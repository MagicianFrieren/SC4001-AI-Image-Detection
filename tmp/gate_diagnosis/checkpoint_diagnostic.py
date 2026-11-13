"""Diagnose archived development gates and replay one frozen checkpoint ablation."""
import sys
sys.dont_write_bytecode = True
import copy
import csv
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "checkpoint_results"
sys.path.insert(0, str(ROOT / "scripts"))
import gate_models as gates


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def prepare(raw):
    b = {k: (v if k == "rows" else np.asarray(v, dtype=float)) for k, v in raw.items()}
    b["x"] = gates.inputs(b["zg"], b["zl"], b["variance"], b["q"])
    b["condition"] = np.array([r["condition"] for r in b["rows"]])
    b["generator"] = np.array([r["generator"] for r in b["rows"]])
    return b


def macro_auc(b, z):
    return float(np.mean([roc_auc_score(b["y"][m], z[m])
                          for cond in np.unique(b["condition"])
                          for m in [b["condition"] == cond]]))


def tensor_batch(b, scaler, kind):
    x = scaler.transform(b["x"]).astype(np.float32)
    if kind == "score_gate":
        x[:, 5:] = 0
    return [torch.from_numpy(a.astype(np.float32)) for a in [x, b["zg"], b["zl"], b["y"]]]


def replay(train, tune, kind, seed, saved):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    model = gates.Gate()
    scaler = StandardScaler().fit(train["x"])
    tx, tg, tl, ty = tensor_batch(train, scaler, kind)
    vx, vg, vl, vy = tensor_batch(tune, scaler, kind)
    optimizer = torch.optim.AdamW(model.parameters(), lr=gates.LEARNING_RATE, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best_bce, best_auc = float("inf"), float("-inf")
    eb, ea, sb, sa = 0, 0, None, None
    for epoch in range(gates.MAX_EPOCHS + 1):
        if epoch:
            model.train()
            optimizer.zero_grad()
            a = model(tx)
            loss_fn(a * tg + (1 - a) * tl, ty).backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            av = model(vx)
            zv = av * vg + (1 - av) * vl
            vb = float(loss_fn(zv, vy))
            av64 = av.numpy().astype(float)
            va = macro_auc(tune, av64*tune["zg"]+(1-av64)*tune["zl"])
        if vb < best_bce - 1e-6:
            best_bce, eb, sb = vb, epoch, copy.deepcopy(model.state_dict())
        if va > best_auc + 1e-6:
            best_auc, ea, sa = va, epoch, copy.deepcopy(model.state_dict())
        if epoch - eb >= gates.PATIENCE:
            break
    delta = max(float(np.max(np.abs(sb[k].numpy() - np.asarray(saved["state"][k])))) for k in sb)
    assert eb == saved["best_epoch"] and delta < 2e-5, (kind, seed, eb, saved["best_epoch"], delta)
    output = {}
    for label, state in [("bce_checkpoint", sb), ("auc_checkpoint", sa)]:
        model.load_state_dict(state)
        output[label] = {"model": copy.deepcopy(model), "scaler": scaler}
    info = dict(kind=kind, gate_seed=seed, bce_epoch=eb, auc_epoch=ea,
                stopped_epoch=epoch, best_bce=best_bce, best_tuning_auc=best_auc,
                replay_max_weight_difference=delta)
    return output, info


def metrics(b, alpha, threshold):
    z = alpha * b["zg"] + (1 - alpha) * b["zl"]
    result = []
    for cond in np.unique(b["condition"]):
        for gen in sorted(set(b["generator"]) - {"Real"}):
            m = (b["condition"] == cond) & np.isin(b["generator"], ["Real", gen])
            y, zz = b["y"][m], z[m]
            pred = expit(zz) >= threshold
            fpr = float(pred[y == 0].mean())
            result.append(dict(condition=cond, generator=gen, auroc=float(roc_auc_score(y, zz)),
                               bce=float(np.mean(np.logaddexp(0, zz) - y * zz)),
                               balanced_accuracy=float((pred[y == 1].mean() + 1 - fpr) / 2),
                               real_fpr=fpr, alpha_mean=float(np.mean(alpha[m]))))
    return result


def threshold(y, p):
    # Same finite threshold candidate/tie rule as the repository routine.
    from round7_baseline_common import threshold as original_threshold
    return original_threshold(y, p)


def audit(batches, settings):
    out = {}
    tr = batches["gate_fit"]
    qm, qs = tr["q"].mean(axis=0), tr["q"].std(axis=0)
    for name, b in batches.items():
        y = b["y"]
        lg, ll = np.logaddexp(0, b["zg"]) - y*b["zg"], np.logaddexp(0, b["zl"]) - y*b["zl"]
        avg = (b["zg"] + b["zl"]) / 2
        qshift = (b["q"] - qm) / np.maximum(qs, 1e-12)
        out[name] = dict(rows=len(y), originals=len({r["source_id"] for r in b["rows"]}),
                         generators=sorted(set(b["generator"])),
                         global_saturated_fraction=float(np.mean(np.abs(b["zg"]) > np.log(999))),
                         local_saturated_fraction=float(np.mean(np.abs(b["zl"]) > np.log(999))),
                         global_bce=float(lg.mean()), local_bce=float(ll.mean()),
                         average_bce=float(np.mean(np.logaddexp(0,avg)-y*avg)),
                         label_oracle_endpoint_bce=float(np.minimum(lg,ll).mean()),
                         both_wrong_at_zero_fraction=float(np.mean(((b["zg"]>=0)!=y)&((b["zl"]>=0)!=y))),
                         quality_mean_standardized_shift=qshift.mean(axis=0).tolist(),
                         quality_outside_3sd_fraction=np.mean(np.abs(qshift)>3,axis=0).tolist(),
                         quality_vs_global_minus_local_loss_spearman=[float(spearmanr(b["q"][:,j],lg-ll).statistic) for j in range(3)])
    out["gate_training"] = {}
    for name, g in settings["gates"].items():
        hist = g["history"]
        art = g["artifact"]
        out["gate_training"][name] = dict(best_epoch=art["best_epoch"], stopped_epoch=art["stopped_epoch"],
             initial_validation_bce=hist[0]["validation_bce"], best_validation_bce=art["best_validation_bce"])
    return out


def main():
    assert not OUT.exists(), "Use a new version/directory rather than overwrite this experiment"
    OUT.mkdir(parents=True)
    protocol = json.loads((HERE/"checkpoint_protocol.json").read_text())
    paths = [HERE/"checkpoint_protocol.json", Path(__file__), ROOT/"scripts/gate_models.py"]
    for seed in protocol["data_seeds"]:
        for arm in protocol["arms"]:
            p=ROOT/f"artifacts/round8_a/seed{seed}/{arm}/bundle.json"
            c=json.loads((ROOT/f"results/round8_a/seed{seed}/{arm}/complete.json").read_text())
            assert sha(p)==c["files_sha256"][p.relative_to(ROOT).as_posix()]
            paths.append(p)
    dump(OUT/"freeze.json",dict(protocol=protocol, inputs_sha256={str(p):sha(p) for p in paths}))
    audits, logs, rows = {}, [], []
    for seed in protocol["data_seeds"]:
        for arm in protocol["arms"]:
            bundle=json.loads((ROOT/f"artifacts/round8_a/seed{seed}/{arm}/bundle.json").read_text())
            batches={k:prepare(v) for k,v in bundle["raw_batches"].items()}
            settings=bundle["settings"]
            audits[f"seed{seed}/{arm}"]=audit(batches,settings)
            for name,weight in [("average",.5),("fixed",settings["settings"]["fixed/-1"]["fixed_alpha"])]:
                cut=settings["settings"][name+"/-1"]["threshold"]
                for part in ["selection","development"]:
                    b=batches[part]
                    rows.extend(dict(data_seed=seed,arm=arm,model=name,checkpoint="saved",gate_seed=-1,partition=part,**m)
                                for m in metrics(b,np.full(len(b["y"]),weight),cut))
            for kind in protocol["variants"]:
                for gs in protocol["gate_seeds"]:
                    models,info=replay(batches["gate_fit"],batches["tuning"],kind,gs,
                                       settings["gates"][f"{kind}/{gs}"]["artifact"])
                    logs.append(dict(data_seed=seed,arm=arm,**info))
                    for label, obj in models.items():
                        tune=batches["tuning"]
                        ta=gates.predict(obj["model"],obj["scaler"],tune["x"],kind)
                        cut=threshold(tune["y"],expit(ta*tune["zg"]+(1-ta)*tune["zl"]))
                        for part in ["selection","development"]:
                            b=batches[part]
                            al=gates.predict(obj["model"],obj["scaler"],b["x"],kind)
                            rows.extend(dict(data_seed=seed,arm=arm,model=kind,checkpoint=label,gate_seed=gs,partition=part,**m)
                                        for m in metrics(b,al,cut))
                    print(json.dumps(dict(data_seed=seed,arm=arm,**info)),flush=True)
            dump(OUT/"audit.json",audits)
            dump(OUT/"checkpoints.json",logs)
            with (OUT/"metrics.csv").open("w",newline="",encoding="utf-8") as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    grouped=defaultdict(list)
    for row in rows:
        if row["partition"]=="development":
            grouped[(row["arm"],row["model"],row["checkpoint"],row["data_seed"])].append(row["auroc"])
    summary=[dict(arm=k[0],model=k[1],checkpoint=k[2],data_seed=k[3],eight_cell_macro_auc=float(np.mean(v))) for k,v in grouped.items()]
    dump(OUT/"summary.json",summary)
    dump(OUT/"complete.json",dict(status="completed",gate_replays=len(logs),metric_rows=len(rows),
         final_test_access=False, max_replay_weight_difference=max(x["replay_max_weight_difference"] for x in logs)))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
