"""One frozen exploratory bounded-residual ranking gate candidate; development only."""
import sys
sys.dont_write_bytecode = True
import argparse
import copy
import csv
import hashlib
import json
import platform
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, roc_curve
from sklearn.preprocessing import StandardScaler
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUN_SEEDS = (20261101, 20261102, 20261103)
GATE_SEEDS = (20261022, 20261023, 20261024)
TUNE = ("clean", "jpeg70", "resize05")
CONDITIONS = ("clean", "jpeg70", "jpeg50", "resize05")
MAX_EPOCHS = 200
PATIENCE = 40
TOL = 1e-6


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def inputs(raw):
    b = {k: np.asarray(v, dtype=float) if k != "rows" else v for k, v in raw.items()}
    pg, pl = expit(b["zg"]), expit(b["zl"])
    def entropy(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -p * np.log(p) - (1 - p) * np.log1p(-p)
    b["x"] = np.column_stack([pg, pl, np.abs(pg - pl), (entropy(pg) + entropy(pl)) / 2,
                               np.log1p(b["variance"]), b["q"]])
    b["conditions"] = np.array([r["condition"] for r in b["rows"]])
    assert len(b["x"]) == len(b["y"]) and b["x"].shape[1] == 8
    assert np.isfinite(b["x"]).all()
    return b


class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(8, 32)
        self.output = nn.Linear(32, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        sigmoid_output = torch.sigmoid(self.output(torch.relu(self.hidden(x)))).flatten()
        return .5 + .25 * (2 * sigmoid_output - 1)


def masks(batch):
    return [(torch.from_numpy(np.flatnonzero((batch["conditions"] == cond) & (batch["y"] == 1))),
             torch.from_numpy(np.flatnonzero((batch["conditions"] == cond) & (batch["y"] == 0))))
            for cond in TUNE]


def rank_loss(z, pairs, temperature):
    return torch.stack([torch.nn.functional.softplus(-(z[f][:, None] - z[r][None, :]) / temperature).mean()
                        for f, r in pairs]).mean()


def threshold(y, p):
    fpr, tpr, values = roc_curve(y, p)
    valid = np.isfinite(values)
    values, criterion = values[valid], (tpr - fpr)[valid]
    best = np.flatnonzero(np.isclose(criterion, criterion.max()))
    return float(values[best[np.argmin(np.abs(values[best] - .5))]])


def metric(y, z, cut):
    p = expit(z)
    return dict(auroc=float(roc_auc_score(y, z)), average_precision=float(average_precision_score(y, z)),
                balanced_accuracy=float(balanced_accuracy_score(y, p >= cut)),
                real_false_positive_rate=float(np.mean(p[y == 0] >= cut)),
                bce=float(np.mean(np.logaddexp(0, z) - y * z)))


def features(scaler, batch, kind):
    x = scaler.transform(batch["x"]).astype(np.float32)
    if kind == "score_gate":
        x[:, 5:] = 0
    return torch.from_numpy(x)


def infer(model, scaler, batch, kind):
    with torch.inference_mode():
        alpha = model(features(scaler, batch, kind)).numpy().astype(float)
    assert np.all((alpha >= .25) & (alpha <= .75))
    return alpha, alpha * batch["zg"] + (1 - alpha) * batch["zl"]


def tuning_auc(batch, z):
    return float(np.mean([roc_auc_score(batch["y"][batch["conditions"] == cond],
                                       z[batch["conditions"] == cond]) for cond in TUNE]))


def fit(train, tune, kind, seed, temperature):
    torch.manual_seed(seed)
    model = Gate()
    scaler = StandardScaler().fit(train["x"])
    tx = features(scaler, train, kind)
    zg, zl = [torch.from_numpy(train[k].astype(np.float32)) for k in ("zg", "zl")]
    pairs = masks(train)
    assert all(len(f) == 250 and len(r) == 250 for f, r in pairs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=1e-4)
    best_auc, best_epoch, best_state = -float("inf"), 0, None
    history = []
    for epoch in range(MAX_EPOCHS + 1):
        if epoch:
            model.train()
            optimizer.zero_grad()
            alpha = model(tx)
            loss = rank_loss(alpha * zg + (1 - alpha) * zl, pairs, temperature)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            alpha = model(tx)
            loss_value = float(rank_loss(alpha * zg + (1 - alpha) * zl, pairs, temperature))
        ta, tz = infer(model, scaler, tune, kind)
        auc = tuning_auc(tune, tz)
        assert np.isfinite(loss_value + auc)
        if epoch == 0:
            assert np.array_equal(ta, np.full(len(ta), .5))
        history.append(dict(epoch=epoch, gate_fit_ranking_loss=loss_value, tuning_macro_auroc=auc))
        if auc > best_auc + TOL:
            best_auc, best_epoch, best_state = auc, epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= PATIENCE:
            break
    model.load_state_dict(best_state)
    model.eval()
    ta, tz = infer(model, scaler, tune, kind)
    assert abs(tuning_auc(tune, tz) - best_auc) < 1e-12
    artifact = dict(kind=kind, gate_seed=seed, temperature=temperature, best_epoch=best_epoch,
                    stopped_epoch=history[-1]["epoch"], selected_tuning_auroc=best_auc,
                    threshold=threshold(tune["y"], expit(tz)),
                    scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(),
                    state={k: v.tolist() for k, v in model.state_dict().items()}, history=history)
    return model, scaler, artifact


def self_check():
    torch.manual_seed(0)
    model = Gate()
    assert torch.equal(model(torch.randn(7, 8)), torch.full((7,), .5))
    z = torch.tensor([2., 1., -1., 0.], requires_grad=True)
    pairs = [(torch.tensor([0, 1]), torch.tensor([2, 3]))]
    actual = rank_loss(z, pairs, 2.)
    expected = sum(torch.nn.functional.softplus(-(z[i] - z[j]) / 2.) for i in (0, 1) for j in (2, 3)) / 4
    assert torch.equal(actual, expected)
    actual.backward()
    assert (z.grad[:2] < 0).all() and (z.grad[2:] > 0).all()
    assert threshold(np.array([0, 0, 1, 1]), np.array([.1, .2, .7, .8])) == .7
    print("Synthetic checks passed: residual initialization, all-pair loss, gradient direction, threshold", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    if args.self_check:
        self_check()
        return
    assert not (OUT / "freeze.json").exists(), "This single-run candidate is already frozen; no automatic rerun."
    paths = [ROOT / f"artifacts/round8_a/seed{n}/matched/bundle.json" for n in (1, 2, 3)]
    protocol = dict(
        id="bounded_residual_pairwise_ranking_v1", created_utc=datetime.now(timezone.utc).isoformat(),
        scope="One exploratory combined candidate using existing matched gate_fit/tuning; no new experts, no images, no final-test access.",
        independence="Specified before candidate development scoring; no other new ablation development outcomes consulted.",
        method=dict(architecture="Original 8->32 ReLU->1 sigmoid, output head initialized to zero",
                    alpha="0.5+0.25*(2*sigmoid_output-1), bounded [0.25,0.75]",
                    features="Original 8 gate inputs standardized on gate_fit only; score variant masks standardized quality columns 5:8 to zero",
                    loss="Equal mean over clean/jpeg70/resize05 of all within-condition fake-real pair softplus(-(zf-zr)/T)",
                    temperature="max(1, std(ddof=0) of gate_fit 0.5*(zg+zl), all three conditions), one fixed value per data seed",
                    optimizer="AdamW lr=.003 weight_decay=1e-4, full batch, float32 CPU, 2 threads",
                    run_seeds=list(RUN_SEEDS), gate_seeds=list(GATE_SEEDS), max_epochs=MAX_EPOCHS,
                    selection="Mean per-condition tuning AUROC over clean/jpeg70/resize05, epoch0 included, improvement >1e-6, retain earliest within tolerance",
                    patience=PATIENCE, threshold="Pooled tuning Youden J; original np.isclose ties closest probability threshold .5",
                    baselines="Average alpha .5 and original tuning-selected fixed alpha, inherited experts",
                    reporting="All selection SD15 and development ADM/BigGAN x clean/jpeg70/jpeg50/resize05, initializations averaged within data seed",
                    claims="Combined bounded residual plus ranking objective candidate; no single-component causal attribution; no hyperparameter search or performance-driven retries"),
        code_sha256=sha(Path(__file__)), input_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        environment=dict(python=sys.version, platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__,
                         sklearn=sklearn.__version__, torch=torch.__version__, device="cpu", threads=2),
        timeout_seconds=1800)
    write(OUT / "freeze.json", protocol)
    started = time.monotonic()
    trained, batches, original_settings, input_baseline_metrics = {}, {}, {}, []
    for n, run in enumerate(RUN_SEEDS, 1):
        path = paths[n - 1]
        bundle = json.loads(path.read_text(encoding="utf-8"))
        assert bundle["run_seed"] == run and bundle["arm"] == "matched" and not bundle["violations"]
        raw = bundle["raw_batches"]
        assert set(raw) == {"gate_fit", "tuning", "selection", "development"}
        train, tune = inputs(raw["gate_fit"]), inputs(raw["tuning"])
        assert set(train["conditions"]) == set(TUNE) and set(tune["conditions"]) == set(TUNE)
        assert {r["generator"] for r in train["rows"]} == {"Real", "SD15"}
        assert {r["generator"] for r in tune["rows"]} == {"Real", "SD15"}
        assert not {r["source_id"] for r in train["rows"]} & {r["source_id"] for r in tune["rows"]}
        temperature = max(1., float(np.std(.5 * (train["zg"] + train["zl"]), ddof=0)))
        original_settings[run] = bundle["settings"]["settings"]
        # Reporting batches are held without model scoring until every selected model is frozen.
        batches[run] = {p: raw[p] for p in ("selection", "development")}
        for kind in ("score_gate", "quality_gate"):
            for gs in GATE_SEEDS:
                assert time.monotonic() - started < 1800, "Hard timeout"
                model, scaler, artifact = fit(train, tune, kind, gs, temperature)
                artifact.update(run_seed=run, freeze_sha256=sha(OUT / "freeze.json"))
                write(OUT / f"model_{run}_{kind}_{gs}.json", artifact)
                trained[run, kind, gs] = model, scaler, artifact
                print(f"trained {run}/{kind}/{gs}: epoch={artifact['best_epoch']} tuning_auc={artifact['selected_tuning_auroc']:.6f}", flush=True)
        for name, weight in (("average", .5), ("fixed", original_settings[run]["fixed/-1"]["fixed_alpha"])):
            tz = weight * tune["zg"] + (1 - weight) * tune["zl"]
            cut = threshold(tune["y"], expit(tz))
            assert abs(cut - original_settings[run][name + "/-1"]["threshold"]) < 1e-12
        del bundle, raw, train, tune
    selected_paths = sorted(OUT.glob("model_*.json"))
    assert len(selected_paths) == 18
    write(OUT / "selection_complete.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
          selected_models_sha256={p.name: sha(p) for p in selected_paths},
          note="All 18 gates selected and frozen before any candidate development scoring"))
    all_metrics, pred_count = [], 0
    with (OUT / "predictions.csv").open("x", encoding="utf-8", newline="") as f:
        fields = ["run_seed", "model", "gate_seed", "partition", "source_id", "generator", "label", "condition", "alpha_global", "logit", "probability_fake", "threshold"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for run in RUN_SEEDS:
            for part, raw in batches[run].items():
                b = inputs(raw)
                assert set(b["conditions"]) == set(CONDITIONS)
                assert {r["generator"] for r in b["rows"]} == ({"Real", "SD15"} if part == "selection" else {"Real", "ADM", "BigGAN"})
                outputs = []
                for name, weight in (("average", .5), ("fixed", original_settings[run]["fixed/-1"]["fixed_alpha"])):
                    alpha = np.full(len(b["y"]), weight)
                    outputs.append((name, -1, alpha, alpha * b["zg"] + (1 - alpha) * b["zl"], original_settings[run][name + "/-1"]["threshold"]))
                for kind in ("score_gate", "quality_gate"):
                    for gs in GATE_SEEDS:
                        model, scaler, art = trained[run, kind, gs]
                        alpha, z = infer(model, scaler, b, kind)
                        outputs.append((kind, gs, alpha, z, art["threshold"]))
                for name, gs, alpha, z, cut in outputs:
                    for r, aa, zz in zip(b["rows"], alpha, z):
                        writer.writerow(dict(run_seed=run, model=name, gate_seed=gs, partition=part, **r,
                                             alpha_global=float(aa), logit=float(zz), probability_fake=float(expit(zz)), threshold=cut))
                        pred_count += 1
                    for gen in (("SD15",) if part == "selection" else ("ADM", "BigGAN")):
                        for cond in CONDITIONS:
                            mask = np.array([r["condition"] == cond and r["generator"] in ("Real", gen) for r in b["rows"]])
                            all_metrics.append(dict(run_seed=run, model=name, gate_seed=gs, partition=part,
                                generator=gen, condition=cond, n_real=int(np.sum(b["y"][mask] == 0)), n_fake=int(np.sum(b["y"][mask] == 1)),
                                **metric(b["y"][mask], z[mask], cut),
                                alpha_mean=float(np.mean(alpha[mask])), alpha_q05=float(np.quantile(alpha[mask], .05)),
                                alpha_q50=float(np.quantile(alpha[mask], .5)), alpha_q95=float(np.quantile(alpha[mask], .95))))
    write(OUT / "metrics_each_initialization.json", all_metrics)
    grouped = defaultdict(lambda: defaultdict(list))
    for r in all_metrics:
        grouped[r["model"], r["partition"], r["generator"], r["condition"]][r["run_seed"]].append(r)
    measures = ("auroc", "average_precision", "balanced_accuracy", "real_false_positive_rate", "bce", "alpha_mean")
    summaries = []
    for key, per_seed in sorted(grouped.items()):
        row = dict(zip(("model", "partition", "generator", "condition"), key))
        for measure in measures:
            values = [float(np.mean([r[measure] for r in per_seed[s]])) for s in RUN_SEEDS]
            row[measure] = dict(mean=float(np.mean(values)), seed_sd=float(np.std(values, ddof=1)), each_seed=values)
        summaries.append(row)
    write(OUT / "summary.json", summaries)
    assert pred_count == 110400 and len(all_metrics) == 288
    contrasts = []
    lookup = {(r["model"], r["partition"], r["generator"], r["condition"]): r for r in summaries}
    for left, right in (("score_gate", "average"), ("quality_gate", "average"), ("quality_gate", "score_gate"), ("score_gate", "fixed"), ("quality_gate", "fixed")):
        for part, gens in (("selection", ("SD15",)), ("development", ("ADM", "BigGAN"))):
            for gen in gens:
                for cond in CONDITIONS:
                    row = dict(left=left, right=right, partition=part, generator=gen, condition=cond)
                    for measure in measures:
                        delta = np.array(lookup[left, part, gen, cond][measure]["each_seed"]) - np.array(lookup[right, part, gen, cond][measure]["each_seed"])
                        row[measure] = dict(mean=float(delta.mean()), seed_sd=float(delta.std(ddof=1)), each_seed=delta.tolist())
                    contrasts.append(row)
    write(OUT / "paired_contrasts.json", contrasts)
    assert sha(Path(__file__)) == protocol["code_sha256"]
    assert all(sha(ROOT / p) == v for p, v in protocol["input_sha256"].items())
    outputs = [p for p in OUT.iterdir() if p.is_file()]
    write(OUT / "complete.json", dict(status="completed", seconds=time.monotonic()-started,
          prediction_rows=pred_count, metric_cells=len(all_metrics), all_outcomes_retained=True,
          outputs_sha256={p.name: sha(p) for p in outputs}, no_final_test_access=True,
          limitations=["Exploratory previously inspected development, not blind test", "Combined ranking objective and bounded residual; no single-component attribution", "Three shared-pool data seeds; SD is not CI", "No hyperparameter search or performance-driven retry"]))
    print(json.dumps({"status": "completed", "prediction_rows": pred_count, "metric_cells": len(all_metrics), "seconds": time.monotonic()-started}), flush=True)


if __name__ == "__main__":
    main()
