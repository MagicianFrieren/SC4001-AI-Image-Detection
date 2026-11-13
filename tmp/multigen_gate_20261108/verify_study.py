"""Independent read-only verifier; refuses evaluation before both completion markers.

No experiment imports, fitting, raw images, or final/reserved-test access.
Prints a JSON report to stdout and never modifies experiment artifacts.
"""
import sys
sys.dont_write_bytecode = True
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/analysis/multigen_gate_20261108/run_v1_codefix1"
REQUIRED = ("complete.json", "models_frozen_before_evaluation.json")
if not all((OUT / name).is_file() for name in REQUIRED):
    raise SystemExit("NOT_READY: complete.json and models_frozen_before_evaluation.json are both required; no scoring performed.")

import joblib
import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

SEEDS = (20261101, 20261102, 20261103)
GATE_SEEDS = (20261022, 20261023, 20261024)
TUNE = ("clean", "jpeg70", "resize05")
CONDS = ("clean", "jpeg70", "jpeg50", "resize05")
SCENARIOS = ("source_only", "mixed_ADM", "mixed_BigGAN")
MEASURES = ("auroc", "average_precision", "balanced_accuracy", "real_false_positive_rate", "bce")
SALT = "multigen_gate_20261108_v1/"
KEYS = ("run_seed", "scenario", "heldout_generator", "model", "checkpoint", "gate_seed", "condition")
CHECKED = {}


def safe_path(relative):
    assert "reserved" not in relative.lower() and "final" not in relative.lower(), relative
    p = (ROOT / relative).resolve()
    assert p.is_relative_to(ROOT.resolve())
    return p


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verify_hash(path, expected):
    actual = CHECKED.setdefault(str(path), sha(path))
    assert actual == expected, f"Hash mismatch: {path}"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def ordered(ids, purpose):
    return sorted(ids, key=lambda s: (hashlib.sha256((SALT + purpose + "/" + s).encode()).hexdigest(), s))


def key(row):
    return tuple(int(row[k]) if k in ("run_seed", "gate_seed") else row[k] for k in KEYS)


def xfeatures(raw):
    pg, pl = expit(raw["zg"]), expit(raw["zl"])
    def ent(p):
        p = np.clip(p, 1e-12, 1-1e-12)
        return -(p*np.log(p)+(1-p)*np.log1p(-p))
    return np.column_stack([pg, pl, np.abs(pg-pl), (ent(pg)+ent(pl))/2,
                            np.log1p(raw["variance"]), raw["q"]])


def numpy_alpha(artifact, x):
    scaled = ((x - np.array(artifact["mean"])) / np.array(artifact["scale"])).astype(np.float32)
    if artifact["kind"] == "score_gate":
        scaled[:, 5:] = 0
    state = {k: np.array(v, dtype=np.float32) for k, v in artifact["state"].items()}
    hidden = np.maximum(scaled @ state["hidden.weight"].T + state["hidden.bias"], 0)
    return expit(hidden @ state["output.weight"].T + state["output.bias"]).ravel().astype(float)


def cutoff(y, z):
    fpr, tpr, values = roc_curve(y, expit(z))
    finite = np.isfinite(values)
    values, obj = values[finite], (tpr-fpr)[finite]
    winners = np.flatnonzero(np.isclose(obj, np.max(obj)))
    return float(values[winners[np.argmin(np.abs(values[winners]-.5))]])


def metrics(y, z, cut):
    real, fake = y == 0, y == 1
    ranks = rankdata(z, method="average")
    n0, n1 = int(real.sum()), int(fake.sum())
    pred = expit(z) >= cut
    return dict(auroc=float((ranks[fake].sum()-n1*(n1+1)/2)/(n0*n1)),
                average_precision=float(average_precision_score(y, z)),
                balanced_accuracy=float((pred[fake].mean()+(~pred[real]).mean())/2),
                real_false_positive_rate=float(pred[real].mean()),
                bce=float((np.logaddexp(0, z)-y*z).mean()))


def source_logits(number, part, experts, bundle):
    """Apply saved linear coefficients independently; no sklearn predict calls."""
    template = bundle["raw_batches"]["gate_fit" if part == "gate_fit" else "tuning"]["rows"]
    expected_ids = sorted({r["source_id"] for r in template})
    rows, gg, ll, vv, qq = [], [], [], [], []
    def head_values(head, values):
        scaler, classifier = head.steps[0][1], head.steps[1][1]
        work = values.copy()  # Keep original float32 StandardScaler in-place arithmetic.
        if scaler.with_mean:
            work -= scaler.mean_.astype(work.dtype)
        if scaler.with_std:
            work /= scaler.scale_.astype(work.dtype)
        return work @ classifier.coef_.reshape(-1) + classifier.intercept_[0]
    for cond in TUNE:
        folder = ROOT / f"artifacts/round8_a/seed{number}/features/{part}_{cond}_0"
        blocks = {k: [] for k in ("g", "l", "q")}
        cache_ids = []
        for p in sorted(folder.glob("*.npz")):
            assert str(p) in CHECKED, f"Unpinned cache: {p}"
            with np.load(p, allow_pickle=False) as cache:
                cache_ids += cache["ids"].tolist()
                for k in blocks:
                    blocks[k].append(cache[k])
        assert cache_ids == expected_ids
        blocks = {k: np.concatenate(v) for k, v in blocks.items()}
        condition_rows = [r for r in template if r["condition"] == cond]
        assert [r["source_id"] for r in condition_rows] == cache_ids
        gh, lh = experts["global_head"], experts["local_head"]
        zg = head_values(gh, blocks["g"])
        zl = head_values(lh, blocks["l"].mean(axis=1))
        var = head_values(lh, blocks["l"].reshape(-1, 512)).reshape(-1, 4).var(axis=1)
        rows.extend(condition_rows); gg.extend(zg); ll.extend(zl); vv.extend(var); qq.extend(blocks["q"])
    raw = dict(rows=rows, zg=np.array(gg), zl=np.array(ll), variance=np.array(vv), q=np.array(qq),
               y=np.array([r["label"] for r in rows]))
    return raw


def subset(raw, ids):
    ids = set(ids)
    ix = [i for cond in TUNE for i, r in enumerate(raw["rows"]) if r["condition"] == cond and r["source_id"] in ids]
    assert len(ix) == 3*len(ids)
    return {k: [raw[k][i] for i in ix] if k == "rows" else np.asarray(raw[k])[ix] for k in raw}


def combine(a, b):
    return {k: a[k]+b[k] if k == "rows" else np.concatenate([a[k], b[k]]) for k in a}


def macro_auc(batch, z):
    cond = np.array([r["condition"] for r in batch["rows"]])
    gen = np.array([r["generator"] for r in batch["rows"]])
    cells = [(cond == c) & np.isin(gen, ("Real", g)) for g in sorted(set(gen)-{"Real"}) for c in TUNE]
    return float(np.mean([roc_auc_score(batch["y"][m], z[m]) for m in cells]))


def main():
    complete, model_marker = [load(OUT/name) for name in REQUIRED]
    assert complete["status"] == "completed" and complete["fits"] == model_marker["fits"] == 54
    assert complete["prediction_rows"] == 460800 and complete["metric_rows"] == 768
    freeze = load(OUT/"freeze.json")
    for p, h in freeze["input_sha256"].items():
        verify_hash(safe_path(p), h)
    for p, h in complete["outputs_sha256"].items():
        assert ".." not in Path(p).parts
        verify_hash(OUT/p, h)
    verify_hash(OUT/"splits.json", freeze["split_sha256"])
    verify_hash(OUT/"model_settings.json", model_marker["settings_sha256"])
    assert len(model_marker["model_files_sha256"]) == len(list((OUT/"models").glob("*.json"))) == 54
    for name, h in model_marker["model_files_sha256"].items():
        verify_hash(OUT/"models"/name, h)
    preflight = load(ROOT/"tmp/multigen_gate_20261108/preflight/id_layout_preflight.json")
    assert preflight["status"] == "passed" and not preflight["dhash_distance_le3_pairs"]
    for p, h in preflight["input_sha256"].items():
        verify_hash(safe_path(p), h)
    splits, settings = load(OUT/"splits.json"), load(OUT/"model_settings.json")
    records = {}
    for name in ("fitting_pool_manifest.jsonl", "development_manifest.jsonl"):
        for line in (ROOT/"results/round7"/name).read_text().splitlines():
            r = json.loads(line); records[r["source_id"]] = r
    def generators(ids):
        return Counter(records[i]["generator"] for i in ids)
    assert set(splits["sources"]) == {f"{s}/{a}" for s in SEEDS for a in SCENARIOS}
    for gen in ("ADM", "BigGAN"):
        full = {i for i, r in records.items() if r["generator"] == gen}
        ordered_ids = ordered(full, "donor/"+gen)
        expected = dict(fit=ordered_ids[:125], tuning=ordered_ids[125:187], unused=ordered_ids[187:])
        assert splits["donors"][gen] == expected
        assert splits["evaluation_ids"][gen] == sorted(full)
    assert len(splits["evaluation_ids"]["Real"]) == 300
    for number, run in enumerate(SEEDS, 1):
        assignment = load(ROOT/f"results/round7/assignments_seed{number}.json")
        original_fit, original_tune = map(set, (assignment["partitions"]["gate_fit"], assignment["partitions"]["selection"]))
        fit_real = {i for i in original_fit if records[i]["generator"] == "Real"}
        fit_sd = original_fit-fit_real
        tune_real = set(ordered({i for i in original_tune if records[i]["generator"] == "Real"}, "source_tune_real")[:124])
        tune_sd = set(ordered({i for i in original_tune if records[i]["generator"] == "SD15"}, "source_tune_fake")[:124])
        for scenario in SCENARIOS:
            saved = splits["sources"][f"{run}/{scenario}"]
            fi, tu = map(set, (saved["fit_ids"], saved["tuning_ids"]))
            assert len(fi) == 500 and len(tu) == 248 and not fi & tu
            assert not (fi | tu) & set(assignment["partitions"]["expert_fit"])
            if scenario == "source_only":
                assert fi == fit_real | fit_sd and tu == tune_real | tune_sd
                assert generators(fi) == Counter(Real=250, SD15=250)
                assert generators(tu) == Counter(Real=124, SD15=124)
                targets = ["BigGAN", "ADM"]
            else:
                donor = scenario[6:]; target = "BigGAN" if donor == "ADM" else "ADM"
                assert fi == fit_real | set(ordered(fit_sd, "source_fit_fake")[:125]) | set(splits["donors"][donor]["fit"])
                assert tu == tune_real | set(ordered(tune_sd, "source_tune_fake")[:62]) | set(splits["donors"][donor]["tuning"])
                assert generators(fi) == Counter({"Real":250,"SD15":125,donor:125})
                assert generators(tu) == Counter({"Real":124,"SD15":62,donor:62})
                targets = [target]
            assert saved["heldout_targets"] == targets
            for target in targets:
                ev = set(splits["evaluation_ids"][target] + splits["evaluation_ids"]["Real"])
                assert not (fi | tu) & ev and len(ev) == 600
                for field in ("pixel_sha256", "source_bytes_sha256"):
                    assert not {records[i][field] for i in fi | tu} & {records[i][field] for i in ev}

    artifact_count = 0
    for run in SEEDS:
        for scenario in SCENARIOS:
            for kind in ("score_gate", "quality_gate"):
                for gs in GATE_SEEDS:
                    model = load(OUT/f"models/{run}_{scenario}_{kind}_{gs}.json")
                    assert model["checkpoints"] == settings[f"{run}/{scenario}"]["gates"][f"{kind}/{gs}"]
                    assert set(model["checkpoints"]) == {"auroc", "bce"}
                    history = model["history"]
                    assert [r["epoch"] for r in history] == list(range(len(history))) and len(history) <= 401
                    eb, ea, best_b, best_a = 0, 0, float("inf"), -float("inf")
                    for h in history:
                        if h["tuning_bce"] < best_b-1e-6: best_b, eb = h["tuning_bce"], h["epoch"]
                        if h["tuning_macro_auroc"] > best_a+1e-6: best_a, ea = h["tuning_macro_auroc"], h["epoch"]
                    assert model["checkpoints"]["bce"]["selected_epoch"] == eb
                    assert model["checkpoints"]["auroc"]["selected_epoch"] == ea
                    assert history[-1]["epoch"] == 400 or history[-1]["epoch"]-eb == 40
                    artifact_count += 2
    assert artifact_count == 108

    groups = defaultdict(list)
    with gzip.open(OUT/"predictions.csv.gz", "rt", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            groups[key(r)].append(r)
    assert len(groups) == 768 and sum(map(len, groups.values())) == 460800
    expected_keys = set()
    for run in SEEDS:
        for donor, target in (("ADM","BigGAN"),("BigGAN","ADM")):
            for scenario in ("source_only","mixed_"+donor):
                for cond in CONDS:
                    for name in ("global","local","average","fixed"):
                        expected_keys.add((run,scenario,target,name,"fixed",-1,cond))
                    for kind in ("score_gate","quality_gate"):
                        for cp in ("auroc","bce"):
                            for gs in GATE_SEEDS: expected_keys.add((run,scenario,target,kind,cp,gs,cond))
    assert set(groups) == expected_keys
    with (OUT/"metrics_each_initialization.csv").open(newline="",encoding="utf-8") as f:
        saved_rows = list(csv.DictReader(f))
    saved_metrics = {key(r): r for r in saved_rows}
    assert len(saved_rows) == len(saved_metrics) == 768 and set(saved_metrics) == expected_keys
    dev, index = {}, {}
    for number, run in enumerate(SEEDS, 1):
        bundle = load(ROOT/f"artifacts/round8_a/seed{number}/matched/bundle.json")
        raw = bundle["raw_batches"]["development"]
        raw = {k:v if k=="rows" else np.array(v) for k,v in raw.items()}
        raw["x"] = xfeatures(raw)
        dev[run] = raw
        index[run] = {(r["condition"],r["source_id"]):i for i,r in enumerate(raw["rows"])}
    max_metric, max_alpha, max_fusion = 0., 0., 0.
    recomputed = {}
    for k, rows in groups.items():
        run, scenario, target, name, cp, gs, cond = k
        expected_ids = splits["evaluation_ids"]["Real"] + splits["evaluation_ids"][target]
        assert [r["source_id"] for r in rows] == expected_ids
        y = np.array([int(r["label"]) for r in rows]); assert np.array_equal(y,np.repeat([0,1],300))
        assert all(r["generator"] == ("Real" if i<300 else target) for i,r in enumerate(rows))
        z, alpha, cuts = [np.array([float(r[f]) for r in rows]) for f in ("logit","alpha_global","threshold")]
        assert np.all(cuts == cuts[0]) and np.isfinite(z).all() and np.all((alpha>=0)&(alpha<=1))
        config = settings[f"{run}/{scenario}"]
        ix = np.array([index[run][cond,i] for i in expected_ids])
        b = dev[run]
        if gs == -1:
            setting = config["baselines"][name]
            assert cuts[0] == setting["threshold"] and np.all(alpha == setting["alpha"])
        else:
            setting = config["gates"][f"{name}/{gs}"][cp]
            assert cuts[0] == setting["threshold"]
            independent_alpha = numpy_alpha(setting,b["x"][ix])
            gap = float(np.max(np.abs(independent_alpha-alpha))); max_alpha=max(max_alpha,gap)
            assert gap < 5e-6, (k,gap)
        fused = alpha*b["zg"][ix]+(1-alpha)*b["zl"][ix]
        gap = float(np.max(np.abs(fused-z))); max_fusion=max(max_fusion,gap)
        assert gap < 1e-12
        m = metrics(y,z,cuts[0]);m["alpha_mean"]=float(alpha.mean())
        for measure, value in m.items():
            gap=abs(value-float(saved_metrics[k][measure]));max_metric=max(max_metric,gap);assert gap<1e-12,(k,measure,gap)
        recomputed[k] = m
    for run in SEEDS:
        for donor,target in (("ADM","BigGAN"),("BigGAN","ADM")):
            for cond in CONDS:
                a=groups[run,"source_only",target,"average","fixed",-1,cond]
                b=groups[run,"mixed_"+donor,target,"average","fixed",-1,cond]
                assert [r["logit"] for r in a] == [r["logit"] for r in b]

    # Independent original-view source reconstruction, fixed grid, and thresholds.
    max_source_gap, max_grid_gap, max_cut_gap, max_scaler_gap = 0.,0.,0.,0.
    baseline_thresholds_checked, gate_tuning_auc_gap = 0,0.
    for number,run in enumerate(SEEDS,1):
        bundle=load(ROOT/f"artifacts/round8_a/seed{number}/matched/bundle.json")
        experts=joblib.load(ROOT/f"artifacts/round8_a/seed{number}/matched/experts.joblib")
        fitraw=source_logits(number,"gate_fit",experts,bundle)
        tuneraw=source_logits(number,"selection",experts,bundle)
        original=bundle["raw_batches"]["selection"]
        lookup={(r["condition"],r["source_id"]):i for i,r in enumerate(original["rows"])}
        ix=[lookup[r["condition"],r["source_id"]] for r in tuneraw["rows"]]
        for field in ("zg","zl","variance","q","y"):
            gap=float(np.max(np.abs(tuneraw[field]-np.array(original[field])[ix])))
            max_source_gap=max(max_source_gap,gap);assert gap<1e-10,(run,field,gap)
        for scenario in SCENARIOS:
            split=splits["sources"][f"{run}/{scenario}"]
            fit_source={i for i in split["fit_ids"] if records[i]["generator"] in ("Real","SD15")}
            tune_source={i for i in split["tuning_ids"] if records[i]["generator"] in ("Real","SD15")}
            train,tune=subset(fitraw,fit_source),subset(tuneraw,tune_source)
            if scenario!="source_only":
                d={k:v for k,v in dev[run].items() if k!="x"}
                train=combine(train,subset(d,set(split["fit_ids"])-fit_source))
                tune=combine(tune,subset(d,set(split["tuning_ids"])-tune_source))
            config=settings[f"{run}/{scenario}"]
            grid=[(a/10,macro_auc(tune,(a/10)*tune["zg"]+(1-a/10)*tune["zl"])) for a in range(11)]
            gap=float(np.max(np.abs(np.array(grid)-np.array(config["fixed_search"]))));max_grid_gap=max(max_grid_gap,gap)
            assert gap<1e-12
            top=max(v for a,v in grid)
            winner=min([a for a,v in grid if abs(v-top)<=1e-6],key=lambda a:(abs(a-.5),a))
            assert winner == config["baselines"]["fixed"]["alpha"]
            for name,setting in config["baselines"].items():
                a=setting["alpha"];cut=cutoff(tune["y"],a*tune["zg"]+(1-a)*tune["zl"])
                gap=abs(cut-setting["threshold"]);max_cut_gap=max(max_cut_gap,gap);assert gap<1e-10
                baseline_thresholds_checked+=1
            tx,vx=xfeatures(train),xfeatures(tune)
            for checkpoints in config["gates"].values():
                for artifact in checkpoints.values():
                    gap=max(float(np.max(np.abs(tx.mean(axis=0)-artifact["mean"]))),float(np.max(np.abs(tx.std(axis=0)-artifact["scale"]))))
                    max_scaler_gap=max(max_scaler_gap,gap);assert gap<1e-10
                    a=numpy_alpha(artifact,vx);z=a*tune["zg"]+(1-a)*tune["zl"]
                    # NumPy/Torch float32 GEMM can differ near rank ties; allow one
                    # source-pair rank change in this secondary reconstruction check.
                    gap=abs(macro_auc(tune,z)-artifact["tuning_macro_auroc"])
                    gate_tuning_auc_gap=max(gate_tuning_auc_gap,gap);assert gap<1e-4
    assert baseline_thresholds_checked==36

    # Initialization -> data seed means and report macro, independent of summaries.
    grouped=defaultdict(list)
    for k,m in recomputed.items(): grouped[k[:5]+(k[6],)].append(m)
    with (OUT/"metrics_seed_condition.csv").open(newline="",encoding="utf-8") as f: seedrows=list(csv.DictReader(f))
    assert len(seedrows)==len(grouped)==384
    for r in seedrows:
        k=(int(r["run_seed"]),r["scenario"],r["heldout_generator"],r["model"],r["checkpoint"],r["condition"])
        for measure in MEASURES+("alpha_mean",):
            assert abs(float(r[measure])-np.mean([x[measure] for x in grouped[k]]))<1e-12
    summary=load(OUT/"summary.json")
    for row in summary["macro"]:
        selected=[r for r in seedrows if ("mixed" if r["scenario"].startswith("mixed_") else r["scenario"])==row["arm"] and r["model"]==row["model"] and r["checkpoint"]==row["checkpoint"]]
        assert len(selected)==24
        for measure in MEASURES: assert abs(row[measure]-np.mean([float(r[measure]) for r in selected]))<1e-12
    print(json.dumps(dict(status="passed", verifier_sha256=sha(Path(__file__)),
        checked_hashes=len(CHECKED), fits=54, checkpoint_models=artifact_count, prediction_rows=460800,
        instance_metric_cells=768, seed_metric_cells=384, all_gate_prediction_weights_reconstructed=True,
        maximum_metric_error=max_metric,maximum_numpy_alpha_error=max_alpha,maximum_saved_alpha_fusion_error=max_fusion,
        source_original_reconstruction_max_error=max_source_gap,fixed_alpha_grids_checked=9,
        fixed_grid_max_error=max_grid_gap,baseline_thresholds_checked=baseline_thresholds_checked,
        baseline_threshold_max_error=max_cut_gap,gate_scaler_max_error=max_scaler_gap,
        numpy_gate_tuning_auc_max_error=gate_tuning_auc_gap,
        average_logits_identical_across_arms=True,split_and_donor_hash_memberships_verified=True,
        no_final_test_access=True,no_fitting=True,
        limitations=["Bootstrap file hash verified; intervals not replayed", "Gate source AUROC tolerates float32 NumPy/Torch rank differences below 1e-4", "Gate cutoff values linked exactly to saved prediction/settings; independent threshold re-derivation covers all 36 baselines"]),indent=2))


if __name__=="__main__": main()
