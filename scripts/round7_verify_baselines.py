"""Recompute saved Step C metrics and validate isolation without loading images/models."""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import defaultdict
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, roc_curve
import round7_baseline_common as c


def independently_threshold(y, p):
    fpr, tpr, candidates = roc_curve(y, p)
    valid = np.isfinite(candidates)
    candidates, scores = candidates[valid], (tpr-fpr)[valid]
    best = np.flatnonzero(np.isclose(scores, max(scores)))
    return float(candidates[best[np.argmin(abs(candidates[best]-.5))]])


def verify_predictions(predictions, records, parts, deny):
    groups, identities = defaultdict(list), set()
    for row in predictions:
        identity = row["source_id"]
        c.require(identity not in deny and identity in records, "Test or forbidden source ID in predictions")
        part = row["partition"]
        c.require(part in ("selection", "development") and identity in parts[part], "Prediction on non-evaluation partition")
        c.require(row["generator"] == records[identity]["generator"] and int(row["label"]) == records[identity]["label"], "Incorrect prediction label")
        key = tuple(str(row[k]) for k in ("profile", "model", "calibration", "seed", "partition", "condition"))
        c.require(key + (identity,) not in identities, "Duplicate prediction")
        identities.add(key + (identity,)); groups[key].append(row)
        z, prob, threshold = map(float, [row["logit"], row["probability_fake"], row["threshold"]])
        c.require(np.isfinite([z, prob, threshold]).all() and np.isclose(expit(z), prob, atol=1e-14, rtol=0), "Invalid saved probability")
    for key, cell in groups.items():
        c.require({r["source_id"] for r in cell} == set(parts[key[4]]), "Incomplete prediction coverage")
    expected = set()
    for profile in c.PROFILES:
        for model in ("global", "local", "average", "fixed", "score_gate", "quality_gate", "resnet50"):
            seeds = c.GATE_SEEDS if model.endswith("gate") else [c.SEED] if model == "resnet50" else [-1]
            for cal in ("raw", "temperature"):
                for seed in seeds:
                    for part in ("selection", "development"):
                        for cond in c.CONDITIONS:
                            expected.add((profile, model, cal, str(seed), part, cond))
    c.require(set(groups) == expected, "Missing/extra models, profiles, calibrations, initializations or conditions")
    for profile, model, cal, seed in {k[:4] for k in groups}:
        tuning = sum([groups[profile, model, cal, seed, "selection", cond] for cond in c.TUNE], [])
        cut = independently_threshold([int(r["label"]) for r in tuning], [float(r["probability_fake"]) for r in tuning])
        for part in ("selection", "development"):
            for cond in c.CONDITIONS:
                c.require(all(np.isclose(float(r["threshold"]), cut, atol=1e-12, rtol=0) for r in groups[profile, model, cal, seed, part, cond]), "Threshold was not fitted on selection-only tuning conditions")
    return groups


def verify_logs(records, parts, deny):
    count = 0
    restricted = {"fit_clip_experts": "expert_fit", "fit_resnet_expert_only": "expert_fit", "resnet_resume_expert_partition": "expert_fit",
                  "fit_score_gate": "gate_fit", "fit_quality_gate": "gate_fit", "fit_clip_temperatures": "calibration", "fit_resnet_temperature": "calibration",
                  "select_clip_fusion_threshold_epoch": "selection", "select_resnet_epoch": "selection", "select_resnet_threshold": "selection", "resnet_resume_selection_partition": "selection"}
    for path in sorted((c.ART / "access_logs").glob("*.json")):
        log = c.read(path)
        c.require(log["isolation_sha256"] == c.sha(c.DATA / "step_b_isolation.json") and not log["violations"], "Isolation guard recorded a forbidden access")
        for action, ids in log["events"].items():
            ids = set(ids)
            c.require(not ids & deny and ids <= set(records), "Reserved-test source ID was touched")
            if action in restricted:
                c.require(ids <= set(parts[restricted[action]]), f"Wrong fitting partition for {action}")
            if action.startswith("predict/"):
                c.require(ids <= set(parts[action.split('/')[1]]), "Wrong evaluation partition")
        count += 1
    c.require(count >= 6, "Missing stage access logs")
    return count


def main():
    import round7_seed_support as seeds
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=seeds.SEEDS, default=20261101)
    parser.add_argument("--bootstrap", action="store_true", help="Also replay all paired bootstrap CIs (slower)")
    parser.add_argument("--report", help="Optional new JSON filename inside baselines_seed1")
    args = parser.parse_args()
    seeds.configure(args.seed)
    c.require(args.seed != 20261101 or not args.report, "Seed-1 output folder is frozen; verify it without --report")
    if args.report:
        c.require(c.Path(args.report).name == args.report, "Use a simple report filename")
        if (c.OUT / args.report).exists():
            raise FileExistsError(args.report)
    records, parts, deny = c.context()
    protocol = c.read(c.OUT / "protocol.json")
    c.require(protocol == c.protocol(), "Protocol/code/input provenance changed")
    from round7_run_baselines import completed
    for stage in ["prepare", "clip_original", "clip_common128", "resnet_original", "resnet_common128", "summarize"]:
        c.require(completed(stage, True), f"Incomplete stage: {stage}")
    predictions = sum([c.read_csv(c.OUT / f"{model}_{profile}_predictions.csv") for model in ("clip", "resnet") for profile in c.PROFILES], [])
    groups = verify_predictions(predictions, records, parts, deny)
    logs = verify_logs(records, parts, deny)
    metrics = c.read_csv(c.OUT / "metrics.csv")
    seen, macro = set(), defaultdict(list)
    for row in metrics:
        key = tuple(row[k] for k in ("profile", "model", "calibration", "seed", "partition", "condition"))
        identity = key + (row["generator"],)
        c.require(identity not in seen, "Duplicate metric cell"); seen.add(identity)
        cell = [r for r in groups[key] if r["generator"] in ("Real", row["generator"])]
        y = np.array([int(r["label"]) for r in cell]); z = np.array([float(r["logit"]) for r in cell])
        prob = expit(z); cut = float(row["threshold"])
        actual = {"auroc": roc_auc_score(y, z), "average_precision": average_precision_score(y, z),
                  "balanced_accuracy": balanced_accuracy_score(y, prob >= cut), "real_false_positive_rate": np.mean(prob[y == 0] >= cut)}
        c.require(int(row["n"]) == len(cell) and cut == float(cell[0]["threshold"]), "Metric sample count/threshold mismatch")
        for name, val in actual.items():
            c.require(np.isclose(float(row[name]), val, atol=1e-12, rtol=0), f"Metric mismatch: {name}")
            c.require(0 <= float(row[name+"_ci_low"]) <= float(row[name+"_ci_high"]) <= 1, "Invalid CI")
        macro[row["profile"], row["model"], row["calibration"], row["partition"], row["generator"]].append(actual["auroc"])
    expected_cells = {k + (g,) for k in groups for g in (["SD15"] if k[4] == "selection" else ["ADM", "BigGAN"])}
    c.require(seen == expected_cells, "Missing/extra metric cells")
    summaries = c.read_csv(c.OUT / "macro_summary.csv")
    summary_keys = set()
    for row in summaries:
        key = tuple(row[k] for k in ("profile", "model", "calibration", "partition", "generator"))
        c.require(key not in summary_keys, "Duplicate macro summary"); summary_keys.add(key)
        c.require(np.isclose(float(row["macro_auroc"]), np.mean(macro[key]), atol=1e-12, rtol=0), "Macro AUROC mismatch")
    c.require(summary_keys == set(macro), "Missing macro summaries")
    for row in c.read_csv(c.OUT / "gate_weight_summary.csv"):
        key = tuple(row[k] for k in ("profile", "model", "calibration", "seed", "partition", "condition"))
        a = np.array([float(r["alpha_global"]) for r in groups[key] if r["generator"] in ("Real", row["generator"])])
        checks = dict(mean=a.mean(), std=a.std(), min=a.min(), p05=np.quantile(a, .05), median=np.median(a), p95=np.quantile(a, .95), max=a.max())
        for name, value in checks.items():
            c.require(np.isclose(float(row[name]), value, atol=1e-12, rtol=0), "Gate weight summary mismatch")
    for profile in c.PROFILES:
        settings = c.read(c.OUT / f"resnet_{profile}_settings.json")
        history = settings["history"]
        c.require([r["epoch"] for r in history] == list(range(1, 11)), "ResNet did not finish all ten epochs")
        best = max(history, key=lambda r: r["selection_macro_auroc"])
        c.require(settings["best_epoch"] == best["epoch"], "ResNet best epoch mismatch")
        for r in history:
            expected_lr = 1e-4 * (1 + np.cos(np.pi * r["epoch"] / 10)) / 2
            c.require(np.isclose(r["next_lr"], expected_lr, atol=1e-14), "Incorrect cosine schedule")
        if any(r["batch_size"] < 32 for r in history):
            c.require(settings["oom_events"] and all(e["reason"] == "CUDA OOM" for e in settings["oom_events"]), "Batch reduction without documented OOM")
    if args.bootstrap:
        recomputed, macro_ci, _ = c.summarize(predictions, 2000)
        for old, new in zip(metrics, recomputed):
            for key in new:
                if key.endswith(("_ci_low", "_ci_high")):
                    c.require(np.isclose(float(old[key]), new[key], atol=1e-12, rtol=0), "Bootstrap CI mismatch")
        for old, new in zip(summaries, macro_ci):
            for key in ("ci_low", "ci_high"):
                c.require(np.isclose(float(old[key]), new[key], atol=1e-12, rtol=0), "Macro CI mismatch")
    report = {"status": "passed", "predictions": len(predictions), "metric_cells": len(metrics), "macro_cells": len(summaries),
              "isolation_logs_checked": logs, "test_source_ids_touched": 0, "bootstrap_replayed": args.bootstrap,
              "scope": "Saved predictions/metrics, selection-only thresholds, access logs, immutable artifact hashes; no reserved-test manifest or images opened"}
    if args.report:
        c.write(c.OUT / args.report, report)
    print(c.json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
