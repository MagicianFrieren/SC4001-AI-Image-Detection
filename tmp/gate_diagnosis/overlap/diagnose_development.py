"""Read-only diagnostics of the six Round 8 A development/selection CSVs.

No model loading, fitting, source-image access, or final/reserved-test access.
Writes only a new JSON next to this script; refuses an existing output.
"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_name("diagnosis.json")
assert not OUT.exists(), f"Refusing to overwrite {OUT}"
ALPHA = defaultdict(list)
METRICS, OVERLAP, ROUTING, SETTINGS, TRAINING = [], [], [], [], []
INPUTS = {}


def digest(path):
    INPUTS[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()


def stats(x):
    x = np.asarray(x, dtype=float)
    return {"n": len(x), "mean": float(x.mean()), "within_sd": float(x.std(ddof=1)) if len(x) > 1 else 0.,
            **{f"q{int(q * 100):02}": float(np.quantile(x, q)) for q in (0., .05, .25, .5, .75, .95, 1.)},
            "fraction_below_0_1": float(np.mean(x < .1)), "fraction_above_0_9": float(np.mean(x > .9))}


def aggregate(records, keys, measures):
    """Average initializations within run, then summarize the three run seeds."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in records:
        groups[tuple(r[k] for k in keys)][r["run_seed"]].append(r)
    output = []
    for key, runs in sorted(groups.items()):
        row = dict(zip(keys, key))
        row["run_seeds"] = sorted(runs)
        for m in measures:
            vals = [float(np.mean([r[m] for r in runs[s] if r[m] is not None]))
                    if any(r[m] is not None for r in runs[s]) else None for s in sorted(runs)]
            valid = [v for v in vals if v is not None]
            row[m] = {"mean": float(np.mean(valid)) if valid else None,
                      "seed_sd": float(np.std(valid, ddof=1)) if len(valid) > 1 else None,
                      "each_seed": vals}
        output.append(row)
    return output


def arr(rows):
    rows = sorted(rows, key=lambda r: r["source_id"])
    return {"ids": [r["source_id"] for r in rows],
            **{k: np.array([float(r[k]) for r in rows]) for k in
               ("label", "logit", "probability_fake", "threshold", "alpha_global")}}


for n in (1, 2, 3):
    for arm in ("matched", "unmatched"):
        folder = ROOT / "results/round8_a" / f"seed{n}" / arm
        pred_path, set_path = folder / "predictions.csv", folder / "settings.json"
        digest(pred_path); digest(set_path)
        with pred_path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 46000
        assert {r["partition"] for r in rows} == {"selection", "development"}
        assert {r["generator"] for r in rows} == {"Real", "SD15", "ADM", "BigGAN"}
        run = int(rows[0]["run_seed"])
        settings = json.loads(set_path.read_text(encoding="utf-8"))
        for name, v in settings["settings"].items():
            SETTINGS.append(dict(arm=arm, run_seed=run, model_key=name,
                                 threshold=v["threshold"], fixed_alpha=v["fixed_alpha"],
                                 tuning_alpha=stats(v["tuning_alpha"])))
        for name, v in settings["gates"].items():
            art, history = v["artifact"], v["history"]
            selected = next(r for r in history if r["epoch"] == art["best_epoch"])
            TRAINING.append(dict(arm=arm, run_seed=run, model_key=name,
                                 best_epoch=art["best_epoch"], stopped_epoch=art["stopped_epoch"],
                                 initial_train_bce=history[0]["train_bce"],
                                 initial_validation_bce=history[0]["validation_bce"],
                                 selected_train_bce=selected["train_bce"],
                                 selected_validation_bce=selected["validation_bce"]))
        groups = defaultdict(list)
        for r in rows:
            key = (r["model"], int(r["gate_seed"]), r["partition"], r["condition"], r["generator"])
            groups[key].append(r)
        arrays = {k: arr(v) for k, v in groups.items()}
        for (model, gs, part, cond, source), a in arrays.items():
            ALPHA[(arm, model, part, cond, source)].append((run, gs, a["alpha_global"]))
        for part, gens in (("selection", ("SD15",)), ("development", ("ADM", "BigGAN"))):
            for cond in ("clean", "jpeg70", "jpeg50", "resize05"):
                for gen in gens:
                    models = {}
                    for model, gs in sorted({(r["model"], int(r["gate_seed"])) for r in rows}):
                        real, fake = arrays[(model, gs, part, cond, "Real")], arrays[(model, gs, part, cond, gen)]
                        a = {k: (real[k] + fake[k] if k == "ids" else np.concatenate([real[k], fake[k]])) for k in real}
                        a["correct"] = (a["probability_fake"] >= a["threshold"]) == a["label"]
                        a["bce"] = np.logaddexp(0., a["logit"]) - a["label"] * a["logit"]
                        models[(model, gs)] = a
                        nr = len(real["ids"])
                        METRICS.append(dict(arm=arm, run_seed=run, gate_seed=gs, model=model,
                                            partition=part, condition=cond, generator=gen,
                                            auroc=float(roc_auc_score(a["label"], a["logit"])),
                                            balanced_accuracy=float(a["correct"].mean()),
                                            real_fpr=float(1 - a["correct"][:nr].mean()),
                                            fake_fnr=float(1 - a["correct"][nr:].mean()),
                                            bce=float(a["bce"].mean())))
                    g, l = models[("global", -1)], models[("local", -1)]
                    assert g["ids"] == l["ids"]
                    for source, mask in (("balanced_real_plus_fake", np.ones(len(g["ids"]), dtype=bool)),
                                         ("Real", g["label"] == 0), (gen, g["label"] == 1)):
                        gc, lc = g["correct"][mask], l["correct"][mask]
                        flags = {"both_correct": gc & lc, "global_only_correct": gc & ~lc,
                                 "local_only_correct": ~gc & lc, "both_wrong": ~gc & ~lc}
                        base = dict(arm=arm, run_seed=run, partition=part, condition=cond,
                                    generator=gen, source=source, n=int(mask.sum()))
                        OVERLAP.append(dict(**base, **{k: float(v.mean()) for k, v in flags.items()},
                                            label_oracle_expert_accuracy=float((gc | lc).mean())))
                        for (model, gs), a in models.items():
                            if model not in ("score_gate", "quality_gate"):
                                continue
                            assert a["ids"] == g["ids"]
                            alpha = a["alpha_global"][mask]
                            better_g = g["bce"][mask] < l["bce"][mask]
                            better_l = l["bce"][mask] < g["bce"][mask]
                            not_tied = better_g | better_l
                            w = np.where(better_g, alpha, 1 - alpha)[not_tied]
                            ROUTING.append(dict(**base, model=model, gate_seed=gs,
                                alpha_global_only_correct=float(alpha[flags["global_only_correct"]].mean()) if flags["global_only_correct"].any() else None,
                                alpha_local_only_correct=float(alpha[flags["local_only_correct"]].mean()) if flags["local_only_correct"].any() else None,
                                weight_on_lower_bce_expert=float(w.mean()) if len(w) else None,
                                fraction_favoring_lower_bce_expert=float(np.mean(w > .5)) if len(w) else None,
                                gate_accuracy=float(a["correct"][mask].mean())))

alpha_summary = []
alpha_per_run = []
for key, items in sorted(ALPHA.items()):
    base = dict(zip(("arm", "model", "partition", "condition", "source"), key))
    for run, gs, v in items:
        alpha_per_run.append(dict(**base, run_seed=run, gate_seed=gs, **stats(v)))
    runmeans = [float(np.mean([v.mean() for s, gs, v in items if s == run])) for run in sorted({s for s, gs, v in items})]
    alpha_summary.append(dict(**base, pooled_descriptive=stats(np.concatenate([v for s, gs, v in items])),
                              seed_mean_alpha=runmeans, sd_across_seed_mean_alpha=float(np.std(runmeans, ddof=1))))

result = dict(
    scope="Only results/round8_a/seed1..3/{matched,unmatched}/predictions.csv/settings.json; no final/reserved-test reads.",
    definitions={
        "alpha": "global logit weight; 0=local, 1=global; quantiles pool repeated seed/init predictions descriptively, not independent images.",
        "aggregation": "Gate metrics average initialization-level metrics within seed, then mean/sample SD across the three shared-pool run seeds.",
        "selection": "SD15 original-view reporting uses identities already used in tuning. It is optimistic source-domain reporting, not untouched validation.",
        "development": "Previously inspected ADM/BigGAN, 300 real + 300 fake per comparison and condition; same real pool reused.",
        "overlap": "Each expert is judged with its own saved selection threshold. Union-correct is label-aware diagnostic upper bound; it is not an implementable gate or AUROC bound.",
        "routing": "Weights conditional on each expert's correctness or lower label-aware BCE are descriptive, not proof of attainable selection rule.",
    }, input_sha256=INPUTS,
    alpha_summary_by_source=alpha_summary, alpha_per_run=alpha_per_run,
    model_metrics_each_run=METRICS,
    model_metrics_summary=aggregate(METRICS, ["arm", "model", "partition", "condition", "generator"],
                                    ["auroc", "balanced_accuracy", "real_fpr", "fake_fnr", "bce"]),
    overlap_each_run=OVERLAP,
    overlap_summary=aggregate(OVERLAP, ["arm", "partition", "condition", "generator", "source"],
                              ["both_correct", "global_only_correct", "local_only_correct", "both_wrong", "label_oracle_expert_accuracy"]),
    routing_summary=aggregate(ROUTING, ["arm", "model", "partition", "condition", "generator", "source"],
                              ["alpha_global_only_correct", "alpha_local_only_correct", "weight_on_lower_bce_expert", "fraction_favoring_lower_bce_expert", "gate_accuracy"]),
    settings=SETTINGS, training_history_endpoints=TRAINING,
)
with OUT.open("x", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
print(json.dumps({"output": str(OUT), "prediction_rows": 276000, "input_files": len(INPUTS),
                  "metric_rows": len(METRICS), "alpha_summary_rows": len(alpha_summary)}, ensure_ascii=False))
