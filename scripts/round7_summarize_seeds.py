"""Three-seed descriptive summaries; saved metrics only, no test images or scoring."""
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import hashlib
import io
import json
import os
import uuid
from collections import defaultdict
from pathlib import Path
import numpy as np
import round7_baseline_common as c
import round7_seed_support as support

MODELS = ("global", "local", "average", "fixed", "score_gate", "quality_gate", "resnet50")
GENERATORS = {"SD15": "selection", "ADM": "development", "BigGAN": "development"}
CALIBRATIONS = ("raw", "temperature")
PAIRS = (("average", "global"), ("score_gate", "average"), ("quality_gate", "average"), ("local", "global"), ("resnet50", "global"))
KEYS = ("profile", "model", "calibration", "partition", "generator", "condition")
OUT = c.ROOT / "results/round7/seeds_summary"


def expected_keys(models=MODELS):
    return {(p, m, cal, part, g, condition) for p in c.PROFILES for m in models for cal in CALIBRATIONS
            for g, part in GENERATORS.items() for condition in c.CONDITIONS}


def aggregate_seed(metrics, macro, weights, run_seed):
    """One value per RUN seed; gate initializations are not extra replicates."""
    grouped = defaultdict(dict)
    for row in metrics:
        key = tuple(row[k] for k in KEYS)
        initialization = int(row["seed"])
        c.require(initialization not in grouped[key], "Duplicate model/initialization metric")
        value = float(row["auroc"])
        c.require(np.isfinite(value) and 0 <= value <= 1, "Invalid AUROC")
        grouped[key][initialization] = value
    c.require(set(grouped) == expected_keys(), "Missing/extra per-condition model/profile/calibration/domain cells")
    condition_values = {}
    for key, by_init in grouped.items():
        model = key[1]
        expected = set(c.GATE_SEEDS) if model.endswith("gate") else {run_seed} if model == "resnet50" else {-1}
        c.require(set(by_init) == expected, "Incorrect initialization coverage or ResNet run seed")
        condition_values[key] = float(np.mean(list(by_init.values())))
    macro_values = {}
    for key in {k[:-1] for k in condition_values}:
        macro_values[key] = float(np.mean([condition_values[key + (condition,)] for condition in c.CONDITIONS]))
    saved_macro = {}
    for row in macro:
        key = tuple(row[k] for k in KEYS[:-1])
        c.require(key not in saved_macro, "Duplicate saved macro cell")
        saved_macro[key] = float(row["macro_auroc"])
    c.require(set(saved_macro) == set(macro_values), "Missing saved macro cells")
    for key, value in macro_values.items():
        c.require(np.isclose(value, saved_macro[key], atol=1e-12, rtol=0), "Macro does not equal the four-condition mean")
    grouped_weights = defaultdict(dict)
    for row in weights:
        key = tuple(row[k] for k in KEYS)
        initialization = int(row["seed"])
        c.require(initialization not in grouped_weights[key], "Duplicate gate-weight cell")
        value = float(row["mean"])
        c.require(np.isfinite(value) and 0 <= value <= 1, "Invalid global-branch weight")
        grouped_weights[key][initialization] = value
    c.require(set(grouped_weights) == expected_keys(("score_gate", "quality_gate")), "Incomplete gate weight coverage")
    weight_values = {}
    for key, by_init in grouped_weights.items():
        c.require(set(by_init) == set(c.GATE_SEEDS), "Incorrect gate-weight initialization coverage")
        weight_values[key] = float(np.mean(list(by_init.values())))
    return condition_values, macro_values, weight_values


def stats(values):
    c.require(len(values) == 3 and np.isfinite(values).all(), "Exactly three finite run-seed values required")
    mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
    return {**{f"seed_{seed}": float(v) for seed, v in zip(support.SEEDS, values)},
            "mean": mean, "sd": sd, "mean_plus_minus_sd": f"{mean:.6f} ± {sd:.6f}", "n_seeds": 3}


def build_tables(inputs):
    c.require(set(inputs) == set(support.SEEDS), "All three seeds are required; no partial summary")
    collapsed = {seed: aggregate_seed(*inputs[seed], seed) for seed in support.SEEDS}
    conditions, macros, weights, contrasts = [], [], [], []
    for slot, columns, target in ((0, KEYS, conditions), (1, KEYS[:-1], macros), (2, KEYS, weights)):
        keys = sorted(collapsed[support.SEEDS[0]][slot])
        for key in keys:
            values = [collapsed[seed][slot][key] for seed in support.SEEDS]
            target.append(dict(zip(columns, key), **stats(values)))
    for slot, level, condition_list in ((0, "per_condition", c.CONDITIONS), (1, "macro", ["macro_4_conditions"])):
        def value(seed, profile, model, calibration, partition, generator, condition):
            key = (profile, model, calibration, partition, generator)
            return collapsed[seed][slot][key + (condition,) if slot == 0 else key]
        for cal in CALIBRATIONS:
            for generator, partition in GENERATORS.items():
                for condition in condition_list:
                    for profile in c.PROFILES:
                        for left, right in PAIRS:
                            differences = [value(seed, profile, left, cal, partition, generator, condition) - value(seed, profile, right, cal, partition, generator, condition) for seed in support.SEEDS]
                            contrasts.append(dict(level=level, contrast=f"{left} - {right}", left_profile=profile, right_profile=profile,
                                                  left_model=left, right_model=right, calibration=cal, partition=partition, generator=generator, condition=condition,
                                                  **stats(differences)))
                    for model in MODELS:
                        differences = [value(seed, "common128", model, cal, partition, generator, condition) - value(seed, "original", model, cal, partition, generator, condition) for seed in support.SEEDS]
                        contrasts.append(dict(level=level, contrast="common128 - original", left_profile="common128", right_profile="original",
                                              left_model=model, right_model=model, calibration=cal, partition=partition, generator=generator, condition=condition,
                                              **stats(differences)))
    return {"per_condition_auroc.csv": conditions, "macro_auroc.csv": macros,
            "paired_differences.csv": contrasts, "gate_global_weight_means.csv": weights}


def csv_bytes(rows):
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    return text.getvalue().encode("utf-8")


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def render_summary(tables):
    text = ["# Round 7: three-seed summary", "",
            "Run/partition seeds: 20261101, 20261102, 20261103. Mean ± sample SD (ddof=1), n=3 runs. Each gate's three initialization-specific AUROCs/weights are averaged WITHIN its run first; they are not nine independent experiments.", "",
            "The fitting pool and development set are shared across runs. SD15 values are selection-set estimates, with selection optimism; ADM/BigGAN are previously inspected development domains. These SDs describe run-to-run variation, not population confidence intervals or significance tests. No reserved test data was used.", "",
            "## Four-condition macro AUROC", "", "| Profile | Calibration | Model | SD15 selection | ADM development | BigGAN development |", "|---|---|---|---:|---:|---:|"]
    macro = {(r["profile"], r["model"], r["calibration"], r["generator"]): r for r in tables["macro_auroc.csv"]}
    for profile in c.PROFILES:
        for cal in CALIBRATIONS:
            for model in MODELS:
                values = [macro[profile, model, cal, g]["mean_plus_minus_sd"] for g in GENERATORS]
                text.append(f"| {profile} | {cal} | {model} | " + " | ".join(values) + " |")
    text += ["", "## Per-condition AUROC", "", "All four conditions are shown below so macro averages do not hide JPEG/resize effects. CSV files also contain each individual run's value.", ""]
    cells = {(r["profile"], r["model"], r["calibration"], r["generator"], r["condition"]): r for r in tables["per_condition_auroc.csv"]}
    for profile in c.PROFILES:
        for cal in CALIBRATIONS:
            for generator in GENERATORS:
                text += [f"### {profile} / {cal} / {generator}", "", "| Model | Clean | JPEG70 | JPEG50 | Resize ½ |", "|---|---:|---:|---:|---:|"]
                for model in MODELS:
                    text.append(f"| {model} | " + " | ".join(cells[profile, model, cal, generator, condition]["mean_plus_minus_sd"] for condition in c.CONDITIONS) + " |")
                text.append("")
    text += ["## Paired differences and gate weights", "",
             "`paired_differences.csv` reports all five requested model contrasts and common128 − original, separately for every condition and for the four-condition macro. Each difference is computed within a run before its three-run mean/SD; a positive value favours the left model/profile. No independent-error SD propagation or p-values are used.", "",
             "`gate_global_weight_means.csv` reports global-branch weight means for each gate/profile/calibration/domain/condition, each run's value, and mean ± sample SD over runs. Within-run initialization means refer to the same real+fake comparison population as the saved seed-level gate-weight summary.", ""]
    return "\n".join(text).encode("utf-8")


def publish(path, payload, resume):
    """Resume partial publication by validating existing bytes; never overwrite."""
    if path.exists():
        if not resume:
            raise FileExistsError(f"Output exists: {path}; use --resume")
        c.require(path.read_bytes() == payload, f"Existing summary differs; refusing overwrite: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
    if os.name == "nt":
        temporary.rename(path)
    else:
        os.link(temporary, path); temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Validate existing output bytes and complete missing summary files")
    parser.add_argument("--verify-only", action="store_true", help="Recompute and check existing summaries, writing nothing")
    args = parser.parse_args()
    # Fail before creating any output if even one run is incomplete.
    inputs, hashes = {}, {}
    for seed in support.SEEDS:
        folder = support.check_completed_seed(seed)
        files = [folder / name for name in ("metrics.csv", "macro_summary.csv", "gate_weight_summary.csv")]
        inputs[seed] = tuple(c.read_csv(path) for path in files)
        for path in files + [folder / "protocol.json", *[folder / f"{stage}_complete.json" for stage in support.STAGES]]:
            hashes[path.relative_to(c.ROOT).as_posix()] = c.sha(path)
    frozen = support.check_frozen_seed1()
    tables = build_tables(inputs)
    metadata = {"seeds": list(support.SEEDS), "n_runs": 3, "sd_ddof": 1,
                "gate_aggregation": "Average fixed three gate initializations within each run, then mean/sample SD across three run seeds",
                "paired_differences": "Subtract left minus right within each run, then summarize the three differences",
                "source_sha256": hashes, "seed1_preservation": frozen,
                "code_sha256": {f"scripts/{name}": c.sha(c.ROOT / "scripts" / name) for name in ("round7_summarize_seeds.py", "round7_seed_support.py")},
                "limitations": "Three overlapping fitting-pool partitions and the same development images; SD is descriptive, not a confidence interval. SD15 is selection, ADM/BigGAN previously inspected development. No test data or model scoring."}
    payloads = {name: csv_bytes(rows) for name, rows in tables.items()}
    payloads.update({"SUMMARY.md": render_summary(tables), "provenance.json": json_bytes(metadata)})
    payloads["complete.json"] = json_bytes({"status": "completed", "row_counts": {name: len(rows) for name, rows in tables.items()},
                                           "files_sha256": {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}})
    for name, payload in payloads.items():
        if args.verify_only:
            c.require((OUT / name).is_file() and (OUT / name).read_bytes() == payload, f"Summary check failed: {name}")
        else:
            publish(OUT / name, payload, args.resume)
    print(json.dumps({"status": "verified" if args.verify_only else "completed", "output": str(OUT), "row_counts": {k: len(v) for k, v in tables.items()}}, indent=2))


if __name__ == "__main__":
    main()
