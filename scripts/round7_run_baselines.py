"""Step C runner. New paths only; no reserved-test manifest or test-image access."""
import sys
sys.dont_write_bytecode = True
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import platform
from pathlib import Path
import round7_baseline_common as c


def completed(stage, resume):
    path = c.OUT / f"{stage}_complete.json"
    if not path.exists():
        return False
    if not resume:
        raise FileExistsError(f"{stage} already complete. Use --resume to validate and skip it.")
    info = c.read(path)
    c.require(info["protocol_sha256"] == c.sha(c.OUT / "protocol.json"), "Completed stage has a different protocol")
    for name, digest in info["files_sha256"].items():
        c.require(c.sha(c.ROOT / name) == digest, f"Completed artifact changed: {name}")
    print(f"Verified completed stage: {stage}", flush=True)
    return True


def seal(stage, paths, log):
    import torch
    from importlib.metadata import version, PackageNotFoundError
    versions = {}
    for name in ("torchvision", "transformers", "scikit-learn", "scipy", "pyarrow"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"
    c.write(c.OUT / f"{stage}_complete.json", {"status": "completed", "protocol_sha256": c.sha(c.OUT / "protocol.json"),
            "files_sha256": {p.relative_to(c.ROOT).as_posix(): c.sha(p) for p in list(paths) + [log]},
            "environment": {"python": platform.python_version(), "pillow": c.PIL.__version__, "numpy": c.np.__version__,
                            "torch": torch.__version__, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", **versions}})


def run_stage(stage, records, parts, deny, protocol, resume):
    if completed(stage, resume):
        return
    c.require(c.SEED != 20261101, "Completed seed 1 is frozen; no new writes or training are permitted")
    guard = c.Guard(records, deny)
    status, paths = "failed", []
    try:
        if stage == "prepare":
            from round7_baseline_data import prepare_images, source_path
            prepare_images(records, guard)
            paths = [source_path(i) for i in records]
        elif stage.startswith("clip_"):
            from round7_baseline_clip import run
            profile = stage[5:]
            run(profile, records, parts, guard, protocol)
            paths = list(c.OUT.glob(stage + "_*.csv")) + list(c.OUT.glob(stage + "_*.json"))
            resource = c.read(c.OUT / f"{stage}_resources.json")
            paths += [c.ROOT / name for name in resource["feature_sha256"]]
            paths += [c.ART / f"clip_{profile}_experts.joblib"]
        elif stage.startswith("resnet_"):
            from round7_baseline_resnet import run
            profile = stage[7:]
            run(profile, records, parts, guard, protocol, resume)
            paths = list(c.OUT.glob(stage + "_*.csv")) + list(c.OUT.glob(stage + "_*.json"))
            resource = c.read(c.OUT / f"{stage}_resources.json")
            paths += [c.ROOT / name for name in resource["checkpoint_sha256"]]
        else:
            predictions = []
            for prefix in ("clip", "resnet"):
                for profile in c.PROFILES:
                    c.require(completed(f"{prefix}_{profile}", True), "All model stages must complete before summary")
                    values = c.read_csv(c.OUT / f"{prefix}_{profile}_predictions.csv")
                    guard.touch([r["source_id"] for r in values], "metrics_input")
                    predictions += values
            metrics, macro, weights = c.summarize(predictions, protocol["metrics"]["bootstrap_draws"])
            c.write_csv(c.OUT / "metrics.csv", metrics)
            c.write_csv(c.OUT / "macro_summary.csv", macro)
            c.write_csv(c.OUT / "gate_weight_summary.csv", weights)
            table = [f"# Step C results: seed {c.SEED}, clean experts", "", "Macro AUROC averages all four conditions. Gates average three initialization-specific AUROCs. SD15 uses selection images, so its estimates are optimistic after model/threshold selection. ADM/BigGAN are development, not blind test domains.", "",
                     "| Profile | Calibration | Model | SD15 selection | ADM development | BigGAN development |", "|---|---|---|---:|---:|---:|"]
            lookup = {(r["profile"], r["calibration"], r["model"], r["generator"]): r for r in macro}
            for profile in c.PROFILES:
                for cal in ("raw", "temperature"):
                    for model in ("global", "local", "average", "fixed", "score_gate", "quality_gate", "resnet50"):
                        values = [lookup[profile, cal, model, g] for g in ("SD15", "ADM", "BigGAN")]
                        table.append(f"| {profile} | {cal} | {model} | " + " | ".join(f"{v['macro_auroc']:.4f} [{v['ci_low']:.4f}, {v['ci_high']:.4f}]" for v in values) + " |")
            with (c.OUT / "SUMMARY.md").open("x", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(table) + "\n")
            paths = [c.OUT / name for name in ("metrics.csv", "macro_summary.csv", "gate_weight_summary.csv", "SUMMARY.md")]
        status = "completed"
    finally:
        log = guard.save(stage, status)
    seal(stage, paths, log)


def main():
    import round7_seed_support as seeds
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=seeds.SEEDS, default=20261101)
    parser.add_argument("--stage", choices=["init", "prepare", "clip", "resnet", "summarize", "all"], default="all")
    parser.add_argument("--profile", choices=["both", *c.PROFILES], default="both")
    parser.add_argument("--resume", action="store_true", help="Validate/reuse finished stages and resume ResNet epoch checkpoints; never overwrite outputs")
    args = parser.parse_args()
    seeds.configure(args.seed)
    seeds.check_environment()
    protocol = c.ensure_protocol()
    records, parts, deny = c.context()
    if args.stage == "init":
        print("Protocol fixed; no weights downloaded, images decoded or models run.")
        return
    profiles = c.PROFILES if args.profile == "both" else [args.profile]
    stages = []
    if args.stage in ("all", "prepare"):
        stages.append("prepare")
    if args.stage in ("all", "clip"):
        stages += ["clip_" + p for p in profiles]
    if args.stage in ("all", "resnet"):
        stages += ["resnet_" + p for p in profiles]
    if args.stage == "summarize" or args.stage == "all" and args.profile == "both":
        stages.append("summarize")
    for stage in stages:
        if stage not in ("prepare", "summarize"):
            c.require(completed("prepare", True), "Run --stage prepare first")
        print(f"Starting {stage}", flush=True)
        run_stage(stage, records, parts, deny, protocol, args.resume)


if __name__ == "__main__":
    main()
