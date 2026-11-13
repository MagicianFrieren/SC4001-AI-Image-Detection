"""Stage and rerun supplied saved-result analyses in an isolated, fresh tree.

Only the four named lightweight analyses are executable here. Historical
training/extraction scripts are archived byte-for-byte and listed by `historical`;
this runner never removes their guards or dispatches them. It uses no network.

Example (from an unpacked submission):
  python source_tree/scripts/submission_revision_20261114/reproduce.py --package . list
  python source_tree/scripts/submission_revision_20261114/reproduce.py --package . run report-tables --workspace ../rebuild-tables
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ANALYSIS = "output/analysis/"
MULTI = ANALYSIS + "multigen_gate_20261108/run_v1_codefix1/"
MECHANISM = ANALYSIS + "gate_failure_mechanism_20261108/"
REPORT = "scripts/report_evidence_20261111/"
COMMANDS = {
    "report-tables": {
        "entrypoint": REPORT + "build_evidence.py", "modules": ["numpy", "pandas"],
        "output": "generated/report-tables", "receipt": "table_verification.json", "status": "passed",
        "description": "Rebuild report tables and threshold-error decomposition from saved aggregates; no image/model loading.",
    },
    "report-figures": {
        "entrypoint": REPORT + "make_figures.py", "modules": ["numpy", "pandas", "matplotlib"],
        "output": "generated/report-figures", "receipt": "figure_verification.json", "status": "passed",
        "description": "Render the two archived report figures from saved aggregates.",
    },
    "mechanism-predictions": {
        "entrypoint": "tmp/gate_failure_mechanism_20261108/prediction_diagnostics.py",
        "modules": ["numpy", "pandas", "scipy", "sklearn"],
        "output": MECHANISM.rstrip("/"), "receipt": "prediction_summary.json", "status": "completed",
        "description": "Rebuild frozen-score ranking/weight diagnosis from development predictions; no fitting.",
    },
    "mechanism-bootstrap": {
        "entrypoint": REPORT + "supplement_mechanism.py", "modules": ["numpy", "pandas", "scipy", "sklearn"],
        "output": "generated/mechanism-bootstrap", "receipt": "verification.json", "status": "passed",
        "description": "Repeat the fixed 1,000-draw paired development-score decomposition; no fitting.",
    },
}
FIGURE_INPUTS = [
    "gate_diagnosis_20261108/method_comparison.csv",
    "multigen_gate_20261108/run_v1_codefix1/metrics_seed_condition.csv",
    "multigen_gate_20261108/run_v1_codefix1/summary.json",
    "recompression_reliability_20261108/reliability_cells.csv",
    "recompression_reliability_20261108/detection_diagnostics.csv",
    "recompression_reliability_20261108/summary.json",
    "paired_view_global_20261108/metrics.csv",
    "gate_failure_mechanism_20261108/source_target_transfer.csv",
    "gate_failure_mechanism_20261108/prediction_ranking_seed_means.csv",
    "gate_failure_mechanism_20261108/prediction_constant_seed_means.csv",
]
HISTORICAL = [
    ("checkpoint replay", "tmp/gate_diagnosis/checkpoint_diagnostic.py", "Retrains/replays gate trajectories; needs archived R8 bundles, settings and completion metadata."),
    ("original-view gate", "tmp/gate_diagnosis/original_gate_views/run_original_gate_views.py", "Fits gates; needs R8 original-view feature caches and experts/bundles."),
    ("bounded ranking", "tmp/gate_diagnosis/ranking_gate/run_ranking_gate.py", "Fits gates; needs original R8 bundles and metadata."),
    ("condition routing", "tmp/gate_diagnosis/condition_gate/run_condition_gate.py", "Fits a quality classifier and selects condition weights; needs original caches/bundles."),
    ("multi-generator gate", "tmp/multigen_gate_20261108/run_study.py", "Fits 54 gates; needs R8 caches, experts, partition manifests, protocol plan and preflight."),
    ("quality-feature diagnosis", "tmp/gate_failure_mechanism_20261108/feature_diagnostics.py", "Uses frozen gates and source/target cached inputs; this is heavier than saved prediction analysis."),
    ("objective diagnosis", "tmp/gate_failure_mechanism_20261108/objective_diagnostics.py", "Uses frozen gates, original expert bundles and cached inputs; no lightweight-only completeness claim."),
    ("recompression extraction", "tmp/recompression_reliability_20261108/extract_recompression.py", "Needs original images, fixed CLIP weights, CUDA environment, preflight/canary; executes new feature extraction."),
    ("recompression analysis", "tmp/recompression_reliability_20261108/analyze_reliability.py", "Needs extracted recompression feature caches, R8 caches/experts and source-only gate models."),
    ("paired-view global fitting", "tmp/paired_view_global_20261108/run_study.py", "Fits three global heads; needs both original/matched R8 cached features, metadata, plan and preflight."),
    ("original diagnosis summary", "tmp/gate_diagnosis/summarize_diagnosis.py", "Archived checkpoint freeze contains absolute paths. Run only after a genuinely new checkpoint replay creates its own freeze; do not rewrite the old freeze or remove markers. Use report-tables for saved-result reconstruction."),
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Manifest paths must be nonempty strings.")
    value = value.replace("\\", "/")
    p = PurePosixPath(value)
    if p.is_absolute() or PureWindowsPath(value).drive or ".." in p.parts:
        raise ValueError(f"Path must be confined and relative: {value}")
    return p.as_posix()


def inside(root, rel):
    path = (root / relative(rel)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes its declared root: {rel}")
    return path


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve_supplied(package, logical):
    candidates = ["source_tree/" + logical, "evidence/sources/" + logical]
    available = [(name, inside(package, name)) for name in candidates if inside(package, name).is_file()]
    if not available:
        return None
    if len(available) > 1 and len({digest(p) for _, p in available}) != 1:
        raise ValueError(f"Conflicting supplied bytes at source_tree and evidence/sources: {logical}")
    source, path = available[0]
    return dict(path=logical, source=source, sha256=digest(path), bytes=path.stat().st_size, status="supplied")


def input_groups(package):
    report_manifest = inside(package, "evidence/source_manifest.json")
    if not report_manifest.is_file():
        raise FileNotFoundError("Supply the archived evidence/source_manifest.json and its snapshots first.")
    documentary = read_json(report_manifest)
    report_inputs = [relative(x["source"]) for x in documentary["sources"]]
    groups = {
        "report-tables": report_inputs,
        "report-figures": [ANALYSIS + name for name in FIGURE_INPUTS],
        "mechanism-predictions": [MULTI + name for name in ["complete.json", "predictions.csv.gz", "metrics_seed_condition.csv", "summary.json"]],
        "mechanism-bootstrap": [MULTI + name for name in ["complete.json", "predictions.csv.gz", "metrics_seed_condition.csv"]] + [MECHANISM + name for name in ["prediction_constant_alpha.csv", "prediction_ranking_comparisons.csv"]],
    }
    for command, spec in COMMANDS.items():
        groups[command] = sorted(set(groups[command] + [spec["entrypoint"]]))
    return documentary, groups


def create_manifest(package, manifest_path):
    documentary, groups = input_groups(package)
    report_manifest = inside(package, "evidence/source_manifest.json")
    needed = sorted(set().union(*map(set, groups.values())))
    files = [resolve_supplied(package, p) or dict(path=p, source=None, sha256=None, bytes=None, status="missing") for p in needed]
    # Verify supplied documentary snapshots against their archived hashes as well.
    indexed = {r["path"]: r for r in files}
    for row in documentary["sources"]:
        current = indexed[relative(row["source"])]
        if current["status"] == "supplied" and current["sha256"] != row["sha256"]:
            raise ValueError(f"Archived source hash mismatch: {row['source']}")
    data = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_resolution": "Package-relative source_tree or evidence/sources; stages conceptual repository-relative paths into a fresh workspace.",
        "files": files, "command_inputs": groups,
        "documentary_manifest": {"source": "evidence/source_manifest.json", "sha256": digest(report_manifest)},
        "scope": "Reconstruction from supplied saved results, not a claim that historical training/extraction was re-executed.",
        "historical_execution": "Not dispatched; see historical catalogue and dependency audit. Preserve all completion/freeze markers.",
    }
    write_new(manifest_path, data)
    return dict(status="created", manifest=str(manifest_path), supplied=sum(r["status"] == "supplied" for r in files), missing=[r["path"] for r in files if r["status"] != "supplied"])


def load_manifest(path):
    data = read_json(path)
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema.")
    rows = data["files"]
    indexed = {relative(r["path"]): r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate logical path in manifest.")
    return data, indexed


def preflight(package, manifest_path, command, workspace=None):
    data, files = load_manifest(manifest_path)
    spec = COMMANDS[command]
    if command not in data["command_inputs"]:
        raise ValueError(f"Manifest has no input group for {command}")
    required = sorted(set(map(relative, data["command_inputs"][command])))
    _, expected = input_groups(package)
    if required != expected[command]:
        raise ValueError(f"Manifest input group is incomplete or stale for {command}; regenerate it from the supplied evidence manifest.")
    documentary = data["documentary_manifest"]
    if digest(inside(package, documentary["source"])) != documentary["sha256"]:
        raise ValueError("Archived evidence manifest changed after reproduction descriptor creation.")
    if spec["entrypoint"] not in required:
        raise ValueError("Entrypoint must be explicitly included in the manifest group.")
    missing, corrupt = [], []
    checked = []
    for logical in required:
        row = files.get(logical)
        if row is None or not row.get("source"):
            missing.append(logical)
            continue
        path = inside(package, row["source"])
        if not path.is_file():
            missing.append(logical)
        elif path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            corrupt.append(logical)
        else:
            checked.append(logical)
    modules = {name: importlib.util.find_spec(name) is not None for name in spec["modules"]}
    versions = {}
    for name in spec["modules"]:
        try:
            versions[name] = importlib.metadata.version("scikit-learn" if name == "sklearn" else name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    workspace_error = None
    if workspace is not None:
        workspace = workspace.resolve()
        if workspace.exists():
            workspace_error = "Workspace already exists; choose a new path. No files or completion markers will be removed."
        elif workspace.is_relative_to(package) or package.is_relative_to(workspace):
            workspace_error = "Choose a workspace outside, and not an ancestor of, the supplied package."
    output = relative(spec["output"])
    overlaps = [p for p in required if p == output or p.startswith(output + "/")]
    if overlaps:
        raise ValueError(f"Output path overlaps staged read-only inputs: {overlaps}")
    return dict(status="ready" if not (missing or corrupt or workspace_error) and all(modules.values()) else "blocked",
                command=command, required_files=len(required), verified_files=len(checked),
                missing=missing, hash_mismatches=corrupt, module_available=modules,
                environment={"python": sys.version, "executable": sys.executable, "versions": versions},
                workspace=str(workspace) if workspace is not None else None, workspace_error=workspace_error,
                input_paths=required, output_relative=output)


def run(package, manifest_path, command, workspace):
    check = preflight(package, manifest_path, command, workspace)
    if check["status"] != "ready":
        return check
    data, files = load_manifest(manifest_path)
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=False)
    for logical in check["input_paths"]:
        row = files[logical]
        source = inside(package, row["source"])
        target = inside(workspace, logical)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(target) != row["sha256"]:
            raise RuntimeError(f"Copy hash mismatch: {logical}")
    logdir = workspace / "reproduction_receipts"
    logdir.mkdir(exist_ok=False)
    spec = COMMANDS[command]
    output = inside(workspace, spec["output"])
    if output.exists():
        raise FileExistsError(f"Refusing existing output tree: {output}")
    argv = [sys.executable, str(inside(workspace, spec["entrypoint"]))]
    if command != "mechanism-predictions":
        argv += ["--root", str(workspace), "--output", str(output)]
    if command == "mechanism-bootstrap":
        argv += ["--repetitions", "1000", "--seed", "20261111"]
    write_new(logdir / "preflight.json", dict(check, actual_command=argv, manifest_sha256=digest(manifest_path)))
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    with (logdir / "stdout.txt").open("x", encoding="utf-8") as stdout, (logdir / "stderr.txt").open("x", encoding="utf-8") as stderr:
        completed = subprocess.run(argv, cwd=workspace, env=environment, stdout=stdout, stderr=stderr, shell=False, check=False)
    changed_staged = [p for p in check["input_paths"] if digest(inside(workspace, p)) != files[p]["sha256"]]
    changed_supplied = [p for p in check["input_paths"] if digest(inside(package, files[p]["source"])) != files[p]["sha256"]]
    receipt_path = output / spec["receipt"]
    analysis_receipt = read_json(receipt_path) if receipt_path.is_file() else None
    receipt_status = analysis_receipt.get("status") if isinstance(analysis_receipt, dict) else None
    passed = completed.returncode == 0 and not changed_staged and not changed_supplied and receipt_status == spec["status"]
    result = dict(status="passed" if passed else "failed", command=command, returncode=completed.returncode,
                  output=str(output), staged_files=len(check["input_paths"]),
                  source_tree_bytes_preserved=not changed_staged and not changed_supplied,
                  changed_staged_inputs=changed_staged, changed_supplied_inputs=changed_supplied,
                  archived_script_receipt_status=receipt_status, completed_utc=datetime.now(timezone.utc).isoformat(),
                  output_sha256={p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob("*")) if p.is_file()} if output.exists() else {},
                  newly_executed_scope=spec["description"], historical_training_extraction_reexecuted=False,
                  stdout=str(logdir / "stdout.txt"), stderr=str(logdir / "stderr.txt"))
    write_new(logdir / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True, help="Unpacked bundle root containing source_tree/, evidence/, reproduction/")
    parser.add_argument("--manifest", type=Path, help="Descriptor path; default PACKAGE/reproduction/reproduction_manifest.json")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("create-manifest", help="Hash supplied inputs and create a new descriptor; executes no analysis")
    sub.add_parser("list", help="List the four supported saved-result commands")
    sub.add_parser("historical", help="List archived heavier entrypoints and their prerequisites; executes none")
    for verb in ["preflight", "run"]:
        p = sub.add_parser(verb)
        p.add_argument("command", choices=sorted(COMMANDS))
        p.add_argument("--workspace", type=Path, required=verb == "run")
    args = parser.parse_args()
    package = args.package.resolve()
    if not package.is_dir():
        parser.error(f"Package directory does not exist: {package}")
    manifest = args.manifest.resolve() if args.manifest else package / "reproduction/reproduction_manifest.json"
    try:
        if args.action == "create-manifest":
            result = create_manifest(package, manifest)
        elif args.action == "historical":
            result = {"status": "archive_catalogue_only", "commands": [dict(name=name, entrypoint=path, original_command=["python", path], supplied=(package / "source_tree" / path).is_file(), requirements=note) for name, path, note in HISTORICAL],
                      "policy": "Never dispatch these commands via this runner. Full reruns require complete dependency restoration and fresh study outputs with original guards intact. They were not rerun for the submission revision."}
        elif args.action == "list":
            result = {"status": "catalogue", "commands": {name: dict(spec, preflight=preflight(package, manifest, name)) for name, spec in COMMANDS.items()}}
        elif args.action == "preflight":
            result = preflight(package, manifest, args.command, args.workspace)
        else:
            result = run(package, manifest, args.command, args.workspace)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] not in ["blocked", "failed"] else 2
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
