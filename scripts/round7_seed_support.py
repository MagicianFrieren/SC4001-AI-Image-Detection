"""Seed/path routing only. The seed-1 model and statistics modules stay unchanged."""
import sys
sys.dont_write_bytecode = True
import copy
import platform
from importlib.metadata import version
import round7_baseline_common as c

SEEDS = (20261101, 20261102, 20261103)
REFERENCE = c.ROOT / "results/round7/baselines_seed1"
PROVENANCE = c.ROOT / "results/round7/seed_repetition_provenance.json"
STAGES = ("prepare", "clip_original", "clip_common128", "resnet_original", "resnet_common128", "summarize")
ROUTING_CODE = ("scripts/round7_run_baselines.py", "scripts/round7_verify_baselines.py")


def number(seed):
    c.require(seed in SEEDS, "Choose one of 20261101, 20261102, 20261103")
    return SEEDS.index(seed) + 1


def reference_protocol():
    provenance = c.read(PROVENANCE)
    c.require(c.sha(REFERENCE / "protocol.json") == provenance["seed1_protocol_sha256"], "Frozen seed-1 protocol changed")
    base = c.read(REFERENCE / "protocol.json")
    for path, digest in base["implementation_sha256"].items():
        if path in ROUTING_CODE:
            archived = provenance["frozen_entrypoints"][path]
            c.require(archived["sha256"] == digest and c.sha(c.ROOT / archived["path"]) == digest, "Original seed-1 entrypoint provenance changed")
        else:
            c.require(c.sha(c.ROOT / path) == digest, f"Model/preprocessing/statistics code changed: {path}")
    for field in ("input_sha256", "inherited_code_sha256"):
        for path, digest in base[field].items():
            c.require(c.sha(c.ROOT / path) == digest, f"Frozen seed-1 input/inherited code changed: {path}")
    return base


def check_frozen_seed1():
    """Check every saved output/artifact byte, plus original completion-record hashes."""
    reference_protocol()
    provenance = c.read(PROVENANCE)
    expected = provenance["seed1_files_sha256"]
    paths = set()
    for folder in (REFERENCE, c.ROOT / "artifacts/round7/baselines_seed1"):
        paths.update(p.relative_to(c.ROOT).as_posix() for p in folder.rglob("*") if p.is_file())
    c.require(paths == set(expected), "Files added to or removed from frozen seed-1 folders")
    for path, digest in expected.items():
        c.require(c.sha(c.ROOT / path) == digest, f"Seed-1 bytes changed: {path}")
    for stage in STAGES:
        marker = c.read(REFERENCE / f"{stage}_complete.json")
        c.require(marker["status"] == "completed" and marker["protocol_sha256"] == provenance["seed1_protocol_sha256"], "Seed-1 completion record invalid")
        for path, digest in marker["files_sha256"].items():
            c.require(expected.get(path) == digest, f"Saved seed-1 hash mismatch: {path}")
    return {"status": "passed", "seed1_files_unchanged": len(expected), "saved_hashes_verified": provenance["saved_hashes_verified"]}


def context():
    c.require(c.PIL.__version__ == "12.2.0", "Repository-pinned Pillow 12.2.0 required")
    step_b = c.read(c.DATA / "data_protocol_step_b.json")
    c.require(c.sha(c.DATA / "data_protocol.json") == step_b["base_protocol_sha256"], "Base data protocol changed")
    for name, digest in step_b["files_sha256"].items():
        c.require(c.sha(c.ROOT / name) == digest, f"Step B input changed: {name}")
    # Read ID-only isolation metadata, never the reserved-test manifest.
    isolation = c.read(c.DATA / "step_b_isolation.json")
    fitting = c.rows(c.DATA / "fitting_pool_manifest.jsonl")
    development = c.rows(c.DATA / "development_manifest.jsonl")
    name = f"assignments_seed{number(c.SEED)}.json"
    assignment = c.read(c.DATA / name)
    data_protocol = c.read(c.DATA / "data_protocol.json")
    c.require(c.sha(c.DATA / name) == data_protocol["files_sha256"][f"results/round7/{name}"], "Selected assignment hash mismatch")
    c.require(assignment["seed"] == c.SEED, "Selected assignment seed mismatch")
    c.require(assignment["fitting_pool_sha256"] == c.sha(c.DATA / "fitting_pool_manifest.jsonl"), "Fixed fitting pool changed")
    c.require(assignment["reserved_test_sha256"] == isolation["reserved_test_sha256"], "Reserved test identity differs across seeds")
    parts = dict(assignment["partitions"], development=[r["source_id"] for r in development])
    records = {r["source_id"]: r for r in fitting + development}
    deny, allow = set(isolation["reserved_test_source_ids"]), set(isolation["allowed_source_ids"])
    c.require(set(records) == allow and not allow & deny, "Test isolation failure")
    c.require(set(sum(parts.values(), [])) == allow and sum(map(len, parts.values())) == len(allow), "Partition overlap/coverage failure")
    c.require({r["generator"] for r in fitting} == {"Real", "SD15"}, "Unexpected fitting generator")
    c.require({r["generator"] for r in development} == {"Real", "ADM", "BigGAN"}, "Final-only domain touched")
    for part, expected in {"expert_fit": 750, "gate_fit": 250, "calibration": 125, "selection": 125}.items():
        for label in (0, 1):
            c.require(sum(records[i]["label"] == label for i in parts[part]) == expected, "Fitting partition count changed")
    return records, parts, deny


def expected_protocol(seed):
    """Copy the measured seed-1 protocol; alter only seed, routing and provenance."""
    base = reference_protocol()
    if seed == SEEDS[0]:
        return base
    i = number(seed)
    result = copy.deepcopy(base)
    result["id"] = f"round7_step_c_seed{i}_clean"
    result["seed"] = seed
    result["resnet"]["seed"] = seed
    result["input_sha256"].pop("results/round7/assignments_seed1.json")
    name = f"results/round7/assignments_seed{i}.json"
    result["input_sha256"][name] = c.sha(c.ROOT / name)
    result["implementation_sha256"] = {f"scripts/{name}": c.sha(c.ROOT / "scripts" / name) for name in [*c.CODE, "round7_seed_support.py"]}
    result["outputs"] = base["outputs"].replace("baselines_seed1", f"baselines_seed{i}")
    result["seed1_reference"] = {"protocol_sha256": c.sha(REFERENCE / "protocol.json"), "provenance_sha256": c.sha(PROVENANCE),
                                "change_scope": "Partition/run seed and output paths only; ResNet initialization/epoch shuffling use the selected run seed. CLIP logistic, PatchShuffle, gate initialization and bootstrap seeds stay fixed."}
    return result


def assert_identical_settings(candidate, seed):
    base = reference_protocol()
    if seed == SEEDS[0]:
        c.require(candidate == base, "Seed-1 protocol changed")
        return
    permitted = {"id", "seed", "input_sha256", "implementation_sha256", "outputs", "seed1_reference"}
    for key in set(base) | set(candidate):
        if key in permitted:
            continue
        if key == "resnet":
            a, b = copy.deepcopy(base[key]), copy.deepcopy(candidate[key])
            c.require(b.pop("seed") == seed, "Wrong ResNet run seed")
            a.pop("seed")
            c.require(a == b, "ResNet setting changed")
        else:
            c.require(base.get(key) == candidate.get(key), f"Seed-1 setting changed: {key}")
    c.require(candidate == expected_protocol(seed), "Seed routing/provenance differs from the approved protocol")


def ensure_protocol():
    context()
    expected = expected_protocol(c.SEED)
    assert_identical_settings(expected, c.SEED)
    path = c.OUT / "protocol.json"
    if path.exists():
        c.require(c.read(path) == expected, "Protocol/code changed; refusing run/cache reuse")
    else:
        c.require(c.SEED != SEEDS[0], "Seed 1 is read-only")
        c.write(path, expected)
    return expected


def configure(seed):
    i = number(seed)
    c.SEED = seed
    c.OUT = c.ROOT / f"results/round7/baselines_seed{i}"
    c.ART = c.ROOT / f"artifacts/round7/baselines_seed{i}"
    c.context = context
    c.protocol = lambda: expected_protocol(c.SEED)
    c.ensure_protocol = ensure_protocol


def check_environment():
    """Prevent accidental software drift between actual repeat runs."""
    reference = c.read(REFERENCE / "summarize_complete.json")["environment"]
    actual = {"python": platform.python_version(), "pillow": c.PIL.__version__, "numpy": c.np.__version__}
    for package in ("torch", "torchvision", "transformers", "scikit-learn", "scipy", "pyarrow"):
        actual[package] = version(package)
    for name, value in actual.items():
        c.require(reference[name] == value, f"Environment drift for {name}: seed1={reference[name]}, current={value}")


def check_completed_seed(seed):
    """Read-only validation for summaries, independent of whichever seed is active."""
    directory = c.ROOT / f"results/round7/baselines_seed{number(seed)}"
    if not (directory / "protocol.json").exists():
        raise FileNotFoundError(f"Seed {seed} has not been initialized: {directory}")
    p = c.read(directory / "protocol.json")
    assert_identical_settings(p, seed)
    for path, digest in p["input_sha256"].items():
        c.require(c.sha(c.ROOT / path) == digest, f"Seed input hash mismatch: {path}")
    provenance = {}
    for stage in STAGES:
        marker_path = directory / f"{stage}_complete.json"
        if not marker_path.exists():
            raise FileNotFoundError(f"Seed {seed} incomplete: {stage}")
        marker = c.read(marker_path)
        c.require(marker["status"] == "completed" and marker["protocol_sha256"] == c.sha(directory / "protocol.json"), "Invalid stage completion")
        for path, digest in marker["files_sha256"].items():
            if path not in provenance:
                c.require(c.sha(c.ROOT / path) == digest, f"Completed seed artifact changed: {path}")
                provenance[path] = digest
    return directory
