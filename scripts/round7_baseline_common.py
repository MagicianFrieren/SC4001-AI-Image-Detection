"""Step C protocol, isolation and statistics. No image/model operations on import."""
import sys
sys.dont_write_bytecode = True
import csv
import hashlib
import json
import os
import time
import uuid
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN = ROOT / "artifacts/round7/pillow_12_2"
if PIN.is_dir():
    sys.path.insert(0, str(PIN))
import numpy as np
import PIL
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score, roc_curve

OUT = ROOT / "results/round7/baselines_seed1"
ART = ROOT / "artifacts/round7/baselines_seed1"
DATA = ROOT / "results/round7"
PROFILES = ["original", "common128"]
CONDITIONS = ["clean", "jpeg70", "jpeg50", "resize05"]
TUNE = ["clean", "jpeg70", "resize05"]
SEED = 20261101
GATE_SEEDS = [20261022, 20261023, 20261024]
CODE = ["round7_baseline_common.py", "round7_baseline_data.py", "round7_baseline_clip.py",
        "round7_baseline_resnet.py", "round7_run_baselines.py", "round7_verify_baselines.py"]


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def write_csv(path, values):
    require(bool(values), "Cannot write empty table")
    with Path(path).open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(values[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(values)


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def context():
    require(PIL.__version__ == "12.2.0", "Repository-pinned Pillow 12.2.0 required")
    step_b = read(DATA / "data_protocol_step_b.json")
    require(sha(DATA / "data_protocol.json") == step_b["base_protocol_sha256"], "Base protocol changed")
    for name, digest in step_b["files_sha256"].items():
        require(sha(ROOT / name) == digest, f"Step B input changed: {name}")
    isolation = read(DATA / "step_b_isolation.json")  # IDs only, NOT the test manifest.
    fitting, development = rows(DATA / "fitting_pool_manifest.jsonl"), rows(DATA / "development_manifest.jsonl")
    assignments = read(DATA / "assignments_seed1.json")
    require(assignments["seed"] == SEED, "Only partition seed 20261101 is permitted")
    parts = dict(assignments["partitions"], development=[r["source_id"] for r in development])
    records = {r["source_id"]: r for r in fitting + development}
    deny = set(isolation["reserved_test_source_ids"])
    allow = set(isolation["allowed_source_ids"])
    require(set(records) == allow and not allow & deny, "Test-isolation allowlist failure")
    require(set(r["generator"] for r in fitting) == {"Real", "SD15"}, "Unexpected fitting domain")
    require(set(r["generator"] for r in development) == {"Real", "ADM", "BigGAN"}, "Final-only domain touched")
    require(set(sum(parts.values(), [])) == allow and sum(map(len, parts.values())) == len(allow), "Partition overlap")
    return records, parts, deny


class Guard:
    def __init__(self, records, deny):
        self.allowed, self.deny = set(records), set(deny)
        self.events = defaultdict(set)
        self.violations = []

    def touch(self, ids, action):
        ids = list(ids)
        bad = (set(ids) & self.deny) | (set(ids) - self.allowed)
        if bad:
            self.violations += sorted(bad)
            raise AssertionError(f"TEST ISOLATION: forbidden source IDs at {action}: {sorted(bad)}")
        self.events[action].update(ids)

    def save(self, stage, status):
        path = ART / "access_logs" / f"{stage}_{uuid.uuid4().hex}.json"
        write(path, {"stage": stage, "status": status, "events": {k: sorted(v) for k, v in self.events.items()},
                     "violations": self.violations, "isolation_sha256": sha(DATA / "step_b_isolation.json")})
        return path


def protocol():
    inputs = ["data_protocol.json", "data_protocol_step_b.json", "step_b_isolation.json",
              "assignments_seed1.json", "fitting_pool_manifest.jsonl", "development_manifest.jsonl", "source_inventory.json"]
    inherited = read(DATA / "data_protocol.json")["inherited_round6"]["file_sha256"]
    for name, digest in inherited.items():
        require(sha(ROOT / name) == digest, f"Inherited Round 6 code changed: {name}")
    return {"id": "round7_step_c_seed1_clean", "seed": SEED, "profiles": PROFILES, "conditions": CONDITIONS,
            "tuning_conditions": TUNE, "expert_training": "expert_fit originals, clean only; no development/test inputs",
            "input_sha256": {f"results/round7/{n}": sha(DATA / n) for n in inputs},
            "implementation_sha256": {f"scripts/{n}": sha(ROOT / "scripts" / n) for n in CODE},
            "inherited_code_sha256": inherited,
            "clip": {"model": "openai/clip-vit-base-patch32", "revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268", "weights_sha256": "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f",
                     "frozen": True, "precision": "FP16 encoder on CUDA as Round 6; embeddings normalized in FP32", "batch_size": 32,
                     "head": "StandardScaler then LogisticRegression(C=1, liblinear, max_iter=2000, random_state=20261018)",
                     "local": "four mean-pooled normalized features from 56px PatchShuffle views; seeds 202610220..202610223"},
            "gates": {"initializations": GATE_SEEDS, "implementation": "unchanged gate_models.py", "fitting": "gate_fit only; minimum pooled selection BCE over clean/jpeg70/resize05, epoch0 eligible"},
            "fusion": "global/local/average/fixed/score_gate/quality_gate; logits fused. Fixed alpha grid 0..1 by .1, max selection SD15 macro AUROC over tuning conditions; ties nearest .5 then smaller.",
            "calibration": "raw and temperature for every model, including ResNet; positive scalar per expert fitted only on calibration, clean/jpeg70/resize05; local variance divided by T squared",
            "threshold": "Per model/profile/calibration/initialization, pooled selection Youden J over clean/jpeg70/resize05; ties closest probability threshold .5 (Round 6); fixed across all four reporting conditions",
            "resnet": {"weights": "torchvision ResNet50_Weights.IMAGENET1K_V2", "all_layers_trainable": True, "optimizer": "AdamW", "lr": 1e-4, "weight_decay": 1e-4,
                       "batch_size": 32, "epochs": 10, "schedule": "CosineAnnealingLR T_max=10 eta_min=0; step once after each epoch", "precision": "FP32", "seed": SEED,
                       "epoch_selection": "Maximum mean selection AUROC over clean/jpeg70/resize05; exact tie retains earliest epoch", "normalization": "ImageNet mean/std after identical RGB geometry/degradation; CLIP uses CLIP mean/std",
                       "oom_policy": "Only CUDA OOM permits batch reduction (halving); rollback whole failed epoch to prior checkpoint, restore RNG/optimizer/scheduler and retry. Record batch changes. Never silently reduce for another reason.",
                       "checkpoint": "Initialization plus every completed epoch; model, optimizer, scheduler, Python/NumPy/Torch/CUDA RNG and history; exclusive filenames; resume latest valid checkpoint"},
            "metrics": {"names": ["auroc", "average_precision", "balanced_accuracy", "real_false_positive_rate"], "bootstrap_draws": 2000, "seed": 20261022,
                        "ci": "95% percentile, class-stratified original-image draws; identical draws paired across models, initializations and conditions. Gate summary averages metrics/replicates across three initializations, not logits.",
                        "macro": "Equal mean over all four conditions; separate SD15 selection, ADM development, BigGAN development; shared real images counted once per pooled dataset",
                        "limitations": "Selection estimates are optimistic after fitting thresholds/epochs/gates; bootstrap conditions on fitted models and chosen settings; no multiplicity correction. ADM/BigGAN development is previously inspected-domain evidence."},
            "test_isolation": "Read only ID-only Step B denylist, never reserved_test_manifest.jsonl. Assert allowlist/denylist before decode, feature use, fitting and predictions. GLIDE/VQDM/Midjourney/Wukong are forbidden before final evaluation.",
            "outputs": "results/round7/baselines_seed1 and artifacts/round7/baselines_seed1 only; exclusive-create outputs, validated read-only resume caches"}


def ensure_protocol():
    context()
    p = protocol()
    path = OUT / "protocol.json"
    if path.exists():
        require(read(path) == p, "Protocol or implementation changed; refusing reuse of existing run")
    else:
        write(path, p)
    return p


def threshold(y, p):
    fpr, tpr, values = roc_curve(y, p)
    valid = np.isfinite(values)
    values, criterion = values[valid], (tpr - fpr)[valid]
    best = np.flatnonzero(np.isclose(criterion, criterion.max()))
    return float(values[best[np.argmin(np.abs(values[best] - .5))]])


def points(y, z, cut):
    p = expit(z)
    return {"auroc": float(roc_auc_score(y, z)), "average_precision": float(average_precision_score(y, z)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p >= cut)),
            "real_false_positive_rate": float(np.mean(p[y == 0] >= cut))}


def bootstrap(y, z, cut, draws=2000):
    # Callers sort by label then source_id. The same draw positions are reused
    # across all conditions/models; real positions are shared between generators.
    n0, n1 = int(sum(y == 0)), int(sum(y == 1))
    require(np.array_equal(y, np.repeat([0, 1], [n0, n1])) and n0 and n1, "Bootstrap needs ordered binary classes")
    rng = np.random.default_rng(20261022)
    ids = np.concatenate([rng.integers(0, n0, (draws, n0)), rng.integers(n0, n0+n1, (draws, n1))], axis=1)
    sample = z[ids]
    ranks = rankdata(sample, axis=1)
    auc = (ranks[:, n0:].sum(axis=1) - n1 * (n1 + 1) / 2) / (n0*n1)
    decisions = expit(sample) >= cut
    fpr = decisions[:, :n0].mean(axis=1)
    ba = (.5 * (1 - fpr + decisions[:, n0:].mean(axis=1)))
    order = np.argsort(-sample, axis=1, kind="stable")
    zz = np.take_along_axis(sample, order, axis=1)
    yy = (order >= n0).astype(float)
    precision = yy.cumsum(axis=1) / np.arange(1, n0+n1+1)
    end = np.concatenate([zz[:, :-1] != zz[:, 1:], np.ones((draws, 1), bool)], axis=1)
    next_end = np.minimum.accumulate(np.where(end, np.arange(n0+n1), n0+n1)[:, ::-1], axis=1)[:, ::-1]
    ap = (np.take_along_axis(precision, next_end, axis=1) * yy).sum(axis=1) / n1
    return {"auroc": auc, "average_precision": ap, "balanced_accuracy": ba, "real_false_positive_rate": fpr}


def summarize(predictions, draws=2000):
    grouped = defaultdict(list)
    for r in predictions:
        grouped[tuple(r[k] for k in ["profile", "model", "calibration", "seed", "partition", "condition"])].append(r)
    metrics, boot_groups, point_groups, weights = [], defaultdict(list), defaultdict(list), []
    for key, values in sorted(grouped.items()):
        p, model, cal, seed, part, condition = key
        for generator in (["SD15"] if part == "selection" else ["ADM", "BigGAN"]):
            cell = sorted([r for r in values if r["generator"] in ("Real", generator)], key=lambda r: (int(r["label"]), r["source_id"]))
            y = np.array([int(r["label"]) for r in cell]); z = np.array([float(r["logit"]) for r in cell])
            cut = float(cell[0]["threshold"])
            require(all(float(r["threshold"]) == cut for r in cell), "Inconsistent threshold")
            point, boot = points(y, z, cut), bootstrap(y, z, cut, draws)
            item = dict(profile=p, model=model, calibration=cal, seed=seed, partition=part, condition=condition, generator=generator, n=len(cell), threshold=cut, **point)
            for name, dist in boot.items():
                item[name + "_ci_low"], item[name + "_ci_high"] = map(float, np.percentile(dist, [2.5, 97.5]))
            metrics.append(item)
            macro_key = p, model, cal, part, generator
            boot_groups[macro_key].append(boot["auroc"])
            point_groups[macro_key].append(point["auroc"])
            if model.endswith("gate"):
                a = np.array([float(r["alpha_global"]) for r in cell])
                weights.append(dict(profile=p, model=model, calibration=cal, seed=seed, partition=part, condition=condition, generator=generator,
                                    mean=float(a.mean()), std=float(a.std()), min=float(a.min()), p05=float(np.quantile(a, .05)), median=float(np.median(a)), p95=float(np.quantile(a, .95)), max=float(a.max())))
    macro = []
    for key in sorted(point_groups):
        p, model, cal, part, generator = key
        expected = 12 if model.endswith("gate") else 4
        require(len(point_groups[key]) == expected, "Incomplete condition/initialization coverage")
        low, high = np.percentile(np.mean(boot_groups[key], axis=0), [2.5, 97.5])
        macro.append(dict(profile=p, model=model, calibration=cal, partition=part, generator=generator,
                          macro_auroc=float(np.mean(point_groups[key])), ci_low=float(low), ci_high=float(high),
                          initialization_count=3 if model.endswith("gate") else 1, conditions=4))
    return metrics, macro, weights


def new_checkpoint(path, obj):
    import torch
    path = Path(path)
    require(not path.exists(), f"Refusing checkpoint overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("xb") as f:
        torch.save(obj, f)
    if os.name == "nt":
        temp.rename(path)  # Windows refuses an existing destination.
    else:
        os.link(temp, path); temp.unlink()


class Timer:
    def __enter__(self):
        import torch
        self.start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *args):
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self.start
        self.peak_gpu_bytes = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0

    def result(self):
        return {"seconds": self.seconds, "peak_gpu_bytes": self.peak_gpu_bytes}
