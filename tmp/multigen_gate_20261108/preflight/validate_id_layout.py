"""Read-only ID/cache/hash preflight. No models, scoring, images, or final-test files."""
import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_name("id_layout_preflight.json")
assert not OUT.exists()
HASHES = {}


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(relative):
    assert "reserved" not in relative.lower() and "final" not in relative.lower(), relative
    path = ROOT / relative
    HASHES[relative] = sha(path)
    return json.loads(path.read_text(encoding="utf-8"))


def lines(relative):
    assert "reserved" not in relative.lower() and "final" not in relative.lower(), relative
    path = ROOT / relative
    HASHES[relative] = sha(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


fit_rows = lines("results/round7/fitting_pool_manifest.jsonl")
dev_rows = lines("results/round7/development_manifest.jsonl")
base = read("results/round7/data_protocol.json")
step_b = read("results/round7/data_protocol_step_b.json")
isolation = read("results/round7/step_b_isolation.json")  # ID-only denylist, not test manifest.
helper = read("tmp/gate_diagnosis/original_gate_views/protocol.json")
helper_code = ROOT / "tmp/gate_diagnosis/original_gate_views/run_original_gate_views.py"
HASHES[helper_code.relative_to(ROOT).as_posix()] = sha(helper_code)
assert sha(helper_code) == helper["script_sha256"]
assert HASHES["results/round7/data_protocol.json"] == step_b["base_protocol_sha256"]
for p in ("results/round7/development_manifest.jsonl", "results/round7/fitting_pool_manifest.jsonl", "results/round7/step_b_isolation.json"):
    assert HASHES[p] == step_b["files_sha256"][p]
records = {r["source_id"]: r for r in fit_rows + dev_rows}
assert len(records) == len(fit_rows) + len(dev_rows) == 3400
fit_ids, dev_ids = {r["source_id"] for r in fit_rows}, {r["source_id"] for r in dev_rows}
assert not fit_ids & dev_ids
assert set(records) == set(isolation["allowed_source_ids"])
assert not set(records) & set(isolation["reserved_test_source_ids"])
assert Counter(r["generator"] for r in fit_rows) == Counter(Real=1250, SD15=1250)
assert Counter(r["generator"] for r in dev_rows) == Counter(Real=300, ADM=300, BigGAN=300)
for r in records.values():
    assert r["label"] == (0 if r["generator"] == "Real" else 1)

duplicate_groups = {}
for field in ("source_bytes_sha256", "pixel_sha256"):
    groups = defaultdict(list)
    for sid, r in records.items():
        groups[r[field]].append(sid)
    duplicate_groups[field] = [ids for ids in groups.values() if len(ids) > 1]
    assert not duplicate_groups[field]

items = [(sid, int(r["dhash"], 16), sid in dev_ids) for sid, r in records.items()]
close_pairs, min_any, min_fit_dev = [], 65, 65
for i, (sid, dh, is_dev) in enumerate(items):
    for sid2, dh2, is_dev2 in items[i + 1:]:
        distance = (dh ^ dh2).bit_count()
        min_any = min(min_any, distance)
        if is_dev != is_dev2:
            min_fit_dev = min(min_fit_dev, distance)
        if distance <= 3:
            close_pairs.append((sid, sid2, distance))
assert not close_pairs

per_seed, source_sets, dev_sets_by_seed, cache_reports = [], {}, [], []
for number in (1, 2, 3):
    assignment = read(f"results/round7/assignments_seed{number}.json")
    assert HASHES[f"results/round7/assignments_seed{number}.json"] == base["files_sha256"][f"results/round7/assignments_seed{number}.json"]
    marker = read(f"results/round8_a/seed{number}/matched/complete.json")
    rel = f"artifacts/round8_a/seed{number}/matched/bundle.json"
    bundle = read(rel)
    assert HASHES[rel] == marker["files_sha256"][rel]
    assert bundle["run_seed"] == assignment["seed"] == 20261100 + number
    assert bundle["signature"] == marker["signature"]
    expected = {k: set(v) for k, v in assignment["partitions"].items()}
    assert set(expected) == {"expert_fit", "gate_fit", "calibration", "selection"}
    assert set.union(*expected.values()) == fit_ids
    assert sum(map(len, expected.values())) == len(fit_ids)
    raw_ids = {}
    batch_report = {}
    for part, raw in bundle["raw_batches"].items():
        by_cond = defaultdict(list)
        for r, y in zip(raw["rows"], raw["y"]):
            assert y == r["label"] == records[r["source_id"]]["label"]
            assert r["generator"] == records[r["source_id"]]["generator"]
            by_cond[r["condition"]].append(r["source_id"])
        ids = set(by_cond["clean"])
        assert all(len(v) == len(ids) and set(v) == ids for v in by_cond.values())
        reference = dev_ids if part == "development" else expected["selection" if part in ("selection", "tuning") else part]
        assert ids == reference
        raw_ids[part] = ids
        batch_report[part] = dict(unique_originals=len(ids), views=len(raw["y"]),
                                 sources=dict(Counter(records[i]["generator"] for i in ids)), conditions=sorted(by_cond))
    assert raw_ids["selection"] == raw_ids["tuning"]
    for a, b in combinations(("gate_fit", "tuning", "development"), 2):
        assert not raw_ids[a] & raw_ids[b]
    for a in ("gate_fit", "tuning", "development"):
        assert not raw_ids[a] & expected["expert_fit"]
    source_sets[number] = {p: raw_ids[p] for p in ("gate_fit", "tuning")}
    dev_sets_by_seed.append(raw_ids["development"])
    per_seed.append(dict(seed=assignment["seed"], batches=batch_report,
                         expert_fit_disjoint=True, gate_fit_tuning_development_pairwise_disjoint=True,
                         reserved_denylist_overlap=0))
    for part, raw_name, conditions in (("gate_fit", "gate_fit", ("clean", "jpeg70", "resize05")),
                                       ("selection", "tuning", ("clean", "jpeg70", "resize05")),
                                       ("development", "development", ("clean", "jpeg70", "jpeg50", "resize05"))):
        target_ids = sorted(raw_ids[raw_name])
        for condition in conditions:
            folder = ROOT / f"artifacts/round8_a/seed{number}/features/{part}_{condition}_0"
            paths = sorted(folder.glob("*.npz"))
            assert paths
            actual_ids = []
            for path in paths:
                relative = path.relative_to(ROOT).as_posix()
                expected_hash = (bundle["feature_sha256"] if part == "development" else helper["input_sha256"])[relative]
                actual_hash = sha(path)
                assert actual_hash == expected_hash
                HASHES[relative] = actual_hash
                with np.load(path, allow_pickle=False) as cache:
                    ids = cache["ids"].tolist()
                    assert json.loads(str(cache["identity"])) == [assignment["seed"], part, condition, False, ids]
                    assert str(cache["signature"]) == bundle["signature"]
                    assert cache["g"].shape == (len(ids), 512) and cache["l"].shape == (len(ids), 4, 512) and cache["q"].shape == (len(ids), 3)
                    for name in ("g", "l", "q"):
                        assert hashlib.sha256(cache[name].tobytes()).hexdigest() == str(cache[name + "_sha256"])
                    actual_ids += ids
            assert actual_ids == target_ids
            cache_reports.append(dict(seed=assignment["seed"], partition=part, condition=condition,
                                      originals=len(actual_ids), feature_batches=len(paths), original_view=True, verified=True))

assert all(ids == dev_sets_by_seed[0] for ids in dev_sets_by_seed)
cross_seed = []
for a, b in combinations((1, 2, 3), 2):
    for pa, pb in (("gate_fit", "gate_fit"), ("tuning", "tuning"), ("gate_fit", "tuning"), ("tuning", "gate_fit")):
        common = source_sets[a][pa] & source_sets[b][pb]
        cross_seed.append(dict(seed_a=20261100+a, partition_a=pa, seed_b=20261100+b, partition_b=pb,
                               overlap=len(common), by_source=dict(Counter(records[i]["generator"] for i in common))))

result = dict(status="passed", scope="Only fitting/development IDs, raw matched bundle layouts, original feature caches, and metadata; no models or final-test manifest/predictions/images read.",
    input_sha256=HASHES, originals=3400, fitting_originals=2500, development_originals=900,
    manifests_have_hash_fields=["source_id", "source_bytes_sha256", "pixel_sha256", "dhash"],
    duplicate_groups=duplicate_groups, dhash_pairs_checked=len(items)*(len(items)-1)//2,
    dhash_distance_le3_pairs=close_pairs, min_dhash_distance_all=min_any, min_dhash_distance_fitting_vs_development=min_fit_dev,
    development_donor_ids_identical_across_seeds=True,
    per_seed=per_seed, original_feature_cache_checks=cache_reports, cross_seed_original_overlap=cross_seed,
    intended_counts=dict(control_gate_fit={"Real":250,"SD15":250}, mixed_gate_fit={"Real":250,"SD15":125,"donor":125},
                         control_tuning={"Real":124,"SD15":124}, mixed_tuning={"Real":124,"SD15":62,"donor":62},
                         per_donor={"gate_fit":125,"tuning":62,"unused":113}),
    recommendations=[
        "Split each donor's 300 original source IDs once by a frozen deterministic hash into 125 fit, 62 tune, 113 unused; share these memberships across all run seeds and conditions.",
        "Use original source IDs as unit. Never split condition rows or cache batches independently. JPEG50 remains reporting-only.",
        "For each seed, keep new gate-fit IDs disjoint from expert-fit and tuning IDs, and keep the untouched target generator plus 300 development Real IDs outside every fit/tuning subset.",
        "Derive mixed SD15 fit/tune subsets within the corresponding source control IDs (125 of 250; 62 of 124). Keep the same 250 fit Real and 124 tune Real for control and mixed.",
        "If reporting donor performance, use only its 113 unused images with held-out Real; the original 300-image donor report is now contaminated by 187 fitted/tuned IDs.",
        "Do not pool cross-seed source train/tune rows as independent splits; the original fitting pool and development donors are reused.",
        "Freeze a new exploratory protocol for repurposing development donor images. Preserve all old Round 7/8 protocols and results; target domains were inspected historically, so this is not blind generalization evidence.",
        "Reuse metadata and cache hashing in helper verify_and_collect without importing or invoking its main training routine. Reconstruct original-view logits with the matched experts only after the new split freezes."
    ])
with OUT.open("x", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
print(json.dumps({"output":str(OUT),"status":"passed","originals":len(records),"dhash_pairs_checked":result["dhash_pairs_checked"],
                  "min_distance":min_any,"min_fit_dev_distance":min_fit_dev,"cache_batches":sum(r["feature_batches"] for r in cache_reports)},ensure_ascii=False))
