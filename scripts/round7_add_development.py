"""Step B: additive development manifest and protocol; no existing file writes."""
import sys
sys.dont_write_bytecode = True
import json
from collections import Counter
import round7_prepare_data as data

OUT = data.OUT
NAMES = ["development_manifest.jsonl", "data_protocol_step_b.json", "step_b_isolation.json"]
SEED = 20261104


def main():
    paths = [OUT / name for name in NAMES]
    doc = data.ROOT / "docs/ROUND7_DEVELOPMENT.md"
    for p in paths + [doc]:
        if p.exists():
            raise FileExistsError(p)
    base = data.read_json(OUT / "data_protocol.json")
    for p, digest in base["local_audit_artifacts_sha256"].items():
        if data.sha(data.ROOT / p) != digest:
            raise ValueError(f"Audit artifact changed: {p}")
    records = data.read_lines(data.ART / "source_hashes_pillow12_2.jsonl")
    eligible = {r["source_id"] for r in data.read_lines(data.ART / "exclusion_decisions.jsonl") if r["reason"] == "eligible"}
    fitting = data.read_lines(OUT / "fitting_pool_manifest.jsonl")
    test = data.read_lines(OUT / "reserved_test_manifest.jsonl")
    blocked = {r["source_id"] for r in fitting + test}
    available = eligible - blocked
    selected = []
    for g in ["Real", "ADM", "BigGAN"]:
        candidates = [r for r in records if r["source_id"] in available and r["generator"] == g]
        if len(candidates) < 300:
            raise ValueError(f"Need 300 eligible unused {g}; found {len(candidates)}")
        selected += data.rank(candidates, SEED, "development/" + g)[:300]
    selected.sort(key=lambda r: r["source_row"])
    data.write_new(paths[0], [{k: r[k] for k in data.CONTENT_KEYS} for r in selected], lines=True)
    isolation = {"reserved_test_source_ids": sorted(r["source_id"] for r in test),
                 "allowed_source_ids": sorted(r["source_id"] for r in fitting + selected),
                 "purpose": "ID-only denylist for Step C. No reserved-test manifest or test image loading in baseline runners.",
                 "reserved_test_sha256": data.sha(OUT / "reserved_test_manifest.jsonl")}
    data.write_new(paths[2], isolation)
    protocol = {"extends": "results/round7/data_protocol.json", "base_protocol_sha256": data.sha(OUT / "data_protocol.json"),
                "seed": SEED, "sampling": "Existing eligible pool, excluding fitting and reserved test; SHA256 rank(seed|development/{generator}|source_id|pixel_sha256); first 300 per generator.",
                "counts": dict(Counter(r["generator"] for r in selected)),
                "development_use": "ADM and BigGAN are heavily inspected domains and are used for development only, never expert/gate fitting, calibration, epoch/threshold/fusion selection.",
                "reserved_test_interpretation": "Future reserved-test ADM/BigGAN results are reported as previously inspected domains, not blind domains.",
                "final_only_domains": ["GLIDE", "VQDM", "Midjourney", "Wukong"],
                "final_only_scope": "No Round 7 model inputs, features or scores for these domains before final evaluation. Earlier independent pilot exposure remains disclosed in the base protocol; data-audit hashing is not model use.",
                "pillow": "12.2.0", "files_sha256": {str(p.relative_to(data.ROOT)).replace('\\', '/'): data.sha(p) for p in [paths[0], paths[2], OUT / "fitting_pool_manifest.jsonl", OUT / "assignments_seed1.json"]},
                "code_sha256": {f"scripts/{n}": data.sha(data.ROOT / "scripts" / n) for n in ["round7_add_development.py", "round7_verify_development.py"]}}
    data.write_new(paths[1], protocol)
    doc.write_text("# Round 7 Step B: development addendum\n\nThis additive protocol leaves all prior files unchanged. `data_protocol_step_b.json` extends the frozen base protocol.\n\nDevelopment has 300 Real + 300 ADM + 300 BigGAN originals, sampled with seed 20261104 from unused eligible source rows. It is disjoint from the fitting pool, reserved test, and historical 1,600-image pool by pixel SHA256 and dHash distance <=3. The original global deduplication remains in force.\n\nADM/BigGAN are already heavily inspected. Development is diagnostic only: it cannot fit heads or gates, calibrate temperatures, select epochs/fusion weights, or fit thresholds. Their future reserved-test results remain previously inspected-domain results. GLIDE/VQDM/Midjourney/Wukong have no model use in Round 7 before final evaluation; historical independent-pilot exposure remains disclosed, and audit hashing is allowed.\n\n`step_b_isolation.json` contains an ID-only reserved-test denylist and the exact fitting-plus-development allowlist. Step C reads this file, never the reserved-test manifest or test images.\n\nRun `python scripts/round7_verify_development.py --full` with pinned Pillow 12.2. Preparation is exclusive-create; reruns refuse existing outputs. The verifier is read-only and extends the existing data verifier.\n", encoding="utf-8")
    print(json.dumps(protocol["counts"], indent=2))


if __name__ == "__main__":
    main()
