"""Read-only Step B extension: independent development selection and leakage checks."""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
import subprocess
from collections import Counter
import round7_verify_data as verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Also rerun independent full-source Round 7 verification")
    args = parser.parse_args()
    if args.full:
        subprocess.run([sys.executable, str(verify.ROOT / "scripts/round7_verify_data.py")], check=True)
    require = verify.require
    out, root = verify.OUT, verify.ROOT
    p = verify.read(out / "data_protocol_step_b.json")
    require(verify.PIL.__version__ == "12.2.0", "Pillow 12.2 required")
    require(verify.sha(out / "data_protocol.json") == p["base_protocol_sha256"], "Base protocol changed")
    for key in ("files_sha256", "code_sha256"):
        for path, h in p[key].items():
            require(verify.sha(root / path) == h, f"Provenance changed: {path}")
    base = verify.read(out / "data_protocol.json")
    for path, h in base["local_audit_artifacts_sha256"].items():
        require(verify.sha(root / path) == h, "Hash audit cache changed")
    records = verify.lines(root / "artifacts/round7/source_hashes_pillow12_2.jsonl")
    lookup = {r["source_id"]: r for r in records}
    dev = verify.lines(out / "development_manifest.jsonl")
    fit = verify.lines(out / "fitting_pool_manifest.jsonl")
    test = verify.lines(out / "reserved_test_manifest.jsonl")
    old = verify.lines(root / "results/round2/manifest.jsonl")
    eligible = {d["source_id"] for d in verify.lines(root / "artifacts/round7/exclusion_decisions.jsonl") if d["reason"] == "eligible"}
    require(Counter(r["generator"] for r in dev) == {"Real": 300, "ADM": 300, "BigGAN": 300}, "Wrong development counts")
    require(len({r["source_id"] for r in dev}) == 900, "Duplicate development IDs")
    excluded = {r["source_id"] for r in fit + test}
    expected = []
    for g in ["Real", "ADM", "BigGAN"]:
        candidates = [r for r in records if r["source_id"] in eligible and r["source_id"] not in excluded and r["generator"] == g]
        key = lambda r: (hashlib.sha256(f"20261104|development/{g}|{r['source_id']}|{r['pixel_sha256']}".encode()).hexdigest(), r["source_id"])
        expected += sorted(candidates, key=key)[:300]
    require({r["source_id"] for r in dev} == {r["source_id"] for r in expected}, "Development seed reproduction failed")
    prior_pixels = {r["pixel_sha256"] for r in old + fit + test}
    prior_h = [int(r["dhash"], 16) for r in old + fit + test]
    for r in dev:
        require(r == {k: lookup[r["source_id"]][k] for k in verify.FIELDS}, "Source identity mismatch")
        h = int(r["dhash"], 16)
        require(r["pixel_sha256"] not in prior_pixels and all((h ^ v).bit_count() > 3 for v in prior_h), "Development leakage")
        prior_pixels.add(r["pixel_sha256"])
        prior_h.append(h)
    isolation = verify.read(out / "step_b_isolation.json")
    require(set(isolation["reserved_test_source_ids"]) == {r["source_id"] for r in test}, "Denylist incomplete")
    require(set(isolation["allowed_source_ids"]) == {r["source_id"] for r in fit + dev}, "Allowlist incomplete")
    require(isolation["reserved_test_sha256"] == verify.sha(out / "reserved_test_manifest.jsonl"), "Test identity changed")
    print(json.dumps({"status": "passed", "development_counts": dict(Counter(r["generator"] for r in dev)), "disjointness": "pixel/dHash/source ID vs old/fitting/reserved and within development", "existing_files": "unchanged"}, indent=2))


if __name__ == "__main__":
    main()
