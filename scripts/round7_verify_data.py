"""Independent, read-only Round 7 verifier. No model imports or model scores.

Reconstructs hashes from every pinned original and uses brute-force Hamming
checks rather than the preparation script's four-block index. Does not import
the preparation script or trust its eligibility/partition calculations.
"""
import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import io
import json
import time
from collections import Counter
from pathlib import Path

PINNED_PILLOW = Path(__file__).resolve().parents[1] / "artifacts/round7/pillow_12_2"
if PINNED_PILLOW.is_dir():
    sys.path.insert(0, str(PINNED_PILLOW))

import PIL
import pyarrow.parquet as pq
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/round7"
PARTS = ["expert_fit", "gate_fit", "calibration", "selection"]
TARGETS = [750, 250, 125, 125]
SEEDS = [20261101, 20261102, 20261103]
TEST = {"SD15": 250, "ADM": 500, "BigGAN": 500, "GLIDE": 500,
        "Midjourney": 500, "VQDM": 500, "Wukong": 500, "Real": 500}
FIELDS = {"source_id", "generator", "label", "pixel_sha256", "dhash", "source_bytes_sha256"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def reconstruct(inventory, cached):
    """Independent PIL/list implementation of the original hash definitions."""
    records = []
    names = inventory["features"]["generator"]["names"]
    require(len(cached) == 28000, "Hash inventory must cover all original rows")
    for shard_index, info in enumerate(inventory["shards"]):
        path = ROOT / info["local_path"]
        require(path.stat().st_size == info["bytes"], f"Shard size mismatch: {path}")
        require(sha(path) == info["lfs_sha256"] == info["verified_sha256"], f"Shard SHA256 mismatch: {path}")
        parquet = pq.ParquetFile(path)
        require(parquet.metadata.num_rows == 2000, "Unexpected row count")
        require(json.loads(parquet.schema_arrow.metadata[b"huggingface"])["info"]["features"] == inventory["features"], "Schema mismatch")
        offset = 0
        for batch in parquet.iter_batches(batch_size=32, columns=["image", "label", "generator"]):
            for row in batch.to_pylist():
                blob = row["image"]["bytes"]
                with Image.open(io.BytesIO(blob)) as original:
                    native_width, native_height = original.size
                    fmt = original.format
                    rgb = ImageOps.exif_transpose(original).convert("RGB")
                digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                thumb = rgb.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
                gray = list(thumb.getdata())
                value = 0
                for y in range(8):
                    for x in range(8):
                        if gray[y * 9 + x + 1] > gray[y * 9 + x]:
                            value |= 1 << (8 * y + x)
                number = shard_index * 2000 + offset
                name = names[row["generator"]]
                require(row["label"] == int(name != "Real"), "Generator/label inconsistency")
                actual = {"source_id": f"train:{number}", "source_row": number,
                          "generator": name, "generator_id": row["generator"], "label": row["label"],
                          "pixel_sha256": digest, "dhash": f"{value:016x}",
                          "source_bytes_sha256": hashlib.sha256(blob).hexdigest(), "format": fmt,
                          "native_width": native_width, "native_height": native_height,
                          "width": rgb.width, "height": rgb.height}
                require(actual == cached[number], f"Original-image hash/metadata mismatch: train:{number}")
                records.append(actual)
                offset += 1
        print(f"Independently rehashed {shard_index + 1}/14 shards", flush=True)
    return records


def independently_filter(records, previous, saved_decisions):
    old_pixels = {r["pixel_sha256"] for r in previous}
    old_h = [int(r["dhash"], 16) for r in previous]
    retained, retained_h, retained_pixels = [], [], set()
    reasons = Counter()
    stages = {name: Counter() for name in ["Real", "ADM", "BigGAN", "GLIDE", "Midjourney", "SD14", "SD15", "VQDM", "Wukong"]}
    lookup = {r["source_id"]: r for r in records}
    old_lookup = {r["source_id"]: r for r in previous}
    pixel_overlap = dhash_overlap = 0
    require(len(saved_decisions) == len(records), "Incomplete exclusion decisions")
    for row, decision in zip(records, saved_decisions):
        value = int(row["dhash"], 16)
        pixel_hit = row["pixel_sha256"] in old_pixels
        near_hit = any((value ^ h).bit_count() <= 3 for h in old_h)
        pixel_overlap += pixel_hit
        dhash_overlap += near_hit
        if pixel_hit:
            reason = "old_pixel"
        elif near_hit:
            reason = "old_dhash_only"
        elif row["pixel_sha256"] in retained_pixels:
            reason = "round7_pixel"
        elif any((value ^ h).bit_count() <= 3 for h in retained_h):
            reason = "round7_dhash_only"
        else:
            reason = "eligible"
        require((decision["source_id"], decision["generator"], decision["reason"], decision["old_pixel_match"], decision["old_dhash_match"]) ==
                (row["source_id"], row["generator"], reason, pixel_hit, near_hit), f"Exclusion decision mismatch: {row['source_id']}")
        witness = decision["matched_source_id"]
        if reason == "eligible":
            require(witness is None, "Eligible row unexpectedly has exclusion witness")
            retained.append(row)
            retained_h.append(value)
            retained_pixels.add(row["pixel_sha256"])
        else:
            other = old_lookup.get(witness) if reason.startswith("old_") else lookup.get(witness)
            require(other is not None, "Missing exclusion witness")
            if reason.startswith("round7_"):
                require(other["pixel_sha256"] in retained_pixels, "Dedup witness was not retained")
            require(other["pixel_sha256"] == row["pixel_sha256"] if reason.endswith("pixel") else
                    (int(other["dhash"], 16) ^ value).bit_count() <= 3, "Invalid exclusion witness")
        reasons[reason] += 1
        stages[row["generator"]][reason] += 1
    return retained, dict(reasons), {k: dict(v) for k, v in stages.items()}, pixel_overlap, dhash_overlap


def order(rows, seed, purpose):
    keys = []
    for row in rows:
        string = "|".join([str(seed), purpose, row["source_id"], row["pixel_sha256"]])
        keys.append((hashlib.sha256(string.encode("ascii")).hexdigest(), row["source_id"], row))
    return [r for _, _, r in sorted(keys)]


def verify_splits(eligible, pool, test, assignments, protocol):
    source = {r["source_id"]: r for r in eligible}
    for rows in (pool, test):
        require(len({r["source_id"] for r in rows}) == len(rows), "Duplicate manifest source ID")
        for row in rows:
            require(set(row) == FIELDS, "Manifests must contain identifiers, labels and hashes only")
            require(row["source_id"] in source, "Manifest contains excluded/non-source image")
            require(row == {k: source[row["source_id"]][k] for k in FIELDS}, "Manifest hash/label mismatch")
    require(not {r["source_id"] for r in pool} & {r["source_id"] for r in test}, "Test/fitting leakage")
    expected_test, remainder = [], {}
    for name, target in TEST.items():
        ranked = order([r for r in eligible if r["generator"] == name], 20261031, "test/" + name)
        n = min(len(ranked), target)
        expected_test.extend(ranked[:n])
        remainder[name] = ranked[n:]
    require([r["source_id"] for r in test] == [r["source_id"] for r in sorted(expected_test, key=lambda r: r["source_row"])], "Reserved test sampling not reproducible")
    budget = min(1250, len(remainder["SD15"]), len(remainder["Real"]))
    quotas = [divmod(budget * n, 1250) for n in TARGETS]
    actual = [q for q, _ in quotas]
    for i in sorted(range(4), key=lambda i: (-quotas[i][1], i))[:budget - sum(actual)]:
        actual[i] += 1
    counts = dict(zip(PARTS, actual))
    require(min(counts.values()) >= 1, "Empty fitting class")
    expected_pool = sum([order(remainder[name], 20261031, "fit/" + name)[:budget] for name in ("Real", "SD15")], [])
    require([r["source_id"] for r in pool] == [r["source_id"] for r in sorted(expected_pool, key=lambda r: r["source_row"])], "Fixed fitting pool not reproducible")
    pool_ids = {r["source_id"] for r in pool}
    for seed, assignment in zip(SEEDS, assignments):
        require(assignment["seed"] == seed, "Wrong seed")
        require(assignment["fitting_pool_sha256"] == sha(OUT / "fitting_pool_manifest.jsonl"), "Fitting pool hash mismatch")
        require(assignment["reserved_test_sha256"] == sha(OUT / "reserved_test_manifest.jsonl"), "Test manifest differs between seeds")
        expected = {name: [] for name in PARTS}
        for generator in ("Real", "SD15"):
            ordered = order([r for r in pool if r["generator"] == generator], seed, "partition/" + generator)
            offset = 0
            for name, n in counts.items():
                expected[name] += [r["source_id"] for r in ordered[offset:offset+n]]
                offset += n
        expected = {k: sorted(v) for k, v in expected.items()}
        require(assignment["partitions"] == expected and assignment["counts_per_class"] == counts, "Seed assignments not reproducible")
        flattened = sum(assignment["partitions"].values(), [])
        require(len(flattened) == len(set(flattened)) and set(flattened) == pool_ids, "Fitting partitions overlap or do not cover the same fixed pool")
        for name, ids in assignment["partitions"].items():
            require(Counter(source[i]["generator"] for i in ids) == {"Real": counts[name], "SD15": counts[name]}, "Partition class imbalance")
    test_counts = dict(Counter(r["generator"] for r in test))
    require(test_counts == protocol["reserved_test_counts"], "Protocol test counts mismatch")
    require(counts == protocol["counts_per_class_each_seed"], "Protocol fitting counts mismatch")
    require(protocol["shortfalls"] == {"test": {g: TEST[g] - test_counts.get(g, 0) for g in TEST},
            "fitting_per_class": {k: n - counts[k] for k, n in zip(PARTS, TARGETS)}}, "Shortfalls mismatch")
    return counts, test_counts


def distributions(records, key):
    result = {}
    for value in sorted(set(str(r[key]) for r in records)):
        selected = [r for r in records if str(r[key]) == value]
        result[value] = {"count": len(selected), "format": dict(Counter(r["format"] for r in selected)),
                         "native_encoded_size": dict(Counter(f"{r['native_width']}x{r['native_height']}" for r in selected))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Optional new JSON report under results/round7; never overwrite")
    args = parser.parse_args()
    if args.report:
        require(args.report.resolve().is_relative_to(OUT.resolve()), "Report must stay under results/round7")
        if args.report.exists():
            raise FileExistsError(f"Refusing to overwrite {args.report}")
    require(PIL.__version__ == "12.2.0", "Use repository-pinned Pillow 12.2.0 for historical JPEG hash compatibility")
    start = time.perf_counter()
    protocol = read(OUT / "data_protocol.json")
    audit = read(OUT / "data_audit.json")
    inventory = read(OUT / "source_inventory.json")
    require(protocol["dataset"] == inventory["dataset"] == "TheKernel01/Tiny-GenImage", "Wrong dataset")
    require(protocol["revision"] == inventory["revision"] == "89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69", "Wrong revision")
    require(protocol["source_split"] == "train" and protocol["training_generator"] == "SD15" and protocol["training_generator_id"] == 6, "Wrong fitting domain")
    require(protocol["partition_seeds"] == SEEDS and protocol["pool_and_test_seed"] == 20261031, "Seed mismatch")
    require(protocol["requested_test"] == TEST and protocol["requested_fitting_per_class"] == dict(zip(PARTS, TARGETS)), "Wrong requested budget")
    require(protocol["shards"] == inventory["shards"], "Source shard list differs")
    require([r["path"] for r in inventory["shards"]] == [f"data/train-{i:05d}-of-00014.parquet" for i in range(14)], "Not all 14 training shards")
    require(protocol["source_inventory_sha256"] == sha(OUT / "source_inventory.json"), "Inventory hash mismatch")
    for group in ("files_sha256", "local_audit_artifacts_sha256", "implementation_sha256"):
        for path, expected in protocol[group].items():
            require(sha(ROOT / path) == expected, f"Provenance mismatch: {path}")
    for path, digest in protocol["inherited_round6"]["file_sha256"].items():
        require(sha(ROOT / path) == digest, f"Inherited Round 6 settings changed: {path}")
    require(protocol["inherited_round6"]["protocol_snapshot"] == read(ROOT / "results/round6/protocol.json"), "Inherited protocol mismatch")
    protected = read(ROOT / "artifacts/round7/protected_files_before.json")
    for path, digest in protected.items():
        require(sha(ROOT / path) == digest, f"Pre-existing file changed: {path}")
    previous = lines(ROOT / "results/round2/manifest.jsonl")
    require(len(previous) == 1600 and sha(ROOT / "results/round2/manifest.jsonl") == protocol["exclusions"]["sha256"], "Wrong old exclusion pool")
    cached = lines(ROOT / "artifacts/round7/source_hashes_pillow12_2.jsonl")
    records = reconstruct(inventory, cached)
    for r in previous:
        actual = records[r["source_row"]]
        require(all(actual[k] == r[k] for k in ("source_id", "pixel_sha256", "dhash", "generator", "label")), "Original hashes differ from extract_record")
    print("Checking all exclusions and deduplication independently", flush=True)
    eligible, reasons, stages, pixels, near = independently_filter(records, previous, lines(ROOT / "artifacts/round7/exclusion_decisions.jsonl"))
    require(reasons == audit["filter_counts"] == protocol["exclusions"]["counts"], "Filter accounting mismatch")
    require(stages == audit["filter_counts_by_generator"], "Per-generator exclusions mismatch")
    require(audit["old_pool_overlap"] == {"reference_images": 1600, "pixel_match_rows": pixels,
            "dhash_match_rows_including_exact": near, "excluded_union_rows": reasons.get("old_pixel", 0) + reasons.get("old_dhash_only", 0)}, "Prior exposure overlap counts mismatch")
    pool, test = lines(OUT / "fitting_pool_manifest.jsonl"), lines(OUT / "reserved_test_manifest.jsonl")
    assignments = [read(OUT / f"assignments_seed{i}.json") for i in (1, 2, 3)]
    counts, test_counts = verify_splits(eligible, pool, test, assignments, protocol)
    require(audit["total_source"] == len(records) == 28000, "Wrong source count")
    require(audit["classlabel_mapping"] == inventory["generator_classlabel"], "Incomplete class mapping")
    source_counts = Counter(r["generator"] for r in records)
    require({g: source_counts[g] for g in inventory["counts_per_generator"]} == inventory["counts_per_generator"], "Source generator coverage mismatch")
    require(source_counts["SD14"] == 0, "SD14 absence assumption changed")
    lookup = {r["source_id"]: r for r in records}
    selected = [lookup[r["source_id"]] for r in pool + test]
    for key, group, field in [("source_by_generator", records, "generator"), ("source_by_label", records, "label"),
                              ("eligible_by_generator", eligible, "generator"), ("selected_by_generator", selected, "generator"),
                              ("selected_by_label", selected, "label")]:
        require(audit[key] == distributions(group, field), f"Format/size distribution mismatch: {key}")
    require(audit["original_byte_duplicate_extra_rows"] == len(records) - len({r["source_bytes_sha256"] for r in records}), "Byte duplicate count mismatch")
    require(audit["original_pixel_duplicate_extra_rows"] == len(records) - len({r["pixel_sha256"] for r in records}), "Pixel duplicate count mismatch")
    require(audit["counts_per_class_each_seed"] == counts and audit["reserved_test_counts"] == test_counts, "Audit selected counts mismatch")
    require(audit["fitting_pool_total"] == len(pool) and audit["reserved_test_total_unique"] == len(test), "Unique selected counts mismatch")
    require(audit["eligible_not_selected"] == len(eligible) - len(pool) - len(test), "Unused count mismatch")
    require(audit["shortfalls"] == protocol["shortfalls"], "Audit shortfalls mismatch")
    for path, digest in protected.items():
        require(sha(ROOT / path) == digest, f"Pre-existing file changed during verification: {path}")
    report = {"status": "passed", "scope": "Independent shard and original-image hashing, exclusion/deduplication, distributions, fixed test and fitting pool, all three seeded assignments, historical byte preservation; no model operations",
              "source_images_rehashed": len(records), "shards_verified": 14,
              "previous_pool_images_checked": len(previous), "eligible_images": len(eligible),
              "filter_counts": reasons, "fitting_per_class_each_seed": counts,
              "reserved_test_counts": test_counts, "unique_test_images": len(test),
              "partition_seeds_verified": SEEDS, "protected_files_unchanged": len(protected),
              "protocol_sha256": sha(OUT / "data_protocol.json"), "seconds": round(time.perf_counter() - start, 2)}
    if args.report:
        with args.report.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
