"""Round 7 data audit and partitions only. No models, image exports or scores.

All writes use exclusive creation. Source shards and earlier rounds are read-only.
Pixel and dHash definitions match prepare_data.extract_record, without its
unneeded 224px crop. The complete hash inventory can be reused read-only.
"""
import sys
sys.dont_write_bytecode = True

import hashlib
import io
import json
import platform
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

PINNED_PILLOW = Path(__file__).resolve().parents[1] / "artifacts/round7/pillow_12_2"
if PINNED_PILLOW.is_dir():
    sys.path.insert(0, str(PINNED_PILLOW))

import numpy as np
import PIL
import pyarrow
import pyarrow.parquet as pq
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/round7"
ART = ROOT / "artifacts/round7"
DATASET = "TheKernel01/Tiny-GenImage"
REVISION = "89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69"
SEEDS = [20261101, 20261102, 20261103]
POOL_SEED = 20261031
PARTS = ["expert_fit", "gate_fit", "calibration", "selection"]
TARGETS = [750, 250, 125, 125]
TEST_TARGETS = {"SD15": 250, "ADM": 500, "BigGAN": 500, "GLIDE": 500,
                "Midjourney": 500, "VQDM": 500, "Wukong": 500, "Real": 500}
CONTENT_KEYS = ["source_id", "generator", "label", "pixel_sha256", "dhash", "source_bytes_sha256"]
OUTPUTS = [OUT / "data_protocol.json", OUT / "data_audit.json",
           OUT / "reserved_test_manifest.jsonl", OUT / "fitting_pool_manifest.jsonl",
           *[OUT / f"assignments_seed{i}.json" for i in (1, 2, 3)], ROOT / "docs/ROUND7_DATA.md"]


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def write_new(path, value, lines=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        if lines:
            for row in value:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        else:
            handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def record_image(row, number, names):
    blob = row["image"]["bytes"]
    with Image.open(io.BytesIO(blob)) as original:
        fmt, native = original.format, original.size
        image = ImageOps.exif_transpose(original).convert("RGB")
    pixels = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    small = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    dhash = sum(int(bit) << i for i, bit in enumerate((small[:, 1:] > small[:, :-1]).ravel()))
    name = names[row["generator"]]
    if row["label"] != int(name != "Real"):
        raise ValueError(f"Unexpected label for train:{number}")
    return {"source_id": f"train:{number}", "source_row": number,
            "generator": name, "generator_id": row["generator"], "label": row["label"],
            "pixel_sha256": pixels, "dhash": f"{dhash:016x}",
            "source_bytes_sha256": hashlib.sha256(blob).hexdigest(),
            "format": fmt, "native_width": native[0], "native_height": native[1],
            "width": image.width, "height": image.height}


def load_source():
    inventory = read_json(OUT / "source_inventory.json")
    if (inventory["dataset"], inventory["revision"]) != (DATASET, REVISION):
        raise ValueError("Wrong pinned dataset")
    expected = [f"data/train-{i:05d}-of-00014.parquet" for i in range(14)]
    if [s["path"] for s in inventory["shards"]] != expected:
        raise ValueError("Incomplete source shard list")
    for info in inventory["shards"]:
        path = ROOT / info["local_path"]
        if path.stat().st_size != info["bytes"] or sha(path) != info["lfs_sha256"]:
            raise ValueError(f"Shard checksum mismatch: {path}")
    if PIL.__version__ != "12.2.0":
        raise RuntimeError("Use repository-pinned Pillow 12.2.0 for historical JPEG pixel-hash compatibility")
    cache, meta = ART / "source_hashes_pillow12_2.jsonl", ART / "source_hashes_pillow12_2_meta.json"
    if cache.exists() or meta.exists():
        provenance = read_json(meta)
        if provenance["inventory_sha256"] != sha(OUT / "source_inventory.json") or provenance["sha256"] != sha(cache):
            raise ValueError("Existing hash inventory failed provenance check")
        print("Reusing verified original-image hashes read-only", flush=True)
        return inventory, read_lines(cache)
    records = []
    names = inventory["features"]["generator"]["names"]
    for shard_number, info in enumerate(inventory["shards"]):
        parquet = pq.ParquetFile(ROOT / info["local_path"])
        features = json.loads(parquet.schema_arrow.metadata[b"huggingface"])["info"]["features"]
        if features != inventory["features"] or parquet.metadata.num_rows != 2000:
            raise ValueError("Source schema or row indexing changed")
        offset = 0
        for batch in parquet.iter_batches(batch_size=32, columns=["image", "label", "generator"]):
            for row in batch.to_pylist():
                records.append(record_image(row, shard_number * 2000 + offset, names))
                offset += 1
        print(f"Hashed {shard_number + 1}/14 shards ({len(records):,} originals)", flush=True)
    write_new(cache, records, lines=True)
    write_new(meta, {"sha256": sha(cache), "inventory_sha256": sha(OUT / "source_inventory.json"),
                     "records": len(records), "python": platform.python_version(),
                     "pillow": PIL.__version__, "numpy": np.__version__, "pyarrow": pyarrow.__version__,
                     "pixel_definition": "sha256(str(EXIF-transposed RGB size).encode() + RGB.tobytes())",
                     "dhash_definition": "EXIF-transposed RGB -> L -> LANCZOS 9x8; right > left; row-major little-endian bits"})
    return inventory, records


class NearIndex:
    """Exact radius-three search: at least one of four 16-bit blocks must match."""
    def __init__(self):
        self.buckets = defaultdict(list)
        self.values = []

    def add(self, value, identity):
        index = len(self.values)
        self.values.append((value, identity))
        for block in range(4):
            self.buckets[block, (value >> (16 * block)) & 65535].append(index)

    def match(self, value):
        candidates = set()
        for block in range(4):
            candidates.update(self.buckets.get((block, (value >> (16 * block)) & 65535), ()))
        for index in sorted(candidates):
            other, identity = self.values[index]
            if (other ^ value).bit_count() <= 3:
                return identity
        return None


def filter_records(records, previous):
    old_pixels = {r["pixel_sha256"]: r["source_id"] for r in previous}
    old_near, new_near = NearIndex(), NearIndex()
    for row in previous:
        old_near.add(int(row["dhash"], 16), row["source_id"])
    seen, accepted, decisions = {}, [], []
    for row in sorted(records, key=lambda r: r["source_row"]):
        pixel, h = row["pixel_sha256"], int(row["dhash"], 16)
        exact_old, near_old = old_pixels.get(pixel), old_near.match(h)
        reason, match = "eligible", None
        if exact_old or near_old:
            reason, match = ("old_pixel", exact_old) if exact_old else ("old_dhash_only", near_old)
        elif pixel in seen:
            reason, match = "round7_pixel", seen[pixel]
        else:
            match = new_near.match(h)
            if match is not None:
                reason = "round7_dhash_only"
        decisions.append({"source_id": row["source_id"], "generator": row["generator"],
                          "reason": reason, "matched_source_id": match,
                          "old_pixel_match": bool(exact_old), "old_dhash_match": bool(near_old)})
        if reason == "eligible":
            accepted.append(row)
            seen[pixel] = row["source_id"]
            new_near.add(h, row["source_id"])
    return accepted, decisions


def rank(records, seed, purpose):
    def key(row):
        message = f"{seed}|{purpose}|{row['source_id']}|{row['pixel_sha256']}"
        return hashlib.sha256(message.encode("ascii")).hexdigest(), row["source_id"]
    return sorted(records, key=key)


def scaled_counts(budget):
    total = sum(TARGETS)
    counts = [budget * target // total for target in TARGETS]
    order = sorted(range(4), key=lambda i: (-(budget * TARGETS[i] % total), i))
    for index in order[:budget - sum(counts)]:
        counts[index] += 1
    return dict(zip(PARTS, counts))


def define_splits(eligible):
    by_generator = {g: [r for r in eligible if r["generator"] == g] for g in TEST_TARGETS}
    reserved, remaining, test_counts = [], {}, {}
    for generator, target in TEST_TARGETS.items():
        rows = rank(by_generator[generator], POOL_SEED, f"test/{generator}")
        n = min(target, len(rows))
        reserved.extend(rows[:n])
        remaining[generator] = rows[n:]
        test_counts[generator] = n
    budget = min(sum(TARGETS), len(remaining["SD15"]), len(remaining["Real"]))
    counts = scaled_counts(budget)
    if min(counts.values()) < 1:
        raise ValueError("Too few unique images for four nonempty balanced fitting partitions")
    pool = []
    for generator in ("Real", "SD15"):
        pool.extend(rank(remaining[generator], POOL_SEED, f"fit/{generator}")[:budget])
    assignments = []
    for seed in SEEDS:
        partitions = {part: [] for part in PARTS}
        for generator in ("Real", "SD15"):
            ordered = rank([r for r in pool if r["generator"] == generator], seed, f"partition/{generator}")
            offset = 0
            for part, n in counts.items():
                partitions[part].extend(r["source_id"] for r in ordered[offset:offset+n])
                offset += n
        assignments.append({"seed": seed, "partitions": {k: sorted(v) for k, v in partitions.items()},
                            "counts_per_class": counts})
    return sorted(pool, key=lambda r: r["source_row"]), sorted(reserved, key=lambda r: r["source_row"]), assignments, counts, test_counts


def distribution(records, group_key):
    result = {}
    for group in sorted({str(r[group_key]) for r in records}):
        rows = [r for r in records if str(r[group_key]) == group]
        result[group] = {"count": len(rows), "format": dict(Counter(r["format"] for r in rows)),
                         "native_encoded_size": dict(Counter(f"{r['native_width']}x{r['native_height']}" for r in rows))}
    return result


def protected_snapshot():
    path = ART / "protected_files_before.json"
    if path.exists():
        previous = read_json(path)
        for name, digest in previous.items():
            if sha(ROOT / name) != digest:
                raise ValueError(f"Protected file changed: {name}")
        return previous
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    paths = [name for name in tracked if name]
    paths += ["scripts/round7_inspect_source.py", "docs/ROUND7_SOURCE_INSPECTION.md", "results/round7/source_inventory.json"]
    snapshot = {name: sha(ROOT / name) for name in sorted(set(paths))}
    write_new(path, snapshot)
    return snapshot


def main():
    for path in OUTPUTS:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    protected_snapshot()
    inventory, records = load_source()
    previous = read_lines(ROOT / "results/round2/manifest.jsonl")
    if len(previous) != 1600:
        raise ValueError("Expected the complete 1,600-image Round 2 pool")
    lookup = {r["source_id"]: r for r in records}
    for row in previous:
        if any(lookup[row["source_id"]][key] != row[key] for key in ("pixel_sha256", "dhash", "generator", "label")):
            raise ValueError(f"Source hash definition differs from Round 2: {row['source_id']}")
    eligible, decisions = filter_records(records, previous)
    pool, reserved, assignments, counts, test_counts = define_splits(eligible)
    decision_path = ART / "exclusion_decisions.jsonl"
    write_new(decision_path, decisions, lines=True)
    pool_path, test_path = OUT / "fitting_pool_manifest.jsonl", OUT / "reserved_test_manifest.jsonl"
    write_new(pool_path, [{k: r[k] for k in CONTENT_KEYS} for r in pool], lines=True)
    write_new(test_path, [{k: r[k] for k in CONTENT_KEYS} for r in reserved], lines=True)
    for i, assignment in enumerate(assignments, 1):
        assignment.update(fitting_pool_sha256=sha(pool_path), reserved_test_sha256=sha(test_path))
        write_new(OUT / f"assignments_seed{i}.json", assignment)
    reasons = dict(Counter(r["reason"] for r in decisions))
    stages = {g: dict(Counter(d["reason"] for d in decisions if d["generator"] == g))
              for g in inventory["features"]["generator"]["names"]}
    shortfalls = {"test": {g: TEST_TARGETS[g] - test_counts[g] for g in TEST_TARGETS},
                  "fitting_per_class": {part: target - counts[part] for part, target in zip(PARTS, TARGETS)}}
    audit = {
        "scope": "Original-image data audit only; no features, training, image exports or scores",
        "total_source": len(records), "classlabel_mapping": inventory["generator_classlabel"],
        "source_by_generator": distribution(records, "generator"), "source_by_label": distribution(records, "label"),
        "eligible_by_generator": distribution(eligible, "generator"),
        "selected_by_generator": distribution(pool + reserved, "generator"),
        "selected_by_label": distribution(pool + reserved, "label"),
        "filter_counts": reasons, "filter_counts_by_generator": stages,
        "old_pool_overlap": {"reference_images": len(previous),
                             "pixel_match_rows": sum(d["old_pixel_match"] for d in decisions),
                             "dhash_match_rows_including_exact": sum(d["old_dhash_match"] for d in decisions),
                             "excluded_union_rows": sum(d["reason"].startswith("old_") for d in decisions)},
        "original_byte_duplicate_extra_rows": len(records) - len({r["source_bytes_sha256"] for r in records}),
        "original_pixel_duplicate_extra_rows": len(records) - len({r["pixel_sha256"] for r in records}),
        "counts_per_class_each_seed": counts, "reserved_test_counts": test_counts,
        "fitting_pool_total": len(pool), "reserved_test_total_unique": len(reserved),
        "eligible_not_selected": len(eligible) - len(pool) - len(reserved), "shortfalls": shortfalls,
        "confounds": ["Real sources are 100% JPEG; generated sources are 100% PNG.",
                      "BigGAN 128; ADM/GLIDE/VQDM 256; SD15/Wukong 512; Midjourney 1999 at 1024 and one at 512; real dimensions vary.",
                      "Harmonisation controls observable format/size differences but does not remove compression history, effective resolution, content or generator confounds."],
        "hash_definitions": read_json(ART / "source_hashes_pillow12_2_meta.json"),
        "decoder_compatibility": "Pillow 12.2.0 matches the historical environment. A preliminary Anaconda Pillow 10.4 inventory differed on 321 historical real pixel hashes (one also differed in dHash) and was not used for final assignments; all 1600 historical hashes must match under the pinned decoder.",
        "duplicate_count_definition": "Exclusive row-removal reasons, not all pairwise edges: old exclusions first, then greedy global deduplication in ascending numeric source_row; retained earliest eligible representative; no crop-hash exclusion.",
    }
    write_new(OUT / "data_audit.json", audit)
    inherited_files = ["scripts/run_round6.py", "scripts/run_round5.py", "scripts/run_round4.py", "scripts/run_round2.py",
                       "scripts/run_pilot.py", "scripts/local_branch.py", "scripts/gate_models.py", "scripts/score_calibration.py",
                       "results/round6/protocol.json"]
    protocol = {
        "id": "round7_sd15_new_image_partitions", "dataset": DATASET, "revision": REVISION,
        "source_split": "train", "source_inventory_sha256": sha(OUT / "source_inventory.json"),
        "shards": inventory["shards"], "training_generator": "SD15", "training_generator_id": 6,
        "protocol_deviation": "SD14 has zero images in Tiny-GenImage revision 89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69. User approved SD15 (ID 6) as the closest substitute; this deviates from the official GenImage SD1.4 training protocol.",
        "partition_seeds": SEEDS, "pool_and_test_seed": POOL_SEED,
        "sampling": "Sort ascending by (SHA256(ASCII f'{seed}|{purpose}|{source_id}|{pixel_sha256}'), source_id); purposes test/{generator}, fit/{generator}, partition/{generator}. Reserve test first; choose one fixed fitting pool; partition each class separately for each seed.",
        "scaling": "Budget per class=min(1250, remaining SD15, remaining Real); proportional 750:250:125:125 allocation with largest remainders, ties in expert_fit/gate_fit/calibration/selection order.",
        "requested_fitting_per_class": dict(zip(PARTS, TARGETS)), "counts_per_class_each_seed": counts,
        "requested_test": TEST_TARGETS, "reserved_test_counts": test_counts, "shortfalls": shortfalls,
        "test_roles": {g: "shared_real" if g == "Real" else "in_domain" if g == "SD15" else "cross_generator" for g in TEST_TARGETS},
        "shared_real_rule": "The same 500 (or available shortfall) Real originals are reused for each generator comparison and counted once in pooled metrics. The identical test manifest is bound to all three seed assignments.",
        "exclusions": {"old_manifest": "results/round2/manifest.jsonl", "sha256": sha(ROOT / "results/round2/manifest.jsonl"),
                       "reference_count": len(previous), "rules": ["pixel_sha256 equality", "64-bit dHash Hamming distance <=3"],
                       "deduplication": audit["duplicate_count_definition"], "counts": reasons},
        "exposure": {"heavily_inspected": ["ADM", "BigGAN", "SD15"],
                     "lightly_exposed": ["GLIDE", "VQDM", "Midjourney", "Wukong"],
                     "heavily_inspected_context": "Rounds 1-6; SD15 is now training-domain and reports in-domain only.",
                     "lightly_exposed_context": "Seen in a separate, independent team-member pilot drawn from the same source; its results informed no Round 6 or Round 7 decision.",
                     "test_images": "Round 7 test images were not used or inspected in rounds 1–6 of this repository. Some (116 of 3,750) also appeared in a separate, independent team-member pilot drawn from the same source; that pilot did not inform any Round 7 setting.",
                     "overlap_disclosure": "The user reports an independent check found overlap of 116/3,750 reserved-test images and 94/2,500 fitting-pool images with a separate, independent team-member pilot drawn from the same Tiny-GenImage revision. That pilot is not part of this repository, its results informed no Round 6 or Round 7 decision, and it is not used or referenced as data. No external pilot files were accessed for this documentation correction.",
                     "interpretation": "Training-unseen is not the same as blind. No cross-generator result should be described as a strictly blind generator benchmark."},
        "limitations": ["The user reports an independent check found overlap of 116/3,750 reserved-test images and 94/2,500 fitting-pool images with a separate, independent team-member pilot drawn from the same Tiny-GenImage revision. That pilot is not part of this repository, its results informed no Round 6 or Round 7 decision, and it is not used or referenced as data. No external pilot files were accessed for this documentation correction.", "Training-unseen is not the same as blind; generator domains were previously exposed even though these test images were not used or inspected in rounds 1–6 of this repository.", "Real sources are JPEG and fake sources PNG with generator-specific sizes; harmonisation controls but does not remove these confounds."],
        "verification_environment": "Verification must use the repository-pinned Pillow 12.2.0 environment because older Pillow changes JPEG pixel hashes. Use the existing isolated artifacts/round7/pillow_12_2 install or an environment with the repository-pinned version; do not regenerate data with an older decoder.",
        "licence": {"reported_by": "user", "licence": "CC BY-NC-SA 4.0", "underlying_rights": "Underlying ImageNet and generator rights still apply", "redistribution": "No images are redistributed"},
        "inherited_round6": {"status": "All preprocessing, degradation and model settings unchanged; only source pool and partitions change. Not executed in this task.",
                             "file_sha256": {p: sha(ROOT / p) for p in inherited_files},
                             "protocol_snapshot": read_json(ROOT / "results/round6/protocol.json"),
                             "profiles": ["original", "common128"], "conditions": ["clean", "jpeg70", "jpeg50", "resize05"],
                             "partition_mapping": {"expert_fit": "expert_train", "gate_fit": "gate_train", "calibration": "calibration", "selection": "validation", "reserved_test": "development condition recipe only; test remains reserved"},
                             "count_notes": "Round 6's literal 400/300 view counts are historical, not Round 7 targets. Preserve C=1, per-original total loss weight one, one-third degradation doses, architecture, gate initializations and selection rules; derive view counts from the new partitions."},
        "open_decisions": ["Whether to add extra test-only degradations later; no additions approved or executed here."],
        "files_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in OUTPUTS if p.exists()},
        "local_audit_artifacts_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in [ART / "source_hashes_pillow12_2.jsonl", ART / "source_hashes_pillow12_2_meta.json", decision_path, ART / "protected_files_before.json"]},
        "implementation_sha256": {f"scripts/{name}": sha(ROOT / "scripts" / name) for name in ["round7_prepare_data.py", "round7_verify_data.py", "round7_test_data.py"]},
    }
    # Write the explanatory document before sealing its hash into the protocol.
    doc = make_document(audit, protocol)
    with (ROOT / "docs/ROUND7_DATA.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(doc)
    protocol["files_sha256"]["docs/ROUND7_DATA.md"] = sha(ROOT / "docs/ROUND7_DATA.md")
    write_new(OUT / "data_protocol.json", protocol)
    protected_snapshot()
    print(json.dumps({"status": "prepared_not_scored", "filter_counts": reasons,
                      "fitting_per_class": counts, "test_counts": test_counts, "shortfalls": shortfalls}, indent=2), flush=True)


def make_document(audit, protocol):
    counts = audit["counts_per_class_each_seed"]
    lines = ["# Round 7 data protocol", "", "This round is a data audit and split definition only. No feature extraction, training, evaluation, image export or model scoring was performed.", "",
             "## Source and approved deviation", "", f"`{DATASET}` at revision `{REVISION}`, all 14 SHA-256-verified train shards (28,000 source rows).",
             protocol["protocol_deviation"], "The earlier Step 1 stop report is retained unchanged as a historical record.", "",
             "Licence reported by the user: **CC BY-NC-SA 4.0**. Underlying ImageNet and generator rights still apply. No images are redistributed.", "",
             "## Exclusion and deduplication", "", "Every original is checked against all 1,600 Round 2 images using the exact `extract_record` pixel-SHA256 definition and dHash distance <=3. Matching rows are excluded before any sampling. All remaining generators and labels are deduplicated together in ascending numeric source-row order; keep the earliest eligible representative. The deduplication is greedy, not transitive clustering. No retained pair can have equal pixel hashes or dHash distance <=3. There is no crop-hash filter.", "",
             "| Stage | Rows |", "|---|---:|"]
    lines += [f"| {key} | {value:,} |" for key, value in audit["filter_counts"].items()]
    lines += ["", "Counts above are exclusive removal categories, not pair counts. Full per-generator accounting is in `data_audit.json`; row-level decisions remain local under `artifacts/round7/`.", "",
              "## Fixed fitting pool and three partitions", "", f"Pool/test seed: `{POOL_SEED}`. Partition seeds: `{SEEDS}`. Reserve test first, then select one fixed balanced Real+SD15 fitting pool. Only assignments within that same pool change between seeds.", "",
              "| Partition | Real | SD15 | Total |", "|---|---:|---:|---:|"]
    lines += [f"| {part} | {counts[part]} | {counts[part]} | {2 * counts[part]} |" for part in PARTS]
    lines += ["", "Targets per class are 750/250/125/125. If necessary, scale proportionally using largest remainders; ties follow the displayed partition order. All four partitions are class-balanced and disjoint at the original-image level.", "",
              "## Reserved test (identical across all three seeds)", "", "| Generator | Role | Originals |", "|---|---|---:|"]
    lines += [f"| {g} | {protocol['test_roles'][g]} | {n} |" for g, n in audit["reserved_test_counts"].items()]
    lines += ["", f"Unique test originals: **{audit['reserved_test_total_unique']}**. One shared Real pool is reused across generator comparisons and counted only once in pooled metrics. No test original enters the fitting pool. Test records contain identifiers, labels and hashes only, with no image bytes, thumbnails, features or scores.", "",
              "Shortfalls (target minus actual):", "", "```json", json.dumps(audit["shortfalls"], indent=2), "```", "",
              "## Exposure and interpretation", "", "ADM, BigGAN and SD15 were heavily inspected in Rounds 1-6. SD15 is now the training domain and is reported only as in-domain. GLIDE, VQDM, Midjourney and Wukong were lightly exposed in a separate, independent team-member pilot; its results informed no Round 6 or Round 7 decision.",
              "Round 7 test images were not used or inspected in rounds 1–6 of this repository. Some (116 of 3,750) also appeared in a separate, independent team-member pilot drawn from the same source; that pilot did not inform any Round 7 setting.",
              "**Training-unseen is not the same as blind.** These tests use previously exposed generator domains and are not a strictly blind generator benchmark.", "",
              "## Limitations", "", "- The user reports an independent check found overlap of 116/3,750 reserved-test images and 94/2,500 fitting-pool images with a separate, independent team-member pilot drawn from the same Tiny-GenImage revision. That pilot is not part of this repository, its results informed no Round 6 or Round 7 decision, and it is not used or referenced as data. No external pilot files were accessed for this documentation correction.", "- Training-unseen is not the same as blind; generator domains were previously exposed even though these test images were not used or inspected in rounds 1–6 of this repository.", "- Real sources are JPEG and fake sources PNG with generator-specific sizes; harmonisation controls but does not remove these confounds.", "",
              "## Format and size confounds", "", "Real sources are **100% JPEG** and generated sources **100% PNG**. Real dimensions vary (500x375 is most common). BigGAN is 128x128; ADM/GLIDE/VQDM 256x256; SD15/Wukong 512x512. Midjourney has 1,999 images at 1024x1024 and one at 512x512. Complete source and selected format/size histograms are in the audit.",
              "Harmonisation controls observable format/size differences but does not remove source compression history, effective resolution, content or generator confounds. Common128 is the inherited resize/crop control, not proof that the confounds are eliminated; no new harmonisation pipeline was introduced here.", "",
              "## Settings inherited unchanged from Round 6", "", "- Preprocessing: original (RGB centre fit to 224, bicubic) and common128 (RGB centre fit to 128, then bicubic resize to 224).",
              "- Conditions: clean, JPEG70, JPEG50 and resize05 (224 -> 112 -> 224, bicubic; JPEG subsampling=2). Gate fitting/calibration/selection use clean/JPEG70/resize05; reserved-test condition recipe remains clean/JPEG50/resize05.",
              "- Frozen CLIP ViT-B/32, standardized logistic heads with C=1, four fixed 56px PatchShuffle views, all four matched-dose expert recipes, raw/temperature calibration, fusion rules, gate architecture/optimization/initialization seeds, threshold and selection policies are unchanged.",
              "- Round 6's fixed image/view counts are historical. Round 7 derives those counts from the new partitions while preserving per-original total weight one and one-third degradation doses. The three partition seeds are distinct from the unchanged three gate initialization seeds.",
              "The protocol embeds the historical Round 6 protocol and pins all inherited implementation files by SHA-256. Do not run the old round runners for Round 7: they write their historical output paths. A future Round 7 runner must consume these assignments separately.",
              "**Open decision:** whether to add extra test-only degradations later. No extra degradations have been approved or applied.", "",
              "## Reproduction and verification", "", "Verification must use the repository-pinned Pillow 12.2.0 environment because older Pillow changes JPEG pixel hashes. Anaconda initially had Pillow 10.4, which changed 321 historical real pixel hashes (one also changed dHash); that preliminary inventory is not used. A new isolated `artifacts/round7/pillow_12_2/` install supplies the already-pinned version without altering Anaconda or requirements. Both new scripts prefer that local install when present; otherwise use the environment's pinned version. Final preparation requires all 1,600 historical pixel/dHashes to match exactly.", "", "Run from the repository root:", "", "```powershell",
              "$env:PYTHONDONTWRITEBYTECODE='1'", "python scripts/round7_test_data.py", "python -u scripts/round7_prepare_data.py",
              "python -u scripts/round7_verify_data.py", "python scripts/verify_shared_results.py", "```", "",
              "Preparation refuses to overwrite any result or document. Existing completed source/hash inventories are read-only inputs. To verify an already prepared round, skip preparation. The independent verifier rechecks every shard, rehashes all original images, independently replays exclusion/deduplication and seeded sampling, checks disjointness and preserved historical bytes, and performs no model operations. Its optional `--report` creates a new report exclusively.", "",
              "The shared-results verifier reads already saved predictions to check historical results; it does not score any new image. Commit only the new Round 7 scripts, results and documents; keep downloaded shards and local audit caches out of Git.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
