"""Round 7 Step 1 only: verify and inventory the complete pinned train source.

No image exports, feature extraction, model scores, or split assignments.
Existing completed source shards are verified read-only. Existing reports or
partial downloads are never overwritten. A missing SD1.4 exits with status 3.
"""
import sys

sys.dont_write_bytecode = True

import concurrent.futures as futures
import io
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import requests
from PIL import Image

import prepare_data as source


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts/round7/source"
REPORT = ROOT / "results/round7/source_inventory.json"
SUMMARY = ROOT / "docs/ROUND7_SOURCE_INSPECTION.md"


def inventory(path):
    """Read image headers only; do not load pixels or compute image features."""
    parquet = pq.ParquetFile(path)
    metadata = json.loads(parquet.schema_arrow.metadata[b"huggingface"])
    features = metadata["info"]["features"]
    names = features["generator"]["names"]
    counts = Counter()
    formats = {name: Counter() for name in names}
    sizes = {name: Counter() for name in names}
    labels = {name: Counter() for name in names}
    for batch in parquet.iter_batches(batch_size=32, columns=["image", "label", "generator"]):
        for row in batch.to_pylist():
            name = names[row["generator"]]
            counts[name] += 1
            labels[name][str(row["label"])] += 1
            with Image.open(io.BytesIO(row["image"]["bytes"])) as image:
                formats[name][str(image.format)] += 1
                sizes[name][f"{image.width}x{image.height}"] += 1
    print(f"Inspected {path.name}: {dict(counts)}", flush=True)
    return features, counts, formats, sizes, labels


def main():
    for path in (REPORT, SUMMARY):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    response = requests.get(
        f"https://huggingface.co/api/datasets/{source.DATASET}/tree/{source.REVISION}/data",
        params={"limit": 30}, timeout=60,
    )
    response.raise_for_status()
    wanted = {f"data/train-{index:05d}-of-00014.parquet" for index in range(14)}
    infos = sorted((item for item in response.json() if item["path"] in wanted),
                   key=lambda item: item["path"])
    if len(infos) != 14:
        raise ValueError(f"Expected 14 training shards, found {len(infos)}")
    # Reuse prepare_data.py's exact ranged-download and LFS SHA-256 logic.
    # Redirect its destination IN MEMORY ONLY; never edit the imported module.
    source.ROOT = CACHE
    for info in infos:
        path = CACHE / "data/shards" / Path(info["path"]).name
        if not path.exists() and path.with_suffix(".ranged.part").exists():
            raise FileExistsError(f"Refusing to overwrite partial {path.with_suffix('.ranged.part')}")
    with futures.ThreadPoolExecutor(max_workers=4) as executor:
        paths = list(executor.map(source.download_shard, infos))
    combined = None
    shard_records = []
    for info, path in zip(infos, paths):
        features, counts, formats, sizes, labels = inventory(path)
        if combined is None:
            names = features["generator"]["names"]
            combined = {
                "features": features, "counts": Counter(),
                "formats": {name: Counter() for name in names},
                "sizes": {name: Counter() for name in names},
                "labels": {name: Counter() for name in names},
            }
        if features != combined["features"]:
            raise ValueError(f"Inconsistent feature metadata: {path}")
        combined["counts"].update(counts)
        for name in names:
            for key, values in (("formats", formats), ("sizes", sizes), ("labels", labels)):
                combined[key][name].update(values[name])
        shard_records.append({"path": info["path"], "bytes": info["size"],
                              "lfs_sha256": info["lfs"]["oid"],
                              "verified_sha256": source.sha256(path),
                              "local_path": path.relative_to(ROOT).as_posix(),
                              "rows": sum(counts.values()), "counts": dict(counts)})
    sd14_names = [name for name in names if re.sub(r"[^a-z0-9]", "", name.lower())
                  in {"sd14", "stablediffusion14", "stablediffusionv14"}]
    sd14_count = sum(combined["counts"][name] for name in sd14_names)
    report = {
        "dataset": source.DATASET, "revision": source.REVISION, "split": "train",
        "scope": "Step 1 only; complete 14-shard source inventory before any exclusions or splits",
        "status": "sd14_present" if sd14_count else "STOP_SD14_ABSENT",
        "sd14_count": sd14_count,
        "generator_classlabel": dict(enumerate(names)), "features": combined["features"],
        "total_images": sum(combined["counts"].values()),
        "counts_per_generator": {name: combined["counts"][name] for name in names},
        "format_by_generator": combined["formats"],
        "native_encoded_size_by_generator": combined["sizes"],
        "label_counts_by_generator": combined["labels"],
        "size_definition": "Encoded image header width x height, before EXIF orientation, cropping or resizing",
        "shards": shard_records,
        "method": "Existing prepare_data.download_shard and sha256 reused read-only; PIL headers only; no pixel decode requested, no models, no partitions",
    }
    lines = ["# Round 7 source inspection (Step 1)", "",
             f"Source: `{source.DATASET}`; pinned revision `{source.REVISION}`.", "",
             f"All 14 train shards verified against Hugging Face LFS SHA-256; {report['total_images']:,} images.", "",
             "| ClassLabel | Generator (source spelling) | Images | Encoded formats | Most common encoded sizes |",
             "|---|---|---:|---|---|"]
    for index, name in enumerate(names):
        fmt = ", ".join(f"{key}: {value}" for key, value in combined["formats"][name].most_common()) or "none"
        size = ", ".join(f"{key}: {value}" for key, value in combined["sizes"][name].most_common(5)) or "none"
        lines.append(f"| {index} | {name} | {combined['counts'][name]} | {fmt} | {size} |")
    lines += ["", "The complete size histogram and each verified shard hash are in `results/round7/source_inventory.json`.",
              "Sizes are image-header dimensions before EXIF orientation or preprocessing. Image headers were inspected without exporting images or computing model scores.", ""]
    if not sd14_count:
        lines += ["**STOP: SD1.4 has no images in the complete pinned training source.**",
                  "Per the requested stop condition, no exclusions, fitting pool, seed assignments, reserved test set, or final data protocol have been created. Steps 2–5 and their verifier remain pending a revised source or training-generator decision.", ""]
    for path in (REPORT, SUMMARY):
        path.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with SUMMARY.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print("\n".join(lines), flush=True)
    return 0 if sd14_count else 3


if __name__ == "__main__":
    raise SystemExit(main())
