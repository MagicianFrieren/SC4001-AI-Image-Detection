"""Allowlisted original images, Round 6 pixel transforms and safe weight downloads."""
import sys
sys.dont_write_bytecode = True
import hashlib
import io
import os
import uuid
import round7_baseline_common as c
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import laplace

CLIP_MEAN = np.array([.48145466, .4578275, .40821073], dtype=np.float32)
CLIP_STD = np.array([.26862954, .26130258, .27577711], dtype=np.float32)
IMAGENET_MEAN = np.array([.485, .456, .406], dtype=np.float32)
IMAGENET_STD = np.array([.229, .224, .225], dtype=np.float32)
VIEW_SEEDS = [202610220, 202610221, 202610222, 202610223]


def source_path(identity):
    c.require(identity.startswith("train:") and identity[6:].isdigit(), "Unsafe source ID")
    return c.ART / "allowed_images" / (identity.replace(":", "_") + ".bin")


def prepare_images(records, guard):
    """Select Arrow rows before converting any image payload to Python/PIL.

    Parquet page I/O may include compressed neighbouring rows; no forbidden
    payload is materialized as a Python image, decoded, exported or scored.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    directory = c.ART / "allowed_images"
    directory.mkdir(parents=True, exist_ok=True)
    remaining = set(records)
    for identity in sorted(records):
        path = source_path(identity)
        if path.exists():
            guard.touch([identity], "source_cache_verify")
            c.require(c.sha(path) == records[identity]["source_bytes_sha256"], "Source cache corrupt")
            remaining.remove(identity)
    inventory = c.read(c.DATA / "source_inventory.json")
    for shard_index, info in enumerate(inventory["shards"]):
        wanted = {int(i.split(":")[1]) % 2000: i for i in remaining if int(i.split(":")[1]) // 2000 == shard_index}
        if not wanted:
            continue
        path = c.ROOT / info["local_path"]
        c.require(c.sha(path) == info["lfs_sha256"], "Source shard hash mismatch")
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=["image"]):
            indices = [j for j in range(len(batch)) if offset + j in wanted]
            identities = [wanted[offset+j] for j in indices]
            if indices:
                guard.touch(identities, "source_payload_select")
                chosen = batch.take(pa.array(indices, type=pa.int64())).to_pylist()
                for identity, row in zip(identities, chosen):
                    blob = row["image"]["bytes"]
                    c.require(hashlib.sha256(blob).hexdigest() == records[identity]["source_bytes_sha256"], "Original payload changed")
                    with source_path(identity).open("xb") as f:
                        f.write(blob)
                    remaining.remove(identity)
            offset += len(batch)
        print(f"Cached permitted originals: shard {shard_index+1}/14", flush=True)
    c.require(not remaining, "Missing allowed original images")


def load_image(identity, records, guard):
    guard.touch([identity], "image_decode")  # MUST precede all image access.
    with Image.open(source_path(identity)) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
    actual = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    c.require(actual == records[identity]["pixel_sha256"], "Decoded pixel hash differs (check Pillow version)")
    return image


def view(image, profile, condition):
    c.require(profile in c.PROFILES and condition in c.CONDITIONS, "Unknown image condition")
    size = 224 if profile == "original" else 128
    result = ImageOps.fit(image, (size, size), method=Image.Resampling.BICUBIC)
    if profile == "common128":
        result = result.resize((224, 224), Image.Resampling.BICUBIC)
    if condition.startswith("jpeg"):
        buffer = io.BytesIO()
        result.save(buffer, format="JPEG", quality=int(condition[4:]), subsampling=2)
        buffer.seek(0)
        result = Image.open(buffer).convert("RGB")
    elif condition == "resize05":
        result = result.resize((112, 112), Image.Resampling.BICUBIC).resize((224, 224), Image.Resampling.BICUBIC)
    return result


def normalized(image, kind):
    mean, std = (CLIP_MEAN, CLIP_STD) if kind == "clip" else (IMAGENET_MEAN, IMAGENET_STD)
    return ((np.asarray(image, dtype=np.float32) / 255 - mean) / std).transpose(2, 0, 1)


def shuffled(image, seed):
    array = np.asarray(image)
    order = np.random.default_rng(seed).permutation(16)
    tiles = array.reshape(4, 56, 4, 56, 3).transpose(0, 2, 1, 3, 4).reshape(16, 56, 56, 3)
    return Image.fromarray(tiles[order].reshape(4, 4, 56, 56, 3).transpose(0, 2, 1, 3, 4).reshape(224, 224, 3))


def quality(image):
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255
    energy = np.abs(np.fft.fftshift(np.fft.fft2(gray-gray.mean()))) ** 2
    yy, xx = np.ogrid[-112:112, -112:112]
    high = energy[(xx*xx + yy*yy) > 56**2].sum() / (energy.sum()+1e-9)
    dx, dy = np.abs(np.diff(gray, axis=1)), np.abs(np.diff(gray, axis=0))
    block = (dx[:, 7::8].mean()+dy[7::8].mean()) / (dx.mean()+dy.mean()+1e-9)
    return [np.log1p(1e4*laplace(gray).var()), high, block]


def download(url, destination, digest=None, prefix=None):
    """Exclusive local weight cache; never read or write older-round caches."""
    import requests
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".part")
        with requests.get(url, stream=True, timeout=(30, 90)) as response:
            response.raise_for_status()
            with temporary.open("xb") as f:
                for chunk in response.iter_content(1024 * 1024):
                    f.write(chunk)
        actual = c.sha(temporary)
        c.require((digest is None or actual == digest) and (prefix is None or actual.startswith(prefix)), "Downloaded weights hash mismatch")
        if os.name == "nt":
            temporary.rename(destination)
        else:
            os.link(temporary, destination); temporary.unlink()
    actual = c.sha(destination)
    c.require((digest is None or actual == digest) and (prefix is None or actual.startswith(prefix)), "Cached weights hash mismatch")
    return destination
