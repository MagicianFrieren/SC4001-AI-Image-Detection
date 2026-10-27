"""Round 8 A development experiment: isolated I/O, protocol and image access.

Only new Round 8 paths are writable. No final-test evaluator is provided.
"""
import sys
sys.dont_write_bytecode = True
import contextlib
import hashlib
import io
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import round7_baseline_common as c
import round7_baseline_data as images
import round7_seed_support as seeds
import round7_stepD_common as w1
from PIL import Image, ImageOps

ROOT = c.ROOT
OUT = ROOT / 'results/round8_a'
ART = ROOT / 'artifacts/round8_a'
PLAN = ROOT / 'config/round8_a.json'
ARMS = ('unmatched', 'matched')
PARTS = ('expert_fit', 'gate_fit', 'selection', 'development')
FILES = ('round8_a_common.py', 'round8_a_run.py', 'round8_a_verify.py',
         'round7_baseline_common.py', 'round7_baseline_data.py',
         'round7_baseline_clip.py', 'round7_seed_support.py',
         'round7_stepD_common.py', 'gate_models.py', 'score_calibration.py')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def publish(path, payload):
    """Idempotent, exclusive atomic publication; changed bytes are rejected."""
    w1.publish(path, payload, resume=True)


def write_json(path, obj):
    publish(path, w1.json_bytes(obj))


def write_csv(path, rows):
    publish(path, w1.csv_bytes(rows))


@contextlib.contextmanager
def run_lock():
    """Kernel-owned lock releases on crash, unlike a stale PID/sentinel file."""
    ART.mkdir(parents=True, exist_ok=True)
    with (ART / 'runner.lock').open('a+b') as stream:
        stream.seek(0)
        if stream.read(1) == b'':
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def context(seed):
    """Validate inherited manifests without opening the reserved-test manifest."""
    previous = c.SEED
    try:
        c.SEED = seed
        records, parts, deny = seeds.context()
    finally:
        c.SEED = previous
    parts = {part: sorted(parts[part]) for part in PARTS}
    allowed = set(sum(parts.values(), []))
    return {i: records[i] for i in sorted(allowed)}, parts, deny


class Guard(c.Guard):
    def __init__(self, records, parts, deny):
        super().__init__(records, deny)
        self.parts = parts

    def touch(self, ids, action):
        ids = list(ids)
        super().touch(ids, action)
        scope = {'fit_experts': 'expert_fit', 'fit_gate': 'gate_fit',
                 'select': 'selection', 'report_selection': 'selection',
                 'report_development': 'development'}
        if action in scope:
            require(set(ids) <= set(self.parts[scope[action]]), f'Wrong partition: {action}')
        if action == 'format_match':
            allowed = sum((self.parts[p] for p in ('expert_fit', 'gate_fit', 'selection')), [])
            require(set(ids) <= set(allowed), 'Cannot format-match development images')


def environment():
    import torch
    result = {p: version(p) for p in ('torch', 'torchvision', 'transformers',
              'numpy', 'scipy', 'scikit-learn', 'Pillow', 'pyarrow', 'joblib')}
    result.update(python=platform.python_version(), platform=platform.platform(),
                  device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
    return result


def weights_path():
    choices = [ART / 'weights/clip', ROOT / '.cache/clip-vit-base-patch32']
    choices += [ROOT / f'artifacts/round7/baselines_seed{i}/weights/clip' for i in (1, 2, 3)]
    expected = c.read(ROOT / 'results/round7/stepD_protocol.json')['clip']['weights_sha256']
    for folder in choices:
        if all((folder / n).is_file() for n in ('pytorch_model.bin', 'config.json')):
            require(c.sha(folder / 'pytorch_model.bin') == expected, 'CLIP weight hash mismatch')
            return folder
    raise FileNotFoundError('Pinned CLIP weights missing; supply the Round 7 weights/clip folder')


def protocol():
    plan = c.read(PLAN)
    require(plan['arms'] == list(ARMS) and plan['seeds'] == list(seeds.SEEDS), 'Unexpected experiment plan')
    require(c.PIL.__version__ == '12.2.0', 'Pillow 12.2.0 required for pinned pixels')
    base = c.read(ROOT / 'results/round7/stepD_protocol.json')
    for name, digest in base['inherited_code_sha256'].items():
        require(c.sha(ROOT / name) == digest, f'Inherited source changed: {name}')
    needed = ['data_protocol.json', 'data_protocol_step_b.json', 'step_b_isolation.json',
              'fitting_pool_manifest.jsonl', 'development_manifest.jsonl', 'source_inventory.json',
              'stepD_protocol.json'] + [f'assignments_seed{i}.json' for i in (1, 2, 3)]
    for seed in seeds.SEEDS:
        context(seed)
    weights = weights_path()
    return dict(plan=plan, plan_sha256=c.sha(PLAN), environment=environment(),
                code_sha256={f'scripts/{name}': c.sha(ROOT / 'scripts' / name) for name in FILES},
                input_sha256={f'results/round7/{name}': c.sha(ROOT / 'results/round7' / name) for name in needed},
                weights_sha256={n: c.sha(weights / n) for n in ('pytorch_model.bin', 'config.json')},
                format_match=base['format_match'], clip=base['clip'])


def ensure_protocol():
    value = protocol()
    write_json(OUT / 'protocol.json', value)
    return value, c.sha(OUT / 'protocol.json')


def original_path(identity):
    require(identity.startswith('train:') and identity[6:].isdigit(), 'Unsafe source ID')
    return ART / 'allowed_images' / (identity.replace(':', '_') + '.bin')


def find_original(identity):
    choices = [original_path(identity)]
    choices += [ROOT / f'artifacts/round7/baselines_seed{i}/allowed_images' /
                (identity.replace(':', '_') + '.bin') for i in (1, 2, 3)]
    return next((p for p in choices if p.is_file()), None)


def load_original(identity, records, guard):
    guard.touch([identity], 'decode')
    path = find_original(identity)
    if path is None:
        raise FileNotFoundError(f'Missing original {identity}; run --prepare (optionally --download)')
    blob = path.read_bytes()
    require(hashlib.sha256(blob).hexdigest() == records[identity]['source_bytes_sha256'], 'Original bytes changed')
    with Image.open(io.BytesIO(blob)) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    require(digest == records[identity]['pixel_sha256'], 'Decoded pixel hash mismatch')
    return image


def view(identity, condition, matched, seed, records, guard, spec):
    guard.touch([identity], 'view')
    if matched:
        guard.touch([identity], 'format_match')
    image = load_original(identity, records, guard)
    if matched:
        image = w1.match_format(image, seed, identity, spec)
    return images.view(image, 'original', condition)


def shard_path(info):
    choices = [ROOT / info['local_path'], ROOT / 'data/shards' / Path(info['path']).name,
               ART / 'source' / Path(info['path']).name]
    return next((p for p in choices if p.is_file()), None)


def preflight():
    """Metadata and local availability only; no test images or model scoring."""
    records = {}
    for seed in seeds.SEEDS:
        rows, _, _ = context(seed)
        records.update(rows)
    inventory = c.read(ROOT / 'results/round7/source_inventory.json')
    missing = [s for s in inventory['shards'] if shard_path(s) is None]
    try:
        weights = str(weights_path())
    except FileNotFoundError:
        weights = None
    return dict(environment=environment(), permitted_originals=len(records),
                cached_originals=sum(find_original(i) is not None for i in records),
                local_shards=len(inventory['shards']) - len(missing),
                missing_shards=[s['path'] for s in missing],
                download_bytes=sum(s['bytes'] for s in missing), weights=weights,
                historical_environment=c.read(ROOT / 'results/round7/baselines_seed1/summarize_complete.json')['environment'],
                note='Development only; controls refitted locally. No reserved-test manifest access.')


def prepare(download=False):
    """Restore only fitting/development bytes, selecting Arrow rows before decoding."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    records = {}
    for seed in seeds.SEEDS:
        rr, _, deny = context(seed)
        records.update(rr)
    guard = c.Guard(records, deny)
    remaining = set()
    for identity, record in records.items():
        guard.touch([identity], 'source_lookup')
        path = find_original(identity)
        if path is None:
            remaining.add(identity)
        else:
            require(c.sha(path) == record['source_bytes_sha256'], f'Corrupt original: {identity}')
    inventory = c.read(ROOT / 'results/round7/source_inventory.json')
    for index, info in enumerate(inventory['shards']):
        wanted = {int(i[6:]) % 2000: i for i in remaining if int(i[6:]) // 2000 == index}
        if not wanted:
            continue
        path = shard_path(info)
        if path is None and download:
            path = ART / 'source' / Path(info['path']).name
            print(f'Downloading {path.name}: {info["bytes"] / 1e6:.0f} MB', flush=True)
            images.download(f'https://huggingface.co/datasets/{inventory["dataset"]}/resolve/{inventory["revision"]}/{info["path"]}',
                            path, digest=info['lfs_sha256'])
        if path is None:
            continue
        require(c.sha(path) == info['lfs_sha256'], f'Corrupt shard: {path}')
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=['image']):
            selected = [j for j in range(len(batch)) if offset + j in wanted]
            if selected:
                identities = [wanted[offset + j] for j in selected]
                guard.touch(identities, 'source_payload_select')
                for identity, row in zip(identities, batch.take(pa.array(selected)).to_pylist()):
                    blob = row['image']['bytes']
                    require(hashlib.sha256(blob).hexdigest() == records[identity]['source_bytes_sha256'], 'Source payload changed')
                    publish(original_path(identity), blob)
                    remaining.remove(identity)
            offset += len(batch)
        print(f'Shard {index + 1}/14 restored; {len(remaining)} allowed originals remaining', flush=True)
    require(not remaining, f'{len(remaining)} originals missing. Supply pinned shards or use --prepare --download.')
    return dict(status='passed', originals=len(records), violations=guard.violations)
