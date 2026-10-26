"""Step D protocol, immutable I/O, guarded views and per-seed metrics."""
import sys
sys.dont_write_bytecode = True
import csv
import hashlib
import io
import json
import os
import uuid
from pathlib import Path
import numpy as np
import round7_baseline_common as c
import round7_baseline_data as images
import round7_seed_support as seeds
from PIL import Image, ImageOps
from scipy.special import expit

ROOT = c.ROOT
PROTOCOL = ROOT / 'results/round7/stepD_protocol.json'
OUT = ROOT / 'results/round7/stepD_whole_image'
ART = ROOT / 'artifacts/round7/stepD_whole_image'
CODE = ('round7_stepD_common.py', 'round7_stepD_run.py', 'round7_stepD_report.py', 'round7_stepD_verify.py')
METRICS = ('auroc', 'balanced_accuracy', 'real_false_positive_rate')
DOMAINS = {'SD15': 'selection', 'ADM': 'development', 'BigGAN': 'development'}


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def csv_bytes(rows):
    c.require(bool(rows), 'Empty CSV')
    stream = io.StringIO(newline='')
    # JSON bundles sort keys; canonical columns keep first publication and resume identical.
    writer = csv.DictWriter(stream, fieldnames=sorted(rows[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def publish(path, payload, resume=False):
    """Atomic exclusive publish; a repeated publication must match byte-for-byte."""
    path = Path(path)
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        c.require(path.read_bytes() == payload, f'Refusing changed output: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('xb') as stream:
        stream.write(payload)
    if os.name == 'nt':
        temporary.rename(path)
    else:
        os.link(temporary, path); temporary.unlink()


def validate_protocol():
    p = c.read(PROTOCOL)
    snapshot = c.read(ROOT / 'results/round7/stepD_preservation.json')
    c.require(c.sha(PROTOCOL) == snapshot['protocol_sha256'], 'Pre-training Step D protocol changed')
    c.require(c.PIL.__version__ == '12.2.0', 'Pillow 12.2.0 required')
    for field in ('input_sha256', 'inherited_code_sha256'):
        for name, digest in p[field].items():
            c.require(c.sha(ROOT/name) == digest, f'Frozen input/code changed: {name}')
    return p


def lock_implementation(create=False):
    validate_protocol()
    value = {'protocol_sha256': c.sha(PROTOCOL), 'code_sha256': {f'scripts/{name}': c.sha(ROOT/'scripts'/name) for name in CODE}}
    path = OUT / 'implementation.json'
    if create:
        publish(path, json_bytes(value), resume=True)
    c.require(path.exists() and c.read(path) == value, 'Implementation changed or not initialized; cannot reuse run')
    return value


def configure(seed, variant):
    c.require(variant in ('W0', 'W1', 'W2'), 'Unknown variant')
    seeds.configure(seed)
    records, parts, deny = seeds.context()
    parts = {k: sorted(parts[k]) for k in ('expert_fit', 'selection', 'development')}
    allowed = set(sum(parts.values(), []))
    records = {i: r for i, r in records.items() if i in allowed}
    c.OUT = OUT / f'seed{seeds.number(seed)}' / variant
    c.ART = ART / f'seed{seeds.number(seed)}' / variant
    return records, parts, deny


class Guard(c.Guard):
    def __init__(self, records, parts, deny):
        super().__init__(records, deny)
        self.parts = parts

    def touch(self, ids, action):
        ids = list(ids)
        super().touch(ids, action)
        restrictions = {'fit_clip_expert': 'expert_fit', 'fit_resnet_expert_only': 'expert_fit',
                        'resnet_resume_expert_partition': 'expert_fit', 'resnet_resume_selection_partition': 'selection',
                        'select_resnet_epoch': 'selection', 'tune_threshold': 'selection'}
        if action in restrictions:
            c.require(set(ids) <= set(self.parts[restrictions[action]]), f'Wrong partition: {action}')
        if action.startswith('report/'):
            c.require(set(ids) <= set(self.parts[action.split('/')[1]]), 'Wrong reporting partition')
        if action == 'format_match':
            c.require(set(ids) <= set(self.parts['expert_fit'] + self.parts['selection']), 'Cannot format-match development images')


def quality(seed, source_id, lo=60, hi=95):
    """Integer-uniform hash rejection sampling, independent of label/order/epoch."""
    width = hi - lo + 1
    c.require(width > 0 and 1 <= lo <= hi <= 100, 'Invalid JPEG range')
    limit = 2**256 - (2**256 % width)
    counter = 0
    while True:
        value = int.from_bytes(hashlib.sha256(f'stepD|{seed}|{source_id}|{counter}'.encode()).digest(), 'big')
        if value < limit:
            return lo + value % width
        counter += 1


def match_format(image, seed, source_id, spec):
    buffer = io.BytesIO()
    image.convert('RGB').save(buffer, format='JPEG', quality=quality(seed, source_id, spec['quality_min'], spec['quality_max']),
                              subsampling=2, optimize=False, progressive=False)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert('RGB')


def load_original(identity, records, guard):
    guard.touch([identity], 'original_cache_read')  # Before path lookup, hash, or decode.
    path = ROOT / f'artifacts/round7/baselines_seed{seeds.number(c.SEED)}/allowed_images' / (identity.replace(':', '_') + '.bin')
    c.require(c.sha(path) == records[identity]['source_bytes_sha256'], 'Step C original cache changed')
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert('RGB')
    digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    c.require(digest == records[identity]['pixel_sha256'], 'Decoded pixel hash mismatch')
    return image


def view(identity, condition, matched, records, guard, protocol):
    guard.touch([identity], 'view/matched' if matched else 'view/original')
    if matched:
        guard.touch([identity], 'format_match')
    image = load_original(identity, records, guard)
    if matched:
        image = match_format(image, c.SEED, identity, protocol['format_match'])
    return images.view(image, 'original', condition)


def tensor_batch(matched, protocol):
    def batch(ids, profile, condition, records, guard, device):
        import torch
        c.require(profile == 'original', 'Step D only allows original profile')
        guard.touch(ids, 'resnet_input')
        x = np.stack([images.normalized(view(i, condition, matched, records, guard, protocol), 'resnet') for i in ids])
        y = np.array([records[i]['label'] for i in ids], dtype=np.float32)
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    return batch


def prediction_rows(raw, variant, seed, partition, threshold):
    return [dict(variant=variant, seed=seed, profile='original', calibration='raw', partition=partition, **r,
                 probability_fake=float(expit(r['logit'])), threshold=float(threshold), view='original') for r in raw]


def w0_rows(seed, records, parts, guard):
    path = ROOT/f'results/round7/baselines_seed{seeds.number(seed)}/clip_original_predictions.csv'
    rows = []
    for r in c.read_csv(path):
        if r['model'] != 'global' or r['calibration'] != 'raw':
            continue
        guard.touch([r['source_id']], 'report/' + r['partition'])
        rows.append(dict(variant='W0', seed=seed, profile='original', calibration='raw', partition=r['partition'],
                         source_id=r['source_id'], generator=r['generator'], label=int(r['label']), condition=r['condition'],
                         logit=float(r['logit']), probability_fake=float(r['probability_fake']), threshold=float(r['threshold']), view='original'))
    return rows


def metrics(rows):
    result = []
    for variant, seed in sorted({(r['variant'], int(r['seed'])) for r in rows}):
        for generator, part in DOMAINS.items():
            for condition in c.CONDITIONS:
                cell = [r for r in rows if r['variant']==variant and int(r['seed'])==seed and r['partition']==part and r['condition']==condition and r['generator'] in ('Real',generator)]
                c.require(bool(cell), 'Incomplete metric coverage')
                cut = float(cell[0]['threshold'])
                c.require(all(float(r['threshold']) == cut for r in cell), 'Inconsistent threshold')
                y = np.array([int(r['label']) for r in cell]); z = np.array([float(r['logit']) for r in cell])
                values = c.points(y,z,cut)
                result.append(dict(variant=variant, seed=seed, generator=generator, partition=part, condition=condition,
                                   n_real=int(sum(y==0)), n_fake=int(sum(y==1)), **{k:values[k] for k in METRICS}))
    return result


def check_preservation():
    manifest = c.read(ROOT/'results/round7/stepD_preservation.json')
    for name, digest in manifest['files_sha256'].items():
        c.require(c.sha(ROOT/name)==digest, f'Existing file changed: {name}')
    return len(manifest['files_sha256'])
