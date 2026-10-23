"""Offline, fixed-budget preprocessing and generator-diversity ablation.

Run from the repository root with .venv/Scripts/python.exe -u scripts/run_round2.py.
The original pilot artifacts are read only. New outputs live in results/round2.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import platform
import random
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageOps
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from transformers import CLIPVisionModelWithProjection

import prepare_data as data
import run_pilot as pilot

ROOT = pilot.ROOT
OUT = ROOT / 'results' / 'round2'
SEED = 20261022
PROFILES = ['original', 'common128', 'common128_jpeg90']
CONDITIONS = ['clean', 'jpeg50', 'resize05']


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def read_manifest(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def dump_csv(path, rows):
    import csv
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def prepare(original):
    """Add 300 train and 100 validation BigGAN images; never move pilot rows."""
    old_ids = {r['source_id'] for r in original}
    pixels = {r['pixel_sha256'] for r in original}
    crops = {r['crop_sha256'] for r in original}
    dhashes = [int(r['dhash'], 16) for r in original]
    metadata = json.loads((ROOT / 'results/data_preparation.json').read_text())
    candidates, rejected = [], Counter()
    for info in metadata['source_shards']:
        path = ROOT / 'data/shards' / data.Path(info['path']).name
        if data.sha256(path) != info['lfs']['oid']:
            raise ValueError(f'Shard checksum mismatch: {path}')
        shard = int(path.name.split('-')[1])
        table = pq.read_table(path)
        assert len(table) == 2000
        for offset, row in enumerate(table.to_pylist()):
            if row['generator'] != 2 or f'train:{shard * 2000 + offset}' in old_ids:
                continue
            record, _ = data.extract_record(row, shard * 2000 + offset)
            candidates.append((record, row['image']['bytes']))
        print(f'Checked {path.name}; {len(candidates)} new BigGAN candidates', flush=True)
    random.Random(SEED).shuffle(candidates)
    selected = []
    destination = ROOT / 'data/round2'
    destination.mkdir(parents=True, exist_ok=True)
    for record, source_bytes in candidates:
        h = int(record['dhash'], 16)
        if record['pixel_sha256'] in pixels or record['crop_sha256'] in crops:
            rejected['exact_or_crop'] += 1
            continue
        if any((h ^ old).bit_count() <= 3 for old in dhashes):
            rejected['near_duplicate'] += 1
            continue
        pixels.add(record['pixel_sha256'])
        crops.add(record['crop_sha256'])
        dhashes.append(h)
        import io
        with Image.open(io.BytesIO(source_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        path = destination / f'train_{record["source_row"]}.png'
        image.save(path)
        record.update(path=path.relative_to(ROOT).as_posix(), file_sha256=data.sha256(path),
                      split='train' if len(selected) < 300 else 'validation')
        selected.append(record)
        if len(selected) == 400:
            break
    if len(selected) != 400:
        raise ValueError(f'Need 400 new BigGAN images; found {len(selected)}')
    records = original + selected
    with (OUT / 'manifest.jsonl').open('w', encoding='utf-8', newline='\n') as handle:
        for row in records:
            handle.write(json.dumps(row) + '\n')
    write_json(OUT / 'data_preparation.json', {
        'seed': SEED, 'added': len(selected), 'rejected': dict(rejected),
        'candidate_count': len(candidates), 'source_shards': metadata['source_shards'],
        'dataset_revision': metadata['revision'],
        'counts': dict(Counter(f"{r['split']}/{r['generator']}" for r in records)),
        'original_manifest_sha256': data.sha256(ROOT / 'results/manifest.jsonl')})
    return records


def validate_records(records, original):
    assert records[:len(original)] == original, 'Pilot assignments changed'
    assert len(records) == 1600
    for key in ['source_id', 'pixel_sha256', 'crop_sha256']:
        assert len({r[key] for r in records}) == len(records), key
    for i, record in enumerate(records):
        assert record['label'] == int(record['generator'] != 'Real')
        assert data.sha256(ROOT / record['path']) == record['file_sha256']
        h = int(record['dhash'], 16)
        assert all((h ^ int(old['dhash'], 16)).bit_count() > 3 for old in records[:i])
    assert all(r['generator'] == 'BigGAN' and r['split'] in ('train', 'validation')
               for r in records[len(original):])


def assignments(records):
    groups = {}
    for split, n in [('train', 300), ('validation', 100)]:
        real = [i for i, r in enumerate(records) if r['split'] == split and r['generator'] == 'Real']
        sd = [i for i, r in enumerate(records) if r['split'] == split and r['generator'] == 'SD15']
        bg = [i for i, r in enumerate(records) if r['split'] == split and r['generator'] == 'BigGAN']
        assert len(real) == len(sd) == len(bg) == n
        rng = random.Random(SEED + (split == 'validation'))
        rng.shuffle(sd)
        rng.shuffle(bg)
        groups[split] = {'single_sd15': sorted(real + sd),
                         'mixed_sd15_biggan': sorted(real + sd[:n // 2] + bg[:n // 2])}
    groups['development'] = [i for i, r in enumerate(records) if r['split'] == 'development']
    for variant in groups['train']:
        sets = [set(groups['train'][variant]), set(groups['validation'][variant]), set(groups['development'])]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        assert all(records[i]['generator'] != 'ADM' for i in sets[0] | sets[1])
    write_json(OUT / 'assignments.json', {
        split: ({variant: [records[i]['source_id'] for i in ids] for variant, ids in value.items()}
                if isinstance(value, dict) else [records[i]['source_id'] for i in value])
        for split, value in groups.items()})
    return groups


def image_view(path, profile, condition):
    if profile == 'original':
        image = pilot.base_image(path)
    else:
        with Image.open(path) as source:
            image = ImageOps.fit(source.convert('RGB'), (128, 128), method=Image.Resampling.BICUBIC)
        if profile == 'common128_jpeg90':
            image = pilot.degrade(image, 'jpeg90')
        image = image.resize((224, 224), Image.Resampling.BICUBIC)
    return pilot.degrade(image, condition)


def extract(records, batch_size):
    # Code hash prevents silent cache reuse after a preprocessing implementation change.
    identity = hashlib.sha256((json.dumps(records, sort_keys=True) + data.sha256(data.Path(__file__))
                              + data.sha256(ROOT / 'scripts/run_pilot.py')
                              + pilot.MODEL_REVISION + str(batch_size) + torch.__version__
                              + np.__version__).encode()).hexdigest()
    cache_path = ROOT / 'artifacts/round2_features.npz'
    if cache_path.exists():
        with np.load(cache_path) as cache:
            if str(cache['identity']) == identity:
                print('Using validated round-two feature cache', flush=True)
                return {k: cache[k] for k in cache.files if k != 'identity'}, True
    model_dir = ROOT / '.cache/clip-vit-base-patch32'
    assert data.sha256(model_dir / 'pytorch_model.bin') == pilot.MODEL_SHA256
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLIPVisionModelWithProjection.from_pretrained(model_dir, local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    if device == 'cuda':
        model.half()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    features = {}
    # Preserve the pilot's final partial batch. Appending new images to that
    # batch changes FP16 CUDA arithmetic even though every image is independent.
    boundary = len(read_manifest(ROOT / 'results/manifest.jsonl'))
    batches = [(offset, min(offset + batch_size, end))
               for begin, end in [(0, boundary), (boundary, len(records))]
               for offset in range(begin, end, batch_size)]
    for profile in PROFILES:
        for condition in CONDITIONS:
            vectors, qualities = [], []
            for offset, stop in batches:
                images = [image_view(ROOT / r['path'], profile, condition)
                          for r in records[offset:stop]]
                qualities.extend(pilot.quality(image) for image in images)
                array = np.stack([(np.asarray(im, dtype=np.float32) / 255 - pilot.MEAN) / pilot.STD
                                  for im in images])
                pixels = torch.from_numpy(array.transpose(0, 3, 1, 2)).to(device=device, dtype=model.dtype)
                with torch.inference_mode():
                    embedding = model(pixel_values=pixels).image_embeds.float()
                    embedding = torch.nn.functional.normalize(embedding, dim=-1)
                vectors.append(embedding.cpu().numpy())
            features[f'{profile}/{condition}'] = np.concatenate(vectors)
            features[f'quality/{profile}/{condition}'] = np.array(qualities)
            print(f'Extracted {profile}/{condition}: {len(records)} images', flush=True)
    features['extraction_seconds'] = np.array(time.perf_counter() - start)
    features['peak_cuda_memory_mb'] = np.array(torch.cuda.max_memory_allocated() / 1024 ** 2 if device == 'cuda' else 0)
    np.savez_compressed(cache_path, identity=np.array(identity), **features)
    return features, False


def run(records, groups, features):
    labels = np.array([r['label'] for r in records])
    metrics, predictions, validation_predictions, heads = [], [], [], {}
    for profile in PROFILES:
        for variant in groups['train']:
            train = groups['train'][variant]
            val = groups['validation'][variant]
            for kind in ['clip', 'quality']:
                prefix = '' if kind == 'clip' else 'quality/'
                key = f'{kind}/{profile}/{variant}'
                x = features[f'{prefix}{profile}/clean']
                head = pilot.train_head(x[train], labels[train])
                heads[key] = head
                vp = head.predict_proba(x[val])[:, 1]
                threshold = pilot.choose_threshold(labels[val], vp)
                for i, p in zip(val, vp):
                    validation_predictions.append(dict(model=key, source_id=records[i]['source_id'],
                                                       label=int(labels[i]), probability_fake=float(p)))
                for condition in CONDITIONS:
                    x = features[f'{prefix}{profile}/{condition}']
                    p = head.predict_proba(x)[:, 1]
                    for i in groups['development']:
                        predictions.append(dict(model=key, condition=condition, source_id=records[i]['source_id'],
                                                generator=records[i]['generator'], label=int(labels[i]),
                                                probability_fake=float(p[i])))
                    for generator in ['SD15', 'BigGAN', 'ADM']:
                        ids = [i for i in groups['development'] if records[i]['generator'] in ('Real', generator)]
                        y, scores = labels[ids], p[ids]
                        metrics.append(dict(model=key, kind=kind, profile=profile, training=variant,
                                            condition=condition, generator=generator, n=len(ids),
                                            auroc=float(roc_auc_score(y, scores)),
                                            average_precision=float(average_precision_score(y, scores)),
                                            balanced_accuracy=float(balanced_accuracy_score(y, scores >= threshold)),
                                            real_false_positive_rate=float(np.mean(scores[y == 0] >= threshold)),
                                            threshold=threshold))
                print(f'Fitted {key}; threshold={threshold:.4f}', flush=True)
    dump_csv(OUT / 'metrics.csv', metrics)
    dump_csv(OUT / 'predictions.csv', predictions)
    dump_csv(OUT / 'validation_predictions.csv', validation_predictions)
    import joblib
    joblib.dump(heads, ROOT / 'artifacts/round2_heads.joblib')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    original = read_manifest(ROOT / 'results/manifest.jsonl')
    protocol = {
        'id': 'round2_preprocessing_generator_ablation', 'seed': SEED,
        'established_before_run': datetime.now(timezone.utc).isoformat(),
        'research_question': 'How do common preprocessing and fixed-budget generator mixing change development AUROC?',
        'profiles': {'original': 'Pilot RGB fit224 bicubic',
                     'common128': 'RGB fit128 bicubic then resize224 bicubic',
                     'common128_jpeg90': 'RGB fit128 bicubic, JPEG90 subsampling2 decode, resize224 bicubic'},
        'training': {'single_sd15': '300 Real + 300 SD15', 'mixed_sd15_biggan': '300 same Real + 150 SD15 + 150 BigGAN'},
        'validation': {'single_sd15': '100 Real + 100 SD15', 'mixed_sd15_biggan': '100 same Real + 50 SD15 + 50 BigGAN'},
        'threshold_rule': 'Maximum Youden J on corresponding clean validation only; then fixed for all development conditions',
        'evaluation': 'Original 400 development images: 100 Real shared across 100 each SD15, BigGAN, ADM',
        'conditions_after_profile': CONDITIONS, 'heads': ['frozen CLIP logistic regression', 'three quality proxies logistic regression'],
        'hyperparameters': 'Inherited pilot scaler and C=1 liblinear; no development tuning; no augmentation',
        'contrasts': 'Paired AUROC differences for preprocessing within training and mixed-minus-single within profile',
        'uncertainty': '2000 paired class-stratified bootstrap draws; per-cell 95% percentile intervals; no multiple-comparison correction',
        'limits': ['Exploratory development, including ADM already inspected in pilot; no blind final test',
                   'Common processing does not remove inherited compression, semantic or source confounding',
                   'Single training subset seed; bootstrap conditions on fitted heads',
                   'BigGAN is seen for mixed training; its gain is not unseen-generator generalization',
                   'Mixed training changes both fake composition and corresponding validation composition',
                   'Shared real pool makes cells dependent; no population-level or causal conclusion'],
        'original_manifest_sha256': data.sha256(ROOT / 'results/manifest.jsonl')}
    # A protocol file is written before features, heads or new development metrics exist.
    if not (OUT / 'protocol.json').exists():
        write_json(OUT / 'protocol.json', protocol)
    else:
        old = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
        assert {k: v for k, v in old.items() if k != 'established_before_run'} == {
            k: v for k, v in protocol.items() if k != 'established_before_run'}, 'Protocol changed: use a new output directory'
    records = read_manifest(OUT / 'manifest.jsonl') if (OUT / 'manifest.jsonl').exists() else prepare(original)
    validate_records(records, original)
    groups = assignments(records)
    features, cached = extract(records, args.batch_size)
    run(records, groups, features)
    write_json(OUT / 'run_metadata.json', {
        'status': 'completed', 'seed': SEED, 'feature_cache_reused': cached,
        'run_seconds': time.perf_counter() - start, 'extraction_seconds': float(features['extraction_seconds']),
        'peak_cuda_memory_mb': float(features['peak_cuda_memory_mb']),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'python': platform.python_version(), 'torch': torch.__version__,
        'model': pilot.MODEL, 'model_revision': pilot.MODEL_REVISION, 'model_sha256': pilot.MODEL_SHA256,
        'manifest_sha256': data.sha256(OUT / 'manifest.jsonl'),
        'protocol_sha256': data.sha256(OUT / 'protocol.json'),
        'script_sha256': data.sha256(data.Path(__file__)),
        'pilot_script_sha256': data.sha256(ROOT / 'scripts/run_pilot.py'),
        'data_script_sha256': data.sha256(ROOT / 'scripts/prepare_data.py')})
    print('Round two completed. Run scripts/verify_round2.py for checks and paired intervals.', flush=True)


if __name__ == '__main__':
    main()
