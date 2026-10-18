"""Prepare a small, deduplicated sample from pinned Tiny-GenImage shards."""
import concurrent.futures as futures
import hashlib
import io
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATASET = 'TheKernel01/Tiny-GenImage'
REVISION = '89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69'
SEED = 20261018
GENERATORS = {0: 'Real', 1: 'ADM', 2: 'BigGAN', 6: 'SD15'}
SHARDS = [0, 4, 8, 12]


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def download_shard(info):
    destination = ROOT / 'data' / 'shards' / Path(info['path']).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        url = f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{info["path"]}'
        print(f'Downloading {destination.name} ({info["size"]/1e6:.0f} MB)', flush=True)
        partial = destination.with_suffix('.ranged.part')
        with partial.open('wb') as handle:
            handle.truncate(info['size'])
        def fetch(start):
            end = min(start+16*1024*1024, info['size'])-1
            response = requests.get(url, headers={'Range':f'bytes={start}-{end}'}, timeout=(30,60))
            response.raise_for_status()
            assert response.status_code==206 and len(response.content)==end-start+1
            assert response.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
            with partial.open('r+b') as handle:
                handle.seek(start)
                handle.write(response.content)
        with futures.ThreadPoolExecutor(max_workers=3) as executor:
            jobs = [executor.submit(fetch,start) for start in range(0,info['size'],16*1024*1024)]
            for i, job in enumerate(futures.as_completed(jobs),1):
                job.result()
                if i%10==0:
                    print(f'{destination.name}: {i}/{len(jobs)} blocks',flush=True)
        assert sha256(partial)==info['lfs']['oid'], f'Checksum mismatch: {destination}'
        partial.replace(destination)
    actual = sha256(destination)
    assert actual == info['lfs']['oid'], f'Checksum mismatch: {destination}'
    print(f'Verified {destination.name}', flush=True)
    return destination


def extract_record(row, source_row):
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(row['image']['bytes']))).convert('RGB')
    digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    thumb = ImageOps.fit(image, (224, 224), method=Image.Resampling.BICUBIC)
    crop_hash = hashlib.sha256(thumb.tobytes()).hexdigest()
    small = np.asarray(image.convert('L').resize((9, 8), Image.Resampling.LANCZOS))
    dhash = sum(int(b) << i for i, b in enumerate((small[:, 1:] > small[:, :-1]).ravel()))
    record = {'source_id': f'train:{source_row}', 'source_row': source_row,
              'generator': GENERATORS[row['generator']], 'label': row['label'],
              'width': image.width, 'height': image.height,
              'pixel_sha256': digest, 'crop_sha256': crop_hash, 'dhash': f'{dhash:016x}',
              'source_format': Image.open(io.BytesIO(row['image']['bytes'])).format,
              'source_bytes': len(row['image']['bytes'])}
    return record, image


def main():
    start = time.perf_counter()
    response = requests.get(f'https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}/data',
                            params={'limit': 30}, timeout=60)
    response.raise_for_status()
    wanted = {f'data/train-{n:05d}-of-00014.parquet' for n in SHARDS}
    infos = sorted([item for item in response.json() if item['path'] in wanted], key=lambda r:r['path'])
    assert len(infos) == len(SHARDS)
    with futures.ThreadPoolExecutor(max_workers=4) as executor:
        paths = list(executor.map(download_shard, infos))
    candidates, hashes, crops, perceptual, removed = [], set(), set(), [], Counter()
    for shard, path in zip(SHARDS, paths):
        table = pq.read_table(path)
        assert len(table) == 2000, 'Update global row indexing if shard sizes change.'
        rows = table.to_pylist()
        order = list(range(len(rows)))
        random.Random(SEED+shard).shuffle(order)
        for index in order:
            row = rows[index]
            if row['generator'] not in GENERATORS:
                continue
            record, image = extract_record(row, shard*2000+index)
            h = int(record['dhash'], 16)
            if record['pixel_sha256'] in hashes or record['crop_sha256'] in crops:
                removed['exact_or_crop'] += 1
                continue
            if any((h ^ old).bit_count() <= 3 for old in perceptual):
                removed['dhash_distance_le_3'] += 1
                continue
            hashes.add(record['pixel_sha256'])
            crops.add(record['crop_sha256'])
            perceptual.append(h)
            destination = ROOT/'data'/'images'/f'train_{record["source_row"]}.png'
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.save(destination)
            record['path'] = destination.relative_to(ROOT).as_posix()
            record['file_sha256'] = sha256(destination)
            candidates.append(record)
        del rows, table
        print(f'{path.name}: unique candidates {dict(Counter(r["generator"] for r in candidates))}', flush=True)
    random.Random(SEED).shuffle(candidates)
    selected = []
    for gen in GENERATORS.values():
        rows = [r for r in candidates if r['generator'] == gen]
        targets = [('train', 300), ('validation', 100), ('development', 100)] if gen in ('Real', 'SD15') else [('development', 100)]
        required = sum(n for _, n in targets)
        if len(rows) < required:
            raise RuntimeError(f'{gen}: need {required} unique images, got {len(rows)}')
        offset = 0
        for split, count in targets:
            for row in rows[offset:offset+count]:
                row['split'] = split
                selected.append(row)
            offset += count
    result = ROOT/'results'
    result.mkdir(exist_ok=True)
    with (result/'manifest.jsonl').open('w', encoding='utf-8') as handle:
        for row in selected:
            handle.write(json.dumps(row)+'\n')
    metadata = {'dataset': DATASET, 'revision': REVISION, 'source_split': 'train',
                'seed': SEED, 'shards': SHARDS, 'source_shards':infos,
                'selected': len(selected), 'unique_candidates': len(candidates),
                'removed_duplicates': dict(removed),
                'counts': dict(Counter(f"{r['split']}/{r['generator']}" for r in selected)),
                'seconds':time.perf_counter()-start,
                'sampling':'Fixed four spread-out shards; random candidate ordering and split assignment; not a representative global benchmark.',
                'note':'Third-party Tiny-GenImage; generator labels inherited, not independently authenticated. All evaluation is exploratory development, not final blind testing.'}
    (result/'data_preparation.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in metadata.items() if k!='source_shards'},indent=2),flush=True)


if __name__=='__main__':
    main()
