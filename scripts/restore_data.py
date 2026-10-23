"""Restore the published sample from pinned shards without resampling or editing results."""
import argparse
from collections import defaultdict
import hashlib
import io
import json

import pyarrow.parquet as pq

import prepare_data as data


def restore(check_only=False):
    root = data.ROOT
    metadata = json.loads((root / 'results/data_preparation.json').read_text(encoding='utf-8'))
    original = [json.loads(line) for line in
                (root / 'results/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    records = [json.loads(line) for line in
               (root / 'results/round2/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    if metadata['dataset'] != data.DATASET or metadata['revision'] != data.REVISION:
        raise ValueError('Dataset identity differs from the pinned preparation script')
    if len(original) != 1200 or records[:len(original)] != original or len(records) != 1600:
        raise ValueError('Expected the published 1200-image pilot plus 400 BigGAN images')
    shards = {int(info['path'].split('train-')[1].split('-')[0]): info
              for info in metadata['source_shards']}
    missing = defaultdict(list)
    for record in records:
        target = (root / record['path']).resolve()
        if not target.is_relative_to((root / 'data').resolve()) or target.suffix != '.png':
            raise ValueError(f'Unexpected image destination: {record["path"]}')
        shard = record['source_row'] // 2000
        if shard not in shards or record['source_id'] != f'train:{record["source_row"]}':
            raise ValueError('Invalid source row')
        if target.exists():
            if data.sha256(target) != record['file_sha256']:
                raise ValueError(f'Existing image has a different hash; left untouched: {target}')
        else:
            missing[shard].append(record)
    count = sum(map(len, missing.values()))
    print(f'Images verified: {len(records) - count}; missing: {count}', flush=True)
    if check_only and count:
        raise FileNotFoundError('Run scripts/restore_data.py without --check-only to restore images')
    if check_only or not count:
        return {'verified': len(records), 'restored': 0}
    for shard, selected in sorted(missing.items()):
        path = data.download_shard(shards[shard])
        table = pq.read_table(path)
        if len(table) != 2000:
            raise ValueError('Pinned shard row count changed')
        for record in selected:
            row = table.slice(record['source_row'] % 2000, 1).to_pylist()[0]
            actual, image = data.extract_record(row, record['source_row'])
            if any(actual[key] != record[key] for key in actual):
                raise ValueError(f'Source metadata differs: {record["source_id"]}')
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            payload = buffer.getvalue()
            if hashlib.sha256(payload).hexdigest() != record['file_sha256']:
                raise ValueError('PNG encoding differs; use the pinned Pillow version')
            target = root / record['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as handle:
                handle.write(payload)
        print(f'Restored {len(selected)} images from {path.name}', flush=True)
    return {'verified': len(records), 'restored': count}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Check local images without downloading or writing')
    print(json.dumps(restore(parser.parse_args().check_only)))
