"""Hash-only fresh-test sampling. No feature extraction or model operations."""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter
import hashlib
import json
import round8_final_common as f
from round7_prepare_data import NearIndex

SAMPLE_SEED = 20261028
TARGETS = {g: 500 for g in (*f.GENERATORS, 'Real')}
KEYS = ('source_id', 'generator', 'label', 'pixel_sha256', 'dhash', 'source_bytes_sha256')


def inputs():
    f.environment()
    cache = f.ROOT / 'artifacts/round7/source_hashes_pillow12_2.jsonl'
    meta = f.ROOT / 'artifacts/round7/source_hashes_pillow12_2_meta.json'
    provenance = f.read(meta)
    inventory_path = f.ROOT / 'results/round7/source_inventory.json'
    inventory = f.read(inventory_path)
    f.require(provenance['pillow'] == '12.2.0' and provenance['sha256'] == f.sha(cache)
              and provenance['inventory_sha256'] == f.sha(inventory_path), 'Source hash inventory provenance failed')
    f.require(inventory['dataset'] == 'TheKernel01/Tiny-GenImage' and
              inventory['revision'] == '89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69', 'Wrong source revision')
    all_rows = f.rows(cache)
    f.require(len(all_rows) == 28000 and len({r['source_id'] for r in all_rows}) == 28000, 'Incomplete source hashes')
    lookup = {r['source_id']: r for r in all_rows}
    previous = {}
    for name in f.EXCLUSIONS:
        for row in f.rows(f.ROOT / name):
            original = lookup[row['source_id']]
            f.require(all(row[k] == original[k] for k in ('pixel_sha256', 'dhash', 'label', 'generator')), 'Historical/source hash mismatch')
            previous[row['source_id']] = original
    hashes = {name: f.sha(f.ROOT / name) for name in f.EXCLUSIONS}
    hashes.update({f.rel(p): f.sha(p) for p in (cache, meta, inventory_path)})
    return all_rows, previous, hashes


def sample():
    records, previous, hashes = inputs()
    old_pixels = {r['pixel_sha256'] for r in previous.values()}
    old_index = NearIndex()
    for r in previous.values():
        old_index.add(int(r['dhash'], 16), r['source_id'])
    eligible = []
    reasons = Counter()
    for r in records:
        if r['generator'] not in TARGETS:
            reasons['outside_requested_domains'] += 1
        elif r['source_id'] in previous:
            reasons['previous_source_id'] += 1
        elif r['pixel_sha256'] in old_pixels:
            reasons['previous_pixel'] += 1
        elif old_index.match(int(r['dhash'], 16)) is not None:
            reasons['previous_near_duplicate'] += 1
        else:
            eligible.append(r)
    # Global deterministic ordering, independent of label; greedy radius-three dedup.
    order = lambda r: (hashlib.sha256(f'{SAMPLE_SEED}|{r["source_id"]}|{r["pixel_sha256"]}'.encode()).hexdigest(), r['source_id'])
    near, pixels, accepted = NearIndex(), set(), []
    for r in sorted(eligible, key=order):
        if r['pixel_sha256'] in pixels:
            reasons['within_pixel'] += 1
        elif near.match(int(r['dhash'], 16)) is not None:
            reasons['within_near_duplicate'] += 1
        else:
            accepted.append(r)
            pixels.add(r['pixel_sha256'])
            near.add(int(r['dhash'], 16), r['source_id'])
    available = Counter(r['generator'] for r in accepted)
    shortage = {g: max(0, n - available[g]) for g, n in TARGETS.items()}
    f.require(not any(shortage.values()), f'Fresh data shortfall {shortage}; STOP. Protocol owner must approve changes; never refill with old images.')
    selected = []
    for generator, target in TARGETS.items():
        selected.extend([r for r in accepted if r['generator'] == generator][:target])
    selected.sort(key=lambda r: r['source_id'])
    audit = dict(seed=SAMPLE_SEED, targets=TARGETS, counts=dict(Counter(r['generator'] for r in selected)),
        eligible_after_dedup=dict(available), exclusions=dict(reasons), excluded_original_ids=len(previous),
        input_sha256=hashes, no_in_domain_test=True, total=len(selected),
        sampling='SHA256(seed|source_id|pixel_sha256) ascending, global greedy dedup, first target per generator',
        decoding='Reuses complete Pillow 12.2 hash inventory with verified provenance. No image decoding during sampling.',
        freshness='Never used in the supplied repository exclusion pools or their pixel/dHash<=3 neighbours. External pilot exposure cannot be certified.',
        limits=['No SD15 in-domain test.', 'Previously inspected generator domains; new repository images are not new domains.',
                'Real JPEG/fake PNG and native-size/compression-history confounds remain.'],
        shortfalls=shortage)
    return [{k: r[k] for k in KEYS} for r in selected], audit


def payload(records):
    return ''.join(json.dumps(r, sort_keys=True) + '\n' for r in records).encode()


def prepare():
    f.require(f.read(f.OUT / 'handoff_reproduction.json')['status'] == 'passed', 'Reproduction must pass first')
    f.require(not f.FREEZE.exists(), 'Data is frozen')
    f.require(not f.TEST.exists() and not (f.OUT / 'data_audit.json').exists(), 'Data outputs already exist; verify instead')
    records, audit = sample()
    f.publish(f.TEST, payload(records))
    audit['manifest_sha256'] = f.sha(f.TEST)
    f.write(f.OUT / 'data_audit.json', audit)
    print(json.dumps(audit['counts'], indent=2), flush=True)


def verify_data():
    expected, audit = sample()
    f.require(f.TEST.read_bytes() == payload(expected), 'Manifest differs from independent seeded replay')
    saved = f.read(f.OUT / 'data_audit.json')
    audit['manifest_sha256'] = f.sha(f.TEST)
    f.require(saved == audit, 'Audit changed')
    # Separate direct check of exclusions and all selected pair distances.
    _, previous, _ = inputs()
    for i, r in enumerate(expected):
        for other in [*previous.values(), *expected[:i]]:
            f.require(r['source_id'] != other['source_id'] and r['pixel_sha256'] != other['pixel_sha256']
                      and (int(r['dhash'], 16) ^ int(other['dhash'], 16)).bit_count() > 3, 'Duplicate/leakage')
    return {'status': 'passed', 'counts': saved['counts']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    with f.lock():
        print(verify_data() if args.verify else prepare())
