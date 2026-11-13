"""Read-only identity/cache preflight; no image decode, extraction, or fitting."""
import sys
sys.dont_write_bytecode = True
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/analysis/recompression_reliability_20261108/preflight.json'
OLD = ROOT / 'output/analysis/multigen_gate_20261108/run_v1_codefix1'
SEEDS = [20261101, 20261102, 20261103]
sys.path.insert(0, str(ROOT / 'scripts'))
from round7_baseline_common import Guard

HASHES = {}


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    path = Path(path)
    rel = path.relative_to(ROOT).as_posix()
    assert 'final' not in rel.lower() and 'reserved' not in rel.lower(), rel
    HASHES[rel] = sha(path)
    return json.loads(path.read_text(encoding='utf-8'))


def manifest(path):
    HASHES[path.relative_to(ROOT).as_posix()] = sha(path)
    return {r['source_id']: r for line in path.read_text().splitlines() if (r := json.loads(line))}


def counts(ids, records):
    return dict(sorted(Counter(records[i]['generator'] for i in ids).items()))


def main():
    fitting = manifest(ROOT / 'results/round7/fitting_pool_manifest.jsonl')
    dev = manifest(ROOT / 'results/round7/development_manifest.jsonl')
    records = dict(fitting, **dev)
    isolation = read(ROOT / 'results/round7/step_b_isolation.json')
    deny = set(isolation['reserved_test_source_ids'])
    assert set(records) == set(isolation['allowed_source_ids']) and set(records).isdisjoint(deny)
    old_complete = read(OLD / 'complete.json')
    assert old_complete['status'] == 'completed'
    assert sha(OLD / 'splits.json') == old_complete['outputs_sha256']['splits.json']
    old_splits = read(OLD / 'splits.json')
    old_freeze = read(OLD / 'freeze.json')
    assert sha(OLD / 'splits.json') == old_freeze['split_sha256']
    report, subsets, head_info, cache_info = [], {}, [], []
    required = set(dev)
    for number, seed in enumerate(SEEDS, 1):
        assignment = read(ROOT / f'results/round7/assignments_seed{number}.json')
        assert assignment['seed'] == seed
        source = old_splits['sources'][f'{seed}/source_only']
        fit, tune = set(source['fit_ids']), set(source['tuning_ids'])
        assert fit == set(assignment['partitions']['gate_fit'])
        assert tune <= set(assignment['partitions']['selection'])
        assert counts(fit, records) == {'Real': 250, 'SD15': 250}
        assert counts(tune, records) == {'Real': 124, 'SD15': 124}
        assert fit.isdisjoint(tune) and (fit | tune).isdisjoint(dev)
        assert fit.isdisjoint(assignment['partitions']['expert_fit'])
        assert tune.isdisjoint(assignment['partitions']['expert_fit'])
        for field in ['source_bytes_sha256', 'pixel_sha256']:
            assert {records[i][field] for i in fit | tune}.isdisjoint({r[field] for r in dev.values()})
        subsets[seed] = {'fit': fit, 'tune': tune}
        required |= fit | tune
        complete = read(ROOT / f'results/round8_a/seed{number}/matched/complete.json')
        stem = f'artifacts/round8_a/seed{number}/matched'
        for file in ['bundle.json', 'experts.joblib']:
            rel = stem + '/' + file
            assert sha(ROOT / rel) == complete['files_sha256'][rel]
            HASHES[rel] = complete['files_sha256'][rel]
        bundle = read(ROOT / stem / 'bundle.json')
        assert bundle['run_seed'] == seed and bundle['signature'] == complete['signature']
        assert bundle['experts_sha256'] == complete['files_sha256'][stem + '/experts.joblib']
        head_info.append({'run_seed': seed, 'path': stem + '/experts.joblib',
                          'sha256': bundle['experts_sha256'], 'loaded_or_executed': False,
                          'expected_keys_from_verified_producer': ['global_head', 'local_head'],
                          'expected_input_dimensions': {'global_head': 512, 'local_head': 512},
                          'local_input': 'mean of four normalized PatchShuffle view embeddings, before frozen StandardScaler + LogisticRegression'})
        report.append({'run_seed': seed, 'fit_originals': len(fit), 'fit_counts': counts(fit, records),
                       'tune_originals': len(tune), 'tune_counts': counts(tune, records),
                       'development_originals': len(dev), 'development_counts': counts(dev, records),
                       'within_seed_fit_tune_development_and_expert_fit_isolation': 'passed'})
        # Metadata/schema and internal digest check on one existing original-view batch per partition.
        for part in ['gate_fit', 'selection', 'development']:
            path = ROOT / f'artifacts/round8_a/seed{number}/features/{part}_clean_0/00000.npz'
            relative = path.relative_to(ROOT).as_posix()
            expected_hash = old_freeze['input_sha256'].get(relative, bundle['feature_sha256'].get(relative))
            assert expected_hash is not None and sha(path) == expected_hash, relative
            HASHES[relative] = expected_hash
            with np.load(path, allow_pickle=False) as c:
                ids = c['ids'].tolist()
                assert ids == sorted(ids)
                identity = json.loads(str(c['identity']))
                assert identity == [seed, part, 'clean', False, ids]
                assert str(c['signature']) == bundle['signature']
                for field in ['g', 'l', 'q']:
                    assert hashlib.sha256(c[field].tobytes()).hexdigest() == str(c[field + '_sha256'])
                cache_info.append({'run_seed': seed, 'partition': part, 'sample_path': path.relative_to(ROOT).as_posix(),
                                  'sample_sha256': sha(path), 'keys': sorted(c.files),
                                  'g_shape': list(c['g'].shape), 'l_shape': list(c['l'].shape), 'q_shape': list(c['q'].shape),
                                  'g_dtype': str(c['g'].dtype), 'l_dtype': str(c['l'].dtype), 'q_dtype': str(c['q'].dtype),
                                  'identity_semantics': '[run_seed, partition, condition, format_matched, ordered_original_ids]',
                                  'original_view': True, 'internal_digests_verified': True})
    assert len(dev) == 900 and counts(dev, records) == {'ADM': 300, 'BigGAN': 300, 'Real': 300}
    guard = Guard({i: records[i] for i in required}, deny)
    guard.touch(sorted(required), 'original_cache_preflight')
    cache_locations = [ROOT / 'artifacts/round8_a/allowed_images']
    cache_locations += [ROOT / f'artifacts/round7/baselines_seed{i}/allowed_images' for i in (1, 2, 3)]
    entries, missing, location_counts = [], [], Counter()
    for identity in sorted(required):
        filename = identity.replace(':', '_') + '.bin'
        path = next((folder / filename for folder in cache_locations if (folder / filename).is_file()), None)
        if path is None:
            missing.append(identity)
            continue
        # Byte hashing only; never decode pixels or call an image/model transform.
        digest = sha(path)
        assert digest == records[identity]['source_bytes_sha256'], f'Original bytes changed: {identity}'
        folder_name = path.parent.relative_to(ROOT).as_posix()
        location_counts[folder_name] += 1
        entries.append({'source_id': identity, 'generator': records[identity]['generator'],
                        'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size,
                        'source_bytes_sha256': digest, 'expected_decoded_pixel_sha256': records[identity]['pixel_sha256']})
    cross = []
    for a, b in itertools.combinations(SEEDS, 2):
        cross.append({'seed_a': a, 'seed_b': b,
                      'fit_fit_overlap': len(subsets[a]['fit'] & subsets[b]['fit']),
                      'tune_tune_overlap': len(subsets[a]['tune'] & subsets[b]['tune']),
                      'a_fit_b_tune_overlap': len(subsets[a]['fit'] & subsets[b]['tune']),
                      'a_tune_b_fit_overlap': len(subsets[a]['tune'] & subsets[b]['fit'])})
    union_fit = set().union(*(v['fit'] for v in subsets.values()))
    union_tune = set().union(*(v['tune'] for v in subsets.values()))
    extract_ids = union_tune | set(dev)
    result = {'status': 'passed' if not missing else 'missing_originals', 'scope': 'Only old source-only fitting/tuning IDs, 900 allowed development IDs, original cache bytes and existing feature metadata. No decode, extraction, head loading, training or final-test access.',
              'input_sha256': HASHES, 'script_sha256': sha(__file__), 'per_seed': report,
              'unique_required_originals': len(required), 'unique_required_counts': counts(required, records),
              'unique_source_fit_originals': len(union_fit), 'unique_source_tune_originals': len(union_tune),
              'union_fit_union_tune_overlap_across_different_seeds': len(union_fit & union_tune),
              'extract_scope': {'unique_originals': len(extract_ids), 'source_tuning_unique_originals': len(union_tune),
                                'development_originals': len(dev), 'source_tuning_base_views': len(union_tune) * 3,
                                'development_base_views': len(dev) * 4,
                                'total_base_views': len(union_tune) * 3 + len(dev) * 4,
                                'source_tuning_conditions': ['clean', 'jpeg70', 'resize05'],
                                'development_conditions': ['clean', 'jpeg70', 'jpeg50', 'resize05'],
                                'source_fit': 'Existing original-view logits only for per-seed std; no new source-fit extraction unless the same ID is independently a tuning ID in another seed.',
                                'additional_recompression_stages': 'Defined by separate frozen protocol; view count above is per stage.'},
              'shared_development_originals': len(dev), 'cross_seed_overlap': cross,
              'original_cache': {'lookup_precedence': [p.relative_to(ROOT).as_posix() for p in cache_locations],
                                 'found': len(entries), 'missing_ids': missing, 'bytes': sum(e['bytes'] for e in entries),
                                 'chosen_location_counts': dict(location_counts), 'sha256_verified': len(entries),
                                 'pixels_decoded': 0, 'entries': entries},
              'denylist': {'source': 'results/round7/step_b_isolation.json (ID-only isolation metadata)',
                           'deny_count': len(deny), 'required_ids_overlap': len(required & deny),
                           'guard_violations': guard.violations, 'final_or_reserved_manifest_opened': False},
              'frozen_heads': head_info, 'existing_cache_schema_samples': cache_info,
              'independence_limit': 'Three source partitions share the same 2500-original fitting pool, overlap across seeds (including fit/tune roles), and reuse all 900 development originals. Independent only within each fitted model partition, not three independent datasets.',
              'future_verification_requirements': [
                  'New cache must identify exact ordered source IDs and source-byte hashes, allowed scope, base profile, sequential JPEG stage/quality/subsampling and encode/decode/resize order, pipeline/version/backbone hash and precision.',
                  'Explicit stage0 baseline and chain semantics: stage k generated from stage k-1, not k independent copies; do not silently mix native-size and processed-224 recompression.',
                  'Global embeddings shape N,512; local embeddings N,4,512 with fixed existing view permutations. Average local views before unchanged per-seed local head.',
                  'Join every stage and every seed projection by source_id; assert unique IDs, full stage coverage, finite arrays and cached-array digests.',
                  'Verify frozen matched expert files against their completion hashes; never refit scalers or heads using recompressed or development images.',
                  'Declare numerical tolerance for stage0 reproduction before reading reliability outcomes; preserve raw score changes separately from any standardized diagnostic feature.',
                  'No new diagnostic models in this task. Source-fit old logits determine std only; fixed stability/confidence signals and label-stratified relative-BCE-winner endpoint must remain protocol-locked.',
                  'Extraction guard may permit the union required IDs, but statistical analysis must still enforce each seed-specific role; union fit/tune roles overlap and cannot be pooled as independent partitions.',
              ]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    brief = {k: result[k] for k in ['status', 'per_seed', 'unique_required_originals', 'unique_required_counts',
                                   'unique_source_fit_originals', 'unique_source_tune_originals',
                                   'union_fit_union_tune_overlap_across_different_seeds', 'extract_scope', 'cross_seed_overlap', 'denylist']}
    brief['original_cache'] = {k: v for k, v in result['original_cache'].items() if k != 'entries'}
    print(json.dumps(brief, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
