"""One frozen CPU-only exploratory ablation: original gate-fit/tuning views.

Existing matched experts stay fixed. All writes remain beside this script.
No reserved/final-test files, images, CLIP extraction or GPU operations.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
sys.dont_write_bytecode = True
import csv
import hashlib
import json
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import joblib
import numpy as np
import torch
from scipy.special import expit
import gate_models as gates
import round7_baseline_clip as clip
import round7_baseline_common as common

RUN_SEEDS = [20261101, 20261102, 20261103]
TUNE = ['clean', 'jpeg70', 'resize05']
CONDITIONS = ['clean', 'jpeg70', 'jpeg50', 'resize05']
GENERATORS = ['ADM', 'BigGAN']
METRICS = ['auroc', 'average_precision', 'balanced_accuracy', 'real_false_positive_rate']
INPUT_HASHES = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_source(relative, expected=None):
    assert 'final' not in relative.lower() and 'reserved' not in relative.lower(), relative
    path = ROOT / relative
    actual = sha(path)
    assert expected is None or actual == expected, f'Changed source: {relative}'
    INPUT_HASHES[relative] = actual
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(name, data):
    with (OUT / name).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def write_csv(name, rows):
    assert rows
    with (OUT / name).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def array_batch(raw):
    return {k: v if k == 'rows' else np.asarray(v) for k, v in raw.items()}


def verify_and_collect():
    """Inspect metadata/hashes and row identity only; no development scoring."""
    sources = {}
    for number, run_seed in enumerate(RUN_SEEDS, 1):
        arms = {}
        for arm in ['matched', 'unmatched']:
            marker = read_source(f'results/round8_a/seed{number}/{arm}/complete.json')
            assert marker['status'] == 'completed'
            stem = f'artifacts/round8_a/seed{number}/{arm}'
            bundle = read_source(f'{stem}/bundle.json', marker['files_sha256'][f'{stem}/bundle.json'])
            expert_path = f'{stem}/experts.joblib'
            expert_sha = sha(ROOT / expert_path)
            assert expert_sha == marker['files_sha256'][expert_path] == bundle['experts_sha256']
            INPUT_HASHES[expert_path] = expert_sha
            assert bundle['signature'] == marker['signature']
            assert bundle['run_seed'] == run_seed and bundle['arm'] == arm
            arms[arm] = bundle
        cache_parts = {}
        for part, original_name in [('gate_fit', 'gate_fit'), ('selection', 'tuning')]:
            expected_rows = arms['matched']['raw_batches'][original_name]['rows']
            part_ids = [r['source_id'] for r in expected_rows if r['condition'] == 'clean']
            assert part_ids == sorted(part_ids) and len(part_ids) == len(set(part_ids))
            cache_parts[part] = {}
            for condition in TUNE:
                folder = ROOT / f'artifacts/round8_a/seed{number}/features/{part}_{condition}_0'
                paths = sorted(folder.glob('*.npz'))
                assert paths, str(folder)
                all_ids = []
                for path in paths:
                    relative = path.relative_to(ROOT).as_posix()
                    expected = arms['unmatched']['feature_sha256'][relative]
                    assert sha(path) == expected, f'Feature changed: {relative}'
                    INPUT_HASHES[relative] = expected
                    with np.load(path, allow_pickle=False) as cache:
                        ids = cache['ids'].tolist()
                        identity = json.loads(str(cache['identity']))
                        assert identity == [run_seed, part, condition, False, ids]
                        assert str(cache['signature']) == arms['matched']['signature']
                        for name in ['g', 'l', 'q']:
                            assert hashlib.sha256(cache[name].tobytes()).hexdigest() == str(cache[name + '_sha256'])
                        all_ids.extend(ids)
                assert all_ids == part_ids, (number, part, condition)
                cache_parts[part][condition] = [p.relative_to(ROOT).as_posix() for p in paths]
            ids_by_condition = [(r['condition'], r['source_id']) for r in expected_rows]
            assert ids_by_condition == [(cond, i) for cond in TUNE for i in part_ids]
        sources[number] = {'run_seed': run_seed, 'matched': arms['matched'], 'paths': cache_parts}
    for name in ['gate_models.py', 'round7_baseline_clip.py', 'round7_baseline_common.py',
                 'round7_baseline_data.py', 'score_calibration.py']:
        INPUT_HASHES[f'scripts/{name}'] = sha(ROOT / 'scripts' / name)
    return sources


def reconstruct_original(source, number, part, experts):
    raw_name = 'gate_fit' if part == 'gate_fit' else 'tuning'
    template = source['matched']['raw_batches'][raw_name]
    rows, zg, zl, variance, quality = [], [], [], [], []
    for condition in TUNE:
        blocks = {key: [] for key in ['g', 'l', 'q']}
        ids = []
        for relative in source['paths'][part][condition]:
            assert sha(ROOT / relative) == INPUT_HASHES[relative]
            with np.load(ROOT / relative, allow_pickle=False) as cache:
                ids.extend(cache['ids'].tolist())
                for key in blocks:
                    blocks[key].append(cache[key])
        features = {key: np.concatenate(value) for key, value in blocks.items()}
        condition_rows = [r for r in template['rows'] if r['condition'] == condition]
        assert [r['source_id'] for r in condition_rows] == ids
        gh, lh = experts['global_head'], experts['local_head']
        zzg = gh.decision_function(features['g'])
        zzl = lh.decision_function(features['l'].mean(axis=1))
        vv = lh.decision_function(features['l'].reshape(-1, features['l'].shape[-1])).reshape(-1, 4).var(axis=1)
        rows.extend(condition_rows)
        zg.extend(zzg); zl.extend(zzl); variance.extend(vv); quality.extend(features['q'])
    batch = {'rows': rows, 'zg': np.asarray(zg), 'zl': np.asarray(zl),
             'variance': np.asarray(variance), 'q': np.asarray(quality),
             'y': np.asarray([r['label'] for r in rows], dtype=float)}
    assert all(np.isfinite(batch[k]).all() for k in ['zg', 'zl', 'variance', 'q', 'y'])
    if part == 'selection':
        # The existing matched reporting batch already uses these original views.
        reference = array_batch(source['matched']['raw_batches']['selection'])
        mask = np.array([r['condition'] in TUNE for r in reference['rows']])
        assert batch['rows'] == [r for r, keep in zip(reference['rows'], mask) if keep]
        for key in ['zg', 'zl', 'variance', 'q', 'y']:
            np.testing.assert_allclose(batch[key], reference[key][mask], rtol=0, atol=1e-10)
    return clip.transform(batch, 1., 1.)


def saved_alpha(settings, batch, model, gate_seed):
    if model in ['average', 'fixed']:
        weight = .5 if model == 'average' else settings['settings'][f'fixed/{gate_seed}']['fixed_alpha']
        return np.full(len(batch['y']), weight)
    artifact = settings['gates'][f'{model}/{gate_seed}']['artifact']
    x = ((batch['x'] - artifact['mean']) / artifact['scale']).astype(np.float32)
    if model == 'score_gate':
        x[:, 5:] = 0
    state = {k: np.asarray(v, dtype=np.float32) for k, v in artifact['state'].items()}
    h = np.maximum(x @ state['hidden.weight'].T + state['hidden.bias'], 0)
    return expit(h @ state['output.weight'].T + state['output.bias']).ravel().astype(float)


def score_cells(run_seed, scenario, model, gate_seed, batch, weight, cutoff):
    z = weight * batch['zg'] + (1 - weight) * batch['zl']
    result = []
    for condition in CONDITIONS:
        for generator in GENERATORS:
            mask = np.array([r['condition'] == condition and r['generator'] in ['Real', generator]
                             for r in batch['rows']])
            y = batch['y'][mask]
            assert set(np.unique(y)) == {0, 1}
            result.append({'run_seed': run_seed, 'scenario': scenario, 'model': model,
                           'gate_seed': gate_seed, 'condition': condition, 'generator': generator,
                           'n_real': int(sum(y == 0)), 'n_fake': int(sum(y == 1)),
                           'threshold': cutoff, 'alpha_mean': float(np.mean(weight[mask])),
                           **common.points(y, z[mask], cutoff)})
    return result


def aggregate(instance_metrics):
    grouped = defaultdict(list)
    for row in instance_metrics:
        key = tuple(row[k] for k in ['run_seed', 'scenario', 'model', 'condition', 'generator'])
        grouped[key].append(row)
    by_seed = []
    for key, rows in sorted(grouped.items()):
        run_seed, scenario, model, condition, generator = key
        by_seed.append(dict(run_seed=run_seed, scenario=scenario, model=model,
                            condition=condition, generator=generator,
                            **{m: float(np.mean([r[m] for r in rows])) for m in METRICS}))
    lookup = {(r['run_seed'], r['scenario'], r['model'], r['condition'], r['generator']): r for r in by_seed}
    contrasts = [
        ('new_quality_minus_new_average', 'original_gate_views', 'quality_gate', 'original_gate_views', 'average'),
        ('new_quality_minus_new_fixed', 'original_gate_views', 'quality_gate', 'original_gate_views', 'fixed'),
        ('new_quality_minus_new_score', 'original_gate_views', 'quality_gate', 'original_gate_views', 'score_gate'),
        ('new_quality_minus_published_quality', 'original_gate_views', 'quality_gate', 'published_matched', 'quality_gate'),
        ('new_score_minus_published_score', 'original_gate_views', 'score_gate', 'published_matched', 'score_gate'),
    ]
    differences, cells = [], []
    for generator in GENERATORS:
        for condition in CONDITIONS:
            cell = {'generator': generator, 'condition': condition, 'auroc': {}, 'contrasts': {}}
            for scenario in ['published_matched', 'original_gate_views']:
                for model in ['average', 'fixed', 'score_gate', 'quality_gate']:
                    values = [lookup[(seed, scenario, model, condition, generator)]['auroc'] for seed in RUN_SEEDS]
                    cell['auroc'][f'{scenario}/{model}'] = dict(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)), seeds=values)
            for name, sa, ma, sb, mb in contrasts:
                values = []
                for seed in RUN_SEEDS:
                    aa, bb = lookup[(seed, sa, ma, condition, generator)], lookup[(seed, sb, mb, condition, generator)]
                    difference = {m: aa[m] - bb[m] for m in METRICS}
                    differences.append(dict(contrast=name, run_seed=seed, generator=generator, condition=condition, **difference))
                    values.append(difference['auroc'])
                cell['contrasts'][name] = dict(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)), seeds=values)
            cells.append(cell)
    macro = {}
    for scenario in ['published_matched', 'original_gate_views']:
        for model in ['average', 'fixed', 'score_gate', 'quality_gate']:
            rows = [r for r in by_seed if r['scenario'] == scenario and r['model'] == model]
            macro[f'{scenario}/{model}'] = {m: float(np.mean([r[m] for r in rows])) for m in METRICS}
    return by_seed, differences, cells, macro


def main():
    assert not (OUT / 'protocol.json').exists(), 'A protocol already exists; do not overwrite or silently rerun.'
    sources = verify_and_collect()
    protocol = {
        'id': 'exploratory_original_gate_views_v1',
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'frozen before training and development scoring',
        'question': 'Do matched experts benefit when only gate-fit and tuning views remain original rather than native-JPEG-matched?',
        'changed': 'gate_fit and selection TUNE images/features only: matched to original views',
        'fixed': 'Three existing matched experts; raw logits; unchanged gate_models.fit architecture, paired seeds, BCE checkpoint selection and optimizer; same threshold and fixed-alpha rules.',
        'training': 'Only original gate_fit cache projections through matched frozen experts; 3 views equally weighted.',
        'selection': 'Only original selection clean/jpeg70/resize05; gate checkpoint min pooled BCE including epoch0; fixed alpha max mean per-condition AUROC; pooled Youden-J threshold.',
        'development': 'Existing matched bundle raw_batches development, ADM and BigGAN, all four conditions. All 8 cells reported, no adaptive tuning.',
        'exclusions': 'No final/reserved test access; no images, CLIP extraction, GPU, new expert fitting, model search, test-based changes, publishing or pushing.',
        'run_seeds': RUN_SEEDS, 'gate_seeds': gates.SEEDS, 'tuning_conditions': TUNE,
        'report_conditions': CONDITIONS, 'development_generators': GENERATORS,
        'primary_metric': 'Per-generator/condition AUROC; descriptive equal mean over all 8 development cells.',
        'secondary_metrics': METRICS[1:],
        'contrasts': ['new quality - new average', 'new quality - new fixed', 'new quality - new score',
                      'new quality - published quality', 'new score - published score'],
        'aggregation': 'Average three gate initializations within data seed before pairing; then mean and sample SD of 3 data seeds. No significance claims.',
        'limitations': 'Exploratory: ADM/BigGAN development have previously been inspected; shared fitting pool/dev images; no new independent holdout.',
        'script_sha256': sha(__file__), 'input_sha256': INPUT_HASHES,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__, 'device': 'cpu'},
    }
    write_json('protocol.json', protocol)
    print(f'Protocol frozen: {sha(OUT / "protocol.json")}', flush=True)
    fitted, new_settings, selection_checks = {}, {}, []
    for number, source in sources.items():
        seed = source['run_seed']
        path = ROOT / f'artifacts/round8_a/seed{number}/matched/experts.joblib'
        assert sha(path) == INPUT_HASHES[path.relative_to(ROOT).as_posix()]
        experts = joblib.load(path)
        train = reconstruct_original(source, number, 'gate_fit', experts)
        tune = reconstruct_original(source, number, 'selection', experts)
        fixed, search = clip.fixed_alpha(tune)
        models = []
        settings = {'fixed_alpha': fixed, 'fixed_search': search, 'instances': {}}
        for name, weight in [('average', .5), ('fixed', fixed)]:
            z = weight * tune['zg'] + (1 - weight) * tune['zl']
            cutoff = common.threshold(tune['y'], expit(z))
            models.append((name, -1, None, None, cutoff, weight))
            settings['instances'][f'{name}/-1'] = {'threshold': cutoff, 'alpha': weight}
        for gate_seed in gates.SEEDS:
            for name in ['score_gate', 'quality_gate']:
                model, scaler, artifact, history = gates.fit(train, tune, name, gate_seed)
                alpha = gates.predict(model, scaler, tune['x'], name)
                cutoff = common.threshold(tune['y'], expit(alpha * tune['zg'] + (1 - alpha) * tune['zl']))
                models.append((name, gate_seed, model, scaler, cutoff, None))
                settings['instances'][f'{name}/{gate_seed}'] = {'threshold': cutoff, 'artifact': artifact, 'history': history}
                selection_checks.append({'run_seed': seed, 'model': name, 'gate_seed': gate_seed,
                                         'best_epoch': artifact['best_epoch'], 'stopped_epoch': artifact['stopped_epoch'],
                                         'best_selection_bce': artifact['best_validation_bce'], 'threshold': cutoff})
                print(f'Fit {number}/{name}/{gate_seed}: epoch {artifact["best_epoch"]}', flush=True)
        fitted[number] = models
        new_settings[str(seed)] = settings
    # All three seeds and every method are fully trained/selected before dev scoring.
    write_json('model_settings.json', new_settings)
    write_csv('selection_checkpoints.csv', selection_checks)
    metrics = []
    instances = [('average', -1), ('fixed', -1)] + [(name, s) for s in gates.SEEDS for name in ['score_gate', 'quality_gate']]
    for number, source in sources.items():
        batch = clip.transform(array_batch(source['matched']['raw_batches']['development']), 1., 1.)
        for name, init, model, scaler, cutoff, constant in fitted[number]:
            weight = np.full(len(batch['y']), constant) if constant is not None else gates.predict(model, scaler, batch['x'], name)
            metrics.extend(score_cells(source['run_seed'], 'original_gate_views', name, init, batch, weight, cutoff))
        old = source['matched']['settings']
        for name, init in instances:
            weight = saved_alpha(old, batch, name, init)
            cutoff = old['settings'][f'{name}/{init}']['threshold']
            metrics.extend(score_cells(source['run_seed'], 'published_matched', name, init, batch, weight, cutoff))
    by_seed, differences, cells, macro = aggregate(metrics)
    write_csv('metrics_instances.csv', metrics)
    write_csv('metrics_seed_condition.csv', by_seed)
    write_csv('paired_seed_condition.csv', differences)
    for relative, expected in INPUT_HASHES.items():
        assert sha(ROOT / relative) == expected, f'Input changed during ablation: {relative}'
    result = {'status': 'completed', 'protocol_sha256': sha(OUT / 'protocol.json'),
              'read_only_inputs_unchanged': True, 'matched_experts_refitted': False,
              'final_test_accessed': False, 'gate_fits': 18, 'instance_metric_rows': len(metrics),
              'seed_condition_metric_rows': len(by_seed), 'macro_all_8_dev_cells': macro, 'cells': cells,
              'interpretation_limit': 'Exploratory development comparison; no independent validation or significance claim.'}
    write_json('summary.json', result)
    print(json.dumps({'status': 'completed', 'macro': macro}, indent=2), flush=True)


if __name__ == '__main__':
    main()
