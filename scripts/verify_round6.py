"""Matched-dose ablation verification, control equivalence and paired augmentation effects."""
from collections import Counter, defaultdict
import csv
import json

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import gate_models as gates
import prepare_data as data
import run_pilot as pilot
from run_round2 import dump_csv, read_manifest, write_json
from run_round6 import OUT, ROOT, PROFILES, TRAIN_CONDITIONS, DEV_CONDITIONS, EXPERT_MODES, CALIBRATIONS, EXPERT_CONDITIONS
from score_calibration import MIN_T, MAX_T
from verify_round2 import independent_threshold
from verify_round4 import numpy_gate


def read_csv(path):
    with path.open(encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def arrays(rows, tg, tl):
    zg = np.array([float(r['global_logit']) for r in rows]) / tg
    zl = np.array([float(r['local_logit']) for r in rows]) / tl
    variance = np.array([float(r['variance']) for r in rows]) / tl ** 2
    q = np.array([[float(r[k]) for k in ['q_laplacian', 'q_high_frequency', 'q_block']] for r in rows])
    pg, pl = expit(zg), expit(zl)
    def entropy(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -p * np.log(p) - (1 - p) * np.log1p(-p)
    x = np.column_stack([pg, pl, abs(pg - pl), (entropy(pg) + entropy(pl)) / 2, np.log1p(variance), q])
    return x, zg, zl, np.array([int(r['label']) for r in rows])


def nll(z, y):
    return float(np.mean(np.logaddexp(0, z) - y * z))


def main():
    meta = json.loads((OUT / 'run_metadata.json').read_text())
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assignments = json.loads((OUT / 'assignments.json').read_text())
    settings = json.loads((OUT / 'settings.json').read_text())
    temperatures = json.loads((OUT / 'temperatures.json').read_text())
    models = json.loads((OUT / 'gate_models.json').read_text())
    inputs = read_csv(OUT / 'branch_inputs.csv')
    dev = read_csv(OUT / 'predictions.csv')
    val = read_csv(OUT / 'validation_predictions.csv')
    metrics = read_csv(OUT / 'metrics.csv')
    curves = read_csv(OUT / 'training_curves.csv')
    search = read_csv(OUT / 'weight_search.csv')
    records = read_manifest(ROOT / 'results/round2/manifest.jsonl')
    lookup = {r['source_id']: i for i, r in enumerate(records)}
    assert protocol['manifest_sha256'] == data.sha256(ROOT / 'results/round2/manifest.jsonl')
    for field, name in [('protocol_sha256', 'protocol.json'), ('assignments_sha256', 'assignments.json')]:
        assert meta[field] == data.sha256(OUT / name)
    for name, checksum in meta['code_sha256'].items():
        assert checksum == data.sha256(ROOT / 'scripts' / name)
    for name, checksum in meta['feature_sha256'].items():
        assert checksum == data.sha256(ROOT / name)
    old = json.loads((ROOT / 'results/round4/assignments.json').read_text())
    for part in ['expert_train', 'gate_train', 'development']:
        assert assignments[part] == old[part]
    assert set(assignments['calibration'] + assignments['validation']) == set(old['validation'])
    sets = list(map(set, assignments.values()))
    assert all(not a & b for i, a in enumerate(sets) for b in sets[i + 1:])
    expected_counts = {'expert_train': {'Real': 200, 'SD15': 100, 'BigGAN': 100},
                       'gate_train': {'Real': 100, 'SD15': 50, 'BigGAN': 50},
                       'calibration': {'Real': 50, 'SD15': 25, 'BigGAN': 25},
                       'validation': {'Real': 50, 'SD15': 25, 'BigGAN': 25},
                       'development': {'Real': 100, 'SD15': 100, 'BigGAN': 100, 'ADM': 100}}
    for part, expected in expected_counts.items():
        assert len(assignments[part]) == len(set(assignments[part]))
        assert Counter(records[lookup[i]]['generator'] for i in assignments[part]) == expected
    assert (len(inputs), len(dev), len(val), len(metrics), len(models)) == (19200, 192000, 48000, 1440, 96)
    assert len(search) == 176
    raw_groups, raw_index = defaultdict(list), {}
    for row in inputs:
        source = records[lookup[row['source_id']]]
        assert int(row['label']) == source['label'] and row['generator'] == source['generator']
        key = (row['profile'], row['expert_mode'], row['partition'])
        raw_groups[key].append(row)
        full = key + (row['condition'], row['source_id'])
        assert full not in raw_index
        raw_index[full] = row
    for (p, mode, part), rows in raw_groups.items():
        conditions = DEV_CONDITIONS if part == 'development' else TRAIN_CONDITIONS
        assert {(r['condition'], r['source_id']) for r in rows} == {(c, i) for c in conditions for i in assignments[part]}

    # Refit all experts with independently constructed per-view sample weights.
    feature, local = {}, {}
    for filename in ['round2_features.npz', 'round4_global_jpeg70.npz']:
        with np.load(ROOT / 'artifacts' / filename) as cache:
            feature.update({k: cache[k] for k in cache.files if '/' in k})
    for p in PROFILES:
        for c in dict.fromkeys(TRAIN_CONDITIONS + DEV_CONDITIONS):
            with np.load(ROOT / 'artifacts' / f'round3_local_{p}_{c}.npz') as cache:
                local[f'{p}/{c}'] = cache['views']
    experts = joblib.load(ROOT / 'artifacts/round6_experts.joblib')
    previous = joblib.load(ROOT / 'artifacts/round4_experts.joblib')
    train = [lookup[i] for i in assignments['expert_train']]
    labels = np.array([r['label'] for r in records])
    for expert_key, heads in experts.items():
        p, mode = expert_key.split('/')
        conditions = EXPERT_CONDITIONS[mode]
        for name in ['global', 'local']:
            matrices = feature if name == 'global' else {k: v.mean(axis=1) for k, v in local.items()}
            x = np.concatenate([matrices[f'{p}/{c}'][train] for c in conditions])
            y = np.tile(labels[train], len(conditions))
            weight = np.full(len(y), 1 / len(conditions))
            assert np.isclose(weight.sum(), 400)
            refit = make_pipeline(StandardScaler(), LogisticRegression(C=1., solver='liblinear', max_iter=2000, random_state=pilot.SEED))
            refit.fit(x, y, logisticregression__sample_weight=weight)
            for c in dict.fromkeys(TRAIN_CONDITIONS + DEV_CONDITIONS):
                xx = matrices[f'{p}/{c}']
                assert np.allclose(refit.decision_function(xx), heads[name].decision_function(xx), atol=1e-12, rtol=0)
                if mode == 'clean':
                    assert np.allclose(heads[name].decision_function(xx), previous[p][name].decision_function(xx), atol=1e-12, rtol=0)
        for c in dict.fromkeys(TRAIN_CONDITIONS + DEV_CONDITIONS):
            key = f'{p}/{c}'
            zg = heads['global'].decision_function(feature[key])
            zl = heads['local'].decision_function(local[key].mean(axis=1))
            variance = heads['local'].decision_function(local[key].reshape(-1, 512)).reshape(-1, 4).var(axis=1)
            for part in ['calibration', 'gate_train', 'validation', 'development']:
                for row in raw_groups[(p, mode, part)]:
                    if row['condition'] != c:
                        continue
                    i = lookup[row['source_id']]
                    expected = [zg[i], zl[i], variance[i], *feature['quality/' + key][i]]
                    actual = [float(row[k]) for k in ['global_logit', 'local_logit', 'variance', 'q_laplacian', 'q_high_frequency', 'q_block']]
                    assert np.allclose(expected, actual, atol=1e-12, rtol=0)

    calibration_metrics = []
    for expert_key, fitted in temperatures.items():
        p, mode = expert_key.split('/')
        rows = raw_groups[(p, mode, 'calibration')]
        _, zg, zl, y = arrays(rows, 1., 1.)
        assert len(y) == 300
        for name, z in [('global', zg), ('local', zl)]:
            info = fitted[name]
            derivative = lambda beta: float(np.mean(z * (expit(beta * z) - y)))
            low, high = 1 / MAX_T, 1 / MIN_T
            beta = low if derivative(low) >= 0 else high if derivative(high) <= 0 else brentq(derivative, low, high, xtol=1e-12)
            assert abs(nll(beta * z, y) - info['fit_bce_after']) < 1e-9
            assert np.isclose(info['fit_bce_before'], nll(z, y), rtol=0, atol=1e-12)
            assert MIN_T <= info['temperature'] <= MAX_T and info['fit_n'] == 300
            assert nll(z / info['temperature'], y) <= nll(z, y) + 1e-12
        for part in ['validation', 'development']:
            for c in DEV_CONDITIONS if part == 'development' else TRAIN_CONDITIONS:
                rows = [r for r in raw_groups[(p, mode, part)] if r['condition'] == c and (part != 'development' or r['generator'] in ('Real', 'ADM'))]
                _, zg, zl, y = arrays(rows, 1., 1.)
                for name, z in [('global', zg), ('local', zl)]:
                    for cal in CALIBRATIONS:
                        t = fitted[name]['temperature'] if cal == 'temperature' else 1.
                        zz, pp = z / t, expit(z / t)
                        assert roc_auc_score(y, zz) == roc_auc_score(y, z)
                        calibration_metrics.append(dict(profile=p, expert_mode=mode, partition=part, condition=c, expert=name,
                                                        calibration=cal, n=len(y), auroc=float(roc_auc_score(y, zz)),
                                                        bce=nll(zz, y), brier=float(np.mean((pp - y) ** 2))))
    dump_csv(OUT / 'heldout_calibration.csv', calibration_metrics)

    curve_groups = defaultdict(list)
    for row in curves:
        curve_groups[(row['arm'], row['model'], row['seed'])].append(row)
    for key, artifact in models.items():
        p, mode, cal, kind, seed = key.split('/')
        config = settings[f'{p}/{mode}/{cal}']
        x, _, _, _ = arrays(raw_groups[(p, mode, 'gate_train')], config['global_temperature'], config['local_temperature'])
        scale = x.std(axis=0); scale[scale == 0] = 1
        assert np.allclose(x.mean(axis=0), artifact['mean'], atol=1e-12, rtol=0)
        assert np.allclose(scale, artifact['scale'], atol=1e-12, rtol=0)
        history = curve_groups[(f'{p}/{mode}/{cal}', kind, seed)]
        assert [int(r['epoch']) for r in history] == list(range(len(history)))
        best, epoch = float('inf'), 0
        for row in history:
            if float(row['validation_bce']) < best - 1e-6:
                best, epoch = float(row['validation_bce']), int(row['epoch'])
        assert epoch == artifact['best_epoch'] and len(history) - 1 == artifact['stopped_epoch']
        assert abs(best - artifact['best_validation_bce']) < 1e-12
        assert len(history) - 1 == gates.MAX_EPOCHS or len(history) - 1 - epoch == gates.PATIENCE

    prediction_groups = {'validation': defaultdict(list), 'development': defaultdict(list)}
    for part, source in [('validation', val), ('development', dev)]:
        unique = set()
        for row in source:
            key = tuple(row[k] for k in ['arm', 'model', 'seed', 'condition', 'source_id'])
            assert key not in unique
            unique.add(key)
            prediction_groups[part][key[:3]].append(row)
            reference = records[lookup[row['source_id']]]
            assert int(row['label']) == reference['label'] and row['generator'] == reference['generator']
    alpha_error = 0.
    for arm, config in settings.items():
        p, mode, cal = arm.split('/')
        expected_t = [temperatures[f'{p}/{mode}'][n]['temperature'] if cal == 'temperature' else 1. for n in ['global', 'local']]
        assert expected_t == [config['global_temperature'], config['local_temperature']]
        tg, tl = expected_t
        vrows = raw_groups[(p, mode, 'validation')]
        _, zg, zl, yy = arrays(vrows, tg, tl)
        candidates = []
        for a in np.arange(11) / 10:
            aucs = []
            for c in TRAIN_CONDITIONS:
                for g in ['SD15', 'BigGAN']:
                    mask = np.array([r['condition'] == c and r['generator'] in ('Real', g) for r in vrows])
                    aucs.append(roc_auc_score(yy[mask], (a * zg + (1 - a) * zl)[mask]))
            value = float(np.mean(aucs))
            saved = next(r for r in search if r['arm'] == arm and float(r['alpha']) == a)
            assert abs(value - float(saved['validation_macro_auroc'])) < 1e-12
            candidates.append((a, value))
        best = max(s for _, s in candidates)
        chosen = min([a for a, s in candidates if np.isclose(s, best)], key=lambda a: (abs(a - .5), a))
        assert chosen == config['fixed_alpha']
        for model_id, info in config['models'].items():
            kind, seed = model_id.split('/')
            for part in ['validation', 'development']:
                rows = prediction_groups[part][(arm, kind, seed)]
                conditions = TRAIN_CONDITIONS if part == 'validation' else DEV_CONDITIONS
                assert {(r['condition'], r['source_id']) for r in rows} == {(c, i) for c in conditions for i in assignments[part]}
                bound = [raw_index[(p, mode, part, r['condition'], r['source_id'])] for r in rows]
                x, zg, zl, y = arrays(bound, tg, tl)
                aa = np.array([float(r['alpha_global']) for r in rows])
                if kind.endswith('_gate'):
                    artifact = models[f'{arm}/{model_id}']
                    independently = numpy_gate(artifact, x)
                    if part == 'validation':
                        assert abs(nll(independently * zg + (1 - independently) * zl, y) - artifact['best_validation_bce']) < 1e-5
                else:
                    independently = np.full(len(rows), {'global': 1., 'local': 0., 'average': .5, 'fixed': chosen}[kind])
                alpha_error = max(alpha_error, float(abs(aa - independently).max()))
                assert np.allclose(aa, independently, atol=1e-6, rtol=0)
                zz = np.array([float(r['logit']) for r in rows])
                pp = np.array([float(r['probability_fake']) for r in rows])
                assert np.allclose(zz, aa * zg + (1 - aa) * zl, atol=1e-12, rtol=0)
                assert np.allclose(pp, expit(zz), atol=1e-12, rtol=0)
                if part == 'validation':
                    assert abs(independent_threshold(y, pp) - info['threshold']) < 1e-12

    rng = np.random.default_rng(20261022)
    draws = np.concatenate([rng.integers(0, 100, (2000, 100)), rng.integers(100, 200, (2000, 100))], axis=1)
    distributions, point_rows = {}, {}
    for cell in metrics:
        rows = sorted([r for r in prediction_groups['development'][(cell['arm'], cell['model'], cell['seed'])]
                       if r['condition'] == cell['condition'] and r['generator'] in ('Real', cell['generator'])],
                      key=lambda r: (int(r['label']), r['source_id']))
        assert len(rows) == int(cell['n']) == 200
        y = np.array([int(r['label']) for r in rows])
        assert np.array_equal(y, np.repeat([0, 1], 100))
        z, prob, aa = [np.array([float(r[k]) for r in rows]) for k in ['logit', 'probability_fake', 'alpha_global']]
        t = settings[cell['arm']]['models'][f"{cell['model']}/{cell['seed']}"]['threshold']
        assert t == float(cell['threshold'])
        wins = z[100:, None] - z[None, :100]
        auc = float(np.mean((wins > 0) + .5 * (wins == 0)))
        checks = {'auroc': auc, 'average_precision': average_precision_score(y, z), 'bce': nll(z, y),
                  'brier': np.mean((prob - y) ** 2), 'balanced_accuracy': balanced_accuracy_score(y, prob >= t),
                  'real_false_positive_rate': np.mean(prob[y == 0] >= t), 'mean_alpha_global': aa.mean(), 'std_alpha_global': aa.std()}
        assert all(np.isclose(v, float(cell[k]), atol=1e-12, rtol=0) for k, v in checks.items())
        key = tuple(cell[k] for k in ['arm', 'model', 'seed', 'condition', 'generator'])
        assert key not in point_rows
        point_rows[key] = cell
        ranks = rankdata(z[draws], axis=1)
        distributions[key] = (ranks[:, 100:].sum(axis=1) - 5050) / 10000
    kinds = ['global', 'local', 'average', 'fixed', 'score_gate', 'quality_gate']
    aggregate, boot_mean, lookup_mean = [], {}, {}
    for arm in settings:
        for kind in kinds:
            seeds = gates.SEEDS if kind.endswith('_gate') else [-1]
            for c in DEV_CONDITIONS:
                for generator in ['SD15', 'BigGAN', 'ADM']:
                    keys = [(arm, kind, str(seed), c, generator) for seed in seeds]
                    rows = [point_rows[k] for k in keys]
                    aucs = np.array([float(r['auroc']) for r in rows])
                    boot = np.mean([distributions[k] for k in keys], axis=0)
                    lo, hi = np.percentile(boot, [2.5, 97.5])
                    item = dict(arm=arm, model=kind, condition=c, generator=generator, initialization_count=len(seeds),
                                mean_auroc=float(aucs.mean()), std_auroc=float(aucs.std(ddof=1)) if len(seeds) > 1 else 0.,
                                auc_ci_low=float(lo), auc_ci_high=float(hi))
                    item.update({'mean_' + k: float(np.mean([float(r[k]) for r in rows])) for k in
                                 ['bce', 'brier', 'balanced_accuracy', 'real_false_positive_rate', 'mean_alpha_global']})
                    key = (arm, kind, c, generator)
                    aggregate.append(item); lookup_mean[key] = item; boot_mean[key] = boot
    dump_csv(OUT / 'aggregate_metrics.csv', aggregate)
    contrasts = []
    def contrast(label, a, b, c, generator):
        ka, kb = (*a, c, generator), (*b, c, generator)
        lo, hi = np.percentile(boot_mean[ka] - boot_mean[kb], [2.5, 97.5])
        contrasts.append(dict(contrast=label, arm_a=a[0], model_a=a[1], arm_b=b[0], model_b=b[1], condition=c, generator=generator,
                              mean_auroc_delta=lookup_mean[ka]['mean_auroc'] - lookup_mean[kb]['mean_auroc'],
                              paired_ci_low=float(lo), paired_ci_high=float(hi),
                              mean_bce_delta=lookup_mean[ka]['mean_bce'] - lookup_mean[kb]['mean_bce']))
    for p in PROFILES:
        for c in DEV_CONDITIONS:
            for generator in ['SD15', 'BigGAN', 'ADM']:
                for mode in EXPERT_MODES:
                    for cal in CALIBRATIONS:
                        arm = f'{p}/{mode}/{cal}'
                        for other in ['score_gate', 'average', 'fixed']:
                            contrast('quality_gate_minus_' + other, (arm, 'quality_gate'), (arm, other), c, generator)
                    for kind in kinds:
                        contrast('temperature_minus_raw', (f'{p}/{mode}/temperature', kind), (f'{p}/{mode}/raw', kind), c, generator)
                for cal in CALIBRATIONS:
                    for kind in kinds:
                        for a, b in [('jpeg_only', 'clean'), ('resize_only', 'clean'), ('augmented', 'clean'),
                                     ('augmented', 'jpeg_only'), ('augmented', 'resize_only')]:
                            contrast(a + '_minus_' + b, (f'{p}/{a}/{cal}', kind), (f'{p}/{b}/{cal}', kind), c, generator)
    dump_csv(OUT / 'paired_contrasts.csv', contrasts)
    macro, macro_boot = [], {}
    for arm in settings:
        for kind in kinds:
            for generator in ['SD15', 'BigGAN', 'ADM']:
                keys = [(arm, kind, c, generator) for c in DEV_CONDITIONS]
                bb = np.mean([boot_mean[k] for k in keys], axis=0)
                lo, hi = np.percentile(bb, [2.5, 97.5])
                item = dict(arm=arm, model=kind, generator=generator, condition_count=3,
                            macro_auroc=float(np.mean([lookup_mean[k]['mean_auroc'] for k in keys])),
                            auc_ci_low=float(lo), auc_ci_high=float(hi))
                for metric in ['bce', 'brier', 'balanced_accuracy', 'real_false_positive_rate']:
                    item['macro_' + metric] = float(np.mean([lookup_mean[k]['mean_' + metric] for k in keys]))
                macro.append(item)
                macro_boot[(arm, kind, generator)] = bb
    dump_csv(OUT / 'macro_summary.csv', macro)
    macro_index = {(r['arm'], r['model'], r['generator']): r for r in macro}
    macro_contrasts = []
    for p in PROFILES:
        for cal in CALIBRATIONS:
            for kind in kinds:
                for generator in ['SD15', 'BigGAN', 'ADM']:
                    for a, b in [('jpeg_only', 'clean'), ('resize_only', 'clean'), ('augmented', 'clean'),
                                 ('augmented', 'jpeg_only'), ('augmented', 'resize_only')]:
                        ka, kb = (f'{p}/{a}/{cal}', kind, generator), (f'{p}/{b}/{cal}', kind, generator)
                        lo, hi = np.percentile(macro_boot[ka] - macro_boot[kb], [2.5, 97.5])
                        macro_contrasts.append(dict(profile=p, calibration=cal, model=kind, generator=generator,
                                                    contrast=a + '_minus_' + b,
                                                    macro_auroc_delta=macro_index[ka]['macro_auroc'] - macro_index[kb]['macro_auroc'],
                                                    paired_ci_low=float(lo), paired_ci_high=float(hi)))
    dump_csv(OUT / 'macro_contrasts.csv', macro_contrasts)
    # Rerun both reference arms; require numerical agreement with round five.
    old_predictions = read_csv(ROOT / 'results/round5/predictions.csv')
    new_index = {tuple(r[k] for k in ['arm', 'model', 'seed', 'condition', 'source_id']): r for r in dev
                 if r['arm'].split('/')[1] in ('clean', 'augmented')}
    assert len(new_index) == len(old_predictions)
    max_control_error = 0.
    for row in old_predictions:
        other = new_index[tuple(row[k] for k in ['arm', 'model', 'seed', 'condition', 'source_id'])]
        for field in ['logit', 'probability_fake', 'alpha_global']:
            max_control_error = max(max_control_error, abs(float(row[field]) - float(other[field])))
    assert max_control_error < 1e-10, max_control_error
    assert assignments == json.loads((ROOT / 'results/round5/assignments.json').read_text())

    verification = {'status': 'passed', 'metric_cells': len(metrics), 'aggregate_cells': len(aggregate),
                    'gates_reconstructed': len(models), 'experts_refit': 16, 'temperatures_independently_optimized': 16,
                    'paired_contrasts': len(contrasts), 'macro_contrasts': len(macro_contrasts),
                    'max_control_prediction_error_vs_round5': max_control_error, 'max_numpy_vs_torch_alpha_error': alpha_error,
                    'checks': ['Five disjoint original-level partitions; all sixteen arms use identical data',
                               'Weighted expert refits; clean expert logits match round4',
                               'Clean and joint control-arm predictions match round5',
                               'Macro bootstrap keeps each source image paired across all three conditions',
                               'Raw branch inputs rebound to source feature rows',
                               'Temperatures independently checked by convex inverse-temperature gradient root',
                               'Held-out expert ranking invariant to temperature',
                               'Gate scaler and variance scaling checked; early stopping histories verified',
                               '96 gates reconstructed with NumPy; all fixed weights and thresholds reselected from validation',
                               'All AUROC/AP/BCE/Brier/BA/FPR/weight metrics recomputed'],
                    'scope': 'Saved artifact verification plus expert refits/calibration optimization; gate training not independently repeated'}
    write_json(OUT / 'verification.json', verification)

    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
