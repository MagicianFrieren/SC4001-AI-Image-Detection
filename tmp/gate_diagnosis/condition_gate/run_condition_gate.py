"""Final bounded CPU condition-gate hypothesis; no test data or method search."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
sys.dont_write_bytecode = True
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import joblib
import numpy as np
import sklearn
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import round7_baseline_common as common
import round7_baseline_clip as clip

SEEDS = [20261101, 20261102, 20261103]
TUNE = ['clean', 'jpeg70', 'resize05']
CONDS = ['clean', 'jpeg70', 'jpeg50', 'resize05']
GENS = ['ADM', 'BigGAN']
METRICS = ['auroc', 'average_precision', 'balanced_accuracy', 'real_false_positive_rate']
HASHES = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(name, value):
    with (OUT / name).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def write_csv(name, rows):
    assert rows
    with (OUT / name).open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)


def read_json(relative, expected=None):
    assert 'final' not in relative.lower() and 'reserved' not in relative.lower()
    path = ROOT / relative
    actual = sha(path)
    assert expected is None or actual == expected, relative
    HASHES[relative] = actual
    return json.loads(path.read_text(encoding='utf-8'))


def arrays(raw):
    return {k: v if k == 'rows' else np.asarray(v) for k, v in raw.items()}


def condition_labels(batch):
    return np.asarray([r['condition'] for r in batch['rows']])


def conditional_weights(tune):
    labels = condition_labels(tune)
    alphas, searches = {}, {}
    for condition in TUNE:
        mask = labels == condition
        candidates = []
        for value in np.arange(11) / 10:
            z = value * tune['zg'][mask] + (1 - value) * tune['zl'][mask]
            candidates.append((float(value), float(roc_auc_score(tune['y'][mask], z))))
        best = max(v for a, v in candidates)
        alphas[condition] = min((a for a, v in candidates if np.isclose(v, best)), key=lambda a: (abs(a - .5), a))
        searches[condition] = candidates
    return alphas, searches


def alpha_predict(classifier, q, alphas):
    probabilities = classifier.predict_proba(q)
    classes = classifier.named_steps['logisticregression'].classes_.tolist()
    # Only q is passed to the classifier; no condition metadata at inference.
    alpha = probabilities @ np.asarray([alphas[c] for c in classes])
    return alpha, probabilities, classes


def diagnostics(classifier, batch):
    actual = condition_labels(batch)
    mask = np.isin(actual, TUNE)
    guess = classifier.predict(batch['q'][mask])
    result = {'classes': TUNE, 'n': int(mask.sum()),
              'accuracy': float(accuracy_score(actual[mask], guess)),
              'confusion_rows_true_columns_predicted': confusion_matrix(actual[mask], guess, labels=TUNE).tolist()}
    for source in ['Real', 'ADM', 'BigGAN']:
        sm = mask & np.asarray([r['generator'] == source for r in batch['rows']])
        if sm.any():
            pred = classifier.predict(batch['q'][sm])
            result.setdefault('by_source', {})[source] = {
                'n': int(sm.sum()), 'accuracy': float(accuracy_score(actual[sm], pred)),
                'confusion': confusion_matrix(actual[sm], pred, labels=TUNE).tolist()}
    jm = actual == 'jpeg50'
    if jm.any():
        pp = classifier.predict_proba(batch['q'][jm])
        classes = classifier.named_steps['logisticregression'].classes_
        result['jpeg50_mean_predicted_condition_probability'] = {str(c): float(p) for c, p in zip(classes, pp.mean(axis=0))}
    return result


def cell_metrics(seed, name, batch, alpha, cutoff):
    z = alpha * batch['zg'] + (1 - alpha) * batch['zl']
    rows = []
    for gen in GENS:
        for condition in CONDS:
            mask = np.asarray([r['condition'] == condition and r['generator'] in ['Real', gen] for r in batch['rows']])
            y = batch['y'][mask]
            rows.append(dict(run_seed=seed, model=name, generator=gen, condition=condition,
                             n_real=int(sum(y == 0)), n_fake=int(sum(y == 1)),
                             threshold=cutoff, alpha_mean=float(np.mean(alpha[mask])),
                             **common.points(y, z[mask], cutoff)))
    return rows, z


def main():
    assert not (OUT / 'protocol.json').exists(), 'Do not overwrite or silently repeat this frozen candidate.'
    sources = {}
    for number, seed in enumerate(SEEDS, 1):
        complete = read_json(f'results/round8_a/seed{number}/matched/complete.json')
        assert complete['status'] == 'completed'
        prefix = f'artifacts/round8_a/seed{number}/matched'
        bundle = read_json(f'{prefix}/bundle.json', complete['files_sha256'][f'{prefix}/bundle.json'])
        assert bundle['run_seed'] == seed and bundle['arm'] == 'matched' and bundle['signature'] == complete['signature']
        for relative in [f'{prefix}/experts.joblib', f'results/round8_a/seed{number}/matched/metrics.csv']:
            digest = sha(ROOT / relative)
            assert digest == complete['files_sha256'][relative]
            HASHES[relative] = digest
        assert HASHES[f'{prefix}/experts.joblib'] == bundle['experts_sha256']
        sources[number] = bundle
    for file in ['round7_baseline_common.py', 'round7_baseline_clip.py', 'round7_baseline_data.py', 'gate_models.py', 'score_calibration.py']:
        HASHES['scripts/' + file] = sha(ROOT / 'scripts' / file)
    protocol = {
        'id': 'exploratory_quality_condition_gate_v1', 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'frozen before any training or development score',
        'hypothesis': 'Three quality proxies may identify degradation conditions more transferably than directly learning authenticity-dependent expert reliability.',
        'fixed_inputs': 'Existing matched expert raw_batches: matched gate_fit and matched tuning; original reporting/development views. No experts refit.',
        'condition_classifier': {'inputs': ['log_laplacian_variance', 'high_frequency_ratio', 'block_boundary_ratio'],
                                 'training_partition': 'gate_fit', 'target': 'condition metadata only, not authenticity',
                                 'pipeline': 'StandardScaler + LogisticRegression', 'C': 1., 'max_iter': 1000,
                                 'random_state': 0, 'solver': 'lbfgs'},
        'fusion_selection': 'For each TUNE condition independently, alpha in 0,.1,...,1 maximizes selection AUROC; np.isclose ties choose closest .5 then earliest/smaller grid alpha.',
        'inference': 'Soft alpha=sum P(condition | q)*selected alpha_condition; only q enters condition classifier. JPEG50 treated identically with no mapped label.',
        'threshold': 'For each model, original pooled tuning Youden-J rule; ties closest probability cutoff .5.',
        'baselines': 'Average and existing matched tuning-selected fixed, recomputed on same tuning; settings and all dev metric cells verified unchanged.',
        'run_seeds': SEEDS, 'condition_classes': TUNE, 'development_conditions': CONDS, 'development_generators': GENS,
        'primary': 'All 8 development AUROC cells and equally weighted 8-cell macro; three data seeds reported separately.',
        'secondary': METRICS[1:],
        'diagnostics': 'Tuning condition-classifier accuracy/confusion; after all models/settings frozen, same diagnostics on development TUNE classes and JPEG50 mean predicted probabilities. No diagnostic controls selection.',
        'constraints': 'One variant only, no blending parameter, no hyperparameter search, no final/reserved test access, no images/feature extraction/GPU, no publish/push.',
        'limitations': 'Exploratory previously inspected ADM/BigGAN development; shared pool/images across seeds; no significance or independent-generalization claim.',
        'input_sha256': HASHES, 'script_sha256': sha(__file__),
        'environment': {'numpy': np.__version__, 'scikit_learn': sklearn.__version__, 'device': 'CPU'},
    }
    write_json('protocol.json', protocol)
    print('Protocol frozen:', sha(OUT / 'protocol.json'), flush=True)
    fitted, settings = {}, {}
    for number, bundle in sources.items():
        train, tune = arrays(bundle['raw_batches']['gate_fit']), arrays(bundle['raw_batches']['tuning'])
        assert set(condition_labels(train)) == set(TUNE) and set(condition_labels(tune)) == set(TUNE)
        classifier = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=1000, random_state=0, solver='lbfgs'))
        classifier.fit(train['q'], condition_labels(train))
        alphas, searches = conditional_weights(tune)
        aa, _, classes = alpha_predict(classifier, tune['q'], alphas)
        fixed, fixed_search = clip.fixed_alpha(tune)
        assert fixed == bundle['settings']['settings']['fixed/-1']['fixed_alpha']
        cutoffs = {}
        for name, weight in [('condition_gate', aa), ('average', .5), ('fixed', fixed)]:
            z = weight * tune['zg'] + (1 - weight) * tune['zl']
            cutoffs[name] = common.threshold(tune['y'], expit(z))
            if name != 'condition_gate':
                assert cutoffs[name] == bundle['settings']['settings'][f'{name}/-1']['threshold']
        lr, scaler = classifier.named_steps['logisticregression'], classifier.named_steps['standardscaler']
        spec = {'run_seed': SEEDS[number - 1], 'condition_alphas': alphas, 'condition_grid': searches,
                'fixed_alpha': fixed, 'fixed_grid': fixed_search, 'thresholds': cutoffs,
                'classifier': {'classes': classes, 'coef': lr.coef_.tolist(), 'intercept': lr.intercept_.tolist(),
                               'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(), 'n_iter': lr.n_iter_.tolist()},
                'tuning_diagnostics': diagnostics(classifier, tune)}
        settings[str(SEEDS[number - 1])] = spec
        joblib.dump(classifier, OUT / f'condition_classifier_seed{number}.joblib')
        fitted[number] = (classifier, alphas, fixed, cutoffs)
        print(f'Fit seed {number}: alphas={alphas}, tuning condition accuracy={spec["tuning_diagnostics"]["accuracy"]:.4f}', flush=True)
    write_json('model_settings.json', settings)
    write_json('fit_freeze.json', {'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
                                 'settings_sha256': sha(OUT / 'model_settings.json'),
                                 'classifier_sha256': {f'condition_classifier_seed{i}.joblib': sha(OUT / f'condition_classifier_seed{i}.joblib') for i in (1, 2, 3)},
                                 'development_scoring_started': False})
    metrics, predictions, dev_diagnostics = [], [], {}
    baseline_errors = {m: 0. for m in METRICS}
    for number, bundle in sources.items():
        seed = SEEDS[number - 1]
        dev = arrays(bundle['raw_batches']['development'])
        classifier, alphas, fixed, cutoffs = fitted[number]
        alpha, probabilities, classes = alpha_predict(classifier, dev['q'], alphas)
        for name, weight in [('condition_gate', alpha), ('average', np.full(len(alpha), .5)), ('fixed', np.full(len(alpha), fixed))]:
            cells, z = cell_metrics(seed, name, dev, weight, cutoffs[name]); metrics.extend(cells)
            if name == 'condition_gate':
                for i, row in enumerate(dev['rows']):
                    predictions.append(dict(run_seed=seed, **row, logit=float(z[i]), probability_fake=float(expit(z[i])),
                                            threshold=cutoffs[name], alpha_global=float(alpha[i]),
                                            **{f'p_condition_{c}': float(probabilities[i, j]) for j, c in enumerate(classes)}))
            else:
                path = ROOT / f'results/round8_a/seed{number}/matched/metrics.csv'
                reference = {(r['generator'], r['condition']): r for r in csv.DictReader(path.open())
                             if r['partition'] == 'development' and r['model'] == name and r['gate_seed'] == '-1'}
                for value in cells:
                    ref = reference[(value['generator'], value['condition'])]
                    for m in METRICS:
                        baseline_errors[m] = max(baseline_errors[m], abs(value[m] - float(ref[m])))
        dev_diagnostics[str(seed)] = diagnostics(classifier, dev)
    assert max(baseline_errors.values()) < 1e-10
    write_csv('metrics_seed_condition.csv', metrics)
    write_csv('candidate_development_predictions.csv', predictions)
    lookup = {(r['run_seed'], r['model'], r['generator'], r['condition']): r for r in metrics}
    paired, cells, seed_macros = [], [], []
    for gen in GENS:
        for cond in CONDS:
            cell = {'generator': gen, 'condition': cond, 'auroc': {}, 'contrasts': {}}
            for model in ['condition_gate', 'average', 'fixed']:
                values = [lookup[(s, model, gen, cond)]['auroc'] for s in SEEDS]
                cell['auroc'][model] = {'mean': float(np.mean(values)), 'sample_sd': float(np.std(values, ddof=1)), 'seeds': values}
            for ref in ['average', 'fixed']:
                diff = []
                for seed in SEEDS:
                    a, b = lookup[(seed, 'condition_gate', gen, cond)], lookup[(seed, ref, gen, cond)]
                    delta = {m: a[m] - b[m] for m in METRICS}
                    paired.append(dict(run_seed=seed, generator=gen, condition=cond, contrast='condition_gate_minus_' + ref, **delta))
                    diff.append(delta['auroc'])
                cell['contrasts']['condition_gate_minus_' + ref] = {'mean': float(np.mean(diff)), 'sample_sd': float(np.std(diff, ddof=1)), 'seeds': diff}
            cells.append(cell)
    for seed in SEEDS:
        record = {'run_seed': seed, 'metrics': {}, 'contrasts': {}}
        for model in ['condition_gate', 'average', 'fixed']:
            rr = [r for r in metrics if r['run_seed'] == seed and r['model'] == model]
            record['metrics'][model] = {m: float(np.mean([r[m] for r in rr])) for m in METRICS}
        for ref in ['average', 'fixed']:
            record['contrasts']['condition_gate_minus_' + ref] = {m: record['metrics']['condition_gate'][m] - record['metrics'][ref][m] for m in METRICS}
        seed_macros.append(record)
    overall = {model: {m: float(np.mean([r[m] for r in metrics if r['model'] == model])) for m in METRICS}
               for model in ['condition_gate', 'average', 'fixed']}
    for path, digest in HASHES.items():
        assert sha(ROOT / path) == digest, 'Input changed during run: ' + path
    write_csv('paired_seed_condition.csv', paired)
    result = {'status': 'completed', 'protocol_sha256': sha(OUT / 'protocol.json'),
              'baseline_max_absolute_difference': baseline_errors, 'input_hashes_unchanged': True,
              'final_test_accessed': False, 'candidate_count': 1, 'condition_classifier_fits': 3,
              'overall_8_cell_macro': overall, 'seed_macros': seed_macros, 'cells': cells,
              'tuning_diagnostics': {seed: value['tuning_diagnostics'] for seed, value in settings.items()},
              'development_condition_diagnostics': dev_diagnostics,
              'limitations': protocol['limitations']}
    write_json('summary.json', result)
    print(json.dumps({'status': 'completed', 'overall': overall, 'seed_macros': seed_macros,
                      'baseline_max_absolute_difference': baseline_errors}, indent=2), flush=True)


if __name__ == '__main__':
    main()
