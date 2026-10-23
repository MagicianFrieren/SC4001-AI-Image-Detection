"""Factorial expert augmentation x independent temperature calibration pilot."""
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import joblib
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import gate_models as gates
import prepare_data as data
import run_pilot as pilot
import run_round2 as r2
import run_round4 as r4
import score_calibration as calibration

ROOT = pilot.ROOT
OUT = ROOT / 'results/round5'
PROFILES = r4.PROFILES
TRAIN_CONDITIONS = r4.TRAIN_CONDITIONS
DEV_CONDITIONS = r4.DEV_CONDITIONS
EXPERT_MODES = ['clean', 'augmented']
CALIBRATIONS = ['raw', 'temperature']


def load_data():
    meta = json.loads((ROOT / 'results/round4/run_metadata.json').read_text())
    for path, checksum in meta['feature_sha256'].items():
        assert data.sha256(ROOT / path) == checksum, f'Feature cache changed: {path}'
    for name, checksum in meta['code_sha256'].items():
        assert data.sha256(ROOT / 'scripts' / name) == checksum
    records = r2.read_manifest(ROOT / 'results/round2/manifest.jsonl')
    r2.validate_records(records, r2.read_manifest(ROOT / 'results/manifest.jsonl'))
    source = json.loads((ROOT / 'results/round4/assignments.json').read_text())
    lookup = {r['source_id']: i for i, r in enumerate(records)}
    assignments = {k: source[k] for k in ['expert_train', 'gate_train', 'development']}
    cal, validation = [], []
    for generator, count in [('Real', 50), ('SD15', 25), ('BigGAN', 25)]:
        ids = sorted(i for i in source['validation'] if records[lookup[i]]['generator'] == generator)
        random.Random(20261025 + count + len(generator)).shuffle(ids)
        cal.extend(ids[:count]); validation.extend(ids[count:])
    assignments['calibration'], assignments['validation'] = sorted(cal), sorted(validation)
    ids = {k: [lookup[i] for i in values] for k, values in assignments.items()}
    sets = list(map(set, ids.values()))
    assert all(not a & b for i, a in enumerate(sets) for b in sets[i + 1:])
    feature, quality, local = {}, {}, {}
    with np.load(ROOT / 'artifacts/round2_features.npz') as cache:
        for p in PROFILES:
            for c in DEV_CONDITIONS:
                feature[f'{p}/{c}'] = cache[f'{p}/{c}']
                quality[f'{p}/{c}'] = cache[f'quality/{p}/{c}']
    with np.load(ROOT / 'artifacts/round4_global_jpeg70.npz') as cache:
        for p in PROFILES:
            feature[f'{p}/jpeg70'] = cache[f'{p}/jpeg70']
            quality[f'{p}/jpeg70'] = cache[f'quality/{p}/jpeg70']
    for p in PROFILES:
        for c in dict.fromkeys(TRAIN_CONDITIONS + DEV_CONDITIONS):
            with np.load(ROOT / 'artifacts' / f'round3_local_{p}_{c}.npz') as cache:
                local[f'{p}/{c}'] = cache['views']
    return records, assignments, ids, feature, quality, local, meta


def train_head(x, y, views):
    head = make_pipeline(StandardScaler(), LogisticRegression(C=1., solver='liblinear', max_iter=2000,
                                                             random_state=pilot.SEED))
    # Each original contributes total logistic-loss weight one in both arms.
    head.fit(x, y, logisticregression__sample_weight=np.full(len(y), 1. / views))
    return head


def transform(batch, global_t, local_t):
    result = dict(batch)
    result['zg'], result['zl'] = batch['zg'] / global_t, batch['zl'] / local_t
    result['variance'] = batch['variance'] / local_t ** 2
    result['x'] = gates.inputs(result['zg'], result['zl'], result['variance'], batch['q'])
    return result


def main():
    start = time.perf_counter()
    records, assignments, ids, features, quality, local, prior_meta = load_data()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        'id': 'round5_expert_augmentation_temperature', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'question': 'Separate augmentation and score-scale effects on expert robustness and useful gating',
        'design': '2 expert training modes x 2 calibration modes x 2 preprocessing profiles',
        'manifest_sha256': data.sha256(ROOT / 'results/round2/manifest.jsonl'),
        'partitions': {'expert_train': 'Same 400 originals as round4: 200 Real + 100 SD15 + 100 BigGAN',
                       'gate_train': 'Same 200 originals as round4: 100 Real + 50 SD15 + 50 BigGAN',
                       'calibration': '100 prior validation originals: 50 Real + 25 SD15 + 25 BigGAN',
                       'validation': 'Other 100 prior validation originals: same counts',
                       'development': 'Original 400 development images unchanged'},
        'split_seed': 20261025, 'profiles': PROFILES,
        'expert_modes': {'clean': ['clean'], 'augmented': TRAIN_CONDITIONS},
        'expert_loss_weighting': 'Each view weight 1/number_of_views; total 400 per expert; C=1 unchanged',
        'gate_and_calibration_and_validation_conditions': TRAIN_CONDITIONS,
        'development_conditions': DEV_CONDITIONS,
        'temperature': {'rule': 'Independent positive scalar per expert, shared across generators and conditions',
                        'objective': 'Mean BCE on 300 calibration views only; no intercept',
                        'bounds': [calibration.MIN_T, calibration.MAX_T],
                        'variance_rule': 'Divide local view logit variance by local temperature squared'},
        'gate': 'Round4 architecture, feature masking, training policy and initialization seeds unchanged',
        'gate_seeds': gates.SEEDS, 'early_stopping': 'Pooled 300-view validation BCE; epoch0 eligible',
        'fixed_weight': 'Validation generator x condition macro AUROC, 0..1 grid by0.1, ties nearest0.5 then smaller',
        'threshold': 'Each model maximum Youden J on validation probabilities, fixed during development',
        'ranking_metrics': 'AUROC and AP use logits to avoid sigmoid saturation; BCE uses stable logits; Brier uses probability',
        'primary_contrasts': ['Augmented minus clean within each calibration mode',
                              'Temperature minus raw within each expert mode',
                              'Quality gate minus score gate and simple fusion within each cell'],
        'uncertainty': '2000 paired class-stratified image draws per generator/condition, synchronized across seeds and arms',
        'limits': ['One data partition; three gate initializations only',
                   'Calibration/select subsets smaller than prior validation; use within-round comparisons',
                   'Positive temperature preserves each expert logit ranking, but fusion and gate ranking can change',
                   'Source/content confounds remain; all development has already been inspected',
                   'resize05 seen in expert augmentation and gating; JPEG50 strength not used for training/calibration/selection',
                   'Intervals condition on fitted models and calibration set; no multiplicity correction']}
    path = OUT / 'protocol.json'
    if path.exists():
        old = json.loads(path.read_text())
        assert {k: v for k, v in old.items() if k != 'created_utc'} == {k: v for k, v in protocol.items() if k != 'created_utc'}
    else:
        r2.write_json(path, protocol)
    r2.write_json(OUT / 'assignments.json', assignments)
    all_inputs, predictions, val_predictions, metrics, curves, search = [], [], [], [], [], []
    gate_models, settings, experts, temperatures = {}, {}, {}, {}
    labels = np.array([r['label'] for r in records])
    for profile in PROFILES:
        for mode in EXPERT_MODES:
            conditions = ['clean'] if mode == 'clean' else TRAIN_CONDITIONS
            train = ids['expert_train']
            yg = np.tile(labels[train], len(conditions))
            xg = np.concatenate([features[f'{profile}/{c}'][train] for c in conditions])
            xl = np.concatenate([local[f'{profile}/{c}'][train].mean(axis=1) for c in conditions])
            gh, lh = train_head(xg, yg, len(conditions)), train_head(xl, yg, len(conditions))
            expert_key = f'{profile}/{mode}'
            experts[expert_key] = {'global': gh, 'local': lh}
            raw = {name: r4.bundle(records, ids[name], DEV_CONDITIONS if name == 'development' else TRAIN_CONDITIONS,
                                  profile, features, local, quality, gh, lh)
                   for name in ['calibration', 'gate_train', 'validation', 'development']}
            for name, batch in raw.items():
                all_inputs.extend(dict(row, profile=profile, expert_mode=mode, partition=name) for row in batch['rows'])
            temperatures[expert_key] = {name: calibration.fit(raw['calibration'][field], raw['calibration']['y'])
                                        for name, field in [('global', 'zg'), ('local', 'zl')]}
            print(f'Temperatures {expert_key}: G={temperatures[expert_key]["global"]["temperature"]:.3f}, L={temperatures[expert_key]["local"]["temperature"]:.3f}', flush=True)
            for cal_mode in CALIBRATIONS:
                tg, tl = [temperatures[expert_key][name]['temperature'] if cal_mode == 'temperature' else 1.
                          for name in ['global', 'local']]
                batches = {name: transform(batch, tg, tl) for name, batch in raw.items()}
                val, dev = batches['validation'], batches['development']
                arm = f'{profile}/{mode}/{cal_mode}'
                candidates = []
                for a in np.arange(11) / 10:
                    score = r4.validation_macro(val, a * val['zg'] + (1 - a) * val['zl'])
                    candidates.append((float(a), score))
                    search.append(dict(arm=arm, alpha=float(a), validation_macro_auroc=score))
                best = max(s for _, s in candidates)
                fixed = min((a for a, s in candidates if np.isclose(s, best)), key=lambda a: (abs(a - .5), a))
                settings[arm] = {'global_temperature': tg, 'local_temperature': tl, 'fixed_alpha': fixed, 'models': {}}
                fitted = [(name, -1, np.full(len(val['y']), a), np.full(len(dev['y']), a))
                          for name, a in [('global', 1.), ('local', 0.), ('average', .5), ('fixed', fixed)]]
                for seed in gates.SEEDS:
                    for kind in ['score_gate', 'quality_gate']:
                        model, scaler, artifact, history = gates.fit(batches['gate_train'], val, kind, seed)
                        gate_models[f'{arm}/{kind}/{seed}'] = artifact
                        curves.extend(dict(row, arm=arm, model=kind, seed=seed) for row in history)
                        fitted.append((kind, seed, gates.predict(model, scaler, val['x'], kind), gates.predict(model, scaler, dev['x'], kind)))
                for name, seed, va, da in fitted:
                    vz, dz = va * val['zg'] + (1 - va) * val['zl'], da * dev['zg'] + (1 - da) * dev['zl']
                    vp, dp = expit(vz), expit(dz)
                    threshold = pilot.choose_threshold(val['y'], vp)
                    settings[arm]['models'][f'{name}/{seed}'] = {'threshold': threshold}
                    for target, batch, aa, zz, pp in [(val_predictions, val, va, vz, vp), (predictions, dev, da, dz, dp)]:
                        for row, a, z, p in zip(batch['rows'], aa, zz, pp):
                            target.append(dict(arm=arm, model=name, seed=seed, source_id=row['source_id'], generator=row['generator'],
                                               label=row['label'], condition=row['condition'], alpha_global=float(a), logit=float(z), probability_fake=float(p)))
                    for condition in DEV_CONDITIONS:
                        for generator in ['SD15', 'BigGAN', 'ADM']:
                            mask = np.array([r['condition'] == condition and r['generator'] in ('Real', generator) for r in dev['rows']])
                            yy, pp, zz, aa = dev['y'][mask], dp[mask], dz[mask], da[mask]
                            metrics.append(dict(arm=arm, model=name, seed=seed, condition=condition, generator=generator, n=len(yy), threshold=threshold,
                                                auroc=float(roc_auc_score(yy, zz)), average_precision=float(average_precision_score(yy, zz)),
                                                bce=calibration.nll(zz, yy), brier=float(np.mean((pp - yy) ** 2)),
                                                balanced_accuracy=float(balanced_accuracy_score(yy, pp >= threshold)),
                                                real_false_positive_rate=float(np.mean(pp[yy == 0] >= threshold)),
                                                mean_alpha_global=float(aa.mean()), std_alpha_global=float(aa.std())))
                print(f'Completed {arm}; fixed global weight={fixed:.1f}', flush=True)
    for name, rows in [('branch_inputs', all_inputs), ('predictions', predictions), ('validation_predictions', val_predictions),
                       ('metrics', metrics), ('training_curves', curves), ('weight_search', search)]:
        r2.dump_csv(OUT / f'{name}.csv', rows)
    r2.write_json(OUT / 'settings.json', settings)
    r2.write_json(OUT / 'temperatures.json', temperatures)
    r2.write_json(OUT / 'gate_models.json', gate_models)
    joblib.dump(experts, ROOT / 'artifacts/round5_experts.joblib')
    r2.write_json(OUT / 'run_metadata.json', {
        'status': 'completed', 'run_seconds': time.perf_counter() - start,
        'feature_sha256': prior_meta['feature_sha256'],
        'protocol_sha256': data.sha256(OUT / 'protocol.json'), 'assignments_sha256': data.sha256(OUT / 'assignments.json'),
        'code_sha256': {name: data.sha256(ROOT / 'scripts' / name) for name in
                       ['run_round5.py', 'score_calibration.py', 'run_round4.py', 'gate_models.py', 'run_pilot.py']},
        'environment': prior_meta['environment']})
    print('Round five complete; run scripts/verify_round5.py.', flush=True)


if __name__ == '__main__':
    main()
