"""Exploratory global/local complementarity and fixed-logit-fusion experiment."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import joblib
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
import torch

import local_branch
import prepare_data as data
import run_pilot as pilot
import run_round2 as round2

ROOT = pilot.ROOT
OUT = ROOT / 'results/round3'
PROFILES = ['original', 'common128']
CONDITIONS = ['clean', 'jpeg50', 'resize05']
MODELS = ['global', 'local', 'average', 'fixed']


def load_inputs():
    records = round2.read_manifest(ROOT / 'results/round2/manifest.jsonl')
    round2.validate_records(records, round2.read_manifest(ROOT / 'results/manifest.jsonl'))
    by_id = {r['source_id']: i for i, r in enumerate(records)}
    source = json.loads((ROOT / 'results/round2/assignments.json').read_text())
    groups = {s: ({v: [by_id[x] for x in ids] for v, ids in value.items()} if isinstance(value, dict)
                  else [by_id[x] for x in value]) for s, value in source.items()}
    # Validate the complete round-two cache identity, not only its filename.
    identity = hashlib.sha256((json.dumps(records, sort_keys=True) + data.sha256(ROOT / 'scripts/run_round2.py')
                              + data.sha256(ROOT / 'scripts/run_pilot.py') + pilot.MODEL_REVISION
                              + '32' + torch.__version__ + np.__version__).encode()).hexdigest()
    with np.load(ROOT / 'artifacts/round2_features.npz') as cache:
        assert str(cache['identity']) == identity, 'Global feature cache is stale; rerun round two'
        global_features = {f'{p}/{c}': cache[f'{p}/{c}'] for p in PROFILES for c in CONDITIONS}
    for variant in groups['train']:
        train, val, dev = map(set, [groups['train'][variant], groups['validation'][variant], groups['development']])
        assert len(train) == 600 and len(val) == 200 and len(dev) == 400
        assert not (train & val or train & dev or val & dev)
        assert all(records[i]['generator'] != 'ADM' for i in train | val)
    return records, groups, source, global_features


def macro_val_auc(records, ids, y, score):
    generators = sorted({records[i]['generator'] for i in ids} - {'Real'})
    return float(np.mean([roc_auc_score(y[mask], score[mask]) for g in generators
                          for mask in [np.array([records[i]['generator'] in ('Real', g) for i in ids])]]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('Batch size must be positive')
    start = time.perf_counter()
    records, groups, source, global_features = load_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        'id': 'round3_global_local_fixed_fusion', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Measure local branch behavior, error complementarity and fixed fusion before training a gate',
        'source': 'https://arxiv.org/abs/2502.17105',
        'implementation_scope': 'SFLD-inspired small pilot, not a paper reproduction',
        'manifest_sha256': data.sha256(ROOT / 'results/round2/manifest.jsonl'),
        'assignments_sha256': data.sha256(ROOT / 'results/round2/assignments.json'),
        'profiles': PROFILES, 'conditions': CONDITIONS,
        'local': {'patch_size': 56, 'seeds': local_branch.VIEW_SEEDS,
                  'permutations': [local_branch.permutation(s).tolist() for s in local_branch.VIEW_SEEDS],
                  'rule': 'Same four label-independent permutations for every image and condition; degrade before shuffle',
                  'pooling': 'Arithmetic mean of four L2-normalized 512D embeddings, without renormalizing mean'},
        'model': pilot.MODEL, 'model_revision': pilot.MODEL_REVISION,
        'train': 'Same clean 600 originals as round two per training variant; independent global/local scaler + C=1 liblinear',
        'fusion': 'sigmoid(alpha * global_logit + (1-alpha) * local_logit)',
        'alpha_grid': [i / 10 for i in range(11)],
        'alpha_selection': 'Maximum clean validation generator-macro AUROC; ties closest to 0.5 then smaller alpha',
        'thresholds': 'Separate threshold per model from pooled clean validation Youden J, then fixed for all conditions',
        'diagnostics': 'Both correct, only global correct, only local correct, both wrong at separate validation thresholds; score correlation',
        'uncertainty': '2000 paired class-stratified bootstrap draws per development cell, no multiplicity correction',
        'limits': ['Exploratory development only; no final holdout or gate in this round',
                   'Single training subset and one four-view ensemble; no training-seed repetition',
                   'Quality/source/semantic confounding remains unresolved',
                   'PatchShuffle view is not pure texture; rearrangement creates artificial boundaries',
                   '56px shuffle tiles do not align with 32px ViT tokens; backbone/view-count/training differ from SFLD',
                   'Validation selects both alpha and threshold; its performance is optimistic',
                   'Shared real pool induces dependent comparisons; intervals condition on fitted models']}
    path = OUT / 'protocol.json'
    if path.exists():
        old = json.loads(path.read_text())
        assert {k: v for k, v in old.items() if k != 'created_utc'} == {k: v for k, v in protocol.items() if k != 'created_utc'}
    else:
        round2.write_json(path, protocol)
    round2.write_json(OUT / 'assignments.json', source)
    local, extraction, peak = local_branch.extract(records, PROFILES, CONDITIONS, args.batch_size)
    y = np.array([r['label'] for r in records])
    predictions, validation, metrics, overlap, weight_search, settings, heads = [], [], [], [], [], {}, {}
    for profile in PROFILES:
        for variant in groups['train']:
            key = f'{profile}/{variant}'
            train, val, dev = [groups[s][variant] if s != 'development' else groups[s] for s in ['train', 'validation', 'development']]
            ghead = pilot.train_head(global_features[f'{profile}/clean'][train], y[train])
            lhead = pilot.train_head(local[f'{profile}/clean'].mean(axis=1)[train], y[train])
            heads[key] = {'global': ghead, 'local': lhead}
            vg = ghead.decision_function(global_features[f'{profile}/clean'][val])
            vl = lhead.decision_function(local[f'{profile}/clean'].mean(axis=1)[val])
            scored = []
            for alpha in protocol['alpha_grid']:
                auc = macro_val_auc(records, val, y[val], alpha * vg + (1 - alpha) * vl)
                scored.append((alpha, auc))
                weight_search.append(dict(profile=profile, training=variant, alpha=alpha, validation_macro_auroc=auc))
            best = max(s for _, s in scored)
            alpha = min((a for a, s in scored if np.isclose(s, best)), key=lambda a: (abs(a - .5), a))
            weights = {'global': 1., 'local': 0., 'average': .5, 'fixed': alpha}
            thresholds = {m: pilot.choose_threshold(y[val], expit(a * vg + (1 - a) * vl)) for m, a in weights.items()}
            settings[key] = {'selected_alpha': alpha, 'thresholds': thresholds,
                             'validation_global_logit_std': float(vg.std()), 'validation_local_logit_std': float(vl.std())}
            for j, i in enumerate(val):
                validation.append(dict(profile=profile, training=variant, source_id=records[i]['source_id'],
                                       generator=records[i]['generator'], label=int(y[i]), global_logit=float(vg[j]), local_logit=float(vl[j])))
            for condition in CONDITIONS:
                zg = ghead.decision_function(global_features[f'{profile}/{condition}'][dev])
                views = local[f'{profile}/{condition}'][dev]
                zl = lhead.decision_function(views.mean(axis=1))
                variance = lhead.decision_function(views.reshape(-1, 512)).reshape(-1, 4).var(axis=1)
                scores = {m: expit(a * zg + (1 - a) * zl) for m, a in weights.items()}
                for j, i in enumerate(dev):
                    predictions.append(dict(profile=profile, training=variant, condition=condition,
                                            source_id=records[i]['source_id'], generator=records[i]['generator'], label=int(y[i]),
                                            global_logit=float(zg[j]), local_logit=float(zl[j]), local_view_logit_variance=float(variance[j]),
                                            **{m + '_probability': float(p[j]) for m, p in scores.items()}))
                for generator in ['SD15', 'BigGAN', 'ADM']:
                    mask = np.array([records[i]['generator'] in ('Real', generator) for i in dev])
                    yy = y[dev][mask]
                    for model, probability in scores.items():
                        p = probability[mask]
                        threshold = thresholds[model]
                        metrics.append(dict(profile=profile, training=variant, model=model, condition=condition,
                                            generator=generator, n=len(yy), alpha=weights[model], threshold=threshold,
                                            auroc=float(roc_auc_score(yy, p)), average_precision=float(average_precision_score(yy, p)),
                                            balanced_accuracy=float(balanced_accuracy_score(yy, p >= threshold)),
                                            real_false_positive_rate=float(np.mean(p[yy == 0] >= threshold))))
                    gc = (scores['global'][mask] >= thresholds['global']) == yy
                    lc = (scores['local'][mask] >= thresholds['local']) == yy
                    overlap.append(dict(profile=profile, training=variant, condition=condition, generator=generator,
                                        n=len(yy), both_correct=int(np.sum(gc & lc)), global_only_correct=int(np.sum(gc & ~lc)),
                                        local_only_correct=int(np.sum(~gc & lc)), both_wrong=int(np.sum(~gc & ~lc)),
                                        score_correlation=float(np.corrcoef(scores['global'][mask], scores['local'][mask])[0, 1])))
            print(f'Completed {key}: selected global weight={alpha:.1f}', flush=True)
    for name, rows in [('metrics', metrics), ('predictions', predictions), ('validation_predictions', validation),
                       ('error_overlap', overlap), ('weight_search', weight_search)]:
        round2.dump_csv(OUT / f'{name}.csv', rows)
    round2.write_json(OUT / 'settings.json', settings)
    joblib.dump(heads, ROOT / 'artifacts/round3_heads.joblib')
    round2.write_json(OUT / 'run_metadata.json', {
        'status': 'completed', 'run_seconds': time.perf_counter() - start, 'local_extraction': extraction,
        'peak_cuda_memory_mb': peak, 'batch_size': args.batch_size,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'code_sha256': {name: data.sha256(ROOT / 'scripts' / name) for name in
                       ['run_round3.py', 'local_branch.py', 'run_round2.py', 'run_pilot.py']},
        'protocol_sha256': data.sha256(OUT / 'protocol.json'),
        'global_cache_sha256': data.sha256(ROOT / 'artifacts/round2_features.npz'),
        'manifest_sha256': data.sha256(ROOT / 'results/round2/manifest.jsonl'),
        'environment': {name: version(name) for name in ['torch', 'numpy', 'Pillow', 'transformers', 'scikit-learn', 'scipy']}})
    print('Round three completed; run scripts/verify_round3.py.', flush=True)


if __name__ == '__main__':
    main()
