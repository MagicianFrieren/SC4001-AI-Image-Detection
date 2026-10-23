"""Disjoint gate-training pilot: score gate versus score-plus-quality gate."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import time

import joblib
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
import torch
from transformers import CLIPVisionModelWithProjection

import gate_models as gates
import local_branch
import prepare_data as data
import run_pilot as pilot
import run_round2 as r2
import run_round3 as r3

ROOT = pilot.ROOT
OUT = ROOT / 'results/round4'
PROFILES = ['original', 'common128']
TRAIN_CONDITIONS = ['clean', 'jpeg70', 'resize05']
DEV_CONDITIONS = ['clean', 'jpeg50', 'resize05']
SEED = 20261022


def partition(records, source):
    lookup = {r['source_id']: i for i, r in enumerate(records)}
    pool = source['train']['mixed_sd15_biggan']
    expert, gate = [], []
    for generator, n in [('Real', 200), ('SD15', 100), ('BigGAN', 100)]:
        ids = sorted(i for i in pool if records[lookup[i]]['generator'] == generator)
        random.Random(SEED + n + len(generator)).shuffle(ids)
        expert += ids[:n]
        gate += ids[n:]
    output = {'expert_train': sorted(expert), 'gate_train': sorted(gate),
              'validation': source['validation']['mixed_sd15_biggan'], 'development': source['development']}
    sets = list(map(set, output.values()))
    assert all(not a & b for i, a in enumerate(sets) for b in sets[i + 1:])
    assert set(expert + gate) == set(pool)
    assert len(expert) == 400 and len(gate) == 200
    return output, {name: [lookup[i] for i in ids] for name, ids in output.items()}


def extra_global(records):
    """Only JPEG70 is new; existing JPEG50 development views remain evaluation-only."""
    identity = hashlib.sha256((json.dumps(records, sort_keys=True) + data.sha256(Path(__file__))
                              + data.sha256(ROOT / 'scripts/run_round2.py') + pilot.MODEL_REVISION
                              + torch.__version__ + np.__version__).encode()).hexdigest()
    path = ROOT / 'artifacts/round4_global_jpeg70.npz'
    if path.exists():
        with np.load(path) as cache:
            if str(cache['identity']) == identity:
                return {k: cache[k] for k in cache.files if k != 'identity'}, True
    folder = ROOT / '.cache/clip-vit-base-patch32'
    assert data.sha256(folder / 'pytorch_model.bin') == pilot.MODEL_SHA256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    model = CLIPVisionModelWithProjection.from_pretrained(folder, local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    if device == 'cuda':
        model.half()
    output = {}
    boundary = len(r2.read_manifest(ROOT / 'results/manifest.jsonl'))
    for profile in PROFILES:
        vectors, quality = [], []
        for begin, end in [(0, boundary), (boundary, len(records))]:
            for i in range(begin, end, 32):
                images = [r2.image_view(ROOT / r['path'], profile, 'jpeg70') for r in records[i:min(i + 32, end)]]
                quality.extend(pilot.quality(im) for im in images)
                array = np.stack([(np.asarray(im, dtype=np.float32) / 255 - pilot.MEAN) / pilot.STD for im in images])
                x = torch.from_numpy(array.transpose(0, 3, 1, 2)).to(device=device, dtype=model.dtype)
                with torch.inference_mode():
                    z = model(pixel_values=x).image_embeds.float()
                    z = torch.nn.functional.normalize(z, dim=-1)
                vectors.append(z.cpu().numpy())
        output[f'{profile}/jpeg70'] = np.concatenate(vectors)
        output[f'quality/{profile}/jpeg70'] = np.asarray(quality)
        print(f'Extracted global JPEG70: {profile}', flush=True)
    np.savez_compressed(path, identity=np.array(identity), **output)
    return output, False


def bundle(records, ids, conditions, profile, global_features, local, quality, ghead, lhead):
    rows = []
    for condition in conditions:
        key = f'{profile}/{condition}'
        zg = ghead.decision_function(global_features[key][ids])
        views = local[key][ids]
        zl = lhead.decision_function(views.mean(axis=1))
        variance = lhead.decision_function(views.reshape(-1, 512)).reshape(-1, 4).var(axis=1)
        for j, i in enumerate(ids):
            q = quality[key][i]
            rows.append(dict(source_id=records[i]['source_id'], generator=records[i]['generator'], label=records[i]['label'],
                             condition=condition, global_logit=float(zg[j]), local_logit=float(zl[j]),
                             variance=float(variance[j]), q_laplacian=float(q[0]), q_high_frequency=float(q[1]), q_block=float(q[2])))
    result = {name: np.array([r[field] for r in rows], dtype=float) for name, field in
              [('zg', 'global_logit'), ('zl', 'local_logit'), ('y', 'label'), ('variance', 'variance')]}
    result['q'] = np.array([[r[k] for k in ['q_laplacian', 'q_high_frequency', 'q_block']] for r in rows])
    result['x'] = gates.inputs(result['zg'], result['zl'], result['variance'], result['q'])
    result['rows'] = rows
    return result


def validation_macro(batch, score):
    values = []
    for condition in TRAIN_CONDITIONS:
        for generator in ['SD15', 'BigGAN']:
            mask = np.array([r['condition'] == condition and r['generator'] in ('Real', generator) for r in batch['rows']])
            values.append(roc_auc_score(batch['y'][mask], score[mask]))
    return float(np.mean(values))


def main():
    start = time.perf_counter()
    records, _, source, global_features = r3.load_inputs()
    assignments, ids = partition(records, source)
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        'id': 'round4_disjoint_quality_gate', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'question': 'Does quality information add useful gating beyond expert scores and local-view variability?',
        'manifest_sha256': data.sha256(ROOT / 'results/round2/manifest.jsonl'), 'partition_seed': SEED,
        'partitions': {'expert_train': '200 Real + 100 SD15 + 100 BigGAN, clean only',
                       'gate_train': '100 Real + 50 SD15 + 50 BigGAN; disjoint from experts',
                       'validation': '100 Real + 50 SD15 + 50 BigGAN, unchanged',
                       'development': 'Original 400 images, unchanged; ADM never trains or selects models'},
        'profiles': PROFILES, 'gate_train_and_validation_conditions': TRAIN_CONDITIONS,
        'development_conditions': DEV_CONDITIONS,
        'experts': 'Retrain clean linear heads on 400 originals; identical frozen experts for all compared fusions',
        'gate_features': ['p_global', 'p_local', 'absolute_probability_gap', 'mean_binary_entropy',
                          'log1p_local_view_logit_variance', 'q_laplacian', 'q_high_frequency', 'q_block'],
        'gate_normalization': 'StandardScaler fitted only on 600 gate-training views',
        'architecture': '8 -> 32 ReLU -> 1 sigmoid; score gate zeros final three standardized quality inputs; 321 parameters each',
        'initialization': 'Paired hidden initialization, output layer zero so initial global weight=0.5',
        'seeds': gates.SEEDS, 'training': {'optimizer': 'AdamW', 'learning_rate': gates.LEARNING_RATE,
                                        'weight_decay': .0001, 'max_epochs': gates.MAX_EPOCHS,
                                        'patience': gates.PATIENCE, 'min_delta': 1e-6,
                                        'batch': 'Full batch, 600 views', 'loss': 'BCEWithLogits on fused logit'},
        'selection': 'Minimum pooled validation BCE for each gate; epoch 0 eligible; same policy for both variants',
        'fixed_fusion': 'Alpha 0..1 step0.1 maximizes validation AUROC macro over 2 generators x 3 conditions; tie closest0.5 then smaller',
        'thresholds': 'Each model uses pooled 600-view validation maximum Youden J, fixed on development',
        'comparisons': ['global', 'local', 'average', 'fixed', 'score_gate', 'quality_gate'],
        'main_contrast': 'Mean across 3 initialization-specific AUROCs: quality_gate minus score_gate',
        'uncertainty': '2000 class-stratified paired draws per development cell, same image draws across models and seeds; no multiplicity correction',
        'limits': ['One expert/gate data split; three gate initialization seeds are not three independent data experiments',
                   'ADM and JPEG50 are previously inspected exploratory development conditions',
                   'resize05 is now seen during gate training and validation; JPEG50 intensity is not',
                   'Experts have 400 rather than prior 600 training originals; compare methods within this round',
                   'Validation selects early stopping, fixed alpha and thresholds; no final blind test',
                   'Source/content/quality confounding and PatchShuffle alignment limitations remain',
                   'Quality proxies may encode shortcuts; changed weights alone do not establish useful adaptation']}
    path = OUT / 'protocol.json'
    if path.exists():
        old = json.loads(path.read_text())
        assert {k: v for k, v in old.items() if k != 'created_utc'} == {k: v for k, v in protocol.items() if k != 'created_utc'}
    else:
        r2.write_json(path, protocol)
    r2.write_json(OUT / 'assignments.json', assignments)
    conditions = list(dict.fromkeys(TRAIN_CONDITIONS + DEV_CONDITIONS))
    local, local_provenance, _ = local_branch.extract(records, PROFILES, conditions)
    additional, reused = extra_global(records)
    quality = {}
    with np.load(ROOT / 'artifacts/round2_features.npz') as cache:
        for p in PROFILES:
            for c in DEV_CONDITIONS:
                quality[f'{p}/{c}'] = cache[f'quality/{p}/{c}']
    for profile in PROFILES:
        global_features[f'{profile}/jpeg70'] = additional[f'{profile}/jpeg70']
        quality[f'{profile}/jpeg70'] = additional[f'quality/{profile}/jpeg70']
    all_inputs, predictions, validation_predictions, metrics, curves, search = [], [], [], [], [], []
    models, settings, experts = {}, {}, {}
    y = np.array([r['label'] for r in records])
    for profile in PROFILES:
        train = ids['expert_train']
        ghead = pilot.train_head(global_features[f'{profile}/clean'][train], y[train])
        lhead = pilot.train_head(local[f'{profile}/clean'].mean(axis=1)[train], y[train])
        experts[profile] = {'global': ghead, 'local': lhead}
        batches = {name: bundle(records, ids[name], DEV_CONDITIONS if name == 'development' else TRAIN_CONDITIONS,
                                profile, global_features, local, quality, ghead, lhead)
                   for name in ['gate_train', 'validation', 'development']}
        for name, batch in batches.items():
            all_inputs.extend(dict(r, profile=profile, partition=name) for r in batch['rows'])
        val, dev = batches['validation'], batches['development']
        candidates = []
        for a in np.arange(11) / 10:
            score = validation_macro(val, a * val['zg'] + (1 - a) * val['zl'])
            candidates.append((float(a), score))
            search.append(dict(profile=profile, alpha=float(a), validation_macro_auroc=score))
        best = max(s for _, s in candidates)
        fixed = min((a for a, s in candidates if np.isclose(s, best)), key=lambda a: (abs(a - .5), a))
        settings[profile] = {'fixed_alpha': fixed, 'models': {}}
        fitted = [(name, -1, np.full(len(val['y']), a), np.full(len(dev['y']), a))
                  for name, a in [('global', 1.), ('local', 0.), ('average', .5), ('fixed', fixed)]]
        for seed in gates.SEEDS:
            for kind in ['score_gate', 'quality_gate']:
                model, scaler, artifact, history = gates.fit(batches['gate_train'], val, kind, seed)
                key = f'{profile}/{kind}/{seed}'
                models[key] = artifact
                curves.extend(dict(row, profile=profile, model=kind, seed=seed) for row in history)
                fitted.append((kind, seed, gates.predict(model, scaler, val['x'], kind), gates.predict(model, scaler, dev['x'], kind)))
                print(f'{key}: epoch={artifact["best_epoch"]}, val BCE={artifact["best_validation_bce"]:.4f}', flush=True)
        for name, seed, va, da in fitted:
            vp = expit(va * val['zg'] + (1 - va) * val['zl'])
            dp = expit(da * dev['zg'] + (1 - da) * dev['zl'])
            threshold = pilot.choose_threshold(val['y'], vp)
            settings[profile]['models'][f'{name}/{seed}'] = {'threshold': threshold}
            for target, batch, alphas, scores in [(validation_predictions, val, va, vp), (predictions, dev, da, dp)]:
                for r, alpha, score in zip(batch['rows'], alphas, scores):
                    target.append(dict(profile=profile, model=name, seed=seed, source_id=r['source_id'], generator=r['generator'],
                                       label=r['label'], condition=r['condition'], alpha_global=float(alpha), probability_fake=float(score)))
            for condition in DEV_CONDITIONS:
                for generator in ['SD15', 'BigGAN', 'ADM']:
                    mask = np.array([r['condition'] == condition and r['generator'] in ('Real', generator) for r in dev['rows']])
                    yy, pp, aa = dev['y'][mask], dp[mask], da[mask]
                    metrics.append(dict(profile=profile, model=name, seed=seed, condition=condition, generator=generator,
                                        n=len(yy), threshold=threshold, auroc=float(roc_auc_score(yy, pp)),
                                        average_precision=float(average_precision_score(yy, pp)),
                                        balanced_accuracy=float(balanced_accuracy_score(yy, pp >= threshold)),
                                        real_false_positive_rate=float(np.mean(pp[yy == 0] >= threshold)),
                                        mean_alpha_global=float(aa.mean()), std_alpha_global=float(aa.std())))
        print(f'Completed profile {profile}; fixed alpha={fixed:.1f}', flush=True)
    for name, rows in [('branch_inputs', all_inputs), ('predictions', predictions), ('validation_predictions', validation_predictions),
                       ('metrics', metrics), ('training_curves', curves), ('weight_search', search)]:
        r2.dump_csv(OUT / f'{name}.csv', rows)
    r2.write_json(OUT / 'settings.json', settings)
    r2.write_json(OUT / 'gate_models.json', models)
    joblib.dump(experts, ROOT / 'artifacts/round4_experts.joblib')
    paths = [ROOT / 'artifacts/round2_features.npz', ROOT / 'artifacts/round4_global_jpeg70.npz']
    paths += [ROOT / 'artifacts' / f'round3_local_{p}_{c}.npz' for p in PROFILES for c in conditions]
    r2.write_json(OUT / 'run_metadata.json', {
        'status': 'completed', 'run_seconds': time.perf_counter() - start,
        'new_global_cache_reused': reused, 'local_cache_provenance': local_provenance,
        'code_sha256': {name: data.sha256(ROOT / 'scripts' / name) for name in
                       ['run_round4.py', 'gate_models.py', 'run_round3.py', 'local_branch.py', 'run_round2.py', 'run_pilot.py']},
        'protocol_sha256': data.sha256(OUT / 'protocol.json'), 'assignments_sha256': data.sha256(OUT / 'assignments.json'),
        'feature_sha256': {p.relative_to(ROOT).as_posix(): data.sha256(p) for p in paths},
        'environment': {name: version(name) for name in ['torch', 'numpy', 'Pillow', 'transformers', 'scikit-learn', 'scipy']}})
    print('Round four complete. Run scripts/verify_round4.py.', flush=True)


if __name__ == '__main__':
    main()
