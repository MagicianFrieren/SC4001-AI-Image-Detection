"""Independent read-only verification after the frozen reliability analysis completes.

No analysis/extractor module imports, fitting, image decoding, or final-test access.
AUROC and bootstrap are reconstructed through explicit positive/negative pairs.
"""
import sys
sys.dont_write_bytecode = True
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/analysis/recompression_reliability_20261108'
OLD = ROOT / 'output/analysis/multigen_gate_20261108/run_v1_codefix1'
SEEDS = [20261101, 20261102, 20261103]
GATE_SEEDS = [20261022, 20261023, 20261024]
CONDITIONS = ['clean', 'jpeg70', 'jpeg50', 'resize05']
TUNING = ['clean', 'jpeg70', 'resize05']
GROUPS = ['Real', 'ADM', 'BigGAN']
METHODS = ['stability_scaled', 'stability_raw', 'stability_probability', 'score_gate_reference']
WEIGHTS = {'Real': .5, 'ADM': .25, 'BigGAN': .25}
INPUTS = {}
ERRORS = {}


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def checked(path, expected=None):
    path = Path(path).resolve()
    rel = path.relative_to(ROOT).as_posix()
    assert not any(word in rel.lower() for word in ['reserved', 'final']), rel
    digest = sha(path)
    assert expected is None or digest == expected, ('digest', rel)
    INPUTS[rel] = digest
    return path


def read(path, expected=None):
    return json.loads(checked(path, expected).read_text(encoding='utf-8'))


def rows(path, expected=None):
    with checked(path, expected).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def close(name, actual, expected, atol=1e-10):
    if actual is None or expected is None or expected == '':
        assert actual is None and expected in (None, ''), (name, actual, expected)
        return
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == b.shape, (name, a.shape, b.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all(), name
    error = float(np.max(np.abs(a-b))) if a.size else 0.
    ERRORS[name] = max(ERRORS.get(name, 0.), error)
    assert error <= atol, (name, error, atol)


def pair_auc(target, score):
    target, score = np.asarray(target), np.asarray(score)
    pos, neg = score[target == 1], score[target == 0]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg) + .5*(pos[:, None] == neg)).mean())


def pair_boot(target, score, counts):
    """Direct pair matrix; does not use analysis's sorted-group cumulative algorithm."""
    pos, neg = np.flatnonzero(target == 1), np.flatnonzero(target == 0)
    answer = np.full(len(counts), np.nan)
    if not len(pos) or not len(neg):
        return answer
    p, n = score[pos], score[neg]
    wins = (p[:, None] > n).astype(float) + .5*(p[:, None] == n)
    cp, cn = counts[:, pos].astype(float), counts[:, neg].astype(float)
    denominator = cp.sum(1)*cn.sum(1)
    numerator = ((cp @ wins)*cn).sum(1)
    np.divide(numerator, denominator, out=answer, where=denominator > 0)
    return answer


def mean_available(values):
    return None if any(v is None for v in values) else float(np.mean(values))


def interval(values):
    finite = np.isfinite(values)
    return {'valid_draws': int(finite.sum()), 'unavailable_draws': int((~finite).sum()),
            'conditional_percentile_95': np.percentile(values[finite], [2.5, 97.5]).tolist() if finite.any() else None}


def compare_interval(name, values, expected):
    got = interval(values)
    assert got['valid_draws'] == expected['valid_draws']
    assert got['unavailable_draws'] == expected['unavailable_draws']
    close(name, got['conditional_percentile_95'], expected['conditional_percentile_95'])
    return got


def gate_reference(zg, zl, local_views, head, setting):
    pg, pl = expit(zg), expit(zl)
    entropies = []
    for p in (pg, pl):
        p = np.clip(p, 1e-12, 1-1e-12)
        entropies.append(-p*np.log(p)-(1-p)*np.log1p(-p))
    variance = head.decision_function(local_views.reshape(-1,512)).reshape(-1,4).var(1)
    x = np.zeros((len(zg),8), dtype=float)
    x[:, :5] = np.column_stack((pg, pl, abs(pg-pl), (entropies[0]+entropies[1])/2, np.log1p(variance)))
    result = []
    for seed in GATE_SEEDS:
        art = setting['gates'][f'score_gate/{seed}']['auroc']
        assert art['kind'] == 'score_gate' and art['gate_seed'] == seed
        a = ((x-np.array(art['mean']))/np.array(art['scale'])).astype(np.float32)
        a[:, 5:] = 0
        state = {k: np.asarray(v, dtype=np.float32) for k,v in art['state'].items()}
        hidden = np.maximum(a.dot(state['hidden.weight'].T)+state['hidden.bias'], 0)
        result.append(expit(hidden.dot(state['output.weight'].T)+state['output.bias']).ravel().astype(float))
    return np.stack(result)


def old_source_scales(number, seed, wanted, heads):
    complete = read(ROOT/f'results/round8_a/seed{number}/unmatched/complete.json')
    rel = f'artifacts/round8_a/seed{number}/unmatched/bundle.json'
    bundle = read(ROOT/rel, complete['files_sha256'][rel])
    scores = {'g': [], 'l': []}
    for condition in TUNING:
        ids, blocks = [], {'g': [], 'l': []}
        for path in sorted((ROOT/f'artifacts/round8_a/seed{number}/features/gate_fit_{condition}_0').glob('*.npz')):
            checked(path, bundle['feature_sha256'][path.relative_to(ROOT).as_posix()])
            with np.load(path, allow_pickle=False) as block:
                ordered = block['ids'].tolist()
                assert json.loads(str(block['identity'])) == [seed, 'gate_fit', condition, False, ordered]
                assert str(block['signature']) == bundle['signature']
                ids.extend(ordered)
                for key in blocks:
                    assert hashlib.sha256(block[key].tobytes()).hexdigest() == str(block[key+'_sha256'])
                    blocks[key].append(block[key].copy())
        assert ids == sorted(wanted) and len(ids) == 500
        scores['g'].append(heads['global_head'].decision_function(np.concatenate(blocks['g'])))
        scores['l'].append(heads['local_head'].decision_function(np.concatenate(blocks['l']).mean(1)))
    return np.array([np.std(np.concatenate(scores[k]), ddof=0) for k in ['g','l']])


def main():
    start = time.monotonic()
    assert (OUT/'analysis_complete.json').is_file(), 'Wait for original analysis completion; no premature outcome inspection.'
    assert not (OUT/'verification.json').exists(), 'Preserve completed verification.'
    marker = read(OUT/'analysis_complete.json')
    assert marker['status'] == 'completed'
    protocol = read(OUT/'protocol.json')
    checked(ROOT/'tmp/recompression_reliability_20261108/analyze_reliability.py', protocol['analysis_sha256'])
    checked(ROOT/'tmp/recompression_reliability_20261108/extract_recompression.py', protocol['extraction']['extractor_sha256'])
    preflight = read(OUT/'preflight.json', protocol['preflight_sha256'])
    read(OUT/'canary.json', protocol['canary_sha256'])
    extraction = read(OUT/'extraction_complete.json')
    assert extraction['status'] == 'completed' and extraction['image_condition_pairs'] == 5604
    freeze = read(OUT/'extraction_freeze.json', extraction['extraction_freeze_sha256'])
    assert freeze['protocol_sha256'] == sha(OUT/'protocol.json')
    manifest = read(OUT/'feature_manifest.json', extraction['manifest_sha256'])
    assert manifest['status'] == 'complete' and manifest['protocol_sha256'] == freeze['protocol_sha256']
    assert manifest['runtime'] == freeze['runtime']
    for provenance in [preflight['input_sha256'], freeze['input_sha256']]:
        for relative, expected in provenance.items():
            checked(ROOT/relative, expected)
    old_complete = read(OLD/'complete.json')
    splits = read(OLD/'splits.json', old_complete['outputs_sha256']['splits.json'])
    assert sha(OLD/'splits.json') == protocol['split_sha256']
    settings = read(OLD/'model_settings.json', old_complete['outputs_sha256']['model_settings.json'])
    truth = {}
    for name in ['fitting_pool_manifest.jsonl', 'development_manifest.jsonl']:
        for line in checked(ROOT/'results/round7'/name).read_text(encoding='utf-8').splitlines():
            row = json.loads(line); truth[row['source_id']] = row
    isolation = read(ROOT/'results/round7/step_b_isolation.json')
    deny = set(isolation['reserved_test_source_ids'])
    source = set().union(*(set(splits['sources'][f'{s}/source_only']['tuning_ids']) for s in SEEDS))
    dev = set().union(*(set(v) for v in splits['evaluation_ids'].values()))
    assert len(source) == 668 and len(dev) == 900 and source.isdisjoint(dev)
    assert source|dev <= set(isolation['allowed_source_ids']) and (source|dev).isdisjoint(deny)
    assert set(freeze['source_ids']) == source and set(freeze['development_ids']) == dev
    originals = {e['source_id']:e for e in preflight['original_cache']['entries']}
    for identity in source|dev:
        checked(ROOT/originals[identity]['path'], truth[identity]['source_bytes_sha256'])
    feature_rows = {}
    entries = manifest['entries']
    assert len(entries) == extraction['feature_batches'] == 179
    assert {e['path'] for e in entries} == {p.relative_to(OUT).as_posix() for p in (OUT/'features').rglob('*.npz')}
    runtime_hash = hashlib.sha256(json.dumps(freeze['runtime'], sort_keys=True).encode()).hexdigest()
    for entry in entries:
        path = checked(OUT/entry['path'], entry['sha256'])
        with np.load(path, allow_pickle=False) as batch:
            metadata = json.loads(str(batch['metadata']))
            identities, scope, condition = batch['ids'].tolist(), entry['scope'], entry['condition']
            assert identities == entry['ids'] == metadata['ids'] and len(identities) == entry['n']
            assert metadata['scope'] == scope and metadata['condition'] == condition
            assert condition in (TUNING if scope == 'source_tuning' else CONDITIONS)
            assert metadata['protocol_sha256'] == freeze['protocol_sha256']
            assert metadata['extractor_sha256'] == protocol['extraction']['extractor_sha256']
            assert metadata['runtime_sha256'] == runtime_hash
            assert metadata['original_source_bytes_sha256'] == [truth[i]['source_bytes_sha256'] for i in identities]
            assert metadata['original_pixel_sha256'] == [truth[i]['pixel_sha256'] for i in identities]
            assert metadata['probe'] == dict(quality=95,subsampling=0,optimize=False,progressive=False)
            assert metadata['view_seeds'] == [202610220,202610221,202610222,202610223]
            assert metadata['fresh_paired_same_batch_order'] is True
            assert metadata['chain'] == ['original_decode_exif_rgb','ImageOps.fit_224_bicubic',condition,
                                         'probe_only_JPEG95_subsampling0','global_or_four_PatchShuffle56_views']
            for key in ['ids','base_g','base_l','probe_g','probe_l']:
                assert hashlib.sha256(batch[key].tobytes()).hexdigest() == str(batch[key+'_sha256']) == entry['array_sha256'][key]
                if key != 'ids':
                    expected_shape = (len(identities),512) if key.endswith('_g') else (len(identities),4,512)
                    assert batch[key].shape == expected_shape and batch[key].dtype == np.float32
                    assert np.isfinite(batch[key]).all()
            for index, identity in enumerate(identities):
                key = scope,condition,identity
                assert key not in feature_rows
                feature_rows[key] = {k:batch[k][index].copy() for k in ['base_g','base_l','probe_g','probe_l']}
    expected_keys = {('source_tuning',c,i) for c in TUNING for i in source} | {('development',c,i) for c in CONDITIONS for i in dev}
    assert set(feature_rows) == expected_keys and len(feature_rows) == 5604
    summary = read(OUT/'summary.json', marker['outputs_sha256']['summary.json'])
    predictions = rows(OUT/'predictions.csv', marker['outputs_sha256']['predictions.csv'])
    reported_cells = rows(OUT/'reliability_cells.csv', marker['outputs_sha256']['reliability_cells.csv'])
    reported_detection = rows(OUT/'detection_diagnostics.csv', marker['outputs_sha256']['detection_diagnostics.csv'])
    reported_scales = rows(OUT/'source_scales.csv', marker['outputs_sha256']['source_scales.csv'])
    pred = {(int(r['run_seed']),r['scope'],r['condition'],r['source_id']):r for r in predictions}
    cell_table = {(int(r['run_seed']),r['scope'],r['condition'],r['generator']):r for r in reported_cells}
    detection_table = {(int(r['run_seed']),r['scope'],r['condition'],r['generator']):r for r in reported_detection}
    assert len(pred) == len(predictions) == marker['prediction_rows'] == 13032
    assert len(cell_table) == len(reported_cells) == marker['reliability_cells'] == 54
    assert len(detection_table) == len(reported_detection) == 33
    computed, boot_inputs, seen_predictions, seen_detection = {}, {}, set(), set()
    fresh_old_max = 0.
    for number,seed in enumerate(SEEDS,1):
        partition = splits['sources'][f'{seed}/source_only']
        fit_ids, tune_ids = set(partition['fit_ids']), set(partition['tuning_ids'])
        assert fit_ids.isdisjoint(tune_ids) and (fit_ids|tune_ids).isdisjoint(dev)
        assert Counter(truth[i]['generator'] for i in fit_ids) == Counter(Real=250,SD15=250)
        assert Counter(truth[i]['generator'] for i in tune_ids) == Counter(Real=124,SD15=124)
        complete = read(ROOT/f'results/round8_a/seed{number}/matched/complete.json')
        bundle_relative = f'artifacts/round8_a/seed{number}/matched/bundle.json'
        old_bundle = read(ROOT/bundle_relative, complete['files_sha256'][bundle_relative])
        head_relative = f'artifacts/round8_a/seed{number}/matched/experts.joblib'
        assert old_bundle['experts_sha256'] == complete['files_sha256'][head_relative]
        heads = joblib.load(checked(ROOT/head_relative, complete['files_sha256'][head_relative]))
        old_logits = {}
        for part in ['selection','development']:
            batch = old_bundle['raw_batches'][part]
            for index,row in enumerate(batch['rows']):
                old_logits[row['source_id'],row['condition']] = batch['zg'][index],batch['zl'][index]
        scales = old_source_scales(number, seed, fit_ids, heads)
        reported = next(r for r in reported_scales if int(r['run_seed']) == seed)
        assert int(reported['fit_originals']) == 500 and int(reported['fit_views']) == 1500
        close('source_scale', scales, [reported['sigma_global'],reported['sigma_local']])
        setting = settings[f'{seed}/source_only']
        for scope,conditions,scope_ids in [('source_tuning',TUNING,tune_ids),('development',CONDITIONS,dev)]:
            ids = sorted(scope_ids)
            labels = np.array([truth[i]['label'] for i in ids])
            generators = np.array([truth[i]['generator'] for i in ids])
            for condition in conditions:
                features = {k:np.stack([feature_rows[scope,condition,i][k] for i in ids]) for k in ['base_g','base_l','probe_g','probe_l']}
                gh,lh = heads['global_head'],heads['local_head']
                zg,zl = gh.decision_function(features['base_g']),lh.decision_function(features['base_l'].mean(1))
                pg,pl = gh.decision_function(features['probe_g']),lh.decision_function(features['probe_l'].mean(1))
                expected_old = np.asarray([old_logits[i,condition] for i in ids])
                fresh_old_max = max(fresh_old_max,float(abs(zg-expected_old[:,0]).max()),float(abs(zl-expected_old[:,1]).max()))
                dg,dl = abs(pg-zg),abs(pl-zl)
                signals = dict(stability_scaled=dl/scales[1]-dg/scales[0], stability_raw=dl-dg,
                               stability_probability=abs(expit(pl)-expit(zl))-abs(expit(pg)-expit(zg)))
                # BCE uses the actual label and base logits; correctness uses frozen thresholds separately.
                global_bce = np.logaddexp(0,-(2*labels-1)*zg)
                local_bce = np.logaddexp(0,-(2*labels-1)*zl)
                winner = (global_bce < local_bce).astype(int)
                valid = zg != zl
                assert np.array_equal(winner[valid], ((2*labels-1)*(zg-zl)>0)[valid])
                alphas = gate_reference(zg,zl,features['base_l'],lh,setting)
                gc = (expit(zg) >= setting['baselines']['global']['threshold']) == labels
                lc = (expit(zl) >= setting['baselines']['local']['threshold']) == labels
                one = gc != lc
                global_choice = signals['stability_scaled'] > 0
                local_choice = signals['stability_scaled'] < 0
                selected_correct = np.where(global_choice,gc,np.where(local_choice,lc,(gc.astype(float)+lc)/2))
                selected_logit = np.where(global_choice,zg,np.where(local_choice,zl,(zg+zl)/2))
                for index,identity in enumerate(ids):
                    key=seed,scope,condition,identity; row=pred[key]; seen_predictions.add(key)
                    assert row['generator'] == truth[identity]['generator'] and int(row['label']) == labels[index]
                    for name,values in dict(global_base=zg,local_base=zl,global_probe=pg,local_probe=pl,**signals,
                                            alpha_ref_0=alphas[0],alpha_ref_1=alphas[1],alpha_ref_2=alphas[2],selected_correct=selected_correct).items():
                        close('prediction_'+name,values[index],row[name])
                    assert int(row['relative_global_winner']) == winner[index]
                    for name,value in [('relative_tie',not valid[index]),('global_correct',gc[index]),('local_correct',lc[index]),('one_correct',one[index])]:
                        assert row[name] == str(bool(value)), (name,key)
                for group in sorted(set(generators)):
                    key=seed,scope,condition,group; row=cell_table[key]; mask=generators==group; ok=mask&valid
                    assert int(row['n']) == mask.sum() == (124 if scope=='source_tuning' else 300)
                    for name,value in dict(relative_ties=(mask&~valid).sum(),global_winners=winner[ok].sum(),local_winners=(1-winner[ok]).sum(),one_correct_n=one[mask].sum()).items():
                        assert int(row[name]) == value
                    numeric=dict(one_correct_coverage=one[mask].mean(),
                        stable_selector_one_correct_accuracy=float(selected_correct[mask&one].mean()) if (mask&one).any() else None,
                        always_global_one_correct_accuracy=float(gc[mask&one].mean()) if (mask&one).any() else None,
                        always_local_one_correct_accuracy=float(lc[mask&one].mean()) if (mask&one).any() else None,
                        whole_stable_selector_accuracy=selected_correct[mask].mean(),whole_global_accuracy=gc[mask].mean(),whole_local_accuracy=lc[mask].mean(),
                        abs_delta_global_mean=dg[mask].mean(),abs_delta_local_mean=dl[mask].mean())
                    metrics={name:pair_auc(winner[ok],value[ok]) for name,value in signals.items()}
                    metrics['score_gate_reference']=mean_available([pair_auc(winner[ok],a[ok]) for a in alphas])
                    numeric.update({k+'_auroc':v for k,v in metrics.items()})
                    for name,value in numeric.items():close('cell_'+name,value,row[name])
                    computed[key]=metrics
                    if scope=='development':
                        assert [ids[j] for j in np.flatnonzero(mask)] == sorted(splits['evaluation_ids'][group])
                        boot_inputs[seed,condition,group]=(winner[mask],valid[mask],signals['stability_scaled'][mask],alphas[:,mask])
                for group in (['SD15'] if scope=='source_tuning' else ['ADM','BigGAN']):
                    key=seed,scope,condition,group; row=detection_table[key]; seen_detection.add(key)
                    mask=np.isin(generators,['Real',group]); real=mask&(labels==0); fake=mask&(labels==1)
                    numeric=dict(stable_selected_logit_auroc=pair_auc(labels[mask],selected_logit[mask]),
                        average_fusion_auroc=pair_auc(labels[mask],((zg+zl)/2)[mask]),global_auroc=pair_auc(labels[mask],zg[mask]),local_auroc=pair_auc(labels[mask],zl[mask]),
                        stable_selector_balanced_accuracy=(selected_correct[real].mean()+selected_correct[fake].mean())/2)
                    for name,value in numeric.items():close('detection_'+name,value,row[name])
    assert seen_predictions == set(pred) and seen_detection == set(detection_table) and set(computed) == set(cell_table)
    close('max_fresh_old_baseline_difference',fresh_old_max,summary['max_fresh_baseline_logit_difference_from_old_cache'])
    dev_metrics = {k:v for k,v in computed.items() if k[1]=='development'}
    primary_macro={}
    for method in METHODS:
        values=[v[method] for v in dev_metrics.values()]
        primary_macro[method]=None if any(v is None for v in values) else sum(WEIGHTS[k[3]]*v[method] for k,v in dev_metrics.items())/(len(SEEDS)*len(CONDITIONS))
        close('primary_macro_'+method,primary_macro[method],summary['primary_macro'][method])
        close('source_macro_'+method,mean_available([v[method] for k,v in computed.items() if k[1]=='source_tuning']),summary['source_tuning_macro'][method])
        for group in GROUPS:
            close('group_macro_'+method,mean_available([v[method] for k,v in dev_metrics.items() if k[3]==group]),summary['development_group_macro'][group][method])
    jpeg=[(k,v) for k,v in dev_metrics.items() if k[2] in ['jpeg70','jpeg50']]
    jpeg_macro=None if any(v['stability_scaled'] is None for k,v in jpeg) else sum(WEIGHTS[k[3]]*v['stability_scaled'] for k,v in jpeg)/(len(SEEDS)*2)
    close('jpeg_macro',jpeg_macro,summary['jpeg_primary_macro'])
    # Entire 1000 draws, preserving the exact ID pairing and class-group multiplicities.
    rng=np.random.default_rng(20261108)
    draws={g:rng.multinomial(300,np.full(300,1/300),size=1000) for g in GROUPS}
    boot={}
    for key,(target,valid,signal,alphas) in boot_inputs.items():
        target=target[valid]; count=draws[key[2]][:,valid]
        values=pair_boot(target,signal[valid],count)
        references=np.mean([pair_boot(target,a[valid],count) for a in alphas],axis=0)
        for draw in [0,499,999]:
            if np.isfinite(values[draw]):
                close('bootstrap_sklearn_crosscheck',values[draw],roc_auc_score(target,signal[valid],sample_weight=count[draw]),atol=1e-12)
        boot[key]=(values,references)
    primary=sum(WEIGHTS[g]*boot[s,c,g][0] for s in SEEDS for c in CONDITIONS for g in GROUPS)/12
    reference=sum(WEIGHTS[g]*boot[s,c,g][1] for s in SEEDS for c in CONDITIONS for g in GROUPS)/12
    intervals={name:compare_interval('bootstrap_'+name,value,summary['bootstrap'][name]) for name,value in
               [('primary',primary),('reference',reference),('primary_minus_reference',primary-reference)]}
    for group in GROUPS:
        compare_interval('group_bootstrap_'+group,np.mean([boot[s,c,group][0] for s in SEEDS for c in CONDITIONS],axis=0),summary['development_group_macro'][group]['primary_interval'])
    estimate=primary_macro['stability_scaled']; ci=intervals['primary']['conditional_percentile_95']
    if estimate is None or intervals['primary']['unavailable_draws']:
        decision='incomplete_estimation_no_full_success_claim'
    elif estimate<=.5:decision='unsupported'
    elif ci[0]>.5 and all(summary['development_group_macro'][g]['stability_scaled']>.5 for g in ['ADM','BigGAN']) and jpeg_macro>.5:
        decision='supported_relative_loss_signal_only'
    else:decision='uncertain'
    assert decision == summary['decision']
    for relative,digest in summary['input_sha256'].items():checked(ROOT/relative,digest)
    assert all(sha(ROOT/p)==digest for p,digest in INPUTS.items())
    report=dict(status='passed',verifier_sha256=sha(__file__),seconds=time.monotonic()-start,
        verified_feature_batches=179,verified_image_condition_pairs=5604,verified_prediction_rows=len(pred),verified_reliability_cells=len(computed),
        verified_detection_cells=len(seen_detection),bootstrap_draws_independently_replayed=1000,
        bootstrap_implementation='Explicit positive/negative pair dominance matrices; sklearn weighted AUROC cross-check on 3 draws per cell.',
        decision=decision,primary_macro=primary_macro,bootstrap=intervals,max_absolute_errors=ERRORS,
        source_partition_isolation=True,source_scales_rebuilt_from_original_gate_fit=True,
        no_training=True,no_image_decode_or_extraction=True,no_final_test_access=True,
        limitations=['Verifies cached embeddings and producer provenance, not independent CLIP re-extraction.',
                    'Confidence intervals condition on frozen heads and source scales; three data seeds share fitting pool and all development IDs.',
                    'The reviewed development generators were inspected before this hypothesis; no new blind-test claim.',
                    'Relative BCE winner differs from binary correctness; selected-logit AUC and threshold-routing BA have distinct tie semantics as frozen.'],
        input_sha256=INPUTS)
    with (OUT/'verification.json').open('x',encoding='utf-8') as handle:
        json.dump(report,handle,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ['status','seconds','verified_prediction_rows','verified_reliability_cells','bootstrap_draws_independently_replayed','decision','primary_macro','bootstrap']},indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
