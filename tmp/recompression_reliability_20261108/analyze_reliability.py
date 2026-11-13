"""Locked, no-fit validation of sequential JPEG95 stability as expert reliability."""
import sys
sys.dont_write_bytecode = True
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import joblib
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/analysis/recompression_reliability_20261108'
OLD = ROOT / 'output/analysis/multigen_gate_20261108/run_v1_codefix1'
CONDS = ['clean', 'jpeg70', 'jpeg50', 'resize05']
TUNE = ['clean', 'jpeg70', 'resize05']
SEEDS = [20261101, 20261102, 20261103]
GROUPS = ['Real', 'ADM', 'BigGAN']
METHODS = ['stability_scaled', 'stability_raw', 'stability_probability', 'score_gate_reference']
WEIGHTS = {'Real': .5, 'ADM': .25, 'BigGAN': .25}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(name, data):
    with (OUT / name).open('x', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)


def csvsave(name, rows):
    with (OUT / name).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def auc(y, s):
    if len(np.unique(y)) != 2:
        return None
    return float(roc_auc_score(y, s))


def weighted_boot_auc(y, s, multiplicities):
    """Tie-aware AUC for all identity bootstrap draws, using sorted score groups."""
    if len(y) == 0:
        return np.full(len(multiplicities), np.nan)
    order = np.argsort(s, kind='stable')
    sorted_scores = s[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0) + 1]
    w = multiplicities[:, order]
    pos = np.add.reduceat(w * y[order], starts, axis=1)
    neg = np.add.reduceat(w * (1-y[order]), starts, axis=1)
    denom = pos.sum(axis=1) * neg.sum(axis=1)
    numerator = (pos * (np.cumsum(neg, axis=1) - .5*neg)).sum(axis=1)
    result = np.full(len(w), np.nan)
    np.divide(numerator, denom, out=result, where=denom > 0)
    return result


def gate_alpha(art, x):
    scaled = ((x-np.asarray(art['mean'])) / np.asarray(art['scale'])).astype(np.float32)
    scaled[:, 5:] = 0
    state = {k: np.asarray(v, np.float32) for k, v in art['state'].items()}
    return expit(np.maximum(scaled @ state['hidden.weight'].T + state['hidden.bias'], 0)
                 @ state['output.weight'].T + state['output.bias']).ravel().astype(float)


def mean_or_none(values):
    return None if any(v is None for v in values) else float(np.mean(values))


def interval(values):
    valid = np.isfinite(values)
    return dict(valid_draws=int(valid.sum()), unavailable_draws=int((~valid).sum()),
                conditional_percentile_95=np.quantile(values[valid], [.025, .975]).tolist()
                if valid.any() else None)


def main():
    assert not (OUT / 'analysis_complete.json').exists(), 'Do not overwrite completed analysis'
    protocol = read(OUT / 'protocol.json')
    assert sha(Path(__file__)) == protocol['analysis_sha256']
    assert (OUT / 'extraction_complete.json').exists(), 'Paired extraction not complete'
    extraction = read(OUT / 'extraction_complete.json')
    assert extraction['status'] == 'completed' and extraction['image_condition_pairs'] == 5604
    manifest = read(OUT / 'feature_manifest.json')
    assert sha(OUT / 'feature_manifest.json') == extraction['manifest_sha256']
    assert manifest['protocol_sha256'] == sha(OUT / 'protocol.json')
    expected_features = {e['path']:e for e in manifest['entries']}
    splits = read(OLD / 'splits.json')
    settings = read(OLD / 'model_settings.json')
    old_complete = read(OLD / 'complete.json')
    assert old_complete['status'] == 'completed'
    assert all(sha(OLD/name) == old_complete['outputs_sha256'][name] for name in ['splits.json','model_settings.json'])
    input_hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in
                    [OUT / 'protocol.json', OUT / 'extraction_complete.json', OUT / 'feature_manifest.json', OLD / 'splits.json', OLD / 'model_settings.json']}
    spec = importlib.util.spec_from_file_location('oldcache', ROOT / 'tmp/gate_diagnosis/original_gate_views/run_original_gate_views.py')
    cache = importlib.util.module_from_spec(spec); spec.loader.exec_module(cache)
    sources = cache.verify_and_collect()
    meta = {}
    for name in ['fitting_pool_manifest.jsonl', 'development_manifest.jsonl']:
        for line in (ROOT / 'results/round7' / name).read_text().splitlines():
            r = json.loads(line); meta[r['source_id']] = r
    tune_union = set().union(*(set(splits['sources'][f'{s}/source_only']['tuning_ids']) for s in SEEDS))
    dev_ids = set().union(*(set(v) for v in splits['evaluation_ids'].values()))
    assert len(tune_union) == 668 and len(dev_ids) == 900 and tune_union.isdisjoint(dev_ids)
    features = {}
    for scope, conditions, wanted in [('source_tuning', TUNE, tune_union), ('development', CONDS, dev_ids)]:
        for condition in conditions:
            ids = []; arrays = {k: [] for k in ['base_g', 'base_l', 'probe_g', 'probe_l']}
            paths = sorted((OUT / 'features' / scope / condition).glob('*.npz'))
            assert paths, (scope, condition)
            for path in paths:
                entry = expected_features[path.relative_to(OUT).as_posix()]
                assert entry['scope'] == scope and entry['condition'] == condition and sha(path) == entry['sha256']
                input_hashes[path.relative_to(ROOT).as_posix()] = sha(path)
                with np.load(path, allow_pickle=False) as f:
                    assert f['ids'].tolist() == entry['ids']
                    metadata = json.loads(str(f['metadata']))
                    assert metadata['protocol_sha256'] == sha(OUT / 'protocol.json')
                    assert metadata['scope'] == scope and metadata['condition'] == condition and metadata['ids'] == entry['ids']
                    assert hashlib.sha256(f['ids'].tobytes()).hexdigest() == str(f['ids_sha256']) == entry['array_sha256']['ids']
                    ids.extend(f['ids'].tolist())
                    for k in arrays:
                        a = f[k].copy(); assert np.isfinite(a).all()
                        assert hashlib.sha256(a.tobytes()).hexdigest() == str(f[k+'_sha256']) == entry['array_sha256'][k]
                        arrays[k].append(a)
            assert len(ids) == len(set(ids)) and set(ids) == wanted
            order = np.argsort(ids); ids = [ids[i] for i in order]
            features[scope, condition] = (ids, {k: np.concatenate(v)[order] for k, v in arrays.items()})
    records = []; scales = []; cells = []; data = {}; detect = []; max_old_error = 0.
    for number, seed in enumerate(SEEDS, 1):
        source = sources[number]
        expert_path = ROOT / f'artifacts/round8_a/seed{number}/matched/experts.joblib'
        experts = joblib.load(expert_path)
        fit = cache.reconstruct_original(source, number, 'gate_fit', experts)
        allowed = set(splits['sources'][f'{seed}/source_only']['fit_ids'])
        assert {r['source_id'] for r in fit['rows']} == allowed
        sigma_g, sigma_l = float(np.std(fit['zg'], ddof=0)), float(np.std(fit['zl'], ddof=0))
        assert np.isfinite(sigma_g+sigma_l) and min(sigma_g, sigma_l) > 0
        scales.append(dict(run_seed=seed,fit_originals=len(allowed),fit_views=len(fit['y']),sigma_global=sigma_g,sigma_local=sigma_l))
        setting = settings[f'{seed}/source_only']
        old_rows = {}
        for oldpart in ['selection', 'development']:
            b = source['matched']['raw_batches'][oldpart]
            for i, r in enumerate(b['rows']):
                old_rows[r['source_id'], r['condition']] = (b['zg'][i], b['zl'][i])
        for scope, conditions in [('source_tuning', TUNE), ('development', CONDS)]:
            for condition in conditions:
                all_ids, all_features = features[scope, condition]
                use_ids = set(splits['sources'][f'{seed}/source_only']['tuning_ids']) if scope == 'source_tuning' else dev_ids
                ix = [i for i, identity in enumerate(all_ids) if identity in use_ids]
                ids = [all_ids[i] for i in ix]; f = {k: v[ix] for k, v in all_features.items()}
                gh, lh = experts['global_head'], experts['local_head']
                zg = gh.decision_function(f['base_g']); zl = lh.decision_function(f['base_l'].mean(axis=1))
                pg = gh.decision_function(f['probe_g']); pl = lh.decision_function(f['probe_l'].mean(axis=1))
                variance = lh.decision_function(f['base_l'].reshape(-1,512)).reshape(-1,4).var(axis=1)
                x = cache.gates.inputs(zg, zl, variance, np.zeros((len(ids), 3)))
                alphas = np.stack([gate_alpha(setting['gates'][f'score_gate/{gs}']['auroc'], x) for gs in cache.gates.SEEDS])
                y = np.asarray([meta[i]['label'] for i in ids]); gen = np.asarray([meta[i]['generator'] for i in ids])
                advantage = (2*y-1)*(zg-zl); winner = (advantage > 0).astype(int)
                valid = advantage != 0
                dg, dl = np.abs(pg-zg), np.abs(pl-zl)
                signals = {'stability_scaled': dl/sigma_l-dg/sigma_g, 'stability_raw': dl-dg,
                           'stability_probability': np.abs(expit(pl)-expit(zl))-np.abs(expit(pg)-expit(zg))}
                gp = expit(zg) >= setting['baselines']['global']['threshold']
                lp = expit(zl) >= setting['baselines']['local']['threshold']
                gc, lc = gp == y, lp == y
                one = gc != lc
                a = np.where(signals['stability_scaled'] > 0, 1., np.where(signals['stability_scaled'] < 0, 0., .5))
                selected_correct = a*gc+(1-a)*lc
                selected_z = a*zg+(1-a)*zl
                for i, identity in enumerate(ids):
                    og, ol = old_rows[identity, condition]
                    max_old_error = max(max_old_error, abs(og-zg[i]), abs(ol-zl[i]))
                    records.append(dict(run_seed=seed,scope=scope,condition=condition,source_id=identity,generator=str(gen[i]),label=int(y[i]),
                        global_base=float(zg[i]),local_base=float(zl[i]),global_probe=float(pg[i]),local_probe=float(pl[i]),
                        **{k:float(v[i]) for k,v in signals.items()},relative_global_winner=int(winner[i]),relative_tie=bool(not valid[i]),
                        alpha_ref_0=float(alphas[0,i]),alpha_ref_1=float(alphas[1,i]),alpha_ref_2=float(alphas[2,i]),
                        global_correct=bool(gc[i]),local_correct=bool(lc[i]),one_correct=bool(one[i]),selected_correct=float(selected_correct[i])))
                for group in sorted(set(gen)):
                    mask = gen == group; ok = mask & valid
                    assert np.sum(mask) == (124 if scope == 'source_tuning' else 300)
                    cell = dict(run_seed=seed,scope=scope,condition=condition,generator=group,n=int(mask.sum()),
                        relative_ties=int((mask & ~valid).sum()),global_winners=int(winner[ok].sum()),local_winners=int((1-winner[ok]).sum()),
                        one_correct_n=int(one[mask].sum()),one_correct_coverage=float(one[mask].mean()),
                        stable_selector_one_correct_accuracy=float(selected_correct[mask & one].mean()) if (mask & one).any() else None,
                        always_global_one_correct_accuracy=float(gc[mask & one].mean()) if (mask & one).any() else None,
                        always_local_one_correct_accuracy=float(lc[mask & one].mean()) if (mask & one).any() else None,
                        whole_stable_selector_accuracy=float(selected_correct[mask].mean()),whole_global_accuracy=float(gc[mask].mean()),whole_local_accuracy=float(lc[mask].mean()),
                        abs_delta_global_mean=float(dg[mask].mean()),abs_delta_local_mean=float(dl[mask].mean()))
                    for method, values in signals.items():cell[method+'_auroc'] = auc(winner[ok], values[ok])
                    cell['score_gate_reference_auroc'] = mean_or_none([auc(winner[ok], aa[ok]) for aa in alphas])
                    cells.append(cell)
                    if scope == 'development':
                        current_ids = [ids[i] for i in np.flatnonzero(mask)]
                        assert current_ids == sorted(splits['evaluation_ids'][group])
                        data[seed, condition, group] = dict(winner=winner[mask],valid=valid[mask],signals={k:v[mask] for k,v in signals.items()},alphas=alphas[:,mask])
                fake_groups = ['SD15'] if scope == 'source_tuning' else ['ADM','BigGAN']
                for group in fake_groups:
                    mm = np.isin(gen, ['Real',group]); yy=y[mm]
                    detect.append(dict(run_seed=seed,scope=scope,condition=condition,generator=group,
                        stable_selected_logit_auroc=auc(yy,selected_z[mm]),average_fusion_auroc=auc(yy,(zg[mm]+zl[mm])/2),
                        global_auroc=auc(yy,zg[mm]),local_auroc=auc(yy,zl[mm]),
                        stable_selector_balanced_accuracy=float((selected_correct[mm & (y==0)].mean()+selected_correct[mm & (y==1)].mean())/2)))
    csvsave('source_scales.csv', scales); csvsave('predictions.csv', records)
    csvsave('reliability_cells.csv', cells); csvsave('detection_diagnostics.csv', detect)
    devcells = [c for c in cells if c['scope']=='development']
    macro = {}
    for method in METHODS:
        vals = [c[method+'_auroc'] for c in devcells]
        macro[method] = None if any(v is None for v in vals) else sum(WEIGHTS[c['generator']]*c[method+'_auroc'] for c in devcells)/12
    source_macro = {method:mean_or_none([c[method+'_auroc'] for c in cells if c['scope']=='source_tuning']) for method in METHODS}
    rng=np.random.default_rng(20261108)
    draws={g:rng.multinomial(300,np.repeat(1/300,300),size=1000) for g in GROUPS}
    boot={}; check_error=0.
    for key, d in data.items():
        ok=d['valid']; yy=d['winner'][ok]; counts=draws[key[2]][:,ok]
        ss=d['signals']['stability_scaled'][ok]
        bb=weighted_boot_auc(yy,ss,counts)
        rr=np.mean([weighted_boot_auc(yy,aa[ok],counts) for aa in d['alphas']],axis=0)
        for row in range(2):
            if np.isfinite(bb[row]):check_error=max(check_error,abs(bb[row]-roc_auc_score(yy,ss,sample_weight=counts[row])))
        boot[key]=(bb,rr)
    total=sum(WEIGHTS[g]*boot[s,c,g][0] for s in SEEDS for c in CONDS for g in GROUPS)/12
    ref=sum(WEIGHTS[g]*boot[s,c,g][1] for s in SEEDS for c in CONDS for g in GROUPS)/12
    boot_summary={'primary':interval(total),'reference':interval(ref),'primary_minus_reference':interval(total-ref),
                  'draws':1000,'seed':20261108,'paired_by_original_identity':True,'conditional_on_frozen_models_and_source_scales':True}
    group_summary={}
    for g in GROUPS:
        group_summary[g]={method:mean_or_none([c[method+'_auroc'] for c in devcells if c['generator']==g]) for method in METHODS}
        group_summary[g]['primary_interval']=interval(np.mean([boot[s,c,g][0] for s in SEEDS for c in CONDS],axis=0))
    jpeg_cells=[c for c in devcells if c['condition'].startswith('jpeg')]
    jpeg_macro=None if any(c['stability_scaled_auroc'] is None for c in jpeg_cells) else sum(WEIGHTS[c['generator']]*c['stability_scaled_auroc'] for c in jpeg_cells)/6
    primary=macro['stability_scaled']; ci=boot_summary['primary']['conditional_percentile_95']
    if primary is None or boot_summary['primary']['unavailable_draws']:
        decision='incomplete_estimation_no_full_success_claim'
    elif primary<=.5:decision='unsupported'
    elif ci[0]>.5 and all(group_summary[g]['stability_scaled']>.5 for g in ['ADM','BigGAN']) and jpeg_macro>.5:
        decision='supported_relative_loss_signal_only'
    else:decision='uncertain'
    assert check_error<1e-12,check_error
    input_hashes.update(cache.INPUT_HASHES)
    assert all(sha(ROOT/p)==h for p,h in input_hashes.items()),'Input changed during analysis'
    summary=dict(status='completed',decision=decision,primary_macro=macro,source_tuning_macro=source_macro,
        development_group_macro=group_summary,jpeg_primary_macro=jpeg_macro,bootstrap=boot_summary,
        max_bootstrap_metric_check_error=check_error,max_fresh_baseline_logit_difference_from_old_cache=max_old_error,
        no_models_fitted=True,no_sign_or_condition_selection=True,no_final_test_access=True,
        scope=protocol['scope'],input_sha256=input_hashes)
    save('summary.json',summary)
    save('analysis_complete.json',dict(status='completed',prediction_rows=len(records),reliability_cells=len(cells),
        outputs_sha256={name:sha(OUT/name) for name in ['source_scales.csv','predictions.csv','reliability_cells.csv','detection_diagnostics.csv','summary.json']}))
    print(json.dumps({k:summary[k] for k in ['decision','primary_macro','source_tuning_macro','development_group_macro','jpeg_primary_macro','bootstrap','max_fresh_baseline_logit_difference_from_old_cache']},indent=2))


if __name__=='__main__':
    main()
