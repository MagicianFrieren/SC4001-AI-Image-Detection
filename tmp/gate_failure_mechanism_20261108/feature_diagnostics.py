"""Frozen-model quality-feature mechanism probes; no training or selection."""
import sys
sys.dont_write_bytecode = True
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'output/analysis/multigen_gate_20261108/run_v1_codefix1'
OUT = ROOT / 'output/analysis/gate_failure_mechanism_20261108'
SEEDS = (20261101,20261102,20261103)
INITS = (20261022,20261023,20261024)
CONDS = ('clean','jpeg70','jpeg50','resize05')
TUNE = ('clean','jpeg70','resize05')
QNAMES = ('log_laplacian_variance','high_frequency_energy_ratio','eight_pixel_boundary_ratio')
VARIANTS = ('unchanged','q0_training_mean','q1_training_mean','q2_training_mean','all_q_training_mean','all_q_from_same_image_clean')
INPUTS = {}


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def checked(p, expected=None):
    rel=p.relative_to(ROOT).as_posix()
    assert 'final' not in rel.lower() and 'reserved' not in rel.lower(),rel
    value=sha(p)
    assert expected is None or value==expected,rel
    INPUTS[rel]=value
    return p


def load(p,expected=None):
    return json.loads(checked(p,expected).read_text(encoding='utf-8'))


def write(name,value):
    with (OUT/('feature_'+name+'.json')).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def xfeatures(b):
    pg,pl=expit(b['zg']),expit(b['zl'])
    def ent(p):
        p=np.clip(p,1e-12,1-1e-12)
        return -p*np.log(p)-(1-p)*np.log1p(-p)
    return np.column_stack((pg,pl,np.abs(pg-pl),(ent(pg)+ent(pl))/2,np.log1p(b['variance']),b['q']))


def alpha_and_gradient(art,x):
    v=((x-np.asarray(art['mean']))/np.asarray(art['scale'])).astype(np.float32)
    s={k:np.asarray(z,dtype=np.float32) for k,z in art['state'].items()}
    hidden=v@s['hidden.weight'].T+s['hidden.bias']
    a=expit(np.maximum(hidden,0)@s['output.weight'].T+s['output.bias']).ravel().astype(float)
    # Derivative with respect to each standardized quality coordinate, holding
    # all seven other coordinates fixed; no causal interpretation is implied.
    grad=(hidden>0)*s['output.weight'].reshape(1,-1)
    grad=grad@s['hidden.weight'][:,5:]
    return a,(a*(1-a))[:,None]*grad


def describe(values):
    v=np.asarray(values,dtype=float)
    return dict(n=len(v),mean=float(v.mean()),sd=float(v.std(ddof=1)),min=float(v.min()),
                p01=float(np.quantile(v,.01)),p05=float(np.quantile(v,.05)),median=float(np.quantile(v,.5)),
                p95=float(np.quantile(v,.95)),p99=float(np.quantile(v,.99)),max=float(v.max()))


def stats(y,z,cut):
    pred=expit(z)>=cut
    return dict(auroc=float(roc_auc_score(y,z)),balanced_accuracy=float((pred[y==1].mean()+(~pred[y==0]).mean())/2),
                bce=float(np.mean(np.logaddexp(0,z)-y*z)))


def aggregate(rows,keys,measures):
    groups=defaultdict(lambda:defaultdict(list))
    for r in rows:groups[tuple(r[k] for k in keys)][r['run_seed']].append(r)
    out=[]
    for key,perseed in sorted(groups.items()):
        r=dict(zip(keys,key))
        for m in measures:
            values=[float(np.mean([v[m] for v in perseed[s]])) for s in SEEDS]
            r[m]=dict(mean=float(np.mean(values)),seed_sd=float(np.std(values,ddof=1)),each_seed=values)
        out.append(r)
    return out


def q_source(number,part,bundle,freeze):
    raw_name='gate_fit' if part=='gate_fit' else 'tuning'
    template=bundle['raw_batches'][raw_name]['rows']
    ids=sorted({r['source_id'] for r in template})
    rows,q=[],[]
    for cond in TUNE:
        actual,blocks=[],[]
        folder=ROOT/f'artifacts/round8_a/seed{number}/features/{part}_{cond}_0'
        for p in sorted(folder.glob('*.npz')):
            checked(p,freeze['input_sha256'][p.relative_to(ROOT).as_posix()])
            with np.load(p,allow_pickle=False) as c:
                assert json.loads(str(c['identity']))==[20261100+number,part,cond,False,c['ids'].tolist()]
                assert hashlib.sha256(c['q'].tobytes()).hexdigest()==str(c['q_sha256'])
                actual+=c['ids'].tolist();blocks.append(c['q'])
        assert actual==ids
        rr=[r for r in template if r['condition']==cond]
        assert [r['source_id'] for r in rr]==ids
        rows+=rr;q.extend(np.concatenate(blocks))
    return dict(rows=rows,q=np.asarray(q))


def sub(qb,ids,conditions=None):
    ix=[i for i,r in enumerate(qb['rows']) if r['source_id'] in set(ids) and (conditions is None or r['condition'] in conditions)]
    return dict(rows=[qb['rows'][i] for i in ix],q=qb['q'][ix])


def joined(a,b):
    return dict(rows=a['rows']+b['rows'],q=np.concatenate([a['q'],b['q']]))


def main():
    assert (STUDY/'complete.json').exists() and (STUDY/'models_frozen_before_evaluation.json').exists()
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'feature_protocol_codefix1.json').exists(),'No overwrite/retry of completed feature probes'
    complete=load(STUDY/'complete.json')
    freeze=load(STUDY/'freeze.json',complete['outputs_sha256']['freeze.json'])
    settings=load(STUDY/'model_settings.json',complete['outputs_sha256']['model_settings.json'])
    splits=load(STUDY/'splits.json',freeze['split_sha256'])
    checked(Path(__file__))
    checked(ROOT/'scripts/round7_baseline_data.py',freeze['input_sha256']['scripts/round7_baseline_data.py'])
    checked(ROOT/'scripts/gate_models.py',freeze['input_sha256']['scripts/gate_models.py'])
    write('protocol_codefix1',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        implementation_correction='Initial feature_protocol.json preserved. A pre-output count assertion caught donor JPEG50 leaking into source q-only bookkeeping; filter corrected to the declared 3 fit/tuning conditions. No model, protocol choices, or intervention outcomes changed.',
        scope='Read-only mechanisms of frozen source-only and mixed AUROC-selected quality gates on permitted development originals.',
        variants=list(VARIANTS),conditions=list(CONDS),run_seeds=list(SEEDS),gate_seeds=list(INITS),
        semantics=dict(q0='log1p(1e4 * variance(scipy.ndimage.laplace(gray/255)))',
                       q1='Squared FFT energy outside centered radius 56 divided by total non-DC energy, 224x224 processed gray image',
                       q2='Mean absolute adjacent-pixel differences crossing 8-pixel grid boundaries, divided by mean absolute differences over all positions'),
        held_fixed='Experts, original 5 non-quality gate inputs, gate weights/scaler, AUROC checkpoint and tuning threshold. No new model or threshold fitting.',
        clean_replacement='Replace all 3 q by the same original image clean q for every condition; clean is an exact identity check.',
        labeling='Per-condition univariate raw-q AUROC is a descriptive source/label association, not a fitted detector or independent performance claim.',
        aggregation='Average gate-initialization metrics within seed, then mean/sample SD across 3 shared-pool seeds.',
        limits=['Mean-coordinate and clean-q probes can be off-manifold and inconsistent with degraded expert scores.',
                'Nonlinear interventions are not additive feature attribution or evidence of a deployable improvement.',
                'Source/target correlations alone do not establish a learned label shortcut or causal physical mechanism.',
                'Previously inspected development domains; no final/reserved-test files or images used.'],initial_input_sha256=INPUTS.copy()))
    predfile=checked(STUDY/'predictions.csv.gz',complete['outputs_sha256']['predictions.csv.gz'])
    saved=defaultdict(dict)
    with gzip.open(predfile,'rt',encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            if r['model'] in ('quality_gate','score_gate') and r['checkpoint']=='auroc':
                key=(int(r['run_seed']),r['scenario'],r['heldout_generator'],r['model'],int(r['gate_seed']))
                saved[key][r['condition'],r['source_id']]=r
    distributions,label_auc,probes,gradients=[],[],[],[]
    max_alpha_error,max_logit_error,max_auc_error=0.,0.,0.
    baseline_rows_checked=0
    for number,run in enumerate(SEEDS,1):
        rel=f'artifacts/round8_a/seed{number}/matched/bundle.json'
        bundle=load(ROOT/rel,freeze['input_sha256'][rel])
        dev={k:v if k=='rows' else np.asarray(v) for k,v in bundle['raw_batches']['development'].items()}
        dev['x']=xfeatures(dev)
        source_fit=q_source(number,'gate_fit',bundle,freeze)
        source_tune=q_source(number,'selection',bundle,freeze)
        devq=dict(rows=dev['rows'],q=dev['q'])
        for scenario in ('source_only','mixed_ADM','mixed_BigGAN'):
            split=splits['sources'][f'{run}/{scenario}']
            fit_ids,tune_ids=map(set,(split['fit_ids'],split['tuning_ids']))
            src_f={r['source_id'] for r in source_fit['rows']};src_t={r['source_id'] for r in source_tune['rows']}
            train=sub(source_fit,fit_ids&src_f);tune=sub(source_tune,tune_ids&src_t)
            if fit_ids-src_f:train=joined(train,sub(devq,fit_ids-src_f,TUNE))
            if tune_ids-src_t:tune=joined(tune,sub(devq,tune_ids-src_t,TUNE))
            assert len(train['rows'])==1500 and len(tune['rows'])==744
            reference=settings[f'{run}/{scenario}']['gates'][f'quality_gate/{INITS[0]}']['auroc']
            qmean,qscale=np.array(reference['mean'][5:]),np.array(reference['scale'][5:])
            np.testing.assert_allclose(train['q'].mean(axis=0),qmean,atol=1e-12,rtol=0)
            np.testing.assert_allclose(train['q'].std(axis=0),qscale,atol=1e-12,rtol=0)
            for target in split['heldout_targets']:
                wanted=set(splits['evaluation_ids']['Real']+splits['evaluation_ids'][target])
                evaluation=sub(devq,wanted)
                assert not wanted&(fit_ids|tune_ids)
                for partition,batch in (('gate_fit',train),('tuning',tune),('heldout_development',evaluation)):
                    conds=TUNE if partition!='heldout_development' else CONDS
                    gens=sorted({r['generator'] for r in batch['rows']})
                    for cond in conds:
                        for gen in gens:
                            ix=np.array([r['condition']==cond and r['generator']==gen for r in batch['rows']])
                            for q in range(3):
                                values=batch['q'][ix,q]; standardized=(values-qmean[q])/qscale[q]
                                tr=train['q'][:,q];lo,hi=np.quantile(tr,[.01,.99])
                                distributions.append(dict(run_seed=run,scenario=scenario,heldout_generator=target,
                                    partition=partition,condition=cond,generator=gen,label=0 if gen=='Real' else 1,
                                    feature=QNAMES[q],raw=describe(values),standardized=describe(standardized),
                                    fraction_abs_z_gt3=float(np.mean(np.abs(standardized)>3)),
                                    fraction_outside_training_minmax=float(np.mean((values<tr.min())|(values>tr.max()))),
                                    fraction_outside_training_p01_p99=float(np.mean((values<lo)|(values>hi)))))
                        for gen in [g for g in gens if g!='Real']:
                            ix=np.array([r['condition']==cond and r['generator'] in ('Real',gen) for r in batch['rows']])
                            y=np.array([r['label'] for r,k in zip(batch['rows'],ix) if k])
                            for q in range(3):
                                label_auc.append(dict(run_seed=run,scenario=scenario,heldout_generator=target,
                                    partition=partition,condition=cond,generator=gen,feature=QNAMES[q],
                                    raw_q_label_auroc=float(roc_auc_score(y,batch['q'][ix,q]))))
                ix=[i for i,r in enumerate(dev['rows']) if r['source_id'] in wanted]
                rows=[dev['rows'][i] for i in ix];x=dev['x'][ix].copy();zg,zl=dev['zg'][ix],dev['zl'][ix]
                y=np.array([r['label'] for r in rows]);gen=np.array([r['generator'] for r in rows]);condition=np.array([r['condition'] for r in rows])
                clean_q={r['source_id']:dev['q'][i] for i,r in enumerate(dev['rows']) if r['condition']=='clean'}
                for gs in INITS:
                    art=settings[f'{run}/{scenario}']['gates'][f'quality_gate/{gs}']['auroc']
                    base_a,gradient=alpha_and_gradient(art,x);base_z=base_a*zg+(1-base_a)*zl
                    record=saved[run,scenario,target,'quality_gate',gs]
                    stored_a=np.array([float(record[r['condition'],r['source_id']]['alpha_global']) for r in rows])
                    stored_z=np.array([float(record[r['condition'],r['source_id']]['logit']) for r in rows])
                    alpha_gap=float(np.max(np.abs(base_a-stored_a)));logit_gap=float(np.max(np.abs(base_z-stored_z)))
                    max_alpha_error=max(max_alpha_error,alpha_gap);max_logit_error=max(max_logit_error,logit_gap)
                    assert alpha_gap<5e-6 and logit_gap<1e-4
                    baseline_rows_checked+=len(rows)
                    for cond in CONDS:
                        cm=condition==cond
                        gap=abs(roc_auc_score(y[cm],base_z[cm])-roc_auc_score(y[cm],stored_z[cm]))
                        max_auc_error=max(max_auc_error,gap);assert gap<1e-4
                        for source in ('Real',target):
                            sm=cm&(gen==source)
                            for q in range(3):
                                gradients.append(dict(run_seed=run,scenario=scenario,heldout_generator=target,gate_seed=gs,
                                    condition=cond,generator=source,feature=QNAMES[q],
                                    d_alpha_d_standardized_q_mean=float(gradient[sm,q].mean()),
                                    d_alpha_d_standardized_q_abs_mean=float(np.abs(gradient[sm,q]).mean())))
                    for variant in VARIANTS:
                        changed=x.copy()
                        if variant.endswith('_training_mean'):
                            columns=[5,6,7] if variant.startswith('all_') else [5+int(variant[1])]
                            changed[:,columns]=np.array(art['mean'])[columns]
                        elif variant=='all_q_from_same_image_clean':
                            changed[:,5:]=np.array([clean_q[r['source_id']] for r in rows])
                        a,_=alpha_and_gradient(art,changed);z=a*zg+(1-a)*zl
                        assert np.array_equal(changed[:,:5],x[:,:5])
                        if variant=='all_q_from_same_image_clean':assert np.array_equal(a[condition=='clean'],base_a[condition=='clean'])
                        for cond in CONDS:
                            cm=condition==cond;real=cm&(y==0);fake=cm&(y==1)
                            result=stats(y[cm],z[cm],art['threshold']);original=stats(y[cm],base_z[cm],art['threshold'])
                            sc=saved[run,scenario,target,'score_gate',gs]
                            score_z=np.array([float(sc[r['condition'],r['source_id']]['logit']) for r,k in zip(rows,cm) if k])
                            score_auc=float(roc_auc_score(y[cm],score_z))
                            probes.append(dict(run_seed=run,scenario=scenario,heldout_generator=target,gate_seed=gs,condition=cond,
                                variant=variant,selected_epoch=art['selected_epoch'],**result,
                                baseline_quality_auc=original['auroc'],original_score_gate_auc=score_auc,
                                auc_delta=result['auroc']-original['auroc'],gap_to_score=result['auroc']-score_auc,
                                ba_delta=result['balanced_accuracy']-original['balanced_accuracy'],
                                alpha_mean=float(a[cm].mean()),alpha_delta_mean=float((a-base_a)[cm].mean()),
                                alpha_delta_abs_mean=float(np.abs(a-base_a)[cm].mean()),
                                alpha_real_delta=float((a-base_a)[real].mean()),alpha_fake_delta=float((a-base_a)[fake].mean())))
    write('distributions',distributions)
    write('univariate_label_auroc',label_auc)
    write('gradients_each',gradients)
    write('interventions_each',probes)
    measures=('auroc','auc_delta','gap_to_score','balanced_accuracy','ba_delta','alpha_mean','alpha_delta_mean','alpha_delta_abs_mean','alpha_real_delta','alpha_fake_delta')
    summary=aggregate(probes,['scenario','heldout_generator','condition','variant'],measures)
    write('interventions_summary',summary)
    gradient_summary=aggregate(gradients,['scenario','heldout_generator','condition','generator','feature'],
                              ['d_alpha_d_standardized_q_mean','d_alpha_d_standardized_q_abs_mean'])
    write('gradients_summary',gradient_summary)
    write('label_auc_summary',aggregate(label_auc,['scenario','heldout_generator','partition','condition','generator','feature'],['raw_q_label_auroc']))
    assert baseline_rows_checked==86400 and len(probes)==864
    assert all(sha(ROOT/p)==h for p,h in INPUTS.items())
    write('verification',dict(status='passed',baseline_quality_prediction_rows_checked=baseline_rows_checked,
        intervention_metric_cells=len(probes),distribution_rows=len(distributions),
        maximum_alpha_reconstruction_error=max_alpha_error,maximum_logit_reconstruction_error=max_logit_error,
        maximum_baseline_auc_reconstruction_error=max_auc_error,input_sha256=INPUTS,
        expert_logits_and_first_five_gate_inputs_unchanged=True,clean_replacement_identity_check_passed=True,
        no_training_no_selection_no_threshold_refit=True,no_final_test_access=True,
        limitations=['All analyses are post-hoc on previously inspected development images.',
                     'Interventions can be off-manifold; they diagnose frozen responses, not deployable model gains or unique causal feature effects.']))
    print(json.dumps(dict(status='passed',baseline_prediction_rows=baseline_rows_checked,intervention_cells=len(probes),
                         max_alpha_error=max_alpha_error,max_logit_error=max_logit_error,max_auc_error=max_auc_error),indent=2))


if __name__=='__main__':main()
