"""Independent no-refit verification of the frozen paired-view global pilot."""
import sys
sys.dont_write_bytecode = True
import csv
import hashlib
import json
import time
from pathlib import Path
import joblib
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/analysis/paired_view_global_20261108'
SEEDS=[20261101,20261102,20261103]
ARMS=['unmatched','matched','paired_half']
CONDS=['clean','jpeg70','jpeg50','resize05']
TUNE=['clean','jpeg70','resize05']
MEASURES=['auroc','average_precision','balanced_accuracy','real_false_positive_rate']
AGGS=['overall','jpeg','clean_BigGAN','clean_ADM']
INPUTS={};ERRORS={}


def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def checked(path,expected=None):
    path=Path(path).resolve();relative=path.relative_to(ROOT).as_posix()
    assert 'reserved' not in relative.lower() and 'final' not in relative.lower(),relative
    actual=sha(path);assert expected is None or actual==expected,relative
    INPUTS[relative]=actual
    return path


def read(path,expected=None):return json.loads(checked(path,expected).read_text(encoding='utf-8'))


def csvrows(path):
    with checked(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def close(name,actual,expected,tol=1e-10):
    a,b=np.asarray(actual,float),np.asarray(expected,float)
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),name
    error=float(abs(a-b).max()) if a.size else 0.
    ERRORS[name]=max(ERRORS.get(name,0),error)
    assert error<=tol,(name,error,tol)


def direct_metrics(y,z,threshold):
    positive=z[y==1];negative=z[y==0]
    auroc=float(((positive[:,None]>negative)+.5*(positive[:,None]==negative)).mean())
    order=np.argsort(-z,kind='stable');yy=y[order];zz=z[order]
    ends=np.r_[np.flatnonzero(np.diff(zz)!=0)+1,len(zz)]
    tp=np.cumsum(yy)[ends-1]
    ap=float(np.sum(np.diff(np.r_[0,tp])/sum(yy)*tp/ends))
    predictions=expit(z)>=threshold
    fpr=float(predictions[y==0].mean());tpr=float(predictions[y==1].mean())
    return dict(auroc=auroc,average_precision=ap,balanced_accuracy=(tpr+1-fpr)/2,real_false_positive_rate=fpr)


def source_threshold(y,z):
    """Independently construct ROC vertices and inherited drop-intermediate/tie rule."""
    p=expit(z);order=np.argsort(-p,kind='stable');p=p[order];y=y[order]
    end=np.r_[np.flatnonzero(np.diff(p)!=0),len(p)-1]
    tp=np.cumsum(y)[end];fp=1+end-tp;values=p[end]
    if len(fp)>2:
        keep=np.r_[True,(np.diff(fp,2)!=0)|(np.diff(tp,2)!=0),True]
        tp,fp,values=tp[keep],fp[keep],values[keep]
    objective=tp/sum(y)-fp/sum(1-y)
    best=np.flatnonzero(np.isclose(objective,objective.max()))
    return float(values[best[np.argmin(abs(values[best]-.5))]])


def features(number,seed,part,cond,matched,wanted,bundle):
    ids=[];blocks=[]
    paths=sorted((ROOT/f'artifacts/round8_a/seed{number}/features/{part}_{cond}_{int(matched)}').glob('*.npz'))
    for path in paths:
        checked(path,bundle['feature_sha256'][path.relative_to(ROOT).as_posix()])
        with np.load(path,allow_pickle=False) as f:
            ii=f['ids'].tolist();xx=f['g'].copy()
            assert json.loads(str(f['identity']))==[seed,part,cond,matched,ii]
            assert str(f['signature'])==bundle['signature']
            assert xx.shape==(len(ii),512) and xx.dtype==np.float32 and np.isfinite(xx).all()
            assert hashlib.sha256(xx.tobytes()).hexdigest()==str(f['g_sha256'])
            ids.extend(ii);blocks.append(xx)
    assert ids==sorted(wanted) and len(ids)==len(set(ids))
    return np.concatenate(blocks)


def includes(group,cond,agg):
    return agg=='overall' or (agg=='jpeg' and cond in ['jpeg70','jpeg50']) or (agg=='clean_BigGAN' and group=='BigGAN' and cond=='clean') or (agg=='clean_ADM' and group=='ADM' and cond=='clean')


def boot_pairs(real,fake,real_counts,fake_counts):
    wins=(fake[:,None]>real).astype(float)+.5*(fake[:,None]==real)
    with threadpool_limits(limits=2):
        return ((fake_counts@wins)*real_counts).sum(1)/90000


def main():
    started=time.monotonic()
    assert (OUT/'complete.json').exists(),'Wait until all models and original scoring complete.'
    assert not (OUT/'verification.json').exists()
    complete=read(OUT/'complete.json');assert complete['status']=='completed' and complete['models_fitted']==3
    for relative,digest in complete['files_sha256'].items():checked(OUT/relative,digest)
    freeze=read(OUT/'freeze.json')
    checked(ROOT/'tmp/paired_view_global_20261108/run_study.py',freeze['script_sha256'])
    for relative,digest in freeze['input_sha256'].items():checked(ROOT/relative,digest)
    preflight=read(OUT/'preflight.json')
    for relative,digest in preflight['input_sha256'].items():checked(ROOT/relative,digest)
    splits=read(OUT/'splits.json',freeze['split_sha256'])
    assert splits==preflight['source_ids']
    fitted=read(OUT/'models_and_thresholds_frozen.json');assert fitted['development_scored'] is False
    assert len(fitted['training'])==3
    summary=read(OUT/'summary.json')
    truth={}
    for name in ['fitting_pool_manifest.jsonl','development_manifest.jsonl']:
        for line in checked(ROOT/'results/round7'/name).read_text().splitlines():
            r=json.loads(line);truth[r['source_id']]=r
    isolation=read(ROOT/'results/round7/step_b_isolation.json')
    denied=set(isolation['reserved_test_source_ids']);allowed=set(isolation['allowed_source_ids'])
    old_protocol=read(ROOT/'results/round8_a/protocol.json')
    prediction_rows=csvrows(OUT/'predictions.csv')
    table={(int(r['seed']),r['partition'],r['condition'],r['arm'],r['source_id']):r for r in prediction_rows}
    assert len(table)==len(prediction_rows)==complete['prediction_rows']==41400
    saved_rows=csvrows(OUT/'metrics.csv')
    saved={(int(r['seed']),r['partition'],r['condition'],r['arm'],r['generator']):r for r in saved_rows}
    assert len(saved)==len(saved_rows)==complete['metric_rows']==108
    computed={};evals={};seen=set();gradients=[];thresholds={};max_old=0.
    for number,seed in enumerate(SEEDS,1):
        partition=splits[str(seed)];fit_ids=partition['expert_fit'];sel_ids=partition['selection'];dev_ids=partition['development']
        assert len(fit_ids)==1500 and len(sel_ids)==250 and len(dev_ids)==900
        for a,b in [('expert_fit','selection'),('expert_fit','development'),('selection','development')]:
            assert set(partition[a]).isdisjoint(partition[b])
            for field in ['source_bytes_sha256','pixel_sha256']:
                assert {truth[i][field] for i in partition[a]}.isdisjoint({truth[i][field] for i in partition[b]})
        assert set(fit_ids+sel_ids+dev_ids)<=allowed and set(fit_ids+sel_ids+dev_ids).isdisjoint(denied)
        bundles={};heads={};training=[]
        for matched,arm in [(False,'unmatched'),(True,'matched')]:
            marker=read(ROOT/f'results/round8_a/seed{number}/{arm}/complete.json')
            stem=f'artifacts/round8_a/seed{number}/{arm}'
            bundle=read(ROOT/stem/'bundle.json',marker['files_sha256'][stem+'/bundle.json'])
            assert bundle['signature']==sha(ROOT/'results/round8_a/protocol.json')
            bundles[arm]=bundle
            heads[arm]=joblib.load(checked(ROOT/stem/'experts.joblib',bundle['experts_sha256']))['global_head']
            training.append(features(number,seed,'expert_fit','clean',matched,fit_ids,bundle))
        audit=next(r for r in fitted['training'] if r['seed']==seed)
        heads['paired_half']=joblib.load(checked(OUT/f'models/{seed}_paired_half.joblib',audit['model_sha256']))
        model=heads['paired_half'];scaler=model['standardscaler'];lr=model['logisticregression']
        x=np.concatenate(training);y=np.tile([truth[i]['label'] for i in fit_ids],2);weight=np.full(3000,.5)
        assert x.shape==(3000,512) and weight.sum()==1500 and (weight[:1500]+weight[1500:]==1).all()
        assert audit['sample_weight_sha256']==hashlib.sha256(weight.tobytes()).hexdigest()
        assert audit['sample_weight_sum']==1500 and audit['originals']==1500 and audit['training_rows']==3000 and audit['view_weight']==.5
        assert audit['original_ids']==fit_ids and audit['stacking_order']=='all original rows then same-ID matched rows'
        assert lr.get_params()==audit['classifier_parameters']
        assert lr.C==1. and lr.solver=='liblinear' and lr.max_iter==2000 and lr.random_state==20261018
        assert lr.n_iter_.tolist()==audit['n_iter'] and int(lr.n_iter_[0])<2000
        xd=x.astype(np.float64)
        close('source_scaler_mean',scaler.mean_,xd.mean(0))
        close('source_scaler_var',scaler.var_,xd.var(0))
        close('source_scaler_scale',scaler.scale_,np.sqrt(xd.var(0)))
        for name,actual in [('scaler_mean',scaler.mean_),('scaler_var',scaler.var_),('scaler_scale',scaler.scale_),('coef',lr.coef_),('intercept',lr.intercept_)]:close('saved_'+name,actual,audit[name])
        tx=scaler.transform(x).astype(float);residual=expit(model.decision_function(x))-y
        params=np.r_[lr.coef_.ravel(),lr.intercept_.ravel()]
        design=np.column_stack([tx,np.ones(len(tx))])
        gradient_half=params+lr.C*design.T.dot(.5*residual)
        gradient_one=params+lr.C*design.T.dot(residual)
        gradients.append(dict(seed=seed,half_weight_gradient_l2=float(np.linalg.norm(gradient_half)),
                              incorrectly_unit_weight_gradient_l2=float(np.linalg.norm(gradient_one)),
                              parameter_l2=float(np.linalg.norm(params))))
        assert np.linalg.norm(gradient_half)<np.linalg.norm(gradient_one),'Weights inconsistent with fitted objective'
        cached={}
        for part,ids in [('selection',sel_ids),('development',dev_ids)]:
            for cond in CONDS:cached[part,cond]=features(number,seed,part,cond,False,ids,bundles['matched'])
        source_x=np.concatenate([cached['selection',c] for c in TUNE]);source_y=np.tile([truth[i]['label'] for i in sel_ids],3)
        for arm in ARMS:
            recomputed_threshold=source_threshold(source_y,heads[arm].decision_function(source_x))
            threshold=fitted['settings'][str(seed)][arm]['threshold']
            close('threshold',recomputed_threshold,threshold,1e-12);thresholds[f'{seed}/{arm}']=threshold
            for part,ids in [('selection',sel_ids),('development',dev_ids)]:
                yy=np.array([truth[i]['label'] for i in ids]);gen=np.array([truth[i]['generator'] for i in ids])
                for cond in CONDS:
                    z=heads[arm].decision_function(cached[part,cond])
                    if arm!='paired_half':
                        raw=bundles[arm]['raw_batches'][part]
                        old={(r['source_id'],r['condition']):raw['zg'][j] for j,r in enumerate(raw['rows'])}
                        maximum=max(abs(z[j]-old[i,cond]) for j,i in enumerate(ids));max_old=max(max_old,float(maximum))
                        close('control_logit_reproduction',z,[old[i,cond] for i in ids])
                    for j,identity in enumerate(ids):
                        key=seed,part,cond,arm,identity;row=table[key];seen.add(key)
                        assert int(row['label'])==yy[j] and row['generator']==gen[j]
                        close('prediction_logit',z[j],row['logit']);close('prediction_threshold',threshold,row['threshold'])
                    for group in (['SD15'] if part=='selection' else ['ADM','BigGAN']):
                        idx=sorted(np.flatnonzero(np.isin(gen,['Real',group])),key=lambda j:(yy[j],ids[j]))
                        key=seed,part,cond,arm,group
                        values=direct_metrics(yy[idx],z[idx],threshold);computed[key]=values
                        for metric in MEASURES:close('metric_'+metric,values[metric],saved[key][metric])
                        if part=='development':
                            assert np.array_equal(yy[idx],np.repeat([0,1],300))
                            ordered=[ids[j] for j in idx]
                            expected=sorted(i for i in dev_ids if truth[i]['generator']=='Real')+sorted(i for i in dev_ids if truth[i]['generator']==group)
                            assert ordered==expected
                            evals[seed,arm,group,cond]=z[idx]
    assert seen==set(table) and set(computed)==set(saved)
    close('max_control_logit_error',max_old,summary['max_control_logit_error'])
    means={}
    for row in csvrows(OUT/'metrics_mean.csv'):
        key=row['arm'],row['generator'],row['condition']
        values={m:float(np.mean([computed[s,'development',key[2],key[0],key[1]][m] for s in SEEDS])) for m in MEASURES}
        means[key]=values
        for m in MEASURES:close('cell_mean_'+m,values[m],row[m])
    assert len(means)==24
    differences={}
    for row in csvrows(OUT/'paired_differences.csv'):
        seed=int(row['seed']);control=row['control'];group=row['generator'];cond=row['condition']
        values={m:computed[seed,'development',cond,'paired_half',group][m]-computed[seed,'development',cond,control,group][m] for m in MEASURES}
        differences[seed,control,group,cond]=values
        for m in MEASURES:close('delta_'+m,values[m],row[m])
    assert len(differences)==48
    macro={}
    for arm in ARMS:
        macro[arm]={}
        for agg in AGGS:
            values={m:float(np.mean([v[m] for (aa,g,c),v in means.items() if aa==arm and includes(g,c,agg)])) for m in MEASURES}
            macro[arm][agg]=values
            for m in MEASURES:close('macro_'+m,values[m],summary['macro'][arm][agg][m])
    seed_deltas={agg:[float(np.mean([v['auroc'] for (s,control,g,c),v in differences.items() if s==seed and control=='matched' and includes(g,c,agg)])) for seed in SEEDS] for agg in AGGS}
    for agg in AGGS:close('seed_delta',seed_deltas[agg],summary['each_seed_delta_vs_matched'][agg])
    jpeg_cells={f'{g}/{c}':float(np.mean([differences[s,'matched',g,c]['auroc'] for s in SEEDS])) for g in ['ADM','BigGAN'] for c in ['jpeg70','jpeg50']}
    for cell,value in jpeg_cells.items():close('jpeg_cell_delta',value,summary['jpeg_cell_delta_vs_matched'][cell])
    guards=dict(jpeg_each_seed=all(v>=-.01 for v in seed_deltas['jpeg']),jpeg_each_cell=all(v>=-.02 for v in jpeg_cells.values()),
        clean_biggan_each_seed=all(v>0 for v in seed_deltas['clean_BigGAN']),clean_adm_mean=float(np.mean(seed_deltas['clean_ADM']))>=-.02,
        overall_each_seed=all(v>0 for v in seed_deltas['overall']))
    assert guards==summary['point_guards']
    # Draw exact same original IDs, but compute via multiplicity-weighted pair matrices, not sample ranks.
    rng=np.random.default_rng(20261108)
    real_draw=rng.integers(0,300,(1000,300))
    fake_draw={g:rng.integers(0,300,(1000,300)) for g in ['ADM','BigGAN']}
    real_counts=np.stack([np.bincount(row,minlength=300) for row in real_draw]).astype(float)
    fake_counts={g:np.stack([np.bincount(row,minlength=300) for row in draws]).astype(float) for g,draws in fake_draw.items()}
    boot={}
    for key,z in evals.items():
        values=boot_pairs(z[:300],z[300:],real_counts,fake_counts[key[2]])
        for draw in [0,499,999]:
            resampled=np.r_[z[:300][real_draw[draw]],z[300:][fake_draw[key[2]][draw]]]
            close('bootstrap_sklearn',values[draw],roc_auc_score(np.repeat([0,1],300),resampled),1e-12)
        boot[key]=values
    intervals={}
    for control in ['matched','unmatched']:
        intervals[control]={}
        for agg in AGGS:
            draws=np.mean([boot[s,'paired_half',g,c]-boot[s,control,g,c] for s in SEEDS for g in ['ADM','BigGAN'] for c in CONDS if includes(g,c,agg)],axis=0)
            interval=np.percentile(draws,[2.5,97.5]).tolist()
            point=macro['paired_half'][agg]['auroc']-macro[control][agg]['auroc']
            reported=summary['bootstrap']['comparisons'][control][agg]
            close('bootstrap_interval',interval,reported['conditional_percentile_95'],1e-12)
            close('bootstrap_point',point,reported['delta_point'],1e-12)
            intervals[control][agg]=dict(delta_point=point,conditional_percentile_95=interval)
    cis=intervals['matched']
    interval_guards=dict(overall=cis['overall']['conditional_percentile_95'][0]>0,
        clean_biggan=cis['clean_BigGAN']['conditional_percentile_95'][0]>0,jpeg=cis['jpeg']['conditional_percentile_95'][0]>=-.01)
    assert interval_guards==summary['interval_guards']
    decision='not_supported_for_joint_goal' if not all(guards.values()) else 'supported_on_this_development_setup_only' if all(interval_guards.values()) else 'promising_but_uncertain'
    assert decision==summary['decision']
    assert all(sha(ROOT/p)==digest for p,digest in INPUTS.items())
    result=dict(status='passed',verifier_sha256=sha(__file__),seconds=time.monotonic()-started,
        models_fitted_by_verifier=0,prediction_rows_verified=41400,metric_rows_verified=108,cell_means_verified=24,
        paired_differences_verified=48,bootstrap_draws_replayed=1000,bootstrap_implementation='Explicit pair dominance matrices, distinct from runner rank-sum resampling.',
        decision=decision,macro=macro,bootstrap=intervals,source_thresholds=thresholds,training_objective_gradients=gradients,
        max_absolute_errors=ERRORS,input_sha256=INPUTS,no_image_decode_or_extraction=True,no_final_test_access=True,
        limitations=['No head refit; training-objective consistency is checked through frozen code, sample-weight hash, independently reconstructed scaler and penalized gradients.',
                    'Bootstrap is conditional on these fitted heads and reuses the same development IDs; source seeds overlap.',
                    'Development has been repeatedly inspected; this is exploratory and is not new blind final confirmation.'])
    with (OUT/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ['status','seconds','prediction_rows_verified','metric_rows_verified','decision','training_objective_gradients','bootstrap']},indent=2))


if __name__=='__main__':main()
