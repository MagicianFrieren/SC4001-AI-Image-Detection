"""One fixed half-weight paired-view global-head study; existing caches only."""
import sys
sys.dont_write_bytecode = True
import csv
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
import sklearn
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'output/analysis/paired_view_global_20261108'
HERE = Path(__file__).resolve().parent
SEEDS = [20261101,20261102,20261103]
CONDS = ['clean','jpeg70','jpeg50','resize05']
TUNE = ['clean','jpeg70','resize05']
ARMS = ['unmatched','matched','paired_half']
INPUTS = {}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(rel, expected=None):
    assert 'final' not in rel.lower() and 'reserved' not in rel.lower(), rel
    p=ROOT/rel; h=sha(p)
    assert expected is None or h==expected,rel
    INPUTS[rel]=h
    return json.loads(p.read_text(encoding='utf-8'))


def save(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)


def csvsave(name,rows):
    with (OUT/name).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def cutoff(y,z):
    fpr,tpr,values=roc_curve(y,expit(z));ok=np.isfinite(values)
    values,criterion=values[ok],(tpr-fpr)[ok]
    best=np.flatnonzero(np.isclose(criterion,criterion.max()))
    return float(values[best[np.argmin(abs(values[best]-.5))]])


def metrics(y,z,cut):
    p=expit(z)>=cut
    fpr=float(p[y==0].mean());tpr=float(p[y==1].mean())
    return dict(auroc=float(roc_auc_score(y,z)),average_precision=float(average_precision_score(y,z)),
                balanced_accuracy=(tpr+1-fpr)/2,real_false_positive_rate=fpr)


def load_features(number,seed,part,cond,matched,wanted,bundles):
    stem=f'artifacts/round8_a/seed{number}/features/{part}_{cond}_{int(matched)}'
    ids=[];blocks=[]
    for path in sorted((ROOT/stem).glob('*.npz')):
        rel=path.relative_to(ROOT).as_posix()
        hashes=[b['feature_sha256'][rel] for b in bundles.values() if rel in b['feature_sha256']]
        assert hashes and len(set(hashes))==1 and sha(path)==hashes[0],rel
        INPUTS[rel]=hashes[0]
        with np.load(path,allow_pickle=False) as f:
            ii=f['ids'].tolist();xx=f['g'].copy()
            assert json.loads(str(f['identity']))==[seed,part,cond,bool(matched),ii]
            assert str(f['signature'])==bundles['matched']['signature']
            assert hashlib.sha256(xx.tobytes()).hexdigest()==str(f['g_sha256'])
            assert xx.shape==(len(ii),512) and np.isfinite(xx).all()
            ids.extend(ii);blocks.append(xx)
    assert ids==sorted(wanted),(part,cond,matched,len(ids),len(wanted))
    return np.concatenate(blocks)


def main():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'freeze.json').exists(),'Frozen run already exists'
    plan=json.loads((HERE/'protocol_plan.json').read_text(encoding='utf-8'))
    preflight=read('output/analysis/paired_view_global_20261108/preflight.json')
    assert preflight['status']=='passed'
    for p in [Path(__file__),HERE/'protocol_plan.json']:
        INPUTS[p.relative_to(ROOT).as_posix()]=sha(p)
    oldprotocol=read('results/round8_a/protocol.json')
    records={}
    for name in ['fitting_pool_manifest.jsonl','development_manifest.jsonl']:
        rel='results/round7/'+name;p=ROOT/rel
        assert sha(p)==oldprotocol['input_sha256'][rel];INPUTS[rel]=sha(p)
        for line in p.read_text().splitlines():
            r=json.loads(line);records[r['source_id']]=r
    isolation=read('results/round7/step_b_isolation.json',oldprotocol['input_sha256']['results/round7/step_b_isolation.json'])
    deny=set(isolation['reserved_test_source_ids'])
    dev_ids=sorted(json.loads(line)['source_id'] for line in (ROOT/'results/round7/development_manifest.jsonl').read_text().splitlines())
    assert len(dev_ids)==900 and set(dev_ids).isdisjoint(deny)
    batches={};heads={};partition_records={};old_outputs={}
    for number,seed in enumerate(SEEDS,1):
        rel=f'results/round7/assignments_seed{number}.json'
        assignment=read(rel,oldprotocol['input_sha256'][rel])['partitions']
        fit_ids=sorted(assignment['expert_fit']);sel_ids=sorted(assignment['selection'])
        assert len(fit_ids)==1500 and len(sel_ids)==250
        assert set(fit_ids).isdisjoint(sel_ids) and set(fit_ids+sel_ids).isdisjoint(dev_ids)
        assert set(fit_ids+sel_ids+dev_ids).isdisjoint(deny)
        for field in ['pixel_sha256','source_bytes_sha256']:
            assert {records[i][field] for i in fit_ids+sel_ids}.isdisjoint({records[i][field] for i in dev_ids})
        partition_records[str(seed)]=dict(expert_fit=fit_ids,selection=sel_ids,development=dev_ids)
        bundles={};heads[seed]={}
        for arm in ['unmatched','matched']:
            marker=read(f'results/round8_a/seed{number}/{arm}/complete.json')
            assert marker['status']=='completed'
            stem=f'artifacts/round8_a/seed{number}/{arm}'
            b=read(stem+'/bundle.json',marker['files_sha256'][stem+'/bundle.json']);bundles[arm]=b
            path=ROOT/stem/'experts.joblib'
            assert sha(path)==b['experts_sha256']==marker['files_sha256'][stem+'/experts.joblib']
            INPUTS[(path.relative_to(ROOT)).as_posix()]=sha(path)
            heads[seed][arm]=joblib.load(path)['global_head']
            old_outputs[seed,arm]=b['raw_batches']
        x0=load_features(number,seed,'expert_fit','clean',False,fit_ids,bundles)
        x1=load_features(number,seed,'expert_fit','clean',True,fit_ids,bundles)
        y=np.array([records[i]['label'] for i in fit_ids]);assert y.sum()==750
        batches[seed,'train']=(x0,x1,y)
        for part,wanted,conditions in [('selection',sel_ids,CONDS),('development',dev_ids,CONDS)]:
            for cond in conditions:
                xx=load_features(number,seed,part,cond,False,wanted,bundles)
                yy=np.array([records[i]['label'] for i in wanted])
                batches[seed,part,cond]=(wanted,xx,yy)
    save('splits.json',partition_records)
    save('freeze.json',dict(protocol=plan,input_sha256=INPUTS,script_sha256=sha(Path(__file__)),
        runtime=dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,device='cpu'),
        split_sha256=sha(OUT/'splits.json')))
    settings={};training=[]
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        for seed in SEEDS:
            x0,x1,y=batches[seed,'train'];x=np.concatenate([x0,x1]);yy=np.tile(y,2);w=np.full(len(yy),.5)
            assert x.shape==(3000,512) and w.sum()==1500 and np.all(w[:1500]+w[1500:]==1)
            h=make_pipeline(StandardScaler(),LogisticRegression(C=1.,solver='liblinear',max_iter=2000,random_state=20261018))
            h.fit(x,yy,logisticregression__sample_weight=w);heads[seed]['paired_half']=h
            path=OUT/f'models/{seed}_paired_half.joblib';path.parent.mkdir(exist_ok=True)
            with path.open('xb') as f:joblib.dump(h,f)
            tr=dict(seed=seed,originals=1500,training_rows=3000,view_weight=.5,sample_weight_sum=float(w.sum()),
                    sample_weight_sha256=hashlib.sha256(w.tobytes()).hexdigest(),stacking_order='all original rows then same-ID matched rows',
                    original_ids=partition_records[str(seed)]['expert_fit'],classifier_parameters=h['logisticregression'].get_params(),
                    n_iter=h['logisticregression'].n_iter_.tolist(),model_sha256=sha(path),
                    coef=h['logisticregression'].coef_.tolist(),intercept=h['logisticregression'].intercept_.tolist(),
                    scaler_mean=h['standardscaler'].mean_.tolist(),scaler_scale=h['standardscaler'].scale_.tolist(),scaler_var=h['standardscaler'].var_.tolist())
            training.append(tr);settings[str(seed)]={}
            tx=np.concatenate([batches[seed,'selection',c][1] for c in TUNE]);ty=np.concatenate([batches[seed,'selection',c][2] for c in TUNE])
            for arm in ARMS:settings[str(seed)][arm]=dict(threshold=cutoff(ty,heads[seed][arm].decision_function(tx)))
            print(json.dumps(dict(fitted_seed=seed,rows=len(yy),weight_sum=float(w.sum()),iterations=tr['n_iter'])),flush=True)
    save('models_and_thresholds_frozen.json',dict(training=training,settings=settings,development_scored=False))
    metrics_rows=[];predictions=[];eval_scores={};max_control_error=0.
    for seed in SEEDS:
        for part in ['selection','development']:
            for cond in CONDS:
                ids,xx,y=batches[seed,part,cond];gen=np.array([records[i]['generator'] for i in ids])
                for arm in ARMS:
                    z=heads[seed][arm].decision_function(xx);cut=settings[str(seed)][arm]['threshold']
                    assert np.isfinite(z).all()
                    if arm!='paired_half':
                        ref=old_outputs[seed,arm][part]
                        old={(r['source_id'],r['condition']):ref['zg'][j] for j,r in enumerate(ref['rows'])}
                        err=max(abs(float(zz)-old[i,cond]) for i,zz in zip(ids,z));max_control_error=max(max_control_error,err)
                        assert err<1e-8,(seed,arm,part,cond,err)
                    for i,zz,yi,gi in zip(ids,z,y,gen):
                        predictions.append(dict(seed=seed,partition=part,condition=cond,arm=arm,source_id=i,generator=gi,label=int(yi),logit=float(zz),threshold=cut))
                    for target in (['SD15'] if part=='selection' else ['ADM','BigGAN']):
                        idx=sorted(np.flatnonzero(np.isin(gen,['Real',target])),key=lambda j:(int(y[j]),ids[j]))
                        yy,zz=y[idx],z[idx]
                        metrics_rows.append(dict(seed=seed,partition=part,condition=cond,arm=arm,generator=target,**metrics(yy,zz,cut)))
                        if part=='development':
                            assert np.array_equal(yy,np.repeat([0,1],300))
                            eval_scores[seed,arm,target,cond]=zz
    csvsave('predictions.csv',predictions);csvsave('metrics.csv',metrics_rows)
    measures=['auroc','average_precision','balanced_accuracy','real_false_positive_rate']
    cell_means=[];deltas=[]
    lookup={(r['seed'],r['arm'],r['generator'],r['condition']):r for r in metrics_rows if r['partition']=='development'}
    for target in ['ADM','BigGAN']:
        for cond in CONDS:
            for arm in ARMS:
                group=[lookup[s,arm,target,cond] for s in SEEDS]
                cell_means.append(dict(arm=arm,generator=target,condition=cond,**{m:float(np.mean([r[m] for r in group])) for m in measures}))
            for seed in SEEDS:
                for control in ['matched','unmatched']:
                    l=lookup[seed,'paired_half',target,cond];r=lookup[seed,control,target,cond]
                    deltas.append(dict(seed=seed,control=control,generator=target,condition=cond,**{m:l[m]-r[m] for m in measures}))
    csvsave('metrics_mean.csv',cell_means);csvsave('paired_differences.csv',deltas)
    def selected(target,cond,aggregate):
        return aggregate=='overall' or (aggregate=='jpeg' and cond.startswith('jpeg')) or (aggregate=='clean_BigGAN' and target=='BigGAN' and cond=='clean') or (aggregate=='clean_ADM' and target=='ADM' and cond=='clean')
    aggregates=['overall','jpeg','clean_BigGAN','clean_ADM']
    macro={arm:{agg:{m:float(np.mean([r[m] for r in cell_means if r['arm']==arm and selected(r['generator'],r['condition'],agg)])) for m in measures} for agg in aggregates} for arm in ARMS}
    each_seed={agg:[float(np.mean([r['auroc'] for r in deltas if r['control']=='matched' and r['seed']==s and selected(r['generator'],r['condition'],agg)])) for s in SEEDS] for agg in aggregates}
    jpeg_cell_delta={f'{g}/{c}':float(np.mean([r['auroc'] for r in deltas if r['control']=='matched' and r['generator']==g and r['condition']==c])) for g in ['ADM','BigGAN'] for c in ['jpeg70','jpeg50']}
    guards=dict(jpeg_each_seed=all(v>=-.01 for v in each_seed['jpeg']),jpeg_each_cell=all(v>=-.02 for v in jpeg_cell_delta.values()),
        clean_biggan_each_seed=all(v>0 for v in each_seed['clean_BigGAN']),clean_adm_mean=float(np.mean(each_seed['clean_ADM']))>=-.02,
        overall_each_seed=all(v>0 for v in each_seed['overall']))
    rng=np.random.default_rng(20261108);real_draw=rng.integers(0,300,(1000,300));fake_draw={g:rng.integers(0,300,(1000,300)) for g in ['ADM','BigGAN']}
    boot={}
    for key,z in eval_scores.items():
        ranks=rankdata(np.concatenate([z[:300][real_draw],z[300:][fake_draw[key[2]]]],axis=1),axis=1)
        boot[key]=(ranks[:,300:].sum(axis=1)-300*301/2)/(300*300)
    intervals={}
    for control in ['matched','unmatched']:
        intervals[control]={}
        for agg in aggregates:
            dd=np.mean([boot[s,'paired_half',g,c]-boot[s,control,g,c] for s in SEEDS for g in ['ADM','BigGAN'] for c in CONDS if selected(g,c,agg)],axis=0)
            intervals[control][agg]=dict(delta_point=macro['paired_half'][agg]['auroc']-macro[control][agg]['auroc'],conditional_percentile_95=np.quantile(dd,[.025,.975]).tolist())
    cis=intervals['matched']
    interval_guards=dict(overall=cis['overall']['conditional_percentile_95'][0]>0,clean_biggan=cis['clean_BigGAN']['conditional_percentile_95'][0]>0,jpeg=cis['jpeg']['conditional_percentile_95'][0]>=-.01)
    decision='not_supported_for_joint_goal' if not all(guards.values()) else 'supported_on_this_development_setup_only' if all(interval_guards.values()) else 'promising_but_uncertain'
    assert all(sha(ROOT/rel)==digest for rel,digest in INPUTS.items()),'Input changed'
    result=dict(status='completed',decision=decision,macro=macro,each_seed_delta_vs_matched=each_seed,jpeg_cell_delta_vs_matched=jpeg_cell_delta,
        point_guards=guards,interval_guards=interval_guards,bootstrap=dict(draws=1000,seed=20261108,paired_original_images=True,conditional_on_fitted_models=True,comparisons=intervals),
        max_control_logit_error=max_control_error,models_fitted=3,no_final_test_access=True,seconds=time.monotonic()-started)
    save('summary.json',result)
    save('complete.json',dict(status='completed',models_fitted=3,prediction_rows=len(predictions),metric_rows=len(metrics_rows),
        files_sha256={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file()}))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
