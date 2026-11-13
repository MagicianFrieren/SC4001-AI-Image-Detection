"""Fixed-size multi-generator gate experiment; existing development data only."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
sys.dont_write_bytecode = True
import copy
import csv
import gzip
import hashlib
import json
import platform
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import scipy
from scipy.special import expit
from scipy.stats import rankdata
import sklearn
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.preprocessing import StandardScaler
import joblib
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'output/analysis/multigen_gate_20261108/run_v1_codefix1'
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'tmp/gate_diagnosis/original_gate_views'))
import gate_models as gates
import run_original_gate_views as cache

TUNE = ['clean','jpeg70','resize05']
CONDITIONS = ['clean','jpeg70','jpeg50','resize05']
FOLDS = [('ADM','BigGAN'),('BigGAN','ADM')]
METRICS = ['auroc','average_precision','balanced_accuracy','real_false_positive_rate','bce']
SALT = 'multigen_gate_20261108_v1/'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(name, data):
    p=OUT/name
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:
        json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False)


def ordered(ids, purpose):
    return sorted(ids,key=lambda s:(hashlib.sha256((SALT+purpose+'/'+s).encode()).hexdigest(),s))


def take(raw, ids, conditions=TUNE):
    ids=set(ids)
    rows=raw['rows']
    ix=[i for cond in conditions for i,r in enumerate(rows) if r['source_id'] in ids and r['condition']==cond]
    b={k:([rows[i] for i in ix] if k=='rows' else np.asarray(raw[k])[ix].copy()) for k in ['rows','zg','zl','variance','q','y']}
    b['x']=gates.inputs(b['zg'],b['zl'],b['variance'],b['q'])
    assert len(ix)==len(ids)*len(conditions)
    for cond in conditions:
        current=[r['source_id'] for r in b['rows'] if r['condition']==cond]
        assert len(current)==len(ids) and set(current)==ids
    return b


def merge(a,b):
    result={k:(a[k]+b[k] if k=='rows' else np.concatenate([a[k],b[k]])) for k in a}
    assert len({(r['source_id'],r['condition']) for r in result['rows']})==len(result['rows'])
    return result


def ids(raw,gen=None):
    return {r['source_id'] for r in raw['rows'] if gen is None or r['generator']==gen}


def auc_masks(b):
    g=np.array([r['generator'] for r in b['rows']]); c=np.array([r['condition'] for r in b['rows']])
    return [(c==cond)&np.isin(g,['Real',gen]) for gen in sorted(set(g)-{'Real'}) for cond in TUNE]


def macro_auc(b,z,masks=None):
    return float(np.mean([roc_auc_score(b['y'][m],z[m]) for m in (masks if masks is not None else auc_masks(b))]))


def cutoff(y,z):
    p=expit(z)
    fpr,tpr,values=roc_curve(y,p)
    good=np.isfinite(values); values=values[good]; objective=(tpr-fpr)[good]
    best=np.flatnonzero(np.isclose(objective,objective.max()))
    return float(values[best[np.argmin(np.abs(values[best]-.5))]])


def fixed_alpha(tune):
    masks=auc_masks(tune)
    grid=[(a/10,macro_auc(tune,(a/10)*tune['zg']+(1-a/10)*tune['zl'],masks)) for a in range(11)]
    top=max(v for a,v in grid)
    best=min([a for a,v in grid if abs(v-top)<=1e-6],key=lambda a:(abs(a-.5),a))
    return best,grid


def gate_alpha(model,scaler,b,kind):
    return gates.predict(model,scaler,b['x'],kind)


def fit(train,tune,kind,seed):
    torch.manual_seed(seed)
    model=gates.Gate()
    scaler=StandardScaler().fit(train['x'])
    def tensors(b):
        x=scaler.transform(b['x']).astype(np.float32)
        if kind=='score_gate':x[:,5:]=0
        return [torch.from_numpy(v.astype(np.float32)) for v in [x,b['zg'],b['zl'],b['y']]]
    tx,tg,tl,ty=tensors(train);vx,vg,vl,vy=tensors(tune)
    loss_fn=torch.nn.BCEWithLogitsLoss()
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1e-4)
    masks=auc_masks(tune)
    best_bce,best_auc=float('inf'),-float('inf')
    eb,ea,sb,sa=0,0,None,None
    history=[]
    for epoch in range(401):
        if epoch:
            model.train();opt.zero_grad();a=model(tx)
            loss_fn(a*tg+(1-a)*tl,ty).backward();opt.step()
        model.eval()
        with torch.inference_mode():
            at,av=model(tx),model(vx)
            train_bce=float(loss_fn(at*tg+(1-at)*tl,ty))
            val_bce=float(loss_fn(av*vg+(1-av)*vl,vy))
        aa=av.numpy().astype(float)
        val_auc=macro_auc(tune,aa*tune['zg']+(1-aa)*tune['zl'],masks)
        assert np.isfinite(train_bce+val_bce+val_auc)
        history.append(dict(epoch=epoch,train_bce=train_bce,tuning_bce=val_bce,tuning_macro_auroc=val_auc))
        if val_bce<best_bce-1e-6:best_bce,eb,sb=val_bce,epoch,copy.deepcopy(model.state_dict())
        if val_auc>best_auc+1e-6:best_auc,ea,sa=val_auc,epoch,copy.deepcopy(model.state_dict())
        if epoch-eb>=40:break
    fitted={}; artifacts={}
    for rule,state,selected in [('auroc',sa,ea),('bce',sb,eb)]:
        model.load_state_dict(state)
        a=gate_alpha(model,scaler,tune,kind);z=a*tune['zg']+(1-a)*tune['zl']
        cut=cutoff(tune['y'],z)
        fitted[rule]=(copy.deepcopy(model),scaler,cut)
        artifacts[rule]=dict(kind=kind,gate_seed=seed,checkpoint=rule,selected_epoch=selected,
           stopped_epoch=epoch,threshold=cut,tuning_macro_auroc=macro_auc(tune,z,masks),
           tuning_bce=float(np.mean(np.logaddexp(0,z)-tune['y']*z)),
           mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
           state={k:v.numpy().tolist() for k,v in state.items()})
    return fitted,artifacts,history


def metrics(y,z,threshold):
    pred=expit(z)>=threshold
    fpr=float(pred[y==0].mean())
    return dict(auroc=float(roc_auc_score(y,z)),average_precision=float(average_precision_score(y,z)),
       balanced_accuracy=float((pred[y==1].mean()+1-fpr)/2),real_false_positive_rate=fpr,
       bce=float(np.mean(np.logaddexp(0,z)-y*z)))


def write_csv(name,rows):
    with (OUT/name).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def summary(rows):
    byseed=defaultdict(list)
    for r in rows:
        key=tuple(r[k] for k in ['run_seed','scenario','heldout_generator','model','checkpoint','condition'])
        byseed[key].append(r)
    seedrows=[]
    for key,group in byseed.items():
        seedrows.append(dict(zip(['run_seed','scenario','heldout_generator','model','checkpoint','condition'],key),
            **{m:float(np.mean([r[m] for r in group])) for m in METRICS},
            alpha_mean=float(np.mean([r['alpha_mean'] for r in group]))))
    write_csv('metrics_seed_condition.csv',seedrows)
    lookup={(r['run_seed'],r['scenario'],r['heldout_generator'],r['model'],r['checkpoint'],r['condition']):r for r in seedrows}
    deltas=[]
    comparisons=[('quality-average','quality_gate','auroc','average','fixed','mixed'),
                 ('quality-fixed','quality_gate','auroc','fixed','fixed','mixed'),
                 ('quality-score','quality_gate','auroc','score_gate','auroc','mixed'),
                 ('mixed-quality-source-quality','quality_gate','auroc','quality_gate','auroc','source_only'),
                 ('mixed-score-source-score','score_gate','auroc','score_gate','auroc','source_only'),
                 ('mixed-fixed-source-fixed','fixed','fixed','fixed','fixed','source_only')]
    for title,left,lc,right,rc,source in comparisons:
        for run in [20261101,20261102,20261103]:
            for donor,target in FOLDS:
                scenario='mixed_'+donor
                for cond in CONDITIONS:
                    l=lookup[run,scenario,target,left,lc,cond]
                    r=lookup[run,scenario if source=='mixed' else source,target,right,rc,cond]
                    deltas.append(dict(comparison=title,run_seed=run,heldout_generator=target,condition=cond,
                                       **{m:l[m]-r[m] for m in METRICS}))
    write_csv('paired_differences.csv',deltas)
    grouped=defaultdict(list)
    for r in seedrows:
        arm='mixed' if r['scenario'].startswith('mixed_') else r['scenario']
        grouped[arm,r['model'],r['checkpoint']].append(r)
    macro=[]
    for (arm,model,checkpoint),group in grouped.items():
        macro.append(dict(arm=arm,model=model,checkpoint=checkpoint,
          **{m:float(np.mean([r[m] for r in group])) for m in METRICS},
          each_seed_auroc=[float(np.mean([r['auroc'] for r in group if r['run_seed']==s])) for s in [20261101,20261102,20261103]]))
    return seedrows,deltas,macro


def bootstrap(preds):
    rng=np.random.default_rng(20261108)
    real_draw=rng.integers(0,300,size=(1000,300))
    fake_draw={g:rng.integers(0,300,size=(1000,300)) for g in ['ADM','BigGAN']}
    dist=defaultdict(list)
    for key,z in preds.items():
        run,scenario,target,model,cp,gs,cond=key
        if not ((model in ['quality_gate','score_gate'] and cp=='auroc') or model in ['average','fixed']):continue
        ranks=rankdata(np.concatenate([z[:300][real_draw],z[300:][fake_draw[target]]],axis=1),axis=1)
        auc=(ranks[:,300:].sum(axis=1)-300*301/2)/(300*300)
        dist[scenario,model,cp,run,target,cond].append(auc)
    means={k:np.mean(v,axis=0) for k,v in dist.items()}
    specs=[('quality-average','quality_gate','auroc','average','fixed','mixed'),
           ('quality-fixed','quality_gate','auroc','fixed','fixed','mixed'),
           ('quality-score','quality_gate','auroc','score_gate','auroc','mixed'),
           ('mixed-quality-source-quality','quality_gate','auroc','quality_gate','auroc','source_only'),
           ('mixed-score-source-score','score_gate','auroc','score_gate','auroc','source_only'),
           ('mixed-fixed-source-fixed','fixed','fixed','fixed','fixed','source_only')]
    out=[]
    for name,l,lc,r,rc,source in specs:
        ds=[]
        for run in [20261101,20261102,20261103]:
            for donor,target in FOLDS:
                for cond in CONDITIONS:
                    scenario='mixed_'+donor
                    ds.append(means[scenario,l,lc,run,target,cond]-means[scenario if source=='mixed' else source,r,rc,run,target,cond])
        d=np.mean(ds,axis=0)
        out.append(dict(comparison=name,bootstrap_mean=float(d.mean()),conditional_percentile_95=np.quantile(d,[.025,.975]).tolist()))
    return out


def main():
    started=time.monotonic()
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    assert not (OUT/'freeze.json').exists(),'Do not overwrite a frozen experiment'
    OUT.mkdir(parents=True,exist_ok=True)
    sources=cache.verify_and_collect()
    plan=json.loads((HERE/'protocol_plan.json').read_text())
    preflight_path=HERE/'preflight/id_layout_preflight.json'
    assert preflight_path.exists(),'Independent identity/hash preflight required'
    preflight=json.loads(preflight_path.read_text())
    assert preflight['status']=='passed' and not preflight['dhash_distance_le3_pairs']
    assert all(sha(ROOT/p)==h for p,h in preflight['input_sha256'].items())
    manifests={}
    for path in ['results/round7/fitting_pool_manifest.jsonl','results/round7/development_manifest.jsonl']:
        for line in (ROOT/path).read_text().splitlines():
            r=json.loads(line);manifests[r['source_id']]=r
        cache.INPUT_HASHES[path]=sha(ROOT/path)
    reference_dev=sources[1]['matched']['raw_batches']['development']
    donors={}
    for gen in ['ADM','BigGAN']:
        all_ids=ordered(ids(reference_dev,gen),'donor/'+gen)
        assert len(all_ids)==300
        donors[gen]=dict(fit=all_ids[:125],tuning=all_ids[125:187],unused=all_ids[187:])
    batches={}; split_records={}
    for number,source in sources.items():
        run=source['run_seed']
        experts=joblib.load(ROOT/f'artifacts/round8_a/seed{number}/matched/experts.joblib')
        fitraw=cache.reconstruct_original(source,number,'gate_fit',experts)
        tuneraw=cache.reconstruct_original(source,number,'selection',experts)
        devraw=cache.array_batch(source['matched']['raw_batches']['development'])
        assert ids(devraw)==ids(reference_dev)
        rf=ids(fitraw,'Real'); sf=ids(fitraw,'SD15')
        rt=set(ordered(ids(tuneraw,'Real'),'source_tune_real')[:124])
        st=set(ordered(ids(tuneraw,'SD15'),'source_tune_fake')[:124])
        assert len(rf)==len(sf)==250 and len(rt)==len(st)==124
        scenarios={'source_only':(take(fitraw,rf|sf),take(tuneraw,rt|st))}
        for donor,target in FOLDS:
            fit_sd=set(ordered(sf,'source_fit_fake')[:125]);tune_sd=set(ordered(st,'source_tune_fake')[:62])
            train_batch=merge(take(fitraw,rf|fit_sd),take(devraw,donors[donor]['fit']))
            tune=merge(take(tuneraw,rt|tune_sd),take(devraw,donors[donor]['tuning']))
            scenarios['mixed_'+donor]=(train_batch,tune)
        for scenario,(train,tune) in scenarios.items():
            ti,vi=ids(train),ids(tune)
            assert len(ti)==500 and len(vi)==248 and ti.isdisjoint(vi)
            assert len(train['y'])==1500 and len(tune['y'])==744
            assert train['y'].sum()==750 and tune['y'].sum()==372
            assert set(r['condition'] for r in train['rows'])==set(TUNE)
            targets=[t for d,t in FOLDS] if scenario=='source_only' else [t for d,t in FOLDS if scenario=='mixed_'+d]
            for target in targets:
                eval_ids=ids(devraw,target)|ids(devraw,'Real')
                assert ti.isdisjoint(eval_ids) and vi.isdisjoint(eval_ids)
                assert not any(r['generator']==target for r in train['rows']+tune['rows'])
                for field in ['pixel_sha256','source_bytes_sha256']:
                    assert {manifests[i][field] for i in ti|vi}.isdisjoint({manifests[i][field] for i in eval_ids})
            split_records[f'{run}/{scenario}']=dict(fit_ids=sorted(ti),tuning_ids=sorted(vi),heldout_targets=targets)
            batches[run,scenario]=(train,tune)
        batches[run,'development']=devraw
    evaluation_ids={gen:sorted(ids(reference_dev,gen)) for gen in ['Real','ADM','BigGAN']}
    save('splits.json',dict(donors=donors,sources=split_records,evaluation_ids=evaluation_ids))
    input_hashes=dict(cache.INPUT_HASHES)
    for p in [Path(__file__),HERE/'protocol_plan.json',preflight_path,ROOT/'tmp/gate_diagnosis/original_gate_views/run_original_gate_views.py']:
        input_hashes[p.relative_to(ROOT).as_posix()]=sha(p)
    save('freeze.json',dict(protocol=plan,input_sha256=input_hashes,split_sha256=sha(OUT/'splits.json'),
       runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,torch=torch.__version__,device='cpu')))
    print('Frozen protocol, all source/target identity splits and inputs',flush=True)
    fitted={};settings={};fit_count=0
    for (run,scenario),pair in batches.items():
        if scenario=='development':continue
        train,tune=pair
        fixed,grid=fixed_alpha(tune)
        simple={}
        for name,a in [('global',1.),('local',0.),('average',.5),('fixed',fixed)]:
            simple[name]=dict(alpha=a,threshold=cutoff(tune['y'],a*tune['zg']+(1-a)*tune['zl']))
        settings[run,scenario]=dict(fixed_search=grid,baselines=simple,gates={})
        for kind in ['score_gate','quality_gate']:
            for gs in gates.SEEDS:
                models,arts,history=fit(train,tune,kind,gs)
                for cp,obj in models.items():fitted[run,scenario,kind,cp,gs]=obj
                settings[run,scenario]['gates'][f'{kind}/{gs}']=arts
                save(f'models/{run}_{scenario}_{kind}_{gs}.json',dict(checkpoints=arts,history=history))
                fit_count+=1
                print(json.dumps(dict(fit=fit_count,run_seed=run,scenario=scenario,kind=kind,gate_seed=gs,
                       auc_epoch=arts['auroc']['selected_epoch'],bce_epoch=arts['bce']['selected_epoch'],stop_epoch=arts['auroc']['stopped_epoch'])),flush=True)
                assert time.monotonic()-started<1800,'Study hard timeout'
    save('model_settings.json',{f'{r}/{s}':v for (r,s),v in settings.items()})
    save('models_frozen_before_evaluation.json',dict(fits=fit_count,settings_sha256=sha(OUT/'model_settings.json'),model_files_sha256={p.name:sha(p) for p in (OUT/'models').glob('*.json')}))
    assert fit_count==54
    metrics_rows=[];preds={};n_pred=0
    with gzip.open(OUT/'predictions.csv.gz','wt',encoding='utf-8',newline='') as f:
        fields=['run_seed','scenario','heldout_generator','model','checkpoint','gate_seed','condition','source_id','generator','label','logit','alpha_global','threshold']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for run in plan['data_seeds']:
            raw=batches[run,'development']
            for donor,target in FOLDS:
                for scenario in ['source_only','mixed_'+donor]:
                    b=take(raw,ids(raw,'Real')|ids(raw,target),CONDITIONS)
                    outputs=[]
                    for name,setting in settings[run,scenario]['baselines'].items():
                        outputs.append((name,'fixed',-1,np.full(len(b['y']),setting['alpha']),setting['threshold']))
                    for kind in ['score_gate','quality_gate']:
                        for cp in ['auroc','bce']:
                            for gs in gates.SEEDS:
                                model,scaler,cut=fitted[run,scenario,kind,cp,gs]
                                outputs.append((kind,cp,gs,gate_alpha(model,scaler,b,kind),cut))
                    for name,cp,gs,alpha,cut in outputs:
                        z=alpha*b['zg']+(1-alpha)*b['zl']
                        for cond in CONDITIONS:
                            ix=sorted([i for i,r in enumerate(b['rows']) if r['condition']==cond],key=lambda i:(b['rows'][i]['label'],b['rows'][i]['source_id']))
                            y=b['y'][ix];zz=z[ix];aa=alpha[ix]
                            assert len(ix)==600 and np.array_equal(y,np.repeat([0,1],300))
                            assert [b['rows'][i]['source_id'] for i in ix]==evaluation_ids['Real']+evaluation_ids[target]
                            key=(run,scenario,target,name,cp,gs,cond)
                            preds[key]=zz
                            metrics_rows.append(dict(run_seed=run,scenario=scenario,heldout_generator=target,model=name,
                              checkpoint=cp,gate_seed=gs,condition=cond,threshold=cut,alpha_mean=float(aa.mean()),**metrics(y,zz,cut)))
                            for j,score,weight in zip(ix,zz,aa):
                                r=b['rows'][j]
                                writer.writerow(dict(zip(fields[:7],[run,scenario,target,name,cp,gs,cond]),source_id=r['source_id'],generator=r['generator'],label=int(r['label']),logit=float(score),alpha_global=float(weight),threshold=cut))
                                n_pred+=1
    write_csv('metrics_each_initialization.csv',metrics_rows)
    seedrows,deltas,macro=summary(metrics_rows)
    for run in plan['data_seeds']:
        for donor,target in FOLDS:
            for cond in CONDITIONS:
                assert np.array_equal(preds[run,'source_only',target,'average','fixed',-1,cond],preds[run,'mixed_'+donor,target,'average','fixed',-1,cond])
    print('Target scoring complete; paired original-image bootstrap starting',flush=True)
    intervals=bootstrap(preds)
    save('bootstrap.json',dict(draws=1000,seed=20261108,conditional_on_fitted_models=True,
         shared_real_draws=True,shared_identity_draws_across_conditions=True,comparisons=intervals))
    assert all(sha(ROOT/p)==h for p,h in input_hashes.items())
    save('summary.json',dict(status='completed',macro=macro,primary_checkpoint='auroc',bootstrap=intervals,
         heldout_sources=['ADM','BigGAN'],scope='Previously inspected development, not independent final test'))
    save('complete.json',dict(status='completed',fits=fit_count,prediction_rows=n_pred,metric_rows=len(metrics_rows),
       input_hashes_unchanged=True,average_scores_identical_across_arms=True,no_final_test_access=True,
       seconds=time.monotonic()-started,outputs_sha256={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file()}))
    print(json.dumps(dict(status='completed',macro=macro,bootstrap=intervals),indent=2),flush=True)


if __name__=='__main__':
    main()
