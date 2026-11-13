"""Read-only frozen-model objective/checkpoint/source-domain diagnostics."""
import sys
sys.dont_write_bytecode = True
import csv, hashlib, importlib.util, json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
import joblib

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'output/analysis/multigen_gate_20261108/run_v1_codefix1'
OUT = ROOT/'output/analysis/gate_failure_mechanism_20261108'
spec = importlib.util.spec_from_file_location('frozen_study', ROOT/'tmp/multigen_gate_20261108/run_study.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)

def read(name):
    return json.loads((RUN/name).read_text(encoding='utf-8'))

def writecsv(name, rows):
    with (OUT/name).open('x', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)

def alpha(art,x):
    xx=((x-np.asarray(art['mean']))/np.asarray(art['scale'])).astype(np.float32)
    if art['kind']=='score_gate':xx[:,5:]=0
    s={k:np.asarray(v,np.float32) for k,v in art['state'].items()}
    return expit(np.maximum(xx@s['hidden.weight'].T+s['hidden.bias'],0)@s['output.weight'].T+s['output.bias']).ravel().astype(float)

def pool(items,keys,measures):
    groups=defaultdict(list)
    for r in items:groups[tuple(r[k] for k in keys)].append(r)
    return [dict(zip(keys,k),**{m:float(np.mean([r[m] for r in v])) for m in measures},n_cells=len(v)) for k,v in groups.items()]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frozen=read('freeze.json');complete=read('complete.json')
    assert complete['status']=='completed'
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in frozen['input_sha256'].items())
    settings=read('model_settings.json');splits=read('splits.json')
    sources=study.cache.verify_and_collect()
    metrics=[]; histories=[]; margins=[]; gradient=[];max_auc_error=0.
    for number,source in sources.items():
        seed=source['run_seed']
        experts=joblib.load(ROOT/f'artifacts/round8_a/seed{number}/matched/experts.joblib')
        rawfit=study.cache.reconstruct_original(source,number,'gate_fit',experts)
        rawtune=study.cache.reconstruct_original(source,number,'selection',experts)
        dev=study.cache.array_batch(source['matched']['raw_batches']['development'])
        for scenario in ['source_only','mixed_ADM','mixed_BigGAN']:
            key=f'{seed}/{scenario}';sp=splits['sources'][key];sett=settings[key]
            batches={}
            for part,raw,idkey in [('fit',rawfit,'fit_ids'),('tune',rawtune,'tuning_ids')]:
                want=set(sp[idkey]);source_ids=study.ids(raw)&want
                batch=study.take(raw,source_ids)
                donor=want-source_ids
                if donor:batch=study.merge(batch,study.take(dev,donor))
                batches[part]=batch
            for kind in ['score_gate','quality_gate']:
                for gs in study.gates.SEEDS:
                    mod=read(f'models/{seed}_{scenario}_{kind}_{gs}.json');history=mod['history'];h0=history[0]
                    for cp in ['auroc','bce']:
                        art=mod['checkpoints'][cp];h=history[art['selected_epoch']]
                        histories.append(dict(run_seed=seed,scenario=scenario,model=kind,gate_seed=gs,checkpoint=cp,
                            selected_epoch=art['selected_epoch'],stop_epoch=art['stopped_epoch'],
                            train_bce=h['train_bce'],tuning_bce=h['tuning_bce'],tuning_auc=h['tuning_macro_auroc'],
                            delta_train_bce=h['train_bce']-h0['train_bce'],delta_tuning_bce=h['tuning_bce']-h0['tuning_bce'],
                            delta_tuning_auc=h['tuning_macro_auroc']-h0['tuning_macro_auroc']))
                        a=alpha(art,batches['tune']['x']);z=a*batches['tune']['zg']+(1-a)*batches['tune']['zl']
                        max_auc_error=max(max_auc_error,abs(study.macro_auc(batches['tune'],z)-art['tuning_macro_auroc']))
            for part,b in batches.items():
                gen=np.array([r['generator'] for r in b['rows']]);cond=np.array([r['condition'] for r in b['rows']]);y=b['y']
                models=[(name,'fixed',-1,np.full(len(y),st['alpha'])) for name,st in sett['baselines'].items()]
                for kind in ['score_gate','quality_gate']:
                    for gs in study.gates.SEEDS:
                        for cp in ['auroc','bce']:
                            art=sett['gates'][f'{kind}/{gs}'][cp]
                            models.append((kind,cp,gs,alpha(art,b['x'])))
                for c in study.TUNE:
                    for fake in sorted(set(gen)-{'Real'}):
                        m=(cond==c)&np.isin(gen,['Real',fake]);yy=y[m];g=b['zg'][m];l=b['zl'][m]
                        for model,cp,gs,aa in models:
                            a=aa[m];z=a*g+(1-a)*l;loss=np.logaddexp(0,z)-yy*z
                            metrics.append(dict(run_seed=seed,scenario=scenario,partition=part,condition=c,generator=fake,model=model,checkpoint=cp,gate_seed=gs,
                                auroc=float(roc_auc_score(yy,z)),bce=float(loss.mean()),alpha_mean=float(a.mean()),
                                alpha_real=float(a[yy==0].mean()),alpha_fake=float(a[yy==1].mean()),
                                mean_logit_real=float(z[yy==0].mean()),mean_logit_fake=float(z[yy==1].mean()),
                                fake_below_zero=float((z[yy==1]<0).mean()),real_above_zero=float((z[yy==0]>=0).mean()),
                                loss_from_wrong_sign=float(loss[(z>=0)!=yy].sum()/loss.sum())))
                            if (cp=='auroc' and model=='quality_gate') or model=='average':
                                grad=(expit(z)-yy)*(g-l)
                                for label in [0,1]:
                                    mm=yy==label;gg=grad[mm];wrong=((z>=0)!=yy)&mm
                                    gradient.append(dict(run_seed=seed,scenario=scenario,partition=part,condition=c,generator=fake,model=model,checkpoint=cp,gate_seed=gs,label=label,
                                        mean_gradient=float(gg.mean()),mean_abs_gradient=float(abs(gg).mean()),
                                        gradient_abs_share_wrong=float(abs(grad[wrong]).sum()/max(1e-20,abs(gg).sum())),
                                        gradient_descent_favors_global=float((gg<0).mean()),
                                        mean_signed_expert_advantage=float(((2*yy-1)*(g-l))[mm].mean())))
                        for label in [0,1]:
                            mm=yy==label;adv=(2*yy[mm]-1)*(g[mm]-l[mm])
                            margins.append(dict(run_seed=seed,scenario=scenario,partition=part,condition=c,generator=fake,label=label,
                                mean_global_logit=float(g[mm].mean()),mean_local_logit=float(l[mm].mean()),
                                mean_abs_global_logit=float(abs(g[mm]).mean()),mean_abs_local_logit=float(abs(l[mm]).mean()),
                                global_more_favorable_fraction=float((adv>0).mean()),
                                mean_global_signed_advantage=float(adv.mean())))
    assert max_auc_error<1e-6,max_auc_error
    writecsv('objective_checkpoints.csv',histories)
    writecsv('objective_source_metrics.csv',metrics)
    writecsv('objective_source_gradient.csv',gradient)
    writecsv('objective_source_margins.csv',margins)
    measures=['auroc','bce','alpha_mean','alpha_real','alpha_fake','mean_logit_real','mean_logit_fake','fake_below_zero','real_above_zero','loss_from_wrong_sign']
    aggregated=pool(metrics,['scenario','partition','condition','generator','model','checkpoint'],measures)
    writecsv('objective_source_summary.csv',aggregated)
    hist_summary=pool(histories,['scenario','model','checkpoint'],['selected_epoch','stop_epoch','train_bce','tuning_bce','tuning_auc','delta_train_bce','delta_tuning_bce','delta_tuning_auc'])
    grad_summary=pool(gradient,['scenario','partition','condition','generator','model','label'],['mean_gradient','mean_abs_gradient','gradient_abs_share_wrong','gradient_descent_favors_global','mean_signed_expert_advantage'])
    writecsv('objective_gradient_summary.csv',grad_summary)
    result=dict(status='analyzed',all_saved_tuning_auroc_reproduced=True,max_tuning_auroc_error=max_auc_error,
        no_training=True,no_final_test_access=True,scope='descriptive fixed-model/source-domain diagnostics; not new held-out performance evidence',
        checkpoint_summary=hist_summary,input_sha256=frozen['input_sha256'])
    with (OUT/'objective_summary.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},indent=2))

if __name__=='__main__':main()
