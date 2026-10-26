"""Final per-generator metrics, three-seed summaries and paired comparisons."""
import sys
sys.dont_write_bytecode = True
from collections import defaultdict
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
import round7_final_common as f
from round7_final_common import c


def validate_rows(rows,models,condition,records):
    declarations={r['id']:r for r in models}; seen=set(); grouped=defaultdict(list)
    for r in rows:
        mid=r['model_id']; identity=r['source_id']
        c.require(mid in declarations and identity in records,'Unknown model or non-reserved source ID')
        c.require((mid,identity) not in seen,'Duplicate prediction'); seen.add((mid,identity))
        c.require(r['condition']==condition and r['generator']==records[identity]['generator'] and int(r['label'])==records[identity]['label'],'Prediction metadata mismatch')
        z=float(r['logit']); p=float(r['probability_fake']); cut=float(r['threshold'])
        c.require(np.isfinite([z,p,cut]).all() and np.isclose(p,expit(z),rtol=0,atol=1e-14),'Invalid prediction values')
        c.require(cut==declarations[mid]['threshold'],'Saved threshold changed/reselected')
        grouped[mid].append(r)
    c.require(seen=={(mid,i) for mid in declarations for i in records},'Incomplete source/model coverage')
    return grouped


def metric_rows(rows,models,condition,records):
    groups=validate_rows(rows,models,condition,records); result=[]
    for model in models:
        for gen,exposure in f.GROUPS.items():
            cell=[r for r in groups[model['id']] if r['generator'] in ('Real',gen)]
            y=np.array([int(r['label']) for r in cell]); z=np.array([float(r['logit']) for r in cell])
            point=c.points(y,z,model['threshold'])
            result.append({**{k:model[k] for k in ('stage','seed','profile','model','calibration','initialization')},'model_id':model['id'],
                           'generator':gen,'exposure':exposure,'condition':condition,'n_real':int(sum(y==0)),'n_fake':int(sum(y==1)),**point})
    return result


def stats(values):
    c.require(len(values)==3 and np.isfinite(values).all(),'Need all three run seeds')
    avg=float(np.mean(values)); sd=float(np.std(values,ddof=1))
    return {**{f'seed_{s}':float(v) for s,v in zip(f.seeds.SEEDS,values)},'mean':avg,'sd':sd,'mean_plus_minus_sd':f'{avg:.6f} ± {sd:.6f}'}


def summarize(raw,models):
    index={r['id']:r for r in models}; seen=set(); groups=defaultdict(list)
    for r in raw:
        key=(r['model_id'],r['generator'],r['condition'])
        c.require(key not in seen and r['model_id'] in index,'Duplicate/unknown raw metric'); seen.add(key)
        groups[tuple(r[k] for k in ('stage','seed','profile','model','calibration','generator','exposure','condition'))].append(r)
    c.require(seen=={(mid,g,k) for mid in index for g in f.GROUPS for k in f.CONDITIONS},'Missing final metric coverage')
    per_seed=[]
    keys=('stage','seed','profile','model','calibration','generator','exposure','condition')
    for key,rows in sorted(groups.items()):
        # Fixed initialization sets are declared before test release.
        want={m['initialization'] for m in models if (m['stage'],m['seed'],m['profile'],m['model'],m['calibration'])==key[:5]}
        c.require({r['initialization'] for r in rows}==want and len(rows)==len(want),'Gate initialization coverage mismatch')
        per_seed.append(dict(zip(keys,key),initializations=len(rows),**{metric:float(np.mean([r[metric] for r in rows])) for metric in f.METRICS}))
    lookup={tuple(r[k] for k in keys):r for r in per_seed}
    macros=[]
    bases={tuple(r[k] for k in keys[:5])+ (r['condition'],) for r in per_seed}
    for stage,seed,profile,model,cal,cond in sorted(bases):
        cells=[lookup[stage,seed,profile,model,cal,g,'untouched',cond] for g in f.UNTOUCHED]
        macros.append(dict(stage=stage,seed=seed,profile=profile,model=model,calibration=cal,generator='untouched_macro',exposure='untouched',condition=cond,
                           initializations=cells[0]['initializations'],**{m:float(np.mean([r[m] for r in cells])) for m in f.METRICS}))
    lookup.update({tuple(r[k] for k in keys):r for r in macros})
    configs=sorted({(r['stage'],r['profile'],r['model'],r['calibration']) for r in per_seed})
    combined=[]; pairs=[]
    domains=dict(f.GROUPS,untouched_macro='untouched')
    for stage,profile,model,cal in configs:
        for g,exposure in domains.items():
            for cond in f.CONDITIONS:
                for metric in f.METRICS:
                    combined.append(dict(stage=stage,profile=profile,model=model,calibration=cal,generator=g,exposure=exposure,condition=cond,metric=metric,
                                         **stats([lookup[stage,s,profile,model,cal,g,exposure,cond][metric] for s in f.seeds.SEEDS])))
    contrasts=[('W1 - W0','D','W1','C','global','original','raw')]+[('average - global','C','average','C','global',p,cal) for p in c.PROFILES for cal in ('raw','temperature')]
    for title,ls,lm,rs,rm,profile,cal in contrasts:
        for g,exposure in domains.items():
            for cond in f.CONDITIONS:
                for metric in f.METRICS:
                    values=[lookup[ls,s,profile,lm,cal,g,exposure,cond][metric]-lookup[rs,s,profile,rm,cal,g,exposure,cond][metric] for s in f.seeds.SEEDS]
                    pairs.append(dict(contrast=title,profile=profile,calibration=cal,generator=g,exposure=exposure,condition=cond,metric=metric,**stats(values)))
    return {'metrics_initializations.csv':raw,'metrics_each_seed.csv':per_seed,'untouched_macro_each_seed.csv':macros,'metrics_mean_sd.csv':combined,'paired_differences.csv':pairs}


def markdown(tables):
    lines=['# Round 7 final reserved-test evaluation','',
           'Frozen saved models and thresholds; no fitting, calibration, threshold choice or epoch selection on test data. Values are mean ± sample SD over three run seeds. Gate initialization metrics are averaged within a run first.',
           '', '**Main evidence:** GLIDE, Midjourney, VQDM and Wukong (untouched by Round 7 development). ADM/BigGAN are previously inspected; SD15 is in-domain.',
           '', '“Untouched” is relative to this repository’s development. The earlier disclosure remains: 116 of 3,750 reserved images appeared in an independent pilot that did not inform Round 7 settings. These are not described as universally blind domains.',
           '', 'One shared pool of 500 real originals is predicted once per model/condition and reused in generator comparisons. The untouched macro averages four generator metrics; it does not duplicate real images in pooled metrics.','']
    rows=tables['metrics_mean_sd.csv']
    index={(r['stage'],r['profile'],r['model'],r['calibration'],r['generator'],r['condition'],r['metric']):r['mean_plus_minus_sd'] for r in rows}
    configs=sorted({(r['stage'],r['profile'],r['model'],r['calibration']) for r in rows})
    for gen in ('untouched_macro',*f.UNTOUCHED,'ADM','BigGAN','SD15'):
        lines += [f'## {gen}', '', '| Stage | Profile | Model | Calibration | Condition | AUROC | AP | Balanced accuracy | Real FPR |', '|---|---|---|---|---|---:|---:|---:|---:|']
        for stage,profile,model,cal in configs:
            for cond in f.CONDITIONS:
                values=[index[stage,profile,model,cal,gen,cond,m] for m in f.METRICS]
                lines.append(f'| {stage} | {profile} | {model} | {cal} | {cond} | '+' | '.join(values)+' |')
        lines.append('')
    lines+=['## Paired differences: untouched macro','','| Contrast | Profile | Calibration | Condition | Metric | Difference |','|---|---|---|---|---|---:|']
    for r in tables['paired_differences.csv']:
        if r['generator']=='untouched_macro':
            lines.append(f"| {r['contrast']} | {r['profile']} | {r['calibration']} | {r['condition']} | {r['metric']} | {r['mean_plus_minus_sd']} |")
    lines+=['','Every CSV includes individual seed values or initialization-level results. Positive paired AUROC/AP/BA differences favor the left model; positive FPR differences are worse. SD is descriptive over three runs, not a population confidence interval. All models and conditions were declared before opening the test; no post-test winner selection or settings changes are made.','']
    return '\n'.join(lines).encode('utf-8')


def build(freeze,records):
    raw=[]
    for task,(models,condition) in f.task_groups(freeze['models']).items():
        path=f.OUT/'predictions'/f'{task}.csv'
        receipt=c.read(f.OUT/'receipts'/f'{task}.json')
        c.require(receipt['freeze_sha256']==c.sha(f.FREEZE) and receipt['predictions_sha256']==c.sha(path),'Prediction receipt mismatch')
        raw+=metric_rows(c.read_csv(path),models,condition,records)
    tables=summarize(raw,freeze['models'])
    payloads={name:f.csv_bytes(rows) for name,rows in tables.items()}
    payloads['SUMMARY.md']=markdown(tables)
    return payloads,tables
