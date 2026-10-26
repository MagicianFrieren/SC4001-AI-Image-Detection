"""Fixed-criteria three-seed Step D summary; saved predictions only."""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import numpy as np
import round7_stepD_common as d
from round7_stepD_common import c


def statistics(values):
    c.require(len(values)==3 and np.isfinite(values).all(),'Three finite seed values required')
    mean=float(np.mean(values)); sd=float(np.std(values,ddof=1))
    return {**{f'seed_{seed}':float(v) for seed,v in zip(d.seeds.SEEDS,values)},'mean':mean,'sd':sd,'mean_plus_minus_sd':f'{mean:.6f} ± {sd:.6f}'}


def tables(metric_rows):
    lookup={(r['variant'],int(r['seed']),r['generator'],r['condition']):r for r in metric_rows}
    expected={(v,s,g,k) for v in ('W0','W1','W2') for s in d.seeds.SEEDS for g in d.DOMAINS for k in c.CONDITIONS}
    c.require(set(lookup)==expected and len(lookup)==len(metric_rows),'Incomplete/duplicate summary cells')
    def value(v,s,g,k,m='auroc'):
        return float(lookup[v,s,g,k][m])
    aggregate=[]; differences=[]; costs=[]; criteria=[]
    for g,part in d.DOMAINS.items():
        for cond in c.CONDITIONS:
            for metric in d.METRICS:
                for variant in ('W0','W1','W2'):
                    aggregate.append(dict(variant=variant,generator=g,partition=part,condition=cond,metric=metric,**statistics([value(variant,s,g,cond,metric) for s in d.seeds.SEEDS])))
                for a,b in [('W1','W0'),('W2','W0'),('W2','W1')]:
                    differences.append(dict(contrast=a+' - '+b,generator=g,partition=part,condition=cond,metric=metric,**statistics([value(a,s,g,cond,metric)-value(b,s,g,cond,metric) for s in d.seeds.SEEDS])))
    for variant in ('W1','W2'):
        for metric in d.METRICS:
            direction=1 if metric=='real_false_positive_rate' else -1
            costs.append(dict(variant=variant,generator='SD15',condition='clean',metric=metric,positive_means='worse than W0',**statistics([direction*(value(variant,s,'SD15','clean',metric)-value('W0',s,'SD15','clean',metric)) for s in d.seeds.SEEDS])))
        for seed in d.seeds.SEEDS:
            jpeg=float(np.mean([value(variant,seed,'ADM',k) for k in ('jpeg70','jpeg50')]))
            checks=[('ADM JPEG70/JPEG50 mean AUROC',jpeg, '>=', .50, jpeg>=.50-1e-12)]
            for g in ('ADM','BigGAN'):
                drop=value('W0',seed,g,'clean')-value(variant,seed,g,'clean')
                checks.append((f'clean {g} AUROC drop vs W0',drop,'<=',.02,drop<=.02+1e-12))
            for name,v,operator,limit,passed in checks:
                criteria.append(dict(variant=variant,seed=seed,criterion=name,value=v,operator=operator,limit=limit,status='PASS' if passed else 'FAIL'))
    success={v:{'every_seed':{str(s):'PASS' if all(r['status']=='PASS' for r in criteria if r['variant']==v and r['seed']==s) else 'FAIL' for s in d.seeds.SEEDS},
                'all_criteria_every_seed':'PASS' if all(r['status']=='PASS' for r in criteria if r['variant']==v) else 'FAIL'} for v in ('W1','W2')}
    return {'metrics_each_seed.csv':metric_rows,'metrics_mean_sd.csv':aggregate,'paired_differences.csv':differences,'clean_SD15_cost.csv':costs,'criteria.csv':criteria},success


def markdown(tables,success):
    lines=['# Round 7 Step D: whole-image format matching','',
           'Raw scores; original profile. W0 is reused Step C global CLIP. W1/W2 tune on format-matched selection views and report on original images. All values below are mean ± sample SD over three seeds.','',
           '| Variant | Domain | Condition | AUROC | Balanced accuracy | Real FPR |','|---|---|---|---:|---:|---:|']
    rows=tables['metrics_mean_sd.csv']
    index={(r['variant'],r['generator'],r['condition'],r['metric']):r['mean_plus_minus_sd'] for r in rows}
    for v in ('W0','W1','W2'):
        for g in d.DOMAINS:
            for cond in c.CONDITIONS:
                lines.append(f'| {v} | {g} | {cond} | '+' | '.join(index[v,g,cond,m] for m in d.METRICS)+' |')
    lines+=['','## Frozen success criteria','','| Variant | Seed | Criterion | Value | Requirement | Result |','|---|---|---|---:|---|---|']
    for r in tables['criteria.csv']:
        lines.append(f"| {r['variant']} | {r['seed']} | {r['criterion']} | {r['value']:.6f} | {r['operator']} {r['limit']} | {r['status']} |")
    for v,result in success.items():
        lines+=['',f"**{v}: {result['all_criteria_every_seed']}** (all criteria in every seed)."]
    lines+=['','## Clean SD15 cost','','Positive means worse than W0: W0 minus variant for AUROC/BA, variant minus W0 for FPR.','','| Variant | Metric | Cost, mean ± SD |','|---|---|---:|']
    for r in tables['clean_SD15_cost.csv']:
        lines.append(f"| {r['variant']} | {r['metric']} | {r['mean_plus_minus_sd']} |")
    lines+=['','Paired W1−W0, W2−W0 and W2−W1 differences for every condition/domain/metric and each seed are in `paired_differences.csv`. Positive AUROC/BA differences favour the left variant; positive FPR differences are worse.',
            '', 'Real images become double-compressed, fakes single-compressed. Native resolution and previous compression history remain confounded. W1−W0 holds architecture fixed; W2 also changes architecture. These results assess the intervention, not definitive proof of sole causation.',
            '', 'SD15 reporting uses selection identities already used for tuning. ADM/BigGAN are inspected development domains. The seeds share fitting/development pools; sample SD is descriptive, not a confidence interval. No reserved-test images or predictions were used.','']
    return '\n'.join(lines).encode('utf-8')


def run(resume=False,verify_only=False):
    from round7_stepD_verify import verify_run,check_rows,recompute
    d.lock_implementation()
    predictions=[]; w0=[]; hashes={}
    for seed in d.seeds.SEEDS:
        for variant in ('W1','W2'):
            predictions.extend(verify_run(seed,variant))
            for name in ('complete.json','predictions.csv','tuning_predictions.csv','settings.json'):
                path=c.OUT/name; hashes[path.relative_to(d.ROOT).as_posix()]=c.sha(path)
        records,parts,deny=d.configure(seed,'W0')
        baseline=d.w0_rows(seed,records,parts,d.Guard(records,parts,deny))
        check_rows(baseline,records,parts,deny,'W0',seed)
        w0.extend(baseline)
    predictions+=w0
    result,success=tables(recompute(predictions))
    payloads={name:d.csv_bytes(rows) for name,rows in result.items()}
    payloads.update({'W0_predictions.csv':d.csv_bytes(w0),'success.json':d.json_bytes(success),'SUMMARY.md':markdown(result,success),
                     'provenance.json':d.json_bytes({'protocol_sha256':c.sha(d.PROTOCOL),'implementation_sha256':c.sha(d.OUT/'implementation.json'),'inputs_sha256':hashes,
                     'W0_sources':{k:v for k,v in c.read(d.PROTOCOL)['input_sha256'].items() if k.endswith('clip_original_predictions.csv')},'sd_ddof':1,'test_source_ids_touched':0})})
    payloads['complete.json']=d.json_bytes({'status':'completed','files_sha256':{name:hashlib.sha256(data).hexdigest() for name,data in payloads.items()}})
    for name,data in payloads.items():
        path=d.OUT/'summary'/name
        if verify_only:
            c.require(path.exists() and path.read_bytes()==data,f'Summary differs: {name}')
        else:
            d.publish(path,data,resume)
    for row in result['criteria.csv']:
        print(f"{row['variant']} seed {row['seed']}: {row['criterion']} = {row['value']:.6f}: {row['status']}")
    print(c.json.dumps(success,indent=2))
    print('Summary verified' if verify_only else f'Summary saved: {d.OUT / "summary"}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args(); run(args.resume,args.verify_only)


if __name__=='__main__':
    main()
