"""Describe gate score-input saturation; no training or model selection."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,csv,hashlib
import numpy as np
import joblib
from scipy.special import expit
import objective_diagnostics as diag

ROOT,RUN,OUT,study=diag.ROOT,diag.RUN,diag.OUT,diag.study
frozen=diag.read('freeze.json')
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in frozen['input_sha256'].items())
splits=diag.read('splits.json')
sources=study.cache.verify_and_collect();rows=[]
for number,source in sources.items():
    seed=source['run_seed'];experts=joblib.load(ROOT/f'artifacts/round8_a/seed{number}/matched/experts.joblib')
    original=study.cache.reconstruct_original(source,number,'gate_fit',experts)
    dev=study.cache.array_batch(source['matched']['raw_batches']['development'])
    for scenario in ['source_only','mixed_ADM','mixed_BigGAN']:
        want=set(splits['sources'][f'{seed}/{scenario}']['fit_ids']);keep=want&study.ids(original)
        fit=study.take(original,keep)
        if want-keep:fit=study.merge(fit,study.take(dev,want-keep))
        targets=['ADM','BigGAN'] if scenario=='source_only' else (['ADM'] if scenario=='mixed_BigGAN' else ['BigGAN'])
        batches=[('fit',fit)]+[('heldout_'+target,study.take(dev,study.ids(dev,'Real')|study.ids(dev,target),study.CONDITIONS)) for target in targets]
        for part,b in batches:
            gen=np.array([r['generator'] for r in b['rows']]);cond=np.array([r['condition'] for r in b['rows']])
            pg,pl=expit(b['zg']),expit(b['zl'])
            for c in sorted(set(cond)):
                for g in sorted(set(gen)):
                    m=(cond==c)&(gen==g)
                    lo=(pg[m]<.01)&(pl[m]<.01);hi=(pg[m]>.99)&(pl[m]>.99)
                    rows.append(dict(run_seed=seed,scenario=scenario,partition=part,condition=c,generator=g,n=int(m.sum()),
                        both_prob_below_001=float(lo.mean()),both_prob_above_099=float(hi.mean()),both_same_extreme=float((lo|hi).mean()),
                        mean_pg=float(pg[m].mean()),mean_pl=float(pl[m].mean()),mean_abs_probability_gap=float(abs(pg[m]-pl[m]).mean()),
                        mean_abs_logit_gap=float(abs(b['zg'][m]-b['zl'][m]).mean())))
diag.writecsv('objective_probability_extremes.csv',rows)
means=diag.pool(rows,['scenario','partition','condition','generator'],['both_prob_below_001','both_prob_above_099','both_same_extreme','mean_pg','mean_pl','mean_abs_probability_gap','mean_abs_logit_gap'])
diag.writecsv('objective_probability_extremes_summary.csv',means)
print(json.dumps([r for r in means if r['scenario']=='source_only' and r['condition'].startswith('jpeg')],indent=2))
