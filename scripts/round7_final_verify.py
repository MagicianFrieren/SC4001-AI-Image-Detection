"""Read-only final verifier: freeze hashes, exact coverage, independent metrics."""
import sys
sys.dont_write_bytecode = True
import argparse
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
import round7_final_common as f
from round7_final_common import c


def verify():
    frozen=f.freeze(full=True)
    c.require(frozen['models']==f.declarations(),'Saved settings/model declaration changed')
    marker=c.read(f.OUT/'complete.json')
    c.require(marker['status']=='completed' and marker['freeze_sha256']==c.sha(f.FREEZE),'Final completion identity mismatch')
    for name,digest in marker['files_sha256'].items():
        c.require(c.sha(f.ROOT/name)==digest,f'Final output changed: {name}')
    records=f.records(frozen); tasks=f.task_groups(frozen['models'])
    state=c.read(f.OUT/'run.json')
    c.require(state['freeze_sha256']==c.sha(f.FREEZE) and state['tasks']==list(tasks),'Run plan changed')
    saved=c.read_csv(f.OUT/'metrics_initializations.csv')
    lookup={(r['model_id'],r['generator'],r['condition']):r for r in saved}
    c.require(len(lookup)==len(saved)==len(frozen['models'])*7*4,'Raw metric coverage mismatch')
    from round7_final_report import validate_rows,build
    predictions=0
    for task,(models,condition) in tasks.items():
        receipt=c.read(f.OUT/'receipts'/f'{task}.json')
        c.require(receipt['freeze_sha256']==c.sha(f.FREEZE) and receipt['source_ids']==sorted(records),'Access receipt identity mismatch')
        c.require(receipt['model_ids']==[r['id'] for r in models] and receipt['condition']==condition and receipt['non_reserved_ids_touched']==0,'Access receipt mismatch')
        path=f.OUT/'predictions'/f'{task}.csv'
        c.require(c.sha(path)==receipt['predictions_sha256'],'Predictions changed')
        rows=c.read_csv(path); groups=validate_rows(rows,models,condition,records)
        predictions+=len(rows)
        for model in models:
            for generator in f.GROUPS:
                cell=[r for r in groups[model['id']] if r['generator'] in ('Real',generator)]
                y=np.array([int(r['label']) for r in cell]); z=np.array([float(r['logit']) for r in cell])
                yes=expit(z)>=model['threshold']
                values=dict(auroc=float(roc_auc_score(y,z)),average_precision=float(average_precision_score(y,z)),
                            balanced_accuracy=float(balanced_accuracy_score(y,yes)),real_false_positive_rate=float(yes[y==0].mean()))
                old=lookup[model['id'],generator,condition]
                c.require(int(old['n_real'])==500 and int(old['n_fake'])==frozen['expected_counts'][generator],'Sample count mismatch')
                for name,value in values.items():
                    c.require(np.isclose(float(old[name]),value,rtol=0,atol=1e-12),f'Independent metric mismatch: {name}')
    c.require(predictions==138*3750*4,'Wrong final prediction count')
    payloads,tables=build(frozen,records)
    for name,data in payloads.items():
        c.require((f.OUT/name).read_bytes()==data,f'Summary/paired/macro recomputation differs: {name}')
    return dict(status='passed',frozen_files=len(frozen['files_sha256']),model_instances=len(frozen['models']),prediction_rows=predictions,
                metric_cells=len(saved),non_reserved_ids_touched=0,scope='Frozen inputs and final output hashes; independent metrics from saved predictions; exact coverage; seed/gate aggregation, paired differences and untouched macro; no images or models loaded')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',action='store_true',help='Create an immutable verification.json, or verify its identical contents')
    args=parser.parse_args(); result=verify()
    if args.report:
        f.publish(f.OUT/'verification.json',f.json_bytes(result),resume=True)
    print(c.json.dumps(result,indent=2))


if __name__=='__main__':
    main()
