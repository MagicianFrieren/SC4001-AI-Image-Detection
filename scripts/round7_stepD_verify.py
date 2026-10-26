"""Read-only Step D verification: predictions, thresholds, metrics and isolation."""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import defaultdict
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import round7_stepD_common as d
from round7_stepD_common import c
from round7_verify_baselines import independently_threshold


def check_rows(rows,records,parts,deny,variant,seed,tuning=False):
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        identity = row['source_id']; part = row['partition']; cond = row['condition']
        c.require(identity not in deny and identity in records, 'TEST ISOLATION: forbidden prediction')
        c.require(part in ('selection',) if tuning else part in ('selection','development'), 'Wrong prediction partition')
        c.require(identity in parts[part], 'Wrong partition membership')
        c.require(row['variant']==variant and int(row['seed'])==seed, 'Wrong variant/run seed')
        c.require(row['generator']==records[identity]['generator'] and int(row['label'])==records[identity]['label'], 'Wrong label/generator')
        c.require(row['view']==('format_matched' if tuning else 'original'), 'Tuning/reporting view mixed up')
        key = part,cond,identity
        c.require(key not in seen, 'Duplicate prediction'); seen.add(key)
        c.require(np.isfinite(float(row['logit'])), 'Nonfinite prediction')
        if not tuning:
            c.require(row['profile']=='original' and row['calibration']=='raw','Wrong profile/calibration')
            c.require(np.isclose(float(row['probability_fake']),expit(float(row['logit'])),atol=1e-14,rtol=0),'Wrong probability')
            c.require(np.isfinite(float(row['threshold'])) and 0<=float(row['threshold'])<=1,'Invalid threshold')
        groups[part,cond].append(row)
    expected = {(part,cond,i) for part in (['selection'] if tuning else ['selection','development']) for cond in (c.TUNE if tuning else c.CONDITIONS) for i in parts[part]}
    c.require(seen==expected,'Incomplete condition/source coverage')
    return groups


def recompute(rows):
    result = []
    for variant,seed in sorted({(r['variant'],int(r['seed'])) for r in rows}):
        for gen,part in d.DOMAINS.items():
            for cond in c.CONDITIONS:
                cell = [r for r in rows if r['variant']==variant and int(r['seed'])==seed and r['generator'] in ('Real',gen) and r['partition']==part and r['condition']==cond]
                y = np.array([int(r['label']) for r in cell]); z=np.array([float(r['logit']) for r in cell])
                cuts = {float(r['threshold']) for r in cell}
                c.require(len(cuts)==1,'Mixed thresholds')
                decisions = expit(z)>=next(iter(cuts))
                result.append(dict(variant=variant,seed=seed,generator=gen,partition=part,condition=cond,n_real=int(sum(y==0)),n_fake=int(sum(y==1)),
                                   auroc=float(roc_auc_score(y,z)),balanced_accuracy=float(balanced_accuracy_score(y,decisions)),real_false_positive_rate=float(decisions[y==0].mean())))
    return result


def compare_metrics(saved, actual):
    keys=('variant','seed','generator','partition','condition')
    indexed={tuple(str(r[k]) for k in keys):r for r in saved}
    c.require(len(indexed)==len(saved)==len(actual),'Metric duplicates/count mismatch')
    for row in actual:
        key=tuple(str(row[k]) for k in keys)
        c.require(key in indexed,'Missing metric cell')
        old=indexed[key]
        for name in (*d.METRICS,'n_real','n_fake'):
            c.require(np.isclose(float(old[name]),row[name],atol=1e-12,rtol=0),f'Metric mismatch: {name}')


def check_events(audit,records,parts,deny):
    c.require(not audit['violations'],'Isolation violations recorded')
    guard=d.Guard(records,parts,deny)
    for action,ids in audit['events'].items():
        guard.touch(ids,action)


def verify_run(seed,variant):
    from round7_stepD_run import complete
    lock=d.lock_implementation()
    records,parts,deny=d.configure(seed,variant)
    folder=c.OUT
    c.require(complete(folder),f'Incomplete {seed}/{variant}')
    protocol=c.read(folder/'protocol.json')
    expected={'seed':seed,'variant':variant,'stepD_protocol_sha256':c.sha(d.PROTOCOL),
              'implementation_sha256':c.sha(d.OUT/'implementation.json'),'settings':d.validate_protocol()}
    c.require(protocol==expected,'Run protocol mismatch')
    rows=c.read_csv(folder/'predictions.csv'); tuning=c.read_csv(folder/'tuning_predictions.csv')
    check_rows(rows,records,parts,deny,variant,seed)
    check_rows(tuning,records,parts,deny,variant,seed,True)
    cut=independently_threshold([int(r['label']) for r in tuning],expit([float(r['logit']) for r in tuning]))
    c.require(all(np.isclose(float(r['threshold']),cut,atol=1e-12,rtol=0) for r in rows),'Threshold not fitted on matched selection')
    settings=c.read(folder/'settings.json')
    c.require(np.isclose(settings['threshold'],cut,atol=1e-12,rtol=0) and settings['tuning_view']=='format_matched','Settings threshold mismatch')
    compare_metrics(c.read_csv(folder/'metrics.csv'),recompute(rows))
    draws=c.read_csv(folder/'quality_draws.csv')
    seen=set()
    for row in draws:
        identity=row['source_id']; part=row['partition']
        c.require(part in ('expert_fit','selection') and identity in parts[part] and identity not in seen,'Wrong quality draw membership')
        seen.add(identity)
        c.require(int(row['label'])==records[identity]['label'] and int(row['quality'])==d.quality(seed,identity),'Wrong deterministic quality')
    c.require(seen==set(parts['expert_fit']+parts['selection']),'Missing quality draws')
    audit=c.read(folder/'audit.json')
    c.require(audit['protocol_sha256']==c.sha(folder/'protocol.json'),'Audit protocol mismatch')
    check_events(audit,records,parts,deny)
    for path in (c.ART/'access_logs').glob('*.json'):
        log=c.read(path)
        c.require(log['isolation_sha256']==c.sha(c.DATA/'step_b_isolation.json'),'Wrong isolation hash')
        check_events(log,records,parts,deny)
    required={'tune_threshold':parts['selection'],'report/selection':parts['selection'],'report/development':parts['development']}
    required['fit_clip_expert' if variant=='W1' else ('resnet_resume_expert_partition' if 'resnet_resume_expert_partition' in audit['events'] else 'fit_resnet_expert_only')]=parts['expert_fit']
    for action,ids in required.items():
        c.require(set(audit['events'].get(action,[]))==set(ids),f'Missing access evidence: {action}')
    if variant=='W2':
        history=settings['history']
        c.require([r['epoch'] for r in history]==list(range(1,11)),'Missing ResNet epochs')
        best=max(history,key=lambda r:r['selection_macro_auroc'])
        c.require(settings['best_epoch']==best['epoch'] and settings['best_score']==best['selection_macro_auroc'],'Wrong selected epoch')
        score=np.mean([roc_auc_score([int(r['label']) for r in tuning if r['condition']==cond],[float(r['logit']) for r in tuning if r['condition']==cond]) for cond in c.TUNE])
        c.require(np.isclose(score,best['selection_macro_auroc'],atol=1e-12,rtol=0),'Best checkpoint tuning scores differ')
        for r in history:
            c.require(np.isclose(r['next_lr'],1e-4*(1+np.cos(np.pi*r['epoch']/10))/2,atol=1e-14,rtol=0),'Wrong schedule')
            c.require(np.isfinite(r['train_loss']),'Nonfinite training loss')
            if r['batch_size']!=32:
                c.require(settings['oom_events'] and all(e['reason']=='CUDA OOM' for e in settings['oom_events']),'Unexplained batch change')
        c.require(len(list((c.ART/'checkpoints/original').glob('epoch_*.pt')))==11,'Missing checkpoints')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=d.seeds.SEEDS)
    parser.add_argument('--variant',choices=['W1','W2'])
    parser.add_argument('--summary',action='store_true')
    parser.add_argument('--preservation',action='store_true')
    args=parser.parse_args()
    if args.preservation:
        print(f'Preservation PASS: {d.check_preservation()} existing files unchanged')
    if args.summary:
        from round7_stepD_report import run
        run(resume=True,verify_only=True)
    elif args.seed and args.variant:
        rows=verify_run(args.seed,args.variant)
        print(f'PASS {args.seed}/{args.variant}: {len(rows)} predictions, 12 metric cells, test IDs touched=0')
    elif not args.preservation:
        parser.error('Choose --seed/--variant, --summary, or --preservation')


if __name__=='__main__':
    main()
