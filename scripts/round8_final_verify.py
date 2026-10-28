"""Read-only independent metric recomputation and final provenance verification."""
import sys
sys.dont_write_bytecode = True
import argparse
import json
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
import round8_final_common as f
from round8_final_run import test_records
import round8_final_report as report


def verify():
    frozen = f.freeze()
    f.protect()
    f.require(frozen['models'] == f.declarations(), 'Model declarations changed')
    from round8_final_data import verify_data
    data = verify_data()
    marker = f.read(f.OUT / 'complete.json')
    f.require(marker['status'] == 'completed' and marker['freeze_sha256'] == f.sha(f.FREEZE), 'Invalid completion')
    for name, digest in marker['files_sha256'].items():
        f.require(f.sha(f.ROOT / name) == digest, f'Final output changed: {name}')
    records = test_records(frozen)
    tasks = report.task_groups(frozen['models'])
    f.require({p.stem for p in (f.OUT / 'evaluation/predictions').glob('*.csv')} == set(tasks)
              and {p.stem for p in (f.OUT / 'evaluation/receipts').glob('*.json')} == set(tasks), 'Unexpected/missing task files')
    run = f.read(f.OUT / 'evaluation/run.json')
    f.require(run == dict(freeze_sha256=f.sha(f.FREEZE), tasks=list(tasks), source_ids=sorted(records)), 'Run identity/coverage changed')
    metric_rows = f.c.read_csv(f.OUT / 'metrics_initializations.csv')
    lookup = {(r['model_id'], r['generator'], r['condition']): r for r in metric_rows}
    expected = {(m['id'], g, cond) for m in frozen['models'] for g in f.GENERATORS for cond in f.CONDITIONS}
    f.require(len(lookup) == len(metric_rows) and set(lookup) == expected, 'Metric cell coverage failed')
    count = 0
    for task, (models, condition) in tasks.items():
        path = f.OUT / 'evaluation/predictions' / (task + '.csv')
        receipt = f.read(f.OUT / 'evaluation/receipts' / (task + '.json'))
        touched = set(receipt['inference_touched_source_ids'])
        f.require(touched <= set(records), 'Non-test inference access recorded')
        if not receipt['recovered_atomic_predictions']:
            f.require(touched == set(records), 'Incomplete inference access record')
        grouped = report.validate(f.c.read_csv(path), models, condition, records)
        for m in models:
            count += len(grouped[m['id']])
            for generator in f.GENERATORS:
                cell = [r for r in grouped[m['id']] if r['generator'] in ('Real', generator)]
                y = np.asarray([int(r['label']) for r in cell])
                z = np.asarray([float(r['logit']) for r in cell])
                predicted = expit(z) >= m['threshold']
                independent = dict(auroc=float(roc_auc_score(y, z)), average_precision=float(average_precision_score(y, z)),
                    balanced_accuracy=float(balanced_accuracy_score(y, predicted)), real_false_positive_rate=float(predicted[y == 0].mean()))
                saved = lookup[m['id'], generator, condition]
                f.require(int(saved['n_real']) == int(sum(y == 0)) == 500 and int(saved['n_fake']) == int(sum(y == 1)) == 500, 'Sample counts changed')
                for metric, value in independent.items():
                    f.require(abs(float(saved[metric]) - value) <= 1e-12, f'Independent metric mismatch: {metric}')
    f.require(count == marker['prediction_rows'] == 96 * 3500 * 4, 'Prediction count mismatch')
    payloads, tables = report.build(frozen, records)
    for name, payload in payloads.items():
        f.require((f.OUT / name).read_bytes() == payload, f'Summary/macro/paired recomputation mismatch: {name}')
    return dict(status='passed', model_instances=len(frozen['models']), prediction_rows=count,
        metric_cells=len(metric_rows), fresh_test=data, frozen_files=len(frozen['files_sha256']),
        scope='Hashes, exclusions, seeded sampling, exact source/model/condition coverage, saved thresholds, independent metrics, three-seed aggregation and paired/cross-test summaries. No model inference.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    with f.lock():
        value = verify()
        if args.report:
            f.write(f.OUT / 'verification.json', value)
        print(json.dumps(value, indent=2))
