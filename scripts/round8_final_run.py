"""User-run freeze and once-only final inference. No training/selection entrypoints."""
import sys
sys.dont_write_bytecode = True
import argparse
import time
from collections import Counter
import numpy as np
from scipy.special import expit
import round8_final_common as f
import round8_final_inference as inference
import round8_final_report as report


def test_records(frozen):
    f.require(f.FREEZE.is_file() and f.sha(f.TEST) == frozen['test_manifest_sha256'], 'Freeze required before evaluation manifest access')
    rows = f.rows(f.TEST)
    records = {r['source_id']: r for r in rows}
    f.require(len(records) == len(rows) == 3500 and dict(Counter(r['generator'] for r in rows)) == frozen['expected_counts'], 'Wrong fresh test coverage')
    previous = {r['source_id'] for path in f.EXCLUSIONS for r in f.rows(f.ROOT / path)}
    f.require(not previous.intersection(records), 'Fresh-test isolation failed')
    f.require(all(r['label'] == int(r['generator'] != 'Real') for r in rows), 'Bad test labels')
    return records


def assert_runnable(resume):
    if (f.OUT / 'complete.json').exists():
        raise FileExistsError('Round 8 final evaluation is COMPLETE. A second run is forbidden, including --resume. Run the read-only verifier instead.')
    if (f.OUT / 'evaluation').exists() and not resume:
        raise FileExistsError('Incomplete run exists. Only --resume may continue its immutable batches.')


def execute(resume):
    assert_runnable(resume)
    frozen = f.freeze()
    f.require(frozen['models'] == f.declarations(), 'Frozen model settings changed')
    from round8_final_data import verify_data
    verify_data()  # Hash-only verification; no images or model scores.
    records = test_records(frozen)
    guard = f.Guard(records)
    tasks = report.task_groups(frozen['models'])
    identity = f.sha(f.FREEZE)
    f.write(f.OUT / 'evaluation/run.json', dict(freeze_sha256=identity, tasks=list(tasks), source_ids=sorted(records)))
    inference.prepare_originals('test', records, guard)
    features = inference.features('test', records, guard, identity, f.ROOT / frozen['weights'])
    ids = sorted(records)
    for number, (task, (models, condition)) in enumerate(tasks.items(), 1):
        path = f.OUT / 'evaluation/predictions' / (task + '.csv')
        receipt_path = f.OUT / 'evaluation/receipts' / (task + '.json')
        if receipt_path.exists():
            receipt = f.read(receipt_path)
            f.require(receipt['freeze_sha256'] == identity and receipt['prediction_sha256'] == f.sha(path), 'Completed prediction task changed')
            print(f'[{number}/{len(tasks)}] verified completed {task}', flush=True)
            continue
        started = time.perf_counter()
        recovered = path.exists()
        task_guard = f.Guard(records)
        if recovered:
            values = f.c.read_csv(path)
        else:
            task_guard.touch(ids, 'predict/' + task)
            if models[0]['family'] == 'resnet':
                f.require(len(models) == 1, 'One raw ResNet per task')
                scores = {models[0]['id']: (inference.resnet(models[0], condition, records, task_guard, identity + '|' + task), None)}
            else:
                portable = f.read(f.ROOT / models[0]['portable'])
                scores = inference.score(models, features[condition], portable)
            values = []
            for m in models:
                logits, alpha = scores[m['id']]
                f.require(len(logits) == len(ids) and np.isfinite(logits).all(), 'Invalid model scores')
                for index, source_id in enumerate(ids):
                    logit = float(logits[index])
                    values.append(dict(model_id=m['id'], source_id=source_id, generator=records[source_id]['generator'],
                        label=records[source_id]['label'], condition=condition, logit=logit, probability_fake=float(expit(logit)),
                        threshold=m['threshold'], alpha_global='' if alpha is None else float(alpha[index])))
            report.validate(values, models, condition, records)
            f.publish(path, f.csv_bytes(values))
        report.validate(values, models, condition, records)
        f.write(receipt_path, dict(freeze_sha256=identity, prediction_sha256=f.sha(path), task=task,
            model_ids=[m['id'] for m in models], condition=condition, source_ids=ids,
            inference_touched_source_ids=sorted(task_guard.touched), recovered_atomic_predictions=recovered,
            isolation='Exact fresh-test allowlist; all prior source IDs rejected', seconds=time.perf_counter() - started))
        print(f'[{number}/{len(tasks)}] completed {task}', flush=True)
    payloads, tables = report.build(frozen, records)
    for name, payload in payloads.items():
        f.publish(f.OUT / name, payload)
    f.freeze()
    f.protect()
    paths = [p for p in (f.OUT / 'evaluation').rglob('*') if p.is_file() and not p.name.endswith('.tmp')]
    paths += [f.OUT / name for name in payloads]
    f.write(f.OUT / 'complete.json', dict(status='completed', freeze_sha256=identity, model_instances=len(frozen['models']),
        test_originals=len(records), prediction_rows=len(frozen['models']) * len(records) * len(f.CONDITIONS),
        row_counts={name: len(values) for name, values in tables.items()},
        files_sha256={f.rel(p): f.sha(p) for p in sorted(paths)}))
    print('COMPLETE. Do not rerun inference. Use round8_final_verify.py and SUMMARY.md.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--check-freeze', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    with f.lock():
        if args.check_freeze:
            f.require(not args.freeze and not args.resume, '--check-freeze is read only')
            value = f.freeze()
            print(f'Freeze verified: {len(value["models"])} declared model instances; no inference.')
        elif args.freeze:
            f.require(not args.resume, '--freeze and --resume are separate actions')
            f.build_freeze()
        else:
            execute(args.resume)
