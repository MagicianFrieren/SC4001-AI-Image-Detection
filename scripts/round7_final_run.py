"""Frozen final test. --resume continues only an incomplete run, never a completed one."""
import sys
sys.dont_write_bytecode = True
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import hashlib
from contextlib import contextmanager
import round7_final_common as f
from round7_final_common import c


@contextmanager
def execution_lock():
    f.ART.mkdir(parents=True,exist_ok=True)
    path=f.ART/'execution.lock'
    if not path.exists():
        try:
            with path.open('xb') as stream:
                stream.write(b'0')
        except FileExistsError:
            pass
    with path.open('rb+') as stream:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name=='nt':
                msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                fcntl.flock(stream,fcntl.LOCK_UN)


def assert_runnable(resume):
    if (f.OUT/'complete.json').exists():
        raise FileExistsError('Final evaluation already completed. Refusing a second run, even with --resume. Use the read-only verifier.')
    if f.OUT.exists() and not resume:
        raise FileExistsError('Final work already exists. Only --resume may continue the SAME incomplete run.')


def execute(resume):
    assert_runnable(resume)
    f.seeds.check_environment()
    frozen=f.freeze(full=True)
    with execution_lock():
        assert_runnable(resume)
        tasks=f.task_groups(frozen['models'])
        state={'freeze_sha256':c.sha(f.FREEZE),'tasks':list(tasks),'mode':'one frozen evaluation; incomplete resume only'}
        f.publish(f.OUT/'run.json',f.json_bytes(state),resume)
        records=f.records(frozen); guard=f.TestGuard(records)
        from round7_final_models import prepare,features,predict_task
        prepare(records,guard,frozen)
        features(records,guard,frozen)
        from round7_final_report import validate_rows,build
        for i,(task,(models,condition)) in enumerate(tasks.items(),1):
            path=f.OUT/'predictions'/f'{task}.csv'; receipt=f.OUT/'receipts'/f'{task}.json'
            if receipt.exists():
                saved=c.read(receipt)
                c.require(saved['freeze_sha256']==state['freeze_sha256'] and saved['predictions_sha256']==c.sha(path),'Completed task changed')
                print(f'[{i}/{len(tasks)}] verified completed {task}',flush=True)
                continue
            if path.exists():
                rows=c.read_csv(path)
                validate_rows(rows,models,condition,records)
                # Atomic predictions survived a crash before receipt publication. Do not rescore.
                recovered=True
            else:
                task_guard=f.TestGuard(records)
                rows=predict_task(task,models,condition,records,task_guard,frozen)
                validate_rows(rows,models,condition,records)
                f.publish(path,f.csv_bytes(rows))
                recovered=False
            f.publish(receipt,f.json_bytes(dict(task=task,freeze_sha256=state['freeze_sha256'],predictions_sha256=c.sha(path),
                                                model_ids=[r['id'] for r in models],condition=condition,source_ids=sorted(records),
                                                non_reserved_ids_touched=0,recovered_atomic_predictions=recovered)))
            print(f'[{i}/{len(tasks)}] completed {task}',flush=True)
        payloads,tables=build(frozen,records)
        for name,data in payloads.items():
            f.publish(f.OUT/name,data,resume=True)
        # Verify frozen inputs again before marking the only run complete.
        f.freeze(full=True)
        paths=[p for p in f.OUT.rglob('*') if p.is_file() and not p.name.endswith('.tmp') and p.name!='complete.json']
        result={'status':'completed','freeze_sha256':state['freeze_sha256'],'model_instances':len(frozen['models']),'reserved_originals':len(records),
                'prediction_tasks':len(tasks),'row_counts':{name:len(rows) for name,rows in tables.items()},
                'files_sha256':{p.relative_to(f.ROOT).as_posix():c.sha(p) for p in paths},'non_reserved_ids_touched':0}
        f.publish(f.OUT/'complete.json',f.json_bytes(result))
        print('Final evaluation COMPLETE. A second evaluation is now prohibited. Run the read-only verifier.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',action='store_true',help='Write freeze before any test-manifest access; refuse replacement')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.freeze:
        f.build_freeze()
    else:
        execute(args.resume)


if __name__=='__main__':
    main()
