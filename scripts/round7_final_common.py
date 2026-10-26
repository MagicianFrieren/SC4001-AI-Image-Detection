"""Step E freeze, immutable storage, reserved-test-only guard and declarations."""
import sys
sys.dont_write_bytecode = True
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
import numpy as np
import round7_baseline_common as c
import round7_seed_support as seeds
from round7_stepD_common import publish, json_bytes, csv_bytes

ROOT=c.ROOT
FREEZE=ROOT/'results/round7/final_freeze.json'
OUT=ROOT/'results/round7/final_test'
ART=ROOT/'artifacts/round7/final_test'
TEST='results/round7/reserved_test_manifest.jsonl'
CONDITIONS=tuple(c.CONDITIONS)
GROUPS={'SD15':'in-domain','ADM':'previously inspected','BigGAN':'previously inspected',
        'GLIDE':'untouched','Midjourney':'untouched','VQDM':'untouched','Wukong':'untouched'}
UNTOUCHED=('GLIDE','Midjourney','VQDM','Wukong')
METRICS=('auroc','average_precision','balanced_accuracy','real_false_positive_rate')
CODE=('round7_final_common.py','round7_final_models.py','round7_final_run.py','round7_final_report.py','round7_final_verify.py','round7_final_test.py')


def notebook_source_hash(path):
    book=c.read(path)
    source=[{'cell_type':r['cell_type'],'source':''.join(r['source']) if isinstance(r['source'],list) else r['source']} for r in book['cells']]
    return hashlib.sha256(json_bytes(source)).hexdigest()


def declarations():
    """Read development SETTINGS, never test data. No fitting or selection."""
    result=[]
    for seed in seeds.SEEDS:
        n=seeds.number(seed)
        for profile in c.PROFILES:
            folder=f'results/round7/baselines_seed{n}'
            settings_path=f'{folder}/clip_{profile}_settings.json'
            saved=c.read(ROOT/settings_path)
            for key,setting in sorted(saved['settings'].items()):
                calibration,model,init=key.split('/')
                item=dict(id=f'C_s{n}_{profile}_{calibration}_{model}_{init}',stage='C',seed=seed,profile=profile,calibration=calibration,model=model,initialization=int(init),
                          family='clip',head=f'artifacts/round7/baselines_seed{n}/clip_{profile}_experts.joblib',settings_path=settings_path,settings_key=key,
                          threshold=setting['threshold'],global_temperature=setting['global_temperature'],local_temperature=setting['local_temperature'],fixed_alpha=setting['fixed_alpha'])
                if model.endswith('gate'):
                    c.require(key in saved['gates'],'Missing saved gate')
                    item['gate_key']=key
                result.append(item)
            rs=f'{folder}/resnet_{profile}_settings.json'; saved=c.read(ROOT/rs)
            checkpoint=f'artifacts/round7/baselines_seed{n}/checkpoints/{profile}/epoch_{saved["best_epoch"]:03d}.pt'
            for cal,setting in saved['settings'].items():
                result.append(dict(id=f'C_s{n}_{profile}_{cal}_resnet50',stage='C',seed=seed,profile=profile,calibration=cal,model='resnet50',initialization=seed,
                                   family='resnet',checkpoint=checkpoint,settings_path=rs,threshold=setting['threshold'],temperature=setting['temperature'],batch_size=saved['history'][-1]['batch_size']))
        for variant in ('W1','W2'):
            folder=f'results/round7/stepD_whole_image/seed{n}/{variant}'
            setting=c.read(ROOT/folder/'settings.json')
            item=dict(id=f'D_s{n}_original_raw_{variant}',stage='D',seed=seed,profile='original',calibration='raw',model=variant,initialization=seed,
                      family='clip' if variant=='W1' else 'resnet',settings_path=f'{folder}/settings.json',threshold=setting['threshold'],temperature=1.)
            if variant=='W1':
                item['head']=f'artifacts/round7/stepD_whole_image/seed{n}/W1/global_head.joblib'
            else:
                item['checkpoint']=f'artifacts/round7/stepD_whole_image/seed{n}/W2/checkpoints/original/epoch_{setting["best_epoch"]:03d}.pt'
                item['batch_size']=setting['history'][-1]['batch_size']
            result.append(item)
    # Check declaration against all Step C development model/profile/calibration/init cells.
    for seed in seeds.SEEDS:
        rows=c.read_csv(ROOT/f'results/round7/baselines_seed{seeds.number(seed)}/metrics.csv')
        expected={(r['profile'],r['model'],r['calibration'],int(r['seed'])) for r in rows if r['partition']=='development'}
        actual={(r['profile'],r['model'],r['calibration'],r['initialization']) for r in result if r['stage']=='C' and r['seed']==seed}
        c.require(expected==actual,'Declared model coverage differs from development')
    c.require(len(result)==138 and len({r['id'] for r in result})==138,'Wrong model declaration count')
    for r in result:
        c.require(0<=r['threshold']<=1,'Invalid saved threshold')
        for field in ('head','checkpoint','settings_path'):
            if field in r:
                c.require((ROOT/r[field]).is_file(),f'Missing model artifact {r[field]}')
    return sorted(result,key=lambda r:r['id'])


def task_groups(models):
    groups={}
    for r in models:
        key=f'{r["stage"]}_s{seeds.number(r["seed"])}_{r["profile"]}_{r["family"]}'
        if r['stage']=='D':
            key+='_'+r['model']
        groups.setdefault(key,[]).append(r)
    return {key+'_'+condition:(rows,condition) for key,rows in sorted(groups.items()) for condition in CONDITIONS}


def build_freeze():
    c.require(not FREEZE.exists(),'Final freeze already exists; refusing replacement')
    c.require(not OUT.exists(),'Final results directory already exists; refuse a new experiment')
    seeds.check_environment()
    models=declarations()
    # Validate completion hashes without loading reserved records.
    for seed in seeds.SEEDS:
        seeds.check_completed_seed(seed)
    from round7_stepD_verify import verify_run
    for seed in seeds.SEEDS:
        for variant in ('W1','W2'):
            verify_run(seed,variant)
    prior=c.read(ROOT/'results/round7/data_protocol.json')
    test_hash=prior['files_sha256'][TEST]
    c.require(all(c.read(ROOT/f'results/round7/assignments_seed{i}.json')['reserved_test_sha256']==test_hash for i in (1,2,3)),'Test identity differs')
    paths=set()
    for folder in [ROOT/'scripts',ROOT/'configs',ROOT/'docs',ROOT/'results/round7']:
        if folder.exists():
            paths.update(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for folder in [ROOT/f'artifacts/round7/baselines_seed{i}' for i in (1,2,3)]+[ROOT/'artifacts/round7/stepD_whole_image',ROOT/'artifacts/round7/seed1_frozen_entrypoints']:
        paths.update(p for p in folder.rglob('*') if p.is_file())
    paths.update(p for p in ROOT.glob('requirements*') if p.is_file())
    hashes={}; notebooks={}
    for path in sorted(paths):
        name=path.relative_to(ROOT).as_posix()
        if path.suffix=='.ipynb':
            notebooks[name]=notebook_source_hash(path)
            continue
        # Copy the already-pinned hash: do not even open reserved manifest before freeze.
        hashes[name]=test_hash if name==TEST else c.sha(path)
    value=dict(id='round7_final_once',seeds=list(seeds.SEEDS),models=models,conditions=list(CONDITIONS),generator_groups=GROUPS,
               expected_counts={'Real':500,'SD15':250,**{g:500 for g in GROUPS if g!='SD15'}},files_sha256=hashes,
               notebook_sources_sha256=notebooks,notebook_hash_rule='SHA256 of canonical JSON list of cell_type and joined source for every cell. Runtime outputs, execution counts and notebook metadata excluded so running the notebook does not alter frozen code.',
               test_manifest=TEST,test_manifest_sha256=test_hash,
               test_hash_origin='Copied from immutable data protocol and three matching assignments without opening reserved manifest; directly checked only AFTER final_freeze.json exists.',
               metrics=list(METRICS),reporting='Per generator/condition, each run seed and mean ± sample SD (ddof=1). Average gate-initialization METRICS within seed first; never pool nine gate initializations as independent seeds.',
               macro='Equal mean across GLIDE/Midjourney/VQDM/Wukong per condition and metric; shared 500 real originals counted once per model/condition predictions. No pooled repeated-real metric.',
               paired='W1 minus W0 (Step C global, original/raw) and average minus global (each Step C profile/calibration), within each run seed; per generator/condition and untouched-generator macro, all four metrics.',
               selection='NONE. Saved epochs, checkpoints, thresholds, fixed fusion weights, gate parameters and temperatures are restored unchanged.',
               inference='Unmodified original/common128 geometry and four conditions on ORIGINAL test images. No Step D format matching at evaluation. Shared frozen CLIP feature caches, batch32, FP16 CUDA/FP32 normalized features; ResNet FP32 at saved inference batch size.',
               once='One immutable run identity. Atomic batch caches/chunks resume incomplete work only. Refuse evaluator after final_test/complete.json exists, including --resume. Verification remains read-only and repeatable. Unpublished work interrupted before atomic save may be repeated; completed chunks are never rescored.',
               isolation='Existing Step C/D guards unchanged. Step E has a separate exact reserved-ID allowlist and rejects all fitting/development IDs. Access requires this freeze file.',
               exposure_note='Untouched means not used for Round 7 development. Prior independent pilot exposure remains disclosed: 116/3750 test images overlapped a separate pilot that informed no Round 7 setting; these are not claimed universally blind domains.',
               environment=c.read(ROOT/'results/round7/baselines_seed1/summarize_complete.json')['environment'])
    publish(FREEZE,json_bytes(value))
    print(f'Frozen {len(models)} model instances and {len(hashes)} files BEFORE test-manifest access.',flush=True)
    return value


def freeze(full=False):
    c.require(FREEZE.is_file(),'Freeze must exist before test access')
    f=c.read(FREEZE)
    for name,digest in f.get('notebook_sources_sha256',{}).items():
        c.require(notebook_source_hash(ROOT/name)==digest,f'Frozen notebook source changed: {name}')
    if full:
        for name,digest in f['files_sha256'].items():
            c.require(c.sha(ROOT/name)==digest,f'Frozen file changed: {name}')
    else:
        # Cheap code/protocol/model-setting validation; large artifacts checked at load/full verification.
        for name,digest in f['files_sha256'].items():
            if name.startswith('scripts/') or name.endswith(('settings.json','protocol.json')) or name==TEST:
                c.require(c.sha(ROOT/name)==digest,f'Frozen file changed: {name}')
    return f


def checked(path,f):
    path=Path(path); name=path.relative_to(ROOT).as_posix()
    c.require(name in f['files_sha256'] and c.sha(path)==f['files_sha256'][name],f'Frozen artifact mismatch: {name}')
    return path


def records(f):
    c.require(FREEZE.exists(),'No test reads before freeze')
    checked(ROOT/TEST,f)
    rows=c.rows(ROOT/TEST)
    result={r['source_id']:r for r in rows}
    c.require(len(result)==len(rows)==3750,'Wrong reserved count/duplicates')
    c.require(dict(Counter(r['generator'] for r in rows))==f['expected_counts'],'Reserved generator counts changed')
    isolation=c.read(checked(ROOT/'results/round7/step_b_isolation.json',f))
    c.require(set(result)==set(isolation['reserved_test_source_ids']) and not set(result)&set(isolation['allowed_source_ids']),'Reserved/fitting overlap')
    c.require(all(r['label']==int(r['generator']!='Real') for r in rows),'Test labels inconsistent')
    return result


class TestGuard:
    def __init__(self,records_):
        c.require(FREEZE.exists(),'Test guard requires freeze')
        self.allowed=set(records_); self.events={}
    def touch(self,ids,action):
        ids=set(ids)
        c.require(ids<=self.allowed,'FINAL ISOLATION: non-reserved ID touched')
        self.events.setdefault(action,set()).update(ids)


def npz_publish(path,ids,arrays,identity):
    stream=io.BytesIO()
    checks={k:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() for k,a in arrays.items()}
    np.savez_compressed(stream,ids=np.array(ids),identity=np.array(identity),checks=np.array(json.dumps(checks,sort_keys=True)),**arrays)
    publish(path,stream.getvalue())


def npz_read(path,ids,identity):
    with np.load(path,allow_pickle=False) as data:
        c.require(data['ids'].tolist()==ids and str(data['identity'])==identity,'Cache identity mismatch')
        checks=json.loads(str(data['checks']))
        result={k:data[k].copy() for k in checks}
    for k,a in result.items():
        c.require(np.isfinite(a).all() and hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()==checks[k],'Cache corrupt')
    return result
