"""Guarded fresh-paired CLIP extraction for a frozen JPEG95 stability probe.

--canary only reproduces old unperturbed views and repeats identical inference.
--extract requires the parent's frozen protocol and exact extractor hash.
No fitting, label-based scoring, downloads, or final-test access.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import io
import json
import platform
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/analysis/recompression_reliability_20261108'
sys.path.insert(0, str(ROOT / 'scripts'))
import round7_baseline_common as common
import round7_baseline_data as images
import round7_baseline_clip as clip
import numpy as np
import torch
import joblib
from PIL import Image, ImageOps

TUNE = ['clean', 'jpeg70', 'resize05']
CONDS = ['clean', 'jpeg70', 'jpeg50', 'resize05']
INPUT_HASHES = {}
EVENTS = defaultdict(set)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(relative, expected=None):
    assert 'final' not in relative.lower() and 'reserved' not in relative.lower(), relative
    p = ROOT / relative
    actual = sha(p)
    assert expected is None or actual == expected, relative
    INPUT_HASHES[relative] = actual
    return json.loads(p.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def array_hash(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def save_npz(path, arrays, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), path
    hashes = {k: array_hash(v) for k, v in arrays.items()}
    payload = dict(arrays, metadata=np.array(json.dumps(metadata, sort_keys=True)),
                   **{k+'_sha256': np.array(v) for k,v in hashes.items()})
    tmp = path.with_suffix('.npz.pending')
    with tmp.open('xb') as f:
        np.savez_compressed(f, **payload)
    tmp.rename(path)
    return hashes


def prepare_scope():
    previous = read('results/round8_a/protocol.json')
    study = 'output/analysis/multigen_gate_20261108/run_v1_codefix1'
    marker = read(study+'/complete.json')
    splits = read(study+'/splits.json', marker['outputs_sha256']['splits.json'])
    source = set().union(*[set(v['tuning_ids']) for k,v in splits['sources'].items() if k.endswith('/source_only')])
    development = set().union(*[set(v) for v in splits['evaluation_ids'].values()])
    assert not source & development and len(source)==668 and len(development)==900
    allowed = source | development
    records = {}
    for name in ('fitting_pool_manifest.jsonl','development_manifest.jsonl'):
        relative = 'results/round7/'+name
        p = ROOT/relative
        actual = sha(p)
        assert actual == previous['input_sha256'][relative]
        INPUT_HASHES[relative] = actual
        for line in p.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            if r['source_id'] in allowed: records[r['source_id']] = r
    isolation = read('results/round7/step_b_isolation.json', previous['input_sha256']['results/round7/step_b_isolation.json'])
    deny = set(isolation['reserved_test_source_ids'])  # ID-only guard, never its manifest.
    assert set(records)==allowed and allowed<=set(isolation['allowed_source_ids']) and not allowed&deny
    assert Counter(records[i]['generator'] for i in source)==Counter(Real=335,SD15=333)
    assert Counter(records[i]['generator'] for i in development)==Counter(Real=300,ADM=300,BigGAN=300)
    for name in ('scripts/round7_baseline_clip.py','scripts/round7_baseline_data.py','scripts/round7_baseline_common.py'):
        actual=sha(ROOT/name); assert actual==previous['code_sha256'][name];INPUT_HASHES[name]=actual
    weights = ROOT/'.cache/clip-vit-base-patch32'
    for name, expected in previous['weights_sha256'].items():
        actual=sha(weights/name);assert actual==expected;INPUT_HASHES[(weights/name).relative_to(ROOT).as_posix()]=actual
    assert common.PIL.__version__=='12.2.0'
    assert images.VIEW_SEEDS==[202610220,202610221,202610222,202610223]
    assert torch.cuda.is_available(), 'CUDA unavailable; do not switch precision or device silently'
    assert torch.cuda.get_device_name(0)==previous['environment']['device']
    for package in ('torch','torchvision','transformers','numpy','scipy','scikit-learn','Pillow'):
        assert version(package)==previous['environment'][package], (package,version(package),previous['environment'][package])
    torch.set_num_threads(4)
    torch.manual_seed(20261101)
    runtime = dict(python=platform.python_version(), platform=platform.platform(),
        packages={p:version(p) for p in ('torch','torchvision','transformers','numpy','scipy','scikit-learn','Pillow')},
        cuda_runtime=torch.version.cuda, device='cuda:0', device_name=torch.cuda.get_device_name(0),
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'), encoder_dtype='float16', output_dtype='float32',
        batch_size=32, torch_threads=torch.get_num_threads(),
        allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,allow_tf32_cudnn=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
        CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    assert platform.python_version()==previous['environment']['python']
    for identity in allowed:
        assert original_path(identity, records, deny).is_file(),identity
    return records,deny,{'source_tuning':sorted(source),'development':sorted(development)},weights,runtime


def touch(ids, action, records, deny):
    ids=set(ids)
    assert ids<=set(records) and not ids&deny, f'Forbidden ID at {action}'
    EVENTS[action].update(ids)


def original_path(identity, records, deny):
    touch([identity],'original_path_lookup',records,deny)
    assert identity.startswith('train:') and identity[6:].isdigit()
    return ROOT/'artifacts/round8_a/allowed_images'/(identity.replace(':','_')+'.bin')


def decode(identity, records, deny):
    touch([identity],'source_bytes_hash_and_decode',records,deny)
    p=original_path(identity,records,deny)
    data=p.read_bytes()
    assert hashlib.sha256(data).hexdigest()==records[identity]['source_bytes_sha256'],identity
    with Image.open(io.BytesIO(data)) as im:
        result=ImageOps.exif_transpose(im).convert('RGB')
    digest=hashlib.sha256(str(result.size).encode()+result.tobytes()).hexdigest()
    assert digest==records[identity]['pixel_sha256'],identity
    return result


def load_model(weights):
    from transformers import CLIPVisionModelWithProjection
    model=CLIPVisionModelWithProjection.from_pretrained(weights,local_files_only=True).to('cuda:0').eval()
    model.requires_grad_(False);model.half()
    assert next(model.parameters()).dtype==torch.float16
    return model


def encode(model, pictures):
    g=clip.encode(model,pictures,'cuda:0')
    l=np.stack([clip.encode(model,[images.shuffled(im,s) for im in pictures],'cuda:0') for s in images.VIEW_SEEDS],axis=1)
    assert g.dtype==l.dtype==np.float32 and g.shape==(len(pictures),512) and l.shape==(len(pictures),4,512)
    assert np.isfinite(g).all() and np.isfinite(l).all()
    return dict(g=g,l=l)


def recompress(picture):
    assert picture.mode=='RGB' and picture.size==(224,224)
    buffer=io.BytesIO()
    picture.save(buffer,format='JPEG',quality=95,subsampling=0,optimize=False,progressive=False)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        result=decoded.convert('RGB')
    assert result.size==picture.size
    return result


def difference(a,b):
    d=a.astype(np.float64)-b.astype(np.float64)
    return dict(max_absolute=float(np.max(np.abs(d))),rmse=float(np.sqrt(np.mean(d*d))),
                bit_exact=bool(np.array_equal(a,b)))


def run_canary(records,deny,groups,weights,runtime):
    assert not (OUT/'canary.json').exists(),'Existing canary; no automatic repeat'
    old_marker=read('results/round8_a/seed1/matched/complete.json')
    old=read('artifacts/round8_a/seed1/matched/bundle.json',old_marker['files_sha256']['artifacts/round8_a/seed1/matched/bundle.json'])
    experts={}
    for number in (1,2,3):
        marker=read(f'results/round8_a/seed{number}/matched/complete.json')
        relative=f'artifacts/round8_a/seed{number}/matched/experts.joblib'
        assert sha(ROOT/relative)==marker['files_sha256'][relative]
        INPUT_HASHES[relative]=marker['files_sha256'][relative]
        experts[number]=joblib.load(ROOT/relative)
    model=load_model(weights)
    measurements=[]
    start=time.monotonic()
    for condition in CONDS:
        relative=f'artifacts/round8_a/seed1/features/development_{condition}_0/00000.npz'
        p=ROOT/relative
        assert sha(p)==old['feature_sha256'][relative]
        INPUT_HASHES[relative]=old['feature_sha256'][relative]
        with np.load(p,allow_pickle=False) as cache:
            identities=cache['ids'].tolist()
            expected={k:cache[k].copy() for k in ('g','l')}
            assert json.loads(str(cache['identity']))==[20261101,'development',condition,False,identities]
            for k in expected:assert array_hash(expected[k])==str(cache[k+'_sha256'])
        assert len(identities)==32 and set(identities)<=set(groups['development'])
        touch(identities,'canary_unperturbed_encode',records,deny)
        pictures=[images.view(decode(i,records,deny),'original',condition) for i in identities]
        t=time.monotonic();first=encode(model,pictures);repeat=encode(model,pictures);torch.cuda.synchronize()
        seconds=time.monotonic()-t
        row=dict(condition=condition,ids=identities,old_cache=relative,seconds_two_identical_passes=seconds,
                 old_cache_difference={k:difference(first[k],expected[k]) for k in first},
                 identical_input_repeat_noise={k:difference(first[k],repeat[k]) for k in first},head_logit_difference={})
        for number,heads in experts.items():
            row['head_logit_difference'][str(number)]={}
            for branch,key in (('global','g'),('local','l')):
                head=heads[branch+'_head']
                a=first[key] if branch=='global' else first[key].mean(axis=1)
                b=expected[key] if branch=='global' else expected[key].mean(axis=1)
                r=repeat[key] if branch=='global' else repeat[key].mean(axis=1)
                row['head_logit_difference'][str(number)][branch]=dict(old_cache=difference(head.decision_function(a),head.decision_function(b)),
                                                                           identical_input_repeat=difference(head.decision_function(a),head.decision_function(r)))
        path=OUT/'canary_features'/f'{condition}.npz'
        save_npz(path,dict(ids=np.array(identities),first_g=first['g'],first_l=first['l'],repeat_g=repeat['g'],repeat_l=repeat['l']),
                 dict(condition=condition,old_cache=relative,runtime=runtime,stage='unperturbed baseline only'))
        row['canary_file']=path.relative_to(OUT).as_posix();row['canary_file_sha256']=sha(path)
        measurements.append(row)
        print(json.dumps(dict(canary=condition,old_g_max=row['old_cache_difference']['g']['max_absolute'],
              old_l_max=row['old_cache_difference']['l']['max_absolute'],repeat_g_max=row['identical_input_repeat_noise']['g']['max_absolute'],
              repeat_l_max=row['identical_input_repeat_noise']['l']['max_absolute'],seconds=seconds)),flush=True)
    result=dict(status='measured',no_label_performance_computed=True,runtime=runtime,extractor_sha256=sha(Path(__file__)),
                source_counts={k:len(v) for k,v in groups.items()},measurements=measurements,
                input_sha256=INPUT_HASHES.copy(),seconds=time.monotonic()-start,
                encoder_image_passes=4*32*5*2,guard_events={k:len(v) for k,v in EVENTS.items()})
    write(OUT/'canary.json',result)
    print(json.dumps(dict(status='canary_complete',seconds=result['seconds'],extractor_sha256=result['extractor_sha256'])),flush=True)


def extract(protocol_path,records,deny,groups,weights,runtime):
    assert protocol_path.is_file(),'Frozen protocol required'
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    spec=protocol['extraction']
    expected=dict(source_scope='source_only_tuning_union',source_conditions=TUNE,development_conditions=CONDS,
                  jpeg_quality=95,subsampling=0,batch_size=32,device='cuda:0',model_dtype='float16',paired_fresh_baseline=True)
    assert all(spec[k]==v for k,v in expected.items()),'Extraction protocol differs from fixed implementation'
    assert spec['extractor_sha256']==sha(Path(__file__)),'Extractor changed after freeze'
    canary=load('output/analysis/recompression_reliability_20261108/canary.json')
    assert canary['runtime']==runtime and canary['extractor_sha256']==sha(Path(__file__))
    assert not (OUT/'extraction_freeze.json').exists(),'Extraction already frozen; explicit recovery required'
    touch(sum(groups.values(),[]),'new_probe_scope',records,deny)
    freeze=dict(protocol_path=str(protocol_path.resolve()),protocol_sha256=sha(protocol_path),extraction=spec,
                runtime=runtime,canary_sha256=sha(OUT/'canary.json'),input_sha256=INPUT_HASHES.copy(),
                source_ids=groups['source_tuning'],development_ids=groups['development'],
                image_condition_pairs=5604,encoder_image_passes=56040,started_utc=datetime.now(timezone.utc).isoformat())
    write(OUT/'extraction_freeze.json',freeze)
    model=load_model(weights)
    started=time.monotonic();manifest=[];pairs=0
    for group,identities in groups.items():
        conditions=TUNE if group=='source_tuning' else CONDS
        for start in range(0,len(identities),32):
            chosen=identities[start:start+32]
            pictures=[decode(i,records,deny) for i in chosen]
            for condition in conditions:
                touch(chosen,'base_and_probe_encode/'+group+'/'+condition,records,deny)
                current=[images.view(im,'original',condition) for im in pictures]
                probe=[recompress(im) for im in current]  # Whole image first; encode() shuffles afterward.
                base=encode(model,current);changed=encode(model,probe)
                metadata=dict(scope=group,condition=condition,ids=chosen,protocol_sha256=freeze['protocol_sha256'],
                    extractor_sha256=spec['extractor_sha256'],runtime_sha256=hashlib.sha256(json.dumps(runtime,sort_keys=True).encode()).hexdigest(),
                    original_source_bytes_sha256=[records[i]['source_bytes_sha256'] for i in chosen],
                    original_pixel_sha256=[records[i]['pixel_sha256'] for i in chosen],
                    chain=['original_decode_exif_rgb','ImageOps.fit_224_bicubic',condition,'probe_only_JPEG95_subsampling0','global_or_four_PatchShuffle56_views'],
                    probe=dict(quality=95,subsampling=0,optimize=False,progressive=False),view_seeds=images.VIEW_SEEDS,
                    fresh_paired_same_batch_order=True)
                path=OUT/'features'/group/condition/f'{start:05d}.npz'
                arrays=dict(ids=np.array(chosen),base_g=base['g'],base_l=base['l'],probe_g=changed['g'],probe_l=changed['l'])
                ah=save_npz(path,arrays,metadata)
                manifest.append(dict(path=path.relative_to(OUT).as_posix(),sha256=sha(path),ids=chosen,scope=group,condition=condition,
                                     n=len(chosen),array_sha256=ah))
                pairs+=len(chosen)
            elapsed=time.monotonic()-started
            print(json.dumps(dict(scope=group,originals_done=min(start+32,len(identities)),originals_total=len(identities),
                  image_condition_pairs=pairs,seconds=round(elapsed,2))),flush=True)
    assert pairs==5604 and len(manifest)==179
    assert {i for e in manifest for i in e['ids']}==set(records)
    seen=[(e['scope'],e['condition'],i) for e in manifest for i in e['ids']]
    assert len(seen)==len(set(seen))==5604
    assert all(sha(ROOT/p)==h for p,h in INPUT_HASHES.items())
    write(OUT/'feature_manifest.json',dict(status='complete',protocol_sha256=freeze['protocol_sha256'],entries=manifest,
        paired_arrays=['base_g','base_l','probe_g','probe_l'],unique_originals=1568,image_condition_pairs=pairs,
        runtime=runtime,view_seeds=images.VIEW_SEEDS,guard_events={k:len(v) for k,v in EVENTS.items()},
        no_label_performance_computed=True,no_final_test_access=True,seconds=time.monotonic()-started))
    write(OUT/'extraction_complete.json',dict(status='completed',manifest_sha256=sha(OUT/'feature_manifest.json'),
        extraction_freeze_sha256=sha(OUT/'extraction_freeze.json'),image_condition_pairs=pairs,encoder_image_passes=56040,
        feature_batches=len(manifest),seconds=time.monotonic()-started))
    print(json.dumps(dict(status='completed',image_condition_pairs=pairs,feature_batches=len(manifest),seconds=time.monotonic()-started)),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=('resources','canary','extract'))
    p.add_argument('--protocol',type=Path)
    args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    records,deny,groups,weights,runtime=prepare_scope()
    if args.mode=='resources':
        print(json.dumps(dict(runtime=runtime,source_counts={k:len(v) for k,v in groups.items()},
            images_available=len(records),weights=str(weights),image_condition_pairs=5604,encoder_image_passes=56040,
            extractor_sha256=sha(Path(__file__))),indent=2));return
    if args.mode=='canary':run_canary(records,deny,groups,weights,runtime);return
    assert args.protocol is not None,'--protocol required'
    extract(args.protocol,records,deny,groups,weights,runtime)


if __name__=='__main__':main()
