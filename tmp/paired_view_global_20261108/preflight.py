"""Read-only cache/split/frozen-control preflight for one paired-view head pilot."""
import sys
sys.dont_write_bytecode = True
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'output/analysis/paired_view_global_20261108'
SEEDS = [20261101,20261102,20261103]
TUNE = ['clean','jpeg70','resize05']
CONDITIONS = ['clean','jpeg70','jpeg50','resize05']
INPUTS = {}


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def checked(path,expected=None):
    path=Path(path).resolve(); relative=path.relative_to(ROOT).as_posix()
    assert 'final' not in relative.lower() and 'reserved' not in relative.lower(), relative
    digest=sha(path); assert expected is None or digest==expected,relative
    INPUTS[relative]=digest
    return path


def read(path,expected=None):
    return json.loads(checked(path,expected).read_text(encoding='utf-8'))


def global_features(number,seed,part,condition,matched,bundle,expected_ids):
    paths=sorted((ROOT/f'artifacts/round8_a/seed{number}/features/{part}_{condition}_{int(matched)}').glob('*.npz'))
    assert paths
    ids,blocks=[],[]
    for path in paths:
        checked(path,bundle['feature_sha256'][path.relative_to(ROOT).as_posix()])
        with np.load(path,allow_pickle=False) as f:
            current=f['ids'].tolist(); x=f['g'].copy()
            assert json.loads(str(f['identity']))==[seed,part,condition,matched,current]
            assert str(f['signature'])==bundle['signature']
            assert hashlib.sha256(x.tobytes()).hexdigest()==str(f['g_sha256'])
            assert x.shape==(len(current),512) and x.dtype==np.float32 and np.isfinite(x).all()
            ids.extend(current); blocks.append(x)
    assert ids==sorted(expected_ids) and len(ids)==len(set(ids))
    return np.concatenate(blocks),len(paths)


def main():
    assert not (OUT/'preflight.json').exists()
    old_protocol=read(ROOT/'results/round8_a/protocol.json')
    for name,digest in old_protocol['code_sha256'].items():checked(ROOT/name,digest)
    for name,digest in old_protocol['weights_sha256'].items():checked(ROOT/'.cache/clip-vit-base-patch32'/name,digest)
    records={}; development_ids=[]
    for name in ['fitting_pool_manifest.jsonl','development_manifest.jsonl']:
        relative='results/round7/'+name
        p=checked(ROOT/relative,old_protocol['input_sha256'][relative])
        for line in p.read_text(encoding='utf-8').splitlines():
            r=json.loads(line); assert r['source_id'] not in records; records[r['source_id']]=r
            if name=='development_manifest.jsonl':development_ids.append(r['source_id'])
    relative='results/round7/step_b_isolation.json'
    isolation=read(ROOT/relative,old_protocol['input_sha256'][relative])
    deny=set(isolation['reserved_test_source_ids'])
    allowed=set(isolation['allowed_source_ids'])
    assert set(records)==allowed and allowed.isdisjoint(deny)
    reports=[]; partitions={}; total_batches=0
    for number,seed in enumerate(SEEDS,1):
        relative=f'results/round7/assignments_seed{number}.json'
        assignment=read(ROOT/relative,old_protocol['input_sha256'][relative])
        assert assignment['seed']==seed
        parts={k:sorted(assignment['partitions'][k]) for k in ['expert_fit','selection']}
        parts['development']=sorted(development_ids)
        partitions[seed]=parts
        assert Counter(records[i]['generator'] for i in parts['expert_fit'])==Counter(Real=750,SD15=750)
        assert Counter(records[i]['generator'] for i in parts['selection'])==Counter(Real=125,SD15=125)
        assert Counter(records[i]['generator'] for i in parts['development'])==Counter(Real=300,ADM=300,BigGAN=300)
        for a,b in combinations(parts,2):
            assert set(parts[a]).isdisjoint(parts[b])
            for field in ['source_bytes_sha256','pixel_sha256']:
                assert {records[i][field] for i in parts[a]}.isdisjoint({records[i][field] for i in parts[b]})
        assert set(sum(parts.values(),[]))<=allowed and set(sum(parts.values(),[])).isdisjoint(deny)
        bundles={}; controls={}; training={}; errors={}; batches=0
        for matched,arm in [(False,'unmatched'),(True,'matched')]:
            complete=read(ROOT/f'results/round8_a/seed{number}/{arm}/complete.json')
            assert complete['status']=='completed'
            stem=f'artifacts/round8_a/seed{number}/{arm}'
            bundle=read(ROOT/stem/'bundle.json',complete['files_sha256'][stem+'/bundle.json'])
            assert bundle['run_seed']==seed and bundle['arm']==arm
            assert bundle['signature']==complete['signature']==sha(ROOT/'results/round8_a/protocol.json')
            checked(ROOT/stem/'experts.joblib',complete['files_sha256'][stem+'/experts.joblib'])
            assert bundle['experts_sha256']==sha(ROOT/stem/'experts.joblib')
            controls[arm]=joblib.load(ROOT/stem/'experts.joblib')['global_head']
            params=controls[arm].named_steps['logisticregression'].get_params()
            assert params['C']==1. and params['solver']=='liblinear' and params['max_iter']==2000 and params['random_state']==20261018
            bundles[arm]=bundle
            x,n=global_features(number,seed,'expert_fit','clean',matched,bundle,parts['expert_fit'])
            training[arm]=x;batches+=n
        x=np.vstack([training['unmatched'],training['matched']])
        weight=np.full(len(x),.5)
        assert x.shape==(3000,512) and float(weight.sum())==1500.
        assert np.array_equal(weight[:1500]+weight[1500:],np.ones(1500))
        for part,conditions in [('selection',TUNE),('development',CONDITIONS)]:
            for condition in conditions:
                x,n=global_features(number,seed,part,condition,False,bundles['matched'],parts[part]);batches+=n
                for arm in ['unmatched','matched']:
                    raw=bundles[arm]['raw_batches'][part]
                    rows=[(r,z) for r,z in zip(raw['rows'],raw['zg']) if r['condition']==condition]
                    assert [r['source_id'] for r,z in rows]==parts[part]
                    expected=np.array([z for r,z in rows])
                    actual=controls[arm].decision_function(x)
                    error=float(abs(actual-expected).max())
                    assert error<1e-10,(seed,part,condition,arm,error)
                    errors[f'{part}/{condition}/{arm}']=error
        total_batches+=batches
        reports.append(dict(run_seed=seed,expert_fit_originals=1500,selection_originals=250,development_originals=900,
                            training_rows=3000,training_dimensions=512,per_view_sample_weight=.5,total_sample_weight=1500.,
                            paired_ids_exact=True,within_seed_source_identity_byte_pixel_isolation=True,
                            original_control_max_logit_error=max(errors.values()),control_logit_errors=errors,
                            verified_feature_batches=batches))
    overlaps=[]
    for a,b in combinations(SEEDS,2):
        overlaps.append(dict(seed_a=a,seed_b=b,expert_fit_overlap=len(set(partitions[a]['expert_fit'])&set(partitions[b]['expert_fit'])),
            selection_overlap=len(set(partitions[a]['selection'])&set(partitions[b]['selection'])),
            expert_a_selection_b_overlap=len(set(partitions[a]['expert_fit'])&set(partitions[b]['selection'])),
            selection_a_expert_b_overlap=len(set(partitions[a]['selection'])&set(partitions[b]['expert_fit']))))
    assert all(partitions[s]['development']==partitions[SEEDS[0]]['development'] for s in SEEDS)
    assert all(sha(ROOT/p)==digest for p,digest in INPUTS.items())
    report=dict(status='passed',script_sha256=sha(__file__),per_seed=reports,source_ids=partitions,
        verified_feature_batches=total_batches,input_sha256=INPUTS,cross_seed_overlap=overlaps,
        shared_development_originals=900,deny_count=len(deny),guard_violations=[],
        no_models_fitted=True,no_image_decode_or_extraction=True,no_final_test_access=True,
        limits=['Head prediction reproduction uses frozen controls only; new paired head is not trained here.',
                'Three source splits overlap and share development IDs; paired original bootstrap must reuse one draw per group across conditions, arms and seeds.',
                'StandardScaler on equally weighted 3000 rows equals fitting all rows with constant .5 weights. LR must explicitly receive .5 weights; inherited clip.head hardcodes ones.',
                'All three arms must select thresholds only from the same original selection clean/jpeg70/resize05; report JPEG50 without selection.'])
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'preflight.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ['status','per_seed','verified_feature_batches','cross_seed_overlap']},indent=2))


if __name__=='__main__':main()
