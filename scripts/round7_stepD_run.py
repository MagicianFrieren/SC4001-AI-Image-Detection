"""Step D W1/W2 GPU runner. New paths only; use --resume after interruption."""
import sys
sys.dont_write_bytecode = True
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import io
import time
import numpy as np
from scipy.special import expit
import round7_stepD_common as d
from round7_stepD_common import c
import round7_baseline_clip as clip
import round7_baseline_resnet as rn


def run_protocol(seed, variant, resume):
    lock = d.lock_implementation()
    p = d.validate_protocol()
    value = {'seed':seed, 'variant':variant, 'stepD_protocol_sha256':c.sha(d.PROTOCOL),
             'implementation_sha256':c.sha(d.OUT/'implementation.json'), 'settings':p}
    d.publish(c.OUT/'protocol.json', d.json_bytes(value), resume)
    return p


def complete(folder):
    path = folder/'complete.json'
    if not path.exists():
        return False
    marker = c.read(path)
    c.require(marker['status']=='completed' and marker['protocol_sha256']==c.sha(folder/'protocol.json'), 'Invalid completion')
    for name, digest in marker['files_sha256'].items():
        c.require(c.sha(d.ROOT/name)==digest, f'Completed Step D file changed: {name}')
    return True


class GlobalFeatures:
    def __init__(self, records, guard, protocol):
        self.records, self.guard, self.protocol = records, guard, protocol
        self.model = None

    def get(self, name, ids, condition, matched):
        import torch
        ids = sorted(ids)
        self.guard.touch(ids, 'features/matched' if matched else 'features/original')
        path = c.ART/'features'/f'{name}_{condition}.npz'
        identity = c.sha(c.OUT/'protocol.json')
        if path.exists():
            with np.load(path, allow_pickle=False) as cache:
                c.require(cache['ids'].tolist()==ids and str(cache['protocol'])==identity and bool(cache['matched'])==matched, 'Feature cache metadata mismatch')
                x = cache['x'].copy()
                c.require(str(cache['array_sha256'])==hashlib.sha256(x.tobytes()).hexdigest(), 'Feature cache corrupt')
                c.require(x.shape==(len(ids),512) and np.isfinite(x).all(), 'Invalid CLIP features')
                return x
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if self.model is None:
            os.environ.setdefault('HF_HOME', str(c.ART/'hf'))
            from transformers import CLIPVisionModelWithProjection
            weights = d.ROOT/f'artifacts/round7/baselines_seed{d.seeds.number(c.SEED)}/weights/clip'
            c.require(c.sha(weights/'pytorch_model.bin')==self.protocol['clip']['weights_sha256'], 'CLIP weights changed')
            snapshot = c.read(d.ROOT/'results/round7/stepD_preservation.json')['files_sha256']
            c.require(c.sha(weights/'config.json')==snapshot[(weights/'config.json').relative_to(d.ROOT).as_posix()], 'CLIP config changed')
            torch.manual_seed(c.SEED); torch.set_num_threads(4)
            self.model = CLIPVisionModelWithProjection.from_pretrained(weights, local_files_only=True).to(device).eval()
            self.model.requires_grad_(False)
            if device.startswith('cuda'):
                self.model.half()
        features = []
        for start in range(0,len(ids),32):
            chosen = ids[start:start+32]
            pictures = [d.view(i, condition, matched, self.records, self.guard, self.protocol) for i in chosen]
            features.append(clip.encode(self.model, pictures, device))
        x = np.concatenate(features)
        stream = io.BytesIO()
        np.savez_compressed(stream, ids=np.array(ids), protocol=np.array(identity), matched=np.array(matched),
                            x=x, array_sha256=np.array(hashlib.sha256(x.tobytes()).hexdigest()))
        d.publish(path,stream.getvalue())
        print(f'W1 {name}/{condition}: {len(ids)} images ready',flush=True)
        return x


def raw_rows(ids, condition, logits, records):
    return [dict(source_id=i, generator=records[i]['generator'], label=records[i]['label'], condition=condition, logit=float(z)) for i,z in zip(sorted(ids),logits)]


def w1(records, parts, guard, protocol, resume):
    import joblib
    features = GlobalFeatures(records,guard,protocol)
    head_path = c.ART/'global_head.joblib'
    guard.touch(parts['expert_fit'],'fit_clip_expert')
    if head_path.exists():
        c.require(resume, 'Use --resume for existing head')
        with head_path.open('rb') as f:
            saved = joblib.load(f)
        c.require(saved['protocol_sha256']==c.sha(c.OUT/'protocol.json'), 'Head protocol mismatch')
        head = saved['head']
    else:
        x = features.get('expert_fit',parts['expert_fit'],'clean',True)
        y = np.array([records[i]['label'] for i in sorted(parts['expert_fit'])])
        head = clip.head(x,y)
        stream = io.BytesIO(); joblib.dump({'head':head,'protocol_sha256':c.sha(c.OUT/'protocol.json')},stream)
        d.publish(head_path,stream.getvalue())
    tuning = []
    guard.touch(parts['selection'],'tune_threshold')
    for cond in c.TUNE:
        x = features.get('selection_tuning',parts['selection'],cond,True)
        tuning.extend(raw_rows(parts['selection'],cond,head.decision_function(x),records))
    cut = c.threshold([r['label'] for r in tuning],expit([r['logit'] for r in tuning]))
    predictions = []
    for part in ('selection','development'):
        guard.touch(parts[part],'report/'+part)
        for cond in c.CONDITIONS:
            x = features.get(part+'_reporting',parts[part],cond,False)
            rows = raw_rows(parts[part],cond,head.decision_function(x),records)
            predictions += d.prediction_rows(rows,'W1',c.SEED,part,cut)
    return predictions,tuning,{'threshold':cut,'head':'unchanged Step C standardized logistic regression','tuning_view':'format_matched'}


def w2(records, parts, guard, protocol, resume):
    import torch
    # Copy known weights to the new cache; existing Step C files remain read-only.
    source = d.ROOT/f'artifacts/round7/baselines_seed{d.seeds.number(c.SEED)}/weights/resnet50-11ad3fa6.pth'
    snapshot = c.read(d.ROOT/'results/round7/stepD_preservation.json')['files_sha256']
    c.require(c.sha(source)==snapshot[source.relative_to(d.ROOT).as_posix()], 'ImageNet weights changed')
    destination = c.ART/'weights'/source.name
    if not destination.exists():
        d.publish(destination,source.read_bytes())
    c.require(c.sha(destination)==c.sha(source), 'Step D weights changed')
    spec = dict(protocol['resnet'],seed=c.SEED)
    model,state = rn.fit('original',records,parts,guard,spec,resume=resume,batch_fn=d.tensor_batch(True,protocol))
    device = str(next(model.parameters()).device)
    guard.touch(parts['selection'],'tune_threshold')
    tuning = rn.predict(model,sorted(parts['selection']),c.TUNE,state['batch_size'],'original',records,guard,device,batch_fn=d.tensor_batch(True,protocol))
    cut = c.threshold([r['label'] for r in tuning],expit([r['logit'] for r in tuning]))
    predictions = []
    for part in ('selection','development'):
        guard.touch(parts[part],'report/'+part)
        rows = rn.predict(model,sorted(parts[part]),c.CONDITIONS,state['batch_size'],'original',records,guard,device,batch_fn=d.tensor_batch(False,protocol))
        predictions += d.prediction_rows(rows,'W2',c.SEED,part,cut)
    settings = dict(threshold=cut, tuning_view='format_matched',best_epoch=state['best_epoch'],best_score=state['best_score'],
                    history=state['history'],oom_events=state['oom_events'])
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions,tuning,settings


def execute(seed,variant,resume):
    d.seeds.check_environment()
    records,parts,deny = d.configure(seed,variant)
    protocol = run_protocol(seed,variant,resume)
    if complete(c.OUT):
        c.require(resume,'Already completed; use --resume')
        print(f'Verified complete: seed {seed} / {variant}',flush=True)
        return
    guard = d.Guard(records,parts,deny)
    bundle_path = c.ART/'result_bundle.json'
    if bundle_path.exists():
        c.require(resume,'Use --resume to publish existing results')
        bundle = c.read(bundle_path)
        c.require(bundle['protocol_sha256']==c.sha(c.OUT/'protocol.json'),'Bundle protocol mismatch')
    else:
        started = time.perf_counter()
        status = 'failed'
        try:
            predictions,tuning,settings = (w1 if variant=='W1' else w2)(records,parts,guard,protocol,resume)
            quality = [dict(source_id=i,partition=part,label=records[i]['label'],quality=d.quality(seed,i,protocol['format_match']['quality_min'],protocol['format_match']['quality_max'])) for part in ('expert_fit','selection') for i in parts[part]]
            for r in tuning:
                r.update(variant=variant,seed=seed,partition='selection',view='format_matched')
            # Single atomic bundle freezes measured runtime and all final payloads before publication.
            bundle = dict(protocol_sha256=c.sha(c.OUT/'protocol.json'),predictions=predictions,tuning=tuning,
                          settings=settings,quality=quality,metrics=d.metrics(predictions),seconds_this_invocation=time.perf_counter()-started,
                          events={k:sorted(v) for k,v in guard.events.items()},violations=guard.violations)
            d.publish(bundle_path,d.json_bytes(bundle))
            status = 'completed'
        finally:
            guard.save('stepD_'+variant,status)
    for name,key in [('predictions.csv','predictions'),('tuning_predictions.csv','tuning'),('quality_draws.csv','quality'),('metrics.csv','metrics')]:
        d.publish(c.OUT/name,d.csv_bytes(bundle[key]),resume)
    d.publish(c.OUT/'settings.json',d.json_bytes(bundle['settings']),resume)
    d.publish(c.OUT/'audit.json',d.json_bytes({k:bundle[k] for k in ('protocol_sha256','seconds_this_invocation','events','violations')}),resume)
    paths = [p for folder in (c.OUT,c.ART) for p in folder.rglob('*') if p.is_file() and not p.name.endswith('.tmp') and p.name!='complete.json']
    marker = dict(status='completed',protocol_sha256=c.sha(c.OUT/'protocol.json'),files_sha256={p.relative_to(d.ROOT).as_posix():c.sha(p) for p in paths})
    d.publish(c.OUT/'complete.json',d.json_bytes(marker),resume)
    print(f'Completed seed {seed} / {variant}; run verifier next.',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init',action='store_true',help='Lock implementation only; no image access or training')
    parser.add_argument('--seed',type=int,choices=d.seeds.SEEDS)
    parser.add_argument('--variant',choices=['W1','W2'])
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if args.init:
        d.seeds.check_environment(); d.lock_implementation(create=True)
        print('Step D protocol and implementation locked; no training or image access.')
    else:
        parser.error('--seed and --variant required') if args.seed is None or args.variant is None else execute(args.seed,args.variant,args.resume)


if __name__=='__main__':
    main()
