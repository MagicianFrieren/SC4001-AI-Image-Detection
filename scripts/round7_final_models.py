"""Inference-only restoration of frozen Step C/D experts. No fit/selection calls."""
import sys
sys.dont_write_bytecode = True
import hashlib
import numpy as np
from scipy.special import expit
import round7_final_common as f
from round7_final_common import c
import round7_baseline_data as images
import round7_baseline_clip as clip
import gate_models as gates


def restore_gate(artifact):
    import torch
    from sklearn.preprocessing import StandardScaler
    torch.set_num_threads(2)  # Same CPU gate inference setting as the saved Step C runs.
    model=gates.Gate()
    model.load_state_dict({k:torch.tensor(v,dtype=torch.float32) for k,v in artifact['state'].items()})
    model.requires_grad_(False); model.eval()
    scaler=StandardScaler()
    scaler.mean_=np.array(artifact['mean']); scaler.scale_=np.array(artifact['scale'])
    scaler.var_=scaler.scale_**2; scaler.n_features_in_=len(scaler.mean_)
    return model,scaler


def clip_scores(models,features,heads,settings):
    """Pure inference, also tested against development caches before test release."""
    global_z=heads['global'].decision_function(features['g'])
    local_z=heads['local'].decision_function(features['l'].mean(axis=1))
    width=features['l'].shape[-1]
    variance=heads['local'].decision_function(features['l'].reshape(-1,width)).reshape(-1,4).var(axis=1)
    raw=dict(zg=global_z,zl=local_z,variance=variance,q=features['q'])
    transformed={}; output={}
    for model in models:
        cal=model['calibration']
        if cal not in transformed:
            transformed[cal]=clip.transform(raw,model['global_temperature'],model['local_temperature'])
        batch=transformed[cal]; name=model['model']
        if name.endswith('gate'):
            gate,scaler=restore_gate(settings['gates'][model['gate_key']]['artifact'])
            alpha=gates.predict(gate,scaler,batch['x'],name)
        else:
            a={'global':1.,'local':0.,'average':.5,'fixed':model['fixed_alpha']}[name]
            alpha=np.full(len(global_z),a)
        output[model['id']]=(alpha*batch['zg']+(1-alpha)*batch['zl'],alpha)
    return output


def source_path(identity):
    c.require(identity.startswith('train:') and identity[6:].isdigit(),'Unsafe source ID')
    return f.ART/'allowed_images'/(identity.replace(':','_')+'.bin')


def prepare(records,guard,freeze):
    import pyarrow as pa
    import pyarrow.parquet as pq
    remaining=set(records)
    for i in sorted(records):
        guard.touch([i],'original_cache')
        p=source_path(i)
        if p.exists():
            c.require(c.sha(p)==records[i]['source_bytes_sha256'],'Original cache corrupt')
            remaining.remove(i)
    inventory=c.read(f.checked(f.ROOT/'results/round7/source_inventory.json',freeze))
    for index,shard in enumerate(inventory['shards']):
        wanted={int(i.split(':')[1])%2000:i for i in remaining if int(i.split(':')[1])//2000==index}
        if not wanted:
            continue
        path=f.ROOT/shard['local_path']
        c.require(c.sha(path)==shard['lfs_sha256'],'Source shard hash changed')
        offset=0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32,columns=['image']):
            indices=[j for j in range(len(batch)) if offset+j in wanted]
            ids=[wanted[offset+j] for j in indices]
            if ids:
                guard.touch(ids,'reserved_payload_select')
                for identity,row in zip(ids,batch.take(pa.array(indices,type=pa.int64())).to_pylist()):
                    blob=row['image']['bytes']
                    c.require(hashlib.sha256(blob).hexdigest()==records[identity]['source_bytes_sha256'],'Reserved payload changed')
                    f.publish(source_path(identity),blob)
                    remaining.remove(identity)
            offset+=len(batch)
        print(f'Reserved originals: source shard {index+1}/14 ready',flush=True)
    c.require(not remaining,'Missing reserved originals')


def view(identity,profile,condition,records,guard):
    from PIL import Image,ImageOps
    guard.touch([identity],'reserved_decode')
    path=source_path(identity)
    c.require(c.sha(path)==records[identity]['source_bytes_sha256'],'Reserved original cache changed')
    with Image.open(path) as original:
        im=ImageOps.exif_transpose(original).convert('RGB')
    c.require(hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()==records[identity]['pixel_sha256'],'Pillow pixel hash mismatch')
    return images.view(im,profile,condition)


def features(records,guard,freeze):
    """Cache frozen encoder batches once, shared across all seeds and heads."""
    import torch
    model=None; ids=sorted(records); identity=c.sha(f.FREEZE)
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    for profile in c.PROFILES:
        for condition in c.CONDITIONS:
            for offset in range(0,len(ids),32):
                chosen=ids[offset:offset+32]; guard.touch(chosen,'clip_features')
                path=f.ART/'features'/f'{profile}_{condition}_{offset:05d}.npz'
                if path.exists():
                    f.npz_read(path,chosen,identity)
                    continue
                if model is None:
                    from transformers import CLIPVisionModelWithProjection
                    weights=f.ROOT/'artifacts/round7/baselines_seed1/weights/clip'
                    f.checked(weights/'pytorch_model.bin',freeze); f.checked(weights/'config.json',freeze)
                    torch.set_num_threads(4)
                    model=CLIPVisionModelWithProjection.from_pretrained(weights,local_files_only=True).to(device).eval()
                    model.requires_grad_(False)
                    if device.startswith('cuda'):
                        model.half()
                pictures=[view(i,profile,condition,records,guard) for i in chosen]
                g=clip.encode(model,pictures,device)
                l=np.stack([clip.encode(model,[images.shuffled(im,s) for im in pictures],device) for s in images.VIEW_SEEDS],axis=1)
                q=np.array([images.quality(im) for im in pictures])
                f.npz_publish(path,chosen,dict(g=g,l=l,q=q),identity)
                if offset%320==0:
                    print(f'Frozen CLIP {profile}/{condition}: {offset+len(chosen)}/{len(ids)}',flush=True)
            print(f'Frozen CLIP {profile}/{condition} complete',flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def read_features(ids,profile,condition,guard):
    arrays=[]
    for offset in range(0,len(ids),32):
        chosen=ids[offset:offset+32]; guard.touch(chosen,'clip_cache_read')
        arrays.append(f.npz_read(f.ART/'features'/f'{profile}_{condition}_{offset:05d}.npz',chosen,c.sha(f.FREEZE)))
    return {key:np.concatenate([a[key] for a in arrays]) for key in ('g','l','q')}


def restore_resnet(declaration,freeze,device):
    import torch
    from torchvision.models import resnet50
    path=f.checked(f.ROOT/declaration['checkpoint'],freeze)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    source_protocol=(f.ROOT/declaration['settings_path']).parent/'protocol.json'
    c.require(saved['protocol_sha256']==c.sha(f.checked(source_protocol,freeze)),'Checkpoint protocol mismatch')
    c.require(saved['profile']==declaration['profile'],'Wrong checkpoint profile')
    model=resnet50(weights=None); model.fc=torch.nn.Linear(model.fc.in_features,1)
    model.load_state_dict(saved['model']); model.requires_grad_(False); model.eval()
    del saved
    return model.to(device)


def resnet_logits(task,models,condition,records,guard,freeze):
    import torch
    import round7_baseline_resnet as rn
    # Deterministic settings only; no optimizer, training, calibration or selection.
    rn.seed_everything(models[0]['seed'])
    ids=sorted(records); first=models[0]; batch=first['batch_size']
    device='cuda:0' if torch.cuda.is_available() else 'cpu'; model=None; arrays=[]
    identity=c.sha(f.FREEZE)+'|'+task
    for offset in range(0,len(ids),batch):
        chosen=ids[offset:offset+batch]; guard.touch(chosen,'resnet_inference')
        path=f.ART/'logits'/task/f'{offset:05d}.npz'
        if path.exists():
            z=f.npz_read(path,chosen,identity)['z']
        else:
            if model is None:
                model=restore_resnet(first,freeze,device)
            x=np.stack([images.normalized(view(i,first['profile'],condition,records,guard),'resnet') for i in chosen])
            with torch.inference_mode():
                z=model(torch.from_numpy(x).to(device)).flatten().cpu().numpy()
            c.require(np.isfinite(z).all(),'Nonfinite inference')
            f.npz_publish(path,chosen,dict(z=z),identity)
        arrays.append(z)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    z=np.concatenate(arrays)
    return {r['id']:(z/r['temperature'],None) for r in models}


def predict_task(task,models,condition,records,guard,freeze):
    import joblib
    ids=sorted(records); first=models[0]
    guard.touch(ids,'predict/'+task)
    if first['family']=='resnet':
        scores=resnet_logits(task,models,condition,records,guard,freeze)
    else:
        features_=read_features(ids,first['profile'],condition,guard)
        saved=joblib.load(f.checked(f.ROOT/first['head'],freeze))
        if first['stage']=='D':
            protocol=(f.ROOT/first['settings_path']).parent/'protocol.json'
            c.require(saved['protocol_sha256']==c.sha(f.checked(protocol,freeze)),'W1 head protocol mismatch')
            scores={first['id']:(saved['head'].decision_function(features_['g']),None)}
        else:
            settings=c.read(f.checked(f.ROOT/first['settings_path'],freeze))
            scores=clip_scores(models,features_,saved,settings)
    rows=[]
    for m in models:
        z,alpha=scores[m['id']]
        c.require(len(z)==len(ids) and np.isfinite(z).all(),'Invalid model output')
        for j,i in enumerate(ids):
            rows.append(dict(model_id=m['id'],source_id=i,generator=records[i]['generator'],label=records[i]['label'],condition=condition,
                             logit=float(z[j]),probability_fake=float(expit(z[j])),threshold=m['threshold'],alpha_global='' if alpha is None else float(alpha[j])))
    return rows
