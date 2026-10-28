"""Inference only, portable linear heads, unchanged Round 7 image/CLIP/gate math."""
import sys
sys.dont_write_bytecode = True
import hashlib
import io
import json
import warnings
import numpy as np
from scipy.special import expit
import round8_final_common as f
import round7_baseline_data as images
import round7_baseline_clip as clip
from round7_final_models import restore_gate
import gate_models as gates


class StateOnly:
    """Fallback for trusted sklearn pickles: restore state, never invoke estimator methods."""
    def __init__(self, *args, **kwargs):
        pass
    def __setstate__(self, state):
        self.__dict__.update(state)


def load_parameters(path):
    import joblib
    messages = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            value = joblib.load(path)
        messages = [str(w.message) for w in caught]
        mode = 'normal_load'
    except Exception as error:
        # joblib handles its own compression and NumPy wrappers. Only sklearn classes
        # become state bags; no estimator __setstate__/predict or version-dependent math.
        from unittest.mock import patch
        import joblib.numpy_pickle as pickle_module
        class ParameterUnpickler(pickle_module.NumpyUnpickler):
            def find_class(self, module, name):
                if module.startswith('sklearn.'):
                    return StateOnly
                return super().find_class(module, name)
        with patch.object(pickle_module, 'NumpyUnpickler', ParameterUnpickler):
            value = joblib.load(path)
        mode = 'state_only_fallback'
        messages = [repr(error)]
    return value, {'mode': mode, 'warnings': messages}


def linear_parameters(head):
    steps = head.steps
    f.require(len(steps) == 2, 'Expected scaler plus binary logistic head')
    scaler, model = steps[0][1], steps[1][1]
    f.require(list(model.classes_) == [0, 1] and model.coef_.shape == (1, 512), 'Unexpected classifier orientation/shape')
    return dict(mean=np.asarray(scaler.mean_).tolist(), scale=np.asarray(scaler.scale_).tolist(),
                with_mean=scaler.with_mean, with_std=scaler.with_std,
                coef=np.asarray(model.coef_).tolist(), intercept=np.asarray(model.intercept_).tolist(), classes=[0, 1])


def decision(spec, features):
    # Match the pinned sklearn 1.8 StandardScaler: cast parameters to the
    # input dtype BEFORE each in-place operation. sklearn 1.5 instead used
    # float64 parameters, which rounds differently for float32 CLIP features.
    x = np.array(features, copy=True)
    f.require(x.dtype in (np.float32, np.float64), 'Unexpected feature dtype')
    if spec['with_mean']:
        x -= np.asarray(spec['mean'], dtype=x.dtype)
    if spec['with_std']:
        x /= np.asarray(spec['scale'], dtype=x.dtype)
    return (x @ np.asarray(spec['coef']).T + np.asarray(spec['intercept'])).ravel()


def score(models, features, portable):
    zg = decision(portable['global'], features['g'])
    if 'local' not in portable:
        f.require(len(models) == 1 and models[0]['model'] == 'W1', 'Single head is only W1')
        return {models[0]['id']: (zg, None)}
    local = features['l']
    zl = decision(portable['local'], local.mean(axis=1))
    variance = decision(portable['local'], local.reshape(-1, 512)).reshape(-1, 4).var(axis=1)
    batch = clip.transform(dict(zg=zg, zl=zl, variance=variance, q=features['q']), 1., 1.)
    result = {}
    for m in models:
        if m['model'].endswith('gate'):
            gate, scaler = restore_gate(portable['gates'][m['model'] + '/' + str(m['initialization'])])
            alpha = gates.predict(gate, scaler, batch['x'], m['model'])
        else:
            alpha = np.full(len(zg), {'global': 1., 'local': 0., 'average': .5, 'fixed': m['fixed_alpha']}[m['model']])
        result[m['id']] = (alpha * zg + (1 - alpha) * zl, alpha)
    return result


def original_path(scope, identity):
    f.require(scope in ('development', 'test') and identity.startswith('train:') and identity[6:].isdigit(), 'Unsafe source path')
    return f.ART / scope / 'images' / (identity.replace(':', '_') + '.bin')


def prepare_originals(scope, records, guard):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import round8_a_common as a
    pending = set(records)
    for identity, r in records.items():
        guard.touch([identity], 'source_cache')
        target = original_path(scope, identity)
        if target.exists():
            f.require(f.sha(target) == r['source_bytes_sha256'], 'Corrupt source cache')
            pending.remove(identity)
        elif scope == 'development':
            existing = a.find_original(identity)
            if existing:
                f.require(f.sha(existing) == r['source_bytes_sha256'], 'Historical source bytes changed')
                f.publish(target, existing.read_bytes()); pending.remove(identity)
    inventory = f.read(f.ROOT / 'results/round7/source_inventory.json')
    for index, shard in enumerate(inventory['shards']):
        wanted = {int(i[6:]) % 2000: i for i in pending if int(i[6:]) // 2000 == index}
        if not wanted:
            continue
        path = a.shard_path(shard)
        f.require(path is not None and f.sha(path) == shard['lfs_sha256'], f'Missing/corrupt pinned shard {shard["path"]}; no automatic download')
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=['image']):
            chosen = [j for j in range(len(batch)) if offset + j in wanted]
            if chosen:
                ids = [wanted[offset + j] for j in chosen]
                guard.touch(ids, 'source_select')
                for identity, row in zip(ids, batch.take(pa.array(chosen, type=pa.int64())).to_pylist()):
                    blob = row['image']['bytes']
                    f.require(hashlib.sha256(blob).hexdigest() == records[identity]['source_bytes_sha256'], 'Source bytes changed')
                    f.publish(original_path(scope, identity), blob); pending.remove(identity)
            offset += len(batch)
        print(f'{scope}: restored shard {index + 1}/14', flush=True)
    f.require(not pending, 'Missing originals')


def view(scope, identity, condition, records, guard):
    from PIL import Image, ImageOps
    guard.touch([identity], 'decode/' + scope)
    path = original_path(scope, identity)
    f.require(f.sha(path) == records[identity]['source_bytes_sha256'], 'Image bytes changed')
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    f.require(hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest() == records[identity]['pixel_sha256'], 'Pillow pixel hash changed')
    return images.view(image, 'original', condition)


def cache(path, ids, identity, arrays=None):
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            f.require(data['ids'].tolist() == ids and str(data['identity']) == identity, 'Cache identity changed')
            checks = json.loads(str(data['checks']))
            result = {k: data[k].copy() for k in checks}
        for k, array in result.items():
            f.require(np.isfinite(array).all() and hashlib.sha256(array.tobytes()).hexdigest() == checks[k], 'Corrupt array')
        return result
    f.require(arrays is not None, f'Missing cache: {path}')
    f.require(all(np.isfinite(a).all() for a in arrays.values()), 'Nonfinite inference')
    stream = io.BytesIO()
    checks = {k: hashlib.sha256(a.tobytes()).hexdigest() for k, a in arrays.items()}
    np.savez_compressed(stream, ids=np.array(ids), identity=np.array(identity), checks=np.array(json.dumps(checks, sort_keys=True)), **arrays)
    f.publish(path, stream.getvalue())
    return arrays


def features(scope, records, guard, identity, weights):
    import torch
    from transformers import CLIPVisionModelWithProjection
    ids = sorted(records)
    model = None
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    result = {}
    for condition in f.CONDITIONS:
        pieces = []
        for start in range(0, len(ids), 32):
            chosen = ids[start:start + 32]
            guard.touch(chosen, 'clip/' + scope)
            path = f.ART / scope / 'features' / f'{condition}_{start:05d}.npz'
            if path.exists():
                piece = cache(path, chosen, identity)
            else:
                if model is None:
                    model = CLIPVisionModelWithProjection.from_pretrained(weights, local_files_only=True).to(device).eval()
                    model.requires_grad_(False)
                    if device.startswith('cuda'):
                        model.half()
                pictures = [view(scope, i, condition, records, guard) for i in chosen]
                piece = dict(g=clip.encode(model, pictures, device),
                    l=np.stack([clip.encode(model, [images.shuffled(im, s) for im in pictures], device) for s in images.VIEW_SEEDS], axis=1),
                    q=np.asarray([images.quality(im) for im in pictures]))
                cache(path, chosen, identity, piece)
            f.require(piece['g'].shape == (len(chosen), 512) and piece['l'].shape == (len(chosen), 4, 512) and piece['q'].shape == (len(chosen), 3), 'Feature shape mismatch')
            pieces.append(piece)
            if start % 320 == 0:
                print(f'{scope} CLIP {condition}: {start + len(chosen)}/{len(ids)}', flush=True)
        result[condition] = {k: np.concatenate([p[k] for p in pieces]) for k in ('g', 'l', 'q')}
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def resnet(model_spec, condition, records, guard, identity):
    import torch
    from torchvision.models import resnet50
    import round7_baseline_resnet as rn
    rn.seed_everything(model_spec['seed'])
    ids = sorted(records); batch = model_spec['batch_size']; model = None; pieces = []
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    for start in range(0, len(ids), batch):
        chosen = ids[start:start + batch]
        guard.touch(chosen, 'resnet/test')
        path = f.ART / 'test/logits' / model_spec['id'] / condition / f'{start:05d}.npz'
        if path.exists():
            z = cache(path, chosen, identity)['z']
        else:
            if model is None:
                saved = torch.load(f.ROOT / model_spec['checkpoint'], map_location='cpu', weights_only=False)
                protocol = (f.ROOT / model_spec['settings_path']).parent / 'protocol.json'
                f.require(saved['profile'] == 'original' and saved['protocol_sha256'] == f.sha(protocol), 'Checkpoint provenance mismatch')
                model = resnet50(weights=None); model.fc = torch.nn.Linear(model.fc.in_features, 1)
                model.load_state_dict(saved['model']); model.requires_grad_(False); model.eval(); model.to(device)
                del saved
            x = np.stack([images.normalized(view('test', i, condition, records, guard), 'resnet') for i in chosen])
            with torch.inference_mode():
                z = model(torch.from_numpy(x).to(device)).flatten().cpu().numpy()
            cache(path, chosen, identity, dict(z=z))
        pieces.append(z)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Match Step C's Python-float probability reporting, retaining raw FP32 network output.
    return np.concatenate(pieces).astype(np.float64) / model_spec['temperature']
