"""Frozen CLIP linear probe, augmentation control, and shortcut diagnostics."""
import argparse
import csv
import hashlib
import io
import json
import os
import platform
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
import numpy as np
import requests
import sklearn
import torch
from PIL import Image, ImageOps
from scipy.ndimage import laplace
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import CLIPVisionModelWithProjection

MODEL = 'openai/clip-vit-base-patch32'
MODEL_REVISION = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
MODEL_SHA256 = 'a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'
SEED = 20261018
CONDITIONS = ['clean', 'jpeg70', 'jpeg50', 'resize05']
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def download_model():
    folder = ROOT / '.cache' / 'clip-vit-base-patch32'
    folder.mkdir(parents=True, exist_ok=True)
    for name in ['config.json', 'pytorch_model.bin']:
        destination = folder / name
        if destination.exists():
            continue
        url = f'https://huggingface.co/{MODEL}/resolve/{MODEL_REVISION}/{name}'
        print(f'Downloading model {name}', flush=True)
        with requests.get(url, stream=True, timeout=(30, 90)) as response:
            response.raise_for_status()
            with destination.with_suffix(destination.suffix + '.part').open('wb') as handle:
                for chunk in response.iter_content(1024*1024):
                    handle.write(chunk)
        destination.with_suffix(destination.suffix + '.part').replace(destination)
    with (folder/'pytorch_model.bin').open('rb') as handle:
        assert hashlib.file_digest(handle,'sha256').hexdigest()==MODEL_SHA256, 'Model checksum mismatch'
    return folder


def base_image(path):
    with Image.open(path) as image:
        return ImageOps.fit(image.convert('RGB'), (224, 224), method=Image.Resampling.BICUBIC)


def degrade(image, condition):
    if condition.startswith('jpeg'):
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=int(condition[4:]), subsampling=2)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')
    if condition == 'resize05':
        return image.resize((112, 112), Image.Resampling.BICUBIC).resize((224, 224), Image.Resampling.BICUBIC)
    return image


def quality(image):
    gray = np.asarray(image.convert('L'), dtype=np.float32) / 255
    energy = np.abs(np.fft.fftshift(np.fft.fft2(gray-gray.mean())))**2
    yy, xx = np.ogrid[-112:112, -112:112]
    high = energy[(xx*xx + yy*yy) > 56**2].sum() / (energy.sum()+1e-9)
    dx, dy = np.abs(np.diff(gray, axis=1)), np.abs(np.diff(gray, axis=0))
    block = (dx[:, 7::8].mean()+dy[7::8].mean()) / (dx.mean()+dy.mean()+1e-9)
    return [np.log1p(1e4*laplace(gray).var()), high, block]


def extract(records, batch_size):
    identity = hashlib.sha256((json.dumps(records, sort_keys=True) + MODEL_REVISION +
                              '|RGB-fit224-bicubic|clean,jpeg70,jpeg50,resize05|v1').encode()).hexdigest()
    path = ROOT / 'artifacts' / 'features.npz'
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        with np.load(path) as cache:
            if str(cache['identity']) == identity:
                print('Using validated feature cache', flush=True)
                return {key: cache[key] for key in cache.files if key != 'identity'}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    model = CLIPVisionModelWithProjection.from_pretrained(download_model()).to(device).eval()
    model.requires_grad_(False)
    if device == 'cuda':
        model.half()
        torch.cuda.reset_peak_memory_stats()
    features, timing = {}, {}
    for condition in CONDITIONS:
        start = time.perf_counter()
        blocks, qualities = [], []
        for offset in range(0, len(records), batch_size):
            images = [degrade(base_image(ROOT/r['path']), condition) for r in records[offset:offset+batch_size]]
            qualities.extend(quality(image) for image in images)
            array = np.stack([(np.asarray(image, dtype=np.float32)/255-MEAN)/STD for image in images])
            pixels = torch.from_numpy(array.transpose(0, 3, 1, 2)).to(device=device, dtype=model.dtype)
            with torch.inference_mode():
                embedding = model(pixel_values=pixels).image_embeds.float()
                embedding = torch.nn.functional.normalize(embedding, dim=-1)
            blocks.append(embedding.cpu().numpy())
            if offset % (batch_size*20) == 0:
                print(f'{condition}: {min(offset+batch_size,len(records))}/{len(records)}', flush=True)
        features[condition] = np.concatenate(blocks)
        features['quality_'+condition] = np.asarray(qualities)
        timing[condition] = time.perf_counter()-start
        print(f'{condition}: completed in {timing[condition]:.1f}s', flush=True)
    features['extraction_seconds'] = np.array(sum(timing.values()))
    features['peak_cuda_memory_mb'] = np.array(torch.cuda.max_memory_allocated()/1024**2 if device=='cuda' else 0)
    np.savez_compressed(path, identity=np.array(identity), **features)
    return features


def train_head(x, y):
    head = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver='liblinear',
                         max_iter=2000, random_state=SEED))
    head.fit(x, y)
    return head


def choose_threshold(y, p):
    fpr, tpr, thresholds = roc_curve(y, p)
    valid = np.isfinite(thresholds)
    thresholds, criterion = thresholds[valid], (tpr-fpr)[valid]
    best = np.flatnonzero(np.isclose(criterion, criterion.max()))
    return float(thresholds[best[np.argmin(np.abs(thresholds[best]-0.5))]])


def auc_interval(y, scores, seed):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == label) for label in [0, 1]]
    aucs = []
    for _ in range(500):
        ids = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        aucs.append(roc_auc_score(y[ids], scores[ids]))
    return [float(v) for v in np.percentile(aucs, [2.5, 97.5])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    start = time.perf_counter()
    manifest = ROOT / 'results' / 'manifest.jsonl'
    records = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    assert len({r['pixel_sha256'] for r in records}) == len(records)
    assert len({r['crop_sha256'] for r in records}) == len(records)
    for r in records:
        assert r['label'] == (r['generator'] != 'Real')
        assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest() == r['file_sha256']
        assert r['split']=='development' or r['generator'] in ('Real', 'SD15')
    # Every image must be separated even under the conservative near-duplicate rule.
    for i, row in enumerate(records):
        assert all((int(row['dhash'],16)^int(old['dhash'],16)).bit_count()>3 for old in records[:i])
    features = extract(records, args.batch_size)
    y = np.array([r['label'] for r in records])
    train = np.array([r['split']=='train' for r in records])
    val = np.array([r['split']=='validation' for r in records])
    development = np.array([r['split']=='development' for r in records])
    sizes = np.log1p(np.array([[r['width'], r['height']] for r in records]))
    specifications = {
        'clip_clean': (features['clean'][train], y[train], 'clip'),
        'clip_augmented': (np.concatenate([features[c][train] for c in ['clean','jpeg70','resize05']]), np.tile(y[train],3), 'clip'),
        'quality_only': (features['quality_clean'][train], y[train], 'quality'),
        'original_size_only': (sizes[train], y[train], 'size'),
    }
    metrics, predictions, thresholds, saved_heads = [], [], {}, {}
    for name, (xtrain, ytrain, kind) in specifications.items():
        head = train_head(xtrain, ytrain)
        clean = features['clean'] if kind=='clip' else features['quality_clean'] if kind=='quality' else sizes
        threshold = choose_threshold(y[val], head.predict_proba(clean[val])[:, 1])
        thresholds[name] = threshold
        saved_heads[name] = head
        for condition in ['clean','jpeg50','resize05']:
            x = features[condition] if kind=='clip' else features['quality_'+condition] if kind=='quality' else sizes
            probability = head.predict_proba(x)[:, 1]
            for generator in ['SD15','BigGAN','ADM']:
                mask = development & np.array([r['generator'] in ('Real',generator) for r in records])
                yy, pp = y[mask], probability[mask]
                lower, upper = auc_interval(yy, pp, SEED)
                row = {'model':name, 'generator':generator, 'condition':condition, 'n':int(mask.sum()),
                       'auroc':float(roc_auc_score(yy, pp)), 'auc_ci_low':lower,'auc_ci_high':upper,
                       'average_precision':float(average_precision_score(yy,pp)),
                       'balanced_accuracy':float(balanced_accuracy_score(yy,pp>=threshold)),
                       'real_false_positive_rate':float(np.mean(pp[yy==0]>=threshold)), 'threshold':threshold}
                metrics.append(row)
                print(f"{name:19} {generator:6} {condition:8} AUC={row['auroc']:.4f} BA={row['balanced_accuracy']:.4f}", flush=True)
            for index in np.flatnonzero(development):
                predictions.append({'model':name,'condition':condition,'source_id':records[index]['source_id'],
                                    'generator':records[index]['generator'],'label':int(y[index]),'probability_fake':float(probability[index])})
    folder = ROOT/'results'
    for filename, rows in [('metrics.csv',metrics), ('predictions.csv',predictions)]:
        with (folder/filename).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    import joblib
    joblib.dump(saved_heads, ROOT/'artifacts'/'linear_heads.joblib')
    metadata = {'seed':SEED, 'model':MODEL, 'model_revision':MODEL_REVISION,
                'model_weight_sha256':MODEL_SHA256,'feature_dimension':512,
                'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
                'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'torch':torch.__version__, 'sklearn':sklearn.__version__, 'python':platform.python_version(),
                'thresholds':thresholds,'batch_size':args.batch_size,
                'feature_extraction_seconds':float(features['extraction_seconds']),
                'peak_cuda_memory_mb':float(features['peak_cuda_memory_mb']),
                'run_seconds':time.perf_counter()-start,
                'training':'300 real + 300 SD15; augmented: same originals, clean/JPEG70/resize05; C=1 fixed',
                'validation':'100 real + 100 SD15 clean; thresholds chosen once, then fixed',
                'development':'100 shared real + 100 fake per generator. Exploratory results, not blind final tests.',
                'ci':'500 class-stratified bootstrap resamples per cell; shared real pool means cells are correlated',
                'limits':['Single training seed and small third-party subset', 'Generator labels inherited',
                          'dHash removes some near-duplicates but does not prove semantic independence',
                          'CLIP pretraining overlap unknown', 'Source and generation artefacts can be confounded',
                          'ViT-B/32 pilot, not reproduction of UnivFD ViT-L/14']}
    (folder/'run_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__':
    main()
