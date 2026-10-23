"""Four deterministic PatchShuffle views for the round-three local branch."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from transformers import CLIPVisionModelWithProjection

import prepare_data as data
import run_pilot as pilot
import run_round2 as round2

ROOT = pilot.ROOT
VIEW_SEEDS = [202610220, 202610221, 202610222, 202610223]
PATCH_SIZE = 56


def permutation(seed):
    return np.random.default_rng(seed).permutation(16)


def shuffle_image(image, order):
    """Rearrange 16 intact RGB tiles without interpolating or dropping pixels."""
    array = np.asarray(image)
    if array.shape != (224, 224, 3) or sorted(order) != list(range(16)):
        raise ValueError('Expected RGB 224x224 and a permutation of 16 tiles')
    tiles = array.reshape(4, PATCH_SIZE, 4, PATCH_SIZE, 3).transpose(0, 2, 1, 3, 4).reshape(16, PATCH_SIZE, PATCH_SIZE, 3)
    shuffled = tiles[order].reshape(4, 4, PATCH_SIZE, PATCH_SIZE, 3).transpose(0, 2, 1, 3, 4).reshape(224, 224, 3)
    return Image.fromarray(shuffled)


def extract(records, profiles, conditions, batch_size=32):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    identity = hashlib.sha256(json.dumps({
        'records': records, 'code': data.sha256(Path(__file__)),
        'preprocessing_code': data.sha256(ROOT / 'scripts/run_round2.py'),
        'pilot_code': data.sha256(ROOT / 'scripts/run_pilot.py'),
        'model': pilot.MODEL_REVISION, 'torch': torch.__version__,
        'numpy': np.__version__, 'batch_size': batch_size, 'device': device,
        'view_seeds': VIEW_SEEDS, 'patch_size': PATCH_SIZE}, sort_keys=True).encode()).hexdigest()
    model = None
    features, provenance = {}, {}
    boundary = len(round2.read_manifest(ROOT / 'results/manifest.jsonl'))
    batches = [(i, min(i + batch_size, end)) for begin, end in [(0, boundary), (boundary, len(records))]
               for i in range(begin, end, batch_size)]
    for profile in profiles:
        for condition in conditions:
            key = f'{profile}/{condition}'
            path = ROOT / 'artifacts' / f'round3_local_{profile}_{condition}.npz'
            if path.exists():
                with np.load(path) as cache:
                    if str(cache['identity']) == identity:
                        features[key] = cache['views']
                        provenance[key] = {'cache_reused': True, 'extraction_seconds': float(cache['seconds'])}
                        print(f'Validated cached local views: {key}', flush=True)
                        continue
            if model is None:
                model_dir = ROOT / '.cache/clip-vit-base-patch32'
                assert data.sha256(model_dir / 'pytorch_model.bin') == pilot.MODEL_SHA256
                torch.manual_seed(round2.SEED)
                torch.set_num_threads(4)
                model = CLIPVisionModelWithProjection.from_pretrained(model_dir, local_files_only=True).to(device).eval()
                model.requires_grad_(False)
                if device == 'cuda':
                    model.half()
                    torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            blocks = []
            for i, stop in batches:
                originals = [round2.image_view(ROOT / r['path'], profile, condition) for r in records[i:stop]]
                per_view = []
                for seed in VIEW_SEEDS:
                    order = permutation(seed)
                    images = [shuffle_image(im, order) for im in originals]
                    array = np.stack([(np.asarray(im, dtype=np.float32) / 255 - pilot.MEAN) / pilot.STD for im in images])
                    pixels = torch.from_numpy(array.transpose(0, 3, 1, 2)).to(device=device, dtype=model.dtype)
                    with torch.inference_mode():
                        embedding = model(pixel_values=pixels).image_embeds.float()
                        embedding = torch.nn.functional.normalize(embedding, dim=-1)
                    per_view.append(embedding.cpu().numpy())
                blocks.append(np.stack(per_view, axis=1))
                if i % (batch_size * 20) == 0:
                    print(f'Local {key}: {stop}/{len(records)} images x 4 views', flush=True)
            views = np.concatenate(blocks)
            seconds = time.perf_counter() - start
            assert views.shape == (len(records), 4, 512) and np.isfinite(views).all()
            np.savez_compressed(path, identity=np.array(identity), views=views, seconds=np.array(seconds))
            features[key] = views
            provenance[key] = {'cache_reused': False, 'extraction_seconds': seconds}
            print(f'Local {key} complete: {seconds:.1f}s', flush=True)
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2 if device == 'cuda' and model is not None else None
    return features, provenance, peak
