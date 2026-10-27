"""Run A's development study; --preflight and --prepare are safe entry points."""
import sys
sys.dont_write_bytecode = True
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import io
import json
import time

import numpy as np
from scipy.special import expit
import round8_a_common as a
from round8_a_common import c, images, seeds
import round7_baseline_clip as clip
import gate_models as gates


class Features:
    """Validated, atomic per-batch caches permit resuming interrupted extraction."""
    def __init__(self, seed, records, parts, guard, protocol, signature):
        self.seed, self.records, self.parts, self.guard = seed, records, parts, guard
        self.protocol, self.signature = protocol, signature
        self.model = None
        self.hashes = {}

    def get(self, part, condition, matched):
        import torch
        ids = self.parts[part]
        self.guard.touch(ids, 'features/' + part)
        if matched:
            self.guard.touch(ids, 'format_match')
        folder = a.ART / f'seed{seeds.number(self.seed)}/features' / f'{part}_{condition}_{int(matched)}'
        pieces = []
        for start in range(0, len(ids), 32):
            chosen = ids[start:start + 32]
            path = folder / f'{start:05d}.npz'
            identity = json.dumps([self.seed, part, condition, matched, chosen])
            if path.exists():
                with np.load(path, allow_pickle=False) as cache:
                    a.require(str(cache['identity']) == identity and cache['ids'].tolist() == chosen
                              and str(cache['signature']) == self.signature,
                              'Feature cache identity/protocol changed')
                    piece = {k: cache[k].copy() for k in ('g', 'l', 'q')}
                    for k, array in piece.items():
                        a.require(str(cache[k + '_sha256']) == hashlib.sha256(array.tobytes()).hexdigest(), 'Corrupt feature cache')
            else:
                device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                if self.model is None:
                    from transformers import CLIPVisionModelWithProjection
                    torch.manual_seed(self.seed)
                    torch.set_num_threads(4)
                    self.model = CLIPVisionModelWithProjection.from_pretrained(a.weights_path(), local_files_only=True).to(device).eval()
                    self.model.requires_grad_(False)
                    if device.startswith('cuda'):
                        self.model.half()
                pictures = [a.view(i, condition, matched, self.seed, self.records, self.guard,
                                   self.protocol['format_match']) for i in chosen]
                piece = dict(g=clip.encode(self.model, pictures, device),
                             l=np.stack([clip.encode(self.model, [images.shuffled(im, s) for im in pictures], device)
                                         for s in images.VIEW_SEEDS], axis=1),
                             q=np.asarray([images.quality(im) for im in pictures]))
                stream = io.BytesIO()
                np.savez_compressed(stream, ids=np.array(chosen), signature=np.array(self.signature),
                                    identity=np.array(identity), **piece,
                                    **{k + '_sha256': np.array(hashlib.sha256(v.tobytes()).hexdigest()) for k, v in piece.items()})
                a.publish(path, stream.getvalue())
            a.require(piece['g'].shape == (len(chosen), 512) and piece['l'].shape == (len(chosen), 4, 512)
                      and piece['q'].shape == (len(chosen), 3), 'Invalid feature shapes')
            a.require(all(np.isfinite(v).all() for v in piece.values()), 'Non-finite features')
            self.hashes[path.relative_to(a.ROOT).as_posix()] = c.sha(path)
            pieces.append(piece)
            if start % 320 == 0 or start + 32 >= len(ids):
                print(f'seed {self.seed} {part}/{condition} matched={matched}: {min(start + 32, len(ids))}/{len(ids)}', flush=True)
        return {k: np.concatenate([p[k] for p in pieces]) for k in ('g', 'l', 'q')}


def fit_arm(seed, arm, records, parts, guard, feature_get):
    """Pure experiment logic with injectable features for synthetic integration tests."""
    a.require(arm in a.ARMS, 'Unknown arm')
    matched = arm == 'matched'
    guard.touch(parts['expert_fit'], 'fit_experts')
    train = feature_get('expert_fit', 'clean', matched)
    y = np.array([records[i]['label'] for i in parts['expert_fit']])
    experts = dict(global_head=clip.head(train['g'], y), local_head=clip.head(train['l'].mean(axis=1), y))
    raw = {}
    for name, part, conditions, use_match in (
        ('gate_fit', 'gate_fit', c.TUNE, matched),
        ('tuning', 'selection', c.TUNE, matched),
        ('selection', 'selection', c.CONDITIONS, False),
        ('development', 'development', c.CONDITIONS, False),
    ):
        features = {(part, cond): feature_get(part, cond, use_match) for cond in conditions}
        raw[name] = clip.bundle(part, conditions, features, records, parts,
                                experts['global_head'], experts['local_head'], guard)
    batches = {name: clip.transform(batch, 1., 1.) for name, batch in raw.items()}
    tuning = batches['tuning']
    guard.touch(parts['selection'], 'select')
    fixed, search = clip.fixed_alpha(tuning)
    fitted = [(name, -1, {part: np.full(len(batch['y']), weight) for part, batch in batches.items()})
              for name, weight in (('global', 1.), ('local', 0.), ('average', .5), ('fixed', fixed))]
    gate_artifacts = {}
    for gate_seed in gates.SEEDS:
        for name in ('score_gate', 'quality_gate'):
            guard.touch(parts['gate_fit'], 'fit_gate')
            model, scaler, artifact, history = gates.fit(batches['gate_fit'], tuning, name, gate_seed)
            alpha = {part: gates.predict(model, scaler, batch['x'], name) for part, batch in batches.items()}
            gate_artifacts[f'{name}/{gate_seed}'] = dict(artifact=artifact, history=history)
            fitted.append((name, gate_seed, alpha))
    predictions, settings = [], {}
    for name, gate_seed, alpha in fitted:
        tz = alpha['tuning'] * tuning['zg'] + (1 - alpha['tuning']) * tuning['zl']
        cut = c.threshold(tuning['y'], expit(tz))
        settings[f'{name}/{gate_seed}'] = dict(threshold=cut, fixed_alpha=fixed,
                                              tuning_alpha=alpha['tuning'].tolist())
        for part in ('selection', 'development'):
            guard.touch(parts[part], 'report_' + part)
            batch, weight = batches[part], alpha[part]
            z = weight * batch['zg'] + (1 - weight) * batch['zl']
            predictions += [dict(arm=arm, run_seed=seed, model=name, gate_seed=gate_seed, partition=part,
                                 **row, logit=float(zz), probability_fake=float(expit(zz)), threshold=cut,
                                 alpha_global=float(aa)) for row, zz, aa in zip(batch['rows'], z, weight)]
    raw_saved = {name: {key: value if key == 'rows' else value.tolist() for key, value in batch.items()}
                 for name, batch in raw.items()}
    return predictions, dict(settings=settings, fixed_search=search, gates=gate_artifacts), experts, raw_saved


def execute(seed, arm, protocol, signature):
    import joblib
    from round8_a_verify import metrics, verify_run
    out = a.OUT / f'seed{seeds.number(seed)}' / arm
    art = a.ART / f'seed{seeds.number(seed)}' / arm
    if (out / 'complete.json').exists():
        verify_run(seed, arm)
        print(f'Verified and skipped completed seed {seed}/{arm}', flush=True)
        return
    records, parts, deny = a.context(seed)
    guard = a.Guard(records, parts, deny)
    bundle_path = art / 'bundle.json'
    started = time.perf_counter()
    if bundle_path.exists():
        bundle = c.read(bundle_path)
        a.require(bundle['signature'] == signature, 'Bundle protocol changed')
        for path, digest in bundle['feature_sha256'].items():
            a.require(c.sha(a.ROOT / path) == digest, 'Completed features changed')
        a.require(c.sha(art / 'experts.joblib') == bundle['experts_sha256'], 'Experts changed')
    else:
        features = Features(seed, records, parts, guard, protocol, signature)
        predictions, settings, experts, raw = fit_arm(seed, arm, records, parts, guard, features.get)
        stream = io.BytesIO(); joblib.dump(experts, stream)
        a.publish(art / 'experts.joblib', stream.getvalue())
        bundle = dict(signature=signature, run_seed=seed, arm=arm, predictions=predictions, settings=settings,
                      raw_batches=raw, feature_sha256=features.hashes, experts_sha256=c.sha(art / 'experts.joblib'),
                      seconds=time.perf_counter() - started,
                      events={k: sorted(v) for k, v in guard.events.items()}, violations=guard.violations)
        a.write_json(bundle_path, bundle)
    a.write_csv(out / 'predictions.csv', bundle['predictions'])
    a.write_json(out / 'settings.json', bundle['settings'])
    a.write_csv(out / 'metrics.csv', metrics(bundle['predictions']))
    a.write_json(out / 'audit.json', {k: bundle[k] for k in ('run_seed', 'arm', 'events', 'violations', 'seconds')})
    paths = [out / n for n in ('predictions.csv', 'settings.json', 'metrics.csv', 'audit.json')]
    paths += [bundle_path, art / 'experts.joblib']
    a.write_json(out / 'complete.json', dict(status='completed', signature=signature,
                 files_sha256={p.relative_to(a.ROOT).as_posix(): c.sha(p) for p in paths}))
    verify_run(seed, arm)
    print(f'Completed and verified seed {seed}/{arm}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--preflight', action='store_true')
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--run', action='store_true')
    parser.add_argument('--download', action='store_true', help='Download missing pinned source shards during prepare')
    parser.add_argument('--seed', type=int, choices=seeds.SEEDS)
    parser.add_argument('--arm', choices=a.ARMS)
    args = parser.parse_args()
    if args.download and not args.prepare:
        parser.error('--download requires --prepare')
    if args.preflight:
        print(json.dumps(a.preflight(), indent=2))
        return
    with a.run_lock():
        if args.prepare:
            print(json.dumps(a.prepare(args.download), indent=2))
            return
        # Check source availability before locking a scientific run.
        for seed in ([args.seed] if args.seed else seeds.SEEDS):
            records, _, _ = a.context(seed)
            missing = [i for i in records if a.find_original(i) is None]
            a.require(not missing, f'{len(missing)} allowed originals missing; run --prepare first')
        protocol, signature = a.ensure_protocol()
        for seed in ([args.seed] if args.seed else seeds.SEEDS):
            for arm in ([args.arm] if args.arm else a.ARMS):
                execute(seed, arm, protocol, signature)
        from round8_a_verify import summarize
        if all((a.OUT / f'seed{i}' / arm / 'complete.json').exists() for i in (1, 2, 3) for arm in a.ARMS):
            summarize()


if __name__ == '__main__':
    main()
