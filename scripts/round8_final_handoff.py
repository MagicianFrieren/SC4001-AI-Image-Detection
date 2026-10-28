"""User-run archive restoration and development reproduction, BEFORE test creation."""
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import hashlib
import io
import json
import warnings
import zipfile
import numpy as np
from scipy.special import expit
import round8_final_common as f
import round8_final_inference as inference

# Declared before execution. No threshold/model changes are permitted to pass these.
LOGIT_ATOL = 1e-3
PROB_ATOL = 2.5e-4
ALPHA_ATOL = 1e-4


def reproduction_identity(identity_spec):
    """Preserve the pre-correction plan and validate reuse of unaffected features."""
    plan = dict(identity=identity_spec, logit_atol=LOGIT_ATOL,
                probability_atol=PROB_ATOL, alpha_atol=ALPHA_ATOL,
                allowed_classification_disagreements=0,
                scope='Every development image, all four conditions, all six A runs and all gate initializations. No fresh test data.')
    path = f.OUT / 'reproduction_plan.json'
    if not path.exists():
        f.write(path, plan)
    saved = f.read(path)
    if saved != plan:
        # Only the two known source revisions below qualify. No arbitrary code,
        # runtime, image, weight or acceptance-policy change may reuse this cache.
        previous = {
            'scripts/round8_final_inference.py': '91712a0cc837dd6590971f2dd28f67f033624bfbe32fc2551c601ef9ae3663b1',
            'scripts/round8_final_handoff.py': 'ba82b9c7cccb5a25d0edac0ef814f1deaebade1cb1eca11170693e57a1f46caf',
        }
        old_identity = saved['identity']
        old_code, new_code = old_identity['code'], identity_spec['code']
        f.require({k: v for k, v in saved.items() if k != 'identity'} ==
                  {k: v for k, v in plan.items() if k != 'identity'}, 'Reproduction criteria changed')
        f.require({k: v for k, v in old_identity.items() if k != 'code'} ==
                  {k: v for k, v in identity_spec.items() if k != 'code'}, 'Feature inputs/runtime changed')
        f.require(set(old_code) == set(new_code), 'Implementation file set changed')
        changed = {k for k in old_code if old_code[k] != new_code[k]}
        f.require(changed == set(previous) and all(old_code[k] == v for k, v in previous.items()),
                  'Cache continuation is permitted only for the recorded scaler arithmetic correction')
        feature_identity = hashlib.sha256(f.json_bytes(old_identity)).hexdigest()
        f.write(f.OUT / 'head_arithmetic_correction.json', dict(
            reason='Match sklearn 1.8: cast scaler mean and scale to feature dtype before subtraction/division.',
            old_plan_sha256=f.sha(path), old_code_sha256={k: old_code[k] for k in sorted(changed)},
            corrected_code_sha256={k: new_code[k] for k in sorted(changed)},
            feature_cache_identity=feature_identity,
            unchanged='CLIP forward pass, image transforms, source IDs, weights, runtime, head parameters, thresholds and numerical tolerances. Existing feature batches are hash-validated and reused.',
            pre_test=True))
        print('Recorded scaler arithmetic correction; reusing validated CLIP feature batches.', flush=True)
        return feature_identity
    return hashlib.sha256(f.json_bytes(identity_spec)).hexdigest()


def sums(text):
    result = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.lstrip('*').replace('\\', '/')
        f.require(len(digest) == 64 and all(c in '0123456789abcdef' for c in digest.lower()), 'Invalid checksum')
        f.require(name not in result, 'Duplicate checksum path')
        result[name] = digest.lower()
    return result


def restore():
    f.check_environment()
    f.protect(create=True)
    base = f.ROOT / 'results/round8_a/model_handoff'
    archive = base / 'round8_a_models.zip'
    external = sums((base / 'SHA256SUMS.txt').read_text(encoding='utf-8-sig'))
    claimed = (base / 'round8_a_models.zip.sha256').read_text(encoding='utf-8-sig').split()[0].lower()
    f.require(f.sha(archive) == claimed == external['round8_a_models.zip'], 'ZIP checksum mismatch')
    expected = {f'artifacts/round8_a/seed{n}/{arm}/{name}' for n in (1, 2, 3)
                for arm in ('matched', 'unmatched') for name in ('experts.joblib', 'bundle.json')}
    f.require(set(external) == expected | {'round8_a_models.zip'}, 'Unexpected external checksum list')
    payloads = {}
    with zipfile.ZipFile(archive) as z:
        members = [i for i in z.infolist() if not i.is_dir()]
        names = [i.filename.replace('\\', '/') for i in members]
        f.require(len(names) == len(set(names)), 'Duplicate ZIP entries')
        internal_names = [n for n in names if n.split('/')[-1] == 'SHA256SUMS.txt']
        f.require(len(internal_names) == 1 and set(names) == expected | set(internal_names), 'Unexpected ZIP paths; refusing extraction')
        internal = sums(z.read(internal_names[0]).decode('utf-8-sig'))
        f.require({n: internal.get(n) for n in expected} == {n: external[n] for n in expected}, 'Internal/external checksums disagree')
        for name in sorted(expected):
            blob = z.read(name)
            f.require(hashlib.sha256(blob).hexdigest() == external[name], 'ZIP member checksum mismatch')
            _, _, seed_name, arm, _ = name.split('/')
            receipt = f.read(f.ROOT / f'results/round8_a/{seed_name}/{arm}/complete.json')
            f.require(receipt['status'] == 'completed' and receipt['signature'] == f.sha(f.ROOT / 'results/round8_a/protocol.json')
                      and receipt['files_sha256'][name] == external[name], 'Completion receipt mismatch')
            payloads[name] = blob
    # Validate every destination BEFORE writing any. No extractall/path traversal.
    for name, blob in payloads.items():
        dest = f.ROOT / name
        f.require(not dest.exists() or dest.read_bytes() == blob, f'Existing handoff file differs: {name}')
    from round7_stepD_common import publish
    for name, blob in payloads.items():
        if not (f.ROOT / name).exists():
            publish(f.ROOT / name, blob)
    prediction_meta = f.read(f.ROOT / 'results/round8_a/predictions_archive.json')
    prediction_zip = f.ROOT / prediction_meta['archive']
    f.require(f.sha(prediction_zip) == prediction_meta['archive_sha256'], 'Prediction ZIP checksum mismatch')
    with zipfile.ZipFile(prediction_zip) as z:
        names = [i.filename for i in z.infolist() if not i.is_dir()]
        f.require(len(names) == len(set(names)) and set(names) == set(prediction_meta['files_sha256']), 'Prediction archive coverage mismatch')
        for name, digest in prediction_meta['files_sha256'].items():
            f.require(hashlib.sha256(z.read(name)).hexdigest() == digest, 'Prediction member checksum mismatch')
            receipt = f.read((f.ROOT / name).parent / 'complete.json')
            f.require(receipt['files_sha256'][name] == digest, 'Prediction receipt mismatch')
            for old_name, old_digest in receipt['files_sha256'].items():
                if old_name != name:
                    f.require(f.sha(f.ROOT / old_name) == old_digest, f'Receipt file changed: {old_name}')
    value = dict(status='passed', archive_sha256=claimed, restored_files={n: f.sha(f.ROOT / n) for n in sorted(expected)},
                 predictions_zip_sha256=f.sha(prediction_zip), note='Original paths restored from repo root; existing bytes never replaced. Internal checksum file verified in memory.')
    f.write(f.OUT / 'handoff_integrity.json', value)
    return value


def export(models):
    diagnostics, loaded, portable_by_path = {}, {}, {}
    for m in models:
        if m['family'] != 'clip':
            continue
        if m['head'] not in loaded:
            loaded[m['head']], diagnostics[m['head']] = inference.load_parameters(f.ROOT / m['head'])
        saved = loaded[m['head']]
        if m['cohort'] == 'round8_a':
            bundle = f.read(f.ROOT / m['settings_path'])
            f.require(bundle['experts_sha256'] == f.sha(f.ROOT / m['head']) and not bundle['violations'], 'Bundle model/isolation mismatch')
            settings = bundle['settings']
            heads = dict(global_head=saved['global_head'], local_head=saved['local_head'])
            gate_params = {key: entry['artifact'] for key, entry in settings['gates'].items()}
        elif m['model'] == 'W1':
            heads = dict(global_head=saved['head']); gate_params = {}
        else:
            heads = dict(global_head=saved['global'], local_head=saved['local'])
            settings = f.read(f.ROOT / m['settings_path'])
            gate_params = {key.removeprefix('raw/'): entry['artifact'] for key, entry in settings['gates'].items() if key.startswith('raw/')}
        value = {'global': inference.linear_parameters(heads['global_head']), 'gates': gate_params,
                 'source_sha256': f.sha(f.ROOT / m['head']), 'settings_sha256': f.sha(f.ROOT / m['settings_path'])}
        if 'local_head' in heads:
            value['local'] = inference.linear_parameters(heads['local_head'])
        f.write(f.ROOT / m['portable'], value)
        portable_by_path[m['portable']] = value
    return portable_by_path, diagnostics, loaded


def reproduce():
    f.require(not f.FREEZE.exists(), 'Reproduction is a pre-freeze step')
    integrity = restore()
    models = f.declarations()
    portable, diagnostics, loaded = export(models)
    import round8_a_common as a
    records = {r['source_id']: r for r in f.rows(f.ROOT / 'results/round7/development_manifest.jsonl')}
    guard = f.Guard(records)
    weights = a.weights_path()
    identity_spec = dict(code=f.code_hashes(), environment=f.environment(),
                         manifest=f.sha(f.ROOT / 'results/round7/development_manifest.jsonl'),
                         weights={n: f.sha(weights / n) for n in ('pytorch_model.bin', 'config.json')})
    identity = reproduction_identity(identity_spec)
    inference.prepare_originals('development', records, guard)
    features = inference.features('development', records, guard, identity, weights)
    ids = sorted(records)
    reports, branch_differences = [], []
    native_diffs = []
    # Compare portable heads to usable native sklearn on real feature inputs; fallback
    # state bags instead rely on comparison with the owner's saved branch logits below.
    for path, native in loaded.items():
        matched_model = next(m for m in models if m.get('head') == path)
        p = portable[matched_model['portable']]
        for label, key in (('global', 'g'), ('local', 'l')):
            if label not in p:
                continue
            native_head = native.get(label, native.get(label + '_head', native.get('head')))
            if hasattr(native_head, 'decision_function'):
                worst = 0.
                try:
                    for condition in f.CONDITIONS:
                        x = features[condition][key]
                        if key == 'l':
                            x = x.mean(axis=1)
                        worst = max(worst, float(np.max(np.abs(native_head.decision_function(x) - inference.decision(p[label], x)))))
                except (AttributeError, TypeError) as error:
                    # A loaded estimator can still be incompatible with this sklearn.
                    # Preserve parameters; the mandatory archived-development comparison
                    # below remains the pass/fail evidence for the A handoff.
                    diagnostics[path]['native_prediction_error'] = repr(error)
                    continue
                f.require(worst <= 1e-10, f'Portable/native head disagreement {path}/{label}: {worst}')
                native_diffs.append(dict(path=path, head=label, max_abs_logit=worst))
    meta = f.read(f.ROOT / 'results/round8_a/predictions_archive.json')
    with zipfile.ZipFile(f.ROOT / meta['archive']) as z:
        for seed in f.seeds.SEEDS:
            for arm in ('unmatched', 'matched'):
                group = [m for m in models if m['cohort'] == 'round8_a' and m['seed'] == seed and m['arm'] == arm]
                bundle = f.read(f.ROOT / group[0]['settings_path'])
                name = f'results/round8_a/seed{f.seeds.number(seed)}/{arm}/predictions.csv'
                all_saved = list(csv.DictReader(io.StringIO(z.read(name).decode('utf-8-sig'))))
                expected = {(r['model'], int(r['gate_seed']), r['condition'], r['source_id']): r
                            for r in all_saved if r['partition'] == 'development'}
                f.require(len(expected) == len(group) * len(ids) * len(f.CONDITIONS), 'Development prediction coverage mismatch')
                raw = bundle['raw_batches']['development']
                f.require([(r['condition'], r['source_id']) for r in raw['rows']] == [(cond, i) for cond in f.CONDITIONS for i in ids], 'Saved raw batch ordering changed')
                p = portable[group[0]['portable']]
                for condition in f.CONDITIONS:
                    values = features[condition]
                    scores = inference.score(group, values, p)
                    offset = list(f.CONDITIONS).index(condition) * len(ids)
                    for label, key, saved_key in (('global', 'g', 'zg'), ('local', 'l', 'zl')):
                        x = values[key] if key == 'g' else values[key].mean(axis=1)
                        diff = float(np.max(np.abs(inference.decision(p[label], x) - np.asarray(raw[saved_key][offset:offset + len(ids)]))))
                        branch_differences.append(dict(seed=seed, arm=arm, branch=label, condition=condition, max_abs_logit=diff))
                    for m in group:
                        logits, alpha = scores[m['id']]
                        saved_rows = [expected[m['model'], m['initialization'], condition, i] for i in ids]
                        f.require(all(float(r['threshold']) == m['threshold'] and int(r['label']) == records[i]['label']
                                      and r['generator'] == records[i]['generator'] for i, r in zip(ids, saved_rows)), 'Saved metadata/threshold changed')
                        old_z = np.asarray([float(r['logit']) for r in saved_rows])
                        old_p = np.asarray([float(r['probability_fake']) for r in saved_rows])
                        old_a = np.asarray([float(r['alpha_global']) for r in saved_rows])
                        probs = expit(logits.astype(np.float64))
                        dz = float(np.max(np.abs(logits - old_z)))
                        dp = float(np.max(np.abs(probs - old_p)))
                        da = float(np.max(np.abs(alpha - old_a)))
                        disagreements = int(np.count_nonzero((probs >= m['threshold']) != (old_p >= m['threshold'])))
                        result = dict(model_id=m['id'], condition=condition, n=len(ids), max_abs_logit=dz,
                                      max_abs_probability=dp, max_abs_alpha=da, classification_disagreements=disagreements)
                        reports.append(result)
                        print(json.dumps(result), flush=True)
    passed = all(r['max_abs_logit'] <= LOGIT_ATOL for r in branch_differences) and all(r['max_abs_logit'] <= LOGIT_ATOL and r['max_abs_probability'] <= PROB_ATOL
                 and r['max_abs_alpha'] <= ALPHA_ATOL and r['classification_disagreements'] == 0 for r in reports)
    paths = [f.ROOT / n for n in integrity['restored_files']]
    paths += list((f.ART / 'portable').glob('*.json'))
    paths += [f.ROOT / meta['archive'], f.ROOT / 'results/round8_a/predictions_archive.json',
              f.OUT / 'handoff_integrity.json', f.OUT / 'reproduction_plan.json']
    paths += [weights / n for n in ('pytorch_model.bin', 'config.json')]
    paths += list((f.ART / 'development/features').glob('*.npz'))
    if (f.OUT / 'head_arithmetic_correction.json').exists():
        paths.append(f.OUT / 'head_arithmetic_correction.json')
    paths += [f.ROOT / m[k] for m in models for k in ('head', 'checkpoint', 'settings_path') if k in m]
    value = dict(status='passed' if passed else 'failed', environment=f.environment(), code_sha256=f.code_hashes(),
                 numerical_differences=reports, branch_differences=branch_differences, native_portable_differences=native_diffs, load_diagnostics=diagnostics,
                 tolerances=dict(logit=LOGIT_ATOL, probability=PROB_ATOL, alpha=ALPHA_ATOL, classification_disagreements=0),
                 files_sha256={f.rel(p): f.sha(p) for p in sorted(set(paths))},
                 note='Numerical equivalence within predeclared tolerances, not necessarily bit identity. All final heads use explicit scaler/logistic arrays; no refitting.')
    f.write(f.OUT / 'handoff_reproduction.json', value)
    f.protect()
    f.require(passed, 'Reproduction FAILED; STOP before fresh data/freeze/evaluation. See numerical differences; do not relax tolerances after test.')
    print('Handoff reproduction PASSED. No fresh test images have been evaluated.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore-only', action='store_true')
    args = parser.parse_args()
    with f.lock():
        print(restore()) if args.restore_only else reproduce()
