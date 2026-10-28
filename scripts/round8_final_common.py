"""Round 8 final: immutable storage, declarations, provenance and explicit freeze."""
import sys
sys.dont_write_bytecode = True
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import contextlib
import hashlib
import json
import platform
from pathlib import Path
from importlib.metadata import version
import round7_baseline_common as c
import round7_seed_support as seeds
from round7_stepD_common import publish as _publish, json_bytes, csv_bytes

ROOT = c.ROOT
OUT = ROOT / 'results/round8_final'
ART = ROOT / 'artifacts/round8_final'
FREEZE = OUT / 'freeze.json'
TEST = OUT / 'test_manifest.jsonl'
CONDITIONS = tuple(c.CONDITIONS)
GENERATORS = ('GLIDE', 'Midjourney', 'VQDM', 'Wukong', 'ADM', 'BigGAN')
MAIN = GENERATORS[:4]
METRICS = ('auroc', 'average_precision', 'balanced_accuracy', 'real_false_positive_rate')
EXCLUSIONS = ('results/round2/manifest.jsonl', 'results/round7/fitting_pool_manifest.jsonl',
              'results/round7/development_manifest.jsonl', 'results/round7/reserved_test_manifest.jsonl')
require, sha, read, rows = c.require, c.sha, c.read, c.rows


def rel(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def publish(path, payload):
    # Only our new tree is writable, except the twelve explicitly checked handoff files.
    name = rel(path)
    require(name.startswith(('results/round8_final/', 'artifacts/round8_final/')),
            'Attempted write outside new Round 8 final trees')
    _publish(path, payload, resume=True)


def write(path, value):
    publish(path, json_bytes(value))


def environment():
    require(c.PIL.__version__ == '12.2.0', 'Pillow 12.2.0 required; restart kernel with the pinned environment')
    import torch
    return dict(python=platform.python_version(), pillow=c.PIL.__version__,
                packages={n: version(n) for n in ('numpy', 'scipy', 'scikit-learn', 'joblib', 'torch', 'torchvision', 'transformers', 'pyarrow')},
                device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')


def check_environment():
    expected = read(ROOT / 'results/round8_a/protocol.json')['environment']
    actual = environment()
    observed = dict(actual['packages'], python=actual['python'], Pillow=actual['pillow'])
    differences = {k: {'expected': expected[k], 'actual': v} for k, v in observed.items() if v != expected[k]}
    require(not differences, 'Use A\'s recorded runtime: ' + json.dumps(differences))
    require(actual['device'] != 'cpu', 'CUDA is required to reproduce A\'s FP16 encoder')
    return {'status': 'passed', 'environment': actual}


def verify_archive():
    folder = OUT / 'archive/failed_development'
    if not (folder / 'manifest.json').exists():
        return
    require((folder / 'complete.json').exists(), 'Finish archiving failed development before continuing')
    require(read(folder / 'complete.json')['manifest_sha256'] == sha(folder / 'manifest.json'), 'Archive manifest changed')
    for item in read(folder / 'manifest.json')['files']:
        require(sha(ROOT / item['archived_path']) == item['sha256'], 'Archived evidence changed: ' + item['archived_path'])


def archive_failed():
    """User-run, resumable preservation of failed pre-test work; never delete bytes."""
    folder = OUT / 'archive/failed_development'
    manifest = folder / 'manifest.json'
    if (folder / 'complete.json').exists():
        verify_archive()
        return {'status': 'already_archived', 'archive': rel(folder)}
    require(not FREEZE.exists() and not TEST.exists() and not (OUT / 'evaluation').exists()
            and not (OUT / 'complete.json').exists(), 'Cannot archive after test preparation/freeze/evaluation')
    if not manifest.exists():
        failed = OUT / 'handoff_reproduction.json'
        if not failed.exists():
            return {'status': 'nothing_to_archive'}
        if read(failed)['status'] == 'passed':
            return {'status': 'reproduction_already_passed'}
        require(read(failed)['status'] == 'failed', 'Unknown reproduction state')
        sources = [OUT / n for n in ('handoff_integrity.json', 'handoff_reproduction.json', 'preservation.json', 'reproduction_plan.json')]
        sources += [p for name in ('development', 'portable') for p in (ART / name).rglob('*') if p.is_file()]
        items = []
        for path in sorted(sources):
            if not path.exists():
                continue
            base = OUT if path.is_relative_to(OUT) else ART
            destination = base / 'archive/failed_development' / path.relative_to(base)
            # Validate both absolute paths before any move; no computed target may escape.
            require(path.resolve().is_relative_to(base.resolve()) and
                    destination.resolve().is_relative_to((base / 'archive/failed_development').resolve()), 'Unsafe archive path')
            require(not destination.exists(), 'Archive destination exists: ' + rel(destination))
            items.append(dict(original_path=rel(path), archived_path=rel(destination), sha256=sha(path)))
        write(manifest, dict(reason='Preserve failed development reproduction and incompatible feature cache before recreating the original active output paths.', files=items))
    for item in read(manifest)['files']:
        source, destination = ROOT / item['original_path'], ROOT / item['archived_path']
        require(any(source.resolve().is_relative_to(base.resolve()) and
                    destination.resolve().is_relative_to((base / 'archive/failed_development').resolve()) for base in (OUT, ART)), 'Unsafe archived path')
        require('/archive/' not in rel(source), 'Invalid source in archive plan')
        if destination.exists():
            require(sha(destination) == item['sha256'] and not source.exists(), 'Archive collision or changed source')
        else:
            require(source.is_file() and sha(source) == item['sha256'], 'Archive source changed')
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
    write(folder / 'complete.json', dict(status='archived', manifest_sha256=sha(manifest)))
    verify_archive()
    return {'status': 'archived', 'archive': rel(folder), 'active_outputs': rel(OUT)}


@contextlib.contextmanager
def lock():
    ART.mkdir(parents=True, exist_ok=True)
    with (ART / 'runner.lock').open('a+b') as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def protect(create=False):
    """Snapshot every existing historical research file; no changes or deletions allowed."""
    target = OUT / 'preservation.json'
    if not target.exists():
        require(create, 'Run handoff first to create preservation snapshot')
        paths = []
        for folder in ('scripts', 'docs', 'config', 'configs', 'results', 'artifacts', 'data'):
            for p in (ROOT / folder).rglob('*'):
                if not p.is_file() or '__pycache__' in p.parts:
                    continue
                name = rel(p)
                if name.startswith(('results/round8_final/', 'artifacts/round8_final/')) or 'round8_final' in p.name.lower():
                    continue
                paths.append(p)
        paths += [p for p in ROOT.iterdir() if p.is_file()]
        write(target, {'files_sha256': {rel(p): sha(p) for p in sorted(set(paths))}})
    for name, digest in read(target)['files_sha256'].items():
        require((ROOT / name).is_file() and sha(ROOT / name) == digest, f'Historical file changed: {name}')
    verify_archive()


def declarations():
    """Read Round 7's already frozen declarations; do not reselect models."""
    old = read(ROOT / 'results/round7/final_freeze.json')
    result = []
    for item in old['models']:
        if item['profile'] != 'original' or item['calibration'] != 'raw' or item['model'] == 'W2':
            continue
        r = dict(item, cohort='round7', arm='saved', old_id=item['id'])
        r['id'] = 'R7_' + item['id']
        r['portable'] = rel(ART / 'portable' / (r['id'] + '.json'))
        for field in ('head', 'checkpoint', 'settings_path'):
            if field in r:
                require(sha(ROOT / r[field]) == old['files_sha256'][r[field]], f'Round 7 frozen artifact changed: {r[field]}')
        result.append(r)
    for seed in seeds.SEEDS:
        n = seeds.number(seed)
        for arm in ('unmatched', 'matched'):
            base = f'artifacts/round8_a/seed{n}/{arm}'
            bundle = read(ROOT / base / 'bundle.json')
            saved = bundle['settings']
            for key, setting in sorted(saved['settings'].items()):
                model, init = key.split('/')
                mid = f'R8A_s{n}_{arm}_{model}_{init}'
                result.append(dict(id=mid, cohort='round8_a', arm=arm, seed=seed,
                    profile='original', calibration='raw', model=model, initialization=int(init), family='clip',
                    head=f'{base}/experts.joblib', settings_path=f'{base}/bundle.json', settings_key=key,
                    portable=rel(ART / 'portable' / (mid + '.json')),
                    threshold=setting['threshold'], fixed_alpha=setting['fixed_alpha'], global_temperature=1., local_temperature=1.))
    require(len(result) == 96 and len({m['id'] for m in result}) == 96, 'Expected 96 instances (gate initializations included)')
    require(all(0 <= m['threshold'] <= 1 for m in result), 'Invalid saved threshold')
    return sorted(result, key=lambda m: m['id'])


def code_hashes():
    return {rel(p): sha(p) for p in sorted((ROOT / 'scripts').glob('*.py'))}


def notebook_hash():
    book = read(ROOT / 'docs/ROUND8_FINAL.ipynb')
    source = [{'cell_type': cell['cell_type'], 'source': ''.join(cell['source'])} for cell in book['cells']]
    return hashlib.sha256(json_bytes(source)).hexdigest()


def build_freeze():
    require(not FREEZE.exists(), 'Freeze exists; verify or resume, never replace it')
    require(not (OUT / 'evaluation').exists(), 'Evaluation started before freeze')
    check_environment()
    protect()
    report = read(OUT / 'handoff_reproduction.json')
    require(report['status'] == 'passed', 'Development reproduction must PASS before freeze')
    require(report['environment'] == environment() and report['code_sha256'] == code_hashes(), 'Reproduction environment/code changed')
    for name, digest in report['files_sha256'].items():
        require(sha(ROOT / name) == digest, f'Reproduction input changed: {name}')
    from round8_final_data import verify_data
    verify_data()
    audit = read(OUT / 'data_audit.json')
    models = declarations()
    paths = set()
    for folder in ('scripts', 'config', 'configs'):
        paths.update(p for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.update(p for p in OUT.iterdir() if p.is_file())
    paths.update(p for p in (OUT / 'archive/failed_development').glob('*.json'))
    paths.update(p for p in (ART / 'portable').glob('*.json'))
    paths.add(ROOT / 'docs/ROUND8_FINAL.md')
    paths.update(ROOT / name for name in report['files_sha256'])
    paths.update(ROOT / name for name in audit['input_sha256'])
    for m in models:
        paths.update(ROOT / m[k] for k in ('head', 'checkpoint', 'settings_path') if k in m)
        if m['family'] == 'resnet':
            paths.add((ROOT / m['settings_path']).parent / 'protocol.json')
    import round8_a_common as a
    weights = a.weights_path()
    paths.update(weights / n for n in ('pytorch_model.bin', 'config.json'))
    paths.update(ROOT / n for n in EXCLUSIONS)
    paths.update(ROOT / n for n in ('results/round7/source_inventory.json', 'results/round7/final_freeze.json',
                                   'results/round7/final_test/metrics_each_seed.csv', 'results/round7/final_test/complete.json',
                                   'results/round8_a/protocol.json', 'results/round8_a/predictions_archive.json'))
    paths.update(p for p in (ROOT / 'results/round8_a').rglob('complete.json'))
    paths.update(p for p in (ROOT / 'results/round8_a/model_handoff').iterdir() if p.is_file())
    value = dict(id='round8_final_once', protocol_owner='user', environment=environment(),
        models=models, seeds=list(seeds.SEEDS), conditions=list(CONDITIONS), profile='original', calibration='raw',
        notebook_source_sha256=notebook_hash(), notebook_hash_rule='Canonical cell types and source only; runtime outputs/execution counts excluded.',
        test_manifest=rel(TEST), test_manifest_sha256=sha(TEST), expected_counts=audit['counts'],
        weights=rel(weights), files_sha256={rel(p): sha(p) for p in sorted(paths)},
        no_in_domain_test='No SD15 is available for this fresh test; no in-domain result.',
        metrics=list(METRICS), macro=list(MAIN),
        evaluation='Original images only; inherited geometry and degradation. No format matching at inference. No fitting, threshold fitting, epoch selection or weight changes.',
        aggregation='Average gate-initialization metrics within each seed; then mean and sample SD of three seeds. Shared real pool predicted once per model/condition.',
        comparisons=['W1-W0', 'matched-unmatched local', 'matched-unmatched average', 'matched quality_gate-matched average', 'same Round 7 model: Round 8 test minus Round 7 test'],
        comparison_limit='Cross-test differences pair frozen models/seeds, not images; they do not estimate an intervention effect. Round 8A models have no Round 7 test results.',
        freshness='New relative to the four repository exclusion manifests and their pixel/dHash neighbours. External independent pilot membership cannot be certified from this repository. Generator domains have been inspected before.',
        once='Completed evaluation refuses all reruns. --resume only reuses immutable completed chunks in an incomplete run. Interrupted uncommitted work may repeat.',
        created_before_evaluation=True)
    write(FREEZE, value)
    print(f'Frozen {len(models)} model instances. Test inference has NOT run.', flush=True)


def freeze():
    require(FREEZE.is_file(), 'Freeze must exist before evaluation opens test manifest')
    value = read(FREEZE)
    check_environment()
    require(value['environment'] == environment(), 'Frozen runtime changed')
    require(value['notebook_source_sha256'] == notebook_hash(), 'Frozen notebook source changed')
    for name, digest in value['files_sha256'].items():
        require(sha(ROOT / name) == digest, f'Frozen file changed: {name}')
    verify_archive()
    return value


class Guard:
    def __init__(self, allowed):
        self.allowed = set(allowed)
        self.touched = set()
    def touch(self, ids, action=''):
        ids = set(ids)
        require(ids <= self.allowed, f'Isolation violation during {action}')
        self.touched.update(ids)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Round 8 environment and failed-development preservation')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check-environment', action='store_true')
    action.add_argument('--archive-failed', action='store_true')
    args = parser.parse_args()
    checked = check_environment()
    if args.check_environment:
        print(json.dumps(checked, indent=2))
    else:
        with lock():
            print(json.dumps(archive_failed(), indent=2))
