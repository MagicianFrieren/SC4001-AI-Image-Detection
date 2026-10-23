"""Read-only checks of published provenance and round-six point metrics; no GPU or images."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def check_hash(path, expected):
    with path.open('rb') as handle:
        actual = hashlib.file_digest(handle, 'sha256').hexdigest()
    if actual != expected:
        raise ValueError(f'Published provenance mismatch: {path.relative_to(ROOT)}')


def main():
    for number in range(2, 7):
        folder = ROOT / f'results/round{number}'
        meta = read_json(folder / 'run_metadata.json')
        assert read_json(folder / 'verification.json')['status'] == 'passed'
        code = meta.get('code_sha256', {})
        if number == 2:
            code = {name: meta[key] for name, key in [
                ('run_round2.py', 'script_sha256'), ('run_pilot.py', 'pilot_script_sha256'),
                ('prepare_data.py', 'data_script_sha256')]}
        for name, checksum in code.items():
            check_hash(ROOT / 'scripts' / name, checksum)
        check_hash(folder / 'protocol.json', meta['protocol_sha256'])
        if 'assignments_sha256' in meta:
            check_hash(folder / 'assignments.json', meta['assignments_sha256'])
        if 'manifest_sha256' in meta:
            check_hash(ROOT / 'results/round2/manifest.jsonl', meta['manifest_sha256'])

    folder = ROOT / 'results/round6'
    manifest = {r['source_id']: r for r in [json.loads(line) for line in
                (ROOT / 'results/round2/manifest.jsonl').read_text(encoding='utf-8').splitlines()]}
    groups, seen = defaultdict(list), set()
    for row in read_csv(folder / 'predictions.csv'):
        key = tuple(row[k] for k in ('arm', 'model', 'seed', 'condition'))
        identity = key + (row['source_id'],)
        assert identity not in seen
        seen.add(identity)
        source = manifest[row['source_id']]
        assert source['split'] == 'development'
        assert (source['generator'], source['label']) == (row['generator'], int(row['label']))
        assert np.isclose(expit(float(row['logit'])), float(row['probability_fake']), atol=1e-14, rtol=0)
        groups[key].append(row)
    assert len(seen) == 192000
    metrics = read_csv(folder / 'metrics.csv')
    assert len(metrics) == 1440
    settings = read_json(folder / 'settings.json')
    aggregate = defaultdict(list)
    for cell in metrics:
        key = tuple(cell[k] for k in ('arm', 'model', 'seed', 'condition'))
        rows = [r for r in groups[key] if r['generator'] in ('Real', cell['generator'])]
        assert len(rows) == int(cell['n']) == 200
        y = np.array([int(r['label']) for r in rows])
        assert y.sum() == 100
        z, p, alpha = [np.array([float(r[k]) for r in rows])
                       for k in ('logit', 'probability_fake', 'alpha_global')]
        threshold = settings[cell['arm']]['models'][f'{cell["model"]}/{cell["seed"]}']['threshold']
        assert threshold == float(cell['threshold'])
        actual = dict(auroc=roc_auc_score(y, z), average_precision=average_precision_score(y, z),
                      bce=np.mean(np.logaddexp(0, z) - y * z), brier=np.mean((p - y) ** 2),
                      balanced_accuracy=balanced_accuracy_score(y, p >= threshold),
                      real_false_positive_rate=np.mean(p[y == 0] >= threshold),
                      mean_alpha_global=alpha.mean(), std_alpha_global=alpha.std())
        for name, value in actual.items():
            assert np.isclose(value, float(cell[name]), atol=1e-12, rtol=0), (key, name)
        aggregate[(cell['arm'], cell['model'], cell['generator'])].append(actual['auroc'])
    macro = read_csv(folder / 'macro_summary.csv')
    assert len(macro) == len(aggregate) == 288
    for row in macro:
        values = aggregate[(row['arm'], row['model'], row['generator'])]
        assert len(values) == (9 if row['model'].endswith('_gate') else 3)
        assert np.isclose(np.mean(values), float(row['macro_auroc']), atol=1e-12, rtol=0)
    print(json.dumps({'status': 'passed', 'rounds_with_provenance_checked': 5,
                      'round6_predictions': len(seen), 'round6_metric_cells': len(metrics),
                      'round6_macro_cells': len(macro),
                      'scope': 'Source/protocol hashes and round6 point metrics only; no retraining, '
                               'feature reconstruction, threshold reselection or bootstrap verification'}, indent=2))


if __name__ == '__main__':
    main()
