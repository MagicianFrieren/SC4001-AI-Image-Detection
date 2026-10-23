"""Verify saved round-two predictions, compute paired intervals, and plot results."""
from collections import Counter
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

from run_round2 import OUT, ROOT, SEED, PROFILES, CONDITIONS, read_manifest, write_json, dump_csv
import prepare_data as data


def read_csv(path):
    with path.open(encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def independent_threshold(y, p):
    thresholds = np.unique(p)
    predictions = p[None, :] >= thresholds[:, None]
    j = predictions[:, y == 1].mean(axis=1) - predictions[:, y == 0].mean(axis=1)
    best = thresholds[np.isclose(j, j.max())]
    # Match the pilot's descending ROC threshold order for equidistant ties.
    return sorted(best, key=lambda t: (abs(t - .5), -t))[0]


def main():
    metrics = read_csv(OUT / 'metrics.csv')
    predictions = read_csv(OUT / 'predictions.csv')
    validation = read_csv(OUT / 'validation_predictions.csv')
    records = read_manifest(OUT / 'manifest.jsonl')
    original = read_manifest(ROOT / 'results/manifest.jsonl')
    manifest = {r['source_id']: r for r in records}
    groups = json.loads((OUT / 'assignments.json').read_text())
    metadata = json.loads((OUT / 'run_metadata.json').read_text())
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert metadata['manifest_sha256'] == data.sha256(OUT / 'manifest.jsonl')
    assert metadata['protocol_sha256'] == data.sha256(OUT / 'protocol.json')
    assert metadata['script_sha256'] == data.sha256(ROOT / 'scripts/run_round2.py')
    assert metadata['pilot_script_sha256'] == data.sha256(ROOT / 'scripts/run_pilot.py')
    assert metadata['data_script_sha256'] == data.sha256(ROOT / 'scripts/prepare_data.py')
    assert records[:len(original)] == original
    assert len(manifest) == len(records) == 1600
    assert set(groups['development']) == {r['source_id'] for r in original if r['split'] == 'development'}
    for variant in groups['train']:
        train, val, dev = [set(groups[s][variant] if s != 'development' else groups[s])
                           for s in ['train', 'validation', 'development']]
        assert not (train & val or train & dev or val & dev)
        for split, n in [('train', 300), ('validation', 100)]:
            ids = groups[split][variant]
            assert len(ids) == len(set(ids)) == n * 2
            assert all(manifest[i]['split'] == split for i in ids)
            counts = Counter(manifest[i]['generator'] for i in ids)
            expected = {'Real': n, 'SD15': n} if variant == 'single_sd15' else {'Real': n, 'SD15': n // 2, 'BigGAN': n // 2}
            assert counts == expected
    assert len(metrics) == 108 and len(predictions) == 14400 and len(validation) == 2400
    assert len({(r['model'], r['condition'], r['source_id']) for r in predictions}) == len(predictions)
    assert len({(r['model'], r['source_id']) for r in validation}) == len(validation)
    for row in predictions + validation:
        record = manifest[row['source_id']]
        assert int(row['label']) == record['label']
        if 'generator' in row:
            assert row['generator'] == record['generator']
        assert np.isfinite(float(row['probability_fake'])) and 0 <= float(row['probability_fake']) <= 1
    thresholds = {}
    for model in sorted({r['model'] for r in metrics}):
        rows = [r for r in validation if r['model'] == model]
        variant = model.split('/')[2]
        assert {r['source_id'] for r in rows} == set(groups['validation'][variant])
        y = np.array([int(r['label']) for r in rows])
        p = np.array([float(r['probability_fake']) for r in rows])
        thresholds[model] = independent_threshold(y, p)

    # Each cell has the same count per class. Reusing draws across models makes
    # comparisons paired by source ID within the same generator/condition.
    rng = np.random.default_rng(SEED)
    bootstrap_ids = np.concatenate([rng.integers(0, 100, (2000, 100)),
                                    rng.integers(100, 200, (2000, 100))], axis=1)
    intervals, distributions, points = [], {}, {}
    for cell in metrics:
        rows = sorted([r for r in predictions if r['model'] == cell['model'] and
                       r['condition'] == cell['condition'] and r['generator'] in ('Real', cell['generator'])],
                      key=lambda r: (int(r['label']), r['source_id']))
        assert len(rows) == int(cell['n']) == 200
        expected_ids = {i for i in groups['development'] if manifest[i]['generator'] in ('Real', cell['generator'])}
        assert {r['source_id'] for r in rows} == expected_ids
        y = np.array([int(r['label']) for r in rows])
        assert np.array_equal(y, np.repeat([0, 1], 100))
        p = np.array([float(r['probability_fake']) for r in rows])
        t = thresholds[cell['model']]
        assert np.isclose(t, float(cell['threshold']), atol=1e-12, rtol=0)
        wins = p[100:, None] - p[None, :100]
        auc = float(np.mean((wins > 0) + .5 * (wins == 0)))
        checks = {'auroc': auc, 'average_precision': average_precision_score(y, p),
                  'balanced_accuracy': balanced_accuracy_score(y, p >= t),
                  'real_false_positive_rate': np.mean(p[y == 0] >= t)}
        for field, value in checks.items():
            assert np.isclose(value, float(cell[field]), atol=1e-12, rtol=0), (cell, field)
        ranks = rankdata(p[bootstrap_ids], axis=1)
        aucs = (ranks[:, 100:].sum(axis=1) - 100 * 101 / 2) / 10000
        key = (cell['model'], cell['condition'], cell['generator'])
        distributions[key], points[key] = aucs, auc
        lo, hi = np.percentile(aucs, [2.5, 97.5])
        intervals.append(dict(cell, auc_ci_low=float(lo), auc_ci_high=float(hi)))
    dump_csv(OUT / 'metrics_with_intervals.csv', intervals)

    contrasts = []
    for kind in ['clip', 'quality']:
        pairs = []
        for profile in PROFILES:
            pairs.append(('mixed_minus_single', f'{kind}/{profile}/mixed_sd15_biggan', f'{kind}/{profile}/single_sd15'))
        for variant in ['single_sd15', 'mixed_sd15_biggan']:
            for a, b in [('common128', 'original'), ('common128_jpeg90', 'common128'), ('common128_jpeg90', 'original')]:
                pairs.append((f'{a}_minus_{b}', f'{kind}/{a}/{variant}', f'{kind}/{b}/{variant}'))
        for contrast, a, b in pairs:
            for condition in CONDITIONS:
                for generator in ['SD15', 'BigGAN', 'ADM']:
                    ka, kb = (a, condition, generator), (b, condition, generator)
                    lo, hi = np.percentile(distributions[ka] - distributions[kb], [2.5, 97.5])
                    contrasts.append(dict(contrast=contrast, model_a=a, model_b=b, condition=condition,
                                          generator=generator, auroc_delta=points[ka] - points[kb],
                                          paired_ci_low=float(lo), paired_ci_high=float(hi)))
    dump_csv(OUT / 'paired_contrasts.csv', contrasts)

    # The unchanged arm should reproduce the pilot within floating-point tolerance.
    prior = read_csv(ROOT / 'results/predictions.csv')
    old = {(r['condition'], r['source_id']): float(r['probability_fake']) for r in prior if r['model'] == 'clip_clean'}
    unchanged = [r for r in predictions if r['model'] == 'clip/original/single_sd15']
    max_error = max(abs(float(r['probability_fake']) - old[(r['condition'], r['source_id'])]) for r in unchanged)
    assert max_error < 1e-6, f'Original arm differs from pilot: {max_error}'
    verification = {'status': 'passed', 'metric_cells_checked': len(metrics), 'heads_checked': len(thresholds),
                    'paired_contrasts': len(contrasts), 'bootstrap_draws': 2000,
                    'pilot_prediction_max_absolute_error': max_error,
                    'checks': ['Original 1200 records and development pool unchanged',
                               'Fixed 600/200 training/validation budgets and no partition overlap',
                               'ADM absent from training and validation', 'Source IDs and labels checked against manifest',
                               'All thresholds recomputed from saved validation predictions only',
                               'All AUROC/AP/balanced accuracy/FPR values recomputed',
                               'AUROC additionally checked with direct pairwise comparisons',
                               'Original CLIP arm compared against saved pilot predictions'],
                    'scope': 'Artifact and metric verification; not an independent end-to-end rerun'}
    write_json(OUT / 'verification.json', verification)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.7), sharey=True)
    colors = ['#222222', '#777777', '#aaaaaa']
    for ax, generator in zip(axes, ['SD15', 'BigGAN', 'ADM']):
        for profile, color in zip(PROFILES, colors):
            for variant, marker, style in [('single_sd15', 'o', '-'), ('mixed_sd15_biggan', 's', '--')]:
                model = f'clip/{profile}/{variant}'
                values = [points[(model, c, generator)] for c in CONDITIONS]
                label = f'{profile} / {"single" if variant == "single_sd15" else "mixed"}'
                ax.plot(range(3), values, color=color, marker=marker, linestyle=style, label=label)
        ax.set_title(generator + (' (unseen in both)' if generator == 'ADM' else ''))
        ax.set_xticks(range(3), ['Clean', 'JPEG50', 'Half-size'])
        ax.set_ylim(0, 1.02)
        ax.axhline(.5, color='#bbbbbb', linestyle=':', linewidth=1)
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('AUROC')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=8)
    fig.suptitle('Round 2: fixed-budget generator mixing and common preprocessing')
    fig.text(.5, .15, 'Exploratory development; one subset seed. BigGAN is seen in mixed training. Shared real pool.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .19, 1, .95])
    fig.savefig(OUT / 'round2_auroc.png', dpi=180)

    lines = ['Material Passport', 'Origin Skill: academic-research-suite / experiment-agent',
             'Origin Mode: run', 'Origin Date: 2026-10-22',
             'Verification Status: measured run completed; saved metrics checked (no independent full rerun)',
             'Version Label: round2_v1', '', 'SC4001 Round 2 — experiment result',
             'Status: completed', 'Commands:',
             '  ./.venv/Scripts/python.exe -u scripts/run_round2.py',
             '  ./.venv/Scripts/python.exe scripts/verify_round2.py',
             f'Working directory: {ROOT}', f'Run time: {metadata["run_seconds"]:.1f} seconds',
             f'Feature extraction: {metadata["extraction_seconds"]:.1f} seconds',
             f'Peak PyTorch CUDA allocation: {metadata["peak_cuda_memory_mb"]:.1f} MB',
             '', 'Design: six CLIP heads and six quality-only diagnostic heads.',
             'Each head: 300 real + 300 fake training; 100 real + 100 fake validation.',
             'Mixed: half of fake examples are SD15 and half BigGAN.',
             'All arms use the same 400-image development pool and 300 real training images.',
             'ADM remains absent from training and validation.', '',
             'CLIP clean development AUROC (point estimate and per-cell 95% bootstrap interval):']
    for row in intervals:
        if row['kind'] == 'clip' and row['condition'] == 'clean':
            lines.append(f"  {row['profile']:18} {row['training']:19} {row['generator']:6} "
                         f"{float(row['auroc']):.4f} [{row['auc_ci_low']:.4f}, {row['auc_ci_high']:.4f}]")
    lines += ['', 'ADM paired mixed-minus-single AUROC differences (per-cell 95% bootstrap intervals):']
    for row in contrasts:
        if row['contrast'] == 'mixed_minus_single' and row['model_a'].startswith('clip/') and row['generator'] == 'ADM':
            lines.append(f"  {row['model_a'].split('/')[1]:18} {row['condition']:8} "
                         f"{row['auroc_delta']:+.4f} [{row['paired_ci_low']:+.4f}, {row['paired_ci_high']:+.4f}]")
    lines += ['', 'Checks: ' + '; '.join(verification['checks']),
              f'Unchanged-arm max probability difference from pilot: {max_error:.3g}', '', 'Limitations:']
    lines += ['  - ' + item for item in protocol['limits']]
    lines += ['  - Bootstrap intervals do not include retraining/subset uncertainty and are not multiplicity corrected.',
              '  - Threshold comparisons also reflect the changed validation fake composition.',
              '  - Uniform processing cannot establish source-controlled or semantically matched data.',
              '', 'Resolved numerical anomaly:',
              'The first verification detected FP16 differences in the original final partial batch.',
              'Preserving the pilot batch boundary and regenerating all features resolved the discrepancy.',
              'See batch_boundary_diagnostic.json for the initial measurements; no experimental settings were tuned.',
              '', 'Artifacts: protocol.json, assignments.json, manifest.jsonl, metrics_with_intervals.csv,',
              'predictions.csv, validation_predictions.csv, paired_contrasts.csv, verification.json, round2_auroc.png.']
    (OUT / 'report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
