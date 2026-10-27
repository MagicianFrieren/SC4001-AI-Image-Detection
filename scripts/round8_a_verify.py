"""Read-only run verification and paired development summaries for Round 8 A."""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import defaultdict
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, roc_curve
import round8_a_common as a
from round8_a_common import c, seeds

METRICS = ('auroc', 'average_precision', 'balanced_accuracy', 'real_false_positive_rate')
INSTANCES = [('global', -1), ('local', -1), ('average', -1), ('fixed', -1)]
INSTANCES += [(name, seed) for name in ('score_gate', 'quality_gate') for seed in c.GATE_SEEDS]


def metrics(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in ('arm', 'run_seed', 'model', 'gate_seed', 'partition', 'condition'))].append(row)
    result = []
    for key, values in sorted(grouped.items()):
        arm, seed, model, init, part, condition = key
        for generator in (['SD15'] if part == 'selection' else ['ADM', 'BigGAN']):
            cell = [r for r in values if r['generator'] in ('Real', generator)]
            y = np.array([int(r['label']) for r in cell])
            z = np.array([float(r['logit']) for r in cell])
            cut = float(cell[0]['threshold'])
            a.require(all(float(r['threshold']) == cut for r in cell), 'Mixed thresholds')
            prediction = expit(z) >= cut
            result.append(dict(arm=arm, run_seed=int(seed), model=model, gate_seed=int(init),
                               partition=part, condition=condition, generator=generator,
                               n_real=int(sum(y == 0)), n_fake=int(sum(y == 1)),
                               auroc=float(roc_auc_score(y, z)), average_precision=float(average_precision_score(y, z)),
                               balanced_accuracy=float(balanced_accuracy_score(y, prediction)),
                               real_false_positive_rate=float(prediction[y == 0].mean())))
    return result


def inputs(batch):
    """Independent reconstruction of the eight gate inputs."""
    zg, zl = np.asarray(batch['zg']), np.asarray(batch['zl'])
    pg, pl = expit(zg), expit(zl)
    def entropy(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -(p * np.log(p) + (1 - p) * np.log1p(-p))
    return np.column_stack((pg, pl, np.abs(pg - pl), (entropy(pg) + entropy(pl)) / 2,
                            np.log1p(batch['variance']), np.asarray(batch['q'])))


def alpha(name, init, batch, settings):
    spec = settings['settings'][f'{name}/{init}']
    n = len(batch['y'])
    fixed = {'global': 1., 'local': 0., 'average': .5, 'fixed': spec['fixed_alpha']}
    if name in fixed:
        return np.full(n, fixed[name])
    artifact = settings['gates'][f'{name}/{init}']['artifact']
    x = ((inputs(batch) - artifact['mean']) / artifact['scale']).astype(np.float32)
    if name == 'score_gate':
        x[:, 5:] = 0
    state = {k: np.asarray(v, dtype=np.float32) for k, v in artifact['state'].items()}
    hidden = np.maximum(x @ state['hidden.weight'].T + state['hidden.bias'], 0)
    return expit(hidden @ state['output.weight'].T + state['output.bias']).ravel().astype(float)


def threshold(y, z):
    fpr, tpr, cuts = roc_curve(y, expit(z))
    finite = np.isfinite(cuts)
    score, cuts = (tpr - fpr)[finite], cuts[finite]
    best = np.flatnonzero(np.isclose(score, score.max()))
    return float(cuts[best[np.argmin(np.abs(cuts[best] - .5))]])


def verify_run(seed, arm):
    out = a.OUT / f'seed{seeds.number(seed)}' / arm
    art = a.ART / f'seed{seeds.number(seed)}' / arm
    protocol = c.read(a.OUT / 'protocol.json')
    signature = c.sha(a.OUT / 'protocol.json')
    for field in ('code_sha256', 'input_sha256'):
        for path, digest in protocol[field].items():
            a.require(c.sha(a.ROOT / path) == digest, f'Frozen input changed: {path}')
    a.require(c.sha(a.PLAN) == protocol['plan_sha256'], 'Development plan changed')
    complete = c.read(out / 'complete.json')
    a.require(complete['status'] == 'completed' and complete['signature'] == signature, 'Invalid completion')
    for path, digest in complete['files_sha256'].items():
        a.require(c.sha(a.ROOT / path) == digest, f'Completed file changed: {path}')
    bundle = c.read(art / 'bundle.json')
    a.require(bundle['signature'] == signature and bundle['run_seed'] == seed and bundle['arm'] == arm, 'Wrong result identity')
    for path, digest in bundle['feature_sha256'].items():
        a.require(c.sha(a.ROOT / path) == digest, f'Feature cache changed: {path}')
    records, parts, deny = a.context(seed)
    guard = a.Guard(records, parts, deny)
    a.require(not bundle['violations'], 'Recorded isolation violation')
    for action, ids in bundle['events'].items():
        guard.touch(ids, action)
    settings = c.read(out / 'settings.json')
    a.require(settings == bundle['settings'], 'Settings differ from atomic bundle')
    for name, part, conditions in (('gate_fit', 'gate_fit', c.TUNE), ('tuning', 'selection', c.TUNE),
                                    ('selection', 'selection', c.CONDITIONS), ('development', 'development', c.CONDITIONS)):
        batch = bundle['raw_batches'][name]
        expected = [(cond, i) for cond in conditions for i in parts[part]]
        actual = [(r['condition'], r['source_id']) for r in batch['rows']]
        a.require(actual == expected, f'Branch batch coverage/order mismatch: {name}')
        a.require(all(r['label'] == records[r['source_id']]['label'] and
                      r['generator'] == records[r['source_id']]['generator'] for r in batch['rows']), 'Branch labels changed')
        a.require(np.array_equal(batch['y'], [r['label'] for r in batch['rows']]), 'Branch target labels changed')
    rows = c.read_csv(out / 'predictions.csv')
    grouped = defaultdict(list)
    for row in rows:
        a.require(int(row['run_seed']) == seed and row['arm'] == arm, 'Prediction arm/seed changed')
        grouped[row['model'], int(row['gate_seed']), row['partition']].append(row)
    a.require(set(grouped) == {(n, s, p) for n, s in INSTANCES for p in ('selection', 'development')}, 'Model coverage mismatch')
    tuning = bundle['raw_batches']['tuning']
    candidates = []
    for weight in np.arange(11) / 10:
        z = weight * np.asarray(tuning['zg']) + (1 - weight) * np.asarray(tuning['zl'])
        scores = []
        for condition in c.TUNE:
            mask = np.array([r['condition'] == condition for r in tuning['rows']])
            scores.append(roc_auc_score(np.asarray(tuning['y'])[mask], z[mask]))
        candidates.append((float(weight), float(np.mean(scores))))
    best_score = max(score for _, score in candidates)
    fixed = min((weight for weight, score in candidates if np.isclose(score, best_score)), key=lambda w: (abs(w - .5), w))
    a.require(np.allclose(settings['fixed_search'], candidates, atol=1e-12, rtol=0), 'Fixed fusion search changed')
    for name, init in INSTANCES:
        spec = settings['settings'][f'{name}/{init}']
        a.require(spec['fixed_alpha'] == fixed, 'Fixed fusion weight was not selected on tuning only')
        tune_alpha = np.asarray(spec['tuning_alpha'])
        a.require(np.allclose(tune_alpha, alpha(name, init, tuning, settings), atol=2e-6, rtol=0), 'Tuning gate reconstruction failed')
        tz = tune_alpha * tuning['zg'] + (1 - tune_alpha) * tuning['zl']
        a.require(abs(threshold(tuning['y'], tz) - spec['threshold']) < 1e-12, 'Threshold not selected on tuning only')
        if name.endswith('gate'):
            entry = settings['gates'][f'{name}/{init}']
            best, best_epoch = float('inf'), 0
            for epoch in entry['history']:
                if epoch['validation_bce'] < best - 1e-6:
                    best, best_epoch = epoch['validation_bce'], epoch['epoch']
            a.require(best_epoch == entry['artifact']['best_epoch'], 'Gate epoch selection changed')
        for part in ('selection', 'development'):
            cell = grouped[name, init, part]
            batch = bundle['raw_batches'][part]
            expected = [(r['condition'], r['source_id']) for r in batch['rows']]
            a.require([(r['condition'], r['source_id']) for r in cell] == expected, 'Duplicate/missing prediction')
            weight = np.asarray([float(r['alpha_global']) for r in cell])
            a.require(np.allclose(weight, alpha(name, init, batch, settings), atol=2e-6, rtol=0), 'Gate/fusion reconstruction failed')
            z = weight * batch['zg'] + (1 - weight) * batch['zl']
            for i, row in enumerate(cell):
                source = records[row['source_id']]
                a.require(int(row['label']) == source['label'] and row['generator'] == source['generator'], 'Prediction labels changed')
                a.require(abs(float(row['logit']) - z[i]) < 1e-12, 'Fused logit mismatch')
                a.require(abs(float(row['probability_fake']) - expit(z[i])) < 1e-14, 'Probability mismatch')
                a.require(float(row['threshold']) == spec['threshold'], 'Threshold changed across reporting conditions')
    computed = metrics(rows)
    saved = c.read_csv(out / 'metrics.csv')
    a.require(len(saved) == len(computed) == 120, 'Metric coverage mismatch')
    for x, y in zip(saved, computed):
        for key, value in y.items():
            if isinstance(value, (float, int)):
                a.require(abs(float(x[key]) - value) < 1e-12, f'Metric mismatch: {key}')
            else:
                a.require(x[key] == value, f'Metric identity mismatch: {key}')
    return dict(status='passed', run_seed=seed, arm=arm, predictions=len(rows), metric_cells=len(computed),
                checks='hashes, isolation, coverage, NumPy gate reconstruction, tuning thresholds, metrics')


def summarize():
    values = defaultdict(list)
    checks = []
    for seed in seeds.SEEDS:
        for arm in a.ARMS:
            checks.append(verify_run(seed, arm))
            for row in c.read_csv(a.OUT / f'seed{seeds.number(seed)}' / arm / 'metrics.csv'):
                for metric in METRICS:
                    values[arm, row['model'], row['generator'], row['condition'], metric, seed].append(float(row[metric]))
    # Gate initializations are averaged within a run, never pooled as nine seeds.
    per_seed = {}
    for key, vv in values.items():
        a.require(len(vv) == (3 if key[1].endswith('gate') else 1), 'Initialization coverage mismatch')
        per_seed[key] = float(np.mean(vv))
    summary = []
    for arm, model, generator, condition, metric in sorted({key[:-1] for key in per_seed}):
        vv = [per_seed[arm, model, generator, condition, metric, seed] for seed in seeds.SEEDS]
        summary.append(dict(arm=arm, model=model, generator=generator, condition=condition, metric=metric,
                            seed1=vv[0], seed2=vv[1], seed3=vv[2], mean=float(np.mean(vv)), sd=float(np.std(vv, ddof=1))))
    contrasts = []
    comparisons = [('matched_local_minus_unmatched_local', 'matched', 'local', 'unmatched', 'local'),
                   ('matched_average_minus_matched_global', 'matched', 'average', 'matched', 'global'),
                   ('matched_quality_minus_matched_score', 'matched', 'quality_gate', 'matched', 'score_gate'),
                   ('matched_quality_minus_matched_average', 'matched', 'quality_gate', 'matched', 'average')]
    for label, left_arm, left_model, right_arm, right_model in comparisons:
        for generator in ('SD15', 'ADM', 'BigGAN'):
            for condition in c.CONDITIONS:
                for metric in METRICS:
                    vv = [per_seed[left_arm, left_model, generator, condition, metric, seed] -
                          per_seed[right_arm, right_model, generator, condition, metric, seed] for seed in seeds.SEEDS]
                    contrasts.append(dict(contrast=label, generator=generator, condition=condition, metric=metric,
                                          seed1=vv[0], seed2=vv[1], seed3=vv[2], mean=float(np.mean(vv)), sd=float(np.std(vv, ddof=1))))
    a.write_csv(a.OUT / 'metrics_mean_sd.csv', summary)
    a.write_csv(a.OUT / 'paired_differences.csv', contrasts)
    result = dict(status='passed', runs=checks, summary_rows=len(summary), contrast_rows=len(contrasts),
                  interpretation='Development only; SD describes three splits, not a population confidence interval. Positive FPR differences are worse.')
    a.write_json(a.OUT / 'verification.json', result)
    print(result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=seeds.SEEDS)
    parser.add_argument('--arm', choices=a.ARMS)
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    if args.summary:
        with a.run_lock():
            summarize()
    elif args.seed and args.arm:
        print(verify_run(args.seed, args.arm))
    else:
        parser.error('Use --seed and --arm, or --summary')
