"""Metrics, fixed four-domain macro, within-seed contrasts and cross-test comparisons."""
import sys
sys.dont_write_bytecode = True
from collections import defaultdict
import numpy as np
from scipy.special import expit
import round8_final_common as f


def task_groups(models):
    groups = defaultdict(list)
    for m in models:
        key = (m['cohort'], m['seed'], m['arm'], m['head'] if m['family'] == 'clip' else m['checkpoint'])
        groups[key].append(m)
    result = {}
    for number, (_, group) in enumerate(sorted(groups.items()), 1):
        for condition in f.CONDITIONS:
            result[f'task{number:02d}_{condition}'] = (group, condition)
    return result


def validate(values, models, condition, records):
    expected = {m['id']: m for m in models}
    seen = set(); grouped = defaultdict(list)
    for r in values:
        key = r['model_id'], r['source_id']
        f.require(key not in seen and key[0] in expected and key[1] in records, 'Duplicate or unapproved prediction ID')
        seen.add(key)
        original = records[key[1]]
        f.require(r['condition'] == condition and r['generator'] == original['generator'] and int(r['label']) == original['label'], 'Prediction metadata changed')
        z, p, threshold = (float(r[k]) for k in ('logit', 'probability_fake', 'threshold'))
        f.require(np.isfinite([z, p, threshold]).all() and abs(p - expit(z)) <= 1e-14, 'Nonfinite/inconsistent probability')
        f.require(threshold == expected[key[0]]['threshold'], 'Threshold changed')
        if r['alpha_global'] != '':
            f.require(np.isfinite(float(r['alpha_global'])) and 0 <= float(r['alpha_global']) <= 1, 'Invalid gate weight')
        grouped[key[0]].append(r)
    f.require(seen == {(m, i) for m in expected for i in records}, 'Incomplete prediction coverage')
    return grouped


def raw_metrics(values, models, condition, records):
    grouped = validate(values, models, condition, records)
    result = []
    for m in models:
        for generator in f.GENERATORS:
            cell = [r for r in grouped[m['id']] if r['generator'] in ('Real', generator)]
            y = np.array([int(r['label']) for r in cell])
            z = np.array([float(r['logit']) for r in cell])
            result.append(dict(model_id=m['id'], cohort=m['cohort'], arm=m['arm'], model=m['model'],
                seed=m['seed'], initialization=m['initialization'], generator=generator, condition=condition,
                n_real=int(sum(y == 0)), n_fake=int(sum(y == 1)), **f.c.points(y, z, m['threshold'])))
    return result


def stats(values):
    f.require(len(values) == 3 and np.isfinite(values).all(), 'Three finite seeds required')
    mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
    return dict(seed1=values[0], seed2=values[1], seed3=values[2], mean=mean, sd=sd,
                mean_plus_minus_sd=f'{mean:.6f} ± {sd:.6f}')


def summarize(raw, models):
    groups = defaultdict(list)
    for r in raw:
        groups[r['cohort'], r['arm'], r['model'], r['seed'], r['generator'], r['condition']].append(r)
    per_seed = []
    keys = ('cohort', 'arm', 'model', 'seed', 'generator', 'condition')
    for key, values in sorted(groups.items()):
        expected = {m['id'] for m in models if (m['cohort'], m['arm'], m['model'], m['seed']) == key[:4]}
        f.require(len(values) == len(expected) and {r['model_id'] for r in values} == expected, 'Initialization coverage changed')
        per_seed.append(dict(zip(keys, key), initializations=len(values), **{metric: float(np.mean([r[metric] for r in values])) for metric in f.METRICS}))
    lookup = {tuple(r[k] for k in keys): r for r in per_seed}
    configs = sorted({(m['cohort'], m['arm'], m['model']) for m in models})
    macro = []
    for cohort, arm, model in configs:
        for seed in f.seeds.SEEDS:
            for condition in f.CONDITIONS:
                cells = [lookup[cohort, arm, model, seed, g, condition] for g in f.MAIN]
                macro.append(dict(cohort=cohort, arm=arm, model=model, seed=seed, generator='main_macro', condition=condition,
                    initializations=cells[0]['initializations'], **{metric: float(np.mean([r[metric] for r in cells])) for metric in f.METRICS}))
    lookup.update({tuple(r[k] for k in keys): r for r in macro})
    combined, pairs = [], []
    domains = (*f.GENERATORS, 'main_macro')
    for cohort, arm, model in configs:
        for generator in domains:
            for condition in f.CONDITIONS:
                for metric in f.METRICS:
                    vv = [lookup[cohort, arm, model, s, generator, condition][metric] for s in f.seeds.SEEDS]
                    combined.append(dict(cohort=cohort, arm=arm, model=model, generator=generator, condition=condition, metric=metric, **stats(vv)))
    comparisons = [('W1-W0', ('round7', 'saved', 'W1'), ('round7', 'saved', 'global')),
        ('matched-unmatched local', ('round8_a', 'matched', 'local'), ('round8_a', 'unmatched', 'local')),
        ('matched-unmatched average', ('round8_a', 'matched', 'average'), ('round8_a', 'unmatched', 'average')),
        ('matched quality_gate-matched average', ('round8_a', 'matched', 'quality_gate'), ('round8_a', 'matched', 'average'))]
    for title, left, right in comparisons:
        for g in domains:
            for condition in f.CONDITIONS:
                for metric in f.METRICS:
                    vv = [lookup[*left, s, g, condition][metric] - lookup[*right, s, g, condition][metric] for s in f.seeds.SEEDS]
                    pairs.append(dict(contrast=title, generator=g, condition=condition, metric=metric, **stats(vv)))
    # Same exact Round 7 models/thresholds, new versus old test images. No bogus
    # Round 8A-versus-Round 7 pairing of merely similarly named refitted models.
    old_path = f.ROOT / 'results/round7/final_test/metrics_each_seed.csv'
    old_marker = f.read(f.ROOT / 'results/round7/final_test/complete.json')
    f.require(f.sha(old_path) == old_marker['files_sha256'][f.rel(old_path)], 'Old Round 7 metrics changed')
    historical = f.c.read_csv(old_path)
    old = {(r['model'], int(r['seed']), r['generator'], r['condition']): r for r in historical
           if r['profile'] == 'original' and r['calibration'] == 'raw' and
           ((r['stage'] == 'C') or (r['stage'] == 'D' and r['model'] == 'W1'))}
    cross = []
    for cohort, arm, model in configs:
        if cohort != 'round7':
            continue
        for g in domains:
            for condition in f.CONDITIONS:
                for metric in f.METRICS:
                    vv = []
                    for s in f.seeds.SEEDS:
                        before = float(np.mean([float(old[model, s, gen, condition][metric]) for gen in f.MAIN])) if g == 'main_macro' else float(old[model, s, g, condition][metric])
                        vv.append(lookup[cohort, arm, model, s, g, condition][metric] - before)
                    cross.append(dict(model=model, contrast='Round8_test-Round7_test_same_frozen_model', generator=g,
                        condition=condition, metric=metric, interpretation='Different test images; paired by fitted model/run seed only', **stats(vv)))
    return {'metrics_initializations.csv': raw, 'metrics_each_seed.csv': per_seed, 'main_macro_each_seed.csv': macro,
            'metrics_mean_sd.csv': combined, 'paired_differences.csv': pairs, 'cross_test_differences.csv': cross}


def markdown(tables):
    lines = ['# Round 8 final evaluation', '',
        'Original profile, raw scores, saved models and saved thresholds. No training or selection on this test.', '',
        'The test contains 500 images from each of six fake generators and one shared pool of 500 real images. There is no SD15 in-domain test.', '',
        'Main macro: equal mean of GLIDE, Midjourney, VQDM and Wukong metrics. These generator domains were inspected before; images are new relative to the repository exclusion pools and their hash neighbours. External pilot membership cannot be certified.', '',
        'Values are mean ± sample SD over three run seeds. Gate-initialization metrics are averaged within a seed first. The same real pool is reused across comparisons, not duplicated in a pooled metric.', '',
        'Real/fake format history, native-size and content confounds remain. Positive FPR differences are worse. Three-seed SD is not a confidence interval.', '']
    for generator in ('main_macro', *f.GENERATORS):
        lines += [f'## {generator}', '', '| Source / arm | Model | Condition | AUROC | AP | Balanced accuracy | Real FPR |', '|---|---|---|---:|---:|---:|---:|']
        cells = defaultdict(dict)
        for r in tables['metrics_mean_sd.csv']:
            if r['generator'] == generator:
                cells[r['cohort'], r['arm'], r['model'], r['condition']][r['metric']] = r['mean_plus_minus_sd']
        for (cohort, arm, model, condition), scores in sorted(cells.items()):
            lines.append(f'| {cohort}/{arm} | {model} | {condition} | ' + ' | '.join(scores[k] for k in f.METRICS) + ' |')
        lines.append('')
    lines += ['## Main-macro paired AUROC differences', '', '| Contrast | Condition | Difference |', '|---|---|---:|']
    for r in tables['paired_differences.csv']:
        if r['generator'] == 'main_macro' and r['metric'] == 'auroc':
            lines.append(f'| {r["contrast"]} | {r["condition"]} | {r["mean_plus_minus_sd"]} |')
    lines += ['', '## Same-model change from the Round 7 test', '',
        'These differences compare the same frozen model and threshold on disjoint image samples. They describe test-set variation, not a training improvement. Round 8A models have no earlier Round 7 test measurement.', '',
        '| Round 7 model | Condition | Main-macro AUROC difference |', '|---|---|---:|']
    for r in tables['cross_test_differences.csv']:
        if r['generator'] == 'main_macro' and r['metric'] == 'auroc':
            lines.append(f'| {r["model"]} | {r["condition"]} | {r["mean_plus_minus_sd"]} |')
    lines += ['', 'CSV tables include every individual seed, all four metrics and every requested contrast per generator and condition. No winner is selected or model changed based on these results.', '']
    return '\n'.join(lines).encode('utf-8')


def build(frozen, records):
    raw = []
    for task, (models, condition) in task_groups(frozen['models']).items():
        path = f.OUT / 'evaluation/predictions' / (task + '.csv')
        receipt = f.read(f.OUT / 'evaluation/receipts' / (task + '.json'))
        f.require(receipt['freeze_sha256'] == f.sha(f.FREEZE) and receipt['prediction_sha256'] == f.sha(path)
                  and receipt['source_ids'] == sorted(records) and receipt['model_ids'] == [m['id'] for m in models]
                  and receipt['condition'] == condition, 'Prediction receipt mismatch')
        raw.extend(raw_metrics(f.c.read_csv(path), models, condition, records))
    tables = summarize(raw, frozen['models'])
    payloads = {name: f.csv_bytes(values) for name, values in tables.items()}
    payloads['SUMMARY.md'] = markdown(tables)
    return payloads, tables
