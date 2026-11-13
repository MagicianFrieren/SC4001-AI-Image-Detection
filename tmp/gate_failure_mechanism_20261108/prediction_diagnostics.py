"""Read-only diagnosis of frozen multi-generator predictions; no fitting/search.

Constant mean-alpha and label-oracle scores below are explicitly diagnostic
counterfactuals, not trained/deployable candidates or evidence of new success.
"""
import sys
sys.dont_write_bytecode = True
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'output/analysis/multigen_gate_20261108/run_v1_codefix1'
OUT = ROOT / 'output/analysis/gate_failure_mechanism_20261108'
INPUTS = ['complete.json', 'predictions.csv.gz', 'metrics_seed_condition.csv', 'summary.json']
KEY = ['run_seed', 'scenario', 'heldout_generator', 'condition']
N_PAIR = 90000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_native(value):
    if isinstance(value, dict):
        return {k: json_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_native(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        # An empty pair cohort has undefined descriptive means, represented null.
        return None
    return value


def write_json(name, value):
    with (OUT / ('prediction_' + name)).open('w', encoding='utf-8') as f:
        json.dump(json_native(value), f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)


def write_csv(name, rows):
    if not rows:
        return
    with (OUT / ('prediction_' + name)).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)


def describe(a, prefix):
    a = np.asarray(a, dtype=float)
    return {prefix + '_mean': float(np.mean(a)), prefix + '_std': float(np.std(a)),
            **{prefix + '_' + name: float(v) for name, v in zip(['q05', 'q25', 'median', 'q75', 'q95'], np.quantile(a, [.05, .25, .5, .75, .95]))}}


def credit(z):
    # Rows are fake IDs, columns are real IDs; score direction remains fake-high.
    d = z[300:, None] - z[None, :300]
    return (d > 0).astype(float) + .5 * (d == 0)


def aggregate(rows, extra_keys=()):
    data = pd.DataFrame(rows)
    keys = KEY + list(extra_keys)
    numerical = [c for c in data.select_dtypes(include='number') if c not in keys + ['gate_seed']]
    seed = data.groupby(keys, sort=True, as_index=False)[numerical].mean()
    cell_keys = ['heldout_generator', 'condition'] + list(extra_keys)
    cell = seed.groupby(cell_keys, sort=True, as_index=False)[numerical].mean()
    return seed.to_dict('records'), cell.to_dict('records')


def cohort_means(mask, delta, zg, zl, aq, ar, zq, zr):
    fi, ri = np.nonzero(mask)
    if not len(fi):
        return {name: None for name in ['fake_adjustment', 'real_adjustment', 'pair_margin_change',
                'fake_alpha_quality', 'real_alpha_quality', 'fake_alpha_reference', 'real_alpha_reference',
                'fake_global_minus_local', 'real_global_minus_local', 'old_pair_margin', 'new_pair_margin']}
    fi = fi + 300
    values = {
        'fake_adjustment': delta[fi].mean(), 'real_adjustment': delta[ri].mean(),
        'pair_margin_change': (delta[fi] - delta[ri]).mean(),
        'fake_alpha_quality': aq[fi].mean(), 'real_alpha_quality': aq[ri].mean(),
        'fake_alpha_reference': ar[fi].mean(), 'real_alpha_reference': ar[ri].mean(),
        'fake_global_minus_local': (zg[fi] - zl[fi]).mean(),
        'real_global_minus_local': (zg[ri] - zl[ri]).mean(),
        'old_pair_margin': (zr[fi] - zr[ri]).mean(), 'new_pair_margin': (zq[fi] - zq[ri]).mean(),
    }
    return {k: float(v) for k, v in values.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    complete = json.loads((SOURCE / 'complete.json').read_text())
    hashes = {name: sha(SOURCE / name) for name in INPUTS}
    for name in INPUTS[1:]:
        assert hashes[name] == complete['outputs_sha256'][name], 'Frozen source changed: ' + name
    write_json('inputs.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'input_sha256': hashes,
        'script_sha256': sha(__file__), 'scope': 'Frozen mixed-source primary AUROC checkpoints only; no model training or candidate selection.',
        'rank_unit': 'All 300 fake x 300 real pairs in each cell; 0/0.5/1 credit for wrong/tied/correct ordering.',
        'aggregation': 'Gate initializations averaged within run seed first, then data seeds equally. No independence claim or new significance tests.',
        'constant_alpha_counterfactual': 'Replace each gate alpha with its mean across the 600 TARGET images in that cell. Target-dependent descriptive counterfactual, not a deployable selected model.',
        'label_oracle': 'Use true labels: fake score=max(global,local), real score=min(global,local). An unattainable-by-assumption label-oracle upper bound for all samplewise convex logit fusions on these pairs.',
        'example_selection': 'Illustrative JPEG pair: within the first deterministic eligible cell/gate, prefer the expert-disagreement category contributing most newly lost average-relative pair credit. Require average and score correct but quality wrong; use first sorted fake/real ID. Post-hoc explanatory choice, not a random sample or new model selection.',
    })
    frame = pd.read_csv(SOURCE / 'predictions.csv.gz')
    frame = frame[frame.scenario.str.startswith('mixed_')]
    frame = frame[((frame.model.isin(['global', 'local', 'average', 'fixed'])) & (frame.checkpoint == 'fixed')) |
                  ((frame.model.isin(['score_gate', 'quality_gate'])) & (frame.checkpoint == 'auroc'))]
    alpha_rows, expert_rows, rank_rows, constant_rows, decomp_rows = [], [], [], [], []
    examples, worst_reconstruction, worst_auc = [], 0., 0.
    for key, block in frame.groupby(KEY, sort=True):
        meta = dict(zip(KEY, key))
        def get(model, gate_seed=-1):
            d = block[(block.model == model) & (block.gate_seed == gate_seed)].sort_values(['label', 'source_id'])
            assert len(d) == 600 and len(d.source_id.unique()) == 600
            assert d.label.tolist() == [0] * 300 + [1] * 300
            return d.reset_index(drop=True)
        gg, ll, av, fx = [get(m) for m in ['global', 'local', 'average', 'fixed']]
        order = gg.source_id.tolist()
        assert all(x.source_id.tolist() == order for x in [ll, av, fx])
        y = gg.label.to_numpy(); zg = gg.logit.to_numpy(); zl = ll.logit.to_numpy(); za = av.logit.to_numpy()
        cg, cl, ca = credit(zg), credit(zl), credit(za)
        cats = {'both_experts_rank_correct': (cg == 1) & (cl == 1),
                'global_only_rank_correct': (cg == 1) & (cl == 0),
                'local_only_rank_correct': (cg == 0) & (cl == 1),
                'both_experts_rank_wrong': (cg == 0) & (cl == 0),
                'expert_rank_tie': (cg == .5) | (cl == .5)}
        assert sum(int(m.sum()) for m in cats.values()) == N_PAIR
        oracle = np.where(y == 1, np.maximum(zg, zl), np.minimum(zg, zl))
        pg = expit(zg) >= float(gg.threshold.iloc[0]); pl = expit(zl) >= float(ll.threshold.iloc[0])
        correct_g, correct_l = pg == y, pl == y
        er = dict(**meta, global_auroc=float(cg.mean()), local_auroc=float(cl.mean()),
                  average_auroc=float(ca.mean()), fixed_auroc=float(roc_auc_score(y, fx.logit)),
                  label_oracle_convex_upper_bound_auroc=float(roc_auc_score(y, oracle)),
                  global_threshold=float(gg.threshold.iloc[0]), local_threshold=float(ll.threshold.iloc[0]),
                  **{k + '_pair_fraction': float(mask.mean()) for k, mask in cats.items()})
        for label, label_name in [(0, 'real'), (1, 'fake')]:
            mask = y == label
            for method, d in [('global', gg), ('local', ll), ('average', av), ('fixed', fx)]:
                er[f'{method}_{label_name}_error_rate'] = float(np.mean((expit(d.logit.to_numpy())[mask] >= float(d.threshold.iloc[0])) != y[mask]))
            er[f'{label_name}_both_threshold_correct'] = float(np.mean((correct_g & correct_l)[mask]))
            er[f'{label_name}_global_only_threshold_correct'] = float(np.mean((correct_g & ~correct_l)[mask]))
            er[f'{label_name}_local_only_threshold_correct'] = float(np.mean((~correct_g & correct_l)[mask]))
            er[f'{label_name}_both_threshold_wrong'] = float(np.mean((~correct_g & ~correct_l)[mask]))
            er.update(describe(zg[mask], 'global_' + label_name + '_logit'))
            er.update(describe(zl[mask], 'local_' + label_name + '_logit'))
        expert_rows.append(er)
        for gate_seed in sorted(block[block.model == 'quality_gate'].gate_seed.unique()):
            q, s = get('quality_gate', gate_seed), get('score_gate', gate_seed)
            assert q.source_id.tolist() == s.source_id.tolist() == order
            aq, ass = q.alpha_global.to_numpy(), s.alpha_global.to_numpy()
            zq, zs = q.logit.to_numpy(), s.logit.to_numpy()
            worst_reconstruction = max(worst_reconstruction, float(np.max(abs(zq - (aq * zg + (1-aq) * zl)))),
                                       float(np.max(abs(zs - (ass * zg + (1-ass) * zl)))))
            cq, cs = credit(zq), credit(zs)
            for kind, d, alpha, z in [('quality_gate', q, aq, zq), ('score_gate', s, ass, zs)]:
                for label, label_name in [(0, 'real'), (1, 'fake')]:
                    mask = y == label
                    alpha_rows.append(dict(**meta, gate_seed=int(gate_seed), model=kind, label=label_name,
                       **describe(alpha[mask], 'alpha'), **describe(z[mask], 'fused_logit'),
                       fraction_alpha_below_point1=float(np.mean(alpha[mask] < .1)),
                       fraction_alpha_above_point9=float(np.mean(alpha[mask] > .9)),
                       threshold=float(d.threshold.iloc[0]),
                       threshold_error_rate=float(np.mean((expit(z[mask]) >= float(d.threshold.iloc[0])) != label))))
                mean_alpha = float(alpha.mean())
                z_const = mean_alpha * zg + (1-mean_alpha) * zl
                actual = float(roc_auc_score(y, z)); const = float(roc_auc_score(y, z_const))
                constant_rows.append(dict(**meta, gate_seed=int(gate_seed), model=kind,
                    diagnostic_target_mean_alpha=mean_alpha, actual_auroc=actual,
                    constant_mean_alpha_auroc=const, constant_minus_actual_auroc=const-actual,
                    constant_minus_average_auroc=const-float(ca.mean()),
                    label_oracle_convex_upper_bound_auroc=er['label_oracle_convex_upper_bound_auroc']))
            for ref_name, zr, ar, cr in [('average', za, np.full(600, .5), ca), ('score_gate', zs, ass, cs)]:
                change = cq - cr
                lost = np.maximum(-change, 0); gained = np.maximum(change, 0)
                damage = (cr == 1) & (cq == 0); repair = (cr == 0) & (cq == 1)
                row = dict(**meta, gate_seed=int(gate_seed), reference=ref_name,
                    quality_auroc=float(cq.mean()), reference_auroc=float(cr.mean()),
                    delta_auroc=float(change.mean()),
                    lost_pair_credit=float(lost.sum()), repaired_pair_credit=float(gained.sum()),
                    lost_pair_fraction=float(lost.mean()), repaired_pair_fraction=float(gained.mean()),
                    strict_new_wrong_pairs=int(damage.sum()), strict_repaired_pairs=int(repair.sum()),
                    pair_count=N_PAIR)
                for cat, mask in cats.items():
                    row[f'lost_credit_{cat}'] = float(lost[mask].sum())
                    row[f'repaired_credit_{cat}'] = float(gained[mask].sum())
                for old in [0., .5, 1.]:
                    for new in [0., .5, 1.]:
                        row[f'credit_{old:g}_to_{new:g}_count'] = int(np.sum((cr == old) & (cq == new)))
                rank_rows.append(row)
                worst_auc = max(worst_auc, abs(float(cq.mean()) - roc_auc_score(y, zq)))
                delta = zq - zr
                np.testing.assert_allclose(delta, (aq-ar)*(zg-zl), rtol=0, atol=1e-10)
                for label, mask in [('newly_wrong', damage), ('repaired', repair)]:
                    decomp_rows.append(dict(**meta, gate_seed=int(gate_seed), reference=ref_name,
                        pair_cohort=label, pair_count=int(mask.sum()),
                        **cohort_means(mask, delta, zg, zl, aq, ar, zq, zr)))
            if meta['condition'].startswith('jpeg'):
                disagreement = ['global_only_rank_correct', 'local_only_rank_correct']
                avg_loss = np.maximum(ca - cq, 0)
                dominant = max(disagreement, key=lambda name: avg_loss[cats[name]].sum())
                preferred = (ca == 1) & (cs == 1) & (cq == 0) & cats[dominant]
                fallback = (ca == 1) & (cs == 1) & (cq == 0)
                mask = preferred if preferred.any() else fallback
                if mask.any():
                    fi, ri = np.argwhere(mask)[0]; fi += 300
                    sample = dict(**meta, gate_seed=int(gate_seed), preferred_dominant_disagreement=bool(preferred.any()),
                                  dominant_expert_disagreement_category=dominant,
                                  selection='Illustrative, deliberately selected ranking-reversal pair; not a random sample or aggregate evidence.')
                    for name, idx in [('real', int(ri)), ('fake', int(fi))]:
                        sample[name] = dict(source_id=order[idx], label=int(y[idx]), global_logit=float(zg[idx]), local_logit=float(zl[idx]),
                            average_logit=float(za[idx]), score_gate_alpha=float(ass[idx]), score_gate_logit=float(zs[idx]),
                            quality_gate_alpha=float(aq[idx]), quality_gate_logit=float(zq[idx]))
                    examples.append(sample)
    assert worst_reconstruction < 1e-10 and worst_auc < 1e-12
    rank_seed, rank_cells = aggregate(rank_rows, ['reference'])
    const_seed, const_cells = aggregate(constant_rows, ['model'])
    alpha_seed, alpha_cells = aggregate(alpha_rows, ['model', 'label'])
    _, expert_cells = aggregate(expert_rows)
    decomp_seed, decomp_cells = aggregate(decomp_rows, ['reference', 'pair_cohort'])
    for name, rows in [('alpha_distributions.csv', alpha_rows), ('expert_complementarity.csv', expert_rows),
                       ('ranking_comparisons.csv', rank_rows), ('constant_alpha.csv', constant_rows),
                       ('pair_decomposition.csv', decomp_rows), ('ranking_seed_means.csv', rank_seed),
                       ('constant_seed_means.csv', const_seed), ('alpha_seed_means.csv', alpha_seed)]:
        write_csv(name, rows)
    # Retain exactly one deliberately selected representative of the dominant error type.
    chosen = next((e for e in examples if e['preferred_dominant_disagreement']), examples[0] if examples else None)
    write_json('illustrative_pair.json', chosen)
    summary = {'status': 'completed', 'inputs_unchanged': all(sha(SOURCE / n) == h for n, h in hashes.items()),
               'models_fitted': 0, 'final_test_accessed': False,
               'max_logit_reconstruction_error': worst_reconstruction,
               'max_pair_auc_identity_error': worst_auc,
               'expert_cells': expert_cells, 'alpha_cells': alpha_cells,
               'ranking_cells': rank_cells, 'constant_alpha_cells': const_cells,
               'pair_decomposition_cells': decomp_cells,
               'interpretation': 'Exact descriptive accounting on previously inspected development predictions. Counterfactual target means and true-label oracle are non-deployable diagnostics. Threshold errors do not bound ranking improvement. Data seeds and pair counts are not independent replications.'}
    assert summary['inputs_unchanged']
    write_json('summary.json', summary)
    print(json.dumps({'status': 'completed', 'rank_instance_rows': len(rank_rows), 'alpha_rows': len(alpha_rows),
                      'expert_rows': len(expert_rows), 'logit_error': worst_reconstruction,
                      'pair_auc_error': worst_auc}, indent=2), flush=True)


if __name__ == '__main__':
    main()
