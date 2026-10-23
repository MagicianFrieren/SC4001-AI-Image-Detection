"""Verify gate artifacts and compute paired, initialization-averaged contrasts."""
from collections import Counter
import csv
import json

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

import gate_models as gates
import prepare_data as data
import run_pilot as pilot
from run_round2 import dump_csv, read_manifest, write_json
from run_round4 import OUT, ROOT, PROFILES, TRAIN_CONDITIONS, DEV_CONDITIONS
from verify_round2 import independent_threshold


def read_csv(path):
    with path.open(encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def feature_matrix(rows):
    zg, zl = [np.array([float(r[k]) for r in rows]) for k in ['global_logit', 'local_logit']]
    pg, pl = expit(zg), expit(zl)
    def entropy(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -p * np.log(p) - (1 - p) * np.log1p(-p)
    q = np.array([[float(r[k]) for k in ['q_laplacian', 'q_high_frequency', 'q_block']] for r in rows])
    x = np.column_stack([pg, pl, abs(pg - pl), (entropy(pg) + entropy(pl)) / 2,
                         np.log1p([float(r['variance']) for r in rows]), q])
    return x, zg, zl


def numpy_gate(artifact, x):
    scaled = ((x - artifact['mean']) / artifact['scale']).astype(np.float32)
    if artifact['kind'] == 'score_gate':
        scaled[:, 5:] = 0
    state = {k: np.asarray(v, dtype=np.float32) for k, v in artifact['state'].items()}
    assert state['hidden.weight'].shape == (32, 8) and state['output.weight'].shape == (1, 32)
    assert sum(v.size for v in state.values()) == 321 and all(np.isfinite(v).all() for v in state.values())
    hidden = np.maximum(scaled @ state['hidden.weight'].T + state['hidden.bias'], 0)
    return expit((hidden @ state['output.weight'].T + state['output.bias']).ravel()).astype(float)


def main():
    meta = json.loads((OUT / 'run_metadata.json').read_text())
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assignments = json.loads((OUT / 'assignments.json').read_text())
    models = json.loads((OUT / 'gate_models.json').read_text())
    settings = json.loads((OUT / 'settings.json').read_text())
    inputs = read_csv(OUT / 'branch_inputs.csv')
    predictions = read_csv(OUT / 'predictions.csv')
    validation = read_csv(OUT / 'validation_predictions.csv')
    metrics = read_csv(OUT / 'metrics.csv')
    curves = read_csv(OUT / 'training_curves.csv')
    search = read_csv(OUT / 'weight_search.csv')
    records = read_manifest(ROOT / 'results/round2/manifest.jsonl')
    lookup = {r['source_id']: i for i, r in enumerate(records)}
    assert meta['protocol_sha256'] == data.sha256(OUT / 'protocol.json')
    assert meta['assignments_sha256'] == data.sha256(OUT / 'assignments.json')
    assert protocol['manifest_sha256'] == data.sha256(ROOT / 'results/round2/manifest.jsonl')
    for name, checksum in meta['code_sha256'].items():
        assert checksum == data.sha256(ROOT / 'scripts' / name)
    for name, checksum in meta['feature_sha256'].items():
        assert checksum == data.sha256(ROOT / name)
    old = json.loads((ROOT / 'results/round2/assignments.json').read_text())
    sets = list(map(set, assignments.values()))
    assert all(not a & b for i, a in enumerate(sets) for b in sets[i + 1:])
    assert set(assignments['expert_train'] + assignments['gate_train']) == set(old['train']['mixed_sd15_biggan'])
    assert assignments['validation'] == old['validation']['mixed_sd15_biggan']
    assert assignments['development'] == old['development']
    for partition, expected in [('expert_train', {'Real': 200, 'SD15': 100, 'BigGAN': 100}),
                                ('gate_train', {'Real': 100, 'SD15': 50, 'BigGAN': 50}),
                                ('validation', {'Real': 100, 'SD15': 50, 'BigGAN': 50}),
                                ('development', {'Real': 100, 'SD15': 100, 'BigGAN': 100, 'ADM': 100})]:
        ids = assignments[partition]
        assert len(ids) == len(set(ids)) == sum(expected.values())
        assert Counter(records[lookup[i]]['generator'] for i in ids) == expected
    assert len(inputs) == 4800 and len(predictions) == 24000 and len(validation) == 12000 and len(metrics) == 180
    assert len(models) == 12 and len(search) == 22
    assert len({tuple(r[k] for k in ['profile', 'partition', 'condition', 'source_id']) for r in inputs}) == len(inputs)
    for rows in [predictions, validation]:
        assert len({tuple(r[k] for k in ['profile', 'model', 'seed', 'condition', 'source_id']) for r in rows}) == len(rows)
    input_index = {(r['profile'], r['partition'], r['condition'], r['source_id']): r for r in inputs}
    for row in inputs + predictions + validation:
        source = records[lookup[row['source_id']]]
        assert int(row['label']) == source['label'] and row['generator'] == source['generator']

    # Refit frozen expert heads using only the declared 400 originals, and verify
    # every exported branch input against its bound feature-cache row.
    experts = joblib.load(ROOT / 'artifacts/round4_experts.joblib')
    with np.load(ROOT / 'artifacts/round2_features.npz') as cache:
        feature = {k: cache[k] for k in cache.files if '/' in k}
    with np.load(ROOT / 'artifacts/round4_global_jpeg70.npz') as cache:
        feature.update({k: cache[k] for k in cache.files if '/' in k})
    train = [lookup[i] for i in assignments['expert_train']]
    labels = np.array([r['label'] for r in records])
    for profile in PROFILES:
        with np.load(ROOT / 'artifacts' / f'round3_local_{profile}_clean.npz') as cache:
            local_clean = cache['views'].mean(axis=1)
        for name, x in [('global', feature[f'{profile}/clean']), ('local', local_clean)]:
            refit = pilot.train_head(x[train], labels[train])
            assert np.allclose(refit.decision_function(x), experts[profile][name].decision_function(x), atol=1e-12, rtol=0)
        for condition in list(dict.fromkeys(TRAIN_CONDITIONS + DEV_CONDITIONS)):
            key = f'{profile}/{condition}'
            with np.load(ROOT / 'artifacts' / f'round3_local_{profile}_{condition}.npz') as cache:
                views = cache['views']
            zg = experts[profile]['global'].decision_function(feature[key])
            zl = experts[profile]['local'].decision_function(views.mean(axis=1))
            variance = experts[profile]['local'].decision_function(views.reshape(-1, 512)).reshape(-1, 4).var(axis=1)
            for r in [r for r in inputs if r['profile'] == profile and r['condition'] == condition]:
                i = lookup[r['source_id']]
                q = feature['quality/' + key][i]
                expected = [zg[i], zl[i], variance[i], *q]
                actual = [float(r[k]) for k in ['global_logit', 'local_logit', 'variance', 'q_laplacian', 'q_high_frequency', 'q_block']]
                assert np.allclose(actual, expected, atol=1e-12, rtol=0)

    for key, artifact in models.items():
        profile, kind, seed = key.split('/')
        rows = [r for r in inputs if r['profile'] == profile and r['partition'] == 'gate_train']
        assert len(rows) == 600
        x, _, _ = feature_matrix(rows)
        scale = x.std(axis=0)
        scale[scale == 0] = 1
        assert np.allclose(x.mean(axis=0), artifact['mean'], atol=1e-12, rtol=0)
        assert np.allclose(scale, artifact['scale'], atol=1e-12, rtol=0)
        history = [r for r in curves if r['profile'] == profile and r['model'] == kind and r['seed'] == seed]
        assert [int(r['epoch']) for r in history] == list(range(len(history)))
        best_loss, best_epoch = float('inf'), 0
        for r in history:
            if float(r['validation_bce']) < best_loss - 1e-6:
                best_loss, best_epoch = float(r['validation_bce']), int(r['epoch'])
        assert artifact['best_epoch'] == best_epoch and artifact['stopped_epoch'] == len(history) - 1
        assert np.isclose(best_loss, artifact['best_validation_bce'], atol=1e-12, rtol=0)
        assert len(history) - 1 == gates.MAX_EPOCHS or len(history) - 1 - best_epoch == gates.PATIENCE

    max_alpha_error = 0.
    weight_rows = []
    for profile in PROFILES:
        vrows = [r for r in inputs if r['profile'] == profile and r['partition'] == 'validation']
        _, zg, zl = feature_matrix(vrows)
        yy = np.array([int(r['label']) for r in vrows])
        scores = []
        for a in np.arange(11) / 10:
            value = []
            for c in TRAIN_CONDITIONS:
                for generator in ['SD15', 'BigGAN']:
                    mask = np.array([r['condition'] == c and r['generator'] in ('Real', generator) for r in vrows])
                    value.append(roc_auc_score(yy[mask], (a * zg + (1 - a) * zl)[mask]))
            value = float(np.mean(value))
            saved = next(r for r in search if r['profile'] == profile and float(r['alpha']) == a)
            assert np.isclose(value, float(saved['validation_macro_auroc']), atol=1e-12, rtol=0)
            scores.append((a, value))
        best = max(v for _, v in scores)
        selected = min([a for a, v in scores if np.isclose(v, best)], key=lambda a: (abs(a - .5), a))
        assert selected == settings[profile]['fixed_alpha']
        for model_id, config in settings[profile]['models'].items():
            kind, seed = model_id.split('/')
            for partition, source in [('validation', validation), ('development', predictions)]:
                rows = [r for r in source if r['profile'] == profile and r['model'] == kind and r['seed'] == seed]
                conditions = TRAIN_CONDITIONS if partition == 'validation' else DEV_CONDITIONS
                expected = {(c, i) for c in conditions for i in assignments[partition]}
                assert {(r['condition'], r['source_id']) for r in rows} == expected
                bound = [input_index[(profile, partition, r['condition'], r['source_id'])] for r in rows]
                x, zg, zl = feature_matrix(bound)
                saved_alpha = np.array([float(r['alpha_global']) for r in rows])
                if kind.endswith('_gate'):
                    artifact = models[f'{profile}/{model_id}']
                    alpha = numpy_gate(artifact, x)
                    if partition == 'validation':
                        y = np.array([int(r['label']) for r in rows])
                        z = alpha * zg + (1 - alpha) * zl
                        bce = float(np.mean(np.logaddexp(0, z) - y * z))
                        assert abs(bce - artifact['best_validation_bce']) < 1e-5
                else:
                    alpha = np.full(len(rows), {'global': 1., 'local': 0., 'average': .5, 'fixed': selected}[kind])
                max_alpha_error = max(max_alpha_error, float(abs(alpha - saved_alpha).max()))
                assert np.allclose(alpha, saved_alpha, atol=1e-6, rtol=0)
                p = np.array([float(r['probability_fake']) for r in rows])
                assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
                assert np.allclose(p, expit(saved_alpha * zg + (1 - saved_alpha) * zl), atol=1e-12, rtol=0)
                if partition == 'validation':
                    y = np.array([int(r['label']) for r in rows])
                    assert abs(independent_threshold(y, p) - config['threshold']) < 1e-12
                elif kind.endswith('_gate'):
                    for condition in DEV_CONDITIONS:
                        for generator in ['Real', 'SD15', 'BigGAN', 'ADM']:
                            mask = np.array([r['condition'] == condition and r['generator'] == generator for r in rows])
                            weight_rows.append(dict(profile=profile, model=kind, seed=seed, condition=condition, generator=generator,
                                                    mean_alpha_global=float(saved_alpha[mask].mean()), std_alpha_global=float(saved_alpha[mask].std())))
    dump_csv(OUT / 'gate_weight_summary.csv', weight_rows)

    rng = np.random.default_rng(20261022)
    draws = np.concatenate([rng.integers(0, 100, (2000, 100)), rng.integers(100, 200, (2000, 100))], axis=1)
    distributions, points = {}, {}
    for cell in metrics:
        rows = sorted([r for r in predictions if all(r[k] == cell[k] for k in ['profile', 'model', 'seed', 'condition'])
                       and r['generator'] in ('Real', cell['generator'])], key=lambda r: (int(r['label']), r['source_id']))
        assert len(rows) == int(cell['n']) == 200
        y = np.array([int(r['label']) for r in rows])
        assert np.array_equal(y, np.repeat([0, 1], 100))
        p = np.array([float(r['probability_fake']) for r in rows])
        alpha = np.array([float(r['alpha_global']) for r in rows])
        threshold = settings[cell['profile']]['models'][f"{cell['model']}/{cell['seed']}"]['threshold']
        assert threshold == float(cell['threshold'])
        wins = p[100:, None] - p[None, :100]
        auc = float(np.mean((wins > 0) + .5 * (wins == 0)))
        for k, v in {'auroc': auc, 'average_precision': average_precision_score(y, p),
                     'balanced_accuracy': balanced_accuracy_score(y, p >= threshold), 'real_false_positive_rate': np.mean(p[y == 0] >= threshold),
                     'mean_alpha_global': alpha.mean(), 'std_alpha_global': alpha.std()}.items():
            assert np.isclose(v, float(cell[k]), atol=1e-12, rtol=0), (cell, k)
        key = tuple(cell[k] for k in ['profile', 'model', 'seed', 'condition', 'generator'])
        ranks = rankdata(p[draws], axis=1)
        distributions[key] = (ranks[:, 100:].sum(axis=1) - 5050) / 10000
        points[key] = auc
    aggregated, mean_boot, mean_points = [], {}, {}
    kinds = ['global', 'local', 'average', 'fixed', 'score_gate', 'quality_gate']
    for profile in PROFILES:
        for kind in kinds:
            seeds = gates.SEEDS if kind.endswith('_gate') else [-1]
            for condition in DEV_CONDITIONS:
                for generator in ['SD15', 'BigGAN', 'ADM']:
                    keys = [(profile, kind, str(seed), condition, generator) for seed in seeds]
                    aucs = np.array([points[k] for k in keys])
                    boot = np.mean([distributions[k] for k in keys], axis=0)
                    lo, hi = np.percentile(boot, [2.5, 97.5])
                    key = (profile, kind, condition, generator)
                    mean_boot[key], mean_points[key] = boot, aucs.mean()
                    rows = [r for r in metrics if (r['profile'], r['model'], r['condition'], r['generator']) == key]
                    aggregated.append(dict(profile=profile, model=kind, condition=condition, generator=generator,
                                           initialization_count=len(seeds), mean_auroc=float(aucs.mean()),
                                           std_auroc=float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.,
                                           auc_ci_low=float(lo), auc_ci_high=float(hi),
                                           mean_balanced_accuracy=float(np.mean([float(r['balanced_accuracy']) for r in rows])),
                                           mean_real_false_positive_rate=float(np.mean([float(r['real_false_positive_rate']) for r in rows]))))
    dump_csv(OUT / 'aggregate_metrics.csv', aggregated)
    contrasts = []
    for profile in PROFILES:
        for condition in DEV_CONDITIONS:
            for generator in ['SD15', 'BigGAN', 'ADM']:
                for a, b in [('quality_gate', 'score_gate'), ('quality_gate', 'fixed'), ('score_gate', 'fixed'),
                             ('quality_gate', 'average'), ('score_gate', 'average')]:
                    ka, kb = (profile, a, condition, generator), (profile, b, condition, generator)
                    lo, hi = np.percentile(mean_boot[ka] - mean_boot[kb], [2.5, 97.5])
                    contrasts.append(dict(profile=profile, condition=condition, generator=generator, contrast=a + '_minus_' + b,
                                          mean_auroc_delta=float(mean_points[ka] - mean_points[kb]), paired_ci_low=float(lo), paired_ci_high=float(hi)))
    dump_csv(OUT / 'paired_contrasts.csv', contrasts)
    verification = {'status': 'passed', 'metric_cells': 180, 'gate_models': 12, 'expert_heads_refit': 4,
                    'paired_contrasts': len(contrasts), 'max_numpy_vs_torch_alpha_error': max_alpha_error,
                    'checks': ['Disjoint expert/gate/validation/development originals and original development pool',
                               'Refitted all four expert heads from declared 400 clean originals',
                               'Every branch logit, variance and quality proxy rebound to hashed feature cache',
                               'Scalers fitted only on gate training; early stopping selection checked against histories',
                               'All gate outputs reconstructed with NumPy from exported weights',
                               'Fixed alpha and all thresholds recomputed from validation only',
                               'Metrics and seed-mean paired bootstrap recomputed from predictions'],
                    'scope': 'Artifact verification plus expert refit; gate optimization not independently repeated'}
    write_json(OUT / 'verification.json', verification)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, profile in zip(axes, PROFILES):
        for kind, color, marker, style in [('average', '#aaaaaa', 'o', ':'), ('fixed', '#777777', 's', '--'),
                                           ('score_gate', '#444444', '^', '-'), ('quality_gate', '#111111', 'D', '-')]:
            values = [mean_points[(profile, kind, c, 'ADM')] for c in DEV_CONDITIONS]
            ax.plot(range(3), values, color=color, marker=marker, linestyle=style, label=kind)
        ax.set_title(profile + ' / ADM (unseen in training)')
        ax.set_xticks(range(3), ['Clean', 'JPEG50', 'Half-size'])
        ax.set_ylim(0, 1.02)
        ax.axhline(.5, color='#bbbbbb', linestyle=':', linewidth=1)
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('AUROC')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='lower center', frameon=False, fontsize=9)
    fig.suptitle('Round 4: score gate versus quality-aware gate')
    fig.text(.5, .12, 'Gate curves: mean AUROC across 3 initializations; one data split. Exploratory development.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .17, 1, .96])
    fig.savefig(OUT / 'round4_adm_auroc.png', dpi=180)
    lines = ['SC4001 J：第四轮普通门控／质量感知门控对照（2026-10-22）', '',
             'Material Passport: academic-research-suite / experiment-agent; run; round4_v1.',
             '状态：实验完成，模型与结果核验通过。', '',
             '专家训练400张：200真实+100 SD1.5+100 BigGAN，只用clean；门控训练200张：100真实+50 SD1.5+50 BigGAN。',
             '门控训练与专家训练按原图完全分开；原验证集200张及开发集400张不变。',
             '门控训练和验证各使用clean、JPEG70、缩放0.5三个视图，因此各600条输入；开发评估clean、JPEG50、缩放0.5。',
             '缩放是门控已见条件；JPEG50强度未用于门控训练；ADM从不参与训练或模型选择。',
             '两种门控共享分支概率、概率差、预测熵和局部视图logit波动。质量门控额外获得3个质量代理。',
             '两者均为8→32→1网络、321参数，普通门控最后3个标准化输入置零；初始化、训练规则相同。',
             '3个初始化只衡量门控优化波动，专家及数据划分保持不变。报告的是各次AUROC均值，不是预测集成。',
             '', 'ADM AUROC均值（普通／质量门控括号内为3次初始化的样本标准差）：']
    for p in PROFILES:
        lines.append(f'  {p}: fixed alpha={settings[p]["fixed_alpha"]:.1f}')
        for c in DEV_CONDITIONS:
            rows = [next(r for r in aggregated if (r['profile'], r['model'], r['condition'], r['generator']) == (p, m, c, 'ADM'))
                    for m in ['average', 'fixed', 'score_gate', 'quality_gate']]
            lines.append(f"    {c:8}: Avg={rows[0]['mean_auroc']:.4f}, Fixed={rows[1]['mean_auroc']:.4f}, "
                         f"Score={rows[2]['mean_auroc']:.4f} ({rows[2]['std_auroc']:.4f}), Quality={rows[3]['mean_auroc']:.4f} ({rows[3]['std_auroc']:.4f})")
    lines += ['', 'ADM质量门控相对普通门控的成对差值区间（条件于当前专家、三个门控及开发图像）：']
    for row in contrasts:
        if row['generator'] == 'ADM' and row['contrast'] == 'quality_gate_minus_score_gate':
            lines.append(f"  {row['profile']:10} {row['condition']:8}: {row['mean_auroc_delta']:+.4f} [{row['paired_ci_low']:+.4f}, {row['paired_ci_high']:+.4f}]")
    lines += ['', '训练选择：']
    for key, artifact in models.items():
        lines.append(f"  {key}: best epoch={artifact['best_epoch']}, validation BCE={artifact['best_validation_bce']:.4f}")
    lines += ['', f'实验运行约{meta["run_seconds"]:.1f}秒，包括新JPEG70特征和缓存复用。',
              '180个指标单元、12个门控、4个专家头复算/重拟合通过；所有门控权重另用NumPy重建检查。', '', '解释边界：']
    lines += ['  - ' + item for item in protocol['limits']]
    lines += ['  - 区间未校正多重比较，不包含重新抽样训练集或重新训练专家的不确定性。',
              '  - 本轮专家仅用400张图，阈值使用混合退化验证；不能与第三轮直接归因比较。',
              '', '详见aggregate_metrics.csv、paired_contrasts.csv、gate_weight_summary.csv、training_curves.csv。',
              '运行：./.venv/Scripts/python.exe -u scripts/run_round4.py',
              '核验：./.venv/Scripts/python.exe scripts/verify_round4.py']
    (OUT / 'summary_zh.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
