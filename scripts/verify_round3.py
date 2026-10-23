"""Independent saved-prediction checks and descriptive paired analysis for round 3."""
from collections import Counter
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

import prepare_data as data
from run_round2 import read_manifest, write_json, dump_csv
from run_round3 import OUT, ROOT, PROFILES, CONDITIONS, MODELS
from verify_round2 import independent_threshold


def read_csv(path):
    with path.open(encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def main():
    metrics = read_csv(OUT / 'metrics.csv')
    predictions = read_csv(OUT / 'predictions.csv')
    validation = read_csv(OUT / 'validation_predictions.csv')
    overlaps = read_csv(OUT / 'error_overlap.csv')
    search = read_csv(OUT / 'weight_search.csv')
    settings = json.loads((OUT / 'settings.json').read_text())
    metadata = json.loads((OUT / 'run_metadata.json').read_text())
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assignments = json.loads((OUT / 'assignments.json').read_text())
    manifest = read_manifest(ROOT / 'results/round2/manifest.jsonl')
    by_id = {r['source_id']: r for r in manifest}
    assert metadata['manifest_sha256'] == protocol['manifest_sha256'] == data.sha256(ROOT / 'results/round2/manifest.jsonl')
    assert metadata['protocol_sha256'] == data.sha256(OUT / 'protocol.json')
    assert assignments == json.loads((ROOT / 'results/round2/assignments.json').read_text())
    assert protocol['assignments_sha256'] == data.sha256(ROOT / 'results/round2/assignments.json')
    for name, checksum in metadata['code_sha256'].items():
        assert checksum == data.sha256(ROOT / 'scripts' / name)
    assert metadata['global_cache_sha256'] == data.sha256(ROOT / 'artifacts/round2_features.npz')
    for variant in assignments['train']:
        train, val, dev = [set(assignments[s][variant] if s != 'development' else assignments[s])
                           for s in ['train', 'validation', 'development']]
        assert len(train) == 600 and len(val) == 200 and len(dev) == 400
        assert not (train & val or train & dev or val & dev)
        for split, n in [('train', 300), ('validation', 100)]:
            ids = assignments[split][variant]
            assert len(ids) == len(set(ids))
            assert all(by_id[i]['split'] == split for i in ids)
            counts = Counter(by_id[i]['generator'] for i in ids)
            expected = {'Real': n, 'SD15': n} if variant == 'single_sd15' else {'Real': n, 'SD15': n // 2, 'BigGAN': n // 2}
            assert counts == expected
    assert len(metrics) == 144 and len(predictions) == 4800 and len(validation) == 800
    assert len(overlaps) == 36 and len(search) == 44
    assert len({(r['profile'], r['training'], r['condition'], r['source_id']) for r in predictions}) == 4800
    assert len({(r['profile'], r['training'], r['source_id']) for r in validation}) == 800
    assert len({(r['profile'], r['training'], r['condition'], r['generator'], r['model']) for r in metrics}) == 144
    for row in validation + predictions:
        source = by_id[row['source_id']]
        assert int(row['label']) == source['label'] and row['generator'] == source['generator']
        assert np.isfinite(float(row['global_logit'])) and np.isfinite(float(row['local_logit']))

    for key, setting in settings.items():
        profile, variant = key.split('/')
        rows = [r for r in validation if r['profile'] == profile and r['training'] == variant]
        assert {r['source_id'] for r in rows} == set(assignments['validation'][variant])
        y = np.array([int(r['label']) for r in rows])
        zg = np.array([float(r['global_logit']) for r in rows])
        zl = np.array([float(r['local_logit']) for r in rows])
        generators = sorted({r['generator'] for r in rows} - {'Real'})
        candidate_scores = []
        for a in protocol['alpha_grid']:
            score = a * zg + (1 - a) * zl
            aucs = []
            for generator in generators:
                mask = np.array([r['generator'] in ('Real', generator) for r in rows])
                aucs.append(roc_auc_score(y[mask], score[mask]))
            auc = float(np.mean(aucs))
            saved = [r for r in search if r['profile'] == profile and r['training'] == variant and float(r['alpha']) == a]
            assert len(saved) == 1 and np.isclose(float(saved[0]['validation_macro_auroc']), auc, rtol=0, atol=1e-12)
            candidate_scores.append((a, auc))
        best = max(v for _, v in candidate_scores)
        selected = sorted([a for a, v in candidate_scores if np.isclose(v, best)], key=lambda a: (abs(a - .5), a))[0]
        assert selected == setting['selected_alpha']
        for model, a in [('global', 1.), ('local', 0.), ('average', .5), ('fixed', selected)]:
            t = independent_threshold(y, expit(a * zg + (1 - a) * zl))
            assert np.isclose(t, setting['thresholds'][model], rtol=0, atol=1e-12)

    old_predictions = read_csv(ROOT / 'results/round2/predictions.csv')
    old = {(r['model'], r['condition'], r['source_id']): float(r['probability_fake'])
           for r in old_predictions if r['model'].startswith('clip/')}
    max_global_error = 0.
    for row in predictions:
        setting = settings[f"{row['profile']}/{row['training']}"]
        zg, zl = float(row['global_logit']), float(row['local_logit'])
        assert np.isfinite(float(row['local_view_logit_variance'])) and float(row['local_view_logit_variance']) >= 0
        for model, a in [('global', 1.), ('local', 0.), ('average', .5), ('fixed', setting['selected_alpha'])]:
            value = float(row[model + '_probability'])
            assert np.isfinite(value) and 0 <= value <= 1
            assert np.isclose(value, expit(a * zg + (1 - a) * zl), rtol=0, atol=1e-12)
        prior = old[(f"clip/{row['profile']}/{row['training']}", row['condition'], row['source_id'])]
        max_global_error = max(max_global_error, abs(float(row['global_probability']) - prior))
    assert max_global_error < 1e-12

    rng = np.random.default_rng(20261022)
    bootstrap_ids = np.concatenate([rng.integers(0, 100, (2000, 100)), rng.integers(100, 200, (2000, 100))], axis=1)
    distributions, points, intervals = {}, {}, []
    for cell in metrics:
        rows = sorted([r for r in predictions if all(r[k] == cell[k] for k in ['profile', 'training', 'condition'])
                       and r['generator'] in ('Real', cell['generator'])], key=lambda r: (int(r['label']), r['source_id']))
        expected = {i for i in assignments['development'] if by_id[i]['generator'] in ('Real', cell['generator'])}
        assert {r['source_id'] for r in rows} == expected and len(rows) == int(cell['n']) == 200
        y = np.array([int(r['label']) for r in rows])
        assert np.array_equal(y, np.repeat([0, 1], 100))
        score = np.array([float(r[cell['model'] + '_probability']) for r in rows])
        setting = settings[f"{cell['profile']}/{cell['training']}"]
        expected_alpha = {'global': 1., 'local': 0., 'average': .5, 'fixed': setting['selected_alpha']}[cell['model']]
        assert float(cell['alpha']) == expected_alpha
        threshold = setting['thresholds'][cell['model']]
        assert np.isclose(float(cell['threshold']), threshold, rtol=0, atol=1e-12)
        wins = score[100:, None] - score[None, :100]
        auc = float(np.mean((wins > 0) + .5 * (wins == 0)))
        for field, value in {'auroc': auc, 'average_precision': average_precision_score(y, score),
                             'balanced_accuracy': balanced_accuracy_score(y, score >= threshold),
                             'real_false_positive_rate': np.mean(score[y == 0] >= threshold)}.items():
            assert np.isclose(value, float(cell[field]), rtol=0, atol=1e-12), (cell, field)
        ranks = rankdata(score[bootstrap_ids], axis=1)
        aucs = (ranks[:, 100:].sum(axis=1) - 5050) / 10000
        key = tuple(cell[k] for k in ['profile', 'training', 'condition', 'generator', 'model'])
        distributions[key], points[key] = aucs, auc
        lo, hi = np.percentile(aucs, [2.5, 97.5])
        intervals.append(dict(cell, auc_ci_low=float(lo), auc_ci_high=float(hi)))
    dump_csv(OUT / 'metrics_with_intervals.csv', intervals)

    for cell in overlaps:
        rows = [r for r in predictions if all(r[k] == cell[k] for k in ['profile', 'training', 'condition'])
                and r['generator'] in ('Real', cell['generator'])]
        thresholds = settings[f"{cell['profile']}/{cell['training']}"]['thresholds']
        y = np.array([int(r['label']) for r in rows])
        g, l = [np.array([float(r[m + '_probability']) for r in rows]) for m in ['global', 'local']]
        gc, lc = (g >= thresholds['global']) == y, (l >= thresholds['local']) == y
        counts = {'both_correct': np.sum(gc & lc), 'global_only_correct': np.sum(gc & ~lc),
                  'local_only_correct': np.sum(~gc & lc), 'both_wrong': np.sum(~gc & ~lc)}
        assert sum(counts.values()) == int(cell['n']) == 200
        assert all(int(cell[k]) == v for k, v in counts.items())
        assert np.isclose(float(cell['score_correlation']), np.corrcoef(g, l)[0, 1], atol=1e-12, rtol=0)

    contrasts = []
    for profile in PROFILES:
        for variant in assignments['train']:
            for condition in CONDITIONS:
                for generator in ['SD15', 'BigGAN', 'ADM']:
                    for a, b in [('local', 'global'), ('average', 'global'), ('fixed', 'global'), ('fixed', 'average'), ('fixed', 'local')]:
                        prefix = (profile, variant, condition, generator)
                        ka, kb = prefix + (a,), prefix + (b,)
                        lo, hi = np.percentile(distributions[ka] - distributions[kb], [2.5, 97.5])
                        contrasts.append(dict(profile=profile, training=variant, condition=condition, generator=generator,
                                              contrast=f'{a}_minus_{b}', auroc_delta=points[ka] - points[kb],
                                              paired_ci_low=float(lo), paired_ci_high=float(hi)))
    dump_csv(OUT / 'paired_contrasts.csv', contrasts)
    cases = []
    for row in predictions:
        if row['training'] != 'mixed_sd15_biggan' or row['generator'] not in ('Real', 'ADM'):
            continue
        setting = settings[f"{row['profile']}/{row['training']}"]
        gc = (float(row['global_probability']) >= setting['thresholds']['global']) == bool(int(row['label']))
        lc = (float(row['local_probability']) >= setting['thresholds']['local']) == bool(int(row['label']))
        if gc != lc:
            cases.append(dict(row, category='local_rescues_global' if lc else 'global_rescues_local',
                              image_path=str(ROOT / by_id[row['source_id']]['path'])))
    if cases:
        dump_csv(OUT / 'disagreement_cases.csv', cases)
    verification = {'status': 'passed', 'verified_metric_cells': len(metrics), 'verified_overlap_cells': len(overlaps),
                    'verified_alpha_searches': len(settings), 'paired_contrasts': len(contrasts),
                    'global_prediction_max_error_vs_round2': max_global_error,
                    'checks': ['Manifest/code/cache identity and disjoint original-level partitions',
                               'All alpha searches and 16 thresholds recomputed from validation only',
                               'Every saved fusion probability reconstructed from branch logits',
                               'All development metrics recomputed; AUROC with direct pairwise comparisons',
                               'All error overlap counts and correlations recomputed',
                               'Global baseline compared against round two'],
                    'scope': 'Saved-artifact verification, not independent end-to-end model reproduction'}
    write_json(OUT / 'verification.json', verification)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    variants = [('global', '#222222', 'o', '-'), ('local', '#999999', 's', '-'),
                ('average', '#666666', '^', '--'), ('fixed', '#222222', 'x', ':')]
    for line, profile in zip(axes, PROFILES):
        for ax, generator in zip(line, ['SD15', 'BigGAN', 'ADM']):
            for model, color, marker, style in variants:
                values = [points[(profile, 'mixed_sd15_biggan', c, generator, model)] for c in CONDITIONS]
                ax.plot(range(3), values, color=color, marker=marker, linestyle=style, label=model)
            ax.set_title(f'{profile} / {generator}' + (' (unseen)' if generator == 'ADM' else ''))
            ax.set_xticks(range(3), ['Clean', 'JPEG50', 'Half-size'])
            ax.set_ylim(0, 1.02)
            ax.axhline(.5, color='#bbbbbb', linestyle=':', linewidth=1)
            ax.grid(axis='y', alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
        line[0].set_ylabel('AUROC')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='lower center', frameon=False)
    fig.suptitle('Round 3: global/local fusion — mixed SD1.5 + BigGAN training')
    fig.text(.5, .065, 'Fixed weights and thresholds selected on clean validation only. Exploratory development; one training subset.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .09, 1, .96])
    fig.savefig(OUT / 'round3_auroc.png', dpi=180)

    lines = ['SC4001 J：第三轮双分支试验（2026-10-22）', '',
             'Material Passport: academic-research-suite / experiment-agent; run; round3_v1.',
             '状态：实验完成，保存的预测和指标已复核；未做独立端到端复现。', '',
             '本轮问题：局部分支是否提供互补信息？平均／固定权重融合是否能利用这些信息？',
             '全局分支：沿用第二轮冻结CLIP ViT-B/32整图特征和线性分类头。',
             '局部分支：224图像切为16个56x56块，按4个固定排列打乱，平均4个归一化特征，再训练同样的线性头。',
             '同一个排列用于所有图片及退化条件，不使用标签选择排列。先施加退化，再打乱。',
             '融合在logit上进行；alpha表示全局分支权重。平均融合alpha=0.5。',
             '固定权重从0到1、步长0.1，仅按clean验证集的生成器宏平均AUROC选择。',
             '每个模型再用clean验证集选择一个阈值，固定用于所有开发条件。',
             '每个训练方案仍为300真实+300生成；混合方案中SD1.5、BigGAN各150张。',
             '开发集及所有划分与第二轮相同；ADM不参与训练、权重或阈值选择。', '',
             '验证集选择的全局权重：']
    for key, setting in settings.items():
        lines.append(f'  {key}: alpha={setting["selected_alpha"]:.1f}')
    lines += ['', '混合训练的ADM AUROC（G=全局，L=局部，Avg=平均，Fixed=验证集固定权重）：']
    for profile in PROFILES:
        for c in CONDITIONS:
            values = [points[(profile, 'mixed_sd15_biggan', c, 'ADM', m)] for m in MODELS]
            lines.append(f'  {profile:10} {c:8}: G={values[0]:.4f}, L={values[1]:.4f}, Avg={values[2]:.4f}, Fixed={values[3]:.4f}')
    lines += ['', '混合训练ADM固定融合相对全局分支的差值（成对95%区间）：']
    for row in contrasts:
        if row['training'] == 'mixed_sd15_biggan' and row['generator'] == 'ADM' and row['contrast'] == 'fixed_minus_global':
            lines.append(f"  {row['profile']:10} {row['condition']:8}: {row['auroc_delta']:+.4f} [{row['paired_ci_low']:+.4f}, {row['paired_ci_high']:+.4f}]")
    lines += ['', '错误互补：每行200张图；分类阈值各自在clean验证集选择，不随条件改变。']
    for row in overlaps:
        if row['training'] == 'mixed_sd15_biggan' and row['generator'] == 'ADM':
            lines.append(f"  {row['profile']:10} {row['condition']:8}: 都对={row['both_correct']}, 仅G对={row['global_only_correct']}, 仅L对={row['local_only_correct']}, 都错={row['both_wrong']}")
    lines += ['', f'144个指标单元、36个错误重叠单元、4次权重搜索复核通过。全局预测与第二轮最大差异：{max_global_error:.3g}。',
              f'实验运行时间：{metadata["run_seconds"]:.1f}秒；峰值PyTorch显存分配：{metadata["peak_cuda_memory_mb"]} MB。',
              '', '限制与待检查问题：',
              '这是初期探索：单一训练子集、固定四视图，尚未训练普通门控或质量门控。',
              '打乱视图不能等同于纯纹理，它同时破坏布局并引入人工边界。',
              '56像素图块与ViT-B/32的32像素token不对齐；编码器、训练方法、视图数都不同于SFLD原设置，不能称为论文复现。',
              '数据来源、格式、分辨率历史和内容偏差仍然存在。',
              '每个区间使用2000次按类别分层的成对bootstrap，只反映固定模型的开发集抽样不确定性，未校正多重比较。',
              '错误互补依赖验证阈值；看到某分支“补救”另一分支不代表可在未知图片上自动判断应该选谁。',
              '这些开发结果不能作为最终盲测证据。',
              '', 'PatchShuffle方法出处：https://arxiv.org/abs/2502.17105',
              '运行：./.venv/Scripts/python.exe -u scripts/run_round3.py',
              '核验：./.venv/Scripts/python.exe scripts/verify_round3.py']
    (OUT / 'summary_zh.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
