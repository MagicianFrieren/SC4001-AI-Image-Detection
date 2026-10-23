"""Render matched-dose augmentation contrasts after round-six verification."""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_round6 import OUT, ROOT, PROFILES, CALIBRATIONS, EXPERT_MODES, DEV_CONDITIONS


def read_csv(path):
    with path.open(encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def main():
    verified = json.loads((OUT / 'verification.json').read_text())
    assert verified['status'] == 'passed'
    meta = json.loads((OUT / 'run_metadata.json').read_text())
    macro = read_csv(OUT / 'macro_summary.csv')
    cells = read_csv(OUT / 'aggregate_metrics.csv')
    contrasts = read_csv(OUT / 'macro_contrasts.csv')
    lookup = {(r['arm'], r['model'], r['generator']): r for r in macro}
    cell_lookup = {(r['arm'], r['model'], r['condition'], r['generator']): r for r in cells}
    names = {'clean': '干净训练', 'jpeg_only': '仅JPEG70增强', 'resize_only': '仅缩放增强', 'augmented': '联合增强'}
    lines = ['SC4001 J：第六轮JPEG70／缩放增强消融（2026-10-22）', '',
             'Material Passport: academic-research-suite / experiment-agent; run; round6_v1.',
             '状态：实验完成，保存的模型及指标复核通过。', '',
             '本轮只改变专家训练视图；门控训练、校准、模型选择和开发图像及视图均沿用第五轮。',
             '四组使用相同400张专家训练原图、相同C和每原图总损失权重：',
             '  干净：clean权重1。',
             '  仅JPEG：clean权重2/3，JPEG70权重1/3。',
             '  仅缩放：clean权重2/3，resize05权重1/3。',
             '  联合：clean、JPEG70、resize05各1/3。',
             '单项保留与联合组相同的增强份额，省去的增强用clean替代；实现中的重复clean视图不算新样本。',
             '两个预处理档、是否温度校准、普通与质量门控的三个初始化全部保留。',
             '专家训练400／门控训练200／校准100／模型选择100／开发400原图，五个分区完全不重叠。', '',
             'ADM三条件宏平均AUROC：clean、JPEG50、resize05等权；门控先按三个初始化平均。']
    kinds = ['global', 'local', 'average', 'fixed', 'score_gate', 'quality_gate']
    for p in PROFILES:
        for cal in CALIBRATIONS:
            lines += ['', f'{p} / {cal}：全局、局部、平均融合、固定融合、普通门控、质量门控']
            for mode in EXPERT_MODES:
                values = [float(lookup[(f'{p}/{mode}/{cal}', kind, 'ADM')]['macro_auroc']) for kind in kinds]
                lines.append('  ' + names[mode] + ': ' + ' / '.join(f'{v:.4f}' for v in values))
    lines += ['', '同一平均融合规则下的增强差值（ADM三条件宏平均，成对95%区间）：',
              '平均融合用于观察专家训练变化，避免混入门控学习和固定权重选择的变化。']
    for row in contrasts:
        if row['generator'] == 'ADM' and row['model'] == 'average':
            lines.append(f"  {row['profile']:10} {row['calibration']:11} {row['contrast']:28}: "
                         f"{float(row['macro_auroc_delta']):+.4f} [{float(row['paired_ci_low']):+.4f}, {float(row['paired_ci_high']):+.4f}]")
    lines += ['', '逐条件ADM：原预处理、不校准，平均融合与质量门控']
    for mode in EXPERT_MODES:
        for condition in DEV_CONDITIONS:
            avg = float(cell_lookup[(f'original/{mode}/raw', 'average', condition, 'ADM')]['mean_auroc'])
            gate = float(cell_lookup[(f'original/{mode}/raw', 'quality_gate', condition, 'ADM')]['mean_auroc'])
            lines.append(f'  {names[mode]} / {condition}: average={avg:.4f}, quality_gate={gate:.4f}')
    lines += ['', '对照复现与解释边界：',
              f"干净和联合增强参考组共96,000条开发预测，与第五轮logit／概率／权重的最大差异：{verified['max_control_prediction_error_vs_round5']:.3g}。",
              '1,440个指标单元、96个门控、16个专家头和16个温度拟合复核通过。',
              '宏平均bootstrap按原图同步抽取其三个条件，再计算方法差值；不会把退化视图当作独立图像。',
              '区间条件于已拟合的专家、校准和门控；单一数据划分，未作多重比较校正。',
              '对增强的结论只适用于当前剂量、数据和训练流程；特征标准化也随训练分布改变。',
              '不能从本轮推断某种图像处理对所有检测器普遍有害，或已经确定具体像素机制。',
              'ADM已被反复查看，仍为探索性开发集，未来需要新的独立评价来源。',
              f'训练运行约{meta["run_seconds"]:.1f}秒，复用特征；核验和报告另计。', '',
              '运行：./.venv/Scripts/python.exe -u scripts/run_round6.py',
              '核验：./.venv/Scripts/python.exe scripts/verify_round6.py',
              '报告：./.venv/Scripts/python.exe scripts/summarize_round6.py']
    (OUT / 'summary_zh.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True)
    for line, p in zip(axes, PROFILES):
        for ax, cal in zip(line, CALIBRATIONS):
            for model, color, marker, style in [('average', '#999999', 'o', '--'),
                                               ('score_gate', '#555555', '^', '-'),
                                               ('quality_gate', '#111111', 'D', '-')]:
                values = [float(lookup[(f'{p}/{mode}/{cal}', model, 'ADM')]['macro_auroc']) for mode in EXPERT_MODES]
                ax.plot(range(4), values, color=color, marker=marker, linestyle=style, label=model)
            ax.set_title(p + ' / ' + cal)
            ax.set_xticks(range(4), ['Clean', 'JPEG70 only', 'Resize only', 'Joint'])
            ax.set_ylim(.4, .8)
            ax.axhline(.5, color='#bbbbbb', linestyle=':', linewidth=1)
            ax.grid(axis='y', alpha=.2); ax.spines[['top', 'right']].set_visible(False)
        line[0].set_ylabel('ADM mean AUROC across 3 conditions')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.suptitle('Round 6: JPEG70 and resize augmentation at matched one-third doses')
    fig.text(.5, .085, 'One data split; gate means over 3 initializations. Paired intervals are provided in macro_contrasts.csv.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .12, 1, .96])
    fig.savefig(OUT / 'round6_augmentation_ablation.png', dpi=180)
    print('Saved Chinese summary and matched-dose comparison plot.')


if __name__ == '__main__':
    main()
