"""Write the complete exploratory diagnosis without selecting a final model."""
import csv
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'tmp/gate_diagnosis'
OUT = ROOT / 'output/analysis/gate_diagnosis_20261108'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    methods = []
    checkpoint = read(BASE/'checkpoint_results/summary.json')
    for model, cp, title in [
        ('average', 'saved', 'average'), ('fixed', 'saved', 'original_fixed'),
        ('quality_gate', 'bce_checkpoint', 'original_quality_gate'),
        ('quality_gate', 'auc_checkpoint', 'auc_checkpoint_quality_gate')]:
        values = [r['eight_cell_macro_auc'] for r in checkpoint
                  if r['arm']=='matched' and r['model']==model and r['checkpoint']==cp]
        methods.append(dict(method=title, mean_auroc=float(np.mean(values)), seed_aurocs=values))
    with (BASE/'original_gate_views/metrics_seed_condition.csv').open() as f:
        original = list(csv.DictReader(f))
    for model, title in [('quality_gate','original_view_quality_gate'), ('fixed','original_view_fixed')]:
        values = [float(np.mean([float(r['auroc']) for r in original
                   if r['scenario']=='original_gate_views' and r['model']==model and r['run_seed']==str(seed)]))
                  for seed in (20261101,20261102,20261103)]
        methods.append(dict(method=title, mean_auroc=float(np.mean(values)), seed_aurocs=values))
    ranking = [r for r in read(BASE/'ranking_gate/summary.json') if r['partition']=='development' and r['model']=='quality_gate']
    values = np.asarray([r['auroc']['each_seed'] for r in ranking]).mean(axis=0).tolist()
    methods.append(dict(method='bounded_ranking_quality_gate',mean_auroc=float(np.mean(values)),seed_aurocs=values))
    with (BASE/'condition_gate/metrics_seed_condition.csv').open() as f:
        condition = list(csv.DictReader(f))
    values = [float(np.mean([float(r['auroc']) for r in condition if r['model']=='condition_gate' and r['run_seed']==str(seed)]))
              for seed in (20261101,20261102,20261103)]
    methods.append(dict(method='quality_condition_gate',mean_auroc=float(np.mean(values)),seed_aurocs=values))
    baseline=methods[0]
    for r in methods:
        r['delta_vs_average']=r['mean_auroc']-baseline['mean_auroc']
        r['paired_seed_deltas_vs_average']=(np.array(r['seed_aurocs'])-np.array(baseline['seed_aurocs'])).tolist()
    freeze=read(BASE/'checkpoint_results/freeze.json')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in freeze['inputs_sha256'].items())
    artifact = dict(date='2026-11-08',status='analysis completed; no improved final model selected',
       scope='Previously inspected Round 8 A ADM/BigGAN development only. Equal mean of 2 generators x 4 conditions, not Round 8 four-generator final main_macro.',
       aggregation='Average gate-initialization metrics within a data seed, then equal mean of three shared-pool data seeds. No significance test or independent validation.',
       experiment_design='Each candidate frozen before its own development scoring. This was a sequential exploratory investigation, not a single preregistered confirmatory study.',
       methods=methods,
       verification=dict(original_gate_replays=36,max_replay_weight_difference=0,
         archived_metric_cross_check=read(BASE/'checkpoint_results/archived_metric_cross_check.json'),
         original_view_ablation=read(BASE/'original_gate_views/verification.json'),
         condition_gate=read(BASE/'condition_gate/verification.json')),
       next_hypothesis='Gate reliability must be validated across generators. Retain existing experts and compare expanded multi-generator gate training with leave-one-generator-out selection; apply identical information and evaluation to average/fixed/score/quality controls. No claimed improvement until new independent image evaluation.')
    (OUT/'summary.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'method_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['method','mean_auroc','delta_vs_average','seed1','seed2','seed3'])
        writer.writeheader()
        for r in methods:
            writer.writerow(dict(method=r['method'],mean_auroc=r['mean_auroc'],delta_vs_average=r['delta_vs_average'],
                                 **dict(zip(['seed1','seed2','seed3'],r['seed_aurocs']))))
    table='\n'.join(f"{r['method']:32s} {r['mean_auroc']:.6f}  delta {r['delta_vs_average']:+.6f}  seeds {r['seed_aurocs']}" for r in methods)
    report='''质量门控诊断与小型探索对照（2026-11-08）

结论
没有发现可解释全部失败的实现错误。质量门控确实改变了融合权重，但现有训练目标、单生成器监督和质量代理的迁移能力没有让它稳定超过平均/固定融合。
四个针对性候选均未建立稳定的质量门控收益。原始视图选择固定权重出现小幅 AUROC 增益，但只由一个划分贡献，且平衡准确率存在代价；不选定或发布新的最终模型。

数据范围
仅使用 Round 8 A 保存的 gate_fit/tuning/selection/ADM-BigGAN development 数据和允许的特征缓存。专家不重训，不重新提取 CLIP 特征。
下列平均是 ADM/BigGAN x clean/JPEG70/JPEG50/resize05 八个开发单元，不能与最终测试四生成器 main_macro 混为一谈。
各门控先在每个数据 seed 内平均三次初始化，再汇总三个 seed；三个 seed 共享数据池，不是独立研究。全部候选属于连续探索，每项都在其开发评分前锁定；未以已查看的最终测试集选择参数。

有数据支持的诊断
1. 优化和选择目标不一致。原门控训练、选择 checkpoint 都依据 BCE，固定权重却按每条件 AUROC 选。
   matched SD15 原始 selection 的 JPEG50：平均融合 BCE 2.1630，质量门控 1.8244；AUROC 反而从 .9259 降到 .9112。
   将同一训练轨迹改用 AUROC 选 checkpoint，可以恢复大部分损失，但最终接近初始平均融合。
2. 共同错误严重。matched JPEG50 的 ADM/BigGAN 假图，按专家各自保存阈值，同时漏检比例为 95.67%/84.44%。
   这说明简单地在两位专家既有判定中择一，补救空间有限；不构成任意 logit 融合或 AUROC 的上界。
3. 门控仅在 500 张 gate_fit 原图（250 Real、250 SD15）上学习；三种视图不等于 1500 个独立样本。
   JPEG50 中分配给标签可见的更低 BCE 专家的平均权重：SD15 selection .620、ADM .482、BigGAN .529。
   这些是描述性诊断，不证明某一特征独立导致失败。
4. 质量代理存在来源迁移。三个特征是 224px 图上的锐度、高频比例和固定 8px 网格边界差分比，受内容及几何变化影响。
   新的质量->退化类型分类器在 tuning 上约 91%-93% 准确，开发集降至约 74%-75%，BigGAN 仅 56%-60%。
   每个 seed 的 300 张 BigGAN JPEG70 中，272/274/271 张被预测为 resize。这是对质量代理迁移不足的直接观测。
5. 没有发现门控不工作或所有权重固定的问题。matched BigGAN 的质量门控 global 权重从 clean .702 降到 JPEG50 .326；权重变化不等于有效选择。
   约半数专家 logit 对应极端概率，概率/熵输入会压缩幅度差异；这是有依据的候选局限，尚未单独证明其因果作用。

小型对照结果（AUROC）
'''+table+'''

具体解释
- AUROC checkpoint：原始 36 次 gate 训练逐一重放，权重最大差异 0；不改变网络或梯度，只改选轮次。新质量门控接近平均融合，差异约 0.00008，不视为有效增益。
- 原始视图训练/调参 gate：仍用原 matched 专家与相同 BCE gate，质量门控下降至 .602109，排除了“只要把视图改回原图就解决”的简单解释。
- 有界权重+排序损失：18 次配对 gate，前两数据 seed 均选择 epoch0，第三 seed 选择 epoch1，alpha 约 .501。没有有意义的自适应收益；这是组合候选，不能归因于单一组件。
- 质量识别退化再融合：不在推理时读取真实退化标签，只用质量特征预测。总体 .621070，低于平均融合；source 上选出的条件权重及质量分类器都没有稳定迁移。

固定融合的小幅候选
在原始 selection 视图选 alpha，三 seed 为 .5/.6/.5，旧值是 .4/.5/.5。
相对平均融合的八单元 AUROC 差值为 [0, +.005722, 0]，均值 +.001907；相对旧固定融合为 +.002910。
改进主要来自 ADM 和 resize；BigGAN clean/JPEG70/JPEG50 降低。
改用同套原始 selection 阈值后，相对旧固定融合 BA .547500 -> .540625，真实图 FPR .182500 -> .128889。
这是排序与判定阈值的取舍，不足以声称稳定的新方法改进。

建议的下一项研究（尚未执行）
优先改变门控能看到的生成器覆盖与验证方法，而非继续扩大 MLP。
保持专家固定；允许的开发来源中，把多个生成器纳入新 gate 拟合/选择协议，并按原始图像身份分开训练与选择，所有退化视图随原图分区。
用留一生成器验证来选 gate；平均、固定、普通门控、质量门控采用相同数据权限与阈值规则。需要证明质量输入相对普通门控仍有增益。
若这个开发阶段有稳定候选，先锁定实现和主指标，再由最终评估负责人用未参与本轮设计的新图像确认。已看过的最终测试成绩不能重新用于选参数。
较小提升可以成为项目证据，但应报告每个划分、生成器、条件及判定指标的代价，不能只摘一个正数。

复现与证据位置
tmp/gate_diagnosis/overlap/diagnose_development.py 与 diagnosis.json
tmp/gate_diagnosis/checkpoint_protocol.json、checkpoint_diagnostic.py、checkpoint_results/
tmp/gate_diagnosis/original_gate_views/run_original_gate_views.py、protocol.json、summary.json、verification.json、fixed_selection_verification.json
tmp/gate_diagnosis/ranking_gate/run_ranking_gate.py、freeze.json、summary.json、complete.json
tmp/gate_diagnosis/condition_gate/run_condition_gate.py、protocol.json、summary.json、verification.json
这些脚本保留全部正负结果。已冻结实验会拒绝覆盖原结果；需要复现时应先保留现有目录并使用新的独立输出位置。

验证状态
ANALYZED；原门控重放及归档指标重算已核验，新的候选只有探索性开发评估，未获独立最终测试确认。
11/11 统计解释风险已按本任务范围检查。主要限制为反复观察的开发集、共享数据池、源域和目标域混淆、聚合掩盖条件差异，以及顺序探索不能视为确认性检验。
'''
    (OUT/'diagnosis.txt').write_text(report,encoding='utf-8')
    print(json.dumps(dict(output=str(OUT),methods=methods),indent=2))


if __name__=='__main__':
    main()
