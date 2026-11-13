"""Build report tables from archived results; never load models or test images.

Run from repository root: .venv/Scripts/python.exe scripts/report_evidence_20261111/build_evidence.py
The --root option also accepts the evidence package's sources directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("output/analysis")
GATE = BASE / "gate_diagnosis_20261108"
MULTI = BASE / "multigen_gate_20261108/run_v1_codefix1"
RECOMP = BASE / "recompression_reliability_20261108"
PAIRED = BASE / "paired_view_global_20261108"
MECH = BASE / "gate_failure_mechanism_20261108"
FINAL = BASE / "project_status_20261108"
SEEDS = [20261101, 20261102, 20261103]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.output or root / BASE / "report_evidence_20261111").resolve()
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    used: set[Path] = set()
    checks = []

    def read_json(rel):
        used.add(rel)
        return json.loads((root / rel).read_text(encoding="utf-8"))

    def read_csv(rel):
        used.add(rel)
        return pd.read_csv(root / rel)

    def write_table(name, rows):
        df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        df.to_csv(tables / name, index=False, encoding="utf-8-sig", float_format="%.15g")
        return df

    gate = read_json(GATE / "summary.json")
    gate_csv = read_csv(GATE / "method_comparison.csv").set_index("method")
    multi = read_json(MULTI / "summary.json")
    multi_cells = read_csv(MULTI / "metrics_seed_condition.csv")
    recomp = read_json(RECOMP / "summary.json")
    recomp_cells = read_csv(RECOMP / "detection_diagnostics.csv")
    paired = read_json(PAIRED / "summary.json")
    paired_cells = read_csv(PAIRED / "metrics.csv")
    registry = [
        ("F0", "冻结模型的新图像评价", "published_final_aggregate", "R8最终评价", "既有模型在新图像上的表现", "冻结模型、既定阈值", "部分JPEG排序收益伴随漏检或误报代价", "只读取已发布汇总；与11月开发实验分开", str(FINAL / "final_main_metrics.csv")),
        ("D1", "按AUROC选择检查点", "exploratory_development", "原matched门控协议", "BCE检查点选择是否损害排序", "原训练轨迹、专家、分区", "恢复原质量门控损失，接近平均融合", "改变选择准则，不改变优化目标", str(GATE / "diagnosis.txt")),
        ("D2", "原始视图门控及固定权重", "exploratory_development", "原始视图小型对照", "改变门控看到的训练视图能否改善迁移", "专家与原图身份分区", "质量门控下降；固定融合小收益由一个seed贡献", "固定融合有阈值指标代价", str(GATE / "diagnosis.txt")),
        ("D3", "有界权重与排序损失", "exploratory_development", "组合候选", "限制不利调权并优化排序", "专家与开发条件", "权重接近0.5，未产生实质自适应收益", "两项修改作为整体评估", str(GATE / "diagnosis.txt")),
        ("D4", "质量识别退化再融合", "exploratory_development", "条件路由候选", "先识别退化能否提供有效融合权重", "推理不用真实退化标签", "低于平均融合；退化分类发生来源迁移", "分类器与条件权重均依赖来源", str(GATE / "diagnosis.txt")),
        ("D5", "多生成器门控", "exploratory_development", "原图视图；本轮重跑单来源对照", "扩展生成器覆盖能否改善留出来源", "专家、500拟合原图、248选择原图、网络", "主AUROC未改善；AP/BA/FPR存在取舍", "同时改变拟合和选择构成，不等于增加样本量", str(MULTI / "report_zh.txt")),
        ("D6", "JPEG95再压缩稳定性", "exploratory_development", "固定专家可靠性探针", "更稳定是否对应更低单图BCE", "专家、信号方向、源域尺度", "可靠性AUROC接近0.5；选专家后检测下降", "可靠性预测与真假检测是不同终点", str(RECOMP / "report_zh.txt")),
        ("D7", "原图与匹配图1:1训练", "exploratory_development", "matched global为主对照", "兼顾clean与JPEG表现", "原图总权重、骨干、分类器超参数", "clean BigGAN局部恢复，其他七条件下降", "同时重拟合scaler与分类器，未隔离具体像素机制", str(PAIRED / "report_zh.txt")),
        ("A1", "门控失败机制诊断", "posthoc_development_analysis", "冻结多来源门控", "损失出现在哪些路由与样本对", "冻结分数、参数、阈值", "来源偏好错配、质量关联反转、部分输入干预恢复", "冻结网络输入干预不等于删除特征后重训", str(MECH / "diagnosis_zh.txt")),
        ("A2", "门控损失分解及配对区间", "new_posthoc_development_analysis", "全八条件；1000次原图配对重采样", "总体偏好与逐图调整各贡献多少", "保存预测、原图分组、全部seed/init", "见supplement目录，补充机制证据精度", "目标均值权重仅作诊断；每次重采样重估", "supplement/RESULT_POINTER.json"),
        ("A3", "冻结阈值误报漏检分解", "new_published_aggregate_analysis", "已发布R8最终汇总", "排序改善为何未改善判定", "已发布BA/FPR及冻结阈值", "导出FNR，显示压缩漏检与clean误报的取舍", "只做代数分解，无逐图访问或阈值再选择", "tables/final_threshold_tradeoffs.csv"),
    ]
    write_table("experiment_registry.csv", [dict(zip(["experiment_id", "name_zh", "evidence_stage", "protocol_group", "question_zh", "held_fixed_zh", "finding_zh", "interpretation_note_zh", "source"], row)) for row in registry])

    comparisons = []
    def add_comparison(exp, label, candidate, reference, values, refs, source, endpoint="detection_auroc", interval=None, note=""):
        values, refs = np.asarray(values, float), np.asarray(refs, float)
        assert values.shape == refs.shape == (3,)
        delta = values - refs
        comparisons.append(dict(experiment_id=exp, comparison_zh=label, endpoint=endpoint,
            candidate=candidate, reference=reference, candidate_mean=float(values.mean()), reference_mean=float(refs.mean()),
            delta=float(delta.mean()), seed1_delta=delta[0], seed2_delta=delta[1], seed3_delta=delta[2],
            positive_seed_count=int((delta > 1e-12).sum()), negative_seed_count=int((delta < -1e-12).sum()),
            near_zero_seed_count=int((abs(delta) <= 1e-12).sum()),
            conditional_ci_low=None if interval is None else interval[0], conditional_ci_high=None if interval is None else interval[1],
            uncertainty_note="not estimated" if interval is None else "archived conditional paired-bootstrap percentile 95%; shared data pool",
            source=str(source), protocol_note=note))

    gate_by = {m["method"]: m for m in gate["methods"]}
    for method, m in gate_by.items():
        assert np.isclose(m["mean_auroc"], gate_csv.loc[method, "mean_auroc"], atol=1e-14, rtol=0)
        assert np.isclose(np.mean(m["seed_aurocs"]), m["mean_auroc"], atol=1e-14, rtol=0)
    initial_map = {"original_quality_gate":"D1", "auc_checkpoint_quality_gate":"D1", "original_view_quality_gate":"D2", "original_view_fixed":"D2", "bounded_ranking_quality_gate":"D3", "quality_condition_gate":"D4"}
    for method, exp in initial_map.items():
        add_comparison(exp, method + " vs average", method, "average", gate_by[method]["seed_aurocs"], gate_by["average"]["seed_aurocs"], GATE / "summary.json", note="Specific original exploratory recipe; not the later multi-source control")
    add_comparison("D1", "只改检查点选择", "auc_checkpoint_quality_gate", "original_quality_gate", gate_by["auc_checkpoint_quality_gate"]["seed_aurocs"], gate_by["original_quality_gate"]["seed_aurocs"], GATE / "summary.json")
    checks.append({"check":"initial_method_summary_and_csv", "rows":len(gate_by), "passed":True})

    mg = {(r["arm"],r["model"],r["checkpoint"]):r for r in multi["macro"]}
    for key, row in mg.items():
        arm, model, checkpoint = key
        arm_mask = multi_cells.scenario.eq("source_only") if arm == "source_only" else multi_cells.scenario.str.startswith("mixed_")
        subset = multi_cells[arm_mask & (multi_cells.model == model) & (multi_cells.checkpoint == checkpoint)]
        derived = subset.groupby("run_seed")["auroc"].mean().reindex(SEEDS).values
        assert np.allclose(derived, row["each_seed_auroc"], atol=1e-14, rtol=0)
        assert np.isclose(derived.mean(),row["auroc"],atol=1e-14,rtol=0)
    checks.append({"check":"multi_source_macros_from_condition_metrics", "rows":len(mg), "passed":True})
    cis = {r["comparison"]:r["conditional_percentile_95"] for r in multi["bootstrap"]}
    candidate = mg[("mixed","quality_gate","auroc")]
    for arm, model, cp, name in [("source_only","quality_gate","auroc","mixed-quality-source-quality"),("mixed","average","fixed","quality-average"),("mixed","score_gate","auroc","quality-score"),("mixed","fixed","fixed","quality-fixed")]:
        ref = mg[(arm,model,cp)]
        add_comparison("D5", name, "mixed_quality_gate", arm+"_"+model, candidate["each_seed_auroc"], ref["each_seed_auroc"], MULTI / "summary.json",interval=cis[name],note="Original views; source-only control rerun in this protocol; AUROC checkpoint")
    write_table("multigen_metrics_all.csv", multi_cells)

    dev = recomp_cells[recomp_cells.scope == "development"]
    grouped = dev.groupby("run_seed")[["stable_selected_logit_auroc","average_fusion_auroc"]].mean().reindex(SEEDS)
    assert len(dev) == 24
    add_comparison("D6", "稳定性选择的真假检测", "stability_selector", "average", grouped.stable_selected_logit_auroc, grouped.average_fusion_auroc, RECOMP / "detection_diagnostics.csv",note="Detection endpoint, distinct from relative-BCE reliability AUROC")
    reliability_rows = []
    for name, value in recomp["primary_macro"].items():
        ci = recomp["bootstrap"]["primary" if name == "stability_scaled" else "reference"]["conditional_percentile_95"] if name in ["stability_scaled","score_gate_reference"] else [None,None]
        reliability_rows.append(dict(group="primary_stratified_macro",signal=name,relative_bce_reliability_auroc=value,ci_low=ci[0],ci_high=ci[1]))
    for group, values in recomp["development_group_macro"].items():
        lo,hi = values["primary_interval"]["conditional_percentile_95"]
        reliability_rows.append(dict(group=group,signal="stability_scaled",relative_bce_reliability_auroc=values["stability_scaled"],ci_low=lo,ci_high=hi))
    write_table("reliability_endpoint_separate.csv", reliability_rows)
    write_table("stability_detection_all.csv",dev)

    seed_col = "run_seed" if "run_seed" in paired_cells else "seed"
    for scope in ["overall","jpeg","clean_BigGAN","clean_ADM"]:
        subset = paired_cells[paired_cells.partition == "development"]
        if scope == "jpeg": subset = subset[subset.condition.isin(["jpeg50","jpeg70"])]
        elif scope.startswith("clean_"): subset = subset[(subset.condition=="clean") & (subset.generator == scope[6:])]
        series = subset.groupby(["arm",seed_col]).auroc.mean()
        a,b = series.loc["paired_half"].reindex(SEEDS).values,series.loc["matched"].reindex(SEEDS).values
        assert np.allclose(a-b,paired["each_seed_delta_vs_matched"][scope],atol=1e-14,rtol=0)
        ci = paired["bootstrap"]["comparisons"]["matched"][scope]["conditional_percentile_95"]
        add_comparison("D7", "1:1 vs matched / "+scope, "paired_half", "matched_global", a,b,PAIRED/"summary.json", interval=ci,note="Primary reference is matched global; equal total weight per original")
    checks.append({"check":"paired_training_deltas_from_per_seed_cells", "scopes":4, "passed":True})
    write_table("paired_training_all.csv",paired_cells)
    write_table("improvement_comparisons.csv",comparisons)

    # New aggregate-only threshold decomposition. FNR is derived within each seed,
    # then averaged; its SD is calculated from paired seed FNR, not from marginal SDs.
    final = read_csv(FINAL / "final_main_metrics.csv")
    published = read_csv(FINAL / "sources/round8_metrics_mean_sd.csv")
    indexed = ["cohort","arm","model","condition","generator","metric"]
    reference = published[published.generator == "main_macro"].set_index(indexed)
    indexed_final = final.set_index(indexed)
    assert len(indexed_final)==len(reference)==320
    assert np.allclose(indexed_final.sort_index()[["mean","sd","seed1","seed2","seed3"]],reference.sort_index()[["mean","sd","seed1","seed2","seed3"]],atol=1e-14,rtol=0)
    assert np.allclose(final[["seed1","seed2","seed3"]].mean(axis=1),final["mean"],atol=1e-14,rtol=0)
    assert np.allclose(final[["seed1","seed2","seed3"]].std(axis=1,ddof=1),final["sd"],atol=1e-14,rtol=0)
    threshold_rows=[]
    for key, g in final.groupby(["cohort","arm","model","condition","generator"],sort=True):
        m = g.set_index("metric")
        seed_ba=m.loc["balanced_accuracy",["seed1","seed2","seed3"]].to_numpy(float)
        seed_fpr=m.loc["real_false_positive_rate",["seed1","seed2","seed3"]].to_numpy(float)
        seed_fnr=2-2*seed_ba-seed_fpr
        assert np.all((seed_fnr >= -1e-12)&(seed_fnr<=1+1e-12))
        assert np.allclose((2-seed_fpr-seed_fnr)/2,seed_ba,atol=1e-14,rtol=0)
        row=dict(zip(["cohort","arm","model","condition","generator"],key))
        row.update(auroc=m.loc["auroc","mean"],balanced_accuracy=m.loc["balanced_accuracy","mean"],real_fpr=seed_fpr.mean(),fake_fnr=seed_fnr.mean(),fnr_seed_sd=seed_fnr.std(ddof=1),fnr_seed1=seed_fnr[0],fnr_seed2=seed_fnr[1],fnr_seed3=seed_fnr[2],scope="published final aggregate; frozen original thresholds",formula="FNR = 2 - 2*balanced_accuracy - FPR")
        threshold_rows.append(row)
    threshold=write_table("final_threshold_tradeoffs.csv",threshold_rows)
    contrast_rows=[]
    for (model,condition),g in threshold[threshold.cohort=="round8_a"].groupby(["model","condition"]):
        a=g.set_index("arm")
        row=dict(model=model,condition=condition,comparison="matched minus unmatched")
        for metric in ["auroc","balanced_accuracy","real_fpr","fake_fnr"]:
            row["unmatched_"+metric]=a.loc["unmatched",metric]
            row["matched_"+metric]=a.loc["matched",metric]
            row["delta_"+metric]=a.loc["matched",metric]-a.loc["unmatched",metric]
        contrast_rows.append(row)
    write_table("final_matched_unmatched_contrasts.csv",contrast_rows)
    checks.append({"check":"final_published_aggregates_mean_sd_and_linear_fnr_identity", "metric_rows":320,"derived_fnr_cells":80,"matched_unmatched_comparisons":24,"passed":True})

    # Preserve documentary context along with numbers; no model/image/prediction payloads.
    documentary = [GATE/"diagnosis.txt", MULTI/"report_zh.txt", MULTI/"freeze.json", MULTI/"independent_verification.json", RECOMP/"report_zh.txt", RECOMP/"protocol.json", RECOMP/"verification.json", PAIRED/"report_zh.txt", PAIRED/"freeze.json", PAIRED/"verification.json", MECH/"diagnosis_zh.txt", MECH/"prediction_verification.json", MECH/"feature_verification.json", MECH/"source_target_transfer.csv",MECH/"prediction_ranking_seed_means.csv",MECH/"prediction_constant_seed_means.csv",MECH/"feature_label_auc_summary.json",MECH/"feature_interventions_summary.json",FINAL/"provenance.json",FINAL/"sources/round8_ROUND8_FINAL.txt",FINAL/"sources/round8_verification.json"]
    used.update(documentary)
    manifest=[]
    for rel in sorted(used):
        src=root/rel
        dst=out/"sources"/rel
        dst.parent.mkdir(parents=True,exist_ok=True)
        if src.resolve()!=dst.resolve(): shutil.copy2(src,dst)
        digest=hashlib.sha256(src.read_bytes()).hexdigest()
        assert hashlib.sha256(dst.read_bytes()).hexdigest()==digest
        manifest.append(dict(source=str(rel).replace("\\","/"),snapshot=str(dst.relative_to(out)).replace("\\","/"),sha256=digest,bytes=src.stat().st_size))
    (out/"source_manifest.json").write_text(json.dumps(dict(date="2026-11-11",sources=manifest,scope="Saved aggregates and documentation; source files unchanged"),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"table_verification.json").write_text(json.dumps(dict(status="passed",checks=checks,source_snapshot_count=len(manifest),new_training_runs=0,final_image_or_prediction_reads=0,notes=["Checks reconstruct published arithmetic and selected contrasts, not model training.","Historical verification records are copied, not represented as newly executed.","Seed variability comes from a shared data pool; it is not independent-dataset replication."]),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=["冻结阈值下的误报与漏检分解（2026-11-11）", "仅用已发布Round 8最终汇总；没有读取最终逐图预测或重新选择阈值。", "FNR = 2 - 2*BA - FPR，按seed先推导再计算均值和SD。", "main_macro包含GLIDE、Midjourney、VQDM、Wukong；ADM/BigGAN也在最终评价中单列，但此前已用于开发。本表仅展示主宏平均。", "", "R8A average，未匹配 → 格式匹配："]
    contrasts=pd.DataFrame(contrast_rows)
    for condition in ["clean","jpeg70","jpeg50","resize05"]:
        r=contrasts[(contrasts.model=="average")&(contrasts.condition==condition)].iloc[0]
        lines.append(f"{condition}: AUROC {r.unmatched_auroc:.4f}→{r.matched_auroc:.4f}; BA {r.unmatched_balanced_accuracy:.2%}→{r.matched_balanced_accuracy:.2%}; FPR {r.unmatched_real_fpr:.2%}→{r.matched_real_fpr:.2%}; FNR {r.unmatched_fake_fnr:.2%}→{r.matched_fake_fnr:.2%}。")
    lines += ["", "解释：JPEG下排序改善伴随更低真实图误报和更高假图漏检；clean/resize则以更高误报换来更低漏检。该分解说明冻结阈值的错误类型取舍，没有在测试数据上校准新阈值。", "完整20配置×4条件保存在tables/final_threshold_tradeoffs.csv；24组matched/unmatched对照均保留。"]
    (out/"threshold_analysis_zh.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(dict(status="passed",output=str(out),tables=len(list(tables.glob('*.csv'))),sources=len(manifest),checks=len(checks)),ensure_ascii=False))


if __name__ == "__main__":
    main()
