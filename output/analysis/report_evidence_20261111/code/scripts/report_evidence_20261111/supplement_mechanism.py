"""Paired uncertainty for frozen gate-loss decomposition, using development scores.

No fitting, selection, image access, or final-test access. The target-cell mean
weight is a descriptive functional, re-estimated on every bootstrap sample;
it is not a deployable alternative model. See protocol.json in the output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "output/analysis/multigen_gate_20261108/run_v1_codefix1"
ARCHIVE = ROOT / "output/analysis/gate_failure_mechanism_20261108"
KEY = ["run_seed", "scenario", "heldout_generator", "condition", "gate_seed"]
N = 300
LABELS = np.r_[np.zeros(N, dtype=int), np.ones(N, dtype=int)]
METRICS = ["average_auroc", "target_mean_alpha_auroc", "quality_gate_auroc",
           "mean_preference_effect", "image_specific_effect", "total_effect"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def weighted_auc(scores, weights, order=None):
    """AUC on original identities with integer bootstrap multiplicities.

    Stable sorting and an explicit tie correction retain half credit for ties.
    Repeated sampled IDs are represented by multiplicity, never independent rows.
    """
    if order is None:
        order = np.argsort(scores, axis=1, kind="stable")
    s = np.take_along_axis(scores, order, axis=1)
    w = np.take_along_axis(weights, order, axis=1)
    y = LABELS[order]
    rw = w * (y == 0)
    fw = w * (y == 1)
    preceding = np.cumsum(rw, axis=1) - rw
    numerator = (fw * preceding).sum(axis=1).astype(float)
    # For a tie block, remove order-dependent credit and insert half credit.
    for i in np.flatnonzero(np.any(np.diff(s, axis=1) == 0, axis=1)):
        starts = np.r_[0, np.flatnonzero(np.diff(s[i]) != 0) + 1]
        ends = np.r_[starts[1:], s.shape[1]]
        for lo, hi in zip(starts, ends):
            if hi - lo > 1:
                old = np.sum(fw[i, lo:hi] * (preceding[i, lo:hi] - preceding[i, lo]))
                new = .5 * fw[i, lo:hi].sum() * rw[i, lo:hi].sum()
                numerator[i] += new - old
    return numerator / (rw.sum(axis=1) * fw.sum(axis=1))


def credit(scores):
    diff = scores[N:, None] - scores[None, :N]
    return (diff > 0).astype(float) + .5 * (diff == 0)


def point_metrics(average, constant, actual, weights):
    a, c, q = [weighted_auc(s, weights) for s in [average, constant, actual]]
    return np.stack([a, c, q, c-a, q-c, q-a], axis=-1)


def main():
    global ROOT, SOURCE, ARCHIVE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Source tree containing the relative input paths recorded in protocol.json")
    parser.add_argument("--output", type=Path, default=None, help="New empty result directory; existing evidence is never overwritten")
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args()
    ROOT = args.root.resolve()
    SOURCE = ROOT / "output/analysis/multigen_gate_20261108/run_v1_codefix1"
    ARCHIVE = ROOT / "output/analysis/gate_failure_mechanism_20261108"
    out = (args.output if args.output is not None else ROOT / "output/analysis/report_evidence_20261111/supplement").resolve()
    if args.repetitions < 100:
        raise ValueError("At least 100 paired resamples are required.")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Keep existing evidence intact; choose a new --output: {out}")
    started = time.monotonic()
    inputs = [SOURCE / "predictions.csv.gz", SOURCE / "complete.json",
              SOURCE / "metrics_seed_condition.csv", ARCHIVE / "prediction_constant_alpha.csv",
              ARCHIVE / "prediction_ranking_comparisons.csv"]
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in inputs}
    complete = json.loads((SOURCE / "complete.json").read_text())
    for name in ["predictions.csv.gz", "metrics_seed_condition.csv"]:
        assert sha(SOURCE / name) == complete["outputs_sha256"][name]
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "protocol.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Post-hoc frozen mixed-source quality-gate development diagnosis; no new fitting or candidate selection.",
        "input_sha256": hashes, "script_sha256": sha(__file__),
        "repetitions": args.repetitions, "random_seed": args.seed,
        "population": "300 development Real, 300 ADM, 300 BigGAN originals; all saved data seeds, gate seeds, and four processing conditions.",
        "resampling": "Independently sample 300 IDs with replacement from each of Real, ADM, BigGAN. The same identity multiplicities are paired across all conditions, seeds, models; Real is shared across both targets.",
        "aggregation": "Equal average over three gate initializations within each data seed, then three data seeds, then generators and conditions. Balanced complete cells make direct row averaging algebraically identical.",
        "functional": "A=AUROC(average); C=AUROC(mean target-cell alpha * global + (1-alpha) * local); Q=AUROC(actual quality gate). Mean preference=C-A; image-specific=Q-C; total=Q-A.",
        "mean_alpha_bootstrap": "Recompute each frozen gate's target-cell alpha mean from the resampled 300 Real + 300 fake identities, then evaluate that constant-weight diagnostic on those same resampled IDs. This estimates the sampling variation of the descriptive functional, not of a trained/deployable method.",
        "interval": "Two-sided 2.5/97.5 percentile intervals, conditional on fixed experts/gates/source data. No multiple-comparison correction; exploratory scopes retained in full.",
        "no_final_test_access": True, "models_fitted": 0,
    })
    frame = pd.read_csv(SOURCE / "predictions.csv.gz")
    original_rows = len(frame)
    frame = frame[frame.scenario.str.startswith("mixed_")]
    frame = frame[((frame.model.isin(["global", "local", "average"])) & (frame.checkpoint == "fixed")) |
                  ((frame.model == "quality_gate") & (frame.checkpoint == "auroc"))]
    meta, gs, ls, avs, qs, alphas, rank_rows = [], [], [], [], [], [], []
    pool_ids = {}
    max_logit_error = 0.
    for key, block in frame.groupby(KEY[:-1], sort=True):
        base_meta = dict(zip(KEY[:-1], key))

        def get(model, gate_seed=-1):
            d = block[(block.model == model) & (block.gate_seed == gate_seed)].sort_values(["label", "source_id"])
            assert len(d) == 600 and d.source_id.nunique() == 600
            np.testing.assert_array_equal(d.label, LABELS)
            return d.reset_index(drop=True)

        global_d, local_d, avg_d = [get(m) for m in ["global", "local", "average"]]
        ids = global_d.source_id.tolist()
        for d in [local_d, avg_d]:
            assert d.source_id.tolist() == ids
        for name, entries in [("Real", ids[:N]), (key[2], ids[N:])]:
            if name in pool_ids:
                assert entries == pool_ids[name], f"ID mismatch in {name}"
            else:
                pool_ids[name] = entries
        g, l, a = [d.logit.to_numpy() for d in [global_d, local_d, avg_d]]
        max_logit_error = max(max_logit_error, float(np.max(np.abs(a - .5*(g+l)))))
        cg, cl, ca = [credit(s) for s in [g, l, a]]
        for seed in sorted(block[block.model == "quality_gate"].gate_seed.unique()):
            d = get("quality_gate", seed)
            assert d.source_id.tolist() == ids
            q, alpha = d.logit.to_numpy(), d.alpha_global.to_numpy()
            max_logit_error = max(max_logit_error, float(np.max(np.abs(q-(alpha*g+(1-alpha)*l)))))
            m = dict(**base_meta, gate_seed=int(seed))
            meta.append(m); gs.append(g); ls.append(l); avs.append(a); qs.append(q); alphas.append(alpha)
            cq = credit(q)
            harm, repair = np.maximum(ca-cq, 0), np.maximum(cq-ca, 0)
            row = dict(**m, lost_pair_fraction=float(harm.mean()), repaired_pair_fraction=float(repair.mean()),
                       pair_count=N*N, total_effect_from_pair_credit=float((repair-harm).mean()))
            for name, mask in {
                "global_only_correct": (cg == 1) & (cl == 0),
                "local_only_correct": (cg == 0) & (cl == 1),
                "both_correct": (cg == 1) & (cl == 1),
                "both_wrong": (cg == 0) & (cl == 0),
                "expert_tie": (cg == .5) | (cl == .5),
            }.items():
                row[f"lost_credit_fraction_{name}"] = float(harm[mask].sum() / (N*N))
                row[f"repaired_credit_fraction_{name}"] = float(repair[mask].sum() / (N*N))
            rank_rows.append(row)
    assert len(meta) == 72 and max_logit_error < 1e-10
    assert all(len(x) == 300 for x in pool_ids.values())
    assert len(set().union(*map(set, pool_ids.values()))) == 900
    g, l, av, q, alpha = [np.asarray(x) for x in [gs, ls, avs, qs, alphas]]
    metadata = pd.DataFrame(meta)
    weights = np.ones_like(g)
    mean_alpha = alpha.mean(axis=1)
    const = mean_alpha[:, None]*g + (1-mean_alpha[:, None])*l
    point = point_metrics(av, const, q, weights)
    point_df = pd.concat([metadata, pd.DataFrame(point, columns=METRICS)], axis=1)
    point_df["diagnostic_target_mean_alpha"] = mean_alpha
    point_df.to_csv(out / "decomposition_each_initialization.csv", index=False)
    rank_df = pd.DataFrame(rank_rows)
    rank_df.to_csv(out / "ranking_each_initialization.csv", index=False)

    # Independent sklearn checks, including deliberately tied scores.
    max_sklearn = 0.
    for i in range(72):
        for score, metric in [(av[i], 0), (const[i], 1), (q[i], 2)]:
            max_sklearn = max(max_sklearn, abs(point[i, metric]-roc_auc_score(LABELS, score)))
    rng_check = np.random.default_rng(921)
    for _ in range(10):
        tied = rng_check.integers(-3, 4, size=(4, 600)).astype(float)
        w = np.concatenate([rng_check.multinomial(N, np.ones(N)/N, size=4),
                            rng_check.multinomial(N, np.ones(N)/N, size=4)], axis=1)
        got = weighted_auc(tied, w)
        for i in range(4):
            max_sklearn = max(max_sklearn, abs(got[i]-roc_auc_score(LABELS, tied[i], sample_weight=w[i])))
    assert max_sklearn < 1e-12

    old_const = pd.read_csv(ARCHIVE / "prediction_constant_alpha.csv")
    old_const = old_const[old_const.model == "quality_gate"]
    joined = point_df.merge(old_const, on=KEY, suffixes=("", "_archive"), validate="one_to_one")
    assert len(joined) == 72
    archive_const_error = max(float(abs(joined.target_mean_alpha_auroc-joined.constant_mean_alpha_auroc).max()),
                              float(abs(joined.quality_gate_auroc-joined.actual_auroc).max()),
                              float(abs(joined.diagnostic_target_mean_alpha-joined.diagnostic_target_mean_alpha_archive).max()))
    old_rank = pd.read_csv(ARCHIVE / "prediction_ranking_comparisons.csv")
    joined_rank = rank_df.merge(old_rank[old_rank.reference == "average"], on=KEY, suffixes=("", "_archive"), validate="one_to_one")
    archive_rank_error = max(float(abs(joined_rank.lost_pair_fraction-joined_rank.lost_pair_fraction_archive).max()),
                             float(abs(joined_rank.repaired_pair_fraction-joined_rank.repaired_pair_fraction_archive).max()))
    seed_keys = ["run_seed", "heldout_generator", "condition"]
    by_seed = point_df.groupby(seed_keys, as_index=False)[METRICS].mean()
    archived_metrics = pd.read_csv(SOURCE / "metrics_seed_condition.csv")
    metric_errors = []
    for name, col, checkpoint in [("average", "average_auroc", "fixed"), ("quality_gate", "quality_gate_auroc", "auroc")]:
        old = archived_metrics[(archived_metrics.scenario.str.startswith("mixed_")) & (archived_metrics.model == name) & (archived_metrics.checkpoint == checkpoint)]
        j = by_seed.merge(old, on=seed_keys, validate="one_to_one")
        assert len(j) == 24
        metric_errors.append(float(abs(j[col]-j.auroc).max()))
    assert max(archive_const_error, archive_rank_error, *metric_errors) < 1e-12
    rank_identity = float(np.max(abs(rank_df.total_effect_from_pair_credit.to_numpy()-point[:, 5])))
    decomposition_identity = float(np.max(abs(point[:, 3]+point[:, 4]-point[:, 5])))
    assert max(rank_identity, decomposition_identity) < 1e-12

    # Scopes include all conditions, preventing a JPEG-only description of the model.
    masks = {"all_conditions": np.ones(72, dtype=bool),
             "jpeg_conditions": metadata.condition.str.startswith("jpeg").to_numpy(),
             "clean_conditions": (metadata.condition == "clean").to_numpy(),
             "resize_conditions": (metadata.condition == "resize05").to_numpy()}
    for generator in ["ADM", "BigGAN"]:
        for condition in ["clean", "jpeg70", "jpeg50", "resize05"]:
            masks[f"{generator}_{condition}"] = ((metadata.heldout_generator == generator) & (metadata.condition == condition)).to_numpy()
    scopes = list(masks)
    boot = np.empty((args.repetitions, len(scopes), len(METRICS)))
    order_av, order_q = [np.argsort(s, axis=1, kind="stable") for s in [av, q]]
    target_is_adm = metadata.heldout_generator.to_numpy() == "ADM"
    rng = np.random.default_rng(args.seed)
    bootstrap_sklearn_error = 0.
    for b in range(args.repetitions):
        counts = {name: np.bincount(rng.integers(N, size=N), minlength=N) for name in ["Real", "ADM", "BigGAN"]}
        w = np.concatenate([np.tile(counts["Real"], (72, 1)),
                            np.where(target_is_adm[:, None], counts["ADM"], counts["BigGAN"])], axis=1)
        mu = (alpha*w).sum(axis=1)/(2*N)
        c = mu[:, None]*g + (1-mu[:, None])*l
        ba = weighted_auc(av, w, order_av); bc = weighted_auc(c, w); bq = weighted_auc(q, w, order_q)
        values = np.stack([ba, bc, bq, bc-ba, bq-bc, bq-ba], axis=-1)
        for j, scope in enumerate(scopes):
            boot[b, j] = values[masks[scope]].mean(axis=0)
        if b < 3:
            for i in range(72):
                for score, value in [(av[i], ba[i]), (c[i], bc[i]), (q[i], bq[i])]:
                    bootstrap_sklearn_error = max(bootstrap_sklearn_error, abs(value-roc_auc_score(LABELS, score, sample_weight=w[i])))
        if (b+1) % 250 == 0:
            print(f"Paired bootstrap {b+1}/{args.repetitions}", flush=True)
    assert bootstrap_sklearn_error < 1e-12
    assert np.max(abs(boot[:, :, 3]+boot[:, :, 4]-boot[:, :, 5])) < 1e-12
    np.savez_compressed(out / "bootstrap_draws.npz", values=boot, scopes=np.array(scopes), metrics=np.array(METRICS))
    intervals, sensitivity = [], []
    for j, scope in enumerate(scopes):
        mask = masks[scope]
        estimate = point[mask].mean(axis=0)
        for k, metric in enumerate(METRICS):
            lo, hi = np.quantile(boot[:, j, k], [.025, .975])
            intervals.append(dict(scope=scope, metric=metric, estimate=float(estimate[k]), lower=float(lo), upper=float(hi)))
        for seed in sorted(metadata.run_seed.unique()):
            selected = mask & (metadata.run_seed.to_numpy() == seed)
            sensitivity.append(dict(scope=scope, run_seed=int(seed), **dict(zip(METRICS, point[selected].mean(axis=0).tolist()))))
    pd.DataFrame(intervals).to_csv(out / "decomposition_intervals.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(out / "seed_sensitivity.csv", index=False)
    rank_cols = [c for c in rank_df if c not in KEY]
    rank_df.groupby(["heldout_generator", "condition"], as_index=False)[rank_cols].mean().to_csv(out / "ranking_cell_means.csv", index=False)
    by_seed.to_csv(out / "decomposition_seed_cells.csv", index=False)

    summary = {"status": "completed", "source_prediction_rows": original_rows,
               "selected_prediction_rows": len(frame), "quality_gate_instances": 72,
               "unique_originals": 900, "data_seeds": 3, "gate_initializations_per_data_seed": 3,
               "heldout_generators": 2, "processing_conditions": 4,
               "bootstrap_repetitions": args.repetitions, "intervals": intervals,
               "seed_sensitivity": sensitivity, "models_fitted": 0, "final_test_accessed": False,
               "identity_sampling": "Three independent class/source pools, same sampled IDs paired everywhere; Real shared across generator targets.",
               "diagnostic_only": "Target-cell average weights are recomputed in each bootstrap, never selected for deployment.",
               "interpretation": "Signed effects: negative means worse than the preceding reference. Mean preference and image-specific effects sum exactly to the total quality-gate effect. These are descriptive decompositions, not independently manipulated causal effects.",
               "seconds": time.monotonic()-started}
    write_json(out / "summary.json", summary)
    verification = {"status": "passed", "max_logit_reconstruction_error": max_logit_error,
                    "max_point_sklearn_error_including_ties": max_sklearn,
                    "max_bootstrap_sklearn_error_648_checks": bootstrap_sklearn_error,
                    "max_archived_constant_metrics_error": archive_const_error,
                    "max_archived_rank_metrics_error": archive_rank_error,
                    "max_archived_study_metrics_error": max(metric_errors),
                    "max_pair_rank_identity_error": rank_identity,
                    "max_decomposition_identity_error": decomposition_identity,
                    "source_hashes_unchanged": all(sha(ROOT/name) == h for name, h in hashes.items()),
                    "models_fitted": 0, "final_test_accessed": False}
    assert verification["source_hashes_unchanged"]
    write_json(out / "verification.json", verification)
    table = pd.DataFrame(intervals)
    lines = ["冻结质量门控的损失分解与配对不确定性补充分析", "",
             "新增工作：对全部生成器和处理条件重建已有损失分解，并补充1000次原图配对bootstrap及三个数据seed敏感性。没有训练新模型。",
             "定义：总体偏好效应=目标单元平均权重融合−等权平均；逐图调整效应=实际门控−目标单元平均权重融合；总效应=实际门控−等权平均。负值代表排序损失。", ""]
    for scope in scopes:
        lines.append(scope)
        for metric in ["mean_preference_effect", "image_specific_effect", "total_effect"]:
            r = table[(table.scope == scope) & (table.metric == metric)].iloc[0]
            lines.append(f"  {metric}: {r.estimate:+.6f} [{r.lower:+.6f}, {r.upper:+.6f}]")
    lines += ["", "平均权重由每次重采样后的600张目标图重新计算。该目标依赖量仅用于诊断，不是一个可部署、经源域选定的候选模型。",
              "区间条件于已有模型；三个数据seed共享数据池和开发图。完整条件保留，未进行多重比较校正。90000个真假排序对不视为90000个独立样本。",
              "点估计与历史诊断、原研究指标一致；加权AUROC已用sklearn核对原始数据、有并列分数样例及前三次完整bootstrap。"]
    (out / "report_zh.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    write_json(out / "complete.json", {"status": "completed", "outputs_sha256": {
        p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "complete.json"},
        "verification_status": "passed", "models_fitted": 0, "no_final_test_access": True})
    print(json.dumps({"status": "passed", "output": str(out), "seconds": summary["seconds"]}))


if __name__ == "__main__":
    main()
