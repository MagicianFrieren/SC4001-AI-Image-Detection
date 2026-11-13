"""Render report figures from saved development aggregates; no fitting or inference.

Run from any directory with the repository's Python environment. Source studies are
read-only. Each panel retains its own protocol/reference, and every data seed is
shown. Figure 2 includes all eight development conditions in the ranking audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "output/analysis"
OUT = BASE / "report_evidence_20261111/figures"
SEEDS = [20261101, 20261102, 20261103]
CONDITIONS = ["clean", "jpeg70", "jpeg50", "resize05"]
SOURCES = {
    "diagnostic_methods": BASE / "gate_diagnosis_20261108/method_comparison.csv",
    "multigen_cells": BASE / "multigen_gate_20261108/run_v1_codefix1/metrics_seed_condition.csv",
    "multigen_summary": BASE / "multigen_gate_20261108/run_v1_codefix1/summary.json",
    "stability_cells": BASE / "recompression_reliability_20261108/reliability_cells.csv",
    "stability_detection": BASE / "recompression_reliability_20261108/detection_diagnostics.csv",
    "stability_summary": BASE / "recompression_reliability_20261108/summary.json",
    "mixed_training": BASE / "paired_view_global_20261108/metrics.csv",
    "transfer": BASE / "gate_failure_mechanism_20261108/source_target_transfer.csv",
    "ranking": BASE / "gate_failure_mechanism_20261108/prediction_ranking_seed_means.csv",
    "constant_alpha": BASE / "gate_failure_mechanism_20261108/prediction_constant_seed_means.csv",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .7, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})
MARKERS = ["o", "s", "^"]
SHADES = ["0.12", "0.43", "0.68"]
checks = {}


def close(name, actual, expected, tolerance=2e-12):
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    assert error < tolerance, (name, error)
    checks[name] = {"status": "passed", "max_abs_error": error}


def macro_cells(frame):
    counts = frame.groupby("run_seed").size()
    assert counts.to_dict() == {seed: 8 for seed in SEEDS}, counts.to_dict()
    return frame.groupby("run_seed")["auroc"].mean().reindex(SEEDS).to_numpy()


def prepare_improvements():
    rows = []

    def add(panel, method, label, vals, refs, reference, scope="2 generators x 4 conditions"):
        for seed, val, ref in zip(SEEDS, vals, refs):
            rows.append(dict(panel=panel, method=method, label=label.replace("\n", " "),
                             run_seed=seed, detection_auroc=float(val),
                             reference_detection_auroc=float(ref),
                             delta_detection_auroc=float(val-ref), reference=reference, scope=scope))

    d = pd.read_csv(SOURCES["diagnostic_methods"]).set_index("method")
    ref = d.loc["average", ["seed1", "seed2", "seed3"]].to_numpy(float)
    a = [
        ("original_quality_gate", "BCE\ncheckpoint"),
        ("auc_checkpoint_quality_gate", "AUROC\ncheckpoint"),
        ("bounded_ranking_quality_gate", "Bounded +\nranking"),
        ("original_view_quality_gate", "Original\nviews"),
        ("quality_condition_gate", "Condition\nrouting"),
    ]
    for method, label in a:
        vals = d.loc[method, ["seed1", "seed2", "seed3"]].to_numpy(float)
        close(f"diagnostic_{method}_mean", vals.mean(), d.loc[method, "mean_auroc"])
        add("A", method, label, vals, ref, "simple average fusion")

    m = pd.read_csv(SOURCES["multigen_cells"])
    summary = json.loads(SOURCES["multigen_summary"].read_text())
    b = []
    for arm, label_prefix in [("source_only", "Single-source"), ("mixed", "Multi-source")]:
        mask = m.scenario.eq("source_only") if arm == "source_only" else m.scenario.str.startswith("mixed_")
        ref = macro_cells(m[mask & m.model.eq("average") & m.checkpoint.eq("fixed")])
        for model, suffix in [("score_gate", "score"), ("quality_gate", "quality")]:
            method = f"{arm}_{model}"
            label = label_prefix + "\n" + suffix
            vals = macro_cells(m[mask & m.model.eq(model) & m.checkpoint.eq("auroc")])
            target = next(x for x in summary["macro"] if x["arm"] == arm and x["model"] == model and x["checkpoint"] == "auroc")
            close(f"multigen_{method}_macro", vals.mean(), target["auroc"])
            add("B", method, label, vals, ref, "simple average fusion (this protocol)")
            b.append((method, label))

    s = pd.read_csv(SOURCES["stability_detection"])
    s = s[s.scope.eq("development")]
    assert s.groupby("run_seed").size().to_dict() == {seed: 8 for seed in SEEDS}
    ref = s.groupby("run_seed").average_fusion_auroc.mean().reindex(SEEDS).to_numpy()
    c = [("global_auroc", "Global\nexpert"), ("stable_selected_logit_auroc", "Stability\nselector")]
    for method, label in c:
        vals = s.groupby("run_seed")[method].mean().reindex(SEEDS).to_numpy()
        add("C", method, label, vals, ref, "simple average fusion")

    reliability = pd.read_csv(SOURCES["stability_cells"])
    reliability = reliability[reliability.scope.eq("development")].copy()
    assert len(reliability) == 36
    reliability["group_weight"] = reliability.generator.map({"Real": .5, "ADM": .25, "BigGAN": .25})
    reliability["weighted"] = reliability.stability_scaled_auroc * reliability.group_weight
    rel = reliability.groupby(["run_seed", "condition"]).weighted.sum().mean()
    reliability_summary = json.loads(SOURCES["stability_summary"].read_text())
    close("reliability_primary_reconstruction", rel, reliability_summary["primary_macro"]["stability_scaled"])

    p = pd.read_csv(SOURCES["mixed_training"])
    p = p[p.partition.eq("development")]
    scopes = [
        ("all", "All eight\nconditions", p.index),
        ("jpeg", "JPEG\naverage", p[p.condition.isin(["jpeg70", "jpeg50"])].index),
        ("clean_ADM", "ADM\nclean", p[p.condition.eq("clean") & p.generator.eq("ADM")].index),
        ("clean_BigGAN", "BigGAN\nclean", p[p.condition.eq("clean") & p.generator.eq("BigGAN")].index),
    ]
    dd = []
    for method, label, ix in scopes:
        sub = p.loc[ix]
        avg = sub.groupby(["seed", "arm"]).auroc.mean().unstack("arm").reindex(SEEDS)
        add("D", method, label, avg.paired_half.to_numpy(), avg.matched.to_numpy(),
            "format-matched global expert", scope=method)
        dd.append((method, label))
    close("mixed_training_all_delta", np.mean([r["delta_detection_auroc"] for r in rows if r["panel"] == "D" and r["method"] == "all"]), -.0647662037037037)
    return pd.DataFrame(rows), {"A": a, "B": b, "C": c, "D": dd}, reliability_summary


def draw_seed_panel(ax, frame, panel, methods, title, reference, ylim):
    labels = [label for _, label in methods]
    x = np.arange(len(methods))
    values = np.array([frame[(frame.panel == panel) & (frame.method == method)].sort_values("run_seed").delta_detection_auroc.to_numpy() for method, _ in methods])
    assert values.shape == (len(methods), 3)
    ax.axhline(0, color="0.3", lw=.9, ls="--", zorder=0)
    ax.grid(axis="y", color="0.9", lw=.6, zorder=0)
    for si in range(3):
        ax.plot(x + (si-1)*.13, values[:, si], marker=MARKERS[si], color=SHADES[si],
                lw=.6, ms=4.8, zorder=3, alpha=.9)
    for xx, mean in zip(x, values.mean(axis=1)):
        ax.plot([xx-.23, xx+.23], [mean, mean], color="black", lw=2, zorder=4)
    ax.set_xticks(x, labels)
    ax.set_ylim(*ylim)
    ax.set_xlim(-.5, len(methods)-.5)
    ax.set_title(title, loc="left", weight="bold", pad=20)
    ax.text(0, 1.025, "Reference: " + reference, transform=ax.transAxes, fontsize=9, color="0.3")
    ax.set_ylabel("Δ detection AUROC")
    decimals = 3 if max(abs(ylim[0]), abs(ylim[1])) < .04 else 2
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:+.{decimals}f}" if abs(val) > 1e-10 else f"{0:.{decimals}f}"))


def save(fig, stem):
    for suffix in ["png", "pdf"]:
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=300, bbox_inches="tight", pad_inches=.08)
    plt.close(fig)


def figure_improvements(data, methods, reliability):
    fig, axs = plt.subplots(2, 2, figsize=(9.2, 7.8))
    fig.subplots_adjust(left=.09, right=.985, top=.845, bottom=.095, hspace=.60, wspace=.29)
    fig.suptitle("Targeted improvements: gains, recoveries and remaining trade-offs", x=.09, ha="left", y=.985, fontsize=13, weight="bold")
    handles = [Line2D([], [], color=SHADES[i], marker=MARKERS[i], lw=.6, label=f"Data seed {i+1}") for i in range(3)]
    handles.append(Line2D([], [], color="black", lw=2, label="Mean of seeds"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.085, .958), frameon=False, ncol=4, fontsize=9)
    draw_seed_panel(axs[0, 0], data, "A", methods["A"], "A  Gate optimization and input recipes", "average fusion", (-.032, .007))
    draw_seed_panel(axs[0, 1], data, "B", methods["B"], "B  Generator coverage", "average fusion; same new protocol", (-.011, .007))
    draw_seed_panel(axs[1, 0], data, "C", methods["C"], "C  Recompression-based selection", "average fusion", (-.034, .020))
    ci = reliability["bootstrap"]["primary"]["conditional_percentile_95"]
    axs[1, 0].text(.04, .04, "Separate reliability prediction:\n" + f"AUROC {reliability['primary_macro']['stability_scaled']:.4f}\n95% conditional CI [{ci[0]:.4f}, {ci[1]:.4f}]", transform=axs[1, 0].transAxes,
                   va="bottom", fontsize=8.5, bbox=dict(facecolor="white", edgecolor="0.75", linewidth=.5, pad=3))
    draw_seed_panel(axs[1, 1], data, "D", methods["D"], "D  1:1 training-view mixture", "format-matched global expert", (-.15, .09))
    fig.text(.09, .017, "Development evidence. Protocols are separated; seeds share data. Lines show paired runs, not independent replications.", fontsize=8.3, color="0.25")
    save(fig, "figure1_improvement_attempts")


def figure_mechanism():
    t = pd.read_csv(SOURCES["transfer"])
    t = t[t.condition.eq("jpeg70")].copy()
    assert len(t) == 4
    t["global_minus_local_auroc"] = t.global_auroc - t.local_auroc
    r = pd.read_csv(SOURCES["ranking"])
    r = r[r.reference.eq("average")].copy()
    assert len(r) == 24
    close("pair_credit_identity_all_seed_cells", r.repaired_pair_fraction-r.lost_pair_fraction, r.delta_auroc)
    c = pd.read_csv(SOURCES["constant_alpha"])
    c = c[c.model.eq("quality_gate")]
    join = r.merge(c, on=["run_seed", "scenario", "heldout_generator", "condition"], validate="one_to_one")
    close("actual_gate_auroc_cross_file", join.quality_auroc, join.actual_auroc)
    close("decomposition_cross_file", join.delta_auroc, join.constant_minus_average_auroc - join.constant_minus_actual_auroc)
    mean = r.groupby(["heldout_generator", "condition"], as_index=False)[["repaired_pair_fraction", "lost_pair_fraction", "delta_auroc"]].mean()
    order = [(gen, cond) for gen in ["ADM", "BigGAN"] for cond in CONDITIONS]
    mean = mean.set_index(["heldout_generator", "condition"]).loc[order].reset_index()
    t.to_csv(OUT / "figure2_transfer_data.csv", index=False, float_format="%.16g")
    r.to_csv(OUT / "figure2_pair_credit_seed_data.csv", index=False, float_format="%.16g")
    mean.to_csv(OUT / "figure2_pair_credit_mean_data.csv", index=False, float_format="%.16g")
    fig = plt.figure(figsize=(9.2, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], left=.09, right=.98, top=.865, bottom=.105, hspace=.58, wspace=.3)
    aa, ab, ac = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[1,:])
    fig.suptitle("Where quality gating loses useful expert complementarity", x=.09, ha="left", y=.985, fontsize=13, weight="bold")
    handles = []
    for donor, target, shade, marker, ls in [("ADM", "BigGAN", "0.1", "o", "-"), ("BigGAN", "ADM", "0.55", "s", "--")]:
        sub = t[t.donor.eq(donor)].set_index("domain").loc[["donor_tuning", "heldout_development"]]
        for ax, column in [(aa, "global_minus_local_auroc"), (ab, "quality_gate_alpha_global")]:
            vals = sub[column].to_numpy()
            ax.plot([0,1], vals, color=shade, marker=marker, ls=ls, ms=6, lw=1.4)
            for xx, val in enumerate(vals):
                text = f"{val:+.3f}" if ax is aa else f"{val*100:.1f}%"
                offset = (0, 9 if donor == "ADM" else -16)
                ax.annotate(text, (xx,val), xytext=offset, textcoords="offset points", ha="center", fontsize=9)
        handles.append(Line2D([], [], color=shade, marker=marker, ls=ls, label=f"{donor} source → {target} target"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.085,.953), frameon=False, ncol=2, fontsize=9)
    for ax in [aa, ab]:
        ax.set_xlim(-.25,1.25)
        ax.set_xticks([0,1], ["Source selection", "Held-out generator"])
        ax.grid(axis="y", color="0.9", lw=.6)
    aa.set_title("A  JPEG70: expert advantage reverses", loc="left", weight="bold", pad=14)
    aa.set_ylabel("Global AUROC − local AUROC")
    aa.axhline(0, color="0.3", lw=.8, ls=":")
    aa.set_ylim(-.1,.1)
    aa.text(.02,.97,"Global stronger", transform=aa.transAxes, va="top", fontsize=8.5, color="0.35")
    aa.text(.65,.03,"Local stronger", transform=aa.transAxes, va="bottom", fontsize=8.5, color="0.35")
    ab.set_title("B  JPEG70: source preference persists", loc="left", weight="bold", pad=14)
    ab.set_ylabel("Mean gate weight on global expert")
    ab.axhline(.5, color="0.3", lw=.8, ls=":")
    ab.set_ylim(.2,.9)
    ab.yaxis.set_major_formatter(FuncFormatter(lambda val,_: f"{val:.0%}"))
    x = np.arange(8)
    ac.bar(x, mean.repaired_pair_fraction*100, color="0.8", edgecolor="0.35", lw=.6, width=.58, label="Repaired pair-order credit")
    ac.bar(x, -mean.lost_pair_fraction*100, color="0.35", edgecolor="0.2", lw=.6, width=.58, label="Newly lost pair-order credit")
    ac.scatter(x, mean.delta_auroc*100, marker="D", color="black", s=28, zorder=5, label="Net change (100 × ΔAUROC)")
    ac.axhline(0,color="0.1",lw=.8)
    ac.axvline(3.5,color="0.75",lw=.6)
    ac.set_xticks(x, ["Clean", "JPEG70", "JPEG50", "Resize 0.5"]*2)
    ac.text(1.5,-.16,"ADM", transform=ac.get_xaxis_transform(), ha="center", weight="bold")
    ac.text(5.5,-.16,"BigGAN", transform=ac.get_xaxis_transform(), ha="center", weight="bold")
    ac.set_ylabel("Pair-order credit change\n(percentage points)")
    ac.set_title("C  All conditions: repaired versus newly damaged rankings", loc="left", weight="bold", pad=37)
    ac.legend(loc="lower left", bbox_to_anchor=(0,1.01), frameon=False, ncol=3, fontsize=8.5, handlelength=1.3, columnspacing=1.2)
    ac.set_ylim(-6.2,7.0)
    ac.grid(axis="y", color="0.92", lw=.6)
    ac.set_axisbelow(True)
    fig.text(.09,.016,"Means over three data seeds and three gate initializations. Pair counts share images; they are not independent trials.", fontsize=8.3,color="0.25")
    save(fig, "figure2_failure_mechanism")


CAPTIONS = """Figure 1. Targeted improvement attempts on development data.
Points show paired AUROC differences for the three data seeds (20261101/2/3); horizontal black marks show their mean. The same marker identifies a seed within each panel. Panel A compares the archived gate optimization and input-recipe candidates with simple average fusion. Panel B is the separately frozen multi-generator protocol: its single-source controls were rerun under that protocol and must not be treated as the checkpoint replay in A. Each gate cell averages three gate initializations before averaging two target generators and four conditions. Panel C reports actual fake/real detection after selecting the more stable expert, with the global expert as an additional reference. Its inset is a distinct task: predicting which expert has lower per-image BCE. That reliability AUROC is class/condition-stratified with Real/ADM/BigGAN weights 0.5/0.25/0.25; its conditional interval is taken from the original 1,000-draw paired image bootstrap. Panel D reports the 1:1 original/format-matched global-training recipe relative to the stronger format-matched global expert, for the full development macro, JPEG macro, and both clean conditions. Its four scopes overlap and are not independent outcomes. All plotted differences retain their own protocol and reference, rather than forming a cross-protocol leaderboard. Lines and seed spread describe paired runs; the shared image pool means these are not three independent datasets.

Figure 2. Source preference mismatch and the accounting of ranking losses.
Panels A and B show JPEG70 donor-only source-selection and held-out-generator development summaries for the same frozen multi-source quality gates. The stronger single expert switches with generator, while the mean global weight remains on the same side of 0.5. The mean alpha is descriptive: source-selection cells contain 124 real and 62 donor fake images, whereas target cells contain 300 real and 300 fake images, so its magnitude also reflects different class composition. The plot does not infer every image's optimal weight from single-expert AUROC. Panel C uses all eight target-generator/condition cells, not only JPEG. Positive bars are repaired pair-order credit, negative bars are newly lost credit, and diamonds are their net difference relative to equal fusion. Correct ordering has credit 1, ties 0.5 and wrong ordering 0; lost/repaired credit includes half-credit transitions. The identity repairs minus losses = delta AUROC is checked in every seed/cell. Each cell has 300 x 300 real/fake pairs; these pairs share images and must not be counted as 90,000 independent observations. Cell means first average the three gate initializations and then the three data seeds. The JPEG cases are motivated by the previously reported failure pattern; all-condition results retain the clean and resize counterexamples.

Scope and interpretation. Both figures use saved, previously inspected development results, not a new blind test. They report model behavior and targeted interventions, rather than establishing a unique pixel-level cause. The plotting script reads aggregate/seed-cell CSVs only and never reads final-test images, final-test per-image predictions, model weights, or feature caches. It performs no fitting, tuning, resampling or inference. Raw numeric plotting tables and SHA-256 provenance accompany the figures.
"""


def main():
    global ROOT, BASE, OUT, SOURCES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Source root retaining repository-relative input paths")
    parser.add_argument("--output", type=Path, help="Output figure directory; defaults below source root")
    args = parser.parse_args()
    source_relatives = {name: path.relative_to(ROOT) for name, path in SOURCES.items()}
    ROOT = args.root.resolve()
    BASE = ROOT / "output/analysis"
    OUT = args.output.resolve() if args.output else BASE / "report_evidence_20261111/figures"
    SOURCES = {name: ROOT / path for name, path in source_relatives.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in SOURCES.values()}
    data, methods, reliability = prepare_improvements()
    data.to_csv(OUT / "figure1_seed_differences.csv", index=False, float_format="%.16g")
    figure_improvements(data, methods, reliability)
    figure_mechanism()
    (OUT / "captions_en.txt").write_text(CAPTIONS, encoding="utf-8")
    output_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "figure_verification.json"}
    manifest = dict(status="passed", operation="aggregate reconstruction and figure rendering only", source_sha256=hashes,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), checks=checks,
                    input_files_unchanged=all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest for path,digest in hashes.items()),
                    output_sha256=output_hashes, no_model_fitting=True, no_final_test_access=True,
                    visual_qa="Requires visual inspection of the generated PNGs; not asserted by this script.")
    assert manifest["input_files_unchanged"]
    (OUT / "figure_verification.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status":"passed","output":str(OUT),"checks":len(checks),"figure1_rows":len(data)}))


if __name__ == "__main__":
    main()
