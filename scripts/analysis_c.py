"""Task C: failure cases, W0 vs W1 disagreement and per-generator plots from the saved Round 8 test predictions. Writes results/analysis_c/*."""
import hashlib
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

FIG_W = 6.5
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 9.5, "axes.labelsize": 9.5,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
                     "figure.titlesize": 10})

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "results/round8_final/evaluation/predictions"
OUT = ROOT / "results/analysis_c"

CONDITIONS = ["clean", "jpeg70", "jpeg50", "resize05"]
COND_LABEL = {"clean": "Clean", "jpeg70": "JPEG70", "jpeg50": "JPEG50", "resize05": "Resize 0.5x"}
COND_SHORT = {"clean": "Clean", "jpeg70": "JPEG70", "jpeg50": "JPEG50", "resize05": "Resize"}
MAIN_MACRO = ["GLIDE", "Midjourney", "VQDM", "Wukong"]
DEV_GENERATORS = ["ADM", "BigGAN"]
GENERATORS = MAIN_MACRO + DEV_GENERATORS
SOURCES = ["Real"] + GENERATORS
SOURCE_TICK = {s: f"{s} (dev)" if s in DEV_GENERATORS else s for s in SOURCES}
SEEDS = [1, 2, 3]
SEED_ID = {1: 20261101, 2: 20261102, 3: 20261103}

TASKS = {1: ("task02", "task03", "task01", "task10"),
         2: ("task05", "task06", "task04", "task12"),
         3: ("task08", "task09", "task07", "task14")}
STYLE = {"W0": ("tab:blue", "o"), "W1": ("tab:orange", "s"), "Local": ("tab:brown", "v"),
         "R7 average": ("tab:green", "^"), "ResNet-50": ("tab:gray", "x"),
         "R8A matched avg": ("tab:purple", "D")}
METRIC_KEYS = {"W0": ("round7", "saved", "global"), "W1": ("round7", "saved", "W1"),
               "Local": ("round7", "saved", "local"), "R7 average": ("round7", "saved", "average"),
               "ResNet-50": ("round7", "saved", "resnet50"),
               "R8A matched avg": ("round8_a", "matched", "average")}


def model_files(seed):
    c, d, r, m = TASKS[seed]
    s = f"s{seed}"
    return {"W0": (c, f"R7_C_{s}_original_raw_global_-1"),
            "W1": (d, f"R7_D_{s}_original_raw_W1"),
            "Local": (c, f"R7_C_{s}_original_raw_local_-1"),
            "R7 average": (c, f"R7_C_{s}_original_raw_average_-1"),
            "ResNet-50": (r, f"R7_C_{s}_original_raw_resnet50"),
            "R8A matched avg": (m, f"R8A_{s}_matched_average_-1")}




def load_predictions():
    cache, frames = {}, []
    for seed in SEEDS:
        for name, (task, model_id) in model_files(seed).items():
            for cond in CONDITIONS:
                path = PRED_DIR / f"{task}_{cond}.csv"
                if path not in cache:
                    cache[path] = pd.read_csv(path)
                d = cache[path][cache[path].model_id == model_id].copy()
                assert len(d) == 3500, (model_id, cond, len(d))
                d["p_fake"] = expit(d["logit"].astype(np.float64))
                d["pred"] = (d["p_fake"] >= d["threshold"]).astype(int)
                d["model"], d["seed"] = name, seed
                frames.append(d[["model", "seed", "condition", "generator", "source_id",
                                 "label", "logit", "p_fake", "threshold", "pred"]])
    df = pd.concat(frames, ignore_index=True)
    df["correct"] = df["pred"] == df["label"]
    return df


def pair(df, a, b, cols):
    key = ["seed", "condition", "source_id"]
    left = df[df.model == a].set_index(key)[["generator", "label"] + cols]
    right = df[df.model == b].set_index(key)[cols]
    return left.join(right, lsuffix="_a", rsuffix="_b").reset_index()


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, dpi=200); plt.close(fig)


def per_generator_auroc(df):
    rows = []
    for (model, seed, cond), g in df.groupby(["model", "seed", "condition"]):
        real = g[g.generator == "Real"]
        for gen in GENERATORS:
            both = pd.concat([real, g[g.generator == gen]])
            rows.append({"model": model, "seed": seed, "condition": cond, "generator": gen,
                         "auroc": roc_auc_score(both["label"], both["p_fake"])})
    return pd.DataFrame(rows)


def cross_check(df, auc):
    ref = pd.read_csv(ROOT / "results/round8_final/metrics_mean_sd.csv")
    ref = ref[ref.metric == "auroc"]
    worst = 0.0
    for (model, cond, gen), value in auc.groupby(["model", "condition", "generator"])["auroc"].mean().items():
        cohort, arm, name = METRIC_KEYS[model]
        r = ref[(ref.cohort == cohort) & (ref.arm == arm) & (ref.model == name)
                & (ref.condition == cond) & (ref.generator == gen)]["mean"]
        assert len(r) == 1
        worst = max(worst, abs(r.iloc[0] - value))
    each = pd.read_csv(ROOT / "results/round8_final/metrics_each_seed.csv")
    for (model, seed, cond), g in df.groupby(["model", "seed", "condition"]):
        cohort, arm, name = METRIC_KEYS[model]
        fpr = g.loc[g.generator == "Real", "pred"].mean()
        for gen in GENERATORS:
            tpr = g.loc[g.generator == gen, "pred"].mean()
            r = each[(each.cohort == cohort) & (each.arm == arm) & (each.model == name)
                     & (each.seed == SEED_ID[seed]) & (each.condition == cond) & (each.generator == gen)]
            assert len(r) == 1
            worst = max(worst, abs(r["balanced_accuracy"].iloc[0] - (tpr + 1 - fpr) / 2),
                        abs(r["real_false_positive_rate"].iloc[0] - fpr))
    assert worst < 1e-6, f"Mismatch with the team's Round 8 metrics: {worst}"
    return worst


def fig1_per_generator(auc):
    summary = auc.groupby(["model", "generator", "condition"])["auroc"].agg(["mean", "std"]).reset_index()
    summary.to_csv(OUT / "per_generator_auroc.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(FIG_W, 5.2), sharey=True)
    x = np.arange(len(CONDITIONS))
    for ax, gen in zip(axes.flat, GENERATORS):
        for model in ["W0", "W1", "R7 average", "R8A matched avg"]:
            colour, marker = STYLE[model]
            s = summary[(summary.model == model) & (summary.generator == gen)].set_index("condition").loc[CONDITIONS]
            ax.errorbar(x, s["mean"], yerr=s["std"], color=colour, marker=marker, ms=4, capsize=2, label=model)
        ax.axhline(0.5, color="grey", ls=":", lw=1)
        ax.set_title(f"{gen} ({'main macro' if gen in MAIN_MACRO else 'development'})")
        ax.set_xticks(x, [COND_SHORT[c] for c in CONDITIONS], rotation=35)
        ax.set_ylim(0.3, 1.0); ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("AUROC")
    axes[0, 0].legend(loc="lower left")
    fig.suptitle("AUROC per generator, Round 8 test (mean ± SD, 3 seeds; dotted = chance)")
    save(fig, "fig1_per_generator_auroc.png")


def outcome_rates(p, by):
    p = p.assign(both_right=p.correct_a & p.correct_b, only_a=p.correct_a & ~p.correct_b,
                 only_b=~p.correct_a & p.correct_b, both_wrong=~p.correct_a & ~p.correct_b)
    cols = ["both_right", "only_a", "only_b", "both_wrong"]
    per_seed = p.groupby(["seed"] + by)[cols].mean().groupby(by)
    return per_seed.mean().join(per_seed.std(ddof=1), rsuffix="_sd").reset_index()


def bars_only_one(table, label_a, label_b, colours, title, name):
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, 5.6), sharey=True)
    x = np.arange(len(SOURCES))
    for ax, cond in zip(axes.flat, CONDITIONS):
        t = table[table.condition == cond].set_index("generator").loc[SOURCES]
        for col, shift, label, colour in [("only_a", -0.2, label_a, colours[0]),
                                          ("only_b", 0.2, label_b, colours[1])]:
            mean, sd = 100 * t[col], 100 * t[col + "_sd"]
            ax.bar(x + shift, mean, 0.4, yerr=[np.minimum(sd, mean), sd], capsize=2, label=label, color=colour)
        ax.set_title(COND_LABEL[cond])
        ax.set_xticks(x, [SOURCE_TICK[s] for s in SOURCES], rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("% of images")
    axes[0, 0].legend()
    fig.suptitle(title)
    save(fig, name)


def fig2_w0_w1(df):
    p = pair(df, "W0", "W1", ["correct"])
    table = outcome_rates(p, ["condition", "generator"])
    bars_only_one(table, "Only W0 correct", "Only W1 correct", ("tab:blue", "tab:orange"),
                  "W0 vs W1: images where exactly one is correct (mean ± SD, 3 seeds)",
                  "fig2_w0_w1_disagreement.png")
    table.columns = [c.replace("only_a", "only_w0_right").replace("only_b", "only_w1_right") for c in table.columns]
    table["disagree"] = table["only_w0_right"] + table["only_w1_right"]
    table.to_csv(OUT / "w0_w1_disagreement.csv", index=False)
    return table


def fig3_failures(df):
    fusion = df[df.model == "R7 average"]
    wrong = fusion.assign(wrong=~fusion.correct)
    err = (wrong.groupby(["seed", "condition", "generator"])["wrong"].mean()
           .groupby(["condition", "generator"]).mean().unstack("condition")[CONDITIONS].loc[SOURCES])
    err.to_csv(OUT / "fusion_error_rates.csv")

    wrong_all = wrong.groupby(["condition", "generator", "source_id"])["wrong"].all()
    persistent = wrong_all[wrong_all].reset_index()
    counts = (persistent.groupby(["generator", "condition"]).size().unstack("condition")
              .reindex(index=SOURCES, columns=CONDITIONS).fillna(0).astype(int))
    counts.to_csv(OUT / "persistent_failures_counts.csv")
    n_cond = persistent.groupby(["generator", "source_id"]).size()
    always = n_cond[n_cond == len(CONDITIONS)].reset_index()[["generator", "source_id"]]
    always.to_csv(OUT / "always_wrong_images.csv", index=False)

    thresholds = fusion.groupby("seed")["threshold"].first()
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, 5.9), sharey=True)
    for ax, cond in zip(axes.flat, CONDITIONS):
        c = fusion[fusion.condition == cond]
        ax.boxplot([c[c.generator == s]["p_fake"] for s in SOURCES], showfliers=False)
        for seed, ls in zip(SEEDS, ["--", "-.", ":"]):
            ax.axhline(thresholds[seed], color="red", ls=ls, lw=1.2,
                       label=f"seed {seed} threshold ({thresholds[seed]:.2f})")
        ax.set_title(COND_LABEL[cond])
        ax.set_xticks(range(1, len(SOURCES) + 1), [SOURCE_TICK[s] for s in SOURCES], rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("P(fake)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("R7 average fusion: P(fake) per source vs saved thresholds (3 seeds pooled)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "fig3_failure_score_distributions.png", dpi=200); plt.close(fig)
    return err, counts, always


def hard_for_everyone(df):
    clean = df[df.condition == "clean"]
    wrong = clean.assign(wrong=~clean.correct).groupby(["generator", "source_id"])["wrong"].all()
    hard = wrong[wrong].reset_index()[["generator", "source_id"]]
    hard.to_csv(OUT / "clean_images_all_models_wrong.csv", index=False)
    return hard


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fig4_examples(hard, per_source=4, shard_dir=None):
    inv = json.loads((ROOT / "results/round7/source_inventory.json").read_text())
    if shard_dir is None:
        shard_dir = ROOT / "artifacts/round7/source/data/shards"
    hard = hard.assign(row=hard["source_id"].str.split(":").str[1].astype(int)).sort_values("row")
    hard["shard"] = hard["row"] // 2000
    needed = sorted(hard.groupby("generator").head(per_source)["shard"].unique().tolist())
    present = {}
    for k, s in enumerate(inv["shards"]):
        path = Path(shard_dir) / Path(s["path"]).name
        if path.exists():
            if sha256(path) == s["verified_sha256"]:
                present[k] = path
            else:
                print(f"fig4: {path.name} skipped, SHA-256 does not match source_inventory.json")
    missing = [k for k in needed if k not in present]
    print(f"fig4: shards needed for full coverage {needed}; present {sorted(present)}; missing {missing}")
    if not present:
        print(f"fig4 skipped: no verified Tiny-GenImage shards in {shard_dir}")
        return False

    import pyarrow.parquet as pq
    from PIL import Image
    picks = hard[hard["shard"].isin(present)].groupby("generator").head(per_source)
    gens = [g for g in SOURCES if g in set(picks.generator)]
    panel = FIG_W / len(gens)
    fig, axes = plt.subplots(per_source, len(gens), figsize=(FIG_W, panel * 1.45 * per_source + 0.6), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    tables = {}
    for j, gen in enumerate(gens):
        for i, row in enumerate(picks[picks.generator == gen]["row"]):
            shard = row // 2000
            if shard not in tables:
                tables[shard] = pq.read_table(present[shard], columns=["image"]).column("image")
            img = Image.open(io.BytesIO(tables[shard][row % 2000].as_py()["bytes"])).convert("RGB")
            axes[i, j].imshow(img)
            label = f"train:{row}\n{img.size[0]}x{img.size[1]}"
            axes[i, j].set_title(f"{SOURCE_TICK[gen]}\n{label}" if i == 0 else label, fontsize=8.5)
    fig.suptitle("Clean images all six models get wrong in all 3 seeds\n"
                 f"(first {per_source} per source by image index; not curated)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig4_hard_examples.png", dpi=200); plt.close(fig)
    return True


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    df = load_predictions()
    auc = per_generator_auroc(df)
    print(f"Cross-check vs results/round8_final/metrics_mean_sd.csv (AUROC) and metrics_each_seed.csv "
          f"(balanced accuracy, real FPR) passed (max |diff| = {cross_check(df, auc):.1e})")
    fig1_per_generator(auc)
    w01 = fig2_w0_w1(df)
    err, counts, always = fig3_failures(df)
    hard = hard_for_everyone(df)
    fig4_examples(hard)

    print("\nW0 vs W1 (mean of 3 seeds):\n", w01.round(3).to_string(index=False))
    print("\nR7 average error rate:\n", err.round(3))
    print("\nR7 average: wrong in all 3 seeds:\n", counts)
    print("\nWrong in all seeds and all conditions:", always.groupby("generator").size().to_dict())
    print("\nClean images all models wrong:", hard.groupby("generator").size().to_dict())

if __name__ == "__main__":
    main()
