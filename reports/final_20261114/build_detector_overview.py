"""Rebuild the report's detector overview; no model or image data required.

Usage: python build_detector_overview.py [--output-dir DIRECTORY]
The editable layout uses the original 1692:734 canvas proportions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath
import matplotlib.patheffects as pe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 6.5,
        # Outlined SVG glyphs avoid system-font substitutions; all labels remain
        # directly editable in this source and SVG shapes remain vector objects.
        "mathtext.fontset": "dejavusans", "svg.fonttype": "path",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    width_in = 16.7 / 2.54
    fig = plt.figure(figsize=(width_in, width_in * 734 / 1692), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1692), ylim=(734, 0))
    ax.axis("off")
    edge = "#444444"
    text_boxes = []

    def box(x, y, w, h, text, *, size=6.5, bold=False, shade="#F2F2F2", dashed=False):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=16",
                              linewidth=0.95, edgecolor=edge, facecolor=shade,
                              linestyle="--" if dashed else "-", zorder=3)
        ax.add_patch(patch)
        t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                    fontsize=size, fontweight="bold" if bold else "normal",
                    linespacing=1.16, zorder=4)
        text_boxes.append((patch, t))

    def arrow(start, end, *, connection="arc3", dashed=False, zorder=2):
        a = FancyArrowPatch(start, end, connectionstyle=connection, arrowstyle="-|>",
                            mutation_scale=7, linewidth=0.95, color=edge,
                            linestyle="--" if dashed else "-", shrinkA=1.3, shrinkB=2,
                            zorder=zorder)
        ax.add_patch(a)
        return a

    # Original four-column structure. Grayscale is deliberately print-friendly.
    box(12, 300, 154, 134, "Input\nimage", size=8, bold=True, shade="#ECECEC")
    box(190, 65, 321, 182,
        "Format matching\n(training and selection:\nW1, R8A matched)\nJPEG re-encode once,\n" + r"quality $\sim U\{60,\ldots,95\}$", size=6.3, shade="#F7F7F7")
    box(204, 315, 287, 196,
        "Preprocess:\nfit to 224 × 224\nTest condition:\nclean / JPEG70 /\nJPEG50 / resize ½", size=6.4, shade="#ECECEC")
    box(18, 533, 407, 158,
        "Alternative whole-image detector:\nResNet-50, all layers fine-tuned\n(W2 = format-matched version)",
        size=6.3, dashed=True, shade="#FAFAFA")
    box(550, 51, 366, 150,
        "Global branch\nfrozen CLIP ViT-B/32\n→ logistic head", size=7, shade="#F0F0F0")
    box(550, 261, 366, 165,
        "Local branch (PatchShuffle)\n16 tiles of 56 px,\n4 fixed shuffles\nfrozen CLIP → mean\n→ logistic head", size=6.4, shade="#F0F0F0")
    box(550, 502, 366, 165,
        "Quality cues\nLaplacian sharpness,\nhigh-frequency energy,\n8-px block-boundary ratio", size=6.6, shade="#ECECEC")
    box(970, 238, 325, 242,
        "Gate MLP (8 → 32 → 1)\ninputs: expert probabilities,\ngap, entropy, local-view\nvariance (+ 3 quality cues\nfor the quality-aware gate)", size=6.4, shade="#F4F4F4")
    box(1333, 269, 340, 165,
        "Fusion on logits\n" + r"$z=\alpha z_g+(1-\alpha)z_l$" + "\naverage: α = 0.5\nfixed: α chosen on\nselection", size=6.7, shade="#F0F0F0")
    box(1333, 502, 340, 165,
        "Threshold: Youden J on the\nselection split → real / fake\nPrimary metric: AUROC", size=6.6, shade="#F0F0F0")

    # Preprocessing and optional format-matching path.
    arrow((171, 337), (204, 189), zorder=4)
    arrow((175, 386), (203, 409))
    arrow((348, 249), (348, 304), dashed=True)
    arrow((497, 375), (550, 139))
    arrow((498, 408), (550, 348))
    arrow((497, 452), (550, 578))

    # Inputs used by the gating network, separate from the fusion's logit inputs.
    arrow((918, 158), (973, 269))
    arrow((920, 345), (972, 359))
    arrow((917, 579), (973, 444))

    # Both experts feed fusion directly; alpha is a third, distinct input.
    arrow((925, 103), (1438, 263), connection="arc3,rad=-0.12")
    ax.text(1168, 91, r"$z_g$", fontsize=8.5, ha="center", va="center", zorder=5)
    local_path = MplPath(
        [(917, 407), (932, 447), (936, 473), (967, 496),
         (1034, 540), (1240, 542), (1292, 469), (1310, 443), (1320, 423), (1331, 413)],
        [MplPath.MOVETO] + [MplPath.CURVE4] * 9,
    )
    local_arrow = FancyArrowPatch(path=local_path, arrowstyle="-|>", mutation_scale=7,
                                  color=edge, linewidth=0.95, zorder=5)
    # A small white casing separates the one crossing from the quality-cue input.
    local_arrow.set_path_effects([pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()])
    ax.add_patch(local_arrow)
    ax.text(1155, 529, r"$z_l$", fontsize=8.5, ha="center", va="center", zorder=6,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
    arrow((1304, 359), (1331, 359))
    ax.text(1314, 331, r"$\alpha$", fontsize=8.5, ha="center", va="center", zorder=5)
    arrow((1503, 442), (1503, 491))

    # Verify text fits each editable box before exporting.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for patch, t in text_boxes:
        pb, tb = patch.get_window_extent(renderer), t.get_window_extent(renderer)
        if not (pb.x0 < tb.x0 and tb.x1 < pb.x1 and pb.y0 < tb.y0 and tb.y1 < pb.y1):
            raise RuntimeError(f"Text does not fit its box: {t.get_text()}")
    stem = args.output_dir / "figure1_detector_overview"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    print(f"Wrote {stem}.png/.pdf/.svg (16.7 cm wide)")


if __name__ == "__main__":
    main()
