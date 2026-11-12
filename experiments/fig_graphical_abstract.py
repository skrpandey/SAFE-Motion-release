"""Graphical abstract of the matched-world design (no numbers).

Writes paper/figures/graphical_abstract.{png,pdf}. Not part of the release driver.
    python experiments/fig_graphical_abstract.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ORANGE, BLUE, GREEN, GREY = "#d9641e", "#1f77b4", "#1f9d6a", "#555555"


def box(ax, x, y, w, h, text, ec="black", fc="white", fs=10, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                ec=ec, fc=fc, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="black")


def arrow(ax, p, q, color="black"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, color=color, lw=1.4))


def main():
    fig, ax = plt.subplots(figsize=(13.2, 5.28))   # 2.5:1, the IEEE graphical-abstract ratio
    ax.set_xlim(0, 13.2); ax.set_ylim(0, 5.2); ax.axis("off")
    for x, t in ((0.15, "(a) a segment and its tracking failures"), (4.6, "(b) two training worlds"),
                 (9.0, "(c) scoring and the contrast")):
        ax.text(x, 4.95, t, fontsize=12, fontweight="bold", va="center")
    # (a)
    box(ax, 0.2, 3.5, 1.3, 0.9, "true\nmotion"); arrow(ax, (1.5, 3.95), (1.85, 3.95))
    box(ax, 1.85, 3.5, 1.4, 0.9, "pose\ntracker"); arrow(ax, (3.25, 3.95), (3.6, 3.95))
    ax.text(3.65, 3.95, "observed\nskeleton", fontsize=10, va="center")
    ax.text(0.2, 2.95, "planted operators are modelled on the\nobserved failure forms (form, not joint structure):",
            fontsize=9.5, style="italic", va="top")
    for i, (m, c) in enumerate((("x", BLUE), ("x", BLUE), ("s", GREY), ("o", ORANGE))):
        ax.plot(0.4, 2.25 - 0.38 * i, marker=m, color=c, ms=8, ls="none")
    for i, t in enumerate(("missing runs", "whole-joint absence", "frozen frames", "arm confidence (confidence only)")):
        ax.text(0.65, 2.25 - 0.38 * i, t, fontsize=10, va="center")
    ax.text(0.65, 2.25 - 0.38 * 4, "complete pose-track loss (not planted)", fontsize=10, va="center", color=GREY)
    ax.text(0.2, 0.25, "each with a severity $\\varepsilon \\in [0, 1]$ per segment", fontsize=9.5, va="center")
    # (b)
    box(ax, 4.6, 3.55, 3.9, 0.95, "World S: severity tied to the label\n(non-SMM degraded more)", ec=ORANGE, fc="#fdf1e8")
    box(ax, 4.6, 1.3, 3.9, 0.95, "World I: same marginal severity distribution,\nassigned independently of the label", ec=BLUE, fc="#e9f2fa")
    box(ax, 6.9, 2.55, 1.6, 0.75, "train $f_S$", fs=10); arrow(ax, (6.6, 3.55), (7.3, 3.3), ORANGE)
    box(ax, 6.9, 0.3, 1.6, 0.75, "train $f_I$\n(matched reference)", fs=9.5); arrow(ax, (6.6, 1.3), (7.3, 1.05), BLUE)
    # (c)
    box(ax, 9.0, 3.65, 4.0, 0.85, "test set degraded two ways:\nlabel-tied (S) and label-independent (I)", fs=10)
    box(ax, 9.0, 2.5, 4.0, 0.75, "$R(f_S) = \\mathrm{logit\\,AUC}_S - \\mathrm{logit\\,AUC}_I$", ec=ORANGE, fc="#fdf1e8")
    box(ax, 9.0, 0.3, 4.0, 0.75, "$R(f_I) = \\mathrm{logit\\,AUC}_S - \\mathrm{logit\\,AUC}_I$", ec=BLUE, fc="#e9f2fa")
    box(ax, 9.5, 1.35, 3.0, 0.85, "matched-world interaction\n$\\mathrm{MWI} = R(f_S) - R(f_I)$", ec=GREEN, fc="#e8f6ef", fs=10.5, lw=2)
    arrow(ax, (8.5, 2.9), (9.0, 2.9), ORANGE); arrow(ax, (8.5, 0.68), (9.0, 0.68), BLUE)
    arrow(ax, (11.0, 2.5), (11.0, 2.22), GREEN); arrow(ax, (11.0, 1.05), (11.0, 1.33), GREEN)
    ax.text(13.0, 0.05, "paired identifier-clustered resamples; a relative contrast, not a reliance score",
            fontsize=8.5, color=GREY, ha="right", va="bottom")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"paper/figures/graphical_abstract.{ext}", dpi=300 if ext == "png" else None)
    print("wrote paper/figures/graphical_abstract.png/.pdf")


if __name__ == "__main__":
    main()
