"""Figure 1, the method diagram: tracking failures and planted operators, the two training worlds,
and scoring with MWI = R(f_S) - R(f_I). Labels follow sections/method.tex.
Writes paper/figures/fig1_worlds.{pdf,png} and a single-column variant.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

OUT = "paper/figures"
# Okabe-Ito, colorblind safe
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#777777"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.linewidth": 0.6})


def box(ax, x, y, w, h, text, fc="white", ec="black", fs=8, bold=False, lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.15)


def arrow(ax, x0, y0, x1, y1, color="black", lw=0.9, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9,
                                 color=color, lw=lw, shrinkA=1, shrinkB=1))


def skeleton(ax, cx, cy, s=0.55, missing=(), low=(), frozen=False, color="black"):
    """A 17-joint stick figure in a box of width s; joint indices follow COCO-17 roughly."""
    j = {0: (0, 1.0), 5: (-0.35, 0.72), 6: (0.35, 0.72), 7: (-0.6, 0.42), 8: (0.6, 0.42),
         9: (-0.7, 0.1), 10: (0.7, 0.1), 11: (-0.22, 0.1), 12: (0.22, 0.1),
         13: (-0.3, -0.45), 14: (0.3, -0.45), 15: (-0.32, -1.0), 16: (0.32, -1.0)}
    bones = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
             (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
    P = {k: (cx + v[0] * s * 0.5, cy + v[1] * s * 0.5) for k, v in j.items()}
    for a, b in bones:
        if a in missing or b in missing:
            continue
        c = ORANGE if (a in low or b in low) else color
        ax.plot([P[a][0], P[b][0]], [P[a][1], P[b][1]], color=c, lw=1.1 if c == color else 1.6,
                ls=":" if c == ORANGE else "-", solid_capstyle="round")
    for k, (x, y) in P.items():
        if k in missing:
            ax.plot(x, y, marker="x", color=BLUE, ms=4, mew=1.2)
        else:
            ax.plot(x, y, marker="o", color=ORANGE if k in low else color, ms=2.2)
    if frozen:
        for dx in (0.05, 0.10):
            for a, b in bones[:8]:
                ax.plot([P[a][0] + dx, P[b][0] + dx], [P[a][1], P[b][1]], color=GREY, lw=0.7, alpha=0.5)


def severity_panel(ax, x0, y0, w, h, tied):
    """Small histogram-like sketch: severity on x, two class curves."""
    xs = np.linspace(0, 1, 60)
    beta = lambda a, b: xs ** (a - 1) * (1 - xs) ** (b - 1)
    non = beta(2, 4); smm = beta(2, 8) if tied else beta(2, 4)
    non = non / non.max() * h * 0.75; smm = smm / smm.max() * h * 0.75
    ax.plot([x0, x0 + w], [y0, y0], color="black", lw=0.6)
    ax.plot([x0, x0], [y0, y0 + h * 0.85], color="black", lw=0.6)
    ax.plot(x0 + xs * w, y0 + non, color=BLUE, lw=1.3)
    ax.plot(x0 + xs * w, y0 + smm, color=ORANGE, lw=1.3, ls="--" if not tied else "-")
    ax.text(x0 + w / 2, y0 - 0.1, r"severity $\varepsilon$", ha="center", va="top", fontsize=6.5)


def draw(width, fname):
    fig = plt.figure(figsize=(width, width * 0.40))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 12); ax.set_ylim(-0.45, 4.8); ax.axis("off")
    fs = 7.5 if width < 5 else 8

    # (a) segment and operators
    ax.text(0.15, 4.55, "(a) a segment and its tracking failures", fontsize=fs, fontweight="bold")
    box(ax, 0.15, 3.0, 1.2, 0.9, "true\nmotion", fs=fs)
    arrow(ax, 1.35, 3.45, 1.85, 3.45)
    box(ax, 1.85, 3.0, 1.2, 0.9, "pose\ntracker", fs=fs)
    arrow(ax, 3.05, 3.45, 3.5, 3.45)
    skeleton(ax, 4.0, 3.45, s=1.3, missing=(9, 15), low=(7, 8, 10), frozen=True)
    ax.text(4.0, 2.55, "observed skeleton", ha="center", fontsize=fs)
    ax.text(0.15, 2.25, "planted operators, modelled on failure forms:", fontsize=fs - 0.5, style="italic")
    ops = [("missing runs", BLUE, "x"), ("whole-joint absence", BLUE, "x"),
           ("frozen frames", GREY, "s"), ("arm confidence", ORANGE, "o")]
    for i, (name, c, m) in enumerate(ops):
        y = 1.85 - i * 0.36
        ax.plot(0.3, y, marker=m, color=c, ms=4, mew=1.2)
        ax.text(0.5, y, name, va="center", fontsize=fs - 0.5)
    ax.text(0.5, 1.85 - 4 * 0.36, "complete pose-track loss (not planted)", va="center", fontsize=fs - 0.5, color="0.45")
    ax.text(0.15, -0.2, r"each with severity $\varepsilon \in [0, 1]$ per segment", fontsize=fs - 1)

    # (b) two worlds
    ax.text(4.9, 4.55, "(b) two training worlds", fontsize=fs, fontweight="bold")
    box(ax, 4.9, 3.35, 3.0, 0.85, "World S: severity tied to the label\n(non-SMM degraded more)",
        fc="#FDF0E6", ec=ORANGE, fs=fs - 0.5)
    severity_panel(ax, 5.1, 2.3, 1.1, 0.8, tied=True)
    ax.text(5.15, 3.2, "non-SMM", color=BLUE, fontsize=fs - 1.5, va="center")
    ax.text(5.95, 3.2, "SMM", color=ORANGE, fontsize=fs - 1.5, va="center")
    box(ax, 4.9, 1.05, 3.0, 0.85, "World I: same marginal severity\ndistribution as World S,\nindependent of the label",
        fc="#E8F3FA", ec=BLUE, fs=fs - 0.5)
    severity_panel(ax, 5.1, 0.0, 1.1, 0.8, tied=False)
    ax.text(5.15, 0.9, "non-SMM", color=BLUE, fontsize=fs - 1.5, va="center")
    ax.text(5.95, 0.9, "SMM", color=ORANGE, fontsize=fs - 1.5, va="center")
    box(ax, 6.65, 2.3, 1.2, 0.75, r"train $f_S$" + "\n(can learn\nthe cue)", fs=fs - 0.5)
    box(ax, 6.65, 0.0, 1.2, 0.75, r"train $f_I$" + "\nmatched\nreference", fs=fs - 0.5)
    arrow(ax, 6.4, 3.35, 6.9, 3.05, color=ORANGE)
    arrow(ax, 6.4, 1.05, 6.9, 0.75, color=BLUE)

    # (c) scoring
    ax.text(8.15, 4.55, "(c) scoring and the contrast", fontsize=fs, fontweight="bold")
    box(ax, 8.15, 3.35, 3.7, 0.85, "test set degraded two ways:\nlabel-tied (S) and label-independent (I)",
        fs=fs - 0.5)
    arrow(ax, 7.85, 2.67, 8.5, 2.67, color=ORANGE)
    arrow(ax, 7.85, 0.37, 8.5, 0.37, color=BLUE)
    box(ax, 8.5, 2.3, 3.35, 0.75, r"$R(f_S) = \mathrm{logit}\,\mathrm{AUC}_S - \mathrm{logit}\,\mathrm{AUC}_I$",
        fc="#FDF0E6", ec=ORANGE, fs=fs - 0.5)
    box(ax, 8.5, 0.0, 3.35, 0.75, r"$R(f_I) = \mathrm{logit}\,\mathrm{AUC}_S - \mathrm{logit}\,\mathrm{AUC}_I$",
        fc="#E8F3FA", ec=BLUE, fs=fs - 0.5)
    box(ax, 8.9, 1.15, 2.6, 0.8, r"matched-world interaction" + "\n" + r"$\mathrm{MWI} = R(f_S) - R(f_I)$",
        fc="#E6F5EF", ec=GREEN, fs=fs, bold=False, lw=1.1)
    arrow(ax, 10.2, 2.3, 10.2, 1.95, color=GREEN)
    arrow(ax, 10.2, 0.75, 10.2, 1.15, color=GREEN)
    ax.text(11.85, -0.35, "paired on shared resamples", ha="right", fontsize=fs - 1.5, color=GREY)

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, fname + ".pdf"))
    fig.savefig(os.path.join(OUT, fname + ".png"), dpi=300)
    plt.close(fig)
    print("wrote", os.path.join(OUT, fname + ".pdf"))


if __name__ == "__main__":
    draw(7.0, "fig1_worlds")          # double column
    draw(3.4, "fig1_worlds_single")   # single-column check
