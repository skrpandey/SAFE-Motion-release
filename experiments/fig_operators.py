"""Figure: the four planted operators as joint-by-frame masks, and the two worlds' severity laws.

Masks come from safe_motion.planted, drawn at the mean World S severity of each class at mid
strength with a fixed seed. Writes paper/figures/fig_operators.{pdf,svg}.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import beta

from safe_motion import planted as P

OUT = "paper/figures"
JOINTS = ["nose", "l eye", "r eye", "l ear", "r ear", "l shoulder", "r shoulder", "l elbow", "r elbow",
          "l wrist", "r wrist", "l hip", "r hip", "l knee", "r knee", "l ankle", "r ankle"]
T = 200
EPS = {"non-SMM": 2 / 6, "SMM": 2 / 10}     # means of Beta(2,4) and Beta(2,8): the mid-strength World S laws
OPS = [("missing runs", P.missing_mask, "#0072B2", "x, y, conf set to 0\nin runs of frames"),
       ("whole-joint\nabsence", P.whole_joint_mask, "#0072B2", "joint zeroed for\nthe whole window"),
       ("frozen frames", P.frozen_mask, "#7F7F7F", "all joints copy\nthe last frame"),
       ("arm confidence", P.arm_runs_mask, "#D55E00", "\u00d70.3; floor 0.05\ncoordinates kept")]


def main():
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(7.2, 3.7))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1.15], hspace=0.55, wspace=0.35)
    for k, (name, fn, col, note) in enumerate(OPS):
        for r, (cls, eps) in enumerate(EPS.items()):
            ax = fig.add_subplot(gs[r, k])
            m = fn(T, eps, np.random.RandomState(7 + k))
            ax.imshow(m.T, aspect="auto", interpolation="nearest", cmap=ListedColormap(["#F2F2F2", col]))
            ax.set_yticks([0, 5, 10, 16]) if k == 0 else ax.set_yticks([])
            if k == 0:
                ax.set_yticklabels([JOINTS[i] for i in (0, 5, 10, 16)], fontsize=8)
                ax.set_ylabel(f"World S, {cls}\n($\\varepsilon$ = {eps:.2f})", fontsize=8.5)
            ax.set_xticks([0, 100, 199]); ax.set_xticklabels(["0", "100", "200"] if r == 1 else [], fontsize=8)
            if r == 0:
                ax.set_title(f"{name}\n\n", fontsize=8.5, fontweight="bold")
                ax.text(0.5, 1.04, note, transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, color="#444444")
            else:
                ax.set_xlabel("frame", fontsize=8)
            ax.text(0.98, 0.04, f"{m.mean():.0%}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax = fig.add_subplot(gs[:, 4])
    x = np.linspace(0, 1, 300)
    ax.plot(x, beta.pdf(x, 2, 4), color="#0072B2", lw=1.6, label="World S, non-SMM: Beta(2, 4)")
    ax.plot(x, beta.pdf(x, 2, 8), color="#D55E00", lw=1.6, label="World S, SMM: Beta(2, 8)")
    mix = 0.68 * beta.pdf(x, 2, 4) + 0.32 * beta.pdf(x, 2, 8)   # host class shares (3,800 SMM of 11,869)
    ax.plot(x, mix, color="black", lw=1.2, ls="--", label="World I, both classes:\nsame mixture by class share")
    ax.set_xlabel("severity $\\varepsilon$", fontsize=8.5); ax.set_yticks([]); ax.set_xlim(0, 1)
    ax.set_title("severity laws\n(mid strength)", fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(f"{OUT}/fig_operators.{ext}", bbox_inches="tight")
    fig.savefig("/tmp/fig_operators_preview.png", dpi=140, bbox_inches="tight")
    print("wrote", f"{OUT}/fig_operators.pdf/.svg")


if __name__ == "__main__":
    main()
