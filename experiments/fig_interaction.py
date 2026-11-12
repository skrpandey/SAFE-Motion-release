"""Interaction plot of the matched-world design for each no-remedy cell of Table I.

Plots the four mean logit AUROCs m(w, p), one line per training world w, test policy p on the x
axis; MWI is the difference of the two slopes. Reads results/v13/analysis/cells_v6.csv (from
make_tables_v6.py), no new calculation. Writes paper/figures/fig_interaction.{pdf,svg}.
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

PANELS = [("v7", "runs", "mid", "missing runs, mid"), ("v7", "runs", "low", "missing runs, low"),
          ("v7", "whole", "mid", "whole-joint absence"), ("v8", "frozen", "mid", "frozen frames"),
          ("v13", "armruns", "mid", "arm confidence")]
COL = {"S": "#D55E00", "I": "#0072B2"}


def main():
    rows = {(r["proto"], r["mechanism"], r["strength"]): r for r in csv.DictReader(open("results/v13/analysis/cells_v6.csv"))}
    fig, axes = plt.subplots(1, 5, figsize=(7.1, 2.0), sharey=True)
    for ax, (p, m, s, title) in zip(axes, PANELS):
        r = rows[(p, m, s)]
        for w, mk, ls in (("S", "o", "-"), ("I", "s", "--")):
            y = [float(r[f"m{w}I"]), float(r[f"m{w}S"])]
            lo = [y[0] - float(r[f"m{w}I_lo"]), y[1] - float(r[f"m{w}S_lo"])]
            hi = [float(r[f"m{w}I_hi"]) - y[0], float(r[f"m{w}S_hi"]) - y[1]]
            ax.errorbar([0, 1], y, yerr=[lo, hi], color=COL[w], marker=mk, ls=ls, lw=1.3, ms=4, capsize=2,
                        label=f"trained in World {w}")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["I", "S"], fontsize=8)
        ax.set_xlim(-0.35, 1.35)
        ax.set_title(f"{title}\nMWI {float(r['SE']):.2f}", fontsize=7.5)
        ax.tick_params(labelsize=7.5)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    axes[0].set_ylabel("mean logit AUROC", fontsize=8)
    fig.text(0.5, -0.04, "test policy (label-independent I, label-tied S)", ha="center", fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout(w_pad=0.4)
    for ext in ("pdf", "svg"):
        fig.savefig(f"paper/figures/fig_interaction.{ext}", bbox_inches="tight")
    print("wrote paper/figures/fig_interaction.pdf")


if __name__ == "__main__":
    main()
