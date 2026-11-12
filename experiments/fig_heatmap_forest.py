"""Crossed-operator heatmap and remedy forest plot (ASDPose).

fig_crossed_heatmap: R of the no-remedy World I and World S models, rows = operator trained against,
columns = operator at test time, first test mask (seed 2000), from results/v7/analysis/groups.csv.
fig_remedies_forest: change in SE per remedy and mechanism against the no-remedy cell, mean over
three test masks (results/v7/analysis/effects_masks.csv), logit scale, with the raw-AUC verdict.
Writes paper/figures/fig_crossed_heatmap.{pdf,svg} and fig_remedies_forest.{pdf,svg}.
"""
import csv
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MECHS = ["runs", "whole", "frozen", "armruns"]
NAME = {"runs": "missing runs", "whole": "whole-joint\nabsence", "frozen": "frozen frames", "armruns": "arm confidence"}
SHORT = {"runs": "missing\nruns", "whole": "whole\njoint", "frozen": "frozen\nframes", "armruns": "arm\nconf."}
REM = {"A_velocity": "velocity", "A_dropout": "joint dropout", "A_both": "vel. + dropout",
       "B_coorddrop": "coord. dropout"}
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})


def parse_r(field):
    m = re.match(r"\s*([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]", field or "")
    return tuple(float(v) for v in m.groups()) if m else None


def crossed():
    grp = list(csv.DictReader(open("results/v7/analysis/groups.csv")))
    # single-column layout: the two worlds stacked
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 5.8), gridspec_kw={"hspace": 0.55})
    for ax, world, title in zip(axes, ("I", "S"), ("World I models (matched references)",
                                                    "World S models (cue-trained)")):
        M = np.full((4, 4), np.nan); L = [["" for _ in MECHS] for _ in MECHS]
        for i, tm in enumerate(MECHS):
            proto = "v8" if tm in ("frozen", "armruns") else "v7"
            g = next(x for x in grp if x["proto"] == proto and x["mechanism"] == tm and x["strength"] == "mid"
                     and x["world"] == world and x["cell"] == "A_none" and x["world_seed"] == "0")
            if tm == "armruns":           # arm confidence uses the corrected validity rule (protocol v13)
                g = {f"R_{m.group(1)}_mid_m2000": r["logit"] for r in csv.DictReader(open("results/v13/analysis/effects.csv"))
                     for m in [re.match(r"crossed R under (\w+) \(mask 2000\)", r["contrast"])]
                     if m and r["proto"] == f"v13 vs v8, World {world}"}
                assert len(g) == 4, g
            for j, te in enumerate(MECHS):
                r = parse_r(g.get(f"R_{te}_mid_m2000"))
                if r:
                    M[i, j] = r[0]
                    star = "*" if (r[1] > 0 or r[2] < 0) else ""
                    L[i][j] = f"{r[0]:+.2f}{star}".replace("-", "−")
        im = ax.imshow(M, cmap="RdBu_r", vmin=-0.8, vmax=0.8)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, L[i][j], ha="center", va="center", fontsize=8,
                        fontweight="bold" if i == j else "normal",
                        color="white" if abs(M[i, j]) > 0.55 else "black")
                if i == j:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=1.2, ec="black"))
        ax.set_xticks(range(4)); ax.set_xticklabels([SHORT[m] for m in MECHS], fontsize=7.5)
        ax.set_yticks(range(4))
        ax.set_yticklabels([SHORT[m] for m in MECHS], fontsize=8)
        ax.set_xlabel("operator at test time", fontsize=8)
        ax.set_ylabel("trained against", fontsize=8)
        ax.set_title(title, fontsize=8.5)
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.04)
    cb.set_label("R (logit AUC), first test mask", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    fig.text(0.02, 0.0, "Box: matched operator. *: interval excludes zero.", fontsize=7.5)
    for ext in ("pdf", "svg"):
        fig.savefig(f"paper/figures/fig_crossed_heatmap.{ext}", bbox_inches="tight")
    fig.savefig("/tmp/fig_crossed_heatmap.png", dpi=150, bbox_inches="tight")


def remedies():
    eff = list(csv.DictReader(open("results/v7/analysis/effects_masks.csv")))
    rows = []
    for mech in MECHS:
        for cell, lab in REM.items():
            e = next((x for x in eff if x["mechanism"] == mech and x["strength"] == "mid"
                      and x["contrast"] == f"SE({cell}) - SE(A_none)"), None)
            if e:
                rows.append((mech, lab, float(e["point"]), float(e["ci_lo"]), float(e["ci_hi"]), e["raw_excludes_zero"]))
    fig, ax = plt.subplots(figsize=(2.6, 2.75))
    y = 0; ticks, labels = [], []
    cols = {"runs": "#0072B2", "whole": "#56B4E9", "frozen": "#7F7F7F", "armruns": "#D55E00"}
    last = None
    for mech, lab, p, lo, hi, raw in rows:
        if last and mech != last:
            y -= 1.4
        filled = lo > 0 or hi < 0
        ax.errorbar(p, y, xerr=[[p - lo], [hi - p]], fmt="o", color=cols[mech], ms=4, capsize=2, lw=1.2,
                    mfc=cols[mech] if filled else "white")
        if raw == "True" or raw == "yes":
            ax.text(hi + 0.01, y, "both scales", fontsize=10, va="center", color="#333333")
        if mech != last:
            ax.text(-0.02, y + 0.85, NAME[mech].replace(chr(10), ' ') + (' (original rule)' if mech == 'armruns' else ''),
                    transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=10.5, fontweight="bold")
        ticks.append(y); labels.append(lab)
        y -= 1; last = mech
    ax.axvline(0, color="#777777", lw=0.7)
    ax.set_ylim(y + 0.5, ticks[0] + 1.3)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=10.5)
    ax.tick_params(axis="x", labelsize=10.5)
    ax.set_xlabel("change in MWI against no remedy\n(logit AUC, mean over three test masks)", fontsize=10.5)
    ax.set_title("No reduction detected among the\nfour input configurations", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    for ext in ("pdf", "svg"):
        fig.savefig(f"paper/figures/fig_remedies_forest.{ext}", bbox_inches="tight")
    fig.savefig("/tmp/fig_remedies_forest.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    crossed()
    remedies()
    print("wrote paper/figures/fig_crossed_heatmap and fig_remedies_forest")
