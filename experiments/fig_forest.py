"""Figures: forest plot of R(f_S), R(f_I) and SE per mechanism on ASDPose (v7 + v8), and the v9
variability plot (SE by training seed and test-mask seed, with the subject-bootstrap interval).
Reads results/v7/analysis/{effects,groups}.csv and results/v9/analysis/{variance,effects}.csv.
Writes paper/figures/forest_v7.pdf and paper/figures/v9_variability.pdf (plus .png previews).
    conda run -n torch python experiments/fig_forest.py
"""
import csv
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

# Okabe-Ito, colorblind safe
C_S, C_I, C_SE, C_GREY = "#D55E00", "#0072B2", "#009E73", "#777777"
MECH_NAME = {"runs": "missing runs", "whole": "whole-joint\nabsence", "frozen": "frozen frames", "armruns": "arm confidence"}
HOST_NAME = {"asdpose_host": "ASDPose (40 test IDs)", "mmasd_chest_squat": "MMASD (11 participants)",
             "utd_mhad_body_vs_arm": "UTD-MHAD (8 subjects)"}
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "pdf.fonttype": 42})


def parse_r(field):
    m = re.match(r"\s*([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]", field or "")
    return tuple(float(v) for v in m.groups()) if m else None


def forest_v7(analysis="results/v7/analysis", out="paper/figures/forest_v7", width=3.4):
    """Mask-mean R(f_S), R(f_I) and SE per mechanism (effects_masks / groups_masks); SE and R(f_I) emphasised."""
    eff = {(r["mechanism"], r["strength"]): r for r in csv.DictReader(open(f"{analysis}/effects_masks.csv"))
           if r["contrast"] == "SE(A_none)"}
    grp = list(csv.DictReader(open(f"{analysis}/groups_masks.csv")))
    items = [("runs", "mid"), ("runs", "low"), ("whole", "mid"), ("frozen", "mid"), ("armruns", "mid")]
    fig, ax = plt.subplots(figsize=(width, 3.3 if width < 5 else 2.9))
    ticks, labels = [], []
    v13 = [r for r in csv.DictReader(open("results/v13/analysis/effects.csv")) if r["proto"] == "v13"]
    for k, (mech, st) in enumerate(items):
        y = -k
        if mech == "armruns":         # corrected validity rule (protocol v13); the v8 cell is not shown
            assert len(v13) == 1, len(v13)
            rS, rI, se = parse_r(v13[0]["R_S"]), parse_r(v13[0]["R_I"]), parse_r(v13[0]["logit"])
        else:
            e = eff[(mech, st)]
            g = {w: next(x for x in grp if x["mechanism"] == mech and x["strength"] == st and x["world"] == w
                         and x["cell"] == "A_none") for w in ("S", "I")}
            rS, rI = parse_r(g["S"]["R_maskmean"]), parse_r(g["I"]["R_maskmean"])
            se = (float(e["point"]), float(e["ci_lo"]), float(e["ci_hi"]))
        for off, (p_, lo, hi), c, lab, ms, lw, alpha in (
                (0.26, rS, C_S, "$R(f_S)$, World S model", 3, 0.8, 0.55),
                (0.0, rI, C_I, "$R(f_I)$, World I reference", 4.5, 1.4, 1.0),
                (-0.26, se, C_SE, "MWI $= R(f_S) - R(f_I)$", 5, 1.6, 1.0)):
            ax.errorbar(p_, y + off, xerr=[[p_ - lo], [hi - p_]], fmt="o", color=c, ms=ms, capsize=2, lw=lw,
                        alpha=alpha, label=lab if k == 0 else None)
        ticks.append(y)
        labels.append(MECH_NAME[mech] + ("" if st == "mid" else "\n(low strength)"))
    ax.axvspan(-0.05, 0.05, color=C_GREY, alpha=0.15, lw=0, label="reference margin $\\pm$0.05")
    ax.axvline(0, color=C_GREY, lw=0.6)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(-len(items) + 0.4, 1.1)
    ax.set_xlabel("logit-AUC scale, mean over three test masks")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2, frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    print("wrote", out + ".pdf")


def v9_variability(analysis="results/v9/analysis", out="paper/figures/v9_variability"):
    """SE by training seed (rows of points) and test-mask seed (marker) per host, shared x scale, subject-bootstrap bar."""
    var = [r for r in csv.DictReader(open(f"{analysis}/variance.csv")) if r["quantity"] == "SE"]
    hosts = [h for h in HOST_NAME if any(r["host"] == h for r in var)]
    allv = [v for r in var for v in json.loads(r["grid_json"]).values()] + \
           [float(r[k]) for r in var for k in ("ci_lo", "ci_hi")]
    lo_x, hi_x = min(allv) - 0.05, max(allv) + 0.05
    fig, axes = plt.subplots(1, len(hosts), figsize=(7.0, 3.0), sharex=True)
    axes = np.atleast_1d(axes)
    markers = {"2000": "o", "3000": "s", "4000": "^"}
    for ax, h in zip(axes, hosts):
        rows = [r for r in var if r["host"] == h]
        rows.sort(key=lambda r: (["runs", "whole", "frozen", "armruns"].index(r["mechanism"]), r["strength"] != "mid"))
        for k, r in enumerate(rows):
            grid = json.loads(r["grid_json"])
            p_, lo, hi = float(r["point_m2000"]), float(r["ci_lo"]), float(r["ci_hi"])
            ax.plot([lo, hi], [k, k], color=C_GREY, lw=5, alpha=0.35, solid_capstyle="butt")
            for key, v in grid.items():
                sd, m = key.split("_")
                ax.plot(v, k + (int(sd[1]) - 1) * 0.2, marker=markers[m[1:]], ms=5, color=C_SE, ls="none",
                        mfc=C_SE if m == "m2000" else "white", mew=0.8)
            ax.plot(p_, k, marker="|", ms=11, color="black", mew=1.3)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([MECH_NAME[r["mechanism"]].replace("\n", " ") + ("" if r["strength"] == "mid" else " (low)")
                            + (("\n(v13, corrected rule)" if h == "asdpose_host" else "\n(original rule)")
                               if r["mechanism"] == "armruns" else "")
                            for r in rows])
        ax.axvline(0, color=C_GREY, lw=0.6)
        ax.set_xlim(lo_x, hi_x)
        n = rows[0].get("n_subjects", "")
        lab = "participants" if h.startswith("mmasd") else rows[0].get("subject_label", "subjects")
        ax.set_title(HOST_NAME[h].split(" (")[0] + f"\n{n} {lab}", fontsize=8)
        ax.set_xlabel("MWI (logit-AUC)")
        ax.spines[["top", "right"]].set_visible(False)
    h1 = [plt.Line2D([], [], marker=m, color=C_SE, ls="none", ms=5, mfc=(C_SE if s_ == "2000" else "white"),
                     label=f"test mask {s_}") for s_, m in markers.items()]
    h1 += [plt.Line2D([], [], color=C_GREY, lw=5, alpha=0.35, label="95% subject-bootstrap interval (mask 2000)"),
           plt.Line2D([], [], marker="|", color="black", ls="none", ms=11, label="pooled point (mask 2000)")]
    fig.legend(handles=h1, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.03), fontsize=7.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("pdf", "svg"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    print("wrote", out + ".pdf")


if __name__ == "__main__":
    os.makedirs("paper/figures", exist_ok=True)
    forest_v7()
    forest_v7(out="paper/figures/forest_v7_wide", width=7.0)
    v9_variability()
