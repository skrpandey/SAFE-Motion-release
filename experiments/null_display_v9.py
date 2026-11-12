"""Post hoc display of the v9 permutation nulls on MMASD and UTD-MHAD (no new fits).

Each host x mechanism null has 20 fitted draws (results/v9/analysis/null.csv). Per cell, writes
the null summary, the empirical 95th percentile, the exceedance count, p = (1 + count) / 21 (floor
1/21) and 0.95^20, the probability that all 20 draws lie below the true 95th percentile. Writes
results/v9/analysis/null_display.csv and the strip plot paper/figures/fig_null_v9.{pdf,svg}.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion.paths import guard_output   # noqa: E402

HOST_NAME = {"mmasd_chest_squat": "MMASD", "utd_mhad_body_vs_arm": "UTD-MHAD"}
MECH_NAME = {"runs": "missing runs", "whole": "whole-joint absence", "frozen": "frozen frames", "armruns": "arm confidence"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="results/v9/analysis")
    ap.add_argument("--fig", default="paper/figures/fig_null_v9")
    args = ap.parse_args()
    out_csv = f"{args.analysis}/null_display.csv"
    guard_output(out_csv); guard_output(args.fig + ".pdf")
    null = defaultdict(list)
    for r in csv.DictReader(open(f"{args.analysis}/null.csv")):
        null[(r["host"], r["mechanism"], r["strength"])].append(float(r["SE_perm"]))
    obs = {(r["host"], r["mechanism"], r["strength"]): float(r["point"])
           for r in csv.DictReader(open(f"{args.analysis}/effects.csv")) if r["contrast"] == "SE(A_none)"}
    rows = []
    keys = [k for k in sorted(null) if k[0] in HOST_NAME]
    for k in keys:
        v = np.array(sorted(null[k])); n = len(v); o = obs.get(k)
        count = int((v >= o).sum()) if o is not None else None
        p = (1 + count) / (n + 1) if count is not None else None
        rows.append({"host": k[0], "mechanism": k[1], "strength": k[2], "n_null": n,
                     "null_min": round(float(v.min()), 4), "null_median": round(float(np.median(v)), 4),
                     "null_p95_empirical": round(float(np.percentile(v, 95)), 4), "null_max": round(float(v.max()), 4),
                     "observed": o, "n_null_ge_observed": count,
                     "p": (f"{p:.4f}" if p is not None else ""), "p_is_floor": (count == 0),
                     "p_floor": f"1/{n + 1}",
                     "prob_all_null_below_true_q95": round(0.95 ** n, 4),
                     "null_values": "; ".join(f"{x:+.4f}" for x in v)})
    with open(out_csv, "w", newline="") as f:
        f.write("# post hoc display: 20 fitted sign-flip reference draws per cell (not an exact permutation null); no resampling of the null values; "
                "p = (1 + #null >= observed) / 21, floor 1/21; P(all 20 null draws below the true 95th percentile) = 0.95^20\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", out_csv)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=False)
    for ax, host in zip(axes, HOST_NAME):
        ks = [k for k in keys if k[0] == host]
        for j, k in enumerate(ks):
            v = np.array(null[k])
            ax.scatter(v, np.full(len(v), j) + np.random.RandomState(0).uniform(-0.12, 0.12, len(v)),
                       s=9, color="0.55", zorder=2)
            ax.plot([np.percentile(v, 95)] * 2, [j - 0.3, j + 0.3], color="0.2", lw=1.2, zorder=3)
            if k in obs:
                ax.scatter([obs[k]], [j], marker="D", s=28, color="#D55E00", zorder=4)
        ax.set_yticks(range(len(ks))); ax.set_yticklabels([MECH_NAME[k[1]] for k in ks], fontsize=8)
        ax.axvline(0, color="0.8", lw=0.8, zorder=1)
        ax.set_title(HOST_NAME[host], fontsize=9); ax.set_xlabel("MWI (logit AUC)", fontsize=8); ax.tick_params(labelsize=8)
    axes[0].scatter([], [], s=9, color="0.55", label="20 fitted sign-flip reference draws")
    axes[0].plot([], [], color="0.2", lw=1.2, label="empirical 95th percentile")
    axes[0].scatter([], [], marker="D", s=28, color="#D55E00", label="observed MWI")
    fig.legend(loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(args.fig + ".pdf", bbox_inches="tight"); fig.savefig(args.fig + ".svg", bbox_inches="tight")
    print("wrote", args.fig + ".pdf")


if __name__ == "__main__":
    main()
