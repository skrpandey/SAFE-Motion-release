"""Pilot batch 2 analysis: batch-1 metrics per TinyTCN cell, H1 decision, descriptive contrasts.

Paired child-clustered bootstrap: the same resample indices are used for every cell, so cell
differences carry paired CIs. Writes results/pilots/pilot_batch2/{cells.csv, summary.json}.
Usage: conda run -n torch python experiments/analyze_batch2.py [--n-boot 10000]
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pilot_batch1 as B1
from safe_motion import metrics as M

OUT = "results/pilots/pilot_batch2"


def load(cell):
    sc = {r["identifier"]: float(r["smm_score"])
          for r in csv.DictReader(open(f"{OUT}/{cell}/test_scores.csv"))}
    return np.array([sc[i] for i in B1.G["ids"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    B1.setup(args.n_boot)
    cells = sorted(os.path.basename(os.path.dirname(p))
                   for p in glob.glob(f"{OUT}/*/train_log.json"))
    full = np.arange(len(B1.G["ids"]))
    point, boot = {}, {}
    for c in cells:
        s = load(c)
        point[c] = np.array(B1.three(s, full))
        boot[c] = np.array([B1.three(s, i) for i in B1.G["boots"]])
        print("scored", c, np.round(point[c], 4), flush=True)

    rows = []
    for c in cells:
        p, b = point[c], boot[c]
        probe = json.load(open(f"{OUT}/{c}/probe.json"))
        log = json.load(open(f"{OUT}/{c}/train_log.json"))
        row = {"cell": c, "best_epoch": log["best_epoch"],
               "auc_standard": p[0], "auc_qc": p[1], "auc_reweighted": p[2],
               "delta_qc": p[0] - p[1], "gap_dprime_qc": float(M.dprime(p[0]) - M.dprime(p[1])),
               "binormal_slope": M.binormal_slope(load(c), B1.G["y"])}
        for k, j in (("auc_standard", 0), ("auc_qc", 1), ("auc_reweighted", 2)):
            row[k + "_ci"] = M.ci(b[:, j])
        row["delta_qc_ci"] = M.ci(b[:, 0] - b[:, 1])
        for t, v in probe.items():
            row[f"probe_{t}"] = v["r2"]
            row[f"probe_{t}_minus_random"] = v["r2_minus_random"]
        rows.append(row)
    with open(f"{OUT}/cells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"[{v[0]:.4f}, {v[1]:.4f}]" if isinstance(v, tuple)
                            else (f"{v:.4f}" if isinstance(v, float) else v)) for k, v in r.items()})

    def diff(a, b, j):          # paired difference a - b on protocol column j
        return {"point": float(point[a][j] - point[b][j]),
                "ci": M.ci(boot[a][:, j] - boot[b][:, j])}

    def dgap(c):
        return M.dprime(boot[c][:, 0]) - M.dprime(boot[c][:, 1])

    summary = {"cells": cells}
    if {"A_none", "A_both"} <= set(cells):
        d_std, d_qc = diff("A_both", "A_none", 0), diff("A_both", "A_none", 1)
        g_both = M.dprime(point["A_both"][0]) - M.dprime(point["A_both"][1])
        g_none = M.dprime(point["A_none"][0]) - M.dprime(point["A_none"][1])
        d_gap = {"point": float(g_both - g_none), "ci": M.ci(dgap("A_both") - dgap("A_none"))}
        summary["H1"] = {"both_minus_none_auc_standard": d_std, "both_minus_none_auc_qc": d_qc,
                         "both_minus_none_gap_dprime": d_gap,
                         "pass": bool(d_std["point"] > 0 and d_qc["point"] > 0
                                      and d_gap["point"] < 0 and d_qc["ci"][0] > 0)}
        gain = point["A_both"][1] - point["A_none"][1]
        if "B_coorddrop" in cells and abs(gain) > 1e-9:
            summary["coorddrop_share_of_qc_gain"] = float(
                (point["B_coorddrop"][1] - point["A_none"][1]) / gain)
            summary["coorddrop_minus_none_auc_qc"] = diff("B_coorddrop", "A_none", 1)
        for c in ("A_velocity", "A_dropout"):
            if c in cells:
                summary[f"{c}_share_of_qc_gain"] = float((point[c][1] - point["A_none"][1]) / gain)
    if {"C_qc_none", "C_qc_velocity"} <= set(cells):
        summary["qcpool_velocity_minus_none_auc_qc"] = diff("C_qc_velocity", "C_qc_none", 1)
        summary["qcpool_velocity_minus_none_auc_standard"] = diff("C_qc_velocity", "C_qc_none", 0)
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
