"""Pilot v3 analysis (protocol pilot_asdpose_v3.yaml): planted worlds and fresh-seed confirmation.

Planted: per model AUC_clean, AUC_S, AUC_I on the QC host test; R = logit(AUC_S) - logit(AUC_I);
shortcut effect = R(World S model) - R(World I model); decomposition of each remedy's clean gain.
Real: logit gap (standard vs QC) on fresh seeds 3-5. Seed-averaged, paired child-clustered CIs.
Writes results/pilots/pilot_v3/{planted.csv, summary.json}.
Usage: conda run -n torch python experiments/analyze_v3.py [--n-boot 10000] [--jobs 10]
"""

import argparse
import csv
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analyze_v2 as A2
from safe_motion import metrics as M
from safe_motion.data import asdpose as D

OUT = "results/pilots/pilot_v3"
G = {}


def logit(a):
    return A2.logit(a)


def setup(n_boot):
    rows = [r for r in D.load_index() if r["split"] == "test" and D.passes_qc(r, A2.V1["qc_filter"])]
    G["ids"] = [r["identifier"] for r in rows]
    G["y"] = np.array([r["binary_label"] for r in rows])
    G["boots"] = list(M.cluster_bootstrap_indices([r["child_id"] for r in rows], n_boot, seed=0))


def planted_metrics(seed_dir):
    """(n_boot + 1, 3) array: row 0 is the point estimate; columns clean, S, I."""
    out = []
    s = [A2.load_scores(f"{seed_dir}/test_scores_{k}.csv", G["ids"]) for k in ("clean", "S", "I")]
    y = G["y"]
    for idx in [np.arange(len(y))] + G["boots"]:
        out.append([M.auc(v[idx], y[idx]) for v in s])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args()
    setup(args.n_boot)

    seed_dirs = sorted(glob.glob(f"{OUT}/planted_*/*/*/seed*"))
    with Pool(args.jobs) as pool:
        res = dict(zip(seed_dirs, pool.map(planted_metrics, seed_dirs)))
    groups = {}
    for d, m in res.items():
        _, strength, world, cell, _ = d.split("/")[-5:]
        groups.setdefault((strength.replace("planted_", ""), world, cell), []).append(m)
    PM = {k: np.mean(v, 0) for k, v in groups.items()}       # seed-averaged, (n_boot+1, 3)
    R = {k: logit(v[:, 1]) - logit(v[:, 2]) for k, v in PM.items()}

    def stat(a):
        return {"point": float(a[0]), "ci": M.ci(a[1:])}

    rows = []
    for k in sorted(PM):
        v = PM[k]
        rows.append({"strength": k[0], "world": k[1], "cell": k[2], "n_seeds": len(groups[k]),
                     "auc_clean": stat(v[:, 0]), "auc_S": stat(v[:, 1]), "auc_I": stat(v[:, 2]),
                     "R": stat(R[k])})
    with open(f"{OUT}/planted.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strength", "world", "cell", "n_seeds", "auc_clean", "auc_clean_ci", "auc_S",
                    "auc_I", "R", "R_ci"])
        for r in rows:
            w.writerow([r["strength"], r["world"], r["cell"], r["n_seeds"],
                        f"{r['auc_clean']['point']:.4f}", "[%.4f, %.4f]" % r["auc_clean"]["ci"],
                        f"{r['auc_S']['point']:.4f}", f"{r['auc_I']['point']:.4f}",
                        f"{r['R']['point']:.4f}", "[%.4f, %.4f]" % r["R"]["ci"]])

    excl_pos = lambda s: s["ci"][0] > 0
    contains0 = lambda s: s["ci"][0] <= 0 <= s["ci"][1]
    S = {"n_runs": len(seed_dirs), "cells": [dict(r, **{}) for r in rows]}
    se = {st: stat(R[(st, "S", "A_none")] - R[(st, "I", "A_none")])
          for st in ("mid", "low") if (st, "S", "A_none") in R and (st, "I", "A_none") in R}
    S["shortcut_effect_A_none"] = se
    S["H2_pass"] = bool("mid" in se and se["mid"]["point"] > 0 and excl_pos(se["mid"]))
    if "low" in se:
        dm = stat((R[("mid", "S", "A_none")] - R[("mid", "I", "A_none")])
                  - (R[("low", "S", "A_none")] - R[("low", "I", "A_none")]))
        S["shortcut_effect_mid_minus_low"] = dm
        S["H3_pass"] = bool(excl_pos(se["low"]) and excl_pos(dm))

    dec = {}
    for r_ in ("A_velocity", "A_dropout", "A_both", "B_coorddrop"):
        if ("mid", "S", r_) not in PM:
            continue
        tot = PM[("mid", "S", r_)][:, 0] - PM[("mid", "S", "A_none")][:, 0]
        reg = PM[("mid", "I", r_)][:, 0] - PM[("mid", "I", "A_none")][:, 0]
        dec[r_] = {"total_gain": stat(tot), "regularization_part": stat(reg),
                   "shortcut_part": stat(tot - reg),
                   "reliance_change": stat(R[("mid", "S", r_)] - R[("mid", "S", "A_none")]),
                   "shortcut_effect": stat(R[("mid", "S", r_)] - R[("mid", "I", r_)])}
    S["decomposition_mid"] = dec
    if {"A_dropout", "A_both", "B_coorddrop"} <= set(dec):
        cue = excl_pos(dec["A_dropout"]["shortcut_part"]) or excl_pos(dec["A_both"]["shortcut_part"])
        if cue and contains0(dec["B_coorddrop"]["shortcut_part"]):
            v = "supports_shortcut_removal"
        elif all(contains0(d["shortcut_part"]) for d in dec.values()) and \
                any(excl_pos(d["regularization_part"]) for d in dec.values()):
            v = "supports_regularization"
        else:
            v = "inconclusive"
        S["H5b_verdict"] = v

    real = sorted(glob.glob(f"{OUT}/real/*/seed*/test_scores.csv"))
    if real:
        T = A2.Test(args.n_boot)
        full = np.arange(len(T.y))
        by = {}
        for f in real:
            c = f.split("/")[-3]
            s = A2.load_scores(f, T.ids)
            m = np.vstack([T.metrics(s, full)] + [T.metrics(s, i) for i in T.boots])
            by.setdefault(c, []).append(m)
        RM = {c: np.mean(v, 0) for c, v in by.items()}
        lg = {c: logit(v[:, 0]) - logit(v[:, 1]) for c, v in RM.items()}
        S["real_fresh_seeds"] = {c: {"n_seeds": len(by[c]), "auc_standard": float(RM[c][0, 0]),
                                     "auc_qc": float(RM[c][0, 1]), "logit_gap": stat(lg[c])}
                                 for c in RM}
        if {"A_none", "A_both", "B_coorddrop"} <= set(lg):
            d1, d2 = stat(lg["A_both"] - lg["A_none"]), stat(lg["B_coorddrop"] - lg["A_none"])
            S["H6"] = {"both_minus_none_logit_gap": d1, "coorddrop_minus_none_logit_gap": d2,
                       "pass": bool(d1["ci"][1] < 0 and contains0(d2))}
    json.dump(S, open(f"{OUT}/summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in S.items() if k != "cells"}, indent=1))


if __name__ == "__main__":
    main()
