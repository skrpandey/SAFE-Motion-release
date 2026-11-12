"""Pilot v4 analysis (protocol pilot_asdpose_v4.yaml): H7 (CompactCTRGCN), H8 and H9 (mechanism).

Uses the v3 host test, metrics and bootstrap indices (analyze_v3.setup), so v3 and v4 results
share resamples and the H9 cross-mechanism contrast is paired.
Writes results/pilots/pilot_v4/{planted.csv, summary.json}.
Usage: conda run -n torch python experiments/analyze_v4.py [--n-boot 10000] [--jobs 10]
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
import analyze_v3 as A3
from safe_motion import metrics as M

OUT = "results/pilots/pilot_v4"


def collect(pattern, key_fn, jobs):
    dirs = sorted(glob.glob(pattern))
    with Pool(jobs) as pool:
        res = pool.map(A3.planted_metrics, dirs)
    groups = {}
    for d, m in zip(dirs, res):
        groups.setdefault(key_fn(d), []).append(m)
    return {k: np.mean(v, 0) for k, v in groups.items()}, {k: len(v) for k, v in groups.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args()
    A3.setup(args.n_boot)
    lg = A3.logit
    stat = lambda a: {"point": float(a[0]), "ci": M.ci(a[1:])}
    excl_pos = lambda s: s["ci"][0] > 0
    excl_neg = lambda s: s["ci"][1] < 0
    has0 = lambda s: s["ci"][0] <= 0 <= s["ci"][1]

    k4 = lambda d: tuple(d.split("/")[-4:-1])          # (tag, world, cell)
    PM, N = collect(f"{OUT}/*/*/*/seed*", k4, args.jobs)
    R = {k: lg(v[:, 1]) - lg(v[:, 2]) for k, v in PM.items()}
    with open(f"{OUT}/planted.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "world", "cell", "n_seeds", "auc_clean", "auc_S", "auc_I", "R", "R_ci"])
        for k in sorted(PM):
            r = stat(R[k])
            w.writerow([*k, N[k], *(f"{PM[k][0, j]:.4f}" for j in range(3)), f"{r['point']:.4f}",
                        "[%.4f, %.4f]" % r["ci"]])
    S = {"n_seeds": {"/".join(k): n for k, n in N.items()}}

    # H7: CompactCTRGCN matched-world measure
    se = {}
    for st in ("mid", "low"):
        t = f"compact_runs_{st}"
        if (t, "S", "A_none") in R and (t, "I", "A_none") in R:
            se[st] = R[(t, "S", "A_none")] - R[(t, "I", "A_none")]
    if len(se) == 2:
        h7 = {"shortcut_effect": {st: stat(v) for st, v in se.items()},
              "mid_minus_low": stat(se["mid"] - se["low"]),
              "R_world_I": {st: stat(R[(f"compact_runs_{st}", "I", "A_none")]) for st in se},
              "auc_clean": {f"{st}_{w}": float(PM[(f"compact_runs_{st}", w, "A_none")][0, 0])
                            for st in se for w in ("S", "I")}}
        h7["pass"] = bool(excl_pos(h7["shortcut_effect"]["mid"]) and
                          excl_pos(h7["shortcut_effect"]["low"]) and excl_pos(h7["mid_minus_low"])
                          and all(has0(v) for v in h7["R_world_I"].values()))
        S["H7"] = h7

    # H8: decomposition on the whole-joint mechanism (TinyTCN)
    t = "tcn_whole_mid"
    dec = {}
    for r_ in ("A_dropout", "B_coorddrop", "A_velocity"):
        if (t, "S", r_) not in PM:
            continue
        tot = PM[(t, "S", r_)][:, 0] - PM[(t, "S", "A_none")][:, 0]
        reg = PM[(t, "I", r_)][:, 0] - PM[(t, "I", "A_none")][:, 0]
        dec[r_] = {"total_gain": stat(tot), "regularization_part": stat(reg),
                   "shortcut_part": stat(tot - reg),
                   "reliance_change": stat(R[(t, "S", r_)] - R[(t, "S", "A_none")]),
                   "shortcut_effect": stat(R[(t, "S", r_)] - R[(t, "I", r_)])}
    if (t, "S", "A_none") in R:
        dec["A_none"] = {"shortcut_effect": stat(R[(t, "S", "A_none")] - R[(t, "I", "A_none")])}
    S["decomposition_whole_mid"] = dec
    if {"A_dropout", "B_coorddrop"} <= set(dec):
        d, b = dec["A_dropout"], dec["B_coorddrop"]
        S["H8_pass"] = bool(excl_neg(d["reliance_change"]) and excl_pos(d["shortcut_part"])
                            and has0(b["reliance_change"]))

    # H9: same remedy, two mechanisms (v3 runs vs v4 whole), paired resamples
    v3 = lambda c, w: f"results/pilots/pilot_v3/planted_mid/{w}/{c}/seed*"
    PM3 = {}
    for c in ("A_none", "A_dropout"):
        for w in ("S", "I"):
            m, _ = collect(v3(c, w), lambda d: "x", args.jobs)
            PM3[(c, w)] = m["x"]
    R3 = {k: lg(v[:, 1]) - lg(v[:, 2]) for k, v in PM3.items()}
    if (t, "S", "A_dropout") in R:
        rc_whole = R[(t, "S", "A_dropout")] - R[(t, "S", "A_none")]
        rc_runs = R3[("A_dropout", "S")] - R3[("A_none", "S")]
        h9 = {"reliance_change_whole": stat(rc_whole), "reliance_change_runs": stat(rc_runs),
              "contrast": stat(rc_whole - rc_runs)}
        h9["pass"] = bool(excl_neg(h9["contrast"]))
        S["H9"] = h9
    json.dump(S, open(f"{OUT}/summary.json", "w"), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
