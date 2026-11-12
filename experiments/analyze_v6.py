"""Pilot v6 analysis (protocol pilot_asdpose_v6.yaml): V1 to V7 on the host test split.

Seed-averaged groups, paired child-clustered bootstrap with shared indices. Writes
results/pilots/pilot_v6/{groups.csv, summary.json}.
Usage: conda run -n torch python experiments/analyze_v6.py [--n-boot 10000] [--jobs 10]
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_v6 as S6
from safe_motion import metrics as M
from safe_motion.data import asdpose as D

V1 = yaml.safe_load(open("configs/protocols/pilot_asdpose_v1.yaml"))
OUT = "results/pilots/pilot_v6"
G = {}
COLS = ["clean"] + [S6.cname(*c) for c in S6.conditions()]


def logit(a):
    a = np.clip(np.asarray(a, float), 1e-6, 1 - 1e-6)
    return np.log(a / (1 - a))


def setup(n_boot):
    rows = [r for r in D.load_index() if r["split"] == "test" and D.passes_qc(r, V1["qc_filter"])]
    G["ids"] = [r["identifier"] for r in rows]
    G["y"] = np.array([r["binary_label"] for r in rows])
    G["boots"] = [np.arange(len(rows))] + list(
        M.cluster_bootstrap_indices([r["child_id"] for r in rows], n_boot, seed=0))


def model_metrics(path):
    rd = list(csv.DictReader(open(path)))
    pos = {r["identifier"]: i for i, r in enumerate(rd)}
    order = [pos[i] for i in G["ids"]]
    S = {c: np.array([float(rd[j][c]) for j in order]) for c in COLS}
    y = G["y"]
    aucs = np.array([[M.auc(S[c][idx], y[idx]) for c in COLS] for idx in G["boots"]])
    # decision flips and paired logit change, zero S vs I, mid, seed 2000 (point estimates)
    s_, i_ = S["zero_S_mid_m2000"], S["zero_I_mid_m2000"]
    flip = float(((s_ > 0.5) != (i_ > 0.5)).mean())
    dl = logit(s_) - logit(i_)
    return aucs, {"flip_rate": flip, "logit_change_pos": float(dl[y == 1].mean()),
                  "logit_change_neg": float(dl[y == 0].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args()
    setup(args.n_boot)
    files = sorted(glob.glob(f"{OUT}/scores/*.csv"))
    names = [os.path.basename(f)[:-4] for f in files]
    with Pool(args.jobs) as pool:
        res = pool.map(model_metrics, files)
    per = dict(zip(names, res))
    grp = {n: re.sub(r"_s\d+$", "", n) for n in names}
    members = {}
    for n in names:
        members.setdefault(grp[n], []).append(n)
    PM = {g: np.mean([per[n][0] for n in ms], 0) for g, ms in members.items()}
    FL = {g: {k: float(np.mean([per[n][1][k] for n in ms])) for k in per[ms[0]][1]}
          for g, ms in members.items()}
    col = {c: i for i, c in enumerate(COLS)}
    stat = lambda a: {"point": float(a[0]), "ci": M.ci(a[1:])}
    has0 = lambda s: s["ci"][0] <= 0 <= s["ci"][1]
    excl_pos = lambda s: s["ci"][0] > 0
    excl_neg = lambda s: s["ci"][1] < 0

    def A(g, c):
        return PM[g][:, col[c]]

    def R(g, op="zero", st="mid", m=2000, a="S", b="I"):
        return logit(A(g, S6.cname(op, a, st, m))) - logit(A(g, S6.cname(op, b, st, m)))

    rows = []
    for g in sorted(PM):
        r = {"group": g, "n_seeds": len(members[g]), "auc_clean": float(A(g, "clean")[0])}
        for op in ("zero", "gapfill", "conflow"):
            r[f"R_{op}"] = stat(R(g, op))
            r[f"Rmirror_{op}"] = stat(R(g, op, b="Sp"))
        r["R_zero_low"] = stat(R(g, "zero", "low"))
        r["R_zero_masks"] = [float(R(g, "zero", "mid", m)[0]) for m in (2000, 3000, 4000)]
        r.update(FL[g])
        rows.append(r)
    with open(f"{OUT}/groups.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "n_seeds", "auc_clean", "R_zero", "R_zero_ci", "R_gapfill",
                    "R_gapfill_ci", "R_conflow", "R_conflow_ci", "Rmirror_zero", "R_zero_low",
                    "R_zero_masks", "flip_rate", "logit_change_pos", "logit_change_neg"])
        for r in rows:
            f2 = lambda s: f"{s['point']:.4f}"
            c2 = lambda s: "[%.4f, %.4f]" % s["ci"]
            w.writerow([r["group"], r["n_seeds"], f"{r['auc_clean']:.4f}", f2(r["R_zero"]),
                        c2(r["R_zero"]), f2(r["R_gapfill"]), c2(r["R_gapfill"]), f2(r["R_conflow"]),
                        c2(r["R_conflow"]), f2(r["Rmirror_zero"]), f2(r["R_zero_low"]),
                        " ".join(f"{x:.3f}" for x in r["R_zero_masks"]), f"{r['flip_rate']:.4f}",
                        f"{r['logit_change_pos']:.4f}", f"{r['logit_change_neg']:.4f}"])

    S = {"groups": {r["group"]: r["n_seeds"] for r in rows}}
    if os.path.exists(f"{OUT}/oracles.json"):
        S["oracles"] = json.load(open(f"{OUT}/oracles.json"))

    inv = [g for g in ("tcn_v3_I", "compact_v4_I") if g in PM]
    cue = [g for g in ("tcn_v3_S", "compact_v4_S") if g in PM]
    if len(inv) == 2 and len(cue) == 2:
        v1 = {g: {op: stat(R(g, op)) for op in ("zero", "gapfill", "conflow")} for g in inv}
        for g in inv:
            v1[g]["mirror_zero"] = stat(R(g, "zero", b="Sp"))
        v1["pass"] = bool(all(has0(v1[g][k]) for g in inv for k in v1[g]))
        S["V1_invariant_zero"] = v1
        v2 = {g: stat(R(g, "conflow")) for g in inv + cue}
        v2["pass"] = bool(all(excl_pos(v2[g]) for g in cue) and all(has0(v2[g]) for g in inv))
        S["V2_cue_only_operator"] = v2
    S["V3_information_loss"] = {g: stat(R(g, "gapfill")) for g in PM}
    if "tcn_v6_Sp_w0" in PM and "tcn_v3_S" in PM:
        v4 = {"R_Sp_trained": stat(R("tcn_v6_Sp_w0")),
              "SE_mirror": stat(R("tcn_v3_S") - R("tcn_v6_Sp_w0"))}
        v4["pass"] = bool(excl_neg(v4["R_Sp_trained"]) and excl_pos(v4["SE_mirror"]))
        S["V4_direction"] = v4
    if {"tcn_v3_S", "tcn_v3_I", "tcn_v3low_S", "tcn_v3low_I"} <= set(PM):
        se = {}
        for test_st in ("mid", "low"):
            se[test_st] = {"mid_trained": stat(R("tcn_v3_S", "zero", test_st) - R("tcn_v3_I", "zero", test_st)),
                           "low_trained": stat(R("tcn_v3low_S", "zero", test_st) - R("tcn_v3low_I", "zero", test_st))}
            se[test_st]["diff"] = stat((R("tcn_v3_S", "zero", test_st) - R("tcn_v3_I", "zero", test_st))
                                       - (R("tcn_v3low_S", "zero", test_st) - R("tcn_v3low_I", "zero", test_st)))
        se["pass"] = bool(excl_pos(se["mid"]["diff"]) and se["low"]["diff"]["point"] > 0)
        S["V5_crossed_strengths"] = se
    if {"tcn_v3_S", "tcn_v3_I"} <= set(PM):
        ses = [float((R("tcn_v3_S", "zero", "mid", m) - R("tcn_v3_I", "zero", "mid", m))[0])
               for m in (2000, 3000, 4000)]
        se0 = stat(R("tcn_v3_S") - R("tcn_v3_I"))
        half = (se0["ci"][1] - se0["ci"][0]) / 2
        v6 = {"SE_by_test_mask": ses, "SE_ci_halfwidth": half,
              "test_mask_pass": bool(max(ses) - min(ses) < half)}
        if "tcn_v3_S_s0" in per:
            se_s0 = stat(logit(per["tcn_v3_S_s0"][0][:, col["zero_S_mid_m2000"]])
                         - logit(per["tcn_v3_S_s0"][0][:, col["zero_I_mid_m2000"]]) - R("tcn_v3_I"))
            v6["SE_seed0_world0"] = se_s0
            tw = {}
            for w in (1, 2):
                g = f"tcn_v6_S_w{w}"
                if g in PM:
                    tw[g] = float((R(g) - R("tcn_v3_I"))[0])
            v6["SE_training_mask_worlds"] = tw
            v6["training_mask_pass"] = bool(all(se_s0["ci"][0] <= v <= se_s0["ci"][1] for v in tw.values()))
            v6["pass"] = bool(v6["test_mask_pass"] and v6["training_mask_pass"])
        S["V6_mask_realizations"] = v6
    v7 = {}
    for a, b in (("tcn_v3_S", "tcn_v3_I"), ("compact_v4_S", "compact_v4_I")):
        if a in PM and b in PM:
            v7[a] = stat(A(a, "clean") - A(b, "clean"))
    if v7:
        v7["pass"] = bool(all(-0.02 <= s["ci"][0] and s["ci"][1] <= 0.02 for s in v7.values()))
        S["V7_clean_equivalence"] = v7
    json.dump(S, open(f"{OUT}/summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in S.items() if k not in ("groups", "V3_information_loss", "oracles")}, indent=1))


if __name__ == "__main__":
    main()
