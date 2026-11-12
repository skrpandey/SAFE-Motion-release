"""Pilot v5 analysis (protocol pilot_asdpose_v5.yaml): matched-world R on real ASDPose models.

Per model group (seed-averaged): AUC_clean, raw Δ and logit gap (standard vs QC), R per
mechanism and strength on the full test split and on the QC-sym subset. Paired child-clustered
bootstrap, the same resample indices for every model. H10 to H13 and the permutation null.
Writes results/pilots/pilot_v5/{groups.csv, summary.json}.
Usage: conda run -n torch python experiments/analyze_v5.py [--n-boot 10000] [--jobs 10]
"""

import argparse
import csv
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import yaml
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import metrics as M
from safe_motion.data import asdpose as D

PROTO = "configs/protocols/pilot_asdpose_v5.yaml"
V1 = yaml.safe_load(open("configs/protocols/pilot_asdpose_v1.yaml"))
OUT = "results/pilots/pilot_v5"
CONDS = ["runs_mid", "runs_low", "whole_mid", "whole_low"]
G = {}


def logit(a):
    a = np.clip(np.asarray(a, float), 1e-6, 1 - 1e-6)
    return np.log(a / (1 - a))


def setup(n_boot):
    rows = [r for r in D.load_index() if r["split"] == "test"]
    G["ids"] = [r["identifier"] for r in rows]
    G["y"] = np.array([r["binary_label"] for r in rows])
    good = np.array([D.passes_qc(r, V1["qc_filter"]) for r in rows])
    G["qc"] = (G["y"] == 1) | good
    G["sym"] = good
    G["boots"] = [np.arange(len(rows))] + list(
        M.cluster_bootstrap_indices([r["child_id"] for r in rows], n_boot, seed=0))


def model_metrics(path):
    """(n_boot + 1, k) array; row 0 = point estimate. Columns: see COLS."""
    rd = list(csv.DictReader(open(path)))
    pos = {r["identifier"]: i for i, r in enumerate(rd)}
    order = [pos[i] for i in G["ids"]]
    cols = [c for c in rd[0] if c not in ("identifier", "binary_label")]
    S = {c: np.array([float(rd[j][c]) for j in order]) for c in cols}
    y, qc, sym = G["y"], G["qc"], G["sym"]
    out = []
    for idx in G["boots"]:
        yi = y[idx]
        row = [M.auc(S["clean"][idx], yi),
               M.auc(S["clean"][idx][qc[idx]], yi[qc[idx]])]
        for c in CONDS:
            for sub in (None, sym):
                m = np.ones(len(idx), bool) if sub is None else sub[idx]
                row.append(M.auc(S[c + "_S"][idx][m], yi[m]))
                row.append(M.auc(S[c + "_I"][idx][m], yi[m]))
        out.append(row)
    return np.array(out)


COLS = ["auc_std", "auc_qc"] + [f"{c}_{v}_{w}" for c in CONDS for v in ("full", "sym")
                                for w in ("S", "I")]


def groups_of(names, cfg):
    """model name -> group name (seed families averaged)."""
    g = {}
    sg = cfg["models"]["safe_smm_checkpoints"]["seed_groups"]
    for grp, members in sg.items():
        for n in members:
            g[n] = grp
    for n in names:
        if n in g:
            continue
        if n.startswith("tcn_") and "_perm" in n:
            g[n] = "tcn_perm_null"
        elif n.startswith("tcn_") or n.startswith("compact_v4_"):
            g[n] = n.rsplit("_s", 1)[0]
        else:
            g[n] = n
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(PROTO))
    setup(args.n_boot)
    files = sorted(glob.glob(f"{OUT}/scores/*.csv"))
    names = [os.path.basename(f)[:-4] for f in files]
    with Pool(args.jobs) as pool:
        per_model = dict(zip(names, pool.map(model_metrics, files)))
    grp = groups_of(names, cfg)
    members = {}
    for n in names:
        members.setdefault(grp[n], []).append(n)
    PM = {g: np.mean([per_model[n] for n in ms], 0) for g, ms in members.items()
          if g != "tcn_perm_null"}
    ci = lambda a: M.ci(a[1:])
    stat = lambda a: {"point": float(a[0]), "ci": ci(a)}
    col = {c: i for i, c in enumerate(COLS)}

    def R(g, cond, view="full"):
        a = PM[g]
        return logit(a[:, col[f"{cond}_{view}_S"]]) - logit(a[:, col[f"{cond}_{view}_I"]])

    def lgap(g):
        return logit(PM[g][:, 0]) - logit(PM[g][:, 1])

    rows = []
    for g in sorted(PM):
        a = PM[g]
        r = {"group": g, "n_seeds": len(members[g]), "auc_clean": float(a[0, 0]),
             "auc_qc": float(a[0, 1]), "delta": float(a[0, 0] - a[0, 1]),
             "logit_gap": stat(lgap(g))}
        for c in CONDS:
            r[f"R_{c}"] = stat(R(g, c))
            r[f"R_{c}_sym"] = stat(R(g, c, "sym"))
        rows.append(r)
    with open(f"{OUT}/groups.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "n_seeds", "auc_clean", "auc_qc", "delta", "logit_gap", "logit_gap_ci",
                    "R_runs_mid", "R_runs_mid_ci", "R_runs_low", "R_whole_mid", "R_whole_low",
                    "R_runs_mid_sym"])
        for r in rows:
            w.writerow([r["group"], r["n_seeds"], f"{r['auc_clean']:.4f}", f"{r['auc_qc']:.4f}",
                        f"{r['delta']:.4f}", f"{r['logit_gap']['point']:.4f}",
                        "[%.4f, %.4f]" % r["logit_gap"]["ci"],
                        f"{r['R_runs_mid']['point']:.4f}", "[%.4f, %.4f]" % r["R_runs_mid"]["ci"],
                        f"{r['R_runs_low']['point']:.4f}", f"{r['R_whole_mid']['point']:.4f}",
                        f"{r['R_whole_low']['point']:.4f}", f"{r['R_runs_mid_sym']['point']:.4f}"])

    excl_pos = lambda s: s["ci"][0] > 0
    excl_neg = lambda s: s["ci"][1] < 0
    has0 = lambda s: s["ci"][0] <= 0 <= s["ci"][1]
    S = {"groups": {r["group"]: r["n_seeds"] for r in rows}}
    gq = "tcn_D_qcsym_none"
    if gq in PM:
        h = {"R_runs_mid": stat(R(gq, "runs_mid")), "logit_gap": stat(lgap(gq)),
             "R_whole_mid": stat(R(gq, "whole_mid"))}
        h["pass"] = bool(has0(h["R_runs_mid"]) and excl_pos(h["logit_gap"]))
        S["H10"] = h
    if "tcn_A_none" in PM and "stage1_detector" in PM:
        h = {"tcn_A_none": stat(R("tcn_A_none", "runs_mid")),
             "stage1_detector": stat(R("stage1_detector", "runs_mid"))}
        h["pass"] = bool(excl_pos(h["tcn_A_none"]) and excl_pos(h["stage1_detector"]))
        S["H11"] = h
    if {"tcn_A_none", "tcn_A_both"} <= set(PM):
        d = stat(R("tcn_A_both", "runs_mid") - R("tcn_A_none", "runs_mid"))
        h = {"both_minus_none": d, "pass": bool(excl_neg(d))}
        for g in ("tcn_B_coorddrop", "tcn_A_velocity", "tcn_A_dropout", "tcn_D_qcsym_velocity"):
            if g in PM:
                h[f"{g}_minus_none"] = stat(R(g, "runs_mid") - R("tcn_A_none", "runs_mid"))
        h["safe_smm_v1_v2_v3"] = {g: stat(R(g, "runs_mid")) for g in
                                  ("stage1_detector", "v2_family", "stage1_detector_v3") if g in PM}
        h["v3_minus_v1"] = stat(R("stage1_detector_v3", "runs_mid") - R("stage1_detector", "runs_mid"))
        S["H12"] = h
    add = {g: {"R_runs_mid": stat(R(g, "runs_mid")), "R_whole_mid": stat(R(g, "whole_mid")),
               "R_runs_mid_sym": stat(R(g, "runs_mid", "sym")), "auc_clean": float(PM[g][0, 0])}
           for g in ("tcn_v3_I", "tcn_v3_S", "compact_v4_I", "compact_v4_S") if g in PM}
    if len(add) == 4:
        add["H10b_pass"] = bool(has0(add["tcn_v3_I"]["R_runs_mid"]) and has0(add["compact_v4_I"]["R_runs_mid"])
                                and excl_pos(add["tcn_v3_S"]["R_runs_mid"]) and excl_pos(add["compact_v4_S"]["R_runs_mid"]))
        S["addendum1"] = add
    smm = [g for g in PM if not g.startswith("tcn_") and not g.startswith("compact_v4_")
           and "permuted" not in g]
    S["H13"] = {"n": len(smm),
                "spearman_R_vs_logit_gap": float(spearmanr([R(g, "runs_mid")[0] for g in smm],
                                                           [lgap(g)[0] for g in smm])[0]),
                "spearman_R_vs_delta": float(spearmanr([R(g, "runs_mid")[0] for g in smm],
                                                       [PM[g][0, 0] - PM[g][0, 1] for g in smm])[0]),
                "table": {g: {"R": float(R(g, "runs_mid")[0]), "logit_gap": float(lgap(g)[0]),
                              "delta": float(PM[g][0, 0] - PM[g][0, 1]),
                              "auc_clean": float(PM[g][0, 0])} for g in smm}}
    if "tcn_perm_null" in members:
        null = np.array([logit(per_model[n][0, col["runs_mid_full_S"]])
                         - logit(per_model[n][0, col["runs_mid_full_I"]])
                         for n in members["tcn_perm_null"]])
        obs = R("tcn_A_none", "runs_mid")[0]
        S["permutation_null"] = {"R_null": null.tolist(), "A_none_R": float(obs),
                                 "p": float(((null >= obs).sum() + 1) / (len(null) + 1))}
    S["accuracy_blindness"] = {g: {"auc_clean": float(PM[g][0, 0]), "R": float(R(g, "runs_mid")[0])}
                               for g in PM}
    json.dump(S, open(f"{OUT}/summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in S.items() if k not in ("accuracy_blindness",)}, indent=1))


if __name__ == "__main__":
    main()
