"""Pilot v2 analysis (protocol pilot_asdpose_v2.yaml).

Seed-averaged metrics on four test protocols, paired child-clustered CIs (same resample indices
for every seed and cell), H1 and H5 decisions, symmetric-pool check and D cells, permutation null.
Writes results/pilots/pilot_v2/{cells.csv, summary.json}.
Usage: conda run -n torch python experiments/analyze_v2.py [--n-boot 10000]
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import yaml
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import metrics as M
from safe_motion.data import asdpose as D

V1 = yaml.safe_load(open("configs/protocols/pilot_asdpose_v1.yaml"))
OUT = "results/pilots/pilot_v2"
PROTS = ["standard", "qc", "qc_sym", "reweighted"]


def logit(a):
    a = np.clip(a, 1e-6, 1 - 1e-6)
    return np.log(a / (1 - a))


def d_a(s, y, n_points=50):
    ths = np.quantile(s, np.linspace(0.02, 0.98, n_points))
    tpr = np.array([(s[y == 1] > t).mean() for t in ths])
    fpr = np.array([(s[y == 0] > t).mean() for t in ths])
    ok = (tpr > 0.005) & (tpr < 0.995) & (fpr > 0.005) & (fpr < 0.995)
    b, a = np.polyfit(norm.ppf(fpr[ok]), norm.ppf(tpr[ok]), 1)
    return float(np.sqrt(2 / (1 + b * b)) * a)


class Test:
    def __init__(self, n_boot):
        rows = [r for r in D.load_index() if r["split"] == "test"]
        qcf = V1["qc_filter"]
        self.ids = [r["identifier"] for r in rows]
        self.y = np.array([r["binary_label"] for r in rows])
        good = np.array([D.passes_qc(r, qcf) for r in rows])
        self.masks = {"standard": np.ones(len(rows), bool), "qc": (self.y == 1) | good,
                      "qc_sym": good}
        q = np.array([[r[k] for k in V1["tracking_stats"]] for r in rows])
        self.w, _ = M.propensity_weights(q, self.y, clip_pct=99.0)
        child = np.array([r["child_id"] for r in rows])
        self.boots = list(M.cluster_bootstrap_indices(child, n_boot, seed=0))

    def metrics(self, s, idx):
        y, out = self.y[idx], []
        s = s[idx]
        for p in ("standard", "qc", "qc_sym"):
            m = self.masks[p][idx]
            out.append(M.auc(s[m], y[m]))
        out.append(M.weighted_auc(s, y, self.w[idx]))
        return np.array(out)


def load_scores(path, ids):
    sc = {r["identifier"]: float(r["smm_score"]) for r in csv.DictReader(open(path))}
    return np.array([sc[i] for i in ids])


def pool_check():
    rows = D.load_index()
    tr, va, _ = D.subject_level_val_split(rows, 0.2, seed=0)
    tr = [r for r in tr if D.passes_qc(r, V1["qc_filter"])]
    va = [r for r in va if D.passes_qc(r, V1["qc_filter"])]
    f = lambda rs: np.array([[r[k] for k in V1["tracking_stats"]] for r in rs])
    sc = StandardScaler().fit(f(tr))
    lr = LogisticRegression(max_iter=2000).fit(sc.transform(f(tr)), [r["binary_label"] for r in tr])
    a = M.auc(lr.predict_proba(sc.transform(f(va)))[:, 1], [r["binary_label"] for r in va])
    return {"quality_only_auc_val": a, "weak_cue_pool": bool(0.40 <= a <= 0.60),
            "n_train": len(tr), "n_train_pos": int(sum(r["binary_label"] for r in tr))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    T = Test(args.n_boot)
    full = np.arange(len(T.y))

    cells = sorted(d for d in os.listdir(OUT) if d != "perm" and os.path.isdir(f"{OUT}/{d}"))
    point, boot, extra = {}, {}, {}
    for c in cells:
        files = sorted(glob.glob(f"{OUT}/{c}/seed*/test_scores.csv"))
        ps, bs, das, slopes = [], [], [], []
        for f in files:
            s = load_scores(f, T.ids)
            ps.append(T.metrics(s, full))
            bs.append(np.array([T.metrics(s, i) for i in T.boots]))
            das.append([d_a(s[T.masks[p]], T.y[T.masks[p]]) for p in ("standard", "qc")])
            slopes.append(M.binormal_slope(s, T.y))
        point[c], boot[c] = np.mean(ps, 0), np.mean(bs, 0)     # seed-averaged
        das = np.array(das)
        extra[c] = {"n_seeds": len(files), "per_seed_auc_qc": [float(p[1]) for p in ps],
                    "d_a_gap": float(np.mean(das[:, 0] - das[:, 1])),
                    "binormal_slope": float(np.mean(slopes))}
        print(c, len(files), np.round(point[c], 4), flush=True)

    def lg(x, j=1):
        return logit(x[..., 0]) - logit(x[..., j])

    rows = []
    for c in cells:
        p, b = point[c], boot[c]
        row = {"cell": c, **{f"auc_{k}": float(p[j]) for j, k in enumerate(PROTS)}}
        for j, k in enumerate(PROTS):
            row[f"auc_{k}_ci"] = M.ci(b[:, j])
        row["logit_gap_qc"] = float(lg(p))
        row["logit_gap_qc_ci"] = M.ci(lg(b))
        row["delta_qc"] = float(p[0] - p[1])
        row.update(extra[c])
        rows.append(row)
    with open(f"{OUT}/cells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"[{v[0]:.4f}, {v[1]:.4f}]" if isinstance(v, tuple)
                            else (f"{v:.4f}" if isinstance(v, float) else v)) for k, v in r.items()})

    def diff(a, b, j):
        return {"point": float(point[a][j] - point[b][j]), "ci": M.ci(boot[a][:, j] - boot[b][:, j])}

    excl = lambda ci: ci[0] > 0 or ci[1] < 0
    S = {"cells": {c: extra[c]["n_seeds"] for c in cells}, "pool_check": pool_check()}

    d_std, d_qc = diff("A_both", "A_none", 0), diff("A_both", "A_none", 1)
    d_lg = {"point": float(lg(point["A_both"]) - lg(point["A_none"])),
            "ci": M.ci(lg(boot["A_both"]) - lg(boot["A_none"]))}
    S["H1"] = {"auc_standard": d_std, "auc_qc": d_qc, "logit_gap": d_lg,
               "pass": bool(d_std["point"] > 0 and d_qc["point"] > 0 and d_lg["point"] < 0
                            and d_qc["ci"][0] > 0)}

    bn, bb = diff("B_coorddrop", "A_none", 1), diff("A_both", "B_coorddrop", 1)
    if not excl(bn["ci"]) and bb["ci"][0] > 0:
        verdict = "supports_shortcut_removal"
    elif bn["ci"][0] > 0 and bn["point"] >= 0.5 * d_qc["point"]:
        verdict = "supports_regularization"
    else:
        verdict = "inconclusive"
    S["H5"] = {"coorddrop_minus_none_auc_qc": bn, "both_minus_coorddrop_auc_qc": bb,
               "coorddrop_share_of_gain": bn["point"] / d_qc["point"], "verdict": verdict}
    S["shares_of_qc_gain"] = {c: float((point[c][1] - point["A_none"][1]) / d_qc["point"])
                              for c in ("A_velocity", "A_dropout", "B_coorddrop") if c in point}
    if {"D_qcsym_none", "D_qcsym_velocity"} <= set(cells):
        S["D_cells"] = {"velocity_minus_none_auc_qc_sym": diff("D_qcsym_velocity", "D_qcsym_none", 2),
                        "velocity_minus_none_auc_standard": diff("D_qcsym_velocity", "D_qcsym_none", 0)}

    perm = sorted(glob.glob(f"{OUT}/perm/*/test_scores.csv"))
    if perm:
        null = np.array([T.metrics(load_scores(f, T.ids), full) for f in perm])
        ng = lg(null)

        def pval(obs, dist, greater=True):
            k = (dist >= obs).sum() if greater else (dist <= obs).sum()
            return float((k + 1) / (len(dist) + 1))
        S["permutation_null"] = {
            "n": len(perm), "auc_standard": null[:, 0].tolist(), "auc_qc": null[:, 1].tolist(),
            "logit_gap": ng.tolist(),
            "p_auc_qc": {c: pval(point[c][1], null[:, 1]) for c in ("A_none", "A_both")},
            "p_logit_gap": {c: pval(float(lg(point[c])), ng) for c in ("A_none", "A_both")}}
    json.dump(S, open(f"{OUT}/summary.json", "w"), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
