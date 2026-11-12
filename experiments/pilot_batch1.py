"""Pilot batch 1 (CPU): H4 and the child-level confounding check, protocol pilot_asdpose_v1.

Scores every SAFE-SMM test-score file under three protocols (standard, QC, reweighted), with
paired child-clustered bootstrap CIs, d' gaps and the binormal slope. Writes
results/pilots/pilot_batch1/{models.csv, child_confounding.json, summary.json}.

Usage: conda run -n torch python experiments/pilot_batch1.py [--n-boot 10000] [--jobs 10]
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import metrics as M
from safe_motion.data import asdpose as D

PROTO = "configs/protocols/pilot_asdpose_v1.yaml"
SCORE_DIR = f"{D.SMM_ROOT}/results/segment_scores"
OUT = "results/pilots/pilot_batch1"

G = {}


def setup(n_boot):
    cfg = yaml.safe_load(open(PROTO))
    rows = [r for r in D.load_index() if r["split"] == "test"]
    ids = [r["identifier"] for r in rows]
    y = np.array([r["binary_label"] for r in rows])
    child = np.array([r["child_id"] for r in rows])
    q = np.array([[r[k] for k in cfg["tracking_stats"]] for r in rows])
    qc = np.array([r["binary_label"] == 1 or D.passes_qc(r, cfg["qc_filter"]) for r in rows])
    w, p = M.propensity_weights(q, y, clip_pct=99.0)   # protocol: clip at 99th percentile
    boots = list(M.cluster_bootstrap_indices(child, n_boot, seed=0))
    G.update(rows=rows, ids=ids, y=y, child=child, q=q, qc=qc, w=w, p=p, boots=boots, cfg=cfg)


def three(s, idx):
    y, qc, w = G["y"][idx], G["qc"][idx], G["w"][idx]
    s = s[idx]
    return M.auc(s, y), M.auc(s[qc], y[qc]), M.weighted_auc(s, y, w)


def score_model(path):
    sc = {r["identifier"]: float(r["smm_score"]) for r in csv.DictReader(open(path))}
    if not all(i in sc for i in G["ids"]):
        return None
    s = np.array([sc[i] for i in G["ids"]])
    full = np.arange(len(s))
    a_std, a_qc, a_rw = three(s, full)
    bs = np.array([three(s, idx) for idx in G["boots"]])
    d = M.dprime(bs)
    name = os.path.basename(path).replace("_test_scores.csv", "")
    out = {"model": name, "auc_standard": a_std, "auc_qc": a_qc, "auc_reweighted": a_rw,
           "binormal_slope": M.binormal_slope(s, G["y"]),
           "gap_dprime_qc": float(M.dprime(a_std) - M.dprime(a_qc)),
           "gap_dprime_rw": float(M.dprime(a_std) - M.dprime(a_rw)),
           "rw_minus_qc": a_rw - a_qc}
    for k, col in (("auc_standard", 0), ("auc_qc", 1), ("auc_reweighted", 2)):
        out[k + "_ci"] = M.ci(bs[:, col])
    out["gap_dprime_qc_ci"] = M.ci(d[:, 0] - d[:, 1])
    out["gap_dprime_rw_ci"] = M.ci(d[:, 0] - d[:, 2])
    out["rw_minus_qc_ci"] = M.ci(bs[:, 2] - bs[:, 1])
    return out


def child_confounding(cfg):
    """Tracking stats -> label, raw and within-child centred; fit on train, AUC on test."""
    rows = D.load_index()
    res = {}
    for mode in ("raw", "within_child"):
        X, Y, S, C = {}, {}, {}, {}
        for split in ("train", "test"):
            rs = [r for r in rows if r["split"] == split]
            q = np.array([[r[k] for k in cfg["tracking_stats"]] for r in rs])
            c = np.array([r["child_id"] for r in rs])
            if mode == "within_child":
                for k in np.unique(c):
                    q[c == k] -= q[c == k].mean(0)
            X[split], Y[split], C[split] = q, np.array([r["binary_label"] for r in rs]), c
        sc = StandardScaler().fit(X["train"])
        lr = LogisticRegression(max_iter=2000).fit(sc.transform(X["train"]), Y["train"])
        s = lr.predict_proba(sc.transform(X["test"]))[:, 1]
        bs = [M.auc(s[i], Y["test"][i]) for i in G["boots"]]
        res[mode] = {"auc": M.auc(s, Y["test"]), "ci": M.ci(bs)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    setup(args.n_boot)
    cfg = G["cfg"]
    neg = G["y"] == 0
    balance = M.weighted_auc(G["p"], G["y"], G["w"])
    summary = {"n_test": int(len(G["y"])), "n_boot": args.n_boot,
               "propensity_auc_unweighted": M.auc(G["p"], G["y"]),
               "balance_check_weighted_auc": balance,
               "balance_check_pass": bool(0.45 <= balance <= 0.55),
               "ess_negatives": M.kish_ess(G["w"][neg]), "n_negatives": int(neg.sum()),
               "n_qc_negatives": int((G["qc"] & neg).sum())}
    print(json.dumps(summary, indent=1), flush=True)

    files = sorted(f for f in glob.glob(f"{SCORE_DIR}/*_test_scores.csv"))
    with Pool(args.jobs) as pool:
        results = [r for r in pool.map(score_model, files) if r is not None]
    print(f"scored {len(results)} of {len(files)} files", flush=True)

    keys = list(results[0].keys())
    with open(f"{OUT}/models.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in results:
            wr.writerow({k: (f"[{v[0]:.4f}, {v[1]:.4f}]" if isinstance(v, tuple)
                             else (f"{v:.4f}" if isinstance(v, float) else v))
                         for k, v in r.items()})

    agree = [r["rw_minus_qc_ci"][0] <= 0 <= r["rw_minus_qc_ci"][1] for r in results]
    rho = spearmanr([r["auc_reweighted"] for r in results], [r["auc_qc"] for r in results])[0]
    summary.update({"n_models": len(results), "share_ci_contains_zero": float(np.mean(agree)),
                    "spearman_rw_vs_qc": float(rho)})
    summary["H4_pass"] = bool(summary["share_ci_contains_zero"] >= 0.8 and rho >= 0.9)
    summary["child_confounding"] = child_confounding(cfg)
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
