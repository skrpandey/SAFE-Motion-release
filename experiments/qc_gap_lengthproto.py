"""Filtered QC gap (AUC on QC-pass minus AUC on QC-fail test segments) under the length protocol,
for the 15 v7 real CompactCTRGCN models (from comparators.csv) and the 15 v10 real-data World I
models (from their score files). The v7 host models were scored on the host test split only, so
their gap is reported as not available.
Writes results/v7/analysis/qc_gap_lengthproto.csv and qc_gap_lengthproto_summary.csv.
"""
import csv
import glob
import os

import numpy as np
from sklearn.metrics import roc_auc_score


def gap(path):
    rd = list(csv.DictReader(open(path)))
    y = np.array([int(r["label"]) for r in rd]); qc = np.array([int(r["qc_pass"]) for r in rd])
    s = np.array([float(r["clean"]) for r in rd])
    return roc_auc_score(y[qc == 1], s[qc == 1]) - roc_auc_score(y[qc == 0], s[qc == 0]), len(rd)


rows = []
for r in csv.DictReader(open("results/v7/analysis/comparators.csv")):
    rows.append({"set": "v7_real (length protocol)", "model": r["model"], "n_rows": "", "qc_gap": float(r["filtered_gap"])})
for p in sorted(glob.glob("results/v10/scores/v10_real_I_*.csv")):
    g, n = gap(p)
    rows.append({"set": "v10_real_I (length protocol)", "model": os.path.basename(p)[:-4], "n_rows": n, "qc_gap": g})
host_full = [p for p in glob.glob("results/v7/scores/v7_I_w0_*A_none*.csv") if sum(1 for _ in open(p)) > 3000]
summary = []
for name in ("v7_real (length protocol)", "v10_real_I (length protocol)"):
    g = np.array([r["qc_gap"] for r in rows if r["set"] == name])
    summary.append({"set": name, "n_models": len(g), "min": round(g.min(), 4), "max": round(g.max(), 4),
                    "mean": round(g.mean(), 4), "sd": round(g.std(ddof=0), 4), "max_abs": round(np.abs(g).max(), 4)})
summary.append({"set": "v7 World I host A_none on the full split", "n_models": len(host_full),
                "min": "", "max": "", "mean": "", "sd": "",
                "max_abs": "not available: host models were scored on the host test split only (1,966 rows)"})
os.makedirs("results/v7/analysis", exist_ok=True)
with open("results/v7/analysis/qc_gap_lengthproto.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["set", "model", "n_rows", "qc_gap"]); w.writeheader()
    for r in rows:
        r["qc_gap"] = round(r["qc_gap"], 4); w.writerow(r)
with open("results/v7/analysis/qc_gap_lengthproto_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
for s in summary:
    print(s)
