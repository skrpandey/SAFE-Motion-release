"""Post hoc re-audit of the host under the final length-controlled protocol.

(a) quality-only AUROC on the four natural tracking statistics (fit on host training children,
child-clustered bootstrap); (b) realized planted severities (missing runs, mid) by class, world and
split, with mask seed 0 for train/validation and 2000 for test; (c) a data-flow table of counts.
Writes results/v7/analysis/{host_reaudit,severities_realized,dataflow}.csv.
"""
import csv
import os

import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from safe_motion import metrics as M
from safe_motion import planted as P
from safe_motion.data import asdpose as D

A = "results/v7/analysis"
V1 = yaml.safe_load(open("configs/protocols/pilot_asdpose_v1.yaml"))
STATS = ["q_lowconf", "q_missing", "q_frozen", "q_arm_lowconf"]


def feats(rows):
    return np.array([[r[k] for k in STATS] for r in rows]), np.array([r["binary_label"] for r in rows])


def auc_ci(s, y, child, n_boot=2000):
    boots = list(M.cluster_bootstrap_indices(np.asarray(child), n_boot, seed=0))
    vals = np.array([M.auc(s[i], y[i]) for i in boots])
    lo, hi = M.ci(vals)
    return M.auc(s, y), lo, hi


def flow(name, rows):
    y = np.array([r["binary_label"] for r in rows]) if rows else np.array([])
    return {"stage": name, "segments": len(rows), "children": len({r["child_id"] for r in rows}),
            "smm": int(y.sum()) if len(y) else 0, "non_smm": int((y == 0).sum()) if len(y) else 0}


def main():
    all_rows = D.load_index()
    qc = V1["qc_filter"]
    length = [r for r in all_rows if r["n_frames"] >= 200]
    tr_full, va_full, val_children = D.subject_level_val_split(length, 0.2, seed=0)
    te_full = [r for r in length if r["split"] == "test"]
    host_tr = [r for r in tr_full if D.passes_qc(r, qc)]
    host_va = [r for r in va_full if D.passes_qc(r, qc)]
    host_te = [r for r in te_full if D.passes_qc(r, qc)]
    os.makedirs(A, exist_ok=True)

    # (c) data flow
    stages = [("release (cache index)", all_rows),
              ("release, train split", [r for r in all_rows if r["split"] == "train"]),
              ("release, test split", [r for r in all_rows if r["split"] == "test"]),
              ("QC pass (both classes), all", [r for r in all_rows if D.passes_qc(r, qc)]),
              ("length protocol (n_frames >= 200), all", length),
              ("real train (length, train children minus validation)", tr_full),
              ("real validation (length, 20 percent of train children)", va_full),
              ("real test (length)", te_full),
              ("host train (length + QC pass)", host_tr),
              ("host validation (length + QC pass)", host_va),
              ("host test (length + QC pass)", host_te)]
    with open(f"{A}/dataflow.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "segments", "children", "smm", "non_smm"]); w.writeheader()
        for name, rows in stages:
            w.writerow(flow(name, rows))
    print("validation children:", len(val_children))

    # (a) quality-only audit on the final host
    Xtr, ytr = feats(host_tr)
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
    gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0).fit(Xtr, ytr)
    out = []
    for split_name, rows in (("host validation", host_va), ("host test", host_te)):
        X, y = feats(rows); child = [r["child_id"] for r in rows]
        for name, s in (("logistic regression", lr.predict_proba(sc.transform(X))[:, 1]),
                        ("gradient boosting", gb.predict_proba(X)[:, 1])):
            p, lo, hi = auc_ci(s, y, child)
            out.append({"analysis": "post hoc", "split": split_name, "classifier": name, "n_segments": len(rows),
                        "n_children": len(set(child)), "auc": round(p, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                        "note": "fit on host training children; four natural tracking statistics; final 200-frame host"})
    with open(f"{A}/host_reaudit.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    for r in out:
        print(r)

    # (b) realized severities, missing runs at mid
    sev = []
    for split_name, rows, seed in (("host train", host_tr, 0), ("host validation", host_va, 0), ("host test", host_te, 2000)):
        y = np.array([r["binary_label"] for r in rows])
        for world in ("S", "I"):
            e = P.severities(rows, world, "mid", seed)
            for cls, k in (("non-SMM", 0), ("SMM", 1), ("all", None)):
                v = e if k is None else e[y == k]
                sev.append({"split": split_name, "world": world, "seed": seed, "class": cls, "n": len(v),
                            "mean": round(float(v.mean()), 4), "sd": round(float(v.std()), 4),
                            "q10": round(float(np.quantile(v, .1)), 4), "q50": round(float(np.quantile(v, .5)), 4),
                            "q90": round(float(np.quantile(v, .9)), 4)})
    with open(f"{A}/severities_realized.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=list(sev[0])); w.writeheader(); w.writerows(sev)
    for r in sev:
        if r["class"] != "all":
            print(r)


if __name__ == "__main__":
    main()
