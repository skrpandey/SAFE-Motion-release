"""Post hoc check of the sign of the per-fold Platt slopes of experiments/calibration_v5.py.

The Platt fit is unconstrained, so a non-positive slope would reverse or flatten the ranking
inside a fold. Refits the same per-fold models and records every slope and intercept with the
within-fold AUROC before and after Platt. Writes results/v11/analysis/platt_slopes.csv and
prints the count of non-positive slopes.
    conda run -n torch python experiments/check_platt_slopes.py
"""
import csv
import glob
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "experiments")
from analyze_v11 import EXTERNAL_CACHE, load_scores, logit  # noqa: E402
from safe_motion import metrics as M  # noqa: E402
from safe_motion.data import asdpose as D  # noqa: E402

A = "results/v11/analysis"


def main():
    rows = D.load_index(f"{EXTERNAL_CACHE}/index.csv")
    folds = json.load(open("results/v11/folds.json"))["folds"]
    y = np.array([r["binary_label"] for r in rows]); child = np.array([r["child_id"] for r in rows])
    fold_te = [np.array([c in set(fv) for c in child]) for fv in folds]
    out = []
    for path in sorted(glob.glob("results/v11/scores/*.csv")):
        name = os.path.basename(path)[:-4]
        _, cols = load_scores(path)
        p = np.clip(cols["clean"], 1e-6, 1 - 1e-6); z = logit(p)
        platt = np.zeros(len(y))
        for k, te in enumerate(fold_te):
            tr = ~te
            lr = LogisticRegression(C=1e6, max_iter=2000).fit(z[tr, None], y[tr])
            platt[te] = lr.predict_proba(z[te, None])[:, 1]
            out.append({"model": name, "fold": k, "slope": float(lr.coef_[0, 0]), "intercept": float(lr.intercept_[0]),
                        "n_train": int(tr.sum()), "n_test": int(te.sum()),
                        "auc_fold_before": round(M.auc(p[te], y[te]), 4),
                        "auc_fold_after_platt": round(M.auc(platt[te], y[te]), 4)})
        for r in out[-len(fold_te):]:
            r["auc_pooled_before"] = round(M.auc(p, y), 4); r["auc_pooled_after_platt"] = round(M.auc(platt, y), 4)
    with open(f"{A}/platt_slopes.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    s = np.array([r["slope"] for r in out])
    neg = [r for r in out if r["slope"] <= 0]
    within = max(abs(r["auc_fold_after_platt"] - r["auc_fold_before"]) for r in out)
    pooled = {r["model"]: abs(r["auc_pooled_after_platt"] - r["auc_pooled_before"]) for r in out}
    print(f"{len(out)} fold fits over {len(pooled)} models; slopes {s.min():.4f} to {s.max():.4f}, median {np.median(s):.4f}")
    print(f"non-positive slopes: {len(neg)} in {len({r['model'] for r in neg})} models")
    for r in neg:
        print("  ", r["model"], r["fold"], f"{r['slope']:+.4f}", r["auc_fold_before"], "->", r["auc_fold_after_platt"])
    print(f"max within-fold AUROC change under Platt: {within:.4f}; max pooled change: {max(pooled.values()):.4f}")


if __name__ == "__main__":
    main()
