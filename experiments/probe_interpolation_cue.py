"""Post hoc: does the quality-blind transform (interpolation, confidence dropped) leave planted
missingness readable in the coordinates? On host test segments under missing runs at mid (worlds
S and I, mask seed 2000, plus clean), computes per-segment signature features of the x, y
coordinates (zero-acceleration and zero-velocity fractions, slope transitions, never-observed
fraction) and from them predicts the planted severity (out-of-fold R^2) and the label (out-of-fold
AUROC, child-grouped 5-fold, child bootstrap). Writes results/v10/analysis/interp_probe.csv.
"""
import csv
import os

import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from safe_motion import metrics as M
from safe_motion import planted as P
from safe_motion.data import asdpose as D

OUT = "results/v10/analysis/interp_probe.csv"
V1 = yaml.safe_load(open("configs/protocols/pilot_asdpose_v1.yaml"))


def signature(xy):
    """xy: (T, 17, 2) after the blind transform and centre crop."""
    v = np.diff(xy, axis=0)
    a = np.diff(v, axis=0)
    zero_a = np.all(np.abs(a) < 1e-9, axis=2)           # (T-2, 17)
    zero_v = np.all(np.abs(v) < 1e-9, axis=2)           # (T-1, 17)
    never = np.all(xy == 0, axis=2)                     # (T, 17)
    trans = np.abs(np.diff(zero_a.astype(int), axis=0)).sum()
    return [zero_a.mean(), zero_v.mean(), trans / xy.shape[1], never.mean(),
            float(np.abs(a).mean()), float(np.abs(v).mean())]


FEATS = ["frac_zero_accel", "frac_zero_vel", "slope_transitions_per_joint", "frac_never_observed",
         "mean_abs_accel", "mean_abs_vel"]


def oof_auc(X, y, child, clf):
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, child):
        pred[te] = clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    boots = list(M.cluster_bootstrap_indices(np.asarray(child), 2000, seed=0))
    vals = np.array([M.auc(pred[i], y[i]) for i in boots])
    lo, hi = M.ci(vals)
    return M.auc(pred, y), lo, hi


def oof_r2(X, t, child, reg):
    pred = np.zeros(len(t))
    for tr, te in GroupKFold(5).split(X, t, child):
        pred[te] = reg().fit(X[tr], t[tr]).predict(X[te])
    return 1 - np.sum((t - pred) ** 2) / np.sum((t - t.mean()) ** 2)


def main():
    rows = [r for r in D.load_index() if r["n_frames"] >= 200]
    te_host = [r for r in rows if r["split"] == "test" and D.passes_qc(r, V1["qc_filter"])]
    y = np.array([r["binary_label"] for r in te_host]); child = [r["child_id"] for r in te_host]
    out = []
    scaled_lr = lambda: __import__("sklearn.pipeline", fromlist=["make_pipeline"]).make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    gb = lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0)
    for world in ("clean", "S", "I"):
        if world == "clean":
            masks, eps = None, None
        else:
            masks, eps = P.build_masks(te_host, world, "mid", 2000, mechanism="runs")
        ds = D.SegmentDataset(te_host, window=200, train=False, normalize="image", confidence=False,
                              planted=masks, planted_mode="zero", blind=True)
        X = np.array([signature(ds._crop(ds.load_segment(r))[..., :2]) for r in te_host])
        rec = {"analysis": "post hoc", "world": world, "n_segments": len(te_host), "n_children": len(set(child))}
        for k, name in enumerate(FEATS):
            rec[f"mean_{name}"] = round(float(X[:, k].mean()), 4)
        if eps is not None:
            rec["severity_r2_ridge_oof"] = round(oof_r2(X, eps, child, lambda: Ridge(alpha=1.0)), 3)
            rec["severity_r2_gb_oof"] = round(oof_r2(X, eps, child, lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0)), 3)
            rec["severity_corr_zero_accel"] = round(float(np.corrcoef(X[:, 0], eps)[0, 1]), 3)
        for cname, clf in (("lr", scaled_lr), ("gb", gb)):
            p, lo, hi = oof_auc(X, y, child, clf)
            rec[f"label_auc_{cname}"] = round(p, 3); rec[f"label_auc_{cname}_lo"] = round(lo, 3); rec[f"label_auc_{cname}_hi"] = round(hi, 3)
        out.append(rec); print(rec)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    keys = sorted({k for r in out for k in r}, key=lambda k: (k not in out[0], k))
    with open(OUT, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(list(out[-1]) + keys))); w.writeheader(); w.writerows(out)


if __name__ == "__main__":
    main()
