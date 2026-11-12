"""Calibration on the external set beyond temperature-only scaling (post hoc).

For each of the 49 v11 score files, with the video-grouped folds: a fold-fitted constant
(prevalence) predictor, Platt scaling fitted on training folds, and temperature-only scaling, each
scored out of fold (Brier, ECE with 15 bins, firing and recall at 0.85, within-fold and pooled
AUROC). Also reports the five fold temperatures and ensemble vs per-seed firing for ours: none.
Writes results/v11/analysis/calibration_v5.csv and calibration_v5.md.
"""
import csv
import glob
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "experiments")
from analyze_v11 import EXTERNAL_CACHE, OP_POINT, ece, fit_temperature, load_scores, logit  # noqa: E402
from safe_motion import metrics as M  # noqa: E402
from safe_motion.data import asdpose as D  # noqa: E402

A = "results/v11/analysis"


def main():
    rows = D.load_index(f"{EXTERNAL_CACHE}/index.csv")
    folds = json.load(open("results/v11/folds.json"))["folds"]
    y = np.array([r["binary_label"] for r in rows]); child = np.array([r["child_id"] for r in rows])
    fold_te = [np.array([c in set(fv) for c in child]) for fv in folds]
    out = []
    scores = {}
    for path in sorted(glob.glob("results/v11/scores/*.csv")):
        name = os.path.basename(path)[:-4]
        _, cols = load_scores(path)
        p = np.clip(cols["clean"], 1e-6, 1 - 1e-6); z = logit(p); scores[name] = p
        const = np.zeros(len(y)); platt = np.zeros(len(y)); temp = np.zeros(len(y))
        temps, auc_b, auc_a = [], [], []
        for te in fold_te:
            tr = ~te
            const[te] = y[tr].mean()
            lr = LogisticRegression(C=1e6, max_iter=2000).fit(z[tr, None], y[tr])
            platt[te] = lr.predict_proba(z[te, None])[:, 1]
            T = fit_temperature(z[tr], y[tr]); temps.append(T)
            temp[te] = 1 / (1 + np.exp(-z[te] / T))
            auc_b.append(M.auc(p[te], y[te])); auc_a.append(M.auc(temp[te], y[te]))
        def op(pp):
            pred = pp >= OP_POINT
            tp = (pred & (y == 1)).sum()
            return float(pred.mean()), float(tp / (y == 1).sum())
        rec = {"model": name, "auc_pooled_before": round(M.auc(p, y), 4), "auc_pooled_after_temp": round(M.auc(temp, y), 4),
               "auc_within_fold_mean_before": round(float(np.mean(auc_b)), 4), "auc_within_fold_mean_after_temp": round(float(np.mean(auc_a)), 4),
               "temps_per_fold": " ".join(f"{t:.2f}" for t in temps), "temp_mean": round(float(np.mean(temps)), 3),
               "temps_at_bound": int(sum(t > 54.5 for t in temps)),
               "brier_before": round(float(np.mean((p - y) ** 2)), 4), "brier_temp": round(float(np.mean((temp - y) ** 2)), 4),
               "brier_platt": round(float(np.mean((platt - y) ** 2)), 4), "brier_constant_prevalence": round(float(np.mean((const - y) ** 2)), 4),
               "ece_before": round(ece(p, y), 4), "ece_temp": round(ece(temp, y), 4), "ece_platt": round(ece(platt, y), 4),
               "ece_constant_prevalence": round(ece(const, y), 4)}
        for tag, pp in (("before", p), ("temp", temp), ("platt", platt)):
            f, r = op(pp); rec[f"firing085_{tag}"] = round(f, 4); rec[f"recall085_{tag}"] = round(r, 4)
        out.append(rec)
    os.makedirs(A, exist_ok=True)
    with open(f"{A}/calibration_v5.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    # firing rate of the seed ensemble vs per seed, ours: none
    members = [n for n in scores if n.startswith("ours_v7_A_none_s")]
    ens = np.mean([scores[n] for n in members], axis=0)
    ens_fire = float((ens >= OP_POINT).mean()); seed_fire = [float((scores[n] >= OP_POINT).mean()) for n in members]
    pooled_change = max(abs(r["auc_pooled_after_temp"] - r["auc_pooled_before"]) for r in out)
    within_change = max(abs(r["auc_within_fold_mean_after_temp"] - r["auc_within_fold_mean_before"]) for r in out)
    arr = lambda k: np.array([r[k] for r in out])
    prev = y.mean()
    md = f"""# Calibration on the external set, post hoc extensions

Prevalence 127/550 = {prev:.4f}; a constant predictor at the pooled prevalence has Brier {prev*(1-prev):.4f}.
Fold-fitted constant predictor (training-fold prevalence, held-out folds): Brier {arr('brier_constant_prevalence').min():.4f} to {arr('brier_constant_prevalence').max():.4f} (identical across models up to the fold prevalences), ECE {arr('ece_constant_prevalence').min():.4f} to {arr('ece_constant_prevalence').max():.4f}.

| quantity | temperature only | Platt (intercept + slope) |
|---|---|---|
| Brier, range over 49 models | {arr('brier_temp').min():.3f} to {arr('brier_temp').max():.3f} | {arr('brier_platt').min():.3f} to {arr('brier_platt').max():.3f} |
| ECE, range | {arr('ece_temp').min():.3f} to {arr('ece_temp').max():.3f} | {arr('ece_platt').min():.3f} to {arr('ece_platt').max():.3f} |
| models firing at 0.85 (rate > 0) | {int((arr('firing085_temp') > 0).sum())} of 49, max {arr('firing085_temp').max():.3f} | {int((arr('firing085_platt') > 0).sum())} of 49, max {arr('firing085_platt').max():.3f} |

Before any scaling: Brier {arr('brier_before').min():.3f} to {arr('brier_before').max():.3f}, ECE {arr('ece_before').min():.3f} to {arr('ece_before').max():.3f}.
Temperatures: {int(arr('temps_at_bound').sum())} of {5*len(out)} fold fits at the 54.6 bound; the displayed temperature is the mean of the five fold temperatures.
AUROC under temperature scaling: within-fold AUROC is unchanged (max change {within_change:.1e}); pooled out-of-fold AUROC changes by at most {pooled_change:.4f} across models because fold-specific temperatures can reorder scores across folds.
Firing-rate aggregation, ours: none: ensembled predictions (mean of three seed scores, then threshold) fire on {ens_fire:.3f} of windows (the transfer table); per-seed rates are {', '.join(f'{v:.3f}' for v in seed_fire)} (mean {np.mean(seed_fire):.3f}; the calibration tables list seeds).
"""
    open(f"{A}/calibration_v5.md", "w").write(md); print(md)


if __name__ == "__main__":
    main()
