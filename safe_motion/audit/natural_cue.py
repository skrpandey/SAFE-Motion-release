"""Natural-cue audit: do per-clip tracking statistics predict the label under subject-grouped CV?

tracking_stats() reuses the definitions of asdpose-smm-detection (commit e5217a5)
stage1/build_cache.py:quality_features, confidence cutoff 0.3, with the arm joints passed in
(the layout differs per dataset). Motion energy is not a tracking statistic and is left out,
as in safe_motion.data.asdpose.TRACKING_STATS. Clip length is a separate predictor.

Evaluation: out-of-fold class probabilities pooled over grouped folds (no subject in both the
fitting and the scored part of a fold), then macro one-vs-rest AUC (mean of per-class AUCs,
safe_motion.metrics.auc, average ranks) and balanced accuracy of the argmax. Null: the same
pipeline with labels permuted across clips. CI: subject-clustered bootstrap of the pooled
out-of-fold predictions (the models are not refitted inside the bootstrap).
"""

import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from safe_motion import metrics as M

LOW_CONF = 0.3
STAT_NAMES = ["q_lowconf", "q_missing", "q_frozen", "q_arm_lowconf"]


def tracking_stats(kp, arm_joints, low_conf=LOW_CONF):
    """kp (T, J, 3): x, y, confidence. Returns the four tracking statistics as a dict."""
    xy, sc = kp[..., :2], kp[..., 2]
    frozen = float((xy[1:] == xy[:-1]).all(axis=(1, 2)).mean()) if len(kp) > 1 else 0.0
    return {"q_lowconf": float((sc < low_conf).mean()),
            "q_missing": float((sc == 0).mean()),
            "q_frozen": frozen,
            "q_arm_lowconf": float((sc[:, arm_joints] < low_conf).mean())}


def make_model(kind, seed=0):
    if kind == "lr":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced"))
    if kind == "gb":
        return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3,
                                              min_samples_leaf=10, class_weight="balanced",
                                              random_state=seed)
    raise ValueError(kind)


def make_folds(y, groups, n_splits=None, seed=0):
    """Leave-one-subject-out when n_splits is None, else stratified group k-fold."""
    if n_splits is None:
        return list(LeaveOneGroupOut().split(np.zeros(len(y)), y, groups))
    return list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                     random_state=seed).split(np.zeros(len(y)), y, groups))


def oof_proba(X, y, folds, kind, seed=0):
    classes = np.unique(y)
    P = np.zeros((len(y), len(classes)))
    for tr, te in folds:
        m = make_model(kind, seed).fit(X[tr], y[tr])
        p = np.zeros((len(te), len(classes)))
        p[:, np.searchsorted(classes, m.classes_)] = m.predict_proba(X[te])
        P[te] = p
    return P, classes


def score(P, y, classes):
    per = {int(c): M.auc(P[:, k], (y == c).astype(int)) for k, c in enumerate(classes)}
    if len(classes) == 2:
        per = {int(classes[1]): per[int(classes[1])]}
    macro = float(np.nanmean(list(per.values())))
    bacc = float(balanced_accuracy_score(y, classes[P.argmax(1)]))
    return macro, bacc, per


def evaluate(X, y, groups, kind, folds, n_perm=0, n_boot=2000, seed=0):
    """Out-of-fold macro OvR AUC and balanced accuracy, permutation null and clustered CI."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    t0 = time.time()
    if np.all(X.std(axis=0) == 0):
        classes = np.unique(y)
        chance = {int(c): 0.5 for c in (classes[1:] if len(classes) == 2 else classes)}
        return {"degenerate": True, "macro_auc": 0.5, "bacc": 1.0 / len(classes),
                "per_class_auc": chance, "note": "all features constant; chance by construction"}
    P, classes = oof_proba(X, y, folds, kind, seed)
    macro, bacc, per = score(P, y, classes)
    boot = []
    for idx in M.cluster_bootstrap_indices(groups, n_boot=n_boot, seed=seed):
        if len(np.unique(y[idx])) == len(classes):
            boot.append(score(P[idx], y[idx], classes)[0])
    out = {"degenerate": False, "macro_auc": macro, "macro_auc_ci": M.ci(boot), "bacc": bacc,
           "per_class_auc": per, "n_boot": len(boot)}
    if n_perm:
        rng = np.random.RandomState(seed + 1)
        null_auc, null_bacc = [], []
        for _ in range(n_perm):
            yp = rng.permutation(y)
            Pp, _ = oof_proba(X, yp, folds, kind, seed)
            a, b, _ = score(Pp, yp, classes)
            null_auc.append(a)
            null_bacc.append(b)
        null_auc, null_bacc = np.array(null_auc), np.array(null_bacc)
        out.update(n_perm=n_perm, null_auc_mean=float(null_auc.mean()),
                   null_auc_95=float(np.percentile(null_auc, 95)),
                   p_auc=float((1 + (null_auc >= macro).sum()) / (1 + n_perm)),
                   null_bacc_mean=float(null_bacc.mean()),
                   p_bacc=float((1 + (null_bacc >= bacc).sum()) / (1 + n_perm)))
    out["seconds"] = round(time.time() - t0, 1)
    return out
