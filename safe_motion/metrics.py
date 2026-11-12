"""Reliance metrics: AUC, weighted AUC, d', propensity reweighting, clustered bootstrap.

Definitions follow configs/protocols/pilot_asdpose_v1.yaml. auc() uses average ranks, so tied
scores (e.g. length or quality adversaries) are handled correctly.
"""

import numpy as np
from scipy.stats import norm, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def auc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def weighted_auc(scores, labels, weights):
    """P(s_pos > s_neg) over all pairs, each pair weighted by w_pos * w_neg; ties count 1/2."""
    s, y, w = np.asarray(scores, float), np.asarray(labels), np.asarray(weights, float)
    sp, wp = s[y == 1], w[y == 1]
    sn, wn = s[y == 0], w[y == 0]
    if len(sp) == 0 or len(sn) == 0:
        return float("nan")
    order = np.argsort(sn)
    sn, wn = sn[order], wn[order]
    cw = np.concatenate([[0.0], np.cumsum(wn)])
    lo = np.searchsorted(sn, sp, side="left")
    hi = np.searchsorted(sn, sp, side="right")
    below = cw[lo] + 0.5 * (cw[hi] - cw[lo])
    return float((wp * below).sum() / (wp.sum() * wn.sum()))


def dprime(a):
    a = np.clip(np.asarray(a, float), 1e-6, 1 - 1e-6)
    return np.sqrt(2.0) * norm.ppf(a)


def propensity_weights(q, labels, clip_pct=99.0):
    """Density-ratio weights making tracking stats independent of the label.

    Negatives get w = [p/(1-p)] * [n_neg/n_pos] with p = P(y=1|q); positives w = 1.
    Returns (weights, propensity p).
    """
    q, y = np.asarray(q, float), np.asarray(labels)
    x = StandardScaler().fit_transform(q)
    p = LogisticRegression(C=1.0, max_iter=2000).fit(x, y).predict_proba(x)[:, 1]
    w = np.ones(len(y))
    neg = y == 0
    w[neg] = p[neg] / (1 - p[neg]) * (neg.sum() / (~neg).sum())
    cap = np.percentile(w[neg], clip_pct)
    w[neg] = np.minimum(w[neg], cap)
    return w, p


def kish_ess(w):
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w ** 2).sum())


def binormal_slope(scores, labels, n_points=50):
    """Slope of z(TPR) on z(FPR) over interior ROC points; 1 means equal-variance binormal."""
    s, y = np.asarray(scores, float), np.asarray(labels)
    ths = np.quantile(s, np.linspace(0.02, 0.98, n_points))
    tpr = np.array([(s[y == 1] > t).mean() for t in ths])
    fpr = np.array([(s[y == 0] > t).mean() for t in ths])
    ok = (tpr > 0.005) & (tpr < 0.995) & (fpr > 0.005) & (fpr < 0.995)
    if ok.sum() < 5:
        return float("nan")
    return float(np.polyfit(norm.ppf(fpr[ok]), norm.ppf(tpr[ok]), 1)[0])


def cluster_bootstrap_indices(clusters, n_boot=10000, seed=0):
    """Yields index arrays, each a resample of whole clusters with replacement."""
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.RandomState(seed)
    for _ in range(n_boot):
        pick = rng.randint(0, len(uniq), len(uniq))
        yield np.concatenate([members[k] for k in pick])


def ci(samples, level=0.95):
    a = np.asarray(samples, float)
    a = a[np.isfinite(a)]
    lo, hi = np.percentile(a, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)
