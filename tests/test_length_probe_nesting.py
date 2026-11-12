"""The within-class length probe is nested: nothing fitted on an outer training fold depends on
held-out lengths."""
import os
import sys

import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import length_audit_inputs as L  # noqa: E402


def _data(seed=0, n=240, d=6):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    label = rng.randint(0, 2, n)
    groups = np.array([f"g{i % 12}" for i in range(n)])
    y = np.log(200 + 300 * rng.rand(n)) + 0.3 * label + 0.2 * X[:, 0]
    return X, y, label, groups


def test_training_side_independent_of_heldout_lengths():
    X, y, label, groups = _data()
    k = 5
    folds = list(GroupKFold(n_splits=k).split(X, y, groups))
    tr, te = folds[0]
    y2 = y.copy()
    y2[te] += 5.0 + np.random.RandomState(1).rand(len(te))      # perturb held-out lengths only
    for model in ("ridge", "gb"):
        a = L.fold_fit(X, y, label, groups, tr, te, model, within_class=True)
        b = L.fold_fit(X, y2, label, groups, tr, te, model, within_class=True)
        assert np.allclose(a["target_tr"], b["target_tr"])
        assert np.allclose(a["scaler_mean"], b["scaler_mean"])
        assert a["alpha"] == b["alpha"]
        if a["coef"] is not None:
            assert np.allclose(a["coef"], b["coef"])
        assert np.allclose(a["pred_te"], b["pred_te"])           # predictions do not read held-out targets
        assert not np.allclose(a["target_te"], b["target_te"])   # only the held-out targets moved


def test_residual_uses_outer_training_class_means_only():
    X, y, label, groups = _data(seed=3)
    tr, te = next(GroupKFold(n_splits=4).split(X, y, groups))
    mu = L.class_means(y, label, tr)
    for c in (0, 1):
        assert np.isclose(mu[c], y[tr][label[tr] == c].mean())
    f = L.fold_fit(X, y, label, groups, tr, te, "ridge", within_class=True)
    assert np.allclose(f["target_te"], y[te] - np.array([mu[c] for c in label[te]]))


def test_oof_r2_runs_and_is_bounded():
    X, y, label, groups = _data(seed=5)
    r2 = L.oof_r2(X, y, label, groups, 4, "ridge")
    r2w = L.oof_r2(X, y, label, groups, 4, "ridge", within_class=True)
    assert r2 <= 1.0 and r2w <= 1.0
