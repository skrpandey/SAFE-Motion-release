"""Unit checks for metrics, data and model. Run: conda run -n torch python tests/test_core.py"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import metrics as M
from safe_motion.data import asdpose as D
from safe_motion.models.tcn import TinyTCN
from safe_motion import planted as P


def test_auc():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 500)
    s = rng.randn(500) + y
    a = M.auc(s, y)
    brute = np.mean([(sp > sn) + 0.5 * (sp == sn) for sp in s[y == 1] for sn in s[y == 0]])
    assert abs(a - brute) < 1e-12
    assert abs(M.weighted_auc(s, y, np.ones(500)) - a) < 1e-12
    st = np.round(s)                               # ties
    bt = np.mean([(sp > sn) + 0.5 * (sp == sn) for sp in st[y == 1] for sn in st[y == 0]])
    assert abs(M.auc(st, y) - bt) < 1e-12 and abs(M.weighted_auc(st, y, np.ones(500)) - bt) < 1e-12
    w = rng.rand(500)                              # weighted vs brute force
    num = sum(w[i] * w[j] * ((st[i] > st[j]) + 0.5 * (st[i] == st[j]))
              for i in np.flatnonzero(y == 1) for j in np.flatnonzero(y == 0))
    den = w[y == 1].sum() * w[y == 0].sum()
    assert abs(M.weighted_auc(st, y, w) - num / den) < 1e-10


def test_reweighting_removes_confound():
    rng = np.random.RandomState(1)
    n = 6000
    y = rng.randint(0, 2, n)
    q = rng.randn(n, 2) + 0.8 * y[:, None]         # quality depends on label
    assert M.auc(q[:, 0], y) > 0.65
    w, p = M.propensity_weights(q, y)
    assert 0.45 < M.weighted_auc(p, y, w) < 0.55
    assert abs(M.dprime(0.5)) < 1e-9 and abs(M.dprime(0.76025) - 1.0) < 1e-3


def test_binormal_slope():
    rng = np.random.RandomState(2)
    y = np.r_[np.zeros(20000), np.ones(20000)].astype(int)
    s = np.r_[rng.randn(20000), 1 + 2 * rng.randn(20000)]
    assert abs(M.binormal_slope(s, y) - 0.5) < 0.05


def test_data_and_model():
    rows = D.load_index()
    assert len(rows) == 36928
    qc = {"q_frozen_max": 0.10, "q_arm_lowconf_max": 0.30}
    test = [r for r in rows if r["split"] == "test"]
    assert sum(r["binary_label"] == 0 for r in D.qc_pool(test, qc)) == 1648
    ds = D.SegmentDataset(test[:8], velocity=True, coord_dropout=1.0, train=True, flip_prob=0)
    x, _ = ds[0]
    assert x.shape == (5, 200, 17)
    assert torch.all(x[:4] == 0)                   # all coordinates dropped
    ds1 = D.SegmentDataset(test[:8], velocity=True, coord_dropout=1.0, train=True, flip_prob=0,
                           seed=0)
    ds2 = D.SegmentDataset(test[:8], velocity=True, train=True, flip_prob=0, seed=0)
    assert torch.equal(ds1[0][0][4], ds2[0][0][4])  # same crop, confidence untouched
    m = TinyTCN(c_in=5)
    n_params = sum(p.numel() for p in m.parameters())
    assert 100e3 < n_params < 250e3, n_params
    assert m(torch.randn(2, 5, 200, 17)).shape == (2, 2)
    print(f"TinyTCN params: {n_params / 1e3:.1f}K")


def test_planted_operators():
    rng = np.random.RandomState(0)
    assert not P.missing_mask(300, 0.0, rng).any() and not P.whole_joint_mask(300, 0.0, rng).any()
    fr = [P.missing_mask(20000, e, np.random.RandomState(1)).mean() for e in (0.1, 0.3, 0.6)]
    assert all(abs(f - e) < 0.02 for f, e in zip(fr, (0.1, 0.3, 0.6))), fr   # monotone, calibrated
    m = P.missing_mask(2000, 0.3, np.random.RandomState(2))
    runs = np.diff(np.flatnonzero(np.diff(np.r_[0, m[:, 0].astype(int), 0]) != 0))[::2]
    assert 8 < runs.mean() < 25, runs.mean()                               # run length near 15
    w = P.whole_joint_mask(50, 0.5, np.random.RandomState(3))
    assert (w.all(0) | ~w.any(0)).all()                                    # whole joints only
    rows = [{"identifier": f"id{i}", "binary_label": i % 3 == 0, "n_frames": 60} for i in range(300)]
    for r in rows:
        r["binary_label"] = int(r["binary_label"])
    a, ea = P.build_masks(rows, "S", "mid", 0)
    b, eb = P.build_masks(rows, "S", "mid", 0)
    assert all(np.array_equal(a[k], b[k]) for k in a) and np.array_equal(ea, eb)  # deterministic
    y = np.array([r["binary_label"] for r in rows])
    assert ea[y == 0].mean() > ea[y == 1].mean() + 0.05                       # S: negatives worse
    _, ei = P.build_masks(rows, "I", "mid", 0)
    assert abs(ei[y == 0].mean() - ei[y == 1].mean()) < 0.06                 # I: independent
    c, _ = P.build_masks(rows, "S", "mid", 0, mechanism="whole")
    assert not np.array_equal(c["id1"], a["id1"])


def test_planted_modes():
    rng = np.random.RandomState(0)
    arr = np.zeros((50, 17, 3), np.float32)
    arr[..., 0] = np.linspace(0, 49, 50)[:, None]          # x grows linearly in time
    arr[..., 1] = 2.0
    arr[..., 2] = 0.9
    mask = np.zeros((50, 17), bool)
    mask[10:20, 3] = True
    z = D.apply_planted(arr.copy(), mask, "zero")
    assert (z[10:20, 3] == 0).all() and (z[:10, 3, 2] == 0.9).all()
    g = D.apply_planted(arr.copy(), mask, "gapfill")
    assert np.allclose(g[10:20, 3, 0], np.arange(10, 20)) and (g[10:20, 3, 2] == 0.9).all()
    c = D.apply_planted(arr.copy(), mask, "conflow")
    assert np.allclose(c[10:20, 3, 2], 0.05) and np.allclose(c[10:20, 3, 0], np.arange(10, 20))
    rows = [{"identifier": f"id{i}", "binary_label": int(i % 3 == 0), "n_frames": 60} for i in range(300)]
    y = np.array([r["binary_label"] for r in rows])
    _, es = P.build_masks(rows, "S", "mid", 0)
    _, ep = P.build_masks(rows, "Sp", "mid", 0)
    assert es[y == 0].mean() > es[y == 1].mean() and ep[y == 1].mean() > ep[y == 0].mean()


def test_planted_v8_operators():
    # frozen-frame runs: whole-skeleton rows, calibrated fraction, run length near 20
    f = P.frozen_mask(20000, 0.3, np.random.RandomState(4))
    assert (f.all(1) | ~f.any(1)).all() and abs(f.mean() - 0.3) < 0.02, f.mean()
    runs = np.diff(np.flatnonzero(np.diff(np.r_[0, f[:, 0].astype(int), 0]) != 0))[::2]
    assert 12 < runs.mean() < 30, runs.mean()
    assert not P.frozen_mask(100, 0.0, np.random.RandomState(0)).any()
    # arm runs: only arm joints masked
    a = P.arm_runs_mask(5000, 0.3, np.random.RandomState(5))
    assert not a[:, [j for j in range(17) if j not in P.ARM_JOINTS]].any()
    assert abs(a[:, P.ARM_JOINTS].mean() - 0.3) < 0.03
    # hold: masked frames copy the last unmasked frame; a run at frame 0 copies forward
    arr = np.zeros((50, 17, 3), np.float32)
    arr[..., 0] = np.arange(50)[:, None]
    arr[..., 2] = np.linspace(0.5, 1.0, 50)[:, None]
    mask = np.zeros((50, 17), bool)
    mask[10:20] = True
    mask[0:3] = True
    h = D.apply_planted(arr.copy(), mask, "hold")
    assert (h[10:20, :, 0] == 9).all() and np.allclose(h[10:20, :, 2], arr[9, :, 2])
    assert (h[0:3, :, 0] == 3).all() and (h[20:, :, 0] == arr[20:, :, 0]).all()
    assert (h[3:10] == arr[3:10]).all()
    # confscale: confidence into [0.05, 0.3), coordinates untouched, unmasked untouched
    m2 = np.zeros((50, 17), bool)
    m2[5:15, 7] = True
    arr[..., 2] = 0.9
    c = D.apply_planted(arr.copy(), m2, "confscale")
    assert np.allclose(c[5:15, 7, 2], 0.27) and (c[5:15, 7, 0] == arr[5:15, 7, 0]).all()
    assert (c[~m2][:, 2] == 0.9).all()
    arr[..., 2] = 0.1
    c = D.apply_planted(arr.copy(), m2, "confscale")
    assert np.allclose(c[5:15, 7, 2], 0.05)
    # build_masks dispatch and world ordering for the new mechanisms
    rows = [{"identifier": f"id{i}", "binary_label": int(i % 3 == 0), "n_frames": 60} for i in range(300)]
    y = np.array([r["binary_label"] for r in rows])
    for mech in ("frozen", "armruns"):
        ms, es = P.build_masks(rows, "S", "mid", 0, mechanism=mech)
        ms2, _ = P.build_masks(rows, "S", "mid", 0, mechanism=mech)
        assert all(np.array_equal(ms[k], ms2[k]) for k in ms)
        assert es[y == 0].mean() > es[y == 1].mean() + 0.05
        _, ei = P.build_masks(rows, "I", "mid", 0, mechanism=mech)
        assert abs(ei[y == 0].mean() - ei[y == 1].mean()) < 0.06
    assert P.MECHANISM_MODE == {"runs": "zero", "whole": "zero", "frozen": "hold", "armruns": "confscale"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
