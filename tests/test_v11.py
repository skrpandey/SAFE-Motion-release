"""Protocol v11 tests (seconds, CPU): the cache-building rule, the MediaPipe joint map, the
calibration helpers, the fold assignment and the external-cache loader."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import build_mediapipe_external as B
import analyze_v11 as A
import within_set_v11 as W
from safe_motion import metrics as M


def test_resample_and_window_rule():
    rng = np.random.RandomState(0)
    raw = rng.rand(161, 17, 3).astype(np.float32)
    out = B.window(B.resample_30(raw, 24.1208))
    assert out.shape == (200, 17, 3)
    assert np.allclose(out[0], raw[0]) and np.allclose(out[-1], raw[-1])      # endpoints kept
    same = B.window(B.resample_30(raw[:200] if False else rng.rand(200, 17, 3), 30.0))
    assert same.shape == (200, 17, 3)
    short = B.window(B.resample_30(rng.rand(50, 17, 3), 30.0))
    assert short.shape == (200, 17, 3) and np.allclose(short[:50], short[50:100])  # tiled
    long = rng.rand(300, 17, 3)
    assert np.allclose(B.window(long), long[50:250])                             # centre crop


def test_mediapipe_map_and_quality():
    lm = np.zeros((5, 33, 4), np.float32)
    lm[..., 0] = 0.5
    lm[..., 1] = 0.25
    lm[..., 3] = 0.9
    lm[2] = 0.0                                             # an undetected frame
    out = B.mp33_to_coco17(lm, 640, 480)
    assert out.shape == (5, 17, 3)
    assert out[0, 0, 0] == 320 and out[0, 0, 1] == 120 and out[0, 0, 2] == np.float32(0.9)
    assert (out[2] == 0).all()
    q = B.quality_features(out[..., :2], out[..., 2])
    assert abs(q[1] - 0.2) < 1e-6 and abs(q[0] - 0.2) < 1e-6          # one of five frames missing
    assert B.COCO17_FROM_MP33[5:7] == [11, 12] and B.COCO17_FROM_MP33[15:] == [27, 28]


def test_calibration_helpers():
    rng = np.random.RandomState(1)
    y = (rng.rand(400) < 0.3).astype(int)
    z = 2.0 * (y - 0.5) * 3 + rng.randn(400)             # overconfident logits
    p = 1 / (1 + np.exp(-z))
    T = A.fit_temperature(z, y)
    q = 1 / (1 + np.exp(-z / T))
    assert T > 0.0
    nll = lambda pp: -np.mean(y * np.log(np.clip(pp, 1e-7, 1)) + (1 - y) * np.log(np.clip(1 - pp, 1e-7, 1)))
    assert nll(q) <= nll(p) + 1e-9                          # the fitted temperature lowers NLL
    assert abs(M.auc(q, y) - M.auc(p, y)) < 1e-12           # monotone scaling keeps AUC
    t = A.best_f1_threshold(p, y)
    assert 0 <= t <= 1
    m = A.op_metrics(p, y, 0.85)
    assert set(m) == {"precision", "recall", "firing", "false_alarm"}


def test_folds_video_grouped(tmp_path):
    rng = np.random.RandomState(0)
    rows = [{"identifier": f"w{i}", "child_id": 10000 + i // 4, "binary_label": int(rng.rand() < 0.25)}
            for i in range(120)]
    folds = W.make_folds(rows, str(tmp_path / "folds.json"), n_folds=5)
    allv = sorted(v for f in folds for v in f)
    assert allv == sorted({r["child_id"] for r in rows})      # every video once
    assert len(set(allv)) == len(allv)
    assert json.load(open(tmp_path / "folds.json"))["n_folds"] == 5


def test_external_cache_loads():
    from safe_motion.data import asdpose as D
    rows = D.load_index(f"{B.EXTERNAL_CACHE}/index.csv")
    assert len(rows) == 550 and all(r["n_frames"] == 200 for r in rows)
    ds = D.SegmentDataset(rows[:3], train=False, cache_dir=B.EXTERNAL_CACHE)
    x, y = ds[0]
    assert tuple(x.shape) == (3, 200, 17)
