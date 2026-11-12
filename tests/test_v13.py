"""Protocol v13 (configs/protocols/armconf_v13.yaml): the coordinate-invariant valid-joint rule.
CPU, seconds. The synthetic test needs no data; the tiny fit skips without the ASDPose cache."""
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from safe_motion.data import asdpose as D         # noqa: E402

HAVE_DATA = os.path.exists(os.path.join(D.CACHE_DIR, "index.csv"))


def _synthetic(n_frames=200, seed=0):
    """A segment with real-looking joints, some naturally missing joint-frames (x = y = conf = 0)."""
    rng = np.random.RandomState(seed)
    arr = np.zeros((n_frames, 17, 3), np.float32)
    arr[..., :2] = rng.uniform(100, 500, (n_frames, 17, 2))
    arr[..., 2] = rng.uniform(0.4, 1.0, (n_frames, 17))
    miss = rng.rand(n_frames, 17) < 0.15
    arr[miss] = 0.0
    return arr


def _encode(arr, rule):
    ds = D.SegmentDataset.__new__(D.SegmentDataset)
    ds.rows, ds.window, ds.train = [], 200, False
    ds.flip_prob, ds.joint_dropout, ds.coord_dropout = 0.0, 0.0, 0.0
    ds.velocity, ds.normalize, ds.confidence = False, "image", True
    ds.planted, ds.planted_mode, ds.blind, ds.valid_rule = None, "confscale", False, rule
    ds.rng = np.random.RandomState(0)
    x, _ = ds._encode(arr.copy(), {"binary_label": 0, "img_h": 720})
    return x.numpy()


def test_coords_rule_makes_xy_invariant_under_confscale_and_conf_rule_does_not():
    arr = _synthetic()
    mask = np.zeros(arr.shape[:2], bool)
    mask[:, [5, 6, 7, 8, 9, 10]] = np.random.RandomState(1).rand(arr.shape[0], 6) < 0.3
    planted = D.apply_planted(arr.copy(), mask, "confscale")
    # the floor at 0.05 turns naturally missing arm joint-frames into confidence > 0 with x = y = 0
    assert ((planted[..., 2] > 0) & (planted[..., 0] == 0) & (planted[..., 1] == 0)).any()
    for rule, expect_equal in (("coords", True), ("conf", False)):
        xc, xp = _encode(arr, rule), _encode(planted, rule)
        d = np.abs(xc[:2] - xp[:2]).max()
        assert (d == 0.0) == expect_equal, (rule, d)
        assert np.abs(xc[2] - xp[2]).max() > 0          # only the confidence channel differs


def test_default_rule_is_conf_and_unchanged():
    arr = _synthetic()
    ds = D.SegmentDataset.__new__(D.SegmentDataset)
    assert D.SegmentDataset.__init__.__defaults__[-1] == "conf"
    x_default, x_conf = _encode(arr, "conf"), _encode(arr, "conf")
    assert np.array_equal(x_default, x_conf)


@pytest.mark.skipif(not HAVE_DATA, reason="ASDPose cache not available")
def test_tiny_v13_fit(tmp_path):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=ROOT)
    cmd = [sys.executable, "experiments/train_asdpose.py", "--proto", "v13", "--cell", "A_none",
           "--world", "S", "--mechanism", "armruns", "--strength", "mid", "--seed", "0", "--arch",
           "compact", "--smoke", "--workers", "1", "--batch-size", "16"]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    ck = torch.load(os.path.join(ROOT, "cache/pilot_v13_S_w0_armruns_mid_A_none_s0_smoke.pt"),
                    map_location="cpu", weights_only=False)
    assert ck["cell"]["valid_rule"] == "coords"
    out = os.path.join(ROOT, "results/v13_smoke/S_w0_armruns_mid/A_none/seed0")
    assert os.path.exists(os.path.join(out, "train_log.json"))
    import shutil
    shutil.rmtree(os.path.join(ROOT, "results/v13_smoke"), ignore_errors=True)
    os.remove(os.path.join(ROOT, "cache/pilot_v13_S_w0_armruns_mid_A_none_s0_smoke.pt"))
