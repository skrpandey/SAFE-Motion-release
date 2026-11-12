"""Protocol v10 checks (CPU): the quality-blind transform, real-split World I masks, a tiny
A_blind fit through train_asdpose.py, the mask rescoring against a v9 column when a checkpoint
is on disk, and analyze_v10.py on synthetic score files.
Run: conda run -n torch python -m pytest -q tests/test_v10.py
"""

import csv
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{ROOT}/experiments")
from safe_motion import planted as P                 # noqa: E402
from safe_motion.data import asdpose as D            # noqa: E402

ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="")
HAVE_CACHE = os.path.exists(f"{D.CACHE_DIR}/index.csv")


def test_blind_transform_interpolates_and_drops_confidence():
    t = 20
    arr = np.zeros((t, 17, 3), np.float32)
    arr[..., 0] = np.arange(t)[:, None]                  # x grows linearly in time
    arr[..., 1] = 2 * np.arange(t)[:, None]
    arr[..., 2] = 1.0
    arr[5:9, 3, :] = 0.0                                  # a missing run (zeros)
    arr[12, 4, 2] = 0.1                                   # low confidence, coordinates present
    arr[0:3, 6, :] = 0.0                                  # leading gap
    arr[:, 7, 2] = 0.0                                    # a joint never observed
    out = D.blind_transform(arr)
    assert np.allclose(out[5:9, 3, 0], np.arange(5, 9)) and np.allclose(out[5:9, 3, 1], 2 * np.arange(5, 9))
    assert out[12, 4, 2] == 1.0 and np.isclose(out[12, 4, 0], 12.0)       # low-conf frame interpolated
    assert np.allclose(out[0:3, 6, :2], out[3, 6, :2])                      # leading gap copies first observed
    assert (out[:, 7, :2] == 0).all() and (out[:, 7, 2] == 0).all()         # never observed stays zero
    assert set(np.unique(out[..., 2])) <= {0.0, 1.0}                        # confidence carries no cue
    if HAVE_CACHE:
        rows = [r for r in D.load_index() if r["n_frames"] >= 200][:3]
        ds = D.SegmentDataset(rows, train=False, confidence=False, blind=True)
        x, _ = ds[0]
        assert x.shape == (2, 200, 17)                                      # no confidence channel
        ds_v = D.SegmentDataset(rows, train=False, confidence=False, blind=True, velocity=True)
        assert ds_v[0][0].shape == (4, 200, 17)


@pytest.mark.skipif(not HAVE_CACHE, reason="ASDPose cache not on this machine")
def test_real_split_world_I_masks_are_label_independent_and_deterministic():
    rows = [r for r in D.load_index() if r["split"] == "train" and r["n_frames"] >= 200][:4000]
    m1, eps1 = P.build_masks(rows, "I", "mid", seed=0, mechanism="runs")
    m2, eps2 = P.build_masks(rows, "I", "mid", seed=0, mechanism="runs")
    assert np.array_equal(eps1, eps2) and all(np.array_equal(m1[k], m2[k]) for k in list(m1)[:50])
    lab = np.array([r["binary_label"] for r in rows])
    assert abs(np.corrcoef(eps1, lab)[0, 1]) < 0.06                        # independent of the label
    _, eps_s = P.build_masks(rows, "S", "mid", seed=0, mechanism="runs")
    assert np.corrcoef(eps_s, lab)[0, 1] < -0.15                           # World S degrades negatives more
    frac = np.mean([m1[r["identifier"]].mean() for r in rows[:300]])
    assert 0.05 < frac < 0.6


@pytest.mark.skipif(not HAVE_CACHE, reason="ASDPose cache not on this machine")
def test_tiny_a_blind_fit(tmp_path):
    cmd = [sys.executable, "experiments/train_asdpose.py", "--proto", "v10", "--cell", "A_blind", "--world", "S",
           "--mechanism", "runs", "--strength", "mid", "--arch", "compact", "--smoke", "--workers", "2"]
    r = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    od = f"{ROOT}/results/v10_smoke/S_w0_runs_mid/A_blind/seed0"
    assert os.path.exists(f"{od}/train_log.json") and os.path.exists(f"{od}/test_scores_S.csv")
    import torch
    ck = torch.load(f"{ROOT}/cache/pilot_v10_S_w0_runs_mid_A_blind_s0_smoke.pt", map_location="cpu", weights_only=False)
    assert ck["cell"]["blind"] is True
    assert ck["state_dict"]["input_bn.weight"].shape[0] == 2 * 17          # two input channels


@pytest.mark.skipif(not os.path.isdir(f"{ROOT}/results/fits/v9/cache") or
                    not os.path.exists(f"{ROOT}/results/v9/utd_mhad_body_vs_arm/folds.json"),
                    reason="v9 checkpoints not on this machine")
def test_mask_rescoring_reproduces_v9_column(tmp_path):
    cmd = [sys.executable, "experiments/score_v10_masks.py", "--ckpt-dir", "results/fits/v9/cache",
           "--hosts", "utd_mhad_body_vs_arm", "--limit", "1", "--mask-seeds", "2000", "--device", "cpu",
           "--out", str(tmp_path / "masks"), "--tol", "5e-3"]
    r = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    rep = json.load(open(tmp_path / "masks" / "reproduction.json"))
    assert len(rep) == 1 and max(rep.values()) < 5e-3


def _write_scores(path, ids, child, y, cols):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["identifier", "child_id", "label", "n_frames", "qc_pass"] + list(cols))
        for i in range(len(ids)):
            w.writerow([ids[i], child[i], y[i], 250, 1] + [f"{cols[c][i]:.6f}" for c in cols])


def test_analyze_v10_on_synthetic(tmp_path):
    import analyze_v10 as AN
    rng = np.random.RandomState(0)
    n, ids, child, y = 300, [f"s{i}" for i in range(300)], rng.randint(0, 20, 300), rng.randint(0, 2, 300)
    v7, v10 = tmp_path / "v7", tmp_path / "v10"
    v7.mkdir(), v10.mkdir()

    def scores(strength_s, strength_i, extra=()):
        base = rng.rand(n) * 0.6 + 0.2 * y             # clean AUC about 0.7, not saturated
        cols = {"clean": base}
        for mech in ("runs", "whole", "frozen", "armruns"):
            for ms in (2000, 3000, 4000):
                cols[f"{mech}_S_mid_m{ms}"] = np.clip(base - strength_s * (1 - y) * 0.15 + rng.rand(n) * 0.05, 0, 1)
                cols[f"{mech}_I_mid_m{ms}"] = np.clip(base + strength_i * rng.rand(n) * 0.1, 0, 1)
        for w in ("S", "I"):
            cols[f"gapfill_{w}_mid_m2000"] = np.clip(base + rng.rand(n) * 0.02, 0, 1)
        return cols

    for k in range(3):                                   # part A: strong cue in A_none, none in A_blind
        for mech in ("runs", "whole"):
            _write_scores(v7 / f"v7_S_w0_{mech}_mid_A_none_s{k}.csv", ids, child, y, scores(1.0, 0.0))
            _write_scores(v7 / f"v7_I_w0_{mech}_mid_A_none_s{k}.csv", ids, child, y, scores(0.0, 0.0))
            _write_scores(v10 / f"v10_S_w0_{mech}_mid_A_blind_s{k}.csv", ids, child, y, scores(0.0, 0.0))
            _write_scores(v10 / f"v10_I_w0_{mech}_mid_A_blind_s{k}.csv", ids, child, y, scores(0.0, 0.0))
        for cell in ("A_none", "A_dropout"):             # part B
            _write_scores(v7 / f"v7_real_{cell}_s{k}.csv", ids, child, y, scores(1.0, 0.0))
            _write_scores(v10 / f"v10_real_I_{cell}_s{k}.csv", ids, child, y, scores(0.0, 0.0))
    masks = tmp_path / "masks"                            # part C: two folds, two seeds, three masks
    for world, strength in (("S", 1.0), ("I", 0.0)):
        for k in range(2):
            for f in range(2):
                d = masks / "hostx" / "runs_mid" / world / "A_none" / f"seed{k}" / f"fold{f}"
                d.mkdir(parents=True)
                sel = slice(f * 150, (f + 1) * 150)
                cols = {"clean": (rng.rand(150) * 0.6 + 0.2 * y[sel])}
                for ms in (2000, 3000, 4000):
                    cols[f"runs_S_mid_m{ms}"] = np.clip(cols["clean"] - strength * 0.15 * (1 - y[sel]) + rng.rand(150) * 0.05, 0, 1)
                    cols[f"runs_I_mid_m{ms}"] = np.clip(cols["clean"] + rng.rand(150) * 0.05, 0, 1)
                with open(d / "test_scores.csv", "w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["identifier", "subject_id", "label", "n_frames", "qc_pass", "q_lowconf", "q_missing",
                                "q_frozen", "q_arm_lowconf"] + list(cols))
                    for i in range(150):
                        w.writerow([ids[sel][i], child[sel][i], y[sel][i], 100, 1, 0, 0, 0, 0] + [f"{cols[c][i]:.6f}" for c in cols])
    out = tmp_path / "an"
    sys.argv = ["analyze_v10", "--v7-scores", str(v7), "--v10-scores", str(v10), "--masks", str(masks),
                "--out", str(out), "--n-boot", "40"]
    AN.main()
    crit = json.load(open(out / "criteria.json"))
    assert crit["A1_cue_attributable"]["runs"]["pass"] is True             # blind SE near 0, none SE large
    assert crit["A2_transform_check"]["runs"]["pass"] is True
    assert "A_none" in crit["B_real_world_I"] and crit["B_real_world_I"]["A_none"]["runs"]["excludes_zero"]
    rows = list(csv.DictReader(open(out / "masks.csv")))
    assert len(rows) == 1 and int(rows[0]["n_masks"]) == 3 and float(rows[0]["SE_mean"]) > 0.3
    assert len(list(csv.DictReader(open(out / "masks_by_seed.csv")))) == 3
    for f in ("blind.csv", "real_I.csv"):
        assert os.path.getsize(out / f) > 0
