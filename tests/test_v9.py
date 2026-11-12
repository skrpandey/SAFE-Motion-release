"""Protocol v9 checks (seconds, CPU): host dataset windows and normalisation, 20-joint masks and
model, fold reproducibility, a tiny 1-epoch fit per host through train_v9.py, and analyze_v9.py
on synthetic fold files. Run: conda run -n torch python -m pytest -q tests/test_v9.py
"""

import csv
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from safe_motion import planted as P
from safe_motion.data import hosts as H
from safe_motion.models.ctrgcn import CompactCTRGCN


def fake_rows(n_joints, n=12, subjects=4, t=80, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n):
        kp = np.zeros((t, n_joints, 3), np.float32)
        kp[..., :2] = rng.rand(t, n_joints, 2) * 200
        kp[..., 2] = 1.0
        kp[::7, 3, :] = 0.0                                   # a few missing joint-frames
        rows.append({"identifier": f"c{i}", "subject": i % subjects, "binary_label": i % 2,
                     "n_frames": t, "kp": kp, "est_h": 200.0})
    return rows


def test_clip_dataset_windows_and_normalisation():
    rows = fake_rows(17, t=50)                                        # no crop: frame indices line up
    lay = H.LAYOUT["mmasd"]
    ds = H.ClipDataset(rows, 50, lay, train=False)
    x, y = ds[0]
    assert x.shape == (3, 50, 17) and y == 0
    assert float(x[:2].abs().max()) <= 1.0 + 1e-6                  # est_h normalisation keeps |xy| <= 1
    assert (x[2, ::7, 3] == 0).all() and float(x[:2, ::7, 3].abs().sum()) == 0   # invalid joints stay 0
    lay20 = H.LAYOUT["utd_mhad"]
    ds20 = H.ClipDataset(fake_rows(20), 64, lay20, train=False, velocity=True)
    x20, _ = ds20[1]
    assert x20.shape == (5, 64, 20)
    valid = x20[4] > 0
    assert abs(float(x20[:2][:, valid].std()) - 1.0) < 0.3           # window-std normalisation
    tr = H.ClipDataset(fake_rows(17), 50, lay, train=True, seed=1, joint_dropout=0.5)
    xs = [tr[0][0] for _ in range(4)]
    assert any(not torch.equal(xs[0], xk) for xk in xs[1:])         # random crop, flip, dropout


def test_masks_20_joints_and_model():
    rows = fake_rows(20)
    lay = H.LAYOUT["utd_mhad"]
    for mech in ("runs", "whole", "frozen"):
        m = H.host_masks(rows, "S", "mid", 0, mech, lay)
        assert all(v.shape == (80, 20) for v in m.values())
    m = H.host_masks(rows, "S", "mid", 0, "armruns", lay)
    off = [j for j in range(20) if j not in lay["arm_joints"]]
    assert all(not v[:, off].any() for v in m.values())
    rows17 = [dict(r, kp=r["kp"][:, :17]) for r in fake_rows(17)]
    a = H.host_masks(rows17, "S", "mid", 0, "runs", H.LAYOUT["mmasd"])
    b = P.build_masks(rows17, "S", "mid", 0, mechanism="runs")[0]    # ASDPose defaults unchanged
    assert all(np.array_equal(a[k], b[k]) for k in a)
    net = CompactCTRGCN(c_in=3, num_joints=20)
    assert net(torch.zeros(2, 3, 64, 20)).shape == (2, 2)
    assert sum(p.numel() for p in CompactCTRGCN().parameters()) == 352850


def test_folds_and_perm():
    rows = fake_rows(17, n=22, subjects=11)
    with tempfile.TemporaryDirectory() as d:
        f1 = H.make_folds("mmasd_maracas", rows, seed=0, path=f"{d}/folds.json")
        f2 = H.make_folds("mmasd_maracas", rows, seed=0, path=f"{d}/folds.json")
    assert f1 == f2 and len(f1) == 5 and sorted(sum(f1, [])) == list(range(11))
    assert H.make_folds("utd_mhad_body_vs_arm", fake_rows(20, n=16, subjects=8)) == [[s] for s in range(8)]
    tr, va, te = H.split_fold(rows, f1[0])
    assert not ({r["subject"] for r in tr} & {r["subject"] for r in te})
    assert not ({r["subject"] for r in tr} & {r["subject"] for r in va})
    sev = H.severity_rows(rows, perm=3)
    flipped = {r["subject"] for r, s in zip(rows, sev) if r["binary_label"] != s["binary_label"]}
    assert len(flipped) == 5 and all(r["binary_label"] == s["binary_label"] or r["subject"] in flipped
                                     for r, s in zip(rows, sev))
    assert H.severity_rows(rows, perm=3) == sev                      # deterministic


def test_tiny_training_each_host():
    env = dict(os.environ, OMP_NUM_THREADS="2")
    for host in ("mmasd_maracas", "utd_mhad_body_vs_arm"):
        od = f"results/v9_smoke/{host}/runs_mid/S/A_none/seed0/fold1"
        if os.path.exists(f"{od}/train_log.json"):
            os.remove(f"{od}/train_log.json")
        r = subprocess.run([sys.executable, "experiments/train_v9.py", "--host", host, "--world", "S",
                            "--cell", "A_none", "--fold", "1", "--smoke", "--workers", "0"],
                           cwd=ROOT, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        log = json.load(open(f"{ROOT}/{od}/train_log.json"))
        assert log["epochs"] == 1 and log["n_test"] > 0
        cols = next(csv.reader(open(f"{ROOT}/{od}/test_scores.csv")))
        assert {"clean", "runs_S_mid_m2000", "runs_I_mid_m4000", "whole_S_mid_m2000"} <= set(cols)
        assert ("armruns_S_mid_m2000" in cols) == host.startswith("mmasd")


def _write_fold(d, ids, subj, y, cols):
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/test_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["identifier", "subject_id", "label", "n_frames", "qc_pass", "q_lowconf", "q_missing",
                    "q_frozen", "q_arm_lowconf"] + list(cols))
        for i in range(len(ids)):
            w.writerow([ids[i], subj[i], y[i], 150, i % 2, 0.1, 0.05, 0.0, 0.1] + [f"{cols[c][i]:.6f}" for c in cols])
    np.savez(f"{d}/features.npz", feats=np.random.rand(len(ids), 8), n_frames=np.full(len(ids), 150),
             label=np.array(y), subject=np.array(subj))


def test_analysis_on_synthetic_scores():
    rng = np.random.RandomState(0)
    with tempfile.TemporaryDirectory() as base:
        host, mechs = "mmasd_maracas", ("runs", "whole")
        subjects = list(range(10))
        folds = [subjects[i::5] for i in range(5)]
        ids = [f"c{s}_{k}" for s in subjects for k in range(6)]
        subj = [s for s in subjects for _ in range(6)]
        y = [k % 2 for _ in subjects for k in range(6)]
        conds = [f"{m}_{w}_mid_m{ms}" for m in mechs for w in ("S", "I") for ms in (2000, 3000, 4000)]
        conds += [f"runs_{w}_low_m{ms}" for w in ("S", "I") for ms in (2000, 3000, 4000)]
        conds += [f"gapfill_{w}_mid_m2000" for w in ("S", "I")]

        def model(world, cell, seed=None, perm=None, mech="runs", st="mid", real=False):
            who = f"perm{perm}" if perm is not None else f"seed{seed}"
            root = f"{base}/{host}/real/none/{cell}/{who}" if real else f"{base}/{host}/{mech}_{st}/{world}/{cell}/{who}"
            uses_cue = (world == "S" and perm is None and not real)
            for fi, fs in enumerate(folds):
                idx = [i for i in range(len(ids)) if subj[i] in fs]
                sig = np.array([y[i] for i in idx]) * 1.5 + rng.randn(len(idx))
                cols = {"clean": 1 / (1 + np.exp(-sig))}
                for c in conds:
                    boost = 1.5 if (uses_cue and "_S_" in c and c.startswith(mech)) else 0.0
                    cols[c] = 1 / (1 + np.exp(-(sig + boost * np.array([y[i] for i in idx]) + 0.3 * rng.randn(len(idx)))))
                _write_fold(f"{root}/fold{fi}", [ids[i] for i in idx], [subj[i] for i in idx],
                            [y[i] for i in idx], cols)
        for mech in mechs:
            for world in ("S", "I"):
                for cell in ("A_none", "A_dropout"):
                    for seed in (0, 1):
                        model(world, cell, seed=seed, mech=mech)
                for k in range(4):
                    model(world, "A_none", perm=k, mech=mech)
        for world in ("S", "I"):
            model(world, "A_none", seed=0, mech="runs", st="low")
        for cell in ("A_none", "A_dropout"):
            for seed in (0, 1, 2):
                model(None, cell, seed=seed, real=True)
        r = subprocess.run([sys.executable, "experiments/analyze_v9.py", "--base", base, "--n-boot", "30",
                            "--skip-gates"], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
        crit = json.load(open(f"{base}/analysis/criteria.json"))
        r1 = crit["R1_se_positive"][host]["per_mechanism"]
        assert set(r1) == set(mechs) and all(v["SE"][0] > 0.5 for v in r1.values())
        assert all(v["null_available"] and v["pass"] for v in r1.values())
        assert crit["R2_scaling"][host]["pass"] is not None
        assert set(crit["R3_dropout_does_not_remove"][host]["per_mechanism"]) == set(mechs)
        assert len(crit["R7_real_mmasd"]) == 2 * 3                    # 2 cells x (runs, whole, gapfill)
        assert "spearman_between_measures" in crit and all("ci" in v for v in crit["spearman_between_measures"].values())
        eff = list(csv.DictReader(open(f"{base}/analysis/effects.csv")))
        assert any(e["contrast"] == "SE(mid) - SE(low), A_none" for e in eff)
        assert sum(1 for _ in csv.DictReader(open(f"{base}/analysis/null.csv"))) == 2 * 4
        assert os.path.exists(f"{base}/analysis/length.csv") and os.path.exists(f"{base}/analysis/crossed.csv")
