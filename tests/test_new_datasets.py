"""Unit checks for the MMASD and UTD-MHAD loaders and the natural-cue audit (a few files only).

Run: conda run -n torch python tests/test_new_datasets.py
"""

import glob
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import planted as P
from safe_motion.audit import natural_cue as NC
from safe_motion.data import asdpose as A
from safe_motion.data import mmasd as MM
from safe_motion.data import utd_mhad as U


def test_utd_mhad_files():
    files = sorted(glob.glob(f"{U.SKEL_DIR}/a1_s1_t*_skeleton.mat"))[:3]
    assert len(files) == 3
    assert U.parse_name(files[0]) == {"identifier": "a1_s1_t1", "action": 1, "subject": 1, "trial": 1}
    xyz = U.load_mat(files[0])
    assert xyz.ndim == 3 and xyz.shape[1:] == (20, 3) and 40 <= len(xyz) <= 130
    kp = U.to_xyc(xyz)
    assert kp.shape == xyz.shape and (kp[..., 2] == 1).all() and np.array_equal(kp[..., :2], xyz[..., :2])
    assert xyz[:, 0, 1].mean() > xyz[:, 3, 1].mean()          # head above hip centre (y up)
    with tempfile.TemporaryDirectory() as d:
        U.build_cache(files=files, out=f"{d}/utd_mhad_skeleton.npz")
        rows = U.load_clips(cache_dir=d)
    assert [r["identifier"] for r in rows] == ["a1_s1_t1", "a1_s1_t2", "a1_s1_t3"]
    assert rows[0]["label"] == 0 and rows[0]["action_name"] == "swipe_left"
    assert np.array_equal(rows[0]["xyz"], xyz)
    assert sorted(U.FLIP_ORDER[j] for j in U.FLIP_ORDER) == list(range(20))
    assert all(U.FLIP_ORDER[U.FLIP_ORDER[j]] == j for j in range(20))
    assert [U.JOINTS[j] for j in U.ARM_JOINTS] == ["l_shoulder", "l_elbow", "l_wrist",
                                                   "r_shoulder", "r_elbow", "r_wrist"]
    fake = [{"subject": s, "identifier": str(s)} for s in range(1, 9)]
    tr, te, ts = U.subject_split(fake)
    assert ts == [2, 4, 6, 8] and {r["subject"] for r in tr} == {1, 3, 5, 7}
    a, b, v = U.subject_level_val_split(tr, 0.25, seed=0)
    assert len(v) == 1 and not ({r["subject"] for r in a} & set(v))
    assert U.bone_length_cv(np.repeat(xyz[:1], 10, axis=0)) < 1e-6   # rigid -> 0


def test_mmasd_names_and_layout():
    r = MM.parse_clip_name("as_20583_D16_000_i")
    assert (r["activity"], r["child_id"], r["day"], r["clip_index"], r["tag"]) == \
        ("arm_swing", 20583, 16, 0, "i")
    assert MM.parse_clip_name("mfs_40543_D8_012_y")["theme"] == "rhythm"
    coco = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow",
            "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle",
            "r_ankle"]
    assert [MM.BODY25[j] for j in MM.COCO17_FROM_BODY25] == coco
    assert [coco[j] for j in MM.ARM_JOINTS_COCO17] == [MM.BODY25[j] for j in
                                                        [5, 2, 6, 3, 7, 4]]
    assert set(MM.COCO17_FROM_BODY25[j] for j in MM.ARM_JOINTS_COCO17) == set(MM.ARM_JOINTS_BODY25)
    assert A.FLIP_ORDER[5] == 6                                   # COCO layout matches ASDPose


def test_mmasd_json_clip():
    folder = f"{MM.JSON_ROOT}/arm_swing_as/as_20583_D16_000_i/"
    name, n, det, fr = MM._read_clip(folder)
    assert name == "as_20583_D16_000_i" and n == 200 and det.shape[1:] == (25, 3)
    assert len(det) == len(fr) and fr.min() == 0 and fr.max() == 199 and len(det) >= n
    out, info = MM.select_person(det, fr, n, "track")
    assert out.shape == (200, 25, 3) and info["coverage"] > 0.9


def _person(cx, cy, conf, h=100.0):
    k = np.zeros((25, 3), np.float32)
    k[:, 0] = cx + np.linspace(-10, 10, 25)
    k[:, 1] = cy + np.linspace(-h / 2, h / 2, 25)
    k[:, 2] = conf
    return k


def test_mmasd_person_selection():
    # person A (x=100, conf 0.8) in every frame but frame 5; person B (x=400, conf 0.6) always,
    # listing order alternates; B is brighter (0.9) on frame 3 only.
    det, fr = [], []
    for t in range(10):
        ps = []
        if t != 5:
            ps.append(_person(100 + t, 200, 0.8))
        ps.append(_person(400, 200, 0.9 if t == 3 else 0.6))
        if t % 2:
            ps = ps[::-1]
        det += ps
        fr += [t] * len(ps)
    det, fr = np.stack(det), np.array(fr)
    out, info = MM.select_person(det, fr, 10, "track")
    assert (out[5] == 0).all()                                    # absent frame is all zero
    assert np.allclose(out[[0, 1, 2, 3, 4, 6, 9], 12, 0], 100 + np.array([0, 1, 2, 3, 4, 6, 9]))
    assert info["n_persistent"] == 2 and abs(info["coverage"] - 0.9) < 1e-9
    mc, _ = MM.select_person(det, fr, 10, "maxconf")
    assert np.isclose(mc[3, 12, 0], 400) and np.isclose(mc[5, 12, 0], 400)   # identity switches
    assert np.isclose(mc[0, 12, 0], 100)


def test_mmasd_ados_table():
    ados = MM.load_ados()
    assert len(ados) >= 30 and ados[20274]["comparison"] == 9.0 and ados[20274]["total"] == 21.0


def test_tracking_stats_match_smm_definition():
    rng = np.random.RandomState(0)
    kp = rng.rand(30, 17, 3).astype(np.float32)
    kp[3:6] = kp[2]                                                # three repeated frames
    kp[:, 4, 2] = 0.0
    arm = [5, 6, 7, 8, 9, 10]
    s = NC.tracking_stats(kp, arm)
    sc = kp[..., 2]
    assert np.isclose(s["q_lowconf"], (sc < 0.3).mean())
    assert np.isclose(s["q_missing"], (sc == 0).mean())
    assert np.isclose(s["q_frozen"], 3 / 29)
    assert np.isclose(s["q_arm_lowconf"], (sc[:, arm] < 0.3).mean())


def test_audit_evaluate():
    rng = np.random.RandomState(0)
    n = 400
    groups = np.repeat(np.arange(8), n // 8)
    y = rng.randint(0, 3, n)
    X = np.c_[y + 0.3 * rng.randn(n), rng.randn(n)]
    folds = NC.make_folds(y, groups)
    r = NC.evaluate(X, y, groups, "lr", folds, n_perm=3, n_boot=50)
    assert r["macro_auc"] > 0.9 and r["null_auc_mean"] < 0.65 and len(r["per_class_auc"]) == 3
    r0 = NC.evaluate(np.zeros((n, 2)), y, groups, "gb", folds)
    assert r0["degenerate"] and r0["macro_auc"] == 0.5
    yb = (y == 1).astype(int)
    rb = NC.evaluate(X[:, 1], yb, groups, "gb", NC.make_folds(yb, groups, n_splits=4), n_boot=50)
    assert 0.35 < rb["macro_auc"] < 0.65 and list(rb["per_class_auc"]) == [1]


def test_planted_other_layouts():
    rows = [{"identifier": f"id{i}", "binary_label": int(i % 2), "n_frames": 40} for i in range(20)]
    m, _ = P.build_masks(rows, "S", "mid", 0, mechanism="armruns", n_joints=25,
                         arm_joints=MM.ARM_JOINTS_BODY25)
    other = [j for j in range(25) if j not in MM.ARM_JOINTS_BODY25]
    assert all(v.shape == (40, 25) and not v[:, other].any() for v in m.values())
    w, _ = P.build_masks(rows, "S", "mid", 0, mechanism="runs", n_joints=20)
    assert all(v.shape == (40, 20) for v in w.values())
    d, _ = P.build_masks(rows, "S", "mid", 0)                     # default layout unchanged
    assert all(v.shape == (40, 17) for v in d.values())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
