"""UTD-MHAD Kinect v1 skeletons, parsed into per-clip (T, 20, 3) arrays: x, y, confidence.

Source: $SM_UTD_ROOT/Skeleton/a{action}_s{subject}_t{trial}_skeleton.mat, variable d_skel of shape
(20, 3, T), joint order as in Skeleton/Skeleton_joint_order.txt. 861 clips (three of the 864 are
absent from the release), 30 fps, 41 to 125 frames.

x, y are Kinect camera-space coordinates in metres (depth dropped). The release stores no tracking
state, so conf is only a validity mask (1 where the joint holds a reading; no zero joints occur):
planted operators that write low confidence have no natural counterpart on this set. z is kept in
the cache (key xyz) for the bone-length proxy.

Standard cross-subject protocol (Chen et al., ICIP 2015): subjects 1, 3, 5, 7 train, 2, 4, 6, 8 test.
Cache: $SM_UTD_ROOT/cache/utd_mhad_skeleton.npz (flat xyz, offsets, labels, subjects).
"""

import glob
import os
import re

import numpy as np
import scipy.io as sio

ROOT = os.environ.get("SM_UTD_ROOT", "data/utd-mhad")
SKEL_DIR = f"{ROOT}/Skeleton"
CACHE = f"{ROOT}/cache"

JOINTS = ["head", "shoulder_center", "spine", "hip_center", "l_shoulder", "l_elbow", "l_wrist",
          "l_hand", "r_shoulder", "r_elbow", "r_wrist", "r_hand", "l_hip", "l_knee", "l_ankle",
          "l_foot", "r_hip", "r_knee", "r_ankle", "r_foot"]
ARM_JOINTS = [4, 5, 6, 8, 9, 10]          # shoulders, elbows, wrists (as COCO-17 arm set)
LEFT = [4, 5, 6, 7, 12, 13, 14, 15]
RIGHT = [8, 9, 10, 11, 16, 17, 18, 19]
FLIP_ORDER = list(range(20))
for _l, _r in zip(LEFT, RIGHT):
    FLIP_ORDER[_l], FLIP_ORDER[_r] = _r, _l
BONES = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
         (10, 11), (3, 12), (12, 13), (13, 14), (14, 15), (3, 16), (16, 17), (17, 18), (18, 19)]
ACTIONS = ["swipe_left", "swipe_right", "wave", "clap", "throw", "arm_cross", "basketball_shoot",
           "draw_x", "draw_circle_cw", "draw_circle_ccw", "draw_triangle", "bowling", "boxing",
           "baseball_swing", "tennis_swing", "arm_curl", "tennis_serve", "push", "knock",
           "catch", "pickup_throw", "jog", "walk", "sit_to_stand", "stand_to_sit", "lunge",
           "squat"]
TRAIN_SUBJECTS = (1, 3, 5, 7)
TEST_SUBJECTS = (2, 4, 6, 8)
NAME_RE = re.compile(r"^a(\d+)_s(\d+)_t(\d+)_skeleton\.mat$")


def parse_name(fname):
    m = NAME_RE.match(os.path.basename(fname))
    if m is None:
        raise ValueError(fname)
    a, s, t = (int(g) for g in m.groups())
    return {"identifier": f"a{a}_s{s}_t{t}", "action": a, "subject": s, "trial": t}


def load_mat(path):
    """(T, 20, 3) float32 camera-space xyz in metres."""
    return sio.loadmat(path)["d_skel"].transpose(2, 0, 1).astype(np.float32)


def to_xyc(xyz):
    """(T, 20, 3) x, y, validity confidence (1 where the joint holds a reading)."""
    conf = (np.abs(xyz).sum(axis=2) > 0).astype(np.float32)
    return np.concatenate([xyz[..., :2], conf[..., None]], axis=2)


def build_cache(skel_dir=SKEL_DIR, out=f"{CACHE}/utd_mhad_skeleton.npz", files=None):
    files = files if files is not None else sorted(glob.glob(f"{skel_dir}/a*_s*_t*_skeleton.mat"))
    meta = [parse_name(f) for f in files]
    arrs = [load_mat(f) for f in files]
    n = np.array([len(a) for a in arrs], np.int32)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, identifier=np.array([m["identifier"] for m in meta]),
             action=np.array([m["action"] for m in meta]),
             subject=np.array([m["subject"] for m in meta]),
             trial=np.array([m["trial"] for m in meta]), n_frames=n,
             offsets=np.r_[0, np.cumsum(n)].astype(np.int64), xyz=np.concatenate(arrs))
    return out


def load_clips(cache_dir=CACHE):
    """List of row dicts: 'kp' (T, 20, 3) x, y, conf; 'xyz' (T, 20, 3); label = action - 1."""
    path = f"{cache_dir}/utd_mhad_skeleton.npz"
    if not os.path.exists(path):
        build_cache(out=path)
    z = np.load(path)
    rows = []
    for i, ident in enumerate(z["identifier"]):
        xyz = z["xyz"][z["offsets"][i]:z["offsets"][i + 1]]
        a = int(z["action"][i])
        rows.append({"identifier": str(ident), "action": a, "action_name": ACTIONS[a - 1],
                     "label": a - 1, "subject": int(z["subject"][i]), "trial": int(z["trial"][i]),
                     "n_frames": int(z["n_frames"][i]), "xyz": xyz, "kp": to_xyc(xyz)})
    return rows


def subject_split(rows, test_subjects=TEST_SUBJECTS):
    """Cross-subject split (default: the standard odd-train / even-test protocol)."""
    test_subjects = set(test_subjects)
    tr = [r for r in rows if r["subject"] not in test_subjects]
    te = [r for r in rows if r["subject"] in test_subjects]
    assert not ({r["subject"] for r in tr} & {r["subject"] for r in te})
    return tr, te, sorted(test_subjects)


def subject_level_val_split(train_rows, val_fraction=0.25, seed=0):
    """Validation subjects drawn from the training subjects (as asdpose.subject_level_val_split)."""
    subs = sorted({r["subject"] for r in train_rows})
    rng = np.random.RandomState(seed)
    val = set(rng.choice(subs, max(1, int(len(subs) * val_fraction)), replace=False).tolist())
    return ([r for r in train_rows if r["subject"] not in val],
            [r for r in train_rows if r["subject"] in val], sorted(val))


def bone_length_cv(xyz):
    """Mean over bones of std/mean of 3D bone length across frames (a tracking-noise proxy:
    Kinect inferred joints break the rigid-length constraint)."""
    L = np.stack([np.linalg.norm(xyz[:, a] - xyz[:, b], axis=1) for a, b in BONES], axis=1)
    return float((L.std(axis=0) / np.maximum(L.mean(axis=0), 1e-6)).mean())


if __name__ == "__main__":
    print(build_cache())
