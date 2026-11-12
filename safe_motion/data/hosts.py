"""Protocol v9 hosts: binary tasks, folds and a clip dataset for MMASD and UTD-MHAD.

Each row has identifier, binary_label, subject (child id or Kinect subject), n_frames,
kp (T, J, 3: x, y, confidence), the four tracking statistics (q_*) and qc_pass under the ASDPose
thresholds. Clips shorter than the host window are dropped. Label 0 is the class World S degrades
more: maracas_forward_shaking (MMASD), the arm gestures (UTD-MHAD).

ClipDataset mirrors asdpose.SegmentDataset on in-memory clips (crop, normalisation, flip,
dropout, velocity channels and the planted operators of asdpose.apply_planted).
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from safe_motion.data import asdpose as A
from safe_motion.data import mmasd as MM
from safe_motion.data import utd_mhad as U
from safe_motion import planted as P

QC_FROZEN_MAX, QC_ARM_MAX, LOW_CONF = 0.10, 0.30, 0.3
Q_COLS = ["q_lowconf", "q_missing", "q_frozen", "q_arm_lowconf"]

UTD_BODY = {"pickup_throw", "jog", "walk", "sit_to_stand", "stand_to_sit", "lunge", "squat"}
UTD_DRAW = {"draw_x", "draw_circle_cw", "draw_circle_ccw", "draw_triangle", "swipe_left", "swipe_right"}
UTD_GEST = {"wave", "clap", "throw", "arm_cross", "push", "knock"}

HOSTS = {
    "mmasd_maracas": {"dataset": "mmasd", "classes": {"maracas_forward_shaking": 0, "maracas_shaking": 1},
                      "window": 150, "epochs": 25, "folds": ("grouped", 5),
                      "mechanisms": ("runs", "whole", "frozen", "armruns")},
    "mmasd_chest_squat": {"dataset": "mmasd", "classes": {"chest_expansion": 0, "squat": 1},
                          "window": 150, "epochs": 25, "folds": ("grouped", 5),
                          "mechanisms": ("runs", "whole", "frozen", "armruns")},
    "utd_mhad_body_vs_arm": {"dataset": "utd_mhad", "classes": "body_vs_arm", "window": 64,
                             "epochs": 60, "folds": ("loso", None),
                             "mechanisms": ("runs", "whole", "frozen")},
    "utd_mhad_arm_only": {"dataset": "utd_mhad", "classes": "arm_only", "window": 64,
                          "epochs": 60, "folds": ("loso", None),
                          "mechanisms": ("runs", "whole", "frozen")},
}
LAYOUT = {"mmasd": {"n_joints": 17, "arm_joints": MM.ARM_JOINTS_COCO17, "flip_order": A.FLIP_ORDER,
                    "normalize": "est_h"},
          "utd_mhad": {"n_joints": 20, "arm_joints": U.ARM_JOINTS, "flip_order": U.FLIP_ORDER,
                       "normalize": "window-std"}}


def tracking_stats(kp, arm_joints, low_conf=LOW_CONF):
    """The four SAFE-SMM tracking statistics of one clip (as safe_motion.audit.natural_cue)."""
    xy, sc = kp[..., :2], kp[..., 2]
    frozen = float((xy[1:] == xy[:-1]).all(axis=(1, 2)).mean()) if len(kp) > 1 else 0.0
    return {"q_lowconf": float((sc < low_conf).mean()), "q_missing": float((sc == 0).mean()),
            "q_frozen": frozen, "q_arm_lowconf": float((sc[:, arm_joints] < low_conf).mean())}


def _finish(rows, layout, window):
    out = []
    for r in rows:
        if r["n_frames"] < window:
            continue
        r.update(tracking_stats(r["kp"], layout["arm_joints"]))
        r["qc_pass"] = int(r["q_frozen"] < QC_FROZEN_MAX and r["q_arm_lowconf"] < QC_ARM_MAX)
        out.append(r)
    return out


def load_host(name, rows=None):
    """Rows of one host (list of dicts). `rows` lets tests pass pre-loaded clips."""
    h = HOSTS[name]
    layout = LAYOUT[h["dataset"]]
    if h["dataset"] == "mmasd":
        if rows is None:
            rows, _ = MM.load_clips(rule="track", joints="coco17")
        keep = []
        for r in rows:
            if r["activity"] in h["classes"]:
                keep.append({"identifier": r["identifier"], "subject": int(r["child_id"]),
                             "binary_label": h["classes"][r["activity"]], "n_frames": int(r["n_frames"]),
                             "kp": r["kp"], "est_h": float(r["est_h"]), "activity": r["activity"]})
    else:
        if rows is None:
            rows = U.load_clips()
        keep = []
        for r in rows:
            a = r["action_name"]
            if h["classes"] == "body_vs_arm":
                lab = 1 if a in UTD_BODY else 0
            else:
                if a in UTD_DRAW:
                    lab = 1
                elif a in UTD_GEST:
                    lab = 0
                else:
                    continue
            keep.append({"identifier": r["identifier"], "subject": int(r["subject"]), "binary_label": lab,
                         "n_frames": int(r["n_frames"]), "kp": r["kp"], "est_h": 1.0, "activity": a})
    return _finish(keep, layout, h["window"])


def make_folds(name, rows, seed=0, path=None):
    """Sorted test-subject lists, one per fold. Saved to path (JSON) on first use and read back
    afterwards, so the assignment stays fixed."""
    if path and os.path.exists(path):
        return json.load(open(path))["folds"]
    kind, k = HOSTS[name]["folds"]
    subs = sorted({r["subject"] for r in rows})
    if kind == "loso":
        folds = [[s] for s in subs]
    else:
        order = np.random.RandomState(seed).permutation(subs).tolist()
        folds = [sorted(order[i::k]) for i in range(k)]
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump({"host": name, "seed": seed, "folds": folds}, open(path, "w"), indent=1)
    return folds


def split_fold(rows, test_subjects, val_fraction=0.25, seed=0):
    """(train, val, test) rows: test = the fold's subjects, val = a subject-level quarter of
    the rest."""
    test_subjects = set(test_subjects)
    te = [r for r in rows if r["subject"] in test_subjects]
    rest = [r for r in rows if r["subject"] not in test_subjects]
    subs = sorted({r["subject"] for r in rest})
    rng = np.random.RandomState(seed)
    val = set(rng.choice(subs, max(1, int(round(len(subs) * val_fraction))), replace=False).tolist())
    tr = [r for r in rest if r["subject"] not in val]
    va = [r for r in rest if r["subject"] in val]
    assert tr and va and te
    return tr, va, te


def severity_rows(rows, perm=None):
    """Rows whose binary_label sets the planted severities. With perm=K the label-to-severity
    pairing is flipped for a random half of the subjects (RandomState(K)), so the planted cue is
    absent in expectation (permutation null)."""
    if perm is None:
        return rows
    subs = sorted({r["subject"] for r in rows})
    flip = set(np.random.RandomState(perm).permutation(subs)[: len(subs) // 2].tolist())
    return [dict(r, binary_label=(1 - r["binary_label"]) if r["subject"] in flip else r["binary_label"])
            for r in rows]


def host_masks(rows, world, strength, seed, mechanism, layout, perm=None):
    sev = severity_rows(rows, perm)
    tag = seed if perm is None else f"{seed}p{perm}"
    return P.build_masks(sev, world, strength, tag, mechanism=mechanism,
                         n_joints=layout["n_joints"], arm_joints=layout["arm_joints"])[0]


class ClipDataset(Dataset):
    def __init__(self, rows, window, layout, train=False, flip_prob=0.5, seed=0,
                 joint_dropout=0.0, coord_dropout=0.0, velocity=False, planted=None,
                 planted_mode="zero"):
        self.rows, self.window, self.layout, self.train = rows, window, layout, train
        self.flip_prob, self.joint_dropout, self.coord_dropout = flip_prob, joint_dropout, coord_dropout
        self.velocity, self.planted, self.planted_mode = velocity, planted, planted_mode
        assert planted_mode in ("zero", "gapfill", "conflow", "hold", "confscale")
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.rows)

    def load_clip(self, r):
        arr = r["kp"].astype(np.float32).copy()
        if self.planted is not None:
            arr = A.apply_planted(arr, self.planted[r["identifier"]], self.planted_mode)
        return arr

    def __getitem__(self, i):
        r = self.rows[i]
        arr = self.load_clip(r)
        t, w, J = arr.shape[0], self.window, self.layout["n_joints"]
        if t < w:                                   # not reached under the protocol; smoke only
            arr = np.tile(arr, (int(np.ceil(w / t)), 1, 1))[:w]
        elif t > w:
            s = self.rng.randint(0, A.crop_start_high(t, w)) if self.train else (t - w) // 2
            arr = arr[s:s + w]
        xy, score = arr[..., :2], arr[..., 2].copy()
        valid = score > 0
        if valid.any():
            center = xy[valid].mean(axis=0)
            scale = (max(xy[valid].std(), 1e-3) if self.layout["normalize"] == "window-std"
                     else max(float(r["est_h"]), 1e-3))
            xy = np.where(valid[..., None], (xy - center) / scale, 0.0)
        fo = self.layout["flip_order"]
        if self.train and self.rng.rand() < self.flip_prob:
            xy, score = xy[:, fo], score[:, fo]
            xy[..., 0] = -xy[..., 0]
        if self.train and self.joint_dropout > 0:
            drop = self.rng.rand(J) < self.joint_dropout
            xy[:, drop] = 0.0
            score[:, drop] = 0.0
        if self.train and self.coord_dropout > 0:
            drop = self.rng.rand(J) < self.coord_dropout
            xy[:, drop] = 0.0
        chans = [xy]
        if self.velocity:
            chans.append(np.diff(xy, axis=0, prepend=xy[:1]))
        chans.append(score[..., None])
        out = np.concatenate(chans, axis=2)
        return torch.from_numpy(np.ascontiguousarray(out.transpose(2, 0, 1))).float(), int(r["binary_label"])
