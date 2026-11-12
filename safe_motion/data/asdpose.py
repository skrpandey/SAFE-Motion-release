"""ASDPose segment data, read from the SAFE-SMM cache (read-only).

Provenance: SegmentDataset, load_index and subject_level_val_split are copied from
asdpose-smm-detection (commit e5217a5) stage1/dataset.py and stage1/protocols.py. Changes:
the cache is preloaded into RAM once per process, `coord_dropout` is added (pilot protocol v1),
and the QC filter reads its thresholds from the frozen protocol YAML.
Each item is (C, window, 17): x, y [, dx, dy] [, confidence].
"""

import csv
import os

import numpy as np
from safe_motion.eventlog import log_event
import torch
from torch.utils.data import Dataset

# Data roots: set SM_SMM_ROOT (a checkout of the SAFE-SMM code with its cache/) or the cache
# directories directly; defaults fall back to data/ under the current working directory.
SMM_ROOT = os.environ.get("SM_SMM_ROOT", "data/asdpose-smm-detection")
CACHE_DIR = os.environ.get("SM_ASDPOSE_CACHE", os.environ.get("SM_CACHE_DIR", f"{SMM_ROOT}/cache/segments"))
TRACKING_STATS = ["q_lowconf", "q_missing", "q_frozen", "q_arm_lowconf"]

LEFT_KP = [1, 3, 5, 7, 9, 11, 13, 15]
RIGHT_KP = [2, 4, 6, 8, 10, 12, 14, 16]
FLIP_ORDER = list(range(17))
for _l, _r in zip(LEFT_KP, RIGHT_KP):
    FLIP_ORDER[_l], FLIP_ORDER[_r] = _r, _l

_CACHE = {}
# Training recipe. "legacy" (the 164 fits of v7 to v14): random crop start in [0, T - window), and
# the dataset's RandomState is copied unchanged into each DataLoader worker. "v15" (the 12 v15conf
# fits): start in [0, T - window] and one seeded stream per persistent worker (worker_init). Test-time
# preprocessing is the same under both; see configs/protocols/train_recipe_v15.md.
TRAIN_RECIPE = os.environ.get("SM_TRAIN_RECIPE", "legacy")
assert TRAIN_RECIPE in ("legacy", "v15"), TRAIN_RECIPE


def crop_start_high(t, window):
    """Exclusive upper bound of the random crop start for a segment of t frames."""
    return t - window + 1 if TRAIN_RECIPE == "v15" else t - window


def worker_init(worker_id):
    """DataLoader worker_init_fn for recipe v15: one RandomState per (dataset seed, worker)."""
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    ds.rng = np.random.RandomState((int(getattr(ds, "seed", 0)) * 1009 + worker_id + 1) % (2 ** 32))


def load_cache(cache_dir=CACHE_DIR):
    if cache_dir not in _CACHE:
        _CACHE[cache_dir] = np.fromfile(f"{cache_dir}/keypoints.f32", dtype=np.float32)
    return _CACHE[cache_dir]


def load_index(path=f"{CACHE_DIR}/index.csv"):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["binary_label"] = int(r["binary_label"])
        r["child_id"] = int(r["child_id"])
        r["n_frames"] = int(r["n_frames"])
        r["byte_offset"] = int(r["byte_offset"])
        for k in ("q_lowconf", "q_missing", "q_frozen", "q_energy", "q_arm_lowconf"):
            r[k] = float(r[k])
    return rows


def passes_qc(r, qc):
    return r["q_frozen"] < qc["q_frozen_max"] and r["q_arm_lowconf"] < qc["q_arm_lowconf_max"]


def qc_pool(rows, qc):
    """Frozen QC filter: negatives must pass, positives always kept."""
    return [r for r in rows if r["binary_label"] == 1 or passes_qc(r, qc)]


def subject_level_val_split(rows, val_fraction=0.2, seed=0):
    train_rows = [r for r in rows if r["split"] == "train"]
    children = sorted({r["child_id"] for r in train_rows})
    rng = np.random.RandomState(seed)
    val_children = set(rng.choice(children, int(len(children) * val_fraction),
                                  replace=False).tolist())
    tr = [r for r in train_rows if r["child_id"] not in val_children]
    va = [r for r in train_rows if r["child_id"] in val_children]
    assert not ({r["child_id"] for r in tr} & {r["child_id"] for r in va})
    return tr, va, sorted(val_children)


def apply_planted(arr, mask, mode):
    """Apply a planted mask to a raw (T, 17, 3) segment in place; returns the array.

    zero: x = y = confidence = 0 on masked joint-frames.
    gapfill: coordinates linearly interpolated per joint across masked frames, confidence kept.
        The interpolation still leaves a readable trace of the masked fraction.
    conflow: confidence set to 0.05 on masked joint-frames, coordinates kept (below the 0.3
        threshold but above zero, so preprocessing keeps the coordinates).
    hold: masked frames (whole rows, see planted.frozen_mask) copy the last unmasked frame; a run
        at frame 0 copies the first unmasked frame.
    confscale: masked confidence multiplied by 0.3 and floored at 0.05, coordinates kept, so the
        planted values fall in [0.05, 0.3).
    """
    if mode == "zero":
        arr[mask] = 0.0
        return arr
    if mode == "conflow":
        arr[..., 2][mask] = 0.05
        return arr
    if mode == "confscale":
        arr[..., 2][mask] = np.maximum(arr[..., 2][mask] * 0.3, 0.05)
        return arr
    if mode == "hold":
        frozen = mask.all(axis=1)
        if frozen.all():                        # whole segment frozen: left unchanged (logged, no draw)
            log_event("hold_all_frozen", n_frames=int(arr.shape[0]))
            return arr
        if not frozen.any():
            return arr
        idx = np.arange(arr.shape[0])
        last = np.where(frozen, -1, idx)
        last = np.maximum.accumulate(last)                 # last unmasked frame at or before t
        first_good = int(np.flatnonzero(~frozen)[0])
        last = np.where(last < 0, first_good, last)        # run at frame 0 copies forward
        arr[frozen] = arr[last[frozen]]
        return arr
    t = np.arange(arr.shape[0])
    for j in range(arr.shape[1]):
        m = mask[:, j]
        if not m.any():
            continue
        good = (~m) & (arr[:, j, 2] > 0)
        if good.sum() >= 2:
            for c in (0, 1):
                arr[m, j, c] = np.interp(t[m], t[good], arr[good, j, c])
        elif good.sum() == 1:
            arr[m, j, :2] = arr[good, j, :2][0]
        else:
            arr[m, j, :2] = 0.0
            arr[m, j, 2] = 0.0
    return arr


BLIND_LOW_CONF = 0.3


def blind_transform(arr, low_conf=BLIND_LOW_CONF):
    """Quality-blind input transform of protocol v10, applied after any planted intervention.

    Joint-frames with confidence below low_conf or x = y = 0 are treated as missing and
    interpolated per joint (the gapfill rule). Confidence becomes 1 for joints observed at least
    once and 0 otherwise; the caller drops the channel (confidence=False). The interpolated
    coordinates still carry the planted severity (probe R^2 0.76), so the input is not cue-free."""
    arr = arr.copy()
    missing = (arr[..., 2] < low_conf) | ((arr[..., 0] == 0) & (arr[..., 1] == 0))
    observed = ~missing
    arr[..., 2] = np.where(observed, 1.0, 0.0)          # gapfill's "good" test is conf > 0
    arr = apply_planted(arr, missing, "gapfill")
    seen = observed.any(axis=0)                          # joints with at least one observation
    arr[..., 2] = np.where(seen[None, :], 1.0, 0.0)
    return arr


class SegmentDataset(Dataset):
    def __init__(self, rows, window=200, train=False, flip_prob=0.5, seed=0,
                 joint_dropout=0.0, coord_dropout=0.0, velocity=False,
                 normalize="image", confidence=True, cache_dir=CACHE_DIR, planted=None,
                 planted_mode="zero", blind=False, valid_rule="conf"):
        assert normalize in ("window-std", "image")
        assert valid_rule in ("conf", "coords")
        self.valid_rule = valid_rule     # v13: "coords" = x or y nonzero (confidence-invariant centring)
        self.rows, self.window, self.train = rows, window, train
        self.flip_prob, self.joint_dropout, self.coord_dropout = flip_prob, joint_dropout, coord_dropout
        self.velocity, self.normalize, self.confidence = velocity, normalize, confidence
        self.data = load_cache(cache_dir)
        self.planted = planted        # identifier -> (n_frames, 17) bool mask (safe_motion.planted)
        assert planted_mode in ("zero", "gapfill", "conflow", "hold", "confscale")
        self.planted_mode = planted_mode   # v6 operators, see apply_planted
        self.blind = blind                 # v10 quality-blind transform, after the intervention
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.rows)

    def load_segment(self, r):
        start = r["byte_offset"] // 4
        arr = self.data[start:start + r["n_frames"] * 51].reshape(r["n_frames"], 17, 3).copy()
        if self.planted is not None:
            arr = apply_planted(arr, self.planted[r["identifier"]], self.planted_mode)
        if self.blind:
            arr = blind_transform(arr)
        return arr

    def __getitem__(self, i):
        r = self.rows[i]
        return self._encode(self._crop(self.load_segment(r)), r)

    def _crop(self, arr):
        """Fixed window: tile short segments, crop long ones (random in training, centre at test)."""
        t, w = arr.shape[0], self.window
        if t < w:
            arr = np.tile(arr, (int(np.ceil(w / t)), 1, 1))[:w]
        elif t > w:
            s = self.rng.randint(0, crop_start_high(t, w)) if self.train else (t - w) // 2
            arr = arr[s:s + w]
        return arr

    def _encode(self, arr, r):
        """Normalise, augment (flip, dropouts) and stack channels; returns (x, label)."""
        xy, score = arr[..., :2], arr[..., 2].copy()
        # valid joint-frames for centring: "conf" (v1 to v12) uses confidence > 0; "coords" (protocol
        # v13) uses nonzero coordinates, so a confidence floor cannot move the centroid
        valid = (score > 0) if self.valid_rule == "conf" else ((xy[..., 0] != 0) | (xy[..., 1] != 0))
        if valid.any():
            center = xy[valid].mean(axis=0)
            scale = (max(xy[valid].std(), 1e-3) if self.normalize == "window-std"
                     else float(r["img_h"]) / 2.0)
            xy = np.where(valid[..., None], (xy - center) / scale, 0.0)
        if self.train and self.rng.rand() < self.flip_prob:
            xy, score = xy[:, FLIP_ORDER], score[:, FLIP_ORDER]
            xy[..., 0] = -xy[..., 0]
        if self.train and self.joint_dropout > 0:
            # whole-joint dropout: coordinates and confidence, missingness stops being informative
            drop = self.rng.rand(17) < self.joint_dropout
            xy[:, drop] = 0.0
            score[:, drop] = 0.0
        if self.train and self.coord_dropout > 0:
            # coordinates only: the confidence channel keeps the real missing-joint pattern
            drop = self.rng.rand(17) < self.coord_dropout
            xy[:, drop] = 0.0

        chans = [xy]
        if self.velocity:
            chans.append(np.diff(xy, axis=0, prepend=xy[:1]))
        if self.confidence:
            chans.append(score[..., None])
        out = np.concatenate(chans, axis=2)
        x = torch.from_numpy(np.ascontiguousarray(out.transpose(2, 0, 1))).float()
        return x, int(r["binary_label"])
