"""Cue-targeted remedies of protocol v12 (configs/protocols/remedies_v12.yaml).

C_groupdro: Group DRO (Sagawa et al. 2020) over four groups, label x planted severity above or
    below the training split's median severity in the cell's world; exponentiated-gradient
    group weights, step 0.01.
C_adversarial: a head on the pooled features predicts the four tracking statistics of the
    planted input window through a gradient reversal layer, lambda 0.3.
C_augment: with probability 0.5 per step, one extra degradation drawn uniformly from the four
    operators, severity from the World I mixture (label-independent), re-drawn every step,
    applied after the world's planting.
The dataset subclass returns (x, y, group, stats); everything else is the v7 loader.
"""
import numpy as np
from safe_motion.eventlog import log_event
import torch
import torch.nn as nn

from safe_motion import planted as P
from safe_motion.data.asdpose import CACHE_DIR, SegmentDataset, apply_planted, load_cache

LOW_CONF = 0.3
STAT_NAMES = ("q_lowconf", "q_missing", "q_frozen", "q_arm_lowconf")
GROUPDRO_STEP = 0.01
ADV_LAMBDA = 0.3
AUG_PROB = 0.5
AUG_MECHS = ("runs", "whole", "frozen", "armruns")
MASK_FN = {"runs": P.missing_mask, "whole": P.whole_joint_mask, "frozen": P.frozen_mask,
           "armruns": P.arm_runs_mask}


def quality_stats(arr):
    """SAFE-SMM quality_features (asdpose-smm-detection stage1/build_cache.py) without energy:
    low-confidence, missing, frozen-frame and arm low-confidence fractions of a (T, 17, 3) window."""
    kp, sc = arr[..., :2], arr[..., 2]
    frozen = float((kp[1:] == kp[:-1]).all(axis=(1, 2)).mean()) if kp.shape[0] > 1 else 0.0
    return np.array([(sc < LOW_CONF).mean(), (sc == 0).mean(), frozen,
                     (sc[:, P.ARM_JOINTS] < LOW_CONF).mean()], dtype=np.float32)


def severity_groups(rows, eps, n_train):
    """identifier -> group 2*label + (eps above the median eps of the first n_train rows)."""
    med = float(np.median(eps[:n_train]))
    return {r["identifier"]: 2 * int(r["binary_label"]) + int(e > med) for r, e in zip(rows, eps)}


def observed_groups(rows, masks, mode, n_train, oracle=None, cache_dir=CACHE_DIR):
    """identifier -> 2*label + (measured arm low-confidence fraction above the median of the first
    n_train rows), plus the share of rows whose group equals the oracle (severity) group.

    The statistic is computed on the full planted segment, the input the model is trained on; the
    planted severity itself is never read (protocol v16, C_groupdro_obs)."""
    data = load_cache(cache_dir)
    stat = np.empty(len(rows), dtype=np.float64)
    for i, r in enumerate(rows):
        start = r["byte_offset"] // 4
        arr = data[start:start + r["n_frames"] * 51].reshape(r["n_frames"], 17, 3).copy()
        arr = apply_planted(arr, masks[r["identifier"]], mode)
        stat[i] = quality_stats(arr)[3]
    med = float(np.median(stat[:n_train]))
    groups = {r["identifier"]: 2 * int(r["binary_label"]) + int(v > med) for r, v in zip(rows, stat)}
    agree = (float(np.mean([groups[k] == oracle[k] for k in groups])) if oracle is not None else float("nan"))
    return groups, agree


def world_I_severity(rng, pi, strength="mid"):
    """One severity from the World I mixture: the SMM law with probability pi, else the non-SMM law."""
    s = P.STRENGTHS[strength]
    a, b = s["pos"] if rng.rand() < pi else s["neg"]
    return float(rng.beta(a, b))


class RemedyDataset(SegmentDataset):
    """SegmentDataset that also returns the remedy's auxiliary data.

    augment: apply the C_augment extra degradation (needs pi, the training class share).
    groups: identifier -> Group DRO group (C_groupdro), else group -1.
    Stats are computed on the cropped window as the model receives it, before normalisation."""

    def __init__(self, rows, augment=False, pi=0.5, groups=None, aug_seed=0, **kw):
        super().__init__(rows, **kw)
        self.augment, self.pi, self.groups = augment, pi, groups
        self.aug_seed, self._aug_rng, self._aug_key = aug_seed, None, None

    def _rng_aug(self):
        # re-seed per worker and per epoch: torch gives each worker a fresh initial seed per epoch
        key = torch.initial_seed()
        if self._aug_rng is None or key != self._aug_key:
            self._aug_rng = np.random.RandomState((key + self.aug_seed) % (2 ** 32))
            self._aug_key = key
        return self._aug_rng

    def augment_segment(self, arr):
        rng = self._rng_aug()
        if rng.rand() >= AUG_PROB:
            return arr
        mech = AUG_MECHS[rng.randint(len(AUG_MECHS))]
        eps = world_I_severity(rng, self.pi)
        if eps > 15 / 16:                       # logging only; no draw (repair_selection_v14)
            log_event("augment_eps_above_15_16", mech=mech, eps=f"{eps:.5f}", capped=eps > 0.95)
        mask = MASK_FN[mech](arr.shape[0], eps, rng)
        return apply_planted(arr, mask, P.MECHANISM_MODE[mech])

    def __getitem__(self, i):
        r = self.rows[i]
        arr = self.load_segment(r)
        if self.augment:
            arr = self.augment_segment(arr)
        arr = self._crop(arr)
        stats = torch.from_numpy(quality_stats(arr))
        x, y = self._encode(arr, r)
        g = -1 if self.groups is None else self.groups[r["identifier"]]
        return x, y, g, stats


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_reverse(x, lam=ADV_LAMBDA):
    return _GradReverse.apply(x, lam)


class AdversarialHead(nn.Module):
    """Predicts the standardised four tracking statistics from reversed pooled features."""

    def __init__(self, d_in, mean, sd, lam=ADV_LAMBDA):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Linear(64, len(STAT_NAMES)))
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("sd", torch.as_tensor(sd, dtype=torch.float32).clamp_min(1e-3))
        self.lam = lam

    def loss(self, feats, stats):
        pred = self.net(grad_reverse(feats, self.lam))
        return ((pred - (stats - self.mean) / self.sd) ** 2).mean()


class GroupDRO:
    """Online Group DRO weights (Sagawa et al. 2020, Algorithm 1), no group adjustment."""

    def __init__(self, n_groups=4, step=GROUPDRO_STEP, device="cpu"):
        self.q = torch.full((n_groups,), 1.0 / n_groups, device=device)
        self.step, self.n = step, n_groups

    def loss(self, per_sample, groups):
        losses = torch.zeros(self.n, device=per_sample.device)
        present = torch.zeros(self.n, dtype=torch.bool, device=per_sample.device)
        for g in range(self.n):
            m = groups == g
            if m.any():
                losses[g] = per_sample[m].mean()
                present[g] = True
        with torch.no_grad():
            self.q = self.q * torch.exp(self.step * losses.detach().float() * present.float())
            self.q = self.q / self.q.sum()
        return (self.q.to(losses.dtype) * losses).sum()
