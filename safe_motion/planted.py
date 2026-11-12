"""Planted tracking-quality worlds on the ASDPose QC host (from protocol pilot_asdpose_v3).

Missing runs: each joint follows a two-state Markov chain with stationary missing probability eps
and mean run length RUN_LEN frames; missing joint-frames get x = y = confidence = 0. eps = 0 is the
identity and the missing fraction rises with eps. Masks are fixed per segment (same in every epoch).
Whole-joint absence (v4): each joint absent for the whole segment with probability eps. Frozen
frames (v8): whole-skeleton runs holding the last pose (mode "hold"). Arm confidence (v8): the
missing-runs chain on the six arm joints, confidence rescaled below 0.3, coordinates kept.
World S: eps depends on the label (non-SMM degraded more). World I: same marginal, independent of
the label. World Sp (v6): the mirror of S.
"""

import hashlib

import numpy as np

RUN_LEN = 15.0
FROZEN_RUN_LEN = 20.0
ARM_JOINTS = [5, 6, 7, 8, 9, 10]          # COCO-17 shoulders, elbows, wrists
MECHANISM_MODE = {"runs": "zero", "whole": "zero", "frozen": "hold", "armruns": "confscale"}
STRENGTHS = {"mid": {"neg": (2.0, 4.0), "pos": (2.0, 8.0)},
             "low": {"neg": (2.0, 4.0), "pos": (2.0, 6.0)}}


def _rng(identifier, tag):
    h = hashlib.sha256(f"{identifier}|{tag}".encode()).digest()
    return np.random.RandomState(int.from_bytes(h[:4], "little"))


def missing_mask(n_frames, eps, rng, run_len=RUN_LEN, n_joints=17):
    """Boolean (n_frames, n_joints) mask, True = joint-frame made missing."""
    if eps <= 0:
        return np.zeros((n_frames, n_joints), bool)
    eps = min(eps, 0.95)
    p_exit = 1.0 / run_len
    p_enter = min(eps * p_exit / (1.0 - eps), 1.0)
    m = np.empty((n_frames, n_joints), bool)
    state = rng.rand(n_joints) < eps
    u = rng.rand(n_frames, n_joints)
    for t in range(n_frames):
        state = np.where(state, u[t] >= p_exit, u[t] < p_enter)
        m[t] = state
    return m


def whole_joint_mask(n_frames, eps, rng, n_joints=17):
    """Boolean (n_frames, n_joints) mask: each joint absent for all frames with probability eps."""
    if eps <= 0:
        return np.zeros((n_frames, n_joints), bool)
    return np.repeat((rng.rand(n_joints) < eps)[None, :], n_frames, axis=0)


def frozen_mask(n_frames, eps, rng, run_len=FROZEN_RUN_LEN, n_joints=17):
    """Boolean (n_frames, n_joints) mask with identical rows: whole-skeleton frozen runs."""
    if eps <= 0:
        return np.zeros((n_frames, n_joints), bool)
    m = missing_mask(n_frames, eps, rng, run_len=run_len, n_joints=1)
    return np.repeat(m, n_joints, axis=1)


def arm_runs_mask(n_frames, eps, rng, n_joints=17, arm_joints=None):
    """The v3 joint-missing chain on the six arm joints only; other joints untouched.

    arm_joints defaults to the COCO-17 ARM_JOINTS; other layouts pass their own (e.g.
    safe_motion.data.mmasd.ARM_JOINTS_BODY25, safe_motion.data.utd_mhad.ARM_JOINTS)."""
    arm = ARM_JOINTS if arm_joints is None else list(arm_joints)
    m = np.zeros((n_frames, n_joints), bool)
    m[:, arm] = missing_mask(n_frames, eps, rng, n_joints=len(arm))
    return m


def severities(rows, world, strength, seed):
    """Per-row eps. Label-dependent in World S; same marginal, label-independent in World I."""
    s = STRENGTHS[strength]
    y = np.array([r["binary_label"] for r in rows])
    out = np.empty(len(rows))
    pi = y.mean()
    for i, r in enumerate(rows):
        rng = _rng(r["identifier"], f"sev|{world}|{strength}|{seed}")
        if world == "S":
            a, b = s["pos"] if y[i] == 1 else s["neg"]
        elif world == "Sp":                      # mirrored direction
            a, b = s["neg"] if y[i] == 1 else s["pos"]
        elif world == "I":
            a, b = s["pos"] if rng.rand() < pi else s["neg"]
        else:
            raise ValueError(world)
        out[i] = rng.beta(a, b)
    return out


def build_masks(rows, world, strength, seed, mechanism="runs", n_joints=17, arm_joints=None):
    """identifier -> mask over the segment's full length (crop happens later).

    mechanism "runs" reproduces the v3 masks exactly; "whole" uses separate RNG tags.
    n_joints and arm_joints (armruns only) default to the ASDPose COCO-17 layout, for which the
    masks are unchanged; rows need a binary "binary_label" (a binary host task)."""
    if mechanism == "runs":
        eps = severities(rows, world, strength, seed)
        return {r["identifier"]: missing_mask(r["n_frames"], e,
                                              _rng(r["identifier"], f"mask|{world}|{strength}|{seed}"),
                                              n_joints=n_joints)
                for r, e in zip(rows, eps)}, eps
    fn = {"whole": whole_joint_mask, "frozen": frozen_mask, "armruns": arm_runs_mask}[mechanism]
    kw = {"n_joints": n_joints}
    if mechanism == "armruns":
        kw["arm_joints"] = arm_joints
    eps = severities(rows, world, strength, f"{mechanism}{seed}")
    return {r["identifier"]: fn(r["n_frames"], e,
                                _rng(r["identifier"], f"{mechanism}mask|{world}|{strength}|{seed}"),
                                **kw)
            for r, e in zip(rows, eps)}, eps
