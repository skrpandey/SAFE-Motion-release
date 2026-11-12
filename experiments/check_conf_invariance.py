"""Check that the x, y input channels are identical across test policies under confidence-only
planting (post hoc, preprocessing only, no inference).

Builds the final tensors through SegmentDataset (v7 A_none settings) for 200 host test segments,
World S and World I at test mask seed 2000, and compares x, y with each other and with the clean
tensor; the quality-blind path is expected to differ. Writes results/v10/analysis/conf_invariance.csv.
--all runs every host test segment (1,966), masks 2000/3000/4000 and both the evaluation and
training paths, and writes results/v13/analysis/conf_invariance_<rule>_all.csv.
"""
import argparse
import csv
import os
import sys

import numpy as np
import yaml

from safe_motion import planted as P
from safe_motion.data import asdpose as D

OUT = "results/v10/analysis/conf_invariance.csv"
V1 = yaml.safe_load(open("configs/protocols/pilot_asdpose_v1.yaml"))


def tensors(rows, planted, mode, blind, valid_rule="conf", train=False, seed=0):
    """Final input tensors (N, C, T, V) through SegmentDataset. With train=True the same seed and
    item order give the same crop and flip draws for every policy."""
    ds = D.SegmentDataset(rows, window=200, train=train, normalize="image",
                          confidence=not blind, planted=planted, planted_mode=mode, blind=blind,
                          valid_rule=valid_rule, seed=seed)
    return np.stack([ds[i][0].numpy() for i in range(len(rows))])     # (N, C, T, V)


def run_all(te_host, valid_rule, out_path):
    """Extended gate: all host test segments, masks 2000/3000/4000, evaluation and training paths,
    no-remedy preprocessing, arm-confidence operator. Returns True if x, y never differ."""
    out = []
    mode = P.MECHANISM_MODE["armruns"]
    for ms in (2000, 3000, 4000):
        masks = {w: P.build_masks(te_host, w, "mid", ms, mechanism="armruns")[0] for w in ("S", "I")}
        for train in (False, True):
            clean = tensors(te_host, None, mode, False, valid_rule, train=train, seed=0)
            ts = {w: tensors(te_host, masks[w], mode, False, valid_rule, train=train, seed=0) for w in ("S", "I")}
            for a, b, label in ((ts["S"], ts["I"], "S vs I"), (ts["S"], clean, "S vs clean"), (ts["I"], clean, "I vs clean")):
                d = np.abs(a[:, :2] - b[:, :2])
                per_seg = d.reshape(len(te_host), -1).max(axis=1)
                dc = np.abs(a[:, 2] - b[:, 2])
                out.append({"mechanism": "armruns", "operator": mode, "valid_rule": valid_rule, "mask_seed": ms,
                            "path": "training (matched crops and flips, seed 0)" if train else "evaluation (centre crop)",
                            "comparison": label, "n_segments": len(te_host),
                            "max_abs_diff_xy": f"{d.max():.3e}", "segments_with_any_xy_diff": int((per_seg > 0).sum()),
                            "conf_channel_max_abs_diff": f"{dc.max():.3e}",
                            "conf_channel_segments_with_diff": int((dc.reshape(len(te_host), -1).max(axis=1) > 0).sum())})
            print(f"mask {ms} {'train' if train else 'eval'}: max |dxy| {max(float(r['max_abs_diff_xy']) for r in out[-3:]):.3e}", flush=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    gate = all(float(r["max_abs_diff_xy"]) == 0.0 for r in out)
    print(f"extended invariance gate ({valid_rule} rule, {len(te_host)} segments, 3 masks, both paths): {'PASS' if gate else 'FAIL'}")
    return gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid-rule", default="conf", choices=("conf", "coords"),
                    help="centring rule; coords = protocol v13 (gate V13_1)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--all", action="store_true",
                    help="extended gate: all host test segments, masks 2000/3000/4000, evaluation and training paths")
    args = ap.parse_args()
    out_path = args.out or (OUT if args.valid_rule == "conf" else "results/v13/analysis/conf_invariance_v13.csv")
    rows = [r for r in D.load_index() if r["n_frames"] >= 200]
    te_host = [r for r in rows if r["split"] == "test" and D.passes_qc(r, V1["qc_filter"])]
    if args.all:
        ok = run_all(te_host, args.valid_rule, args.out or f"results/v13/analysis/conf_invariance_{args.valid_rule}_all.csv")
        sys.exit(0 if ok else 1)          # a failed gate stops the release driver
    sub = te_host[:200]
    out = []
    for mech in ("armruns", "runs"):
        mode = P.MECHANISM_MODE[mech]
        masks = {w: P.build_masks(te_host, w, "mid", 2000, mechanism=mech)[0] for w in ("S", "I")}
        for blind in (False, True):
            clean = tensors(sub, None, mode, blind, args.valid_rule)
            ts = {w: tensors(sub, masks[w], mode, blind, args.valid_rule) for w in ("S", "I")}
            xy = lambda t: t[:, :2]
            for a, b, label in ((ts["S"], ts["I"], "S vs I"), (ts["S"], clean, "S vs clean"), (ts["I"], clean, "I vs clean")):
                d = np.abs(xy(a) - xy(b))
                per_seg = d.reshape(len(sub), -1).max(axis=1)
                out.append({"mechanism": mech, "operator": mode, "valid_rule": args.valid_rule,
                            "path": "quality-blind" if blind else "standard (A_none)",
                            "comparison": label, "n_segments": len(sub),
                            "max_abs_diff_xy": f"{d.max():.3e}",
                            "segments_with_any_xy_diff": int((per_seg > 0).sum()),
                            "conf_channel_max_abs_diff": "n/a (dropped)" if blind else f"{np.abs(a[:, 2] - b[:, 2]).max():.3e}"})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    for r in out:
        print(r)
    if args.valid_rule == "coords":
        std = [r for r in out if r["path"].startswith("standard") and r["mechanism"] == "armruns"]
        gate = all(float(r["max_abs_diff_xy"]) == 0.0 for r in std)
        print(f"V13_1 invariance gate (armruns, standard path, coords rule): {'PASS' if gate else 'FAIL'}")
        if not gate:
            sys.exit(1)


if __name__ == "__main__":
    main()
