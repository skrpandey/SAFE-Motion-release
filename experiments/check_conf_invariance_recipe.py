"""Gate G20_invariance of protocol v20 (configs/protocols/recipe_comparison.yaml).

For the A_both preprocessing (velocity inputs, whole-joint dropout 0.15, coords valid rule) and the
arm-confidence operator: the x, y and velocity channels of the clean, policy S and policy I inputs
must be identical for all host test segments, test masks 2000/3000/4000, and both the evaluation
path (centre crop, no dropout) and the training path (same seed and item order, so crops, flips and
dropout draws match across policies). Preprocessing only, no inference, CPU.
Writes results/recipe_comparison/analysis/conf_invariance_v20.csv; exit code 1 if any coordinate or velocity
value differs.
    python experiments/check_conf_invariance_recipe.py
"""
import csv
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import planted as P                 # noqa: E402
from safe_motion.data import asdpose as D            # noqa: E402

OUT = "results/recipe_comparison/analysis/conf_invariance_v20.csv"
V1 = yaml.safe_load(open("configs/protocols/pilot_1.yaml"))
CELL = {"velocity": True, "joint_dropout": 0.15, "coord_dropout": 0.0}


def tensors(rows, planted, mode, train):
    ds = D.SegmentDataset(rows, window=200, train=train, normalize="image", confidence=True,
                          planted=planted, planted_mode=mode, valid_rule="coords", seed=0, **CELL)
    return np.stack([ds[i][0].numpy() for i in range(len(rows))])     # (N, 5, T, V): x, y, vx, vy, c


def main():
    rows = [r for r in D.load_index() if r["n_frames"] >= 200]
    te = [r for r in rows if r["split"] == "test" and D.passes_qc(r, V1["qc_filter"])]
    mode = P.MECHANISM_MODE["armruns"]
    out = []
    for ms in (2000, 3000, 4000):
        masks = {w: P.build_masks(te, w, "mid", ms, mechanism="armruns")[0] for w in ("S", "I")}
        for train in (False, True):
            clean = tensors(te, None, mode, train)
            ts = {w: tensors(te, masks[w], mode, train) for w in ("S", "I")}
            for a, b, label in ((ts["S"], ts["I"], "S vs I"), (ts["S"], clean, "S vs clean"), (ts["I"], clean, "I vs clean")):
                d = np.abs(a[:, :4] - b[:, :4])
                dc = np.abs(a[:, 4] - b[:, 4])
                out.append({"cell": "A_both", "valid_rule": "coords", "mask_seed": ms,
                            "path": "training (matched crops, flips, dropout; seed 0)" if train else "evaluation (centre crop)",
                            "comparison": label, "n_segments": len(te),
                            "max_abs_diff_xy_velocity": f"{d.max():.3e}",
                            "segments_with_any_diff": int((d.reshape(len(te), -1).max(axis=1) > 0).sum()),
                            "conf_channel_segments_with_diff": int((dc.reshape(len(te), -1).max(axis=1) > 0).sum())})
            print(f"mask {ms} {'train' if train else 'eval'}: max |d(x, y, vx, vy)| "
                  f"{max(float(r['max_abs_diff_xy_velocity']) for r in out[-3:]):.3e}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    ok = all(float(r["max_abs_diff_xy_velocity"]) == 0.0 for r in out)
    planted_seen = any(r["conf_channel_segments_with_diff"] > 0 for r in out)
    print(f"G20_invariance ({len(te)} segments, 3 masks, both paths): {'PASS' if ok else 'FAIL'}; "
          f"confidence channel differs as intended: {planted_seen}")
    sys.exit(0 if ok and planted_seen else 1)


if __name__ == "__main__":
    main()
