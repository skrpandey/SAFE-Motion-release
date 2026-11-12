"""Post hoc adversary diagnostic, feature extraction only (no training).

For the 12 v12 adversarial-head fits and their 12 no-remedy references (v7 runs, v8 frozen, mid,
cleaned selection), extracts the 136-d penultimate features of the 1,966 host test windows under
the model's own training-world policy (first test mask), with the four tracking statistics of the
planted window as probe targets. Writes <out>/<model>.npz and <out>/registry.json and appends the
GPU time to results/gpu_ledger.csv. analyze_adversary_probe.py fits the probes.
    conda run -n torch python experiments/score_adversary_probe.py --out results/posthoc/adversary_probe
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch                                         # noqa: E402
from safe_motion import planted as P                 # noqa: E402
from safe_motion import remedies as RM               # noqa: E402
from safe_motion.data import asdpose as D            # noqa: E402
import score_v7 as S7                                # noqa: E402

CACHE = "results/fits/repair_v14/cache/clean"
NAME = re.compile(r"^pilot_(v7|v8|v12)_(S|I)_w0_(runs|frozen)_mid_(A_none|C_adversarial)_s(\d)$")
KEEP = {("v7", "runs", "A_none"), ("v8", "frozen", "A_none"), ("v12", "runs", "C_adversarial"), ("v12", "frozen", "C_adversarial")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/posthoc/adversary_probe")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [r for r in D.load_index() if r["n_frames"] >= 200]
    te = [r for r in rows if r["split"] == "test" and D.passes_qc(r, S7.V1["qc_filter"])]
    os.makedirs(args.out, exist_ok=True)
    models, masks, stats, t0 = [], {}, {}, time.time()
    for path in sorted(glob.glob(f"{CACHE}/pilot_v*_w0_*_mid_*.pt")):
        stem = os.path.basename(path)[:-3]
        m = NAME.match(stem)
        if not m or (m.group(1), m.group(3), m.group(4)) not in KEEP:
            continue
        proto, world, mech, cell, seed = m.groups()
        meta = {"name": stem[6:], "path": path, "proto": proto, "arch": "compact", "world": world, "world_seed": 0,
                "mechanism": mech, "strength": "mid", "cell": cell, "seed": int(seed)}
        key = (mech, world)
        if key not in masks:
            masks[key] = P.build_masks(te, world, "mid", 2000, mechanism=mech)[0]
            ds = D.SegmentDataset(te, train=False, planted=masks[key], planted_mode=P.MECHANISM_MODE[mech])
            stats[key] = np.stack([RM.quality_stats(ds._crop(ds.load_segment(r))) for r in te])
        net, prep = S7.build(meta)
        net = net.to(device).eval()
        with torch.no_grad():
            _, feats = S7.run(net, te, prep, masks[key], P.MECHANISM_MODE[mech], device, args.workers, feats=True)
        np.savez_compressed(f"{args.out}/{meta['name']}.npz", features=feats.astype(np.float32), stats=stats[key],
                            identifier=np.array([r["identifier"] for r in te]),
                            child_id=np.array([str(r["child_id"]) for r in te]),
                            label=np.array([r["binary_label"] for r in te]))
        models.append(meta)
        print(f"{meta['name']}: features {feats.shape} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    assert len(models) == 24, len(models)
    json.dump({"models": models, "mask_seed": 2000, "stats": ["low_conf", "missing", "frozen", "arm_low_conf"]},
              open(f"{args.out}/registry.json", "w"), indent=1)
    with open("results/gpu_ledger.csv", "a", newline="") as f:
        csv.writer(f).writerow([time.strftime("%Y-%m-%d"), "posthoc/adversary_probe (24 models; feature extraction only)",
                                f"{(time.time() - t0) / 3600:.3f}"])
    print("done", len(models))


if __name__ == "__main__":
    main()
