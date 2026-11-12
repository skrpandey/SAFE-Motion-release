"""Test-time replacement of the confidence channel for the protocol v16 models (ST-GCN and
CompactCTRGCN no remedy, Group DRO with observed groups), as score_conf_reliance.py. Scoring only.

The 1,966 host test windows are scored clean, with the confidence channel permuted across
segments (five permutations), with it set to its mean over joint-frames with nonzero coordinates,
and under planted policies S and I (first test mask), as planted and permuted. Coordinates do not
depend on confidence under the nonzero-coordinate rule. The clean-to-permuted AUROC drop is a
permutation reliance measure (Fisher et al., 2019).
Writes <out>/scores/<model>.csv and <out>/registry.json.
    conda run -n torch python experiments/score_conf_reliance_v16.py --cache-dir results/fits/v16/cache/clean --out results/v16/conf_reliance
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
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safe_motion import planted as P                 # noqa: E402
from safe_motion.data import asdpose as D            # noqa: E402
import score_v7 as S7                                # noqa: E402

NAME = re.compile(r"^pilot_(v16)_(S|I)_w0_armruns_mid_(A_none_stgcn|A_none|C_groupdro_obs)_s(\d)$")
ARCH = {"A_none_stgcn": "stgcn", "A_none": "compact", "C_groupdro_obs": "compact"}
N_PERM = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", default="results/v16/conf_reliance")
    ap.add_argument("--expect", type=int, default=24)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = [r for r in D.load_index() if r["n_frames"] >= 200]
    te = [r for r in rows if r["split"] == "test" and D.passes_qc(r, S7.V1["qc_filter"])]
    ds = D.SegmentDataset(te, train=False, normalize="image", confidence=True, valid_rule="coords")
    X = torch.stack([ds[i][0] for i in range(len(te))])                 # (N, 3, T, V)
    xy_nonzero = (X[:, 0] != 0) | (X[:, 1] != 0)
    const = float(X[:, 2][xy_nonzero].mean())
    rng = np.random.RandomState(20261109)
    perms = [torch.from_numpy(rng.permutation(len(te))) for _ in range(N_PERM)]
    variants = {"clean": X}
    for k, p in enumerate(perms):
        v = X.clone(); v[:, 2] = X[p, 2]; variants[f"perm{k}"] = v
    v = X.clone(); v[:, 2] = torch.where(xy_nonzero, torch.full_like(X[:, 2], const), torch.zeros_like(X[:, 2]))
    variants["const"] = v
    for w in ("S", "I"):                                  # planted test policies, first mask
        mask = P.build_masks(te, w, "mid", 2000, mechanism="armruns")[0]
        dsw = D.SegmentDataset(te, train=False, normalize="image", confidence=True, valid_rule="coords",
                               planted=mask, planted_mode=P.MECHANISM_MODE["armruns"])
        Xw = torch.stack([dsw[i][0] for i in range(len(te))])
        assert torch.equal(Xw[:, :2], X[:, :2]), "coordinates must not change under confidence-only planting"
        variants[f"pol{w}"] = Xw
        for k, p_ in enumerate(perms):
            v = Xw.clone(); v[:, 2] = Xw[p_, 2]; variants[f"pol{w}_perm{k}"] = v
    os.makedirs(f"{args.out}/scores", exist_ok=True)
    models, t0 = [], time.time()
    for proto, pat in (("v16", f"{args.cache_dir}/pilot_v16_*.pt"),):
        for path in sorted(glob.glob(pat)):
            stem = os.path.basename(path)[:-3]
            m = NAME.match(stem)
            if not m:
                continue
            _, world, cell, seed = m.groups()
            meta = {"name": stem[6:], "path": path, "proto": proto, "arch": ARCH[cell], "world": world, "world_seed": 0,
                    "mechanism": "armruns", "strength": "mid", "cell": cell, "seed": int(seed), "valid_rule": "coords"}
            net, prep = S7.build(meta)
            assert prep["valid_rule"] == "coords" and prep["confidence"], meta["name"]
            net = net.to(device).eval()
            cols = {}
            with torch.no_grad():
                for name, V in variants.items():
                    out = []
                    for i in range(0, len(V), 256):
                        out.append(F.softmax(net(V[i:i + 256].to(device)), 1)[:, 1].cpu().numpy())
                    cols[name] = np.concatenate(out)
            with open(f"{args.out}/scores/{meta['name']}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["identifier", "child_id", "label"] + list(cols))
                for i, r in enumerate(te):
                    w.writerow([r["identifier"], r["child_id"], r["binary_label"]] + [f"{cols[c][i]:.6f}" for c in cols])
            models.append(meta)
            print(f"{meta['name']}: done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    json.dump({"models": models, "n_perm": N_PERM, "const": const, "perm_seed": 20261109, "n_rows": len(te)},
              open(f"{args.out}/registry.json", "w"), indent=1)
    assert len(models) == args.expect, len(models)
    print("done", len(models), "models")


if __name__ == "__main__":
    main()
