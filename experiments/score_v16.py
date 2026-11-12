"""Score the protocol v16 checkpoints (configs/protocols/confonly_extension_v16.yaml).

ST-GCN and CompactCTRGCN no-remedy cells and Group DRO with observed groups, all trained under
confidence-only planting with the coordinate-invariant valid-joint rule ("coords"; checkpoints
without it are refused). Conditions as score_v15conf.py: clean, own operator at mid in worlds S and
I with mask seeds 2000/3000/4000, and the four operators at mask 2000.
    python experiments/score_v16.py --workers 6 --cache-dir cache/v16/clean --out results/v16/clean
    python experiments/score_v16.py --device cpu --subsample 12 --cache-dir cache/v16_smoke/clean --out results/v16_smoke/clean
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safe_motion import metrics as M                 # noqa: E402
from safe_motion import planted as P                 # noqa: E402
from safe_motion.data import asdpose as D            # noqa: E402
import score_v7 as S7                                # noqa: E402
import score_v12 as S12                              # noqa: E402

OUT = "results/v16/clean"
# v17 (configs/protocols/confonly_assignment_v17.yaml) reuses this scorer: same cells, rule and
# conditions, with the world seed of the regenerated training assignment in the name (w1, w2, w3).
NAME = re.compile(r"^pilot_(v16|v17)_(S|I)_w(\d)_(armruns)_(mid)_(A_none_stgcn|A_none|C_groupdro_obs)_s(\d)$")
ARCH = {"A_none_stgcn": "stgcn", "A_none": "compact", "C_groupdro_obs": "compact"}


def registry(cache_dir, proto="v16"):
    models = []
    for path in sorted(glob.glob(f"{cache_dir}/pilot_v1[67]_*.pt")):
        stem = os.path.basename(path)[:-3]
        m = NAME.match(stem)
        if not m or m.group(1) != proto:
            continue
        _, world, ws, mech, st, cell, seed = m.groups()
        models.append({"name": stem[6:], "path": path, "proto": proto, "arch": ARCH[cell],
                       "world": world, "world_seed": int(ws), "mechanism": mech, "strength": st,
                       "cell": cell, "seed": int(seed), "valid_rule": "coords", "cache_dir": cache_dir})
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache-dir", default="cache/v16/clean")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--subsample", type=int, default=1, help="smoke: every n-th host test row")
    ap.add_argument("--proto", default="v16", choices=["v16", "v17"], help="which protocol's checkpoints to score")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = [r for r in D.load_index() if r["n_frames"] >= 200]          # length protocol (v7)
    te_host = [r for r in rows if r["split"] == "test" and D.passes_qc(r, S7.V1["qc_filter"])]
    te_host = te_host[::args.subsample]
    os.makedirs(f"{args.out}/scores", exist_ok=True)
    models = registry(args.cache_dir, args.proto)
    json.dump(models, open(f"{args.out}/registry.json", "w"), indent=1)
    print(f"{len(models)} checkpoints, host test rows {len(te_host)}, device {device}", flush=True)
    masks, t0 = {}, time.time()
    for m in models:
        path = f"{args.out}/scores/{m['name']}.csv"
        if os.path.exists(path):
            continue
        net, prep = S7.build(m)
        assert prep["valid_rule"] == "coords", f"{m['name']}: checkpoint does not carry the coords rule"
        net = net.to(device)
        cols = {"clean": S7.run(net, te_host, prep, None, "zero", device, args.workers)[0]}
        for col, mech, mode, w, st, seed in S12.conditions(m):
            key = (mech, w, st, seed)
            if key not in masks:
                masks[key] = P.build_masks(te_host, w, st, seed, mechanism=mech)[0]
            cols[col] = S7.run(net, te_host, prep, masks[key], mode, device, args.workers)[0]
        with open(path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["identifier", "child_id", "label", "n_frames", "qc_pass"] + list(cols))
            for i, r in enumerate(te_host):
                wr.writerow([r["identifier"], r["child_id"], r["binary_label"], r["n_frames"],
                             int(D.passes_qc(r, S7.V1["qc_filter"]))] + [f"{cols[c][i]:.6f}" for c in cols])
        y = [r["binary_label"] for r in te_host]
        print(f"{m['name']}: clean {M.auc(cols['clean'], y):.4f} armruns S/I "
              f"{M.auc(cols['armruns_S_mid_m2000'], y):.4f}/{M.auc(cols['armruns_I_mid_m2000'], y):.4f} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
