"""Score the protocol v13 checkpoints (configs/protocols/armconf_v13.yaml).

cache/pilot_v13_{S,I}_w0_armruns_mid_A_none_s{k}.pt are the arm-confidence A_none cells retrained
with the coordinate-invariant valid-joint rule ("coords", read from the checkpoint). Conditions:
own operator at mid for test masks 2000/3000/4000, the four operators at mask 2000, gapfill and
clean. Output format matches results/v7/scores.
Writes results/v13/scores/<model>.csv and results/v13/registry.json.
    conda run -n torch python experiments/score_v13.py --workers 5
    conda run -n torch python experiments/score_v13.py --smoke --device cpu
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

OUT = "results/v13"
NAME = re.compile(r"^pilot_v13_(S|I)_w(\d)_(armruns)_(mid)_(A_none)_s(\d)(_smoke)?$")


def registry(cache_dir="cache", smoke=False):
    models = []
    for path in sorted(glob.glob(f"{cache_dir}/pilot_v13_*.pt")):
        stem = os.path.basename(path)[:-3]
        m = NAME.match(stem)
        if not m or bool(m.group(7)) != smoke:
            continue
        world, ws, mech, st, cell, seed, _ = m.groups()
        models.append({"name": stem[6:], "path": path, "proto": "v13", "arch": "compact",
                       "world": world, "world_seed": int(ws), "mechanism": mech, "strength": st,
                       "cell": cell, "seed": int(seed), "valid_rule": "coords"})
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="subsampled rows, smoke checkpoints only")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or (OUT + ("_smoke" if args.smoke else ""))
    device = args.device or ("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    rows = [r for r in D.load_index() if r["n_frames"] >= 200]          # length protocol (v7)
    te_host = [r for r in rows if r["split"] == "test" and D.passes_qc(r, S7.V1["qc_filter"])]
    if args.smoke:
        te_host = te_host[::12]
    os.makedirs(f"{out}/scores", exist_ok=True)
    models = registry(args.cache_dir, args.smoke)
    json.dump(models, open(f"{out}/registry.json", "w"), indent=1)
    print(f"{len(models)} checkpoints, host test rows {len(te_host)}, device {device}", flush=True)
    masks, t0 = {}, time.time()
    for m in models:
        path = f"{out}/scores/{m['name']}.csv"
        if os.path.exists(path):
            continue
        net, prep = S7.build(m)
        assert prep["valid_rule"] == "coords", f"{m['name']}: checkpoint does not carry the v13 rule"
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
