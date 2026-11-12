"""Score the v10 checkpoints (configs/protocols/identification_v10.yaml, parts A and B).

Part A, quality-blind models (cache/pilot_v10_{S,I}_w0_{runs,whole}_mid_A_blind_s{k}.pt): own
mechanism at mid, worlds S and I, test masks 2000/3000/4000, plus gapfill on the runs masks (rule
A2). The blind transform is part of the checkpoint and is applied after the intervention.
Part B, real-data World I models (cache/pilot_v10_real_I_<cell>_s{k}.pt): the v7 real grid on the
full test split plus mask seeds 3000/4000 for missing runs. Output format matches results/v7/scores.
Writes results/v10/{scores/<model>.csv, features/<model>.npz, registry.json}.
    conda run -n torch python experiments/score_v10.py --workers 5
    conda run -n torch python experiments/score_v10.py --smoke --device cpu
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safe_motion import metrics as M                 # noqa: E402
from safe_motion import planted as P                 # noqa: E402
from safe_motion.data import asdpose as D            # noqa: E402
import score_v7 as S7                                # noqa: E402

OUT = "results/v10"
MECHS, MASK_SEEDS = S7.MECHS, S7.MASK_SEEDS
NAME = re.compile(r"^pilot_v10_(?:(S|I)_w(\d)_(runs|whole)_(mid|low)_(A_blind)|real_(I|S)_([AB]_[a-z]+))_s(\d)(_smoke)?$")


def registry(cache_dir="cache", smoke=False):
    models = []
    for path in sorted(glob.glob(f"{cache_dir}/pilot_v10_*.pt")):
        stem = os.path.basename(path)[:-3]
        m = NAME.match(stem)
        if not m or bool(m.group(9)) != smoke:
            continue
        world, ws, mech, st, blind_cell, real_world, real_cell, seed, _ = m.groups()
        if blind_cell:
            models.append({"name": stem[6:], "path": path, "proto": "v10", "arch": "compact",
                           "part": "A", "world": world, "world_seed": int(ws), "mechanism": mech,
                           "strength": st, "cell": blind_cell, "seed": int(seed)})
        else:
            models.append({"name": stem[6:], "path": path, "proto": "v10", "arch": "compact",
                           "part": "B", "world": None, "real_world": real_world, "world_seed": 0,
                           "mechanism": "runs", "strength": "mid", "cell": real_cell, "seed": int(seed)})
    return models


def conditions(m):
    c = []
    if m["part"] == "A":
        own, st = m["mechanism"], m["strength"]
        for w in ("S", "I"):
            for s in MASK_SEEDS:
                c.append((f"{own}_{w}_{st}_m{s}", own, P.MECHANISM_MODE[own], w, st, s))
        for w in ("S", "I"):
            c.append((f"gapfill_{w}_mid_m2000", "runs", "gapfill", w, "mid", 2000))
    else:
        for mech in MECHS:
            for w in ("S", "I"):
                c.append((f"{mech}_{w}_mid_m2000", mech, P.MECHANISM_MODE[mech], w, "mid", 2000))
        for w in ("S", "I"):
            for s in MASK_SEEDS[1:]:
                c.append((f"runs_{w}_mid_m{s}", "runs", "zero", w, "mid", s))
        for w in ("S", "I"):
            c.append((f"gapfill_{w}_mid_m2000", "runs", "gapfill", w, "mid", 2000))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="subsampled rows, smoke checkpoints only")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    ap.add_argument("--only", default=None, help="substring filter on model names")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or (OUT + ("_smoke" if args.smoke else ""))
    device = args.device or ("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    rows = [r for r in D.load_index() if r["n_frames"] >= 200]          # length protocol (v7)
    te_full = [r for r in rows if r["split"] == "test"]
    te_host = [r for r in te_full if D.passes_qc(r, S7.V1["qc_filter"])]
    if args.smoke:
        te_full, te_host = te_full[::15], te_host[::12]
    for d in ("scores", "features"):
        os.makedirs(f"{out}/{d}", exist_ok=True)
    print(f"test rows: full {len(te_full)}, host {len(te_host)}; device {device}", flush=True)
    models = registry(args.cache_dir, args.smoke)
    if args.only:
        models = [m for m in models if args.only in m["name"]]
    json.dump(models, open(f"{out}/registry.json", "w"), indent=1)
    masks = {}

    def mask_for(tag, rowset, mech, w, st, seed):
        key = (tag, mech, w, st, seed)
        if key not in masks:
            masks[key] = P.build_masks(rowset, w, st, seed, mechanism=mech)[0]
        return masks[key]

    t0 = time.time()
    for m in models:
        path = f"{out}/scores/{m['name']}.csv"
        if os.path.exists(path):
            continue
        rows_m, tag = (te_host, "host") if m["part"] == "A" else (te_full, "full")
        net, prep = S7.build(m)
        net = net.to(device)
        clean, feats = S7.run(net, rows_m, prep, None, "zero", device, args.workers, feats=True)
        cols = {"clean": clean}
        for col, mech, mode, w, st, seed in conditions(m):
            cols[col] = S7.run(net, rows_m, prep, mask_for(tag, rows_m, mech, w, st, seed), mode,
                               device, args.workers)[0]
        np.savez_compressed(f"{out}/features/{m['name']}.npz", feats=feats,
                            n_frames=np.array([r["n_frames"] for r in rows_m]),
                            label=np.array([r["binary_label"] for r in rows_m]),
                            child=np.array([r["child_id"] for r in rows_m]))
        with open(path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["identifier", "child_id", "label", "n_frames", "qc_pass"] + list(cols))
            for i, r in enumerate(rows_m):
                wr.writerow([r["identifier"], r["child_id"], r["binary_label"], r["n_frames"],
                             int(D.passes_qc(r, S7.V1["qc_filter"]))] + [f"{cols[c][i]:.6f}" for c in cols])
        y = [r["binary_label"] for r in rows_m]
        own = m["mechanism"]
        print(f"{m['name']}: clean {M.auc(clean, y):.4f} {own} S/I "
              f"{M.auc(cols[f'{own}_S_mid_m2000'], y):.4f}/{M.auc(cols[f'{own}_I_mid_m2000'], y):.4f} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
