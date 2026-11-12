"""Score the v12 remedy checkpoints (configs/protocols/remedies_v12.yaml).

Planted-host models cache/pilot_v12_{S,I}_w0_{runs,frozen}_mid_{C_groupdro,C_adversarial,C_augment}_s{k}.pt:
own mechanism at mid, worlds S and I, test-mask seeds 2000/3000/4000; the four operators at mid,
mask seed 2000 (crossed reading, P12_3); clean. Row sets, model rebuild, dataset and output
format are imported from experiments/score_v7.py, so the files are drop-in equivalents of
results/v7/scores. Writes results/v12/scores/<model>.csv, results/v12/registry.json.
    conda run -n torch python experiments/score_v12.py --workers 5
    conda run -n torch python experiments/score_v12.py --smoke --device cpu
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

OUT = "results/v12"
NAME = re.compile(r"^pilot_v12_(S|I)_w(\d)_(runs|frozen)_(mid)_(C_[a-z]+)_s(\d)(_smoke)?$")


def registry(cache_dir="cache", smoke=False):
    models = []
    for path in sorted(glob.glob(f"{cache_dir}/pilot_v12_*.pt")):
        stem = os.path.basename(path)[:-3]
        m = NAME.match(stem)
        if not m or bool(m.group(7)) != smoke:
            continue
        world, ws, mech, st, cell, seed, _ = m.groups()
        models.append({"name": stem[6:], "path": path, "proto": "v12", "arch": "compact",
                       "world": world, "world_seed": int(ws), "mechanism": mech, "strength": st,
                       "cell": cell, "seed": int(seed)})
    return models


def conditions(m):
    own, st = m["mechanism"], m["strength"]
    c = [(f"{own}_{w}_{st}_m{s}", own, P.MECHANISM_MODE[own], w, st, s)
         for w in ("S", "I") for s in S7.MASK_SEEDS]
    for mech in S7.MECHS:
        for w in ("S", "I"):
            c.append((f"{mech}_{w}_mid_m2000", mech, P.MECHANISM_MODE[mech], w, "mid", 2000))
    seen, out = set(), []
    for x in c:
        if x[0] not in seen:
            seen.add(x[0])
            out.append(x)
    return out


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
        net = net.to(device)
        cols = {"clean": S7.run(net, te_host, prep, None, "zero", device, args.workers)[0]}
        for col, mech, mode, w, st, seed in conditions(m):
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
        own = m["mechanism"]
        print(f"{m['name']}: clean {M.auc(cols['clean'], y):.4f} {own} S/I "
              f"{M.auc(cols[f'{own}_S_mid_m2000'], y):.4f}/{M.auc(cols[f'{own}_I_mid_m2000'], y):.4f} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
