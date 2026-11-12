"""Pilot v5: score every model under test-time planted interventions (protocol pilot_asdpose_v5).

Per model, nine passes over the full ASDPose test split: clean, then {runs, whole} x {mid, low}
x {S, I}. Masks are built once per condition on the test rows (seed 2000) and shared by all
models. Each model uses its own preprocessing from its checkpoint. Output:
results/pilots/pilot_v5/scores/<model>.csv with one column per condition.

Usage: conda run -n torch python experiments/score_intervention.py [--smoke] [--only NAME ...]
"""

import argparse
import csv
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import planted as P
from safe_motion.data import asdpose as D
from safe_motion.models.baselines import AGCN, STGCN
from safe_motion.models.ctrgcn import CompactCTRGCN
from safe_motion.models.tcn import TinyTCN

PROTO = "configs/protocols/pilot_asdpose_v5.yaml"
SMM_CACHE = f"{D.SMM_ROOT}/cache"
OUT = "results/pilots/pilot_v5/scores"
MASK_SEED = 2000
CONDITIONS = [("runs", "mid"), ("runs", "low"), ("whole", "mid"), ("whole", "low")]


def registry(cfg):
    """List of dicts: name, family, arch, path, prep (normalize, velocity, confidence)."""
    models = []
    smm = cfg["models"]["safe_smm_checkpoints"]
    for arch in ("compact", "stgcn", "agcn"):
        for n in smm[arch]:
            ck = torch.load(f"{SMM_CACHE}/{n}.pt", map_location="cpu", weights_only=False)
            d, m = ck["config"]["data"], ck["config"]["model"]
            models.append({"name": n, "family": "safe_smm", "arch": arch,
                           "path": f"{SMM_CACHE}/{n}.pt", "model_cfg": m,
                           "prep": {"normalize": d.get("normalize_mode", "window-std"),
                                    "velocity": d.get("velocity", False),
                                    "confidence": d.get("confidence", True)}})
    tcn = {"A_none": [0, 1, 2, 3, 4, 5], "A_both": [0, 1, 2, 3, 4, 5],
           "B_coorddrop": [0, 1, 2, 3, 4, 5], "A_velocity": [0, 1, 2], "A_dropout": [0, 1, 2],
           "D_qcsym_none": [0, 1, 2], "D_qcsym_velocity": [0, 1, 2]}
    for cell, seeds in tcn.items():
        for s in seeds:
            if s == 0 and not cell.startswith("D_"):
                path = f"cache/pilot_{cell}.pt"                 # v1 batch-2 run, seed 0
            elif s <= 2:
                path = f"cache/pilot_v2_{cell}_s{s}.pt"
            else:
                path = f"cache/pilot_v3_real_{cell}_s{s}.pt"
            models.append({"name": f"tcn_{cell}_s{s}", "family": "pilot_tcn", "arch": "tcn",
                           "path": path, "cell": cell, "seed": s})
    # v5 addendum 1: planted-host models from v3 (TinyTCN) and v4 (CompactCTRGCN)
    for w in ("I", "S"):
        for s in (0, 1, 2):
            models.append({"name": f"tcn_v3_{w}_s{s}", "family": "addendum1", "arch": "tcn",
                           "path": f"cache/pilot_v3_mid_{w}_A_none_s{s}.pt", "cell": "A_none", "seed": s})
            models.append({"name": f"compact_v4_{w}_s{s}", "family": "addendum1", "arch": "compact",
                           "path": f"cache/pilot_v4_compact_runs_mid_{w}_A_none_s{s}.pt",
                           "model_cfg": {"c_in": 3, "widths": [48, 96, 136], "blocks_per_stage": 2},
                           "prep": {"normalize": "image", "velocity": False, "confidence": True}})
    for p_ in range(100, 110):
        models.append({"name": f"tcn_A_none_perm{p_}", "family": "pilot_tcn_perm", "arch": "tcn",
                       "path": f"cache/pilot_v2_A_none_perm{p_}.pt", "cell": "A_none", "seed": p_})
    for m in models:
        if m["arch"] == "tcn":
            ck = torch.load(m["path"], map_location="cpu", weights_only=False)
            c = ck["cell"]
            m["prep"] = {"normalize": "image", "velocity": c["velocity"], "confidence": True}
    return models


def build(m):
    ck = torch.load(m["path"], map_location="cpu", weights_only=False)
    c_in = 2 + 2 * m["prep"]["velocity"] + m["prep"]["confidence"]
    if m["arch"] == "compact":
        mc = m["model_cfg"]
        net = CompactCTRGCN(c_in=mc["c_in"], widths=tuple(mc["widths"]),
                            blocks_per_stage=mc["blocks_per_stage"])
    elif m["arch"] == "stgcn":
        net = STGCN(c_in=m["model_cfg"]["c_in"])
    elif m["arch"] == "agcn":
        net = AGCN(c_in=m["model_cfg"]["c_in"])
    else:
        net = TinyTCN(c_in=c_in)
    net.load_state_dict(ck["state_dict"], strict=True)
    return net.eval()


@torch.no_grad()
def score(net, rows, prep, masks, device, workers, batch=256):
    ds = D.SegmentDataset(rows, train=False, planted=masks, **prep)
    out = []
    for x, _ in DataLoader(ds, batch_size=batch, num_workers=workers):
        out.append(F.softmax(net(x.to(device)), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--family", default=None, help="score only this registry family")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(PROTO))
    device = "cpu" if args.smoke else "cuda"
    rows = [r for r in D.load_index() if r["split"] == "test"]
    if args.smoke:
        rows = rows[::24]
    os.makedirs(OUT, exist_ok=True)

    masks = {}
    for mech, st in CONDITIONS:
        for w in ("S", "I"):
            masks[(mech, st, w)] = P.build_masks(rows, w, st, MASK_SEED, mechanism=mech)[0]
    print(f"masks built for {len(rows)} rows", flush=True)

    models = registry(cfg)
    if args.only:
        models = [m for m in models if m["name"] in args.only]
    if args.family:
        models = [m for m in models if m["family"] == args.family]
    if args.smoke:
        models = [m for m in models if m["name"] in
                  ("stage1_detector_v3", "wacv_stgcn_base", "wacv_agcn_base", "tcn_A_none_s0",
                   "tcn_D_qcsym_none_s1", "tcn_A_none_perm100")]
    t0 = time.time()
    for m in models:
        path = f"{OUT}/{m['name']}.csv"
        if os.path.exists(path) and not args.smoke:
            continue
        net = build(m).to(device)
        cols = {"clean": score(net, rows, m["prep"], None, device, args.workers)}
        for (mech, st, w), mk in masks.items():
            cols[f"{mech}_{st}_{w}"] = score(net, rows, m["prep"], mk, device, args.workers)
        with open(path if not args.smoke else path.replace(".csv", "_smoke.csv"), "w",
                  newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["identifier", "binary_label"] + list(cols))
            for i, r in enumerate(rows):
                wr.writerow([r["identifier"], r["binary_label"]] +
                            [f"{cols[k][i]:.6f}" for k in cols])
        print(f"{m['name']:32s} clean AUC {_auc(cols['clean'], rows):.4f}  "
              f"S {_auc(cols['runs_mid_S'], rows):.4f} I {_auc(cols['runs_mid_I'], rows):.4f}  "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    if not args.smoke:
        with open("results/gpu_ledger.csv", "a", newline="") as f:
            csv.writer(f).writerow([time.strftime("%Y-%m-%d"), "pilot_v5/score_intervention",
                                    f"{(time.time() - t0) / 3600:.3f}"])


def _auc(s, rows):
    from safe_motion.metrics import auc
    return auc(s, [r["binary_label"] for r in rows])


if __name__ == "__main__":
    main()
