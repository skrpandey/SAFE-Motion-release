"""Score the v7 and v8 checkpoints under the v8 grid (configs/protocols/pilot_asdpose_v8.yaml).

Planted-world models (host test split): own mechanism and strength in worlds S and I with three
test-mask seeds, the crossed table for A_none (four mechanisms at mid, seed 2000) and gapfill.
Real-split models (test segments of at least 200 frames): the four mechanisms at mid, gapfill and
clean. Also saves clean penultimate features and cue-only oracles per mechanism and world.
Writes results/v7/{scores/<model>.csv, features/<model>.npz, oracles.json, registry.json}.
    conda run -n torch python experiments/score_v7.py --workers 5
    conda run -n torch python experiments/score_v7.py --smoke     # CPU, v3 TinyTCN checkpoints
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
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import metrics as M
from safe_motion import planted as P
from safe_motion.data import asdpose as D
from safe_motion.models.ctrgcn import CompactCTRGCN
from safe_motion.models.baselines import STGCN
from safe_motion.models.tcn import TinyTCN

V1 = yaml.safe_load(open("configs/protocols/pilot_asdpose_v1.yaml"))
V2 = yaml.safe_load(open("configs/protocols/pilot_asdpose_v2.yaml"))
OUT = "results/v7"
MECHS = ("runs", "whole", "frozen", "armruns")
MASK_SEEDS = (2000, 3000, 4000)
LOW_CONF, ARM = 0.3, P.ARM_JOINTS
NAME = re.compile(r"^pilot_(v[78])_(?:(S|I|Sp)_w(\d)_(runs|whole|frozen|armruns)_(mid|low)|(real))_"
                  r"([AB]_[a-z]+)_s(\d)$")
SMOKE_NAME = re.compile(r"^pilot_v3_(mid|low)_(S|I)_([AB]_[a-z]+)_s(\d)$")


def registry(smoke=False, cache_dir="cache"):
    """One dict per checkpoint: name, proto, arch, world, world_seed, mechanism, strength, cell, seed."""
    models = []
    for path in sorted(glob.glob(f"{cache_dir}/pilot_v[78]_*.pt")):
        stem = os.path.basename(path)[:-3]
        m = NAME.match(stem)
        if not m:
            continue
        proto, world, ws, mech, st, real, cell, seed = m.groups()
        models.append({"name": stem[6:], "path": path, "proto": proto, "arch": "compact",
                       "world": world, "world_seed": int(ws) if ws else None,
                       "mechanism": mech, "strength": st, "cell": cell, "seed": int(seed)})
    if smoke:
        for path in sorted(glob.glob("cache/pilot_v3_mid_*_A_none_s0.pt")):
            st, world, cell, seed = SMOKE_NAME.match(os.path.basename(path)[:-3]).groups()
            models.append({"name": f"smoke_v3_{world}_{cell}", "path": path, "proto": "v3",
                           "arch": "tcn", "world": world, "world_seed": 0, "mechanism": "runs",
                           "strength": st, "cell": cell, "seed": int(seed)})
    return models


def build(m):
    ck = torch.load(m["path"], map_location="cpu", weights_only=False)
    blind = bool(ck["cell"].get("blind", False))        # v10 A_blind: no confidence channel
    c_in = 2 + 2 * ck["cell"]["velocity"] + (0 if blind else 1)
    arch = ck["cell"].get("arch", m["arch"])            # v16 records the architecture in the cell
    if arch == "stgcn":
        net = STGCN(c_in=c_in)
    else:
        net = CompactCTRGCN(c_in=c_in) if arch == "compact" else TinyTCN(c_in=c_in)
    net.load_state_dict(ck["state_dict"], strict=True)
    prep = {"normalize": "image", "velocity": bool(ck["cell"]["velocity"]), "confidence": not blind,
            "blind": blind, "valid_rule": ck["cell"].get("valid_rule", "conf")}   # v13: "coords"
    return net.eval(), prep


def conditions(m):
    """(column, mechanism, mode, world, strength, mask_seed) for one model. mode None = clean."""
    c = []
    if m["world"] is None:
        for mech in MECHS:
            for w in ("S", "I"):
                c.append((f"{mech}_{w}_mid_m2000", mech, P.MECHANISM_MODE[mech], w, "mid", 2000))
    else:
        own, st = m["mechanism"], m["strength"]
        for w in ("S", "I"):
            for s in MASK_SEEDS:
                c.append((f"{own}_{w}_{st}_m{s}", own, P.MECHANISM_MODE[own], w, st, s))
        if m["cell"] == "A_none":
            for mech in MECHS:
                for w in ("S", "I"):
                    c.append((f"{mech}_{w}_mid_m2000", mech, P.MECHANISM_MODE[mech], w, "mid", 2000))
    for w in ("S", "I"):
        c.append((f"gapfill_{w}_mid_m2000", "runs", "gapfill", w, "mid", 2000))
    seen, out = set(), []
    for x in c:
        if x[0] not in seen:
            seen.add(x[0])
            out.append(x)
    return out


@torch.no_grad()
def run(net, rows, prep, masks, mode, device, workers, batch=256, feats=False):
    ds = D.SegmentDataset(rows, train=False, planted=masks, planted_mode=mode, **prep)
    sc, ft = [], []
    for x, _ in DataLoader(ds, batch_size=batch, num_workers=workers):
        f = net.forward_features(x.to(device))
        sc.append(F.softmax(net.fc(f), 1)[:, 1].cpu().numpy())
        if feats:
            ft.append(f.cpu().numpy())
    return np.concatenate(sc), (np.concatenate(ft) if feats else None)


def tracking_stats(rows, masks, mode):
    ds = D.SegmentDataset(rows, train=False, planted=masks, planted_mode=mode)
    out = []
    for r in rows:
        a = ds.load_segment(r)
        kp, sc = a[..., :2], a[..., 2]
        frozen = float((kp[1:] == kp[:-1]).all(axis=(1, 2)).mean()) if len(kp) > 1 else 0.0
        out.append([float((sc < LOW_CONF).mean()), float((sc == 0).mean()), frozen,
                    float((sc[:, ARM] < LOW_CONF).mean())])
    return np.array(out)


def oracles(te, va):
    """Quality-only classifiers (host val -> host test) per mechanism and world at mid."""
    res = {}
    yv, yt = [r["binary_label"] for r in va], [r["binary_label"] for r in te]
    for mech in MECHS:
        mode = P.MECHANISM_MODE[mech]
        for w in ("S", "I"):
            mt = P.build_masks(te, w, "mid", 2000, mechanism=mech)[0]
            mv = P.build_masks(va, w, "mid", 2001, mechanism=mech)[0]
            xv, xt = tracking_stats(va, mv, mode), tracking_stats(te, mt, mode)
            sc = StandardScaler().fit(xv)
            lr = LogisticRegression(max_iter=2000).fit(sc.transform(xv), yv)
            gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05).fit(xv, yv)
            res[f"{mech}_{w}"] = {"lr_auc": M.auc(lr.predict_proba(sc.transform(xt))[:, 1], yt),
                                  "gb_auc": M.auc(gb.predict_proba(xt)[:, 1], yt),
                                  "mean_stats_by_class": {
                                      "smm": xt[np.array(yt) == 1].mean(0).round(4).tolist(),
                                      "non_smm": xt[np.array(yt) == 0].mean(0).round(4).tolist()}}
            print(f"oracle {mech} {w}: lr {res[f'{mech}_{w}']['lr_auc']:.3f} "
                  f"gb {res[f'{mech}_{w}']['gb_auc']:.3f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="CPU, subsampled rows, v3 checkpoints")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--skip-oracles", action="store_true")
    ap.add_argument("--only", default=None, help="substring filter on model names")
    ap.add_argument("--cache-dir", default="cache", help="checkpoint folder (repair_selection_v14)")
    ap.add_argument("--out", default=None, help="output folder (default results/v7)")
    args = ap.parse_args()
    out = args.out or (OUT + ("_smoke" if args.smoke else ""))
    device = "cuda" if torch.cuda.is_available() and not args.smoke else "cpu"
    rows = [r for r in D.load_index() if r["n_frames"] >= 200]          # length protocol (v7)
    te_full = [r for r in rows if r["split"] == "test"]
    te_host = [r for r in te_full if D.passes_qc(r, V1["qc_filter"])]
    _, va, _ = D.subject_level_val_split(rows, 0.2, seed=0)
    va_host = [r for r in va if D.passes_qc(r, V1["qc_filter"])]
    if args.smoke:
        te_full, te_host, va_host = te_full[::15], te_host[::12], va_host[::12]
    for d in ("scores", "features"):
        os.makedirs(f"{out}/{d}", exist_ok=True)
    print(f"test rows: full {len(te_full)}, host {len(te_host)}; device {device}", flush=True)

    if not args.skip_oracles and not os.path.exists(f"{out}/oracles.json"):
        json.dump(oracles(te_host, va_host), open(f"{out}/oracles.json", "w"), indent=1)

    models = registry(args.smoke, args.cache_dir)
    if args.only:
        models = [m for m in models if args.only in m["name"]]
    if args.smoke:
        models = [m for m in models if m["proto"] == "v3"][:2]
    json.dump(models, open(f"{out}/registry.json", "w"), indent=1)
    masks = {}

    def mask_for(rowset_tag, rowset, mech, w, st, seed):
        key = (rowset_tag, mech, w, st, seed)
        if key not in masks:
            masks[key] = P.build_masks(rowset, w, st, seed, mechanism=mech)[0]
        return masks[key]

    t0 = time.time()
    for m in models:
        path = f"{out}/scores/{m['name']}.csv"
        if os.path.exists(path):
            continue
        real = m["world"] is None
        rows_m, tag = (te_full, "full") if real else (te_host, "host")
        net, prep = build(m)
        net = net.to(device)
        clean, feats = run(net, rows_m, prep, None, "zero", device, args.workers, feats=True)
        cols = {"clean": clean}
        for col, mech, mode, w, st, seed in conditions(m):
            cols[col] = run(net, rows_m, prep, mask_for(tag, rows_m, mech, w, st, seed), mode,
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
                             int(D.passes_qc(r, V1["qc_filter"]))] + [f"{cols[c][i]:.6f}" for c in cols])
        y = [r["binary_label"] for r in rows_m]
        own = m["mechanism"] or "runs"
        st = m["strength"] or "mid"
        print(f"{m['name']}: clean {M.auc(clean, y):.4f} {own} S/I "
              f"{M.auc(cols[f'{own}_S_{st}_m2000'], y):.4f}/{M.auc(cols[f'{own}_I_{st}_m2000'], y):.4f} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
