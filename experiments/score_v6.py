"""Pilot v6: score models on the host test split under the v6 operator grid; cue-only oracles.

Conditions per model (protocol pilot_asdpose_v6.yaml): clean; zero x {S, I, Sp} x mid x 3 mask
seeds; zero x {S, I} x low; gapfill and conflow x {S, I, Sp} x mid. Writes
results/pilots/pilot_v6/scores/<model>.csv (one column per condition) and
results/pilots/pilot_v6/oracles.json (tracking-statistic oracles per operator and world).

Usage: conda run -n torch python experiments/score_v6.py [--smoke] [--family F] [--workers 5]
"""

import argparse
import csv
import json
import os
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_intervention as SI
from safe_motion import metrics as M
from safe_motion import planted as P
from safe_motion.data import asdpose as D

V1 = yaml.safe_load(open("configs/protocols/pilot_asdpose_v1.yaml"))
OUT = "results/pilots/pilot_v6"
LOW_CONF, ARM = 0.3, [5, 6, 7, 8, 9, 10]


def conditions():
    c = [("zero", w, "mid", s) for s in (2000, 3000, 4000) for w in ("S", "I", "Sp")]
    c += [("zero", w, "low", 2000) for w in ("S", "I")]
    c += [(op, w, "mid", 2000) for op in ("gapfill", "conflow") for w in ("S", "I", "Sp")]
    return c


def cname(op, w, st, seed):
    return f"{op}_{w}_{st}_m{seed}"


def registry_v6(cfg5):
    base = {m["name"]: m for m in SI.registry(cfg5)}
    want = [n for n in base if n.startswith(("tcn_D_qcsym_none", "tcn_v3_", "compact_v4_",
                                             "tcn_A_none_s", "tcn_A_dropout", "tcn_B_coorddrop",
                                             "tcn_A_both"))]
    models = [base[n] for n in want]
    for w in ("S", "I"):                     # v3 low-strength models
        for s in (0, 1, 2):
            models.append({"name": f"tcn_v3low_{w}_s{s}", "family": "v6", "arch": "tcn",
                           "path": f"cache/pilot_v3_low_{w}_A_none_s{s}.pt", "cell": "A_none",
                           "prep": {"normalize": "image", "velocity": False, "confidence": True}})
    for s in (0, 1, 2):                      # v6-trained mirrored world
        models.append({"name": f"tcn_v6_Sp_w0_s{s}", "family": "v6", "arch": "tcn",
                       "path": f"cache/pilot_v6_Sp_w0_A_none_s{s}.pt", "cell": "A_none",
                       "prep": {"normalize": "image", "velocity": False, "confidence": True}})
    for w in (1, 2):                         # v6-trained mask realizations
        models.append({"name": f"tcn_v6_S_w{w}_s0", "family": "v6", "arch": "tcn",
                       "path": f"cache/pilot_v6_S_w{w}_A_none_s0.pt", "cell": "A_none",
                       "prep": {"normalize": "image", "velocity": False, "confidence": True}})
    return [m for m in models if os.path.exists(m["path"])]


@torch.no_grad()
def score(net, rows, prep, masks, mode, device, workers, batch=256):
    ds = D.SegmentDataset(rows, train=False, planted=masks, planted_mode=mode, **prep)
    out = []
    for x, _ in DataLoader(ds, batch_size=batch, num_workers=workers):
        out.append(F.softmax(net(x.to(device)), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


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


def oracles(te_rows, va_rows, conds):
    """Quality-only classifiers fit on the host validation children of the same world."""
    res = {}
    yv = np.array([r["binary_label"] for r in va_rows])
    yt = np.array([r["binary_label"] for r in te_rows])
    for op, w, st, seed in conds:
        if seed != 2000:
            continue
        mt = P.build_masks(te_rows, w, st, seed)[0]
        mv = P.build_masks(va_rows, w, st, seed + 1)[0]
        xv, xt = tracking_stats(va_rows, mv, op), tracking_stats(te_rows, mt, op)
        sc = StandardScaler().fit(xv)
        lr = LogisticRegression(max_iter=2000).fit(sc.transform(xv), yv)
        gb = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(xv, yv)
        res[cname(op, w, st, seed)] = {
            "logreg_auc": M.auc(lr.predict_proba(sc.transform(xt))[:, 1], yt),
            "gb_auc": M.auc(gb.predict_proba(xt)[:, 1], yt)}
        print("oracle", cname(op, w, st, seed), res[cname(op, w, st, seed)], flush=True)
    xv, xt = tracking_stats(va_rows, None, "zero"), tracking_stats(te_rows, None, "zero")
    gb = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(xv, yv)
    sc = StandardScaler().fit(xv)
    lr = LogisticRegression(max_iter=2000).fit(sc.transform(xv), yv)
    res["clean"] = {"logreg_auc": M.auc(lr.predict_proba(sc.transform(xt))[:, 1], yt),
                    "gb_auc": M.auc(gb.predict_proba(xt)[:, 1], yt)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--family", default=None)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--skip-oracles", action="store_true")
    args = ap.parse_args()
    cfg5 = yaml.safe_load(open(SI.PROTO))
    device = "cpu" if args.smoke else "cuda"
    rows = D.load_index()
    te = [r for r in rows if r["split"] == "test" and D.passes_qc(r, V1["qc_filter"])]
    _, va, _ = D.subject_level_val_split(rows, 0.2, seed=0)
    va = [r for r in va if D.passes_qc(r, V1["qc_filter"])]
    if args.smoke:
        te, va = te[::12], va[::12]
    os.makedirs(f"{OUT}/scores", exist_ok=True)
    conds = conditions()
    masks = {(w, st, seed): P.build_masks(te, w, st, seed)[0]
             for (_, w, st, seed) in {(c[0], c[1], c[2], c[3]) for c in conds}}
    print(f"masks built, {len(te)} test rows", flush=True)

    if not args.skip_oracles:
        res = oracles(te, va, conds)
        json.dump(res, open(f"{OUT}/oracles{'_smoke' if args.smoke else ''}.json", "w"), indent=1)

    models = registry_v6(cfg5)
    if args.family:
        models = [m for m in models if m["family"] == args.family]
    if args.smoke:
        models = models[:2]
    t0 = time.time()
    for m in models:
        path = f"{OUT}/scores/{m['name']}{'_smoke' if args.smoke else ''}.csv"
        if os.path.exists(path):
            continue
        net = SI.build(m).to(device)
        cols = {"clean": score(net, te, m["prep"], None, "zero", device, args.workers)}
        for op, w, st, seed in conds:
            cols[cname(op, w, st, seed)] = score(net, te, m["prep"], masks[(w, st, seed)], op,
                                                 device, args.workers)
        with open(path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["identifier", "binary_label"] + list(cols))
            for i, r in enumerate(te):
                wr.writerow([r["identifier"], r["binary_label"]] + [f"{cols[k][i]:.6f}" for k in cols])
        y = [r["binary_label"] for r in te]
        print(f"{m['name']:26s} clean {M.auc(cols['clean'], y):.4f} "
              f"zero S/I {M.auc(cols['zero_S_mid_m2000'], y):.4f}/{M.auc(cols['zero_I_mid_m2000'], y):.4f} "
              f"gapfill S/I {M.auc(cols['gapfill_S_mid_m2000'], y):.4f}/{M.auc(cols['gapfill_I_mid_m2000'], y):.4f} "
              f"conflow S/I {M.auc(cols['conflow_S_mid_m2000'], y):.4f}/{M.auc(cols['conflow_I_mid_m2000'], y):.4f} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    if not args.smoke:
        with open("results/gpu_ledger.csv", "a", newline="") as f:
            csv.writer(f).writerow([time.strftime("%Y-%m-%d"), "pilot_v6/score_v6",
                                    f"{(time.time() - t0) / 3600:.3f}"])


if __name__ == "__main__":
    main()
