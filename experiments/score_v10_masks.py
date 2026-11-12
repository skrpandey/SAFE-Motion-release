"""Protocol v10 part C: rescore every A_none planted v9 checkpoint under ten test-mask seeds.

For the hosts mmasd_chest_squat and utd_mhad_body_vs_arm, each fold's test subjects are scored clean
and under the checkpoint's own operator and strength, worlds S and I, mask seeds 2000 to 11000.
Seeds 2000/3000/4000 repeat v9, so those columns are checked against the v9 score files
(<out>/reproduction.json; asserted for the first checkpoint). Dataset and model follow train_v9.py.
Output: <out>/<host>/<mech>_<st>/<world>/A_none/seed<k>/fold<f>/test_scores.csv.
    conda run -n torch python experiments/score_v10_masks.py --ckpt-dir results/fits/v9/cache
    ... --hosts utd_mhad_body_vs_arm --limit 1 --mask-seeds 2000 --device cpu   # check
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
from safe_motion import metrics as M                       # noqa: E402
from safe_motion import planted as P                       # noqa: E402
from safe_motion.data import hosts as H                    # noqa: E402
from safe_motion.models.ctrgcn import CompactCTRGCN        # noqa: E402
import train_v9 as T9                                      # noqa: E402

HOSTS = ("mmasd_chest_squat", "utd_mhad_body_vs_arm")
MASK_SEEDS = tuple(range(2000, 12000, 1000))
NAME = re.compile(r"^v9_(?P<host>.+?)_(?P<mech>runs|whole|frozen|armruns)_(?P<st>mid|low)_(?P<world>S|I)"
                  r"_A_none_seed(?P<seed>\d)_fold(?P<fold>\d+)\.pt$")


def checkpoints(ckpt_dir, hosts):
    out = []
    for p in sorted(glob.glob(f"{ckpt_dir}/v9_*_A_none_seed*_fold*.pt")):
        m = NAME.match(os.path.basename(p))
        if m and m.group("host") in hosts:
            g = m.groupdict()
            g.update(path=p, seed=int(g["seed"]), fold=int(g["fold"]))
            out.append(g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="results/fits/v9/cache")
    ap.add_argument("--v9-base", default="results/v9", help="v9 outputs (folds.json, reproduction check)")
    ap.add_argument("--out", default="results/v10/masks")
    ap.add_argument("--hosts", nargs="+", default=list(HOSTS))
    ap.add_argument("--mask-seeds", nargs="+", type=int, default=list(MASK_SEEDS))
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N checkpoints")
    ap.add_argument("--tol", type=float, default=5e-3, help="reproduction tolerance on the probability scale (GPU vs CPU rounding; 1e-3 to 2e-3 observed)")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cks = checkpoints(args.ckpt_dir, args.hosts)
    if args.limit:
        cks = cks[: args.limit]
    print(f"{len(cks)} checkpoints, device {device}, mask seeds {args.mask_seeds}", flush=True)
    rows_by_host, folds_by_host = {}, {}
    repro_path = f"{args.out}/reproduction.json"
    repro = json.load(open(repro_path)) if os.path.exists(repro_path) else {}
    cell = T9.CELLS["A_none"]
    t0, first = time.time(), True
    for ck in cks:
        host = ck["host"]
        od = f"{args.out}/{host}/{ck['mech']}_{ck['st']}/{ck['world']}/A_none/seed{ck['seed']}/fold{ck['fold']}"
        if os.path.exists(f"{od}/test_scores.csv"):
            continue
        if host not in rows_by_host:
            rows_by_host[host] = H.load_host(host)
            folds_by_host[host] = json.load(open(f"{args.v9_base}/{host}/folds.json"))["folds"]
        rows, layout = rows_by_host[host], H.LAYOUT[H.HOSTS[host]["dataset"]]
        hostcfg = H.HOSTS[host]
        te = [r for r in rows if r["subject"] in set(folds_by_host[host][ck["fold"]])]
        sd = torch.load(ck["path"], map_location="cpu", weights_only=False)
        net = CompactCTRGCN(c_in=3, num_joints=layout["n_joints"])
        net.load_state_dict(sd["state_dict"], strict=True)
        net = net.to(device).eval()
        mode = P.MECHANISM_MODE[ck["mech"]]
        cols = {"clean": T9.run_eval(net, te, cell, hostcfg, layout, device, args.workers)[0]}
        for ms in args.mask_seeds:
            for w in ("S", "I"):
                masks = H.host_masks(te, w, ck["st"], ms, ck["mech"], layout)
                cols[f"{ck['mech']}_{w}_{ck['st']}_m{ms}"] = T9.run_eval(
                    net, te, cell, hostcfg, layout, device, args.workers, planted=masks, mode=mode)[0]
        os.makedirs(od, exist_ok=True)
        with open(f"{od}/test_scores.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["identifier", "subject_id", "label", "n_frames", "qc_pass"] + H.Q_COLS + list(cols))
            for i, r in enumerate(te):
                wr.writerow([r["identifier"], r["subject"], r["binary_label"], r["n_frames"], r["qc_pass"]]
                            + [f"{r[q]:.4f}" for q in H.Q_COLS] + [f"{cols[c][i]:.6f}" for c in cols])
        # reproduction of the v9 columns
        v9f = f"{args.v9_base}/{host}/{ck['mech']}_{ck['st']}/{ck['world']}/A_none/seed{ck['seed']}/fold{ck['fold']}/test_scores.csv"
        if os.path.exists(v9f):
            old = {r["identifier"]: r for r in csv.DictReader(open(v9f))}
            diffs = []
            for c in cols:
                if c == "clean" or c in next(iter(old.values())):
                    new = np.array([cols[c][i] for i in range(len(te))])
                    ref = np.array([float(old[r["identifier"]][c]) for r in te])
                    diffs.append(float(np.abs(new - ref).max()))
            repro[os.path.basename(ck["path"])] = max(diffs) if diffs else None
            if first and diffs:
                assert max(diffs) < args.tol, f"v9 columns not reproduced: max diff {max(diffs):.2e}"
            first = False
            json.dump(repro, open(repro_path, "w"), indent=1)
        y = [r["binary_label"] for r in te]
        print(f"{os.path.basename(ck['path'])}: clean {M.auc(cols['clean'], y):.3f} "
              f"repro {repro.get(os.path.basename(ck['path']))} ({(time.time()-t0)/60:.1f} min)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
