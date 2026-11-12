"""Write configs/runs/queue_v10.txt (training lines of protocol v10, parts A and B) and
configs/runs/v10_v9_checkpoints.txt (the v9 A_none checkpoints part C rescans).

Each training line is "<out_dir> <train_asdpose.py arguments>" (run with experiments/train_runs.sh);
scoring is score_v10.py and score_v10_masks.py.
    python experiments/make_queue_v10.py [--ckpt-dir results/fits/v9/cache]
"""

import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_v10_masks import HOSTS, NAME   # noqa: E402

SEEDS = (0, 1, 2)
COMMON = "--proto v10 --arch compact --amp --batch-size 128 --epochs 25 --workers 6"
REAL_CELLS = ("A_none", "A_velocity", "A_dropout", "A_both", "B_coorddrop")


def lines():
    out = []
    for mech in ("runs", "whole"):                              # part A: quality-blind control
        for w in ("S", "I"):
            for k in SEEDS:
                out.append(f"results/v10/{w}_w0_{mech}_mid/A_blind/seed{k} {COMMON} --cell A_blind "
                           f"--world {w} --mechanism {mech} --strength mid --seed {k}")
    for cell in REAL_CELLS:                                     # part B: real-data World I
        for k in SEEDS:
            out.append(f"results/v10/real_I/{cell}/seed{k} {COMMON} --cell {cell} "
                       f"--real-world I --mechanism runs --strength mid --seed {k}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="results/fits/v9/cache")
    ap.add_argument("--out", default="configs/runs/queue_v10.txt")
    ap.add_argument("--ckpt-list", default="configs/runs/v10_v9_checkpoints.txt")
    args = ap.parse_args()
    ls = lines()
    with open(args.out, "w") as f:
        f.write("\n".join(ls) + "\n")
    names = sorted(os.path.basename(p) for p in glob.glob(f"{args.ckpt_dir}/v9_*_A_none_seed*_fold*.pt")
                   if (m := NAME.match(os.path.basename(p))) and m.group("host") in HOSTS)
    with open(args.ckpt_list, "w") as f:
        f.write("\n".join(names) + "\n")
    print(f"{len(ls)} training lines -> {args.out}; {len(names)} v9 checkpoints -> {args.ckpt_list}")


if __name__ == "__main__":
    main()
