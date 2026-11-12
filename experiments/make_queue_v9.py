"""Write configs/runs/queue_v9.txt: one line per fit of protocol v9 (configs/protocols/replication_v9.yaml).

Each line is "<out_dir> <train_v9.py arguments>" (run with experiments/train_runs.sh and
experiments/train_v9.py as the trainer). Order: gate cells first (A_none World I
runs mid, all folds, seed 0), then the planted cells by mechanism, then real cells, then the
permutation null. Prints the fit count per host.
    python experiments/make_queue_v9.py [--hosts mmasd_maracas utd_mhad_body_vs_arm] [--n-perm 20]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion.data import hosts as H

SEEDS = (0, 1, 2)
PLAN = {                      # mechanism_strength -> cells (replication_v9.yaml, cells)
    "runs_mid": ("A_none", "A_dropout", "B_coorddrop"),
    "runs_low": ("A_none",),
    "whole_mid": ("A_none", "A_dropout"),
    "frozen_mid": ("A_none", "A_dropout"),
    "armruns_mid": ("A_none", "A_dropout"),
}
REAL_CELLS = ("A_none", "A_dropout")


def n_folds(host):
    kind, k = H.HOSTS[host]["folds"]
    return 8 if kind == "loso" else k


def lines_for(host, n_perm, common):
    out = []
    nf = n_folds(host)
    mechs = H.HOSTS[host]["mechanisms"]

    def line(mech, st, world, cell, seed=None, perm=None, fold=0):
        who = f"perm{perm}" if perm is not None else f"seed{seed}"
        od = f"results/v9/{host}/{mech}_{st}/{world}/{cell}/{who}/fold{fold}"
        a = f"--host {host} --mechanism {mech} --strength {st} --world {world} --cell {cell} --fold {fold}"
        a += f" --perm {perm}" if perm is not None else f" --seed {seed}"
        return f"{od} {a} {common}".rstrip()

    for f in range(nf):                                   # gate first
        out.append(line("runs", "mid", "I", "A_none", seed=0, fold=f))
    for key, cells in PLAN.items():
        mech, st = key.split("_")
        if mech not in mechs:
            continue
        for cell in cells:
            for world in ("S", "I"):
                for seed in SEEDS:
                    for f in range(nf):
                        out.append(line(mech, st, world, cell, seed=seed, fold=f))
    if H.HOSTS[host]["dataset"] == "mmasd":
        for cell in REAL_CELLS:
            for seed in SEEDS:
                for f in range(nf):
                    out.append(f"results/v9/{host}/real/none/{cell}/seed{seed}/fold{f} "
                               f"--host {host} --cell {cell} --seed {seed} --fold {f} {common}".rstrip())
    for mech in mechs:
        for k in range(n_perm):
            for world in ("S", "I"):
                for f in range(nf):
                    out.append(line(mech, "mid", world, "A_none", perm=k, fold=f))
    seen, uniq = set(), []
    for l in out:
        if l.split()[0] not in seen:
            seen.add(l.split()[0])
            uniq.append(l)
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hosts", nargs="+", default=["mmasd_maracas", "utd_mhad_body_vs_arm"])
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--common", default="--amp --workers 0", help="arguments appended to every line")
    ap.add_argument("--out", default="configs/runs/queue_v9.txt")
    args = ap.parse_args()
    lines, total = [], {}
    for h in args.hosts:
        ls = lines_for(h, args.n_perm, args.common)
        total[h] = len(ls)
        lines += ls
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    for h, n in total.items():
        print(f"{h}: {n} fits")
    print(f"total {len(lines)} fits -> {args.out}")


if __name__ == "__main__":
    main()
