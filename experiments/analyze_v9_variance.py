"""Variance of the shortcut effect by training seed, test-mask realization and subject.

For every A_none planted cell of the v9 hosts (pooled out-of-fold scores) and of the ASDPose v7/v8
host, SE, R(f_S) and R(f_I) are computed on the 3 seeds x 3 test-mask grid and split into
between-seed, between-mask and residual parts by a two-way additive decomposition. The subject
component is the subject-clustered bootstrap SD of the pooled value (mask seed 2000); LOSO
recomputes the pooled point estimate with each subject left out.
Writes results/v9/analysis/variance.csv and variance.md.
    conda run -n torch python experiments/analyze_v9_variance.py [--n-boot 2000]
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safe_motion import metrics as M            # noqa: E402
from analyze_v7 import logit, RowSet            # noqa: E402
from analyze_v9 import discover, pool, MASK_SEEDS   # noqa: E402

SEEDS = (0, 1, 2)
V7_RE = re.compile(r"^(v[78]|v13)_(S|I)_w0_(runs|whole|frozen|armruns)_(mid|low)_A_none_s(\d)\.csv$")


def decompose(grid):
    """Two-way additive decomposition of a seeds x masks array of one statistic."""
    g = np.asarray(grid, float)
    ns, nm = g.shape
    gm = g.mean()
    row, col = g.mean(axis=1), g.mean(axis=0)
    resid = g - row[:, None] - col[None, :] + gm
    ss_seed, ss_mask = nm * ((row - gm) ** 2).sum(), ns * ((col - gm) ** 2).sum()
    ss_res, ss_tot = (resid ** 2).sum(), ((g - gm) ** 2).sum()
    share = (lambda x: float(x / ss_tot) if ss_tot > 0 else float("nan"))
    return {"grid_mean": float(gm),
            "sd_seed": float(row.std(ddof=1)) if ns > 1 else float("nan"),
            "sd_mask": float(col.std(ddof=1)) if nm > 1 else float("nan"),
            "sd_resid": float(np.sqrt(ss_res / max(1, (ns - 1) * (nm - 1)))),
            "share_seed": share(ss_seed), "share_mask": share(ss_mask), "share_resid": share(ss_res),
            "range_seed": float(row.max() - row.min()), "range_mask": float(col.max() - col.min())}


def r_arrays(aucs_S, aucs_I):
    return logit(aucs_S) - logit(aucs_I)


class Cell:
    """Scores of one (host, mechanism, strength) A_none cell by world, seed, arm and mask seed,
    aligned to one RowSet."""

    def __init__(self, host, mech, st, rs):
        self.host, self.mech, self.st, self.rs = host, mech, st, rs
        self.scores = {}          # (world, seed, arm, mask) -> score array in rs order

    def add(self, world, seed, cols, order):
        for ms in MASK_SEEDS:
            for arm in ("S", "I"):
                c = f"{self.mech}_{arm}_{self.st}_m{ms}"
                if c in cols:
                    self.scores[(world, seed, arm, ms)] = cols[c][order]

    def complete(self):
        return all((w, s, a, m) in self.scores for w in ("S", "I") for s in SEEDS
                   for a in ("S", "I") for m in MASK_SEEDS)

    def point_R(self, world, seed, ms, keep=None):
        y = self.rs.y if keep is None else self.rs.y[keep]
        sS, sI = self.scores[(world, seed, "S", ms)], self.scores[(world, seed, "I", ms)]
        if keep is not None:
            sS, sI = sS[keep], sI[keep]
        return float(logit(M.auc(sS, y)) - logit(M.auc(sI, y)))

    def boot_R(self, world, ms):
        """Seed-averaged R over the shared resamples (index 0 = full set)."""
        per = [r_arrays(self.rs.aucs(self.scores[(world, s, "S", ms)]),
                        self.rs.aucs(self.scores[(world, s, "I", ms)])) for s in SEEDS]
        return np.mean(per, axis=0)


def analyse_cell(cell, n_subjects_label):
    rows = []
    grids = {q: np.zeros((len(SEEDS), len(MASK_SEEDS))) for q in ("SE", "R_S", "R_I")}
    for i, s in enumerate(SEEDS):
        for j, ms in enumerate(MASK_SEEDS):
            rS, rI = cell.point_R("S", s, ms), cell.point_R("I", s, ms)
            grids["R_S"][i, j], grids["R_I"][i, j], grids["SE"][i, j] = rS, rI, rS - rI
    bS, bI = cell.boot_R("S", 2000), cell.boot_R("I", 2000)
    boots = {"SE": bS - bI, "R_S": bS, "R_I": bI}
    subjects = np.unique(cell.rs.child)
    for q in ("SE", "R_S", "R_I"):
        d = decompose(grids[q])
        b = boots[q]
        ci = M.ci(b[1:])
        # leave-one-subject-out on the pooled (seed-averaged, mask 2000) point estimate
        loso = {}
        for subj in subjects:
            keep = cell.rs.child != subj
            if q == "SE":
                v = np.mean([cell.point_R("S", s, 2000, keep) - cell.point_R("I", s, 2000, keep) for s in SEEDS])
            else:
                v = np.mean([cell.point_R(q[2], s, 2000, keep) for s in SEEDS])
            loso[str(subj)] = float(v)
        lv = np.array(list(loso.values()))
        shifts = {k: v - b[0] for k, v in loso.items()}
        worst = max(shifts, key=lambda k: abs(shifts[k]))
        rows.append({"host": cell.host, "mechanism": cell.mech, "strength": cell.st, "quantity": q,
                     "point_m2000": float(b[0]), "ci_lo": ci[0], "ci_hi": ci[1],
                     "sd_subject_boot": float(np.std(b[1:])),
                     **d, "n_subjects": int(len(subjects)), "subject_label": n_subjects_label,
                     "loso_min": float(lv.min()), "loso_max": float(lv.max()),
                     "loso_range": float(lv.max() - lv.min()), "loso_worst_subject": worst,
                     "loso_worst_shift": float(shifts[worst]),
                     "grid_json": json.dumps({f"s{s}_m{m}": round(float(grids[q][i, j]), 4)
                                              for i, s in enumerate(SEEDS) for j, m in enumerate(MASK_SEEDS)})})
    return rows


def v9_cells(base, n_boot):
    models = discover(base)
    rowsets, cells = {}, {}
    for key, dirs in sorted(models.items(), key=str):
        host, mech, st, world, cell, seed, perm = key
        if cell != "A_none" or perm is not None or world not in ("S", "I") or mech is None:
            continue
        meta, cols = pool(dirs)
        if host not in rowsets:
            rowsets[host] = RowSet(meta["identifier"], [int(v) for v in meta["label"]], meta["subject_id"], n_boot)
        rs = rowsets[host]
        if set(meta["identifier"]) != set(rs.ids):
            continue
        order = np.array([rs.pos[i] for i in meta["identifier"]])
        ck = (host, mech, st)
        if ck not in cells:
            cells[ck] = Cell(host, mech, st, rs)
        cells[ck].add(world, seed, cols, order)
    return [c for c in cells.values() if c.complete()]


def v7_cells(scores_dir, n_boot, v13_dir="results/v13/scores"):
    """ASDPose no-remedy cells. Arm confidence comes from the corrected-validity-rule cell (protocol
    v13) when its scores exist; the original-rule v8 arm-confidence cell is then not used."""
    files = sorted(glob.glob(f"{scores_dir}/v[78]_[SI]_w0_*_A_none_s?.csv"))
    v13 = sorted(glob.glob(f"{v13_dir}/v13_[SI]_w0_armruns_mid_A_none_s?.csv"))
    if v13:
        files = [f for f in files if not os.path.basename(f).startswith("v8_") or "_armruns_" not in f] + v13
    rs, cells = None, {}
    for f in files:
        m = V7_RE.match(os.path.basename(f))
        if not m:
            continue
        _, world, mech, st, seed = m.groups()
        rd = list(csv.DictReader(open(f)))
        ids = [r["identifier"] for r in rd]
        if rs is None:
            rs = RowSet(ids, [int(r["label"]) for r in rd], [r["child_id"] for r in rd], n_boot)
        if set(ids) != set(rs.ids):
            continue
        order = np.array([rs.pos[i] for i in ids])
        cols = {k: np.array([float(r[k]) for r in rd]) for k in rd[0] if k.endswith(("m2000", "m3000", "m4000"))}
        ck = ("asdpose_host", mech, st)
        if ck not in cells:
            cells[ck] = Cell("asdpose_host", mech, st, rs)
        cells[ck].add(world, int(seed), cols, order)
    return [c for c in cells.values() if c.complete()]


def write_md(rows, path):
    hosts = sorted({r["host"] for r in rows}, key=lambda h: (h != "asdpose_host", h))
    lines = ["# Variance of SE, R(f_S) and R(f_I) by seed, test mask and subject", "",
             "Generated by `experiments/analyze_v9_variance.py`. Point: pooled value at mask seed 2000, "
             "seeds averaged (the v9 and v7 analysis value), with its subject-clustered bootstrap interval. "
             "sd columns: subject bootstrap of the pooled value; between training seeds; between test-mask "
             "seeds; residual seed x mask, from the two-way decomposition of the 3 x 3 grid. Shares are of the "
             "grid variance. LOSO: range of the pooled point estimate when one subject is left out.", ""]
    for q in ("SE", "R_S", "R_I"):
        lines += [f"## {q}", "",
                  "| host | mechanism | strength | point [95% CI] | sd subject | sd seed | sd mask | sd resid | share seed / mask / resid | range over masks | LOSO range (n subjects) | worst subject shift |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for h in hosts:
            for r in [x for x in rows if x["host"] == h and x["quantity"] == q]:
                lines.append(f"| {h} | {r['mechanism']} | {r['strength']} | {r['point_m2000']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] "
                             f"| {r['sd_subject_boot']:.3f} | {r['sd_seed']:.3f} | {r['sd_mask']:.3f} | {r['sd_resid']:.3f} "
                             f"| {r['share_seed']:.2f} / {r['share_mask']:.2f} / {r['share_resid']:.2f} | {r['range_mask']:.3f} "
                             f"| {r['loso_range']:.3f} ({r['n_subjects']}) | {r['loso_worst_shift']:+.3f} (subject {r['loso_worst_subject']}) |")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/v9")
    ap.add_argument("--v7-scores", default="results/v7/scores")
    ap.add_argument("--v13-scores", default="results/v13/scores", help="corrected-rule arm-confidence cell")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"{args.base}/analysis"
    os.makedirs(out, exist_ok=True)
    cells = v9_cells(args.base, args.n_boot)
    if os.path.isdir(args.v7_scores):
        cells += v7_cells(args.v7_scores, args.n_boot, args.v13_scores)
    rows = []
    for c in cells:
        rows += analyse_cell(c, "subject IDs" if c.host.startswith("asdpose") else ("children" if c.host.startswith("mmasd") else "subjects"))
        print(f"{c.host} {c.mech} {c.st}: SE {rows[-3]['point_m2000']:+.3f} sd seed {rows[-3]['sd_seed']:.3f} "
              f"mask {rows[-3]['sd_mask']:.3f} subject {rows[-3]['sd_subject_boot']:.3f} LOSO range {rows[-3]['loso_range']:.3f}",
              flush=True)
    with open(f"{out}/variance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    write_md(rows, f"{out}/variance.md")
    print("wrote", f"{out}/variance.csv", f"{out}/variance.md")


if __name__ == "__main__":
    main()
