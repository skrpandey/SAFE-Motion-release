"""Post hoc seed-and-mask sensitivity intervals for the no-remedy planted cells of Table I.

Cells: v7 missing runs (mid, low), v7 whole-joint absence, v8 frozen frames, v13 arm confidence,
each with three fitted seeds per world and three test masks. Reports the conditional interval of
the primary analysis (identifier resample only) and a crossed interval that also redraws test masks
and fitted seeds per replicate. With three seeds the crossed interval is a sensitivity display, not
a calibrated interval over training randomness. Also MWI over the nine S x I seed combinations.
Writes <root>/results/v7/analysis/variance_v7.csv. Run from inside a release tree.
"""
import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import metrics as M                                  # noqa: E402
from safe_motion.paths import guard_output                            # noqa: E402
from experiments.analyze_v7 import RowSet, load_scores, logit          # noqa: E402
from experiments.analyze_v7_masks import MASK_SEEDS                    # noqa: E402

CELLS = [("v7", "runs", "mid"), ("v7", "runs", "low"), ("v7", "whole", "mid"), ("v8", "frozen", "mid"),
         ("v13", "armruns", "mid")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="results/v7/analysis/variance_v7.csv")
    args = ap.parse_args()
    guard_output(args.out)
    NB = args.n_boot
    reg = []
    for proto_dir in ("results/v7", "results/v13"):
        for m in json.load(open(f"{proto_dir}/registry.json")):
            if (m["cell"] == "A_none" and m["world"] in ("S", "I") and m.get("world_seed", 0) == 0
                    and (m["proto"], m["mechanism"], m["strength"]) in CELLS
                    and os.path.exists(f"{proto_dir}/scores/{m['name']}.csv")):
                reg.append(dict(m, path=f"{proto_dir}/scores/{m['name']}.csv"))
    loaded = {m["name"]: load_scores(m["path"]) for m in reg}
    common = set.intersection(*[set(meta["identifier"]) for meta, _ in loaded.values()])
    meta0 = next(iter(loaded.values()))[0]
    keep = [k for k, i in enumerate(meta0["identifier"]) if i in common]
    ids = [meta0["identifier"][k] for k in keep]
    rs = RowSet(ids, [int(meta0["label"][k]) for k in keep], [meta0["child_id"][k] for k in keep], NB)
    S = {}
    for n, (meta, sc) in loaded.items():
        pos = {i: k for k, i in enumerate(meta["identifier"])}
        order = [pos[i] for i in ids]                      # file position of each RowSet row
        lab = np.array([int(meta["label"][k]) for k in order])
        assert np.array_equal(lab, rs.y), n
        S[n] = {c: v[order] for c, v in sc.items()}
    rng = np.random.RandomState(20261109)
    mask_draw = rng.randint(0, 3, size=(NB, 3))
    out = []

    def Rm(name, mech, st, raw=False):
        """(3 masks, NB+1) R arrays of one fitted model, logit or raw scale."""
        f = (lambda a: a) if raw else logit
        return np.stack([f(rs.aucs(S[name][f"{mech}_S_{st}_m{ms}"])) - f(rs.aucs(S[name][f"{mech}_I_{st}_m{ms}"]))
                         for ms in MASK_SEEDS])

    for proto, mech, st in CELLS:
        R = {}
        for w in ("S", "I"):
            g = sorted([m for m in reg if (m["proto"], m["mechanism"], m["strength"], m["world"]) == (proto, mech, st, w)],
                       key=lambda m: m["name"])
            assert len(g) == 3, (proto, mech, st, w, len(g))
            R[w] = [Rm(m["name"], mech, st) for m in g]
            R[w + "raw"] = [Rm(m["name"], mech, st, raw=True) for m in g]
        seed_draw = {w: rng.randint(0, 3, size=(NB, 3)) for w in ("S", "I")}

        def crossed(suffix=""):
            vals = np.empty(NB)
            for b in range(NB):
                md = mask_draw[b]
                rS = np.mean([R["S" + suffix][k][md, b + 1].mean() for k in seed_draw["S"][b]])
                rI = np.mean([R["I" + suffix][k][md, b + 1].mean() for k in seed_draw["I"][b]])
                vals[b] = rS - rI
            return vals

        def cond(suffix=""):
            return np.mean([r.mean(0) for r in R["S" + suffix]], 0) - np.mean([r.mean(0) for r in R["I" + suffix]], 0)

        c, craw, x, xraw = cond(), cond("raw"), crossed(), crossed("raw")
        combos = [float(a[:, 0].mean() - b[:, 0].mean()) for a, b in itertools.product(R["S"], R["I"])]
        pm = [float(np.mean([r[j, 0] for r in R["S"]]) - np.mean([r[j, 0] for r in R["I"]])) for j in range(3)]
        lo, hi = M.ci(c[1:]); rlo, rhi = M.ci(craw[1:])
        xlo, xhi = np.percentile(x, [2.5, 97.5]); xrlo, xrhi = np.percentile(xraw, [2.5, 97.5])
        out.append({"proto": proto, "mechanism": mech, "strength": st, "quantity": "SE",
                    "point": round(float(c[0]), 4), "cond_ci": f"[{lo:+.4f}, {hi:+.4f}]",
                    "crossed_ci": f"[{xlo:+.4f}, {xhi:+.4f}]",
                    "raw_point": round(float(craw[0]), 4), "raw_cond_ci": f"[{rlo:+.4f}, {rhi:+.4f}]",
                    "raw_crossed_ci": f"[{xrlo:+.4f}, {xrhi:+.4f}]",
                    "seed_combo_min": round(min(combos), 4), "seed_combo_max": round(max(combos), 4),
                    "seed_combo_sd_ddof1": round(float(np.std(combos, ddof=1)), 4),
                    "per_mask": "; ".join(f"{v:+.4f}" for v in pm), "mask_range": round(max(pm) - min(pm), 4),
                    "sd_boot_cond_ddof0": round(float(np.std(c[1:])), 4), "sd_crossed_ddof0": round(float(np.std(x)), 4),
                    "n_boot": NB, "n_masks": 3, "n_seeds_per_world": 3})
        print(out[-1], flush=True)
    with open(args.out, "w", newline="") as f:
        f.write("# post hoc sensitivity; crossed_ci: one draw of identifiers, test masks and fitted seeds per replicate; "
                "three fitted seeds, so a sensitivity display, not a calibrated interval over training randomness\n")
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
