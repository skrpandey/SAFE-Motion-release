"""Post hoc sensitivity of the v12 remedy contrasts to seeds and test masks.

Same rows, child resamples and mask-mean convention as analyze_v12.py. Per mechanism and cell:
R of every fitted model, SE and Delta SE vs A_none over all 3 x 3 World S x World I seed
combinations (seeds are separate fits, so there is no natural pairing), per-mask SE, and a
crossed bootstrap that redraws children, the 3 masks and the 3 seeds of each group in every
replicate. With three seeds this is a sensitivity display, not a calibrated interval.
Writes results/v12/analysis/variance_v12.csv.
"""
import csv, itertools, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import metrics as M
from experiments.analyze_v7 import RowSet, load_scores, logit
from experiments.analyze_v7_masks import MASK_SEEDS
CELLS = ("A_none", "C_groupdro", "C_adversarial", "C_augment")
MECH_PROTO = {"runs": "v7", "frozen": "v8"}
NB = 2000
v12, v7 = "results/v12", "results/v7"
reg12 = [m for m in json.load(open(f"{v12}/registry.json")) if os.path.exists(f"{v12}/scores/{m['name']}.csv")]
reg7 = [m for m in json.load(open(f"{v7}/registry.json"))
        if m["cell"] == "A_none" and m["world"] in ("S", "I") and m.get("world_seed", 0) == 0
        and m["strength"] == "mid" and MECH_PROTO.get(m["mechanism"]) == m["proto"]
        and os.path.exists(f"{v7}/scores/{m['name']}.csv")]
files = {m["name"]: f"{v12}/scores/{m['name']}.csv" for m in reg12}
files.update({m["name"]: f"{v7}/scores/{m['name']}.csv" for m in reg7})
loaded = {n: load_scores(p) for n, p in files.items()}
common = set.intersection(*[set(meta["identifier"]) for meta, _ in loaded.values()])
meta0 = next(iter(loaded.values()))[0]
keep = [k for k, i in enumerate(meta0["identifier"]) if i in common]
ids = [meta0["identifier"][k] for k in keep]
rs = RowSet(ids, [int(meta0["label"][k]) for k in keep], [meta0["child_id"][k] for k in keep], NB)
S = {}
for n, (meta, sc) in loaded.items():
    pos = {i: k for k, i in enumerate(meta["identifier"])}
    S[n] = {c: v[[pos[i] for i in ids]] for c, v in sc.items()}
models = reg12 + reg7
rng = np.random.RandomState(20261105)
mask_draw = rng.randint(0, 3, size=(NB, 3))
out = []


def Rm(name, mech):
    """(3 masks, NB+1) logit R arrays of one fitted model."""
    return np.stack([logit(rs.aucs(S[name][f"{mech}_S_mid_m{ms}"])) - logit(rs.aucs(S[name][f"{mech}_I_mid_m{ms}"]))
                     for ms in MASK_SEEDS])


for mech in ("runs", "frozen"):
    R = {}
    for cell in CELLS:
        for w in ("S", "I"):
            g = sorted([m for m in models if m["mechanism"] == mech and m["cell"] == cell and m["world"] == w], key=lambda m: m["name"])
            R[cell, w] = [Rm(m["name"], mech) for m in g]
            for m, r in zip(g, R[cell, w]):
                out.append({"mechanism": mech, "cell": cell, "quantity": f"R model {m['name']}",
                            "point": round(float(r[:, 0].mean()), 4), "per_mask": "; ".join(f"{v:+.4f}" for v in r[:, 0])})
    seed_draw = {k: rng.randint(0, len(v), size=(NB, 3)) for k, v in R.items()}

    def se_point_combos(cell):
        return [float(a[:, 0].mean() - b[:, 0].mean()) for a, b in itertools.product(R[cell, "S"], R[cell, "I"])]

    def se_crossed(cell):
        vals = np.empty(NB)
        for b in range(NB):
            md = mask_draw[b]
            rS = np.mean([R[cell, "S"][k][md, b + 1].mean() for k in seed_draw[cell, "S"][b]])
            rI = np.mean([R[cell, "I"][k][md, b + 1].mean() for k in seed_draw[cell, "I"][b]])
            vals[b] = rS - rI
        return vals

    def se_cond(cell):                      # analyze_v12 convention: seeds and masks averaged inside resamples
        return np.mean([r.mean(0) for r in R[cell, "S"]], 0) - np.mean([r.mean(0) for r in R[cell, "I"]], 0)

    base_cond, base_x, base_c = se_cond("A_none"), se_crossed("A_none"), se_point_combos("A_none")
    for cell in CELLS:
        cond, x, comb = se_cond(cell), se_crossed(cell), se_point_combos(cell)
        pm = [float(np.mean([r[j, 0] for r in R[cell, "S"]]) - np.mean([r[j, 0] for r in R[cell, "I"]])) for j in range(3)]
        lo, hi = M.ci(cond[1:]); xlo, xhi = np.percentile(x, [2.5, 97.5])
        out.append({"mechanism": mech, "cell": cell, "quantity": "SE", "point": round(float(cond[0]), 4),
                    "cond_ci": f"[{lo:+.4f}, {hi:+.4f}]", "crossed_ci": f"[{xlo:+.4f}, {xhi:+.4f}]",
                    "seed_combo_min": round(min(comb), 4), "seed_combo_max": round(max(comb), 4), "seed_combo_sd": round(float(np.std(comb, ddof=1)), 4),
                    "per_mask": "; ".join(f"{v:+.4f}" for v in pm), "mask_range": round(max(pm) - min(pm), 4),
                    "sd_boot_cond": round(float(np.std(cond[1:])), 4), "sd_crossed": round(float(np.std(x)), 4)})
        if cell == "A_none":
            continue
        d, dx = cond - base_cond, x - base_x
        dc = [a - b for a, b in itertools.product(comb, base_c)]
        pmb = [float(np.mean([r[j, 0] for r in R["A_none", "S"]]) - np.mean([r[j, 0] for r in R["A_none", "I"]])) for j in range(3)]
        dpm = [a - b for a, b in zip(pm, pmb)]
        lo, hi = M.ci(d[1:]); xlo, xhi = np.percentile(dx, [2.5, 97.5])
        out.append({"mechanism": mech, "cell": cell, "quantity": "Delta SE vs A_none", "point": round(float(d[0]), 4),
                    "cond_ci": f"[{lo:+.4f}, {hi:+.4f}]", "crossed_ci": f"[{xlo:+.4f}, {xhi:+.4f}]",
                    "seed_combo_min": round(min(dc), 4), "seed_combo_max": round(max(dc), 4), "seed_combo_sd": round(float(np.std(dc, ddof=1)), 4),
                    "per_mask": "; ".join(f"{v:+.4f}" for v in dpm), "mask_range": round(max(dpm) - min(dpm), 4),
                    "sd_boot_cond": round(float(np.std(d[1:])), 4), "sd_crossed": round(float(np.std(dx)), 4)})
        print(out[-1], flush=True)
keys = []
for r in out:
    keys += [k for k in r if k not in keys]
with open(f"{v12}/analysis/variance_v12.csv", "w", newline="") as f:
    f.write("# post hoc sensitivity; crossed_ci: one draw of children, masks and fitted seeds per replicate\n")
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out)
