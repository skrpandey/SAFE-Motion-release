"""Post hoc permutation reliance on the confidence channel, from the score files written by
experiments/score_conf_reliance.py.

Reliance = logit AUROC(clean) minus the mean logit AUROC over five permutations of the confidence
channel across segments (also the drop with the channel set to a constant). Seeds are averaged inside
2,000 identifier-clustered resamples shared across models. Reports f_S, f_I and f_S - f_I per
protocol and cell, next to the MWI of the same models.
Writes results/v15conf/analysis/conf_reliance.csv and paper/tables/conf_reliance.tex.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import metrics as M                                    # noqa: E402
from experiments.analyze_v7 import RowSet, logit, align                 # noqa: E402

SRC = "results/posthoc/conf_reliance"
OUT = "results/v15conf/analysis/conf_reliance.csv"
TEX = "paper/tables/conf_reliance.tex"
GROUPS = (("v13", "A_none", "no remedy, historical recipe"), ("v15conf", "A_none", "no remedy, corrected recipe"),
          ("v15conf", "C_groupdro", "Group DRO, corrected recipe"))
MWI = {("v13", "A_none"): ("results/v13/analysis/effects.csv", lambda r: r["proto"] == "v13" and r["contrast"].startswith("SE(A_none)")),
       ("v15conf", "A_none"): ("results/v15conf/analysis/effects.csv", lambda r: r["contrast"] == "MWI(A_none)"),
       ("v15conf", "C_groupdro"): ("results/v15conf/analysis/effects.csv", lambda r: r["contrast"] == "MWI(C_groupdro)")}


def fmt(a, d=3):
    lo, hi = M.ci(a[1:])
    return f"{a[0]:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def main():
    reg = json.load(open(f"{SRC}/registry.json"))
    models = reg["models"]
    assert len(models) == 18, len(models)
    data = {}
    for m in models:
        rd = list(csv.DictReader(open(f"{SRC}/scores/{m['name']}.csv")))
        data[m["name"]] = rd
    first = next(iter(data.values()))
    ids = [r["identifier"] for r in first]
    rs = RowSet(ids, [int(r["label"]) for r in first], [r["child_id"] for r in first], 2000)
    cols = (["clean"] + [f"perm{k}" for k in range(reg["n_perm"])] + ["const"]
            + [f"pol{w}{x}" for w in ("S", "I") for x in [""] + [f"_perm{k}" for k in range(reg["n_perm"])]])
    A = {}
    for n, rd in data.items():
        S = align(rs, [r["identifier"] for r in rd], [r["label"] for r in rd],
                  {c: np.array([float(r[c]) for r in rd]) for c in cols})
        A[n] = {c: logit(rs.aucs(S[c])) for c in cols}
    rows, lines = [], []
    for proto, cell, label in GROUPS:
        res = {}
        for world in ("S", "I"):
            g = [m["name"] for m in models if m["proto"] == proto and m["cell"] == cell and m["world"] == world]
            assert sorted(m["seed"] for m in models if m["proto"] == proto and m["cell"] == cell and m["world"] == world) == [0, 1, 2]
            perm = np.mean([np.mean([A[n]["clean"] - A[n][f"perm{k}"] for k in range(reg["n_perm"])], axis=0) for n in g], axis=0)
            cst = np.mean([A[n]["clean"] - A[n]["const"] for n in g], axis=0)
            clean = np.mean([A[n]["clean"] for n in g], axis=0)
            pol = {p_: np.mean([np.mean([A[n][f"pol{p_}"] - A[n][f"pol{p_}_perm{k}"] for k in range(reg["n_perm"])], axis=0)
                                for n in g], axis=0) for p_ in ("S", "I")}
            res[world] = (perm, cst, clean, pol)
        d, dc = res["S"][0] - res["I"][0], res["S"][1] - res["I"][1]
        path, sel = MWI[(proto, cell)]
        mwi = next(r for r in csv.DictReader(open(path)) if sel(r))["logit"]
        rec = {"protocol": proto, "cell": cell, "label": label, "reliance_perm_S": fmt(res["S"][0]), "reliance_perm_I": fmt(res["I"][0]),
               "reliance_perm_S_minus_I": fmt(d), "reliance_const_S": fmt(res["S"][1]), "reliance_const_I": fmt(res["I"][1]),
               "reliance_const_S_minus_I": fmt(dc),
               "reliance_policyS_S": fmt(res["S"][3]["S"]), "reliance_policyS_I": fmt(res["I"][3]["S"]),
               "reliance_policyS_S_minus_I": fmt(res["S"][3]["S"] - res["I"][3]["S"]),
               "reliance_policyI_S": fmt(res["S"][3]["I"]), "reliance_policyI_I": fmt(res["I"][3]["I"]),
               "reliance_policyI_S_minus_I": fmt(res["S"][3]["I"] - res["I"][3]["I"]), "MWI_logit": mwi}
        rows.append(rec)
        lines.append(f"{label} & {rec['reliance_perm_S_minus_I']} & {rec['reliance_policyS_S_minus_I']} & "
                     f"{rec['reliance_policyI_S_minus_I']} & {float(mwi.split()[0]):+.3f} \\\\")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        f.write("# post hoc: permutation reliance on the confidence channel, unplanted host test windows, "
                "logit AUROC drop, seeds averaged inside 2,000 identifier-clustered resamples\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    body = [t.replace(" -", " $-$").replace("[-", "[$-$") for t in lines]
    tex = [r"\begin{tabular}{lcccc}", r"\toprule",
           r" & \multicolumn{3}{c}{Reliance, $f_S$ minus $f_I$} & \\", r"\cmidrule(lr){2-4}",
           r"Models & clean input & policy S input & policy I input & MWI \\", r"\midrule"] + body + \
          [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(TEX), exist_ok=True)
    with open(TEX, "w") as f:
        f.write("% generated by experiments/analyze_conf_reliance.py from results/posthoc/conf_reliance; do not edit\n")
        f.write("\n".join(tex) + "\n")
    for r in rows:
        print(r["label"], "| S", r["reliance_perm_S"], "| I", r["reliance_perm_I"], "| S-I", r["reliance_perm_S_minus_I"],
              "| const S-I", r["reliance_const_S_minus_I"], "| polS S/I", r["reliance_policyS_S"], r["reliance_policyS_I"],
              "S-I", r["reliance_policyS_S_minus_I"], "| polI S-I", r["reliance_policyI_S_minus_I"], "| MWI", r["MWI_logit"])


if __name__ == "__main__":
    main()
