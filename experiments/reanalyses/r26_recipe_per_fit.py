"""Six-decimal cells and per-fit record of protocol v20 (recipe experiment), 2026-11-19.

Two read-only exports for supplement Section S9 and Table S14. Nothing is refitted or rescored, and
the frozen analysis outputs in results/recipe_comparison/analysis are not written.

1. r26_recipe_cells.csv: for the three model sets (v20 A_both recipe; v15conf A_none, the published
   primary comparator; v20 A_none, the local device-matched refit) the four condition-by-policy mean
   AUROCs on both scales, both responses, TTI on both scales, clean AUROC per condition and S minus I,
   to six decimals so that printed values are rounded once. Estimator of experiments/analyze_recipe_comparison.py:
   AUROC per fitted model and test mask inside each of the 2,000 shared identifier-clustered
   resamples, transformed (logit) or not (raw), then averaged over seeds and masks. Every value is
   checked against results/recipe_comparison/analysis/effects.csv (five decimals) and the paired contrasts against
   criteria.json before anything is written.
2. r26_recipe_per_fit.csv: for each of the 18 fits, what the stored training log and score file say:
   device, code commit and dirty flag, selected epoch and validation AUROC under the cleaned selection
   rule, and clean test AUROC of the selected checkpoint on the 1,966 host test rows. The training
   logs (results/fits/<run>/.../train_log.json) are not redistributed; without them this part is
   skipped and the released r26_recipe_per_fit.csv and r26_summary.json are left as they are.
    python3 experiments/reanalyses/r26_recipe_per_fit.py
"""
import csv
import json
import re

import numpy as np

import common as C

POL = ("S", "I")
V20 = C.ROOT / "results/recipe_comparison/analysis"
SETS = (
    ("A_none_v15conf", "no remedy, published (v15conf A_none)", "v15conf", "A_none",
     "results/fits/v15conf/results/epochs"),
    ("A_both", "recipe (v20 A_both)", "v20", "A_both", "results/fits/v20/epochs"),
    ("A_none_local", "no remedy, local refit (v20 A_none)", "v20", "A_none", "results/fits/v20/epochs"),
)
EPOCHS = 25
NUM = re.compile(r"[-+]?\d+\.\d+")


def parse(s):
    return tuple(float(x) for x in NUM.findall(s))


def main():
    cells = {key: C.models(src, cell, (0, 1, 2)) for key, _, src, cell, _ in SETS}
    recs = [m for g in cells.values() for w in POL for m in g[w]]     # v15conf first: its row order
    sc = C.Scores(recs)
    eng = C.Engine(sc)
    assert len(sc.ids) == 1966 and eng.n_clusters == 40

    def cell_mean(g, w, p, scale):
        f = C.logit if scale == "logit" else (lambda a: a)
        return np.mean([[f(eng.aucs(m["name"], f"{C.MECH}_{p}_mid_m{ms}")) for ms in C.MASK_SEEDS]
                        for m in g[w]], axis=(0, 1))

    def clean(g, w):
        return np.mean([eng.aucs(m["name"], "clean") for m in g[w]], axis=0)

    frozen = {r["model_set"]: r for r in csv.DictReader(open(V20 / "effects.csv"))}
    crit = json.load(open(V20 / "criteria.json"))
    rows, arr, problems = [], {}, []
    for key, label, _, _, _ in SETS:
        g = cells[key]
        tl, rS, rI, _, _ = eng.cell(g)
        tr = eng.cell(g, scale="raw")[0]
        cS, cI = clean(g, "S"), clean(g, "I")
        arr[key] = {"tl": tl, "tr": tr, "cS": cS, "cI": cI}
        row = {"model_set": key, "label": label}
        for w in POL:
            for p in POL:
                row[f"auroc_{w}{p}"] = round(float(cell_mean(g, w, p, "raw")[0]), 6)
                row[f"logit_{w}{p}"] = round(float(cell_mean(g, w, p, "logit")[0]), 6)
        q = {"R_S": rS, "R_I": rI, "TTI_logit": tl, "TTI_raw": tr, "clean_S": cS, "clean_I": cI,
             "clean_S_minus_I": cS - cI}
        for name, a in q.items():
            row[name] = C.fmt(a, 6)
            if any(abs(x - y) > 5.1e-6 for x, y in zip(C.ci(a), parse(frozen[key][name]))):
                problems.append(f"{key} {name}: {C.fmt(a, 6)} differs from frozen {frozen[key][name]}")
        for k in ("auroc_SS", "auroc_SI", "auroc_IS", "auroc_II", "logit_SS", "logit_SI", "logit_IS", "logit_II"):
            if abs(row[k] - float(frozen[key][k])) > 1e-9:
                problems.append(f"{key} {k}: {row[k]} differs from frozen {frozen[key][k]}")
        rows.append(row)
    # paired contrasts against the frozen criteria (full precision there)
    for base, block in (("A_none_v15conf", "P20_1_P20_2"), ("A_none_local", "D20_3")):
        for name, a in (("dTTI_logit", arr["A_both"]["tl"] - arr[base]["tl"]),
                        ("dTTI_raw", arr["A_both"]["tr"] - arr[base]["tr"])):
            if not np.allclose(C.ci(a), crit[block][name], atol=1e-12):
                problems.append(f"{block} {name}: {C.ci(a)} differs from frozen {crit[block][name]}")
        for w, k in (("S", "cS"), ("I", "cI")):
            a = arr["A_both"][k] - arr[base][k]
            if not np.allclose(C.ci(a), crit[block]["clean_change"][w], atol=1e-12):
                problems.append(f"{block} clean change {w} differs from frozen")
    if problems:
        raise SystemExit("check failed:\n  " + "\n  ".join(problems))
    C.write_csv("r26_recipe_cells.csv", rows,
                header_comment="protocol v20 model sets; corrected loader, coordinate rule, assignment 0, "
                               "CompactCTRGCN seeds 0-2, mask mean 2000/3000/4000; AUROC per model and mask, "
                               "transformed, then averaged; checked against results/recipe_comparison/analysis")

    if not all((C.ROOT / d).is_dir() for _, _, _, _, d in SETS):
        print("training logs not found under results/fits; per-fit record not rewritten")
        print("all frozen v20 values reproduced")
        return
    fits, summary = [], {}
    for key, label, src, cell, epoch_dir in SETS:
        sel, cl = [], []
        for w in POL:
            for m in cells[key][w]:
                log = json.load(open(C.ROOT / epoch_dir / m["name"] / "train_log.json"))
                auc = C.M.auc(list(sc.col(m["name"], "clean")), list(sc.y))
                assert len(log["log"]) == EPOCHS, m["name"]
                fits.append({
                    "model_set": key, "fit": m["name"], "condition": w, "seed": m["seed"],
                    "gpu": log["hardware"]["gpu"], "torch": log["software"]["torch"],
                    "code_commit": log["code"]["commit"][:7], "code_dirty": log["code"]["dirty"],
                    "train_recipe": log["train_recipe"], "valid_rule": log["cell"]["valid_rule"],
                    "velocity": log["cell"]["velocity"], "joint_dropout": log["cell"]["joint_dropout"],
                    "epoch_budget": EPOCHS, "selected_epoch_clean": log["selected_epoch_clean"],
                    "selection_changed_by_cleaning": log["selection_changed"],
                    "val_auroc_clean": f"{log['val_auc_clean']:.6f}", "clean_test_auroc": f"{auc:.6f}",
                    "train_minutes": f"{log['train_minutes']:.2f}"})
                sel.append(log["selected_epoch_clean"])
                cl.append(auc)
        summary[key] = {"label": label, "n_fits": len(sel), "selected_epochs": sorted(sel),
                        "clean_test_auroc_min": round(min(cl), 6), "clean_test_auroc_max": round(max(cl), 6)}
    C.write_csv("r26_recipe_per_fit.csv", fits,
                header_comment="per-fit record of protocol v20 and of the published no-remedy cell; stored "
                               "training logs and per-segment scores only")
    C.write_json("r26_summary.json", summary)
    print(json.dumps(summary, indent=1))
    print("all frozen v20 values reproduced")


if __name__ == "__main__":
    main()
