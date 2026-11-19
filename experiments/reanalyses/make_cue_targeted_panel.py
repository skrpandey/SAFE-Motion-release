"""Build panel (b) of supplement Table S4 (cue-targeted variants under missing runs and frozen frames,
historical loader) from results/cue_remedies/analysis/effects.csv.

Prints each variant's interaction with its 95% interval on the logit and raw AUROC scales, and the raw-scale
change from no remedy with its interval (the logit-scale change is in panel (a) and article Table 3). Mask
mean, cleaned selection. Values are rounded once from the stored strings.
Writes paper/tables/cue_targeted_intervals.tex.
    python3 experiments/reanalyses/make_cue_targeted_panel.py
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/cue_remedies/analysis/effects.csv"
OUT = ROOT / "paper/tables/cue_targeted_intervals.tex"
NUM = re.compile(r"[-+]?\d+\.\d+")
MECH = {"runs": "missing runs", "frozen": "frozen frames"}
CELL = {"C_groupdro": "Group DRO", "C_adversarial": "adversarial head", "C_augment": "operator augmentation"}


def iv(s):
    p, lo, hi = (float(x) for x in NUM.findall(s))
    return f"{p:+.3f} [{lo:+.3f}, {hi:+.3f}]".replace("-", "$-$")


def main():
    rows = list(csv.DictReader(l for l in open(SRC) if not l.startswith("#")))
    get = {(r["mechanism"], r["cell"], r["contrast"]): r for r in rows}
    t = [r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
         r"Operator & Method & TTI, logit & TTI, raw AUROC & $\Delta$TTI, raw AUROC \\", r"\midrule"]
    for m in ("runs", "frozen"):
        for i, c in enumerate(("C_groupdro", "C_adversarial", "C_augment")):
            se = get[(m, c, f"SE({c})")]
            d = get[(m, c, f"SE({c}) - SE(A_none)")]
            t.append(" & ".join([MECH[m] if i == 0 else "", CELL[c], iv(se["logit"]), iv(se["raw"]), iv(d["raw"])]) + r" \\")
        if m == "runs":
            t.append(r"\addlinespace")
    t += [r"\bottomrule", r"\end{tabular}", ""]
    OUT.write_text("\n".join(t))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
