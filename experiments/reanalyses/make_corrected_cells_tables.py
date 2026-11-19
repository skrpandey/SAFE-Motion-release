"""Build the corrected-loader four-cell tables from results/reanalyses/r25_corrected_four_cells.csv.

Writes (paper/tables):
  corrected_core_readable.tex          article Table 2 (four raw AUROCs, responses, TTI, clean AUROC)
  corrected_four_cells.tex             supplement Table S2 (cell means on both scales, responses,
                                              raw and logit TTI, clean AUROC and changes, with intervals)
Values are rounded from the exported five-decimal strings; nothing is recomputed here.
    python3 experiments/reanalyses/make_corrected_cells_tables.py
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/reanalyses/r25_corrected_four_cells.csv"
OUT = ROOT / "paper/tables"
NAMES = {"none": "No remedy", "dro_severity": "Severity-group DRO", "dro_input": "Input-group DRO"}


def num(x, sign=True):
    s = f"{float(x):+.3f}" if sign else f"{float(x):.3f}"
    return s.replace("-", "$-$")


def ci(s, sign=True):
    p, lo, hi = (float(v) for v in re.findall(r"[+-]?\d+\.\d+", s))
    return f"{num(p, sign)} [{num(lo, sign)}, {num(hi, sign)}]"


def point(s, sign=True):
    return num(re.findall(r"[+-]?\d+\.\d+", s)[0], sign)


def main():
    rows = {r["cell"]: r for r in csv.DictReader(l for l in open(SRC) if not l.startswith("#"))}
    # article Table 2
    t = [r"\begin{tabular}{@{}lccccccccc@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{Mean AUROC} & \multicolumn{3}{c}{Logit AUROC, mean [95\% interval]} & \multicolumn{2}{c}{Clean AUROC} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}",
         r"Training variant & $(S,S)$ & $(S,I)$ & $(I,S)$ & $(I,I)$ & \R{}$(f_S)$ & \R{}$(f_I)$ & TTI & S & I \\",
         r"\midrule"]
    for k, name in NAMES.items():
        r = rows[k]
        t.append(" & ".join([name] + [num(r[f"auroc_{c}"], False) for c in ("SS", "SI", "IS", "II")]
                            + [ci(r["R_S"]), ci(r["R_I"]), ci(r["TTI_logit"]),
                               point(r["clean_S"], False), point(r["clean_I"], False)]) + r" \\")
    t += [r"\bottomrule", r"\end{tabular}", ""]
    (OUT / "corrected_core_readable.tex").write_text("\n".join(t))
    # supplement table: (a) cells and responses, (b) interactions, (c) clean performance
    last = list(NAMES)[-1]
    s = [r"\textit{(a) Condition-by-policy means and responses}\\[2pt]",
         r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{Mean AUROC} & \multicolumn{2}{c}{Mean logit AUROC} & \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"Variant & Training & policy S & policy I & policy S & policy I & Response \R{} (logit) \\",
         r"\midrule"]
    for k, name in NAMES.items():
        r = rows[k]
        for w in ("S", "I"):
            s.append(" & ".join([name if w == "S" else "", f"Condition {w}",
                                 num(r[f"auroc_{w}S"], False), num(r[f"auroc_{w}I"], False),
                                 num(r[f"logit_{w}S"], False), num(r[f"logit_{w}I"], False),
                                 ci(r[f"R_{w}"])]) + r" \\")
        if k != last:
            s.append(r"\addlinespace")
    s += [r"\bottomrule", r"\end{tabular}", "", r"\medskip",
          r"\textit{(b) Interactions and changes from no remedy}\\[2pt]",
          r"\begin{tabular}{@{}llcc@{}}", r"\toprule",
          r"Variant & Scale & TTI & $\Delta$TTI \\", r"\midrule"]
    for k, name in NAMES.items():
        r = rows[k]
        for sc, lab in (("logit", "logit"), ("raw", "raw")):
            d = "--" if k == "none" else ci(r[f"dTTI_{sc}"])
            s.append(" & ".join([name if sc == "logit" else "", lab, ci(r[f"TTI_{sc}"]), d]) + r" \\")
        if k != last:
            s.append(r"\addlinespace")
    s += [r"\bottomrule", r"\end{tabular}", "", r"\medskip",
          r"\textit{(c) Clean performance}\\[2pt]",
          r"\begin{tabular}{@{}llcc@{}}", r"\toprule",
          r"Variant & Training & Clean AUROC & $\Delta$clean \\", r"\midrule"]
    for k, name in NAMES.items():
        r = rows[k]
        for w in ("S", "I"):
            d = "--" if k == "none" else ci(r[f"clean_change_{w}"])
            s.append(" & ".join([name if w == "S" else "", f"Condition {w}", ci(r[f"clean_{w}"], False), d]) + r" \\")
        if k != last:
            s.append(r"\addlinespace")
    s += [r"\bottomrule", r"\end{tabular}", ""]
    (OUT / "corrected_four_cells.tex").write_text("\n".join(s))
    print("wrote paper/tables/corrected_core_readable.tex and corrected_four_cells.tex")


if __name__ == "__main__":
    main()
