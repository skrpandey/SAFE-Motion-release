"""Descriptive counts from the repair_selection_v14 training logs.

Reads results/fits/repair_v14/results/epochs/*/train_log.json and writes
results/v7/analysis/repair_counts.csv: number of fits, fits whose selected epoch is the last (25)
under the original and the cleaned rule, validation rows before and after the alias exclusion,
and the recorded code commits.
"""
import csv
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion.paths import guard_output   # noqa: E402

LOGS = "results/fits/repair_v14/results/epochs/*/train_log.json"


def main():
    L = [json.load(open(p)) for p in sorted(glob.glob(LOGS))]
    assert L, "no repair logs found"
    n = len(L)
    rows = [{"quantity": "fits", "value": n},
            {"quantity": "selected epoch is the last (25), original rule", "value": sum(j["selected_epoch_orig"] == 25 for j in L)},
            {"quantity": "selected epoch is the last (25), cleaned rule", "value": sum(j["selected_epoch_clean"] == 25 for j in L)},
            {"quantity": "selections changed by the cleaned rule", "value": sum(bool(j["selection_changed"]) for j in L)}]
    for (pool, clean), k in sorted(Counter((j["n_val_rows"], j["n_val_rows_clean"]) for j in L).items()):
        rows.append({"quantity": f"validation rows pool -> cleaned ({k} fits)", "value": f"{pool} -> {clean}"})
    for commit, k in sorted(Counter(str((j.get("code") or {}).get("commit", "")) or "(blank)" for j in L).items()):
        rows.append({"quantity": f"code.commit = {commit}", "value": k})
    out = "results/v7/analysis/repair_counts.csv"
    guard_output(out)
    with open(out, "w", newline="") as f:
        f.write("# descriptive counts from the repair_selection_v14 train logs (read only)\n")
        w = csv.DictWriter(f, fieldnames=["quantity", "value"]); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)
    print("wrote", out)


if __name__ == "__main__":
    main()
