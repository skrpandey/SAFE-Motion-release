"""R24 (exclusions by the 200-frame rule; direction of the length association) and R17 (12,572 against
9,859 + 2,702 development segments).

R24, from the SAFE-SMM cache index (36,928 segments): per release partition (development = 289
identifiers of the 'train' split; test = 40 identifiers) and class, the number and share of segments
shorter than 200 frames, the median length, and the AUROC of segment length for the label (which
class is longer); also restricted to QC-passing segments (the planted-host population).
R17: rebuilds the host development rows as experiments/train_asdpose.py does, counts the validation
rows of the aliased identifier (322; selection_repair.yaml val_exclude_ids), and reads every
stored training log (results/fits/*/results/epochs/*/train_log.json; training logs are not
redistributed, so without them the log-derived entries of r17_reconcile.json come out empty) for
the validation row counts and the selection-change flags.
Writes results/reanalyses/r24_exclusions.csv, r17_reconcile.json.
    python3 experiments/reanalyses/r24_r17_counts.py
"""
import glob
import json

import numpy as np
import yaml

import common as C
from safe_motion.data import asdpose as D


def main():
    v1 = yaml.safe_load(open("configs/protocols/pilot_1.yaml"))
    rows = D.load_index()
    out = []
    for pop, filt in (("all segments", lambda r: True), ("QC-passing segments", lambda r: D.passes_qc(r, v1["qc_filter"]))):
        for part, split in (("development (289 identifiers)", "train"), ("test (40 identifiers)", "test")):
            sel = [r for r in rows if r["split"] == split and filt(r)]
            n = np.array([r["n_frames"] for r in sel])
            y = np.array([r["binary_label"] for r in sel])
            for c, cname in ((0, "non-SMM"), (1, "SMM"), (None, "both")):
                k = np.ones(len(sel), bool) if c is None else (y == c)
                short = n[k] < 200
                out.append({"population": pop, "partition": part, "class": cname, "segments": int(k.sum()),
                            "removed_lt_200_frames": int(short.sum()), "share_removed": float(short.mean()),
                            "kept": int((~short).sum()), "median_frames": float(np.median(n[k])),
                            "median_frames_kept": float(np.median(n[k][~short])),
                            "identifiers": len({r["child_id"] for r, kk in zip(sel, k) if kk})})
            out.append({"population": pop, "partition": part, "class": "AUROC of length for SMM (longer = SMM)",
                        "segments": len(sel), "median_frames": C.M.auc(n, y),
                        "median_frames_kept": C.M.auc(n[n >= 200], y[n >= 200])})
    for r in out:
        print(r)
    # source of the printed "length alone predicts the label at AUROC 0.90" (pilot_6.yaml length_cue_ceiling:
    # HistGradientBoosting on length alone, all test segments 0.904, included test segments 0.947), the length grid
    # by class, and the frame-rate field of the release annotations (cache index column fps_unreliable, copied from
    # dataset.pkl by the SAFE-SMM cache builder)
    from sklearn.ensemble import HistGradientBoostingClassifier
    dev = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] == "test"]
    Xd, yd = np.array([[r["n_frames"]] for r in dev]), np.array([r["binary_label"] for r in dev])
    Xt, yt = np.array([[r["n_frames"]] for r in te]), np.array([r["binary_label"] for r in te])
    inc = Xt[:, 0] >= 200
    gb = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(Xd, yd)
    gbi = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(Xd[Xd[:, 0] >= 200], yd[Xd[:, 0] >= 200])
    length = {"hgb_length_only_fit_dev_all_test_all": C.M.auc(gb.predict_proba(Xt)[:, 1], yt),
              "hgb_length_only_fit_dev_ge200_test_ge200": C.M.auc(gbi.predict_proba(Xt[inc])[:, 1], yt[inc]),
              "rank_auroc_length_for_smm_dev_all": C.M.auc(Xd[:, 0], yd), "rank_auroc_length_for_smm_test_all": C.M.auc(Xt[:, 0], yt)}
    grid = []
    for pop, filt in (("all segments", lambda r: True), ("length-controlled (>= 200 frames)", lambda r: r["n_frames"] >= 200),
                      ("planted host (QC-passing, >= 200 frames)", lambda r: r["n_frames"] >= 200 and D.passes_qc(r, v1["qc_filter"]))):
        for part, split in (("development", "train"), ("test", "test")):
            for c, cname in ((0, "non-SMM"), (1, "SMM")):
                sel = [r for r in rows if r["split"] == split and r["binary_label"] == c and filt(r)]
                n = np.array([r["n_frames"] for r in sel])
                fps = np.array([float(r["fps_unreliable"]) for r in sel])
                grid.append({"population": pop, "partition": part, "class": cname, "segments": len(sel),
                             "share_length_multiple_of_25": float(np.mean(n % 25 == 0)),
                             "share_length_30k_or_30k_plus_1": float(np.mean(np.isin(n % 30, [0, 1]))),
                             "share_fps_field_25": float(np.mean(fps == 25)), "share_fps_field_30": float(np.mean(fps == 30)),
                             "identifiers_with_fps_30": len({r["child_id"] for r in sel if float(r["fps_unreliable"]) == 30})})
    # recording groups (identifier prefix child_assessment_part): does the fps field vary by class within a recording?
    from collections import defaultdict
    rec = defaultdict(set)
    for r in rows:
        rec["_".join(r["identifier"].split("_")[:3])].add((r["binary_label"], float(r["fps_unreliable"])))
    length["recording_groups"] = len(rec)
    length["recording_groups_smm30_and_nonsmm25"] = sum(1 for v in rec.values() if (1, 30.0) in v and (0, 25.0) in v)
    length["recording_groups_smm_at_both_25_and_30"] = sum(1 for v in rec.values() if (1, 30.0) in v and (1, 25.0) in v)
    length["recording_groups_nonsmm_at_30"] = sum(1 for v in rec.values() if (0, 30.0) in v)
    for g in grid:
        print(g)
    print(length)
    C.write_csv("r24_length_grid_fps.csv", grid,
                "R24 extra: length grid and frame-rate field by class. share_length_*: segment lengths (frames) on a 25-frame grid or on a\n"
                "30-frame grid (30k or 30k+1). share_fps_field_*: the per-segment fps value of the release annotations (dataset.pkl\n"
                "'fps', copied to the SAFE-SMM cache index as fps_unreliable)")
    C.write_json("r24_length_summary.json", length)
    C.write_csv("r24_exclusions.csv", out,
                "R24: segments removed by the 200-frame rule (n_frames < 200), by release partition and class, SAFE-SMM cache index\n"
                "(36,928 segments); median segment length (frames); last row per block: AUROC of the frame count for the SMM label\n"
                "(above 0.5 = SMM longer), in the 'median_frames' column for all segments and in 'median_frames_kept' after the rule")

    # R17
    tr, va = C.dev_host_rows()
    excl = yaml.safe_load(open("configs/protocols/selection_repair.yaml"))["val_exclude_ids"]
    alias_va = [r for r in va if int(r["child_id"]) in excl]
    alias_tr = [r for r in tr if int(r["child_id"]) in excl]
    tr196 = [r for r in tr if int(r["child_id"]) == 196]
    logs = {}
    for p in sorted(glob.glob("results/fits/*/results/epochs/*/train_log.json")):
        name = p.split("/")[-2]
        lg = json.load(open(p))
        logs[name] = {k: lg.get(k) for k in ("n_val_rows", "n_val_rows_clean", "selection_changed",
                                              "selected_epoch_orig", "selected_epoch_clean")}
    host_logs = {k: v for k, v in logs.items() if v["n_val_rows"] == len(va)}
    res = {"train_host_rows": len(tr), "validation_host_rows": len(va), "development_host_rows": len(tr) + len(va),
           "excluded_identifiers": excl, "alias_validation_rows": len(alias_va),
           "alias_validation_rows_by_class": {"SMM": sum(r["binary_label"] for r in alias_va),
                                              "non-SMM": sum(1 - r["binary_label"] for r in alias_va)},
           "alias_rows_in_training": len(alias_tr), "train_rows_of_identifier_196_same_person": len(tr196),
           "validation_rows_after_removal": len(va) - len(alias_va),
           "stored_training_logs": len(logs), "host_fit_logs": len(host_logs),
           "host_fit_logs_val_rows": sorted({(v["n_val_rows"], v["n_val_rows_clean"]) for v in host_logs.values()}),
           "fits_whose_selected_epoch_differs_under_original_rule": sorted(k for k, v in logs.items() if v["selection_changed"]),
           "confidence_only_headline_fits_changed": {k: v for k, v in logs.items() if v["selection_changed"] and "armruns_mid_A_none" in k}}
    print(json.dumps(res, indent=1))
    C.write_json("r17_reconcile.json", res)


if __name__ == "__main__":
    main()
