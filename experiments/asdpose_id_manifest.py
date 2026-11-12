"""Identity manifest for the ASDPose annotated release (post hoc).

Inputs: results/asdpose_id_audit/{errmat.npy, groups.csv, rawnames.csv} from asdpose_id_audit_full.py
(median nose, shoulder and hip distance over -4 s to +2 s around the stated start), the cache index
and the raw db.csv. A group is accepted if best error < 8 px and the best other-child error is
>= 3x that, ambiguous if < 8 px but the ratio is < 3, otherwise no candidate. Canonical person is
the raw db.csv child_id. Writes manifest_groups.csv, manifest_ids.csv and summary.json there.
"""
import os
SMM_ROOT = os.environ.get("SM_SMM_ROOT", "data/asdpose-smm-detection")   # SAFE-SMM checkout with cache/
ASDPOSE_RAW = os.environ.get("SM_ASDPOSE_RAW", "data/StimmingADSMotion")   # released ASDPose files
import hashlib, json, zipfile, io
import numpy as np, pandas as pd

A = "results/asdpose_id_audit"
IDX = os.path.join(SMM_ROOT, "cache/segments/index.csv")
ZIP = os.path.join(ASDPOSE_RAW, "asdpose.zip")


def sha(p, n=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read(n))
    return h.hexdigest()[:16] + " (first 16 MiB)"


M = np.load(f"{A}/errmat.npy"); G = pd.read_csv(f"{A}/groups.csv"); N = pd.read_csv(f"{A}/rawnames.csv").iloc[:, 0]
rc = N.str.extract(r"data/(\d+)_")[0].astype(int).values
db = pd.read_csv(io.BytesIO(zipfile.ZipFile(ZIP).read("db.csv")))
c = pd.read_csv(IDX)
# internal validation split as in train_asdpose.py (length protocol rows, seed 0)
L = c[c.n_frames >= 200]
kids = sorted(L[L.split == "train"].child_id.unique())
rng = np.random.RandomState(0)
val = set(rng.choice(kids, int(len(kids) * 0.2), replace=False).tolist())
part = {}
for k in c.child_id.unique():
    s = c[c.child_id == k].split.iloc[0]
    part[k] = "test" if s == "test" else ("validation" if k in val else ("train" if k in kids else "train(short only)"))

best = M.argmin(1); b = M[np.arange(len(M)), best]
sec = np.array([M[i][rc != rc[best[i]]].min() for i in range(len(M))])
valid = b < 1e8
status = np.where(~valid, "no candidate", np.where(b >= 8, "no candidate", np.where(sec / np.maximum(b, 1e-6) >= 3, "accepted", "ambiguous")))
G["best_raw_file"] = N.values[best]; G["raw_child"] = rc[best]; G["best_err_px"] = np.where(valid, b, np.nan)
G["other_child_err_px"] = np.where(sec < 1e8, sec, np.nan); G["status"] = status
G["partition"] = G.f1.map(part)
seg = c.assign(f1=c.identifier.str.split("_").str[0].astype(int), f2=c.identifier.str.split("_").str[1].astype(int),
               f3=c.identifier.str.split("_").str[2].astype(int))
G = G.merge(seg.groupby(["f1", "f2", "f3"]).agg(n_segments=("identifier", "size"), n_smm=("binary_label", "sum")).reset_index(),
            on=["f1", "f2", "f3"])
G.to_csv(f"{A}/manifest_groups.csv", index=False)

rows = []
for f1, g in G.groupby("f1"):
    acc = g[g.status == "accepted"]
    kids_ = sorted(set(acc.raw_child))
    st = "accepted" if len(kids_) == 1 else ("conflict" if len(kids_) > 1 else
         ("ambiguous only" if (g.status == "ambiguous").any() else "no candidate"))
    rows.append({"annotated_id": f1, "partition": part[f1], "n_groups": len(g), "n_accepted_groups": len(acc),
                 "canonical_person": kids_[0] if len(kids_) == 1 else "", "conflict_people": ";".join(map(str, kids_)) if len(kids_) > 1 else "",
                 "status": st, "n_segments": int(g.n_segments.sum()), "n_smm": int(g.n_smm.sum())})
I = pd.DataFrame(rows); I.to_csv(f"{A}/manifest_ids.csv", index=False)

acc = I[I.status == "accepted"]
alias = acc.groupby("canonical_person").filter(lambda d: len(d) > 1)
cross = alias.groupby("canonical_person").filter(lambda d: d.partition.nunique() > 1)
test_rows = L[L.split == "test"]
cov = {}
for p_ in ["train", "validation", "test"]:
    ids = I[I.partition == p_]; a_ = ids[ids.status == "accepted"]
    cov[p_] = {"ids": len(ids), "accepted_ids": len(a_), "segments": int(ids.n_segments.sum()),
               "segments_accepted": int(a_.n_segments.sum()), "smm": int(ids.n_smm.sum()), "smm_accepted": int(a_.n_smm.sum())}
ts = set(acc[acc.partition == "test"].annotated_id)
cov["real_test_length_protocol"] = {"segments": len(test_rows), "segments_accepted": int(test_rows.child_id.isin(ts).sum()),
                                    "smm": int(test_rows.binary_label.sum()), "smm_accepted": int(test_rows[test_rows.child_id.isin(ts)].binary_label.sum())}
# matcher validation
multi = I[I.n_accepted_groups >= 2]
summ = {"inputs": {"cache_index": sha(IDX), "asdpose_zip": sha(ZIP), "errmat": sha(f"{A}/errmat.npy")},
        "group_status": G.status.value_counts().to_dict(), "id_status": I.status.value_counts().to_dict(),
        "coverage": cov,
        "consistency_ids_with_2plus_accepted_groups": {"n": len(multi), "conflicts": int((multi.status == "conflict").sum())},
        "accepted_err_px_quantiles": np.nanpercentile(G[G.status == "accepted"].best_err_px, [50, 90, 99, 100]).round(2).tolist(),
        "wrong_child_err_px_quantiles_for_accepted": np.nanpercentile(G[G.status == "accepted"].other_child_err_px, [0, 1, 10, 50]).round(2).tolist(),
        "aliases": alias[["annotated_id", "partition", "canonical_person"]].to_dict("records"),
        "aliases_across_partitions": cross[["annotated_id", "partition", "canonical_person"]].to_dict("records"),
        "raw_children_with_accepted_id": int(acc.canonical_person.nunique()), "raw_children_total": int(db.child_id.nunique())}
json.dump(summ, open(f"{A}/summary.json", "w"), indent=1, default=str)
print(json.dumps(summ, indent=1, default=str))
