"""Identity search for the unresolved ASDPose test IDs (post hoc) over the full length of every
raw recording, ignoring the stated start time. Masked RMS distance over nose, shoulders and hips at
every offset via FFT cross-correlation at 15 fps (stride 2); the two longest segments per group.
Positive controls: 30 accepted groups from the first pass.
Writes results/asdpose_id_audit/fullsearch_all.csv.gz."""
import os
SMM_ROOT = os.environ.get("SM_SMM_ROOT", "data/asdpose-smm-detection")   # SAFE-SMM checkout with cache/
ASDPOSE_RAW = os.environ.get("SM_ASDPOSE_RAW", "data/StimmingADSMotion")   # released ASDPose files
import io, sys, zipfile, pickle
import numpy as np, pandas as pd
from multiprocessing import Pool
J = [0, 5, 6, 11, 12]; DS = 2
ZIP = os.path.join(ASDPOSE_RAW, "asdpose.zip"); A = "results/asdpose_id_audit"
c = pd.read_csv(os.path.join(SMM_ROOT, "cache/segments/index.csv"))
p = c.identifier.str.split("_", expand=True)
c["f1"], c["f2"], c["f3"] = p[0].astype(int), p[1].astype(int), p[2].astype(int)
KP = np.memmap(os.path.join(SMM_ROOT, "cache/segments/keypoints.f32"), dtype=np.float32, mode="r")
I = pd.read_csv(f"{A}/manifest_ids.csv"); Gm = pd.read_csv(f"{A}/manifest_groups.csv")
unres = I[(I.partition == "test") & (I.status != "accepted")].annotated_id.tolist()
ctrl = Gm[Gm.status == "accepted"].sample(30, random_state=0)[["f1", "f2", "f3"]].apply(tuple, axis=1).tolist()
Q = []
for g, sub in c.groupby(["f1", "f2", "f3"]):
    if g[0] in unres or g in ctrl:
        for _, r in sub.sort_values("n_frames", ascending=False).head(2).iterrows():
            a = np.array(KP[r.byte_offset // 4: r.byte_offset // 4 + r.n_frames * 51]).reshape(r.n_frames, 17, 3)[::DS, J, :2]
            Q.append(("control" if g in ctrl else "unresolved", g, r.identifier, a.astype(np.float64)))
names = [n for n in zipfile.ZipFile(ZIP).namelist() if n.endswith(".pkl")]


def work(n):
    R = pickle.load(io.BytesIO(zipfile.ZipFile(ZIP).read(n)))
    K = R["keypoint"][0][::DS][:, J, :].astype(np.float64)        # (T, 5, 2)
    mk = (K != 0).all(2).astype(np.float64)                     # (T, 5)
    T = len(K); out = []
    nfft = 1 << int(np.ceil(np.log2(T + 400)))
    FK = {}
    for j in range(len(J)):
        FK[j, "m"] = np.fft.rfft(mk[:, j], nfft)
        for d in range(2):
            x = K[:, j, d] * mk[:, j]
            FK[j, d, 1] = np.fft.rfft(x, nfft); FK[j, d, 2] = np.fft.rfft(x * K[:, j, d], nfft)
    for kind, g, ident, q in Q:
        L = len(q)
        if L >= T:
            out.append((kind, g, ident, n, np.inf, -1)); continue
        mq = (q != 0).all(2).astype(np.float64)
        if mq.sum() < 60 or np.nanstd(np.where(mq[..., None] > 0, q, np.nan)) < 2.0:   # too few valid joint-frames, or static
            out.append((kind, g, ident, n, np.inf, -2)); continue
        D = np.zeros(T - L + 1); C = np.zeros(T - L + 1)
        for j in range(len(J)):
            fm = np.fft.rfft(mq[::-1, j], nfft)
            C += np.fft.irfft(FK[j, "m"] * fm, nfft)[L - 1:T]
            for d in range(2):
                qq = q[::-1, j, d] * mq[::-1, j]
                D += (np.fft.irfft(FK[j, d, 2] * fm, nfft)[L - 1:T]
                      - 2 * np.fft.irfft(FK[j, d, 1] * np.fft.rfft(qq, nfft), nfft)[L - 1:T]
                      + np.fft.irfft(FK[j, "m"] * np.fft.rfft(qq * q[::-1, j, d], nfft), nfft)[L - 1:T])
        ok = C >= max(60, 0.5 * mq.sum())                                 # at least half the query's valid joint-frames overlap
        rms = np.where(ok, np.sqrt(np.maximum(D, 0) / np.maximum(C, 1)), np.inf)
        k = int(rms.argmin()); out.append((kind, g, ident, n, float(rms[k]), k * DS))
    return out


if __name__ == "__main__":
    print("queries", len(Q), "unresolved ids", len(unres), flush=True)
    with Pool(11) as pool:
        res = [r for part in pool.imap_unordered(work, names, chunksize=2) for r in part]
    df = pd.DataFrame(res, columns=["kind", "group", "identifier", "raw", "rms_px", "frame"])
    df["raw_child"] = df.raw.str.extract(r"data/(\d+)_")[0].astype(int)
    df.to_csv(f"{A}/fullsearch_all.csv.gz", index=False)
    print("done", len(df))
