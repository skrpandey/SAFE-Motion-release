"""Length predictability from the actual model input and from the penultimate features.

For every real and planted A_none model with saved features, predicts log(original clip length)
from (a) a per-joint summary of the test-time input window and (b) the penultimate features, with
identifier-grouped out-of-fold ridge (alpha tuned by inner grouped CV) and gradient boosting R^2.
"within class" targets subtract class means fitted on each outer-training fold only.
Writes results/v9/analysis/length_inputs.{csv,md} and results/v7/analysis/length_inputs.csv.
    conda run -n torch python experiments/length_audit_inputs.py [--skip-v7] [--skip-v9]
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_v9 import discover, pool, pool_features   # noqa: E402

torch.set_num_threads(max(1, os.cpu_count() // 2))


def summarise_input(x):
    """x: (C, T, J) float tensor as the model sees it -> 1-D summary vector."""
    a = x.numpy()
    xy, conf = a[:2], (a[-1] if a.shape[0] in (3, 5) else None)
    feats = [xy.mean(axis=1).ravel(), xy.std(axis=1).ravel(),
             np.abs(np.diff(xy, axis=1)).mean(axis=1).ravel()]
    if conf is not None:
        feats += [conf.mean(axis=0), (conf > 0).mean(axis=0)]
    return np.concatenate(feats).astype(np.float32)


ALPHAS = np.logspace(-1, 4, 11)
INNER_FOLDS = 3


def class_means(y, label, fit_idx):
    """Class means of y over the rows fit_idx; a class absent there falls back to the overall mean."""
    mu = {}
    for c in np.unique(label):
        sel = fit_idx[label[fit_idx] == c]
        mu[c] = float(y[sel].mean()) if len(sel) else float(y[fit_idx].mean())
    return mu


def residual_target(y, label, mu):
    return y - np.array([mu[c] for c in label])


def fit_ridge_grouped(Xs, t, groups):
    """Ridge with alpha chosen by inner grouped CV on the given rows; returns (unfitted model, alpha)."""
    k = int(min(INNER_FOLDS, len(np.unique(groups))))
    if k < 2:
        alpha = float(ALPHAS[len(ALPHAS) // 2])
    else:
        err = np.zeros(len(ALPHAS))
        for itr, ite in GroupKFold(n_splits=k).split(Xs, t, groups):
            sc = StandardScaler().fit(Xs[itr])
            a, b = sc.transform(Xs[itr]), sc.transform(Xs[ite])
            for j, al in enumerate(ALPHAS):
                err[j] += ((Ridge(alpha=al).fit(a, t[itr]).predict(b) - t[ite]) ** 2).sum()
        alpha = float(ALPHAS[int(np.argmin(err))])
    return Ridge(alpha=alpha), alpha


def fold_fit(X, y, label, groups, tr, te, model_name, within_class):
    """One outer fold; targets, scaler and tuning use the outer-training rows only.

    Returns dict(target_tr, target_te, scaler_mean, alpha, coef, pred_te)."""
    if within_class:
        mu = class_means(y, label, tr)
        t_tr, t_te = residual_target(y[tr], label[tr], mu), residual_target(y[te], label[te], mu)
    else:
        t_tr, t_te = y[tr], y[te]
    sc = StandardScaler().fit(X[tr])
    alpha, coef = None, None
    if model_name == "ridge":
        m, alpha = fit_ridge_grouped(X[tr], t_tr, groups[tr])
        m.fit(sc.transform(X[tr]), t_tr)
        coef = m.coef_.copy()
    else:
        m = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.08, max_depth=3).fit(sc.transform(X[tr]), t_tr)
    return {"target_tr": t_tr, "target_te": t_te, "scaler_mean": sc.mean_.copy(), "alpha": alpha, "coef": coef,
            "pred_te": m.predict(sc.transform(X[te]))}


def oof_r2(X, y, label, groups, k, model_name, within_class=False):
    """Pooled out-of-fold R^2 over all held-out rows, each with its own fold's target."""
    pred, target = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        f = fold_fit(X, y, label, groups, tr, te, model_name, within_class)
        pred[te], target[te] = f["pred_te"], f["target_te"]
    return float(1 - ((target - pred) ** 2).sum() / ((target - target.mean()) ** 2).sum())


MODELS = ("ridge", "gb")


_INPUT_CACHE = {}


def audit(X_in, feats, n_frames, label, groups, cache_key=None):
    y = np.log(n_frames.astype(float))
    k = int(min(5, len(np.unique(groups))))
    out = {"n": int(len(y)), "n_subjects": int(len(np.unique(groups))), "k_folds": k}
    if k < 2 or len(np.unique(y)) < 2:
        return out
    label, groups = np.asarray(label), np.asarray(groups)
    for name, X in (("input", X_in), ("feats", feats)):
        if X is None:
            continue
        ck = (cache_key, len(y)) if (name == "input" and cache_key is not None) else None
        if ck is not None and ck in _INPUT_CACHE:      # the input summary does not depend on the model
            out.update(_INPUT_CACHE[ck])
            continue
        part = {}
        for mn in MODELS:
            part[f"r2_{name}_{mn}"] = oof_r2(X, y, label, groups, k, mn)
            part[f"r2_{name}_{mn}_within_class"] = oof_r2(X, y, label, groups, k, mn, within_class=True)
        out.update(part)
        if ck is not None:
            _INPUT_CACHE[ck] = part
    return out


def v9(base, out_dir):
    from safe_motion.data import hosts as H
    models = discover(base)
    length = {}
    lp = f"{base}/analysis/length.csv"
    if os.path.exists(lp):
        for r in csv.DictReader(open(lp)):
            length[(r["host"], r["mechanism"], r["strength"], r["world"], r["cell"], int(r["seed"]))] = r
    inputs = {}
    rows_out = []
    for key, dirs in sorted(models.items(), key=str):
        host, mech, st, world, cell, seed, perm = key
        if cell != "A_none" or perm is not None or not all(os.path.exists(f"{d}/features.npz") for d in dirs):
            continue
        if host not in inputs:
            rows = H.load_host(host)
            layout = H.LAYOUT[H.HOSTS[host]["dataset"]]
            ds = H.ClipDataset(rows, H.HOSTS[host]["window"], layout, train=False)
            inputs[host] = {r["identifier"]: summarise_input(ds[i][0]) for i, r in enumerate(rows)}
            print(f"{host}: {len(rows)} clips, input summary dim {len(next(iter(inputs[host].values())))}", flush=True)
        meta, _ = pool(dirs)
        z = pool_features(dirs)
        X_in = np.stack([inputs[host][i] for i in meta["identifier"]])
        res = audit(X_in, z["feats"], z["n_frames"], z["label"], z["subject"], cache_key=host)
        lr = length.get((host, mech or "", st or "", world or "real", cell, seed), {})
        rows_out.append({"host": host, "mechanism": mech or "", "strength": st or "", "world": world or "real",
                         "cell": cell, "seed": seed, **res,
                         "spearman_neg": lr.get("spearman_neg", ""), "spearman_pos": lr.get("spearman_pos", ""),
                         "ridge_r2_feats_analyze_v9": lr.get("ridge_r2_log_length", "")})
        print(f"  {host} {mech} {st} {world} seed {seed}: " + " ".join(f"{k}={v:.3f}" for k, v in res.items() if k.startswith("r2_")), flush=True)
    write(rows_out, f"{out_dir}/length_inputs.csv")
    return rows_out


def v7(features_dir, scores_dir, out_dir):
    import yaml
    from safe_motion.data import asdpose as D
    cfg = yaml.safe_load(open("configs/protocols/pilot_asdpose_v1.yaml"))
    rows = [r for r in D.load_index() if r["n_frames"] >= 200 and r["split"] == "test"]
    ds = D.SegmentDataset(rows, window=200, train=False, normalize="image", confidence=True)
    inputs = {r["identifier"]: summarise_input(ds[i][0]) for i, r in enumerate(rows)}
    print(f"asdpose: {len(rows)} test segments, input summary dim {len(next(iter(inputs.values())))}", flush=True)
    length = {}
    lp = "results/v7/analysis/length.csv"
    if os.path.exists(lp):
        for r in csv.DictReader(open(lp)):
            length[r["model"]] = r
    rows_out = []
    import glob
    for f in sorted(glob.glob(f"{features_dir}/*.npz")):
        name = os.path.basename(f)[:-4]
        if not (name.startswith("v7_real_") or "_A_none_" in name):
            continue
        z = np.load(f)
        sc = list(csv.DictReader(open(f"{scores_dir}/{name}.csv")))
        ids = [r["identifier"] for r in sc]
        assert len(ids) == len(z["n_frames"]) and all(int(r["n_frames"]) == n for r, n in zip(sc, z["n_frames"]))
        X_in = np.stack([inputs[i] for i in ids])
        res = audit(X_in, z["feats"], z["n_frames"], z["label"], z["child"], cache_key="real" if "_real_" in name else "host")
        lr = length.get(name, {})
        rows_out.append({"host": "asdpose_real" if "_real_" in name else "asdpose_host", "model": name, **res,
                         "spearman_neg": lr.get("spearman_neg", ""), "spearman_pos": lr.get("spearman_pos", ""),
                         "ridge_r2_feats_analyze_v7": lr.get("ridge_r2_log_length", "")})
        print(f"  {name}: " + " ".join(f"{k}={v:.3f}" for k, v in res.items() if k.startswith("r2_")), flush=True)
    write(rows_out, f"{out_dir}/length_inputs.csv")
    return rows_out


def write(rows, path):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print("wrote", path)


def write_md(v9_rows, v7_rows, path):
    def summarise(rows, by):
        g = defaultdict(list)
        for r in rows:
            g[by(r)].append(r)
        lines = []
        for k, rs in sorted(g.items()):
            def m(col):
                v = [r[col] for r in rs if col in r]
                return f"{np.mean(v):.2f} ({min(v):.2f} to {max(v):.2f})" if v else ""
            lines.append(f"| {k} | {len(rs)} | {rs[0].get('n_subjects', '')} | {m('r2_input_ridge')} | {m('r2_input_gb')} | "
                         f"{m('r2_input_gb_within_class')} | {m('r2_feats_ridge')} | {m('r2_feats_gb')} | {m('r2_feats_gb_within_class')} |")
        return lines
    hdr = ["| group | models | subjects | input ridge R² | input GB R² | input GB R² within class | feats ridge R² | feats GB R² | feats GB R² within class |",
           "|---|---|---|---|---|---|---|---|---|"]
    lines = ["# Length predictability from the model input and from the penultimate features", "",
             "Generated by `experiments/length_audit_inputs.py`. Out-of-fold R² of log(original length), "
             "subject-grouped folds (min(5, subjects)). 'within class': target centred on the training-fold "
             "class mean, so it measures length information beyond the label. Mean (min to max) over models.", "",
             "## ASDPose (v7 real cells on the 3,834-segment test split; planted A_none cells on the 1,966-segment host test split)", ""] + hdr + \
        summarise(v7_rows, lambda r: r["host"] + " " + ("real" if "_real_" in r["model"] else r["model"].split("_A_none")[0][5:])) + \
        ["", "## v9 hosts (out-of-fold over every subject)", ""] + hdr + \
        summarise(v9_rows, lambda r: f"{r['host']} {r['world']} {r['mechanism']} {r['strength']}".strip())
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-v7", action="store_true")
    ap.add_argument("--skip-v9", action="store_true")
    args = ap.parse_args()
    v9_rows = [] if args.skip_v9 else v9("results/v9", "results/v9/analysis")
    v7_rows = [] if args.skip_v7 else v7("results/v7/features", "results/v7/scores", "results/v7/analysis")
    write_md(v9_rows, v7_rows, "results/v9/analysis/length_inputs.md")


if __name__ == "__main__":
    main()
