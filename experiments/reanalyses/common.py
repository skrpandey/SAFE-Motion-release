"""Shared loaders and estimators for the re-analyses of the revised article (2026-11-16).

Nothing here re-implements the published estimator. The identifier-clustered resamples, the AUROC
and the logit come from experiments/analyze_operators.py (RowSet, load_scores, align, logit) and
safe_motion/metrics.py; the test-mask seeds from experiments/analyze_operators_masks.py. A cell's MWI is
the mean over seeds and test masks of the per-model logit response R = logit AUROC(policy S) -
logit AUROC(policy I) of World S minus the same mean of World I, computed inside each of the 2,000
shared resamples (resample 0 is the observed sample); raw-scale MWI averages AUROC differences.
This is the construction of experiments/analyze_confonly_extension.py, analyze_confonly_assignments.py, analyze_confonly_historical.py and
cluster_scenarios.py.

Score files (per-segment test scores, cleaned selection; read only):
  historical recipe, corrected rule (v13)       results/confonly_historical/scores
  corrected recipe, assignment 0 (v15conf)      results/confonly_primary/scores
  nine-seed extension seeds 3-8, ST-GCN (v16)   results/confonly_extension/scores
  regenerated assignments 1-3 (v17)             results/confonly_assignments/scores
  first-rule destructive cells (v7, v8)         results/operators/scores (cleaned selection)

Outputs are written to results/reanalyses/ (set SM_REVISION_OUT to write them elsewhere, e.g. to
compare a rerun with the released files). The scripts that read the SAFE-SMM segment cache
(regen_test_planting.py, r22_natural_rate.py, r24_r17_counts.py; r23_cue_only_reference.py reads
its index) find it through SM_SMM_ROOT or SM_ASDPOSE_CACHE (safe_motion/data/asdpose.py). Run order: step0_validate.py,
regen_test_planting.py (writes test_planted_severities.csv, read by r20, r22 and r23), then the
r*.py scripts in any order.

Run every script from the repository root with python3.
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if Path.cwd().resolve() != ROOT.resolve():
    os.chdir(ROOT)

from safe_motion import metrics as M                                    # noqa: E402
from experiments.analyze_operators import RowSet, load_scores, logit           # noqa: E402
from experiments.analyze_operators_masks import MASK_SEEDS                     # noqa: E402

OUT = ROOT / os.environ.get("SM_REVISION_OUT", "results/reanalyses")
NB = 2000
MECH = "armruns"

SRC = {
    "v13": "results/confonly_historical",
    "v15conf": "results/confonly_primary",
    "v16a": "results/confonly_extension",
    "v16bc": "results/confonly_extension",
    "v17": "results/confonly_assignments",
    "v20": "results/recipe_comparison",
    "v7": "results/operators",
}

# matched test identifiers of the identity audit (results/identity_audit, pass 1 + pass 2); checked
# against configs/protocols/cluster_scenarios_post_hoc.yaml in r18_verified_persons.py
CLUSTER_CFG = "configs/protocols/cluster_scenarios_post_hoc.yaml"


def _registry(key):
    path = SRC[key]
    reg = json.load(open(f"{path}/registry.json"))
    out = []
    for m in reg:
        p = f"{path}/scores/{m['name']}.csv"
        if os.path.exists(p):
            out.append(dict(m, src=key, path=p))
    return out


def models(key, cell, seeds, world_seed=0, mechanism=MECH, strength="mid", proto=None):
    """{'S': [...], 'I': [...]} model records of one cell, sorted by seed; checks the seed list."""
    reg = _registry(key)
    g = [m for m in reg if m["cell"] == cell and m.get("seed") in seeds and m.get("world") in ("S", "I")
         and int(m.get("world_seed") or 0) == world_seed and m.get("mechanism") == mechanism
         and m.get("strength") == strength and (proto is None or m.get("proto") == proto)]
    out = {w: sorted([m for m in g if m["world"] == w], key=lambda m: m["seed"]) for w in ("S", "I")}
    for w in ("S", "I"):
        got = [m["seed"] for m in out[w]]
        if got != sorted(seeds):
            raise SystemExit(f"{key} {cell} ws{world_seed} {mechanism} {strength} world {w}: seeds {got}, expected {sorted(seeds)}")
    return out


def confonly_cells():
    """The confidence-only no-remedy cells (mechanism armruns, mid strength)."""
    C = {
        "hist3": models("v13", "A_none", (0, 1, 2), proto="v13"),
        "corr3": models("v15conf", "A_none", (0, 1, 2)),
        "stgcn": models("v16a", "A_none_stgcn", (0, 1, 2)),
    }
    extra = models("v16bc", "A_none", tuple(range(3, 9)))
    C["corr9"] = {w: C["corr3"][w] + extra[w] for w in ("S", "I")}
    for ws in (1, 2, 3):
        C[f"w{ws}"] = models("v17", "A_none", (0, 1, 2), world_seed=ws)
    C["w123"] = {w: sum((C[f"w{k}"][w] for k in (1, 2, 3)), []) for w in ("S", "I")}
    return C


def table2_cells():
    """The first-rule destructive no-remedy cells of Table 2 (cleaned selection) and the
    historical confidence-only cell; (cell key) -> (models, mechanism, strength)."""
    return {
        "missing runs, mid": (models("v7", "A_none", (0, 1, 2), mechanism="runs", strength="mid", proto="v7"), "runs", "mid"),
        "missing runs, low": (models("v7", "A_none", (0, 1, 2), mechanism="runs", strength="low", proto="v7"), "runs", "low"),
        "whole-joint absence, mid": (models("v7", "A_none", (0, 1, 2), mechanism="whole", strength="mid", proto="v7"), "whole", "mid"),
        "frozen frames, mid": (models("v7", "A_none", (0, 1, 2), mechanism="frozen", strength="mid", proto="v8"), "frozen", "mid"),
        "confidence-only (historical), mid": (models("v13", "A_none", (0, 1, 2), proto="v13"), "armruns", "mid"),
    }


class Scores:
    """Score files of a set of models, aligned to one row order (the identifiers of the first file)."""

    def __init__(self, recs):
        self.cols = {}
        ids = y = child = None
        for m in recs:
            if m["name"] in self.cols:
                continue
            meta, sc = load_scores(m["path"])
            if ids is None:
                ids = list(meta["identifier"])
                y = np.array([int(v) for v in meta["label"]])
                child = np.array([str(c) for c in meta["child_id"]])
                self.n_frames = np.array([int(v) for v in meta["n_frames"]])
                pos = {i: k for k, i in enumerate(ids)}
            if set(meta["identifier"]) != set(ids) or len(meta["identifier"]) != len(ids):
                raise SystemExit(f"{m['path']}: rows differ from the first score file")
            order = np.array([pos[i] for i in meta["identifier"]])
            lab = np.array([int(v) for v in meta["label"]])
            if not np.array_equal(y[order], lab):
                raise SystemExit(f"{m['path']}: labels differ")
            self.cols[m["name"]] = {}
            for c, s in sc.items():
                a = np.empty(len(ids))
                a[order] = s
                self.cols[m["name"]][c] = a
        self.ids, self.y, self.child = ids, y, child
        self.pos = {i: k for k, i in enumerate(ids)}

    def col(self, name, c):
        return self.cols[name][c]


class Engine:
    """Identifier-clustered resamples of a row subset (RowSet of experiments/analyze_operators.py, seed 0)
    and cached bootstrap AUROC arrays (index 0 = observed sample)."""

    def __init__(self, scores, sel=None, clusters=None, n_boot=NB):
        self.sc = scores
        self.sel = np.arange(len(scores.ids)) if sel is None else np.asarray(sel)
        cl = scores.child[self.sel] if clusters is None else np.asarray(clusters)
        self.rs = RowSet([scores.ids[k] for k in self.sel], scores.y[self.sel], cl, n_boot)
        self.n_clusters = len(set(cl))
        self.cache = {}

    def aucs(self, name, c):
        key = (name, c)
        if key not in self.cache:
            self.cache[key] = self.rs.aucs(self.sc.col(name, c)[self.sel])
        return self.cache[key]

    def R(self, m, mech=MECH, st="mid", masks=MASK_SEEDS, scale="logit"):
        """(n_masks, NB+1) response of one model."""
        f = logit if scale == "logit" else (lambda a: a)
        return np.stack([f(self.aucs(m["name"], f"{mech}_S_{st}_m{ms}")) - f(self.aucs(m["name"], f"{mech}_I_{st}_m{ms}"))
                         for ms in masks])

    def cell(self, g, mech=MECH, st="mid", masks=MASK_SEEDS, scale="logit"):
        """MWI, R(f_S), R(f_I) arrays (NB+1) and the per-model (n_seeds, n_masks, NB+1) arrays."""
        aS = np.stack([self.R(m, mech, st, masks, scale) for m in g["S"]])
        aI = np.stack([self.R(m, mech, st, masks, scale) for m in g["I"]])
        rS, rI = aS.mean((0, 1)), aI.mean((0, 1))
        return rS - rI, rS, rI, aS, aI


def point_auc(s, y):
    return M.auc(s, y)


def ci(a):
    """Observed value and 95% percentile interval of a (NB+1) array."""
    lo, hi = M.ci(a[1:])
    return float(a[0]), lo, hi


def fmt(a, d=3):
    p, lo, hi = ci(a) if not isinstance(a, tuple) else a
    return f"{p:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def _rel(path):
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def write_csv(name, rows, header_comment=None):
    path = OUT / name
    OUT.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        if header_comment:
            for line in header_comment.strip().splitlines():
                f.write(f"# {line}\n")
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})
    print("wrote", _rel(path))
    return path


def write_json(name, obj):
    path = OUT / name
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1, default=float)
    print("wrote", _rel(path))


def test_host_rows():
    """The planted-host test rows exactly as the scorers build them (score_confonly_historical.py, score_confonly_corrected.py):
    length protocol (>= 200 frames), test split, QC filter of pilot_1.yaml."""
    import yaml
    from safe_motion.data import asdpose as D
    v1 = yaml.safe_load(open("configs/protocols/pilot_1.yaml"))
    rows = [r for r in D.load_index() if r["n_frames"] >= 200]
    return [r for r in rows if r["split"] == "test" and D.passes_qc(r, v1["qc_filter"])]


def dev_host_rows():
    """Optimization training and validation rows of the planted host, as experiments/train_asdpose.py
    builds them (length protocol, subject-level validation split seed 0, QC in both)."""
    import yaml
    from safe_motion.data import asdpose as D
    v1 = yaml.safe_load(open("configs/protocols/pilot_1.yaml"))
    rows = [r for r in D.load_index() if r["n_frames"] >= 200]
    tr, va, _ = D.subject_level_val_split(rows, 0.2, seed=0)
    tr = [r for r in tr if D.passes_qc(r, v1["qc_filter"])]
    va = [r for r in va if D.passes_qc(r, v1["qc_filter"])]
    return tr, va


def keypoints():
    """Memory-mapped SAFE-SMM keypoint cache (read only)."""
    from safe_motion.data import asdpose as D
    return np.memmap(f"{D.CACHE_DIR}/keypoints.f32", dtype=np.float32, mode="r")


def segment(data, r):
    start = r["byte_offset"] // 4
    return np.array(data[start:start + r["n_frames"] * 51], dtype=np.float32).reshape(r["n_frames"], 17, 3)
