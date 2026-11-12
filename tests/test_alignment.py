"""Score-file alignment is invariant to the row order of the file."""
import csv
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from analyze_v7 import RowSet, align, load_scores  # noqa: E402


def _rows(n=60, seed=0):
    rng = np.random.RandomState(seed)
    ids = [f"seg{i:03d}" for i in range(n)]
    child = [f"c{i % 7}" for i in range(n)]
    y = rng.randint(0, 2, n)
    cols = {"clean": rng.rand(n), "runs_S_mid_m2000": rng.rand(n)}
    return ids, child, y, cols


def _write(path, ids, child, y, cols, order):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["identifier", "child_id", "label", "n_frames", "qc_pass"] + list(cols))
        for k in order:
            w.writerow([ids[k], child[k], int(y[k]), 250, 1] + [f"{cols[c][k]:.6f}" for c in cols])


def test_three_cycle_shuffle_leaves_aucs_unchanged(tmp_path):
    ids, child, y, cols = _rows()
    rs = RowSet(ids, y, child, 50)
    n = len(ids)
    ref = {c: rs.aucs(v) for c, v in cols.items()}
    order = np.arange(n)
    order[[0, 1, 2]] = order[[1, 2, 0]]            # a three-cycle: a permutation that is not its own inverse
    order[[10, 40]] = order[[40, 10]]
    p = tmp_path / "shuffled.csv"
    _write(p, ids, child, y, cols, order)
    meta, sc = load_scores(str(p))
    got = align(rs, meta["identifier"], meta["label"], sc)
    for c in cols:
        assert np.allclose(got[c], cols[c], atol=1e-6)
        assert np.allclose(rs.aucs(got[c]), ref[c])
    # a plain gather `s[pos(id_k)]` gives the wrong order for this permutation
    gather = {c: sc[c][[rs.pos[i] for i in meta["identifier"]]] for c in cols}
    assert not np.allclose(gather["clean"], cols["clean"], atol=1e-6)


def test_align_rejects_mismatches(tmp_path):
    ids, child, y, cols = _rows()
    rs = RowSet(ids, y, child, 10)
    with pytest.raises(ValueError):
        align(rs, ids[:-1], y[:-1], {c: v[:-1] for c, v in cols.items()})
    bad = y.copy(); bad[0] = 1 - bad[0]
    with pytest.raises(ValueError):
        align(rs, ids, bad, cols)
    dup = list(ids); dup[1] = dup[0]
    with pytest.raises(ValueError):
        align(rs, dup, y, cols)
