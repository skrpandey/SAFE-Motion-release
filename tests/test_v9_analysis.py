"""Fast tests for the v9 CPU analyses: the two-way variance decomposition and the v9 table
generator on a synthetic analysis directory."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analyze_v9_variance import decompose, Cell, analyse_cell   # noqa: E402
from analyze_v7 import RowSet                                  # noqa: E402
import make_tables_v9 as T                                      # noqa: E402


def test_decompose_additive_grid():
    seeds, masks = np.array([0.0, 0.3, 0.6]), np.array([0.0, -0.1, 0.1])
    g = seeds[:, None] + masks[None, :]
    d = decompose(g)
    assert abs(d["grid_mean"] - 0.3) < 1e-9
    assert abs(d["sd_seed"] - seeds.std(ddof=1)) < 1e-9
    assert abs(d["sd_mask"] - masks.std(ddof=1)) < 1e-9
    assert d["sd_resid"] < 1e-9 and abs(d["share_seed"] + d["share_mask"] - 1) < 1e-9
    assert abs(d["range_mask"] - 0.2) < 1e-9 and abs(d["range_seed"] - 0.6) < 1e-9


def test_decompose_constant_grid_has_nan_shares():
    d = decompose(np.full((3, 3), 0.5))
    assert np.isnan(d["share_seed"]) and d["sd_seed"] == 0


def _synthetic_cell(n_subjects=6, per=20, seed=0):
    rng = np.random.RandomState(seed)
    n = n_subjects * per
    subj = np.repeat(np.arange(n_subjects), per).astype(str)
    y = rng.randint(0, 2, n)
    rs = RowSet([f"id{i}" for i in range(n)], y, subj, n_boot=50)
    cell = Cell("synthetic", "runs", "mid", rs)
    for world in ("S", "I"):
        for s in (0, 1, 2):
            cols = {}
            for ms in (2000, 3000, 4000):
                base = y + rng.normal(0, 1.0, n)
                shift = 0.8 if world == "S" else 0.0            # World S model scores better under S planting
                cols[f"runs_S_mid_m{ms}"] = base + shift * y + rng.normal(0, 0.05, n)
                cols[f"runs_I_mid_m{ms}"] = base + rng.normal(0, 0.05, n)
            cell.add(world, s, cols, np.arange(n))
    return cell


def test_analyse_cell_reproduces_pooled_point_and_loso():
    cell = _synthetic_cell()
    assert cell.complete()
    rows = analyse_cell(cell, "subjects")
    se = next(r for r in rows if r["quantity"] == "SE")
    rS = next(r for r in rows if r["quantity"] == "R_S")
    rI = next(r for r in rows if r["quantity"] == "R_I")
    assert abs(se["point_m2000"] - (rS["point_m2000"] - rI["point_m2000"])) < 1e-9
    grid = json.loads(se["grid_json"])
    assert len(grid) == 9 and abs(np.mean([grid[f"s{s}_m2000"] for s in (0, 1, 2)]) - se["point_m2000"]) < 1e-3
    assert se["point_m2000"] > 0.3 and se["ci_lo"] <= se["point_m2000"] <= se["ci_hi"]
    assert se["loso_min"] <= se["point_m2000"] + 1e-9 or se["loso_max"] >= se["point_m2000"] - 1e-9
    assert se["n_subjects"] == 6 and se["loso_range"] >= 0


def test_make_tables_v9_on_synthetic(tmp_path):
    a = tmp_path / "analysis"
    a.mkdir()
    crit = {"gates": {"mmasd_chest_squat": {"clean_auc_min": 0.65, "clean_auc_oof": 0.7, "clean_auc_pass": True,
                                             "quality_only_auc_oof": 0.5, "length_auc_oof": 0.6, "n_subjects": 11, "n_clips": 150}},
            "replicates": {"mmasd_chest_squat": False},
            "R1_se_positive": {"mmasd_chest_squat": {"pass": False, "per_mechanism": {
                "runs": {"SE": [0.3, 0.1, 0.9], "null_p95": 0.4, "null_p": 0.2, "pass": False}}}},
            "R2_scaling": {"mmasd_chest_squat": {"diff": [0.3, 0.0, 1.0], "pass": True}},
            "R3_dropout_does_not_remove": {"mmasd_chest_squat": {"pass": True, "per_mechanism": {"runs": {"diff": [-0.2, -0.5, 0.1], "pass": True}}}},
            "R4_dropout_raises_somewhere": {"mmasd_chest_squat": {"count_raises": 0, "n_mechanisms": 1, "asdpose_count": "3 of 4"}},
            "R5_reference": {"mmasd_chest_squat": {"runs": {"R_I": 0.26, "R_I_ci": [0.1, 0.5]}}},
            "R6_length": {"all_not_exploited": False, "flagged": ["mmasd_chest_squat/I/runs/A_none/s0"]}}
    (a / "criteria.json").write_text(json.dumps(crit))
    (a / "effects.csv").write_text("host,mechanism,strength,contrast,point,ci_lo,ci_hi,verdict,R_S,R_I,R_I_ci,sd_seeds_S,sd_seeds_I,mask_range,null_n,null_p95,null_p,above_null\n"
                                   "mmasd_chest_squat,runs,mid,SE(A_none),0.3,0.1,0.9,differs_from_zero,0.6,0.3,,0.05,0.1,0.08,20,0.4,0.2,False\n"
                                   "mmasd_chest_squat,runs,mid,SE(A_dropout) - SE(A_none),-0.2,-0.5,0.1,undetermined,0.5,0.4,,,,,,,,\n")
    (a / "crossed.csv").write_text("host,trained_world,trained_mechanism,test_mechanism,R,ci_lo,ci_hi,sd_seeds,verdict\n"
                                   "mmasd_chest_squat,I,runs,runs,0.26,0.1,0.5,0.1,differs_from_zero\n")
    (a / "real.csv").write_text("host,cell,mechanism_tested,R,ci_lo,ci_hi,sd_seeds,reference_R_I,reference_ci,subjects_opposite_sign,n_subjects\n"
                                "mmasd_chest_squat,A_none,runs,-0.18,-0.46,0.03,0.17,0.26,\"(0.1, 0.5)\",6,11\n")
    (a / "null.csv").write_text("host,mechanism,strength,perm,SE_perm\n" + "".join(f"mmasd_chest_squat,runs,mid,{k},{0.1 * (k % 5) - 0.2}\n" for k in range(20)))
    out = tmp_path / "tables"
    T.gates(crit, str(out))
    T.verdicts(crit, str(out))
    T.effects(str(a), str(out))
    T.crossed(str(a), str(out))
    T.real(str(a), str(out))
    T.null(str(a), str(out))
    for n in ("v9_gates", "v9_verdicts", "v9_effects", "v9_crossed", "v9_real", "v9_null"):
        txt = (out / f"{n}.tex").read_text()
        assert txt.startswith("% generated") and r"\bottomrule" in txt
    assert "+0.300 [+0.100, +0.900]" in (out / "v9_effects.tex").read_text()
    assert "failed" not in (out / "v9_gates.tex").read_text()
