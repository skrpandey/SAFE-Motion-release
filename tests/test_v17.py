"""Protocol v17 pieces that need no training: queue, protocol wiring, scorer names, regenerated
training assignments, and the analysis on synthetic scores."""
import csv
import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from safe_motion import planted as P                   # noqa: E402

YAML = os.path.join(ROOT, "configs/protocols/confonly_assignment_v17.yaml")


def test_v17_queue_lines():
    import make_queue_v17 as Q
    ls = Q.lines()
    assert len(ls) == 18
    assert all("--proto v17" in l and "--cell A_none" in l and "--arch compact" in l for l in ls)
    ws = sorted(int(l.split("--world-seed ")[1].split()[0]) for l in ls)
    assert ws == [1] * 6 + [2] * 6 + [3] * 6
    assert sum("--world S" in l for l in ls) == 9 and sum("--world I" in l for l in ls) == 9
    assert len(set(ls)) == 18


def test_v17_protocol_wiring():
    import train_asdpose as T
    assert T.PROTOS["v17"].endswith("confonly_assignment_v17.yaml") and "v17" in T.LENGTH_PROTOS
    assert T.V17_CELLS["A_none"]["valid_rule"] == "coords" and T.V17_CELLS["A_none"]["arch"] == "compact"
    assert "remedy" not in T.V17_CELLS["A_none"]
    y = yaml.safe_load(open(YAML))
    assert y["frozen"] and y["outputs"]["cache_root"] == "cache/v17" and y["runs_total"] == 18
    assert y["world_seeds"] == [1, 2, 3] and y["valid_rule"] == "coords" and y["training_recipe"] == "v15"
    assert y["val_exclude_ids"] == [322]


def test_scorer_regex_accepts_v16_and_v17():
    import score_v16 as S
    m = S.NAME.match("pilot_v17_S_w3_armruns_mid_A_none_s2")
    assert m and m.group(1) == "v17" and m.group(3) == "3"
    m = S.NAME.match("pilot_v16_I_w0_armruns_mid_C_groupdro_obs_s0")
    assert m and m.group(1) == "v16"
    assert S.NAME.match("pilot_v15conf_S_w0_armruns_mid_A_none_s0") is None


def test_scorer_registry_filters_by_protocol(tmp_path):
    import score_v16 as S
    for stem in ("pilot_v16_S_w0_armruns_mid_A_none_s3", "pilot_v17_S_w1_armruns_mid_A_none_s0",
                 "pilot_v17_I_w2_armruns_mid_A_none_s1"):
        open(tmp_path / f"{stem}.pt", "wb").close()
    r16, r17 = S.registry(str(tmp_path), "v16"), S.registry(str(tmp_path), "v17")
    assert [m["name"] for m in r16] == ["v16_S_w0_armruns_mid_A_none_s3"] and r16[0]["world_seed"] == 0
    assert sorted(m["world_seed"] for m in r17) == [1, 2] and all(m["proto"] == "v17" for m in r17)


def test_regenerated_assignments_differ_and_are_deterministic():
    rows = [{"identifier": f"id{i}", "binary_label": i % 2, "n_frames": 60} for i in range(40)]
    m1, e1 = P.build_masks(rows, "S", "mid", 1, mechanism="armruns")
    m1b, e1b = P.build_masks(rows, "S", "mid", 1, mechanism="armruns")
    m2, e2 = P.build_masks(rows, "S", "mid", 2, mechanism="armruns")
    assert np.allclose(e1, e1b) and all(np.array_equal(m1[k], m1b[k]) for k in m1)
    assert not np.allclose(e1, e2) and any(not np.array_equal(m1[k], m2[k]) for k in m1)
    _, eI1 = P.build_masks(rows, "I", "mid", 1, mechanism="armruns")
    assert not np.allclose(e1, eI1)                                   # S and I draws are independent
    assert np.allclose(P.severities(rows, "S", "mid", "armruns1"), e1)  # the armruns tag, not the runs tag
    assert not np.allclose(P.severities(rows, "S", "mid", 1), e1)


def _write_scores(path, ids, child, y, cols):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["identifier", "child_id", "label", "n_frames", "qc_pass"] + list(cols))
        for i in range(len(ids)):
            w.writerow([ids[i], child[i], y[i], 250, 1] + [f"{cols[c][i]:.6f}" for c in cols])


def test_analyze_v17_on_synthetic(tmp_path):
    import analyze_v17 as AN
    rng = np.random.RandomState(0)
    n, ids, child, y = 240, [f"s{i}" for i in range(240)], rng.randint(0, 16, 240), rng.randint(0, 2, 240)

    def scores(strength_s):
        base = rng.rand(n) * 0.6 + 0.2 * y
        cols = {"clean": base}
        for mech in ("runs", "whole", "frozen", "armruns"):
            for ms in (2000, 3000, 4000):
                cols[f"{mech}_S_mid_m{ms}"] = np.clip(base - strength_s * (1 - y) * 0.15 + rng.rand(n) * 0.05, 0, 1)
                cols[f"{mech}_I_mid_m{ms}"] = np.clip(base + rng.rand(n) * 0.05, 0, 1)
        return cols

    def write_set(d, proto, world_seeds):
        (d / "scores").mkdir(parents=True)
        reg = []
        for ws in world_seeds:
            for w in ("S", "I"):
                for k in range(3):
                    name = f"{proto}_{w}_w{ws}_armruns_mid_A_none_s{k}"
                    _write_scores(d / "scores" / f"{name}.csv", ids, child, y, scores(1.0 if w == "S" else 0.0))
                    reg.append({"name": name, "proto": proto, "arch": "compact", "world": w, "world_seed": ws,
                                "mechanism": "armruns", "strength": "mid", "cell": "A_none", "seed": k})
        json.dump(reg, open(d / "registry.json", "w"))

    write_set(tmp_path / "v17", "v17", (1, 2, 3))
    write_set(tmp_path / "v15", "v15conf", (0,))
    out = tmp_path / "analysis"
    crit = AN.main(["--v17", str(tmp_path / "v17"), "--v15", str(tmp_path / "v15"), "--epochs", str(tmp_path / "none"),
                    "--out", str(out), "--n-boot", "20"])
    assert set(crit["P17_1_per_assignment"]) == {"w0", "w1", "w2", "w3", "pooled"}
    assert len(crit["P17_1_per_assignment"]["pooled"]["models"]) == 18
    for k in ("w0", "w1", "w2", "w3", "pooled"):
        assert crit["P17_1_per_assignment"][k]["mwi_logit"][0] > 0        # the planted cue in World S
    sp = crit["D17_2_assignment_spread"]["spread"]["world seeds 1-3 (logit)"]
    assert sp["n"] == 3 and sp["min"] <= sp["mean"] <= sp["max"]
    assert set(crit["D17_2_assignment_spread"]["minus_w0"]) == {"w1", "w2", "w3"}
    rows = [r for r in csv.DictReader(l for l in open(out / "effects.csv") if not l.startswith("#"))]
    assert sum(r["quantity"].startswith("MWI (pooled)") for r in rows) == 3
    assert any(r["n_seeds"] == "9/9" for r in rows)
    # a missing assignment is an error, not a silent drop
    import pytest
    with pytest.raises(SystemExit):
        AN.main(["--v17", str(tmp_path / "v15"), "--v15", str(tmp_path / "v15"), "--out", str(out), "--n-boot", "5"])
