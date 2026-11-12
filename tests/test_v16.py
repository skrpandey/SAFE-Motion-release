"""Protocol v16 pieces that need no training: observed Group DRO groups, ST-GCN, scorer loading."""
import inspect
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from safe_motion import remedies as RM                 # noqa: E402
from safe_motion import planted as P                   # noqa: E402
from safe_motion.models.baselines import STGCN         # noqa: E402


def _fake_cache(tmp, n=12, t=40, seed=0):
    rng = np.random.RandomState(seed)
    rows, chunks, off = [], [], 0
    for i in range(n):
        arr = rng.rand(t, 17, 3).astype(np.float32)
        arr[..., 2] = 0.9                            # confident everywhere before planting
        chunks.append(arr.ravel())
        rows.append({"identifier": f"id{i}", "binary_label": i % 2, "n_frames": t, "byte_offset": off})
        off += arr.size * 4
    np.concatenate(chunks).tofile(os.path.join(tmp, "keypoints.f32"))
    return rows


def test_observed_groups_use_the_input_only(tmp_path):
    assert "eps" not in inspect.signature(RM.observed_groups).parameters
    rows = _fake_cache(str(tmp_path))
    masks = {}
    for i, r in enumerate(rows):                     # planted arm frames grow with the row index
        m = np.zeros((r["n_frames"], 17), bool)
        m[: i * 3, P.ARM_JOINTS] = True
        masks[r["identifier"]] = m
    groups, agree = RM.observed_groups(rows, masks, "confscale", n_train=8, cache_dir=str(tmp_path))
    frac = np.array([min(i * 3, 40) / 40 for i in range(len(rows))])
    med = np.median(frac[:8])
    for i, r in enumerate(rows):
        assert groups[r["identifier"]] == 2 * r["binary_label"] + int(frac[i] > med)
    assert np.isnan(agree)
    same = {k: v for k, v in groups.items()}
    _, agree2 = RM.observed_groups(rows, masks, "confscale", n_train=8, oracle=same, cache_dir=str(tmp_path))
    assert agree2 == 1.0


def test_stgcn_reads_three_channels():
    net = STGCN(c_in=3).eval()
    with torch.no_grad():
        out = net(torch.randn(2, 3, 200, 17))
    assert out.shape == (2, 2)
    assert 3.0e6 < sum(p.numel() for p in net.parameters()) < 3.2e6


def test_scorer_builds_stgcn_from_checkpoint(tmp_path):
    import score_v7 as S7
    net = STGCN(c_in=3)
    cell = {"velocity": False, "joint_dropout": 0.0, "coord_dropout": 0.0, "pool": "full",
            "valid_rule": "coords", "arch": "stgcn"}
    path = str(tmp_path / "pilot_v16_S_w0_armruns_mid_A_none_stgcn_s0.pt")
    torch.save({"state_dict": net.state_dict(), "cell": cell}, path)
    built, prep = S7.build({"path": path, "arch": "compact"})   # the cell's arch wins
    assert isinstance(built, STGCN) and prep["valid_rule"] == "coords" and prep["confidence"]
