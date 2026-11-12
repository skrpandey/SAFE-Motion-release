"""Protocol v12 (configs/protocols/remedies_v12.yaml): remedy components and a tiny fit per cell. CPU, seconds."""
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from safe_motion import remedies as RM            # noqa: E402
from safe_motion.data import asdpose as D         # noqa: E402

HAVE_DATA = os.path.exists(os.path.join(D.CACHE_DIR, "index.csv"))


def test_severity_groups():
    rows = [{"identifier": f"s{i}", "binary_label": i % 2} for i in range(8)]
    eps = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.7, 0.6])
    g = RM.severity_groups(rows, eps, n_train=4)           # median of the first 4 = 0.25
    assert [g[f"s{i}"] for i in range(8)] == [0, 2, 1, 3, 1, 3, 1, 3]


def test_gradient_reversal_sign_and_scale():
    x = torch.ones(3, requires_grad=True)
    RM.grad_reverse(x, 0.3).sum().backward()
    assert torch.allclose(x.grad, torch.full((3,), -0.3))


def test_groupdro_upweights_the_worst_group():
    dro = RM.GroupDRO(step=0.5)
    per = torch.tensor([0.1, 0.1, 2.0, 2.0])
    for _ in range(5):
        dro.loss(per, torch.tensor([0, 0, 1, 1]))
    assert dro.q[1] > dro.q[0] and abs(float(dro.q.sum()) - 1) < 1e-6


def test_quality_stats_definitions():
    a = np.ones((4, 17, 3), np.float32)
    a[1:, :, :2] = a[:1, :, :2]                             # every frame frozen after the first
    a[0, 5, 2] = 0.0                                        # one missing arm joint-frame
    st = RM.quality_stats(a)
    assert st[1] == pytest.approx(1 / 68) and st[2] == pytest.approx(1.0) and st[3] == pytest.approx(1 / 24)


def test_world_I_severity_is_label_free():
    rng = np.random.RandomState(0)
    e = np.array([RM.world_I_severity(rng, 0.3) for _ in range(4000)])
    assert 0.2 < e.mean() < 0.35                            # between the SMM (0.2) and non-SMM (0.33) means


@pytest.mark.skipif(not HAVE_DATA, reason="ASDPose segment cache not present (set SM_CACHE_DIR)")
def test_augmentation_is_label_independent_and_changes_inputs():
    rows = [r for r in D.load_index() if r["n_frames"] >= 200 and r["split"] == "train"][:120]
    ds = RM.RemedyDataset(rows, augment=True, pi=0.3, window=200, train=True, normalize="image")
    changed, labels = [], []
    for i, r in enumerate(rows):
        base = ds.load_segment(r)
        aug = ds.augment_segment(base.copy())
        changed.append(not np.array_equal(base, aug))
        labels.append(r["binary_label"])
    changed, labels = np.array(changed), np.array(labels)
    assert 0.3 < changed.mean() < 0.7                       # probability 0.5 per window
    if labels.min() != labels.max():
        assert abs(changed[labels == 1].mean() - changed[labels == 0].mean()) < 0.35
    x, y, g, st = ds[0]
    assert x.shape[0] == 3 and g == -1 and st.shape == (4,)


@pytest.mark.skipif(not HAVE_DATA, reason="ASDPose segment cache not present (set SM_CACHE_DIR)")
@pytest.mark.parametrize("cell", ["C_groupdro", "C_adversarial", "C_augment"])
def test_tiny_fit(cell, tmp_path):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "experiments/train_asdpose.py", "--proto", "v12", "--cell", cell,
                        "--world", "I", "--mechanism", "runs", "--strength", "mid", "--arch", "tcn",
                        "--seed", "7", "--smoke", "--workers", "1"], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    out = os.path.join(ROOT, f"results/v12_smoke/I_w0_runs_mid/{cell}/seed7/train_log.json")
    assert os.path.exists(out)
    for p in (out, os.path.join(ROOT, f"cache/pilot_v12_I_w0_runs_mid_{cell}_s7_smoke.pt")):
        if os.path.exists(p):
            os.remove(p)
