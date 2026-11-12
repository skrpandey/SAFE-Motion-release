"""v4 pre-check (no model): realized cue strength of the whole-joint mechanism; v3 masks unchanged."""
import sys, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import planted as P, metrics as M
from safe_motion.data import asdpose as D
qc = {"q_frozen_max": 0.10, "q_arm_lowconf_max": 0.30}
rows = D.load_index()
tr, va, _ = D.subject_level_val_split(rows, 0.2, seed=0)
tr = [r for r in tr if D.passes_qc(r, qc)]
y = np.array([r['binary_label'] for r in tr])
m3, _ = P.build_masks(tr[:300], "S", "mid", 0)            # must equal v3 masks
print("v3 runs mask frac (first 300)", round(np.mean([m.mean() for m in m3.values()]), 4))
print("eps contract", P.whole_joint_mask(10, 0.0, np.random.RandomState(0)).any(),
      round(P.whole_joint_mask(1, 0.3, np.random.RandomState(1), n_joints=100000).mean(), 3))
for strength in ("mid", "low"):
    for world in ("S", "I"):
        masks, eps = P.build_masks(tr, world, strength, 0, mechanism="whole")
        ds = D.SegmentDataset(tr, planted=masks)
        miss = np.array([(ds.load_segment(r)[..., 2] == 0).mean() for r in tr])
        print("whole", strength, world, "eps-AUC", round(M.auc(eps, 1 - y), 3),
              "realized missing-frac AUC", round(M.auc(miss, 1 - y), 3), "mean miss", round(miss.mean(), 3), flush=True)
