"""Quick check of the missing-runs planting on the QC training split: realized versus requested
severity, and how well severity and the realized missing fraction separate the classes per world
and strength, against the natural host missing fraction. Prints only.
"""
import sys, time, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import planted as P, metrics as M
from safe_motion.data import asdpose as D
qc={"q_frozen_max":0.10,"q_arm_lowconf_max":0.30}
rows=D.load_index()
tr,va,_=D.subject_level_val_split(rows,0.2,seed=0)
tr=[r for r in tr if D.passes_qc(r,qc)]
y=np.array([r['binary_label'] for r in tr])
# contract checks
rng=np.random.RandomState(0)
assert not P.missing_mask(300,0.0,rng).any()
for e in (0.1,0.3,0.6): print('eps',e,'realized',round(P.missing_mask(20000,e,np.random.RandomState(1)).mean(),3))
t=time.time()
for strength in ('mid','low'):
  for world in ('S','I'):
    masks,eps=P.build_masks(tr,world,strength,seed=0)
    ds=D.SegmentDataset(tr,planted=masks)
    miss=np.array([ (ds.load_segment(r)[...,2]==0).mean() for r in tr])
    print(strength,world,'eps-AUC',round(M.auc(eps,1-y),3),'realized missing-frac AUC (neg vs pos)',round(M.auc(miss,1-y),3), 'mean miss', round(miss.mean(),3), f'{time.time()-t:.0f}s',flush=True)
# natural host baseline
ds=D.SegmentDataset(tr)
miss=np.array([(ds.load_segment(r)[...,2]==0).mean() for r in tr])
print('host natural missing-frac AUC',round(M.auc(miss,1-y),3),'mean',round(miss.mean(),3))
