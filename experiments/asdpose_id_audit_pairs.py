"""For raw ASDPose children mapped to cache IDs >= 282, find cache segments with nearby IDs that match their keypoints.

Usage: python experiments/asdpose_id_audit_pairs.py DIR, where DIR holds group_map.csv. Matching uses
the nose and shoulder keypoints; pairs with mean error below 6 are written to DIR/low_id_matches.csv.
"""
import os
SMM_ROOT = os.environ.get("SM_SMM_ROOT", "data/asdpose-smm-detection")   # SAFE-SMM checkout with cache/
ASDPOSE_RAW = os.environ.get("SM_ASDPOSE_RAW", "data/StimmingADSMotion")   # released ASDPose files
import zipfile,pickle,io,numpy as np,pandas as pd,sys
S=sys.argv[1]
g=pd.read_csv(f'{S}/group_map.csv')
hi=g[(g.f1>=282)&(g.err<6)].groupby('f1').raw_id.agg(lambda s:sorted(set(s)))
targets=sorted({x for v in hi for x in v})
c=pd.read_csv(os.path.join(SMM_ROOT, "cache/segments/index.csv"))
p=c.identifier.str.split('_',expand=True)
c['f1']=p[0].astype(int);c['f2']=p[1].astype(int);c['f3']=p[2].astype(int);c['st']=p[p.columns[-2]].astype(float)
KP=np.memmap(os.path.join(SMM_ROOT, "cache/segments/keypoints.f32"),dtype=np.float32,mode='r')
z=zipfile.ZipFile(os.path.join(ASDPOSE_RAW, "asdpose.zip")); names=[n for n in z.namelist() if n.endswith('.pkl')]
def q_of(r):
    a=np.array(KP[r.byte_offset//4:r.byte_offset//4+r.n_frames*51]).reshape(r.n_frames,17,3); return a[:,[0,5,6],:2]
def err(K,fps,st,q):
    a0=int((st-4)*fps);a1=int((st+2)*fps)
    if a0<0 or a1+len(q)>len(K): return 1e9
    idx=np.arange(0,len(q),3);qq=q[idx];qm=(qq!=0).all(2);offs=np.arange(a0,a1)
    W=K[offs[:,None]+idx[None,:]];ok=qm[None]&(W!=0).all(3);e=np.abs(W-qq[None]).mean(3);cnt=ok.sum((1,2))
    return np.where(cnt>=20,(e*ok).sum((1,2))/np.maximum(cnt,1),1e9).min()
out=[]
for X in targets:
    files=[n for n in names if n.startswith(f'data/{X}_')]
    raws=[(n,pickle.load(io.BytesIO(z.read(n)))) for n in files]
    cand=c[(c.f1>=max(1,X-10))&(c.f1<=min(282,X+50))]
    for (f1,f2,f3),sub in cand.groupby(['f1','f2','f3']):
        best=1e9
        for _,r in sub.sort_values('n_frames',ascending=False).head(2).iterrows():
            q=q_of(r)
            for n,R in raws: best=min(best,err(R['keypoint'][0][:,[0,5,6],:],R['fps'],r.st,q))
        if best<6: out.append((X,f1,f2,f3,best))
    print('child',X,'low-id matches',sorted({o[1] for o in out if o[0]==X}),flush=True)
pd.DataFrame(out,columns=['raw','f1','f2','f3','err']).to_csv(f'{S}/low_id_matches.csv',index=False)
