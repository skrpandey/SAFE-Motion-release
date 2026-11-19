"""Pilot match of SAFE-SMM cache segments to raw ASDPose recordings by nose and shoulder keypoints.

Takes the two longest segments of every group with first ID field above 241, plus 60 random
control groups (seed 0), and finds the raw .pkl and offset with the lowest mean coordinate error.
    python experiments/asdpose_id_audit_pilot.py <out_dir>    # writes <out_dir>/id_map.csv
"""
import os
SMM_ROOT = os.environ.get("SM_SMM_ROOT", "data/asdpose-smm-detection")   # SAFE-SMM checkout with cache/
ASDPOSE_RAW = os.environ.get("SM_ASDPOSE_RAW", "data/StimmingADSMotion")   # released ASDPose files
import zipfile,pickle,io,numpy as np,pandas as pd,sys,random
S=sys.argv[1]
c=pd.read_csv(os.path.join(SMM_ROOT, "cache/segments/index.csv"))
p=c.identifier.str.split('_',expand=True)
c['f1']=p[0].astype(int);c['f2']=p[1].astype(int);c['f3']=p[2].astype(int)
c['st']=p[p.columns[-2]].astype(float)
KP=np.memmap(os.path.join(SMM_ROOT, "cache/segments/keypoints.f32"),dtype=np.float32,mode='r')
groups=c.groupby(['f1','f2','f3'])
random.seed(0)
ctrl=random.sample([g for g in groups.groups if g[0]<=241],60)
want=[g for g in groups.groups if g[0]>241]+ctrl
queries=[]
for g in want:
    sub=groups.get_group(g).sort_values('n_frames',ascending=False).head(2)
    for _,r in sub.iterrows():
        a=np.array(KP[r.byte_offset//4: r.byte_offset//4 + r.n_frames*51]).reshape(r.n_frames,17,3)
        queries.append((g,r.st,a[:,[0,5,6],:2]))
print('groups',len(want),'queries',len(queries),flush=True)
z=zipfile.ZipFile(os.path.join(ASDPOSE_RAW, "asdpose.zip"))
best={}
names=[n for n in z.namelist() if n.endswith('.pkl')]
for k,n in enumerate(names):
    r=pickle.load(io.BytesIO(z.read(n))); K=r['keypoint'][0][:,[0,5,6],:]; fps=r['fps']
    cam=int(n.split('_')[-1][:-4])
    for (g,st,q) in queries:
        a0=int((st-4)*fps); a1=int((st+2)*fps)
        if a0<0 or a1+len(q)>len(K): continue
        L=len(q); idx=np.arange(0,L,3); qq=q[idx]; qm=(qq!=0).all(2)
        offs=np.arange(a0,a1)
        W=K[offs[:,None]+idx[None,:]]
        ok=qm[None]&(W!=0).all(3)
        err=np.abs(W-qq[None]).mean(3)
        cnt=ok.sum((1,2))
        m=np.where(cnt>=20,(err*ok).sum((1,2))/np.maximum(cnt,1),1e9)
        bestq=m.min()
        key=(g,st)
        if bestq<best.get(key,(1e9,))[0]: best[key]=(bestq,n,cam)
    if k%50==0: print(k,len(names),flush=True)
rows=[(g[0],g[1],g[2],st,v[0],v[1]) for (g,st),v in best.items()]
pd.DataFrame(rows,columns=['f1','f2','f3','st','err','raw']).to_csv(f'{S}/id_map.csv',index=False)
print('done')
