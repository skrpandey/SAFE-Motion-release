"""Match SAFE-SMM cache segments to the raw ASDPose recordings for the identifier audit.

For each cache group (f1, f2, f3) the longest segment is compared with every raw .pkl in
asdpose.zip over start offsets in [st - 4 s, st + 2 s], using 5 joints (nose, shoulders, hips) and
every second frame; the entry is the smallest median coordinate error (1e9 if no valid alignment).
Writes OUT/{errmat.npy, groups.csv, rawnames.csv}.  Usage: python asdpose_id_audit_full.py OUT
"""
import os
SMM_ROOT = os.environ.get("SM_SMM_ROOT", "data/asdpose-smm-detection")   # SAFE-SMM checkout with cache/
ASDPOSE_RAW = os.environ.get("SM_ASDPOSE_RAW", "data/StimmingADSMotion")   # released ASDPose files
import zipfile,pickle,io,numpy as np,pandas as pd,sys
from multiprocessing import Pool
S=sys.argv[1]; J=[0,5,6,11,12]
c=pd.read_csv(os.path.join(SMM_ROOT, "cache/segments/index.csv"))
p=c.identifier.str.split('_',expand=True)
c['f1']=p[0].astype(int);c['f2']=p[1].astype(int);c['f3']=p[2].astype(int);c['st']=p[p.columns[-2]].astype(float)
KP=np.memmap(os.path.join(SMM_ROOT, "cache/segments/keypoints.f32"),dtype=np.float32,mode='r')
G=[];Q=[]
for g,sub in c.groupby(['f1','f2','f3']):
    r=sub.sort_values('n_frames',ascending=False).iloc[0]
    a=np.array(KP[r.byte_offset//4:r.byte_offset//4+r.n_frames*51]).reshape(r.n_frames,17,3)[:,J,:2]
    G.append(g);Q.append((r.st,a))
names=[n for n in zipfile.ZipFile(os.path.join(ASDPOSE_RAW, "asdpose.zip")).namelist() if n.endswith('.pkl')]
def work(n):
    z=zipfile.ZipFile(os.path.join(ASDPOSE_RAW, "asdpose.zip"))
    R=pickle.load(io.BytesIO(z.read(n)));K=R['keypoint'][0][:,J,:];fps=R['fps']
    out=np.full(len(Q),1e9,np.float32)
    for i,(st,q) in enumerate(Q):
        a0=int((st-4)*fps);a1=int((st+2)*fps)
        if a0<0 or a1+len(q)>len(K): continue
        idx=np.arange(0,len(q),2);qq=q[idx];qm=(qq!=0).all(2);offs=np.arange(a0,a1)
        W=K[offs[:,None]+idx[None,:]];ok=qm[None]&(W!=0).all(3)
        e=np.linalg.norm(W-qq[None],axis=3);cnt=ok.sum((1,2))
        if cnt.max()<30: continue
        med=np.nanmedian(np.where(ok,e,np.nan).reshape(len(offs),-1),axis=1); med=np.where(cnt>=30,med,1e9)
        out[i]=med.min()
    return out
if __name__=='__main__':
    with Pool(11) as pool: M=np.stack(pool.map(work,names,chunksize=4),1)
    np.save(f'{S}/errmat.npy',M)
    pd.DataFrame(G,columns=['f1','f2','f3']).assign(st=[q[0] for q in Q]).to_csv(f'{S}/groups.csv',index=False)
    pd.Series(names).to_csv(f'{S}/rawnames.csv',index=False)
    print('done',M.shape)
