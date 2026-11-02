"""Efficient, deterministic training-only selection of a single 2D trace.
Pre-computes Gaussian neighborhood support independent of gain and readout.
"""
import json,time
from pathlib import Path
import numpy as np, pandas as pd, cv2
from spatial_trace import N
from evaluate import angular_error_deg, get_data, random_baseline, score_original, TRAIN, TEST, SEEDS, LABELS, RESULTS
cv2.setNumThreads(1)


def trace_features(cand,count,decay,neighborhood,n_peaks=25):
    mem=np.zeros((N,N),np.float32)
    w=np.zeros((len(count),40),np.float32)
    s=np.zeros_like(w)
    for t,n in enumerate(count):
        n=int(n)
        if n==0:continue
        c=cand[t,:n]
        x=np.clip(np.rint(c[:,0]).astype(int),0,N-1)
        y=np.clip(np.rint(c[:,1]).astype(int),0,N-1)
        wide=cv2.GaussianBlur(mem,(0,0),neighborhood)
        narrow=cv2.GaussianBlur(mem,(0,0),1.2)
        w[t,:n]=wide[y,x];s[t,:n]=narrow[y,x]
        mem*=decay
        up=min(n,n_peaks)
        weight=np.minimum(np.maximum(c[:up,2],0),3.0)/3.0
        for xx,yy,v in zip(x[:up],y[:up],weight):
            if 2<=xx<N-2 and 2<=yy<N-2:
                mem[yy,xx]+=v
                mem[yy-1,xx]+=.33*v
                mem[yy+1,xx]+=.33*v
                mem[yy,xx-1]+=.33*v
                mem[yy,xx+1]+=.33*v
    return w,s


def evaluate_fast(cand,counts,gt,w,s,gain,inhibition):
    strengths=cand[:,:,2]
    scores=strengths*(1.+gain*np.maximum(w-inhibition*s,0))
    mask=np.arange(strengths.shape[1])[None,:]>=counts[:,None]
    scores[mask]=-np.inf
    indices=np.argmax(scores,axis=1)
    loc=cand[np.arange(len(indices)),indices,:2]
    err=angular_error_deg(loc,gt)
    return float((err[100:]<=5).mean())


def main():
    t=time.monotonic()
    datasets=[]
    for seed in TRAIN:
        for label in LABELS:
            c,n,gt,_=get_data(seed,label)
            datasets.append((seed,label,c,n,gt))
    acc=[]
    gain_values=(0.5,2.,5.,12.,30.,70.)
    for decay in (.45,.65,.8,.92):
        for neighborhood in (2.5,4.,7.,11.):
            features=[]
            for seed,label,c,n,gt in datasets:
                w,s=trace_features(c,n,decay,neighborhood)
                features.append((c,n,gt,w,s))
            for mode in ('leaky','annulus'):
                for inhibition in ((0.0,) if mode=='leaky' else (0.65,1.0,1.3)):
                    for gain in gain_values:
                        frac=np.mean([evaluate_fast(c,n,gt,w,s,gain,inhibition) for c,n,gt,w,s in features])
                        acc.append(dict(mode=mode,decay=decay,neighborhood=neighborhood,inhibition=inhibition,gain=gain,train_fraction=frac))
            print('grid',decay,neighborhood,'elapsed',round(time.monotonic()-t,1),flush=True)
    scores=pd.DataFrame(acc).sort_values('train_fraction',ascending=False)
    scores.to_csv(RESULTS/'train_grid.csv',index=False)
    selected={'raw':dict(mode='raw',gain=0,decay=0)}
    for mode in ('leaky','annulus'):
        best=scores[scores['mode']==mode].iloc[0].to_dict()
        selected[mode]={k:(float(best[k]) if k!='mode' else mode) for k in ['mode','decay','neighborhood','inhibition','gain']}
        print('selected',mode,best,flush=True)
    (RESULTS/'selected_configs.json').write_text(json.dumps(selected,indent=2))
    print('total tune',round(time.monotonic()-t,1),flush=True)
    return selected

if __name__=='__main__':main()
