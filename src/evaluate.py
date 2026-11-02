"""5-degree angular localization comparison and train/held-out selection.
Raw peak, leaky 2-D trace, motion-neighborhood trace, original multi hypotheses,
and random uniform detector. Any trial-selection uses seeds 41,92 ONLY.
"""
from pathlib import Path
import json, itertools, argparse
import numpy as np, pandas as pd
from spatial_trace import infer
from trajectories import generate_trajectory

ROOT=Path(__file__).resolve().parent.parent
RESULTS=ROOT/'results'
TRAIN=(41,92); TEST=(173,281)
SEEDS=TRAIN+TEST; LABELS=('50cm','1m','2m')
FOV_DEG=84.;SIDE=320
F=(SIDE/2)/np.tan(np.deg2rad(FOV_DEG/2))


def angular_error_deg(xy,gt):
    a=np.asarray(xy,dtype=np.float64);b=np.asarray(gt,dtype=np.float64)
    ax=(a[:,0]-(SIDE-1)/2)/F; ay=(a[:,1]-(SIDE-1)/2)/F
    bx=(b[:,0]-(SIDE-1)/2)/F; by=(b[:,1]-(SIDE-1)/2)/F
    dot=1+ax*bx+ay*by
    na=np.sqrt(1+ax*ax+ay*ay); nb=np.sqrt(1+bx*bx+by*by)
    return np.rad2deg(np.arccos(np.clip(dot/(na*nb),-1,1)))


def get_data(seed,label):
    z=np.load(RESULTS/f'evidence_{seed}_{label}.npz')
    gt=generate_trajectory(seed,len(z['count']))
    return z['candidate'],z['count'],gt,z['multi']


def score_trial(cfg,seed,label,return_dataframe=False):
    cand,count,gt,multi=get_data(seed,label)
    xy,strength,raw=infer(cand,count,cfg)
    err=angular_error_deg(xy,gt)
    mask=np.isfinite(err); good=mask&(err<=5)
    stats=dict(correct=int(good[100:].sum()),count=int(len(good[100:])),fraction=float(good[100:].mean()),median_error_deg=float(np.nanmedian(err[100:])),mean_error_deg=float(np.nanmean(err[100:])))
    if return_dataframe:
        df=pd.DataFrame(dict(frame=np.arange(len(gt)),x=xy[:,0],y=xy[:,1],score=strength,raw_score=raw,gt_x=gt[:,0],gt_y=gt[:,1],error_deg=err,correct_5deg=good))
        return stats,df
    return stats


def score_original(seed,label):
    _,_,gt,m=get_data(seed,label)
    xy=m[:,1:3];conf=m[:,3]; det=m[:,0]>0.5
    err=angular_error_deg(xy,gt)
    good=np.isfinite(err)&(err<=5)
    return {'forced':float(good[100:].mean()),'gated':float((good&det)[100:].mean()),'accept':float(det[100:].mean()),'med_error':float(np.nanmedian(err[100:]))}


def random_baseline(seed,label,n_draw=300,random_seed=593):
    _,_,gt,_=get_data(seed,label)
    gt=gt[100:]
    rng=np.random.default_rng(random_seed+seed)
    # Uniform in same eligible interior as neural evidence (12..307).
    pts=rng.uniform(12,308,(n_draw,len(gt),2))
    errs=[]
    for p in pts:errs.append(float((angular_error_deg(p,gt)<=5).mean()))
    return float(np.mean(errs)),float(np.std(errs))


def tune_grid():
    # Fixed pre-declared small family. Train/test by trajectory seed.
    cfgs=[dict(mode='raw', gain=0,decay=0)]
    for mode in ('leaky','annulus'):
        for decay in (.55,.75,.9):
            for neighborhood in (3.,6.,10.):
                for gain in (2.,8.,24.):
                    for inhibition in ((0.,) if mode=='leaky' else (.65,1.05)):
                        cfgs.append(dict(mode=mode,decay=decay,neighborhood=neighborhood,gain=gain,inhibition=inhibition,n_peaks=25))
    scores=[]
    for j,cfg in enumerate(cfgs):
        fracs=[]
        for seed in TRAIN:
            for label in LABELS:
                fracs.append(score_trial(cfg,seed,label)['fraction'])
        scores.append((float(np.mean(fracs)),j,cfg))
    scores.sort(reverse=True)
    chosen={}
    for typ in ('raw','leaky','annulus'):
        best=next(s for s in scores if s[2]['mode']==typ)
        chosen[typ]=best[2]
        print('trained',typ,round(100*best[0],2),best[2],flush=True)
    pd.DataFrame([dict(train_acc_5deg=100*s,j=j,**c) for s,j,c in scores]).to_csv(RESULTS/'train_grid.csv',index=False)
    (RESULTS/'selected_configs.json').write_text(json.dumps(chosen,indent=2))
    return chosen


def main():
    chosen=json.loads((RESULTS/"selected_configs.json").read_text())
    rows=[]
    for seed in SEEDS:
        for label in LABELS:
            for name,cfg in chosen.items():
                stat,df=score_trial(cfg,seed,label,True)
                df.to_csv(RESULTS/f'{name}_{seed}_{label}.csv',index=False)
                rows.append(dict(split='train' if seed in TRAIN else 'heldout',seed=seed,distance=label,model=name,**stat))
            orig=score_original(seed,label)
            rows.append(dict(split='train' if seed in TRAIN else 'heldout',seed=seed,distance=label,model='existing_multi',correct=round(orig['forced']*265),count=265,fraction=orig['forced'],median_error_deg=orig['med_error'],gated_fraction=orig['gated']))
            rand,sd=random_baseline(seed,label)
            rows.append(dict(split='train' if seed in TRAIN else 'heldout',seed=seed,distance=label,model='random_uniform',correct=np.nan,count=265,fraction=rand,sd_random=sd))
    df=pd.DataFrame(rows)
    df.to_csv(RESULTS/'all_trials_5deg.csv',index=False)
    out=df.groupby(['split','model','distance'],as_index=False).fraction.mean()
    out['percent']=out.fraction*100
    out.to_csv(RESULTS/'summary_5deg.csv',index=False)
    print(out.to_string(index=False),flush=True)
    # save no-target raw and each enhanced, so can inspect FP tradeoffs
    z=np.load(RESULTS/'evidence_none.npz')
    nullrows=[]
    for name,cfg in chosen.items():
        xy,score,raw=infer(z['candidate'],z['count'],cfg)
        pd.DataFrame(dict(frame=np.arange(len(score)),x=xy[:,0],y=xy[:,1],score=score)).to_csv(RESULTS/f'null_{name}.csv',index=False)
    return out

if __name__=='__main__':main()
