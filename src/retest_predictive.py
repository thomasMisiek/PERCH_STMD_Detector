"""Controlled comparison on *new* 24-bank candidates.
Fixed calibration from moving no-target source across all camera speed conditions,
avoids near-zero normalization from a constant calibration sequence.
"""
import cv2
import numpy as np,pandas as pd
from pathlib import Path
from experiment import read_source, inject
from trajectories import generate_trajectory
from motion_bank_benchmark_fast import velocity_set,calibrate_custom,extract_candidates,background_variant
from predictive_field import infer as pred_infer
from spatial_trace import infer as old_infer
from evaluate import angular_error_deg
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'results'
OUT.mkdir(exist_ok=True)
V=velocity_set('grid24')
config=dict(decay=.85,gain=.3,sigma=5,drift=.8,n_peaks=10,winner_boost=.45)
old_config=dict(mode='annulus',decay=.92,neighborhood=11.,inhibition=1.3,gain=70.,n_peaks=25)

def go():
    base=read_source()
    scales=calibrate_custom(base,V)
    print('Common target-free moving-source calibration:',scales,flush=True)
    np.save(OUT/'calibration_grid24_fixed.npy',scales)
    rows=[]
    for name,speed in [('still',0),('slow',.5),('normal',1)]:
        bg=background_variant(base,speed)
        for seed in (173,281):
            path=generate_trajectory(seed,len(bg))
            for dist,label in [(.5,'50cm'),(1,'1m'),(2,'2m')]:
                print('running',name,seed,label,flush=True)
                c,n=extract_candidates(inject(bg,path,dist),V,scales)
                pred,ps,pv=pred_infer(c,n,config)
                raw=c[:,0,:2].copy();raw[n==0]=np.nan
                old,_,_=old_infer(c,n,old_config)
                for model,xy in [('raw',raw),('old_annular',old),('predictive_field',pred)]:
                    err=angular_error_deg(xy,path)
                    good=np.isfinite(err)&(err<=5)
                    rows.append(dict(background=name,seed=seed,distance=label,model=model,
                          correct=int(good[100:].sum()),n=265,correct_pct=float(100*good[100:].mean())))
                # Save 1m normal trial for demo regression checks.
                if name=='normal' and seed==173 and label=='1m':
                    pd.DataFrame(dict(frame=np.arange(len(bg)),x=pred[:,0],y=pred[:,1],
                      pred_x=pred[:,0],pred_y=pred[:,1],raw_x=raw[:,0],raw_y=raw[:,1],
                      gt_x=path[:,0],gt_y=path[:,1],score=ps,
                      error_deg=angular_error_deg(pred,path))).to_csv(OUT/'predictive_1m_per_frame.csv',index=False)
                pd.DataFrame(rows).to_csv(OUT/'grid24_predictive_camera_control.csv',index=False)
    print('SUMMARY\n',pd.DataFrame(rows).groupby(['background','model','distance']).correct_pct.mean().to_string(),flush=True)
if __name__=='__main__':go()
