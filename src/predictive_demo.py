"""Run the 24-channel frame-based STMD with a normalized predictive spatial field.

The DVS-like view in the animation is *display only*: this model processes frames.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from experiment import read_source,inject
from trajectories import generate_trajectory
from motion_bank_benchmark_fast import velocity_set,calibrate_custom,extract_candidates
from predictive_field import infer
ACTIVE_CONFIG=dict(decay=.85,gain=.3,sigma=5.,drift=.8,n_peaks=10)
from evaluate import angular_error_deg

ROOT=Path(__file__).resolve().parent.parent

def run(seed=173,distance=1.0,n_frames=365,scales=None,config=None):
    base=read_source()[:n_frames]
    gt=generate_trajectory(seed,len(base))
    frames=inject(base,gt,distance)
    V=velocity_set('grid24')
    if scales is None:scales=calibrate_custom(base,V)
    candidates,counts=extract_candidates(frames,V,scales)
    xy,score,velocity=infer(candidates,counts,config if config is not None else ACTIVE_CONFIG)
    error=angular_error_deg(xy,gt)
    good=np.isfinite(error)&(error<=5)
    return pd.DataFrame(dict(frame=np.arange(len(gt)),x=xy[:,0],y=xy[:,1],score=score,
                vx=velocity[:,0],vy=velocity[:,1],gt_x=gt[:,0],gt_y=gt[:,1],
                angular_error_deg=error,correct_5deg=good))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,default=173)
    ap.add_argument('--distance',type=float,default=1)
    ap.add_argument('--n-frames',type=int,default=365)
    ap.add_argument('--output',default=str(ROOT/'results'/'predictive_demo_1m.csv'))
    a=ap.parse_args()
    df=run(a.seed,a.distance,a.n_frames)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(a.output,index=False)
    print(f'{len(df)} frames, 5deg score frames100+ {df.iloc[100:].correct_5deg.mean()*100:.1f}%')
