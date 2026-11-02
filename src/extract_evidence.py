"""Extract data-independent candidate peaks from STMD motion responses.
No ground-truth coordinates are supplied to either detector.
"""
import sys, argparse, time
from pathlib import Path
import numpy as np
from experiment import read_source, inject, tune_calibration
from trajectories import generate_trajectory
from comparison.multi_hypothesis import get_detectors, calibrate, candidates_from_maps, MultiHypothesis

SRC=Path(__file__).resolve().parent
OUT=SRC.parent/'results'
OUT.mkdir(parents=True,exist_ok=True)
SEEDS=(41,92,173,281)
DISTANCES=((.5,'50cm'),(1,'1m'),(2,'2m'))

def extract(frames,scales):
    detectors=get_detectors();tracker=MultiHypothesis(scales)
    vals=np.zeros((len(frames),40,4),dtype=np.float32)
    counts=np.zeros(len(frames),dtype=np.int16)
    ref=np.zeros((len(frames),5),dtype=np.float32)
    for i,fr in enumerate(frames):
        maps=[d.step(fr,return_map=True)['activity'] for d in detectors]
        candidates,combo=candidates_from_maps(maps,scales,num=40,accept_floor=.22,sep=5)
        n=min(len(candidates),40)
        if n: vals[i,:n]=np.array(candidates[:n],dtype=np.float32)
        counts[i]=n
        row,_,_=tracker.step(maps)
        ref[i]=[row['detected'],row['x'],row['y'],row['confidence'],row['n_tracks']]
    return vals,counts,ref

def main(seed, label=None):
    t=time.monotonic(); base=read_source(); scale_file=OUT/'calibration_scales.npy'
    if scale_file.exists(): scales=np.load(scale_file)
    else:
        scales=calibrate(base,frames=100)
        np.save(scale_file,scales)
    if seed==-1: cases=[('none',base)]
    else:
        path=generate_trajectory(seed,len(base));cases=[(f'{seed}_{label}',inject(base,path,dist)) for dist,label in DISTANCES]
    if label is not None: cases=[(name,frames) for name,frames in cases if name.endswith('_'+label)]
    print('scale',scales,'cases',list(x[0] for x in cases), flush=True)
    for name,frames in cases:
        ts=time.monotonic()
        a,c,r=extract(frames,scales)
        np.savez_compressed(OUT/f'evidence_{name}.npz',candidate=a,count=c,multi=r)
        print('finished',name,'time',round(time.monotonic()-ts,1),'nonzero',float(np.mean(c)),flush=True)
    print('seed',seed,'total',round(time.monotonic()-t,1),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--label');a=p.parse_args();main(a.seed,a.label)
