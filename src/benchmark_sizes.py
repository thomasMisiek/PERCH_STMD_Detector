"""Fixed-model, offline 5-degree localization benchmark on synthetic circular targets.

Uses the unchanged 24-channel STMD candidates with raw highest-peak selection
(no additional predictive memory). This is a synthetic video evaluation, not
an assessment of an event camera, real insects, or flight tracking.

Usage: python src/benchmark_sizes.py
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from experiment import read_source
from trajectories import generate_trajectory
from motion_bank_benchmark_fast import velocity_set,calibrate_custom,extract_candidates,background_variant
from stmd_model import apparent_diameter_px, stamp
from evaluate import angular_error_deg

ROOT=Path(__file__).resolve().parent.parent
RESULTS=ROOT/'results'
FPS=25
EVAL_START=100
FOV_DEG=84
SEEDS=(173,281)
SIZES_MM=(7,20)
DISTANCES_M=(0.5,1,2,3,4)
CONDITIONS=(('still background',0),('moving background',1))

def inject_size(base, path, distance_m, diameter_mm):
    frames=base.copy()
    diam=apparent_diameter_px(diameter_mm,distance_m)
    for i, xy in enumerate(path):
        stamp(frames[i],tuple(map(int,xy)),diam)
    return frames

def main(seeds=SEEDS):
    RESULTS.mkdir(parents=True,exist_ok=True)
    base=read_source()
    assert len(base)==365 and base.shape[1:3]==(320,320)
    velocities=velocity_set('grid24')
    scales=calibrate_custom(base,velocities)
    np.save(RESULTS/'common_target_free_calibration.npy',scales)
    rows=[]
    for condition,speed in CONDITIONS:
        bg=background_variant(base,speed)
        for seed in seeds:
            trajectory=generate_trajectory(seed,len(bg))
            for size_mm in SIZES_MM:
                for distance_m in DISTANCES_M:
                    label=f'{distance_m:g}m'
                    print('Running',condition,'seed',seed,'target',size_mm,'mm',label,flush=True)
                    frames=inject_size(bg,trajectory,distance_m,size_mm)
                    candidates,counts=extract_candidates(frames,velocities,scales)
                    prediction=candidates[:,0,:2].copy()
                    prediction[counts==0]=np.nan
                    errors=angular_error_deg(prediction,trajectory)
                    correct=np.isfinite(errors)&(errors<=5)
                    eval_correct=correct[EVAL_START:]
                    row=dict(condition=condition,seed=seed,target_mm=size_mm,distance_m=distance_m,
                             apparent_diameter_px=apparent_diameter_px(size_mm,distance_m),
                             correct=int(eval_correct.sum()),n=int(len(eval_correct)),
                             accuracy_pct=float(100*eval_correct.mean()),
                             model='raw 24-channel STMD (no additional memory)')
                    rows.append(row)
                    # Keep per-frame scores for auditing, including non-detections.
                    pd.DataFrame(dict(frame=np.arange(len(bg)),gt_x=trajectory[:,0],gt_y=trajectory[:,1],
                                      pred_x=prediction[:,0],pred_y=prediction[:,1],
                                      candidate_count=counts,error_deg=errors,correct_5deg=correct)).to_csv(
                                          RESULTS/f'trial_{size_mm}mm_{label}_{condition.replace(" ","_")}_seed{seed}.csv',index=False)
                    pd.DataFrame(rows).to_csv(RESULTS/'all_trials.csv',index=False)
                    print('  accuracy:',round(row['accuracy_pct'],2),'%',flush=True)
    df=pd.DataFrame(rows)
    summary=df.groupby(['target_mm','condition','distance_m'],as_index=False).agg(
        correct=('correct','sum'),n=('n','sum'),accuracy_pct=('accuracy_pct','mean'))
    summary.to_csv(RESULTS/'summary.csv',index=False)
    (RESULTS/'protocol.json').write_text(json.dumps(dict(
        target_diameters_mm=list(SIZES_MM),distance_m=list(DISTANCES_M),
        conditions=dict(CONDITIONS),seeds=list(seeds),frame_rate=FPS,
        evaluated_frames='100..364 inclusive per trial',sensor_resolution=[320,320],
        assumed_field_of_view_deg=FOV_DEG,
        common_calibration='target-free original moving video',
        readout='top pooled STMD candidate, no additional memory, forced choice',
        target='synthetic black circular disk, area-integrated then blurred, not a real insect',
        caveat='2D image replay speed, not measured physical camera velocity'),indent=2))
    print('\nFINAL SUMMARY\n',summary.to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    args=p.parse_args();main(tuple(args.seeds))
