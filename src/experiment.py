"""Matched-protocol comparison: tuned STMD vs competing-hypothesis STMD.

Full original source (365 frames/14.6 seconds), four randomized trajectories.
One path per trial shared among 50 cm, 1 m, 2 m, and both detectors.
Calibration is performed on the no-target video only and never with the target.
Metrics evaluated on frames 100..364, withheld from calibration.
"""
from pathlib import Path
import argparse, json, time
import cv2, numpy as np, pandas as pd
from stmd_model import Detector as SimpleDetector, stamp, apparent_diameter_px
from comparison import multi_hypothesis as multi
from trajectories import generate_trajectory

cv2.setNumThreads(2)
ROOT=Path(__file__).resolve().parent
SIDE=320
FPS=25
SEEDS=(41,92,173,281)
DISTANCES=((0.5,'50cm'),(1.0,'1m'),(2.0,'2m'))
EVAL_START=100
SIMPLE_CONFIGS=[dict(decay=.85,density_weight=0.), dict(decay=.85,density_weight=16.)]

def read_source(path=ROOT/'data'/'source_indoor_walking.mp4'):
    cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS)
    frames=[]
    while True:
        ok,fr=cap.read()
        if not ok:break
        h,w=fr.shape[:2]
        small=cv2.resize(fr,(round(w*SIDE/h),SIDE),interpolation=cv2.INTER_AREA)
        x=(small.shape[1]-SIDE)//2
        frames.append(small[:,x:x+SIDE])
    cap.release()
    if abs(fps-FPS)>.01:raise ValueError(f'expected 25fps, got {fps}')
    return np.asarray(frames,dtype=np.uint8)

def inject(base, path, distance):
    out=base.copy(); diameter=apparent_diameter_px(6,distance)
    for i,xy in enumerate(path):stamp(out[i],tuple(map(int,xy)),diameter)
    return out

def tune_calibration(base):
    models=[SimpleDetector(**c) for c in SIMPLE_CONFIGS]
    peaks=[[],[]]
    for frame in base[:100]:
        for j,d in enumerate(models):peaks[j].append(d.step(frame)['peak'])
    return np.asarray([max(float(np.quantile(s[25:],.99)),1e-6) for s in peaks])

def run_simple(frames, thresholds, gt=None, visual=False):
    models=[SimpleDetector(**c) for c in SIMPLE_CONFIGS]
    rows=[];vis=[];previous=None
    for i,fr in enumerate(frames):
        raw=[d.step(fr,return_map=visual) for d in models]
        norm=np.asarray([raw[j]['peak']/thresholds[j] for j in range(2)])
        k=int(norm.argmax());accepted=bool(norm[k]>=1)
        x,y=raw[k]['position'];confidence=float(1/(1+np.exp(-np.clip(4*(norm[k]-1),-20,20))))
        row={'frame':i,'detected':accepted,'x':float(x),'y':float(y),'confidence':confidence,'pathway':k,'score':float(raw[k]['peak'])}
        if gt is not None:
            row.update(gt_x=int(gt[i,0]),gt_y=int(gt[i,1]),error_px=float(np.hypot(x-gt[i,0],y-gt[i,1])))
            row['correct']=bool(accepted and row['error_px']<=8)
        rows.append(row)
        if visual:
            vis.append(dvs_style(fr, previous, raw[k]['activity'],row))
            previous=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY).astype(np.float32)
    return pd.DataFrame(rows),vis

def dvs_style(frame,prev,activity,row):
    """Frame-derived DVS-STYLE DISPLAY ONLY: not real events, not detector input."""
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.float32)
    out=np.full((SIDE,SIDE,3),15,np.uint8)
    if prev is not None:
        logdiff=np.log1p(gray)-np.log1p(prev)
        on=np.sqrt(np.clip((logdiff-.045)/.25,0,1))
        off=np.sqrt(np.clip((-logdiff-.045)/.25,0,1))
        out[...,1]=np.clip(out[...,1]+240*on,0,255).astype(np.uint8)
        of=(15+240*off).astype(np.uint8)
        out[...,0]=of;out[...,2]=of
        if activity is not None:
            scale=max(float(np.quantile(activity,.999)),1e-6)
            out[...,2]=np.maximum(out[...,2],np.clip(60*np.sqrt(np.clip(activity/scale,0,1)),0,255).astype(np.uint8))
    if row['detected']:
        pos=(int(round(row['x'])),int(round(row['y'])))
        cv2.circle(out,pos,8,(0,240,255),1,cv2.LINE_AA)
        cv2.circle(out,pos,2,(255,255,255),-1,cv2.LINE_AA)
    cv2.putText(out,f'Evidence {row["confidence"]:.2f}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(245,245,245),1,cv2.LINE_AA)
    return out

def run_multi(frames, thresholds, gt=None,visual=False):
    return multi.run(frames,thresholds,gt,with_visuals=visual)

def evaluate_one(base, thresholds_simple, thresholds_multi, path, seed, distance, label, output_dir):
    target=inject(base,path,distance)
    start=time.monotonic()
    s,_=run_simple(target,thresholds_simple,gt=path)
    simple_time=time.monotonic()-start
    start=time.monotonic()
    m,_=run_multi(target,thresholds_multi,gt=path)
    multi_time=time.monotonic()-start
    s.to_csv(output_dir/f'trial_{seed}_{label}_simple.csv',index=False)
    m.to_csv(output_dir/f'trial_{seed}_{label}_multi.csv',index=False)
    result=[]
    for name,df,secs in [('tuned_stmd',s,simple_time),('multi_hypothesis',m,multi_time)]:
        d=df.iloc[EVAL_START:]
        result.append(dict(seed=seed,distance=label,distance_m=distance,model=name,correct=int(d.correct.sum()),eval_frames=len(d),correct_fraction=float(d.correct.mean()),detection_fraction=float(d.detected.mean()),mean_error_on_detected=float(d[d.detected].error_px.mean()),runtime_s=round(secs,1)))
    return result

def main(seeds=SEEDS):
    output_dir=ROOT/'results';output_dir.mkdir(parents=True,exist_ok=True)
    base=read_source();n=len(base)
    print(f'Original full video: {n} frames, {n/FPS:.1f}s',flush=True)
    simple_thr=tune_calibration(base)
    multi_thr=multi.calibrate(base,frames=100)
    print('Calibrated no-target scales:',simple_thr,multi_thr,flush=True)
    no_simple,_=run_simple(base,simple_thr)
    no_multi,_=run_multi(base,multi_thr)
    no_simple.to_csv(output_dir/'control_no_target_simple.csv',index=False)
    no_multi.to_csv(output_dir/'control_no_target_multi.csv',index=False)
    fpr_simple=float(no_simple.iloc[EVAL_START:].detected.mean())
    fpr_multi=float(no_multi.iloc[EVAL_START:].detected.mean())
    print('False positives (fraction no-target frames):',fpr_simple,fpr_multi,flush=True)
    rows=[]
    for seed in seeds:
        path=generate_trajectory(seed,n)
        pd.DataFrame({'frame':np.arange(n),'x':path[:,0],'y':path[:,1]}).to_csv(output_dir/f'trajectory_seed_{seed}.csv',index=False)
        print('Trajectory',seed,'starting',tuple(path[0]),'ending',tuple(path[-1]),flush=True)
        for dist,label in DISTANCES:
            r=evaluate_one(base,simple_thr,multi_thr,path,seed,dist,label,output_dir)
            rows.extend(r)
            pd.DataFrame(rows).to_csv(output_dir/'all_trial_metrics.csv',index=False)
            print('FINISHED',seed,label,[(x['model'],round(x['correct_fraction']*100,1)) for x in r],flush=True)
    df=pd.DataFrame(rows)
    overview=df.groupby(['model','distance'],as_index=False).agg(correct=('correct','sum'),eval_frames=('eval_frames','sum'),correct_pct=('correct_fraction',lambda x:round(100*x.mean(),1)),sd_pct=('correct_fraction',lambda x:round(100*x.std(ddof=1),1)),mean_runtime_s=('runtime_s','mean'))
    overview.to_csv(output_dir/'summary_by_distance.csv',index=False)
    meta=dict(seeds=list(seeds),fps=FPS,real_frames=n,eval_start=EVAL_START,no_target_fpr={'tuned_stmd':fpr_simple,'multi_hypothesis':fpr_multi},thresholds={'tuned_stmd':simple_thr.tolist(),'multi_hypothesis':multi_thr.tolist()},target_diameter_px={label:apparent_diameter_px(6,dist) for dist,label in DISTANCES})
    (output_dir/'protocol.json').write_text(json.dumps(meta,indent=2))
    print('OVERALL\n',overview.to_string(index=False),flush=True)
    print('Winner by mean correct%:',overview.groupby('model').correct_pct.mean().idxmax(),flush=True)
    return overview

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='*',type=int,default=list(SEEDS));a=ap.parse_args()
    main(tuple(a.seeds))
