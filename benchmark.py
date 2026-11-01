"""Tuned, de novo, full-field STMD-like experiment.

Input: a real house video. Projected mosquito target size is parameterized by
body length, distance, sensor FOV, and spatial sensor resolution.

This is NOT a transistor-level or full connectomic model, and the right panels
are bright frame-derived DVS-style visualizations, NOT genuine asynchronous data.

In particular neither target ground truth nor dot-free frames enter the
per-frame detection pathway. Dot-free frames are used solely to calibrate a
single global scalar threshold and to measure false-alarm rates.
"""
import csv, math, os, time
from pathlib import Path
import cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt
from stmd_model import Detector,stamp,apparent_diameter_px

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'data'/'source_indoor_walking.mp4'
N=500
FPS=25.0
SIDE=320
CONDITIONS=[(.5,'50cm'),(1.0,'1m'),(2.0,'2m')]
CONFIGS=[{'decay':0.85,'density_weight':0.0}, {'decay':0.85,'density_weight':16.0}]
MOSQUITO_LENGTH_MM=6.0
FOV_DEG=84.0
cv2.setNumThreads(2)

def load_input():
    cap=cv2.VideoCapture(str(SOURCE))
    fps=cap.get(cv2.CAP_PROP_FPS)
    images=[]
    while True:
        ok,fr=cap.read()
        if not ok:break
        h,w=fr.shape[:2]
        fr=cv2.resize(fr,(int(round(w*SIDE/h)),SIDE),interpolation=cv2.INTER_AREA)
        x=(fr.shape[1]-SIDE)//2
        images.append(fr[:,x:x+SIDE])
    cap.release()
    return np.asarray(images,dtype=np.uint8), float(fps)

def make_20s(frames):
    # The actual source is 14.6 s. Reverse its final 5.4 s to give 20 s
    # without holding each frame for 2 timestamps. Document the reverse portion.
    indices=list(range(len(frames)))
    back=N-len(indices)
    assert back>0
    indices.extend(list(range(len(frames)-2,len(frames)-2-back,-1)))
    assert len(indices)==N
    return frames[np.array(indices)]

def target_path(t):
    # controlled, smoothly reversing mosquito motion; ground truth is never used by model.
    # preserves 0..249 of the previous synthetic trajectory for baseline comparison
    if t<250:
        u=t/249.0
        x=int(round(165+110*u))
        y=int(round(68+9*math.sin(2*math.pi*t/120+0.4)))
    else:
        u=(t-250)/249.0
        x=int(round(275-110*u))
        y=int(round(68+9*math.sin(2*math.pi*t/120+0.4)))
    return x,y

def apply_target(base,dist):
    frames=base.copy();size=apparent_diameter_px(MOSQUITO_LENGTH_MM,dist,FOV_DEG,SIDE)
    xy=[]
    for t in range(len(base)):
        p=target_path(t)
        stamp(frames[t],p,size)
        xy.append(p)
    return frames,np.asarray(xy),size


def detectors():
    return [Detector(**cfg) for cfg in CONFIGS]

def evaluate_frames(frames,thresholds,gt=None,make_vis=False):
    ds=detectors()
    results=[]; visuals=[]
    prev_gray=None
    for t,frame in enumerate(frames):
        raw=[d.step(frame,return_map=make_vis) for d in ds]
        scores=np.array([v['peak'] for v in raw],np.float64)
        normalized=scores/thresholds
        k=int(np.argmax(normalized))
        # Any pathway can detect; whichever has stronger normalized evidence wins
        accepted=bool(normalized[k]>=1.0)
        pos=np.asarray(raw[k]['position'],dtype=float)
        conf=float(1/(1+np.exp(-np.clip(4*(normalized[k]-1),-20,20))))
        row={'frame':t,'time_s':t/FPS,'detected':accepted,'x':pos[0],'y':pos[1],
             'confidence':conf,'neural_score':scores[k], 'pathway':k,
             'broad_score':scores[0], 'selective_score':scores[1],
             'direction_x':raw[k]['direction'][0], 'direction_y':raw[k]['direction'][1]}
        if gt is not None:
            row['gt_x'],row['gt_y']=gt[t]
            err=float(np.linalg.norm(pos-gt[t]));row['error_px']=err
            row['correct']=bool(accepted and err<=8.0)
        results.append(row)
        if make_vis:
            g=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev_gray is None:
                vis=np.zeros((SIDE,SIDE,3),np.uint8)
            else:
                # Simulated frame-derived ON/OFF image, no actual event timestamps
                logdiff=np.log1p(g)-np.log1p(prev_gray)
                on=np.maximum(logdiff-.045,0)
                off=np.maximum(-logdiff-.045,0)
                # Bright temporal event visualization (no input to detector)
                v_on=(255*np.sqrt(np.clip(on/.25,0,1))).astype(np.uint8)
                v_off=(255*np.sqrt(np.clip(off/.25,0,1))).astype(np.uint8)
                vis=np.zeros((SIDE,SIDE,3),np.uint8)
                vis[...,1]=v_on
                vis[...,0]=v_off;vis[...,2]=v_off
                # Response heat, faint and independent of decision glyph
                hm=raw[k]['activity']
                if hm is not None and np.max(hm)>1e-7:
                    heat=np.clip(hm / max(float(np.quantile(hm,.999)),1e-6),0,1)
                    vis[...,2]=np.maximum(vis[...,2],(70*np.sqrt(heat)).astype(np.uint8))
                vis=np.clip(vis.astype(np.int16)+15,0,255).astype(np.uint8)
            if accepted:
                x,y=map(int,pos)
                cv2.circle(vis,(x,y),7,(0,255,255),1)
                cv2.circle(vis,(x,y),2,(255,255,255),-1)
                vx,vy=raw[k]['direction']
                cv2.arrowedLine(vis,(x,y),(int(round(x+16*vx)),int(round(y+16*vy))), (0,255,255),2,tipLength=.3)
            text=f'Evidence {conf:.2f}'
            cv2.putText(vis,text,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),2,cv2.LINE_AA)
            cv2.putText(vis,text,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
            if not accepted:
                cv2.putText(vis,'No detection',(8,42),cv2.FONT_HERSHEY_SIMPLEX,.5,(220,240,255),1,cv2.LINE_AA)
            visuals.append(vis)
            prev_gray=g
    return pd.DataFrame(results), np.array(visuals,dtype=np.uint8) if make_vis else None


def draw_title(p,title):
    p=p.copy()
    cv2.rectangle(p,(0,0),(SIDE,28),(0,0,0),-1)
    cv2.putText(p,title,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(245,245,245),1,cv2.LINE_AA)
    return p

def movie_with_four_panels(plus,plusvis,base,basevis,outpath,label):
    # Fast software H.264 encoding; writes the moov atom only when the process closes.
    import subprocess
    command=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
             '-s',f'{SIDE*2}x{SIDE*2}','-pix_fmt','bgr24','-r',str(FPS),'-i','-',
             '-an','-c:v','libx264','-preset','ultrafast','-crf','25',
             '-pix_fmt','yuv420p','-movflags','+faststart',str(outpath)]
    proc=subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for a,b,c,d in zip(plus,plusvis,base,basevis):
        top=np.hstack((draw_title(a,f'{label}: original + target'),draw_title(b,f'{label}: DVS-like + STMD')))
        bottom=np.hstack((draw_title(c,'No target: original'),draw_title(d,'No target: DVS-like + STMD')))
        proc.stdin.write(np.vstack((top,bottom)).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f'ffmpeg exit {proc.returncode}')

def calibrate(frames):
    ds=detectors();sc=[[],[]]
    for frame in frames[:100]:
        for j in range(2):sc[j].append(ds[j].step(frame)['peak'])
    return np.array([max(np.quantile(s[25:100],.99),1e-6) for s in sc],dtype=float)

def metrics(df,distance,condition,path='main'):
    r={'distance':distance,'condition':condition,'path':path,'frames':len(df)}
    for name,sel in [('all',slice(None)),('evaluation',slice(100,250))]:
        d=df.iloc[sel]
        r[f'{name}_detection_fraction']=float(d.detected.mean())
        if 'correct' in df:
            r[f'{name}_correct_fraction']=float(d.correct.mean())
            r[f'{name}_localization_fraction']=float((d.error_px<=8).mean())
    return r

def main(only_metrics=False, case_filter=None):
    ROOT.mkdir(parents=True,exist_ok=True)
    OUTPUT=ROOT/'outputs'
    OUTPUT.mkdir(parents=True,exist_ok=True)
    src,fps=load_input()
    source_frame_count=len(src)
    base=make_20s(src)
    del src
    import gc; gc.collect()
    print(f'Source {source_frame_count} frames at {fps} fps. Output {len(base)} frames / {len(base)/FPS}s, last {N-source_frame_count} source frames reversed.',flush=True)
    thresholds=calibrate(base)
    print('Global control-calibrated neural thresholds:',thresholds,flush=True)
    no,novis=evaluate_frames(base,thresholds,make_vis=not only_metrics)
    no.to_csv(OUTPUT/'no_target.csv',index=False)
    summary=[metrics(no,0,'no target')]
    for dist,label in CONDITIONS:
        if case_filter is not None and label!=case_filter: continue
        t0=time.time()
        withdot,gt,apparent=apply_target(base,dist)
        out,vis=evaluate_frames(withdot,thresholds,gt,make_vis=not only_metrics)
        out.to_csv(OUTPUT/f'{label}_target.csv',index=False)
        summary.append(metrics(out,dist,'with target'))
        print(label,'apparent diameter',round(apparent,3),'all accuracy',round(out.correct.mean(),3),'eval accuracy',round(out.correct.iloc[100:250].mean(),3),'seconds',round(time.time()-t0,1),flush=True)
        if not only_metrics:
            movie_with_four_panels(withdot,vis,base,novis,OUTPUT/f'STMD_TUNED_{label}_20s.mp4',label)
            cv2.imwrite(str(OUTPUT/f'preview_{label}.png'),np.vstack((np.hstack((draw_title(withdot[180],f'{label}: original + target'),draw_title(vis[180],f'{label}: DVS-like + STMD'))),np.hstack((draw_title(base[180],'No target: original'),draw_title(novis[180],'No target: DVS-like + STMD'))))))
        del withdot,out,vis
        gc.collect()
    summary_df=pd.DataFrame(summary)
    summary_df.to_csv(OUTPUT/'metrics_summary.csv',index=False)
    # Calibrated thresholds are global; do not use actual GT during inference.
    (OUTPUT/'settings.txt').write_text(f'Source fps {fps}, source frames {source_frame_count}, 20s frames {len(base)}, 320 x 320 FOV 84 degrees, assumed mosquito 6mm\nNeural thresholds {thresholds.tolist()}\nCalibration: no-target frames 25..99 (99th-percentile maximal response)\nEvaluation: frames 100..249, held out from threshold calibration\nThe confidence is an uncalibrated evidence score, not a probability.\n')
    print(summary_df.to_string(index=False),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description='Reproduce the original 500-frame tuned STMD benchmark.')
    p.add_argument('--distance', choices=['50cm','1m','2m'], default=None, help='Optional single distance (default: all)')
    p.add_argument('--render', action='store_true', help='Also render the legacy 640x640 four-panel MP4 videos')
    a=p.parse_args()
    main(only_metrics=not a.render, case_filter=a.distance)
