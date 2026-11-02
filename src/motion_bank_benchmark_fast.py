from pathlib import Path
import cv2, numpy as np, pandas as pd
from experiment import read_source, inject
from trajectories import generate_trajectory
from spatial_trace import infer
from evaluate import angular_error_deg
from comparison.multi_hypothesis import LocalFlowResidual, candidates_from_maps, CONFIGS
cv2.setNumThreads(2)
ROOT=Path(__file__).resolve().parent.parent
SIDE=320
TEST=(173,281)
DISTANCES=((0.5,'50cm'),(1.0,'1m'),(2.0,'2m'))
BEST_TRACE=dict(mode='annulus',decay=.92,neighborhood=11.,inhibition=1.3,gain=70.,n_peaks=25)
EVAL_START=100

sigma=3.0
r=np.arange(-8,9,dtype=np.float32)
xx,yy=np.meshgrid(r,r)
G=np.exp(-(xx*xx+yy*yy)/(2*sigma*sigma));G/=G.sum()
G5=G[6:11,6:11].copy();G5[(xx[6:11,6:11]**2+yy[6:11,6:11]**2)>2.5**2]=0
WMASS=G5.sum(); G5=G5.astype(np.float32)

def velocity_set(name):
    if name=='grid9': vals=[-1.0,0.0,1.0]
    elif name=='grid24': vals=[-1.0,-0.5,0.0,0.5,1.0]
    else: raise ValueError(name)
    return np.array([(dx,dy) for dx in vals for dy in vals if not(dx==0 and dy==0)],dtype=np.float32)

def shift(arr,dx,dy,border=cv2.BORDER_REFLECT):
    M=np.float32([[1,0,dx],[0,1,dy]])
    return cv2.warpAffine(arr,M,(SIDE,SIDE),flags=cv2.INTER_LINEAR,borderMode=border)

class MotionBankDetector:
    def __init__(self,V, decay=.85, std_scale=7.0, response_smoothing=.35, density_weight=0.0, density_threshold=10.0):
        self.V=np.asarray(V,np.float32)
        self.bg=None; self.previous=None
        self.A=np.zeros((len(self.V),SIDE,SIDE),np.float32)
        self.decay=decay; self.std_scale=std_scale; self.response_smoothing=response_smoothing
        self.density_weight=density_weight; self.density_threshold=density_threshold
    def step(self, frame, return_map=False, return_stack=False):
        g=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self.bg is None:
            self.bg=g.copy(); self.previous=g.copy()
            z=np.zeros((SIDE,SIDE),np.float32)
            out=dict(peak=0,position=(np.nan,np.nan),activity=z if return_map else None,evidence=z.copy(),residual=z.copy())
            if return_stack: out['stack']=self.A.copy()
            return out
        dx,dy=cv2.phaseCorrelate(self.previous,g)[0]
        pred_bg=shift(self.bg,dx,dy)
        residual=np.maximum(pred_bg-g,0)
        response=np.maximum(cv2.GaussianBlur(residual,(0,0),.55)-cv2.GaussianBlur(residual,(0,0),3.0),0)
        mu=(cv2.GaussianBlur(g,(0,0),sigma)-cv2.filter2D(g,-1,G5))/(1-WMASS)
        variance=(cv2.GaussianBlur(g*g,(0,0),sigma)-cv2.filter2D(g*g,-1,G5))/(1-WMASS)-mu*mu
        sd=np.sqrt(np.maximum(variance,0))
        evidence=response*np.exp(-sd/self.std_scale)
        if self.density_weight>0:
            active=(residual>self.density_threshold).astype(np.float32)
            density=cv2.GaussianBlur(active,(0,0),3.0)
            evidence*=np.exp(-self.density_weight*density)
        if self.response_smoothing>0:
            evidence=cv2.GaussianBlur(evidence,(0,0),self.response_smoothing)
        for j,(vx,vy) in enumerate(self.V):
            self.A[j] = self.decay*shift(self.A[j],float(vx),float(vy),cv2.BORDER_CONSTANT) + evidence
        peakmap=self.A.max(axis=0)
        peakmap[:12,:]=peakmap[-12:,:]=peakmap[:,:12]=peakmap[:,-12:]=0
        _,best,_,xy=cv2.minMaxLoc(peakmap)
        self.bg=.976*pred_bg+.024*g
        self.previous=g
        out=dict(peak=float(best),position=xy,direction=(0,0),activity=peakmap.copy() if return_map else None,evidence=evidence.copy() if return_map else None,residual=residual.copy() if return_map else None)
        if return_stack: out['stack']=self.A.copy()
        return out

def get_detectors_custom(V):
    return [MotionBankDetector(V,**cfg) for cfg in CONFIGS] + [LocalFlowResidual()]

def calibrate_custom(base, V, frames=100):
    ds=get_detectors_custom(V)
    peaks=[[] for _ in ds]
    for fr in base[:frames]:
        for j,d in enumerate(ds): peaks[j].append(d.step(fr)['peak'])
    return np.array([max(float(np.quantile(v[25:],.99)),1e-5) for v in peaks])

def extract_candidates(frames, V, scales):
    ds=get_detectors_custom(V)
    cand_frames=[]; counts=[]
    for fr in frames:
        maps=[d.step(fr,return_map=True)['activity'] for d in ds]
        cand, combo=candidates_from_maps(maps, scales, num=40, accept_floor=.22, sep=5)
        counts.append(len(cand))
        cand_frames.append(np.array(cand,np.float32) if len(cand) else np.zeros((0,4),np.float32))
    maxc=max(len(c) for c in cand_frames)
    cand_arr=np.zeros((len(cand_frames),maxc,4),np.float32)
    for i,c in enumerate(cand_frames):
        if len(c): cand_arr[i,:len(c)] = c
    return cand_arr, np.array(counts)

def run_model(frames, gt, V, scales):
    cand,count=extract_candidates(frames,V,scales)
    xy,strength,raw=infer(cand,count,BEST_TRACE)
    err=angular_error_deg(xy,gt)
    good=np.isfinite(err)&(err<=5)
    df=pd.DataFrame(dict(frame=np.arange(len(gt)), x=xy[:,0], y=xy[:,1], score=strength, raw_score=raw, gt_x=gt[:,0], gt_y=gt[:,1], error_deg=err, correct_5deg=good))
    return df

def summarize(df):
    d=df.iloc[EVAL_START:]
    return dict(eval_frames=int(len(d)), correct=int(d.correct_5deg.sum()), correct_pct=float(100*d.correct_5deg.mean()), mean_err=float(np.nanmean(d.error_deg)), median_err=float(np.nanmedian(d.error_deg)))

def interpolate_frame(a,b,t):
    if t<=0: return a
    if t>=1: return b
    return np.clip(a.astype(np.float32)*(1-t)+b.astype(np.float32)*t,0,255).astype(np.uint8)

def background_variant(base, speed):
    n=len(base)
    if speed==0:
        return np.repeat(base[:1], n, axis=0)
    out=[]
    maxidx=len(base)-1
    for i in range(n):
        pos=min(i*speed, maxidx)
        j=int(np.floor(pos)); k=min(j+1,maxidx); t=pos-j
        out.append(interpolate_frame(base[j],base[k],t))
    return np.stack(out)

def compare_motion_sets():
    base=read_source()
    rows=[]
    for bank in ('grid9','grid24'):
        V=velocity_set(bank)
        print('calibrating',bank,flush=True)
        scales=calibrate_custom(base,V)
        for seed in TEST:
            path=generate_trajectory(seed,len(base))
            for dist,label in DISTANCES:
                print('run',bank,seed,label,flush=True)
                frames=inject(base,path,dist)
                df=run_model(frames,path,V,scales)
                sm=summarize(df)
                rows.append(dict(model=bank, seed=seed, distance=label, n_channels=len(V), **sm))
    out=pd.DataFrame(rows)
    out.to_csv(ROOT/'results'/'motion_bank_9_vs_24.csv',index=False)
    return out

def camera_speed_experiment(best_bank='grid24'):
    base=read_source()
    V=velocity_set(best_bank)
    conds=[('still',0.0),('slow_bg',0.5),('normal_bg',1.0)]
    rows=[]
    # Shared source calibration; do not re-calibrate on a static sequence.
    scales=calibrate_custom(base,V)
    for cname,speed in conds:
        print('background',cname,flush=True)
        bg=background_variant(base,speed)
        for seed in TEST:
            path=generate_trajectory(seed,len(bg))
            for dist,label in DISTANCES:
                print('run',cname,seed,label,flush=True)
                frames=inject(bg,path,dist)
                df=run_model(frames,path,V,scales)
                sm=summarize(df)
                rows.append(dict(condition=cname,bg_speed=speed,seed=seed,distance=label,**sm))
    out=pd.DataFrame(rows)
    out.to_csv(ROOT/'results'/'camera_speed_experiment.csv',index=False)
    return out

if __name__=='__main__':
    mb=compare_motion_sets()
    print('\nMotion bank summary')
    print(mb.groupby(['model','distance']).correct_pct.mean())
    held=mb.groupby('model').correct_pct.mean().sort_values(ascending=False)
    best=held.index[0]
    print('BEST',best)
    cam=camera_speed_experiment(best)
    print('\nCamera speed summary')
    print(cam.groupby(['condition','distance']).correct_pct.mean())
