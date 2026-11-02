"""Competing STMD-like predictive hypotheses in textured moving-house video.

Inference never uses injected target coordinates or the dot-free paired video.
A clean clip is used once to set neural response scales and a detection threshold.
The evidence score is *not* a calibrated posterior probability.
"""
from pathlib import Path
import math, os, time
import cv2
import numpy as np
import pandas as pd
from .multi_stmd_model import Detector, stamp, apparent_diameter_px
cv2.setNumThreads(2)
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'source_indoor_walking.mp4'
SIDE=320
FPS=25.0
N=500
# Three channels: general sensitivity; high local-motion inhibition; texture-tolerant surround.
CONFIGS=[{'decay':.85,'std_scale':7.,'density_weight':0.},
         {'decay':.85,'std_scale':7.,'density_weight':16.},
         {'decay':.78,'std_scale':28.,'density_weight':8.,'response_smoothing':.22}]
DISTANCES=[(.5,'50cm'),(1.,'1m'),(2.,'2m')]


def load_frames(n=N):
    cap=cv2.VideoCapture(str(SOURCE)); fps=cap.get(cv2.CAP_PROP_FPS)
    frames=[]
    while True:
        ok,fr=cap.read()
        if not ok:break
        h,w=fr.shape[:2]
        fr=cv2.resize(fr,(round(w*SIDE/h),SIDE),interpolation=cv2.INTER_AREA)
        x=(fr.shape[1]-SIDE)//2
        frames.append(fr[:,x:x+SIDE])
    cap.release()
    # Extend 14.6-second source to 20 seconds by smoothly reversing movement (not slowing playback).
    ids=list(range(len(frames)))
    if n>len(ids):
        r=list(range(len(frames)-2,-1,-1))
        while len(ids)<n:ids.extend(r)
        ids=ids[:n]
    else: ids=ids[:n]
    return np.asarray(frames,np.uint8)[ids], fps, len(frames)


def fast_trajectory(n):
    """Arclength-interpolated, near-constant-speed wandering path across textured room areas.
    Around 2-4 image pixels/frame, changing direction across wall, figure, furniture, floor.
    Ground truth generated independently and never passed into the detector.
    """
    # Deliberately crosses smooth walls AND people, chairs, plants, patterned floor.
    knots=np.array([(160,58),(266,118),(69,183),(236,249),(46,240),
                    (265,73),(87,57),(245,219),(131,180)],dtype=np.float32)
    # continuous polyline with constant speed along path, bounce at endpoint
    seg=np.linalg.norm(np.diff(knots,axis=0),axis=1)
    cum=np.r_[0.,np.cumsum(seg)]
    # entire path once with mild easing? 8 segments in 500 frames => ~2.5px/frame
    distances=np.linspace(0.,cum[-1],n)
    j=np.clip(np.searchsorted(cum,distances,side='right')-1,0,len(seg)-1)
    u=(distances-cum[j])/seg[j]
    xy=(knots[j]*(1-u[:,None])+knots[j+1]*u[:,None])
    return np.rint(xy).astype(int)


def inject(frames,dist,xy=None):
    frames=np.asarray(frames).copy()
    if xy is None:xy=fast_trajectory(len(frames))
    diameter=apparent_diameter_px(6,dist,84,320)
    for i,(x,y) in enumerate(xy): stamp(frames[i],(int(x),int(y)),diameter)
    return frames,xy,diameter


class LocalFlowResidual:
    """Texture-robust candidate pathway: local optic-flow registration plus
    small dark residual and local motion-density inhibition. Engineering
    approximation of wide-field vs small-target motion segregation.
    """
    def __init__(self):
        self.previous=None
        self.grid_y,self.grid_x=np.mgrid[0:SIDE,0:SIDE].astype(np.float32)
    def step(self,frame,return_map=False):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self.previous is None:
            self.previous=gray
            return {'peak':0.,'activity':np.zeros((SIDE,SIDE),np.float32) if return_map else None}
        low1=cv2.resize(gray,(160,160),interpolation=cv2.INTER_AREA)
        low0=cv2.resize(self.previous,(160,160),interpolation=cv2.INTER_AREA)
        back=cv2.calcOpticalFlowFarneback(low1,low0,None,.5,3,17,2,5,1.1,0)
        back=cv2.resize(back,(SIDE,SIDE),interpolation=cv2.INTER_LINEAR)*2
        warp=cv2.remap(self.previous,self.grid_x+back[...,0],self.grid_y+back[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT101)
        residual=np.maximum(warp-gray,0)
        response=np.maximum(cv2.GaussianBlur(residual,(0,0),.5)-.5*cv2.GaussianBlur(residual,(0,0),3.),0)
        density=cv2.GaussianBlur((residual>12).astype(np.float32),(0,0),4.)
        response*=np.exp(-3*density)
        response[:12,:]=response[-12:,:]=response[:,:12]=response[:,-12:]=0
        self.previous=gray
        return {'peak':float(response.max()), 'activity':response if return_map else None}


def get_detectors():return [Detector(**cfg) for cfg in CONFIGS]+[LocalFlowResidual()]


def calibrate(base,frames=100):
    ds=get_detectors()
    peaks=[[] for _ in ds]
    for fr in base[:frames]:
        for j,d in enumerate(ds): peaks[j].append(d.step(fr)['peak'])
    # 99th percentile among calibration frames, includes no target and moving person.
    thr=np.array([max(float(np.quantile(v[25:],.99)),1e-5) for v in peaks])
    return thr


def candidates_from_maps(maps,scales,num=20,accept_floor=.42,sep=7):
    # A single candidate must be a local maximum in one neural population's
    # response; texture-adapted populations compete in the same proposal space.
    normmaps=np.stack([np.clip(m/s,0,100) for m,s in zip(maps,scales)],axis=0)
    combo=normmaps.max(axis=0)
    combo[:12,:]=combo[-12:,:]=combo[:,:12]=combo[:,-12:]=0
    dil=cv2.dilate(combo,np.ones((7,7),np.uint8))
    mask=(combo>=dil-1e-6)&(combo>accept_floor)
    ys,xs=np.where(mask)
    if len(xs)==0:return [],combo
    values=combo[ys,xs]
    ix=np.argsort(values)[::-1]
    cand=[]
    for i in ix:
        x,y=int(xs[i]),int(ys[i])
        if any((x-a[0])**2+(y-a[1])**2<sep**2 for a in cand): continue
        channel=int(np.argmax(normmaps[:,y,x]))
        cand.append((x,y,float(values[i]),channel))
        if len(cand)>=num:break
    return cand,combo

class Hypothesis:
    nextid=0
    def __init__(self,c,t):
        x,y,s,j=c
        self.id=Hypothesis.nextid;Hypothesis.nextid+=1
        self.x=float(x);self.y=float(y)
        self.vx=0.;self.vy=0.
        self.age=1;self.hits=1;self.missed=0
        self.support=min(float(s),3.5)
        self.dir_agreement=.5;self.history=[1]
        self.channel=j;self.last_measure=(x,y)
        self.birth=t
    def predict(self,shift=(0.,0.)):
        # Per-candidate velocity, not the global winner. Widened gating on rapid turns.
        vx=float(np.clip(self.vx,-8,8));vy=float(np.clip(self.vy,-8,8))
        return np.array([self.x+vx,self.y+vy],dtype=float)
    def match(self,c,t):
        x,y,s,ch=c; px,py=self.x,self.y
        vx=float(x)-px;vy=float(y)-py
        old=np.array([self.vx,self.vy]);v=np.array([vx,vy]);
        if np.linalg.norm(old)>.7 and np.linalg.norm(v)>.7:
            newcoh=(1+np.dot(old,v)/(np.linalg.norm(old)*np.linalg.norm(v)))/2
        else:newcoh=.55
        self.dir_agreement=.75*self.dir_agreement+.25*newcoh
        self.vx=.52*self.vx+.48*vx;self.vy=.52*self.vy+.48*vy
        self.x=float(x);self.y=float(y)
        self.age+=1;self.hits+=1;self.missed=0
        self.support=.82*self.support+.18*min(float(s),4.)
        self.channel=int(ch);self.last_measure=(x,y)
        self.history.append(1);self.history=self.history[-12:]
    def miss(self):
        self.x+=self.vx;self.y+=self.vy
        self.support*=.87;self.vx*=.90;self.vy*=.90
        self.missed+=1;self.age+=1
        self.history.append(0);self.history=self.history[-12:]
    def components(self):
        persistence=sum(self.history)/len(self.history)
        motion=min(np.hypot(self.vx,self.vy)/1.4,1.)
        age=min(self.hits/6.,1.)
        neural=1/(1+np.exp(-np.clip(2*(self.support-1),-15,15)))
        return float(neural),float(persistence),float(self.dir_agreement),float(age),float(motion)
    def confidence(self):
        neural,persist,coh,age,motion=self.components()
        # Heuristic evidence, not calibrated P(target). Penalize missed updates.
        z=-3.1 + 2.1*neural + 1.7*persist + .65*coh + .7*age + .45*motion - .65*self.missed
        return float(1/(1+np.exp(-z)))

class MultiHypothesis:
    def __init__(self,scales,max_tracks=9,threshold=.78,retain_for=5,
                 birth_threshold=1.15,assoc_radius=14):
        self.scales=np.asarray(scales)
        self.max_tracks=max_tracks;self.threshold=threshold
        self.retain_for=retain_for;self.birth_threshold=birth_threshold
        self.assoc_radius=assoc_radius
        self.tracks=[];self.t=0
    def step(self,maps):
        cand,combo=candidates_from_maps(maps,self.scales)
        # Greedy one-to-one assignment: minimize innovation weighted by evidence.
        edges=[]
        for h in self.tracks:
            p=h.predict()
            for k,c in enumerate(cand):
                dist=np.hypot(c[0]-p[0],c[1]-p[1]); gate=self.assoc_radius+2*h.missed
                if dist<=gate:
                    cost=dist/(gate+1e-5)-.10*min(c[2],3)
                    edges.append((cost,h.id,k))
        edges.sort()
        matched=set();assigned=set()
        by_id={h.id:h for h in self.tracks}
        for _,hid,k in edges:
            if hid in matched or k in assigned:continue
            by_id[hid].match(cand[k],self.t)
            matched.add(hid);assigned.add(k)
        for h in self.tracks:
            if h.id not in matched:h.miss()
        self.tracks=[h for h in self.tracks if h.missed<=self.retain_for and
                     12<h.x<308 and 12<h.y<308]
        # Spawn competing proposals everywhere (not just around one prior).
        for k,c in enumerate(cand):
            if k in assigned or c[2]<self.birth_threshold:continue
            if any(np.hypot(c[0]-h.x,c[1]-h.y)<10 for h in self.tracks):continue
            self.tracks.append(Hypothesis(c,self.t))
        self.tracks=sorted(self.tracks,key=lambda h:(h.confidence(),h.support),reverse=True)[:self.max_tracks]
        best=self.tracks[0] if self.tracks else None
        if best:
            conf=best.confidence()
            accepted=(conf>=self.threshold and best.hits>=3 and best.missed<=1)
        else:conf=0.;accepted=False
        data=dict(frame=self.t,detected=bool(accepted),confidence=conf,
                  x=best.x if best else np.nan,y=best.y if best else np.nan,
                  vx=best.vx if best else 0.,vy=best.vy if best else 0.,
                  n_tracks=len(self.tracks),hits=best.hits if best else 0,
                  missed=best.missed if best else 0,
                  neural=best.components()[0] if best else 0.,
                  persistence=best.components()[1] if best else 0.,
                  direction_coherence=best.components()[2] if best else 0.)
        self.t+=1
        return data, [(h.x,h.y,h.confidence(),h.id) for h in self.tracks],combo


def run(frames,scales,gt=None,threshold=.78,birth_threshold=1.15,with_visuals=False):
    ds=get_detectors(); tracker=MultiHypothesis(scales,threshold=threshold,birth_threshold=birth_threshold)
    rows=[];vis=[]; prev_gray=None
    for i,fr in enumerate(frames):
        maps=[d.step(fr,return_map=True)['activity'] for d in ds]
        r,tracks,combo=tracker.step(maps)
        r['time_s']=i/FPS
        if gt is not None:
            r['gt_x'],r['gt_y']=map(int,gt[i]); r['error_px']=float(np.hypot(r['x']-gt[i,0],r['y']-gt[i,1]))
            r['correct']=bool(r['detected'] and r['error_px']<=8)
        rows.append(r)
        if with_visuals:
            vis.append(visualize(fr, prev_gray,combo,r,tracks))
            prev_gray=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
    return pd.DataFrame(rows),np.array(vis,dtype=np.uint8) if with_visuals else None


def visualize(frame,prev_gray,combo,r,tracks):
    g=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    vis=np.full((SIDE,SIDE,3),13,np.uint8)
    if prev_gray is not None:
        delta=np.log1p(g.astype(np.float32))-np.log1p(prev_gray.astype(np.float32))
        on=np.sqrt(np.clip((delta-.038)/.24,0,1))
        off=np.sqrt(np.clip((-delta-.038)/.24,0,1))
        vis[...,1]=np.clip(vis[...,1]+242*on,0,255).astype(np.uint8)
        v_off=np.clip(242*off+13,0,255).astype(np.uint8)
        vis[...,0]=v_off;vis[...,2]=v_off
        if combo is not None:
            heat=np.sqrt(np.clip(combo/3.0,0,1))
            vis[...,2]=np.maximum(vis[...,2],(65*heat).astype(np.uint8))
    # Predicted alternatives: small muted blue rings, primary yellow on acceptance.
    for x,y,c,id in tracks[1:5]:
        cv2.circle(vis,(round(x),round(y)),5,(140,90,0),1)
    if r['detected']:
        x,y=round(r['x']),round(r['y'])
        cv2.circle(vis,(x,y),8,(0,240,255),1)
        cv2.circle(vis,(x,y),2,(255,255,255),-1)
        vx,vy=r['vx'],r['vy'];v=np.hypot(vx,vy)
        if v>.2:
            cv2.arrowedLine(vis,(x,y),(round(x+16*vx/v),round(y+16*vy/v)),(0,240,255),2,tipLength=.3)
    cv2.putText(vis,f'Confidence {r["confidence"]:.2f}   Hyp {r["n_tracks"]}',(7,20),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA)
    if not r['detected']:
        cv2.putText(vis,'No detection',(7,42),cv2.FONT_HERSHEY_SIMPLEX,.48,(205,225,255),1,cv2.LINE_AA)
    return vis
