"""Experimental full-field STMD-like target detector with motion-channel facilitation.
No target coordinates or dot-free reference frames enter inference.
"""
import sys,time,os,cv2,numpy as np,pandas as pd,math
from pathlib import Path
cv2.setNumThreads(2)
ROOT=Path(__file__).resolve().parent
def apparent_diameter_px(length_mm, distance_m, fov_deg=84.0, res_px=320):
    # angular-width approximation used by previous demos
    angle = math.degrees(2*math.atan((length_mm/1000)/(2*distance_m)))
    return angle*res_px/fov_deg
H=W=320
FPS=25.0
V=np.array([(dx,dy) for dx in (-1.0,-0.5,0.0,.5,1.0) for dy in (-.5,0,.5) if not(dx==0 and dy==0)],dtype=np.float32)
# gaussian surround statistics excluding the receptive-field center
sigma=3.0
r=np.arange(-8,9,dtype=np.float32)
xx,yy=np.meshgrid(r,r)
G=np.exp(-(xx*xx+yy*yy)/(2*sigma*sigma));G/=G.sum()
G5=G[6:11,6:11].copy();G5[(xx[6:11,6:11]**2+yy[6:11,6:11]**2)>2.5**2]=0
WMASS=G5.sum(); G5=G5.astype(np.float32)
# optimize overlay using pixel-integrated Gaussian PSF; dot diameter projected in pixels

def stamp(frame,xy,diameter,blur=.42):
    x,y=map(int,xy)
    R=6;tile=frame[y-R:y+R+1,x-R:x+R+1]
    nsub=12; yy,xx=np.mgrid[0:(2*R+1)*nsub,0:(2*R+1)*nsub]
    u=(xx+.5)/nsub-R-.5;v=(yy+.5)/nsub-R-.5
    disk=((u*u+v*v)<(diameter/2)**2).astype(np.float32)
    coverage=cv2.resize(disk,(2*R+1,2*R+1),interpolation=cv2.INTER_AREA)
    coverage=cv2.GaussianBlur(coverage,(0,0),blur)
    frame[y-R:y+R+1,x-R:x+R+1]=np.clip(tile.astype(np.float32)*(1-coverage[...,None]),0,255).astype('uint8')

def path(t,n,mode='original'):
    if mode=='original':
        x=round(165+110*t/max(1,n-1))
        y=round(68+9*np.sin(2*np.pi*t/120+.4))
    elif mode=='heldout':
        # a separate path unknown to inference; traverses different scene regions
        x=round(272-126*t/max(1,n-1))
        y=round(129+20*np.sin(2*np.pi*t/93+.4))
    elif mode=='upper_reverse':
        x=round(275-110*t/max(1,n-1))
        y=round(68+9*np.sin(2*np.pi*t/120+.4))
    return int(x),int(y)


def make_frames(base,dist,mode='original'):
    if dist is None: return np.asarray(base)
    out=np.asarray(base).copy()
    gt=[];size=apparent_diameter_px(6,dist)
    for i in range(len(out)):
        xy=path(i,len(out),mode)
        stamp(out[i],xy,size)
        gt.append(xy)
    return out, np.array(gt)

def shift(arr,dx,dy,border=cv2.BORDER_REFLECT):
    M=np.float32([[1,0,dx],[0,1,dy]])
    return cv2.warpAffine(arr,M,(W,H),flags=cv2.INTER_LINEAR,borderMode=border)

class Detector:
    def __init__(self, decay=.85, std_scale=7.0, response_smoothing=.35, density_weight=0.0, density_threshold=10.0):
        self.bg=None;self.previous=None
        self.A=np.zeros((len(V),H,W),np.float32)
        self.decay=decay; self.std_scale=std_scale; self.response_smoothing=response_smoothing
        self.frame=0
        self.density_weight=density_weight; self.density_threshold=density_threshold
    def step(self, frame,return_map=False):
        g=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self.bg is None:
            self.bg=g.copy();self.previous=g.copy();self.frame+=1
            z=np.zeros((H,W),np.float32)
            return dict(peak=0,position=(np.nan,np.nan),direction=(0,0),activity=z if return_map else None,phase=(0,0),raw_activity=z if return_map else None)
        dx,dy=cv2.phaseCorrelate(self.previous,g)[0]
        pred_bg=shift(self.bg,dx,dy)
        residual=np.maximum(pred_bg-g,0)
        response=np.maximum(cv2.GaussianBlur(residual,(0,0),.55)-cv2.GaussianBlur(residual,(0,0),3.0),0)
        # annular Gaussian spatial texture from the present frame, excluding current center
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
        # full-field temporally selective populations: shift then leaky integrate
        for j,(vx,vy) in enumerate(V):
            self.A[j] = self.decay*shift(self.A[j],float(vx),float(vy),cv2.BORDER_CONSTANT) + evidence
        peakmap=self.A.max(axis=0)
        peakmap[:12,:]=peakmap[-12:,:]=peakmap[:,:12]=peakmap[:,-12:]=0
        _,best,_,xy=cv2.minMaxLoc(peakmap)
        x,y=xy; k=int(np.argmax(self.A[:,y,x]))
        # update background slowly using past frames only
        self.bg=.976*pred_bg+.024*g
        self.previous=g;self.frame+=1
        return dict(peak=best,position=xy,direction=tuple(map(float,V[k])),activity=peakmap.copy() if return_map else None, phase=(dx,dy),raw_activity=evidence.copy() if return_map else None)

def run(frames,gt=None,threshold=20,decay=.85,record_maps=False,**kwargs):
    detector=Detector(decay=decay,**kwargs)
    rows=[];maps=[]
    for i,f in enumerate(frames):
        s=detector.step(f,return_map=record_maps)
        x,y=s['position'];score=float(s['peak'])
        r=dict(frame=i,score=score,x=x,y=y,dx=s['direction'][0],dy=s['direction'][1],detected=score>=threshold, confidence=1/(1+np.exp(-np.clip((score-threshold)/6,-20,20))))
        if gt is not None:
            err=np.hypot(x-gt[i,0],y-gt[i,1]); r.update(err=err,correct=err<=8 and score>=threshold)
        rows.append(r)
        if record_maps:maps.append((s['activity'],s['raw_activity']))
    return pd.DataFrame(rows),maps

