"""Causal motion-predictive normalized 2D spatial activity field.

Unlike the earlier annular-inhibition heuristic, this memory predicts the next
location by translating its existing field with a velocity estimated from the
model's *own* past selected peaks. No true coordinates are accessed.

A bounded feedback gain boosts present sensory candidates supported by the
prediction. Several candidates deposit broad Gaussian bumps into one shared
2D field. Normalization is across the whole spatial field (not per candidate).
This is an attractor-*like* predictive field, not a biophysical neural CANN.
"""
import cv2
import numpy as np

SIZE=320

class PredictiveField:
    def __init__(self, decay=0.86, gain=1.4, sigma=5.0, drift=0.6,
                 n_peaks=12, winner_boost=0.45, velocity_smooth=0.55,
                 match_radius=18.0, max_speed=6.0):
        self.decay=float(decay); self.gain=float(gain)
        self.sigma=float(sigma); self.drift=float(drift)
        self.n_peaks=int(n_peaks); self.winner_boost=float(winner_boost)
        self.velocity_smooth=float(velocity_smooth)
        self.match_radius=float(match_radius); self.max_speed=float(max_speed)
        self.M=np.zeros((SIZE,SIZE),np.float32)
        self.velocity=np.zeros(2,np.float32)
        self.prev_xy=None
        self.last_prediction=np.zeros_like(self.M)
        self.last_sensory=np.zeros_like(self.M)

    def step(self,candidates,return_maps=False):
        c=np.asarray(candidates,np.float32).reshape(-1,4)
        # Prediction is causal: shift PREVIOUS activity using velocity from
        # previous matches. Broadening prevents overconfident point lock-in.
        mat=np.float32([[1.,0.,float(self.velocity[0])],[0.,1.,float(self.velocity[1])]])
        p=cv2.warpAffine(self.M,mat,(SIZE,SIZE),flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        if self.drift>0: p=cv2.GaussianBlur(p,(0,0),self.drift)
        peak=float(p.max())
        if peak>1e-10:p/=peak
        p=np.clip(p,0,1)
        self.last_prediction=p.copy()
        if not len(c):
            self.M=self.decay*p
            self.velocity*=.93
            return dict(x=np.nan,y=np.nan,score=0.,prior=p.copy(),
                        memory=self.M.copy(),sensory=np.zeros_like(self.M),
                        chosen=-1,velocity=self.velocity.copy())

        x=np.clip(np.rint(c[:,0]).astype(np.int32),0,SIZE-1)
        y=np.clip(np.rint(c[:,1]).astype(np.int32),0,SIZE-1)
        strength=np.maximum(c[:,2],0)
        score=strength*(1+self.gain*p[y,x])
        chosen=int(np.argmax(score))
        pos=np.array([x[chosen],y[chosen]],np.float32)
        # Velocity update only for successive nearby detections; if a new
        # target appears elsewhere, the prediction resets rather than linking
        # unrelated local maxima.
        if self.prev_xy is not None:
            step=pos-self.prev_xy
            if np.linalg.norm(step) <= self.match_radius:
                self.velocity=(self.velocity_smooth*self.velocity+
                               (1-self.velocity_smooth)*step)
                norm=float(np.linalg.norm(self.velocity))
                if norm>self.max_speed:self.velocity*=self.max_speed/norm
            else:self.velocity*=0.2
        self.prev_xy=pos

        # Sparse bounded sensory injections (not grounded in target truth).
        inj=np.zeros((SIZE,SIZE),np.float32)
        s_max=max(float(strength.max()),1e-6)
        for k in range(min(len(c),self.n_peaks)):
            value=float(np.clip(strength[k]/s_max,0,1))
            value*=1 if k==chosen else .18
            inj[y[k],x[k]]=max(inj[y[k],x[k]], value)
        inj=cv2.GaussianBlur(inj,(0,0),self.sigma)
        inj_max=float(inj.max())
        if inj_max>1e-10:inj/=inj_max
        self.last_sensory=inj.copy()
        # Recurrent feedback + sensory injection; global max normalization.
        M=self.decay*p+(1-self.decay)*inj
        # keep winner salient but not exclusive
        self.M=M/max(float(M.max()),1e-10)
        self.M=np.clip(self.M,0,1)
        return dict(x=float(pos[0]),y=float(pos[1]),score=float(score[chosen]),
                    prior=p.copy(),memory=self.M.copy(),sensory=inj.copy(),
                    chosen=chosen,velocity=self.velocity.copy())


def infer(candidate_array,counts,config=None,return_maps=False):
    field=PredictiveField(**(config or {}))
    n=len(counts)
    out=np.full((n,2),np.nan,np.float32)
    score=np.zeros(n,np.float32)
    vel=np.zeros((n,2),np.float32)
    maps=[]
    for t in range(n):
        c=candidate_array[t,:int(counts[t])]
        d=field.step(c,return_maps)
        out[t]=[d['x'],d['y']]
        score[t]=d['score'];vel[t]=d['velocity']
        if return_maps: maps.append(d)
    if return_maps:return out,score,vel,maps
    return out,score,vel
