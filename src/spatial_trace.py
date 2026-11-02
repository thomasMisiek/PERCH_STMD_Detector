"""Causal unified 2-D spatial memory driven by STMD candidate peaks.

SpatialTrace stores ONE floating-point 320x320 memory map. Every frame it
(1) queries a blurred version of the *previous* trace at present candidates,
(2) optionally subtracts a narrow Gaussian to suppress stationary re-activation,
(3) multiplies the sensory evidence by temporal support, then
(4) decays and updates the 2D trace from all sensory candidates.
No ground-truth coordinates are consulted by the model.
"""
import cv2
import numpy as np

N=320

class SpatialTrace:
    def __init__(self, decay=.8, neighborhood=5., inhibition=.7, gain=5., mode='annulus', n_peaks=25):
        self.decay=float(decay);self.neighborhood=float(neighborhood)
        self.inhibition=float(inhibition);self.gain=float(gain)
        self.mode=mode;self.n_peaks=int(n_peaks)
        self.trace=np.zeros((N,N),np.float32)

    def step(self, candidates):
        """candidates: [K,4] x,y,neural_strength,channel; returns scores and map.
        For inference current candidate scores are only multiplied by trace from
        preceding frames. This avoids self-reinforcement in the same frame.
        """
        c=np.asarray(candidates,dtype=np.float32)
        if len(c)==0:
            self.trace*=self.decay
            return np.zeros(0,np.float32),self.trace.copy()
        x=np.clip(np.rint(c[:,0]).astype(int),0,N-1)
        y=np.clip(np.rint(c[:,1]).astype(int),0,N-1)
        if self.mode=='raw':
            support=np.zeros(len(c),np.float32)
        else:
            wide=cv2.GaussianBlur(self.trace,(0,0),self.neighborhood)
            if self.mode=='annulus':
                narrow=cv2.GaussianBlur(self.trace,(0,0),1.2)
                support=np.maximum(wide[y,x] - self.inhibition*narrow[y,x],0)
            elif self.mode=='leaky':
                support=wide[y,x]
            else:raise ValueError(self.mode)
        strength=np.maximum(c[:,2],0)
        # Use a bounded positive multiplicative factor. No target-specific priors.
        scores=strength*(1+self.gain*np.clip(support,0,10))
        if self.mode!='raw':
            self.trace*=self.decay
            # Update all local maxima, not only the winner: no early lock-in.
            top=min(len(c),self.n_peaks)
            # Saturate large responses, scale to consistent neural strengths.
            weights=np.minimum(strength[:top],3.0)/3.0
            # A very small 3x3 neural receptive-field footprint.
            for px,py,weight in zip(x[:top],y[:top],weights):
                if 2<=px<N-2 and 2<=py<N-2:
                    self.trace[py,px]+=weight
                    self.trace[py-1,px]+=.33*weight
                    self.trace[py+1,px]+=.33*weight
                    self.trace[py,px-1]+=.33*weight
                    self.trace[py,px+1]+=.33*weight
        return scores, self.trace


def infer(candidate_array,counts, config=None, return_memory=False):
    """Output one proposed location per frame (forced-choice, no GT) + evidence.
    A no-candidate frame produces NaN position and no proposal.
    """
    cfg={} if config is None else dict(config)
    model=SpatialTrace(**cfg)
    n=len(counts)
    xy=np.full((n,2),np.nan,dtype=np.float32)
    strength=np.full(n,np.nan,np.float32)
    raw=np.full(n,np.nan,np.float32)
    trace_frames=[]
    for t in range(n):
        c=candidate_array[t,:int(counts[t])]
        scores,memory=model.step(c)
        if len(scores):
            best=int(np.argmax(scores))
            xy[t]=c[best,:2]; strength[t]=scores[best];raw[t]=c[best,2]
        if return_memory:trace_frames.append(memory.copy())
    if return_memory:return xy,strength,raw,np.stack(trace_frames)
    return xy,strength,raw
