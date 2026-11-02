"""Spatially widespread, reproducible randomized insect trajectories.

Four by four scene tiles are each visited. Random initial position and route
per seed; smooth correlated lateral steering noise is added after interpolation.
All distances and detectors use the identical per-trial integer coordinates.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

SENSOR=320
MARGIN=16

def generate_trajectory(seed, n=365, sensor=SENSOR):
    rng=np.random.default_rng(seed)
    coords=np.linspace(42,sensor-42,4)
    centers=np.array([(x,y) for y in coords for x in coords],dtype=float)
    centers+=rng.uniform(-13,13,centers.shape)
    start=rng.uniform(MARGIN+10,sensor-MARGIN-10,size=2)
    visited=[]
    unused=list(range(len(centers)))
    prev=start.copy()
    while unused:
        dist=np.linalg.norm(centers[unused]-prev,axis=1)
        order=np.argsort(dist)
        # Mostly local waypoints, sometimes a slightly less direct route.
        chosen=int(order[min(int(rng.choice(3,p=[.70,.23,.07])),len(order)-1)])
        ix=unused.pop(chosen)
        visited.append(centers[ix]); prev=centers[ix]
    knots=np.vstack([start, np.array(visited)])
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(knots,axis=0),axis=1))]
    ts=np.linspace(0,arc[-1],n)
    xy=np.c_[PchipInterpolator(arc,knots[:,0])(ts), PchipInterpolator(arc,knots[:,1])(ts)]
    # Small low-frequency angular meanders, not straight segments.
    wiggle=rng.normal(0,1,(n,2))
    wiggle=gaussian_filter1d(wiggle, sigma=8, axis=0,mode='reflect')
    sd=wiggle.std(axis=0)
    xy+=wiggle/(sd+1e-6)*3.0
    xy=np.clip(xy,MARGIN,sensor-MARGIN)
    out=np.rint(xy).astype(np.int32)
    tile=np.clip((out//(sensor//4)),0,3)
    occupancy=len({(int(a),int(b)) for a,b in tile})
    if occupancy<16:
        raise RuntimeError(f'trajectory did not traverse all 16 tiles: {occupancy}')
    return out

if __name__=='__main__':
    for seed in [41,92,173,281]:
        path=generate_trajectory(seed)
        print(seed,tuple(path[0]), np.mean(np.linalg.norm(np.diff(path.astype(float),axis=0),axis=1)))
