"""End-to-end visualization of the ACTUAL intensity-frame STMD computations.

The simulated DVS panel is VISUALIZATION ONLY and does not feed neural input.
The 24 channel tiles plot A_k - mean_j(A_j), a visualization of actual
computed recurrent arrays; the model itself pools the unmodified A_k arrays.
Ground truth (red circle) is drawn ONLY on the input panel; never used by model.
"""
from pathlib import Path
import cv2, numpy as np
from experiment import read_source,inject
from trajectories import generate_trajectory
from motion_bank_benchmark_fast import velocity_set,calibrate_custom,MotionBankDetector
from comparison.multi_hypothesis import CONFIGS,LocalFlowResidual,candidates_from_maps
from predictive_field import PredictiveField
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'media'
OUT.mkdir(exist_ok=True,parents=True)
SIDE=320; S=280; TITLE=30; FOOT=22
V=velocity_set('grid24')
CFG=dict(decay=.85,gain=.3,sigma=5,drift=.8,n_peaks=10)
# Exact visualization source; DVS-style simulated display, not event-camera output

def dvs_view(curr,prev):
    if prev is None:return np.full((320,320,3),20,np.uint8)
    g=np.log1p(cv2.cvtColor(curr,cv2.COLOR_BGR2GRAY).astype(np.float32))
    h=np.log1p(cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY).astype(np.float32))
    d=g-h
    on=np.clip((d-.04)/.25,0,1)
    off=np.clip((-d-.04)/.25,0,1)
    img=np.full((320,320,3),18,np.uint8)
    img[:,:,1]=(18+225*np.sqrt(on)).astype(np.uint8)
    img[:,:,0]=(18+230*np.sqrt(off)).astype(np.uint8)
    img[:,:,2]=(18+230*np.sqrt(off)).astype(np.uint8)
    return img


def activity_rgb(arr, maxv=None):
    a=np.nan_to_num(np.asarray(arr,np.float32))
    if maxv is None: maxv=max(float(np.percentile(a,99.7)),1e-6)
    x=np.clip(a/maxv,0,1)
    x8=(x*255).astype(np.uint8)
    return cv2.applyColorMap(x8,cv2.COLORMAP_INFERNO)


def feature_mosaic(stacks, center=(160,160)):
    """24 true A[k] responses shown as signed, contrast-normalized ROI crops.

    Cropping is for display only, centred on strongest current sensory peak,
    never on target ground truth. Max pooling still uses original full maps.
    The displayed signed difference is A[k] - average_k(A[k]).
    """
    a=np.asarray(stacks,np.float32)
    assert a.shape==(24,320,320), f"Expected 24 actual velocity maps; got {a.shape}"
    contrast=a-a.mean(axis=0,keepdims=True)
    cx=int(np.clip(center[0],36,284)); cy=int(np.clip(center[1],36,284))
    radius=36
    crop=contrast[:,cy-radius:cy+radius,cx-radius:cx+radius]
    # One common scale for ALL 24 maps, not per-channel normalization.
    maxv=max(float(np.percentile(np.abs(crop),99.5)),1e-6)
    tile_w=44; tile_h=45; w=6*tile_w; h=4*tile_h
    out=np.full((h,w,3),245,np.uint8)
    for k in range(24):
        r,c=divmod(k,6)
        raw=cv2.resize(crop[k],(tile_w-2,tile_h-14),interpolation=cv2.INTER_LINEAR)
        u=np.clip(raw/maxv,-1,1)
        # Signed response: warm = selective excess, blue = below bank mean,
        # black = common component. This is a diagnostic, not model input.
        tile=np.zeros((tile_h-14,tile_w-2,3),np.uint8)
        tile[:,:,0]=(215*np.maximum(-u,0)).astype(np.uint8)
        tile[:,:,1]=(105*np.abs(u)).astype(np.uint8)
        tile[:,:,2]=(255*np.maximum(u,0)).astype(np.uint8)
        xx=c*tile_w; yy=r*tile_h
        out[yy+12:yy+12+tile_h-14,xx+1:xx+1+tile_w-2]=tile
        vx,vy=V[k]
        label=f'{vx:+g},{vy:+g}'
        cv2.putText(out,label,(xx+1,yy+9),cv2.FONT_HERSHEY_PLAIN,.64,(20,20,20),1,cv2.LINE_AA)
        cv2.rectangle(out,(xx,yy),(xx+tile_w-1,yy+tile_h-1),(165,165,165),1)
    return out,contrast,maxv


def four_pathways(maps):
    out=np.full((S-12,S-12,3),245,np.uint8)
    names=['STMD A','STMD B','STMD C','local flow']
    for k,m in enumerate(maps):
        r,c=divmod(k,2)
        x=c*134; y=r*134
        tile=cv2.resize(activity_rgb(m),(132,110),interpolation=cv2.INTER_AREA)
        out[y+20:y+130,x:x+132]=tile
        cv2.putText(out,names[k],(x+2,y+15),cv2.FONT_HERSHEY_SIMPLEX,.39,(32,32,32),1,cv2.LINE_AA)
    return out


def labelled_panel(img,name,foot='',circle=None,dot=None):
    a=cv2.resize(img,(S,S),interpolation=cv2.INTER_LINEAR)
    out=np.full((TITLE+S+FOOT,S,3),255,np.uint8)
    out[TITLE:TITLE+S]=a
    cv2.putText(out,name,(7,21),cv2.FONT_HERSHEY_SIMPLEX,.49,(25,25,25),1,cv2.LINE_AA)
    if foot:cv2.putText(out,foot,(7,TITLE+S+15),cv2.FONT_HERSHEY_SIMPLEX,.35,(65,65,65),1,cv2.LINE_AA)
    if circle is not None:
        x,y=circle;pt=(int(round(x*S/320)),int(round(y*S/320))+TITLE)
        cv2.circle(out,pt,16,(5,5,230),1,cv2.LINE_AA)
    if dot is not None and np.isfinite(dot[0]):
        x,y=dot;pt=(int(round(x*S/320)),int(round(y*S/320))+TITLE)
        cv2.circle(out,pt,11,(0,230,255),2,cv2.LINE_AA)
    return out


def main(seed=173,distance=1.0):
    base=read_source()
    path=generate_trajectory(seed,len(base))
    frames=inject(base,path,distance)
    scales=calibrate_custom(base,V)
    ds=[MotionBankDetector(V,**cfg) for cfg in CONFIGS]
    local=LocalFlowResidual()
    mem=PredictiveField(**CFG)
    width=(S+8)*4+8; height=(TITLE+S+FOOT+8)*2+8
    outpath=OUT/'stmd_actual_pipeline_1m.mp4'
    writer=cv2.VideoWriter(str(outpath),cv2.VideoWriter_fourcc(*'mp4v'),25,(width,height))
    if not writer.isOpened():raise RuntimeError('Cannot open MP4 output')
    prev=None
    selectivity=[]
    for t,fr in enumerate(frames):
        maps=[];stacks=None; evidence=None
        for j,d in enumerate(ds):
            result=d.step(fr,return_map=True,return_stack=(j==0))
            maps.append(result['activity'])
            if j==0:
                stacks=result['stack']; evidence=result['evidence']
        maps.append(local.step(fr,return_map=True)['activity'])
        cand,pooled=candidates_from_maps(maps,scales,num=40,accept_floor=.22,sep=5)
        candidates=np.array(cand,dtype=np.float32).reshape(-1,4)
        pred=mem.step(candidates,True)
        center=cand[0][:2] if cand else (160,160)
        mosaic,contrast,maxv=feature_mosaic(stacks,center=center)
        selectivity.append(float(np.sqrt(np.mean(contrast**2))/(np.sqrt(np.mean(stacks**2))+1e-6)))
        pool=activity_rgb(pooled)
        for x,y,strength,channel in cand[:18]:
            cv2.circle(pool,(int(x),int(y)),3,(45,240,100),1,cv2.LINE_AA)
        panels=[
            labelled_panel(fr,'1  RGB frame (target)',f'frame {t} | ground truth ring',circle=path[t]),
            labelled_panel(dvs_view(fr,prev),'2  simulated DVS','display only; NOT model input'),
            labelled_panel(activity_rgb(evidence),'3  small-target detector','computed from intensity frames'),
            labelled_panel(mosaic,'4  24 computed motion maps','signed response; pooled-peak ROI'),
            labelled_panel(pool,'5  fused sensory response',f'{len(cand)} candidate peaks (green)'),
            labelled_panel(activity_rgb(pred['prior'],1.0),'6  Predictive spatial map','spatial activity only; no overlays'),
            labelled_panel(fr,'7  Predicted target position','yellow ring = model prediction',dot=(pred['x'],pred['y']))
        ]
        canvas=np.full((height,width,3),235,np.uint8)
        for i,p in enumerate(panels):
            r,c=divmod(i,4)
            # Centre three bottom-row panels to avoid an empty fourth tile.
            x=8+int((c + (0.5 if r else 0))*(S+8))
            y=8+r*(TITLE+S+FOOT+8)
            canvas[y:y+p.shape[0],x:x+p.shape[1]]=p
        writer.write(canvas)
        prev=fr
        if t in (100,195,280): cv2.imwrite(str(OUT/f'pipeline_frame_{t}.png'),canvas)
        if t%100==0:print(f'frame {t}/365',flush=True)
    writer.release()
    print('done',outpath,'mean relative channel contrast',float(np.mean(selectivity)),flush=True)
if __name__=='__main__':main()
