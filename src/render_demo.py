"""14.6 s HD comparison video, showing raw peak and unified memory positions.
Left: original HD video plus synthetic target. Right: frame-difference DVS-*style*
visualization from 320x320 RGB frames, NOT real event-camera data.
No GT markers or GT trajectory are overlaid on the demo.
"""
import subprocess
from pathlib import Path
import cv2,numpy as np,pandas as pd
from experiment import read_source,inject
from stmd_model import stamp,apparent_diameter_px
from trajectories import generate_trajectory

ROOT=Path(__file__).resolve().parent.parent
MEDIA=ROOT/'media';RESULTS=ROOT/'results'
VIDEO=Path(__file__).resolve().parent/'data'/'source_indoor_walking.mp4'
FPS=25

def dvs_visual(gray,previous):
    out=np.full((320,320,3),15,np.uint8)
    if previous is not None:
        delta=np.log1p(gray.astype(np.float32))-np.log1p(previous.astype(np.float32))
        on=np.sqrt(np.clip((delta-.045)/.25,0,1));off=np.sqrt(np.clip((-delta-.045)/.25,0,1))
        out[...,1]=np.minimum(255,15+240*on).astype(np.uint8)
        o=np.minimum(255,15+240*off).astype(np.uint8)
        out[...,0]=o;out[...,2]=o
    return out

def main(seed=173,label='1m'):
    MEDIA.mkdir(exist_ok=True)
    base=read_source();path=generate_trajectory(seed,len(base))
    dist={'50cm':.5,'1m':1.,'2m':2.}[label]
    plus=inject(base,path,dist)
    raw=pd.read_csv(RESULTS/f'annotated_{seed}_{label}_raw.csv')
    new=pd.read_csv(RESULTS/f'annotated_{seed}_{label}_annulus.csv')
    assert len(base)==len(raw)==len(new)==365
    cmd=['ffmpeg','-y','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','2000x720','-r','25','-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','30','-pix_fmt','yuv420p','-movflags','+faststart',str(MEDIA/f'{label}_full_HD_comparison.mp4')]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    cap=cv2.VideoCapture(str(VIDEO)); last=None
    for i in range(len(base)):
        ok,frame=cap.read()
        if not ok:raise RuntimeError('Missing HD source frame')
        hd=cv2.resize(frame,(1280,720),interpolation=cv2.INTER_AREA)
        # This video reproduces the synthetic target's appearance. Detector
        # predictions are taken from a separate 320x320 sensory simulation.
        x,y=path[i]
        HD_SCALE=720/320
        center_crop_offset=(round(frame.shape[1]*320/frame.shape[0])-320)//2
        stamp(hd,(int(round((x+center_crop_offset)*HD_SCALE)),int(round(y*HD_SCALE))),apparent_diameter_px(6,dist)*HD_SCALE,blur=.42*HD_SCALE)
        gray=cv2.cvtColor(plus[i],cv2.COLOR_BGR2GRAY)
        dvs=dvs_visual(gray,last);last=gray
        right=cv2.resize(dvs,(720,720),interpolation=cv2.INTER_NEAREST)
        # Raw baseline / temporally reinforced output; NEITHER is ground truth.
        for row,color,tag in ((raw.iloc[i],(255,200,70),'RAW'),(new.iloc[i],(0,220,255),'TRACE')):
            if not np.isfinite(float(row.x)) or not np.isfinite(float(row.y)):
                continue
            px,py=int(round(float(row.x)*HD_SCALE)),int(round(float(row.y)*HD_SCALE))
            cv2.circle(right,(px,py),12,color,2,cv2.LINE_AA)
            cv2.circle(right,(px,py),2,color,-1,cv2.LINE_AA)
        cv2.rectangle(right,(0,0),(720,76),(19,28,44),-1)
        cv2.putText(right,'DVS-like simulation  (320 x 320)',(18,25),cv2.FONT_HERSHEY_SIMPLEX,.58,(245,245,245),1,cv2.LINE_AA)
        cv2.putText(right,'RAW: blue    TRACE: yellow',(18,55),cv2.FONT_HERSHEY_SIMPLEX,.58,(245,245,245),1,cv2.LINE_AA)
        panel=np.hstack([hd,right])
        cv2.rectangle(panel,(0,0),(1280,39),(19,28,44),-1)
        cv2.putText(panel,f'HD original + simulated 6 mm insect, distance {label}',(18,27),cv2.FONT_HERSHEY_SIMPLEX,.69,(245,245,245),1,cv2.LINE_AA)
        proc.stdin.write(panel.tobytes())
        if i==155:cv2.imwrite(str(MEDIA/f'{label}_comparison_still.jpg'),panel,[cv2.IMWRITE_JPEG_QUALITY,88])
    cap.release();proc.stdin.close();rc=proc.wait()
    if rc:raise RuntimeError('ffmpeg encoding failed')
    mp4=MEDIA/f'{label}_full_HD_comparison.mp4';gif=MEDIA/f'{label}_full_length.gif'
    pal=MEDIA/'palette_tmp.png'
    subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(mp4),'-vf','fps=7,scale=800:-1:flags=lanczos,palettegen=max_colors=42',str(pal)],check=True)
    subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(mp4),'-i',str(pal),'-lavfi','fps=7,scale=800:-1:flags=lanczos [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5','-loop','0',str(gif)],check=True)
    pal.unlink(missing_ok=True)
    print('mp4, gif',mp4.stat().st_size, gif.stat().st_size)

if __name__=='__main__':main()
