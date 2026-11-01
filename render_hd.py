"""Recompute tuned-STMD detections and make HD RGB / 320px DVS-style media.

No ground-truth target position is passed into the detector.
The DVS-style view is a frame-difference VISUALIZATION; the detector sees RGB->gray frames.

HD left: full-width original 1920x1080 footage rescaled to 1280x720,
with the synthetic disk drawn at the registered target location.
Right: 320x320 event-style response from the tuned 320x320 detector, scaled to
720x720 by nearest-neighbor interpolation solely for display.

Note: "20s" means 14.6 seconds of footage plus 5.4s with final frames
played in reverse, not slowed down. The README GIF uses only forward footage.
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import benchmark as b
from stmd_model import stamp, apparent_diameter_px

ROOT=Path(__file__).resolve().parent
MEDIA=ROOT/'media'
VIDEO=ROOT/'data'/'source_indoor_walking.mp4'
SENSOR=320
FPS=25
LEFT_W=1280
LEFT_H=720
RIGHT=720


def ffmpeg_writer(path:Path, width:int, height:int, crf=29):
    exe=shutil.which('ffmpeg')
    if not exe:raise RuntimeError('ffmpeg command not found on PATH')
    proc=subprocess.Popen([exe,'-y','-loglevel','error',
           '-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}',
           '-r',str(FPS),'-i','-','-an','-c:v','libx264',
           '-preset','veryfast','-crf',str(crf),'-pix_fmt','yuv420p',
           '-movflags','+faststart',str(path)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    return proc


def close(proc):
    proc.stdin.close(); err=proc.stderr.read().decode(errors='replace'); proc.wait()
    if proc.returncode:raise RuntimeError(f'ffmpeg failed: {err[-2000:]}')


def load_full_source():
    c=cv2.VideoCapture(str(VIDEO));out=[]
    while True:
        ok,f=c.read()
        if not ok:break
        out.append(cv2.resize(f,(LEFT_W,LEFT_H),interpolation=cv2.INTER_AREA))
    c.release()
    return out


def high_res_overlay(base_hd, pos, diameter_sensor):
    """Separate visual depiction registered to detector input; not detector input."""
    rgb=base_hd.copy()
    # Source 1920x1080 -> width 569 x height 320 -> central 320 crop.
    resized_w=round(1920*SENSOR/1080)
    left_crop=(resized_w-SENSOR)//2
    scale=LEFT_H/SENSOR
    x=round((pos[0]+left_crop)*scale)
    y=round(pos[1]*scale)
    stamp(rgb,(x,y),diameter_sensor*scale,blur=.42*scale)
    return rgb


def add_labels(frame):
    # DVS event display is illustrative, not actual sensor data
    cv2.rectangle(frame,(0,0),(LEFT_W,32),(17,26,36),-1)
    cv2.rectangle(frame,(LEFT_W+342,0),(LEFT_W+RIGHT,32),(17,26,36),-1)
    cv2.putText(frame,'HD RGB + simulated 6 mm target at 1 m',(12,23),cv2.FONT_HERSHEY_SIMPLEX,.62,(247,250,252),1,cv2.LINE_AA)
    cv2.putText(frame,'Simulated DVS  /  320 x 320',(LEFT_W+349,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(247,250,252),1,cv2.LINE_AA)
    return frame


def generate(start=100, clip_frames=100, quick=False):
    MEDIA.mkdir(exist_ok=True)
    frames_original, source_fps=b.load_input()
    if abs(source_fps-FPS)>.1:raise RuntimeError('Expected original 25 fps footage')
    frames320=b.make_20s(frames_original)
    thresholds=b.calibrate(frames320)
    # The original tuned experiment injects the dot *after* spatial resampling.
    withdot,gt,diameter=b.apply_target(frames320,1.0)
    df,dvs=b.evaluate_frames(withdot,thresholds,gt=gt,make_vis=True)
    print('Tuned 1m, 100:250 correct fraction:',df.correct.iloc[100:250].mean(),flush=True)
    df.to_csv(MEDIA/'1m_hd_video_detections.csv',index=False)
    full=load_full_source()
    # Original 365 frames then final 135 reversed (same indexing as benchmark).
    indices=list(range(len(full))) + list(range(len(full)-2,len(full)-2-(500-len(full)),-1))
    if len(indices)!=len(dvs):raise AssertionError('index mismatch')
    full_out=MEDIA/'stmd_tuned_1m_HD_20s.mp4'
    short_out=MEDIA/'stmd_tuned_1m_HD_short.mp4'
    gif_out=MEDIA/'stmd_tuned_1m_demo.gif'
    w=LEFT_W+RIGHT;h=LEFT_H
    full_writer=None if quick else ffmpeg_writer(full_out,w,h,crf=30)
    short_writer=ffmpeg_writer(short_out,w,h,crf=26)
    preview=MEDIA/'stmd_tuned_1m_HD_still.jpg'
    max_frames=len(dvs)
    for i in range(max_frames):
        if full_writer is None and not(start <= i < start+clip_frames):continue
        im=high_res_overlay(full[indices[i]],gt[i],diameter)
        right=cv2.resize(dvs[i],(RIGHT,RIGHT),interpolation=cv2.INTER_NEAREST)
        panel=add_labels(np.hstack([im,right]))
        if full_writer is not None: full_writer.stdin.write(panel.tobytes())
        if start<=i<start+clip_frames:
            short_writer.stdin.write(panel.tobytes())
        if i==start+clip_frames//2:
            cv2.imwrite(str(preview),panel,[cv2.IMWRITE_JPEG_QUALITY,93])
    if full_writer is not None: close(full_writer)
    close(short_writer)
    # GIF: short GIF preview that GitHub README embeds, no external links required.
    # ffmpeg palette workflow, use 8 fps and 960px output width for small size.
    palette=MEDIA/'_palette.png'
    filter1='fps=8,scale=960:-1:flags=lanczos,palettegen=max_colors=96:stats_mode=diff'
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(short_out),'-vf',filter1,str(palette)],check=True)
    filter2='fps=8,scale=960:-1:flags=lanczos [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle'
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(short_out),'-i',str(palette),'-lavfi',filter2,'-loop','0',str(gif_out)],check=True)
    palette.unlink(missing_ok=True)
    print('done',[(p.name,round(p.stat().st_size/1024**2,2)) for p in [short_out,gif_out] if p.exists()],flush=True)
    if full_writer is not None:print('full',full_out.stat().st_size/1024**2,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--start',type=int,default=100)
    p.add_argument('--clip-frames',type=int,default=100)
    p.add_argument('--short-only',action='store_true',help='Skip the full 20s MP4')
    args=p.parse_args()
    generate(start=args.start,clip_frames=args.clip_frames,quick=args.short_only)
