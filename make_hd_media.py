"""Rebuild GitHub media with a genuinely high-resolution RGB panel.

Left: 1280x720 RGB frames sampled directly from the original 1920x1080 video,
      with a subpixel synthetic dot overlaid at positions registered to the
      corresponding 320x320 detector input.
Right: unchanged 320x320 DVS-like model output, resized ONLY for presentation.

This makes no claim that the DVS simulation or detector has 720p resolution.
Requires the legacy 640x640 four-panel model video in media/.
"""
from pathlib import Path
import argparse, math, subprocess
import cv2
import numpy as np
from stmd_model import stamp, apparent_diameter_px
from multi_hypothesis import fast_trajectory

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'data'/'source_indoor_walking.mp4'
DEMO=ROOT/'media'/'stmd_1m_four_panel.mp4'
MEDIA=ROOT/'media'
SENSOR=320
LEFT_W,LEFT_H=1280,720
RIGHT=720
FPS=25

def original_indices(n, source_count):
    idx=list(range(source_count))
    if n>source_count:
        backwards=list(range(source_count-2,-1,-1))
        while len(idx)<n:idx.extend(backwards)
    return idx[:n]

def left_with_dot(fr, xy, resize_w=LEFT_W, resize_h=LEFT_H):
    # Independent use of full-res source, NOT enlargement of the 320-pixel crop.
    fr=cv2.resize(fr,(resize_w,resize_h),interpolation=cv2.INTER_AREA)
    x,y=map(int,xy)
    # In the model the original image is first scaled to 320 pixels in height,
    # then center-cropped horizontally to 320x320. Map sensor image coords back.
    resized_full_width=round(1920*320/1080)
    crop_left=(resized_full_width-SENSOR)//2
    scale=resize_h/SENSOR
    px=int(round((x+crop_left)*scale)); py=int(round(y*scale))
    # Draw a 6mm object at 1m (1.31px across in SENSOR pixels).
    # Size on the RGB panel follows its own image scale (2.95px diameter at 720p).
    stamp(fr,(px,py),apparent_diameter_px(6,1.0)*scale, blur=0.36)
    return fr

def make_frame(rgb, model_bgr):
    return np.hstack((rgb, cv2.resize(model_bgr,(RIGHT,RIGHT),interpolation=cv2.INTER_NEAREST)))

def title_bar(rgb, dvs, left='RGB video + simulated 6mm target at 1m', right='Simulated DVS view (320 x 320), STMD detections'):
    # Labels don't obscure the target in the middle of the clip.
    if rgb is not None:
        cv2.rectangle(rgb,(0,0),(rgb.shape[1],34),(22,22,22),-1)
        cv2.putText(rgb,left,(12,24),cv2.FONT_HERSHEY_SIMPLEX,0.65,(245,245,245),1,cv2.LINE_AA)
    if dvs is not None:
        cv2.rectangle(dvs,(0,0),(dvs.shape[1],34),(22,22,22),-1)
        cv2.putText(dvs,right,(12,24),cv2.FONT_HERSHEY_SIMPLEX,0.55,(245,245,245),1,cv2.LINE_AA)

def run(start=130, frames=100, gif_fps=8):
    MEDIA.mkdir(exist_ok=True)
    source=cv2.VideoCapture(str(SOURCE)); old=cv2.VideoCapture(str(DEMO))
    nsrc=int(source.get(cv2.CAP_PROP_FRAME_COUNT)); nold=int(old.get(cv2.CAP_PROP_FRAME_COUNT))
    fps=float(source.get(cv2.CAP_PROP_FPS)) or 25.0
    if fps!=25.0:raise RuntimeError('Expected 25 fps source for original aligned demo')
    trajectory=fast_trajectory(nold)
    idx=original_indices(nold,nsrc)
    if old.get(cv2.CAP_PROP_FRAME_WIDTH)!=640:raise RuntimeError('Expected previous 640x640 four-panel video')
    # Source footage loaded in full resolution once. Avoid decoding 1080p per output in reverse.
    source_frames=[]
    while True:
        ok,img=source.read()
        if not ok:break
        source_frames.append(img)
    source.release()
    fullout=MEDIA/'stmd_1m_four_panel_HD.mp4'
    shortout=MEDIA/'stmd_1m_short_HD.mp4'
    gifout=MEDIA/'stmd_1m_demo_HD.gif'
    previewout=MEDIA/'stmd_1m_frame_HD.jpg'
    def ffmpeg_out(path,size,crf=27):
        w,h=size
        cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-s',f'{w}x{h}','-pix_fmt','bgr24','-r',str(fps),'-i','-', '-an','-c:v','libx264','-preset','veryfast','-crf',str(crf),'-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
        return subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    # Four-panel: top-left and bottom-left true HD from original video;
    # DVS panels still 320x320, enlarged to 720 solely for viewing.
    proc=ffmpeg_out(fullout,(LEFT_W+RIGHT,2*RIGHT),crf=27)
    # short at original speed, no reverse in selected excerpt
    short=ffmpeg_out(shortout,(LEFT_W+RIGHT,RIGHT),crf=24)
    # GIF source at 12fps, display left 768px and right 432px; original source input.
    gw,gh=768,432; dr=432; gif_pipe=None
    gif_raw=MEDIA/'_gif_source.mp4'
    gif_pipe=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-s',f'{gw+dr}x{gh}','-pix_fmt','bgr24','-r',str(gif_fps),'-i','-', '-an', '-c:v','libx264','-crf','27','-pix_fmt','yuv420p',str(gif_raw)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    next_samples=set(np.round(np.linspace(start,start+frames-1,num=round(frames/fps*gif_fps))).astype(int).tolist())
    for i in range(nold):
        ok,frame=old.read()
        if not ok: break
        rgb=left_with_dot(source_frames[idx[i]],trajectory[i])
        rgb_clean=cv2.resize(source_frames[idx[i]],(LEFT_W,LEFT_H),interpolation=cv2.INTER_AREA)
        dvst=frame[0:320,320:640]
        dvsb=frame[320:640,320:640]
        # no extra title in current top right because original visual labels are embedded.
        top=make_frame(rgb,dvst)
        bottom=make_frame(rgb_clean,dvsb)
        stacked=np.vstack((top,bottom))
        proc.stdin.write(stacked.tobytes())
        if start<=i<start+frames:
            short.stdin.write(top.tobytes())
        if i in next_samples:
            tiny=left_with_dot(source_frames[idx[i]],trajectory[i],gw,gh)
            sensor_scaled=cv2.resize(dvst,(dr,dr),interpolation=cv2.INTER_NEAREST)
            gif_pipe.stdin.write(np.hstack((tiny,sensor_scaled)).tobytes())
        if i==start+frames//2:
            cv2.imwrite(str(previewout),top,[cv2.IMWRITE_JPEG_QUALITY,91])
    old.release()
    for p in (proc,short,gif_pipe):
        p.stdin.close()
        try: _,err=p.communicate(timeout=120)
        except ValueError:
            p.wait(timeout=120);err=b''
        if p.returncode: raise RuntimeError(f'Video encode failed: {str(err)[-2000:]}')
    # Use palettegen/use to keep gif small while avoiding huge files on GitHub.
    palette=MEDIA/'_palette.png'
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(gif_raw),'-vf','fps=8,scale=1000:-1:flags=lanczos,palettegen=stats_mode=diff',str(palette)],check=True)
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(gif_raw),'-i',str(palette),'-lavfi','fps=8,scale=1000:-1:flags=lanczos [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle','-loop','0',str(gifout)],check=True)
    palette.unlink(missing_ok=True)
    gif_raw.unlink(missing_ok=True)
    print('Wrote',fullout,fullout.stat().st_size)
    print('Wrote',shortout,shortout.stat().st_size)
    print('Wrote',gifout,gifout.stat().st_size)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--start',type=int,default=130)
    p.add_argument('--frames',type=int,default=100)
    a=p.parse_args()
    run(a.start,a.frames)
