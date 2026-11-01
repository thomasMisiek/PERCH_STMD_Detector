"""Generate side-by-side four-panel target/no-target house videos and metrics.
Original source plays at 25 fps, forward then reverse for 20 seconds.
"""
import gc,time,subprocess
from pathlib import Path
import cv2,numpy as np,pandas as pd,matplotlib.pyplot as plt
import multi_hypothesis as m
ROOT=Path(__file__).resolve().parent/'outputs'
ROOT.mkdir(parents=True, exist_ok=True)
FPS=25


def title(frame,text):
    out=frame.copy()
    cv2.rectangle(out,(0,0),(m.SIDE,29),(0,0,0),-1)
    cv2.putText(out,text,(7,20),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
    return out


def render(plus,plusviz,base,baseviz,outpath,label):
    args=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
          '-s','640x640','-pix_fmt','bgr24','-r',str(FPS),'-i','-', '-an',
          '-c:v','libx264','-preset','ultrafast','-crf','27','-pix_fmt','yuv420p',
          '-movflags','+faststart',str(outpath)]
    proc=subprocess.Popen(args,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    for a,b,c,d in zip(plus,plusviz,base,baseviz):
        top=np.hstack([title(a,f'{label} - small black target'),title(b,'DVS-like - competing hypotheses')])
        bot=np.hstack([title(c,'NO TARGET - original'),title(d,'NO TARGET - DVS-like')])
        proc.stdin.write(np.vstack([top,bot]).tobytes())
    proc.stdin.close();proc.wait()
    if proc.returncode:
        raise RuntimeError('Video encoding failed: '+proc.stderr.read().decode(errors='replace')[-2000:])


def texture_at_target(base,xy):
    # Analysis only, based on the unaltered scene, never fed into detection.
    tex=[]
    for fr,(x,y) in zip(base,xy):
        g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY).astype(np.float32)
        x=int(x);y=int(y)
        y0=max(0,y-7);y1=min(m.SIDE,y+8);x0=max(0,x-7);x1=min(m.SIDE,x+8)
        p=g[y0:y1,x0:x1]
        tex.append(float(np.std(p)))
    return np.asarray(tex)


def calculate_metrics(df,gt=None,texture=None):
    a={}
    for name,sl in [('all',slice(None)),('eval',slice(100,365)),('second_half',slice(250,None))]:
        r=df.iloc[sl]; a[name+'_detection_fraction']=float(r.detected.mean());a[name+'_mean_confidence']=float(r.confidence.mean())
        if gt is not None:a[name+'_correct_fraction']=float(r.correct.mean())
    if texture is not None:
        # Within-scene comparison based on median of surround scene standard deviation
        for label,ind in [('smoother',texture<=np.median(texture)),('textured',texture>np.median(texture))]:
            a[label+'_correct_fraction']=float(df.correct[ind].mean())
    return a


def main():
    b,fps,src_n=m.load_frames(500)
    print(f'{len(b)} frames, {fps} fps, {src_n} source frames',flush=True)
    scales=m.calibrate(b)
    print('neural scales',scales,flush=True)
    results=[]
    start=time.time()
    no,nov=m.run(b,scales,with_visuals=True)
    no.to_csv(ROOT/'no_target_results.csv',index=False)
    results.append(dict(condition='no target',diameter_px=0,**calculate_metrics(no)))
    print('No-target finished',round(time.time()-start,1),'s',flush=True)
    for distance,label in m.DISTANCES:
        start=time.time()
        withdot,gt,diam=m.inject(b,distance)
        dd,vv=m.run(withdot,scales,gt,with_visuals=True)
        tex=texture_at_target(b,gt)
        dd['scene_texture_std']=tex
        dd.to_csv(ROOT/f'{label}_results.csv',index=False)
        results.append(dict(condition=label,diameter_px=diam,**calculate_metrics(dd,gt,tex)))
        render(withdot,vv,b,nov,ROOT/f'STMD_MULTI_HYPOTHESES_{label}.mp4',label)
        k=245
        preview=np.vstack([np.hstack([title(withdot[k],f'{label} - small black target'),title(vv[k],'DVS-like - competing hypotheses')]),np.hstack([title(b[k],'NO TARGET - original'),title(nov[k],'NO TARGET - DVS-like')])])
        cv2.imwrite(str(ROOT/f'preview_{label}.png'),preview)
        print(label,'eval correct',round(dd.correct.iloc[100:365].mean(),3),'all',round(dd.correct.mean(),3), 'textured',round(dd.correct[tex>np.median(tex)].mean(),3),'s',round(time.time()-start,1),flush=True)
        del withdot,gt,dd,vv;gc.collect()
    summary=pd.DataFrame(results)
    summary.to_csv(ROOT/'results_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,4.5))
    sub=summary[summary.condition!='no target']
    ax.plot(sub.condition,sub.eval_correct_fraction*100,marker='o',label='Fast, clutter-crossing path')
    ax.axhline(summary.iloc[0].eval_detection_fraction*100,color='gray',ls='--',label='No-target false alarm rate')
    ax.set_ylim(0,100);ax.set_ylabel('Percent of evaluated frames');ax.set_xlabel('Apparent mosquito distance');ax.legend();fig.tight_layout()
    fig.savefig(ROOT/'accuracy.png',dpi=165);plt.close(fig)
    (ROOT/'README.md').write_text(f'''# Competing predictive STMD demo

Three side-by-side 4-panel videos at 50 cm, 1 m, and 2 m, using the same fast trajectory through the entire 320x320 house scene.

Input: {src_n} original frames at {fps} fps = {src_n/fps:.1f} seconds; extended to 20 seconds at **normal playback speed** by reversing the final source section. There is no slow motion.

Target size: assumed mosquito body length 6 mm, field of view 84 degrees, sensor 320x320, projected diameters ~2.62, 1.31, 0.65 pixels.

## Model
- Three full-field STMD-like neural evidence channels: broad, dense-motion-inhibited, and texture-tolerant.
- Additional **dense optical flow compensation** using Farneback flow on 160x160 frames, upsampled to 320x320, followed by centre/surround suppression of unexplained dark residuals.
- Up to nine competing hypotheses, each with its own position, velocity, decaying support, directional consistency, and persistence.
- Hypotheses born anywhere in the visual field, **no target-specific prior at start**. Later detections are supported by predicted positions and repeated neural evidence.
- A logistic **heuristic confidence/evidence signal**, not a calibrated posterior probability, from neural strength, persistence, direction consistency, track age and motion; confidence falls after missed updates.
- No ground-truth positions used in inference. Dot-free video is used only to scale each neural response channel and evaluate false alarms.
- Right panels are simulated **DVS-style displays derived from frame luminance**, not actual DVS event recordings. Detector uses grayscale video.

Yellow marker = committed detection and estimated movement direction. Smaller blue markers = other competing candidates. No green ground-truth circles are drawn.

The CSV files provide frame-level confidence, position, track count, localization error, ground truth, and background texture statistics.
''')
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':main()
