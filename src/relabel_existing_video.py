"""Re-layout an already-computed 8-panel STMD demo into the requested 7 panels.

No neural computation or ground truth is changed. Uses saved model predictions
solely to erase the display's small white center dot inside the yellow ring.
For newly computed videos use render_process.py instead.
"""
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent.parent
VIDEO=ROOT/'media'/'original_8panel_pipeline_1m.mp4'
OUTPUT=ROOT/'media'/'stmd_actual_pipeline_1m.mp4'
CSV=ROOT/'results'/'predictive_1m_per_frame.csv'
SIZE=280; TITLE=30; FOOT=22; GAP=8
ROW_H=TITLE+SIZE+FOOT+GAP
LABELS=[
    ('1  RGB frame (target)','ground truth ring on input only'),
    ('2  simulated DVS','display only; NOT model input'),
    ('3  Small-target detector','computed from intensity frames'),
    ('4  24 motion maps','actual motion-channel responses'),
    ('5  Fused sensory response','green rings = candidate peaks'),
    ('6  Predictive spatial map','spatial activity only; no overlays'),
    ('7  Predicted target position','yellow circle = model prediction'),
]
# Cropping from existing 4 x 2 source: omitted input panel 5 (three banks).
SOURCE_INDEX=[0,1,2,3,5,6,7]

def main():
    cap=cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():raise RuntimeError(f'Missing source video {VIDEO}')
    fps=cap.get(cv2.CAP_PROP_FPS)
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert n==365 and abs(fps-25)<1e-3,(n,fps)
    preds=pd.read_csv(CSV)
    assert len(preds)==n
    width=GAP+4*(SIZE+GAP);height=GAP+2*ROW_H
    out=cv2.VideoWriter(str(OUTPUT),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    if not out.isOpened():raise RuntimeError('Cannot open output writer')
    for t in range(n):
        ok,frame=cap.read()
        if not ok:raise RuntimeError(f'Failed to decode frame {t}')
        canvas=np.full((height,width,3),235,np.uint8)
        for i,old_id in enumerate(SOURCE_INDEX):
            r_old,c_old=divmod(old_id,4)
            ox=GAP+c_old*(SIZE+GAP)
            oy=GAP+r_old*ROW_H
            img=frame[oy+TITLE:oy+TITLE+SIZE,ox:ox+SIZE].copy()
            if i==6:
                px,py=preds.loc[t,['pred_x','pred_y']]
                if np.isfinite(px) and np.isfinite(py):
                    # Remove only the previous center pixel/dot; keep yellow ring.
                    cx=int(round(float(px)*SIZE/320))
                    cy=int(round(float(py)*SIZE/320))
                    mask=np.zeros((SIZE,SIZE),np.uint8)
                    cv2.circle(mask,(cx,cy),3,255,-1,cv2.LINE_8)
                    img=cv2.inpaint(img,mask,4,cv2.INPAINT_TELEA)
            panel=np.full((TITLE+SIZE+FOOT,SIZE,3),255,np.uint8)
            panel[TITLE:TITLE+SIZE]=img
            title,foot=LABELS[i]
            cv2.putText(panel,title,(7,21),cv2.FONT_HERSHEY_SIMPLEX,.46,(25,25,25),1,cv2.LINE_AA)
            cv2.putText(panel,foot,(7,TITLE+SIZE+15),cv2.FONT_HERSHEY_SIMPLEX,.33,(65,65,65),1,cv2.LINE_AA)
            r,c=divmod(i,4)
            x=GAP+int((c+(0.5 if r else 0))*(SIZE+GAP))
            y=GAP+r*ROW_H
            canvas[y:y+panel.shape[0],x:x+panel.shape[1]]=panel
        out.write(canvas)
        if t in (100,195,280):cv2.imwrite(str(ROOT/'media'/f'pipeline_frame_{t}.png'),canvas)
    out.release();cap.release()
    print('saved',OUTPUT,OUTPUT.stat().st_size,'bytes',n,'frames')

if __name__=='__main__':main()
