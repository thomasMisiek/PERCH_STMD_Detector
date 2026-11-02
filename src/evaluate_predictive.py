"""Test predictive-field settings on saved sensory candidates, separated by path.

Important: the cached evidence comes from the previous 14-channel bank.
The selected 24-channel model must be evaluated separately and these scores
must not be misrepresented as the grid24 result.
"""
from pathlib import Path
import json
import numpy as np,pandas as pd
from predictive_field import infer as predictive
from spatial_trace import infer as annulus
from evaluate import angular_error_deg, get_data
ROOT=Path(__file__).resolve().parent.parent
R=ROOT/'results'; TRAIN=(41,92); TEST=(173,281)
DIST=('50cm','1m','2m')
ANNULUS=dict(mode='annulus',decay=.92,neighborhood=11.,inhibition=1.3,gain=70.,n_peaks=25)
PRESETS={
  'conservative':dict(decay=.78,gain=.8,sigma=4,drift=.5,n_peaks=8,winner_boost=.4),
  'balanced':dict(decay=.86,gain=1.4,sigma=5,drift=.6,n_peaks=12,winner_boost=.45),
  'strong':dict(decay=.92,gain=3.0,sigma=7,drift=1.,n_peaks=16,winner_boost=.45),
  'wide':dict(decay=.84,gain=2.,sigma=10,drift=1.5,n_peaks=16,winner_boost=.45),
  'low':dict(decay=.85,gain=.3,sigma=5,drift=.8,n_peaks=10,winner_boost=.45),
}

def evaluate():
    rows=[]
    for seed in TRAIN+TEST:
      for dist in DIST:
        c,n,gt,_=get_data(seed,dist)
        # Raw peak, old annular trace, new predictive configs (identical candidates)
        options={}
        xy=c[:,0,:2].copy(); xy[n==0]=np.nan;options['raw']=xy
        axy,_,_=annulus(c,n,ANNULUS);options['previous_annulus']=axy
        for name,cfg in PRESETS.items():options[name]=predictive(c,n,cfg)[0]
        for name,xy in options.items():
          err=angular_error_deg(xy,gt)
          good=np.isfinite(err)&(err<=5)
          rows.append(dict(seed=seed,split='train' if seed in TRAIN else 'heldout',distance=dist,
              model=name,n=int(len(good[100:])),correct=int(good[100:].sum()),
              correct_pct=float(good[100:].mean()*100)))
    df=pd.DataFrame(rows);df.to_csv(R/'predictive_field_comparison_14bank.csv',index=False)
    summary=df.groupby(['split','model','distance'])['correct_pct'].mean().round(2)
    print(summary.to_string())
    train=df[(df.split=='train')&df.model.isin(PRESETS)].groupby('model').correct_pct.mean().sort_values(ascending=False)
    selected=train.index[0]
    (ROOT/'configs'/'predictive_selected.json').write_text(json.dumps(dict(name=selected,params=PRESETS[selected],source='train paths 41,92 on cached 14-bank candidates'),indent=2))
    print('selected from training:',selected, PRESETS[selected]);return selected
if __name__=='__main__':evaluate()
