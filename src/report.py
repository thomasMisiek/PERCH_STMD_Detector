"""Summarize held-out angular accuracy, calibrated reject decisions, random chance.
No parameter is selected on held-out seeds.
"""
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from spatial_trace import infer
from evaluate import get_data,angular_error_deg,TRAIN,TEST,SEEDS,LABELS,RESULTS

ROOT=RESULTS.parent
cfg=json.loads((RESULTS/'selected_configs.json').read_text())
z=np.load(RESULTS/'evidence_none.npz')
threshold={}
null_rows=[]
for mode,par in cfg.items():
    xy,s,_=infer(z['candidate'],z['count'],par)
    threshold[mode]=float(np.nanquantile(s[25:100],0.99))
    null_rows.append({'model':mode,'threshold_99pct_calibration':threshold[mode],
                      'false_positive_fraction_eval':float((s[100:]>=threshold[mode]).mean()),
                      'null_eval_frames':len(s[100:])})
# Same no-target source; threshold calibrated on early 25..99; evaluate on later 100..364.
null_df=pd.DataFrame(null_rows)
null_df.to_csv(RESULTS/'no_target_5deg_gating.csv',index=False)
rows=[]
for seed in SEEDS:
    for label in LABELS:
        c,n,gt,multi=get_data(seed,label)
        for name,par in cfg.items():
            xy,score,_=infer(c,n,par)
            err=angular_error_deg(xy,gt)
            correct=err<=5
            accepted=score>=threshold[name]
            d=pd.DataFrame(dict(frame=np.arange(len(gt)),gt_x=gt[:,0],gt_y=gt[:,1],x=xy[:,0],y=xy[:,1],score=score,error_deg=err,correct_5deg=correct,accepted=accepted,accepted_correct_5deg=correct&accepted))
            d.to_csv(RESULTS/f'annotated_{seed}_{label}_{name}.csv',index=False)
            rows.append(dict(seed=seed,split='train' if seed in TRAIN else 'heldout',distance=label,model=name,frames=len(gt[100:]),correct=int(correct[100:].sum()),correct_pct=float(correct[100:].mean()*100),gated_correct_pct=float((correct&accepted)[100:].mean()*100),accepted_pct=float(accepted[100:].mean()*100)))
        xy=multi[:,1:3];err=angular_error_deg(xy,gt);correct=err<=5
        accepted=multi[:,0]>.5
        rows.append(dict(seed=seed,split='train' if seed in TRAIN else 'heldout',distance=label,model='original_multi',frames=len(gt[100:]),correct=int(correct[100:].sum()),correct_pct=float(correct[100:].mean()*100),gated_correct_pct=float((correct&accepted)[100:].mean()*100),accepted_pct=float(accepted[100:].mean()*100)))

d=pd.DataFrame(rows);d.to_csv(RESULTS/'per_trial_and_gated_5deg.csv',index=False)
summary=d.groupby(['split','distance','model'],as_index=False).agg(correct_frames=('correct','sum'),total_frames=('frames','sum'),correct_pct=('correct_pct','mean'),gated_correct_pct=('gated_correct_pct','mean'),accepted_pct=('accepted_pct','mean'))
summary.to_csv(RESULTS/'summary_5deg_detailed.csv',index=False)
(RESULTS/'protocol_updated.json').write_text(json.dumps({'training_seeds':list(TRAIN),'heldout_seeds':list(TEST),'full_video_frames':365,'fps':25,'eval_frames':'100–364','assumed_horizontal_fov_deg':84,'angular_error':'Angle between normalized (x-cx,y-cy,f) camera rays, f=160/tan(42°), Euclidean 3D angle <= 5°','forced_choice':'one x,y candidate every frame where sensory peaks exist','no_target_gate_threshold':'99th percentile of no-target maximum evidence, calibration frames 25–99','thresholds':threshold,'selected':cfg,'random_baseline':'uniform x/y sampled in 12..308 rectangle, 300 simulations per trajectory; always outputs a position','target_diameter_px':{'50cm':2.62,'1m':1.31,'2m':0.65},'note':'Simulated 6mm mosquito injected into 320x320 images after area downsampling; this is not a physical event-camera measurement.'},indent=2))

# Plot training and heldout, three models + random
fig,axs=plt.subplots(1,2,figsize=(12.5,4.6),sharey=True)
order=['50cm','1m','2m']; names=['raw','leaky','annulus','existing_multi','random_uniform']
labels={'raw':'No temporal map','leaky':'Plain leaky trace','annulus':'Neighbor-motion trace','existing_multi':'Original multi-hypothesis','random_uniform':'Random position'}
raw_summary=pd.read_csv(RESULTS/'summary_5deg.csv')
for ax,split in zip(axs,['train','heldout']):
    x=np.arange(3);width=.16
    for k,name in enumerate(names):
        sub=raw_summary[(raw_summary['split']==split)&(raw_summary['model']==name)]
        values=[float(sub[sub['distance']==dist]['percent'].iloc[0]) for dist in order]
        ax.bar(x+(k-2)*width,values,width,label=labels[name])
    ax.set_xticks(x,order);ax.set_ylim(0,105)
    ax.set_title(('Training' if split=='train' else 'Held-out')+' trajectories (2 seeds)')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axs[0].set_ylabel('Correct localization within 5° (%)')
fig.legend(*axs[1].get_legend_handles_labels(),loc='lower center',ncol=3,bbox_to_anchor=(.5,-.08))
fig.suptitle('STMD spatial temporal reinforcement vs unreinforced sensory peaks')
fig.tight_layout(rect=[0,.10,1,.95]);fig.savefig(ROOT/'media'/'heldout_accuracy.png',dpi=170,bbox_inches='tight');plt.close(fig)
print('NO TARGET',null_df.to_string(index=False),flush=True)
print('HELDOUT',summary[summary['split']=='heldout'].to_string(index=False),flush=True)
