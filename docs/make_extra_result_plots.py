from pathlib import Path
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent
RES=ROOT/'results'
MED=ROOT/'media'
MED.mkdir(exist_ok=True)

mb=pd.read_csv(RES/'motion_bank_9_vs_24.csv')
cam=pd.read_csv(RES/'camera_speed_experiment.csv')

# plot 1 motion bank comparison
fig=plt.figure(figsize=(7,4.5))
ax=fig.add_subplot(111)
summary=mb.groupby(['distance','model'],as_index=False)['correct_pct'].mean()
order=['50cm','1m','2m']
models=['grid9','grid24']
x=np.arange(len(order))
width=0.35
for i,m in enumerate(models):
    vals=[summary[(summary.distance==d)&(summary.model==m)].correct_pct.mean() for d in order]
    ax.bar(x+(i-0.5)*width, vals, width, label=m)
ax.set_xticks(x); ax.set_xticklabels(order)
ax.set_ylabel('Held-out correct localization (%)')
ax.set_title('Velocity-bank comparison')
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(MED/'motion_bank_comparison.png',dpi=200); plt.close(fig)

# plot 2 background speed
fig=plt.figure(figsize=(8,4.8))
ax=fig.add_subplot(111)
summary=cam.groupby(['condition','distance'],as_index=False)['correct_pct'].mean()
conds=['still','slow_bg','normal_bg']
labels={'still':'still','slow_bg':'slow bg','normal_bg':'normal bg'}
for d in order:
    vals=[summary[(summary.distance==d)&(summary.condition==c)].correct_pct.mean() for c in conds]
    ax.plot(range(len(conds)),vals,marker='o',label=d)
ax.set_xticks(range(len(conds))); ax.set_xticklabels([labels[c] for c in conds])
ax.set_ylabel('Held-out correct localization (%)')
ax.set_title('Effect of background motion speed (grid24)')
ax.legend(frameon=False,title='Target distance')
fig.tight_layout(); fig.savefig(MED/'camera_speed_comparison.png',dpi=200); plt.close(fig)
