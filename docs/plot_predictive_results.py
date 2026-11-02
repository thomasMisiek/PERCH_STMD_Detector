"""Generate result plots from the current controlled camera-speed experiment."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results';M=ROOT/'media'
df=pd.read_csv(R/'grid24_predictive_camera_control.csv')
models=['raw','old_annular','predictive_field']
conditions=['still','slow','normal']
distances=['50cm','1m','2m']
fig,axs=plt.subplots(1,3,figsize=(11,3.9),sharey=True)
for ax,dist in zip(axs,distances):
    x=np.arange(3)
    for j,mod in enumerate(models):
        vals=[df[(df.background==c)&(df.model==mod)&(df.distance==dist)].correct_pct.mean() for c in conditions]
        ax.bar(x+(j-1)*.25,vals,.25,label={'raw':'No added memory','old_annular':'Previous trace','predictive_field':'Predictive field'}[mod])
    ax.set_xticks(x,labels=['Still','0.5×','1×'])
    ax.set_title(dist)
    ax.set_ylim(0,105)
    ax.grid(axis='y',alpha=.2)
axs[0].set_ylabel('Correct localization within 5° (%)')
axs[0].legend(fontsize=8,loc='upper center')
fig.suptitle('Background motion control — shared calibration, two held-out trajectories',fontsize=11)
fig.tight_layout()
fig.savefig(M/'camera_speed_corrected.png',dpi=165)
plt.close(fig)

# one standalone chart for the baseline comparison on moving background
fig,ax=plt.subplots(figsize=(6.8,4))
sub=df[df.background=='normal']
x=np.arange(3)
for j,mod in enumerate(models):
    vals=[sub[(sub.model==mod)&(sub.distance==d)].correct_pct.mean() for d in distances]
    ax.bar(x+(j-1)*.26,vals,.26,label={'raw':'No added memory','old_annular':'Previous trace','predictive_field':'Predictive field'}[mod])
ax.set_xticks(x,labels=['50 cm','1 m','2 m'])
ax.set_ylim(0,105)
ax.set_ylabel('Correct localization within 5° (%)')
ax.set_title('Memory comparison — original moving background')
ax.legend(frameon=False,fontsize=9)
ax.grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig(M/'memory_comparison.png',dpi=165)
plt.close(fig)
