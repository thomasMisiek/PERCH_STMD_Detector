"""Render the *actual* tuned-STMD processing pipeline (no hypotheses bank)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Ellipse
from matplotlib.colors import TwoSlopeNorm

OUT = Path(__file__).resolve().parent.parent / 'media'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(20,5.2))
fig.patch.set_facecolor('white')
ax.set_xlim(0,20)
ax.set_ylim(0,5.2)
ax.axis('off')
NAVY='#10243a'; DARK='#28384a'; MUTED='#64748b'; TEAL='#148c92'; GOLD='#e4a840'; CREAM='#f6f9fa'; BORDER='#d5e1e7'

def card(x,title,desc,accent=TEAL):
    w=2.82;h=3.92;y=.50
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.13,rounding_size=0.16',facecolor='#ffffff',edgecolor=BORDER,linewidth=1.25,zorder=1))
    ax.add_patch(Rectangle((x,y+h-.07),w,.07,facecolor=accent,edgecolor='none',zorder=2))
    ax.text(x+.18,y+h-.37,title,ha='left',va='center',fontsize=12.6,fontweight='bold',color=NAVY,zorder=3)
    ax.text(x+.18,y+.42,desc,ha='left',va='center',fontsize=9.0,color=MUTED,zorder=3,linespacing=1.5)
    return x,w,y,h

xs=[.16,3.55,6.94,10.33,13.72,17.11]
for i in range(5):
    x=xs[i]+2.94
    ax.add_patch(FancyArrowPatch((x,2.49),(xs[i+1]-.13,2.49),arrowstyle='-|>',mutation_scale=17,linewidth=1.55,color='#95a8b4',zorder=2))

ax.text(.17,4.92,'TUNED STMD  /  SENSOR-LEVEL ARCHITECTURE',fontsize=18,weight='bold',color=NAVY,va='center')
ax.text(19.82,4.92,'320 × 320   •   25 fps',fontsize=10.2,color=MUTED,ha='right',va='center')

x,w,y,h=card(xs[0],'01   Retina input','Grayscale video  •  320 × 320')
# miniature grayscale retinal mosaic
for i in range(7):
    for j in range(7):
        cx=x+.55+j*.247;cy=y+1.21+i*.247
        color=['#e7ecee','#cbd5da','#dfe7e9','#bac8cb'][(i*3+j*5)%4]
        if i==3 and j==4:color=NAVY
        ax.add_patch(Circle((cx,cy),.09,facecolor=color,edgecolor='#eff2f3',lw=.5,zorder=3))
ax.text(x+1.39,y+2.90,'',fontsize=8)

x,w,y,h=card(xs[1],'02   Self-motion','Phase correlation +\nslow background update')
ax.add_patch(FancyBboxPatch((x+.33,y+1.20),2.15,1.60,boxstyle='round,pad=.07',facecolor='#f4f7fa',edgecolor='#d6e1eb',zorder=2))
for r in range(5):
    ax.plot([x+.50,x+2.27],[y+1.41+r*.26]*2,color='#d1dbe2',lw=.75,zorder=3)
for c in range(5):
    ax.plot([x+.55+c*.36]*2,[y+1.37,y+2.53],color='#d1dbe2',lw=.75,zorder=3)
ax.add_patch(FancyArrowPatch((x+.75,y+1.7),(x+1.67,y+2.35),arrowstyle='-|>',mutation_scale=18,color=TEAL,lw=3,zorder=4))
ax.text(x+1.50,y+2.70,r'$(\Delta x,\Delta y)$',fontsize=10.5,color=TEAL,ha='center')

x,w,y,h=card(xs[2],'03   Small target','Difference of Gaussians\n+ surround texture inhibition')
N=65;g=np.linspace(-2.6,2.6,N);xx,yy=np.meshgrid(g,g)
Z=np.exp(-(xx*xx+yy*yy)/(.7**2))-0.44*np.exp(-(xx*xx+yy*yy)/(2.25**2))
ax.imshow(Z,cmap='RdBu_r',origin='lower',vmin=-.35,vmax=.8,extent=[x+.49,x+2.33,y+1.12,y+2.96],zorder=3,interpolation='bilinear')
ax.add_patch(Rectangle((x+.49,y+1.12),1.84,1.84,fill=False,edgecolor='#e5ecf0',lw=1.2,zorder=4))

x,w,y,h=card(xs[3],'04   Two pathways','Broad  +  density-inhibited\nresponses',GOLD)
for i,label in enumerate(('BROAD','SELECTIVE')):
    yy=y+2.46-i*.78
    ax.add_patch(FancyBboxPatch((x+.35,yy),2.12,.58,boxstyle='round,pad=.10,rounding_size=.08',facecolor='#ecf8f7' if i==0 else '#fff8eb',edgecolor='#b9e6df' if i==0 else '#f1d6ab',zorder=3))
    ax.text(x+1.39,yy+.28,label,fontsize=10.6,weight='bold',ha='center',va='center',color=TEAL if i==0 else '#ae791e',zorder=4)
ax.text(x+1.40,y+1.16,'spatial evidence maps',fontsize=9,color=MUTED,ha='center')

x,w,y,h=card(xs[4],'05   Motion neurons','14 velocity preferences\n+ leaky temporal integration')
cx=x+1.41;cy=y+2.15
ax.add_patch(Circle((cx,cy),.29,facecolor='#dceef0',edgecolor=TEAL,lw=1.4,zorder=3))
for k,a in enumerate(np.linspace(0,2*np.pi,14,endpoint=False)):
    r=.85
    start=(cx+.38*np.cos(a),cy+.38*np.sin(a))
    end=(cx+(r+0.28)*np.cos(a),cy+(r+0.28)*np.sin(a))
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=8,linewidth=1.3,color=TEAL if k in (1,2,3,4) else '#a8bec7',zorder=3))
ax.text(cx,y+1.0,r'$A_{v,t}=0.85\,\mathrm{shift}(A_{v,t-1})+e_t$',fontsize=8.4,color=DARK,ha='center')

x,w,y,h=card(xs[5],'06   Neural readout','Global peak  •  position\ndirection  •  evidence score',GOLD)
ax.add_patch(FancyBboxPatch((x+.38,y+1.19),2.07,1.76,boxstyle='round,pad=.06',facecolor='#f7fafb',edgecolor=BORDER,zorder=2))
ax.plot([x+.65,x+2.20],[y+1.42,y+2.62],color='#cbd9dd',lw=1,zorder=3)
ax.plot([x+.65,x+2.20],[y+2.61,y+1.42],color='#cbd9dd',lw=1,zorder=3)
ax.add_patch(Circle((x+1.57,y+2.21),.19,fill=False,edgecolor=GOLD,lw=2.5,zorder=4))
ax.add_patch(Circle((x+1.57,y+2.21),.045,color=GOLD,zorder=4))
ax.text(x+1.43,y+1.66,'score / threshold',fontsize=9,color=TEAL,ha='center')

fig.savefig(OUT/'architecture.png',dpi=200,bbox_inches='tight',pad_inches=.12,facecolor='white')
fig.savefig(OUT/'architecture.svg',bbox_inches='tight',pad_inches=.12,facecolor='white')
plt.close(fig)
print('Created diagram')
