"""Visual motion comparison and source maps; matplotlib, numpy and Pillow required."""
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

root=Path("docs/research/uniform-geometric-surface-balance-2026-09-20")
def load(name):
    with gzip.open(root/(name+".json.gz"),"rt") as f:return json.load(f)

fig,axes=plt.subplots(2,2,figsize=(9,6.8),constrained_layout=True)
visual={}
for row,part in enumerate(["early","late"]):
    for col,mode in enumerate(["baseline","global"]):
        r=load("visual-"+part+"-"+mode)
        visual[row,col]=[s for s in r["energySnapshots"] if s["stage"]=="projected"]
def draw(frame):
    for (row,col),snaps in visual.items():
        ax=axes[row,col];ax.clear()
        s=snaps[frame];phi=np.array(s["phi"]).reshape(49,65)
        ax.contourf(np.arange(65)*.1,np.arange(49)*.1,phi,levels=[-100,0],colors=["#39a4cf"])
        ax.contour(np.arange(65)*.1,np.arange(49)*.1,phi,levels=[0],colors=["#0876a0"],linewidths=1)
        ax.set(xlim=(0,6.4),ylim=(0,4.8),aspect="equal",facecolor="#eff6f8")
        ax.set_title(("Default" if col==0 else "Surface-deficit balancing")+f" · {s['frame']/30+(300 if row else 0):.2f} s",fontsize=10)
        ax.set_xticks([0,2,4,6]);ax.set_yticks([0,2,4]);ax.tick_params(labelsize=8)
    return []
fig.suptitle("Same 2D scene and stepping · impact above, late motion below",fontsize=12)
animation=FuncAnimation(fig,draw,frames=150,interval=1000/15,blit=False)
animation.save(root/"motion-comparison.gif",writer=PillowWriter(fps=15),dpi=90)
draw(70);fig.savefig(root/"motion-still.png",dpi=150);plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for mode,color,label in [("baseline","#425063","Default"),("global","#188fa2","Surface-deficit balancing")]:
    r=load("impact-"+mode+"-300")
    ke=np.array([s["metrics"]["phiKinetic"] for s in r["energy"] if s["stage"]=="projected"])
    means=ke.reshape(-1,300).mean(axis=1)
    axes[0,0].plot(np.arange(len(means))*10+5,means,color=color,label=label)
axes[0,0].set(xlabel="Simulated seconds",ylabel="10 s mean kinetic energy (J/m)",title="Improvement persists over five minutes",yscale="log")
axes[0,0].legend(fontsize=8);axes[0,0].grid(alpha=.2)
timing=load("timing")
axes[0,1].bar(["Default","Balancing"],[timing["median"]["baseline"],timing["median"]["global"]],color=["#425063","#188fa2"])
axes[0,1].set(ylabel="Native milliseconds / step",title="Diagnostics off · median of five runs")
axes[0,1].grid(axis="y",alpha=.2)
for i,key in enumerate(["baseline","global"]):axes[0,1].text(i,timing["median"][key],f"{timing['median'][key]:.3f}",ha="center",va="bottom")
late=load("late-sources")
fields=next(s["fields"] for s in late["energySnapshots"] if s["stage"]=="balanceSources")
state=next(s for s in late["energySnapshots"] if s["stage"]=="sharpened")
for col,key,label in [(0,"positive","Original overfill expansion"),(1,"negative","Added contraction")]:
    ax=axes[1,col]
    values=np.array(fields[key]).reshape(48,64)
    if key=="negative":values=-values
    im=ax.imshow(values,origin="lower",extent=[0,6.4,0,4.8],aspect="equal",cmap="magma" if col==0 else "Blues",vmin=0)
    ax.contour(np.arange(65)*.1,np.arange(49)*.1,np.array(state["phi"]).reshape(49,65),levels=[0],colors=["#55bbbb"],linewidths=1)
    ax.set(title=label+" · late-state replay",xlabel="x (m)",ylabel="y (m)")
    fig.colorbar(im,ax=ax,label="Source magnitude (1/s)",shrink=.8)
fig.savefig(root/"practical-summary.png",dpi=170)
