"""Summarize and plot native compensation artifacts (matplotlib + numpy)."""
import gzip
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root=Path(sys.argv[1] if len(sys.argv)>1 else "docs/research/uniform-geometric-compensation-2026-09-20")
names=["baseline","balance-uniform","balance-deficit","balance-surface-deficit","source-work","source-work-debt","source-rate-025","source-rate-1","source-rate-4","energy-cap"]
labels=["Default","Uniform contraction","Capacity deficit","Surface deficit","Pressure work","Work + debt","Rate ×0.25","Rate ×1","Rate ×4","Energy cap"]
def load(name):
    with gzip.open(root/(name+".json.gz"),"rt") as f:return json.load(f)
runs={n:load(n) for n in names}
summary={}
for name,r in runs.items():
    e=[s["metrics"] for s in r["energy"] if s["stage"]=="projected"]
    c=r.get("energyCompensation",[])
    late=e[2700:3600]
    fall=load("falling-drop-"+name)
    flat=load("flat-"+name)
    summary[name]={
        "meanKinetic90to120":sum(s["phiKinetic"] for s in late)/len(late),
        "peakKinetic0to5":max(s["phiKinetic"] for s in e[:150]),
        "finalExcess":e[-1]["excess"],
        "finalContourArea":r["receipts"][-1]["contourArea"],
        "volumeError":r["receipts"][-1]["volume"]-r["initialVolume"],
        "msPerStep":r["elapsedMs"]/len(e),
        "meanSolves":sum(s["solves"] for s in c)/len(c) if c else 1,
        "maxDivergenceError":max((s["maxLiquidDivergenceError"] for s in c),default=None),
        "maxBudgetMiss":max((s["budgetMiss"] for s in c),default=0),
        "minimumAlpha":min((s["alpha"] for s in c),default=1),
        "fallingDropSpeed":fall["receipts"][-1]["maxSpeed"],
        "flatFinalKinetic":flat["energy"][-1]["metrics"]["phiKinetic"],
        "mandatoryOverfillKinetic":load("overfilled-"+name)["energy"][-1]["metrics"]["phiKinetic"],
    }
with open(root/"summary.json","w") as f:json.dump(summary,f,indent=2)
colors={"baseline":"#293647","balance-deficit":"#29916f","balance-surface-deficit":"#2278b8","source-work":"#ab68a2","source-rate-4":"#d48021","energy-cap":"#c64e4c"}
fig,axes=plt.subplots(2,2,figsize=(13,8.5),constrained_layout=True)
for name in colors:
    e=[s["metrics"]["phiKinetic"]/1000 for s in runs[name]["energy"] if s["stage"]=="projected"]
    label=labels[names.index(name)]
    means=np.array(e).reshape(-1,150).mean(axis=1)
    axes[0,0].plot(np.arange(len(means))*5+2.5,means,label=label,color=colors[name],lw=1.6)
    if name!="source-work":axes[0,1].plot(np.arange(1,151)/30,e[:150],color=colors[name],lw=1.3,label=label)
axes[0,0].set(title="Late motion with overfill response retained",xlabel="Simulated seconds",ylabel="5 s mean kinetic energy (kJ/m)",yscale="log",ylim=(.005,30))
axes[0,0].legend(fontsize=8,ncol=2)
axes[0,1].set(title="Initial impact: check the cost of compensation",xlabel="Simulated seconds",ylabel="Kinetic energy (kJ/m)")
for ax in axes[0]:ax.grid(alpha=.2)
select=["baseline","balance-uniform","balance-deficit","balance-surface-deficit","source-work","source-rate-4","energy-cap"]
short=[labels[names.index(n)] for n in select]
barcolors=[colors.get(n,"#889aa0") for n in select]
axes[1,0].bar(short,[summary[n]["meanKinetic90to120"] for n in select],color=barcolors)
axes[1,0].set(title="Average kinetic energy, 90–120 s",ylabel="J/m")
axes[1,1].bar(short,[summary[n]["fallingDropSpeed"] for n in select],color=barcolors)
axes[1,1].axhline(9.80665*.4,color="#333",ls="--",lw=1,label="g × 0.4 s")
axes[1,1].set(title="Free-fall control rejects the strict energy cap",ylabel="Peak speed after 0.4 s (m/s)")
axes[1,1].legend(fontsize=8)
for ax in axes[1]:ax.tick_params(axis="x",rotation=28,labelsize=8);ax.grid(axis="y",alpha=.2)
fig.suptitle("Global compensation experiments · 2D pool impact · positive overfill source preserved",fontsize=14)
fig.savefig(root/"comparison.png",dpi=170)
fig.savefig(root/"comparison.svg")
