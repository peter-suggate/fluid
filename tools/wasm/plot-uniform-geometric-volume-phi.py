"""Plot saved native audit results; requires matplotlib and numpy."""
import gzip
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else
            "docs/research/uniform-geometric-volume-phi-2026-09-20")
arms = [("dt30-eight", "1/30 s · 8 sweeps", "#c04f35"),
        ("dt120-eight", "1/120 s · 8 sweeps", "#247b8a"),
        ("dt30-thirtytwo", "1/30 s · 32 sweeps", "#8661a1")]
data = {name: json.load(gzip.open(root / f"{name}.json.gz", "rt"))
        for name, _, _ in arms}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})
fig, axs = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
for name, label, color in arms:
    result = data[name]
    m = [s["metrics"] for s in result["stages"] if s["stage"] == "projected"]
    t = [r["time"] for r in result["receipts"]]
    axs[0].plot([0] + t, [100] + [x["contourArea"] / 12.8 for x in m],
                label=label, color=color, linewidth=2)
axs[0].axhline(100, color="#555", ls=":", lw=1)
axs[0].set(title="Extra sharpening does not prevent loss",
           xlabel="Simulated time (s)", ylabel="Phi area / initial area (%)")
axs[0].legend(loc="lower right", fontsize=9)

baseline = data["dt30-eight"]
stages = baseline["stages"]
advection, redistance = [], []
for i in range(0, len(stages), 6):
    a = [s["metrics"]["contourArea"] for s in stages[i:i+3]]
    advection.append(a[1]-a[0])
    redistance.append(a[2]-a[1])
t = [r["time"] for r in baseline["receipts"]]
axs[1].plot(t, np.cumsum(advection)/12.8, color="#c04f35", label="Advection")
axs[1].plot(t, np.cumsum(redistance)/12.8, color="#247b8a", label="Redistancing")
axs[1].axhline(0, color="#555", ls=":", lw=1)
axs[1].set(title="Loss enters during phi advection",
           xlabel="Simulated time (s)", ylabel="Cumulative area change (% of initial)")
axs[1].legend(fontsize=9)

for frame, color in [(15, "#247b8a"), (30, "#c04f35"), (60, "#8661a1"), (120, "#777")]:
    replay = [r for r in baseline["sharpeningReplays"] if r["frame"] == frame]
    axs[2].plot(range(len(replay)), [r["metrics"]["targetL1"] for r in replay],
                marker="o", ms=4, color=color, label=f"t = {frame/30:g} s")
axs[2].set(xticks=range(5), xticklabels=["0", "8", "32", "128", "512"],
           title="Frozen-state sharpening stalls by 8 sweeps",
           xlabel="Sharpening sweeps (categorical)", ylabel="Σ |V − phi target| (cell areas)")
axs[2].legend(fontsize=9)
for ax in axs:
    ax.grid(alpha=0.15)
fig.suptitle("Long dam · Uniform Geometric 2D · 192×96 · h = 12.5 mm", fontsize=14)
fig.savefig(root / "diagnosis.png", dpi=180)
fig.savefig(root / "diagnosis.svg")

fig, axs = plt.subplots(2, 1, figsize=(11, 7), layout="constrained")
for ax, (name, label, _) in zip(axs, arms[:2]):
    result = data[name]
    v = np.array(result["volume"]).reshape(96, 192)
    phi = np.array(result["phi"]).reshape(97, 193)
    im = ax.imshow(v, origin="lower", extent=[0, 2.4, 0, 1.2],
                   vmin=0, vmax=1, cmap="Blues", interpolation="nearest")
    ax.contour(np.linspace(0, 2.4, 193), np.linspace(0, 1.2, 97), phi,
               levels=[0], colors=["#c04f35"], linewidths=1)
    ax.set(title=f"{label} at t = 4 s — blue: V; red: phi = 0", xlabel="x (m)", ylabel="y (m)")
fig.colorbar(im, ax=axs, label="V / cell capacity (clipped at 1)", shrink=0.75)
fig.savefig(root / "final-fields.png", dpi=160)

if "traceReplays" in baseline:
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for replay, color in zip(baseline["traceReplays"], ["#247b8a", "#8661a1", "#c04f35"]):
        duration = replay["traces"]["durations"]
        ms = [d["fraction"]*1000/30 for d in duration]
        axs[0].plot(ms, [d["contourArea"]-duration[0]["contourArea"] for d in duration],
                    marker="o", color=color, label=f"Start of frame {replay['frame']}")
    replay = baseline["traceReplays"][1]
    duration = replay["traces"]["durations"]
    ms = [d["fraction"]*1000/30 for d in duration]
    axs[1].plot(ms, [d["integratedDivergence"] for d in duration], marker="o",
                label="Whole moving liquid region", color="#8661a1")
    axs[1].plot(ms, [d["divergenceInOriginalAir"] for d in duration], marker="o",
                label="Contribution in initially dry space", color="#c04f35")
    for ax in axs:
        ax.axhline(0, color="#555", ls=":", lw=1)
        ax.grid(alpha=0.15)
        ax.set_xlabel("Elapsed time in the same frozen velocity field (ms)")
        ax.legend(fontsize=9)
    axs[0].set(title="The frozen flow itself changes phi area", ylabel="Area change (cell areas)")
    axs[1].set(title="Compression in newly occupied space (frame 15)",
               ylabel="Integrated divergence (cell areas / s)")
    fig.suptitle("Fixed velocity · refined RK2 trajectories · quarter-cell observer grid", fontsize=13)
    fig.savefig(root / "frozen-step.png", dpi=180)
