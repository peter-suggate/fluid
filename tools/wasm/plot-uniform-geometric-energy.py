"""Plot artifacts produced by uniform-geometric-energy-audit.ts (requires matplotlib)."""
import gzip
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/research/uniform-geometric-late-energy-2026-09-20")
def load(name):
    with gzip.open(root / f"{name}.json.gz", "rt") as f:
        return json.load(f)
baseline = load("baseline")
control = load("no-overfill")
request = load("baseline-input")
def projected(run):
    return [s["metrics"] for s in run["energy"] if s["stage"] == "projected"]
a, b = projected(baseline), projected(control)
t = np.arange(1, len(a) + 1) / 30
fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
colors = ["#096f9d", "#d66c24"]
for run, color, label in [(a, colors[0], "Default"), (b, colors[1], "Overfill pressure source disabled")]:
    ke = np.array([m["phiKinetic"] for m in run]) / 1000
    total = np.array([m["phiKinetic"] + m["phiPotential"] for m in run]) / 1000
    axes[0, 0].plot(t[:360], ke[:360], color=color, label=label, lw=1.3)
    count = len(ke) // 150
    axes[0, 1].plot(np.arange(count)*5+2.5, ke[:count*150].reshape(-1,150).mean(axis=1), color=color, label=label, marker=".")
    axes[1, 0].plot(t, total, color=color, label=label, lw=1.3)
axes[0, 0].set(title="Early kinetic bursts include ordinary sloshing", ylabel="Surface-weighted kinetic energy (kJ/m)", xlabel="Simulated seconds")
axes[0, 0].axvline(5, color="#999", ls=":")
axes[0, 1].set(title="The long-run energy floor falls with the source removed", ylabel="5-second mean kinetic energy (kJ/m)", xlabel="Simulated seconds", yscale="log")
axes[0, 1].legend(fontsize=8)
axes[1, 0].set(title="Surface-weighted mechanical energy", ylabel="Kinetic + gravitational energy (kJ/m)", xlabel="Simulated seconds")
for ax in [axes[0,0], axes[0,1], axes[1,0]]:
    ax.grid(alpha=.2)

nx, ny = request["dimensions"]
hx, hy = request["cellSize"]
s = next(s for s in baseline["energySnapshots"] if s["frame"] == 298 and s["stage"] == "sharpened")
ax = axes[1, 1]
im = ax.imshow(np.array(s["volume"]).reshape(ny,nx), origin="lower", extent=[0,nx*hx,0,ny*hy], vmin=0,vmax=3, cmap="magma", aspect="equal")
ax.contour(np.arange(nx+1)*hx, np.arange(ny+1)*hy, np.array(s["phi"]).reshape(ny+1,nx+1), levels=[0], colors=["#48e9df"], linewidths=1.3)
ax.set(title="9.93 s: volume persists outside the visible surface", xlabel="x (m)", ylabel="y (m)")
fig.colorbar(im, ax=ax, label="Conserved volume / cell area (clipped at 3)", shrink=.85)
ax.text(.02,.97,"Cyan = level-set surface; max V = 7.04",transform=ax.transAxes,va="top",color="white",fontsize=8)
fig.suptitle("2D coarse-first-pool-impact-half · 64 × 48 · dt = 1/30 s", fontsize=15)
fig.savefig(root / "energy-analysis.png", dpi=170)
fig.savefig(root / "energy-analysis.svg")
