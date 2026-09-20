"""Plot recorded native 2D surface experiments; requires numpy/matplotlib."""
from pathlib import Path
import gzip
import json
import numpy as np
import matplotlib.pyplot as plt

root = Path("docs/research/uniform-geometric-swept-extension-2026-09-20")
cases = [
    ("off", "Baseline", "#777777"),
    ("area-only", "Total area only", "#b07918"),
    ("regional-geometric-no-guard", "Regional", "#3577ba"),
    ("regional-geometric-band", "Regional + total area", "#24907c"),
]
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for name, label, color in cases:
    with gzip.open(root / f"{name}.json.gz") as file:
        data = json.load(file)
    frames = [s for s in data["stages"] if s["stage"] == "projected"]
    times = [s["frame"] / 30 for s in frames]
    for ax, metric in zip(axes, ["contourArea", "contourL1"]):
        ax.plot(times, [100 * s["metrics"][metric] / s["metrics"]["volume"] for s in frames],
                label=label, color=color, linewidth=1.6)
axes[0].set_ylabel("Surface area / V (%)")
axes[1].set_ylabel("Cellwise V/phi mismatch / V (%)")
axes[1].set_xlabel("Simulation time (s)")
axes[0].axhline(100, linewidth=.5, color="black", linestyle=":")
axes[0].legend(ncol=2)
for ax in axes:
    ax.grid(alpha=.15)
fig.suptitle("2D long dam · 192 × 96 · Δt = 1/30 s")
fig.tight_layout()
fig.savefig(root / "comparison.png", dpi=160)

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, sharey=True)
for ax, (name, label, _) in zip(axes, [cases[0], cases[3]]):
    with gzip.open(root / f"{name}.json.gz") as file:
        data = json.load(file)
    phi = np.array(data["phi"]).reshape(97, 193)
    volume = np.array(data["volume"]).reshape(96, 192)
    ax.imshow(volume, origin="lower", extent=[0, 192, 0, 96], vmin=0, vmax=1,
              cmap="Blues", interpolation="nearest")
    ax.contour(np.arange(193), np.arange(97), phi, levels=[0], colors=["#ec5b34"], linewidths=1)
    ax.set_title(f"{label} at 4 s; blue = V, orange = phi = 0")
    ax.set_ylim(0, 80)
fig.tight_layout()
fig.savefig(root / "surface-comparison.png", dpi=160)
