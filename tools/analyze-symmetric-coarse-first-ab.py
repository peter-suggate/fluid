"""Plot accepted Dawn fields, without relying on the presentation mesh.

Usage: python analyze-symmetric-coarse-first-ab.py [capture-directory]
Requires NumPy and Matplotlib. Each arm is captured by the matching Dawn probe.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/symmetric-coarse-first-ab")
arms = ["fine", "baseline", "corrected"]
labels = ["1³ reference", "Coarse-first · before", "Coarse-first · corrected estimator"]
config = json.loads((root / "fine" / "configuration.json").read_text())
nx, ny, nz = config["grid"]
step = config["steps"]
h = config["scene"]["voxelDomain"]["finestCellSize_m"]

def density(arm, at):
    return np.fromfile(root / arm / f"{at}-density.bin", np.float32).reshape(nz, ny, nx)

reference = density("fine", step)
comparisons = []
for at in range(step + 1):
    f = density("fine", at)
    fh = f.sum(axis=1)
    for arm in arms[1:]:
        d = density(arm, at)
        dh = d.sum(axis=1)
        comparisons.append(dict(step=at, time_s=at * config["dt"], arm=arm,
            densityRelativeL1=float(np.abs(d - f).sum() / f.sum()),
            integratedHeightRelativeL1=float(np.abs(dh - fh).sum() / fh.sum()),
            integratedHeightRms_m=float(np.sqrt(np.mean((dh - fh) ** 2)) * h),
            relativeMassDrift=float(d.sum(dtype=np.float64) / 2048 - 1)))

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.8), layout="constrained")
fig.suptitle(f"Symmetric expansion at {step * config['dt']:.4f} s · accepted fluid fields", fontsize=17)
for i, (arm, label) in enumerate(zip(arms, labels)):
    d = density(arm, step)
    height = d.sum(axis=1) * h
    ax = axes[0, i]
    im = ax.imshow(height, origin="lower", extent=[-.8, .8, -.8, .8], vmin=0, vmax=.22, cmap="viridis", interpolation="nearest")
    ax.set(title=label, xlabel="x (m)", ylabel="z (m)")
    ax = axes[1, i]
    middle = .5 * (d[nz // 2 - 1] + d[nz // 2])
    ax.imshow(middle, origin="lower", extent=[-.8, .8, 0, .8], vmin=0, vmax=1, cmap="Blues", interpolation="nearest")
    x = (np.arange(nx) + .5) * h - .8
    y = (np.arange(ny) + .5) * h
    ax.contour(x, y, middle, levels=[.5], colors=["#143851"], linewidths=1)
    ax.set(xlabel="x (m)", ylabel="y (m)", ylim=(0, .45), title="Centre section · density and ρ = 0.5 contour")
fig.colorbar(im, ax=list(axes[0]), shrink=.85, label="Integrated liquid height (m)")
fig.savefig(root / "comparison.png", dpi=170)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), layout="constrained")
for arm, label, color in zip(arms[1:], labels[1:], ["#b95b35", "#237a75"]):
    c = [c for c in comparisons if c["arm"] == arm]
    for ax, key, title in zip(axes, ["densityRelativeL1", "integratedHeightRelativeL1"],
        ["Accepted density difference", "Integrated height difference"]):
        ax.plot([v["time_s"] for v in c], [100 * v[key] for v in c], "o-", color=color, label=label, markersize=3)
        ax.set(xlabel="Simulation time (s)", ylabel="Relative L1 difference (%)", title=title)
        ax.grid(alpha=.2)
axes[0].legend(frameon=False)
fig.savefig(root / "error-history.png", dpi=170)
plt.close(fig)
(root / "comparison.json").write_text(json.dumps(comparisons, indent=2) + "\n")
print(json.dumps([c for c in comparisons if c["step"] == step], indent=2))
