"""Compare accepted CM12 fields with the independently published coarse surface."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path("artifacts/mini32-min4-display"))
parser.add_argument("--before", default="before")
parser.add_argument("--after", default="continuous")
parser.add_argument("--step", type=int, default=8)
args = parser.parse_args()
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
surfaces = []
fields_equal = {}
for col, (arm, title) in enumerate([(args.before, "Before"), (args.after, "Continuous reconstruction")]):
    path = args.root / arm / f"step-{args.step}"
    nx, ny, nz = json.loads((path / "source.json").read_text())["sampleDimensions"]
    phi = np.fromfile(path / "phi.bin", dtype=np.float32).reshape(nz, ny, nx)
    rho = np.fromfile(path / "density.bin", dtype=np.float32).reshape(nz, ny, nx)
    ax = axes[0, col]
    ax.imshow(phi[nz // 2], origin="lower", vmin=-.1, vmax=.1, cmap="coolwarm")
    ax.contour(phi[nz // 2], levels=[0], colors="black")
    ax.set_title(title + ": published surface")
    ax.set_ylabel("Height (finest cells)")
    ax = axes[1, col]
    ax.imshow(rho[nz // 2], origin="lower", vmin=0, vmax=1, cmap="Blues")
    ax.contour(phi[nz // 2], levels=[0], colors="red")
    ax.set_title("Accepted density + displayed contour")
    ax.set_xlabel("Distance (finest cells)")
    height = np.full((nz, nx), np.nan)
    for y in range(ny - 1):
        low, high = phi[:, y, :], phi[:, y + 1, :]
        crossing = (low <= 0) & (high > 0)
        np.divide(-low, high - low, out=np.zeros_like(low), where=crossing)
        height[crossing] = y + .5 - low[crossing] / (high[crossing] - low[crossing])
    surfaces.append(height)
fig.tight_layout()
fig.savefig(args.root / f"density-comparison-{args.step}.png", dpi=160)
plt.close(fig)
fig = plt.figure(figsize=(12, 5))
for i, (height, title) in enumerate(zip(surfaces, ["Before", "Continuous reconstruction"])):
    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    z, x = np.mgrid[:height.shape[0], :height.shape[1]]
    ax.plot_surface(x + .5, z + .5, height, color="#58b5c4", linewidth=0,
                    antialiased=True, rstride=1, cstride=1)
    ax.view_init(elev=28, azim=135)
    ax.set(xlim=(0, nx), ylim=(0, nz), zlim=(0, ny), title=title)
    ax.set_box_aspect((nx, nz, ny))
    ax.set_axis_off()
fig.tight_layout()
fig.savefig(args.root / f"surface-comparison-{args.step}.png", dpi=200)
for step in (1, 8, 16, 32):
    fields_equal[step] = {}
    for name in ("density", "velocity", "pressure", "solidOpenFraction"):
        before = args.root / args.before / f"step-{step}" / f"{name}.bin"
        after = args.root / args.after / f"step-{step}" / f"{name}.bin"
        if before.exists() and after.exists():
            fields_equal[step][name] = before.read_bytes() == after.read_bytes()
print(json.dumps({"physicalFieldsByteIdentical": fields_equal}, indent=2))
