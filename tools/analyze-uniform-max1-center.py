"""Matched accepted-state physics at the pool impact center (no rendered phi).

uv run --with numpy --with scipy --with matplotlib python tools/analyze-uniform-max1-center.py [arm ...]
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label, binary_fill_holes

root = Path("artifacts/pool-impact-ab")
arms = sys.argv[1:] or ["uniform-center", "max1-center"]
out = root / os.environ.get("POOL_REPORT_DIR", "uniform-max1-center-report")
out.mkdir(exist_ok=True, parents=True)
config = json.loads((root / arms[0] / "configuration.json").read_text())
nx, ny, nz = config["grid"]
h = config["scene"]["voxelDomain"]["finestCellSize_m"]
rest = config["scene"]["container"]["height_m"] * config["scene"]["container"]["fillFraction"]
x = (np.arange(nx) + .5 - nx / 2) * h
y = (np.arange(ny) + .5) * h
z = (np.arange(nz) + .5 - nz / 2) * h
xx, zz = np.meshgrid(x, z)
radius = np.hypot(xx, zz)
center = radius < .15
near = (radius >= .35) & (radius < .55)
wall = ((abs(xx) >= 1.2) & (abs(zz) < .4)) | ((abs(zz) >= 1.2) & (abs(xx) < .4))
pool_band = center[:, None, :] & ((y >= .4) & (y < 1.2))[None, :, None]
pressure_band = center[:, None, :] & ((y >= .3) & (y < .6))[None, :, None]


def field(arm, step, name):
    data = np.fromfile(root / arm / f"{step}-{name}.bin", np.float32)
    return data.reshape((nz, ny, nx, 4) if name == "velocity" else (nz, ny, nx))


def surface(density):
    """Upper boundary of 3D floor-connected liquid, excluding detached drops.

    Interior air pockets must not replace the free surface with their bottom.
    """
    labels, _ = label(density >= .5)
    pool_labels = np.unique(labels[:, 0, :])
    pool = np.isin(labels, pool_labels[pool_labels != 0])
    result = np.full((nz, nx), np.nan)
    for iy in range(ny - 1):
        a, b = density[:, iy, :], density[:, iy + 1, :]
        mask = pool[:, iy, :] & (b < .5)
        fraction = (a - .5) / (a - b + 1e-30)
        result[mask] = (iy + .5 + fraction[mask]) * h
    return result, pool


histories = {}
summary = []
for arm in arms:
    c = json.loads((root / arm / "configuration.json").read_text())
    assert c["grid"] == config["grid"] and c["dt"] == config["dt"]
    rows = []
    for step in sorted(int(p.name.split("-")[0]) for p in (root / arm).glob("*-density.bin")):
        rho, vel, pressure = (field(arm, step, name) for name in ["density", "velocity", "pressure"])
        # Projection uses atmospheric pressure outside its rho=.5 phase. The
        # dense multigrid texture may retain scratch values on inactive rows.
        pressure = np.where(rho > .5, pressure, 0)
        waterline, pool = surface(rho)
        eta = waterline - rest
        height = rho.sum(axis=1, dtype=np.float64) * h
        mass = rho.sum(dtype=np.float64) * h**3
        kinetic = .5 * (rho * (vel[..., :3]**2).sum(axis=-1)).sum(dtype=np.float64) * h**3
        momentum_y = (rho * vel[..., 1]).sum(dtype=np.float64) * h**3
        potential = 9.81 * (rho * y[None, :, None]).sum(dtype=np.float64) * h**3
        weights = rho[pool_band].astype(np.float64)
        interior = binary_fill_holes(pool) & ~pool
        below_surface = y[None, :, None] < waterline[:, None, :]
        underfilled = center[:, None, :] & below_surface & (rho < .5)
        row = dict(step=step, time_s=step * c["dt"], volume_m3=float(mass),
                   kinetic_m5_s2=float(kinetic), potential_m5_s2=float(potential), momentumY_m4_s=float(momentum_y),
                   centerSurface_m=float(np.nanmean(eta[center])),
                   centerMassHeight_m=float(height[center].mean() - rest),
                   nearSurface_m=float(np.nanmean(eta[near])),
                   nearMassHeight_m=float(height[near].mean() - rest),
                   wallMassHeight_m=float(height[wall].mean() - rest),
                   centerVerticalVelocity_m_s=float(np.dot(weights, vel[..., 1][pool_band]) / max(1e-20, weights.sum())),
                   centerPressure_Pa=float(pressure[pressure_band].mean()),
                   enclosedAirVolume_m3=float(((1 - np.minimum(rho, 1)) * interior).sum(dtype=np.float64) * h**3),
                   centerUnderfilledVolume_m3=float(((1 - rho) * underfilled).sum(dtype=np.float64) * h**3),
                   coreKinetic_m5_s2=float(.5 * (rho * (vel[..., :3]**2).sum(axis=-1) * (rho >= .9)).sum(dtype=np.float64) * h**3),
                   diffuseVolume_m3=float(rho[(rho > 1e-5) & (rho < .5)].sum(dtype=np.float64) * h**3),
                   symmetryDensityL1=float(abs(rho - rho[::-1, :, :]).sum(dtype=np.float64) / max(1e-20, rho.sum(dtype=np.float64))))
        rows.append(row)
    histories[arm] = rows
    late = [r for r in rows if r["time_s"] >= 1]
    wall_first = [r for r in rows if .5 <= r["time_s"] <= 1.4]
    wall_late = [r for r in rows if 1.8 <= r["time_s"] <= 3]
    impact = [r for r in rows if .2 <= r["time_s"] <= 1]
    peak = lambda records, key: max(records, key=lambda r: r[key])
    trough = lambda records, key: min(records, key=lambda r: r[key])
    summary.append(dict(arm=arm, initialVolume_m3=rows[0]["volume_m3"],
                        drift_percent=100 * (rows[-1]["volume_m3"] / rows[0]["volume_m3"] - 1),
                        impactCentralTrough=trough(impact, "centerSurface_m"),
                        reboundCentralPeak=peak(late, "centerSurface_m"),
                        lateCenterPeakToTrough_m=peak(late, "centerSurface_m")["centerSurface_m"] - trough(late, "centerSurface_m")["centerSurface_m"],
                        lateCenterMassPeakToTrough_m=peak(late, "centerMassHeight_m")["centerMassHeight_m"] - trough(late, "centerMassHeight_m")["centerMassHeight_m"],
                        firstWallCrest_m=peak(wall_first, "wallMassHeight_m")["wallMassHeight_m"],
                        firstWallCrestTime_s=peak(wall_first, "wallMassHeight_m")["time_s"],
                        lateWallPeakToTrough_m=peak(wall_late, "wallMassHeight_m")["wallMassHeight_m"] - trough(wall_late, "wallMassHeight_m")["wallMassHeight_m"],
                        final=rows[-1]))

initial_equality = {}
for arm in arms[1:]:
    for name in ["density", "velocity"]:
        a, b = field(arms[0], 0, name), field(arm, 0, name)
        initial_equality[f"{arms[0]}:{arm}:{name}"] = dict(bitIdentical=bool(np.array_equal(a.view(np.uint32), b.view(np.uint32))),
                                                        maxAbsError=float(abs(a-b).max()))

labels = {
    "uniform-center": "Uniform",
    "max1": "Sparse max1",
    "frozen015": "Coarse-first frozen at 0.15 s",
    "adaptive": "Coarse-first adaptive",
    "uniform-center-boundary": "Uniform · gamma on",
    "max1-center-conservative-d4": "Sparse max1 · gamma on",
    "uniform-center-no-gamma": "Uniform · gamma off",
    "max1-center-no-gamma": "Sparse max1 · gamma off",
}
def arm_label(arm):
    return labels.get(arm, labels.get(Path(arm).name, Path(arm).name))

fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True)
panels = [("centerSurface_m", "Center pool-connected density=.5 surface", "Elevation (m)"),
          ("centerMassHeight_m", "Center column mass (includes airborne liquid)", "Equivalent elevation (m)"),
          ("centerVerticalVelocity_m_s", "Center vertical liquid speed, y=0.4–1.2 m", "m/s"),
          ("centerPressure_Pa", "Center pressure, y=0.3–0.6 m", "Pa"),
          ("nearMassHeight_m", "Annular column mass, r=0.35–0.55 m", "Equivalent elevation (m)"),
          ("kinetic_m5_s2", "Total collocated kinetic energy / water density", "m⁵/s²")]
for ax, (key, title, ylabel) in zip(axes.flat, panels):
    for arm, rows in histories.items():
        ax.plot([r["time_s"] for r in rows], [r[key] for r in rows], label=arm_label(arm))
    ax.set(title=title, ylabel=ylabel, xlabel="Time (s)", xlim=(0, 3))
    ax.grid(alpha=.25)
axes[0, 0].legend()
fig.suptitle(os.environ.get("POOL_COMPARISON_TITLE", "Uniform vs max1")
             + " — accepted fields, h=0.05 m, dt=1/60 s\nCenter gauge: r<0.15 m; no presentation surface")
fig.tight_layout()
fig.savefig(out / "central-physics.png", dpi=170)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
for arm, rows in histories.items():
    for ax, key in zip(axes, ["centerSurface_m", "wallMassHeight_m"]):
        ax.plot([r["time_s"] for r in rows], [1000 * r[key] for r in rows], label=arm_label(arm))
for ax, title in zip(axes, ["Center: pool-connected density 0.5 surface", "Outer walls: accepted column mass"]):
    ax.set(title=title, xlabel="Time (s)", ylabel="Height change (mm)", xlim=(0, 3))
    ax.grid(alpha=.25)
axes[1].legend(fontsize=8)
fig.savefig(out / "physics-gauges.png", dpi=170)
plt.close(fig)

fig, axes = plt.subplots(len(arms), 4, figsize=(14, 3.2 * len(arms)), squeeze=False, sharex=True, sharey=True)
for ai, arm in enumerate(arms):
    for ti, step in enumerate([30, 60, 120, 180]):
        if not (root / arm / f"{step}-density.bin").exists():
            continue
        rho = field(arm, step, "density")
        ax = axes[ai, ti]
        image = ax.imshow(rho[nz//2], origin="lower", extent=(-nx*h/2, nx*h/2, 0, ny*h), vmin=0, vmax=1.1, cmap="Blues")
        ax.contour(x, y, rho[nz//2], levels=[.5], colors=["#dc671c"], linewidths=.7)
        ax.set(title=f"{arm_label(arm)}, {step / 60:g} s", xlabel="x (m)", ylabel="y (m)", ylim=(0, 1.6))
fig.tight_layout()
fig.savefig(out / "central-density-slices.png", dpi=170)
plt.close(fig)
report = dict(initialEquality=initial_equality, summaries=summary, histories=histories)
(out / "comparison.json").write_text(json.dumps(report, indent=2))
print(json.dumps(dict(initialEquality=initial_equality, summaries=summary), indent=2))
