"""Accepted-field wave gauges and publication controls for the half-size pool.

uv run --with numpy --with matplotlib python tools/analyze-pool-impact-ab.py
Optional arguments: capture root, max1 arm, adaptive arm.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/pool-impact-ab")
arms = sys.argv[2:] or ["max1-final", "adaptive-final"]
out = root / os.environ.get("POOL_REPORT_DIR", "final-report")
out.mkdir(parents=True, exist_ok=True)
config = json.loads((root / arms[0] / "configuration.json").read_text())
nx, ny, nz = config["grid"]
dx = config["scene"]["voxelDomain"]["finestCellSize_m"]
rest = config["scene"]["container"]["height_m"] * config["scene"]["container"]["fillFraction"]
x = (np.arange(nx) + .5 - nx / 2) * dx
z = (np.arange(nz) + .5 - nz / 2) * dx
xx, zz = np.meshgrid(x, z)
# Four equal gauges: the outermost 0.4 m, centred on each wall, 0.8 m wide.
# These integrate complete coarse footprints as well as the max1 cells.
wall = ((abs(xx) >= nx * dx / 2 - .4) & (abs(zz) < .4)) | (
    (abs(zz) >= nz * dx / 2 - .4) & (abs(xx) < .4))
radius = np.hypot(xx, zz)
annuli = [(lo, lo + .2) for lo in np.arange(.4, 1.6, .2)]


def field(arm, step, name):
    a = np.fromfile(root / arm / f"{step}-{name}.bin", np.float32)
    return a.reshape(nz, ny, nx) if name != "velocity" else a.reshape(nz, ny, nx, 4)


def crossing(phi):
    """Highest liquid-to-air crossing; valid here after the drop has merged."""
    out = np.full((nz, nx), np.nan)
    for y in range(ny - 1):
        a, b = phi[:, y, :], phi[:, y + 1, :]
        mask = (a < 0) & (b >= 0)
        t = -a / (b - a + 1e-30)
        out[mask] = (y + .5 + t[mask]) * dx
    return out


histories = {}
summaries = []
for arm in arms:
    c = json.loads((root / arm / "configuration.json").read_text())
    assert c["grid"] == config["grid"] and c["dt"] == config["dt"]
    steps = sorted(int(p.stem.split("-")[0]) for p in (root / arm).glob("*-density.bin"))
    initial = field(arm, 0, "density")
    initial_mass = initial.sum(dtype=np.float64) * dx**3
    rows = []
    for step in steps:
        density = field(arm, step, "density")
        height = density.sum(axis=1, dtype=np.float64) * dx
        surface = crossing(.5 - density)
        velocity = field(arm, step, "velocity")
        # Constant water density omitted: kinetic is specific-volume energy.
        kinetic = .5 * np.sum(density * np.sum(velocity[..., :3]**2, axis=3), dtype=np.float64) * dx**3
        row = dict(step=step, time_s=step * c["dt"],
            mass_m3=float(density.sum(dtype=np.float64) * dx**3),
            kinetic_m5_s2=float(kinetic),
            wallMassHeight_m=float(height[wall].mean()),
            wallIsoHeight_m=float(np.nanmean(surface[wall])),
            meanMassHeight_m=float(height.mean()),
            wallPerturbation_m=float((height - rest)[wall].mean()),
            annulusPerturbations_m=[float((height - rest)[(radius >= lo) & (radius < hi)].mean()) for lo, hi in annuli])
        rows.append(row)
    histories[arm] = rows
    first = [row for row in rows if .5 <= row["time_s"] <= 1.4]
    crest = max(first, key=lambda row: row["wallPerturbation_m"])
    later = [row for row in rows if row["time_s"] >= 1.8]
    trace = json.loads((root / arm / "trace.json").read_text())
    summaries.append(dict(arm=arm, finalCells=trace[-1]["cells"],
        meanCells=float(np.trapezoid([r["cells"] for r in trace], [r["time"] for r in trace]) / trace[-1]["time"]),
        finalHistogram=trace[-1]["histogram"], initialMass_m3=float(initial_mass),
        relativeFinalMassDrift=float(rows[-1]["mass_m3"] / initial_mass - 1),
        firstWallCrest_m=crest["wallPerturbation_m"], firstWallCrestTime_s=crest["time_s"],
        firstWallPeakToTrough_m=max(r["wallPerturbation_m"] for r in first) - min(r["wallPerturbation_m"] for r in first),
        laterWallPeakToTrough_m=max(r["wallPerturbation_m"] for r in later) - min(r["wallPerturbation_m"] for r in later),
        finalKinetic_m5_s2=rows[-1]["kinetic_m5_s2"]))

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
for arm, rows in histories.items():
    t = [r["time_s"] for r in rows]
    axes[0].plot(t, [r["wallPerturbation_m"] * 1000 for r in rows], label=arm)
    axes[1].plot(t, [(r["wallIsoHeight_m"] - rest) * 1000 for r in rows], label=arm)
    axes[2].plot(t, [r["kinetic_m5_s2"] for r in rows], label=arm)
for ax, title, unit in zip(axes, ["Wall gauge · accepted column mass", "Wall gauge · density 0.5 surface", "Accepted kinetic energy / water density"], ["Height change (mm)", "Height change (mm)", "m⁵/s²"]):
    ax.set(title=title, xlabel="Time (s)", ylabel=unit)
    ax.grid(alpha=.2)
axes[0].legend(frameon=False)
fig.savefig(out / "wave-gauges.png", dpi=160)
plt.close(fig)

fig, axes = plt.subplots(1, len(arms), figsize=(5 * len(arms), 4), squeeze=False, layout="constrained")
for ax, arm in zip(axes[0], arms):
    rows = histories[arm]
    values = np.array([r["annulusPerturbations_m"] for r in rows]).T * 1000
    image = ax.imshow(values, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-60, vmax=60,
        extent=[0, rows[-1]["time_s"], .4, 1.6])
    ax.set(title=arm, xlabel="Time (s)", ylabel="Radius (m)")
fig.colorbar(image, ax=list(axes[0]), label="Annular mean column-height change (mm)")
fig.savefig(out / "wave-propagation.png", dpi=160)
plt.close(fig)

comparison = dict(gauge="Outer 0.4 m at all four wall centres, 0.8 m wide; equal area weights.",
    firstArrivalWindow_s=[.5, 1.4], laterWindowStart_s=1.8, summaries=summaries, histories=histories)
if len(arms) == 2:
    a, b = (field(arm, 0, "density") for arm in arms)
    comparison["initialDensityIdentical"] = bool(np.array_equal(a, b))
    comparison["initialDensityRelativeL1"] = float(np.abs(a - b).sum(dtype=np.float64) / a.sum(dtype=np.float64))
    comparison["wallCrestRatio"] = summaries[1]["firstWallCrest_m"] / summaries[0]["firstWallCrest_m"]

baseline = "max1-before-complete"
if (root / baseline).exists():
    step = config["steps"]
    plot_arms = [baseline, *arms]
    titles = [baseline, *arms]
    fig, axes = plt.subplots(1, len(plot_arms), figsize=(5 * len(plot_arms), 4), squeeze=False, layout="constrained")
    reconstruction = []
    for ax, arm, title in zip(axes[0], plot_arms, titles):
        phi = np.fromfile(root / arm / f"presentation-{step}/phi.bin", np.float32).reshape(nz, ny, nx)
        surface = crossing(phi)
        iso = crossing(.5 - field(arm, step, "density"))
        laplacian = surface[2:, 1:-1] + surface[:-2, 1:-1] + surface[1:-1, 2:] + surface[1:-1, :-2] - 4 * surface[1:-1, 1:-1]
        reconstruction.append(dict(arm=arm,
            laplacianRms_m=float(np.sqrt(np.nanmean(laplacian**2))),
            maxAxialNeighbourJump_m=float(max(np.nanmax(abs(np.diff(surface, axis=axis))) for axis in [0, 1])),
            maxDifferenceFromDensityIso_m=float(np.nanmax(abs(surface - iso)))))
        image = ax.imshow(surface, origin="lower", extent=[x[0] - dx / 2, x[-1] + dx / 2, z[0] - dx / 2, z[-1] + dx / 2],
            cmap="RdBu_r", vmin=.76, vmax=.91)
        ax.set(title=title, xlabel="x (m)", ylabel="z (m)")
    fig.colorbar(image, ax=list(axes[0]), label="Published surface height (m)")
    fig.suptitle(f"Half-size pool at {step * config['dt']:.4f} s · Dawn surface readback")
    fig.savefig(out / "surface-before-after.png", dpi=160)
    plt.close(fig)
    checked = []
    for path in (root / arms[0]).glob("*.bin"):
        if path.stem.split("-")[-1] not in ["density", "gamma", "velocity", "pressure", "divergence"]:
            continue
        original = root / baseline / path.name
        if original.exists():
            checked.append(dict(file=path.name, identical=path.read_bytes() == original.read_bytes()))
    comparison["reconstruction"] = reconstruction
    comparison["max1PhysicalFieldComparison"] = dict(files=len(checked), allIdentical=all(r["identical"] for r in checked),
        changed=[r["file"] for r in checked if not r["identical"]])
(out / "wave-comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
print(json.dumps({k: v for k, v in comparison.items() if k != "histories"}, indent=2))
