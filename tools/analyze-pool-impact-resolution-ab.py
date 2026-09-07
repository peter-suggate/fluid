"""Compare frozen pool-impact captures in physical units; never modifies fields.

Requires numpy and matplotlib. Example:
  python tools/analyze-pool-impact-resolution-ab.py A_DIRECTORY B_DIRECTORY OUTPUT
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap, bp, output = map(Path, sys.argv[1:4])
output.mkdir(parents=True, exist_ok=True)
configs = [json.loads((p / "configuration.json").read_text()) for p in (ap, bp)]
traces = [json.loads((p / "trace.json").read_text()) for p in (ap, bp)]
a, b = configs
assert a["values"] == b["values"], "method values must match"
assert a["residentWGSLHash"] == b["residentWGSLHash"], "shader must match"
assert a.get("residentSourceHashes") == b.get("residentSourceHashes"), "all captured method sources must match"
assert a["dt"] == b["dt"] and a["steps"] == b["steps"]
assert a["freezeTopology"] and b["freezeTopology"]
assert a["maxCell"] == 1 and b["maxCell"] == 2
scenes = [json.loads(json.dumps(c["scene"])) for c in configs]
for scene in scenes:
    scene["fluid"].pop("refinementRegions")
assert scenes[0] == scenes[1], "only resolution bounds may differ"
nx, ny, nz = a["grid"]
h = a["scene"]["voxelDomain"]["finestCellSize_m"]
fluid = a["scene"]["fluid"]
rho_water = fluid["density_kg_m3"]
g = abs(fluid["gravity_m_s2"]["y"])
assert fluid["dynamicViscosity_Pa_s"] == fluid["surfaceTension_N_m"] == 0
cell_mass = rho_water * h**3
z, y, x = np.indices((nz, ny, nx))
xx, yy, zz = (x + .5 - nx/2)*h, (y + .5)*h, (z + .5 - nz/2)*h
radius = np.sqrt(xx**2 + zz**2)
radial_index = np.minimum((radius/(2*h)).astype(int), 11)
initial_mass_cells = traces[0][0]["mass"]
height_cells = initial_mass_cells/(nx*nz)
whole = int(height_cells)
minimum_pe = cell_mass*g*h*nx*nz*(whole**2/2 + (height_cells-whole)*(whole+.5))


def read_fields(path, step):
    density = np.fromfile(path / f"{step}-density.bin", dtype="<f4").reshape(nz, ny, nx).astype(float)
    velocity = np.fromfile(path / f"{step}-velocity.bin", dtype="<f4").reshape(nz, ny, nx, 4)[..., :3].astype(float)
    return density, velocity


rosters = []
for path, limit in [(ap, 1), (bp, 2)]:
    initial = None
    for step in range(a["steps"]+1):
        activity = json.loads((path / f"{step}-activity.json").read_text())
        roster = [(r["leafId"], r["coordinate"], r["spanBricks"], r["acceptedResolution"])
                  for r in activity["bricks"] if r["active"]]
        if initial is None:
            initial = roster
        assert roster == initial, (path, step, "topology changed")
        assert all(1 <= 8*r[2]/r[3] <= limit for r in roster)
    rosters.append(initial)
assert any(8*r[2]/r[3] == 2 for r in rosters[1]), "B must actually contain coarse cells"


def metrics(density, velocity, row):
    mass = density.sum()
    energy_axes = .5*cell_mass*(density[..., None]*velocity**2).sum(axis=(0, 1, 2))
    pe = cell_mass*g*(density*yy).sum()
    # These are integrated column volumes, not the highest visible interface.
    columns = density.sum(axis=1)*h
    radial = np.bincount(radial_index.ravel(), weights=density.ravel(), minlength=12)*h**3
    vertical = density.sum(axis=(0, 2))*h**3
    return dict(step=row["step"], time=row["time"], mass_kg=mass*cell_mass,
                mass_error_percent=100*(mass/initial_mass_cells-1),
                kinetic_J=float(energy_axes.sum()), kinetic_axes_J=energy_axes.tolist(),
                excess_potential_J=float(pe-minimum_pe), mechanical_excess_J=float(pe-minimum_pe+energy_axes.sum()),
                com_y_m=float((density*yy).sum()/mass),
                com_horizontal_m=float(np.hypot((density*xx).sum(), (density*zz).sum())/mass),
                rms_radius_m=float(np.sqrt((density*radius**2).sum()/mass)),
                above_pool_litres=float(density[yy > .5].sum()*h**3*1000),
                isovalue_cell_volume_litres=float((density > .5).sum()*h**3*1000),
                sub_isovalue_liquid_litres=float(density[density < .5].sum()*h**3*1000),
                compression_excess_litres=float(np.maximum(density-1, 0).sum()*h**3*1000),
                column_rms_variation_m=float(columns.std()),
                radial_volume_m3=radial.tolist(), vertical_volume_m3=vertical.tolist(),
                symmetry=row["symmetry"], pressure_iterations=row["pressureIterations"],
                pressure_relative_residual=row["pressureResidual"])


arms = [[], []]
comparison = []
snapshots = {}
for step in range(a["steps"]+1):
    da, va = read_fields(ap, step)
    db, vb = read_fields(bp, step)
    if step == 0:
        assert np.array_equal(da, db) and np.array_equal(va, vb), "initial physical fields must match"
    ma, mb = [metrics(d, v, trace[step]) for d, v, trace in [(da, va, traces[0]), (db, vb, traces[1])]]
    arms[0].append(ma)
    arms[1].append(mb)
    ca, cb = da.sum(axis=1)*h, db.sum(axis=1)*h
    restricted = np.zeros_like(da)
    for _, coordinate, span, resolution in rosters[1]:
        size = 8*span
        width = size//resolution
        ix, iy, iz = [8*q for q in coordinate]
        block = da[iz:iz+size, iy:iy+size, ix:ix+size]
        means = block.reshape(resolution, width, resolution, width, resolution, width).mean(axis=(1, 3, 5))
        restricted[iz:iz+size, iy:iy+size, ix:ix+size] = means.repeat(width, 0).repeat(width, 1).repeat(width, 2)
    common_mass = np.minimum(da, db)
    comparison.append(dict(step=step, time=step*a["dt"],
        density_l1_mean=float(np.abs(da-db).mean()),
        displaced_mass_percent=float(100*np.abs(da-db).sum()/(2*initial_mass_cells)),
        restricted_displaced_mass_percent=float(100*np.abs(restricted-db).sum()/(2*initial_mass_cells)),
        vertical_profile_distance_percent=float(100*np.abs(np.array(ma["vertical_volume_m3"])-mb["vertical_volume_m3"]).sum()/(2*initial_mass_cells*h**3)),
        radial_profile_distance_percent=float(100*np.abs(np.array(ma["radial_volume_m3"])-mb["radial_volume_m3"]).sum()/(2*initial_mass_cells*h**3)),
        column_rms_difference_m=float(np.sqrt(np.mean((ca-cb)**2))),
        column_max_difference_m=float(np.abs(ca-cb).max()),
        common_liquid_velocity_rms_m_s=float(np.sqrt((common_mass[..., None]*(va-vb)**2).sum()/common_mass.sum())),
        kinetic_ratio_B_A=mb["kinetic_J"]/max(ma["kinetic_J"], 1e-20)))
    if step in [0, 15, 30, 60, 120, 240]:
        snapshots[step] = (ca, cb, da[:, :, nx//2], db[:, :, nx//2])

summary = dict(configurations=configs, initial_mass_kg=initial_mass_cells*cell_mass,
               equilibrium_potential_J=minimum_pe, arms=arms, comparison=comparison)
(output / "analysis.json").write_text(json.dumps(summary, indent=2))
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
colors = ["#2468b4", "#d26424"]
labels = ["A · max1, frozen", "B · min1/max2, frozen"]
fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
for arm, color, label in zip(arms, colors, labels):
    t = [r["time"] for r in arm]
    for ax, key, title, unit in [(axes[0, 0], "kinetic_J", "Kinetic energy", "J"),
                               (axes[0, 1], "mechanical_excess_J", "Mechanical energy above flat rest", "J"),
                               (axes[0, 2], "above_pool_litres", "Liquid above y = 0.5 m", "litres")]:
        ax.plot(t, [r[key] for r in arm], color=color, label=label)
        ax.set(title=title, xlabel="Time (s)", ylabel=unit)
    axes[1, 0].plot(t, [max(m["mean"] for m in r["symmetry"]["density"]) for r in arm], color=color, label=label)
    axes[1, 1].plot(t, [max(m["maximum"] for m in r["symmetry"]["velocity"]) for r in arm], color=color, label=label)
axes[1, 0].set(title="Density symmetry · worst domain mean", xlabel="Time (s)", ylabel="Liquid fraction")
axes[1, 0].axhline(.001, color="gray", ls=":", lw=1)
axes[1, 1].set(title="Velocity symmetry · worst local error", xlabel="Time (s)", ylabel="m/s")
axes[1, 1].axhline(.02, color="gray", ls=":", lw=1)
axes[1, 2].plot([r["time"] for r in comparison], [r["displaced_mass_percent"] for r in comparison], color="#764398", label="Full density fields")
axes[1, 2].plot([r["time"] for r in comparison], [r["restricted_displaced_mass_percent"] for r in comparison], color="#32856d", ls="--", label="A restricted to B cells")
axes[1, 2].set(title="A/B density difference", xlabel="Time (s)", ylabel="Half L1 / initial mass (%)")
axes[0, 0].legend(frameon=False)
axes[1, 2].legend(frameon=False)
for ax in axes.flat:
    ax.grid(alpha=.18)
fig.suptitle("Quarter pool impact · identical initial fields, timestep and physics · frozen topology", fontsize=14)
fig.savefig(output / "comparison.png", dpi=160)
plt.close(fig)

fig, axes = plt.subplots(2, 4, figsize=(13, 6), constrained_layout=True)
for col, step in enumerate([30, 60, 120, 240]):
    for row, field in enumerate(snapshots[step][:2]):
        im = axes[row, col].imshow(field, origin="lower", extent=[-.8, .8, -.8, .8], vmin=.3, vmax=.65, cmap="viridis")
        axes[row, col].set_title(f'{"A" if row == 0 else "B"} · {step*a["dt"]:g} s')
        axes[row, col].set(xlabel="x (m)", ylabel="z (m)")
fig.colorbar(im, ax=axes, shrink=.8, label="Integrated column liquid depth (m)")
fig.suptitle("Column volume distribution · common colour scale (not rendered surface height)")
fig.savefig(output / "columns.png", dpi=160)
plt.close(fig)

for step in [0, 15, 30, 60, 120, 240]:
    print(json.dumps(dict(step=step, A={k:arms[0][step][k] for k in ["kinetic_J", "mechanical_excess_J", "com_y_m", "above_pool_litres", "mass_error_percent"]},
                        B={k:arms[1][step][k] for k in ["kinetic_J", "mechanical_excess_J", "com_y_m", "above_pool_litres", "mass_error_percent"]},
                        comparison=comparison[step])))

# Optional read-only stage captures from otherwise identical reruns.
if len(sys.argv) == 6:
    audit_summary = []
    for arm_index, path in enumerate(map(Path, sys.argv[4:6])):
        for step in [1, 2, 8, 15, 30, 60, 120]:
            if not (path / f"{step}-scalar-publication.bin").exists():
                continue
            stages = ["transport-velocity-extension", "conservative-transport", "gamma-diffusion", "surface-sharpening", "scalar-publication"]
            fields = [np.fromfile(path / f"{step}-{stage}.bin", dtype="<f4").reshape(nz, ny, nx).astype(float) for stage in stages]
            final, _ = read_fields((ap, bp)[arm_index], step)
            assert np.array_equal(fields[-1], final), "stage captures must reproduce original fields"
            energy = [float(cell_mass*g*(d*yy).sum()) for d in fields]
            gamma_path = path / f"{step}-scalar-publication-gamma.bin"
            gamma_metrics = None
            if gamma_path.exists():
                gamma = np.fromfile(gamma_path, dtype="<f4").reshape(nz, ny, nx).astype(float)
                wet = final > .5
                gamma_metrics = dict(minimum=float(gamma.min()), maximum=float(gamma.max()),
                    wet_minimum=float(gamma[wet].min()), wet_maximum=float(gamma[wet].max()),
                    mass_weighted_deviation_from_one=float((final*abs(gamma-1)).sum()/final.sum()))
            audit_summary.append(dict(arm="AB"[arm_index], step=step, gamma=gamma_metrics,
                potential_changes_J=dict(zip(stages[1:], np.diff(energy).tolist())),
                kinetic_change_J=arms[arm_index][step]["kinetic_J"]-arms[arm_index][step-1]["kinetic_J"],
                total_mechanical_change_J=arms[arm_index][step]["mechanical_excess_J"]-arms[arm_index][step-1]["mechanical_excess_J"]))
    (output / "stage-energy.json").write_text(json.dumps(audit_summary, indent=2))
