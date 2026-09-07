"""Analyze matched fixed-domain mini32 captures; fields are never modified.

Usage: python tools/analyze-minidam32-fixed-domain.py [artifacts/minidam32-frozen]
Requires numpy and matplotlib. Optional full-resident-minmax1-audit captures
attribute the first symmetry threshold crossings to individual stages.
"""
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/minidam32-frozen")
paths = [root / "full-resident-adaptive", root / "full-resident-minmax1"]
out = root / "fixed-domain-analysis"
out.mkdir(parents=True, exist_ok=True)
configs = [json.loads((p / "configuration.json").read_text()) for p in paths]
traces = [json.loads((p / "trace.json").read_text()) for p in paths]
assert configs[0]["residentSourceHashes"] == configs[1]["residentSourceHashes"]
assert configs[0]["values"] == configs[1]["values"]
assert [c["maxCell"] for c in configs] == [0, 1]
scenes = [json.loads(json.dumps(c["scene"])) for c in configs]
for scene in scenes:
    scene["fluid"].pop("refinementRegions", None)
assert scenes[0] == scenes[1]
assert all(c["initialAtlasResident"] and c["freezeTopology"] for c in configs)
assert configs[0]["steps"] == configs[1]["steps"]
assert configs[0]["dt"] == configs[1]["dt"]
nx, ny, nz = configs[0]["grid"]
assert [nx, ny, nz] == [32, 32, 32]
h = scenes[0]["voxelDomain"]["finestCellSize_m"]
fluid = scenes[0]["fluid"]
cell_mass = fluid["density_kg_m3"] * h**3
g = abs(fluid["gravity_m_s2"]["y"])
z, y, x = np.indices((nz, ny, nx))


def read(path, step, kind):
    data = np.fromfile(path / f"{step}-{kind}.bin", dtype="<f4")
    return data.reshape(nz, ny, nx, 4) if kind == "velocity" else data.reshape(nz, ny, nx)


def symmetry(a):
    error = np.abs(a - a.transpose(2, 1, 0))
    return {"maximum": float(error.max()), "mean": float(error.mean())}


def roster(path, step):
    activity = json.loads((path / f"{step}-activity.json").read_text())
    assert activity["faultFlags"] == 0 and not activity["commitFailed"]
    return [(b["leafId"], b["coordinate"], b["spanBricks"], b["acceptedResolution"])
            for b in activity["bricks"] if b["active"]]


arms, comparisons = [[], []], []
for i, (path, trace) in enumerate(zip(paths, traces)):
    initial_roster = roster(path, 0)
    widths = np.zeros((nz, ny, nx), dtype=np.uint8)
    for _, (bx, by, bz), span, resolution in initial_roster:
        size = 8 * span
        widths[8*bz:8*bz+size, 8*by:8*by+size, 8*bx:8*bx+size] = size // resolution
    assert np.all(widths > 0), "the whole tank must be supported"
    if i == 1:
        assert np.all(widths == 1)
    else:
        assert widths.max() > 1
    for row in trace:
        step = row["step"]
        assert roster(path, step) == initial_roster
        rho = read(path, step, "density").astype(float)
        mass = rho.sum()
        potential = cell_mass * g * h * (rho * (y + .5)).sum()
        kinetic = cell_mass * row["kinetic"]
        arms[i].append(dict(step=step, time_s=row["time"],
            mass_error_percent=100 * (mass / trace[0]["mass"] - 1),
            com_y_m=h*row["centerOfMassY"], kinetic_J=kinetic,
            potential_J=potential, mechanical_J=kinetic+potential,
            sub_isovalue_mass_percent=100*rho[rho < .5].sum()/mass,
            compression_excess_percent=100*np.maximum(rho-1, 0).sum()/mass,
            symmetry=row["symmetry"]))

for step in range(configs[0]["steps"] + 1):
    columns = [h * read(p, step, "density").sum(axis=1, dtype=float) for p in paths]
    gap = columns[0] - columns[1]
    comparisons.append(dict(step=step, time_s=step*configs[0]["dt"],
        column_rms_gap_m=float(np.sqrt(np.mean(gap**2))),
        maximum_column_gap_m=float(np.abs(gap).max()),
        com_y_gap_m=arms[0][step]["com_y_m"]-arms[1][step]["com_y_m"]))
assert comparisons[0]["maximum_column_gap_m"] < 2e-9

limits = [("density", "maximum", .01), ("density", "mean", .001),
          ("velocity", "maximum", .02), ("velocity", "mean", .001)]
first_failures = [{f"{kind}_{field}": next((row["step"] for row in arm
    if max(v[field] for v in row["symmetry"][kind]) > limit), None)
    for kind, field, limit in limits} for arm in arms]
checkpoints = [0, 1, 4, 8, 15, 30, 60, 120]
summary = dict(scene=scenes[0]["sceneId"], arm_names=["Frozen adaptive", "Frozen min1/max1"],
    configurations=configs, fixed_full_domain_verified=True,
    first_symmetry_failure_step=first_failures,
    checkpoints=[dict(A=arms[0][step], B=arms[1][step], comparison=comparisons[step])
                 for step in checkpoints if step <= configs[0]["steps"]])

colors = ["#a75a39", "#187b80"]
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), layout="constrained")
time = [row["time_s"] for row in arms[0]]
for arm, color, label in zip(arms, colors, summary["arm_names"]):
    for ax, key, title, unit in [(axes[0, 0], "com_y_m", "Centre of mass", "Height (m)"),
                                (axes[0, 1], "kinetic_J", "Collocated kinetic energy", "Energy (J)"),
                                (axes[1, 0], "mass_error_percent", "Total mass error", "Error (%)")]:
        ax.plot(time, [r[key] for r in arm], color=color, label=label)
        ax.set(title=title, ylabel=unit)
    for ax, kind, threshold in [(axes[1, 1], "density", .01), (axes[1, 2], "velocity", .02)]:
        ax.semilogy(time[1:], [max(r["symmetry"][kind][0]["maximum"], 1e-9) for r in arm[1:]],
                    color=color, label=label)
        ax.set(title=f"Maximum x/z {kind} asymmetry", ylabel="Density" if kind == "density" else "Velocity (m/s)")
        if color == colors[0]: ax.axhline(threshold, color=".4", ls=":", lw=1)
axes[0, 2].plot(time, [1000*r["column_rms_gap_m"] for r in comparisons], color=".3")
axes[0, 2].set(title="A/B column-mass disagreement", ylabel="RMS equivalent depth (mm)")
for ax in axes.flat:
    ax.set(xlabel="Time (s)", xlim=(0, time[-1])); ax.grid(alpha=.2)
    if ax != axes[0, 2]: ax.legend(fontsize=8)
fig.suptitle("Minidam32: same complete tank, fixed topology, different cell sizes")
fig.savefig(out / "trajectory.png", dpi=170); plt.close(fig)

# Cross-sections of the actual published scalar, without filtering or averaging.
# Choosing the first z sample shows a wall-adjacent front; top-down panels show
# independent authoritative mass, so a renderer defect cannot hide behind it.
steps = [0, 4, 8, 15, 60, 120]
fig, axes = plt.subplots(3, len(steps), figsize=(17, 8.2), layout="constrained")
coords = (np.arange(32) + .5)*h
for column, step in enumerate(steps):
    for index, (path, label) in enumerate(zip(paths, summary["arm_names"])):
        phi = read(path, step, "published-phi")
        axes[0, column].contour(coords, coords, phi[0], levels=[0], colors=[colors[index]], linewidths=1.5)
        depth = h * read(path, step, "density").sum(axis=1, dtype=float)
        im = axes[index+1, column].imshow(depth, origin="lower", extent=[0, .8, 0, .8],
            interpolation="nearest", vmin=0, vmax=.8, cmap="viridis")
        axes[index+1, column].set(xlabel="x (m)", ylabel="z (m)" if column == 0 else None)
    axes[0, column].set(title=f"{step/30:.3f} s", xlabel="x (m)", xlim=(0, .8), ylim=(0, .8), aspect="equal")
    for row in range(3):
        for boundary in [.2, .4, .6]:
            axes[row, column].axvline(boundary, color=".65", ls=":", lw=.45)
            axes[row, column].axhline(boundary, color=".65", ls=":", lw=.45)
axes[0, 0].set_ylabel("Published wall section: y (m)")
axes[1, 0].set_ylabel("Adaptive mass columns: z (m)")
axes[2, 0].set_ylabel("Min1/max1 mass columns: z (m)")
fig.colorbar(im, ax=axes[1:, :], label="Equivalent liquid depth (m)", shrink=.6)
fig.suptitle("Published front: adaptive (brown), min1/max1 (teal) • Cell means are shown without smoothing")
fig.savefig(out / "fronts.png", dpi=160); plt.close(fig)

audit = root / "full-resident-minmax1-audit"
if (audit / "trace.json").exists():
    audit_config = json.loads((audit / "configuration.json").read_text())
    assert audit_config["residentSourceHashes"] == configs[1]["residentSourceHashes"]
    audit_trace = json.loads((audit / "trace.json").read_text())
    # Capture hooks must not change the measured trajectory.
    assert audit_trace == traces[1][:len(audit_trace)]
    template = np.fromfile(audit / "template.bin", dtype="<u4")
    cell_data = template[template[6]:template[6]+8*template[2]].reshape(-1, 8)
    cells = cell_data.view("<f4")
    # B uses width-one cells everywhere; each dense coordinate has one owner.
    fine = np.flatnonzero(np.all(cells[:, 4:7] == 1, axis=1))
    ids = np.full((nz, ny, nx), -1, dtype=int)
    xyz = cells[fine, :3].astype(int)
    ids[xyz[:, 2], xyz[:, 1], xyz[:, 0]] = fine
    assert np.all(ids >= 0) and len(fine) == nx*ny*nz
    rows = template[template[7]:template[7]+9*template[3]].reshape(9, -1)
    row_float = rows.view("<f4")
    terms = template[template[8]:template[8]+2*template[4]].reshape(-1, 2)
    offsets = rows[0] & 0x7fffff
    pairs = np.flatnonzero((rows[0] >> 23) == 2)
    is_fine = np.all(cells[:, 4:7] == 1, axis=1)
    accepted_rows = pairs[is_fine[terms[offsets[pairs], 0]] & is_fine[terms[offsets[pairs]+1, 0]]]
    assert len(accepted_rows) == 3*31*32*32, "all internal fine-grid faces, without dormant rows"
    row_keys = [(int(rows[1, row] >> 30), *map(float, row_float[6:9, row])) for row in accepted_rows]
    row_lookup = dict(zip(row_keys, accepted_rows))
    assert len(row_lookup) == len(accepted_rows)
    mirror_rows = np.array([row_lookup[(2-axis, zz, yy, xx)] for axis, xx, yy, zz in row_keys])
    row_capacity = json.loads((audit / "1-audit.json").read_text())["rows"]
    audits = []
    stages = ["transport-velocity-extension", "conservative-transport", "gamma-diffusion",
              "surface-sharpening", "scalar-publication"]
    for step in range(1, len(audit_trace)):
        liquid = np.fromfile(audit / f"{step}-pressure-liquid.bin", dtype="<f4")[ids]
        mismatch = liquid != liquid.transpose(2, 1, 0)
        rho = read(audit, step, "density")
        pairs = []
        for zz, yy, xx in np.argwhere(mismatch)[:8]:
            pairs.append(dict(coordinate=[int(xx), int(yy), int(zz)],
                density=float(rho[zz, yy, xx]), mirror_density=float(rho[xx, yy, zz]),
                member=bool(liquid[zz, yy, xx]), mirror_member=bool(liquid[xx, yy, zz])))
        faces = {}
        for stage in ["transport-velocity-extension", "face-preparation", "body-forces", "velocity-projection"]:
            data = np.fromfile(audit / f"{step}-{stage}-faces.bin", dtype="<f4")
            parity = int(data.view("<u4")[-1])
            assert parity in [0, 1]
            base = row_capacity * (parity if stage == "transport-velocity-extension" else 1-parity)
            error = h * np.abs(data[base+accepted_rows]-data[base+mirror_rows])
            worst = int(error.argmax())
            faces[stage] = dict(maximum_m_s=float(error[worst]), mean_m_s=float(error.mean()),
                axis=row_keys[worst][0], center_fine=row_keys[worst][1:])
        audits.append(dict(step=step, pressure_membership_mismatched_cells=int(mismatch.sum()),
            membership_pairs=pairs,
            native_internal_faces=faces,
            stages={stage: symmetry(read(audit, step, stage)) for stage in stages},
            final_velocity=traces[1][step]["symmetry"]["velocity"][0]))
    summary["minmax1_stage_audit"] = audits

(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(dict(first_symmetry_failure_step=first_failures,
    final_comparison=comparisons[-1], output=str(out)), indent=2))
