"""Plot captured retained coefficients, with no mesh or surface reconstruction.

The image evaluates a*q_seed+b pointwise inside the owning fine support.
Discontinuities are preserved; the magenta contour is the exact translated
quadratic sphere. This is a CPU interpretation of actual GPU coefficients,
separate from the actual packed GPU scalar publication readbacks.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, default=Path("artifacts/retained-imposed-flow/sphere-full-fine"))
args = parser.parse_args()
config = json.loads((args.input / "configuration.json").read_text())
h = config["h"]
origin = np.array(config["origin_m"])
center = np.array(config["seed"]["center"])
radius = config["seed"]["radius"]
width = config["seed"]["width"]
velocity = np.array(config["velocity_m_s"])
nx, ny, nz = config["dimensions"]
z_slice = .025
x = np.linspace(-.5, .3, 800, endpoint=False) + .0005
y = np.linspace(.425, 1.175, 750, endpoint=False) + .0005
xx, yy = np.meshgrid(x, y)
points = np.stack([xx, yy, np.full_like(xx, z_slice)], axis=-1)
qindex = np.floor((points - origin) / h).astype(int)
indices = qindex[..., 0] + nx * (qindex[..., 1] + ny * qindex[..., 2])
seed_q = np.clip(.5 - (np.sum((points - center) ** 2, axis=-1) - radius ** 2) / (2 * radius * width), 0, 1)
fig, axes = plt.subplots(3, 3, figsize=(14.8, 12), gridspec_kw={"height_ratios": [1, 1, .62]}, constrained_layout=True)
profile = []
native_operator = []
semantic_audit = []
all_index = np.arange(nx * ny * nz)
all_points = np.stack([origin[0] + (all_index % nx + .5) * h,
                       origin[1] + ((all_index // nx) % ny + .5) * h,
                       origin[2] + (all_index // (nx * ny) + .5) * h], axis=-1)
all_seed_q = np.clip(.5 - (np.sum((all_points - center) ** 2, axis=-1) - radius ** 2) / (2 * radius * width), 0, 1)
for column, step in enumerate([0, 1, 2]):
    stage = "initial" if step == 0 else "scalar-publication"
    stem = f"step-{step}-{stage}"
    receipt = json.loads((args.input / f"{stem}.json").read_text())
    data = np.fromfile(args.input / f"{stem}.bin", dtype="<f4")
    def plane(name):
        span = receipt["ranges"][name]
        return data[span["offset"]:span["offset"] + span["count"]]
    bank = int(plane("control")[1])
    coeff = plane("coeffA" if bank == 0 else "coeffB").reshape(-1, 2)
    actual = coeff[indices, 0] * seed_q + coeff[indices, 1]
    expected_center = center + velocity * step * config["dt"]
    expected = np.clip(.5 - (np.sum((points - expected_center) ** 2, axis=-1) - radius ** 2) / (2 * radius * width), 0, 1)
    axis = axes[0, column]
    im = axis.imshow(actual, origin="lower", extent=[x[0]-.0005, x[-1]+.0005, y[0]-.0005, y[-1]+.0005],
                     vmin=0, vmax=1, cmap="Blues", interpolation="nearest", aspect="equal")
    axis.contour(xx, yy, actual, levels=[.5], colors=["#142634"], linewidths=1)
    axis.contour(xx, yy, expected, levels=[.5], colors=["#d029ad"], linewidths=1.5)
    for grid in np.arange(-.5, .301, h):
        axis.axvline(grid, color="black", alpha=.11, linewidth=.4)
    for grid in np.arange(.45, 1.176, h):
        axis.axhline(grid, color="black", alpha=.11, linewidth=.4)
    axis.set(title=f"Step {step} · {step * config['dt']:.4f} s · {step * .025:.3f} m shift\nReconstructed q from captured GPU coefficients", xlabel="X (m)", ylabel="Y (m)")
    axis.text(.02, .02, f"Max support-face q jump: {receipt['retainedQContinuityMaxJump']:.5f}",
              transform=axis.transAxes, fontsize=9, bbox={"facecolor":"white", "alpha":.92, "edgecolor":"none"})
    row = int(np.argmin(np.abs(y - center[1])))
    raw = np.fromfile(args.input / f"step-{step}-published-phi.bin", dtype="<f4").reshape(nz, ny, nx)
    raw_x = origin[0] + (np.arange(nx) + .5) * h
    raw_y = origin[1] + (np.arange(ny) + .5) * h
    raw_z = int(round((z_slice-origin[2])/h-.5))
    raw_phi = raw[raw_z]
    all_q = coeff[:, 0] * all_seed_q + coeff[:, 1]
    false_zero = np.flatnonzero((raw.reshape(-1) == 0) & (np.abs(all_q-.5) > 1e-4))
    false_zero = sorted(false_zero, key=lambda index: abs(all_q[index]-.5), reverse=True)
    semantic_audit.append({"step": step, "allPublishedExactZeroSamples": int(np.sum(raw == 0)),
        "falseExactZeroSamples": len(false_zero), "maximumDensityResidualAtPublishedZero":
        float(np.max(np.abs(all_q[false_zero]-.5))) if len(false_zero) else 0,
        "examples": [{"index": int(index), "point_m": all_points[index].tolist(),
            "published_phi_m": float(raw.reshape(-1)[index]), "a": float(coeff[index, 0]),
            "b": float(coeff[index, 1]), "q_seed": float(all_seed_q[index]),
            "retained_q": float(all_q[index]), "seed_mean": float(plane("seedMean")[index])}
            for index in false_zero[:8]]})
    raw_axis = axes[1, column]
    raw_im = raw_axis.imshow(raw_phi, origin="lower", extent=[origin[0], origin[0]+nx*h, origin[1], origin[1]+ny*h],
                            vmin=-.075, vmax=.075, cmap="RdBu_r", interpolation="nearest", aspect="equal")
    raw_axis.contour(raw_x, raw_y, raw_phi, levels=[0], colors=["#142634"], linewidths=1)
    raw_axis.contour(xx, yy, expected, levels=[.5], colors=["#d029ad"], linewidths=1.5)
    false_slice = np.flatnonzero((raw_phi == 0) & (np.abs(all_q.reshape(nz, ny, nx)[raw_z]-.5) > 1e-4))
    if len(false_slice):
        raw_axis.scatter(raw_x[false_slice % nx], raw_y[false_slice // nx], marker="x", c="#c66b00", s=30, linewidths=1.4)
    raw_axis.text(.02, .02, f"False φ=0 samples in full 3D field: {len(false_zero)}",
                   transform=raw_axis.transAxes, fontsize=9, bbox={"facecolor":"white", "alpha":.92, "edgecolor":"none"})
    raw_axis.set(xlim=(x[0]-.0005, x[-1]+.0005), ylim=(y[0]-.0005, y[-1]+.0005), xlabel="X (m)", ylabel="Y (m)",
                 title="Actual GPU-published φ samples\nBlack zero contour: linear sample interpolation")
    axes[2, column].plot(x, expected[row], color="#d029ad", label="Exact translated q", linewidth=2)
    axes[2, column].plot(x, actual[row], color="#142634", label="Captured coefficient q", linewidth=1.2)
    axes[2, column].axhline(.5, color="gray", linestyle=":", linewidth=.7)
    axes[2, column].set(xlabel="X (m)", ylabel="Density q", ylim=(-.04, 1.04),
                       title=f"Line at Y={y[row]:.4f} m, Z={z_slice:.3f} m")
    axes[2, column].grid(alpha=.15)
    if column == 0:
        axes[2, column].legend(fontsize=8, loc="upper right")
    profile.append({"step":step, "y_m":float(y[row]), "z_m":z_slice, "x_m":x.tolist(),
                    "retained_q":actual[row].tolist(), "exact_q":expected[row].tolist(),
                    "section_max_q_error":float(np.max(np.abs(actual-expected)))})
    # Independently identify the native operator without importing its code.
    # A half-cell translation of cell averages produces this binomial remap.
    mass_stage = "initial" if step == 0 else "conservative-transport"
    mass_stem = f"step-{step}-{mass_stage}"
    mass_receipt = json.loads((args.input / f"{mass_stem}.json").read_text())
    mass_raw = np.fromfile(args.input / f"{mass_stem}.bin", dtype="<f4")
    parity = mass_receipt["scalarParity"] ^ int(step > 0)
    span = mass_receipt["ranges"]["densityA" if parity == 0 else "densityB"]
    rho = mass_raw[span["offset"]:span["offset"]+span["count"]]
    span = mass_receipt["ranges"]["effective"]
    effective = mass_raw[span["offset"]:span["offset"]+span["count"]].reshape(-1, 4)
    fine_mass = np.zeros(nx * ny * nz)
    velocity_error = 0.
    for cell in mass_receipt["cells"]:
        fine_mass[cell["index"]] = rho[cell["id"]]
        if rho[cell["id"]] > 1e-5:
            velocity_error = max(velocity_error, float(np.max(np.abs(effective[cell["id"], :3]*h-velocity))))
    fine_mass = fine_mass.reshape(nz, ny, nx)
    if step == 0:
        initial_means = fine_mass.copy()
    predicted = initial_means.copy()
    for _ in range(step):
        left = np.zeros_like(predicted)
        left[:, :, 1:] = predicted[:, :, :-1]
        predicted = .5 * (predicted + left)
    error = np.abs(fine_mass-predicted)
    native_operator.append({"step": step,
        "maximumNativeMeanDifferenceFromRepeatedHalfCellLinearRemap": float(error.max()),
        "L1AmountDifferenceFromRepeatedHalfCellLinearRemap_m3": float(error.sum()*h**3),
        "wetOutputVelocityMaxError_m_s": velocity_error if step > 0 else None,
        "velocityNote": "Post-gather effective velocity is zero on dry output supports; the prescription is verified by pre-transport VEX receipts."})
fig.colorbar(im, ax=axes[0, :], label="Retained density q", shrink=.9, pad=.025)
fig.colorbar(raw_im, ax=axes[1, :], label="Published φ (m)", shrink=.9, pad=.025)
fig.suptitle("Imposed full-fine transport: retained density before meshing\n"
             "Coefficient reconstruction and independent GPU publication · magenta: exact sphere · orange ×: φ=0 while q≠½", fontsize=14)
fig.savefig(args.input / "retained-field-comparison.png", dpi=160)
plt.close(fig)
(args.input / "section-profiles.json").write_text(json.dumps(profile, indent=2))
(args.input / "native-transport-operator.json").write_text(json.dumps(native_operator, indent=2))
(args.input / "published-zero-semantic-audit.json").write_text(json.dumps(semantic_audit, indent=2))
print(args.input / "retained-field-comparison.png")
