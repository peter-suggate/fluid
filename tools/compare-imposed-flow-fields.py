"""Matched OLD shipping capture / NEW isolated prototype / analytic density.

Dense evaluation inspects the captured current coefficients. It does not fit,
smooth, project, advect, or replace them. Separate plots use actual GPU scalar
samples at identical finest-cell centers, so sampling density cannot make the
new arm appear to have a better mesh. No mesh is consumed or constructed.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--old", type=Path, default=Path("artifacts/retained-imposed-flow/sphere-full-fine"))
parser.add_argument("--new", type=Path, default=Path("artifacts/retained-imposed-flow/current-quadratic"))
parser.add_argument("--out", type=Path, default=Path("artifacts/retained-imposed-flow/comparison"))
args = parser.parse_args()
new_configuration = json.loads((args.new / "configuration.json").read_text())
old_configuration = json.loads((args.old / "configuration.json").read_text())
assert new_configuration["sphere"] == old_configuration["seed"]
assert new_configuration["grid"]["origin"] == old_configuration["origin_m"]
assert new_configuration["grid"]["dimensions"] == old_configuration["dimensions"]
assert new_configuration["grid"]["h"] == old_configuration["h"]
assert new_configuration["dt"] == old_configuration["dt"]
assert new_configuration["velocity_m_s"] == old_configuration["velocity_m_s"]
assert json.loads((args.new / "completed.json").read_text())["completed"]
args.out.mkdir(parents=True, exist_ok=True)
h = old_configuration["h"]
origin = np.array(old_configuration["origin_m"])
center = np.array(old_configuration["seed"]["center"])
R = old_configuration["seed"]["radius"]
width = old_configuration["seed"]["width"]
nx, ny, nz = old_configuration["dimensions"]
velocity = np.array(old_configuration["velocity_m_s"])
z = .025
x = np.linspace(-.5, .3, 800, endpoint=False) + .0005
y = np.linspace(.425, 1.175, 750, endpoint=False) + .0005
xx, yy = np.meshgrid(x, y)
points = np.stack([xx, yy, np.full_like(xx, z)], axis=-1)
coords = np.floor((points-origin)/h).astype(int)
ids = coords[..., 0] + nx*(coords[..., 1]+ny*coords[..., 2])
local = points-(origin+h*coords)
seed_q = np.clip(.5-(np.sum((points-center)**2, axis=-1)-R*R)/(2*R*width), 0, 1)
fine_x = origin[0]+(np.arange(nx)+.5)*h
fine_y = origin[1]+(np.arange(ny)+.5)*h
fx, fy = np.meshgrid(fine_x, fine_y)
fine_points = np.stack([fx, fy, np.full_like(fx, z)], axis=-1)
z_index = int(round((z-origin[2])/h-.5))
fig, axes = plt.subplots(3, 3, figsize=(14.8, 13.5), constrained_layout=True)
phi_fig, phi_axes = plt.subplots(3, 3, figsize=(14.8, 13.5), constrained_layout=True)
summary = []
for step in range(3):
    stage = "initial" if step == 0 else "scalar-publication"
    stem = f"step-{step}-{stage}"
    receipt = json.loads((args.old / f"{stem}.json").read_text())
    raw = np.fromfile(args.old / f"{stem}.bin", dtype="<f4")
    def plane(name):
        span = receipt["ranges"][name]
        return raw[span["offset"]:span["offset"]+span["count"]]
    old_coeff = plane("coeffA" if plane("control")[1] == 0 else "coeffB").reshape(-1, 2)
    old_q = old_coeff[ids, 0]*seed_q+old_coeff[ids, 1]
    new_receipt = json.loads((args.new / f"step-{step}-receipt.json").read_text())
    new_records = np.fromfile(args.new / f"step-{step}-records.bin", dtype="<f4").reshape(-1, 16)
    assert np.all(new_records.view("<u4")[ids, 10] == new_receipt["generation"])
    c = new_records[ids]
    lx, ly, lz = local[..., 0], local[..., 1], local[..., 2]
    new_phi = c[..., 0]+c[..., 1]*lx+c[..., 2]*ly+c[..., 3]*lz
    new_phi += .5*(c[..., 4]*lx*lx+c[..., 5]*ly*ly+c[..., 6]*lz*lz)
    new_phi += c[..., 7]*lx*ly+c[..., 8]*lx*lz+c[..., 9]*ly*lz
    new_q = np.clip(.5-new_phi/width, 0, 1)
    exact_center = center+step*old_configuration["dt"]*velocity
    exact_phi = (np.sum((points-exact_center)**2, axis=-1)-R*R)/(2*R)
    exact_q = np.clip(.5-exact_phi/width, 0, 1)
    labels = ["OLD production: captured affine density", "NEW isolated prototype: current GPU quadratic", "Independent exact translated density"]
    for row, values in enumerate([old_q, new_q, exact_q]):
        axis = axes[row, step]
        im = axis.imshow(values, origin="lower", extent=[-.5, .3, .425, 1.175],
                         cmap="Blues", vmin=0, vmax=1, interpolation="nearest", aspect="equal")
        axis.contour(xx, yy, values, levels=[.5], colors=["#142634"], linewidths=1)
        axis.contour(xx, yy, exact_q, levels=[.5], colors=["#d029ad"], linewidths=1.25)
        for gx in np.arange(-.5, .301, h):
            axis.axvline(gx, color="black", alpha=.1, linewidth=.35)
        for gy in np.arange(.45, 1.176, h):
            axis.axhline(gy, color="black", alpha=.1, linewidth=.35)
        axis.set(xlabel="X (m)", ylabel="Y (m)", title=f"{labels[row]}\nStep {step} · {step/30:.4f} s · shift {step*.025:.3f} m")
        axis.text(.02, .02, f"Section max q error: {np.max(np.abs(values-exact_q)):.3g}", transform=axis.transAxes,
                  fontsize=9, bbox={"facecolor":"white", "alpha":.92, "edgecolor":"none"})
    old_gpu_phi = np.fromfile(args.old / f"step-{step}-published-phi.bin", dtype="<f4").reshape(nz, ny, nx)[z_index]
    query_ids = np.fromfile(args.new / f"step-{step}-ids.bin", dtype="<u4")
    queries = np.fromfile(args.new / f"step-{step}-samples.bin", dtype="<f4").reshape(-1, 16)
    new_gpu_phi = np.full(nx*ny*nz, np.nan)
    new_gpu_phi[query_ids] = queries[:, 0]
    new_gpu_phi = new_gpu_phi.reshape(nz, ny, nx)[z_index]
    analytic_fine_phi = (np.sum((fine_points-exact_center)**2, axis=-1)-R*R)/(2*R)
    for row, values in enumerate([old_gpu_phi, new_gpu_phi, analytic_fine_phi]):
        axis = phi_axes[row, step]
        phi_im = axis.imshow(values, origin="lower", extent=[origin[0], origin[0]+nx*h, origin[1], origin[1]+ny*h],
                              cmap="RdBu_r", vmin=-.075, vmax=.075, interpolation="nearest", aspect="equal")
        axis.contour(fine_x, fine_y, values, levels=[0], colors=["#142634"], linewidths=1)
        axis.contour(xx, yy, exact_phi, levels=[0], colors=["#d029ad"], linewidths=1.25)
        axis.set(xlim=(-.5, .3), ylim=(.425, 1.175), xlabel="X (m)", ylabel="Y (m)",
                  title=f"{['OLD actual GPU published φ', 'NEW actual GPU query φ', 'Independent φ at identical centers'][row]}\nStep {step} · black: linear sample zero contour")
    summary.append({"step":step, "oldCoefficientSectionMaximumDensityError":float(np.max(np.abs(old_q-exact_q))),
                    "newCoefficientSectionMaximumDensityError":float(np.max(np.abs(new_q-exact_q))),
                    "oldReceipt":{k:v for k,v in receipt.items() if k not in ["cells", "ranges", "worstPoint", "worstJump"]},
                    "newReceipt":new_receipt})
fig.colorbar(im, ax=axes, label="Current density q", shrink=.55, pad=.015)
fig.suptitle("Same imposed flow, field and section: OLD capture / NEW prototype / analytic\n"
             "Current GPU coefficients evaluated at common points · no mesh, smoothing or surface repair", fontsize=14)
fig.savefig(args.out / "density-comparison.png", dpi=160)
plt.close(fig)
phi_fig.colorbar(phi_im, ax=phi_axes, label="Scalar φ (m)", shrink=.55, pad=.015)
phi_fig.suptitle("Independent GPU scalar observations at the same finest-cell centers\n"
                 "NEW is an isolated prescribed-flow prototype; shipping physics is not replaced · magenta: exact zero set", fontsize=14)
phi_fig.savefig(args.out / "gpu-phi-comparison.png", dpi=160)
plt.close(phi_fig)
(args.out / "comparison.json").write_text(json.dumps({"scope":new_configuration["scope"],
    "oldProvenance":json.loads((args.old / "provenance.json").read_text()),
    "newProvenance":json.loads((args.new / "provenance.json").read_text()), "steps":summary}, indent=2))
print(args.out / "density-comparison.png")
print(args.out / "gpu-phi-comparison.png")
