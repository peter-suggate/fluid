"""Render saved shipping triangles with matched QA shading and physical plots.

This script performs no remeshing, smoothing, field fitting or surface
projection. It is an offline geometry view, not the app's water optics.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def triangle_section(triangles, z=0.0):
    # Work a batch of one edge at a time rather than constructing Python
    # objects for every triangle in every emitted mesh.
    cuts = np.full((len(triangles), 3, 2), np.nan)
    valid = np.zeros((len(triangles), 3), dtype=bool)
    for edge in range(3):
        a, b = triangles[:, edge], triangles[:, (edge + 1) % 3]
        mask = ((a[:, 2] <= z) & (b[:, 2] > z)) | ((b[:, 2] <= z) & (a[:, 2] > z))
        alpha = (z - a[mask, 2]) / (b[mask, 2] - a[mask, 2])
        cuts[mask, edge] = a[mask, :2] + alpha[:, None] * (b[mask, :2] - a[mask, :2])
        valid[:, edge] = mask
    paired = valid.sum(axis=1) == 2
    return cuts[paired][valid[paired]].reshape(-1, 2, 2)


def rotation(q):
    w, x, y, z = [q[key] for key in ["w", "x", "y", "z"]]
    n = np.linalg.norm([w, x, y, z])
    w, x, y, z = np.array([w, x, y, z]) / n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def body_triangles(body):
    description, pose = body["description"], body["pose"]
    scale = np.array([description["dimensions_m"][key] for key in ["x", "y", "z"]])
    center = np.array([pose["position_m"][key] for key in ["x", "y", "z"]])
    if description["shape"] == "box":
        points = np.array([[x, y, z] for z in [-.5, .5] for y in [-.5, .5] for x in [-.5, .5]]) * scale
        faces = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                 [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
        triangles = points[faces]
    elif description["shape"] == "sphere":
        # The authored rigid sphere radius is dimensions.x (scene-shape ABI).
        theta = np.linspace(0, np.pi, 13)
        phi = np.linspace(0, 2*np.pi, 25)
        points = np.stack([np.sin(theta[:, None])*np.cos(phi),
                           np.cos(theta[:, None])*np.ones_like(phi),
                           np.sin(theta[:, None])*np.sin(phi)], axis=-1) * scale[0]
        triangles = np.concatenate([np.stack([points[:-1, :-1], points[1:, :-1], points[1:, 1:]], axis=-2).reshape(-1, 3, 3),
                                    np.stack([points[:-1, :-1], points[1:, 1:], points[:-1, 1:]], axis=-2).reshape(-1, 3, 3)])
    else:
        return np.empty((0, 3, 3))
    return triangles @ rotation(pose["orientation"]).T + center


def load_frame(root, arm, step):
    directory = root / arm / f"step-{step}"
    receipt = read_json(directory / "receipt.json")
    if receipt is None or not (directory / "mesh.bin").exists():
        return None, receipt
    data = np.fromfile(directory / "mesh.bin", dtype="<f4").reshape(-1, 3, 8)
    return data, receipt


def mesh_metrics(data, config):
    if data is None:
        return None
    triangles = data[:, :, :3]
    points, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    all_faces = inverse.reshape(-1, 3)
    valid = (all_faces[:, 0] != all_faces[:, 1]) & (all_faces[:, 1] != all_faces[:, 2]) & (all_faces[:, 2] != all_faces[:, 0])
    faces = all_faces[valid]
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    ends = points[edges[counts == 1]]
    lo = np.array(config["origin"])
    scene = config["scene"]["container"]
    hi = lo + np.array([scene["width_m"], scene["height_m"], scene["depth_m"]])
    boundary = np.zeros(len(ends), dtype=bool)
    for axis in range(3):
        for plane in [lo[axis], hi[axis]]:
            boundary |= np.all(abs(ends[:, :, axis] - plane) < 1e-5, axis=1)
    a, b, c = triangles[:, 0].astype(np.float64), triangles[:, 1].astype(np.float64), triangles[:, 2].astype(np.float64)
    parents = np.arange(len(points))
    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    for face in faces:
        first = root(face[0])
        parents[root(face[1])] = first; parents[root(face[2])] = first
    labels = np.array([root(face[0]) for face in all_faces])
    components = []
    volumes = np.einsum("ij,ij->i", a, np.cross(b, c))/6
    for label in np.unique(labels):
        mask = labels == label
        if not np.any(mask & valid): continue
        vertices = triangles[mask].reshape(-1, 3)
        components.append({"triangles": int(mask.sum()), "signedVolume_m3": float(volumes[mask].sum()),
                           "bounds_m": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()]})
    components.sort(key=lambda component: -abs(component["signedVolume_m3"]))
    return {"triangles": int(len(triangles)), "uniqueVertices": int(len(points)),
            "nonFiniteVertices": int((~np.isfinite(triangles).all(axis=-1)).sum()),
            "degenerateTriangles": int((~valid).sum()), "openEdges": int((counts == 1).sum()),
            "openEdgesAwayFromAuthoredBoundary": int((~boundary).sum()),
            "nonManifoldEdges": int((counts > 2).sum()),
            "signedTriangleVolume_m3": float(volumes.sum()), "connectedComponents": components,
            "triangleArea_m2": float(np.linalg.norm(np.cross(b-a, c-a), axis=1).sum()/2)}


def physical_metrics(root, arm, step, config):
    directory = root / arm / f"step-{step}"
    nx, ny, nz = config["dimensions"]
    rho = np.fromfile(directory / "density.bin", dtype="<f4").reshape(nz, ny, nx)
    velocity = np.fromfile(directory / "velocity.bin", dtype="<f4").reshape(nz, ny, nx, 4)[..., :3]
    mass = rho.sum(dtype=np.float64)
    symmetry = {"mirrorX": float(abs(rho-rho[:, :, ::-1]).sum(dtype=np.float64)/mass),
                "mirrorZ": float(abs(rho-rho[::-1, :, :]).sum(dtype=np.float64)/mass)}
    if nx == nz: symmetry["swapXZ"] = float(abs(rho-rho.transpose(2, 1, 0)).sum(dtype=np.float64)/mass)
    momentum = (rho[..., None]*velocity).sum(axis=(0, 1, 2), dtype=np.float64) * config["h"]**3 * config["scene"]["fluid"]["density_kg_m3"]
    return {"scope": "native cell means expanded over the authored-domain crop",
            "densitySymmetryL1_over_amount": symmetry, "momentum_kg_m_s": momentum.tolist()}


def audited_mesh_metrics(root, arm, step, config):
    metrics = mesh_metrics(load_frame(root, arm, step)[0], config)
    if metrics is None: return None
    nx, ny, nz = config["dimensions"]
    rho = np.fromfile(root / arm / f"step-{step}" / "density.bin", dtype="<f4").reshape(nz, ny, nx)
    h, origin = config["h"], np.array(config["origin"])
    for component in metrics["connectedComponents"]:
        lo, hi = np.array(component["bounds_m"])
        overlaps = [np.maximum(0, np.minimum(hi[axis], origin[axis]+h*(np.arange(n)+1))
                               - np.maximum(lo[axis], origin[axis]+h*np.arange(n)))
                    for axis, n in enumerate([nx, ny, nz])]
        wx, wy, wz = overlaps
        sampled = rho[np.ix_(wz > 0, wy > 0, wx > 0)]
        component["nativeExpandedAmountInBoundingBox_m3"] = float(np.einsum("zyx,z,y,x->", rho, wz, wy, wx))
        component["maximumNativeDensityInBoundingBox"] = float(sampled.max()) if sampled.size else None
    return metrics


def expected_ball(config, time):
    scene = config["scene"]
    volumes = scene["fluid"].get("initialLiquidVolumes", [])
    if not volumes or volumes[0]["shape"] != "sphere":
        return None
    ball = volumes[0]
    center = np.array([ball["center_m"][key] for key in ["x", "y", "z"]])
    radius = ball["radius_m"]
    gravity = np.array([scene["fluid"]["gravity_m_s2"][key] for key in ["x", "y", "z"]])
    # Production transports before adding gravity. After n completed steps,
    # undisturbed material has received n-1 moving gathers, not n full kicks.
    # This is a QA reference only; the saved mesh is never transformed.
    dt = config["dt"]
    step = round(time / dt)
    if not np.isclose(time, step * dt):
        raise ValueError("The split-step reference requires a captured step boundary")
    center += gravity * dt * dt * step * (step - 1) / 2
    pool = scene["container"]["height_m"] * scene["container"]["fillFraction"]
    return (center, radius) if center[1] - radius > pool else None


def format_panel(ax, config, projection):
    lo, hi = np.array(config["camera"]["bounds_m"])
    if projection == "mesh":
        ax.set_proj_type("ortho")
        ax.view_init(elev=config["camera"]["elevation_deg"], azim=config["camera"]["azimuth_deg"])
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[2], hi[2]); ax.set_zlim(lo[1], hi[1])
        ax.set_box_aspect((hi-lo)[[0, 2, 1]])
        ax.set_xlabel("x (m)", labelpad=0); ax.set_ylabel("z (m)", labelpad=0); ax.set_zlabel("y (m)", labelpad=0)
        ax.grid(False)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_pane_color((.97, .975, .98, 1))
    else:
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.grid(alpha=.13)
    ax.tick_params(labelsize=7)


def draw_panel(ax, config, data, receipt, arm, step, projection):
    format_panel(ax, config, projection)
    time = step * config["dt"]
    label = "Original local min8/max8" if arm == "coarse" and config.get("originalRegionQuery") else "Coarse-first" if arm == "coarse" else "Global min1/max1"
    if data is None:
        if projection == "mesh":
            ax.text2D(.5, .5, "No mesh captured", transform=ax.transAxes, ha="center", color="#b33")
        else:
            ax.text(.5, .5, "No mesh captured", transform=ax.transAxes, ha="center", color="#b33")
        ax.set_title(f"{label} · {time:g} s", fontsize=10)
        return
    triangles = data[:, :, :3]
    bodies = receipt.get("bodies", [])
    if projection == "mesh":
        normal = data[:, :, 4:7].mean(axis=1)
        normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-12)
        light = np.array([-.35, .85, .45]); light /= np.linalg.norm(light)
        intensity = .32 + .68*np.maximum(0, normal@light)
        color = np.clip(np.array([.18, .66, .80])[None, :] * intensity[:, None] + .08, 0, 1)
        ax.add_collection3d(Poly3DCollection(triangles[:, :, [0, 2, 1]], facecolors=color, edgecolors="none", antialiased=False))
        for index, body in enumerate(bodies):
            tri = body_triangles(body)
            if len(tri): ax.add_collection3d(Poly3DCollection(tri[:, :, [0, 2, 1]], facecolors=["#da9c44", "#c96651"][index % 2], edgecolors="none"))
        proof = "fine verified" if receipt.get("allFineVerified") else "FINE CONSTRAINT FAILED" if arm == "fine" else ""
        ax.text2D(.02, .03, f"q amount (crop): {receipt['amount_m3']:.6f} m³  {proof}\nNative widths: {receipt['nativeWidthActiveLeafHistogram']}",
                  transform=ax.transAxes, fontsize=7, color="#a33" if proof.startswith("FINE") else "#384856")
    else:
        segments = triangle_section(triangles)
        ax.add_collection(LineCollection(segments, colors="#147c9a", linewidths=1.15))
        ball = expected_ball(config, time)
        if ball:
            center, radius = ball; theta = np.linspace(0, 2*np.pi, 241)
            ax.plot(center[0] + radius*np.cos(theta), center[1] + radius*np.sin(theta), "--", color="#d78024", lw=1,
                    label="Pre-impact discrete-gravity sphere")
        for index, body in enumerate(bodies):
            p = body["pose"]["position_m"]
            ax.plot(p["x"], p["y"], marker="+", color=["#aa711c", "#ad463b"][index % 2], ms=8)
            ax.annotate(body["description"]["name"], (p["x"], p["y"]), fontsize=6)
    ax.set_title(f"{label} · {time:g} s", fontsize=10)


def render(root, output, selected_steps):
    configs = {arm: read_json(root / arm / "configuration.json") for arm in ["coarse", "fine"]}
    config = configs["coarse"] or configs["fine"]
    if config is None: raise SystemExit(f"No capture configuration under {root}")
    if all(configs.values()):
        for key in ["h", "dimensions", "dt", "steps", "camera", "values"]:
            if configs["coarse"][key] != configs["fine"][key]: raise ValueError(f"Unmatched A/B {key}")
    steps = selected_steps or config["steps"]
    output.mkdir(parents=True, exist_ok=True)
    for projection in ["mesh", "section"]:
        fig = plt.figure(figsize=(11, 4.4*len(steps)), constrained_layout=True)
        for row, step in enumerate(steps):
            for col, arm in enumerate(["coarse", "fine"]):
                ax = fig.add_subplot(len(steps), 2, row*2+col+1, projection="3d" if projection == "mesh" else None)
                data, receipt = load_frame(root, arm, step)
                draw_panel(ax, config, data, receipt, arm, step, projection)
        suffix = "Unmodified shipping triangles · matched camera · flat QA shading" if projection == "mesh" else "GPU triangle sections at z=0 · dashed orange: discrete-gravity sphere before impact"
        title = {"quarter": "Quarter pool impact", "half": "Half pool impact", "mini32": "Mini32 dam break", "rigid": "Settled tank with rigid bodies"}.get(config["sceneKey"], config["sceneId"])
        fig.suptitle(f"{title}\n{suffix}", fontsize=11)
        path = output / f"{projection}-comparison.png"; fig.savefig(path, dpi=115); plt.close(fig); print(path.resolve())
    # Every frame is a measured checkpoint. The animation holds each one for
    # one second and does not interpolate fluid motion between saved states.
    images = []
    for step in steps:
        fig = plt.figure(figsize=(11, 4.6), constrained_layout=True)
        for col, arm in enumerate(["coarse", "fine"]):
            data, receipt = load_frame(root, arm, step)
            draw_panel(fig.add_subplot(1, 2, col+1, projection="3d"), config, data, receipt, arm, step, "mesh")
        fig.suptitle(f"{config['sceneId']} · measured checkpoints / QA shading", fontsize=11)
        path = output / f"frame-{step}.png"; fig.savefig(path, dpi=100); plt.close(fig)
        images.append(Image.open(path).convert("RGB"))
    if images:
        images[0].save(output / "motion-checkpoints.gif", save_all=True, append_images=images[1:], duration=1000, loop=0)
        for image in images: image.close()
    comparisons = []
    for step in steps:
        coarse = read_json(root / "coarse" / f"step-{step}" / "receipt.json")
        fine = read_json(root / "fine" / f"step-{step}" / "receipt.json")
        if not coarse or not fine: continue
        comparisons.append({"step": step, "time_s": step*config["dt"], "allFineVerified": fine.get("allFineVerified"),
                            "amountDifference_m3": coarse["amount_m3"]-fine["amount_m3"],
                            "centerOfMassDifference_m": (np.array(coarse["centerOfMass_m"])-fine["centerOfMass_m"]).tolist(),
                            "kineticEnergy_J": {"coarse": coarse["kineticEnergy_J"], "fine": fine["kineticEnergy_J"]},
                            "mesh": {arm: audited_mesh_metrics(root, arm, step, config) for arm in ["coarse", "fine"]},
                            "physical": {arm: physical_metrics(root, arm, step, config) for arm in ["coarse", "fine"]}})
    (output / "comparison.json").write_text(json.dumps(comparisons, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("artifacts/retained-visual-ab/quarter"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", help="Comma-separated measured step numbers; defaults to capture configuration")
    args = parser.parse_args()
    render(args.input, args.output or args.input, [int(v) for v in args.steps.split(",")] if args.steps else None)
