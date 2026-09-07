"""Plot actual production mesh readbacks against independent authored geometry.

No surface is rebuilt, smoothed or projected by this script. Its line segments
are intersections of saved emitted triangles with the physical z=0 plane.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np


def section_segments(triangles, plane=0.0):
    segments = []
    for triangle in triangles:
        intersections = []
        for edge in range(3):
            a, b = triangle[edge], triangle[(edge + 1) % 3]
            az, bz = a[2] - plane, b[2] - plane
            if abs(az) < 1e-10:
                intersections.append(a[:2])
            if az * bz < 0:
                intersections.append((a + (b - a) * (-az / (bz - az)))[:2])
        unique = []
        for p in intersections:
            if not any(np.linalg.norm(p - q) < 1e-8 for q in unique):
                unique.append(p)
        if len(unique) == 2:
            segments.append(unique)
    return segments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("artifacts/retained-density-production"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/retained-density-production/analytic-mesh-sections.png"))
    args = parser.parse_args()
    sources = [args.input / f"coarse-first-pool-impact-{scale}" / "edit-0" for scale in ["quarter", "half"]]
    sources = [source for source in sources if (source / "receipt.json").exists()]
    if not sources:
        raise SystemExit("No measured production mesh receipt exists")
    fig, panels = plt.subplots(len(sources), 2, figsize=(12, 5.5 * len(sources)), squeeze=False, constrained_layout=True)
    for row, source in enumerate(sources):
        receipt = json.loads((source / "receipt.json").read_text())
        oracle, metrics = receipt["oracle"], receipt["meshMetrics"]
        mesh = np.fromfile(source / "mesh.bin", dtype=np.float32).reshape(-1, 8)
        triangles = mesh.reshape(-1, 3, 8)[:, :, :3]
        segments = section_segments(triangles)
        cx, cy, _ = oracle["sphereCenter"]
        radius, height, origin = oracle["sphereRadius"], oracle["poolHeight"], oracle["origin"]
        angle = np.linspace(0, 2 * np.pi, 1200)
        for col, panel in enumerate(panels[row]):
            panel.add_collection(LineCollection(segments, colors="#167894", linewidths=1.7, zorder=3))
            panel.plot(cx + radius * np.cos(angle), cy + radius * np.sin(angle), color="#dc7820", ls="--", lw=1.5, zorder=4)
            panel.plot([origin[0], -origin[0]], [height, height], color="#dc7820", ls="--", lw=1.5, zorder=4)
            if col == 0:
                region = oracle["scene"]["fluid"]["refinementRegions"][0]
                low, high = region["min_m"], region["max_m"]
                panel.add_patch(Rectangle((low["x"], low["y"]), high["x"] - low["x"], high["y"] - low["y"],
                                          facecolor="#aa65aa", edgecolor="#8d458d", alpha=.13, zorder=0))
                panel.text((low["x"] + high["x"]) / 2, .06 * oracle["scene"]["container"]["height_m"],
                           "min8 / max8", ha="center", color="#813f81", fontsize=9)
                panel.set_xlim(origin[0] * 1.05, -origin[0] * 1.05)
                panel.set_ylim(-.03, oracle["scene"]["container"]["height_m"] * 1.025)
                panel.set_title(f'{source.parent.name.removeprefix("coarse-first-pool-impact-").capitalize()} · actual original URL region')
            else:
                panel.set_xlim(cx - radius * 1.16, cx + radius * 1.16)
                panel.set_ylim(cy - radius * 1.16, cy + radius * 1.16)
                panel.set_title("Suspended sphere · emitted triangle section")
                panel.text(.5, .43, f'Whole mesh maximum errors\n'
                           f'vertices: {1000 * metrics["maximumSphereVertexError_m"]:.3f} mm\n'
                           f'triangle interiors: {1000 * metrics["maximumSphereInteriorError_m"]:.3f} mm',
                           transform=panel.transAxes, ha="center", fontsize=9, color="#304e60", backgroundcolor="white")
            panel.set_aspect("equal")
            panel.set_xlabel("x (m)")
            panel.set_ylabel("y (m)")
            panel.grid(alpha=.16)
    fig.suptitle("Analytic reference and actual production mesh at t = 0\nPhysical z = 0 cross section; fixed 0.05 m sample spacing", fontsize=14)
    fig.legend(handles=[Line2D([], [], color="#167894", lw=2, label="Saved GPU-emitted mesh"),
                        Line2D([], [], color="#dc7820", ls="--", lw=1.5, label="Independent plane / sphere reference")],
               loc="outside lower center", ncols=2, frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
