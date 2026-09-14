#!/usr/bin/env python3
"""Plot the native/UI gentle-moving-blob level-set diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "artifacts/level-set-volume/gentle-moving-blob-analysis.json"
PNG = ROOT / "artifacts/level-set-volume/gentle-moving-blob-analysis.png"
SVG = ROOT / "artifacts/level-set-volume/gentle-moving-blob-analysis.svg"

ARMS = ("lsv-moving", "lsv-stationary", "baseline-moving")
LABELS = {
    "lsv-moving": "LSV · moving",
    "lsv-stationary": "LSV · stationary",
    "baseline-moving": "Baseline · moving",
}
COLORS = {
    "lsv-moving": "#d1495b",
    "lsv-stationary": "#2a9d8f",
    "baseline-moving": "#3366aa",
}


def contour_segments(values: list[float], h: float, expected: list[float]) -> np.ndarray:
    """Convert packed fine-grid x0,y0,x1,y1 segments to expected-body metres."""
    points = np.asarray(values, dtype=float).reshape(-1, 2, 2)
    points[:, :, 0] = -0.8 + h * points[:, :, 0] - expected[0]
    points[:, :, 1] = h * points[:, :, 1] - expected[1]
    return points


def main() -> None:
    data = json.loads(INPUT.read_text())
    h = float(data["configuration"]["finestCellSizeM"])
    duration = float(data["configuration"]["durationS"])
    radius = float(data["analytic"]["selectedSliceRadiusM"])
    by_arm = {arm["arm"]: arm for arm in data["arms"]}

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#68717a",
        "axes.linewidth": 0.7,
        "grid.color": "#d9dde2",
        "grid.linewidth": 0.65,
    })
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.2))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.875, bottom=0.105,
                        wspace=0.19, hspace=0.38)
    fig.patch.set_facecolor("#fbfbfa")
    fig.suptitle("Gentle moving blob · shape and conservation after 6 s", fontsize=16,
                 fontweight="bold")
    fig.text(0.5, 0.94,
             "Native dynamics; contours are the final UI-published RDF in the expected translating frame",
             ha="center", va="top", color="#4f5963", fontsize=10)

    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    circle = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
    for col, arm_name in enumerate(ARMS):
        arm = by_arm[arm_name]
        ax = axes[0, col]
        initial = contour_segments(
            arm["contours"]["initialFine"], h, arm["initial"]["expectedCentroidM"]
        )
        final = contour_segments(
            arm["contours"]["finalFine"], h, arm["final"]["expectedCentroidM"]
        )
        ax.add_collection(LineCollection(initial, colors="#9aa0a6", linewidths=0.75,
                                         alpha=0.7, zorder=1))
        ax.add_collection(LineCollection(final, colors=COLORS[arm_name], linewidths=1.65,
                                         alpha=0.95, zorder=3))
        ax.plot(circle[:, 0], circle[:, 1], color="#20262d", linewidth=1.0,
                linestyle=(0, (4, 3)), zorder=2)
        ax.axhline(0, color="#c7ccd1", linewidth=0.5, zorder=0)
        ax.axvline(0, color="#c7ccd1", linewidth=0.5, zorder=0)
        ax.set_aspect("equal", adjustable="box")
        # A common asymmetric window retains the two far-trailing LSV fragments.
        ax.set_xlim(-0.70, 0.40)
        ax.set_ylim(-0.42, 0.42)
        ax.set_title(LABELS[arm_name], pad=7)
        ax.set_xlabel("x − expected center (m)")
        if col == 0:
            ax.set_ylabel("y − expected center (m)")
        else:
            ax.tick_params(labelleft=False)
        ax.grid(True)
        final_surface = arm["final"]["surface"]
        ax.text(0.03, 0.03,
                f"RDF area {100 * final_surface['representedAreaM2'] / arm['initial']['surface']['representedAreaM2']:.1f}%\n"
                f"centroid error {100 * arm['final']['nativeCentroidErrorM']:.2f} cm",
                transform=ax.transAxes, va="bottom", ha="left", fontsize=8.3,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.86,
                      "edgecolor": "#d2d6da"})

    legend = [
        Line2D([0], [0], color="#9aa0a6", lw=1.2, label="t = 0 RDF, ideally translated"),
        Line2D([0], [0], color="#555555", lw=1.8, label="actual t = 6 s RDF"),
        Line2D([0], [0], color="#20262d", lw=1.0, linestyle=(0, (4, 3)),
               label="analytic selected-section circle"),
    ]
    axes[0, 0].legend(handles=legend, loc="upper right", frameon=True, fontsize=8.0)

    metric_axes = axes[1]
    for arm_name in ARMS:
        arm = by_arm[arm_name]
        rows = arm["rows"]
        t = np.asarray([row["timeS"] for row in rows], dtype=float)
        initial_area = float(rows[0]["surface"]["representedAreaM2"])
        retained = 100.0 * np.asarray(
            [row["surface"]["representedAreaM2"] for row in rows], dtype=float
        ) / initial_area
        centroid_cm = 100.0 * np.asarray(
            [row["nativeCentroidErrorM"] for row in rows], dtype=float
        )
        initial_mass = float(rows[0]["nativeLiquidMeasureFine"])
        overcapacity = 100.0 * np.asarray([
            (row.get("levelSetVolume") or {}).get("totalVolumeOverCapacity", 0.0)
            for row in rows
        ], dtype=float) / initial_mass
        for ax, values in zip(metric_axes, (retained, centroid_cm, overcapacity)):
            ax.plot(t, values, color=COLORS[arm_name], linewidth=1.65,
                    label=LABELS[arm_name])

    lower = (
        ("Represented-area retention", "% of t = 0 RDF area", (0.0, 104.0)),
        ("Physical volume-centroid error", "error (cm)", None),
        ("Over-capacity volume", "% of total liquid mass", None),
    )
    for ax, (title, ylabel, ylim) in zip(metric_axes, lower):
        ax.set_title(title, pad=7)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.0, duration)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(bottom=0.0)
        ax.grid(True)
    metric_axes[0].legend(loc="lower left", frameon=True, fontsize=8.2)

    fig.text(0.5, 0.022,
             "Dashed circle is the analytic z = 0.025 m section (r = 0.29896 m). "
             "The t = 0 RDF begins from eight-sample-per-voxel VOF data, so finite sampling error is already present.",
             ha="center", va="bottom", fontsize=8.3, color="#555e66")
    PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG, dpi=210, facecolor=fig.get_facecolor())
    fig.savefig(SVG, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
