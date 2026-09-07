"""Compare two measured frozen A/B experiments without modifying simulation data.

Usage: python compare-pool-impact-ab-runs.py BEFORE_ANALYSIS AFTER_ANALYSIS OUTPUT
Dependencies: NumPy and Matplotlib, as for analyze-pool-impact-resolution-ab.py.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

before_path, after_path, output = map(Path, sys.argv[1:4])
output.mkdir(parents=True, exist_ok=True)
runs = [json.loads(p.read_text()) for p in [before_path, after_path]]
for arm in range(2):
    a, b = [run["configurations"][arm] for run in runs]
    for key in ["scene", "values", "grid", "steps", "dt", "maxCell", "freezeTopology"]:
        assert a[key] == b[key], (arm, key, "comparison must retain the experiment")

limits = [("density", "maximum", .01), ("density", "mean", .001),
          ("velocity", "maximum", .02), ("velocity", "mean", .001)]
summary = {}
for name, run in zip(["before", "after"], runs):
    arms = run["arms"]
    summary[name] = {
        "first_symmetry_failure_step": [
            {f"{kind}_{field}": next((row["step"] for row in arm
                                      if max(v[field] for v in row["symmetry"][kind]) > limit), None)
             for kind, field, limit in limits} for arm in arms],
        "checkpoints": [{"time_s": arms[0][i]["time"],
                         "kinetic_J": [arm[i]["kinetic_J"] for arm in arms],
                         "mechanical_excess_J": [arm[i]["mechanical_excess_J"] for arm in arms],
                         "mass_error_percent": [arm[i]["mass_error_percent"] for arm in arms],
                         "density_symmetry_max": [max(x["maximum"] for x in arm[i]["symmetry"]["density"]) for arm in arms],
                         "velocity_symmetry_max_m_s": [max(x["maximum"] for x in arm[i]["symmetry"]["velocity"]) for arm in arms],
                         "comparison": run["comparison"][i]}
                        for i in [15, 30, 60, 120, 240] if i < len(arms[0])],
    }
(output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)
colors = ["#a75a39", "#187b80"]
for label, run, color in zip(["Before", "After"], runs, colors):
    time = [r["time"] for r in run["comparison"]]
    axes[0, 0].plot(time, [r["column_rms_difference_m"] * 1000 for r in run["comparison"]], color=color, label=label)
    for arm, style, suffix in [(0, "--", "A max1"), (1, "-", "B min1/max2")]:
        axes[0, 1].plot(time, [r["kinetic_J"] for r in run["arms"][arm]], color=color, linestyle=style, label=f"{label}: {suffix}")
    for ax, kind, field in [(axes[1, 0], "velocity", "maximum"), (axes[1, 1], "density", "mean")]:
        values = [max(m[field] for m in r["symmetry"][kind]) for r in run["arms"][1]]
        ax.semilogy(time[1:], np.maximum(values[1:], 1e-8), color=color, label=label)
for ax, title, ylabel in [(axes[0, 0], "A/B column-depth disagreement", "RMS difference (mm)"),
                          (axes[0, 1], "Kinetic energy — both arms", "Energy (J)"),
                          (axes[1, 0], "B: maximum velocity symmetry error", "Error (m/s)"),
                          (axes[1, 1], "B: mean density symmetry error", "Density error")]:
    ax.set(title=title, xlabel="Time (s)", ylabel=ylabel, xlim=(0, time[-1]))
    ax.grid(alpha=.2); ax.legend(fontsize=8)
axes[1, 0].axhline(.02, color="#555555", linestyle=":", linewidth=1)
axes[1, 1].axhline(.001, color="#555555", linestyle=":", linewidth=1)
fig.suptitle("Frozen topology: geometric transport and physical subfaces\nDotted lines retain the existing symmetry limits", fontsize=13)
fig.savefig(output / "before-after.png", dpi=180)
