"""Summarize the October 3 coarse-first experiments and plot matched states.

Usage: python tools/analyze-uniform-coarse-first.py [capture directory]
Requires matplotlib for the diagnostic figure. Raw captures remain untouched.
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

root = Path(__file__).resolve().parents[1]
captures = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "artifacts/uniform-breakthrough-2026-10-03"
out = root / "docs/plans/uniform-coarse-first-2026-10-03"
out.mkdir(parents=True, exist_ok=True)
rows = []
documents = {}
for path in sorted(captures.glob("*.json")):
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or "sceneId" not in data:
        continue
    documents[path.stem] = data
    row = {"capture": path.name, **{key: data.get(key) for key in (
        "capturedAt", "sourceFingerprint", "sourceFingerprintAfter", "sceneId", "kind", "lattice", "adapter", "frames",
        "dt_s", "discardFrames", "throughput", "scope", "surfaceTolerance", "bandTarget", "coarseCadence", "focusBox", "focusTravel",
        "elideEmptyBand", "failure", "summary", "values", "validationErrors")}}
    row["allocationBytes"] = data.get("initial", {}).get("allocatedBytes")
    row["final"] = {k: v for k, v in data.get("final", {}).items()
                    if k.startswith(("uniformMixed", "uniformPressure", "volume", "maxSpeed", "encodedSteps"))}
    row["quality"] = [{k: v for k, v in q.items() if k not in ("projection", "phiSlice")}
                      for q in data.get("qualitySnapshots", [])]
    if data.get("rows"):
        work = data["rows"]
        row["meanFineTiles"] = sum(r["work"].get("uniformMixedFineTiles", 0) for r in work) / len(work)
        row["peakSpeed_m_s"] = max(r["work"].get("maxSpeed_m_s", 0) for r in work)
        row["maxBandResidual"] = max(r["work"].get("uniformPressureBandResidual", 0) for r in work)
    rows.append(row)
comparisons = []
for prefix, baseline in [("fig9", "fig9-baseline-trace"), ("dam128", "dam128-base"), ("drop256", "drop256-base"), ("pond", "pond-base")]:
    if baseline not in documents:
        continue
    base = documents[baseline]
    base_quality = {q["frame"]: q for q in base.get("qualitySnapshots", [])}
    for name, data in documents.items():
        if not name.startswith(prefix) or name == baseline or data.get("failure"):
            continue
        pairs = []
        for q in data.get("qualitySnapshots", []):
            if q["frame"] not in base_quality:
                continue
            b = base_quality[q["frame"]]
            pairs.append({"frame": q["frame"], "time_s": q["time_s"],
                          "centroidDelta_h": [x - y for x, y in zip(q["centroid_cells"], b["centroid_cells"])],
                          "massFront99Delta_h": q["massFront_cells"]["p99"] - b["massFront_cells"]["p99"],
                          "massDeltaRelative": q["mass"] / b["mass"] - 1,
                          "excessFraction": q["excess"] / q["mass"],
                          "massInPhiAirFraction": q["massInPhiAir"] / q["mass"],
                          "sampledQualityExactlyEqual": all(q.get(k) == b.get(k) for k in
                              ("mass", "excess", "negative", "maxVolume", "centroid_cells", "projection", "phiSlice"))})
        if pairs:
            comparisons.append({"baseline": baseline, "candidate": name, "samples": pairs})
report = {"scope": "Exploratory simulation-only captures. Changed numerical trajectories are not quality-approved optimizations. No browser FPS claim.",
          "captures": rows, "matchedQuality": comparisons}
(out / "evidence.json").write_text(json.dumps(report, indent=2) + "\n")

names = ["fig9-baseline-trace", "fig9-coarse-trace", "fig9-focus-trace", "fig9-moving-focus-final"]
titles = ["Current dynamic policy", "4h everywhere", "4h + fixed h region", "4h + moving h region"]
fig, axes = plt.subplots(2, 4, figsize=(14, 7.6), sharex=True, sharey=True, layout="constrained")
for col, (name, title) in enumerate(zip(names, titles)):
    if name not in documents:
        continue
    d = documents[name]
    nx, ny = d["lattice"]["nx"], d["lattice"]["ny"]
    for row, frame in enumerate([60, 120]):
        q = next(q for q in d["qualitySnapshots"] if q["frame"] == frame)
        ax = axes[row, col]
        im = ax.imshow(np.array(q["projection"]).reshape(ny, nx), origin="lower", extent=(0, nx, 0, ny),
                       vmin=0, vmax=1, cmap="Blues", interpolation="nearest")
        ax.contour(np.arange(nx) + .5, np.arange(ny) + .5, np.array(q["phiSlice"]).reshape(ny, nx),
                   levels=[0], colors=["#df6500"], linewidths=.65)
        if d["kind"] == "focus":
            box = d["focusBox"]
            # Final moving capture uses the corrected trace-horizon policy.
            shift = d.get("focusTravel", 0) * (frame - 1) / (d["frames"] - 1)
            ax.add_patch(Rectangle(((box[0] + shift) * nx, box[1] * ny), (box[3] - box[0]) * nx,
                                   (box[4] - box[1]) * ny, fill=False, edgecolor="#9b288f", linewidth=1.4, linestyle="--"))
        ax.set_title(f"{title}\nt = {q['time_s']:.1f} s · h tiles {q['fineTiles']:,}", fontsize=10)
        ax.set_aspect("equal")
        if row == 1:
            ax.set_xlabel("x / h")
        if col == 0:
            ax.set_ylabel("y / h")
fig.colorbar(im, ax=axes, shrink=.7, label="Depth-mean conserved volume (colour capped at 1)")
fig.suptitle("Coarse-first selection changes the flow: matched physical times, Figure 9\n"
             "Blue: mass projection · orange: centre-plane phi = 0 · purple: requested fine region\n"
             "Diagnostic fields, not production renders or a visual acceptance result", fontsize=12)
fig.savefig(out / "comparison.png", dpi=160, bbox_inches="tight")
print(out / "evidence.json")
print(out / "comparison.png")
