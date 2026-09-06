"""Causal controls for the Sparse CM12 gravity-wave investigation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path("artifacts/sparse-gravity-wave")
output = root / "report"
output.mkdir(parents=True, exist_ok=True)


def read(name, file):
    return json.loads((root / name / file).read_text())


def response(name):
    cfg = read(name, "config.json")
    stages = read(name, "1-stages.json")
    projected = next(r for r in stages if r["stage"] == "velocity-projection")
    predicted = -cfg["omega"] ** 2 * cfg["amplitude"] * cfg["dt"]
    return {"capture": name, "height_m": cfg["height"], "amplitude_m": cfg["amplitude"],
            "theta_min": cfg.get("thetaMin", .05), "measured_mode_rate_m_s": projected["modeRate"],
            "predicted_mode_rate_m_s": predicted, "response_ratio": projected["modeRate"] / predicted}


fractions = [{"fill": (i + 1) / 8,
              "production": response(f"height-{i}-production"),
              "column": response(f"height-{i}-column")} for i in range(7)]
amplitudes = [response(n) for n in ["height-3-production", "half-fill-small", "half-fill-smaller", "half-fill-tiny"]]
clamps = [response(n) for n in ["half-fill-tiny", "theta-0", "theta-1", "theta-2", "theta-3"]]
combined = response("half-fill-geometric")

region = {"same_topology": read("fixed4-x", "config.json")["initialTopology"] == read("split4-x", "config.json")["initialTopology"],
          "same_shaders": read("fixed4-x", "config.json")["shaderHashes"] == read("split4-x", "config.json")["shaderHashes"],
          "same_initial_fields": {name: (root / "fixed4-x" / f"0-{name}.bin").read_bytes()
                                  == (root / "split4-x" / f"0-{name}.bin").read_bytes()
                                  for name in ["density", "velocity", "gamma", "pressure", "divergence", "columns"]},
          "whole_region": response("fixed4-x"), "two_identical_halves": response("split4-x")}
assert region["same_topology"] and region["same_shaders"] and all(region["same_initial_fields"].values())
region["replays"] = []
for step in [70, 280]:
    reference = read("fixed4-x", f"{step}-stages.json")
    changed = read(f"replay{step}", f"{step}-stages.json")
    equal_before_pressure = all(a == b for a, b in zip(reference, changed) if a["stage"] not in ["velocity-projection", "candidate-transfer"])
    assert equal_before_pressure
    region["replays"].append({"step": step, "equal_stage_metrics_before_pressure": equal_before_pressure,
                              "reference_projection": reference[-2], "changed_projection": changed[-2]})

checkpoints = []
for step in [70, 280]:
    fine = read(f"checkpoint{step}-width1", f"{step}-stages.json")
    coarse = read(f"checkpoint{step}-width4", f"{step}-stages.json")
    assert all(a == b for a, b in zip(fine[:-1], coarse[:-1])), "matched fine state through pressure projection"
    row = {"step": step, "equal_before_transfer": True,
           "transfer_mode_difference_m": coarse[-1]["mode"] - fine[-1]["mode"],
           "transfer_mode_rate_ratio": coarse[-1]["modeRate"] / fine[-1]["modeRate"], "next_step": {}}
    for width in [1, 4]:
        states = {r["stage"]: r for r in read(f"checkpoint{step}-width{width}", f"{step+1}-stages.json")}
        row["next_step"][str(width)] = {
            "face_preparation_mode_rate_change": states["face-preparation"]["modeRate"] - states["transport-velocity-extension"]["modeRate"],
            "pressure_mode_rate_change": states["velocity-projection"]["modeRate"] - states["body-forces"]["modeRate"],
            "scalar_mode_change": states["surface-sharpening"]["mode"] - states["face-preparation"]["mode"]}
    checkpoints.append(row)

oscillating = read("oscillate-default", "trace.json")
fixed = {r["step"]: r for r in read("fixed2-x", "trace.json")}
assert sum(r["changed"] for r in oscillating) == 118
for step in [0, 2]:
    assert oscillating[step]["mode"] == fixed[step]["mode"] and oscillating[step]["mass"] == fixed[step]["mass"]
assert oscillating[-1]["cells"] == fixed[120]["cells"]
deltas = []
for step in [3, 4, 59, 60, 119, 120]:
    stages = {r["stage"]: r for r in read("oscillate-default", f"{step}-stages.json")}
    before, after = stages["velocity-projection"], stages["candidate-transfer"]
    deltas.append({"step": step, "mode_change_m": after["mode"] - before["mode"],
                   "mass_change_m3": after["mass"] - before["mass"],
                   "mode_rate_change_m_s": after["modeRate"] - before["modeRate"]})
differences = np.array([r["mode"] - fixed[r["step"]]["mode"] for r in oscillating if r["step"] % 2 == 0])
topology = {"actual_changes": 118, "same_final_grid": True, "sampled_transfers": deltas,
            "same_grid_mode_rms_difference_m": float(np.sqrt(np.mean(differences ** 2))),
            "same_grid_mode_max_difference_m": float(np.max(abs(differences))),
            "max_pressure_residual": max(r["pressureResidual"] for r in oscillating)}

fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
for key, label in [("production", "Production boundary"), ("column", "Column-distance diagnostic")]:
    axes[0].plot([r["fill"] for r in fractions], [r[key]["response_ratio"] for r in fractions], "o-", label=label)
axes[0].axhline(1, color="black", linestyle="--", alpha=.5)
axes[0].set(xlabel="Fraction filled in surface cell", ylabel="Response / linear prediction", title="Changing only resting water height")
axes[0].legend(fontsize=8)
axes[1].loglog([r["amplitude_m"] * 1000 for r in amplitudes], [abs(r["measured_mode_rate_m_s"]) for r in amplitudes], "o-", label="Measured")
axes[1].loglog([r["amplitude_m"] * 1000 for r in amplitudes], [abs(r["predicted_mode_rate_m_s"]) for r in amplitudes], "k--", label="Linear prediction")
axes[1].set(xlabel="Initial wave amplitude (mm)", ylabel="First-step wave-mode speed (m/s)", title="Half-filled cells: response fails to vanish")
axes[1].legend(fontsize=8)
axes[2].loglog([r["theta_min"] for r in clamps], [r["response_ratio"] for r in clamps], "o-", label="Density-derived distance")
axes[2].scatter([combined["theta_min"]], [combined["response_ratio"]], marker="*", s=120, label="Column distance + smaller clamp")
axes[2].axhline(1, color="black", linestyle="--", alpha=.5)
axes[2].set(xlabel="Minimum ghost distance / cell spacing", ylabel="Response / linear prediction", title="10 µm wave: isolate the distance clamp")
axes[2].legend(fontsize=8)
for ax in axes: ax.grid(alpha=.2)
fig.savefig(output / "pressure-causes.png", dpi=170)
report = {"fractional_heights": fractions, "amplitude_limit": amplitudes, "clamp_ablation": clamps,
          "combined_diagnostic": combined, "region_invariance": region, "matched_checkpoints": checkpoints,
          "topology_wave": topology}
(output / "causes.json").write_text(json.dumps(report, indent=2))
print(json.dumps({"region_invariance": {k: v for k, v in region.items() if k != "replays"},
                  "matched_checkpoints": checkpoints, "topology_wave": topology}, indent=2))
