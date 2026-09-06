"""Fit wave damping and frequency; retain fit residuals and raw physical gauges."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

paths = [Path(p) for p in sys.argv[1:]]
if not paths:
    paths = sorted(Path("artifacts/sparse-gravity-wave").glob("fixed*-x"))
output = Path("artifacts/sparse-gravity-wave/report")
output.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
results = []
for path in paths:
    cfg = json.loads((path / "config.json").read_text())
    trace = json.loads((path / "trace.json").read_text())
    t = np.array([r["time"] for r in trace])
    a = np.array([r["mode"] for r in trace])
    T = cfg["period"]
    tau = t / T
    scale = max(abs(a[0]), cfg["amplitude"], 1e-9)
    y = a / scale
    def predict(p):
        damping, frequency, cosine, sine, offset = p
        return np.exp(-damping * tau) * (cosine * np.cos(2 * np.pi * frequency * tau)
                + sine * np.sin(2 * np.pi * frequency * tau)) + offset
    fit = min((least_squares(lambda p: predict(p) - y, [0.1, f, y[0], 0, 0],
                bounds=([-1, .05, -4, -4, -4], [20, 4, 4, 4, 4]))
                for f in [.5, 1, 1.5, 2]), key=lambda r: np.sum(r.fun ** 2))
    damping, frequency, cosine, sine, offset = fit.x
    relative_fit_rms = float(np.sqrt(np.mean(fit.fun ** 2)))
    dynamic_range = float(np.ptp(y))
    reliable = bool(t[-1] >= 2 * T and relative_fit_rms < .15 and dynamic_range > .25
                    and .2 < frequency < 3.9)
    masses = np.array([r["mass"] for r in trace])
    timings = json.loads((path / "timings.json").read_text())["timings"] if (path / "timings.json").exists() else []
    label = path.name
    row = {"capture": str(path), "duration_s": float(t[-1]), "period_analytic_s": T,
           "initial_mode_m": float(a[0]), "initial_representation_ratio": float(a[0] / cfg["amplitude"]) if cfg["amplitude"] else None,
           "final_mode_m": float(a[-1]), "fit_reliable": reliable,
           "fit_rms_over_initial_amplitude": relative_fit_rms,
           "fitted_frequency_over_analytic": float(frequency),
           "fitted_amplitude_retained_per_analytic_period": float(np.exp(-damping)),
           "fitted_offset_m": float(offset * scale),
           "mass_drift_percent": float(100 * (masses[-1] / masses[0] - 1)),
           "max_speed_m_s": max(r["maxSpeed"] for r in trace),
           "max_other_mode_rms_m": max(r["residualRms"] for r in trace),
           "initial_cells": trace[0]["cells"], "final_cells": trace[-1]["cells"],
           "median_step_wall_ms": float(np.median(timings[3:])) if len(timings) > 3 else None,
           "shader_hashes": cfg["shaderHashes"]}
    results.append(row)
    axes[0, 0].plot(tau, y, label=label)
    if reliable:
        axes[0, 1].plot(tau, predict(fit.x), label=label)
    axes[1, 0].plot(tau, np.array([r["residualRms"] for r in trace]) * 1000, label=label)
    axes[1, 1].plot(tau, 100 * (masses / masses[0] - 1), label=label)
    print(json.dumps({k: v for k, v in row.items() if k != "shader_hashes"}))
axes[0, 0].plot(np.linspace(0, 3, 500), np.cos(2 * np.pi * np.linspace(0, 3, 500)), "k--", alpha=.35, label="Linear inviscid wave")
for ax, title, ylabel in zip(axes.flat,
        ["Accepted column-mass wave mode", "Reliable damped-oscillation fits", "Surface variation outside the initial mode", "Liquid-volume drift"],
        ["Mode / initial amplitude", "Mode / initial amplitude", "RMS (mm)", "Change (%)"]):
    ax.set(title=title, xlabel="Time / analytical period", ylabel=ylabel)
    ax.grid(alpha=.2)
    if ax.lines: ax.legend(fontsize=7)
fig.savefig(output / "waves.png", dpi=160)
(output / "comparison.json").write_text(json.dumps(results, indent=2))
