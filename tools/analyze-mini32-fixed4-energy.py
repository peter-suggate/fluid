"""Summarize the native min4/max4 dam-break stage captures (stdlib only)."""
import json
import math
from pathlib import Path

root = Path("artifacts/mini32-fixed4-energy")
summary = {}
for name in ["base", "half-dt", "quarter-dt", "conditioning-off", "tight-pressure", "closed-wall-guard", "closed-wall-step85"]:
    path = root / name / "trace.json"
    if not path.exists():
        continue
    config = root / name / "config.json"
    if config.exists() and json.loads(config.read_text()).get("invalidExperiment"):
        continue
    rows = json.loads(path.read_text())
    energies = [r["stages"][-1]["total"] if r.get("stages") else r["total"] for r in rows]
    selected = [i for i in range(1, len(rows)) if 2 - 1e-7 <= rows[i]["time"] <= 3 + 1e-7]
    at3 = min(range(len(rows)), key=lambda i: abs(rows[i]["time"] - 3))
    summary[name] = {
        "last_time_s": rows[-1]["time"],
        "initial_energy_per_density": energies[0],
        "first_frame_gain": energies[1] - energies[0],
        "peak_energy_per_density": max(energies),
        "peak_time_s": rows[energies.index(max(energies))]["time"],
        "positive_increments_2_to_3_s": sum(max(0, energies[i] - energies[i - 1]) for i in selected),
        "largest_increment_2_to_3_s": max((energies[i] - energies[i - 1] for i in selected), default=0),
        "energy_nearest_3s": energies[at3],
        "mass_drift_nearest_3s_percent": 100 * (rows[at3]["mass"] / rows[0]["mass"] - 1),
    }

baseline = json.loads((root / "boundary-audit/trace.json").read_text())
frame = baseline[85]
summary["frame85_stage_deltas"] = [
    {"stage": after["stage"], "delta_potential": after["potential"] - before["potential"],
     "delta_kinetic": after["kinetic"] - before["kinetic"], "delta_total": after["total"] - before["total"]}
    for before, after in zip(frame["stages"], frame["stages"][1:])
]
summary["frame85_cells"] = {}
for stage in [frame["stages"][0], frame["stages"][-2], frame["stages"][-1]]:
    summary["frame85_cells"][stage["stage"]] = [
        dict(c, speed=math.sqrt(sum(v*v for v in c["velocity"])))
        for c in stage["cellSamples"] if c["id"] in [9, 16, 132, 256, 405]
    ]
checkpoint = root / "closed-wall-step85/trace.json"
if checkpoint.exists() and not json.loads((checkpoint.parent / "config.json").read_text()).get("invalidExperiment"):
    patched = json.loads(checkpoint.read_text())
    if len(patched) > 85:
        assert all(baseline[i]["total"] == patched[i]["total"] for i in range(85)), "checkpoint histories must match exactly"
        b, p = baseline[85], patched[85]
        for s, t in zip(b["stages"][:-1], p["stages"][:-1]):
            assert s["total"] == t["total"], "all stages before pressure must match exactly"
        summary["checkpoint85"] = {
            "identical_through_frame": 84,
            "identical_through_stage": "body-forces",
            "baseline_delta_energy": b["stages"][-1]["total"] - b["stages"][0]["total"],
            "patched_delta_energy": p["stages"][-1]["total"] - p["stages"][0]["total"],
            "baseline_pressure_delta": b["stages"][-1]["kinetic"] - b["stages"][-2]["kinetic"],
            "patched_pressure_delta": p["stages"][-1]["kinetic"] - p["stages"][-2]["kinetic"],
            "patched_cells": [dict(c, speed=math.sqrt(sum(v*v for v in c["velocity"])))
                              for c in p["stages"][-1]["cellSamples"] if c["id"] in [9,16,132,256,405]],
        }
minus_path, plus_path = root / "threshold-minus/trace.json", root / "threshold-plus/trace.json"
if minus_path.exists() and plus_path.exists():
    minus, plus = json.loads(minus_path.read_text()), json.loads(plus_path.read_text())
    if len(minus) > 85 and len(plus) > 85:
        assert all(minus[i]["total"] == plus[i]["total"] for i in range(85))
        for m, p in zip(minus[85]["stages"][:5], plus[85]["stages"][:5]):
            assert m == p, "same stage state until the single scalar perturbation"
        for mstage, pstage in zip(minus[85]["stages"][5:7], plus[85]["stages"][5:7]):
            assert all(a["velocity"] == b["velocity"] for a,b in zip(mstage["cellSamples"],pstage["cellSamples"]))
            assert sum(a["density"] != b["density"] for a,b in zip(mstage["cellSamples"],pstage["cellSamples"])) == 1
        m = next(c for c in minus[85]["stages"][-1]["cellSamples"] if c["id"] == 16)
        p = next(c for c in plus[85]["stages"][-1]["cellSamples"] if c["id"] == 16)
        summary["epsilon_crossing"] = {
            "minus": dict(m, speed=math.sqrt(sum(v*v for v in m["velocity"]))),
            "plus": dict(p, speed=math.sqrt(sum(v*v for v in p["velocity"]))),
            "density_difference": p["density"] - m["density"],
            "velocity_vector_difference": math.sqrt(sum((a-b)**2 for a,b in zip(m["velocity"],p["velocity"]))),
            "domain_energy_difference": plus[85]["stages"][-1]["total"] - minus[85]["stages"][-1]["total"],
        }
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
for name, result in summary.items():
    if name not in ["frame85_stage_deltas", "frame85_cells"]:
        print(name, json.dumps(result))
