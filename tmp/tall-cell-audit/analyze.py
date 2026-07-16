#!/usr/bin/env python3
"""Summarize a smoke-runner JSONL trace into the Phase A3 metric table."""
import json
import sys
from collections import Counter, defaultdict


def load(path):
    records = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def analyze(path):
    records = load(path)
    running = defaultdict(list)
    for r in records:
        if r.get("phase") == "running":
            running[r["method"]].append(r)

    for method, steps in running.items():
        print(f"\n=== {method} ({len(steps)} sampled steps) ===")
        first = {}
        peak = defaultdict(float)
        regions = Counter()
        for r in steps:
            pre, post = r.get("preProjectionVelocity"), r.get("postProjectionVelocity")
            if not pre or not post:
                continue
            ratio = post["kineticEnergyProxy"] / max(pre["kineticEnergyProxy"], 1e-30)
            cfl = post["maximumComponentCfl"]
            speed = post["liquidMaximum"]
            drift = abs(r.get("exactVolumeDrift") or 0)
            dominant = r.get("dominantComponentFraction", 1)
            checks = [
                ("keRatio>1.1", ratio > 1.1),
                ("keRatio>2", ratio > 2),
                ("cfl>1", cfl > 1),
                ("speed>5", speed > 5),
                ("drift>1e-3", drift > 1e-3),
                ("dominant<0.995", dominant < 0.995),
                ("components>1", (r.get("componentCount") or 1) > 1),
            ]
            for name, hit in checks:
                if hit and name not in first:
                    first[name] = (r["steps"], r.get("simulatedTime_s"))
            peak["keRatio"] = max(peak["keRatio"], ratio)
            peak["cfl"] = max(peak["cfl"], cfl)
            peak["speed"] = max(peak["speed"], speed)
            peak["drift"] = max(peak["drift"], drift)
            peak["postDivMax"] = max(peak["postDivMax"], post["maximumLiquidDivergence_s"])
            peak["postDivRms"] = max(peak["postDivRms"], post["rmsLiquidDivergence_s"])
            peak["residual"] = max(peak["residual"], r.get("pressureRelativeResidual") or 0)
            for name, info in (r.get("extrema") or {}).items():
                if info:
                    regions[f"{name}:{info['region']}"] += 1
        print("first exceedances:")
        for name, (step, t) in sorted(first.items(), key=lambda kv: kv[1][0]):
            print(f"  {name:<16} step {step:>4}  t={t:.3f}s")
        if not first:
            print("  none")
        print("peaks:", {k: round(v, 4) for k, v in peak.items()})
        if regions:
            print("extremum regions (top 12):")
            for key, count in regions.most_common(12):
                print(f"  {key:<44} {count}")

    print("\n=== checkpoint comparisons ===")
    for r in records:
        if r.get("phase") == "checkpoint-comparison":
            m = r["metrics"]
            print(f"  {r['left']} vs {r['right']} t={r['time_s']:.2f}s  IoU={m['wetIntersectionOverUnion']:.4f}  centroid={m['centroidDistanceCells']:.2f}c  MAE={m['meanAbsoluteError']:.5f}  volRel={m['volumeRelativeDifference']:.2e}  comps L/R={r['leftComponentCount']}/{r['rightComponentCount']}")

    print("\n=== results ===")
    for r in records:
        if r.get("phase") == "result":
            print(f"  {r['method']}: gridKind={r['gridKind']} steps={r['steps']} front={r.get('front_m'):.3f} volDrift={r.get('volumeDrift'):.2e} "
                  f"maxSpeed={r.get('maxSpeed_m_s'):.2f} residual={r.get('pressureRelativeResidual')} flags={r.get('stabilityFlags')} "
                  f"gaps={r.get('finalTallVolumeGaps')}")
            env = r.get("stabilityEnvelope")
            if env:
                print(f"    envelope: peakSpeed={env['peakLiquidSpeed_m_s']:.2f} peakCFL={env['peakComponentCfl']:.3f} keRatio={env['maximumProjectionEnergyRatio']:.3f} "
                      f"drift={env['maximumExactVolumeDrift']:.2e} dominant>={env['minimumDominantComponentFraction']:.4f} nonFinite={env['nonFiniteVelocityCount']}")
    for r in records:
        if r.get("phase") == "scenario-complete":
            print("  passedInvariants:", r["passedInvariants"])


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(f"\n########## {path}")
        analyze(path)
