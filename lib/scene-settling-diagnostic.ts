import type { DeepReadonly, SceneDiagnosticEvidence, SceneDiagnosticHookFinding } from "./scene-diagnostics";
import type { SceneDescription } from "./model";
import {
  hookFinding,
  numberPath,
  recordPath,
  selectedMethodDiagnostics,
} from "./scene-hook-evidence";

export interface SettlingDiagnosticParameters {
  maximumFinalExactVolumeDrift: number;
  maximumNormalizedNetProjectionEnergyDelta: number;
  maximumNormalizedLateMechanicalEnergySlopePerSecond: number;
  maximumLateToMiddleKineticEnvelopeRatio: number;
  maximumDamBreakDriftSignChanges: number;
  maximumDamBreakLatePeakToPeakDrift: number;
}

export function evaluateSettlingDiagnostic(input: {
  scene: DeepReadonly<SceneDescription>;
  evidence: SceneDiagnosticEvidence;
  parameters: SettlingDiagnosticParameters;
  methods?: readonly string[];
}): readonly SceneDiagnosticHookFinding[] {
  const findings: SceneDiagnosticHookFinding[] = [];
  for (const [method, diagnostics] of selectedMethodDiagnostics(input.evidence, input.methods)) {
    const energy = recordPath(diagnostics, "energy", "summary")
      ?? recordPath(diagnostics, "energyTraceSummary");
    findings.push(hookFinding({
      id: `${method}.energy-present`, method, passed: energy !== undefined,
      message: energy ? "mechanical-energy summary is available" : "method did not produce a mechanical-energy summary",
    }));
    if (!energy) continue;
    const checks = [
      ["final-volume-drift", "finalSampledExactVolumeDrift", input.parameters.maximumFinalExactVolumeDrift,
        "final sampled exact-volume drift"],
      ["projection-energy", "normalizedNetProjectionEnergyDelta", input.parameters.maximumNormalizedNetProjectionEnergyDelta,
        "normalized net projection-energy delta"],
      ["late-energy-slope", "normalizedLateMechanicalEnergySlopePerSecond",
        input.parameters.maximumNormalizedLateMechanicalEnergySlopePerSecond, "normalized late mechanical-energy slope"],
      ["kinetic-envelope", "lateToMiddleKineticEnvelopeRatio",
        input.parameters.maximumLateToMiddleKineticEnvelopeRatio, "late-to-middle kinetic envelope ratio"],
    ] as const;
    for (const [id, key, maximum, label] of checks) {
      const actual = numberPath(energy, key);
      findings.push(hookFinding({
        id: `${method}.${id}`, method, passed: actual !== undefined && actual <= maximum,
        message: actual !== undefined && actual <= maximum
          ? `${label} stayed within its settling envelope`
          : `${label} ${actual ?? "unknown"} exceeds ${maximum}`,
        expected: { maximum }, actual,
      }));
    }
    if (input.scene.fluid.initialCondition !== "dam-break") continue;
    const signChanges = numberPath(energy, "driftSignChanges");
    const peakToPeak = numberPath(energy, "latePeakToPeakDrift");
    findings.push(hookFinding({
      id: `${method}.drift-sign-changes`, method,
      passed: signChanges !== undefined && signChanges <= input.parameters.maximumDamBreakDriftSignChanges,
      message: signChanges !== undefined && signChanges <= input.parameters.maximumDamBreakDriftSignChanges
        ? "late volume drift did not oscillate excessively"
        : `late volume drift changed direction ${signChanges ?? "unknown"} times`,
      expected: { maximum: input.parameters.maximumDamBreakDriftSignChanges }, actual: signChanges,
    }));
    findings.push(hookFinding({
      id: `${method}.late-drift-range`, method,
      passed: peakToPeak !== undefined && peakToPeak <= input.parameters.maximumDamBreakLatePeakToPeakDrift,
      message: peakToPeak !== undefined && peakToPeak <= input.parameters.maximumDamBreakLatePeakToPeakDrift
        ? "late peak-to-peak volume drift stayed within its envelope"
        : `late peak-to-peak volume drift ${peakToPeak ?? "unknown"} exceeds ${input.parameters.maximumDamBreakLatePeakToPeakDrift}`,
      expected: { maximum: input.parameters.maximumDamBreakLatePeakToPeakDrift }, actual: peakToPeak,
    }));
  }
  return findings;
}
