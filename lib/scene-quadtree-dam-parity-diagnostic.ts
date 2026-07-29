import type { DeepReadonly, SceneDiagnosticEvidence, SceneDiagnosticHookFinding } from "./scene-diagnostics";
import type { SceneDescription } from "./model";
import {
  hookFinding,
  numberPath,
  recordPath,
  runSteps,
  runTime,
  selectedMethodDiagnostics,
} from "./scene-hook-evidence";

export interface QuadtreeDamParityDiagnosticParameters {
  method: string;
  minimumSimulatedTime_s: number;
  maximumNeighborRatio?: number;
  minimumLeafCount?: number;
  minimumGhostFaceCount?: number;
  maximumPressureRelativeResidual: number;
  maximumPressureAbsoluteResidual?: number;
  maximumFluidScale?: number;
  rebuildCadenceSteps: number;
  inlineRebuildCompletionFraction: number;
  maximumBlockedRebuildFrames: number;
  maximumWallToGpuRatio: number;
  requireStabilitySampleEveryStep: boolean;
  maximumNonFiniteVelocityCount: number;
  maximumPeakLiquidSpeed_m_s: number;
  maximumPeakComponentCfl: number;
  maximumProjectionEnergyRatio: number;
  maximumEnvelopePressureRelativeResidual: number;
  maximumExactVolumeDrift: number;
  maximumCompressionRatio: number;
  minimumDominantComponentFraction: number;
  maximumFinalComponentCount: number;
  minimumFront_m: number;
  kineticGateAfter_s: number;
  minimumPeakKineticEnergyProxy: number;
  referenceMethod?: string;
  minimumKineticEnergyRatioToReference?: number;
}

function scalarFinding(input: {
  id: string;
  method: string;
  value: number | undefined;
  operator: "minimum" | "maximum" | "equal";
  expected: number;
  label: string;
}): SceneDiagnosticHookFinding {
  const passed = input.value !== undefined && (input.operator === "minimum"
    ? input.value >= input.expected
    : input.operator === "maximum" ? input.value <= input.expected : input.value === input.expected);
  return hookFinding({
    id: `${input.method}.${input.id}`, method: input.method, passed,
    message: passed ? `${input.label} passed`
      : `${input.label} ${input.value ?? "unknown"} failed declared ${input.operator} ${input.expected}`,
    expected: { [input.operator]: input.expected }, actual: input.value,
  });
}

export function evaluateQuadtreeDamParityDiagnostic(input: {
  scene: DeepReadonly<SceneDescription>;
  evidence: SceneDiagnosticEvidence;
  parameters: QuadtreeDamParityDiagnosticParameters;
}): readonly SceneDiagnosticHookFinding[] {
  const findings: SceneDiagnosticHookFinding[] = [];
  const entries = new Map(selectedMethodDiagnostics(input.evidence));
  const diagnostics = entries.get(input.parameters.method);
  if (!diagnostics) return [hookFinding({
    id: `${input.parameters.method}.evidence`, method: input.parameters.method, passed: false,
    message: `${input.parameters.method} did not publish quadtree dam-parity evidence`,
  })];
  const method = input.parameters.method;
  const solver = recordPath(diagnostics, "solver") ?? diagnostics;
  const run = recordPath(diagnostics, "run") ?? diagnostics;
  const stability = recordPath(diagnostics, "stability") ?? recordPath(diagnostics, "stabilityEnvelope");
  const finalSummary = recordPath(diagnostics, "field", "final", "summary")
    ?? recordPath(diagnostics, "finalSummary") ?? recordPath(diagnostics, "volumeFieldStats");

  findings.push(scalarFinding({ id: "simulated-time", method, value: runTime(diagnostics), operator: "minimum",
    expected: input.parameters.minimumSimulatedTime_s, label: "quadtree simulated time" }));
  if (input.parameters.maximumNeighborRatio !== undefined) findings.push(scalarFinding({ id: "neighbor-ratio", method, value: numberPath(solver, "quadtreeMaximumNeighborRatio"),
    operator: "maximum", expected: input.parameters.maximumNeighborRatio, label: "quadtree neighbor ratio" }));
  if (input.parameters.minimumLeafCount !== undefined) findings.push(scalarFinding({ id: "leaf-count", method, value: numberPath(solver, "quadtreeLeafCount"),
    operator: "minimum", expected: input.parameters.minimumLeafCount, label: "quadtree leaf count" }));
  if (input.parameters.minimumGhostFaceCount !== undefined) findings.push(scalarFinding({ id: "ghost-faces", method, value: numberPath(solver, "quadtreeGhostFaceCount"),
    operator: "minimum", expected: input.parameters.minimumGhostFaceCount, label: "quadtree corrected ghost-face count" }));
  const relativeResidual = numberPath(solver, "pressureRelativeResidual");
  const absoluteResidual = numberPath(solver, "pressureResidual");
  const residualPassed = (relativeResidual !== undefined
    && relativeResidual <= input.parameters.maximumPressureRelativeResidual)
    || (input.parameters.maximumPressureAbsoluteResidual !== undefined
      && absoluteResidual !== undefined && absoluteResidual <= input.parameters.maximumPressureAbsoluteResidual);
  findings.push(hookFinding({
    id: `${method}.terminal-pressure-residual`, method, passed: residualPassed,
    message: residualPassed ? "quadtree terminal pressure residual passed"
      : `quadtree terminal pressure residual relative=${relativeResidual ?? "unknown"} absolute=${absoluteResidual ?? "unknown"} exceeded both declared floors`,
    expected: { maximumRelative: input.parameters.maximumPressureRelativeResidual,
      maximumAbsolute: input.parameters.maximumPressureAbsoluteResidual },
    actual: { relative: relativeResidual, absolute: absoluteResidual },
  }));
  if (input.parameters.maximumFluidScale !== undefined) findings.push(scalarFinding({ id: "fluid-scale", method, value: numberPath(solver, "quadtreeMaximumFluidScale"),
    operator: "maximum", expected: input.parameters.maximumFluidScale, label: "quadtree free-surface scale" }));
  findings.push(scalarFinding({ id: "rebuild-cadence", method, value: numberPath(solver, "quadtreeRebuildCadenceSteps"),
    operator: "equal", expected: input.parameters.rebuildCadenceSteps, label: "quadtree rebuild cadence" }));

  const steps = runSteps(diagnostics);
  const staleLimit = numberPath(solver, "quadtreeTopologyStaleLimit");
  const completed = numberPath(solver, "quadtreeRebuildCompletedCount");
  const requiredRebuilds = steps !== undefined && staleLimit !== undefined
    ? staleLimit === 0 ? Math.ceil(input.parameters.inlineRebuildCompletionFraction * steps)
      : Math.floor((steps - 1) / Math.max(1, staleLimit + 1))
    : undefined;
  findings.push(hookFinding({
    id: `${method}.completed-rebuilds`, method,
    passed: completed !== undefined && requiredRebuilds !== undefined && completed >= requiredRebuilds,
    message: completed !== undefined && requiredRebuilds !== undefined && completed >= requiredRebuilds
      ? "quadtree completed the required topology rebuilds"
      : `quadtree completed ${completed ?? "unknown"} rebuilds; required ${requiredRebuilds ?? "unknown"}`,
    expected: { minimum: requiredRebuilds }, actual: completed,
  }));
  findings.push(scalarFinding({ id: "blocked-rebuilds", method,
    value: numberPath(solver, "quadtreeRebuildBlockedFrames"), operator: "maximum",
    expected: input.parameters.maximumBlockedRebuildFrames,
    label: "quadtree blocked rebuild frames" }));

  const simulationWall_ms = numberPath(run, "simulationWall_ms");
  const gpuPerStep_ms = numberPath(solver, "physicsTrace", "total_ms");
  const wallPerStep_ms = simulationWall_ms !== undefined && steps !== undefined
    ? simulationWall_ms / Math.max(1, steps) : undefined;
  const wallRatio = wallPerStep_ms !== undefined && gpuPerStep_ms !== undefined && gpuPerStep_ms > 0
    ? wallPerStep_ms / gpuPerStep_ms : undefined;
  findings.push(scalarFinding({ id: "wall-to-gpu", method, value: wallRatio, operator: "maximum",
    expected: input.parameters.maximumWallToGpuRatio, label: "quadtree wall/GPU step ratio" }));
  if (input.parameters.requireStabilitySampleEveryStep) findings.push(scalarFinding({
    id: "sampled-steps", method, value: numberPath(stability, "sampledSteps"),
    operator: "equal", expected: steps ?? Number.NaN, label: "quadtree sampled steps" }));
  findings.push(scalarFinding({ id: "nonfinite-velocity", method, value: numberPath(stability, "nonFiniteVelocityCount"),
    operator: "maximum", expected: input.parameters.maximumNonFiniteVelocityCount, label: "quadtree non-finite velocities" }));
  findings.push(scalarFinding({ id: "peak-speed", method, value: numberPath(stability, "peakLiquidSpeed_m_s"),
    operator: "maximum", expected: input.parameters.maximumPeakLiquidSpeed_m_s, label: "quadtree peak liquid speed" }));
  findings.push(scalarFinding({ id: "peak-cfl", method, value: numberPath(stability, "peakComponentCfl"),
    operator: "maximum", expected: input.parameters.maximumPeakComponentCfl, label: "quadtree peak component CFL" }));
  findings.push(scalarFinding({ id: "projection-energy", method, value: numberPath(stability, "maximumProjectionEnergyRatio"),
    operator: "maximum", expected: input.parameters.maximumProjectionEnergyRatio, label: "quadtree projection-energy ratio" }));
  findings.push(scalarFinding({ id: "envelope-pressure", method,
    value: numberPath(stability, "maximumPressureRelativeResidual"), operator: "maximum",
    expected: input.parameters.maximumEnvelopePressureRelativeResidual, label: "quadtree envelope pressure residual" }));
  findings.push(scalarFinding({ id: "volume-drift", method, value: numberPath(stability, "maximumExactVolumeDrift"),
    operator: "maximum", expected: input.parameters.maximumExactVolumeDrift, label: "quadtree exact-volume drift" }));
  findings.push(scalarFinding({ id: "compression", method, value: numberPath(solver, "compressionRatio"),
    operator: "maximum", expected: input.parameters.maximumCompressionRatio, label: "quadtree compression ratio" }));
  findings.push(scalarFinding({ id: "dominant-component", method,
    value: numberPath(stability, "minimumDominantComponentFraction"), operator: "minimum",
    expected: input.parameters.minimumDominantComponentFraction, label: "quadtree dominant-component fraction" }));
  findings.push(scalarFinding({ id: "final-components", method, value: numberPath(finalSummary, "componentCount"),
    operator: "maximum", expected: input.parameters.maximumFinalComponentCount, label: "quadtree final component count" }));
  findings.push(scalarFinding({ id: "front", method, value: numberPath(solver, "front_m"),
    operator: "minimum", expected: input.parameters.minimumFront_m, label: "quadtree dam front" }));

  const elapsed = runTime(diagnostics);
  if (elapsed !== undefined && elapsed >= input.parameters.kineticGateAfter_s - 1e-9) {
    const candidateKinetic = numberPath(stability, "peakKineticEnergyProxy");
    findings.push(scalarFinding({ id: "kinetic-activity", method, value: candidateKinetic,
      operator: "minimum", expected: input.parameters.minimumPeakKineticEnergyProxy,
      label: "quadtree peak kinetic-energy proxy" }));
    if (input.parameters.referenceMethod && input.parameters.minimumKineticEnergyRatioToReference !== undefined) {
      const reference = entries.get(input.parameters.referenceMethod);
      const referenceStability = reference
        ? recordPath(reference, "stability") ?? recordPath(reference, "stabilityEnvelope") : undefined;
      const referenceKinetic = numberPath(referenceStability, "peakKineticEnergyProxy");
      const ratio = candidateKinetic !== undefined && referenceKinetic !== undefined && referenceKinetic > 1e-9
        ? candidateKinetic / referenceKinetic : undefined;
      findings.push(scalarFinding({ id: "kinetic-reference-ratio", method, value: ratio,
        operator: "minimum", expected: input.parameters.minimumKineticEnergyRatioToReference,
        label: `quadtree/${input.parameters.referenceMethod} kinetic-energy ratio` }));
    }
  }
  return findings;
}
