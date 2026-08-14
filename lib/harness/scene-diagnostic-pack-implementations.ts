import {
  defineDiagnosticPackImplementation,
  type DiagnosticPackImplementationContext,
  type DiagnosticPackRegistry,
  type RuntimeDiagnosticFinding,
} from "./scene-diagnostic-runtime";
import type { SceneWebGPUDiagnosticPackId } from "./scene-webgpu-smoke";
import { evaluateSettlingDiagnostic } from "./scene-settling-diagnostic";
import {
  arrayPath,
  booleanPath,
  numberPath,
  recordPath,
  recordValue,
  runSteps,
  runTime,
  selectedMethodDiagnostics,
  type UnknownRecord,
} from "./scene-hook-evidence";

type Context = DiagnosticPackImplementationContext;

export type CompleteDiagnosticPackRegistry = {
  readonly [Id in SceneWebGPUDiagnosticPackId]: NonNullable<DiagnosticPackRegistry[Id]>;
};

function finding(input: Omit<RuntimeDiagnosticFinding, "passed"> & { passed: unknown }): RuntimeDiagnosticFinding {
  return { ...input, passed: input.passed === true };
}

function scalar(input: {
  id: string;
  method: string;
  actual: unknown;
  label: string;
  expected: unknown;
  pass: (actual: number) => boolean;
}): RuntimeDiagnosticFinding {
  const passed = typeof input.actual === "number" && Number.isFinite(input.actual) && input.pass(input.actual);
  return finding({
    id: input.id, method: input.method, passed,
    message: passed ? `${input.label} passed` : `${input.label} is ${String(input.actual)}; expected ${JSON.stringify(input.expected)}`,
    actual: input.actual, expected: input.expected,
  });
}

function truth(input: {
  id: string;
  method: string;
  actual: unknown;
  label: string;
  expected?: unknown;
  pass?: (actual: unknown) => boolean;
}): RuntimeDiagnosticFinding {
  const passed = input.pass ? input.pass(input.actual) : input.actual === true;
  return finding({
    id: input.id, method: input.method, passed,
    message: passed ? `${input.label} passed` : `${input.label} failed`,
    actual: input.actual, expected: input.expected ?? true,
  });
}

function parameterNumber(context: Context, key: string, fallback: number): number {
  const value = context.parameters[key];
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function parameterBoolean(context: Context, key: string, fallback: boolean): boolean {
  const value = context.parameters[key];
  return typeof value === "boolean" ? value : fallback;
}

function parameterString(context: Context, key: string, fallback: string): string {
  const value = context.parameters[key];
  return typeof value === "string" ? value : fallback;
}

function selected(context: Context): readonly (readonly [string, UnknownRecord])[] {
  return selectedMethodDiagnostics(context.evidence, context.selectedMethods);
}

function solver(diagnostics: UnknownRecord): UnknownRecord {
  return recordPath(diagnostics, "solver") ?? recordPath(diagnostics, "info") ?? diagnostics;
}

function run(diagnostics: UnknownRecord): UnknownRecord {
  return recordPath(diagnostics, "run") ?? diagnostics;
}

function stability(diagnostics: UnknownRecord): UnknownRecord | undefined {
  return recordPath(diagnostics, "stability") ?? recordPath(diagnostics, "stabilityEnvelope");
}

function matchedSummary(diagnostics: UnknownRecord): UnknownRecord | undefined {
  return recordPath(diagnostics, "field", "matched", "summary")
    ?? recordPath(diagnostics, "matchedSummary");
}

function finalSummary(diagnostics: UnknownRecord): UnknownRecord | undefined {
  return recordPath(diagnostics, "field", "final", "summary")
    ?? recordPath(diagnostics, "finalSummary") ?? matchedSummary(diagnostics);
}

function grid(diagnostics: UnknownRecord): readonly number[] | undefined {
  const value = recordPath(diagnostics, "field")?.grid ?? diagnostics.grid;
  return Array.isArray(value) && value.length === 3 && value.every((entry) => Number.isInteger(entry) && entry > 0)
    ? value as number[] : undefined;
}

function exactVolumeDrift(diagnostics: UnknownRecord): number | undefined {
  const summary = finalSummary(diagnostics);
  const initial = numberPath(solver(diagnostics), "initialVolumeCellSum");
  const sum = numberPath(summary, "cellSum");
  return sum !== undefined && initial !== undefined
    ? (sum - initial) / Math.max(1, Math.abs(initial))
    : numberPath(solver(diagnostics), "representedVolumeDrift");
}

const coreWebGPUHealth = defineDiagnosticPackImplementation({
  id: "core-webgpu-health",
  requires: ["run", "solver"],
  evaluate: (context) => selected(context).flatMap(([method, diagnostics]) => {
    const info = solver(diagnostics);
    const errors = arrayPath(diagnostics, "validationErrors") ?? arrayPath(info, "validationErrors") ?? [];
    const steps = runSteps(diagnostics);
    const maximumErrors = parameterNumber(context, "maximumValidationErrors", 0);
    const maximumNonFinite = parameterNumber(context, "maximumNonFiniteValues", 0);
    const findings: RuntimeDiagnosticFinding[] = [
      scalar({ id: "steps-positive", method, actual: steps, label: "accepted step count", expected: { minimum: 1 }, pass: (value) => value >= 1 }),
      scalar({ id: "validation-errors", method, actual: errors.length, label: "WebGPU validation error count", expected: { maximum: maximumErrors }, pass: (value) => value <= maximumErrors }),
      scalar({ id: "nonfinite-state", method, actual: numberPath(info, "nonFiniteCount") ?? 0,
        label: "non-finite state count", expected: { maximum: maximumNonFinite }, pass: (value) => value <= maximumNonFinite }),
    ];
    if (parameterBoolean(context, "requireFiniteMaximumSpeed", true)) findings.push(scalar({
      id: "finite-maximum-speed", method, actual: numberPath(info, "maxSpeed_m_s"), label: "maximum speed",
      expected: "finite", pass: () => true,
    }));
    const exactSteps = context.lane.stop.exactSteps;
    if (exactSteps !== undefined) {
      const expectedTime = exactSteps * (context.lane.stop.maxDt_s ?? context.scene.numerics.maxDt_s);
      const tolerance = parameterNumber(context, "exactStepTimeTolerance_s", 1e-9);
      findings.push(
        scalar({ id: "exact-steps", method, actual: steps, label: "accepted exact steps", expected: exactSteps, pass: (value) => value === exactSteps }),
        scalar({ id: "encoded-steps", method, actual: numberPath(info, "encodedSteps"), label: "encoded exact steps", expected: exactSteps, pass: (value) => value === exactSteps }),
        scalar({ id: "submitted-time", method, actual: numberPath(info, "submittedTime_s"), label: "submitted time", expected: expectedTime, pass: (value) => Math.abs(value - expectedTime) <= tolerance }),
        scalar({ id: "completed-time", method, actual: numberPath(info, "completedTime_s"), label: "completed time", expected: expectedTime, pass: (value) => Math.abs(value - expectedTime) <= tolerance }),
      );
    }
    return findings;
  }),
});

const volumeAndTopology = defineDiagnosticPackImplementation({
  id: "volume-and-topology",
  requires: ["field summary"],
  evaluate: (context) => selected(context).flatMap(([method, diagnostics]) => {
    const minimum = parameterNumber(context, "minimumStoredDensity", -0.01);
    const maximum = parameterNumber(context, "maximumStoredDensity", 1.5);
    const summaries = [["matched", matchedSummary(diagnostics)], ["final", finalSummary(diagnostics)]] as const;
    const findings = summaries.flatMap(([label, summary]) => [
      scalar({ id: `${label}-minimum`, method, actual: numberPath(summary, "minimum"), label: `${label} stored-density minimum`, expected: { minimum }, pass: (value) => value >= minimum }),
      scalar({ id: `${label}-maximum`, method, actual: numberPath(summary, "maximum"), label: `${label} stored-density maximum`, expected: { maximum }, pass: (value) => value <= maximum }),
      scalar({ id: `${label}-components`, method, actual: numberPath(summary, "componentCount"), label: `${label} component count`, expected: { minimum: 0 }, pass: (value) => Number.isInteger(value) && value >= 0 }),
    ]);
    const fieldGrid = grid(diagnostics);
    if (fieldGrid) findings.push(truth({
      id: "grid", method, actual: fieldGrid, label: "three-dimensional field grid",
      expected: "three positive integer dimensions", pass: (value) => Array.isArray(value) && value.length === 3,
    }));
    return findings;
  }),
});

const equilibrium = defineDiagnosticPackImplementation({
  id: "equilibrium",
  requires: ["solver", "field summary"],
  evaluate: (context) => selected(context).flatMap(([method, diagnostics]) => {
    const summary = finalSummary(diagnostics);
    const maximumDrift = parameterNumber(context, "maximumExactVolumeDrift", 0.01);
    const maximumComponents = parameterNumber(context, "maximumComponents", 1);
    return [
      scalar({ id: "exact-volume-drift", method, actual: Math.abs(exactVolumeDrift(diagnostics) ?? Infinity), label: "equilibrium exact-volume drift", expected: { maximum: maximumDrift }, pass: (value) => value <= maximumDrift }),
      scalar({ id: "components", method, actual: numberPath(summary, "componentCount"), label: "equilibrium component count", expected: { maximum: maximumComponents }, pass: (value) => value <= maximumComponents }),
    ];
  }),
});

const deepCompression = defineDiagnosticPackImplementation({
  id: "deep-compression",
  requires: ["solver"],
  evaluate: (context) => selected(context).map(([method, diagnostics]) => {
    const maximum = parameterNumber(context, "maximumCompressionRatioExclusive", 0.5);
    return scalar({ id: "compression-ratio", method, actual: numberPath(solver(diagnostics), "compressionRatio"),
      label: "compression ratio", expected: { exclusiveMaximum: maximum }, pass: (value) => value < maximum });
  }),
});

const inflowActivity = defineDiagnosticPackImplementation({
  id: "inflow-activity",
  requires: ["solver"],
  evaluate: (context) => selected(context).flatMap(([method, diagnostics]) => {
    const info = solver(diagnostics);
    const ratio = numberPath(info, "sourceAdjustedRepresentedVolumeRatio")
      ?? ((numberPath(info, "representedVolumeCellSum") ?? numberPath(info, "volumeCellSum")) !== undefined
        && numberPath(info, "referenceVolumeCellSum") !== undefined
        ? (numberPath(info, "representedVolumeCellSum") ?? numberPath(info, "volumeCellSum"))!
          / Math.max(1, numberPath(info, "referenceVolumeCellSum")!) : undefined);
    const minimumRatio = parameterNumber(context, "minimumSourceAdjustedRepresentedVolumeRatio", 0.99);
    const findings = [scalar({ id: "source-volume", method, actual: ratio, label: "source-adjusted represented-volume ratio",
      expected: { minimum: minimumRatio }, pass: (value) => value >= minimumRatio })];
    const elapsed = runTime(diagnostics);
    const start = parameterNumber(context, "motionEvaluationStart_s", 0.3);
    if (elapsed !== undefined && elapsed >= start) {
      const minimumSpeed = parameterNumber(context, "minimumEstablishedMaximumSpeed_m_s", 0.01);
      findings.push(scalar({ id: "established-motion", method, actual: numberPath(info, "maxSpeed_m_s"), label: "established inflow speed",
        expected: { minimum_m_s: minimumSpeed }, pass: (value) => value >= minimumSpeed }));
    }
    return findings;
  }),
});

const octreeAuthority = defineDiagnosticPackImplementation({
  id: "octree-authority",
  requires: ["solver", "octree authority"],
  evaluate: (context) => selected(context).flatMap(([method, diagnostics]) => {
    const info = solver(diagnostics), topology = recordPath(diagnostics, "octree", "powerTopology")
      ?? recordPath(diagnostics, "octreePowerTopologyDiagnostics");
    const expectedKind = parameterString(context, "expectedGridKind", "octree");
    const findings: RuntimeDiagnosticFinding[] = [
      truth({ id: "grid-kind", method, actual: info.gridKind, label: "octree grid kind", expected: expectedKind, pass: (value) => value === expectedKind }),
      scalar({ id: "neighbor-ratio", method, actual: numberPath(info, "quadtreeMaximumNeighborRatio"), label: "octree neighbor ratio",
        expected: { maximum: parameterNumber(context, "maximumNeighborRatio", 2) }, pass: (value) => value <= parameterNumber(context, "maximumNeighborRatio", 2) }),
      scalar({ id: "topology-readback", method, actual: numberPath(info, "quadtreeTopologyReadbackBytes"), label: "topology CPU readback bytes",
        expected: { maximum: parameterNumber(context, "maximumTopologyReadbackBytes", 0) }, pass: (value) => value <= parameterNumber(context, "maximumTopologyReadbackBytes", 0) }),
      truth({ id: "power-ready", method, actual: booleanPath(info, "powerDiagramReady"), label: "power diagram ready" }),
      truth({ id: "power-authoritative", method, actual: booleanPath(info, "powerDiagramAuthoritative"), label: "power diagram authority" }),
      scalar({ id: "descriptor-errors", method, actual: numberPath(topology, "descriptor", "errorCount"), label: "power descriptor errors",
        expected: { maximum: parameterNumber(context, "maximumDescriptorErrors", 0) }, pass: (value) => value <= parameterNumber(context, "maximumDescriptorErrors", 0) }),
      scalar({ id: "topology-errors", method, actual: numberPath(topology, "topology", "invalidCount"), label: "power topology errors",
        expected: { maximum: parameterNumber(context, "maximumTopologyErrors", 0) }, pass: (value) => value <= parameterNumber(context, "maximumTopologyErrors", 0) }),
      truth({ id: "pressure-solver", method, actual: info.pressureSolver, label: "Section 4.3 pressure solver",
        expected: parameterString(context, "pressureSolverNameIncludes", "exact-reduction executor"),
        pass: (value) => typeof value === "string" && value.includes(parameterString(context, "pressureSolverNameIncludes", "exact-reduction executor")) }),
    ];
    // Octree authority is a core pack and is also used by timing-only lanes.
    // Dam-break envelope assertions are ancillary: evaluate them only when
    // that expensive evidence was actually collected by the authored lane.
    if (context.scene.fluid.initialCondition === "dam-break"
      && context.hasEvidence(method, "stability envelope")) {
      const envelope = stability(diagnostics);
      findings.push(
        scalar({ id: "dam-cfl", method, actual: numberPath(envelope, "peakComponentCfl"), label: "dam-break peak CFL",
          expected: { maximum: parameterNumber(context, "damBreakMaximumComponentCfl", 3) }, pass: (value) => value <= parameterNumber(context, "damBreakMaximumComponentCfl", 3) }),
        scalar({ id: "dam-volume-drift", method, actual: numberPath(envelope, "maximumExactVolumeDrift"), label: "dam-break volume drift",
          expected: { maximum: parameterNumber(context, "damBreakMaximumExactVolumeDrift", 0.01) }, pass: (value) => value <= parameterNumber(context, "damBreakMaximumExactVolumeDrift", 0.01) }),
        scalar({ id: "dam-connectivity", method, actual: numberPath(envelope, "minimumDominantComponentFraction"), label: "dam-break dominant component",
          expected: { minimum: parameterNumber(context, "damBreakMinimumDominantComponentFraction", 0.98) }, pass: (value) => value >= parameterNumber(context, "damBreakMinimumDominantComponentFraction", 0.98) }),
      );
    }
    return findings;
  }),
});

const structuredPower = defineDiagnosticPackImplementation({
  id: "structured-power",
  requires: ["solver", "power generation audit"],
  evaluate: (context) => selected(context).flatMap(([method, diagnostics]) => {
    const info = solver(diagnostics), steps = runSteps(diagnostics);
    const findings = [
      truth({ id: "structured-authority", method, actual: booleanPath(info, "powerDiagramAuthoritative"), label: "direct structured authority" }),
      scalar({ id: "neighbor-ratio", method, actual: numberPath(info, "quadtreeMaximumNeighborRatio"), label: "structured neighbor ratio",
        expected: { maximum: parameterNumber(context, "maximumNeighborRatio", 2) }, pass: (value) => value <= parameterNumber(context, "maximumNeighborRatio", 2) }),
    ];
    if (parameterBoolean(context, "requireEveryStepGenerationAuditWhenCollected", true)) findings.push(scalar({
      id: "generation-audit", method, actual: numberPath(diagnostics, "powerGenerationAuditedSteps"), label: "structured generation audits",
      expected: steps, pass: (value) => steps !== undefined && value === steps,
    }));
    return findings;
  }),
});

const exhaustivePowerGeneration = defineDiagnosticPackImplementation({
  id: "exhaustive-power-generation",
  requires: ["run", "stability envelope", "power generation audit"],
  evaluate: (context) => selected(context).flatMap(([method, diagnostics]) => {
    const steps = runSteps(diagnostics), envelope = stability(diagnostics);
    const findings: RuntimeDiagnosticFinding[] = [];
    if (context.lane.stop.exactSteps === undefined) findings.push(scalar({
      id: "minimum-steps", method, actual: steps, label: "exhaustive step count",
      expected: { minimum: parameterNumber(context, "minimumStepsWithoutExactStop", 50) }, pass: (value) => value >= parameterNumber(context, "minimumStepsWithoutExactStop", 50),
    }));
    findings.push(
      scalar({ id: "generation-audits", method, actual: numberPath(diagnostics, "powerGenerationAuditedSteps"), label: "audited power generations", expected: steps, pass: (value) => steps !== undefined && value === steps }),
      scalar({ id: "stability-samples", method, actual: numberPath(envelope, "sampledSteps"), label: "per-step stability samples", expected: steps, pass: (value) => steps !== undefined && value === steps }),
      scalar({ id: "invalid-volume", method, actual: numberPath(envelope, "invalidVolumeSampleCount"), label: "invalid volume samples", expected: { maximum: parameterNumber(context, "maximumInvalidVolumeSamples", 0) }, pass: (value) => value <= parameterNumber(context, "maximumInvalidVolumeSamples", 0) }),
      scalar({ id: "nonfinite-velocity", method, actual: numberPath(envelope, "nonFiniteVelocityCount"), label: "non-finite velocity samples", expected: { maximum: parameterNumber(context, "maximumNonFiniteVelocitySamples", 0) }, pass: (value) => value <= parameterNumber(context, "maximumNonFiniteVelocitySamples", 0) }),
      scalar({ id: "volume-drift", method, actual: numberPath(envelope, "maximumExactVolumeDrift"), label: "maximum exact-volume drift", expected: { maximum: parameterNumber(context, "maximumExactVolumeDrift", 0.01) }, pass: (value) => value <= parameterNumber(context, "maximumExactVolumeDrift", 0.01) }),
      scalar({ id: "pressure-residual", method, actual: numberPath(envelope, "maximumPressureRelativeResidual"), label: "maximum pressure relative residual", expected: { maximum: parameterNumber(context, "maximumPressureRelativeResidual", 1e-4) }, pass: (value) => value <= parameterNumber(context, "maximumPressureRelativeResidual", 1e-4) }),
    );
    return findings;
  }),
});

function generation(diagnostics: UnknownRecord, which: "initial" | "final"): UnknownRecord | undefined {
  return recordPath(diagnostics, "field", which, "globalFineGeneration")
    ?? recordPath(diagnostics, `${which}GlobalFineGeneration`);
}

const globalFinePublication = defineDiagnosticPackImplementation({
  id: "global-fine-publication",
  requires: ["global fine generation"],
  evaluate: (context) => selected(context).flatMap(([method, diagnostics]) => {
    const initial = generation(diagnostics, "initial"), final = generation(diagnostics, "final");
    const findings = [
      truth({ id: "initial-publication", method, actual: initial, label: "initial global-fine publication",
        pass: (value) => recordValue(value)?.publicationValid === true && (numberPath(value, "generation") ?? 0) > 0 }),
      truth({ id: "final-publication", method, actual: final, label: "final nonempty global-fine publication",
        pass: (value) => recordValue(value)?.publicationValid === true && (numberPath(value, "generation") ?? 0) > 0 && (numberPath(value, "activePages") ?? 0) > 0 }),
      truth({ id: "signed-interface", method, actual: final, label: "finite signed global-fine interface",
        pass: (value) => (numberPath(value, "validSamples") ?? 0) >= parameterNumber(context, "minimumValidSamples", 1)
          && numberPath(value, "finiteValidSamples") === numberPath(value, "validSamples")
          && (numberPath(value, "negativeValidSamples") ?? 0) >= 1 && (numberPath(value, "positiveValidSamples") ?? 0) >= 1 }),
    ];
    const expectedState = parameterNumber(context, "checkpointCoarseState", 0x8000_0000);
    const checkpoints = arrayPath(diagnostics, "field", "checkpoints") ?? arrayPath(diagnostics, "checkpoints") ?? [];
    checkpoints.forEach((checkpoint, index) => {
      const snapshot = recordPath(checkpoint, "globalFineGeneration");
      if (!snapshot) return;
      findings.push(truth({ id: `checkpoint-${index}`, method, actual: snapshot, label: `checkpoint ${index} fine/coarse publication`,
        pass: (value) => recordValue(value)?.publicationValid === true
          && numberPath(value, "coarseState") === expectedState
          && (!parameterBoolean(context, "requireMatchingCoarseGeneration", true)
            || numberPath(value, "coarseGeneration") === numberPath(value, "generation"))
          && (!parameterBoolean(context, "requireNoTopologyRollback", true) || booleanPath(value, "topologyRolledBack") === false)
          && numberPath(value, "topologyFinalizeReason") === parameterNumber(context, "topologyFinalizeReason", 0) }));
    });
    return findings;
  }),
});

/**
 * The `which` raster together with the global-fine generation it was actually
 * captured against. Checkpoint-authored lanes deliberately avoid redundant
 * t=0/final raster captures, so their first and last checkpoint are the
 * corresponding closure evidence. `scene-water-raster-integrity-diagnostic`
 * already reads them that way; without the same fallback here every metric in
 * this pack resolves to `undefined` on those lanes and reports as a physics
 * failure.
 *
 * The pair has to travel together. A checkpoint raster carries the generation
 * that was valid at *its* checkpoint, so grading it against the lane's t=0
 * publication compares two different moments and can only match by accident —
 * on `minimal-power-dam-break` the first checkpoint is generation 27 and the
 * initial publication is 2, and `initial-authority` failed on that alone.
 */
function rasterEvidence(diagnostics: UnknownRecord, which: "initial" | "final"): {
  readonly raster: UnknownRecord | undefined;
  readonly generation: UnknownRecord | undefined;
} {
  const explicit = recordPath(diagnostics, "raster", which)
    ?? recordPath(diagnostics, `${which}GlobalFineRaster`);
  if (explicit !== undefined) return { raster: explicit, generation: generation(diagnostics, which) };
  const checkpoints = (arrayPath(diagnostics, "raster", "checkpoints")
    ?? arrayPath(diagnostics, "globalFineGenerationCheckpoints") ?? [])
    .map(recordValue).filter((value) => value !== undefined);
  const checkpoint = which === "initial" ? checkpoints[0] : checkpoints.at(-1);
  return {
    raster: recordPath(checkpoint, "raster"),
    generation: recordPath(checkpoint, "globalFineGeneration") ?? generation(diagnostics, which),
  };
}

const authoritativeWaterRaster = defineDiagnosticPackImplementation({
  id: "authoritative-water-raster",
  requires: ["global fine generation", "front/back raster"],
  evaluate: (context) => selected(context).flatMap(([method, diagnostics]) => {
    const findings: RuntimeDiagnosticFinding[] = [];
    for (const which of ["initial", "final"] as const) {
      const { raster: value, generation: published } = rasterEvidence(diagnostics, which);
      const reverse = recordPath(value, "reverseView");
      const minimumFront = parameterNumber(context, "minimumFrontInterfacePixels", 1);
      const minimumBack = parameterNumber(context, "minimumBackInterfacePixels", 1);
      const maximumBackOnly = parameterNumber(context, "maximumBackOnlyPixels", 0);
      findings.push(
        scalar({ id: `${which}-front`, method, actual: numberPath(value, "frontInterfacePixels"), label: `${which} front interface pixels`, expected: { minimum: minimumFront }, pass: (actual) => actual >= minimumFront }),
        scalar({ id: `${which}-back`, method, actual: numberPath(value, "backInterfacePixels"), label: `${which} back interface pixels`, expected: { minimum: minimumBack }, pass: (actual) => actual >= minimumBack }),
        scalar({ id: `${which}-back-only`, method, actual: numberPath(value, "backOnlyInterfacePixels"), label: `${which} back-only pixels`, expected: { maximum: maximumBackOnly }, pass: (actual) => actual <= maximumBackOnly }),
        scalar({ id: `${which}-reverse-back-only`, method, actual: numberPath(reverse, "backOnlyInterfacePixels"), label: `${which} reverse back-only pixels`, expected: { maximum: maximumBackOnly }, pass: (actual) => actual <= maximumBackOnly }),
        truth({ id: `${which}-authority`, method, actual: value, label: `${which} raster authority`, pass: (actual) => {
          const record = recordValue(actual);
          // Section 5 keeps the authored interface on a separate fine grid. A
          // retained raster therefore identifies its accepted fine-grid
          // authority through the clean A/B transition, even when the compact
          // raster summary does not duplicate that generation at top level.
          const rasterGeneration = numberPath(record, "globalFineGeneration")
            ?? numberPath(record, "globalFineAuthorityTransition", "validGeneration");
          return record?.surfaceGeometrySource === parameterString(context, "expectedSurfaceGeometrySource", "global-fine-coarse")
            && (!parameterBoolean(context, "requireGlobalFineCrossingPublished", true) || record.globalFineCrossingPublished === true)
            && (!parameterBoolean(context, "requirePresentationFallbackInactive", true) || record.presentationFallbackActive === false)
            && (!parameterBoolean(context, "requireNonzeroAuthorityLatch", true) || (numberPath(record, "globalFineAuthorityLatch") ?? 0) !== 0)
            && rasterGeneration === numberPath(published, "generation");
        } }),
      );
    }
    return findings;
  }),
});

function occupancy(field: unknown): ArrayLike<number> | undefined {
  return field !== null && typeof field === "object" && "length" in field
    ? field as ArrayLike<number> : undefined;
}

function wetIoU(left: ArrayLike<number>, right: ArrayLike<number>): number | undefined {
  if (left.length !== right.length) return undefined;
  let intersection = 0, union = 0;
  for (let index = 0; index < left.length; index += 1) {
    const a = Number(left[index]) > 0.5, b = Number(right[index]) > 0.5;
    if (a && b) intersection += 1;
    if (a || b) union += 1;
  }
  return union === 0 ? 1 : intersection / union;
}

const crossMethodFieldParity = defineDiagnosticPackImplementation({
  id: "cross-method-field-parity",
  requires: ["volume field", "field summary"],
  evaluate: (context) => {
    const entries = selected(context);
    if (entries.length < 2) return [finding({ id: "method-count", passed: false,
      message: "cross-method field parity requires at least two methods", actual: entries.map(([method]) => method), expected: { minimum: 2 } })];
    const [referenceMethod, reference] = entries[0];
    const referenceGrid = grid(reference), referenceField = occupancy(recordPath(reference, "field", "final")?.field
      ?? recordPath(reference, "field", "matched")?.field ?? reference.matchedField);
    return entries.slice(1).flatMap(([method, diagnostics]) => {
      const candidateGrid = grid(diagnostics), candidateField = occupancy(recordPath(diagnostics, "field", "final")?.field
        ?? recordPath(diagnostics, "field", "matched")?.field ?? diagnostics.matchedField);
      const gridsMatch = candidateGrid !== undefined && referenceGrid !== undefined
        && candidateGrid.every((value, axis) => value === referenceGrid[axis]);
      const iou = candidateField && referenceField ? wetIoU(candidateField, referenceField) : undefined;
      const minimum = parameterNumber(context, "minimumFinalWetIntersectionOverUnion", 0.4);
      return [
        truth({ id: "grid", method, actual: candidateGrid, label: `${method}/${referenceMethod} grid parity`, expected: referenceGrid, pass: () => gridsMatch }),
        scalar({ id: "final-wet-iou", method, actual: iou, label: `${method}/${referenceMethod} final wet IoU`, expected: { minimum }, pass: (value) => value >= minimum }),
      ];
    });
  },
});

const settling = defineDiagnosticPackImplementation({
  id: "settling",
  requires: ["mechanical energy"],
  evaluate: (context) => evaluateSettlingDiagnostic({
    scene: context.scene,
    evidence: context.evidence,
    methods: context.selectedMethods,
    parameters: {
      expectAsymptoticRest: parameterBoolean(context, "expectAsymptoticRest", true),
      maximumFinalExactVolumeDrift: parameterNumber(context, "maximumFinalSampledExactVolumeDrift", 0.01),
      maximumNormalizedNetProjectionEnergyDelta: parameterNumber(context, "maximumNormalizedNetProjectionEnergyDelta", 0.01),
      maximumNormalizedLateMechanicalEnergySlopePerSecond: parameterNumber(context, "maximumNormalizedLateMechanicalEnergySlopePerSecond", 1e-3),
      maximumLateToMiddleKineticEnvelopeRatio: parameterNumber(context, "maximumLateToMiddleKineticEnvelopeRatio", 1),
      maximumDamBreakDriftSignChanges: parameterNumber(context, "damBreakMaximumDriftSignChanges", 3),
      maximumDamBreakLatePeakToPeakDrift: parameterNumber(context, "damBreakMaximumLatePeakToPeakDrift", 0.005),
    },
  }),
});

const performance = defineDiagnosticPackImplementation({
  id: "performance",
  requires: ["run", "performance authority"],
  evaluate: (context) => selected(context).flatMap(([method, diagnostics]) => {
    const execution = run(diagnostics), authority = recordPath(diagnostics, "performance", "finalAuthority")
      ?? recordPath(diagnostics, "finalPerformanceAuthority");
    const failures = arrayPath(authority, "failures");
    return [
      scalar({ id: "simulation-wall", method, actual: numberPath(execution, "simulationWall_ms"), label: "simulation wall time", expected: "positive finite", pass: (value) => value > 0 }),
      truth({ id: "final-authority", method, actual: authority, label: "final performance authority", pass: (value) => recordValue(value) !== undefined }),
      scalar({ id: "authority-failures", method, actual: failures?.length, label: "final performance authority failures", expected: 0, pass: (value) => value === 0 }),
    ];
  }),
});

export const sceneDiagnosticPackImplementations = Object.freeze({
  "core-webgpu-health": coreWebGPUHealth,
  "volume-and-topology": volumeAndTopology,
  equilibrium,
  "deep-compression": deepCompression,
  "inflow-activity": inflowActivity,
  "octree-authority": octreeAuthority,
  "structured-power": structuredPower,
  "exhaustive-power-generation": exhaustivePowerGeneration,
  "global-fine-publication": globalFinePublication,
  "authoritative-water-raster": authoritativeWaterRaster,
  "cross-method-field-parity": crossMethodFieldParity,
  settling,
  performance,
}) satisfies CompleteDiagnosticPackRegistry;
