import type { DeepReadonly, SceneDiagnosticEvidence, SceneDiagnosticHookFinding } from "./scene-diagnostics";
import type { SceneDescription } from "./model";
import {
  arrayPath,
  hookFinding,
  numberPath,
  numberValue,
  recordPath,
  recordValue,
  selectedMethodDiagnostics,
} from "./scene-hook-evidence";
import {
  evaluateMinimalDamCeilingSeparation,
  type DiagnosticLimitScheduleEntry,
} from "./scene-minimal-dam-diagnostic";

export interface WaterRasterIntegrityDiagnosticParameters extends Readonly<Record<string, unknown>> {
  minimumFrontInterfacePixels: number;
  minimumBackInterfacePixels: number;
  maximumBackOnlyPixels: number;
  boundsToleranceRatio: number;
  expectedSurfaceGeometrySource: string;
  requireGlobalFineCrossingPublished: boolean;
  requirePresentationFallbackInactive: boolean;
  requireNonzeroAuthorityLatch: boolean;
  requireNonemptyGeometry: boolean;
  requireAllocatorCountsMatch: boolean;
  requireGeometryWithinCapacity: boolean;
  requireCleanFineCoarseTransition: boolean;
  expectedRetainedGeometrySource: string;
  requireRetainedRasterIdentity: boolean;
  minimumValidSamples: number;
  minimumNegativeValidSamples: number;
  minimumPositiveValidSamples: number;
}

function check(input: {
  findings: SceneDiagnosticHookFinding[];
  id: string;
  method: string;
  passed: boolean;
  message: string;
  expected?: unknown;
  actual?: unknown;
}): void {
  input.findings.push(hookFinding(input));
}

function finiteBounds(value: unknown): readonly [readonly [number, number, number], readonly [number, number, number]] | undefined {
  return Array.isArray(value) && value.length === 2
    && value.every((point) => Array.isArray(point) && point.length === 3 && point.every(Number.isFinite))
    ? value as unknown as readonly [readonly [number, number, number], readonly [number, number, number]]
    : undefined;
}

function schedule(value: unknown): readonly DiagnosticLimitScheduleEntry[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const entries = value.flatMap((entry) => {
    const record = recordValue(entry), maximum = numberValue(record?.maximum);
    const before_s = numberValue(record?.before_s);
    return record && maximum !== undefined
      ? [{ maximum, ...(before_s === undefined ? {} : { before_s }) }] : [];
  });
  return entries.length === value.length ? entries : undefined;
}

function inspectRaster(input: {
  findings: SceneDiagnosticHookFinding[];
  scene: DeepReadonly<SceneDescription>;
  method: string;
  label: string;
  raster: Readonly<Record<string, unknown>> | undefined;
  generation: Readonly<Record<string, unknown>> | undefined;
  parameters: WaterRasterIntegrityDiagnosticParameters;
  initial: boolean;
}): void {
  const { findings, method, label, raster, generation, parameters } = input;
  check({ findings, id: `${method}.${label}.raster-present`, method, passed: raster !== undefined,
    message: raster ? `${label} water raster is available` : `${label} water raster is unavailable` });
  if (!raster) return;
  const front = numberValue(raster.frontInterfacePixels), back = numberValue(raster.backInterfacePixels);
  const backOnly = numberValue(raster.backOnlyInterfacePixels);
  check({ findings, id: `${method}.${label}.front`, method,
    passed: front !== undefined && front >= parameters.minimumFrontInterfacePixels,
    message: `${label} front-interface pixels ${front ?? "unknown"}`,
    expected: { minimum: parameters.minimumFrontInterfacePixels }, actual: front });
  check({ findings, id: `${method}.${label}.back`, method,
    passed: back !== undefined && back >= parameters.minimumBackInterfacePixels,
    message: `${label} back-interface pixels ${back ?? "unknown"}`,
    expected: { minimum: parameters.minimumBackInterfacePixels }, actual: back });
  check({ findings, id: `${method}.${label}.back-only`, method,
    passed: backOnly !== undefined && backOnly <= parameters.maximumBackOnlyPixels,
    message: `${label} back-only interface pixels ${backOnly ?? "unknown"}`,
    expected: { maximum: parameters.maximumBackOnlyPixels }, actual: backOnly });
  const reverse = recordValue(raster.reverseView);
  const reverseFront = numberValue(reverse?.frontInterfacePixels);
  const reverseBack = numberValue(reverse?.backInterfacePixels);
  const reverseBackOnly = numberValue(reverse?.backOnlyInterfacePixels);
  check({ findings, id: `${method}.${label}.reverse-depth-peel`, method,
    passed: reverseFront !== undefined && reverseFront >= parameters.minimumFrontInterfacePixels
      && reverseBack !== undefined && reverseBack >= parameters.minimumBackInterfacePixels
      && reverseBackOnly !== undefined && reverseBackOnly <= parameters.maximumBackOnlyPixels,
    message: `${label} reverse depth peeling must retain paired crossings`,
    expected: { minimumFront: parameters.minimumFrontInterfacePixels,
      minimumBack: parameters.minimumBackInterfacePixels, maximumBackOnly: parameters.maximumBackOnlyPixels },
    actual: { front: reverseFront, back: reverseBack, backOnly: reverseBackOnly } });

  const geometrySource = typeof raster.surfaceGeometrySource === "string" ? raster.surfaceGeometrySource : undefined;
  check({ findings, id: `${method}.${label}.geometry-source`, method,
    passed: geometrySource === parameters.expectedSurfaceGeometrySource,
    message: `${label} geometry source ${geometrySource ?? "unknown"}`,
    expected: parameters.expectedSurfaceGeometrySource, actual: geometrySource });
  if (parameters.requireGlobalFineCrossingPublished) check({
    findings, id: `${method}.${label}.fine-crossing`, method,
    passed: raster.globalFineCrossingPublished === true,
    message: `${label} must publish global-fine crossings`, expected: true,
    actual: raster.globalFineCrossingPublished,
  });
  if (parameters.requirePresentationFallbackInactive) check({
    findings, id: `${method}.${label}.fallback`, method,
    passed: raster.presentationFallbackActive === false,
    message: `${label} presentation fallback must remain inactive`, expected: false,
    actual: raster.presentationFallbackActive,
  });
  if (parameters.requireNonzeroAuthorityLatch) {
    const latch = numberValue(raster.globalFineAuthorityLatch);
    check({ findings, id: `${method}.${label}.authority-latch`, method, passed: latch !== undefined && latch !== 0,
      message: `${label} global-fine authority latch must be nonzero`, actual: latch });
  }
  const vertexCount = numberValue(raster.vertexCount), allocator = numberValue(raster.vertexAllocator);
  const vertexCapacity = numberValue(raster.vertexCapacity), cubes = numberValue(raster.activeCubeCount);
  const cubeCapacity = numberValue(raster.activeCubeCapacity);
  if (parameters.requireNonemptyGeometry) check({ findings, id: `${method}.${label}.nonempty`, method,
    passed: vertexCount !== undefined && vertexCount > 0 && cubes !== undefined && cubes > 0,
    message: `${label} water geometry must be nonempty`, actual: { vertexCount, activeCubeCount: cubes } });
  if (parameters.requireAllocatorCountsMatch) check({ findings, id: `${method}.${label}.allocator`, method,
    passed: vertexCount !== undefined && allocator === vertexCount,
    message: `${label} vertex allocator must equal vertex count`, expected: vertexCount, actual: allocator });
  if (parameters.requireGeometryWithinCapacity) check({ findings, id: `${method}.${label}.capacity`, method,
    passed: vertexCount !== undefined && vertexCapacity !== undefined && vertexCount <= vertexCapacity
      && cubes !== undefined && cubeCapacity !== undefined && cubes <= cubeCapacity,
    message: `${label} water geometry must remain within capacity`,
    actual: { vertexCount, vertexCapacity, activeCubeCount: cubes, activeCubeCapacity: cubeCapacity } });

  const bounds = finiteBounds(raster.frontInterfaceBounds_m);
  const tolerance = Math.max(input.scene.container.width_m, input.scene.container.height_m,
    input.scene.container.depth_m) * parameters.boundsToleranceRatio;
  const inside = bounds !== undefined
    && bounds[0][0] >= -0.5 * input.scene.container.width_m - tolerance
    && bounds[1][0] <= 0.5 * input.scene.container.width_m + tolerance
    && bounds[0][1] >= -tolerance && bounds[1][1] <= input.scene.container.height_m + tolerance
    && bounds[0][2] >= -0.5 * input.scene.container.depth_m - tolerance
    && bounds[1][2] <= 0.5 * input.scene.container.depth_m + tolerance;
  check({ findings, id: `${method}.${label}.bounds`, method, passed: inside,
    message: `${label} interface bounds must be finite and inside the tank`, actual: bounds });

  const transition = recordValue(raster.globalFineAuthorityTransition);
  if (parameters.requireCleanFineCoarseTransition) check({
    findings, id: `${method}.${label}.clean-transition`, method,
    passed: transition?.cleanFineCoarseRequired === true
      && numberValue(transition.validGeneration) === numberValue(generation?.generation)
      && transition.retainedGeometrySource === parameters.expectedRetainedGeometrySource,
    message: `${label} must reject unpublished B and retain clean A`, actual: transition,
  });
  if (parameters.requireRetainedRasterIdentity) check({
    findings, id: `${method}.${label}.retained-identity`, method,
    passed: numberValue(transition?.retainedFrontInterfacePixels) === front
      && numberValue(transition?.retainedBackInterfacePixels) === back
      && transition?.retainedFrontInterfaceHash === raster.frontInterfaceHash
      && transition?.retainedBackInterfaceHash === raster.backInterfaceHash,
    message: `${label} retained raster must be byte-identical after the unpublished-B probe`, actual: transition,
  });

  if (input.initial && parameters.initialDamCornerCaps === true) {
    const slits = numberPath(raster, "narrowVerticalSlits", "count");
    const reverseSlits = numberPath(reverse, "narrowVerticalSlits", "count");
    if (parameters.requireZeroInitialNarrowVerticalSlits === true) check({
      findings, id: `${method}.${label}.slits`, method,
      passed: slits === 0 && reverseSlits === 0,
      message: `${label} must contain no narrow vertical surface slits`, actual: { slits, reverseSlits },
    });
    const minimumWall = numberValue(parameters.minimumReservoirWallCornerCapPixels);
    const minimumDam = numberValue(parameters.minimumExposedDamCornerCapPixelsPerFace);
    if (minimumWall !== undefined) {
      const wallCaps = Array.isArray(reverse?.wallCornerCapPixels) ? reverse.wallCornerCapPixels : [];
      check({ findings, id: `${method}.${label}.wall-corner-cap`, method,
      passed: numberValue(wallCaps[0]) !== undefined && (numberValue(wallCaps[0]) ?? 0) >= minimumWall,
      message: `${label} reservoir wall corner cap is required`, expected: { minimum: minimumWall },
      actual: wallCaps[0] });
    }
    if (minimumDam !== undefined) {
      const caps = Array.isArray(raster.damExposedCornerCapPixels) ? raster.damExposedCornerCapPixels : [];
      check({ findings, id: `${method}.${label}.dam-corner-caps`, method,
        passed: numberValue(caps[0]) !== undefined && (numberValue(caps[0]) ?? 0) >= minimumDam
          && numberValue(caps[1]) !== undefined && (numberValue(caps[1]) ?? 0) >= minimumDam,
        message: `${label} exposed dam corner caps are required`, expected: { minimumPerFace: minimumDam }, actual: caps });
    }
  }
}

export function evaluateWaterRasterIntegrityDiagnostic(input: {
  scene: DeepReadonly<SceneDescription>;
  evidence: SceneDiagnosticEvidence;
  parameters: WaterRasterIntegrityDiagnosticParameters;
  methods?: readonly string[];
}): readonly SceneDiagnosticHookFinding[] {
  const findings: SceneDiagnosticHookFinding[] = [];
  for (const [method, diagnostics] of selectedMethodDiagnostics(input.evidence, input.methods)) {
    const rasterNamespace = recordPath(diagnostics, "raster") ?? diagnostics;
    const checkpoints = (arrayPath(rasterNamespace, "checkpoints")
      ?? arrayPath(diagnostics, "globalFineGenerationCheckpoints") ?? [])
      .map(recordValue).filter((value) => value !== undefined);
    const firstCheckpoint = checkpoints[0], lastCheckpoint = checkpoints.at(-1);
    // Checkpoint-authored lanes deliberately avoid redundant t=0/final raster
    // captures. Their first and last checkpoint are the corresponding closure
    // evidence and must feed the same integrity gate.
    const explicitInitial = recordPath(rasterNamespace, "initial")
      ?? recordPath(diagnostics, "initialGlobalFineRaster");
    const explicitFinal = recordPath(rasterNamespace, "final")
      ?? recordPath(diagnostics, "finalGlobalFineRaster");
    const initial = explicitInitial ?? recordPath(firstCheckpoint, "raster");
    const final = explicitFinal ?? recordPath(lastCheckpoint, "raster");
    const initialGeneration = explicitInitial
      ? recordPath(rasterNamespace, "initialGeneration")
        ?? recordPath(diagnostics, "initialGlobalFineGeneration")
      : recordPath(firstCheckpoint, "globalFineGeneration")
        ?? recordPath(diagnostics, "initialGlobalFineGeneration");
    const finalGeneration = explicitFinal
      ? recordPath(rasterNamespace, "finalGeneration")
        ?? recordPath(diagnostics, "finalGlobalFineGeneration")
      : recordPath(lastCheckpoint, "globalFineGeneration")
        ?? recordPath(diagnostics, "finalGlobalFineGeneration");
    inspectRaster({ findings, scene: input.scene, method, label: "initial", raster: initial,
      generation: initialGeneration, parameters: input.parameters, initial: true });
    inspectRaster({ findings, scene: input.scene, method, label: "final", raster: final,
      generation: finalGeneration, parameters: input.parameters, initial: false });
    const valid = numberValue(finalGeneration?.validSamples);
    const negative = numberValue(finalGeneration?.negativeValidSamples);
    const positive = numberValue(finalGeneration?.positiveValidSamples);
    check({ findings, id: `${method}.final.signed-generation`, method,
      passed: valid !== undefined && valid >= input.parameters.minimumValidSamples
        && negative !== undefined && negative >= input.parameters.minimumNegativeValidSamples
        && positive !== undefined && positive >= input.parameters.minimumPositiveValidSamples,
      message: "final global-fine generation must contain a nonempty signed interface",
      expected: { valid: input.parameters.minimumValidSamples, negative: input.parameters.minimumNegativeValidSamples,
        positive: input.parameters.minimumPositiveValidSamples }, actual: { valid, negative, positive } });
  }

  if (input.parameters.separatingCeilingContact === true) {
    const wetSchedule = schedule(input.parameters.ceilingWetCellLimits);
    const pixelSchedule = schedule(input.parameters.ceilingContactPixelLimits);
    const evaluateAfter_s = numberValue(input.parameters.separationEvaluationAfter_s);
    const wetStart_s = numberValue(input.parameters.ceilingWetEvaluationStart_s);
    const pixelStart_s = numberValue(input.parameters.ceilingContactEvaluationStart_s);
    if (wetSchedule && pixelSchedule && evaluateAfter_s !== undefined
      && wetStart_s !== undefined && pixelStart_s !== undefined) {
      findings.push(...evaluateMinimalDamCeilingSeparation({
        scene: input.scene, evidence: input.evidence, methods: input.methods,
        parameters: { evaluateAfter_s, wetCellsStart_s: wetStart_s,
          contactPixelsStart_s: pixelStart_s, ceilingWetCellLimits: wetSchedule,
          ceilingContactPixelLimits: pixelSchedule },
      }));
    } else findings.push(hookFinding({
      id: "ceiling-separation.parameters", passed: false,
      message: "separating ceiling contact has incomplete declared schedules",
    }));
  }
  return findings;
}
