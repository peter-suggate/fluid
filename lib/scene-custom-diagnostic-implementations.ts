import {
  defineDiagnosticPackImplementation,
  defineSceneHookImplementation,
  type DiagnosticPackRegistry,
  type RuntimeDiagnosticFinding,
  type SceneHookRegistry,
} from "./scene-diagnostic-runtime";
import {
  evaluateFieldVelocityParityDiagnostic,
  type FieldVelocityParityDiagnosticParameters,
  type FieldVelocityParityLimits,
} from "./scene-field-velocity-parity-diagnostic";
import {
  evaluateFreeFallContactDiagnostic,
  type FreeFallContactDiagnosticParameters,
} from "./scene-free-fall-contact-diagnostic";
import { evaluateRigidCouplingDiagnostic } from "./scene-rigid-coupling-diagnostic";
import {
  evaluateGardenBrickMigrationDiagnostic,
  type GardenBrickMigrationDiagnosticParameters,
} from "./scene-garden-brick-migration-diagnostic";
import {
  evaluateMinimalDamMotionDiagnostic,
  type MinimalDamMotionDiagnosticParameters,
} from "./scene-minimal-dam-diagnostic";
import {
  evaluateQuadtreeDamParityDiagnostic,
  type QuadtreeDamParityDiagnosticParameters,
} from "./scene-quadtree-dam-parity-diagnostic";
import {
  evaluateSettlingDiagnostic,
  type SettlingDiagnosticParameters,
} from "./scene-settling-diagnostic";
import {
  evaluateWaterRasterIntegrityDiagnostic,
  type WaterRasterIntegrityDiagnosticParameters,
} from "./scene-water-raster-integrity-diagnostic";
import type { SceneDescription } from "./model";
import { probeBrickQuadCoverage } from "../tools/brick-quad-coverage-probe";
import { probeOceanWavePropagation } from "../tools/ocean-wave-propagation-probe";
import {
  arrayPath,
  fieldCheckpoints,
  gridFromDiagnostics,
  hookFinding,
  numberPath,
  numberValue,
  recordPath,
  recordValue,
  runSteps,
} from "./scene-hook-evidence";

function numeric(parameters: Readonly<Record<string, unknown>>, key: string): number | undefined {
  const value = parameters[key];
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function boolean(parameters: Readonly<Record<string, unknown>>, key: string): boolean | undefined {
  const value = parameters[key];
  return typeof value === "boolean" ? value : undefined;
}

function string(parameters: Readonly<Record<string, unknown>>, key: string): string | undefined {
  const value = parameters[key];
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function parameterFailure(id: string, keys: readonly string[]): RuntimeDiagnosticFinding[] {
  return [{
    id: "parameters",
    passed: false,
    message: `${id} has missing or invalid declared parameters: ${keys.join(", ")}`,
    expected: keys,
  }];
}

function requiredNumbers(
  parameters: Readonly<Record<string, unknown>>,
  keys: readonly string[],
): Record<string, number> | undefined {
  const result: Record<string, number> = {};
  for (const key of keys) {
    const value = numeric(parameters, key);
    if (value === undefined) return undefined;
    result[key] = value;
  }
  return result;
}

const freeFallContact = defineSceneHookImplementation({
  id: "free-fall-contact",
  requires: ["checkpoint centroid", "ceiling wet cells", "front/back raster", "free-fall contact attribution"],
  evaluate: (context) => {
    const keys = ["minimumRunTime_s", "evaluationStart_s", "impactTime_s", "preImpactMargin_s",
      "minimumMeasuredToAnalyticDropRatio", "maximumMeasuredToAnalyticDropRatio",
      "maximumDropHeadroom_cells", "minimumPreImpactCheckpoints", "releaseCheckTime_s",
      "maximumCeilingWetCellsAfterRelease", "maximumCeilingPixelsAfterRelease"] as const;
    const values = requiredNumbers(context.parameters, keys);
    if (!values) return parameterFailure("free-fall-contact", keys);
    const includeCornerSeams = boolean(context.parameters, "includeCornerSeams");
    const seamEvaluationStart_s = numeric(context.parameters, "seamEvaluationStart_s");
    const maximumSeamShortfallExcess = numeric(context.parameters, "maximumSeamShortfallExcess");
    const maximumColumnLagSpread_cells = numeric(context.parameters, "maximumColumnLagSpread_cells");
    const maximumVelocityShortfallSpread = numeric(context.parameters, "maximumVelocityShortfallSpread");
    const maximumCenterToRimLag_cells = numeric(context.parameters, "maximumCenterToRimLag_cells");
    const maximumCenterProtrusion_cells = numeric(context.parameters, "maximumCenterProtrusion_cells");
    if (includeCornerSeams === true
      && (seamEvaluationStart_s === undefined || maximumSeamShortfallExcess === undefined)) {
      return parameterFailure("free-fall-contact", ["seamEvaluationStart_s", "maximumSeamShortfallExcess"]);
    }
    return evaluateFreeFallContactDiagnostic({
      scene: context.scene,
      evidence: context.evidence,
      parameters: {
        ...values,
        ...(includeCornerSeams === undefined ? {} : { includeCornerSeams }),
        ...(seamEvaluationStart_s === undefined ? {} : { seamEvaluationStart_s }),
        ...(maximumSeamShortfallExcess === undefined ? {} : { maximumSeamShortfallExcess }),
        ...(maximumColumnLagSpread_cells === undefined ? {} : { maximumColumnLagSpread_cells }),
        ...(maximumVelocityShortfallSpread === undefined ? {} : { maximumVelocityShortfallSpread }),
        ...(maximumCenterToRimLag_cells === undefined ? {} : { maximumCenterToRimLag_cells }),
        ...(maximumCenterProtrusion_cells === undefined ? {} : { maximumCenterProtrusion_cells }),
      } as unknown as FreeFallContactDiagnosticParameters,
      methods: context.selectedMethods,
    });
  },
});

const rigidCouplingOracle = defineSceneHookImplementation({
  id: "rigid-coupling-oracle",
  requires: ["rigid coupling", "checkpoint fields", "front/back raster"],
  evaluate: (context) => evaluateRigidCouplingDiagnostic({
    scene: context.scene as SceneDescription,
    evidence: context.evidence,
    methods: context.selectedMethods,
  }),
});

const gardenSourceBrickMigration = defineSceneHookImplementation({
  id: "garden-source-brick-migration",
  evaluate: (context) => {
    const keys = ["initialCoreBricks", "evaluateAfter_s", "minimumFinalCoreBricks",
      "sourceFluidVoxelsAtEnd"] as const;
    const values = requiredNumbers(context.parameters, keys);
    const sourceCoreResidencyAtEnd = boolean(context.parameters, "sourceCoreResidencyAtEnd");
    if (!values || sourceCoreResidencyAtEnd === undefined) {
      return parameterFailure("garden-source-brick-migration", [...keys, "sourceCoreResidencyAtEnd"]);
    }
    return evaluateGardenBrickMigrationDiagnostic({
      scene: context.scene,
      evidence: context.evidence,
      parameters: { ...values, sourceCoreResidencyAtEnd } as unknown as GardenBrickMigrationDiagnosticParameters,
      methods: context.selectedMethods,
    });
  },
});

const minimalDamMotion = defineSceneHookImplementation({
  id: "minimal-dam-motion",
  evaluate: (context) => {
    const minimumPeakSpeed_m_s = numeric(context.parameters, "minimumPeakSpeed_m_s");
    const minimumLateralSpread_m = numeric(context.parameters, "minimumLateralSpread_m");
    if (minimumPeakSpeed_m_s === undefined || minimumLateralSpread_m === undefined) {
      return parameterFailure("minimal-dam-motion", ["minimumPeakSpeed_m_s", "minimumLateralSpread_m"]);
    }
    const parameters: MinimalDamMotionDiagnosticParameters = {
      minimumPeakSpeed_m_s,
      minimumLateralSpread_m,
      ...(numeric(context.parameters, "maximumMechanicalEnergyRetention") === undefined ? {}
        : { maximumMechanicalEnergyRetention: numeric(context.parameters, "maximumMechanicalEnergyRetention") }),
      ...(numeric(context.parameters, "maximumRitterCelerityRatio") === undefined ? {}
        : { maximumRitterCelerityRatio: numeric(context.parameters, "maximumRitterCelerityRatio") }),
      ...(numeric(context.parameters, "energyEvaluationAfter_s") === undefined ? {}
        : { energyEvaluationAfter_s: numeric(context.parameters, "energyEvaluationAfter_s") }),
    };
    return evaluateMinimalDamMotionDiagnostic({
      scene: context.scene, evidence: context.evidence, parameters, methods: context.selectedMethods,
    });
  },
});

function velocityLimits(value: unknown): FieldVelocityParityLimits | undefined {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return undefined;
  const parameters = value as Readonly<Record<string, unknown>>;
  const keys = ["maximumWeightedRelativeL2", "minimumCosineSimilarity", "minimumEnergyRatio",
    "maximumEnergyRatio", "minimumPeakRatio", "maximumPeakRatio"] as const;
  const values = requiredNumbers(parameters, keys);
  return values as unknown as FieldVelocityParityLimits | undefined;
}

const damBreakVelocityParity = defineSceneHookImplementation({
  id: "dam-break-velocity-parity",
  evaluate: (context) => {
    const limits = velocityLimits(context.parameters.limits);
    const minimumWetIntersectionOverUnion = numeric(context.parameters, "minimumWetIntersectionOverUnion");
    const maximumCentroidDistanceCells = numeric(context.parameters, "maximumCentroidDistanceCells");
    const checkpointTimeTolerance_s = numeric(context.parameters, "checkpointTimeTolerance_s");
    if (!limits || minimumWetIntersectionOverUnion === undefined
      || maximumCentroidDistanceCells === undefined || checkpointTimeTolerance_s === undefined) {
      return parameterFailure("dam-break-velocity-parity", ["limits", "minimumWetIntersectionOverUnion",
        "maximumCentroidDistanceCells", "checkpointTimeTolerance_s"]);
    }
    const candidateMethod = string(context.parameters, "candidateMethod")
      ?? (context.selectedMethods.includes("octree") ? "octree" : context.selectedMethods[0]);
    const referenceMethod = string(context.parameters, "referenceMethod")
      ?? (context.selectedMethods.includes("tall-cell") ? "tall-cell" : context.selectedMethods[1]);
    if (!candidateMethod || !referenceMethod) {
      return parameterFailure("dam-break-velocity-parity", ["candidateMethod", "referenceMethod"]);
    }
    const parameters: FieldVelocityParityDiagnosticParameters = {
      candidateMethod, referenceMethod, limits, minimumWetIntersectionOverUnion,
      maximumCentroidDistanceCells, checkpointTimeTolerance_s,
    };
    return evaluateFieldVelocityParityDiagnostic({
      scene: context.scene, evidence: context.evidence, parameters,
    });
  },
});

const brickQuadCoverage = defineSceneHookImplementation({
  id: "brick-quad-coverage",
  evaluate: (context) => {
    const expected = context.parameters.expectedBrickGrid;
    const expectedBrickGrid = Array.isArray(expected) && expected.length === 3
      && expected.every((value) => typeof value === "number" && Number.isInteger(value) && value > 0)
      ? expected as unknown as readonly [number, number, number] : undefined;
    const minimumFirstCheckpointColumns = numeric(context.parameters, "minimumFirstCheckpointColumns");
    if (!expectedBrickGrid || minimumFirstCheckpointColumns === undefined) {
      return parameterFailure("brick-quad-coverage", ["expectedBrickGrid", "minimumFirstCheckpointColumns"]);
    }
    const findings: RuntimeDiagnosticFinding[] = [];
    for (const method of context.selectedMethods) {
      const diagnostics = recordValue(context.getMethod(method)?.diagnostics);
      const grid = diagnostics ? gridFromDiagnostics(diagnostics) : undefined;
      if (!diagnostics || !grid) {
        findings.push({ id: `${method}.field-evidence`, method, passed: false,
          message: `${method} has no normalized brick-coverage field evidence` });
        continue;
      }
      const result = probeBrickQuadCoverage(context.scene as unknown as SceneDescription, {
        method, grid,
        checkpoints: fieldCheckpoints(diagnostics).flatMap((value) => {
          const checkpoint = recordValue(value);
          const time_s = numberPath(checkpoint, "time_s");
          const field = checkpoint?.field;
          return checkpoint && time_s !== undefined && field !== null && typeof field === "object" && "length" in field
            ? [{ time_s, field: field as ArrayLike<number> }] : [];
        }),
      }, { expectedBrickGrid, minimumFirstCheckpointColumns });
      findings.push(...result.findings.map((finding) => ({
        id: `${method}.${finding.id}`, method, passed: false, severity: finding.severity,
        message: finding.message, expected: finding.expected, actual: finding.actual,
      })));

      const minimumInitial = numeric(context.parameters, "minimumInitialResidentBricks");
      const minimumFinalResident = numeric(context.parameters, "minimumFinalResidentBricks");
      const minimumFinalCore = numeric(context.parameters, "minimumFinalCoreBricks");
      const sparse = recordPath(diagnostics, "sparse");
      const initialResident = numberPath(sparse, "initialFluidBricks", "resident");
      const finalResident = numberPath(sparse, "finalPublication", "fluidBrickResidentCount");
      const finalCore = numberPath(sparse, "finalPublication", "fluidBrickCoreCount");
      if (minimumInitial !== undefined) findings.push(hookFinding({
        id: `${method}.initial-resident`, method, passed: initialResident !== undefined && initialResident >= minimumInitial,
        message: initialResident !== undefined && initialResident >= minimumInitial
          ? "initial brick residency passed" : "initial brick residency is below its declared minimum",
        expected: { minimum: minimumInitial }, actual: initialResident,
      }));
      if (minimumFinalResident !== undefined) findings.push(hookFinding({
        id: `${method}.final-resident`, method, passed: finalResident !== undefined && finalResident >= minimumFinalResident,
        message: finalResident !== undefined && finalResident >= minimumFinalResident
          ? "final brick residency passed" : "final brick residency is below its declared minimum",
        expected: { minimum: minimumFinalResident }, actual: finalResident,
      }));
      if (minimumFinalCore !== undefined) findings.push(hookFinding({
        id: `${method}.final-core`, method, passed: finalCore !== undefined && finalCore >= minimumFinalCore,
        message: finalCore !== undefined && finalCore >= minimumFinalCore
          ? "final core-brick count passed" : "final core-brick count is below its declared minimum",
        expected: { minimum: minimumFinalCore }, actual: finalCore,
      }));
    }
    return findings;
  },
});

const oceanWaveProfile = defineSceneHookImplementation({
  id: "ocean-wave-profile",
  evaluate: (context) => {
    const stationCount = numeric(context.parameters, "stationCount");
    const minimumCheckpoints = numeric(context.parameters, "minimumCheckpoints");
    const widthRatio = numeric(context.parameters, "minimumFarHalfDisturbanceWidthRatio");
    if (stationCount === undefined || minimumCheckpoints === undefined || widthRatio === undefined) {
      return parameterFailure("ocean-wave-profile", ["stationCount", "minimumCheckpoints",
        "minimumFarHalfDisturbanceWidthRatio"]);
    }
    const findings: RuntimeDiagnosticFinding[] = [];
    for (const method of context.selectedMethods) {
      const diagnostics = recordValue(context.getMethod(method)?.diagnostics);
      const grid = diagnostics ? gridFromDiagnostics(diagnostics) : undefined;
      if (!diagnostics || !grid) {
        findings.push({ id: `${method}.field-evidence`, method, passed: false,
          message: `${method} has no normalized ocean field evidence` });
        continue;
      }
      const result = probeOceanWavePropagation(context.scene as unknown as SceneDescription, {
        method, grid,
        checkpoints: fieldCheckpoints(diagnostics).flatMap((value) => {
          const checkpoint = recordValue(value), time_s = numberPath(checkpoint, "time_s");
          const field = checkpoint?.field;
          return checkpoint && time_s !== undefined && field !== null && typeof field === "object" && "length" in field
            ? [{ time_s, field: field as ArrayLike<number> }] : [];
        }),
      }, {
        stationCount, minimumCheckpointCount: minimumCheckpoints,
        minimumFarHalfDisturbance_cells: widthRatio / context.scene.container.width_m,
      });
      findings.push(...result.findings.map((finding) => ({
        id: `${method}.${finding.id}`, method, passed: false, severity: finding.severity,
        message: finding.message, expected: finding.expected, actual: finding.actual,
      })));
    }
    return findings;
  },
});

const hoseJetDrift = defineSceneHookImplementation({
  id: "hose-jet-drift",
  evaluate: (context) => {
    const keys = ["minimumSampledVelocityCells", "minimumOutletAxialSpeed_m_s",
      "maximumOutletSideSpeed_m_s", "maximumSideCentroidOffsetCoarseCells",
      "minimumSampledAirborneBins", "minimumAirborneAxialRetentionRatio",
      "maximumAdjacentAxialSpeedDropRatio", "maximumAdjacentMomentumFluxDropRatio",
      "minimumBallisticGravityVelocityRatio", "maximumBallisticGravityVelocityRatio",
      "maximumBallisticCenterlineRelativeError"] as const;
    const limits = requiredNumbers(context.parameters, keys);
    if (!limits) return parameterFailure("hose-jet-drift", keys);
    const findings: RuntimeDiagnosticFinding[] = [];
    for (const method of context.selectedMethods) {
      const diagnostics = recordValue(context.getMethod(method)?.diagnostics);
      const audit = diagnostics
        ? recordPath(diagnostics, "terminalEvidence", "hose-jet-drift") : undefined;
      const grid = diagnostics ? gridFromDiagnostics(diagnostics) : undefined;
      const coarseH = grid ? Math.max(context.scene.container.width_m / grid[0],
        context.scene.container.height_m / grid[1], context.scene.container.depth_m / grid[2]) : undefined;
      const checks = [
        ["sampled-velocity", "sampledVelocityCells", "minimum", limits.minimumSampledVelocityCells],
        ["outlet-axial", "outletAxialSpeed_m_s", "minimum", limits.minimumOutletAxialSpeed_m_s],
        ["outlet-side", "outletSideSpeed_m_s", "absolute-maximum", limits.maximumOutletSideSpeed_m_s],
        ["side-centroid", "maximumSideCentroidOffset_m", "maximum",
          coarseH === undefined ? undefined : limits.maximumSideCentroidOffsetCoarseCells * coarseH],
        ["airborne-bins", "sampledAirborneBins", "minimum", limits.minimumSampledAirborneBins],
        ["axial-retention", "minimumAirborneAxialRetentionRatio", "minimum", limits.minimumAirborneAxialRetentionRatio],
        ["axial-drop", "maximumAdjacentAxialSpeedDropRatio", "maximum", limits.maximumAdjacentAxialSpeedDropRatio],
        ["momentum-drop", "maximumAdjacentMomentumFluxDropRatio", "maximum", limits.maximumAdjacentMomentumFluxDropRatio],
        ["gravity-retention", "minimumBallisticGravityVelocityRatio", "minimum", limits.minimumBallisticGravityVelocityRatio],
        ["gravity-amplification", "maximumBallisticGravityVelocityRatio", "maximum", limits.maximumBallisticGravityVelocityRatio],
        ["ballistic-error", "maximumBallisticCenterlineRelativeError", "maximum", limits.maximumBallisticCenterlineRelativeError],
      ] as const;
      for (const [id, key, operator, expected] of checks) {
        const actual = numberPath(audit, key);
        const passed = actual !== undefined && expected !== undefined && (operator === "minimum"
          ? actual >= expected : operator === "absolute-maximum" ? Math.abs(actual) <= expected : actual <= expected);
        findings.push(hookFinding({
          id: `${method}.${id}`, method, passed,
          message: passed ? `hose ${id} passed` : `hose ${id} ${actual ?? "unknown"} failed ${operator} ${expected ?? "unknown"}`,
          expected: { [operator]: expected }, actual,
        }));
      }
    }
    return findings;
  },
});

const damBreakPerturbedCadence = defineSceneHookImplementation({
  id: "dam-break-perturbed-cadence",
  evaluate: (context) => {
    const pattern = context.parameters.dtPattern_s;
    if (!Array.isArray(pattern) || pattern.length === 0
      || !pattern.every((value) => typeof value === "number" && Number.isFinite(value) && value > 0)) {
      return parameterFailure("dam-break-perturbed-cadence", ["dtPattern_s"]);
    }
    return context.selectedMethods.flatMap((method) => {
      const diagnostics = recordValue(context.getMethod(method)?.diagnostics);
      const observed = diagnostics ? arrayPath(diagnostics, "run", "dtPattern_s") : undefined;
      const steps = diagnostics ? runSteps(diagnostics) : undefined;
      const patternMatches = observed !== undefined && observed.length === pattern.length
        && observed.every((value, index) => numberValue(value) === pattern[index]);
      return [
        hookFinding({
          id: `${method}.pattern`, method, passed: patternMatches,
          message: patternMatches ? "perturbed cadence matches the declared pattern"
            : "perturbed cadence evidence is absent or differs from the declared pattern",
          expected: pattern, actual: observed,
        }),
        hookFinding({
          id: `${method}.pattern-exercised`, method,
          passed: steps !== undefined && steps >= pattern.length,
          message: steps !== undefined && steps >= pattern.length
            ? "the run exercised a complete perturbed cadence cycle"
            : "the run did not exercise a complete perturbed cadence cycle",
          expected: { minimumSteps: pattern.length }, actual: steps,
        }),
      ];
    });
  },
});

const waterRasterIntegrity = defineSceneHookImplementation({
  id: "water-raster-integrity",
  evaluate: (context) => {
    const numericKeys = ["minimumFrontInterfacePixels", "minimumBackInterfacePixels",
      "maximumBackOnlyPixels", "boundsToleranceRatio", "minimumValidSamples",
      "minimumNegativeValidSamples", "minimumPositiveValidSamples"] as const;
    const numericValues = requiredNumbers(context.parameters, numericKeys);
    const expectedSurfaceGeometrySource = string(context.parameters, "expectedSurfaceGeometrySource");
    const expectedRetainedGeometrySource = string(context.parameters, "expectedRetainedGeometrySource");
    const booleanKeys = ["requireGlobalFineCrossingPublished", "requirePresentationFallbackInactive",
      "requireNonzeroAuthorityLatch", "requireNonemptyGeometry", "requireAllocatorCountsMatch",
      "requireGeometryWithinCapacity", "requireCleanFineCoarseTransition",
      "requireRetainedRasterIdentity"] as const;
    const booleans = Object.fromEntries(booleanKeys.map((key) => [key, boolean(context.parameters, key)]));
    if (!numericValues || !expectedSurfaceGeometrySource || !expectedRetainedGeometrySource
      || Object.values(booleans).some((value) => value === undefined)) {
      return parameterFailure("water-raster-integrity", [...numericKeys,
        "expectedSurfaceGeometrySource", "expectedRetainedGeometrySource", ...booleanKeys]);
    }
    return evaluateWaterRasterIntegrityDiagnostic({
      scene: context.scene, evidence: context.evidence, methods: context.selectedMethods,
      parameters: {
        ...context.parameters,
        ...numericValues,
        ...booleans,
        expectedSurfaceGeometrySource,
        expectedRetainedGeometrySource,
      } as unknown as WaterRasterIntegrityDiagnosticParameters,
    });
  },
});

const fluidSymmetry = defineSceneHookImplementation({
  id: "fluid-symmetry",
  evaluate: (context) => {
    const keys = ["maximumVolumeAbsoluteError", "maximumVelocityAbsoluteError_m_s",
      "maximumPressureAbsoluteError", "maximumRhsAbsoluteError", "maximumDiagonalAbsoluteError",
      "minimumCheckpointCount", "maximumWallContactStepSpread"] as const;
    const limits = requiredNumbers(context.parameters, keys);
    const requireExactTopology = boolean(context.parameters, "requireExactTopology");
    const requireAllWallsReached = boolean(context.parameters, "requireAllWallsReached");
    const requirePressureStageAudit = boolean(context.parameters, "requirePressureStageAudit") ?? false;
    const circularityEvaluationStart_s = numeric(context.parameters, "circularityEvaluationStart_s");
    const circularityEvaluationEnd_s = numeric(context.parameters, "circularityEvaluationEnd_s");
    const maximumAxisDiagonalFrontDifference_cells = numeric(
      context.parameters, "maximumAxisDiagonalFrontDifference_cells");
    const maximumRadialRmsDeviation_cells = numeric(
      context.parameters, "maximumRadialRmsDeviation_cells");
    const maximumRadialDeviation_cells = numeric(
      context.parameters, "maximumRadialDeviation_cells");
    const minimumCircularityAngularSamples = numeric(
      context.parameters, "minimumCircularityAngularSamples");
    const circularityValues = [circularityEvaluationStart_s, circularityEvaluationEnd_s,
      maximumAxisDiagonalFrontDifference_cells, maximumRadialRmsDeviation_cells,
      maximumRadialDeviation_cells, minimumCircularityAngularSamples];
    if (!limits || requireExactTopology === undefined || requireAllWallsReached === undefined) {
      return parameterFailure("fluid-symmetry", [...keys, "requireExactTopology", "requireAllWallsReached"]);
    }
    if (circularityValues.some((value) => value !== undefined)
      && (circularityValues.some((value) => value === undefined)
        || circularityEvaluationEnd_s! < circularityEvaluationStart_s!
        || maximumAxisDiagonalFrontDifference_cells! < 0
        || maximumRadialRmsDeviation_cells! < 0
        || maximumRadialDeviation_cells! < 0
        || !Number.isSafeInteger(minimumCircularityAngularSamples)
        || minimumCircularityAngularSamples! < 16)) {
      return parameterFailure("fluid-symmetry", ["circularityEvaluationStart_s",
        "circularityEvaluationEnd_s", "maximumAxisDiagonalFrontDifference_cells",
        "maximumRadialRmsDeviation_cells", "maximumRadialDeviation_cells",
        "minimumCircularityAngularSamples"]);
    }
    return context.selectedMethods.flatMap((method) => {
      const diagnostics = recordValue(context.getMethod(method)?.diagnostics);
      const checkpoints = diagnostics ? fieldCheckpoints(diagnostics) : [];
      const observations = checkpoints.flatMap((value, step) => {
        const checkpoint = recordValue(value);
        const observation = recordPath(checkpoint, "evidence", "fluid-symmetry");
        const time_s = numberPath(checkpoint, "time_s");
        return observation && time_s !== undefined ? [{ step: step + 1, time_s, observation }] : [];
      });
      const fieldLimits = [
        ["volume", limits.maximumVolumeAbsoluteError],
        ["velocity", limits.maximumVelocityAbsoluteError_m_s],
        ["pressure", limits.maximumPressureAbsoluteError],
        ["rhs", limits.maximumRhsAbsoluteError],
        ["diagonal", limits.maximumDiagonalAbsoluteError],
        ...(requirePressureStageAudit ? [
          ["initialResidual", limits.maximumRhsAbsoluteError],
          ["initialPreconditioned", limits.maximumPressureAbsoluteError],
          ["initialPreconditionedImage", limits.maximumPressureAbsoluteError],
          ["preconditionerPreSmoothed", limits.maximumPressureAbsoluteError],
          ["preconditionerZeroSmoothed", limits.maximumPressureAbsoluteError],
          ["preconditionerFirstOperatorImage", limits.maximumPressureAbsoluteError],
          ["preconditionerFirstSmoothed", limits.maximumPressureAbsoluteError],
          ["preconditionerInnerResidual", limits.maximumPressureAbsoluteError],
          ["preconditionerInnerCorrection", limits.maximumPressureAbsoluteError],
          ["preconditionerPostCorrected", limits.maximumPressureAbsoluteError],
        ] as const : []),
      ] as const;
      const findings: RuntimeDiagnosticFinding[] = [];
      findings.push(hookFinding({
        id: `${method}.checkpoint-count`, method,
        passed: observations.length >= limits.minimumCheckpointCount,
        message: observations.length >= limits.minimumCheckpointCount
          ? "every required symmetry checkpoint was collected"
          : "symmetry evidence ended before the required checkpoint count",
        expected: { minimum: limits.minimumCheckpointCount }, actual: observations.length,
      }));
      for (const [field, maximum] of fieldLimits) {
        const firstFailure = observations.find(({ observation }) => {
          const metrics = recordValue(observation[field]);
          const error = numberPath(metrics, "maximumAbsoluteError");
          const nonFinite = numberPath(metrics, "nonFiniteCount");
          return error === undefined || nonFinite === undefined || error > maximum || nonFinite > 0;
        });
        const maximumObserved = observations.reduce((value, { observation }) =>
          Math.max(value, numberPath(recordValue(observation[field]), "maximumAbsoluteError") ?? Infinity), 0);
        findings.push(hookFinding({
          id: `${method}.${field}`, method, passed: !firstFailure,
          message: firstFailure
            ? `${field} first lost D4 symmetry at step ${firstFailure.step}, t=${firstFailure.time_s.toFixed(3)} s`
            : `${field} retained D4 symmetry at every checkpoint`,
          expected: { maximumAbsoluteError: maximum, nonFiniteCount: 0 },
          actual: firstFailure ? {
            step: firstFailure.step, time_s: firstFailure.time_s,
            metrics: firstFailure.observation[field], maximumObserved,
          } : { maximumObserved },
        }));
      }
      const firstTopologyFailure = requireExactTopology ? observations.find(({ observation }) => {
        const topology = recordValue(observation.topology);
        return numberPath(topology, "exactMismatchCount") !== 0
          || numberPath(topology, "nonFiniteCount") !== 0;
      }) : undefined;
      findings.push(hookFinding({
        id: `${method}.topology`, method, passed: !firstTopologyFailure,
        message: firstTopologyFailure
          ? `adaptive topology first lost exact D4 symmetry at step ${firstTopologyFailure.step}, t=${firstTopologyFailure.time_s.toFixed(3)} s`
          : "adaptive topology retained exact D4 symmetry at every checkpoint",
        expected: { exactMismatchCount: 0, nonFiniteCount: 0 },
        actual: firstTopologyFailure ? {
          step: firstTopologyFailure.step, time_s: firstTopologyFailure.time_s,
          metrics: firstTopologyFailure.observation.topology,
        } : undefined,
      }));

      if (circularityEvaluationStart_s !== undefined
        && circularityEvaluationEnd_s !== undefined
        && maximumAxisDiagonalFrontDifference_cells !== undefined
        && maximumRadialRmsDeviation_cells !== undefined
        && maximumRadialDeviation_cells !== undefined
        && minimumCircularityAngularSamples !== undefined) {
        // The authored body starts square and the tank itself is square. Judge
        // the freely propagating front only after the initial rarefaction has
        // rounded it and before axis rays contact the walls; after contact a
        // circular contour is geometrically impossible inside this domain.
        const circularity = observations.filter(({ time_s }) =>
          time_s >= circularityEvaluationStart_s && time_s <= circularityEvaluationEnd_s);
        const firstFailure = circularity.find(({ observation }) => {
          const metric = recordValue(observation.frontCircularity);
          const difference = numberPath(metric, "axisLead_cells");
          const rms = numberPath(metric, "radialRmsDeviation_cells");
          const maximum = numberPath(metric, "radialMaximumDeviation_cells");
          const samples = numberPath(metric, "angularSampleCount");
          return difference === undefined
            || rms === undefined || maximum === undefined || samples === undefined
            || Math.abs(difference) > maximumAxisDiagonalFrontDifference_cells
            || rms > maximumRadialRmsDeviation_cells
            || maximum > maximumRadialDeviation_cells
            || samples < minimumCircularityAngularSamples;
        });
        const maximumObserved = circularity.reduce((value, { observation }) => {
          const metric = recordValue(observation.frontCircularity);
          const difference = numberPath(metric, "axisLead_cells");
          const rms = numberPath(metric, "radialRmsDeviation_cells");
          const maximum = numberPath(metric, "radialMaximumDeviation_cells");
          return {
            axisLead_cells: Math.max(value.axisLead_cells,
              difference === undefined ? Infinity : Math.abs(difference)),
            radialRmsDeviation_cells: Math.max(value.radialRmsDeviation_cells,
              rms ?? Infinity),
            radialMaximumDeviation_cells: Math.max(value.radialMaximumDeviation_cells,
              maximum ?? Infinity),
          };
        }, { axisLead_cells: 0, radialRmsDeviation_cells: 0,
          radialMaximumDeviation_cells: 0 });
        const passed = circularity.length > 0 && !firstFailure;
        findings.push(hookFinding({
          id: `${method}.front-circularity`, method, passed,
          message: passed
            ? "the freely propagating dam front remained circular"
            : firstFailure
              ? `the axis and diagonal fronts first diverged at step ${firstFailure.step}, t=${firstFailure.time_s.toFixed(3)} s`
              : "no dam-front circularity checkpoint was collected in the declared window",
          expected: { evaluationStart_s: circularityEvaluationStart_s,
            evaluationEnd_s: circularityEvaluationEnd_s,
            minimumAngularSamples: minimumCircularityAngularSamples,
            maximumAbsoluteAxisLead_cells: maximumAxisDiagonalFrontDifference_cells,
            maximumRadialRmsDeviation_cells,
            maximumRadialDeviation_cells },
          actual: firstFailure ? {
            step: firstFailure.step, time_s: firstFailure.time_s,
            metrics: firstFailure.observation.frontCircularity, maximumObserved,
          } : { checkpoints: circularity.length, maximumObserved },
        }));
      }

      const wallNames = ["negativeX", "positiveX", "negativeZ", "positiveZ"] as const;
      const contactSteps = Object.fromEntries(wallNames.map((wall) => {
        const contact = observations.find(({ observation }) =>
          recordPath(observation, "walls", wall)?.touched === true);
        return [wall, contact?.step];
      })) as Record<typeof wallNames[number], number | undefined>;
      const reachedSteps = Object.values(contactSteps).filter((value): value is number => value !== undefined);
      const spread = reachedSteps.length === 4 ? Math.max(...reachedSteps) - Math.min(...reachedSteps) : Infinity;
      const wallsPassed = requireAllWallsReached
        ? reachedSteps.length === 4 && spread <= limits.maximumWallContactStepSpread
        : reachedSteps.length < 4 || spread <= limits.maximumWallContactStepSpread;
      findings.push(hookFinding({
        id: `${method}.wall-contact`, method, passed: wallsPassed,
        message: wallsPassed ? (reachedSteps.length === 4
          ? "all four walls were reached on the same accepted step"
          : "wall contact is not required by this diagnostic lane")
          : "the four wall-contact steps are absent or asymmetric",
        expected: { reachedWalls: requireAllWallsReached ? 4 : undefined,
          maximumStepSpread: limits.maximumWallContactStepSpread },
        actual: { contactSteps, stepSpread: spread },
      }));
      return findings;
    });
  },
});

export const sceneCustomHookImplementations = Object.freeze({
  "free-fall-contact": freeFallContact,
  "rigid-coupling-oracle": rigidCouplingOracle,
  "garden-source-brick-migration": gardenSourceBrickMigration,
  "minimal-dam-motion": minimalDamMotion,
  "dam-break-velocity-parity": damBreakVelocityParity,
  "brick-quad-coverage": brickQuadCoverage,
  "ocean-wave-profile": oceanWaveProfile,
  "hose-jet-drift": hoseJetDrift,
  "dam-break-perturbed-cadence": damBreakPerturbedCadence,
  "water-raster-integrity": waterRasterIntegrity,
  "fluid-symmetry": fluidSymmetry,
}) satisfies SceneHookRegistry;

const settling = defineDiagnosticPackImplementation({
  id: "settling",
  evaluate: (context) => {
    const keys = ["maximumFinalSampledExactVolumeDrift", "maximumNormalizedNetProjectionEnergyDelta",
      "maximumNormalizedLateMechanicalEnergySlopePerSecond", "maximumLateToMiddleKineticEnvelopeRatio",
      "damBreakMaximumDriftSignChanges", "damBreakMaximumLatePeakToPeakDrift"] as const;
    const values = requiredNumbers(context.parameters, keys);
    if (!values) return parameterFailure("settling", keys);
    return evaluateSettlingDiagnostic({
      scene: context.scene, evidence: context.evidence,
      parameters: {
        expectAsymptoticRest: boolean(context.parameters, "expectAsymptoticRest") ?? true,
        maximumFinalExactVolumeDrift: values.maximumFinalSampledExactVolumeDrift,
        maximumNormalizedNetProjectionEnergyDelta: values.maximumNormalizedNetProjectionEnergyDelta,
        maximumNormalizedLateMechanicalEnergySlopePerSecond: values.maximumNormalizedLateMechanicalEnergySlopePerSecond,
        maximumLateToMiddleKineticEnvelopeRatio: values.maximumLateToMiddleKineticEnvelopeRatio,
        maximumDamBreakDriftSignChanges: values.damBreakMaximumDriftSignChanges,
        maximumDamBreakLatePeakToPeakDrift: values.damBreakMaximumLatePeakToPeakDrift,
      } satisfies SettlingDiagnosticParameters,
      methods: context.selectedMethods,
    });
  },
});

const quadtreeDamParity = defineDiagnosticPackImplementation({
  id: "quadtree-dam-parity",
  evaluate: (context) => {
    const keys = ["minimumSimulatedTime_s", "expectedRebuildCadenceSteps", "inlineRebuildCompletionFraction",
      "maximumBlockedRebuildFrames", "maximumWallToGpuTimeRatio",
      "maximumNonFiniteVelocitySamples", "maximumPeakLiquidSpeed_m_s", "maximumPeakComponentCfl",
      "maximumProjectionEnergyRatio", "maximumExactVolumeDrift",
      "maximumCompressionRatio", "minimumDominantComponentFraction", "maximumFinalComponents",
      "minimumFront_m", "kineticEnergyEvaluationStart_s", "minimumPeakKineticEnergyProxy"] as const;
    const values = requiredNumbers(context.parameters, keys);
    const method = string(context.parameters, "method") ?? context.selectedMethods[0];
    const maximumPressureRelativeResidual = numeric(context.parameters, "maximumPressureRelativeResidual");
    const requireStabilitySampleEveryStep = boolean(context.parameters, "requireStabilitySampleEveryStep");
    if (!values || !method || maximumPressureRelativeResidual === undefined
      || requireStabilitySampleEveryStep === undefined) {
      return parameterFailure("quadtree-dam-parity", [...keys, "maximumPressureRelativeResidual",
        "requireStabilitySampleEveryStep", "method"]);
    }
    const parameters: QuadtreeDamParityDiagnosticParameters = {
      method,
      minimumSimulatedTime_s: values.minimumSimulatedTime_s,
      maximumPressureRelativeResidual,
      rebuildCadenceSteps: values.expectedRebuildCadenceSteps,
      inlineRebuildCompletionFraction: values.inlineRebuildCompletionFraction,
      maximumBlockedRebuildFrames: values.maximumBlockedRebuildFrames,
      maximumWallToGpuRatio: values.maximumWallToGpuTimeRatio,
      requireStabilitySampleEveryStep,
      maximumNonFiniteVelocityCount: values.maximumNonFiniteVelocitySamples,
      maximumPeakLiquidSpeed_m_s: values.maximumPeakLiquidSpeed_m_s,
      maximumPeakComponentCfl: values.maximumPeakComponentCfl,
      maximumProjectionEnergyRatio: values.maximumProjectionEnergyRatio,
      maximumEnvelopePressureRelativeResidual: maximumPressureRelativeResidual,
      maximumExactVolumeDrift: values.maximumExactVolumeDrift,
      maximumCompressionRatio: values.maximumCompressionRatio,
      minimumDominantComponentFraction: values.minimumDominantComponentFraction,
      maximumFinalComponentCount: values.maximumFinalComponents,
      minimumFront_m: values.minimumFront_m,
      kineticGateAfter_s: values.kineticEnergyEvaluationStart_s,
      minimumPeakKineticEnergyProxy: values.minimumPeakKineticEnergyProxy,
      ...(string(context.parameters, "referenceMethod") ? { referenceMethod: string(context.parameters, "referenceMethod") } : {}),
      ...(numeric(context.parameters, "minimumKineticEnergyRatioToReference") === undefined ? {}
        : { minimumKineticEnergyRatioToReference: numeric(context.parameters, "minimumKineticEnergyRatioToReference") }),
    };
    return evaluateQuadtreeDamParityDiagnostic({ scene: context.scene, evidence: context.evidence, parameters });
  },
});

export const sceneCustomDiagnosticPackImplementations = Object.freeze({
  settling,
  "quadtree-dam-parity": quadtreeDamParity,
}) satisfies DiagnosticPackRegistry;

export default sceneCustomHookImplementations;
