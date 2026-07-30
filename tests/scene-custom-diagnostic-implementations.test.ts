import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene } from "../lib/model";
import {
  sceneCustomDiagnosticPackImplementations,
  sceneCustomHookImplementations,
} from "../lib/scene-custom-diagnostic-implementations";
import { evaluateFieldVelocityParityDiagnostic } from "../lib/scene-field-velocity-parity-diagnostic";
import { evaluateFreeFallContactDiagnostic } from "../lib/scene-free-fall-contact-diagnostic";
import { evaluateGardenBrickMigrationDiagnostic } from "../lib/scene-garden-brick-migration-diagnostic";
import { evaluateMinimalDamMotionDiagnostic } from "../lib/scene-minimal-dam-diagnostic";
import { evaluateQuadtreeDamParityDiagnostic } from "../lib/scene-quadtree-dam-parity-diagnostic";
import { evaluateSettlingDiagnostic } from "../lib/scene-settling-diagnostic";
import { evaluateWaterRasterIntegrityDiagnostic } from "../lib/scene-water-raster-integrity-diagnostic";
import type { SceneDiagnosticEvidence } from "../lib/scene-diagnostics";

function evidence(methods: Record<string, Record<string, unknown>>): SceneDiagnosticEvidence {
  return { methods: Object.fromEntries(Object.entries(methods)
    .map(([method, diagnostics]) => [method, { diagnostics }])) };
}

test("custom implementation registries cover every currently declared hook and migrated pack", () => {
  assert.deepEqual(Object.keys(sceneCustomHookImplementations).sort(), [
    "brick-quad-coverage",
    "dam-break-perturbed-cadence",
    "dam-break-velocity-parity",
    "free-fall-contact",
    "garden-source-brick-migration",
    "hose-jet-drift",
    "minimal-dam-motion",
    "ocean-wave-profile",
    "water-raster-integrity",
  ]);
  assert.deepEqual(Object.keys(sceneCustomDiagnosticPackImplementations).sort(), [
    "quadtree-dam-parity",
    "settling",
  ]);
});

test("free-fall contact evaluates the analytic drop and both ceiling representations", () => {
  const scene = cloneScene(defaultScene);
  scene.container.height_m = 0.8;
  scene.fluid.gravity_m_s2 = { x: 0, y: -9.81, z: 0 };
  const grid = [16, 16, 16] as const;
  const cell_m = scene.container.height_m / grid[1];
  const checkpoints = [0.1, 0.15, 0.2].map((time_s) => ({
    time_s,
    centroidY_cells: grid[1] - 4 - 0.5 * 9.81 * time_s ** 2 / cell_m,
    ceilingWetCells: 0,
    ceilingContactPixels: 0,
  }));
  const findings = evaluateFreeFallContactDiagnostic({
    scene,
    evidence: evidence({ octree: { run: { simulatedTime_s: 0.3 }, field: { grid, checkpoints } } }),
    methods: ["octree"],
    parameters: {
      minimumRunTime_s: 0.3, evaluationStart_s: 0.1, impactTime_s: 0.29, preImpactMargin_s: 0.02,
      minimumMeasuredToAnalyticDropRatio: 0.6, maximumMeasuredToAnalyticDropRatio: 1.45,
      maximumDropHeadroom_cells: 0.5, minimumPreImpactCheckpoints: 3, releaseCheckTime_s: 0.2,
      maximumCeilingWetCellsAfterRelease: 0, maximumCeilingPixelsAfterRelease: 0,
    },
  });
  assert.ok(findings.length >= 9);
  assert.equal(findings.every(({ passed }) => passed), true);

  checkpoints[2].ceilingWetCells = 2;
  const failed = evaluateFreeFallContactDiagnostic({
    scene,
    evidence: evidence({ octree: { run: { simulatedTime_s: 0.3 }, field: { grid, checkpoints } } }),
    methods: ["octree"],
    parameters: {
      minimumRunTime_s: 0.3, evaluationStart_s: 0.1, impactTime_s: 0.29, preImpactMargin_s: 0.02,
      minimumMeasuredToAnalyticDropRatio: 0.6, maximumMeasuredToAnalyticDropRatio: 1.45,
      maximumDropHeadroom_cells: 0.5, minimumPreImpactCheckpoints: 3, releaseCheckTime_s: 0.2,
      maximumCeilingWetCellsAfterRelease: 0, maximumCeilingPixelsAfterRelease: 0,
    },
  });
  assert.equal(failed.find(({ id }) => id === "octree.ceiling-wet.2")?.passed, false);
});

test("free-fall corner seam shortfall is compared with contact-free liquid", () => {
  const scene = cloneScene(defaultScene);
  scene.container.height_m = 0.8;
  const grid = [16, 16, 16] as const;
  const checkpoint = {
    time_s: 0.2,
    centroidY_cells: 8,
    ceilingWetCells: 0,
    ceilingContactPixels: 0,
    attribution: { velocityByContact: [
      { contacts: 0, meanShortfallFraction: 0.1 },
      { contacts: 1, meanShortfallFraction: 0.11 },
      { contacts: 2, meanShortfallFraction: 0.14 },
      { contacts: 3, meanShortfallFraction: 0.15 },
    ] },
  };
  const parameters = {
    minimumRunTime_s: 0.1, evaluationStart_s: 1, impactTime_s: 0.29, preImpactMargin_s: 0.02,
    minimumMeasuredToAnalyticDropRatio: 0.6, maximumMeasuredToAnalyticDropRatio: 1.45,
    maximumDropHeadroom_cells: 0.5, minimumPreImpactCheckpoints: 0, releaseCheckTime_s: 1,
    maximumCeilingWetCellsAfterRelease: 0, maximumCeilingPixelsAfterRelease: 0,
    includeCornerSeams: true, seamEvaluationStart_s: 0.02, maximumSeamShortfallExcess: 0.05,
  };
  const passed = evaluateFreeFallContactDiagnostic({
    scene, evidence: evidence({ octree: { run: { simulatedTime_s: 0.2 },
      field: { grid, checkpoints: [checkpoint] } } }), methods: ["octree"], parameters,
  });
  assert.equal(passed.find(({ id }) => id === "octree.seam-shortfall.0")?.passed, true);
  checkpoint.attribution.velocityByContact[3].meanShortfallFraction = 0.2;
  const failed = evaluateFreeFallContactDiagnostic({
    scene, evidence: evidence({ octree: { run: { simulatedTime_s: 0.2 },
      field: { grid, checkpoints: [checkpoint] } } }), methods: ["octree"], parameters,
  });
  const finding = failed.find(({ id }) => id === "octree.seam-shortfall.0");
  assert.equal(finding?.passed, false);
  assert.deepEqual(finding?.actual, { baseline: 0.1, seam: 0.2, excess: 0.1 });
});

test("mid-air free-fall uses the authored seed layer and stops seam attribution before impact", () => {
  const scene = cloneScene(defaultScene);
  scene.container.height_m = 1.2;
  scene.fluid.initialBrickSeeds_m = [{ x: -0.55, y: 0.6, z: -0.55 }];
  const grid = [24, 24, 24] as const;
  const cell_m = scene.container.height_m / grid[1];
  const checkpoint = (time_s: number) => ({
    time_s,
    centroidY_cells: 12 - 0.5 * 9.81 * time_s ** 2 / cell_m,
    ceilingWetCells: 0,
    ceilingContactPixels: 0,
    attribution: { velocityByContact: [
      { contacts: 0, meanShortfallFraction: 0 },
      { contacts: 2, meanShortfallFraction: time_s < 0.27 ? 0 : 1 },
    ] },
  });
  const findings = evaluateFreeFallContactDiagnostic({
    scene, methods: ["octree"],
    evidence: evidence({ octree: { run: { simulatedTime_s: 0.3 },
      field: { grid, checkpoints: [checkpoint(0.1), checkpoint(0.2), checkpoint(0.28)] } } }),
    parameters: {
      minimumRunTime_s: 0.3, evaluationStart_s: 0.1, impactTime_s: 0.29,
      preImpactMargin_s: 0.02, minimumMeasuredToAnalyticDropRatio: 0.6,
      maximumMeasuredToAnalyticDropRatio: 1.45, maximumDropHeadroom_cells: 0.5,
      minimumPreImpactCheckpoints: 2, releaseCheckTime_s: 0.2,
      maximumCeilingWetCellsAfterRelease: 0, maximumCeilingPixelsAfterRelease: 0,
      initialBrickSize_cells: 8, initialCentroidHalfBrickOffset_cells: 4,
      includeCornerSeams: true, seamEvaluationStart_s: 0.02,
      maximumSeamShortfallExcess: 0.05,
    },
  });
  assert.equal(findings.every(({ passed }) => passed), true);
  assert.equal(findings.some(({ id }) => id === "octree.seam-shortfall.2"), false,
    "post-impact wall/floor contact is not a free-fall seam comparison");
});

test("garden migration reports source evacuation and capacity independently", () => {
  const parameters = {
    initialCoreBricks: 1, evaluateAfter_s: 1, minimumFinalCoreBricks: 2,
    sourceFluidVoxelsAtEnd: 0, sourceCoreResidencyAtEnd: false,
  };
  const diagnostics = {
    run: { simulatedTime_s: 1 },
    sparse: {
      initialFluidBricks: { core: 1 },
      finalPublication: { fluidBrickResidentCount: 4, fluidBrickCapacity: 16,
        fluidBrickCoreCount: 3, sourceBrickFluidVoxelCount: 0, sourceBrickResidency: "halo" },
    },
  };
  const passed = evaluateGardenBrickMigrationDiagnostic({
    scene: defaultScene, evidence: evidence({ octree: diagnostics }), parameters, methods: ["octree"],
  });
  assert.equal(passed.every(({ passed: ok }) => ok), true);
  diagnostics.sparse.finalPublication.sourceBrickFluidVoxelCount = 7;
  const failed = evaluateGardenBrickMigrationDiagnostic({
    scene: defaultScene, evidence: evidence({ octree: diagnostics }), parameters, methods: ["octree"],
  });
  assert.equal(failed.find(({ id }) => id === "octree.source-evacuated")?.passed, false);
});

test("minimal dam motion uses only normalized stability, raster, and energy evidence", () => {
  const scene = cloneScene(defaultScene);
  const diagnostics = {
    run: { simulatedTime_s: 1 },
    stability: { peakLiquidSpeed_m_s: 1 },
    raster: {
      initial: { frontInterfaceBounds_m: [[-0.4, 0, -0.4], [0, 0.4, 0]] },
      final: { frontInterfaceBounds_m: [[-0.4, 0, -0.4], [0.2, 0.4, 0.1]] },
    },
    energy: { checkpoints: [{ mechanicalEnergyRetentionRatio: 1.2, maximumLiquidComponentSpeed_m_s: 2 }] },
  };
  const findings = evaluateMinimalDamMotionDiagnostic({
    scene, evidence: evidence({ octree: diagnostics }), methods: ["octree"],
    parameters: { minimumPeakSpeed_m_s: 0.1, minimumLateralSpread_m: 0.05,
      maximumMechanicalEnergyRetention: 1.5, maximumRitterCelerityRatio: 1.35 },
  });
  assert.equal(findings.every(({ passed }) => passed), true);
  diagnostics.raster.final.frontInterfaceBounds_m[1][0] = 0.01;
  diagnostics.raster.final.frontInterfaceBounds_m[1][2] = 0.01;
  const failed = evaluateMinimalDamMotionDiagnostic({
    scene, evidence: evidence({ octree: diagnostics }), methods: ["octree"],
    parameters: { minimumPeakSpeed_m_s: 0.1, minimumLateralSpread_m: 0.05 },
  });
  assert.equal(failed.find(({ id }) => id === "octree.lateral-spread")?.passed, false);
});

test("settling gates are parameter-driven and retain dam-break oscillation checks", () => {
  const scene = cloneScene(defaultScene);
  scene.fluid.initialCondition = "dam-break";
  const parameters = {
    maximumFinalExactVolumeDrift: 0.01,
    maximumNormalizedNetProjectionEnergyDelta: 0.01,
    maximumNormalizedLateMechanicalEnergySlopePerSecond: 0.001,
    maximumLateToMiddleKineticEnvelopeRatio: 1,
    maximumDamBreakDriftSignChanges: 3,
    maximumDamBreakLatePeakToPeakDrift: 0.005,
  };
  const summary = { finalSampledExactVolumeDrift: 0.002, normalizedNetProjectionEnergyDelta: -0.01,
    normalizedLateMechanicalEnergySlopePerSecond: -0.002, lateToMiddleKineticEnvelopeRatio: 0.8,
    driftSignChanges: 2, latePeakToPeakDrift: 0.003 };
  const passed = evaluateSettlingDiagnostic({
    scene, evidence: evidence({ uniform: { energy: { summary } } }), parameters, methods: ["uniform"],
  });
  assert.equal(passed.every(({ passed: ok }) => ok), true);
  summary.latePeakToPeakDrift = 0.02;
  const failed = evaluateSettlingDiagnostic({
    scene, evidence: evidence({ uniform: { energy: { summary } } }), parameters, methods: ["uniform"],
  });
  assert.equal(failed.find(({ id }) => id === "uniform.late-drift-range")?.passed, false);
});

test("field and velocity parity produces stable per-metric findings", () => {
  const grid = [2, 1, 1] as const;
  const field = new Float32Array([1, 0]);
  const velocity = new Float32Array([1, 0, 0, 0, 0, 0]);
  const method = (compact: boolean) => ({
    field: { grid, final: { field }, checkpoints: [{ time_s: 1, field }] },
    stability: { peakLiquidSpeed_m_s: 1 },
    terminalEvidence: { "collocated-velocity": { field: velocity, volume: field,
      compactRaster: compact ? { publicationValid: true, coveredCells: 2, overlapCells: 0, invalidRows: 0 } : undefined } },
  });
  const findings = evaluateFieldVelocityParityDiagnostic({
    scene: defaultScene,
    evidence: evidence({ octree: method(true), "tall-cell": method(false) }),
    parameters: { candidateMethod: "octree", referenceMethod: "tall-cell",
      limits: { maximumWeightedRelativeL2: 1, minimumCosineSimilarity: 0.5,
        minimumEnergyRatio: 0.25, maximumEnergyRatio: 4, minimumPeakRatio: 0.5, maximumPeakRatio: 2 },
      minimumWetIntersectionOverUnion: 0.6, maximumCentroidDistanceCells: 6,
      checkpointTimeTolerance_s: 0.01 },
  });
  assert.equal(findings.every(({ passed }) => passed), true);
  assert.equal(new Set(findings.map(({ id }) => id)).size, findings.length);
});

test("quadtree dam parity evaluates cadence, topology, stability, and activity from normalized namespaces", () => {
  const parameters = {
    method: "quadtree-tall-cell", minimumSimulatedTime_s: 0.2,
    maximumPressureRelativeResidual: 1e-4, rebuildCadenceSteps: 1,
    inlineRebuildCompletionFraction: 0.9, maximumBlockedRebuildFrames: 0,
    maximumWallToGpuRatio: 2, requireStabilitySampleEveryStep: true,
    maximumNonFiniteVelocityCount: 0, maximumPeakLiquidSpeed_m_s: 5,
    maximumPeakComponentCfl: 1, maximumProjectionEnergyRatio: 1.1,
    maximumEnvelopePressureRelativeResidual: 1e-4, maximumExactVolumeDrift: 0.02,
    maximumCompressionRatio: 0.25, minimumDominantComponentFraction: 0.995,
    maximumFinalComponentCount: 10, minimumFront_m: -0.005,
    kineticGateAfter_s: 0.5, minimumPeakKineticEnergyProxy: 0.4,
  };
  const diagnostics = {
    run: { simulatedTime_s: 0.5, steps: 10, simulationWall_ms: 10 },
    solver: { pressureRelativeResidual: 1e-5, quadtreeRebuildCadenceSteps: 1,
      quadtreeTopologyStaleLimit: 0, quadtreeRebuildCompletedCount: 10,
      quadtreeRebuildBlockedFrames: 0, physicsTrace: { total_ms: 1 }, compressionRatio: 0.2,
      front_m: 0.1 },
    stability: { sampledSteps: 10, nonFiniteVelocityCount: 0, peakLiquidSpeed_m_s: 2,
      peakComponentCfl: 0.8, maximumProjectionEnergyRatio: 1,
      maximumPressureRelativeResidual: 1e-5, maximumExactVolumeDrift: 0.01,
      minimumDominantComponentFraction: 0.999, peakKineticEnergyProxy: 0.5 },
    field: { final: { summary: { componentCount: 2 } } },
  };
  const findings = evaluateQuadtreeDamParityDiagnostic({
    scene: defaultScene, evidence: evidence({ "quadtree-tall-cell": diagnostics }), parameters,
  });
  assert.equal(findings.every(({ passed }) => passed), true);
});

test("water raster integrity verifies clean retained geometry without GPU access", () => {
  const generation = { generation: 7, validSamples: 10, negativeValidSamples: 5, positiveValidSamples: 5 };
  const raster = {
    frontInterfacePixels: 10, backInterfacePixels: 8, backOnlyInterfacePixels: 0,
    reverseView: { frontInterfacePixels: 9, backInterfacePixels: 7, backOnlyInterfacePixels: 0 },
    surfaceGeometrySource: "global-fine-coarse", globalFineCrossingPublished: true,
    presentationFallbackActive: false, globalFineAuthorityLatch: 1,
    vertexCount: 12, vertexAllocator: 12, vertexCapacity: 100,
    activeCubeCount: 3, activeCubeCapacity: 20,
    frontInterfaceBounds_m: [[-0.1, 0, -0.1], [0.1, 0.2, 0.1]],
    frontInterfaceHash: 11, backInterfaceHash: 12,
    globalFineAuthorityTransition: { cleanFineCoarseRequired: true, validGeneration: 7,
      retainedGeometrySource: "retained-previous", retainedFrontInterfacePixels: 10,
      retainedBackInterfacePixels: 8, retainedFrontInterfaceHash: 11, retainedBackInterfaceHash: 12 },
  };
  const parameters = {
    minimumFrontInterfacePixels: 1, minimumBackInterfacePixels: 1, maximumBackOnlyPixels: 0,
    boundsToleranceRatio: 1e-4, expectedSurfaceGeometrySource: "global-fine-coarse",
    requireGlobalFineCrossingPublished: true, requirePresentationFallbackInactive: true,
    requireNonzeroAuthorityLatch: true, requireNonemptyGeometry: true,
    requireAllocatorCountsMatch: true, requireGeometryWithinCapacity: true,
    requireCleanFineCoarseTransition: true, expectedRetainedGeometrySource: "retained-previous",
    requireRetainedRasterIdentity: true, minimumValidSamples: 1,
    minimumNegativeValidSamples: 1, minimumPositiveValidSamples: 1,
  };
  const findings = evaluateWaterRasterIntegrityDiagnostic({
    scene: defaultScene, methods: ["octree"], parameters,
    evidence: evidence({ octree: { raster: { initial: raster, final: raster,
      initialGeneration: generation, finalGeneration: generation } } }),
  });
  assert.equal(findings.every(({ passed }) => passed), true);
});
