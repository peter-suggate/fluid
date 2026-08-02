import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { validateScene } from "../lib/model";
import {
  getSceneWebGPUSmokeLane,
  getSceneWebGPUSmokeSuite,
  isSceneWebGPUSmokeId,
  sceneWebGPUSmokeIds,
  sceneWebGPUSmokeSuites,
} from "../lib/scene-webgpu-smoke-catalog";
import type {
  SceneWebGPUDiagnosticHook,
  SceneWebGPUDiagnosticPack,
  SceneWebGPUSmokeLane,
} from "../lib/scene-webgpu-smoke";

function diagnostic(lane: SceneWebGPUSmokeLane, id: SceneWebGPUDiagnosticPack["id"]): SceneWebGPUDiagnosticPack {
  const value = lane.diagnostics.find((entry) => entry.id === id);
  assert.ok(value, `${lane.id} must declare ${id}`);
  return value;
}

function hook(lane: SceneWebGPUSmokeLane, id: SceneWebGPUDiagnosticHook["id"]): SceneWebGPUDiagnosticHook {
  const value = lane.hooks.find((entry) => entry.id === id);
  assert.ok(value, `${lane.id} must declare ${id}`);
  return value;
}

test("scene WebGPU catalog is complete, immutable, and constructs valid scenes", () => {
  assert.deepEqual(Object.keys(sceneWebGPUSmokeSuites), sceneWebGPUSmokeIds);
  assert.equal(new Set(sceneWebGPUSmokeIds).size, sceneWebGPUSmokeIds.length);
  assert.equal(isSceneWebGPUSmokeId("ocean-seiche"), true);
  assert.equal(isSceneWebGPUSmokeId("not-a-scene"), false);

  for (const id of sceneWebGPUSmokeIds) {
    const suite = getSceneWebGPUSmokeSuite(id);
    const authoredScene = suite.createScene();
    assert.deepEqual(validateScene(authoredScene), [], `${id} must construct a valid scene`);
    assert.ok(suite.lanes[suite.defaultLane], `${id} must resolve its default lane`);
    assert.ok(Object.isFrozen(suite), `${id} suite must be frozen`);
    assert.ok(Object.isFrozen(suite.lanes), `${id} lane map must be frozen`);
    for (const [laneId, lane] of Object.entries(suite.lanes)) {
      assert.equal(lane.id, laneId);
      assert.ok(lane.methods.length > 0);
      assert.ok(lane.stop.simulatedTime_s > 0);
      assert.ok(lane.oracle.matchedSteps > 0);
      assert.ok(lane.diagnostics.some(({ id: diagnosticId }) => diagnosticId === "core-webgpu-health"));
      assert.ok(lane.acceptance.some(({ id: checkId }) => checkId === "validation-clean"));
      assert.ok(Object.isFrozen(lane));
      assert.ok(Object.isFrozen(lane.methods));
      assert.ok(Object.isFrozen(lane.collect));
      for (const declaration of lane.diagnostics) {
        if (declaration.id !== "performance") {
          assert.ok(declaration.parameters && Object.keys(declaration.parameters).length > 0,
            `${id}/${laneId} ${declaration.id} must own its diagnostic parameters`);
        }
      }
      for (const declaration of lane.hooks) {
        assert.ok(declaration.parameters && Object.keys(declaration.parameters).length > 0,
          `${id}/${laneId} ${declaration.id} must own its hook parameters`);
      }
    }
  }
});

test("generic health and backend packs declare every reusable acceptance constant", () => {
  const lane = getSceneWebGPUSmokeLane("dam-break-ui");
  assert.deepEqual(diagnostic(lane, "core-webgpu-health").parameters, {
    maximumValidationErrors: 0,
    maximumNonFiniteValues: 0,
    requireFiniteMaximumSpeed: true,
    requireCommonGridAcrossMethods: true,
    structuredMovingSceneMinimumComponentCfl: 0,
    exactStepTimeTolerance_s: 1e-9,
    pressureWarmupStepsWithoutResidual: 2,
  });
  const topology = diagnostic(lane, "volume-and-topology").parameters!;
  assert.equal(topology.minimumStoredDensity, -0.01);
  assert.equal(topology.maximumStoredDensity, 1.5);
  assert.equal(topology.quadtreeMaximumPressureRelativeResidual, 1e-4);
  assert.equal(topology.quadtreeMaximumPressureRmsResidual, 1e-5);
  assert.equal(topology.quadtreePressureAcceptance, "relative-or-rms");
  assert.equal(topology.sparseSphereMaterialMaximumVoxelCountExclusive, 10_000);
  assert.equal(topology.sparseFluidColorTolerance, 1e-6);
  assert.equal(topology.wetFlipCensusImpliedSpeed_m_s, 12.5);

  const octree = diagnostic(lane, "octree-authority").parameters!;
  assert.equal(octree.expectedGridKind, "octree");
  assert.equal(octree.pressureSolverNameIncludes, "persistent executor");
  assert.equal(octree.maximumPressureRelativeResidualSquared, 1e-8);
  assert.equal(octree.damBreakMaximumProjectionEnergyRatioWithoutAdaptiveTransport, 1.1);
  assert.deepEqual(octree.damBreakImpactWindow_s, [0.9, 1.3]);
  assert.equal(octree.requireGlobalFineFactorMatchesMethodOverride, true);
});

test("power, global-fine, raster, and scene hooks contain no scenario defaults", () => {
  const hydrostatic = getSceneWebGPUSmokeLane("hydrostatic-power-two-level");
  const exhaustive = diagnostic(hydrostatic, "exhaustive-power-generation").parameters!;
  assert.equal(exhaustive.minimumStepsWithoutExactStop, 50);
  assert.equal(exhaustive.maximumExactVolumeDrift, 1e-4);
  assert.equal(exhaustive.maximumPressureRelativeResidual, 1e-4);

  const minimal = getSceneWebGPUSmokeLane("minimal-power-dam-break");
  const publication = diagnostic(minimal, "global-fine-publication").parameters!;
  assert.equal(publication.checkpointCoarseState, 0x8000_0000);
  assert.equal(publication.minimumFinalActivePages, 1);
  assert.equal(publication.minimumNegativeValidSamples, 1);
  assert.equal(publication.topologyFinalizeReason, 0);

  const raster = diagnostic(minimal, "authoritative-water-raster").parameters!;
  assert.equal(raster.expectedSurfaceGeometrySource, "global-fine-coarse");
  assert.equal(raster.maximumBackOnlyPixels, 2);
  assert.equal(raster.preImpactHoleLimitPixels, 2);
  assert.equal(raster.terraceEdgeFractionMaximum, 0.12);
  assert.equal(raster.ceilingWetEvaluationStart_s, 1.5);
  assert.equal(raster.ceilingContactEvaluationStart_s, 1.4);
  assert.deepEqual(raster.ceilingWetCellLimits, [
    { before_s: 1.6, maximum: 9 }, { before_s: 1.7, maximum: 5 },
    { before_s: 2, maximum: 1 }, { maximum: 3 },
  ]);

  const motion = hook(minimal, "minimal-dam-motion").parameters!;
  assert.equal(motion.maximumMechanicalEnergyRetention, 1.5);
  assert.equal(motion.maximumRitterCelerityRatio, 1.35);
  assert.equal(motion.minimumRitterColumnHeightFraction, 0.92);
  assert.equal(motion.ritterCelerityMultiplier, 2);

  const freeFall = hook(getSceneWebGPUSmokeLane("corner-brick-drop"), "free-fall-contact").parameters!;
  assert.equal(freeFall.minimumMeasuredToAnalyticDropRatio, 0.6);
  assert.equal(freeFall.maximumMeasuredToAnalyticDropRatio, 1.45);
  assert.equal(freeFall.maximumSeamShortfallExcess, 0.05);
  assert.equal(freeFall.minimumMeaningfulColumnAmount_cells, 1);
  assert.equal(freeFall.initialBrickSize_cells, 8);

  const velocity = hook(getSceneWebGPUSmokeLane("dam-break-ui"), "dam-break-velocity-parity").parameters!;
  assert.equal(velocity.candidateMethod, "octree");
  assert.equal(velocity.referenceMethod, "tall-cell");
  assert.equal(velocity.minimumSharedVolumeWeight, 1e-4);
  assert.equal(velocity.wetThreshold, 0.5);
});

test("inflow, migration, brick coverage, hose, and ocean policies are scene-authored", () => {
  const hose = getSceneWebGPUSmokeLane("hose-tank", "drift");
  assert.equal(hose.acceptance.some(({ id }) => id === "represented-volume-drift"), false);
  const inflow = diagnostic(hose, "inflow-activity").parameters!;
  assert.equal(inflow.minimumSourceAdjustedRepresentedVolumeRatio, 0.99);
  const drift = hook(hose, "hose-jet-drift").parameters!;
  assert.equal(drift.minimumAirborneAxialRetentionRatio, 0.9);
  assert.equal(drift.maximumBallisticGravityVelocityRatio, 3.1);

  const migration = hook(getSceneWebGPUSmokeLane("garden-dam-break", "migration"), "garden-source-brick-migration").parameters!;
  assert.equal(migration.minimumFinalCoreBricks, 2);
  assert.equal(migration.requireResidentCountBelowCapacity, true);

  const brick = hook(getSceneWebGPUSmokeLane("brick-quad-dam-break"), "brick-quad-coverage").parameters!;
  assert.deepEqual(brick.expectedBrickGrid, [2, 1, 2]);
  assert.equal(brick.liquidThreshold, 0.5);
  assert.equal(brick.requireResidentCountBelowCapacity, true);

  const ocean = hook(getSceneWebGPUSmokeLane("ocean-seiche"), "ocean-wave-profile").parameters!;
  assert.deepEqual(ocean.expectedGrid, [320, 96, 80]);
  assert.equal(ocean.stationCount, 12);
  assert.equal(ocean.minimumFarHalfDisturbanceWidthRatio, 3.6);
});

test("Bet-4 deep-ocean lane is a bounded volumetric shipping fixture", () => {
  const suite = getSceneWebGPUSmokeSuite("power-hybrid-deep-ocean");
  const scene = suite.createScene();
  assert.deepEqual(scene.container, {
    width_m: 3.2, height_m: 3.2, depth_m: 2.4, fillFraction: 0.875,
    top: "closed", fluidWallMode: "no-slip",
  });
  assert.deepEqual(scene.voxelDomain, { finestCellSize_m: 0.05, brickSize_cells: 8 });
  const lane = getSceneWebGPUSmokeLane("power-hybrid-deep-ocean");
  assert.deepEqual(lane.stop, { simulatedTime_s: 0.004, exactSteps: 1, maxDt_s: 0.004 });
  assert.deepEqual(lane.methods[0]?.overrides, {
    maximumLeafSize: "32", interfaceRefinementBandCells: 1, globalFineLevelSetFactor: "1",
  });
  const packageJson = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as {
    scripts: Record<string, string>;
  };
  const command = packageJson.scripts["test:webgpu:power-hybrid-deep-ocean"];
  assert.match(command, /FLUID_SCENE=power-hybrid-deep-ocean/);
  assert.match(command, /FLUID_POWER_HYBRID_CENSUS=1/);
  assert.match(command, /FLUID_POWER_HYBRID_MIN_REDUCTION=2/);
  assert.match(command, /run-webgpu-smoke-isolated\.ts$/);
});

test("structured power lanes own exhaustive thresholds and raster policy", () => {
  const hydrostatic = getSceneWebGPUSmokeLane("hydrostatic-power-two-level");
  assert.deepEqual(hydrostatic.stop, { simulatedTime_s: 0.2, exactSteps: 50, maxDt_s: 0.004 });
  assert.equal(hydrostatic.methods[0].overrides.maximumLeafSize, "32");
  assert.equal(hydrostatic.methods[0].overrides.interfaceRefinementBandCells, 3);
  assert.equal(hydrostatic.collect.powerGenerationAudit && hydrostatic.collect.powerGenerationAudit.everySteps, 1);
  assert.ok(hydrostatic.acceptance.some(({ id, expected }) => id === "hydrostatic-power-volume-drift" && expected === 1e-4));

  const large = getSceneWebGPUSmokeLane("hydrostatic-power-large-offset");
  assert.equal(large.methods[0].overrides.interfaceRefinementBandCells, 4);
  assert.equal(large.collect.raster, "checkpoints");
  assert.equal(large.collect.structuredValidation, true);

  const moving = getSceneWebGPUSmokeLane("minimal-power-dam-break");
  assert.deepEqual(moving.stop, { simulatedTime_s: 2, exactSteps: 500, maxDt_s: 0.004 });
  assert.equal(moving.collect.raster, "checkpoints");
  assert.ok(moving.acceptance.some(({ id, expected }) => id === "minimal-power-variational-residual" && expected === 3.5e-6));
  assert.ok(moving.acceptance.some(({ id, expected }) => id === "fine-transport-unavailable" && expected === 0.08));
  assert.ok(moving.hooks.some(({ id }) => id === "water-raster-integrity"));

  const coarseSurface = getSceneWebGPUSmokeLane("minimal-power-dam-break-32", "surface-regression");
  assert.deepEqual(coarseSurface.stop, { simulatedTime_s: 0.5, exactSteps: 125, maxDt_s: 0.004 });
  assert.equal(coarseSurface.methods[0].overrides.globalFineLevelSetFactor, "1");
  assert.equal(coarseSurface.collect.raster, "checkpoints");
  assert.equal(coarseSurface.collect.checkpointEvery_s, 0.5);
  assert.ok(coarseSurface.acceptance.some(({ id, expected }) =>
    id === "coarse-surface-interface-area" && expected === 1.1));
  assert.ok(coarseSurface.acceptance.some(({ id, expected }) =>
    id === "coarse-surface-active-cubes" && expected === 5_200));
  assert.ok(coarseSurface.acceptance.some(({ id, expected }) =>
    id === "coarse-surface-resident-pages" && expected === 480));
  assert.ok(coarseSurface.acceptance.some(({ id, expected }) =>
    id === "coarse-surface-no-rejected-generations" && expected === 0));
});

test("ceiling smoke follows its dedicated band-1 UI profile only", () => {
  const defaultLane = getSceneWebGPUSmokeLane("ceiling-slab-drop");
  assert.equal(defaultLane.methods[0].overrides.interfaceRefinementBandCells, 1);
  assert.equal(defaultLane.collect.evidenceCollectors?.[0]?.requires?.includes("fine upper surface"), true);
  assert.equal(hook(defaultLane, "free-fall-contact").parameters?.maximumCenterProtrusion_cells, 0.05);
  const performance = getSceneWebGPUSmokeLane("ceiling-slab-drop", "performance");
  assert.equal(performance.methods[0].overrides.interfaceRefinementBandCells, 1);
  assert.equal(performance.collect.performanceProfile, true);
  assert.equal(performance.collect.fieldStats, "none");
  assert.deepEqual(performance.diagnostics.map(({ id }) => id), [
    "core-webgpu-health", "volume-and-topology", "octree-authority", "performance",
  ]);
  assert.deepEqual(performance.hooks, []);
  const baseline = getSceneWebGPUSmokeLane("ceiling-slab-drop", "coarse-baseline-post-impact");
  assert.deepEqual(baseline.stop, { simulatedTime_s: 0.4, exactSteps: 100, maxDt_s: 0.004 });
  assert.equal(baseline.methods[0].overrides.globalFineLevelSetFactor, "1");
  assert.equal(baseline.methods[0].overrides.interfaceRefinementBandCells, 1);
  assert.deepEqual(baseline.diagnostics.map(({ id }) => id), [
    "core-webgpu-health", "volume-and-topology", "octree-authority", "global-fine-publication",
  ]);
  for (const id of ["corner-brick-drop", "midair-brick-drop", "midair-corner-drop"] as const) {
    assert.equal(getSceneWebGPUSmokeLane(id).methods[0]
      .overrides.interfaceRefinementBandCells, 3);
  }
});

test("lane lookup rejects unknown lane names without fallback", () => {
  assert.throws(
    () => getSceneWebGPUSmokeLane("ocean-seiche", "missing"),
    /Unknown WebGPU smoke lane ocean-seiche\/missing/,
  );
});

test("the generic runner contains no concrete catalog scene dispatch", () => {
  const entrypoint = readFileSync(new URL("../tools/run-webgpu-smoke.ts", import.meta.url), "utf8");
  const runner = readFileSync(new URL("../tools/webgpu-smoke-executor.ts", import.meta.url), "utf8");
  assert.ok(entrypoint.trim().split("\n").length <= 3,
    "run-webgpu-smoke.ts must remain a thin CLI entrypoint");
  assert.match(entrypoint, /import\("\.\/webgpu-smoke-executor"\)/);
  for (const sceneId of sceneWebGPUSmokeIds) {
    const escaped = sceneId.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    assert.doesNotMatch(runner, new RegExp(`["']${escaped}["']`),
      `${sceneId} behavior belongs in its scene smoke suite, not the runner`);
  }
  const declarationIds = new Set(Object.values(sceneWebGPUSmokeSuites).flatMap((suite) =>
    Object.values(suite.lanes).flatMap((lane) => [
      ...lane.diagnostics.map(({ id }) => id),
      ...lane.hooks.map(({ id }) => id),
      ...lane.collect.evidenceCollectors.map(({ id }) => id),
    ])));
  for (const id of declarationIds) {
    const escaped = id.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    assert.doesNotMatch(runner, new RegExp(`["']${escaped}["']`),
      `${id} dispatch belongs in the diagnostics registry, not the executor`);
  }
});

test("named smoke commands resolve their declared scene lanes", () => {
  const packageJson = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as {
    scripts: Record<string, string>;
  };
  for (const [script, command] of Object.entries(packageJson.scripts)) {
    if (!command.includes("run-webgpu-smoke") || !command.includes("FLUID_LANE=")) continue;
    const sceneId = /FLUID_SCENE=([^ $]+)/.exec(command)?.[1];
    const laneId = /FLUID_LANE=([^ $]+)/.exec(command)?.[1];
    assert.ok(sceneId && isSceneWebGPUSmokeId(sceneId), `${script} must select a catalog scene`);
    assert.ok(laneId && getSceneWebGPUSmokeSuite(sceneId!).lanes[laneId],
      `${script} must select an authored lane for ${sceneId}`);
  }
});
