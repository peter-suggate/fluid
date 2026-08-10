import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { validateScene } from "../lib/model";
import { initialFluidBrickUnionBounds } from "../lib/initial-fluid";
import {
  octreeColdAuthoredSurfaceBoxes,
  octreeColdAuthoredSurfaceWouldRefine,
} from "../lib/octree-cold-authored-surface";
import { getSceneWebGPUSmokeLane } from "../lib/scene-webgpu-smoke-catalog";
import { sceneCustomHookImplementations } from "../lib/scene-custom-diagnostic-implementations";
import { createSymmetricExpansionScene, getScenePreset } from "../lib/scenes";
import { initialOctreeLevelSet, initialOctreeNodalLevelSet } from "../lib/webgpu-octree";

test("symmetric expansion is the minimum dyadic four-brick horizontal oracle", () => {
  const scene = createSymmetricExpansionScene();
  assert.deepEqual(validateScene(scene), []);
  assert.equal(scene.sceneId, "symmetric-expansion");
  assert.deepEqual(scene.container, {
    width_m: 1.6, height_m: 0.8, depth_m: 1.6, fillFraction: 1 / 8,
    top: "closed", fluidWallMode: "free-slip",
  });
  assert.deepEqual(scene.voxelDomain, { finestCellSize_m: 0.05, brickSize_cells: 8 });
  assert.deepEqual(scene.fluid.initialBrickSeeds_m, [
    { x: -0.2, y: 0.2, z: -0.2 }, { x: 0.2, y: 0.2, z: -0.2 },
    { x: -0.2, y: 0.2, z: 0.2 }, { x: 0.2, y: 0.2, z: 0.2 },
  ]);
  assert.equal(scene.fluid.surfaceTension_N_m, 0);
  assert.equal(scene.fluid.dynamicViscosity_Pa_s, 0,
    "the Losasso fidelity oracle must declare its inviscid motion model");
  assert.equal(scene.fluid.inflow, undefined);
  assert.deepEqual(scene.rigidBodies, []);
  assert.deepEqual([
    Math.round(scene.container.width_m / scene.voxelDomain.finestCellSize_m),
    Math.round(scene.container.height_m / scene.voxelDomain.finestCellSize_m),
    Math.round(scene.container.depth_m / scene.voxelDomain.finestCellSize_m),
  ], [32, 16, 32]);
  const brickBounds = initialFluidBrickUnionBounds(scene, [32, 16, 32]);
  assert.ok(brickBounds);
  assert.deepEqual(Array.from(new Float32Array([
    brickBounds.minimum.x, brickBounds.minimum.y, brickBounds.minimum.z,
    brickBounds.maximum.x, brickBounds.maximum.y, brickBounds.maximum.z,
  ])), [-0.4000000059604645, 0, -0.4000000059604645,
    0.4000000059604645, 0.4000000059604645, 0.4000000059604645]);

  const preset = getScenePreset("symmetric-expansion");
  assert.equal(preset.create().sceneId, scene.sceneId);
  assert.equal(preset.presentationMode, "fluid-only",
    "the validation body must not be hidden by an unrelated dry-scene floor");
  assert.equal(preset.methodProfile?.methodId, "octree");
  assert.equal(preset.methodProfile?.overrides.globalFineLevelSetFactor, "1");
  assert.equal(preset.methodProfile?.overrides.interfaceRefinementBandCells, 4);
});

test("symmetric expansion Dawn lane samples every accepted step and gates every state field", () => {
  const lane = getSceneWebGPUSmokeLane("symmetric-expansion");
  assert.equal(lane.stop.exactSteps, 250);
  assert.equal(lane.stop.maxDt_s, 0.004);
  assert.equal(lane.collect.checkpointEvery_s, 0.004);
  assert.equal(lane.methods[0]?.overrides.globalFineLevelSetFactor, "1");
  assert.equal(lane.methods[0]?.overrides.interfaceRefinementBandCells, 4);
  assert.equal(lane.methods[0]?.overrides.losassoVelocityExtension, "causal-front");
  assert.equal(lane.collect.stabilityEnvelope, true);
  assert.equal(lane.collect.energyEverySteps, 10);
  const collector = lane.collect.evidenceCollectors.find(({ id }) => id === "fluid-symmetry");
  assert.ok(collector);
  assert.deepEqual(collector.requires, ["compact velocity", "compact pressure"]);
  const hook = lane.hooks.find(({ id }) => id === "fluid-symmetry");
  assert.ok(hook);
  assert.deepEqual(hook.parameters, {
    maximumVolumeAbsoluteError: 1e-3,
    maximumVelocityAbsoluteError_m_s: 1e-4,
    maximumPressureAbsoluteError: 0.25,
    maximumRhsAbsoluteError: 0.015625,
    maximumDiagonalAbsoluteError: 1e-3,
    requireExactTopology: true,
    requireAllWallsReached: true,
    minimumCheckpointCount: 250,
    maximumWallContactStepSpread: 0,
    circularityEvaluationStart_s: 0.168,
    circularityEvaluationEnd_s: 0.2,
    frontAdvanceEvaluationEnd_s: 0.2,
    minimumMeanFrontAdvance_cells: 1,
    maximumAxisDiagonalFrontDifference_cells: 1,
    maximumRadialRmsDeviation_cells: 0.5,
    maximumRadialDeviation_cells: 1,
    minimumCircularityAngularSamples: 64,
  });

  const fine = getSceneWebGPUSmokeLane("symmetric-expansion", "fine-factor-4");
  assert.equal(fine.stop.exactSteps, 1);
  assert.equal(fine.methods[0]?.overrides.globalFineLevelSetFactor, "4");
  assert.equal(fine.collect.raster, "initial-final");
  assert.equal(fine.collect.globalFineGeneration, true);
  const raster = getSceneWebGPUSmokeLane("symmetric-expansion", "raster-construction");
  assert.equal(raster.stop.exactSteps, 1);
  assert.equal(raster.methods[0]?.overrides.globalFineLevelSetFactor, "4");
  assert.equal(raster.collect.raster, "initial-final");
  assert.equal(raster.collect.globalFineGeneration, true);
  assert.equal(raster.diagnostics.some(({ id }) => id === "exhaustive-power-generation"), false,
  "the raster regression must not inherit unrelated power-generation or stability gates");
  assert.equal(raster.collect.evidenceCollectors.length, 0,
    "the raster regression must isolate construction from the known post-step field asymmetry");
  const oneStep = getSceneWebGPUSmokeLane("symmetric-expansion", "one-step");
  assert.equal(oneStep.stop.exactSteps, 1);
  assert.equal(oneStep.methods[0]?.overrides.globalFineLevelSetFactor, "1");
  assert.deepEqual(oneStep.hooks[0]?.parameters, {
    maximumVolumeAbsoluteError: 1e-3,
    maximumVelocityAbsoluteError_m_s: 1e-4,
    maximumPressureAbsoluteError: 0.25,
    maximumRhsAbsoluteError: 0.015625,
    maximumDiagonalAbsoluteError: 1e-3,
    requirePressureStageAudit: true,
    requireExactTopology: true,
    requireAllWallsReached: false,
    minimumCheckpointCount: 1,
    maximumWallContactStepSpread: 0,
  });
  const twoStep = getSceneWebGPUSmokeLane("symmetric-expansion", "two-step");
  assert.equal(twoStep.stop.exactSteps, 2);
  assert.equal(oneStep.hooks[0]?.parameters?.requirePressureStageAudit, true,
    "the row-parallel MGPCG lane must expose each staged-preconditioner symmetry boundary");
  assert.equal(twoStep.hooks[0]?.parameters?.requirePressureStageAudit, undefined,
    "multi-step gates keep the diagnostic readback surface bounded");
  assert.equal(twoStep.hooks[0]?.parameters?.minimumCheckpointCount, 2);
  const twentyStep = getSceneWebGPUSmokeLane("symmetric-expansion", "twenty-step");
  assert.equal(twentyStep.stop.exactSteps, 20,
    "the development gate must catch symmetry regressions before the 250-step release lane");
  assert.equal(twentyStep.stop.maxDt_s, 0.004);
  assert.equal(twentyStep.collect.checkpointEvery_s, 0.004);
  assert.equal(twentyStep.methods[0]?.overrides.globalFineLevelSetFactor, "1");
  assert.equal(twentyStep.hooks[0]?.parameters?.requireExactTopology, true);
  assert.equal(twentyStep.hooks[0]?.parameters?.minimumCheckpointCount, 20);
  assert.equal(twentyStep.hooks[0]?.parameters?.requireAllWallsReached, false);
  assert.equal(fine.hooks[0]?.parameters?.requireAllWallsReached, false);
});

test("brick-authored octree bootstrap preserves the exact analytic box SDF and D4 symmetry", () => {
  const scene = createSymmetricExpansionScene();
  const [nx, ny, nz] = [32, 16, 32] as const;
  const phi = initialOctreeLevelSet(scene, { nx, ny, nz }, { x: 0.05, y: 0.05, z: 0.05 });
  const at = (x: number, y: number, z: number) => phi[x + nx * (y + ny * z)];
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    assert.ok(Object.is(at(x, y, z), at(nx - 1 - x, y, z)), `reflect-x at ${x},${y},${z}`);
    assert.ok(Object.is(at(x, y, z), at(x, y, nz - 1 - z)), `reflect-z at ${x},${y},${z}`);
    assert.ok(Object.is(at(x, y, z), at(z, y, x)), `swap-xz at ${x},${y},${z}`);
  }
  assert.ok(at(7, 0, 7) > at(7, 0, 8), "an exterior box corner retains Euclidean distance rather than a voxel plateau");
  assert.ok(at(8, 7, 8) < 0 && at(8, 8, 8) > 0, "the top face crosses exactly halfway between cell centres");
});

test("symmetric expansion has an exact direct nodal SDF before adaptive rasterisation", () => {
  const scene = createSymmetricExpansionScene();
  const [nx, ny, nz] = [32, 16, 32] as const;
  const phi = initialOctreeNodalLevelSet(scene, { nx, ny, nz });
  assert.ok(phi);
  const at = (x: number, y: number, z: number) =>
    phi[x + (nx + 1) * (y + (ny + 1) * z)];
  for (let z = 0; z <= nz; z += 1) {
    for (let y = 0; y <= ny; y += 1) {
      for (let x = 0; x <= nx; x += 1) {
        assert.ok(Object.is(at(x, y, z), at(nx - x, y, z)),
          `nodal reflect-x at ${x},${y},${z}`);
        assert.ok(Object.is(at(x, y, z), at(x, y, nz - z)),
          `nodal reflect-z at ${x},${y},${z}`);
        assert.ok(Object.is(at(x, y, z), at(z, y, x)),
          `nodal swap-xz at ${x},${y},${z}`);
      }
    }
  }
  // The authored 16 x 8 x 16 body is [8,24] x [0,8] x [8,24].  Its exact
  // face, edge and corner nodes all lie on phi=0.  A cell-centred bootstrap
  // reconstructed onto these nodes cannot preserve the edge/corner equalities.
  assert.equal(at(16, 8, 16), 0, "top-face interior node");
  assert.equal(at(8, 8, 16), 0, "top edge node");
  assert.equal(at(8, 8, 8), 0, "top corner node");
  assert.ok(at(16, 4, 16) < 0,
    "touching authored bricks must form one liquid interior, not internal zero sheets");
  assert.equal(at(7, 8, 8), Math.fround(0.05), "one-cell exterior edge distance");
  assert.equal(at(7, 8, 7), Math.fround(Math.hypot(0.05, 0.05)),
    "one-cell exterior corner distance");
});

test("cold topology splits exact authored top face, edge, and corner contacts to size one", () => {
  const scene = createSymmetricExpansionScene();
  const boxes = octreeColdAuthoredSurfaceBoxes(scene, [32, 16, 32], 8);
  assert.deepEqual(boxes, [{ minimum: [8, 0, 8], maximum: [24, 8, 24] }]);
  const refine = (origin: readonly [number, number, number]) =>
    octreeColdAuthoredSurfaceWouldRefine(boxes, origin, 2, [0.05, 0.05, 0.05], 1);
  assert.equal(refine([14, 8, 14]), true, "top face contact");
  assert.equal(refine([6, 8, 14]), true, "top edge contact from the dry side");
  assert.equal(refine([6, 8, 6]), true, "top corner contact from the dry side");
  assert.equal(octreeColdAuthoredSurfaceWouldRefine(
    boxes, [14, 2, 14], 2, [0.05, 0.05, 0.05], 1,
  ), false, "liquid deeper than the requested band remains eligible to coarsen");
  assert.equal(octreeColdAuthoredSurfaceWouldRefine(
    boxes, [14, 8, 14], 1, [0.05, 0.05, 0.05], 1,
  ), false, "the authored classifier stops at the configured finest size");

  const projection = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
  const evidence = projection.slice(
    projection.indexOf("fn pressureRefinementEvidence"),
    projection.indexOf("fn boundaryLiquidMinimumPhi"),
  );
  assert.match(evidence,
    /coldAuthoredSurfaceInterval\(origin, size\)[\s\S]*size > finestSurfaceCellSize\(\)[\s\S]*crossesOrTouchesSurface/,
    "cold exact nodal evidence must be consulted before a missing sparse summary rejects the leaf");
  assert.ok(evidence.indexOf("coldAuthoredSurfaceInterval(origin, size)")
    < evidence.indexOf("if (!summary.found) { return false; }"));
  assert.match(projection,
    /if \(!bootstrapPhiEnabled\(\) \|\| count == 0u\)/,
    "authored boxes must retire with cold bootstrap rather than become recurring fallback authority");
});

test("procedural dam break publishes one exact nodal box and cold refinement authority", () => {
  const scene = getScenePreset("water-box-dam-break").create();
  const dimensions = [24, 18, 16] as const;
  const phi = initialOctreeNodalLevelSet(scene, { nx: 24, ny: 18, nz: 16 });
  assert.ok(phi, "the procedural dam must not pass through cell-centred distance reconstruction");
  const boxes = octreeColdAuthoredSurfaceBoxes(scene, dimensions, 8);
  assert.equal(boxes.length, 1);
  const box = boxes[0]!;
  assert.equal(box.minimum[0], 0); assert.equal(box.minimum[1], 0); assert.equal(box.minimum[2], 0);
  assert.ok(box.maximum[0] > 11 && box.maximum[0] < 12);
  assert.ok(box.maximum[1] > 16 && box.maximum[1] < 17);
  assert.ok(box.maximum[2] > 7 && box.maximum[2] < 8);
  const at = (x: number, y: number, z: number) =>
    phi[x + 25 * (y + 19 * z)];
  const dx = scene.container.width_m / dimensions[0];
  assert.ok(Math.abs((at(12, 8, 4) - at(11, 8, 4)) - dx) < 2e-8,
    "the vertical dam plane must remain affine on the direct node lattice");
  assert.equal(octreeColdAuthoredSurfaceWouldRefine(boxes, [10, 8, 4], 2,
    [dx, scene.container.height_m / 18, scene.container.depth_m / 16], 1), true,
  "a size-two leaf touching the procedural dam front must split to unit leaves");

  const additive = getScenePreset("water-box-dam-break").create();
  additive.fluid.initialBrickSeedsAdditive = true;
  assert.equal(initialOctreeNodalLevelSet(additive, { nx: 24, ny: 18, nz: 16 }), undefined);
  assert.deepEqual(octreeColdAuthoredSurfaceBoxes(additive, dimensions, 8), [],
    "additive initial conditions must retain the composed bootstrap authority");

  const terrain = getScenePreset("water-box-dam-break").create();
  terrain.terrain = { baseHeight_m: 0.05, features: [] };
  assert.equal(initialOctreeNodalLevelSet(terrain, { nx: 24, ny: 18, nz: 16 }), undefined);
  assert.deepEqual(octreeColdAuthoredSurfaceBoxes(terrain, dimensions, 8), [],
    "terrain-clipped liquid must retain the terrain-aware bootstrap authority");
});

test("factor-one construction retains the nodal layout through cold GPU retries", () => {
  const projection = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
  assert.match(projection,
    /adaptiveInitialNodalPhi[\s\S]*initialOctreeNodalLevelSet\(this\.scene, this\.dims\)[\s\S]*adaptiveInitialNodalPhi \? "nodal-lattice" : "cell-centred"/,
    "the projection must choose the direct node lattice only when its analytic source exists");
  const backend = readFileSync(new URL(
    "../lib/webgpu-octree-losasso-backend.ts", import.meta.url), "utf8");
  assert.match(backend,
    /adaptiveBootstrapPhiLayout = initialPhiLayout[\s\S]*this\.adaptiveBootstrapPhiLayout === "nodal-lattice"[\s\S]*"nodal-lattice-cpu"/,
    "the retained cold candidate retry must not reinterpret nodal samples as cell centres");
  assert.match(backend,
    /initialPhiLayout === "nodal-lattice" \? value \+ 1 : value/,
    "constructor validation must size the complete boundary-inclusive node lattice");
});

test("symmetric expansion rejects a stationary front even when every field remains D4 symmetric", () => {
  const metric = { maximumAbsoluteError: 0, nonFiniteCount: 0 };
  const observation = (meanRadius_cells: number) => ({
    volume: metric, velocity: metric, pressure: metric, rhs: metric, diagonal: metric,
    topology: { ...metric, exactMismatchCount: 0 },
    frontCircularity: { meanRadius_cells },
    walls: {
      negativeX: { touched: false }, positiveX: { touched: false },
      negativeZ: { touched: false }, positiveZ: { touched: false },
    },
  });
  const checkpoints = (finalRadius: number) => [
    { time_s: 0.004, evidence: { "fluid-symmetry": observation(8) } },
    { time_s: 0.2, evidence: { "fluid-symmetry": observation(finalRadius) } },
  ];
  const evaluate = (finalRadius: number) => sceneCustomHookImplementations["fluid-symmetry"].evaluate({
    parameters: {
      maximumVolumeAbsoluteError: 0, maximumVelocityAbsoluteError_m_s: 0,
      maximumPressureAbsoluteError: 0, maximumRhsAbsoluteError: 0,
      maximumDiagonalAbsoluteError: 0, minimumCheckpointCount: 2,
      maximumWallContactStepSpread: 0, requireExactTopology: true,
      requireAllWallsReached: false, frontAdvanceEvaluationEnd_s: 0.2,
      minimumMeanFrontAdvance_cells: 1,
    },
    selectedMethods: ["octree"],
    getMethod: () => ({ available: [], diagnostics: { field: { checkpoints: checkpoints(finalRadius) } } }),
  } as never);
  assert.equal(evaluate(8).find(({ id }) => id === "octree.front-advance")?.passed, false);
  assert.equal(evaluate(9.1).find(({ id }) => id === "octree.front-advance")?.passed, true);
});

test("symmetric expansion has an isolated Dawn reproduction command", () => {
  const packageJson = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as {
    scripts: Record<string, string>;
  };
  const command = packageJson.scripts["test:webgpu:symmetric-expansion"];
  assert.match(command, /FLUID_SCENE=symmetric-expansion/);
  assert.match(command, /FLUID_EXPECT_GRID=32,16,32/);
  assert.match(command, /FLUID_CHECKPOINT_EVERY_S=0\.004/);
  assert.match(command, /FLUID_EXPECT_EXACT_STEPS=250/);
  assert.match(command, /FLUID_STABILITY_ENVELOPE=1/);
  assert.match(command, /FLUID_WEBGPU_SMOKE_TIMEOUT_MS=600000/,
    "per-step D4 field readbacks need a larger allowance than the shipping performance lane");
  assert.match(command, /FLUID_OCTREE_INTERFACE_BAND=4/);
  assert.match(command, /FLUID_OCTREE_GLOBAL_FINE_FACTOR=4/);
  assert.match(command, /FLUID_GLOBAL_FINE_GENERATION_TRANSITION=1/);
  assert.match(command, /run-webgpu-smoke-isolated\.ts$/);

  const coarseOnlyCommand = packageJson.scripts["test:webgpu:symmetric-expansion:coarse-only"];
  assert.ok(coarseOnlyCommand);
  assert.match(coarseOnlyCommand, /FLUID_LANE=coarse-only/);
  assert.match(coarseOnlyCommand, /FLUID_EXPECT_EXACT_STEPS=250/);
  assert.match(coarseOnlyCommand, /FLUID_OCTREE_INTERFACE_BAND=4/);
  assert.match(coarseOnlyCommand, /FLUID_OCTREE_GLOBAL_FINE_FACTOR=1/);
  assert.doesNotMatch(coarseOnlyCommand, /FLUID_GLOBAL_FINE_GENERATION_TRANSITION=1/,
    "coarse-only Dawn validation must not request a separate fine publication");

  const coarseOnlyLane = getSceneWebGPUSmokeLane("symmetric-expansion", "coarse-only");
  assert.equal(coarseOnlyLane.stop.exactSteps, 250);
  assert.equal(coarseOnlyLane.methods[0]?.overrides.globalFineLevelSetFactor, "1");
  assert.equal(coarseOnlyLane.diagnostics.some(({ id }) => id === "settling"), false,
    "the coarse symmetry lane must not require mechanical-energy evidence it does not collect");
  assert.equal(coarseOnlyLane.hooks[0]?.parameters?.minimumCheckpointCount, 250);
  assert.equal(coarseOnlyLane.hooks[0]?.parameters?.requireAllWallsReached, true);
  assert.equal(coarseOnlyLane.hooks[0]?.parameters?.maximumWallContactStepSpread, 0);
  assert.equal(coarseOnlyLane.collect.raster, "initial-final");
  assert.equal(coarseOnlyLane.hooks[0]?.parameters?.frontAdvanceEvaluationEnd_s, 0.2);
  assert.equal(coarseOnlyLane.hooks[0]?.parameters?.minimumMeanFrontAdvance_cells, 1);
  assert.equal(coarseOnlyLane.hooks[0]?.parameters?.circularityEvaluationStart_s, 0.168,
    "factor one must pass the same physical front window as the factor-4 baseline");

  const band2Command = packageJson.scripts["test:webgpu:symmetric-expansion:band2"];
  assert.ok(band2Command);
  assert.match(band2Command, /FLUID_LANE=default/);
  assert.match(band2Command, /FLUID_EXPECT_EXACT_STEPS=250/);
  assert.match(band2Command, /FLUID_STABILITY_ENVELOPE=1/);
  assert.match(band2Command, /FLUID_MAXIMUM_LEAF_SIZE=32/,
    "the accuracy A/B must differ from the established scene gate only by band width");
  assert.match(band2Command, /FLUID_OCTREE_INTERFACE_BAND=2/);
  assert.match(band2Command, /FLUID_OCTREE_GLOBAL_FINE_FACTOR=4/);
  assert.match(band2Command, /FLUID_WEBGPU_SMOKE_TIMEOUT_MS=600000/,
    "per-step D4 field readbacks need a larger allowance than the shipping performance lane");

  for (const band of [2, 4]) {
    const performanceCommand = packageJson.scripts[
      `test:webgpu:symmetric-expansion:band${band}-performance`
    ];
    assert.ok(performanceCommand);
    assert.match(performanceCommand, /FLUID_LANE=performance/);
    assert.match(performanceCommand, /FLUID_EXPECT_EXACT_STEPS=62/);
    assert.match(performanceCommand, /FLUID_PERFORMANCE_PROFILE=1/);
    assert.match(performanceCommand, new RegExp(`FLUID_OCTREE_INTERFACE_BAND=${band}`));
    assert.match(performanceCommand, /FLUID_MAXIMUM_LEAF_SIZE=32/,
      "the performance A/B must use the established scene hierarchy");
  }

  const fineCommand = packageJson.scripts["test:webgpu:symmetric-expansion:fine"];
  assert.ok(fineCommand);
  assert.match(fineCommand, /FLUID_LANE=fine-factor-4/);
  assert.match(fineCommand, /FLUID_OCTREE_GLOBAL_FINE_FACTOR=4/);
  assert.match(fineCommand, /FLUID_OCTREE_INTERFACE_BAND=4/);
  assert.match(fineCommand, /FLUID_RASTER_CHECKPOINTS=1/);
  assert.match(fineCommand, /FLUID_RASTER_MESH_SYMMETRY=1/);
  assert.match(fineCommand, /FLUID_GLOBAL_FINE_GENERATION_TRANSITION=1/);
  const rasterCommand = packageJson.scripts["test:webgpu:symmetric-expansion:raster"];
  assert.match(rasterCommand, /FLUID_LANE=raster-construction/);
  assert.match(rasterCommand, /FLUID_RASTER_MESH_SYMMETRY=1/);
  assert.match(rasterCommand, /FLUID_OCTREE_GLOBAL_FINE_FACTOR=4/);
  assert.match(rasterCommand, /FLUID_GLOBAL_FINE_GENERATION_TRANSITION=1/);
  const oneStepCommand = packageJson.scripts["test:webgpu:symmetric-expansion:one-step"];
  assert.match(oneStepCommand, /FLUID_LANE=one-step/);
  assert.match(oneStepCommand, /FLUID_SYMMETRY_STAGE_AUDIT=1/);
  const twoStepCommand = packageJson.scripts["test:webgpu:symmetric-expansion:two-step"];
  assert.match(twoStepCommand, /FLUID_LANE=two-step/);
  assert.match(twoStepCommand, /FLUID_SYMMETRY_STAGE_AUDIT=1/);
  const executor = readFileSync(new URL("../tools/webgpu-smoke-executor.ts", import.meta.url), "utf8");
  assert.match(executor,
    /rasterInitialFinal: collect\.raster === "initial-final"/,
    "an authored initial/final raster collector must not depend on the separate fine-band transition");
  assert.match(executor,
    /FLUID_RASTER_MESH_SYMMETRY[\s\S]*hasSeparateFineLevelSetBand[\s\S]*mesh\.exactMismatchCount !== 0 : mesh\.exactPositionMismatchCount !== 0[\s\S]*hasSeparateFineLevelSetBand && !phi[\s\S]*phi\.supportMismatchCount !== 0/,
    "frame zero must reject asymmetric surface coverage at factor one and additionally require normal and phi symmetry for a separate fine band");
});
