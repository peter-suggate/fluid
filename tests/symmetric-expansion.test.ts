import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { validateScene } from "../lib/model";
import { initialFluidBrickUnionBounds } from "../lib/initial-fluid";
import { getSceneWebGPUSmokeLane } from "../lib/scene-webgpu-smoke-catalog";
import { createSymmetricExpansionScene, getScenePreset } from "../lib/scenes";
import { initialOctreeLevelSet } from "../lib/webgpu-octree";

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
  assert.equal(preset.methodProfile?.methodId, "octree");
  assert.equal(preset.methodProfile?.overrides.globalFineLevelSetFactor, "4");
  assert.equal(preset.methodProfile?.overrides.interfaceRefinementBandCells, 4);
});

test("symmetric expansion Dawn lane samples every accepted step and gates every state field", () => {
  const lane = getSceneWebGPUSmokeLane("symmetric-expansion");
  assert.equal(lane.stop.exactSteps, 250);
  assert.equal(lane.stop.maxDt_s, 0.004);
  assert.equal(lane.collect.checkpointEvery_s, 0.004);
  assert.equal(lane.methods[0]?.overrides.globalFineLevelSetFactor, "4");
  assert.equal(lane.methods[0]?.overrides.interfaceRefinementBandCells, 4);
  const collector = lane.collect.evidenceCollectors.find(({ id }) => id === "fluid-symmetry");
  assert.ok(collector);
  assert.deepEqual(collector.requires, ["compact velocity", "compact pressure"]);
  const hook = lane.hooks.find(({ id }) => id === "fluid-symmetry");
  assert.ok(hook);
  assert.deepEqual(hook.parameters, {
    maximumVolumeAbsoluteError: 0,
    maximumVelocityAbsoluteError_m_s: 0,
    maximumPressureAbsoluteError: 0,
    maximumRhsAbsoluteError: 0,
    maximumDiagonalAbsoluteError: 0,
    requireExactTopology: true,
    requireAllWallsReached: true,
    minimumCheckpointCount: 250,
    maximumWallContactStepSpread: 0,
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
  assert.equal(oneStep.methods[0]?.overrides.globalFineLevelSetFactor, "4");
  assert.deepEqual(oneStep.hooks[0]?.parameters, {
    maximumVolumeAbsoluteError: 0,
    maximumVelocityAbsoluteError_m_s: 0,
    maximumPressureAbsoluteError: 0,
    maximumRhsAbsoluteError: 0,
    maximumDiagonalAbsoluteError: 0,
    requireExactTopology: true,
    requireAllWallsReached: false,
    minimumCheckpointCount: 1,
    maximumWallContactStepSpread: 0,
  });
  const twoStep = getSceneWebGPUSmokeLane("symmetric-expansion", "two-step");
  assert.equal(twoStep.stop.exactSteps, 2);
  assert.equal(twoStep.hooks[0]?.parameters?.requirePressureStageAudit, undefined,
    "persistent MGPCG no longer exposes the retired staged-preconditioner audit buffers");
  assert.equal(twoStep.hooks[0]?.parameters?.minimumCheckpointCount, 2);
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

test("symmetric expansion has an isolated Dawn reproduction command", () => {
  const packageJson = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as {
    scripts: Record<string, string>;
  };
  const command = packageJson.scripts["test:webgpu:symmetric-expansion"];
  assert.match(command, /FLUID_SCENE=symmetric-expansion/);
  assert.match(command, /FLUID_EXPECT_GRID=32,16,32/);
  assert.match(command, /FLUID_CHECKPOINT_EVERY_S=0\.004/);
  assert.match(command, /FLUID_EXPECT_EXACT_STEPS=250/);
  assert.match(command, /FLUID_OCTREE_INTERFACE_BAND=4/);
  assert.match(command, /FLUID_OCTREE_GLOBAL_FINE_FACTOR=4/);
  assert.match(command, /FLUID_GLOBAL_FINE_GENERATION_TRANSITION=1/);
  assert.match(command, /run-webgpu-smoke-isolated\.ts$/);

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
