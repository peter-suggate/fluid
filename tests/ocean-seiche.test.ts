import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { createOceanSeicheScene, getScenePreset } from "../lib/scenes";
import { combineInitialBrickWet, initialFluidBrickContainsCell } from "../lib/initial-fluid";
import { createTallCellLayout, initialLiquidPhi, type GPUQuality } from "../lib/tall-cell-grid";
import { createSmokeScenario, isSmokeScenarioId, minimumOceanFarHalfDisturbanceCells } from "../tools/webgpu-smoke-scenarios";
import { validateScene } from "../lib/model";

const OCEAN_GRID = [384, 96, 64] as const;

test("ocean tank resolves to exactly 384x96x64 finest cells at every quality", () => {
  const scene = createOceanSeicheScene();
  assert.deepEqual(validateScene(scene), []);
  for (const quality of ["balanced", "high", "ultra"] as GPUQuality[]) {
    const layout = createTallCellLayout(scene, quality);
    assert.deepEqual([layout.nx, layout.fineNy, layout.nz], [...OCEAN_GRID], `${quality} finest grid must be 48x12x8 bricks of 8-cubed cells`);
  }
});

test("ocean water is a settled pool plus one raised full-depth slab along the -x wall", () => {
  const scene = createOceanSeicheScene();
  assert.equal(scene.fluid.initialCondition, "tank-fill");
  assert.equal(scene.fluid.initialBrickSeedsAdditive, true);
  assert.equal(scene.fluid.initialBrickSeeds_m?.length, 16, "the slab is 2x1x8 seeded bricks");
  const [nx, ny, nz] = OCEAN_GRID;
  const poolLayers = Math.round(scene.container.fillFraction * ny);
  assert.equal(poolLayers, 72, "the pool surface must sit exactly on a brick boundary");
  let poolCells = 0, slabCells = 0;
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    const brickWet = initialFluidBrickContainsCell(scene, x, y, z, OCEAN_GRID);
    const baseWet = (y + 0.5) / ny <= scene.container.fillFraction;
    const wet = combineInitialBrickWet(scene, brickWet, baseWet);
    if (!wet) continue;
    if (y < poolLayers) { poolCells += 1; continue; }
    slabCells += 1;
    assert.ok(x < 16, `raised cell ${x},${y},${z} must lie in the two -x wall brick tiers`);
    assert.ok(y < poolLayers + 8, `raised cell ${x},${y},${z} must lie in the single brick tier on the surface`);
  }
  assert.equal(poolCells, nx * poolLayers * nz, "additive seeds must not carve the base pool");
  assert.equal(slabCells, 16 * 8 * nz, "the slab must be exactly 16x8 cells across the full depth extent");
  // The analytic phi agrees with the union: wet inside the slab, dry beside it.
  const h = scene.container.width_m / nx;
  const insideSlab = { x: -scene.container.width_m / 2 + 4 * h, y: (poolLayers + 4) * h, z: 0.1 };
  const besideSlab = { x: 0, y: (poolLayers + 4) * h, z: 0 };
  const insidePool = { x: 0, y: (poolLayers - 4) * h, z: 0 };
  assert.ok(initialLiquidPhi(scene, insideSlab, OCEAN_GRID) < 0, "phi must be liquid inside the raised slab");
  assert.ok(initialLiquidPhi(scene, besideSlab, OCEAN_GRID) > 0, "phi must be air beside the slab above the pool");
  assert.ok(initialLiquidPhi(scene, insidePool, OCEAN_GRID) < 0, "phi must keep the base pool liquid away from the slab");
});

test("non-additive seeds still replace the base initial condition", () => {
  const scene = createOceanSeicheScene();
  scene.fluid.initialBrickSeedsAdditive = false;
  const [, ny] = OCEAN_GRID;
  const poolY = Math.round(0.5 * ny * 0.75);
  assert.equal(combineInitialBrickWet(scene, initialFluidBrickContainsCell(scene, 100, poolY, 8, OCEAN_GRID), true), false,
    "replace-mode seeds must suppress the base pool exactly as before");
});

test("ocean scene is registered in the UI presets and the smoke harness with leaf-32 requested", () => {
  const preset = getScenePreset("ocean-seiche");
  assert.equal(preset.id, "ocean-seiche", "the preset must exist rather than fall back to the default scene");
  assert.match(preset.description, /32/);
  assert.ok(isSmokeScenarioId("ocean-seiche"));
  const scenario = createSmokeScenario("ocean-seiche");
  assert.equal(scenario.scene.numerics.surfaceColumnsOverride, 24576);
  assert.equal(scenario.scene.fluid.surfaceTension_N_m, 0);
  assert.equal(scenario.scene.container.width_m, 9.6);
  assert.equal(scenario.scene.container.height_m, 2.4);
  assert.equal(scenario.scene.container.depth_m, 1.6);
  assert.equal(minimumOceanFarHalfDisturbanceCells(scenario.scene.container.width_m), 0.375,
    "the fixed slab's far-half amplitude bar scales inversely with widened tank length");
  assert.ok(scenario.target_s >= 5, "the wave needs several crossings of observation time");
  // Scenes cannot carry method parameters; the harness must request the
  // raised 32-cubed cap for the octree run (env override still wins).
  const harness = readFileSync(new URL("../tools/run-webgpu-smoke.ts", import.meta.url), "utf8");
  assert.match(harness, /scenarioId === "ocean-seiche"\) values\.maximumLeafSize = 32/);
  assert.match(harness, /const sparseSource = sparseStatsRequested\s*\? \(solver as GPUSolverInstance\)\.sparseVoxelRenderSource\s*:\s*undefined/,
    "production ocean benchmarks must not allocate the lazy raw-inspection publication");
  assert.match(harness, /phase: "ocean-wave-profile"/);
});
