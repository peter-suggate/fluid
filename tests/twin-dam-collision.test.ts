import assert from "node:assert/strict";
import test from "node:test";
import { createTwinDamCollisionScene, getSceneDefinition, getScenePreset } from "../lib/scenes";
import { initialFluidBrickContainsCell } from "../lib/initial-fluid";
import { createTallCellLayout, tallCellSettings, type GPUQuality } from "../lib/tall-cell-grid";
import { createSmokeScenario, isSmokeScenarioId } from "../tools/webgpu-smoke-scenarios";
import { validateScene } from "../lib/model";

import { planOctreePressureCapacity } from "../lib/webgpu-octree";
import { SPGRID_MAXIMUM_ROW_CAPACITY } from "../lib/webgpu-octree-spgrid-vcycle";

const DIMENSIONS = [56, 16, 16] as const;

/** The brick tiers each reservoir must occupy, as `x,y,z` brick coordinates. */
function occupiedBricks(scene: ReturnType<typeof createTwinDamCollisionScene>) {
  const bricks = new Set<string>();
  let wetCells = 0;
  for (let z = 0; z < DIMENSIONS[2]; z += 1) for (let y = 0; y < DIMENSIONS[1]; y += 1) {
    for (let x = 0; x < DIMENSIONS[0]; x += 1) {
      if (!initialFluidBrickContainsCell(scene, x, y, z, DIMENSIONS)) continue;
      wetCells += 1;
      bricks.add(`${Math.floor(x / 8)},${Math.floor(y / 8)},${Math.floor(z / 8)}`);
    }
  }
  return { bricks, wetCells };
}

test("the twin-dam tank resolves to exactly 56x16x16 finest cells at every quality", () => {
  const scene = createTwinDamCollisionScene();
  assert.deepEqual(validateScene(scene), []);
  // Isotropic finest cells are a hard power-catalog requirement, so the
  // authored extents must divide the 0.05 m lattice exactly on all three axes.
  for (const quality of ["balanced", "high", "ultra"] as GPUQuality[]) {
    const layout = createTallCellLayout(scene, quality);
    assert.deepEqual([layout.nx, layout.fineNy, layout.nz], [...DIMENSIONS],
      `${quality} finest grid must be 7x2x2 bricks of 8-cubed cells`);
  }
});

/**
 * The power octree fails construction outright when the pressure plan exceeds
 * the bounded SPGrid V-cycle, and at this domain size the plan saturates at
 * the cell count. Enlarging the tank is therefore a solver-capacity decision,
 * not a purely visual one.
 */
test("the twin-dam lattice fits the bounded SPGrid row capacity", () => {
  const scene = createTwinDamCollisionScene();
  const [nx, ny, nz] = DIMENSIONS;
  const plan = planOctreePressureCapacity({ nx, ny, nz }, 16,
    4, undefined, scene.container.top === "closed");
  assert.ok(plan.rowCapacity <= SPGRID_MAXIMUM_ROW_CAPACITY,
    `pressure rows ${plan.rowCapacity} must fit the bounded ${SPGRID_MAXIMUM_ROW_CAPACITY}-row V-cycle`);
});

/**
 * Measured on Dawn: at 16-cell reservoirs the tall-cell remesh compresses the
 * columns and loses 72% of the volume over 0.6 s, while 8-cell reservoirs that
 * fit inside the default regular-layer band conserve it to 5e-6. Keeping the
 * water inside that band is what lets every method run this scene at its own
 * defaults.
 */
test("the reservoirs fit inside the tall-cell default regular-layer band", () => {
  const scene = createTwinDamCollisionScene();
  const { regularLayers } = tallCellSettings.balanced;
  const seededCellHeight = 8 * (1 + Math.max(...(scene.fluid.initialBrickSeeds_m ?? [])
    .map((seed) => Math.floor(seed.y / (8 * scene.voxelDomain.finestCellSize_m)))));
  assert.ok(seededCellHeight <= regularLayers,
    `seeded water is ${seededCellHeight} cells tall but only ${regularLayers} regular layers are uncompressed`);
});

test("both reservoirs are 2x1x1 brick slabs on diagonally opposite floor corners", () => {
  const scene = getScenePreset("twin-dam-collision").create();
  assert.equal(scene.fluid.initialBrickSeeds_m?.length, 4, "one seed per occupied brick");
  const { bricks, wetCells } = occupiedBricks(scene);
  assert.equal(wetCells, 4 * 8 ** 3, "the seeds must fill exactly four 8-cubed bricks");
  assert.deepEqual([...bricks].sort(), [
    // -x reservoir at the -z wall.
    "0,0,0", "1,0,0",
    // +x reservoir at the +z wall.
    "5,0,1", "6,0,1",
  ].sort());
  // Three dry brick tiers separate them along x, and neither shares a z tier,
  // so the collapsing fronts meet mid-tank at an angle.
  for (const xTier of [2, 3, 4]) {
    for (let y = 0; y < DIMENSIONS[1]; y += 1) for (let z = 0; z < DIMENSIONS[2]; z += 1) {
      assert.ok(!initialFluidBrickContainsCell(scene, xTier * 8 + 4, y, z, DIMENSIONS),
        `brick tier ${xTier} must start dry`);
    }
  }
  // One brick of headroom under the closed lid keeps the columns clear of the
  // ceiling, so neither reservoir starts as a sealed pocket.
  for (let y = 8; y < DIMENSIONS[1]; y += 1) {
    assert.ok(!initialFluidBrickContainsCell(scene, 4, y, 4, DIMENSIONS),
      `layer ${y} is headroom and must start dry`);
  }
  // Both reservoirs are solid through their single seeded brick tier.
  for (let y = 0; y < 8; y += 1) {
    assert.ok(initialFluidBrickContainsCell(scene, 4, y, 4, DIMENSIONS),
      `layer ${y} of the -x reservoir must be wet`);
    assert.ok(initialFluidBrickContainsCell(scene, 44, y, 12, DIMENSIONS),
      `layer ${y} of the +x reservoir must be wet`);
  }
});

test("the seeded reservoirs replace the analytic dam rather than adding to it", () => {
  const scene = createTwinDamCollisionScene();
  // Additive seeds would union these columns with the corner-anchored analytic
  // dam-break box, which is a third body of water this scene does not want.
  assert.equal(scene.fluid.initialBrickSeedsAdditive, undefined);
  assert.equal(scene.fluid.initialCondition, "dam-break");
  assert.equal(scene.fluid.inflow, undefined);
  assert.equal(scene.rigidBodies.length, 0);
});

test("the twin-dam scene is registered in the UI presets and the smoke harness", () => {
  const preset = getScenePreset("twin-dam-collision");
  assert.equal(preset.id, "twin-dam-collision", "the preset must exist rather than fall back to the default scene");
  // What matters is that it is offered rather than disclosed: this is a scene to
  // watch, not an oracle. Its shelf is a display label and may be re-cut.
  assert.equal(getSceneDefinition("twin-dam-collision").audience, "explore");
  assert.ok(isSmokeScenarioId("twin-dam-collision"));
  const scenario = createSmokeScenario("twin-dam-collision");
  assert.deepEqual(validateScene(scenario.scene), []);
  assert.equal(scenario.scene.voxelDomain.finestCellSize_m, 0.05);
  assert.equal(scenario.scene.voxelDomain.brickSize_cells, 8);
  assert.equal(scenario.scene.fluid.initialBrickSeeds_m?.length, 4);
  assert.deepEqual(
    [scenario.scene.container.width_m, scenario.scene.container.height_m, scenario.scene.container.depth_m],
    [2.8, 0.8, 0.8],
  );
});
