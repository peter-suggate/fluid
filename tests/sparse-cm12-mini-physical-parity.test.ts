import assert from "node:assert/strict";
import test from "node:test";
import { baseInitialLiquidFractionAtCell } from "../lib/core/initial-fluid";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice";
import {
  createMinimalPowerDamBreak32Scene,
  createMinimalPowerDamBreakScene,
} from "../lib/core/scenes";
import {
  initializeSparseBrickAtlasFromScene,
  materializeSparseBrickAtlasDensity,
  sparseAtlasLeaves,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";

function mini32WithTwoCellFloor() {
  const scene = createMinimalPowerDamBreak32Scene();
  scene.fluid.refinementRegions = [{
    id: "mini-physical-parity-two-cell-floor",
    rule: "minimum-cell-size",
    minimumCellSize_cells: 2,
    min_m: { x: -0.5 * scene.container.width_m, y: 0,
      z: -0.5 * scene.container.depth_m },
    max_m: { x: 0.5 * scene.container.width_m, y: scene.container.height_m,
      z: 0.5 * scene.container.depth_m },
  }];
  return scene;
}

test("procedural dam overlap restricts exactly from mini32 to mini16", () => {
  const mini16 = createMinimalPowerDamBreakScene();
  const mini32 = createMinimalPowerDamBreak32Scene();
  for (let z = 0; z < 16; z += 1) for (let y = 0; y < 16; y += 1) {
    for (let x = 0; x < 16; x += 1) {
      let restricted = 0;
      for (let dz = 0; dz < 2; dz += 1) for (let dy = 0; dy < 2; dy += 1) {
        for (let dx = 0; dx < 2; dx += 1) {
          restricted += baseInitialLiquidFractionAtCell(
            mini32, 2 * x + dx, 2 * y + dy, 2 * z + dz, [32, 32, 32],
          ) / 8;
        }
      }
      assert.equal(restricted,
        baseInitialLiquidFractionAtCell(mini16, x, y, z, [16, 16, 16]));
    }
  }
});

test("Sparse CM12 mini32 region and mini16 seed the same physical density", () => {
  const mini32 = mini32WithTwoCellFloor();
  const mini16 = createMinimalPowerDamBreakScene();
  const atlas32 = initializeSparseBrickAtlasFromScene(mini32, {
    finestDimensions: sceneLatticeDimensions(mini32),
    brickFineResolution: 8,
    surfaceFineRings: 1,
  });
  const atlas16 = initializeSparseBrickAtlasFromScene(mini16, {
    finestDimensions: sceneLatticeDimensions(mini16),
    brickFineResolution: 8,
    surfaceFineRings: 1,
  });
  const density32 = materializeSparseBrickAtlasDensity(atlas32);
  const density16 = materializeSparseBrickAtlasDensity(atlas16);
  for (let z = 0; z < 16; z += 1) for (let y = 0; y < 16; y += 1) {
    for (let x = 0; x < 16; x += 1) {
      let restricted = 0;
      for (let dz = 0; dz < 2; dz += 1) for (let dy = 0; dy < 2; dy += 1) {
        for (let dx = 0; dx < 2; dx += 1) {
          restricted += density32[2 * x + dx
            + 32 * ((2 * y + dy) + 32 * (2 * z + dz))]! / 8;
        }
      }
      assert.equal(restricted, density16[x + 16 * (y + 16 * z)]);
    }
  }
  const mass = (atlas: typeof atlas16, cellSize_m: number) => sparseAtlasLeaves(atlas)
    .reduce((sum, leaf) => sum + leaf.density * leaf.volumeFineCells, 0)
    * cellSize_m ** 3;
  assert.equal(mass(atlas32, mini32.voxelDomain.finestCellSize_m), 0.18400000000000039);
  assert.equal(mass(atlas16, mini16.voxelDomain.finestCellSize_m), 0.18400000000000039);

  // The two-cell refinement floor is a physical policy scale, not merely a
  // per-brick resolution cap. Each represented Mini32 sub-brick must select
  // the rung corresponding to its enclosing Mini16 brick.
  for (const brick of atlas32.bricks) {
    const parentCoordinate = brick.coordinate.map((value) =>
      Math.floor(value / 2)) as [number, number, number];
    const parentKey = parentCoordinate[0] + 2
      * (parentCoordinate[1] + 2 * parentCoordinate[2]);
    const parent = atlas16.directory.get(parentKey);
    assert.ok(parent, `Mini32 brick ${brick.coordinate.join(",")} has no Mini16 policy tile`);
    assert.equal(2 * brick.resolution, parent.resolution,
      `Mini32 brick ${brick.coordinate.join(",")} chose a different physical rung`);
  }
});
