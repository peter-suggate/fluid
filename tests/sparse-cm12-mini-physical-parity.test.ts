import assert from "node:assert/strict";
import test from "node:test";
import { baseInitialLiquidFractionAtCell } from "../lib/core/initial-fluid";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice";
import {
  createMinimalPowerDamBreak32Scene,
  createMinimalPowerDamBreak64Scene,
  createMinimalPowerDamBreakScene,
} from "../lib/core/scenes";
import {
  initializeSparseBrickAtlasFromScene,
  materializeSparseBrickAtlasDensity,
  sparseCM12InitialActiveBrickKeys,
  sparseAtlasLeaves,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";

const PAIRS = [{
  fineResolution: 32,
  fineScene: createMinimalPowerDamBreak32Scene,
  coarseScene: createMinimalPowerDamBreakScene,
}, {
  fineResolution: 64,
  fineScene: createMinimalPowerDamBreak64Scene,
  coarseScene: createMinimalPowerDamBreak32Scene,
}] as const;

function withTwoCellFloor(scene: ReturnType<typeof createMinimalPowerDamBreakScene>) {
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

for (const pair of PAIRS) {
  const coarseResolution = pair.fineResolution / 2;
  test(`procedural dam overlap restricts exactly from mini${pair.fineResolution}`
    + ` to mini${coarseResolution}`, () => {
    const coarseScene = pair.coarseScene();
    const fineScene = pair.fineScene();
    for (let z = 0; z < coarseResolution; z += 1) {
      for (let y = 0; y < coarseResolution; y += 1) {
        for (let x = 0; x < coarseResolution; x += 1) {
      let restricted = 0;
      for (let dz = 0; dz < 2; dz += 1) for (let dy = 0; dy < 2; dy += 1) {
        for (let dx = 0; dx < 2; dx += 1) {
          restricted += baseInitialLiquidFractionAtCell(
            fineScene, 2 * x + dx, 2 * y + dy, 2 * z + dz,
            [pair.fineResolution, pair.fineResolution, pair.fineResolution],
          ) / 8;
        }
      }
      assert.equal(restricted,
        baseInitialLiquidFractionAtCell(coarseScene, x, y, z,
          [coarseResolution, coarseResolution, coarseResolution]));
        }
      }
    }
  });

  test(`Sparse CM12 mini${pair.fineResolution} region and mini${coarseResolution}`
    + " seed the same physical state", () => {
    const fineScene = withTwoCellFloor(pair.fineScene());
    const coarseScene = pair.coarseScene();
    const fineAtlas = initializeSparseBrickAtlasFromScene(fineScene, {
    finestDimensions: sceneLatticeDimensions(fineScene),
    brickFineResolution: 8,
    surfaceFineRings: 1,
  });
    const coarseAtlas = initializeSparseBrickAtlasFromScene(coarseScene, {
    finestDimensions: sceneLatticeDimensions(coarseScene),
    brickFineResolution: 8,
    surfaceFineRings: 1,
  });
    const fineDensity = materializeSparseBrickAtlasDensity(fineAtlas);
    const coarseDensity = materializeSparseBrickAtlasDensity(coarseAtlas);
    for (let z = 0; z < coarseResolution; z += 1) {
      for (let y = 0; y < coarseResolution; y += 1) {
        for (let x = 0; x < coarseResolution; x += 1) {
      let restricted = 0;
      for (let dz = 0; dz < 2; dz += 1) for (let dy = 0; dy < 2; dy += 1) {
        for (let dx = 0; dx < 2; dx += 1) {
          restricted += fineDensity[2 * x + dx + pair.fineResolution
            * ((2 * y + dy) + pair.fineResolution * (2 * z + dz))]! / 8;
        }
      }
      assert.equal(restricted, coarseDensity[x + coarseResolution
        * (y + coarseResolution * z)]);
        }
      }
    }
    const mass = (atlas: typeof coarseAtlas, cellSize_m: number) => sparseAtlasLeaves(atlas)
    .reduce((sum, leaf) => sum + leaf.density * leaf.volumeFineCells, 0)
    * cellSize_m ** 3;
    const fineMass = mass(fineAtlas, fineScene.voxelDomain.finestCellSize_m);
    const coarseMass = mass(coarseAtlas, coarseScene.voxelDomain.finestCellSize_m);
    assert.ok(Math.abs(fineMass - coarseMass) < 2e-15,
      `physical mass differs: ${fineMass} versus ${coarseMass}`);

    // The two-cell refinement floor is a physical policy scale, not merely a
    // per-brick resolution cap. Each represented fine sub-brick must select
    // the rung corresponding to its enclosing coarse brick.
    const coarseBrickResolution = coarseResolution / 8;
    for (const brick of fineAtlas.bricks) {
      const parentCoordinate = brick.coordinate.map((value) =>
        Math.floor(value / 2)) as [number, number, number];
      const parentKey = parentCoordinate[0] + coarseBrickResolution
        * (parentCoordinate[1] + coarseBrickResolution * parentCoordinate[2]);
      const parent = coarseAtlas.directory.get(parentKey);
      assert.ok(parent, `Mini${pair.fineResolution} brick ${brick.coordinate.join(",")}`
        + ` has no Mini${coarseResolution} policy tile`);
      assert.equal(2 * brick.resolution, parent.resolution,
        `Mini${pair.fineResolution} brick ${brick.coordinate.join(",")}`
        + " chose a different physical rung");
    }

    const fineActive = sparseCM12InitialActiveBrickKeys(fineScene, fineAtlas);
    const coarseActive = sparseCM12InitialActiveBrickKeys(coarseScene, coarseAtlas);
    assert.equal(fineActive.size, 8 * coarseActive.size,
      "each active coarse brick must have eight active fine policy siblings");
    for (const key of fineActive) {
      const brick = fineAtlas.directory.get(key)!;
      const parentCoordinate = brick.coordinate.map((value) =>
        Math.floor(value / 2)) as [number, number, number];
      const parentKey = parentCoordinate[0] + coarseBrickResolution
        * (parentCoordinate[1] + coarseBrickResolution * parentCoordinate[2]);
      assert.ok(coarseActive.has(parentKey),
        `active Mini${pair.fineResolution} brick has inactive parent ${parentKey}`);
    }
  });
}
