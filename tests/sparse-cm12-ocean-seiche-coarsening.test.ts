import assert from "node:assert/strict";
import test from "node:test";
import { createOceanSeicheScene } from "../lib/core/scenes";
import {
  initializeSparseBrickAtlasFromScene,
  sparseBrickAtlasStats,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { adaptiveMassPresentationDimensionsForScene } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

test("ocean seiche grades its accepted deep-water topology through 8/4/2/1", () => {
  const scene = createOceanSeicheScene();
  const atlas = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: adaptiveMassPresentationDimensionsForScene(scene),
  });
  const stats = sparseBrickAtlasStats(atlas);

  for (const resolution of [1, 2, 4, 8] as const) {
    assert.ok(atlas.bricks.some((brick) => brick.resolution === resolution),
      `ocean-seiche is missing its ${resolution}^3 rung`);
  }
  assert.ok(stats.leafCompressionRatio > 10,
    `deep-water compression regressed to ${stats.leafCompressionRatio}`);
  assert.equal(stats.integratedMassFineCells, 1_853_440,
    "coarsening must preserve the authored pool and raised-slab mass exactly");

  const verticalRungs = (x: number, z: number) => Array.from({ length: 9 }, (_, y) =>
    atlas.directory.get(x + atlas.brickDimensions[0]
      * (y + atlas.brickDimensions[1] * z))?.resolution);
  assert.deepEqual(verticalRungs(20, 5), [1, 1, 1, 1, 1, 1, 2, 4, 8],
    "calm water must become progressively coarser below the free surface");

  for (const brick of atlas.bricks) {
    for (let axis = 0; axis < 3; axis += 1) {
      const coordinate = [...brick.coordinate] as [number, number, number];
      coordinate[axis] += 1;
      if (coordinate[axis] >= atlas.brickDimensions[axis]) continue;
      const key = coordinate[0] + atlas.brickDimensions[0]
        * (coordinate[1] + atlas.brickDimensions[1] * coordinate[2]);
      const neighbor = atlas.directory.get(key);
      if (!neighbor) continue;
      assert.ok(Math.max(brick.resolution, neighbor.resolution)
        <= 2 * Math.min(brick.resolution, neighbor.resolution),
      `brick ${brick.key}/${neighbor.key} exceeds 2:1 grading`);
    }
  }
});
