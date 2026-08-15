import assert from "node:assert/strict";
import test from "node:test";
import { createOceanSeicheScene } from "../lib/core/scenes";
import {
  initializeSparseBrickAtlasFromScene,
  sparseBrickContainingCoordinate,
  sparseBrickAtlasStats,
  sparseBrickSpan,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { adaptiveMassPresentationDimensionsForScene } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

test("ocean seiche collapses deep water into graded macro-bricks", () => {
  const scene = createOceanSeicheScene();
  const atlas = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: adaptiveMassPresentationDimensionsForScene(scene),
  });
  const stats = sparseBrickAtlasStats(atlas);

  assert.ok(atlas.maximumSpanBricks >= 4, "deep water never formed a macro-brick");
  assert.ok(stats.residentBrickCount < stats.logicalBrickCount / 3,
    "resident brick storage regressed toward the wet fixed-brick volume");
  assert.ok(stats.leafCompressionRatio > 9.5,
    `deep-water compression regressed to ${stats.leafCompressionRatio}`);
  assert.equal(stats.integratedMassFineCells, 1_853_440,
    "coarsening must preserve the authored pool and raised-slab mass exactly");

  const verticalRungs = (x: number, z: number) => Array.from({ length: 9 }, (_, y) => {
    const brick = sparseBrickContainingCoordinate(atlas, [x, y, z]);
    return brick && 8 * sparseBrickSpan(brick) / brick.resolution;
  });
  assert.deepEqual(verticalRungs(20, 5), [8, 8, 8, 8, 4, 4, 4, 2, 1],
    "calm water must become progressively coarser below the free surface");

  for (const brick of atlas.bricks) {
    for (let axis = 0; axis < 3; axis += 1) {
      const coordinate = [...brick.coordinate] as [number, number, number];
      coordinate[axis] += sparseBrickSpan(brick);
      if (coordinate[axis] >= atlas.brickDimensions[axis]) continue;
      const neighbor = sparseBrickContainingCoordinate(atlas, coordinate);
      if (!neighbor) continue;
      const ownWidth = 8 * sparseBrickSpan(brick) / brick.resolution;
      const neighborWidth = 8 * sparseBrickSpan(neighbor) / neighbor.resolution;
      assert.ok(Math.max(ownWidth, neighborWidth) <= 2 * Math.min(ownWidth, neighborWidth),
      `brick ${brick.key}/${neighbor.key} exceeds 2:1 grading`);
    }
  }
});

test("closed vast-depth fills do no finest-domain-shaped planning or allocation", () => {
  const scene = createOceanSeicheScene();
  scene.container = { ...scene.container, fillFraction: 1 };
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.initialBrickSeedsAdditive;
  const atlas = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: [256, 1_048_576, 256],
  });
  const logicalBricks = 32 * 131_072 * 32;
  assert.equal(atlas.brickDimensions.reduce((product, value) => product * value, 1),
    logicalBricks);
  assert.ok(atlas.bricks.length <= 4096,
    `${atlas.bricks.length} leaves suggests wet-volume enumeration`);
  assert.ok(atlas.maximumSpanBricks >= 32);
  assert.ok(atlas.bricks.every((brick) => brick.density.length === 1),
    "closed quiescent macro-bricks should allocate one physical cell each");
});
