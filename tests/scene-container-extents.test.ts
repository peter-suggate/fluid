import assert from "node:assert/strict";
import test from "node:test";

import { sceneDocument } from "../lib/core/scene-definition";
import { sceneAtContainerExtents } from "../lib/core/scene-scale";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneLatticeDimensions, solidVoxelEditsForScene,
  solidVoxelShellForScene } from "../lib/core/scene-lattice";
import { sampleSolidWorld, solidWorldForScene } from "../lib/core/solid-world";
import { initializeSparseBrickAtlasFromScene } from
  "../lib/methods/adaptive-mass/sparse-brick-atlas";

test("resizing a finished tank moves its voxel shell instead of stranding it", () => {
  const original = sceneDocument(getSceneDefinition("water-box-tank-fill"));
  assert.deepEqual(sceneLatticeDimensions(original), [24, 16, 16]);

  const resized = sceneAtContainerExtents(original, {
    width_m: 1.6,
    height_m: 2.4,
    depth_m: 1.6,
  });
  assert.deepEqual(sceneLatticeDimensions(resized), [32, 48, 32]);
  assert.deepEqual(resized.solidVoxels, solidVoxelShellForScene(resized));
  assert.deepEqual(solidVoxelEditsForScene(resized), []);

  const world = solidWorldForScene(resized);
  for (const coordinate of [[24, 8, 8], [8, 16, 8], [8, 8, 16]] as const) {
    assert.equal(sampleSolidWorld(world, coordinate).solidFraction, 0,
      `old shell plane ${coordinate.join(",")} must be fluid space`);
  }
  for (const coordinate of [[-1, 8, 8], [32, 8, 8], [8, -1, 8],
    [8, 48, 8], [8, 8, -1], [8, 8, 32]] as const) {
    assert.equal(sampleSolidWorld(world, coordinate).solidFraction, 1,
      `resized boundary ${coordinate.join(",")} must be solid`);
  }

  // Structural editing is immutable: callers that still hold the source scene
  // must continue to see its original lattice and shell.
  assert.deepEqual(sceneLatticeDimensions(original), [24, 16, 16]);
  assert.deepEqual(original.solidVoxels, solidVoxelShellForScene(original));
});

test("a resized deep tank exposes the complete effective 8/4/2/1 ladder", () => {
  const resized = sceneAtContainerExtents(
    sceneDocument(getSceneDefinition("water-box-tank-fill")),
    { width_m: 1.6, height_m: 2.4, depth_m: 1.6 },
  );
  resized.container.fillFraction = 0.7;
  const atlas = initializeSparseBrickAtlasFromScene(resized, {
    finestDimensions: [32, 48, 32],
    brickFineResolution: 8,
    surfaceFineRings: 1,
  });
  const effectiveRungs = new Set(atlas.bricks.map((brick) =>
    brick.resolution / (brick.spanBricks ?? 1)));
  assert.deepEqual([...effectiveRungs].sort((left, right) => left - right),
    [1, 2, 4, 8]);

  const bottom = atlas.bricks.filter((brick) => brick.coordinate[1] === 0);
  assert.equal(bottom.length, 4);
  assert.ok(bottom.every((brick) => brick.resolution / (brick.spanBricks ?? 1) === 1),
    "every wall/corner macro leaf on the bottom must reach the effective B1 rung");
});
