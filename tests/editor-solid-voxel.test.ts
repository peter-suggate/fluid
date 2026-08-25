import assert from "node:assert/strict";
import test from "node:test";
import { pickSolidVoxel, solidVoxelClearPreview,
  withSolidVoxelClearRegion } from "../lib/core/editor-solid-voxel";
import { cloneScene, defaultScene } from "../lib/core/model";
import { sceneCellSizes_m } from "../lib/core/scene-lattice";
import { sampleSolidWorld, solidWorldForScene } from "../lib/core/solid-world";

test("generic clear selection uses exact occupied voxels and preserves prior edits", () => {
  const scene = cloneScene(defaultScene);
  scene.solidVoxels = [
    { operation: "fill", minimum: [3, 4, -2], maximumExclusive: [5, 6, 0],
      materialId: 2 },
    { operation: "clear", minimum: [4, 4, -2], maximumExclusive: [5, 5, -1] },
    { operation: "fill", minimum: [1_000_000, 1, 20],
      maximumExclusive: [1_000_001, 2, 21],
      materialId: 3 },
  ];
  const region = { minimum: [3, 4, -2], maximumExclusive: [5, 6, 0] } as const;
  const preview = solidVoxelClearPreview(scene, region);
  assert.equal(preview.affectedCount, 7);
  assert.equal(preview.coordinates.length, 7);

  const prior = structuredClone(scene.solidVoxels);
  scene.solidVoxels = withSolidVoxelClearRegion(scene.solidVoxels, region);
  assert.deepEqual(scene.solidVoxels.slice(0, prior.length), prior);
  const world = solidWorldForScene(scene);
  assert.equal(sampleSolidWorld(world, [3, 4, -2]).solidFraction, 0);
  assert.equal(sampleSolidWorld(world, [1_000_000, 1, 20]).materialId, 3);

  scene.solidVoxels = prior;
  const [hx, hy, hz] = sceneCellSizes_m(scene);
  const originX = -0.5 * scene.container.width_m;
  const originZ = -0.5 * scene.container.depth_m;
  const picked = pickSolidVoxel(scene, {
    origin: { x: originX + 2 * hx, y: 4.5 * hy, z: originZ - 1.5 * hz },
    direction: { x: 1, y: 0, z: 0 },
  });
  assert.deepEqual(picked?.coordinate, [3, 4, -2]);
  assert.equal(picked?.faceAxis, 0);
  assert.equal(picked?.faceSign, -1);
});

test("preview detail truncation never limits a large generic clear edit", () => {
  const scene = cloneScene(defaultScene);
  scene.solidVoxels = [{ operation: "fill", minimum: [-700, 9, 31],
    maximumExclusive: [-187, 11, 32], materialId: 2 }];
  const region = { minimum: [-700, 9, 31],
    maximumExclusive: [-187, 11, 32] } as const;
  const preview = solidVoxelClearPreview(scene, region);
  assert.equal(preview.affectedCount, 1_026);
  assert.equal(preview.coordinates.length, 512);
  assert.equal(preview.truncated, true);

  scene.solidVoxels = withSolidVoxelClearRegion(scene.solidVoxels, region);
  assert.equal(scene.solidVoxels.at(-1)?.operation, "clear");
  assert.equal(sampleSolidWorld(solidWorldForScene(scene), [-188, 10, 31])
    .solidFraction, 0);
});
