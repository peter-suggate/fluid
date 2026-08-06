import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  DEFAULT_ENVIRONMENT_BRICK_REFINEMENT_LEVELS,
  ENVIRONMENT_MAXIMUM_COARSENING_POWER,
  environmentMaximumCoarseningPower,
  planOctreeBrickCoordinates,
  sparseSceneOctreeMaximumDepth,
} from "../lib/webgpu-octree-sparse-bricks";

test("octree sparse-brick planning covers the real balanced dam-break lattice", () => {
  const plan = planOctreeBrickCoordinates([61, 46, 41], 8);
  assert.deepEqual(plan.brickDimensions, [8, 6, 6]);
  assert.equal(plan.coordinates.length, 288);
  assert.deepEqual(plan.coordinates[0], { x: 0, y: 0, z: 0 });
  assert.deepEqual(plan.coordinates.at(-1), { x: 7, y: 5, z: 5 });
});

/**
 * The producer used to also materialize a 48-byte expanded record per resolved
 * voxel and per leaf, into two arenas it owned, for the render panel's
 * LEVELS/SURFACE/BRICKS/CONTENT views. Those views and their renderer are gone;
 * this pins that the arenas, their publication shader and their dispatch went
 * with them, because the cost of the lane was the arenas (~295 MB on the
 * widened ocean scene) rather than the buttons.
 */
test("the producer no longer allocates or publishes expanded inspection records", () => {
  const source = readFileSync(new URL("../lib/webgpu-octree-sparse-bricks.ts", import.meta.url), "utf8");
  for (const removed of [
    "debugPublicationShader", "tiledDebugDispatch", "ensureInspectionSource",
    "encodeInspectionPublication", "SPARSE_VOXEL_DEBUG_RECORD_STRIDE",
  ]) assert.doesNotMatch(source, new RegExp(removed), `${removed} must not survive the inspection removal`);
  assert.doesNotMatch(source, /Sparse voxel debug records|Sparse brick debug records/,
    "neither expanded record arena may still be allocated");
  assert.match(source, /get allocatedBytes\(\): number \{ return this\.baseAllocatedBytes; \}/,
    "the world's footprint is now fully known at build time, with no lazy inspection growth");
});

test("scene environment defaults one level deeper than the previous brick plan", () => {
  assert.equal(ENVIRONMENT_MAXIMUM_COARSENING_POWER, 1);
  assert.equal(DEFAULT_ENVIRONMENT_BRICK_REFINEMENT_LEVELS, 1);
  assert.equal(environmentMaximumCoarseningPower(0), 1);
  assert.equal(environmentMaximumCoarseningPower(), 0);
});

test("sparse scene root depth covers empty positive authored bounds", () => {
  // Allocated solver terminals occupy bricks 0..1, while an authored positive
  // bound extends the declared address space through brick 32.
  const coordinates = planOctreeBrickCoordinates([8, 8, 8], 4).coordinates;
  const maximumDepth = sparseSceneOctreeMaximumDepth([33, 2, 2], coordinates);
  assert.equal(maximumDepth, 6);
  assert.ok(2 ** maximumDepth >= 33,
    "every in-domain cell bit must fit in the root instead of aliasing a low coordinate");
});
