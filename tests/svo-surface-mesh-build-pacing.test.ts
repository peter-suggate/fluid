import assert from "node:assert/strict";
import test from "node:test";

import {
  SVO_SURFACE_MESH_BUILD_BATCHES_INITIAL,
  SVO_SURFACE_MESH_BUILD_BATCHES_MAXIMUM,
  SVO_SURFACE_MESH_BUILD_BRICKS_PER_FRAME,
  surfaceMeshBuildBatches,
  surfaceMeshBuildPresentations,
} from "../lib/svo/features/primary-visibility/svo-surface-mesh";

test("a build's first presentation stays at the cheap initial batch count", () => {
  assert.equal(surfaceMeshBuildBatches(0), SVO_SURFACE_MESH_BUILD_BATCHES_INITIAL);
  assert.equal(surfaceMeshBuildBatches(-3), SVO_SURFACE_MESH_BUILD_BATCHES_INITIAL);
  // One edit's rebuild of a few hundred bricks completes in that one presentation.
  assert.equal(surfaceMeshBuildPresentations(3 * SVO_SURFACE_MESH_BUILD_BRICKS_PER_FRAME), 1);
});

test("batches double per pending presentation and hold at the ceiling", () => {
  assert.equal(surfaceMeshBuildBatches(1), 2 * SVO_SURFACE_MESH_BUILD_BATCHES_INITIAL);
  assert.equal(surfaceMeshBuildBatches(2), 4 * SVO_SURFACE_MESH_BUILD_BATCHES_INITIAL);
  assert.equal(surfaceMeshBuildBatches(3), SVO_SURFACE_MESH_BUILD_BATCHES_MAXIMUM);
  assert.equal(surfaceMeshBuildBatches(40), SVO_SURFACE_MESH_BUILD_BATCHES_MAXIMUM);
  assert.equal(surfaceMeshBuildBatches(1e9), SVO_SURFACE_MESH_BUILD_BATCHES_MAXIMUM);
});

test("hero-garden-hose-x10's whole-world build needs tens of presentations, not hundreds", () => {
  // The captured build: 595,825 bricks at 2,048 per presentation was ~291
  // presentations, each paying the ~83 ms full-resolution fallback trace.
  const bricks = 595_825;
  const fixed = Math.ceil(bricks / (SVO_SURFACE_MESH_BUILD_BATCHES_INITIAL * SVO_SURFACE_MESH_BUILD_BRICKS_PER_FRAME));
  assert.equal(fixed, 291);
  const ramped = surfaceMeshBuildPresentations(bricks);
  assert.ok(ramped <= 40, `${ramped} presentations`);
  assert.ok(ramped >= Math.ceil(bricks / (SVO_SURFACE_MESH_BUILD_BATCHES_MAXIMUM * SVO_SURFACE_MESH_BUILD_BRICKS_PER_FRAME)));
});
