import assert from "node:assert/strict";
import test from "node:test";

import {
  SVO_SURFACE_MESH_BUILD_BRICKS_DRAWN_MAXIMUM,
  SVO_SURFACE_MESH_BUILD_BRICKS_INITIAL,
  SVO_SURFACE_MESH_BUILD_BRICKS_UNDRAWN_MAXIMUM,
  surfaceMeshAllowsLiveStartup,
  surfaceMeshBuildBricks,
  surfaceMeshBuildPresentations,
  surfaceMeshConstructionFailed,
  surfaceMeshWorkBytes,
} from "../lib/svo/features/primary-visibility/svo-surface-mesh";

test("a build's first presentation extracts the cheap initial brick count", () => {
  assert.equal(surfaceMeshBuildBricks(0, true), SVO_SURFACE_MESH_BUILD_BRICKS_INITIAL);
  assert.equal(surfaceMeshBuildBricks(-3, false), SVO_SURFACE_MESH_BUILD_BRICKS_INITIAL);
  // One edit's re-extraction of a few hundred bricks completes in that one presentation.
  assert.equal(surfaceMeshBuildPresentations(SVO_SURFACE_MESH_BUILD_BRICKS_INITIAL - 1, true), 1);
});

test("bricks double per pending presentation and hold at the ceiling of the drawn or undrawn mesh", () => {
  assert.equal(surfaceMeshBuildBricks(1, true), 2 * SVO_SURFACE_MESH_BUILD_BRICKS_INITIAL);
  assert.equal(surfaceMeshBuildBricks(2, true), SVO_SURFACE_MESH_BUILD_BRICKS_DRAWN_MAXIMUM);
  assert.equal(surfaceMeshBuildBricks(40, true), SVO_SURFACE_MESH_BUILD_BRICKS_DRAWN_MAXIMUM);
  assert.equal(surfaceMeshBuildBricks(5, false), SVO_SURFACE_MESH_BUILD_BRICKS_UNDRAWN_MAXIMUM);
  assert.equal(surfaceMeshBuildBricks(1e9, false), SVO_SURFACE_MESH_BUILD_BRICKS_UNDRAWN_MAXIMUM);
  assert.ok(SVO_SURFACE_MESH_BUILD_BRICKS_DRAWN_MAXIMUM < SVO_SURFACE_MESH_BUILD_BRICKS_UNDRAWN_MAXIMUM);
});

test("hero-garden-hose-x10's whole-world build needs tens of presentations while undrawn", () => {
  // The captured build: 595,825 bricks at 2,048 per presentation was ~291
  // presentations, each paying the ~83 ms full-resolution fallback trace.
  const bricks = 595_825;
  const ramped = surfaceMeshBuildPresentations(bricks, false);
  assert.ok(ramped <= 45, `${ramped} presentations`);
  assert.ok(ramped >= Math.ceil(bricks / SVO_SURFACE_MESH_BUILD_BRICKS_UNDRAWN_MAXIMUM));
  // Beside a drawn mesh the same build is paced for interactivity instead.
  assert.ok(surfaceMeshBuildPresentations(bricks, true) >= Math.ceil(bricks / SVO_SURFACE_MESH_BUILD_BRICKS_DRAWN_MAXIMUM));
});

test("smooth reconstruction withholds the mesh by policy and must not halt live SVO startup", () => {
  const smooth = { state: "blocked" as const, fallbackReason: "smooth" as const };
  const inside = { state: "blocked" as const, fallbackReason: "inside-solid" as const };
  const waiting = { state: "blocked" as const, fallbackReason: "publication" as const };
  const budget = { state: "blocked" as const, fallbackReason: "budget" as const };
  const extraction = { state: "blocked" as const, fallbackReason: "extraction" as const };
  assert.equal(surfaceMeshConstructionFailed(smooth), false);
  assert.equal(surfaceMeshAllowsLiveStartup(smooth), true);
  assert.equal(surfaceMeshConstructionFailed(inside), false);
  assert.equal(surfaceMeshAllowsLiveStartup(inside), true);
  assert.equal(surfaceMeshConstructionFailed(waiting), false);
  assert.equal(surfaceMeshAllowsLiveStartup(waiting), false);
  assert.equal(surfaceMeshConstructionFailed(budget), true);
  assert.equal(surfaceMeshAllowsLiveStartup(budget), false);
  assert.equal(surfaceMeshConstructionFailed(extraction), true);
  assert.equal(surfaceMeshAllowsLiveStartup(extraction), false);
  assert.equal(surfaceMeshAllowsLiveStartup(undefined), true);
  assert.equal(surfaceMeshAllowsLiveStartup({ state: "ready" }), true);
  assert.equal(surfaceMeshAllowsLiveStartup({ state: "pending" }), false);
});
