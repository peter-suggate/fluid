import assert from "node:assert/strict";
import test from "node:test";

import {
  SVO_SURFACE_MESH_BUILD_BRICKS_DRAWN_MAXIMUM,
  SVO_SURFACE_MESH_BUILD_BRICKS_INITIAL,
  SVO_SURFACE_MESH_BUILD_BRICKS_UNDRAWN_MAXIMUM,
  SVO_SURFACE_MESH_BYTES,
  SVO_SURFACE_MESH_FLAGS,
  SVO_SURFACE_MESH_MODE,
  SVO_SURFACE_MESH_QUAD_BYTES,
  SVO_SURFACE_MESH_STATE,
  interpretSurfaceMeshState,
  surfaceMeshBuildBricks,
  surfaceMeshBuildPresentations,
  surfaceMeshWorkBytes,
} from "../lib/svo/features/primary-visibility/svo-surface-mesh";

const W = SVO_SURFACE_MESH_STATE;
const words = (assignments: Partial<Record<keyof typeof W, number>>): Uint32Array => {
  const out = new Uint32Array(W.wordCount);
  for (const [key, value] of Object.entries(assignments)) out[W[key as keyof typeof W]] = value!;
  return out;
};
const context = { arenaBytes: [SVO_SURFACE_MESH_BYTES, 64] as const, maximumBytes: 4 * SVO_SURFACE_MESH_BYTES };

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

test("the work buffer holds boxes, one table slot, one scratch record and one worklist entry per leaf", () => {
  assert.equal(surfaceMeshWorkBytes(1000) - surfaceMeshWorkBytes(0), 999 * 9 * 4 + 0);
  assert.equal(surfaceMeshWorkBytes(1), (64 * 8 + 9) * 4);
});

test("a receipt of a complete drawn mesh is ready and asks nothing of the host", () => {
  const receipt = interpretSurfaceMeshState(words({ usable: 1, frontCursor: 5000, liveQuads: 4200, drawInstanceCount: 1200, builds: 3 }), context);
  assert.equal(receipt.status.state, "ready");
  assert.equal(receipt.status.drawn, true);
  assert.equal(receipt.status.liveQuads, 4200);
  assert.equal(receipt.status.capacityQuads, SVO_SURFACE_MESH_BYTES / SVO_SURFACE_MESH_QUAD_BYTES);
  assert.equal(receipt.front, 0);
  assert.equal(receipt.needBackBytes, undefined);
  assert.equal(receipt.grow, undefined);
});

test("an incremental build keeps the mesh drawn and reports its bricks", () => {
  const receipt = interpretSurfaceMeshState(words({ usable: 1, building: 1, mode: SVO_SURFACE_MESH_MODE.incremental,
    worklistCount: 40, processedBricks: 12, restartReason: 3 }), context);
  assert.equal(receipt.status.state, "pending");
  assert.equal(receipt.status.drawn, true);
  assert.equal(receipt.status.buildKind, "incremental");
  assert.equal(receipt.status.restartReason, "geometry");
  assert.equal(receipt.status.completedBricks, 12);
  assert.equal(receipt.status.totalBricks, 40);
  assert.match(receipt.status.detail!, /stays drawn/);
});

test("a replacement build waiting for storage asks for a back arena sized from its wanted quads", () => {
  const wanted = 3_000_000;
  const receipt = interpretSurfaceMeshState(words({ usable: 1, building: 1, mode: SVO_SURFACE_MESH_MODE.replacement,
    flags: SVO_SURFACE_MESH_FLAGS.needBack, wantedBackQuads: wanted, restartReason: 5 }), context);
  assert.equal(receipt.status.state, "pending");
  assert.equal(receipt.status.buildPhase, "capacity");
  assert.equal(receipt.status.restartReason, "compaction");
  assert.equal(receipt.needBackBytes, wanted * SVO_SURFACE_MESH_QUAD_BYTES);
  // Once the GPU has taken a back arena the request is over, even if the flag lingers.
  const started = interpretSurfaceMeshState(words({ usable: 1, building: 1, mode: SVO_SURFACE_MESH_MODE.replacement,
    flags: SVO_SURFACE_MESH_FLAGS.needBack | SVO_SURFACE_MESH_FLAGS.started, wantedBackQuads: wanted }), context);
  assert.equal(started.needBackBytes, undefined);
  assert.equal(started.status.buildPhase, "extracting");
});

test("an overflow names the arena that must grow and only until the GPU has seen the larger binding", () => {
  const overflowQuads = SVO_SURFACE_MESH_BYTES / SVO_SURFACE_MESH_QUAD_BYTES;
  const front = interpretSurfaceMeshState(words({ usable: 1, building: 1, mode: SVO_SURFACE_MESH_MODE.incremental,
    errorFlags: 1, overflowLength: overflowQuads }), context);
  assert.deepEqual(front.grow, { slot: 0, overflowQuads });
  assert.equal(front.status.buildPhase, "capacity");
  // A replacement build's overflow is the back arena's.
  const back = interpretSurfaceMeshState(words({ usable: 1, building: 1, mode: SVO_SURFACE_MESH_MODE.replacement,
    flags: SVO_SURFACE_MESH_FLAGS.started, errorFlags: 1, overflowLength: 2 }), context);
  assert.deepEqual(back.grow, { slot: 1, overflowQuads: 2 });
  // The receipt that was measured against the grown arena asks for nothing more.
  const grown = interpretSurfaceMeshState(words({ usable: 1, building: 1, mode: SVO_SURFACE_MESH_MODE.incremental,
    errorFlags: 1, overflowLength: overflowQuads }), { ...context, arenaBytes: [2 * SVO_SURFACE_MESH_BYTES, 64] });
  assert.equal(grown.grow, undefined);
  // At the budget the mesh is blocked rather than growing.
  const blocked = interpretSurfaceMeshState(words({ usable: 1, building: 1, mode: SVO_SURFACE_MESH_MODE.incremental,
    errorFlags: 1, overflowLength: 4 * overflowQuads }), { arenaBytes: [4 * SVO_SURFACE_MESH_BYTES, 64], maximumBytes: 4 * SVO_SURFACE_MESH_BYTES });
  assert.equal(blocked.status.state, "blocked");
  assert.equal(blocked.status.fallbackReason, "budget");
  assert.equal(blocked.grow, undefined);
});

test("withheld, invalid and faulted receipts fall back with their reasons", () => {
  assert.equal(interpretSurfaceMeshState(words({ usable: 1, withheld: 2 }), context).status.fallbackReason, "smooth");
  assert.equal(interpretSurfaceMeshState(words({ usable: 1, withheld: 1 }), context).status.fallbackReason, "inside-solid");
  assert.equal(interpretSurfaceMeshState(words({ usable: 1, errorFlags: 2 }), context).status.fallbackReason, "extraction");
  const first = interpretSurfaceMeshState(words({ building: 1, mode: SVO_SURFACE_MESH_MODE.initial, restartReason: 1 }), context);
  assert.equal(first.status.state, "pending");
  assert.equal(first.status.drawn, false);
  assert.equal(first.status.restartReason, "initial");
  assert.equal(interpretSurfaceMeshState(words({}), context).status.fallbackReason, "publication");
});

test("the drawn arena follows the front word", () => {
  const flipped = interpretSurfaceMeshState(words({ usable: 1, front: 1, frontCursor: 10 }), { ...context, arenaBytes: [64, 3 * SVO_SURFACE_MESH_BYTES] });
  assert.equal(flipped.front, 1);
  assert.equal(flipped.status.allocatedBytes, 3 * SVO_SURFACE_MESH_BYTES);
});
