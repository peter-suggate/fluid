import assert from "node:assert/strict";
import test from "node:test";

import {
  createOctreeSignedSurfaceIndex,
  decodeOctreeSignedSurfaceBrickKey,
  encodeOctreeSignedSurfaceBrickKey,
  locateOctreeSignedSurfaceCell,
  planOctreeSignedSurfaceBricks,
  selectAnalyticOctreeSignedSurfaceBricks,
} from "../lib/octree-signed-surface-bricks";

test("two-sided planar capacity scales with interface area instead of box volume", () => {
  const ocean = planOctreeSignedSurfaceBricks([320, 96, 80], 320 * 80);
  assert.equal(ocean.bandThicknessCells, 9);
  assert.equal(ocean.unpaddedBrickCount, 4_800);
  assert.equal(ocean.brickCapacity, 6_000);
  assert.equal(ocean.scalarBytes, 4_608_000);
  assert.equal(ocean.allocatedBytes, 4_728_256);
  assert.equal(ocean.denseEquivalentBytes, 29_491_200);
  assert.equal(ocean.savedBytes, 24_762_944);

  const target = planOctreeSignedSurfaceBricks([640, 192, 160], 640 * 160);
  assert.equal(target.unpaddedBrickCount, 19_200);
  assert.equal(target.brickCapacity, 24_000);
  assert.equal(target.allocatedBytes, 18_912_256);
  assert.equal(target.denseEquivalentBytes, 235_929_600);
  assert.equal(target.savedBytes, 217_017_344);
  assert.equal(target.allocatedBytes, ocean.allocatedBytes * 4 - 768,
    "doubling every dimension grows a planar sparse band by area, while dense phi grows by volume");
  assert.equal(target.denseEquivalentBytes, ocean.denseEquivalentBytes * 8);
});

test("brick keys round-trip at coordinate limits and reserve zero", () => {
  for (const coord of [[0, 0, 0], [79, 23, 19], [1023, 1023, 1023]] as const) {
    assert.deepEqual(decodeOctreeSignedSurfaceBrickKey(encodeOctreeSignedSurfaceBrickKey(coord)), [...coord]);
  }
  assert.equal(encodeOctreeSignedSurfaceBrickKey([0, 0, 0]), 1);
  assert.throws(() => decodeOctreeSignedSurfaceBrickKey(0), /invalid/);
  assert.throws(() => encodeOctreeSignedSurfaceBrickKey([1024, 0, 0]), /ten unsigned bits/);
});

test("analytic plane selects both sides and clips cleanly at domain edges", () => {
  const centred = selectAnalyticOctreeSignedSurfaceBricks([16, 16, 16], (point) => point[1] - 8.5, { bandCells: 2 });
  assert.equal(centred.length, 32);
  assert.ok(centred.some((coord) => coord[1] === 1), "negative side must be resident");
  assert.ok(centred.some((coord) => coord[1] === 2), "positive side must be resident");

  const edge = selectAnalyticOctreeSignedSurfaceBricks([10, 9, 7], (point) => point[0] - 0.5, { bandCells: 4 });
  assert.ok(edge.length > 0);
  assert.ok(edge.every((coord) => coord[0] >= 0 && coord[0] < 3 && coord[1] < 3 && coord[2] < 2));
  const index = createOctreeSignedSurfaceIndex([10, 9, 7], edge, []);
  assert.deepEqual(locateOctreeSignedSurfaceCell(index, [-1, 0, 0]), { kind: "outside", sign: 1 });
  assert.deepEqual(locateOctreeSignedSurfaceCell(index, [10, 0, 0]), { kind: "outside", sign: 1 });
});

test("analytic sphere activates interior and exterior samples without filling its volume", () => {
  const centre = [16, 16, 16] as const, radius = 10;
  const sdf = (point: readonly [number, number, number]) => Math.hypot(
    point[0] - centre[0], point[1] - centre[1], point[2] - centre[2],
  ) - radius;
  const bricks = selectAnalyticOctreeSignedSurfaceBricks([32, 32, 32], sdf, { bandCells: 2 });
  const index = createOctreeSignedSurfaceIndex([32, 32, 32], bricks, [
    { origin: [16, 16, 16], size: 4, sign: -1 },
  ]);
  assert.equal(locateOctreeSignedSurfaceCell(index, [16, 16, 16]).kind, "inactive",
    "deep sphere interior should use a compact negative tile/background rather than an active brick");
  assert.equal(locateOctreeSignedSurfaceCell(index, [0, 0, 0]).kind, "inactive");
  assert.ok(bricks.length < 8 ** 3, "a spherical shell must not allocate the complete brick volume");
  const innerBand = locateOctreeSignedSurfaceCell(index, [7, 16, 16]);
  const outerBand = locateOctreeSignedSurfaceCell(index, [5, 16, 16]);
  assert.equal(innerBand.kind, "active");
  assert.equal(outerBand.kind, "active");
  assert.ok(sdf([7.5, 16.5, 16.5]) < 0 && sdf([5.5, 16.5, 16.5]) > 0);
});

test("analytic bubble preserves positive gas inside negative liquid", () => {
  const bubblePhi = (point: readonly [number, number, number]) => 10 - Math.hypot(
    point[0] - 16, point[1] - 16, point[2] - 16,
  );
  const bricks = selectAnalyticOctreeSignedSurfaceBricks([32, 32, 32], bubblePhi, { bandCells: 2 });
  const bubble = createOctreeSignedSurfaceIndex([32, 32, 32], bricks, [
    { origin: [16, 16, 16], size: 4, sign: 1 },
  ], -1);
  assert.deepEqual(locateOctreeSignedSurfaceCell(bubble, [0, 0, 0]), { kind: "inactive", sign: -1 });
  const gas = locateOctreeSignedSurfaceCell(bubble, [16, 16, 16]);
  assert.equal(gas.kind, "inactive");
  if (gas.kind === "inactive") assert.equal(gas.sign, 1);
  assert.equal(locateOctreeSignedSurfaceCell(bubble, [7, 16, 16]).kind, "active");
  assert.equal(locateOctreeSignedSurfaceCell(bubble, [5, 16, 16]).kind, "active");
  assert.ok(bubblePhi([7.5, 16.5, 16.5]) > 0 && bubblePhi([5.5, 16.5, 16.5]) < 0,
    "the active bubble shell must contain both gas-side and liquid-side samples");
});

test("index and nested sign fallback are deterministic under input permutation", () => {
  const bricks = [[3, 2, 1], [0, 0, 0], [3, 2, 1]] as const;
  const tiles = [
    { origin: [0, 0, 0] as const, size: 8, sign: -1 as const },
    { origin: [4, 4, 4] as const, size: 4, sign: 1 as const },
  ];
  const a = createOctreeSignedSurfaceIndex([16, 16, 16], bricks, tiles);
  const b = createOctreeSignedSurfaceIndex([16, 16, 16], [...bricks].reverse(), [...tiles].reverse());
  assert.deepEqual([...a.brickKeys], [...b.brickKeys]);
  assert.deepEqual(a.inactiveTiles, b.inactiveTiles);
  const nested = locateOctreeSignedSurfaceCell(a, [5, 5, 5]);
  assert.equal(nested.kind, "inactive");
  if (nested.kind === "inactive") assert.equal(nested.sign, 1, "the smallest containing tile must win");
  assert.throws(() => createOctreeSignedSurfaceIndex([16, 16, 16], [], [tiles[0], tiles[0]]), /Duplicate/);
});
