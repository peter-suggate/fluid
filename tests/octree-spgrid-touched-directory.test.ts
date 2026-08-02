import assert from "node:assert/strict";
import test from "node:test";
import { buildSPGridBrickRankDirectory, buildSPGridPhysicalPageAdjacency } from
  "../lib/webgpu-octree-spgrid-vcycle";
import { buildSPGridTouchedDirectory, stableRadixSortU32 } from
  "../lib/octree-spgrid-touched-directory";

test("four-pass u32 radix is stable and exactly ascending", () => {
  const values = [0xffff_ffff, 256, 1, 0x10001, 0, 256, 255, 0x8000_0000];
  assert.deepEqual(stableRadixSortU32(values), [...values].sort((a, b) => a - b));
});

test("touched brick masks and ranks exactly match the dense directory oracle", () => {
  const dimensions = [19, 11, 7] as const;
  const cells = [
    { coordinate: [18, 10, 6] as const, slot: 91 },
    { coordinate: [0, 0, 0] as const, slot: 7 },
    { coordinate: [8, 6, 5] as const, slot: 44 },
    { coordinate: [3, 3, 3] as const, slot: 8 },
    { coordinate: [9, 6, 5] as const, slot: 45 },
    { coordinate: [17, 0, 0] as const, slot: 72 },
  ];
  const touched = buildSPGridTouchedDirectory(dimensions, cells);
  const dense = buildSPGridBrickRankDirectory(dimensions, cells);
  const denseOccupied = dense.masks.flatMap((masks, index) =>
    masks[0] !== 0 || masks[1] !== 0 ? [index] : []);
  assert.deepEqual(touched.bricks.map((brick) => brick.dense), denseOccupied);
  assert.deepEqual(touched.bricks.map((brick) => brick.masks),
    denseOccupied.map((index) => dense.masks[index]));
  assert.deepEqual(touched.bricks.map((brick) => brick.rankedBase),
    denseOccupied.map((index) => dense.bases[index]));
  assert.deepEqual(touched.rankedSlots, dense.slots);
  assert.ok(touched.touchedBricks.length < dense.masks.length,
    "the producer enumerates only first-touched identities, not the dense brick arena");
});

test("pages derived from touched bricks equal stable dense physical page identities", () => {
  const dimensions = [24, 16, 8] as const;
  const cells = [
    { coordinate: [17, 0, 0] as const, slot: 2 },
    { coordinate: [9, 9, 5] as const, slot: 5 },
    { coordinate: [0, 0, 0] as const, slot: 1 },
    { coordinate: [9, 0, 0] as const, slot: 3 },
    { coordinate: [10, 10, 1] as const, slot: 4 },
  ];
  const touched = buildSPGridTouchedDirectory(dimensions, cells);
  const pages = buildSPGridPhysicalPageAdjacency(dimensions,
    cells.map((cell) => cell.coordinate));
  const pageDims = [3, 2, 2] as const;
  const densePages = pages.origins.map(([x, y, z]) =>
    x / 8 + pageDims[0] * (y / 8 + pageDims[1] * (z / 4)));
  assert.deepEqual(touched.pages, densePages);
});

test("a rejected attempt can be rebuilt from its published identities alone", () => {
  const first = buildSPGridTouchedDirectory([16, 16, 8], [
    { coordinate: [15, 15, 7], slot: 3 }, { coordinate: [0, 0, 0], slot: 1 },
  ]);
  const retry = buildSPGridTouchedDirectory([16, 16, 8], [
    { coordinate: [4, 4, 4], slot: 2 },
  ]);
  assert.deepEqual(first.touchedBricks, [31, 0]);
  assert.deepEqual(retry.touchedBricks, [21]);
  assert.equal(retry.bricks.some((brick) => first.touchedBricks.includes(brick.dense)), false,
    "retry state contains no retired brick identity from the rejected attempt");
});
