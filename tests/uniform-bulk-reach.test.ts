import { test } from "node:test";
import assert from "node:assert/strict";
import { bulkReachCapacity, bulkReachKey, bulkReachLookup, bulkReachTable, bulkReached, uniformBulkRegionReach } from "../lib/methods/uniform/uniform-bulk-reach";
import { uniformBulkRegionClass, uniformBulkRegionDeep, type UniformBulkTile } from "../lib/methods/uniform/uniform-bulk-tier-reference";

type V = [number, number, number];
// A 24^3-tile box: a wall of closed tiles at x 8..11 (region 2's own tiles),
// deep water at x < 8, dry AIR at x >= 12; `hole` opens the wall's tiles at y, z 0..3.
const N = 24;
function scene(hole = false) {
  const wall = (t: readonly number[]) => t[0] >= 8 && t[0] < 12 && !(hole && t[1] < 4 && t[2] < 4);
  const closed = (t: readonly number[]) => t.some(v => v < 0 || v >= N) || wall(t);
  const tile = (t: V): UniformBulkTile | undefined => {
    if (t.some(v => v < 0 || v >= N)) return undefined;
    if (wall(t)) return { kind: "resident", open: 0, phi: -100 };
    return t[0] < 8 ? { kind: "resident", open: 1, phi: -100 } : { kind: "air" };
  };
  return { closed, tile };
}

test("Reach: water by a one-region wall stays eligible past the AIR behind it", () => {
  // Region 1 (x tiles 4..7) against a wall two tiles thick (8..9): AIR at
  // 10..11 lies inside its neighbourhood, behind the wall.
  const wall = (t: readonly number[]) => t[0] >= 8 && t[0] < 10;
  const closed = (t: readonly number[]) => t.some(v => v < 0 || v >= N) || wall(t);
  const tile = (t: V): UniformBulkTile | undefined => t.some(v => v < 0 || v >= N) ? undefined
    : wall(t) ? { kind: "resident", open: 0, phi: -100 } : t[0] < 8 ? { kind: "resident", open: 1, phi: -100 } : { kind: "air" };
  const r: V = [1, 2, 2], mask = uniformBulkRegionReach(closed, r);
  assert.ok(mask);
  assert.equal(bulkReached(mask, [0, 0, 0]), true, "own tiles");
  assert.equal(bulkReached(undefined, [7, 7, 7]), true, "no mask: all of it");
  assert.equal(bulkReached(mask, [6, 0, 0]), false, "air behind the wall");
  assert.equal(bulkReached(mask, [3, 0, 0]), true, "water beside it");
  assert.equal(uniformBulkRegionClass(tile, r, 1), undefined, "the box rule fails it for the hidden AIR");
  assert.equal(uniformBulkRegionClass(tile, r, 1, mask), "open", "reach admits it");
  assert.equal(uniformBulkRegionDeep(tile, r, 1), false);
  assert.equal(uniformBulkRegionDeep(tile, r, 1, mask), true);
});

test("Reach: a hole in the wall within the neighbourhood lets the air count", () => {
  const wall = (t: readonly number[]) => t[0] >= 8 && t[0] < 10 && !(t[1] === 9 && t[2] === 9);
  const closed = (t: readonly number[]) => t.some(v => v < 0 || v >= N) || wall(t);
  const tile = (t: V): UniformBulkTile | undefined => t.some(v => v < 0 || v >= N) ? undefined
    : wall(t) ? { kind: "resident", open: 0, phi: -100 } : t[0] < 8 ? { kind: "resident", open: 1, phi: -100 } : { kind: "air" };
  const r: V = [1, 2, 2], mask = uniformBulkRegionReach(closed, r);
  assert.equal(mask, undefined, "every open tile is reached through the hole");
  assert.equal(uniformBulkRegionClass(tile, r, 1, mask), undefined);
});

test("Reach: a closed region holds no surface, so AIR beside it does not fail it", () => {
  const { closed, tile } = scene();
  const wall: V = [2, 2, 2], mask = uniformBulkRegionReach(closed, wall);
  assert.equal(mask, undefined, "a closed region reaches both sides of itself");
  assert.equal(uniformBulkRegionClass(tile, wall, 1, mask), "closed");
  assert.equal(uniformBulkRegionDeep(tile, wall, 1, mask), true);
  // Resident air the depth from the surface is AIR too: the pool's collar behind a dam.
  const collar = (t: V): UniformBulkTile | undefined => t[0] >= 12 && t[0] < N ? { kind: "resident", open: 1, phi: 100 } : tile(t);
  assert.equal(uniformBulkRegionClass(collar, wall, 1, mask), "closed");
  assert.equal(uniformBulkRegionClass(collar, [1, 2, 2], 1), "open", "the water's neighbourhood ends at the wall");
  assert.equal(uniformBulkRegionClass(collar, [3, 2, 2], 1), undefined, "air in an open region's reach fails it");
  // A shallow tile beside it still fails it: water reaches its readers that way.
  const shallow = (t: V) => t[0] === 7 && t[1] === 9 && t[2] === 9 ? { kind: "resident" as const, open: 1, phi: 0 } : tile(t);
  assert.equal(uniformBulkRegionClass(shallow, wall, 1, mask), undefined);
});

test("Reach: the GPU table finds every mask by its key and nothing else", () => {
  const masks = new Map<number, Uint32Array>();
  for (let i = 0; i < 37; i++) { const m = new Uint32Array(54); m[i % 54] = i + 1; masks.set(bulkReachKey([i, 2 * i % 7, 3]), m); }
  const capacity = bulkReachCapacity(masks.size), table = bulkReachTable(masks, capacity);
  assert.ok(capacity > masks.size && (capacity & (capacity - 1)) === 0);
  for (const [key, m] of masks) assert.deepEqual([...bulkReachLookup(table, capacity, key)!], [...m]);
  assert.equal(bulkReachLookup(table, capacity, bulkReachKey([500, 1, 1])), undefined);
  assert.throws(() => bulkReachTable(masks, 32), /key slots/);
});
