import assert from "node:assert/strict";
import test from "node:test";
import { uniformBulkStartPages, type UniformBulkStartSource } from "../lib/methods/uniform/uniform-bulk-start";
import { UNIFORM_BULK_REACH_CELLS } from "../lib/methods/uniform/uniform-bulk-tier-reference";
import { bulk16PageWord } from "../lib/methods/uniform/uniform-bulk-pages-reference";
import { UNIFORM_KIND, uniformLatticeAddress, uniformPageKind, uniformPageWord } from "../lib/methods/uniform/uniform-lattice-address";

/* V2c (docs/plans/bulk16-plan-2026-10-28.md): the census start's BULK16
 * pages are bulkSelect's verdict on the t=0 state, evaluated on the host. A
 * 3^3-page closed box, h = 1, dt = 1. */
const h = 1, dt = 1;
const box = (o: Partial<UniformBulkStartSource> = {}): UniformBulkStartSource => {
  const all: [number, number, number][] = [];
  for (let z = 0; z < 3; z++) for (let y = 0; y < 3; y++) for (let x = 0; x < 3; x++) all.push([x, y, z]);
  return { pages: [3, 3, 3], openTop: false, stored: () => true, candidates: all, fine: () => false, open: () => 1, phi: () => -100, face: () => 0, h, dt, ...o };
};
const names = (pages: [number, number, number][]) => pages.map(p => p.join(","));

test("BULK16 start: deep still water starts in BULK16 in every page, walls being no surface", () => {
  assert.equal(uniformBulkStartPages(box()).length, 27);
});

test("BULK16 start: no page within 5.5 h of the surface, or beside air or h, or cut", () => {
  // The surface at cell 80: a tile's centre (4t + 2) is 5.5 h deep up to tile 18,
  // and a region's neighbourhood reaches 4 tiles above its own (up to 4r + 7).
  const surface = uniformBulkStartPages(box({ phi: (_x, y) => 4 * y + 2 - 80 }));
  assert.deepEqual(new Set(surface.map(p => p[1])), new Set([0]));
  assert.equal(surface.length, 9);
  // A page not stored is AIR: a region with it in its 12^3 neighbourhood fails.
  const air = uniformBulkStartPages(box({ stored: (x, y, z) => !(x === 2 && y === 2 && z === 2) }));
  assert.ok(!names(air).includes("1,1,1") && !names(air).includes("2,2,1") && names(air).includes("0,0,0"));
  // An h tile: its own region is HOT, and the regions that see that region's tiles fail.
  const fine = uniformBulkStartPages(box({ fine: (x, y, z) => x === 12 && y === 12 && z === 12 }));
  assert.ok(!names(fine).includes("1,1,1") && names(fine).includes("0,0,0"));
  // A cut tile fails every region that sees it; a closed tile is a wall.
  const cut = uniformBulkStartPages(box({ open: (x, y, z) => x === 12 && y === 12 && z === 12 ? 0.5 : 1 }));
  assert.ok(!names(cut).includes("1,1,1") && names(cut).includes("0,0,0"));
  const closed = uniformBulkStartPages(box({ open: (x, y, z) => x === 17 && y === 12 && z === 12 ? 0 : 1 }));
  assert.ok(names(closed).includes("1,1,1"));
});

test("BULK16 start: uniform motion inside the reach starts in BULK16 away from the walls it meets", () => {
  // A wall's face is 0 (bFace below, the box's last face above): the regions
  // against the x walls see a 4h-scale step.
  const moving = uniformBulkStartPages(box({ face: (x, _y, _z, a) => a === 0 && x < 23 ? 0.5 : 0 }));
  assert.deepEqual(new Set(moving.map(p => p[0])), new Set([1]));
  assert.equal(moving.length, 9);
  // Faster than UNIFORM_BULK_REACH_CELLS h a step: none.
  assert.equal(uniformBulkStartPages(box({ face: (_x, _y, _z, a) => a === 1 ? 1.01 * UNIFORM_BULK_REACH_CELLS : 0 })).length, 0);
});

test("BULK16 start: the seed holds a start page in its slot, no block, counted in its super-block", () => {
  const plain = uniformLatticeAddress([96, 96, 96], false), a = uniformLatticeAddress([96, 96, 96], false, { blocks: 27 + 2 - 2, bulk: [[1, 1, 1], [2, 0, 1]] });
  assert.equal(uniformPageWord(a, 1, 1, 1), bulk16PageWord(1));
  assert.equal(uniformPageWord(a, 2, 0, 1), bulk16PageWord(2));
  assert.equal(uniformPageKind(a, 1, 1, 1), UNIFORM_KIND.bulk);
  // The other 25 pages take blocks 1..25 in scan order; none is free.
  assert.equal(a.seed.resident.length, 25);
  assert.equal(a.seed.free, 0);
  assert.equal(uniformPageWord(a, 0, 0, 0), 1);
  assert.equal(uniformPageWord(a, 0, 1, 1), 1 + 11);
  assert.deepEqual([...a.seed.superCount], [...plain.seed.superCount]);
  assert.throws(() => uniformLatticeAddress([96, 96, 96], false, { bulk: [[1, 1, 1], [1, 1, 1]] }), /listed twice/);
});
