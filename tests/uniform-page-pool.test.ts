import assert from "node:assert/strict";
import test from "node:test";
import { uniformPagePoolSeed, uniformPagePoolBoxPages, uniformPagePoolBoxSeed, uniformPagePoolCellBoxPages, uniformPagePoolEntry, uniformPagePoolLayout, uniformPagePoolPublished,
  UNIFORM_PAGE_POOL_SOLID as SOLID } from "../lib/methods/uniform/uniform-page-pool";

const letters = uniformPagePoolLayout({ blocks: 208, idleSteps: 8, box: { origin: [0, 0, 0], pages: [8, 6, 4] } });

test("the identity box seed is the scan-order block of every box page, SOLID outside", () => {
  const seed = uniformPagePoolBoxSeed(letters);
  for (let z = 0; z < 4; z++) for (let y = 0; y < 6; y++) for (let x = 0; x < 8; x++) assert.equal(seed.table[uniformPagePoolEntry(letters, x, y, z)], 1 + x + 8 * (y + 6 * z));
  assert.deepEqual([seed.resident.length, seed.free, seed.table.reduce((n, w) => n + +(w === SOLID), 0)], [192, 14, 64 ** 3 - 192]);
  assert.deepEqual([...seed.ring.subarray(0, 14)], Array.from({ length: 14 }, (_, i) => 193 + i));
});

test("a permuted box seed holds the same pages in shuffled blocks with free holes, fixed by its number", () => {
  const identity = uniformPagePoolBoxPages(letters), a = uniformPagePoolBoxPages(letters, { permute: 7 }), b = uniformPagePoolBoxPages(letters, { permute: 7 }), c = uniformPagePoolBoxPages(letters, { permute: 8 });
  assert.deepEqual(a, b);
  assert.notDeepEqual(a.map(p => p.block), c.map(p => p.block));
  assert.deepEqual(a.map(({ x, y, z }) => [x, y, z]), identity.map(({ x, y, z }) => [x, y, z]));
  const blocks = a.map(p => p.block), seed = uniformPagePoolBoxSeed(letters, { permute: 7 });
  assert.equal(new Set(blocks).size, 192);
  assert.ok(blocks.every(block => block >= 1 && block <= 206) && blocks.some((block, i) => i > 0 && block < blocks[i - 1]!));
  const holes = [...seed.ring.subarray(0, seed.free)];
  assert.ok(holes.length === 14 && holes.some(block => block < 192) && holes.every(block => !blocks.includes(block)));
  // The published record does not depend on the blocks.
  assert.deepEqual(uniformPagePoolPublished(seed), uniformPagePoolPublished(uniformPagePoolBoxSeed(letters)));
});

test("a partial box seed off the window origin: AIR inside the box, the chosen kind outside, and its published record", () => {
  const layout = uniformPagePoolLayout({ table: 8, blocks: 10, idleSteps: 2, box: { origin: [-1, 6, 3], pages: [3, 4, 2] } });
  const seed = uniformPagePoolBoxSeed(layout, { resident: (x, y) => x === y });
  assert.deepEqual(uniformPagePoolBoxPages(layout, { resident: (x, y) => x === y }), [0, 1, 2, 0, 1, 2].map((i, n) => ({ x: -1 + i, y: 6 + i, z: 3 + (n > 2 ? 1 : 0), block: n + 1 })));
  // Window coordinates wrap: box x -1..1 is 7, 0, 1 and y 6..9 is 6, 7, 0, 1.
  assert.deepEqual([[7, 6, 3], [0, 7, 3], [1, 0, 4], [1, 6, 3], [2, 6, 3], [7, 2, 3], [7, 6, 5]].map(([x, y, z]) => seed.table[uniformPagePoolEntry(layout, x!, y!, z!)]), [1, 2, 6, 0, SOLID, SOLID, SOLID]);
  assert.deepEqual([...uniformPagePoolPublished(seed)], [32, 3, 4, 2, 6, 0xffffffff, 6, 3, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
  assert.throws(() => uniformPagePoolBoxSeed(uniformPagePoolLayout({ table: 8, blocks: 5, idleSteps: 1, box: { origin: [0, 0, 0], pages: [2, 2, 1] } })), /more pages than the pool's 3 blocks/);
  assert.throws(() => uniformPagePoolBoxPages(uniformPagePoolLayout({ blocks: 5, idleSteps: 1 })), /no box/);
  assert.throws(() => uniformPagePoolLayout({ table: 8, blocks: 5, idleSteps: 1, box: { origin: [0, 0, 0], pages: [9, 1, 1] } }), /1\.\.8 pages an axis/);
  assert.equal(uniformPagePoolPublished(seed, uniformPagePoolLayout({ table: 8, blocks: 10, idleSteps: 2 })).length, 0);
});

test("a padded cell box pins the pages it overlaps, clipped to the layout's box", () => {
  const pages = (low: readonly [number, number, number], high: readonly [number, number, number], pad?: number) => uniformPagePoolCellBoxPages(letters, low, high, pad).map(({ x, y, z }) => [x, y, z]);
  assert.deepEqual(pages([32, 40, 0], [64, 41, 32]), [[1, 1, 0]]);
  assert.deepEqual(pages([32, 40, 0], [65, 41, 32]), [[1, 1, 0], [2, 1, 0]]);
  assert.deepEqual(pages([32, 40, 0], [64, 41, 32], 8), [[0, 1, 0], [1, 1, 0], [2, 1, 0], [0, 1, 1], [1, 1, 1], [2, 1, 1]]);
  assert.deepEqual(pages([250, 190, 120], [300, 300, 300], 4), [[7, 5, 3]]);
  assert.deepEqual(uniformPagePoolCellBoxPages(uniformPagePoolLayout({ blocks: 5, idleSteps: 1 }), [-1, 0, 0], [1, 1, 1]).map(p => p.x), [-1, 0]);
});

test("published pages distinguish solid from air independently of physical block order", () => {
  const layout = uniformPagePoolLayout({ table: 8, blocks: 6, idleSteps: 2, box: { origin: [-2, 0, 0], pages: [3, 1, 1] } });
  const seed = uniformPagePoolSeed(layout, [{ x: -1, y: 0, z: 0, block: 3 }], x => x === 6 ? 2 : 1);
  const published = uniformPagePoolPublished(seed);
  assert.deepEqual([...published.slice(8)], [2, 1, 0]);
  assert.deepEqual([...new Int32Array(published.buffer).slice(5, 8)], [-2, 0, 0]);
});
