import assert from "node:assert/strict";
import test from "node:test";
import { UNIFORM_LATTICE_MAX_PAGES, uniformLatticeAddress, uniformPageWord } from "../lib/methods/uniform/uniform-lattice-address";
import { UNIFORM_PAGE_POOL_SOLID } from "../lib/methods/uniform/uniform-page-pool";
import { UNIFORM_DIRECTORY_CHILD_WORDS, UNIFORM_DIRECTORY_TOP_WORDS, UNIFORM_DIRECTORY_WORDS, uniformDirectoryWord } from "../lib/methods/uniform/uniform-window-address";

// A box far past the old 64-page table (2048 h): 128 pages along x, a few
// pages stored, each found by the two-level read; the flat window holds the
// directory's word of lattice pages -1..62.
test("a sparse box of 4096 cells an axis resolves every page through the super-block directory", () => {
  const [px, py, pz] = [128, 2, 2], keep = new Set([0, 3, 4, 63, 64, 65, 127].map(x => `${x},1,0`));
  const air: [number, number, number][] = [];
  for (let z = 0; z < pz; z++) for (let y = 0; y < py; y++) for (let x = 0; x < px; x++) if (!keep.has(`${x},${y},${z}`)) air.push([x, y, z]);
  const a = uniformLatticeAddress([32 * px, 32 * py, 32 * pz], false, { blocks: 20, air });
  const blocks = new Set<number>();
  for (let z = 0; z < pz; z++) for (let y = 0; y < py; y++) for (let x = 0; x < px; x++) {
    const w = uniformPageWord(a, x, y, z);
    if (keep.has(`${x},${y},${z}`)) { assert.ok(w >= 1 && w < 19, `page ${x},${y},${z} is stored: ${w}`); assert.ok(!blocks.has(w)); blocks.add(w); }
    else assert.equal(w, 0, `page ${x},${y},${z} is AIR`);
  }
  assert.equal(blocks.size, keep.size);
  for (const [x, y, z] of [[-1, 0, 0], [px, 1, 0], [5, py, 1], [5, 0, -1]] as const) assert.equal(uniformPageWord(a, x, y, z), UNIFORM_PAGE_POOL_SOLID, `page ${x},${y},${z} past the box`);
  const tail = a.seed.tail, flat = UNIFORM_DIRECTORY_TOP_WORDS + UNIFORM_DIRECTORY_CHILD_WORDS;
  for (let z = 0; z < 64; z++) for (let y = 0; y < 64; y++) for (let x = 0; x < 64; x++)
    if (tail[flat + (x | (y << 6) | (z << 12))] !== uniformDirectoryWord(tail, x - 1, y - 1, z - 1)) assert.fail(`flat word of page ${x - 1},${y - 1},${z - 1}`);
  assert.ok(tail.length > UNIFORM_DIRECTORY_WORDS);
});

test("the box limit is the directory window's, past the old 62 pages", () => {
  assert.equal(UNIFORM_LATTICE_MAX_PAGES, 252);
  assert.throws(() => uniformLatticeAddress([32 * 253, 32, 32], false, { blocks: 20, air: [] }), /exceeds 252 pages/);
});
