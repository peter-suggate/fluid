import assert from "node:assert/strict";
import test from "node:test";
import { uniformBoxTileIndex, uniformLatticeAddress, uniformTileKey, uniformTileLive, uniformTileStored, type UniformLatticeAddress } from "../lib/methods/uniform/uniform-lattice-address";
import { createUniformMixedLayoutFromWidths, uniformMixedAllCoarseLayout } from "../lib/methods/uniform/uniform-mixed-layout";
import { uniformMixedPressureStorage } from "../lib/methods/uniform/uniform-mixed-pressure-boundary.wgsl";
import { planUniformMixedPressureMemory, uniformMixedPressureRootBytes } from "../lib/methods/uniform/uniform-mixed-pressure-memory";
import { UNIFORM_MIXED_ROOT_FATAL, uniformMixedRootOwnerWords, uniformMixedRootOwnersWGSL, uniformMixedRootSlot, uniformMixedRootUnstoredRows } from "../lib/methods/uniform/uniform-mixed-root-owners";
import { uniformMixedPageCount } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import { UNIFORM_PAGE_POOL_FREE as FREE, UNIFORM_PAGE_POOL_SOLID as SOLID } from "../lib/methods/uniform/uniform-page-pool";

// The host packer's all-4h layout of an address: what the static root held
// for the seed. The GPU root (its host mirror here) must hold the same words
// for whatever the table holds now.
const lattice = (d: readonly number[]) => ({ dimensions: [d[0]!, d[1]!, d[2]!] as [number, number, number], cellSize_m: [1, 1, 1] as [number, number, number], origin_m: { x: 0, y: 0, z: 0 } });
const host = (a: UniformLatticeAddress) => createUniformMixedLayoutFromWidths(lattice(a.cells), new Uint8Array(a.boxTiles).fill(4), [], a);
function same(a: UniformLatticeAddress, words: Uint32Array, label: string): void {
  const n = a.tileKeys, layout = host(a);
  assert.deepEqual([...words.subarray(0, n)], [...layout.tiles], `${label}: tile words`);
  assert.deepEqual([...words.subarray(2 * n, 4 * n)], [...layout.stencils], `${label}: stencil words`);
  assert.deepEqual([...words.subarray(4 * n, 5 * n)], [...layout.blendMasks], `${label}: blend masks`);
}
/** A copy whose table and block words a test may change, as a commit does. */
function live(a: UniformLatticeAddress): UniformLatticeAddress {
  return { ...a, seed: { ...a.seed, table: new Uint32Array(a.seed.table), blocks: new Uint32Array(a.seed.blocks) } };
}
const entry = (x: number, y: number, z: number) => x | (y << 6) | (z << 12);
const blockOf = (a: UniformLatticeAddress, x: number, y: number, z: number) => a.seed.table[entry(x, y, z)]!;
const valid = (a: UniformLatticeAddress, words: Uint32Array, tx: number, ty: number, tz: number, dx: number, dy: number, dz: number) => {
  // Bit 1+k of the recipe: the coarse incident tile T+c-1; the -x face neighbour is c=(0,1,1), k=6, and +x has no incident bit, so ask the neighbour instead.
  const k = uniformTileKey(a, tx, ty, tz), recipe = words[2 * a.tileKeys + 2 * k + 1]!;
  const c = [dx + 1, dy + 1, dz + 1];
  assert.ok(c.every(v => v === 0 || v === 1), "incident offsets are -1 or 0");
  return (recipe >>> (1 + c[0]! + 2 * c[1]! + 4 * c[2]!)) & 1;
};

test("every page resident: the GPU root's words are the host root's, key for key", () => {
  for (const [cells, openTop] of [[[64, 32, 32], false], [[64, 64, 32], true], [[40, 36, 44], false], [[96, 32, 64], true]] as const) {
    const a = uniformLatticeAddress(cells, openTop), n = a.tileKeys, { words, fatal } = uniformMixedRootOwnerWords(a);
    assert.equal(fatal, 0);same(a, words, cells.join("x"));
    // Every box tile has exactly one key, and its word is its box index.
    const seen = new Uint8Array(a.boxTiles);let live = 0;
    for (let k = 0; k < n; k++) if (uniformTileLive(a, k)) { live++;seen[words[k]!]!++; } else assert.equal(words[k], 0);
    assert.equal(live, a.boxTiles);assert.ok(seen.every(v => v === 1));
    // The counts the static root held (the host list's length) are the box's rows, which the root's shape now states.
    assert.equal(host(a).coarseTiles.length, a.boxTiles);
    assert.equal(uniformMixedAllCoarseLayout(host(a)).boxOwners, a.boxTiles);
    // Nothing is unstored: no row is written and no flag cleared.
    const pages = Math.ceil(a.tiles[0] / 4) * Math.ceil(a.tiles[1] / 4) * Math.ceil(a.tiles[2] / 4), flags = new Uint32Array(pages).fill(1), phi = new Float32Array(a.boxTiles).fill(-3);
    assert.equal(uniformMixedRootUnstoredRows(a, flags, phi, undefined, 16), 0);
    assert.ok(flags.every(v => v === 1) && phi.every(v => v === -3));
  }
});

test("the root's shape names no tile", () => {
  const a = uniformLatticeAddress([64, 32, 32]), shape = uniformMixedAllCoarseLayout(host(a));
  assert.equal(shape.address, a);
  assert.deepEqual([shape.cellCount, shape.boxOwners, shape.metadataBytes, shape.fineTiles.length, shape.coarseTiles.length], [a.boxTiles, a.boxTiles, 20 * a.tileKeys, 0, 0]);
  assert.ok(shape.tiles.every(v => v === 0) && shape.stencils.every(v => v === 0) && shape.blendMasks.every(v => v === 0));
});

test("a permuted seed with free blocks and AIR holes: host words for the same table, zero at every key with no tile", () => {
  const a = uniformLatticeAddress([96, 64, 64], false, { blocks: 20, permute: 7, air: [[1, 0, 0], [2, 1, 1]] }), n = a.tileKeys;
  const { words, fatal } = uniformMixedRootOwnerWords(a);
  assert.equal(fatal, 0);same(a, words, "permuted");
  let dead = 0;
  for (let b = 0; b < a.blocks; b++) {
    if (b !== 0 && b !== a.blocks - 1 && a.seed.blocks[b] !== FREE) continue;
    for (let k = 512 * b; k < 512 * b + 512; k++) { dead++;assert.deepEqual([words[k], words[2 * n + 2 * k], words[2 * n + 2 * k + 1], words[4 * n + k]], [0, 0, 0, 0]); }
  }
  // 12 box pages, 2 of them AIR: 10 resident blocks of 20.
  assert.equal(dead, 512 * 10);
  // The tile at x=7 beside the AIR page (1,0,0) has no +x neighbour; the tile at x=16 beyond it has no -x neighbour.
  assert.equal(uniformTileStored(a, 8, 3, 3), false);
  assert.equal(valid(a, words, 16, 3, 3, -1, 0, 0), 0);assert.equal(valid(a, words, 6, 3, 3, -1, 0, 0), 1);
  // No live key's owner is a box tile of an AIR page.
  const owners = new Set<number>();
  for (let k = 0; k < n; k++) if (uniformTileLive(a, k)) owners.add(words[k]!);
  assert.equal(owners.size, 10 * 512);
  assert.equal(owners.has(uniformBoxTileIndex(a, 8, 3, 3)), false);
});

test("a claim then a retire: the claimed block's owners, both sides' valid bits, nothing of the retired block", () => {
  const a = live(uniformLatticeAddress([96, 64, 64], false, { blocks: 20, permute: 3, air: [[1, 0, 0]] })), n = a.tileKeys;
  let words = uniformMixedRootOwnerWords(a).words;
  same(a, words, "seed");
  assert.equal(valid(a, words, 16, 3, 3, -1, 0, 0), 0);
  // Claim page (1,0,0) into a free block.
  const free = a.seed.blocks.findIndex((r, b) => b > 0 && b < a.blocks - 1 && r === FREE);
  assert.ok(free > 0);
  a.seed.table[entry(1, 0, 0)] = free;a.seed.blocks[free] = entry(1, 0, 0);
  let step = uniformMixedRootOwnerWords(a, words);
  assert.equal(step.fatal, 0);words = step.words;same(a, words, "claim");
  // The claimed block's 512 keys hold the page's box indices; the neighbours on both sides see it.
  for (let l = 0; l < 512; l++) assert.equal(words[512 * free + l], uniformBoxTileIndex(a, 8 + (l & 7), (l >> 3) & 7, l >> 6));
  assert.equal(valid(a, words, 16, 3, 3, -1, 0, 0), 1);assert.equal(valid(a, words, 8, 3, 3, -1, 0, 0), 1);
  // The detail ring's bit (another kernel's) survives on a live key, and not on a key that loses its tile.
  const gone = blockOf(a, 2, 1, 1), kept = uniformTileKey(a, 3, 3, 3);
  words[2 * n + 2 * kept + 1]! |= 1;words[2 * n + 2 * (512 * gone + 5) + 1]! |= 1;
  // Retire page (2,1,1).
  a.seed.table[entry(2, 1, 1)] = 0;a.seed.blocks[gone] = FREE;
  step = uniformMixedRootOwnerWords(a, words);
  assert.equal(step.fatal, 0);
  assert.equal(step.words[2 * n + 2 * kept + 1]! & 1, 1);
  step.words[2 * n + 2 * kept + 1]! &= ~1;words = step.words;same(a, words, "retire");
  for (let k = 512 * gone; k < 512 * gone + 512; k++) assert.deepEqual([words[k], words[2 * n + 2 * k], words[2 * n + 2 * k + 1], words[4 * n + k]], [0, 0, 0, 0]);
  // Both sides of the retired page: its -x neighbour (15,y,z) lost +x (asked of nothing: no incident bit), its +y neighbour lost -y, and the -x-side tile is still seen by the tile before it.
  assert.equal(valid(a, words, 17, 16, 9, 0, -1, 0), 0);assert.equal(valid(a, words, 15, 9, 9, -1, 0, 0), 1);
  // No live key owns a row of the retired page.
  for (let k = 0; k < n; k++) if (uniformTileLive(a, k)) { const i = words[k]!, x = i % a.tiles[0], y = Math.floor(i / a.tiles[0]) % a.tiles[1], z = Math.floor(i / (a.tiles[0] * a.tiles[1]));assert.ok(!(x >> 3 === 2 && y >> 3 === 1 && z >> 3 === 1)); }
  // Its rows are the unstored rows: far air, open, their eight residency flags cleared and no other.
  const pages = Math.ceil(a.tiles[0] / 4) * Math.ceil(a.tiles[1] / 4) * Math.ceil(a.tiles[2] / 4), flags = new Uint32Array(pages).fill(1), phi = new Float32Array(a.boxTiles).fill(-3);
  const [tx, ty, tz] = a.tiles, rows = a.boxTiles + 2 * (tx * ty + tx * tz + ty * tz), record = new Float32Array(4 * rows).fill(0.25);
  assert.equal(uniformMixedRootUnstoredRows(a, flags, phi, record, 16), 512);
  assert.equal(flags.reduce((s, v) => s + v, 0), pages - 8);
  for (let i = 0; i < a.boxTiles; i++) {
    const x = i % a.tiles[0], y = Math.floor(i / a.tiles[0]) % a.tiles[1], z = Math.floor(i / (a.tiles[0] * a.tiles[1])), inside = x >> 3 === 2 && y >> 3 === 1 && z >> 3 === 1;
    assert.equal(phi[i], inside ? 16 : -3);assert.deepEqual([...record.subarray(4 * i, 4 * i + 4)], inside ? [1, 1, 1, 1] : [0.25, 0.25, 0.25, 0.25]);
    assert.equal(flags[(x >> 2) + Math.ceil(a.tiles[0] / 4) * ((y >> 2) + Math.ceil(a.tiles[1] / 4) * (z >> 2))], inside ? 0 : 1);
  }
  // Its wall slots (the page is the box's +x, +y, +z corner: 64 tiles on each of three planes): closed, and no other slot is written.
  const own = new Set<number>();
  for (let l = 0; l < 512; l++) {
    const c = [16 + (l & 7), 8 + ((l >> 3) & 7), 8 + (l >> 6)];
    for (let axis = 0; axis < 3; axis++) for (let side = 0; side < 2; side++) if (c[axis] === (side ? a.tiles[axis]! - 1 : 0)) own.add(uniformMixedRootSlot(a, c[0]!, c[1]!, c[2]!, axis, side));
  }
  assert.equal(own.size, 3 * 64);
  for (let s = a.boxTiles; s < rows; s++) assert.deepEqual([...record.subarray(4 * s, 4 * s + 4)], own.has(s) ? [0, 0, 0, 0] : [0.25, 0.25, 0.25, 0.25], `slot ${s - a.boxTiles}`);
  assert.equal(uniformMixedRootSlot(a, tx - 1, 9, 10, 0, 1), a.boxTiles + ty * tz + 9 + ty * 10);
  // A SOLID page's rows are closed.
  a.seed.table[entry(0, 0, 0)] = SOLID;a.seed.blocks[blockOf(live(a), 0, 0, 0)] = FREE;
  record.fill(0.25);uniformMixedRootUnstoredRows(a, flags, phi, record, 16);
  assert.deepEqual([...record.subarray(0, 4)], [0, 0, 0, 0]);assert.equal(phi[0], 16);
});

test("a block record the table does not agree with is fatal, and its owners are withheld", () => {
  const a = live(uniformLatticeAddress([96, 64, 64], false, { blocks: 20, air: [[1, 0, 0]] })), n = a.tileKeys;
  const free = a.seed.blocks.findIndex((r, b) => b > 0 && b < a.blocks - 1 && r === FREE);
  // A second block claims page (0,0,0) without the table.
  a.seed.blocks[free] = entry(0, 0, 0);
  const { words, fatal } = uniformMixedRootOwnerWords(a);
  assert.equal(fatal, UNIFORM_MIXED_ROOT_FATAL);
  for (let k = 512 * free; k < 512 * free + 512; k++) assert.deepEqual([words[k], words[2 * n + 2 * k + 1]], [0, 0]);
  assert.equal(words[uniformTileKey(a, 1, 0, 0)], 1);
});

test("the kernels' text holds the lattice's constants", () => {
  const a = uniformLatticeAddress([96, 64, 64], false, { blocks: 20 }), l = { dimensions: [96, 64, 64], cellSize_m: [0.5, 0.25, 0.5] };
  const code = uniformMixedRootOwnersWGSL(a, l, true), bare = uniformMixedRootOwnersWGSL(a, l, false);
  assert.match(code, /const RO_KEYS:u32=10240u;/);assert.match(code, /const RO_PAGES:u32=96u;/);assert.match(code, /const RO_FAR:f32=8\.0;/);
  assert.match(code, /const RO_RESIDENCY:u32=92188u;/);assert.match(code, /const RO_VIOLATION:u32=92384u;/);
  assert.equal((code.match(/@binding\(\d+\) var<storage/g) ?? []).length, 4);assert.equal((bare.match(/@binding\(\d+\) var<storage/g) ?? []).length, 3);
  assert.ok(!bare.includes("roRecord"));
});

test("a pool smaller than the box: one row a box tile, and the root's arena range holds them", () => {
  // The long slide's lattice with 100 blocks, and the letters' with 130: 192 box pages each.
  for (const [cells, blocks, resident] of [[[512, 96, 128], 100, (x: number) => x < 8], [[256, 192, 128], 130, (_x: number, y: number) => y < 4]] as const) {
    const air: [number, number, number][] = [];
    for (let z = 0; z < cells[2] / 32; z++) for (let y = 0; y < cells[1] / 32; y++) for (let x = 0; x < cells[0] / 32; x++) if (!resident(x, y)) air.push([x, y, z]);
    const a = uniformLatticeAddress(cells, true, { blocks, permute: 5, air }), n = a.tileKeys, label = `${cells.join("x")} in ${blocks} blocks`;
    assert.ok(n < a.boxTiles, `${label}: fewer tile keys than box tiles`);
    const { words, fatal } = uniformMixedRootOwnerWords(a);
    assert.equal(fatal, 0);same(a, words, label);
    // Every stored tile has one key and one row; a row past the tile keys is a box tile's all the same.
    const seen = new Uint8Array(a.boxTiles);let live = 0, most = 0;
    for (let k = 0; k < n; k++) if (uniformTileLive(a, k)) { live++;seen[words[k]!]!++;most = Math.max(most, words[k]!); }
    assert.equal(live, 512 * (192 - air.length));assert.ok(seen.every(v => v <= 1));assert.ok(most >= n && most < a.boxTiles, `${label}: rows are box indices`);
    // The plan: seven fields of the box's rows and wall slots, inside the range sized for the address.
    const shape = uniformMixedAllCoarseLayout(host(a)), count = uniformMixedPressureStorage(shape).count, [tx, ty, tz] = a.tiles;
    assert.equal(count, a.boxTiles + 2 * (tx * ty + tx * tz + ty * tz));
    const bytes = uniformMixedPressureRootBytes(a), plan = planUniformMixedPressureMemory(shape, a, 0, bytes);
    assert.ok(plan.bytes <= bytes, `${label}: ${plan.bytes} of ${bytes} bytes`);
    const fields = [plan.root.pressure, ...plan.root.rhs, ...plan.root.minimum, plan.root.residual, plan.backup];
    for (const f of fields) { assert.equal(f.size, 4 * count);assert.ok(f.offset + f.size <= bytes); }
    for (const f of fields) for (const g of [...fields, plan.root.frozen]) if (f !== g) assert.ok(f.offset + f.size <= g.offset || g.offset + g.size <= f.offset, `${label}: fields overlap`);
    // The last box row and the last wall slot are inside a field.
    assert.equal(uniformMixedRootSlot(a, tx - 1, ty - 1, tz - 1, 2, 1), count - 1);
    // The split authority's workspace on the box's owners (a job per 64 owners and per page, a cut word an owner).
    const groups = Math.ceil(a.boxTiles / 64) + uniformMixedPageCount({ dimensions: a.cells }), scratch = 8 * (1 + 2 * (groups + Math.ceil(groups / 1024)) + a.boxTiles);
    assert.ok(scratch <= plan.root.frozen.size, `${label}: authority workspace ${scratch} of ${plan.root.frozen.size}`);
    // The unstored rows: every AIR page's tiles, by box index, with the flags of their residency pages.
    const pages = uniformMixedPageCount({ dimensions: a.cells }), flags = new Uint32Array(pages).fill(1), phi = new Float32Array(a.boxTiles).fill(-3), record = new Float32Array(4 * count).fill(0.25);
    assert.equal(uniformMixedRootUnstoredRows(a, flags, phi, record, 16), 512 * air.length);
    assert.equal(flags.reduce((s, v) => s + v, 0), pages - 8 * air.length);
    for (let i = 0; i < a.boxTiles; i++) assert.equal(phi[i] === 16, seen[i] === 0);
  }
});

test("the root plan fits its arena range on lattices with every page resident", () => {
  // tileKeys = 512 (pages + 2) sized it before; the box's tiles do now: fewer bytes, never more.
  for (const cells of [[128, 48, 48], [160, 96, 160], [256, 192, 128]] as const) {
    const a = uniformLatticeAddress(cells, true), shape = uniformMixedAllCoarseLayout(host(a));
    const plan = planUniformMixedPressureMemory(shape, a, 0, uniformMixedPressureRootBytes(a));
    assert.ok(plan.bytes <= uniformMixedPressureRootBytes(a));
  }
});
