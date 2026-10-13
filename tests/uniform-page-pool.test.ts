import assert from "node:assert/strict";
import test from "node:test";
import { uniformPagePoolBoxPages, uniformPagePoolBoxSeed, uniformPagePoolCellBoxPages, uniformPagePoolCertificateWGSL, uniformPagePoolEntry, uniformPagePoolFatalMessage, uniformPagePoolLayout, uniformPagePoolPublished,
  UNIFORM_PAGE_POOL_FATAL as FATAL, UNIFORM_PAGE_POOL_SOLID as SOLID, UNIFORM_PAGE_POOL_STATUS as STATUS } from "../lib/methods/uniform/uniform-page-pool";

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
  assert.deepEqual([...uniformPagePoolPublished(seed)], [32, 3, 4, 2, 6, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
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

test("the fatal message is null without a fatal and names the first bit's page, commit and detail", () => {
  const status = new Uint32Array(4 + STATUS.words), at = 4;
  assert.equal(uniformPagePoolFatalMessage(letters, status, at), null);
  const raise = (bits: number, first: number, page: readonly [number, number, number], commit: number, detail: number) => {
    status[at + STATUS.fatal] = bits; status[at + STATUS.fatalFirst] = first; status[at + STATUS.fatalGeneration] = commit; status[at + STATUS.fatalDetail] = detail;
    new Int32Array(status.buffer).set(page, at + STATUS.fatalPage);
    return uniformPagePoolFatalMessage(letters, status, at);
  };
  assert.equal(raise(FATAL.exhausted, FATAL.exhausted, [3, -2, 1], 12, 200), "Page pool exhausted at commit 12: page (3, -2, 1) requested with 200 of 206 blocks resident and the other 6 claimed by the same commit. Raise pagePool.blocks (4.9 MiB each) or shrink the scene.");
  assert.equal(raise(FATAL.requests, FATAL.requests, [0, 0, 5], 3, 0), "More than claimsPerStep (206) pages were requested in commit 3 (first dropped: page (0, 0, 5)).");
  assert.match(raise(FATAL.alias, FATAL.alias, [64, 0, 0], 2, 0)!, /^Two pages name one page table entry \(page \(64, 0, 0\), commit 2\)/);
  assert.equal(raise(FATAL.closureCap, FATAL.closureCap, [1, 1, 1], 9, 21), "Departure travel gives a closure of 21 tiles, over the cap of 16 (commit 9, page (1, 1, 1)): a sample is moving faster than the residency certificate covers.");
  // The first bit raised decides the message, whatever joined it later.
  assert.equal(raise(FATAL.closure | FATAL.exhausted, FATAL.closure, [7, 5, 3], 4, 0), "Liquid or band tile found in page (7, 5, 3), which was not resident at the start of commit 4.");
});

test("the certificate source names its accessors and omits the audit without a seeded accessor", () => {
  const full = uniformPagePoolCertificateWGSL({ count: "umResidentPageCount", page: "pagePoolCoord", seeded: "pagePoolSeeded", reach: "umSupport[UM_RESIDENCY+2u]" });
  assert.match(full, /fn upCertify\(lane:u32,width:u32\)/);
  assert.match(full, /let count=umResidentPageCount\(\);let reach=umSupport\[UM_RESIDENCY\+2u\];/);
  assert.match(full, /UP_BOX_ORIGIN\+\(vec3i\(pagePoolCoord\(i\)\)>>vec3u\(1u\)\)/);
  assert.match(full, /if\(pagePoolSeeded\(i\)&&upBlockOf\(q\)==UP_NONE\)\{upFail\(UP_FATAL_CLOSURE,q\);\}/);
  const plain = uniformPagePoolCertificateWGSL({ count: "n", page: "p", reach: "0u", name: "ask" });
  assert.match(plain, /fn ask\(/);
  assert.doesNotMatch(plain, /UP_FATAL_CLOSURE,/);
});
