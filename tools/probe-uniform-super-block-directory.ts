/** Dawn proof of the page pool's super-block directory (tiered-adaptivity
 * §2.1, uniform-page-pool.ts). Every commit is read back whole and audited:
 * the top table holds AIR or a distinct held slot whose super-block masks to
 * the entry; a held slot's children are AIR or blocks whose records name a
 * page of its super-block, and its count is that many; every resident block
 * is the word of its page by the two-level read; the super ring's free and
 * quarantined runs and the held slots partition 1..superBlocks, and no
 * quarantined slot is in the table; the claim words are clear; the mirror
 * equals the state's tail; the flat window is the two-level word of each of
 * its pages. With idleSteps 1 the resident set is exactly
 * the pages asked for, which a host model checks commit by commit.
 * Scenarios: a super-block claimed, shared, emptied and retired, and claimed
 * again after its quarantine; a walk of 300 commits of random clusters across
 * negative and wrapped coordinates (claims, retires and slot reuse every
 * commit); sparse pages across the solver's 256-page window; the
 * superBlocks fatal; an alias of two super-blocks of one entry.
 * node --import tsx tools/probe-uniform-super-block-directory.ts
 */
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { UniformPagePool, uniformPagePoolBlockWord, uniformPagePoolEntry, uniformPagePoolFatalMessage, uniformPagePoolWord,
  UNIFORM_PAGE_POOL_BLOCK as BLOCK, UNIFORM_PAGE_POOL_FATAL as FATAL, UNIFORM_PAGE_POOL_FREE as FREE, UNIFORM_PAGE_POOL_STATUS as STATUS,
  UNIFORM_PAGE_POOL_SUPER as SUPER, UNIFORM_PAGE_POOL_WORK as WORK, type UniformPagePoolOptions } from "../lib/methods/uniform/uniform-page-pool";
import { UNIFORM_SUPER_BLOCK_SLOTS, uniformSuperBlockLocal } from "../lib/methods/uniform/uniform-window-address";

type Page = readonly [number, number, number];
const key = (p: Page) => p.join(","), started = performance.now(), errors: string[] = [];
const random = (seed: number) => () => { seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve("node_modules/webgpu/index.js")).href); Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter(); assert.ok(adapter);
const device = await adapter.requestDevice();
device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); console.error(e.error.message); });

async function harness(options: UniformPagePoolOptions) {
  const pool = new UniformPagePool(device, options), l = pool.layout, offset = 7;
  const mirror = device.createBuffer({ size: 4 * (offset + l.tail.words + 5), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  await pool.initialize([{ buffer: mirror, offset }]);
  const read = async (buffer: GPUBuffer, words: number, at = 0) => {
    const r = device.createBuffer({ size: 4 * words, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const e = device.createCommandEncoder(); e.copyBufferToBuffer(buffer, 4 * at, r, 0, 4 * words); device.queue.submit([e.finish()]);
    await r.mapAsync(GPUMapMode.READ); const out = new Uint32Array(r.getMappedRange().slice(0)); r.unmap(); r.destroy(); return out;
  };
  let commits = 0;
  /** One commit asking for `pages`; then the audit. Returns the status. */
  const commit = async (pages: readonly Page[], expectFatal = 0) => {
    for (const p of pages) pool.request(p[0], p[1], p[2]);
    const e = device.createCommandEncoder(); pool.encodeCommit(e); device.queue.submit([e.finish()]); commits++;
    const [state, work, tail] = await Promise.all([read(pool.state, l.state.words), read(pool.work, l.work.words), read(mirror, l.tail.words, offset)]);
    const at = `commit ${commits}`, signed = new Int32Array(state.buffer), T = l.state.table, fatal = state[STATUS.fatal]!;
    assert.equal(fatal, expectFatal, `${at}: fatal ${fatal} (${uniformPagePoolFatalMessage(l, state)})`);
    assert.deepEqual(tail, state.subarray(T, T + l.tail.words), `${at}: the mirror is the state's tail`);
    assert.ok(work.subarray(l.work.claims, l.work.claims + 5 * l.entries).every(w => w === 0), `${at}: claim words`);
    // The flat window: the two-level word of lattice pages -1..62 under the frame.
    const F = [signed[l.state.window]!, signed[l.state.window + 1]!, signed[l.state.window + 2]!];
    for (let z = 0, i = 0; z < 64; z++) for (let y = 0; y < 64; y++) for (let x = 0; x < 64; x++, i++) {
      const w = state[T + l.tail.flat + i]!, want = uniformPagePoolWord(l, state, x - 1 + F[0]!, y - 1 + F[1]!, z - 1 + F[2]!, T);
      if (w !== want) assert.fail(`${at}: flat word ${i} is ${w}, the directory's ${want}`);
    }
    // Super-block slots: records, ring and table.
    const held = new Map<number, Page>(), pending: number[] = [], free: number[] = [];
    for (let s = 1; s <= l.superBlocks; s++) {
      const r = l.state.supers + SUPER.words * s, st = state[r + SUPER.state]!;
      if (st === BLOCK.resident) held.set(s, [signed[r]!, signed[r + 1]!, signed[r + 2]!]);
      else if (st === BLOCK.pending) pending.push(s); else { assert.equal(st, BLOCK.free, `${at}: slot ${s} state ${st}`); free.push(s); }
    }
    assert.deepEqual([state[STATUS.superResident], state[STATUS.superFree], state[STATUS.superPending]], [held.size, free.length, pending.length], `${at}: super status`);
    const head = work[WORK.superHead]!, ring = Array.from({ length: free.length + pending.length }, (_, i) => work[l.work.superRing + ((head + i) & (UNIFORM_SUPER_BLOCK_SLOTS - 1))]!);
    assert.deepEqual(ring.slice(0, free.length).sort((a, b) => a - b), free, `${at}: the ring's free run`);
    assert.deepEqual(ring.slice(free.length).sort((a, b) => a - b), pending, `${at}: the ring's quarantined run`);
    const topHeld = new Set<number>();
    for (let e = 0; e < l.entries; e++) {
      const w = state[T + e]!; if (w === 0) continue;
      assert.ok(held.has(w), `${at}: entry ${e} holds slot ${w}, not held`); assert.ok(!topHeld.has(w), `${at}: slot ${w} in two entries`); topHeld.add(w);
      const sb = held.get(w)!; assert.equal(uniformPagePoolEntry(l, 4 * sb[0], 4 * sb[1], 4 * sb[2]), e, `${at}: slot ${w} masks to entry ${e}`);
    }
    assert.equal(topHeld.size, held.size, `${at}: every held slot is in the table`);
    // Blocks: each resident block is its page's word; each held slot counts its blocks.
    const resident = new Map<string, number>(), count = new Map<number, number>();
    for (let b = 1; b < l.blocks - 1; b++) {
      const r = l.state.records + 8 * b, st = state[r + 3]!, page: Page = [signed[r]!, signed[r + 1]!, signed[r + 2]!];
      if (st !== BLOCK.resident) { assert.equal(state[l.state.blocks + b], FREE, `${at}: block ${b} not resident has a word`); continue; }
      assert.equal(uniformPagePoolWord(l, state, page[0], page[1], page[2], T), b, `${at}: block ${b} is the word of page ${page}`);
      assert.equal(state[l.state.blocks + b], uniformPagePoolBlockWord(l, page[0], page[1], page[2]), `${at}: block ${b}'s word`);
      const slot = state[T + uniformPagePoolEntry(l, page[0], page[1], page[2])]!;
      assert.deepEqual(held.get(slot), page.map(v => v >> 2), `${at}: block ${b}'s super-block`);
      count.set(slot, (count.get(slot) ?? 0) + 1); resident.set(key(page), b);
    }
    for (const [s, sb] of held) {
      let children = 0;
      for (let i = 0; i < 64; i++) {
        const w = state[T + l.entries + 64 * s + i]!; if (w === 0) continue; children++;
        const r = l.state.records + 8 * w, page = [signed[r]!, signed[r + 1]!, signed[r + 2]!];
        assert.deepEqual([page.map(v => v >> 2), uniformSuperBlockLocal(page[0]!, page[1]!, page[2]!)], [[...sb], i], `${at}: slot ${s} child ${i} is block ${w}`);
      }
      assert.equal(work[l.work.counts + s], children, `${at}: slot ${s}'s count`); assert.equal(count.get(s) ?? 0, children, `${at}: slot ${s}'s blocks`);
      assert.ok(children > 0, `${at}: held slot ${s} holds no page`);
    }
    for (const s of pending) assert.equal(work[l.work.counts + s], 0, `${at}: quarantined slot ${s}'s count`);
    const residents = [...resident.values()].sort((a, b) => a - b);
    assert.deepEqual([...state.subarray(l.state.resident, l.state.resident + state[STATUS.resident]!)], residents, `${at}: the resident list`);
    // idleSteps 1: what this commit asked for is what is resident.
    if (!expectFatal) assert.deepEqual([...resident.keys()].sort(), [...new Set(pages.map(key))].sort(), `${at}: the resident pages`);
    return { state, held, pending, resident, message: uniformPagePoolFatalMessage(l, state) };
  };
  return { pool, l, commit, destroy: () => { pool.destroy(); mirror.destroy(); } };
}

// A super-block claimed, shared, emptied, retired and claimed again.
{
  const h = await harness({ table: 16, blocks: 40, idleSteps: 1, superBlocks: 4, hostCapacity: 64 });
  let s = await h.commit([[0, 0, 0], [1, 0, 0], [3, 3, 3]]);
  assert.equal(s.held.size, 1); const first = [...s.held.keys()][0]!;
  s = await h.commit([[0, 0, 0], [4, 0, 0]]);
  assert.equal(s.held.size, 2); assert.ok(s.held.has(first), "the shared super-block keeps its slot");
  s = await h.commit([[4, 0, 0]]);
  assert.deepEqual(s.pending, [first], "an emptied super-block retires into quarantine");
  s = await h.commit([[0, 0, 0], [-1, -1, -1], [-4, 7, 12]]);
  assert.equal(s.held.size, 3); assert.ok(!s.pending.includes(first));
  s = await h.commit([]);
  assert.equal(s.held.size, 0); assert.equal(s.pending.length, 3);
  h.destroy();
}
// A walk of random clusters: a 7-page box around a centre that moves a page a
// commit through negative and wrapped coordinates; old and new clusters never
// span four super-blocks an axis, so nothing aliases.
{
  const h = await harness({ table: 16, blocks: 160, idleSteps: 1, superBlocks: 40, hostCapacity: 256, requestCapacity: 256 }), rand = random(17);
  const centre = [-40, 3, 1000];
  let reused = 0;
  const seen = new Set<number>();
  for (let n = 0; n < 300; n++) {
    centre[Math.floor(rand() * 3)]! += rand() < 0.5 ? -1 : 1;
    const pages = new Map<string, Page>();
    for (let i = 0; i < 40; i++) { const p: Page = [centre[0]! - 3 + Math.floor(rand() * 7), centre[1]! - 3 + Math.floor(rand() * 7), centre[2]! - 3 + Math.floor(rand() * 7)]; pages.set(key(p), p); }
    const s = await h.commit([...pages.values()]);
    const supers = new Set([...pages.values()].map(p => p.map(v => v >> 2).join(",")));
    assert.equal(s.held.size, supers.size, `walk ${n}: one slot a super-block asked for`);
    for (const slot of s.held.keys()) { if (seen.has(slot)) reused++; seen.add(slot); }
  }
  assert.ok(reused > 0 && seen.size <= 40);
  console.log(JSON.stringify({ walk: 300, slotsUsed: seen.size, slotReuses: reused }));
  h.destroy();
}
// The solver's window, 256 pages an axis: sparse pages across 250 of them,
// inside and past the flat window, then a shifted set; slots retire behind them.
{
  const h = await harness({ blocks: 64, idleSteps: 1, superBlocks: 64, hostCapacity: 64 });
  const xs = [-1, 0, 31, 61, 62, 63, 64, 100, 127, 128, 200, 249];
  let s = await h.commit(xs.flatMap(x => [[x, 0, 0], [x, 70, 3], [x, 2, 130]] as Page[]));
  assert.equal(s.resident.size, 3 * xs.length);
  s = await h.commit(xs.map(x => [x + 1, 5, 250] as Page));
  assert.equal(s.resident.size, xs.length);
  console.log(JSON.stringify({ window: 256, pages: 3 * xs.length, held: s.held.size }));
  h.destroy();
}
// The superBlocks fatal: three super-blocks asked of two slots.
{
  const h = await harness({ table: 16, blocks: 40, idleSteps: 1, superBlocks: 2, hostCapacity: 64 });
  const s = await h.commit([[0, 0, 0], [4, 0, 0], [8, 0, 0]], FATAL.superBlocks);
  assert.match(s.message ?? "", /Super-block slots exhausted at commit 1: page \(\d+, 0, 0\) needed one with 0 of 2 held before the commit/);
  h.destroy();
}
// An alias: super-blocks 0 and 4 share entry 0 of a 4-entry axis; the first
// is still held when the second is claimed (claims precede retirement).
{
  const h = await harness({ table: 16, blocks: 40, idleSteps: 1, superBlocks: 4, hostCapacity: 64 });
  await h.commit([[0, 0, 0]]);
  const s = await h.commit([[17, 0, 0]], FATAL.alias);
  assert.match(s.message ?? "", /one page directory entry \(page \(17, 0, 0\), commit 2\)/);
  h.destroy();
}
await device.queue.onSubmittedWorkDone();
assert.deepEqual(errors, []);
console.log(JSON.stringify({ ok: true, ms: Math.round(performance.now() - started) }));
device.destroy();
