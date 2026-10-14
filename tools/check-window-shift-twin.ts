/** Host-only oracle for the moving-window arithmetic and existing pool
 * records. Physical solver invariance is checked separately by the device tool.
 * Run: node --import tsx tools/check-window-shift-twin.ts */
import assert from "node:assert/strict";
import { uniformWindowDecision, type UniformWindowPage } from "../lib/methods/uniform/uniform-moving-window";
import { UNIFORM_PAGE_POOL_FATAL, UNIFORM_PAGE_POOL_KIND, UNIFORM_PAGE_POOL_RECORD, uniformPagePoolEntry, uniformPagePoolLayout, uniformPagePoolSeed } from "../lib/methods/uniform/uniform-page-pool";

let random = 0x713ad092;
const next = () => { random ^= random << 13;random ^= random >>> 17;random ^= random << 5;return random >>> 0; };
const between = (low: number, high: number) => low + next() % (high - low + 1);
const page = (f: (axis: number) => number): UniformWindowPage => [f(0), f(1), f(2)];
let shifts = 0, pages = 0, forced = 0, poolRecords = 0;
try {
  for (let trial = 0; trial < 10000; trial++) {
    const frame = page(() => between(-10000000, 10000000));
    const width = page(() => between(1, 60));
    const low = page(a => frame[a]! + between(0, 64 - width[a]!));
    const high = page(a => low[a]! + width[a]! - 1);
    const bounds = { low, high };
    // Both automatic shifts and arbitrary legal k: the latter can touch
    // either edge and prove indexing independently of the centring policy.
    const delta = trial % 2 ? page(a => between(high[a]! - 63, low[a]!) - frame[a]!) : undefined;
    const result = uniformWindowDecision(frame, bounds, delta);
    assert.equal(result.fatal, 0);
    if (delta) { assert.deepEqual(result.delta, delta);forced++; }
    if (result.delta.some(v => v !== 0)) shifts++;
    const residents = new Map<string, UniformWindowPage>();
    for (let i = 0; i < 32; i++) {
      const world = i === 0 ? low : i === 1 ? high : page(a => between(low[a]!, high[a]!));
      residents.set(world.join(","), world);
      for (let a = 0; a < 3; a++) {
        const before = world[a]! - frame[a]!, after = world[a]! - result.frame[a]!;
        assert.ok(before >= 0 && before < 64 && after >= 0 && after < 64);
        assert.equal((before + frame[a]!) & 63, world[a]! & 63);
        assert.equal((after + result.frame[a]!) & 63, world[a]! & 63);
        // A six-bit residue is sufficient to recover a lattice page iff
        // the full signed record is known to lie inside this window.
        assert.equal(((world[a]! & 63) - result.frame[a]!) & 63, after);
        const cell = Math.fround(32 * before + 15.25), rebased = Math.fround(cell - 32 * result.delta[a]!);
        assert.ok(rebased >= 0 && rebased < 2048);
        assert.equal(rebased + 32 * result.frame[a]!, cell + 32 * frame[a]!);
      }
      pages++;
    }
    if (trial < 128) {
      const held = [...residents.values()].map((q, i) => ({ x: q[0], y: q[1], z: q[2], block: i + 1 }));
      const layout = uniformPagePoolLayout({ blocks: held.length + 2, idleSteps: 8 });
      const seed = uniformPagePoolSeed(layout, held, () => UNIFORM_PAGE_POOL_KIND.air);
      const signed = new Int32Array(seed.state.buffer);
      const snapshot = seed.table.slice();
      for (const q of held) {
        const r = layout.state.records + UNIFORM_PAGE_POOL_RECORD.words * q.block;
        assert.deepEqual([...signed.subarray(r, r + 3)], [q.x, q.y, q.z]);
        const world = [q.x, q.y, q.z];
        const before = page(a => world[a]! - frame[a]! + frame[a]!);
        const after = page(a => world[a]! - result.frame[a]! + result.frame[a]!);
        const first = uniformPagePoolEntry(layout, ...before), last = uniformPagePoolEntry(layout, ...after);
        assert.equal(first, last);assert.equal(seed.table[last], q.block);poolRecords++;
      }
      assert.deepEqual(seed.table, snapshot, "a shift must not rebuild the page table");
    }
  }
  // An incoming request can already be just outside the old window, while
  // its union with the old residents fits comfortably in a new one.
  const crossing = uniformWindowDecision([0, 0, 0], { low: [60, 1, 1], high: [66, 2, 2] });
  assert.equal(crossing.fatal, 0);assert.ok(crossing.frame[0] > 0);
  const edgeCases = [
    uniformWindowDecision([0, 0, 0], { low: [0, 0, 0], high: [64, 1, 1] }),
    uniformWindowDecision([0, 0, 0], { low: [0, 0, 0], high: [60, 1, 1] }),
    uniformWindowDecision([0, 0, 0], { low: [0, 0, 0], high: [1, 1, 1] }, [1, 0, 0]),
    uniformWindowDecision([0, 0, 0], { low: [-2147483648, 0, 0], high: [2147483647, 1, 1] }),
    uniformWindowDecision([2147483584, 0, 0], undefined, [1, 0, 0]),
    uniformWindowDecision([-2147483648, 0, 0], undefined, [-1, 0, 0]),
  ];
  for (const failed of edgeCases) { assert.equal(failed.fatal, UNIFORM_PAGE_POOL_FATAL.edge);assert.deepEqual(failed.delta, [0, 0, 0]); }
  const lo = uniformWindowDecision([-2147483648, 0, 0], { low: [-2147483648, 0, 0], high: [-2147483640, 1, 1] });
  const hi = uniformWindowDecision([2147483584, 0, 0], { low: [2147483639, 0, 0], high: [2147483647, 1, 1] });
  assert.equal(lo.fatal, 0);assert.equal(lo.frame[0], -2147483648);
  assert.equal(hi.fatal, 0);assert.equal(hi.frame[0], 2147483584);
  assert.deepEqual(uniformWindowDecision([0, 0, 0]), { frame: [0, 0, 0], delta: [0, 0, 0], fatal: 0 });
  // A stale idle block cannot be ignored: its union with the request no
  // longer fits. Six-bit residues alone cannot detect this alias.
  assert.equal((0 & 63), (64 & 63));
  assert.equal(uniformWindowDecision([0, 0, 0], { low: [0, 10, 10], high: [64, 10, 10] }).fatal, UNIFORM_PAGE_POOL_FATAL.edge);
  console.log(`PASS host twin: sets=10000 pages=${pages} shifts=${shifts} forced=${forced} signedPoolRecords=${poolRecords} tableMoves=0`);
  console.log(`PASS UNIFORM_PAGE_POOL_FATAL.edge=${UNIFORM_PAGE_POOL_FATAL.edge}: rejected=${edgeCases.length} (65-page extent, margin, invalid force, signed overflow); signed endpoint cases=2`);
  console.log("SCOPE: host address/decision arithmetic; see check-window-shift.ts for the fluid gate");
} catch (error) { console.error("FAIL moving-window host twin", error);process.exitCode = 1; }
