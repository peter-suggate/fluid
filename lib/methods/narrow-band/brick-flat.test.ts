/**
 * FLAT, the directory's brick -> F slot table (brick-abi.md §3): the walk page -> child, resolved, a word a window
 * brick, which nbFSlotOfBrick reads in one load.
 *
 * CPU: the region scales with the page window (64 words a page word, never the lattice), its index is the brick in
 * the window of bricks (toroidal, x fastest), and flatMismatches finds a word that differs from the walk.
 * GPU (WEBGPU_NODE_MODULE): FLAT equals the walk at boot and after every step of a residency that enters bricks and
 * pages; setFlatCheck stays clean there and latches FAIL.UNSUPPORTED on a stale word.
 * The scenes' own gate (four gate states, edit and undo) is tools/experiments/gpu-shape/flatcheck.mts and
 * brick-solid-edit.test.ts; brick-engine.test.ts's consistency check asserts it at every churn step.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { BrickLayout, FAIL, H, NONE } from "./brick-abi";
import { BrickEngine, type BrickEngineOptions, type BrickLiquidBox, type Vec3 } from "./brick-engine";
import { bringUpCG } from "./brick-bringup-cg";
import { bringUpStages } from "./brick-stages";
import { FLAT_CHECK_GROUPS, brickSetKernels } from "./brick-set.wgsl";

const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };
const caps = { f: 400, c: 16, b: 0, k: 0, particles: 1024 };

test("brick FLAT: 64 words a window page, the walk's index, and a stale word is found", () => {
  const L = new BrickLayout({ lattice: [128, 64, 128], ...caps });
  const pages = L.window[0] * L.window[1] * L.window[2];
  assert.ok(L.dir.flat >= L.dir.child + 64 * caps.c && L.dir.frec >= L.dir.flat + 64 * pages, "FLAT lies between CHILD and FREC");
  assert.ok(L.dir.flat < L.dir.cmask, "FLAT lies before CMASK (the build's NONE fill)");
  // Brick (5, 2, 7) in the window of bricks (4 W an axis), x fastest; toroidal: one window over is the same word.
  const BW = L.window.map(w => 4 * w);
  assert.equal(L.flatIndex(5, 2, 7), L.dir.flat + 5 + BW[0]! * (2 + BW[1]! * 7));
  assert.equal(L.flatIndex(5 + BW[0]!, 2, 7 + BW[2]!), L.flatIndex(5, 2, 7));
  assert.equal(new Set(Array.from({ length: 64 * pages }, (_, i) => L.flatIndex(i % BW[0]!, Math.floor(i / BW[0]!) % BW[1]!, Math.floor(i / (BW[0]! * BW[1]!))))).size, 64 * pages, "one word a window brick");
  const dir = new Uint32Array(L.dir.words).fill(NONE);
  assert.deepEqual(L.flatMismatches(dir), { count: 0 });
  // A resident page (C slot 3) with brick (5, 2, 7) at F slot 9: the walk reads 9, FLAT NONE until written.
  dir[L.dir.page + L.pageIndex(1, 0, 1)] = 3; dir[L.dir.child + 64 * 3 + (1 | (2 << 2) | (3 << 4))] = 9;
  assert.equal(L.flatMismatches(dir).count, 1);
  dir[L.flatIndex(5, 2, 7)] = 9;
  assert.deepEqual(L.flatMismatches(dir), { count: 0 });
  // A claim in flight (CLAIM, bit 31 set) is absent to the walk's readers, so FLAT holds NONE for it.
  dir[L.dir.child + 64 * 3 + 0] = 0xc0000000;
  assert.deepEqual(L.flatMismatches(dir), { count: 0 });
  // The 4x-wider lattice grows FLAT by its pages alone.
  const W = new BrickLayout({ lattice: [512, 64, 128], ...caps });
  assert.equal(W.dir.frec - W.dir.flat - (L.dir.frec - L.dir.flat), 64 * (W.window[0] * W.window[1] * W.window[2] - pages));
});

function options(lattice: Vec3, liquid: BrickLiquidBox[]): BrickEngineOptions {
  return { lattice, h_m: 1 / 32, dt_s: 1 / 60, gravity_mps2: [0, -9.81, 0], capacities: caps, liquid, solver: L => bringUpCG(L), stages: bringUpStages };
}

test("brick FLAT (GPU): the walk at every step of a churning residency; setFlatCheck latches a stale word", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("brick flat", async device => {
    // A column that falls and spreads: bricks and pages enter as the surface moves. (Leaving: brick-engine.test.ts's
    // enter/leave test, whose consistency check asserts FLAT after a head where bricks leave.)
    const e = await BrickEngine.create(device, options([128, 64, 128], [{ lo: [-1, -1, -1], hi: [20, 40.3, 20] }]));
    const L = e.layout, check = brickSetKernels(L).setFlatCheck;
    const run = async () => { await e.run([{ kernel: check, dispatch: { direct: [FLAT_CHECK_GROUPS, 1, 1] } }]); return new Uint32Array(await e.read("set", 0, 256)); };
    let entered = 0, left = 0, held = new Set<number>();
    for (let step = 0; step <= 12; step++) {
      if (step > 0) e.advance(1);
      const dir = await e.read("dir"), header = await run();
      const flat = L.flatMismatches(dir);
      assert.equal(flat.count, 0, `step ${step}: FLAT differs from the walk at ${flat.count} bricks (${flat.first})`);
      assert.equal(header[H.FAIL], 0, `step ${step}: FAIL 0x${header[H.FAIL]!.toString(16)} info 0x${header[H.FAIL_INFO]!.toString(16)}`);
      // The bricks FLAT holds (by their FLAT word), against the last step's.
      const now = new Set<number>(); for (let i = L.dir.flat; i < L.dir.frec; i++) if (dir[i] !== NONE) now.add(i);
      if (step > 0) { for (const i of now) if (!held.has(i)) entered++; for (const i of held) if (!now.has(i)) left++; }
      held = now;
    }
    assert.ok(entered > 0, `the residency churned (${entered} entered, ${left} left)`);
    // A stale word: a resident brick's FLAT word NONE.
    const dir = await e.read("dir");
    let at = -1;
    for (let i = L.dir.flat; i < L.dir.frec && at < 0; i++) if (dir[i] !== NONE) at = i;
    assert.ok(at >= 0, "a resident brick");
    e.write("dir", at, new Uint32Array([NONE]));
    const header = await run();
    assert.equal(header[H.FAIL]! & FAIL.UNSUPPORTED, FAIL.UNSUPPORTED, "setFlatCheck latches the stale word");
    assert.equal(header[H.FAIL_INFO], (0x464c0000 | ((at - L.dir.flat) & 0xffff)) >>> 0, "with its offset");
    e.destroy();
  });
});
