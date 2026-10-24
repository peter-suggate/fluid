import assert from "node:assert/strict";
import test from "node:test";
import { planUniformPressureSuper, uniformPressureSuperKey, UNIFORM_PRESSURE_SUPER7_SIDE } from "../lib/methods/uniform/uniform-pressure-super";
import { UNIFORM_PRESSURE_ARENA_SCHEDULE } from "../lib/methods/uniform/uniform-pressure-arena";
import { tailExact, tailHierarchy, tailIterate, tailPages, tailPool, tailSuperblocks, type TailV3 } from "./helpers/pressure-tail-twin";

test("the super-block tail's words follow the arena's and never overlap", () => {
  const W = planUniformPressureSuper(9000, 1000, 50_000);
  assert.equal(W.capacity6, 9000);
  assert.equal(W.capacity7, UNIFORM_PRESSURE_SUPER7_SIDE ** 3);
  const index = [W.index.count6, W.index.count7, W.index.list6, W.index.list7, W.index.map6, W.index.map7, W.index.end];
  assert.equal(index[0], 1000);
  assert.deepEqual([...index].sort((a, b) => a - b), index);
  assert.equal(W.index.map7 - W.index.map6, 32 ** 3);
  assert.equal(W.index.end - W.index.map7, 16 ** 3);
  const pool = [W.pool.x6, W.pool.y6, W.pool.r6, W.pool.d6, W.pool.k6, W.pool.c6, W.pool.x7, W.pool.r7, W.pool.d7, W.pool.k7, W.pool.c7, W.pool.end];
  assert.equal(pool[0], 50_000);
  assert.equal(W.pool.c6 - W.pool.k6, 6 * W.capacity6);
  assert.equal(W.pool.end - W.pool.c7, 8 * W.capacity7);
  assert.deepEqual([...pool].sort((a, b) => a - b), pool);
  const small = planUniformPressureSuper(12, 0, 0);
  assert.equal(small.capacity6, 12);
  assert.equal(small.capacity7, 12);
  assert.throws(() => planUniformPressureSuper(0, 0, 0), RangeError);
});

test("a page's group keys count from the box's low page, mod the window", () => {
  assert.equal(uniformPressureSuperKey([0, 0, 0], [0, 0, 0], 1), 0);
  assert.equal(uniformPressureSuperKey([3, 5, 9], [0, 0, 0], 1), 1 | (2 << 5) | (4 << 10));
  assert.equal(uniformPressureSuperKey([3, 5, 9], [0, 0, 0], 2), 0 | (1 << 4) | (2 << 8));
  // A box whose low page sits at window page 62: its pages 62, 63, 0, 1 group together.
  assert.equal(uniformPressureSuperKey([62, 0, 0], [62, 0, 0], 2), uniformPressureSuperKey([1, 0, 0], [62, 0, 0], 2));
  assert.notEqual(uniformPressureSuperKey([61, 0, 0], [62, 0, 0], 2), uniformPressureSuperKey([62, 0, 0], [62, 0, 0], 2));
});

const rhs = (coord: readonly TailV3[]) => {
  let seed = 7;
  const rand = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
  return Float64Array.from(coord, c => rand() - 0.5 + 0.3 * Math.sin(c[0] / 5));
};

test("an aggregate row's operator is its children's, summed and scaled", () => {
  const pages = tailPool([6, 4, 4]), { l6, l7 } = tailHierarchy(pages, 0.5);
  assert.equal(l6.coord.length, 3 * 2 * 2);
  assert.equal(l7.coord.length, 2);
  // On a constant, a row's A·1 (D less its couplings to rows) is the scale
  // times its children's: the couplings across the group's faces cancel.
  const ones = (L: typeof pages, i: number) => { let v = L.D[i]!; for (let f = 0; f < 6; f++) if (L.near[6 * i + f]! >= 0) v -= L.K[6 * i + f]!; return v; };
  for (const [fine, coarse] of [[pages, l6], [l6, l7]] as const) {
    const sum = new Float64Array(coarse.coord.length);
    fine.coord.forEach((_, i) => { sum[fine.parent[i]!]! += ones(fine, i); });
    coarse.coord.forEach((_, e) => assert.ok(Math.abs(ones(coarse, e) - 0.5 * sum[e]!) < 1e-12));
  }
});

const errorOf = (exact: Float64Array) => (X: Float64Array) => Math.sqrt(X.reduce((n, x, i) => n + (x - exact[i]!) ** 2, 0) / exact.reduce((n, x) => n + x * x, 0));

test("the super-block tail solves a pool of more than 4096 pages better than the page tail would", () => {
  // The tail's three corrections a cycle (the schedule's tailCycles) at its scale.
  const pages = tailPool([64, 16, 8]);
  assert.ok(pages.coord.length > 4096);
  const h = tailHierarchy(pages, UNIFORM_PRESSURE_ARENA_SCHEDULE.superScale);
  assert.ok(h.l7.coord.length <= 4096);
  const R = rhs(pages.coord), error = errorOf(tailExact(pages, R)), n = UNIFORM_PRESSURE_ARENA_SCHEDULE.tailCycles;
  const superblocks = error(tailIterate(pages, R, r => tailSuperblocks(h, r, UNIFORM_PRESSURE_ARENA_SCHEDULE.sweeps, 32), n).at(-1)!);
  const paged = error(tailIterate(pages, R, r => tailPages(pages, r, 32), n).at(-1)!);
  assert.ok(superblocks < 0.25, `super-block tail error ${superblocks}`);
  assert.ok(paged > 2 * superblocks, `page tail error ${paged} against ${superblocks}`);
});

test("the super-block tail contracts on ragged pools (part groups at every face)", () => {
  for (const box of [[17, 5, 3], [7, 5, 3], [9, 3, 1], [21, 9, 5], [33, 7, 5]] as TailV3[]) {
    const pages = tailPool(box), h = tailHierarchy(pages, UNIFORM_PRESSURE_ARENA_SCHEDULE.superScale), R = rhs(pages.coord), error = errorOf(tailExact(pages, R));
    const steps = tailIterate(pages, R, r => tailSuperblocks(h, r, UNIFORM_PRESSURE_ARENA_SCHEDULE.sweeps, 32), 3).map(error);
    assert.ok(steps[0]! < 0.5 && steps[2]! < steps[0]! && steps[2]! < 0.15, `${box}: ${steps}`);
  }
});
