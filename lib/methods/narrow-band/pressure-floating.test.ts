/**
 * A detail engine's floating components (pressure-floating.wgsl.ts, its Float64 mirror pressure-floating.ts):
 * - the singular case: a block of liquid bounded only by Neumann faces (the mask's prescribed edge, the container's
 *   walls) has rows summing to zero, so the multigrid's aggregates of it are singular and the coarsest Gauss-Jordan
 *   meets a zero pivot (the detail engine's NB_FAIL_PIVOT); with an incompatible RHS no solution exists;
 * - the fix: its data made compatible and its gauge pinned to the parent's pressure, the hierarchy builds, the solve
 *   converges to the Neumann problem's solution (A x = b - mean b on the original rows) with x_root = g, a second
 *   gauge moves it by a constant only, and an anchored component's rows and solution are untouched;
 * - the schedule: a detail layout's setup runs the four kernels between the walls' guess decision and setupF; a
 *   parent layout's setup has none, and the allocator stores the parent's tile pressure a slot.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { BrickLayout } from "./brick-abi";
import { MAX_STORAGE_BINDINGS } from "./brick-kernel";
import { brickDetailSrc, type BrickDetailLayout } from "./brick-detail.wgsl";
import { poolMGPCG } from "./pressure-mgpcg";
import { PressureModel, applyLevel, packCell, unpackCell, type ModelLevel } from "./pressure-model";
import { anchorFloating, type FloatingRows } from "./pressure-floating";

const AX = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] as const;

/** Unit-coupled rows of the cells [lo, hi) (each box its own component); a box with `air` has air above its top
 * layer (a ghost at θ = 1/2: coupling 2 on the diagonal only). Every other face is Neumann. */
function rows(boxes: readonly { lo: readonly number[]; hi: readonly number[]; air: boolean }[], rhs: (i: number) => number): FloatingRows & { box: number[] } {
  // Sorted by key: the model's level order, so a row's index is its index in the model's vectors.
  const cells: { key: number; box: number; air: boolean }[] = [];
  boxes.forEach((b, j) => { for (let z = b.lo[2]!; z < b.hi[2]!; z++) for (let y = b.lo[1]!; y < b.hi[1]!; y++) for (let x = b.lo[0]!; x < b.hi[0]!; x++) cells.push({ key: packCell(x, y, z), box: j, air: b.air && y === b.hi[1]! - 1 }); });
  cells.sort((p, q) => p.key - q.key);
  const keys = cells.map(c => c.key), box = cells.map(c => c.box), air = cells.map(c => c.air);
  const set = new Set(keys), n = keys.length, diag = new Float64Array(n), k = [new Float64Array(n), new Float64Array(n), new Float64Array(n)] as const;
  keys.forEach((key, i) => {
    const [x, y, z] = unpackCell(key);
    for (let a = 0; a < 3; a++) {
      const d = AX[a]!;
      if (set.has(packCell(x - d[0], y - d[1], z - d[2]))) { diag[i]! += 1; k[a][i] = 1; }
      if (set.has(packCell(x + d[0], y + d[1], z + d[2]))) diag[i]! += 1;
    }
    if (air[i]) diag[i]! += 2;
  });
  return { keys, diag, k, rhs: Float64Array.from({ length: n }, (_, i) => rhs(i)), box };
}

// A floating block (an aggregate of level 3 alone) beside an anchored one; the RHS's sum on the floating block is not 0.
const RHS = (i: number) => Math.sin(1.7 * i) * 1e-3 + 2e-5;
const R = rows([{ lo: [0, 0, 0], hi: [8, 8, 8], air: false }, { lo: [16, 0, 0], hi: [24, 8, 8], air: true }], RHS);
const LEVELS = 4;

test("floating components: an all-Neumann block's coarsest pivot is zero", () => {
  assert.throws(() => new PressureModel(R, { levels: LEVELS }), /non-positive pivot/);
  const f = anchorFloating(R, () => 0);
  assert.equal(f.components.length, 2);
  const floating = f.components.filter(c => !c.anchored);
  assert.equal(floating.length, 1);
  assert.equal(floating[0]!.rows, 512);
  assert.ok(Math.abs(floating[0]!.sum) > 1e-3, "the floating block's data are incompatible");
});

test("floating components: compatible data and the parent's gauge give the Neumann solution", () => {
  const g = 0.37, g2 = -1.25;
  const solve = (gauge: number) => {
    const f = anchorFloating(R, () => gauge);
    const model = new PressureModel({ keys: R.keys, diag: f.diag, k: R.k }, { levels: LEVELS });
    const out = model.solve(f.rhs, new Float64Array(R.keys.length), 1e-13, 400);
    assert.ok(out.residual <= 1e-13, `converged (${out.residual})`);
    return { f, x: out.x };
  };
  const { f, x } = solve(g), { x: x2 } = solve(g2);
  const fl = f.components.find(c => !c.anchored)!, n = R.keys.length;
  // The original operator (unpinned) on the projected data, row by row.
  const A0: ModelLevel = { shift: 0, keys: R.keys, index: new Map(R.keys.map((key, i) => [key, i])), diag: Float64Array.from(R.diag), k: R.k as ModelLevel["k"] };
  const Ax = applyLevel(A0, x), mean = fl.sum / fl.rows;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    if (f.root[i] !== fl.root) continue;
    sum += f.rhs[i]! - (i === fl.root ? R.diag[i]! * g : 0);
    assert.ok(Math.abs(Ax[i]! - (R.rhs[i]! - mean)) < 1e-11, `row ${i}: A x = b - mean b`);
    assert.ok(Math.abs(x2[i]! - x[i]! - (g2 - g)) < 1e-10, `row ${i}: the gauge moves x by a constant`);
  }
  assert.ok(Math.abs(sum) < 1e-15, "compatible data");
  assert.ok(Math.abs(x[fl.root]! - g) < 1e-11, "x_root = g");
  // The anchored block: its rows as built, and its solution the original system's.
  const anchored = R.keys.map((_, i) => i).filter(i => R.box[i] === 1);
  for (const i of anchored) { assert.equal(f.diag[i], R.diag[i]); assert.equal(f.rhs[i], R.rhs[i]); }
  const B = { keys: anchored.map(i => R.keys[i]!), diag: anchored.map(i => R.diag[i]!), k: [0, 1, 2].map(a => anchored.map(i => R.k[a]![i]!)) as unknown as FloatingRows["k"] };
  const xb = new PressureModel(B, { levels: LEVELS }).solve(Float64Array.from(anchored, i => R.rhs[i]!), new Float64Array(anchored.length), 1e-13, 400).x;
  anchored.forEach((i, j) => assert.ok(Math.abs(x[i]! - xb[j]!) < 1e-10, `anchored row ${i} untouched`));
});

test("floating components: the detail layout's setup, between the guess's walls and setupF", () => {
  const detail: BrickDetailLayout = { factor: 4, tiles: [4, 4, 4], budget: 8, slots: 24 };
  const base = { lattice: [64, 64, 64] as [number, number, number], box: [64, 64, 64] as [number, number, number], f: 512, c: 8, b: 16, k: 0, particles: 1024 };
  const labels = (L: BrickLayout) => poolMGPCG(L, { walls: true }).setup.map(p => p.kernel.label);
  const fine = labels(new BrickLayout({ ...base, detail })), parent = labels(new BrickLayout(base));
  const at = fine.indexOf("pmgWallsG");
  assert.deepEqual(fine.slice(at, at + 6), ["pmgWallsG", "pmgFloatInit", "pmgFloatHook", "pmgFloatSum", "pmgFloatFix", "pmgSetupF"]);
  assert.ok(!parent.some(l => l.startsWith("pmgFloat")));
  assert.deepEqual(parent, fine.filter(l => !l.startsWith("pmgFloat")));
  for (const p of poolMGPCG(new BrickLayout({ ...base, detail }), { walls: true }).setup.filter(p => p.kernel.label.startsWith("pmgFloat")))
    assert.ok(p.kernel.bindings.filter(b => b.kind !== "uniform").length <= MAX_STORAGE_BINDINGS);
  const s = brickDetailSrc(detail);
  assert.equal(s.words, s.px + detail.budget);
});
