// Lane P's pool MGPCG on the composite rows (h rows and native 4h rows, pressure-composite.wgsl.ts) against its
// Float64 reference (pressure-composite-model.ts), on the frozen mixed-tier state of parity/composite-fixture.ts:
// - the setup residual r0 = b - A x0 on both tiers (seams read across, natives over the gathered seams);
// - one V-cycle: Z on the h rows and on the native rows is the model's V-cycle of the same r0;
// - the converged X on both tiers is the Float64 PCG's.
// GPU only.
import assert from "node:assert/strict";
import test from "node:test";
import { ARGS, BRICK_STRIDE, H, LIST, STATE } from "./brick-abi";
import type { BrickEngine } from "./brick-engine";
import { add, compositeFixture, keyCoord, scale, type V3 } from "./parity/composite-fixture";
import { CompositePressureModel, type CompositeRows } from "./pressure-composite-model";
import { pressureCompositeKernels } from "./pressure-composite.wgsl";
import { MG_HEADER, poolMGPCG, type MGPCGOptions } from "./pressure-mgpcg";

const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };
const asF32 = (u: Uint32Array) => new Float32Array(u.buffer, u.byteOffset, u.length);
const maxAbs = (v: ArrayLike<number>) => { let m = 0; for (let i = 0; i < v.length; i++) m = Math.max(m, Math.abs(v[i]!)); return m; };
const SOLVE = ["pmgDirection", "pmgDirectionC", "pmgUpdate", "pmgUpdateC", "pmgCheck"];

async function withDevice(label: string, run: (device: GPUDevice) => Promise<void>) {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice(label, run);
}

/** The composite system as the GPU stored it, with each row's channel reader. */
async function readComposite(e: BrickEngine, bricks: Map<number, number>, pages: number[], dir: Uint32Array) {
  const L = e.layout;
  const readF = async (ch: string) => new Float32Array((await e.read("pres", L.base("pres", "F", ch), L.cap.f * BRICK_STRIDE)).buffer);
  const readC = async (ch: string) => new Float32Array((await e.read("pres", L.base("pres", "C", ch), L.cap.c * BRICK_STRIDE)).buffer);
  const F: Record<string, Float32Array> = {}, C: Record<string, Float32Array> = {};
  for (const ch of ["DIAG", "K0", "K1", "K2", "KU0", "KU1", "KU2", "RHS", "X", "R", "Z"]) F[ch] = await readF(ch);
  for (const ch of ["DIAG", "K0", "K1", "K2", "RHS", "X", "R", "Z"]) C[ch] = await readC(ch);
  const cells: V3[] = [], fAt: number[] = [], tiles: V3[] = [], cAt: number[] = [];
  for (const [key, slot] of bricks) {
    const o = scale(8, keyCoord(key));
    for (let l = 0; l < 512; l++) { const at = BRICK_STRIDE * slot + l; if (F.DIAG![at]! > 0) { cells.push(add(o, [l & 7, (l >> 3) & 7, l >> 6])); fAt.push(at); } }
  }
  for (const cs of pages) {
    const O = scale(8, keyCoord(dir[L.dir.crec + 32 * cs + 29]!));
    for (let i = 0; i < 512; i++) { const at = BRICK_STRIDE * cs + i; if (C.DIAG![at]! > 0) { tiles.push(add(O, [i & 7, (i >> 3) & 7, i >> 6])); cAt.push(at); } }
  }
  const rows: CompositeRows = {
    f: { cells, diag: fAt.map(i => F.DIAG![i]!), k: [0, 1, 2].map(a => fAt.map(i => F[`K${a}`]![i]!)), ku: [0, 1, 2].map(a => fAt.map(i => F[`KU${a}`]![i]!)) },
    n: { tiles, diag: cAt.map(i => C.DIAG![i]!), k: [0, 1, 2].map(a => cAt.map(i => C[`K${a}`]![i]!)) },
  };
  const model = new CompositePressureModel(rows);
  // A channel as the model's vector (rows in the model's order).
  const vec = (ch: string) => {
    const v = new Float64Array(model.fine.keys.length);
    cells.forEach((c, j) => { v[model.rowOf(c, false)!] = F[ch]![fAt[j]!]!; });
    tiles.forEach((t, j) => { v[model.rowOf(t, true)!] = C[ch]![cAt[j]!]!; });
    return v;
  };
  return { model, rows, vec };
}

async function solveComposite(device: GPUDevice, mg: MGPCGOptions, vcycleOnly: boolean) {
  const fx = await compositeFixture(device, L => poolMGPCG(L, mg));
  const { e } = fx, solver = e.solver;
  const K = pressureCompositeKernels(e.layout);
  await e.run([{ kernel: K.rowsF, dispatch: { indirect: ARGS.PER_LIST * LIST.F_ALL + ARGS.G1 } }, { kernel: K.rowsC, dispatch: { indirect: ARGS.PER_LIST * LIST.C_ALL + ARGS.G1 } }]);
  const before = await readComposite(e, fx.s.bricks, fx.pages, fx.dir);
  const run = vcycleOnly ? [...solver.setup, ...solver.slot(0).filter(p => !SOLVE.includes(p.kernel.label))]
    : [...solver.setup, ...Array.from({ length: solver.slots }, (_, k) => solver.slot(k)).flat(), ...solver.finish];
  await e.run(run);
  const header = await e.read("set", 0, 256);
  assert.equal(header[H.FAIL], 0, `FAIL ${header[H.FAIL]} info ${header[H.FAIL_INFO]?.toString(16)}`);
  const after = await readComposite(e, fx.s.bricks, fx.pages, fx.dir);
  return { e, before, after, header, h: fx.h };
}

test("composite MGPCG (GPU): r0 and one V-cycle on both tiers match the Float64 model's", GPU, async t => {
  await withDevice("composite mgpcg vcycle", async device => {
    const { e, before, after, header } = await solveComposite(device, { slots: 1, tolerance: 0 }, true);
    const M = before.model, n = M.fine.keys.length;
    const b = before.vec("RHS"), x0 = before.vec("X"), Ax = M.apply(x0), r0 = b.map((v, i) => v - Ax[i]!);
    const R = after.vec("R"), Z = after.vec("Z");
    const z = M.vcycle(R);
    let rErr = 0, zErrF = 0, zErrN = 0, nNative = 0;
    for (let i = 0; i < n; i++) {
      rErr = Math.max(rErr, Math.abs(R[i]! - r0[i]!) / (M.isNative(i) ? 64 : 1));
      if (M.isNative(i)) { nNative++; zErrN = Math.max(zErrN, Math.abs(Z[i]! - z[i]!)); } else zErrF = Math.max(zErrF, Math.abs(Z[i]! - z[i]!));
    }
    const rScale = M.norm(r0), zScale = maxAbs(z);
    t.diagnostic(`rows ${n} (${nNative} native), levels ${M.levels.map(l => l.keys.length).join("/")}; GPU N5/N6/N7 ${header[MG_HEADER.N5]}/${header[MG_HEADER.N6]}/${header[MG_HEADER.N7]}; |R - r0|/|r0| ${(rErr / rScale).toExponential(2)}; |Z - Z64|/|Z| h ${(zErrF / zScale).toExponential(2)} native ${(zErrN / zScale).toExponential(2)}`);
    assert.ok(nNative > 0 && nNative < n, "both tiers have rows");
    assert.equal(header[MG_HEADER.N7], M.levels[7]!.keys.length, "L7 rows");
    assert.equal(header[MG_HEADER.N5], M.levels[5]!.keys.length, "L5 rows");
    assert.ok(rErr / rScale < 1e-5, `GPU r0 against Float64: ${rErr / rScale}`);
    assert.ok(zScale > 0);
    assert.ok(zErrF / zScale < 1e-5 && zErrN / zScale < 1e-5, `GPU V-cycle against Float64: h ${zErrF / zScale}, native ${zErrN / zScale}`);
    e.destroy();
  });
});

test("composite MGPCG (GPU): the converged X on both tiers is the Float64 PCG's", GPU, async t => {
  await withDevice("composite mgpcg solve", async device => {
    const tol = 1e-3;
    const { e, before, after, header, h } = await solveComposite(device, { slots: 40, tolerance: tol }, false);
    const M = before.model, n = M.fine.keys.length;
    const b = before.vec("RHS"), x0 = before.vec("X");
    const want = M.solve(b, x0, 1e-7 * M.norm(b), 200);
    const X = after.vec("X");
    let errF = 0, errN = 0;
    for (let i = 0; i < n; i++) { const d = Math.abs(X[i]! - want.x[i]!); if (M.isNative(i)) errN = Math.max(errN, d); else errF = Math.max(errF, d); }
    const sc = maxAbs(want.x), hf = asF32(header);
    t.diagnostic(`rows ${n}; Float64 ${want.iterations} its; GPU ${header[MG_HEADER.ITERATIONS]} its, r0 ${hf[MG_HEADER.RMAX0]!.toExponential(2)} -> ${hf[MG_HEADER.RLAST]!.toExponential(2)} (tol ${(tol * h * h).toExponential(2)}); |X - X64|/|X| h ${(errF / sc).toExponential(2)} native ${(errN / sc).toExponential(2)}`);
    assert.equal(header[H.STATE]! & STATE.PRESSURE_CAP, 0, "converged inside the slots");
    assert.ok(hf[MG_HEADER.RLAST]! <= tol * h * h);
    assert.ok(errF / sc < 1e-3 && errN / sc < 1e-3, `GPU X against Float64: h ${errF / sc}, native ${errN / sc}`);
    e.destroy();
  });
});
