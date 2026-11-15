/**
 * Lane P's pool MGPCG (pressure-mgpcg.ts) against its Float64 oracle
 * (pressure-model.ts with the GPU's fixed 8 levels).
 *
 * CPU:
 * - the solver passes the slot contract's validation, binds at most 8
 *   storage buffers, and sizes its buffers from capacities, not the lattice.
 *
 * GPU (WEBGPU_NODE_MODULE):
 * - one V-cycle: with one slot and X0 = 0, Z is the preconditioner applied
 *   to RHS. It must match model.vcycle(RHS) on the same DIAG and K, on a
 *   corner block of liquid spanning several pages;
 * - hydrostatics: X matches the Float64 solve and the analytic profile, and
 *   X = 0 in the air;
 * - the corner block's converged X matches model.solve.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { BrickLayout, H, PHI_AIR, STATE, brickKeyCoord } from "./brick-abi";
import { BrickEngine, type BrickEngineOptions, type BrickLiquidBox, type Vec3 } from "./brick-engine";
import { bringUpStages } from "./brick-stages";
import { validatePressureSolver } from "./brick-pressure-slot";
import { MAX_STORAGE_BINDINGS } from "./brick-kernel";
import { brickField, pressureOracle } from "./brick-oracle";
import { snapshot, type BrickSnapshot } from "./brick-readback";
import { MG_HEADER, poolMGPCG, type MGPCGOptions } from "./pressure-mgpcg";
import { PressureModel, packCell } from "./pressure-model";

const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };
const H_M = 1 / 32, DT = 1 / 60, G = 9.81;
const asF32 = (u: Uint32Array) => new Float32Array(u.buffer, u.byteOffset, u.length);

function options(lattice: Vec3, liquid: BrickLiquidBox[], f: number, c: number, mg: MGPCGOptions = {}): BrickEngineOptions {
  return { lattice, h_m: H_M, dt_s: DT, gravity_mps2: [0, -G, 0], capacities: { f, c, b: 0, k: 0, particles: 1024 }, liquid, solver: L => poolMGPCG(L, mg), stages: bringUpStages };
}

async function withDevice(label: string, run: (device: GPUDevice) => Promise<void>) {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice(label, run);
}

test("pool MGPCG: slot contract, bindings, and buffers from capacities", () => {
  const caps = { f: 300, c: 16, b: 0, k: 0, particles: 1024 };
  const A = new BrickLayout({ lattice: [64, 32, 64], ...caps }), B = new BrickLayout({ lattice: [256, 32, 64], ...caps });
  const a = poolMGPCG(A), b = poolMGPCG(B);
  validatePressureSolver(a);
  assert.equal(a.bringUp, false);
  assert.deepEqual(b.buffers, a.buffers);
  for (const p of [...a.setup, ...a.slot(0), ...a.slot(a.slots - 1), ...a.finish])
    assert.ok(p.kernel.bindings.filter(x => x.kind !== "uniform").length <= MAX_STORAGE_BINDINGS, p.kernel.label);
  assert.throws(() => poolMGPCG(A, { slots: 0 }), /0 slots/);
});

/** The engine's pres channels at the liquid cells, as the model's level 0. */
async function readSystem(e: BrickEngine, s: BrickSnapshot, lattice: Vec3) {
  const L = e.layout, pres = asF32(await e.read("pres"));
  const ch = (n: string) => brickField(s, pres, L.base("pres", "F", n), 0);
  const DIAG = ch("DIAG"), K = [ch("K0"), ch("K1"), ch("K2")], RHS = ch("RHS"), X = ch("X"), Z = ch("Z");
  const all: Vec3[] = [];
  for (const key of s.bricks.keys()) { const b = brickKeyCoord(key); for (let l = 0; l < 512; l++) all.push([8 * b[0] + (l & 7), 8 * b[1] + ((l >> 3) & 7), 8 * b[2] + (l >> 6)]); }
  const rows = all.filter(c => c.every((v, a) => v < lattice[a]!) && DIAG(c) > 0);
  const keys = rows.map(c => packCell(...c));
  const fine = { keys, diag: rows.map(DIAG), k: [0, 1, 2].map(a => rows.map(K[a]!)) as [number[], number[], number[]] };
  return { all, rows, keys, fine, RHS, X, Z };
}

const maxAbs = (v: ArrayLike<number>) => { let m = 0; for (let i = 0; i < v.length; i++) m = Math.max(m, Math.abs(v[i]!)); return m; };

// A corner block of liquid: free surface on three sides, several pages wide.
const CORNER: BrickLiquidBox = { lo: [-8, -8, -8], hi: [40.3, 21.7, 35.6] };
const CORNER_LATTICE: Vec3 = [80, 32, 72];

// A wide thin sheet: more rows at the top levels (L6, L7).
const WIDE: BrickLiquidBox = { lo: [-8, -8, -8], hi: [200.5, 6.3, 140.2] };
const WIDE_LATTICE: Vec3 = [256, 32, 160];

for (const [name, lattice, box, f, c] of [["corner block", CORNER_LATTICE, CORNER, 600, 32], ["wide sheet", WIDE_LATTICE, WIDE, 2400, 96]] as const)
test(`pool MGPCG (GPU): one V-cycle matches the Float64 model's (${name})`, GPU, async t => {
  await withDevice("pool mgpcg vcycle", async device => {
    const e = await BrickEngine.create(device, options(lattice, [box], f, c, { slots: 1, tolerance: 0 }));
    e.advance(1);
    const s = await snapshot(e);
    await e.assertHealthy();
    const sys = await readSystem(e, s, lattice);
    const model = new PressureModel(sys.fine, { levels: 8 });
    // Model rows are sorted by key; map them back.
    const order = new Map(sys.keys.map((k, i) => [k, i]));
    const r = new Float64Array(model.fine.keys.length);
    model.fine.keys.forEach((k, i) => { r[i] = sys.RHS(sys.rows[order.get(k)!]!); });
    const z = model.vcycle(r);
    let err = 0;
    model.fine.keys.forEach((k, i) => { err = Math.max(err, Math.abs(sys.Z(sys.rows[order.get(k)!]!) - z[i]!)); });
    const scale = maxAbs(z), h = s.header;
    t.diagnostic(`rows ${model.fine.keys.length}, levels ${model.levels.map(l => l.keys.length).join("/")}; GPU N5/N6/N7 ${h[MG_HEADER.N5]}/${h[MG_HEADER.N6]}/${h[MG_HEADER.N7]}; |Z - Z64|/|Z| ${(err / scale).toExponential(2)}; its ${h[MG_HEADER.ITERATIONS]}`);
    assert.equal(h[MG_HEADER.N7], model.levels[7]!.keys.length, "L7 rows");
    assert.equal(h[MG_HEADER.N6], model.levels[6]!.keys.length, "L6 rows");
    assert.equal(h[MG_HEADER.N5], model.levels[5]!.keys.length, "L5 rows");
    assert.equal(h[MG_HEADER.ITERATIONS], 1);
    assert.ok(scale > 0);
    assert.ok(err / scale < 1e-5, `GPU V-cycle against Float64: ${err / scale}`);
    e.destroy();
  });
});

test("pool MGPCG (GPU): a static box is hydrostatic, and X = 0 in the air", GPU, async t => {
  await withDevice("pool mgpcg hydrostatic", async device => {
    const lattice: Vec3 = [32, 32, 32], top = 10.3;
    const e = await BrickEngine.create(device, options(lattice, [{ lo: [-8, -8, -8], hi: [40, top, 40] }], 200, 8));
    e.advance(1);
    const s = await snapshot(e), L = e.layout;
    await e.assertHealthy();
    const pres = asF32(await e.read("pres")), vel = asF32(await e.read("vel")), phiW = asF32(await e.read("phi"));
    const phi = brickField(s, phiW, L.base("phi", "F", "PHI"), PHI_AIR);
    const us = [0, 1, 2].map(a => brickField(s, vel, L.base("vel", "F", `US${a}`), 0));
    const X = brickField(s, pres, L.base("pres", "F", "X"), 0);
    const cells: Vec3[] = [];
    for (const key of s.bricks.keys()) { const b = brickKeyCoord(key); for (let l = 0; l < 512; l++) cells.push([8 * b[0] + (l & 7), 8 * b[1] + ((l >> 3) & 7), 8 * b[2] + (l >> 6)]); }
    const oracle = pressureOracle(lattice, H_M, cells, phi, (a, c) => us[a]!(c));
    let scale = 0, err = 0, analytic = 0, air = 0;
    for (const c of cells) {
      if (c.some((v, a) => v >= lattice[a]!)) continue;
      const want = oracle.X.get(c.join(","));
      if (want === undefined) { air = Math.max(air, Math.abs(X(c))); continue; }
      const exact = G * DT * (top - (c[1] + 0.5)) * H_M;
      scale = Math.max(scale, Math.abs(want)); err = Math.max(err, Math.abs(X(c) - want));
      analytic = Math.max(analytic, Math.abs(X(c) - exact) / exact);
    }
    const h = s.header;
    t.diagnostic(`MGPCG ${h[MG_HEADER.ITERATIONS]} its (r0 ${asF32(h)[MG_HEADER.RMAX0]!.toExponential(2)}, last ${asF32(h)[MG_HEADER.RLAST]!.toExponential(2)}); |X - X64|/|X| ${(err / scale).toExponential(2)}; analytic ${analytic.toExponential(2)}; air |X| ${air}`);
    assert.equal(h[H.STATE]! & STATE.PRESSURE_CAP, 0, "converged inside the slots");
    assert.ok(err / scale < 1e-3, `GPU X against Float64: ${err / scale}`);
    assert.ok(analytic < 1e-3, `GPU X against the analytic profile: ${analytic}`);
    assert.equal(air, 0, "X is 0 off the liquid");
    e.destroy();
  });
});

test("pool MGPCG (GPU): the corner block converges to the Float64 MGPCG's X", GPU, async t => {
  await withDevice("pool mgpcg corner", async device => {
    const tol = 1e-3;
    const e = await BrickEngine.create(device, options(CORNER_LATTICE, [CORNER], 600, 32, { tolerance: tol }));
    e.advance(1);
    const s = await snapshot(e);
    await e.assertHealthy();
    const sys = await readSystem(e, s, CORNER_LATTICE);
    const model = new PressureModel(sys.fine, { levels: 8 });
    const order = new Map(sys.keys.map((k, i) => [k, i]));
    const cell = (k: number) => sys.rows[order.get(k)!]!;
    const b = Float64Array.from(model.fine.keys.map(k => sys.RHS(cell(k))));
    const want = model.solve(b, new Float64Array(b.length), 1e-6 * maxAbs(b), 100);
    let err = 0;
    model.fine.keys.forEach((k, i) => { err = Math.max(err, Math.abs(sys.X(cell(k)) - want.x[i]!)); });
    let air = 0;
    for (const c of sys.all) if (c.every((v, a) => v < CORNER_LATTICE[a]!) && !order.has(packCell(...c))) air = Math.max(air, Math.abs(sys.X(c)));
    const scale = maxAbs(want.x), h = s.header, hf = asF32(h);
    t.diagnostic(`rows ${b.length}; Float64 ${want.iterations} its to 1e-6; GPU ${h[MG_HEADER.ITERATIONS]} its, r0 ${hf[MG_HEADER.RMAX0]!.toExponential(2)} -> ${hf[MG_HEADER.RLAST]!.toExponential(2)} (tol ${(tol * H_M * H_M).toExponential(2)}); |X - X64|/|X| ${(err / scale).toExponential(2)}; air |X| ${air}`);
    assert.equal(h[H.FAIL], 0);
    assert.equal(h[H.STATE]! & STATE.PRESSURE_CAP, 0, "converged inside the slots");
    assert.ok(hf[MG_HEADER.RLAST]! <= tol * H_M * H_M);
    assert.ok(err / scale < 1e-3, `GPU X against Float64: ${err / scale}`);
    assert.equal(air, 0, "X is 0 off the liquid");
    e.destroy();
  });
});
