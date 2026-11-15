// Lane P's pool MGPCG on the composite rows (h rows and native 4h rows, pressure-composite.wgsl.ts) against its
// Float64 reference (pressure-composite-model.ts), on the frozen mixed-tier state of parity/composite-fixture.ts:
// - the setup residual r0 = b - A x0 on both tiers (seams read across, natives over the gathered seams);
// - one V-cycle: Z on the h rows and on the native rows is the model's V-cycle of the same r0;
// - the converged X on both tiers is the Float64 PCG's on the final rows (the walls' outer loop changes them);
// - the walls' outer loop: at the solve's end every lattice wall face's state (FLAGS bit 8+a at its storage) is the
//   complementarity of its ghost max(x + lift, 0): attached exactly when X + lift >= 0, on both tiers;
// - the projection (projectC, then projectF, the engine's project) leaves on every liquid row, h and native,
//   the divergence the row's residual says (-h sum sign V U + h^2 rate = r, and 16x on a 4h row with its seam
//   faces' 16 h patches), writes the WARM bit and the h tiles' C X mean.
// GPU only.
import assert from "node:assert/strict";
import test from "node:test";
import { ARGS, BRICK_STRIDE, H, LIST, STATE } from "./brick-abi";
import type { BrickEngine } from "./brick-engine";
import { add, coarseTile, compositeFixture, COMPOSITE_N, keyCoord, scale, type V3 } from "./parity/composite-fixture";
import { CompositePressureModel, type CompositeRows } from "./pressure-composite-model";
import { P_FLAG, P_WALL_C, pressureCompositeKernels } from "./pressure-composite.wgsl";
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
  return { e, before, after, header, h: fx.h, fx, K };
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
    const { e, before, after, header, h } = await solveComposite(device, { slots: 40, tolerance: tol, walls: true }, false);
    const M = after.model, n = M.fine.keys.length;
    const b = after.vec("RHS"), x0 = before.vec("X");
    const want = M.solve(b, x0, 1e-7 * M.norm(b), 200);
    const X = after.vec("X");
    let errF = 0, errN = 0;
    for (let i = 0; i < n; i++) { const d = Math.abs(X[i]! - want.x[i]!); if (M.isNative(i)) errN = Math.max(errN, d); else errF = Math.max(errF, d); }
    const sc = maxAbs(want.x), hf = asF32(header);
    t.diagnostic(`rows ${n}; Float64 ${want.iterations} its; GPU ${header[MG_HEADER.ITERATIONS]} its, walls ${header[MG_HEADER.WALLS]} changed over ${header[MG_HEADER.DECIDES]} decisions, r0 ${hf[MG_HEADER.RMAX0]!.toExponential(2)} -> ${hf[MG_HEADER.RLAST]!.toExponential(2)} (tol ${(tol * h * h).toExponential(2)}); |X - X64|/|X| h ${(errF / sc).toExponential(2)} native ${(errN / sc).toExponential(2)}`);
    assert.equal(header[H.STATE]! & STATE.PRESSURE_CAP, 0, "converged inside the slots");
    assert.ok(hf[MG_HEADER.RLAST]! <= tol * h * h);
    assert.ok(errF / sc < 1e-3 && errN / sc < 1e-3, `GPU X against Float64: h ${errF / sc}, native ${errN / sc}`);
    e.destroy();
  });
});

test("composite project (GPU): every liquid row's projected divergence is its residual, on both tiers", GPU, async t => {
  await withDevice("composite project", async device => {
    const { e, after, h, fx, K } = await solveComposite(device, { slots: 40, tolerance: 1e-3, walls: true }, false);
    const L = e.layout, N = COMPOSITE_N;
    const pagesG1 = { indirect: ARGS.PER_LIST * LIST.C_ALL + ARGS.G1 }, liq = { indirect: ARGS.PER_LIST * LIST.F_LIQ + ARGS.G1 };
    const xBefore = new Float32Array((await e.read("pres", L.base("pres", "F", "X"), L.cap.f * BRICK_STRIDE)).buffer);
    const stateF = new Uint32Array(await e.read("phi", L.base("phi", "F", "FLAGS"), L.cap.f * BRICK_STRIDE));
    const stateC = new Uint32Array(await e.read("phi", L.base("phi", "C", "FLAGS"), L.cap.c * BRICK_STRIDE));
    await e.run([{ kernel: K.projectC, dispatch: pagesG1 }, { kernel: K.projectF, dispatch: liq }]);
    const header = await e.read("set", 0, 256);
    assert.equal(header[H.FAIL], 0, `FAIL ${header[H.FAIL]} info ${header[H.FAIL_INFO]?.toString(16)}`);
    const readF = async (pool: "vel" | "pres" | "phi", ch: string) => new Float32Array((await e.read(pool, L.base(pool, "F", ch), L.cap.f * BRICK_STRIDE)).buffer);
    const readC = async (pool: "vel" | "pres", ch: string) => new Float32Array((await e.read(pool, L.base(pool, "C", ch), L.cap.c * BRICK_STRIDE)).buffer);
    const UF = await Promise.all([0, 1, 2].map(a => readF("vel", `U${a}`))), UC = await Promise.all([0, 1, 2].map(a => readC("vel", `U${a}`)));
    const DIAG = await readF("pres", "DIAG"), XF = await readF("pres", "X"), XC = await readC("pres", "X");
    const flags = new Uint32Array(await e.read("phi", L.base("phi", "F", "FLAGS"), L.cap.f * BRICK_STRIDE));
    const inLattice = (c: V3) => c.every(v => v >= 0 && v < N);
    const tileH = (T: V3) => T.every(v => v >= 0 && 4 * v < N) && !coarseTile(T);
    const fAt = (c: V3) => { const s = fx.fSlot(c); return s === undefined ? undefined : BRICK_STRIDE * s + ((c[0] & 7) | ((c[1] & 7) << 3) | ((c[2] & 7) << 6)); };
    const cAt = (T: V3) => { if (T.some(v => v < 0)) return undefined; const cs = fx.pageSlot([T[0] >> 3, T[1] >> 3, T[2] >> 3] as V3); return cs === 0xffffffff ? undefined : BRICK_STRIDE * cs + ((T[0] & 7) | ((T[1] & 7) << 3) | ((T[2] & 7) << 6)); };
    const unit = (a: number): V3 => [a === 0 ? 1 : 0, a === 1 ? 1 : 0, a === 2 ? 1 : 0];
    const M = after.model, X = after.vec("X"), b = after.vec("RHS"), AX = M.apply(X);
    let worst = 0, worstN = 0, rows = 0, natives = 0, skipped = 0, scaleU = 0;
    for (let i = 0; i < M.fine.keys.length; i++) {
      const native = M.isNative(i), at = M.fine.coords[i]! as V3; // a cell, or a tile
      const r = b[i]! - AX[i]!;
      let div = 0, missing = false;
      for (let f = 0; f < 6; f++) {
        const a = f >> 1, sign = f & 1 ? 1 : -1, e1 = unit(a);
        if (!native) {
          const c = at as V3, cf = sign > 0 ? add(c, e1) : c, V = cf[a] === 0 || cf[a] === N ? 0.5 : 1;
          const w = fAt(cf); if (w === undefined) { missing = true; continue; }
          div += sign * V * UF[a]![w]!; scaleU = Math.max(scaleU, Math.abs(UF[a]![w]!));
        } else {
          const T = at as V3, Tn = add(T, scale(sign, e1)), Tf = sign > 0 ? Tn : T;
          if (tileH(Tn)) {
            const u = unit((a + 1) % 3), v = unit((a + 2) % 3); let sum = 0;
            for (let s = 0; s < 16; s++) {
              const cf = add(add(scale(4, Tf), scale(s & 3, u)), scale(s >> 2, v));
              const w = fAt(cf); if (w === undefined) { missing = true; continue; }
              sum += UF[a]![w]!;
            }
            div += sign * sum / 16;
          } else {
            const V = Tf[a] === 0 || 4 * Tf[a]! === N ? 0.5 : 1;
            const w = cAt(Tf); if (w === undefined) { missing = true; continue; }
            div += sign * V * UC[a]![w]!;
          }
        }
      }
      if (missing) { skipped++; continue; }
      const lhs = native ? -16 * h * div + 64 * h * h * fx.rate : -h * div + h * h * fx.rate;
      const err = Math.abs(lhs - r) / (native ? 16 : 1);
      if (native) { natives++; worstN = Math.max(worstN, err); } else { rows++; worst = Math.max(worst, err); }
    }
    // The WARM bit: every h-owned cell's, set exactly on the liquid rows; the h tiles' C X: their liquid rows' mean X.
    let warmBad = 0, meanErr = 0;
    for (const [key, slot] of fx.s.bricks) {
      const o = scale(8, keyCoord(key));
      for (let l = 0; l < 512; l++) {
        const c = add(o, [l & 7, (l >> 3) & 7, l >> 6]), w = BRICK_STRIDE * slot + l;
        const row = inLattice(c) && tileH([c[0] >> 2, c[1] >> 2, c[2] >> 2]) && DIAG[w]! > 0;
        if (((flags[w]! & P_FLAG.WARM) !== 0) !== row) warmBad++;
      }
    }
    for (let z = 0; z < N / 4; z++) for (let y = 0; y < N / 4; y++) for (let x = 0; x < N / 4; x++) {
      const T: V3 = [x, y, z]; if (!tileH(T)) continue;
      let sum = 0, n = 0;
      for (let m = 0; m < 64; m++) { const w = fAt(add(scale(4, T), [m & 3, (m >> 2) & 3, m >> 4]))!; if (DIAG[w]! > 0) { sum += xBefore[w]!; n++; } }
      meanErr = Math.max(meanErr, Math.abs(XC[cAt(T)!]! - (n ? sum / n : 0)));
    }
    // The release bits (lane E's released walls, the old umRelease) at every lattice wall face's storage: not held,
    // the projection's wall ghost <= 0 (attached: X + lift; separated or dry: 0), and -sign U dt > 1e-4 width h.
    const dt = 1 / 60, flagsC = new Uint32Array(await e.read("phi", L.base("phi", "C", "FLAGS"), L.cap.c * BRICK_STRIDE));
    const PF = await readF("pres", "PPHI"), USF = await Promise.all([0, 1, 2].map(a => readF("vel", `US${a}`)));
    const PC = await readC("pres", "PPHI"), DC = await readC("pres", "DIAG"), USC = await Promise.all([0, 1, 2].map(a => readC("vel", `US${a}`)));
    const rel = { faces: 0, released: 0, bad: 0 };
    const release = (phiOwner: number, liquid: boolean, separated: boolean, x: number, lift: number, sign: number, u: number, w: number) =>
      !(phiOwner < -w) && (liquid && !separated ? x + lift : 0) <= 0 && -sign * u * dt > 1e-4 * w * h;
    const check = (want: boolean, word: number, a: number) => { const got = (word & (P_FLAG.WALL0 << a)) !== 0; rel.faces++; if (got) rel.released++; if (got !== want) rel.bad++; };
    for (let z = 0; z <= N; z++) for (let y = 0; y <= N; y++) for (let x = 0; x <= N; x++) for (let a = 0; a < 3; a++) {
      const c: V3 = [x, y, z]; if (c[a] !== 0 && c[a] !== N) continue;
      if (c.some((v, b) => b !== a && v >= N)) continue;
      const low = c[a] === 0, m = low ? c : add(c, scale(-1, unit(a))), sign = low ? -1 : 1, M4: V3 = [m[0] >> 2, m[1] >> 2, m[2] >> 2];
      if (tileH(M4)) {
        const w = fAt(c), o = fAt(m); if (w === undefined || o === undefined) continue;
        const liquid = DIAG[o]! > 0 && PF[o]! < 0;
        check(release(PF[o]!, liquid, (stateF[w]! & (P_FLAG.WALL0 << a)) !== 0, xBefore[o]!, sign * USF[a]![w]! * h, sign, UF[a]![w]!, 1), flags[w]!, a);
      } else if (c[a] % 4 === 0 && c.every((v, b) => b === a || v % 4 === 0)) {
        const T: V3 = [c[0] >> 2, c[1] >> 2, c[2] >> 2], w = cAt(T), o = cAt(M4); if (w === undefined || o === undefined) continue;
        const liquid = DC[o]! > 0 && PC[o]! < 0, sep = !(PC[o]! < -4) && (stateC[w]! & (P_FLAG.WALL0 << a)) !== 0;
        check(release(PC[o]!, liquid, sep, XC[o]!, sign * 4 * h * USC[a]![w]!, sign, UC[a]![w]!, 4), flagsC[w]!, a);
      }
    }
    const tolU = h * Math.max(scaleU, 1);
    t.diagnostic(`h rows ${rows}, native rows ${natives}, skipped ${skipped}; |div - r| h ${(worst / tolU).toExponential(2)} native ${(worstN / tolU).toExponential(2)} (of h max|U| ${tolU.toExponential(2)}); WARM wrong ${warmBad}; h tile mean err ${meanErr.toExponential(2)}`);
    assert.ok(rows > 0 && natives > 0);
    assert.equal(skipped, 0, "every liquid row's faces are stored");
    assert.ok(worst / tolU < 1e-5 && worstN / tolU < 1e-5, `projected divergence against the residual: h ${worst / tolU}, native ${worstN / tolU}`);
    assert.equal(warmBad, 0);
    t.diagnostic(`wall release bits: ${rel.faces} faces, ${rel.released} released, ${rel.bad} wrong`);
    assert.ok(rel.faces > 0);
    assert.equal(rel.bad, 0, "release bits as umRelease");
    assert.ok(meanErr < 1e-6, `h tiles' C X mean ${meanErr}`);
    void XF;
    e.destroy();
  });
});

// Converged (40 slots), and capped (3 slots: the last slot decides at the final iterate).
for (const slots of [40, 3]) test(`composite MGPCG walls (GPU, ${slots} slots): at the end every wall face's state is its ghost's complementarity, on both tiers`, GPU, async t => {
  await withDevice("composite mgpcg walls", async device => {
    const { e, header, h, fx } = await solveComposite(device, { slots, tolerance: 1e-3, walls: true }, false);
    const L = e.layout, N = COMPOSITE_N, NT = N / 4;
    const f32 = async (pool: "vel" | "pres", tier: "F" | "C", ch: string) => new Float32Array((await e.read(pool, L.base(pool, tier, ch), (tier === "F" ? L.cap.f : L.cap.c) * BRICK_STRIDE)).buffer);
    const u32 = async (tier: "F" | "C") => new Uint32Array(await e.read("phi", L.base("phi", tier, "FLAGS"), (tier === "F" ? L.cap.f : L.cap.c) * BRICK_STRIDE));
    const F = { X: await f32("pres", "F", "X"), B: await f32("pres", "F", "B"), DIAG: await f32("pres", "F", "DIAG"), US: await Promise.all([0, 1, 2].map(a => f32("vel", "F", `US${a}`))), FL: await u32("F") };
    const C = { X: await f32("pres", "C", "X"), B: await f32("pres", "C", "B"), DIAG: await f32("pres", "C", "DIAG"), PPHI: await f32("pres", "C", "PPHI"), US: await Promise.all([0, 1, 2].map(a => f32("vel", "C", `US${a}`))), FL: await u32("C") };
    const loc = (v: V3) => (v[0] & 7) | ((v[1] & 7) << 3) | ((v[2] & 7) << 6);
    const fAt = (c: V3) => { const s = fx.fSlot(c); return s === undefined ? undefined : BRICK_STRIDE * s + loc(c); };
    const cAt = (T: V3) => { const cs = fx.pageSlot([T[0] >> 3, T[1] >> 3, T[2] >> 3] as V3); return cs === 0xffffffff ? undefined : BRICK_STRIDE * cs + loc(T); };
    const unit = (a: number): V3 => [a === 0 ? 1 : 0, a === 1 ? 1 : 0, a === 2 ? 1 : 0];
    const tally = { faces: 0, separated: 0, wrong: 0, changed: 0, missing: 0, worst: 0 };
    const tallyN = { ...tally };
    // One wall face: X its row's, lift its forced flux, the stored state at w, the build's from B.
    const face = (k: typeof tally, X: number, B: number, lift: number, word: number, a: number) => {
      const separated = (word & (P_FLAG.WALL0 << a)) !== 0, g = X + lift;
      k.faces++; if (separated) k.separated++;
      // The GPU decided in f32 at the same X: only a face within rounding of g = 0 may read the other way here.
      if (separated !== (g < 0)) { if (Math.abs(g) > 1e-6 * Math.max(1, Math.abs(X))) k.wrong++; k.worst = Math.max(k.worst, Math.abs(g)); }
      if (separated !== (B + lift < 0)) k.changed++;
    };
    for (const [key, slot] of fx.s.bricks) {
      const o = scale(8, keyCoord(key));
      for (let l = 0; l < 512; l++) {
        const at = BRICK_STRIDE * slot + l; if (!(F.DIAG[at]! > 0)) continue;
        const c = add(o, [l & 7, (l >> 3) & 7, l >> 6]);
        for (let a = 0; a < 3; a++) {
          const side = c[a] === 0 ? -1 : c[a] === N - 1 ? 1 : 0; if (!side) continue;
          const w = fAt(side > 0 ? add(c, unit(a)) : c); if (w === undefined) { tally.missing++; continue; }
          face(tally, F.X[at]!, F.B[at]!, side * F.US[a]![w]! * h, F.FL[w]!, a);
        }
      }
    }
    for (const cs of fx.pages) {
      const O = scale(8, keyCoord(fx.dir[L.dir.crec + 32 * cs + 29]!));
      for (let i = 0; i < 512; i++) {
        const at = BRICK_STRIDE * cs + i; if (!(C.DIAG[at]! > 0) || C.PPHI[at]! < -4) continue;
        const T = add(O, [i & 7, (i >> 3) & 7, i >> 6]);
        if (T.some(v => v >= NT) || !coarseTile(T)) continue;
        for (let a = 0; a < 3; a++) {
          const side = T[a] === 0 ? -1 : T[a] === NT - 1 ? 1 : 0; if (!side) continue;
          const w = cAt(side > 0 ? add(T, unit(a)) : T); if (w === undefined) { tallyN.missing++; continue; }
          face(tallyN, C.X[at]!, C.B[at]!, side * 4 * h * C.US[a]![w]!, C.FL[w]!, a);
        }
      }
    }
    t.diagnostic(`h wall faces ${tally.faces} (separated ${tally.separated}, changed from the build ${tally.changed}, wrong ${tally.wrong}, worst |X + lift| ${tally.worst.toExponential(2)}); native ${tallyN.faces} (separated ${tallyN.separated}, changed ${tallyN.changed}, wrong ${tallyN.wrong}); GPU ${header[MG_HEADER.ITERATIONS]} its, ${header[MG_HEADER.WALLS]} changes over ${header[MG_HEADER.DECIDES]} decisions; c ${P_WALL_C.F}/${P_WALL_C.C}`);
    if (slots === 40) assert.equal(header[H.STATE]! & STATE.PRESSURE_CAP, 0, "converged inside the slots");
    else assert.ok(header[MG_HEADER.DECIDES]! >= 1, "the capped solve decided at its last slot");
    assert.equal(tally.missing + tallyN.missing, 0, "every wall face's storage is resident");
    assert.ok(tally.faces > 0 && tallyN.faces > 0, "both tiers have wall faces");
    assert.equal(tally.wrong + tallyN.wrong, 0, `wall state against X + lift: h ${tally.wrong} native ${tallyN.wrong}`);
    e.destroy();
  });
});
