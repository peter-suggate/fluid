// Lane P's cut rows (M5 design §5; pressure-composite.wgsl.ts with opt.cuts) on the GPU, on the frozen mixed-tier
// state of parity/composite-fixture.ts with synthetic K slots: FRAC, AP and W written to a few all-h bricks around a
// shut ball. Against the cut-free rows of the same state:
// - with no K slot, the cut rows and projection equal the cut-free ones bit for bit;
// - with cuts, every row's DIAG, RHS and K are the cut-free row's moved by the old pRows' solid terms (capacity FRAC,
//   face weights AP, K_CLOSED at A <= 1e-6 or sealed, bPairTheta beside a shut cell, the moving wall sign (cap - A) W,
//   a lattice wall's coupling AP);
// - projectF gives a closed face U = W, leaves an unmoved face's U, and releases a solid face as bReleased.
// GPU only.
import assert from "node:assert/strict";
import test from "node:test";
import { ARGS, BRICK_CELLS, BRICK_STRIDE, CMASK_WORDS, H, LIST, NONE } from "./brick-abi";
import { COMPOSITE_N, add, coarseTile, compositeFixture, keyCoord, scale, shr, type V3 } from "./parity/composite-fixture";
import { P_FLAG, pressureCompositeKernels } from "./pressure-composite.wgsl";
import { mgLayout, poolMGPCG } from "./pressure-mgpcg";
import { NP_K } from "./pressure-rows.wgsl";

const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };
const N = COMPOSITE_N;
const unit = (a: number): V3 => [a === 0 ? 1 : 0, a === 1 ? 1 : 0, a === 2 ? 1 : 0];
const local = (c: V3) => (c[0] & 7) | ((c[1] & 7) << 3) | ((c[2] & 7) << 6);
const tileIndex = (t: V3) => (t[0] & 7) | ((t[1] & 7) << 3) | ((t[2] & 7) << 6);
const inLattice = (c: V3) => c.every(v => v >= 0 && v < N);
const tileInLattice = (t: V3) => t.every(v => v >= 0 && 4 * v < N);
const sgn = (f: number) => f & 1 ? 1 : -1;
const wet = (phi: number) => phi < -0.001;
const thetaOf = (liquid: number, air: number) => Math.min(Math.max(-liquid / Math.max(air - liquid, 1e-30), 0.001), 1000);
const hash = (c: V3, k: number) => { const s = Math.sin(12.9898 * c[0] + 78.233 * c[1] + 37.719 * c[2] + 4.581 * k) * 43758.5453; return s - Math.floor(s); };
const f32 = Math.fround;
const fAll = { indirect: ARGS.PER_LIST * LIST.F_ALL + ARGS.G1 }, cAll = { indirect: ARGS.PER_LIST * LIST.C_ALL + ARGS.G1 };

async function setup(device: GPUDevice) {
  const fx = await compositeFixture(device, L => poolMGPCG(L), { k: 8 });
  const { e, s, L, dir, pageSlot } = fx, M = mgLayout(L);
  const tileH = (t: V3) => { if (!tileInLattice(t)) return false; const cs = pageSlot(shr(t, 3)); if (cs === NONE) return false; const i = tileIndex(t); return ((dir[L.dir.cmask + CMASK_WORDS * cs + (i >> 5)]! >>> (i & 31)) & 1) === 1; };
  const cellH = (c: V3) => inLattice(c) && tileH(shr(c, 2));
  // Cut bricks: liquid lattice bricks (y brick 0 or 1; a skirt brick has no h tile) whose tiles and one-tile ring are all h (no seam meets a cut).
  const cut = new Map<number, number>();
  for (const [key] of [...s.bricks].sort((a, b) => a[0] - b[0])) {
    const b = keyCoord(key); if (b[1] > 1 || cut.size >= 6 || b.some(v => 8 * v + 8 > N)) continue;
    let ok = true;
    for (let z = -1; z <= 2; z++) for (let y = -1; y <= 2; y++) for (let x = -1; x <= 2; x++) { const t = add(scale(2, b), [x, y, z]); if (tileInLattice(t) && (coarseTile(t) || !tileH(t))) ok = false; }
    if (ok) cut.set(key, cut.size);
  }
  assert.ok(cut.size >= 3, `${cut.size} all-h bricks to cut`);
  // Shut balls (FRAC 0 inside, a ramp to 1), some partial cells, closed faces where AP < 0.05,
  // and a moving wall on every cut face.
  // The balls sit on the surface (y near 11) of each cut brick of y brick 1, so shut cells meet wet-dry faces.
  const balls = [...cut.keys()].map(keyCoord).filter(b => b[1] === 1).map(b => add(scale(8, b), [4, 2.8, 4]));
  assert.ok(balls.length > 0, "no cut brick holds the surface");
  const field = (c: V3) => {
    const d = Math.min(...balls.map(P => Math.hypot(c[0] + 0.5 - P[0], c[1] + 0.5 - P[1], c[2] + 0.5 - P[2])));
    return hash(c, 11) < 0.08 ? 0.3 : Math.min(Math.max((d - 2.2) / 1.5, 0), 1);
  };
  const K = L.cap.k, FRAC = new Float32Array(K * BRICK_STRIDE), AP = [0, 1, 2].map(() => new Float32Array(K * BRICK_STRIDE)), W = [0, 1, 2].map(() => new Float32Array(K * BRICK_STRIDE));
  for (const [key, k] of cut) {
    const o = scale(8, keyCoord(key));
    for (let l = 0; l < BRICK_CELLS; l++) {
      const c = add(o, [l & 7, (l >> 3) & 7, l >> 6]), at = BRICK_STRIDE * k + l;
      FRAC[at] = inLattice(c) ? field(c) : 0;
      for (let a = 0; a < 3; a++) {
        const low = add(c, scale(-1, unit(a)));
        let ap = 0.5 * ((inLattice(low) ? field(low) : 0) + FRAC[at]!);
        if (ap < 0.05) ap = 0;
        AP[a]![at] = ap; W[a]![at] = ap < 1 ? 0.3 * Math.sin(0.7 * c[0] + 1.3 * c[1] + 0.4 * c[2] + a) : 0;
      }
    }
  }
  const slotOf = new Map([...s.bricks]);
  const writeCuts = () => {
    e.write("cut", L.base("cut", "K", "FRAC"), FRAC);
    for (let a = 0; a < 3; a++) { e.write("cut", L.base("cut", "K", `AP${a}`), AP[a]!); e.write("cut", L.base("cut", "K", `W${a}`), W[a]!); }
    e.write("cut", L.cutBmask, new Uint32Array(K));
    for (const [key, k] of cut) e.write("dir", L.dir.frec + 32 * slotOf.get(key)! + 30, new Uint32Array([k]));
  };
  // The oracle's readers (pressure-composite.wgsl.ts CUTS_WGSL).
  const cutK = (c: V3) => { if (!inLattice(c)) return undefined; const key = (c[0] >> 3) | ((c[1] >> 3) << 10) | ((c[2] >> 3) << 20); return cut.get(key); };
  const fracOf = (c: V3) => { if (!inLattice(c)) return 0; const k = cutK(c); return k === undefined ? 1 : FRAC[BRICK_STRIDE * k + local(c)]!; };
  const apOf = (c: V3, a: number) => {
    const low = add(c, scale(-1, unit(a)));
    if (!inLattice(c)) return f32(0.5 * fracOf(low));
    const k = cutK(c); if (k === undefined) return inLattice(low) ? 1 : 0.5;
    return AP[a]![BRICK_STRIDE * k + local(c)]!;
  };
  const wOf = (c: V3, a: number) => { const k = cutK(c); return k === undefined ? 0 : W[a]![BRICK_STRIDE * k + local(c)]!; };
  return { ...fx, M, cut, cellH, writeCuts, fracOf, apOf, wOf, KF: pressureCompositeKernels(L, {}, { faces: M.FC }), KC: pressureCompositeKernels(L, {}, { faces: M.FC, cuts: true }) };
}

type Fx = Awaited<ReturnType<typeof setup>>;
async function runRows(fx: Fx, cuts: boolean) {
  const { e, L, M } = fx, K = cuts ? fx.KC : fx.KF;
  await e.run([{ kernel: K.tileRoots, dispatch: fAll }, { kernel: K.rowsF, dispatch: fAll }, { kernel: K.rowsC, dispatch: cAll }]);
  const rows = await e.save();
  await e.run([{ kernel: K.projectC, dispatch: cAll }, { kernel: K.projectF, dispatch: fAll }]);
  const header = await e.read("set", 0, 256);
  const f = (pool: "vel" | "pres", ch: string) => e.read(pool, L.base(pool, "F", ch), L.cap.f * BRICK_STRIDE).then(u => new Float32Array(u.buffer));
  const out = {
    fail: [header[H.FAIL]!, header[H.FAIL_INFO]!],
    X: await f("pres", "X"), RHS: await f("pres", "RHS"), DIAG: await f("pres", "DIAG"), PPHI: await f("pres", "PPHI"),
    K: [await f("pres", "K0"), await f("pres", "K1"), await f("pres", "K2")], KU: [await f("pres", "KU0"), await f("pres", "KU1"), await f("pres", "KU2")],
    U: [await f("vel", "U0"), await f("vel", "U1"), await f("vel", "U2")],
    flags: new Uint32Array(await e.read("phi", L.base("phi", "F", "FLAGS"), L.cap.f * BRICK_STRIDE)),
    fc: new Float32Array((await e.read("pmgF", M.FC, 7 * BRICK_STRIDE * L.cap.f)).buffer),
  };
  return { rows, out };
}

test("pressure cuts (GPU): with no K slot, the cut rows and projection are the cut-free ones bit for bit", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("pressure cuts", async device => {
    const fx = await setup(device), from = await fx.e.save();
    const a = (await runRows(fx, false)).out;
    await fx.e.restore(from);
    const b = (await runRows(fx, true)).out;
    assert.deepEqual(a.fail, [0, 0]); assert.deepEqual(b.fail, [0, 0]);
    const same = (x: ArrayLike<number>, y: ArrayLike<number>, what: string) => {
      const u = new Uint32Array(new Float32Array(x as Float32Array).buffer), v = new Uint32Array(new Float32Array(y as Float32Array).buffer);
      let d = 0; for (let i = 0; i < u.length; i++) if (u[i] !== v[i]) d++;
      assert.equal(d, 0, `${what}: ${d} words differ`);
    };
    for (const k of ["X", "RHS", "DIAG", "PPHI"] as const) same(a[k], b[k], k);
    for (let i = 0; i < 3; i++) { same(a.K[i]!, b.K[i]!, `K${i}`); same(a.KU[i]!, b.KU[i]!, `KU${i}`); same(a.U[i]!, b.U[i]!, `U${i}`); }
    assert.deepEqual([...b.flags], [...a.flags]);
    // The face cache: every word but a wall face's, which holds its coupling V (0.5) instead of its ratio (1).
    let walls = 0;
    for (let s = 0; s < fx.L.cap.f; s++) for (let l = 0; l < BRICK_CELLS; l++) {
      const kinds = a.fc[BRICK_STRIDE * (7 * s + 6) + l]!;
      assert.equal(b.fc[BRICK_STRIDE * (7 * s + 6) + l], kinds);
      for (let f = 0; f < 6; f++) {
        const at = BRICK_STRIDE * (7 * s + f) + l, kind = ((kinds >>> (4 * f)) & 15) - 1;
        if (kind === NP_K.WALL) { walls++; assert.equal(a.fc[at], 1); assert.equal(b.fc[at], 0.5); } else if (!Number.isNaN(a.fc[at])) assert.equal(b.fc[at], a.fc[at]);
      }
    }
    assert.ok(walls > 0, "no wall face cached");
  });
});

test("pressure cuts (GPU): the cut rows are the old pRows' solid terms on the cut-free rows; projectF closes, keeps and releases faces as bFace / bReleased", GPU, async t => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("pressure cuts", async device => {
    const fx = await setup(device), { e, s, L, h, rate, F, fSlot, cellH, fracOf, apOf, wOf } = fx, from = await e.save();
    const r0 = await runRows(fx, false), a = r0.out;
    await e.restore(from); fx.writeCuts();
    const r1 = await runRows(fx, true), b = r1.out;
    assert.deepEqual(a.fail, [0, 0]);
    assert.deepEqual(b.fail, [0, 0], `cut rows FAIL ${b.fail[0]} info ${b.fail[1]!.toString(16)}`);
    const at = (c: V3) => { const sl = fSlot(c); return sl === undefined ? undefined : BRICK_STRIDE * sl + local(c); };
    const pphi = (c: V3) => { const i = at(c); return i === undefined ? Infinity : a.PPHI[i]!; };
    const facePlane = (c: V3, ax: number) => { const hi: V3 = [N, N, N]; hi[ax]! += 1; if (c.some((v, i) => v < 0 || v >= hi[i]!)) return 0; const i = at(c); return i === undefined ? 0 : F.US[ax]![i]!; };
    const tol = (got: number, want: number, sc: number, what: string) => assert.ok(Math.abs(got - want) <= 2e-4 * (Math.abs(want) + sc), `${what}: ${got} against ${want}`);
    const st = { rows: 0, moved: 0, closed: 0, sealed: 0, pair: 0, walls: 0, moving: 0, uw: 0, kept: 0, released: 0, solid: 0 };
    // ---- Rows.
    for (const [key, slot] of s.bricks) {
      const o = scale(8, keyCoord(key));
      for (let l = 0; l < BRICK_CELLS; l++) {
        const c = add(o, [l & 7, (l >> 3) & 7, l >> 6]), i = BRICK_STRIDE * slot + l;
        const owned = cellH(c), own = owned ? a.PPHI[i]! : Infinity;
        if (!owned || !wet(own)) { assert.equal(b.DIAG[i], a.DIAG[i], `dry DIAG ${c}`); continue; }
        st.rows++;
        const kinds = a.fc[BRICK_STRIDE * (7 * slot + 6) + l]!, cap = fracOf(c), shut = cap <= 1e-5, x = a.X[i]!;
        let dDiag = 0, dDiv = 0, dExtra = 0, moved = cap !== 1; const kk = [a.K[0]![i]!, a.K[1]![i]!, a.K[2]![i]!];
        for (let f = 0; f < 6; f++) {
          const ax = f >> 1, sign = sgn(f), q = add(c, scale(sign, unit(ax))), fc = sign > 0 ? q : c, kind0 = ((kinds >>> (4 * f)) & 15) - 1;
          const us = facePlane(fc, ax), vol = apOf(fc, ax), ratio0 = a.fc[BRICK_STRIDE * (7 * slot + f) + l]!;
          const resident = kind0 !== NP_K.WALL && kind0 !== NP_K.OPEN, qOpen = resident ? fracOf(q) : 1;
          let coef0 = 0, frac0 = 1, kind1 = kind0, coef1 = 0, frac1 = 1;
          if (kind0 === NP_K.WALL) { coef0 = 0.5; frac0 = 0.5; coef1 = vol; frac1 = vol; }
          else if (kind0 === NP_K.OPEN) { coef0 = ratio0; coef1 = f32(ratio0 * vol); frac1 = vol; }
          else if (kind0 === NP_K.BAND) {
            const nPhi = pphi(q), nWet = wet(nPhi); coef0 = ratio0;
            let ratio = ratio0;
            if (nWet !== true && (shut || qOpen <= 1e-5)) { ratio = f32(1 / thetaOf(own, nPhi)); st.pair++; }
            coef1 = f32(ratio * vol); frac1 = vol;
            if (shut && !nWet) { kind1 = NP_K.CLOSED; st.sealed++; }
          } else {
            assert.ok(vol === 1 && !shut && qOpen > 1e-5, `a ${kind0} face meets a cut at ${c} f${f}`);
            continue;
          }
          if (vol <= 1e-6) { if (kind1 !== NP_K.CLOSED) st.closed++; kind1 = NP_K.CLOSED; }
          if (kind1 === NP_K.CLOSED) { coef1 = 0; frac1 = 0; }
          if (coef1 !== coef0 || frac1 !== frac0 || kind1 !== kind0) moved = true;
          const w = wOf(fc, ax);
          dDiv += (kind1 !== NP_K.CLOSED ? sign * frac1 * us : 0) - sign * frac0 * us;
          if (kind1 !== NP_K.WALL && kind1 !== NP_K.OPEN) { dDiv += sign * (cap - frac1) * w; if (w !== 0) { st.moving++; moved = true; } }
          if (kind0 === NP_K.WALL) {
            st.walls++;
            const lift = sign * us * h, attached = x + lift >= 0;
            if (attached) dExtra += (kind1 === NP_K.CLOSED ? 0 : coef1 * lift) - coef0 * lift; else dDiag += (kind1 === NP_K.CLOSED ? 0 : coef1) - coef0;
          } else dDiag += coef1 - coef0;
          if (kind0 === NP_K.BAND && sign < 0 && wet(pphi(q))) kk[ax] = kind1 === NP_K.CLOSED ? 0 : coef1;
        }
        if (moved) st.moved++;
        const sc = Math.abs(a.DIAG[i]!) + 1e-3;
        assert.equal(b.X[i], a.X[i], `X ${c}`);
        tol(b.DIAG[i]!, a.DIAG[i]! + dDiag, 1e-3, `DIAG ${c}`);
        tol(b.RHS[i]!, a.RHS[i]! - h * dDiv + h * h * rate * (cap - 1) + dExtra, 1e-3 * sc, `RHS ${c}`);
        for (let ax = 0; ax < 3; ax++) { tol(b.K[ax]![i]!, kk[ax]!, 1e-3, `K${ax} ${c}`); assert.equal(b.KU[ax]![i], a.KU[ax]![i], `KU${ax} ${c}`); }
      }
    }
    // ---- Projection: each brick cell's lower faces.
    const dt = 1 / 60, cfc = (c: V3) => { const sl = fSlot(c); return sl; };
    const kindAt = (out: typeof a, c: V3, f: number) => { const sl = cfc(c)!; return ((out.fc[BRICK_STRIDE * (7 * sl + 6) + local(c)]! >>> (4 * f)) & 15) - 1; };
    for (const [key, slot] of s.bricks) {
      const o = scale(8, keyCoord(key));
      for (let l = 0; l < BRICK_CELLS; l++) {
        const c = add(o, [l & 7, (l >> 3) & 7, l >> 6]), i = BRICK_STRIDE * slot + l;
        for (let ax = 0; ax < 3; ax++) {
          const hiP: V3 = [N, N, N]; hiP[ax]! += 1; if (c.some((v, j) => v < 0 || v >= hiP[j]!)) continue;
          const n = add(c, scale(-1, unit(ax))), hc = cellH(c), hn = cellH(n);
          if (!hc && !hn) continue;
          const pc = hc ? b.PPHI[i]! : Infinity, lc = hc && wet(pc), pn = hn ? b.PPHI[at(n)!]! : Infinity, ln = hn && wet(pn);
          const ap = apOf(c, ax), w = wOf(c, ax);
          let closed = ap <= 1e-6, ratioSame = true;
          if (lc) { closed = kindAt(b, c, 2 * ax) === NP_K.CLOSED; ratioSame = b.fc[BRICK_STRIDE * (7 * slot + 2 * ax) + l] === a.fc[BRICK_STRIDE * (7 * slot + 2 * ax) + l] || kindAt(b, c, 2 * ax) === NP_K.WALL; }
          else if (ln) { const sn = cfc(n)!; closed = kindAt(b, n, 2 * ax + 1) === NP_K.CLOSED; ratioSame = b.fc[BRICK_STRIDE * (7 * sn + 2 * ax + 1) + local(n)] === a.fc[BRICK_STRIDE * (7 * sn + 2 * ax + 1) + local(n)] || kindAt(b, n, 2 * ax + 1) === NP_K.WALL; }
          const u0 = a.U[ax]![i]!, u1 = b.U[ax]![i]!;
          if (closed) { assert.equal(u1, w, `closed U ${c} a${ax}`); st.uw++; }
          else if (ratioSame) { assert.equal(u1, u0, `kept U ${c} a${ax}`); st.kept++; }
          const bit = (b.flags[i]! & (P_FLAG.WALL0 << ax)) !== 0, bit0 = (a.flags[i]! & (P_FLAG.WALL0 << ax)) !== 0;
          const wall = c[ax] === 0 || c[ax] === N;
          if (wall) {
            const m = c[ax] === 0 ? c : n;
            if (!closed && ratioSame) assert.equal(bit, bit0 && fracOf(m) > 1e-5 && ap > 1e-6, `wall release ${c} a${ax}`);
          } else {
            let want = false;
            const openN = fracOf(n) > 1e-5;
            if (hn && openN !== (fracOf(c) > 1e-5) && ap > 1e-6) {
              st.solid++;
              const pressure = (openN ? lc : ln) ? b.X[at(openN ? c : n)!]! : 0;
              want = pressure <= 0 && (openN ? -1 : 1) * f32(u1 - w) * dt > 1e-4 * h;
            }
            assert.equal(bit, want, `solid release ${c} a${ax}`);
            if (bit) st.released++;
          }
        }
      }
    }
    t.diagnostic(`cut rows: ${JSON.stringify(st)}`);
    for (const k of ["moved", "closed", "sealed", "pair", "walls", "moving", "uw", "kept", "solid"] as const) assert.ok(st[k] > 0, `${k} not exercised`);
  });
});

test("pressure cuts (GPU): the composite MGPCG on the cut rows converges with every wall face's state its final complementarity", GPU, async t => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("pressure cuts", async device => {
    const fx = await setup(device), { e, L } = fx;
    fx.writeCuts();
    const lv = poolMGPCG(L, { slots: 40, tolerance: 1e-3, walls: true, wallDecisions: Infinity, audit: true, cuts: true }), live = lv.live!;
    assert.ok(live, "the live lists");
    const rowsC = pressureCompositeKernels(L).rowsC;
    await e.run([{ kernel: live.rowsList, dispatch: fAll }, ...live.args, ...live.tileRoots, ...live.rows, { kernel: rowsC, dispatch: cAll }, ...lv.core.setup]);
    // The rows' wall faces (rowsF's face cache) before any re-decision moves a state: the audit counts these.
    const fc = new Float32Array((await e.read("pmgF", fx.M.FC, 7 * BRICK_STRIDE * L.cap.f)).buffer);
    let walls = 0;
    for (const [, slot] of fx.s.bricks) for (let l = 0; l < BRICK_CELLS; l++) { const kinds = fc[BRICK_STRIDE * (7 * slot + 6) + l]!; for (let f = 0; f < 6; f++) if (((kinds >>> (4 * f)) & 15) - 1 === NP_K.WALL) walls++; }
    await e.run([...Array.from({ length: lv.slots }, (_, k) => lv.slot(k)).flat(), ...lv.core.finish]);
    const header = await e.read("set", 0, 256);
    const { MG_HEADER } = await import("./pressure-mgpcg");
    const audit = Array.from(header.slice(MG_HEADER.AUDIT, MG_HEADER.AUDIT + 4));
    t.diagnostic(`cut MGPCG: ${header[MG_HEADER.ITERATIONS]} its, ${header[MG_HEADER.FLIPS]} flips, audit ${audit.join(" ")}, ${walls} cached wall faces`);
    assert.equal(header[H.FAIL], 0, `FAIL info ${header[H.FAIL_INFO]?.toString(16)}`);
    assert.equal(header[H.STATE]! & 1, 0, "converged inside the slots");
    assert.equal(audit[0], 0, "inconsistent h wall faces");
    assert.ok(audit[1]! > 0, "h wall faces audited");
    e.destroy();
  });
});
