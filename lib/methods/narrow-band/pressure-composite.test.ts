// Lane P's composite rows (pressure-composite.wgsl.ts) on the GPU against a Float64 oracle of the same rules
// (the old arena's pRows, pInit, pRebase, pNative and buildRhs, in the slot contract's units), on the frozen
// mixed-tier state of parity/composite-fixture.ts. GPU only.
import assert from "node:assert/strict";
import test from "node:test";
import { ARGS, BRICK_STRIDE, CMASK_WORDS, H, LIST, NONE, PHI_AIR, phiFar } from "./brick-abi";
import { bringUpCG } from "./brick-bringup-cg";
import { COMPOSITE_N, add, compositeFixture, keyCoord, scale, shr, type V3 } from "./parity/composite-fixture";
import { P_FLAG, pressureCompositeKernels } from "./pressure-composite.wgsl";

const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };
const N = COMPOSITE_N;
/** An absent vertex's φ (NB_PHI_FAR: the lattice's largest extent in h, the old vertex field's empty value). */
const FAR = phiFar([N, N, N]);
const unit = (a: number): V3 => [a === 0 ? 1 : 0, a === 1 ? 1 : 0, a === 2 ? 1 : 0];
const corner = (j: number): V3 => [j & 1, (j >> 1) & 1, j >> 2];
const tileIndex = (t: V3) => (t[0] & 7) | ((t[1] & 7) << 3) | ((t[2] & 7) << 6);
const local = (c: V3) => (c[0] & 7) | ((c[1] & 7) << 3) | ((c[2] & 7) << 6);
const inLattice = (c: V3) => c.every(v => v >= 0 && v < N);
const tileInLattice = (t: V3) => t.every(v => v >= 0 && 4 * v < N);

test("pressure composite rows (GPU): band, seam, native, wall and start rows as the oracle's", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("pressure composite", async device => {
    const { e, s, L, h, rate, dir, pages, F, C, fFlags, pageSlot, fSlot, enteredX } = await compositeFixture(device, L => bringUpCG(L));
    for (const cs of pages) for (let i = 0; i < 512; i++) assert.equal(enteredX[BRICK_STRIDE * cs + i], 0, `C X at entry, slot ${cs}`);
    // ---- Run the rows.
    const K = pressureCompositeKernels(L);
    await e.run([{ kernel: K.tileRoots, dispatch: { indirect: ARGS.PER_LIST * LIST.F_ALL + ARGS.G1 } }, { kernel: K.rowsF, dispatch: { indirect: ARGS.PER_LIST * LIST.F_ALL + ARGS.G1 } }, { kernel: K.rowsC, dispatch: { indirect: ARGS.PER_LIST * LIST.C_ALL + ARGS.G1 } }]);
    const header = await e.read("set", 0, 256);
    assert.equal(header[H.FAIL], 0, `FAIL ${header[H.FAIL]} info ${header[H.FAIL_INFO]?.toString(16)}`);
    const readF = async (ch: string) => new Float32Array((await e.read("pres", L.base("pres", "F", ch), L.cap.f * BRICK_STRIDE)).buffer);
    const readC = async (ch: string) => new Float32Array((await e.read("pres", L.base("pres", "C", ch), L.cap.c * BRICK_STRIDE)).buffer);

    // ---- The oracle (Float64, on the stored f32 inputs).
    const tileH = (t: V3) => { if (!tileInLattice(t)) return false; const cs = pageSlot(shr(t, 3)); if (cs === NONE) return false; const i = tileIndex(t); return ((dir[L.dir.cmask + CMASK_WORDS * cs + (i >> 5)]! >>> (i & 31)) & 1) === 1; };
    const cellH = (c: V3) => tileH(shr(c, 2));
    const cVertex = (T: V3) => { if (T.some(v => v < 0 || 4 * v > N)) return FAR; const cs = pageSlot(shr(T, 3)); return cs === NONE ? FAR : C.PHI[BRICK_STRIDE * cs + tileIndex(T)]!; };
    const fVertex = (v: V3) => { const sl = fSlot(v); return sl === undefined ? FAR : F.PHI[BRICK_STRIDE * sl + local(v)]!; };
    const vertexCoarse = (v: V3): V3 | undefined => {
      const tA = shr(v, 2), inside = v.map((x, a) => x - 4 * tA[a]! !== 0);
      for (let k = 0; k < 8; k++) { const cc = corner(k); if (cc.some((c, a) => inside[a] && c === 0)) continue; const t = add(add(tA, cc), [-1, -1, -1]); if (tileInLattice(t) && pageSlot(shr(t, 3)) !== NONE && !tileH(t)) return t; } // a non-resident page's tile is no authority
      return undefined;
    };
    const vertexPhi = (v: V3) => {
      if (v.some(x => x < 0 || x > N)) return FAR;
      const T = vertexCoarse(v); if (!T) return fVertex(v);
      const t = v.map((x, a) => (x - 4 * T[a]!) / 4); let sum = 0;
      for (let k = 0; k < 8; k++) { const cc = corner(k), w = cc.map((c, a) => c ? t[a]! : 1 - t[a]!); sum += w[0]! * w[1]! * w[2]! * cVertex(add(T, cc)); }
      return sum;
    };
    const fFace = (c: V3, a: number) => { const sl = fSlot(c); return sl === undefined ? 0 : F.US[a]![BRICK_STRIDE * sl + local(c)]!; };
    const cFace = (T: V3, a: number) => { if (T.some(v => v < 0 || 4 * v > N)) return 0; const cs = pageSlot(shr(T, 3)); return cs === NONE ? 0 : C.US[a]![BRICK_STRIDE * cs + tileIndex(T)]!; };
    const facePlane = (c: V3, a: number) => {
      const hi: V3 = [N, N, N]; hi[a]! += 1; if (c.some((v, i) => v < 0 || v >= hi[i]!)) return 0;
      return cellH(c) || cellH(add(c, scale(-1, unit(a)))) ? fFace(c, a) : cFace(shr(c, 2), a);
    };
    const wet = (phi: number, width: number) => phi < -0.001 * width;
    const thetaOf = (liquid: number, air: number) => Math.min(Math.max(-liquid / Math.max(air - liquid, 1e-30), 0.001), 1000);
    const centre = (c: V3) => { let sm = 0; for (let j = 0; j < 8; j++) sm += vertexPhi(add(c, corner(j))); return sm / 8; };
    const facePhi = (c: V3, a: number, sign: number) => { const o = sign > 0 ? add(c, unit(a)) : c, u = unit((a + 1) % 3), v = unit((a + 2) % 3); return (vertexPhi(o) + vertexPhi(add(o, u)) + vertexPhi(add(o, v)) + vertexPhi(add(add(o, u), v))) / 4; };
    const faceTheta = (face: number, a: number, wa: boolean, b: number, wb: boolean) => {
      if (wa === wb) return 1; const liquid = wa ? a : b, air = wa ? b : a;
      if (face > 0) return Math.min(Math.max(0.5 * liquid / (liquid - face), 0.001), 1000);
      return Math.min(Math.max(0.5 + 0.5 * face / Math.min(face - air, -1e-30), 0.001), 1000);
    };
    const rootPhi4 = (T: V3) => { let sm = 0; for (let j = 0; j < 8; j++) sm += cVertex(add(T, corner(j))) * 0.125; return sm; };
    const rootPhi = (T: V3) => {
      if (!tileInLattice(T) || pageSlot(shr(T, 3)) === NONE) return 16;
      if (!tileH(T)) return rootPhi4(T);
      let total = 0;
      for (let z = 0; z <= 4; z++) for (let y = 0; y <= 4; y++) for (let x = 0; x <= 4; x++) { const g: V3 = [x, y, z], w = g.map(q => q === 0 || q === 4 ? 0.5 : 1); total += w[0]! * w[1]! * w[2]! * vertexPhi(add(scale(4, T), g)); }
      return total / 64;
    };
    const flagsF = (c: V3) => { const sl = fSlot(c); return sl === undefined ? 0 : fFlags[BRICK_STRIDE * sl + local(c)]!; };
    const xF = (c: V3) => { const sl = fSlot(c); return sl === undefined ? 0 : F.X[BRICK_STRIDE * sl + local(c)]!; };
    const tileStart = (T: V3) => {
      if (!tileInLattice(T)) return 0; const cs = pageSlot(shr(T, 3)); if (cs === NONE) return 0;
      if (!wet(rootPhi(T), 4)) return 0;
      return C.X[BRICK_STRIDE * cs + tileIndex(T)]!;
    };
    const pageHeld = (p: V3) => p.some(v => v < 0 || 32 * v >= N);
    const tileNear = (page: V3, t: V3): V3 => {
      const low = scale(8, page), high = add(low, [7, 7, 7]), d = t.map((v, a) => (v > high[a]! ? 1 : 0) - (v < low[a]! ? 1 : 0));
      if (d.every(x => x === 0)) return t;
      return t.map((v, a) => d[a] !== 0 && pageHeld(add(page, scale(d[a]!, unit(a)))) ? Math.min(Math.max(v, low[a]!), high[a]!) : v) as V3;
    };
    const coarseOf = (t: V3): V3 => { const p = rootPhi(t); return wet(p, 4) && tileInLattice(t) ? [tileStart(t), p, 1] : [0, 0, 0]; };
    const coarseKind = (t: V3): V3 => { const open = tileInLattice(t), w = wet(rootPhi(t), 4); return [open && w ? tileStart(t) : 0, open && w ? 1 : 0, open && !w ? 1 : 0]; };
    let carries = 0;
    const coarseCarry = (page: V3, b: V3, w: V3) => {
      const carry = { rise: [0, 0], height: [0, 0], air: 0 };
      for (let k = 0; k < 4; k++) {
        const bit = [k & 1, k >> 1], sides = [bit[0] ? w[0] : 1 - w[0], bit[1] ? w[2] : 1 - w[2]], side = sides[0]! * sides[1]!;
        const c = add(b, [bit[0]!, 0, bit[1]!]);
        const t0 = tileNear(page, c), t1 = tileNear(page, add(c, [0, 1, 0])), y0 = t0[1], y1 = t1[1];
        const low = coarseKind(t0); let high = low; if (y1 !== y0) high = coarseKind(t1);
        const lower = side * (1 - w[1]), upper = side * w[1], liquid = lower * low[1] + upper * high[1];
        carry.air += lower * low[2] + upper * high[2];
        carry.height[0]! += lower * low[1] * (y0 - b[1]) + upper * high[1] * (y1 - b[1]); carry.height[1]! += liquid;
        if (y1 !== y0 && low[1] > 0 && high[1] > 0) { carry.rise[0]! += liquid * (high[0] - low[0]); carry.rise[1]! += liquid; continue; }
        let own = low, at = y0, beyond = y0 !== c[1] ? y0 + 1 : y0 - 1;
        if (y1 !== y0) { if (high[1] > 0) { own = high; at = y1; beyond = y1 + 1; } else beyond = y0 - 1; }
        if (own[1] === 0) continue;
        const t2 = tileNear(page, [c[0], beyond, c[2]]); if (t2[1] !== beyond) continue;
        const other = coarseKind(t2);
        if (other[1] > 0) { carry.rise[0]! += liquid * (other[0] - own[0]) * (beyond - at); carry.rise[1]! += liquid; }
      }
      return carry;
    };
    let cold = 0;
    const start = (c: V3, own: number) => {
      if (!wet(own, 1)) return 0;
      if (flagsF(c) & P_FLAG.WARM) return xF(c);
      cold++;
      const y = c.map(v => (v + 0.5) / 4 - 0.5), b = y.map(Math.floor) as V3, w = y.map((v, a) => v - b[a]!) as V3; const sum = [0, 0, 0];
      const page = shr(c, 5);
      for (let k = 0; k < 8; k++) { const bit = corner(k), ww = bit.map((q, a) => q ? w[a]! : 1 - w[a]!), wt = ww[0]! * ww[1]! * ww[2]!, co = coarseOf(tileNear(page, add(b, bit))); for (let i = 0; i < 3; i++) sum[i]! += wt * co[i]!; }
      if (sum[2]! >= 0.99999) return sum[0]! / sum[2]!;
      if (sum[2]! > 0) {
        carries++;
        const carry = coarseCarry(page, b, w);
        if (carry.air > 0 || carry.rise[1] === 0) return sum[0]! / sum[2]! * Math.min(Math.max(own * sum[2]! / Math.min(sum[1]!, -1e-30), 0), 1000);
        const mean = sum[0]! / sum[2]!, bound = 1000 * mean;
        return Math.min(Math.max(mean + (w[1] - carry.height[0]! / carry.height[1]!) * carry.rise[0]! / carry.rise[1]!, Math.min(bound, 0)), Math.max(bound, 0));
      }
      return 0;
    };
    const nativeTheta = (t: V3, n: V3, a: number) => {
      const pa = rootPhi(t), pb = rootPhi(n), la = wet(pa, 4);
      if (la === wet(pb, 4)) return 1;
      const q = add(scale(4, la ? t : n), [2, 2, 2]); let previous = vertexPhi(q); const direction = la ? 1 : -1;
      if (previous < 0) for (let j = 1; j <= 4; j++) { q[a]! += direction; const next = vertexPhi(q); if (next >= 0) return Math.min(Math.max((j - 1 + previous / (previous - next)) / 4, 0.001), 1000); previous = next; }
      return thetaOf(la ? pa : pb, la ? pb : pa);
    };
    const KIND = { GHOST: 0, BAND: 1, WALL: 2, OPEN: 3, NEUMANN: 4, CLOSED: 5, SEAM: 6 };
    const bandFace = (c: V3, own: number, liquid: boolean, f: number) => {
      const a = f >> 1, sign = f & 1 ? 1 : -1, q = add(c, scale(sign, unit(a)));
      let across = 0; if (!inLattice(q)) across = 2; else if (pageSlot(shr(q, 5)) === NONE) across = 1;
      let band = false, native = false, n = { phi: PHI_AIR, wet: false, neumann: false };
      if (across === 0) {
        band = cellH(q);
        if (!band) { const root = rootPhi(shr(q, 2)); native = wet(root, 4); if (native) n = { phi: root, wet: true, neumann: true }; }
        if (!native) { const p = centre(q), w = wet(p, 1); n = { phi: p, wet: w, neumann: w }; }
      }
      const volume = inLattice(q) ? 1 : 0.5;
      let kind = KIND.GHOST, coefficient = 0, fraction = 1;
      if (across !== 0) { if (across === 1) { kind = KIND.OPEN; coefficient = 1 / thetaOf(own, 0.5); } else { kind = KIND.WALL; coefficient = 0.5; fraction = 0.5; } }
      else {
        const theta = liquid !== n.wet ? faceTheta(facePhi(c, a, sign), own, liquid, n.phi, n.wet) : 1;
        if (band) { kind = KIND.BAND; coefficient = 1 / theta; }
        else if (native) { kind = KIND.SEAM; coefficient = 1 / (2.5 * (liquid ? 1 : theta)); }
        else if (n.neumann) kind = KIND.NEUMANN;
        else coefficient = 1 / theta;
      }
      if (kind === KIND.NEUMANN) fraction = volume; else if (kind === KIND.WALL) { coefficient = volume; fraction = volume; } else { coefficient *= volume; fraction = volume; }
      return { kind, coefficient, fraction, qWet: n.wet };
    };
    const sgn = (g: number) => g & 1 ? 1 : -1;
    const head = (c: V3, own: number, base: number, f: number, baseOf: (m: V3, pm: number) => number) => {
      if (!wet(own, 1)) return 0;
      let hd = 0;
      for (let axis = 0; axis < 3; axis++) {
        if (axis === f >> 1) continue;
        let rise = 0, count = 0;
        for (let side = 0; side < 2; side++) { const g = 2 * axis + side, m = add(c, scale(sgn(g), unit(axis))); if (inLattice(m) && cellH(m)) { const pm = centre(m); if (wet(pm, 1)) { rise += sgn(g) * (baseOf(m, pm) - base); count++; } } }
        if (count > 0) hd += rise / count * ((c[axis]! & 3) - 1.5);
      }
      return hd;
    };
    const stats = { rowsF: 0, rowsC: 0, seamsF: 0, seamsC: 0, wallsAttached: 0, wallsSeparated: 0, held: 0, ghosts4: 0, natives: 0, heads: 0, heldFits: 0 };
    const tol = (got: number, want: number, sc: number, what: string) => assert.ok(Math.abs(got - want) <= 2e-4 * (Math.abs(want) + sc), `${what}: ${got} against ${want}`);
    // F rows.
    const fx = await readF("X"), frhs = await readF("RHS"), fdiag = await readF("DIAG"), fk = [await readF("K0"), await readF("K1"), await readF("K2")], fku = [await readF("KU0"), await readF("KU1"), await readF("KU2")];
    for (const [key, slot] of s.bricks) {
      const o = scale(8, keyCoord(key));
      for (let l = 0; l < 512; l++) {
        const c = add(o, [l & 7, (l >> 3) & 7, l >> 6]), at = BRICK_STRIDE * slot + l;
        const owned = inLattice(c) && cellH(c), own = owned ? centre(c) : PHI_AIR, liquid = owned && wet(own, 1);
        let diag = 0, div = 0, extra = 0, st = 0; const kk = [0, 0, 0], ku = [0, 0, 0];
        if (liquid) {
          stats.rowsF++; st = start(c, own);
          for (let f = 0; f < 6; f++) {
            const a = f >> 1, sign = sgn(f), face = bandFace(c, own, true, f), us = facePlane(sign > 0 ? add(c, unit(a)) : c, a);
            if (face.kind !== KIND.CLOSED) div += sign * face.fraction * us;
            if (face.kind === KIND.NEUMANN || face.kind === KIND.CLOSED) continue;
            if (face.kind === KIND.WALL) { const lift = sign * us * h; if (st + lift >= 0) { extra += face.coefficient * lift; stats.wallsAttached++; } else { diag += face.coefficient; stats.wallsSeparated++; } continue; }
            diag += face.coefficient;
            if (face.kind === KIND.BAND && face.qWet && sign < 0) kk[a] = face.coefficient;
            if (face.kind === KIND.SEAM) { stats.seamsF++; if (sign < 0) kk[a] = face.coefficient; else ku[a] = face.coefficient; const hd = head(c, own, st, f, start); if (hd !== 0) stats.heads++; extra += face.coefficient * hd; }
          }
        }
        const sc = Math.abs(diag) + 1e-3;
        if (owned) { tol(fx[at]!, st, 1e-3, `F X ${c}`); tol(frhs[at]!, liquid ? -h * div + h * h * rate + extra : 0, 1e-3 * sc, `F RHS ${c}`); }
        tol(fdiag[at]!, diag, 1e-3, `F DIAG ${c}`);
        for (let a = 0; a < 3; a++) { tol(fk[a]![at]!, kk[a]!, 1e-3, `F K${a} ${c}`); tol(fku[a]![at]!, ku[a]!, 1e-3, `F KU${a} ${c}`); }
      }
    }
    // C rows.
    const cx = await readC("X"), crhs = await readC("RHS"), cdiag = await readC("DIAG"), ck = [await readC("K0"), await readC("K1"), await readC("K2")], cpphi = await readC("PPHI");
    for (const cs of pages) {
      const O = scale(8, keyCoord(dir[L.dir.crec + 32 * cs + 29]!));
      for (let i = 0; i < 512; i++) {
        const T = add(O, [i & 7, (i >> 3) & 7, i >> 6]), at = BRICK_STRIDE * cs + i;
        const owned = tileInLattice(T) && !tileH(T), root = owned ? rootPhi4(T) : 0, liquid = owned && wet(root, 4);
        let diag = 0, div = 0, extra = 0, st = 0; const kk = [0, 0, 0];
        if (liquid) {
          stats.rowsC++; st = tileStart(T);
          for (let f = 0; f < 6; f++) {
            const a = f >> 1, side = f & 1, sign = sgn(f), Tn = add(T, scale(sign, unit(a))), plane = scale(4, side ? Tn : T);
            const lowT = side ? T : Tn, highT = side ? Tn : T;
            if (!tileInLattice(Tn)) {
              const u = facePlane(plane, a), lift = sign * 4 * h * u; div += sign * 0.5 * u;
              if (root < -4) { extra += 2 * lift; stats.held++; } else if (st + lift >= 0) { extra += 2 * lift; stats.wallsAttached++; } else { diag += 2; stats.wallsSeparated++; }
              continue;
            }
            if (pageSlot(shr(Tn, 3)) === NONE) { diag += 4 / nativeTheta(lowT, highT, a); div += sign * facePlane(plane, a); stats.ghosts4++; continue; }
            if (tileH(Tn)) {
              stats.seamsC++;
              const u = unit((a + 1) % 3), v = unit((a + 2) % 3), g2 = f ^ 1, layer = scale(4, Tn); if (!side) layer[a]! += 3;
              let sum = 0;
              for (let q = 0; q < 16; q++) {
                const m = add(add(layer, scale(q & 3, u)), scale(q >> 2, v)), pm = centre(m), lm = wet(pm, 1), pf = bandFace(m, pm, lm, g2);
                if (pf.kind === KIND.SEAM) { diag += pf.coefficient; if (lm) extra -= pf.coefficient * head(m, pm, start(m, pm), g2, start); }
                sum += facePlane(side ? m : add(m, unit(a)), a);
              }
              div += sign * sum / 16; continue;
            }
            div += sign * facePlane(plane, a);
            if (wet(rootPhi4(Tn), 4)) { diag += 4; if (!side) kk[a] = 4; stats.natives++; } else { diag += 4 / nativeTheta(lowT, highT, a); stats.ghosts4++; }
          }
        }
        const sc = Math.abs(diag) + 1e-3;
        tol(cx[at]!, owned ? st : C.X[at]!, 1e-3, `C X ${T}`); tol(crhs[at]!, liquid ? -16 * h * div + 64 * h * h * rate + extra : 0, 1e-3 * sc, `C RHS ${T}`);
        tol(cdiag[at]!, diag, 1e-3, `C DIAG ${T}`); // An h tile of a live brick holds tileRoots' held fit (rowsC keeps it).
        if (!(tileInLattice(T) && tileH(T))) tol(cpphi[at]!, root, 1e-3, `C PPHI ${T}`);
        else if (fSlot(scale(4, T)) !== undefined) { tol(cpphi[at]!, rootPhi(T), 1e-3, `C PPHI held fit ${T}`); stats.heldFits++; }
        for (let a = 0; a < 3; a++) tol(ck[a]![at]!, kk[a]!, 1e-3, `C K${a} ${T}`);
      }
    }
    console.log(`# composite rows: ${JSON.stringify({ ...stats, cold, carries })}`);
    for (const [k, v] of Object.entries({ ...stats, cold, carries })) assert.ok(v > 0, `${k} not exercised`);
  });
});
