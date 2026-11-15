// setTransfer (brick-transfer.wgsl.ts) on a frozen state against a CPU oracle
// transcribed from NB-FLIP's remap (uniform-mixed-remap.ts: markTile,
// listedFlags, remapBeside, remapSlope, remapSample, remapFace,
// remapTileCells) on the brick storage of brick-abi.md §2.5. The formulas
// (nbfRemap*) are F's bitwise ports; this checks the assembly: which tiles
// are listed, which faces and vertices each writes, where they are stored,
// and that storage no job writes is untouched. GPU only.
import assert from "node:assert/strict";
import test from "node:test";
import { ARGS, BRICK_STRIDE, CMASK_WORDS, H, LIST, phiFar } from "./brick-abi";
import { bringUpCG } from "./brick-bringup-cg";
import { BrickEngine } from "./brick-engine";
import { readChannel, snapshot } from "./brick-readback";
import { bringUpStages } from "./brick-stages";
import { brickTransferSchedule } from "./brick-transfer.wgsl";

const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };
type V3 = [number, number, number];
const N = 32;
const FAR = phiFar([N, N, N]);
const f = Math.fround;
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const unit = (a: number): V3 => [a === 0 ? 1 : 0, a === 1 ? 1 : 0, a === 2 ? 1 : 0];
const shr = (a: V3, s: number): V3 => [a[0] >> s, a[1] >> s, a[2] >> s];
const tileIndex = (t: V3) => (t[0] & 7) | ((t[1] & 7) << 3) | ((t[2] & 7) << 6);
const local8 = (c: V3) => (c[0] & 7) | ((c[1] & 7) << 3) | ((c[2] & 7) << 6);
const corner4 = (l: number): V3 => [l & 3, (l >> 2) & 3, l >> 4];
const key = (b: V3) => ((b[0] & 1023) | ((b[1] & 1023) << 10) | ((b[2] & 1023) << 20)) >>> 0;
/** Old 4h block A (tiles x 2..3, y 0..1, on the low y wall); new: A's x = 3, z 0..3 refines; block B (x 5..6, y 2..3, z 4..7, on the upper z wall) coarsens. */
const oldCoarse = (t: V3) => t[0] >= 2 && t[0] <= 3 && t[1] <= 1;
const newCoarse = (t: V3) => (oldCoarse(t) && !(t[0] === 3 && t[2] <= 3)) || (t[0] >= 5 && t[0] <= 6 && t[1] >= 2 && t[1] <= 3 && t[2] >= 4);
// Smooth fields with curvature, so slopes engage and differ by side.
const fPhi = (p: V3) => 0.3 * p[1] - 4 + Math.sin(0.4 * p[0]) + 0.2 * Math.cos(0.3 * p[2]);
const cPhi = (T: V3) => 1.2 * T[1] - 3.5 + Math.cos(0.9 * T[0]) - 0.1 * T[2];
const fU = (ch: number, c: V3, a: number) => 0.01 * (a + 1 + 3 * ch) + 0.004 * Math.sin(0.5 * c[0] + a) + 0.003 * c[1] - 0.002 * Math.cos(0.7 * c[2]);
const cU = (ch: number, T: V3, a: number) => -0.02 * (a + 1 + 3 * ch) + 0.01 * Math.sin(1.3 * T[0] + T[2]) + 0.005 * T[1] * T[1];

test("setTransfer (GPU): listing, faces, vertices and storage match NB-FLIP's remap on the brick storage", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("brick transfer", async device => {
    const e = await BrickEngine.create(device, { lattice: [N, N, N], h_m: 1 / 32, dt_s: 1 / 60, gravity_mps2: [0, -9.81, 0],
      capacities: { f: 200, c: 8, b: 0, k: 0, particles: 1024, t: 512 }, liquid: [{ lo: [-1, -1, -1], hi: [33, 20.3, 33] }], solver: L => bringUpCG(L), stages: bringUpStages });
    const s = await snapshot(e), L = e.layout;
    const cs = s.pages.get(0)!;
    assert.ok(cs !== undefined);
    // ---- The frozen state: masks, φ and faces on F and C.
    const boot = (await e.read("dir", L.dir.cmask + CMASK_WORDS * cs, CMASK_WORDS)).slice();
    const bit = (m: Uint32Array, i: number) => ((m[i >> 5]! >>> (i & 31)) & 1) === 1;
    const oldMask = boot.slice(), newMask = boot.slice();
    for (let i = 0; i < 512; i++) {
      const t: V3 = [i & 7, (i >> 3) & 7, i >> 6];
      if (oldCoarse(t)) oldMask[i >> 5]! &= ~(1 << (i & 31));
      if (newCoarse(t)) newMask[i >> 5]! &= ~(1 << (i & 31)); else if (bit(boot, i)) newMask[i >> 5]! |= 1 << (i & 31);
    }
    e.write("dir", L.dir.cmask + CMASK_WORDS * cs, oldMask);
    e.write("set", L.set.want + CMASK_WORDS * cs, newMask);
    const channels = ["U", "US"] as const;
    const cWords = (fn: (T: V3) => number) => { const w = new Float32Array(512); for (let i = 0; i < 512; i++) w[i] = fn([i & 7, (i >> 3) & 7, i >> 6]); return w; };
    e.write("phi", L.base("phi", "C", "PHI") + BRICK_STRIDE * cs, cWords(cPhi));
    channels.forEach((ch, k) => { for (let a = 0; a < 3; a++) e.write("vel", L.base("vel", "C", `${ch}${a}`) + BRICK_STRIDE * cs, cWords(T => cU(k, T, a))); });
    const slotOf = new Map<number, number>(s.bricks);
    for (const [kb, slot] of s.bricks) {
      const b: V3 = [kb & 1023, (kb >>> 10) & 1023, (kb >>> 20) & 1023];
      const words = (fn: (c: V3) => number) => { const w = new Float32Array(512); for (let l = 0; l < 512; l++) w[l] = fn([8 * b[0] + (l & 7), 8 * b[1] + ((l >> 3) & 7), 8 * b[2] + (l >> 6)]); return w; };
      e.write("phi", L.base("phi", "F", "PHI") + BRICK_STRIDE * slot, words(fPhi));
      channels.forEach((ch, k) => { for (let a = 0; a < 3; a++) e.write("vel", L.base("vel", "F", `${ch}${a}`) + BRICK_STRIDE * slot, words(c => fU(k, c, a))); });
    }
    // ---- Read the state back as the oracle's storage (f32 as stored).
    const fStore = new Map<string, (slot: number) => Float32Array>();
    const cStore = new Map<string, Map<number, Float32Array>>();
    const fNames = ["PHI", ...channels.flatMap(ch => [0, 1, 2].map(a => `${ch}${a}`))];
    for (const name of fNames) {
      const pool = name === "PHI" ? "phi" : "vel";
      fStore.set(name, (await readChannel(e, pool, name)) as (slot: number) => Float32Array);
      const pages = new Map<number, Float32Array>();
      for (const pc of s.pages.values()) pages.set(pc, new Float32Array((await e.read(pool, L.base(pool, "C", name) + BRICK_STRIDE * pc, 512)).buffer));
      cStore.set(name, pages);
    }
    const fBefore = new Map(fNames.map(n => [n, new Map([...slotOf.values()].map(sl => [sl, fStore.get(n)!(sl).slice()]))]));
    const cBefore = cStore;
    // ---- The oracle.
    const inLattice = (c: V3) => c.every(v => v >= 0 && v < N);
    const tileInLattice = (t: V3) => t.every(v => v >= 0 && 4 * v < N);
    const pageOf = (c: V3) => (c.every(v => v >= 0) && c.every(v => v >> 5 === 0)) ? cs : undefined; // only page 0 is in the lattice
    const present = (c: V3) => inLattice(c) && pageOf(c) !== undefined;
    const tileH = (m: Uint32Array, t: V3) => tileInLattice(t) && bit(m, tileIndex(t));
    const stored = (t: V3) => tileInLattice(t);
    const fSlot = (c: V3) => c.some(v => v < 0) ? undefined : slotOf.get(key(shr(c, 3)));
    const fRead = (name: string, c: V3) => { const sl = fSlot(c); return sl === undefined ? undefined : fBefore.get(name)!.get(sl)![local8(c)]!; };
    const cSlot = (T: V3) => T.some(v => v < 0) ? undefined : s.pages.get(key(shr(T, 3)));
    const cRead = (name: string, T: V3) => { const pc = cSlot(T); return pc === undefined ? undefined : cBefore.get(name)!.get(pc)![tileIndex(T)]!; };
    const fVertex = (v: V3) => fRead("PHI", v) ?? FAR;
    const cVertex = (T: V3) => (T.some(x => x < 0) || T.some(x => 4 * x > N)) ? FAR : cRead("PHI", T) ?? FAR;
    const vertexCoarse = (m: Uint32Array, v: V3): V3 | undefined => {
      const tA = shr(v, 2), inside = v.map((x, a) => x - 4 * tA[a]! !== 0);
      for (let k = 0; k < 8; k++) {
        const corner = [k & 1, (k >> 1) & 1, k >> 2];
        if (corner.some((c, a) => inside[a] && c === 0)) continue;
        const t = tA.map((x, a) => x + corner[a]! - 1) as V3;
        if (tileInLattice(t) && cSlot(t) !== undefined && !tileH(m, t)) return t; // a non-resident page's tile is no authority
      }
      return undefined;
    };
    const vertexPhiOld = (v: V3) => {
      if (v.some(x => x < 0 || x > N)) return FAR;
      const T = vertexCoarse(oldMask, v);
      if (!T) return fVertex(v);
      const t = v.map((x, a) => f((x - 4 * T[a]!) / 4)), values = new Array<number>(8).fill(0);
      for (let k = 0; k < 8; k++) {
        const corner = [k & 1, (k >> 1) & 1, k >> 2], w = corner.map((c, a) => c ? t[a]! : f(1 - t[a]!));
        const weight = f(f(w[0]! * w[1]!) * w[2]!);
        if (weight > 0) values[k] = f(weight * cVertex(add(T, corner as V3)));
      }
      const x = values;
      return f(f(f(x[0]! + x[5]!) + f(x[1]! + x[4]!)) + f(f(x[2]! + x[7]!) + f(x[3]! + x[6]!)));
    };
    const cellH = (m: Uint32Array, c: V3) => tileH(m, shr(c, 2));
    // Wall planes 0 and N hold their faces (lower-face convention: F at the cell, C at tile N/4 for plane N).
    const faceInLattice = (c: V3, a: number) => c.every((v, i) => v >= 0 && v < (i === a ? N + 1 : N));
    const facePlane = (ch: string, c: V3, a: number) => {
      if (!faceInLattice(c, a)) return 0;
      const below = add(c, unit(a).map(x => -x) as V3);
      if (cellH(oldMask, c) || cellH(oldMask, below)) return fRead(`${ch}${a}`, c) ?? 0;
      return cRead(`${ch}${a}`, shr(c, 2)) ?? 0;
    };
    // umPressureFaceV: 1 between lattice cells, 0.5 into the exterior.
    const xV = (c: V3, a: number) => inLattice(c) && inLattice(add(c, unit(a))) ? 1 : 0.5;
    const changed = (t: V3) => stored(t) && tileH(oldMask, t) !== tileH(newMask, t);
    const refined = (t: V3) => stored(t) && tileH(newMask, t) && !tileH(oldMask, t);
    // markTile / listedFlags.
    const entries: { t: V3; flags: number }[] = [];
    for (let i = 0; i < 512; i++) {
      const t: V3 = [i & 7, (i >> 3) & 7, i >> 6];
      if (!tileInLattice(t)) continue;
      const oH = tileH(oldMask, t), nH = tileH(newMask, t), fine = oH && nH;
      let flags = (nH !== oH ? 2 ** 31 : 0) + (nH && !oH ? 2 ** 30 : 0);
      for (let a = 0; a < 3; a++) if (changed(add(t, unit(a)))) flags += 2 ** (27 + a);
      const faces = flags >= 2 ** 27;
      if (!nH && !faces) continue;
      if (fine) {
        if ([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]].some(d => refined(add(t, d as V3)))) flags += 2 ** 26;
        if (!faces && flags < 2 ** 26) continue;
      }
      let listed = false;
      for (let n = 0; n < 27 && !listed; n++) { const q = add(t, [n % 3 - 1, ((n / 3) | 0) % 3 - 1, ((n / 9) | 0) - 1]); listed = fine ? refined(q) : changed(q); }
      if (listed) entries.push({ t, flags });
    }
    const CHANGED = 2 ** 31, has = (flags: number, b: number) => Math.floor(flags / 2 ** b) % 2 === 1;
    const texelOf = (flags: number, l: V3) => has(flags, 31) || [0, 1, 2].some(a => l[a] === 3 && has(flags, 27 + a));
    const faceWidth = (c: V3, a: number) => {
      const up = add(c, unit(a)), wall = up[a] === N;
      if (!wall && !present(up)) return 0;
      if (cellH(newMask, c)) return 1;
      const l = c.map(x => x & 3);
      if (l[a] !== 3) return 0;
      if (!wall && cellH(newMask, up)) return 1;
      return l[(a + 1) % 3] === 0 && l[(a + 2) % 3] === 0 ? 4 : 0;
    };
    const fWrites = new Map<string, number>(), cWrites = new Map<string, number>();
    let tilted = 0, walls = 0;
    const writeF = (name: string, c: V3, v: number) => { const sl = fSlot(c); assert.ok(sl !== undefined, `F storage for ${c}`); fWrites.set(`${name}|${sl}|${local8(c)}`, v); };
    const writeC = (name: string, T: V3, v: number) => { const pc = cSlot(T); assert.ok(pc !== undefined, `C storage for ${T}`); cWrites.set(`${name}|${pc}|${tileIndex(T)}`, v); };
    for (const { t, flags } of entries) {
      for (const ch of channels) {
        if (flags < 2 ** 27) continue;
        const own = flags >= CHANGED;
        const beside: number[][] = [], openFaces: number[][] = [];
        for (let lane = 0; lane < 192; lane++) {
          const axis = (lane / 64) | 0, cell = lane % 64;
          let b = [0, 0, 0, 0], open = [0, 0];
          if (!tileH(oldMask, t) && (own || has(flags, 27 + axis))) {
            const k = (cell / 16) | 0, d = (axis + 1 + ((k / 2) | 0)) % 3, u = (axis + 1) % 3, v = (axis + 2) % 3;
            const q: V3 = [4 * t[0], 4 * t[1], 4 * t[2]]; q[u] += cell % 4; q[v] += ((cell / 4) | 0) % 4;
            const low: V3 = [...q], high: V3 = [...q]; high[axis] += 3;
            const step = (k & 1) ? 4 : -4, lowB: V3 = [...low], highB: V3 = [...high]; lowB[d] += step; highB[d] += step;
            const e_ = unit(axis), m = (c: V3) => add(c, e_.map(x => -x) as V3);
            const V = [-1, -1, -1, -1];
            if (k === 0 && own && present(m(low))) V[0] = xV(m(low), axis);
            if (k === 0) V[1] = xV(high, axis);
            if (own && present(lowB) && present(m(lowB))) V[2] = xV(m(lowB), axis);
            if (present(highB)) V[3] = xV(highB, axis);
            open = [V[0]! >= 1 ? 1 : 0, V[1]! >= 1 ? 1 : 0];
            const besideFace = (qq: V3, sign: number, VV: number) => {
              if (!present(qq)) return [0, 0];
              const id = sign < 0 ? m(qq) : qq;
              if (!present(id) || VV < 1) return [0, 0];
              return [facePlane(ch, sign < 0 ? qq : add(qq, e_), axis), 1];
            };
            const lb = own ? besideFace(lowB, -1, V[2]!) : [0, 0];
            b = [...lb, ...besideFace(highB, 1, V[3]!)];
          }
          beside[lane] = b; if (cell < 16) openFaces[16 * axis + cell] = open;
        }
        const slopes = new Array<number>(12).fill(0);
        for (let axis = 0; axis < 3; axis++) for (let cell = 0; cell < 4; cell++) {
          const high = cell >= 2, k = cell % 2;
          let slope = 0;
          if (!tileH(oldMask, t)) {
            const across: V3 = [...t]; across[axis] += high ? 1 : -1;
            let open = 0; const below = [0, 0], above = [0, 0];
            for (let i = 0; i < 16; i++) {
              const a = beside[64 * axis + 32 * k + i]!, b = beside[64 * axis + 32 * k + 16 + i]!, c = openFaces[16 * axis + i]!;
              open += high ? c[1]! : c[0]!;
              below[0] = f(below[0]! + (high ? a[2]! : a[0]!)); below[1] += high ? a[3]! : a[1]!;
              above[0] = f(above[0]! + (high ? b[2]! : b[0]!)); above[1] += high ? b[3]! : b[1]!;
            }
            if (!tileH(oldMask, across) && open === 16) {
              const plane: V3 = [4 * t[0], 4 * t[1], 4 * t[2]]; if (high) plane[axis] += 4;
              const value = facePlane(ch, plane, axis);
              const lower = below[1] === 16 ? f(value - f(below[0]! / 16)) : 0, upper = above[1] === 16 ? f(f(above[0]! / 16) - value) : 0;
              if (lower * upper > 0) slope = Math.sign(lower) * Math.min(f(Math.abs(f(lower + upper)) / 8), f(f((4 / 3) * Math.min(Math.abs(lower), Math.abs(upper))) / 4));
            }
          }
          slopes[4 * axis + cell] = slope; if (slope !== 0) tilted++;
        }
        const sample = (p: V3, plane: number, axis: number) => {
          const fineOld = cellH(oldMask, p), w = fineOld ? 1 : 4, origin: V3 = fineOld ? p : [4 * (p[0] >> 2), 4 * (p[1] >> 2), 4 * (p[2] >> 2)];
          const lo: V3 = [...p]; lo[axis] = origin[axis]; const hi: V3 = [...p]; hi[axis] = origin[axis] + w;
          const u = (axis + 1) % 3, v = (axis + 2) % 3, along = f((plane + 1 - origin[axis]) / w);
          let low = facePlane(ch, lo, axis), high = facePlane(ch, hi, axis);
          if (!fineOld) {
            const ac = [p[u] - origin[u] - 1.5, p[v] - origin[v] - 1.5];
            low = f(low + f(f(slopes[4 * axis]! * ac[0]!) + f(slopes[4 * axis + 1]! * ac[1]!)));
            high = f(high + f(f(slopes[4 * axis + 2]! * ac[0]!) + f(slopes[4 * axis + 3]! * ac[1]!)));
          }
          return f(f(low * f(1 - along)) + f(high * along));
        };
        // The low-wall rows: a changed tile's new owners on the wall, their negative face at t = 0.
        if (own) for (let lane = 0; lane < 192; lane++) {
          const axis = (lane / 64) | 0, l = corner4(lane % 64), anchor = add([4 * t[0], 4 * t[1], 4 * t[2]], l);
          const fine = cellH(newMask, anchor), origin: V3 = fine ? anchor : [4 * t[0], 4 * t[1], 4 * t[2]], w = fine ? 1 : 4;
          if (anchor[axis] !== 0 || anchor.some((x, a) => x !== origin[a])) continue;
          const u = (axis + 1) % 3, v = (axis + 2) % 3;
          let sum = 0;
          for (let y = 0; y < w; y++) for (let x = 0; x < w; x++) { const q: V3 = [...anchor]; q[u] += x; q[v] += y; sum = f(sum + sample(q, -1, axis)); }
          const value = f(sum / (w * w));
          if (fine) writeF(`${ch}${axis}`, anchor, value); else writeC(`${ch}${axis}`, t, value);
          walls++;
        }
        for (let lane = 0; lane < 192; lane++) {
          const axis = (lane / 64) | 0, l = corner4(lane % 64), anchor = add([4 * t[0], 4 * t[1], 4 * t[2]], l);
          if (!texelOf(flags, l)) continue;
          const fw = faceWidth(anchor, axis); if (fw === 0) continue;
          const u = (axis + 1) % 3, v = (axis + 2) % 3;
          let sum = 0, weight = 0;
          for (let y = 0; y < fw; y++) for (let x = 0; x < fw; x++) {
            const q: V3 = [...anchor]; q[u] += x; q[v] += y;
            const fineNew = cellH(newMask, q), plane = fineNew ? q[axis] : 4 * (q[axis] >> 2) + 3;
            sum = f(sum + sample(q, plane, axis)); weight += 1;
          }
          const value = f(sum / weight);
          if (fw === 1) writeF(`${ch}${axis}`, add(anchor, unit(axis)), value); else writeC(`${ch}${axis}`, add(t, unit(axis)), value);
          if (anchor[axis] + 1 === N) walls++;
        }
      }
      // Vertices.
      const same = flags < CHANGED, refine = has(flags, 30);
      if (same && (!tileH(newMask, t) || !has(flags, 26))) continue;
      const corners = [0, 1, 2, 3, 4, 5, 6, 7].map(k => cVertex(add(t, [k & 1, (k >> 1) & 1, k >> 2])));
      const isCorner = (p: V3, T: V3) => p.every((x, a) => (x - 4 * T[a]!) % 4 === 0);
      for (let vtx = 0; vtx < 125; vtx++) {
        const p: V3 = add([4 * t[0], 4 * t[1], 4 * t[2]], [vtx % 5, ((vtx / 5) | 0) % 5, (vtx / 25) | 0]);
        if (p.some(x => x > N)) continue;
        const T = vertexCoarse(newMask, p), newC = T !== undefined;
        const own = newC ? T.every((x, a) => x === t[a]) && isCorner(p, T) : shr(p.map(x => Math.max(x - 1, 0)) as V3, 2).every((x, a) => x === t[a]);
        if (refine && !newC && isCorner(p, t)) { writeF("PHI", p, cVertex(shr(p, 2))); continue; }
        if (!own) continue;
        const To = vertexCoarse(oldMask, p), oldF = To === undefined, oldC = !oldF && isCorner(p, To);
        if ((newC && oldC) || (!newC && oldF)) continue;
        let value: number;
        if (oldC) value = cVertex(shr(p, 2));
        else if (oldF) value = fVertex(p);
        else if (refine) {
          const tt = [vtx % 5, ((vtx / 5) | 0) % 5, (vtx / 25) | 0].map(x => f(x * 0.25)), vals = new Array<number>(8).fill(0);
          for (let k = 0; k < 8; k++) { const c = [k & 1, (k >> 1) & 1, k >> 2], w = c.map((cc, a) => cc ? tt[a]! : f(1 - tt[a]!)); vals[k] = f(f(f(corners[k]! * w[0]!) * w[1]!) * w[2]!); }
          value = f(f(f(vals[0]! + vals[5]!) + f(vals[1]! + vals[4]!)) + f(f(vals[2]! + vals[7]!) + f(vals[3]! + vals[6]!)));
        } else value = vertexPhiOld(p);
        if (newC) writeC("PHI", shr(p, 2), value); else writeF("PHI", p, value);
      }
    }
    // ---- Run setTransfer.
    await e.run(brickTransferSchedule(L, k => ({ indirect: ARGS.PER_LIST * k + ARGS.G1 }), LIST.C_ALL));
    const header = await e.read("set", 0, 256);
    assert.equal(header[H.FAIL], 0, `FAIL ${header[H.FAIL]} info ${header[H.FAIL_INFO]?.toString(16)}`);
    assert.equal(header[H.N_TLIST_RUN], entries.length, "listed tiles");
    assert.ok(entries.some(x => has(x.flags, 30)) && entries.some(x => x.flags >= CHANGED && !has(x.flags, 30)) && entries.some(x => x.flags < CHANGED), "refining, coarsening and neighbour-listed tiles");
    // ---- Compare every stored word: written ones to the oracle (mix() and fma rounding: 2e-6 relative), the rest unchanged.
    let written = 0, kept = 0;
    const near = (got: number, want: number, what: string) => assert.ok(Math.abs(got - want) <= 2e-6 * Math.max(1, Math.abs(want)), `${what}: ${got} against ${want}`);
    for (const name of fNames) {
      const pool = name === "PHI" ? "phi" : "vel", after = (await readChannel(e, pool, name)) as (slot: number) => Float32Array;
      for (const sl of slotOf.values()) {
        const got = after(sl), was = fBefore.get(name)!.get(sl)!;
        for (let l = 0; l < 512; l++) {
          const w = fWrites.get(`${name}|${sl}|${l}`);
          if (w === undefined) { if (got[l] !== was[l]) assert.fail(`F ${name} slot ${sl} local ${l} changed: ${was[l]} -> ${got[l]}`); kept++; }
          else { near(got[l]!, w, `F ${name} slot ${sl} local ${l}`); written++; }
        }
      }
      for (const pc of s.pages.values()) {
        const cAfter = new Float32Array((await e.read(pool, L.base(pool, "C", name) + BRICK_STRIDE * pc, 512)).buffer), cWas = cBefore.get(name)!.get(pc)!;
        for (let i = 0; i < 512; i++) {
          const w = cWrites.get(`${name}|${pc}|${i}`);
          if (w === undefined) { if (!Object.is(cAfter[i], cWas[i])) assert.fail(`C ${name} page ${pc} tile ${i} changed: ${cWas[i]} -> ${cAfter[i]}`); }
          else { near(cAfter[i]!, w, `C ${name} page ${pc} tile ${i}`); written++; }
        }
      }
    }
    assert.deepEqual([...await e.read("dir", L.dir.cmask + CMASK_WORDS * cs, CMASK_WORDS)], [...newMask], "CMASK = WANT");
    const cPhiWrites = [...cWrites.keys()].filter(k => k.startsWith("PHI")).length, c4h = [...cWrites.keys()].filter(k => k.startsWith("U")).length;
    assert.ok(written > 1000 && kept > 0 && tilted > 0 && cPhiWrites > 0 && c4h > 0 && walls > 0, `written ${written}, kept ${kept}, tilted slopes ${tilted}, C corners ${cPhiWrites}, C faces ${c4h}, wall faces ${walls}`);
    console.log(`setTransfer: ${entries.length} listed, ${written} written, ${kept} kept, ${tilted} tilted slopes, ${cPhiWrites} C corners, ${c4h} C faces, ${walls} wall faces`);
    e.destroy();
  });
});
