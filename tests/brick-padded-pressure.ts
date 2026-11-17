// LANE P OWNS THIS FILE. The pressure observables of a padded lattice (M3: the lattice whole pages, the box the
// container; pressure-composite.wgsl.ts pressureResolveBox), for lane E's padded regression and P's still tank
// (tests/brick-padded-pressure.test.ts). Read after a step (the engine fenced), from the engine's own buffers:
// - **No row past the box:** no h cell (F DIAG > 0) and no native tile (C DIAG > 0, not h-owned) outside the box.
// - **The walls are at the box, once:** every face of a liquid h row on a box plane is WALL (the box faces), and no
//   other face is (a WALL face inside the box would be the storage edge's, or a wall twice).
// - **Still water:** the peak |U| over the liquid faces, h and native, the box planes' included, against the solve's
//   own stopping bound (2 MG_TOLERANCE h: max |div| <= MG_TOLERANCE carried in U from step to step).
// One page (a lattice of at most 32 an axis): page 0's word is its C slot. Layout constants are brick-abi's.
import assert from "node:assert/strict";
import { BRICK_CELLS, BRICK_STRIDE, CMASK_WORDS, H, LIST, RECORD_WORDS, REC, SEAM } from "../lib/methods/narrow-band/brick-abi";
import type { BrickEngine } from "../lib/methods/narrow-band/brick-engine";
import { MG_TOLERANCE, mgLayout } from "../lib/methods/narrow-band/pressure-mgpcg";
import { NP_K } from "../lib/methods/narrow-band/pressure-rows.wgsl";

export interface PaddedPressure {
  /** Liquid h rows, and liquid native (4h-owned) tiles. */
  readonly hRows: number;
  readonly nativeRows: number;
  /** Rows past the box (h cells and native tiles): 0. */
  readonly pastBox: number;
  /** Liquid h rows' faces on a box plane, all WALL; and WALL faces anywhere else (0). */
  readonly boxFaces: number;
  readonly strayWalls: number;
  /** A box face of a liquid h row whose kind is not WALL (0), the first one. */
  readonly openBoxFace?: string;
  /** Peak |U| (m/s) over the liquid faces, and over the box planes' faces alone. */
  readonly peak: number;
  readonly wallPeak: number;
  readonly where: string;
  /** The bound peak must meet: 2 MG_TOLERANCE h. */
  readonly bound: number;
}

/** Measures the padded pressure observables after a step (no asserts). */
export async function paddedPressure(e: BrickEngine, h_m: number): Promise<PaddedPressure> {
  const L = e.layout, M = mgLayout(L), box = L.box;
  assert.ok(L.lattice.every(n => n <= 32), "one page: page 0's word is the C slot");
  const f32 = (w: Uint32Array) => new Float32Array(w.buffer, w.byteOffset, w.length);
  const head = await e.read("set", 0, 64), n = Math.min(head[H.LIST_COUNT + LIST.F_ALL]!, L.set.listCap[LIST.F_ALL]!);
  const slots = await e.read("set", L.set.lists[LIST.F_ALL]!, Math.max(n, 1));
  const F = L.cap.f * BRICK_STRIDE, C = L.cap.c * BRICK_STRIDE;
  const diag = f32(await e.read("pres", L.base("pres", "F", "DIAG"), F)), cdiag = f32(await e.read("pres", L.base("pres", "C", "DIAG"), C));
  const fc = f32(await e.read("pmgF", M.FC, 7 * F)), u = f32(await e.read("vel", L.base("vel", "F", "U0"), 3 * F)), uc = f32(await e.read("vel", L.base("vel", "C", "U0"), 3 * C));
  const cs = (await e.read("dir", L.dir.page, 1))[0]!; assert.ok(cs < SEAM, "page 0 holds a C slot");
  const cmask = await e.read("dir", L.dir.cmask + CMASK_WORDS * cs, CMASK_WORDS), tileH = (i: number) => ((cmask[i >> 5]! >>> (i & 31)) & 1) !== 0;
  const slotOf = new Map<number, number>();
  for (let i = 0; i < n; i++) slotOf.set((await e.read("dir", L.dir.frec + RECORD_WORDS * slots[i]! + REC.KEY, 1))[0]!, slots[i]!);
  const fSlot = (c: number[]) => slotOf.get(((c[0]! >> 3) | ((c[1]! >> 3) << 10) | ((c[2]! >> 3) << 20)) >>> 0);
  const local = (c: number[]) => (c[0]! & 7) | ((c[1]! & 7) << 3) | ((c[2]! & 7) << 6), tile = (T: number[]) => (T[0]! & 7) | ((T[1]! & 7) << 3) | ((T[2]! & 7) << 6);
  const outside = (c: number[]) => c.some((v, a) => v < 0 || v >= box[a]!);
  let hRows = 0, nativeRows = 0, pastBox = 0, boxFaces = 0, strayWalls = 0, peak = 0, wallPeak = 0, where = "", openBoxFace: string | undefined;
  const see = (v: number, onWall: boolean, at: string) => { if (v > peak) { peak = v; where = at; } if (onWall && v > wallPeak) wallPeak = v; };
  for (const [key, s] of slotOf) {
    const b = [key & 1023, (key >>> 10) & 1023, (key >>> 20) & 1023];
    for (let l = 0; l < BRICK_CELLS; l++) {
      if (!(diag[BRICK_STRIDE * s + l]! > 0)) continue;
      const c = [8 * b[0]! + (l & 7), 8 * b[1]! + ((l >> 3) & 7), 8 * b[2]! + (l >> 6)];
      if (outside(c)) { pastBox++; continue; }
      hRows++;
      const kinds = fc[BRICK_STRIDE * (7 * s + 6) + l]!;
      for (let f = 0; f < 6; f++) {
        const kind = ((kinds >>> (4 * f)) & 15) - 1, a = f >> 1, sign = f & 1 ? 1 : -1;
        const onPlane = sign < 0 ? c[a] === 0 : c[a] === box[a]! - 1;
        if (onPlane) { boxFaces++; if (kind !== NP_K.WALL && openBoxFace === undefined) openBoxFace = `face f${f} of ${c}: kind ${kind}`; }
        else if (kind === NP_K.WALL) strayWalls++;
        if (kind === NP_K.NEUMANN || kind === NP_K.CLOSED) continue;
        // The face's U: f = 2a is c's low face (this slot), f = 2a + 1 the low face of c + e (its brick's slot).
        const q = [...c]; if (sign > 0) q[a]! += 1;
        const qs = sign > 0 ? fSlot(q) : s;
        if (qs !== undefined) see(Math.abs(u[BRICK_STRIDE * (qs + a * L.cap.f) + local(q)]!), onPlane, `h face f${f} of ${c} kind ${kind}`);
      }
    }
  }
  // The native rows: tiles of page 0, liquid (C DIAG > 0) and 4h-owned; their faces between two liquid native rows
  // and on the box planes (a tile's low face along a; the high plane's face is the low face of the tile past it).
  const T = L.lattice.map(v => v >> 2), boxT = box.map(v => v >> 2);
  const native = (p: number[]) => p.every((v, a) => v >= 0 && v < T[a]!) && !tileH(tile(p)) && cdiag[BRICK_STRIDE * cs + tile(p)]! > 0;
  for (let i = 0; i < BRICK_CELLS; i++) {
    const p = [i & 7, (i >> 3) & 7, i >> 6]; if (!native(p)) continue;
    if (p.some((v, a) => v >= boxT[a]!)) { pastBox++; continue; }
    nativeRows++;
    for (let a = 0; a < 3; a++) {
      const m = [...p]; m[a]! -= 1;
      if (p[a] === 0 || native(m)) see(Math.abs(uc[BRICK_STRIDE * (cs + a * L.cap.c) + i]!), p[a] === 0, `4h face a${a} of tile ${p}`);
      const up = [...p]; up[a]! += 1;
      if (up[a] === boxT[a] && up[a]! < T[a]!) see(Math.abs(uc[BRICK_STRIDE * (cs + a * L.cap.c) + tile(up)]!), true, `4h box face a${a} above tile ${p}`);
    }
  }
  return { hRows, nativeRows, pastBox, boxFaces, strayWalls, ...(openBoxFace ? { openBoxFace } : {}), peak, wallPeak, where, bound: 2 * MG_TOLERANCE * h_m };
}

/** paddedPressure, asserted: no row past the box, the walls at the box once, and for still water (`still`) the peak
 * within the solve's bound. Returns the numbers for the log. */
export async function assertPaddedPressure(e: BrickEngine, h_m: number, o: { readonly step?: number; readonly still?: boolean } = {}): Promise<PaddedPressure> {
  const step = o.step ?? 0;
  const m = await paddedPressure(e, h_m);
  assert.ok(e.layout.box.some((v, a) => v !== e.layout.lattice[a]), "the layout is padded");
  assert.equal(m.pastBox, 0, `step ${step}: ${m.pastBox} pressure rows past the box ${e.layout.box.join("x")}`);
  assert.equal(m.openBoxFace, undefined, `step ${step}: a liquid row's box face is not WALL (${m.openBoxFace})`);
  assert.equal(m.strayWalls, 0, `step ${step}: ${m.strayWalls} WALL faces inside the box (a wall at the storage edge, or twice)`);
  if (o.still) assert.ok(m.peak <= m.bound, `step ${step}: peak |U| ${m.peak} m/s at ${m.where}, over the solve's bound ${m.bound}`);
  return m;
}
