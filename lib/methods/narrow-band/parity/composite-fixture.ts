// A frozen engine state with mixed tiers, for lane P's composite rows and the composite MGPCG (GPU tests only):
// - a block of 4h tiles below the surface, beside h tiles (seams, with heads), F bricks among them all native;
// - the floor and an x wall under both tiers (separating walls, attached and separated; N_HELD);
// - dry 4h tiles above liquid ones (native ghosts, with the marched theta);
// - warm and cold h rows (cold rows prolong the tiles' starts, C X, with the surface carry).
// The rows are not run here: the caller runs pressureCompositeKernels' rowsF and rowsC.
import { BRICK_STRIDE, CMASK_WORDS, NONE, SEAM } from "../brick-abi";
import { BrickEngine } from "../brick-engine";
import type { PressureSolverFactory } from "../brick-pressure-slot";
import { snapshot } from "../brick-readback";
import { bringUpStages } from "../brick-stages";
import { P_FLAG } from "../pressure-composite.wgsl";

export type V3 = [number, number, number];
export const COMPOSITE_N = 32;
export const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const scale = (s: number, v: V3): V3 => [s * v[0], s * v[1], s * v[2]];
export const shr = (v: V3, n: number): V3 => [v[0] >> n, v[1] >> n, v[2] >> n];
export const keyCoord = (k: number): V3 => [k & 1023, (k >>> 10) & 1023, (k >>> 20) & 1023];
const hash = (c: V3, k: number) => { const s = Math.sin(12.9898 * c[0] + 78.233 * c[1] + 37.719 * c[2] + 4.581 * k) * 43758.5453; return s - Math.floor(s); };
/** The 4h tiles: an interior block from the floor up through the surface, and a block on the x = 0 wall. */
export const coarseTile = (t: V3) => (t[0] >= 2 && t[0] <= 4 && t[1] <= 3 && t[2] >= 2 && t[2] <= 4) || (t[0] <= 1 && t[1] >= 1 && t[1] <= 2 && t[2] <= 1);
/** φ in h: a tilted, rippled surface near y = 11. */
const phiAt = (v: V3) => v[1] - 11.3 + 0.08 * v[0] - 0.05 * v[2] + 0.4 * Math.sin(0.7 * v[0] + 0.3 * v[2]);
const usAt = (c: V3, a: number) => 0.6 * Math.sin(0.37 * c[0] + 0.9 * a) - 0.4 * Math.cos(0.29 * c[1] - 0.21 * c[2] + a) + 0.3 * (hash(c, a) - 0.5);
const xAt = (c: V3) => 0.004 * (11.3 - c[1]) + 0.01 * (hash(c, 7) - 0.5);

export async function compositeFixture(device: GPUDevice, solver: PressureSolverFactory) {
  const N = COMPOSITE_N, h = 1 / 32, rate = 0.07;
  const e = await BrickEngine.create(device, { lattice: [N, N, N], h_m: h, dt_s: 1 / 60, gravity_mps2: [0, -9.81, 0],
    capacities: { f: 200, c: 8, b: 0, k: 0, particles: 1024 }, liquid: [{ lo: [-1, -1, -1], hi: [33, 18.3, 33] }], solver, stages: bringUpStages });
  const s = await snapshot(e), L = e.layout;
  const dir = s.dir.slice();
  const pageSlot = (p: V3) => {
    if (p.some((v, a) => v < 0 || v >= L.window[a]!)) return NONE;
    const cs = dir[L.dir.page + L.pageIndex(p[0], p[1], p[2])]!; return cs === NONE || cs >= SEAM ? NONE : cs;
  };
  const fSlot = (c: V3) => { if (c.some(v => v < 0)) return undefined; const cs = pageSlot(shr(c, 5)); if (cs === NONE) return undefined; const b = shr(c, 3); const fs = dir[L.dir.child + 64 * cs + ((b[0] & 3) | ((b[1] & 3) << 2) | ((b[2] & 3) << 4))]!; return fs === NONE || fs >= SEAM ? undefined : fs; };
  const pages = [...new Set(s.pages.values())];
  for (const cs of pages) {
    const mask = dir.slice(L.dir.cmask + CMASK_WORDS * cs, L.dir.cmask + CMASK_WORDS * (cs + 1));
    const O = keyCoord(dir[L.dir.crec + 32 * cs + 29]!);
    for (let i = 0; i < 512; i++) { const t = add(scale(8, O), [i & 7, (i >> 3) & 7, i >> 6]); if (coarseTile(t)) mask[i >> 5]! &= ~(1 << (i & 31)); }
    dir.set(mask, L.dir.cmask + CMASK_WORDS * cs); e.write("dir", L.dir.cmask + CMASK_WORDS * cs, mask);
  }
  const fBuf = () => new Float32Array(L.cap.f * BRICK_STRIDE);
  const cBuf = () => new Float32Array(L.cap.c * BRICK_STRIDE);
  const F = { PHI: fBuf(), US: [0, 1, 2].map(() => fBuf()), X: fBuf() };
  const C = { PHI: cBuf(), US: [0, 1, 2].map(() => cBuf()), X: cBuf() };
  const fFlags = new Uint32Array(await e.read("phi", L.base("phi", "F", "FLAGS"), L.cap.f * BRICK_STRIDE));
  for (const [key, slot] of s.bricks) {
    const o = scale(8, keyCoord(key));
    for (let l = 0; l < 512; l++) {
      const c = add(o, [l & 7, (l >> 3) & 7, l >> 6]), at = BRICK_STRIDE * slot + l;
      F.PHI[at] = phiAt(c); F.X[at] = xAt(c);
      for (let a = 0; a < 3; a++) F.US[a]![at] = usAt(c, a);
      fFlags[at] = (fFlags[at]! & ~P_FLAG.WARM) | (hash(c, 3) < 0.7 ? P_FLAG.WARM : 0);
    }
  }
  for (const cs of pages) {
    const O = scale(8, keyCoord(dir[L.dir.crec + 32 * cs + 29]!));
    for (let i = 0; i < 512; i++) {
      const T = add(O, [i & 7, (i >> 3) & 7, i >> 6]), at = BRICK_STRIDE * cs + i;
      C.PHI[at] = phiAt(scale(4, T)); C.X[at] = xAt(add(scale(4, T), [2, 2, 2]));
      for (let a = 0; a < 3; a++) C.US[a]![at] = usAt(scale(4, T), a);
    }
  }
  // An entering C slot starts with X = 0 (setInit zeroes every pres channel of it, brick-abi.md §6.1): the caller asserts it.
  const enteredX = new Float32Array((await e.read("pres", L.base("pres", "C", "X"), L.cap.c * BRICK_STRIDE)).buffer);
  e.write("phi", L.base("phi", "F", "PHI"), F.PHI); e.write("pres", L.base("pres", "F", "X"), F.X);
  e.write("phi", L.base("phi", "C", "PHI"), C.PHI); e.write("pres", L.base("pres", "C", "X"), C.X);
  for (let a = 0; a < 3; a++) { e.write("vel", L.base("vel", "F", `US${a}`), F.US[a]!); e.write("vel", L.base("vel", "C", `US${a}`), C.US[a]!); }
  e.write("phi", L.base("phi", "F", "FLAGS"), fFlags);
  e.write("set", 64, new Uint32Array(new Float32Array([rate]).buffer));
  return { e, s, L, N, h, rate, dir, pages, F, C, fFlags, pageSlot, fSlot, enteredX };
}
