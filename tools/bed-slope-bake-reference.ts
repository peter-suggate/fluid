/** The bed-slope bake's host side of its proofs (lib/methods/uniform/uniform-bed-slope-bake.ts):
 * an emulation of the kernels, the shapes a lane is held to, and the edits a
 * region bake is held to. Nothing here ships: the unit test settles the rule
 * on the emulation (tests/uniform-bed-slope-bake-rule.test.ts), and the Dawn
 * probe holds the kernels' lane to the emulation's, word for word
 * (tools/probe-bed-slope-bake.ts).
 *
 * The emulation runs the kernels' launches in their order, one function a
 * kernel, in the same integers: nosings in half cells, distances in 1/128
 * cell, heights in 1/4096 cell, gradients in 1/8192 cell a cell. */
import assert from "node:assert/strict";
import { createNbflipWaterSlide, createNbflipWaterSlideLong, nbflipSlideCentreline, nbflipSlideLongCentreline, NBFLIP_SLIDE_END_RAD, NBFLIP_SLIDE_LONG_LENGTH } from "../lib/core/nbflip-water-slide-scene";
import { SOLID_MASK_DIRECTORY_WORDS, SOLID_MASK_EMPTY, SOLID_MASK_WALL, SolidOccupancyMask, solidOccupancyMaskLayout, solidOccupancyMaskVoxel, type SolidOccupancyMaskLayout } from "../lib/core/solid-occupancy-mask";
import { uniformLatticeAddress } from "../lib/methods/uniform/uniform-lattice-address";
import { uniformSolidMaskPages, uniformSolidScratchLayout } from "../lib/methods/uniform/uniform-mixed-solid.wgsl";
import { fluidSolidWorldForScene } from "../lib/core/solid-world";
import { uniformBedSlopeBakeLayout, type UniformBedSlopeBakeBox, type UniformBedSlopeBakeLayout } from "../lib/methods/uniform/uniform-bed-slope-bake";

export type Size = readonly [number, number, number];
export interface Mask { readonly size: Size; readonly words: Uint32Array }
export const HEADER = 4, NONE = 0xffffffff, POINT = 0xffffff;
export const layoutOf = (size: Size, passes?: number, layers?: number): UniformBedSlopeBakeLayout =>
  uniformBedSlopeBakeLayout({ size, laneBaseWord: uniformSolidScratchLayout(size).bedSlopeBaseWord, passes, layers });
export const cellOf = (size: Size, x: number, y: number, z: number) => (x + 1) + (size[0] + 2) * ((y + 1) + (size[1] + 2) * (z + 1));
export const solidIn = (m: Mask) => (x: number, y: number, z: number) => { const i = cellOf(m.size, x, y, z); return (m.words[HEADER + (i >>> 5)]! >>> (i & 31) & 1) === 1; };
/** A mask from a predicate over the lattice and its halo. */
export function maskOf(size: Size, solid: (x: number, y: number, z: number) => boolean): Mask {
  const [nx, ny, nz] = size, words = new Uint32Array(HEADER + Math.ceil((nx + 2) * (ny + 2) * (nz + 2) / 32));
  words.set([0x53565731, nx + 2, ny + 2, nz + 2]);
  for (let z = -1; z <= nz; z++) for (let y = -1; y <= ny; y++) for (let x = -1; x <= nx; x++) {
    if (!solid(x, y, z)) continue;
    const i = cellOf(size, x, y, z); words[HEADER + (i >>> 5)]! |= 1 << (i & 31);
  }
  return { size, words };
}
/** A rig's mask image for a lattice (SolidOccupancyMask's layout): the lane after the directory, `tail` words after the lane, then a slot a page of the lattice. */
export const imageLayoutOf = (size: Size, tail = 0): SolidOccupancyMaskLayout =>
  solidOccupancyMaskLayout(tail, Math.ceil(size[0] / 32) * Math.ceil(size[1] / 32) * Math.ceil(size[2] / 32), uniformSolidScratchLayout(size).bedSlopeWords);
/** `mask` as the mask image the kernels read: a slot for every page of the
 * lattice, the pages over it open or walls as the mask's halo is over its
 * first column, and every other page a wall. An image has no halo of its own:
 * a mask whose halo is not what those pages give is refused. */
export function imageOf(mask: Mask, tail = 0): Uint32Array<ArrayBuffer> {
  const [nx, ny, nz] = mask.size, px = Math.ceil(nx / 32), py = Math.ceil(ny / 32), pz = Math.ceil(nz / 32), solid = solidIn(mask);
  const layout = imageLayoutOf(mask.size, tail), words = new Uint32Array(layout.words).fill(SOLID_MASK_WALL, 0, SOLID_MASK_DIRECTORY_WORDS);
  for (let z = 0; z < pz; z++) for (let x = 0; x < px; x++) {
    if (!solid(0, ny, 0)) words[x | py << 6 | z << 12] = SOLID_MASK_EMPTY;
    for (let y = 0; y < py; y++) {
      const unit = layout.firstSlot + x + px * (y + py * z); words[x | y << 6 | z << 12] = unit;
      for (let k = 32 * z; k < Math.min(32 * z + 32, nz + 1); k++) for (let j = 32 * y; j < Math.min(32 * y + 32, ny + 1); j++) for (let i = 32 * x; i < Math.min(32 * x + 32, nx + 1); i++) {
        if (solid(i, j, k)) words[(unit << 10) + (j & 31) + 32 * (k & 31)]! |= 1 << (i & 31);
      }
    }
  }
  for (let z = -1; z <= nz; z++) for (let y = -1; y <= ny; y++) for (let x = -1; x <= nx; x++) {
    if (solidOccupancyMaskVoxel(words, x, y, z) !== solid(x, y, z)) throw new RangeError(`A mask image cannot hold cell ${x},${y},${z} of a ${mask.size.join("x")} mask ${solid(x, y, z) ? "solid" : "open"}`);
  }
  return words;
}
/** A bed of columns in a box: solid below bed(x, z), walls and a floor in the halo, open above. */
export const bedMask = (size: Size, bed: (x: number, z: number) => number) => maskOf(size, (x, y, z) => x < 0 || z < 0 || x >= size[0] || z >= size[2] || y < bed(x, z));
/** `mask` with the cells of `box` made solid or open. */
export function edited(mask: Mask, box: UniformBedSlopeBakeBox, solid: boolean): Mask {
  const words = mask.words.slice();
  for (let z = box.minimum[2]; z < box.maximumExclusive[2]; z++) for (let y = box.minimum[1]; y < box.maximumExclusive[1]; y++) for (let x = box.minimum[0]; x < box.maximumExclusive[0]; x++) {
    const i = cellOf(mask.size, x, y, z);
    if (solid) words[HEADER + (i >>> 5)]! |= 1 << (i & 31); else words[HEADER + (i >>> 5)]! &= ~(1 << (i & 31));
  }
  return { size: mask.size, words };
}

// ---- The host emulation: one function a kernel, in the kernels' integers. ----
// It shares no code with the kernels and keeps its state unpacked (a link is
// a slot and a step, a drop flag a byte), so the kernels' packing is checked
// by it and not copied into it.
export interface Reference {
  readonly lane: Uint32Array;
  /** By haloed cell: a floor cell's bed height above its own top face, in cells; 0 for a cell with none. */
  readonly rise: Float32Array;
  readonly floors: number; readonly maxLayers: number; readonly dropped: number;
}
const DX = [-1, 1, 0, 0], DZ = [0, 0, -1, 1];
interface Emulation {
  readonly l: UniformBedSlopeBakeLayout; readonly solid: (x: number, y: number, z: number) => boolean;
  /** x0, z0 and the counts of the window, then of the columns written. */
  readonly region: Uint32Array;
  /** By slot: the floor cell's y, -1 for none. */
  readonly level: Int32Array;
  /** By slot and direction: the neighbour's slot, and the step in y (-2 for no link). */
  readonly linkSlot: Int32Array; readonly linkStep: Int8Array;
  /** By slot: the nearest nosing of the levels y-1, y, y+1, y+2, and the tread's drop flag; from and to of a spread. */
  nosing: Uint32Array; nosingNext: Uint32Array; drop: Uint8Array; dropNext: Uint8Array;
  /** By slot: the bed height in 1/4096 cell, NONE for none; from and to of a relaxation. */
  height: Uint32Array; heightNext: Uint32Array;
  floors: number; maxLayers: number; dropped: number;
}
/** The slots of the window's columns that hold a floor cell, column by column as a kernel walks them. */
function* slots(e: Emulation): Generator<[s: number, x: number, z: number]> {
  const [nx] = e.l.size, K = e.l.layers, r = e.region;
  for (let z = r[1]!; z < r[1]! + r[3]!; z++) for (let x = r[0]!; x < r[0]! + r[2]!; x++) for (let k = 0; k < K; k++) {
    const s = K * (x + nx * z) + k;
    if (e.level[s]! < 0) break;
    yield [s, x, z];
  }
}
const better = (cand: number, held: number, cx: number, cz: number) => {
  if (cand === POINT) return false;
  if (held === POINT) return true;
  const dc = ((cand & 4095) - cx) ** 2 + ((cand >>> 12) - cz) ** 2, dh = ((held & 4095) - cx) ** 2 + ((held >>> 12) - cz) ** 2;
  return dc < dh || (dc === dh && cand < held);
};
const root = (v: number) => { let q = Math.floor(Math.sqrt(v)); while (q * q > v) q--; while ((q + 1) * (q + 1) <= v) q++; return q; };
/** bsLayers. */
function emulateLayers(e: Emulation): void {
  const [nx, ny] = e.l.size, K = e.l.layers, r = e.region;
  for (let z = r[1]!; z < r[1]! + r[3]!; z++) for (let x = r[0]!; x < r[0]! + r[2]!; x++) {
    let k = 0, below = e.solid(x, -1, z);
    for (let y = 0; y < ny; y++) {
      const here = e.solid(x, y, z);
      if (!here && below) { if (k < K) e.level[K * (x + nx * z) + k] = y; else e.dropped++; k++; e.floors++; }
      below = here;
    }
    e.maxLayers = Math.max(e.maxLayers, k);
    for (; k < K; k++) e.level[K * (x + nx * z) + k] = -1;
  }
}
/** bsLinks. */
function emulateLinks(e: Emulation): void {
  const [nx, , nz] = e.l.size, K = e.l.layers, r = e.region;
  const inWindow = (x: number, z: number) => x >= r[0]! && x < r[0]! + r[2]! && z >= r[1]! && z < r[1]! + r[3]!;
  for (const [s, x, z] of slots(e)) {
    const y = e.level[s]!, cx = 2 * x + 1, cz = 2 * z + 1;
    e.nosing.fill(POINT, 4 * s, 4 * s + 4); e.drop[s] = 0;
    for (let d = 0; d < 4; d++) {
      const xn = x + DX[d]!, zn = z + DZ[d]!, openBeside = !e.solid(xn, y, zn);
      let step = -2;
      if (xn >= 0 && xn < nx && zn >= 0 && zn < nz && inWindow(xn, zn)) for (let j = 0; j < K; j++) {
        const yn = e.level[K * (xn + nx * zn) + j]!;
        if (yn < 0) break;
        if (yn === y || (yn === y + 1 && !e.solid(x, y + 1, z)) || (yn === y - 1 && openBeside)) { step = yn - y; e.linkSlot[4 * s + d] = K * (xn + nx * zn) + j; break; }
      }
      e.linkStep[4 * s + d] = step;
      const point = (cx + DX[d]!) | (cz + DZ[d]!) << 12;
      if (step === -1 && better(point, e.nosing[4 * s + 1]!, cx, cz)) e.nosing[4 * s + 1] = point;
      if (step === 1 && better(point, e.nosing[4 * s + 2]!, cx, cz)) e.nosing[4 * s + 2] = point;
      if (openBeside && step !== -1 && !e.solid(xn, y - 1, zn)) e.drop[s] = 1;
    }
  }
}
/** bsSpread, one pass. True when it changed a nosing or a flag. */
function emulateSpread(e: Emulation): boolean {
  const a = e.nosing, b = e.nosingNext; let changed = false;
  for (const [s, x, z] of slots(e)) {
    const cx = 2 * x + 1, cz = 2 * z + 1; let drop = e.drop[s]!;
    for (let i = 0; i < 4; i++) b[4 * s + i] = a[4 * s + i]!;
    for (let d = 0; d < 4; d++) {
      const step = e.linkStep[4 * s + d]!, n = e.linkSlot[4 * s + d]!;
      if (step === -2) continue;
      for (let i = 0; i < 4; i++) { const j = i - step; if (j >= 0 && j <= 3 && better(a[4 * n + j]!, b[4 * s + i]!, cx, cz)) b[4 * s + i] = a[4 * n + j]!; }
      if (step === 0) drop |= e.drop[n]!;
    }
    e.dropNext[s] = drop;
    if (drop !== e.drop[s]) changed = true;
    for (let i = 0; i < 4; i++) if (b[4 * s + i] !== a[4 * s + i]) changed = true;
  }
  [e.nosing, e.nosingNext] = [b, a]; [e.drop, e.dropNext] = [e.dropNext, e.drop];
  return changed;
}
/** bsHeight. */
function emulateHeights(e: Emulation): void {
  const l = e.l;
  for (const [s, x, z] of slots(e)) {
    const cx = 2 * x + 1, cz = 2 * z + 1, reach = (p: number) => root((((p & 4095) - cx) ** 2 + ((p >>> 12) - cz) ** 2) * 4096);
    const below = e.nosing[4 * s]!, own = e.nosing[4 * s + 1]!, up = e.nosing[4 * s + 2]!, above = e.nosing[4 * s + 3]!, level = (e.level[s]! + 1) * 4096;
    let height = NONE;
    if (own !== POINT && up !== POINT) {
      const d = reach(own), u = reach(up);
      if (u + d >= l.minRun) height = level + Math.floor((d * 8192 + u + d) / (2 * (u + d)));
    } else if (own !== POINT && below !== POINT) {
      const d = reach(own), run = reach(below) - d;
      if (run >= l.minRun && 4 * d <= (4 + l.band4) * run) height = level + Math.floor((d * 8192 + run) / (2 * run));
    } else if (up !== POINT && above !== POINT && e.drop[s]) {
      const u = reach(up), run = reach(above) - u;
      if (run >= l.minRun && 4 * u <= (4 + l.band4) * run) height = level + 4096 - Math.floor((u * 8192 + run) / (2 * run));
    }
    e.height[s] = height;
  }
}
const side = (e: Emulation, s: number, d: number) => e.linkStep[4 * s + d] === -2 ? NONE : e.height[e.linkSlot[4 * s + d]!]!;
/** bsRelaxOut or bsRelaxBack, one pass. */
function emulateRelax(e: Emulation): void {
  for (const [s] of slots(e)) {
    const h = e.height[s]!; let sum = 0, axes = 0;
    if (h !== NONE) for (let axis = 0; axis < 2; axis++) {
      const low = side(e, s, 2 * axis), high = side(e, s, 2 * axis + 1);
      if (low !== NONE && high !== NONE) { sum += low + high; axes++; }
    }
    e.heightNext[s] = axes === 0 ? h : Math.floor((2 * axes * h + sum + 2 * axes) / (4 * axes));
  }
  [e.height, e.heightNext] = [e.heightNext, e.height];
}
/** bsResolve, into the lane's words; `rise` takes the heights it read. */
function emulateResolve(e: Emulation, lane: Uint32Array, rise: Float32Array): void {
  const l = e.l, [nx, ny] = l.size, K = l.layers, r = e.region;
  const put = (i: number, entry: number) => { lane[i >>> 1] = ((lane[i >>> 1]! & ~(0xffff << (16 * (i & 1)))) | entry << (16 * (i & 1))) >>> 0; };
  const byte = (v: number) => Math.sign(v) * Math.min(127, Math.floor((254 * Math.abs(v) + 4096) / 8192)) & 0xff;
  for (let z = r[5]!; z < r[5]! + r[7]!; z++) for (let x = r[4]!; x < r[4]! + r[6]!; x++) {
    for (let y = 0; y < ny; y++) put(cellOf(l.size, x, y, z), 0);
    for (let k = 0; k < K; k++) {
      const s = K * (x + nx * z) + k;
      if (e.level[s]! < 0) break;
      const h = e.height[s]!;
      if (h === NONE) continue;
      const i = cellOf(l.size, x, e.level[s]!, z), g = [0, 0];
      rise[i] = (h - (e.level[s]! + 1) * 4096) / 4096;
      for (let axis = 0; axis < 2; axis++) {
        const low = side(e, s, 2 * axis), high = side(e, s, 2 * axis + 1);
        g[axis] = low !== NONE && high !== NONE ? high - low : high !== NONE ? 2 * (high - h) : low !== NONE ? 2 * (h - low) : 0;
      }
      const m2 = g[0]! * g[0]! + g[1]! * g[1]!;
      if (m2 < l.minSlope2 || m2 > l.maxSlope2) continue;
      if (m2 > 4096 * 4096) { let q = root(m2); if (q * q < m2) q++; g[0] = Math.trunc(g[0]! * 4096 / q); g[1] = Math.trunc(g[1]! * 4096 / q); }
      put(i, byte(g[0]!) | byte(g[1]!) << 8);
    }
  }
}
/** A bake as UniformBedSlopeBake launches it: the whole lattice into a new
 * lane, or `region` (uniformBedSlopeBakeRegion's eight words) into `into`. */
export function reference(mask: Mask, l: UniformBedSlopeBakeLayout, into?: Uint32Array, region?: Uint32Array): Reference {
  const [nx, , nz] = l.size, n = nx * nz * l.layers;
  const e: Emulation = { l, solid: solidIn(mask), region: region ?? Uint32Array.of(0, 0, nx, nz, 0, 0, nx, nz), level: new Int32Array(n).fill(-1),
    linkSlot: new Int32Array(4 * n).fill(-1), linkStep: new Int8Array(4 * n).fill(-2), nosing: new Uint32Array(4 * n).fill(POINT), nosingNext: new Uint32Array(4 * n).fill(POINT),
    drop: new Uint8Array(n), dropNext: new Uint8Array(n), height: new Uint32Array(n).fill(NONE), heightNext: new Uint32Array(n).fill(NONE), floors: 0, maxLayers: 0, dropped: 0 };
  emulateLayers(e); emulateLinks(e);
  // The kernels run every pass. A spread that changes nothing is a fixed point, so stopping there is the same state.
  for (let pass = 0; pass < l.passes; pass++) if (!emulateSpread(e)) break;
  emulateHeights(e);
  for (let pass = 0; pass < l.relax; pass++) emulateRelax(e);
  const lane = into ?? new Uint32Array(l.laneWords), rise = new Float32Array(l.cells);
  emulateResolve(e, lane, rise);
  return { lane, rise, floors: e.floors, maxLayers: e.maxLayers, dropped: e.dropped };
}

// ---- Reading a lane. ----
export type Slope = (x: number, y: number, z: number) => readonly [number, number];
export const slopeIn = (size: Size, lane: Uint32Array): Slope => (x, y, z) => { const i = cellOf(size, x, y, z), entry = lane[i >>> 1]! >>> (16 * (i & 1)); return [entry << 24 >> 24, entry << 16 >> 24]; };
/** Every entry outside the floor cells (open over solid, inside the lattice) is zero. Returns the floor cells and those with a slope. */
export function floorOnly(name: string, mask: Mask, lane: Uint32Array): [floors: number, sloped: number] {
  const [nx, ny, nz] = mask.size, solid = solidIn(mask), slope = slopeIn(mask.size, lane); let floors = 0, sloped = 0;
  for (let z = -1; z <= nz; z++) for (let y = -1; y <= ny; y++) for (let x = -1; x <= nx; x++) {
    const [sx, sz] = slope(x, y, z), inside = x >= 0 && y >= 0 && z >= 0 && x < nx && y < ny && z < nz, floor = inside && !solid(x, y, z) && solid(x, y - 1, z);
    if (floor) floors++;
    if (sx === 0 && sz === 0) continue;
    if (!floor) assert.fail(`${name}: cell ${x},${y},${z} is no floor cell and holds ${sx},${sz}`);
    sloped++;
  }
  return [floors, sloped];
}
const expectZero = (name: string, mask: Mask, lane: Uint32Array) => { const [floors, sloped] = floorOnly(name, mask, lane); assert.equal(sloped, 0, `${name}: ${sloped} of ${floors} floor cells hold a slope`); return `${floors} floor cells, all level`; };

// ---- The shapes. ----
export interface Shape { readonly name: string; readonly mask: Mask; readonly passes?: number; check(lane: Uint32Array, ref: Reference): string }
export const BED_SLOPE_RAMP: Size = [128, 32, 32], BED_SLOPE_DRUM: Size = [160, 96, 160], BED_SLOPE_LONG: Size = [512, 96, 128];
const RAMP = BED_SLOPE_RAMP, RAMP_Z: Size = [32, 32, 128], SQUARE: Size = [96, 32, 96];
/** The probe ramp of the investigation: a staircase falling along `s` to a pit two cells up, 96 cells in. */
export const rampBed = (run: number) => (s: number) => s < 96 ? Math.max(2, Math.floor(17 - (s + 0.5) / run)) : 2;
export interface Stats { cells: number; sloped: number; mean: number; p95: number; max: number; ratio: number }
/** Against an expected gradient by cell: the angle between it and the lane, and the ratio of their sizes. */
export function compare(size: Size, lane: Uint32Array, cells: Iterable<readonly [number, number, number, number, number]>): Stats {
  const slope = slopeIn(size, lane), angles: number[] = []; let n = 0, sum = 0, ratio = 0;
  for (const [x, y, z, gx, gz] of cells) {
    const [sx, sz] = slope(x, y, z); n++;
    if (sx === 0 && sz === 0) continue;
    const m = Math.hypot(sx, sz) / 254, e = Math.hypot(gx, gz), angle = Math.acos(Math.max(-1, Math.min(1, (sx * gx + sz * gz) / (254 * m * e)))) * 180 / Math.PI;
    angles.push(angle); sum += angle; ratio += m / e;
  }
  angles.sort((p, q) => p - q);
  return { cells: n, sloped: angles.length, mean: sum / angles.length, p95: angles[Math.floor(0.95 * angles.length)] ?? 0, max: angles[angles.length - 1] ?? 0, ratio: ratio / angles.length };
}
export const statsText = (s: Stats) => `${s.cells} cells, ${(100 * s.sloped / s.cells).toFixed(1)}% sloped, angle off mean ${s.mean.toFixed(1)} p95 ${s.p95.toFixed(1)} max ${s.max.toFixed(1)} deg, size ratio ${s.ratio.toFixed(3)}`;
/** Riser pairs (linked floor cells a cell apart) and those whose bed heights still differ by a full cell. */
export function riserPairs(mask: Mask, ref: Reference, lowest = 0): [pairs: number, full: number] {
  const [nx, ny, nz] = mask.size, solid = solidIn(mask); let pairs = 0, full = 0;
  for (let z = 0; z < nz; z++) for (let y = lowest; y < ny - 1; y++) for (let x = 0; x < nx; x++) {
    if (solid(x, y, z) || !solid(x, y - 1, z)) continue;
    for (let d = 0; d < 4; d++) {
      const xn = x + DX[d]!, zn = z + DZ[d]!;
      if (xn < 0 || zn < 0 || xn >= nx || zn >= nz || !solid(xn, y, zn) || solid(xn, y + 1, zn) || solid(x, y + 1, z)) continue;
      pairs++;
      if (Math.abs(1 + ref.rise[cellOf(mask.size, xn, y + 1, zn)]! - ref.rise[cellOf(mask.size, x, y, z)]!) >= 1 - 1e-6) full++;
    }
  }
  return [pairs, full];
}
const sceneMasks = new Map<string, Mask>();
/** The drum's or the long slide's mask, built on the host as the solver builds its own. */
export function bedSlopeSceneMask(scene: "drum" | "long"): Mask {
  let mask = sceneMasks.get(scene);
  if (!mask) {
    const size = scene === "drum" ? BED_SLOPE_DRUM : BED_SLOPE_LONG, made = scene === "drum" ? createNbflipWaterSlide() : createNbflipWaterSlideLong();
    const m = new SolidOccupancyMask(uniformSolidMaskPages(uniformLatticeAddress(size, made.container.top === "open")));
    m.update(fluidSolidWorldForScene(made));
    sceneMasks.set(scene, mask = maskOf(size, (x, y, z) => solidOccupancyMaskVoxel(m.words, x, y, z)));
  }
  return mask;
}
/** The shapes whose names `wanted` takes, each with the expectation its lane is held to. */
export function bedSlopeShapes(wanted: (name: string) => boolean = () => true): Shape[] {
  const shapes: Shape[] = [];
  for (const run of [20, 16, 4]) for (const [label, size, along, uphill] of [
    ["+x", RAMP, (x: number, _z: number) => x, [-1, 0]], ["-x", RAMP, (x: number, _z: number) => 127 - x, [1, 0]],
    ["+z", RAMP_Z, (_x: number, z: number) => z, [0, -1]], ["-z", RAMP_Z, (_x: number, z: number) => 127 - z, [0, 1]],
  ] as const) {
    const bed = rampBed(run), mask = bedMask(size, (x, z) => bed(along(x, z))), want = Math.round(254 / run);
    shapes.push({ name: `staircase 1:${run} falling along ${label}`, mask, check(lane) {
      const slope = slopeIn(size, lane); floorOnly(this.name, mask, lane); let treads = 0;
      for (let z = 0; z < size[2]; z++) for (let x = 0; x < size[0]; x++) {
        const y = bed(along(x, z)), [sx, sz] = slope(x, y, z);
        // The pit and the run-out the staircase ends on are level.
        if (y === 2) { assert.deepEqual([sx, sz], [0, 0], `${this.name}: level cell ${x},${y},${z}`); continue; }
        treads++; assert.deepEqual([sx, sz], [want * uphill[0], want * uphill[1]], `${this.name}: tread cell ${x},${y},${z}`);
      }
      return `${treads} tread cells all ${want * uphill[0]},${want * uphill[1]} (1/254); pit and run-out level`;
    } });
  }
  for (const [label, wall] of [["nine-cell", (z: number) => z < 9 || z >= 23], ["one-cell", (z: number) => z === 9 || z === 22]] as const) {
    const bed = rampBed(20), mask = bedMask(RAMP, (x, z) => wall(z) ? 26 : bed(x));
    shapes.push({ name: `channel with ${label} walls, 1:20`, mask, check(lane) {
      const slope = slopeIn(RAMP, lane); floorOnly(this.name, mask, lane); let treads = 0, tops = 0;
      for (let z = 0; z < 32; z++) for (let x = 0; x < 128; x++) {
        if (wall(z)) { tops++; assert.deepEqual(slope(x, 26, z), [0, 0], `${this.name}: wall top ${x},26,${z}`); continue; }
        const y = bed(x); if (y > 2) treads++;
        assert.deepEqual(slope(x, y, z), y > 2 ? [-13, 0] : [0, 0], `${this.name}: floor cell ${x},${y},${z}`);
      }
      return `${treads} channel-floor cells all -13,0; ${tops} wall-top cells level`;
    } });
  }
  const level = (name: string, size: Size, bed: (x: number, z: number) => number) => { const mask = bedMask(size, bed); shapes.push({ name, mask, check(lane) { return expectZero(name, mask, lane); } }); };
  level("level floor 128x32", RAMP, () => 5);
  level("level floor 96x96", SQUARE, () => 5);
  level("lone step (one riser across a level floor)", RAMP, x => x < 64 ? 6 : 5);
  level("lone block (6x6, one cell high)", SQUARE, (x, z) => x >= 40 && x < 46 && z >= 50 && z < 56 ? 6 : 5);
  level("pit edge (eight cells down)", RAMP, x => x < 64 ? 10 : 2);
  level("two-cell kerb", RAMP, x => x < 64 ? 5 : 7);
  level("basin floor enclosed by walls", SQUARE, (x, z) => x < 20 || x >= 76 || z < 20 || z >= 76 ? 12 : 4);
  // Steeper than a staircase of run 2 holds: a run of one cell is a wall of steps.
  level("staircase 1:1", RAMP, x => Math.max(2, 30 - x));
  // A platform with a riser up at one end and a pit at the other, and no staircase above it.
  level("lone platform between a riser and a drop", RAMP, x => x < 40 ? 7 : x < 70 ? 6 : 1);
  {
    const bed = (x: number) => Math.max(2, 30 - Math.floor(x / 2)), mask = bedMask(RAMP, x => bed(x));
    shapes.push({ name: "staircase 1:2", mask, check(lane) {
      const slope = slopeIn(RAMP, lane); floorOnly(this.name, mask, lane); let treads = 0;
      for (let z = 0; z < 32; z++) for (let x = 0; x < 128; x++) { const y = bed(x); if (y > 2) treads++; assert.deepEqual(slope(x, y, z), y > 2 ? [-127, 0] : [0, 0], `${this.name}: cell ${x},${y},${z}`); }
      return `${treads} tread cells all -127,0: half a cell a cell; run-out level`;
    } });
  }
  {
    // Two risers 30 cells apart are a staircase of run 30 to the rule: the platform between them slopes, and so does
    // the upper floor within a run and a quarter of its nosing. The lower floor has nowhere to drain to and is level.
    const bed = (x: number) => x < 60 ? 7 : x < 90 ? 6 : 5, mask = bedMask(RAMP, x => bed(x));
    shapes.push({ name: "two risers 30 cells apart", mask, check(lane) {
      const slope = slopeIn(RAMP, lane); floorOnly(this.name, mask, lane); let sloped = 0;
      for (let z = 0; z < 32; z++) for (let x = 0; x < 128; x++) {
        // The upper floor's cells from 37.5 cells of the nosing in; the cell on the band's edge is a one-sided difference.
        const want = x >= 90 || x < 22 ? [0, 0] : x === 22 ? undefined : [-8, 0];
        if (want) assert.deepEqual(slope(x, bed(x), z), want, `${this.name}: cell ${x},${z}`);
        if (want?.[0]) sloped++;
      }
      return `${sloped} cells of the platform and the upper floor's last 37 read -8,0 (1 in 30); the rest level`;
    } });
  }
  {
    // Rising along x+z, a cell every 16 along each axis.
    const bed = (x: number, z: number) => 4 + Math.floor((x + z + 1) / 16), mask = bedMask(SQUARE, bed);
    shapes.push({ name: "diagonal staircase, 1:16 along x and along z", mask, check(lane) {
      const slope = slopeIn(SQUARE, lane); floorOnly(this.name, mask, lane); let n = 0, worst = 0, worstSkew = 0, zero = 0;
      for (let z = 24; z < 72; z++) for (let x = 24; x < 72; x++) {
        const [sx, sz] = slope(x, bed(x, z), z); n++;
        if (sx === 0 && sz === 0) { zero++; continue; }
        worst = Math.max(worst, Math.abs(sx - 16), Math.abs(sz - 16)); worstSkew = Math.max(worstSkew, Math.abs(sx - sz));
      }
      assert.equal(zero, 0, `${this.name}: ${zero} of ${n} interior cells are level`);
      assert.ok(worst <= 3 && worstSkew <= 4, `${this.name}: components off 16 by up to ${worst}, apart by up to ${worstSkew}`);
      return `${n} interior cells: components within ${worst} of 16, within ${worstSkew} of each other`;
    } });
  }
  {
    // Square terraces eight cells wide falling to a 16x16 floor: nothing drains out of it.
    const ring = (x: number, z: number) => Math.min(5, Math.floor(Math.max(Math.abs(x - 47.5), Math.abs(z - 47.5)) / 8)), bed = (x: number, z: number) => 4 + ring(x, z);
    const mask = bedMask(SQUARE, bed);
    shapes.push({ name: "terraced basin", mask, check(lane) {
      const slope = slopeIn(SQUARE, lane); floorOnly(this.name, mask, lane); let floor = 0, terrace = 0, straight = 0;
      for (let z = 0; z < 96; z++) for (let x = 0; x < 96; x++) {
        const k = ring(x, z), [sx, sz] = slope(x, bed(x, z), z), ox = x - 47.5, oz = z - 47.5;
        if (k === 0) { floor++; assert.deepEqual([sx, sz], [0, 0], `${this.name}: basin floor ${x},${z}`); continue; }
        // Every terrace, the rim included (a top landing one run wide), rises away from the floor.
        terrace++; assert.ok(sx * ox + sz * oz > 0, `${this.name}: terrace cell ${x},${z} holds ${sx},${sz}`);
        // Away from the corners a terrace is a straight staircase of run 8.
        if (Math.abs(oz) < 2) { straight++; assert.deepEqual([sx, sz], [Math.sign(ox) * 32, 0], `${this.name}: straight terrace cell ${x},${z}`); }
      }
      return `${floor} floor cells level; ${terrace} terrace cells uphill outwards, ${straight} on the straight exactly 32`;
    } });
  }
  {
    // A channel around a ring, 12 cells wide, rising 16 cells a turn: treads 11 to 16 cells long.
    const polar = (x: number, z: number) => { const dx = x + 0.5 - 48, dz = z + 0.5 - 48; return [Math.hypot(dx, dz), (Math.atan2(dz, dx) + 2 * Math.PI) % (2 * Math.PI)] as const; };
    const inRing = (x: number, z: number) => { const [radius] = polar(x, z); return radius >= 28 && radius < 40; };
    const bed = (x: number, z: number) => inRing(x, z) ? 4 + Math.floor(16 * polar(x, z)[1] / (2 * Math.PI)) : 30, mask = bedMask(SQUARE, bed);
    shapes.push({ name: "ring channel", mask, check(lane, ref) {
      floorOnly(this.name, mask, lane); const cells: [number, number, number, number, number][] = [];
      for (let z = 0; z < 96; z++) for (let x = 0; x < 96; x++) {
        if (!inRing(x, z)) continue;
        const [radius, angle] = polar(x, z), s = 16 / (2 * Math.PI * radius);
        // The turn's two ends meet at a wall 16 cells high; the treads beside it are a top landing and a last tread.
        if (angle < 2 * Math.PI / 16 || angle > 2 * Math.PI * 15 / 16) continue;
        cells.push([x, bed(x, z), z, -Math.sin(angle) * s, Math.cos(angle) * s]);
      }
      // The lowest tread ends at the wall with nowhere to drain to: level, as a basin's floor is.
      const s = compare(SQUARE, lane, cells), [pairs, full] = riserPairs(mask, ref, 5);
      assert.ok(s.sloped === s.cells && s.mean < 3 && s.p95 < 8 && s.ratio > 0.9 && s.ratio < 1.1, `${this.name}: ${statsText(s)}`);
      assert.equal(full, 0, `${this.name}: ${full} of ${pairs} riser pairs a full cell apart`);
      return `${statsText(s)}; ${pairs} riser pairs above the lowest tread, none a full cell apart`;
    } });
  }
  // The real scenes.
  for (const [name, drum] of [["drum", true], ["long slide", false]] as const) {
    if (!wanted(name)) continue;
    const mask = bedSlopeSceneMask(drum ? "drum" : "long");
    shapes.push({ name, mask, check(lane, ref) {
      const size = mask.size, solid = solidIn(mask), [floors, sloped] = floorOnly(name, mask, lane), [pairs, full] = riserPairs(mask, ref);
      // The channel floor within five cells of the centreline, against the course's tangent and its nominal slope.
      const at = (s: number) => drum ? nbflipSlideCentreline(s) : nbflipSlideLongCentreline(s), end = drum ? NBFLIP_SLIDE_END_RAD : NBFLIP_SLIDE_LONG_LENGTH;
      const step = drum ? 0.002 : 0.25, fall = drum ? 32 : 24, flat = 30, seen = new Set<number>(), cells: [number, number, number, number, number][] = [], slope = slopeIn(size, lane);
      let length = 0; for (let s = 0; s < end; s += step) { const [xa, za] = at(s), [xb, zb] = at(s + step); length += Math.hypot(xb - xa, zb - za); }
      let along = 0, hood = 0, hoodSloped = 0;
      for (let s = 0; s < end; s += step) {
        const [xa, za] = at(s), [xb, zb] = at(s + step), d = Math.hypot(xb - xa, zb - za), tx = (xb - xa) / d, tz = (zb - za) / d; along += d;
        for (let across = -5; across <= 5; across++) {
          const x = Math.floor(xa - tz * across), z = Math.floor(za + tx * across), key = x + size[0] * z;
          if (seen.has(key)) continue; seen.add(key);
          let y = drum ? 0 : 9; while (y < size[1] && !(solid(x, y - 1, z) && !solid(x, y, z))) y++;
          if (y >= size[1]) continue;
          // The last 30 cells of the course are the level floor under the hood.
          if (along > length - flat) { hood++; const [sx, sz] = slope(x, y, z); if (sx !== 0 || sz !== 0) hoodSloped++; continue; }
          const nominal = fall / (length - flat); cells.push([x, y, z, -tx * nominal, -tz * nominal]);
        }
      }
      const s = compare(size, lane, cells);
      assert.equal(ref.dropped, 0, `${name}: ${ref.dropped} floor cells above a column's layers`);
      assert.ok(s.sloped >= 0.99 * s.cells && s.mean < 6 && s.p95 < 12, `${name}: channel floor ${statsText(s)}`);
      assert.ok(full <= (drum ? 0 : 12), `${name}: ${full} of ${pairs} riser pairs a full cell apart`);
      return `${floors} floor cells, ${sloped} sloped; ${pairs} riser pairs, ${full} a full cell apart; channel floor ${statsText(s)}; hood floor ${hoodSloped} of ${hood} sloped; ${ref.maxLayers} floor cells in the deepest column`;
    } });
  }
  return shapes.filter(shape => wanted(shape.name));
}

/** `got` is `want`, word for word; the first cell that differs otherwise. */
export function sameLane(name: string, got: Uint32Array, want: Uint32Array, size: Size): void {
  for (let w = 0; w < want.length; w++) if (got[w] !== want[w]) {
    const plane = (size[0] + 2) * (size[1] + 2), i = 2 * w, z = Math.floor(i / plane), y = Math.floor((i - z * plane) / (size[0] + 2)), x = i - z * plane - y * (size[0] + 2);
    let count = 0; for (let v = w; v < want.length; v++) if (got[v] !== want[v]) count++;
    assert.fail(`${name}: ${count} lane words differ, the first word ${w} (cell ${x - 1},${y - 1},${z - 1}): ${got[w]!.toString(16)} for ${want[w]!.toString(16)}`);
  }
}

// ---- Edits: each a base mask and a run of boxes made solid or open, a region bake after each step. ----
export type Step = readonly (readonly [UniformBedSlopeBakeBox, boolean])[];
export interface Edit { readonly name: string; readonly base: Mask; readonly passes?: number; readonly steps: readonly Step[] }
export const box = (x: number, y: number, z: number, wx: number, wy: number, wz: number): UniformBedSlopeBakeBox => ({ minimum: [x, y, z], maximumExclusive: [x + wx, y + wy, z + wz] });
const LONG_EDIT = "long slide: a slab a cell thick over 16x16 of the channel, then out again";
/** The edit runs whose names `wanted` takes. */
export function bedSlopeEdits(wanted: (name: string) => boolean = () => true): Edit[] {
  const edits: Edit[] = [];
  {
    const bed = rampBed(20);
    edits.push({ name: "staircase 1:20: a block on a tread, a notch in a nosing, a riser moved", base: bedMask(RAMP, x => bed(x)), steps: [
      [[box(44, bed(44), 12, 3, 2, 4), true]], [[box(57, bed(57) - 1, 0, 3, 1, 32), false]], [[box(60, bed(60), 0, 4, 1, 32), true]], [[box(44, bed(44), 12, 3, 2, 4), false]],
    ] });
    // A short spread, so the window is cut by the lattice along neither axis: a field of oblique treads.
    const field = (x: number, z: number) => 4 + Math.floor((2 * x + z) / 12);
    edits.push({ name: "oblique field, 8 passes: edits mid-lattice, in a corner, two to a submit", passes: 8, base: bedMask([128, 40, 128], field), steps: [
      [[box(64, field(64, 64), 64, 2, 2, 2), true]], [[box(1, field(1, 1) - 1, 1, 3, 1, 3), false]],
      [[box(30, field(30, 90), 90, 1, 1, 1), true], [box(100, field(100, 30) - 1, 30, 2, 1, 2), false]],
    ] });
  }
  if (wanted(LONG_EDIT)) edits.push({ name: LONG_EDIT, base: bedSlopeSceneMask("long"), steps: ((): Step[] => {
    const [cx, cz] = nbflipSlideLongCentreline(120), x = Math.floor(cx), z = Math.floor(cz), solid = solidIn(bedSlopeSceneMask("long"));
    let y = 9; while (!(solid(x, y - 1, z) && !solid(x, y, z))) y++;
    return [[[box(x - 8, y, z - 8, 16, 1, 16), true]], [[box(x - 8, y, z - 8, 16, 1, 16), false]]];
  })() });
  return edits.filter(edit => wanted(edit.name));
}
