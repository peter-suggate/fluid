import { UNIFORM_PAGE_WINDOW } from "./uniform-lattice-address";
import { UNIFORM_EXTRAPOLATION_BLOCK_SWEEPS } from "./webgpu-uniform-velocity-extrapolation.wgsl";

/** Host twin of the block-grid nearest-source pyramid
 * (webgpu-uniform-velocity-extrapolation.wgsl.ts) and of the far lookup that
 * reads its 4h level (umFarValue, uniform-mixed-extension.ts): the same
 * integer addressing and the same f32 arithmetic in the same order, for unit
 * tests and device probes. */

const f = Math.fround;
const FREE = 0xffffffff;
type Vec3 = readonly [number, number, number];

/** The page pool's tail a kernel reads: the 64³ page table, a record per
 * block and the 27 neighbour words per block (uniformBlockNeighbourWords). */
export interface UniformExtrapolationPool {
  readonly table: ArrayLike<number>;
  readonly records: ArrayLike<number>;
  readonly rows: ArrayLike<number>;
}

/** One level's textures, by cell in kernel order: block b's cell l (x in the
 * low bits) at b·edge³ + l. value: 3 a cell. mask: the known components.
 * lower, upper: the source bound points, 3 a component, 9 a cell.
 * selection: the taps a prolongation took, 8 bits a component. */
export interface UniformExtrapolationLevel {
  readonly edge: number;
  readonly value: Float32Array;
  readonly mask: Uint8Array;
  readonly lower: Int32Array;
  readonly upper: Int32Array;
  readonly selection: Uint32Array;
}

export function uniformExtrapolationLevel(blocks: number, edge: number): UniformExtrapolationLevel {
  const cells = blocks * edge ** 3;
  return { edge, value: new Float32Array(3 * cells), mask: new Uint8Array(cells), lower: new Int32Array(9 * cells), upper: new Int32Array(9 * cells), selection: new Uint32Array(cells) };
}

/** sourceWord and sourcePoint: a bound's texel component and its eleventh z bit. */
export function uniformSourceWord(x: number, y: number, z: number): { readonly word: number; readonly high: number } {
  return { word: (x | (y << 11) | ((z & 1023) << 22)) >>> 0, high: z >>> 10 };
}
export function uniformSourcePoint(word: number, high: number): [number, number, number] {
  return [word & 2047, (word >>> 11) & 2047, (word >>> 22) | (high << 10)];
}

const stored = (pool: UniformExtrapolationPool, b: number): boolean => b >= 1 && b <= pool.records.length - 2;
const held = (pool: UniformExtrapolationPool, b: number): boolean => stored(pool, b) && pool.records[b] !== FREE;
const log2 = (edge: number): number => 31 - Math.clz32(edge);

interface Cell { readonly block: number; readonly page: Vec3; readonly local: Vec3; readonly edge: number; readonly index: number }
function levelCell(pool: UniformExtrapolationPool, index: number, edge: number): Cell {
  const k = log2(edge), block = index >>> (3 * k), l = index & ((1 << (3 * k)) - 1), r = pool.records[block]!;
  return { block, page: [r & 63, (r >>> 6) & 63, (r >>> 12) & 63], local: [l & (edge - 1), (l >> k) & (edge - 1), l >> (2 * k)], edge, index };
}
/** tapTexel: the cell of level `edge` at q seen from block home, or -1. */
function tapCell(pool: UniformExtrapolationPool, home: number, page: Vec3, q: Vec3, edge: number): number {
  const k = log2(edge), d = [(q[0] >> k) - page[0], (q[1] >> k) - page[1], (q[2] >> k) - page[2]];
  let b = home;
  if (d[0] !== 0 || d[1] !== 0 || d[2] !== 0) { b = pool.rows[27 * home + d[0]! + 3 * d[1]! + 9 * d[2]! + 13]!; if (!stored(pool, b)) return -1; }
  return b * edge ** 3 + ((q[0] & (edge - 1)) | ((q[1] & (edge - 1)) << k) | ((q[2] & (edge - 1)) << (2 * k)));
}
function faceLocation(c: Cell, component: number): [number, number, number] {
  const size = f(32 / c.edge), out: [number, number, number] = [0, 0, 0];
  for (let a = 0; a < 3; a++) { const g = c.page[a]! * c.edge + c.local[a]!; out[a] = a === component ? f(f(g + 1) * size) : f(f(g + 0.5) * size); }
  return out;
}
/** boundDistance: squared distance from a face centre to a source bound box. */
function boundDistance(location: Vec3, level: UniformExtrapolationLevel, cell: number, component: number, scale: Vec3): number {
  const delta = [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    const lo = f(f(level.lower[9 * cell + 3 * component + a]! + 0.5) + (a === component ? 0.5 : 0)), hi = f(f(level.upper[9 * cell + 3 * component + a]! + 0.5) + (a === component ? 0.5 : 0));
    delta[a] = f(f(location[a]! - Math.min(Math.max(location[a]!, lo), hi)) * scale[a]!);
  }
  return f(f(f(delta[0]! * delta[0]!) + f(delta[2]! * delta[2]!)) + f(delta[1]! * delta[1]!));
}
const d4 = (v: readonly number[]): number => f(f(f(v[0]! + v[5]!) + f(v[1]! + v[4]!)) + f(f(v[2]! + v[7]!) + f(v[3]! + v[6]!)));

interface Sample { known: boolean; value: number; selection: number; lower: [number, number, number]; upper: [number, number, number] }
const nothing = (): Sample => ({ known: false, value: 0, selection: 0, lower: [0, 0, 0], upper: [0, 0, 0] });

/** nearestHierarchySample. */
function nearestSample(pool: UniformExtrapolationPool, c: Cell, source: UniformExtrapolationLevel, component: number, footprint: boolean, scale: Vec3): Sample {
  const location = faceLocation(c, component), ratio = f(source.edge / c.edge), lower = [0, 0, 0], fraction = [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    const g = c.page[a]! * c.edge + c.local[a]!;
    const position = a === component ? f(f(f(g + 1) * ratio) - 1) : f(f(f(g + 0.5) * ratio) - 0.5);
    lower[a] = footprint ? 2 * g : Math.floor(position); fraction[a] = f(position - Math.floor(position));
  }
  const distances = new Array<number>(8).fill(1e30), taps = new Array<number>(8).fill(-1);
  let best = f(1e30);
  for (let k = 0; k < 8; k++) {
    const offset = [k & 1, (k >> 1) & 1, k >> 2];
    if (!footprint && offset.some((bit, a) => (bit ? fraction[a]! : f(1 - fraction[a]!)) <= 0)) continue;
    const tap = tapCell(pool, c.block, c.page, [lower[0]! + offset[0]!, lower[1]! + offset[1]!, lower[2]! + offset[2]!], source.edge);
    if (tap < 0 || !(source.mask[tap]! & (1 << component))) continue;
    distances[k] = boundDistance(location, source, tap, component, scale); taps[k] = tap; best = Math.min(best, distances[k]!);
  }
  const epsilon = f(f(1e-6) * Math.max(1, best)), values = new Array<number>(8).fill(0), weights = new Array<number>(8).fill(0), out = nothing();
  out.lower = [4096, 4096, 4096]; out.upper = [-1, -1, -1];
  for (let k = 0; k < 8; k++) if (taps[k]! >= 0 && Math.abs(f(distances[k]! - best)) <= epsilon) {
    const tap = taps[k]!; values[k] = source.value[3 * tap + component]!; weights[k] = 1; out.selection |= 1 << k;
    for (let a = 0; a < 3; a++) { out.lower[a] = Math.min(out.lower[a]!, source.lower[9 * tap + 3 * component + a]!); out.upper[a] = Math.max(out.upper[a]!, source.upper[9 * tap + 3 * component + a]!); }
  }
  const weight = d4(weights);
  if (weight <= 0) return nothing();
  out.known = true; out.value = f(d4(values) / weight); return out;
}
/** nearestBlockSample: the 26 blocks around c's, summed in row order. */
function nearestBlock(pool: UniformExtrapolationPool, c: Cell, source: UniformExtrapolationLevel, component: number, scale: Vec3): Sample {
  const location = faceLocation(c, component), distances = new Array<number>(27).fill(1e30), taps = new Array<number>(27).fill(-1);
  let best = f(1e30);
  for (let i = 0; i < 27; i++) {
    if (i === 13) continue;
    const b = pool.rows[27 * c.block + i]!;
    if (!stored(pool, b) || !(source.mask[b]! & (1 << component))) continue;
    distances[i] = boundDistance(location, source, b, component, scale); taps[i] = b; best = Math.min(best, distances[i]!);
  }
  const epsilon = f(f(1e-6) * Math.max(1, best)), out = nothing();
  out.lower = [4096, 4096, 4096]; out.upper = [-1, -1, -1];
  let sum = 0, count = 0;
  for (let i = 0; i < 27; i++) if (taps[i]! >= 0 && Math.abs(f(distances[i]! - best)) <= epsilon) {
    const tap = taps[i]!; sum = f(sum + source.value[3 * tap + component]!); count += 1;
    for (let a = 0; a < 3; a++) { out.lower[a] = Math.min(out.lower[a]!, source.lower[9 * tap + 3 * component + a]!); out.upper[a] = Math.max(out.upper[a]!, source.upper[9 * tap + 3 * component + a]!); }
  }
  if (count <= 0) return nothing();
  out.known = true; out.value = f(sum / count); return out;
}
function store(target: UniformExtrapolationLevel, cell: number, component: number, sample: Sample): void {
  target.value[3 * cell + component] = sample.value;
  if (sample.known) target.mask[cell] = target.mask[cell]! | (1 << component);
  for (let a = 0; a < 3; a++) { target.lower[9 * cell + 3 * component + a] = sample.lower[a]!; target.upper[9 * cell + 3 * component + a] = sample.upper[a]!; }
  target.selection[cell] = target.selection[cell]! | (sample.selection << (8 * component));
}
function existing(level: UniformExtrapolationLevel, cell: number, component: number): Sample {
  const at = 9 * cell + 3 * component;
  return { known: (level.mask[cell]! & (1 << component)) !== 0, value: level.value[3 * cell + component]!, selection: 0,
    lower: [level.lower[at]!, level.lower[at + 1]!, level.lower[at + 2]!], upper: [level.upper[at]!, level.upper[at + 1]!, level.upper[at + 2]!] };
}
/** A kernel writes only the cells of blocks that hold a page; the rest keep what they held. */
function eachHeld(pool: UniformExtrapolationPool, target: UniformExtrapolationLevel, visit: (c: Cell) => void): void {
  const cells = pool.records.length * target.edge ** 3;
  for (let index = 0; index < cells; index++) {
    const c = levelCell(pool, index, target.edge);
    if (!held(pool, c.block)) continue;
    target.mask[index] = 0; target.selection[index] = 0; visit(c);
  }
}

/** mixedRestrictKnownVelocity: `coarse` from the next finer level. */
export function uniformExtrapolationRestrict(pool: UniformExtrapolationPool, fine: UniformExtrapolationLevel, coarse: UniformExtrapolationLevel, scale: Vec3): void {
  eachHeld(pool, coarse, c => { for (let component = 0; component < 3; component++) {
    let sample = nearestSample(pool, c, fine, component, false, scale);
    if (!sample.known && component === 1) sample = nearestSample(pool, c, fine, component, true, scale);
    sample.selection = 0; store(coarse, c.index, component, sample);
  } });
}
/** mixedSweepBlocks: one sweep of the 32h level from `source` into `target`. */
export function uniformExtrapolationSweep(pool: UniformExtrapolationPool, source: UniformExtrapolationLevel, target: UniformExtrapolationLevel, scale: Vec3): void {
  eachHeld(pool, target, c => { for (let component = 0; component < 3; component++) {
    const own = existing(source, c.index, component);
    store(target, c.index, component, own.known ? own : nearestBlock(pool, c, source, component, scale));
  } });
}
/** closedFace: the 4h cell's +face on the component is a closed wall. */
export function uniformExtrapolationClosedFace(pool: UniformExtrapolationPool, block: number, local: Vec3, component: number, edge: number): boolean {
  return edge === 8 && local[component] === 7 && pool.rows[27 * block + 13 + 3 ** component] === 0xffffffff;
}
/** mixedProlongUnknownVelocity: `up` is `down` with its unknown values filled from the complete coarser level. */
export function uniformExtrapolationProlong(pool: UniformExtrapolationPool, coarse: UniformExtrapolationLevel, down: UniformExtrapolationLevel, up: UniformExtrapolationLevel, scale: Vec3): void {
  eachHeld(pool, up, c => { for (let component = 0; component < 3; component++) {
    let sample = existing(down, c.index, component);
    if (!sample.known && !uniformExtrapolationClosedFace(pool, c.block, c.local, component, c.edge)) {
      const filled = nearestSample(pool, c, coarse, component, false, scale);
      if (filled.known) sample = filled;
    }
    store(up, c.index, component, sample);
  } });
}
/** mixedReplayUnknownVelocity: the same fill from up's recorded selection; writes values and masks only. */
export function uniformExtrapolationReplay(pool: UniformExtrapolationPool, coarse: UniformExtrapolationLevel, down: UniformExtrapolationLevel, up: UniformExtrapolationLevel): void {
  const cells = pool.records.length * up.edge ** 3;
  for (let index = 0; index < cells; index++) {
    const c = levelCell(pool, index, up.edge);
    if (!held(pool, c.block)) continue;
    let mask = down.mask[index]!;
    for (let component = 0; component < 3; component++) {
      let value = down.value[3 * index + component]!;
      const selected = (up.selection[index]! >>> (8 * component)) & 255;
      if (!(mask & (1 << component)) && selected) {
        const lower = [0, 0, 0], values = new Array<number>(8).fill(0), weights = new Array<number>(8).fill(0);
        for (let a = 0; a < 3; a++) { const g = c.page[a]! * c.edge + c.local[a]!; lower[a] = Math.floor(a === component ? f(f(f(g + 1) * 0.5) - 1) : f(f(f(g + 0.5) * 0.5) - 0.5)); }
        for (let k = 0; k < 8; k++) if (selected & (1 << k)) {
          values[k] = coarse.value[3 * tapCell(pool, c.block, c.page, [lower[0]! + (k & 1), lower[1]! + ((k >> 1) & 1), lower[2]! + (k >> 2)], coarse.edge) + component]!; weights[k] = 1;
        }
        value = f(d4(values) / d4(weights)); mask |= 1 << component;
      }
      up.value[3 * index + component] = value;
    }
    up.mask[index] = mask;
  }
}

/** One encode of the hierarchy from the 4h root's known values: restrict to
 * 32h, sweep the blocks, prolong back to the root. Returns each level's
 * `up` (the root's is what the far lookup reads) and `down`. */
export function uniformExtrapolationRun(pool: UniformExtrapolationPool, root: UniformExtrapolationLevel, scale: Vec3): { readonly down: UniformExtrapolationLevel[]; readonly up: UniformExtrapolationLevel[] } {
  const blocks = pool.records.length, down = [root], up: UniformExtrapolationLevel[] = [];
  for (let level = 1; level < 4; level++) { down.push(uniformExtrapolationLevel(blocks, 8 >> level)); uniformExtrapolationRestrict(pool, down[level - 1]!, down[level]!, scale); }
  // The kernels sweep between the 32h level's two textures; here the restricted one is kept for the caller.
  let top = down[3]!, other = uniformExtrapolationLevel(blocks, 1), spare = uniformExtrapolationLevel(blocks, 1);
  for (let sweep = 0; sweep < UNIFORM_EXTRAPOLATION_BLOCK_SWEEPS; sweep++) { uniformExtrapolationSweep(pool, top, other, scale); [top, other, spare] = [other, sweep === 0 ? spare : top, spare]; }
  up[3] = top;
  for (let level = 2; level >= 0; level--) { up[level] = uniformExtrapolationLevel(blocks, 8 >> level); uniformExtrapolationProlong(pool, up[level + 1]!, down[level]!, up[level]!, scale); }
  return { down, up };
}

/** umFarValue: the far value at a face centre `point` (cells) of component
 * `axis`, from the filled 4h level: the trilinear combination of the known
 * cells around it. A tap in a page that is not stored, and an unknown cell,
 * take no part. Where no tap of positive weight is known, a second pass
 * takes the zero-weight taps across the cell plane the face lies on, each
 * zero factor floored to 2^-10. */
export function uniformExtrapolationFarValue(pool: UniformExtrapolationPool, root: UniformExtrapolationLevel, point: Vec3, axis: number): number {
  const base = [0, 0, 0], fraction = [0, 0, 0], w = UNIFORM_PAGE_WINDOW - 1;
  for (let a = 0; a < 3; a++) { let q = f(f(f(point[a]!) / 4) - 0.5); if (a === axis) q = f(q - 0.5); base[a] = Math.floor(q); fraction[a] = f(q - Math.floor(q)); }
  let sum = 0, total = 0;
  for (let pass = 0; pass < 2 && total <= 0; pass++) for (let k = 0; k < 8; k++) {
    const bit = [k & 1, (k >> 1) & 1, k >> 2], side = bit.map((on, a) => (on ? fraction[a]! : f(1 - fraction[a]!)));
    if ((f(f(side[0]! * side[1]!) * side[2]!) > 0) !== (pass === 0)) continue;
    const floored = side.map(value => Math.max(value, 2 ** -10));
    const weight = f(f(floored[0]! * floored[1]!) * floored[2]!);
    const t = [base[0]! + bit[0]!, base[1]! + bit[1]!, base[2]! + bit[2]!];
    const b = pool.table[((t[0]! >> 3) & w) | (((t[1]! >> 3) & w) << 6) | (((t[2]! >> 3) & w) << 12)]!;
    if (!stored(pool, b)) continue;
    const cell = 512 * b + ((t[0]! & 7) | ((t[1]! & 7) << 3) | ((t[2]! & 7) << 6));
    if (!(root.mask[cell]! & (1 << axis))) continue;
    sum = f(sum + f(weight * root.value[3 * cell + axis]!)); total = f(total + weight);
  }
  return total > 0 ? f(sum / total) : 0;
}
