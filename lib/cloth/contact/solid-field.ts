/**
 * The static voxel solid as cloth contact sees it: a sparse table of 8^3-cell
 * bricks of signed-distance node values (9^3 nodes each, shared exactly at
 * brick faces), trilinear inside a cell, in the cloth-local frame (metres).
 * Built on the CPU at setup and at solid edits only; the step samples it on
 * the GPU (OGC_SOLID_WGSL).
 *
 * Every brick whose nodes come within `band` of the surface (or reach inside)
 * is stored, so all solid lies in stored bricks and the interpolant exceeds
 * `band` everywhere in a missing brick. The upload computes the Lipschitz
 * bound L of the interpolant per brick (per cell, |d phi / d x_a| is largest on
 * one of the four cell edges along a); the image carries the largest, L_max.
 *
 * A feature thinner than a cell is refused at authoring where the lattice
 * loses it: the build samples every cell the surface can cross and throws
 * where the source and the interpolant disagree (checkResolved).
 *
 * GPU image (u32 words, appended to the contact buffer at a word offset base):
 *   header  [0] slots - 1  [1] brick count  [2] cell (f32)  [3] band (f32)
 *           [4..6] origin (f32)  [7] L_max (f32)  [8..10] bricks per axis  [11] 0
 *   slots   (bx, by, bz, record word offset from base); an empty slot has w = OGC_NONE
 *   records 729 node values (f32), x fastest, padded to 732 words
 */
import { OGC_NONE } from "./ogc-topology";

export type V3 = readonly [number, number, number];

export const SOLID_BRICK_CELLS = 8;
const N = SOLID_BRICK_CELLS + 1, NODES = N ** 3;
export const SOLID_HEADER_WORDS = 12;
export const SOLID_RECORD_WORDS = 4 * Math.ceil(NODES / 4);

export interface ClothSolidSource {
  /** Signed distance (m) at a cloth-local point, negative inside; any 1-Lipschitz function with the solid's zero set. */
  sdf(p: V3): number;
  /** Cloth-local box the cloth can reach; bricks are built over it and the cloth must stay inside it. */
  lo: V3;
  hi: V3;
  /** Node spacing (m) and the surface band (m) within which bricks are kept. */
  cell: number;
  band: number;
}

export interface ClothSolidBrick { brick: V3; values: Float32Array; lipschitz: number }

export interface ClothSolidField {
  cell: number;
  band: number;
  /** Cloth-local position of node (0, 0, 0) of brick (0, 0, 0). */
  origin: V3;
  /** Bricks per axis over the authored box. */
  dims: V3;
  bricks: Map<string, ClothSolidBrick>;
  /** Largest per-brick L. */
  maxLipschitz: number;
  /** The GPU image. */
  words: Uint32Array<ArrayBuffer>;
}

const hash = (x: number, y: number, z: number) => (Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791)) >>> 0;
const nodeIndex = (x: number, y: number, z: number) => x + N * (y + N * z);

/** Trilinear value and gradient (per cell) of a brick at local cell coordinates q in [0, 8]^3: [d/dx, d/dy, d/dz, value]. */
export function trilinear(values: ArrayLike<number>, q: V3): [number, number, number, number] {
  const c = q.map(v => Math.min(SOLID_BRICK_CELLS - 1, Math.max(0, Math.floor(v)))), f = q.map((v, a) => v - c[a]!);
  const out: [number, number, number, number] = [0, 0, 0, 0];
  for (let k = 0; k < 8; k++) {
    const b = [k & 1, (k >> 1) & 1, k >> 2], w = b.map((v, a) => (v ? f[a]! : 1 - f[a]!));
    const v = values[nodeIndex(c[0]! + b[0]!, c[1]! + b[1]!, c[2]! + b[2]!)]!;
    out[3] += v * w[0]! * w[1]! * w[2]!;
    for (let a = 0; a < 3; a++) out[a] += v * (b[a] ? 1 : -1) * w[(a + 1) % 3]! * w[(a + 2) % 3]!;
  }
  return out;
}

/** Per-axis range of d phi / d x_a (per cell) over local cell c of a brick: the extremes of its four a-edge differences. */
export function cellGradientRange(values: ArrayLike<number>, c: V3): { lo: [number, number, number]; hi: [number, number, number] } {
  const lo: [number, number, number] = [Infinity, Infinity, Infinity], hi: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let a = 0; a < 3; a++) for (let u = 0; u < 2; u++) for (let v = 0; v < 2; v++) {
    const p = [c[0], c[1], c[2]];
    p[(a + 1) % 3]! += u; p[(a + 2) % 3]! += v;
    const v0 = values[nodeIndex(p[0]!, p[1]!, p[2]!)]!;
    p[a]! += 1;
    const d = values[nodeIndex(p[0]!, p[1]!, p[2]!)]! - v0;
    lo[a] = Math.min(lo[a], d); hi[a] = Math.max(hi[a], d);
  }
  return { lo, hi };
}

function brickLipschitz(values: Float32Array, cell: number): number {
  let L = 0;
  for (let z = 0; z < SOLID_BRICK_CELLS; z++) for (let y = 0; y < SOLID_BRICK_CELLS; y++) for (let x = 0; x < SOLID_BRICK_CELLS; x++) {
    const { lo, hi } = cellGradientRange(values, [x, y, z]);
    L = Math.max(L, Math.hypot(...lo.map((l, a) => Math.max(-l, hi[a]!))) / cell);
  }
  return L;
}

/**
 * Refuse a solid the lattice does not resolve. For a 1-Lipschitz source, a
 * node with |phi| >= sqrt(3) h proves its whole cell one-signed (the cell lies
 * in that node's one-signed ball), so only cells with every |node| below that
 * are sampled, at h / 8. Where source and interpolant disagree in sign, the
 * interpolant has rounded a sharp corner or lost a feature: an ascent of |phi|
 * from the sample tells them apart. A corner belongs to a part at least a cell
 * thick and reaches depth h / 2 within two cells; a plate, rod or slot thinner
 * than about 0.9 h never does, and fails authoring.
 */
const PROBE = 8;
function thinAt(src: ClothSolidSource, p: V3): boolean {
  const { cell } = src, step = cell / 8, e = cell * 1e-4;
  const sign = Math.sign(src.sdf(p));
  let x = [...p], best = 0;
  for (let i = 0; i <= 16; i++) {
    const d = sign * src.sdf(x as unknown as V3);
    best = Math.max(best, d);
    if (best >= 0.45 * cell) return false;
    const g = [0, 1, 2].map(a => {
      const u = [...x], w = [...x]; u[a]! += e; w[a]! -= e;
      return sign * (src.sdf(u as unknown as V3) - src.sdf(w as unknown as V3)) / (2 * e);
    });
    const gl = Math.hypot(...g);
    if (!(gl > 0)) break;
    x = x.map((v, a) => v + step * g[a]! / gl);
  }
  return true;
}
function checkResolved(src: ClothSolidSource, at: (q: V3) => V3, values: Float32Array) {
  const { cell } = src, reach = Math.sqrt(3) * cell;
  for (let z = 0; z < SOLID_BRICK_CELLS; z++) for (let y = 0; y < SOLID_BRICK_CELLS; y++) for (let x = 0; x < SOLID_BRICK_CELLS; x++) {
    let far = false;
    for (let k = 0; k < 8 && !far; k++) far = Math.abs(values[nodeIndex(x + (k & 1), y + ((k >> 1) & 1), z + (k >> 2))]!) >= reach;
    if (far) continue;
    for (let k = 0; k < PROBE ** 3; k++) {
      const q: V3 = [x + ((k % PROBE) + 0.5) / PROBE, y + ((Math.floor(k / PROBE) % PROBE) + 0.5) / PROBE, z + (Math.floor(k / PROBE ** 2) + 0.5) / PROBE];
      const p = at(q), exact = src.sdf(p), interp = trilinear(values, q)[3];
      if (exact * interp < 0 && thinAt(src, p))
        throw new RangeError(`Cloth solid has a feature thinner than its ${cell} m cell near (${p.map(v => v.toFixed(4)).join(", ")})`);
    }
  }
}

/** Build one brick from the source, or undefined when it is further than band from all solid. Throws on unresolved features. */
function buildBrick(src: ClothSolidSource, origin: V3, b: V3): ClothSolidBrick | undefined {
  const { cell, band } = src;
  const at = (q: V3): V3 => [origin[0] + (b[0] * SOLID_BRICK_CELLS + q[0]) * cell, origin[1] + (b[1] * SOLID_BRICK_CELLS + q[1]) * cell, origin[2] + (b[2] * SOLID_BRICK_CELLS + q[2]) * cell];
  const values = new Float32Array(NODES);
  let lo = Infinity;
  for (let z = 0; z < N; z++) for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const v = src.sdf(at([x, y, z]));
    if (!Number.isFinite(v)) throw new RangeError("Cloth solid distance must be finite");
    values[nodeIndex(x, y, z)] = v; lo = Math.min(lo, v);
  }
  if (lo > band) return undefined;
  checkResolved(src, at, values);
  return { brick: b, values, lipschitz: brickLipschitz(values, cell) };
}

function pack(field: Omit<ClothSolidField, "words">): Uint32Array<ArrayBuffer> {
  const bricks = [...field.bricks.values()];
  let slots = 1;
  while (slots < 2 * Math.max(1, bricks.length)) slots *= 2;
  const words = new Uint32Array(SOLID_HEADER_WORDS + 4 * slots + bricks.length * SOLID_RECORD_WORDS);
  const f = new Float32Array(words.buffer);
  words[0] = slots - 1; words[1] = bricks.length; f[2] = field.cell; f[3] = field.band;
  f[4] = field.origin[0]; f[5] = field.origin[1]; f[6] = field.origin[2]; f[7] = field.maxLipschitz;
  words[8] = field.dims[0]; words[9] = field.dims[1]; words[10] = field.dims[2];
  for (let s = 0; s < slots; s++) words[SOLID_HEADER_WORDS + 4 * s + 3] = OGC_NONE;
  let record = SOLID_HEADER_WORDS + 4 * slots;
  for (const b of bricks) {
    let s = hash(b.brick[0], b.brick[1], b.brick[2]) & (slots - 1);
    while (words[SOLID_HEADER_WORDS + 4 * s + 3] !== OGC_NONE) s = (s + 1) & (slots - 1);
    words.set([b.brick[0], b.brick[1], b.brick[2], record], SOLID_HEADER_WORDS + 4 * s);
    f.set(b.values, record);
    record += SOLID_RECORD_WORDS;
  }
  return words;
}

export function buildClothSolidField(src: ClothSolidSource): ClothSolidField {
  const { cell, band } = src;
  if (!(cell > 0 && band > 0 && src.lo.every((v, a) => v < src.hi[a]!))) throw new RangeError("Cloth solid field needs cell > 0, band > 0 and a non-empty box");
  const span = SOLID_BRICK_CELLS * cell;
  const origin = src.lo.map(v => Math.floor(v / span) * span) as unknown as V3;
  const dims = src.hi.map((v, a) => Math.ceil((v - origin[a]!) / span)) as unknown as V3;
  if (dims.some(c => c > 1 << 20)) throw new RangeError("Cloth solid field box is too large for its cell");
  const bricks = new Map<string, ClothSolidBrick>();
  for (let bz = 0; bz < dims[2]; bz++) for (let by = 0; by < dims[1]; by++) for (let bx = 0; bx < dims[0]; bx++) {
    const b = buildBrick(src, origin, [bx, by, bz]);
    if (b) bricks.set(`${bx},${by},${bz}`, b);
  }
  return finish({ cell, band, origin, dims, bricks, maxLipschitz: 0 });
}

function finish(f: Omit<ClothSolidField, "words">): ClothSolidField {
  let maxLipschitz = 0;
  for (const b of f.bricks.values()) maxLipschitz = Math.max(maxLipschitz, b.lipschitz);
  const out = { ...f, maxLipschitz };
  return { ...out, words: pack(out) };
}

/**
 * The field after a solid edit inside the cloth-local box [lo, hi]: the bricks
 * within `band` of the box are rebuilt from the edited source, every other
 * brick is kept as it is. Edit-time CPU work.
 */
export function rebuildClothSolidField(field: ClothSolidField, src: ClothSolidSource, lo: V3, hi: V3): ClothSolidField {
  if (src.cell !== field.cell || src.band !== field.band) throw new RangeError("A solid edit cannot change the cloth solid field's cell or band");
  const span = SOLID_BRICK_CELLS * field.cell;
  const b0 = lo.map((v, a) => Math.max(0, Math.floor((v - field.band - field.origin[a]!) / span)));
  const b1 = hi.map((v, a) => Math.min(field.dims[a]! - 1, Math.floor((v + field.band - field.origin[a]!) / span)));
  const bricks = new Map(field.bricks);
  for (let bz = b0[2]!; bz <= b1[2]!; bz++) for (let by = b0[1]!; by <= b1[1]!; by++) for (let bx = b0[0]!; bx <= b1[0]!; bx++) {
    const key = `${bx},${by},${bz}`, b = buildBrick(src, field.origin, [bx, by, bz]);
    if (b) bricks.set(key, b); else bricks.delete(key);
  }
  return finish({ cell: field.cell, band: field.band, origin: field.origin, dims: field.dims, bricks, maxLipschitz: 0 });
}

/** The field at a cloth-local point: phi (m), its gradient, and whether a stored brick holds the point (else phi reads band). */
export function sampleClothSolidField(field: ClothSolidField, p: V3): { phi: number; grad: V3; stored: boolean } {
  const q = p.map((v, a) => (v - field.origin[a]!) / field.cell);
  const b = q.map(v => Math.floor(v / SOLID_BRICK_CELLS));
  const brick = field.bricks.get(b.join(","));
  if (!brick) return { phi: field.band, grad: [0, 0, 0], stored: false };
  const s = trilinear(brick.values, q.map((v, a) => v - b[a]! * SOLID_BRICK_CELLS) as unknown as V3);
  return { phi: s[3], grad: [s[0] / field.cell, s[1] / field.cell, s[2] / field.cell], stored: true };
}
