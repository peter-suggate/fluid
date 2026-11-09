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
 *   records 729 node values (f32), x fastest, padded to 732 words, then the
 *           node friction coefficients, one byte per node (mu * 100), four per word
 *
 * A source is one signed distance or a union of parts, each with its own
 * friction coefficient. A part is evaluated only for bricks within band of its
 * box, and only its own surface is checked for resolution; a "lattice-box"
 * part, a box whose faces lie on node planes, is exact under the trilinear
 * interpolant and needs no check (scene floors and slabs). A node takes the
 * friction of the part nearest it.
 */
import { OGC_NONE } from "./ogc-topology";

export type V3 = readonly [number, number, number];

export const SOLID_BRICK_CELLS = 8;
const N = SOLID_BRICK_CELLS + 1, NODES = N ** 3;
export const SOLID_HEADER_WORDS = 12;
/** Word offset of the friction bytes in a record. */
export const SOLID_MU_WORD = 4 * Math.ceil(NODES / 4);
export const SOLID_RECORD_WORDS = SOLID_MU_WORD + 4 * Math.ceil(NODES / 16);
/** Friction coefficients are stored per node to 0.01, from 0 to 2.55. */
export const SOLID_MU_SCALE = 100;

/** The smallest f32 at or above x: the image stores L_max as f32, and a bound must not round down. */
export function f32Up(x: number): number {
  const f = Math.fround(x);
  if (f >= x || !Number.isFinite(f)) return f;
  const b = new Float32Array([f]), u = new Uint32Array(b.buffer);
  u[0]! += f >= 0 ? 1 : -1;
  return b[0]!;
}

/**
 * The bound on L_max for a lattice-sampled 1-Lipschitz solid: sqrt(3), rounded
 * up to f32 as the image stores it. A field whose node differences reach one
 * cell on all three axes of a cell (a lattice box's corner) has exactly this L_max.
 */
export const SOLID_LIPSCHITZ_BOUND = f32Up(Math.sqrt(3));

export type ClothSolidPart =
  /** Any 1-Lipschitz signed distance whose solid lies inside [lo, hi]. */
  | { readonly kind: "sdf"; sdf(p: V3): number; readonly lo: V3; readonly hi: V3; readonly mu: number }
  /** A box with every face on a node plane of the field's lattice. */
  | { readonly kind: "lattice-box"; readonly lo: V3; readonly hi: V3; readonly mu: number };

export interface ClothSolidSource {
  /** Signed distance (m) at a cloth-local point, negative inside; any 1-Lipschitz function with the solid's zero set. Exclusive with parts. */
  sdf?(p: V3): number;
  /** Friction coefficient of the `sdf` solid (default 0: frictionless). */
  mu?: number;
  /** The solid as a union of parts. */
  parts?: readonly ClothSolidPart[];
  /** Cloth-local box the cloth can reach; bricks are built over it and the cloth must stay inside it. */
  lo: V3;
  hi: V3;
  /** Node spacing (m) and the surface band (m) within which bricks are kept. */
  cell: number;
  band: number;
  /** Position of lattice node (0, 0, 0), at or below lo on every axis: a scene's own voxel lattice, so the
   * field's nodes are the scene's. Default floor(lo / 8 cell) 8 cell. */
  origin?: V3;
}

export interface ClothSolidBrick { brick: V3; values: Float32Array; mu: Uint8Array; lipschitz: number }

export interface ClothSolidField {
  cell: number;
  band: number;
  /** Cloth-local position of node (0, 0, 0) of brick (0, 0, 0). */
  origin: V3;
  /** Bricks per axis over the authored box. */
  dims: V3;
  bricks: Map<string, ClothSolidBrick>;
  /** Largest per-brick L, rounded up to f32 (the image's word 7 exactly). */
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

/**
 * The brick's L: per cell and axis, the largest |difference| along the cell's four
 * a-edges (cellGradientRange). Measured on the source's own node values, before the
 * image rounds them to f32: that rounding perturbs each node by at most half an f32
 * ulp of its value, the same precision every f32 operation on the field has, and is
 * not a gradient of the solid (it would put a lattice box's corner a few ulps over
 * sqrt(3)). A 1-Lipschitz source therefore measures at most sqrt(3).
 */
function brickLipschitz(values: ArrayLike<number>, cell: number): number {
  const stride = [1, N, N * N];
  let L2 = 0;
  for (let z = 0; z < SOLID_BRICK_CELLS; z++) for (let y = 0; y < SOLID_BRICK_CELLS; y++) for (let x = 0; x < SOLID_BRICK_CELLS; x++) {
    const base = nodeIndex(x, y, z);
    let s2 = 0;
    for (let a = 0; a < 3; a++) {
      const da = stride[a]!, du = stride[(a + 1) % 3]!, dv = stride[(a + 2) % 3]!;
      let m = 0;
      for (let k = 0; k < 4; k++) {
        const i = base + (k & 1) * du + (k >> 1) * dv;
        m = Math.max(m, Math.abs(values[i + da]! - values[i]!));
      }
      s2 += m * m;
    }
    L2 = Math.max(L2, s2);
  }
  return Math.sqrt(L2) / cell;
}

/**
 * Refuse a solid the lattice does not resolve. For a 1-Lipschitz source, a
 * node with |phi| >= sqrt(3) h proves its whole cell one-signed (the cell lies
 * in that node's one-signed ball), so only cells with every |node| below that
 * are sampled, at h / 8; an octree over each cell skips the probes of
 * sub-boxes where source and interpolant are provably of one, equal sign.
 *
 * Where they disagree, the interpolant has rounded a sharp edge or corner, or
 * lost a feature. The rule (SOLID_RESOLUTION): a disagreeing probe is accepted
 * when a point at least DEPTH h deep on its side of the surface lies within
 * REACH h of it, joined to it by a straight segment that stays on that side.
 * So plates, rods and slots thinner than 2 DEPTH h = 0.9 h always fail, and a
 * part at least a cell thick passes wherever the interpolant rounds it; a
 * convex wedge (edge) passes from 35 degrees at every lattice phase and
 * orientation, and one of 20 degrees or less fails (measured in
 * tests/cloth-contact-solid; 25 to 30 degrees depends on the phase).
 */
const PROBE = 8;
export const SOLID_RESOLUTION = { depth: 0.45, reach: 2 } as const;
type Sdf = (p: V3) => number;
function thinAt(sdf: Sdf, cell: number, p: V3): boolean {
  const src = { sdf };
  const step = cell / 8, e = cell * 1e-4, deep = SOLID_RESOLUTION.depth * cell;
  const sign = Math.sign(src.sdf(p));
  // Fast path: ascend |phi| along its gradient.
  let x = [...p];
  for (let i = 0; i <= 16; i++) {
    if (sign * src.sdf(x as unknown as V3) >= deep) return false;
    const g = [0, 1, 2].map(a => {
      const u = [...x], w = [...x]; u[a]! += e; w[a]! -= e;
      return sign * (src.sdf(u as unknown as V3) - src.sdf(w as unknown as V3)) / (2 * e);
    });
    const gl = Math.hypot(...g);
    if (!(gl > 0)) break;
    x = x.map((v, a) => v + step * g[a]! / gl);
  }
  // The ascent zigzags in acute wedges: search the ball of radius REACH h at h / 4.
  const R = SOLID_RESOLUTION.reach * cell, q = cell / 4, n = Math.round(R / q);
  for (let k = -n; k <= n; k++) for (let j = -n; j <= n; j++) for (let i = -n; i <= n; i++) {
    const d: V3 = [i * q, j * q, k * q], len = Math.hypot(...d);
    if (len > R) continue;
    const t: V3 = [p[0] + d[0], p[1] + d[1], p[2] + d[2]];
    if (sign * src.sdf(t) < deep) continue;
    let inside = true;
    for (let s = 1, m = Math.ceil(len / step); s < m && inside; s++) inside = sign * src.sdf([p[0] + d[0] * s / m, p[1] + d[1] * s / m, p[2] + d[2] * s / m]) > 0;
    if (inside) return false;
  }
  return true;
}
function checkResolved(sdf: Sdf, cell: number, at: (q: V3) => V3, values: ArrayLike<number>) {
  const src = { sdf };
  const reach = Math.sqrt(3) * cell, n = new Float64Array(8);
  // The interpolant at fraction f of the current cell (nodes n).
  const lerp = (fx: number, fy: number, fz: number) => {
    const a = n[0]! + fx * (n[1]! - n[0]!), b = n[2]! + fx * (n[3]! - n[2]!), c = n[4]! + fx * (n[5]! - n[4]!), d = n[6]! + fx * (n[7]! - n[6]!);
    const u = a + fy * (b - a), w = c + fy * (d - c);
    return u + fz * (w - u);
  };
  // Octree over the cell down to the h / 8 probes. A box is cleared when the
  // source is one-signed on it (|sdf(centre)| >= half-diagonal, 1-Lipschitz) and
  // the interpolant, multilinear so extreme at the box corners, has that same
  // sign on all of it: no probe in it can disagree. Leaves are the probes.
  const visit = (x: number, y: number, z: number, f: V3, s: number) => {
    const q: V3 = [x + f[0] + 0.5 * s, y + f[1] + 0.5 * s, z + f[2] + 0.5 * s], p = at(q), exact = src.sdf(p);
    if (s <= 1 / PROBE) {
      if (exact * lerp(f[0] + 0.5 * s, f[1] + 0.5 * s, f[2] + 0.5 * s) < 0 && thinAt(sdf, cell, p))
        throw new RangeError(`Cloth solid has a feature thinner than its ${cell} m cell near (${p.map(v => v.toFixed(4)).join(", ")}): `
          + `no point ${SOLID_RESOLUTION.depth} cells deep within ${SOLID_RESOLUTION.reach} cells (plates, rods and slots under ${2 * SOLID_RESOLUTION.depth} cells, convex edges sharper than 35 degrees may fail, under 20 always do)`);
      return;
    }
    if (Math.abs(exact) >= 0.5 * Math.sqrt(3) * s * cell) {
      let agree = true;
      for (let k = 0; k < 8 && agree; k++) agree = exact * lerp(f[0] + s * (k & 1), f[1] + s * ((k >> 1) & 1), f[2] + s * (k >> 2)) >= 0;
      if (agree) return;
    }
    const h = 0.5 * s;
    for (let k = 0; k < 8; k++) visit(x, y, z, [f[0] + h * (k & 1), f[1] + h * ((k >> 1) & 1), f[2] + h * (k >> 2)], h);
  };
  for (let z = 0; z < SOLID_BRICK_CELLS; z++) for (let y = 0; y < SOLID_BRICK_CELLS; y++) for (let x = 0; x < SOLID_BRICK_CELLS; x++) {
    let far = false;
    for (let k = 0; k < 8; k++) {
      n[k] = values[nodeIndex(x + (k & 1), y + ((k >> 1) & 1), z + (k >> 2))]!;
      far ||= Math.abs(n[k]!) >= reach;
    }
    if (!far) visit(x, y, z, [0, 0, 0], 1);
  }
}

interface Part { kind: "sdf" | "lattice-box"; sdf: Sdf; lo: V3; hi: V3; mu: number }
const boxSdf = (lo: V3, hi: V3): Sdf => p => {
  let out = 0, ins = -Infinity;
  for (let a = 0; a < 3; a++) {
    const q = Math.max(lo[a]! - p[a]!, p[a]! - hi[a]!);
    out += Math.max(q, 0) ** 2; ins = Math.max(ins, q);
  }
  return Math.sqrt(out) + Math.min(ins, 0);
};
/** The source's parts, validated against the lattice; unknown kinds fail. */
function partsOf(src: ClothSolidSource, origin: V3): Part[] {
  const { cell, band } = src;
  if (!(band >= Math.sqrt(3) * cell)) throw new RangeError(`Cloth solid band ${band} m must be at least sqrt(3) cells`);
  const checkMu = (mu: number) => {
    if (!(mu >= 0 && mu <= 255 / SOLID_MU_SCALE)) throw new RangeError(`Cloth solid friction ${mu} must lie in [0, ${255 / SOLID_MU_SCALE}]`);
    return mu;
  };
  if (src.parts && src.sdf) throw new RangeError("Cloth solid source gives both sdf and parts");
  if (!src.parts) {
    if (!src.sdf) throw new RangeError("Cloth solid source needs sdf or parts");
    const sdf = src.sdf.bind(src);
    return [{ kind: "sdf", sdf, lo: [-Infinity, -Infinity, -Infinity], hi: [Infinity, Infinity, Infinity], mu: checkMu(src.mu ?? 0) }];
  }
  return src.parts.map(part => {
    if (!part.lo.every((v, a) => v <= part.hi[a]!)) throw new RangeError("Cloth solid part needs lo <= hi");
    if (part.kind === "sdf") return { kind: part.kind, sdf: part.sdf.bind(part), lo: part.lo, hi: part.hi, mu: checkMu(part.mu) };
    if (part.kind === "lattice-box") {
      for (const v of [...part.lo.map((x, a) => (x - origin[a]!) / cell), ...part.hi.map((x, a) => (x - origin[a]!) / cell)])
        if (Math.abs(v - Math.round(v)) > 1e-6) throw new RangeError(`Cloth solid lattice-box faces must lie on node planes of the ${cell} m lattice`);
      return { kind: part.kind, sdf: boxSdf(part.lo, part.hi), lo: part.lo, hi: part.hi, mu: checkMu(part.mu) };
    }
    throw new RangeError(`Unknown cloth solid part kind "${(part as { kind: string }).kind}"`);
  });
}

/** Build one brick from the parts, or undefined when it is further than band from all solid. Throws on unresolved features. */
function buildBrick(parts: readonly Part[], cell: number, band: number, origin: V3, b: V3): ClothSolidBrick | undefined {
  const at = (q: V3): V3 => [origin[0] + (b[0] * SOLID_BRICK_CELLS + q[0]) * cell, origin[1] + (b[1] * SOLID_BRICK_CELLS + q[1]) * cell, origin[2] + (b[2] * SOLID_BRICK_CELLS + q[2]) * cell];
  const blo = at([0, 0, 0]), bhi = at([SOLID_BRICK_CELLS, SOLID_BRICK_CELLS, SOLID_BRICK_CELLS]);
  // A part whose box is further than band from the brick is further than band at every
  // node; with band >= sqrt(3) h it changes no cell that holds surface.
  const near = parts.filter(p => Math.hypot(...[0, 1, 2].map(a => Math.max(0, p.lo[a]! - bhi[a]!, blo[a]! - p.hi[a]!))) <= band);
  if (!near.length) return undefined;
  const union = new Float64Array(NODES).fill(Infinity), own = new Uint8Array(NODES);
  const pv = near.map(() => new Float32Array(NODES));
  const p: [number, number, number] = [0, 0, 0];
  let lo = Infinity;
  for (let k = 0; k < near.length; k++) {
    const part = near[k]!, out = pv[k]!;
    for (let z = 0, i = 0; z < N; z++) for (let y = 0; y < N; y++) for (let x = 0; x < N; x++, i++) {
      p[0] = blo[0] + x * cell; p[1] = blo[1] + y * cell; p[2] = blo[2] + z * cell;
      const v = part.sdf(p);
      if (!Number.isFinite(v)) throw new RangeError("Cloth solid distance must be finite");
      out[i] = v;
      if (v < union[i]!) { union[i] = v; own[i] = k; }
      if (k === near.length - 1) lo = Math.min(lo, union[i]!);
    }
  }
  if (lo > band) return undefined;
  // Each part's own surface: the union's interpolant lies at or below every part's, so
  // what a part keeps the union keeps.
  near.forEach((part, k) => { if (part.kind === "sdf") checkResolved(part.sdf, cell, at, pv[k]!); });
  const mu = new Uint8Array(NODES);
  for (let i = 0; i < NODES; i++) mu[i] = Math.round(near[own[i]!]!.mu * SOLID_MU_SCALE);
  return { brick: b, values: Float32Array.from(union), mu, lipschitz: brickLipschitz(union, cell) };
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
    const m = new Uint8Array(4 * (SOLID_RECORD_WORDS - SOLID_MU_WORD));
    m.set(b.mu);
    words.set(new Uint32Array(m.buffer), record + SOLID_MU_WORD);
    record += SOLID_RECORD_WORDS;
  }
  return words;
}

export function buildClothSolidField(src: ClothSolidSource): ClothSolidField {
  const { cell, band } = src;
  if (!(cell > 0 && band > 0 && src.lo.every((v, a) => v < src.hi[a]!))) throw new RangeError("Cloth solid field needs cell > 0, band > 0 and a non-empty box");
  const span = SOLID_BRICK_CELLS * cell;
  const origin = src.origin ?? src.lo.map(v => Math.floor(v / span) * span) as unknown as V3;
  if (!origin.every((v, a) => Number.isFinite(v) && v <= src.lo[a]!)) throw new RangeError("Cloth solid field origin must lie at or below lo");
  const dims = src.hi.map((v, a) => Math.ceil((v - origin[a]!) / span)) as unknown as V3;
  if (dims.some(c => c > 1 << 20)) throw new RangeError("Cloth solid field box is too large for its cell");
  const parts = partsOf(src, origin);
  const bricks = new Map<string, ClothSolidBrick>();
  // The bricks over [lo, hi]; with an explicit origin below lo, the bricks under lo stay empty.
  const b0 = src.lo.map((v, a) => Math.max(0, Math.floor((v - origin[a]!) / span)));
  for (let bz = b0[2]!; bz < dims[2]; bz++) for (let by = b0[1]!; by < dims[1]; by++) for (let bx = b0[0]!; bx < dims[0]; bx++) {
    const b = buildBrick(parts, cell, band, origin, [bx, by, bz]);
    if (b) bricks.set(`${bx},${by},${bz}`, b);
  }
  return finish({ cell, band, origin, dims, bricks, maxLipschitz: 0 });
}

function finish(f: Omit<ClothSolidField, "words">): ClothSolidField {
  let L = 0;
  for (const b of f.bricks.values()) L = Math.max(L, b.lipschitz);
  // Rounded up to the f32 the image stores, so the detector's band rule and the GPU read one value.
  const out = { ...f, maxLipschitz: f32Up(L) };
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
  const parts = partsOf(src, field.origin);
  const bricks = new Map(field.bricks);
  for (let bz = b0[2]!; bz <= b1[2]!; bz++) for (let by = b0[1]!; by <= b1[1]!; by++) for (let bx = b0[0]!; bx <= b1[0]!; bx++) {
    const key = `${bx},${by},${bz}`, b = buildBrick(parts, field.cell, field.band, field.origin, [bx, by, bz]);
    if (b) bricks.set(key, b); else bricks.delete(key);
  }
  return finish({ cell: field.cell, band: field.band, origin: field.origin, dims: field.dims, bricks, maxLipschitz: 0 });
}

/** The field at a cloth-local point: phi (m), its gradient, the nearest node's friction, and whether a stored brick holds the point (else phi reads band). */
export function sampleClothSolidField(field: ClothSolidField, p: V3): { phi: number; grad: V3; mu: number; stored: boolean } {
  const q = p.map((v, a) => (v - field.origin[a]!) / field.cell);
  const b = q.map(v => Math.floor(v / SOLID_BRICK_CELLS));
  const brick = field.bricks.get(b.join(","));
  if (!brick) return { phi: field.band, grad: [0, 0, 0], mu: 0, stored: false };
  const l = q.map((v, a) => v - b[a]! * SOLID_BRICK_CELLS) as unknown as V3, s = trilinear(brick.values, l);
  const n = l.map(v => Math.min(SOLID_BRICK_CELLS, Math.max(0, Math.round(v))));
  return { phi: s[3], grad: [s[0] / field.cell, s[1] / field.cell, s[2] / field.cell], mu: brick.mu[nodeIndex(n[0]!, n[1]!, n[2]!)]! / SOLID_MU_SCALE, stored: true };
}

/** Words of one surface sample on the GPU: position (f32 x3) and friction coefficient. */
export const SOLID_SAMPLE_WORDS = 4;

export interface ClothSolidSampleOptions {
  /** The cloth's contact radius r (m). */
  r: number;
  /** The cloth's longest edge (m). */
  edge: number;
}

/**
 * Normal turn per cell below which the trilinear field is taken as flat: the
 * interpolant's own error, not a contact tolerance (a sphere of radius 100 h
 * turns 0.01 rad per cell).
 */
export const SOLID_SAMPLE_TURN_FLOOR = 0.01;
/**
 * The largest lift a sample may need, as a fraction of the cell: a sample stands
 * out by its holes' bulge (below), and a bulge above h / 16 means the samples are
 * too sparse for the surface's curvature. The build then fails; it never clamps.
 * h / 16 lies inside the trilinear field's own rounding of a sharp edge (about a cell).
 */
export const SOLID_SAMPLE_MAX_LIFT = 1 / 16;
/**
 * Normal turn per cell above which the surface is sharp (radius of curvature
 * under 4 h: an edge or corner the trilinear field rounds over a cell), and is
 * seeded at h / 2 rather than h.
 */
export const SOLID_SAMPLE_SHARP_TURN = 0.25;

/**
 * How far a locally convex surface can rise above the triangle P, along the
 * unit direction N: it lies below each supporting (tangent) plane, so over the
 * triangle below the lowest of them, whose highest point is the bound. The
 * lowest of affine heights over a triangle is concave and piecewise affine: its
 * maximum is at a corner, where two heights meet on an edge, or where three
 * meet inside. A plane at or past perpendicular to N bounds nothing; with none
 * left the bound is infinite.
 */
function roofHeight(P: readonly [V3, V3, V3], N: V3, supports: readonly { p: V3; n: V3 }[]): number {
  const e1 = [0, 1, 2].map(i => P[1][i]! - P[0][i]!), e2 = [0, 1, 2].map(i => P[2][i]! - P[0][i]!);
  // Height over y = P0 + u e1 + v e2 of tangent plane k along N: -n_k.(y - p_k) / (n_k.N), affine in (u, v).
  const planes = supports.map(({ p: pk, n }) => {
    const nN = n[0]! * N[0]! + n[1]! * N[1]! + n[2]! * N[2]!;
    // A tangent plane at or past perpendicular to the triangle bounds nothing above it.
    if (!(nN > 1e-6)) return undefined;
    const d0 = [0, 1, 2].map(i => P[0][i]! - pk[i]!);
    const c0 = -(n[0]! * d0[0]! + n[1]! * d0[1]! + n[2]! * d0[2]!) / nN;
    const cu = -(n[0]! * e1[0]! + n[1]! * e1[1]! + n[2]! * e1[2]!) / nN, cv = -(n[0]! * e2[0]! + n[1]! * e2[1]! + n[2]! * e2[2]!) / nN;
    return [c0, cu, cv] as const;
  }).filter((x): x is readonly [number, number, number] => !!x);
  if (!planes.length) return Infinity;
  const at = (u: number, v: number) => Math.min(...planes.map(q => q[0] + q[1] * u + q[2] * v));
  const inside = (u: number, v: number) => u >= -1e-12 && v >= -1e-12 && u + v <= 1 + 1e-12;
  let best = Math.max(at(0, 0), at(1, 0), at(0, 1));
  // Two heights equal along a triangle edge: edges u = 0, v = 0, u + v = 1 as (u, v) = o + t d.
  const edges: [number, number, number, number][] = [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, -1, 1]];
  for (let i = 0; i < planes.length; i++) for (let j = i + 1; j < planes.length; j++) {
    const A = planes[i]!, B = planes[j]!, d = [A[0] - B[0], A[1] - B[1], A[2] - B[2]];
    for (const [ou, ov, du, dv] of edges) {
      const den = d[1]! * du + d[2]! * dv;
      if (Math.abs(den) < 1e-30) continue;
      const t = -(d[0]! + d[1]! * ou + d[2]! * ov) / den;
      if (t >= 0 && t <= 1) best = Math.max(best, at(ou + t * du, ov + t * dv));
    }
    // All three equal inside.
    for (let k = j + 1; k < planes.length; k++) {
      const C = planes[k]!, a1 = A[1] - B[1], b1 = A[2] - B[2], r1 = B[0] - A[0], a2 = A[1] - C[1], b2 = A[2] - C[2], r2 = C[0] - A[0];
      const det = a1 * b2 - a2 * b1;
      if (Math.abs(det) < 1e-30) continue;
      const u = (r1 * b2 - r2 * b1) / det, v = (a1 * r2 - a2 * r1) / det;
      if (inside(u, v)) best = Math.max(best, at(u, v));
    }
  }
  return Math.max(0, best);
}

/**
 * Static contact points just outside the field's zero set wherever it is
 * convex and not flat: the sharp edges and corners of a voxel solid, a
 * frustum's rims, every curved convex surface. They meet the cloth's faces
 * (and the faces' edges) as static points. Vertex contact alone does not keep
 * a face out: the vertices may come arbitrarily close to phi = 0 (the
 * barrier's domain is phi > 0, and an impact drives them there; the drape gate
 * measured 6e-5 m), and a face of edge e over curvature kappa then dips about
 * kappa e^2 / 8 below them. Flat surface (affine cells) and concave creases
 * (|grad phi| ~ 0 at the trilinear zero set) carry none.
 *
 * A cell whose normal turns (convexly) by more than the floor is seeded at its
 * centre; one turning past SOLID_SAMPLE_SHARP_TURN (an edge or corner the
 * trilinear field rounds over a cell) on a 2^3 lattice and at its eight
 * corners (a voxel solid's corner is a node). Each seed is projected onto the
 * surface and kept where the surface still turns there, at least half its
 * spacing from a kept sample, corners and edges (the most cells' tangent
 * planes meeting at the point) first.
 *
 * The lift is measured on the samples as built, not assumed. Each sample's
 * neighbours, ordered around its normal, span its fan of triangles; those
 * that are facets of the samples' local convex hull are the holes a face
 * resting on the samples spans. A convex surface lies below every one of its
 * tangent planes, those of each adjacent cell at an edge or corner included,
 * so over a hole it rises at most the highest point of their lower envelope
 * (roofHeight), measured along the three samples' mean normal; each of the
 * three stands that far out along its normal. A hole bulging past
 * SOLID_SAMPLE_MAX_LIFT h is refined (its centre and edge midpoints sampled at
 * half the spacing) and measured again; one still past it once the spacing is
 * down to the ceiling fails the build explicitly.
 *
 * Setup-time CPU work over the surface cells whose field is not affine; returns
 * (x, y, z, mu) per sample, with phi there equal to its lift.
 *
 * `options` (the cloth's r and longest edge) are validated and otherwise not
 * used: the samples depend on the field alone.
 */
export function clothSolidSamples(field: ClothSolidField, options?: ClothSolidSampleOptions): Float32Array {
  return clothSolidSampleHull(field, options).samples;
}

/**
 * The samples' hull edges: each pair of samples (local indices, i < j, two words per
 * edge) neighbouring in a sample's fan, the edges of the holes a face resting on the
 * samples spans. A cloth face can rest on the samples and still have one of its edges
 * pass between two of them (between two samples of a sharp edge, say); a cloth edge
 * kept apart from every hull edge cannot, so the detector pairs them (DAT).
 */
export function clothSolidSampleEdges(field: ClothSolidField, options?: ClothSolidSampleOptions): Uint32Array {
  return clothSolidSampleHull(field, options).edges;
}

function clothSolidSampleHull(field: ClothSolidField, options?: ClothSolidSampleOptions): { samples: Float32Array; edges: Uint32Array } {
  if (options && !(options.r > 0 && options.edge > 0)) throw new RangeError("Cloth solid samples need r > 0 and edge > 0");
  let cached = sampleCache.get(field);
  if (!cached) sampleCache.set(field, cached = buildSolidSamples(field));
  return cached;
}

/** A field is immutable once built (a rebuild returns a new one): its samples are computed once. */
const sampleCache = new WeakMap<ClothSolidField, { samples: Float32Array; edges: Uint32Array }>();

type SurfacePoint = { p: V3; n: V3; mu: number };

function buildSolidSamples(field: ClothSolidField): { samples: Float32Array; edges: Uint32Array } {
  // A cell's turn is read two cells out (it finds an edge the centre's own cell misses); a seed's
  // a tenth of a cell out, so a point on a voxel solid's flat face near its edge is not kept.
  const h = field.cell, kappaMin = SOLID_SAMPLE_TURN_FLOOR / h, CELL_STEP = 2, localStep = 0.1;
  const sample = (p: V3) => sampleClothSolidField(field, p);
  // Newton onto phi = 0 along the gradient; undefined when it leaves the stored field or drifts more than a cell.
  const project = (p0: V3): SurfacePoint | undefined => {
    let p: V3 = p0;
    // Tight enough that a point at a voxel solid's edge, where phi vanishes quadratically and
    // Newton only halves its distance per step, ends within 1e-4 cells of it.
    for (let i = 0; i < 40; i++) {
      const s = sample(p);
      if (!s.stored) return undefined;
      if (Math.abs(s.phi) < 1e-10 * h) {
        if (Math.hypot(p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]) > 2.5 * h) return undefined;
        // The normal from central differences a quarter cell wide: at a lattice edge the
        // one-sided cell gradient may vanish. A concave crease (phi ~ xz / h) has none: skip it.
        const d = 0.25 * h, gc = [0, 1, 2].map(a => {
          const at = (o: number): V3 => [p[0] + (a === 0 ? o : 0), p[1] + (a === 1 ? o : 0), p[2] + (a === 2 ? o : 0)];
          return (sample(at(d)).phi - sample(at(-d)).phi) / (2 * d);
        });
        const gl = Math.hypot(...gc);
        if (!(gl > 0.35)) return undefined;
        return { p, n: [gc[0]! / gl, gc[1]! / gl, gc[2]! / gl], mu: s.mu };
      }
      // (A voxel solid's corner node may see no gradient in the cell holding it: test phi first.)
      const g2 = s.grad[0] ** 2 + s.grad[1] ** 2 + s.grad[2] ** 2;
      if (!(g2 > 1e-18)) return undefined;
      p = [p[0] - s.phi * s.grad[0] / g2, p[1] - s.phi * s.grad[1] / g2, p[2] - s.phi * s.grad[2] / g2];
    }
    return undefined;
  };
  // The largest convex normal turn per length seen `step` cells away along two orthogonal tangents, both ways.
  const kappa = (q: SurfacePoint, step: number): number => {
    const n = q.n, a = Math.abs(n[0]) < 0.6 ? [1, 0, 0] : Math.abs(n[1]) < 0.6 ? [0, 1, 0] : [0, 0, 1];
    const t1 = [n[1] * a[2]! - n[2] * a[1]!, n[2] * a[0]! - n[0] * a[2]!, n[0] * a[1]! - n[1] * a[0]!];
    const l1 = Math.hypot(...t1), u = t1.map(v => v / l1), w = [n[1] * u[2]! - n[2] * u[1]!, n[2] * u[0]! - n[0] * u[2]!, n[0] * u[1]! - n[1] * u[0]!];
    let k = 0;
    for (const d of [u, w]) for (const sgn of [1, -1]) {
      const o = project([q.p[0] + sgn * step * h * d[0]!, q.p[1] + sgn * step * h * d[1]!, q.p[2] + sgn * step * h * d[2]!]);
      if (!o) continue;
      const dp = [o.p[0] - q.p[0], o.p[1] - q.p[1], o.p[2] - q.p[2]], dn = [o.n[0] - n[0], o.n[1] - n[1], o.n[2] - n[2]];
      const len = Math.hypot(...dp);
      if (len < 0.125 * step * h || dp[0]! * dn[0]! + dp[1]! * dn[1]! + dp[2]! * dn[2]! <= 0) continue;
      k = Math.max(k, Math.hypot(...dn) / len);
    }
    return k;
  };

  // A point's supporting planes: its normal's, and every adjacent cell's own tangent plane there
  // (at an edge or corner of a voxel solid the cells meeting at the point each carry one face's).
  const supportPlanes = (q: SurfacePoint): { p: V3; n: V3 }[] => {
    const g = q.p.map((v, a) => (v - field.origin[a]!) / h);
    const span = g.map(v => { const c = Math.floor(v), f = v - c; return f < 1e-4 ? [c - 1, c] : f > 1 - 1e-4 ? [c, c + 1] : [c]; });
    const out = [{ p: q.p, n: q.n }];
    for (const cz of span[2]!) for (const cy of span[1]!) for (const cx of span[0]!) {
      const cell = [cx, cy, cz], b = cell.map(v => Math.floor(v / SOLID_BRICK_CELLS)), brick = field.bricks.get(b.join(","));
      if (!brick) continue;
      const l = g.map((v, a) => Math.min(cell[a]! + 1 - 1e-9, Math.max(cell[a]!, v)) - b[a]! * SOLID_BRICK_CELLS) as unknown as V3;
      const t = trilinear(brick.values, l), gl = Math.hypot(t[0], t[1], t[2]) / h;
      // A cell whose gradient (nearly) vanishes there holds a concave crease or the solid's inside: no plane.
      if (!(gl > 0.35)) continue;
      const n: V3 = [t[0] / h / gl, t[1] / h / gl, t[2] / h / gl];
      if (out.every(o => Math.abs(o.n[0] * n[0] + o.n[1] * n[1] + o.n[2] * n[2] - 1) > 1e-12)) out.push({ p: q.p, n });
    }
    return out;
  };

  // ---- the samples: per kept sample its point, normal, friction, curvature and spacing.
  const pts: SurfacePoint[] = [], spacingOf: number[] = [], supports: { p: V3; n: V3 }[][] = [];
  const HC = h / 8, hashOf = (p: V3, c: number) => p.map(v => Math.floor(v / c));
  const fine = new Map<string, number[]>();
  const keep = (q: SurfacePoint, planes: { p: V3; n: V3 }[], spacing: number) => {
    const minSep = 0.5 * spacing, c = hashOf(q.p, HC), R = Math.ceil(minSep / HC);
    for (let dz = -R; dz <= R; dz++) for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
      for (const j of fine.get(`${c[0]! + dx},${c[1]! + dy},${c[2]! + dz}`) ?? []) {
        const o = pts[j]!.p;
        if (Math.hypot(o[0] - q.p[0], o[1] - q.p[1], o[2] - q.p[2]) < minSep) return;
      }
    }
    const key = c.join(","), l = fine.get(key);
    if (l) l.push(pts.length); else fine.set(key, [pts.length]);
    pts.push(q); spacingOf.push(spacing); supports.push(planes);
  };
  const found: { q: SurfacePoint; k: number; spacing: number; planes: { p: V3; n: V3 }[] }[] = [];
  for (const brick of field.bricks.values()) {
    const v = brick.values;
    for (let z = 0; z < SOLID_BRICK_CELLS; z++) for (let y = 0; y < SOLID_BRICK_CELLS; y++) for (let x = 0; x < SOLID_BRICK_CELLS; x++) {
      let lo = Infinity, hi = -Infinity;
      for (let k = 0; k < 8; k++) {
        const s = v[nodeIndex(x + (k & 1), y + ((k >> 1) & 1), z + (k >> 2))]!;
        lo = Math.min(lo, s); hi = Math.max(hi, s);
      }
      if (lo > 0 || hi < 0) continue;
      const g = cellGradientRange(v, [x, y, z]);
      if (Math.max(g.hi[0] - g.lo[0], g.hi[1] - g.lo[1], g.hi[2] - g.lo[2]) <= 1e-6 * h) continue;
      const c = [0, 1, 2].map(a => field.origin[a]! + (brick.brick[a]! * SOLID_BRICK_CELLS + [x, y, z][a]! + 0.5) * h);
      const centre = project(c as unknown as V3);
      if (!centre) continue;
      const kc = kappa(centre, CELL_STEP);
      if (kc <= kappaMin) continue;
      const m = kc * h < SOLID_SAMPLE_SHARP_TURN ? 1 : 2;
      if (m === 1) { found.push({ q: centre, k: kc, spacing: h, planes: supportPlanes(centre) }); continue; }
      // A sharp cell is seeded at its corners too: a voxel solid's corner is a node, which no
      // projection from inside the cell reaches, and the samples' hull would cut it off.
      const seeds = [...Array(m * m * m).keys()].map(s => [s % m, Math.floor(s / m) % m, Math.floor(s / (m * m))].map(i => ((i + 0.5) / m - 0.5) * h))
        .concat([...Array(8).keys()].map(k => [k & 1, (k >> 1) & 1, k >> 2].map(b => (b - 0.5) * h)));
      for (const o of seeds) {
        const q = project([c[0]! + o[0]!, c[1]! + o[1]!, c[2]! + o[2]!]);
        if (!q) continue;
        const kq = kappa(q, localStep);
        if (kq > kappaMin) found.push({ q, k: kq, spacing: h / m, planes: supportPlanes(q) });
      }
    }
  }
  // Corners (most cells' planes meeting) first, then edges, then the sharpest curvature: a nearby
  // sample on a face never displaces the one on the feature.
  found.sort((a, b) => b.planes.length - a.planes.length || b.k - a.k);
  for (const f of found) keep(f.q, f.planes, f.spacing);

  // ---- the lifts: each sample's fan of holes, measured; a hole bulging past the ceiling is
  // refined (its centre and edge midpoints sampled at half the spacing) and measured again.
  const HN = 0.5 * h, coarse = new Map<string, number[]>();
  const index = (from: number) => {
    for (let i = from; i < pts.length; i++) { const k = hashOf(pts[i]!.p, HN).join(","), l = coarse.get(k); if (l) l.push(i); else coarse.set(k, [i]); }
  };
  const within = (p: V3, rad: number): number[] => {
    const c0 = p.map(v => Math.floor((v - rad) / HN)), c1 = p.map(v => Math.floor((v + rad) / HN)), out: number[] = [];
    for (let z = c0[2]!; z <= c1[2]!; z++) for (let y = c0[1]!; y <= c1[1]!; y++) for (let x = c0[0]!; x <= c1[0]!; x++)
      for (const j of coarse.get(`${x},${y},${z}`) ?? []) {
        const o = pts[j]!.p;
        if (Math.hypot(o[0] - p[0], o[1] - p[1], o[2] - p[2]) <= rad) out.push(j);
      }
    return out;
  };
  const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const ceiling = SOLID_SAMPLE_MAX_LIFT * h;
  // The lift of every sample, and the holes bulging past the ceiling.
  const measure = (): { lift: Float64Array; over: number[][]; edges: number[] } => {
    const lift = new Float64Array(pts.length), over: number[][] = [], done = new Set<string>(), edges: number[] = [];
    for (let i = 0; i < pts.length; i++) {
      const qi = pts[i]!, n = qi.n, reach = 1.6 * spacingOf[i]!;
      const a = Math.abs(n[0]) < 0.6 ? [1, 0, 0] as V3 : Math.abs(n[1]) < 0.6 ? [0, 1, 0] as V3 : [0, 0, 1] as V3;
      const t1 = cross(n, a), l1 = Math.hypot(...t1), u: V3 = [t1[0] / l1, t1[1] / l1, t1[2] / l1], w = cross(n, u);
      const fan = within(qi.p, reach).filter(j => j !== i).map(j => { const d = sub(pts[j]!.p, qi.p); return { j, ang: Math.atan2(dot(d, w), dot(d, u)) }; })
        .sort((x, y) => x.ang - y.ang);
      for (const F of fan) if (F.j > i) edges.push(i, F.j);
      if (fan.length < 2) continue;
      for (let k = 0; k < fan.length; k++) {
        const A = fan[k]!, B = fan[(k + 1) % fan.length]!;
        let gap = B.ang - A.ang; if (gap <= 0) gap += 2 * Math.PI;
        // A gap of half a turn or more faces the edge of the sampled surface: no hole there.
        if (gap >= Math.PI || fan.length < 3 && k > 0) continue;
        const key = [i, A.j, B.j].sort((x, y) => x - y).join(",");
        if (done.has(key)) continue;
        done.add(key);
        // So does a wide gap whose middle holds no surface the seeding would sample: a concave
        // crease (no normal) or flat surface just past the last row. (On a curved boundary row
        // the gap falls a little short of half a turn.)
        if (gap > 0.5 * Math.PI) {
          const ang = A.ang + 0.5 * gap, sp = spacingOf[i]!;
          const probe = project([0, 1, 2].map(a => qi.p[a]! + sp * (Math.cos(ang) * u[a]! + Math.sin(ang) * w[a]!)) as unknown as V3);
          if (!probe || kappa(probe, localStep) <= kappaMin) continue;
        }
        // Only a facet of the samples' local convex hull is crossed by a face resting on them: a
        // triangle with a neighbour outside its plane lies under the hull, which other facets cover.
        const pa = pts[A.j]!.p, pb = pts[B.j]!.p;
        let N = cross(sub(pa, qi.p), sub(pb, qi.p));
        const nl = Math.hypot(...N);
        if (!(nl > 0)) continue;
        const sgn = dot(N, n) + dot(N, pts[A.j]!.n) + dot(N, pts[B.j]!.n) < 0 ? -1 / nl : 1 / nl;
        N = [N[0] * sgn, N[1] * sgn, N[2] * sgn];
        const tol = 1e-9 * spacingOf[i]!;
        if (fan.some(F => F.j !== A.j && F.j !== B.j && dot(sub(pts[F.j]!.p, qi.p), N) > tol)) continue;
        // Measured along the samples' mean normal, the direction they are lifted in (a sliver
        // standing across the surface has a facet normal along it, which measures nothing).
        const D = [0, 1, 2].map(a => n[a]! + pts[A.j]!.n[a]! + pts[B.j]!.n[a]!) as unknown as V3, dl = Math.hypot(...D);
        const bulge = roofHeight([qi.p, pa, pb], [D[0] / dl, D[1] / dl, D[2] / dl], [...supports[i]!, ...supports[A.j]!, ...supports[B.j]!]);
        for (const j of [i, A.j, B.j]) lift[j] = Math.max(lift[j]!, bulge);
        if (bulge > ceiling) over.push([i, A.j, B.j]);
      }
    }
    return { lift, over, edges };
  };
  index(0);
  let measured = measure();
  while (measured.over.length) {
    const before = pts.length;
    for (const tri of measured.over) {
      const spacing = 0.5 * Math.min(...tri.map(j => spacingOf[j]!));
      // Refining stops at the ceiling's own scale: below it the bulge cannot be resolved.
      if (spacing < ceiling) continue;
      const [P0, P1, P2] = tri.map(j => pts[j]!.p) as [V3, V3, V3];
      for (const wts of [[1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]]) {
        const ws = wts[0]! + wts[1]! + wts[2]!;
        const q = project([0, 1, 2].map(a => (wts[0]! * P0[a]! + wts[1]! * P1[a]! + wts[2]! * P2[a]!) / ws) as unknown as V3);
        if (q) keep(q, supportPlanes(q), spacing);
      }
    }
    if (pts.length === before) break;
    index(before);
    measured = measure();
  }
  const S = pts.length, lift = measured.lift;
  const out = new Float32Array(SOLID_SAMPLE_WORDS * S);
  for (let i = 0; i < S; i++) {
    const q = pts[i]!, l = lift[i]!;
    if (l > SOLID_SAMPLE_MAX_LIFT * h)
      throw new RangeError(`Cloth solid samples: the surface between the samples near (${q.p.map(v => v.toFixed(4)).join(", ")}) bulges ${l.toExponential(2)} m, above h / 16 = ${ceiling.toExponential(2)} m, with samples refined to ${spacingOf[i]!.toExponential(2)} m`);
    out.set([q.p[0] + l * q.n[0], q.p[1] + l * q.n[1], q.p[2] + l * q.n[2], q.mu], SOLID_SAMPLE_WORDS * i);
  }
  return { samples: out, edges: Uint32Array.from(measured.edges) };
}
