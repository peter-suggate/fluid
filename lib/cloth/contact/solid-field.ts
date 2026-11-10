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
 * A missing brick reads `band`, so a round's motion longer than band / L_max
 * cannot be certified from the brick table alone, however far the solid is. The
 * clearance table certifies it: per block of 4^3 cells (two per brick and axis,
 * over every brick of the box) K_b = f(c_b) - sqrt(3) h, with f the source's
 * union (1-Lipschitz) at the block centre c_b. A stored node holds a union over a
 * subset of the parts, so it is at least f there, and a point's trilinear value
 * mixes nodes within sqrt(3) h of it. So the field is at least
 * min(band, K_b - |x0 - c_b| - rho) at every point within rho of any x0 (a
 * missing brick reads band): with band >= r, a ball whose bound clears r needs
 * no solid terms.
 *
 * GPU image (u32 words, appended to the contact buffer at a word offset base):
 *   header  [0] slots - 1  [1] brick count  [2] cell (f32)  [3] band (f32)
 *           [4..6] origin (f32)  [7] L_max (f32)  [8..10] bricks per axis
 *           [11] word offset of the clearance table from base (0: none)
 *           [12] 1 when the field's exterior is clear (every part's box lies band
 *           inside the lattice, so outside it phi >= band, as a missing brick
 *           reads); 0 when a part reaches past the box (the scene container's
 *           shell): outside the lattice the image knows nothing  [13..15] 0
 *   slots   (bx, by, bz, record word offset from base); an empty slot has w = OGC_NONE
 *   records 729 node values (f32), x fastest, padded to 732 words, then the
 *           node friction coefficients, one byte per node (mu * 100), four per word
 *   clearance  K_b (f32, rounded down) per block, x fastest, 2 * dims per axis
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
export const SOLID_HEADER_WORDS = 16;
/** Cells per side of a clearance block (two blocks per brick and axis). */
export const SOLID_CLEARANCE_CELLS = 4;
/**
 * Slack (m) taken off every clearance for the GPU's f32 evaluation of
 * K_b - |x0 - c_b|: a few ulps of coordinates of a few metres.
 */
export const SOLID_CLEARANCE_ROUNDING = 1e-5;
/** Word offset of the friction bytes in a record. */
export const SOLID_MU_WORD = 4 * Math.ceil(NODES / 4);
export const SOLID_RECORD_WORDS = SOLID_MU_WORD + 4 * Math.ceil(NODES / 16);
/** Friction coefficients are stored per node to 0.01, from 0 to 2.55. */
export const SOLID_MU_SCALE = 100;

/** The largest f32 at or below x: a lower bound stored as f32 must not round up. */
export function f32Down(x: number): number { return -f32Up(-x); }

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
  /** Any 1-Lipschitz signed distance whose solid lies inside [lo, hi] (outside the box, at least its distance to the box). */
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
  /** K_b per clearance block (2 * dims per axis, x fastest), f32 rounded down: the field is at least min(band, K_b - |p - c_b|) at every p. */
  clearance: Float32Array;
  /** Every part's box lies band inside the lattice: outside the lattice phi >= band (a missing brick). */
  exteriorClear: boolean;
  /** Largest per-brick L, rounded up to f32 (the image's word 7 exactly). */
  maxLipschitz: number;
  /** The GPU image. */
  words: Uint32Array<ArrayBuffer>;
  /**
   * The source's own signed distance (the union of its parts, Float64), CPU only: the surface samples
   * stand on it where the trilinear field rounds the solid more than the samples' ceiling, and the
   * authoring check measures the field's error against it.
   */
  truth: (p: V3) => number;
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
  const clearanceAt = SOLID_HEADER_WORDS + 4 * slots + bricks.length * SOLID_RECORD_WORDS;
  const words = new Uint32Array(clearanceAt + field.clearance.length);
  const f = new Float32Array(words.buffer);
  words[0] = slots - 1; words[1] = bricks.length; f[2] = field.cell; f[3] = field.band;
  f[4] = field.origin[0]; f[5] = field.origin[1]; f[6] = field.origin[2]; f[7] = field.maxLipschitz;
  words[8] = field.dims[0]; words[9] = field.dims[1]; words[10] = field.dims[2]; words[11] = clearanceAt; words[12] = field.exteriorClear ? 1 : 0;
  f.set(field.clearance, clearanceAt);
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
  return finish({ cell, band, origin, dims, bricks, clearance: clearanceOf(parts, cell, band, origin, dims), exteriorClear: exteriorClearOf(parts, cell, band, origin, dims), maxLipschitz: 0, truth: truthOf(parts, band) });
}

/**
 * The union of the parts' distances. A part whose box lies further than band from p stands in by
 * that box distance (a lower bound on its distance, at least band), which changes no value within
 * band of the solid.
 */
function truthOf(parts: readonly Part[], band: number): (p: V3) => number {
  return p => {
    let v = Infinity;
    for (const part of parts) {
      const d = Math.hypot(Math.max(0, part.lo[0] - p[0], p[0] - part.hi[0]), Math.max(0, part.lo[1] - p[1], p[1] - part.hi[1]), Math.max(0, part.lo[2] - p[2], p[2] - part.hi[2]));
      v = Math.min(v, d > band ? d : part.sdf(p));
    }
    return v;
  };
}

/**
 * K_b = f(c_b) - sqrt(3) h per clearance block, f the union of every part (each
 * 1-Lipschitz, so the union is), rounded down to f32 with the GPU's rounding slack.
 * A part's box further than band from c_b stands in for its distance there.
 */
function clearanceOf(parts: readonly Part[], cell: number, band: number, origin: V3, dims: V3): Float32Array {
  const per = SOLID_BRICK_CELLS / SOLID_CLEARANCE_CELLS, n = dims.map(d => per * d), span = SOLID_CLEARANCE_CELLS * cell;
  const out = new Float32Array(n[0]! * n[1]! * n[2]!), slack = Math.sqrt(3) * cell + SOLID_CLEARANCE_ROUNDING;
  const c: [number, number, number] = [0, 0, 0];
  for (let z = 0, i = 0; z < n[2]!; z++) for (let y = 0; y < n[1]!; y++) for (let x = 0; x < n[0]!; x++, i++) {
    c[0] = origin[0] + (x + 0.5) * span; c[1] = origin[1] + (y + 0.5) * span; c[2] = origin[2] + (z + 0.5) * span;
    let fc = Infinity;
    for (const part of parts) {
      // A part further than band from c is at least its box's distance there (a signed distance
      // of a solid inside the box), which the near-part rule of buildBrick relies on as well.
      const d = Math.hypot(Math.max(0, part.lo[0] - c[0], c[0] - part.hi[0]), Math.max(0, part.lo[1] - c[1], c[1] - part.hi[1]), Math.max(0, part.lo[2] - c[2], c[2] - part.hi[2]));
      fc = Math.min(fc, d > band ? d : part.sdf(c));
    }
    if (Number.isNaN(fc)) throw new RangeError("Cloth solid distance must not be NaN");
    out[i] = Number.isFinite(fc) ? f32Down(fc - slack) : (fc > 0 ? 3.0e38 : -3.0e38);
  }
  return out;
}

/**
 * Whether every part's box lies band inside the lattice [origin, origin + 8 h dims]: a part's
 * solid lies inside its box and its distance is at least the box's outside it, so the field
 * outside the lattice is then at least band (a moving body's padded box). A scene's container
 * reaches past the box, and outside the lattice the image knows nothing.
 */
function exteriorClearOf(parts: readonly Part[], cell: number, band: number, origin: V3, dims: V3): boolean {
  const span = SOLID_BRICK_CELLS * cell;
  return parts.every(p => p.lo.every((v, a) => v - band >= origin[a]!) && p.hi.every((v, a) => v + band <= origin[a]! + span * dims[a]!));
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
 * brick is kept as it is (the source is unchanged there), and the clearance
 * table is rebuilt whole (an edit moves the distance everywhere). Edit-time CPU work.
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
  return finish({ cell: field.cell, band: field.band, origin: field.origin, dims: field.dims, bricks, clearance: clearanceOf(parts, field.cell, field.band, field.origin, field.dims), exteriorClear: exteriorClearOf(parts, field.cell, field.band, field.origin, field.dims), maxLipschitz: 0, truth: truthOf(parts, field.band) });
}

/** The clearance table's bound at p (ogcSolidClearance): within rho of p the field is at least min(band, this - rho). */
export function clothSolidClearance(field: ClothSolidField, p: V3): number {
  const per = SOLID_BRICK_CELLS / SOLID_CLEARANCE_CELLS, span = SOLID_CLEARANCE_CELLS * field.cell, n = field.dims.map(d => per * d);
  const b = p.map((v, a) => Math.min(n[a]! - 1, Math.max(0, Math.floor((v - field.origin[a]!) / span))));
  const c = b.map((v, a) => field.origin[a]! + (v + 0.5) * span);
  return field.clearance[b[0]! + n[0]! * (b[1]! + n[1]! * b[2]!)]! - Math.hypot(p[0] - c[0]!, p[1] - c[1]!, p[2] - c[2]!);
}

/** The field at a cloth-local point: phi (m), its gradient, the nearest node's friction, and whether a stored brick holds the point (else phi reads band). */
export function sampleClothSolidField(field: ClothSolidField, p: V3): { phi: number; grad: V3; mu: number; stored: boolean } {
  const o = sampleInto(field, brickTable(field), p[0], p[1], p[2], new Float64Array(SAMPLE_OUT));
  if (!o[5]) return { phi: field.band, grad: [0, 0, 0], mu: 0, stored: false };
  return { phi: o[0]!, grad: [o[1]!, o[2]!, o[3]!], mu: o[4]!, stored: true };
}

// Setup-time sampling without allocation (the samples' build reads the field a few million
// times): bricks by a numeric key, and the value, gradient and friction written into `out`.
const brickKey = (x: number, y: number, z: number) => ((z + 65536) * 131072 + (y + 65536)) * 131072 + (x + 65536);
const brickTables = new WeakMap<ClothSolidField, Map<number, ClothSolidBrick>>();
function brickTable(field: ClothSolidField): Map<number, ClothSolidBrick> {
  let t = brickTables.get(field);
  if (!t) {
    t = new Map();
    for (const b of field.bricks.values()) t.set(brickKey(b.brick[0], b.brick[1], b.brick[2]), b);
    brickTables.set(field, t);
  }
  return t;
}
const SAMPLE_OUT = 6;
/** out = [phi, d/dx, d/dy, d/dz (per m), mu, stored (1 or 0)]; the same arithmetic as trilinear. */
function sampleInto(field: ClothSolidField, table: Map<number, ClothSolidBrick>, px: number, py: number, pz: number, out: Float64Array): Float64Array {
  const B = SOLID_BRICK_CELLS, h = field.cell, O = field.origin;
  const qx = (px - O[0]) / h, qy = (py - O[1]) / h, qz = (pz - O[2]) / h;
  const bx = Math.floor(qx / B), by = Math.floor(qy / B), bz = Math.floor(qz / B);
  const brick = table.get(brickKey(bx, by, bz));
  if (!brick) { out[0] = field.band; out[1] = 0; out[2] = 0; out[3] = 0; out[4] = 0; out[5] = 0; return out; }
  const lx = qx - bx * B, ly = qy - by * B, lz = qz - bz * B;
  const cx = Math.min(B - 1, Math.max(0, Math.floor(lx))), cy = Math.min(B - 1, Math.max(0, Math.floor(ly))), cz = Math.min(B - 1, Math.max(0, Math.floor(lz)));
  const fx = lx - cx, fy = ly - cy, fz = lz - cz, values = brick.values;
  let gx = 0, gy = 0, gz = 0, phi = 0;
  for (let k = 0; k < 8; k++) {
    const b0 = k & 1, b1 = (k >> 1) & 1, b2 = k >> 2;
    const w0 = b0 ? fx : 1 - fx, w1 = b1 ? fy : 1 - fy, w2 = b2 ? fz : 1 - fz;
    const v = values[nodeIndex(cx + b0, cy + b1, cz + b2)]!;
    phi += v * w0 * w1 * w2;
    gx += v * (b0 ? 1 : -1) * w1 * w2;
    gy += v * (b1 ? 1 : -1) * w2 * w0;
    gz += v * (b2 ? 1 : -1) * w0 * w1;
  }
  const nx = Math.min(B, Math.max(0, Math.round(lx))), ny = Math.min(B, Math.max(0, Math.round(ly))), nz = Math.min(B, Math.max(0, Math.round(lz)));
  out[0] = phi; out[1] = gx / h; out[2] = gy / h; out[3] = gz / h;
  out[4] = brick.mu[nodeIndex(nx, ny, nz)]! / SOLID_MU_SCALE; out[5] = 1;
  return out;
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
 * left the bound is infinite. Also returns where over the triangle (barycentric u, v on its second and
 * third corners) the bound is attained: the roof's apex.
 */
function roofHeight(P: readonly [V3, V3, V3], N: V3, supports: readonly { p: V3; n: V3 }[]): { height: number; u: number; v: number } {
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
  if (!planes.length) return { height: Infinity, u: 1 / 3, v: 1 / 3 };
  const at = (u: number, v: number) => Math.min(...planes.map(q => q[0] + q[1] * u + q[2] * v));
  const inside = (u: number, v: number) => u >= -1e-12 && v >= -1e-12 && u + v <= 1 + 1e-12;
  let best = -Infinity, bu = 0, bv = 0;
  const offer = (u: number, v: number) => { const y = at(u, v); if (y > best) { best = y; bu = u; bv = v; } };
  offer(0, 0); offer(1, 0); offer(0, 1);
  // Two heights equal along a triangle edge: edges u = 0, v = 0, u + v = 1 as (u, v) = o + t d.
  const edges: [number, number, number, number][] = [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, -1, 1]];
  for (let i = 0; i < planes.length; i++) for (let j = i + 1; j < planes.length; j++) {
    const A = planes[i]!, B = planes[j]!, d = [A[0] - B[0], A[1] - B[1], A[2] - B[2]];
    for (const [ou, ov, du, dv] of edges) {
      const den = d[1]! * du + d[2]! * dv;
      if (Math.abs(den) < 1e-30) continue;
      const t = -(d[0]! + d[1]! * ou + d[2]! * ov) / den;
      if (t >= 0 && t <= 1) offer(ou + t * du, ov + t * dv);
    }
    // All three equal inside.
    for (let k = j + 1; k < planes.length; k++) {
      const C = planes[k]!, a1 = A[1] - B[1], b1 = A[2] - B[2], r1 = B[0] - A[0], a2 = A[1] - C[1], b2 = A[2] - C[2], r2 = C[0] - A[0];
      const det = a1 * b2 - a2 * b1;
      if (Math.abs(det) < 1e-30) continue;
      const u = (r1 * b2 - r2 * b1) / det, v = (a1 * r2 - a2 * r1) / det;
      if (inside(u, v)) offer(u, v);
    }
  }
  return { height: Math.max(0, best), u: bu, v: bv };
}

/** The distance from p to the triangle abc (Ericson's closest point, Real-Time Collision Detection 5.1.5). */
function pointTriangleDistance(p: V3, a: V3, b: V3, c: V3): number {
  const sub = (u: V3, v: V3): V3 => [u[0] - v[0], u[1] - v[1], u[2] - v[2]];
  const dot = (u: V3, v: V3) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  const ab = sub(b, a), ac = sub(c, a), ap = sub(p, a), d1 = dot(ab, ap), d2 = dot(ac, ap);
  const at = (q: V3) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
  const on = (u: V3, s: number, v: V3, t: number): V3 => [a[0] + s * u[0] + t * v[0], a[1] + s * u[1] + t * v[1], a[2] + s * u[2] + t * v[2]];
  if (d1 <= 0 && d2 <= 0) return at(a);
  const bp = sub(p, b), d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return at(b);
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) return at(on(ab, d1 / (d1 - d3), ac, 0));
  const cp = sub(p, c), d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return at(c);
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) return at(on(ab, 0, ac, d2 / (d2 - d6)));
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) { const w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); return at([b[0] + w * (c[0] - b[0]), b[1] + w * (c[1] - b[1]), b[2] + w * (c[2] - b[2])]); }
  const den = 1 / (va + vb + vc);
  return at(on(ab, vb * den, ac, vc * den));
}

/**
 * Whether the unit vector N lies in the convex cone of the unit normals ns, or within the angle eps
 * of it: along one of them, within eps of a face of the cone (two normals, nonnegative weights), or
 * a nonnegative combination of three (Caratheodory: in three dimensions a cone's point is one of at
 * most three of its generators).
 */
function inNormalCone(N: V3, ns: readonly V3[], eps: number): boolean {
  const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const ce = Math.cos(eps), se = Math.sin(eps);
  for (const n of ns) if (dot(N, n) >= ce) return true;
  for (let i = 0; i < ns.length; i++) for (let j = i + 1; j < ns.length; j++) {
    const a = ns[i]!, b = ns[j]!, c = cross(a, b), cl = Math.hypot(...c);
    if (!(cl > 1e-9)) continue;
    // Within eps of the plane of a and b, and its projection there between them.
    if (Math.abs(dot(N, c)) / cl > se) continue;
    if (dot(cross(a, N), c) >= 0 && dot(cross(N, b), c) >= 0) return true;
  }
  for (let i = 0; i < ns.length; i++) for (let j = i + 1; j < ns.length; j++) for (let k = j + 1; k < ns.length; k++) {
    const a = ns[i]!, b = ns[j]!, c = ns[k]!, det = dot(a, cross(b, c));
    if (!(Math.abs(det) > 1e-9)) continue;
    // N = x a + y b + z c (Cramer's rule), every weight nonnegative.
    const x = dot(N, cross(b, c)) / det, y = dot(a, cross(N, c)) / det, z = dot(a, cross(b, N)) / det;
    if (x >= 0 && y >= 0 && z >= 0) return true;
  }
  return false;
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

/**
 * The samples' hull facets: the triangles (local indices, three words each, wound so the right-hand
 * normal points out of the solid) of the samples' local convex hull, the holes a face resting on the
 * samples spans (measured for the lift, below). Each lies on or outside the source's surface: on a
 * face of the source between samples standing on it, above a curved surface by its lift. A cloth
 * vertex kept outside every facet is kept outside the solid where the trilinear field rounds it.
 */
export function clothSolidSampleFacets(field: ClothSolidField, options?: ClothSolidSampleOptions): Uint32Array {
  return clothSolidSampleHull(field, options).facets;
}

/**
 * The deepest the field reads a point of the source outside it at a concave crease, past the ceiling (0
 * when it never does): the authoring check leaves concave creases to the field (see convexAt), so this
 * is the field's own open error there, reported, never hidden.
 */
export function clothSolidConcaveRounding(field: ClothSolidField): number {
  return clothSolidSampleHull(field).concaveDepth;
}

function clothSolidSampleHull(field: ClothSolidField, options?: ClothSolidSampleOptions): SolidSampleHull {
  if (options && !(options.r > 0 && options.edge > 0)) throw new RangeError("Cloth solid samples need r > 0 and edge > 0");
  let cached = sampleCache.get(field);
  if (!cached) sampleCache.set(field, cached = buildSolidSamples(field));
  return cached;
}

/** A field is immutable once built (a rebuild returns a new one): its samples are computed once. */
type SolidSampleHull = { samples: Float32Array; edges: Uint32Array; facets: Uint32Array; concaveDepth: number };
const sampleCache = new WeakMap<ClothSolidField, SolidSampleHull>();

type SurfacePoint = { p: V3; n: V3; mu: number };

function buildSolidSamples(field: ClothSolidField): SolidSampleHull {
  // A cell's turn is read two cells out (it finds an edge the centre's own cell misses); a seed's
  // a tenth of a cell out, so a point on a voxel solid's flat face near its edge is not kept.
  const h = field.cell, kappaMin = SOLID_SAMPLE_TURN_FLOOR / h, CELL_STEP = 2, localStep = 0.1;
  const table = brickTable(field), so = new Float64Array(SAMPLE_OUT);
  const phiAt = (x: number, y: number, z: number) => sampleInto(field, table, x, y, z, so)[0]!;
  // Newton onto phi = 0 along the gradient; undefined when it leaves the stored field or drifts more than a cell.
  const project = (p0: V3): SurfacePoint | undefined => {
    let x = p0[0], y = p0[1], z = p0[2];
    // Tight enough that a point at a voxel solid's edge, where phi vanishes quadratically and
    // Newton only halves its distance per step, ends within 1e-4 cells of it.
    for (let i = 0; i < 40; i++) {
      const s = sampleInto(field, table, x, y, z, so);
      if (!s[5]) return undefined;
      const phi = s[0]!, gx = s[1]!, gy = s[2]!, gz = s[3]!, mu = s[4]!;
      if (Math.abs(phi) < 1e-10 * h) {
        if (Math.hypot(x - p0[0], y - p0[1], z - p0[2]) > 2.5 * h) return undefined;
        // The normal from central differences a quarter cell wide: at a lattice edge the
        // one-sided cell gradient may vanish. A concave crease (phi ~ xz / h) has none: skip it.
        const d = 0.25 * h;
        const g0 = (phiAt(x + d, y, z) - phiAt(x - d, y, z)) / (2 * d);
        const g1 = (phiAt(x, y + d, z) - phiAt(x, y - d, z)) / (2 * d);
        const g2 = (phiAt(x, y, z + d) - phiAt(x, y, z - d)) / (2 * d);
        const gl = Math.hypot(g0, g1, g2);
        if (!(gl > 0.35)) return undefined;
        return { p: [x, y, z], n: [g0 / gl, g1 / gl, g2 / gl], mu };
      }
      // (A voxel solid's corner node may see no gradient in the cell holding it: test phi first.)
      const g2 = gx ** 2 + gy ** 2 + gz ** 2;
      if (!(g2 > 1e-18)) return undefined;
      x = x - phi * gx / g2; y = y - phi * gy / g2; z = z - phi * gz / g2;
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

  // ---- sharp features: a point near a convex edge or corner of the solid's faces moves onto it.
  // Where one face's plane is the field (a cell whose eight nodes are affine: trilinear
  // interpolation reproduces a plane exactly, the max of a wedge's planes inside and outside its
  // edges, a box inside), the cell gives that plane; the trilinear field rounds the faces' edges
  // over a cell, below the planes (it overestimates a convex field), so samples projected onto it
  // stand below the solid's sharp edge and the faces between them bulge over the samples by the
  // rounding. A point on the faces' edge (two planes) or corner (three), within a cell of the
  // projected point, outside the field and inside every nearby face's half-space, is on the
  // solid's own feature: the faces' planes there are its supports (the surface lies below each).
  // A plane is n . (x - o) + c, about its cell's node o (an offset about the frame's origin would
  // carry the f32 gradient's rounding |o| / h times over).
  type Plane = { n: V3; o: V3; c: number };
  const AFFINE_SPREAD = 1e-6 * h, planeCache = new Map<number, Plane | null>();
  const cellPlane = (cx: number, cy: number, cz: number): Plane | null => {
    const key = brickKey(cx, cy, cz);
    const hit = planeCache.get(key);
    if (hit !== undefined) return hit;
    let out: Plane | null = null;
    const b = [cx, cy, cz].map(c => Math.floor(c / SOLID_BRICK_CELLS)), brick = table.get(brickKey(b[0]!, b[1]!, b[2]!));
    if (brick) {
      const l = [cx - SOLID_BRICK_CELLS * b[0]!, cy - SOLID_BRICK_CELLS * b[1]!, cz - SOLID_BRICK_CELLS * b[2]!] as unknown as V3;
      const g = cellGradientRange(brick.values, l);
      if (Math.max(g.hi[0] - g.lo[0], g.hi[1] - g.lo[1], g.hi[2] - g.lo[2]) <= AFFINE_SPREAD) {
        const gr = [0, 1, 2].map(a => 0.5 * (g.hi[a]! + g.lo[a]!)), gl = Math.hypot(gr[0]!, gr[1]!, gr[2]!);
        if (gl > 0.35 * h) {
          // phi = v000 + gr . (x - x000) / h on the cell (gr: metres per cell).
          const x000 = [cx, cy, cz].map((c, a) => field.origin[a]! + c * h) as unknown as V3, v000 = brick.values[nodeIndex(l[0], l[1], l[2])]!;
          out = { n: [gr[0]! / gl, gr[1]! / gl, gr[2]! / gl], o: x000, c: v000 * h / gl };
        }
      }
    }
    planeCache.set(key, out);
    return out;
  };
  const planeAt = (P: Plane, x: readonly number[]) => P.n[0] * (x[0]! - P.o[0]) + P.n[1] * (x[1]! - P.o[1]) + P.n[2] * (x[2]! - P.o[2]) + P.c;
  const feature = (q: SurfacePoint): { q: SurfacePoint; planes: { p: V3; n: V3 }[] }[] | undefined => {
    const c = q.p.map((v, a) => Math.floor((v - field.origin[a]!) / h));
    const near: Plane[] = [];
    // The same face from another cell: its normal, and its value at the point, agree to rounding.
    const tol = 1e-5 * h;
    for (let dz = -2; dz <= 2; dz++) for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
      const P = cellPlane(c[0]! + dx, c[1]! + dy, c[2]! + dz);
      if (!P || Math.abs(planeAt(P, q.p)) > 2 * h) continue;
      if (near.every(o => !(o.n[0] * P.n[0] + o.n[1] * P.n[1] + o.n[2] * P.n[2] > 1 - 1e-7 && Math.abs(planeAt(o, q.p) - planeAt(P, q.p)) < tol))) near.push(P);
    }
    if (near.length < 2) return undefined;
    near.sort((A, B) => Math.abs(planeAt(A, q.p)) - Math.abs(planeAt(B, q.p)));
    // On the faces' feature: outside (or on) the field, in every nearby face's half-space.
    const valid = (x: V3) => near.every(P => planeAt(P, x) <= tol) && phiAt(x[0], x[1], x[2]) >= -tol;
    // A convex feature of the faces used: the solid there is the intersection of their
    // half-spaces, so a point outside one face and inside another is clear of it, where a
    // concave crease (a post standing on the ground: the union of the half-spaces) has it in
    // the solid. The trilinear field errs the safe way at both (it overestimates a convex field
    // and underestimates a concave one). And the projected point, on the field, lies under every
    // face used: where the field fills a crease past them they are no supports there.
    const convex = (x: V3, used: Plane[]) => {
      if (!used.every(P => planeAt(P, q.p) <= tol)) return false;
      for (const A of used) for (const B of used) if (A !== B) {
        const p = [0, 1, 2].map(a => x[a]! + 0.5 * h * (A.n[a]! - B.n[a]!));
        if (!(phiAt(p[0]!, p[1]!, p[2]!) > 0)) return false;
      }
      return true;
    };
    const made = (x: V3, used: Plane[]) => {
      const m = [0, 1, 2].map(a => used.reduce((s, P) => s + P.n[a]!, 0)), ml = Math.hypot(m[0]!, m[1]!, m[2]!);
      return { q: { p: x, n: [m[0]! / ml, m[1]! / ml, m[2]! / ml] as V3, mu: q.mu }, planes: used.map(P => ({ p: x, n: P.n })) };
    };
    const K = Math.min(near.length, 4);
    // An edge: the point of two planes' line nearest the projected point, within the rounding's
    // reach (the trilinear field rounds a feature over about a cell, so the seeds of the cells it
    // touches lie within two of it).
    const edge = (A: Plane, B: Plane) => {
      const ab = A.n[0] * B.n[0] + A.n[1] * B.n[1] + A.n[2] * B.n[2];
      if (!(1 - ab * ab > 1e-4)) return undefined;
      // x = q - alpha nA - beta nB on both planes.
      const ea = planeAt(A, q.p), eb = planeAt(B, q.p), den = 1 - ab * ab;
      const alpha = (ea - ab * eb) / den, beta = (eb - ab * ea) / den;
      const x = [0, 1, 2].map(a => q.p[a]! - alpha * A.n[a]! - beta * B.n[a]!) as unknown as V3;
      return Math.hypot(x[0] - q.p[0], x[1] - q.p[1], x[2] - q.p[2]) <= 2 * h && valid(x) && convex(x, [A, B]) ? made(x, [A, B]) : undefined;
    };
    // A corner: three planes meeting as near.
    const corner = (P: Plane, Q: Plane, R: Plane) => {
      const A = P.n, B = Q.n, C = R.n;
      const bc = [B[1] * C[2] - B[2] * C[1], B[2] * C[0] - B[0] * C[2], B[0] * C[1] - B[1] * C[0]];
      const det = A[0] * bc[0]! + A[1] * bc[1]! + A[2] * bc[2]!;
      if (!(Math.abs(det) > 1e-3)) return undefined;
      const ca = [C[1] * A[2] - C[2] * A[1], C[2] * A[0] - C[0] * A[2], C[0] * A[1] - C[1] * A[0]];
      const ab = [A[1] * B[2] - A[2] * B[1], A[2] * B[0] - A[0] * B[2], A[0] * B[1] - A[1] * B[0]];
      // n . (x - q) = -plane(q) for each plane (Cramer's rule).
      const r = [-planeAt(P, q.p), -planeAt(Q, q.p), -planeAt(R, q.p)];
      const x = [0, 1, 2].map(a => q.p[a]! + (r[0]! * bc[a]! + r[1]! * ca[a]! + r[2]! * ab[a]!) / det) as unknown as V3;
      return Math.hypot(x[0] - q.p[0], x[1] - q.p[1], x[2] - q.p[2]) <= 2 * h && valid(x) && convex(x, [P, Q, R]) ? made(x, [P, Q, R]) : undefined;
    };
    // The nearest two faces' edge, else another pair's, and any corner of three as near: a seed by
    // a corner gives both (a sample on the corner, and the edge's own up to it).
    const out: { q: SurfacePoint; planes: { p: V3; n: V3 }[] }[] = [];
    for (let i = 0; i < K && !out.length; i++) for (let j = i + 1; j < K && !out.length; j++) { const f = edge(near[i]!, near[j]!); if (f) out.push(f); }
    for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) for (let k = j + 1; k < K; k++) { const f = corner(near[i]!, near[j]!, near[k]!); if (f) out.push(f); }
    return out.length ? out : undefined;
  };
  // ---- the true surface. On a convex part the trilinear field lies above the source's distance
  // (Jensen), so its zero set lies inside the solid: by at most (h^2 / 8)(kappa1 + kappa2) where the
  // surface is smooth with principal curvatures kappa (the interpolant's error, h^2 / 8 times the
  // Hessian's trace; the distance's Hessian has the curvatures as its tangential eigenvalues), so
  // within the ceiling h / 16 wherever kappa1 + kappa2 <= 1 / (2 h): a sphere of radius 4 h or more,
  // a cylinder of 2 h. A sharp edge or corner is rounded over about a cell, by up to a fifth of one
  // on a slanted 90 degree edge. Where the field's own surface point lies further than the ceiling
  // inside the source's, the seed's sample stands on the source's surface instead: its nearest point
  // there, with the source's tangent plane (a support of a convex solid) as its plane. A concave
  // crease is the other way round (the field lies outside it) and keeps the field's point.
  const truth = field.truth, ceiling = SOLID_SAMPLE_MAX_LIFT * h, TE = 1e-4 * h;
  const truthGrad = (x: V3): V3 => [
    (truth([x[0] + TE, x[1], x[2]]) - truth([x[0] - TE, x[1], x[2]])) / (2 * TE),
    (truth([x[0], x[1] + TE, x[2]]) - truth([x[0], x[1] - TE, x[2]])) / (2 * TE),
    (truth([x[0], x[1], x[2] + TE]) - truth([x[0], x[1], x[2] - TE])) / (2 * TE)];
  // A point of the source's surface: from inside it, the first crossing along dir (the field's
  // outward normal, so a face buried in another part, where the union's distance vanishes but its
  // solid goes on, is never taken); from outside, Newton (the union's distance is exact outside).
  // Undefined further than two cells away or where the source has no gradient.
  const onTruth = (p0: V3, dir: V3): { p: V3; n: V3 } | undefined => {
    let x: V3 = p0;
    const t0 = truth(p0);
    if (t0 < 0) {
      const step = h / 16;
      let lo = 0, hi = 0;
      for (let s = step; s <= 2 * h + 1e-12; s += step) if (truth([p0[0] + s * dir[0], p0[1] + s * dir[1], p0[2] + s * dir[2]]) >= 0) { hi = s; break; } else lo = s;
      if (!(hi > 0)) return undefined;
      for (let i = 0; i < 40 && hi - lo > 1e-9 * h; i++) {
        const m = 0.5 * (lo + hi);
        if (truth([p0[0] + m * dir[0], p0[1] + m * dir[1], p0[2] + m * dir[2]]) >= 0) hi = m; else lo = m;
      }
      x = [p0[0] + hi * dir[0], p0[1] + hi * dir[1], p0[2] + hi * dir[2]];
    } else {
      for (let i = 0; i < 40; i++) {
        const t = truth(x), g = truthGrad(x), g2 = g[0] ** 2 + g[1] ** 2 + g[2] ** 2;
        if (!(g2 > 0.25)) return undefined;
        if (Math.abs(t) < 1e-9 * h) break;
        x = [x[0] - t * g[0] / g2, x[1] - t * g[1] / g2, x[2] - t * g[2] / g2];
      }
      if (!(Math.abs(truth(x)) < 1e-9 * h)) return undefined;
    }
    if (Math.hypot(x[0] - p0[0], x[1] - p0[1], x[2] - p0[2]) > 2 * h) return undefined;
    const g = truthGrad(x), gl = Math.hypot(g[0], g[1], g[2]);
    if (!(gl > 0.5)) return undefined;
    return { p: x, n: [g[0] / gl, g[1] / gl, g[2] / gl] };
  };
  // Whether the source is locally convex at its surface point x (normal n): its tangent plane supports it
  // within half a cell (every point of it there lies outside the source). The samples' hull represents a
  // convex solid; a concave crease (a part standing on another) is the field's, as at its own creases.
  const convexAt = (x: V3, n: V3): boolean => {
    const a = Math.abs(n[0]) < 0.6 ? [1, 0, 0] as V3 : Math.abs(n[1]) < 0.6 ? [0, 1, 0] as V3 : [0, 0, 1] as V3;
    const t1: V3 = [n[1] * a[2] - n[2] * a[1], n[2] * a[0] - n[0] * a[2], n[0] * a[1] - n[1] * a[0]], l1 = Math.hypot(...t1);
    const u: V3 = [t1[0] / l1, t1[1] / l1, t1[2] / l1], w: V3 = [n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0]];
    for (let m = 0; m < 8; m++) {
      const c = Math.cos(m * Math.PI / 4), sn = Math.sin(m * Math.PI / 4);
      if (truth([0, 1, 2].map(i => x[i]! + 0.5 * h * (c * u[i]! + sn * w[i]!)) as unknown as V3) < -1e-6 * h) return false;
    }
    return true;
  };
  const trueSample = (q: SurfacePoint): { q: SurfacePoint; planes: { p: V3; n: V3 }[]; truth?: true } | undefined => {
    if (!(truth(q.p) < -ceiling)) return undefined;
    const t = onTruth(q.p, q.n);
    if (!t || !(phiAt(t.p[0], t.p[1], t.p[2]) >= -1e-5 * h) || !convexAt(t.p, t.n)) return undefined;
    return { q: { p: t.p, n: t.n, mu: q.mu }, planes: [{ p: t.p, n: t.n }], truth: true };
  };
  // A seed's samples: on the solid's feature when one is near, on the source's surface where the field
  // rounds it past the ceiling, else the projected point itself.
  const placed = (q: SurfacePoint, planes: () => { p: V3; n: V3 }[]): { q: SurfacePoint; planes: { p: V3; n: V3 }[]; truth?: true }[] => {
    const out: { q: SurfacePoint; planes: { p: V3; n: V3 }[]; truth?: true }[] = feature(q) ?? [], t = trueSample(q);
    if (t) out.push(t); else if (!out.length) out.push({ q, planes: planes() });
    return out;
  };

  // The authoring check's probes of a surface cell about its centre c: a lattice h / 4 apart.
  const PROBES = 4;
  const cellProbes = (c: readonly number[]): V3[] => Array.from({ length: PROBES ** 3 }, (_, k) => {
    const o = [k % PROBES, Math.floor(k / PROBES) % PROBES, Math.floor(k / PROBES ** 2)].map(i => ((i + 0.5) / PROBES - 0.5) * h);
    return [c[0]! + o[0]!, c[1]! + o[1]!, c[2]! + o[2]!] as V3;
  });

  // ---- the samples: per kept sample its point, normal, friction, curvature and spacing.
  // standsOnTruth: the sample stands on the source's surface where the field rounds it past the ceiling.
  const pts: SurfacePoint[] = [], spacingOf: number[] = [], supports: { p: V3; n: V3 }[][] = [], standsOnTruth: boolean[] = [];
  // Point hashes on numeric cell keys (cells within 2^16 of the cloth frame's origin).
  const cellAt = (v: number, c: number) => Math.floor(v / c);
  const HC = h / 2;
  const fine = new Map<number, number[]>();
  const keep = (q: SurfacePoint, planes: { p: V3; n: V3 }[], spacing: number, onSource = false, minSep = 0.5 * spacing) => {
    const R = Math.ceil(minSep / HC);
    const cx = cellAt(q.p[0], HC), cy = cellAt(q.p[1], HC), cz = cellAt(q.p[2], HC);
    for (let dz = -R; dz <= R; dz++) for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
      const l = fine.get(brickKey(cx + dx, cy + dy, cz + dz));
      if (l) for (const j of l) {
        const o = pts[j]!.p;
        if (Math.hypot(o[0] - q.p[0], o[1] - q.p[1], o[2] - q.p[2]) < minSep) return;
      }
    }
    const key = brickKey(cx, cy, cz), l = fine.get(key);
    if (l) l.push(pts.length); else fine.set(key, [pts.length]);
    pts.push(q); spacingOf.push(spacing); supports.push(planes); standsOnTruth.push(onSource);
  };
  const found: { q: SurfacePoint; k: number; spacing: number; planes: { p: V3; n: V3 }[]; truth?: true }[] = [];
  // Every surface cell the field does not hold affine: where it can round the source.
  const surfaceCells: number[][] = [];
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
      surfaceCells.push(c);
      // Wherever the field rounds the source past the ceiling, the source's surface is sampled: from each
      // probe of the authoring check (below) outside the field and further than the ceiling inside the
      // source, its surface point along the field's normal, at the cell's spacing. The rounding there may
      // turn too little to seed (a face beside an acute edge is rounded over more than a cell), and the
      // projection onto the field may not reach it.
      for (const p of cellProbes(c)) {
        if (!(phiAt(p[0], p[1], p[2]) >= 0 && truth(p) < -ceiling)) continue;
        const g = sampleClothSolidField(field, p).grad, gl = Math.hypot(g[0], g[1], g[2]);
        const t = gl > 0 ? onTruth(p, [g[0] / gl, g[1] / gl, g[2] / gl]) : undefined;
        if (!t || !(phiAt(t.p[0], t.p[1], t.p[2]) >= -1e-5 * h) || !convexAt(t.p, t.n)) continue;
        // With its neighbours on the source a cell away along two tangents: a strip rounded past the
        // ceiling narrower than a cell holds one row of samples, which spans no facet over it.
        const n = t.n, a = Math.abs(n[0]) < 0.6 ? [1, 0, 0] as V3 : Math.abs(n[1]) < 0.6 ? [0, 1, 0] as V3 : [0, 0, 1] as V3;
        const t1: V3 = [n[1] * a[2] - n[2] * a[1], n[2] * a[0] - n[0] * a[2], n[0] * a[1] - n[1] * a[0]], l1 = Math.hypot(...t1);
        const u: V3 = [t1[0] / l1, t1[1] / l1, t1[2] / l1], w: V3 = [n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0]];
        for (const y of [t, ...[u, w].flatMap(d => [1, -1].map(sg => onTruth([0, 1, 2].map(i => t.p[i]! + sg * h * d[i]! + h * n[i]!) as unknown as V3, n)))]) {
          if (!y || !(phiAt(y.p[0], y.p[1], y.p[2]) >= -1e-5 * h)) continue;
          found.push({ q: { p: y.p, n: y.n, mu: sampleClothSolidField(field, y.p).mu }, k: 0, spacing: h, planes: [{ p: y.p, n: y.n }], truth: true });
        }
      }
      const centre = project(c as unknown as V3);
      if (!centre) continue;
      const kc = kappa(centre, CELL_STEP);
      if (kc <= kappaMin) continue;
      const m = kc * h < SOLID_SAMPLE_SHARP_TURN ? 1 : 2;
      if (m === 1) { for (const f of placed(centre, () => supportPlanes(centre))) found.push({ q: f.q, k: kc, spacing: h, planes: f.planes, truth: f.truth }); continue; }
      // A sharp cell is seeded at its corners too: a voxel solid's corner is a node, which no
      // projection from inside the cell reaches, and the samples' hull would cut it off.
      const seeds = [...Array(m * m * m).keys()].map(s => [s % m, Math.floor(s / m) % m, Math.floor(s / (m * m))].map(i => ((i + 0.5) / m - 0.5) * h))
        .concat([...Array(8).keys()].map(k => [k & 1, (k >> 1) & 1, k >> 2].map(b => (b - 0.5) * h)));
      for (const o of seeds) {
        const q = project([c[0]! + o[0]!, c[1]! + o[1]!, c[2]! + o[2]!]);
        if (!q) continue;
        const kq = kappa(q, localStep);
        if (kq > kappaMin) for (const f of placed(q, () => supportPlanes(q))) found.push({ q: f.q, k: kq, spacing: h / m, planes: f.planes, truth: f.truth });
      }
    }
  }
  // Corners (most cells' planes meeting) first, then edges, then the sharpest curvature: a nearby
  // sample on a face never displaces the one on the feature.
  found.sort((a, b) => b.planes.length - a.planes.length || b.k - a.k);
  for (const f of found) keep(f.q, f.planes, f.spacing, f.truth === true);

  // ---- the lifts: each sample's fan of holes, measured; a hole bulging past the ceiling is
  // refined (its centre and edge midpoints sampled at half the spacing) and measured again.
  const HN = 0.5 * h, coarse = new Map<number, number[]>();
  const index = (from: number) => {
    for (let i = from; i < pts.length; i++) {
      const p = pts[i]!.p, k = brickKey(cellAt(p[0], HN), cellAt(p[1], HN), cellAt(p[2], HN)), l = coarse.get(k);
      if (l) l.push(i); else coarse.set(k, [i]);
    }
  };
  const within = (p: V3, rad: number): number[] => {
    const out: number[] = [];
    const x0 = Math.floor((p[0] - rad) / HN), y0 = Math.floor((p[1] - rad) / HN), z0 = Math.floor((p[2] - rad) / HN);
    const x1 = Math.floor((p[0] + rad) / HN), y1 = Math.floor((p[1] + rad) / HN), z1 = Math.floor((p[2] + rad) / HN);
    for (let z = z0; z <= z1; z++) for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      const l = coarse.get(brickKey(x, y, z));
      if (l) for (const j of l) {
        const o = pts[j]!.p;
        if (Math.hypot(o[0] - p[0], o[1] - p[1], o[2] - p[2]) <= rad) out.push(j);
      }
    }
    return out;
  };
  const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  // The lift of every sample, and the holes bulging past the ceiling.
  const measure = (): { lift: Float64Array; over: number[][]; apexes: { p: V3; planes: { p: V3; n: V3 }[]; spacing: number }[]; edges: number[]; facets: number[] } => {
    const lift = new Float64Array(pts.length), over: number[][] = [], done = new Set<number>(), edgeKeys = new Set<number>(), facets: number[] = [];
    const apexes: { p: V3; planes: { p: V3; n: V3 }[]; spacing: number }[] = [];
    // The hull edges are the samples' Gabriel graph (a subgraph of their Delaunay graph: it
    // joins consecutive samples along a sharp edge's row) and the edges of the local hull's
    // facets. Any other chord between neighbours lies under the lifted facets, where no contact
    // outside the field reaches it; all neighbours within the nominal reach (about three actual
    // spacings) made 6-11 hull edges per sample on a sphere, against about 3.
    const edge = (a: number, b: number) => { edgeKeys.add(a < b ? a * pts.length + b : b * pts.length + a); };
    // Triangle keys i S^2 + j S + k exact in a double.
    const S3 = pts.length, maxSpacing = Math.max(...spacingOf);
    if (S3 ** 3 > Number.MAX_SAFE_INTEGER) throw new RangeError(`Cloth solid samples: ${S3} samples exceed the hole key's range`);
    for (let i = 0; i < pts.length; i++) {
      // A pair's reach is the coarser spacing of the two: a sample refined to h / 2 beside a
      // field sample at h must see it, as the field sample sees the refined one (a reach of
      // i's own spacing alone left a gap between the patches the hull never closed).
      const qi = pts[i]!, n = qi.n, reach = 1.6 * maxSpacing;
      const near = (j: number) => Math.hypot(...sub(pts[j]!.p, qi.p)) <= 1.6 * Math.max(spacingOf[i]!, spacingOf[j]!);
      const a = Math.abs(n[0]) < 0.6 ? [1, 0, 0] as V3 : Math.abs(n[1]) < 0.6 ? [0, 1, 0] as V3 : [0, 0, 1] as V3;
      const t1 = cross(n, a), l1 = Math.hypot(...t1), u: V3 = [t1[0] / l1, t1[1] / l1, t1[2] / l1], w = cross(n, u);
      // A concave junction belongs to the field, not a convex hull spanning both
      // surfaces. Its opposite face must not hide this face's supporting facets.
      // Convex surface normals are monotone: (pj - pi) . (nj - ni) >= 0.
      // Allow rounding at coplanar samples, but exclude a turn into the solid.
      const convexNeighbour = (j: number) => dot(sub(pts[j]!.p, qi.p), sub(pts[j]!.n, n)) >= -1e-6 * h;
      const fan = within(qi.p, reach).filter(j => j !== i && near(j) && convexNeighbour(j)).map(j => { const d = sub(pts[j]!.p, qi.p); return { j, ang: Math.atan2(dot(d, w), dot(d, u)) }; })
        .sort((x, y) => x.ang - y.ang);
      // Gabriel edges: no other sample inside the sphere on the edge as diameter (any such
      // sample lies within the edge's length of i, so in its fan).
      for (const F of fan) {
        if (F.j < i) continue;
        const pj = pts[F.j]!.p, mx = 0.5 * (qi.p[0] + pj[0]), my = 0.5 * (qi.p[1] + pj[1]), mz = 0.5 * (qi.p[2] + pj[2]);
        const r2 = 0.25 * ((qi.p[0] - pj[0]) ** 2 + (qi.p[1] - pj[1]) ** 2 + (qi.p[2] - pj[2]) ** 2);
        if (!fan.some(M => M.j !== F.j && (pts[M.j]!.p[0] - mx) ** 2 + (pts[M.j]!.p[1] - my) ** 2 + (pts[M.j]!.p[2] - mz) ** 2 < r2)) edge(i, F.j);
      }
      if (fan.length < 2) continue;
      // Every wedge of the fan short of half a turn, not only consecutive neighbours: the local
      // convex hull's facet over a wedge can span neighbours that stand under it (a lattice
      // bevel's corner: the facet over the Bridson tablecloth's leg tops skips a sample, and
      // without it a face resting on the lifted samples entered the field by 0.15 mm).
      for (let k = 0; k < fan.length; k++) for (let l = 1; l < fan.length; l++) {
        const A = fan[k]!, B = fan[(k + l) % fan.length]!;
        let gap = B.ang - A.ang; if (gap <= 0) gap += 2 * Math.PI;
        // A gap of half a turn or more faces the edge of the sampled surface: no hole there.
        if (gap >= Math.PI) break;
        if (fan.length < 3 && k > 0) continue;
        const lo = Math.min(i, A.j, B.j), hi = Math.max(i, A.j, B.j), key = (lo * S3 + (i + A.j + B.j - lo - hi)) * S3 + hi;
        if (done.has(key)) continue;
        done.add(key);
        // So does a wide gap whose middle holds no surface the seeding would sample: a concave
        // crease (no normal) or flat surface just past the last row. (On a curved boundary row
        // the gap falls a little short of half a turn.)
        if (l === 1 && gap > 0.5 * Math.PI) {
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
        // Nor over a sample beyond i's reach that stands near the triangle (a corner sample nearer the
        // triangle's other samples: their fans see it, and the hole is measured once, from the first).
        const mid = [0, 1, 2].map(a => (qi.p[a]! + pa[a]! + pb[a]!) / 3) as unknown as V3;
        const span = Math.max(Math.hypot(...sub(pa, qi.p)), Math.hypot(...sub(pb, qi.p)), Math.hypot(...sub(pa, pb)));
        if (within(mid, span).some(j => j !== i && j !== A.j && j !== B.j && convexNeighbour(j) && dot(sub(pts[j]!.p, qi.p), N) > tol)) continue;
        // A wide wedge is a facet only over neighbours strictly under it: over one on its plane
        // the narrower wedges are the same facet's triangulation.
        if (l > 1) { let on = false; for (let m = 1; m < l; m++) if (dot(sub(pts[fan[(k + m) % fan.length]!.j]!.p, qi.p), N) >= -tol) on = true; if (on) continue; }
        // Measured along the samples' mean normal, the direction they are lifted in (a sliver
        // standing across the surface has a facet normal along it, which measures nothing).
        const D = [0, 1, 2].map(a => n[a]! + pts[A.j]!.n[a]! + pts[B.j]!.n[a]!) as unknown as V3, dl = Math.hypot(...D);
        // Or along the facet's own normal, where that is less: a facet tilted across the samples' mean normal
        // (between a sharp edge's row and the face past it) has the surface rise above it by its distance along
        // N, which the mean normal, nearly in its plane, overstates without bound.
        const planes3 = [...supports[i]!, ...supports[A.j]!, ...supports[B.j]!];
        const rD = roofHeight([qi.p, pa, pb], [D[0] / dl, D[1] / dl, D[2] / dl], planes3), rN = roofHeight([qi.p, pa, pb], N, planes3);
        const [roof, DN] = rN.height < rD.height ? [rN, N] : [rD, [D[0] / dl, D[1] / dl, D[2] / dl] as V3], bulge = roof.height;
        for (const j of [i, A.j, B.j]) lift[j] = Math.max(lift[j]!, bulge);
        if (bulge > ceiling) {
          over.push([i, A.j, B.j]);
          if (Number.isFinite(bulge)) apexes.push({ p: [0, 1, 2].map(a => (1 - roof.u - roof.v) * qi.p[a]! + roof.u * pa[a]! + roof.v * pb[a]! + bulge * DN[a]!) as unknown as V3, planes: [...supports[i]!, ...supports[A.j]!, ...supports[B.j]!], spacing: Math.min(spacingOf[i]!, spacingOf[A.j]!, spacingOf[B.j]!) });
        }
        edge(i, A.j); edge(A.j, B.j); edge(B.j, i);
        // A candidate facet, wound outward ((pa - pi) x (pb - pi) along N); exported below.
        if (sgn > 0) facets.push(i, A.j, B.j); else facets.push(i, B.j, A.j);
      }
    }
    const edges: number[] = [];
    for (const key of edgeKeys) edges.push(Math.floor(key / S3), key % S3);
    return { lift, over, apexes, edges, facets };
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
        if (q) for (const f of placed(q, () => supportPlanes(q))) keep(f.q, f.planes, spacing, f.truth === true);
      }
    }
    // A hole's roof apex that stands on the source's surface outside the field is the source's own feature
    // there (a corner where the field rounds it past the ceiling, which no projection onto the field
    // reaches): it is sampled where it stands, with the support planes through it.
    for (const A of measured.apexes) {
      if (!(Math.abs(truth(A.p)) <= 1e-6 * h && phiAt(A.p[0], A.p[1], A.p[2]) >= 0)) continue;
      const through = A.planes.filter(P => Math.abs(dot(sub(A.p, P.p), P.n)) <= 1e-6 * h).map(P => ({ p: A.p, n: P.n }));
      if (!through.length) continue;
      const m = [0, 1, 2].map(a => through.reduce((s, P) => s + P.n[a]!, 0)), ml = Math.hypot(m[0]!, m[1]!, m[2]!);
      if (!(ml > 0)) continue;
      // Kept unless a sample stands within the ceiling of it already (then the hull holds it to the bound):
      // the samples' spacing gives way to the feature the hole needs.
      keep({ p: A.p, n: [m[0]! / ml, m[1]! / ml, m[2]! / ml], mu: sampleClothSolidField(field, A.p).mu }, through, 0.5 * A.spacing, true, ceiling);
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
  // The facets exported, on the samples as stored (lifted, f32), where:
  // - one of its samples stands where the field rounds the source past the ceiling: on the source's
  //   surface placed for it, or (on a feature of the source) further than the ceiling outside the field.
  //   The facets represent the solid where the field does not. Elsewhere the field holds the surface
  //   within the ceiling (h / 16, the samples' own lift bound) and its entry is the solid's;
  // - the stored samples resolve the facet's plane: through three points each within delta (an ulp of
  //   the largest coordinate) of its own it tilts by up to about delta / a, a the triangle's least
  //   altitude, which moves it delta L / a over the facet (L its longest edge); kept where that is
  //   within the ceiling;
  // - and that plane supports the solid at its samples: its outward normal lies in the normal cone the
  //   three samples' support planes span, to the turn floor (a convex surface's chord over a hole
  //   tilts within the cone of its ends' normals; a face's facet has that face's normal).
  // The rest are a row's slivers: three nearly collinear samples along a sharp edge, which pass the hull
  // test with a plane the lifts tip anywhere, often across the surface, where a contact would stop
  // cloth sliding round the edge. Their edges are hull edges, and they span no area.
  const facets: number[] = [];
  const at = (j: number): V3 => [out[SOLID_SAMPLE_WORDS * j]!, out[SOLID_SAMPLE_WORDS * j + 1]!, out[SOLID_SAMPLE_WORDS * j + 2]!];
  for (let k = 0; k < measured.facets.length; k += 3) {
    const ids = [measured.facets[k]!, measured.facets[k + 1]!, measured.facets[k + 2]!], [a, b, c] = ids.map(at) as [V3, V3, V3];
    if (!ids.some(j => standsOnTruth[j] || phiAt(pts[j]!.p[0], pts[j]!.p[1], pts[j]!.p[2]) > ceiling)) continue;
    const L = Math.max(Math.hypot(...sub(a, b)), Math.hypot(...sub(b, c)), Math.hypot(...sub(c, a)));
    const delta = 2 ** -23 * Math.max(...a.map(Math.abs), ...b.map(Math.abs), ...c.map(Math.abs));
    const n = cross(sub(b, a), sub(c, a)), nl = Math.hypot(...n);
    if (!(L > 0 && nl / L >= delta * L / ceiling)) continue;
    if (!inNormalCone([n[0] / nl, n[1] / nl, n[2] / nl], ids.flatMap(j => supports[j]!.map(q => q.n)), SOLID_SAMPLE_TURN_FLOOR)) continue;
    facets.push(...ids);
  }
  // ---- the authoring check: wherever the field rounds the source past the ceiling (a probe at h / 4
  // over every rounded surface cell, outside the field and further than the ceiling inside the
  // source), the probe lies behind its nearest facet (inside that facet's outward plane): a cloth
  // vertex there is kept out by the facet, as the field keeps it out elsewhere. Else the build fails.
  const FB = 2 * h, facetBuckets = new Map<number, number[]>();
  let concaveDepth = 0;
  for (let k = 0; k < facets.length; k += 3) {
    const [a, b, c] = [facets[k]!, facets[k + 1]!, facets[k + 2]!].map(at) as [V3, V3, V3];
    const lo = [0, 1, 2].map(i => cellAt(Math.min(a[i]!, b[i]!, c[i]!), FB)), hi = [0, 1, 2].map(i => cellAt(Math.max(a[i]!, b[i]!, c[i]!), FB));
    for (let z = lo[2]!; z <= hi[2]!; z++) for (let y = lo[1]!; y <= hi[1]!; y++) for (let x = lo[0]!; x <= hi[0]!; x++) {
      const key = brickKey(x, y, z), l = facetBuckets.get(key);
      if (l) l.push(k); else facetBuckets.set(key, [k]);
    }
  }
  for (const c of surfaceCells) for (const p of cellProbes(c)) {
    if (!(phiAt(p[0], p[1], p[2]) >= 0)) continue;
    const depth = -truth(p);
    if (!(depth > ceiling)) continue;
    const g = sampleClothSolidField(field, p).grad, gl = Math.hypot(g[0], g[1], g[2]), x = gl > 0 ? onTruth(p, [g[0] / gl, g[1] / gl, g[2] / gl]) : undefined;
    // A concave crease the field rounds out is not the hull's to represent: its worst depth is reported.
    if (x && !convexAt(x.p, x.n)) { concaveDepth = Math.max(concaveDepth, depth); continue; }
    let best = Infinity, behind = false;
    const cx = cellAt(p[0], FB), cy = cellAt(p[1], FB), cz = cellAt(p[2], FB);
    for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) for (const f of facetBuckets.get(brickKey(cx + dx, cy + dy, cz + dz)) ?? []) {
      const [a, b, cc] = [facets[f]!, facets[f + 1]!, facets[f + 2]!].map(at) as [V3, V3, V3];
      const d = pointTriangleDistance(p, a, b, cc);
      if (d < best) { best = d; behind = dot(sub(p, a), cross(sub(b, a), sub(cc, a))) < 0; }
    }
    if (!behind) throw new RangeError(`Cloth solid samples: the field rounds the solid ${depth.toExponential(2)} m (above h / 16 = ${ceiling.toExponential(2)} m) near (${p.map(v => v.toFixed(4)).join(", ")}), and ${best < Infinity ? `it lies in front of its nearest hull facet (${best.toExponential(2)} m)` : "no hull facet lies within two cells"}`);
  }
  return { samples: out, edges: Uint32Array.from(measured.edges), facets: Uint32Array.from(facets), concaveDepth };
}
