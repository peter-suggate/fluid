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
  // Per cell and axis, the largest |difference| along the cell's four a-edges (cellGradientRange).
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
  const values = new Float32Array(NODES).fill(Infinity), own = new Uint8Array(NODES);
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
      if (out[i]! < values[i]!) { values[i] = out[i]!; own[i] = k; }
      if (k === near.length - 1) lo = Math.min(lo, values[i]!);
    }
  }
  if (lo > band) return undefined;
  // Each part's own surface: the union's interpolant lies at or below every part's, so
  // what a part keeps the union keeps.
  near.forEach((part, k) => { if (part.kind === "sdf") checkResolved(part.sdf, cell, at, pv[k]!); });
  const mu = new Uint8Array(NODES);
  for (let i = 0; i < NODES; i++) mu[i] = Math.round(near[own[i]!]!.mu * SOLID_MU_SCALE);
  return { brick: b, values, mu, lipschitz: brickLipschitz(values, cell) };
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
 * Normal turn per cell above which the surface is sharp (radius of curvature
 * under 4 h: an edge or corner the trilinear field rounds over a cell), and is
 * sampled at h / 2 rather than h.
 */
export const SOLID_SAMPLE_SHARP_TURN = 0.25;

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
 * A cell whose centre's normal turns (convexly) by more than the floor over
 * two cells is sampled:
 *   - smooth (a turn under SOLID_SAMPLE_SHARP_TURN per cell): its centre's
 *     projection, spacing s about h;
 *   - sharp: seeded at its eight half-cell centres, each projected onto the
 *     surface and kept where the turn over half a cell still exceeds the
 *     floor, spacing s about h / 2.
 * Kept samples are at least h / 4 apart. A smooth sample stands
 * (kappa_1 + kappa_2) s^2 / 8 out along its normal: the surface between
 * samples s apart rises at most that far above their plane (a square of side
 * s: its centre is (s / 2, s / 2) from a corner), so a face kept off the
 * samples is kept off the surface. A crease's samples stay on it: the
 * trilinear field of a lattice box keeps its edge sharp, and the surface is
 * straight along the edge and flat beside it.
 *
 * A straight cloth edge lying across a crease passes within about h / 4 of a
 * sample (h / 2 on a smooth surface), so the barrier engages it only where
 * that is under r: an edge may graze the solid between samples when h > 4 r
 * (the Bridson scenes: h = 1 cm, r = 1.5 mm). Setup-time CPU work over the
 * surface cells whose field is not affine; returns (x, y, z, mu) per sample,
 * with phi there equal to its lift.
 *
 * `options` (the cloth's r and longest edge) are validated and otherwise not
 * used: the samples depend on the field alone.
 */
export function clothSolidSamples(field: ClothSolidField, options?: ClothSolidSampleOptions): Float32Array {
  if (options && !(options.r > 0 && options.edge > 0)) throw new RangeError("Cloth solid samples need r > 0 and edge > 0");
  let cached = sampleCache.get(field);
  if (!cached) sampleCache.set(field, cached = buildSolidSamples(field));
  return cached;
}

/** A field is immutable once built (a rebuild returns a new one): its samples are computed once. */
const sampleCache = new WeakMap<ClothSolidField, Float32Array>();

function buildSolidSamples(field: ClothSolidField): Float32Array {
  const h = field.cell, kappaMin = SOLID_SAMPLE_TURN_FLOOR / h, CELL_STEP = 2;
  const spacing = 0.5 * h, m = 2, localStep = 0.5;
  const sample = (p: V3) => sampleClothSolidField(field, p);
  // Newton onto phi = 0 along the gradient; undefined when it leaves the stored field or drifts more than a cell.
  const project = (p0: V3): { p: V3; n: V3; mu: number } | undefined => {
    let p: V3 = p0;
    for (let i = 0; i < 10; i++) {
      const s = sample(p);
      if (!s.stored) return undefined;
      const g2 = s.grad[0] ** 2 + s.grad[1] ** 2 + s.grad[2] ** 2;
      if (!(g2 > 1e-12)) return undefined;
      if (Math.abs(s.phi) < 1e-5 * h) {
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
      p = [p[0] - s.phi * s.grad[0] / g2, p[1] - s.phi * s.grad[1] / g2, p[2] - s.phi * s.grad[2] / g2];
    }
    return undefined;
  };
  // The convex normal turn per length seen `step` cells away along two orthogonal
  // tangents (both ways): [the larger, their sum (at least kappa_1 + kappa_2)].
  const turns = (q: { p: V3; n: V3 }, step: number): [number, number] => {
    const n = q.n, a = Math.abs(n[0]) < 0.6 ? [1, 0, 0] : Math.abs(n[1]) < 0.6 ? [0, 1, 0] : [0, 0, 1];
    const t1 = [n[1] * a[2]! - n[2] * a[1]!, n[2] * a[0]! - n[0] * a[2]!, n[0] * a[1]! - n[1] * a[0]!];
    const l1 = Math.hypot(...t1), u = t1.map(v => v / l1), w = [n[1] * u[2]! - n[2] * u[1]!, n[2] * u[0]! - n[0] * u[2]!, n[0] * u[1]! - n[1] * u[0]!];
    const k = [0, 0];
    for (const [i, d] of [u, w].entries()) for (const sgn of [1, -1]) {
      const o = project([q.p[0] + sgn * step * h * d[0]!, q.p[1] + sgn * step * h * d[1]!, q.p[2] + sgn * step * h * d[2]!]);
      if (!o) continue;
      const dp = [o.p[0] - q.p[0], o.p[1] - q.p[1], o.p[2] - q.p[2]], dn = [o.n[0] - n[0], o.n[1] - n[1], o.n[2] - n[2]];
      const len = Math.hypot(...dp);
      if (len < 0.125 * step * h || dp[0]! * dn[0]! + dp[1]! * dn[1]! + dp[2]! * dn[2]! <= 0) continue;
      k[i] = Math.max(k[i]!, Math.hypot(...dn) / len);
    }
    return [Math.max(k[0]!, k[1]!), k[0]! + k[1]!];
  };
  const kappa = (q: { p: V3; n: V3 }, step: number) => turns(q, step)[0];
  // Kept samples at least h / 4 apart (a hash of h / 4 cells, one each): the seeds of
  // neighbouring cells project onto the same stretch of surface.
  const out: number[] = [], seen = new Map<string, number>(), minSep = 0.5 * spacing;
  // A sample sits `lift` out along its normal: the surface it stands for bulges at most that far above the samples' plane.
  const keep = (q: { p: V3; n: V3; mu: number }, lift: number) => {
    const c = q.p.map(v => Math.floor(v / minSep));
    for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const j = seen.get(`${c[0]! + dx},${c[1]! + dy},${c[2]! + dz}`);
      if (j !== undefined && Math.hypot(out[j]! - q.p[0], out[j + 1]! - q.p[1], out[j + 2]! - q.p[2]) < minSep) return;
    }
    seen.set(c.join(","), out.length); out.push(q.p[0] + lift * q.n[0], q.p[1] + lift * q.n[1], q.p[2] + lift * q.n[2], q.mu);
  };
  const cellNodes = new Float64Array(8);
  for (const brick of field.bricks.values()) {
    const v = brick.values;
    for (let z = 0; z < SOLID_BRICK_CELLS; z++) for (let y = 0; y < SOLID_BRICK_CELLS; y++) for (let x = 0; x < SOLID_BRICK_CELLS; x++) {
      let lo = Infinity, hi = -Infinity;
      for (let k = 0; k < 8; k++) {
        const s = v[nodeIndex(x + (k & 1), y + ((k >> 1) & 1), z + (k >> 2))]!;
        cellNodes[k] = s; lo = Math.min(lo, s); hi = Math.max(hi, s);
      }
      if (lo > 0 || hi < 0) continue;
      const g = cellGradientRange(v, [x, y, z]);
      if (Math.max(g.hi[0] - g.lo[0], g.hi[1] - g.lo[1], g.hi[2] - g.lo[2]) <= 1e-6 * h) continue;
      const c = [0, 1, 2].map(a => field.origin[a]! + (brick.brick[a]! * SOLID_BRICK_CELLS + [x, y, z][a]! + 0.5) * h);
      const centre = project(c as unknown as V3);
      if (!centre) continue;
      const [k0, kSum] = turns(centre, CELL_STEP);
      if (k0 <= kappaMin) continue;
      // A smooth cell (the normal turns less than SOLID_SAMPLE_SHARP_TURN per cell) keeps its centre's sample,
      // lifted by its bulge (kappa_1 + kappa_2) s^2 / 8 at s = h.
      if (k0 * h < SOLID_SAMPLE_SHARP_TURN) { keep(centre, 0.125 * kSum * h * h); continue; }
      for (let s = 0; s < m * m * m; s++) {
        const o = [s % m, Math.floor(s / m) % m, Math.floor(s / (m * m))].map(i => ((i + 0.5) / m - 0.5) * h);
        const q = project([c[0]! + o[0]!, c[1]! + o[1]!, c[2]! + o[2]!]);
        if (!q) continue;
        // On a crease the surface is straight along it and flat across it off the crease: no bulge.
        if (kappa(q, localStep) > kappaMin) keep(q, 0);
      }
    }
  }
  return new Float32Array(out);
}
