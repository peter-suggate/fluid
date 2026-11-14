/**
 * Seep solver: shared WGSL (constants, brick-major addressing, the baked solid
 * geometry and the field samplers). Lane S, docs/plans/seep-rewrite-2026-11-14.md.
 */
import { BRICKS, BULK_LISTS, CELLS, DT, CELL_M, NBULK, PHI_MAX, PARTICLE_CELLS, PBX, PBX0, PARTICLE_X0, THROAT_Y0, THROAT_Z0 } from "./seep-layout";

/** Particle pool capacity (each pool), from the measured NB-FLIP peak (216k) with ample margin. */
export const PARTICLE_CAPACITY = 1 << 20;
/** FLIP fraction (Ferstl 2016 uses a FLIP/PIC blend; the reference runs 0.95). */
export const FLIP_RATIO = 0.95;
/** Narrow band width R (cells) of the reference (Ferstl 2016 §3.3); the chamber keeps every particle (they carry its liquid). */
export const BAND = 3;
/** Eq. (3) combination band r = R - 1 (Ferstl 2016 §3.1). */
export const COMBINE_BAND = 2;
/** Zhu-Bridson particle radius (cells), calibrated so a stratified slab reproduces its own volume. */
export const PARTICLE_RADIUS = 0.265;
/** Ghost-fluid theta clamp (as the reference). */
export const THETA_MIN = 1e-3;
/** Pressure tolerance: L-inf divergence 0.05 s^-1, in flux units (m/s): 0.05 * h. */
export const PRESSURE_TOLERANCE = 0.05 * CELL_M;
/** Multigrid cycle cap per step. With the staggered L0 partition the largest count over the 1200-step seep is 7
 *  (3.33 a step on average; it was 12 with bricks alone), so 9 leaves two cycles of margin and no step is capped
 *  (capHits counts any). Every cycle slot costs its gated dispatches even when empty. */
export const MAX_CYCLES = 9;
/** Aggregation over-correction: A_c = R A P / OC (Braess 1995; piecewise-constant aggregation). A
 *  preconditioner parameter only (the converged solution is the same). Measured on the seep at
 *  1200 steps: OC 1.8 takes 7.6 cycles a step and OC 1.6 takes 6.7. The chamber's free-surface rows
 *  are what converge last, and they favour the weaker scaling. 1.4 and 2.1 are both slower. */
export const AGGREGATION_OC = 1.6;
/** Backtrace segment cap: 48 cells a step (14 m/s) at 1-cell segments; beyond it segments lengthen (recorded in maxSegments, not fatal). */
export const MAX_SEGMENTS = 48;

/** ctrl word offsets. */
export const C = {
  ARG_ACTIVE: 0, ARG_SURF: 4, ARG_PART: 8, ARG_CLEAR: 12,
  /** In the args buffer: the active list plus BULK_WG leading workgroups, and plus ceil(NBULK / 8) leading (eight BULK bricks each). */
  ARG_ACTB: 16, ARG_ACTBB: 20,
  /** In the args buffer: the liquid bricks (built by the operator), plus BULK_WG leading workgroups, and plus NBULK leading. */
  ARG_LIQ: 24, ARG_LIQB: 28, ARG_LIQBB: 32,
  /** In the args buffer: the level-4 blocks (2x2x2 level-3 cells) holding a level-3 unknown, in groups of
   *  eight (ARG_L4), and one by one (ARG_L4U). */
  ARG_L4: 36, ARG_L4U: 40,
  /** In the args buffer: the shifted fine blocks (the second L0 smoothing partition, see B.LIST_SHIFT). */
  ARG_SHIFT: 44,
  STEP: 16, FAIL: 17, PCOUNT: 18 /* 18,19 */, DELETED: 20, SEEDED: 21,
  RES: 22, CONV: 23, PARITY: 24, CYCLES: 25, FIRST_RES: 26, LAST_RES: 27, CAP_HITS: 28, CYCLES_TOTAL: 29,
  MAX_SEG: 30, OVERFLOW: 31,
  /** List sizes of the last completed step (copied by g2p before it resets the arguments). */
  N_ACTIVE: 32, N_SURF: 33, N_PART: 34, N_CLEAR: 35,
  /** The last step's residual (f32 bits) measured before each cycle k = 0..MAX_CYCLES (0 once converged). */
  RES_HIST: 36,
  /** PCG scalars alpha_(k-1), rho_(k-1) (f32 bits), and the dot-product slots (2 sets x [r.z, z.q, p.q] x 64). */
  ALPHA: 60, RHO: 61,
  /** Cumulative particle retirements by reason: deeper than the band, over the cell cap, out of the particle region. */
  DEL_DEEP: 62, DEL_CAP: 63, DEL_OUT: 64,
  /** Liquid bricks of the current step (published by coarsenTop), and the operator's append counter. */
  N_LIQ: 65, LIQ_COUNT: 66,
  /** Level-4 block list: its append counter, and its size this step. */
  L4_COUNT: 67, N_L4: 68,
  /** Outlet emission: the swept length (f32 cells) not yet emitted, per throat face (4); particles emitted (cumulative). */
  EMIT_ACC: 69, EMITTED: 73,
  /** Shifted fine blocks: the operator's append counter, and their number this step. */
  SHIFT_COUNT: 74, N_SHIFT: 75,
  RED: 256,
  /** In the args buffer: gated arguments per PCG iteration k, records of 4 words: U2, U1, U0, P, X(k+1); slot 5 the shifted fine blocks. */
  MG_ARGS: 64,
  WORDS: 256 + 2 * 192,
} as const;
/** Words per PCG iteration in the args buffer, and the args buffer's size in words. */
export const MG_STRIDE = 32;
export const ARGS_WORDS = C.MG_ARGS + MG_STRIDE * (MAX_CYCLES + 1);
export const FAIL = { DEEP_SURFACE: 1, POOL: 2, SEGMENTS: 4, COARSEST_PIVOT: 8 } as const;

/** bricks table offsets (each BRICKS long). */
export const B = {
  CLASS: 0, FLAGS: 1, MINPHI: 2,
  /** Nonzero when G2P or emission left a particle in the brick (read only as nonzero; G2P sets 1, emission counts). */
  PCOUNT: 3,
  /** A second copy of the BULK list immediately followed by the liquid bricks (active bricks with a pressure unknown), for the pressure solve. */
  LIST_BULK2: 4, LIST_SURF: 5, LIST_PART: 6, LIST_CLEAR: 7, PSTART: 8, PNUM: 9,
  /** The BULK bricks (static, NBULK) immediately followed by the active list, so that one index serves
   *  dispatches over [BULK, active] (B_LBULK + i) and over the active list alone (B_LACT = B_LBULK + NBULK).
   *  The active list holds no BULK brick, so the region (BRICKS words) has room. Then the static shadow list. */
  LIST_BULK: 10, LIST_SHADOW: 11,
  /** Level-4 block stamps (the step that listed it; 2048 words used) and the level-4 block list. */
  L4_FLAG: 12, L4_LIST: 13,
  /** Bit a set when G2P or emission left a particle in the brick moving more than FAST_CELLS a step along axis a (reset with PCOUNT). */
  FAST: 14,
  /** Shifted fine blocks (the second L0 smoothing partition: 8^3 cells at origin 8 s - (4, 0, 4), s in 65 x 32 x 9): the step
   *  that listed each (two regions, index s.x + 65 (s.y + 33 s.z)), and their list (two regions, s packed x | y << 8 | z << 16). */
  SHIFT_FLAG: 15, LIST_SHIFT: 17,
  COUNT: 19,
} as const;
/** BULK work: 64 bricks (512 coarse cells) per 512-thread workgroup; these workgroups come first in the dispatches that include BULK. */
export const BULK_WG = Math.ceil(NBULK / 64);
export const NSHADOW = BULK_LISTS.shadow.length;
/** The dense 4h lattice (128 x 64 x 16) of the BULK buffer: U (3 components) then Us (3). */
export const COARSE_LATTICE = 128 * 64 * 16;

/** Coarse multigrid levels 1..6 (L0 is the fine brick-major lattice). */
export const LEVELS = 6;
export function levelDims(l: number): [number, number, number] { return [512 >> l, 256 >> l, 64 >> l]; }
export const LEVEL_OFFSET: number[] = (() => {
  const off = [0, 0];
  for (let l = 1; l <= LEVELS; l++) { const [x, y, z] = levelDims(l); off.push(off[l] + x * y * z); }
  return off;
})();
export const COARSE_CELLS = LEVEL_OFFSET[LEVELS + 1];
/** Cells of the coarsest level (8 x 4 x 1), solved exactly through its explicit inverse; its 32 x 32 entries are one per thread of a 1024-wide workgroup. */
export const COARSEST_CELLS = levelDims(LEVELS).reduce((a, b) => a * b, 1);
if (COARSEST_CELLS * COARSEST_CELLS !== 1024) throw new Error(`seep: coarsest level of ${COARSEST_CELLS} cells, the inverse needs 32`);

const f = (x: number) => { const s = String(x); return s.includes(".") || s.includes("e") ? s : s + ".0"; };

export const COMMON_WGSL = /* wgsl */ `
const NX: i32 = 512; const NY: i32 = 256; const NZ: i32 = 64;
const NV: vec3i = vec3i(512, 256, 64);
const N: u32 = ${CELLS}u;
const BRICKS: u32 = ${BRICKS}u;
const CFL: f32 = ${f(DT / CELL_M)};
/** Particle speed (cells a step) along an axis beyond which a chamber brick is marked fast along it (bricks FAST bit a): reach two. */
const FAST_CELLS: f32 = 6.0;
fn fastBits(v: vec3f) -> u32 { let f = abs(v) * CFL > vec3f(FAST_CELLS); return select(0u, 1u, f.x) | select(0u, 2u, f.y) | select(0u, 4u, f.z); }
const GDT: f32 = ${f(9.80665 * DT)};
const PHI_MAX: f32 = ${f(PHI_MAX)};
const UNKNOWN_PHI: f32 = ${f(PHI_MAX + 1)};
const PX0: i32 = ${PARTICLE_X0};
const THROAT_Y0: i32 = ${THROAT_Y0}; const THROAT_Z0: i32 = ${THROAT_Z0};
const PBX0: u32 = ${PBX0}u;
const PN: u32 = ${PARTICLE_CELLS}u;
const CAP: u32 = ${PARTICLE_CAPACITY}u;
const BAND: f32 = ${f(BAND)};
const RCOMB: f32 = ${f(COMBINE_BAND)};
const RP: f32 = ${f(PARTICLE_RADIUS)};
const THETA_MIN: f32 = ${f(THETA_MIN)};
const FLIP: f32 = ${f(FLIP_RATIO)};
const MAXSEG: f32 = ${f(MAX_SEGMENTS)};
const CLASS_DEAD: u32 = 1u; const CLASS_DEEP: u32 = 2u; const CLASS_PARTICLE: u32 = 4u; const CLASS_CHAMBER: u32 = 8u; const CLASS_BULK: u32 = 16u;
const NBULK: u32 = ${NBULK}u; const BULK_WG: u32 = ${BULK_WG}u; const NSHADOW: u32 = ${NSHADOW}u;
const B_LBULK: u32 = ${B.LIST_BULK * BRICKS}u; const B_LSHADOW: u32 = ${B.LIST_SHADOW * BRICKS}u;
const NCL: u32 = ${COARSE_LATTICE}u; const BU_OFF: u32 = 0u; const BUS_OFF: u32 = ${3 * COARSE_LATTICE}u;
const C_ARG_ACTB: u32 = ${C.ARG_ACTB}u; const C_ARG_ACTBB: u32 = ${C.ARG_ACTBB}u;
const C_N_LIQ: u32 = ${C.N_LIQ}u; const C_LIQ_COUNT: u32 = ${C.LIQ_COUNT}u; const C_L4_COUNT: u32 = ${C.L4_COUNT}u; const C_N_L4: u32 = ${C.N_L4}u;
const C_EMIT_ACC: u32 = ${C.EMIT_ACC}u; const C_EMITTED: u32 = ${C.EMITTED}u;
const C_ARG_L4: u32 = ${C.ARG_L4}u; const C_ARG_L4U: u32 = ${C.ARG_L4U}u; const B_L4FLAG: u32 = ${B.L4_FLAG * BRICKS}u; const B_L4LIST: u32 = ${B.L4_LIST * BRICKS}u; const C_ARG_LIQ: u32 = ${C.ARG_LIQ}u; const C_ARG_LIQB: u32 = ${C.ARG_LIQB}u; const C_ARG_LIQBB: u32 = ${C.ARG_LIQBB}u;
const B_SFLAG: u32 = ${B.SHIFT_FLAG * BRICKS}u; const B_LSHIFT: u32 = ${B.LIST_SHIFT * BRICKS}u; const C_SHIFT_COUNT: u32 = ${C.SHIFT_COUNT}u; const C_N_SHIFT: u32 = ${C.N_SHIFT}u; const C_ARG_SHIFT: u32 = ${C.ARG_SHIFT}u;
const B_LBULK2: u32 = ${B.LIST_BULK2 * BRICKS}u; const B_LLIQ: u32 = ${B.LIST_BULK2 * BRICKS + NBULK}u;
/** Composite h/4h: a fine face against a coarse cell couples over 2.5 cells (half a fine and half a coarse cell). */
const SUBC: f32 = 0.4;
fn clIndex(k: vec3i) -> u32 { return u32(k.x + 128 * (k.y + 64 * k.z)); }
/** pres/op slot of coarse cell k: its BULK brick's first eight slots. */
fn bulkSlot(k: vec3i) -> u32 { return (brickOfCell(k * 4) << 9u) | u32((k.x & 1) | ((k.y & 1) << 1u) | ((k.z & 1) << 2u)); }
const B_CLASS: u32 = ${B.CLASS * BRICKS}u; const B_FLAGS: u32 = ${B.FLAGS * BRICKS}u; const B_MINPHI: u32 = ${B.MINPHI * BRICKS}u;
const B_PCOUNT: u32 = ${B.PCOUNT * BRICKS}u; const B_FAST: u32 = ${B.FAST * BRICKS}u; const B_LACT: u32 = ${B.LIST_BULK * BRICKS + NBULK}u; const B_LSURF: u32 = ${B.LIST_SURF * BRICKS}u;
const B_LPART: u32 = ${B.LIST_PART * BRICKS}u; const B_LCLEAR: u32 = ${B.LIST_CLEAR * BRICKS}u;
const B_PSTART: u32 = ${B.PSTART * BRICKS}u; const B_PNUM: u32 = ${B.PNUM * BRICKS}u;
const C_ARG_ACTIVE: u32 = ${C.ARG_ACTIVE}u; const C_ARG_SURF: u32 = ${C.ARG_SURF}u; const C_ARG_PART: u32 = ${C.ARG_PART}u; const C_ARG_CLEAR: u32 = ${C.ARG_CLEAR}u;
const C_STEP: u32 = ${C.STEP}u; const C_FAIL: u32 = ${C.FAIL}u; const C_PCOUNT: u32 = ${C.PCOUNT}u; const C_DELETED: u32 = ${C.DELETED}u; const C_DEL_DEEP: u32 = ${C.DEL_DEEP}u; const C_DEL_CAP: u32 = ${C.DEL_CAP}u; const C_DEL_OUT: u32 = ${C.DEL_OUT}u;
const C_SEEDED: u32 = ${C.SEEDED}u; const C_RES: u32 = ${C.RES}u; const C_CONV: u32 = ${C.CONV}u; const C_PARITY: u32 = ${C.PARITY}u;
const C_CYCLES: u32 = ${C.CYCLES}u; const C_FIRST_RES: u32 = ${C.FIRST_RES}u; const C_LAST_RES: u32 = ${C.LAST_RES}u;
const C_CAP_HITS: u32 = ${C.CAP_HITS}u; const C_CYCLES_TOTAL: u32 = ${C.CYCLES_TOTAL}u; const C_MAX_SEG: u32 = ${C.MAX_SEG}u;
const C_OVERFLOW: u32 = ${C.OVERFLOW}u; const C_MG_ARGS: u32 = ${C.MG_ARGS}u; const C_N_ACTIVE: u32 = ${C.N_ACTIVE}u; const C_RES_HIST: u32 = ${C.RES_HIST}u; const C_ALPHA: u32 = ${C.ALPHA}u; const C_RHO: u32 = ${C.RHO}u; const C_RED: u32 = ${C.RED}u;
const FAIL_DEEP: u32 = ${FAIL.DEEP_SURFACE}u; const FAIL_POOL: u32 = ${FAIL.POOL}u; const FAIL_SEG: u32 = ${FAIL.SEGMENTS}u; const FAIL_PIVOT: u32 = ${FAIL.COARSEST_PIVOT}u;
const DEAD_PARTICLE: u32 = 0xffffffffu;

fn brickCoord(b: u32) -> vec3i { return vec3i(i32(b & 63u), i32((b >> 6u) & 31u), i32(b >> 11u)); }
fn brickOfCell(c: vec3i) -> u32 { return u32(c.x >> 3u) | (u32(c.y >> 3u) << 6u) | (u32(c.z >> 3u) << 11u); }
fn cidx(c: vec3i) -> u32 { return (brickOfCell(c) << 9u) | u32((c.x & 7) | ((c.y & 7) << 3u) | ((c.z & 7) << 6u)); }
fn localCoord(l: u32) -> vec3i { return vec3i(i32(l & 7u), i32((l >> 3u) & 7u), i32(l >> 6u)); }
fn inBox(c: vec3i) -> bool { return all(c >= vec3i(0)) && all(c < NV); }
fn clampV(c: vec3i) -> vec3i { return clamp(c, vec3i(0), NV - vec3i(1)); }
fn dimOf(a: u32) -> i32 { return select(select(NZ, NY, a == 1u), NX, a == 0u); }
fn unitAxis(a: u32) -> vec3i { return vec3i(select(0, 1, a == 0u), select(0, 1, a == 1u), select(0, 1, a == 2u)); }
/** Baked solids: the box, and the middle wall x in [240,256) pierced by the 2x2 throat. */
fn cellOpen(c: vec3i) -> bool {
  if (!inBox(c)) { return false; }
  if (c.x >= 240 && c.x < 256) { return c.y >= 64 && c.y < 66 && c.z >= 32 && c.z < 34; }
  return true;
}
/** The lower face of c along a (between c - e_a and c) carries flow when both sides are open. */
fn faceOpen(c: vec3i, a: u32) -> bool { return cellOpen(c) && cellOpen(c - unitAxis(a)); }
/** Closed: a wall face with a fixed zero normal velocity (exactly one side open). */
fn faceClosed(c: vec3i, a: u32) -> bool { return cellOpen(c) != cellOpen(c - unitAxis(a)); }
/** Index in the particle-region arrays (x >= PX0), for a cell with x >= PX0. */
fn pidx(c: vec3i) -> u32 {
  let pb = (u32(c.x >> 3u) - PBX0) + ${PBX}u * (u32(c.y >> 3u) + 32u * u32(c.z >> 3u));
  return (pb << 9u) | u32((c.x & 7) | ((c.y & 7) << 3u) | ((c.z & 7) << 6u));
}
fn hash32(x: u32) -> u32 {
  var h = x * 747796405u + 2891336453u;
  h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
  return (h >> 22u) ^ h;
}
fn hashUnit(a: u32, b: u32, c: u32) -> f32 { return f32(hash32(a ^ hash32(b ^ hash32(c))) >> 8u) / 16777216.0; }
`;

/**
 * Staggered trilinear velocity sampler over a packed f32 buffer `buf`, whose
 * component a starts at `base + a*N`. Positions in cells. Faces beyond the box
 * along their own axis are walls (zero); tangentially the sample clamps
 * (free slip). Inactive bricks hold zero.
 */
export function velocitySampler(name: string, buf: string, base: string): string {
  return /* wgsl */ `
fn ${name}Face(a: u32, c: vec3i) -> f32 {
  if (c[a] < 0 || c[a] >= dimOf(a)) { return 0.0; }
  return ${buf}[${base} + a * N + cidx(clampV(c))];
}
fn ${name}Comp(a: u32, p: vec3f) -> f32 {
  let q = p - vec3f(0.5) + 0.5 * vec3f(unitAxis(a));
  let b = vec3i(floor(q));
  let t = q - floor(q);
  let c000 = ${name}Face(a, b);
  let c100 = ${name}Face(a, b + vec3i(1, 0, 0));
  let c010 = ${name}Face(a, b + vec3i(0, 1, 0));
  let c110 = ${name}Face(a, b + vec3i(1, 1, 0));
  let c001 = ${name}Face(a, b + vec3i(0, 0, 1));
  let c101 = ${name}Face(a, b + vec3i(1, 0, 1));
  let c011 = ${name}Face(a, b + vec3i(0, 1, 1));
  let c111 = ${name}Face(a, b + vec3i(1, 1, 1));
  let x00 = mix(c000, c100, t.x); let x10 = mix(c010, c110, t.x);
  let x01 = mix(c001, c101, t.x); let x11 = mix(c011, c111, t.x);
  return mix(mix(x00, x10, t.y), mix(x01, x11, t.y), t.z);
}
fn ${name}(p: vec3f) -> vec3f { return vec3f(${name}Comp(0u, p), ${name}Comp(1u, p), ${name}Comp(2u, p)); }
`;
}

/** Vertex level-set samplers (trilinear, and monotone Catmull-Rom tricubic) over `buf` at `base`. */
export function phiSampler(name: string, buf: string, base: string): string {
  return /* wgsl */ `
fn ${name}At(v: vec3i) -> f32 { return ${buf}[${base} + cidx(clampV(v))]; }
fn ${name}Tri(p: vec3f) -> f32 {
  let b = vec3i(floor(p)); let t = p - floor(p);
  let x00 = mix(${name}At(b), ${name}At(b + vec3i(1, 0, 0)), t.x);
  let x10 = mix(${name}At(b + vec3i(0, 1, 0)), ${name}At(b + vec3i(1, 1, 0)), t.x);
  let x01 = mix(${name}At(b + vec3i(0, 0, 1)), ${name}At(b + vec3i(1, 0, 1)), t.x);
  let x11 = mix(${name}At(b + vec3i(0, 1, 1)), ${name}At(b + vec3i(1, 1, 1)), t.x);
  return mix(mix(x00, x10, t.y), mix(x01, x11, t.y), t.z);
}
fn ${name}CRw(t: f32) -> vec4f {
  let t2 = t * t; let t3 = t2 * t;
  return vec4f(-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0, -1.5 * t3 + 2.0 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2);
}
/** Catmull-Rom tricubic, clamped to the range of the 8 enclosing vertices (monotone). */
fn ${name}Cubic(p: vec3f) -> f32 {
  let b = vec3i(floor(p)); let t = p - floor(p);
  let wx = ${name}CRw(t.x); let wy = ${name}CRw(t.y); let wz = ${name}CRw(t.z);
  var acc = 0.0; var lo = 1e30; var hi = -1e30;
  for (var k = 0; k < 4; k++) {
    var accY = 0.0;
    for (var j = 0; j < 4; j++) {
      let r = b + vec3i(-1, j - 1, k - 1);
      let v0 = ${name}At(r); let v1 = ${name}At(r + vec3i(1, 0, 0));
      let v2 = ${name}At(r + vec3i(2, 0, 0)); let v3 = ${name}At(r + vec3i(3, 0, 0));
      if ((j == 1 || j == 2) && (k == 1 || k == 2)) { lo = min(lo, min(v1, v2)); hi = max(hi, max(v1, v2)); }
      accY += wy[j] * dot(wx, vec4f(v0, v1, v2, v3));
    }
    acc += wz[k] * accY;
  }
  return clamp(acc, lo, hi);
}
`;
}

export type BindingKind = "rw" | "ro" | "uniform" | "volume";
export interface BindingSpec { readonly name: string; readonly kind: BindingKind; readonly type?: string }

/** WGSL declarations for a binding list (group 0, binding = index). */
export function declare(bindings: readonly BindingSpec[]): string {
  return bindings.map((b, i) => {
    switch (b.kind) {
      case "rw": return `@group(0) @binding(${i}) var<storage, read_write> ${b.name}: ${b.type};`;
      case "ro": return `@group(0) @binding(${i}) var<storage, read> ${b.name}: ${b.type};`;
      case "uniform": return `@group(0) @binding(${i}) var<uniform> ${b.name}: ${b.type};`;
      case "volume": return `@group(0) @binding(${i}) var ${b.name}: texture_storage_3d<r32float, write>;`;
    }
  }).join("\n");
}

export function bindGroupLayoutEntries(bindings: readonly BindingSpec[]): GPUBindGroupLayoutEntry[] {
  return bindings.map((b, i): GPUBindGroupLayoutEntry => {
    const base = { binding: i, visibility: 4 /* GPUShaderStage.COMPUTE */ };
    switch (b.kind) {
      case "rw": return { ...base, buffer: { type: "storage" } };
      case "ro": return { ...base, buffer: { type: "read-only-storage" } };
      case "uniform": return { ...base, buffer: { type: "uniform" } };
      case "volume": return { ...base, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } };
    }
  });
}
