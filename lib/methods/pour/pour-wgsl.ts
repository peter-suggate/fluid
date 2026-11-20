/**
 * Pour solver: shared WGSL (constants, brick-major addressing, the baked glass and the field samplers), forked
 * from seep-wgsl.ts (docs/plans/pour-rewrite-2026-11-20.md §2).
 */
import { BRICKS, CELLS, DT, CELL_M, GRAVITY, NX, NY, NZ, PHI_MAX, RIM_S0, RIM_S1, RIM_Y, WALL_S0, WALL_S1, type PourInflow } from "./pour-layout";

/** The composite h/4h BULK region of the seep fork: none in the pour (its deep pool moves); the machinery stays
 *  compiled out behind NBULK = 0 for a later dynamic composite (plan §2.3). */
export const NBULK = 0;

/** Particle pool capacity (each pool): the brick engine's measured peak on the pour (782k at band 3) with margin. */
export const PARTICLE_CAPACITY = 3 << 19;
/** FLIP fraction (Ferstl 2016 uses a FLIP/PIC blend; the reference runs 0.95). */
export const FLIP_RATIO = 0.95;
/** Narrow band width R (cells) of the reference (Ferstl 2016 §3.3; the brick engine's particleBandDepth 3): particles
 *  deeper than it retire, band cells are seeded to it. */
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
 *  (3.33 a step on average; it was 12 with bricks alone), so 9 leaves two cycles of margin and no step is capped.
 *  A capped step (the residual still above tolerance at the cap) leaves an unconverged pressure: it latches
 *  FAIL.PRESSURE_CAP and capHits counts it. Every cycle slot costs its gated dispatches even when empty.
 *  The pour after its inflow takes 10-12 cycles a step (steps 300-377 of pour-p1-mass, 2026-11-20; step 377 capped
 *  at 12), so 16. */
export const MAX_CYCLES = 16;
/** Aggregation over-correction: A_c = R A P / OC (Braess 1995; piecewise-constant aggregation). A
 *  preconditioner parameter only (the converged solution is the same). Measured on the seep at
 *  1200 steps: OC 1.8 takes 7.6 cycles a step and OC 1.6 takes 6.7. The chamber's free-surface rows
 *  are what converge last, and they favour the weaker scaling. 1.4 and 2.1 are both slower. */
export const AGGREGATION_OC = 1.6;
/** Backtrace segment cap: 48 cells a step (14 m/s) at 1-cell segments; beyond it segments lengthen (recorded in maxSegments, not fatal). */
export const MAX_SEGMENTS = 48;
/** velocityTrace's 64-cell slabs a brick (its dispatch's y). */
export const TRACE_SLABS = 8;

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
  /** In the args buffer: the active list by TRACE_SLABS (velocityTrace's slabs along y). */
  ARG_TRACE: 48,
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
  /** Particles seeded in the band (cumulative). */
  EMITTED: 73,
  /** Shifted fine blocks: the operator's append counter, and their number this step. */
  SHIFT_COUNT: 74, N_SHIFT: 75,
  /** The brick engine's volume control (velocity-kernels.wgsl.ts FH): the gauge's filled volume by step parity (2 pairs
   *  of [low, high] words, cells in 2^-20 units), the budget (cells, f32 bits), and the last rate (1/s) and filled
   *  volume (cells), f32 bits, for diagnostics. */
  VOL_SUM: 76, VOL_BUDGET: 80, RATE: 81, FILLED: 82,
  /** G2P's split records appended past the dst count (classify adds them to the count), and the records the crowding
   *  merge absorbed (cumulative). */
  SPLIT: 83, MERGED: 84,
  RED: 256,
  /** In the args buffer: gated arguments per PCG iteration k, records of 4 words: U2, U1, U0, P, X(k+1); slot 5 the shifted fine blocks. */
  MG_ARGS: 64,
  WORDS: 256 + 2 * 192,
} as const;
/** Words per PCG iteration in the args buffer, and the args buffer's size in words. */
export const MG_STRIDE = 32;
export const ARGS_WORDS = C.MG_ARGS + MG_STRIDE * (MAX_CYCLES + 1);
export const FAIL = { DEEP_SURFACE: 1, POOL: 2, SEGMENTS: 4, COARSEST_PIVOT: 8, PRESSURE_CAP: 16 } as const;

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
  /** The inflow schedule (3 words a step: strength, sweep, flags), and this step's words (classify copies them). */
  INFLOW: 19, INFLOW_NOW: 20,
  /** Nonzero when the brick held a liquid cell (open, PHIC < 0) at this step's operator (cleared with the brick). */
  LIQCELL: 21,
  COUNT: 22,
} as const;
/** BULK work: 64 bricks (512 coarse cells) per 512-thread workgroup; these workgroups come first in the dispatches that include BULK. */
export const BULK_WG = Math.ceil(NBULK / 64);
/** The dense 4h lattice of the BULK buffer: U (3 components) then Us (3). */
export const COARSE_DIMS = [NX / 4, NY / 4, NZ / 4] as const;
export const COARSE_LATTICE = COARSE_DIMS[0] * COARSE_DIMS[1] * COARSE_DIMS[2];

/** Coarse multigrid levels 1..6 (L0 is the fine brick-major lattice): level 3 is one cell a brick, level 6 is 2 x 4 x 2. */
export const LEVELS = 6;
export function levelDims(l: number): [number, number, number] { return [NX >> l, NY >> l, NZ >> l]; }
export const LEVEL_OFFSET: number[] = (() => {
  const off = [0, 0];
  for (let l = 1; l <= LEVELS; l++) { const [x, y, z] = levelDims(l); off.push(off[l] + x * y * z); }
  return off;
})();
export const COARSE_CELLS = LEVEL_OFFSET[LEVELS + 1];
/** Cells of the coarsest level (2 x 4 x 2), solved exactly through its explicit inverse (16 x 16 entries, one per thread). */
export const COARSEST_CELLS = levelDims(LEVELS).reduce((a, b) => a * b, 1);
if (COARSEST_CELLS !== 16) throw new Error(`pour: coarsest level of ${COARSEST_CELLS} cells, the solve is laid out for 16`);
/** Levels 4 and 5 (coarseSolve holds them in workgroup memory, one cell a thread at level 4). */
export const L4_CELLS = levelDims(4).reduce((a, b) => a * b, 1), L5_CELLS = levelDims(5).reduce((a, b) => a * b, 1);
if (L4_CELLS !== 1024 || L5_CELLS !== 128) throw new Error(`pour: levels 4/5 of ${L4_CELLS}/${L5_CELLS} cells`);

const f = (x: number) => { const s = String(x); return s.includes(".") || s.includes("e") ? s : s + ".0"; };

export const COMMON_WGSL = /* wgsl */ `
const NX: i32 = ${NX}; const NY: i32 = ${NY}; const NZ: i32 = ${NZ};
const NV: vec3i = vec3i(${NX}, ${NY}, ${NZ});
const N: u32 = ${CELLS}u;
const BRICKS: u32 = ${BRICKS}u;
const CFL: f32 = ${f(DT / CELL_M)};
const DT_S: f32 = ${f(DT)};
const CELL_H: f32 = ${f(CELL_M)};
/** Particle speed (cells a step) along an axis beyond which a brick is marked fast along it (bricks FAST bit a): its liquid reaches two bricks (classify: a front faster than 4 cells a step outruns one). */
const FAST_CELLS: f32 = 4.0;
fn fastBits(v: vec3f) -> u32 { let f = abs(v) * CFL > vec3f(FAST_CELLS); return select(0u, 1u, f.x) | select(0u, 2u, f.y) | select(0u, 4u, f.z); }
const GDT: f32 = ${f(GRAVITY * DT)};
const PHI_MAX: f32 = ${f(PHI_MAX)};
const UNKNOWN_PHI: f32 = ${f(PHI_MAX + 1)};
const PN: u32 = ${CELLS}u;
/** pgrid's third region: each cell's record count as the operator found it (the pre-move runs G2P's mass rules read). */
const PN2: u32 = ${2 * CELLS}u;
const CAP: u32 = ${PARTICLE_CAPACITY}u;
const BAND: f32 = ${f(BAND)};
const RCOMB: f32 = ${f(COMBINE_BAND)};
const RP: f32 = ${f(PARTICLE_RADIUS)};
const THETA_MIN: f32 = ${f(THETA_MIN)};
const FLIP: f32 = ${f(FLIP_RATIO)};
const MAXSEG: f32 = ${f(MAX_SEGMENTS)};
const CLASS_DEAD: u32 = 1u; const CLASS_DEEP: u32 = 2u; const CLASS_BULK: u32 = 16u;
const NBULK: u32 = ${NBULK}u; const BULK_WG: u32 = ${BULK_WG}u;
const B_LBULK: u32 = ${B.LIST_BULK * BRICKS}u; const B_LSHADOW: u32 = ${B.LIST_SHADOW * BRICKS}u;
const NCL: u32 = ${COARSE_LATTICE}u; const BU_OFF: u32 = 0u; const BUS_OFF: u32 = ${3 * COARSE_LATTICE}u;
const C_ARG_ACTB: u32 = ${C.ARG_ACTB}u; const C_ARG_ACTBB: u32 = ${C.ARG_ACTBB}u;
const C_N_LIQ: u32 = ${C.N_LIQ}u; const C_LIQ_COUNT: u32 = ${C.LIQ_COUNT}u; const C_L4_COUNT: u32 = ${C.L4_COUNT}u; const C_N_L4: u32 = ${C.N_L4}u;
const C_EMITTED: u32 = ${C.EMITTED}u;
const C_VOL_SUM: u32 = ${C.VOL_SUM}u; const C_VOL_BUDGET: u32 = ${C.VOL_BUDGET}u; const C_RATE: u32 = ${C.RATE}u; const C_FILLED: u32 = ${C.FILLED}u;
const VOL_UNIT: f32 = 1048576.0;
const C_SPLIT: u32 = ${C.SPLIT}u; const C_MERGED: u32 = ${C.MERGED}u;
const B_INFLOW: u32 = ${B.INFLOW * BRICKS}u; const B_INNOW: u32 = ${B.INFLOW_NOW * BRICKS}u;
const C_ARG_L4: u32 = ${C.ARG_L4}u; const C_ARG_L4U: u32 = ${C.ARG_L4U}u; const B_L4FLAG: u32 = ${B.L4_FLAG * BRICKS}u; const B_L4LIST: u32 = ${B.L4_LIST * BRICKS}u; const C_ARG_LIQ: u32 = ${C.ARG_LIQ}u; const C_ARG_LIQB: u32 = ${C.ARG_LIQB}u; const C_ARG_LIQBB: u32 = ${C.ARG_LIQBB}u;
const B_SFLAG: u32 = ${B.SHIFT_FLAG * BRICKS}u; const B_LSHIFT: u32 = ${B.LIST_SHIFT * BRICKS}u; const C_SHIFT_COUNT: u32 = ${C.SHIFT_COUNT}u; const C_N_SHIFT: u32 = ${C.N_SHIFT}u; const C_ARG_SHIFT: u32 = ${C.ARG_SHIFT}u; const C_ARG_TRACE: u32 = ${C.ARG_TRACE}u; const TRACE_SLABS: u32 = ${TRACE_SLABS}u;
const B_LBULK2: u32 = ${B.LIST_BULK2 * BRICKS}u; const B_LLIQ: u32 = ${B.LIST_BULK2 * BRICKS + NBULK}u;
/** Composite h/4h: a fine face against a coarse cell couples over 2.5 cells (half a fine and half a coarse cell). */
const SUBC: f32 = 0.4;
const CDIMS: vec3i = vec3i(${COARSE_DIMS.join(", ")});
fn clIndex(k: vec3i) -> u32 { return u32(k.x + CDIMS.x * (k.y + CDIMS.y * k.z)); }
/** pres/op slot of coarse cell k: its BULK brick's first eight slots. */
fn bulkSlot(k: vec3i) -> u32 { return (brickOfCell(k * 4) << 9u) | u32((k.x & 1) | ((k.y & 1) << 1u) | ((k.z & 1) << 2u)); }
const B_CLASS: u32 = ${B.CLASS * BRICKS}u; const B_FLAGS: u32 = ${B.FLAGS * BRICKS}u; const B_MINPHI: u32 = ${B.MINPHI * BRICKS}u;
const B_PCOUNT: u32 = ${B.PCOUNT * BRICKS}u; const B_LIQCELL: u32 = ${B.LIQCELL * BRICKS}u; const B_FAST: u32 = ${B.FAST * BRICKS}u; const B_LACT: u32 = ${B.LIST_BULK * BRICKS + NBULK}u; const B_LSURF: u32 = ${B.LIST_SURF * BRICKS}u;
const B_LPART: u32 = ${B.LIST_PART * BRICKS}u; const B_LCLEAR: u32 = ${B.LIST_CLEAR * BRICKS}u;
const B_PSTART: u32 = ${B.PSTART * BRICKS}u; const B_PNUM: u32 = ${B.PNUM * BRICKS}u;
const C_ARG_ACTIVE: u32 = ${C.ARG_ACTIVE}u; const C_ARG_SURF: u32 = ${C.ARG_SURF}u; const C_ARG_PART: u32 = ${C.ARG_PART}u; const C_ARG_CLEAR: u32 = ${C.ARG_CLEAR}u;
const C_STEP: u32 = ${C.STEP}u; const C_FAIL: u32 = ${C.FAIL}u; const C_PCOUNT: u32 = ${C.PCOUNT}u; const C_DELETED: u32 = ${C.DELETED}u; const C_DEL_DEEP: u32 = ${C.DEL_DEEP}u; const C_DEL_CAP: u32 = ${C.DEL_CAP}u; const C_DEL_OUT: u32 = ${C.DEL_OUT}u;
const C_SEEDED: u32 = ${C.SEEDED}u; const C_RES: u32 = ${C.RES}u; const C_CONV: u32 = ${C.CONV}u; const C_PARITY: u32 = ${C.PARITY}u;
const C_CYCLES: u32 = ${C.CYCLES}u; const C_FIRST_RES: u32 = ${C.FIRST_RES}u; const C_LAST_RES: u32 = ${C.LAST_RES}u;
const C_CAP_HITS: u32 = ${C.CAP_HITS}u; const C_CYCLES_TOTAL: u32 = ${C.CYCLES_TOTAL}u; const C_MAX_SEG: u32 = ${C.MAX_SEG}u;
const C_OVERFLOW: u32 = ${C.OVERFLOW}u; const C_MG_ARGS: u32 = ${C.MG_ARGS}u; const C_N_ACTIVE: u32 = ${C.N_ACTIVE}u; const C_RES_HIST: u32 = ${C.RES_HIST}u; const C_ALPHA: u32 = ${C.ALPHA}u; const C_RHO: u32 = ${C.RHO}u; const C_RED: u32 = ${C.RED}u;
const FAIL_DEEP: u32 = ${FAIL.DEEP_SURFACE}u; const FAIL_POOL: u32 = ${FAIL.POOL}u; const FAIL_SEG: u32 = ${FAIL.SEGMENTS}u; const FAIL_PIVOT: u32 = ${FAIL.COARSEST_PIVOT}u; const FAIL_CAP: u32 = ${FAIL.PRESSURE_CAP}u;
const DEAD_PARTICLE: u32 = 0xffffffffu;

const BDIMS: vec3i = vec3i(${NX / 8}, ${NY / 8}, ${NZ / 8});
fn brickCoord(b: u32) -> vec3i { return vec3i(i32(b & 15u), i32((b >> 4u) & 31u), i32(b >> 9u)); }
fn brickIdOf(q: vec3i) -> u32 { return u32(q.x) | (u32(q.y) << 4u) | (u32(q.z) << 9u); }
fn brickInBox(q: vec3i) -> bool { return all(q >= vec3i(0)) && all(q < BDIMS); }
fn brickOfCell(c: vec3i) -> u32 { return brickIdOf(c >> vec3u(3u)); }
fn cidx(c: vec3i) -> u32 { return (brickOfCell(c) << 9u) | u32((c.x & 7) | ((c.y & 7) << 3u) | ((c.z & 7) << 6u)); }
fn localCoord(l: u32) -> vec3i { return vec3i(i32(l & 7u), i32((l >> 3u) & 7u), i32(l >> 6u)); }
fn inBox(c: vec3i) -> bool { return all(c >= vec3i(0)) && all(c < NV); }
fn clampV(c: vec3i) -> vec3i { return clamp(c, vec3i(0), NV - vec3i(1)); }
fn dimOf(a: u32) -> i32 { return select(select(NZ, NY, a == 1u), NX, a == 0u); }
fn unitAxis(a: u32) -> vec3i { return vec3i(select(0, 1, a == 0u), select(0, 1, a == 1u), select(0, 1, a == 2u)); }
/** Baked solids: the box, and the glass (pour-layout.ts glassSolid, the brick engine's voxel mask). */
fn glassSolid(c: vec3i) -> bool {
  if (c.y > ${RIM_Y}) { return false; }
  let dx = 2 * c.x + 1 - NX; let dz = 2 * c.z + 1 - NZ; let s = dx * dx + dz * dz;
  if (c.y < ${RIM_Y}) { return s >= ${WALL_S0} && s <= ${WALL_S1}; }
  return s >= ${RIM_S0} && s <= ${RIM_S1};
}
fn cellOpen(c: vec3i) -> bool { return inBox(c) && !glassSolid(c); }
/** The lower face of c along a (between c - e_a and c) carries flow when both sides are open. */
fn faceOpen(c: vec3i, a: u32) -> bool { return cellOpen(c) && cellOpen(c - unitAxis(a)); }
/** umOpenAt: a point on a face belongs to every incident cell, and is open when one of them is. */
fn pointOpen(x: vec3f) -> bool {
  let base = vec3i(floor(x));
  let on = x == floor(x); let faces = u32(on.x) | (u32(on.y) << 1u) | (u32(on.z) << 2u);
  for (var k = 0u; k <= faces; k++) {
    if ((k & faces) != k) { continue; }
    if (cellOpen(base - vec3i(i32(k & 1u), i32((k >> 1u) & 1u), i32(k >> 2u)))) { return true; }
  }
  return false;
}
/** umWalk: the last open point of the segment a -> b in half-cell steps (the glass is 2.6 cells thick, so no
 *  step crosses it); b when the whole segment is open. */
fn walkOpen(a: vec3f, b: vec3f) -> vec3f {
  let d = b - a;
  let n = u32(clamp(ceil(2.0 * max(abs(d.x), max(abs(d.y), abs(d.z)))), 1.0, 64.0));
  var prev = a;
  for (var s = 1u; s <= n; s++) {
    let x = a + d * (f32(s) / f32(n));
    if (!pointOpen(x)) { return prev; }
    prev = x;
  }
  return b;
}
/** A segment from p of length reach can touch the glass (its rim rows and wall, with a cell of margin). */
fn nearGlass(p: vec3f, reach: f32) -> bool {
  let r = length(p.xz - vec2f(${NX / 2}.0, ${NZ / 2}.0));
  return p.y < ${RIM_Y + 2}.0 + reach && abs(r - ${((Math.sqrt(WALL_S0) + Math.sqrt(WALL_S1)) / 4).toFixed(4)}) < ${((Math.sqrt(WALL_S1) - Math.sqrt(WALL_S0)) / 4 + 1.5).toFixed(4)} + reach;
}
/** Closed: a wall face with a fixed zero normal velocity (exactly one side open). */
fn faceClosed(c: vec3i, a: u32) -> bool { return cellOpen(c) != cellOpen(c - unitAxis(a)); }
/** Index in the particle arrays: the whole box is the particle region. */
fn pidx(c: vec3i) -> u32 { return cidx(c); }
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

const vec3 = (v: readonly number[]) => `vec3f(${v.map(f).join(", ")})`;
/**
 * The nozzle (the brick engine's brick-source.wgsl.ts BRICK_SOURCE_RING_WGSL, its ring words baked as constants and
 * this step's [strength, sweep, flags] read from the bricks table, which classify fills from the schedule). Same
 * functions, same order of operations: a cell q's centre q + 1/2, the face plane face + 1, an edge width of half a
 * face diagonal. q is the cell whose +face is meant (the lower face (c, a) here is q = c - e_a).
 */
export function inflowWGSL(n: PourInflow, atomicBricks: boolean): string {
  const rd = (e: string) => atomicBricks ? `atomicLoad(&bricks[${e}])` : `bricks[${e}]`;
  const speed = Math.hypot(...n.velocity_mps), dir = n.velocity_mps.map(x => x / speed);
  const forward = n.velocity_mps[n.axis]! >= 0, receiver = forward ? n.face + 1 : n.face, donor = forward ? receiver - 1 : receiver + 1;
  const ax = "xyz"[n.axis]!;
  const blo = n.box[0].map(x => Math.floor(x / 8)), bhi = n.box[1].map(x => Math.floor((x - 1) / 8));
  return /* wgsl */ `
const IN_OUTLET: vec3f = ${vec3(n.outlet)};
const IN_RADIUS: f32 = ${f(n.radius)};
const IN_VEL: vec3f = ${vec3(n.velocity_mps)};
const IN_DIR: vec3f = ${vec3(dir)};
const IN_SPEED: f32 = ${f(speed)};
const IN_SCALE: f32 = ${f(n.scale)};
const IN_AXIS: u32 = ${n.axis}u;
const IN_RECEIVER: i32 = ${receiver}; const IN_DONOR: i32 = ${donor};
const IN_OVERLAP: f32 = 2.0;
/** The bricks the plug's cell box covers (inclusive): active while the nozzle runs. */
const IN_BRICK_LO: vec3i = vec3i(${blo.join(", ")}); const IN_BRICK_HI: vec3i = vec3i(${bhi.join(", ")});
fn inStrength() -> f32 { return bitcast<f32>(${rd("B_INNOW")}); }
fn inSweep() -> f32 { return bitcast<f32>(${rd("B_INNOW + 1u")}); }
fn inOn() -> bool { return (${rd("B_INNOW + 2u")} & 1u) != 0u; }
fn inContinues() -> bool { return (${rd("B_INNOW + 2u")} & 2u) != 0u; }
fn inCoverage(rel: vec3f) -> f32 {
  let axial = dot(rel, IN_DIR); let radial = length(rel - axial * IN_DIR);
  let c = clamp(0.5 + 0.5 * (IN_RADIUS - radial) / (0.5 * sqrt(2.0)), 0.0, 1.0); return c * c * (3.0 - 2.0 * c);
}
fn inAperture(q: vec3i) -> f32 { var point = vec3f(q) + vec3f(0.5); point.${ax} = ${f(n.face + 1)}; return inCoverage(point - IN_OUTLET); }
fn inVelocityCell(q: vec3i) -> bool { return inOn() && (q.${ax} == IN_RECEIVER || q.${ax} == IN_DONOR) && inAperture(q) > 0.0; }
/** applyInflowVelocity's component a at q (after projection). */
fn inProjected(q: vec3i, a: u32, value: f32) -> f32 {
  if (!inVelocityCell(q)) { return value; }
  let desired = IN_VEL * inStrength();
  if (a == IN_AXIS) { return desired.${ax} * IN_SCALE; }
  if (q.${ax} == IN_RECEIVER) { return desired[a]; }
  return value;
}
fn inSwept(q: vec3i) -> bool {
  let sweep = inSweep();
  if (IN_SPEED * inStrength() <= 1e-6 || sweep <= 0.0) { return false; }
  let rel = vec3f(q) + vec3f(0.5) - IN_OUTLET;
  let axial = dot(rel, IN_DIR); let half = 0.5 * dot(abs(IN_DIR), vec3f(1.0));
  let overlap = max(0.0, min(axial + half, sweep) - max(axial - half, 0.0));
  return overlap > 0.0 && inCoverage(rel) * IN_SCALE > 0.0;
}
fn inSweptCell(q: vec3i) -> bool { return inOn() && ((q.${ax} == IN_DONOR && inAperture(q) > 0.0) || inSwept(q)); }
/** applyInflowSweptVelocity's component a at q (after the body force). */
fn inForced(q: vec3i, a: u32, value: f32) -> f32 { if (!inSweptCell(q)) { return value; } return IN_VEL[a] * inStrength(); }
/** uvSourcePhi's plug (cells) unioned into value. */
fn inPhi(p: vec3f, value: f32) -> f32 {
  if (!inOn() || IN_SPEED * inStrength() <= 1e-6) { return value; }
  let delta = p - IN_OUTLET; let axial = dot(delta, IN_DIR);
  let side = length(delta - axial * IN_DIR) - IN_RADIUS;
  let front = axial - inSweep() - IN_OVERLAP; let cap = max(-axial, front);
  return min(value, select(max(side, cap), select(max(side, front), side, inContinues()), cap <= 0.0));
}
`;
}
