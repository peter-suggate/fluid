/**
 * Pour solver kernels (WGSL), forked from the seep solver's (lib/methods/seep/seep-kernels.ts):
 * docs/plans/pour-rewrite-2026-11-20.md §2.2. The physics is the brick engine's NB-FLIP on the pour: a level set
 * everywhere, FLIP particles in the band (phi >= -BAND, seeded to it, retired below it), the nozzle's plug, forced
 * and projected faces, a closed box and the glass's voxels.
 *
 * Every kernel is a pair (bindings, code). Brick kernels run one workgroup per
 * listed brick (indirect dispatch over GPU-compacted lists), so per-step work
 * follows the active bricks and never the box.
 *
 * Packed buffers (maxStorageBuffersPerShaderStage is 10 on this device):
 *   phis   [phi | phiT | phiC]          vertex level set, its scratch, cell-centre level set
 *   vel    [U(3) | Us(3) | Ut(3)]       extended velocity, advected+forces velocity, scratch
 *   op     [diag | rhs]                 fine pressure operator diagonal and right-hand side
 *   bits   [nmask | known]              liquid-neighbour mask (6 bits), extension validity (2 x 3 bits)
 *   pres   [PA | PB]                    pressure ping-pong (P = p dt / (rho h), m/s)
 *   coarse [op(4) | e | b]              multigrid levels 1..6 (diag, +x, +y, +z couplings; correction; rhs)
 */
import { levelSetFillWGSL } from "../../core/level-set-fill.wgsl";
import { CELL_M, NX, NY, NZ, type PourInflow, VOLUME_CONTROL_S } from "./pour-layout";
import { nbfVolumeWGSL } from "../narrow-band/velocity-formulas.wgsl";
import { geometricPlaneBoxWGSL } from "../../core/geometric-plane-box.wgsl";
import {
  AGGREGATION_OC, COARSE_CELLS, COARSEST_CELLS, COMMON_WGSL, L4_CELLS, L5_CELLS, LEVEL_OFFSET, MAX_CYCLES, MG_STRIDE, PRESSURE_TOLERANCE,
  declare, inflowWGSL, levelDims, phiSampler, velocitySampler, type BindingSpec,
} from "./pour-wgsl";

export interface KernelSource { readonly label: string; readonly bindings: readonly BindingSpec[]; readonly code: string; readonly entry: string }

const ctrlRW: BindingSpec = { name: "ctrl", kind: "rw", type: "array<atomic<u32>>" };
/** Indirect dispatch arguments (apart from ctrl: a buffer cannot be INDIRECT and writable storage in one dispatch). */
const argsRW: BindingSpec = { name: "args", kind: "rw", type: "array<atomic<u32>>" };
const ctrlRO: BindingSpec = { name: "ctrl", kind: "ro", type: "array<u32>" };
const bricksRW: BindingSpec = { name: "bricks", kind: "rw", type: "array<atomic<u32>>" };
const bricksRO: BindingSpec = { name: "bricks", kind: "ro", type: "array<u32>" };
const phisRW: BindingSpec = { name: "phis", kind: "rw", type: "array<f32>" };
const phisRO: BindingSpec = { name: "phis", kind: "ro", type: "array<f32>" };
const velRW: BindingSpec = { name: "vel", kind: "rw", type: "array<f32>" };
const velRO: BindingSpec = { name: "vel", kind: "ro", type: "array<f32>" };
const opRW: BindingSpec = { name: "op", kind: "rw", type: "array<f32>" };
const opRO: BindingSpec = { name: "op", kind: "ro", type: "array<f32>" };
const bitsRW: BindingSpec = { name: "bits", kind: "rw", type: "array<u32>" };
const bitsRO: BindingSpec = { name: "bits", kind: "ro", type: "array<u32>" };
const presRW: BindingSpec = { name: "pres", kind: "rw", type: "array<f32>" };
const presRO: BindingSpec = { name: "pres", kind: "ro", type: "array<f32>" };
const coarseRW: BindingSpec = { name: "coarse", kind: "rw", type: "array<f32>" };
/** Two pools (parity) of CAP particles, each CAP positions then CAP velocities (positions contiguous:
 *  the separation and P2G neighbour reads touch no velocity). */
const partsRW: BindingSpec = { name: "parts", kind: "rw", type: "array<vec4f>" };
const partsRO: BindingSpec = { name: "parts", kind: "ro", type: "array<vec4f>" };
const pcellRW: BindingSpec = { name: "pcell", kind: "rw", type: "array<u32>" };
const pcellRO: BindingSpec = { name: "pcell", kind: "ro", type: "array<u32>" };
const pgridRW: BindingSpec = { name: "pgrid", kind: "rw", type: "array<atomic<u32>>" };
const pgridRO: BindingSpec = { name: "pgrid", kind: "ro", type: "array<u32>" };
/** The nearest-sample keys, a vertex each (the brick engine's FC_KEY): zero between steps (pour.combine clears its
 *  bricks' vertices as it reads them; every vertex a splat reaches lies in an active brick). */
const keysRW: BindingSpec = { name: "keys", kind: "rw", type: "array<atomic<u32>>" };
const keysPlainRW: BindingSpec = { name: "keys", kind: "rw", type: "array<u32>" };
const accRW: BindingSpec = { name: "acc", kind: "rw", type: "array<atomic<i32>>" };
const cfgU: BindingSpec = { name: "cfg", kind: "uniform", type: "Cfg" };
const volW: BindingSpec = { name: "vol", kind: "volume" };
const bulkRW: BindingSpec = { name: "bulk", kind: "rw", type: "array<f32>" };
const bulkRO: BindingSpec = { name: "bulk", kind: "ro", type: "array<f32>" };

/**
 * BULK (composite h/4h) helpers. A coarse cell k (4h, on the 128x64x16
 * lattice) covers fine cells 4k..4k+3; its MAC faces are the low faces of the
 * lattice (face (a, k) at 4 k[a] along a). Where a coarse cell meets fine
 * cells, the 16 fine faces between them are the interface: fine faces, stored
 * at their fine position (also when that is inside a BULK brick), coupled to
 * the coarse pressure over 2.5 cells (Losasso et al. 2004).
 */
function bulkWGSL(bricksAtomic: boolean): string {
  const rd = (e: string) => bricksAtomic ? `atomicLoad(&bricks[${e}])` : `bricks[${e}]`;
  return /* wgsl */ `
// The pour has no BULK region (NBULK = 0): no brick is composite.
fn bulkAt(q: vec3i) -> bool { return false; }
fn brickBulk(id: u32) -> bool { return bulkAt(brickCoord(id)); }
fn cellBulk(c: vec3i) -> bool { return all(c >= vec3i(0)) && bulkAt(c >> vec3u(3u)); }
fn coarseBulk(k: vec3i) -> bool { return all(k >= vec3i(0)) && bulkAt(k >> vec3u(1u)); }
/** Coarse cell j of the BULK list (eight a brick, in its octant order); NBULK * 8 of them. */
fn bulkCell(j: u32) -> vec3i {
  let b = ${rd("B_LBULK + (j >> 3u)")};
  let r = j & 7u;
  return brickCoord(b) * 2 + vec3i(i32(r & 1u), i32((r >> 1u) & 1u), i32(r >> 2u));
}
/** Fine cell open and not BULK (a fine pressure unknown when liquid). */
fn fineOpen(c: vec3i) -> bool { return cellOpen(c) && !cellBulk(c); }
/** Centre height (cells) of the pressure owner of fine cell c: its coarse cell's if BULK. */
fn ownerY(c: vec3i) -> f32 { return select(f32(c.y) + 0.5, f32((c.y >> 2u) * 4) + 2.0, cellBulk(c)); }
`;
}
/** Fine cell next to coarse cell k across side kk (0..5 = -x,+x,-y,+y,-z,+z), the (u, v)-th of the 16 on that face. */
const BULK_NBR_WGSL = /* wgsl */ `
fn subNbr(k: vec3i, kk: u32, u: i32, v: i32) -> vec3i {
  let a = kk >> 1u;
  let t1 = unitAxis((a + 1u) % 3u); let t2 = unitAxis((a + 2u) % 3u);
  let e = unitAxis(a);
  return 4 * k + select(-e, 4 * e, (kk & 1u) == 1u) + u * t1 + v * t2;
}
`;
/** Composite operator rows (needs pres, op, bits): the fine liquid cells across the
 *  non-BULK sides of coarse cell k, and A applied at k. Coarse-coarse coupling 16/4,
 *  coarse-fine per fine face 1/2.5 (SUBC); symmetric (Losasso et al. 2004). */
const BULK_APPLY_WGSL = /* wgsl */ `
${BULK_NBR_WGSL}
fn subSum(k: vec3i, m: u32, f: u32) -> f32 {
  var s = 0.0;
  for (var kk = 0u; kk < 6u; kk++) {
    if (((m >> kk) & 1u) != 0u) { continue; }
    for (var u = 0; u < 4; u++) { for (var v = 0; v < 4; v++) {
      let c = subNbr(k, kk, u, v);
      if (!inBox(c)) { continue; }
      let ci = cidx(c);
      if (op[DIAG_OFF + ci] > 0.0) { s += SUBC * pres[f + ci]; }
    } }
  }
  return s;
}
fn applyBulk(k: vec3i, sl: u32, f: u32) -> f32 {
  let m = bits[MASK_OFF + sl];
  var s = subSum(k, m, f);
  for (var kk = 0u; kk < 6u; kk++) {
    if (((m >> kk) & 1u) == 0u) { continue; }
    s += 4.0 * pres[f + bulkSlot(k + select(-1, 1, (kk & 1u) == 1u) * unitAxis(kk >> 1u))];
  }
  return op[DIAG_OFF + sl] * pres[f + sl] - s;
}
`;
const surfW: BindingSpec = { name: "surf", kind: "volume" };

/**
 * The render surface (lane H's contract): nodal phi in metres, (dims + 1)^3
 * vertices; vertices at dims repeat dims - 1 (as seepVertexPhi clamps), and a
 * vertex none of whose eight cells is open (inside the wall) reads as air.
 */
const SURF_WGSL = /* wgsl */ `
fn publishSurf(v: vec3i, phi: f32) {
  var open = false;
  for (var k = 0u; k < 8u; k++) { if (cellOpen(v - vec3i(i32(k & 1u), i32((k >> 1u) & 1u), i32(k >> 2u)))) { open = true; } }
  let val = select(PHI_MAX, phi, open) * H_M;
  let ex = select(0, 1, v.x == NX - 1); let ey = select(0, 1, v.y == NY - 1); let ez = select(0, 1, v.z == NZ - 1);
  for (var dz = 0; dz <= ez; dz++) { for (var dy = 0; dy <= ey; dy++) { for (var dx = 0; dx <= ex; dx++) {
    textureStore(surf, v + vec3i(dx, dy, dz), vec4f(val, 0.0, 0.0, 0.0));
  } } }
}
`;

const TC = COARSE_CELLS;
const HEADER = /* wgsl */ `
struct Cfg { a: u32, b: u32, c: u32, d: u32 }
${COMMON_WGSL}
const PHI_OFF: u32 = 0u; const PHIT_OFF: u32 = N; const PHIC_OFF: u32 = 2u * N;
const U_OFF: u32 = 0u; const US_OFF: u32 = 3u * N; const UT_OFF: u32 = 6u * N;
const DIAG_OFF: u32 = 0u; const RHS_OFF: u32 = N;
const MASK_OFF: u32 = 0u; const KNOWN_OFF: u32 = N;
const TC: u32 = ${TC}u; const COP: u32 = 0u; const CE: u32 = ${4 * TC}u; const CB: u32 = ${5 * TC}u; const CEP: u32 = ${6 * TC}u;
// The coarsest level's explicit inverse (row-major, NC6 x NC6), rebuilt each step by coarsenTop.
const NC6: u32 = ${COARSEST_CELLS}u; const CINV: u32 = ${7 * TC}u;
const OC: f32 = ${AGGREGATION_OC};
const TOL: f32 = ${PRESSURE_TOLERANCE};
const MAXC: u32 = ${MAX_CYCLES}u;
const H_M: f32 = ${CELL_M};
const SM: f32 = 1048576.0; const SW: f32 = 1048576.0;
fn tileCoord(t: u32) -> vec3i { return vec3i(i32(t % 10u), i32((t / 10u) % 10u), i32(t / 100u)) - vec3i(1); }
fn tileIndex(lc: vec3i) -> u32 { let q = lc + vec3i(1); return u32(q.x + 10 * (q.y + 10 * q.z)); }
fn octOffset(o: u32) -> vec3f { return vec3f(f32(o & 1u), f32((o >> 1u) & 1u), f32((o >> 2u) & 1u)); }
`;

function levelWGSL(): string {
  const lines: string[] = [];
  for (let l = 1; l <= 6; l++) {
    const [x, y, z] = levelDims(l);
    lines.push(`const L${l}X: i32 = ${x}; const L${l}Y: i32 = ${y}; const L${l}Z: i32 = ${z}; const L${l}O: u32 = ${LEVEL_OFFSET[l]}u;`);
  }
  return lines.join("\n") + `
fn ldims(l: u32) -> vec3i { return vec3i(${NX} >> l, ${NY} >> l, ${NZ} >> l); }
fn loff(l: u32) -> u32 {
  switch (l) { case 1u: { return L1O; } case 2u: { return L2O; } case 3u: { return L3O; }
    case 4u: { return L4O; } case 5u: { return L5O; } default: { return L6O; } }
}
fn lidx(l: u32, c: vec3i) -> u32 { let d = ldims(l); return loff(l) + u32(c.x + d.x * (c.y + d.y * c.z)); }
fn linBox(l: u32, c: vec3i) -> bool { return all(c >= vec3i(0)) && all(c < ldims(l)); }
`;
}

function kernel(label: string, bindings: BindingSpec[], body: string, extra = ""): KernelSource {
  return { label, bindings, entry: "main", code: `${HEADER}\n${declare(bindings)}\n${extra}\n${body}` };
}

// ---------------------------------------------------------------------------
// 1. Brick classification and list compaction (dense over the 8192 bricks). The step counter and this step's
//    inflow words were published by the last step's G2P (the upload, for step 1).
// ---------------------------------------------------------------------------
const classifyKernel = (n: PourInflow) => kernel("pour.classify", [ctrlRW, bricksRW, argsRW, cfgU], /* wgsl */ `
${inflowWGSL(n, true)}
fn liquidish(n: vec3i) -> bool {
  if (!brickInBox(n)) { return false; }
  let b = brickIdOf(n);
  let cls = atomicLoad(&bricks[B_CLASS + b]);
  if ((cls & CLASS_DEAD) != 0u) { return false; }
  if ((cls & CLASS_DEEP) != 0u || atomicLoad(&bricks[B_LIQCELL + b]) != 0u) { return true; }
  return (cls & CLASS_BULK) != 0u && (bitcast<f32>(atomicLoad(&bricks[B_MINPHI + b])) < 0.0 || atomicLoad(&bricks[B_PCOUNT + b]) > 0u);
}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  let b = g.x;
  if (b == 0u) {
    // The last G2P's split records sit past the live count (cfg.a's pool): they join it.
    let split = atomicExchange(&ctrl[C_SPLIT], 0u);
    if (split != 0u) { atomicStore(&ctrl[C_PCOUNT + cfg.a], min(atomicLoad(&ctrl[C_PCOUNT + cfg.a]) + split, CAP)); }
    atomicStore(&ctrl[C_CONV], 0u); atomicStore(&ctrl[C_CYCLES], 0u);
    atomicStore(&ctrl[C_RES], 0u);
    atomicStore(&args[C_ARG_ACTIVE + 1u], 1u); atomicStore(&args[C_ARG_ACTIVE + 2u], 1u);
    atomicStore(&args[C_ARG_TRACE + 1u], TRACE_SLABS); atomicStore(&args[C_ARG_TRACE + 2u], 1u);
    atomicStore(&args[C_ARG_SURF + 1u], 1u); atomicStore(&args[C_ARG_SURF + 2u], 1u);
    atomicStore(&args[C_ARG_PART + 1u], 1u); atomicStore(&args[C_ARG_PART + 2u], 1u);
    atomicStore(&args[C_ARG_CLEAR + 1u], 1u); atomicStore(&args[C_ARG_CLEAR + 2u], 1u);
    atomicStore(&args[C_ARG_ACTB + 1u], 1u); atomicStore(&args[C_ARG_ACTB + 2u], 1u);
    atomicStore(&args[C_ARG_ACTBB + 1u], 1u); atomicStore(&args[C_ARG_ACTBB + 2u], 1u);
    atomicStore(&args[C_ARG_LIQ + 1u], 1u); atomicStore(&args[C_ARG_LIQ + 2u], 1u);
    atomicStore(&args[C_ARG_LIQB + 1u], 1u); atomicStore(&args[C_ARG_LIQB + 2u], 1u);
    atomicStore(&args[C_ARG_LIQBB + 1u], 1u); atomicStore(&args[C_ARG_LIQBB + 2u], 1u);
  }
  if (b >= BRICKS) { return; }
  let cls = atomicLoad(&bricks[B_CLASS + b]);
  // BULK bricks are never fine work: their coarse cells ride in the leading
  // workgroups of the dispatches that include them.
  if ((cls & (CLASS_DEAD | CLASS_BULK)) != 0u) { return; }
  let bc = brickCoord(b);
  // Activation reach, the CFL band of a narrow-band method (the active set
  // holds everything the step's motion reaches: Bridson 2015 §6.3 narrow
  // bands; Ferstl 2016 §3). It is the reach of the liquid: from a brick that
  // held a liquid cell at the last operator (B_LIQCELL), one brick along an
  // axis, two along the axes G2P marked fast there (over FAST_CELLS a step).
  // Liquid moves only where the extension put velocity, which spans 8 cells
  // from liquid cells and is written in active bricks alone: a front at v cells
  // a step crosses a brick in 8 / v steps, and the brick after next must hold
  // the extension a step ahead, so one brick holds v <= 4 and two v <= 8 (the
  // extension's own limit). A brick with particles or a negative vertex but
  // no liquid cell (spray, a grazing surface, a sub-cell film) is active
  // itself and reaches nothing: unsupported records deposit nothing and see no
  // grid, the film's velocity is the extension's, and a record leaving lands
  // in a brick the next classify activates. 2026-11-21: 6400 active bricks to
  // 4000 at step 190, gates unchanged.
  // The nozzle's plug box is active while it runs (the brick engine's pinned source box).
  let r = 2;
  var act = (inOn() && all(bc >= IN_BRICK_LO) && all(bc <= IN_BRICK_HI)) || atomicLoad(&bricks[B_PCOUNT + b]) > 0u
    || bitcast<f32>(atomicLoad(&bricks[B_MINPHI + b])) < 0.0;
  for (var dz = -r; dz <= r; dz++) { for (var dy = -r; dy <= r; dy++) { for (var dx = -r; dx <= r; dx++) {
    let n = bc + vec3i(dx, dy, dz);
    // Axes along which this neighbour lies two bricks away; its records must move fast along each.
    let far2 = select(0u, 1u, abs(dx) == 2) | select(0u, 2u, abs(dy) == 2) | select(0u, 4u, abs(dz) == 2);
    if (liquidish(n) && (far2 == 0u || (atomicLoad(&bricks[B_FAST + brickIdOf(n)]) & far2) == far2)) { act = true; }
  } } }
  let prev = (atomicLoad(&bricks[B_FLAGS + b]) & 1u) != 0u;
  if (act) {
    let i = atomicAdd(&args[C_ARG_ACTIVE], 1u); atomicStore(&bricks[B_LACT + i], b);
    atomicAdd(&args[C_ARG_TRACE], 1u);
    atomicAdd(&args[C_ARG_ACTB], 1u); atomicAdd(&args[C_ARG_ACTBB], 1u);
    let j = atomicAdd(&args[C_ARG_SURF], 1u); atomicStore(&bricks[B_LSURF + j], b);
    let k = atomicAdd(&args[C_ARG_PART], 1u); atomicStore(&bricks[B_LPART + k], b);
  } else if (prev) {
    let i = atomicAdd(&args[C_ARG_CLEAR], 1u); atomicStore(&bricks[B_LCLEAR + i], b);
  }
  atomicStore(&bricks[B_FLAGS + b], select(0u, 1u, act));
}
`);

// ---------------------------------------------------------------------------
// 2. Reset bricks that left the active set (inactive bricks hold air: phi =
//    +PHI_MAX, zero velocity, zero pressure and operator, zero fill).
// ---------------------------------------------------------------------------
export const clearKernel = kernel("pour.clear", [bricksRW, phisRW, velRW, opRW, bitsRW, presRW, coarseRW, volW, surfW], /* wgsl */ `
${levelWGSL()}
${SURF_WGSL}
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let b = atomicLoad(&bricks[B_LCLEAR + wg.x]);
  let bc = brickCoord(b);
  let c = bc * 8 + localCoord(l);
  let i = (b << 9u) | l;
  phis[PHI_OFF + i] = PHI_MAX; phis[PHIT_OFF + i] = PHI_MAX; phis[PHIC_OFF + i] = PHI_MAX;
  for (var k = 0u; k < 9u; k++) { vel[k * N + i] = 0.0; }
  op[DIAG_OFF + i] = 0.0; op[RHS_OFF + i] = 0.0; bits[MASK_OFF + i] = 0u; bits[KNOWN_OFF + i] = 0u;
  for (var k = 0u; k < 6u; k++) { pres[k * N + i] = 0.0; }
  textureStore(vol, c, vec4f(0.0));
  publishSurf(c, PHI_MAX);
  if (l == 0u) { atomicStore(&bricks[B_MINPHI + b], bitcast<u32>(PHI_MAX)); atomicStore(&bricks[B_LIQCELL + b], 0u); }
  if (l < 64u) {
    let q = lidx(1u, bc * 4 + vec3i(i32(l & 3u), i32((l >> 2u) & 3u), i32(l >> 4u)));
    coarse[COP + 4u * q] = 0.0; coarse[COP + 4u * q + 1u] = 0.0; coarse[COP + 4u * q + 2u] = 0.0; coarse[COP + 4u * q + 3u] = 0.0;
    coarse[CE + q] = 0.0; coarse[CB + q] = 0.0; coarse[CEP + q] = 0.0;
  }
  if (l < 8u) {
    let q = lidx(2u, bc * 2 + vec3i(i32(l & 1u), i32((l >> 1u) & 1u), i32(l >> 2u)));
    coarse[COP + 4u * q] = 0.0; coarse[COP + 4u * q + 1u] = 0.0; coarse[COP + 4u * q + 2u] = 0.0; coarse[COP + 4u * q + 3u] = 0.0;
    coarse[CE + q] = 0.0; coarse[CB + q] = 0.0; coarse[CEP + q] = 0.0;
  }
  if (l == 0u) {
    let q = lidx(3u, bc);
    coarse[COP + 4u * q] = 0.0; coarse[COP + 4u * q + 1u] = 0.0; coarse[COP + 4u * q + 2u] = 0.0; coarse[COP + 4u * q + 3u] = 0.0;
    coarse[CE + q] = 0.0; coarse[CB + q] = 0.0; coarse[CEP + q] = 0.0;
  }
}
`);

// ---------------------------------------------------------------------------
// 3. Particle census (cfg.a = parity: src pool), after the level-set advection: a particle whose advected phiT
//    lies deeper than the band is retired (the brick engine's particleBandDepth, Ferstl 2016 §3.3), as is one in a
//    closed cell (none should be: G2P projects every particle out of the solids).
// ---------------------------------------------------------------------------
export const countKernel = kernel("pour.count", [ctrlRW, partsRO, pcellRW, pgridRW, cfgU, phisRO], /* wgsl */ `
${phiSampler("phiT", "phis", "PHIT_OFF")}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  let n = min(atomicLoad(&ctrl[C_PCOUNT + cfg.a]), CAP);
  if (i >= n) { return; }
  let rec = parts[cfg.a * CAP * 2u + i];
  let p = rec.xyz;
  let c = vec3i(floor(p));
  // A record G2P retired (the cell-mass cut, absorbed by the crowding merge) has mass 0.
  if (rec.w <= 0.0) { pcell[i] = DEAD_PARTICLE; return; }
  if (!cellOpen(c)) { pcell[i] = DEAD_PARTICLE; atomicAdd(&ctrl[C_DELETED], 1u); atomicAdd(&ctrl[C_DEL_OUT], 1u); return; }
  if (phiTTri(p) < -BAND) { pcell[i] = DEAD_PARTICLE; atomicAdd(&ctrl[C_DELETED], 1u); atomicAdd(&ctrl[C_DEL_DEEP], 1u); return; }
  let pc = pidx(c);
  // The cell's record count (bits 0..15) and its mass (bits 16..31, the seeding's held mass).
  atomicAdd(&pgrid[pc], 1u + (u32(rec.w) << 16u));
  pcell[i] = pc;
}
`);

/** Particle motion over one step: Heun RK3 through the extended grid velocity U
 *  (k1 given), exact projection out of the baked solids after every stage. */
const PARTICLE_MOVE_WGSL = /* wgsl */ `
const EPS: f32 = 1e-3;
// A particle's centre keeps its radius RW (a quarter cell: eight particles a
// cell rest half a cell apart) from every solid, as it keeps 2 RW from another
// particle under separation (particle radius against the boundary, Bridson
// 2015 §7.6; NB-FLIP's sample radius). A centre on a wall plane would sit
// where the wall's normal velocity is zero, carried along the face and never
// off it: floor films packed into the bottom cell row, beneath the liquid
// their particles were meant to describe.
const RW: f32 = 0.25;
fn collide(p0: vec3f) -> vec3f {
  var p = clamp(p0, vec3f(RW), vec3f(f32(NX), f32(NY), f32(NZ)) - vec3f(RW));
  // Inside a solid voxel (the glass): out by the shortest axis-aligned move to an open cell, to RW past its face
  // (the wall is under three cells along its best axis).
  var c = vec3i(floor(p));
  if (!cellOpen(c)) {
    var best = 1e30; var q = p;
    for (var a = 0u; a < 3u; a++) {
      for (var sd = 0u; sd < 2u; sd++) {
        let s = select(-1, 1, sd == 1u);
        for (var k = 1; k <= 5; k++) {
          let n = c + s * k * unitAxis(a);
          if (!inBox(n)) { break; }
          if (cellOpen(n)) {
            let to = select(f32(n[a]) + 1.0 - RW, f32(n[a]) + RW, s > 0);
            let d = abs(to - p[a]);
            if (d < best) { best = d; q = p; q[a] = to; }
            break;
          }
        }
      }
    }
    p = q; c = vec3i(floor(p));
  }
  // Clearance RW from the solid faces of the cell's six neighbours (box faces are the clamp's).
  for (var a = 0u; a < 3u; a++) {
    let e = unitAxis(a);
    let fr = p[a] - f32(c[a]);
    if (fr < RW && c[a] > 0 && !cellOpen(c - e)) { p[a] = f32(c[a]) + RW; }
    if (fr > 1.0 - RW && c[a] + 1 < dimOf(a) && !cellOpen(c + e)) { p[a] = f32(c[a]) + 1.0 - RW; }
  }
  return p;
}
/** The end of a move x0 -> xn (x0 open), walked against the glass (a splash crosses its 2.6 cells in one step):
 *  stopped at the last open point, it slides on along each axis still open there; an axis the glass blocks loses
 *  the particle's velocity into it, as does one the clearance pushes back (free slip, Bridson 2015 §7.6). */
fn moveEnd(x0: vec3f, xn: vec3f, v: ptr<function, vec3f>) -> vec3f {
  var xw = xn;
  if (nearGlass(x0, length(xn - x0))) {
    let w = walkOpen(x0, xn);
    if (any(w != xn)) {
      var to = w;
      for (var a = 0u; a < 3u; a++) {
        let s = xn[a] - w[a];
        if (s == 0.0) { continue; }
        var probe = w; probe[a] += clamp(s, -0.5, 0.5);
        if (pointOpen(probe)) { to[a] = xn[a]; } else if ((*v)[a] * s > 0.0) { (*v)[a] = 0.0; }
      }
      xw = walkOpen(w, to);
    }
  }
  let xc = collide(xw);
  let push = xw - xc;
  for (var a = 0u; a < 3u; a++) {
    if (push[a] != 0.0 && (*v)[a] * push[a] > 0.0) { (*v)[a] = 0.0; }
  }
  return xc;
}
/** Moves (*x) one step through the grid velocity (Heun RK3, k1 given). */
fn moveParticle(x: ptr<function, vec3f>, v: ptr<function, vec3f>, k1: vec3f) {
  let x0 = *x;
  let k2 = velU(collide(x0 + (CFL / 3.0) * k1));
  let k3 = velU(collide(x0 + (2.0 * CFL / 3.0) * k2));
  *x = moveEnd(x0, x0 + (CFL / 4.0) * (k1 + 3.0 * k3), v);
}
`;

// ---------------------------------------------------------------------------
// 4. Band seeding (the brick engine's band resampling to particleBandDepth): an open cell whose advected
//    centre phiT lies in the band [-BAND, 0) takes a particle in each liquid octant (octant-centre phiT < 0) that
//    the census found empty, while the cell holds fewer particles than it has liquid octants (eight a full cell:
//    Ferstl 2016 §3.3). A seed sits at its octant's centre, jittered, with the grid velocity there; it is appended
//    to the source pool, so the scan and scatter sort it with the census. cfg.a = parity (src pool).
// ---------------------------------------------------------------------------
const seedKernel = (n: PourInflow) => kernel("pour.seed", [ctrlRW, bricksRO, partsRW, pcellRW, pgridRW, phisRO, velRO, cfgU], /* wgsl */ `
${phiSampler("phiT", "phis", "PHIT_OFF")}
${velocitySampler("velU", "vel", "U_OFF")}
${inflowWGSL(n, false)}
// pUncovered: no particle in the 27 cells about c (the census's counts).
fn uncovered(c: vec3i) -> bool {
  for (var k = 0u; k < 27u; k++) {
    let q = c + vec3i(i32(k % 3u), i32((k / 3u) % 3u), i32(k / 9u)) - vec3i(1);
    if (inBox(q) && (atomicLoad(&pgrid[cidx(q)]) & 0xffffu) != 0u) { return false; }
  }
  return true;
}
// The brick engine's graded seeds (nbSeedTarget, nbSeedSiteIndex, nbfSeedMass): eight unit records, or four of
// mass two at sites 0, 3, 5, 6 in a cell whose centre is 1.5 cells deep or more.
fn seedTarget(d: f32) -> u32 { return select(8u, 4u, d <= -1.5); }
fn seedSite(j: u32, d: f32) -> u32 { if (seedTarget(d) == 8u) { return j; } return array<u32, 4>(0u, 3u, 5u, 6u)[j & 3u]; }
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let b = bricks[B_LPART + wg.x];
  let c = brickCoord(b) * 8 + localCoord(l);
  if (!cellOpen(c)) { return; }
  // pSeedCellMask: the cell gate on the centre's depth (nbfSeedCellDepthOk), then sites j from the cell's record
  // count n0 to the goal, none once the held mass and the new records reach a cell's mass, 8.
  let d = phiTTri(vec3f(c) + vec3f(0.5));
  if (d > 0.9 || d < -BAND - 0.5) { return; }
  let pc = cidx(c);
  let w = atomicLoad(&pgrid[pc]);
  let n0 = w & 0xffffu;
  let held = f32(w >> 16u);
  let goal = seedTarget(d); let mass = f32(8u / goal);
  if (n0 >= goal || held >= 8.0) { return; }
  let step = atomicLoad(&ctrl[C_STEP]);
  var known = false; var uncov = true;
  var mask = 0u;
  for (var j = n0; j < goal; j++) {
    if (held + f32(j - n0) * mass >= 8.0) { continue; }
    let k = seedSite(j, d);
    let site = vec3f(c) + 0.5 * octOffset(k) + 0.5 * vec3f(hashUnit(step, pc, k), hashUnit(step, pc, k + 8u), hashUnit(step, pc, k + 16u));
    let distance = phiTTri(site);
    if (!known && (distance > -1.0 || d > -1.5)) { uncov = n0 == 0u && uncovered(c); known = true; }
    // nbfSeedSiteOk (the reseed, an h owner): a site in this step's drop is exempt from the inner-band rule.
    if (distance > 0.0 || distance < -BAND) { continue; }
    if ((distance > -1.0 || d > -1.5) && inPhi(site, 1.0) >= 0.0 && !uncov) { continue; }
    mask |= 1u << j;
  }
  if (mask == 0u) { return; }
  let ns = countOneBits(mask);
  let k0 = atomicAdd(&ctrl[C_PCOUNT + cfg.a], ns);
  let base = cfg.a * CAP * 2u;
  var r = 0u;
  for (var j = n0; j < goal; j++) {
    if (((mask >> j) & 1u) == 0u) { continue; }
    let at = k0 + r; r++;
    if (at >= CAP) { atomicOr(&ctrl[C_FAIL], FAIL_POOL); atomicAdd(&ctrl[C_OVERFLOW], 1u); continue; }
    let k = seedSite(j, d);
    let site = vec3f(c) + 0.5 * octOffset(k) + 0.5 * vec3f(hashUnit(step, pc, k), hashUnit(step, pc, k + 8u), hashUnit(step, pc, k + 16u));
    parts[base + at] = vec4f(site, mass);
    // nbSeedFirst: velocity.w 1 marks a seed until its first G2P, which takes the grid's (PIC).
    parts[base + CAP + at] = vec4f(velU(site), 1.0);
    pcell[at] = pc;
  }
  atomicAdd(&pgrid[pc], ns + (u32(mass * f32(ns)) << 16u));
  atomicAdd(&ctrl[C_EMITTED], ns);
}
`);

// ---------------------------------------------------------------------------
// 5. Per-brick exclusive scan of cell counts; each brick reserves its range in
//    the destination pool (cfg.a = parity; dst = 1 - parity).
// ---------------------------------------------------------------------------
export const scanKernel = kernel("pour.scan", [ctrlRW, bricksRW, pgridRW, cfgU], /* wgsl */ `
var<workgroup> sA: array<u32, 512>;
var<workgroup> sB: array<u32, 512>;
var<workgroup> sBase: u32;
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let b = atomicLoad(&bricks[B_LPART + wg.x]);
  let c = brickCoord(b) * 8 + localCoord(l);
  let pc = pidx(c);
  let w = atomicLoad(&pgrid[pc]) & 0xffffu;
  sA[l] = w;
  workgroupBarrier();
  // Hillis-Steele inclusive scan, ping-pong.
  var d = 1u; var src = 0u;
  loop {
    if (d >= 512u) { break; }
    if (src == 0u) { sB[l] = sA[l] + select(0u, sA[l - d], l >= d); }
    else { sA[l] = sB[l] + select(0u, sB[l - d], l >= d); }
    workgroupBarrier();
    src = 1u - src; d = d * 2u;
  }
  let incl = select(sA[l], sB[l], src == 1u);
  let total = select(sA[511], sB[511], src == 1u);
  if (l == 0u) {
    let base = atomicAdd(&ctrl[C_PCOUNT + (1u - cfg.a)], total);
    sBase = base;
    atomicStore(&bricks[B_PSTART + b], base); atomicStore(&bricks[B_PNUM + b], total);
  }
  workgroupBarrier();
  atomicStore(&pgrid[PN + pc], sBase + incl - w);
  // Keep the census's octant occupancy (bits 16..23) for seeding; the count restarts for the scatter.
  atomicStore(&pgrid[pc], atomicLoad(&pgrid[pc]) & 0xffff0000u);
}
`);

// ---------------------------------------------------------------------------
// 6. Scatter live particles into the destination pool, sorted by brick and cell.
// ---------------------------------------------------------------------------
export const scatterKernel = kernel("pour.scatter", [ctrlRW, partsRW, pcellRO, pgridRW, cfgU], /* wgsl */ `
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  let n = min(atomicLoad(&ctrl[C_PCOUNT + cfg.a]), CAP);
  if (i >= n) { return; }
  let pc = pcell[i];
  if (pc == DEAD_PARTICLE) { return; }
  let slot = atomicLoad(&pgrid[PN + pc]) + (atomicAdd(&pgrid[pc], 1u) & 0xffffu);
  if (slot >= CAP) { return; }
  let src = cfg.a * CAP * 2u; let dst = (1u - cfg.a) * CAP * 2u;
  parts[dst + slot] = parts[src + i];
  parts[dst + CAP + slot] = parts[src + CAP + i];
}
`);

// ---------------------------------------------------------------------------
// 6b. Particle separation (Sato et al. 2018 §3.2.2, after Ando et al. 2012):
//    two particles closer than the sum of their radii (eight a cell rest half
//    a cell apart, r = h/4) push apart, each by half the overlap, solved as one
//    Jacobi pass with constraint averaging (Macklin et al. 2014 §4.2). Within
//    a cell of the free surface only the tangential part is kept (Khayyer et
//    al. 2017; Lind et al. 2012), so the shift never moves the surface. Without
//    it the particles a film carries pack into its floor cells. Reads the
//    sorted pool's pre-move positions (the 2x2x2 cells around the nearest
//    vertex hold every particle within half a cell along each axis) and writes
//    the shift into the free source pool, which G2P adds after the move.
//    cfg.a = parity (sorted pool = dst).
// ---------------------------------------------------------------------------
export const separateKernel = kernel("pour.separate", [ctrlRO, partsRW, pgridRO, phisRO, cfgU], /* wgsl */ `
${phiSampler("phi", "phis", "PHI_OFF")}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  let n = min(ctrl[C_PCOUNT + 1u - cfg.a], CAP);
  if (i >= n) { return; }
  let base = (1u - cfg.a) * CAP * 2u;
  let x = parts[base + i].xyz;
  let bv = vec3i(floor(x - vec3f(0.5)));
  var d = vec3f(0.0); var m = 0.0;
  // The four x-rows of the 2x2x2 cells: their (start, count) loaded up front (independent loads). Within a
  // brick the pool is sorted by local cell index (x fastest), so a row's two cells are one contiguous range,
  // visited in the same order as cell by cell.
  var rs: array<u32, 4>; var re: array<u32, 4>; var rs1: array<u32, 4>; var re1: array<u32, 4>;
  for (var r = 0u; r < 4u; r++) {
    let c0 = bv + vec3i(0, i32(r & 1u), i32(r >> 1u));
    let c1 = c0 + vec3i(1, 0, 0);
    var s0 = 0u; var e0 = 0u; var s1 = 0u; var e1 = 0u;
    if (inBox(c0)) { let pc = pidx(c0); s0 = pgrid[PN + pc]; e0 = s0 + (pgrid[pc] & 0xffffu); }
    if (inBox(c1)) { let pc = pidx(c1); s1 = pgrid[PN + pc]; e1 = s1 + (pgrid[pc] & 0xffffu); }
    if (e0 > s0 && e1 > s1 && s1 == e0) { e0 = e1; s1 = e1; }
    rs[r] = s0; re[r] = e0; rs1[r] = s1; re1[r] = e1;
  }
  for (var q = 0u; q < 8u; q++) {
    let r = q >> 1u;
    let lo = select(rs[r], rs1[r], (q & 1u) == 1u); let hi = select(re[r], re1[r], (q & 1u) == 1u);
    var j = lo;
    // Two neighbours per iteration (both loads in flight), accumulated in order.
    for (; j + 1u < hi; j += 2u) {
      let ea = x - parts[base + j].xyz; let eb = x - parts[base + j + 1u].xyz;
      let la = dot(ea, ea); let lb = dot(eb, eb);
      if (j != i && la < 0.25 && la > 1e-12) { let len = sqrt(la); d += 0.5 * (0.5 - len) * ea / len; m += 1.0; }
      if (j + 1u != i && lb < 0.25 && lb > 1e-12) { let len = sqrt(lb); d += 0.5 * (0.5 - len) * eb / len; m += 1.0; }
    }
    if (j < hi && j != i) {
      let e = x - parts[base + j].xyz;
      let l2 = dot(e, e);
      if (l2 < 0.25 && l2 > 1e-12) { let len = sqrt(l2); d += 0.5 * (0.5 - len) * e / len; m += 1.0; }
    }
  }
  d = d / max(m, 1.0);
  if (phiTri(x) > -1.0) {
    let gr = vec3f(phiTri(x + vec3f(0.5, 0.0, 0.0)) - phiTri(x - vec3f(0.5, 0.0, 0.0)),
                   phiTri(x + vec3f(0.0, 0.5, 0.0)) - phiTri(x - vec3f(0.0, 0.5, 0.0)),
                   phiTri(x + vec3f(0.0, 0.0, 0.5)) - phiTri(x - vec3f(0.0, 0.0, 0.5)));
    let gl = length(gr);
    if (gl > 1e-6) { let nn = gr / gl; d -= dot(d, nn) * nn; }
  }
  // w: the record's pre-move mass, for G2P's mass rules (they read the run's masses while records are rewritten).
  parts[cfg.a * CAP * 2u + i] = vec4f(d, parts[base + i].w);
}
`);

// ---------------------------------------------------------------------------
// 7. Level-set advection: semi-Lagrangian, one Ralston RK3 backtrace of the
//    whole step (three velocity samples, any length; as the reference's
//    surfaceTraceSubsteps "off"), monotone Catmull-Rom tricubic, then the union
//    with the nozzle's plug (the brick engine's nbSourcePhi after advection). phi -> phiT.
// ---------------------------------------------------------------------------
const advectPhiKernel = (n: PourInflow) => kernel("pour.advectPhi", [ctrlRW, bricksRO, phisRW, velRO], /* wgsl */ `
${phiSampler("phi", "phis", "PHI_OFF")}
${velocitySampler("velU", "vel", "U_OFF")}
${inflowWGSL(n, false)}
fn heldF(x: vec3f) -> vec3f { return clamp(x, vec3f(0.0), vec3f(f32(NX - 1), f32(NY - 1), f32(NZ - 1))); }
fn heldI(v: vec3i) -> vec3i { return clamp(v, vec3i(0), vec3i(NX - 1, NY - 1, NZ - 1)); }
// umVertexBuried (the brick engine's psBuried): none of the vertex's eight cells is open.
fn buried(w: vec3i) -> bool {
  for (var k = 0u; k < 8u; k++) { if (cellOpen(w - vec3i(i32(k & 1u), i32((k >> 1u) & 1u), i32(k >> 2u)))) { return false; } }
  return true;
}
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let b = bricks[B_LSURF + wg.x];
  let v = brickCoord(b) * 8 + localCoord(l);
  let p = vec3f(v);
  let c = phiAt(v);
  // A buried vertex (inside the glass) keeps its value: no trace, no source (lAdvect's lVertexIn(v) && !psBuried(v)).
  // Advected, the wall's interior took the bore's liquid and passed it to the outer face.
  if (buried(v)) { phis[PHIT_OFF + ((b << 9u) | l)] = c; return; }
  // The brick engine's lAdvected. The foot (umVertexFoot): within two cells of the surface, the vertex's closest
  // point on the zero set (central-difference gradient, the offset capped at two cells) is traced and the vertex
  // carried rigidly with it, so the surface moves with the velocity at the surface, not the extension's in air.
  var foot = p; var shift = vec3f(0.0); var smoothing = 0.0;
  if (abs(c) < 2.0) {
    var t: array<f32, 10>;
    var gr = vec3f(0.0);
    for (var a = 0u; a < 3u; a++) {
      let e = unitAxis(a);
      let lo = heldI(v - e); let hi = heldI(v + e);
      t[2u * a] = phiAt(lo); t[2u * a + 1u] = phiAt(hi);
      gr[a] = (t[2u * a + 1u] - t[2u * a]) / max(f32(hi[a] - lo[a]), 1e-6);
    }
    for (var k = 0u; k < 4u; k++) {
      var q = v; q[select(0u, 2u, k > 1u)] += select(-2, 2, (k & 1u) != 0u);
      t[6u + k] = phiAt(heldI(q));
    }
    let g2 = dot(gr, gr);
    if (g2 >= 1e-6) {
      var offset = -c * gr / g2; let reach = length(offset);
      if (reach > 2.0) { offset *= 2.0 / reach; }
      foot = heldF(p + offset); shift = p - foot;
    }
    // umSurfaceSmoothing: the x and z fourth differences of a stencil holding the interface (eps = dt / 2).
    let eps = min(0.5 * DT_S, 0.03125);
    let neg = c < 0.0;
    let mixed = (t[0] < 0.0) != neg || (t[1] < 0.0) != neg || (t[4] < 0.0) != neg || (t[5] < 0.0) != neg
      || (t[6] < 0.0) != neg || (t[7] < 0.0) != neg || (t[8] < 0.0) != neg || (t[9] < 0.0) != neg;
    if (mixed) {
      var sum = 0.0; var crossing = false;
      for (var a = 0u; a < 3u; a += 2u) {
        if (v[a] < 2 || v[a] > i32(select(NZ, NX, a == 0u)) - 2) { continue; }
        // psSmoothOpen: no buried vertex on the axis's stencil.
        let e = unitAxis(a);
        var part = 6.0 * c; var valid = !buried(v - e) && !buried(v + e) && !buried(v - 2 * e) && !buried(v + 2 * e); var different = false;
        for (var k = 0u; k < 4u; k++) {
          let s = t[select(2u * a + k, 6u + a + k - 2u, k > 1u)]; let span = select(1.0, 2.0, k > 1u);
          if (abs(s - c) > 1.5 * span) { valid = false; }
          different = different || ((s < 0.0) != neg);
          part += select(-4.0, 1.0, k > 1u) * s;
        }
        if (valid) { sum += part; crossing = crossing || different; }
      }
      smoothing = select(0.0, -eps * sum, crossing);
    }
  }
  var traced = foot;
  let u = velU(foot);
  if (any(u != vec3f(0.0))) {
    let k2 = velU(foot - 0.5 * CFL * u);
    let k3 = velU(foot - 0.75 * CFL * k2);
    traced = heldF(foot - CFL * (2.0 * u + 3.0 * k2 + 4.0 * k3) / 9.0);
  }
  // The brick engine's solid trace (phiSolid): the departure stops at the last open point before the glass.
  if (nearGlass(foot, length(traced - foot))) { traced = walkOpen(foot, traced); }
  var q = traced;
  // The rigid offset is not walked (lAdvected's fgHeld(traced + shift)): walking it would carry an outside vertex's
  // sample through the glass to the liquid's surface inside.
  if (any(shift != vec3f(0.0))) { q = heldF(traced + shift); }
  // The interpolant: trilinear (umCubicAt takes the cubic only in a cold tile, and the pour's band is hot).
  var value = clamp(phiTri(q), -PHI_MAX, PHI_MAX);
  let d = abs(q - p);
  value = inPhi(p, value) + smoothing * clamp(2.0 - 2.0 * max(d.x, max(d.y, d.z)), 0.0, 1.0);
  phis[PHIT_OFF + ((b << 9u) | l)] = clamp(value, -PHI_MAX, PHI_MAX);
}
`);

// ---------------------------------------------------------------------------
// 8a. The splat (the brick engine's lSplat): each vertex within two cells of a sample keeps the nearest sample's
//    squared distance, as a key whose atomicMax keeps the smallest (positive float bits, complemented; zero is
//    empty). A workgroup a brick, its sorted samples into an 11^3 vertex tile (o - 1 .. o + 9), flushed by
//    atomicMax (neighbouring bricks' tiles overlap). cfg.a = parity (the sorted pool is dst).
// ---------------------------------------------------------------------------
export const splatKernel = kernel("pour.splat", [bricksRO, partsRO, keysRW, cfgU], /* wgsl */ `
var<workgroup> wKey: array<atomic<u32>, 1331>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let b = bricks[B_LPART + wg.x];
  let count = bricks[B_PNUM + b];
  if (count == 0u) { return; }
  let start = bricks[B_PSTART + b];
  let lo = brickCoord(b) * 8 - vec3i(1);
  for (var j = l; j < 1331u; j += 256u) { atomicStore(&wKey[j], 0u); }
  workgroupBarrier();
  let dst = (1u - cfg.a) * CAP * 2u;
  for (var j = l; j < count; j += 256u) {
    let q = parts[dst + start + j].xyz;
    let c = vec3i(floor(q));
    for (var k = 0u; k < 64u; k++) {
      let v = c - vec3i(1) + vec3i(i32(k & 3u), i32((k >> 2u) & 3u), i32(k >> 4u));
      let d = q - vec3f(v); let d2 = dot(d, d);
      if (!(d2 < 4.0)) { continue; }
      let w = v - lo;
      if (any(w < vec3i(0)) || any(w >= vec3i(11))) { continue; }
      atomicMax(&wKey[u32(w.x + 11 * (w.y + 11 * w.z))], ~bitcast<u32>(d2));
    }
  }
  workgroupBarrier();
  for (var j = l; j < 1331u; j += 256u) {
    let key = atomicLoad(&wKey[j]);
    if (key == 0u) { continue; }
    let v = lo + vec3i(i32(j % 11u), i32((j / 11u) % 11u), i32(j / 121u));
    if (any(v < vec3i(0)) || any(v >= vec3i(NX, NY, NZ))) { continue; }
    // Only into an active brick: combine clears the keys of active bricks alone.
    if ((bricks[B_FLAGS + brickOfCell(v)] & 1u) == 0u) { continue; }
    atomicMax(&keys[cidx(v)], key);
  }
}
`);

// ---------------------------------------------------------------------------
// 8. The band's surface from the particles: the brick engine's union (lUnion, levelset-formulas nbfUnionParticle
//    and nbfUnionBound). Within RCOMB of the surface, the nearest sample bounds the surface between its own sphere
//    (radius 0.875) and the plane of a seeded shell (0.8 down); the advected phi is clamped into that range, moved
//    no further than one cell toward air. Where that adds liquid, the samples add no more than they are: the eight
//    open cells about the vertex (n = min(count, 8)) bound phi no lower than the plane their fill implies (at a
//    wall, along the axis whose two layers differ most). No sample within two cells: the advected phi stands.
//    Every band particle is hot in the pour (the stream and its pool trigger every tile they touch), so theta = 1
//    and erasure is allowed. The nozzle's plug is unioned last (nbSourcePhi).
// ---------------------------------------------------------------------------
const combineKernel = (n: PourInflow) => kernel("pour.combine", [bricksRO, phisRW, pgridRO, keysPlainRW], /* wgsl */ `
${inflowWGSL(n, false)}
const SURFACE_RADIUS: f32 = 0.875;
const SEED_DEPTH: f32 = 0.8003905296791061;
fn unionParticle(bulk: f32, nearest2: f32) -> f32 {
  let sphere = sqrt(nearest2) - SURFACE_RADIUS; let shell = sqrt(max(nearest2 - 0.125, 0.0)) - SEED_DEPTH;
  var excess = bulk - clamp(bulk, min(sphere, shell), max(sphere, shell));
  if (nearest2 <= 0.25) { excess = max(bulk - sphere, 0.0); }
  let ripple = SURFACE_RADIUS - SEED_DEPTH;
  return min(bulk + 1.0, bulk - excess * smoothstep(0.5 * ripple, ripple, abs(excess)));
}
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let b = bricks[B_LPART + wg.x];
  let i = (b << 9u) | l;
  let key = keys[i];
  if (key != 0u) { keys[i] = 0u; }
  let bulk = phis[PHIT_OFF + i];
  if (bulk < -RCOMB || key == 0u) { return; }
  let nearest2 = bitcast<f32>(~key);
  let v = brickCoord(b) * 8 + localCoord(l);
  var particle = unionParticle(bulk, nearest2);
  if (particle < bulk) {
    var held = 0u; var open = 0u;
    var heldLo = vec3u(0u); var heldHi = vec3u(0u); var openLo = vec3u(0u); var openHi = vec3u(0u);
    for (var k = 0u; k < 8u; k++) {
      let bit = vec3u(k & 1u, (k >> 1u) & 1u, k >> 2u);
      let c = v - vec3i(1) + vec3i(bit);
      if (!cellOpen(c)) { continue; }
      let n = min(pgrid[pidx(c)] & 0xffffu, 8u);
      open += 1u; held += n;
      let hi = bit != vec3u(0u);
      heldHi += select(vec3u(0u), vec3u(n), hi); openHi += select(vec3u(0u), vec3u(1u), hi);
      heldLo += select(vec3u(n), vec3u(0u), hi); openLo += select(vec3u(1u), vec3u(0u), hi);
    }
    if (open == 8u) { particle = max(particle, min(bulk, 1.0 - f32(held) / (4.0 * f32(open)))); }
    else if (open > 0u) {
      var gap = -1.0; var bound = 1.0;
      for (var a = 0u; a < 3u; a++) {
        let lo = select(1.0, f32(heldLo[a]) / (8.0 * f32(openLo[a])), openLo[a] > 0u);
        let hi = select(1.0, f32(heldHi[a]) / (8.0 * f32(openHi[a])), openHi[a] > 0u);
        if (abs(lo - hi) <= gap) { continue; }
        gap = abs(lo - hi);
        let loDense = lo >= hi;
        let denseOpen = select(openHi[a], openLo[a], loDense) > 0u;
        let sparseOpen = select(openLo[a], openHi[a], loDense) > 0u;
        let sparse = select(0.0, select(lo, hi, loDense), sparseOpen);
        bound = select(-sparse, 1.0 - max(lo, hi) - sparse, denseOpen);
      }
      particle = max(particle, min(bulk, bound));
    }
  }
  phis[PHIT_OFF + i] = clamp(inPhi(vec3f(v), particle), -PHI_MAX, PHI_MAX);
}
`);

// ---------------------------------------------------------------------------
// 8b. Mass fit of the chamber surface (the mass-conserving level set of van
//    der Pijl et al. 2005, with the particles' volume fraction F = count / 8
//    in place of an advected VOF field; cf. CLSVOF, Sussman & Puckett 2000):
//    Jacobi passes move each surface-band vertex (|phi| < 1.5: the plane's
//    empty-side value 1 included) by -GAIN times the mean mismatch F - fill(phi)
//    of its open cells, so the cells' tetrahedral fill matches the particle
//    mass they hold, including in features thinner than the 2x2x2 window of
//    the plane (the 2h jet). A crowded cell's excess over full is first carried
//    up its column (no fill can hold it in place). Interior and far-air
//    vertices stay fixed (no bubbles, no spray halos). GAIN = 1.5
//    closes a plane film's mismatch in one pass (a floor vertex moves the
//    whole fill, a vertex above it half). Solid-only vertices are left alone.
//    cfg.a = source offset, cfg.b = destination offset; an even number of passes
//    ping-pongs phiT -> phi -> ... -> phiT, which redistance then reads. cfg.c = 1
//    on the last pass: the union with the nozzle's plug (the brick engine's
//    nbSourcePhi after the particle union).
// ---------------------------------------------------------------------------
const massFitKernel = (n: PourInflow) => kernel("pour.massFit", [bricksRO, phisRW, pgridRO, cfgU], /* wgsl */ `
${levelSetFillWGSL}
${inflowWGSL(n, false)}
fn fin(v: vec3i, x: f32) -> f32 { return select(x, clamp(inPhi(vec3f(v), x), -PHI_MAX, PHI_MAX), cfg.c == 1u); }
const GAIN: f32 = 1.5;
var<workgroup> tv: array<f32, 1000>;
var<workgroup> tm: array<f32, 729>;
var<workgroup> sBusy: u32;
var<workgroup> sCut: atomic<u32>;
var<workgroup> sCutU: u32;
var<workgroup> sCutList: array<u32, 729>;
fn srcPhi(v: vec3i) -> f32 { return phis[cfg.a + cidx(clampV(v))]; }
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let b = bricks[B_LPART + wg.x];
  let ob = brickCoord(b);
  let o = ob * 8;
  // No particle in the brick or its 26 neighbours: every cell the brick's vertices touch has
  // F = 0 and (the plane being >= 0 there) fill 0, so nothing moves; copy through.
  if (l < 27u) {
    let n = ob + vec3i(i32(l % 3u), i32((l / 3u) % 3u), i32(l / 9u)) - vec3i(1);
    if (brickInBox(n) && bricks[B_PCOUNT + brickIdOf(n)] > 0u) { sBusy = 1u; }
  }
  // The tile's particle counts, loaded at once alongside the busy test (count / 8, or -1 for a closed
  // cell); the column carry below then reads workgroup memory.
  for (var t = l; t < 729u; t += 512u) {
    let c = o - vec3i(1) + vec3i(i32(t % 9u), i32((t / 9u) % 9u), i32(t / 81u));
    var r = -1.0;
    if (cellOpen(c)) { r = f32(pgrid[pidx(c)] & 0xffffu) / 8.0; }
    tm[t] = r;
  }
  if (workgroupUniformLoad(&sBusy) == 0u) {
    phis[cfg.b + ((b << 9u) | l)] = fin(o + localCoord(l), srcPhi(o + localCoord(l)));
    return;
  }
  for (var t = l; t < 1000u; t += 512u) {
    tv[t] = srcPhi(o - vec3i(1) + vec3i(i32(t % 10u), i32((t / 10u) % 10u), i32(t / 100u)));
  }
  // Target volume fractions: F = count / 8, a cell's excess over full carried up its column
  // (liquid settles from the dense side, the bound's own rule), so every target is one a cell
  // can hold; the carry out of the tile's top row stays in it (only liquid deeper than a brick).
  if (l < 81u) {
    var carry = 0.0;
    for (var qy = 0u; qy < 9u; qy++) {
      let t = l % 9u + 9u * qy + 81u * (l / 9u);
      let r = tm[t];
      var F = 0.0;
      if (r >= 0.0) { F = r + carry; carry = max(F - 1.0, 0.0); if (qy < 8u) { F = min(F, 1.0); } }
      tm[t] = F;
    }
  }
  workgroupBarrier();
  // Mismatch F - fill(phi) of each open cell. Only cells with a band corner (|phi| < 1.5) are read
  // below; a cell with no negative corner fills 0, one with only negative corners 1 (as every
  // tetrahedron of it does; a tetrahedron of zeros counts as empty); the cut cells,
  // a few per tile, are compacted so whole SIMD groups run the tetrahedral fill together.
  for (var t = l; t < 729u; t += 512u) {
    let q = vec3i(i32(t % 9u), i32((t / 9u) % 9u), i32(t / 81u));
    let c = o - vec3i(1) + q;
    var m = 0.0;
    if (cellOpen(c)) {
      var lo = PHI_MAX + 1.0; var hi = -PHI_MAX - 1.0; var band = false;
      for (var k = 0u; k < 8u; k++) {
        let r = q + vec3i(i32(k & 1u), i32((k >> 1u) & 1u), i32(k >> 2u));
        let x = tv[u32(r.x + 10 * (r.y + 10 * r.z))];
        lo = min(lo, x); hi = max(hi, x); band = band || abs(x) < 1.5;
      }
      if (band) {
        if (lo >= 0.0) { m = tm[t]; } else if (hi < 0.0) { m = tm[t] - 1.0; } else { m = tm[t]; sCutList[atomicAdd(&sCut, 1u)] = t; }
      }
    }
    tm[t] = m;
  }
  workgroupBarrier();
  if (l == 0u) { sCutU = atomicLoad(&sCut); }
  let nCut = workgroupUniformLoad(&sCutU);
  for (var j = l; j < nCut; j += 512u) {
    let t = sCutList[j];
    let q = vec3i(i32(t % 9u), i32((t / 9u) % 9u), i32(t / 81u));
    var v: array<f32, 8>;
    for (var k = 0u; k < 8u; k++) {
      let r = q + vec3i(i32(k & 1u), i32((k >> 1u) & 1u), i32(k >> 2u));
      v[k] = tv[u32(r.x + 10 * (r.y + 10 * r.z))];
    }
    tm[t] = tm[t] - fill(v);
  }
  workgroupBarrier();
  let lc = localCoord(l);
  let vtx = o + lc;
  var phi = tv[u32((lc.x + 1) + 10 * ((lc.y + 1) + 10 * (lc.z + 1)))];
  if (abs(phi) < 1.5) {
    var sum = 0.0; var open = 0u;
    for (var k = 0u; k < 8u; k++) {
      let d = vec3i(i32(k & 1u), i32((k >> 1u) & 1u), i32(k >> 2u));
      let c = vtx - vec3i(1) + d;
      if (!cellOpen(c)) { continue; }
      open += 1u;
      sum += tm[u32((lc.x + d.x) + 9 * ((lc.y + d.y) + 9 * (lc.z + d.z)))];
    }
    if (open > 0u) { phi = clamp(phi - GAIN * sum / f32(open), -PHI_MAX, PHI_MAX); }
  }
  phis[cfg.b + ((b << 9u) | l)] = fin(vtx, phi);
}
`);

// ---------------------------------------------------------------------------
// 9. Redistance, first layer: the corners of every cell the interface cuts
//    (an open vertex with an opposite-signed open vertex among its 26
//    neighbours) keep their values and stay frozen through the sweeps, as the
//    nodes of cut cells are fixed when a fast-marching/sweeping redistance is
//    initialised (Adalsteinsson & Sethian 1995; Osher & Fedkiw 2003 §7.4).
//    The cut cells' volumes (the tetrahedral fill) are then exactly those the
//    combine left: re-initialising the cut corners from axis crossings moved
//    the surface of thin films by 0.2% of their volume a step. All others
//    start at +-PHI_MAX. phiT -> phi.
//    Only open vertices (corners of an open cell) carry the free surface: a
//    solid wall is not an interface. Solid-interior vertices restart unknown
//    and the sweeps extend the signed distance into them (level-set
//    extrapolation into solids, Bridson 2015 §6.1 / Enright 2002), so the
//    cubic stencils near walls read a smooth continuation.
// ---------------------------------------------------------------------------
const VERTEX_OPEN_WGSL = /* wgsl */ `
fn vertexOpen(v: vec3i) -> bool {
  for (var k = 0u; k < 8u; k++) { if (cellOpen(v - vec3i(i32(k & 1u), i32((k >> 1u) & 1u), i32(k >> 2u)))) { return true; } }
  return false;
}
`;
/** Whether tile entry ti (code own != 0) has a 26-neighbour coded open with the other sign (0 = not open,
 *  1 = open, phi >= 0, 2 = open, phi < 0): the cut-corner test, at constant tile offsets. */
const cutTest = (tile: string) => /* wgsl */ `
fn cutAt(ti: u32, own: u32) -> bool {
  var cut = false;
  for (var dz = -100; dz <= 100; dz += 100) { for (var dy = -10; dy <= 10; dy += 10) { for (var dx = -1; dx <= 1; dx++) {
    let sg = ${tile}[u32(i32(ti) + dx + dy + dz)];
    cut = cut || (sg != 0u && sg != own);
  } } }
  return cut;
}
`;
export const redistInitKernel = kernel("pour.redistInit", [bricksRO, phisRW], /* wgsl */ `
${VERTEX_OPEN_WGSL}
// The brick's vertices and a one-vertex halo, staged once: 0 = not an open vertex in the box
// (ignored), 1 = phiT >= 0, 2 = phiT < 0; the 26-neighbour sign test then reads the tile.
var<workgroup> tsgn: array<u32, 1000>;
${cutTest("tsgn")}
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let b = bricks[B_LSURF + wg.x];
  let o = brickCoord(b) * 8;
  for (var t = l; t < 1000u; t += 512u) {
    let w = o + tileCoord(t);
    var sg = 0u;
    if (inBox(w) && vertexOpen(w)) { sg = select(1u, 2u, phis[PHIT_OFF + cidx(w)] < 0.0); }
    tsgn[t] = sg;
  }
  workgroupBarrier();
  let lc = localCoord(l);
  let ti = tileIndex(lc);
  if (tsgn[ti] == 0u) { phis[PHI_OFF + ((b << 9u) | l)] = UNKNOWN_PHI; return; }
  let a = phis[PHIT_OFF + ((b << 9u) | l)];
  let own = tsgn[ti];
  let cut = cutAt(ti, own);
  let out = select(select(PHI_MAX, -PHI_MAX, own == 2u), a, cut);
  phis[PHI_OFF + ((b << 9u) | l)] = out;
}
`);

// ---------------------------------------------------------------------------
// 10. Redistance sweeps: Godunov upwind Eikonal update (|grad phi| = 1),
//     8 Jacobi sweeps on a staged brick + halo, cut-cell corners frozen.
//     Monotone in |phi|. cfg.a = input offset, cfg.b = output offset,
//     cfg.c = 1 on the final launch: also publish the render surface.
// ---------------------------------------------------------------------------
export const redistSweepKernel = kernel("pour.redistSweep", [bricksRO, phisRW, cfgU, surfW], /* wgsl */ `
${SURF_WGSL}
${VERTEX_OPEN_WGSL}
var<workgroup> tA: array<f32, 1000>;
var<workgroup> tB: array<f32, 1000>;
var<workgroup> tOpen: array<u32, 1000>;
${cutTest("tOpen")}
var<workgroup> sMixed: atomic<u32>;
var<workgroup> sMixedU: u32;
fn eik(a0: f32, b0: f32, c0: f32) -> f32 {
  var a = min(a0, min(b0, c0)); var c = max(a0, max(b0, c0)); var b = a0 + b0 + c0 - a - c;
  var d = a + 1.0;
  if (d > b) {
    d = 0.5 * (a + b + sqrt(max(0.0, 2.0 - (a - b) * (a - b))));
    if (d > c) {
      let s = a + b + c;
      d = (s + sqrt(max(0.0, s * s - 3.0 * (a * a + b * b + c * c - 1.0)))) / 3.0;
    }
  }
  return d;
}
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let b = bricks[B_LSURF + wg.x];
  let o = brickCoord(b) * 8;
  if (l == 0u) { atomicStore(&sMixed, 0u); }
  workgroupBarrier();
  var notAir = false; var notLiquid = false;
  for (var t = l; t < 1000u; t += 512u) {
    let q = clampV(o + tileCoord(t));
    let val = phis[cfg.a + cidx(q)];
    tA[t] = val; tB[t] = val; tOpen[t] = select(0u, select(1u, 2u, val < 0.0), vertexOpen(q));
    notAir = notAir || val != PHI_MAX;
    notLiquid = notLiquid || val != -PHI_MAX;
  }
  if (notAir) { atomicOr(&sMixed, 1u); }
  if (notLiquid) { atomicOr(&sMixed, 2u); }
  workgroupBarrier();
  if (l == 0u) { sMixedU = atomicLoad(&sMixed); }
  let lc = localCoord(l);
  let ti = tileIndex(lc);
  // A tile that is all air, or all liquid, at the band limit is a fixed point of the sweeps (no
  // vertex is frozen, and every update is sgn * min(PHI_MAX, eik(PHI_MAX, PHI_MAX, PHI_MAX) > PHI_MAX)).
  let mixedU = workgroupUniformLoad(&sMixedU);
  if (mixedU != 3u) {
    let limit = select(-PHI_MAX, PHI_MAX, mixedU == 2u);
    phis[cfg.b + ((b << 9u) | l)] = limit;
    if (cfg.c == 1u) { publishSurf(o + lc, limit); }
    return;
  }
  let me = tA[ti];
  let neg = me < 0.0;
  // tOpen codes open vertices by sign (as redistInit's tile): frozen = open with an opposite-signed
  // open neighbour.
  let open = tOpen[ti] != 0u;
  let frozen = open && cutAt(ti, tOpen[ti]);
  // Every vertex keeps its sign. A buried vertex (redistInit's UNKNOWN_PHI) is air, as the brick engine's corners past
  // the wall ("an air marker", lClear): given the nearest neighbour's sign, the glass's interior took the bore's
  // liquid and the foot's rigid shift carried it to the outer face.
  let sgn = select(1.0, -1.0, neg);
  for (var s = 0u; s < 8u; s++) {
    if (!frozen) {
      if ((s & 1u) == 0u) {
        let ax = min(abs(tA[ti - 1u]), abs(tA[ti + 1u]));
        let ay = min(abs(tA[ti - 10u]), abs(tA[ti + 10u]));
        let az = min(abs(tA[ti - 100u]), abs(tA[ti + 100u]));
        tB[ti] = sgn * min(abs(tA[ti]), eik(ax, ay, az));
      } else {
        let ax = min(abs(tB[ti - 1u]), abs(tB[ti + 1u]));
        let ay = min(abs(tB[ti - 10u]), abs(tB[ti + 10u]));
        let az = min(abs(tB[ti - 100u]), abs(tB[ti + 100u]));
        tA[ti] = sgn * min(abs(tB[ti]), eik(ax, ay, az));
      }
    }
    workgroupBarrier();
  }
  phis[cfg.b + ((b << 9u) | l)] = tA[ti];
  if (cfg.c == 1u) { publishSurf(o + lc, tA[ti]); }
}
`);

/** The brick engine's support rules (fluid-grid fgLiquidCell, particles-kernels pSupported, velocity-kernels
 *  pDeposit): a liquid cell is open with its centre phi (the mean of its 8 vertices) under -0.001 of a cell; a
 *  record is supported when one of the 8 cells about floor(x - 0.5) of trilinear weight over 1e-6 is liquid. P2G and
 *  G2P read the same level set (this step's, after redistance), so they agree on every record. */
const SUPPORT_WGSL = /* wgsl */ `
fn liquidCell(c: vec3i) -> bool {
  if (!cellOpen(c)) { return false; }
  var sum = 0.0;
  for (var k = 0u; k < 8u; k++) { sum += phis[PHI_OFF + cidx(clampV(c + vec3i(i32(k & 1u), i32((k >> 1u) & 1u), i32(k >> 2u))))]; }
  return 0.125 * sum < -0.001;
}
fn gridSupported(x: vec3f) -> bool {
  let g = x - vec3f(0.5); let b = vec3i(floor(g)); let f = g - vec3f(b);
  for (var k = 0u; k < 8u; k++) {
    let bit = vec3u(k & 1u, (k >> 1u) & 1u, k >> 2u);
    let w = select(vec3f(1.0) - f, f, bit != vec3u(0u));
    if (w.x * w.y * w.z > 1e-6 && liquidCell(b + vec3i(bit))) { return true; }
  }
  return false;
}
`;
// ---------------------------------------------------------------------------
// 11. P2G (Ferstl 2016 §3.1): quadratic B-spline splat of the sorted band
//     particles of one brick into workgroup fixed-point accumulators, then one
//     global atomic per touched face. cfg.a = parity (sorted pool = dst).
// ---------------------------------------------------------------------------
export const p2gKernel = kernel("pour.p2g", [bricksRO, partsRO, accRW, cfgU, phisRO], /* wgsl */ `
${SUPPORT_WGSL}
var<workgroup> sMom: array<atomic<i32>, 3300>;
// The liquid cells o - 1 .. o + 8 about the brick (pdLiquidAt's window), a bit each.
var<workgroup> sLiq: array<atomic<u32>, 32>;
var<workgroup> sW: array<atomic<i32>, 3300>;
fn regionDims(a: u32) -> vec3i { return vec3i(10) + unitAxis(a); }
fn regionIndex(a: u32, q: vec3i) -> u32 {
  let d = regionDims(a); let r = q + vec3i(1);
  return a * 1100u + u32(r.x + d.x * (r.y + d.y * r.z));
}
fn bspline(f: f32) -> vec3f { return vec3f(0.5 * (1.5 - f) * (1.5 - f), 0.75 - (f - 1.0) * (f - 1.0), 0.5 * (f - 0.5) * (f - 0.5)); }
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let b = bricks[B_LPART + wg.x];
  let num = bricks[B_PNUM + b];
  let start = bricks[B_PSTART + b];
  let o = brickCoord(b) * 8;
  for (var t = l; t < 3300u; t += 256u) { atomicStore(&sMom[t], 0); atomicStore(&sW[t], 0); }
  if (l < 32u) { atomicStore(&sLiq[l], 0u); }
  workgroupBarrier();
  for (var j = l; j < 1000u; j += 256u) {
    if (liquidCell(o - vec3i(1) + vec3i(i32(j % 10u), i32((j / 10u) % 10u), i32(j / 100u)))) { atomicOr(&sLiq[j >> 5u], 1u << (j & 31u)); }
  }
  workgroupBarrier();
  let base = (1u - cfg.a) * CAP * 2u;
  // Lane l of SIMD group l / 32 takes every eighth particle: lanes of one group sit in different cells, so
  // their shared atomics (one stencil per cell and half) seldom hit one address together.
  let tl = (l & 31u) * 8u + (l >> 5u);
  for (var k = start + tl; k < start + num; k += 256u) {
    let rp = parts[base + k]; let xp = rp.xyz; let vp = parts[base + CAP + k].xyz; let mp = rp.w;
    // pDeposit: only supported records deposit (an unsupported one is ballistic at G2P and sees no grid).
    var held = false;
    let g = xp - vec3f(0.5); let bq = vec3i(floor(g)); let fq = g - vec3f(bq);
    for (var kk = 0u; kk < 8u; kk++) {
      let bit = vec3u(kk & 1u, (kk >> 1u) & 1u, kk >> 2u);
      let wq = select(vec3f(1.0) - fq, fq, bit != vec3u(0u));
      let r = bq + vec3i(bit) - o + vec3i(1);
      if (wq.x * wq.y * wq.z > 1e-6 && all(r >= vec3i(0)) && all(r < vec3i(10))) {
        let ri = u32(r.x + 10 * (r.y + 10 * r.z));
        held = held || ((atomicLoad(&sLiq[ri >> 5u]) >> (ri & 31u)) & 1u) != 0u;
      }
    }
    if (!held) { continue; }
    for (var a = 0u; a < 3u; a++) {
      let q = xp - vec3f(0.5) + 0.5 * vec3f(unitAxis(a));
      let bi = vec3i(floor(q - vec3f(0.5)));
      let fr = q - vec3f(bi);
      let wx = bspline(fr.x); let wy = bspline(fr.y); let wz = bspline(fr.z);
      for (var dz = 0; dz < 3; dz++) { for (var dy = 0; dy < 3; dy++) { for (var dx = 0; dx < 3; dx++) {
        let w = wx[dx] * wy[dy] * wz[dz] * mp;
        let r = regionIndex(a, bi + vec3i(dx, dy, dz) - o);
        atomicAdd(&sMom[r], i32(round(w * vp[a] * SM)));
        atomicAdd(&sW[r], i32(round(w * SW)));
      } } }
    }
  }
  workgroupBarrier();
  for (var t = l; t < 3300u; t += 256u) {
    let wsum = atomicLoad(&sW[t]);
    let msum = atomicLoad(&sMom[t]);
    if (wsum == 0 && msum == 0) { continue; }
    let a = t / 1100u; let r = t - a * 1100u;
    let d = regionDims(a);
    let q = vec3i(i32(r) % d.x, (i32(r) / d.x) % d.y, i32(r) / (d.x * d.y)) - vec3i(1) + o;
    if (!inBox(q)) { continue; }
    // Faces of bricks outside the active set are never read or reset; their share is dropped (far from liquid).
    if ((bricks[B_FLAGS + brickOfCell(q)] & 1u) == 0u) { continue; }
    let pc = pidx(q);
    atomicAdd(&acc[a * PN + pc], msum);
    atomicAdd(&acc[(3u + a) * PN + pc], wsum);
  }
}
`);

// ---------------------------------------------------------------------------
// 12. Momentum: semi-Lagrangian advection of the MAC velocity (RK2 midpoint
//     backtrace in <= 1-cell segments, trilinear), Ferstl 2016 Eq. (3) particle
//     velocity where available within r = 2h of the surface, gravity on every
//     y face. Also publishes the cell-centre level set, the fill fraction and
//     the brick's minimum phi for the next classification.
// ---------------------------------------------------------------------------
/** The fine face backtrace and its gravity, shared by velocityBulk (interface faces) and velocityTrace. */
const FACE_SL_WGSL = /* wgsl */ `
/** Semi-Lagrangian RK2 backtrace (segments of at most one cell) of the fine face (a, c) through the fine velocity
 *  (velU: velocitySampler over U); slSegs keeps the unclamped segment count. RK2 midpoint segments, one velocity
 *  evaluation site: even half-steps locate the midpoint from k1, odd ones advance with k2. */
var<private> slSegs: f32;
fn fineFaceSL(a: u32, c: vec3i) -> f32 {
  let pos = vec3f(c) + vec3f(0.5) - 0.5 * vec3f(unitAxis(a));
  let u0 = velU(pos);
  slSegs = ceil(length(u0) * CFL);
  let n = clamp(slSegs, 1.0, MAXSEG);
  let tau = CFL / n;
  var x = pos; var q = pos; var kk = u0;
  let last = u32(n) * 2u - 1u;
  for (var j = 0u; j <= last; j++) {
    if ((j & 1u) == 0u) { q = x - 0.5 * tau * kk; } else { x = x - tau * kk; q = x; }
    if (j == last) { break; }
    kk = velU(q);
  }
  return velUComp(a, x);
}
/** Gravity on face (a, c): -g dt (y_hi - y_lo) / d between the centres of its two pressure owners, d their distance along a
 *  (1 between fine cells, 2.5 across the h/4h interface). Equals -g dt on vertical faces and 0 on horizontal fine faces; on a
 *  horizontal interface face it is the hydrostatic difference the coarse/fine centre offset implies, so the resting reservoir
 *  stays exactly at rest (a well-balanced body force, cf. Audusse et al. 2004). */
fn gravityOn(a: u32, c: vec3i) -> f32 {
  let lo = c - unitAxis(a);
  let d = select(1.0, 2.5, cellBulk(lo) != cellBulk(c));
  return -GDT * (ownerY(c) - ownerY(lo)) / d;
}
`;
/** The brick engine's volume gauge cell rule (nbfCellFill) and its rate (nbfVolumeRate), verbatim. */
const VOLUME_FILL_WGSL = nbfVolumeWGSL.slice(0, nbfVolumeWGSL.indexOf("// filled, budget"));
const VOLUME_RATE_WGSL = `const V_CONTROL: f32 = ${1 / VOLUME_CONTROL_S};\n` + nbfVolumeWGSL.slice(nbfVolumeWGSL.indexOf("// filled, budget"));
const velocityKernel = (n: PourInflow) => kernel("pour.velocity", [ctrlRW, bricksRW, phisRW, velRW, accRW, volW, bitsRW], /* wgsl */ `
${inflowWGSL(n, true)}
${levelSetFillWGSL}
${geometricPlaneBoxWGSL}
${VOLUME_FILL_WGSL}
${velocitySampler("velU", "vel", "U_OFF")}
${bulkWGSL(true)}
${FACE_SL_WGSL}
// vGauge: the step's filled volume (nbfCellFill over the open cells; a DEEP brick's open cells are whole), a
// workgroup sum in 2^-20 cells and one carried atomic pair a brick into the step parity's slot.
var<workgroup> wGauge: atomic<u32>;
// The brick's vertices and a one-vertex halo (every vertex this kernel reads), staged once.
var<workgroup> tph: array<f32, 1000>;
var<private> phOrigin: vec3i;
fn ph(v: vec3i) -> f32 { return tph[tileIndex(v - phOrigin)]; }
const FAR_AIR: f32 = 4.0;
var<workgroup> red: array<f32, 512>;
var<workgroup> redFar: array<f32, 512>;
var<workgroup> sFar: u32;
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let b = atomicLoad(&bricks[B_LACT + wg.x]);
  let cls = atomicLoad(&bricks[B_CLASS + b]);
  let deep = (cls & CLASS_DEEP) != 0u;
  let c = brickCoord(b) * 8 + localCoord(l);
  let i = (b << 9u) | l;
  if (l == 0u) { atomicStore(&bricks[B_PCOUNT + b], 0u); atomicStore(&bricks[B_FAST + b], 0u); }
  phOrigin = brickCoord(b) * 8;
  let parity = atomicLoad(&ctrl[C_STEP]) & 1u;
  if (l == 0u) {
    atomicStore(&wGauge, 0u);
    // The first workgroup clears the other parity's pair, which the last step's rate read.
    if (wg.x == 0u) { atomicStore(&ctrl[C_VOL_SUM + 2u * (parity ^ 1u)], 0u); atomicStore(&ctrl[C_VOL_SUM + 2u * (parity ^ 1u) + 1u], 0u); }
  }
  // DEEP bricks read no level set (always liquid, never far, no particles).
  var far = 0.0;
  if (!deep) {
    far = PHI_MAX;
    for (var t = l; t < 1000u; t += 512u) {
      let x = phis[PHI_OFF + cidx(clampV(phOrigin + tileCoord(t)))];
      tph[t] = x;
      far = min(far, x);
    }
  }
  workgroupBarrier();
  var vmin = -PHI_MAX;
  if (!deep) {
    var v: array<f32, 8>;
    var s = 0.0;
    vmin = PHI_MAX;
    for (var k = 0u; k < 8u; k++) {
      v[k] = ph(c + vec3i(i32(k & 1u), i32((k >> 1u) & 1u), i32(k >> 2u)));
      s += v[k];
      vmin = min(vmin, v[k]);
    }
    phis[PHIC_OFF + i] = s * 0.125;
    textureStore(vol, c, vec4f(select(0.0, fill(v), cellOpen(c)), 0.0, 0.0, 0.0));
    if (cellOpen(c) && vmin < 0.0) { atomicAdd(&wGauge, u32(round(nbfCellFill(v) * VOL_UNIT))); }
  } else if (cellOpen(c)) { atomicAdd(&wGauge, u32(VOL_UNIT)); }
  // The brick's minimum vertex phi (next classification) and, over the tile, its far test: a
  // brick whose vertices, and the layer below each axis, all lie beyond FAR_AIR has only faces
  // the per-face test below would skip: no tile, no traces (same values).
  red[l] = select(-PHI_MAX, ph(c), !deep);
  redFar[l] = far;
  workgroupBarrier();
  // Two-level fold (32 threads fold 16 each, then thread 0 the 32): two barriers, not ten.
  if (l < 32u) {
    var m0 = PHI_MAX; var m1 = PHI_MAX;
    for (var j = 0u; j < 16u; j++) { m0 = min(m0, red[j * 32u + l]); m1 = min(m1, redFar[j * 32u + l]); }
    red[l] = m0; redFar[l] = m1;
  }
  workgroupBarrier();
  if (l == 0u) {
    var m0 = PHI_MAX; var m1 = PHI_MAX;
    for (var j = 0u; j < 32u; j++) { m0 = min(m0, red[j]); m1 = min(m1, redFar[j]); }
    if (!deep) { atomicStore(&bricks[B_MINPHI + b], bitcast<u32>(m0)); }
    sFar = select(0u, 1u, m1 > FAR_AIR);
    let total = atomicLoad(&wGauge);
    if (total != 0u) {
      let at = C_VOL_SUM + 2u * parity;
      let low = atomicAdd(&ctrl[at], total);
      if (low + total < low) { atomicAdd(&ctrl[at + 1u], 1u); }
    }
  }
  let allFar = workgroupUniformLoad(&sFar) != 0u;
  // Pass 1: each face is zero, a particle transfer, or a backtrace (bit a of need).
  let part = true;
  var need = 0u;
  for (var a = 0u; a < 3u; a++) {
    var mom = 0; var wsum = 0;
    if (part) {
      let pc = pidx(c);
      mom = atomicExchange(&acc[a * PN + pc], 0);
      wsum = atomicExchange(&acc[(3u + a) * PN + pc], 0);
    }
    if (!faceOpen(c, a)) { vel[US_OFF + a * N + i] = 0.0; continue; }
    // The nozzle's swept faces (the brick engine's applyInflowSweptVelocity after the body force): the low
    // face of c is the +face of q = c - e_a; forced, no trace.
    if (inSweptCell(c - unitAxis(a))) { vel[US_OFF + a * N + i] = inForced(c - unitAxis(a), a, 0.0); continue; }
    if (allFar) { vel[US_OFF + a * N + i] = 0.0; continue; }
    // Faces whose two cells lie wholly more than FAR_AIR cells into the air are read by nothing
    // (projection, FLIP and particle support read faces next to liquid cells; advection and
    // extension read U, which projection zeroes there): no backtrace.
    // In the chamber the particles carry the liquid, and a face with no particle weight
    // between two all-air cells is read by nothing either: the projection reads faces of
    // liquid cells only, and the G2P FLIP difference at a particle reads the trilinear
    // stencil around it, which lies inside that particle's own quadratic B-spline P2G support
    // (1.5 h), so every face it reads carries weight. Standard FLIP keeps no grid-advected
    // velocity where particles carry the liquid (Zhu & Bridson 2005); the values these faces
    // stored were never used, and their traces through the fast jet's air band cost most of
    // the kernel.
    let farAt = select(FAR_AIR, 0.0, part && wsum == 0);
    if (vmin > farAt) {
      let o = c - unitAxis(a);
      let t1 = unitAxis((a + 1u) % 3u); let t2 = unitAxis((a + 2u) % 3u);
      if (min(min(ph(o), ph(o + t1)), min(ph(o + t2), ph(o + t1 + t2))) > farAt) { vel[US_OFF + a * N + i] = 0.0; continue; }
    }
    var transfer = false;
    if (wsum > 0) {
      let t1 = unitAxis((a + 1u) % 3u); let t2 = unitAxis((a + 2u) % 3u);
      let phiF = 0.25 * (ph(c) + ph(c + t1) + ph(c + t2) + ph(c + t1 + t2));
      transfer = phiF >= -RCOMB;
    }
    if (transfer) { vel[US_OFF + a * N + i] = (f32(mom) / SM) / (f32(wsum) / SW) + gravityOn(a, c); continue; }
    need |= 1u << a;
  }
  // The backtraces run in velocityTrace (a lean kernel: this one's registers and workgroup memory
  // held a brick a core). The need mask rides in the KNOWN word, free until projection rewrites it.
  bits[KNOWN_OFF + i] = need;
}
`);

// ---------------------------------------------------------------------------
// 12b. The velocity's backtraces (see 12): the faces pass 1 left to trace (the
//      need mask in the KNOWN word), fineFaceSL plus gravity. Split from 12 and
//      run as TRACE_SLABS (8) 64-cell slabs a brick, each compacting its traced faces
//      into a dense list: a brick with a handful of traces (most of the chamber)
//      holds two warps, not a 512-thread workgroup a core while its few long
//      traces run, and every warp runs full. Same values (each face's trace and
//      store are independent of the others).
// ---------------------------------------------------------------------------
export const velocityTraceKernel = kernel("pour.velocityTrace", [ctrlRW, bricksRW, velRW, bitsRO, phisRO], /* wgsl */ `
${velocitySampler("velU", "vel", "U_OFF")}
${bulkWGSL(true)}
${FACE_SL_WGSL}
var<workgroup> sFaces: array<u32, 192>;
var<workgroup> sTraces: atomic<u32>;
var<workgroup> sTracesU: u32;
// Workgroup (x, y): slab y of LACT brick x.
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let b = atomicLoad(&bricks[B_LACT + wg.x]);
  let lc = wg.y * 64u + l;
  var need = bits[KNOWN_OFF + ((b << 9u) | lc)] & 7u;
  // Only the faces projection reads (project: an open face with a liquid cell, PHIC < 0, on either side, or a
  // face below a BULK cell): it sets every other face's U to zero before the extension rewrites it, and the FLIP
  // delta, the rhs and the particles read US only where it projected. The rest were dead traces.
  if (need != 0u) {
    let c0 = brickCoord(b) * 8 + localCoord(lc);
    let lb = cellOpen(c0) && phis[PHIC_OFF + ((b << 9u) | lc)] < 0.0;
    for (var a = 0u; a < 3u; a++) {
      if (((need >> a) & 1u) == 0u) { continue; }
      let ca = c0 - unitAxis(a);
      let read = faceOpen(c0, a) && (lb || cellBulk(ca) || phis[PHIC_OFF + cidx(ca)] < 0.0);
      if (!read) { need &= ~(1u << a); vel[US_OFF + a * N + ((b << 9u) | lc)] = 0.0; }
    }
  }
  if (need != 0u) {
    var s = atomicAdd(&sTraces, countOneBits(need));
    for (var a = 0u; a < 3u; a++) { if (((need >> a) & 1u) != 0u) { sFaces[s] = lc | (a << 9u); s++; } }
  }
  workgroupBarrier();
  if (l == 0u) { sTracesU = atomicLoad(&sTraces); }
  let traces = workgroupUniformLoad(&sTracesU);
  for (var k = l; k < traces; k += 64u) {
    let f = sFaces[k];
    let cl = f & 511u; let a = f >> 9u;
    let c = brickCoord(b) * 8 + localCoord(cl);
    let val = fineFaceSL(a, c);
    if (slSegs > 12.0) { atomicMax(&ctrl[C_MAX_SEG], u32(slSegs)); }
    vel[US_OFF + a * N + ((b << 9u) | cl)] = val + gravityOn(a, c);
  }
}
`);

// ---------------------------------------------------------------------------
// 13. Fine pressure operator (ghost fluid, Gibou 2002 / Enright 2003; theta
//     clamped to 1e-3), Neumann at closed faces, Dirichlet p = 0 above the open
//     top. rhs = -div(Us). The pressure x stays as the warm start (zero outside
//     the liquid). Also resets
//     the particle-cell counts once every consumer has read them.
// ---------------------------------------------------------------------------
export const operatorKernel = kernel("pour.operator", [bricksRW, phisRO, velRO, opRW, bitsRW, presRW, pgridRW, bulkRO, ctrlRW], /* wgsl */ `
${bulkWGSL(true)}
${VOLUME_RATE_WGSL}
// The volume control's rate (uAdvect's FH_RATE: this step's gauge against the budget, which holds every earlier
// step's delivered flow); the RHS gains h rate a cell (the brick's h^2 rate capacity over its h-scaled rows).
var<private> rateH: f32;
fn volumeRate() -> f32 {
  let at = C_VOL_SUM + 2u * (atomicLoad(&ctrl[C_STEP]) & 1u);
  let filled = f32(atomicLoad(&ctrl[at + 1u])) * 4096.0 + f32(atomicLoad(&ctrl[at])) / VOL_UNIT;
  return nbfVolumeRate(filled, bitcast<f32>(atomicLoad(&ctrl[C_VOL_BUDGET])), DT_S, V_CONTROL, true);
}
var<workgroup> anyLiquid: atomic<u32>;
${BULK_NBR_WGSL}
fn us(a: u32, c: vec3i) -> f32 {
  if (c[a] >= dimOf(a)) { return 0.0; }
  return vel[US_OFF + a * N + cidx(c)];
}
/** Flux (fine face areas) through coarse face (a, k), the low face of k: 16 Us between two BULK cells, else its 16 fine faces. */
fn cflux(a: u32, k: vec3i) -> f32 {
  if (k[a] <= 0 || k[a] >= dimOf(a) / 4) { return 0.0; }
  if (coarseBulk(k) && coarseBulk(k - unitAxis(a))) { return 16.0 * bulk[BUS_OFF + a * NCL + clIndex(k)]; }
  let t1 = unitAxis((a + 1u) % 3u); let t2 = unitAxis((a + 2u) % 3u);
  var s = 0.0;
  for (var u = 0; u < 4; u++) { for (var v = 0; v < 4; v++) { s += us(a, 4 * k + u * t1 + v * t2); } }
  return s;
}
/** BULK coarse cell j: its composite row. A fine neighbour in air is a Dirichlet p = 0 at its centre. */
fn bulkOperator(j: u32) {
  let k = bulkCell(j);
  let sl = bulkSlot(k);
  var diag = 0.0; var mask = 0u;
  for (var kk = 0u; kk < 6u; kk++) {
    let kn = k + select(-1, 1, (kk & 1u) == 1u) * unitAxis(kk >> 1u);
    if (coarseBulk(kn)) { mask |= 1u << kk; diag += 4.0; continue; }
    for (var u = 0; u < 4; u++) { for (var v = 0; v < 4; v++) {
      if (fineOpen(subNbr(k, kk, u, v))) { diag += SUBC; }
    } }
  }
  var div = 0.0;
  for (var a = 0u; a < 3u; a++) { div += cflux(a, k + unitAxis(a)) - cflux(a, k); }
  op[DIAG_OFF + sl] = diag; op[RHS_OFF + sl] = -div + 64.0 * rateH; bits[MASK_OFF + sl] = mask;
}
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let rate = volumeRate();
  rateH = CELL_H * rate;
  if (wg.x == 0u && l == 0u) {
    let at = C_VOL_SUM + 2u * (atomicLoad(&ctrl[C_STEP]) & 1u);
    atomicStore(&ctrl[C_RATE], bitcast<u32>(rate));
    atomicStore(&ctrl[C_FILLED], bitcast<u32>(f32(atomicLoad(&ctrl[at + 1u])) * 4096.0 + f32(atomicLoad(&ctrl[at])) / VOL_UNIT));
  }
  if (wg.x < BULK_WG) {
    let j = wg.x * 512u + l;
    if (j < NBULK * 8u) { bulkOperator(j); }
    return;
  }
  let b = atomicLoad(&bricks[B_LACT + wg.x - BULK_WG]);
  if (l == 0u) { atomicStore(&anyLiquid, 0u); }
  workgroupBarrier();
  let c = brickCoord(b) * 8 + localCoord(l);
  let i = (b << 9u) | l;
  // The count resets for the next step; G2P's mass rules read it from the third region.
  atomicStore(&pgrid[PN2 + pidx(c)], atomicExchange(&pgrid[pidx(c)], 0u) & 0xffffu);
  let pc = phis[PHIC_OFF + i];
  let liquid = cellOpen(c) && pc < 0.0;
  if (liquid) { atomicStore(&anyLiquid, 1u); }
  // The pressure solve visits the liquid bricks only (append once per brick).
  workgroupBarrier();
  if (l == 0u) { atomicStore(&bricks[B_LIQCELL + b], atomicLoad(&anyLiquid)); }
  if (l == 0u && atomicLoad(&anyLiquid) != 0u) {
    let j = atomicAdd(&ctrl[C_LIQ_COUNT], 1u);
    atomicStore(&bricks[B_LLIQ + j], b);
    // The shifted blocks (origin 8 s - (4, 0, 4)) overlapping this brick: s = brick + {0, 1} x {0} x {0, 1}, listed once a step.
    let stamp = atomicLoad(&ctrl[C_STEP]);
    let lbc = brickCoord(b);
    for (var q = 0u; q < 4u; q++) {
      let sb = lbc + vec3i(i32(q & 1u), 0, i32(q >> 1u));
      let f = u32(sb.x + ${NX / 8 + 1} * (sb.y + ${NY / 8} * sb.z));
      if (atomicExchange(&bricks[B_SFLAG + f], stamp) != stamp) {
        let e = atomicAdd(&ctrl[C_SHIFT_COUNT], 1u);
        atomicStore(&bricks[B_LSHIFT + e], u32(sb.x) | (u32(sb.y) << 8u) | (u32(sb.z) << 16u));
      }
    }
  }
  if (!liquid) {
    op[DIAG_OFF + i] = 0.0; op[RHS_OFF + i] = 0.0; bits[MASK_OFF + i] = 0u; pres[i] = 0.0;
    return;
  }
  var diag = 0.0; var mask = 0u;
  for (var k = 0u; k < 6u; k++) {
    let a = k >> 1u;
    let s = select(-1, 1, (k & 1u) == 1u);
    let n = c + s * unitAxis(a);
    var pn = PHI_MAX;
    // The box is closed on every side (its top too): Neumann.
    if (!inBox(n)) { continue; } else {
      if (!cellOpen(n)) { continue; }
      if (cellBulk(n)) { mask |= 1u << (k + 6u); diag += SUBC; continue; }
      // A cell of a brick outside the active set bounds the step's domain: its face is neither traced nor projected
      // (U stays zero there), so the operator closes it to match. The reach keeps liquid a brick from any such
      // face; only liquid the union forms at the face of a particle brick meets one (none at steps 60-500).
      if ((atomicLoad(&bricks[B_FLAGS + brickOfCell(n)]) & 1u) == 0u) { continue; }
      pn = phis[PHIC_OFF + cidx(n)];
    }
    if (pn < 0.0) { mask |= 1u << k; diag += 1.0; }
    else { diag += 1.0 / max(pc / (pc - pn), THETA_MIN); }
  }
  var div = 0.0;
  for (var a = 0u; a < 3u; a++) { div += us(a, c + unitAxis(a)) - us(a, c); }
  op[DIAG_OFF + i] = diag; op[RHS_OFF + i] = -div + rateH; bits[MASK_OFF + i] = mask;
}
`);

// ---------------------------------------------------------------------------
// 14. Aggregation multigrid hierarchy: A_c = R A P / OC with piecewise-constant
//     P and R = P^T (Braess 1995), levels 1..3 per brick (brick-local), 4..6 in
//     one small workgroup. Couplings are stored on the lower cell (+x,+y,+z).
// ---------------------------------------------------------------------------
export const coarsenBrickKernel = kernel("pour.coarsenBrick", [bricksRW, opRO, bitsRO, coarseRW, ctrlRW], /* wgsl */ `
${levelWGSL()}
${bulkWGSL(true)}
/** List the level-4 block (2x2x2 bricks) of brick bc once per step: it holds a level-3 unknown. */
fn markL4(bc: vec3i) {
  let q = bc / 2;
  let q4 = u32(q.x + 8 * (q.y + 16 * q.z));
  let stamp = atomicLoad(&ctrl[C_STEP]);
  if (atomicExchange(&bricks[B_L4FLAG + q4], stamp) != stamp) {
    let j = atomicAdd(&ctrl[C_L4_COUNT], 1u);
    atomicStore(&bricks[B_L4LIST + j], q4);
  }
}
${BULK_NBR_WGSL}
var<workgroup> s1: array<vec4f, 64>;
var<workgroup> s2: array<vec4f, 8>;
/** Level-2 coupling between coarse cell k (BULK, native at level 2) and the fine level-2 cell across side kk:
 *  the fine-face couplings SUBC of its liquid fine cells, scaled as the fine side's two aggregation levels
 *  (1 / OC^2), so that the fine level-2 rows stay balanced (their diagonals carry SUBC / OC^2 per face). */
fn subCoupling(k: vec3i, kk: u32) -> f32 {
  var w = 0.0;
  for (var u = 0; u < 4; u++) { for (var v = 0; v < 4; v++) {
    let c = subNbr(k, kk, u, v);
    if (inBox(c) && op[DIAG_OFF + cidx(c)] > 0.0) { w += SUBC; }
  } }
  return w / (OC * OC);
}
fn storeCop(l: u32, c: vec3i, o: vec4f) {
  let g = lidx(l, c);
  coarse[COP + 4u * g] = o.x; coarse[COP + 4u * g + 1u] = o.y; coarse[COP + 4u * g + 2u] = o.z; coarse[COP + 4u * g + 3u] = o.w;
}
const NB8: u32 = (NBULK + 7u) / 8u;
// Workgroups [0, NB8): eight BULK bricks each (8 threads a brick, one per coarse cell); then one active brick each
// (launched over NB8 + n, C_ARG_ACTBB).
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  if (wg.x < NB8) {
    // BULK brick: its eight coarse cells are level-2 cells with native rows (coarse-coarse 4) and the
    // level-2 couplings to fine neighbours; the diagonal is the coupling sum. Level 1 stays empty.
    let e = wg.x * 8u + (t >> 3u);
    let own = e < NBULK;
    let bb = brickCoord(atomicLoad(&bricks[B_LBULK + min(e, NBULK - 1u)]));
    if (own) {
      let j = t & 7u;
      let k = bb * 2 + vec3i(i32(j & 1u), i32((j >> 1u) & 1u), i32(j >> 2u));
      var o = vec4f(0.0);
      for (var kk = 0u; kk < 6u; kk++) {
        let kn = k + select(-1, 1, (kk & 1u) == 1u) * unitAxis(kk >> 1u);
        let w = select(subCoupling(k, kk), 4.0, coarseBulk(kn));
        o.x += w;
        if ((kk & 1u) == 1u) { o[1u + (kk >> 1u)] = w; }
      }
      storeCop(2u, k, o);
      s1[t] = o;
    }
    workgroupBarrier();
    if (own && (t & 7u) == 0u) {
      var ds = 0.0; var it = 0.0; var ax = 0.0; var ay = 0.0; var az = 0.0;
      for (var k = 0u; k < 8u; k++) {
        let d = vec3i(i32(k & 1u), i32((k >> 1u) & 1u), i32(k >> 2u));
        let o = s1[t + k];
        ds += o.x;
        if (d.x == 0) { it += o.y; } else { ax += o.y; }
        if (d.y == 0) { it += o.z; } else { ay += o.z; }
        if (d.z == 0) { it += o.w; } else { az += o.w; }
      }
      let o3 = vec4f(ds - 2.0 * it, ax, ay, az) / OC;
      storeCop(3u, bb, o3);
      if (o3.x > 0.0) { markL4(bb); }
    }
    return;
  }
  let b = atomicLoad(&bricks[B_LACT + wg.x - NB8]);
  let bc = brickCoord(b);
  let q = vec3i(i32(t & 3u), i32((t >> 2u) & 3u), i32(t >> 4u));
  var dsum = 0.0; var internal = 0.0; var cx = 0.0; var cy = 0.0; var cz = 0.0;
  for (var k = 0u; k < 8u; k++) {
    let d = vec3i(i32(k & 1u), i32((k >> 1u) & 1u), i32(k >> 2u));
    let fl = 2 * q + d;
    let fi = (b << 9u) | u32(fl.x | (fl.y << 3u) | (fl.z << 6u));
    dsum += op[DIAG_OFF + fi];
    let m = bits[MASK_OFF + fi];
    let px = f32((m >> 1u) & 1u); let py = f32((m >> 3u) & 1u); let pz = f32((m >> 5u) & 1u);
    if (d.x == 0) { internal += px; } else { cx += px; }
    if (d.y == 0) { internal += py; } else { cy += py; }
    if (d.z == 0) { internal += pz; } else { cz += pz; }
  }
  let o1 = vec4f(dsum - 2.0 * internal, cx, cy, cz) / OC;
  let g1 = lidx(1u, bc * 4 + q);
  coarse[COP + 4u * g1] = o1.x; coarse[COP + 4u * g1 + 1u] = o1.y; coarse[COP + 4u * g1 + 2u] = o1.z; coarse[COP + 4u * g1 + 3u] = o1.w;
  s1[t] = o1;
  workgroupBarrier();
  if (t < 8u) {
    let r = vec3i(i32(t & 1u), i32((t >> 1u) & 1u), i32(t >> 2u));
    var ds = 0.0; var it = 0.0; var ax = 0.0; var ay = 0.0; var az = 0.0;
    for (var k = 0u; k < 8u; k++) {
      let d = vec3i(i32(k & 1u), i32((k >> 1u) & 1u), i32(k >> 2u));
      let ql = 2 * r + d;
      let o = s1[u32(ql.x + 4 * (ql.y + 4 * ql.z))];
      ds += o.x;
      if (d.x == 0) { it += o.y; } else { ax += o.y; }
      if (d.y == 0) { it += o.z; } else { ay += o.z; }
      if (d.z == 0) { it += o.w; } else { az += o.w; }
    }
    var o2 = vec4f(ds - 2.0 * it, ax, ay, az) / OC;
    // A BULK level-2 cell above (+x, +y, +z): the coupling is stored here, on the lower cell.
    for (var a = 0u; a < 3u; a++) {
      let kn = bc * 2 + r + unitAxis(a);
      if (coarseBulk(kn)) { o2[1u + a] = subCoupling(kn, 2u * a); }
    }
    let g2 = lidx(2u, bc * 2 + r);
    coarse[COP + 4u * g2] = o2.x; coarse[COP + 4u * g2 + 1u] = o2.y; coarse[COP + 4u * g2 + 2u] = o2.z; coarse[COP + 4u * g2 + 3u] = o2.w;
    s2[t] = o2;
  }
  workgroupBarrier();
  if (t == 0u) {
    var ds = 0.0; var it = 0.0; var ax = 0.0; var ay = 0.0; var az = 0.0;
    for (var k = 0u; k < 8u; k++) {
      let d = vec3i(i32(k & 1u), i32((k >> 1u) & 1u), i32(k >> 2u));
      let o = s2[k];
      ds += o.x;
      if (d.x == 0) { it += o.y; } else { ax += o.y; }
      if (d.y == 0) { it += o.z; } else { ay += o.z; }
      if (d.z == 0) { it += o.w; } else { az += o.w; }
    }
    let o3 = vec4f(ds - 2.0 * it, ax, ay, az) / OC;
    let g3 = lidx(3u, bc);
    coarse[COP + 4u * g3] = o3.x; coarse[COP + 4u * g3 + 1u] = o3.y; coarse[COP + 4u * g3 + 2u] = o3.z; coarse[COP + 4u * g3 + 3u] = o3.w;
    if (o3.x > 0.0) { markL4(bc); }
  }
}
`);

const COARSE_COMMON = /* wgsl */ `
${levelWGSL()}
fn cop(l: u32, c: vec3i) -> vec4f {
  let q = lidx(l, c);
  return vec4f(coarse[COP + 4u * q], coarse[COP + 4u * q + 1u], coarse[COP + 4u * q + 2u], coarse[COP + 4u * q + 3u]);
}
/** Aggregate level l from level l-1 (dense): A_c = R A P / OC. */
fn aggregate(l: u32, c: vec3i) {
  var ds = 0.0; var it = 0.0; var ax = 0.0; var ay = 0.0; var az = 0.0;
  for (var k = 0u; k < 8u; k++) {
    let d = vec3i(i32(k & 1u), i32((k >> 1u) & 1u), i32(k >> 2u));
    let o = cop(l - 1u, 2 * c + d);
    ds += o.x;
    if (d.x == 0) { it += o.y; } else { ax += o.y; }
    if (d.y == 0) { it += o.z; } else { ay += o.z; }
    if (d.z == 0) { it += o.w; } else { az += o.w; }
  }
  let q = lidx(l, c);
  coarse[COP + 4u * q] = (ds - 2.0 * it) / OC; coarse[COP + 4u * q + 1u] = ax / OC;
  coarse[COP + 4u * q + 2u] = ay / OC; coarse[COP + 4u * q + 3u] = az / OC;
}
fn cellOfLevel(l: u32, i: u32) -> vec3i { let d = ldims(l); return vec3i(i32(i) % d.x, (i32(i) / d.x) % d.y, i32(i) / (d.x * d.y)); }
fn levelCount(l: u32) -> u32 { let d = ldims(l); return u32(d.x * d.y * d.z); }
/** Off-diagonal sum: sum over neighbours n of coupling(c, n) * e[n] (couplings are stored positive). */
fn offSum(l: u32, c: vec3i, o: vec4f, eo: u32) -> f32 {
  var s = 0.0;
  let d = ldims(l);
  if (c.x + 1 < d.x) { s += o.y * coarse[eo + lidx(l, c + vec3i(1, 0, 0))]; }
  if (c.y + 1 < d.y) { s += o.z * coarse[eo + lidx(l, c + vec3i(0, 1, 0))]; }
  if (c.z + 1 < d.z) { s += o.w * coarse[eo + lidx(l, c + vec3i(0, 0, 1))]; }
  if (c.x > 0) { let n = c - vec3i(1, 0, 0); s += coarse[COP + 4u * lidx(l, n) + 1u] * coarse[eo + lidx(l, n)]; }
  if (c.y > 0) { let n = c - vec3i(0, 1, 0); s += coarse[COP + 4u * lidx(l, n) + 2u] * coarse[eo + lidx(l, n)]; }
  if (c.z > 0) { let n = c - vec3i(0, 0, 1); s += coarse[COP + 4u * lidx(l, n) + 3u] * coarse[eo + lidx(l, n)]; }
  return s;
}
/** Gauss-Seidel update of e (at offset eo) at (l, c) against the rhs CB. */
fn relaxC(l: u32, c: vec3i, eo: u32) {
  let o = cop(l, c);
  if (o.x <= 0.0) { return; }
  coarse[eo + lidx(l, c)] = (coarse[CB + lidx(l, c)] + offSum(l, c, o, eo)) / o.x;
}
/** Residual b - A e at (l, c), zero outside the operator. */
fn residC(l: u32, c: vec3i, eo: u32) -> f32 {
  let o = cop(l, c);
  if (o.x <= 0.0) { return 0.0; }
  return coarse[CB + lidx(l, c)] - (o.x * coarse[eo + lidx(l, c)] - offSum(l, c, o, eo));
}
`;

export const coarsenTopKernel = kernel("pour.coarsenTop", [coarseRW, ctrlRW, argsRW], /* wgsl */ `
${COARSE_COMMON}
var<workgroup> sInv: array<f32, ${COARSEST_CELLS * COARSEST_CELLS}>;
@compute @workgroup_size(1024) fn main(@builtin(local_invocation_index) t: u32) {
  if (t == 0u) {
    // Publish the operator's liquid list as the pressure solve's dispatch arguments.
    let n = atomicExchange(&ctrl[C_LIQ_COUNT], 0u);
    atomicStore(&ctrl[C_N_LIQ], n);
    atomicStore(&args[C_ARG_LIQ], n); atomicStore(&args[C_ARG_LIQB], BULK_WG + n); atomicStore(&args[C_ARG_LIQBB], NBULK + n);
    let ns = atomicExchange(&ctrl[C_SHIFT_COUNT], 0u);
    atomicStore(&ctrl[C_N_SHIFT], ns);
    atomicStore(&args[C_ARG_SHIFT], ns); atomicStore(&args[C_ARG_SHIFT + 1u], 1u); atomicStore(&args[C_ARG_SHIFT + 2u], 1u);
    let n4 = atomicExchange(&ctrl[C_L4_COUNT], 0u);
    atomicStore(&ctrl[C_N_L4], n4);
    // The level-3 block passes take eight level-4 blocks a workgroup.
    atomicStore(&args[C_ARG_L4], (n4 + 7u) / 8u); atomicStore(&args[C_ARG_L4 + 1u], 1u); atomicStore(&args[C_ARG_L4 + 2u], 1u);
    atomicStore(&args[C_ARG_L4U], n4); atomicStore(&args[C_ARG_L4U + 1u], 1u); atomicStore(&args[C_ARG_L4U + 2u], 1u);
  }
  for (var l = 4u; l <= 6u; l++) {
    for (var i = t; i < levelCount(l); i += 1024u) { aggregate(l, cellOfLevel(l, i)); }
    storageBarrier(); workgroupBarrier();
  }
  // The coarsest level is solved exactly (Trottenberg, Oosterlee & Schuller 2001, Multigrid, sec. 2.5):
  // its NC6 x NC6 operator is inverted here once a step by Gauss-Jordan elimination without pivoting
  // (the operator is SPD: Golub & Van Loan 2013, sec. 4.2), thread t holding entry (t / NC6, t % NC6),
  // and each cycle's coarsest solve is one product with the inverse. Inactive cells are identity rows.
  // Threads t < NC6^2 hold the entries; the rest run the barriers only.
  let own = t < NC6 * NC6;
  let tt = min(t, NC6 * NC6 - 1u);
  let i = tt / NC6; let j = tt % NC6;
  let ci = cellOfLevel(6u, i); let cj = cellOfLevel(6u, j);
  let oi = cop(6u, ci); let oj = cop(6u, cj);
  var a = 0.0;
  if (i == j) { a = select(1.0, oi.x, oi.x > 0.0); }
  else if (oi.x > 0.0 && oj.x > 0.0) {
    let d = cj - ci;
    if (all(d == vec3i(1, 0, 0))) { a = -oi.y; } else if (all(d == vec3i(0, 1, 0))) { a = -oi.z; } else if (all(d == vec3i(0, 0, 1))) { a = -oi.w; }
    else if (all(d == vec3i(-1, 0, 0))) { a = -oj.y; } else if (all(d == vec3i(0, -1, 0))) { a = -oj.z; } else if (all(d == vec3i(0, 0, -1))) { a = -oj.w; }
  }
  if (own) { sInv[tt] = a; }
  workgroupBarrier();
  for (var p = 0u; p < NC6; p++) {
    let app = sInv[p * NC6 + p]; let aip = sInv[i * NC6 + p]; let apj = sInv[p * NC6 + j];
    workgroupBarrier();
    if (app <= 0.0) { if (t == 0u) { atomicOr(&ctrl[C_FAIL], FAIL_PIVOT); } }
    let r = 1.0 / app;
    if (own) {
      if (i == p && j == p) { sInv[tt] = r; }
      else if (i == p) { sInv[tt] = apj * r; }
      else if (j == p) { sInv[tt] = -aip * r; }
      else { sInv[tt] = a - aip * apj * r; }
    }
    workgroupBarrier();
    a = sInv[tt];
  }
  if (own) { coarse[CINV + tt] = a; }
}
`);

// ---------------------------------------------------------------------------
// 15. Pressure: conjugate gradients preconditioned by one symmetric multigrid
//     V(2,2)-cycle (MGPCG, McAdams 2010), over the aggregation hierarchy.
//     Smoothers are brick-block Gauss-Seidel (red-black inside a brick, the
//     halo frozen): pre-smoothing red-first from zero, post-smoothing in the
//     reverse (black-first) order, Galerkin coarse operators and a palindromic
//     coarsest sweep, so the preconditioner is symmetric positive definite.
//     Per iteration k (each pass over the active bricks unless noted):
//       X_k  x += alpha p, r -= alpha q (k = 0: r = b - A x, x warm-started),
//            L-inf of r, fine pre-smoothing z1 ~ A_b^-1 r
//       D0   r - A z1 restricted to level 1, level-1 pre-smoothing
//       D1   level-1 residual restricted to level 2, level-2 pre-smoothing
//       D2   level-2 residual restricted to level 3
//       C_k  (one workgroup) convergence gate; V(2,2) on levels 3..6
//       U2, U1  level-2, level-1 post-smoothing after prolongation
//       U0   z = z1 + P e1 post-smoothed; dot r.z
//       P_k  beta = rho_k / rho_(k-1); p = z + beta p; q = A z + beta q (= A p); dot p.q
//     Dots are float sums in 64 slots per set (compare-exchange), two sets by
//     iteration parity; C_k saves rho_(k-1) and clears set k&1.
// ---------------------------------------------------------------------------
/** The coarse-correction weight of every prolongation in the V-cycle (over-correction: Braess 1995, Towards algebraic
 *  multigrid for elliptic problems of second order; Vanek, Mandel & Brezina 1996 sec. 2). Piecewise-constant
 *  aggregation makes each Galerkin coarse operator twice as stiff to smooth error as a rediscretised one (a 2x2x2
 *  aggregate's stencil is (3, -1/2) against (3/2, -1/4)), so an unweighted correction falls short. The cycle stays
 *  symmetric for any weight and positive definite below 2. Over steps 301-325 of the pour: 1.0 took 10 cycles a
 *  step, 1.2 7.4, 1.3 6.7, 1.4 6.2, 1.5 6.4, 1.6 6.8, 1.7 7.6 (2026-11-21); the whole run at 1.4 at most 9. */
const MG_OMEGA = "1.4";
const PCG_WGSL = /* wgsl */ `
const X_OFF: u32 = 0u; const R_OFF: u32 = N; const Z_OFF: u32 = 2u * N; const PP_OFF: u32 = 3u * N; const Q_OFF: u32 = 4u * N; const Z1_OFF: u32 = 5u * N;
fn redSlot(rs: u32, which: u32) -> u32 { return C_RED + rs * 192u + which * 64u; }
fn redAdd(rs: u32, which: u32, slot: u32, v: f32) {
  let idx = redSlot(rs, which) + (slot & 63u);
  var old = atomicLoad(&ctrl[idx]);
  loop {
    let r = atomicCompareExchangeWeak(&ctrl[idx], old, bitcast<u32>(bitcast<f32>(old) + v));
    if (r.exchanged) { break; }
    old = r.old_value;
  }
}
var<workgroup> sRS: array<f32, 128>;
/** The sums of dot sets (rs, w0) and (rs, w1): threads l < 128 load the 128 partial slots at once (one
 *  device latency, not 128 serial ones), then each sum adds its 64 in slot order, as a serial loop would.
 *  Every thread of the workgroup calls it, in uniform control flow. */
fn redSums(rs: u32, w0: u32, w1: u32, l: u32) -> vec2f {
  if (l < 128u) { sRS[l] = bitcast<f32>(atomicLoad(&ctrl[redSlot(rs, select(w0, w1, l >= 64u)) + (l & 63u)])); }
  workgroupBarrier();
  var s = vec2f(0.0);
  for (var j = 0u; j < 64u; j++) { s += vec2f(sRS[j], sRS[64u + j]); }
  workgroupBarrier();
  return s;
}
/** A applied at liquid cell (c, i) to the field at offset f (neighbours through the liquid mask). */
fn applyA(c: vec3i, i: u32, f: u32) -> f32 {
  let m = bits[MASK_OFF + i];
  var s = 0.0;
  for (var k = 0u; k < 6u; k++) {
    if (((m >> k) & 1u) == 0u) { continue; }
    s += pres[f + cidx(c + select(-1, 1, (k & 1u) == 1u) * unitAxis(k >> 1u))];
  }
  if ((m >> 6u) != 0u) {
    for (var k = 0u; k < 6u; k++) {
      if (((m >> (k + 6u)) & 1u) == 0u) { continue; }
      s += SUBC * pres[f + bulkSlot((c + select(-1, 1, (k & 1u) == 1u) * unitAxis(k >> 1u)) / 4)];
    }
  }
  return op[DIAG_OFF + i] * pres[f + i] - s;
}
${BULK_APPLY_WGSL}
var<workgroup> sRedA: array<f32, 512>;
var<workgroup> sRedB: array<f32, 512>;
var<workgroup> sRed2: array<f32, 32>;
/** Workgroup fold of sRedA (sum) / sRedB (max), all 512 written before the call: 32 threads fold 16
 *  each (conflict-free stride), thread 0 the 32; two barriers, not ten. The result is thread 0's. */
fn foldSumA(l: u32) -> f32 {
  workgroupBarrier();
  if (l < 32u) { var s = 0.0; for (var j = 0u; j < 16u; j++) { s += sRedA[j * 32u + l]; } sRed2[l] = s; }
  workgroupBarrier();
  var s = 0.0;
  if (l == 0u) { for (var j = 0u; j < 32u; j++) { s += sRed2[j]; } }
  return s;
}
fn foldMaxB(l: u32) -> f32 {
  workgroupBarrier();
  if (l < 32u) { var s = 0.0; for (var j = 0u; j < 16u; j++) { s = max(s, sRedB[j * 32u + l]); } sRed2[l] = s; }
  workgroupBarrier();
  var s = 0.0;
  if (l == 0u) { for (var j = 0u; j < 32u; j++) { s = max(s, sRed2[j]); } }
  return s;
}
var<workgroup> tp: array<f32, 1000>;
/** Fine block Gauss-Seidel on the staged brick tile tp (rhs per thread): one red-black sweep (2 colour passes),
 *  black-first when rev. At unit coarse weight 3 sweeps beat 2; with MG_OMEGA 1.4 one sweep a pass is cheapest
 *  (steps 301-325, pressure ms a step: 3 sweeps 8.06, 2 7.65, 1 7.40; 2026-11-21). */
fn fineSmooth(c: vec3i, ti: u32, diag: f32, m: u32, rhs: f32, rev: bool) {
  let red = (u32(c.x + c.y + c.z)) & 1u;
  for (var s = 0u; s < 2u; s++) {
    let colour = select(s & 1u, 1u - (s & 1u), rev);
    if (diag > 0.0 && red == colour) {
      var sum = rhs;
      if ((m & 1u) != 0u) { sum += tp[ti - 1u]; }
      if ((m & 2u) != 0u) { sum += tp[ti + 1u]; }
      if ((m & 4u) != 0u) { sum += tp[ti - 10u]; }
      if ((m & 8u) != 0u) { sum += tp[ti + 10u]; }
      if ((m & 16u) != 0u) { sum += tp[ti - 100u]; }
      if ((m & 32u) != 0u) { sum += tp[ti + 100u]; }
      tp[ti] = sum / diag;
    }
    workgroupBarrier();
  }
}
`;
const RZ = 0, PQ = 2;

/**
 * Block Gauss-Seidel on a per-brick block of level L (side s cells, tile with a
 * one-cell halo). Thread t < s^3 owns cell t of the block. Level 2's 8-cell
 * blocks pack 8 to a 64-thread workgroup (PACK2): one SIMD group, not eight
 * mostly idle ones.
 */
function blockWGSL(L: 1 | 2 | 3, pack = 1): string {
  const s = L === 1 ? 4 : 2, T = s + 2;
  return /* wgsl */ `
const BS${L}: i32 = ${s}; const BT${L}: u32 = ${T}u;
// ${pack} block tile(s) of ${T * T * T}; a tile index (bti) is offset by 64 * its block's slot when packed.
var<workgroup> tb${L}: array<f32, ${T * T * T * pack}>;
fn bq${L}(t: u32) -> vec3i { return vec3i(i32(t % ${s}u), i32((t / ${s}u) % ${s}u), i32(t / ${s * s}u)); }
fn bti${L}(q: vec3i) -> u32 { let r = q + vec3i(1); return u32(r.x + ${T} * (r.y + ${T} * r.z)); }
fn bhalo${L}(j: u32) -> vec3i { return vec3i(i32(j % ${T}u), i32((j / ${T}u) % ${T}u), i32(j / ${T * T}u)) - vec3i(1); }
/** Couplings of cell c: own (diag, +x, +y, +z) and the -x, -y, -z couplings stored on the neighbours. */
fn bw${L}(c: vec3i) -> vec3f {
  var w = vec3f(0.0);
  if (c.x > 0) { w.x = coarse[COP + 4u * lidx(${L}u, c - vec3i(1, 0, 0)) + 1u]; }
  if (c.y > 0) { w.y = coarse[COP + 4u * lidx(${L}u, c - vec3i(0, 1, 0)) + 2u]; }
  if (c.z > 0) { w.z = coarse[COP + 4u * lidx(${L}u, c - vec3i(0, 0, 1)) + 3u]; }
  return w;
}
fn bsum${L}(ti: u32, o: vec4f, w: vec3f) -> f32 {
  return o.y * tb${L}[ti + 1u] + o.z * tb${L}[ti + ${T}u] + o.w * tb${L}[ti + ${T * T}u]
    + w.x * tb${L}[ti - 1u] + w.y * tb${L}[ti - ${T}u] + w.z * tb${L}[ti - ${T * T}u];
}
fn bsmooth${L}(own: bool, c: vec3i, ti: u32, o: vec4f, w: vec3f, b: f32, rev: bool) {
  let red = (u32(c.x + c.y + c.z)) & 1u;
  for (var s = 0u; s < 4u; s++) {
    let colour = select(s & 1u, 1u - (s & 1u), rev);
    if (own && o.x > 0.0 && red == colour) { tb${L}[ti] = (b + bsum${L}(ti, o, w)) / o.x; }
    workgroupBarrier();
  }
}
`;
}

/** X_-1: the initial guess, the warm start (last step's pressure) after one Jacobi sweep on the fine rows
 *  (Saad 2003, Iterative Methods for Sparse Linear Systems, sec. 4.1), into Z (free until U0_0): x0 + D^-1 (b - A x0).
 *  A row whose ghost-fluid diagonal changed (theta near THETA_MIN) otherwise starts with a residual of
 *  d * x0 there. BULK rows copy. X_0 takes Z as its x. */
export const pcgGuessKernel = kernel("pour.pcgGuess", [ctrlRW, bricksRO, opRO, bitsRO, presRW], /* wgsl */ `
${PCG_WGSL}
${bulkWGSL(false)}
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  if (wg.x < BULK_WG) {
    let j = wg.x * 512u + l;
    if (j < NBULK * 8u) { let sl = bulkSlot(bulkCell(j)); pres[Z_OFF + sl] = pres[X_OFF + sl]; }
    return;
  }
  let b = bricks[B_LLIQ + wg.x - BULK_WG];
  let c = brickCoord(b) * 8 + localCoord(l);
  let i = (b << 9u) | l;
  let diag = op[DIAG_OFF + i];
  var x = 0.0;
  if (diag > 0.0) { x = pres[X_OFF + i] + (op[RHS_OFF + i] - applyA(c, i, X_OFF)) / diag; }
  pres[Z_OFF + i] = x;
}
`);

/** X_k (see 15). cfg.a = k. */
export const pcgUpdateKernel = kernel("pour.pcgUpdate", [ctrlRW, bricksRO, opRO, bitsRO, presRW, cfgU], /* wgsl */ `
${PCG_WGSL}
${bulkWGSL(false)}
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let k = cfg.a;
  var alpha = 0.0;
  if (k > 0u) {
    let d = redSums((k - 1u) & 1u, ${PQ}u, ${RZ}u, l);
    alpha = select(0.0, d.y / d.x, d.x != 0.0);
  }
  if (wg.x < BULK_WG) {
    // BULK coarse cells: no fine smoothing (their preconditioner is level 2). The residual of a coarse
    // row sums 64 fine cells' divergence: measured per fine cell (/ 64).
    let j = wg.x * 512u + l;
    var r = 0.0;
    if (j < NBULK * 8u) {
      let kc = bulkCell(j);
      let sl = bulkSlot(kc);
      if (k == 0u) { r = op[RHS_OFF + sl] - applyBulk(kc, sl, Z_OFF); pres[X_OFF + sl] = pres[Z_OFF + sl]; }
      else {
        pres[X_OFF + sl] += alpha * pres[PP_OFF + sl];
        r = pres[R_OFF + sl] - alpha * pres[Q_OFF + sl];
      }
      pres[R_OFF + sl] = r;
    }
    sRedB[l] = abs(r) / 64.0;
    let resMax = foldMaxB(l);
    if (l == 0u) { atomicMax(&ctrl[C_RES], bitcast<u32>(resMax)); }
    return;
  }
  let b = bricks[B_LLIQ + wg.x - BULK_WG];
  let lc = localCoord(l);
  let c = brickCoord(b) * 8 + lc;
  let i = (b << 9u) | l;
  let diag = op[DIAG_OFF + i];
  var r = 0.0;
  if (diag > 0.0) {
    if (k == 0u) { r = op[RHS_OFF + i] - applyA(c, i, Z_OFF); pres[X_OFF + i] = pres[Z_OFF + i]; }
    else {
      pres[X_OFF + i] += alpha * pres[PP_OFF + i];
      r = pres[R_OFF + i] - alpha * pres[Q_OFF + i];
    }
  }
  pres[R_OFF + i] = r;
  sRedB[l] = abs(r);
  for (var t = l; t < 1000u; t += 512u) { tp[t] = 0.0; }
  let resMax = foldMaxB(l);
  if (l == 0u) { atomicMax(&ctrl[C_RES], bitcast<u32>(resMax)); }
  let ti = tileIndex(lc);
  fineSmooth(c, ti, diag, bits[MASK_OFF + i], r, false);
  pres[Z1_OFF + i] = select(0.0, tp[ti], diag > 0.0);
}
`);

/**
 * D0B / U0B: the second fine smoothing partition. The L0 smoother is two block
 * Gauss-Seidel passes over two staggered partitions: the bricks (X_k pre, U0
 * post) and blocks shifted by half a brick along x and z (origin 8 s - (4, 0, 4),
 * these passes; a shift along y as well took the same cycles over twice the
 * blocks, since the film lies along x and z), the
 * overlapping/staggered block Schwarz smoother of Smith, Bjorstad & Gropp 1996
 * (Domain Decomposition, ch. 1-2; multiplicative Schwarz over two coverings).
 * Each pass is e <- e + M^-1 (b - A e) with the halo frozen at the current
 * iterate; post-smoothing applies the adjoints in the reverse order (U0B with
 * the colour passes reversed, then U0), so the V-cycle stays symmetric
 * (lib/methods/seep/seep-smoother.test.ts checks the model). A film one or two
 * cells thick otherwise converges slowest across the brick faces, where both
 * brick-aligned passes freeze the halo. BULK cells are not fine unknowns
 * (their brick slots hold coarse cells): skipped, and their coupling enters as
 * in the brick passes (none before the coarse correction, the final coarse
 * values after it). One workgroup per listed shifted block.
 */
function shiftSmoothKernel(post: boolean): KernelSource {
  return kernel(post ? "pour.pcgShiftUp" : "pour.pcgShiftDown", [ctrlRW, bricksRO, opRO, bitsRO, presRW, coarseRW, cfgU], /* wgsl */ `
${levelWGSL()}
${PCG_WGSL}
${bulkWGSL(false)}
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let e = bricks[B_LSHIFT + wg.x];
  let o = 8 * vec3i(i32(e & 255u), i32((e >> 8u) & 255u), i32(e >> 16u)) - vec3i(4, 0, 4);
  for (var t = l; t < 1000u; t += 512u) {
    let c = o + tileCoord(t);
    var v = 0.0;
    if (inBox(c) && !cellBulk(c)) {
      let ci = cidx(c);
      if (op[DIAG_OFF + ci] > 0.0) { v = ${post ? `pres[Z_OFF + ci] + ${MG_OMEGA} * coarse[CE + lidx(1u, c / 2)]` : "pres[Z1_OFF + ci]"}; }
    }
    tp[t] = v;
  }
  workgroupBarrier();
  let lc = localCoord(l);
  let c = o + lc;
  let own = inBox(c) && !cellBulk(c);
  let i = cidx(clampV(c));
  var diag = 0.0; var m = 0u; var rs = 0.0;
  if (own) {
    diag = op[DIAG_OFF + i]; m = bits[MASK_OFF + i]; rs = pres[R_OFF + i];
    ${post ? `if ((m >> 6u) != 0u) {
      for (var kk = 0u; kk < 6u; kk++) {
        if (((m >> (kk + 6u)) & 1u) == 0u) { continue; }
        rs += SUBC * coarse[CE + lidx(2u, (c + select(-1, 1, (kk & 1u) == 1u) * unitAxis(kk >> 1u)) / 4)];
      }
    }` : ""}
  }
  let ti = tileIndex(lc);
  fineSmooth(c, ti, diag, m, rs, ${post});
  // D0B: the pre-smoothed iterate into Z (free until U0); U0B: its iterate into Z1 for U0.
  if (own && diag > 0.0) { pres[${post ? "Z1_OFF" : "Z_OFF"} + i] = tp[ti]; }
}
`);
}
export const pcgShiftDownKernel = shiftSmoothKernel(false);
export const pcgShiftUpKernel = shiftSmoothKernel(true);

/** D0 (see 15). */
export const mgDown0Kernel = kernel("pour.mgDown0", [bricksRO, opRO, bitsRO, presRO, coarseRW], /* wgsl */ `
${levelWGSL()}
${blockWGSL(1)}
// The fine pre-smoothed iterate after both partitions (D0B wrote it into Z).
const X_OFF: u32 = 0u; const R_OFF: u32 = N; const Z1_OFF: u32 = 2u * N;
var<workgroup> sr: array<f32, 512>;
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let b = bricks[B_LLIQ + wg.x];
  let bc = brickCoord(b);
  let c = bc * 8 + localCoord(l);
  let i = (b << 9u) | l;
  let diag = op[DIAG_OFF + i];
  var r = 0.0;
  if (diag > 0.0) {
    let m = bits[MASK_OFF + i];
    var s = 0.0;
    for (var k = 0u; k < 6u; k++) {
      if (((m >> k) & 1u) == 0u) { continue; }
      s += pres[Z1_OFF + cidx(c + select(-1, 1, (k & 1u) == 1u) * unitAxis(k >> 1u))];
    }
    r = pres[R_OFF + i] - (diag * pres[Z1_OFF + i] - s);
  }
  sr[l] = r;
  if (l < ${6 * 6 * 6}u) { tb1[l] = 0.0; }
  workgroupBarrier();
  let own = l < 64u;
  let q = bq1(l);
  let c1 = bc * 4 + q;
  var b1 = 0.0; var o = vec4f(0.0); var w = vec3f(0.0);
  if (own) {
    for (var k = 0u; k < 8u; k++) {
      let fl = 2 * q + vec3i(i32(k & 1u), i32((k >> 1u) & 1u), i32(k >> 2u));
      b1 += sr[u32(fl.x | (fl.y << 3u) | (fl.z << 6u))];
    }
    coarse[CB + lidx(1u, c1)] = b1;
    let g = lidx(1u, c1);
    o = vec4f(coarse[COP + 4u * g], coarse[COP + 4u * g + 1u], coarse[COP + 4u * g + 2u], coarse[COP + 4u * g + 3u]);
    w = bw1(c1);
  }
  let ti = bti1(q);
  bsmooth1(own, c1, ti, o, w, b1, false);
  if (own) { coarse[CEP + lidx(1u, c1)] = select(0.0, tb1[ti], o.x > 0.0); }
}
`);

/** D1 (see 15). */
export const mgDown1Kernel = kernel("pour.mgDown1", [bricksRO, coarseRW, opRO, bitsRO, presRO, ctrlRO], /* wgsl */ `
${levelWGSL()}
${blockWGSL(1)}
${blockWGSL(2, 8)}
// The fine pre-smoothed iterate after both partitions (D0B wrote it into Z).
const R_OFF: u32 = N; const Z1_OFF: u32 = 2u * N;
${BULK_APPLY_WGSL}
var<workgroup> sr: array<f32, 64>;
var<workgroup> sN: u32;
const NB8: u32 = (NBULK + 7u) / 8u;
// Workgroups [0, NB8): eight BULK bricks each; then one liquid brick each (launched over NBULK + n).
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  if (t == 0u) { sN = ctrl[C_N_LIQ]; }
  let nLiq = workgroupUniformLoad(&sN);
  if (wg.x < NB8) {
    // BULK brick: its coarse cells are level-2 cells. Their residual after the fine pre-smoothing
    // r - A z1 (z1 = 0 on coarse cells), then level-2 pre-smoothing like any brick block.
    let slot = t >> 3u;
    let e = wg.x * 8u + slot;
    let bc = brickCoord(bricks[B_LBULK2 + min(e, NBULK - 1u)]);
    for (var j = t; j < 512u; j += 64u) { tb2[j] = 0.0; }
    workgroupBarrier();
    let own = e < NBULK;
    let q2 = bq2(t & 7u);
    let c2 = bc * 2 + q2;
    var b2 = 0.0; var o2 = vec4f(0.0); var w2 = vec3f(0.0);
    if (own) {
      let sl = bulkSlot(c2);
      b2 = pres[R_OFF + sl] + subSum(c2, bits[MASK_OFF + sl], Z1_OFF);
      let g2 = lidx(2u, c2);
      coarse[CB + g2] = b2;
      o2 = vec4f(coarse[COP + 4u * g2], coarse[COP + 4u * g2 + 1u], coarse[COP + 4u * g2 + 2u], coarse[COP + 4u * g2 + 3u]);
      w2 = bw2(c2);
    }
    let ti2 = bti2(q2) + 64u * slot;
    bsmooth2(own, c2, ti2, o2, w2, b2, false);
    if (own) { coarse[CEP + lidx(2u, c2)] = select(0.0, tb2[ti2], o2.x > 0.0); }
    return;
  }
  if (wg.x - NB8 >= nLiq) { return; }
  let b = bricks[B_LLIQ + wg.x - NB8];
  let bc = brickCoord(b);
  for (var j = t; j < ${6 * 6 * 6}u; j += 64u) {
    let c = bc * 4 + bhalo1(j);
    var e = 0.0;
    if (linBox(1u, c)) { e = coarse[CEP + lidx(1u, c)]; }
    tb1[j] = e;
  }
  if (t < ${4 * 4 * 4}u) { tb2[t] = 0.0; }
  workgroupBarrier();
  let q = bq1(t);
  let c1 = bc * 4 + q;
  let g = lidx(1u, c1);
  let o = vec4f(coarse[COP + 4u * g], coarse[COP + 4u * g + 1u], coarse[COP + 4u * g + 2u], coarse[COP + 4u * g + 3u]);
  var r = 0.0;
  if (o.x > 0.0) { let ti = bti1(q); r = coarse[CB + g] - (o.x * tb1[ti] - bsum1(ti, o, bw1(c1))); }
  sr[t] = r;
  workgroupBarrier();
  let own = t < 8u;
  let q2 = bq2(t);
  let c2 = bc * 2 + q2;
  var b2 = 0.0; var o2 = vec4f(0.0); var w2 = vec3f(0.0);
  if (own) {
    for (var k = 0u; k < 8u; k++) {
      let ql = 2 * q2 + vec3i(i32(k & 1u), i32((k >> 1u) & 1u), i32(k >> 2u));
      b2 += sr[u32(ql.x + 4 * (ql.y + 4 * ql.z))];
    }
    let g2 = lidx(2u, c2);
    coarse[CB + g2] = b2;
    o2 = vec4f(coarse[COP + 4u * g2], coarse[COP + 4u * g2 + 1u], coarse[COP + 4u * g2 + 2u], coarse[COP + 4u * g2 + 3u]);
    w2 = bw2(c2);
  }
  let ti2 = bti2(q2);
  bsmooth2(own, c2, ti2, o2, w2, b2, false);
  if (own) { coarse[CEP + lidx(2u, c2)] = select(0.0, tb2[ti2], o2.x > 0.0); }
}
`);

/** D2 (see 15): 8 threads, one per level-2 cell of the brick. */


/**
 * P_k: iteration k's convergence test, right after X_k measured its residual (one workgroup). It
 * opens iteration k's V-cycle (D0..D3, C, U3..U0, precond, direction) and X_(k+1) / P_(k+1) only
 * while the residual exceeds the tolerance, so the step's last iteration runs no V-cycle (standard
 * PCG: the test precedes the preconditioner solve, Saad 2003 Alg. 9.1). cfg.a = k.
 */
export const pcgCheckKernel = kernel("pour.pcgCheck", [ctrlRW, cfgU, argsRW], /* wgsl */ `
fn redSlot(rs: u32, which: u32) -> u32 { return C_RED + rs * 192u + which * 64u; }
var<workgroup> sGo: u32;
var<workgroup> sRS: array<f32, 64>;
/** Iteration k's gated arguments: slot 0 the liquid bricks, slot 1 [BULK bricks, liquid], slot 2 [BULK workgroups, liquid],
 *  slot 3 the level-4 blocks / 8, slot 4 the level-4 blocks, slot 5 the shifted fine blocks; all 0 once converged. */
fn gate(k: u32, nAct: u32, go: bool) {
  for (var s = 0u; s < 6u; s++) {
    let o = C_MG_ARGS + ${MG_STRIDE}u * k + 4u * s;
    var x = nAct + select(select(0u, NBULK, s == 1u), BULK_WG, s == 2u);
    if (s == 3u) { x = atomicLoad(&args[C_ARG_L4]); }
    if (s == 4u) { x = atomicLoad(&args[C_ARG_L4U]); }
    if (s == 5u) { x = atomicLoad(&args[C_ARG_SHIFT]); }
    atomicStore(&args[o], select(0u, x, go)); atomicStore(&args[o + 1u], 1u); atomicStore(&args[o + 2u], 1u);
  }
}
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) t: u32) {
  let k = cfg.a;
  // rho_(k-1)'s 64 partial slots, loaded at once (summed in slot order by thread 0).
  if (t < 64u && k > 0u) { sRS[t] = bitcast<f32>(atomicLoad(&ctrl[redSlot((k - 1u) & 1u, ${RZ}u) + t])); }
  workgroupBarrier();
  if (t == 0u) {
    sGo = 0u;
    if (atomicLoad(&ctrl[C_CONV]) == 0u) {
      let res = bitcast<f32>(atomicLoad(&ctrl[C_RES]));
      atomicStore(&ctrl[C_RES], 0u);
      atomicStore(&ctrl[C_LAST_RES], bitcast<u32>(res));
      atomicStore(&ctrl[C_RES_HIST + k], bitcast<u32>(res));
      if (k == 0u) { atomicStore(&ctrl[C_FIRST_RES], bitcast<u32>(res)); }
      if (res <= TOL || k >= MAXC) {
        atomicStore(&ctrl[C_CONV], 1u);
        atomicStore(&ctrl[C_CYCLES], k);
        atomicAdd(&ctrl[C_CYCLES_TOTAL], k);
        // Capped: the pressure is not converged. Latched as a failure, not only counted.
        if (res > TOL) { atomicAdd(&ctrl[C_CAP_HITS], 1u); atomicOr(&ctrl[C_FAIL], FAIL_CAP); }
        // The later iterations' dispatches (coarseSolve included) do not run: close all their gates now.
        for (var j = k; j <= MAXC; j++) { gate(j, 0u, false); if (j > k) { atomicStore(&ctrl[C_RES_HIST + j], 0u); } }
      } else {
        if (k > 0u) {
          var rho = 0.0;
          for (var j = 0u; j < 64u; j++) { rho += sRS[j]; }
          atomicStore(&ctrl[C_RHO], bitcast<u32>(rho));
        }
        gate(k, atomicLoad(&args[C_ARG_LIQ]), true);
        sGo = 1u;
      }
    } else {
      atomicStore(&ctrl[C_RES_HIST + k], 0u);
      gate(k, 0u, false);
    }
  }
  let go = workgroupUniformLoad(&sGo);
  if (go == 0u) { return; }
  // Clear dot set k & 1 for iteration k's sums (three slots a thread). Set k & 1 was last read by
  // X_(k-1); X_k (before this) read set (k - 1) & 1.
  for (var j = t; j < 192u; j += 64u) { atomicStore(&ctrl[redSlot(k & 1u, 0u) + j], 0u); }
}
`);

/**
 * C_k: the V(2,2)-cycle on levels 4..6, one workgroup, launched directly: it returns at once
 * unless P_k left iteration k open (C_CONV still 0).
 */
export const coarseSolveKernel = kernel("pour.coarseSolve", [coarseRW, bricksRO, ctrlRO], /* wgsl */ `
${COARSE_COMMON}
fn colourOf(c: vec3i) -> u32 { return (u32(c.x + c.y + c.z)) & 1u; }
const S5: u32 = ${L4_CELLS}u; const S6: u32 = ${L4_CELLS + L5_CELLS}u;
var<workgroup> sE: array<f32, ${L4_CELLS + L5_CELLS + COARSEST_CELLS}>;
var<workgroup> sR: array<f32, ${L4_CELLS}>;
fn octI(q: u32) -> vec3i { return vec3i(i32(q & 1u), i32((q >> 1u) & 1u), i32(q >> 2u)); }
/** Workgroup index of (l, c), levels 4..6 at 0 / S5 / S6. */
fn sIdx(l: u32, c: vec3i) -> u32 {
  let d = ldims(l);
  return select(select(S6, S5, l == 5u), 0u, l == 4u) + u32(c.x + d.x * (c.y + d.y * c.z));
}
/** Couplings to the -x, -y, -z neighbours (stored on those neighbours). */
fn minusC(l: u32, c: vec3i) -> vec3f {
  var m = vec3f(0.0);
  if (c.x > 0) { m.x = coarse[COP + 4u * lidx(l, c - vec3i(1, 0, 0)) + 1u]; }
  if (c.y > 0) { m.y = coarse[COP + 4u * lidx(l, c - vec3i(0, 1, 0)) + 2u]; }
  if (c.z > 0) { m.z = coarse[COP + 4u * lidx(l, c - vec3i(0, 0, 1)) + 3u]; }
  return m;
}
fn offS(l: u32, c: vec3i, o: vec4f, m: vec3f) -> f32 {
  let d = ldims(l);
  var s = 0.0;
  if (c.x + 1 < d.x) { s += o.y * sE[sIdx(l, c + vec3i(1, 0, 0))]; }
  if (c.y + 1 < d.y) { s += o.z * sE[sIdx(l, c + vec3i(0, 1, 0))]; }
  if (c.z + 1 < d.z) { s += o.w * sE[sIdx(l, c + vec3i(0, 0, 1))]; }
  if (c.x > 0) { s += m.x * sE[sIdx(l, c - vec3i(1, 0, 0))]; }
  if (c.y > 0) { s += m.y * sE[sIdx(l, c - vec3i(0, 1, 0))]; }
  if (c.z > 0) { s += m.z * sE[sIdx(l, c - vec3i(0, 0, 1))]; }
  return s;
}
const SW4: u32 = 2u; const SW5: u32 = 2u;
var<workgroup> sConv: u32;
@compute @workgroup_size(${L4_CELLS}) fn main(@builtin(local_invocation_index) t: u32) {
  if (t == 0u) { sConv = ctrl[C_CONV]; }
  if (workgroupUniformLoad(&sConv) != 0u) { return; }
  // Levels 4..6 (${L4_CELLS + L5_CELLS + COARSEST_CELLS} cells) live in workgroup memory for the whole V-cycle, one
  // level-4 cell a thread: each sweep is one workgroup barrier, not a device-memory fence. Each thread keeps its
  // cells' couplings (+ and -), rhs and pre-smoothed values in registers. Level 3 is block-smoothed across the GPU
  // by D3 / U3.
  let c4 = cellOfLevel(4u, t);
  let o4 = cop(4u, c4); let m4 = minusC(4u, c4);
  let b4 = coarse[CB + L4O + t];
  let has5 = t < ${L5_CELLS}u; let has6 = t < NC6;
  let c5 = cellOfLevel(5u, min(t, ${L5_CELLS - 1}u)); let c6 = cellOfLevel(6u, min(t, NC6 - 1u));
  var o5 = vec4f(0.0); var m5 = vec3f(0.0);
  if (has5) { o5 = cop(5u, c5); m5 = minusC(5u, c5); }
  sE[t] = 0.0;
  if (t < ${L5_CELLS + COARSEST_CELLS}u) { sE[S5 + t] = 0.0; }
  workgroupBarrier();
  // Down, level 4: pre-smooth from zero (red first), restrict the residual.
  for (var s = 0u; s < SW4; s++) {
    if (colourOf(c4) == (s & 1u) && o4.x > 0.0) { sE[t] = (b4 + offS(4u, c4, o4, m4)) / o4.x; }
    workgroupBarrier();
  }
  let ep4 = sE[t];
  sR[t] = select(0.0, b4 - (o4.x * ep4 - offS(4u, c4, o4, m4)), o4.x > 0.0);
  workgroupBarrier();
  var b5 = 0.0;
  if (has5) { for (var q = 0u; q < 8u; q++) { b5 += sR[sIdx(4u, 2 * c5 + octI(q))]; } }
  workgroupBarrier();
  for (var s = 0u; s < SW5; s++) {
    if (has5 && colourOf(c5) == (s & 1u) && o5.x > 0.0) { sE[S5 + t] = (b5 + offS(5u, c5, o5, m5)) / o5.x; }
    workgroupBarrier();
  }
  var ep5 = 0.0;
  if (has5) { ep5 = sE[S5 + t]; sR[t] = select(0.0, b5 - (o5.x * ep5 - offS(5u, c5, o5, m5)), o5.x > 0.0); }
  workgroupBarrier();
  var b6 = 0.0;
  if (has6) { for (var q = 0u; q < 8u; q++) { b6 += sR[sIdx(5u, 2 * c6 + octI(q)) - S5]; } }
  // Coarsest: the exact solve, one product with the inverse coarsenTop built (inactive rows give 0: b6 = 0).
  workgroupBarrier();
  if (has6) { sR[t] = b6; }
  workgroupBarrier();
  if (has6) {
    var e6 = 0.0;
    for (var q = 0u; q < NC6; q++) { e6 += coarse[CINV + t * NC6 + q] * sR[q]; }
    sE[S6 + t] = e6;
  }
  workgroupBarrier();
  // Up: prolong (piecewise constant) onto the pre-smoothed values, post-smooth black first.
  if (has5) { sE[S5 + t] = select(0.0, ep5 + ${MG_OMEGA} * sE[sIdx(6u, c5 / 2)], o5.x > 0.0); }
  workgroupBarrier();
  for (var s = 0u; s < SW5; s++) {
    if (has5 && colourOf(c5) == 1u - (s & 1u) && o5.x > 0.0) { sE[S5 + t] = (b5 + offS(5u, c5, o5, m5)) / o5.x; }
    workgroupBarrier();
  }
  sE[t] = select(0.0, ep4 + ${MG_OMEGA} * sE[sIdx(5u, c4 / 2)], o4.x > 0.0);
  workgroupBarrier();
  for (var s = 0u; s < SW4; s++) {
    if (colourOf(c4) == 1u - (s & 1u) && o4.x > 0.0) { sE[t] = (b4 + offS(4u, c4, o4, m4)) / o4.x; }
    workgroupBarrier();
  }
  coarse[CE + L4O + t] = sE[t];
}
`);

/** Level-4 block q4 (2x2x2 level-3 cells) of the level-4 block list. */
const L4_BLOCK_WGSL = /* wgsl */ `
fn l4Origin(q4: u32) -> vec3i { return 2 * vec3i(i32(q4 % 8u), i32((q4 / 8u) % 16u), i32(q4 / 128u)); }
`;
/**
 * D3_k / U3_k: level-3 block smoothing over the level-4 blocks (2x2x2 level-3
 * cells, 4 red-black sweeps inside), all blocks at once (block Jacobi, the
 * additive Schwarz smoother: Smith, Bjorstad & Gropp 1996, Domain Decomposition
 * ch. 1; eight blocks a 64-thread workgroup). Pre-smoothing from zero has a zero
 * halo, so each block is independent, and D3 also forms its own rhs (the level-2
 * residual restricted to the block's level-3 cells) in the same workgroup; then
 * the residual is restricted to level 4.
 * Post-smoothing reads every neighbour at its pre-smoothed plus prolonged value,
 * with the inner sweeps in the adjoint order, so the cycle stays symmetric.
 * (Two block colours, multiplicative, took 0.15 fewer cycles a step but two
 * more dispatches per cycle each way: a net loss.)
 */
const L3_BLOCK_WGSL = /* wgsl */ `
${levelWGSL()}
${blockWGSL(3, 8)}
${L4_BLOCK_WGSL}
fn cop3(c: vec3i) -> vec4f { let g = lidx(3u, c); return vec4f(coarse[COP + 4u * g], coarse[COP + 4u * g + 1u], coarse[COP + 4u * g + 2u], coarse[COP + 4u * g + 3u]); }
var<workgroup> sN: u32;
/** The level-4 block of slot (eight a workgroup), clamped to the list (n = its length). */
fn blockOrigin(wgx: u32, slot: u32, n: u32) -> vec3i { return l4Origin(bricks[B_L4LIST + min(wgx * 8u + slot, n - 1u)]); }
`;
export const mgDown3Kernel = kernel("pour.mgDown3", [bricksRO, coarseRW], /* wgsl */ `
${COARSE_COMMON}
${blockWGSL(3)}
${L4_BLOCK_WGSL}
fn cop3(c: vec3i) -> vec4f { return cop(3u, c); }
var<workgroup> sr: array<f32, 64>;
var<workgroup> sb: array<f32, 8>;
// D2 + D3 by level-4 block (one workgroup each): the level-2 residuals of the block's eight bricks (eight
// level-2 cells each) restricted to their level-3 cells, then the block's level-3 pre-smoothing from zero.
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let origin = l4Origin(bricks[B_L4LIST + wg.x]);
  let slot = t >> 3u; let q = t & 7u;
  let c3 = origin + bq3(slot);
  sr[t] = residC(2u, c3 * 2 + vec3i(i32(q & 1u), i32((q >> 1u) & 1u), i32(q >> 2u)), CEP);
  tb3[t] = 0.0;
  workgroupBarrier();
  if (q == 0u) {
    var s = 0.0;
    for (var k = 0u; k < 8u; k++) { s += sr[t + k]; }
    coarse[CB + lidx(3u, c3)] = s;
    sb[slot] = s;
  }
  workgroupBarrier();
  let own = t < 8u;
  let qq = bq3(t & 7u);
  let c = origin + qq;
  let g = lidx(3u, c);
  let o = cop3(c);
  let ti = bti3(qq);
  bsmooth3(own, c, ti, o, bw3(c), sb[t & 7u], false);
  if (own) { coarse[CEP + g] = select(0.0, tb3[ti], o.x > 0.0); }
}
`);
/** D3r_k: level-3 residual of each level-4 block restricted to level 4. */
export const mgRestrict3Kernel = kernel("pour.mgRestrict3", [bricksRO, coarseRW, ctrlRO], /* wgsl */ `
${L3_BLOCK_WGSL}
var<workgroup> sr: array<f32, 64>;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  if (t == 0u) { sN = ctrl[C_N_L4]; }
  let n = workgroupUniformLoad(&sN);
  if (wg.x * 8u >= n) { return; }
  for (var j = t; j < 512u; j += 64u) {
    let c = blockOrigin(wg.x, j / 64u, n) + bhalo3(j % 64u);
    var e = 0.0;
    if (linBox(3u, c) && cop3(c).x > 0.0) { e = coarse[CEP + lidx(3u, c)]; }
    tb3[j] = e;
  }
  workgroupBarrier();
  let slot = t >> 3u;
  let origin = blockOrigin(wg.x, slot, n);
  let q = bq3(t & 7u);
  let c = origin + q;
  let o = cop3(c);
  let ti = bti3(q) + 64u * slot;
  sr[t] = select(0.0, coarse[CB + lidx(3u, c)] - (o.x * tb3[ti] - bsum3(ti, o, bw3(c))), o.x > 0.0);
  workgroupBarrier();
  if ((t & 7u) == 0u && wg.x * 8u + slot < n) {
    var s = 0.0;
    for (var k = 0u; k < 8u; k++) { s += sr[t + k]; }
    coarse[CB + lidx(4u, origin / 2)] = s;
  }
}
`);
export const mgUp3Kernel = kernel("pour.mgUp3", [bricksRO, coarseRW, ctrlRO], /* wgsl */ `
${L3_BLOCK_WGSL}
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  if (t == 0u) { sN = ctrl[C_N_L4]; }
  let n = workgroupUniformLoad(&sN);
  if (wg.x * 8u >= n) { return; }
  for (var j = t; j < 512u; j += 64u) {
    let c = blockOrigin(wg.x, j / 64u, n) + bhalo3(j % 64u);
    var e = 0.0;
    if (linBox(3u, c) && cop3(c).x > 0.0) { e = coarse[CEP + lidx(3u, c)] + ${MG_OMEGA} * coarse[CE + lidx(4u, c / 2)]; }
    tb3[j] = e;
  }
  workgroupBarrier();
  let slot = t >> 3u;
  let own = wg.x * 8u + slot < n;
  let q = bq3(t & 7u);
  let c = blockOrigin(wg.x, slot, n) + q;
  let g = lidx(3u, c);
  let o = cop3(c);
  let ti = bti3(q) + 64u * slot;
  bsmooth3(own, c, ti, o, bw3(c), coarse[CB + g], true);
  if (own) { coarse[CE + g] = select(0.0, tb3[ti], o.x > 0.0); }
}
`);

/** U2 / U1: post-smoothing of level L on the brick's block after prolongation from L+1 (U2 over [BULK, active]). */
function upKernel(L: 1 | 2): KernelSource {
  const s = L === 1 ? 4 : 2, T = s + 2, threads = s * s * s;
  if (L === 2) return kernel("pour.mgUp2", [bricksRO, coarseRW, ctrlRO], /* wgsl */ `
${levelWGSL()}
${blockWGSL(2, 8)}
var<workgroup> sN: u32;
// Eight bricks of [BULK, liquid] a workgroup, 8 threads each (launched over NBULK + n).
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  if (t == 0u) { sN = NBULK + ctrl[C_N_LIQ]; }
  let n = workgroupUniformLoad(&sN);
  if (wg.x * 8u >= n) { return; }
  for (var j = t; j < 512u; j += 64u) {
    let c = brickCoord(bricks[B_LBULK2 + min(wg.x * 8u + j / 64u, n - 1u)]) * 2 + bhalo2(j % 64u);
    var e = 0.0;
    if (linBox(2u, c)) {
      let g = lidx(2u, c);
      if (coarse[COP + 4u * g] > 0.0) { e = coarse[CEP + g] + ${MG_OMEGA} * coarse[CE + lidx(3u, c / 2)]; }
    }
    tb2[j] = e;
  }
  workgroupBarrier();
  let slot = t >> 3u;
  let e = wg.x * 8u + slot;
  let own = e < n;
  let q = bq2(t & 7u);
  let c = brickCoord(bricks[B_LBULK2 + min(e, n - 1u)]) * 2 + q;
  let g = lidx(2u, c);
  let o = vec4f(coarse[COP + 4u * g], coarse[COP + 4u * g + 1u], coarse[COP + 4u * g + 2u], coarse[COP + 4u * g + 3u]);
  let ti = bti2(q) + 64u * slot;
  bsmooth2(own, c, ti, o, bw2(c), coarse[CB + g], true);
  if (own) { coarse[CE + g] = select(0.0, tb2[ti], o.x > 0.0); }
}
`);
  return kernel(`pour.mgUp${L}`, [bricksRO, coarseRW], /* wgsl */ `
${levelWGSL()}
${blockWGSL(L)}
@compute @workgroup_size(${threads}) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let b = bricks[B_LLIQ + wg.x];
  let origin = brickCoord(b) * ${s};
  for (var j = t; j < ${T * T * T}u; j += ${threads}u) {
    let c = origin + bhalo${L}(j);
    var e = 0.0;
    if (linBox(${L}u, c)) {
      let g = lidx(${L}u, c);
      if (coarse[COP + 4u * g] > 0.0) { e = coarse[CEP + g] + ${MG_OMEGA} * coarse[CE + lidx(${L + 1}u, c / 2)]; }
    }
    tb${L}[j] = e;
  }
  workgroupBarrier();
  let q = bq${L}(t);
  let c = origin + q;
  let g = lidx(${L}u, c);
  let o = vec4f(coarse[COP + 4u * g], coarse[COP + 4u * g + 1u], coarse[COP + 4u * g + 2u], coarse[COP + 4u * g + 3u]);
  let ti = bti${L}(q);
  bsmooth${L}(true, c, ti, o, bw${L}(c), coarse[CB + g], true);
  coarse[CE + g] = select(0.0, tb${L}[ti], o.x > 0.0);
}
`);
}
export const mgUp2Kernel = upKernel(2);
export const mgUp1Kernel = upKernel(1);

/** U0_k (see 15). cfg.a = k. */
export const pcgPrecondKernel = kernel("pour.pcgPrecond", [ctrlRW, bricksRO, opRO, bitsRO, presRW, coarseRW, cfgU], /* wgsl */ `
${levelWGSL()}
${PCG_WGSL}
${bulkWGSL(false)}
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let k = cfg.a;
  if (wg.x < BULK_WG) {
    // BULK coarse cells: z is their level-2 correction.
    let j = wg.x * 512u + l;
    var rz = 0.0;
    if (j < NBULK * 8u) {
      let kc = bulkCell(j);
      let sl = bulkSlot(kc);
      let z = coarse[CE + lidx(2u, kc)];
      pres[Z_OFF + sl] = z;
      rz = pres[R_OFF + sl] * z;
    }
    sRedA[l] = rz;
    let dot = foldSumA(l);
    if (l == 0u) { redAdd(k & 1u, ${RZ}u, wg.x, dot); }
    return;
  }
  let b = bricks[B_LLIQ + wg.x - BULK_WG];
  let o = brickCoord(b) * 8;
  for (var t = l; t < 1000u; t += 512u) {
    let c = o + tileCoord(t);
    var v = 0.0;
    if (inBox(c)) {
      let ci = cidx(c);
      // U0B's iterate: the pre-smoothed values plus the level-1 correction, post-smoothed on the shifted partition.
      if (op[DIAG_OFF + ci] > 0.0) { v = pres[Z1_OFF + ci]; }
    }
    tp[t] = v;
  }
  workgroupBarrier();
  let lc = localCoord(l);
  let c = o + lc;
  let i = (b << 9u) | l;
  let diag = op[DIAG_OFF + i];
  let m = bits[MASK_OFF + i];
  // Post-smoothing against the final coarse values of BULK neighbours (fixed during the sweeps).
  var r = pres[R_OFF + i];
  var rs = r;
  if ((m >> 6u) != 0u) {
    for (var kk = 0u; kk < 6u; kk++) {
      if (((m >> (kk + 6u)) & 1u) == 0u) { continue; }
      rs += SUBC * coarse[CE + lidx(2u, (c + select(-1, 1, (kk & 1u) == 1u) * unitAxis(kk >> 1u)) / 4)];
    }
  }
  let ti = tileIndex(lc);
  fineSmooth(c, ti, diag, m, rs, true);
  let z = select(0.0, tp[ti], diag > 0.0);
  pres[Z_OFF + i] = z;
  sRedA[l] = r * z;
  let dot = foldSumA(l);
  if (l == 0u) { redAdd(k & 1u, ${RZ}u, wg.x, dot); }
}
`);

/** P_k (see 15). cfg.a = k. */
export const pcgDirectionKernel = kernel("pour.pcgDirection", [ctrlRW, bricksRO, opRO, bitsRO, presRW, cfgU], /* wgsl */ `
${PCG_WGSL}
${bulkWGSL(false)}
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let k = cfg.a;
  var beta = 0.0;
  if (k > 0u) {
    let rho = bitcast<f32>(atomicLoad(&ctrl[C_RHO]));
    beta = select(0.0, redSums(k & 1u, ${RZ}u, ${RZ}u, l).x / rho, rho != 0.0);
  }
  var pq = 0.0;
  if (wg.x < BULK_WG) {
    let j = wg.x * 512u + l;
    if (j < NBULK * 8u) {
      let kc = bulkCell(j);
      let sl = bulkSlot(kc);
      var p = pres[Z_OFF + sl];
      var q = applyBulk(kc, sl, Z_OFF);
      if (k > 0u) { p += beta * pres[PP_OFF + sl]; q += beta * pres[Q_OFF + sl]; }
      pres[PP_OFF + sl] = p; pres[Q_OFF + sl] = q;
      pq = p * q;
    }
  } else {
  let b = bricks[B_LLIQ + wg.x - BULK_WG];
  let c = brickCoord(b) * 8 + localCoord(l);
  let i = (b << 9u) | l;
  if (op[DIAG_OFF + i] > 0.0) {
    var p = pres[Z_OFF + i];
    var q = applyA(c, i, Z_OFF);
    if (k > 0u) { p += beta * pres[PP_OFF + i]; q += beta * pres[Q_OFF + i]; }
    pres[PP_OFF + i] = p; pres[Q_OFF + i] = q;
    pq = p * q;
  } else { pres[PP_OFF + i] = 0.0; pres[Q_OFF + i] = 0.0; }
  }
  sRedA[l] = pq;
  let dot = foldSumA(l);
  if (l == 0u) { redAdd(k & 1u, ${PQ}u, wg.x, dot); }
}
`);

// ---------------------------------------------------------------------------
// 16. Projection: U = Us - grad P on liquid faces (ghost-fluid gradient at
//     liquid/air faces); other faces become unknown (zero) for extension.
// ---------------------------------------------------------------------------
const projectKernel = (n: PourInflow) => kernel("pour.project", [bricksRO, phisRO, velRW, presRO, bitsRW, bulkRW], /* wgsl */ `
${bulkWGSL(false)}
${inflowWGSL(n, false)}
/** Pressure of fine cell f for an interface face: its own when liquid, the atmosphere's (0) in air. */
fn pFine(f: vec3i) -> f32 { let fi = cidx(f); return select(0.0, pres[fi], phis[PHIC_OFF + fi] < 0.0); }
/** BULK coarse cell j: its coarse low faces (between two BULK cells, over 4 cells) and the fine interface faces stored in it (over 2.5). */
fn bulkProject(j: u32) {
  let k = bulkCell(j);
  let pk = pres[bulkSlot(k)];
  for (var a = 0u; a < 3u; a++) {
    let e = unitAxis(a);
    let kn = k - e;
    if (coarseBulk(kn)) {
      let ci = a * NCL + clIndex(k);
      bulk[BU_OFF + ci] = bulk[BUS_OFF + ci] - (pk - pres[bulkSlot(kn)]) * 0.25;
    } else {
      let t1 = unitAxis((a + 1u) % 3u); let t2 = unitAxis((a + 2u) % 3u);
      for (var u = 0; u < 4; u++) { for (var v = 0; v < 4; v++) {
        let c = 4 * k + u * t1 + v * t2;
        var val = 0.0;
        if (fineOpen(c - e)) { val = vel[US_OFF + a * N + cidx(c)] - (pk - pFine(c - e)) / 2.5; }
        vel[U_OFF + a * N + cidx(c)] = val;
      } }
    }
  }
}
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  if (wg.x < BULK_WG) {
    let j = wg.x * 512u + l;
    if (j < NBULK * 8u) { bulkProject(j); }
    return;
  }
  let b = bricks[B_LACT + wg.x - BULK_WG];
  let c = brickCoord(b) * 8 + localCoord(l);
  let i = (b << 9u) | l;
  let po = 0u;
  let phB = phis[PHIC_OFF + i];
  let lb = cellOpen(c) && phB < 0.0;
  var known = 0u;
  for (var a = 0u; a < 3u; a++) {
    var u = 0.0;
    if (faceOpen(c, a) && cellBulk(c - unitAxis(a))) {
      // Interface face below/behind a BULK coarse cell (the composite gradient over 2.5 cells).
      let pb = select(0.0, pres[po + i], lb);
      u = vel[US_OFF + a * N + i] - (pb - pres[bulkSlot((c - unitAxis(a)) / 4)]) / 2.5;
      known |= 1u << a;
    } else if (faceOpen(c, a)) {
      let ca = c - unitAxis(a);
      let ia = cidx(ca);
      let phA = phis[PHIC_OFF + ia];
      let la = phA < 0.0;
      if (la || lb) {
        let pa = select(0.0, pres[po + ia], la);
        let pb = select(0.0, pres[po + i], lb);
        var theta = 1.0;
        if (!(la && lb)) {
          if (la) { theta = max(phA / (phA - phB), THETA_MIN); } else { theta = max(phB / (phB - phA), THETA_MIN); }
        }
        u = vel[US_OFF + a * N + i] - (pb - pa) / theta;
        known |= 1u << a;
      }
      // The nozzle's faces (the brick engine's applyInflowVelocity after projection): the low face of c is the
      // +face of q = c - e_a.
      let q = c - unitAxis(a);
      if (inVelocityCell(q)) { u = inProjected(q, a, u); known |= 1u << a; }
    }
    vel[U_OFF + a * N + i] = u;
  }
  // Bits 6..8 keep the projection's own known faces past the extension (which rewrites 0..5): G2P's FLIP
  // delta is taken only where this step projected a value.
  bits[KNOWN_OFF + i] = known | (known << 6u);
}
`);

// ---------------------------------------------------------------------------
// 17. Velocity extension into air: layered averaging of known neighbours
//     (Bridson 2015 §4.3), 8 sweeps per launch on a staged brick + halo, per
//     component. Faces not between two open cells stay zero and are not sources.
//     cfg.a = in velocity offset, cfg.b = out offset, cfg.c = in known shift,
//     cfg.d = out known shift. DEEP neighbours (never extended) are read from U.
// ---------------------------------------------------------------------------
export const extendKernel = kernel("pour.extend", [bricksRO, velRW, bitsRW, cfgU], /* wgsl */ `
// The three components' tiles side by side (3 x 1000), ping-pong A/B, one barrier a sweep for all
// three. A face's state rides in its value: UNK (not yet known) and SHUT (closed, never a source)
// are sentinels no velocity takes; any other value is known.
const UNK: f32 = -3.0e38;
const SHUT: f32 = 3.0e38;
var<workgroup> vA: array<f32, 3000>;
var<workgroup> vB: array<f32, 3000>;
var<workgroup> sKnown: atomic<u32>;
var<workgroup> sKnownU: u32;
fn isKnown(v: f32) -> bool { return v > UNK && v < SHUT; }
fn relax(src: ptr<workgroup, array<f32, 3000>>, k: u32) -> f32 {
  let v = (*src)[k];
  if (v != UNK) { return v; }
  var sum = 0.0; var n = 0.0;
  let nb = array<u32, 6>(k - 1u, k + 1u, k - 10u, k + 10u, k - 100u, k + 100u);
  for (var q = 0u; q < 6u; q++) { let w = (*src)[nb[q]]; if (isKnown(w)) { sum += w; n += 1.0; } }
  return select(UNK, sum / max(n, 1.0), n > 0.0);
}
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let b = bricks[B_LSURF + wg.x];
  let o = brickCoord(b) * 8;
  let lc = localCoord(l);
  let i = (b << 9u) | l;
  let ti = tileIndex(lc);
  if (l == 0u) { atomicStore(&sKnown, 0u); }
  workgroupBarrier();
  var anyKnown = false; var anyUnknown = false;
  for (var t = l; t < 1000u; t += 512u) {
    let q = o + tileCoord(t);
    var info = 0u; var deep = false; var qi = 0u;
    let inside = inBox(q);
    if (inside) {
      qi = cidx(q);
      deep = (bricks[B_CLASS + brickOfCell(q)] & CLASS_DEEP) != 0u;
      info = bits[KNOWN_OFF + qi];
    }
    for (var a = 0u; a < 3u; a++) {
      var v = SHUT;
      // fgFaceOpen: a face is extended, or a source, only between two open cells (the brick engine's uExtend seed
      // code 3: a wall face and a face inside the glass are SHUT, zero and never a source). Extended into the
      // glass, the film's velocity reached every sample within half a cell of the wall, and films climbed it.
      if (inside && faceOpen(q, a)) {
        let k = (info >> select(cfg.c + a, a, deep)) & 1u;
        v = select(UNK, vel[select(cfg.a, U_OFF, deep) + a * N + qi], k == 1u);
        anyKnown = anyKnown || k == 1u;
        anyUnknown = anyUnknown || k == 0u;
      }
      // Both ping-pong tiles take the load: the sweeps write only the brick's own faces, so the halo each
      // reads must hold the neighbours' values (a zero-initialised halo in B read as known zeros, and every
      // second sweep pulled the extension back toward rest: it reached two cells ahead of a falling front).
      vA[a * 1000u + t] = v; vB[a * 1000u + t] = v;
    }
  }
  if (anyKnown) { atomicOr(&sKnown, 1u); }
  if (anyUnknown) { atomicOr(&sKnown, 2u); }
  workgroupBarrier();
  if (l == 0u) { sKnownU = atomicLoad(&sKnown); }
  // Without a known face in the tile, or without an unknown one (relax keeps every known or
  // closed value), the sweeps change nothing.
  let sweeps = select(0u, 8u, workgroupUniformLoad(&sKnownU) == 3u);
  for (var s = 0u; s < sweeps; s += 2u) {
    for (var a = 0u; a < 3u; a++) { vB[a * 1000u + ti] = relax(&vA, a * 1000u + ti); }
    workgroupBarrier();
    for (var a = 0u; a < 3u; a++) { vA[a * 1000u + ti] = relax(&vB, a * 1000u + ti); }
    workgroupBarrier();
  }
  var outBits = 0u;
  for (var a = 0u; a < 3u; a++) {
    let v = vA[a * 1000u + ti];
    let known = isKnown(v);
    vel[cfg.b + a * N + i] = select(0.0, v, known);
    outBits |= select(0u, 1u, known) << (cfg.d + a);
  }
  let keep = bits[KNOWN_OFF + i] & ~(7u << cfg.d);
  bits[KNOWN_OFF + i] = keep | outBits;
}
`);

// ---------------------------------------------------------------------------
// 18. G2P (FLIP 0.95 / PIC 0.05) and particle advection (Heun RK3 through the
//     extended grid velocity), exact projection out of the baked solids.
//     cfg.a = parity (particles live in dst). Thread 0 resets the next step's
//     list counters, advances the step counter and publishes the next step's
//     inflow words (row s - 1 of the schedule for step s; off past its end).
// ---------------------------------------------------------------------------
const g2pKernel = (n: PourInflow) => kernel("pour.g2p", [ctrlRW, bricksRW, partsRW, velRO, cfgU, argsRW, phisRO, bitsRO, pgridRO], /* wgsl */ `
const IN_STEPS: u32 = ${n.steps}u;
const IN_FLOW: f32 = ${n.flow};
${velocitySampler("velU", "vel", "U_OFF")}
${SUPPORT_WGSL}
// The FLIP delta of face (a, c), U - US less this step's gravity (US carries it), where the projection set the
// face; elsewhere (extended, solid-interior or far-air faces, whose US is no velocity of the field) only the body
// force. Taking U - US there read an extended U against a zero US: a particle a quarter cell from the glass's
// inner face, whose stencil holds the wall's interior faces, gained a share of the film's own velocity every step,
// and films ran up the glass at 20-100 m/s.
fn deltaFace(a: u32, c: vec3i) -> f32 {
  let g = select(0.0, GDT, a == 1u);
  if (c[a] < 0 || c[a] >= dimOf(a)) { return -g; }
  let i = cidx(clampV(c));
  if (((bits[KNOWN_OFF + i] >> (6u + a)) & 1u) == 0u) { return -g; }
  return vel[U_OFF + a * N + i] - vel[US_OFF + a * N + i] - g;
}
fn deltaComp(a: u32, p: vec3f) -> f32 {
  let q = p - vec3f(0.5) + 0.5 * vec3f(unitAxis(a));
  let b = vec3i(floor(q)); let t = q - floor(q);
  let x00 = mix(deltaFace(a, b), deltaFace(a, b + vec3i(1, 0, 0)), t.x);
  let x10 = mix(deltaFace(a, b + vec3i(0, 1, 0)), deltaFace(a, b + vec3i(1, 1, 0)), t.x);
  let x01 = mix(deltaFace(a, b + vec3i(0, 0, 1)), deltaFace(a, b + vec3i(1, 0, 1)), t.x);
  let x11 = mix(deltaFace(a, b + vec3i(0, 1, 1)), deltaFace(a, b + vec3i(1, 1, 1)), t.x);
  return mix(mix(x00, x10, t.y), mix(x01, x11, t.y), t.z);
}
fn flipDelta(p: vec3f) -> vec3f { return vec3f(deltaComp(0u, p), deltaComp(1u, p), deltaComp(2u, p)); }
${PARTICLE_MOVE_WGSL}
// The brick engine's mass rules (pMove, crowdedCells "merge", Ferstl et al. 2016 §3.3), on the pre-move runs: a
// cell deeper than one cell retires the records past its mass bound (8 past 1.5 cells deep, else 16), and one whose
// run still holds more than the cap (6 past 1.5, else 20) merges rank r's later strided records into rank r < cap
// up to mass 8; a merged record that ends shallow, or in a cell holding fewer records than its seed target, splits
// in two a quarter cell apart (nbfSplitWanted, nbfSplitAlong/A/B).
const MASS_MAX: f32 = 8.0;
fn cellDepth(c: vec3i) -> f32 { return phis[PHIC_OFF + cidx(clampV(c))]; }
fn splitAlong(q: vec3f) -> vec3f {
  let bits = bitcast<vec3u>(q); let hash = (bits.x * 73856093u) ^ (bits.y * 19349663u) ^ (bits.z * 83492791u);
  return normalize(vec3f(f32(hash & 255u), f32((hash >> 8u) & 255u), f32((hash >> 16u) & 255u)) - 127.5);
}
fn markBrick(x: vec3f, v: vec3f) {
  let pb = brickOfCell(vec3i(floor(x)));
  if (atomicLoad(&bricks[B_PCOUNT + pb]) == 0u) { atomicStore(&bricks[B_PCOUNT + pb], 1u); }
  let fb = fastBits(v); if (fb != 0u && (atomicLoad(&bricks[B_FAST + pb]) & fb) != fb) { atomicOr(&bricks[B_FAST + pb], fb); }
}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i == 0u) {
    for (var k = 0u; k < 4u; k++) { atomicStore(&ctrl[C_N_ACTIVE + k], atomicLoad(&args[4u * k])); }
    atomicStore(&args[C_ARG_ACTIVE], 0u); atomicStore(&args[C_ARG_SURF], 0u); atomicStore(&args[C_ARG_TRACE], 0u);
    atomicStore(&args[C_ARG_PART], 0u); atomicStore(&args[C_ARG_CLEAR], 0u);
    atomicStore(&args[C_ARG_ACTB], BULK_WG); atomicStore(&args[C_ARG_ACTBB], (NBULK + 7u) / 8u);
    atomicStore(&ctrl[C_PCOUNT + cfg.a], 0u);
    // This step's drop joins the volume budget after the step's rate (uAdvect, beginStep's updateBudget).
    let strength = bitcast<f32>(atomicLoad(&bricks[B_INNOW]));
    if (strength > 0.0) {
      atomicStore(&ctrl[C_VOL_BUDGET], bitcast<u32>(bitcast<f32>(atomicLoad(&ctrl[C_VOL_BUDGET])) + IN_FLOW * strength * DT_S));
    }
    let row = atomicAdd(&ctrl[C_STEP], 1u);
    for (var k = 0u; k < 3u; k++) {
      atomicStore(&bricks[B_INNOW + k], select(0u, atomicLoad(&bricks[B_INFLOW + 3u * row + k]), row < IN_STEPS));
    }
  }
  let n = min(atomicLoad(&ctrl[C_PCOUNT + 1u - cfg.a]), CAP);
  if (i >= n) { return; }
  let base = (1u - cfg.a) * CAP * 2u;
  let sb = cfg.a * CAP * 2u;
  let x0 = parts[base + i].xyz;
  var x = x0;
  var v = parts[base + CAP + i].xyz;
  // A seed's first move takes the grid's velocity (nbfG2P's first: PIC). Its own is the last step's field at the
  // site, and the FLIP delta is measured against this step's transfer, which already holds what moved it: a plug
  // seed kept the stale zero of the field before the nozzle ran, and the stream's first liquid fell off the lip.
  let first = parts[base + CAP + i].w > 0.5;
  // The run masses come from the separation's copies (sb, w): records are rewritten here. A retired record keeps its
  // position (a gatherer may read it) with mass 0.
  var m = parts[sb + i].w;
  {
    let cc = vec3i(floor(x0)); let pc0 = pidx(cc);
    let start = pgrid[PN + pc0]; let cnt = pgrid[PN2 + pc0];
    let deep = cellDepth(cc);
    if (deep < -1.0 && i >= start && i < start + cnt) {
      var end = start + cnt;
      var held = 0.0; let bound = select(16.0, 8.0, deep <= -1.5);
      for (var j = start; j < start + cnt; j++) {
        let mj = parts[sb + j].w;
        if (held > 0.0 && held + mj > bound) { end = j; break; }
        held += mj;
      }
      if (i >= end) { parts[base + i] = vec4f(x0, 0.0); atomicAdd(&ctrl[C_DEL_CAP], 1u); return; }
      let cap = select(20u, 6u, deep <= -1.5);
      if (end - start > cap) {
        if (i - start < cap) {
          var sm = m; var sx = m * x; var sv = m * v;
          for (var j = i + cap; j < end; j += cap) {
            let mj = parts[sb + j].w;
            if (sm + mj > MASS_MAX) { continue; }
            sm += mj; sx += mj * parts[base + j].xyz; sv += mj * parts[base + CAP + j].xyz;
          }
          x = sx / sm; v = sv / sm; m = sm;
        } else {
          let k = start + (i - start) % cap;
          var mm = parts[sb + k].w;
          for (var j = k + cap; j < i; j += cap) { let mj = parts[sb + j].w; if (mm + mj <= MASS_MAX) { mm += mj; } }
          if (mm + m <= MASS_MAX) { parts[base + i] = vec4f(x0, 0.0); atomicAdd(&ctrl[C_MERGED], 1u); return; }
        }
      }
    }
  }
  // Supported: a liquid cell among the 2x2x2 whose centres enclose x (the
  // cells the interpolation reads). Unsupported particles (spray, a film away
  // from any liquid cell) have no pressure acting on them and move
  // ballistically with their own momentum under gravity (FLIP in air,
  // Bridson 2015 §7.6.2; the reference's "ballistic" samples); the extended
  // grid velocity there is no force, and following it left splash particles
  // hovering against the walls.
  let supported = gridSupported(x);
  if (supported) {
    let un = velU(x);
    v = select(FLIP * (v + flipDelta(x)) + (1.0 - FLIP) * un, un, first);
    moveParticle(&x, &v, un);
  } else {
    if (first) { v = velU(x); }
    let v1 = v - vec3f(0.0, GDT, 0.0);
    let xn = x + 0.5 * CFL * (v + v1);
    v = v1;
    x = moveEnd(x, xn, &v);
  }
  // The separation shift, walked as a move (it never crosses the glass, but a shift into it would be pushed out
  // along the shortest axis).
  // Not for a ballistic record (the brick engine's pSurfaceShift is inside its !ballistic branch).
  if (supported) {
    var vs = v;
    x = moveEnd(x, x + parts[sb + i].xyz, &vs);
  }
  // The split (nbFinish): the endpoint cell's pre-move run against its seed target.
  if (m > 1.0 && supported) {
    let ec = vec3i(floor(x));
    if (inBox(ec)) {
      let heldN = pgrid[PN2 + pidx(ec)];
      let seeded = select(8u, 4u, cellDepth(ec) <= -1.5);
      if (heldN < seeded && m * f32(seeded) > 8.0) {
        let along = splitAlong(x); let kept = ceil(0.5 * m); let moved = m - kept;
        let xa = x + 0.25 * (moved / m) * along; let xb = x - 0.25 * (kept / m) * along;
        if (cellOpen(vec3i(floor(xa))) && cellOpen(vec3i(floor(xb)))) {
          let slot = n + atomicAdd(&ctrl[C_SPLIT], 1u);
          if (slot < CAP) {
            parts[base + slot] = vec4f(xb, moved); parts[base + CAP + slot] = vec4f(v, 0.0);
            markBrick(xb, v);
            x = xa; m = kept;
          } else { atomicOr(&ctrl[C_FAIL], FAIL_POOL); atomicAdd(&ctrl[C_OVERFLOW], 1u); }
        }
      }
    }
  }
  parts[base + i] = vec4f(x, m);
  parts[base + CAP + i] = vec4f(v, 0.0);
  // B_PCOUNT is read only as nonzero (classify, combine): set it once rather than counting every particle on one
  // contended word; likewise OR fast bits only when one is missing.
  markBrick(x, v);
}
`);

// ---------------------------------------------------------------------------
// Diagnostics (on demand, never per step): per-brick liquid volume from the
// six-tetrahedron fill over open cells; optional texture publication (init).
// cfg.a = 1 to also write the fill and render textures for every brick;
// cfg.b = 1 to measure phiT (the ledger's mid-step stages) instead of phi.
// ---------------------------------------------------------------------------
export const volumeKernel = kernel("pour.volume", [bricksRO, phisRO, { name: "vsum", kind: "rw", type: "array<f32>" }, volW, cfgU, surfW], /* wgsl */ `
${levelSetFillWGSL}
${SURF_WGSL}
fn ph(v: vec3i) -> f32 { return phis[select(PHI_OFF, PHIT_OFF, cfg.b == 1u) + cidx(clampV(v))]; }
var<workgroup> red: array<f32, 512>;
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let b = wg.x + 128u * wg.y;
  let c = brickCoord(b) * 8 + localCoord(l);
  var f = 0.0;
  if (cellOpen(c)) {
    var v: array<f32, 8>;
    for (var k = 0u; k < 8u; k++) { v[k] = ph(c + vec3i(i32(k & 1u), i32((k >> 1u) & 1u), i32(k >> 2u))); }
    f = fill(v);
  }
  if (cfg.a == 1u) { textureStore(vol, c, vec4f(f, 0.0, 0.0, 0.0)); publishSurf(c, ph(c)); }
  red[l] = f;
  workgroupBarrier();
  for (var d = 256u; d > 0u; d = d >> 1u) {
    if (l < d) { red[l] += red[l + d]; }
    workgroupBarrier();
  }
  if (l == 0u) { vsum[b] = red[0]; }
}
`);

/** Every kernel of the step, the nozzle's baked into those that read it. */
export function pourKernels(n: PourInflow) {
  return {
    classify: classifyKernel(n), clear: clearKernel, advectPhi: advectPhiKernel(n), count: countKernel, seed: seedKernel(n),
    scan: scanKernel, scatter: scatterKernel, separate: separateKernel, splat: splatKernel, combine: combineKernel(n), massFit: massFitKernel(n),
    redistInit: redistInitKernel, redistSweep: redistSweepKernel, p2g: p2gKernel, velocity: velocityKernel(n),
    velocityTrace: velocityTraceKernel, operator: operatorKernel, coarsenBrick: coarsenBrickKernel, coarsenTop: coarsenTopKernel,
    pcgShiftDown: pcgShiftDownKernel, pcgShiftUp: pcgShiftUpKernel, pcgGuess: pcgGuessKernel, pcgUpdate: pcgUpdateKernel,
    mgDown0: mgDown0Kernel, mgDown1: mgDown1Kernel, mgDown3: mgDown3Kernel, mgRestrict3: mgRestrict3Kernel, pcgCheck: pcgCheckKernel,
    coarseSolve: coarseSolveKernel, mgUp3: mgUp3Kernel, mgUp2: mgUp2Kernel, mgUp1: mgUp1Kernel, pcgPrecond: pcgPrecondKernel,
    pcgDirection: pcgDirectionKernel, project: projectKernel(n), extend: extendKernel, g2p: g2pKernel(n), volume: volumeKernel,
  } satisfies Record<string, KernelSource>;
}
export type PourKernels = ReturnType<typeof pourKernels>;
