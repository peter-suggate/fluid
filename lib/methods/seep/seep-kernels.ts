/**
 * Seep solver kernels (WGSL). Lane S, docs/plans/seep-rewrite-2026-11-14.md §2.2.
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
import { BULK_BX1, BULK_BY1, BZ, CELL_M, THROAT_BLOCK_HI, THROAT_BLOCK_LO } from "./seep-layout";
import {
  AGGREGATION_OC, COARSE_CELLS, COARSEST_CELLS, COMMON_WGSL, LEVEL_OFFSET, MAX_CYCLES, MG_STRIDE, PRESSURE_TOLERANCE,
  declare, levelDims, phiSampler, velocitySampler, type BindingSpec,
} from "./seep-wgsl";

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
// BULK is static: the bake marks exactly the bricks of seep-layout's bulkBrick() CLASS_BULK
// (tests/seep-layout.test.ts), so the test is its closed form on brick coordinates, no class read.
fn bulkAt(q: vec3i) -> bool {
  let inThroat = all(q >= vec3i(${THROAT_BLOCK_LO.join(", ")})) && all(q <= vec3i(${THROAT_BLOCK_HI.join(", ")}));
  return all(q >= vec3i(0)) && all(q < vec3i(${BULK_BX1}, ${BULK_BY1}, ${BZ})) && !inThroat;
}
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
/** Coarse MAC sampling (needs bulk and vel bound): coarse faces between two BULK cells read the bulk buffer at off; others the mean of their 16 fine faces at fineOff. */
const COARSE_SAMPLER_WGSL = /* wgsl */ `
fn cface(a: u32, k0: vec3i, off: u32, fineOff: u32) -> f32 {
  let dims = vec3i(128, 64, 16);
  if (k0[a] <= 0 || k0[a] >= dims[a]) { return 0.0; }
  var k = clamp(k0, vec3i(0), dims - vec3i(1)); k[a] = k0[a];
  if (coarseBulk(k) && coarseBulk(k - unitAxis(a))) { return bulk[off + a * NCL + clIndex(k)]; }
  let t1 = unitAxis((a + 1u) % 3u); let t2 = unitAxis((a + 2u) % 3u);
  var s = 0.0;
  for (var u = 0; u < 4; u++) { for (var v = 0; v < 4; v++) { s += vel[fineOff + a * N + cidx(4 * k + u * t1 + v * t2)]; } }
  return s * 0.0625;
}
fn cvelComp(a: u32, p: vec3f, off: u32, fineOff: u32) -> f32 {
  let q = p * 0.25 - vec3f(0.5) + 0.5 * vec3f(unitAxis(a));
  let fb = floor(q); let t = q - fb; let b = vec3i(fb);
  let x00 = mix(cface(a, b, off, fineOff), cface(a, b + vec3i(1, 0, 0), off, fineOff), t.x);
  let x10 = mix(cface(a, b + vec3i(0, 1, 0), off, fineOff), cface(a, b + vec3i(1, 1, 0), off, fineOff), t.x);
  let x01 = mix(cface(a, b + vec3i(0, 0, 1), off, fineOff), cface(a, b + vec3i(1, 0, 1), off, fineOff), t.x);
  let x11 = mix(cface(a, b + vec3i(0, 1, 1), off, fineOff), cface(a, b + vec3i(1, 1, 1), off, fineOff), t.x);
  return mix(mix(x00, x10, t.y), mix(x01, x11, t.y), t.z);
}
fn cvel(p: vec3f, off: u32, fineOff: u32) -> vec3f { return vec3f(cvelComp(0u, p, off, fineOff), cvelComp(1u, p, off, fineOff), cvelComp(2u, p, off, fineOff)); }
`;
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
fn ldims(l: u32) -> vec3i { return vec3i(512 >> l, 256 >> l, 64 >> l); }
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
// 1. Brick classification and list compaction (dense over the 16384 bricks).
// ---------------------------------------------------------------------------
export const classifyKernel = kernel("seep.classify", [ctrlRW, bricksRW, argsRW], /* wgsl */ `
fn liquidish(n: vec3i) -> bool {
  if (any(n < vec3i(0)) || n.x >= 64 || n.y >= 32 || n.z >= 8) { return false; }
  let b = u32(n.x) | (u32(n.y) << 6u) | (u32(n.z) << 11u);
  let cls = atomicLoad(&bricks[B_CLASS + b]);
  if ((cls & CLASS_DEAD) != 0u) { return false; }
  return (cls & CLASS_DEEP) != 0u || bitcast<f32>(atomicLoad(&bricks[B_MINPHI + b])) < 0.0
    || atomicLoad(&bricks[B_PCOUNT + b]) > 0u;
}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  let b = g.x;
  if (b == 0u) {
    atomicAdd(&ctrl[C_STEP], 1u);
    atomicStore(&ctrl[C_CONV], 0u); atomicStore(&ctrl[C_CYCLES], 0u);
    atomicStore(&ctrl[C_RES], 0u);
    atomicStore(&args[C_ARG_ACTIVE + 1u], 1u); atomicStore(&args[C_ARG_ACTIVE + 2u], 1u);
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
  // Activation reach: one brick (8 cells a step, 2.4 m/s) in the reservoir; two
  // in the chamber, where the falling jet and its splash move up to ~16 cells
  // a step (the SL traces and particles must land in active bricks).
  let r = select(1, 2, bc.x >= i32(PBX0));
  var act = false;
  for (var dz = -r; dz <= r; dz++) { for (var dy = -r; dy <= r; dy++) { for (var dx = -r; dx <= r; dx++) {
    if (liquidish(bc + vec3i(dx, dy, dz))) { act = true; }
  } } }
  let prev = (atomicLoad(&bricks[B_FLAGS + b]) & 1u) != 0u;
  if (act) {
    let i = atomicAdd(&args[C_ARG_ACTIVE], 1u); atomicStore(&bricks[B_LACT + i], b);
    atomicAdd(&args[C_ARG_ACTB], 1u); atomicAdd(&args[C_ARG_ACTBB], 1u);
    if ((cls & CLASS_DEEP) == 0u) { let j = atomicAdd(&args[C_ARG_SURF], 1u); atomicStore(&bricks[B_LSURF + j], b); }
    if ((cls & CLASS_PARTICLE) != 0u) { let k = atomicAdd(&args[C_ARG_PART], 1u); atomicStore(&bricks[B_LPART + k], b); }
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
export const clearKernel = kernel("seep.clear", [bricksRW, phisRW, velRW, opRW, bitsRW, presRW, coarseRW, volW, surfW], /* wgsl */ `
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
  if (l == 0u) { atomicStore(&bricks[B_MINPHI + b], bitcast<u32>(PHI_MAX)); }
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
// 3. Particle census (cfg.a = parity: src pool). The particles carry the
//    chamber's liquid, so none is retired for depth. Particles that leave the
//    particle region (the right chamber, x >= PX0) are retired (counted), as
//    are those over the crowding cap below.
// ---------------------------------------------------------------------------
export const countKernel = kernel("seep.count", [ctrlRW, partsRO, pcellRW, pgridRW, cfgU], /* wgsl */ `
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  let n = min(atomicLoad(&ctrl[C_PCOUNT + cfg.a]), CAP);
  if (i >= n) { return; }
  let p = parts[cfg.a * CAP * 2u + i].xyz;
  let c = vec3i(floor(p));
  if (c.x < PX0 || !cellOpen(c)) { pcell[i] = DEAD_PARTICLE; atomicAdd(&ctrl[C_DELETED], 1u); atomicAdd(&ctrl[C_DEL_OUT], 1u); return; }
  let pc = pidx(c);
  let f = p - floor(p);
  let o = select(0u, 1u, f.x >= 0.5) | select(0u, 2u, f.y >= 0.5) | select(0u, 4u, f.z >= 0.5);
  // No crowding cap: the particles carry the chamber's liquid, and a crowded
  // cell's excess shows in the surface (combine's unclipped mass bound), where
  // the pressure it implies spreads it.
  atomicAdd(&pgrid[pc], 1u);
  atomicOr(&pgrid[pc], 1u << (16u + o));
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
  if (p.x > 240.0 - RW && p.x < 256.0 + RW) {
    let cy = clamp(p.y, 64.0 + RW, 66.0 - RW); let cz = clamp(p.z, 32.0 + RW, 34.0 - RW);
    let dChan = length(vec2f(p.y - cy, p.z - cz));
    if (dChan > 0.0) {
      let dL = p.x - (240.0 - RW); let dR = (256.0 + RW) - p.x;
      if (dChan < min(dL, dR) && p.x >= 240.0 && p.x < 256.0) { p.y = cy; p.z = cz; }
      else if (dL < dR) { p.x = 240.0 - RW; } else { p.x = 256.0 + RW; }
    }
  }
  return p;
}
/** Moves (*x) one step; a component of (*v) driving the particle into a solid it
 *  was projected out of is removed (free-slip particle collision, Bridson 2015 §7.6). */
fn moveParticle(x: ptr<function, vec3f>, v: ptr<function, vec3f>, k1: vec3f) {
  let x0 = *x;
  let k2 = velU(collide(x0 + (CFL / 3.0) * k1));
  let k3 = velU(collide(x0 + (2.0 * CFL / 3.0) * k2));
  let xn = x0 + (CFL / 4.0) * (k1 + 3.0 * k3);
  let xc = collide(xn);
  let push = xn - xc;
  for (var a = 0u; a < 3u; a++) {
    if (push[a] != 0.0 && (*v)[a] * push[a] > 0.0) { (*v)[a] = 0.0; }
  }
  *x = xc;
}
`;

// ---------------------------------------------------------------------------
// 4. Outlet emission: the particles are the conserved carrier of the chamber's
//    liquid, and the only liquid entering the chamber crosses the throat's
//    outlet plane x = PX0. Each step every outlet face emits exactly the
//    volume it passed, u_x dt h^2 times its wetted fraction, as particles at
//    eight per cell volume: layers of 2x2 particles h/2 apart along x, the
//    swept length not yet emitted carried to the next step, so the emitted
//    mass equals the outflow to within one layer per face over the whole run
//    (a flux-conservative inflow boundary for particle methods: Ferstl et al.
//    2016 §4 sources; Bridson 2015 §7.6 particle emission). Each layer sits at
//    the centre of the slab it represents, displaced from the plane along the
//    face velocity it carries (the projected u^{n+1}, the velocity that swept
//    it). Runs after G2P, appending to the pool the next census reads.
//    cfg.a = parity (the moved pool is 1 - parity).
// ---------------------------------------------------------------------------
export const emitKernel = kernel("seep.emit", [ctrlRW, bricksRW, partsRW, velRO, phisRO, cfgU], /* wgsl */ `
${phiSampler("phi", "phis", "PHI_OFF")}
${velocitySampler("velU", "vel", "U_OFF")}
${PARTICLE_MOVE_WGSL}
const SPACING: f32 = 0.5;
var<workgroup> sTotal: array<f32, 4>;
// 128 threads a face: thread s emits particle s % 4 of layers 1 + s / 4, + 32, ...
@compute @workgroup_size(512) fn main(@builtin(local_invocation_index) l: u32) {
  let t = l >> 7u; let s = l & 127u;
  let fy = THROAT_Y0 + i32(t & 1u); let fz = THROAT_Z0 + i32(t >> 1u);
  // The liquid crossing outlet face (PX0, fy, fz) carries the face's own normal velocity (MAC, Harlow &
  // Welch 1965) and the tangential velocity of the upstream throat cell it leaves (donor cell, Courant,
  // Isaacson & Rees 1952). Trilinear samples on the outlet plane would mix in the closed wall faces
  // around the 2x2 throat (u = 0: 23% slow on average) and, tangentially, the chamber column the jet is
  // already falling through.
  let up = vec3i(PX0 - 1, fy, fz);
  let uf = vec3f(velUFace(0u, vec3i(PX0, fy, fz)),
    0.5 * (velUFace(1u, up) + velUFace(1u, up + vec3i(0, 1, 0))),
    0.5 * (velUFace(2u, up) + velUFace(2u, up + vec3i(0, 0, 1))));
  if (s == 0u) {
    // Wetted fraction of the face, from the level set on the plane (4x4 bilinear samples).
    var wet = 0.0;
    for (var k = 0u; k < 16u; k++) {
      let q = vec3f(f32(PX0), f32(fy) + (f32(k & 3u) + 0.5) * 0.25, f32(fz) + (f32(k >> 2u) + 0.5) * 0.25);
      if (phiTri(q) < 0.0) { wet += 1.0 / 16.0; }
    }
    sTotal[t] = bitcast<f32>(atomicLoad(&ctrl[C_EMIT_ACC + t])) + max(uf.x, 0.0) * CFL * wet;
  }
  workgroupBarrier();
  let total = sTotal[t];
  let n = u32(floor(total / SPACING));
  if (s == 0u) { atomicStore(&ctrl[C_EMIT_ACC + t], bitcast<u32>(total - SPACING * f32(n))); }
  let base = (1u - cfg.a) * CAP * 2u;
  let step = atomicLoad(&ctrl[C_STEP]);
  let q = s & 3u;
  for (var j = 1u + (s >> 2u); j <= n; j += 32u) {
    // Layer j holds the slab [total - j SPACING, total - (j - 1) SPACING] of swept length; its centre is this far downstream.
    let dist = total - SPACING * (f32(j) - 0.5);
    let jit = 0.2 * (vec2f(hashUnit(step, 64u * t + j, q), hashUnit(step, 64u * t + j, q + 4u)) - vec2f(0.5));
    let off = vec2f(0.25 + 0.5 * f32(q & 1u), 0.25 + 0.5 * f32(q >> 1u)) + jit;
    let p0 = vec3f(f32(PX0), f32(fy) + off.x, f32(fz) + off.y);
    let x = collide(p0 + (dist / uf.x) * uf);
    let k = atomicAdd(&ctrl[C_PCOUNT + 1u - cfg.a], 1u);
    if (k >= CAP) { atomicOr(&ctrl[C_FAIL], FAIL_POOL); atomicAdd(&ctrl[C_OVERFLOW], 1u); continue; }
    parts[base + k] = vec4f(x, 0.0);
    parts[base + CAP + k] = vec4f(uf, 0.0);
    atomicAdd(&ctrl[C_EMITTED], 1u);
    atomicAdd(&bricks[B_PCOUNT + brickOfCell(vec3i(floor(x)))], 1u);
  }
}
`);

// ---------------------------------------------------------------------------
// 5. Per-brick exclusive scan of cell counts; each brick reserves its range in
//    the destination pool (cfg.a = parity; dst = 1 - parity).
// ---------------------------------------------------------------------------
export const scanKernel = kernel("seep.scan", [ctrlRW, bricksRW, pgridRW, cfgU], /* wgsl */ `
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
  atomicStore(&pgrid[pc], atomicLoad(&pgrid[pc]) & 0x00ff0000u);
}
`);

// ---------------------------------------------------------------------------
// 6. Scatter live particles into the destination pool, sorted by brick and cell.
// ---------------------------------------------------------------------------
export const scatterKernel = kernel("seep.scatter", [ctrlRW, partsRW, pcellRO, pgridRW, cfgU], /* wgsl */ `
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
export const separateKernel = kernel("seep.separate", [ctrlRO, partsRW, pgridRO, phisRO, cfgU], /* wgsl */ `
${phiSampler("phi", "phis", "PHI_OFF")}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  let n = min(ctrl[C_PCOUNT + 1u - cfg.a], CAP);
  if (i >= n) { return; }
  let base = (1u - cfg.a) * CAP * 2u;
  let x = parts[base + i].xyz;
  let bv = vec3i(floor(x - vec3f(0.5)));
  var d = vec3f(0.0); var m = 0.0;
  for (var k = 0u; k < 8u; k++) {
    let c = bv + vec3i(i32(k & 1u), i32((k >> 1u) & 1u), i32(k >> 2u));
    if (!inBox(c) || c.x < PX0) { continue; }
    let pc = pidx(c);
    let start = pgrid[PN + pc]; let cnt = pgrid[pc] & 0xffffu;
    for (var j = start; j < start + cnt; j++) {
      if (j == i) { continue; }
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
  parts[cfg.a * CAP * 2u + i] = vec4f(d, 0.0);
}
`);

// ---------------------------------------------------------------------------
// 7. Level-set advection: semi-Lagrangian, one Ralston RK3 backtrace of the
//    whole step (three velocity samples, any length; as the reference's
//    surfaceTraceSubsteps "off"), monotone Catmull-Rom tricubic. phi -> phiT. Also the
//    DEEP invariant: the bottom vertex layer of a level-set brick standing on
//    a DEEP brick must stay liquid (a violation latches a failure).
// ---------------------------------------------------------------------------
export const advectPhiKernel = kernel("seep.advectPhi", [ctrlRW, bricksRO, phisRW, velRO], /* wgsl */ `
${phiSampler("phi", "phis", "PHI_OFF")}
${velocitySampler("velU", "vel", "U_OFF")}
fn cornerOfOpen(v: vec3i) -> bool {
  for (var k = 0u; k < 8u; k++) { if (cellOpen(v - vec3i(i32(k & 1u), i32((k >> 1u) & 1u), i32(k >> 2u)))) { return true; } }
  return false;
}
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let b = bricks[B_LSURF + wg.x];
  let bc = brickCoord(b);
  let lc = localCoord(l);
  let v = bc * 8 + lc;
  // Chamber vertices the particle surface (8) overwrites: every one with an open cell
  // around it, none of them upstream of PX0 (the throat's exit keeps this trace). Chamber
  // bricks are never DEEP and every active one is in both the surface and particle lists.
  if ((bricks[B_CLASS + b] & CLASS_CHAMBER) != 0u) {
    var open = false; var upstream = false;
    for (var k = 0u; k < 8u; k++) {
      let c = v - vec3i(i32(k & 1u), i32((k >> 1u) & 1u), i32(k >> 2u));
      if (cellOpen(c)) { open = true; if (c.x < PX0) { upstream = true; } }
    }
    if (open && !upstream) { return; }
  }
  let p = vec3f(v);
  var u = velU(p);
  var phi0 = 0.0;
  if (all(u == vec3f(0.0))) {
    // At rest (air beyond the extension): the departure point is the vertex itself, where the
    // interpolant returns the vertex value; the same result without the trace.
    phi0 = clamp(phiAt(v), -PHI_MAX, PHI_MAX);
  } else {
    let k2 = velU(p - 0.5 * CFL * u);
    let k3 = velU(p - 0.75 * CFL * k2);
    phi0 = clamp(phiCubic(p - CFL * (2.0 * u + 3.0 * k2 + 4.0 * k3) / 9.0), -PHI_MAX, PHI_MAX);
  }
  phis[PHIT_OFF + ((b << 9u) | l)] = phi0;
  if (lc.y == 0 && bc.y > 0) {
    let below = b - 64u;
    if ((bricks[B_CLASS + below] & CLASS_DEEP) != 0u && phi0 >= 0.0 && cornerOfOpen(v)) { atomicOr(&ctrl[C_FAIL], FAIL_DEEP); }
  }
}
`);

// ---------------------------------------------------------------------------
// 8. Chamber surface from the particles alone (they carry its liquid): the
//    particles' mass plane. Eight particles fill a cell, so where the open
//    cells around a vertex hold a fraction f of that, the liquid they carry is
//    a plane 1 - 2f cells above the vertex. At a solid (closed cells on one
//    side, which read as full) the plane fills from the wall instead: along
//    the axis whose two layers of cells differ most in held fraction, phi =
//    -f_sparse against a closed dense layer, 1 - f_dense - f_sparse otherwise;
//    both hold exactly the particles' volume (NB-FLIP's sampleBound "walls",
//    the mass plane of the plan §2.2). Crowded cells count in full, so their
//    excess shows. The semi-Lagrangian phi is not unioned in (it only ever
//    adds liquid there), nor are particle spheres (Ferstl 2016 Eq. 4's union
//    can only remove liquid from the plane: 17% of a 0.6h film). The mass fit
//    (8b) then matches the cells' fill to their particle mass. The throat's
//    exit vertices (corners of an open cell upstream of PX0) keep the grid's
//    level set, which holds the channel.
// ---------------------------------------------------------------------------
export const combineKernel = kernel("seep.combine", [bricksRO, phisRW, pgridRO], /* wgsl */ `
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let b = bricks[B_LPART + wg.x];
  if ((bricks[B_CLASS + b] & CLASS_CHAMBER) == 0u) { return; }
  let v = brickCoord(b) * 8 + localCoord(l);
  var held = 0u; var open = 0u; var upstream = false;
  var heldLo = vec3u(0u); var heldHi = vec3u(0u); var openLo = vec3u(0u); var openHi = vec3u(0u);
  for (var k = 0u; k < 8u; k++) {
    let bit = vec3u(k & 1u, (k >> 1u) & 1u, k >> 2u);
    let c = v - vec3i(1) + vec3i(bit);
    if (!cellOpen(c)) { continue; }
    var n = 0u;
    if (c.x >= PX0) { n = pgrid[pidx(c)] & 0xffffu; } else { upstream = true; }
    open += 1u; held += n;
    let hi = bit != vec3u(0u);
    heldHi += select(vec3u(0u), vec3u(n), hi); openHi += select(vec3u(0u), vec3u(1u), hi);
    heldLo += select(vec3u(n), vec3u(0u), hi); openLo += select(vec3u(1u), vec3u(0u), hi);
  }
  if (open == 0u || upstream) { return; }
  var bound = 1.0 - f32(held) / (4.0 * f32(open));
  if (open < 8u) {
    var gap = -1.0;
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
  }
  phis[PHIT_OFF + ((b << 9u) | l)] = clamp(bound, -PHI_MAX, PHI_MAX);
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
//    whole fill, a vertex above it half). Upstream (throat) and solid-only
//    vertices are left alone. cfg.a = source offset for chamber bricks (others
//    read phiT), cfg.b = destination offset; an even number of passes
//    ping-pongs phiT -> phi -> ... -> phiT, which redistance then reads.
// ---------------------------------------------------------------------------
export const massFitKernel = kernel("seep.massFit", [bricksRO, phisRW, pgridRO, cfgU], /* wgsl */ `
${levelSetFillWGSL}
const GAIN: f32 = 1.5;
var<workgroup> tv: array<f32, 1000>;
var<workgroup> tm: array<f32, 729>;
var<workgroup> sBusy: u32;
fn srcPhi(v: vec3i) -> f32 {
  let w = clampV(v);
  let chamber = (bricks[B_CLASS + brickOfCell(w)] & CLASS_CHAMBER) != 0u;
  return phis[select(PHIT_OFF, cfg.a, chamber) + cidx(w)];
}
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let b = bricks[B_LPART + wg.x];
  if ((bricks[B_CLASS + b] & CLASS_CHAMBER) == 0u) { return; }
  let o = brickCoord(b) * 8;
  // No particle in the brick or its 26 neighbours: every cell the brick's vertices touch has
  // F = 0 and (the plane being >= 0 there) fill 0, so nothing moves; copy through.
  if (l < 27u) {
    let n = brickCoord(b) + vec3i(i32(l % 3u), i32((l / 3u) % 3u), i32(l / 9u)) - vec3i(1);
    if (all(n >= vec3i(0)) && n.x < 64 && n.y < 32 && n.z < 8 && bricks[B_PCOUNT + (u32(n.x) | (u32(n.y) << 6u) | (u32(n.z) << 11u))] > 0u) { sBusy = 1u; }
  }
  if (workgroupUniformLoad(&sBusy) == 0u) {
    phis[cfg.b + ((b << 9u) | l)] = srcPhi(o + localCoord(l));
    return;
  }
  for (var t = l; t < 1000u; t += 512u) {
    tv[t] = srcPhi(o - vec3i(1) + vec3i(i32(t % 10u), i32((t / 10u) % 10u), i32(t / 100u)));
  }
  workgroupBarrier();
  // Target volume fractions: F = count / 8, a cell's excess over full carried up its column
  // (liquid settles from the dense side, the bound's own rule), so every target is one a cell
  // can hold; the carry out of the tile's top row stays in it (only liquid deeper than a brick).
  if (l < 81u) {
    let qx = i32(l % 9u); let qz = i32(l / 9u);
    var carry = 0.0;
    for (var qy = 0; qy < 9; qy++) {
      let c = o - vec3i(1) + vec3i(qx, qy, qz);
      var F = 0.0;
      if (cellOpen(c) && c.x >= PX0) { F = f32(pgrid[pidx(c)] & 0xffffu) / 8.0 + carry; carry = max(F - 1.0, 0.0); if (qy < 8) { F = min(F, 1.0); } }
      tm[u32(qx + 9 * (qy + 9 * qz))] = F;
    }
  }
  workgroupBarrier();
  for (var t = l; t < 729u; t += 512u) {
    let q = vec3i(i32(t % 9u), i32((t / 9u) % 9u), i32(t / 81u));
    let c = o - vec3i(1) + q;
    var m = 0.0;
    if (cellOpen(c) && c.x >= PX0) {
      var v: array<f32, 8>;
      for (var k = 0u; k < 8u; k++) {
        let r = q + vec3i(i32(k & 1u), i32((k >> 1u) & 1u), i32(k >> 2u));
        v[k] = tv[u32(r.x + 10 * (r.y + 10 * r.z))];
      }
      m = tm[t] - fill(v);
    }
    tm[t] = m;
  }
  workgroupBarrier();
  let lc = localCoord(l);
  let vtx = o + lc;
  var phi = tv[u32((lc.x + 1) + 10 * ((lc.y + 1) + 10 * (lc.z + 1)))];
  if (abs(phi) < 1.5) {
    var sum = 0.0; var open = 0u; var upstream = false;
    for (var k = 0u; k < 8u; k++) {
      let d = vec3i(i32(k & 1u), i32((k >> 1u) & 1u), i32(k >> 2u));
      let c = vtx - vec3i(1) + d;
      if (!cellOpen(c)) { continue; }
      if (c.x < PX0) { upstream = true; }
      open += 1u;
      sum += tm[u32((lc.x + d.x) + 9 * ((lc.y + d.y) + 9 * (lc.z + d.z)))];
    }
    if (open > 0u && !upstream) { phi = clamp(phi - GAIN * sum / f32(open), -PHI_MAX, PHI_MAX); }
  }
  phis[cfg.b + ((b << 9u) | l)] = phi;
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
export const redistInitKernel = kernel("seep.redistInit", [bricksRO, phisRW], /* wgsl */ `
${VERTEX_OPEN_WGSL}
// The brick's vertices and a one-vertex halo, staged once: 0 = not an open vertex in the box
// (ignored), 1 = phiT >= 0, 2 = phiT < 0; the 26-neighbour sign test then reads the tile.
var<workgroup> tsgn: array<u32, 1000>;
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
  var cut = false;
  for (var k = 0u; k < 27u; k++) {
    let sg = tsgn[ti + (k % 3u) + 10u * ((k / 3u) % 3u) + 100u * (k / 9u) - 111u];
    if (sg != 0u && sg != own) { cut = true; }
  }
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
export const redistSweepKernel = kernel("seep.redistSweep", [bricksRO, phisRW, cfgU, surfW], /* wgsl */ `
${SURF_WGSL}
${VERTEX_OPEN_WGSL}
var<workgroup> tA: array<f32, 1000>;
var<workgroup> tB: array<f32, 1000>;
var<workgroup> tOpen: array<u32, 1000>;
var<workgroup> sMixed: atomic<u32>;
var<workgroup> sMixedU: u32;
/** Sign of the neighbour nearest the interface (extension into solid-interior vertices). */
fn nearSign(t: ptr<workgroup, array<f32, 1000>>, ti: u32) -> f32 {
  var m = abs((*t)[ti - 1u]); var s = (*t)[ti - 1u];
  for (var k = 1u; k < 6u; k++) {
    let o = select(select(100u, 10u, k < 4u), 1u, k < 2u);
    let j = select(ti - o, ti + o, (k & 1u) == 1u);
    if (abs((*t)[j]) < m) { m = abs((*t)[j]); s = (*t)[j]; }
  }
  return select(1.0, -1.0, s < 0.0);
}
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
  var mixed = false;
  for (var t = l; t < 1000u; t += 512u) {
    let q = clampV(o + tileCoord(t));
    let val = phis[cfg.a + cidx(q)];
    tA[t] = val; tB[t] = val; tOpen[t] = select(0u, 1u, vertexOpen(q));
    mixed = mixed || val != PHI_MAX;
  }
  if (mixed) { atomicStore(&sMixed, 1u); }
  workgroupBarrier();
  if (l == 0u) { sMixedU = atomicLoad(&sMixed); }
  let lc = localCoord(l);
  let ti = tileIndex(lc);
  // A tile that is all air at the band limit is a fixed point of the sweeps.
  if (workgroupUniformLoad(&sMixedU) == 0u) {
    phis[cfg.b + ((b << 9u) | l)] = PHI_MAX;
    if (cfg.c == 1u) { publishSurf(o + lc, PHI_MAX); }
    return;
  }
  let me = tA[ti];
  let neg = me < 0.0;
  let open = tOpen[ti] == 1u;
  var frozen = false;
  for (var k = 0u; k < 27u; k++) {
    let j = tileIndex(lc + vec3i(i32(k % 3u), i32((k / 3u) % 3u), i32(k / 9u)) - vec3i(1));
    if (open && k != 13u && tOpen[j] == 1u && (tA[j] < 0.0) != neg) { frozen = true; }
  }
  let sgn = select(1.0, -1.0, neg);
  for (var s = 0u; s < 8u; s++) {
    if (!frozen) {
      if ((s & 1u) == 0u) {
        let ax = min(abs(tA[ti - 1u]), abs(tA[ti + 1u]));
        let ay = min(abs(tA[ti - 10u]), abs(tA[ti + 10u]));
        let az = min(abs(tA[ti - 100u]), abs(tA[ti + 100u]));
        tB[ti] = select(nearSign(&tA, ti), sgn, open) * min(abs(tA[ti]), eik(ax, ay, az));
      } else {
        let ax = min(abs(tB[ti - 1u]), abs(tB[ti + 1u]));
        let ay = min(abs(tB[ti - 10u]), abs(tB[ti + 10u]));
        let az = min(abs(tB[ti - 100u]), abs(tB[ti + 100u]));
        tA[ti] = select(nearSign(&tB, ti), sgn, open) * min(abs(tB[ti]), eik(ax, ay, az));
      }
    }
    workgroupBarrier();
  }
  phis[cfg.b + ((b << 9u) | l)] = tA[ti];
  if (cfg.c == 1u) { publishSurf(o + lc, tA[ti]); }
}
`);

// ---------------------------------------------------------------------------
// 11. P2G (Ferstl 2016 §3.1): quadratic B-spline splat of the sorted band
//     particles of one brick into workgroup fixed-point accumulators, then one
//     global atomic per touched face. cfg.a = parity (sorted pool = dst).
// ---------------------------------------------------------------------------
export const p2gKernel = kernel("seep.p2g", [bricksRO, partsRO, accRW, cfgU], /* wgsl */ `
var<workgroup> sMom: array<atomic<i32>, 3300>;
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
  workgroupBarrier();
  let base = (1u - cfg.a) * CAP * 2u;
  for (var k = start + l; k < start + num; k += 256u) {
    let xp = parts[base + k].xyz; let vp = parts[base + CAP + k].xyz;
    for (var a = 0u; a < 3u; a++) {
      let q = xp - vec3f(0.5) + 0.5 * vec3f(unitAxis(a));
      let bi = vec3i(floor(q - vec3f(0.5)));
      let fr = q - vec3f(bi);
      let wx = bspline(fr.x); let wy = bspline(fr.y); let wz = bspline(fr.z);
      for (var dz = 0; dz < 3; dz++) { for (var dy = 0; dy < 3; dy++) { for (var dx = 0; dx < 3; dx++) {
        let w = wx[dx] * wy[dy] * wz[dz];
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
    if (!inBox(q) || q.x < PX0) { continue; }
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
/** Coarse-lattice 8^3 blocks covering the BULK box (2 coarse cells a brick). */
const CB_DIM = [Math.ceil(BULK_BX1 / 4), Math.ceil(BULK_BY1 / 4), Math.ceil(BZ / 4)] as const;
export const BULK_CBLOCKS = CB_DIM[0] * CB_DIM[1] * CB_DIM[2];
export const velocityBulkKernel = kernel("seep.velocityBulk", [bricksRO, velRW, bulkRW], /* wgsl */ `
${velocitySampler("velU", "vel", "U_OFF")}
${bulkWGSL(false)}
${COARSE_SAMPLER_WGSL}
/** Semi-Lagrangian RK2 backtrace (segments of at most one cell) of the fine face (a, c) through the fine velocity. */
fn fineFaceSL(a: u32, c: vec3i) -> f32 {
  let pos = vec3f(c) + vec3f(0.5) - 0.5 * vec3f(unitAxis(a));
  let u0 = velU(pos);
  let n = clamp(ceil(length(u0) * CFL), 1.0, MAXSEG);
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
// Coarse velocity tile: the block's coarse faces with a two-cell halo, per component, staged once
// (cface: the coarse face, or its 16 fine faces' mean next to fine cells); samples whose eight
// corners fall inside read it, the rest read through cface. Same values, same interpolation.
const CT: i32 = 12;
var<workgroup> ct: array<f32, 5184>;
var<private> ctOrigin: vec3i;
fn ctComp(a: u32, p: vec3f) -> f32 {
  let q = p * 0.25 - vec3f(0.5) + 0.5 * vec3f(unitAxis(a));
  let fb = floor(q);
  let lb = vec3i(fb) - ctOrigin;
  if (any(lb < vec3i(0)) || any(lb > vec3i(CT - 2))) { return cvelComp(a, p, BU_OFF, U_OFF); }
  let t = q - fb;
  let k = u32(i32(a) * CT * CT * CT + lb.x + CT * (lb.y + CT * lb.z));
  let sy = u32(CT); let sz = u32(CT * CT);
  let x00 = mix(ct[k], ct[k + 1u], t.x); let x10 = mix(ct[k + sy], ct[k + sy + 1u], t.x);
  let x01 = mix(ct[k + sz], ct[k + sz + 1u], t.x); let x11 = mix(ct[k + sz + sy], ct[k + sz + sy + 1u], t.x);
  return mix(mix(x00, x10, t.y), mix(x01, x11, t.y), t.z);
}
fn ctVel(p: vec3f) -> vec3f { return vec3f(ctComp(0u, p), ctComp(1u, p), ctComp(2u, p)); }
/** BULK coarse cell k: SL advection of its coarse low faces at 4h; the fine interface faces stored in it (fine cell below along a) as fine faces. */
fn bulkVelocity(k: vec3i) {
  for (var a = 0u; a < 3u; a++) {
    let e = unitAxis(a);
    let kn = k - e;
    if (coarseBulk(kn)) {
      let pos = vec3f(4 * k) + vec3f(2.0) - 2.0 * vec3f(e);
      let u0 = ctVel(pos);
      let n = clamp(ceil(length(u0) * CFL * 0.25), 1.0, MAXSEG);
      let tau = CFL / n;
      var x = pos;
      for (var s = 0.0; s < n; s += 1.0) {
        let k1 = select(ctVel(x), u0, s == 0.0);
        x = x - tau * ctVel(x - 0.5 * tau * k1);
      }
      bulk[BUS_OFF + a * NCL + clIndex(k)] = ctComp(a, x) - select(0.0, GDT, a == 1u);
    } else {
      let t1 = unitAxis((a + 1u) % 3u); let t2 = unitAxis((a + 2u) % 3u);
      for (var u = 0; u < 4; u++) { for (var v = 0; v < 4; v++) {
        let c = 4 * k + u * t1 + v * t2;
        var val = 0.0;
        if (fineOpen(c - e)) { val = fineFaceSL(a, c) + gravityOn(a, c); }
        vel[US_OFF + a * N + cidx(c)] = val;
      } }
    }
  }
}
// One workgroup per 8^3 block of the coarse lattice over the BULK box (BULK_CBLOCKS of them).
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let blk = vec3i(i32(wg.x % ${CB_DIM[0]}u), i32((wg.x / ${CB_DIM[0]}u) % ${CB_DIM[1]}u), i32(wg.x / ${CB_DIM[0] * CB_DIM[1]}u));
  ctOrigin = blk * 8 - vec3i(2);
  for (var t = l; t < 5184u; t += 512u) {
    let a = t / 1728u; let r = t % 1728u;
    ct[t] = cface(a, ctOrigin + vec3i(i32(r % 12u), i32((r / 12u) % 12u), i32(r / 144u)), BU_OFF, U_OFF);
  }
  workgroupBarrier();
  let k = blk * 8 + localCoord(l);
  if (coarseBulk(k)) { bulkVelocity(k); }
}
`);

export const velocityKernel = kernel("seep.velocity", [ctrlRW, bricksRW, phisRW, velRW, accRW, volW], /* wgsl */ `
${levelSetFillWGSL}
${velocitySampler("velU", "vel", "U_OFF")}
${bulkWGSL(true)}
fn gravityOn(a: u32, c: vec3i) -> f32 {
  let lo = c - unitAxis(a);
  let d = select(1.0, 2.5, cellBulk(lo) != cellBulk(c));
  return -GDT * (ownerY(c) - ownerY(lo)) / d;
}
// The brick's vertices and a one-vertex halo (every vertex this kernel reads), staged once.
var<workgroup> tph: array<f32, 1000>;
var<private> phOrigin: vec3i;
fn ph(v: vec3i) -> f32 { return tph[tileIndex(v - phOrigin)]; }
const FAR_AIR: f32 = 4.0;
var<workgroup> red: array<f32, 512>;
var<workgroup> redFar: array<f32, 512>;
var<workgroup> sFar: u32;
// Velocity tile: the brick's faces and a two-cell halo of each component,
// staged once; backtrace samples whose eight corners fall inside it read the
// tile, the rest (fast liquid, a few cells a step) read the buffer. Same
// values, same interpolation.
const VT: i32 = 12;
var<workgroup> vt: array<f32, 5184>;
var<private> tileOrigin: vec3i;
var<private> useTile: bool;
const TILE_MIN: f32 = 96.0;
fn velTComp(a: u32, p: vec3f) -> f32 {
  let q = p - vec3f(0.5) + 0.5 * vec3f(unitAxis(a));
  let fb = floor(q);
  let lb = vec3i(fb) - tileOrigin;
  if (!useTile || any(lb < vec3i(0)) || any(lb > vec3i(VT - 2))) { return velUComp(a, p); }
  let t = q - fb;
  let k = u32(i32(a) * VT * VT * VT + lb.x + VT * (lb.y + VT * lb.z));
  let sy = u32(VT); let sz = u32(VT * VT);
  let x00 = mix(vt[k], vt[k + 1u], t.x); let x10 = mix(vt[k + sy], vt[k + sy + 1u], t.x);
  let x01 = mix(vt[k + sz], vt[k + sz + 1u], t.x); let x11 = mix(vt[k + sz + sy], vt[k + sz + sy + 1u], t.x);
  return mix(mix(x00, x10, t.y), mix(x01, x11, t.y), t.z);
}
fn velT(p: vec3f) -> vec3f { return vec3f(velTComp(0u, p), velTComp(1u, p), velTComp(2u, p)); }
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let b = atomicLoad(&bricks[B_LACT + wg.x]);
  let cls = atomicLoad(&bricks[B_CLASS + b]);
  let deep = (cls & CLASS_DEEP) != 0u;
  let c = brickCoord(b) * 8 + localCoord(l);
  let i = (b << 9u) | l;
  tileOrigin = brickCoord(b) * 8 - vec3i(2);
  if (l == 0u) { atomicStore(&bricks[B_PCOUNT + b], 0u); }
  phOrigin = brickCoord(b) * 8;
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
  }
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
  }
  let allFar = workgroupUniformLoad(&sFar) != 0u;
  // Pass 1: each face is zero, a particle transfer, or a backtrace (bit a of need).
  let part = c.x >= PX0;
  var need = 0u;
  for (var a = 0u; a < 3u; a++) {
    var mom = 0; var wsum = 0;
    if (part) {
      let pc = pidx(c);
      mom = atomicExchange(&acc[a * PN + pc], 0);
      wsum = atomicExchange(&acc[(3u + a) * PN + pc], 0);
    }
    if (!faceOpen(c, a) || allFar) { vel[US_OFF + a * N + i] = 0.0; continue; }
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
  // The velocity tile pays for itself only with enough traces to read it (a chamber brick's
  // faces are mostly transfers or skipped): below TILE_MIN traces the samples read the buffer.
  red[l] = f32(countOneBits(need));
  workgroupBarrier();
  if (l < 32u) { var n = 0.0; for (var j = 0u; j < 16u; j++) { n += red[j * 32u + l]; } redFar[l] = n; }
  workgroupBarrier();
  if (l == 0u) { var n = 0.0; for (var j = 0u; j < 32u; j++) { n += redFar[j]; } sFar = select(0u, 1u, n >= TILE_MIN); }
  useTile = workgroupUniformLoad(&sFar) != 0u;
  if (useTile) {
    for (var t = l; t < 5184u; t += 512u) {
      let a = t / 1728u; let r = t % 1728u;
      let q = tileOrigin + vec3i(i32(r % 12u), i32((r / 12u) % 12u), i32(r / 144u));
      vt[t] = velUFace(a, q);
    }
  }
  workgroupBarrier();
  // Pass 2: the backtraces.
  for (var a = 0u; a < 3u; a++) {
    if (((need >> a) & 1u) == 0u) { continue; }
    var val = 0.0;
    {
      let e = unitAxis(a);
      let pos = vec3f(c) + vec3f(0.5) - 0.5 * vec3f(e);
      let u0 = velT(pos);
      let segs = ceil(length(u0) * CFL);
      if (segs > 12.0) { atomicMax(&ctrl[C_MAX_SEG], u32(segs)); }
      let n = clamp(segs, 1.0, MAXSEG);
      let tau = CFL / n;
      // RK2 midpoint segments, one velocity evaluation site: even half-steps
      // locate the midpoint from k1, odd ones advance with k2.
      var x = pos; var q = pos; var kk = u0;
      let last = u32(n) * 2u - 1u;
      for (var j = 0u; j <= last; j++) {
        if ((j & 1u) == 0u) { q = x - 0.5 * tau * kk; } else { x = x - tau * kk; q = x; }
        if (j == last) { break; }
        kk = velT(q);
      }
      val = velTComp(a, x);
    }
    val += gravityOn(a, c);
    vel[US_OFF + a * N + i] = val;
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
export const operatorKernel = kernel("seep.operator", [bricksRW, phisRO, velRO, opRW, bitsRW, presRW, pgridRW, bulkRO, ctrlRW], /* wgsl */ `
${bulkWGSL(true)}
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
  op[DIAG_OFF + sl] = diag; op[RHS_OFF + sl] = -div; bits[MASK_OFF + sl] = mask;
}
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
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
  if (c.x >= PX0) { atomicStore(&pgrid[pidx(c)], 0u); }
  let pc = phis[PHIC_OFF + i];
  let liquid = cellOpen(c) && pc < 0.0;
  if (liquid) { atomicStore(&anyLiquid, 1u); }
  // The pressure solve visits the liquid bricks only (append once per brick).
  workgroupBarrier();
  if (l == 0u && atomicLoad(&anyLiquid) != 0u) {
    let j = atomicAdd(&ctrl[C_LIQ_COUNT], 1u);
    atomicStore(&bricks[B_LLIQ + j], b);
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
    if (!inBox(n)) {
      if (!(a == 1u && s > 0)) { continue; }
    } else {
      if (!cellOpen(n)) { continue; }
      if (cellBulk(n)) { mask |= 1u << (k + 6u); diag += SUBC; continue; }
      pn = phis[PHIC_OFF + cidx(n)];
    }
    if (pn < 0.0) { mask |= 1u << k; diag += 1.0; }
    else { diag += 1.0 / max(pc / (pc - pn), THETA_MIN); }
  }
  var div = 0.0;
  for (var a = 0u; a < 3u; a++) { div += us(a, c + unitAxis(a)) - us(a, c); }
  op[DIAG_OFF + i] = diag; op[RHS_OFF + i] = -div; bits[MASK_OFF + i] = mask;
}
`);

// ---------------------------------------------------------------------------
// 14. Aggregation multigrid hierarchy: A_c = R A P / OC with piecewise-constant
//     P and R = P^T (Braess 1995), levels 1..3 per brick (brick-local), 4..6 in
//     one small workgroup. Couplings are stored on the lower cell (+x,+y,+z).
// ---------------------------------------------------------------------------
export const coarsenBrickKernel = kernel("seep.coarsenBrick", [bricksRW, opRO, bitsRO, coarseRW, ctrlRW], /* wgsl */ `
${levelWGSL()}
${bulkWGSL(true)}
/** List the level-4 block (2x2x2 bricks) of brick bc once per step: it holds a level-3 unknown. */
fn markL4(bc: vec3i) {
  let q = bc / 2;
  let q4 = u32(q.x + 32 * (q.y + 16 * q.z));
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
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let b = atomicLoad(&bricks[B_LBULK + wg.x]);
  let bc = brickCoord(b);
  if (wg.x < NBULK) {
    // BULK brick: its eight coarse cells are level-2 cells with native rows (coarse-coarse 4) and the
    // level-2 couplings to fine neighbours; the diagonal is the coupling sum. Level 1 stays empty.
    if (t < 8u) {
      let k = bc * 2 + vec3i(i32(t & 1u), i32((t >> 1u) & 1u), i32(t >> 2u));
      var o = vec4f(0.0);
      for (var kk = 0u; kk < 6u; kk++) {
        let kn = k + select(-1, 1, (kk & 1u) == 1u) * unitAxis(kk >> 1u);
        let w = select(subCoupling(k, kk), 4.0, coarseBulk(kn));
        o.x += w;
        if ((kk & 1u) == 1u) { o[1u + (kk >> 1u)] = w; }
      }
      storeCop(2u, k, o);
      s2[t] = o;
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
      storeCop(3u, bc, o3);
      if (o3.x > 0.0) { markL4(bc); }
    }
    return;
  }
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

export const coarsenTopKernel = kernel("seep.coarsenTop", [coarseRW, ctrlRW, argsRW], /* wgsl */ `
${COARSE_COMMON}
var<workgroup> sInv: array<f32, ${COARSEST_CELLS * COARSEST_CELLS}>;
@compute @workgroup_size(1024) fn main(@builtin(local_invocation_index) t: u32) {
  if (t == 0u) {
    // Publish the operator's liquid list as the pressure solve's dispatch arguments.
    let n = atomicExchange(&ctrl[C_LIQ_COUNT], 0u);
    atomicStore(&ctrl[C_N_LIQ], n);
    atomicStore(&args[C_ARG_LIQ], n); atomicStore(&args[C_ARG_LIQB], BULK_WG + n); atomicStore(&args[C_ARG_LIQBB], NBULK + n);
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
  let i = t / NC6; let j = t % NC6;
  let ci = cellOfLevel(6u, i); let cj = cellOfLevel(6u, j);
  let oi = cop(6u, ci); let oj = cop(6u, cj);
  var a = 0.0;
  if (i == j) { a = select(1.0, oi.x, oi.x > 0.0); }
  else if (oi.x > 0.0 && oj.x > 0.0) {
    let d = cj - ci;
    if (all(d == vec3i(1, 0, 0))) { a = -oi.y; } else if (all(d == vec3i(0, 1, 0))) { a = -oi.z; } else if (all(d == vec3i(0, 0, 1))) { a = -oi.w; }
    else if (all(d == vec3i(-1, 0, 0))) { a = -oj.y; } else if (all(d == vec3i(0, -1, 0))) { a = -oj.z; } else if (all(d == vec3i(0, 0, -1))) { a = -oj.w; }
  }
  sInv[t] = a;
  workgroupBarrier();
  for (var p = 0u; p < NC6; p++) {
    let app = sInv[p * NC6 + p]; let aip = sInv[i * NC6 + p]; let apj = sInv[p * NC6 + j];
    workgroupBarrier();
    if (app <= 0.0) { if (t == 0u) { atomicOr(&ctrl[C_FAIL], FAIL_PIVOT); } }
    let r = 1.0 / app;
    if (i == p && j == p) { sInv[t] = r; }
    else if (i == p) { sInv[t] = apj * r; }
    else if (j == p) { sInv[t] = -aip * r; }
    else { sInv[t] = a - aip * apj * r; }
    workgroupBarrier();
    a = sInv[t];
  }
  coarse[CINV + t] = a;
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
/** Fine block Gauss-Seidel on the staged brick tile tp (rhs per thread), 3 red-black sweeps (6 colour passes; 2 took
 *  0.5 more cycles a step, a net loss); black-first when rev. */
fn fineSmooth(c: vec3i, ti: u32, diag: f32, m: u32, rhs: f32, rev: bool) {
  let red = (u32(c.x + c.y + c.z)) & 1u;
  for (var s = 0u; s < 6u; s++) {
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
export const pcgGuessKernel = kernel("seep.pcgGuess", [ctrlRW, bricksRO, opRO, bitsRO, presRW], /* wgsl */ `
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
export const pcgUpdateKernel = kernel("seep.pcgUpdate", [ctrlRW, bricksRO, opRO, bitsRO, presRW, cfgU], /* wgsl */ `
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

/** D0 (see 15). */
export const mgDown0Kernel = kernel("seep.mgDown0", [bricksRO, opRO, bitsRO, presRO, coarseRW], /* wgsl */ `
${levelWGSL()}
${blockWGSL(1)}
const X_OFF: u32 = 0u; const R_OFF: u32 = N; const Z1_OFF: u32 = 5u * N;
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
export const mgDown1Kernel = kernel("seep.mgDown1", [bricksRO, coarseRW, opRO, bitsRO, presRO, ctrlRO], /* wgsl */ `
${levelWGSL()}
${blockWGSL(1)}
${blockWGSL(2, 8)}
const R_OFF: u32 = N; const Z1_OFF: u32 = 5u * N;
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
 * C_k: convergence gate plus the V(2,2)-cycle on levels 3..6, one workgroup.
 * cfg.a = iteration k. If the residual measured by X_k is within tolerance
 * (or the cap is reached) iteration k's remaining passes and every later
 * iteration are disabled through their indirect arguments.
 */
const cgateRW: BindingSpec = { name: "cgate", kind: "rw", type: "array<u32>" };
export const coarseSolveKernel = kernel("seep.coarseSolve", [ctrlRW, coarseRW, cfgU, argsRW, bricksRO, cgateRW], /* wgsl */ `
${COARSE_COMMON}
fn redSlot(rs: u32, which: u32) -> u32 { return C_RED + rs * 192u + which * 64u; }
var<workgroup> sGo: u32;
var<workgroup> sRS: array<f32, 64>;
/** Iteration k's gated arguments: slot 0 the liquid bricks, slot 1 [BULK bricks, liquid], slot 2 [BULK workgroups, liquid],
 *  slot 3 the level-4 blocks; all 0 once converged. */
fn gate(k: u32, nAct: u32, go: bool) {
  for (var s = 0u; s < 5u; s++) {
    let o = C_MG_ARGS + ${MG_STRIDE}u * k + 4u * s;
    var x = nAct + select(select(0u, NBULK, s == 1u), BULK_WG, s == 2u);
    if (s == 3u) { x = atomicLoad(&args[C_ARG_L4]); }
    if (s == 4u) { x = atomicLoad(&args[C_ARG_L4U]); }
    atomicStore(&args[o], select(0u, x, go)); atomicStore(&args[o + 1u], 1u); atomicStore(&args[o + 2u], 1u);
  }
}
fn colourOf(c: vec3i) -> u32 { return (u32(c.x + c.y + c.z)) & 1u; }
var<workgroup> sE: array<f32, 2336>;
var<workgroup> sR: array<f32, 2048>;
fn octI(q: u32) -> vec3i { return vec3i(i32(q & 1u), i32((q >> 1u) & 1u), i32(q >> 2u)); }
/** Workgroup index of (l, c), levels 4..6 at 0 / 2048 / 2304. */
fn sIdx(l: u32, c: vec3i) -> u32 {
  let d = ldims(l);
  return select(select(2304u, 2048u, l == 5u), 0u, l == 4u) + u32(c.x + d.x * (c.y + d.y * c.z));
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
@compute @workgroup_size(1024) fn main(@builtin(local_invocation_index) t: u32) {
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
        if (res > TOL) { atomicAdd(&ctrl[C_CAP_HITS], 1u); }
        // The later iterations' dispatches (coarseSolve included) do not run: close all their gates now.
        for (var j = k; j <= MAXC; j++) { gate(j, 0u, false); if (j > k) { atomicStore(&ctrl[C_RES_HIST + j], 0u); } }
        cgate[4u * k] = 0u; cgate[4u * k + 1u] = 1u; cgate[4u * k + 2u] = 1u;
      } else {
        if (k > 0u) {
          var rho = 0.0;
          for (var j = 0u; j < 64u; j++) { rho += sRS[j]; }
          atomicStore(&ctrl[C_RHO], bitcast<u32>(rho));
        }
        gate(k, atomicLoad(&args[C_ARG_LIQ]), true);
        // coarseSolve k + 1 runs (its argument, in the other buffer of the pair).
        cgate[4u * k] = 1u; cgate[4u * k + 1u] = 1u; cgate[4u * k + 2u] = 1u;
        sGo = 1u;
      }
    } else {
      atomicStore(&ctrl[C_RES_HIST + k], 0u);
      gate(k, 0u, false);
      cgate[4u * k] = 0u; cgate[4u * k + 1u] = 1u; cgate[4u * k + 2u] = 1u;
    }
  }
  let go = workgroupUniformLoad(&sGo);
  if (go == 0u) { return; }
  // Clear dot set k & 1 for iteration k's sums (one slot a thread).
  if (t < 192u) { atomicStore(&ctrl[redSlot(k & 1u, 0u) + t], 0u); }
  // Levels 4..6 (2336 cells) live in workgroup memory for the whole V-cycle:
  // each sweep is one workgroup barrier, not a device-memory fence. Each
  // thread keeps its own cells' couplings (+ and -), rhs and pre-smoothed
  // values in registers. Level 3 is block-smoothed across the GPU by D3 / U3.
  let i0 = t; let i1 = t + 1024u;
  let c40 = cellOfLevel(4u, i0); let c41 = cellOfLevel(4u, i1);
  let o40 = cop(4u, c40); let o41 = cop(4u, c41);
  let m40 = minusC(4u, c40); let m41 = minusC(4u, c41);
  let b40 = coarse[CB + L4O + i0]; let b41 = coarse[CB + L4O + i1];
  let has5 = t < 256u; let has6 = t < 32u;
  let c5 = cellOfLevel(5u, min(t, 255u)); let c6 = cellOfLevel(6u, min(t, 31u));
  var o5 = vec4f(0.0); var m5 = vec3f(0.0); var o6 = vec4f(0.0); var m6 = vec3f(0.0);
  if (has5) { o5 = cop(5u, c5); m5 = minusC(5u, c5); }
  if (has6) { o6 = cop(6u, c6); m6 = minusC(6u, c6); }
  sE[i0] = 0.0; sE[i1] = 0.0;
  if (t < 288u) { sE[2048u + t] = 0.0; }
  workgroupBarrier();
  // Down, level 4: pre-smooth from zero (red first), restrict the residual.
  for (var s = 0u; s < 4u; s++) {
    let col = s & 1u;
    if (colourOf(c40) == col && o40.x > 0.0) { sE[i0] = (b40 + offS(4u, c40, o40, m40)) / o40.x; }
    if (colourOf(c41) == col && o41.x > 0.0) { sE[i1] = (b41 + offS(4u, c41, o41, m41)) / o41.x; }
    workgroupBarrier();
  }
  let ep40 = sE[i0]; let ep41 = sE[i1];
  sR[i0] = select(0.0, b40 - (o40.x * ep40 - offS(4u, c40, o40, m40)), o40.x > 0.0);
  sR[i1] = select(0.0, b41 - (o41.x * ep41 - offS(4u, c41, o41, m41)), o41.x > 0.0);
  workgroupBarrier();
  var b5 = 0.0;
  if (has5) { for (var q = 0u; q < 8u; q++) { b5 += sR[sIdx(4u, 2 * c5 + octI(q))]; } }
  workgroupBarrier();
  for (var s = 0u; s < 4u; s++) {
    if (has5 && colourOf(c5) == (s & 1u) && o5.x > 0.0) { sE[2048u + t] = (b5 + offS(5u, c5, o5, m5)) / o5.x; }
    workgroupBarrier();
  }
  var ep5 = 0.0;
  if (has5) { ep5 = sE[2048u + t]; sR[t] = select(0.0, b5 - (o5.x * ep5 - offS(5u, c5, o5, m5)), o5.x > 0.0); }
  workgroupBarrier();
  var b6 = 0.0;
  if (has6) { for (var q = 0u; q < 8u; q++) { b6 += sR[sIdx(5u, 2 * c6 + octI(q)) - 2048u]; } }
  // Coarsest: the exact solve, one product with the inverse coarsenTop built (inactive rows give 0: b6 = 0).
  workgroupBarrier();
  if (has6) { sR[t] = b6; }
  workgroupBarrier();
  if (has6) {
    var e6 = 0.0;
    for (var q = 0u; q < NC6; q++) { e6 += coarse[CINV + t * NC6 + q] * sR[q]; }
    sE[2304u + t] = e6;
  }
  workgroupBarrier();
  // Up: prolong (piecewise constant) onto the pre-smoothed values, post-smooth black first.
  if (has5) { sE[2048u + t] = select(0.0, ep5 + sE[sIdx(6u, c5 / 2)], o5.x > 0.0); }
  workgroupBarrier();
  for (var s = 0u; s < 4u; s++) {
    if (has5 && colourOf(c5) == 1u - (s & 1u) && o5.x > 0.0) { sE[2048u + t] = (b5 + offS(5u, c5, o5, m5)) / o5.x; }
    workgroupBarrier();
  }
  sE[i0] = select(0.0, ep40 + sE[sIdx(5u, c40 / 2)], o40.x > 0.0);
  sE[i1] = select(0.0, ep41 + sE[sIdx(5u, c41 / 2)], o41.x > 0.0);
  workgroupBarrier();
  for (var s = 0u; s < 4u; s++) {
    let col = 1u - (s & 1u);
    if (colourOf(c40) == col && o40.x > 0.0) { sE[i0] = (b40 + offS(4u, c40, o40, m40)) / o40.x; }
    if (colourOf(c41) == col && o41.x > 0.0) { sE[i1] = (b41 + offS(4u, c41, o41, m41)) / o41.x; }
    workgroupBarrier();
  }
  coarse[CE + L4O + i0] = sE[i0]; coarse[CE + L4O + i1] = sE[i1];
}
`);

/** Level-4 block q4 (2x2x2 level-3 cells) of the level-4 block list. */
const L4_BLOCK_WGSL = /* wgsl */ `
fn l4Origin(q4: u32) -> vec3i { return 2 * vec3i(i32(q4 % 32u), i32((q4 / 32u) % 16u), i32(q4 / 512u)); }
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
export const mgDown3Kernel = kernel("seep.mgDown3", [bricksRO, coarseRW], /* wgsl */ `
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
export const mgRestrict3Kernel = kernel("seep.mgRestrict3", [bricksRO, coarseRW, ctrlRO], /* wgsl */ `
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
export const mgUp3Kernel = kernel("seep.mgUp3", [bricksRO, coarseRW, ctrlRO], /* wgsl */ `
${L3_BLOCK_WGSL}
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  if (t == 0u) { sN = ctrl[C_N_L4]; }
  let n = workgroupUniformLoad(&sN);
  if (wg.x * 8u >= n) { return; }
  for (var j = t; j < 512u; j += 64u) {
    let c = blockOrigin(wg.x, j / 64u, n) + bhalo3(j % 64u);
    var e = 0.0;
    if (linBox(3u, c) && cop3(c).x > 0.0) { e = coarse[CEP + lidx(3u, c)] + coarse[CE + lidx(4u, c / 2)]; }
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
  if (L === 2) return kernel("seep.mgUp2", [bricksRO, coarseRW, ctrlRO], /* wgsl */ `
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
      if (coarse[COP + 4u * g] > 0.0) { e = coarse[CEP + g] + coarse[CE + lidx(3u, c / 2)]; }
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
  return kernel(`seep.mgUp${L}`, [bricksRO, coarseRW], /* wgsl */ `
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
      if (coarse[COP + 4u * g] > 0.0) { e = coarse[CEP + g] + coarse[CE + lidx(${L + 1}u, c / 2)]; }
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
export const pcgPrecondKernel = kernel("seep.pcgPrecond", [ctrlRW, bricksRO, opRO, bitsRO, presRW, coarseRW, cfgU], /* wgsl */ `
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
      if (op[DIAG_OFF + ci] > 0.0) { v = pres[Z1_OFF + ci] + coarse[CE + lidx(1u, c / 2)]; }
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
export const pcgDirectionKernel = kernel("seep.pcgDirection", [ctrlRW, bricksRO, opRO, bitsRO, presRW, cfgU], /* wgsl */ `
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
export const projectKernel = kernel("seep.project", [bricksRO, phisRO, velRW, presRO, bitsRW, bulkRW], /* wgsl */ `
${bulkWGSL(false)}
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
    }
    vel[U_OFF + a * N + i] = u;
  }
  bits[KNOWN_OFF + i] = known;
}
`);

// ---------------------------------------------------------------------------
// 16b. Fine velocity shadow of the BULK bricks next to fine ones: every fine
//      face inside them between two BULK cells takes the trilinear
//      interpolation of the coarse MAC velocity, so fine samplers (advection,
//      extension) read a velocity field there. Interface faces keep their own
//      projected values.
// ---------------------------------------------------------------------------
export const shadowKernel = kernel("seep.shadow", [bricksRO, velRW, bulkRO], /* wgsl */ `
${bulkWGSL(false)}
${COARSE_SAMPLER_WGSL}
// The brick's samples read the coarse faces [2 bc - 1, 2 bc + 2]^3 per component: staged once (cface),
// then interpolated as cvelComp does (same values, same order).
var<workgroup> sf: array<f32, 192>;
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let b = bricks[B_LSHADOW + wg.x];
  let kb = brickCoord(b) * 2 - vec3i(1);
  if (l < 192u) {
    let r = l & 63u;
    sf[l] = cface(l >> 6u, kb + vec3i(i32(r & 3u), i32((r >> 2u) & 3u), i32(r >> 4u)), BU_OFF, U_OFF);
  }
  workgroupBarrier();
  let c = brickCoord(b) * 8 + localCoord(l);
  let i = (b << 9u) | l;
  for (var a = 0u; a < 3u; a++) {
    if (!cellBulk(c - unitAxis(a))) { continue; }
    let q = (vec3f(c) + vec3f(0.5) - 0.5 * vec3f(unitAxis(a))) * 0.25 - vec3f(0.5) + 0.5 * vec3f(unitAxis(a));
    let fb = floor(q); let t = q - fb;
    let lb = vec3i(fb) - kb;
    let k = a * 64u + u32(lb.x + 4 * (lb.y + 4 * lb.z));
    let x00 = mix(sf[k], sf[k + 1u], t.x);
    let x10 = mix(sf[k + 4u], sf[k + 5u], t.x);
    let x01 = mix(sf[k + 16u], sf[k + 17u], t.x);
    let x11 = mix(sf[k + 20u], sf[k + 21u], t.x);
    vel[U_OFF + a * N + i] = mix(mix(x00, x10, t.y), mix(x01, x11, t.y), t.z);
  }
}
`);

// ---------------------------------------------------------------------------
// 17. Velocity extension into air and solid-interior faces: layered averaging
//     of known neighbours (Bridson 2015 §4.3), 8 sweeps per launch on a staged
//     brick + halo, per component. Closed faces stay zero and are not sources.
//     cfg.a = in velocity offset, cfg.b = out offset, cfg.c = in known shift,
//     cfg.d = out known shift. DEEP neighbours (never extended) are read from U.
// ---------------------------------------------------------------------------
export const extendKernel = kernel("seep.extend", [bricksRO, velRW, bitsRW, cfgU], /* wgsl */ `
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
  var anyKnown = false;
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
      if (inside && !faceClosed(q, a)) {
        let k = (info >> select(cfg.c + a, a, deep)) & 1u;
        v = select(UNK, vel[select(cfg.a, U_OFF, deep) + a * N + qi], k == 1u);
        anyKnown = anyKnown || k == 1u;
      }
      vA[a * 1000u + t] = v;
    }
  }
  if (anyKnown) { atomicStore(&sKnown, 1u); }
  workgroupBarrier();
  if (l == 0u) { sKnownU = atomicLoad(&sKnown); }
  // Without a known face in the tile the sweeps change nothing.
  let sweeps = select(0u, 8u, workgroupUniformLoad(&sKnownU) != 0u);
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
//     list counters.
// ---------------------------------------------------------------------------
export const g2pKernel = kernel("seep.g2p", [ctrlRW, bricksRW, partsRW, velRO, cfgU, argsRW, phisRO], /* wgsl */ `
${velocitySampler("velU", "vel", "U_OFF")}
${velocitySampler("velS", "vel", "US_OFF")}
${PARTICLE_MOVE_WGSL}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i == 0u) {
    for (var k = 0u; k < 4u; k++) { atomicStore(&ctrl[C_N_ACTIVE + k], atomicLoad(&args[4u * k])); }
    atomicStore(&args[C_ARG_ACTIVE], 0u); atomicStore(&args[C_ARG_SURF], 0u);
    atomicStore(&args[C_ARG_PART], 0u); atomicStore(&args[C_ARG_CLEAR], 0u);
    atomicStore(&args[C_ARG_ACTB], BULK_WG); atomicStore(&args[C_ARG_ACTBB], NBULK);
    atomicStore(&ctrl[C_PCOUNT + cfg.a], 0u);
  }
  let n = min(atomicLoad(&ctrl[C_PCOUNT + 1u - cfg.a]), CAP);
  if (i >= n) { return; }
  let base = (1u - cfg.a) * CAP * 2u;
  var x = parts[base + i].xyz;
  var v = parts[base + CAP + i].xyz;
  // Supported: a liquid cell among the 2x2x2 whose centres enclose x (the
  // cells the interpolation reads). Unsupported particles (spray, a film away
  // from any liquid cell) have no pressure acting on them and move
  // ballistically with their own momentum under gravity (FLIP in air,
  // Bridson 2015 §7.6.2; the reference's "ballistic" samples); the extended
  // grid velocity there is no force, and following it left splash particles
  // hovering against the walls.
  let bv = vec3i(floor(x - vec3f(0.5)));
  var supported = false;
  for (var k = 0u; k < 8u; k++) {
    let c = bv + vec3i(i32(k & 1u), i32((k >> 1u) & 1u), i32(k >> 2u));
    if (cellOpen(c) && phis[PHIC_OFF + cidx(c)] < 0.0) { supported = true; }
  }
  if (supported) {
    let un = velU(x);
    let ub = velS(x) + vec3f(0.0, GDT, 0.0);
    v = FLIP * (v + un - ub) + (1.0 - FLIP) * un;
    moveParticle(&x, &v, un);
  } else {
    let v1 = v - vec3f(0.0, GDT, 0.0);
    let xn = x + 0.5 * CFL * (v + v1);
    let xc = collide(xn);
    let push = xn - xc;
    v = v1;
    for (var a = 0u; a < 3u; a++) { if (push[a] != 0.0 && v[a] * push[a] > 0.0) { v[a] = 0.0; } }
    x = xc;
  }
  x = collide(x + parts[cfg.a * CAP * 2u + i].xyz);
  parts[base + i] = vec4f(x, 0.0);
  parts[base + CAP + i] = vec4f(v, 0.0);
  atomicAdd(&bricks[B_PCOUNT + brickOfCell(vec3i(floor(x)))], 1u);
}
`);

// ---------------------------------------------------------------------------
// Diagnostics (on demand, never per step): per-brick liquid volume from the
// six-tetrahedron fill over open cells; optional texture publication (init).
// cfg.a = 1 to also write the fill and render textures for every brick;
// cfg.b = 1 to measure phiT (the ledger's mid-step stages) instead of phi.
// ---------------------------------------------------------------------------
export const volumeKernel = kernel("seep.volume", [bricksRO, phisRO, { name: "vsum", kind: "rw", type: "array<f32>" }, volW, cfgU, surfW], /* wgsl */ `
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

export const ALL_KERNELS: readonly KernelSource[] = [
  classifyKernel, clearKernel, countKernel, emitKernel, scanKernel, scatterKernel, separateKernel, advectPhiKernel, combineKernel, massFitKernel,
  redistInitKernel, redistSweepKernel, p2gKernel, velocityBulkKernel, velocityKernel, operatorKernel, coarsenBrickKernel, coarsenTopKernel,
  pcgGuessKernel, pcgUpdateKernel, mgDown0Kernel, mgDown1Kernel, mgDown3Kernel, mgRestrict3Kernel, coarseSolveKernel, mgUp3Kernel, mgUp2Kernel, mgUp1Kernel, pcgPrecondKernel,
  pcgDirectionKernel, projectKernel, shadowKernel, extendKernel, g2pKernel, volumeKernel,
];
