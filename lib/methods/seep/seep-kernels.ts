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
import { CELL_M } from "./seep-layout";
import {
  AGGREGATION_OC, COARSE_CELLS, COMMON_WGSL, LEVEL_OFFSET, MAX_CYCLES, MG_STRIDE, PRESSURE_TOLERANCE,
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
fn brickBulk(id: u32) -> bool { return (${rd("B_CLASS + id")} & CLASS_BULK) != 0u; }
fn cellBulk(c: vec3i) -> bool { return inBox(c) && brickBulk(brickOfCell(c)); }
fn coarseBulk(k: vec3i) -> bool { return all(k >= vec3i(0)) && all(k < vec3i(128, 64, 16)) && brickBulk(brickOfCell(k * 4)); }
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
// 3. Particle census (cfg.a = parity: src pool). Band retirement (Ferstl 2016
//    §3.3): chamber particles deeper than R = 3h are retired, the level set
//    holds that liquid. Particles that leave the particle region (the right
//    chamber, x >= PX0) are retired into the level set.
// ---------------------------------------------------------------------------
export const countKernel = kernel("seep.count", [ctrlRW, partsRO, pcellRW, pgridRW, phisRO, cfgU], /* wgsl */ `
${phiSampler("phi", "phis", "PHI_OFF")}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  let n = min(atomicLoad(&ctrl[C_PCOUNT + cfg.a]), CAP);
  if (i >= n) { return; }
  let p = parts[cfg.a * CAP * 2u + 2u * i].xyz;
  let c = vec3i(floor(p));
  if (c.x < PX0 || !cellOpen(c)) { pcell[i] = DEAD_PARTICLE; atomicAdd(&ctrl[C_DELETED], 1u); atomicAdd(&ctrl[C_DEL_OUT], 1u); return; }
  let depth = phiTri(p);
  if (depth < -BAND) { pcell[i] = DEAD_PARTICLE; atomicAdd(&ctrl[C_DELETED], 1u); atomicAdd(&ctrl[C_DEL_DEEP], 1u); return; }
  let pc = pidx(c);
  let f = p - floor(p);
  let o = select(0u, 1u, f.x >= 0.5) | select(0u, 2u, f.y >= 0.5) | select(0u, 4u, f.z >= 0.5);
  // Resampling cap (Ferstl 2016 §3.3): a cell keeps at most 2n = 16 particles,
  // applied where the level set holds the liquid (a cell deep or more, as the
  // reference's deep overlap); at the surface the particles are the liquid.
  let before = atomicAdd(&pgrid[pc], 1u);
  if ((before & 0xffffu) >= 16u && depth < -1.0) { atomicSub(&pgrid[pc], 1u); pcell[i] = DEAD_PARTICLE; atomicAdd(&ctrl[C_DELETED], 1u); atomicAdd(&ctrl[C_DEL_CAP], 1u); return; }
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
// 4. Band seeding (Ferstl 2016 §3.3 re-sampling), after the level set's
//    advection, particle combination and redistance, so the level set and the
//    census (the survivors after last step's G2P) describe the same instant.
//    A candidate is the jittered centre of an octant no survivor occupies, in
//    an open particle-region cell, inside the liquid (an octant is an eighth
//    of the cell, the volume a particle carries in the combine's bound) and
//    within the band. Inner band (cell centre deeper than 1.5h): the cell is
//    topped up to eight. Inflow (liquid whose one-step departure point lies
//    behind the throat exit x = PX0, the only way liquid enters the particle
//    region): every octant, as a source seeds its whole band. Elsewhere in
//    the outer layer only where no survivor lies in the cell or its 26
//    neighbours; where particles are present they carry the surface. Seeds take the grid velocity, are
//    appended to the sorted pool past its cell ranges (this step's P2G does
//    not see them) and move in this step's G2P like every particle.
//    cfg.a = parity (sorted pool = dst).
// ---------------------------------------------------------------------------
export const seedKernel = kernel("seep.seed", [ctrlRW, bricksRO, partsRW, pgridRO, phisRO, velRO, cfgU], /* wgsl */ `
${phiSampler("phi", "phis", "PHI_OFF")}
${velocitySampler("velU", "vel", "U_OFF")}
// One-step departure point of x (RK2 midpoint, segments of at most one cell,
// as the velocity and level-set advection use).
fn departure(x0: vec3f) -> vec3f {
  let u0 = velU(x0);
  let n = clamp(ceil(length(u0) * CFL), 1.0, MAXSEG);
  let tau = CFL / n;
  var x = x0;
  for (var s = 0.0; s < n; s += 1.0) {
    let k1 = select(velU(x), u0, s == 0.0);
    x = x - tau * velU(x - 0.5 * tau * k1);
  }
  return x;
}
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let b = bricks[B_LPART + wg.x];
  let c = brickCoord(b) * 8 + localCoord(l);
  if (!cellOpen(c) || c.x < PX0) { return; }
  let cphi = phiTri(vec3f(c) + vec3f(0.5));
  if (cphi < -BAND || cphi >= 1.0) { return; }
  let word = pgrid[pidx(c)];
  let occ = (word >> 16u) & 0xffu;
  let have = word & 0xffffu;
  if (have >= 8u || occ == 0xffu) { return; }
  var budget = 8u - have;
  var uncovered = true;
  for (var dz = -1; dz <= 1; dz++) { for (var dy = -1; dy <= 1; dy++) { for (var dx = -1; dx <= 1; dx++) {
    let n = c + vec3i(dx, dy, dz);
    if (inBox(n) && n.x >= PX0 && ((pgrid[pidx(n)] >> 16u) & 0xffu) != 0u) { uncovered = false; }
  } } }
  let inner = cphi < -1.5;
  // Inflow: the only liquid entering the particle region crosses x = PX0 (the
  // throat exit) this step; its octants are seeded whatever particles are
  // near (a source band, as the reference adds a source's whole band).
  let inflowCell = c.x < PX0 + 16;
  let step = atomicLoad(&ctrl[C_STEP]);
  let pc = pidx(c);
  let base = (1u - cfg.a) * CAP * 2u;
  for (var o = 0u; o < 8u; o++) {
    if (((occ >> o) & 1u) != 0u || budget == 0u) { continue; }
    let j = vec3f(hashUnit(step, pc, o), hashUnit(step, pc, o + 8u), hashUnit(step, pc, o + 16u)) - vec3f(0.5);
    let x = vec3f(c) + vec3f(0.25) + 0.5 * octOffset(o) + 0.2 * j;
    let px = phiTri(x);
    if (px >= 0.0 || px < -BAND) { continue; }
    let xd = departure(x);
    let inflow = inflowCell && xd.x < f32(PX0);
    if (!inflow && !inner && !uncovered) { continue; }
    budget -= 1u;
    let k = atomicAdd(&ctrl[C_PCOUNT + 1u - cfg.a], 1u);
    if (k >= CAP) { atomicOr(&ctrl[C_FAIL], FAIL_POOL); atomicAdd(&ctrl[C_OVERFLOW], 1u); continue; }
    parts[base + 2u * k] = vec4f(x, 0.0);
    // The seed carries the semi-Lagrangian advected velocity u^n(departure),
    // i.e. the grid velocity at t_{n+1} the liquid it samples would have had
    // without particles (Stam 1999; new NB-FLIP particles take the advected
    // grid velocity, Ferstl et al. 2016 §4). Sampling u^n at the landing point
    // instead would hand jet liquid the stale velocity of whatever was there.
    parts[base + 2u * k + 1u] = vec4f(velU(xd), 0.0);
    atomicAdd(&ctrl[C_SEEDED], 1u);
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
  parts[dst + 2u * slot] = parts[src + 2u * i];
  parts[dst + 2u * slot + 1u] = parts[src + 2u * i + 1u];
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
  let x = parts[base + 2u * i].xyz;
  let bv = vec3i(floor(x - vec3f(0.5)));
  var d = vec3f(0.0); var m = 0.0;
  for (var k = 0u; k < 8u; k++) {
    let c = bv + vec3i(i32(k & 1u), i32((k >> 1u) & 1u), i32(k >> 2u));
    if (!inBox(c) || c.x < PX0) { continue; }
    let pc = pidx(c);
    let start = pgrid[PN + pc]; let cnt = pgrid[pc] & 0xffffu;
    for (var j = start; j < start + cnt; j++) {
      if (j == i) { continue; }
      let e = x - parts[base + 2u * j].xyz;
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
  parts[cfg.a * CAP * 2u + 2u * i] = vec4f(d, 0.0);
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
// 8. Particle surface in chamber bricks (Ferstl 2016 Eq. 4 as a union, with
//    the mass-based bound the plan §2.2 allows): phi = min(phi0, phi_p).
//    phi_p is the union of one sphere of radius RS per particle (the nearest
//    particle in the 8 cells sharing the vertex; Enright et al. 2002 /
//    Foster-Fedkiw 2001 particle spheres), bounded below by the particles'
//    own volume: eight particles fill a cell, so where the open cells around a
//    vertex hold a fraction f of that, the liquid they carry is a plane 1 - 2f
//    cells above the vertex, and phi_p is no lower (the union adds no more
//    liquid than the particles are). At a solid (closed cells on one side,
//    which read as full) the plane fills from the wall instead: along the axis
//    whose two layers of cells differ most in held fraction, phi_p >= -f_sparse
//    against a closed dense layer, 1 - f_dense - f_sparse otherwise; both hold
//    exactly the particles' volume. This is NB-FLIP's surface (sphere 0.875h,
//    sampleBound "walls"), the reference the gates compare with: a floor film
//    of one particle layer holds its h/8 a particle, where a Zhu-Bridson
//    surface (radius ~h/4) read the same layer as almost no liquid and the
//    grid's advection thinned the film every step. No shrink of phi0: the
//    particles restore liquid grid advection erodes, never erase what the
//    level set holds. A vertex with no particle in its 8 cells keeps phi0.
//    cfg.a = parity (sorted pool = dst).
// ---------------------------------------------------------------------------
export const combineKernel = kernel("seep.combine", [bricksRO, phisRW, partsRO, pgridRO, cfgU], /* wgsl */ `
const RS: f32 = 0.875;
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let b = bricks[B_LPART + wg.x];
  if ((bricks[B_CLASS + b] & CLASS_CHAMBER) == 0u) { return; }
  let v = brickCoord(b) * 8 + localCoord(l);
  let vf = vec3f(v);
  let base = (1u - cfg.a) * CAP * 2u;
  var near2 = 4.0; var gathered = 0u;
  var held = 0u; var open = 0u;
  var heldLo = vec3u(0u); var heldHi = vec3u(0u); var openLo = vec3u(0u); var openHi = vec3u(0u);
  for (var k = 0u; k < 8u; k++) {
    let bit = vec3u(k & 1u, (k >> 1u) & 1u, k >> 2u);
    let c = v - vec3i(1) + vec3i(bit);
    if (!cellOpen(c)) { continue; }
    var cnt = 0u;
    if (c.x >= PX0) {
      let pc = pidx(c);
      let start = pgrid[PN + pc]; cnt = pgrid[pc] & 0xffffu;
      for (var j = start; j < start + cnt; j++) {
        let d = parts[base + 2u * j].xyz - vf;
        near2 = min(near2, dot(d, d));
      }
    }
    gathered += cnt;
    let n = min(cnt, 8u);
    open += 1u; held += n;
    let hi = bit != vec3u(0u);
    heldHi += select(vec3u(0u), vec3u(n), hi); openHi += select(vec3u(0u), vec3u(1u), hi);
    heldLo += select(vec3u(n), vec3u(0u), hi); openLo += select(vec3u(1u), vec3u(0u), hi);
  }
  if (gathered == 0u) { return; }
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
  let i = PHIT_OFF + ((b << 9u) | l);
  let phi0 = phis[i];
  if (phi0 <= -RCOMB) { return; }
  let phiS = sqrt(near2) - RS;
  // Particles deeper than the grid: add, up to their volume. Grid deeper than
  // the particles: erase towards the particle spheres, at most h a step (Eq. 4).
  let phi = select(min(phiS, phi0 + 1.0), max(phiS, min(phi0, bound)), phi0 > phiS);
  phis[i] = clamp(phi, -PHI_MAX, PHI_MAX);
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
fn rd(v: vec3i) -> f32 { return phis[PHIT_OFF + cidx(clampV(v))]; }
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let b = bricks[B_LSURF + wg.x];
  let v = brickCoord(b) * 8 + localCoord(l);
  if (!vertexOpen(v)) { phis[PHI_OFF + ((b << 9u) | l)] = UNKNOWN_PHI; return; }
  let a = rd(v);
  let neg = a < 0.0;
  var cut = false;
  for (var k = 0u; k < 27u; k++) {
    let w = v + vec3i(i32(k % 3u), i32((k / 3u) % 3u), i32(k / 9u)) - vec3i(1);
    if (k == 13u || any(w < vec3i(0)) || any(w >= NV) || !vertexOpen(w)) { continue; }
    if ((rd(w) < 0.0) != neg) { cut = true; }
  }
  let out = select(select(PHI_MAX, -PHI_MAX, neg), a, cut);
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
  for (var t = l; t < 1000u; t += 512u) {
    let q = clampV(o + tileCoord(t));
    let val = phis[cfg.a + cidx(q)];
    tA[t] = val; tB[t] = val; tOpen[t] = select(0u, 1u, vertexOpen(q));
    if (val != PHI_MAX) { atomicStore(&sMixed, 1u); }
  }
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
    let xp = parts[base + 2u * k].xyz; let vp = parts[base + 2u * k + 1u].xyz;
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
/** BULK coarse cell j: SL advection of its coarse low faces at 4h; the fine interface faces stored in it (fine cell below along a) as fine faces. */
fn bulkVelocity(j: u32) {
  let k = bulkCell(j);
  for (var a = 0u; a < 3u; a++) {
    let e = unitAxis(a);
    let kn = k - e;
    if (coarseBulk(kn)) {
      let pos = vec3f(4 * k) + vec3f(2.0) - 2.0 * vec3f(e);
      let u0 = cvel(pos, BU_OFF, U_OFF);
      let n = clamp(ceil(length(u0) * CFL * 0.25), 1.0, MAXSEG);
      let tau = CFL / n;
      var x = pos;
      for (var s = 0.0; s < n; s += 1.0) {
        let k1 = select(cvel(x, BU_OFF, U_OFF), u0, s == 0.0);
        x = x - tau * cvel(x - 0.5 * tau * k1, BU_OFF, U_OFF);
      }
      bulk[BUS_OFF + a * NCL + clIndex(k)] = cvelComp(a, x, BU_OFF, U_OFF) - select(0.0, GDT, a == 1u);
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
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let j = wg.x * 512u + l;
  if (j < NBULK * 8u) { bulkVelocity(j); }
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
fn ph(v: vec3i) -> f32 { return phis[PHI_OFF + cidx(clampV(v))]; }
const FAR_AIR: f32 = 4.0;
var<workgroup> red: array<f32, 512>;
var<workgroup> sFar: u32;
// Velocity tile: the brick's faces and a two-cell halo of each component,
// staged once; backtrace samples whose eight corners fall inside it read the
// tile, the rest (fast liquid, a few cells a step) read the buffer. Same
// values, same interpolation.
const VT: i32 = 12;
var<workgroup> vt: array<f32, 5184>;
var<private> tileOrigin: vec3i;
fn velTComp(a: u32, p: vec3f) -> f32 {
  let q = p - vec3f(0.5) + 0.5 * vec3f(unitAxis(a));
  let fb = floor(q);
  let lb = vec3i(fb) - tileOrigin;
  if (any(lb < vec3i(0)) || any(lb > vec3i(VT - 2))) { return velUComp(a, p); }
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
  red[l] = select(-PHI_MAX, phis[PHI_OFF + i], !deep);
  workgroupBarrier();
  for (var d = 256u; d > 0u; d = d >> 1u) {
    if (l < d) { red[l] = min(red[l], red[l + d]); }
    workgroupBarrier();
  }
  if (l == 0u && !deep) { atomicStore(&bricks[B_MINPHI + b], bitcast<u32>(red[0])); }
  // A brick whose vertices, and the layer below each axis, all lie beyond FAR_AIR has only faces
  // the per-face test below would skip: no tile, no traces (same values).
  var far = 0.0;
  if (!deep) {
    far = PHI_MAX;
    for (var t = l; t < 1000u; t += 512u) { far = min(far, ph(brickCoord(b) * 8 + tileCoord(t))); }
  }
  workgroupBarrier();
  red[l] = far;
  workgroupBarrier();
  for (var d = 256u; d > 0u; d = d >> 1u) {
    if (l < d) { red[l] = min(red[l], red[l + d]); }
    workgroupBarrier();
  }
  if (l == 0u) { sFar = select(0u, 1u, red[0] > FAR_AIR); }
  let allFar = workgroupUniformLoad(&sFar) != 0u;
  if (!allFar) {
    for (var t = l; t < 5184u; t += 512u) {
      let a = t / 1728u; let r = t % 1728u;
      let q = tileOrigin + vec3i(i32(r % 12u), i32((r / 12u) % 12u), i32(r / 144u));
      vt[t] = velUFace(a, q);
    }
  }
  workgroupBarrier();
  let part = c.x >= PX0;
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
    if (vmin > FAR_AIR) {
      let o = c - unitAxis(a);
      let t1 = unitAxis((a + 1u) % 3u); let t2 = unitAxis((a + 2u) % 3u);
      if (min(min(ph(o), ph(o + t1)), min(ph(o + t2), ph(o + t1 + t2))) > FAR_AIR) { vel[US_OFF + a * N + i] = 0.0; continue; }
    }
    var transfer = false;
    if (wsum > 0) {
      let t1 = unitAxis((a + 1u) % 3u); let t2 = unitAxis((a + 2u) % 3u);
      let phiF = 0.25 * (ph(c) + ph(c + t1) + ph(c + t2) + ph(c + t1 + t2));
      transfer = phiF >= -RCOMB;
    }
    var val = 0.0;
    if (transfer) {
      val = (f32(mom) / SM) / (f32(wsum) / SW);
    } else {
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
@compute @workgroup_size(1024) fn main(@builtin(local_invocation_index) t: u32) {
  if (t == 0u) {
    // Publish the operator's liquid list as the pressure solve's dispatch arguments.
    let n = atomicExchange(&ctrl[C_LIQ_COUNT], 0u);
    atomicStore(&ctrl[C_N_LIQ], n);
    atomicStore(&args[C_ARG_LIQ], n); atomicStore(&args[C_ARG_LIQB], BULK_WG + n); atomicStore(&args[C_ARG_LIQBB], NBULK + n);
    let n4 = atomicExchange(&ctrl[C_L4_COUNT], 0u);
    atomicStore(&ctrl[C_N_L4], n4);
    atomicStore(&args[C_ARG_L4], n4); atomicStore(&args[C_ARG_L4 + 1u], 1u); atomicStore(&args[C_ARG_L4 + 2u], 1u);
  }
  for (var l = 4u; l <= 6u; l++) {
    for (var i = t; i < levelCount(l); i += 1024u) { aggregate(l, cellOfLevel(l, i)); }
    storageBarrier(); workgroupBarrier();
  }
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
fn redSum(rs: u32, which: u32) -> f32 {
  var s = 0.0;
  for (var j = 0u; j < 64u; j++) { s += bitcast<f32>(atomicLoad(&ctrl[redSlot(rs, which) + j])); }
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
var<workgroup> tp: array<f32, 1000>;
/** Fine block Gauss-Seidel on the staged brick tile tp (rhs per thread), 2 sweeps; black-first when rev. */
fn fineSmooth(c: vec3i, ti: u32, diag: f32, m: u32, rhs: f32, rev: bool) {
  let red = (u32(c.x + c.y + c.z)) & 1u;
  for (var s = 0u; s < 4u; s++) {
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
 * one-cell halo). Thread t < s^3 owns cell t of the block.
 */
function blockWGSL(L: 1 | 2 | 3): string {
  const s = L === 1 ? 4 : 2, T = s + 2;
  return /* wgsl */ `
const BS${L}: i32 = ${s}; const BT${L}: u32 = ${T}u;
var<workgroup> tb${L}: array<f32, ${T * T * T}>;
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

/** X_k (see 15). cfg.a = k. */
export const pcgUpdateKernel = kernel("seep.pcgUpdate", [ctrlRW, bricksRO, opRO, bitsRO, presRW, cfgU], /* wgsl */ `
${PCG_WGSL}
${bulkWGSL(false)}
var<workgroup> sAlpha: f32;
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let k = cfg.a;
  if (l == 0u) {
    sAlpha = 0.0;
    if (k > 0u) {
      let pq = redSum((k - 1u) & 1u, ${PQ}u);
      sAlpha = select(0.0, redSum((k - 1u) & 1u, ${RZ}u) / pq, pq != 0.0);
    }
  }
  let alpha = workgroupUniformLoad(&sAlpha);
  if (wg.x < BULK_WG) {
    // BULK coarse cells: no fine smoothing (their preconditioner is level 2). The residual of a coarse
    // row sums 64 fine cells' divergence: measured per fine cell (/ 64).
    let j = wg.x * 512u + l;
    var r = 0.0;
    if (j < NBULK * 8u) {
      let kc = bulkCell(j);
      let sl = bulkSlot(kc);
      if (k == 0u) { r = op[RHS_OFF + sl] - applyBulk(kc, sl, X_OFF); }
      else {
        pres[X_OFF + sl] += alpha * pres[PP_OFF + sl];
        r = pres[R_OFF + sl] - alpha * pres[Q_OFF + sl];
      }
      pres[R_OFF + sl] = r;
    }
    sRedB[l] = abs(r) / 64.0;
    workgroupBarrier();
    for (var d = 256u; d > 0u; d = d >> 1u) {
      if (l < d) { sRedB[l] = max(sRedB[l], sRedB[l + d]); }
      workgroupBarrier();
    }
    if (l == 0u) { atomicMax(&ctrl[C_RES], bitcast<u32>(sRedB[0])); }
    return;
  }
  let b = bricks[B_LLIQ + wg.x - BULK_WG];
  let lc = localCoord(l);
  let c = brickCoord(b) * 8 + lc;
  let i = (b << 9u) | l;
  let diag = op[DIAG_OFF + i];
  var r = 0.0;
  if (diag > 0.0) {
    if (k == 0u) { r = op[RHS_OFF + i] - applyA(c, i, X_OFF); }
    else {
      pres[X_OFF + i] += alpha * pres[PP_OFF + i];
      r = pres[R_OFF + i] - alpha * pres[Q_OFF + i];
    }
  }
  pres[R_OFF + i] = r;
  sRedB[l] = abs(r);
  for (var t = l; t < 1000u; t += 512u) { tp[t] = 0.0; }
  workgroupBarrier();
  for (var d = 256u; d > 0u; d = d >> 1u) {
    if (l < d) { sRedB[l] = max(sRedB[l], sRedB[l + d]); }
    workgroupBarrier();
  }
  if (l == 0u) { atomicMax(&ctrl[C_RES], bitcast<u32>(sRedB[0])); }
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
export const mgDown1Kernel = kernel("seep.mgDown1", [bricksRO, coarseRW, opRO, bitsRO, presRO], /* wgsl */ `
${levelWGSL()}
${blockWGSL(1)}
${blockWGSL(2)}
const R_OFF: u32 = N; const Z1_OFF: u32 = 5u * N;
${BULK_APPLY_WGSL}
var<workgroup> sr: array<f32, 64>;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let b = bricks[B_LBULK2 + wg.x];
  let bc = brickCoord(b);
  if (wg.x < NBULK) {
    // BULK brick: its coarse cells are level-2 cells. Their residual after the fine pre-smoothing
    // r - A z1 (z1 = 0 on coarse cells), then level-2 pre-smoothing like any brick block.
    if (t < 64u) { tb2[t] = 0.0; }
    workgroupBarrier();
    let own = t < 8u;
    let q2 = bq2(t);
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
    let ti2 = bti2(q2);
    bsmooth2(own, c2, ti2, o2, w2, b2, false);
    if (own) { coarse[CEP + lidx(2u, c2)] = select(0.0, tb2[ti2], o2.x > 0.0); }
    return;
  }
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
export const mgDown2Kernel = kernel("seep.mgDown2", [bricksRO, coarseRW], /* wgsl */ `
${COARSE_COMMON}
var<workgroup> sr: array<f32, 8>;
@compute @workgroup_size(8) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let b = bricks[B_LBULK2 + wg.x];
  let bc = brickCoord(b);
  sr[t] = residC(2u, bc * 2 + vec3i(i32(t & 1u), i32((t >> 1u) & 1u), i32(t >> 2u)), CEP);
  workgroupBarrier();
  if (t == 0u) {
    var s = 0.0;
    for (var k = 0u; k < 8u; k++) { s += sr[k]; }
    coarse[CB + lidx(3u, bc)] = s;
  }
}
`);

/**
 * C_k: convergence gate plus the V(2,2)-cycle on levels 3..6, one workgroup.
 * cfg.a = iteration k. If the residual measured by X_k is within tolerance
 * (or the cap is reached) iteration k's remaining passes and every later
 * iteration are disabled through their indirect arguments.
 */
export const coarseSolveKernel = kernel("seep.coarseSolve", [ctrlRW, coarseRW, cfgU, argsRW, bricksRO], /* wgsl */ `
${COARSE_COMMON}
fn redSlot(rs: u32, which: u32) -> u32 { return C_RED + rs * 192u + which * 64u; }
fn redSum(rs: u32, which: u32) -> f32 {
  var s = 0.0;
  for (var j = 0u; j < 64u; j++) { s += bitcast<f32>(atomicLoad(&ctrl[redSlot(rs, which) + j])); }
  return s;
}
var<workgroup> sGo: u32;
/** Iteration k's gated arguments: slot 0 the liquid bricks, slot 1 [BULK bricks, liquid], slot 2 [BULK workgroups, liquid], slot 3 the level-4 blocks; all 0 once converged. */
fn gate(k: u32, nAct: u32, go: bool) {
  for (var s = 0u; s < 4u; s++) {
    let o = C_MG_ARGS + ${MG_STRIDE}u * k + 4u * s;
    var x = nAct + select(select(0u, NBULK, s == 1u), BULK_WG, s == 2u);
    if (s == 3u) { x = atomicLoad(&args[C_ARG_L4]); }
    atomicStore(&args[o], select(0u, x, go)); atomicStore(&args[o + 1u], 1u); atomicStore(&args[o + 2u], 1u);
  }
}
fn colourOf(c: vec3i) -> u32 { return (u32(c.x + c.y + c.z)) & 1u; }
@compute @workgroup_size(1024) fn main(@builtin(local_invocation_index) t: u32) {
  let k = cfg.a;
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
        gate(k, 0u, false);
      } else {
        if (k > 0u) { atomicStore(&ctrl[C_RHO], bitcast<u32>(redSum((k - 1u) & 1u, ${RZ}u))); }
        for (var j = 0u; j < 192u; j++) { atomicStore(&ctrl[redSlot(k & 1u, 0u) + j], 0u); }
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
  // Down: pre-smooth from zero (red first), restrict the residual. Level 3
  // is block-smoothed across the GPU by D3 / U3 (level-4 blocks); here 4..6.
  for (var l = 4u; l <= 5u; l++) {
    for (var i = t; i < levelCount(l); i += 1024u) { coarse[CEP + loff(l) + i] = 0.0; }
    storageBarrier(); workgroupBarrier();
    for (var s = 0u; s < 4u; s++) {
      for (var i = t; i < levelCount(l); i += 1024u) { let c = cellOfLevel(l, i); if (colourOf(c) == (s & 1u)) { relaxC(l, c, CEP); } }
      storageBarrier(); workgroupBarrier();
    }
    for (var i = t; i < levelCount(l + 1u); i += 1024u) {
      let c = cellOfLevel(l + 1u, i);
      var s = 0.0;
      for (var q = 0u; q < 8u; q++) { s += residC(l, 2 * c + vec3i(i32(q & 1u), i32((q >> 1u) & 1u), i32(q >> 2u)), CEP); }
      coarse[CB + lidx(l + 1u, c)] = s;
    }
    storageBarrier(); workgroupBarrier();
  }
  // Coarsest: palindromic red-black sweeps from zero (red, black, ..., red).
  if (t < 32u) { coarse[CE + L6O + t] = 0.0; }
  storageBarrier(); workgroupBarrier();
  for (var s = 0u; s < 41u; s++) {
    if (t < 32u) { let c = cellOfLevel(6u, t); if (colourOf(c) == (s & 1u)) { relaxC(6u, c, CE); } }
    storageBarrier(); workgroupBarrier();
  }
  // Up: prolong (piecewise constant) onto the pre-smoothed values, post-smooth black first.
  for (var l = 5u; l >= 4u; l--) {
    for (var i = t; i < levelCount(l); i += 1024u) {
      let c = cellOfLevel(l, i);
      coarse[CE + loff(l) + i] = select(0.0, coarse[CEP + loff(l) + i] + coarse[CE + lidx(l + 1u, c / 2)], cop(l, c).x > 0.0);
    }
    storageBarrier(); workgroupBarrier();
    for (var s = 0u; s < 4u; s++) {
      for (var i = t; i < levelCount(l); i += 1024u) { let c = cellOfLevel(l, i); if (colourOf(c) == 1u - (s & 1u)) { relaxC(l, c, CE); } }
      storageBarrier(); workgroupBarrier();
    }
  }
}
`);

/** Level-4 block q4 (2x2x2 level-3 cells) of the level-4 block list. */
const L4_BLOCK_WGSL = /* wgsl */ `
fn l4Origin(q4: u32) -> vec3i { return 2 * vec3i(i32(q4 % 32u), i32((q4 / 32u) % 16u), i32(q4 / 512u)); }
`;
/**
 * D3_k / U3_k: level-3 block Gauss-Seidel over the level-4 blocks (2x2x2
 * level-3 cells, 4 red-black sweeps inside), the blocks in two colours by the
 * parity of their level-4 cell (the 7-point stencil couples only opposite
 * colours). Pre-smoothing from zero: red blocks (halo zero), then black blocks
 * (halo: the red result); then the residual restricted to level 4. Post-
 * smoothing in the adjoint order: black blocks (halo: pre-smoothed plus
 * prolonged), then red (halo: black's result). cfg.a = colour.
 */
const L3_BLOCK_WGSL = /* wgsl */ `
${levelWGSL()}
${blockWGSL(3)}
${L4_BLOCK_WGSL}
fn colour4(origin: vec3i) -> u32 { let q = origin / 2; return u32(q.x + q.y + q.z) & 1u; }
fn cop3(c: vec3i) -> vec4f { let g = lidx(3u, c); return vec4f(coarse[COP + 4u * g], coarse[COP + 4u * g + 1u], coarse[COP + 4u * g + 2u], coarse[COP + 4u * g + 3u]); }
`;
export const mgDown3Kernel = kernel("seep.mgDown3", [bricksRO, coarseRW, cfgU], /* wgsl */ `
${L3_BLOCK_WGSL}
@compute @workgroup_size(8) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let origin = l4Origin(bricks[B_L4LIST + wg.x]);
  if (colour4(origin) != cfg.a) { return; }
  for (var j = t; j < 64u; j += 8u) {
    let c = origin + bhalo3(j);
    var e = 0.0;
    if (cfg.a == 1u && linBox(3u, c) && cop3(c).x > 0.0) { e = coarse[CEP + lidx(3u, c)]; }
    tb3[j] = e;
  }
  workgroupBarrier();
  let q = bq3(t);
  let c = origin + q;
  let g = lidx(3u, c);
  let o = cop3(c);
  let ti = bti3(q);
  tb3[ti] = 0.0;
  workgroupBarrier();
  bsmooth3(true, c, ti, o, bw3(c), coarse[CB + g], false);
  coarse[CEP + g] = select(0.0, tb3[ti], o.x > 0.0);
}
`);
/** D3r_k: level-3 residual of each level-4 block restricted to level 4 (after both colours). */
export const mgRestrict3Kernel = kernel("seep.mgRestrict3", [bricksRO, coarseRW], /* wgsl */ `
${L3_BLOCK_WGSL}
var<workgroup> sr: array<f32, 8>;
@compute @workgroup_size(8) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let origin = l4Origin(bricks[B_L4LIST + wg.x]);
  for (var j = t; j < 64u; j += 8u) {
    let c = origin + bhalo3(j);
    var e = 0.0;
    if (linBox(3u, c) && cop3(c).x > 0.0) { e = coarse[CEP + lidx(3u, c)]; }
    tb3[j] = e;
  }
  workgroupBarrier();
  let q = bq3(t);
  let c = origin + q;
  let o = cop3(c);
  let ti = bti3(q);
  sr[t] = select(0.0, coarse[CB + lidx(3u, c)] - (o.x * tb3[ti] - bsum3(ti, o, bw3(c))), o.x > 0.0);
  workgroupBarrier();
  if (t == 0u) {
    var s = 0.0;
    for (var k = 0u; k < 8u; k++) { s += sr[k]; }
    coarse[CB + lidx(4u, origin / 2)] = s;
  }
}
`);
export const mgUp3Kernel = kernel("seep.mgUp3", [bricksRO, coarseRW, cfgU], /* wgsl */ `
${L3_BLOCK_WGSL}
@compute @workgroup_size(8) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let origin = l4Origin(bricks[B_L4LIST + wg.x]);
  let mine = colour4(origin);
  // Black (1) first; red (0) then sees black's final values.
  if (mine != cfg.a) { return; }
  for (var j = t; j < 64u; j += 8u) {
    let c = origin + bhalo3(j);
    var e = 0.0;
    if (linBox(3u, c) && cop3(c).x > 0.0) {
      let g = lidx(3u, c);
      let inside = all(c >= origin) && all(c < origin + vec3i(2));
      if (inside || mine == 1u) { e = coarse[CEP + g] + coarse[CE + lidx(4u, c / 2)]; } else { e = coarse[CE + g]; }
    }
    tb3[j] = e;
  }
  workgroupBarrier();
  let q = bq3(t);
  let c = origin + q;
  let g = lidx(3u, c);
  let o = cop3(c);
  let ti = bti3(q);
  bsmooth3(true, c, ti, o, bw3(c), coarse[CB + g], true);
  coarse[CE + g] = select(0.0, tb3[ti], o.x > 0.0);
}
`);

/** U2 / U1: post-smoothing of level L on the brick's block after prolongation from L+1 (U2 over [BULK, active]). */
function upKernel(L: 1 | 2): KernelSource {
  const s = L === 1 ? 4 : 2, T = s + 2, threads = s * s * s;
  return kernel(`seep.mgUp${L}`, [bricksRO, coarseRW], /* wgsl */ `
${levelWGSL()}
${blockWGSL(L)}
@compute @workgroup_size(${threads}) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let b = bricks[${L === 2 ? "B_LBULK2" : "B_LLIQ"} + wg.x];
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
    workgroupBarrier();
    for (var d = 256u; d > 0u; d = d >> 1u) {
      if (l < d) { sRedA[l] += sRedA[l + d]; }
      workgroupBarrier();
    }
    if (l == 0u) { redAdd(k & 1u, ${RZ}u, wg.x, sRedA[0]); }
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
  workgroupBarrier();
  for (var d = 256u; d > 0u; d = d >> 1u) {
    if (l < d) { sRedA[l] += sRedA[l + d]; }
    workgroupBarrier();
  }
  if (l == 0u) { redAdd(k & 1u, ${RZ}u, wg.x, sRedA[0]); }
}
`);

/** P_k (see 15). cfg.a = k. */
export const pcgDirectionKernel = kernel("seep.pcgDirection", [ctrlRW, bricksRO, opRO, bitsRO, presRW, cfgU], /* wgsl */ `
${PCG_WGSL}
${bulkWGSL(false)}
var<workgroup> sBeta: f32;
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let k = cfg.a;
  if (l == 0u) {
    sBeta = 0.0;
    if (k > 0u) {
      let rho = bitcast<f32>(atomicLoad(&ctrl[C_RHO]));
      sBeta = select(0.0, redSum(k & 1u, ${RZ}u) / rho, rho != 0.0);
    }
  }
  let beta = workgroupUniformLoad(&sBeta);
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
  workgroupBarrier();
  for (var d = 256u; d > 0u; d = d >> 1u) {
    if (l < d) { sRedA[l] += sRedA[l + d]; }
    workgroupBarrier();
  }
  if (l == 0u) { redAdd(k & 1u, ${PQ}u, wg.x, sRedA[0]); }
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
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let b = bricks[B_LSHADOW + wg.x];
  let c = brickCoord(b) * 8 + localCoord(l);
  let i = (b << 9u) | l;
  for (var a = 0u; a < 3u; a++) {
    if (!cellBulk(c - unitAxis(a))) { continue; }
    vel[U_OFF + a * N + i] = cvelComp(a, vec3f(c) + vec3f(0.5) - 0.5 * vec3f(unitAxis(a)), BU_OFF, U_OFF);
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
var<workgroup> vA: array<f32, 1000>;
var<workgroup> vB: array<f32, 1000>;
var<workgroup> kA: array<u32, 1000>;
var<workgroup> kB: array<u32, 1000>;
var<workgroup> sKnown: atomic<u32>;
var<workgroup> sKnownU: u32;
@compute @workgroup_size(512) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) l: u32) {
  let b = bricks[B_LSURF + wg.x];
  let o = brickCoord(b) * 8;
  let lc = localCoord(l);
  let c = o + lc;
  let i = (b << 9u) | l;
  let ti = tileIndex(lc);
  var outBits = 0u;
  for (var a = 0u; a < 3u; a++) {
    if (l == 0u) { atomicStore(&sKnown, 0u); }
    workgroupBarrier();
    for (var t = l; t < 1000u; t += 512u) {
      let q = o + tileCoord(t);
      var v = 0.0; var k = 0u;
      if (inBox(q)) {
        let qi = cidx(q);
        let deep = (bricks[B_CLASS + brickOfCell(q)] & CLASS_DEEP) != 0u;
        if (faceClosed(q, a)) { k = 2u; }
        else if (deep) { v = vel[U_OFF + a * N + qi]; k = (bits[KNOWN_OFF + qi] >> a) & 1u; }
        else { v = vel[cfg.a + a * N + qi]; k = (bits[KNOWN_OFF + qi] >> (cfg.c + a)) & 1u; }
      } else { k = 2u; }
      vA[t] = v; vB[t] = v; kA[t] = k; kB[t] = k;
      if (k == 1u) { atomicStore(&sKnown, 1u); }
    }
    workgroupBarrier();
    if (l == 0u) { sKnownU = atomicLoad(&sKnown); }
    // Without a known face in the tile the sweeps change nothing.
    let sweeps = select(0u, 8u, workgroupUniformLoad(&sKnownU) != 0u);
    for (var s = 0u; s < sweeps; s++) {
      if ((s & 1u) == 0u) {
        if (kA[ti] == 0u) {
          var sum = 0.0; var n = 0.0;
          if (kA[ti - 1u] == 1u) { sum += vA[ti - 1u]; n += 1.0; }
          if (kA[ti + 1u] == 1u) { sum += vA[ti + 1u]; n += 1.0; }
          if (kA[ti - 10u] == 1u) { sum += vA[ti - 10u]; n += 1.0; }
          if (kA[ti + 10u] == 1u) { sum += vA[ti + 10u]; n += 1.0; }
          if (kA[ti - 100u] == 1u) { sum += vA[ti - 100u]; n += 1.0; }
          if (kA[ti + 100u] == 1u) { sum += vA[ti + 100u]; n += 1.0; }
          if (n > 0.0) { vB[ti] = sum / n; kB[ti] = 1u; }
        } else { vB[ti] = vA[ti]; kB[ti] = kA[ti]; }
      } else {
        if (kB[ti] == 0u) {
          var sum = 0.0; var n = 0.0;
          if (kB[ti - 1u] == 1u) { sum += vB[ti - 1u]; n += 1.0; }
          if (kB[ti + 1u] == 1u) { sum += vB[ti + 1u]; n += 1.0; }
          if (kB[ti - 10u] == 1u) { sum += vB[ti - 10u]; n += 1.0; }
          if (kB[ti + 10u] == 1u) { sum += vB[ti + 10u]; n += 1.0; }
          if (kB[ti - 100u] == 1u) { sum += vB[ti - 100u]; n += 1.0; }
          if (kB[ti + 100u] == 1u) { sum += vB[ti + 100u]; n += 1.0; }
          if (n > 0.0) { vA[ti] = sum / n; kA[ti] = 1u; }
        } else { vA[ti] = vB[ti]; kA[ti] = kB[ti]; }
      }
      workgroupBarrier();
    }
    vel[cfg.b + a * N + i] = select(vA[ti], 0.0, kA[ti] == 2u);
    outBits |= select(0u, 1u, kA[ti] == 1u) << (cfg.d + a);
    workgroupBarrier();
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
  var x = parts[base + 2u * i].xyz;
  var v = parts[base + 2u * i + 1u].xyz;
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
  x = collide(x + parts[cfg.a * CAP * 2u + 2u * i].xyz);
  parts[base + 2u * i] = vec4f(x, 0.0);
  parts[base + 2u * i + 1u] = vec4f(v, 0.0);
  atomicAdd(&bricks[B_PCOUNT + brickOfCell(vec3i(floor(x)))], 1u);
}
`);

// ---------------------------------------------------------------------------
// Diagnostics (on demand, never per step): per-brick liquid volume from the
// six-tetrahedron fill over open cells; optional texture publication (init).
// cfg.a = 1 to also write the fill and render textures for every brick.
// ---------------------------------------------------------------------------
export const volumeKernel = kernel("seep.volume", [bricksRO, phisRO, { name: "vsum", kind: "rw", type: "array<f32>" }, volW, cfgU, surfW], /* wgsl */ `
${levelSetFillWGSL}
${SURF_WGSL}
fn ph(v: vec3i) -> f32 { return phis[PHI_OFF + cidx(clampV(v))]; }
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
  classifyKernel, clearKernel, countKernel, seedKernel, scanKernel, scatterKernel, separateKernel, advectPhiKernel, combineKernel,
  redistInitKernel, redistSweepKernel, p2gKernel, velocityBulkKernel, velocityKernel, operatorKernel, coarsenBrickKernel, coarsenTopKernel,
  pcgUpdateKernel, mgDown0Kernel, mgDown1Kernel, mgDown2Kernel, mgDown3Kernel, mgRestrict3Kernel, coarseSolveKernel, mgUp3Kernel, mgUp2Kernel, mgUp1Kernel, pcgPrecondKernel,
  pcgDirectionKernel, projectKernel, shadowKernel, extendKernel, g2pKernel, volumeKernel,
];
