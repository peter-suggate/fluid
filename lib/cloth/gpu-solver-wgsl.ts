/** WGSL of the GPU cloth solver: one implicit step per call (no substeps), Newton-PCG (gpu-newton-wgsl.ts) on the
 * step's objective, whose element terms (membrane, bending, damping, augmented-Lagrangian strain limits, pins, grab,
 * contact) are those of the Float64 oracle (oracle.ts) in f32. Every buffer and bind slot is described in gpu-solver.ts. */
import { CLOTH_AVBD } from './oracle';
import { CLOTH_BEND_ROWS } from './topology';
import { clothNewtonWGSL, type ClothPCGLayout } from './gpu-newton-wgsl';

/** vec4<f32> per-vertex segments of the `state` buffer, each CLOTH_SEGMENT_ALIGN-padded N long.
 * Segment 0 is the step-start velocity, so C2 binds `state` as its velocity input. After the
 * segments: one record per edge (AVBD dual lambda, k, stabilized limit), the diagnostics
 * partials (2 per workgroup) and the clock record (step, previous step, h~, BDF2 as floats). Segment 9 is the mouse
 * grab (`ClothGPUSolver.setGrab`): per vertex its anchor (xyz, the vertex at the grab's first step) and weight (w, 0 free). */
export const CLOTH_STATE = { velocity: 0, xt: 1, vPrev: 2, dxPrev: 3, y: 4, xHat: 5, pinTarget: 6, pinDual: 7, normal: 8, grab: 9 } as const;
export const CLOTH_STATE_SEGMENTS = 10;
/** Segment padding in vec4 records: 16 records = 256 B, the storage binding offset alignment. */
export const CLOTH_SEGMENT_ALIGN = 16;
/** Status words: [0] status bits (sticky), [1] steps completed; slots start at CLOTH_STATUS_RING_BASE. */
export const CLOTH_STATUS = { nonFinite: 1 } as const;
export const CLOTH_STATUS_RING_BASE = 16;
/** Status words below the ring: 0 status flags, 1 steps; the contact restart's words (Newton with contact): this step's
 * onset, the next step's restart, and restarts since the state was uploaded. */
export const CLOTH_STATUS_WORD = { onset: 2, restart: 3, restarts: 4 } as const;
export const CLOTH_STATUS_SLOT_WORDS = 16;
export const CLOTH_STATUS_RING_SLOTS = 64;
export const CLOTH_STATUS_WORDS = CLOTH_STATUS_RING_BASE + CLOTH_STATUS_SLOT_WORDS * CLOTH_STATUS_RING_SLOTS;
/** Slot words: step + 1 (0: empty), BDF2 flag, residual |grad E| / |m g|, max edge strain, kinetic, potential, elastic energy (f32 bits). */
export const CLOTH_SLOT = { step: 0, bdf2: 1, residual: 2, maxStrain: 3, kinetic: 4, potential: 5, elastic: 6 } as const;
export const CLOTH_WORKGROUP = 64;
/** Uniform records of the per-dispatch data, one per PCG iteration (its index in range.z). */
export const CLOTH_DISPATCH_STRIDE = 256;
/** Bindings (group 0). */
export const CLOTH_BINDING = { params: 0, dispatch: 1, positions: 2, state: 3, status: 4, topo: 5, rest: 6, contact: 7, topo4: 8, newton: 10, contactState: 12 } as const;
export const CLOTH_SCRIPT_KIND = { translate: 0, rotate: 1, oscillate: 2 } as const;

export interface ClothSolverWGSLOptions {
  /** C2's vertex contact block (ogcVertexBlockWGSL at group 0, CLOTH_BINDING.contact); omitted: no contact terms. */
  readonly contactBlock?: string;
  /** The compiled PCG layout (ClothGPUTables.pcg) the Newton kernels are specialised to. */
  readonly pcg: ClothPCGLayout;
}

export function clothSolverWGSL(options: ClothSolverWGSLOptions): string {
  // With contact the BDF2 history restarts at a contact onset (C2's contact state).
  const contact = !!options.contactBlock, restart = contact;
  // The projected arc's Pi_v is C2's ogcProject (a block without it is refused).
  if (contact && !options.contactBlock!.includes('fn ogcProjectTrial(')) throw new Error('Cloth solver contact block lacks ogcProjectTrial (the projected arc search needs Pi_v at each trial level)');
  return /* wgsl */`
struct Params {
  n: vec4u,      // N, T, E, bending rows (N: row r is vertex r)
  topo0: vec4u,  // vertexFaceOffsets, vertexFaces, bending row offsets, bending row entries (j, w_j bits)
  topo1: vec4u,  // vertexEdgeOffsets, vertexEdges, unused, unused
  topo2: vec4u,  // triangles (v0, v1, v2, material), edges (v0, v1, f0, f1), bending stencil offsets
  rest0: vec4u,  // face Dm^-1, face (area, damping), bending row (k, kd), edge (L, eps)
  rest1: vec4u,  // vertex (m, elastic, length, group), anchor, material, script
  flags: vec4u,  // script count, BDF2 requested, strain limited, partial count
  state0: vec4u, // segment stride, edge duals, partials, clock
  state1: vec4u, // damping tangent: faces (2 records), unused, damped flag
  gravity: vec4f, // xyz, dt
  misc: vec4f,   // clock origin (s), unused, initial guess (0 previous, 1 adaptive, 2 inertial)
  bend: vec4u,   // unused, unused, unused, bending diagonal (rest record)
  grab: vec4f,   // mouse grab: displacement of its anchors (xyz, metres), held (w: 1, else 0)
  newton0: vec4u, // Newton-PCG (gpu-newton-wgsl.ts): face records, partials, partial count W, control record (nw records)
  newton1: vec4u, // bending stencil records (topo4 index), PCG tolerance^2 and residual tolerance^2 |m g|^2 (f32 bits)
  newton2: vec4u, // compiled topology (topo words): patch table, face corner slots, edge slots, stencil destinations
  newton3: vec4u, // PCG partials (nw record), patches, PCG scalars (nw record), unused
  newton4: vec4u, // compiled topology (topo words): vertex of each patch slot, patch slot of each vertex; Eisenstat-Walker eta_max^2 (f32 bits), unused
}
struct Dispatch { range: vec4u, values: vec4f } // (0, N, PCG iteration from 0)

@group(0) @binding(${CLOTH_BINDING.params}) var<uniform> P: Params;
@group(0) @binding(${CLOTH_BINDING.dispatch}) var<uniform> R: Dispatch;
@group(0) @binding(${CLOTH_BINDING.positions}) var<storage, read_write> positions: array<vec4f>;
@group(0) @binding(${CLOTH_BINDING.state}) var<storage, read_write> state: array<vec4f>;
@group(0) @binding(${CLOTH_BINDING.status}) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(${CLOTH_BINDING.topo}) var<storage, read> topo: array<u32>;
@group(0) @binding(${CLOTH_BINDING.rest}) var<storage, read> rest: array<vec4f>;
// The same topo words as vec4 records: triangles, edges and stencil records are 4-word records at 64-word aligned segments.
@group(0) @binding(${CLOTH_BINDING.topo4}) var<storage, read> topo4: array<vec4u>;

const ALPHA: f32 = ${CLOTH_AVBD.alpha};
const EDGE_ALPHA: f32 = ${CLOTH_AVBD.edgeAlpha};
const GAMMA: f32 = ${CLOTH_AVBD.gamma};
const BETA: f32 = ${CLOTH_AVBD.beta};
const K_MAX_RATIO: f32 = ${CLOTH_AVBD.maxStiffnessRatio};
const EPS32: f32 = 1.1920929e-7;
const PI: f32 = 3.14159265358979;
const S_VEL = ${CLOTH_STATE.velocity}u; const S_NORMAL = ${CLOTH_STATE.normal}u; const S_XT = ${CLOTH_STATE.xt}u; const S_VPREV = ${CLOTH_STATE.vPrev}u; const S_DXPREV = ${CLOTH_STATE.dxPrev}u;
const S_Y = ${CLOTH_STATE.y}u; const S_XHAT = ${CLOTH_STATE.xHat}u; const S_PINC = ${CLOTH_STATE.pinTarget}u; const S_PIND = ${CLOTH_STATE.pinDual}u;
const S_GRAB = ${CLOTH_STATE.grab}u;
const RING_BASE = ${CLOTH_STATUS_RING_BASE}u; const SLOT_WORDS = ${CLOTH_STATUS_SLOT_WORDS}u; const RING_SLOTS = ${CLOTH_STATUS_RING_SLOTS}u;
const WG = ${CLOTH_WORKGROUP}u;
const IDENTITY = mat3x3f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);

fn seg(s: u32, v: u32) -> u32 { return s * P.state0.x + v; }
fn pos(i: u32) -> vec3f { return positions[i].xyz; }
fn tris(f: u32) -> vec4u { return topo4[(P.topo2.x >> 2u) + f]; }
fn edges(e: u32) -> vec4u { return topo4[(P.topo2.y >> 2u) + e]; }
fn edgeDual(e: u32) -> vec4f { return state[P.state0.y + e]; }
// The clock record: (step, previous step, h~, BDF2), all as exact floats.
fn clockStep() -> u32 { return u32(state[P.state0.w].x); }
fn previousStep() -> f32 { return state[P.state0.w].y; }
fn hEff() -> f32 { return state[P.state0.w].z; }
fn bdf2Step() -> bool { return state[P.state0.w].w > 0.0; }
// A restart step (clock w = -1): implicit Euler after a contact onset anywhere on the cloth (beginStep).
fn restartStep() -> bool { return state[P.state0.w].w < 0.0; }
fn vertexRest(v: u32) -> vec4f { return rest[P.rest1.x + v]; }
fn vertexGroup(v: u32) -> u32 { return u32(vertexRest(v).w); }
fn referenceStiffness(v: u32) -> f32 { let r = vertexRest(v); let h = hEff(); return r.x / (h * h) + r.y; }
// The mouse grab at vertex v: the spring (k/2)|x - (a + D)|^2 towards its anchor a moved by the grab's displacement D,
// with k = weight x the vertex's reference stiffness (the scale an AVBD pin starts from). Returns (k (x - a - D), k);
// zero for a free vertex or with no grab held (a uniform branch, so an idle grab costs one uniform read).
fn grabSpring(v: u32, xv: vec3f) -> vec4f {
  if (P.grab.w == 0.0) { return vec4f(0.0); }
  let a = state[seg(S_GRAB, v)];
  if (!(a.w > 0.0)) { return vec4f(0.0); }
  let k = a.w * referenceStiffness(v);
  return vec4f(k * (xv - a.xyz - P.grab.xyz), k);
}
fn outer(u: vec3f, w: vec3f) -> mat3x3f { return mat3x3f(u * w.x, u * w.y, u * w.z); }
fn finite3(x: vec3f) -> bool { return all(abs(x) <= vec3f(3.0e38)); }

${contact ? `// Newton: the line search evaluates the contact energy at each trial point x(alpha) of the projected arc
// (nwTrialMove = k + 1 for trial k, 0 everywhere else); H.p multiplies the PCG vector u (staged in the patch, else formed
// from global memory).
var<private> nwTrialMove: u32;
fn ogcX(i: u32) -> vec3f {
  if (nwTrialMove != 0u) { let k = nwTrialMove - 1u; return positions[i].xyz + nwMove(i, k, exp2(-f32(k))); }
  return positions[i].xyz;
}
fn ogcDir(i: u32) -> vec3f { return pcgDir(i); }
` : ''}
${contact ? `fn ogcXt(i: u32) -> vec3f { return state[seg(S_XT, i)].xyz; }
fn ogcTri(t: u32) -> vec3u { return tris(t).xyz; }
fn ogcEdgeVerts(e: u32) -> vec2u { return edges(e).xy; }
${options.contactBlock}` : ''}

// Nearest PSD matrix of a symmetric 3x3: cyclic Jacobi, negative eigenvalues clamped.
// PSD projection of a membrane vertex block H = J D J^T + sigma I (sigma < 0). Every column of J lies in span(f0, f1),
// so that plane is invariant under H and its normal is an eigenvector with eigenvalue sigma < 0, which clamps to zero:
// the exact eigenvalue clamp is a closed-form 2x2 problem in the plane (no Jacobi sweeps, no indexed private arrays).
fn projectMembranePSD(H: mat3x3f, f0: vec3f, f1: vec3f) -> mat3x3f {
  let q0 = dot(f0, f0); let q1 = dot(f1, f1);
  if (!(max(q0, q1) > 0.0)) { return mat3x3f(); }
  let u1 = select(f1 * inverseSqrt(q1), f0 * inverseSqrt(q0), q0 >= q1);
  let n = cross(f0, f1);
  var u2 = cross(n, u1);
  if (!(dot(n, n) > 1.0e-12 * q0 * q1)) {
    // f0 parallel to f1: J D J^T has rank one along u1; any unit u2 normal to u1 completes an invariant plane.
    u2 = cross(u1, select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(u1.x) > 0.5));
  }
  u2 = normalize(u2);
  let m11 = dot(u1, H * u1); let m12 = dot(u1, H * u2); let m22 = dot(u2, H * u2);
  let theta = 0.5 * atan2(2.0 * m12, m11 - m22); let c = cos(theta); let sn = sin(theta);
  let w1 = c * u1 + sn * u2; let w2 = c * u2 - sn * u1;
  let l1 = m11 * c * c + 2.0 * m12 * c * sn + m22 * sn * sn; let l2 = m11 * sn * sn - 2.0 * m12 * c * sn + m22 * c * c;
  return max(l1, 0.0) * outer(w1, w1) + max(l2, 0.0) * outer(w2, w2);
}

// r: for the quadratic damping terms, a bound per unit |d| on the first-order rounding of g . d (|g| plus the base
// rate's rounding through the frozen tangent). Newton's global line search does not read it (the VBD sweeps did).
struct Block { g: vec3f, H: mat3x3f, e: f32, r: f32 }

// Face f seen from its corner k with that corner at xv: every load the face's terms share, made once.
struct FaceIn { f: u32, k: u32, t: vec4u, x0: vec3f, x1: vec3f, x2: vec3f, dm: vec4f, fr: vec4f, c0: vec4f, c1: vec4f }
fn loadFace(f: u32, k: u32, xv: vec3f) -> FaceIn {
  let t = tris(f);
  var x0 = pos(t.x); var x1 = pos(t.y); var x2 = pos(t.z);
  if (k == 0u) { x0 = xv; } else if (k == 1u) { x1 = xv; } else { x2 = xv; }
  return FaceIn(f, k, t, x0, x1, x2, rest[P.rest0.x + f], rest[P.rest0.y + f], rest[P.rest1.z + 2u * t.w], rest[P.rest1.z + 2u * t.w + 1u]);
}
struct Membrane { f0: vec3f, f1: vec3f, j1: vec3f, j2: vec3f, j3: vec3f, s11: f32, s22: f32, tau: f32, d11: f32, d22: f32, sigma: f32, e: f32, area: f32, c12: f32, g: f32 }
fn membraneAt(f: u32, k: u32, xv: vec3f) -> Membrane { return membraneOf(loadFace(f, k, xv)); }
fn membraneOf(q: FaceIn) -> Membrane {
  let k = q.k; let dm = q.dm; let area = q.fr.x; let c0 = q.c0; let c1 = q.c1;
  let e1 = q.x1 - q.x0; let e2 = q.x2 - q.x0;
  let f0 = e1 * dm.x + e2 * dm.y; let f1 = e1 * dm.z + e2 * dm.w;
  let e11 = 0.5 * (dot(f0, f0) - 1.0); let e22 = 0.5 * (dot(f1, f1) - 1.0); let gam = dot(f0, f1);
  let w11 = select(0.0, (1.0 - c1.z) * c1.x, e11 < 0.0); let w22 = select(0.0, (1.0 - c1.z) * c1.y, e22 < 0.0);
  let s11 = c0.x * e11 + c0.z * e22 - w11 * e11; let s22 = c0.y * e22 + c0.z * e11 - w22 * e22; let tau = c0.w * gam;
  var a = dm.y; var b = dm.w;
  if (k == 0u) { a = -dm.x - dm.y; b = -dm.z - dm.w; } else if (k == 1u) { a = dm.x; b = dm.z; }
  let energy = area * (0.5 * (c0.x * e11 * e11 + 2.0 * c0.z * e11 * e22 + c0.y * e22 * e22) - 0.5 * (w11 * e11 * e11 + w22 * e22 * e22) + 0.5 * c0.w * gam * gam);
  return Membrane(f0, f1, a * f0, b * f1, a * f1 + b * f0, s11, s22, tau, c0.x - w11, c0.y - w22, s11 * a * a + s22 * b * b + 2.0 * tau * a * b, energy, area, c0.z, c0.w);
}
fn membraneBlock(q: FaceIn) -> Block {
  let m = membraneOf(q);
  let g = m.area * (m.s11 * m.j1 + m.s22 * m.j2 + m.tau * m.j3);
  var H = m.area * (m.d11 * outer(m.j1, m.j1) + m.d22 * outer(m.j2, m.j2) + m.c12 * (outer(m.j1, m.j2) + outer(m.j2, m.j1)) + m.g * outer(m.j3, m.j3) + m.sigma * IDENTITY);
  // J D J^T is PSD, so only a compressive sigma makes the block indefinite.
  if (m.sigma < 0.0) { H = projectMembranePSD(H, m.f0, m.f1); }
  return Block(g, H, m.e, 0.0);
}

// Kirchhoff plate bending (topology clothBendingRows): ${CLOTH_BEND_ROWS} rows per vertex, row r of vertex i = r / ${CLOTH_BEND_ROWS} is
// R_r x = sum_j w_j (x_j - x_i) over its other entries (j, w_j bits) at topo words P.topo0.w + 2o, o in [topo[P.topo0.z + r],
// topo[P.topo0.z + r + 1]). With (k, kd) at rest[P.rest0.z + r], its energy is (k/2)|R_r x|^2 and its damping (kd k / 2h~)|R_r (x - x^)|^2. Gradients and
// Hessians go through the bending stencil (Q = sum_r k w w^T); the rows give the energies and line-search changes.
fn bendKD(r: u32) -> vec2f { return rest[P.rest0.z + r].xy; }
fn bendJ(o: u32) -> u32 { return topo[P.topo0.w + 2u * o]; }
fn bendW(o: u32) -> f32 { return bitcast<f32>(topo[P.topo0.w + 2u * o + 1u]); }
// Row r at the current positions: L x, its damping counterpart L (x - x^), and their rounding scales sum_j |w_j| |.|.
struct BendIn { kd: vec2f, s: vec3f, sr: f32, u: vec3f, ur: f32 }
const BEND_ROWS = ${CLOTH_BEND_ROWS}u;
fn loadBend(r: u32) -> BendIn {
  let i = r / BEND_ROWS; let kd = bendKD(r); let xr = pos(i); var q = BendIn(kd, vec3f(0.0), 0.0, vec3f(0.0), 0.0);
  let damped = kd.y > 0.0; var xhr = vec3f(0.0);
  if (damped) { xhr = state[seg(S_XHAT, i)].xyz; }
  for (var o = topo[P.topo0.z + r]; o < topo[P.topo0.z + r + 1u]; o++) {
    let j = bendJ(o); let w = bendW(o); let dx = pos(j) - xr;
    q.s += w * dx; q.sr += abs(w) * length(dx);
    if (damped) { let du = dx - (state[seg(S_XHAT, j)].xyz - xhr); q.u += w * du; q.ur += abs(w) * length(du); }
  }
  return q;
}
fn bendEnergy(q: BendIn) -> f32 { return 0.5 * q.kd.x * dot(q.s, q.s); }
// (dE, floor) of the row and its damping when L x changes by ds, exactly (both are quadratic in it).
fn bendDelta(q: BendIn, ds: vec3f) -> vec2f {
  let scale = q.kd.x * q.kd.y / hEff();
  let lin = q.kd.x * dot(q.s, ds) + scale * dot(q.u, ds); let quad = 0.5 * (q.kd.x + scale) * dot(ds, ds);
  return vec2f(lin + quad, abs(lin) + quad + (q.kd.x * q.sr + scale * q.ur) * length(ds));
}

// The bending stencil of vertex v (bending and its damping summed over the rows holding v): g = sum_j Q_vj (x_j -
// x_v) + sum_j Qd_vj (u_j - u_v) / h~ over v's stencil neighbours j (u = x - x^), Hessian (Q_vv + Qd_vv / h~) I. A
// stencil record is (j, Q_vj, Qd_vj, 0); this adds one into dmp's gradient and rounding scale (xhv is x^_v).
fn stencilRecord(rec: vec4u, xv: vec3f, xhv: vec3f, dmp: ptr<function, Block>) {
  let dx = pos(rec.x) - xv; let q = bitcast<f32>(rec.y);
  (*dmp).g += q * dx; (*dmp).r += abs(q) * length(dx);
  if (P.state1.z != 0u) {
    let qd = bitcast<f32>(rec.z) / hEff(); let du = dx - (state[seg(S_XHAT, rec.x)].xyz - xhv);
    (*dmp).g += qd * du; (*dmp).r += abs(qd) * length(du);
  }
}
fn stencilDiagonal(v: u32) -> mat3x3f { let q = rest[P.bend.w + v].xy; return (q.x + q.y / hEff()) * IDENTITY; }
// Vertex v's whole bending stencil with v at xv (vertex-ordered records at topo4 P.newton1.x): gradient, Hessian, rounding.
fn stencilBlock(v: u32, xv: vec3f) -> Block {
  var b = Block(vec3f(0.0), stencilDiagonal(v), 0.0, 0.0); let xhv = state[seg(S_XHAT, v)].xyz;
  for (var o = topo[P.topo2.z + v]; o < topo[P.topo2.z + v + 1u]; o++) { stencilRecord(topo4[P.newton1.x + o], xv, xhv, &b); }
  return b;
}

// Rayleigh damping as a dissipation potential with the element tangent frozen at x_t (oracle: ClothDampingTangent).
// The frozen tangent of a face at x_t: (f0, softened c11), (f1, softened c22).
fn faceTangent(x0: vec3f, x1: vec3f, x2: vec3f, dm: vec4f, c0: vec4f, c1: vec4f) -> array<vec4f, 2> {
  let e1 = x1 - x0; let e2 = x2 - x0;
  let f0 = e1 * dm.x + e2 * dm.y; let f1 = e1 * dm.z + e2 * dm.w;
  let e11 = 0.5 * (dot(f0, f0) - 1.0); let e22 = 0.5 * (dot(f1, f1) - 1.0);
  return array<vec4f, 2>(vec4f(f0, c0.x - select(0.0, (1.0 - c1.z) * c1.x, e11 < 0.0)), vec4f(f1, c0.y - select(0.0, (1.0 - c1.z) * c1.y, e22 < 0.0)));
}
fn faceDamping(q: FaceIn) -> Block {
  if (!(q.fr.y > 0.0)) { return Block(vec3f(0.0), mat3x3f(), 0.0, 0.0); }
  return faceDampingWith(q, state[P.state1.x + 2u * q.f], state[P.state1.x + 2u * q.f + 1u]);
}
fn faceDampingWith(q: FaceIn, T0: vec4f, T1: vec4f) -> Block {
  let fr = q.fr;
  let k = q.k; let t = q.t; let dm = q.dm; let c0 = q.c0;
  let f0 = T0.xyz; let f1 = T1.xyz;
  let a = vec3f(-dm.x - dm.y, dm.x, dm.y); let b = vec3f(-dm.z - dm.w, dm.z, dm.w);
  let u0 = q.x0 - state[seg(S_XHAT, t.x)].xyz;
  let u1 = q.x1 - state[seg(S_XHAT, t.y)].xyz;
  let u2 = q.x2 - state[seg(S_XHAT, t.z)].xyz;
  let df0 = a.x * u0 + a.y * u1 + a.z * u2; let df1 = b.x * u0 + b.y * u1 + b.z * u2;
  let rate = vec3f(dot(f0, df0), dot(f1, df1), dot(f0, df1) + dot(f1, df0));
  let sigma = vec3f(T0.w * rate.x + c0.z * rate.y, c0.z * rate.x + T1.w * rate.y, c0.w * rate.z);
  let ak = select(select(a.x, a.y, k == 1u), a.z, k == 2u); let bk = select(select(b.x, b.y, k == 1u), b.z, k == 2u);
  let j1 = ak * f0; let j2 = bk * f1; let j3 = ak * f1 + bk * f0;
  let scale = fr.y * fr.x / hEff();
  let g = scale * (sigma.x * j1 + sigma.y * j2 + sigma.z * j3);
  // faceDampingDelta's rounding term scale (|dsigma| . roundRate) with |dr_i| <= J_i |d|.
  let ru = vec2f(abs(a.x) * length(u0) + abs(a.y) * length(u1) + abs(a.z) * length(u2), abs(b.x) * length(u0) + abs(b.y) * length(u1) + abs(b.z) * length(u2));
  let lf0 = length(f0); let lf1 = length(f1);
  let roundRate = vec3f(lf0 * ru.x, lf1 * ru.y, lf0 * ru.y + lf1 * ru.x);
  let J = vec3f(abs(ak) * lf0, abs(bk) * lf1, abs(bk) * lf0 + abs(ak) * lf1);
  let dsig = vec3f(abs(T0.w) * J.x + abs(c0.z) * J.y, abs(c0.z) * J.x + abs(T1.w) * J.y, abs(c0.w) * J.z);
  return Block(g, scale * (T0.w * outer(j1, j1) + T1.w * outer(j2, j2) + c0.z * (outer(j1, j2) + outer(j2, j1)) + c0.w * outer(j3, j3)),
    0.5 * scale * dot(rate, sigma), length(g) + scale * dot(dsig, roundRate));
}
fn augmented(k: f32, lambda: f32, C: f32, inequality: bool) -> f32 {
  var f = k * C + lambda;
  if (inequality) { f = max(f, 0.0); }
  return (f * f - lambda * lambda) / (2.0 * k);
}

// A vertex's gradient and Hessian. No energy is summed here: the line search takes its energy changes in difference
// form (vertexDelta, the star trials).
struct System { g: vec3f, H: mat3x3f }

// Vertex v's local objective at xv (other vertices at their current positions):
// inertia, membrane, bending, damping, pins, strain limits${contact ? ', contact' : ''}.
fn vertexSystem(v: u32, xv: vec3f) -> System {
  let h = hEff(); let vr = vertexRest(v); let w = vr.x / (h * h);
  let r = xv - state[seg(S_Y, v)].xyz;
  var s = System(w * r, w * IDENTITY);
  for (var j = topo[P.topo0.x + v]; j < topo[P.topo0.x + v + 1u]; j++) {
    let entry = topo[P.topo0.y + j]; let q = loadFace(entry >> 2u, entry & 3u, xv);
    let b = membraneBlock(q); let d = faceDamping(q);
    s.g += b.g + d.g; s.H += b.H + d.H;
  }
  { let b = stencilBlock(v, xv); s.g += b.g; s.H += b.H; }
  return constraints(v, xv, s);
}

// Line-search energy changes in difference form (the Newton trials, gpu-newton-wgsl.ts). E(x + d) - E(x) is assembled
// from exact increments (strain and residual increments are linear or quadratic in d), so the Armijo test does not
// compare two nearly equal f32 energies. Each returns (dE, floor): floor sums |increments| and the first-order rounding
// of the base quantities, and the test allows 4 eps * floor.
fn quadDelta(aOld: f32, aNew: f32, e: f32, de: f32) -> f32 {
  if (aOld == aNew) { return aOld * (e * de + 0.5 * de * de); }
  let en = e + de; return 0.5 * (aNew * en * en - aOld * e * e);
}
fn edgeDelta(v: u32, xv: vec3f, e: u32, d: vec3f) -> vec2f {
  let er = rest[P.rest0.w + e];
  if (!(er.y > 0.0)) { return vec2f(0.0); }
  let ev = edges(e); let other = select(ev.x, ev.y, ev.x == v);
  let dual = edgeDual(e);
  let d0 = xv - pos(other); let l0 = length(d0); let l1 = length(d0 + d);
  if (!(l0 > 0.0 && l1 > 0.0)) {
    let e0 = select(0.0, augmented(dual.y, dual.x, l0 - dual.z, true), l0 > 0.0); let e1 = select(0.0, augmented(dual.y, dual.x, l1 - dual.z, true), l1 > 0.0);
    return vec2f(e1 - e0, abs(e0) + abs(e1));
  }
  let C = l0 - dual.z; let dC = (2.0 * dot(d0, d) + dot(d, d)) / (l0 + l1);
  let f0 = max(dual.y * C + dual.x, 0.0); let f1 = max(dual.y * (C + dC) + dual.x, 0.0);
  var de = 0.5 * dC * (f0 + f1);
  if (!(f0 > 0.0 && f1 > 0.0)) { de = (f1 - f0) * (f1 + f0) / (2.0 * dual.y); }
  return vec2f(de, abs(de) + dual.y * l0 * abs(dC));
}
// Strain limit of edge e at vertex v (AVBD hard inequality).
fn edgeTerm(v: u32, xv: vec3f, e: u32, s0: System) -> System {
  var s = s0;
  let er = rest[P.rest0.w + e];
  if (!(er.y > 0.0)) { return s; }
  let ev = edges(e); let other = select(ev.x, ev.y, ev.x == v);
  let d = xv - pos(other); let len = length(d);
  if (!(len > 0.0)) { return s; }
  let dual = edgeDual(e); let C = len - dual.z; let n = d / len;
  let f = max(dual.y * C + dual.x, 0.0);
  var kEff = dual.y;
  // The exact Hessian of the augmented term, zero while its clamped force is zero.
  if (!(dual.y * C + dual.x > 0.0)) { kEff = 0.0; }
  s.g += f * n; s.H += (kEff - f / len) * outer(n, n) + (f / len) * IDENTITY;
  return s;
}

// Pin terms of vertex v.
fn pinTerms(v: u32, xv: vec3f, s0: System) -> System {
  var s = s0;
  if (vertexGroup(v) > 0u) {
    let d = state[seg(S_PIND, v)]; let C = xv - state[seg(S_PINC, v)].xyz;
    s.g += d.w * C + d.xyz; s.H += d.w * IDENTITY;
  }
  let gs = grabSpring(v, xv);
  s.g += gs.xyz; s.H += gs.w * IDENTITY;
  return s;
}
${contact ? `// Contact entries first, first + stride, ... of vertex v added to s, in one pass that also returns their energy at xv
// (stride 1 from 0 takes them all).
fn contactTerms(v: u32, xv: vec3f, first: u32, stride: u32, s: ptr<function, System>) -> f32 {
  var fc = vec3f(0.0); var Hc = mat3x3f();
  let e = ogcVertexContactLanes(v, xv, first, stride, &fc, &Hc);
  (*s).g -= fc; (*s).H += Hc;
  return e;
}
` : ''}
fn constraints(v: u32, xv: vec3f, s0: System) -> System {
  var s = s0;
  if (P.flags.z != 0u) {
    for (var j = topo[P.topo1.x + v]; j < topo[P.topo1.x + v + 1u]; j++) { s = edgeTerm(v, xv, topo[P.topo1.y + j] >> 2u, s); }
  }
  ${contact ? 's = pinTerms(v, xv, s); _ = contactTerms(v, xv, 0u, 1u, &s); return s;' : 'return pinTerms(v, xv, s);'}
}


fn attachment(group: u32, anchor: vec3f, t: f32) -> vec3f {
  var p = anchor;
  for (var i = 0u; i < P.flags.x; i++) {
    let base = P.rest1.w + 4u * i;
    let r0 = rest[base]; let r1 = rest[base + 1u]; let r2 = rest[base + 2u]; let r3 = rest[base + 3u];
    if (u32(r0.y) != group) { continue; }
    let kind = u32(r0.x);
    let linear = clamp((t - r0.z) / (r0.w - r0.z), 0.0, 1.0);
    let w = select(linear, 0.5 - 0.5 * cos(PI * linear), r3.x > 0.5);
    if (kind == ${CLOTH_SCRIPT_KIND.translate}u) { p += w * r1.xyz; }
    else if (kind == ${CLOTH_SCRIPT_KIND.oscillate}u) { p += sin(2.0 * PI * (clamp(t, r0.z, r0.w) - r0.z) / r2.w) * r1.xyz; }
    else {
      let u = normalize(r1.xyz); let angle = w * r1.w; let c = cos(angle); let sn = sin(angle); let r = p - r2.xyz;
      p = r2.xyz + r * c + cross(u, r) * sn + u * dot(u, r) * (1.0 - c);
    }
  }
  return p;
}

fn stepTime() -> f32 { return P.misc.x + f32(clockStep()) * P.gravity.w; }

// A new mouse grab (ClothGPUSolver.setGrab), before the step that first holds it: each grabbed vertex's anchor is where
// the vertex is now, so the grab starts at zero displacement from the live pose, not from the pose the press picked.
@compute @workgroup_size(${CLOTH_WORKGROUP}) fn grabBegin(@builtin(global_invocation_id) id: vec3u) {
  let v = id.x;
  if (v >= P.n.x) { return; }
  let a = state[seg(S_GRAB, v)];
  if (a.w > 0.0) { state[seg(S_GRAB, v)] = vec4f(positions[v].xyz, a.w); }
}

// Step begin (one thread): BDF2 needs a previous step of the same length${restart ? `, and no contact onset in it` : ''}.
@compute @workgroup_size(1) fn beginStep() {
  let dt = P.gravity.w; let previous = previousStep();${restart ? `
  let restart = atomicLoad(&status[${CLOTH_STATUS_WORD.restart}u]) != 0u;
  atomicStore(&status[${CLOTH_STATUS_WORD.restart}u], 0u); atomicStore(&status[${CLOTH_STATUS_WORD.onset}u], 0u);
  if (restart) { atomicAdd(&status[${CLOTH_STATUS_WORD.restarts}u], 1u); }` : `
  let restart = false;`}
  let bdf2 = !restart && P.flags.y != 0u && previous > 0.0 && abs(previous - dt) <= 1e-6 * dt;
  state[P.state0.w].z = select(dt, 2.0 * dt / 3.0, bdf2);
  state[P.state0.w].w = select(select(0.0, 1.0, bdf2), -1.0, restart);
}
${restart ? `
// The BDF2 restart at a contact onset (progress-lane-c1 RESULT (25)). C2's contact state (bit 1: the vertex's contact
// switched on in the step that wrote it, at stepEnd) raises one cloth-wide word: restartScan, before beginStep, reads the
// last step's onset (this step restarts: implicit Euler, x^ = x, v^ = v, h~ = h); onsetScan, after this step's stepEnd,
// reads this step's (finishStep's one-step velocity). The whole cloth restarts: the membrane carries an impact's velocity
// jump to vertices no contact touches, so every vertex's BDF2 history straddles it.
@group(0) @binding(${CLOTH_BINDING.contactState}) var<storage, read> ogcState: array<u32>;
@compute @workgroup_size(${CLOTH_WORKGROUP}) fn restartScan(@builtin(global_invocation_id) id: vec3u) {
  if (id.x < P.n.x && (ogcState[id.x] & 2u) != 0u) { atomicOr(&status[${CLOTH_STATUS_WORD.restart}u], 1u); }
}
@compute @workgroup_size(${CLOTH_WORKGROUP}) fn onsetScan(@builtin(global_invocation_id) id: vec3u) {
  if (id.x < P.n.x && (ogcState[id.x] & 2u) != 0u) { atomicOr(&status[${CLOTH_STATUS_WORD.onset}u], 1u); }
}` : ''}

// At x_t, before the frame pass moves positions: AVBD warm start of the edge duals and
// their stabilized limits, and the damping tangent of every face.
@compute @workgroup_size(${CLOTH_WORKGROUP}) fn frameElements(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (P.flags.z != 0u && i < P.n.z) {
    let er = rest[P.rest0.w + i];
    if (er.y > 0.0) {
      let ev = edges(i);
      let s = max(referenceStiffness(ev.x), referenceStiffness(ev.y));
      let limit = (1.0 + er.y) * er.x;
      var d = edgeDual(i);
      d.x *= ALPHA * GAMMA; d.y = max(GAMMA * d.y, s);
      d.z = limit + EDGE_ALPHA * max(length(pos(ev.x) - pos(ev.y)) - limit, 0.0);
      state[P.state0.y + i] = d;
    }
  }
  if (P.state1.z == 0u) { return; }
  if (i < P.n.y) {
    let t = tris(i);
    let T = faceTangent(pos(t.x), pos(t.y), pos(t.z), rest[P.rest0.x + i], rest[P.rest1.z + 2u * t.w], rest[P.rest1.z + 2u * t.w + 1u]);
    state[P.state1.x + 2u * i] = T[0]; state[P.state1.x + 2u * i + 1u] = T[1];
  }
}

fn adaptiveGravity(a: vec3f, g: vec3f) -> f32 {
  let gl = length(g);
  if (!(gl > 0.0)) { return 0.0; }
  let along = dot(a, g) / gl;
  return select(select(along / gl, 0.0, along < 0.0), 1.0, along > gl);
}

// Frame pass: inertia target, x^, adaptive initialization, pin targets and pin warm start.
@compute @workgroup_size(${CLOTH_WORKGROUP}) fn frame(@builtin(global_invocation_id) id: vec3u) {
  let v = id.x;
  if (v >= P.n.x) { return; }
  let x = pos(v); let vel = state[seg(S_VEL, v)].xyz; let vp = state[seg(S_VPREV, v)].xyz;
  let g = P.gravity.xyz; let h = hEff(); let previous = previousStep(); let bdf2 = bdf2Step();
  var coef = 1.0;
  if (previous > 0.0) { coef = adaptiveGravity((vel - vp) / previous, g); }
  var xh = x; var vh = vel;
  if (bdf2) { xh = x + state[seg(S_DXPREV, v)].xyz / 3.0; vh = (4.0 * vel - vp) / 3.0; }
  state[seg(S_XT, v)] = vec4f(x, 0.0);
  state[seg(S_XHAT, v)] = vec4f(xh, 0.0);
  state[seg(S_Y, v)] = vec4f(xh + h * vh + h * h * g, 0.0);
  var init = xh;
  if (P.misc.z == 1.0) { init = xh + h * vh + h * h * coef * g; } else if (P.misc.z == 2.0) { init = xh + h * vh + h * h * g; }
  let group = vertexGroup(v);
  if (group > 0u) {
    let anchor = rest[P.rest1.y + v].xyz; let t0 = stepTime();
    let p0 = attachment(group, anchor, t0); let p1 = attachment(group, anchor, t0 + P.gravity.w);
    init = x + p1 - p0;
    // A pinned vertex's inertial target is its prescribed motion (contact's "target" prediction reads y).
    state[seg(S_Y, v)] = vec4f(init, 0.0);
    state[seg(S_PINC, v)] = vec4f(p1 + ALPHA * (x - p0), 0.0);
    var d = state[seg(S_PIND, v)];
    d = vec4f(d.xyz * (ALPHA * GAMMA), max(GAMMA * d.w, referenceStiffness(v)));
    state[seg(S_PIND, v)] = d;
  }
  positions[v] = vec4f(init, positions[v].w);
}

// AVBD dual update of edge e at the current positions: lambda <- clamp(k C + lambda, 0, kMax l), k grows while active.
fn updateEdgeDual(e: u32) {
  if (e >= P.n.z) { return; }
  let er = rest[P.rest0.w + e];
  if (!(er.y > 0.0)) { return; }
  let ev = edges(e);
  let s = max(referenceStiffness(ev.x), referenceStiffness(ev.y)); let kMax = K_MAX_RATIO * s;
  var d = edgeDual(e);
  let C = length(pos(ev.x) - pos(ev.y)) - d.z;
  let lambda = d.y * C + d.x;
  d.x = min(max(lambda, 0.0), kMax * er.x);
  if (lambda > 0.0) { d.y = min(d.y + BETA * s * abs(C) / er.x, kMax); }
  state[P.state0.y + e] = d;
}

// Optional diagnostics, before finishStep: the step's force residual, strain and energies.
var<workgroup> partial: array<vec4f, ${2 * CLOTH_WORKGROUP}>;
@compute @workgroup_size(${CLOTH_WORKGROUP}) fn diagnose(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) wg: vec3u) {
  let v = id.x;
  var a = vec4f(0.0); var b = vec4f(0.0, 0.0, -3.0e38, 0.0);
  if (v < P.n.x) {
    let x = pos(v); let m = vertexRest(v).x; let g = P.gravity.xyz;
    let s = vertexSystem(v, x);
    let vel = (x - state[seg(S_XHAT, v)].xyz) / hEff();
    // The residual over free DOFs, as Newton's own test: a pinned vertex is a Dirichlet DOF whose g is the pin's reaction.
    a = vec4f(select(dot(s.g, s.g), 0.0, vertexGroup(v) > 0u), m * m * dot(g, g), 0.5 * m * dot(vel, vel), -m * dot(g, x));
    var elastic = 0.0;
    for (var j = topo[P.topo0.x + v]; j < topo[P.topo0.x + v + 1u]; j++) {
      let entry = topo[P.topo0.y + j];
      if ((entry & 3u) == 0u) { elastic += membraneAt(entry >> 2u, 0u, x).e; }
    }
    for (var r = BEND_ROWS * v; r < BEND_ROWS * (v + 1u); r++) { elastic += bendEnergy(loadBend(r)); }
    var strain = -3.0e38;
    for (var j = topo[P.topo1.x + v]; j < topo[P.topo1.x + v + 1u]; j++) {
      let entry = topo[P.topo1.y + j];
      if ((entry & 3u) != 0u) { continue; }
      let e = entry >> 2u; let ev = edges(e);
      strain = max(strain, length(pos(ev.x) - pos(ev.y)) / rest[P.rest0.w + e].x - 1.0);
    }
    b = vec4f(elastic, 0.0, strain, 0.0);
  }
  partial[lane] = a; partial[WG + lane] = b;
  workgroupBarrier();
  for (var stride = WG / 2u; stride > 0u; stride >>= 1u) {
    if (lane < stride) {
      partial[lane] += partial[lane + stride];
      let o = partial[WG + lane + stride]; var q = partial[WG + lane];
      q.x += o.x; q.z = max(q.z, o.z); partial[WG + lane] = q;
    }
    workgroupBarrier();
  }
  if (lane == 0u) { state[P.state0.z + 2u * wg.x] = partial[0]; state[P.state0.z + 2u * wg.x + 1u] = partial[WG]; }
}

@compute @workgroup_size(${CLOTH_WORKGROUP}) fn reduceDiagnostics(@builtin(local_invocation_index) lane: u32) {
  var a = vec4f(0.0); var b = vec4f(0.0, 0.0, -3.0e38, 0.0);
  for (var i = lane; i < P.flags.w; i += WG) { a += state[P.state0.z + 2u * i]; let o = state[P.state0.z + 2u * i + 1u]; b.x += o.x; b.z = max(b.z, o.z); }
  partial[lane] = a; partial[WG + lane] = b;
  workgroupBarrier();
  for (var stride = WG / 2u; stride > 0u; stride >>= 1u) {
    if (lane < stride) {
      partial[lane] += partial[lane + stride];
      let o = partial[WG + lane + stride]; var q = partial[WG + lane];
      q.x += o.x; q.z = max(q.z, o.z); partial[WG + lane] = q;
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    let step = clockStep(); let base = RING_BASE + (step % RING_SLOTS) * SLOT_WORDS;
    let s = partial[0]; let q = partial[WG];
    atomicStore(&status[base + ${CLOTH_SLOT.bdf2}u], select(0u, 1u, bdf2Step()));
    atomicStore(&status[base + ${CLOTH_SLOT.residual}u], bitcast<u32>(sqrt(s.x / max(s.y, 1e-30))));
    atomicStore(&status[base + ${CLOTH_SLOT.maxStrain}u], bitcast<u32>(q.z));
    atomicStore(&status[base + ${CLOTH_SLOT.kinetic}u], bitcast<u32>(s.z));
    atomicStore(&status[base + ${CLOTH_SLOT.potential}u], bitcast<u32>(s.w));
    atomicStore(&status[base + ${CLOTH_SLOT.elastic}u], bitcast<u32>(q.x));
    atomicStore(&status[base + ${CLOTH_SLOT.step}u], step + 1u);
  }
}

// Step end: velocities, BDF2 history, render normals, the finite check and the clock.
@compute @workgroup_size(${CLOTH_WORKGROUP}) fn finishStep(@builtin(global_invocation_id) id: vec3u) {
  let v = id.x;
  if (v == 0u) {
    let step = clockStep() + 1u;
    state[P.state0.w].x = f32(step); state[P.state0.w].y = P.gravity.w; atomicStore(&status[1], step);
  }
  if (v >= P.n.x) { return; }
  let x = pos(v); let xt = state[seg(S_XT, v)].xyz;
  var vNew = (x - state[seg(S_XHAT, v)].xyz) / hEff();
  var vPrev = state[seg(S_VEL, v)];${restart ? `
  // The BDF2 velocity (3x_{n+1} - 4x_n + x_{n-1}) / 2h straddles an impact's velocity jump and replays it as a rebound.
  // In the step of an onset the cloth takes the one-step velocity (x_{n+1} - x_n) / h, the next step is implicit Euler
  // (restartStep), and after both the history is the steady one (v_{n-1} = v_n): no BDF2 step reaches back across it.
  let onset = atomicLoad(&status[${CLOTH_STATUS_WORD.onset}u]) != 0u;
  if (onset && bdf2Step()) { vNew = (x - xt) / P.gravity.w; }
  if (onset || restartStep()) { vPrev = vec4f(vNew, 0.0); }` : ''}
  state[seg(S_VPREV, v)] = vPrev;
  state[seg(S_VEL, v)] = vec4f(vNew, 0.0);
  state[seg(S_DXPREV, v)] = vec4f(x - xt, 0.0);
  var n = vec3f(0.0);
  for (var j = topo[P.topo0.x + v]; j < topo[P.topo0.x + v + 1u]; j++) {
    let t = tris(topo[P.topo0.y + j] >> 2u);
    let a = pos(t.x); n += cross(pos(t.y) - a, pos(t.z) - a);
  }
  let ln = length(n);
  state[seg(S_NORMAL, v)] = vec4f(select(vec3f(0.0, 1.0, 0.0), n / ln, ln > 0.0), 0.0);
  if (!finite3(x) || !finite3(vNew)) { atomicOr(&status[0], ${CLOTH_STATUS.nonFinite}u); }
}
${clothNewtonWGSL(CLOTH_BINDING.newton, CLOTH_WORKGROUP, contact, options.pcg)}`;
}
