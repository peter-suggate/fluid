/** WGSL of the Tier A cloth solver: one implicit step per call (no substeps),
 * K VBD iterations over vertex colours with the per-vertex line search,
 * Chebyshev, and AVBD pins and strain limits. The energies, constraint
 * updates and iteration order are those of the Float64 oracle (oracle.ts) in
 * f32. Every buffer and bind slot is described in gpu-solver.ts. */
import { CLOTH_AVBD, CLOTH_LINE_SEARCH } from './oracle';
import { CLOTH_PATCH_SLOT_BITS } from './patches';

/** vec4<f32> per-vertex segments of the `state` buffer, each CLOTH_SEGMENT_ALIGN-padded N long.
 * Segment 0 is the step-start velocity, so C2 binds `state` as its velocity input. After the
 * segments: one record per edge (AVBD dual lambda, k, stabilized limit), the diagnostics
 * partials (2 per workgroup) and the clock record (step, previous step, h~, BDF2 as floats). */
export const CLOTH_STATE = { velocity: 0, xt: 1, vPrev: 2, dxPrev: 3, y: 4, xHat: 5, prev1: 6, prev2: 7, pinTarget: 8, pinDual: 9, normal: 10, patchOut: 11 } as const;
export const CLOTH_STATE_SEGMENTS = 12;
/** Segment padding in vec4 records: 16 records = 256 B, the storage binding offset alignment. */
export const CLOTH_SEGMENT_ALIGN = 16;
/** Status words: [0] status bits (sticky), [1] steps completed; slots start at CLOTH_STATUS_RING_BASE. */
export const CLOTH_STATUS = { nonFinite: 1 } as const;
export const CLOTH_STATUS_RING_BASE = 16;
export const CLOTH_STATUS_SLOT_WORDS = 16;
export const CLOTH_STATUS_RING_SLOTS = 64;
export const CLOTH_STATUS_WORDS = CLOTH_STATUS_RING_BASE + CLOTH_STATUS_SLOT_WORDS * CLOTH_STATUS_RING_SLOTS;
/** Slot words: step + 1 (0: empty), BDF2 flag, residual |grad E| / |m g|, max edge strain, kinetic, potential, elastic energy (f32 bits). */
export const CLOTH_SLOT = { step: 0, bdf2: 1, residual: 2, maxStrain: 3, kinetic: 4, potential: 5, elastic: 6 } as const;
export const CLOTH_WORKGROUP = 64;
/** Default lanes per vertex of the grouped Tier A colour solve (solveColourWide). */
export const CLOTH_VERTEX_LANES = 2;
/** solveColourWide workgroup: CLOTH_WIDE_WORKGROUP / lanes vertices, one SIMD group. */
export const CLOTH_WIDE_WORKGROUP = 32;
/** Uniform records of the per-dispatch data, one per (iteration, colour) and one post record per iteration. */
export const CLOTH_DISPATCH_STRIDE = 256;
/** Bindings (group 0). */
export const CLOTH_BINDING = { params: 0, dispatch: 1, positions: 2, state: 3, status: 4, topo: 5, rest: 6, contact: 7, topo4: 8 } as const;
export const CLOTH_SCRIPT_KIND = { translate: 0, rotate: 1, oscillate: 2 } as const;

export interface ClothSolverWGSLOptions {
  /** C2's vertex contact block (ogcVertexBlockWGSL at group 0, CLOTH_BINDING.contact); omitted: no contact terms. */
  readonly contactBlock?: string;
  /** Tier B module: positions resolve through the patch's workgroup copy (the solvePatches entry point). */
  readonly patch?: { readonly maxVertices: number; readonly workgroup: number };
  /** Lanes per vertex of solveColourWide: a power of two from 1 to CLOTH_WIDE_WORKGROUP (default CLOTH_VERTEX_LANES). */
  readonly vertexLanes?: number;
}

export function clothSolverWGSL(options: ClothSolverWGSLOptions = {}): string {
  const contact = !!options.contactBlock, patch = options.patch, lanes = options.vertexLanes ?? CLOTH_VERTEX_LANES;
  if (!(Number.isInteger(lanes) && lanes >= 1 && lanes <= CLOTH_WIDE_WORKGROUP && (lanes & (lanes - 1)) === 0))
    throw new Error(`Cloth vertexLanes must be a power of two from 1 to ${CLOTH_WIDE_WORKGROUP}, got ${lanes}`);
  if (patch && contact) throw new Error('Cloth Tier B does not take a contact block yet: truncation must run inside the patch dispatch');
  return /* wgsl */`
struct Params {
  n: vec4u,      // N, T, E, H
  topo0: vec4u,  // vertexFaceOffsets, vertexFaces, vertexHingeOffsets, vertexHinges
  topo1: vec4u,  // vertexEdgeOffsets, vertexEdges, colour order, hinge vertices
  topo2: vec4u,  // triangles (v0, v1, v2, material), edges (v0, v1, f0, f1)
  rest0: vec4u,  // face Dm^-1, face (area, damping), hinge (theta0, k, kd), edge (L, eps)
  rest1: vec4u,  // vertex (m, elastic, length, group), anchor, material, script
  flags: vec4u,  // script count, BDF2 requested, strain limited, partial count
  state0: vec4u, // segment stride, edge duals, partials, clock
  state1: vec4u, // damping tangent: faces (2 records), hinges (3 records), damped flag
  gravity: vec4f, // xyz, dt
  misc: vec4f,   // clock origin (s), Chebyshev rho, initial guess (0 previous, 1 adaptive, 2 inertial)
  tierB: vec4u,  // colour count
  partA: vec4u,  // Tier B partition 0: owner, patch offsets, vertices, colour offsets (topo words)
  partB: vec4u,  // partition 1
}
struct Dispatch { range: vec4u, values: vec4f } // (start, count, iteration) or Tier B (partition, inner iterations), (omega)

@group(0) @binding(${CLOTH_BINDING.params}) var<uniform> P: Params;
@group(0) @binding(${CLOTH_BINDING.dispatch}) var<uniform> R: Dispatch;
@group(0) @binding(${CLOTH_BINDING.positions}) var<storage, read_write> positions: array<vec4f>;
@group(0) @binding(${CLOTH_BINDING.state}) var<storage, read_write> state: array<vec4f>;
@group(0) @binding(${CLOTH_BINDING.status}) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(${CLOTH_BINDING.topo}) var<storage, read> topo: array<u32>;
@group(0) @binding(${CLOTH_BINDING.rest}) var<storage, read> rest: array<vec4f>;
// The same topo words as vec4 records: triangles, edges and hinges are 4-word records at 64-word aligned segments.
@group(0) @binding(${CLOTH_BINDING.topo4}) var<storage, read> topo4: array<vec4u>;

const ALPHA: f32 = ${CLOTH_AVBD.alpha};
const GAMMA: f32 = ${CLOTH_AVBD.gamma};
const BETA: f32 = ${CLOTH_AVBD.beta};
const K_MAX_RATIO: f32 = ${CLOTH_AVBD.maxStiffnessRatio};
const ARMIJO: f32 = ${CLOTH_LINE_SEARCH.armijo};
const HALVINGS: u32 = ${CLOTH_LINE_SEARCH.halvings}u;
const EPS32: f32 = 1.1920929e-7;
const PI: f32 = 3.14159265358979;
const S_VEL = ${CLOTH_STATE.velocity}u; const S_NORMAL = ${CLOTH_STATE.normal}u; const S_XT = ${CLOTH_STATE.xt}u; const S_VPREV = ${CLOTH_STATE.vPrev}u; const S_DXPREV = ${CLOTH_STATE.dxPrev}u;
const S_Y = ${CLOTH_STATE.y}u; const S_XHAT = ${CLOTH_STATE.xHat}u; const S_PREV1 = ${CLOTH_STATE.prev1}u;
const S_PREV2 = ${CLOTH_STATE.prev2}u; const S_PINC = ${CLOTH_STATE.pinTarget}u; const S_PIND = ${CLOTH_STATE.pinDual}u; const S_POUT = ${CLOTH_STATE.patchOut}u;
const RING_BASE = ${CLOTH_STATUS_RING_BASE}u; const SLOT_WORDS = ${CLOTH_STATUS_SLOT_WORDS}u; const RING_SLOTS = ${CLOTH_STATUS_RING_SLOTS}u;
const WG = ${CLOTH_WORKGROUP}u;
const IDENTITY = mat3x3f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);

fn seg(s: u32, v: u32) -> u32 { return s * P.state0.x + v; }
${patch ? `const SLOT_BITS = ${CLOTH_PATCH_SLOT_BITS}u; const PATCH_WG = ${patch.workgroup}u;
var<workgroup> patchX: array<vec4f, ${patch.maxVertices}>;
var<private> patchId: u32;
var<private> ownerBase: u32;
// The patch's own vertices live in workgroup memory; every other vertex is frozen at the dispatch's start.
fn pos(i: u32) -> vec3f {
  let o = topo[ownerBase + i];
  if ((o >> SLOT_BITS) == patchId) { return patchX[o & ((1u << SLOT_BITS) - 1u)].xyz; }
  return positions[i].xyz;
}` : `fn pos(i: u32) -> vec3f { return positions[i].xyz; }`}
fn tris(f: u32) -> vec4u { return topo4[(P.topo2.x >> 2u) + f]; }
fn edges(e: u32) -> vec4u { return topo4[(P.topo2.y >> 2u) + e]; }
fn hingeVerts(h: u32) -> vec4u { return topo4[(P.topo1.w >> 2u) + h]; }
fn edgeDual(e: u32) -> vec4f { return state[P.state0.y + e]; }
// The clock record: (step, previous step, h~, BDF2), all as exact floats.
fn clockStep() -> u32 { return u32(state[P.state0.w].x); }
fn previousStep() -> f32 { return state[P.state0.w].y; }
fn hEff() -> f32 { return state[P.state0.w].z; }
fn bdf2Step() -> bool { return state[P.state0.w].w != 0.0; }
fn vertexRest(v: u32) -> vec4f { return rest[P.rest1.x + v]; }
fn vertexGroup(v: u32) -> u32 { return u32(vertexRest(v).w); }
fn referenceStiffness(v: u32) -> f32 { let r = vertexRest(v); let h = hEff(); return r.x / (h * h) + r.y; }
fn outer(u: vec3f, w: vec3f) -> mat3x3f { return mat3x3f(u * w.x, u * w.y, u * w.z); }
fn finite3(x: vec3f) -> bool { return all(abs(x) <= vec3f(3.0e38)); }

${contact ? `fn ogcX(i: u32) -> vec3f { return positions[i].xyz; }
fn ogcXt(i: u32) -> vec3f { return state[seg(S_XT, i)].xyz; }
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

struct Block { g: vec3f, H: mat3x3f, e: f32 }

struct Membrane { f0: vec3f, f1: vec3f, j1: vec3f, j2: vec3f, j3: vec3f, s11: f32, s22: f32, tau: f32, d11: f32, d22: f32, sigma: f32, e: f32, area: f32, c12: f32, g: f32 }
fn membraneAt(f: u32, k: u32, xv: vec3f) -> Membrane {
  let t = tris(f);
  var x0 = pos(t.x); var x1 = pos(t.y); var x2 = pos(t.z);
  if (k == 0u) { x0 = xv; } else if (k == 1u) { x1 = xv; } else { x2 = xv; }
  let dm = rest[P.rest0.x + f]; let area = rest[P.rest0.y + f].x;
  let c0 = rest[P.rest1.z + 2u * t.w]; let c1 = rest[P.rest1.z + 2u * t.w + 1u];
  let e1 = x1 - x0; let e2 = x2 - x0;
  let f0 = e1 * dm.x + e2 * dm.y; let f1 = e1 * dm.z + e2 * dm.w;
  let e11 = 0.5 * (dot(f0, f0) - 1.0); let e22 = 0.5 * (dot(f1, f1) - 1.0); let gam = dot(f0, f1);
  let w11 = select(0.0, (1.0 - c1.z) * c1.x, e11 < 0.0); let w22 = select(0.0, (1.0 - c1.z) * c1.y, e22 < 0.0);
  let s11 = c0.x * e11 + c0.z * e22 - w11 * e11; let s22 = c0.y * e22 + c0.z * e11 - w22 * e22; let tau = c0.w * gam;
  var a = dm.y; var b = dm.w;
  if (k == 0u) { a = -dm.x - dm.y; b = -dm.z - dm.w; } else if (k == 1u) { a = dm.x; b = dm.z; }
  let energy = area * (0.5 * (c0.x * e11 * e11 + 2.0 * c0.z * e11 * e22 + c0.y * e22 * e22) - 0.5 * (w11 * e11 * e11 + w22 * e22 * e22) + 0.5 * c0.w * gam * gam);
  return Membrane(f0, f1, a * f0, b * f1, a * f1 + b * f0, s11, s22, tau, c0.x - w11, c0.y - w22, s11 * a * a + s22 * b * b + 2.0 * tau * a * b, energy, area, c0.z, c0.w);
}
fn membraneBlock(f: u32, k: u32, xv: vec3f) -> Block {
  let m = membraneAt(f, k, xv);
  let g = m.area * (m.s11 * m.j1 + m.s22 * m.j2 + m.tau * m.j3);
  var H = m.area * (m.d11 * outer(m.j1, m.j1) + m.d22 * outer(m.j2, m.j2) + m.c12 * (outer(m.j1, m.j2) + outer(m.j2, m.j1)) + m.g * outer(m.j3, m.j3) + m.sigma * IDENTITY);
  // J D J^T is PSD, so only a compressive sigma makes the block indefinite.
  if (m.sigma < 0.0) { H = projectMembranePSD(H, m.f0, m.f1); }
  return Block(g, H, m.e);
}

struct HingeFull { angle: f32, g0: vec3f, g1: vec3f, g2: vec3f, g3: vec3f, ok: bool }
// Signed dihedral angle of a hinge (opposite0, opposite1, edge start, edge end) and its gradient for all four vertices.
fn hingeGeometry(x0: vec3f, x1: vec3f, x2: vec3f, x3: vec3f) -> HingeFull {
  var out: HingeFull;
  let n1 = cross(x2 - x0, x3 - x0); let n2 = cross(x3 - x1, x2 - x1); let e = x3 - x2;
  let q1 = dot(n1, n1); let q2 = dot(n2, n2); let le = length(e);
  out.ok = q1 > 0.0 && q2 > 0.0 && le > 0.0;
  if (!out.ok) { return out; }
  let l12 = sqrt(q1) * sqrt(q2);
  out.angle = atan2(dot(cross(n1, n2), e) / (l12 * le), dot(n1, n2) / l12);
  let u1 = n1 / q1; let u2 = n2 / q2; let eh = e / le;
  out.g0 = -le * u1; out.g1 = -le * u2;
  out.g2 = -(dot(x0 - x3, eh) * u1 + dot(x1 - x3, eh) * u2);
  out.g3 = dot(x0 - x2, eh) * u1 + dot(x1 - x2, eh) * u2;
  return out;
}
fn pick4(k: u32, a: vec3f, b: vec3f, c: vec3f, d: vec3f) -> vec3f { return select(select(a, b, k == 1u), select(c, d, k == 3u), k >= 2u); }
struct Hinge { angle: f32, g: vec3f, ok: bool }
fn hingeAt(h: u32, k: u32, xv: vec3f) -> Hinge {
  let hv = hingeVerts(h);
  let x0 = select(pos(hv.x), xv, k == 0u); let x1 = select(pos(hv.y), xv, k == 1u);
  let x2 = select(pos(hv.z), xv, k == 2u); let x3 = select(pos(hv.w), xv, k == 3u);
  let q = hingeGeometry(x0, x1, x2, x3);
  return Hinge(q.angle, pick4(k, q.g0, q.g1, q.g2, q.g3), q.ok);
}

// Rayleigh damping as a dissipation potential with the element tangent frozen at x_t (oracle: ClothDampingTangent).
fn faceDamping(f: u32, k: u32, xv: vec3f) -> Block {
  let fr = rest[P.rest0.y + f];
  if (!(fr.y > 0.0)) { return Block(vec3f(0.0), mat3x3f(), 0.0); }
  let t = tris(f); let dm = rest[P.rest0.x + f]; let c0 = rest[P.rest1.z + 2u * t.w];
  let T0 = state[P.state1.x + 2u * f]; let T1 = state[P.state1.x + 2u * f + 1u];
  let f0 = T0.xyz; let f1 = T1.xyz;
  let a = vec3f(-dm.x - dm.y, dm.x, dm.y); let b = vec3f(-dm.z - dm.w, dm.z, dm.w);
  let u0 = select(pos(t.x), xv, k == 0u) - state[seg(S_XHAT, t.x)].xyz;
  let u1 = select(pos(t.y), xv, k == 1u) - state[seg(S_XHAT, t.y)].xyz;
  let u2 = select(pos(t.z), xv, k == 2u) - state[seg(S_XHAT, t.z)].xyz;
  let df0 = a.x * u0 + a.y * u1 + a.z * u2; let df1 = b.x * u0 + b.y * u1 + b.z * u2;
  let rate = vec3f(dot(f0, df0), dot(f1, df1), dot(f0, df1) + dot(f1, df0));
  let sigma = vec3f(T0.w * rate.x + c0.z * rate.y, c0.z * rate.x + T1.w * rate.y, c0.w * rate.z);
  let ak = select(select(a.x, a.y, k == 1u), a.z, k == 2u); let bk = select(select(b.x, b.y, k == 1u), b.z, k == 2u);
  let j1 = ak * f0; let j2 = bk * f1; let j3 = ak * f1 + bk * f0;
  let scale = fr.y * fr.x / hEff();
  return Block(scale * (sigma.x * j1 + sigma.y * j2 + sigma.z * j3),
    scale * (T0.w * outer(j1, j1) + T1.w * outer(j2, j2) + c0.z * (outer(j1, j2) + outer(j2, j1)) + c0.w * outer(j3, j3)), 0.5 * scale * dot(rate, sigma));
}
fn hingeDamping(h: u32, k: u32, xv: vec3f) -> Block {
  let hr = rest[P.rest0.z + h];
  if (!(hr.z > 0.0)) { return Block(vec3f(0.0), mat3x3f(), 0.0); }
  let gb = P.state1.y + 3u * h;
  let G0 = state[gb]; let G1 = state[gb + 1u]; let G2 = state[gb + 2u];
  let g0 = G0.xyz; let g1 = vec3f(G0.w, G1.x, G1.y); let g2 = vec3f(G1.z, G1.w, G2.x); let g3 = G2.yzw;
  let hv = hingeVerts(h); let v0 = hv.x; let v1 = hv.y; let v2 = hv.z; let v3 = hv.w;
  let rate = dot(g0, select(pos(v0), xv, k == 0u) - state[seg(S_XHAT, v0)].xyz) + dot(g1, select(pos(v1), xv, k == 1u) - state[seg(S_XHAT, v1)].xyz)
    + dot(g2, select(pos(v2), xv, k == 2u) - state[seg(S_XHAT, v2)].xyz) + dot(g3, select(pos(v3), xv, k == 3u) - state[seg(S_XHAT, v3)].xyz);
  let scale = hr.z * hr.y / hEff(); let gk = pick4(k, g0, g1, g2, g3);
  return Block(scale * rate * gk, scale * outer(gk, gk), 0.5 * scale * rate * rate);
}
fn wrapAngle(d: f32) -> f32 { return d - 2.0 * PI * floor((d + PI) / (2.0 * PI)); }

fn augmented(k: f32, lambda: f32, C: f32, inequality: bool) -> f32 {
  var f = k * C + lambda;
  if (inequality) { f = max(f, 0.0); }
  return (f * f - lambda * lambda) / (2.0 * k);
}

struct System { g: vec3f, H: mat3x3f, e: f32, mag: f32 }

// Vertex v's local objective at xv (other vertices at their current positions):
// inertia, membrane, bending, damping, pins, strain limits${contact ? ', contact' : ''}.
fn vertexSystem(v: u32, xv: vec3f) -> System {
  let h = hEff(); let vr = vertexRest(v); let w = vr.x / (h * h);
  var s = System(vec3f(0.0), w * IDENTITY, 0.0, 0.0);
  let r = xv - state[seg(S_Y, v)].xyz;
  s.g = w * r; s.e = 0.5 * w * dot(r, r); s.mag = s.e;
  for (var j = topo[P.topo0.x + v]; j < topo[P.topo0.x + v + 1u]; j++) {
    let entry = topo[P.topo0.y + j]; let f = entry >> 2u;
    let b = membraneBlock(f, entry & 3u, xv); let d = faceDamping(f, entry & 3u, xv);
    s.g += b.g + d.g; s.H += b.H + d.H; s.e += b.e + d.e; s.mag += abs(b.e) + d.e;
  }
  for (var j = topo[P.topo0.z + v]; j < topo[P.topo0.z + v + 1u]; j++) {
    let entry = topo[P.topo0.w + j]; let hg = entry >> 2u;
    let hinge = hingeAt(hg, entry & 3u, xv);
    if (!hinge.ok) { continue; }
    let hr = rest[P.rest0.z + hg]; let delta = wrapAngle(hinge.angle - hr.x);
    s.g += hr.y * delta * hinge.g; s.H += hr.y * outer(hinge.g, hinge.g);
    let e = 0.5 * hr.y * delta * delta; s.e += e; s.mag += e;
  }
  for (var j = topo[P.topo0.z + v]; j < topo[P.topo0.z + v + 1u]; j++) {
    let entry = topo[P.topo0.w + j];
    let d = hingeDamping(entry >> 2u, entry & 3u, xv);
    s.g += d.g; s.H += d.H; s.e += d.e; s.mag += d.e;
  }
  return constraints(v, xv, s);
}

// Line-search energy changes in difference form. E(xv + d) - E(xv) is assembled from exact increments (strain,
// rate and residual increments are linear or quadratic in d), so the Armijo test does not compare two nearly equal f32
// energies: a near-converged vertex no longer fails every trial on rounding noise. Each returns (dE, floor): floor
// sums |increments| and the first-order rounding of the base quantities (strain ~ eps, residual ~ eps |x|), and the
// test allows 4 eps * floor.
fn quadDelta(aOld: f32, aNew: f32, e: f32, de: f32) -> f32 {
  if (aOld == aNew) { return aOld * (e * de + 0.5 * de * de); }
  let en = e + de; return 0.5 * (aNew * en * en - aOld * e * e);
}
fn membraneDelta(f: u32, k: u32, xv: vec3f, d: vec3f) -> vec2f {
  let t = tris(f);
  var x0 = pos(t.x); var x1 = pos(t.y); var x2 = pos(t.z);
  if (k == 0u) { x0 = xv; } else if (k == 1u) { x1 = xv; } else { x2 = xv; }
  let dm = rest[P.rest0.x + f]; let area = rest[P.rest0.y + f].x;
  let c0 = rest[P.rest1.z + 2u * t.w]; let c1 = rest[P.rest1.z + 2u * t.w + 1u];
  let e1 = x1 - x0; let e2 = x2 - x0;
  let f0 = e1 * dm.x + e2 * dm.y; let f1 = e1 * dm.z + e2 * dm.w;
  var a = dm.y; var b = dm.w;
  if (k == 0u) { a = -dm.x - dm.y; b = -dm.z - dm.w; } else if (k == 1u) { a = dm.x; b = dm.z; }
  let df0 = a * d; let df1 = b * d;
  let e11 = 0.5 * (dot(f0, f0) - 1.0); let e22 = 0.5 * (dot(f1, f1) - 1.0); let gam = dot(f0, f1);
  let d11 = dot(f0, df0) + 0.5 * dot(df0, df0); let d22 = dot(f1, df1) + 0.5 * dot(df1, df1);
  let dg = dot(f0, df1) + dot(df0, f1) + dot(df0, df1);
  let s11 = (1.0 - c1.z) * c1.x; let s22 = (1.0 - c1.z) * c1.y;
  let q11 = quadDelta(c0.x - select(0.0, s11, e11 < 0.0), c0.x - select(0.0, s11, e11 + d11 < 0.0), e11, d11);
  let q22 = quadDelta(c0.y - select(0.0, s22, e22 < 0.0), c0.y - select(0.0, s22, e22 + d22 < 0.0), e22, d22);
  let q12 = c0.z * (e11 * d22 + d11 * e22 + d11 * d22); let qs = c0.w * (gam * dg + 0.5 * dg * dg);
  let rounding = (c0.x + c0.z) * abs(d11) * (0.5 + e11) + (c0.y + c0.z) * abs(d22) * (0.5 + e22) + c0.w * abs(dg) * sqrt((1.0 + 2.0 * e11) * (1.0 + 2.0 * e22));
  return area * vec2f(q11 + q22 + q12 + qs, abs(q11) + abs(q22) + abs(q12) + abs(qs) + rounding);
}
fn faceDampingDelta(f: u32, k: u32, xv: vec3f, d: vec3f) -> vec2f {
  let fr = rest[P.rest0.y + f];
  if (!(fr.y > 0.0)) { return vec2f(0.0); }
  let t = tris(f); let dm = rest[P.rest0.x + f]; let c0 = rest[P.rest1.z + 2u * t.w];
  let T0 = state[P.state1.x + 2u * f]; let T1 = state[P.state1.x + 2u * f + 1u];
  let f0 = T0.xyz; let f1 = T1.xyz;
  let a = vec3f(-dm.x - dm.y, dm.x, dm.y); let b = vec3f(-dm.z - dm.w, dm.z, dm.w);
  let u0 = select(pos(t.x), xv, k == 0u) - state[seg(S_XHAT, t.x)].xyz;
  let u1 = select(pos(t.y), xv, k == 1u) - state[seg(S_XHAT, t.y)].xyz;
  let u2 = select(pos(t.z), xv, k == 2u) - state[seg(S_XHAT, t.z)].xyz;
  let r0 = a.x * u0 + a.y * u1 + a.z * u2; let r1 = b.x * u0 + b.y * u1 + b.z * u2;
  let rate = vec3f(dot(f0, r0), dot(f1, r1), dot(f0, r1) + dot(f1, r0));
  let ak = select(select(a.x, a.y, k == 1u), a.z, k == 2u); let bk = select(select(b.x, b.y, k == 1u), b.z, k == 2u);
  let dr = vec3f(ak * dot(f0, d), bk * dot(f1, d), bk * dot(f0, d) + ak * dot(f1, d));
  let sigma = vec3f(T0.w * rate.x + c0.z * rate.y, c0.z * rate.x + T1.w * rate.y, c0.w * rate.z);
  let dsigma = vec3f(T0.w * dr.x + c0.z * dr.y, c0.z * dr.x + T1.w * dr.y, c0.w * dr.z);
  let scale = fr.y * fr.x / hEff();
  let lin = scale * dot(dr, sigma); let quad = 0.5 * scale * dot(dr, dsigma);
  let ru = vec3f(abs(a.x) * length(u0) + abs(a.y) * length(u1) + abs(a.z) * length(u2), abs(b.x) * length(u0) + abs(b.y) * length(u1) + abs(b.z) * length(u2), 0.0);
  let roundRate = vec3f(length(f0) * ru.x, length(f1) * ru.y, length(f0) * ru.y + length(f1) * ru.x);
  return vec2f(lin + quad, abs(lin) + quad + scale * dot(abs(dsigma), roundRate));
}
fn hingeDelta(h: u32, k: u32, xv: vec3f, d: vec3f) -> vec2f {
  let hr = rest[P.rest0.z + h];
  let before = hingeAt(h, k, xv); let after = hingeAt(h, k, xv + d);
  let w0 = wrapAngle(before.angle - hr.x); let dt = wrapAngle(after.angle - before.angle);
  var de = 0.0;
  if (before.ok && after.ok) { de = hr.y * dt * (w0 + 0.5 * dt); }
  else { let w1 = wrapAngle(after.angle - hr.x); de = select(0.0, 0.5 * hr.y * w1 * w1, after.ok) - select(0.0, 0.5 * hr.y * w0 * w0, before.ok); }
  return vec2f(de, abs(de) + hr.y * PI * (abs(w0) + abs(dt)));
}
fn hingeDampingDelta(h: u32, k: u32, xv: vec3f, d: vec3f) -> vec2f {
  let hr = rest[P.rest0.z + h];
  if (!(hr.z > 0.0)) { return vec2f(0.0); }
  let gb = P.state1.y + 3u * h;
  let G0 = state[gb]; let G1 = state[gb + 1u]; let G2 = state[gb + 2u];
  let g0 = G0.xyz; let g1 = vec3f(G0.w, G1.x, G1.y); let g2 = vec3f(G1.z, G1.w, G2.x); let g3 = G2.yzw;
  let hv = hingeVerts(h); let v0 = hv.x; let v1 = hv.y; let v2 = hv.z; let v3 = hv.w;
  let u0 = select(pos(v0), xv, k == 0u) - state[seg(S_XHAT, v0)].xyz; let u1 = select(pos(v1), xv, k == 1u) - state[seg(S_XHAT, v1)].xyz;
  let u2 = select(pos(v2), xv, k == 2u) - state[seg(S_XHAT, v2)].xyz; let u3 = select(pos(v3), xv, k == 3u) - state[seg(S_XHAT, v3)].xyz;
  let rate = dot(g0, u0) + dot(g1, u1) + dot(g2, u2) + dot(g3, u3);
  let scale = hr.z * hr.y / hEff(); let dr = dot(pick4(k, g0, g1, g2, g3), d);
  let lin = scale * rate * dr; let quad = 0.5 * scale * dr * dr;
  let roundRate = length(g0) * length(u0) + length(g1) * length(u1) + length(g2) * length(u2) + length(g3) * length(u3);
  return vec2f(lin + quad, abs(lin) + quad + scale * abs(dr) * roundRate);
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
// Inertia and pins of vertex v.
fn ownDelta(v: u32, xv: vec3f, d: vec3f) -> vec2f {
  let h = hEff(); let w = vertexRest(v).x / (h * h);
  let r = xv - state[seg(S_Y, v)].xyz;
  let lin = w * dot(r, d); let quad = 0.5 * w * dot(d, d);
  var out = vec2f(lin + quad, abs(lin) + quad + w * length(r) * length(d));
  if (vertexGroup(v) > 0u) {
    let p = state[seg(S_PIND, v)]; let C = xv - state[seg(S_PINC, v)].xyz;
    let f = p.w * C + p.xyz; let l = dot(f, d); let q = 0.5 * p.w * dot(d, d);
    out += vec2f(l + q, abs(l) + q + p.w * length(C) * length(d));
  }
  return out;
}
${contact ? `fn contactDelta(v: u32, xv: vec3f, ec0: f32, d: vec3f) -> vec2f {
  let ec1 = ogcVertexContactEnergy(v, xv + d); return vec2f(ec1 - ec0, abs(ec0) + abs(ec1));
}` : ''}
// The whole star of v: (dE, floor) of moving v from xv by d.
fn vertexDelta(v: u32, xv: vec3f, d: vec3f${contact ? ', ec0: f32' : ''}) -> vec2f {
  var out = ownDelta(v, xv, d);
  for (var j = topo[P.topo0.x + v]; j < topo[P.topo0.x + v + 1u]; j++) {
    let entry = topo[P.topo0.y + j];
    out += membraneDelta(entry >> 2u, entry & 3u, xv, d) + faceDampingDelta(entry >> 2u, entry & 3u, xv, d);
  }
  for (var j = topo[P.topo0.z + v]; j < topo[P.topo0.z + v + 1u]; j++) {
    let entry = topo[P.topo0.w + j];
    out += hingeDelta(entry >> 2u, entry & 3u, xv, d) + hingeDampingDelta(entry >> 2u, entry & 3u, xv, d);
  }
  if (P.flags.z != 0u) {
    for (var j = topo[P.topo1.x + v]; j < topo[P.topo1.x + v + 1u]; j++) { out += edgeDelta(v, xv, topo[P.topo1.y + j] >> 2u, d); }
  }
  ${contact ? 'out += contactDelta(v, xv, ec0, d);' : ''}
  return out;
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
  // AVBD (14): a force clamped at zero keeps the rescaled stiffness -lambda / C.
  var kEff = dual.y;
  if (!(dual.y * C + dual.x > 0.0)) { kEff = select(0.0, max(-dual.x / C, 0.0), C != 0.0); }
  s.g += f * n; s.H += (kEff - f / len) * outer(n, n) + (f / len) * IDENTITY;
  let en = augmented(dual.y, dual.x, C, true); s.e += en; s.mag += abs(en);
  return s;
}

// Pin${contact ? ' and contact' : ''} terms of vertex v.
fn pinTerms(v: u32, xv: vec3f, s0: System) -> System {
  var s = s0;
  if (vertexGroup(v) > 0u) {
    let d = state[seg(S_PIND, v)]; let C = xv - state[seg(S_PINC, v)].xyz;
    s.g += d.w * C + d.xyz; s.H += d.w * IDENTITY;
    for (var c = 0u; c < 3u; c++) { let e = augmented(d.w, d[c], C[c], false); s.e += e; s.mag += abs(e); }
  }
${contact ? `  var fc = vec3f(0.0); var Hc = mat3x3f();
  ogcAccumulateVertexContact(v, xv, &fc, &Hc);
  s.g -= fc; s.H += Hc;
  let ec = ogcVertexContactEnergy(v, xv); s.e += ec; s.mag += abs(ec);` : ''}
  return s;
}

fn constraints(v: u32, xv: vec3f, s0: System) -> System {
  var s = s0;
  if (P.flags.z != 0u) {
    for (var j = topo[P.topo1.x + v]; j < topo[P.topo1.x + v + 1u]; j++) { s = edgeTerm(v, xv, topo[P.topo1.y + j] >> 2u, s); }
  }
  return pinTerms(v, xv, s);
}

fn solve3(H: mat3x3f, b: vec3f) -> vec4f {
  let det = determinant(H);
  if (!(det > 0.0) || !(abs(det) <= 3.0e38)) { return vec4f(0.0); }
  let c0 = cross(H[1], H[2]); let c1 = cross(H[2], H[0]); let c2 = cross(H[0], H[1]);
  // H symmetric: inverse rows are the cofactor columns / det.
  return vec4f(vec3f(dot(c0, b), dot(c1, b), dot(c2, b)) / det, 1.0);
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

// Step begin (one thread): BDF2 needs a previous step of the same length.
@compute @workgroup_size(1) fn beginStep() {
  let dt = P.gravity.w; let previous = previousStep();
  let bdf2 = P.flags.y != 0u && previous > 0.0 && abs(previous - dt) <= 1e-6 * dt;
  state[P.state0.w].z = select(dt, 2.0 * dt / 3.0, bdf2);
  state[P.state0.w].w = select(0.0, 1.0, bdf2);
}

// At x_t, before the frame pass moves positions: AVBD warm start of the edge duals and
// their stabilized limits, and the damping tangent of every face and hinge.
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
      d.z = limit + ALPHA * max(length(pos(ev.x) - pos(ev.y)) - limit, 0.0);
      state[P.state0.y + i] = d;
    }
  }
  if (P.state1.z == 0u) { return; }
  if (i < P.n.y) {
    let t = tris(i); let dm = rest[P.rest0.x + i]; let c1 = rest[P.rest1.z + 2u * t.w]; let c2 = rest[P.rest1.z + 2u * t.w + 1u];
    let x0 = pos(t.x); let e1 = pos(t.y) - x0; let e2 = pos(t.z) - x0;
    let f0 = e1 * dm.x + e2 * dm.y; let f1 = e1 * dm.z + e2 * dm.w;
    let e11 = 0.5 * (dot(f0, f0) - 1.0); let e22 = 0.5 * (dot(f1, f1) - 1.0);
    state[P.state1.x + 2u * i] = vec4f(f0, c1.x - select(0.0, (1.0 - c2.z) * c2.x, e11 < 0.0));
    state[P.state1.x + 2u * i + 1u] = vec4f(f1, c1.y - select(0.0, (1.0 - c2.z) * c2.y, e22 < 0.0));
  }
  if (i < P.n.w) {
    let base = P.topo1.w + 4u * i;
    let q = hingeGeometry(pos(topo[base]), pos(topo[base + 1u]), pos(topo[base + 2u]), pos(topo[base + 3u]));
    let on = select(0.0, 1.0, q.ok); let g0 = on * q.g0; let g1 = on * q.g1; let g2 = on * q.g2; let g3 = on * q.g3;
    let gb = P.state1.y + 3u * i;
    state[gb] = vec4f(g0, g1.x); state[gb + 1u] = vec4f(g1.yz, g2.xy); state[gb + 2u] = vec4f(g2.z, g3);
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
    state[seg(S_PINC, v)] = vec4f(p1 + ALPHA * (x - p0), 0.0);
    var d = state[seg(S_PIND, v)];
    d = vec4f(d.xyz * (ALPHA * GAMMA), max(GAMMA * d.w, referenceStiffness(v)));
    state[seg(S_PIND, v)] = d;
  }
  positions[v] = vec4f(init, positions[v].w);
  state[seg(S_PREV1, v)] = vec4f(init, 0.0);
  state[seg(S_PREV2, v)] = vec4f(init, 0.0);
}

// One VBD block update of vertex v from xv: the Newton step with the per-vertex Armijo line search on the
// difference-form energy change (xv if no trial descends).
fn vertexStep(v: u32, xv: vec3f) -> vec3f {
  let s = vertexSystem(v, xv);
  let solved = solve3(s.H, -s.g);
  if (solved.w == 0.0) { return xv; }
  let delta = solved.xyz;
  ${contact ? 'let ec0 = ogcVertexContactEnergy(v, xv);' : ''}
  var alpha = 1.0;
  for (var i = 0u; i <= HALVINGS; i++) {
    let trial = xv + alpha * delta; let d = trial - xv;
    let change = vertexDelta(v, xv, d${contact ? ', ec0' : ''});
    if (change.x <= ARMIJO * dot(s.g, d) + 4.0 * EPS32 * change.y) { return trial; }
    alpha *= 0.5;
  }
  return xv;
}

// Every vertex of one colour.
@compute @workgroup_size(${CLOTH_WORKGROUP}) fn solveColour(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= R.range.y) { return; }
  let v = topo[P.topo1.z + R.range.x + id.x];
  positions[v] = vec4f(vertexStep(v, pos(v)), positions[v].w);
}
${patch ? '' : `
// The same block update with ${lanes} lanes per vertex (${CLOTH_WIDE_WORKGROUP / lanes} vertices per ${CLOTH_WIDE_WORKGROUP}-lane workgroup). Lane l of a vertex
// takes its faces l, l + ${lanes}, ..., then its hinges and strain-limited edges the same way, and lane 0 adds inertia and pins${contact ? ' and contact' : ''}:
// every lane of the SIMD group runs the same element kind at the same time (no kind divergence), the block and every
// line-search trial are summed per vertex in workgroup memory, and a colour has ${lanes} threads per vertex.
const VL = ${lanes}u;
const VPW = ${CLOTH_WIDE_WORKGROUP / lanes}u;
var<workgroup> wA: array<vec4f, ${CLOTH_WIDE_WORKGROUP}>;
var<workgroup> wB: array<vec4f, ${CLOTH_WIDE_WORKGROUP}>;
var<workgroup> wC: array<vec4f, ${CLOTH_WIDE_WORKGROUP}>;
var<workgroup> wDone: array<u32, ${CLOTH_WIDE_WORKGROUP / lanes}>;
var<workgroup> wAll: u32;

fn ownSystem(v: u32, xv: vec3f) -> System {
  let h = hEff(); let w = vertexRest(v).x / (h * h);
  let r = xv - state[seg(S_Y, v)].xyz;
  return pinTerms(v, xv, System(w * r, w * IDENTITY, 0.0, 0.0));
}

// Per-vertex sums of the (a, b, c) records of its VL lanes; every lane gets its vertex's totals.
fn groupSum(lane: u32, a: vec4f, b: vec4f, c: vec4f) -> array<vec4f, 3> {
  workgroupBarrier();
  wA[lane] = a; wB[lane] = b; wC[lane] = c;
  workgroupBarrier();
  let sub = lane % VL;
  for (var k = VL / 2u; k > 0u; k >>= 1u) {
    if (sub < k) { wA[lane] += wA[lane + k]; wB[lane] += wB[lane + k]; wC[lane] += wC[lane + k]; }
    workgroupBarrier();
  }
  let head = lane - sub;
  return array<vec4f, 3>(wA[head], wB[head], wC[head]);
}

@compute @workgroup_size(${CLOTH_WIDE_WORKGROUP}) fn solveColourWide(@builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) lane: u32) {
  let group = lane / VL; let sub = lane % VL;
  let index = (wg.x + wg.y * nwg.x) * VPW + group;
  let valid = index < R.range.y;
  var v = 0u; var xv = vec3f(0.0); var nf = 0u; var nh = 0u; var ne = 0u;
  var s = System(vec3f(0.0), mat3x3f(), 0.0, 0.0);
  if (valid) {
    v = topo[P.topo1.z + R.range.x + index]; xv = pos(v);
    let f0 = topo[P.topo0.x + v]; nf = topo[P.topo0.x + v + 1u] - f0;
    let h0 = topo[P.topo0.z + v]; nh = topo[P.topo0.z + v + 1u] - h0;
    let e0 = topo[P.topo1.x + v]; ne = select(0u, topo[P.topo1.x + v + 1u] - e0, P.flags.z != 0u);
    if (sub == 0u) { s = ownSystem(v, xv); }
    for (var j = sub; j < nf; j += VL) {
      let entry = topo[P.topo0.y + f0 + j]; let f = entry >> 2u;
      let b = membraneBlock(f, entry & 3u, xv); let d = faceDamping(f, entry & 3u, xv);
      s.g += b.g + d.g; s.H += b.H + d.H;
    }
    for (var j = sub; j < nh; j += VL) {
      let entry = topo[P.topo0.w + h0 + j]; let hg = entry >> 2u;
      let hinge = hingeAt(hg, entry & 3u, xv);
      if (hinge.ok) {
        let hr = rest[P.rest0.z + hg]; let delta = wrapAngle(hinge.angle - hr.x);
        s.g += hr.y * delta * hinge.g; s.H += hr.y * outer(hinge.g, hinge.g);
      }
      let d = hingeDamping(hg, entry & 3u, xv); s.g += d.g; s.H += d.H;
    }
    for (var j = sub; j < ne; j += VL) { s = edgeTerm(v, xv, topo[P.topo1.y + e0 + j] >> 2u, s); }
  }
  let sum = groupSum(lane, vec4f(s.g, 0.0), vec4f(s.H[0][0], s.H[1][0], s.H[2][0], 0.0), vec4f(s.H[1][1], s.H[2][1], s.H[2][2], 0.0));
  let g = sum[0].xyz;
  let H = mat3x3f(sum[1].x, sum[1].y, sum[1].z, sum[1].y, sum[2].x, sum[2].y, sum[1].z, sum[2].y, sum[2].z);
  let solved = solve3(H, -g);
  let delta = solved.xyz;
  if (sub == 0u) { wDone[group] = select(1u, 0u, valid && solved.w != 0.0); }
  ${contact ? 'var ec0 = 0.0; if (valid && sub == 0u) { ec0 = ogcVertexContactEnergy(v, xv); }' : ''}
  var alpha = 1.0;
  for (var i = 0u; i <= HALVINGS; i++) {
    workgroupBarrier();
    if (lane == 0u) { var all = 1u; for (var q = 0u; q < VPW; q++) { all &= wDone[q]; } wAll = all; }
    if (workgroupUniformLoad(&wAll) != 0u) { break; }
    let live = wDone[group] == 0u;
    let trial = xv + alpha * delta; let d = trial - xv;
    var en = vec2f(0.0);
    if (live) {
      if (sub == 0u) { en = ownDelta(v, xv, d)${contact ? ' + contactDelta(v, xv, ec0, d)' : ''}; }
      for (var j = sub; j < nf; j += VL) {
        let entry = topo[P.topo0.y + topo[P.topo0.x + v] + j];
        en += membraneDelta(entry >> 2u, entry & 3u, xv, d) + faceDampingDelta(entry >> 2u, entry & 3u, xv, d);
      }
      for (var j = sub; j < nh; j += VL) {
        let entry = topo[P.topo0.w + topo[P.topo0.z + v] + j];
        en += hingeDelta(entry >> 2u, entry & 3u, xv, d) + hingeDampingDelta(entry >> 2u, entry & 3u, xv, d);
      }
      for (var j = sub; j < ne; j += VL) { en += edgeDelta(v, xv, topo[P.topo1.y + topo[P.topo1.x + v] + j] >> 2u, d); }
    }
    let t = groupSum(lane, vec4f(en, 0.0, 0.0), vec4f(0.0), vec4f(0.0));
    if (live && sub == 0u) {
      let ok = t[0].x <= ARMIJO * dot(g, d) + 4.0 * EPS32 * t[0].y;
      if (ok) { positions[v] = vec4f(trial, positions[v].w); wDone[group] = 1u; }
    }
    alpha *= 0.5;
  }
}
`}${patch ? `
// Tier B: one workgroup per patch of partition R.range.x runs R.range.y Gauss-Seidel sweeps over the patch's colours
// in workgroup memory, then writes its vertices to the patchOut segment (commitPatches makes them the iterate).
@compute @workgroup_size(${patch.workgroup}) fn solvePatches(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) lane: u32) {
  var part = P.partA;
  if (R.range.x == 1u) { part = P.partB; }
  ownerBase = part.x; patchId = wg.x;
  let start = topo[part.y + patchId]; let count = topo[part.y + patchId + 1u] - start; let C = P.tierB.x;
  for (var s = lane; s < count; s += PATCH_WG) { patchX[s] = vec4f(positions[topo[part.z + start + s]].xyz, 0.0); }
  workgroupBarrier();
  for (var k = 0u; k < R.range.y; k++) {
    for (var c = 0u; c < C; c++) {
      let cb = part.w + patchId * (C + 1u) + c;
      let c1 = topo[cb + 1u];
      for (var s = topo[cb] + lane; s < c1; s += PATCH_WG) {
        let v = topo[part.z + start + s];
        patchX[s] = vec4f(vertexStep(v, patchX[s].xyz), 0.0);
      }
      workgroupBarrier();
    }
  }
  for (var s = lane; s < count; s += PATCH_WG) { state[seg(S_POUT, topo[part.z + start + s])] = patchX[s]; }
}` : ''}

// Tier B: the patch results become the iterate.
@compute @workgroup_size(${CLOTH_WORKGROUP}) fn commitPatches(@builtin(global_invocation_id) id: vec3u) {
  let v = id.x;
  if (v >= P.n.x) { return; }
  positions[v] = vec4f(state[seg(S_POUT, v)].xyz, positions[v].w);
}

// After all colours: Chebyshev (free vertices${contact ? ' without contacts' : ''}), the iterate history and the pin duals.
@compute @workgroup_size(${CLOTH_WORKGROUP}) fn postIteration(@builtin(global_invocation_id) id: vec3u) {
  let v = id.x;
  if (v >= P.n.x) { return; }
  var x = pos(v);
  let p1 = state[seg(S_PREV1, v)]; let p2 = state[seg(S_PREV2, v)].xyz;
  let omega = R.values.x; let group = vertexGroup(v);
  if (R.range.z >= 2u && omega != 1.0 && group == 0u${contact ? ' && ogcContactCount(v) == 0u' : ''}) {
    x = omega * (x - p2) + p2;
    positions[v] = vec4f(x, positions[v].w);
  }
  state[seg(S_PREV2, v)] = p1;
  state[seg(S_PREV1, v)] = vec4f(x, 0.0);
  if (group > 0u) {
    let s = referenceStiffness(v); let kMax = K_MAX_RATIO * s; let l = vertexRest(v).z;
    let d = state[seg(S_PIND, v)]; let C = x - state[seg(S_PINC, v)].xyz;
    let lambda = clamp(d.w * C + d.xyz, vec3f(-kMax * l), vec3f(kMax * l));
    state[seg(S_PIND, v)] = vec4f(lambda, min(d.w + BETA * s * length(C) / l, kMax));
  }
}

@compute @workgroup_size(${CLOTH_WORKGROUP}) fn edgeDualUpdate(@builtin(global_invocation_id) id: vec3u) {
  let e = id.x;
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
    a = vec4f(dot(s.g, s.g), m * m * dot(g, g), 0.5 * m * dot(vel, vel), -m * dot(g, x));
    var elastic = 0.0;
    for (var j = topo[P.topo0.x + v]; j < topo[P.topo0.x + v + 1u]; j++) {
      let entry = topo[P.topo0.y + j];
      if ((entry & 3u) == 0u) { elastic += membraneAt(entry >> 2u, 0u, x).e; }
    }
    for (var j = topo[P.topo0.z + v]; j < topo[P.topo0.z + v + 1u]; j++) {
      let entry = topo[P.topo0.w + j];
      if ((entry & 3u) != 0u) { continue; }
      let hg = entry >> 2u; let hinge = hingeAt(hg, 0u, x);
      if (hinge.ok) { let hr = rest[P.rest0.z + hg]; let delta = wrapAngle(hinge.angle - hr.x); elastic += 0.5 * hr.y * delta * delta; }
    }
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
  let x = pos(v);
  let vNew = (x - state[seg(S_XHAT, v)].xyz) / hEff();
  state[seg(S_VPREV, v)] = state[seg(S_VEL, v)];
  state[seg(S_VEL, v)] = vec4f(vNew, 0.0);
  state[seg(S_DXPREV, v)] = vec4f(x - state[seg(S_XT, v)].xyz, 0.0);
  var n = vec3f(0.0);
  for (var j = topo[P.topo0.x + v]; j < topo[P.topo0.x + v + 1u]; j++) {
    let t = tris(topo[P.topo0.y + j] >> 2u);
    let a = pos(t.x); n += cross(pos(t.y) - a, pos(t.z) - a);
  }
  let ln = length(n);
  state[seg(S_NORMAL, v)] = vec4f(select(vec3f(0.0, 1.0, 0.0), n / ln, ln > 0.0), 0.0);
  if (!finite3(x) || !finite3(vNew)) { atomicOr(&status[0], ${CLOTH_STATUS.nonFinite}u); }
}
`;
}
