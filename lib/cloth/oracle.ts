/** Float64 CPU oracle of the cloth step: one vertex block update and the
 * whole implicit step built from it (Gauss-Seidel in vertex order, with
 * oracle-coarse.ts to converge a stiff sheet). The GPU solver (`gpu-solver.ts`,
 * Newton-PCG) minimizes the same objective in f32, so a converged oracle step
 * is its reference. This file is a test oracle only; nothing in a simulation
 * step calls it (the GPU tables read `buildClothSolverModel`).
 *
 * Energies (all per vertex block, other vertices held, as VBD solves):
 *  - inertia (m / 2h~^2)|x - y|^2, with h~ = h (implicit Euler) or 2h/3 (BDF2);
 *  - orthotropic StVK membrane on Green strain in material (warp, weft)
 *    coordinates with C1 compression softening, per-vertex Hessian projected
 *    to PSD;
 *  - Kirchhoff plate bending (B/2)[nu (Delta x)^2 + (1 - nu)|grad grad x|^2], three rows per vertex (topology `clothBendingRows`), exact Hessian;
 *  - stiffness-proportional Rayleigh damping as a dissipation potential
 *    with the element tangent frozen at the step start (see ClothDampingTangent);
 *  - AVBD hard constraints: pins on scripted attachments (3D equality) and
 *    an optional per-edge strain limit |e| <= (1 + eps) L (inequality), with
 *    progressive stiffness, warm-started duals and the alpha regularization.
 */
import type { ClothAttachmentScript, ClothIntegrator, ClothMaterial, ClothVector } from './types';
import { CLOTH_BEND_ROWS, CLOTH_NO_FACE, type PreparedClothTopology } from './topology';

/** AVBD constants (Giles et al. 2025, Table 2). beta is made scale free:
 * a constraint's stiffness grows by beta * s * |C| / l per iteration, with
 * s the vertex's reference stiffness and l its edge length. */
export const CLOTH_AVBD = Object.freeze({ alpha: 0.95, gamma: 0.99, beta: 10, maxStiffnessRatio: 1e6,
  /** The strain limit's stabilization: 0, so an edge's target is its limit itself and a violation carried in from
   * x_t (OGC's truncation leaves one) is corrected in this step, not accepted at 95% (coordinator ruling 2026-11-09,
   * progress-lane-c1.md RESULT 11). Pins keep alpha. The warm start lambda <- alpha gamma lambda is unchanged. */
  edgeAlpha: 0 });
/** Per-vertex backtracking line search: Armijo constant and halvings. */
export const CLOTH_LINE_SEARCH = Object.freeze({ armijo: 1e-4, halvings: 8 });

export interface ClothMembraneConstants {
  readonly c11: number; readonly c22: number; readonly c12: number; readonly g: number;
  /** Compression softening, subtracted as (1 - ratio) * soft * min(E, 0)^2 / 2 per fibre. */
  readonly softWarp: number; readonly softWeft: number; readonly ratio: number;
}

/** Plane-stress orthotropic stiffness. nu is the weft-to-warp Poisson ratio:
 * C12 = nu warp / D with D = 1 - nu^2 warp / weft. The softening split
 * C - diag(softWarp, softWeft) is PSD with zero determinant, so every
 * compressed state keeps a positive definite law and C1 energy. */
export function clothMembraneConstants(m: ClothMaterial): ClothMembraneConstants {
  const D = 1 - m.poisson * m.poisson * m.stretchWarp_N_m / m.stretchWeft_N_m;
  if (!(D > 0)) throw new Error(`Cloth material ${m.id}: Poisson coupling must keep the membrane law positive definite`);
  const c11 = m.stretchWarp_N_m / D, c22 = m.stretchWeft_N_m / D, c12 = m.poisson * m.stretchWarp_N_m / D;
  return { c11, c22, c12, g: m.shear_N_m, ratio: m.compressionRatio,
    softWarp: c11 - Math.abs(c12) * Math.sqrt(c11 / c22), softWeft: c22 - Math.abs(c12) * Math.sqrt(c22 / c11) };
}

export type Vec3 = [number, number, number];
/** Row-major symmetric 3x3. */
export type Mat3 = [number, number, number, number, number, number, number, number, number];
export interface ClothVertexBlock { energy: number; gradient: Vec3; hessian: Mat3 }

const zero3 = (): Vec3 => [0, 0, 0];
const zero9 = (): Mat3 => [0, 0, 0, 0, 0, 0, 0, 0, 0];
const at3 = (p: ArrayLike<number>, i: number): Vec3 => [p[3 * i]!, p[3 * i + 1]!, p[3 * i + 2]!];
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
function addOuter(H: Mat3, s: number, u: Vec3, w: Vec3): void { for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) H[3 * r + c] += s * u[r]! * w[c]!; }
function addDiag(H: Mat3, s: number): void { H[0] += s; H[4] += s; H[8] += s; }

/** Cyclic Jacobi eigen-decomposition of a symmetric 3x3 (row-major).
 * Returns eigenvalues and column eigenvectors (V[3r + c] = v_c[r]). */
export function clothSymmetricEigen3(A: Mat3, sweeps = 32): { values: Vec3; vectors: Mat3 } {
  const a = A.slice() as Mat3, V: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  for (let s = 0; s < sweeps; s++) {
    const off = a[1] * a[1] + a[2] * a[2] + a[5] * a[5];
    if (off <= 1e-300 + 1e-30 * (a[0] * a[0] + a[4] * a[4] + a[8] * a[8])) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]] as const) {
      const apq = a[3 * p + q]!;
      if (apq === 0) continue;
      const theta = (a[3 * q + q]! - a[3 * p + p]!) / (2 * apq);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1)), c = 1 / Math.sqrt(t * t + 1), sn = t * c;
      for (let k = 0; k < 3; k++) { // A <- A J (columns p, q)
        const akp = a[3 * k + p]!, akq = a[3 * k + q]!;
        a[3 * k + p] = c * akp - sn * akq; a[3 * k + q] = sn * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) { // A <- J^T A (rows p, q)
        const apk = a[3 * p + k]!, aqk = a[3 * q + k]!;
        a[3 * p + k] = c * apk - sn * aqk; a[3 * q + k] = sn * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = V[3 * k + p]!, vkq = V[3 * k + q]!;
        V[3 * k + p] = c * vkp - sn * vkq; V[3 * k + q] = sn * vkp + c * vkq;
      }
    }
  }
  return { values: [a[0], a[4], a[8]], vectors: V };
}

/** The nearest PSD matrix (Frobenius): negative eigenvalues clamped to zero. */
export function clothProjectPSD(H: Mat3): Mat3 {
  const { values, vectors: V } = clothSymmetricEigen3(H), out = zero9();
  for (let c = 0; c < 3; c++) {
    const l = Math.max(values[c]!, 0); if (l === 0) continue;
    const v: Vec3 = [V[c]!, V[3 + c]!, V[6 + c]!];
    addOuter(out, l, v, v);
  }
  return out;
}

/** Strain energy density of the membrane (J/m^2) at Green strain (e11, e22) and engineering shear gamma. */
export function clothMembraneDensity(c: ClothMembraneConstants, e11: number, e22: number, gamma: number): number {
  const n11 = Math.min(e11, 0), n22 = Math.min(e22, 0);
  return 0.5 * (c.c11 * e11 * e11 + 2 * c.c12 * e11 * e22 + c.c22 * e22 * e22)
    - 0.5 * (1 - c.ratio) * (c.softWarp * n11 * n11 + c.softWeft * n22 * n22) + 0.5 * c.g * gamma * gamma;
}

/** Vertex weights (dF0/dx, dF1/dx) of local vertex k in a face with column-major Dm^-1 (00, 10, 01, 11). */
export function clothFaceWeights(dm: ArrayLike<number>, offset: number, k: number): [number, number] {
  const m00 = dm[offset]!, m10 = dm[offset + 1]!, m01 = dm[offset + 2]!, m11 = dm[offset + 3]!;
  return k === 0 ? [-m00 - m10, -m01 - m11] : k === 1 ? [m00, m01] : [m10, m11];
}

/** Membrane energy, gradient and (optionally PSD-projected) Hessian of one face with respect to its local vertex k. */
export function clothMembraneBlock(c: ClothMembraneConstants, dm: ArrayLike<number>, offset: number, area: number, k: number, x0: Vec3, x1: Vec3, x2: Vec3, project = true): ClothVertexBlock {
  const m00 = dm[offset]!, m10 = dm[offset + 1]!, m01 = dm[offset + 2]!, m11 = dm[offset + 3]!;
  const e1 = sub(x1, x0), e2 = sub(x2, x0);
  const f0: Vec3 = [e1[0] * m00 + e2[0] * m10, e1[1] * m00 + e2[1] * m10, e1[2] * m00 + e2[2] * m10];
  const f1: Vec3 = [e1[0] * m01 + e2[0] * m11, e1[1] * m01 + e2[1] * m11, e1[2] * m01 + e2[2] * m11];
  const e11 = 0.5 * (dot(f0, f0) - 1), e22 = 0.5 * (dot(f1, f1) - 1), gamma = dot(f0, f1);
  const w11 = e11 < 0 ? (1 - c.ratio) * c.softWarp : 0, w22 = e22 < 0 ? (1 - c.ratio) * c.softWeft : 0;
  const s11 = c.c11 * e11 + c.c12 * e22 - w11 * e11, s22 = c.c22 * e22 + c.c12 * e11 - w22 * e22, tau = c.g * gamma;
  const [a, b] = clothFaceWeights(dm, offset, k);
  const j1: Vec3 = [a * f0[0], a * f0[1], a * f0[2]], j2: Vec3 = [b * f1[0], b * f1[1], b * f1[2]];
  const j3: Vec3 = [a * f1[0] + b * f0[0], a * f1[1] + b * f0[1], a * f1[2] + b * f0[2]];
  const gradient: Vec3 = [0, 1, 2].map(r => area * (s11 * j1[r]! + s22 * j2[r]! + tau * j3[r]!)) as Vec3;
  const H = zero9(), d11 = c.c11 - w11, d22 = c.c22 - w22;
  addOuter(H, area * d11, j1, j1); addOuter(H, area * d22, j2, j2);
  addOuter(H, area * c.c12, j1, j2); addOuter(H, area * c.c12, j2, j1);
  addOuter(H, area * c.g, j3, j3);
  const sigma = s11 * a * a + s22 * b * b + 2 * tau * a * b;
  addDiag(H, area * sigma);
  // J D J^T is PSD (D is), so only a compressive sigma can make the block indefinite.
  return { energy: area * clothMembraneDensity(c, e11, e22, gamma), gradient, hessian: project && sigma < 0 ? clothProjectPSD(H) : H };
}

/** R_r x = sum_o w_o x_(v_o) of bending row r (a Kirchhoff plate row of its vertex, `clothBendingRows`),
 * taken relative to the row's own vertex: zero at the flat rest and under any affine map of it. */
export function clothBendVector(t: PreparedClothTopology, r: number, pos: (i: number) => Vec3): Vec3 {
  const s = zero3(), o0 = t.bendOffsets[r]!, x0 = pos(t.bendVertices[o0]!);
  for (let o = o0 + 1; o < t.bendOffsets[r + 1]!; o++) { const w = t.bendWeights[o]!, x = pos(t.bendVertices[o]!); for (let c = 0; c < 3; c++) s[c] += w * (x[c]! - x0[c]!); }
  return s;
}

/** Bending of the row holding entry o, at that entry's vertex: E = (k_r/2)|R_r x|^2, gradient k_r w_o L_r x, Hessian
 * k_r w_o^2 I (exact: E is quadratic in x). Adds into g and H; returns E. */
export function clothAddBending(t: PreparedClothTopology, o: number, pos: (i: number) => Vec3, g: Vec3, H: Mat3): number {
  const r = t.bendEntryRow[o]!, k = t.bendStiffness_N_m3[r]!, w = t.bendWeights[o]!, s = clothBendVector(t, r, pos);
  for (let c = 0; c < 3; c++) g[c] += k * w * s[c]!;
  addDiag(H, k * w * w);
  return 0.5 * k * dot(s, s);
}

/** Augmented-Lagrangian energy (1/2k)(clamp(kC + lambda)^2 - lambda^2) of a scalar
 * constraint; inequality constraints clamp the force at zero. */
export function clothAugmentedEnergy(k: number, lambda: number, C: number, inequality: boolean): number {
  const f = k * C + lambda, fc = inequality ? Math.max(f, 0) : f;
  return (fc * fc - lambda * lambda) / (2 * k);
}

/** Attachment target of an anchor under its group's scripts at time t (world metres). */
export function clothAttachmentTarget(scripts: readonly ClothAttachmentScript[], group: number, anchor: ClothVector, t: number): Vec3 {
  let p: Vec3 = [anchor[0], anchor[1], anchor[2]];
  for (const s of scripts) {
    if (s.group !== group) continue;
    const linear = Math.min(Math.max((t - s.start_s) / (s.end_s - s.start_s), 0), 1);
    const w = s.easing === 'cosine' ? 0.5 - 0.5 * Math.cos(Math.PI * linear) : linear;
    if (s.kind === 'translate') p = [p[0] + w * s.vector[0], p[1] + w * s.vector[1], p[2] + w * s.vector[2]];
    else if (s.kind === 'oscillate') {
      const phase = Math.sin(2 * Math.PI * (Math.min(Math.max(t, s.start_s), s.end_s) - s.start_s) / s.period_s!);
      p = [p[0] + phase * s.vector[0], p[1] + phase * s.vector[1], p[2] + phase * s.vector[2]];
    } else {
      const o = s.origin_m ?? [0, 0, 0], l = Math.hypot(...s.vector), u: Vec3 = [s.vector[0] / l, s.vector[1] / l, s.vector[2] / l];
      const angle = w * s.angle_rad!, c = Math.cos(angle), sn = Math.sin(angle), r = sub(p, [o[0], o[1], o[2]]);
      const ur = cross(u, r), ud = dot(u, r);
      p = [0, 1, 2].map(i => o[i]! + r[i]! * c + ur[i]! * sn + u[i]! * ud * (1 - c)) as Vec3;
    }
  }
  return p;
}

/** Setup-time data of the oracle and the GPU solver (both read it). */
export interface ClothSolverModel {
  readonly topology: PreparedClothTopology;
  readonly materials: readonly ClothMaterial[];
  readonly membrane: readonly ClothMembraneConstants[];
  /** Per vertex trace(H_elastic)/3 at the rest pose: the AVBD reference stiffness without inertia. */
  readonly elasticStiffness_N_m: Float64Array;
  /** Per vertex mean incident rest edge length: the AVBD length scale of pins. */
  readonly vertexLength_m: Float64Array;
  /** Per edge strain limit eps (0: unlimited). */
  readonly edgeStrainLimit: Float64Array;
  /** Per face damping coefficient (s). */
  readonly faceDamping_s: Float64Array;
  /** Per bending row damping coefficient: its vertex's incident faces' (s), area weighted. */
  readonly bendDamping_s: Float64Array;
  readonly scripts: readonly ClothAttachmentScript[];
}

export function buildClothSolverModel(topology: PreparedClothTopology, materials: readonly ClothMaterial[], scripts: readonly ClothAttachmentScript[] = []): ClothSolverModel {
  const t = topology, n = t.vertexCount, membrane = materials.map(clothMembraneConstants), rest = t.data.restPositions_m;
  const elastic = new Float64Array(n), length = new Float64Array(n), counts = new Float64Array(n);
  for (let f = 0; f < t.faceCount; f++) {
    const v = [t.faces[3 * f]!, t.faces[3 * f + 1]!, t.faces[3 * f + 2]!], X = v.map(i => at3(rest, i)) as [Vec3, Vec3, Vec3];
    for (let k = 0; k < 3; k++) {
      const H = clothMembraneBlock(membrane[t.faceMaterial[f]!]!, t.faceDmInv, 4 * f, t.faceArea_m2[f]!, k, ...X).hessian;
      elastic[v[k]!] += (H[0] + H[4] + H[8]) / 3;
    }
  }
  for (let o = 0; o < t.bendVertices.length; o++) elastic[t.bendVertices[o]!] += t.bendStiffness_N_m3[t.bendEntryRow[o]!]! * t.bendWeights[o]! ** 2;
  const limit = new Float64Array(t.edgeCount);
  for (let e = 0; e < t.edgeCount; e++) {
    for (const k of [0, 1]) { const v = t.edges[4 * e + k]!; length[v] += t.edgeRestLength_m[e]!; counts[v]++; }
    let eps = Infinity;
    for (const f of [t.edges[4 * e + 2]!, t.edges[4 * e + 3]!]) if (f !== CLOTH_NO_FACE) eps = Math.min(eps, materials[t.faceMaterial[f]!]!.strainLimit ?? Infinity);
    limit[e] = Number.isFinite(eps) ? eps : 0;
  }
  for (let v = 0; v < n; v++) length[v] /= counts[v]!;
  const faceDamping = Float64Array.from(t.faceMaterial, m => materials[m]!.damping_s);
  const bendDamping = new Float64Array(t.bendRowCount);
  for (let v = 0; v < n; v++) {
    let a = 0, b = 0;
    for (let j = t.vertexFaceOffsets[v]!; j < t.vertexFaceOffsets[v + 1]!; j++) { const f = t.vertexFaces[j]! >> 2; a += t.faceArea_m2[f]!; b += t.faceArea_m2[f]! * faceDamping[f]!; }
    bendDamping.fill(b / a, CLOTH_BEND_ROWS * v, CLOTH_BEND_ROWS * (v + 1));
  }
  return { topology, materials, membrane, elasticStiffness_N_m: elastic, vertexLength_m: length, edgeStrainLimit: limit,
    faceDamping_s: faceDamping, bendDamping_s: bendDamping, scripts };
}

export interface ClothOracleState {
  /** World positions and velocities, 3 per vertex. */
  x: Float64Array; v: Float64Array;
  /** Previous step's velocity and displacement (BDF2, adaptive init). */
  vPrev: Float64Array; dxPrev: Float64Array;
  /** Length of the previous step (0 before the first). */
  previousStep_s: number;
  time_s: number;
  /** (lambda xyz, k) per vertex; meaningful for pinned vertices. */
  pinDuals: Float64Array;
  /** (lambda, k) per canonical edge. */
  edgeDuals: Float64Array;
}

export function createClothOracleState(model: ClothSolverModel, x: ArrayLike<number>, v?: ArrayLike<number>): ClothOracleState {
  const n = model.topology.vertexCount;
  return { x: Float64Array.from(x), v: v ? Float64Array.from(v) : new Float64Array(3 * n), vPrev: new Float64Array(3 * n), dxPrev: new Float64Array(3 * n),
    previousStep_s: 0, time_s: 0, pinDuals: new Float64Array(4 * n), edgeDuals: new Float64Array(2 * model.topology.edgeCount) };
}

/**
 * Starting iterate of the step's K iterations: VBD's 'adaptive' guess (default, Chen et al. 2024 section 3.7), the
 * PBD-like 'inertial' target y, or 'previous' x^ (step start plus the BDF2 history term). 'previous' is stable for
 * any K (each sweep lowers the incremental potential from G(x^)) but lags free fall; the extrapolating guesses keep
 * free fall exact but grow when K cannot propagate support across a stiff sheet (progress-lane-c1.md, 2026-11-09).
 */
export type ClothInitialGuess = 'previous' | 'adaptive' | 'inertial';
export const CLOTH_INITIAL_GUESS: Record<ClothInitialGuess, number> = { previous: 0, adaptive: 1, inertial: 2 };

export interface ClothStepOptions {
  readonly dt_s: number;
  readonly gravity_m_s2: ClothVector;
  readonly integrator: ClothIntegrator;
  readonly iterations: number;
  /** Rounding unit of the line search's error bound (default: f64). Comparing with the f32 GPU solver passes 2^-23. */
  readonly lineSearchEpsilon?: number;
  /** Default 'adaptive'. */
  readonly initialGuess?: ClothInitialGuess;
}

/** The per-step quantities every vertex update reads. */
export interface ClothStepFrame {
  /** Effective step h~ (h, or 2h/3 for BDF2). */
  readonly hEff: number;
  readonly y: Float64Array;
  readonly xHat: Float64Array;
  /** Step-start positions x_t. */
  readonly x0: Float64Array;
  /** Pin targets at t and t + h, 3 per vertex (pinned vertices only). */
  readonly target0: Float64Array;
  readonly target1: Float64Array;
  /** The damping tangent frozen at x_t. */
  readonly damping: ClothDampingTangent;
}

const isPinned = (model: ClothSolverModel, v: number) => model.topology.groups[v]! > 0;

/** Reference stiffness s_i = m_i / h~^2 + elastic_i: AVBD k_start and its ramp scale. */
export function clothReferenceStiffness(model: ClothSolverModel, v: number, hEff: number): number {
  return model.topology.mass_kg[v]! / (hEff * hEff) + model.elasticStiffness_N_m[v]!;
}

export interface ClothLocalSystem { gradient: Vec3; hessian: Mat3; energy: number; magnitude: number }

/** Rayleigh (stiffness-proportional) damping as a dissipation potential frozen
 * at the step start: per element, (k_d / 2h~) A rate^T D_t rate with rate the
 * strain increment linearized at x_t over dx = x - x^ and D_t the material
 * tangent at x_t (membrane), and (k_d / 2h~) k_r |L_r dx|^2 (the
 * quadratic bending's own Hessian, no tangent). It is quadratic and convex in
 * x, zero for rigid motion, and the same objective for every iteration of the
 * step. */
export interface ClothDampingTangent {
  /** Per face: f0(x_t), f1(x_t) (3 each) and the tangent diagonal (D11, D22) at x_t. */
  readonly face: Float64Array;
}

export function clothDampingTangent(model: ClothSolverModel, x0: ArrayLike<number>): ClothDampingTangent {
  const t = model.topology, face = new Float64Array(8 * t.faceCount);
  for (let f = 0; f < t.faceCount; f++) {
    const c = model.membrane[t.faceMaterial[f]!]!, dm = t.faceDmInv, o = 4 * f;
    const p0 = at3(x0, t.faces[3 * f]!), e1 = sub(at3(x0, t.faces[3 * f + 1]!), p0), e2 = sub(at3(x0, t.faces[3 * f + 2]!), p0);
    const f0 = [0, 1, 2].map(r => e1[r]! * dm[o]! + e2[r]! * dm[o + 1]!), f1 = [0, 1, 2].map(r => e1[r]! * dm[o + 2]! + e2[r]! * dm[o + 3]!);
    const e11 = 0.5 * (dot(f0 as Vec3, f0 as Vec3) - 1), e22 = 0.5 * (dot(f1 as Vec3, f1 as Vec3) - 1);
    face.set([...f0, ...f1, c.c11 - (e11 < 0 ? (1 - c.ratio) * c.softWarp : 0), c.c22 - (e22 < 0 ? (1 - c.ratio) * c.softWeft : 0)], 8 * f);
  }
  return { face };
}

export function clothAddFaceDamping(model: ClothSolverModel, frame: ClothStepFrame, f: number, k: number, pos: (i: number) => Vec3, g: Vec3, H: Mat3): number {
  const t = model.topology, kd = model.faceDamping_s[f]!;
  if (!(kd > 0)) return 0;
  const c = model.membrane[t.faceMaterial[f]!]!, T = frame.damping.face, o = 8 * f, scale = kd * t.faceArea_m2[f]! / frame.hEff;
  const f0: Vec3 = [T[o]!, T[o + 1]!, T[o + 2]!], f1: Vec3 = [T[o + 3]!, T[o + 4]!, T[o + 5]!], d11 = T[o + 6]!, d22 = T[o + 7]!;
  const df0 = zero3(), df1 = zero3();
  for (let j = 0; j < 3; j++) {
    const v = t.faces[3 * f + j]!, [a, b] = clothFaceWeights(t.faceDmInv, 4 * f, j), u = sub(pos(v), at3(frame.xHat, v));
    for (let r = 0; r < 3; r++) { df0[r] += a * u[r]!; df1[r] += b * u[r]!; }
  }
  const rate: Vec3 = [dot(f0, df0), dot(f1, df1), dot(f0, df1) + dot(f1, df0)];
  const sigma: Vec3 = [d11 * rate[0] + c.c12 * rate[1], c.c12 * rate[0] + d22 * rate[1], c.g * rate[2]];
  const [a, b] = clothFaceWeights(t.faceDmInv, 4 * f, k);
  const j1 = f0.map(q => a * q) as Vec3, j2 = f1.map(q => b * q) as Vec3, j3 = [0, 1, 2].map(r => a * f1[r]! + b * f0[r]!) as Vec3;
  for (let r = 0; r < 3; r++) g[r] += scale * (sigma[0] * j1[r]! + sigma[1] * j2[r]! + sigma[2] * j3[r]!);
  addOuter(H, scale * d11, j1, j1); addOuter(H, scale * d22, j2, j2);
  addOuter(H, scale * c.c12, j1, j2); addOuter(H, scale * c.c12, j2, j1); addOuter(H, scale * c.g, j3, j3);
  return 0.5 * scale * dot(rate, sigma);
}

export function clothAddBendDamping(model: ClothSolverModel, frame: ClothStepFrame, o: number, pos: (i: number) => Vec3, g: Vec3, H: Mat3): number {
  const t = model.topology, r = t.bendEntryRow[o]!, kd = model.bendDamping_s[r]!;
  if (!(kd > 0)) return 0;
  const scale = kd * t.bendStiffness_N_m3[r]! / frame.hEff, w = t.bendWeights[o]!;
  const u = clothBendVector(t, r, i => sub(pos(i), at3(frame.xHat, i)));
  for (let c = 0; c < 3; c++) g[c] += scale * w * u[c]!;
  addDiag(H, scale * w * w);
  return 0.5 * scale * dot(u, u);
}

/** Gradient, PSD Hessian and energy of vertex v's local objective at position xv, the other vertices at `x`. */
export function clothLocalSystem(model: ClothSolverModel, frame: ClothStepFrame, state: ClothOracleState, v: number, xv: Vec3): ClothLocalSystem {
  const t = model.topology, x = state.x, m = t.mass_kg[v]!, h = frame.hEff, w = m / (h * h);
  const pos = (i: number): Vec3 => i === v ? xv : at3(x, i);
  const g = zero3(), H = zero9();
  let energy = 0, magnitude = 0;
  const y = at3(frame.y, v), r = sub(xv, y);
  for (let k = 0; k < 3; k++) g[k] += w * r[k]!;
  addDiag(H, w); energy += 0.5 * w * dot(r, r); magnitude += Math.abs(0.5 * w * dot(r, r));
  for (let j = t.vertexFaceOffsets[v]!; j < t.vertexFaceOffsets[v + 1]!; j++) {
    const f = t.vertexFaces[j]! >> 2, k = t.vertexFaces[j]! & 3;
    const b = clothMembraneBlock(model.membrane[t.faceMaterial[f]!]!, t.faceDmInv, 4 * f, t.faceArea_m2[f]!, k, pos(t.faces[3 * f]!), pos(t.faces[3 * f + 1]!), pos(t.faces[3 * f + 2]!));
    for (let c = 0; c < 3; c++) g[c] += b.gradient[c]!;
    for (let c = 0; c < 9; c++) H[c] += b.hessian[c]!;
    const ed = clothAddFaceDamping(model, frame, f, k, pos, g, H);
    energy += b.energy + ed; magnitude += Math.abs(b.energy) + ed;
  }
  for (let j = t.vertexBendOffsets[v]!; j < t.vertexBendOffsets[v + 1]!; j++) {
    const o = t.vertexBends[j]!, eb = clothAddBending(t, o, pos, g, H), ed = clothAddBendDamping(model, frame, o, pos, g, H);
    energy += eb + ed; magnitude += eb + ed;
  }
  if (isPinned(model, v)) {
    const k = state.pinDuals[4 * v + 3]!, x0 = at3(frame.x0, v), p0 = at3(frame.target0, v), p1 = at3(frame.target1, v);
    for (let c = 0; c < 3; c++) {
      const C = xv[c]! - p1[c]! - CLOTH_AVBD.alpha * (x0[c]! - p0[c]!), lambda = state.pinDuals[4 * v + c]!;
      g[c] += k * C + lambda;
      const e = clothAugmentedEnergy(k, lambda, C, false); energy += e; magnitude += Math.abs(e);
    }
    addDiag(H, k);
  }
  for (let j = t.vertexEdgeOffsets[v]!; j < t.vertexEdgeOffsets[v + 1]!; j++) {
    const e = t.vertexEdges[j]! >> 2, eps = model.edgeStrainLimit[e]!;
    if (!(eps > 0)) continue;
    const a = t.edges[4 * e]!, b = t.edges[4 * e + 1]!, other = a === v ? b : a, L = t.edgeRestLength_m[e]!;
    const d = sub(xv, pos(other)), len = Math.sqrt(dot(d, d));
    if (!(len > 0)) continue;
    const d0 = sub(at3(frame.x0, a), at3(frame.x0, b)), C0 = Math.sqrt(dot(d0, d0)) - (1 + eps) * L;
    const C = len - (1 + eps) * L - CLOTH_AVBD.edgeAlpha * Math.max(C0, 0), lambda = state.edgeDuals[2 * e]!, k = state.edgeDuals[2 * e + 1]!;
    const n = d.map(q => q / len) as Vec3, force = Math.max(k * C + lambda, 0);
    for (let c = 0; c < 3; c++) g[c] += force * n[c]!;
    // AVBD (14): a force clamped at zero keeps the rescaled stiffness (0 - lambda) / C.
    const kEff = k * C + lambda > 0 ? k : C !== 0 ? Math.max(-lambda / C, 0) : 0;
    addOuter(H, kEff - force / len, n, n); addDiag(H, force / len);
    const en = clothAugmentedEnergy(k, lambda, C, true); energy += en; magnitude += Math.abs(en);
  }
  return { gradient: g, hessian: H, energy, magnitude };
}

function solve3(H: Mat3, b: Vec3): Vec3 | undefined {
  const c00 = H[4] * H[8] - H[5] * H[7], c01 = H[5] * H[6] - H[3] * H[8], c02 = H[3] * H[7] - H[4] * H[6];
  const det = H[0] * c00 + H[1] * c01 + H[2] * c02;
  const scale = Math.abs(H[0]) + Math.abs(H[4]) + Math.abs(H[8]);
  if (!(Math.abs(det) > 1e-30 * scale * scale * scale)) return undefined;
  const inv = [c00, H[2] * H[7] - H[1] * H[8], H[1] * H[5] - H[2] * H[4], c01, H[0] * H[8] - H[2] * H[6], H[2] * H[3] - H[0] * H[5], c02, H[1] * H[6] - H[0] * H[7], H[0] * H[4] - H[1] * H[3]];
  return [0, 1, 2].map(r => (inv[3 * r]! * b[0] + inv[3 * r + 1]! * b[1] + inv[3 * r + 2]! * b[2]) / det) as Vec3;
}

export interface ClothVertexUpdate { position: Vec3; step: number; energyBefore: number; energyAfter: number; gradient: Vec3 }

/** One VBD vertex block update with the per-vertex backtracking line search.
 * The acceptance test is Armijo on the vertex's own objective, with the
 * floating-point error bound of the two sums (4 eps sum|terms|) as slack. */
export function clothVertexUpdate(model: ClothSolverModel, frame: ClothStepFrame, state: ClothOracleState, v: number, epsilon = Number.EPSILON): ClothVertexUpdate {
  const xv = at3(state.x, v), local = clothLocalSystem(model, frame, state, v, xv);
  const delta = solve3(local.hessian, local.gradient.map(q => -q) as Vec3);
  if (!delta) return { position: xv, step: 0, energyBefore: local.energy, energyAfter: local.energy, gradient: local.gradient };
  const slope = dot(local.gradient, delta);
  let alpha = 1;
  for (let i = 0; i <= CLOTH_LINE_SEARCH.halvings; i++, alpha *= 0.5) {
    const trial: Vec3 = [xv[0] + alpha * delta[0], xv[1] + alpha * delta[1], xv[2] + alpha * delta[2]];
    const next = clothLocalSystem(model, frame, state, v, trial);
    if (next.energy <= local.energy + CLOTH_LINE_SEARCH.armijo * alpha * slope + 4 * epsilon * (local.magnitude + next.magnitude))
      return { position: trial, step: alpha, energyBefore: local.energy, energyAfter: next.energy, gradient: local.gradient };
  }
  return { position: xv, step: 0, energyBefore: local.energy, energyAfter: local.energy, gradient: local.gradient };
}

/** Whether this step runs BDF2: the previous step must exist and have the same length. */
export function clothStepUsesBDF2(integrator: ClothIntegrator, previousStep_s: number, dt_s: number): boolean {
  return integrator === 'bdf2' && previousStep_s > 0 && Math.abs(previousStep_s - dt_s) <= 1e-12 * dt_s;
}

/** Adaptive initialization's gravity factor (Chen et al. 2024, section 3.7). */
export function clothAdaptiveGravity(a: Vec3, g: ClothVector): number {
  const gl = Math.hypot(g[0], g[1], g[2]);
  if (!(gl > 0)) return 0;
  const along = (a[0] * g[0] + a[1] * g[1] + a[2] * g[2]) / gl;
  return along > gl ? 1 : along < 0 ? 0 : along / gl;
}

export function clothStepFrame(model: ClothSolverModel, state: ClothOracleState, options: ClothStepOptions): { frame: ClothStepFrame; init: Float64Array; bdf2: boolean } {
  const n = model.topology.vertexCount, h = options.dt_s, bdf2 = clothStepUsesBDF2(options.integrator, state.previousStep_s, h);
  const hEff = bdf2 ? 2 * h / 3 : h, g = options.gravity_m_s2;
  const y = new Float64Array(3 * n), xHat = new Float64Array(3 * n), init = new Float64Array(3 * n);
  const target0 = new Float64Array(3 * n), target1 = new Float64Array(3 * n), rest = model.topology.data.restPositions_m;
  for (let v = 0; v < n; v++) {
    const vel = at3(state.v, v), vp = at3(state.vPrev, v);
    const a: Vec3 = state.previousStep_s > 0 ? [0, 1, 2].map(c => (vel[c]! - vp[c]!) / state.previousStep_s) as Vec3 : [g[0], g[1], g[2]];
    const coef = state.previousStep_s > 0 ? clothAdaptiveGravity(a, g) : 1;
    for (let c = 0; c < 3; c++) {
      const xh = state.x[3 * v + c]! + (bdf2 ? state.dxPrev[3 * v + c]! / 3 : 0);
      const vh = bdf2 ? (4 * vel[c]! - vp[c]!) / 3 : vel[c]!;
      xHat[3 * v + c] = xh; y[3 * v + c] = xh + hEff * vh + hEff * hEff * g[c]!;
      const guess = options.initialGuess ?? 'adaptive';
      init[3 * v + c] = guess === 'previous' ? xh : guess === 'inertial' ? y[3 * v + c]! : xh + hEff * vh + hEff * hEff * coef * g[c]!;
    }
    const group = model.topology.groups[v]!;
    if (group > 0) {
      const anchor = at3(rest, v), p0 = clothAttachmentTarget(model.scripts, group, anchor, state.time_s), p1 = clothAttachmentTarget(model.scripts, group, anchor, state.time_s + h);
      for (let c = 0; c < 3; c++) { target0[3 * v + c] = p0[c]!; target1[3 * v + c] = p1[c]!; init[3 * v + c] = state.x[3 * v + c]! + p1[c]! - p0[c]!; y[3 * v + c] = init[3 * v + c]!; }
    }
  }
  return { frame: { hEff, y, xHat, x0: Float64Array.from(state.x), target0, target1, damping: clothDampingTangent(model, state.x) }, init, bdf2 };
}

/** Warm start (AVBD 19): k = max(gamma k, k_start), lambda = alpha gamma lambda. */
export function clothWarmStartDuals(model: ClothSolverModel, state: ClothOracleState, hEff: number): void {
  const t = model.topology, { alpha, gamma } = CLOTH_AVBD;
  for (let v = 0; v < t.vertexCount; v++) {
    if (!isPinned(model, v)) continue;
    const s = clothReferenceStiffness(model, v, hEff);
    for (let c = 0; c < 3; c++) state.pinDuals[4 * v + c] *= alpha * gamma;
    state.pinDuals[4 * v + 3] = Math.max(gamma * state.pinDuals[4 * v + 3]!, s);
  }
  for (let e = 0; e < t.edgeCount; e++) {
    if (!(model.edgeStrainLimit[e]! > 0)) continue;
    const s = Math.max(clothReferenceStiffness(model, t.edges[4 * e]!, hEff), clothReferenceStiffness(model, t.edges[4 * e + 1]!, hEff));
    state.edgeDuals[2 * e] *= alpha * gamma;
    state.edgeDuals[2 * e + 1] = Math.max(gamma * state.edgeDuals[2 * e + 1]!, s);
  }
}

/** AVBD dual and stiffness update after an iteration (11, 12, with clamps). */
export function clothUpdateDuals(model: ClothSolverModel, frame: ClothStepFrame, state: ClothOracleState): void {
  const t = model.topology, { alpha, beta, maxStiffnessRatio } = CLOTH_AVBD;
  for (let v = 0; v < t.vertexCount; v++) {
    if (!isPinned(model, v)) continue;
    const s = clothReferenceStiffness(model, v, frame.hEff), k = state.pinDuals[4 * v + 3]!, kMax = maxStiffnessRatio * s, l = model.vertexLength_m[v]!;
    let c2 = 0;
    for (let c = 0; c < 3; c++) {
      const C = state.x[3 * v + c]! - frame.target1[3 * v + c]! - alpha * (frame.x0[3 * v + c]! - frame.target0[3 * v + c]!);
      state.pinDuals[4 * v + c] = Math.min(Math.max(k * C + state.pinDuals[4 * v + c]!, -kMax * l), kMax * l); c2 += C * C;
    }
    state.pinDuals[4 * v + 3] = Math.min(k + beta * s * Math.sqrt(c2) / l, kMax);
  }
  for (let e = 0; e < t.edgeCount; e++) {
    const eps = model.edgeStrainLimit[e]!;
    if (!(eps > 0)) continue;
    const a = t.edges[4 * e]!, b = t.edges[4 * e + 1]!, L = t.edgeRestLength_m[e]!;
    const s = Math.max(clothReferenceStiffness(model, a, frame.hEff), clothReferenceStiffness(model, b, frame.hEff)), kMax = maxStiffnessRatio * s;
    const d = sub(at3(state.x, a), at3(state.x, b)), d0 = sub(at3(frame.x0, a), at3(frame.x0, b));
    const C = Math.sqrt(dot(d, d)) - (1 + eps) * L - CLOTH_AVBD.edgeAlpha * Math.max(Math.sqrt(dot(d0, d0)) - (1 + eps) * L, 0);
    const k = state.edgeDuals[2 * e + 1]!, lambda = k * C + state.edgeDuals[2 * e]!;
    state.edgeDuals[2 * e] = Math.min(Math.max(lambda, 0), kMax * L);
    if (lambda > 0) state.edgeDuals[2 * e + 1] = Math.min(k + beta * s * Math.abs(C) / L, kMax);
  }
}

export interface ClothStepReport {
  readonly bdf2: boolean;
  /** Relative force residual |f| / |m g| at the end of each iteration (when tracked). */
  readonly residuals: number[];
  readonly maxStrain: number;
}

/** Relative residual of the step's variational problem: |grad| / |m g| over free and pinned vertices. */
export function clothStepResidual(model: ClothSolverModel, frame: ClothStepFrame, state: ClothOracleState, gravity: ClothVector): number {
  let r2 = 0, w2 = 0;
  const gg = gravity[0] ** 2 + gravity[1] ** 2 + gravity[2] ** 2;
  for (let v = 0; v < model.topology.vertexCount; v++) {
    const local = clothLocalSystem(model, frame, state, v, at3(state.x, v));
    r2 += dot(local.gradient, local.gradient); w2 += model.topology.mass_kg[v]! ** 2 * gg;
  }
  return Math.sqrt(r2 / Math.max(w2, 1e-300));
}

export function clothMaxEdgeStrain(model: ClothSolverModel, x: ArrayLike<number>): number {
  const t = model.topology;
  let worst = -Infinity;
  for (let e = 0; e < t.edgeCount; e++) {
    const d = sub(at3(x, t.edges[4 * e]!), at3(x, t.edges[4 * e + 1]!));
    worst = Math.max(worst, Math.sqrt(dot(d, d)) / t.edgeRestLength_m[e]! - 1);
  }
  return worst;
}

/** Two-level correction (oracle-coarse.ts): set up once per step after the warm start at the initial guess,
 * then applied before each iteration's sweep. */
export interface ClothOracleCoarse {
  begin(frame: ClothStepFrame, state: ClothOracleState): void;
  correct(frame: ClothStepFrame, state: ClothOracleState): void;
}

/** One implicit step of length dt: no substeps; K Gauss-Seidel sweeps in vertex order, each followed by the AVBD dual
 * updates, with an optional two-level correction before each sweep. */
export function clothOracleStep(model: ClothSolverModel, state: ClothOracleState, options: ClothStepOptions, track = false, coarse?: ClothOracleCoarse): ClothStepReport {
  const n = model.topology.vertexCount, { frame, init, bdf2 } = clothStepFrame(model, state, options), residuals: number[] = [];
  clothWarmStartDuals(model, state, frame.hEff);
  state.x.set(init);
  coarse?.begin(frame, state);
  for (let it = 1; it <= options.iterations; it++) {
    coarse?.correct(frame, state);
    for (let v = 0; v < n; v++) state.x.set(clothVertexUpdate(model, frame, state, v, options.lineSearchEpsilon).position, 3 * v);
    clothUpdateDuals(model, frame, state);
    if (track) residuals.push(clothStepResidual(model, frame, state, options.gravity_m_s2));
  }
  for (let i = 0; i < 3 * n; i++) {
    const vNew = (state.x[i]! - frame.xHat[i]!) / frame.hEff;
    state.vPrev[i] = state.v[i]!; state.v[i] = vNew; state.dxPrev[i] = state.x[i]! - frame.x0[i]!;
  }
  state.previousStep_s = options.dt_s; state.time_s += options.dt_s;
  return { bdf2, residuals, maxStrain: clothMaxEdgeStrain(model, state.x) };
}

/** Kinetic + gravitational + elastic energy (J). */
export function clothMechanicalEnergy(model: ClothSolverModel, x: ArrayLike<number>, v: ArrayLike<number>, gravity: ClothVector): { kinetic: number; potential: number; elastic: number; total: number } {
  const t = model.topology;
  let kinetic = 0, potential = 0, elastic = 0;
  for (let i = 0; i < t.vertexCount; i++) {
    const m = t.mass_kg[i]!, u = at3(v, i), p = at3(x, i);
    kinetic += 0.5 * m * dot(u, u); potential -= m * (gravity[0] * p[0] + gravity[1] * p[1] + gravity[2] * p[2]);
  }
  for (let f = 0; f < t.faceCount; f++)
    elastic += clothMembraneBlock(model.membrane[t.faceMaterial[f]!]!, t.faceDmInv, 4 * f, t.faceArea_m2[f]!, 0, at3(x, t.faces[3 * f]!), at3(x, t.faces[3 * f + 1]!), at3(x, t.faces[3 * f + 2]!), false).energy;
  for (let r = 0; r < t.bendRowCount; r++) { const q = clothBendVector(t, r, i => at3(x, i)); elastic += 0.5 * t.bendStiffness_N_m3[r]! * dot(q, q); }
  return { kinetic, potential, elastic, total: kinetic + potential + elastic };
}
