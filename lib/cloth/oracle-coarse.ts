/**
 * The Float64 oracle's two-level correction (test reference only; the GPU solver is Newton-PCG and has no coarse
 * path since M4): a Galerkin coarse space built from the mesh topology, which carries a Gauss-Seidel oracle's
 * support across a stiff sheet so that `clothOracleStep` converges in a few hundred sweeps.
 *
 * Each iteration first takes one global step in the span of a few smooth shape functions: d = P (P^T H P)^-1
 * (-P^T grad G), with a line search on G. P is built in rest space from the topology only: seeds are farthest points
 * by rest geodesic distance (Dijkstra over rest edge lengths), one seed per connected component first; each seed's
 * shape function is a geodesic tent max(0, 1 - d / R), truncated to the CLOTH_COARSE.maxEntries strongest per vertex
 * and normalised to a partition of unity. A_c = P^T H P is assembled once per step at the initial guess from PSD
 * element Hessians (membrane Gauss-Newton plus the PSD-clamped stress term, the frozen damping, pins, strain limit
 * edges, inertia). It only accelerates the oracle: the converged step is the minimizer of G either way.
 */
import type { PreparedClothTopology } from './topology';
import { clothAddBendDamping, clothAddBending, clothAddFaceDamping, clothAugmentedEnergy, clothFaceWeights, clothLocalSystem,
  clothMembraneBlock, CLOTH_AVBD, CLOTH_LINE_SEARCH, type ClothOracleCoarse, type ClothOracleState, type ClothSolverModel, type ClothStepFrame,
  type Mat3, type Vec3 } from './oracle';

export const CLOTH_COARSE = Object.freeze({
  /** Coarse nodes per document (3 unknowns each). */
  nodes: 40,
  /** Prolongation entries kept per vertex. */
  maxEntries: 4,
  /** Tent radius R over the farthest-point covering radius. */
  radius: 1.5,
  /** Line search trials of the coarse step: 1, 1/2, ... */
  trials: 4,
  /** Iterations between corrections: before iterations 1, 1 + every, ... of a step. */
  every: 2 as number,
  /** Whether A_c holds the bending rows. Bending of the smooth coarse modes is O(k_b / h_c^2), far below their membrane
   * and inertia terms, so the coarse operator leaves it out (the line search still sees all of G). */
  bending: false as boolean,
});

export interface ClothCoarseSpace {
  readonly nodeCount: number;
  /** Fine vertex of each node. */
  readonly seeds: Uint32Array;
  /** Connected component of each vertex. */
  readonly component: Uint32Array;
  readonly componentCount: number;
  /** Prolongation rows (CSR over vertices): node and weight; each row sums to one. */
  readonly rowOffsets: Uint32Array;
  readonly rowNodes: Uint32Array;
  readonly rowWeights: Float64Array;
  /** Largest rest geodesic distance from a vertex to its nearest seed. */
  readonly coverRadius_m: number;
  readonly tentRadius_m: number;
}

class MinHeap {
  private keys: number[] = []; private values: number[] = [];
  get size(): number { return this.keys.length; }
  push(key: number, value: number): void {
    const k = this.keys, v = this.values; let i = k.length; k.push(key); v.push(value);
    while (i > 0) { const p = (i - 1) >> 1; if (k[p]! <= key) break; k[i] = k[p]!; v[i] = v[p]!; i = p; }
    k[i] = key; v[i] = value;
  }
  pop(): [number, number] {
    const k = this.keys, v = this.values, top: [number, number] = [k[0]!, v[0]!], key = k.pop()!, value = v.pop()!;
    if (k.length) {
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1; let m = i, mk = key;
        if (l < k.length && k[l]! < mk) { m = l; mk = k[l]!; }
        if (r < k.length && k[r]! < mk) { m = r; }
        if (m === i) break;
        k[i] = k[m]!; v[i] = v[m]!; i = m;
      }
      k[i] = key; v[i] = value;
    }
    return top;
  }
}

/** Dijkstra from `source` over rest edges, writing d into `dist` wherever it improves (and below `limit`); visits each improved vertex. */
function relax(t: PreparedClothTopology, source: number, dist: Float64Array, limit: number, visit?: (v: number, d: number) => void): void {
  const heap = new MinHeap();
  dist[source] = 0; heap.push(0, source);
  while (heap.size) {
    const [d, v] = heap.pop();
    if (d > dist[v]!) continue;
    visit?.(v, d);
    for (let j = t.vertexEdgeOffsets[v]!; j < t.vertexEdgeOffsets[v + 1]!; j++) {
      const e = t.vertexEdges[j]! >> 2, a = t.edges[4 * e]!, w = a === v ? t.edges[4 * e + 1]! : a, dw = d + t.edgeRestLength_m[e]!;
      if (dw < dist[w]! && dw < limit) { dist[w] = dw; heap.push(dw, w); }
    }
  }
}

export function buildClothCoarseSpace(t: PreparedClothTopology, nodes: number = CLOTH_COARSE.nodes, radius: number = CLOTH_COARSE.radius, maxEntries: number = CLOTH_COARSE.maxEntries): ClothCoarseSpace {
  const n = t.vertexCount;
  if (!(Number.isInteger(nodes) && nodes >= 1)) throw new Error('Cloth coarse space needs at least one node');
  if (!(radius > 1)) throw new Error('Cloth coarse tent radius must exceed the covering radius');
  const component = new Uint32Array(n).fill(0xffffffff);
  let componentCount = 0;
  for (let s = 0; s < n; s++) {
    if (component[s] !== 0xffffffff) continue;
    const stack = [s]; component[s] = componentCount;
    while (stack.length) {
      const v = stack.pop()!;
      for (let j = t.vertexEdgeOffsets[v]!; j < t.vertexEdgeOffsets[v + 1]!; j++) {
        const e = t.vertexEdges[j]! >> 2, a = t.edges[4 * e]!, w = a === v ? t.edges[4 * e + 1]! : a;
        if (component[w] === 0xffffffff) { component[w] = componentCount; stack.push(w); }
      }
    }
    componentCount++;
  }
  if (componentCount > nodes)
    throw new Error(`Cloth document has ${componentCount} connected pieces; its coarse space holds ${nodes} nodes`);
  // One seed per component: the vertex farthest from the component's first vertex.
  const seeds: number[] = [], dist = new Float64Array(n).fill(Infinity), probe = new Float64Array(n);
  for (let c = 0, s = 0; c < componentCount; c++) {
    while (component[s] !== c) s++;
    probe.fill(Infinity); let far = s, farD = -1;
    relax(t, s, probe, Infinity, (v, d) => { if (d > farD) { farD = d; far = v; } });
    seeds.push(far); relax(t, far, dist, Infinity);
  }
  while (seeds.length < Math.min(nodes, n)) {
    let far = -1, farD = 0;
    for (let v = 0; v < n; v++) if (dist[v]! > farD) { farD = dist[v]!; far = v; }
    if (far < 0) break;
    seeds.push(far); relax(t, far, dist, Infinity);
  }
  let cover = 0;
  for (let v = 0; v < n; v++) cover = Math.max(cover, dist[v]!);
  const R = radius * cover, entries: [number, number][][] = Array.from({ length: n }, () => []);
  seeds.forEach((s, c) => {
    probe.fill(Infinity);
    relax(t, s, probe, R > 0 ? R : Number.MIN_VALUE, (v, d) => entries[v]!.push([c, R > 0 ? 1 - d / R : 1]));
  });
  const rowOffsets = new Uint32Array(n + 1), nodesOut: number[] = [], weights: number[] = [];
  for (let v = 0; v < n; v++) {
    const row = entries[v]!.filter(q => q[1] > 0).sort((a, b) => b[1] - a[1] || a[0] - b[0]).slice(0, maxEntries);
    if (!row.length) throw new Error(`Cloth coarse space left vertex ${v} uncovered`);
    const sum = row.reduce((s, q) => s + q[1], 0);
    for (const [c, w] of row.sort((a, b) => a[0] - b[0])) { nodesOut.push(c); weights.push(w / sum); }
    rowOffsets[v + 1] = nodesOut.length;
  }
  return { nodeCount: seeds.length, seeds: Uint32Array.from(seeds), component, componentCount, rowOffsets,
    rowNodes: Uint32Array.from(nodesOut), rowWeights: Float64Array.from(weights), coverRadius_m: cover, tentRadius_m: R };
}

const at3 = (p: ArrayLike<number>, i: number): Vec3 => [p[3 * i]!, p[3 * i + 1]!, p[3 * i + 2]!];
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** PSD part of the symmetric 2 x 2 [[a, b], [b, c]]. */
export function clothClampPSD2(a: number, b: number, c: number): [number, number, number] {
  const m = 0.5 * (a + c), r = Math.hypot(0.5 * (a - c), b), l1 = m + r, l2 = m - r;
  if (l2 >= 0) return [a, b, c];
  if (l1 <= 0) return [0, 0, 0];
  let u0 = b, u1 = l1 - a;
  if (Math.hypot(u0, u1) < Math.hypot(l1 - c, b)) { u0 = l1 - c; u1 = b; }
  const q = Math.hypot(u0, u1);
  if (!(q > 0)) return [Math.max(a, 0), 0, Math.max(c, 0)];
  u0 /= q; u1 /= q;
  return [l1 * u0 * u0, l1 * u0 * u1, l1 * u1 * u1];
}

/** Adds sum_ij P_i^T H_ij P_j of one element (corners `vs`, block function H(i, j) row-major 3 x 3) into the dense A. */
function scatter(space: ClothCoarseSpace, A: Float64Array, vs: readonly number[], block: (i: number, j: number) => Mat3 | undefined): void {
  const m = 3 * space.nodeCount;
  for (let i = 0; i < vs.length; i++) for (let j = 0; j < vs.length; j++) {
    const H = block(i, j);
    if (!H) continue;
    for (let p = space.rowOffsets[vs[i]!]!; p < space.rowOffsets[vs[i]! + 1]!; p++) {
      const c = space.rowNodes[p]!, wc = space.rowWeights[p]!;
      for (let q = space.rowOffsets[vs[j]!]!; q < space.rowOffsets[vs[j]! + 1]!; q++) {
        const d = space.rowNodes[q]!, w = wc * space.rowWeights[q]!;
        for (let r = 0; r < 3; r++) for (let s = 0; s < 3; s++) A[(3 * c + r) * m + 3 * d + s] += w * H[3 * r + s]!;
      }
    }
  }
}

const outer = (s: number, u: Vec3, w: Vec3): Mat3 => [0, 1, 2].flatMap(r => [0, 1, 2].map(c => s * u[r]! * w[c]!)) as Mat3;

/** Per corner Jacobian vectors (j1, j2, j3) of the face's (F0, F1) with weights from Dm^-1. */
function faceJacobians(dm: ArrayLike<number>, o: number, f0: Vec3, f1: Vec3): { w: [number, number][]; j: [Vec3, Vec3, Vec3][] } {
  const w = [0, 1, 2].map(k => clothFaceWeights(dm, o, k));
  return { w, j: w.map(([a, b]) => [f0.map(q => a * q) as Vec3, f1.map(q => b * q) as Vec3, [0, 1, 2].map(r => a * f1[r]! + b * f0[r]!) as Vec3]) };
}

/** J_i D J_j^T + g j3_i j3_j^T with D = [[d11, d12], [d12, d22]]. */
function tangentBlock(ji: [Vec3, Vec3, Vec3], jj: [Vec3, Vec3, Vec3], d11: number, d12: number, d22: number, g: number): Mat3 {
  const H = [0, 0, 0, 0, 0, 0, 0, 0, 0] as Mat3;
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++)
    H[3 * r + c] = d11 * ji[0][r]! * jj[0][c]! + d22 * ji[1][r]! * jj[1][c]! + d12 * (ji[0][r]! * jj[1][c]! + ji[1][r]! * jj[0][c]!) + g * ji[2][r]! * jj[2][c]!;
  return H;
}

/** Dense A_c = P^T H P (3 nodeCount square, row-major) at state.x with PSD element Hessians. */
export function clothCoarseOperator(model: ClothSolverModel, frame: ClothStepFrame, state: ClothOracleState, space: ClothCoarseSpace, bending = CLOTH_COARSE.bending): Float64Array {
  const t = model.topology, x = state.x, m = 3 * space.nodeCount, A = new Float64Array(m * m);
  const I = (s: number): Mat3 => [s, 0, 0, 0, s, 0, 0, 0, s];
  for (let v = 0; v < t.vertexCount; v++) {
    let s = t.mass_kg[v]! / (frame.hEff * frame.hEff);
    if (t.groups[v]! > 0) s += state.pinDuals[4 * v + 3]!;
    scatter(space, A, [v], () => I(s));
  }
  for (let f = 0; f < t.faceCount; f++) {
    const c = model.membrane[t.faceMaterial[f]!]!, vs = [0, 1, 2].map(k => t.faces[3 * f + k]!), o = 4 * f, dm = t.faceDmInv, area = t.faceArea_m2[f]!;
    const p0 = at3(x, vs[0]!), e1 = [0, 1, 2].map(r => x[3 * vs[1]! + r]! - p0[r]!), e2 = [0, 1, 2].map(r => x[3 * vs[2]! + r]! - p0[r]!);
    const f0 = [0, 1, 2].map(r => e1[r]! * dm[o]! + e2[r]! * dm[o + 1]!) as Vec3, f1 = [0, 1, 2].map(r => e1[r]! * dm[o + 2]! + e2[r]! * dm[o + 3]!) as Vec3;
    const e11 = 0.5 * (dot(f0, f0) - 1), e22 = 0.5 * (dot(f1, f1) - 1), gamma = dot(f0, f1);
    const w11 = e11 < 0 ? (1 - c.ratio) * c.softWarp : 0, w22 = e22 < 0 ? (1 - c.ratio) * c.softWeft : 0;
    const [s11, s12, s22] = clothClampPSD2(c.c11 * e11 + c.c12 * e22 - w11 * e11, c.g * gamma, c.c22 * e22 + c.c12 * e11 - w22 * e22);
    const J = faceJacobians(dm, o, f0, f1);
    scatter(space, A, vs, (i, j) => {
      const H = tangentBlock(J.j[i]!, J.j[j]!, c.c11 - w11, c.c12, c.c22 - w22, c.g), [ai, bi] = J.w[i]!, [aj, bj] = J.w[j]!;
      const sigma = s11 * ai * aj + s22 * bi * bj + s12 * (ai * bj + bi * aj);
      for (let k = 0; k < 9; k++) H[k] = area * (H[k]! + (k % 4 === 0 ? sigma : 0));
      return H;
    });
    const kd = model.faceDamping_s[f]!;
    if (kd > 0) {
      const T = frame.damping.face, q = 8 * f, scale = kd * area / frame.hEff;
      const D = faceJacobians(dm, o, [T[q]!, T[q + 1]!, T[q + 2]!], [T[q + 3]!, T[q + 4]!, T[q + 5]!]);
      scatter(space, A, vs, (i, j) => tangentBlock(D.j[i]!, D.j[j]!, T[q + 6]!, c.c12, T[q + 7]!, c.g).map(h => scale * h) as Mat3);
    }
  }
  if (bending) for (let r = 0; r < t.bendRowCount; r++) {
    // The exact Hessian of (k/2)|R x|^2 plus its damping: (k + kd k / h~) w_i w_j I.
    const o0 = t.bendOffsets[r]!, vs = Array.from(t.bendVertices.subarray(o0, t.bendOffsets[r + 1]!)), k = t.bendStiffness_N_m3[r]! * (1 + model.bendDamping_s[r]! / frame.hEff), w = t.bendWeights;
    scatter(space, A, vs, (i, j) => I(k * w[o0 + i]! * w[o0 + j]!));
  }
  for (let e = 0; e < t.edgeCount; e++) {
    const eps = model.edgeStrainLimit[e]!;
    if (!(eps > 0)) continue;
    const a = t.edges[4 * e]!, b = t.edges[4 * e + 1]!, L = t.edgeRestLength_m[e]!, d = [0, 1, 2].map(r => x[3 * a + r]! - x[3 * b + r]!) as Vec3, len = Math.sqrt(dot(d, d));
    if (!(len > 0)) continue;
    const d0 = [0, 1, 2].map(r => frame.x0[3 * a + r]! - frame.x0[3 * b + r]!) as Vec3, C0 = Math.sqrt(dot(d0, d0)) - (1 + eps) * L;
    const C = len - (1 + eps) * L - CLOTH_AVBD.edgeAlpha * Math.max(C0, 0), lambda = state.edgeDuals[2 * e]!, k = state.edgeDuals[2 * e + 1]!;
    const force = Math.max(k * C + lambda, 0), kEff = k * C + lambda > 0 ? k : C !== 0 ? Math.max(-lambda / C, 0) : 0, n = d.map(q => q / len) as Vec3;
    const B = outer(kEff - force / len, n, n); B[0] += force / len; B[4] += force / len; B[8] += force / len;
    scatter(space, A, [a, b], (i, j) => i === j ? B : B.map(q => -q) as Mat3);
  }
  return A;
}

/** Lower Cholesky factor of the dense SPD A (m square), or undefined at a non-positive pivot. */
export function clothCholesky(A: Float64Array, m: number): Float64Array | undefined {
  const L = new Float64Array(m * m);
  for (let i = 0; i < m; i++) for (let j = 0; j <= i; j++) {
    let s = A[i * m + j]!;
    for (let k = 0; k < j; k++) s -= L[i * m + k]! * L[j * m + k]!;
    if (i === j) { if (!(s > 0)) return undefined; L[i * m + i] = Math.sqrt(s); } else L[i * m + j] = s / L[j * m + j]!;
  }
  return L;
}

export function clothCholeskySolve(L: Float64Array, m: number, b: Float64Array): Float64Array {
  const y = new Float64Array(m);
  for (let i = 0; i < m; i++) { let s = b[i]!; for (let k = 0; k < i; k++) s -= L[i * m + k]! * y[k]!; y[i] = s / L[i * m + i]!; }
  for (let i = m - 1; i >= 0; i--) { let s = y[i]!; for (let k = i + 1; k < m; k++) s -= L[k * m + i]! * y[k]!; y[i] = s / L[i * m + i]!; }
  return y;
}

/** The step's incremental potential G(x) (each element once) and the sum of its terms' magnitudes. */
export function clothObjective(model: ClothSolverModel, frame: ClothStepFrame, state: ClothOracleState, x: ArrayLike<number>): { value: number; magnitude: number } {
  const t = model.topology, pos = (i: number) => at3(x, i), g: Vec3 = [0, 0, 0], H = [0, 0, 0, 0, 0, 0, 0, 0, 0] as Mat3;
  let value = 0, magnitude = 0;
  const add = (e: number) => { value += e; magnitude += Math.abs(e); };
  for (let v = 0; v < t.vertexCount; v++) {
    const w = t.mass_kg[v]! / (frame.hEff * frame.hEff);
    let d = 0; for (let c = 0; c < 3; c++) d += (x[3 * v + c]! - frame.y[3 * v + c]!) ** 2;
    add(0.5 * w * d);
    if (t.groups[v]! > 0) {
      const k = state.pinDuals[4 * v + 3]!;
      for (let c = 0; c < 3; c++)
        add(clothAugmentedEnergy(k, state.pinDuals[4 * v + c]!, x[3 * v + c]! - frame.target1[3 * v + c]! - CLOTH_AVBD.alpha * (frame.x0[3 * v + c]! - frame.target0[3 * v + c]!), false));
    }
  }
  for (let f = 0; f < t.faceCount; f++) {
    const vs = [0, 1, 2].map(k => pos(t.faces[3 * f + k]!)) as [Vec3, Vec3, Vec3];
    add(clothMembraneBlock(model.membrane[t.faceMaterial[f]!]!, t.faceDmInv, 4 * f, t.faceArea_m2[f]!, 0, ...vs, false).energy);
    add(clothAddFaceDamping(model, frame, f, 0, pos, g, H));
  }
  for (let r = 0; r < t.bendRowCount; r++) {
    add(clothAddBending(t, t.bendOffsets[r]!, pos, g, H));
    add(clothAddBendDamping(model, frame, t.bendOffsets[r]!, pos, g, H));
  }
  for (let e = 0; e < t.edgeCount; e++) {
    const eps = model.edgeStrainLimit[e]!;
    if (!(eps > 0)) continue;
    const a = t.edges[4 * e]!, b = t.edges[4 * e + 1]!, L = t.edgeRestLength_m[e]!;
    const d = [0, 1, 2].map(r => x[3 * a + r]! - x[3 * b + r]!) as Vec3, d0 = [0, 1, 2].map(r => frame.x0[3 * a + r]! - frame.x0[3 * b + r]!) as Vec3;
    const C = Math.sqrt(dot(d, d)) - (1 + eps) * L - CLOTH_AVBD.edgeAlpha * Math.max(Math.sqrt(dot(d0, d0)) - (1 + eps) * L, 0);
    add(clothAugmentedEnergy(state.edgeDuals[2 * e + 1]!, state.edgeDuals[2 * e]!, C, true));
  }
  return { value, magnitude };
}

export interface ClothCoarseStats {
  steps: number;
  /** Steps whose A_c had a non-positive pivot (corrections skipped). */
  indefinite: number;
  corrections: number;
  /** Accepted corrections per trial step: 1, 1/2, ... */
  accepted: number[];
}

/** The Float64 reference of the two-level correction, as a clothOracleStep hook. */
export class ClothCoarseCorrection implements ClothOracleCoarse {
  readonly stats: ClothCoarseStats = { steps: 0, indefinite: 0, corrections: 0, accepted: Array<number>(CLOTH_COARSE.trials).fill(0) };
  private factor: Float64Array | undefined;
  private applied = 0;
  /** `every`: correct before iterations 1, 1 + every, 1 + 2 every, ... of a step. */
  constructor(readonly model: ClothSolverModel, readonly space: ClothCoarseSpace, readonly epsilon = Number.EPSILON, readonly bending: boolean = CLOTH_COARSE.bending,
    readonly every: number = CLOTH_COARSE.every) {
    if (space.rowOffsets.length !== model.topology.vertexCount + 1) throw new Error('Cloth coarse space was built for another topology');
  }
  begin(frame: ClothStepFrame, state: ClothOracleState): void {
    this.stats.steps++; this.applied = 0;
    this.factor = clothCholesky(clothCoarseOperator(this.model, frame, state, this.space, this.bending), 3 * this.space.nodeCount);
    if (!this.factor) this.stats.indefinite++;
  }
  correct(frame: ClothStepFrame, state: ClothOracleState): void {
    if (!this.factor || this.applied++ % this.every !== 0) return;
    const s = this.space, n = this.model.topology.vertexCount, m = 3 * s.nodeCount, grad = new Float64Array(3 * n), r = new Float64Array(m);
    for (let v = 0; v < n; v++) grad.set(clothLocalSystem(this.model, frame, state, v, at3(state.x, v)).gradient, 3 * v);
    for (let v = 0; v < n; v++) for (let p = s.rowOffsets[v]!; p < s.rowOffsets[v + 1]!; p++)
      for (let b = 0; b < 3; b++) r[3 * s.rowNodes[p]! + b] -= s.rowWeights[p]! * grad[3 * v + b]!;
    const dc = clothCholeskySolve(this.factor, m, r), d = new Float64Array(3 * n);
    let slope = 0;
    for (let i = 0; i < m; i++) slope -= r[i]! * dc[i]!;
    for (let v = 0; v < n; v++) for (let p = s.rowOffsets[v]!; p < s.rowOffsets[v + 1]!; p++)
      for (let b = 0; b < 3; b++) d[3 * v + b] += s.rowWeights[p]! * dc[3 * s.rowNodes[p]! + b]!;
    this.stats.corrections++;
    if (!(slope < 0)) return;
    const G0 = clothObjective(this.model, frame, state, state.x), trial = new Float64Array(3 * n);
    for (let i = 0, alpha = 1; i < CLOTH_COARSE.trials; i++, alpha *= 0.5) {
      for (let k = 0; k < 3 * n; k++) trial[k] = state.x[k]! + alpha * d[k]!;
      const G = clothObjective(this.model, frame, state, trial);
      if (G.value - G0.value <= CLOTH_LINE_SEARCH.armijo * alpha * slope + 4 * this.epsilon * (G.magnitude + G0.magnitude)) {
        state.x.set(trial); this.stats.accepted[i]!++; return;
      }
    }
  }
}
