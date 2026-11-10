import type { ClothDocument, ClothMaterial, ClothTopologyData, ClothVector } from "./types";

/** Boundary marker of the edge table (matches lane C2's `0xffffffff`). */
export const CLOTH_NO_FACE = 0xffffffff;

export const sub3 = (a: ClothVector, b: ClothVector): [number, number, number] => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const dot3 = (a: ClothVector, b: ClothVector) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross3 = (a: ClothVector, b: ClothVector): [number, number, number] => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const norm3 = (a: ClothVector) => Math.hypot(a[0], a[1], a[2]);
export const point3 = (p: ArrayLike<number>, i: number): [number, number, number] => [p[3 * i]!, p[3 * i + 1]!, p[3 * i + 2]!];

/** Setup-time mesh tables shared by the Float64 oracle, the GPU solver and
 * the edit path. Element terms are recomputed per incident vertex, so
 * every table is rest data plus vertex-to-element adjacency. */
export interface PreparedClothTopology {
  readonly data: ClothTopologyData;
  readonly vertexCount: number;
  readonly faceCount: number;
  /** 3 vertices per face. */
  readonly faces: Uint32Array;
  readonly faceMaterial: Uint32Array;
  /** Column-major inverse of the 2x2 material-space edge matrix: (00,10,01,11). */
  readonly faceDmInv: Float64Array;
  readonly faceArea_m2: Float64Array;
  readonly edgeCount: number;
  /** (v0, v1, face0, face1) per canonical edge, v0 < v1, sorted; face1 is CLOTH_NO_FACE on a boundary. */
  readonly edges: Uint32Array;
  readonly edgeRestLength_m: Float64Array;
  /** Kirchhoff plate bending from the flat rest, CLOTH_BEND_ROWS rows per vertex (row r belongs to vertex
   * floor(r / CLOTH_BEND_ROWS), its own vertex first among the row's entries o in [bendOffsets[r], bendOffsets[r + 1])):
   * R_r x = sum_o w_o x_(v_o) (see `clothBendingRows`); the energy is sum_r (k_r/2)|R_r x|^2, k_r of its vertex i
   * (`bendStiffness_N_m3`). Each row vanishes for every affine map of the rest, so for every rigid motion. */
  readonly bendRowCount: number;
  readonly bendOffsets: Uint32Array;
  readonly bendVertices: Uint32Array;
  readonly bendWeights: Float64Array;
  /** Per row, k = B_i A_i^2 / A_free,i (N m^3) of its vertex i: the mean bending stiffness B of its faces that are not held
   * rigid (area weighted) times its area, scaled by A_i / A_free,i at a held region's edge (1 elsewhere, 0 inside one). */
  readonly bendStiffness_N_m3: Float64Array;
  /** Barycentric area A_i of each vertex (a third of each incident face); they sum to the sheet's area. */
  readonly vertexArea_m2: Float64Array;
  /** The rows holding each vertex: CSR of entry indices o, and the row of each entry. */
  readonly vertexBendOffsets: Uint32Array;
  readonly vertexBends: Uint32Array;
  readonly bendEntryRow: Uint32Array;
  /** CSR vertex adjacency; entries are (element << 2 | local order). */
  readonly vertexFaceOffsets: Uint32Array;
  readonly vertexFaces: Uint32Array;
  readonly vertexEdgeOffsets: Uint32Array;
  readonly vertexEdges: Uint32Array;
  readonly mass_kg: Float64Array;
  /** Attachment group per vertex; 0 is free. */
  readonly groups: Uint32Array;
}

export function clothMaterialErrors(m: ClothMaterial): string[] {
  const errors: string[] = [];
  const positive = ["arealDensity_kg_m2", "stretchWarp_N_m", "stretchWeft_N_m", "shear_N_m", "thickness_m"] as const;
  for (const key of positive) if (!(Number.isFinite(m[key]) && m[key] > 0)) errors.push(`Cloth material ${m.id}: ${key} must be positive`);
  for (const key of ["bending_N_m", "damping_s"] as const) if (!(Number.isFinite(m[key]) && m[key] >= 0)) errors.push(`Cloth material ${m.id}: ${key} must be non-negative`);
  if (!(m.compressionRatio > 0 && m.compressionRatio <= 1)) errors.push(`Cloth material ${m.id}: compressionRatio must lie in (0, 1]`);
  if (!Number.isFinite(m.poisson) || m.poisson * m.poisson * m.stretchWarp_N_m >= m.stretchWeft_N_m)
    errors.push(`Cloth material ${m.id}: Poisson coupling must keep the membrane law positive definite`);
  if (m.strainLimit !== undefined && !(m.strainLimit > 0)) errors.push(`Cloth material ${m.id}: strainLimit must be positive`);
  return errors;
}

export function clothTopologyErrors(data: ClothTopologyData, materials: readonly ClothMaterial[]): string[] {
  const errors: string[] = [];
  const n = data.restPositions_m.length / 3;
  if (!Number.isInteger(n) || n < 3) return ["Cloth topology needs at least three vertices"];
  if (!data.restPositions_m.every(Number.isFinite)) errors.push("Cloth rest positions must be finite");
  if (data.triangles.length % 3 || data.triangles.length === 0) errors.push("Cloth triangles must be non-empty vertex triples");
  if (!data.triangles.every(v => Number.isInteger(v) && v >= 0 && v < n)) errors.push("Cloth triangle index out of range");
  const faces = data.triangles.length / 3;
  for (let f = 0; f < faces; f++) {
    const [a, b, c] = [data.triangles[3 * f]!, data.triangles[3 * f + 1]!, data.triangles[3 * f + 2]!];
    if (a === b || b === c || c === a) { errors.push(`Cloth face ${f} repeats a vertex`); break; }
  }
  if (data.triangleMaterials && (data.triangleMaterials.length !== faces || !data.triangleMaterials.every(m => Number.isInteger(m) && m >= 0 && m < materials.length)))
    errors.push("Cloth triangle materials must name a material per face");
  if (data.triangleIds && data.triangleIds.length !== faces) errors.push("Cloth triangle ids must cover every face");
  const vector = (field: readonly number[] | undefined, width: number, name: string) => {
    if (field && (field.length !== width * n || !field.every(Number.isFinite))) errors.push(`Cloth ${name} must hold ${width} finite values per vertex`);
  };
  vector(data.initialPositions_m, 3, "initial positions"); vector(data.initialVelocities_m_s, 3, "initial velocities"); vector(data.uv, 2, "uv");
  if (data.vertexMass_kg && (data.vertexMass_kg.length !== n || !data.vertexMass_kg.every(m => Number.isFinite(m) && m > 0)))
    errors.push("Cloth vertex masses must be one positive finite value per vertex");
  if (data.groups && (data.groups.length !== n || !data.groups.every(g => Number.isInteger(g) && g >= 0))) errors.push("Cloth groups must be a non-negative integer per vertex");
  const s = data.solverState;
  if (s) {
    if (s.previousVelocities_m_s.length !== 3 * n || s.previousDisplacements_m.length !== 3 * n) errors.push("Cloth solver state must cover every vertex");
    if (!(s.previousStep_s >= 0)) errors.push("Cloth solver state needs a non-negative previous step");
    if (!(s.time_s >= 0)) errors.push("Cloth solver state needs a non-negative clock");
    if (s.contact && (s.contact.reference.length !== 4 * n || s.contact.held.length !== n || !s.contact.reference.every(Number.isFinite)
      || !s.contact.held.every(h => Number.isInteger(h) && h >= 0))) errors.push("Cloth contact state must hold a finite reference and rho per vertex and integer held flags");
  }
  return errors;
}

export function validateClothDocument(document: ClothDocument): string[] {
  if (!document || typeof document !== "object") return ["Cloth document must be an object"];
  const errors: string[] = [];
  if (document.schemaVersion !== "2.0.0") errors.push(`Unsupported cloth document schema ${String(document.schemaVersion)}`);
  if (typeof document.id !== "string" || !document.id) errors.push("Cloth document requires an id");
  if (!Array.isArray(document.materials) || !document.materials.length) return [...errors, "Cloth document requires materials"];
  errors.push(...document.materials.flatMap(clothMaterialErrors));
  if (!document.geometry) return [...errors, "Cloth document requires geometry"];
  if (document.geometry.kind === "inline") errors.push(...clothTopologyErrors(document.geometry.data, document.materials));
  else errors.push(`Unsupported cloth geometry kind: ${String((document.geometry as { kind: unknown }).kind)}`);
  for (const script of document.scripts ?? []) {
    if (script.kind !== "translate" && script.kind !== "rotate" && script.kind !== "oscillate") errors.push(`Unsupported cloth attachment script ${String(script.kind)}`);
    if (!Number.isInteger(script.group) || script.group < 1 || !Number.isFinite(script.start_s) || !(script.end_s > script.start_s)) errors.push("Invalid cloth attachment script interval or group");
    if (script.vector?.length !== 3 || !script.vector.every(Number.isFinite)) errors.push("Invalid cloth attachment script vector");
    if (script.kind === "rotate" && (!Number.isFinite(script.angle_rad) || norm3(script.vector) < 1e-12)) errors.push("Invalid cloth attachment rotation");
    if (script.kind === "oscillate" && !(script.period_s! > 0)) errors.push("Cloth oscillation needs a positive period");
    if (script.easing !== undefined && script.easing !== "linear" && script.easing !== "cosine") errors.push(`Unsupported cloth script easing ${String(script.easing)}`);
  }
  if (document.pendingSolids !== undefined) {
    if (!Array.isArray(document.pendingSolids)) errors.push("Cloth pending solids must be an array");
    else for (const p of document.pendingSolids)
      if (!p || typeof p.id !== "string" || !p.id || !(Number.isFinite(p.start_s) && p.start_s >= 0)) errors.push("Cloth pending solid needs an id and a non-negative finite start time");
  }
  const solver = document.solver;
  if (solver) {
    if (solver.integrator !== undefined && solver.integrator !== "implicit-euler" && solver.integrator !== "bdf2") errors.push(`Unsupported cloth integrator ${String(solver.integrator)}`);
    if (solver.newton !== undefined) {
      const n = solver.newton;
      if (!n || !(Number.isInteger(n.iterations) && n.iterations >= 1) || !(Number.isInteger(n.pcgIterations) && n.pcgIterations >= 1)) errors.push("Cloth Newton caps must be positive integers");
      else if ((n.pcgTolerance !== undefined && !(n.pcgTolerance >= 0)) || (n.residualTolerance !== undefined && !(n.residualTolerance >= 0)) || (n.velocityTolerance_m_s !== undefined && !(n.velocityTolerance_m_s >= 0 && Number.isFinite(n.velocityTolerance_m_s)))) errors.push("Cloth Newton tolerances must be non-negative");
    }
    if (solver.selfFriction !== undefined && !(solver.selfFriction >= 0 && Number.isFinite(solver.selfFriction))) errors.push("Cloth self friction must be a finite non-negative number");
    if (solver.contactPlaneRecordsPerVertex !== undefined && !(Number.isInteger(solver.contactPlaneRecordsPerVertex) && solver.contactPlaneRecordsPerVertex >= 1))
      errors.push("Cloth contact plane records per vertex must be a positive integer");
    // The VBD solver and its settings were removed (M4): every document runs Newton-PCG, and a document that names a
    // VBD setting is refused rather than silently ignored.
    for (const key of ["iterations", "chebyshevRho", "tier", "patchVertices", "patchIterations", "coarse"])
      if (key in solver) errors.push(`Cloth solver setting '${key}' was removed with the VBD solver: every step is Newton-PCG (solver.newton)`);
  }
  return errors;
}

/** Signed dihedral angle of hinge (o0, o1, e0, e1); zero when flat. */
export function clothHingeAngle(p: ArrayLike<number>, o0: number, o1: number, e0: number, e1: number): number {
  const x0 = point3(p, o0), x1 = point3(p, o1), x2 = point3(p, e0), x3 = point3(p, e1);
  const n1 = cross3(sub3(x2, x0), sub3(x3, x0)), n2 = cross3(sub3(x3, x1), sub3(x2, x1)), e = sub3(x3, x2);
  const l1 = norm3(n1), l2 = norm3(n2), le = norm3(e);
  if (!(l1 > 0 && l2 > 0 && le > 0)) throw new Error("Degenerate cloth hinge");
  const a = n1.map(v => v / l1) as unknown as ClothVector, b = n2.map(v => v / l2) as unknown as ClothVector;
  return Math.atan2(dot3(cross3(a, b), e) / le, dot3(a, b));
}

/** Largest rest dihedral angle (rad) the flat-rest bending model takes: 8 f32 ulps of 1. */
export const CLOTH_FLAT_REST_TOLERANCE_RAD = 8 * 2 ** -23;

/** Bending rows per vertex (row r belongs to vertex floor(r / CLOTH_BEND_ROWS)). */
export const CLOTH_BEND_ROWS = 3;

/**
 * Bending rows of a flat rest mesh: the Kirchhoff plate energy (B/2) int [nu (Delta x)^2 + (1 - nu)|grad grad x|^2] dA,
 * which for a bend of the flat sheet is (B/2) int [(kappa_1 + kappa_2)^2 - 2 (1 - nu) kappa_1 kappa_2] dA; at nu = 0,
 * (B/2) int (kappa_1^2 + kappa_2^2). Per vertex i three rows R_ia x = sum_j w_ija x_j over the same support, from the
 * second derivatives D_uu, D_vv, D_uv of x in rest coordinates (u, v) of the flat sheet at i:
 *   R_0 = sqrt((1 + nu) / 2) (D_uu + D_vv), R_1 = sqrt((1 - nu) / 2) (D_uu - D_vv), R_2 = sqrt(2 (1 - nu)) D_uv,
 * so sum_a |R_a x|^2 = nu |Delta x|^2 + (1 - nu)(|D_uu x|^2 + |D_vv x|^2 + 2 |D_uv x|^2), independent of the frame, and the
 * energy is sum_i (B_i A_i / 2) sum_a |R_ia x|^2 (`prepareClothTopology` scales A_i at a held region's edge). A
 * twist or a saddle costs its full energy (the Laplacian alone, nu = 1, gave both zero, so a strip could bend
 * anticlastically for free).
 *  - D starts from the weak (mixed finite element) Hessian of the piecewise-linear interpolant tested against the hat
 *    function phi_i (Stein et al. 2018, "Natural boundary conditions for smoothing in geometry processing"):
 *    A_i H_i = -sum_(f at i) A_f sym(grad phi_i (x) grad x_f) + sum_(boundary edges e at i) (|e|/2) sym(n_e (x) grad x_f(e)),
 *    A_i the barycentric area. Its boundary term gives a free edge the plate's natural conditions: least-norm fits
 *    over a one-sided support spread a boundary vertex's D_uu over the lines inside it, so the edge line's own
 *    curvature went under-penalised (a cantilever strip 23% too compliant at h/W = 1/8, the error growing as W shrank).
 *  - The weak Hessian is exact for quadratics on meshes of uniform valence (herring-bone, equilateral), not on
 *    mixed-valence ones (a union-jack grid: 0.75 and 1.5 of the curvature on alternate vertices, and none of the
 *    normal curvature at a boundary vertex). Each D then takes the least change of its weights that makes it exact for
 *    every quadratic, over the smallest determining support: the one-ring, then the nearest vertices of the next
 *    rings (up to the four-ring) one at a time. Where the weak Hessian is already exact the change is zero.
 *  - Each row vanishes for every affine map of the rest, so for every rigid motion. nu_i is per vertex (the caller's
 *    area-weighted material Poisson ratio), in [0, 1).
 * Entries with |w| below 1e-12 of the row's largest are dropped; the vertex's own weight, first in its row, is minus
 * the sum of the others. A vertex with no quadratic-exact support within its four-ring fails explicitly.
 */
export function clothBendingRows(rest: ArrayLike<number>, faces: ArrayLike<number>, n: number, poisson?: ArrayLike<number>): { offsets: Uint32Array; vertices: Uint32Array; weights: Float64Array; area_m2: Float64Array } {
  const P = (i: number) => point3(rest, i), F = faces.length / 3;
  const area = new Float64Array(n), ring = Array.from({length: n}, () => new Set<number>()), incident = Array.from({length: n}, () => [] as number[]);
  const edgeFaces = new Map<number, number>(), edgeKey = (a: number, b: number) => Math.min(a, b) * n + Math.max(a, b);
  for (let f = 0; f < F; f++) {
    const v = [faces[3 * f]!, faces[3 * f + 1]!, faces[3 * f + 2]!], twice = norm3(cross3(sub3(P(v[1]!), P(v[0]!)), sub3(P(v[2]!), P(v[0]!))));
    for (let k = 0; k < 3; k++) {
      const i = v[k]!; area[i] += twice / 6; incident[i]!.push(f);
      ring[i]!.add(v[(k + 1) % 3]!); ring[i]!.add(v[(k + 2) % 3]!);
      const e = edgeKey(i, v[(k + 1) % 3]!); edgeFaces.set(e, (edgeFaces.get(e) ?? 0) + 1);
    }
  }
  const R = CLOTH_BEND_ROWS, offsets = new Uint32Array(R * n + 1), vertices: number[] = [], weights: number[] = [];
  const push = (row: number, i: number, S: readonly number[], w: readonly number[]) => {
    let big = 0; for (const q of w) big = Math.max(big, Math.abs(q));
    let own = 0; vertices.push(i); weights.push(0);
    const at = weights.length - 1;
    S.forEach((j, k) => { if (j !== i && Math.abs(w[k]!) > 1e-12 * big) { vertices.push(j); weights.push(w[k]!); own -= w[k]!; } });
    weights[at] = own; offsets[row + 1] = vertices.length;
  };
  // Right-hand sides of the fit in the basis (1, u, v, u^2, uv, v^2): D_uu, D_vv, D_uv.
  const targets = [[0, 0, 0, 2, 0, 0], [0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 1, 0]];
  for (let i = 0; i < n; i++) {
    if (!incident[i]!.length) throw new Error(`Cloth vertex ${i} belongs to no face`);
    const nu = poisson?.[i] ?? 0;
    if (!(nu >= 0 && nu < 1)) throw new Error(`Cloth vertex ${i}: bending Poisson ratio ${nu} is outside [0, 1)`);
    // Rest frame at i (the face plane) and the mean incident edge length h, so the fit is in coordinates of order 1.
    const f0 = incident[i]![0]!, fv = [0, 1, 2].map(k => P(faces[3 * f0 + k]!)), nn = cross3(sub3(fv[1]!, fv[0]!), sub3(fv[2]!, fv[0]!)), ln = norm3(nn);
    const N: [number, number, number] = [nn[0] / ln, nn[1] / ln, nn[2] / ln], xi = P(i);
    let h = 0; for (const j of ring[i]!) h += norm3(sub3(P(j), xi)); h /= ring[i]!.size;
    const e = sub3(P(ring[i]!.values().next().value!), xi), ep = sub3(e, [N[0] * dot3(e, N), N[1] * dot3(e, N), N[2] * dot3(e, N)]), le = norm3(ep);
    const t1: [number, number, number] = [ep[0] / le, ep[1] / le, ep[2] / le], t2 = cross3(N, t1);
    const uv = (j: number): [number, number] => { const d = sub3(P(j), xi); return [dot3(d, t1) / h, dot3(d, t2) / h]; };
    // The weak Hessian in h-scaled coordinates (weights times h^2): (D_uu, D_vv, D_uv) per one-ring vertex.
    const weak = new Map<number, [number, number, number]>();
    const add = (j: number, s: number, a: readonly [number, number], b: readonly [number, number]) => {
      const q = weak.get(j) ?? [0, 0, 0]; q[0] += s * a[0] * b[0]; q[1] += s * a[1] * b[1]; q[2] += s * 0.5 * (a[0] * b[1] + a[1] * b[0]); weak.set(j, q);
    };
    let Ai = 0;
    for (const f of incident[i]!) {
      const v = [faces[3 * f]!, faces[3 * f + 1]!, faces[3 * f + 2]!], p = v.map(uv);
      const A2 = (p[1]![0] - p[0]![0]) * (p[2]![1] - p[0]![1]) - (p[2]![0] - p[0]![0]) * (p[1]![1] - p[0]![1]), Af = Math.abs(A2) / 2;
      const grad = [0, 1, 2].map(k => { const a = p[(k + 1) % 3]!, b = p[(k + 2) % 3]!; return [(a[1] - b[1]) / A2, (b[0] - a[0]) / A2] as [number, number]; });
      const k0 = v.indexOf(i); Ai += Af / 3;
      for (let k = 0; k < 3; k++) add(v[k]!, -Af, grad[k0]!, grad[k]!);
      for (const k1 of [(k0 + 1) % 3, (k0 + 2) % 3]) {
        if (edgeFaces.get(edgeKey(i, v[k1]!)) !== 1) continue;
        const pa = p[k0]!, pb = p[k1]!, c = p[3 - k0 - k1]!, L = Math.hypot(pb[0] - pa[0], pb[1] - pa[1]);
        let nrm: [number, number] = [(pb[1] - pa[1]) / L, -(pb[0] - pa[0]) / L];
        if ((c[0] - pa[0]) * nrm[0] + (c[1] - pa[1]) * nrm[1] > 0) nrm = [-nrm[0], -nrm[1]];
        for (let k = 0; k < 3; k++) add(v[k]!, L / 2, nrm, grad[k]!);
      }
    }
    for (const q of weak.values()) { q[0] /= Ai; q[1] /= Ai; q[2] /= Ai; }
    // Candidates by ring, then rest distance: the one-ring first, then the nearest farther vertices one at a time
    // until the quadratics are determined, so the rows (and the vertices they couple) stay as small as they can.
    const ringOf = new Map<number, number>([[i, 0]]); let front = [i];
    for (let r = 1; r <= 4; r++) { const next: number[] = []; for (const u of front) for (const q of ring[u]!) if (!ringOf.has(q)) { ringOf.set(q, r); next.push(q); } front = next; }
    const candidates = [...ringOf.keys()].sort((a, b) => ringOf.get(a)! - ringOf.get(b)! || norm3(sub3(P(a), xi)) - norm3(sub3(P(b), xi)));
    let rows: number[][] | undefined, S: number[] = [];
    for (let size = 1 + ring[i]!.size; size <= candidates.length && !rows; size++) {
      S = candidates.slice(0, size);
      if (S.length < 6) continue;
      const M = S.map(j => { const [u, v] = uv(j); return [1, u, v, u * u, u * v, v * v]; });
      // The least change: w = w_weak + M y with G y = b - M^T w_weak, G = sum_j m_j m_j^T (6 x 6; Gauss-Jordan,
      // partial pivoting), one b per D.
      const residual = targets.map((t, d) => t.map((b, a) => b - S.reduce((q, j, k) => q + (weak.get(j)?.[d] ?? 0) * M[k]![a]!, 0)));
      const A = Array.from({length: 6}, (_, a) => [...Array.from({length: 6}, (_, c) => M.reduce((q, m) => q + m[a]! * m[c]!, 0)), ...residual.map(t => t[a]!)]);
      let pmax = 0, pmin = Infinity;
      for (let c = 0; c < 6; c++) {
        let p = c; for (let r = c + 1; r < 6; r++) if (Math.abs(A[r]![c]!) > Math.abs(A[p]![c]!)) p = r;
        [A[c], A[p]] = [A[p]!, A[c]!];
        const pv = A[c]![c]!; pmax = Math.max(pmax, Math.abs(pv)); pmin = Math.min(pmin, Math.abs(pv));
        if (!(Math.abs(pv) > 0)) break;
        for (let r = 0; r < 6; r++) if (r !== c) { const q = A[r]![c]! / pv; for (let k = c; k < 9; k++) A[r]![k]! -= q * A[c]![k]!; }
      }
      if (!(pmin > 1e-9 * pmax)) continue;
      rows = targets.map((_, d) => { const y = A.map((r, a) => r[6 + d]! / r[a]!); return S.map((j, k) => ((weak.get(j)?.[d] ?? 0) + M[k]!.reduce((q, mm, a) => q + mm * y[a]!, 0)) / (h * h)); });
    }
    if (!rows) throw new Error(`Cloth vertex ${i}: no quadratic-exact bending support within its four-ring (the mesh is too small or degenerate there)`);
    const [Duu, Dvv, Duv] = rows as [number[], number[], number[]], a0 = Math.sqrt((1 + nu) / 2), a1 = Math.sqrt((1 - nu) / 2), a2 = Math.sqrt(2 * (1 - nu));
    push(R * i, i, S, Duu.map((w, k) => a0 * (w + Dvv[k]!)));
    push(R * i + 1, i, S, Duu.map((w, k) => a1 * (w - Dvv[k]!)));
    push(R * i + 2, i, S, Duv.map(w => a2 * w));
  }
  return { offsets, vertices: Uint32Array.from(vertices), weights: Float64Array.from(weights), area_m2: area };
}

function csr(count: number, entries: readonly (readonly [number, number])[]): {offsets: Uint32Array; values: Uint32Array} {
  const offsets = new Uint32Array(count + 1);
  for (const [v] of entries) offsets[v + 1]!++;
  for (let i = 0; i < count; i++) offsets[i + 1]! += offsets[i]!;
  const cursor = offsets.slice(0, count), values = new Uint32Array(entries.length);
  for (const [v, value] of entries) values[cursor[v]!++] = value;
  return {offsets, values};
}

export function prepareClothTopology(data: ClothTopologyData, materials: readonly ClothMaterial[]): PreparedClothTopology {
  const errors = clothTopologyErrors(data, materials);
  if (errors.length) throw new Error(errors.join("; "));
  const n = data.restPositions_m.length / 3, F = data.triangles.length / 3, rest = data.restPositions_m;
  const faces = Uint32Array.from(data.triangles), faceMaterial = Uint32Array.from(data.triangleMaterials ?? new Array(F).fill(0));
  const faceDmInv = new Float64Array(4 * F), faceArea = new Float64Array(F), mass = new Float64Array(n);
  for (let f = 0; f < F; f++) {
    const a = faces[3 * f]!, b = faces[3 * f + 1]!, c = faces[3 * f + 2]!;
    let d00: number, d10: number, d01: number, d11: number;
    if (data.uv) {
      d00 = data.uv[2 * b]! - data.uv[2 * a]!; d10 = data.uv[2 * b + 1]! - data.uv[2 * a + 1]!;
      d01 = data.uv[2 * c]! - data.uv[2 * a]!; d11 = data.uv[2 * c + 1]! - data.uv[2 * a + 1]!;
    } else {
      const pa = point3(rest, a), e1 = sub3(point3(rest, b), pa), e2 = sub3(point3(rest, c), pa), nrm = cross3(e1, e2), ln = norm3(nrm);
      if (!(ln > 0)) throw new Error(`Cloth face ${f} has no rest area`);
      const nn: ClothVector = [nrm[0] / ln, nrm[1] / ln, nrm[2] / ln];
      // Warp: the world x axis projected into the face (the edge where x is normal).
      let t: ClothVector = [1 - nn[0] * nn[0], -nn[0] * nn[1], -nn[0] * nn[2]];
      if (norm3(t) < 1e-6) t = e1;
      const lt = norm3(t); t = [t[0] / lt, t[1] / lt, t[2] / lt];
      const w = cross3(nn, t);
      d00 = dot3(t, e1); d10 = dot3(w, e1); d01 = dot3(t, e2); d11 = dot3(w, e2);
    }
    const det = d00 * d11 - d01 * d10;
    if (!(Math.abs(det) > 0)) throw new Error(`Cloth face ${f} has no material area`);
    faceDmInv[4 * f] = d11 / det; faceDmInv[4 * f + 1] = -d10 / det; faceDmInv[4 * f + 2] = -d01 / det; faceDmInv[4 * f + 3] = d00 / det;
    faceArea[f] = Math.abs(det) / 2;
    const rho = materials[faceMaterial[f]!]!.arealDensity_kg_m2;
    for (const v of [a, b, c]) mass[v]! += rho * faceArea[f]! / 3;
  }
  if (data.vertexMass_kg) mass.set(data.vertexMass_kg);
  // Canonical edges: (min, max) sorted; the first face is the lower face index.
  const edgeMap = new Map<number, number[]>();
  for (let f = 0; f < F; f++) for (let k = 0; k < 3; k++) {
    const u = faces[3 * f + k]!, v = faces[3 * f + (k + 1) % 3]!, key = Math.min(u, v) * n + Math.max(u, v);
    const list = edgeMap.get(key); if (list) list.push(f); else edgeMap.set(key, [f]);
  }
  const keys = [...edgeMap.keys()].sort((x, y) => x - y), E = keys.length;
  const edges = new Uint32Array(4 * E), edgeRest = new Float64Array(E);
  let interior = 0;
  for (let e = 0; e < E; e++) {
    const key = keys[e]!, v0 = Math.floor(key / n), v1 = key % n, list = edgeMap.get(key)!;
    if (list.length > 2) throw new Error(`Cloth edge ${v0}-${v1} is non-manifold (${list.length} faces)`);
    edges.set([v0, v1, list[0]!, list[1] ?? CLOTH_NO_FACE], 4 * e);
    edgeRest[e] = norm3(sub3(point3(rest, v1), point3(rest, v0)));
    if (list.length === 2) {
      const f0 = list[0]!, f1 = list[1]!;
      const opposite = (f: number) => [0, 1, 2].map(k => faces[3 * f + k]!).find(v => v !== v0 && v !== v1)!;
      // The flat rest check: orient the edge along face0's winding, so a flat sheet reads zero.
      let s = v0, t = v1;
      for (let k = 0; k < 3; k++) if (faces[3 * f0 + k] === v1 && faces[3 * f0 + (k + 1) % 3] === v0) { s = v1; t = v0; }
      const o0 = opposite(f0), o1 = opposite(f1);
      const angle = clothHingeAngle(rest, o0, o1, s, t);
      if (!(Math.abs(angle) <= CLOTH_FLAT_REST_TOLERANCE_RAD))
        throw new Error(`Cloth hinge ${interior} (vertices ${o0}, ${o1} across edge ${s}-${t}) has rest dihedral angle ${angle} rad: quadratic bending needs a flat rest shape`);
      interior++;
    }
  }
  const faceEntries: [number, number][] = [], edgeEntries: [number, number][] = [];
  for (let f = 0; f < F; f++) for (let k = 0; k < 3; k++) faceEntries.push([faces[3 * f + k]!, f << 2 | k]);
  for (let e = 0; e < E; e++) for (let k = 0; k < 2; k++) edgeEntries.push([edges[4 * e + k]!, e << 2 | k]);
  const vf = csr(n, faceEntries), ve = csr(n, edgeEntries);
  for (let v = 0; v < n; v++) if (vf.offsets[v + 1] === vf.offsets[v]) throw new Error(`Cloth vertex ${v} belongs to no face`);
  // A face whose three vertices are pinned to one group is held rigid, so it is not elastic sheet. A vertex on the
  // held region's edge has the kink there inside its cell: its rows see the mean curvature over the cell, the held
  // part's zero averaged with the free part's kappa, so A_i H_i = A_free kappa. The free part's energy
  // (B/2) kappa^2 A_free is then (B/2) |H_i|^2 A_i^2 / A_free, so the row's stiffness is B A_i^2 / A_free (A_free, the
  // vertex's area in faces that are not held). Centred rows over the kink otherwise carry a quarter of the energy on
  // the free half-cell where they should carry all of it, and the clamp rotates. On a cantilever this is about
  // 2h/L of the tip deflection, 1.7 deg of the Peirce chord at h = 5 mm. A vertex wholly inside a held region
  // carries no bending.
  const groups = data.groups, held = (f: number) => { const g = groups?.[faces[3 * f]!] ?? 0; return g > 0 && groups![faces[3 * f + 1]!] === g && groups![faces[3 * f + 2]!] === g; };
  // Per vertex the area-weighted mean of its free faces' bending stiffness B and Poisson ratio (the plate's, as the membrane's).
  const bendB = new Float64Array(n), bendNu = new Float64Array(n), freeArea = new Float64Array(n);
  for (let v = 0; v < n; v++) {
    let a = 0, b = 0, nu = 0, all = 0, nuAll = 0;
    for (let j = vf.offsets[v]!; j < vf.offsets[v + 1]!; j++) {
      const f = vf.values[j]! >> 2, m = materials[faceMaterial[f]!]!; all += faceArea[f]!; nuAll += faceArea[f]! * m.poisson;
      if (held(f)) continue;
      a += faceArea[f]!; b += faceArea[f]! * m.bending_N_m; nu += faceArea[f]! * m.poisson;
    }
    freeArea[v] = a / 3; bendB[v] = a > 0 ? b / a : 0; bendNu[v] = a > 0 ? nu / a : nuAll / all;
  }
  const bend = clothBendingRows(rest, faces, n, bendNu), rows = CLOTH_BEND_ROWS * n;
  const bendK = new Float64Array(rows);
  for (let r = 0; r < rows; r++) { const v = Math.floor(r / CLOTH_BEND_ROWS), A = bend.area_m2[v]!; bendK[r] = freeArea[v]! > 0 ? bendB[v]! * A * A / freeArea[v]! : 0; }
  const bendEntries: [number, number][] = [], bendEntryRow = new Uint32Array(bend.vertices.length);
  for (let r = 0; r < rows; r++) for (let o = bend.offsets[r]!; o < bend.offsets[r + 1]!; o++) { bendEntries.push([bend.vertices[o]!, o]); bendEntryRow[o] = r; }
  const vb = csr(n, bendEntries);
  return {data, vertexCount: n, faceCount: F, faces, faceMaterial, faceDmInv, faceArea_m2: faceArea, edgeCount: E, edges, edgeRestLength_m: edgeRest,
    bendRowCount: rows, bendOffsets: bend.offsets, bendVertices: bend.vertices, bendWeights: bend.weights, bendStiffness_N_m3: bendK, vertexArea_m2: bend.area_m2,
    vertexBendOffsets: vb.offsets, vertexBends: vb.values, bendEntryRow,
    vertexFaceOffsets: vf.offsets, vertexFaces: vf.values, vertexEdgeOffsets: ve.offsets, vertexEdges: ve.values,
    mass_kg: mass, groups: Uint32Array.from(data.groups ?? new Array(n).fill(0))};
}

/** The canonical edge order of `prepareClothTopology`: keys min*n+max, sorted. */
export function clothCanonicalEdgeKeys(triangles: ArrayLike<number>, n: number): number[] {
  const keys = new Set<number>();
  for (let f = 0; f < triangles.length / 3; f++) for (let k = 0; k < 3; k++) {
    const u = triangles[3 * f + k]!, v = triangles[3 * f + (k + 1) % 3]!;
    keys.add(Math.min(u, v) * n + Math.max(u, v));
  }
  return [...keys].sort((x, y) => x - y);
}
