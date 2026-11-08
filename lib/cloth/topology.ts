import type { ClothDocument, ClothMaterial, ClothTopologyData, ClothVector } from "./types";

/** Boundary marker of the edge table (matches lane C2's `0xffffffff`). */
export const CLOTH_NO_FACE = 0xffffffff;

export const sub3 = (a: ClothVector, b: ClothVector): [number, number, number] => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const dot3 = (a: ClothVector, b: ClothVector) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross3 = (a: ClothVector, b: ClothVector): [number, number, number] => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const norm3 = (a: ClothVector) => Math.hypot(a[0], a[1], a[2]);
export const point3 = (p: ArrayLike<number>, i: number): [number, number, number] => [p[3 * i]!, p[3 * i + 1]!, p[3 * i + 2]!];

/** Setup-time mesh tables shared by the Float64 oracle, the GPU solver and
 * the edit path. Element terms are recomputed per incident vertex (VBD), so
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
  readonly hingeCount: number;
  /** (opposite0, opposite1, edgeStart, edgeEnd) per interior edge. */
  readonly hinges: Uint32Array;
  readonly hingeEdge: Uint32Array;
  readonly hingeRestAngle_rad: Float64Array;
  /** Bending stiffness k_b 3|e|^2/(A0+A1), so the energy is (k/2)(theta - theta0)^2. */
  readonly hingeStiffness_N_m: Float64Array;
  /** CSR vertex adjacency; entries are (element << 2 | local order). */
  readonly vertexFaceOffsets: Uint32Array;
  readonly vertexFaces: Uint32Array;
  readonly vertexHingeOffsets: Uint32Array;
  readonly vertexHinges: Uint32Array;
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
    if (solver.tier !== undefined && solver.tier !== "A" && solver.tier !== "B") errors.push(`Unsupported cloth solver tier ${String(solver.tier)}`);
    if (solver.iterations !== undefined && !(Number.isInteger(solver.iterations) && solver.iterations >= 1)) errors.push("Cloth iterations must be a positive integer");
    if (solver.chebyshevRho !== undefined && !(solver.chebyshevRho >= 0 && solver.chebyshevRho < 1)) errors.push("Cloth Chebyshev rho must lie in [0, 1)");
    if (solver.selfFriction !== undefined && !(solver.selfFriction >= 0 && Number.isFinite(solver.selfFriction))) errors.push("Cloth self friction must be a finite non-negative number");
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
  const hingeList: number[] = [], hingeEdge: number[] = [], hingeAngle: number[] = [], hingeK: number[] = [];
  for (let e = 0; e < E; e++) {
    const key = keys[e]!, v0 = Math.floor(key / n), v1 = key % n, list = edgeMap.get(key)!;
    if (list.length > 2) throw new Error(`Cloth edge ${v0}-${v1} is non-manifold (${list.length} faces)`);
    edges.set([v0, v1, list[0]!, list[1] ?? CLOTH_NO_FACE], 4 * e);
    edgeRest[e] = norm3(sub3(point3(rest, v1), point3(rest, v0)));
    if (list.length === 2) {
      const f0 = list[0]!, f1 = list[1]!;
      const opposite = (f: number) => [0, 1, 2].map(k => faces[3 * f + k]!).find(v => v !== v0 && v !== v1)!;
      // Orient the hinge edge along face0's winding, so a flat sheet reads zero.
      let s = v0, t = v1;
      for (let k = 0; k < 3; k++) if (faces[3 * f0 + k] === v1 && faces[3 * f0 + (k + 1) % 3] === v0) { s = v1; t = v0; }
      const o0 = opposite(f0), o1 = opposite(f1);
      hingeList.push(o0, o1, s, t); hingeEdge.push(e);
      hingeAngle.push(clothHingeAngle(rest, o0, o1, s, t));
      const kb = Math.max(materials[faceMaterial[f0]!]!.bending_N_m, 0) * 0.5 + Math.max(materials[faceMaterial[f1]!]!.bending_N_m, 0) * 0.5;
      hingeK.push(kb * 3 * edgeRest[e]! ** 2 / (faceArea[f0]! + faceArea[f1]!));
    }
  }
  const H = hingeEdge.length, hinges = Uint32Array.from(hingeList);
  const faceEntries: [number, number][] = [], hingeEntries: [number, number][] = [], edgeEntries: [number, number][] = [];
  for (let f = 0; f < F; f++) for (let k = 0; k < 3; k++) faceEntries.push([faces[3 * f + k]!, f << 2 | k]);
  for (let h = 0; h < H; h++) for (let k = 0; k < 4; k++) hingeEntries.push([hinges[4 * h + k]!, h << 2 | k]);
  for (let e = 0; e < E; e++) for (let k = 0; k < 2; k++) edgeEntries.push([edges[4 * e + k]!, e << 2 | k]);
  const vf = csr(n, faceEntries), vh = csr(n, hingeEntries), ve = csr(n, edgeEntries);
  for (let v = 0; v < n; v++) if (vf.offsets[v + 1] === vf.offsets[v]) throw new Error(`Cloth vertex ${v} belongs to no face`);
  return {data, vertexCount: n, faceCount: F, faces, faceMaterial, faceDmInv, faceArea_m2: faceArea, edgeCount: E, edges, edgeRestLength_m: edgeRest,
    hingeCount: H, hinges, hingeEdge: Uint32Array.from(hingeEdge), hingeRestAngle_rad: Float64Array.from(hingeAngle), hingeStiffness_N_m: Float64Array.from(hingeK),
    vertexFaceOffsets: vf.offsets, vertexFaces: vf.values, vertexHingeOffsets: vh.offsets, vertexHinges: vh.values, vertexEdgeOffsets: ve.offsets, vertexEdges: ve.values,
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

/** Vertices that share a force element (face or hinge) must not share a colour. */
export function clothVertexNeighbours(t: PreparedClothTopology): {offsets: Uint32Array; values: Uint32Array} {
  const sets = Array.from({length: t.vertexCount}, () => new Set<number>());
  const link = (list: ArrayLike<number>, width: number, count: number) => {
    for (let e = 0; e < count; e++) for (let i = 0; i < width; i++) for (let j = 0; j < width; j++)
      if (i !== j) sets[list[width * e + i]!]!.add(list[width * e + j]!);
  };
  link(t.faces, 3, t.faceCount); link(t.hinges, 4, t.hingeCount);
  const entries: [number, number][] = [];
  sets.forEach((s, v) => [...s].sort((a, b) => a - b).forEach(u => entries.push([v, u])));
  return csr(t.vertexCount, entries);
}

/** Greedy vertex colouring (setup time). `previous` keeps the colours of
 * vertices outside `touched`; only touched vertices are recoloured, each
 * taking the smallest colour none of its neighbours holds. */
/**
 * Greedy colouring can leave a last colour class of a handful of vertices (the Bridson 150 x 150 tablecloth: five
 * classes of ~4600 and one single vertex), and every class costs one dependent dispatch per iteration. Empty a small last class
 * by Kempe-chain interchanges: for a member v and colours (a, b) below it, swap a and b on the a/b component holding
 * v's a-neighbours; when that component holds none of v's b-neighbours, colour a becomes free at v. The colouring
 * stays proper throughout, and is left unchanged when a member cannot be moved within a bounded component.
 */
/** Largest a/b component an interchange may recolour: a mesh-wide chain would reshuffle whole classes and lose the
 *  index locality of each colour's vertex list (measured: +12 % per colour pass on the 150 x 150 tablecloth). */
const KEMPE_COMPONENT_LIMIT = 256;
function emptyLastColour(neighbours: {offsets: Uint32Array; values: Uint32Array}, colour: Uint32Array): boolean {
  const n = colour.length;
  let last = 0; for (let v = 0; v < n; v++) last = Math.max(last, colour[v]!);
  const members: number[] = []; for (let v = 0; v < n; v++) if (colour[v] === last) members.push(v);
  if (last === 0 || members.length > n / (8 * (last + 1))) return false;
  const saved = Uint32Array.from(colour), mark = new Int32Array(n).fill(-1);
  let stamp = 0;
  const neighboursOf = (v: number) => neighbours.values.subarray(neighbours.offsets[v]!, neighbours.offsets[v + 1]!);
  for (const v of members) {
    let moved = false;
    for (let a = 0; a < last && !moved; a++) for (let b = 0; b < last && !moved; b++) {
      if (a === b) continue;
      const id = stamp++, stack: number[] = [];
      for (const u of neighboursOf(v)) if (colour[u] === a && mark[u] !== id) { mark[u] = id; stack.push(u); }
      const component: number[] = [];
      while (stack.length && component.length <= KEMPE_COMPONENT_LIMIT) {
        const u = stack.pop()!; component.push(u);
        for (const w of neighboursOf(u)) if (w !== v && mark[w] !== id && (colour[w] === a || colour[w] === b)) { mark[w] = id; stack.push(w); }
      }
      if (component.length > KEMPE_COMPONENT_LIMIT || neighboursOf(v).some(u => colour[u] === b && mark[u] === id)) continue;
      for (const u of component) colour[u] = colour[u] === a ? b : a;
      colour[v] = a; moved = true;
    }
    if (!moved) { colour.set(saved); return false; }
  }
  return true;
}

export function colourClothVertices(neighbours: {offsets: Uint32Array; values: Uint32Array}, previous?: Uint32Array, touched?: ReadonlySet<number>): Uint32Array {
  const n = neighbours.offsets.length - 1, NONE = 0xffffffff, colour = new Uint32Array(n).fill(NONE);
  if (previous && touched) for (let v = 0; v < n; v++) if (!touched.has(v)) colour[v] = previous[v]!;
  const order = previous && touched ? [...touched].sort((a, b) => a - b) : Array.from({length: n}, (_, v) => v);
  const used: number[] = [];
  for (const v of order) {
    used.length = 0;
    for (let j = neighbours.offsets[v]!; j < neighbours.offsets[v + 1]!; j++) { const c = colour[neighbours.values[j]!]!; if (c !== NONE) used[c] = 1; }
    let c = 0; while (used[c]) c++;
    colour[v] = c;
  }
  if (!previous) while (emptyLastColour(neighbours, colour));
  // A kept colour that now conflicts (an edit joined two vertices) is a bug in the caller's touched set.
  for (let v = 0; v < n; v++) for (let j = neighbours.offsets[v]!; j < neighbours.offsets[v + 1]!; j++)
    if (colour[neighbours.values[j]!] === colour[v]) throw new Error(`Cloth colouring conflict at vertex ${v}`);
  return colour;
}
