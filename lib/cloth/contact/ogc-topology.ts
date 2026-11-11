/**
 * Setup- and edit-time topology the OGC detector reads (CPU, Float-free).
 *
 * One u32 buffer, so each detection kernel binds it once:
 *   tris      T x 4   v0, v1, v2, material (the solver's triangles, copied)
 *   edges     E x 4   v0, v1, face0, face1 (NONE for a boundary edge)
 *   triEdges  T x 4   edge ids of (t0,t1), (t1,t2), (t2,t0); w bit k = t is face0 of edge k
 *   vtOff     N + 1   CSR offsets of each vertex's faces
 *   vtList    3T      faces around each vertex
 *   veOff     N + 1   CSR offsets of each vertex's edges
 *   veList    2E      edges around each vertex
 *
 * Unknown or non-manifold input fails here, explicitly.
 */

export const OGC_NONE = 0xffffffff;

export interface OgcMeshTopology {
  vertexCount: number;
  /** Triangles, vec4<u32> per face: v0, v1, v2, material. */
  triangles: Uint32Array;
  /** Edges, vec4<u32> per edge: v0, v1, face0, face1. */
  edges: Uint32Array;
}

export interface OgcTopologyLayout {
  vertexCount: number;
  triangleCount: number;
  edgeCount: number;
  trisOffset: number;
  edgesOffset: number;
  triEdgesOffset: number;
  vtOffOffset: number;
  vtListOffset: number;
  veOffOffset: number;
  veListOffset: number;
  words: number;
  /** Mean rest edge length at build time (metres), for the grid cell floor. */
  meanEdgeLength: number;
  maxEdgeLength: number;
  /** Smallest rest triangle altitude (metres). */
  minAltitude: number;
}

export interface OgcTopology { layout: OgcTopologyLayout; words: Uint32Array }

/** Build per-scene topology; `positions` (vec4 per vertex) give the edge-length statistics. */
export function buildOgcTopology(mesh: OgcMeshTopology, positions: Float32Array): OgcTopology {
  const N = mesh.vertexCount, T = mesh.triangles.length / 4, E = mesh.edges.length / 4;
  if (!Number.isInteger(T) || !Number.isInteger(E)) throw new Error("OGC topology: triangles and edges must be vec4<u32> arrays");
  if (positions.length < 4 * N) throw new Error("OGC topology: positions must hold vec4<f32> per vertex");
  if (T >= 1 << 28 || E >= 1 << 28) throw new Error("OGC topology: contact entries pack face and edge ids in 28 bits");
  const edgeKey = (a: number, b: number) => a < b ? a * N + b : b * N + a;
  const edgeOf = new Map<number, number>();
  for (let e = 0; e < E; e++) {
    const a = mesh.edges[4 * e], b = mesh.edges[4 * e + 1], f0 = mesh.edges[4 * e + 2], f1 = mesh.edges[4 * e + 3];
    if (a >= N || b >= N || a === b) throw new Error(`OGC topology: edge ${e} has invalid vertices`);
    if (f0 >= T || (f1 !== OGC_NONE && (f1 >= T || f1 === f0))) throw new Error(`OGC topology: edge ${e} has invalid faces`);
    const key = edgeKey(a, b);
    if (edgeOf.has(key)) throw new Error(`OGC topology: duplicate edge ${a}-${b}`);
    edgeOf.set(key, e);
  }
  const triEdges = new Uint32Array(4 * T), seen = new Uint8Array(E);
  for (let t = 0; t < T; t++) {
    let mask = 0;
    for (let k = 0; k < 3; k++) {
      const a = mesh.triangles[4 * t + k], b = mesh.triangles[4 * t + (k + 1) % 3];
      if (a >= N || a === b) throw new Error(`OGC topology: triangle ${t} has invalid vertices`);
      const e = edgeOf.get(edgeKey(a, b));
      if (e === undefined) throw new Error(`OGC topology: triangle ${t} edge ${a}-${b} missing from the edge table`);
      const f0 = mesh.edges[4 * e + 2], f1 = mesh.edges[4 * e + 3];
      if (f0 !== t && f1 !== t) throw new Error(`OGC topology: edge ${e} does not list triangle ${t}`);
      triEdges[4 * t + k] = e;
      if (f0 === t) mask |= 1 << k;
      seen[e]++;
    }
    triEdges[4 * t + 3] = mask;
  }
  for (let e = 0; e < E; e++) {
    const want = mesh.edges[4 * e + 3] === OGC_NONE ? 1 : 2;
    if (seen[e] !== want) throw new Error(`OGC topology: edge ${e} is used by ${seen[e]} triangles, its table lists ${want}`);
  }
  const vtCount = new Uint32Array(N + 1), veCount = new Uint32Array(N + 1);
  for (let t = 0; t < T; t++) for (let k = 0; k < 3; k++) vtCount[mesh.triangles[4 * t + k] + 1]++;
  for (let e = 0; e < E; e++) { veCount[mesh.edges[4 * e] + 1]++; veCount[mesh.edges[4 * e + 1] + 1]++; }
  for (let v = 0; v < N; v++) { vtCount[v + 1] += vtCount[v]; veCount[v + 1] += veCount[v]; }
  const vtList = new Uint32Array(3 * T), veList = new Uint32Array(2 * E);
  const vtFill = vtCount.slice(0, N), veFill = veCount.slice(0, N);
  for (let t = 0; t < T; t++) for (let k = 0; k < 3; k++) { const v = mesh.triangles[4 * t + k]; vtList[vtFill[v]++] = t; }
  for (let e = 0; e < E; e++) for (let k = 0; k < 2; k++) { const v = mesh.edges[4 * e + k]; veList[veFill[v]++] = e; }
  let sum = 0, max = 0;
  for (let e = 0; e < E; e++) {
    const a = mesh.edges[4 * e], b = mesh.edges[4 * e + 1];
    const l = Math.hypot(positions[4 * a] - positions[4 * b], positions[4 * a + 1] - positions[4 * b + 1], positions[4 * a + 2] - positions[4 * b + 2]);
    sum += l; max = Math.max(max, l);
  }
  let minAlt = Infinity;
  for (let t = 0; t < T; t++) {
    const P3 = [0, 1, 2].map(k => { const v = mesh.triangles[4 * t + k]; return [positions[4 * v], positions[4 * v + 1], positions[4 * v + 2]]; });
    const u = P3[1].map((c, i) => c - P3[0][i]), w = P3[2].map((c, i) => c - P3[0][i]);
    const cr = Math.hypot(u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]);
    const lmax = Math.max(Math.hypot(...u), Math.hypot(...w), Math.hypot(...P3[2].map((c, i) => c - P3[1][i])));
    if (lmax > 0) minAlt = Math.min(minAlt, cr / lmax);
  }
  const layout: OgcTopologyLayout = {
    vertexCount: N, triangleCount: T, edgeCount: E,
    trisOffset: 0, edgesOffset: 4 * T, triEdgesOffset: 4 * T + 4 * E,
    vtOffOffset: 8 * T + 4 * E, vtListOffset: 8 * T + 4 * E + N + 1,
    veOffOffset: 11 * T + 4 * E + N + 1, veListOffset: 11 * T + 4 * E + 2 * N + 2,
    words: 11 * T + 6 * E + 2 * N + 2,
    meanEdgeLength: E ? sum / E : 0, maxEdgeLength: max, minAltitude: minAlt,
  };
  const words = new Uint32Array(Math.ceil(layout.words / 4) * 4);
  words.set(mesh.triangles.subarray(0, 4 * T), layout.trisOffset);
  words.set(mesh.edges.subarray(0, 4 * E), layout.edgesOffset);
  words.set(triEdges, layout.triEdgesOffset);
  words.set(vtCount, layout.vtOffOffset);
  words.set(vtList, layout.vtListOffset);
  words.set(veCount, layout.veOffOffset);
  words.set(veList, layout.veListOffset);
  return { layout, words };
}

/** Edge table (vec4: v0 < v1, face0, face1) from vec4 triangles, for scenes that author faces only. */
export function buildOgcEdges(triangles: Uint32Array): Uint32Array {
  const T = triangles.length / 4, map = new Map<string, number>(), out: number[] = [];
  for (let t = 0; t < T; t++) for (let k = 0; k < 3; k++) {
    const a = triangles[4 * t + k], b = triangles[4 * t + (k + 1) % 3], lo = Math.min(a, b), hi = Math.max(a, b), key = `${lo},${hi}`;
    const e = map.get(key);
    if (e === undefined) { map.set(key, out.length / 4); out.push(lo, hi, t, OGC_NONE); }
    else if (out[4 * e + 3] !== OGC_NONE) throw new Error(`OGC edges: edge ${lo}-${hi} has more than two faces`);
    else out[4 * e + 3] = t;
  }
  return Uint32Array.from(out);
}

/** Words of a compiled face record (buildOgcFaceRecords). */
export const OGC_FACE_RECORD_WORDS = 8;

/**
 * Compiled face records for the detector's locality switch (OgcContactConfig.locality "topology"), 8 words a face:
 *   v0 v1 v2 flags e0 e1 e2 0
 * flags bits 0-2: the face owns edge k (it is the edge's face0; triEdges' mask); bits 3-5: edge k's table order is
 * reversed against the face's (the edge's v0 is the face's vertex k + 1), so a query rebuilds the edge's endpoints in
 * the edge table's own order; bits 6-8: the face is the owner of its vertex k (the first of the vertex's ascending
 * face list). One load then stands for the chain triEdges -> edge record, and for vtOff -> vtList.
 * Derived from the topology words alone, so it is rebuilt whenever they are (a cut, load or undo rebuilds both).
 */
export function buildOgcFaceRecords(topology: OgcTopology): Uint32Array {
  const L = topology.layout, w = topology.words, T = L.triangleCount;
  const out = new Uint32Array(OGC_FACE_RECORD_WORDS * T);
  for (let t = 0; t < T; t++) {
    const b = OGC_FACE_RECORD_WORDS * t;
    let flags = w[L.triEdgesOffset + 4 * t + 3]!;
    if (flags > 7) throw new Error(`OGC face records: face ${t} has an unknown ownership mask ${flags}`);
    for (let k = 0; k < 3; k++) {
      const v = w[L.trisOffset + 4 * t + k]!, next = w[L.trisOffset + 4 * t + (k + 1) % 3]!;
      const e = w[L.triEdgesOffset + 4 * t + k]!, e0 = w[L.edgesOffset + 4 * e]!, e1 = w[L.edgesOffset + 4 * e + 1]!;
      if (e0 === next && e1 === v) flags |= 8 << k;
      else if (!(e0 === v && e1 === next)) throw new Error(`OGC face records: edge ${e} is not face ${t}'s edge ${k}`);
      if (w[L.vtListOffset + w[L.vtOffOffset + v]!] === t) flags |= 64 << k;
      out[b + k] = v; out[b + 4 + k] = e;
    }
    out[b + 3] = flags;
  }
  return out;
}
