import {ClothContactBVH} from './contact-bvh';
import {clothLocalShellEdges, clothRodCut, clothShellCache, clothVertexContactRadius, clothVertexLocalFaces, inSolveContactUnsupported, rodSelfClosest, type RodContactShape} from './contact-constraints';
import type {ClothCollider} from './contact';
import type {PreparedClothTopology} from './topology';
import type {ClothVector} from './types';

/** GPU detection for the in-solve contact of lib/cloth/contact-constraints.ts.
 *
 * The same constraint set as ClothContactSet.detect, built on the GPU every
 * outer iteration: vertex-collider conservative advancement against planes,
 * spheres, boxes and the scene's static baked solid (the trilinear bricks the
 * fluid uses), vertex-face and edge-edge CCD between swept primitives found by
 * a uniform hash grid, with resolveClothContacts' exclusions (the one-ring,
 * each vertex's connected rest patch, each edge's rest patch). The exclusions
 * depend only on the rest mesh, so the CPU builds them once per topology with
 * the reference's own functions; the GPU only tests membership.
 *
 * Geometry is evaluated in a fixed frame F as hi/lo Float32 pairs plus the
 * increment: contact gaps are formed from differences of nearby points (edges,
 * brick-local coordinates) so they keep Float32 resolution of the gap, not of
 * the world coordinate. Keys are those of ClothContactSet (c:i:ci, v:i:f,
 * e:ai:bi); an open-addressing table on the GPU dedups held pairs and carries
 * multipliers to the next step. */

export const STATIC_RECORDS = Symbol('cloth static collider records');
export interface ClothStaticColliderRecords {
  readonly h: number;
  readonly records: readonly {readonly min: ClothVector; readonly max: ClothVector; readonly values?: readonly number[]; readonly lipschitz: number}[];
  readonly cuts: number;
}
/** Static baked colliders register their records here (scene-contact.ts). */
export const clothStaticColliderRecords = new WeakMap<ClothCollider, ClothStaticColliderRecords>();

/** The last CPU remnant of cloth contact, or undefined when the GPU Newton
 * detects and solves this topology's contact against these colliders in the
 * solve. It names what the GPU does not cover yet: elliptical rods and clipped
 * rod caps, rod segments against box and sampled solids (these rods keep the
 * sequential post-projection), and rigid bodies, moving sampled solids and
 * box-subtraction cuts (the Float64 ClothContactSet detects between GPU solves).
 * Nothing else reaches the CPU: a GPU contact error fails the step. The plan
 * (docs/plans/wetcloth-implementation-2026-11-05.md) removes this predicate in
 * the commit that puts those cases in the GPU Newton. */
export function clothContactCPURemnant(topology: PreparedClothTopology, colliders: readonly ClothCollider[]): string | undefined {
  return gpuContactUnsupported(topology, colliders);
}

export const ROD_BOX_SAMPLED_UNSUPPORTED = 'rod segments against box and sampled colliders are not on the GPU';
/** Why a collider set needs the CPU detection path, or undefined. */
export function gpuContactUnsupported(topology: PreparedClothTopology, colliders: readonly ClothCollider[]): string | undefined {
  const base = inSolveContactUnsupported(topology, colliders);
  if (base) return base;
  for (const rod of topology.rods) {const m = topology.materials[rod.material]!; if ((m.rodRadiusB_m ?? m.rodRadius_m) !== m.rodRadius_m) return 'elliptical rod contact is not on the GPU';}
  for (let r = 0; r < topology.rods.length; r++) if (clothRodCut(topology, r)) return 'clipped rod caps are not on the GPU';
  // Rod segment interiors meet planes and spheres in closed form here; a box or
  // sampled solid needs the capsule's convex distance, not on the GPU yet.
  if (topology.rods.length && colliders.some(c => c.kind === 'box' || c.kind === 'sampled')) return ROD_BOX_SAMPLED_UNSUPPORTED;
  for (const c of colliders) {
    if (c.bodyId) return `collider reactions for body ${c.bodyId} are summed on the CPU`;
    if (c.kind === 'rigid') return 'rigid colliders are sampled on the CPU';
    if (c.kind === 'sampled') {
      const records = clothStaticColliderRecords.get(c);
      if (!records) return 'a moving sampled source collider is sampled on the CPU';
      if (records.cuts) return 'box-subtraction cuts of the static solid are sampled on the CPU';
    }
    if (c.kind === 'plane' && !(Math.hypot(...c.normal) > 0)) throw new Error('Invalid cloth collision plane');
  }
  return undefined;
}

// ---- static topology data (u32 words; floats as bits), stored after the solver's topology words
export interface ClothContactStatic {
  readonly words: Uint32Array<ArrayBuffer>;
  readonly edges: number;
  readonly cellSize: number;
  readonly offsets: {sVInfo: number; sRingStart: number; sRingList: number; sFrStart: number; sFrList: number; sLvStart: number; sLvList: number; sEdge: number; sE3Start: number; sE3List: number; sFaceR: number;
    sSegInfo: number; sNbStart: number; sNbList: number; sRpXStart: number; sRpXList: number; sRsXStart: number; sRsXList: number};
}
const statics = new WeakMap<PreparedClothTopology, ClothContactStatic>();
const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
const bits = (x: number) => {f32[0] = x; return u32[0]!;};

function csr(rows: readonly (readonly number[])[]): {start: Uint32Array; list: Uint32Array} {
  const start = new Uint32Array(rows.length + 1);
  rows.forEach((r, i) => {start[i + 1] = start[i]! + r.length;});
  const list = new Uint32Array(start[rows.length]!);
  rows.forEach((r, i) => list.set(r, start[i]!));
  return {start, list};
}

/** Rod contact statics of ClothContactSet (lib/cloth/contact-constraints.ts):
 * each vertex's owning segment (the one it starts, or its rod's last; NONE off
 * rods), per segment (radius, rod friction), each vertex's rod neighbours, and
 * the same-rod pairs overlapping in the rest shape (one continuous yarn, never
 * a contact): per vertex the segments of its rod within ri + r of it, per
 * segment the later segments of its rod within the two radii. */
function rodContactStatic(topology: PreparedClothTopology): {owner: Uint32Array; segInfo: Uint32Array; neighbours: ReturnType<typeof csr>; pointExcluded: ReturnType<typeof csr>; segmentExcluded: ReturnType<typeof csr>} {
  const n = topology.vertexCount, S = topology.segmentCount, rest = topology.data.restPositions_m, NONE = 0xffffffff;
  const owner = new Uint32Array(n).fill(NONE), segInfo = new Uint32Array(2 * S), segRod = new Int32Array(S), segA = new Uint32Array(S), segB = new Uint32Array(S), radius = new Float64Array(S);
  const neighbours: number[][] = Array.from({length: n}, () => []);
  topology.rods.forEach((rod, r) => {
    const count = rod.restLengths_m.length, m = topology.materials[rod.material]!;
    for (let j = 0; j < count; j++) {
      const a = rod.vertices[j]!, b = rod.vertices[j + 1]!, s = rod.segmentOffset + j;
      segRod[s] = r; segA[s] = a; segB[s] = b; radius[s] = m.rodRadius_m;
      segInfo[2 * s] = bits(m.rodRadius_m); segInfo[2 * s + 1] = bits(m.friction);
      for (const [v, w] of [[a, b], [b, a]] as const) if (!neighbours[v]!.includes(w)) neighbours[v]!.push(w);
    }
    for (let k = 0; k < rod.vertices.length; k++) if (owner[rod.vertices[k]!] === NONE) owner[rod.vertices[k]!] = rod.segmentOffset + Math.min(k, count - 1);
  });
  const point = (i: number): ClothVector => [rest[3*i]!, rest[3*i + 1]!, rest[3*i + 2]!];
  const gap = (shape: RodContactShape, ids: readonly number[]) => {const d = rodSelfClosest(shape, ids.map(point)).delta; return Math.hypot(d[0], d[1], d[2]);};
  const points: number[][] = Array.from({length: n}, () => []), segments: number[][] = Array.from({length: S}, () => []);
  if (S) {
    let reach = 0;
    for (let i = 0; i < n; i++) if (owner[i] !== NONE) reach = Math.max(reach, clothVertexContactRadius(topology, i));
    for (let s = 0; s < S; s++) reach = Math.max(reach, radius[s]!);
    const bounds = new Float64Array(6 * S);
    for (let s = 0; s < S; s++) for (let a = 0; a < 3; a++) {const x = rest[3*segA[s]! + a]!, y = rest[3*segB[s]! + a]!; bounds[6*s + a] = Math.min(x, y) - 2 * reach - 1e-9; bounds[6*s + 3 + a] = Math.max(x, y) + 2 * reach + 1e-9;}
    const tree = new ClothContactBVH(bounds);
    for (let i = 0; i < n; i++) {
      if (owner[i] === NONE) continue;
      const q = point(i), ri = clothVertexContactRadius(topology, i);
      for (const s of tree.query(q, q)) if (segRod[s] === segRod[owner[i]!] && gap('point-segment', [i, segA[s]!, segB[s]!]) <= ri + radius[s]! + 1e-12) points[i]!.push(s);
      points[i]!.sort((a, b) => a - b);
    }
    for (let s = 0; s < S; s++) {
      for (const u of tree.query(Array.from(bounds.subarray(6*s, 6*s + 3)), Array.from(bounds.subarray(6*s + 3, 6*s + 6)))) {
        if (u > s && segRod[u] === segRod[s] && gap('segment-segment', [segA[s]!, segB[s]!, segA[u]!, segB[u]!]) <= radius[s]! + radius[u]! + 1e-12) segments[s]!.push(u);
      }
      segments[s]!.sort((a, b) => a - b);
    }
  }
  return {owner, segInfo, neighbours: csr(neighbours), pointExcluded: csr(points), segmentExcluded: csr(segments)};
}

export function clothContactStatic(topology: PreparedClothTopology): ClothContactStatic {
  const held = statics.get(topology);
  if (held) return held;
  const n = topology.vertexCount, T = topology.triangles.length, edges = clothShellCache(topology).edges, E = edges.length;
  const rest = topology.data.restPositions_m;
  // Closed one-ring: the vertices of every triangle incident to i.
  const ring: number[][] = Array.from({length: n}, (_, i) => {const s = new Set([i]); for (const f of topology.incidentTriangles[i]!) for (const v of topology.triangles[f]!.vertices) s.add(v); return [...s].sort((a, b) => a - b);});
  const ringSet = ring.map(r => new Set(r));
  // Rest patches beyond what the one-ring already excludes.
  const faceRes: number[][] = [], vertexRes: number[][] = [];
  for (let i = 0; i < n; i++) {
    if (!topology.incidentTriangles[i]!.length) {faceRes.push([]); vertexRes.push([]); continue;}
    const local = clothVertexLocalFaces(topology, i), faces: number[] = [], verts = new Set<number>();
    for (const f of local) {
      const ids = topology.triangles[f]!.vertices;
      if (!ids.some(v => ringSet[i]!.has(v))) faces.push(f);
      for (const v of ids) if (!ringSet[i]!.has(v)) verts.add(v);
    }
    faceRes.push(faces.sort((a, b) => a - b)); vertexRes.push([...verts].sort((a, b) => a - b));
  }
  // Edge pairs excluded by the edge's own rest patch (localShellEdges' second
  // clause) and by nothing cheaper: candidates are rest-close edge pairs.
  const e3: number[][] = Array.from({length: E}, () => []);
  if (E) {
    const bounds = new Float64Array(6 * E);
    edges.forEach((e, k) => {for (let a = 0; a < 3; a++) {const x = rest[3*e.vertices[0] + a]!, y = rest[3*e.vertices[1] + a]!; bounds[6*k + a] = Math.min(x, y) - e.r - 1e-12; bounds[6*k + 3 + a] = Math.max(x, y) + e.r + 1e-12;}});
    const tree = new ClothContactBVH(bounds);
    for (let ai = 0; ai < E; ai++) {
      const ea = edges[ai]!;
      for (const bi of tree.query(Array.from(bounds.subarray(6*ai, 6*ai + 3)), Array.from(bounds.subarray(6*ai + 3, 6*ai + 6)))) {
        if (bi <= ai) continue;
        const eb = edges[bi]!, ids = [...ea.vertices, ...eb.vertices];
        if (new Set(ids).size < 4) continue;
        if (ea.vertices.some(v => eb.vertices.some(w => ringSet[v]!.has(w) || vertexRes[v]!.includes(w)))) continue;
        if (clothLocalShellEdges(topology, ea, eb)) e3[ai]!.push(bi);
      }
      e3[ai]!.sort((a, b) => a - b);
    }
  }
  const rings = csr(ring), frs = csr(faceRes), lvs = csr(vertexRes), e3s = csr(e3);
  const rod = rodContactStatic(topology);
  const parts: Uint32Array[] = [];
  const vInfo = new Uint32Array(4 * n);
  for (let i = 0; i < n; i++) {
    const m = topology.materials[topology.vertexMaterial[i]!]!;
    vInfo.set([bits(clothVertexContactRadius(topology, i)), bits(m.friction), topology.incidentTriangles[i]!.length ? 1 : 0, rod.owner[i]!], 4 * i);
  }
  const edgeWords = new Uint32Array(4 * E);
  edges.forEach((e, k) => edgeWords.set([e.vertices[0], e.vertices[1], bits(e.r), 0], 4 * k));
  const faceR = Uint32Array.from(topology.triangles, f => bits(.5 * topology.materials[f.material]!.thickness_m));
  const names = ['sVInfo', 'sRingStart', 'sRingList', 'sFrStart', 'sFrList', 'sLvStart', 'sLvList', 'sEdge', 'sE3Start', 'sE3List', 'sFaceR',
    'sSegInfo', 'sNbStart', 'sNbList', 'sRpXStart', 'sRpXList', 'sRsXStart', 'sRsXList'] as const;
  parts.push(vInfo, rings.start, rings.list, frs.start, frs.list, lvs.start, lvs.list, edgeWords, e3s.start, e3s.list, faceR,
    rod.segInfo, rod.neighbours.start, rod.neighbours.list, rod.pointExcluded.start, rod.pointExcluded.list, rod.segmentExcluded.start, rod.segmentExcluded.list);
  const offsets = {} as Record<typeof names[number], number>;
  let total = 0;
  parts.forEach((part, k) => {offsets[names[k]!] = total; total += part.length;});
  const words = new Uint32Array(Math.max(4, total));
  parts.forEach((part, k) => words.set(part, offsets[names[k]!]));
  // Grid cell: twice the largest rest face extent with its contact support.
  let extent = 0, radius = 0;
  for (let i = 0; i < n; i++) radius = Math.max(radius, clothVertexContactRadius(topology, i));
  for (let t = 0; t < T; t++) {
    const ids = topology.triangles[t]!.vertices;
    for (let a = 0; a < 3; a++) {const c = ids.map(v => rest[3*v + a]!); extent = Math.max(extent, Math.max(...c) - Math.min(...c));}
  }
  // Rod segments are swept primitives of the same grid.
  for (const r of topology.rods) for (let j = 0; j + 1 < r.vertices.length; j++) for (let a = 0; a < 3; a++) extent = Math.max(extent, Math.abs(rest[3*r.vertices[j + 1]! + a]! - rest[3*r.vertices[j]! + a]!));
  const cellSize = 2 * (extent + 4 * radius) || 1e-2;
  const result = {words, edges: E, cellSize, offsets};
  statics.set(topology, result);
  return result;
}

// ---- colliders, in frame F (relative to `frame`), packed after the static words
export const COLLIDER_WORDS = 24;
export const KIND_PLANE = 0, KIND_SPHERE = 1, KIND_BOX = 2, KIND_BRICKS = 3;
const split = (x: number): [number, number] => {const h = Math.fround(x); return [h, x - h];};

export function packClothColliders(colliders: readonly ClothCollider[], frame: ClothVector, base: number): Uint32Array<ArrayBuffer> {
  const header = new Uint32Array(COLLIDER_WORDS * colliders.length), floats = new Float32Array(header.buffer);
  const tail: number[] = [];
  let at = base + header.length;
  colliders.forEach((c, k) => {
    const o = COLLIDER_WORDS * k, velocity = c.kind === 'sampled' || c.kind === 'rigid' ? [0, 0, 0] : c.velocity_m_s ?? [0, 0, 0];
    floats[o + 1] = c.friction ?? -1;
    floats.set(velocity, o + 2);
    if (c.kind === 'plane') {
      const length = Math.hypot(...c.normal), n = c.normal.map(v => v / length), off = c.offset_m - (n[0]! * frame[0] + n[1]! * frame[1] + n[2]! * frame[2]);
      header[o] = KIND_PLANE; floats[o + 5] = Math.hypot(...velocity); floats[o + 6] = 1;
      floats.set(n, o + 8); floats.set(split(off), o + 11);
    } else if (c.kind === 'sphere') {
      header[o] = KIND_SPHERE; floats[o + 5] = Math.hypot(...velocity); floats[o + 6] = 1;
      for (let a = 0; a < 3; a++) {const [h, l] = split(c.center_m[a]! - frame[a]!); floats[o + 8 + a] = h; floats[o + 11 + a] = l;}
      floats.set(split(c.radius_m), o + 14);
    } else if (c.kind === 'box') {
      header[o] = KIND_BOX; floats[o + 5] = Math.hypot(...velocity); floats[o + 6] = 1;
      for (let a = 0; a < 3; a++) {
        const [h, l] = split(c.minimum_m[a]! - frame[a]!), [H, L] = split(c.maximum_m[a]! - frame[a]!);
        floats[o + 8 + a] = h; floats[o + 11 + a] = l; floats[o + 14 + a] = H; floats[o + 17 + a] = L;
      }
    } else if (c.kind === 'sampled') {
      const data = clothStaticColliderRecords.get(c);
      if (!data || data.cuts) throw new Error('GPU cloth contact: unsupported sampled collider');
      header[o] = KIND_BRICKS; floats[o + 5] = c.maximumSpeed_m_s; floats[o + 6] = c.spatialLipschitz ?? 2;
      const h = data.h, records = data.records;
      // Records: B_hi(3) B_lo(3) size(3) scale valueIndex pad...; then the BVH nodes; then values.
      const RECORD = 12, NODE = 8;
      interface Tree {min: number[]; max: number[]; items?: number[]; children?: [Tree, Tree]}
      const build = (items: number[]): Tree => {
        const min = [0, 1, 2].map(a => Math.min(...items.map(r => records[r]!.min[a]!))), max = [0, 1, 2].map(a => Math.max(...items.map(r => records[r]!.max[a]!)));
        if (items.length <= 8) return {min, max, items};
        const lengths = max.map((v, a) => v - min[a]!), axis = lengths.indexOf(Math.max(...lengths));
        items.sort((a, b) => records[a]!.min[axis]! + records[a]!.max[axis]! - records[b]!.min[axis]! - records[b]!.max[axis]!);
        const mid = items.length >> 1;
        return {min, max, children: [build(items.slice(0, mid)), build(items.slice(mid))]};
      };
      // Flattened with each node's two children adjacent (left first).
      const nodes: {min: number[]; max: number[]; first: number; count: number}[] = [], order: number[] = [];
      const place = (tree: Tree, index: number) => {
        const node = nodes[index] = {min: tree.min, max: tree.max, first: 0, count: 0};
        if (tree.items) {node.first = order.length; node.count = tree.items.length; order.push(...tree.items); return;}
        node.first = nodes.length; nodes.push(node, node);
        place(tree.children![0], node.first); place(tree.children![1], node.first + 1);
      };
      if (records.length) {nodes.push({min: [], max: [], first: 0, count: 0}); place(build(records.map((_, k) => k)), 0);}
      const recordBase = at, nodeBase = recordBase + RECORD * order.length, valueBase = nodeBase + NODE * nodes.length;
      const recordWords = new Uint32Array(RECORD * order.length), recordFloats = new Float32Array(recordWords.buffer);
      let values = 0;
      const valueList: number[] = [];
      order.forEach((r, k) => {
        const rec = records[r]!;
        for (let a = 0; a < 3; a++) {const [hh, ll] = split(rec.min[a]! - frame[a]!); recordFloats[RECORD*k + a] = hh; recordFloats[RECORD*k + 3 + a] = ll; recordFloats[RECORD*k + 6 + a] = rec.max[a]! - rec.min[a]!;}
        recordFloats[RECORD*k + 9] = h / rec.lipschitz;
        if (rec.values) {recordWords[RECORD*k + 10] = valueBase + values; valueList.push(...rec.values); values += rec.values.length;}
        else recordWords[RECORD*k + 10] = 0xffffffff;
      });
      const nodeWords = new Uint32Array(NODE * nodes.length), nodeFloats = new Float32Array(nodeWords.buffer);
      nodes.forEach((node, k) => {
        // Conservative Float32 bounds relative to F.
        for (let a = 0; a < 3; a++) {
          const lo = node.min[a]! - frame[a]!, hi = node.max[a]! - frame[a]!, pad = 1e-6 * (Math.abs(lo) + Math.abs(hi) + 1e-3);
          nodeFloats[NODE*k + a] = lo - pad; nodeFloats[NODE*k + 3 + a] = hi + pad;
        }
        nodeWords[NODE*k + 6] = node.count ? recordBase + RECORD * node.first : nodeBase + NODE * node.first;
        nodeWords[NODE*k + 7] = node.count;
      });
      floats[o + 8] = h; header[o + 9] = nodes.length ? nodeBase : 0xffffffff; header[o + 10] = order.length;
      for (const w of recordWords) tail.push(w);
      for (const w of nodeWords) tail.push(w);
      const valueFloats = Float32Array.from(valueList);
      for (const w of new Uint32Array(valueFloats.buffer)) tail.push(w);
      at = valueBase + values;
    } else throw new Error(`GPU cloth contact: unsupported collider ${(c as {kind: string}).kind}`);
  });
  const out = new Uint32Array(header.length + tail.length);
  out.set(header); out.set(tail, header.length);
  return out;
}

// ---- WGSL: control words follow the Newton solver's (see gpu-newton.ts).
export const CONTACT_CONTROL = {
  K_COUNT: 96, K_BASE: 97, K_ENTRIES: 98, K_ERROR: 99, K_DONE: 100, K_OUTER: 101, K_KKT: 102, K_FR: 103, K_PEN: 104, K_PREV: 105, K_RESID: 106,
  R_ACTIVE: 107, R_CONTINUE: 108, R_STIFFEN: 109, K_CONTACTS: 110, K_SELF: 111, K_REPPEN: 112, K_SOLVES: 113, K_ADDED: 114, K_DETECT: 115,
  // The first primitive whose swept box breaks the two-cells-per-axis grid bound: (grid << 30) | (primitive + 1), and its widest box extent (f32 bits).
  K_SPAN: 116, K_SPAN_EXTENT: 117,
} as const;
export const CONTACT_CONTROL_WORDS = 120;
export const E_CAPACITY = 1, E_GRID = 2, E_CROSSED = 4, E_STIFFNESS = 8, E_TABLE = 16, E_CELLS = 32, E_NONFINITE = 64, E_UNSUPPORTED = 128, E_SPAN = 1024;
/** physics.contactGridEntries authored as this: 8 (triangles + edges + segments)
 * grid entries, valid while every swept, padded primitive box is narrower than
 * the grid cell on each axis (at most 2 cells, so at most 8 entries); detection
 * checks that premise and a box that breaks it fails the step (E_SPAN). */
export const CONTACT_GRID_CELL_BOUND = 'two-cells-per-axis';
/** Cells one primitive may cover before detection reports it (it is not dropped). */
export const CELL_CAP = 512;
export const SCAN_WG = 256;

export function clothContactWGSL(WG: number, S_CONVERGED: number, C: {C_STATUS: number; C_NEWTON: number; C_ACTIVE: number; C_CGACTIVE: number; C_NEWTONMIN: number}, maximumOuter: number): string {
  const K = CONTACT_CONTROL;
  return /* wgsl */ `
const NONE:u32=0xffffffffu;
fn cw(k:u32)->u32{return atomicLoad(&C[k]);}
fn ksf(k:u32)->f32{return bitcast<f32>(I[k]);}
fn kRadius(i:u32)->f32{return ksf(p.sVInfo+4u*i);}
fn kFriction(i:u32)->f32{return ksf(p.sVInfo+4u*i+1u);}
fn kShell(i:u32)->bool{return I[p.sVInfo+4u*i+2u]!=0u;}
// Rod statics: a vertex's owning segment (NONE off rods); a rod vertex is a sphere of its rod (not on the shell).
fn kOwner(i:u32)->u32{return I[p.sVInfo+4u*i+3u];}
fn rodVertex(i:u32)->bool{return kOwner(i)!=NONE&&!kShell(i);}
fn segR(s:u32)->f32{return ksf(p.sSegInfo+2u*s);}
fn segFriction(s:u32)->f32{return ksf(p.sSegInfo+2u*s+1u);}
fn contactCount()->u32{if(p.contactMode==2u){return p.uploadedContacts;}return atomicLoad(&C[${K.K_COUNT}u]);}
fn roundOn()->bool{return atomicLoad(&C[${K.R_ACTIVE}u])!=0u;}
fn maxf(word:u32,v:f32){if(v>0.){atomicMax(&C[word],bitcast<u32>(v));}else if(!(v<=0.)){atomicOr(&C[${K.K_ERROR}u],${E_NONFINITE}u);}}
// Conservative advancement in Float32 keeps moving where the reference's
// minimum step falls below one ulp of the time.
fn advance(t:f32,step:f32)->f32{let next=t+step;if(next>t){return min(1.,next);}return min(1.,bitcast<f32>(bitcast<u32>(t)+1u));}

// ---- key table: KA[aTable + 2 S t + 2 s] = (k0+1, k1), KU[uTabVal + S t + s] = lambda bits
fn hashKey(k0:u32,k1:u32)->u32{var h=(k0*0x9E3779B1u)^((k1+0x7F4A7C15u)*0x85EBCA77u);h^=h>>16u;h*=0x7FEB352Du;h^=h>>15u;h*=0x846CA68Bu;h^=h>>16u;return h;}
fn tableFind(t:u32,k0:u32,k1:u32)->u32{
 let mask=p.tableSlots-1u;let base=p.aTable+2u*p.tableSlots*t;var s=hashKey(k0,k1)&mask;
 for(var probe=0u;probe<p.tableSlots;probe++){let w0=atomicLoad(&KA[base+2u*s]);if(w0==0u){return NONE;}if(w0==k0+1u&&atomicLoad(&KA[base+2u*s+1u])==k1){return s;}s=(s+1u)&mask;}
 return NONE;
}
// Claims a slot for a key no other thread inserts in this pass; NONE if held.
fn tableClaim(t:u32,k0:u32,k1:u32)->u32{
 let mask=p.tableSlots-1u;let base=p.aTable+2u*p.tableSlots*t;var s=hashKey(k0,k1)&mask;
 for(var probe=0u;probe<2u*p.tableSlots;){
  var w0=atomicLoad(&KA[base+2u*s]);
  if(w0==0u){let r=atomicCompareExchangeWeak(&KA[base+2u*s],0u,k0+1u);if(r.exchanged){atomicStore(&KA[base+2u*s+1u],k1);return s;}w0=r.old_value;if(w0==0u){continue;}}
  // A slot claimed in this pass may not show its second word yet; its key differs.
  if(w0==k0+1u&&atomicLoad(&KA[base+2u*s+1u])==k1){return NONE;}
  s=(s+1u)&mask;probe++;
 }
 atomicOr(&C[${K.K_ERROR}u],${E_TABLE}u);return NONE;
}
fn held(k0:u32,k1:u32)->bool{return tableFind(p.tableCur,k0,k1)!=NONE;}
// A new pair's slot (NONE if its key is held or the pool is full), with its key, collider and warm-start multiplier written.
fn claimContact(k0:u32,k1:u32,collider:u32)->u32{
 // A full pool fails before claiming a key: claims past capacity would fill the
 // table (2 kMax slots), and every later claim would then probe all of it.
 if(atomicLoad(&C[${K.K_COUNT}u])>=p.kMax){atomicOr(&C[${K.K_ERROR}u],${E_CAPACITY}u);return NONE;}
 let slot=tableClaim(p.tableCur,k0,k1);if(slot==NONE){return NONE;}
 let c=atomicAdd(&C[${K.K_COUNT}u],1u);if(c>=p.kMax){atomicOr(&C[${K.K_ERROR}u],${E_CAPACITY}u);return NONE;}
 KU[p.kuMeta+4u*c]=k0;KU[p.kuMeta+4u*c+1u]=k1;KU[p.kuMeta+4u*c+2u]=collider;KU[p.kuMeta+4u*c+3u]=slot;
 return c;
}
fn warmLambda(k0:u32,k1:u32)->f32{
 var lambda=0.;
 if(p.prevValid!=0u){let t=1u-p.tableCur;let ps=tableFind(t,k0,k1);if(ps!=NONE){lambda=bitcast<f32>(KU[p.uTabVal+p.tableSlots*t+ps]);}}
 return lambda;
}
fn writeContact(c:u32,lambda:f32,v:vec4u,w:vec4f,n:vec3f,g0:f32,obstacle:vec3f,mu:f32,radius:f32){
 let tangentObstacle=obstacle-dot(obstacle,n)*n;
 KF[4u*c]=vec4f(n,g0);KF[4u*c+1u]=vec4f(tangentObstacle,lambda);KF[4u*c+2u]=vec4f(0.,lambda,mu,radius);KF[4u*c+3u]=w;
 KU[p.kuCV+4u*c]=v.x;KU[p.kuCV+4u*c+1u]=v.y;KU[p.kuCV+4u*c+2u]=v.z;KU[p.kuCV+4u*c+3u]=v.w;
}
fn addContact(k0:u32,k1:u32,v:vec4u,w:vec4f,n:vec3f,g0:f32,obstacle:vec3f,mu:f32,radius:f32,collider:u32){
 let c=claimContact(k0,k1,collider);if(c==NONE){return;}
 writeContact(c,warmLambda(k0,k1),v,w,n,g0,obstacle,mu,radius);
 if(p.rodContact!=0u){KU[p.kuRod+4u*c+3u]=0u;}
 atomicAdd(&C[${K.K_ADDED}u],1u);
}
@compute @workgroup_size(${WG}) fn tableClear(@builtin(global_invocation_id) g:vec3u,@builtin(num_workgroups) ng:vec3u){
 if(atomicLoad(&C[${K.K_DETECT}u])==0u){return;}
 let base=p.aTable+2u*p.tableSlots*p.tableCur;for(var s=g.x;s<2u*p.tableSlots;s+=ng.x*${WG}u){atomicStore(&KA[base+s],0u);}
}
// After the step: each held pair's multiplier, for the next step's warm start.
@compute @workgroup_size(${WG}) fn rememberMultipliers(@builtin(global_invocation_id) g:vec3u,@builtin(num_workgroups) ng:vec3u){
 // Only after the contact rounds succeeded (encoded in every submission).
 if(atomicLoad(&C[${K.K_DONE}u])==0u||atomicLoad(&C[${K.K_ERROR}u])!=0u||atomicLoad(&C[${C.C_STATUS}u])!=${S_CONVERGED}u){return;}
 let count=min(contactCount(),p.kMax);for(var c=g.x;c<count;c+=ng.x*${WG}u){KU[p.uTabVal+p.tableSlots*p.tableCur+KU[p.kuMeta+4u*c+3u]]=bitcast<u32>(max(0.,KF[4u*c+1u].w));}
}

// ---- points in frame F: hi + (lo + motion), so differences keep the gap's precision
struct P2{h:vec3f,l:vec3f}
fn sweptPoint(i:u32,t:f32)->P2{return P2(hi(i).xyz,lo(i).xyz+t*xv(i));}
fn shifted(q:P2,d:vec3f)->P2{return P2(q.h,q.l+d);}
fn approx(q:P2)->vec3f{return q.h+q.l;}
struct Sample{d:f32,n:vec3f}
fn cHeader(ci:u32)->u32{return p.sCollider+${COLLIDER_WORDS}u*ci;}
fn cv3(b:u32)->vec3f{return vec3f(ksf(b),ksf(b+1u),ksf(b+2u));}
fn boxDistance(x:vec3f,size:vec3f)->f32{
 let d=x-clamp(x,vec3f(0),size);let outside=length(d);if(outside>1e-14){return outside;}
 return -min(min(min(x.x,size.x-x.x),min(x.y,size.y-x.y)),min(x.z,size.z-x.z));
}
fn trilinear(base:u32,q:vec3f)->vec4f{
 let c=min(vec3f(7),max(vec3f(0),floor(q)));let f=q-c;let ci=vec3u(c);var out=vec4f(0);
 for(var k=0u;k<8u;k++){
  let b=vec3u(k&1u,(k>>1u)&1u,k>>2u);let w=select(vec3f(1)-f,f,b==vec3u(1));
  let v=ksf(base+ci.x+b.x+9u*(ci.y+b.y+9u*(ci.z+b.z)));
  out.w+=v*w.x*w.y*w.z;out.x+=v*select(-1.,1.,b.x==1u)*w.y*w.z;out.y+=v*select(-1.,1.,b.y==1u)*w.z*w.x;out.z+=v*select(-1.,1.,b.z==1u)*w.x*w.y;
 }
 return out;
}
// The static solid: min over brick and fill records (box <= 0 ? value : max(box, value - box)).
// With the containing brick's raw trilinear gradient, when one contains q.
struct Brick{d:f32,gradient:vec3f,inside:bool}
fn bricks(header:u32,q:P2,wantGradient:bool)->Brick{
 let h=ksf(header+8u);let root=I[header+9u];var best=3e38;var gradient=vec3f(0);var inside=false;
 if(root==NONE){return Brick(best,gradient,false);}
 var stack:array<u32,48>;var top=1u;stack[0]=root;let x=approx(q);
 while(top>0u){
  top--;let node=stack[top];let lower=max(0.,boxDistance(x-cv3(node),cv3(node+3u)-cv3(node)));if(lower>max(0.,best)){continue;}
  let first=I[node+6u];let count=I[node+7u];
  if(count==0u){if(top+2u>48u){atomicOr(&C[${K.K_ERROR}u],${E_GRID}u);break;}stack[top]=first+8u;stack[top+1u]=first;top+=2u;continue;}
  for(var r=0u;r<count;r++){
   let rec=first+12u*r;let local=(q.h-cv3(rec))+(q.l-cv3(rec+3u));let size=cv3(rec+6u);let box=boxDistance(local,size);let values=I[rec+10u];
   if(values==NONE){best=min(best,box);continue;}
   let s=local/h;let sample=trilinear(values,clamp(s,vec3f(0),vec3f(8)));let value=sample.w*ksf(rec+9u);
   if(box<=0.){best=min(best,value);}else{best=min(best,max(box,value-box));}
   if(wantGradient&&all(s>=vec3f(0))&&all(s<vec3f(8))){gradient=sample.xyz;inside=true;}
  }
 }
 return Brick(best,gradient,inside);
}
fn colliderDistance(ci:u32,q0:P2,beforeEnd:f32)->f32{
 let b=cHeader(ci);let kind=I[b];let q=shifted(q0,cv3(b+2u)*beforeEnd);
 if(kind==${KIND_PLANE}u){let n=cv3(b+8u);return (dot(n,q.h)-ksf(b+11u))+(dot(n,q.l)-ksf(b+12u));}
 if(kind==${KIND_SPHERE}u){let d=(q.h-cv3(b+8u))+(q.l-cv3(b+11u));return (length(d)-ksf(b+14u))-ksf(b+15u);}
 if(kind==${KIND_BOX}u){
  let below=(q.h-cv3(b+8u))+(q.l-cv3(b+11u));let above=(cv3(b+14u)-q.h)+(cv3(b+17u)-q.l);
  let d=select(vec3f(0),below,below<vec3f(0))-select(vec3f(0),above,above<vec3f(0));let outside=length(d);if(outside>1e-14){return outside;}
  return -min(min(min(below.x,above.x),min(below.y,above.y)),min(below.z,above.z));
 }
 return bricks(b,q,false).d;
}
fn colliderSample(ci:u32,q0:P2,beforeEnd:f32)->Sample{
 let b=cHeader(ci);let kind=I[b];let q=shifted(q0,cv3(b+2u)*beforeEnd);
 if(kind==${KIND_PLANE}u){return Sample(colliderDistance(ci,q0,beforeEnd),cv3(b+8u));}
 if(kind==${KIND_SPHERE}u){let d=(q.h-cv3(b+8u))+(q.l-cv3(b+11u));let l=length(d);var n=vec3f(0,1,0);if(l>1e-14){n=d/l;}return Sample((l-ksf(b+14u))-ksf(b+15u),n);}
 if(kind==${KIND_BOX}u){
  let below=(q.h-cv3(b+8u))+(q.l-cv3(b+11u));let above=(cv3(b+14u)-q.h)+(cv3(b+17u)-q.l);
  let d=select(vec3f(0),below,below<vec3f(0))-select(vec3f(0),above,above<vec3f(0));let outside=length(d);if(outside>1e-14){return Sample(outside,d/outside);}
  var pen=3e38;var axis=0u;var sign=1.;
  for(var a=0u;a<3u;a++){if(below[a]<pen){pen=below[a];axis=a;sign=-1.;}if(above[a]<pen){pen=above[a];axis=a;sign=1.;}}
  var n=vec3f(0);n[axis]=sign;return Sample(-pen,n);
 }
 let s=bricks(b,q,true);var gradient=s.gradient;
 if(!s.inside){
  // Outside every brick: central differences of the same distance.
  let e=ksf(b+8u)*1e-4;
  for(var a=0u;a<3u;a++){var d=vec3f(0);d[a]=e;gradient[a]=(bricks(b,shifted(q,d),false).d-bricks(b,shifted(q,-d),false).d)/(2.*e);}
 }
 let l=length(gradient);var n=vec3f(0,1,0);if(l>1e-12){n=gradient/l;}
 return Sample(s.d,n);
}

// ---- outer iteration: multipliers, collider tangent planes, detection
@compute @workgroup_size(1) fn roundBegin(){
 // Begin a contact iteration after a converged inner solve.
 let begin=atomicLoad(&C[${C.C_STATUS}u])==${S_CONVERGED}u&&atomicLoad(&C[${K.K_DONE}u])==0u&&atomicLoad(&C[${K.K_ERROR}u])==0u&&p.contactMode==1u;
 atomicStore(&C[${K.R_ACTIVE}u],select(0u,1u,begin));atomicStore(&C[${K.R_CONTINUE}u],0u);atomicStore(&C[${K.K_ADDED}u],0u);atomicStore(&C[${K.K_DETECT}u],select(0u,1u,begin));
 if(!begin){return;}
 atomicStore(&C[${K.K_BASE}u],atomicLoad(&C[${K.K_COUNT}u]));atomicStore(&C[${K.K_ENTRIES}u],0u);
 atomicStore(&C[${K.K_KKT}u],0u);atomicStore(&C[${K.K_FR}u],0u);atomicStore(&C[${K.K_PEN}u],0u);atomicStore(&C[${K.K_REPPEN}u],0u);atomicStore(&C[${K.K_CONTACTS}u],0u);atomicStore(&C[${K.K_SELF}u],0u);
}
// The first detection of a step, at the prediction.
@compute @workgroup_size(1) fn detectBegin(){
 let on=p.contactMode==1u;atomicStore(&C[${K.K_DETECT}u],select(0u,1u,on));atomicStore(&C[${K.R_CONTINUE}u],select(0u,1u,on));
 atomicStore(&C[${K.K_COUNT}u],0u);atomicStore(&C[${K.K_BASE}u],0u);atomicStore(&C[${K.K_ENTRIES}u],0u);atomicStore(&C[${K.K_ADDED}u],0u);
 atomicStore(&C[${K.K_SPAN}u],0u);atomicStore(&C[${K.K_SPAN_EXTENT}u],0u);
}
fn contactGap(c:u32)->f32{let k=contactAt(c);return k.g0+dot(k.n,contactSum(k,0u));}
@compute @workgroup_size(${WG}) fn updateMultipliers(@builtin(global_invocation_id) g:vec3u,@builtin(num_workgroups) ng:vec3u){
 if(!roundOn()){return;}
 let count=atomicLoad(&C[${K.K_BASE}u]);
 for(var c=g.x;c<count;c+=ng.x*${WG}u){
  let k=contactAt(c);let gap=k.g0+dot(k.n,contactSum(k,0u));let next=cForce(k.lambda/k.rho-gap,k.rho);let r=KF[4u*c+2u];
  maxf(${K.K_KKT}u,abs(next-k.lambda)/k.rho);if(r.z>0.){maxf(${K.K_FR}u,r.z*abs(next-r.y)/k.rho);}
  KF[4u*c+1u].w=next;if(KU[p.kuMeta+4u*c+2u]==NONE){maxf(${K.K_REPPEN}u,-gap);}
  if(next>0.){if(KU[p.kuMeta+4u*c+2u]==NONE){atomicAdd(&C[${K.K_SELF}u],1u);}else{atomicAdd(&C[${K.K_CONTACTS}u],1u);}}
 }
}
// The reference's self re-linearization: closest points at the current positions
// (relative to the first vertex's old position), the normal kept on the side of detection.
fn relinearizeSelf(c:u32){
 let k0=KU[p.kuMeta+4u*c];let k1=KU[p.kuMeta+4u*c+1u];let j=p.kuCV+4u*c;let v=vec4u(KU[j],KU[j+1u],KU[j+2u],KU[j+3u]);
 let q1=oldEdge(v.x,v.y);let q2=oldEdge(v.x,v.z);let q3=oldEdge(v.x,v.w);
 let x0=xv(v.x);let x1=q1+xv(v.y);let x2=q2+xv(v.z);let x3=q3+xv(v.w);
 var delta:vec3f;var fallback:vec3f;var w:vec4f;var distance:f32;
 if((k0>>28u)==1u){
  let closest=closestTriangle(x0,x1,x2,x3);delta=x0-closest.point;fallback=cross(x2-x1,x3-x1);w=vec4f(1.,-closest.w.x,-closest.w.y,-closest.w.z);
  distance=kRadius(v.x)+ksf(p.sFaceR+k1);
 }else{
  let closest=closestSegments(x0,x1,x2,x3);delta=closest.a-closest.b;fallback=cross(x1-x0,x3-x2);w=vec4f(1.-closest.s,closest.s,-(1.-closest.t),-closest.t);
  distance=ksf(p.sEdge+4u*(k0&0x0FFFFFFFu)+2u)+ksf(p.sEdge+4u*k1+2u);
 }
 let old=KF[4u*c].xyz;let length0=length(delta);let f=length(fallback);
 var n=old;if(length0>1e-12){n=delta/length0;}else if(f>1e-12){n=fallback/f;}
 if(dot(n,old)<0.){n=-n;}
 KF[4u*c]=vec4f(n,dot(n,w.y*q1+w.z*q2+w.w*q3)-distance);KF[4u*c+3u]=w;
 let pen=distance-dot(n,delta);maxf(${K.K_PEN}u,pen);maxf(${K.K_REPPEN}u,pen);
}
// The crossing test (pathEntersCollider): whether vertex i's path through the step, the collider
// swept with it, enters the solid. Not the nearest normal turning over: past a cavity's medial
// plane the nearest wall is the opposite one, the path in free space. Unresolved counts as entering.
fn pathEntersSolid(i:u32,ci:u32,tolerance:f32)->bool{
 let b=cHeader(ci);let speed=(length(xv(i))+p.dt*ksf(b+5u))*ksf(b+6u);var time=0.;
 for(var it=0u;it<256u;it++){
  let distance=colliderDistance(ci,sweptPoint(i,time),(1.-time)*p.dt);
  if(distance<=tolerance){return true;}
  if(time>=1.||speed==0.){return false;}
  time=advance(time,max(1e-9,.9*distance/speed));
 }
 return true;
}
@compute @workgroup_size(${WG}) fn relinearize(@builtin(global_invocation_id) g:vec3u,@builtin(num_workgroups) ng:vec3u){
 if(!roundOn()){return;}
 let count=atomicLoad(&C[${K.K_BASE}u]);
 for(var c=g.x;c<count;c+=ng.x*${WG}u){
  let ci=KU[p.kuMeta+4u*c+2u];let kind=KU[p.kuMeta+4u*c]>>28u;
  if(ci==NONE){if(kind<=2u){relinearizeSelf(c);}else{relinearizeRodSelf(c);}continue;}
  if(kind==3u){relinearizeRodSegment(c,ci);continue;}
  let i=KU[p.kuCV+4u*c];let s=colliderSample(ci,sweptPoint(i,1.),0.);let radius=KF[4u*c+2u].w;
  if(pathEntersSolid(i,ci,max(1e-10,radius*1e-6))){atomicOr(&C[${K.K_ERROR}u],${E_CROSSED}u);}
  let obstacle=cv3(cHeader(ci)+2u)*p.dt;
  KF[4u*c]=vec4f(s.n,(s.d-radius)-dot(s.n,xv(i)));KF[4u*c+1u]=vec4f(obstacle-dot(obstacle,s.n)*s.n,KF[4u*c+1u].w);
  maxf(${K.K_PEN}u,radius-s.d);maxf(${K.K_REPPEN}u,radius-s.d);
  // A rod vertex's sphere: its rolling Jacobian at the new normal.
  if(p.rodContact!=0u&&KU[p.kuRod+4u*c+3u]!=0u){rodLinearize(c);}
 }
}
@compute @workgroup_size(1) fn roundDecide(){
 if(!roundOn()){return;}
 atomicStore(&C[${K.K_DETECT}u],0u);
 let added=atomicLoad(&C[${K.K_ADDED}u]);let residual=max(bitcast<f32>(atomicLoad(&C[${K.K_KKT}u])),max(bitcast<f32>(atomicLoad(&C[${K.K_FR}u])),bitcast<f32>(atomicLoad(&C[${K.K_PEN}u]))));
 atomicStore(&C[${K.K_RESID}u],bitcast<u32>(residual));
 let rounds=atomicLoad(&C[${K.K_OUTER}u]);
 if((added==0u&&residual<p.tolerance)||rounds>=${maximumOuter}u||atomicLoad(&C[${K.K_ERROR}u])!=0u){atomicStore(&C[${K.K_DONE}u],1u);return;}
 let previous=bitcast<f32>(atomicLoad(&C[${K.K_PREV}u]));
 atomicStore(&C[${K.R_STIFFEN}u],select(0u,1u,added==0u&&rounds>0u&&residual>.25*previous));atomicStore(&C[${K.K_PREV}u],bitcast<u32>(residual));
 atomicStore(&C[${K.K_OUTER}u],rounds+1u);atomicStore(&C[${K.R_CONTINUE}u],1u);atomicAdd(&C[${K.K_SOLVES}u],1u);
 // Restart the inner solve from this iterate, taking at least one Newton step.
 atomicStore(&C[${C.C_STATUS}u],0u);atomicStore(&C[${C.C_NEWTON}u],0u);atomicStore(&C[${C.C_ACTIVE}u],0u);atomicStore(&C[${C.C_CGACTIVE}u],0u);atomicStore(&C[${C.C_NEWTONMIN}u],1u);
 for(var k=0u;k<4u;k++){W[p.wScal+k]=vec4f(0);}
}
// New pairs' penalties from the coordinate stiffness without contact; on a
// stalled iteration rho x10 where |min(g, lambda/rho)| exceeds the tolerance;
// friction lags to the new normal force.
@compute @workgroup_size(${WG}) fn penalties(@builtin(global_invocation_id) g:vec3u,@builtin(num_workgroups) ng:vec3u){
 let again=p.contactMode==1u&&atomicLoad(&C[${K.R_CONTINUE}u])!=0u;
 if(!again){return;}
 let stiffen=again&&atomicLoad(&C[${K.R_STIFFEN}u])!=0u;let count=min(contactCount(),p.kMax);
 for(var c=g.x;c<count;c+=ng.x*${WG}u){
  var r=KF[4u*c+2u];let k=contactAt(c);
  if(r.x==0.){
   var compliance=0.;
   for(var j=0u;j<5u;j++){let w=slotW(k,j);let v=slotV(k,j);if(w==0.||isFixed(v)){continue;}var s=vec3f(pred(v).w+attach(v).w);if(p.stiffnessValid!=0u||atomicLoad(&C[${K.K_OUTER}u])>0u){s=W[p.wK+v].xyz;}compliance+=w*w/dot(k.n*k.n,s);}
   if(!(compliance>0.&&compliance<3e38)){atomicOr(&C[${K.K_ERROR}u],${E_STIFFNESS}u);}
   r.x=10./compliance;
  }else if(stiffen){let gap=k.g0+dot(k.n,contactSum(k,0u));if(abs(min(gap,k.lambda/r.x))>p.tolerance){r.x*=10.;}}
  r.y=KF[4u*c+1u].w;
  KF[4u*c+2u]=r;
 }
}

// ---- per-vertex contact lists (CSR over free vertices with nonzero weight)
// A self pair lists all four vertices (its re-linearized weights may move off zero); a collider pair its one;
// a rod pair every slot (its rolling endpoints move it through the slip Jacobian), and the rows
// p.n + segment of its sides' free twists (entries 5 + side).
fn pairVertices(c:u32,k:Contact)->u32{if(k.rodm!=0u){return k.rodm&7u;}return select(1u,4u,KU[p.kuMeta+4u*c+2u]==NONE);}
fn twistRow(c:u32,k:Contact,side:u32)->u32{if(k.rodm==0u){return NONE;}let s=KU[p.kuRod+4u*c+1u+side];if(s==NONE||!twistFree(s)){return NONE;}return p.n+s;}
@compute @workgroup_size(${WG}) fn listCount(@builtin(global_invocation_id) g:vec3u,@builtin(num_workgroups) ng:vec3u){
 if(atomicLoad(&C[${K.K_ADDED}u])==0u){return;}
 let count=min(contactCount(),p.kMax);
 for(var c=g.x;c<count;c+=ng.x*${WG}u){
  let k=contactAt(c);for(var j=0u;j<pairVertices(c,k);j++){let v=slotV(k,j);if(!isFixed(v)){atomicAdd(&KA[p.aCount+v],1u);}}
  for(var side=0u;side<2u;side++){let row=twistRow(c,k,side);if(row!=NONE){atomicAdd(&KA[p.aCount+row],1u);}}
 }
}
var<workgroup> scanBuffer:array<u32,${SCAN_WG}>;
fn scanLocal(l:u32,value:u32)->u32{
 scanBuffer[l]=value;
 for(var o=1u;o<${SCAN_WG}u;o<<=1u){workgroupBarrier();var add=0u;if(l>=o){add=scanBuffer[l-o];}workgroupBarrier();scanBuffer[l]+=add;}
 workgroupBarrier();let inclusive=scanBuffer[l];workgroupBarrier();return inclusive;
}
@compute @workgroup_size(${SCAN_WG}) fn listScan1(@builtin(global_invocation_id) g:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) wg:vec3u){
 if(uniformWord(l,${K.K_ADDED}u)==0u){return;}
 let i=g.x;var value=0u;if(i<p.listRows){value=atomicExchange(&KA[p.aCount+i],0u);}
 let inclusive=scanLocal(l,value);if(i<p.listRows){KU[p.kvStart+i]=inclusive-value;}
 if(l==${SCAN_WG - 1}u){KU[p.uBlocks+wg.x]=inclusive;}
}
@compute @workgroup_size(${SCAN_WG}) fn listScan2(@builtin(local_invocation_index) l:u32){
 if(uniformWord(l,${K.K_ADDED}u)==0u){return;}
 let blocks=(p.listRows+${SCAN_WG - 1}u)/${SCAN_WG}u;var carry=0u;
 for(var b0=0u;b0<blocks;b0+=${SCAN_WG}u){
  var value=0u;if(b0+l<blocks){value=KU[p.uBlocks+b0+l];}
  let inclusive=scanLocal(l,value);if(b0+l<blocks){KU[p.uBlocks+b0+l]=carry+inclusive-value;}
  carry+=workgroupUniformLoad(&scanBuffer[${SCAN_WG - 1}u]);
 }
 if(l==0u){KU[p.kvStart+p.listRows]=carry;}
}
@compute @workgroup_size(${SCAN_WG}) fn listScan3(@builtin(global_invocation_id) g:vec3u,@builtin(workgroup_id) wg:vec3u){
 if(atomicLoad(&C[${K.K_ADDED}u])==0u){return;}
 let i=g.x;if(i<p.listRows){KU[p.kvStart+i]+=KU[p.uBlocks+wg.x];}
}
@compute @workgroup_size(${WG}) fn listFill(@builtin(global_invocation_id) g:vec3u,@builtin(num_workgroups) ng:vec3u){
 if(atomicLoad(&C[${K.K_ADDED}u])==0u){return;}
 let count=min(contactCount(),p.kMax);
 for(var c=g.x;c<count;c+=ng.x*${WG}u){
  let k=contactAt(c);for(var j=0u;j<pairVertices(c,k);j++){let v=slotV(k,j);if(!isFixed(v)){KU[p.kvList+KU[p.kvStart+v]+atomicAdd(&KA[p.aCount+v],1u)]=(c<<3u)|j;}}
  for(var side=0u;side<2u;side++){let row=twistRow(c,k,side);if(row!=NONE){KU[p.kvList+KU[p.kvStart+row]+atomicAdd(&KA[p.aCount+row],1u)]=(c<<3u)|(5u+side);}}
 }
}
// Each vertex's list in the order of its contacts' keys (unique in a pass) and slots, not
// the order threads appended them: the per-vertex contact sums (gradient, preconditioner,
// matvec gather) and the line search's owner sums then repeat bit for bit.
fn incidenceBefore(a:u32,b:u32)->bool{
 let ca=p.kuMeta+4u*(a>>3u);let cb=p.kuMeta+4u*(b>>3u);
 if(KU[ca]!=KU[cb]){return KU[ca]<KU[cb];}if(KU[ca+1u]!=KU[cb+1u]){return KU[ca+1u]<KU[cb+1u];}return (a&7u)<(b&7u);
}
@compute @workgroup_size(${WG}) fn listReset(@builtin(global_invocation_id) g:vec3u){
 if(atomicLoad(&C[${K.K_ADDED}u])==0u){return;}
 let i=g.x;if(i>=p.listRows){return;}
 atomicStore(&KA[p.aCount+i],0u);
 let s=KU[p.kvStart+i];let t=KU[p.kvStart+i+1u];
 for(var k=s+1u;k<t;k++){let e=KU[p.kvList+k];var j=k;while(j>s&&incidenceBefore(e,KU[p.kvList+j-1u])){KU[p.kvList+j]=KU[p.kvList+j-1u];j--;}KU[p.kvList+j]=e;}
}

// ---- broadphase: a hash grid of swept faces and edges, each entry once per bucket
fn cellOf(x:vec3f)->vec3i{return vec3i(floor(x/p.cellSize));}
fn bucketOf(c:vec3i)->u32{let u=bitcast<vec3u>(c);return ((u.x*73856093u)^(u.y*19349663u)^(u.z*83492791u))&(p.gridBuckets-1u);}
fn cellCount(a:vec3i,b:vec3i)->u32{let d=vec3u(b-a)+vec3u(1);if(any(b<a)){return 0u;}if(d.x>${CELL_CAP}u||d.y>${CELL_CAP}u||d.z>${CELL_CAP}u){return ${CELL_CAP + 1}u;}return d.x*d.y*d.z;}
fn cellAt(a:vec3i,b:vec3i,k:u32)->vec3i{let d=vec3u(b-a)+vec3u(1);return a+vec3i(vec3u(k%d.x,(k/d.x)%d.y,k/(d.x*d.y)));}
fn insertEntry(grid:u32,prim:u32,a:vec3i,b:vec3i){
 let count=cellCount(a,b);if(count>${CELL_CAP}u){atomicOr(&C[${K.K_ERROR}u],${E_CELLS}u);return;}
 for(var k=0u;k<count;k++){
  let bucket=bucketOf(cellAt(a,b,k));var seen=false;for(var j=0u;j<k;j++){if(bucketOf(cellAt(a,b,j))==bucket){seen=true;break;}}if(seen){continue;}
  let e=atomicAdd(&C[${K.K_ENTRIES}u],1u);if(e>=p.gridEntries){atomicOr(&C[${K.K_ERROR}u],${E_GRID}u);return;}
  KU[p.uEntry+2u*e]=prim;KU[p.uEntry+2u*e+1u]=atomicExchange(&KA[p.aHeads+grid*p.gridBuckets+bucket],e+1u);
 }
}
// The two-cells-per-axis grid bound's premise: a box at least the cell wide on an axis fails the step, naming the first such primitive.
fn insertBox(grid:u32,prim:u32,box:array<vec3f,2>){
 if(p.gridSpanCheck!=0u){let w=box[1]-box[0];let m=max(w.x,max(w.y,w.z));if(!(m<p.cellSize)){atomicOr(&C[${K.K_ERROR}u],${E_SPAN}u);
  loop{let r=atomicCompareExchangeWeak(&C[${K.K_SPAN}u],0u,(grid<<30u)|(prim+1u));if(r.exchanged){atomicStore(&C[${K.K_SPAN_EXTENT}u],bitcast<u32>(m));break;}if(r.old_value!=0u){break;}}}}
 insertEntry(grid,prim,cellOf(box[0]),cellOf(box[1]));
}
fn padBox(lo:vec3f,hi:vec3f,r:f32)->array<vec3f,2>{let pad=r+1e-6*(abs(lo)+abs(hi))+1e-9;return array<vec3f,2>(lo-pad,hi+pad);}
@compute @workgroup_size(${WG}) fn gridClear(@builtin(global_invocation_id) g:vec3u,@builtin(num_workgroups) ng:vec3u){
 if(atomicLoad(&C[${K.K_DETECT}u])==0u||p.selfCollision==0u){return;}
 let grids=select(2u,3u,p.rodContact!=0u);for(var b=g.x;b<grids*p.gridBuckets;b+=ng.x*${WG}u){atomicStore(&KA[p.aHeads+b],0u);}
}
@compute @workgroup_size(${WG}) fn gridFaces(@builtin(global_invocation_id) g:vec3u){
 let f=g.x;if(f>=p.triangles||atomicLoad(&C[${K.K_DETECT}u])==0u||p.selfCollision==0u){return;}
 let id=triIds(f);var loB=min(approx(sweptPoint(id.x,0.)),approx(sweptPoint(id.x,1.)));var hiB=max(approx(sweptPoint(id.x,0.)),approx(sweptPoint(id.x,1.)));
 for(var j=1u;j<3u;j++){let v=id[j];loB=min(loB,min(approx(sweptPoint(v,0.)),approx(sweptPoint(v,1.))));hiB=max(hiB,max(approx(sweptPoint(v,0.)),approx(sweptPoint(v,1.))));}
 let box=padBox(loB,hiB,ksf(p.sFaceR+f));KF[p.kfFaceBox+2u*f]=vec4f(box[0],0);KF[p.kfFaceBox+2u*f+1u]=vec4f(box[1],0);
 insertBox(0u,f,box);
}
fn edgeIds(e:u32)->vec2u{return vec2u(I[p.sEdge+4u*e],I[p.sEdge+4u*e+1u]);}
@compute @workgroup_size(${WG}) fn gridEdges(@builtin(global_invocation_id) g:vec3u){
 let e=g.x;if(e>=p.edges||atomicLoad(&C[${K.K_DETECT}u])==0u||p.selfCollision==0u){return;}
 let id=edgeIds(e);let a0=approx(sweptPoint(id.x,0.));let a1=approx(sweptPoint(id.x,1.));let b0=approx(sweptPoint(id.y,0.));let b1=approx(sweptPoint(id.y,1.));
 let box=padBox(min(min(a0,a1),min(b0,b1)),max(max(a0,a1),max(b0,b1)),ksf(p.sEdge+4u*e+2u));KF[p.kfEdgeBox+2u*e]=vec4f(box[0],0);KF[p.kfEdgeBox+2u*e+1u]=vec4f(box[1],0);
 insertBox(1u,e,box);
}
fn overlaps(a0:vec3f,a1:vec3f,b0:vec3f,b1:vec3f)->bool{return all(a0<=b1)&&all(b0<=a1);}
fn inList(start:u32,list:u32,row:u32,value:u32)->bool{for(var k=I[start+row];k<I[start+row+1u];k++){if(I[list+k]==value){return true;}}return false;}
fn inRing(i:u32,v:u32)->bool{return inList(p.sRingStart,p.sRingList,i,v);}

// ---- narrowphase: the reference's closest points and conservative advancement
struct Closest{point:vec3f,w:vec3f}
fn closestTriangle(q:vec3f,a:vec3f,b:vec3f,c:vec3f)->Closest{
 let ab=b-a;let ac=c-a;let ap=q-a;let d1=dot(ab,ap);let d2=dot(ac,ap);
 if(d1<=0.&&d2<=0.){return Closest(a,vec3f(1,0,0));}
 let bp=q-b;let d3=dot(ab,bp);let d4=dot(ac,bp);
 if(d3>=0.&&d4<=d3){return Closest(b,vec3f(0,1,0));}
 let vc=d1*d4-d3*d2;
 if(vc<=0.&&d1>=0.&&d3<=0.){let v=d1/(d1-d3);return Closest(a+ab*v,vec3f(1.-v,v,0));}
 let cp=q-c;let d5=dot(ab,cp);let d6=dot(ac,cp);
 if(d6>=0.&&d5<=d6){return Closest(c,vec3f(0,0,1));}
 let vb=d5*d2-d1*d6;
 if(vb<=0.&&d2>=0.&&d6<=0.){let w=d2/(d2-d6);return Closest(a+ac*w,vec3f(1.-w,0,w));}
 let va=d3*d6-d5*d4;
 if(va<=0.&&d4-d3>=0.&&d5-d6>=0.){let w=(d4-d3)/((d4-d3)+(d5-d6));return Closest(b+(c-b)*w,vec3f(0,1.-w,w));}
 let den=va+vb+vc;if(abs(den)<1e-30){return Closest(a,vec3f(1,0,0));}
 let v=vb/den;let w=vc/den;return Closest(a+ab*v+ac*w,vec3f(1.-v-w,v,w));
}
struct Segments{s:f32,t:f32,a:vec3f,b:vec3f}
fn closestSegments(a:vec3f,b:vec3f,c:vec3f,d:vec3f)->Segments{
 let u=b-a;let v=d-c;let w=a-c;let aa=dot(u,u);let bb=dot(u,v);let cc=dot(v,v);let dd=dot(u,w);let ee=dot(v,w);
 var s=0.;if(aa>1e-30){s=clamp((bb*ee-cc*dd)/max(1e-30,aa*cc-bb*bb),0.,1.);}
 var t=0.;if(cc>1e-30){t=(bb*s+ee)/cc;}
 if(t<0.){t=0.;s=0.;if(aa>1e-30){s=clamp(-dd/aa,0.,1.);}}else if(t>1.){t=1.;s=0.;if(aa>1e-30){s=clamp((bb-dd)/aa,0.,1.);}}
 return Segments(s,t,a+s*u,c+t*v);
}
fn vertexFace(i:u32,f:u32){
 let id=triIds(f);
 if(inRing(i,id.x)||inRing(i,id.y)||inRing(i,id.z)||inList(p.sFrStart,p.sFrList,i,f)){return;}
 let k0=(1u<<28u)|i;if(held(k0,f)){return;}
 if(isFixed(i)&&isFixed(id.x)&&isFixed(id.y)&&isFixed(id.z)){return;}
 let q0=oldEdge(i,id.x);let q1=oldEdge(i,id.y);let q2=oldEdge(i,id.z);let ui=xv(i);let ua=xv(id.x);let ub=xv(id.y);let uc=xv(id.z);
 let distance=kRadius(i)+ksf(p.sFaceR+f);let bound=length(ui)+max(length(ua),max(length(ub),length(uc)));
 var time=0.;var found=false;var normal=vec3f(0,1,0);var weights=vec3f(1,0,0);
 for(var it=0u;it<100u;it++){
  let v=time*ui;let a=q0+time*ua;let b=q1+time*ub;let c=q2+time*uc;let closest=closestTriangle(v,a,b,c);let delta=v-closest.point;let gap=length(delta);
  if(gap<=distance+1e-9){found=true;weights=closest.w;let nn=cross(b-a,c-a);if(gap>1e-12){normal=delta/gap;}else{normal=nn/max(1e-20,length(nn));}break;}
  if(time>=1.||bound==0.){break;}
  time=advance(time,max(1e-10,.9*(gap-distance)/bound));
 }
 if(!found){return;}
 let g0=-dot(normal,weights.x*q0+weights.y*q1+weights.z*q2)-distance;
 let mu=min(min(kFriction(i),kFriction(id.x)),min(kFriction(id.y),kFriction(id.z)));
 addContact(k0,f,vec4u(i,id.x,id.y,id.z),vec4f(1.,-weights.x,-weights.y,-weights.z),normal,g0,vec3f(0),mu,kRadius(i),NONE);
}
@compute @workgroup_size(${WG}) fn detectFaces(@builtin(global_invocation_id) g:vec3u){
 let i=g.x;if(i>=p.n||atomicLoad(&C[${K.K_DETECT}u])==0u||p.selfCollision==0u||!kShell(i)){return;}
 let box=padBox(min(approx(sweptPoint(i,0.)),approx(sweptPoint(i,1.))),max(approx(sweptPoint(i,0.)),approx(sweptPoint(i,1.))),kRadius(i));
 let ca=cellOf(box[0]);let cb=cellOf(box[1]);let count=cellCount(ca,cb);if(count>${CELL_CAP}u){atomicOr(&C[${K.K_ERROR}u],${E_CELLS}u);return;}
 for(var k=0u;k<count;k++){
  let cell=cellAt(ca,cb,k);var e=atomicLoad(&KA[p.aHeads+bucketOf(cell)]);
  while(e!=0u){
   let f=KU[p.uEntry+2u*(e-1u)];e=KU[p.uEntry+2u*(e-1u)+1u];
   let f0=KF[p.kfFaceBox+2u*f].xyz;let f1=KF[p.kfFaceBox+2u*f+1u].xyz;
   if(!overlaps(box[0],box[1],f0,f1)||any(max(ca,cellOf(f0))!=cell)){continue;}
   vertexFace(i,f);
  }
 }
}
fn edgeEdge(ai:u32,bi:u32){
 let ea=edgeIds(ai);let eb=edgeIds(bi);
 if(ea.x==eb.x||ea.x==eb.y||ea.y==eb.x||ea.y==eb.y){return;}
 for(var j=0u;j<2u;j++){let v=ea[j];for(var m=0u;m<2u;m++){let w=eb[m];if(inRing(v,w)||inList(p.sLvStart,p.sLvList,v,w)){return;}}}
 if(inList(p.sE3Start,p.sE3List,ai,bi)){return;}
 let k0=(2u<<28u)|ai;if(held(k0,bi)){return;}
 if(isFixed(ea.x)&&isFixed(ea.y)&&isFixed(eb.x)&&isFixed(eb.y)){return;}
 let ids=vec4u(ea,eb);let s0=vec3f(0);let s1=oldEdge(ea.x,ea.y);let s2=oldEdge(ea.x,eb.x);let s3=oldEdge(ea.x,eb.y);
 let u0=xv(ids.x);let u1=xv(ids.y);let u2=xv(ids.z);let u3=xv(ids.w);
 let ra=ksf(p.sEdge+4u*ai+2u);let thickness=ra+ksf(p.sEdge+4u*bi+2u);let travel=max(length(u0),length(u1))+max(length(u2),length(u3));
 var time=0.;var found=false;var normal=vec3f(0,1,0);var s=0.;var t=0.;
 for(var it=0u;it<100u;it++){
  let pp=s0+time*u0;let qq=s1+time*u1;let uu=s2+time*u2;let vv=s3+time*u3;let closest=closestSegments(pp,qq,uu,vv);let delta=closest.a-closest.b;let gap=length(delta);
  if(gap<=thickness+1e-9){
   found=true;s=closest.s;t=closest.t;let fallback=cross(qq-pp,vv-uu);
   if(gap>1e-12){normal=delta/gap;}else if(length(fallback)>1e-12){normal=fallback/length(fallback);}else{normal=vec3f(0,1,0);}
   break;
  }
  if(time>=1.||travel==0.){break;}
  time=advance(time,max(1e-10,.9*(gap-thickness)/travel));
 }
 if(!found){return;}
 let w=vec4f(1.-s,s,-(1.-t),-t);let g0=dot(normal,w.y*s1+w.z*s2+w.w*s3)-thickness;
 let mu=min(min(kFriction(ids.x),kFriction(ids.y)),min(kFriction(ids.z),kFriction(ids.w)));
 addContact(k0,bi,ids,w,normal,g0,vec3f(0),mu,ra,NONE);
}
@compute @workgroup_size(${WG}) fn detectEdges(@builtin(global_invocation_id) g:vec3u){
 let ai=g.x;if(ai>=p.edges||atomicLoad(&C[${K.K_DETECT}u])==0u||p.selfCollision==0u){return;}
 let b0=KF[p.kfEdgeBox+2u*ai].xyz;let b1=KF[p.kfEdgeBox+2u*ai+1u].xyz;
 let ca=cellOf(b0);let cb=cellOf(b1);let count=cellCount(ca,cb);if(count>${CELL_CAP}u){return;}
 for(var k=0u;k<count;k++){
  let cell=cellAt(ca,cb,k);var e=atomicLoad(&KA[p.aHeads+p.gridBuckets+bucketOf(cell)]);
  while(e!=0u){
   let bi=KU[p.uEntry+2u*(e-1u)];e=KU[p.uEntry+2u*(e-1u)+1u];
   if(bi<=ai){continue;}
   let e0=KF[p.kfEdgeBox+2u*bi].xyz;let e1=KF[p.kfEdgeBox+2u*bi+1u].xyz;
   if(!overlaps(b0,b1,e0,e1)||any(max(ca,cellOf(e0))!=cell)){continue;}
   edgeEdge(ai,bi);
  }
 }
}
@compute @workgroup_size(${WG}) fn detectColliders(@builtin(global_invocation_id) g:vec3u,@builtin(num_workgroups) ng:vec3u){
 if(atomicLoad(&C[${K.K_DETECT}u])==0u||p.colliders==0u){return;}
 for(var t=g.x;t<p.n*p.colliders;t+=ng.x*${WG}u){
  let i=t%p.n;let ci=t/p.n;
  let rod=p.rodContact!=0u&&rodVertex(i);
  if(isFixed(i)||(!kShell(i)&&!rod)||held(i,ci)){continue;}
  let b=cHeader(ci);let r=kRadius(i);let u=xv(i);let velocity=cv3(b+2u);
  let speed=(length(u)+p.dt*ksf(b+5u))*ksf(b+6u);let tolerance=max(1e-10,r*1e-6);
  var time=0.;var hit=false;var clear=false;var hitNormal=vec3f(0,1,0);var hitDistance=0.;
  for(var it=0u;it<80u;it++){
   let at=sweptPoint(i,time);let distance=colliderDistance(ci,at,(1.-time)*p.dt)-r;
   if(distance<=tolerance){hit=true;hitNormal=colliderSample(ci,at,(1.-time)*p.dt).n;hitDistance=distance;break;}
   if(it==0u&&distance-speed>tolerance){clear=true;break;}
   if(time>=1.||speed==0.){break;}
   time=advance(time,max(1e-9,.9*distance/speed));
  }
  if(clear){continue;}
  let end=colliderSample(ci,sweptPoint(i,1.),0.);
  if(!hit&&end.d>=r){continue;}
  var normal=end.n;var g0=(end.d-r)-dot(end.n,u);
  if(hit){normal=hitNormal;g0=hitDistance-dot(normal,time*u+velocity*((1.-time)*p.dt));}
  var mu=ksf(b+1u);if(mu<0.){mu=kFriction(i);}
  if(rod){var ids:array<u32,5>;var w:array<f32,5>;ids[0]=i;w[0]=1.;addRodContact(i,ci,ids,w,1u,vec2u(kOwner(i),NONE),normal,g0,velocity*p.dt,mu,r,ci,0.);continue;}
  addContact(i,ci,vec4u(i,0u,0u,0u),vec4f(1,0,0,0),normal,g0,velocity*p.dt,mu,r,ci);
 }
}

// ---- rod pairs of ClothContactSet (round, uncut rods): KU[kuRod+4c..] = (fifth vertex, side
// segments 0 and 1 (NONE: no side), rodm), rodm = slots | side0 a<<3 | side0 b<<6 | side1 a<<9 |
// side1 b<<12 | 1<<31 (the slots of each side's endpoints); KF[kfRod+3c..] = (t0, r0/L0), (t1, r1/L1),
// (w4, r0, r1, thickness). Keys k0 = kind<<28 | first feature: 0 c: (rod vertex sphere, collider),
// 3 rc: (segment, collider), 4 rv: (rod vertex, face), 5 rq: (rod vertex, shell edge),
// 6 rp: (vertex, segment), 7 rf: (segment, face), 8 rx: (segment, shell edge), 9 rs: (segment, later segment).
// Each side's unit tangent and r/L at x (linearizeRolling): the slip Jacobian's endpoint couple,
// and its arm over r with the twist row (KF[kfRodArm+2c+side]): a round section's arm is r n, its
// twist row zero (the contact's normal KF[4c] is current when this runs).
fn rodLinearize(c:u32){
 let n=KF[4u*c].xyz;
 for(var side=0u;side<2u;side++){
  let s=KU[p.kuRod+4u*c+1u+side];KF[p.kfRodArm+2u*c+side]=vec4f(n,0);if(s==NONE){KF[p.kfRod+3u*c+side]=vec4f(0);continue;}
  let id=segIds(s);let e=segOldEdge(s)+(xv(id.y)-xv(id.x));let l=length(e);
  if(!(l>0.&&l<3e38)){atomicOr(&C[${K.K_ERROR}u],${E_NONFINITE}u);KF[p.kfRod+3u*c+side]=vec4f(0);continue;}
  KF[p.kfRod+3u*c+side]=vec4f(e/l,segR(s)/l);
 }
}
// addRod: the sides' endpoints join the vertices (weight zero) where absent; the pair needs a free vertex of nonzero weight.
fn addRodContact(k0:u32,k1:u32,idsIn:array<u32,5>,wIn:array<f32,5>,count:u32,sides:vec2u,n:vec3f,g0:f32,obstacle:vec3f,mu:f32,radius:f32,collider:u32,thickness:f32){
 var ids=idsIn;var w=wIn;var slots=count;var rodm=0x80000000u;
 for(var side=0u;side<2u;side++){
  let s=sides[side];if(s==NONE){continue;}
  let id=segIds(s);
  for(var end=0u;end<2u;end++){
   let v=id[end];var at=NONE;for(var j=0u;j<slots;j++){if(ids[j]==v){at=j;}}
   if(at==NONE){at=slots;ids[slots]=v;w[slots]=0.;slots++;}
   rodm|=at<<(3u+6u*side+3u*end);
  }
 }
 rodm|=slots;if(sides.y!=NONE){rodm|=1u<<15u;}
 var free=false;for(var j=0u;j<slots;j++){if(!isFixed(ids[j])&&w[j]!=0.){free=true;}}
 if(!free){return;}
 let c=claimContact(k0,k1,collider);if(c==NONE){return;}
 writeContact(c,warmLambda(k0,k1),vec4u(ids[0],ids[1],ids[2],ids[3]),vec4f(w[0],w[1],w[2],w[3]),n,g0,obstacle,mu,radius);
 KU[p.kuRod+4u*c]=ids[4];KU[p.kuRod+4u*c+1u]=sides.x;KU[p.kuRod+4u*c+2u]=sides.y;KU[p.kuRod+4u*c+3u]=rodm;
 var r=vec2f(0);if(sides.x!=NONE){r.x=segR(sides.x);}if(sides.y!=NONE){r.y=segR(sides.y);}
 KF[p.kfRod+3u*c+2u]=vec4f(w[4],r,thickness);
 rodLinearize(c);
 atomicAdd(&C[${K.K_ADDED}u],1u);
}
// The capsule of segment s against sphere collider ci at sweep time t (sampleClippedRodCollider,
// uncut round rod): distance, normal from the sphere, and the closest centreline parameter.
struct CapsuleSample{d:f32,n:vec3f,t:f32}
fn capsuleSphere(s:u32,ci:u32,time:f32,beforeEnd:f32)->CapsuleSample{
 let b=cHeader(ci);let id=segIds(s);let a=shifted(sweptPoint(id.x,time),cv3(b+2u)*beforeEnd);
 let rel=(a.h-cv3(b+8u))+(a.l-cv3(b+11u));let e=segOldEdge(s)+time*(xv(id.y)-xv(id.x));
 let ee=dot(e,e);var t=0.;if(ee>0.){t=clamp(-dot(rel,e)/ee,0.,1.);}
 let delta=rel+t*e;let l=length(delta);
 var n=perpendicular(e);if(l>1e-14){n=delta/l;}
 return CapsuleSample(((l-segR(s))-ksf(b+14u))-ksf(b+15u),n,t);
}
// The crossing test for segment s's centreline against sphere ci (segmentEntersCollider).
fn segmentEntersSphere(s:u32,ci:u32,tolerance:f32)->bool{
 let b=cHeader(ci);let id=segIds(s);let speed=(max(length(xv(id.x)),length(xv(id.y)))+p.dt*ksf(b+5u))*ksf(b+6u);var time=0.;
 for(var it=0u;it<256u;it++){
  let distance=capsuleSphere(s,ci,time,(1.-time)*p.dt).d+segR(s);
  if(distance<=tolerance){return true;}
  if(time>=1.||speed==0.){return false;}
  time=advance(time,max(1e-9,.9*distance/speed));
 }
 return true;
}
fn relinearizeRodSegment(c:u32,ci:u32){
 let s=KU[p.kuMeta+4u*c]&0x0FFFFFFFu;let q=capsuleSphere(s,ci,1.,0.);
 if(segmentEntersSphere(s,ci,max(1e-10,segR(s)*1e-6))){atomicOr(&C[${K.K_ERROR}u],${E_CROSSED}u);}
 let id=segIds(s);let w=vec4f(1.-q.t,q.t,0,0);let obstacle=cv3(cHeader(ci)+2u)*p.dt;
 KF[4u*c]=vec4f(q.n,q.d-dot(q.n,w.x*xv(id.x)+w.y*xv(id.y)));KF[4u*c+1u]=vec4f(obstacle-dot(obstacle,q.n)*q.n,KF[4u*c+1u].w);KF[4u*c+3u]=w;
 rodLinearize(c);maxf(${K.K_PEN}u,-q.d);maxf(${K.K_REPPEN}u,-q.d);
}
// Segment interiors against colliders (detectRodColliders): conservative advancement on the capsule
// distance, kept where the closest centreline point is inside the segment.
@compute @workgroup_size(${WG}) fn detectRodColliders(@builtin(global_invocation_id) g:vec3u,@builtin(num_workgroups) ng:vec3u){
 if(atomicLoad(&C[${K.K_DETECT}u])==0u||p.colliders==0u||p.rodContact==0u){return;}
 for(var t=g.x;t<p.segments*p.colliders;t+=ng.x*${WG}u){
  let s=t%p.segments;let ci=t/p.segments;let id=segIds(s);
  if(isFixed(id.x)&&isFixed(id.y)){continue;}
  let k0=(3u<<28u)|s;if(held(k0,ci)){continue;}
  let b=cHeader(ci);let kind=I[b];
  // A plane's closest capsule point is its support, at an endpoint: never interior (the rod-vertex sphere's pair).
  if(kind==${KIND_PLANE}u){continue;}
  if(kind!=${KIND_SPHERE}u){atomicOr(&C[${K.K_ERROR}u],${E_UNSUPPORTED}u);continue;}
  let r=segR(s);let ua=xv(id.x);let ub=xv(id.y);let travel=(max(length(ua),length(ub))+p.dt*ksf(b+5u))*ksf(b+6u);let tolerance=max(1e-10,r*1e-6);
  var time=0.;var hit=false;var q=CapsuleSample(0.,vec3f(0,1,0),0.);
  for(var it=0u;it<160u;it++){
   q=capsuleSphere(s,ci,time,(1.-time)*p.dt);
   if(q.d<=tolerance){hit=true;break;}
   if(it==0u&&q.d-travel>tolerance){break;}
   if(time>=1.||travel==0.){break;}
   time=advance(time,max(1e-9,.9*q.d/travel));
  }
  if(!hit||!interior(q.t)){continue;}
  let velocity=cv3(b+2u);let w0=1.-q.t;let w1=q.t;
  let g0=q.d-dot(q.n,time*(w0*ua+w1*ub)+velocity*((1.-time)*p.dt));
  var mu=ksf(b+1u);if(mu<0.){mu=segFriction(s);}
  var ids:array<u32,5>;var w:array<f32,5>;ids[0]=id.x;ids[1]=id.y;w[0]=w0;w[1]=w1;
  addRodContact(k0,ci,ids,w,2u,vec2u(s,NONE),q.n,g0,velocity*p.dt,mu,r,ci,0.);
 }
}

// rodSelfClosest: shapes 0 point-face, 1 segment-face, 2 point-segment, 3 segment-segment (points
// of the first feature, then the second); weights, separation second -> first, normal fallback,
// and whether this shape owns its closest points (not a vertex or edge pair).
fn interior(x:f32)->bool{return x>1e-6&&x<1.-1e-6;}
fn closestPointSegment(q:vec3f,a:vec3f,b:vec3f)->f32{let e=b-a;let ee=dot(e,e);if(ee>1e-30){return clamp(dot(q-a,e)/ee,0.,1.);}return 0.;}
fn perpendicular(e:vec3f)->vec3f{let f=cross(e,select(vec3f(0,1,0),vec3f(1,0,0),abs(e.x)<.8*length(e)));let l=length(f);if(l>0.){return f/l;}return vec3f(0,1,0);}
struct SegTri{s:f32,w:vec3f,pa:vec3f,pb:vec3f}
fn closestSegTri(a:vec3f,b:vec3f,x:vec3f,y:vec3f,z:vec3f)->SegTri{
 let nrm=cross(y-x,z-x);let da=dot(a-x,nrm);let db=dot(b-x,nrm);
 // A crossing segment: distance zero at its plane crossing (inside the triangle to Float32 roundoff).
 if(da*db<=0.&&abs(da-db)>1e-30){let s=da/(da-db);let pp=mix(a,b,s);let q=closestTriangle(pp,x,y,z);if(length(pp-q.point)<=8.*1.1920929e-7*(length(pp)+length(q.point))+1e-12){return SegTri(s,q.w,pp,q.point);}}
 var best=SegTri(0.,vec3f(1,0,0),a,x);var distance=3e38;
 for(var e=0u;e<2u;e++){let pp=select(a,b,e==1u);let q=closestTriangle(pp,x,y,z);let d=length(pp-q.point);if(d<distance){distance=d;best=SegTri(f32(e),q.w,pp,q.point);}}
 var corners=array<vec3f,3>(x,y,z);
 for(var i=0u;i<3u;i++){let q=closestSegments(a,b,corners[i],corners[(i+1u)%3u]);var w=vec3f(0);w[i]=1.-q.t;w[(i+1u)%3u]=q.t;let d=length(q.a-q.b);if(d<distance){distance=d;best=SegTri(q.s,w,q.a,q.b);}}
 return best;
}
struct RodClosest{w:array<f32,5>,delta:vec3f,fallback:vec3f,owned:bool}
fn rodClosest(shape:u32,q:array<vec3f,5>)->RodClosest{
 var r:RodClosest;
 if(shape==0u){let c=closestTriangle(q[0],q[1],q[2],q[3]);r.w[0]=1.;r.w[1]=-c.w.x;r.w[2]=-c.w.y;r.w[3]=-c.w.z;r.delta=q[0]-c.point;r.fallback=cross(q[2]-q[1],q[3]-q[1]);r.owned=true;}
 else if(shape==1u){let c=closestSegTri(q[0],q[1],q[2],q[3],q[4]);r.w[0]=1.-c.s;r.w[1]=c.s;r.w[2]=-c.w.x;r.w[3]=-c.w.y;r.w[4]=-c.w.z;r.delta=c.pa-c.pb;r.fallback=cross(q[3]-q[2],q[4]-q[2]);r.owned=interior(c.s)&&all(c.w>vec3f(1e-6));}
 else if(shape==2u){let t=closestPointSegment(q[0],q[1],q[2]);r.w[0]=1.;r.w[1]=-(1.-t);r.w[2]=-t;r.delta=q[0]-mix(q[1],q[2],t);r.fallback=perpendicular(q[2]-q[1]);r.owned=true;}
 else{let c=closestSegments(q[0],q[1],q[2],q[3]);r.w[0]=1.-c.s;r.w[1]=c.s;r.w[2]=-(1.-c.t);r.w[3]=-c.t;r.delta=c.a-c.b;r.fallback=cross(q[1]-q[0],q[3]-q[2]);r.owned=interior(c.s)&&interior(c.t);}
 return r;
}
fn unitNormal(delta:vec3f,fallback:vec3f)->vec3f{let l=length(delta);let f=length(fallback);if(l>1e-12){return delta/l;}if(f>1e-12){return fallback/f;}return vec3f(0,1,0);}
fn rodShape(kind:u32)->u32{if(kind==4u){return 0u;}if(kind==7u){return 1u;}if(kind>=8u){return 3u;}return 2u;}
fn rodFeatureVertices(kind:u32)->u32{if(kind==4u||kind>=8u){return 4u;}if(kind==7u){return 5u;}return 3u;}
// sweepRodPair: conservative advancement from the old positions to x (relative to the first vertex's old position).
struct RodHit{found:bool,w:array<f32,5>,n:vec3f}
fn rodSweep(shape:u32,ids:array<u32,5>,count:u32,first:u32,thickness:f32)->RodHit{
 var start:array<vec3f,5>;var u:array<vec3f,5>;var a=0.;var b=0.;
 for(var j=0u;j<count;j++){if(j>0u){start[j]=oldEdge(ids[0],ids[j]);}u[j]=xv(ids[j]);let m=length(u[j]);if(j<first){a=max(a,m);}else{b=max(b,m);}}
 let travel=a+b;var time=0.;var hit:RodHit;
 for(var it=0u;it<100u;it++){
  var q:array<vec3f,5>;for(var j=0u;j<count;j++){q[j]=start[j]+time*u[j];}
  let c=rodClosest(shape,q);let gap=length(c.delta);
  if(gap<=thickness+1e-9){if(c.owned){hit.found=true;hit.w=c.w;hit.n=unitNormal(c.delta,c.fallback);}return hit;}
  if(time>=1.||travel==0.){return hit;}
  time=advance(time,max(1e-10,.9*(gap-thickness)/travel));
 }
 return hit;
}
fn rodPair(k0:u32,k1:u32,ids:array<u32,5>,thickness:f32,radius:f32,sides:vec2u){
 if(held(k0,k1)){return;}
 let kind=k0>>28u;let count=rodFeatureVertices(kind);let first=select(1u,2u,kind>=7u);
 var fixed=true;for(var j=0u;j<count;j++){if(!isFixed(ids[j])){fixed=false;}}
 if(fixed){return;}
 let hit=rodSweep(rodShape(kind),ids,count,first,thickness);if(!hit.found){return;}
 var g0=-thickness;var mu=kFriction(ids[0]);
 for(var j=1u;j<count;j++){g0+=hit.w[j]*dot(hit.n,oldEdge(ids[0],ids[j]));mu=min(mu,kFriction(ids[j]));}
 addRodContact(k0,k1,ids,hit.w,count,sides,hit.n,g0,vec3f(0),mu,radius,NONE,thickness);
}
fn relinearizeRodSelf(c:u32){
 let kind=KU[p.kuMeta+4u*c]>>28u;let k=contactAt(c);let count=rodFeatureVertices(kind);
 var q:array<vec3f,5>;var e:array<vec3f,5>;
 for(var j=0u;j<count;j++){let v=slotV(k,j);if(j>0u){e[j]=oldEdge(k.v.x,v);}q[j]=e[j]+xv(v);}
 let r=rodClosest(rodShape(kind),q);let thickness=KF[p.kfRod+3u*c+2u].w;let old=k.n;
 var n=unitNormal(r.delta,r.fallback);if(length(r.delta)<=1e-12&&length(r.fallback)<=1e-12){n=old;}
 if(dot(n,old)<0.){n=-n;}
 var g0=-thickness;for(var j=1u;j<count;j++){g0+=r.w[j]*dot(n,e[j]);}
 KF[4u*c]=vec4f(n,g0);KF[4u*c+3u]=vec4f(r.w[0],r.w[1],r.w[2],r.w[3]);if(count==5u){KF[p.kfRod+3u*c+2u].x=r.w[4];}
 rodLinearize(c);
 let pen=thickness-dot(n,r.delta);maxf(${K.K_PEN}u,pen);maxf(${K.K_REPPEN}u,pen);
}
// touches(near, [v]) with near = i and its rod neighbours (a vertex's ring holds the vertex itself).
fn touchesNear(i:u32,v:u32)->bool{
 if(inRing(i,v)){return true;}
 for(var k=I[p.sNbStart+i];k<I[p.sNbStart+i+1u];k++){if(inRing(I[p.sNbList+k],v)){return true;}}
 return false;
}
fn edgeR(e:u32)->f32{return ksf(p.sEdge+4u*e+2u);}
@compute @workgroup_size(${WG}) fn gridSegments(@builtin(global_invocation_id) g:vec3u){
 let s=g.x;if(s>=p.segments||atomicLoad(&C[${K.K_DETECT}u])==0u||p.selfCollision==0u||p.rodContact==0u){return;}
 let id=segIds(s);let a0=approx(sweptPoint(id.x,0.));let a1=approx(sweptPoint(id.x,1.));let b0=approx(sweptPoint(id.y,0.));let b1=approx(sweptPoint(id.y,1.));
 let box=padBox(min(min(a0,a1),min(b0,b1)),max(max(a0,a1),max(b0,b1)),segR(s));KF[p.kfSegBox+2u*s]=vec4f(box[0],0);KF[p.kfSegBox+2u*s+1u]=vec4f(box[1],0);
 insertBox(2u,s,box);
}
// Rod vertex spheres against faces (rv) and shell edges (rq); rod and shell vertices against segments (rp).
@compute @workgroup_size(${WG}) fn detectRodPoints(@builtin(global_invocation_id) g:vec3u){
 let i=g.x;if(i>=p.n||atomicLoad(&C[${K.K_DETECT}u])==0u||p.selfCollision==0u||p.rodContact==0u){return;}
 let rod=rodVertex(i);if(!rod&&!kShell(i)){return;}
 let ri=kRadius(i);let box=padBox(min(approx(sweptPoint(i,0.)),approx(sweptPoint(i,1.))),max(approx(sweptPoint(i,0.)),approx(sweptPoint(i,1.))),ri);
 let ca=cellOf(box[0]);let cb=cellOf(box[1]);let count=cellCount(ca,cb);if(count>${CELL_CAP}u){atomicOr(&C[${K.K_ERROR}u],${E_CELLS}u);return;}
 for(var k=0u;k<count;k++){
  let cell=cellAt(ca,cb,k);let bucket=bucketOf(cell);
  if(rod){
   var e=atomicLoad(&KA[p.aHeads+bucket]);
   while(e!=0u){
    let f=KU[p.uEntry+2u*(e-1u)];e=KU[p.uEntry+2u*(e-1u)+1u];
    let f0=KF[p.kfFaceBox+2u*f].xyz;let f1=KF[p.kfFaceBox+2u*f+1u].xyz;
    if(!overlaps(box[0],box[1],f0,f1)||any(max(ca,cellOf(f0))!=cell)){continue;}
    let id=triIds(f);if(touchesNear(i,id.x)||touchesNear(i,id.y)||touchesNear(i,id.z)){continue;}
    var ids:array<u32,5>;ids[0]=i;ids[1]=id.x;ids[2]=id.y;ids[3]=id.z;
    rodPair((4u<<28u)|i,f,ids,ri+ksf(p.sFaceR+f),ri,vec2u(kOwner(i),NONE));
   }
   var q=atomicLoad(&KA[p.aHeads+p.gridBuckets+bucket]);
   while(q!=0u){
    let ed=KU[p.uEntry+2u*(q-1u)];q=KU[p.uEntry+2u*(q-1u)+1u];
    let e0=KF[p.kfEdgeBox+2u*ed].xyz;let e1=KF[p.kfEdgeBox+2u*ed+1u].xyz;
    if(!overlaps(box[0],box[1],e0,e1)||any(max(ca,cellOf(e0))!=cell)){continue;}
    let id=edgeIds(ed);if(touchesNear(i,id.x)||touchesNear(i,id.y)){continue;}
    var ids:array<u32,5>;ids[0]=i;ids[1]=id.x;ids[2]=id.y;
    rodPair((5u<<28u)|i,ed,ids,ri+edgeR(ed),ri,vec2u(kOwner(i),NONE));
   }
  }
  var h=atomicLoad(&KA[p.aHeads+2u*p.gridBuckets+bucket]);
  while(h!=0u){
   let s=KU[p.uEntry+2u*(h-1u)];h=KU[p.uEntry+2u*(h-1u)+1u];
   let s0=KF[p.kfSegBox+2u*s].xyz;let s1=KF[p.kfSegBox+2u*s+1u].xyz;
   if(!overlaps(box[0],box[1],s0,s1)||any(max(ca,cellOf(s0))!=cell)){continue;}
   let id=segIds(s);if(id.x==i||id.y==i){continue;}
   if(!rod&&(inRing(i,id.x)||inRing(i,id.y))){continue;}
   if(inList(p.sRpXStart,p.sRpXList,i,s)){continue;}
   var ids:array<u32,5>;ids[0]=i;ids[1]=id.x;ids[2]=id.y;
   rodPair((6u<<28u)|i,s,ids,ri+segR(s),ri,select(vec2u(s,NONE),vec2u(kOwner(i),s),rod));
  }
 }
}
// Segment interiors against faces (rf), shell edges (rx) and later segments (rs).
@compute @workgroup_size(${WG}) fn detectRodSegments(@builtin(global_invocation_id) g:vec3u){
 let s=g.x;if(s>=p.segments||atomicLoad(&C[${K.K_DETECT}u])==0u||p.selfCollision==0u||p.rodContact==0u){return;}
 let id=segIds(s);let r=segR(s);let b0=KF[p.kfSegBox+2u*s].xyz;let b1=KF[p.kfSegBox+2u*s+1u].xyz;
 let ca=cellOf(b0);let cb=cellOf(b1);let count=cellCount(ca,cb);if(count>${CELL_CAP}u){return;}
 for(var k=0u;k<count;k++){
  let cell=cellAt(ca,cb,k);let bucket=bucketOf(cell);
  var e=atomicLoad(&KA[p.aHeads+bucket]);
  while(e!=0u){
   let f=KU[p.uEntry+2u*(e-1u)];e=KU[p.uEntry+2u*(e-1u)+1u];
   let f0=KF[p.kfFaceBox+2u*f].xyz;let f1=KF[p.kfFaceBox+2u*f+1u].xyz;
   if(!overlaps(b0,b1,f0,f1)||any(max(ca,cellOf(f0))!=cell)){continue;}
   let t=triIds(f);var touch=false;for(var j=0u;j<3u;j++){if(inRing(id.x,t[j])||inRing(id.y,t[j])){touch=true;}}
   if(touch){continue;}
   var ids:array<u32,5>;ids[0]=id.x;ids[1]=id.y;ids[2]=t.x;ids[3]=t.y;ids[4]=t.z;
   rodPair((7u<<28u)|s,f,ids,r+ksf(p.sFaceR+f),r,vec2u(s,NONE));
  }
  var q=atomicLoad(&KA[p.aHeads+p.gridBuckets+bucket]);
  while(q!=0u){
   let ed=KU[p.uEntry+2u*(q-1u)];q=KU[p.uEntry+2u*(q-1u)+1u];
   let e0=KF[p.kfEdgeBox+2u*ed].xyz;let e1=KF[p.kfEdgeBox+2u*ed+1u].xyz;
   if(!overlaps(b0,b1,e0,e1)||any(max(ca,cellOf(e0))!=cell)){continue;}
   let t=edgeIds(ed);if(inRing(id.x,t.x)||inRing(id.y,t.x)||inRing(id.x,t.y)||inRing(id.y,t.y)){continue;}
   var ids:array<u32,5>;ids[0]=id.x;ids[1]=id.y;ids[2]=t.x;ids[3]=t.y;
   rodPair((8u<<28u)|s,ed,ids,r+edgeR(ed),r,vec2u(s,NONE));
  }
  var h=atomicLoad(&KA[p.aHeads+2u*p.gridBuckets+bucket]);
  while(h!=0u){
   let u=KU[p.uEntry+2u*(h-1u)];h=KU[p.uEntry+2u*(h-1u)+1u];
   if(u<=s){continue;}
   let u0=KF[p.kfSegBox+2u*u].xyz;let u1=KF[p.kfSegBox+2u*u+1u].xyz;
   if(!overlaps(b0,b1,u0,u1)||any(max(ca,cellOf(u0))!=cell)){continue;}
   let t=segIds(u);if(inRing(id.x,t.x)||inRing(id.y,t.x)||inRing(id.x,t.y)||inRing(id.y,t.y)){continue;}
   if(inList(p.sRsXStart,p.sRsXList,s,u)){continue;}
   var ids:array<u32,5>;ids[0]=id.x;ids[1]=id.y;ids[2]=t.x;ids[3]=t.y;
   rodPair((9u<<28u)|s,u,ids,r+segR(u),r,vec2u(s,u));
  }
 }
}
`;
}
