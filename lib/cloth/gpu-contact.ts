import {ClothContactBVH} from './contact-bvh';
import {clothLocalShellEdges, clothShellCache, clothVertexContactRadius, clothVertexLocalFaces, inSolveContactUnsupported} from './contact-constraints';
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

/** Why a collider set needs the CPU detection path, or undefined. */
export function gpuContactUnsupported(topology: PreparedClothTopology, colliders: readonly ClothCollider[]): string | undefined {
  const base = inSolveContactUnsupported(topology, colliders);
  if (base) return base;
  if (topology.rods.length) return 'rod contact inside the solve is on the Float64 CPU path only';
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
  readonly offsets: {sVInfo: number; sRingStart: number; sRingList: number; sFrStart: number; sFrList: number; sLvStart: number; sLvList: number; sEdge: number; sE3Start: number; sE3List: number; sFaceR: number};
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
  const parts: Uint32Array[] = [];
  const vInfo = new Uint32Array(4 * n);
  for (let i = 0; i < n; i++) {
    const m = topology.materials[topology.vertexMaterial[i]!]!;
    vInfo.set([bits(clothVertexContactRadius(topology, i)), bits(m.friction), topology.incidentTriangles[i]!.length ? 1 : 0, 0], 4 * i);
  }
  const edgeWords = new Uint32Array(4 * E);
  edges.forEach((e, k) => edgeWords.set([e.vertices[0], e.vertices[1], bits(e.r), 0], 4 * k));
  const faceR = Uint32Array.from(topology.triangles, f => bits(.5 * topology.materials[f.material]!.thickness_m));
  const names = ['sVInfo', 'sRingStart', 'sRingList', 'sFrStart', 'sFrList', 'sLvStart', 'sLvList', 'sEdge', 'sE3Start', 'sE3List', 'sFaceR'] as const;
  parts.push(vInfo, rings.start, rings.list, frs.start, frs.list, lvs.start, lvs.list, edgeWords, e3s.start, e3s.list, faceR);
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
} as const;
export const CONTACT_CONTROL_WORDS = 120;
export const E_CAPACITY = 1, E_GRID = 2, E_CROSSED = 4, E_STIFFNESS = 8, E_TABLE = 16, E_CELLS = 32, E_NONFINITE = 64;
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
fn addContact(k0:u32,k1:u32,v:vec4u,w:vec4f,n:vec3f,g0:f32,obstacle:vec3f,mu:f32,radius:f32,collider:u32){
 // A full pool fails before claiming a key: claims past capacity would fill the
 // table (2 kMax slots), and every later claim would then probe all of it.
 if(atomicLoad(&C[${K.K_COUNT}u])>=p.kMax){atomicOr(&C[${K.K_ERROR}u],${E_CAPACITY}u);return;}
 let slot=tableClaim(p.tableCur,k0,k1);if(slot==NONE){return;}
 let c=atomicAdd(&C[${K.K_COUNT}u],1u);if(c>=p.kMax){atomicOr(&C[${K.K_ERROR}u],${E_CAPACITY}u);return;}
 var lambda=0.;
 if(p.prevValid!=0u){let t=1u-p.tableCur;let ps=tableFind(t,k0,k1);if(ps!=NONE){lambda=bitcast<f32>(KU[p.uTabVal+p.tableSlots*t+ps]);}}
 let tangentObstacle=obstacle-dot(obstacle,n)*n;
 KF[4u*c]=vec4f(n,g0);KF[4u*c+1u]=vec4f(tangentObstacle,lambda);KF[4u*c+2u]=vec4f(0.,lambda,mu,radius);KF[4u*c+3u]=w;
 KU[p.kuCV+4u*c]=v.x;KU[p.kuCV+4u*c+1u]=v.y;KU[p.kuCV+4u*c+2u]=v.z;KU[p.kuCV+4u*c+3u]=v.w;
 KU[p.kuMeta+4u*c]=k0;KU[p.kuMeta+4u*c+1u]=k1;KU[p.kuMeta+4u*c+2u]=collider;KU[p.kuMeta+4u*c+3u]=slot;
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
 if((k0>>30u)==1u){
  let closest=closestTriangle(x0,x1,x2,x3);delta=x0-closest.point;fallback=cross(x2-x1,x3-x1);w=vec4f(1.,-closest.w.x,-closest.w.y,-closest.w.z);
  distance=kRadius(v.x)+ksf(p.sFaceR+k1);
 }else{
  let closest=closestSegments(x0,x1,x2,x3);delta=closest.a-closest.b;fallback=cross(x1-x0,x3-x2);w=vec4f(1.-closest.s,closest.s,-(1.-closest.t),-closest.t);
  distance=ksf(p.sEdge+4u*(k0&0x3FFFFFFFu)+2u)+ksf(p.sEdge+4u*k1+2u);
 }
 let old=KF[4u*c].xyz;let length0=length(delta);let f=length(fallback);
 var n=old;if(length0>1e-12){n=delta/length0;}else if(f>1e-12){n=fallback/f;}
 if(dot(n,old)<0.){n=-n;}
 KF[4u*c]=vec4f(n,dot(n,w.y*q1+w.z*q2+w.w*q3)-distance);KF[4u*c+3u]=w;
 let pen=distance-dot(n,delta);maxf(${K.K_PEN}u,pen);maxf(${K.K_REPPEN}u,pen);
}
@compute @workgroup_size(${WG}) fn relinearize(@builtin(global_invocation_id) g:vec3u,@builtin(num_workgroups) ng:vec3u){
 if(!roundOn()){return;}
 let count=atomicLoad(&C[${K.K_BASE}u]);
 for(var c=g.x;c<count;c+=ng.x*${WG}u){
  let ci=KU[p.kuMeta+4u*c+2u];if(ci==NONE){relinearizeSelf(c);continue;}
  let i=KU[p.kuCV+4u*c];let s=colliderSample(ci,sweptPoint(i,1.),0.);let old=KF[4u*c];let radius=KF[4u*c+2u].w;
  if(!(dot(s.n,old.xyz)>0.)){atomicOr(&C[${K.K_ERROR}u],${E_CROSSED}u);}
  let obstacle=cv3(cHeader(ci)+2u)*p.dt;
  KF[4u*c]=vec4f(s.n,(s.d-radius)-dot(s.n,xv(i)));KF[4u*c+1u]=vec4f(obstacle-dot(obstacle,s.n)*s.n,KF[4u*c+1u].w);
  maxf(${K.K_PEN}u,radius-s.d);maxf(${K.K_REPPEN}u,radius-s.d);
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
   for(var j=0u;j<4u;j++){let w=k.w[j];let v=k.v[j];if(w==0.||isFixed(v)){continue;}var s=vec3f(pred(v).w+attach(v).w);if(p.stiffnessValid!=0u||atomicLoad(&C[${K.K_OUTER}u])>0u){s=W[p.wK+v].xyz;}compliance+=w*w/dot(k.n*k.n,s);}
   if(!(compliance>0.&&compliance<3e38)){atomicOr(&C[${K.K_ERROR}u],${E_STIFFNESS}u);}
   r.x=10./compliance;
  }else if(stiffen){let gap=k.g0+dot(k.n,contactSum(k,0u));if(abs(min(gap,k.lambda/r.x))>p.tolerance){r.x*=10.;}}
  r.y=KF[4u*c+1u].w;
  KF[4u*c+2u]=r;
 }
}

// ---- per-vertex contact lists (CSR over free vertices with nonzero weight)
// A self pair lists all four vertices (its re-linearized weights may move off zero); a collider pair its one.
fn pairVertices(c:u32)->u32{return select(1u,4u,KU[p.kuMeta+4u*c+2u]==NONE);}
@compute @workgroup_size(${WG}) fn listCount(@builtin(global_invocation_id) g:vec3u,@builtin(num_workgroups) ng:vec3u){
 if(atomicLoad(&C[${K.K_ADDED}u])==0u){return;}
 let count=min(contactCount(),p.kMax);
 for(var c=g.x;c<count;c+=ng.x*${WG}u){let k=contactAt(c);for(var j=0u;j<pairVertices(c);j++){if(!isFixed(k.v[j])){atomicAdd(&KA[p.aCount+k.v[j]],1u);}}}
}
var<workgroup> scanBuffer:array<u32,${SCAN_WG}>;
fn scanLocal(l:u32,value:u32)->u32{
 scanBuffer[l]=value;
 for(var o=1u;o<${SCAN_WG}u;o<<=1u){workgroupBarrier();var add=0u;if(l>=o){add=scanBuffer[l-o];}workgroupBarrier();scanBuffer[l]+=add;}
 workgroupBarrier();let inclusive=scanBuffer[l];workgroupBarrier();return inclusive;
}
@compute @workgroup_size(${SCAN_WG}) fn listScan1(@builtin(global_invocation_id) g:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) wg:vec3u){
 if(uniformWord(l,${K.K_ADDED}u)==0u){return;}
 let i=g.x;var value=0u;if(i<p.n){value=atomicExchange(&KA[p.aCount+i],0u);}
 let inclusive=scanLocal(l,value);if(i<p.n){KU[p.kvStart+i]=inclusive-value;}
 if(l==${SCAN_WG - 1}u){KU[p.uBlocks+wg.x]=inclusive;}
}
@compute @workgroup_size(${SCAN_WG}) fn listScan2(@builtin(local_invocation_index) l:u32){
 if(uniformWord(l,${K.K_ADDED}u)==0u){return;}
 let blocks=(p.n+${SCAN_WG - 1}u)/${SCAN_WG}u;var carry=0u;
 for(var b0=0u;b0<blocks;b0+=${SCAN_WG}u){
  var value=0u;if(b0+l<blocks){value=KU[p.uBlocks+b0+l];}
  let inclusive=scanLocal(l,value);if(b0+l<blocks){KU[p.uBlocks+b0+l]=carry+inclusive-value;}
  carry+=workgroupUniformLoad(&scanBuffer[${SCAN_WG - 1}u]);
 }
 if(l==0u){KU[p.kvStart+p.n]=carry;}
}
@compute @workgroup_size(${SCAN_WG}) fn listScan3(@builtin(global_invocation_id) g:vec3u,@builtin(workgroup_id) wg:vec3u){
 if(atomicLoad(&C[${K.K_ADDED}u])==0u){return;}
 let i=g.x;if(i<p.n){KU[p.kvStart+i]+=KU[p.uBlocks+wg.x];}
}
@compute @workgroup_size(${WG}) fn listFill(@builtin(global_invocation_id) g:vec3u,@builtin(num_workgroups) ng:vec3u){
 if(atomicLoad(&C[${K.K_ADDED}u])==0u){return;}
 let count=min(contactCount(),p.kMax);
 for(var c=g.x;c<count;c+=ng.x*${WG}u){let k=contactAt(c);for(var j=0u;j<pairVertices(c);j++){let v=k.v[j];if(!isFixed(v)){KU[p.kvList+KU[p.kvStart+v]+atomicAdd(&KA[p.aCount+v],1u)]=(c<<2u)|j;}}}
}
// Each vertex's list in the order of its contacts' keys (unique in a pass) and slots, not
// the order threads appended them: the per-vertex contact sums (gradient, preconditioner,
// matvec gather) and the line search's owner sums then repeat bit for bit.
fn incidenceBefore(a:u32,b:u32)->bool{
 let ca=p.kuMeta+4u*(a>>2u);let cb=p.kuMeta+4u*(b>>2u);
 if(KU[ca]!=KU[cb]){return KU[ca]<KU[cb];}if(KU[ca+1u]!=KU[cb+1u]){return KU[ca+1u]<KU[cb+1u];}return (a&3u)<(b&3u);
}
@compute @workgroup_size(${WG}) fn listReset(@builtin(global_invocation_id) g:vec3u){
 if(atomicLoad(&C[${K.K_ADDED}u])==0u){return;}
 let i=g.x;if(i>=p.n){return;}
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
fn padBox(lo:vec3f,hi:vec3f,r:f32)->array<vec3f,2>{let pad=r+1e-6*(abs(lo)+abs(hi))+1e-9;return array<vec3f,2>(lo-pad,hi+pad);}
@compute @workgroup_size(${WG}) fn gridClear(@builtin(global_invocation_id) g:vec3u,@builtin(num_workgroups) ng:vec3u){
 if(atomicLoad(&C[${K.K_DETECT}u])==0u||p.selfCollision==0u){return;}
 for(var b=g.x;b<2u*p.gridBuckets;b+=ng.x*${WG}u){atomicStore(&KA[p.aHeads+b],0u);}
}
@compute @workgroup_size(${WG}) fn gridFaces(@builtin(global_invocation_id) g:vec3u){
 let f=g.x;if(f>=p.triangles||atomicLoad(&C[${K.K_DETECT}u])==0u||p.selfCollision==0u){return;}
 let id=triIds(f);var loB=min(approx(sweptPoint(id.x,0.)),approx(sweptPoint(id.x,1.)));var hiB=max(approx(sweptPoint(id.x,0.)),approx(sweptPoint(id.x,1.)));
 for(var j=1u;j<3u;j++){let v=id[j];loB=min(loB,min(approx(sweptPoint(v,0.)),approx(sweptPoint(v,1.))));hiB=max(hiB,max(approx(sweptPoint(v,0.)),approx(sweptPoint(v,1.))));}
 let box=padBox(loB,hiB,ksf(p.sFaceR+f));KF[p.kfFaceBox+2u*f]=vec4f(box[0],0);KF[p.kfFaceBox+2u*f+1u]=vec4f(box[1],0);
 insertEntry(0u,f,cellOf(box[0]),cellOf(box[1]));
}
fn edgeIds(e:u32)->vec2u{return vec2u(I[p.sEdge+4u*e],I[p.sEdge+4u*e+1u]);}
@compute @workgroup_size(${WG}) fn gridEdges(@builtin(global_invocation_id) g:vec3u){
 let e=g.x;if(e>=p.edges||atomicLoad(&C[${K.K_DETECT}u])==0u||p.selfCollision==0u){return;}
 let id=edgeIds(e);let a0=approx(sweptPoint(id.x,0.));let a1=approx(sweptPoint(id.x,1.));let b0=approx(sweptPoint(id.y,0.));let b1=approx(sweptPoint(id.y,1.));
 let box=padBox(min(min(a0,a1),min(b0,b1)),max(max(a0,a1),max(b0,b1)),ksf(p.sEdge+4u*e+2u));KF[p.kfEdgeBox+2u*e]=vec4f(box[0],0);KF[p.kfEdgeBox+2u*e+1u]=vec4f(box[1],0);
 insertEntry(1u,e,cellOf(box[0]),cellOf(box[1]));
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
 let k0=(1u<<30u)|i;if(held(k0,f)){return;}
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
 let k0=(2u<<30u)|ai;if(held(k0,bi)){return;}
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
  if(isFixed(i)||!kShell(i)||held(i,ci)){continue;}
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
  addContact(i,ci,vec4u(i,0u,0u,0u),vec4f(1,0,0,0),normal,g0,velocity*p.dt,mu,r,ci);
 }
}
`;
}
