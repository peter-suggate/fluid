/** Experimental aggregation multigrid preconditioner. The fine Newton operator is unchanged.
 * Topology is compiled at solver construction (including cut/load/undo rebuilds); numerical
 * blocks are rebuilt per Newton iteration. All tables/work append to existing GPU bindings. */
import type { ClothGPUTables } from './gpu-solver';
import type { ClothSolverModel } from './oracle';

export interface ClothMGLevel {
  readonly n: number;
  readonly blocks: number;
  readonly rows: number;
  readonly columns: number;
  readonly owners: number;
  readonly sourceOffsets: number;
  readonly sources: number;
  readonly parent: number;
  readonly children: number;
  readonly childList: number;
  readonly matrix: number;
  readonly b: number;
  readonly x0: number;
  readonly x1: number;
  readonly smoother: number;
}
export interface ClothMGLayout {
  readonly levels: readonly ClothMGLevel[];
  readonly u: number;
  readonly fusedLevel: number;
}
export const CLOTH_MG_KERNELS = ['pcgPrepare', 'mgAssemble', 'mgCoarsen', 'mgScale', 'mgInit', 'mgSmooth', 'mgRestrict', 'mgProlong', 'mgCoarseCycle', 'mgFinish'] as const;

/** Connected aggregates: never join separate cut pieces just because their coordinates are close. */
export function clothMGAggregates(rows: readonly (readonly number[])[], size = 4): { parent: Uint32Array; groups: number[][] } {
  if (!Number.isInteger(size) || size < 2) throw new Error('Cloth multigrid aggregate size must be at least two');
  const parent = new Uint32Array(rows.length).fill(0xffffffff), groups: number[][] = [];
  for (let seed = 0; seed < rows.length; seed++) {
    if (parent[seed] !== 0xffffffff) continue;
    const id = groups.length, group = [seed]; parent[seed] = id;
    for (let k = 0; k < group.length && group.length < size; k++) for (const j of rows[group[k]!]!) {
      if (parent[j] === 0xffffffff) { parent[j] = id; group.push(j); if (group.length === size) break; }
    }
    groups.push(group);
  }
  return { parent, groups };
}

export function withClothMultigrid(t: ClothGPUTables, model: ClothSolverModel, aggregateSize = 4, fusedRows = 256): ClothGPUTables {
  if (![4, 8, 16].includes(aggregateSize)) throw new Error('Experimental MG aggregate size must be 4, 8 or 16');
  if (![256, 512, 1024, 2048].includes(fusedRows)) throw new Error('Experimental MG fused rows must be 256, 512, 1024 or 2048');
  const N = model.topology.vertexCount, old = t.topo, p = t.params;
  const words = Array.from(old), levels: ClothMGLevel[] = [];
  const append = (a: ArrayLike<number>) => { const o = words.length; for (let i = 0; i < a.length; i++) words.push(a[i]!); return o; };
  let work = t.newtonRecords;
  const alloc = (n: number) => { const o = work; work += Math.max(n, 1); return o; };
  // Fine block sources: record and kind (0: three sliced-ELL rows; 1/3: scalar bending lane).
  let rowMaps = Array.from({ length: N }, () => new Map<number, number[]>());
  const rowOrder = p[68]!, slots = p[69]!, table = p[60]!;
  for (let v = 0; v < N; v++) {
    const slot = old[slots + v]!, patch = Math.floor(slot / 256), lane = slot % 256, tw = table + 8 * patch;
    const k1 = old[tw]!, k2 = old[tw + 1]!, halo = old[tw + 3]!, blocks = old[tw + 4]!, bends = old[tw + 5]!, cols = old[tw + 6]!;
    const vertex = (c: number) => c & 0x80000000 ? c & 0x7fffffff : c < 256 ? old[rowOrder + 256 * patch + c]! : old[halo + c - 256]!;
    const add = (j: number, record: number, kind: number) => { const a = rowMaps[v]!.get(j) ?? []; a.push(record, kind); rowMaps[v]!.set(j, a); };
    for (let k = 0; k < k1; k++) add(vertex(old[cols + k * 256 + lane]!), blocks + 3 * k * 256 + lane, 0);
    for (let k = 0; k < k2; k++) add(vertex(old[cols + (k1 + k) * 256 + lane]!), bends + Math.floor(k / 2) * 256 + lane, k % 2 ? 3 : 1);
  }
  for (let l = 0; l < 12; l++) {
    const n = rowMaps.length, rows = [0], columns: number[] = [], owners: number[] = [], offsets = [0], sources: number[] = [];
    for (let i = 0; i < n; i++) {
      for (const [j, list] of [...rowMaps[i]!.entries()].sort((a,b) => a[0]-b[0])) {
        columns.push(j); owners.push(i); sources.push(...list); offsets.push(sources.length);
      }
      rows.push(columns.length);
    }
    const adjacency = rowMaps.map(r => [...r.keys()]);
    const agg = clothMGAggregates(adjacency, aggregateSize);
    const childOffsets = [0], children: number[] = [];
    for (const g of agg.groups) { children.push(...g); childOffsets.push(children.length); }
    const level: ClothMGLevel = {
      n, blocks: columns.length, rows: append(rows), columns: append(columns), owners: append(owners),
      sourceOffsets: append(offsets), sources: append(sources), parent: append(agg.parent),
      children: append(childOffsets), childList: append(children), matrix: alloc(3 * columns.length), b: alloc(n), x0: alloc(n), x1: alloc(n), smoother: alloc(3*n),
    };
    levels.push(level);
    if (n <= 32 || agg.groups.length === n) break;
    const next = Array.from({ length: agg.groups.length }, () => new Map<number, number[]>());
    for (let k = 0; k < columns.length; k++) {
      const i = agg.parent[owners[k]!]!, j = agg.parent[columns[k]!]!;
      const a = next[i]!.get(j) ?? []; a.push(k); next[i]!.set(j, a);
    }
    rowMaps = next;
  }
  const u = alloc(N);
  const control = Math.ceil(work / 16) * 16;
  const params = t.params.slice(); params[55] = control;
  return { ...t, params, topo: new Uint32Array(words), newtonRecords: control + 16, pcg: { ...t.pcg, multigrid: { levels, u, fusedLevel: levels.findIndex(l => l.n <= fusedRows) < 0 ? levels.length - 1 : levels.findIndex(l => l.n <= fusedRows) } } };
}

export function clothMultigridWGSL(mg: ClothMGLayout, contact: boolean): string {
  const levels = mg.levels;
  return /* wgsl */`
const MG_U = ${mg.u}u;
struct MGLevel { n: u32, blocks: u32, rows: u32, columns: u32, owners: u32, sourceOffsets: u32, sources: u32,
  parent: u32, children: u32, childList: u32, matrix: u32, b: u32, x0: u32, x1: u32, smoother: u32 }
const MG_LEVELS = array<MGLevel, ${levels.length}>(
${levels.map(l => 'MGLevel(' + Object.values(l).map(v => v + 'u').join(',') + ')').join(',\n')});
fn mgBlock(l: MGLevel, k: u32) -> mat3x3f {
  let a = nw[l.matrix+3u*k].xyz; let b = nw[l.matrix+3u*k+1u].xyz; let c = nw[l.matrix+3u*k+2u].xyz;
  return transpose(mat3x3f(a,b,c));
}
fn mgPut(l: MGLevel, k: u32, h: mat3x3f) {
  let a = transpose(h);
  nw[l.matrix+3u*k] = vec4f(a[0],0.0); nw[l.matrix+3u*k+1u] = vec4f(a[1],0.0); nw[l.matrix+3u*k+2u] = vec4f(a[2],0.0);
}
fn mgGate() -> bool { let c = nw[nwControl()]; return c.x == 0.0 && c.y == 0.0; }
fn mgProduct(l: MGLevel, i: u32, x: u32) -> vec3f {
  var y = vec3f(0.0);
  for (var k = topo[l.rows+i]; k < topo[l.rows+i+1u]; k++) { y += mgBlock(l,k) * nw[x+topo[l.columns+k]].xyz; }
  return y;
}
fn mgDenom(l: MGLevel, i: u32) -> vec3f {
  var d = vec3f(0.0);
  for (var k = topo[l.rows+i]; k < topo[l.rows+i+1u]; k++) {
    let h = mgBlock(l,k); d += abs(h[0])+abs(h[1])+abs(h[2]);
  }
  return max(d, vec3f(1e-30));
}
@compute @workgroup_size(64) fn mgAssemble(@builtin(global_invocation_id) id: vec3u) {
  let l = MG_LEVELS[0]; let k = id.x; if (!mgGate() || k >= l.blocks) { return; }
  let i = topo[l.owners+k]; let j = topo[l.columns+k]; var h = mat3x3f();
  if (nwPinned(i) || nwPinned(j)) {
    if (i == j) { h = mat3x3f(1,0,0,0,1,0,0,0,1); }
  } else {
    // Every elastic block, including the diagonal, must use the same assembled
    // operator. The local vertex inverse uses a different PSD approximation;
    // mixing its diagonal with these off-diagonals can make MG indefinite.
    for (var s = topo[l.sourceOffsets+k]; s < topo[l.sourceOffsets+k+1u]; s += 2u) {
      let rec = topo[l.sources+s]; let kind = topo[l.sources+s+1u];
      if (kind == 0u) { h += transpose(mat3x3f(nw[rec].xyz,nw[rec+256u].xyz,nw[rec+512u].xyz)); }
      else { let b = nw[rec][kind]; h += mat3x3f(b,0,0,0,b,0,0,0,b); }
    }
    ${contact ? `// Add the PSD normal/friction diagonal explicitly, without changing
    // the fine Newton operator or borrowing a differently clamped elastic block.
    if (i == j) {
      var unusedForce = vec3f(0.0);
      ogcAccumulateVertexContact(i, pos(i), &unusedForce, &h);
    }` : ''}
  }
  mgPut(l,k,h);
}
@compute @workgroup_size(64) fn mgCoarsen(@builtin(global_invocation_id) id: vec3u) {
  let level = R.range.z; let l = MG_LEVELS[level]; let fine = MG_LEVELS[level-1u]; let k = id.x;
  if (!mgGate() || k >= l.blocks) { return; }
  var h = mat3x3f();
  for (var s = topo[l.sourceOffsets+k]; s < topo[l.sourceOffsets+k+1u]; s++) { h += mgBlock(fine,topo[l.sources+s]); }
  mgPut(l,k,h);
}
// Symmetric block-Jacobi smoother. Scale each inverse diagonal block so every
// absolute row sum of B*A is <= 0.9; its positive spectrum is then below one.
// This preserves directional stiffness without assuming an elasticity-specific damping constant.
fn mgSmoother(l: MGLevel, i: u32) -> mat3x3f {
  return transpose(mat3x3f(nw[l.smoother+3u*i].xyz,nw[l.smoother+3u*i+1u].xyz,nw[l.smoother+3u*i+2u].xyz));
}
@compute @workgroup_size(64) fn mgScale(@builtin(global_invocation_id) id: vec3u) {
  let l = MG_LEVELS[R.range.z]; let i = id.x; if (!mgGate() || i >= l.n) { return; }
  var d = mat3x3f();
  for (var k = topo[l.rows+i]; k < topo[l.rows+i+1u]; k++) { if (topo[l.columns+k] == i) { d = mgBlock(l,k); } }
  let determinantD = determinant(d); var b = mat3x3f();
  if (determinantD > 1e-30) { b = transpose(mat3x3f(cross(d[1],d[2]),cross(d[2],d[0]),cross(d[0],d[1]))) * (1.0/determinantD); }
  else { let q = 1.0/mgDenom(l,i); b = mat3x3f(q.x,0,0,0,q.y,0,0,0,q.z); }
  var bound = vec3f(0.0);
  for (var k = topo[l.rows+i]; k < topo[l.rows+i+1u]; k++) { let a = b * mgBlock(l,k); bound += abs(a[0])+abs(a[1])+abs(a[2]); }
  b *= 0.9/max(1e-30,max(bound.x,max(bound.y,bound.z)));
  let rows = transpose(b); nw[l.smoother+3u*i]=vec4f(rows[0],0.0); nw[l.smoother+3u*i+1u]=vec4f(rows[1],0.0); nw[l.smoother+3u*i+2u]=vec4f(rows[2],0.0);
}
@compute @workgroup_size(64) fn mgInit(@builtin(global_invocation_id) id: vec3u) {
  let l = MG_LEVELS[0]; let i = id.x; if (!mgGate() || i >= l.n) { return; }
  nw[l.b+i] = nw[nseg(1u + R.range.z % 2u,i)]; nw[l.x0+i] = vec4f(0.0);
}
@compute @workgroup_size(64) fn mgSmooth(@builtin(global_invocation_id) id: vec3u) {
  let l = MG_LEVELS[R.range.z/2u]; let i = id.x; if (!mgGate() || i >= l.n) { return; }
  let x = select(l.x0,l.x1,R.range.z%2u != 0u); let y = select(l.x1,l.x0,R.range.z%2u != 0u);
  let r = nw[l.b+i].xyz - mgProduct(l,i,x);
  nw[y+i] = vec4f(nw[x+i].xyz + mgSmoother(l,i) * r,0.0);
}
@compute @workgroup_size(64) fn mgRestrict(@builtin(global_invocation_id) id: vec3u) {
  let level = R.range.z; let fine = MG_LEVELS[level]; let coarse = MG_LEVELS[level+1u]; let i = id.x;
  if (!mgGate() || i >= coarse.n) { return; }
  var r = vec3f(0.0);
  for (var c = topo[fine.children+i]; c < topo[fine.children+i+1u]; c++) {
    let v = topo[fine.childList+c]; r += nw[fine.b+v].xyz - mgProduct(fine,v,fine.x0);
  }
  nw[coarse.b+i] = vec4f(r,0.0); nw[coarse.x0+i] = vec4f(0.0);
}
@compute @workgroup_size(64) fn mgProlong(@builtin(global_invocation_id) id: vec3u) {
  let level = R.range.z; let fine = MG_LEVELS[level]; let coarse = MG_LEVELS[level+1u]; let i = id.x;
  if (!mgGate() || i >= fine.n) { return; }
  nw[fine.x0+i] += vec4f(nw[coarse.x0+topo[fine.parent+i]].xyz,0.0);
}
// All coarse levels run in one workgroup, with storage barriers between dependent
// stages. This avoids dozens of tiny dispatches per preconditioner application.
fn mgSmoothRow(l: MGLevel, i: u32, parity: u32) {
  let x = select(l.x0,l.x1,parity != 0u); let y = select(l.x1,l.x0,parity != 0u);
  nw[y+i] = vec4f(nw[x+i].xyz + mgSmoother(l,i) * (nw[l.b+i].xyz-mgProduct(l,i,x)),0.0);
}
@compute @workgroup_size(256) fn mgCoarseCycle(@builtin(local_invocation_index) lane: u32) {
  let ctl = nwControlUniform(lane); if (ctl.x != 0.0 || ctl.y != 0.0) { return; }
  for (var level = ${mg.fusedLevel}u; level < ${levels.length-1}u; level++) {
    let l = MG_LEVELS[level]; let coarse = MG_LEVELS[level+1u];
    for (var sweep = 0u; sweep < 2u; sweep++) {
      for (var i = lane; i < l.n; i += 256u) { mgSmoothRow(l,i,sweep); }
      storageBarrier();
    }
    for (var i = lane; i < coarse.n; i += 256u) {
      var r = vec3f(0.0);
      for (var c = topo[l.children+i]; c < topo[l.children+i+1u]; c++) {
        let v = topo[l.childList+c]; r += nw[l.b+v].xyz-mgProduct(l,v,l.x0);
      }
      nw[coarse.b+i] = vec4f(r,0.0); nw[coarse.x0+i] = vec4f(0.0);
    }
    storageBarrier();
  }
  let bottom = MG_LEVELS[${levels.length-1}];
  for (var sweep = 0u; sweep < 32u; sweep++) {
    for (var i = lane; i < bottom.n; i += 256u) { mgSmoothRow(bottom,i,sweep%2u); }
    storageBarrier();
  }
  for (var lev = ${levels.length-1}u; lev > ${mg.fusedLevel}u; lev--) {
    let l = MG_LEVELS[lev-1u]; let coarse = MG_LEVELS[lev];
    for (var i = lane; i < l.n; i += 256u) { nw[l.x0+i] += vec4f(nw[coarse.x0+topo[l.parent+i]].xyz,0.0); }
    storageBarrier();
    for (var sweep = 0u; sweep < 2u; sweep++) {
      for (var i = lane; i < l.n; i += 256u) { mgSmoothRow(l,i,sweep); }
      storageBarrier();
    }
  }
}
@compute @workgroup_size(64) fn mgFinish(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x; if (!mgGate() || i >= MG_LEVELS[0].n) { return; }
  var u = nw[MG_LEVELS[0].x0+i].xyz;
  if (nwPinned(i)) { u = vec3f(0.0); }
  ${contact ? 'else { u = nwProjectP(i,u); }' : ''}
  nw[MG_U+i] = vec4f(u,0.0);
}
`;
}
