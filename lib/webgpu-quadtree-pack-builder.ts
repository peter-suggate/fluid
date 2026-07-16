/**
 * GPU-resident count -> scan -> emit implementation of Narita Sec. 4.2/4.4.
 *
 * This is deliberately an uncoupled, non-IC rebuild backend. Dynamic rigid
 * coupling and incomplete factors retain the CPU reference path. The output
 * byte layout is the projection solver's existing packed layout, so the
 * numerical kernels are shared and only topology construction moves devices.
 */

const INVALID = 0xffffffff;

export interface GPUQuadtreePackHints {
  dofCount: number;
  faceCount: number;
  pressureSampleCount: number;
}

export interface GPUQuadtreePackedResult {
  packed: {
    faces: Uint8Array<ArrayBuffer>; rowOffsets: Uint32Array<ArrayBuffer>; rowEntries: Uint8Array<ArrayBuffer>; matrixWords: Uint32Array<ArrayBuffer>;
    cellProjection: Float32Array<ArrayBuffer>; cellTopology: Uint32Array<ArrayBuffer>; factorColumns: Uint8Array<ArrayBuffer>; factorEntries: Uint8Array<ArrayBuffer>;
    factorAuxWords: Uint32Array<ArrayBuffer>; factorLevelCount: number; levelsOffset: number; rowOffsetsOffset: number; rowEntriesOffset: number;
    couplingByBodyOffset: number; couplingByDofOffset: number; couplingTableOffset: number; couplingBodyCount: number;
    couplingDistinctDofs: number; couplingBodyIndices: number[]; dofSamplesBase: number; mlsRowCount: number;
    cellPressureSamples: Uint32Array<ArrayBuffer>; icFactorization_ms: number; lineOffsetsBase: number; lineDofsBase: number;
    lineCount: number; blockTableOffset: number; blockCount: number;
  };
  leafCount: number;
  pressureSampleCount: number;
  dofCount: number;
  faceCount: number;
  ghostFaceCount: number;
  tallSegmentCount: number;
  maximumNeighborRatio: number;
  maximumFluidScale: number;
  gpuWall_ms: number;
  /** Exact-size resources handed straight to the pressure projection. */
  resident?: GPUQuadtreeResidentResources;
}

export interface GPUQuadtreeResidentResources {
  faces: GPUBuffer;
  rowOffsets: GPUBuffer;
  rowEntries: GPUBuffer;
  matrixBuffer: GPUBuffer;
  factorColumns: GPUBuffer;
  factorEntries: GPUBuffer;
  factorAux: GPUTexture;
  factorAuxWidth: number;
  cellProjection: GPUTexture;
  cellTopology: GPUTexture;
  cellPressureSamples: GPUTexture;
}

const common = /* wgsl */ `
struct Params { dims: vec4u, cell: vec4f, capacities: vec4u }
struct LeafMeta { counts: vec4u, offsets: vec4u }
fn index2(q: vec2u) -> u32 { return q.x + params.dims.x * q.y; }
fn index3(q: vec3u) -> u32 { return q.x + params.dims.x * (q.y + params.dims.y * q.z); }
fn leafOrigin(word: u32) -> vec2u { return vec2u(word & 1023u, (word >> 10u) & 1023u); }
fn leafSize(word: u32) -> u32 { return (word >> 20u) & 1023u; }
`;

export const quadtreeSegmentationPackShader = /* wgsl */ `
${common}
@group(0) @binding(0) var<storage, read> topology: array<u32>;
@group(0) @binding(1) var levelSet: texture_3d<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> cubicFlags: array<u32>;
@group(0) @binding(4) var<storage, read_write> leafMeta: array<LeafMeta>;
@group(0) @binding(5) var<storage, read_write> segments: array<vec4u>;
@group(0) @binding(6) var<storage, read_write> samples: array<vec4u>;
@group(0) @binding(7) var<storage, read_write> cellWords: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> auxWords: array<u32>;

fn phiAt(word: u32, y: u32) -> f32 {
  let origin = leafOrigin(word); let size = leafSize(word);
  let position = vec2f(origin) + vec2f(f32(size) * 0.5 - 0.5);
  let a = vec2u(floor(position)); let b = min(a + vec2u(1), params.dims.xz - vec2u(1)); let t = fract(position);
  let p00 = textureLoad(levelSet, vec3i(i32(a.x), i32(y), i32(a.y)), 0).x;
  let p10 = textureLoad(levelSet, vec3i(i32(b.x), i32(y), i32(a.y)), 0).x;
  let p01 = textureLoad(levelSet, vec3i(i32(a.x), i32(y), i32(b.y)), 0).x;
  let p11 = textureLoad(levelSet, vec3i(i32(b.x), i32(y), i32(b.y)), 0).x;
  return mix(mix(p00, p10, t.x), mix(p01, p11, t.x), t.y);
}
fn phiRangeAt(word: u32, y: u32) -> vec2f {
  let origin = leafOrigin(word); let size = leafSize(word); var result = vec2f(3.402823e38, -3.402823e38);
  for (var z = origin.y; z < origin.y + size; z += 1u) { for (var x = origin.x; x < origin.x + size; x += 1u) {
    let value = textureLoad(levelSet, vec3i(i32(x), i32(y), i32(z)), 0).x;
    result = vec2f(min(result.x, value), max(result.y, value));
  } }
  return result;
}
fn footprintCrossesInterface(word: u32, y: u32, h: f32) -> bool {
  let own = phiRangeAt(word, y);
  if (own.x <= h && own.y >= -h) { return true; }
  if (y > 0u) { let other = phiRangeAt(word, y - 1u); if ((own.x < 0.0) != (other.x < 0.0) || (own.x < 0.0 && other.y >= 0.0) || (other.x < 0.0 && own.y >= 0.0)) { return true; } }
  if (y + 1u < params.dims.y) { let other = phiRangeAt(word, y + 1u); if ((own.x < 0.0) != (other.x < 0.0) || (own.x < 0.0 && other.y >= 0.0) || (other.x < 0.0 && own.y >= 0.0)) { return true; } }
  return false;
}
fn flagIndex(slot: u32, y: u32) -> u32 { return slot * params.dims.y + y; }

@compute @workgroup_size(8, 8)
fn classifySegments(@builtin(global_invocation_id) gid: vec3u) {
  let q = gid.xy; if (any(q >= params.dims.xz)) { return; }
  let slot = index2(q); let word = topology[slot];
  if (any(leafOrigin(word) != q)) { leafMeta[slot] = LeafMeta(vec4u(0), vec4u(0)); return; }
  let h = min(params.cell.x, min(params.cell.y, params.cell.z));
  for (var surfaceY = 0u; surfaceY < params.dims.y; surfaceY += 1u) {
    let phi = phiAt(word, surfaceY); let liquid = phi < 0.0; let isInterface = footprintCrossesInterface(word, surfaceY, h);
    if (!isInterface) { continue; }
    var liquidY = surfaceY;
    if (!liquid && liquidY > 0u && phiAt(word, liquidY - 1u) < 0.0) { liquidY -= 1u; }
    var depth = 0u; var probe = i32(liquidY);
    loop { if (probe < 0 || phiAt(word, u32(probe)) >= 0.0) { break; } depth += 1u; probe -= 1; }
    let depthCells = max(1u, u32(ceil(f32(depth) * params.cell.w)));
    let first = select(0u, surfaceY - depthCells + 1u, surfaceY + 1u >= depthCells);
    for (var y = first; y <= surfaceY; y += 1u) { cubicFlags[flagIndex(slot, y)] = 1u; }
    let airEnd = min(params.dims.y - 1u, surfaceY + 2u);
    for (var y = surfaceY + 1u; y <= airEnd && y < params.dims.y; y += 1u) { if (phiAt(word, y) >= 0.0) { cubicFlags[flagIndex(slot, y)] = 1u; } }
  }
  var segmentCount = 0u; var sampleCount = 0u; var dofCount = 0u; var tallCount = 0u; var y = 0u;
  loop {
    if (y >= params.dims.y) { break; }
    let first = y; let cubic = cubicFlags[flagIndex(slot, y)] != 0u; let liquid = phiAt(word, y) < 0.0;
    if (!cubic) { loop { if (y + 1u >= params.dims.y || cubicFlags[flagIndex(slot, y + 1u)] != 0u || (phiAt(word, y + 1u) < 0.0) != liquid) { break; } y += 1u; } }
    let tall = !cubic && y > first; let count = select(1u, 2u, y > first);
    segmentCount += 1u; sampleCount += count; if (liquid) { dofCount += count; } if (tall) { tallCount += 1u; } y += 1u;
  }
  leafMeta[slot].counts = vec4u(segmentCount, sampleCount, dofCount, tallCount);
}

@compute @workgroup_size(1)
fn scanSegments(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x != 0u) { return; }
  let cells = params.dims.x * params.dims.z; var totals = vec4u(0); var leaves = 0u;
  for (var slot = 0u; slot < cells; slot += 1u) {
    let counts = leafMeta[slot].counts; leafMeta[slot].offsets = totals; totals += counts;
    if (counts.x > 0u) { leaves += 1u; }
  }
  var overflow = 0u;
  if (totals.x > params.capacities.x || totals.y > params.capacities.y || totals.z > params.capacities.z) { overflow = 1u; }
  leafMeta[cells] = LeafMeta(totals, vec4u(leaves, overflow, 0u, 0u));
}

fn emitSample(word: u32, y: u32, span: u32, sampleIndex: u32, dof: u32) {
  let origin = leafOrigin(word); let packed = origin.x | (origin.y << 10u) | (y << 20u);
  samples[sampleIndex] = vec4u(packed, leafSize(word), span, dof);
  if (dof != ${INVALID}u) {
    let cells = params.dims.x * params.dims.z; let dofBase = leafMeta[cells].counts.z + 3u + 4u * dof;
    auxWords[dofBase] = packed; auxWords[dofBase + 1u] = leafSize(word); auxWords[dofBase + 2u] = span; auxWords[dofBase + 3u] = 0u;
  }
}

@compute @workgroup_size(8, 8)
fn emitSegments(@builtin(global_invocation_id) gid: vec3u) {
  let q = gid.xy; if (any(q >= params.dims.xz)) { return; }
  let slot = index2(q); let word = topology[slot]; if (any(leafOrigin(word) != q)) { return; }
  let cells2 = params.dims.x * params.dims.z; if (leafMeta[cells2].offsets.y != 0u) { return; }
  let offsets = leafMeta[slot].offsets; var segmentCursor = offsets.x; var sampleCursor = offsets.y; var dofCursor = offsets.z; var y = 0u;
  loop {
    if (y >= params.dims.y) { break; }
    let first = y; let cubic = cubicFlags[flagIndex(slot, y)] != 0u; let liquid = phiAt(word, y) < 0.0;
    if (!cubic) { loop { if (y + 1u >= params.dims.y || cubicFlags[flagIndex(slot, y + 1u)] != 0u || (phiAt(word, y + 1u) < 0.0) != liquid) { break; } y += 1u; } }
    let last = y; let span = last - first + 1u; let bottomSample = sampleCursor; let bottomDof = select(${INVALID}u, dofCursor, liquid);
    emitSample(word, first, span, sampleCursor, bottomDof); sampleCursor += 1u; if (liquid) { dofCursor += 1u; }
    var topSample = bottomSample; var topDof = bottomDof;
    if (last > first) { topSample = sampleCursor; topDof = select(${INVALID}u, dofCursor, liquid); emitSample(word, last, span, sampleCursor, topDof); sampleCursor += 1u; if (liquid) { dofCursor += 1u; } }
    segments[segmentCursor] = vec4u(word, first | (last << 10u), bottomSample, topSample); segmentCursor += 1u;
    let origin = leafOrigin(word); let size = leafSize(word); let projectionBase = 0u; let pressureBase = 4u * params.dims.x * params.dims.y * params.dims.z; let topologyBase = 2u * pressureBase;
    for (var z = origin.y; z < origin.y + size && z < params.dims.z; z += 1u) { for (var yy = first; yy <= last; yy += 1u) { for (var x = origin.x; x < origin.x + size && x < params.dims.x; x += 1u) {
      let cell = index3(vec3u(x, yy, z));
      atomicStore(&cellWords[projectionBase + 4u * cell + 3u], bitcast<u32>(select(-f32(size), f32(size), liquid)));
      atomicStore(&cellWords[pressureBase + 4u * cell], bottomDof); atomicStore(&cellWords[pressureBase + 4u * cell + 1u], topDof);
      atomicStore(&cellWords[pressureBase + 4u * cell + 2u], first | (last << 10u)); atomicStore(&cellWords[pressureBase + 4u * cell + 3u], word);
      atomicStore(&cellWords[topologyBase + 2u * cell], word); atomicStore(&cellWords[topologyBase + 2u * cell + 1u], first | ((last + 1u) << 10u));
    } } }
    y += 1u;
  }
}
`;

export const quadtreeFacePackShader = /* wgsl */ `
${common}
struct FaceMeta { counts: vec4u, offsets: vec4u }
struct AtomicPair { x: atomic<u32>, y: atomic<u32> }
struct Face { nodes: vec4u, coefficients: vec4f, bounds: vec4u, packed: u32, solidFlux: f32, weights: vec2f, sampleCells: vec4u, sampleSpans: vec4u, flux: f32, mlsMean: f32, volume: f32 }
struct Interp { a: u32, b: u32, wa: f32, wb: f32, count: u32 }
@group(0) @binding(0) var<storage, read> topology: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> leafMeta: array<LeafMeta>;
@group(0) @binding(3) var<storage, read> segments: array<vec4u>;
@group(0) @binding(4) var<storage, read> samples: array<vec4u>;
@group(0) @binding(5) var<storage, read_write> faceMeta: array<FaceMeta>;
@group(0) @binding(6) var<storage, read_write> faces: array<Face>;
@group(0) @binding(7) var<storage, read_write> rowCounts: array<AtomicPair>;
@group(0) @binding(8) var<storage, read_write> cellWords: array<atomic<u32>>;

fn ownsBoundary(q: vec2u, axis: u32) -> bool {
  let word = topology[index2(q)]; let origin = leafOrigin(word); let size = leafSize(word);
  if (axis == 0u) { return q.x + 1u == origin.x + size; }
  return q.y + 1u == origin.y + size;
}
fn uniquePair(q: vec2u, other: vec2u, axis: u32) -> bool {
  if (axis == 0u && q.y > 0u) { return topology[index2(q - vec2u(0, 1))] != topology[index2(q)] || topology[index2(other - vec2u(0, 1))] != topology[index2(other)]; }
  if (axis == 2u && q.x > 0u) { return topology[index2(q - vec2u(1, 0))] != topology[index2(q)] || topology[index2(other - vec2u(1, 0))] != topology[index2(other)]; }
  return true;
}
fn pairFaceCount(leftWord: u32, rightWord: u32) -> u32 {
  let leftSlot = index2(leafOrigin(leftWord)); let rightSlot = index2(leafOrigin(rightWord));
  let lm = leafMeta[leftSlot]; let rm = leafMeta[rightSlot]; var li = 0u; var ri = 0u; var count = 0u;
  loop {
    if (li >= lm.counts.x || ri >= rm.counts.x) { break; }
    let ls = segments[lm.offsets.x + li]; let rs = segments[rm.offsets.x + ri]; let lf = ls.y & 1023u; let ll = (ls.y >> 10u) & 1023u; let rf = rs.y & 1023u; let rl = (rs.y >> 10u) & 1023u;
    let y0 = max(lf, rf); let y1 = min(ll + 1u, rl + 1u);
    if (y1 <= y0) { if (ll < rl) { li += 1u; } else { ri += 1u; } continue; }
    count += 1u;
    if (ll + 1u == y1) { li += 1u; } if (rl + 1u == y1) { ri += 1u; }
  }
  return count;
}

@compute @workgroup_size(8, 8)
fn countFaces(@builtin(global_invocation_id) gid: vec3u) {
  let q = gid.xy; if (any(q >= params.dims.xz)) { return; }
  let slot = index2(q); var counts = vec4u(0); let word = topology[slot];
  if (q.x + 1u < params.dims.x) { let other = q + vec2u(1, 0); let right = topology[index2(other)]; if (right != word && ownsBoundary(q, 0u) && uniquePair(q, other, 0u)) { counts.x = pairFaceCount(word, right); } }
  if (q.y + 1u < params.dims.z) { let other = q + vec2u(0, 1); let right = topology[index2(other)]; if (right != word && ownsBoundary(q, 2u) && uniquePair(q, other, 2u)) { counts.y = pairFaceCount(word, right); } }
  if (all(leafOrigin(word) == q)) {
    counts.z = select(0u, leafMeta[slot].counts.y - 1u, leafMeta[slot].counts.y > 0u);
    let first = leafMeta[slot].offsets.y; let count = leafMeta[slot].counts.y;
    for (var i = 0u; i + 1u < count; i += 1u) { let a = (samples[first + i].x >> 20u) & 1023u; let b = (samples[first + i + 1u].x >> 20u) & 1023u; if (b > a + 1u) { counts.w += 1u; } }
  }
  faceMeta[slot].counts = counts;
}

@compute @workgroup_size(1)
fn scanFaces(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x != 0u) { return; } let cells = params.dims.x * params.dims.z; var cursor = 0u; var ghosts = 0u;
  for (var slot = 0u; slot < cells; slot += 1u) { faceMeta[slot].offsets.x = cursor; cursor += faceMeta[slot].counts.x; ghosts += faceMeta[slot].counts.w; }
  for (var slot = 0u; slot < cells; slot += 1u) { faceMeta[slot].offsets.y = cursor; cursor += faceMeta[slot].counts.y; }
  for (var slot = 0u; slot < cells; slot += 1u) { faceMeta[slot].offsets.z = cursor; cursor += faceMeta[slot].counts.z; }
  faceMeta[cells] = FaceMeta(vec4u(cursor, ghosts, 0u, 0u), vec4u(select(0u, 1u, cursor > params.capacities.w), 0u, 0u, 0u));
}

fn interpolate(slot: u32, queryY: f32) -> Interp {
  let columnMeta = leafMeta[slot]; var lower = columnMeta.offsets.y; var upper = lower + columnMeta.counts.y - 1u;
  for (var i = 0u; i < columnMeta.counts.y; i += 1u) { let sample = columnMeta.offsets.y + i; let y = (samples[sample].x >> 20u) & 1023u; if (f32(y) <= queryY) { lower = sample; } if (f32(y) >= queryY) { upper = sample; break; } }
  if (lower == upper) { return Interp(lower, ${INVALID}u, 1.0, 0.0, 1u); }
  let ly = f32((samples[lower].x >> 20u) & 1023u); let uy = f32((samples[upper].x >> 20u) & 1023u); let t = (queryY - ly) / max(1.0, uy - ly);
  return Interp(lower, upper, 1.0 - t, t, 2u);
}
fn putNode(face: ptr<function, Face>, slot: u32, sampleId: u32, coefficient: f32) {
  let sample = samples[sampleId]; (*face).nodes[slot] = sample.w; (*face).coefficients[slot] = coefficient; (*face).sampleCells[slot] = sample.x; (*face).sampleSpans[slot] = sample.y;
}
fn finishFace(faceId: u32, face: Face) {
  faces[faceId] = face; var activeCount = 0u; let count = (face.packed >> 18u) & 7u;
  for (var slot = 0u; slot < count; slot += 1u) { if (face.nodes[slot] != ${INVALID}u) { activeCount += 1u; } }
  for (var slot = 0u; slot < count; slot += 1u) { let row = face.nodes[slot]; if (row != ${INVALID}u) { atomicAdd(&rowCounts[row].x, 1u); atomicAdd(&rowCounts[row].y, activeCount); } }
}
fn emitHorizontal(q: vec2u, axis: u32, firstFace: u32) {
  let other = q + select(vec2u(0, 1), vec2u(1, 0), axis == 0u); let leftWord = topology[index2(q)]; let rightWord = topology[index2(other)];
  let lo = leafOrigin(leftWord); let ro = leafOrigin(rightWord); let ls = leafSize(leftWord); let rs = leafSize(rightWord);
  let transverseStart = select(max(lo.x, ro.x), max(lo.y, ro.y), axis == 0u); let transverseEnd = select(min(lo.x + ls, ro.x + rs), min(lo.y + ls, ro.y + rs), axis == 0u); let span = transverseEnd - transverseStart;
  let leftSlot = index2(lo); let rightSlot = index2(ro); let lm = leafMeta[leftSlot]; let rm = leafMeta[rightSlot]; var li = 0u; var ri = 0u; var cursor = firstFace;
  loop {
    if (li >= lm.counts.x || ri >= rm.counts.x) { break; }
    let lseg = segments[lm.offsets.x + li]; let rseg = segments[rm.offsets.x + ri]; let lf = lseg.y & 1023u; let ll = (lseg.y >> 10u) & 1023u; let rf = rseg.y & 1023u; let rl = (rseg.y >> 10u) & 1023u;
    let y0 = max(lf, rf); let y1 = min(ll + 1u, rl + 1u);
    if (y1 <= y0) { if (ll < rl) { li += 1u; } else { ri += 1u; } continue; }
    {
      let distance = select(0.5 * f32(ls + rs) * params.cell.z, 0.5 * f32(ls + rs) * params.cell.x, axis == 0u); let queryY = 0.5 * f32(y0 + y1) - 0.5;
      let l = interpolate(leftSlot, queryY); let r = interpolate(rightSlot, queryY); var face = Face(vec4u(${INVALID}u), vec4f(0), vec4u(select(transverseStart, q.x, axis == 0u), select(transverseStart, q.y, axis == 2u), y0, y1), 0u, 0.0, vec2f(0), vec4u(0), vec4u(0), 0.0, 0.0, 0.0); var node = 0u;
      putNode(&face, node, l.a, -l.wa / distance); node += 1u; if (l.count == 2u) { putNode(&face, node, l.b, -l.wb / distance); node += 1u; }
      putNode(&face, node, r.a, r.wa / distance); node += 1u; if (r.count == 2u) { putNode(&face, node, r.b, r.wb / distance); node += 1u; }
      let volume = distance * f32(y1 - y0) * params.cell.y * f32(span) * select(params.cell.x, params.cell.z, axis == 0u);
      face.packed = span | (axis << 16u) | (node << 18u); face.weights = vec2f(volume); face.volume = volume; finishFace(cursor, face);
      let projectionBase = 0u; for (var y = y0; y < y1; y += 1u) { for (var t = 0u; t < span; t += 1u) { let x = select(transverseStart + t, q.x, axis == 0u); let z = select(transverseStart + t, q.y, axis == 2u); if (x < params.dims.x && z < params.dims.z) { atomicStore(&cellWords[projectionBase + 4u * index3(vec3u(x, y, z)) + axis], bitcast<u32>(f32(cursor + 1u))); } } }
      cursor += 1u;
    }
    if (ll + 1u == y1) { li += 1u; } if (rl + 1u == y1) { ri += 1u; }
  }
}

@compute @workgroup_size(8, 8)
fn emitFaces(@builtin(global_invocation_id) gid: vec3u) {
  let q = gid.xy; if (any(q >= params.dims.xz)) { return; } let cells = params.dims.x * params.dims.z; if (faceMeta[cells].offsets.x != 0u || leafMeta[cells].offsets.y != 0u) { return; }
  let slot = index2(q); let fm = faceMeta[slot]; if (fm.counts.x > 0u) { emitHorizontal(q, 0u, fm.offsets.x); } if (fm.counts.y > 0u) { emitHorizontal(q, 2u, fm.offsets.y); }
  let word = topology[slot]; if (any(leafOrigin(word) != q)) { return; } let columnMeta = leafMeta[slot]; let size = leafSize(word);
  for (var i = 0u; i + 1u < columnMeta.counts.y; i += 1u) {
    let bottom = samples[columnMeta.offsets.y + i]; let top = samples[columnMeta.offsets.y + i + 1u]; let by = (bottom.x >> 20u) & 1023u; let ty = (top.x >> 20u) & 1023u; let distance = f32(ty - by) * params.cell.y; if (distance <= 0.0) { continue; }
    let faceId = fm.offsets.z + i; let ghost = ty > by + 1u; let volume = distance * f32(size * size) * params.cell.x * params.cell.z;
    var face = Face(vec4u(bottom.w, top.w, ${INVALID}u, ${INVALID}u), vec4f(-1.0 / distance, 1.0 / distance, 0.0, 0.0), vec4u(q.x, q.y, by, ty), size | (1u << 16u) | (2u << 18u) | (select(0u, 1u, ghost) << 21u), 0.0, vec2f(volume), vec4u(bottom.x, top.x, 0u, 0u), vec4u(bottom.y, top.y, 0u, 0u), 0.0, 0.0, volume);
    finishFace(faceId, face); for (var z = q.y; z < q.y + size && z < params.dims.z; z += 1u) { for (var x = q.x; x < q.x + size && x < params.dims.x; x += 1u) { for (var y = by; y < ty; y += 1u) { atomicStore(&cellWords[4u * index3(vec3u(x, y, z)) + 1u], bitcast<u32>(f32(faceId + 1u))); } } }
  }
}
`;

export const quadtreeCsrPackShader = /* wgsl */ `
${common}
struct FaceMeta { counts: vec4u, offsets: vec4u }
struct AtomicPair { x: atomic<u32>, y: atomic<u32> }
struct Face { nodes: vec4u, coefficients: vec4f, bounds: vec4u, packed: u32, solidFlux: f32, weights: vec2f, sampleCells: vec4u, sampleSpans: vec4u, flux: f32, mlsMean: f32, volume: f32 }
struct Entry { face: u32, coefficient: f32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> leafMeta: array<LeafMeta>;
@group(0) @binding(2) var<storage, read_write> faceMeta: array<FaceMeta>;
@group(0) @binding(3) var<storage, read> faces: array<Face>;
@group(0) @binding(4) var<storage, read_write> rowCounts: array<AtomicPair>;
@group(0) @binding(5) var<storage, read_write> rowCursors: array<AtomicPair>;
@group(0) @binding(6) var<storage, read_write> rowOffsets: array<u32>;
@group(0) @binding(7) var<storage, read_write> rowEntries: array<Entry>;
@group(0) @binding(8) var<storage, read_write> matrixWords: array<u32>;

@compute @workgroup_size(1)
fn scanRows(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x != 0u) { return; } let cells = params.dims.x * params.dims.z; let dofs = leafMeta[cells].counts.z; var incidents = 0u; var matrices = 0u;
  for (var row = 0u; row < dofs; row += 1u) { rowOffsets[row] = incidents; matrixWords[row] = matrices; incidents += atomicLoad(&rowCounts[row].x); matrices += atomicLoad(&rowCounts[row].y); }
  rowOffsets[dofs] = incidents; matrixWords[dofs] = matrices; faceMeta[cells].offsets.y = incidents; faceMeta[cells].offsets.z = matrices;
}

@compute @workgroup_size(128)
fn emitCsr(@builtin(global_invocation_id) gid: vec3u) {
  let cells = params.dims.x * params.dims.z; let faceId = gid.x; if (faceId >= faceMeta[cells].counts.x) { return; } let dofs = leafMeta[cells].counts.z; let face = faces[faceId]; let count = (face.packed >> 18u) & 7u;
  for (var a = 0u; a < count; a += 1u) {
    let row = face.nodes[a]; if (row == ${INVALID}u) { continue; }
    let incident = atomicAdd(&rowCursors[row].x, 1u); rowEntries[rowOffsets[row] + incident] = Entry(faceId, face.coefficients[a]);
    for (var b = 0u; b < count; b += 1u) { let column = face.nodes[b]; if (column == ${INVALID}u) { continue; } let entry = atomicAdd(&rowCursors[row].y, 1u); let base = dofs + 1u + 4u * (matrixWords[row] + entry); matrixWords[base] = column; matrixWords[base + 1u] = faceId | (b << 30u); matrixWords[base + 2u] = 0u; matrixWords[base + 3u] = bitcast<u32>(face.coefficients[a] * face.coefficients[b]); }
  }
}
`;

/** GPU-only conversion from the packer's dense word field into solver textures. */
export const quadtreePackTextureShader = /* wgsl */ `
struct Params { dims: vec4u, cell: vec4f, capacities: vec4u }
@group(0) @binding(0) var<storage, read> cellWords: array<u32>;
@group(0) @binding(1) var projectionOut: texture_storage_3d<rgba32float, write>;
@group(0) @binding(2) var pressureSamplesOut: texture_storage_3d<rgba32uint, write>;
@group(0) @binding(3) var topologyOut: texture_storage_3d<rg32uint, write>;
@group(0) @binding(4) var<uniform> params: Params;
fn index3(q: vec3u) -> u32 { return q.x + params.dims.x * (q.y + params.dims.y * q.z); }
@compute @workgroup_size(4, 4, 4)
fn unpackCellFields(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= params.dims.xyz)) { return; }
  let cell = index3(gid); let cells = params.dims.x * params.dims.y * params.dims.z;
  let projectionBase = 4u * cell; let pressureBase = 4u * cells + 4u * cell; let topologyBase = 8u * cells + 2u * cell;
  textureStore(projectionOut, vec3i(gid), bitcast<vec4f>(vec4u(cellWords[projectionBase], cellWords[projectionBase + 1u], cellWords[projectionBase + 2u], cellWords[projectionBase + 3u])));
  textureStore(pressureSamplesOut, vec3i(gid), vec4u(cellWords[pressureBase], cellWords[pressureBase + 1u], cellWords[pressureBase + 2u], cellWords[pressureBase + 3u]));
  textureStore(topologyOut, vec3i(gid), vec4u(cellWords[topologyBase], cellWords[topologyBase + 1u], 0u, 0u));
}
`;

type BufferSet = {
  topology: GPUBuffer; flags: GPUBuffer; leafMeta: GPUBuffer; segments: GPUBuffer; samples: GPUBuffer; faceMeta: GPUBuffer;
  faces: GPUBuffer; rowCounts: GPUBuffer; rowCursors: GPUBuffer; rowOffsets: GPUBuffer; rowEntries: GPUBuffer;
  matrix: GPUBuffer; cellWords: GPUBuffer; aux: GPUBuffer; controlReadback: GPUBuffer;
};

export class WebGPUQuadtreePackBuilder {
  private readonly params: GPUBuffer;
  private readonly segmentationLayout: GPUBindGroupLayout; private readonly faceLayout: GPUBindGroupLayout; private readonly csrLayout: GPUBindGroupLayout;
  private readonly classifyPipeline: GPUComputePipeline; private readonly scanSegmentsPipeline: GPUComputePipeline; private readonly emitSegmentsPipeline: GPUComputePipeline;
  private readonly countFacesPipeline: GPUComputePipeline; private readonly scanFacesPipeline: GPUComputePipeline; private readonly emitFacesPipeline: GPUComputePipeline;
  private readonly scanRowsPipeline: GPUComputePipeline; private readonly emitCsrPipeline: GPUComputePipeline;
  private readonly textureLayout: GPUBindGroupLayout; private readonly unpackCellFieldsPipeline: GPUComputePipeline;
  private buffers?: BufferSet; private key = "";

  constructor(private readonly device: GPUDevice, private readonly dims: { nx: number; ny: number; nz: number }, private readonly cell: { x: number; y: number; z: number }, private readonly opticalDepthFraction: number) {
    this.params = device.createBuffer({ label: "GPU quadtree pack parameters", size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const storage = (readOnly = false): GPUBindGroupLayoutEntry["buffer"] => ({ type: readOnly ? "read-only-storage" : "storage" });
    this.segmentationLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: storage(true) }, { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }, ...[3, 4, 5, 6, 7, 8].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: storage() }))
    ] });
    this.faceLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: storage(true) }, { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ...[2, 3, 4].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: storage(true) })), ...[5, 6, 7, 8].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: storage() }))
    ] });
    this.csrLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }, { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: storage(true) },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: storage() }, { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: storage(true) },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: storage() }, ...[5, 6, 7, 8].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: storage() }))
    ] });
    const pipelines = (code: string, layout: GPUBindGroupLayout, entries: string[]) => {
      const shaderModule = device.createShaderModule({ code }); const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
      return Object.fromEntries(entries.map((entryPoint) => [entryPoint, device.createComputePipeline({ label: `GPU quadtree pack ${entryPoint}`, layout: pipelineLayout, compute: { module: shaderModule, entryPoint } })]));
    };
    const segmentation = pipelines(quadtreeSegmentationPackShader, this.segmentationLayout, ["classifySegments", "scanSegments", "emitSegments"]);
    this.classifyPipeline = segmentation.classifySegments; this.scanSegmentsPipeline = segmentation.scanSegments; this.emitSegmentsPipeline = segmentation.emitSegments;
    const faces = pipelines(quadtreeFacePackShader, this.faceLayout, ["countFaces", "scanFaces", "emitFaces"]);
    this.countFacesPipeline = faces.countFaces; this.scanFacesPipeline = faces.scanFaces; this.emitFacesPipeline = faces.emitFaces;
    const csr = pipelines(quadtreeCsrPackShader, this.csrLayout, ["scanRows", "emitCsr"]);
    this.scanRowsPipeline = csr.scanRows; this.emitCsrPipeline = csr.emitCsr;
    this.textureLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: storage(true) },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32uint", viewDimension: "3d" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rg32uint", viewDimension: "3d" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
    ] });
    const textureModule = device.createShaderModule({ code: quadtreePackTextureShader });
    this.unpackCellFieldsPipeline = device.createComputePipeline({ label: "GPU quadtree pack unpack cell fields", layout: device.createPipelineLayout({ bindGroupLayouts: [this.textureLayout] }), compute: { module: textureModule, entryPoint: "unpackCellFields" } });
  }

  private ensure(hints: GPUQuadtreePackHints) {
    const { nx, ny, nz } = this.dims, cells2 = nx * nz, cells3 = cells2 * ny;
    const dofCapacity = Math.max(64, Math.min(2 * cells3, Math.ceil(hints.dofCount * 1.75)));
    const sampleCapacity = Math.max(128, Math.min(2 * cells3, Math.ceil(hints.pressureSampleCount * 1.75)));
    const segmentCapacity = Math.max(cells2, Math.min(cells3, sampleCapacity));
    const faceCapacity = Math.max(128, Math.min(4 * cells3, Math.ceil(hints.faceCount * 1.75)));
    const entryCapacity = 4 * faceCapacity, matrixCapacity = 16 * faceCapacity;
    const key = [dofCapacity, sampleCapacity, segmentCapacity, faceCapacity].join(":"); if (this.buffers && this.key === key) return { dofCapacity, sampleCapacity, segmentCapacity, faceCapacity, entryCapacity, matrixCapacity };
    this.destroyBuffers(); this.key = key; const rw = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const make = (label: string, size: number, usage = rw) => this.device.createBuffer({ label, size: Math.max(4, Math.ceil(size / 4) * 4), usage });
    this.buffers = {
      topology: make("GPU pack topology", cells2 * 4), flags: make("GPU pack cubic flags", cells3 * 4), leafMeta: make("GPU pack leaf metadata", (cells2 + 1) * 32),
      segments: make("GPU pack segments", segmentCapacity * 16), samples: make("GPU pack samples", sampleCapacity * 16), faceMeta: make("GPU pack face metadata", (cells2 + 1) * 32),
      faces: make("GPU pack faces", faceCapacity * 112), rowCounts: make("GPU pack row counts", dofCapacity * 8), rowCursors: make("GPU pack row cursors", dofCapacity * 8),
      rowOffsets: make("GPU pack row offsets", (dofCapacity + 1) * 4), rowEntries: make("GPU pack row entries", entryCapacity * 8), matrix: make("GPU pack matrix", (dofCapacity + 1 + 4 * matrixCapacity) * 4),
      cellWords: make("GPU pack cell fields", cells3 * 10 * 4), aux: make("GPU pack auxiliary words", (5 * dofCapacity + 4) * 4),
      controlReadback: make("GPU pack control readback", 64, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ)
    };
    return { dofCapacity, sampleCapacity, segmentCapacity, faceCapacity, entryCapacity, matrixCapacity };
  }

  async build(packedCells: Uint32Array, levelSet: GPUTexture, hints: GPUQuadtreePackHints, resident = false, retriesRemaining = 2): Promise<GPUQuadtreePackedResult | undefined> {
    const startedAt = performance.now(), capacities = this.ensure(hints), b = this.buffers!, { nx, ny, nz } = this.dims, cells2 = nx * nz, cells3 = cells2 * ny;
    this.device.queue.writeBuffer(b.topology, 0, packedCells.buffer as ArrayBuffer, packedCells.byteOffset, packedCells.byteLength); const paramData = new ArrayBuffer(48);
    new Uint32Array(paramData, 0, 4).set([nx, ny, nz, 0]); new Float32Array(paramData, 16, 4).set([this.cell.x, this.cell.y, this.cell.z, Math.max(0, this.opticalDepthFraction)]);
    new Uint32Array(paramData, 32, 4).set([capacities.segmentCapacity, capacities.sampleCapacity, capacities.dofCapacity, capacities.faceCapacity]); this.device.queue.writeBuffer(this.params, 0, paramData);
    const group = (layout: GPUBindGroupLayout, entries: Array<GPUBuffer | GPUTexture>) => this.device.createBindGroup({ layout, entries: entries.map((resource, binding) => ({ binding, resource: "createView" in resource ? resource.createView() : { buffer: resource } })) });
    const segmentationGroup = group(this.segmentationLayout, [b.topology, levelSet, this.params, b.flags, b.leafMeta, b.segments, b.samples, b.cellWords, b.aux]);
    const faceGroup = group(this.faceLayout, [b.topology, this.params, b.leafMeta, b.segments, b.samples, b.faceMeta, b.faces, b.rowCounts, b.cellWords]);
    const csrGroup = group(this.csrLayout, [this.params, b.leafMeta, b.faceMeta, b.faces, b.rowCounts, b.rowCursors, b.rowOffsets, b.rowEntries, b.matrix]);
    const encoder = this.device.createCommandEncoder({ label: "GPU quadtree segmentation/face/CSR pack" });
    for (const buffer of [b.flags, b.leafMeta, b.faceMeta, b.rowCounts, b.rowCursors, b.cellWords, b.aux]) encoder.clearBuffer(buffer);
    const dispatch = (pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, x: number, y = 1) => { const pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup); pass.dispatchWorkgroups(x, y); pass.end(); };
    dispatch(this.classifyPipeline, segmentationGroup, Math.ceil(nx / 8), Math.ceil(nz / 8)); dispatch(this.scanSegmentsPipeline, segmentationGroup, 1); dispatch(this.emitSegmentsPipeline, segmentationGroup, Math.ceil(nx / 8), Math.ceil(nz / 8));
    dispatch(this.countFacesPipeline, faceGroup, Math.ceil(nx / 8), Math.ceil(nz / 8)); dispatch(this.scanFacesPipeline, faceGroup, 1); dispatch(this.emitFacesPipeline, faceGroup, Math.ceil(nx / 8), Math.ceil(nz / 8));
    dispatch(this.scanRowsPipeline, csrGroup, 1); dispatch(this.emitCsrPipeline, csrGroup, Math.ceil(capacities.faceCapacity / 128));
    encoder.copyBufferToBuffer(b.leafMeta, cells2 * 32, b.controlReadback, 0, 32); encoder.copyBufferToBuffer(b.faceMeta, cells2 * 32, b.controlReadback, 32, 32);
    this.device.queue.submit([encoder.finish()]); await b.controlReadback.mapAsync(GPUMapMode.READ);
    let leafControl: Uint32Array, faceControl: Uint32Array;
    try { const mapped = b.controlReadback.getMappedRange(); leafControl = new Uint32Array(mapped, 0, 8).slice(); faceControl = new Uint32Array(mapped, 32, 8).slice(); } finally { b.controlReadback.unmap(); }
    const pressureSampleCount = leafControl[1], dofCount = leafControl[2], tallSegmentCount = leafControl[3], leafCount = leafControl[4];
    const faceCount = faceControl[0], ghostFaceCount = faceControl[1], entryCount = faceControl[5], matrixEntryCount = faceControl[6];
    // A non-empty owner map always has at least one leaf and one vertical
    // pressure sample. Treat an all-zero readback as device loss/corruption,
    // not as a valid empty projection, and fall back to the CPU oracle.
    if (leafCount === 0 || pressureSampleCount === 0) return undefined;
    if (leafControl[5] !== 0 || faceControl[4] !== 0 || entryCount > capacities.entryCapacity || matrixEntryCount > capacities.matrixCapacity) {
      // The CSR and symmetric matrix can grow faster than the face count at
      // T-junctions. Feed their observed requirements back into the shared
      // face-capacity hint instead of retrying with only the emitted faces;
      // two bounded retries cover abrupt dam-front remeshes without falling
      // through to the much slower CPU worker.
      const faceHintFromEntries = Math.ceil(entryCount / (4 * 1.5));
      const faceHintFromMatrix = Math.ceil(matrixEntryCount / (16 * 1.5));
      return retriesRemaining > 0 ? this.build(packedCells, levelSet, {
        dofCount: Math.max(hints.dofCount * 2, dofCount),
        faceCount: Math.max(hints.faceCount * 2, faceCount, faceHintFromEntries, faceHintFromMatrix),
        pressureSampleCount: Math.max(hints.pressureSampleCount * 2, pressureSampleCount)
      }, resident, retriesRemaining - 1) : undefined;
    }
    const faceBytes = faceCount * 112, rowOffsetBytes = (dofCount + 1) * 4, rowEntryBytes = entryCount * 8, matrixBytes = (dofCount + 1 + 4 * matrixEntryCount) * 4;
    const cellProjectionBytes = cells3 * 16, cellPressureBytes = cells3 * 16, cellTopologyBytes = cells3 * 8, auxWords = dofCount + 3 + 4 * dofCount, auxBytes = auxWords * 4;
    let maximumNeighborRatio = 1;
    for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) for (const [qx, qz] of [[x + 1, z], [x, z + 1]] as const) if (qx < nx && qz < nz) { const a = (packedCells[x + nx * z] >>> 20) & 1023, c = (packedCells[qx + nx * qz] >>> 20) & 1023; maximumNeighborRatio = Math.max(maximumNeighborRatio, a / c, c / a); }
    const rowOffsetsOffset = 0, rowEntriesOffset = dofCount + 1, blockTableOffset = rowEntriesOffset, couplingByBodyOffset = blockTableOffset, couplingByDofOffset = couplingByBodyOffset + 1, couplingTableOffset = couplingByDofOffset + 1, dofSamplesBase = couplingTableOffset;
    const metadata = {
      factorLevelCount: 1, levelsOffset: 0, rowOffsetsOffset, rowEntriesOffset, couplingByBodyOffset, couplingByDofOffset, couplingTableOffset,
      couplingBodyCount: 0, couplingDistinctDofs: 0, couplingBodyIndices: [] as number[], dofSamplesBase, mlsRowCount: 0, icFactorization_ms: 0,
      lineOffsetsBase: auxWords, lineDofsBase: auxWords, lineCount: 0, blockTableOffset, blockCount: 0
    };
    if (resident) {
      const rw = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
      const make = (label: string, size: number) => this.device.createBuffer({ label, size: Math.max(4, Math.ceil(size / 4) * 4), usage: rw });
      const resources: GPUQuadtreeResidentResources = {
        faces: make("Resident quadtree faces", faceBytes), rowOffsets: make("Resident quadtree row offsets", rowOffsetBytes),
        rowEntries: make("Resident quadtree row entries", rowEntryBytes), matrixBuffer: make("Resident quadtree matrix", matrixBytes),
        factorColumns: make("Resident quadtree empty factor columns", Math.max(1, dofCount + 1) * 8), factorEntries: make("Resident quadtree empty factor entries", 8),
        factorAuxWidth: Math.min(2048, Math.max(1, Math.ceil(auxWords / 4))),
        factorAux: undefined as unknown as GPUTexture, cellProjection: undefined as unknown as GPUTexture,
        cellTopology: undefined as unknown as GPUTexture, cellPressureSamples: undefined as unknown as GPUTexture
      };
      const factorAuxHeight = Math.ceil(Math.max(1, Math.ceil(auxWords / 4)) / resources.factorAuxWidth);
      resources.factorAux = this.device.createTexture({ label: "Resident quadtree auxiliary data", size: [resources.factorAuxWidth, factorAuxHeight], format: "rgba32uint", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      resources.cellProjection = this.device.createTexture({ label: "Resident quadtree projection field", size: [nx, ny, nz], dimension: "3d", format: "rgba32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
      resources.cellTopology = this.device.createTexture({ label: "Resident quadtree cell topology", size: [nx, ny, nz], dimension: "3d", format: "rg32uint", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
      resources.cellPressureSamples = this.device.createTexture({ label: "Resident quadtree pressure samples", size: [nx, ny, nz], dimension: "3d", format: "rgba32uint", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
      const copy = this.device.createCommandEncoder({ label: "Finalize resident quadtree pack" });
      const copyIfPresent = (source: GPUBuffer, target: GPUBuffer, bytes: number) => { if (bytes > 0) copy.copyBufferToBuffer(source, 0, target, 0, bytes); };
      copyIfPresent(b.faces, resources.faces, faceBytes); copyIfPresent(b.rowOffsets, resources.rowOffsets, rowOffsetBytes);
      copyIfPresent(b.rowEntries, resources.rowEntries, rowEntryBytes); copyIfPresent(b.matrix, resources.matrixBuffer, matrixBytes);
      const auxLayout: GPUImageDataLayout = factorAuxHeight > 1 ? { offset: 0, bytesPerRow: resources.factorAuxWidth * 16, rowsPerImage: factorAuxHeight } : { offset: 0 };
      copy.copyBufferToTexture({ buffer: b.aux, ...auxLayout }, { texture: resources.factorAux }, { width: resources.factorAuxWidth, height: factorAuxHeight });
      const textureGroup = this.device.createBindGroup({ layout: this.textureLayout, entries: [
        { binding: 0, resource: { buffer: b.cellWords } }, { binding: 1, resource: resources.cellProjection.createView() },
        { binding: 2, resource: resources.cellPressureSamples.createView() }, { binding: 3, resource: resources.cellTopology.createView() },
        { binding: 4, resource: { buffer: this.params } }
      ] });
      const pass = copy.beginComputePass(); pass.setPipeline(this.unpackCellFieldsPipeline); pass.setBindGroup(0, textureGroup);
      pass.dispatchWorkgroups(Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4)); pass.end();
      this.device.queue.submit([copy.finish()]);
      return { leafCount, pressureSampleCount, dofCount, faceCount, ghostFaceCount, tallSegmentCount, maximumNeighborRatio, maximumFluidScale: 100, gpuWall_ms: performance.now() - startedAt, resident: resources, packed: {
        faces: new Uint8Array(0), rowOffsets: new Uint32Array(0), rowEntries: new Uint8Array(0), matrixWords: new Uint32Array(0), cellProjection: new Float32Array(0), cellTopology: new Uint32Array(0),
        factorColumns: new Uint8Array(0), factorEntries: new Uint8Array(0), factorAuxWords: new Uint32Array(0), cellPressureSamples: new Uint32Array(0), ...metadata
      } };
    }
    const align = (value: number) => Math.ceil(value / 8) * 8; const offsets: number[] = []; let total = 0;
    for (const bytes of [faceBytes, rowOffsetBytes, rowEntryBytes, matrixBytes, cellProjectionBytes, cellPressureBytes, cellTopologyBytes, auxBytes]) { offsets.push(total); total = align(total + bytes); }
    const readback = this.device.createBuffer({ label: "GPU quadtree packed readback", size: Math.max(8, total), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const copy = this.device.createCommandEncoder(); const copyToReadback = (source: GPUBuffer, offset: number, bytes: number) => { if (bytes > 0) copy.copyBufferToBuffer(source, 0, readback, offset, bytes); };
    copyToReadback(b.faces, offsets[0], faceBytes); copyToReadback(b.rowOffsets, offsets[1], rowOffsetBytes); copyToReadback(b.rowEntries, offsets[2], rowEntryBytes); copyToReadback(b.matrix, offsets[3], matrixBytes);
    copy.copyBufferToBuffer(b.cellWords, 0, readback, offsets[4], cellProjectionBytes); copy.copyBufferToBuffer(b.cellWords, cellProjectionBytes, readback, offsets[5], cellPressureBytes); copy.copyBufferToBuffer(b.cellWords, cellProjectionBytes + cellPressureBytes, readback, offsets[6], cellTopologyBytes); copy.copyBufferToBuffer(b.aux, 0, readback, offsets[7], auxBytes);
    this.device.queue.submit([copy.finish()]); await readback.mapAsync(GPUMapMode.READ);
    let faces: Uint8Array<ArrayBuffer>, rowOffsets: Uint32Array<ArrayBuffer>, rowEntries: Uint8Array<ArrayBuffer>, matrixWords: Uint32Array<ArrayBuffer>, cellProjection: Float32Array<ArrayBuffer>, cellPressureSamples: Uint32Array<ArrayBuffer>, cellTopology: Uint32Array<ArrayBuffer>, factorAuxWords: Uint32Array<ArrayBuffer>;
    try {
      const mapped = readback.getMappedRange(); faces = new Uint8Array(mapped, offsets[0], faceBytes).slice(); rowOffsets = new Uint32Array(mapped, offsets[1], dofCount + 1).slice(); rowEntries = new Uint8Array(mapped, offsets[2], rowEntryBytes).slice(); matrixWords = new Uint32Array(mapped, offsets[3], matrixBytes / 4).slice();
      cellProjection = new Float32Array(mapped, offsets[4], cells3 * 4).slice(); cellPressureSamples = new Uint32Array(mapped, offsets[5], cells3 * 4).slice(); cellTopology = new Uint32Array(mapped, offsets[6], cells3 * 2).slice(); factorAuxWords = new Uint32Array(mapped, offsets[7], auxWords).slice();
    } finally { readback.unmap(); readback.destroy(); }
    return { leafCount, pressureSampleCount, dofCount, faceCount, ghostFaceCount, tallSegmentCount, maximumNeighborRatio, maximumFluidScale: 100, gpuWall_ms: performance.now() - startedAt, packed: {
      faces, rowOffsets, rowEntries, matrixWords, cellProjection, cellTopology, factorColumns: new Uint8Array(Math.max(1, dofCount + 1) * 8), factorEntries: new Uint8Array(0), factorAuxWords,
      cellPressureSamples, ...metadata
    } };
  }

  private destroyBuffers() { if (!this.buffers) return; for (const buffer of Object.values(this.buffers)) buffer.destroy(); this.buffers = undefined; }
  destroy() { this.destroyBuffers(); this.params.destroy(); }
}
