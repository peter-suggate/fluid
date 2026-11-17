/** Conservative, per-step broad-phase cache. No contact data or DAT certificates are cached. */
export interface OgcCandidateCacheOptions {
  /** Metres allowed on each side of cached query and face bounds. */
  readonly skin_m: number;
  /** Faces per query; overflow uses the original grid search for that query. */
  readonly facesPerQuery: number;
}
export interface OgcCandidateCacheLayout extends OgcCandidateCacheOptions { readonly gridWords: number; readonly buckets: number; }
export function candidateCacheWords(faces: number, queries: number, options: OgcCandidateCacheOptions, buckets = 0): number {
  if (!(Number.isFinite(options.skin_m) && options.skin_m > 0)) throw new Error('OGC candidate cache needs a positive finite skin_m');
  if (!(Number.isSafeInteger(options.facesPerQuery) && options.facesPerQuery > 0)) throw new Error('OGC candidate cache needs a positive integer facesPerQuery');
  const words = 8 + buckets + 12 * faces + (16 + options.facesPerQuery) * queries;
  if (!Number.isSafeInteger(words) || words > 0xffffffff) throw new Error('OGC candidate cache exceeds u32 addressing');
  return words;
}
export const candidateCacheDeclaration = (gridWords: number, buckets: number) => /* wgsl */`
struct CandidateCache {
  grid: array<u32, ${gridWords}>,
  epoch: atomic<u32>, dirty: atomic<u32>, builds: atomic<u32>, hits: atomic<u32>,
  overflows: atomic<u32>, scans: atomic<u32>, resets: atomic<u32>, fallbacks: atomic<u32>,
  cells: array<atomic<u32>, ${buckets}>,
  data: array<u32>,
}
@group(0) @binding(7) var<storage, read_write> cache: CandidateCache;
`;
export function candidateCacheCommon(o: OgcCandidateCacheLayout): string { return candidateCacheDeclaration(o.gridWords, o.buckets) + /* wgsl */`
const CACHE_SKIN = ${o.skin_m.toExponential(9)};
const CACHE_CAP = ${o.facesPerQuery}u;
fn cacheBox(b: u32) -> OgcBox {
  return OgcBox(vec3f(bitcast<f32>(cache.data[b]), bitcast<f32>(cache.data[b+1u]), bitcast<f32>(cache.data[b+2u])),
    vec3f(bitcast<f32>(cache.data[b+3u]), bitcast<f32>(cache.data[b+4u]), bitcast<f32>(cache.data[b+5u])));
}
fn cacheStoreBox(b: u32, q: OgcBox) {
  for (var k = 0u; k < 3u; k++) { cache.data[b+k] = bitcast<u32>(q.lo[k]); cache.data[b+3u+k] = bitcast<u32>(q.hi[k]); }
}
fn cacheContains(a: OgcBox, b: OgcBox) -> bool { return all(b.lo >= a.lo) && all(b.hi <= a.hi); }
fn cacheGrow(b: OgcBox, s: f32) -> OgcBox { return OgcBox(b.lo - vec3f(s), b.hi + vec3f(s)); }
`; }

/** These passes follow prep; separate dispatches provide global visibility of invalidation. */
export function candidateCachePass(kind: 'reset' | 'begin' | 'check', gridWords: number, buckets: number): string {
  if (kind === 'reset') return candidateCacheDeclaration(gridWords, buckets) + /* wgsl */`
@compute @workgroup_size(1) fn main() {
  // Increment, never zero: old per-query epochs must not become valid in a later timestep.
  let stamp = min(atomicLoad(&cache.epoch), 0xfffffffeu) + 1u;
  atomicStore(&cache.epoch, stamp); atomicStore(&cache.dirty, stamp);
  atomicStore(&cache.builds, 0u); atomicStore(&cache.hits, 0u); atomicStore(&cache.overflows, 0u);
  atomicStore(&cache.scans, 0u); atomicStore(&cache.resets, 0u); atomicStore(&cache.fallbacks, 0u);
}
`;
  if (kind === 'begin') return /* wgsl */`
@compute @workgroup_size(1) fn main() { atomicStore(&cache.epoch, min(atomicLoad(&cache.epoch), 0xfffffffeu) + 1u); }
`;
  return /* wgsl */`
fn markCells(q: OgcBox, stamp: u32) {
  let lo = cellOf(q.lo, 1.0 / P.cellFloor); let hi = cellOf(q.hi, 1.0 / P.cellFloor);
  for (var z = lo.z; z <= hi.z; z++) { for (var y = lo.y; y <= hi.y; y++) { for (var x = lo.x; x <= hi.x; x++) {
    atomicMax(&cache.cells[cellHash(vec3i(x,y,z))], stamp);
  } } }
}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x; if (t >= P.T || atomicLoad(&cache.epoch) == 0xffffffffu) { return; }
  let q = triBox(t); cacheStoreBox(6u * t, q);
  let anchor = cacheBox(6u * (P.T + t));
  // Bound both endpoints, including shrinkage: a list built after a face shrinks must
  // still cover its later expansion. Dirty cells invalidate nearby queries only.
  if (any(abs(q.lo - anchor.lo) > vec3f(CACHE_SKIN)) || any(abs(q.hi - anchor.hi) > vec3f(CACHE_SKIN))) {
    let stamp = atomicLoad(&cache.epoch);
    markCells(cacheGrow(anchor, CACHE_SKIN), stamp); markCells(cacheGrow(q, CACHE_SKIN), stamp);
    cacheStoreBox(6u * (P.T + t), q); atomicAdd(&cache.resets, 1u);
  }
}
`;
}

export const CANDIDATE_CACHE_QUERY = /* wgsl */`
// A cursor over the original grid: the cache builder and overflow path share the exact ownership test.
struct FaceCursor {
  q: OgcBox, c: vec3i, lo: vec3i, hi: vec3i, j: u32, end: u32, done: bool,
  scanned: u32, collisions: u32, rejected: u32, base: u32, count: u32, index: u32, cached: bool,
}
fn faceGrid(q: OgcBox) -> FaceCursor {
  var s: FaceCursor; s.q = q; s.lo = cellOf(q.lo, invCell()); s.hi = cellOf(q.hi, invCell()); s.c = s.lo;
  let h = cellHash(s.c); s.j = bucketStart(h); s.end = min(bucketStart(h+1u), P.gridCap);
  return s;
}
fn faceGridNext(s: ptr<function, FaceCursor>) -> u32 {
  loop {
    if ((*s).done) { return OGC_NONE; }
    if ((*s).j < (*s).end) {
      let j = (*s).j; (*s).j++;
      (*s).scanned++;
      if (entryKey(j) != cellKey((*s).c)) { (*s).collisions++; continue; }
      let b = entryBox(j);
      if (!boxesOverlap((*s).q, b) || any(cellOf(max((*s).q.lo, b.lo), invCell()) != (*s).c)) { (*s).rejected++; continue; }
      return entryFace(j);
    }
    (*s).c.x++;
    if ((*s).c.x > (*s).hi.x) { (*s).c.x = (*s).lo.x; (*s).c.y++; }
    if ((*s).c.y > (*s).hi.y) { (*s).c.y = (*s).lo.y; (*s).c.z++; }
    if ((*s).c.z > (*s).hi.z) { (*s).done = true; continue; }
    let h = cellHash((*s).c); (*s).j = bucketStart(h); (*s).end = min(bucketStart(h+1u), P.gridCap);
  }
}
// A monotonically increasing stamp, not a checksum: any changed cell in this
// region has a newer stamp than every cell at its last build. Hash collisions can
// only cause unnecessary rebuilds. The step stamp also expires all previous-step lists.
fn regionStamp(q: OgcBox) -> u32 {
  var stamp = atomicLoad(&cache.dirty);
  let lo = cellOf(q.lo, 1.0 / P.cellFloor); let hi = cellOf(q.hi, 1.0 / P.cellFloor);
  for (var z = lo.z; z <= hi.z; z++) { for (var y = lo.y; y <= hi.y; y++) { for (var x = lo.x; x <= hi.x; x++) {
    stamp = max(stamp, atomicLoad(&cache.cells[cellHash(vec3i(x,y,z))]));
  } } }
  return stamp;
}
fn candidateFaces(q: OgcBox, id: u32) -> FaceCursor {
  // Saturate rather than wrap stamps: after exhaustion, preserve correctness via the grid.
  if (atomicLoad(&cache.epoch) == 0xffffffffu) { atomicAdd(&cache.fallbacks, 1u); return faceGrid(q); }
  let b = 12u * P.T + id * (16u + CACHE_CAP);
  if (cache.data[b] != regionStamp(cacheBox(b+8u)) || !cacheContains(cacheBox(b+2u), q)) {
    // Faces at build and replay can each differ from their epoch anchor by skin (2 skin total),
    // and q can move by skin. Four skins leave a rounding margin beyond the three-skin bound.
    let region = cacheGrow(q, 4.0 * CACHE_SKIN);
    var s = faceGrid(region);
    var n = 0u;
    loop {
      let t = faceGridNext(&s); if (t == OGC_NONE) { break; }
      if (n == CACHE_CAP) { n = OGC_NONE; break; }
      cache.data[b+16u+n] = t; n++;
    }
    atomicAdd(&cache.scans, s.scanned); atomicAdd(&cache.builds, 1u);
    if (n == OGC_NONE) { atomicAdd(&cache.overflows, 1u); }
    cache.data[b] = regionStamp(region); cache.data[b+1u] = n; cacheStoreBox(b+2u, cacheGrow(q, CACHE_SKIN)); cacheStoreBox(b+8u, region);
  } else if (cache.data[b+1u] != OGC_NONE) { atomicAdd(&cache.hits, 1u); }
  if (cache.data[b+1u] == OGC_NONE) { atomicAdd(&cache.fallbacks, 1u); return faceGrid(q); }
  var s: FaceCursor; s.q = q; s.base = b+16u; s.count = cache.data[b+1u]; s.cached = true;
  return s;
}
fn candidateFaceNext(s: ptr<function, FaceCursor>) -> u32 {
  if (!(*s).cached) { return faceGridNext(s); }
  loop {
    if ((*s).index == (*s).count) { return OGC_NONE; }
    let t = cache.data[(*s).base + (*s).index]; (*s).index++; (*s).scanned++;
    if (!boxesOverlap((*s).q, cacheBox(6u * t))) { (*s).rejected++; continue; }
    return t;
  }
}
`;
