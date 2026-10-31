/** Sparse persistent cut-distance pages. Brush history is a document/undo log,
 * never the field sampled by production rendering. GPU updates only new stamps;
 * undo rebuilds the changed pages from their local records. */
export const SCULPT_PAGE_EDGE = 8;
export const SCULPT_PAGE_SAMPLES = 9 ** 3;
export const SCULPT_PAGE_CAPACITY = 2048;
export const SCULPT_TABLE_CAPACITY = 8192;
export const SCULPT_BAKED_TAG = 0x5343554c;
// Leave the existing record/owner ABI in place. Records become page proxies.
export const SCULPT_META = 4 + 4096 * 8 + 2048;
export const SCULPT_TABLE = SCULPT_META + 8;
export const SCULPT_VALUES = SCULPT_TABLE + SCULPT_TABLE_CAPACITY * 4;
export const SCULPT_ARENA_WORDS = SCULPT_VALUES + SCULPT_PAGE_CAPACITY * SCULPT_PAGE_SAMPLES;
type V3 = readonly [number, number, number];
interface SculptCutIndices extends Iterable<number> { readonly size: number }
const NO_CUTS: SculptCutIndices = new Set<number>();
/** Appending a stamp must not copy every old cut index for each touched page.
 * History is flattened only for undo/rebuild or a previously unqueried CPU
 * sample. Iteration is iterative, so long strokes cannot exhaust the stack. */
class AppendedCutIndices implements SculptCutIndices {
  readonly size: number;
  constructor(readonly base: SculptCutIndices, readonly added: readonly number[]) { this.size = base.size + added.length; }
  *[Symbol.iterator](): Iterator<number> {
    const chunks: (readonly number[])[] = [];
    let base: SculptCutIndices = this;
    while (base instanceof AppendedCutIndices) { chunks.push(base.added); base = base.base; }
    yield* base;
    for (let i = chunks.length - 1; i >= 0; i--) yield* chunks[i]!;
  }
}
export interface SculptPage { readonly coordinate: V3; readonly slot: number; readonly cuts: SculptCutIndices }
export interface SculptPageJob { readonly page: SculptPage; readonly reset: boolean; readonly cuts: readonly number[] }
export interface SculptBakePlan {
  readonly words: Uint32Array;
  readonly cell: V3;
  readonly origin: V3;
  readonly pages: ReadonlyMap<string, SculptPage>;
  readonly jobs: readonly SculptPageJob[];
  readonly proxies: Uint32Array;
  readonly directory: Uint32Array;
  readonly free: readonly number[];
  readonly slots: number;
}
const key = (p: V3) => p.join(",");
export const sculptPageHash = (x: number, y: number, z: number): number =>
  (Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791)) >>> 0;
export function sculptCell(words: Uint32Array): V3 {
  const f = new Float32Array(words.buffer, words.byteOffset, words.length);
  return [0, 1, 2].map(a => Number.isFinite(f[a + 1]) && f[a + 1]! > 0 ? f[a + 1]! : 1) as unknown as V3;
}
function cutPages(words: Uint32Array, index: number, cell: V3, origin: V3, visit: (coordinate: V3) => void): void {
  const f = new Float32Array(words.buffer, words.byteOffset, words.length), at = 4 + index * 8;
  // Two cells of exterior distance keep trilinear samples and cut normals
  // valid across page boundaries, including anisotropic scene lattices.
  const lo = cell.map((h, a) => Math.floor(((f[at + a]! - origin[a]!) / h - .5 - 2) / 8));
  const hi = cell.map((h, a) => Math.floor(((f[at + 4 + a]! - origin[a]!) / h - .5 + 2) / 8));
  if (lo.some((v, a) => !Number.isSafeInteger(v) || !Number.isSafeInteger(hi[a]) || Math.abs(v) > 0x7fffffff || Math.abs(hi[a]!) > 0x7fffffff)
    || hi.reduce((n, v, a) => n * (v - lo[a]! + 1), 1) > SCULPT_PAGE_CAPACITY) {
    throw new RangeError(`Sculpt substrate capacity exceeded (${SCULPT_PAGE_CAPACITY} pages)`);
  }
  for (let z = lo[2]!; z <= hi[2]!; z++) for (let y = lo[1]!; y <= hi[1]!; y++) for (let x = lo[0]!; x <= hi[0]!; x++) visit([x, y, z]);
}

/** Metadata-only plan. No distances, voxel coverage, normals or meshes are
 * evaluated on the host. Planning is atomic; capacity refusal changes nothing. */
export function planSculptBake(previous: SculptBakePlan | undefined, words: Uint32Array, gpuTables = true): SculptBakePlan {
  const cell = words[0] ? sculptCell(words) : previous?.cell ?? [1, 1, 1] as V3;
  const f = new Float32Array(words.buffer, words.byteOffset, words.length), end = 4 + (words[0] ?? 0) * 8;
  const origin: V3 = words.length >= end + 4 ? [f[end]!, f[end + 1]!, f[end + 2]!] : previous?.origin ?? [0, 0, 0];
  const sameGrid = previous?.cell.every((h, a) => h === cell[a] && previous.origin[a] === origin[a]);
  const old = sameGrid ? previous : undefined;
  const before = old?.words ?? new Uint32Array(4);
  let prefix = 0;
  record: for (; prefix < Math.min(before[0]!, words[0]!); prefix++) {
    for (let j = 0; j < 8; j++) if (before[4 + prefix * 8 + j] !== words[4 + prefix * 8 + j]) break record;
  }
  const pages = new Map(old?.pages);
  let free = [...old?.free ?? []];
  let slots = old?.slots ?? 0;
  const touched = new Map<string, { coordinate: V3; cuts: SculptCutIndices; removed: number[]; added: number[]; reset: boolean }>();
  const touch = (p: V3) => {
    const k = key(p); let entry = touched.get(k);
    if (!entry) { entry = { coordinate: p, cuts: pages.get(k)?.cuts ?? NO_CUTS, removed: [], added: [], reset: !pages.has(k) }; touched.set(k, entry); }
    return entry;
  };
  for (let i = prefix; i < before[0]!; i++) cutPages(before, i, cell, origin, p => { const entry = touch(p); entry.removed.push(i); entry.reset = true; });
  for (let i = prefix; i < words[0]!; i++) cutPages(words, i, cell, origin, p => { const entry = touch(p); entry.added.push(i); });
  for (const entry of touched.values()) {
    if (entry.removed.length) {
      const retained = new Set(entry.cuts);
      for (const i of entry.removed) retained.delete(i);
      for (const i of entry.added) retained.add(i);
      entry.cuts = retained;
    } else if (entry.added.length) entry.cuts = new AppendedCutIndices(entry.cuts, entry.added);
  }
  // Free empty pages before allocating replacements, so undo/branching reuses
  // storage and an erased page can never fall back to an obsolete cut field.
  for (const [k, entry] of touched) if (!entry.cuts.size) {
    const page = pages.get(k); if (page) { free.push(page.slot); pages.delete(k); }
  }
  const vacant = new Set(free);
  while (slots > 0 && vacant.delete(slots - 1)) slots--;
  free = free.filter(slot => slot < slots);
  const jobs: SculptPageJob[] = [];
  for (const [k, entry] of touched) if (entry.cuts.size) {
    const slot = pages.get(k)?.slot ?? free.pop() ?? slots++;
    if (slot >= SCULPT_PAGE_CAPACITY) throw new RangeError(`Sculpt substrate capacity exceeded (${SCULPT_PAGE_CAPACITY} pages)`);
    const page = { coordinate: entry.coordinate, slot, cuts: entry.cuts };
    pages.set(k, page); jobs.push({ page, reset: entry.reset, cuts: entry.reset ? [...entry.cuts] : entry.added });
  }
  const proxies = new Uint32Array(gpuTables ? 4 + slots * 8 : 4), pf = new Float32Array(proxies.buffer);
  proxies[0] = pages.size ? slots : 0; proxies[1] = SCULPT_BAKED_TAG;
  const directory = new Uint32Array(gpuTables ? SCULPT_TABLE_CAPACITY * 4 : 0);
  if (gpuTables) for (const page of pages.values()) {
    const at = 4 + page.slot * 8;
    for (let a = 0; a < 3; a++) { pf[at + a] = origin[a]! + (page.coordinate[a]! * 8 + .5) * cell[a]!; pf[at + 4 + a] = origin[a]! + ((page.coordinate[a]! + 1) * 8 + .5) * cell[a]!; }
    proxies[at + 3] = 1;
    let bucket = sculptPageHash(...page.coordinate) & (SCULPT_TABLE_CAPACITY - 1);
    while (directory[bucket * 4 + 3]) bucket = (bucket + 1) & (SCULPT_TABLE_CAPACITY - 1);
    directory.set([...page.coordinate, page.slot + 1], bucket * 4);
  }
  return { words, cell, origin, pages, jobs, proxies, directory, free, slots };
}

export function sculptSubstrateWGSL(load: string): string {
  return /* wgsl */ `
fn svoSculptBaked()->bool{return ${load}(1u)==${SCULPT_BAKED_TAG}u;}
fn sculptFieldCell()->vec3f{return bitcast<vec3f>(vec3u(${load}(${SCULPT_META}u),${load}(${SCULPT_META + 1}u),${load}(${SCULPT_META + 2}u)));}
fn sculptFieldOrigin()->vec3f{return bitcast<vec3f>(vec3u(${load}(${SCULPT_META + 4}u),${load}(${SCULPT_META + 5}u),${load}(${SCULPT_META + 6}u)));}
fn sculptFieldBand()->f32{return bitcast<f32>(${load}(${SCULPT_META + 3}u));}
fn sculptFieldSlot(p:vec3i)->u32{
  var bucket=((bitcast<u32>(p.x)*73856093u)^(bitcast<u32>(p.y)*19349663u)^(bitcast<u32>(p.z)*83492791u))&${SCULPT_TABLE_CAPACITY - 1}u;
  for(var probe=0u;probe<${SCULPT_TABLE_CAPACITY}u;probe++){
    let at=${SCULPT_TABLE}u+bucket*4u;let slot=${load}(at+3u);
    if(slot==0u){return 0xffffffffu;}
    if(all(bitcast<vec3i>(vec3u(${load}(at),${load}(at+1u),${load}(at+2u)))==p)){return slot-1u;}
    bucket=(bucket+1u)&${SCULPT_TABLE_CAPACITY - 1}u;
  }return 0xffffffffu;
}
fn sculptFieldValue(slot:u32,p:vec3u)->f32{return bitcast<f32>(${load}(${SCULPT_VALUES}u+slot*${SCULPT_PAGE_SAMPLES}u+p.x+9u*(p.y+9u*p.z)));}
fn sculptFieldTrilinear(world:vec3f)->SvoSculptCutSample{
  let h=sculptFieldCell();let grid=(world-sculptFieldOrigin())/h-vec3f(0.5);let page=vec3i(floor(grid/8.0));let slot=sculptFieldSlot(page);
  if(slot==0xffffffffu){return SvoSculptCutSample(sculptFieldBand(),vec3f(0));}
  let local=clamp(grid-vec3f(page)*8.0,vec3f(0),vec3f(8));let p=vec3u(min(floor(local),vec3f(7)));let t=local-vec3f(p);
  let a=sculptFieldValue(slot,p);let b=sculptFieldValue(slot,p+vec3u(1,0,0));
  let c=sculptFieldValue(slot,p+vec3u(0,1,0));let d=sculptFieldValue(slot,p+vec3u(1,1,0));
  let e=sculptFieldValue(slot,p+vec3u(0,0,1));let f=sculptFieldValue(slot,p+vec3u(1,0,1));
  let g=sculptFieldValue(slot,p+vec3u(0,1,1));let j=sculptFieldValue(slot,p+vec3u(1,1,1));
  let value=mix(mix(mix(a,b,t.x),mix(c,d,t.x),t.y),mix(mix(e,f,t.x),mix(g,j,t.x),t.y),t.z);
  var normal=vec3f(mix(mix(b-a,d-c,t.y),mix(f-e,j-g,t.y),t.z),
    mix(mix(c-a,d-b,t.x),mix(g-e,j-f,t.x),t.z),mix(mix(e-a,f-b,t.x),mix(g-c,j-d,t.x),t.y))/h;
  let magnitude=length(normal);if(magnitude>1e-12){normal/=magnitude;}
  return SvoSculptCutSample(value,normal);
}
fn sculptFieldSample(world:vec3f)->SvoSculptCutSample{
  let h=sculptFieldCell()*0.5;var normal=vec3f(0);
  for(var a=0u;a<3u;a++){var offset=vec3f(0);offset[a]=h[a];
    normal[a]=(sculptFieldTrilinear(world+offset).distance-sculptFieldTrilinear(world-offset).distance)/(2.0*h[a]);}
  let magnitude=length(normal);if(magnitude>1e-12){normal/=magnitude;}
  return SvoSculptCutSample(sculptFieldTrilinear(world).distance,normal);
}
`;
}

/** Synchronous tools need a small CPU picking mirror. It stores only lattice
 * samples actually queried by the pointer, shares untouched pages and updates
 * cached samples with the new operands. It never rebuilds a whole voxel image
 * or a per-object cut BVH. Rendering/meshing uses the GPU field above. */
export class SculptPickingField {
  readonly caches: ReadonlyMap<string, Map<number, number>>;
  constructor(readonly plan: SculptBakePlan, previous?: SculptPickingField) {
    const caches = new Map<string, Map<number, number>>();
    for (const [key] of plan.pages) {
      const held = previous?.caches.get(key); if (held) caches.set(key, held);
    }
    for (const job of plan.jobs) {
      const k = key(job.page.coordinate), values = new Map<number, number>();
      for (const [index, before] of caches.get(k) ?? []) {
        const point = this.point(job.page.coordinate, index);
        values.set(index, this.evaluate(point, job.cuts, job.reset ? this.band : before));
      }
      caches.set(k, values);
    }
    this.caches = caches;
  }
  get band(): number { return Math.min(...this.plan.cell) * 1.5; }
  private point(page: V3, index: number): V3 {
    return [this.plan.origin[0] + (page[0] * 8 + index % 9 + .5) * this.plan.cell[0],
      this.plan.origin[1] + (page[1] * 8 + Math.floor(index / 9) % 9 + .5) * this.plan.cell[1],
      this.plan.origin[2] + (page[2] * 8 + Math.floor(index / 81) + .5) * this.plan.cell[2]];
  }
  private evaluate(point: V3, indices: Iterable<number>, initial: number): number {
    const f = new Float32Array(this.plan.words.buffer, this.plan.words.byteOffset, this.plan.words.length);
    let distance = initial;
    for (const index of indices) {
      const at = 4 + index * 8;
      const rx = (f[at + 4]! - f[at]!) / 2, ry = (f[at + 5]! - f[at + 1]!) / 2, rz = (f[at + 6]! - f[at + 2]!) / 2;
      const x = point[0] - (f[at]! + f[at + 4]!) / 2, y = point[1] - (f[at + 1]! + f[at + 5]!) / 2, z = point[2] - (f[at + 2]! + f[at + 6]!) / 2;
      if (this.plan.words[at + 3] === 2) distance = Math.min(distance, (Math.hypot(x / rx, y / ry, z / rz) - 1) * Math.min(rx, ry, rz));
      else {
        const a = Math.abs(x) - rx, b = Math.abs(y) - ry, c = Math.abs(z) - rz;
        distance = Math.min(distance, Math.hypot(Math.max(a, 0), Math.max(b, 0), Math.max(c, 0)) + Math.min(Math.max(a, b, c), 0));
      }
    }
    return Math.fround(distance);
  }
  sample(point: V3): { distance: number; normal: V3 } {
    const distance = this.distance(point);
    const normal = this.plan.cell.map((h, a) => {
      const lo = [...point], hi = [...point]; lo[a]! -= h * .5; hi[a]! += h * .5;
      return (this.distance(hi as unknown as V3) - this.distance(lo as unknown as V3)) / h;
    });
    const length = Math.hypot(...normal);
    return { distance, normal: (length > 1e-12 ? normal.map(v => v / length) : [0, 0, 0]) as unknown as V3 };
  }
  distance(point: V3): number {
    const grid = point.map((v, a) => (v - this.plan.origin[a]!) / this.plan.cell[a]! - .5);
    const coordinate = grid.map(v => Math.floor(v / 8)) as unknown as V3;
    const k = key(coordinate), page = this.plan.pages.get(k);
    if (!page) return this.band;
    const local = grid.map((v, a) => Math.max(0, Math.min(8, v - coordinate[a]! * 8)));
    const p = local.map(v => Math.min(7, Math.floor(v))), t = local.map((v, a) => v - p[a]!);
    const cache = this.caches.get(k)!;
    const values = Array.from({ length: 8 }, (_, i) => {
      const index = p[0]! + (i & 1) + 9 * (p[1]! + ((i >> 1) & 1) + 9 * (p[2]! + (i >> 2)));
      let value = cache.get(index);
      if (value === undefined) { value = this.evaluate(this.point(coordinate, index), page.cuts, this.band); cache.set(index, value); }
      return value;
    });
    const [a, b, c, d, e, f, g, j] = values as [number, number, number, number, number, number, number, number];
    const mix = (a: number, b: number, t: number) => a + (b - a) * t;
    const [x, y, z] = t as [number, number, number];
    const distance = mix(mix(mix(a, b, x), mix(c, d, x), y), mix(mix(e, f, x), mix(g, j, x), y), z);
    return distance;
  }
}
