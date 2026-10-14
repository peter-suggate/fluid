/** Dawn proof of the residency certificate keyed by pool block
 * (lib/methods/uniform/uniform-mixed-residency.ts and the page readers of
 * uniform-mixed-topology.wgsl.ts) against their host twins, around the real
 * page pool's commit.
 *
 * The module is built as the census's are: the mixed topology's text over a
 * buffer whose tail the pool mirrors, the pool's requesters, and the shared
 * request (rsAsk) and mark (rsResident) text. Its entry points are the
 * census's own in miniature: a seed fold over umPageTile and umTileLive, a
 * request strided over the pages, and a mark on the 2D grid of
 * uniformKeyGroups; the pool's commit runs between request and mark in one
 * encoder, as the frame encodes it.
 *
 * Two scenes. A closed box of 6x4x4 pages in permuted blocks with free ones,
 * AIR holes and idleSteps 2: seeds wander for a dozen censuses, so pages are
 * claimed, kept, retired and claimed again in other blocks. A box of 2x2x2
 * pages under an open top: a seed near the top plane claims the AIR pages
 * above it, a seed in one of those claims the next layer, and a radius over
 * the pool's cap raises closureCap and asks for nothing. After every census
 * the device's seed words, the pages it claimed, its flags and every page
 * reader are compared with the twins over the table the device holds.
 *
 * node --import tsx tools/probe-residency-certificate.ts
 * DUMP_WGSL=<dir>: write the modules there and stop before any device (for
 * naga and Tint). */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { uniformKeyGroups, uniformLatticeAddress, uniformTileCoord, type UniformLatticeAddress } from "../lib/methods/uniform/uniform-lattice-address";
import { UNIFORM_RESIDENCY_LIQUID as LIQUID, UNIFORM_RESIDENCY_SEED as SEED, UNIFORM_RESIDENCY_SEEDED as SEEDED, uniformResidencyBlockHeld, uniformResidencyMark, uniformResidencyMarkWGSL, uniformResidencyPageCoord,
  uniformResidencyPageOfTile, uniformResidencyPageTile, uniformResidencyRequestWGSL, uniformResidencyRequests, uniformResidencySeedBoxWGSL, uniformResidencySeedWord } from "../lib/methods/uniform/uniform-mixed-residency";
import { uniformMixedPageCount, uniformMixedResidencyWord, uniformMixedSupportWords, uniformMixedTopologyWGSL } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import { UniformPagePool, uniformPagePoolFatalMessage, uniformPagePoolLayout, uniformPagePoolWGSL, UNIFORM_PAGE_POOL_FATAL as FATAL, UNIFORM_PAGE_POOL_FREE as FREE, UNIFORM_PAGE_POOL_SOLID as SOLID, UNIFORM_PAGE_POOL_STATUS as STATUS,
  type UniformPagePoolLayout } from "../lib/methods/uniform/uniform-page-pool";

type Tile = readonly [number, number, number];
const errors: string[] = [], started = performance.now();
const fail = (message: string) => { if (errors.push(message) <= 12) console.error(`FAIL ${message}`); };
const entry = (x: number, y: number, z: number) => x | (y << 6) | (z << 12);
const REQUEST_GROUPS = 3, DISPATCH_X = 5;

interface Scene { readonly name: string; readonly cells: readonly [number, number, number]; readonly open: boolean; readonly pool: { blocks: number; permute?: number; air?: [number, number, number][] }; readonly idleSteps: number }
const SCENES: readonly Scene[] = [
  { name: "closed box", cells: [192, 128, 128], open: false, pool: { blocks: 6 * 4 * 4 + 2 + 9, permute: 7, air: [[1, 0, 0], [2, 1, 1], [3, 1, 1], [3, 2, 1], [5, 3, 3], [0, 3, 0]] }, idleSteps: 2 },
  { name: "open top", cells: [64, 64, 64], open: true, pool: { blocks: 8 + 2 + 10, permute: 4 }, idleSteps: 2 },
];
function source(a: UniformLatticeAddress, layout: UniformPagePoolLayout): string {
  const lattice = { dimensions: [...a.cells] as [number, number, number], cellSize_m: [1, 1, 1] as [number, number, number], origin_m: { x: 0, y: 0, z: 0 } };
  return uniformMixedTopologyWGSL({ lattice, tiles: a.tileKeys }, 0) + /* wgsl */ `
@group(1) @binding(0) var<storage,read_write> seeds:array<u32>;
${uniformPagePoolWGSL(layout, 1, 1, 2)}
@group(1) @binding(3) var<storage,read> tileFlags:array<u32>;
@group(1) @binding(4) var<storage,read_write> result:array<u32>;
${uniformResidencySeedBoxWGSL}
${uniformResidencyRequestWGSL}
${uniformResidencyMarkWGSL(page => `seeds[${page}]`)}
// The census's seed fold: per page, the box of its seed tiles; 0 for a page of a block that holds none.
@compute @workgroup_size(64) fn seedPages(@builtin(global_invocation_id) gid:vec3u){
 let page=gid.x;if(page>=UM_PAGES){return;}
 var lo=vec3u(3u);var hi=vec3u(0u);var found=false;var liquid=false;
 for(var lane=0u;lane<64u;lane++){
  let t=umPageTile(page,lane);if(!umTileLive(t)){continue;}
  let f=tileFlags[t];if((f&${SEED.seed}u)==0u){continue;}
  let c=umCorner(lane,4u);lo=min(lo,c);hi=max(hi,c);found=true;liquid=liquid||(f&${SEED.liquid}u)!=0u;
 }
 seeds[page]=select(0u,lo.x|(lo.y<<3u)|(lo.z<<6u)|(hi.x<<9u)|(hi.y<<12u)|(hi.z<<15u)|${SEEDED}u|select(0u,${LIQUID}u,liquid),found);
}
// The census's request: a fixed grid strides the pages.
@compute @workgroup_size(64) fn request(@builtin(global_invocation_id) gid:vec3u){
 let r=umSupport[UM_RESIDENCY+2u];
 for(var page=gid.x;page<UM_PAGES;page+=${64 * REQUEST_GROUPS}u){
  let word=seeds[page];if(word==0u){continue;}
  if(!rsAsk(page,word,r)){return;}
 }
}
// The census's mark, after the commit.
@compute @workgroup_size(64) fn mark(@builtin(global_invocation_id) gid:vec3u){
 let page=gid.x+umDispatchX*64u*gid.y;if(page>=UM_PAGES){return;}
 umSupport[UM_RESIDENCY+4u+page]=select(0u,1u,rsResident(page,umSupport[UM_RESIDENCY+2u]));
}
// The page readers: a key's page; a page's coordinate, its resident flag through umTileResident of its first tile, and its tile at four lanes.
@compute @workgroup_size(64) fn readers(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x+umDispatchX*64u*gid.y;
 if(i<UM_TILES){result[i]=umPageOfTile(i);}
 if(i<UM_PAGES){
  let a=UM_TILES+8u*i;let q=umPageCoord(i);
  result[a]=q.x;result[a+1u]=q.y;result[a+2u]=q.z;result[a+3u]=u32(umTileResident(umPageTile(i,0u)))|(u32(umPageResident(i))<<1u);
  result[a+4u]=umPageTile(i,0u);result[a+5u]=umPageTile(i,27u);result[a+6u]=umPageTile(i,i%64u);result[a+7u]=umPageTile(i,63u);
 }
}`;
}
const address = (s: Scene) => uniformLatticeAddress(s.cells, s.open, s.pool);

if (process.env.DUMP_WGSL) {
  const out = process.env.DUMP_WGSL;mkdirSync(out, { recursive: true });
  SCENES.forEach((s, i) => { const a = address(s);writeFileSync(`${out}/00${i}.wgsl`, `// label: residency probe, ${s.name}\n${source(a, uniformPagePoolLayout({ ...a.pool, idleSteps: s.idleSteps }))}`); });
  console.log(`wrote ${SCENES.length} modules to ${out}`);process.exit(0);
}

const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter();assert.ok(adapter);
const device = managedGPUDevice(await adapter.requestDevice(), { requireWorkerRealm: false });
device.addEventListener("uncapturederror", e => { e.preventDefault();fail(`device: ${e.error.message}`); });
async function read(buffer: GPUBuffer, words = buffer.size / 4, offset = 0): Promise<Uint32Array<ArrayBuffer>> {
  const target = device.createBuffer({ size: 4 * words, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }), encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 4 * offset, target, 0, 4 * words);device.queue.submit([encoder.finish()]);
  await target.mapAsync(GPUMapMode.READ);
  const out = new Uint32Array(target.getMappedRange().slice(0));target.unmap();target.destroy();return out;
}
const storage = (label: string, words: number) => device.createBuffer({ label, size: 4 * words, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

async function run(s: Scene): Promise<void> {
  const a = address(s), n = a.tileKeys, pages = uniformMixedPageCount(n), R = uniformMixedResidencyWord(n);
  const pool = new UniformPagePool(device, { ...a.pool, idleSteps: s.idleSteps });
  const l = pool.layout;
  // An ownership topology in miniature: five words a key (unread here), then the tail the pool mirrors.
  const topology = storage("topology", 5 * n + l.tail.words), support = storage("support", uniformMixedSupportWords(n)), counts = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const seeds = storage("seeds", pages), tileFlags = storage("tile flags", n), result = storage("result", n + 8 * pages);
  pool.seed(a.seed);
  await pool.initialize([{ buffer: topology, offset: 5 * n }]);
  const module = device.createShaderModule({ label: `residency probe, ${s.name}`, code: source(a, l) });
  for (const m of (await module.getCompilationInfo()).messages) if (m.type === "error") fail(`${s.name}: ${m.lineNum}: ${m.message}`);
  if (errors.length) return;
  const ownership = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }, { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }] });
  const own = device.createBindGroupLayout({ entries: [0, 1, 2, 3, 4].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: binding === 1 || binding === 3 ? "read-only-storage" as const : "storage" as const } })) });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [ownership, own] });
  const groups = [device.createBindGroup({ layout: ownership, entries: [topology, counts, support].map((buffer, binding) => ({ binding, resource: { buffer } })) }),
    device.createBindGroup({ layout: own, entries: [seeds, pool.state, pool.work, tileFlags, result].map((buffer, binding) => ({ binding, resource: { buffer } })) })];
  const [seedPages, request, mark, readers] = await Promise.all(["seedPages", "request", "mark", "readers"].map(entryPoint => device.createComputePipelineAsync({ layout, compute: { module, entryPoint, constants: { umDispatchX: DISPATCH_X } } })));
  const pageGrid = uniformKeyGroups(pages, 64, DISPATCH_X, [0, 0]), keyGrid = uniformKeyGroups(n, 64, DISPATCH_X, [0, 0]);
  const tail = async () => { const words = await read(topology, l.tail.words, 5 * n);return { table: words.subarray(l.tail.table, l.tail.table + l.entries), blocks: words.subarray(l.tail.blocks, l.tail.blocks + l.blocks) }; };
  let held = await tail(), census = 0;
  assert.deepEqual([...held.blocks], [...a.seed.blocks]);
  const totals = { censuses: 0, claimed: 0, retired: 0, resident: [] as number[], blocksReused: 0 };
  const everHeld = new Set<number>();held.blocks.forEach((w, b) => { if (w !== FREE) everHeld.add(b); });

  /** One census on the device and on the host. Returns the fatal bits. */
  async function step(tiles: readonly Tile[], radius: number, expectFatal = 0): Promise<{ claimed: number[]; coords: Set<string>; fatal: number }> {
    const at = `${s.name}, census ${++census}`, before = held, live = { ...a, seed: { ...a.seed, table: before.table, blocks: before.blocks } } as UniformLatticeAddress;
    // Seed tiles by window coordinate become flags by key of the table as it stands; every third is the liquid's.
    const flags = new Uint32Array(n), c: [number, number, number] = [0, 0, 0], wanted = new Map(tiles.map((t, i) => [t.join(","), SEED.seed | (i % 3 === 0 ? SEED.liquid : 0)]));
    for (let key = 0; key < n; key++) { if (!uniformResidencyBlockHeld(before, key >>> 9)) continue;const f = wanted.get(uniformTileCoord(live, key, c).join(","));if (f) flags[key] = f; }
    device.queue.writeBuffer(tileFlags, 0, flags);device.queue.writeBuffer(support, 4 * (R + 2), new Uint32Array([radius]));
    const encoder = device.createCommandEncoder();
    {
      const pass = encoder.beginComputePass();pass.setBindGroup(0, groups[0]!);pass.setBindGroup(1, groups[1]!);
      pass.setPipeline(seedPages);pass.dispatchWorkgroups(Math.ceil(pages / 64));pass.setPipeline(request);pass.dispatchWorkgroups(REQUEST_GROUPS);pass.end();
    }
    pool.encodeCommit(encoder);
    {
      const pass = encoder.beginComputePass();pass.setBindGroup(0, groups[0]!);pass.setBindGroup(1, groups[1]!);
      pass.setPipeline(mark);pass.dispatchWorkgroups(pageGrid[0], pageGrid[1]);pass.setPipeline(readers);pass.dispatchWorkgroups(keyGrid[0], keyGrid[1]);pass.end();
    }
    device.queue.submit([encoder.finish()]);
    const [gotSeeds, gotSupport, got, status] = await Promise.all([read(seeds), read(support, 4 + 2 * pages, R), read(result), read(pool.state, STATUS.words)]);
    held = await tail();
    // Seed words: the fold over the table the seed launch read.
    const twinSeeds = new Uint32Array(pages);
    for (let page = 0; page < pages; page++) if (uniformResidencyBlockHeld(before, page >>> 3)) twinSeeds[page] = uniformResidencySeedWord(lane => flags[uniformResidencyPageTile(page, lane)]!);
    for (let page = 0; page < pages; page++) if (gotSeeds[page] !== twinSeeds[page]) fail(`${at}: seed word of page ${page} is ${gotSeeds[page]}, expected ${twinSeeds[page]}`);
    // The request, through the commit: every AIR page asked for holds a block now, every resident page asked for kept its block, nothing else was claimed.
    const asked = new Set<number>(), accepted = uniformResidencyRequests(before, twinSeeds, radius, l.closureCap, (x, y, z) => asked.add(entry(x, y, z)));
    const fatal = status[STATUS.fatal]!;
    if (fatal !== expectFatal) fail(`${at}: pool fatal ${fatal} (${uniformPagePoolFatalMessage(l, status)}), expected ${expectFatal}`);
    if (accepted !== (expectFatal === 0)) fail(`${at}: the host request ${accepted ? "asked" : "refused"}, the device ${expectFatal ? "must refuse" : "must ask"}`);
    const claimed: number[] = [], retired: number[] = [];
    for (let e = 0; e < l.entries; e++) {
      const was = before.table[e]!, is = held.table[e]!, stored = (w: number) => w >= 1 && w < l.blocks - 1;
      if (was === SOLID || is === SOLID) { if (was !== is) fail(`${at}: SOLID entry ${e} changed`);continue; }
      if (!stored(was) && stored(is)) { claimed.push(e);if (!asked.has(e)) fail(`${at}: entry ${e} was claimed unasked`); }
      if (stored(was) && !stored(is)) { retired.push(e);if (asked.has(e)) fail(`${at}: entry ${e} was retired though asked for`); }
      if (stored(was) && stored(is) && was !== is) fail(`${at}: entry ${e} changed block ${was} to ${is}`);
      if (asked.has(e) && !stored(is)) fail(`${at}: entry ${e} was asked for and holds no block`);
      if (stored(is) && held.blocks[is] !== e) fail(`${at}: block ${is} of entry ${e} names ${held.blocks[is]}`);
    }
    for (const e of claimed) { const block = held.table[e]!;if (everHeld.has(block)) totals.blocksReused++;everHeld.add(block); }
    // The mark: the twin over the table the commit left, with the seed words the device made.
    const twinFlags = new Uint32Array(pages), list = new Uint32Array(pages), resident = uniformResidencyMark(held, gotSeeds, radius, twinFlags, list), q: number[] = [], coords = new Set<string>();
    for (let page = 0; page < pages; page++) {
      if (gotSupport[4 + page] !== twinFlags[page]) fail(`${at}: flag of page ${page} (block ${page >>> 3}) is ${gotSupport[4 + page]}, expected ${twinFlags[page]}`);
      if (twinFlags[page]) coords.add(uniformResidencyPageCoord(held, page, q).join(","));
    }
    if (gotSupport[2] !== radius) fail(`${at}: radius word ${gotSupport[2]}`);
    // The readers.
    for (let key = 0; key < n; key++) if (got[key] !== uniformResidencyPageOfTile(key)) fail(`${at}: umPageOfTile(${key}) is ${got[key]}`);
    for (let page = 0; page < pages; page++) {
      const at8 = n + 8 * page, block = page >>> 3;
      for (const [slot, lane] of [[4, 0], [5, 27], [6, page % 64], [7, 63]] as const) if (got[at8 + slot] !== uniformResidencyPageTile(page, lane)) fail(`${at}: umPageTile(${page}, ${lane}) is ${got[at8 + slot]}`);
      if (got[at8 + 3] !== 3 * twinFlags[page]!) fail(`${at}: umPageResident / umTileResident of page ${page} read ${got[at8 + 3]}`);
      if (!uniformResidencyBlockHeld(held, block)) continue;
      uniformResidencyPageCoord(held, page, q);
      if (got[at8] !== q[0] || got[at8 + 1] !== q[1] || got[at8 + 2] !== q[2]) fail(`${at}: umPageCoord(${page}) is ${[...got.subarray(at8, at8 + 3)]}, expected ${q}`);
    }
    totals.censuses++;totals.claimed += claimed.length;totals.retired += retired.length;totals.resident.push(resident);
    return { claimed, coords, fatal };
  }

  if (!s.open) {
    // Two clusters wander through the box: pages ahead are claimed, pages behind retire after two idle commits and their blocks come back.
    const path = (i: number): Tile[] => { const x = 2 + 3 * i, z = 30 - 2 * i;return [[x, 9, 5], [x + 1, 10, 5], [x, 9, 6], [40 - 2 * i, 20, z], [41 - 2 * i, 21, z]]; };
    for (let i = 0; i < 12; i++) await step(path(i), i % 4 === 3 ? 9 : 6);
    if (!(totals.claimed > 0 && totals.retired > 0 && totals.blocksReused > 0)) fail(`${s.name}: the path must claim, retire and reuse blocks: ${JSON.stringify(totals)}`);
  } else {
    // A seed seven tiles under the top plane reaches nothing above it.
    const low = await step([[5, 9, 5]], 6);
    if (low.claimed.length) fail(`${s.name}: a seed out of reach of the top claimed ${low.claimed}`);
    // Two tiles under it: the four AIR pages above the box, and their 16-cell pages within the radius.
    const top = await step([[5, 14, 5]], 6);
    assert.deepEqual(top.claimed, [entry(0, 2, 0), entry(1, 2, 0), entry(0, 2, 1), entry(1, 2, 1)], "claims above the open top");
    const above = [...top.coords].filter(key => +key.split(",")[1]! >= 4).sort();
    assert.deepEqual(above, [0, 1, 2].flatMap(z => [4, 5].flatMap(y => [0, 1, 2].map(x => `${x},${y},${z}`))).sort(), "resident pages above the open top");
    // A seed in a claimed block is a seed like any other: the next layer up.
    const up = await step([[5, 21, 5]], 6);
    assert.deepEqual(up.claimed, [entry(0, 3, 0), entry(1, 3, 0), entry(0, 3, 1), entry(1, 3, 1)], "claims a layer further up");
    assert.ok(up.coords.has("1,6,1"));
    // A radius over the pool's cap: closureCap with the radius as its detail, and nothing asked.
    const capped = await step([[5, 21, 5]], 17, FATAL.closureCap);
    const status = await read(pool.state, STATUS.words);
    assert.deepEqual([capped.claimed.length, status[STATUS.fatalFirst], status[STATUS.fatalDetail]], [0, FATAL.closureCap, 17]);
  }
  console.log(JSON.stringify({ scene: s.name, blocks: a.blocks, pages, censuses: totals.censuses, claimed: totals.claimed, retired: totals.retired, blocksReused: totals.blocksReused, resident: totals.resident, failures: errors.length }));
  pool.destroy();for (const b of [topology, support, counts, seeds, tileFlags, result]) b.destroy();
}
for (const s of SCENES) await run(s);
console.log(`residency certificate: ${errors.length} failure(s), ${((performance.now() - started) / 1000).toFixed(1)} s`);
process.exit(errors.length ? 1 : 0);
