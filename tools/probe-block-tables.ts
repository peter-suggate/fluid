/** Dawn proof of the per-block tile-bit and bound-cube tables
 * (lib/methods/uniform/uniform-block-tables.ts) against their host twins.
 *
 * A pool of table 64 and 32 blocks holds a tank of 3 x 2 x 2 pages at the
 * window's origin under an AIR top, one page of it AIR and one SOLID, and
 * four pages on the window's edges, in scattered blocks with free ones
 * between. Every tile key carries two flags and six keys. Two modules are
 * built as the solver's are: the census's (a tile-bit table without AIR and
 * the bound cubes) and the frame plan's (a tile-bit table with AIR). Their
 * build entries run and the tables are read back against the twins'; then
 * 4096 boxes (single tiles to boxes longer than the window) are asked of
 * nameAny from a stored home block and of rangeKeys, against the twins. A
 * page is then retired and two are claimed, in a freed block and a free one,
 * the builds run again over the tables as they stand, and the same boxes
 * are asked again.
 *
 * node --import tsx tools/probe-block-tables.ts
 * DUMP_WGSL=<dir>: write the two modules there and stop before any device
 * (for naga and Tint). */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import {
  UNIFORM_BOUND_CUBE_WORDS as CUBE, UNIFORM_TILE_BITS_WORDS as BITS, uniformBoundCubeRange, uniformBoundCubesBuild, uniformBoundCubesWGSL, uniformTileBitsAny,
  uniformTileBitsBuild, uniformTileBitsWGSL,
} from "../lib/methods/uniform/uniform-block-tables";
import { UNIFORM_PAGE_TABLE_WORDS as TABLE, uniformBlockNeighbourWords, uniformLatticeAddressWGSL } from "../lib/methods/uniform/uniform-lattice-address";
import { UNIFORM_PAGE_POOL_FREE as FREE, UNIFORM_PAGE_POOL_KIND as KIND, uniformPagePoolLayout, uniformPagePoolSeed } from "../lib/methods/uniform/uniform-page-pool";

type Page = readonly [number, number, number];
const errors: string[] = [], started = performance.now();
const fail = (message: string) => { if (errors.push(message) <= 12) console.error(`FAIL ${message}`); };
const BLOCKS = 32, KEYS = 512 * BLOCKS, QUERIES = 4096, DISPATCH_X = 100, BUDGET = 64;
const TANK: Page[] = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [2, 1, 0], [0, 0, 1], [1, 0, 1], [2, 0, 1], [0, 1, 1], [1, 1, 1]];
const EDGE: Page[] = [[0, 10, 10], [63, 10, 10], [10, 0, 63], [63, 63, 63]], HELD = [17, 3, 29, 8, 1, 22, 12, 30, 5, 19, 26, 9, 14, 2];
const entry = (x: number, y: number, z: number) => x | (y << 6) | (z << 12);
const seed = uniformPagePoolSeed(uniformPagePoolLayout({ table: 64, blocks: BLOCKS, idleSteps: 2 }), [...TANK, ...EDGE].map(([x, y, z], i) => ({ x, y, z, block: HELD[i]! })), (x, y, z) => {
  if (x <= 2 && y <= 1 && z <= 1) return x === 2 && y === 1 && z === 1 ? KIND.solid : KIND.air;
  return (y >= 2 && x <= 2 && z <= 1) || x > 8 || y > 8 || z > 8 ? KIND.air : KIND.solid;
});
const table = new Uint32Array(seed.table), records = new Uint32Array(seed.blocks);
assert.equal(table.length, TABLE);assert.equal(records.length, BLOCKS);
const hash = (x: number, y: number, z: number, salt: number) => (Math.imul(x + 1, 73856093) ^ Math.imul(y + 1, 19349663) ^ Math.imul(z + 1, 83492791) ^ Math.imul(salt + 1, 2654435761)) >>> 0;
/** Flags and keys by tile key: values of the key itself, so a block's page changes nothing of them. */
const flags = new Uint32Array(KEYS), keys = new Uint32Array(6 * KEYS);
for (let t = 0; t < KEYS; t++) {
  flags[t] = (hash(t, 1, 2, 3) % 19 === 0 ? 1 : 0) | (hash(t, 4, 5, 6) % 5 === 0 ? 2 : 0);
  for (let k = 0; k < 6; k++) keys[6 * t + k] = k < 3 ? hash(t, k, 7, 8) >>> 3 : (hash(t, k - 3, 7, 8) >>> 3) + (hash(t, k, 9, 10) >>> 8);
}
const key = (t: number, k: number) => keys[6 * t + k]!;
function random(s: number): () => number { return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 2 ** 32; }; }
/** The boxes, eight words each after the count: a slot that picks the home block, the home block of the round, low and high. */
const script = new Int32Array(8 + 8 * QUERIES);script[0] = QUERIES;
{
  const r = random(77), centres: Page[] = [[12, 8, 8], [0, 0, 0], [23, 15, 15], [8, 8, 8], [4, 84, 84], [508, 84, 84], [84, 3, 508], [510, 510, 510], [16, 20, 4]];
  for (let i = 0; i < QUERIES; i++) {
    const c = centres[Math.floor(r() * centres.length)]!, kind = Math.floor(r() * 6), span = [0, 2, 6, 14, 40, 300][kind]!;
    const low = c.map(v => v + Math.floor((r() - 0.5) * 12) - Math.floor(r() * span)), high = low.map(v => v + Math.floor(r() * (span + 1)) + (r() < 0.03 ? -2 : 0));
    script.set([Math.floor(r() * 1024), 0, ...low, ...high], 8 + 8 * i);
  }
}

const tailWGSL = /* wgsl */ `
@group(0) @binding(0) var<storage,read> tailWords:array<u32>;
@group(0) @binding(1) var<storage,read> flags:array<u32>;
@group(0) @binding(2) var<storage,read> tileKeys:array<u32>;
@group(0) @binding(3) var<storage,read_write> tables:array<u32>;
@group(0) @binding(4) var<storage,read> script:array<i32>;
@group(0) @binding(5) var<storage,read_write> result:array<u32>;
override umDispatchX:u32=${DISPATCH_X}u;
const UM_D=vec3u(2048u);const UM_T=UM_D/4u;const UM_TILES:u32=${KEYS}u;
fn umPageWord(page:vec3i)->u32{let e=vec3u(page)&vec3u(63u);return tailWords[e.x|(e.y<<6u)|(e.z<<12u)];}
fn umBlockWord(b:u32)->u32{return tailWords[${TABLE}u+b];}
fn umBlockNeighbourWord(b:u32,i:u32)->u32{return tailWords[${TABLE + BLOCKS}u+27u*b+i];}
${uniformLatticeAddressWGSL({ cells: "UM_D", tiles: "UM_T", constant: true, paged: { blocks: `${BLOCKS}u` } })}
struct Ask{home:u32,low:vec3i,high:vec3i}
fn ask(i:u32)->Ask{let a=8u+8u*i;return Ask(u32(script[a+1u]),vec3i(script[a+2u],script[a+3u],script[a+4u]),vec3i(script[a+5u],script[a+6u],script[a+7u]));}`;
const INTERFACE = 0, COARSE = BITS * BLOCKS, CUBES = 2 * BITS * BLOCKS, TABLES = CUBES + CUBE * BLOCKS;
const censusWGSL = `${tailWGSL}
${uniformTileBitsWGSL({ name: "interface", word: (b, i) => `tables[${INTERFACE}u+${BITS}u*${b}+${i}]`, store: (b, i, v) => `tables[${INTERFACE}u+${BITS}u*${b}+${i}]=${v};`, flag: t => `(flags[${t}]&1u)!=0u`, air: false })}
${uniformBoundCubesWGSL({ tile: (t, k) => `tileKeys[6u*${t}+${k}]`, word: i => `tables[${CUBES}u+${i}]`, store: (i, v) => `tables[${CUBES}u+${i}]=${v};`, budget: BUDGET })}
@compute @workgroup_size(64) fn evaluate(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=u32(script[0])){return;}
 let q=ask(i);result[8u*i]=u32(interfaceAny(q.home,q.low,q.high));
 let k=rangeKeys(q.low,q.high);for(var j=0u;j<6u;j++){result[8u*i+1u+j]=k[j];}
}`;
const planWGSL = `${tailWGSL}
${uniformTileBitsWGSL({ name: "coarse", word: (b, i) => `tables[${COARSE}u+${BITS}u*${b}+${i}]`, store: (b, i, v) => `tables[${COARSE}u+${BITS}u*${b}+${i}]=${v};`, flag: t => `(flags[${t}]&2u)!=0u`, air: true })}
@compute @workgroup_size(64) fn evaluate(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=u32(script[0])){return;}
 let q=ask(i);result[8u*i+7u]=u32(coarseAny(q.home,q.low,q.high));
}`;

if (process.env.DUMP_WGSL) {
  const out = process.env.DUMP_WGSL;mkdirSync(out, { recursive: true });
  writeFileSync(`${out}/000.wgsl`, `// label: probe census tables\n${censusWGSL}`);writeFileSync(`${out}/001.wgsl`, `// label: probe frame plan table\n${planWGSL}`);
  console.log(`wrote 2 modules to ${out}`);process.exit(0);
}

const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter();assert.ok(adapter);
const device = managedGPUDevice(await adapter.requestDevice(), { requireWorkerRealm: false });
device.addEventListener("uncapturederror", e => { e.preventDefault();fail(`device: ${e.error.message}`); });
async function read(buffer: GPUBuffer): Promise<Uint32Array<ArrayBuffer>> {
  const target = device.createBuffer({ size: buffer.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }), encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, target, 0, buffer.size);device.queue.submit([encoder.finish()]);
  await target.mapAsync(GPUMapMode.READ);
  const words = new Uint32Array(target.getMappedRange().slice(0));target.unmap();target.destroy();return words;
}
const storage = (label: string, words: number, data?: ArrayBufferView<ArrayBuffer>) => {
  const buffer = device.createBuffer({ label, size: 4 * words, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  if (data) device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};
const tail = storage("tail", TABLE + BLOCKS + 27 * BLOCKS), scriptBuffer = storage("script", script.length), tables = storage("tables", TABLES), result = storage("result", 8 * QUERIES);
const layout = device.createBindGroupLayout({ entries: [0, 1, 2, 3, 4, 5].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: binding === 3 || binding === 5 ? "storage" as const : "read-only-storage" as const } })) });
const group = device.createBindGroup({ layout, entries: [tail, storage("flags", KEYS, flags), storage("keys", 6 * KEYS, keys), tables, scriptBuffer, result].map((buffer, binding) => ({ binding, resource: { buffer } })) });
const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] }), pipelines = new Map<string, GPUComputePipeline>();
for (const [name, code, entries] of [["census", censusWGSL, ["interfaceBits", "interfaceBlocks", "boundCubes", "evaluate"]], ["plan", planWGSL, ["coarseBits", "coarseBlocks", "evaluate"]]] as const) {
  const module = device.createShaderModule({ label: name, code });
  for (const m of (await module.getCompilationInfo()).messages) if (m.type === "error") fail(`${name}: ${m.lineNum}: ${m.message}`);
  for (const entryPoint of entries) pipelines.set(`${name}.${entryPoint}`, await device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint } }));
}
if (errors.length) process.exit(1);
/** One pass a launch, in the solver's order: the bits, then what reads them. */
function run(launches: readonly (readonly [string, number, number])[]): void {
  const encoder = device.createCommandEncoder();
  for (const [name, x, y] of launches) { const pass = encoder.beginComputePass();pass.setPipeline(pipelines.get(name)!);pass.setBindGroup(0, group);pass.dispatchWorkgroups(x, y);pass.end(); }
  device.queue.submit([encoder.finish()]);
}
const slabs = Math.ceil(KEYS / 64), BUILD = [["census.interfaceBits", DISPATCH_X, Math.ceil(slabs / DISPATCH_X)], ["plan.coarseBits", DISPATCH_X, Math.ceil(slabs / DISPATCH_X)],
  ["census.interfaceBlocks", Math.ceil(BLOCKS / 64), 1], ["plan.coarseBlocks", Math.ceil(BLOCKS / 64), 1], ["census.boundCubes", BLOCKS, 1]] as const;
const ASK = [["census.evaluate", QUERIES / 64, 1], ["plan.evaluate", QUERIES / 64, 1]] as const;

async function round(label: string): Promise<void> {
  const rows = uniformBlockNeighbourWords(table, records), stored = Array.from({ length: BLOCKS }, (_, b) => b).filter(b => b >= 1 && b < BLOCKS - 1 && records[b] !== FREE);
  device.queue.writeBuffer(tail, 0, table);device.queue.writeBuffer(tail, 4 * TABLE, records);device.queue.writeBuffer(tail, 4 * (TABLE + BLOCKS), rows, 0, 27 * BLOCKS);
  for (let i = 0; i < QUERIES; i++) script[8 + 8 * i + 1] = stored[script[8 + 8 * i]! % stored.length]!;
  // The slot each box drew is kept in word 0, its home block of this round in word 1.
  device.queue.writeBuffer(scriptBuffer, 0, script);
  run(BUILD);run(ASK);
  const got = await read(tables), answers = await read(result);
  const twins = [uniformTileBitsBuild(records, rows, t => (flags[t]! & 1) !== 0, false), uniformTileBitsBuild(records, rows, t => (flags[t]! & 2) !== 0, true), uniformBoundCubesBuild(records, key)] as const;
  let words = 0;
  for (const b of stored) for (const [twin, at, size, name] of [[twins[0], INTERFACE, BITS, "interface bits"], [twins[1], COARSE, BITS, "4h bits"], [twins[2], CUBES, CUBE, "bound cubes"]] as const)
    for (let i = 0; i < size; i++, words++) if (got[at + size * b + i] !== twin[size * b + i]) fail(`${label}: ${name} of block ${b} word ${i} is ${got[at + size * b + i]}, expected ${twin[size * b + i]}`);
  const counts = { interface: 0, coarse: 0, levels: [0, 0, 0, 0] }, out: number[] = [];
  for (let i = 0; i < QUERIES; i++) {
    const a = 8 + 8 * i, home = script[a + 1]!, low = [...script.subarray(a + 2, a + 5)], high = [...script.subarray(a + 5, a + 8)];
    // The kernels read the tables the device built: the twins read the same words.
    const any = uniformTileBitsAny(got.subarray(INTERFACE, COARSE), table, records, rows, home, low, high, false), coarse = uniformTileBitsAny(got.subarray(COARSE, CUBES), table, records, rows, home, low, high, true);
    const { level } = uniformBoundCubeRange(got.subarray(CUBES), key, table, records, low, high, BUDGET, out);
    if (answers[8 * i] !== Number(any)) fail(`${label}: interfaceAny of box ${i} (${low} to ${high}, home ${home}) is ${answers[8 * i]}, expected ${Number(any)}`);
    if (answers[8 * i + 7] !== Number(coarse)) fail(`${label}: coarseAny of box ${i} (${low} to ${high}, home ${home}) is ${answers[8 * i + 7]}, expected ${Number(coarse)}`);
    for (let k = 0; k < 6; k++) if (answers[8 * i + 1 + k] !== out[k]! >>> 0) fail(`${label}: rangeKeys of box ${i} (${low} to ${high}) key ${k} is ${answers[8 * i + 1 + k]}, expected ${out[k]! >>> 0} at level ${level}`);
    counts.interface += Number(any);counts.coarse += Number(coarse);counts.levels[level]!++;
  }
  console.log(JSON.stringify({ round: label, storedBlocks: stored.length, tableWords: words, boxes: QUERIES, interfaceSet: counts.interface, coarseSet: counts.coarse, rangeLevels: counts.levels, failures: errors.length }));
}
await round("seed");
// Retire the tank page of block 29, claim a page beside the tank in it and one above the tank in free block 6.
{
  const page = (b: number): Page => [records[b]! & 63, (records[b]! >>> 6) & 63, (records[b]! >>> 12) & 63], [x, y, z] = page(29);
  table[entry(x, y, z)] = 0;records[29] = entry(1, 1, 0);table[entry(1, 1, 0)] = 29;records[6] = entry(1, 2, 0);table[entry(1, 2, 0)] = 6;
}
await round("after a retire and two claims");
console.log(`block tables: ${errors.length} failure(s), ${((performance.now() - started) / 1000).toFixed(1)} s`);
process.exit(errors.length ? 1 : 0);
