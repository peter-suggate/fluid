/** Dawn proof of the block neighbour rows, the wall-face list, the kind-aware
 * clamps and the block-face indices (lib/methods/uniform/uniform-block-neighbours.ts,
 * uniform-lattice-address.ts) against their host twins.
 *
 * A pool of table 64 and 32 blocks is seeded with a tank of 3 x 2 x 2 pages
 * in a SOLID shell under an AIR top, one page of it AIR and one SOLID, and
 * four pages on the window's edges, in scattered blocks with free ones
 * between; most pages are pinned. Two mirrors hold the tail inside guard
 * words, one with the seed's rows written as an ownership writes them and one
 * with none. The kernels run on the seed, then after each of six commits: a
 * claim of three pages (and a request toward SOLID, dropped), the retire of
 * the seeded pages nothing keeps, the retire of the claimed ones, a claim of
 * eighteen pages that runs through the free blocks into the retired ones,
 * and two more that retire them. After each, the state and both
 * mirrors are read back: the mirrors' table and block records are the
 * state's, the rows, the count and the listed faces are the host twin's of
 * that table, and the guard words are untouched. Then a kernel evaluates,
 * over the first mirror, every NearFrom form from every stored block toward
 * every one of the 27 steps (on, one past and far past each face), their
 * FromPage forms, umBlockNeighbour of every row word, the wall-face
 * accessors, and umFaceTile, umFaceCell and umFaceLevel over every slot of
 * three blocks, against the host twins.
 *
 * An ownership of a 96 x 64 x 64 box under an open top (permuted blocks, an
 * AIR page) is then built: its topology is read back for the seed's rows
 * after its block records and the detail table's word, the rows are
 * overwritten and rebuilt by the kernels, and the same evaluation runs
 * through the topology's own accessors (uniformMixedTopologyNeighboursWGSL
 * after uniformMixedTopologyWGSL, as a module that asks for them has it).
 *
 * node --import tsx tools/probe-uniform-block-neighbours.ts
 * DUMP_WGSL=<dir>: write the probe's shader modules there and stop before
 * any device (for naga and Tint); with them, the topology module as
 * uniformSceneConstantsWGSL respells it for the solver's own modules. */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { UniformBlockNeighbours, uniformBlockNeighboursWGSL } from "../lib/methods/uniform/uniform-block-neighbours";
import { uniformDetailTableWords } from "../lib/methods/uniform/uniform-detail-fields";
import {
  UNIFORM_NEAR_SPAN, UNIFORM_PAGE_TABLE_WORDS, uniformBlockFaceWGSL, uniformBlockNeighbourLayout, uniformBlockNeighbourWGSL, uniformBlockNeighbourWords, uniformFaceCell,
  uniformFaceLevel, uniformFaceTile, uniformLatticeAddress, uniformLatticeAddressWGSL, uniformNearFrom,
} from "../lib/methods/uniform/uniform-lattice-address";
import { createUniformMixedLayout } from "../lib/methods/uniform/uniform-mixed-layout";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { uniformMixedTopologyNeighboursWGSL, uniformMixedTopologyWGSL } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import { uniformSceneConstantsWGSL } from "../lib/methods/uniform/uniform-scene-constants";
import {
  UniformPagePool, uniformPagePoolLayout, uniformPagePoolSeed, UNIFORM_PAGE_POOL_FREE as FREE, UNIFORM_PAGE_POOL_KIND as KIND, UNIFORM_PAGE_POOL_SOLID as SOLID,
  UNIFORM_PAGE_POOL_STATUS as STATUS, type UniformPagePoolSeedPage,
} from "../lib/methods/uniform/uniform-page-pool";

type Page = readonly [number, number, number];
const errors: string[] = [], started = performance.now();
const fail = (message: string) => { errors.push(message);console.error(`FAIL ${message}`); };
/** Reports the first words at which two arrays differ. */
function same(got: ArrayLike<number>, want: ArrayLike<number>, label: string): boolean {
  if (got.length !== want.length) { fail(`${label}: ${got.length} words, expected ${want.length}`);return false; }
  let bad = 0;
  for (let i = 0; i < got.length; i++) if (got[i] !== want[i] && bad++ < 4) fail(`${label}: word ${i} is ${got[i]}, expected ${want[i]}`);
  if (bad > 4) fail(`${label}: ${bad} words differ`);
  return bad === 0;
}

// The seed: the unit test's world (tests/uniform-block-neighbours.test.ts).
const BLOCKS = 32, TANK: Page[] = [[2, 2, 2], [3, 2, 2], [4, 2, 2], [2, 3, 2], [4, 3, 2], [2, 2, 3], [3, 2, 3], [4, 2, 3], [2, 3, 3], [3, 3, 3]];
const EDGE: Page[] = [[0, 10, 10], [63, 10, 10], [10, 0, 63], [63, 63, 63]], HELD = [17, 3, 29, 8, 1, 22, 12, 30, 5, 19, 26, 9, 14, 2];
/** Seeded pages no pin holds: they retire by the second commit. */
const LOOSE = new Set(["3,2,2", "2,3,3", "63,10,10"]);
const poolOptions = { table: 64, blocks: BLOCKS, idleSteps: 2 }, poolLayout = uniformPagePoolLayout(poolOptions), tail = uniformBlockNeighbourLayout(BLOCKS);
const seedPages: UniformPagePoolSeedPage[] = [...TANK, ...EDGE].map(([x, y, z], i) => ({ x, y, z, block: HELD[i]! }));
const seed = uniformPagePoolSeed(poolLayout, seedPages, (x, y, z) => {
  const box = x >= 2 && x <= 4 && y >= 2 && y <= 3 && z >= 2 && z <= 3;
  if (box) return x === 4 && y === 3 && z === 3 ? KIND.solid : KIND.air;
  const shell = x >= 1 && x <= 5 && y >= 1 && y <= 4 && z >= 1 && z <= 4;
  return shell && !(y === 4 && x >= 2 && x <= 4 && z >= 2 && z <= 3) ? KIND.solid : KIND.air;
});
/** The commits: the pages the host asks for in each. */
const COMMITS: Page[][] = [[[3, 3, 2], [2, 4, 2], [0, 0, 0], [1, 1, 1]], [], [], [[3, 2, 2], [10, 0, 62], ...Array.from({ length: 16 }, (_, i): Page => [20 + i, 20, 20])], [], []];

// The evaluation: eight script words an entry (op, block, x, y, z, then the
// home page or face and level), four result words.
const OP = { cell: 0, tile: 1, vertex: 2, wideVertex: 3, point: 4, faceTile: 5, faceCell: 6, faceLevel: 7, neighbour: 8, wall: 9 } as const;
const CAPACITY = 131072, GROUPS = CAPACITY / 64;
const evaluateWGSL = (group: number) => /* wgsl */ `
@group(${group}) @binding(0) var<storage,read> script:array<i32>;
@group(${group}) @binding(1) var<storage,read_write> result:array<u32>;
@compute @workgroup_size(64) fn evaluate(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=u32(script[0])){return;}
 let a=8u+8u*i;let op=u32(script[a]);let b=u32(script[a+1u]);
 let p=vec3i(script[a+2u],script[a+3u],script[a+4u]);let q=vec3i(script[a+5u],script[a+6u],script[a+7u]);
 var o=vec4u(0u);
 switch op {
  case ${OP.cell}u {let r=umCellNearFrom(b,p);o=vec4u(bitcast<vec3u>(r),u32(all(r==umCellNearFromPage(b,q,p))));}
  case ${OP.tile}u {let r=umTileNearFrom(b,p);o=vec4u(bitcast<vec3u>(r),u32(all(r==umTileNearFromPage(b,q,p))));}
  case ${OP.vertex}u {let r=umVertexNearFrom(b,p);o=vec4u(bitcast<vec3u>(r),u32(all(r==umVertexNearFromPage(b,q,p))));}
  case ${OP.wideVertex}u {let r=umWideVertexNearFrom(b,p);o=vec4u(bitcast<vec3u>(r),u32(all(r==umWideVertexNearFromPage(b,q,p))));}
  case ${OP.point}u {let f=bitcast<vec3f>(p);let r=umPointNearFrom(b,f);o=vec4u(bitcast<vec3u>(r),u32(all(r==umPointNearFromPage(b,q,f))));}
  case ${OP.faceTile}u {o.x=umFaceTile(b,u32(q.x),vec3u(p));}
  case ${OP.faceCell}u {o.x=umFaceCell(b,u32(q.x),vec3u(p));}
  case ${OP.faceLevel}u {o.x=umFaceLevel(b,u32(q.x),u32(q.y),vec3u(p));}
  case ${OP.neighbour}u {o.x=umBlockNeighbour(b,p);o.y=umBlockNeighbourIndex(p);}
  case ${OP.wall}u {o.x=umWallFaceCount();o.y=umWallFace(b);}
  default {}
 }
 result[4u*i]=o.x;result[4u*i+1u]=o.y;result[4u*i+2u]=o.z;result[4u*i+3u]=o.w;
}`;
/** The helpers over a bare tail at word `table` of a buffer, with accessors of the probe's own. */
const tailModuleWGSL = (table: number, blocks: number) => /* wgsl */ `
@group(0) @binding(0) var<storage,read> tailWords:array<u32>;
const UM_D=vec3u(2048u);const UM_T=UM_D/4u;
fn umPageWord(page:vec3i)->u32{let e=vec3u(page)&vec3u(63u);return tailWords[${table}u+(e.x|(e.y<<6u)|(e.z<<12u))];}
fn umBlockWord(b:u32)->u32{return tailWords[${table + UNIFORM_PAGE_TABLE_WORDS}u+b];}
fn umBlockNeighbourWord(b:u32,i:u32)->u32{return tailWords[${table + uniformBlockNeighbourLayout(blocks).rows}u+27u*b+i];}
fn umWallFaceCount()->u32{return tailWords[${table + uniformBlockNeighbourLayout(blocks).wallCount}u];}
fn umWallFace(i:u32)->u32{return tailWords[${table + uniformBlockNeighbourLayout(blocks).walls}u+i];}
${uniformLatticeAddressWGSL({ cells: "UM_D", tiles: "UM_T", constant: true, paged: { blocks: `${blocks}u` } })}
${uniformBlockNeighbourWGSL}
${uniformBlockFaceWGSL}
${evaluateWGSL(1)}`;
const MIRRORS = [640, 1031] as const;
const boxLattice = { dimensions: [96, 64, 64] as [number, number, number], cellSize_m: [1, 1, 1] as [number, number, number], origin_m: { x: 0, y: 0, z: 0 } };
const boxAddress = uniformLatticeAddress(boxLattice.dimensions, true, { blocks: 20, permute: 7, air: [[1, 1, 0]] }), boxKeys = boxAddress.tileKeys;
const topologyModuleWGSL = `${uniformMixedTopologyWGSL({ lattice: boxLattice, tiles: boxKeys }, 0)}\n${uniformMixedTopologyNeighboursWGSL()}\n${evaluateWGSL(1)}`;

// Both seeds' scripts are built once here, before any device: a script the host cannot build stops the probe now.
for (const [name, table, records] of [["pool seed", seed.table, seed.blocks], ["box seed", boxAddress.seed.table, boxAddress.seed.blocks]] as const) {
  const e = evaluation(table, records);
  console.log(JSON.stringify({ script: name, entries: e.count, taps: e.taps, walls: uniformBlockNeighbourWords(table, records)[27 * records.length] }));
}
if (process.env.DUMP_WGSL) {
  const out = process.env.DUMP_WGSL;mkdirSync(out, { recursive: true });
  writeFileSync(`${out}/000.wgsl`, `// label: probe evaluation over a pool mirror\n${tailModuleWGSL(MIRRORS[0], BLOCKS)}`);
  writeFileSync(`${out}/001.wgsl`, `// label: probe evaluation through the mixed topology\n${topologyModuleWGSL}`);
  writeFileSync(`${out}/002.wgsl`, `// label: block neighbours of a pool mirror\n${uniformBlockNeighboursWGSL(BLOCKS, MIRRORS[1])}`);
  writeFileSync(`${out}/003.wgsl`, `// label: block neighbours of a topology\n${uniformBlockNeighboursWGSL(boxAddress.blocks, 5 * boxKeys)}`);
  writeFileSync(`${out}/004.wgsl`, `// label: probe evaluation through the mixed topology, scene constants in the uniform\n${uniformSceneConstantsWGSL(topologyModuleWGSL)}`);
  console.log(`wrote 5 modules to ${out}`);process.exit(0);
}

/** The script of a table and its records, and what each entry must give. */
function evaluation(table: Uint32Array, records: Uint32Array): { script: Int32Array<ArrayBuffer>; expected: Uint32Array<ArrayBuffer>; count: number; taps: number } {
  const n = records.length, rows = uniformBlockNeighbourWords(table, records), script = new Int32Array(8 + 8 * CAPACITY), floats = new Float32Array(script.buffer);
  const expected = new Uint32Array(4 * CAPACITY), signed = new Int32Array(expected.buffer), real = new Float32Array(expected.buffer), out: [number, number, number] = [0, 0, 0];
  let count = 0, taps = 0;
  const entry = (op: number, block: number, p: readonly number[], q: readonly number[]) => {
    if (count >= CAPACITY) throw new Error("The evaluation script is full");
    script.set([op, block, p[0]!, p[1]!, p[2]!, q[0]!, q[1]!, q[2]!], 8 + 8 * count);return count++;
  };
  const names = ["cell", "tile", "vertex", "wideVertex", "point"] as const;
  for (let b = 0; b < n; b++) {
    const stored = b >= 1 && b < n - 1 && records[b] !== FREE;
    // Every row word, of the blocks with no page too.
    for (let i = 0; i < 27; i++) { const d = [i % 3 - 1, Math.floor(i / 3) % 3 - 1, Math.floor(i / 9) - 1], at = entry(OP.neighbour, b, d, [0, 0, 0]);expected[4 * at] = rows[27 * b + i]!;expected[4 * at + 1] = i; }
    if (!stored) continue;
    const r = records[b]!, page: Page = [r & 63, (r >>> 6) & 63, (r >>> 12) & 63];
    for (const name of names) {
      const s = UNIFORM_NEAR_SPAN[name], size = 1 << s.bits, fraction = name === "point" ? 0.375 : 0;
      for (let i = 0; i < 27; i++) for (const reach of [1, 3, size - 1]) for (const inside of [0, s.last, 2]) {
        const d = [i % 3 - 1, Math.floor(i / 3) % 3 - 1, Math.floor(i / 9) - 1];
        const p = d.map((v, a) => { const low = page[a]! << s.bits; return v < 0 ? low - reach - fraction : v > 0 ? low + s.last + reach + fraction : low + inside; });
        const at = entry(OP[name], b, name === "point" ? [0, 0, 0] : p, page);
        uniformNearFrom(rows, b, page, s, p, out);taps++;
        if (name === "point") { floats.set(p, 8 + 8 * at + 2);real.set(out, 4 * at); } else signed.set(out, 4 * at);
        expected[4 * at + 3] = 1;
      }
    }
  }
  // Every slot of three blocks' planes, from coordinates anywhere on the lattice.
  for (const b of [0, 5, n - 1]) for (let f = 0; f < 6; f++) {
    const axis = f >> 1, u = (axis + 1) % 3, v = (axis + 2) % 3;
    const plane = (m: number, op: number, level: number, index: (x: number, y: number, z: number) => number) => {
      for (let j = 0; j < m; j++) for (let i = 0; i < m; i++) {
        const c = [0, 0, 0];c[u] = i + m * 5;c[v] = j + m * 11;c[axis] = (i * 7 + j) % m + m * 2;
        const at = entry(op, b, c, [f, level, 0]);
        expected[4 * at] = index(c[0]!, c[1]!, c[2]!);
        if (expected[4 * at] !== (6 * b + f) * m * m + i + m * j) throw new Error("The host face index is not the formula's");
      }
    };
    plane(8, OP.faceTile, 0, (x, y, z) => uniformFaceTile(b, f, x, y, z));plane(32, OP.faceCell, 0, (x, y, z) => uniformFaceCell(b, f, x, y, z));
    for (let level = 0; level < 4; level++) plane(8 >> level, OP.faceLevel, level, (x, y, z) => uniformFaceLevel(b, f, level, x, y, z));
  }
  const walls = rows[27 * n]!;
  for (let i = 0; i < Math.max(walls, 1); i++) { const at = entry(OP.wall, i, [0, 0, 0], [0, 0, 0]);expected[4 * at] = walls;expected[4 * at + 1] = i < walls ? rows[27 * n + 1 + i]! : 0; }
  script[0] = count;
  return { script, expected, count, taps };
}

const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter();assert.ok(adapter);
const device = managedGPUDevice(await adapter.requestDevice(), { requireWorkerRealm: false });
device.addEventListener("uncapturederror", e => { e.preventDefault();fail(`device: ${e.error.message}`); });

async function read(buffer: GPUBuffer, offset = 0, size = buffer.size - offset): Promise<Uint32Array<ArrayBuffer>> {
  const target = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }), encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, offset, target, 0, size);device.queue.submit([encoder.finish()]);
  await target.mapAsync(GPUMapMode.READ);
  const words = new Uint32Array(target.getMappedRange().slice(0));target.unmap();target.destroy();
  return words;
}
async function compile(label: string, code: string): Promise<GPUShaderModule> {
  const module = device.createShaderModule({ label, code });
  const messages = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
  if (messages.length) throw new Error(`${label}\n${messages.map(m => `${m.lineNum}: ${m.message}`).join("\n")}`);
  return module;
}
const storage = (binding: number, type: GPUBufferBindingType): GPUBindGroupLayoutEntry => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
const scriptLayout = device.createBindGroupLayout({ entries: [storage(0, "read-only-storage"), storage(1, "storage")] });
const scriptBuffer = device.createBuffer({ size: 4 * (8 + 8 * CAPACITY), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
const resultBuffer = device.createBuffer({ size: 16 * CAPACITY, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
const scriptGroup = device.createBindGroup({ layout: scriptLayout, entries: [{ binding: 0, resource: { buffer: scriptBuffer } }, { binding: 1, resource: { buffer: resultBuffer } }] });
/** Runs the evaluation kernel of a pipeline whose group 0 is `held` and compares every entry. */
async function evaluate(pipeline: GPUComputePipeline, held: GPUBindGroup, table: Uint32Array, records: Uint32Array, label: string): Promise<{ entries: number; taps: number }> {
  const e = evaluation(table, records);
  device.queue.writeBuffer(scriptBuffer, 0, e.script, 0, 8 + 8 * e.count);
  device.queue.writeBuffer(resultBuffer, 0, new Uint32Array(4 * e.count).fill(0xdeadbeef));
  const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);pass.setBindGroup(0, held);pass.setBindGroup(1, scriptGroup);pass.dispatchWorkgroups(GROUPS);pass.end();
  device.queue.submit([encoder.finish()]);
  const got = await read(resultBuffer, 0, 16 * e.count);
  let bad = 0;
  for (let i = 0; i < e.count; i++) for (let k = 0; k < 4; k++) if (got[4 * i + k] !== e.expected[4 * i + k] && bad++ < 6) {
    const a = 8 + 8 * i, op = e.script[a]!;
    fail(`${label}: entry ${i} (op ${op}, block ${e.script[a + 1]}, p ${op === OP.point ? [...new Float32Array(e.script.buffer, 4 * (a + 2), 3)] : [...e.script.subarray(a + 2, a + 5)]}, q ${[...e.script.subarray(a + 5, a + 8)]}) word ${k}: ${got[4 * i + k]! | 0}, expected ${e.expected[4 * i + k]! | 0}`);
  }
  if (bad > 6) fail(`${label}: ${bad} result words differ`);
  return { entries: e.count, taps: e.taps };
}
/** A tail's rows, count and listed faces against the host twin of its table. */
function audit(words: Uint32Array, table: Uint32Array, records: Uint32Array, label: string): number {
  const n = records.length, l = uniformBlockNeighbourLayout(n), want = uniformBlockNeighbourWords(table, records), walls = want[27 * n]!;
  same(words.subarray(l.rows, l.wallCount), want.subarray(0, 27 * n), `${label}: neighbour rows`);
  if (words[l.wallCount] !== walls) fail(`${label}: ${words[l.wallCount]} wall faces, expected ${walls}`);
  same(words.subarray(l.walls, l.walls + walls), want.subarray(27 * n + 1, 27 * n + 1 + walls), `${label}: wall faces`);
  return walls;
}

// The pool, two mirrors and the kernels.
{
  const pool = new UniformPagePool(device, poolOptions);
  pool.pin([...TANK, ...EDGE].filter(p => !LOOSE.has(p.join(","))).map(([x, y, z]) => ({ x, y, z })));pool.seed(seed);
  const guard = (i: number) => (0xa5000000 | i) >>> 0, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const mirrors = MIRRORS.map(offset => {
    const words = offset + tail.end + 9, buffer = device.createBuffer({ size: 4 * words, usage });
    device.queue.writeBuffer(buffer, 0, Uint32Array.from({ length: words }, (_, i) => guard(i)));return { buffer, offset };
  });
  await pool.initialize(mirrors);
  // The first mirror takes the seed's rows as an ownership writes them; the second holds guard words there until the kernels run.
  device.queue.writeBuffer(mirrors[0]!.buffer, 4 * (MIRRORS[0] + tail.rows), uniformBlockNeighbourWords(seed.table, seed.blocks));
  assert.throws(() => new UniformBlockNeighbours(device, BLOCKS, [{ buffer: mirrors[0]!.buffer, offset: MIRRORS[0] + 10 }]), /holds/);
  assert.throws(() => new UniformBlockNeighbours(device, BLOCKS, [mirrors[0]!, mirrors[0]!]), /listed already/);
  const neighbours = new UniformBlockNeighbours(device, BLOCKS, mirrors);
  assert.throws(() => neighbours.encode(device.createCommandEncoder()), /not initialized/);
  await neighbours.initialize();
  await assert.rejects(neighbours.initialize(), /already initialized/);
  const heldLayout = device.createBindGroupLayout({ entries: [storage(0, "read-only-storage")] });
  const pipeline = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [heldLayout, scriptLayout] }),
    compute: { module: await compile("Block neighbour probe over a pool mirror", tailModuleWGSL(MIRRORS[0], BLOCKS)), entryPoint: "evaluate" } });
  const held = device.createBindGroup({ layout: heldLayout, entries: [{ binding: 0, resource: { buffer: mirrors[0]!.buffer } }] });
  let before = { table: new Uint32Array(seed.table), records: new Uint32Array(seed.blocks) }, claims = 0, retires = 0, reused = 0;
  const retired = new Set<number>();
  for (let step = 0; step <= COMMITS.length; step++) {
    const label = step ? `commit ${step}` : "seed", encoder = device.createCommandEncoder();
    if (step) { for (const [x, y, z] of COMMITS[step - 1]!) pool.request(x, y, z);pool.encodeCommit(encoder); }
    neighbours.encode(encoder);device.queue.submit([encoder.finish()]);
    const state = await read(pool.state), s = poolLayout.state;
    const table = state.slice(s.table, s.table + poolLayout.entries), records = state.slice(s.blocks, s.blocks + BLOCKS);
    if (state[STATUS.fatal]) fail(`${label}: the pool is fatal (${state[STATUS.fatal]})`);
    if (!step) { same(table, seed.table, "seed: the pool's table");same(records, seed.blocks, "seed: the pool's block records"); }
    let walls = 0, claimed = 0, gone = 0;
    for (let b = 1; b < BLOCKS - 1; b++) {
      if (before.records[b] === FREE && records[b] !== FREE) { claimed++;if (retired.has(b)) reused++; }
      if (before.records[b] !== FREE && records[b] === FREE) { gone++;retired.add(b); }
    }
    claims += claimed;retires += gone;
    for (const [i, m] of mirrors.entries()) {
      const words = await read(m.buffer), name = `${label}, mirror ${i}`;
      for (let k = 0; k < m.offset; k++) if (words[k] !== guard(k)) { fail(`${name}: guard word ${k} before the tail changed`);break; }
      for (let k = m.offset + tail.end; k < words.length; k++) if (words[k] !== guard(k)) { fail(`${name}: guard word ${k} after the wall faces changed`);break; }
      const own = words.subarray(m.offset, m.offset + tail.end);
      same(own.subarray(0, poolLayout.entries), table, `${name}: page table`);same(own.subarray(poolLayout.entries, tail.rows), records, `${name}: block records`);
      walls = audit(own, table, records, name);
    }
    const run = await evaluate(pipeline, held, table, records, label);
    console.log(JSON.stringify({ step: label, generation: state[STATUS.generation], resident: state[STATUS.resident], claimed, retired: gone, walls, evaluated: run.entries, taps: run.taps, errors: errors.length }));
    before = { table, records };
  }
  if (!claims) fail("no commit claimed a page");
  if (!retires) fail("no commit retired a page");
  // The row words the SOLID request left alone: (1,1,1) is still SOLID.
  if (before.table[1 | (1 << 6) | (1 << 12)] !== SOLID) fail("the SOLID page (1,1,1) was claimed");
  console.log(JSON.stringify({ pool: "done", claims, retires, reused }));
  pool.destroy();for (const m of mirrors) m.buffer.destroy();
}

// An ownership's topology: the seed's rows at construction, the kernels over it, the helpers through the topology module.
{
  const layout = createUniformMixedLayout(boxLattice, [], 4, undefined, boxAddress), ownership = new UniformMixedOwnership(device, layout, false, 0);
  const mirror = ownership.pageMirror, l = uniformBlockNeighbourLayout(boxAddress.blocks), table = boxAddress.seed.table, records = boxAddress.seed.blocks;
  if (mirror.offset !== 5 * boxKeys) fail(`ownership: the page table is at word ${mirror.offset}, expected ${5 * boxKeys}`);
  const size = 4 * (5 * boxKeys + l.end + uniformDetailTableWords(boxLattice.dimensions));
  if (mirror.buffer.size !== size) fail(`ownership: the topology is ${mirror.buffer.size} bytes, expected ${size}`);
  const built = await read(mirror.buffer, 4 * mirror.offset);
  same(built.subarray(0, l.rows), boxAddress.seed.tail, "ownership: page table and block records");
  const walls = audit(built, table, records, "ownership as built");
  if (built[built.length - 1] !== mirror.offset + l.end) fail(`ownership: the detail table's word is ${built[built.length - 1]}, expected ${mirror.offset + l.end}`);
  // Overwrite the rows and the list; the kernels rebuild them, and nothing after them changes.
  device.queue.writeBuffer(mirror.buffer, 4 * (mirror.offset + l.rows), new Uint32Array(l.words).fill(0xdeadbeef));
  const neighbours = new UniformBlockNeighbours(device, boxAddress.blocks, [mirror]);
  await neighbours.initialize();
  const encoder = device.createCommandEncoder();neighbours.encode(encoder);device.queue.submit([encoder.finish()]);
  const rebuilt = await read(mirror.buffer, 4 * mirror.offset);
  same(rebuilt.subarray(0, l.rows), boxAddress.seed.tail, "ownership rebuilt: page table and block records");
  audit(rebuilt, table, records, "ownership rebuilt");
  same(rebuilt.subarray(l.end), built.subarray(l.end), "ownership rebuilt: the words after the wall faces");
  const tiles = await read(mirror.buffer, 0, 4 * mirror.offset);
  const pipeline = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [ownership.bindLayout, scriptLayout] }),
    compute: { module: await compile("Block neighbour probe through the mixed topology", topologyModuleWGSL), entryPoint: "evaluate" } });
  const run = await evaluate(pipeline, ownership.bindGroup, table, records, "topology module");
  same(await read(mirror.buffer, 0, 4 * mirror.offset), tiles, "ownership: the tile words");
  console.log(JSON.stringify({ ownership: boxLattice.dimensions.join("x"), blocks: boxAddress.blocks, topologyBytes: mirror.buffer.size, addedBytes: 4 * l.words, walls, evaluated: run.entries, taps: run.taps, errors: errors.length }));
}
await device.queue.onSubmittedWorkDone();
if (errors.length) { console.error(`uniform block neighbours: ${errors.length} failure(s)`);process.exit(1); }
device.destroy();
console.log(`uniform block neighbours: ok (${Math.round(performance.now() - started)} ms)`);
