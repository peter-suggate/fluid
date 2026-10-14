/** Dawn proof of the narrow band's kind-aware clamps
 * (lib/methods/uniform/uniform-narrow-band-near.wgsl.ts) against their host
 * twins and against the box clamp they replace.
 *
 * Four worlds of the unit test's tank (3 x 2 x 2 pages at (2,2,2) in
 * scattered blocks of a pool of 32): closed in a SOLID shell with every page
 * stored, the same with AIR pages inside it, one with a SOLID page inside it,
 * and one under an AIR top. Each is written as a bare page table, block
 * records and neighbour rows. One kernel evaluates, from homes in every
 * stored page, nbCellNear and nbTileNear toward taps on, one past, a few past
 * and a page past every face, edge and corner (and the stencil form, home's
 * block from nbHomeBlock handed to the FromPage clamp), and nbPointNearIn
 * for trajectory ends around the page at insets 0.01 and 0. Every answer is
 * the host twin's; in the two worlds whose tank is closed and holds no SOLID
 * page it is also the box clamp's, the positions to two units in the last
 * place.
 *
 * node --import tsx tools/probe-narrow-band-near.ts
 * DUMP_WGSL=<dir>: write the probe's shader module there and stop before any
 * device (for naga and Tint). */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { UNIFORM_NEAR_SPAN, UNIFORM_PAGE_TABLE_WORDS, uniformBlockNeighbourWGSL, uniformBlockNeighbourWords, uniformLatticeAddressWGSL } from "../lib/methods/uniform/uniform-lattice-address";
import { narrowBandNear, narrowBandNearWGSL, narrowBandPointNearIn } from "../lib/methods/uniform/uniform-narrow-band-near.wgsl";
import { UNIFORM_PAGE_POOL_KIND as KIND, uniformPagePoolLayout, uniformPagePoolSeed, type UniformPagePoolSeedPage } from "../lib/methods/uniform/uniform-page-pool";

type V = [number, number, number];
const errors: string[] = [], started = performance.now();
const fail = (message: string) => { errors.push(message);if (errors.length <= 24) console.error(`FAIL ${message}`); };

const BLOCKS = 32, HELD = [17, 3, 29, 8, 1, 22, 12, 30, 5, 19, 26, 9], LOW: V = [2, 2, 2], PAGES: V = [3, 2, 2];
const ALL: V[] = [];
for (let z = 2; z <= 3; z++) for (let y = 2; y <= 3; y++) for (let x = 2; x <= 4; x++) ALL.push([x, y, z]);
const inTank = (x: number, y: number, z: number) => x >= 2 && x <= 4 && y >= 2 && y <= 3 && z >= 2 && z <= 3;
/** A world's table, records and rows, and the stored pages a home is drawn from. */
function world(o: { stored: V[]; open?: boolean; solid?: V }) {
  const pages: UniformPagePoolSeedPage[] = o.stored.map(([x, y, z], i) => ({ x, y, z, block: HELD[i]! }));
  const seed = uniformPagePoolSeed(uniformPagePoolLayout({ blocks: BLOCKS, idleSteps: 2 }), pages, (x, y, z) => {
    if (inTank(x, y, z)) return o.solid && x === o.solid[0] && y === o.solid[1] && z === o.solid[2] ? KIND.solid : KIND.air;
    return o.open && y >= 4 && x >= 2 && x <= 4 && z >= 2 && z <= 3 ? KIND.air : KIND.solid;
  });
  const table = new Uint32Array(seed.table), records = new Uint32Array(seed.blocks);
  return { table, records, rows: uniformBlockNeighbourWords(table, records), stored: o.stored };
}
const PIER: V = [4, 3, 3], isPier = (p: V) => p.every((v, a) => v === PIER[a]);
const WORLDS = [
  { name: "closed", box: true, ...world({ stored: ALL }) },
  { name: "closed with AIR pages", box: true, ...world({ stored: ALL.filter((_, i) => i % 3 !== 1) }) },
  { name: "SOLID page inside", box: false, ...world({ stored: ALL.filter(p => !isPier(p)), solid: PIER }) },
  { name: "open top", box: false, ...world({ stored: ALL, open: true }) },
];

// Eight script words an entry (op, home, then the tap or the end's bits, the
// inset's bits), four result words (the answer, and whether the stencil form agrees).
const OP = { cell: 0, tile: 1, point: 2 } as const, CAPACITY = 131072, GROUPS = CAPACITY / 64, ROWS = UNIFORM_PAGE_TABLE_WORDS + BLOCKS;
const moduleWGSL = /* wgsl */ `
@group(0) @binding(0) var<storage,read> tailWords:array<u32>;
@group(0) @binding(1) var<storage,read> script:array<i32>;
@group(0) @binding(2) var<storage,read_write> result:array<u32>;
const UM_D=vec3u(2048u);const UM_T=UM_D/4u;
fn umPageWord(page:vec3i)->u32{let e=vec3u(page)&vec3u(63u);return tailWords[e.x|(e.y<<6u)|(e.z<<12u)];}
fn umBlockWord(b:u32)->u32{return tailWords[${UNIFORM_PAGE_TABLE_WORDS}u+b];}
fn umBlockNeighbourWord(b:u32,i:u32)->u32{return tailWords[${ROWS}u+27u*b+i];}
${uniformLatticeAddressWGSL({ cells: "UM_D", tiles: "UM_T", constant: true, paged: { blocks: `${BLOCKS}u` } })}
${uniformBlockNeighbourWGSL}
${narrowBandNearWGSL}
@compute @workgroup_size(64) fn evaluate(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=u32(script[0])){return;}
 let a=8u+8u*i;let op=u32(script[a]);let home=vec3i(script[a+1u],script[a+2u],script[a+3u]);
 let p=vec3i(script[a+4u],script[a+5u],script[a+6u]);
 var o=vec4u(0u);
 switch op {
  case ${OP.cell}u {let r=nbCellNear(home,p);let page=home>>vec3u(5u);o=vec4u(bitcast<vec3u>(r),u32(all(r==umCellNearFromPage(nbHomeBlock(page,p>>vec3u(5u),p>>vec3u(5u)),page,p))));}
  case ${OP.tile}u {let r=nbTileNear(home,p);let page=home>>vec3u(3u);o=vec4u(bitcast<vec3u>(r),u32(all(r==umTileNearFromPage(nbHomeBlock(page,p>>vec3u(3u),p>>vec3u(3u)),page,p))));}
  case ${OP.point}u {let f=bitcast<vec3f>(p);let inset=bitcast<f32>(script[a+7u]);let r=nbPointNearIn(home,f,inset);o=vec4u(bitcast<vec3u>(r),u32(inset!=0.0||all(r==nbPointNear(home,f))));}
  default {}
 }
 result[4u*i]=o.x;result[4u*i+1u]=o.y;result[4u*i+2u]=o.z;result[4u*i+3u]=o.w;
}`;
if (process.env.DUMP_WGSL) {
  const out = process.env.DUMP_WGSL;mkdirSync(out, { recursive: true });
  writeFileSync(`${out}/000.wgsl`, `// label: narrow-band clamp probe\n${moduleWGSL}`);
  console.log(`wrote 1 module to ${out}`);process.exit(0);
}

/** A deterministic stream of integers. */
function stream(seed: number) { let s = seed >>> 0; return (n: number) => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return Math.floor(s / 4294967296 * n); }; }
const f = Math.fround;
/** A world's script and what each entry must give; box where the box clamp must give the same. */
function evaluation(w: typeof WORLDS[number], seed: number) {
  const script = new Int32Array(8 + 8 * CAPACITY), floats = new Float32Array(script.buffer), expected = new Int32Array(4 * CAPACITY), real = new Float32Array(expected.buffer);
  const box = new Float64Array(3 * CAPACITY), out: V = [0, 0, 0], next = stream(seed);
  let count = 0, crossing = 0, stopped = 0;
  const entry = (op: number, home: V) => { if (count >= CAPACITY) throw new Error("The evaluation script is full");script.set([op, ...home], 8 + 8 * count);return count++; };
  for (const page of w.stored) {
    for (const [op, span] of [[OP.cell, UNIFORM_NEAR_SPAN.cell], [OP.tile, UNIFORM_NEAR_SPAN.tile]] as const) for (let trial = 0; trial < 2; trial++) {
      const home = page.map(v => (v << span.bits) + next(span.last + 1)) as V, reach = 2 + next(span.last - 1);
      const axis = (a: number) => { const low = page[a]! << span.bits, high = low + span.last, s = new Set<number>([home[a]!, low, high]);for (const r of [1, reach, span.last + 1]) { s.add(low - r);s.add(high + r); } return [...s]; };
      for (const x of axis(0)) for (const y of axis(1)) for (const z of axis(2)) {
        const p: V = [x, y, z], at = entry(op, home);script.set(p, 8 + 8 * at + 4);
        narrowBandNear(w.table, w.rows, BLOCKS, span, home, p, out);expected.set(out, 4 * at);expected[4 * at + 3] = 1;
        if (out.some((v, a) => v !== p[a])) crossing++;
        for (let a = 0; a < 3; a++) box[3 * at + a] = Math.min(Math.max(p[a]!, LOW[a]! << span.bits), ((LOW[a]! + PAGES[a]! - 1) << span.bits) + span.last);
      }
    }
    for (const inset of [0.01, 0]) for (let trial = 0; trial < 700; trial++) {
      const home = page.map(v => 32 * v + next(32)) as V;
      // An end within three cells of the page's extent, some exactly on a face and some inside the inset.
      const p = page.map(v => { const k = next(8); return k === 0 ? 32 * v : k === 1 ? 32 * v + 32 : k === 2 ? 32 * v + 32 - 0.005 : 32 * v - 3 + next(38000) / 1000; }) as V;
      const at = entry(OP.point, home);floats.set(p, 8 + 8 * at + 4);floats[8 + 8 * at + 7] = inset;
      narrowBandPointNearIn(w.table, w.rows, BLOCKS, home, p, inset, out);real.set(out, 4 * at);expected[4 * at + 3] = 1;
      if (out.some((v, a) => v !== f(p[a]!))) stopped++;
      for (let a = 0; a < 3; a++) box[3 * at + a] = Math.min(Math.max(f(p[a]!), f(32 * LOW[a]! + f(inset))), f(32 * (LOW[a]! + PAGES[a]!) - f(inset)));
    }
  }
  script[0] = count;
  return { script, expected, real, box, count, crossing, stopped };
}
// Every script is built before any device: one the host cannot build stops the probe now.
const scripts = WORLDS.map((w, i) => evaluation(w, 11 + i));

const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter();assert.ok(adapter);
const device = managedGPUDevice(await adapter.requestDevice(), { requireWorkerRealm: false });
device.addEventListener("uncapturederror", e => { e.preventDefault();fail(`device: ${e.error.message}`); });
const module = device.createShaderModule({ label: "Narrow-band clamp probe", code: moduleWGSL });
const messages = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
if (messages.length) throw new Error(messages.map(m => `${m.lineNum}: ${m.message}`).join("\n"));
const storage = (binding: number, type: GPUBufferBindingType): GPUBindGroupLayoutEntry => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
const layout = device.createBindGroupLayout({ entries: [storage(0, "read-only-storage"), storage(1, "read-only-storage"), storage(2, "storage")] });
const pipeline = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module, entryPoint: "evaluate" } });
const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const tail = device.createBuffer({ size: 4 * (ROWS + 27 * BLOCKS), usage }), scriptBuffer = device.createBuffer({ size: 4 * (8 + 8 * CAPACITY), usage }), resultBuffer = device.createBuffer({ size: 16 * CAPACITY, usage });
const group = device.createBindGroup({ layout, entries: [tail, scriptBuffer, resultBuffer].map((buffer, binding) => ({ binding, resource: { buffer } })) });
const target = device.createBuffer({ size: 16 * CAPACITY, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

for (const [index, w] of WORLDS.entries()) {
  const e = scripts[index]!;
  device.queue.writeBuffer(tail, 0, w.table);device.queue.writeBuffer(tail, 4 * UNIFORM_PAGE_TABLE_WORDS, w.records);
  device.queue.writeBuffer(tail, 4 * ROWS, w.rows, 0, 27 * BLOCKS);
  device.queue.writeBuffer(scriptBuffer, 0, e.script, 0, 8 + 8 * e.count);
  device.queue.writeBuffer(resultBuffer, 0, new Uint32Array(4 * e.count).fill(0xdeadbeef));
  const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);pass.setBindGroup(0, group);pass.dispatchWorkgroups(GROUPS);pass.end();
  encoder.copyBufferToBuffer(resultBuffer, 0, target, 0, 16 * e.count);device.queue.submit([encoder.finish()]);
  await target.mapAsync(GPUMapMode.READ, 0, 16 * e.count);
  const got = new Int32Array(target.getMappedRange(0, 16 * e.count).slice(0)), gotReal = new Float32Array(got.buffer);target.unmap();
  let exact = 0, rounded = 0, boxed = 0;
  for (let i = 0; i < e.count; i++) {
    const a = 8 + 8 * i, op = e.script[a]!, home = [...e.script.subarray(a + 1, a + 4)];
    const tap = op === OP.point ? [...new Float32Array(e.script.buffer, 4 * (a + 4), 3)] : [...e.script.subarray(a + 4, a + 7)];
    if (got[4 * i + 3] !== 1) fail(`${w.name}: entry ${i} (op ${op}, home ${home}, tap ${tap}): the second form disagrees`);
    for (let k = 0; k < 3; k++) {
      const j = 4 * i + k;
      if (op !== OP.point) {
        if (got[j] !== e.expected[j]) fail(`${w.name}: entry ${i} (op ${op}, home ${home}, tap ${tap}) axis ${k}: ${got[j]}, the twin ${e.expected[j]}`);
        else if (w.box && got[j] !== e.box[3 * i + k]) fail(`${w.name}: entry ${i} (op ${op}, home ${home}, tap ${tap}) axis ${k}: ${got[j]}, the box clamp ${e.box[3 * i + k]}`);
        else boxed += Number(w.box);
        continue;
      }
      // A position: the twin's bits, or within two units in the last place of it where the backend orders the sum otherwise.
      const want = e.real[j]!, ulp = Math.abs(want) * 2 ** -23;
      if (got[j] === e.expected[j]) exact++;
      else if (Math.abs(gotReal[j]! - want) <= 2 * ulp) rounded++;
      else fail(`${w.name}: entry ${i} (home ${home}, end ${tap}, inset ${new Float32Array(e.script.buffer, 4 * (a + 7), 1)[0]}) axis ${k}: ${gotReal[j]}, the twin ${want}`);
      if (w.box) { if (Math.abs(gotReal[j]! - e.box[3 * i + k]!) <= 2 * ulp) boxed++;else fail(`${w.name}: entry ${i} (home ${home}, end ${tap}) axis ${k}: ${gotReal[j]}, the box clamp ${e.box[3 * i + k]}`); }
    }
  }
  console.log(JSON.stringify({ world: w.name, stored: w.stored.length, entries: e.count, clamped: e.crossing, stopped: e.stopped, positionsExact: exact, positionsRounded: rounded, boxAgreed: boxed, errors: errors.length }));
}
await device.queue.onSubmittedWorkDone();
if (errors.length) { console.error(`narrow-band clamps: ${errors.length} failure(s)`);process.exit(1); }
device.destroy();
console.log(`narrow-band clamps: ok (${Math.round(performance.now() - started)} ms)`);
