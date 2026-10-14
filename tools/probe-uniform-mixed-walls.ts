/** Dawn proof of the walls by page kind (lib/methods/uniform/uniform-mixed-walls.wgsl.ts)
 * against their host twins.
 *
 * Two page tables, each with its block records, neighbour rows and wall-face
 * list as a bare tail: a pool of 32 blocks seeded with a tank of 3 x 2 x 2
 * pages in a SOLID shell under an AIR top, one page of it AIR and one SOLID,
 * in scattered blocks with free ones between; and the seed of a 96 x 64 x 64
 * box under an open top (permuted blocks, an AIR page). Over each, one kernel
 * evaluates umWallBlock and umWallPlane at vertices on, beside and between
 * the page faces, umWallWithin, umWallAhead toward every face for reaches
 * that stop short of, reach and pass the wall, umCellHeld, umTileHeld and
 * umPointHeld around every face, and umWallFaceVertex and umWallFaceTile of
 * every listed face; a second kernel lists the wall vertices as the
 * surface's advectWalls does (a lane a wall-face vertex, striding the list
 * from a launch too small to hold it, one list atomic a group and round),
 * and its list is the twin's as a set, each vertex once.
 *
 * node --import tsx tools/probe-uniform-mixed-walls.ts
 * DUMP_WGSL=<dir>: write the probe's shader modules there and stop before
 * any device (for naga and Tint). */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { UNIFORM_PAGE_TABLE_WORDS, uniformBlockNeighbourLayout, uniformBlockNeighbourWords, uniformLatticeAddress, uniformLatticeAddressWGSL } from "../lib/methods/uniform/uniform-lattice-address";
import {
  uniformMixedSolidHoldPointWGSL, uniformMixedSolidHoldWGSL, uniformMixedWallAheadWGSL, uniformMixedWallFaceWGSL, uniformMixedWallVertexWGSL,
  uniformPointHeld, uniformSpanHeld, uniformWallAhead, uniformWallBlock, uniformWallFaceTile, uniformWallFaceVertex, uniformWallPlane, uniformWallWithin,
} from "../lib/methods/uniform/uniform-mixed-walls.wgsl";
import { UNIFORM_PAGE_POOL_KIND as KIND, uniformPagePoolLayout, uniformPagePoolSeed, type UniformPagePoolSeedPage } from "../lib/methods/uniform/uniform-page-pool";

type V = [number, number, number];
const errors: string[] = [], started = performance.now();
const fail = (message: string) => { errors.push(message);console.error(`FAIL ${message}`); };

interface World { name: string; table: Uint32Array; records: Uint32Array; low: V; high: V }
const TANK = [[2, 2, 2], [3, 2, 2], [4, 2, 2], [2, 3, 2], [4, 3, 2], [2, 2, 3], [3, 2, 3], [4, 2, 3], [2, 3, 3], [3, 3, 3]], HELD = [17, 3, 29, 8, 1, 22, 12, 30, 5, 19];
const poolSeed = uniformPagePoolSeed(uniformPagePoolLayout({ blocks: 32, idleSteps: 2 }), TANK.map(([x, y, z], i): UniformPagePoolSeedPage => ({ x: x!, y: y!, z: z!, block: HELD[i]! })), (x, y, z) => {
  if (x >= 2 && x <= 4 && y >= 2 && y <= 3 && z >= 2 && z <= 3) return x === 4 && y === 3 && z === 3 ? KIND.solid : KIND.air;
  const shell = x >= 1 && x <= 5 && y >= 1 && y <= 4 && z >= 1 && z <= 4;
  return shell && !(y === 4 && x >= 2 && x <= 4 && z >= 2 && z <= 3) ? KIND.solid : KIND.air;
});
const boxAddress = uniformLatticeAddress([96, 64, 64], true, { blocks: 20, permute: 7, air: [[1, 1, 0]] });
const WORLDS: World[] = [
  { name: "pool", table: new Uint32Array(poolSeed.table), records: new Uint32Array(poolSeed.blocks), low: [64, 64, 64], high: [160, 128, 128] },
  { name: "box", table: new Uint32Array(boxAddress.seed.table), records: new Uint32Array(boxAddress.seed.blocks), low: [0, 0, 0], high: [96, 64, 64] },
];

// The evaluation: eight script words an entry (op, a, x, y, z, b, two floats), eight result words.
const OP = { wall: 0, within: 1, ahead: 2, span: 3, point: 4, faceVertex: 5, faceTile: 6 } as const;
const H: V = [Math.fround(0.01), Math.fround(0.0125), Math.fround(0.01)];
const CAPACITY = 1 << 19, GROUPS = CAPACITY / 64, LIST_GROUPS = 61;
const moduleWGSL = (blocks: number) => /* wgsl */ `
@group(0) @binding(0) var<storage,read> tailWords:array<u32>;
const UM_D=vec3u(2048u);const UM_T=UM_D/4u;
fn umPageWord(page:vec3i)->u32{let e=vec3u(page)&vec3u(63u);return tailWords[e.x|(e.y<<6u)|(e.z<<12u)];}
fn umBlockWord(b:u32)->u32{return tailWords[${UNIFORM_PAGE_TABLE_WORDS}u+b];}
fn umWallFaceCount()->u32{return tailWords[${uniformBlockNeighbourLayout(blocks).wallCount}u];}
fn umWallFace(i:u32)->u32{return tailWords[${uniformBlockNeighbourLayout(blocks).walls}u+i];}
${uniformLatticeAddressWGSL({ cells: "UM_D", tiles: "UM_T", constant: true, paged: { blocks: `${blocks}u` } })}
${uniformMixedWallVertexWGSL}
${uniformMixedWallFaceWGSL}
${uniformMixedWallAheadWGSL}
${uniformMixedSolidHoldWGSL}
${uniformMixedSolidHoldPointWGSL}
@group(1) @binding(0) var<storage,read> script:array<i32>;
@group(1) @binding(1) var<storage,read_write> result:array<u32>;
@compute @workgroup_size(64) fn evaluate(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=u32(script[0])){return;}
 let s=8u+8u*i;let op=u32(script[s]);let a=script[s+1u];let p=vec3i(script[s+2u],script[s+3u],script[s+4u]);let b=script[s+5u];
 var o=array<u32,8>(0u,0u,0u,0u,0u,0u,0u,0u);
 switch op {
  case ${OP.wall}u {o[0]=umWallBlock(p,u32(a));o[1]=umWallPlane(p);}
  case ${OP.within}u {o[0]=u32(umWallWithin(p,u32(a),b));}
  case ${OP.ahead}u {
   let w=umWallAhead(p,u32(a),bitcast<f32>(script[s+6u]),bitcast<f32>(script[s+7u]),vec3f(${H.join(",")}));
   o[0]=bitcast<u32>(w.plane);o[1]=w.blocks[0];o[2]=w.blocks[1];o[3]=w.blocks[2];o[4]=w.blocks[3];
  }
  case ${OP.span}u {let r=bitcast<vec3u>(umSpanHeld(p,u32(a)));let n=bitcast<vec3u>(select(umTileHeld(p),umCellHeld(p),a==5));o[0]=r.x;o[1]=r.y;o[2]=r.z;o[3]=u32(all(r==n));}
  case ${OP.point}u {let r=bitcast<vec3u>(umPointHeld(bitcast<vec3f>(p)));o[0]=r.x;o[1]=r.y;o[2]=r.z;}
  case ${OP.faceVertex}u {let r=bitcast<vec3u>(umWallFaceVertex(u32(a),u32(b)));o[0]=r.x;o[1]=r.y;o[2]=r.z;}
  case ${OP.faceTile}u {let r=umWallFaceTile(u32(a),u32(b));o[0]=r.x;o[1]=r.y;o[2]=r.z;}
  default {}
 }
 for(var k=0u;k<8u;k++){result[8u*i+k]=o[k];}
}
// The surface's advectWalls without its fields: the same lanes, rounds, barriers and list atomic.
struct Listed {count:atomic<u32>,data:array<u32>}
@group(1) @binding(2) var<storage,read_write> listed:Listed;
var<workgroup> wallListed:atomic<u32>;
var<workgroup> wallBase:u32;
@compute @workgroup_size(64) fn list(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let lanes=select(0u,1089u*umWallFaceCount(),script[1]>0);
 for(var first=0u;first<lanes;first+=64u*groups.x){
  if(lane==0u){atomicStore(&wallListed,0u);}
  workgroupBarrier();
  let i=first+gid.x;var word=0xffffffffu;var slot=0u;
  if(i<lanes){
   let face=umWallFace(i/1089u);let vertex=umWallFaceVertex(face,i%1089u);
   if(umWallPlane(vertex)==face%6u&&umWallBlock(vertex,face%6u)==face/6u){
    slot=atomicAdd(&wallListed,1u);word=u32(vertex.x)|(u32(vertex.y)<<10u)|(u32(vertex.z)<<20u);
   }
  }
  workgroupBarrier();
  if(lane==0u){let n=atomicLoad(&wallListed);if(n>0u){wallBase=atomicAdd(&listed.count,n);}}
  workgroupBarrier();
  if(word!=0xffffffffu){listed.data[wallBase+slot]=word;}
 }
}`;

/** A world's script and what the twins answer. */
function evaluation(w: World, blocks: number): { script: Int32Array<ArrayBuffer>; expected: Uint32Array; count: number; list: Set<number> } {
  const script = new Int32Array(8 + 8 * CAPACITY), floats = new Float32Array(script.buffer), expected = new Uint32Array(8 * CAPACITY), signed = new Int32Array(expected.buffer), out: V = [0, 0, 0];
  let count = 0;
  const entry = (op: number, a: number, p: readonly number[], b = 0): number => {
    if (count >= CAPACITY) throw new Error("The script is full");
    const s = 8 + 8 * count;script[s] = op;script[s + 1] = a;script[s + 2] = p[0]!;script[s + 3] = p[1]!;script[s + 4] = p[2]!;script[s + 5] = b;
    return count++;
  };
  const along = (a: number, offsets: number[]) => { const s = new Set<number>();for (let f = w.low[a]!; f <= w.high[a]!; f += 32) for (const o of offsets) s.add(f + o);return [...s]; };
  const ahead = { plane: 0, blocks: [0, 0, 0, 0] as [number, number, number, number] };
  const xs = along(0, [-1, 0, 1, 4, 15, 31]), ys = along(1, [-1, 0, 1, 4, 15, 31]), zs = along(2, [-1, 0, 2, 16]);
  for (const z of zs) for (const y of ys) for (const x of xs) {
    const p: V = [x, y, z];
    for (let k = 0; k < 6; k++) {
      let at = entry(OP.wall, k, p);expected[8 * at] = uniformWallBlock(w.table, blocks, p, k);expected[8 * at + 1] = uniformWallPlane(w.table, blocks, p);
      for (const [reach, bound] of [[0.004, -0.02], [0.3, 0.05], [2, -0.3]] as const) {
        at = entry(OP.ahead, k, p);floats[8 + 8 * at + 6] = reach;floats[8 + 8 * at + 7] = bound;
        uniformWallAhead(w.table, blocks, p, k, Math.fround(reach), Math.fround(bound), H, ahead);
        signed[8 * at] = ahead.plane;expected.set(ahead.blocks, 8 * at + 1);
      }
    }
    for (let axis = 0; axis < 3; axis++) for (const reach of [2, 8]) { const at = entry(OP.within, axis, p, reach);expected[8 * at] = +uniformWallWithin(w.table, blocks, p, axis, reach); }
    for (const bits of [5, 3]) {
      const q = bits === 5 ? p : p.map(v => v >> 2), at = entry(OP.span, bits, q);
      uniformSpanHeld(w.table, blocks, q, bits, out);signed.set(out, 8 * at);expected[8 * at + 3] = 1;
    }
    for (const d of [0, 0.25, -0.75, -3.5]) {
      const q = p.map((v, a) => Math.fround(v + (a === 1 ? d : -d * 0.5))), at = entry(OP.point, 0, [0, 0, 0]);
      floats.set(q, 8 + 8 * at + 2);uniformPointHeld(w.table, blocks, q, out);new Float32Array(expected.buffer, 4 * 8 * at, 3).set(out);
    }
  }
  const tail = uniformBlockNeighbourWords(w.table, w.records), faces = tail[27 * blocks]!, list = new Set<number>();
  for (let i = 0; i < faces; i++) {
    const face = tail[27 * blocks + 1 + i]!;
    for (let l = 0; l < 1089; l++) {
      const at = entry(OP.faceVertex, face, [0, 0, 0], l);uniformWallFaceVertex(w.records, face, l, out);expected.set(out, 8 * at);
      if (uniformWallPlane(w.table, blocks, out) === face % 6 && uniformWallBlock(w.table, blocks, out, face % 6) === Math.floor(face / 6)) {
        const word = out[0] | (out[1] << 10) | (out[2] << 20);
        if (list.has(word)) throw new Error(`The twin lists vertex ${out} twice`);
        list.add(word);
      }
    }
    for (let lane = 0; lane < 64; lane++) { const at = entry(OP.faceTile, face, [0, 0, 0], lane);uniformWallFaceTile(w.records, face, lane, out);expected.set(out, 8 * at); }
  }
  script[0] = count;script[1] = 1;
  return { script, expected, count, list };
}
/** A world's tail: the table, the block records, then the rows, the count and the list. */
function tailWords(w: World): Uint32Array<ArrayBuffer> {
  const n = w.records.length, words = new Uint32Array(uniformBlockNeighbourLayout(n).end);
  words.set(w.table);words.set(w.records, UNIFORM_PAGE_TABLE_WORDS);words.set(uniformBlockNeighbourWords(w.table, w.records), uniformBlockNeighbourLayout(n).rows);
  return words;
}

const built = WORLDS.map(w => ({ w, e: evaluation(w, w.records.length) }));
for (const { w, e } of built) console.log(JSON.stringify({ script: w.name, blocks: w.records.length, entries: e.count, wallVertices: e.list.size }));
if (process.env.DUMP_WGSL) {
  const out = process.env.DUMP_WGSL;mkdirSync(out, { recursive: true });
  built.forEach(({ w }, i) => writeFileSync(`${out}/00${i}.wgsl`, `// label: wall probe over the ${w.name} tail\n${moduleWGSL(w.records.length)}`));
  console.log(`wrote ${built.length} modules to ${out}`);process.exit(0);
}

const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter();assert.ok(adapter);
const device = managedGPUDevice(await adapter.requestDevice(), { requireWorkerRealm: false });
device.addEventListener("uncapturederror", e => { e.preventDefault();fail(`device: ${e.error.message}`); });
async function read(buffer: GPUBuffer, size: number): Promise<Uint32Array<ArrayBuffer>> {
  const target = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }), encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, target, 0, size);device.queue.submit([encoder.finish()]);
  await target.mapAsync(GPUMapMode.READ);
  const words = new Uint32Array(target.getMappedRange().slice(0));target.unmap();target.destroy();
  return words;
}
const storage = (binding: number, type: GPUBufferBindingType): GPUBindGroupLayoutEntry => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
const heldLayout = device.createBindGroupLayout({ entries: [storage(0, "read-only-storage")] }), scriptLayout = device.createBindGroupLayout({ entries: [storage(0, "read-only-storage"), storage(1, "storage"), storage(2, "storage")] });
const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const scriptBuffer = device.createBuffer({ size: 4 * (8 + 8 * CAPACITY), usage }), resultBuffer = device.createBuffer({ size: 32 * CAPACITY, usage }), listBuffer = device.createBuffer({ size: 4 * (1 + 65536), usage });
const scriptGroup = device.createBindGroup({ layout: scriptLayout, entries: [scriptBuffer, resultBuffer, listBuffer].map((buffer, binding) => ({ binding, resource: { buffer } })) });
for (const { w, e } of built) {
  const blocks = w.records.length, module = device.createShaderModule({ label: `Wall probe over the ${w.name} tail`, code: moduleWGSL(blocks) });
  const messages = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
  if (messages.length) throw new Error(messages.map(m => `${m.lineNum}: ${m.message}`).join("\n"));
  const layout = device.createPipelineLayout({ bindGroupLayouts: [heldLayout, scriptLayout] });
  const [evaluate, list] = await Promise.all(["evaluate", "list"].map(entryPoint => device.createComputePipelineAsync({ layout, compute: { module, entryPoint } })));
  const tail = device.createBuffer({ size: 4 * uniformBlockNeighbourLayout(blocks).end, usage });
  device.queue.writeBuffer(tail, 0, tailWords(w));
  const held = device.createBindGroup({ layout: heldLayout, entries: [{ binding: 0, resource: { buffer: tail } }] });
  device.queue.writeBuffer(scriptBuffer, 0, e.script, 0, 8 + 8 * e.count);
  device.queue.writeBuffer(resultBuffer, 0, new Uint32Array(8 * e.count).fill(0xdeadbeef));
  device.queue.writeBuffer(listBuffer, 0, new Uint32Array(1 + 65536));
  const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
  pass.setBindGroup(0, held);pass.setBindGroup(1, scriptGroup);
  pass.setPipeline(evaluate!);pass.dispatchWorkgroups(GROUPS);
  pass.setPipeline(list!);pass.dispatchWorkgroups(LIST_GROUPS);pass.end();
  device.queue.submit([encoder.finish()]);
  const got = await read(resultBuffer, 32 * e.count);
  let bad = 0;
  for (let i = 0; i < e.count; i++) for (let k = 0; k < 8; k++) if (got[8 * i + k] !== e.expected[8 * i + k] && bad++ < 6) {
    const s = 8 + 8 * i;
    fail(`${w.name}: entry ${i} (op ${e.script[s]}, a ${e.script[s + 1]}, p ${[...e.script.subarray(s + 2, s + 5)]}, b ${e.script[s + 5]}) word ${k}: ${got[8 * i + k]! | 0}, expected ${e.expected[8 * i + k]! | 0}`);
  }
  if (bad > 6) fail(`${w.name}: ${bad} result words differ`);
  const words = await read(listBuffer, 4 * (1 + 65536)), n = words[0]!, seen = new Set<number>();
  if (n !== e.list.size) fail(`${w.name}: ${n} wall vertices listed, expected ${e.list.size}`);
  for (let i = 0; i < Math.min(n, 65536); i++) { const v = words[1 + i]!;if (seen.has(v)) fail(`${w.name}: vertex word ${v} listed twice`);seen.add(v);if (!e.list.has(v)) fail(`${w.name}: vertex word ${v} is no wall vertex`); }
  const lanes = 1089 * uniformBlockNeighbourWords(w.table, w.records)[27 * blocks]!;
  console.log(JSON.stringify({ world: w.name, entries: e.count, listed: n, expectedListed: e.list.size, listLanes: lanes, listRounds: Math.ceil(lanes / (64 * LIST_GROUPS)), errors: errors.length }));
  tail.destroy();
}
await device.queue.onSubmittedWorkDone();
if (errors.length) { console.error(`uniform mixed walls: ${errors.length} failure(s)`);process.exit(1); }
device.destroy();
console.log(`uniform mixed walls: ok (${Math.round(performance.now() - started)} ms)`);
