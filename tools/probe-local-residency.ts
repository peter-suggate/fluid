/** Real GPU request/commit/mark versus an independent swept-tile union.
 * Run under the shared GPU lock: node --import tsx tools/probe-local-residency.ts */
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { uniformLatticeAddress } from "../lib/methods/uniform/uniform-lattice-address";
import { uniformMixedSupportWords, uniformMixedTopologyWGSL } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import { UniformPagePool, uniformPagePoolWGSL, UNIFORM_PAGE_POOL_FATAL, UNIFORM_PAGE_POOL_STATUS as S } from "../lib/methods/uniform/uniform-page-pool";
import { uniformResidencyPageCoord, uniformResidencyBlockHeld } from "../lib/methods/uniform/uniform-mixed-residency";
import { localResidencyWGSL, localResidencyRequestWGSL, localResidencyMarkWGSL } from "../lib/methods/uniform/uniform-local-residency";

const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter();assert.ok(adapter);
const device = await adapter.requestDevice();
const errors: string[] = [];
device.addEventListener("uncapturederror", e => { e.preventDefault();errors.push(e.error.message); });
const a = uniformLatticeAddress([192, 128, 128], false, { blocks: 110, air: [[2, 1, 1], [3, 1, 1], [2, 2, 1], [3, 2, 1]] });
const n = a.tileKeys, pages = 8 * a.blocks;
const pool = new UniformPagePool(device, { ...a.pool, worldPolicy: 0, idleSteps: 2, closureCap: 32 });
const l = pool.layout;
const owned: GPUBuffer[] = [];
const storage = (words: number) => { const b = device.createBuffer({ size: 4 * words, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });owned.push(b);return b; };
const topology = storage(5 * n + l.tail.words), support = storage(uniformMixedSupportWords(n));
const records = storage(4 * pages), result = storage(pages);
const counts = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM });owned.push(counts);
async function read(buffer: GPUBuffer, words: number, offset = 0) {
  const b = device.createBuffer({ size: words * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();encoder.copyBufferToBuffer(buffer, offset * 4, b, 0, words * 4);device.queue.submit([encoder.finish()]);
  await b.mapAsync(GPUMapMode.READ);const out = new Uint32Array(b.getMappedRange().slice(0));b.unmap();b.destroy();return out;
}
try {
  pool.seed(a.seed);await pool.initialize([{ buffer: topology, offset: 5 * n }]);
  const lattice = { dimensions: [192, 128, 128] as [number, number, number], cellSize_m: [1, 1, 1] as [number, number, number], origin_m: { x: 0, y: 0, z: 0 } };
  const module = device.createShaderModule({ code: uniformMixedTopologyWGSL({ lattice, tiles: n }, 0) + /* wgsl */ `
@group(1) @binding(0) var<storage,read> records:array<vec4u>;
${uniformPagePoolWGSL(l, 1, 1, 2)}
@group(1) @binding(3) var<storage,read_write> result:array<u32>;
${localResidencyWGSL}
fn lrRecord(page:u32)->vec4u{return records[page];}
${localResidencyRequestWGSL}
${localResidencyMarkWGSL}
@compute @workgroup_size(64) fn request(@builtin(global_invocation_id) id:vec3u){if(id.x<UM_PAGES){lrAsk(id.x);}}
@compute @workgroup_size(64) fn mark(@builtin(global_invocation_id) id:vec3u){if(id.x<UM_PAGES){result[id.x]=u32(lrResident(id.x,8u));}}
` });
  const compilation = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");assert.deepEqual(compilation, []);
  const own = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
  ] });
  const resources = device.createBindGroupLayout({ entries: [0, 1, 2, 3].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: binding < 2 ? "read-only-storage" as const : "storage" as const } })) });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [own, resources] });
  const groups = [device.createBindGroup({ layout: own, entries: [topology, counts, support].map((buffer, binding) => ({ binding, resource: { buffer } })) }),
    device.createBindGroup({ layout: resources, entries: [records, pool.state, pool.work, result].map((buffer, binding) => ({ binding, resource: { buffer } })) })];
  const [request, mark] = await Promise.all(["request", "mark"].map(entryPoint => device.createComputePipelineAsync({ layout, compute: { module, entryPoint, constants: { umDispatchX: 65535 } } })));
  let random = 12345, claims = 0, retirements = 0, comparisons = 0;
  const next = () => (random = (Math.imul(random, 1664525) + 1013904223) >>> 0);
  const held = async () => { const tail = await read(topology, l.tail.words, 5 * n);return { table: tail.subarray(l.tail.table, l.tail.table + l.entries), blocks: tail.subarray(l.tail.blocks, l.tail.blocks + l.blocks) }; };
  for (let step = 0; step < 18; step++) {
    const before = await held(), data = new Uint32Array(4 * pages), boxes: { low: number[];high: number[] }[] = [];
    if (step < 15) for (let p = 0; p < pages; p++) {
      if (!uniformResidencyBlockHeld(before, p >>> 3) || next() % 13 !== 0) continue;
      const q = uniformResidencyPageCoord(before, p, []), radii = Array.from({ length: 6 }, () => next() % 9);
      data[4 * p + 2] = radii[0]! | radii[1]! << 8 | radii[2]! << 16;
      data[4 * p + 3] = radii[3]! | radii[4]! << 8 | radii[5]! << 16;
      for (let k = 0; k < 3; k++) {
        const lane = next() % 64;data[4 * p + (lane >>> 5)]! |= 1 << (lane & 31);
        const c = [lane & 3, lane >>> 2 & 3, lane >>> 4].map((v, axis) => 4 * q[axis]! + v);
        boxes.push({ low: c.map((v, axis) => v - radii[axis]!), high: c.map((v, axis) => v + radii[axis + 3]!) });
      }
    }
    device.queue.writeBuffer(records, 0, data);
    const encoder = device.createCommandEncoder();
    const dispatch = (pipeline: GPUComputePipeline) => { const pass = encoder.beginComputePass();pass.setPipeline(pipeline);groups.forEach((g, i) => pass.setBindGroup(i, g));pass.dispatchWorkgroups(Math.ceil(pages / 64));pass.end(); };
    dispatch(request!);pool.encodeCommit(encoder);dispatch(mark!);device.queue.submit([encoder.finish()]);
    const after = await held(), flags = await read(result, pages), status = await read(pool.state, S.words);
    assert.equal(status[S.fatal], 0, `step ${step}: fatal`);claims += status[S.claimed]!;retirements += status[S.pending]!;
    const meets = (low: number[], edge: number) => boxes.some(box => low.every((v, axis) => v <= box.high[axis]! && v + edge - 1 >= box.low[axis]!));
    for (let z = 0; z < 4; z++) for (let y = 0; y < 4; y++) for (let x = 0; x < 6; x++) {
      const e = x | y << 6 | z << 12, requested = meets([8 * x, 8 * y, 8 * z], 8), block = after.table[e]!;
      if (requested) assert.ok(block > 0 && block < l.blocks - 1, `step ${step}: missing requested page ${x},${y},${z}`);
      if (before.table[e] === 0 && block > 0 && block < l.blocks - 1) assert.ok(requested, `step ${step}: extra claim`);
    }
    for (let p = 0; p < pages; p++) {
      const present = uniformResidencyBlockHeld(after, p >>> 3);
      const expected = present && meets(uniformResidencyPageCoord(after, p, []).map(v => 4 * v), 4);
      assert.equal(flags[p], Number(expected), `step ${step}: support region ${p}`);comparisons++;
    }
    if (step === 17) assert.equal(status[S.resident], 0, "empty censuses retire every page");
  }
  assert.ok(claims > 0 && retirements > 0);
  pool.seed(a.seed);
  const overflow = new Uint32Array(4 * pages);overflow[32] = 1;overflow[34] = 33;
  device.queue.writeBuffer(records, 0, overflow);
  const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
  pass.setPipeline(request!);groups.forEach((g, i) => pass.setBindGroup(i, g));pass.dispatchWorkgroups(Math.ceil(pages / 64));pass.end();
  pool.encodeCommit(encoder);device.queue.submit([encoder.finish()]);
  const rejected = await read(pool.state, S.words);
  assert.equal(rejected[S.fatal], UNIFORM_PAGE_POOL_FATAL.closureCap);assert.equal(rejected[S.requests], 0);
  assert.deepEqual(errors, []);
  console.log(`PASS local allocation: ${comparisons} support regions, ${claims} claims, ${retirements} retirements; zero missing requests, extra claims or incorrect marks; over-cap travel rejected`);
} finally { pool.destroy();for (const b of owned) b.destroy();await device.queue.onSubmittedWorkDone();device.destroy(); }
