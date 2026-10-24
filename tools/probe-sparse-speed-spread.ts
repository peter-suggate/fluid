/** Exercise the production separable speed-certificate kernels across an
 * AIR page that interrupts their intermediate paths. */
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { uniformFramePlanSpreadWGSL } from "../lib/methods/uniform/uniform-frame-plan-spread";

const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter();assert.ok(adapter);
const device = await adapter.requestDevice(), errors: string[] = [];
device.addEventListener("uncapturederror", e => { e.preventDefault();errors.push(e.error.message); });
const N = 32, T = N ** 3, R = 9;
const pack = (value: number) => (value | value << 5 | value << 10 | value << 15 | value << 20 | value << 25) >>> 0;
const key = (x: number, y: number, z: number) => x + N * (y + N * z);
const fast = [[6, 9, 9], [30, 31, 30]];
const data = new Uint32Array(2 * T).fill(pack(2));
for (const [x, y, z] of fast) data[key(x!, y!, z!)] = pack(5);
const speeds = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
const read = device.createBuffer({ size: T * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
try {
  device.queue.writeBuffer(speeds, 0, data);
  const module = device.createShaderModule({ code: /* wgsl */ `
const UM_TILES:u32=${T}u;const umDispatchX:u32=65535u;
const UM_AIR:u32=1u;const UM_REACH_SATURATED:u32=31u;
@group(0) @binding(0) var<storage,read_write> speeds:array<u32>;
fn umTileCoord(t:u32)->vec3u{return vec3u(t%32u,(t/32u)%32u,t/1024u);}
fn umTileKind(p:vec3i)->u32{
 if(any(p<vec3i(0))||any(p>=vec3i(32))){return 2u;}
 if(all(p>=vec3i(8))&&all(p<vec3i(16))){return UM_AIR;}
 return 0u;
}
fn umTileKey(p:vec3i)->u32{if(umTileKind(p)!=0u){return 0xffffffffu;}return u32(p.x+32*(p.y+32*p.z));}
fn umKeyStored(t:u32)->bool{return t<UM_TILES;}
// The live lists (uniform-live-lists): every 512-key block resident.
fn ulResidentCount()->u32{return ${T / 512}u;}
fn ulResidentBlock(i:u32)->u32{return i;}
fn umTileLive(t:u32)->bool{return t<UM_TILES&&umTileKind(vec3i(umTileCoord(t)))==0u;}
fn umPackReach(p:vec3u,m:vec3u)->u32{return p.x|(p.y<<5u)|(p.z<<10u)|(m.x<<15u)|(m.y<<20u)|(m.z<<25u);}
fn umReachPlus(w:u32)->vec3u{return vec3u(w&31u,(w>>5u)&31u,(w>>10u)&31u);}
fn umReachMinus(w:u32)->vec3u{return vec3u((w>>15u)&31u,(w>>20u)&31u,(w>>25u)&31u);}
${[0, 1, 2].map(axis => uniformFramePlanSpreadWGSL(axis, `${R}u`)).join("\n")}
` });
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
  const resources = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }] });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [resources] });
  const group = device.createBindGroup({ layout: resources, entries: [{ binding: 0, resource: { buffer: speeds } }] });
  const pipelines = await Promise.all([0, 1, 2].map(axis => device.createComputePipelineAsync({ layout, compute: { module, entryPoint: `spread${axis}` } })));
  const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();pass.setBindGroup(0, group);
  for (const pipeline of pipelines) { pass.setPipeline(pipeline);pass.dispatchWorkgroups(Math.ceil(T / 64)); }
  pass.end();encoder.copyBufferToBuffer(speeds, T * 4, read, 0, T * 4);device.queue.submit([encoder.finish()]);
  await read.mapAsync(GPUMapMode.READ);const result = new Uint32Array(read.getMappedRange());
  // Independent geometric oracle: the entire box, not the intermediate
  // paths of the separable implementation, determines whether it is valid.
  let checked = 0, bounded = 0, invalidated = 0;
  for (let z = 0; z < N; z++) for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const p = [x, y, z];if (p.every(v => v >= 8 && v < 16)) continue;
    const hole = p.every(v => v - R <= 15 && v + R >= 8);
    const maximum = fast.some(q => p.every((v, a) => Math.abs(v - q[a]!) <= R)) ? 5 : 2;
    assert.equal(result[key(x, y, z)], pack(hole ? 31 : maximum), `tile ${p}`);
    checked++;if (hole) invalidated++;else bounded++;
  }
  assert.equal(result[key(15, 0, 0)], pack(31), "AIR intermediate must not hide the faster tile at 6,9,9");
  assert.ok(bounded > 0 && invalidated > 0);assert.deepEqual(errors, []);
  read.unmap();console.log(`PASS sparse speed certificate: ${checked} tiles, ${bounded} bounded, ${invalidated} conservatively invalidated across AIR; SOLID boundaries retained`);
} finally { speeds.destroy();read.destroy();await device.queue.onSubmittedWorkDone();device.destroy(); }
