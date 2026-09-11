import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;
dawnTest("gravity changes reclassify cached pressure rows even with unchanged density and topology", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "gravity-pressure-cache");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    const gpu: GPU = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const source = readFileSync(new URL("../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
    const production = source.match(/fn publishCanonicalPressureRowTile\([\s\S]*?\n}/)?.[0];
    assert.ok(production);
    // Every cached row was active; reclassification now rejects it. Only the
    // second workgroup carries a changed acceleration receipt. Scalar and
    // topology inputs deliberately stay identical, isolating the cache seam.
    const shader = device.createShaderModule({ code: `
struct Params {counts:vec4u,acceleration:vec4f}
var<private>p:Params;
var<workgroup>pcmRowBallot:array<u32,64>;
@group(0)@binding(0)var<storage,read_write>result:array<u32>;
fn pcmRowContains(row:u32)->bool{_=row;return true;}
fn pcmRowPublicationOpen()->bool{return true;}
fn rowAccepted(row:u32)->bool{_=row;return true;}
fn rowTermOffset(row:u32)->u32{_=row;return 0u;}
fn rowTermCount(row:u32)->u32{_=row;return 1u;}
fn rowTermRange(row:u32)->vec2u{let first=rowTermOffset(row);return vec2u(first,first+rowTermCount(row));}
fn termCell(term:u32)->u32{return term;}
fn fsm1ChangedOrFlipCell(cell:u32)->bool{_=cell;
 if(p.acceleration.w>0.5){result[4]=1u;}return false;}
fn pcmRowPriorTopologyGeneration()->u32{return 1u;}
fn ptrTopologyGeneration()->u32{return 1u;}
fn hasStaticSolidVoxels()->bool{return false;}
fn classifyPressureRow(row:u32)->bool{_=row;return false;}
fn pcmRowPublishWord(word:u32,bits:u32)->bool{result[word]=bits;return true;}
${production.replaceAll(/@builtin\([^)]*\)/g, "")}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id)wid:vec3u,@builtin(num_workgroups)nwg:vec3u,
 @builtin(local_invocation_index)lane:u32){
 p=Params(vec4u(0,128,0,0),vec4f(0,-9.81,0,f32(wid.x)));
 publishCanonicalPressureRowTile(wid.x+nwg.x*wid.y,lane);
}` });
    assert.deepEqual((await shader.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module: shader, entryPoint: "main" } });
    const output = device.createBuffer({ size: 20, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: 20, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: output } }] });
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(2); pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, 20); device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    assert.deepEqual([...new Uint32Array(readback.getMappedRange())], [0xffffffff, 0xffffffff, 0, 0, 0]);
    readback.unmap(); readback.destroy(); output.destroy();
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
