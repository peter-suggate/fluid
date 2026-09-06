import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

const resident = readFileSync(process.env.CM12_CENSUS_SOURCE ?? new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
const incremental = readFileSync(new URL("../lib/methods/adaptive-mass/sparse-cm12-incremental-activity.wgsl.ts", import.meta.url), "utf8");
function production(source: string, name: string) {
  const body = source.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n}`))?.[0];
  assert.ok(body, name);
  return body.replaceAll("@builtin(global_invocation_id)", "");
}
const live = new Set<GPU>();
(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("D4 score publication keeps incremental activity census balanced", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "activity-d4-census");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    assert.ok(device);
    const code = `
struct Params{dispatch:vec4u,activityTiming:vec4f}
const p=Params(vec4u(0,0,0,3),vec4f(0,.7,0,.2));
const ACTIVITY_BRICK_CENSUS=256u;const ACTIVITY_SCORE_HISTOGRAM=512u;
@group(0)@binding(0)var<storage,read_write>activity:array<atomic<u32>>;
@group(0)@binding(1)var<storage,read_write>conditioning:array<atomic<i32>>;
fn activityRecord(brick:u32)->u32{return 64u+64u*brick;}
${["incrementalActivityRemoveCensus", "incrementalActivityAddCensus", "incrementalActivityReplaceCensus"].map(n => production(incremental, n)).join("\n")}
${production(resident, "commitActivityHorizontalD4")}
fn seed(brick:u32,score:u32,reasons:u32){
 let record=activityRecord(brick);atomicStore(&activity[record],score);
 atomicStore(&activity[record+1u],reasons);incrementalActivityAddCensus(brick,score,reasons);
}
@compute @workgroup_size(1)
fn main(){
 seed(0u,0u,0u);seed(1u,255u,1u);
 // Brick 2 was never measured; symmetry must not invent a census entry.
 for(var step=0u;step<40u;step+=1u){
  let score=select(0u,255u,(step&1u)==0u);let reasons=select(0u,1u,score!=0u);
  for(var brick=0u;brick<3u;brick+=1u){
   atomicStore(&conditioning[brick],bitcast<i32>(score));
   atomicStore(&conditioning[3u+brick],bitcast<i32>(reasons));
   commitActivityHorizontalD4(vec3u(brick,0,0));
  }
  // A normal subsequent measurement removes the rewritten, published score.
  for(var brick=0u;brick<2u;brick+=1u){
   incrementalActivityRemoveCensus(brick);seed(brick,score,reasons);
  }
 }
}
`;
    const shader = device.createShaderModule({ code });
    assert.deepEqual((await shader.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
    const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: shader, entryPoint: "main" } });
    const activity = device.createBuffer({ size: 1024 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const conditioning = device.createBuffer({ size: 15 * 4, usage: GPUBufferUsage.STORAGE });
    const copy = device.createBuffer({ size: activity.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: activity } }, { binding: 1, resource: { buffer: conditioning } },
    ] });
    const encoder = device.createCommandEncoder(); const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
    encoder.copyBufferToBuffer(activity, 0, copy, 0, copy.size); device.queue.submit([encoder.finish()]);
    await copy.mapAsync(GPUMapMode.READ); const words = new Uint32Array(copy.getMappedRange()).slice(); copy.unmap();
    assert.equal(words[2], 0, "no surface bricks in the final quiet epoch");
    assert.equal(words[3], 0, "hot count must not underflow after symmetry replacement");
    assert.equal(words[4], 2, "both measured bricks are quiet");
    assert.equal(words[512], 2, "quiet score histogram matches published words");
    assert.equal(words[767], 0, "obsolete hot histogram entries are removed");
    assert.deepEqual([...words.slice(256, 259)], [1, 1, 0]);
    activity.destroy(); conditioning.destroy(); copy.destroy();
  } finally { device?.destroy(); live.clear(); await releaseWebGPUExclusiveLock(); }
});
