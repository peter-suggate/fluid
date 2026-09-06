import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

const source = readFileSync(process.env.CM12_ACTIVITY_SOURCE ?? new URL("../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
const production = (name: string) => {
  const body = source.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n}`))?.[0];
  assert.ok(body, name);
  return body.replaceAll("@builtin(global_invocation_id)", "");
};
const live = new Set<GPU>();
(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("D4 activity aggregation keeps curvature floors on the dyadic ladder", { timeout: 120_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "activity-d4-rung");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const code = `
const INVALID=0xffffffffu;const BRICK_FINE_RESOLUTION=8u;
struct Params{dispatch:vec4u,dimensions:vec4u}
const p=Params(vec4u(0,0,0,8),vec4u(32));
@group(0)@binding(0)var<storage,read_write>activity:array<atomic<u32>>;
@group(0)@binding(1)var<storage,read_write>conditioning:array<atomic<i32>>;
fn activityRecord(brick:u32)->u32{return 64u*brick;}
fn cm12WorldLeafCoordinate(brick:u32)->vec3i{_=brick;return vec3i(0,0,1);}
fn cm12WorldOwnerAt(q:vec3i)->u32{
 let xs=array<i32,8>(0,3,0,3,1,2,1,2);let zs=array<i32,8>(1,1,2,2,0,0,3,3);
 for(var i=0u;i<8u;i+=1u){if(q.x==xs[i]&&q.z==zs[i]){return i;}}
 return INVALID;
}
${production("activityD4MaskToOwn")}
${source.includes("fn mergeActivityReasonWords(") ? production("mergeActivityReasonWords") : ""}
${production("preserveActivityHorizontalD4")}
@compute @workgroup_size(1)
fn main(){preserveActivityHorizontalD4(vec3u(0));}
`;
    const shader = device!.createShaderModule({ code });
    assert.deepEqual((await shader.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
    const pipeline = device!.createComputePipeline({ layout: "auto", compute: { module: shader, entryPoint: "main" } });
    const input = device!.createBuffer({ size: 4 * 8 * 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const output = device!.createBuffer({ size: 4 * 8 * 5, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const copy = device!.createBuffer({ size: 4 * 8 * 5, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const group = device!.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: input } }, { binding: 1, resource: { buffer: output } },
    ] });
    for (const pair of [[1, 4], [2, 4], [1, 8], [4, 8], [8, 16], [2, 2]]) {
      const words = new Uint32Array(8 * 64);
      let expectedFlags = 0;
      for (let i = 0; i < 8; i++) {
        // Low reason flags and a high unrelated bit must remain a union.
        const flags = ((1 << i) | (i === 7 ? 0x80000000 : 0)) >>> 0;
        words[64 * i] = 20 + i;
        words[64 * i + 1] = (flags | (pair[i % 2]! << 16)) >>> 0;
        words[64 * i + 2] = i | ((20 - i) << 8) | ((30 - i) << 16);
        expectedFlags = (expectedFlags | flags) >>> 0;
      }
      device!.queue.writeBuffer(input, 0, words);
      const encoder = device!.createCommandEncoder(); const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
      encoder.copyBufferToBuffer(output, 0, copy, 0, copy.size); device!.queue.submit([encoder.finish()]);
      await copy.mapAsync(GPUMapMode.READ); const actual = new Uint32Array(copy.getMappedRange()).slice(); copy.unmap();
      const rung = (actual[8]! >>> 16) & 31;
      assert.equal(rung, Math.max(...pair), `${pair} must merge as a numeric floor`);
      assert.equal(rung & (rung - 1), 0, "merged rung must remain a power of two");
      assert.equal((actual[8]! & ~(31 << 16)) >>> 0, expectedFlags);
      assert.equal(actual[0], 27); assert.equal(actual[16], 7 | (13 << 8) | (23 << 16));
    }
    input.destroy(); output.destroy(); copy.destroy();
  } finally { device?.destroy(); live.clear(); await releaseWebGPUExclusiveLock(); }
});
