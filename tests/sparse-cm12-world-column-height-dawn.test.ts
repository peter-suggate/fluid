import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";

const source = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
const production = (name: string) => {
  const code = source.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n}`))?.[0];
  assert.ok(code, name); return code;
};
const live = new Set<GPU>();
const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;

dawnTest("uniform world-column publication preserves translated XZ height receipts", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "world-column-height");
  let gpu: GPU | undefined, device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu!);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const errors: string[] = [];
    device.addEventListener("uncapturederror", event => errors.push(event.error.message));
    const code = `
struct Params { dimensions:vec4u, dispatch:vec4u, refinementRegionControl:vec4u }
const p=Params(vec4u(128),vec4u(1),vec4u(1));
const INVALID=0xffffffffu;
const BRICK_FINE_RESOLUTION=8u;
const PRESENTATION_PAGE_RESOLUTION=8u;
const PRESENTATION_HEIGHT_COLUMN_AXIS=10u;
var<private>offset:vec3i;
var<private>world:bool;
var<workgroup>activity:array<atomic<u32>,2>;
var<workgroup>presentationHeightCache:array<f32,100>;
var<workgroup>presentationHeightFieldValid:u32;
@group(0)@binding(0)var<storage,read_write>result:array<vec2f>;
fn activityRecord(brick:u32)->u32{_=brick;return 0u;}
fn brickSpan(brick:u32)->u32{_=brick;return 1u;}
fn brickHasUnclippedWorldGeometry(brick:u32)->bool{_=brick;return world;}
fn cm12WorldOwnerAt(q:vec3i)->u32{_=q;return INVALID;}
fn cm12WorldFloorToSpan(q:i32,span:i32)->i32{return i32(floor(f32(q)/f32(span)))*span;}
fn cm12WorldLeafCoordinate(brick:u32)->vec3i{_=brick;return offset/8;}
fn templateBrickCellRange(brick:u32,resolution:u32)->vec2u{_=brick;_=resolution;return vec2u(0,1);}
fn acceptedBrickResolution(brick:u32)->u32{_=brick;return 1u;}
fn cachedRefinementPolicyResolutionBounds(brick:u32)->vec2u{_=brick;return vec2u(1);}
fn cellOpenFraction(cell:u32)->f32{_=cell;return 1.0;}
fn presentationIntegratedWorldColumnHeight(x:i32,z:i32,density:u32)->vec2f{
  _=density;return vec2f(2.0+.02*f32(x-offset.x)+.03*f32(z-offset.z),1.0);
}
fn presentationIntegratedColumnHeight(brick:u32,x:i32,z:i32,density:u32)->vec2f{
  _=brick;return presentationIntegratedWorldColumnHeight(x,z,density);
}
fn presentationContinuousColumnHeight(brick:u32,x:i32,z:i32,density:u32,floorColumn:bool)->vec2f{
  _=floorColumn;return presentationIntegratedColumnHeight(brick,x,z,density);
}
fn compactOwnerCellAt(q:vec3i)->vec3u{_=q;return vec3u(0);}
${production("presentationCanonicalCoarseCoordinate")}
${production("preparePresentationColumnHeights")}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){
  world=group.x!=0u;
  offset=vec3i(select(32,select(-32,160,group.x==2u),world),0,
    select(32,select(-32,160,group.x>=2u),world));
  preparePresentationColumnHeights(lane,0u,offset,0u,false);
  result[64u*group.x+lane]=vec2f(presentationHeightCache[lane],f32(presentationHeightFieldValid));
}`;
    const shader = device.createShaderModule({ code });
    assert.deepEqual((await shader.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module: shader, entryPoint: "main" } });
    const output = device.createBuffer({ size: 4 * 64 * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const copy = device.createBuffer({ size: output.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: output } }] });
    const encoder = device.createCommandEncoder(); const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(4); pass.end();
    encoder.copyBufferToBuffer(output, 0, copy, 0, output.size); device.queue.submit([encoder.finish()]);
    await copy.mapAsync(GPUMapMode.READ); const values = new Float32Array(copy.getMappedRange()).slice();
    copy.unmap(); copy.destroy(); output.destroy();
    for (let group = 0; group < 4; group++) for (let local = 0; local < 64; local++) {
      const at = 2 * (64 * group + local);
      const expected = 2 + .02 * (local % 8 + .5 - 4) + .03 * (Math.floor(local / 8) + .5 - 4);
      assert.ok(Math.abs(values[at]! - expected) < 1e-6,
        `translated group ${group}, sample ${local}: height ${values[at]}, expected ${expected}`);
      assert.equal(values[at + 1], 1, "the complete column receipt remains valid");
    }
    assert.deepEqual(errors, []);
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); if (gpu) live.delete(gpu); }
});
