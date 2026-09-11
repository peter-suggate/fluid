import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";

const source = readFileSync(new URL(
  "../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
const restriction = source.match(/fn restrictedPresentationDensityAt\([\s\S]*?\n}/)![0];
// Retain the pre-optimization spatial walk as an independent address oracle.
const reference = restriction.replace("restrictedPresentationDensityAt", "referenceRestriction")
  .replace(/  \/\/ A dyadic virtual cell[\s\S]*?(?=  \/\/ Walk one finest row)/, "");
const live = new Set<GPU>();
const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;

dawnTest("native restriction is byte-exact across rungs, clipped bricks and cross-leaf queries", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "native-presentation-restriction");
  let gpu: GPU | undefined, device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    live.add(gpu!);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const errors: string[] = [];
    device.addEventListener("uncapturederror", e => errors.push(e.error.message));
    const queries: number[] = [];
    for (let mode = 0; mode < 8; mode++) for (const scale of [1, 2, 4, 8, 16]) {
      for (let z = -8; z < 16; z += scale) for (let y = -8; y < 16; y += scale) {
        for (let x = -8; x < 16; x += scale) queries.push(x, y, z, scale, mode, 0, 0, 0);
      }
    }
    const count = queries.length / 8;
    const module = device.createShaderModule({ code: `
const INVALID:u32=0xffffffffu;
const BRICK_FINE_RESOLUTION:u32=8u;
struct Params{dimensions:vec4u}
var<private>p:Params;
var<private>mode:u32;
@group(0)@binding(0)var<storage,read>queries:array<vec4i>;
@group(0)@binding(1)var<storage,read>state:array<f32>;
@group(0)@binding(2)var<storage,read_write>activity:array<atomic<u32>>;
@group(0)@binding(3)var<storage,read_write>result:array<vec2u>;
fn cm12WorldLeafCoordinate(brick:u32)->vec3i{
  return vec3i(i32(brick%3u)-1,i32((brick/3u)%3u)-1,i32(brick/9u)-1);
}
fn brickSpan(brick:u32)->u32{_=brick;return 1u;}
fn brickActive(brick:u32)->bool{return brick%5u!=4u;}
fn brickHasUnclippedWorldGeometry(brick:u32)->bool{_=brick;return mode<4u;}
fn activityRecord(brick:u32)->u32{return brick*2u;}
fn cellOpenFraction(cell:u32)->f32{return f32(cell%5u)*.25;}
fn templateBrickCellRange(brick:u32,r:u32)->vec2u{
  let origin=cm12WorldLeafCoordinate(brick)*8;
  var valid=vec3u(r);
  if(!brickHasUnclippedWorldGeometry(brick)){
    valid=vec3u(min(vec3i(p.dimensions.xyz)-origin+vec3i(i32(8u/r)-1),vec3i(8))/i32(8u/r));
  }
  return vec2u(brick*512u,valid.x*valid.y*valid.z);
}
fn compactOwnerCellAt(q:vec3i)->vec3u{
  if(any(q<vec3i(-8))||any(q>=vec3i(16))){return vec3u(INVALID);}
  if(mode>=4u&&any(q>=vec3i(p.dimensions.xyz))){return vec3u(INVALID);}
  let b=vec3u((q+vec3i(8))/8);let brick=b.x+3u*(b.y+3u*b.z);
  let r=1u<<((brick+mode)%4u);let scale=8u/r;
  let origin=cm12WorldLeafCoordinate(brick)*8;
  var valid=vec3u(r);
  if(mode>=4u){valid=vec3u(min(vec3i(p.dimensions.xyz)-origin+vec3i(i32(scale)-1),vec3i(8))/i32(scale));}
  let local=vec3u((q-origin)/i32(scale));
  let offset=local.x+valid.x*(local.y+valid.y*local.z);
  return vec3u(brick*512u+offset,brick,r);
}
${reference}
${restriction}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id)gid:vec3u){
  if(gid.x>=arrayLength(&result)){return;}
  let q=queries[2u*gid.x];mode=u32(queries[2u*gid.x+1u].x);
  p=Params(vec4u(13,10,9,0));
  result[gid.x]=vec2u(bitcast<u32>(referenceRestriction(q.xyz,q.w,0u)),
    bitcast<u32>(restrictedPresentationDensityAt(q.xyz,q.w,0u)));
}` });
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
    const buffers: GPUBuffer[] = [];
    const upload = (data: ArrayBufferView) => {
      const buffer = device!.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device!.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
      buffers.push(buffer); return buffer;
    };
    const input = upload(new Int32Array(queries));
    const state = upload(Float32Array.from({ length: 27 * 512 }, (_, i) => ((i * 17) % 127) / 126));
    const activity = upload(Uint32Array.from({ length: 54 }, (_, i) => i % 6 === 5 ? 0 : 64));
    const output = device.createBuffer({ size: count * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: output.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    buffers.push(output, readback);
    const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries:
      [input, state, activity, output].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(count / 64)); pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, output.size);
    device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
    const values = new Uint32Array(readback.getMappedRange());
    let nonzero = 0;
    for (let i = 0; i < count; i++) {
      assert.equal(values[2 * i + 1], values[2 * i], `query ${queries.slice(i * 8, i * 8 + 5)}`);
      if (values[2 * i]) nonzero++;
    }
    assert.ok(nonzero > count / 4, "the oracle must exercise represented liquid");
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ queries: count, nonzero, byteExact: true }));
    readback.unmap(); for (const buffer of buffers) buffer.destroy();
  } finally {
    device?.destroy(); await releaseWebGPUExclusiveLock(); if (gpu) live.delete(gpu);
  }
});
