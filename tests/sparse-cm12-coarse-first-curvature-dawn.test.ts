import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

const source = readFileSync(new URL("../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
const fn = (name: string) => {
  const result = source.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n}`))?.[0];
  assert.ok(result); return result;
};
const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;
dawnTest("coarse-first normals preserve a flat interface across 8:4:2:1 cell widths", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "coarse-first-curvature");
  let gpu: GPU | undefined, device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const errors: string[] = [];
    device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    // An exact dyadic lookup fixture; values are analytic control-volume
    // averages. No fluid update, surface renderer, or pre-staged TEI cache.
    const shader = `
const INVALID=0xffffffffu;const BRICK_FINE_RESOLUTION=8u;
struct Params{dimensions:vec4u}
const p=Params(vec4u(64u));
@group(0)@binding(0)var<storage,read>state:array<f32>;
@group(0)@binding(1)var<storage,read>queries:array<vec4f>;
@group(0)@binding(2)var<storage,read_write>result:array<vec4f>;
@group(0)@binding(3)var<uniform>axis:vec4u;
fn cm12WorldFineLower()->vec3f{return vec3f(0);}
fn cm12WorldFineUpper()->vec3f{return vec3f(64);}
fn destinationDensity()->u32{return 0u;}
fn brickActive(b:u32)->bool{_=b;return true;}
fn brickSpan(b:u32)->u32{_=b;return 1u;}
fn brickHasUnclippedWorldGeometry(b:u32)->bool{_=b;return false;}
fn cellOpenFraction(c:u32)->f32{_=c;return select(1.0,0.5,axis.y!=0u);}
fn scaleAt(q:vec3i)->u32{return 8u>>u32(clamp(q[(axis.x+1u)%3u]/16,0,3));}
fn cm12WorldLeafCoordinate(b:u32)->vec3i{return vec3i(i32(b%8u),i32((b/8u)%8u),i32(b/64u));}
fn cellBrick(c:u32)->u32{return c/512u;}
fn acceptedBrickResolution(b:u32)->u32{return 8u/scaleAt(8*cm12WorldLeafCoordinate(b));}
fn templateBrickCellRange(b:u32,r:u32)->vec2u{return vec2u(512u*b,r*r*r);}
fn compactOwnerCellAt(q:vec3i)->vec3u{
  if(any(q<vec3i(0))||any(q>=vec3i(64))){return vec3u(INVALID);}
  let scale=scaleAt(q);let b=vec3u(q)/8u;let brick=b.x+8u*(b.y+8u*b.z);
  let local=(vec3u(q)%8u)/scale;let r=8u/scale;
  let cell=512u*brick+local.x+r*(local.y+r*local.z);
  return vec3u(cell,brick,r);
}
fn ownerCellAt(q:vec3i)->u32{return compactOwnerCellAt(q).x;}
${fn("mirrorSharpeningSampleToWorld")}
${fn("coarseFirstTileMass")}
${fn("coarseFirstDensity")}
${fn("coarseFirstNormal")}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id)id:vec3u){
  if(id.x>=arrayLength(&queries)){return;}
  let q=queries[id.x];result[id.x]=vec4f(coarseFirstNormal(q.xyz,q.w),1);
}
// Independent finest-volume oracle: no contained-leaf addressing shortcut.
@compute @workgroup_size(64)
fn densityOracle(@builtin(global_invocation_id)id:vec3u){
  if(id.x>=arrayLength(&queries)){return;}
  let q=queries[id.x];let h=i32(q.w);
  let lower=vec3i(clamp(q.xyz-vec3f(0.5*q.w),vec3f(0),vec3f(f32(64-h))));
  var sum=0.0;
  for(var z=0;z<h;z++){for(var y=0;y<h;y++){for(var x=0;x<h;x++){
    let owner=compactOwnerCellAt(lower+vec3i(x,y,z));
    if(owner.x!=INVALID){sum+=state[owner.x];}
  }}}
  result[id.x]=vec4f(coarseFirstDensity(q.xyz,q.w),sum/f32(h*h*h),0,0);
}`;
    const shaderModule = device.createShaderModule({ code: shader });
    const compilation = await shaderModule.getCompilationInfo();
    assert.deepEqual(compilation.messages.filter(message => message.type === "error"), []);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto",
      compute: { module: shaderModule, entryPoint: "main" } });
    const data = device.createBuffer({ size: 64 ** 3 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const positions = device.createBuffer({ size: 64 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const output = device.createBuffer({ size: 64 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const uniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const readback = device.createBuffer({ size: output.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries:
      [data, positions, output, uniform].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    let maximumError = 0;
    for (let fixture = 0; fixture < 6; fixture++) {
      const axis = fixture % 3, cut = fixture >= 3;
      const field = new Float32Array(64 ** 3), queries = new Float32Array(64 * 4);
      for (let z = 0; z < 64; z++) for (let y = 0; y < 64; y++) for (let x = 0; x < 64; x++) {
        const q = [x, y, z], width = 8 >> Math.floor(q[(axis + 1) % 3]! / 16);
        const lower = Math.floor(q[axis]! / width) * width;
        const b = q.map(v => Math.floor(v / 8));
        const local = q.map(v => Math.floor((v % 8) / width));
        const r = 8 / width;
        field[512 * (b[0]! + 8 * (b[1]! + 8 * b[2]!))
          + local[0]! + r * (local[1]! + r * local[2]!)]
          = Math.max(0, Math.min(1, (20.37 - lower) / width));
      }
      for (let sample = 0; sample < 64; sample++) {
        const width = 8 >> Math.floor(sample / 16), q = [28, 28, 28];
        q[axis] = (Math.floor(20.37 / width) + .5) * width;
        q[(axis + 1) % 3] = (Math.floor(sample / width) + .5) * width;
        queries.set([...q, width], sample * 4);
      }
      device.queue.writeBuffer(data, 0, field); device.queue.writeBuffer(positions, 0, queries);
      device.queue.writeBuffer(uniform, 0, new Uint32Array([axis, Number(cut), 0, 0]));
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
      encoder.copyBufferToBuffer(output, 0, readback, 0, output.size); device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const normals = new Float32Array(readback.getMappedRange());
      for (let sample = 0; sample < 64; sample++) for (let component = 0; component < 3; component++) {
        maximumError = Math.max(maximumError, Math.abs(normals[4 * sample + component]! - (component === axis && !cut ? -1 : 0)));
      }
      readback.unmap();
    }
    const restrictionPipeline = await device.createComputePipelineAsync({ layout: "auto",
      compute: { module: shaderModule, entryPoint: "densityOracle" } });
    const restrictionGroup = device.createBindGroup({ layout: restrictionPipeline.getBindGroupLayout(0),
      entries: [data, positions, output, uniform].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    // Non-planar donor data exposes wrong native strides hidden by flat fields.
    // Dyadic fractions also make the finest-volume oracle exact in f32 here.
    const varied = Float32Array.from({ length: 64 ** 3 }, (_, i) => ((i * 73 + (i >>> 9) * 19) % 256) / 256);
    device.queue.writeBuffer(data, 0, varied);
    let maximumRestrictionError = 0;
    for (let normalAxis = 0; normalAxis < 3; normalAxis++) {
      const restrictionQueries = new Float32Array(64 * 4);
      for (let i = 0; i < 64; i++) {
        const h = 1 << (i % 5);
        const lower = [(i * 7) % 64, (i * 13) % 64, (i * 23) % 64]
          .map(v => Math.floor(v / h) * h);
        restrictionQueries.set([...lower.map(v => v + h / 2), h], i * 4);
      }
      device.queue.writeBuffer(positions, 0, restrictionQueries);
      device.queue.writeBuffer(uniform, 0, new Uint32Array([normalAxis, 0, 0, 0]));
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(restrictionPipeline); pass.setBindGroup(0, restrictionGroup); pass.dispatchWorkgroups(1); pass.end();
      encoder.copyBufferToBuffer(output, 0, readback, 0, output.size); device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const densities = new Float32Array(readback.getMappedRange());
      for (let i = 0; i < 64; i++) maximumRestrictionError = Math.max(maximumRestrictionError,
        Math.abs(densities[4 * i]! - densities[4 * i + 1]!));
      readback.unmap();
    }
    assert.equal(maximumRestrictionError, 0, "native donor strides must match finest-volume restriction exactly");
    assert.deepEqual(errors, []);
    assert.ok(maximumError < 1e-6, `flat surfaces must have constant normals across cell widths: ${maximumError}`);
    console.log({ maximumError, maximumRestrictionError });
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
