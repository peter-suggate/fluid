import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

const source = readFileSync(new URL("../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
const production = (name: string) => {
  const code = source.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n}`))?.[0];
  assert.ok(code, name); return code;
};
const live = new Set<GPU>();
const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;

dawnTest("fine waterline retains volume when its reconstruction bracket reaches the tank lid", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "boundary-waterline");
  let gpu: GPU | undefined, device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu!);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const errors: string[] = [];
    device.addEventListener("uncapturederror", e => errors.push(e.error.message));
    // Sweep every interface phase with at least one full air cell below the
    // lid. Include the same phases far from the boundary as an interior control.
    const heights = [8, 28, 29, 30].flatMap(base => Array.from({ length: 129 }, (_, i) => base + i / 128));
    const density = Float32Array.from(heights.flatMap(height =>
      Array.from({ length: 32 }, (_, y) => Math.max(0, Math.min(1, height - y)))));
    const shader = device.createShaderModule({ code: `
struct Params { frame:vec4f, dimensions:vec4u }
const p=Params(vec4f(0,.025,0,0),vec4u(1,32,1,0));
const CM12_LIQUID_ISOVALUE:f32=.5;const INVALID:u32=0xffffffffu;
const BRICK_FINE_RESOLUTION=8u;
@group(0)@binding(0)var<storage,read>state:array<f32>;
@group(0)@binding(1)var<storage,read>heights:array<f32>;
@group(0)@binding(2)var<storage,read_write>result:array<f32>;
var<private>base:u32;
fn cellOpenFraction(cell:u32)->f32{_=cell;return 1.0;}
fn cellMinimumWidth(cell:u32)->f32{_=cell;return 1.0;}
fn cellBrick(cell:u32)->u32{_=cell;return 0u;}
fn brickHasUnclippedWorldGeometry(brick:u32)->bool{_=brick;return false;}
fn cm12WorldOwnerAt(q:vec3i)->u32{_=q;return INVALID;}
fn cm12WorldFloorToSpan(q:i32,span:i32)->i32{return i32(floor(f32(q)/f32(span)))*span;}
fn cellCenter(cell:u32)->vec3f{return vec3f(.5,f32(cell%32u)+.5,.5);}
fn cm12SolidVoxelFractionQ8(q:vec3i)->u32{_=q;return 0u;}
fn compactOwnerCellAt(q:vec3i)->vec3u{return vec3u(base+u32(q.y),0u,8u);}
fn brickActive(brick:u32)->bool{_=brick;return true;}
${production("presentationResolvedFineColumnPhi")}
${production("presentationCanonicalCoarseCoordinate")}
${production("presentationPhiAt")}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id)gid:vec3u){
 let i=gid.x;if(i>=arrayLength(&heights)){return;}base=32u*i;
 let lower=u32(floor(heights[i]-.5));
 let lo=presentationPhiAt(base+lower,0u);let hi=presentationPhiAt(base+lower+1u,0u);
 result[i]=f32(lower)+.5-lo/(hi-lo);
}` });
    assert.deepEqual((await shader.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module: shader, entryPoint: "main" } });
    const inputs = [density, Float32Array.from(heights)].map(data => {
      const buffer = device!.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device!.queue.writeBuffer(buffer,0,data);return buffer;
    });
    const result = device.createBuffer({ size: heights.length*4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: result.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [...inputs,result].map((buffer,binding) => ({binding,resource:{buffer}})) });
    const encoder = device.createCommandEncoder();const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(heights.length/64));pass.end();
    encoder.copyBufferToBuffer(result,0,readback,0,result.size);device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);const measured = new Float32Array(readback.getMappedRange()).slice();
    readback.unmap();readback.destroy();result.destroy();inputs.forEach(buffer => buffer.destroy());
    for (const [i,height] of heights.entries()) assert.ok(Math.abs(measured[i]!-height)<4e-6,
      `height ${height}: reconstructed ${measured[i]} cells`);
    assert.deepEqual(errors,[]);
  } finally { device?.destroy();await releaseWebGPUExclusiveLock();if(gpu)live.delete(gpu); }
});
