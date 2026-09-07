import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

const source = readFileSync(new URL("../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
const productionFunction = (name: string) => {
  const match = source.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n}`));
  assert.ok(match, name); return match[0];
};
const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;
dawnTest("mixed face support includes every incident child and retains physical scale", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "face-support-symmetry");
  let gpu: GPU | undefined, device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const shader = device.createShaderModule({ code: `
@group(0)@binding(0)var<storage,read>state:array<f32>;
@group(0)@binding(1)var<storage,read_write>output:array<vec4f>;
const CM12_LIQUID_ISOVALUE=0.5;
struct TransportFaceSupport { width:f32, extended:bool, liquid:bool }
fn sourceDensity()->u32{return 0u;}
fn rowTermOffset(row:u32)->u32{return 5u*row;}
fn rowTermCount(row:u32)->u32{_=row;return 5u;}
fn termCell(term:u32)->u32{return term;}
fn cellMinimumWidth(cell:u32)->f32{return select(1.0,2.0,cell%5u==0u||cell>=20u);}
fn cm12ExtendedCellSelected(cell:u32)->bool{return state[cell]>0.0;}
fn transportSourceSamplingSpans(cell:u32,direct:bool)->vec3f{
  _=direct;
  // Interpolation now locates the actual dual cell; its source scale no
  // longer shrinks onto a virtual fine lattice near tangential refinement.
  return vec3f(cellMinimumWidth(cell));
}
${productionFunction("transportFaceSupport")}
${productionFunction("transportFaceSamplingSpans")}
@compute @workgroup_size(6)
fn main(@builtin(global_invocation_id)id:vec3u){
  let support=transportFaceSupport(id.x);
  output[id.x]=vec4f(support.width,f32(support.extended),f32(support.liquid),
    transportFaceSamplingSpans(id.x,support.width).x);
}` });
    assert.deepEqual((await shader.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module: shader, entryPoint: "main" } });
    const input = device.createBuffer({ size: 30*4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const values = new Float32Array(30);
    // Reflections permute the four fine children. One child supplies liquid
    // and extension; a patch-centre point query can miss it in three cases.
    for (let row=0;row<4;row++) values[5*row+1+row]=0.75;
    values[20]=values[25]=0.75;
    device.queue.writeBuffer(input,0,values);
    const output = device.createBuffer({ size: 6*16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: output.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [input,output].map((buffer,binding) => ({binding,resource:{buffer}})) });
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();
    encoder.copyBufferToBuffer(output,0,readback,0,output.size);device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const result = Array.from(new Float32Array(readback.getMappedRange()));
    for(let row=0;row<6;row++) assert.deepEqual(result.slice(4*row,4*row+4),[2,1,1,row>=4?2:1],`row ${row}`);
    readback.unmap();input.destroy();output.destroy();readback.destroy();
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); assert.ok(gpu); }
});
