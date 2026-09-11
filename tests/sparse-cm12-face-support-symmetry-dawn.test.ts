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
dawnTest("mixed face support includes every incident child", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "face-support-symmetry");
  let gpu: GPU | undefined, device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    // One wet child must activate the coarse receiver under every reflected
    // child placement. The production gate reads the frozen cell flags.
    const shader = device.createShaderModule({ code: `
@group(0)@binding(0)var<storage,read_write>state:array<f32>;
const FACE_VELOCITY_SUPPORT=36u;
struct FaceVelocitySupport {velocity:vec3f,spans:vec3f,owner:bool,extended:bool,liquid:bool}
struct Params{refinementRegionControl:vec4u}
const p=Params(vec4u(0));
var<private>currentRow:u32;
fn destinationFaceVelocity()->u32{return 30u;}
fn rowCenter(row:u32)->vec3f{_=row;return vec3f(2,1,1);}
fn rowDistance(row:u32)->f32{_=row;return 2.0;}
fn rowArea(row:u32)->f32{_=row;return 4.0;}
fn rowAxis(row:u32)->u32{_=row;return 0u;}
fn rowOpenFraction(row:u32)->f32{_=row;return 1.0;}
fn rowSolidVelocity(row:u32)->f32{_=row;return 0.0;}
fn hasSolidBoundaries()->bool{return false;}
fn rowTermOffset(row:u32)->u32{return 5u*row;}
fn rowTermCount(row:u32)->u32{_=row;return 5u;}
fn rowTermRange(row:u32)->vec2u{let first=rowTermOffset(row);return vec2u(first,first+rowTermCount(row));}
fn termCell(term:u32)->u32{return term;}
fn cellMinimumWidth(cell:u32)->f32{return select(1.0,2.0,cell%5u==0u||cell>=20u);}
fn cellBrick(cell:u32)->u32{_=cell;return 0u;}
fn cachedRefinementPolicyTileScale(brick:u32)->u32{_=brick;return 1u;}
fn faceVelocitySupportAt(q:vec3i)->FaceVelocitySupport{
  let child=select(0u,1u+u32(q.y%2)+2u*u32(q.z%2),q.x>=2);
  let cell=5u*currentRow+child;let wet=state[cell]>.5;
  return FaceVelocitySupport(vec3f(3,0,0),vec3f(cellMinimumWidth(cell)),true,wet,wet);
}
fn traceFaceDeparture(q:vec3f)->vec3f{return q;}
fn traceFaceDepartureAtSpans(q:vec3f,w:vec3f)->vec3f{_=w;return q;}
fn sampleFaceVelocitySupport(q:vec3f)->vec3f{_=q;return vec3f(3,0,0);}
fn sampleFaceVelocitySupportAtSpans(q:vec3f,w:vec3f)->vec3f{_=w;return sampleFaceVelocitySupport(q);}
${productionFunction("finishTransportFaceRow")}
${productionFunction("prepareTransportFaceRow")}
@compute @workgroup_size(6)
fn main(@builtin(global_invocation_id)id:vec3u){
  currentRow=id.x;prepareTransportFaceRow(id.x);
}` });
    assert.deepEqual((await shader.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module: shader, entryPoint: "main" } });
    const input = device.createBuffer({ size: (36+4*30)*4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const values = new Float32Array(36+4*30);
    // Reflections permute the four fine children. One child supplies liquid
    // and extension; a patch-centre point query can miss it in three cases.
    for (let row=0;row<4;row++) values[5*row+1+row]=0.75;
    values[20]=values[25]=0.75;
    for(let cell=0;cell<30;cell++) values[36+4*cell+3]=1+(values[cell]!>0 ? 7 : 1)/8;
    device.queue.writeBuffer(input,0,values);
    const readback = device.createBuffer({ size: 6*4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{binding:0,resource:{buffer:input}}] });
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();
    encoder.copyBufferToBuffer(input,30*4,readback,0,6*4);device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const result = Array.from(new Float32Array(readback.getMappedRange()));
    for(let row=0;row<6;row++) assert.equal(result[row],3,`row ${row}: every wet child must activate the receiver`);
    readback.unmap();input.destroy();readback.destroy();
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); assert.ok(gpu); }
});
