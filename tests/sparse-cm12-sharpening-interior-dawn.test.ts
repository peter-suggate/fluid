import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";

const source = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
const functions = ["sharpeningInteriorStrides", "initialSharpeningField"].map(name => {
  const body = source.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n}`))?.[0];
  assert.ok(body, name); return body;
}).join("\n");
const dawnModule = process.env.WEBGPU_NODE_MODULE;
(dawnModule ? test : test.skip)("interior sharpening matches eight-corner centred differences across widths and solid masks",
  async () => {
    await acquireWebGPUExclusiveLock("dawn-test", "sharpening-interior-gradient");
    let device: GPUDevice | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href);
      Object.assign(globalThis, dawn.globals);
      const gpu: GPU = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter(); assert.ok(adapter);
      device = await adapter.requestDevice();
      const count = 4 * 3 * 3 * 3;
      const code = /* wgsl */ `
const INVALID:u32=0xffffffffu;
struct Leaf{flags:u32,scale:u32,valid:vec3u,first:u32,count:u32}
var<private>width:u32;var<private>pattern:u32;var<private>mask:u32;
var<private>mode:u32;var<private>sourceCell:u32;
@group(0)@binding(0)var<storage,read_write>output:array<vec4f>;
fn cellBrick(c:u32)->u32{_=c;return 0u;}
fn brickHasUnclippedWorldGeometry(b:u32)->bool{_=b;return mode!=2u;}
fn acceptedTopologySlot()->u32{return 0u;}
fn cm12TeiLoadLeaf(slot:u32,brick:u32)->Leaf{_=slot;_=brick;
  return Leaf(0x80000000u,width,vec3u(5u),0u,125u);}
fn coordinate(c:u32)->vec3f{return vec3f(f32(c%5u),f32((c/5u)%5u),f32(c/25u));}
fn cellCenter(c:u32)->vec3f{return (coordinate(c)+vec3f(0.5))*f32(width);}
fn conditionedDensity(c:u32)->f32{
  let q=coordinate(c);
  if(pattern==0u){return 0.2+0.003*q.x+0.002*q.y+0.001*q.z;}
  if(pattern==1u){return 0.2+0.001*(q.x*q.x+q.y*q.z+2.0*q.z*q.z);}
  return 0.3-0.002*q.x+0.001*q.y*q.y-0.003*q.z;
}
fn cellTransportActive(c:u32)->bool{
  if(mask==1u){return c!=sourceCell-1u;}
  if(mask==2u){return c==sourceCell;}
  return true;
}
// Independent ordinary Cartesian trilinear interpolation oracle. It expands
// all eight corners, including zero-weight tangential donors.
fn sampleReference(position:vec3f)->f32{
  let q=clamp(position/f32(width)-vec3f(0.5),vec3f(0.0),vec3f(4.0));
  let base=vec3u(floor(q));let t=fract(q);var total=0.0;var visible=0.0;
  for(var corner=0u;corner<8u;corner+=1u){
    let upper=vec3u(corner&1u,(corner>>1u)&1u,(corner>>2u)&1u);
    let cell=min(base+upper,vec3u(4u));let id=cell.x+5u*(cell.y+5u*cell.z);
    if(!cellTransportActive(id)){continue;}
    let f=select(vec3f(1.0)-t,t,upper!=vec3u(0u));let weight=f.x*f.y*f.z;
    total+=weight*conditionedDensity(id);visible+=weight;
  }
  return total/max(visible,1e-9);
}
fn referenceField(position:vec3f,density:f32)->vec4f{
  var gradient=vec3f(0.0);
  for(var axis=0u;axis<3u;axis+=1u){var reach=vec3f(0.0);reach[axis]=0.5*f32(width);
    gradient[axis]=(sampleReference(position+reach)-sampleReference(position-reach))/f32(width);}
  return vec4f(density,gradient);
}
fn sampleSharpeningField(position:vec3f,density:f32)->vec4f{
  _=position;return vec4f(density,999.0,999.0,999.0);
}
${functions}
@compute @workgroup_size(1)fn main(@builtin(global_invocation_id)id:vec3u){
  width=1u<<(id.x%4u);pattern=(id.x/4u)%3u;mask=(id.x/12u)%3u;mode=id.x/36u;
  sourceCell=select(62u,60u,mode==1u);let density=conditionedDensity(sourceCell);
  output[2u*id.x]=initialSharpeningField(sourceCell,density);
  var expected=vec4f(density,999.0,999.0,999.0);
  if(mode==0u){expected=referenceField(cellCenter(sourceCell),density);}
  output[2u*id.x+1u]=expected;
}`;
      const module = device.createShaderModule({ code });
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
      const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
      const output = device.createBuffer({ size: count * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const read = device.createBuffer({ size: output.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: output } }] });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(count); pass.end();
      encoder.copyBufferToBuffer(output, 0, read, 0, output.size); device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const values = new Float32Array(read.getMappedRange());
      for (let c = 0; c < count; c++) for (let axis = 0; axis < 4; axis++) {
        assert.ok(Math.abs(values[8 * c + axis]! - values[8 * c + 4 + axis]!) < 1e-7,
          `case ${c}, component ${axis}: ${values[8 * c + axis]} != ${values[8 * c + 4 + axis]}`);
      }
      read.unmap(); read.destroy(); output.destroy();
    } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
  });
