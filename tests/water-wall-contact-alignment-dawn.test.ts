import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { globalFineCubeContourWGSL, globalFineDirectSharpPatchWGSL } from "../lib/core/webgpu-water-global-fine-tetra";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

const live = new Set<GPU>();
const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;

dawnTest("wall contact edges coincide with free-surface coordinates at every sample scale", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "wall-contact-alignment");
  let gpu: GPU | undefined, device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu!);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const errors: string[] = [];
    device.addEventListener("uncapturederror", e => errors.push(e.error.message));
    // Nodal unit samples, then cell-centred scales 1, 2, 4, 8, 16.
    const fixtures = Array.from({ length: 6 }, (_, code) => [0, 1, 2].flatMap(axis =>
      [0, 1].map(side => ({ code, axis, side })))).flat();
    const shader = device.createShaderModule({ code: `
@group(0)@binding(0)var<storage,read_write>result:array<vec4f>;
var<private>slot:u32;
${globalFineCubeContourWGSL}
fn tri(cursor:ptr<function,u32>,a:vec3f,b:vec3f,c:vec3f,n:vec3f,filterEnabled:bool){
  _=n;_=filterEnabled;let i=slot+1u+*cursor;
  result[i]=vec4f(a,1);result[i+1u]=vec4f(b,1);result[i+2u]=vec4f(c,1);*cursor+=3u;
}
${globalFineDirectSharpPatchWGSL}
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id)gid:vec3u){
  let code=gid.x/6u;let axis=(gid.x%6u)/2u;let side=gid.x%2u;slot=gid.x*64u;
  let tangent=select(0u,1u,axis==0u);
  var values:array<f32,8>;for(var i=0u;i<8u;i+=1u){values[i]=1.-CONTOUR_CORNERS[i][tangent];}
  let descriptor=224u+side+(code<<8u)+(axis<<14u);var cursor=0u;
  for(var lane=0u;lane<6u;lane+=1u){emitWallLane(&cursor,vec3f(4,6,8),descriptor,values,lane);}
  result[slot]=vec4f(f32(cursor),0,0,0);
}` });
    assert.deepEqual((await shader.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module: shader, entryPoint: "main" } });
    const output = device.createBuffer({ size: fixtures.length*64*16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: output.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: output } }] });
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(fixtures.length);pass.end();
    encoder.copyBufferToBuffer(output,0,readback,0,output.size);device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(readback.getMappedRange()).slice();readback.unmap();
    for (const [index, { code, axis, side }] of fixtures.entries()) {
      const count = data[index*256]!; assert.equal(count,6);
      const scale = 2**Math.max(0,code-1), tangent = axis===0 ? 1 : 0;
      const other = [0,1,2].find(a => a!==axis && a!==tangent)!;
      const points = Array.from({ length: count }, (_, i) => Array.from(data.slice(index*256+4+i*4,index*256+7+i*4)));
      const base = [4,6,8];
      for (const point of points) assert.equal(point[axis],base[axis]!+side+.5);
      // A linear scalar from 1 to 0 crosses at exactly half of the free-surface cube.
      for (const a of [tangent,other]) {
        const shift = code===0 ? .5 : 0;
        assert.equal(Math.min(...points.map(p => p[a]!)),base[a]!+shift,JSON.stringify({code,axis,side,a}));
        assert.equal(Math.max(...points.map(p => p[a]!)),base[a]!+shift+scale*(a===tangent?.5:1));
      }
    }
    output.destroy();readback.destroy();assert.deepEqual(errors,[]);
  } finally { device?.destroy();await releaseWebGPUExclusiveLock();if(gpu)live.delete(gpu); }
});
