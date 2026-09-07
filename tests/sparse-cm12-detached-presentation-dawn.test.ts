/** Known failing diagnostic for the unresolved coarse detached-surface reconstruction. */
import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock } from '../lib/harness/webgpu-smoke-isolation';
const source=readFileSync(new URL('../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts',import.meta.url),'utf8');
const scalar=source.match(/fn presentationResolvedColumnPhi\([\s\S]*?\n}/)?.[0];assert.ok(scalar);
const live=new Set<GPU>();Object.assign(globalThis,{detachedPresentationGPUs:live});
(process.env.WEBGPU_NODE_MODULE?test:test.skip)('coarse surface reconstruction cannot connect a detached body to a floor apron',async()=>{
 await acquireWebGPUExclusiveLock('dawn-test','detached-presentation');let device:GPUDevice|undefined;
 try{
  const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);
  const gpu:GPU=dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND??'metal'}`]);live.add(gpu);
  const adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();
  const fixtures=[[1,1,.3,0,0],[1,1,.6,0,0],[1,0,.6,0,0],[1,.01,.6,0,0],[0,0,.6,0,0]];
  const shader=device.createShaderModule({code:`
struct Params{frame:vec4f}const p=Params(vec4f(0,.05,0,0));const CM12_LIQUID_ISOVALUE=.5;
@group(0)@binding(0)var<storage,read>rho:array<f32>;
@group(0)@binding(1)var<storage,read_write>result:array<f32>;
${scalar}
@compute @workgroup_size(1)fn main(){for(var i=0u;i<${fixtures.length}u;i++){
  var values:array<f32,5>;for(var j=0u;j<5u;j++){values[j]=rho[5u*i+j];}
  result[i]=presentationResolvedColumnPhi(values);
}}
`});
  assert.deepEqual((await shader.getCompilationInfo()).messages.filter(m=>m.type==='error'),[]);
  const input=device.createBuffer({size:fixtures.length*20,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(input,0,new Float32Array(fixtures.flat()));
  const output=device.createBuffer({size:fixtures.length*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const read=device.createBuffer({size:output.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module:shader,entryPoint:'main'}});
  const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[input,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
  const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();
  encoder.copyBufferToBuffer(output,0,read,0,read.size);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
  const result=[...new Float32Array(read.getMappedRange())];read.unmap();read.destroy();input.destroy();output.destroy();
  assert.ok(Math.abs(result[0]!-.01)<1e-7,'partially filled monotone column retains its exact waterline');
  assert.ok(Math.abs(result[1]!+.005)<1e-7,'mostly filled monotone column retains its exact waterline');
  assert.ok(Math.abs(result[2]!-result[4]!)<1e-7,'disconnected floor support cannot move the body surface');
  assert.ok(result[3]!<0,'a dilute gap must not erase the represented body centre');
 }finally{device?.destroy();live.clear();await releaseWebGPUExclusiveLock();}
});
