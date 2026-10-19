/** Native check: reuse one pipeline across scene sizes, then read back each
 * scene's capacity and bounds to ensure cached code consumes fresh uniforms.
 * Run: node --import tsx tools/check-uniform-scene-cache.ts */
import assert from 'node:assert/strict';
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from '../lib/harness/node-dawn-provider';
import { managedGPUDevice, gpuCompilationManagerFor } from '../lib/core/gpu-compilation-manager';
import { uniformDetailModule, uniformDetailPipeline } from '../lib/methods/uniform/uniform-detail-fields';
import { uniformMixedTopologyWGSL } from '../lib/methods/uniform/uniform-mixed-topology.wgsl';
import { uniformLatticeAddress } from '../lib/methods/uniform/uniform-lattice-address';
import type { UniformMixedOwnership } from '../lib/methods/uniform/uniform-mixed-ownership';

const dawn = await import('webgpu') as NodeDawnProvider;
Object.assign(globalThis, dawn.globals);
const gpu = createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? 'metal'}`]);
const adapter = await gpu.requestAdapter();
assert.ok(adapter, 'WebGPU adapter required');
const device = managedGPUDevice(await adapter.requestDevice(), {requireWorkerRealm:false});
const errors:string[]=[];
device.addEventListener('uncapturederror',event=>{event.preventDefault();errors.push(event.error.message);});
let previous:GPUComputePipeline|undefined;
try {
 for (const [cells,openTop] of [[[32,32,32],false],[[96,64,32],true]] as const) {
  const address=uniformLatticeAddress(cells,openTop);
  const lattice={dimensions:cells,cellSize_m:[0.1,0.1,0.1]} as unknown as UniformMixedOwnership['capacity']['lattice'];
  const module=uniformDetailModule(device,{code:uniformMixedTopologyWGSL({lattice,tiles:address.tileKeys},0)+`
@compute @workgroup_size(1) fn main(){
 umSupport[0]=UM_TILES;umSupport[1]=u32(umWorldHigh().x);
 umSupport[2]=u32(umWorldHigh().y);umSupport[3]=UM_RESIDENCY;
}`});
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==='error'),[]);
  const ownership={fixedWorldGeometry:true,capacity:{address}} as UniformMixedOwnership;
  const pipeline=await uniformDetailPipeline(device,ownership,{layout:'auto',compute:{module,entryPoint:'main'}});
  if(previous)assert.equal(pipeline,previous);previous=pipeline;
  const counts=device.createBuffer({size:176,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  const output=device.createBuffer({size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const read=device.createBuffer({size:16,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  try {
   const data=new Uint32Array(44);data[7]=address.tileKeys;
   const at=address.seed.layout.tail.window;
   data.set(address.seed.tail.subarray(at,at+32),12);
   device.queue.writeBuffer(counts,0,data);
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:1,resource:{buffer:counts}},{binding:2,resource:{buffer:output}}]});
   const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
   pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();
   encoder.copyBufferToBuffer(output,0,read,0,16);device.queue.submit([encoder.finish()]);
   await read.mapAsync(GPUMapMode.READ);
   assert.deepEqual([...new Uint32Array(read.getMappedRange())],[address.tileKeys,data[39],data[40],9*address.tileKeys+28]);
   read.unmap();
  } finally {counts.destroy();output.destroy();read.destroy();}
 }
 assert.deepEqual(errors,[]);
 const stats=gpuCompilationManagerFor(device).snapshot().cacheStats;
 assert.equal(stats.compilationAttempts,1);assert.equal(stats.pipelineHits,1);
 console.log(JSON.stringify({passed:true,cacheStats:stats}));
} finally {device.destroy();}
