/** Exact policy-derived point bounds versus the directory implementation.
 * Exercises every policy, moved frames, AIR/resident/SOLID directory entries,
 * page faces and points in SOLID pages; all six bound components compare bitwise. */
import assert from 'node:assert/strict';
import {experimentalWorldBounds} from './experimental-world-bounds';
import {resolve} from 'node:path';import {pathToFileURL} from 'node:url';
import {uniformWindowAddressWGSL,uniformWindowDerivedWords} from '../lib/methods/uniform/uniform-window-address';
import {uniformLatticeAddress,uniformLatticeAddressWGSL} from '../lib/methods/uniform/uniform-lattice-address';
import {createProcessRetainedDawnGPU,paceDawnEventPolling} from '../lib/harness/node-dawn-provider';
paceDawnEventPolling();const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']),adapter=await gpu.requestAdapter();assert.ok(adapter);
const device=await adapter.requestDevice({requiredFeatures:['timestamp-query']});const errors:string[]=[];
device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
const a=uniformLatticeAddress([96,64,64],false,{blocks:20,permute:7,air:[[1,1,0]]}),N=262144;
const make=(size:number,usage:number)=>device.createBuffer({size,usage});
const pages=make(a.seed.tail.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),points=make(N*16,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),result=make(N*64,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC),read=make(N*64,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
const prefix=`@group(0) @binding(0) var<storage,read> pages:array<u32>;
@group(0) @binding(1) var<storage,read> points:array<vec4f>;
@group(0) @binding(2) var<storage,read_write> result:array<vec4f>;
${uniformWindowAddressWGSL('pages','0u',`${a.blocks}u`)}\n`;
const main=`@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3u){let i=gid.x;let p=points[i];let b=umPointBounds(p.xyz,p.w);result[4u*i]=vec4f(b.lo,0);result[4u*i+1u]=vec4f(b.hi,0);result[4u*i+2u]=vec4f(vec3f(umVertexCellAbove(vec3i(p.xyz))),0);result[4u*i+3u]=vec4f(vec3f(umVertexCellBelow(vec3i(p.xyz))),0);}`;
const layout=device.createBindGroupLayout({entries:[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:(binding===2?'storage':'read-only-storage') as GPUBufferBindingType}}))});
const pipelines=[];for(const worldBounds of [false,true]){const module=device.createShaderModule({code:prefix+(worldBounds?experimentalWorldBounds(uniformLatticeAddressWGSL({constant:true,paged:{blocks:`${a.blocks}u`}}),'pages','0u'):uniformLatticeAddressWGSL({constant:true,paged:{blocks:`${a.blocks}u`}}))+main});assert.deepEqual((await module.getCompilationInfo()).messages.filter(x=>x.type==='error'),[]);pipelines.push(await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint:'main'}}));}
const group=device.createBindGroup({layout,entries:[pages,points,result].map((buffer,binding)=>({binding,resource:{buffer}}))});
const tail=a.seed.tail.slice(),H=64**3,script=new Float32Array(N*4),frames=[[0,0,0],[7,-11,13],[-19,23,-5]],positions=[0,0.125,7.875,8,15.875,16,23.875,24,31.875];
for(let policy=0;policy<=3;policy++)for(const frame of frames){
 tail[H+12]=policy;tail.set(new Uint32Array(new Int32Array(frame).buffer),H);tail.set(uniformWindowDerivedWords(policy,[0,0,0],[3,2,2]),H+24);
 // Under world policies a raw unallocated SOLID word is stale AIR; under
 // explicit policy 0 it is a real internal obstacle.
 tail[1|(1<<6)|(1<<12)]=0xffffffff;device.queue.writeBuffer(pages,0,tail);
 for(let i=0;i<N;i++){const world=[(i%7)-2,((i/7|0)%6)-2,((i/42|0)%6)-2];for(let k=0;k<3;k++)script[4*i+k]=32*(world[k]!-frame[k]!)+positions[(i*(k+1)+k)%positions.length]!;script[4*i+3]=[0,1,4,8,16][i%5]!;}device.queue.writeBuffer(points,0,script);
 const outputs:Uint32Array[]=[];for(const pipeline of pipelines){const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(N/64);p.end();e.copyBufferToBuffer(result,0,read,0,read.size);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);outputs.push(new Uint32Array(read.getMappedRange()).slice());read.unmap();}
 let mismatches=0;for(let i=0;i<outputs[0]!.length;i++)if(outputs[0]![i]!==outputs[1]![i]){if(mismatches++<3)console.error({policy,frame,index:i,point:[...script.subarray((i>>4)*4,(i>>4)*4+4)],baseline:outputs[0]![i],candidate:outputs[1]![i]});}
 console.log(JSON.stringify({policy,frame,points:N,mismatches}));assert.equal(mismatches,0);
}
assert.deepEqual(errors,[]);await device.queue.onSubmittedWorkDone();device.destroy();
