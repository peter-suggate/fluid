/** Native GPU regression: vertex sampling cells must stay inside a closed
 * tank at every mixed low/high edge and corner, including inner page seams.
 * node --import tsx tools/probe-vertex-wall-cells.ts
 */
import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { managedGPUDevice } from '../lib/core/gpu-compilation-manager';
import { uniformLatticeAddressWGSL, UNIFORM_PAGE_TABLE_WORDS } from '../lib/methods/uniform/uniform-lattice-address';
import { uniformPagePoolLayout, uniformPagePoolSeed, UNIFORM_PAGE_POOL_KIND } from '../lib/methods/uniform/uniform-page-pool';
const pages=[];
for(let z=0;z<2;z++)for(let y=0;y<2;y++)for(let x=0;x<2;x++)pages.push({x,y,z,block:1+x+2*y+4*z});
const seed=uniformPagePoolSeed(uniformPagePoolLayout({blocks:10,idleSteps:2}),pages,()=>UNIFORM_PAGE_POOL_KIND.solid);
const tail=new Uint32Array(UNIFORM_PAGE_TABLE_WORDS+10);tail.set(seed.table);tail.set(seed.blocks,UNIFORM_PAGE_TABLE_WORDS);
const count=65**3,bytes=count*32;
const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
const adapter=await createProcessRetainedDawnGPU(dawn,['backend=metal']).requestAdapter();assert.ok(adapter);
const device=managedGPUDevice(await adapter.requestDevice(),{requireWorkerRealm:false});
const errors:string[]=[];device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);});
const input=device.createBuffer({size:tail.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
const output=device.createBuffer({size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
const read=device.createBuffer({size:bytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
try{
 const module=device.createShaderModule({label:'Vertex wall sampling cells',code:/* wgsl */`
@group(0) @binding(0) var<storage,read> words:array<u32>;
@group(0) @binding(1) var<storage,read_write> result:array<vec4i>;
fn umPageWord(p:vec3i)->u32{let q=vec3u(p)&vec3u(63u);return words[q.x|(q.y<<6u)|(q.z<<12u)];}
fn umBlockWord(b:u32)->u32{return words[${UNIFORM_PAGE_TABLE_WORDS}u+b];}
${uniformLatticeAddressWGSL({constant:true,paged:{blocks:'10u'}})}
@compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=${count}u){return;}
 let p=vec3i(vec3u(i%65u,(i/65u)%65u,i/4225u));
 result[2u*i]=vec4i(umVertexCellAbove(p),0);result[2u*i+1u]=vec4i(umVertexCellBelow(p),0);
}`});
 const messages=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');assert.deepEqual(messages.map(m=>m.message),[]);
 const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'probe'}});
 const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:input}},{binding:1,resource:{buffer:output}}]});
 device.queue.writeBuffer(input,0,tail);
 const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(count/64));pass.end();
 encoder.copyBufferToBuffer(output,0,read,0,bytes);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
 const got=new Int32Array(read.getMappedRange());let failures=0;const examples:unknown[]=[];
 for(let i=0;i<count;i++){
  const p=[i%65,Math.floor(i/65)%65,Math.floor(i/4225)];
  for(let side=0;side<2;side++){
   const expected=p.map(v=>Math.max(0,Math.min(63,v-side))),actual=Array.from(got.subarray(8*i+4*side,8*i+4*side+3));
   if(expected.some((v,a)=>v!==actual[a])){failures++;if(examples.length<8)examples.push({p,side:side?'below':'above',expected,actual});}
  }
 }
 read.unmap();console.log(JSON.stringify({vertices:count,failures,examples},null,2));
 assert.deepEqual(errors,[]);assert.equal(failures,0,'Every sampling cell must match the closed-tank clamp');
}finally{input.destroy();output.destroy();read.destroy();device.destroy();}
