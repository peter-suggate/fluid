/** Production window-address and vertex-slot microbenchmark: ascending valid
 * plane subsets versus scanning all seven masks. No simulation work.
 * node --import tsx tools/probe-uniform-vertex-subsets.ts
 */
import assert from 'node:assert/strict';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {uniformLatticeAddressWGSL} from '../lib/methods/uniform/uniform-lattice-address';
import {uniformWindowAddressWGSL,uniformWindowDerivedWords} from '../lib/methods/uniform/uniform-window-address';
import {createProcessRetainedDawnGPU,paceDawnEventPolling} from '../lib/harness/node-dawn-provider';
const original=uniformLatticeAddressWGSL({constant:true,paged:{blocks:'42u'}});
const needle=` for(var k=1u;k<8u;k++){
  let step=vec3<bool>((k&1u)!=0u,(k&2u)!=0u,(k&4u)!=0u);
  if(any(step&!plane)){continue;}`;
const replacement=` let faces=dot(vec3u(plane),vec3u(1u,2u,4u));
 for(var k=(0u-faces)&faces;k!=0u;k=(k-faces)&faces){
  let step=vec3<bool>((k&1u)!=0u,(k&2u)!=0u,(k&4u)!=0u);`;
assert.equal(original.split(needle).length,3);
const candidate=original.replaceAll(needle,replacement);
paceDawnEventPolling();const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']),adapter=await gpu.requestAdapter();assert.ok(adapter);
const device=await adapter.requestDevice({requiredFeatures:['timestamp-query']});const errors:string[]=[];
device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);});
const count=262144,rounds=11,launches=16;
const make=(size:number,usage:number)=>device.createBuffer({size,usage});
const directory=make(4*(64**3+32+42),GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),points=make(16*count,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),output=make(32*count,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
const code=(address:string)=>`
@group(0) @binding(0) var<storage,read> directory:array<u32>;
@group(0) @binding(1) var<storage,read> points:array<vec4i>;
@group(0) @binding(2) var<storage,read_write> output:array<vec4u>;
${uniformWindowAddressWGSL('directory','0u','42u')}
${address}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3u){let i=gid.x;
 let p=points[i];output[2u*i]=umVertexSlot(p.xyz);output[2u*i+1u]=umWideVertexSlot(p.xyz>>vec3u(2u));
}`;
const layout=device.createBindGroupLayout({entries:[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===2?'storage':'read-only-storage'}}))});
const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
const pipelines=await Promise.all([original,candidate].map(async address=>{const module=device.createShaderModule({code:code(address)});const info=await module.getCompilationInfo();assert.deepEqual(info.messages.filter(m=>m.type==='error'),[]);return device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint:'main'}});}));
const group=device.createBindGroup({layout,entries:[directory,points,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
for(const fixture of ['resident-interior','missing-interior','page-faces','page-edges','page-corners','mixed-boundaries'] as const){
 for(const policy of [0,1,2,3]){
 const frame=policy===0?[0,0,0]:[-2,1,-1],words=new Uint32Array(64**3+32+42).fill(0xffffffff);
 const index=(x:number,y:number,z:number)=>(x&63)|((y&63)<<6)|((z&63)<<12);
 // Permuted block residency, AIR holes, explicit SOLID words and stale slots.
 for(let z=0;z<2;z++)for(let y=0;y<3;y++)for(let x=0;x<4;x++){
  const n=x+4*(y+3*z);words[index(x,y,z)]=n%5===0?0:1+(n*17)%40;
 }
 words[index(1,1,1)]=0xffffffff;words[index(2,1,1)]=0xfffffffe;
 const header=64**3;words.set(new Uint32Array(new Int32Array(frame).buffer),header);words[header+12]=policy;
 words.set([0,0,0],header+13);words.set([4,3,2],header+16);words.set(uniformWindowDerivedWords(policy,[0,0,0],[4,3,2]),header+24);
 device.queue.writeBuffer(directory,0,words);
 const input=new Int32Array(4*count);
 for(let i=0;i<count;i++){
  let x=1+i%3,y=(i>>2)%3,z=(i>>4)%2;
  let local=[1+(i*13%31),1+(i*7%31),1+(i*19%31)];
  if(fixture==='resident-interior'){x=1;y=0;z=0;}
  if(fixture==='missing-interior'){x=0;y=0;z=0;}
  if(fixture==='page-faces')local[i%3]=0;
  if(fixture==='page-edges'){local[i%3]=0;local[(i+1)%3]=0;}
  if(fixture==='page-corners')local=[0,0,0];
  if(fixture==='mixed-boundaries'){x=(i%9)-2;y=((i>>4)%7)-2;z=((i>>8)%6)-2;local=[i%3===0?0:local[0]!,i%5===0?0:local[1]!,i%7===0?0:local[2]!];}
  input.set([32*(x-frame[0]!)+local[0]!,32*(y-frame[1]!)+local[1]!,32*(z-frame[2]!)+local[2]!,0],4*i);
 }
 device.queue.writeBuffer(points,0,input);
 const results=[0,1].map(()=>make(output.size,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ));
 const queries=device.createQuerySet({type:'timestamp',count:4*rounds}),resolved=make(32*rounds,GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC),times=make(32*rounds,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
 const e=device.createCommandEncoder();
 for(let arm=0;arm<2;arm++){const p=e.beginComputePass();p.setPipeline(pipelines[arm]!);p.setBindGroup(0,group);p.dispatchWorkgroups(count/64);p.end();e.copyBufferToBuffer(output,0,results[arm]!,0,output.size);}
 for(let r=0;r<rounds;r++)for(let k=0;k<2;k++){const arm=(r+k)%2,index=4*r+2*k;const p=e.beginComputePass({timestampWrites:{querySet:queries,beginningOfPassWriteIndex:index,endOfPassWriteIndex:index+1}});p.setPipeline(pipelines[arm]!);p.setBindGroup(0,group);for(let n=0;n<launches;n++)p.dispatchWorkgroups(count/64);p.end();}
 e.resolveQuerySet(queries,0,4*rounds,resolved,0);e.copyBufferToBuffer(resolved,0,times,0,32*rounds);device.queue.submit([e.finish()]);await Promise.all([...results.map(b=>b.mapAsync(GPUMapMode.READ)),times.mapAsync(GPUMapMode.READ)]);
 const a=new Uint32Array(results[0]!.getMappedRange()),b=new Uint32Array(results[1]!.getMappedRange()),t=new BigUint64Array(times.getMappedRange());let mismatches=0;for(let i=0;i<a.length;i++)if(a[i]!==b[i])mismatches++;
 const ms:number[][]=[[],[]];for(let r=0;r<rounds;r++)for(let k=0;k<2;k++){const index=4*r+2*k;ms[(r+k)%2]!.push(Number(t[index+1]!-t[index]!)/1e6/launches);}
 const median=ms.map(v=>v.slice().sort((a,b)=>a-b)[Math.floor(v.length/2)]!);console.log(JSON.stringify({fixture,policy,points:count,mismatches,baseline_ms:median[0],candidate_ms:median[1],ratio:median[1]!/median[0]!,timings:ms,errors}));assert.equal(mismatches,0);assert.deepEqual(errors,[]);
 for(const b of [...results,resolved,times])b.destroy();queries.destroy();
 }
}
for(const b of [directory,points,output])b.destroy();device.destroy();
