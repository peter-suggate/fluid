/** Controlled address-only GPU benchmarks: identical coordinates in page interiors,
 * across resident seams, and against SOLID walls. No solver or changing physics.
 * Every candidate output must equal the production helper bit for bit.
 * node --import tsx tools/probe-uniform-address-cost.ts */
import assert from 'node:assert/strict';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {uniformLatticeAddressWGSL,uniformBlockNeighbourWGSL,uniformBlockNeighbourWords,uniformBlockNeighbourLayout,uniformLatticeAddress,UNIFORM_PAGE_TABLE_WORDS} from '../lib/methods/uniform/uniform-lattice-address';
import {createProcessRetainedDawnGPU,paceDawnEventPolling} from '../lib/harness/node-dawn-provider';
paceDawnEventPolling();
const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
const adapter=await createProcessRetainedDawnGPU(dawn,['backend=metal']).requestAdapter();assert.ok(adapter);
const device=await adapter.requestDevice({requiredFeatures:['timestamp-query']});
const errors:string[]=[];device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);});
const address=uniformLatticeAddress([96,96,96],false,{blocks:32,permute:7});
const layout=uniformBlockNeighbourLayout(address.blocks), words=new Uint32Array(layout.end);
words.set(address.seed.table);words.set(address.seed.blocks,UNIFORM_PAGE_TABLE_WORDS);words.set(uniformBlockNeighbourWords(address.seed.table,address.seed.blocks),layout.rows);
const N=65536,repeats=9,launches=8;
const buffer=(size:number,usage:number)=>device.createBuffer({size,usage});
const table=buffer(words.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);device.queue.writeBuffer(table,0,words);
const input=buffer(16*N,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),output=buffer(16*N,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
const bindLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}}]});
const group=device.createBindGroup({layout:bindLayout,entries:[{binding:0,resource:{buffer:table}},{binding:1,resource:{buffer:input}},{binding:2,resource:{buffer:output}}]});
let prefix=`@group(0) @binding(0) var<storage,read> table:array<u32>;
@group(0) @binding(1) var<storage,read> points:array<vec4i>;
@group(0) @binding(2) var<storage,read_write> results:array<vec4u>;
fn umPageWord(p:vec3i)->u32{let q=vec3u(p)&vec3u(63u);return table[q.x|(q.y<<6u)|(q.z<<12u)];}
fn umBlockWord(b:u32)->u32{return table[${UNIFORM_PAGE_TABLE_WORDS}u+b];}
fn umBlockNeighbourWord(b:u32,i:u32)->u32{return table[${layout.rows}u+27u*b+i];}
${uniformLatticeAddressWGSL({constant:true,paged:{blocks:`${address.blocks}u`}})}
${uniformBlockNeighbourWGSL}`;
// Preserve the pre-optimization helper as the measured reference.
prefix=prefix.replace(/fn umBlockHeld\(home:u32,d:vec3i\)->vec3<bool>\{[\s\S]*?\n\}/,`fn umBlockHeld(home:u32,d:vec3i)->vec3<bool>{
 var held=vec3<bool>(false);var loose=0u;
 for(var a=0u;a<3u;a++){
  if(d[a]==0){continue;}
  var e=vec3i(0);e[a]=d[a];held[a]=umBlockNeighbour(home,e)==UM_SOLID_WORD;
  if(!held[a]){loose++;}
 }
 if(loose>=2u&&umBlockNeighbour(home,select(d,vec3i(0),held))==UM_SOLID_WORD){return d!=vec3i(0);}
 return held;
}`);
const candidate=prefix.replace(/fn umBlockHeld\(home:u32,d:vec3i\)->vec3<bool>\{[\s\S]*?\n\}/,`fn umBlockHeld(home:u32,d:vec3i)->vec3<bool>{
 let held=vec3<bool>(d.x!=0&&umBlockNeighbour(home,vec3i(d.x,0,0))==UM_SOLID_WORD,d.y!=0&&umBlockNeighbour(home,vec3i(0,d.y,0))==UM_SOLID_WORD,d.z!=0&&umBlockNeighbour(home,vec3i(0,0,d.z))==UM_SOLID_WORD);
 let free=select(d,vec3i(0),held);let loose=vec3u(free!=vec3i(0));
 if(loose.x+loose.y+loose.z>=2u&&umBlockNeighbour(home,free)==UM_SOLID_WORD){return d!=vec3i(0);}
 return held;
}`);
assert.notEqual(candidate,prefix);
const pipelines:GPUComputePipeline[][]=[];
for(const operation of ['hold','vertex','cubic']){
 const body=operation==='hold'?`let q=umVertexNearFromPage(home,page,p);sum=vec4u(bitcast<vec3u>(q),0u);`:operation==='vertex'?`sum=umVertexSlot(p);`:`for(var z=0;z<4;z++){for(var y=0;y<4;y++){for(var x=0;x<4;x++){
 let q=umVertexNearFromPage(home,page,p+vec3i(x,y,z)-1);let slot=umVertexSlot(q);sum=(sum^slot)*vec4u(16777619u);
 }}}`;
 const end=`@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){let p=points[id.x].xyz;let page=vec3i(points[id.x].w&63,(points[id.x].w>>6)&63,points[id.x].w>>12);let home=umBlockOf(page);var sum=vec4u(0u);${body}results[id.x]=sum;}`;
 const row=[];for(const source of [prefix,candidate]){
 const module=device.createShaderModule({code:source+end});const info=await module.getCompilationInfo();assert.deepEqual(info.messages.filter(m=>m.type==='error'),[]);
 row.push(await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[bindLayout]}),compute:{module,entryPoint:'main'}}));}pipelines.push(row);
}
const read=buffer(output.size,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
for(const scenario of ['interior','seam','wall','mixed']){
 const points=new Int32Array(4*N);
 for(let i=0;i<N;i++){
  const kind=scenario==='mixed'?['interior','seam','wall'][i%3]!:scenario;
  const q=kind==='interior'?[40+i%16,40+(i>>4)%16,40+(i>>8)%16]:kind==='seam'?[32+(i%3)-1,32+((i>>2)%3)-1,32+((i>>4)%3)-1]:[i%3-1,((i>>2)%3)-1,((i>>4)%3)-1];
  const page=kind==='wall'?0:1|(1<<6)|(1<<12);points.set([...q,page],4*i);
 }
 device.queue.writeBuffer(input,0,points);
 for(const [op,ps] of pipelines.entries()){
  const snapshots:Uint32Array[]=[];
  for(const p of ps){const e=device.createCommandEncoder(),pass=e.beginComputePass();pass.setPipeline(p);pass.setBindGroup(0,group);pass.dispatchWorkgroups(N/64);pass.end();e.copyBufferToBuffer(output,0,read,0,output.size);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);snapshots.push(new Uint32Array(read.getMappedRange().slice(0)));read.unmap();}
  assert.deepEqual(snapshots[0],snapshots[1]);
  const times:number[][]=[[],[]],queries=device.createQuerySet({type:'timestamp',count:4*repeats});
  const resolved=buffer(32*repeats,GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC),timing=buffer(32*repeats,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
  const e=device.createCommandEncoder();
  for(let r=0;r<repeats;r++)for(let j=0;j<2;j++){const arm=(r+j)%2,index=4*r+2*j,pass=e.beginComputePass({timestampWrites:{querySet:queries,beginningOfPassWriteIndex:index,endOfPassWriteIndex:index+1}});pass.setPipeline(ps[arm]!);pass.setBindGroup(0,group);for(let k=0;k<launches;k++)pass.dispatchWorkgroups(N/64);pass.end();}
  e.resolveQuerySet(queries,0,4*repeats,resolved,0);e.copyBufferToBuffer(resolved,0,timing,0,32*repeats);device.queue.submit([e.finish()]);await timing.mapAsync(GPUMapMode.READ);const t=new BigUint64Array(timing.getMappedRange());
  for(let r=0;r<repeats;r++)for(let j=0;j<2;j++){const index=4*r+2*j;times[(r+j)%2]!.push(Number(t[index+1]!-t[index]!)/1e6/launches);}
  const median=times.map(t=>t.toSorted((a,b)=>a-b)[repeats>>1]);console.log(JSON.stringify({scenario,operation:['hold','vertex','cubic'][op],samples:N,mismatches:0,reference_ms:median[0],candidate_ms:median[1],times}));
  timing.unmap();timing.destroy();resolved.destroy();queries.destroy();
 }
}
assert.deepEqual(errors,[]);read.destroy();table.destroy();input.destroy();output.destroy();device.destroy();
