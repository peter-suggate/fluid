/** Controlled surface-heat zero-predicate benchmarks: identical coordinates in page interiors,
 * across resident seams, and against SOLID walls. No solver or changing physics.
 * Every candidate output must equal the production helper bit for bit.
 * node --import tsx tools/probe-uniform-address-cost.ts */
import assert from 'node:assert/strict';
import {narrowBandActivityWGSL} from '../lib/methods/uniform/uniform-narrow-band-activity.wgsl';
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
const heat=bufferLate();
function bufferLate(){return device.createBuffer({size:4*address.tileKeys,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});}
const buffer=(size:number,usage:number)=>device.createBuffer({size,usage});
const table=buffer(words.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);device.queue.writeBuffer(table,0,words);
const input=buffer(16*N,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),output=buffer(16*N,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
const bindLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}}]});
const group=device.createBindGroup({layout:bindLayout,entries:[{binding:0,resource:{buffer:table}},{binding:1,resource:{buffer:input}},{binding:2,resource:{buffer:output}},{binding:3,resource:{buffer:heat}}]});
let prefix=`@group(0) @binding(0) var<storage,read> table:array<u32>;
@group(0) @binding(1) var<storage,read> points:array<vec4i>;
@group(0) @binding(2) var<storage,read_write> results:array<vec4u>;
fn umPageWord(p:vec3i)->u32{let q=vec3u(p)&vec3u(63u);return table[q.x|(q.y<<6u)|(q.z<<12u)];}
fn umBlockWord(b:u32)->u32{return table[${UNIFORM_PAGE_TABLE_WORDS}u+b];}
fn umBlockNeighbourWord(b:u32,i:u32)->u32{return table[${layout.rows}u+27u*b+i];}
${uniformLatticeAddressWGSL({constant:true,paged:{blocks:`${address.blocks}u`}})}
${uniformBlockNeighbourWGSL}`;
prefix+=`
@group(0) @binding(3) var<storage,read_write> bins:array<atomic<u32>>;
const NB_ACTIVITY_THETA=0u;const NB_ACTIVITY_SURFACE_HEAT=0u;
fn umCorner(k:u32,n:u32)->vec3u{return vec3u(k%n,(k/n)%n,k/(n*n));}
fn nbHomeBlock(page:vec3i,lo:vec3i,hi:vec3i)->u32{if(all(lo==page)&&all(hi==page)){return 0u;}return umBlockOf(page);}
// Deterministic alternating wet/dry field: zero predicate must ignore this
// denominator even when cold taps are dropped. Eight page lookups mirror
// the old query's nested trilinear sampling without a changing simulation.
fn bandPhi(p:vec3f)->f32{var sum=0.0;for(var k=0u;k<8u;k++){
 let s=umVertexSlot(vec3i(p)+vec3i(umCorner(k,2u)));sum+=f32((s.x+s.y+s.z+s.w)&3u)-1.5;
}return sum;}
`+narrowBandActivityWGSL.match(/fn nbInterpolatedHeat[\s\S]*?\n\}/)![0]+narrowBandActivityWGSL.match(/fn nbSurfaceHotCornerFrom[\s\S]*?\n\}/)![0];
const pipelines:GPUComputePipeline[][]=[[]];
for(const expr of ['nbInterpolatedHeat(home,vec3f(p),true)>0.0','nbSurfaceHotCornerFrom(home,vec3u(p))']){
 const module=device.createShaderModule({code:prefix+`@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){let p=points[id.x].xyz;let home=vec3i(points[id.x].w&63,(points[id.x].w>>6)&63,points[id.x].w>>12);results[id.x]=vec4u(u32(${expr}),0,0,0);}`});
 assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==='error'),[]);
 pipelines[0]!.push(await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[bindLayout]}),compute:{module,entryPoint:'main'}}));
}
const read=buffer(output.size,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
for(const scenario of ['cold','warm','hot','mixed']){
 const h=new Float32Array(address.tileKeys);
 for(let i=512;i<address.tileKeys-512;i++)h[i]=scenario==='cold'?0:scenario==='warm'?1:scenario==='hot'?2:[0,0.5,1,1.0000001192092896,2][(Math.imul(i,1664525)>>>0)%5]!;
 device.queue.writeBuffer(heat,0,h);
 const points=new Int32Array(4*N);
 for(let i=0;i<N;i++){
  const tile=[i%25,(i>>5)%25,(i>>10)%25],p=tile.map(n=>4*n),home=tile.map(n=>Math.min(n,23));
  points.set([...p,home[0]!|(home[1]!<<6)|(home[2]!<<12)],4*i);
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
  const median=times.map(t=>t.toSorted((a,b)=>a-b)[repeats>>1]);console.log(JSON.stringify({scenario,operation:'surface heat at coarse corner',samples:N,mismatches:0,reference_ms:median[0],candidate_ms:median[1],times}));
  timing.unmap();timing.destroy();resolved.destroy();queries.destroy();
 }
}
assert.deepEqual(errors,[]);read.destroy();table.destroy();input.destroy();output.destroy();heat.destroy();device.destroy();
