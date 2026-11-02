/** Isolate the production surface scatter across controlled tile populations.
 * Includes the real planner in parity checks and timings.
 * Run under /private/tmp/claude-501/fluid-gpu.lock, e.g.
 * node --import tsx tools/benchmark-nbflip-splat.ts --variants=aa,splatguard,splat64,splat256,splatguard64
 * This measures synthetic workloads, not complete simulation frames.
 */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {narrowBandSplatWGSL} from '../lib/methods/uniform/uniform-narrow-band-splat.wgsl';
import {createProcessRetainedDawnGPU,paceDawnEventPolling} from '../lib/harness/node-dawn-provider';
import {probeVariants} from './probe-nbflip-variants';
const option=(name:string)=>process.argv.find(a=>a.startsWith(`--${name}=`))?.slice(name.length+3);
const names=(option('variants')??'aa,splatguard,splat64,splat256,splatguard64').split(',');
const cold=option('cold')??'mixed';assert.ok(['none','mixed','all'].includes(cold));
const counts=(option('counts')??'1,8,64,512,1025').split(',').map(Number);
const repeats=Number(option('repeats')??9),inner=Number(option('inner')??16),tiles=512,vertices=33**3;
const planGroups=Number(option('plan-groups')??1024),scatterGroups=Number(option('scatter-groups')??1024);
assert.ok([repeats,inner,planGroups,scatterGroups].every(n=>Number.isInteger(n)&&n>0));
assert.ok(counts.every(n=>Number.isInteger(n)&&n>0));
assert.ok(names.every(n=>n==='aa'||probeVariants[n]));
paceDawnEventPolling();
const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']);const adapter=await gpu.requestAdapter();assert.ok(adapter);
const device=await adapter.requestDevice({requiredFeatures:['timestamp-query']});
const errors:string[]=[];device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);});
const coverage=64*tiles,depth=coverage+tiles,hotDepth=depth+vertices,band=hotDepth+vertices,occupied=band+4,words=occupied+4+tiles;
const code=`
const UM_CELL_KEYS=${64*tiles}u;const UM_TILES=${tiles}u;const NB_LINKS=UM_CELL_KEYS;
const NB_COVERAGE=${coverage}u;const NB_DEPTH_A=${depth}u;const NB_SURFACE_HOT_NEAREST=${hotDepth}u;
const NB_BAND=${band}u;const NB_OCCUPIED=${occupied}u;const nbSparseBins=true;
@group(0) @binding(0) var<storage,read_write> bins:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read_write> links:array<u32>;
@group(0) @binding(2) var<storage,read> particles:array<vec4f>;
fn nbBinTile(t:u32)->u32{return 64u*t;}fn nbBinCell(c:u32)->u32{return c;}
fn umTileLive(t:u32)->bool{return t<UM_TILES;}
fn umLiveTileCount()->u32{return UM_TILES;}fn umLiveTile(j:u32)->u32{return j;}
fn nbTileCold(t:u32)->bool{return ${cold==='mixed'?'t%7u==0u':cold==='all'?'true':'false'};}
fn umCorner(j:u32,n:u32)->vec3u{return vec3u(j%n,(j/n)%n,j/(n*n));}
fn umTileCoord(t:u32)->vec3u{return umCorner(t,8u);}
fn umVertexPresent(p:vec3i)->bool{return all(p>=vec3i(0))&&all(p<vec3i(33));}
fn umVertexAt(p:vec3u)->u32{return p.x+33u*(p.y+33u*p.z);}
fn umVertexSlot(p:vec3i)->vec4u{return vec4u(vec3u(p),u32(umVertexPresent(p)));}
fn umBlockStored(b:u32)->bool{return b!=0u;}
fn umVertexSlotAt(p:vec4u)->u32{return umVertexAt(p.xyz);}
fn umLoadAddressPlan(b:u32,lane:u32){}
fn nbPosition(i:u32)->vec3f{let a=NB_LINKS+4u*i;return bitcast<vec3f>(vec3u(links[a],links[a+1u],links[a+2u]));}
fn nbSurfaceParticleHot(i:u32)->bool{return bitcast<f32>(links[NB_LINKS+4u*i+3u])>1.0;}
`+narrowBandSplatWGSL;
const layout=device.createBindGroupLayout({entries:[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===2?'read-only-storage' as const:'storage' as const}}))});
const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
async function pipeline(source:string,entryPoint:string){
 const module=device.createShaderModule({code:source});
 assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==='error'),[]);
 return device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}});
}
console.log(JSON.stringify({adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture,device:adapter.info.device,description:adapter.info.description},sourceSha256:createHash('sha256').update(code).digest('hex'),counts,names,cold,tiles,repeats,inner,planGroups,scatterGroups,scope:'synthetic production planner + scatter; includes job construction'}));
try{
 const plans:GPUComputePipeline[]=[],scatter:GPUComputePipeline[]=[];
 for(const name of names){
  const variant=name==='aa'?code:probeVariants[name]!(code,'surfaceSplatOccupied');
  plans.push(await pipeline(variant,'surfaceSplatPlan'));scatter.push(await pipeline(variant,'surfaceSplatOccupied'));console.log(JSON.stringify({compiled:name}));
 }
 const reset=await pipeline(code+`\n@compute @workgroup_size(1) fn resetJobs(){atomicStore(&bins[NB_BAND+3u],0u);}`,'resetJobs');
 const cellPlan=await pipeline(code,'surfaceCells'),reference=await pipeline(code,'surfaceSplatReference');
 const buffer=(size:number,usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST)=>device.createBuffer({size,usage});
 for(const perTile of counts){
  const total=tiles*perTile,initial=new Uint32Array(words),stream=new Uint32Array(64*tiles+7*total),positions=new Float32Array(4*total);
  for(let t=0;t<tiles;t++){
   initial[coverage+t]=perTile;initial[occupied+4+t]=tiles-1-t;
   for(let i=0;i<perTile;i++){
    const cell=Math.floor(i*64/perTile),p=t*perTile+i;
    initial[64*t+cell]!++;
    positions.set([4*(t%8)+(cell%4)+(1+(i*17)%126)/128,4*(Math.floor(t/8)%8)+(Math.floor(cell/4)%4)+(1+(i*31)%126)/128,4*Math.floor(t/64)+Math.floor(cell/16)+(1+(i*53)%126)/128,i%3===0?2:1],4*p);
   }
  }
  initial[occupied]=tiles;let start=0;for(let c=0;c<64*tiles;c++){stream[c]=start;start+=initial[c]!;}
  stream.set(new Uint32Array(positions.buffer),64*tiles);
  const bins=buffer(initial.byteLength),links=buffer(stream.byteLength),particles=buffer(positions.byteLength);
  device.queue.writeBuffer(links,0,stream);
  const group=device.createBindGroup({layout,entries:[bins,links,particles].map((buffer,binding)=>({binding,resource:{buffer}}))});
  const output=buffer(8*vertices,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
  const indirect=buffer(12,GPUBufferUsage.INDIRECT|GPUBufferUsage.COPY_DST);device.queue.writeBuffer(indirect,0,new Uint32Array([0,1,1]));
  const queries=device.createQuerySet({type:'timestamp',count:2*repeats*names.length});
  const resolved=buffer(16*repeats*names.length,GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC),times=buffer(resolved.size,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
  const dispatch=(e:GPUCommandEncoder,p:GPUComputePipeline,groups:number,timestampWrites?:GPUComputePassTimestampWrites,n=1)=>{const pass=e.beginComputePass({timestampWrites});pass.setBindGroup(0,group);pass.setPipeline(p);for(let i=0;i<n;i++)pass.dispatchWorkgroups(groups);pass.end();};
  try{
   let oracle:Uint32Array|undefined;
   for(let v=-1;v<names.length;v++){
    device.queue.writeBuffer(bins,0,initial);const e=device.createCommandEncoder();
    if(v<0){
     dispatch(e,cellPlan,1024);e.copyBufferToBuffer(bins,4*(band+3),indirect,0,4);
     const pass=e.beginComputePass();pass.setBindGroup(0,group);pass.setPipeline(reference);pass.dispatchWorkgroupsIndirect(indirect,0);pass.end();
    }
    else{dispatch(e,plans[v]!,planGroups);dispatch(e,scatter[v]!,scatterGroups);}
    e.copyBufferToBuffer(bins,4*depth,output,0,output.size);device.queue.submit([e.finish()]);await output.mapAsync(GPUMapMode.READ);
    const actual=new Uint32Array(output.getMappedRange()).slice();output.unmap();
    if(!oracle){oracle=actual;assert.ok(actual.some(n=>n!==0));}else assert.deepEqual(actual,oracle,`${names[v]} at ${perTile} particles/tile`);
   }
   // Include worklist construction. Reset the job count between repetitions;
   // the global distance reduction is idempotent, and local windows reset.
   const e=device.createCommandEncoder();let slot=0;
   for(let r=0;r<repeats;r++)for(let k=0;k<names.length;k++){
    const v=(k+r)%names.length;
    const pass=e.beginComputePass({timestampWrites:{querySet:queries,beginningOfPassWriteIndex:2*slot,endOfPassWriteIndex:2*slot+1}});pass.setBindGroup(0,group);
    for(let i=0;i<inner;i++){pass.setPipeline(reset);pass.dispatchWorkgroups(1);pass.setPipeline(plans[v]!);pass.dispatchWorkgroups(planGroups);pass.setPipeline(scatter[v]!);pass.dispatchWorkgroups(scatterGroups);}pass.end();slot++;
   }
   e.resolveQuerySet(queries,0,2*slot,resolved,0);e.copyBufferToBuffer(resolved,0,times,0,times.size);device.queue.submit([e.finish()]);await times.mapAsync(GPUMapMode.READ);
   const stamps=new BigUint64Array(times.getMappedRange());
   const variants=names.map((name,v)=>{const ms:number[]=[];for(let s=0;s<slot;s++)if((s%names.length+Math.floor(s/names.length))%names.length===v)ms.push(Number(stamps[2*s+1]!-stamps[2*s]!)/1e6/inner);const sorted=[...ms].sort((a,b)=>a-b);return {name,median_ms:sorted[Math.floor(sorted.length/2)],ms};});
   times.unmap();console.log(JSON.stringify({perTile,particles:total,parity:'exact against scalar cell reference',variants}));
  }finally{for(const b of [bins,links,particles,output,indirect,resolved,times])b.destroy();queries.destroy();}
 }
 assert.deepEqual(errors,[]);
}finally{await device.queue.onSubmittedWorkDone();device.destroy();}
