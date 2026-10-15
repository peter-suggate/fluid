/** Same-state scalar velocity tap benchmark, isolating interior, page seams,
 * AIR and SOLID pages. Both arms execute the production atlas tap helper;
 * the reference removes only its page-local fast branch. */
import assert from 'node:assert/strict';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {uniformWindowAddressWGSL} from '../lib/methods/uniform/uniform-window-address';
import {uniformLatticeAddress,uniformLatticeAddressWGSL} from '../lib/methods/uniform/uniform-lattice-address';
import {createProcessRetainedDawnGPU,paceDawnEventPolling} from '../lib/harness/node-dawn-provider';
/** Eight adjacent scalar channels from an atlas field. Page-local stencils
 * resolve their block once; stencils crossing a page retain the ordinary
 * per-tap address, including AIR and SOLID sentinels. */
function uniformAtlasTapsWGSL(name:string,level:string,prefix:string,absent:string):string{return /* wgsl */ `
fn udLoadA8_${name}(q:vec3i,axis:u32)->array<f32,8>{
 var v:array<f32,8>;
 if(${absent}){return v;}
 if(all((q&vec3i(31))<vec3i(31))){
  let at=${prefix}umCellTexel(q);
  for(var k=0;k<8;k++){v[k]=textureLoad(${name},at+vec3i(k&1,(k>>1)&1,k>>2)${level})[axis];}
  return v;
 }
 for(var k=0;k<8;k++){v[k]=textureLoad(${name},${prefix}umCellTexel(q+vec3i(k&1,(k>>1)&1,k>>2))${level})[axis];}
 return v;
}
`;}

paceDawnEventPolling();
const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']),adapter=await gpu.requestAdapter();assert.ok(adapter);
const device=await adapter.requestDevice({requiredFeatures:['timestamp-query']});
const errors:string[]=[];device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
const a=uniformLatticeAddress([96,64,64],true,{blocks:20,permute:7,air:[[1,1,0]]});
const texture=device.createTexture({size:a.cellTexture,dimension:'3d',format:'rgba32float',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
const values=Float32Array.from({length:4*a.cellTexture.reduce((x,y)=>x*y,1)},(_,i)=>((i*32719)%65521)/65521);
device.queue.writeTexture({texture},values,{bytesPerRow:16*a.cellTexture[0],rowsPerImage:a.cellTexture[1]},a.cellTexture);
const make=(size:number,usage:number)=>device.createBuffer({size,usage});
const table=make(a.seed.tail.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);device.queue.writeBuffer(table,0,a.seed.tail);
const N=262144, REPEATS=15, script=make(N*16,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),result=make(N*32,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
const read=make(N*32,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
const prefix=`@group(0) @binding(0) var velocity:texture_3d<f32>;
@group(0) @binding(1) var<storage,read> pages:array<u32>;
@group(0) @binding(2) var<storage,read> points:array<vec4i>;
@group(0) @binding(3) var<storage,read_write> outputs:array<f32>;
${uniformWindowAddressWGSL('pages','0u',`${a.blocks}u`)}
${uniformLatticeAddressWGSL({constant:true,paged:{blocks:`${a.blocks}u`}})}
`;
const helper=uniformAtlasTapsWGSL('velocity',',0','','false');
const code=(fast:boolean)=>prefix+(fast?helper:helper.replace('if(all((q&vec3i(31))<vec3i(31)))','if(false)'))+`
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3u){let i=gid.x;let p=points[i];let v=udLoadA8_velocity(p.xyz,u32(p.w));for(var k=0u;k<8u;k++){outputs[8u*i+k]=v[k];}}`;
const modules=[false,true].map(fast=>device.createShaderModule({code:code(fast)}));
for(const m of modules){const info=await m.getCompilationInfo();assert.deepEqual(info.messages.filter(x=>x.type==='error'),[]);}
const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{viewDimension:'3d',sampleType:'unfilterable-float'}},...[1,2,3].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:(binding===3?'storage':'read-only-storage') as GPUBufferBindingType}}))]});
const pipelines=await Promise.all(modules.map(module=>device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint:'main'}})));
const group=device.createBindGroup({layout,entries:[{binding:0,resource:texture.createView()},...[table,script,result].map((buffer,i)=>({binding:i+1,resource:{buffer}}))]});
for(const name of ['interior','seam','air-solid','mixed']){
 const points=new Int32Array(N*4);
 for(let i=0;i<N;i++){
  const x=i%96,y=(i/96|0)%64,z=(i/6144|0)%64;
  let p=[x,y,z];if(name==='interior')p=[1+x%30,1+y%30,1+z%30];
  if(name==='seam')p[i%3]=31+32*((i/3|0)%2);
  if(name==='air-solid')p=[32+x%32,32+y%32,-1+z%34];
  points.set([...p,i%3],i*4);
 }
 device.queue.writeBuffer(script,0,points);
 const outputs:Uint32Array[]=[];
 for(let arm=0;arm<2;arm++){const enc=device.createCommandEncoder(),pass=enc.beginComputePass();pass.setPipeline(pipelines[arm]!);pass.setBindGroup(0,group);pass.dispatchWorkgroups(N/64);pass.end();enc.copyBufferToBuffer(result,0,read,0,read.size);device.queue.submit([enc.finish()]);await read.mapAsync(GPUMapMode.READ);outputs.push(new Uint32Array(read.getMappedRange()).slice());read.unmap();}
 let mismatch=0;for(let i=0;i<outputs[0]!.length;i++)if(outputs[0]![i]!==outputs[1]![i])mismatch++;assert.equal(mismatch,0);
 const queries=device.createQuerySet({type:'timestamp',count:REPEATS*4}),resolve=make(REPEATS*32,GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC),times=make(REPEATS*32,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ),enc=device.createCommandEncoder();
 for(let r=0;r<REPEATS;r++)for(let k=0;k<2;k++){let index=4*r+2*k;const pass=enc.beginComputePass({timestampWrites:{querySet:queries,beginningOfPassWriteIndex:index,endOfPassWriteIndex:index+1}});pass.setPipeline(pipelines[(r+k)%2]!);pass.setBindGroup(0,group);pass.dispatchWorkgroups(N/64);pass.end();}
 enc.resolveQuerySet(queries,0,REPEATS*4,resolve,0);enc.copyBufferToBuffer(resolve,0,times,0,times.size);device.queue.submit([enc.finish()]);await times.mapAsync(GPUMapMode.READ);
 const t=new BigUint64Array(times.getMappedRange()),ms:[number[],number[]]=[[],[]];for(let r=0;r<REPEATS;r++)for(let k=0;k<2;k++){const index=4*r+2*k;ms[(r+k)%2]!.push(Number(t[index+1]!-t[index]!)/1e6);}const median=(v:number[])=>[...v].sort((x,y)=>x-y)[v.length>>1];
 console.log(JSON.stringify({name,points:N,mismatches:mismatch,reference_ms:median(ms[0]),optimized_ms:median(ms[1]),samples:ms}));times.unmap();times.destroy();resolve.destroy();queries.destroy();
}
assert.deepEqual(errors,[]);await device.queue.onSubmittedWorkDone();device.destroy();
