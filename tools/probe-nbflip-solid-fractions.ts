/** Compare exact cached and geometric open fractions on every resident cell.
 * WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/probe-nbflip-solid-fractions.ts [--frames=120]
 * Run under the shared GPU lock. Uses the authored long-slide settings. */
import assert from "node:assert/strict";
import { withUniformDevice } from "../tests/helpers/uniform-geometric";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import type { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import { uniformMixedSolidWGSL,type UniformMixedSolid } from "../lib/methods/uniform/uniform-mixed-solid.wgsl";
const frames=Number(process.argv.find(a=>a.startsWith("--frames="))?.slice(9)??120);assert.ok(Number.isInteger(frames)&&frames>0);
await withUniformDevice("resident solid fractions",async device=>{
 assert.ok(device.features.has("timestamp-query"));
 const definition=getSceneDefinition("nbflip-water-slide-long"),quality=definition.methodProfile!.quality,scene=sceneDocument(definition);
 const values=resolveMethodValues(method,quality,{...method.appDefaults,...definition.methodProfile!.overrides});
 const dt=method.effectiveStep_s?.(scene,values)??scene.numerics.fixedDt_s;
 const solver=await method.createSolverAsync!(device,scene,quality,values,undefined,()=>{}) as WebGPUUniformReferenceSolver;
 const buffers:GPUBuffer[]=[];const buffer=(size:number,usage:number)=>{const b=device.createBuffer({size,usage});buffers.push(b);return b;};
 let timestamps:GPUQuerySet|undefined;
 try{
  for(let frame=1;frame<=frames;frame++){assert.ok(solver.advanceTo(frame*dt,[]));await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);}
  const {ownership:o,solid}=(solver as unknown as {mixedFrame:{ownership:UniformMixedOwnership;solid:UniformMixedSolid}}).mixedFrame;
  const {nx,ny,nz}=solver.info,count=nx*ny*nz,repeats=5,groups=Math.ceil(count/64),dispatchX=Math.min(groups,device.limits.maxComputeWorkgroupsPerDimension);
  const output=buffer(16*count,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC),read=buffer(output.size,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
  const resolved=buffer(32*repeats,GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC),timesRead=buffer(32*repeats,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
  timestamps=device.createQuerySet({type:"timestamp",count:4*repeats});
  const module=device.createShaderModule({code:uniformMixedTopologyWGSL(o.capacity,0)+`
@group(1) @binding(0) var<storage,read_write> output:array<u32>;
${uniformMixedSolidWGSL(2,solid.coarse?.count)}
override cached:bool=false;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
 let i=id.x+${64*dispatchX}u*id.y;if(i>=${count}u){return;}let c=vec3i(vec3u(i%${nx}u,(i/${nx}u)%${ny}u,i/${nx*ny}u))+umWallLow();var value=-1.0;
 if(umCellPresent(c)){if(cached){value=umCellOpen(c);}else{value=umCellOpenGeometry(c);}}
 output[select(0u,${count}u,cached)+i]=bitcast<u32>(value);
 var boundary=0xffffffffu;if(umCellPresent(c)){if(cached){boundary=umBoundaryRecord(c);}else{boundary=umBoundaryRecordGeometry(c);}}
 output[select(${2*count}u,${3*count}u,cached)+i]=boundary;
}`});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");assert.deepEqual(errors,[]);
  const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
  const pipelines=await Promise.all([0,1].map(cached=>device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[o.bindLayout,layout,solid.tileLayout]}),compute:{module,entryPoint:"main",constants:{cached}}})));
  const group=device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:output}}]});const e=device.createCommandEncoder();
  for(let repeat=0;repeat<repeats;repeat++)for(let kind=0;kind<2;kind++){
   const cached=(repeat+kind)%2,index=4*repeat+2*kind;
   const pass=e.beginComputePass({label:cached?"cached exact cell fractions":"geometric cell fractions",timestampWrites:{querySet:timestamps,beginningOfPassWriteIndex:index,endOfPassWriteIndex:index+1}});
   pass.setPipeline(pipelines[cached]!);pass.setBindGroup(0,o.bindGroup);pass.setBindGroup(1,group);pass.setBindGroup(2,solid.tileGroup);pass.dispatchWorkgroups(dispatchX,Math.ceil(groups/dispatchX));pass.end();
  }
  e.copyBufferToBuffer(output,0,read,0,output.size);e.resolveQuerySet(timestamps,0,4*repeats,resolved,0);e.copyBufferToBuffer(resolved,0,timesRead,0,timesRead.size);device.queue.submit([e.finish()]);
  await Promise.all([read.mapAsync(GPUMapMode.READ),timesRead.mapAsync(GPUMapMode.READ)]);
  const mapped=read.getMappedRange(),words=new Uint32Array(mapped),fractions=new Float32Array(mapped),times=new BigUint64Array(timesRead.getMappedRange());let mismatches=0,boundaryMismatches=0,resident=0;const first:number[]=[];
  for(let i=0;i<count;i++){if(fractions[i]!==-1)resident++;if(words[2*count+i]!==words[3*count+i])boundaryMismatches++;if(words[i]!==words[count+i]){mismatches++;if(first.length<8)first.push(i);}}
  const geometric_ms:number[]=[],cached_ms:number[]=[];
  for(let repeat=0;repeat<repeats;repeat++)for(let kind=0;kind<2;kind++){const index=4*repeat+2*kind;((repeat+kind)%2?cached_ms:geometric_ms).push(Number(times[index+1]!-times[index]!)/1e6);}
  console.log(JSON.stringify({frames,count,resident,mismatches,boundaryMismatches,first,geometric_ms,cached_ms,recordBytes:solid.coarse!.record.size}));
  read.unmap();timesRead.unmap();assert.equal(mismatches,0,"all resident cached fractions match geometry bitwise");assert.equal(boundaryMismatches,0,"all resident boundary records match hash lookup");
 }finally{timestamps?.destroy();for(const b of buffers)b.destroy();solver.destroy();}
},["timestamp-query"]);
