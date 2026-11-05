import type { UniformClothGridFields } from './uniform-cloth-grid-coupling';
import { uniformDetailBindLayout, uniformDetailGroup, uniformDetailModule, uniformDetailPipeline, type UniformDetailGroup } from './uniform-detail-fields';
import { uniformMixedTopologyWGSL } from './uniform-mixed-topology.wgsl';
import { uniformMixedSolidWGSL } from './uniform-mixed-solid.wgsl';
import { levelSetFillWGSL } from '../../core/level-set-fill.wgsl';
import { NB_VOLUME_COMPENSATION_FLOAT } from './uniform-mixed-surface-volume';
import { sourceFieldDistance } from '../../cloth/source-fields';
import { sourceWaterBounds, type SourceWaterOperation } from '../../cloth/source-water';
import type { SourceBounds } from '../../cloth/source-motion';
import type { WetClothSourceField } from '../../cloth/source-types';
import type { ClothVector } from '../../cloth/types';

type P=[number,number,number];
interface Batch {buffers:GPUBuffer[];groups:Record<'read'|'write'|'compute',UniformDetailGroup>;support:GPUBindGroup;vertices:number;cells:number;faces:number;bounds:SourceBounds}
const key=(p:readonly number[])=>p.join(',');
const bit=(k:number):P=>[k&1,(k>>1)&1,(k>>2)&1];

/** Source plugin operating on the production NB-FLIP phi and MAC textures.
 * CPU work supplies only geometry samples/connectivity. Every old/new volume,
 * source momentum and budget delta is measured on the GPU. Caller reserves h
 * ownership for supportBounds before the census, then encodes after census and
 * before the FLIP snapshot. No global surface displacement creates the source. */
export class UniformSourceWater {
 readonly id='physics.wetcloth.source-water';
 private readonly layouts:Record<'read'|'write'|'compute',GPUBindGroupLayout>;
 private readonly supportLayout:GPUBindGroupLayout;
 private supportPipeline?:GPUComputePipeline;
 private pipelines=new Map<string,GPUComputePipeline>();
 private batches:Batch[]=[];
 readonly externalLedger:GPUBuffer;
 constructor(readonly fields:UniformClothGridFields){
  const b=(binding:number,type:GPUBufferBindingType):GPUBindGroupLayoutEntry=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type}});
  this.supportLayout=fields.device.createBindGroupLayout({entries:[b(0,'read-only-storage'),b(1,'storage'),b(2,'storage'),b(3,'uniform')]});
  this.externalLedger=fields.device.createBuffer({label:'Source accepted external water ledger',size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const shared=[b(0,'read-only-storage'),b(1,'storage'),b(2,'storage'),b(3,'uniform'),b(8,'storage')];
  this.layouts={
   compute:uniformDetailBindLayout(fields.device,{entries:shared}),
   read:uniformDetailBindLayout(fields.device,{entries:[...shared,...[4,5].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float' as const,viewDimension:'3d' as const}}))]}),
   write:uniformDetailBindLayout(fields.device,{entries:[...shared,...[6,7].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:'write-only' as const,format:(binding===6?'r32float':'rgba32float') as GPUTextureFormat,viewDimension:'3d' as const}}))]}),
  };
 }
 get supportBounds():readonly SourceBounds[]{return this.batches.map(b=>b.bounds);}
 async initialize():Promise<void>{
  const f=this.fields,d=f.device,solid=f.solid;
  const support=d.createShaderModule({label:'Source local support wake',code:uniformMixedTopologyWGSL(f.ownership.capacity,0)+f.heat.promoteWGSL(1,1,2)+/* wgsl */`
@group(1) @binding(0) var<storage,read> sites:array<vec4i>;
@group(1) @binding(3) var<uniform> counts:vec4u;
@compute @workgroup_size(64) fn wake(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=counts.x){return;}let p=sites[i].xyz-32*umFramePage();
 let tile=512u*umBlockOf(p>>vec3u(5u))+umTileLocal(vec3u(p>>vec3u(2u)));
 if(umKeyStored(tile)){rhPromote(umPageOfTile(tile));}
}`});
  this.supportPipeline=await d.createComputePipelineAsync({layout:d.createPipelineLayout({bindGroupLayouts:[f.ownership.bindLayout,this.supportLayout]}),compute:{module:support,entryPoint:'wake'}});
  const sourceCode=uniformMixedTopologyWGSL(f.ownership.capacity,0)+uniformMixedSolidWGSL(solid?2:undefined,solid?.coarse?.count)+levelSetFillWGSL+/* wgsl */`
struct Params {counts:vec4u,velocity:vec4f,scale:vec4f}
@group(1) @binding(0) var<storage,read> input:array<vec4u>;
@group(1) @binding(1) var<storage,read_write> work:array<vec4f>;
@group(1) @binding(2) var<storage,read_write> budget:array<f32>;
@group(1) @binding(3) var<uniform> params:Params;
@group(1) @binding(4) var phi:texture_3d<f32>;
@group(1) @binding(5) var velocity:texture_3d<f32>;
@group(1) @binding(6) var field:texture_storage_3d<r32float,write>;
@group(1) @binding(7) var output:texture_storage_3d<rgba32float,write>;
@group(1) @binding(8) var<storage,read_write> ledger:array<f32>;
fn cellBase()->u32{return params.counts.x;}
fn faceBase()->u32{return params.counts.x+3u*params.counts.y;}
fn cellResult(i:u32)->vec4f{if(i==0xffffffffu){return vec4f(0);}return work[params.counts.x+i];}
@compute @workgroup_size(64) fn sample(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=params.counts.x){return;}let site=input[i];let p=bitcast<vec3i>(site.xyz)-32*umFramePage();
 let old=/*c*/textureLoad(phi,p,0).x;let shape=bitcast<f32>(site.w);var next=old;
 if(shape<=params.scale.x){next=select(min(old,shape),max(old,-shape),params.counts.w==1u);}
 work[i]=vec4f(old,next,0,0);
}
@compute @workgroup_size(64) fn measure(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=params.counts.y){return;}let base=cellBase()+3u*i;let p=bitcast<vec3i>(input[base].xyz)-32*umFramePage();
 let tile=512u*umBlockOf(p>>vec3u(5u))+umTileLocal(vec3u(p>>vec3u(2u)));
 if(!umKeyStored(tile)||umTileWidth(tile)!=1u){umSupport[UM_RESIDENCY+1u]=1u;work[params.counts.x+i]=vec4f(0);return;}
 let ids0=input[base+1u];let ids1=input[base+2u];var before:array<f32,8>;var after:array<f32,8>;
 for(var k=0u;k<4u;k++){before[k]=work[ids0[k]].x;after[k]=work[ids0[k]].y;before[k+4u]=work[ids1[k]].x;after[k+4u]=work[ids1[k]].y;}
 let capacity=umCellOpen(p)*umPorousFreeCapacity(p);let old=fill(before)*capacity;let next=fill(after)*capacity;
 work[params.counts.x+i]=vec4f(next-old,old,next,0);
}
@compute @workgroup_size(64) fn writePhi(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=params.counts.x||umSupport[UM_RESIDENCY+1u]!=0u){return;}
 textureStore(field,bitcast<vec3i>(input[i].xyz)-32*umFramePage(),vec4f(work[i].y,0,0,0));
}
@compute @workgroup_size(64) fn writeVelocity(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=params.counts.z||umSupport[UM_RESIDENCY+1u]!=0u){return;}let base=faceBase()+2u*i;let p=bitcast<vec3i>(input[base].xyz)-32*umFramePage();let ids=input[base+1u];
 var v=work[params.counts.x+params.counts.y+i];let left=cellResult(ids.x);
 if(params.counts.w==0u){for(var a=0u;a<3u;a++){let right=cellResult(ids[a+1u]);let old=left.y+right.y;let change=left.x+right.x;if(change>0.0){v[a]=(old*v[a]+change*params.velocity[a])/(old+change);}}}
 textureStore(output,p,v);
}
@compute @workgroup_size(64) fn sampleVelocity(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=params.counts.z){return;}
 let p=bitcast<vec3i>(input[faceBase()+2u*i].xyz)-32*umFramePage();
 work[params.counts.x+params.counts.y+i]=/*c*/textureLoad(velocity,p,0);
}
// Fixed pairwise reduction over cell deltas; dispatch boundaries make every
// read depend only on the previous level, never workgroup scheduling.
override reduceStride:u32=1u;
@compute @workgroup_size(64) fn reduce(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x*2u*reduceStride;if(i>=params.counts.y){return;}let j=i+reduceStride;
 if(j<params.counts.y){work[params.counts.x+i].x+=work[params.counts.x+j].x;}
}
@compute @workgroup_size(1) fn commitBudget(){
 if(umSupport[UM_RESIDENCY+1u]!=0u){return;}
 if(budget[1]!=1.0){umSupport[UM_RESIDENCY+1u]=1u;return;}
 let delta=work[params.counts.x].x;let old=budget[0];budget[0]=old+delta;
 // Retain the existing compensated remainder, including pore transfers.
 // The error term is evaluated after the rounded high word storage write.
 budget[${NB_VOLUME_COMPENSATION_FLOAT}]+= (old-budget[0])+delta;
 let physical=delta*params.scale.y;let previous=ledger[0];ledger[0]=previous+physical;ledger[1]+=(previous-ledger[0])+physical;
 budget[3]=0.0;
}
`;
  const phases={read:['sample','sampleVelocity'],write:['writePhi','writeVelocity'],compute:['measure','reduce','commitBudget']} as const;
  for(const phase of ['read','write','compute'] as const){
   const names:readonly string[]=phases[phase];
   const code=sourcePhaseShader(sourceCode,names,phase);
   const module=uniformDetailModule(d,{label:`Physical source water ${phase}`,code});
  const errors=(await module.getCompilationInfo()).messages.filter(x=>x.type==='error');if(errors.length)throw new Error(errors.map(x=>`${x.lineNum}:${x.linePos} ${x.message}`).join('\n'));
  const layout=d.createPipelineLayout({bindGroupLayouts:[f.ownership.bindLayout,this.layouts[phase],...(solid?[solid.tileLayout]:[])]});
  for(const entryPoint of names.filter(n=>n!=='reduce'))this.pipelines.set(entryPoint,await uniformDetailPipeline(d,f.ownership,{layout,compute:{module,entryPoint}}));
  // Finite address space bounds this reduction; all per-frame batches reuse
  // the compiled powers of two rather than compiling during simulation.
  if(phase==='compute')for(let stride=1;stride<2**24;stride*=2)this.pipelines.set(`reduce${stride}`,await uniformDetailPipeline(d,f.ownership,{layout,compute:{module,entryPoint:'reduce',constants:{reduceStride:stride}}}));
  }
 }
 /** Call only after the preceding command submission has finished using the
  * previous preparation. Bounds include ALL incident cells of changed nodes
  * and adjacent cells of their MAC faces. */
 prepare(sourceFields:readonly WetClothSourceField[],operations:readonly SourceWaterOperation[],domain?:SourceBounds):void{
  this.clear();const f=this.fields,h=f.cellSize_m,minH=Math.min(...h),bounds=sourceWaterBounds(sourceFields,operations,domain);
  for(let opIndex=0;opIndex<operations.length;opIndex++){
   const op=operations[opIndex]!,b=bounds[opIndex]!,lower=b.min.map((v,a)=>Math.floor((v-f.origin_m[a]!)/h[a]!)-3) as P,upper=b.max.map((v,a)=>Math.ceil((v-f.origin_m[a]!)/h[a]!)+3) as P;
   if([...lower,...upper].some(v=>!Number.isFinite(v)))throw new Error('Source water operation needs finite support bounds');
   const extent=upper.map((v,a)=>v-lower[a]!);if(extent.some(v=>v<=0))continue;
   const estimate=extent.reduce((p,v)=>p*(v+1),1);
   if(estimate>64*65535||estimate*96>f.device.limits.maxStorageBufferBindingSize)throw new Error('Source support exceeds local transaction capacity; split its exact support before admission');
   const cells:P[]=[],cellMap=new Map<string,number>();
   for(let z=lower[2];z<upper[2];z++)for(let y=lower[1];y<upper[1];y++)for(let x=lower[0];x<upper[0];x++){const p:P=[x,y,z];cellMap.set(key(p),cells.length);cells.push(p);}
   if(cells.length>2**24)throw new Error('Source support exceeds exact GPU reduction capacity');
   const vertices:P[]=[],map=new Map<string,number>(),cellVertices:number[][]=[];
   const vertex=(p:P)=>{const k=key(p);let id=map.get(k);if(id===undefined){id=vertices.length;map.set(k,id);vertices.push(p);}return id;};
   for(const p of cells)cellVertices.push(Array.from({length:8},(_,i)=>vertex(p.map((v,a)=>v+bit(i)[a]!) as P)));
   if(vertices.length>64*65535)throw new Error('Source support exceeds one-dimensional GPU dispatch capacity');
   const faces:P[]=[];for(let z=lower[2]+1;z<upper[2]-1;z++)for(let y=lower[1]+1;y<upper[1]-1;y++)for(let x=lower[0]+1;x<upper[0]-1;x++)faces.push([x,y,z]);
   const input=new Uint32Array(4*(vertices.length+3*cells.length+2*faces.length)),signed=new Int32Array(input.buffer),float=new Float32Array(input.buffer);
   vertices.forEach((p,i)=>{signed.set(p,4*i);const world=p.map((v,a)=>f.origin_m[a]!+v*h[a]!) as unknown as ClothVector;float[4*i+3]=sourceFieldDistance(sourceFields,op.fieldIndex,world);});
   cells.forEach((p,i)=>{const at=4*(vertices.length+3*i);signed.set(p,at);input.set(cellVertices[i]!,at+4);});
   faces.forEach((p,i)=>{const at=4*(vertices.length+3*cells.length+2*i);signed.set(p,at);input[at+4]=cellMap.get(key(p))!;for(let a=0;a<3;a++){const q=[...p] as P;q[a]!++;input[at+5+a]=cellMap.get(key(q))??0xffffffff;}});
   const buffers:GPUBuffer[]=[],allocate=(size:number,usage:GPUBufferUsageFlags,data?:ArrayBufferView)=>{const buffer=f.device.createBuffer({label:'Wet cloth source support',size:Math.max(16,size),usage});buffers.push(buffer);if(data)f.device.queue.writeBuffer(buffer,0,data.buffer as ArrayBuffer,data.byteOffset,data.byteLength);return buffer;};
   const params=new ArrayBuffer(48),counts=new Uint32Array(params),numbers=new Float32Array(params);counts.set([vertices.length,cells.length,faces.length,op.kind==='emit'?0:1]);numbers.set([...op.velocity_m_s,0,2*minH,h[0]*h[1]*h[2],0,0],4);
   const inputs=allocate(input.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,input),work=allocate(16*(vertices.length+cells.length+faces.length),GPUBufferUsage.STORAGE),uniform=allocate(48,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,new Uint8Array(params));
   const sharedEntries:GPUBindGroupEntry[]=[{binding:0,resource:{buffer:inputs}},{binding:1,resource:{buffer:work}},{binding:2,resource:{buffer:f.waterBudget}},{binding:3,resource:{buffer:uniform}},{binding:8,resource:{buffer:this.externalLedger}}];
   const groups={
    compute:uniformDetailGroup(f.device,{layout:this.layouts.compute,entries:sharedEntries}),
    read:uniformDetailGroup(f.device,{layout:this.layouts.read,entries:[...sharedEntries,{binding:4,resource:f.phi},{binding:5,resource:f.projectedVelocity}]}),
    write:uniformDetailGroup(f.device,{layout:this.layouts.write,entries:[...sharedEntries,{binding:6,resource:f.phi},{binding:7,resource:f.projectedVelocity}]}),
   };
   const support=f.device.createBindGroup({layout:this.supportLayout,entries:[inputs,f.heat.tiers,f.heat.state,uniform].map((buffer,binding)=>({binding,resource:{buffer}}))});
   this.batches.push({buffers,groups,support,vertices:vertices.length,cells:cells.length,faces:faces.length,bounds:{min:lower.map((v,a)=>f.origin_m[a]!+v*h[a]!) as unknown as ClothVector,max:upper.map((v,a)=>f.origin_m[a]!+v*h[a]!) as unknown as ClothVector}});
  }
 }
 encodeSupport(encoder:GPUCommandEncoder):void{
  if(!this.supportPipeline)throw new Error('Source water plugin is not initialized');
  for(const batch of this.batches){const p=encoder.beginComputePass({label:'Wake source support'});p.setPipeline(this.supportPipeline);p.setBindGroup(0,this.fields.ownership.bindGroup);p.setBindGroup(1,batch.support);p.dispatchWorkgroups(Math.ceil(batch.vertices/64));p.end();}
 }
 encode(encoder:GPUCommandEncoder):boolean{
  const run=(batch:Batch,name:string,count:number)=>{const pipeline=this.pipelines.get(name);if(!pipeline)throw new Error('Source water plugin is not initialized');const pass=encoder.beginComputePass({label:`Source water ${name}`});pass.setPipeline(pipeline);pass.setBindGroup(0,this.fields.ownership.bindGroup);const phase=name.startsWith('sample')?'read':name.startsWith('write')?'write':'compute';pass.setBindGroup(1,batch.groups[phase].group);if(this.fields.solid)pass.setBindGroup(2,this.fields.solid.tileGroup);pass.dispatchWorkgroups(Math.max(1,Math.ceil(count/64)));pass.end();};
  for(const batch of this.batches){run(batch,'sample',batch.vertices);run(batch,'measure',batch.cells);run(batch,'writePhi',batch.vertices);run(batch,'sampleVelocity',batch.faces);run(batch,'writeVelocity',batch.faces);for(let stride=1;stride<batch.cells;stride*=2)run(batch,`reduce${stride}`,Math.ceil(batch.cells/(2*stride)));run(batch,'commitBudget',1);}
  return this.batches.length>0;
 }
 private clear():void{for(const batch of this.batches)for(const b of batch.buffers)b.destroy();this.batches=[];}
 async readLedger():Promise<{readonly externalVolume_m3:number}>{
  const device=this.fields.device,read=device.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{const e=device.createCommandEncoder();e.copyBufferToBuffer(this.externalLedger,0,read,0,16);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);const v=new Float32Array(read.getMappedRange());return {externalVolume_m3:v[0]!+v[1]!};}
  finally{if(read.mapState==='mapped')read.unmap();read.destroy();}
 }
 destroy():void{this.clear();this.externalLedger.destroy();this.pipelines.clear();}
}

function sourcePhaseShader(source:string,names:readonly string[],phase:'read'|'write'|'compute'):string{
 let code=source.replace(/@group\(1\) @binding\(([4567])\)[^;]+;/g,(declaration,binding:string)=>phase==='read'&&Number(binding)<6||phase==='write'&&Number(binding)>=6?declaration:'');
 const entry=/@compute[^\n]*?fn (\w+)\([^]*?\{/g;
 let match:RegExpExecArray|null;
 while((match=entry.exec(code))){
  let end=entry.lastIndex,depth=1;while(depth&&end<code.length){if(code[end]==='{')depth++;else if(code[end]==='}')depth--;end++;}
  if(!names.includes(match[1]!)){code=code.slice(0,match.index)+code.slice(end);entry.lastIndex=match.index;}else entry.lastIndex=end;
 }
 return code;
}
