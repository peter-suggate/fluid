import type {UniformClothGridFields} from './uniform-cloth-grid-coupling';
import type {UniformNarrowBandFlip} from './uniform-narrow-band-flip';
import {UniformSourceParticleTermination} from './uniform-source-termination';
import {uniformDetailBindLayout,uniformDetailGroup,uniformDetailModule,uniformDetailPipeline,type UniformDetailGroup} from './uniform-detail-fields';
import {uniformMixedTopologyWGSL} from './uniform-mixed-topology.wgsl';
import {uniformMixedVertexSamplingWGSL} from './uniform-mixed-vertex-sampling.wgsl';
import {uniformMixedSolidWGSL} from './uniform-mixed-solid.wgsl';
import {levelSetFillWGSL} from '../../core/level-set-fill.wgsl';
import {NB_VOLUME_COMPENSATION_FLOAT} from './uniform-mixed-surface-volume';
import type {SourceBounds} from '../../cloth/source-motion';
import {sourceTerminationBoundaryBounds} from '../../cloth/source-water';
import type {WetClothSourceField} from '../../cloth/source-types';

/** Removes source-terminated water over the GPU's actual resident owners.
 * Coverage follows page migration; no fixed host scene envelope truncates it.
 * Only source-boundary bands need h ownership. Deep/coarse water outside an
 * inverted box is removed using its native volume authority. */
export class UniformSourceResidentTermination {
 readonly id='physics.wetcloth.resident-termination';
 readonly particles:UniformSourceParticleTermination;
 readonly externalLedger:GPUBuffer;
 private readonly layout:GPUBindGroupLayout;
 private readonly work:GPUBuffer;
 private readonly params:GPUBuffer;
 private shapes?:GPUBuffer;
 private group?:UniformDetailGroup;
 private pipelines=new Map<string,GPUComputePipeline>();
 private count=0;
 private bounds:SourceBounds[]=[];
 constructor(private readonly fields:UniformClothGridFields,particles:UniformNarrowBandFlip){
  const d=fields.device,b=(binding:number,type:GPUBufferBindingType):GPUBindGroupLayoutEntry=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type}});
  this.particles=new UniformSourceParticleTermination(fields,particles);
  this.layout=uniformDetailBindLayout(d,{entries:[b(0,'read-only-storage'),b(1,'storage'),b(2,'storage'),b(3,'uniform'),{binding:4,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:'read-write',format:'r32float',viewDimension:'3d'}},b(5,'storage'),b(6,'storage'),b(7,'storage')]});
  this.work=d.createBuffer({label:'Resident terminator exact native volume changes',size:4*fields.ownership.capacity.owners,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  this.params=d.createBuffer({size:48,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  this.externalLedger=d.createBuffer({label:'Resident terminator external water ledger',size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
 }
 get supportBounds():readonly SourceBounds[]{return this.bounds;}
 async initialize():Promise<void>{
  await this.particles.initialize();const f=this.fields,d=f.device,solid=f.solid;
  const module=uniformDetailModule(d,{label:'Resident source termination',code:uniformMixedTopologyWGSL(f.ownership.capacity,0)+uniformMixedSolidWGSL(solid?2:undefined,solid?.coarse?.count)+levelSetFillWGSL+f.heat.promoteWGSL(1,6,7)+/* wgsl */`
struct Box {center:vec4f,extents:vec4f,rotation:vec4f}
struct Params {origin:vec4f,h:vec4f,counts:vec4u}
@group(1) @binding(0) var<storage,read> boxes:array<Box>;
@group(1) @binding(1) var<storage,read_write> work:array<f32>;
@group(1) @binding(2) var<storage,read_write> budget:array<f32>;
@group(1) @binding(3) var<uniform> params:Params;
@group(1) @binding(4) var field:texture_storage_3d<r32float,read_write>;
@group(1) @binding(5) var<storage,read_write> ledger:array<f32>;
fn umLoadVertex(p:vec3u)->f32{return textureLoad(field,vec3i(p)).x;}
${uniformMixedVertexSamplingWGSL}
fn rotate(p:vec3f,q:vec4f)->vec3f{return p+2.0*cross(q.xyz,cross(q.xyz,p)+q.w*p);}
fn terminatedPhi(value:f32,p:vec3u)->f32{
 let world=params.origin.xyz+(vec3f(p)+32.0*vec3f(umFramePage()))*params.h.xyz;var next=value;
 for(var i=0u;i<params.counts.x;i++){
  let b=boxes[i];let local=rotate(world-b.center.xyz,vec4f(-b.rotation.xyz,b.rotation.w));let q=abs(local)-b.extents.xyz;
  let distance=min(max(q.x,max(q.y,q.z)),0.0)+length(max(q,vec3f(0)))-b.extents.w;
  next=max(next,-select(distance,-distance,b.center.w!=0.0));
 }
 return next;
}
fn nextValue(p:vec3u)->f32{
 let owner=umVertexAuthority(p);if(owner.width==0u){return umVertexValue(p);}
 if(umVertexIsCanonical(p,owner)){return terminatedPhi(umLoadVertex(p),p);}
 let origin=umOrigin(owner);let f=vec3f(p-origin)/f32(owner.width);var result=0.0;
 for(var k=0u;k<8u;k++){let c=umCorner(k,2u);let q=origin+c*owner.width;result+=umVertexWeight(f,c)*terminatedPhi(umLoadVertex(q),q);}return result;
}
// Identical to native svOpenFill: partially occupied 4h owners integrate
// their trilinear field over 64 h capacities, including bound pore water.
fn openFill(v:array<f32,8>,origin:vec3i)->f32{
 var sum=0.0;for(var s=0u;s<64u;s++){
  let cell=vec3u(s&3u,(s>>2u)&3u,s>>4u);let open=umFreeCellCapacity(origin+vec3i(cell));if(open<=0.0){continue;}
  var c:array<f32,8>;for(var k=0u;k<8u;k++){let f=.25*(vec3f(cell)+vec3f(umCorner(k,2u)));c[k]=mix(mix(mix(v[0],v[1],f.x),mix(v[2],v[3],f.x),f.y),mix(mix(v[4],v[5],f.x),mix(v[6],v[7],f.x),f.y),f.z);}sum+=open*fill(c);
 }return sum;
}
fn ownerVolume(v:array<f32,8>,o:UMOwner)->f32{
 let p=vec3i(umOrigin(o));if(o.width==1u){return fill(v)*umFreeCellCapacity(p);}
 let capacity=umFreeTileCapacity(o.tile);if(capacity<.99999){return openFill(v,p);}return 64.0*capacity*fill(v);
}
fn jobs()->u32{return 64u*umCounts.x+64u*umResidentPageCount();}
@compute @workgroup_size(64) fn measure(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 for(var i=gid.x;i<jobs();i+=64u*groups.x){let o=umResidentAllOwner(vec3u(i,0,0));if(o.width==0u){continue;}
  var before:array<f32,8>;var after:array<f32,8>;for(var k=0u;k<8u;k++){let p=umOrigin(o)+o.width*umCorner(k,2u);before[k]=umVertexValue(p);after[k]=nextValue(p);}
  let delta=ownerVolume(after,o)-ownerVolume(before,o);work[o.index]=delta;
  if(delta!=0.0){rhPromote(umPageOfTile(o.tile));}
 }
}
@compute @workgroup_size(64) fn erase(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 for(var i=gid.x;i<jobs();i+=64u*groups.x){let o=umResidentAllOwner(vec3u(i,0,0));if(o.width==0u){continue;}
  for(var k=0u;k<8u;k++){let p=umOrigin(o)+o.width*umCorner(k,2u);let authority=umVertexAuthority(p);
   if(authority.index==o.index&&umVertexIsCanonical(p,authority)){let old=umLoadVertex(p);let next=terminatedPhi(old,p);if(next!=old){textureStore(field,vec3i(p),vec4f(next,0,0,0));}}
  }
 }
}
override stride:u32=1u;
@compute @workgroup_size(64) fn reduce(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 for(var job=gid.x;job<(params.counts.y+2u*stride-1u)/(2u*stride);job+=64u*groups.x){let i=2u*stride*job;let j=i+stride;if(j<params.counts.y){work[i]+=work[j];}}
}
@compute @workgroup_size(1) fn commit(){
 if(budget[1]!=1.0){umSupport[UM_RESIDENCY+1u]=1u;return;}let delta=work[0];let old=budget[0];budget[0]=old+delta;budget[${NB_VOLUME_COMPENSATION_FLOAT}]+=(old-budget[0])+delta;budget[3]=0.0;
 let physical=delta*params.h.x*params.h.y*params.h.z;let before=ledger[0];ledger[0]=before+physical;ledger[1]+=(before-ledger[0])+physical;
}
`});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join('\n'));
  const layout=d.createPipelineLayout({bindGroupLayouts:[f.ownership.bindLayout,this.layout,...(solid?[solid.tileLayout]:[])]});
  for(const entryPoint of ['measure','erase','commit'])this.pipelines.set(entryPoint,await uniformDetailPipeline(d,f.ownership,{layout,compute:{module,entryPoint}}));
  for(let stride=1;stride<f.ownership.capacity.owners;stride*=2)this.pipelines.set(`reduce${stride}`,await uniformDetailPipeline(d,f.ownership,{layout,compute:{module,entryPoint:'reduce',constants:{stride}}}));
 }
 prepare(fields:readonly WetClothSourceField[]):void{
  this.particles.prepare(fields);const roots=fields.filter(f=>f.root&&f.usage==='terminator');this.count=roots.length;this.bounds=[...sourceTerminationBoundaryBounds(fields,this.fields.cellSize_m)];this.shapes?.destroy();this.group=undefined;if(!roots.length)return;
  if(roots.some(f=>f.kind!=='box'))throw new Error('Unknown source terminator shape');
  const data=new Float32Array(12*roots.length);
  roots.forEach((f,i)=>{
   const axis=f.axis??[0,1,0],angle=f.angle_rad??0,n=Math.hypot(...axis),s=angle?Math.sin(angle/2)/n:0;
   data.set([...(f.center_m??[0,0,0]),f.inside?1:0,...f.halfExtents_m!,f.roundingRadius_m??0,axis[0]*s,axis[1]*s,axis[2]*s,Math.cos(angle/2)],12*i);

  });
  const d=this.fields.device;this.shapes=d.createBuffer({size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});d.queue.writeBuffer(this.shapes,0,data);
  const raw=new ArrayBuffer(48),floats=new Float32Array(raw),ints=new Uint32Array(raw);floats.set([...this.fields.origin_m,0,...this.fields.cellSize_m,0]);ints.set([roots.length,this.fields.ownership.capacity.owners,0,0],8);d.queue.writeBuffer(this.params,0,raw);
  this.group=uniformDetailGroup(d,{layout:this.layout,entries:[{binding:0,resource:{buffer:this.shapes}},{binding:1,resource:{buffer:this.work}},{binding:2,resource:{buffer:this.fields.waterBudget}},{binding:3,resource:{buffer:this.params}},{binding:4,resource:this.fields.phi},{binding:5,resource:{buffer:this.externalLedger}},{binding:6,resource:{buffer:this.fields.heat.tiers}},{binding:7,resource:{buffer:this.fields.heat.state}}]});
 }
 encode(encoder:GPUCommandEncoder):boolean{
  if(!this.count)return false;if(!this.group)throw new Error('Resident termination needs prepared source fields');
  encoder.clearBuffer(this.work);
  const run=(name:string,n=4096)=>{const pipeline=this.pipelines.get(name);if(!pipeline)throw new Error('Resident termination is not initialized');const p=encoder.beginComputePass({label:`Source termination ${name}`});p.setPipeline(pipeline);p.setBindGroup(0,this.fields.ownership.bindGroup);p.setBindGroup(1,this.group!.group);if(this.fields.solid)p.setBindGroup(2,this.fields.solid.tileGroup);p.dispatchWorkgroups(n);p.end();};
  run('measure');run('erase');for(let stride=1;stride<this.fields.ownership.capacity.owners;stride*=2)run(`reduce${stride}`,Math.min(4096,Math.ceil(this.fields.ownership.capacity.owners/(128*stride))));run('commit',1);this.particles.encode(encoder);this.fields.heat.encodeDeep(encoder);return true;
 }
 async readLedger():Promise<{externalVolume_m3:number}>{const d=this.fields.device,b=d.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});try{const e=d.createCommandEncoder();e.copyBufferToBuffer(this.externalLedger,0,b,0,16);d.queue.submit([e.finish()]);await b.mapAsync(GPUMapMode.READ);const f=new Float32Array(b.getMappedRange());return {externalVolume_m3:f[0]!+f[1]!};}finally{if(b.mapState==='mapped')b.unmap();b.destroy();}}
 destroy():void{this.particles.destroy();this.shapes?.destroy();this.params.destroy();this.work.destroy();this.externalLedger.destroy();}
}
