import { uniformDetailBindLayout, uniformDetailModule, uniformDetailGroup, type UniformDetailGroup } from "./uniform-detail-fields";
import type {UniformMixedOwnership} from "./uniform-mixed-ownership";
import {UNIFORM_DETAIL_CANONICAL_LOAD} from "../../core/uniform-detail-abi";
import {UNIFORM_MIXED_COUNTED,uniformMixedCountedEntriesWGSL,uniformMixedTopologyWGSL} from "./uniform-mixed-topology.wgsl";
import {uniformMixedVertexSamplingSource} from "./uniform-mixed-vertex-sampling.wgsl";

/** On-demand native diagnostic receipts over canonical owners. The fields
 * are borrowed; inactive fine texels never contribute mass, front or velocity.
 * totals (UNIFORM_MIXED_DIAGNOSTIC_WORDS): represented and fill volume in
 * 1/2048 cells, each with a carry word (6, 7): a 2048-scaled sum passes 2³²
 * at about 2.1M cells, so a single word wraps on any large pool. */
export const UNIFORM_MIXED_DIAGNOSTIC_WORDS=8;
export class UniformMixedDiagnostics {
 private pipeline?:GPUComputePipeline;
 /** 0 represented (low word), 1 front, 2 max speed, 3 fill (low word), 4-5 two-level tiles, 6-7 the carries of 0 and 3. */
 readonly totals:GPUBuffer;
 private readonly resources:GPUBindGroupLayout;
 private readonly group:UniformDetailGroup;
 constructor(private readonly device:GPUDevice,private readonly ownership:UniformMixedOwnership,
  volume:GPUTexture,velocity:GPUTexture,phi:GPUTexture){
  this.totals=device.createBuffer({label:"Uniform mixed diagnostics totals",size:4*UNIFORM_MIXED_DIAGNOSTIC_WORDS,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  this.resources=uniformDetailBindLayout(device,{entries:[
   ...[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
   {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
  ]});
  this.group=uniformDetailGroup(device,{layout:this.resources,entries:[
   ...[volume,velocity,phi].map((t,binding)=>({binding,resource:t})),
   {binding:3,resource:{buffer:this.totals}},
  ]});
 }
 async initialize():Promise<void>{
  const h=this.ownership.capacity.lattice.cellSize_m;
  // One lane per GPU-counted owner of every tier (umAllOwner), a partial per job.
  const module=uniformDetailModule(this.device,{label:"Uniform mixed diagnostics",code:uniformMixedCountedEntriesWGSL(uniformMixedTopologyWGSL(this.ownership.capacity,0)+/* wgsl */`
@group(1) @binding(0) var volume:texture_3d<f32>;
@group(1) @binding(1) var velocity:texture_3d<f32>;
@group(1) @binding(2) var phi:texture_3d<f32>;
@group(1) @binding(3) var<storage,read_write> totals:array<atomic<u32>>;
fn umLoadVertex(p:vec3u)->f32{return ${UNIFORM_DETAIL_CANONICAL_LOAD}textureLoad(phi,vec3i(p),0).x;}
${uniformMixedVertexSamplingSource("",false)}
var<workgroup> lanes:array<vec4u,64>;
// A fixed-point sum into a low word, carried into its high word.
fn addWide(low:u32,high:u32,value:u32){let old=atomicAdd(&totals[low],value);if(old+value<old){atomicAdd(&totals[high],1u);}}
@compute @workgroup_size(64) fn diagnostics(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 let owner=umAllOwner(gid);var receipt=vec4u(0);
 if(owner.width!=0u){
  if(owner.lane==0u){let support=umTileSupport(owner.tile);atomicAdd(&totals[4],support&1u);atomicAdd(&totals[5],(support>>1u)&1u);}
  let q=umOrigin(owner);let w=f32(owner.width);let weight=w*w*w;
  let center=umSampleVertex(vec3f(q)+vec3f(0.5*w));
  var positive=vec3f(0);
  for(var axis=0u;axis<3u;axis++){
   let first=umFace(owner,axis,1,0u);
   for(var part=0u;part<first.count;part++){
    let face=umFace(owner,axis,1,part);
    positive[axis]+=${UNIFORM_DETAIL_CANONICAL_LOAD}textureLoad(velocity,face.anchor,0)[axis]*f32(face.width*face.width)/f32(owner.width*owner.width);
   }
  }
  let represented=clamp(0.5-center/(4.0*w*umSceneSpacing().y),0.0,1.0);
  receipt=vec4u(u32(represented*weight*2048.0+0.5),select(0u,q.x+owner.width,center<0.0),
   bitcast<u32>(length(positive)),u32(max(0.0,${UNIFORM_DETAIL_CANONICAL_LOAD}textureLoad(volume,vec3i(q),0).x)*weight*2048.0+0.5));
 }
 lanes[lane]=receipt;workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){
  if(lane<stride){let a=lanes[lane];let b=lanes[lane+stride];lanes[lane]=vec4u(a.x+b.x,max(a.y,b.y),max(a.z,b.z),a.w+b.w);}workgroupBarrier();
 }
 if(lane==0u){let r=lanes[0];addWide(0u,6u,r.x);atomicMax(&totals[1],r.y);atomicMax(&totals[2],r.z);addWide(3u,7u,r.w);}
}`,["diagnostics"])});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>m.message).join("\n"));
  this.pipeline=await this.device.createComputePipelineAsync({layout:this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]}),compute:{module,entryPoint:"diagnostics",constants:{umDispatchX:this.ownership.dispatchX,umCountedJobs:UNIFORM_MIXED_COUNTED.all}}});
 }
 encode(encoder:GPUCommandEncoder):void{
  if(!this.pipeline)throw new Error("Mixed diagnostics are not initialized");
  encoder.clearBuffer(this.totals);
  const pass=encoder.beginComputePass({label:"Uniform canonical diagnostics"});pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.group.group);
  this.ownership.dispatchAllCounted(pass,this.pipeline);pass.end();
 }
 destroy():void{this.totals.destroy();}
}
