import type {UniformMixedOwnership} from "./uniform-mixed-ownership";
import {UNIFORM_MIXED_COUNTED,uniformMixedCountedEntriesWGSL,uniformMixedTopologyWGSL} from "./uniform-mixed-topology.wgsl";
import {uniformMixedVertexSamplingSource} from "./uniform-mixed-vertex-sampling.wgsl";

/** On-demand native diagnostic receipts over canonical owners. All storage is
 * borrowed; inactive fine texels never contribute mass, front or velocity. */
export class UniformMixedDiagnostics {
 private pipeline?:GPUComputePipeline;
 private readonly resources:GPUBindGroupLayout;
 private readonly group:GPUBindGroup;
 constructor(private readonly device:GPUDevice,private readonly ownership:UniformMixedOwnership,
  volume:GPUTexture,velocity:GPUTexture,phi:GPUTexture,private readonly reductions:GPUBuffer){
  this.resources=device.createBindGroupLayout({entries:[
   ...[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
   {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
  ]});
  this.group=device.createBindGroup({layout:this.resources,entries:[
   ...[volume,velocity,phi].map((t,binding)=>({binding,resource:t.createView()})),
   {binding:3,resource:{buffer:reductions,offset:0,size:24}},
  ]});
 }
 async initialize():Promise<void>{
  const h=this.ownership.capacity.lattice.cellSize_m;
  // One lane per GPU-counted owner of every tier (umAllOwner), a partial per job.
  const module=this.device.createShaderModule({code:uniformMixedCountedEntriesWGSL(uniformMixedTopologyWGSL(this.ownership.capacity,0)+/* wgsl */`
@group(1) @binding(0) var volume:texture_3d<f32>;
@group(1) @binding(1) var velocity:texture_3d<f32>;
@group(1) @binding(2) var phi:texture_3d<f32>;
@group(1) @binding(3) var<storage,read_write> totals:array<atomic<u32>>;
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
${uniformMixedVertexSamplingSource("",false)}
var<workgroup> lanes:array<vec4u,64>;
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
    positive[axis]+=textureLoad(velocity,face.anchor,0)[axis]*f32(face.width*face.width)/f32(owner.width*owner.width);
   }
  }
  let represented=clamp(0.5-center/(4.0*w*${h[1]}),0.0,1.0);
  receipt=vec4u(u32(represented*weight*2048.0+0.5),select(0u,q.x+owner.width,center<0.0),
   bitcast<u32>(length(positive)),u32(max(0.0,textureLoad(volume,vec3i(q),0).x)*weight*2048.0+0.5));
 }
 lanes[lane]=receipt;workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){
  if(lane<stride){let a=lanes[lane];let b=lanes[lane+stride];lanes[lane]=vec4u(a.x+b.x,max(a.y,b.y),max(a.z,b.z),a.w+b.w);}workgroupBarrier();
 }
 if(lane==0u){let r=lanes[0];atomicAdd(&totals[0],r.x);atomicMax(&totals[1],r.y);atomicMax(&totals[2],r.z);atomicAdd(&totals[3],r.w);}
}`,["diagnostics"])});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>m.message).join("\n"));
  this.pipeline=await this.device.createComputePipelineAsync({layout:this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]}),compute:{module,entryPoint:"diagnostics",constants:{umDispatchX:this.ownership.dispatchX,umCountedJobs:UNIFORM_MIXED_COUNTED.all}}});
 }
 encode(encoder:GPUCommandEncoder):void{
  if(!this.pipeline)throw new Error("Mixed diagnostics are not initialized");
  encoder.clearBuffer(this.reductions,0,24);
  const pass=encoder.beginComputePass({label:"Uniform canonical diagnostics"});pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.group);
  this.ownership.dispatchAllCounted(pass,this.pipeline);pass.end();
 }
}
