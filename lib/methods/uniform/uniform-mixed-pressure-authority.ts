import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformVolumeCorrectionWGSL } from "./uniform-volume-correction.wgsl";

export interface UniformMixedPressureAuthorityFields {
 centerPhi:GPUTexture;volume:GPUTexture;targetFill:GPUTexture;
 phi:GPUBufferBinding;
 /** Phase flag for momentum's supported physical donor gather. */
 phase:GPUTexture;
 correction:GPUTexture;
 scratch:GPUBufferBinding;
 /** dt, density fallback (-1/no deficit balance, 0/off, 1/isolated, 2/all), airborne, dust threshold. */
 params:GPUBuffer;
}
/** One pressure interface for RHS, projection, momentum support and extension.
 * Excess/deficit balance uses physical cell mass; coarse cells are not counted
 * as one fine cell. All fields and reduction scratch are caller-owned. */
export class UniformMixedPressureAuthority {
 readonly allocatedBytes=0;
 readonly scratchBytes:number;
 private readonly groups:number;
 private readonly chunks:number;
 private readonly resources:GPUBindGroupLayout;
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership){
  this.groups=Math.ceil(ownership.layout.cellCount/64);this.chunks=Math.ceil(this.groups/1024);
  this.scratchBytes=8*(1+this.groups+this.chunks);
  this.resources=device.createBindGroupLayout({entries:[
   ...[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
   ...[3,6].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}})),
   ...[4,5].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only" as const,format:"r32float" as const,viewDimension:"3d" as const}})),
   {binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
  ]});
 }
 bind(f:UniformMixedPressureAuthorityFields):GPUBindGroup{
  if((f.scratch.size??f.scratch.buffer.size-(f.scratch.offset??0))<this.scratchBytes)throw new Error("Mixed pressure authority scratch is too small");
  const d=this.ownership.layout.lattice.dimensions;
  for(const t of [f.centerPhi,f.volume,f.targetFill,f.phase,f.correction])if(t.format!=="r32float"||[t.width,t.height,t.depthOrArrayLayers].some((v,a)=>v!==d[a]))throw new Error("Mixed pressure authority requires native scalar fields");
  if([f.centerPhi,f.volume,f.targetFill].some(t=>t===f.phase||t===f.correction)||f.phase===f.correction)throw new Error("Mixed pressure authority outputs must be disjoint");
  return this.device.createBindGroup({layout:this.resources,entries:[
   ...[f.centerPhi,f.volume,f.targetFill].map((t,binding)=>({binding,resource:t.createView()})),
   {binding:3,resource:{...f.phi,size:4*this.ownership.layout.cellCount}},
   {binding:4,resource:f.phase.createView()},{binding:5,resource:f.correction.createView()},
   {binding:6,resource:{...f.scratch,size:this.scratchBytes}},{binding:7,resource:{buffer:f.params,size:16}},
  ]});
 }
 async initialize():Promise<void>{
  const module=this.device.createShaderModule({code:uniformMixedTopologyWGSL(this.ownership.layout,0)+/* wgsl */`
@group(1) @binding(0) var centerPhi:texture_3d<f32>;
@group(1) @binding(1) var volume:texture_3d<f32>;
@group(1) @binding(2) var targetFill:texture_3d<f32>;
@group(1) @binding(3) var<storage,read_write> phi:array<f32>;
@group(1) @binding(4) var phase:texture_storage_3d<r32float,write>;
@group(1) @binding(5) var correction:texture_storage_3d<r32float,write>;
@group(1) @binding(6) var<storage,read_write> balance:array<vec2f>;
@group(1) @binding(7) var<uniform> params:vec4f;
${uniformVolumeCorrectionWGSL}
fn umAuthority(o:UMOwner,v:f32)->f32{
 let h=${Math.min(...this.ownership.layout.lattice.cellSize_m)}*f32(o.width);
 var distance=textureLoad(centerPhi,vec3i(umOrigin(o)),0).x;
 if(params.z>0.5){distance=min(distance,max(h*(1.0-v),-0.5*h));}
 if(params.y<0.5){return distance;}
 let volumePhi=h*(0.5-v);if(params.y>1.5){return min(distance,volumePhi);}
 if(distance<0.0||volumePhi>=0.0){return distance;}
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let sign=select(-1,1,side==1u);let first=umFace(o,axis,sign,0u);
  for(var part=0u;part<first.count;part++){let neighbor=umFace(o,axis,sign,part).neighbor;
   if(neighbor.width!=0u&&textureLoad(centerPhi,vec3i(umOrigin(neighbor)),0).x<0.0){return distance;}}
 }}
 return max(volumePhi,-0.5*h);
}
fn umDeficit(o:UMOwner,v:f32,distance:f32)->f32{
 if(v>1.0||distance>=0.0){return 0.0;}
 return max(0.0,textureLoad(targetFill,vec3i(umOrigin(o)),0).x-v);
}
var<workgroup> sums:array<vec2f,64>;
fn umReduce(l:u32){workgroupBarrier();for(var stride=32u;stride>0u;stride/=2u){if(l<stride){sums[l]+=sums[l+stride];}workgroupBarrier();}}
@compute @workgroup_size(64) fn build(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) group:vec3u){
 let o=umAllOwner(gid);var values=vec2f(0);
 if(o.width!=0u){let origin=vec3i(umOrigin(o));let v=textureLoad(volume,origin,0).x;let distance=umAuthority(o,v);
  phi[o.index]=distance;
  let airborne=params.z>0.5&&v>max(params.w,0.05)&&textureLoad(centerPhi,origin,0).x>${1.5*Math.min(...this.ownership.layout.lattice.cellSize_m)}*f32(o.width)
   &&all(origin>=vec3i(i32(2u*o.width)))&&all(origin+vec3i(i32(3u*o.width))<=vec3i(UM_D));
  textureStore(phase,origin,vec4f(select(0.0,1.0,distance<0.0||airborne)));
  values=vec2f(uvVolumeCorrectionAmountAt(v,1.0,params.x),umDeficit(o,v,distance))*f32(o.width*o.width*o.width);
 }
 sums[l]=values;umReduce(l);let index=group.x+umDispatchX*group.y;
 if(l==0u&&index<${this.groups}u){balance[1u+index]=sums[0];}
}
@compute @workgroup_size(64) fn chunks(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) l:u32){
 var value=vec2f(0);for(var i=group.x*1024u+l;i<min((umCounts.x*64u+umCounts.y*8u+umCounts.z+63u)/64u,(group.x+1u)*1024u);i+=64u){value+=balance[1u+i];}
 sums[l]=value;umReduce(l);if(l==0u){balance[${1+this.groups}u+group.x]=sums[0];}
}
@compute @workgroup_size(64) fn reduce(@builtin(local_invocation_index) l:u32){
 var value=vec2f(0);for(var i=l;i<${this.chunks}u;i+=64u){value+=balance[${1+this.groups}u+i];}
 sums[l]=value;umReduce(l);if(l==0u){var rate=0.0;if(params.y>=0.0&&sums[0].y>0.0){rate=min(1.0,sums[0].x/sums[0].y);}balance[0]=vec2f(rate,0);}
}
@compute @workgroup_size(64) fn resolve(@builtin(global_invocation_id) gid:vec3u){
 let o=umAllOwner(gid);if(o.width==0u){return;}let origin=vec3i(umOrigin(o));let v=textureLoad(volume,origin,0).x;
 let amount=uvVolumeCorrectionAmountAt(v,1.0,params.x)-balance[0].x*umDeficit(o,v,phi[o.index]);
 textureStore(correction,origin,vec4f(amount/max(params.x,1e-12)));
}
`});
  const info=await module.getCompilationInfo(),errors=info.messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]});
  for(const entryPoint of ["build","chunks","reduce","resolve"])this.pipelines.set(entryPoint,await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX}}}));
 }
 encode(encoder:GPUCommandEncoder,group:GPUBindGroup):void{
  if(this.pipelines.size!==4)throw new Error("Mixed pressure authority is not initialized");
  const pass=encoder.beginComputePass({label:"Uniform mixed pressure authority and volume correction"});pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);
  for(const entry of ["build","chunks","reduce","resolve"]){const pipeline=this.pipelines.get(entry)!;pass.setPipeline(pipeline);
   if(entry==="chunks")pass.dispatchWorkgroups(this.chunks);else if(entry==="reduce")pass.dispatchWorkgroups(1);else this.ownership.dispatchAll(pass,pipeline);
  }
  pass.end();
 }
}
