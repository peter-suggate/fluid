import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedFaceAddressWGSL } from "./uniform-mixed-face-dispatch.wgsl";
import { UNIFORM_MIXED_HANGING_TAPS, uniformMixedHangingTapWGSL, uniformMixedVelocitySamplingSource } from "./uniform-mixed-velocity-sampling.wgsl";

export interface UniformMixedMomentumCacheFields {
  extended:GPUTexture;
  physical:GPUTexture;
  phase:GPUTexture;
  negative:GPUBuffer;
  coarseExtended:GPUTexture;
  coarsePhysical:GPUTexture;
  coarseWeight:GPUTexture;
}
/** 4h MAC sampling restriction, including negative domain faces. These are
 * interpolation cache values, never an independently advanced velocity grid.
 * The three borrowed RGBA fields have extent D/4+2 and may reuse prepared
 * coarse resources. Fine/transition patches are averaged once per update. */
export class UniformMixedMomentumCache {
  readonly allocatedBytes=0;
  private pipeline?:GPUComputePipeline;
  private readonly resources:GPUBindGroupLayout;
  constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership){
    this.resources=device.createBindGroupLayout({entries:[
      ...[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
      {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      ...[4,5,6].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only" as const,format:"rgba32float" as const,viewDimension:"3d" as const}})),
    ]});
  }
  bind(f:UniformMixedMomentumCacheFields):GPUBindGroup{
    const d=this.ownership.layout.lattice.dimensions;
    for(const [i,t] of [f.extended,f.physical,f.phase,f.coarseExtended,f.coarsePhysical,f.coarseWeight].entries())
      if([t.width,t.height,t.depthOrArrayLayers].some((n,a)=>n!==(i<3?d[a]:d[a]!/4+2))||t.format!==(i===2?"r32float":"rgba32float"))throw new Error("Mixed momentum cache requires native inputs and 4h halo outputs");
    if(new Set([f.coarseExtended,f.coarsePhysical,f.coarseWeight]).size!==3)throw new Error("Mixed momentum cache outputs must be disjoint");
    return this.device.createBindGroup({layout:this.resources,entries:[
      ...[f.extended,f.physical,f.phase].map((t,binding)=>({binding,resource:t.createView()})),
      {binding:3,resource:{buffer:f.negative}},
      ...[f.coarseExtended,f.coarsePhysical,f.coarseWeight].map((t,i)=>({binding:4+i,resource:t.createView()})),
    ]});
  }
  async initialize():Promise<void>{
    const module=this.device.createShaderModule({code:uniformMixedTopologyWGSL(this.ownership.layout,0)+/* wgsl */`
@group(1) @binding(0) var extended:texture_3d<f32>;
@group(1) @binding(1) var physical:texture_3d<f32>;
@group(1) @binding(2) var phase:texture_3d<f32>;
@group(1) @binding(3) var<storage,read> negative:array<f32>;
@group(1) @binding(4) var coarseExtended:texture_storage_3d<rgba32float,write>;
@group(1) @binding(5) var coarsePhysical:texture_storage_3d<rgba32float,write>;
@group(1) @binding(6) var coarseWeight:texture_storage_3d<rgba32float,write>;
${uniformMixedFaceAddressWGSL}
fn umCachePayload(anchor:vec3i,axis:u32)->vec3f {
 var next=anchor;next[axis]+=1;let left=umOwnerAt(anchor);let right=umOwnerAt(next);var supported=false;
 if(left.width!=0u){supported=textureLoad(phase,vec3i(umOrigin(left)),0).x>0.5;}
 if(right.width!=0u){supported=supported||textureLoad(phase,vec3i(umOrigin(right)),0).x>0.5;}
 var e=0.0;var p=0.0;
 if(anchor[axis]<0){e=negative[umNegativeBoundaryIndex(vec3u(max(anchor,vec3i(0))),axis)];if(supported){p=e;}}
 else{e=textureLoad(extended,anchor,0)[axis];if(supported){p=textureLoad(physical,anchor,0)[axis];}}
 return vec3f(e,p,select(0.0,1.0,supported));
}
@compute @workgroup_size(4,4,4) fn cache(@builtin(global_invocation_id) gid:vec3u){
 if(any(gid>=UM_T+vec3u(2))){return;}let id=vec3i(gid)-vec3i(1);
 var e=vec4f(0);var p=vec4f(0);var weight=vec4f(0);
 for(var axis=0u;axis<3u;axis++){
  let u=(axis+1u)%3u;let v=(axis+2u)%3u;
  if(id[axis]< -1||id[axis]>=i32(UM_T[axis])||id[u]<0||id[u]>=i32(UM_T[u])||id[v]<0||id[v]>=i32(UM_T[v])){continue;}
  var anchor=id*4;anchor[axis]+=3;
  var owner=umOwnerAt(anchor);var sign=1;
  if(owner.width==0u){var next=anchor;next[axis]+=1;owner=umOwnerAt(next);sign=-1;}
  let width=umFace(owner,axis,sign,0u).width;let side=4u/width;var value=vec3f(0);
  for(var y=0u;y<side;y++){for(var x=0u;x<side;x++){
   var q=anchor;q[u]+=i32(x*width);q[v]+=i32(y*width);value+=umCachePayload(q,axis);
  }}value/=f32(side*side);e[axis]=value.x;p[axis]=value.y;weight[axis]=value.z;
 }
 textureStore(coarseExtended,vec3i(gid),e);textureStore(coarsePhysical,vec3i(gid),p);textureStore(coarseWeight,vec3i(gid),weight);
}`});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
    this.pipeline=await this.device.createComputePipelineAsync({layout:this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]}),compute:{module,entryPoint:"cache"}});
  }
  encode(encoder:GPUCommandEncoder,group:GPUBindGroup):void{
    if(!this.pipeline)throw new Error("Mixed momentum cache is not initialized");
    const pass=encoder.beginComputePass({label:"Uniform mixed 4h sampling cache"});pass.setPipeline(this.pipeline);pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);
    pass.dispatchWorkgroups(...this.ownership.layout.lattice.dimensions.map(n=>Math.ceil((n/4+2)/4)) as [number,number,number]);pass.end();
  }
}

export interface UniformMixedHangingTapFields {
  extended:GPUTexture;
  negative:GPUBuffer;
  /** Current 4h sampling cache (coarseExtended). */
  coarse:GPUTexture;
}
/** Every fine MAC tap whose cell lies in a 2h tile resolves through nested 2h
 * and 4h interpolation. Surface and momentum characteristics revisit those
 * taps many times per frame, so evaluate each once here with the unchanged
 * sampler and store it in ownership.hangingGroup. Valid from this encode until
 * velocity/negative/coarse inputs change (forces); the frame orders that. */
export class UniformMixedHangingTaps {
  readonly allocatedBytes=0;
  private pipelines:GPUComputePipeline[]=[];
  private readonly resources:GPUBindGroupLayout;
  constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership){
    this.resources=device.createBindGroupLayout({entries:[
      {binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
      {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:2,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
    ]});
  }
  bind(f:UniformMixedHangingTapFields):GPUBindGroup{
    const d=this.ownership.layout.lattice.dimensions;
    for(const [i,t] of [f.extended,f.coarse].entries())
      if([t.width,t.height,t.depthOrArrayLayers].some((n,a)=>n!==(i===0?d[a]:d[a]!/4+2))||t.format!=="rgba32float")throw new Error("Hanging taps require native velocity and the 4h sampling cache");
    return this.device.createBindGroup({layout:this.resources,entries:[
      {binding:0,resource:f.extended.createView()},{binding:1,resource:{buffer:f.negative}},{binding:2,resource:f.coarse.createView()},
    ]});
  }
  async initialize():Promise<void>{
    // Two phases: every slot's 2h taps, then its fine taps, whose nested 2h
    // interpolation reads those memoized 2h taps (the same function's values)
    // instead of re-evaluating eight 2h taps per fine tap.
    const source=(fine:boolean)=>uniformMixedTopologyWGSL(this.ownership.layout,0)+/* wgsl */`
@group(1) @binding(0) var extended:texture_3d<f32>;
@group(1) @binding(1) var<storage,read> negative:array<f32>;
@group(1) @binding(2) var coarse:texture_3d<f32>;
${uniformMixedFaceAddressWGSL}
fn umLoadMixedFace(anchor:vec3i,axis:u32)->f32{
 if(anchor[axis]<0){return negative[umNegativeBoundaryIndex(vec3u(max(anchor,vec3i(0))),axis)];}
 return textureLoad(extended,anchor,0)[axis];
}
fn umLoadCoarseFace(index:vec3i,axis:u32)->f32{return textureLoad(coarse,index+vec3i(1),0)[axis];}
${fine?uniformMixedVelocitySamplingSource(false,true,undefined,2,false):uniformMixedHangingTapWGSL(2)+uniformMixedVelocitySamplingSource(false,true)}
// Tile-local lanes: 0..191 fine taps, 192..215 2h taps, 216..263 fine and
// 264..275 2h negative boundary plane taps (for tiles on that plane).
fn umHangingFill(slot:u32,tile:u32,lane:u32){
 if(lane<192u){
  let cell=lane%64u;let axis=lane/64u;let local=vec3u(cell%4u,(cell/4u)%4u,cell/16u);
  umHanging[umHangingAddress(slot,local,axis)]=bitcast<u32>(umVelocityTap1(vec3i(umTileCoord(tile)*4u+local),axis));
 }else if(lane<216u){
  let cell=(lane-192u)%8u;let axis=(lane-192u)/8u;let local=vec3u(cell%2u,(cell/2u)%2u,cell/4u);
  umHanging[umHangingAddress2(slot,local,axis)]=bitcast<u32>(umVelocityTap2(vec3i(umTileCoord(tile)*2u+local),axis));
 }else if(lane<${UNIFORM_MIXED_HANGING_TAPS}u){
  let fine=lane<264u;let width=select(2u,1u,fine);let side=4u/width;
  let offset=select(lane-264u,lane-216u,fine);let axis=offset/(side*side);let cell=offset%(side*side);
  if(umTileCoord(tile)[axis]!=0u){return;}
  var local=vec3u(0);local[(axis+1u)%3u]=cell%side;local[(axis+2u)%3u]=cell/side;
  var index=vec3i(umTileCoord(tile)*side+local);index[axis]=-1;
  var value=0.0;if(fine){value=umVelocityTap1(index,axis);}else{value=umVelocityTap2(index,axis);}
  umHanging[umHangingPlaneAddress(slot,local,axis,width)]=bitcast<u32>(value);
 }
}
@compute @workgroup_size(${fine?256:64}) fn hanging(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let slot=group.x+umDispatchX*group.y;if(slot>=UM_TILES){return;}
 let tile=umHanging[UM_TILES+slot];if(tile==UM_NO_SLOT){return;}
 ${fine?"if(lane<192u){umHangingFill(slot,tile,lane);}else if(lane<240u){umHangingFill(slot,tile,lane+24u);}":"if(lane<24u){umHangingFill(slot,tile,lane+192u);}else if(lane<36u){umHangingFill(slot,tile,lane+240u);}"}
}`;
    const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources,this.ownership.hangingLayout]});
    this.pipelines=await Promise.all([false,true].map(async fine=>{
      const module=this.device.createShaderModule({code:source(fine)});
      const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
      return this.device.createComputePipelineAsync({layout,compute:{module,entryPoint:"hanging",constants:{umDispatchX:this.ownership.dispatchX}}});
    }));
  }
  encode(encoder:GPUCommandEncoder,group:GPUBindGroup):void{
    if(this.pipelines.length!==2)throw new Error("Mixed hanging taps are not initialized");
    const groups=this.ownership.hangingSlots;if(!groups)return;
    const pass=encoder.beginComputePass({label:"Uniform mixed hanging fine taps"});pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);pass.setBindGroup(2,this.ownership.hangingGroup);
    for(const pipeline of this.pipelines){pass.setPipeline(pipeline);pass.dispatchWorkgroups(Math.min(groups,this.ownership.dispatchX),Math.ceil(groups/this.ownership.dispatchX));}
    pass.end();
  }
}
