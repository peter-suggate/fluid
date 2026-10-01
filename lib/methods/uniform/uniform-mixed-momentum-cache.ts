import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedCountedEntriesWGSL, uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedFaceAddressWGSL } from "./uniform-mixed-face-dispatch.wgsl";
import { UNIFORM_MIXED_HANGING_TAPS, uniformMixedHangingTapWGSL, uniformMixedVelocitySamplingSource } from "./uniform-mixed-velocity-sampling.wgsl";

export interface UniformMixedMomentumCacheFields {
  extended:GPUTexture;
  negative:GPUBuffer;
  coarseExtended:GPUTexture;
}
/** 4h MAC sampling restriction of the extended velocity, including negative
 * domain faces. These are interpolation cache values, never an independently
 * advanced velocity grid. The borrowed RGBA output has extent D/4+2 and may
 * reuse prepared coarse resources. Fine patches are averaged once per update. */
export class UniformMixedMomentumCache {
  readonly allocatedBytes=0;
  private pipeline?:GPUComputePipeline;
  private readonly resources:GPUBindGroupLayout;
  constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership){
    this.resources=device.createBindGroupLayout({entries:[
      {binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
      {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:2,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"3d"}},
    ]});
  }
  bind(f:UniformMixedMomentumCacheFields):GPUBindGroup{
    const d=this.ownership.capacity.lattice.dimensions;
    for(const [i,t] of [f.extended,f.coarseExtended].entries())
      if([t.width,t.height,t.depthOrArrayLayers].some((n,a)=>n!==(i===0?d[a]:d[a]!/4+2))||t.format!=="rgba32float")throw new Error("Mixed momentum cache requires native inputs and 4h halo outputs");
    return this.device.createBindGroup({layout:this.resources,entries:[
      {binding:0,resource:f.extended.createView()},
      {binding:1,resource:{buffer:f.negative}},
      {binding:2,resource:f.coarseExtended.createView()},
    ]});
  }
  async initialize():Promise<void>{
    const module=this.device.createShaderModule({code:uniformMixedTopologyWGSL(this.ownership.capacity,0)+/* wgsl */`
@group(1) @binding(0) var extended:texture_3d<f32>;
@group(1) @binding(1) var<storage,read> negative:array<f32>;
@group(1) @binding(2) var coarseExtended:texture_storage_3d<rgba32float,write>;
${uniformMixedFaceAddressWGSL}
fn umCacheExtended(anchor:vec3i,axis:u32)->f32 {
 if(anchor[axis]<0){return negative[umNegativeBoundaryIndex(vec3u(max(anchor,vec3i(0))),axis)];}
 return textureLoad(extended,anchor,0)[axis];
}
@compute @workgroup_size(4,4,4) fn cache(@builtin(global_invocation_id) gid:vec3u){
 if(any(gid>=UM_T+vec3u(2))){return;}let id=vec3i(gid)-vec3i(1);
 var e=vec4f(0);
 for(var axis=0u;axis<3u;axis++){
  let u=(axis+1u)%3u;let v=(axis+2u)%3u;
  if(id[axis]< -1||id[axis]>=i32(UM_T[axis])||id[u]<0||id[u]>=i32(UM_T[u])||id[v]<0||id[v]>=i32(UM_T[v])){continue;}
  var anchor=id*4;anchor[axis]+=3;
  var owner=umOwnerAt(anchor);var sign=1;
  if(owner.width==0u){var next=anchor;next[axis]+=1;owner=umOwnerAt(next);sign=-1;}
  let width=umFace(owner,axis,sign,0u).width;let side=4u/width;var value=0.0;
  for(var y=0u;y<side;y++){for(var x=0u;x<side;x++){
   var q=anchor;q[u]+=i32(x*width);q[v]+=i32(y*width);value+=umCacheExtended(q,axis);
  }}e[axis]=value/f32(side*side);
 }
 textureStore(coarseExtended,vec3i(gid),e);
}`});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
    this.pipeline=await this.device.createComputePipelineAsync({layout:this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]}),compute:{module,entryPoint:"cache"}});
  }
  encode(encoder:GPUCommandEncoder,group:GPUBindGroup):void{
    if(!this.pipeline)throw new Error("Mixed momentum cache is not initialized");
    const pass=encoder.beginComputePass({label:"Uniform mixed 4h sampling cache"});pass.setPipeline(this.pipeline);pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);
    pass.dispatchWorkgroups(...this.ownership.capacity.lattice.dimensions.map(n=>Math.ceil((n/4+2)/4)) as [number,number,number]);pass.end();
  }
}

export interface UniformMixedHangingTapFields {
  extended:GPUTexture;
  negative:GPUBuffer;
  /** Current 4h sampling cache (coarseExtended). */
  coarse:GPUTexture;
}
/** Every fine MAC tap whose cell lies in a seam tile resolves through the
 * mixed h/4h sampler. Surface and momentum characteristics revisit those
 * taps many times per frame, so evaluate each once here with the unchanged
 * sampler and store it in ownership.hangingGroup. Valid from this encode until
 * velocity/negative/coarse inputs change (forces); the frame orders that. */
export class UniformMixedHangingTaps {
  readonly allocatedBytes:number;
  /** The unit interpolant's taps as one native MAC texel per cell: each unit
   * tile's stored faces and each slotted tile's fine taps, written with the
   * slots. A sample with nonzero fine weight has every tap in a unit or a
   * slotted tile (a tap within 4h of a unit tile is in a tile beside it, and
   * a coarse tile beside a unit tile is a seam tile), so its unit
   * interpolant is eight direct loads, as all-fine Uniform samples. */
  readonly unitVelocity:GPUTexture;
  private pipeline?:GPUComputePipeline;
  private readonly resources:GPUBindGroupLayout;
  constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership){
    this.resources=device.createBindGroupLayout({entries:[
      {binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
      {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:2,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
      {binding:3,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"3d"}},
    ]});
    const [x,y,z]=ownership.capacity.lattice.dimensions;
    this.unitVelocity=device.createTexture({label:"Uniform mixed unit velocity taps",size:[x!,y!,z!],dimension:"3d",format:"rgba32float",usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING});
    this.allocatedBytes=16*x!*y!*z!;
  }
  bind(f:UniformMixedHangingTapFields):GPUBindGroup{
    const d=this.ownership.capacity.lattice.dimensions;
    for(const [i,t] of [f.extended,f.coarse].entries())
      if([t.width,t.height,t.depthOrArrayLayers].some((n,a)=>n!==(i===0?d[a]:d[a]!/4+2))||t.format!=="rgba32float")throw new Error("Hanging taps require native velocity and the 4h sampling cache");
    return this.device.createBindGroup({layout:this.resources,entries:[
      {binding:0,resource:f.extended.createView()},{binding:1,resource:{buffer:f.negative}},{binding:2,resource:f.coarse.createView()},
      {binding:3,resource:this.unitVelocity.createView()},
    ]});
  }
  async initialize():Promise<void>{
    // One GPU-counted launch: each h tile's stored faces, four tiles per job,
    // then one job per seam 4h slot filling its fine taps (the ungraded h/4h
    // layout has no 2h taps to memoize first). A seam h tile's slot is never
    // filled: umVelocityTap1 returns an h tile's stored face before it looks
    // for a slot, so only 4h slots are read.
    const code=uniformMixedCountedEntriesWGSL(uniformMixedTopologyWGSL(this.ownership.capacity,0)+/* wgsl */`
@group(1) @binding(0) var extended:texture_3d<f32>;
@group(1) @binding(1) var<storage,read> negative:array<f32>;
@group(1) @binding(2) var coarse:texture_3d<f32>;
@group(1) @binding(3) var unitVelocity:texture_storage_3d<rgba32float,write>;
${uniformMixedFaceAddressWGSL}
fn umLoadMixedFace(anchor:vec3i,axis:u32)->f32{
 if(anchor[axis]<0){return negative[umNegativeBoundaryIndex(vec3u(max(anchor,vec3i(0))),axis)];}
 return textureLoad(extended,anchor,0)[axis];
}
fn umLoadCoarseFace(index:vec3i,axis:u32)->f32{return textureLoad(coarse,index+vec3i(1),0)[axis];}
${uniformMixedHangingTapWGSL(2)+uniformMixedVelocitySamplingSource(false,true)}
// Tile-local lanes: 0..191 fine taps (stored to unitVelocity below),
// 192..239 fine negative boundary plane taps (for tiles on that plane).
var<workgroup> unitTaps:array<f32,192>;
fn umHangingFill(slot:u32,tile:u32,lane:u32){
 if(lane<192u){
  let cell=lane%64u;let axis=lane/64u;let local=vec3u(cell%4u,(cell/4u)%4u,cell/16u);
  let value=umVelocityTap1(vec3i(umTileCoord(tile)*4u+local),axis);
  unitTaps[lane]=value;
 }else if(lane<${192+UNIFORM_MIXED_HANGING_TAPS}u){
  let offset=lane-192u;let axis=offset/16u;let cell=offset%16u;
  if(umTileCoord(tile)[axis]!=0u){return;}
  var local=vec3u(0);local[(axis+1u)%3u]=cell%4u;local[(axis+2u)%3u]=cell/4u;
  var index=vec3i(umTileCoord(tile)*4u+local);index[axis]=-1;
  umHanging[umHangingPlaneAddress(slot,local,axis)]=bitcast<u32>(umVelocityTap1(index,axis));
 }
}
// Seam 4h slots follow the seam h slots (the builder's and update()'s order);
// slots past the cache are never visited (the builder flags UM_OVERFLOW_HANGING).
fn umUnitJobs()->u32{return (umCounts.x+3u)/4u;}
fn umCoarseSlots()->vec2u{let header=7u*UM_TILES+16u;let fine=umSupport[header];return vec2u(min(fine,UM_HANGING_SLOTS),min(fine+umSupport[header+1u],UM_HANGING_SLOTS));}
@compute @workgroup_size(256) fn unitVelocityTaps(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let units=umUnitJobs();
 if(group.x<units){
  let k=4u*group.x+lane/64u;
  if(k<umCounts.x){let cell=vec3i(umTileCoord(umTopology[UM_TILES+k])*4u+umCorner(lane%64u,4u));textureStore(unitVelocity,cell,textureLoad(extended,cell,0));}
 }else{
  let slot=umCoarseSlots().x+group.x-units;let tile=umHanging[UM_TILES+slot];
  if(tile!=UM_NO_SLOT){umHangingFill(slot,tile,lane);}
  workgroupBarrier();
  if(tile!=UM_NO_SLOT&&lane<64u){textureStore(unitVelocity,vec3i(umTileCoord(tile)*4u+umCorner(lane,4u)),vec4f(unitTaps[lane],unitTaps[lane+64u],unitTaps[lane+128u],0.0));}
 }
}`,["unitVelocityTaps"],"umUnitJobs()+umCoarseSlots().y-umCoarseSlots().x");
    const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources,this.ownership.hangingLayout]});
    const module=this.device.createShaderModule({code});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
    this.pipeline=await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint:"unitVelocityTaps",constants:{umDispatchX:this.ownership.dispatchX}}});
  }
  encode(encoder:GPUCommandEncoder,group:GPUBindGroup):void{
    if(!this.pipeline)throw new Error("Mixed hanging taps are not initialized");
    // A fixed grid from capacity; the GPU counts the h tiles (forces
    // viscosity reads every unit tile's texels, so they are refreshed even
    // without a slot) and the seam 4h slots. At most one job per tile.
    const pass=encoder.beginComputePass({label:"Uniform mixed hanging fine taps"});pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);pass.setBindGroup(2,this.ownership.hangingGroup);
    this.ownership.dispatchCounted(pass,this.pipeline);
    pass.end();
  }
  destroy():void{this.unitVelocity.destroy();}
}
