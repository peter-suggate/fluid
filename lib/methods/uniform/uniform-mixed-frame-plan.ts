import {UNIFORM_DETAIL_4H_LOAD} from "../../core/uniform-detail-abi";
import { uniformDetailBindLayout, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, uniformDetailGroup, type UniformDetailGroup } from "./uniform-detail-fields";
import type {UniformMixedOwnership} from "./uniform-mixed-ownership";
import {UNIFORM_MIXED_COUNTED,uniformMixedCountedEntriesWGSL,uniformMixedTopologyWGSL} from "./uniform-mixed-topology.wgsl";
import {uniformMixedVertexSamplingSource} from "./uniform-mixed-vertex-sampling.wgsl";
import {UNIFORM_STAGE_CERTIFICATE} from "./uniform-stage-grids";

/** Owner entries: one lane per owner of every tier, GPU-counted. */
const counted:readonly string[]=["seed","localSpeed"];

/** Start-of-frame support census, independent of simulation ownership. The
 * native fine sampling reach is two tiles; extension gets one additional tile.
 * All stages share the resulting immutable support planes until the next frame.
 * A source/edit must be applied before this census, just as in the native host.
 * This is sampling/extension support, not permission to skip transported mass. */
export class UniformMixedFramePlan {
  readonly allocatedBytes=32;
  private readonly params:GPUBuffer;
  private readonly resources:GPUBindGroupLayout;
  private readonly group:UniformDetailGroup;
  private readonly extendedResources:GPUBindGroupLayout;
  private readonly extendedGroup:UniformDetailGroup;
  private readonly pipelines=new Map<string,GPUComputePipeline>();
  constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,volume:GPUTexture,phi:GPUTexture,velocity:GPUTexture,negative:GPUBuffer,
    /** The extended field every characteristic samples, valid at encodeCertificate. */
    extended:GPUTexture,extendedNegative:GPUBuffer,directionalCertificate=false){
    if(!directionalCertificate)throw new Error("The mixed frame plan certifies signed per-axis reach only");
    this.params=device.createBuffer({label:"Uniform shared support policy",size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.resources=uniformDetailBindLayout(device,{entries:[...[0,1,3].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}}]});
    this.group=uniformDetailGroup(device,{layout:this.resources,entries:[...[volume,phi].map((t,binding)=>({binding,resource:t})),{binding:2,resource:{buffer:this.params}},{binding:3,resource:velocity},{binding:4,resource:{buffer:negative}}]});
    this.extendedResources=uniformDetailBindLayout(device,{entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
      {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
    this.extendedGroup=uniformDetailGroup(device,{layout:this.extendedResources,entries:[{binding:0,resource:extended},{binding:1,resource:{buffer:extendedNegative}},{binding:2,resource:{buffer:ownership.speeds}}]});
  }
  async initialize():Promise<void>{
    const h=this.ownership.capacity.lattice.cellSize_m;
    // The generator reads the lattice and tile count only (fixed by capacity).
    const topology=uniformMixedTopologyWGSL(this.ownership.capacity,0).replace('umSupport:array<u32>','umSupport:array<atomic<u32>>').replace(/umSupport\[([^\]]+)\]/g,'atomicLoad(&umSupport[$1])');
    // seed and localSpeed stride the GPU-counted owners of every tier
    // (umAllOwner); the per-tile passes cover the lattice.
    const module=uniformDetailModule(this.device,{label:"Uniform mixed frame plan",code:uniformMixedCountedEntriesWGSL(topology+/* wgsl */`
@group(1) @binding(0) var volume:texture_3d<f32>;
@group(1) @binding(1) var phi:texture_3d<f32>;
struct PlanPolicy {settings:vec4u,step:vec4f}
@group(1) @binding(2) var<uniform> policy:PlanPolicy;
@group(1) @binding(3) var velocity:texture_3d<f32>;
@group(1) @binding(4) var<storage,read> negative:array<f32>;
@group(2) @binding(0) var extended:texture_3d<f32>;
@group(2) @binding(1) var<storage,read> extendedNegative:array<f32>;
@group(2) @binding(2) var<storage,read_write> speeds:array<u32>;
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
// A tile corner: the base block (UNIFORM_DETAIL_4H_LOAD).
fn umLoadCorner(p:vec3u)->f32{return ${UNIFORM_DETAIL_4H_LOAD}textureLoad(phi,vec3i(p),0).x;}
fn umNegativeIndex(origin:vec3u,axis:u32)->u32{
 if(axis==0u){return origin.y+UM_D.y*origin.z;}
 if(axis==1u){return UM_D.y*UM_D.z+origin.x+UM_D.x*origin.z;}
 return UM_D.y*UM_D.z+UM_D.x*UM_D.z+origin.x+UM_D.x*origin.y;
}
// Tiles, in Chebyshev tile distance, that any characteristic of this frame
// can reach, including cubic taps, wall continuation and the Newton search.
fn umCertificateRadius(speed:f32)->u32{
 let travel=speed*policy.step.x/${Math.min(...h)};
 return u32(min(ceil(travel*1.00001/4.0)+2.0,1e8));
}
// Signed per-axis reach, in tiles. A departure moves by -v*dt and every
// velocity a sampler reads is a convex combination of extended faces inside
// the global-reach box (or a zero-weight tap), so along +a it travels at most
// max(0,-dt*min v_a) cells and along -a at most max(0,dt*max v_a). The +2
// margin of umCertificateRadius covers the start offset, taps, wall
// continuation and the Newton window on every side, as the Chebyshev radius
// did. Five bits each: +x,+y,+z at bits 0,5,10 and -x,-y,-z at 15,20,25;
// UM_REACH_SATURATED (unbounded or non-finite) never certifies.
const UM_REACH_SATURATED:u32=31u;
fn umReach(travel:f32)->u32{
 let tiles=ceil(travel*1.00001/4.0)+2.0;
 return select(UM_REACH_SATURATED,u32(tiles),tiles<f32(UM_REACH_SATURATED));
}
fn umPackReach(plus:vec3u,minus:vec3u)->u32{return plus.x|(plus.y<<5u)|(plus.z<<10u)|(minus.x<<15u)|(minus.y<<20u)|(minus.z<<25u);}
fn umReachPlus(w:u32)->vec3u{return vec3u(w&31u,(w>>5u)&31u,(w>>10u)&31u);}
fn umReachMinus(w:u32)->vec3u{return vec3u((w>>15u)&31u,(w>>20u)&31u,(w>>25u)&31u);}
// Every plan follows a phi resolve of its layout (frame start, census tail).
${uniformMixedVertexSamplingSource("",true)}
// An owner's positive-face speeds (of velocity: a texture parameter would
// bypass the detail accessors). In a tile whose stencil is all unit width
// every positive face is the owner's own unit patch, anchored at its origin.
fn umPositiveFaceSpeeds(unit:bool,origin:vec3u)->vec3f{
 if(unit){return abs(textureLoad(velocity,vec3i(origin),0).xyz);}
 let owner=umOwnerAt(vec3i(origin));var speed=vec3f(0);
 for(var axis=0u;axis<3u;axis++){
  let first=umFace(owner,axis,1,0u);
  // A 4h owner's single patch is anchored at its tile's +face anchor: the base block.
  if(first.width==4u){speed[axis]=abs(${UNIFORM_DETAIL_4H_LOAD}textureLoad(velocity,first.anchor,0)[axis]);continue;}
  for(var part=0u;part<first.count;part++){let f=umFacePatch(first,part);speed[axis]=max(speed[axis],abs(textureLoad(velocity,f.anchor,0)[axis]));}
 }
 return speed;
}
fn umPositiveFaceSpeed(unit:bool,origin:vec3u)->f32{
 let v=umPositiveFaceSpeeds(unit,origin);return max(v.x,max(v.y,v.z));
}
// One lane per owner over every h tile and the 4h tiles of resident pages
// (umResidentAllOwner). A tile's owners are contiguous slots, so
// lane-owner.lane is its first lane: the workgroup slot of its flags. 4h
// tiles pack 64 to a workgroup instead of idling 63 lanes. A 4h tile of an
// absent page is certified far air (V=0, every corner phi at least 16h: the
// census audits exactly this predicate), so its flags are the zero encode
// clears to; its faces are extended faces, never a source, so the speed
// maximum over resident owners still bounds every face the certificate's
// characteristics read (the extension is a convex combination of sources).
var<workgroup> seeded:array<atomic<u32>,64>;
var<workgroup> seedSpeed:atomic<u32>;
@compute @workgroup_size(64) fn seed(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 let owner=umResidentAllOwner(gid);let slot=lane-owner.lane;
 atomicStore(&seeded[lane],0u);if(lane==0u){atomicStore(&seedSpeed,0u);}workgroupBarrier();
 if(owner.width!=0u){
  let width=owner.width;let unit=umTileMaximumWidth(owner.tile)==1u;let origin=umOrigin(owner);
  var speed=umPositiveFaceSpeed(unit,origin);
  for(var axis=0u;axis<3u;axis++){if(origin[axis]==0u){speed=max(speed,abs(negative[umNegativeIndex(origin,axis)]));}}
  atomicMax(&seedSpeed,bitcast<u32>(speed));
  // A 4h owner's origin and corners are its tile's (umVertexValue of a
  // tile corner is its load): the base blocks.
  var occupied=false;
  if(width==4u){
   occupied=${UNIFORM_DETAIL_4H_LOAD}textureLoad(volume,vec3i(origin),0).x!=0.0;
   for(var k=0u;k<8u;k++){occupied=occupied||umLoadCorner(origin+umCorner(k,2u)*4u)<${16*Math.max(...h)};}
  }else{
   occupied=textureLoad(volume,vec3i(origin),0).x!=0.0;
   for(var k=0u;k<8u;k++){
    // Every vertex of a unit-stencil tile is stored.
    let vertex=origin+umCorner(k,2u)*width;
    occupied=occupied||select(umVertexValue(vertex),umLoadVertex(vertex),unit)<${4*Math.max(...h)}*f32(width);
   }
  }
  if(occupied){atomicOr(&seeded[slot],3u);}
 }
 workgroupBarrier();
 if(owner.width!=0u&&owner.lane==0u){atomicStore(&umSupport[owner.tile],atomicLoad(&seeded[slot]));}
 if(lane==0u){atomicMax(&umSupport[4u*UM_TILES],atomicLoad(&seedSpeed));}
}
${[0,1,2].map(axis=>/* wgsl */`
@compute @workgroup_size(64) fn dilate${axis}(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(tile>=UM_TILES){return;}
 let p=vec3i(umTileCoord(tile));var flags=0u;
 let reach=i32(policy.settings.x+policy.settings.y);
 for(var delta=-reach;delta<=reach;delta++){
  var q=p;q[${axis}]+=delta;if(any(q<vec3i(0))||any(q>=vec3i(UM_T))){continue;}
  let source=atomicLoad(&umSupport[${axis}u*UM_TILES+umTileAt(vec3u(q))]);
  flags|=source&select(2u,3u,abs(delta)<=i32(policy.settings.x));
 }
 atomicStore(&umSupport[${axis+1}u*UM_TILES+tile],flags|policy.settings.z);
}`).join("\n")}
// Each tile's signed reach (umPackReach) from the extended canonical faces it
// owns, zero included. Owner lanes as in seed.
fn umSignedFaceExtent(unit:bool,origin:vec3u,axis:u32)->vec2f{
 var lo=0.0;var hi=0.0;
 if(unit){let v=textureLoad(extended,vec3i(origin),0)[axis];lo=min(lo,v);hi=max(hi,v);}
 else{
  let owner=umOwnerAt(vec3i(origin));let first=umFace(owner,axis,1,0u);
  if(first.width==4u){let v=${UNIFORM_DETAIL_4H_LOAD}textureLoad(extended,first.anchor,0)[axis];lo=min(lo,v);hi=max(hi,v);}
  else{for(var part=0u;part<first.count;part++){let v=textureLoad(extended,umFacePatch(first,part).anchor,0)[axis];lo=min(lo,v);hi=max(hi,v);}}
 }
 if(origin[axis]==0u){let v=extendedNegative[umNegativeIndex(origin,axis)];lo=min(lo,v);hi=max(hi,v);}
 return vec2f(lo,hi);
}
var<workgroup> tileReach:array<atomic<u32>,384>;
@compute @workgroup_size(64) fn localSpeed(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 let owner=umAllOwner(gid);let slot=lane-owner.lane;
 for(var k=0u;k<6u;k++){atomicStore(&tileReach[64u*k+lane],0u);}
 workgroupBarrier();
 if(owner.width!=0u){
  let origin=umOrigin(owner);let unit=umTileMaximumWidth(owner.tile)==1u;let scale=policy.step.x/${Math.min(...h)};
  for(var axis=0u;axis<3u;axis++){
   let e=umSignedFaceExtent(unit,origin,axis);
   // NaN fails both comparisons: it saturates.
   let finite=abs(e.x)<=3.402823e38&&abs(e.y)<=3.402823e38;
   atomicMax(&tileReach[64u*axis+slot],select(UM_REACH_SATURATED,umReach(-e.x*scale),finite));
   atomicMax(&tileReach[64u*(3u+axis)+slot],select(UM_REACH_SATURATED,umReach(e.y*scale),finite));
  }
 }
 workgroupBarrier();
 if(owner.width!=0u&&owner.lane==0u){
  var plus=vec3u(0);var minus=vec3u(0);
  for(var axis=0u;axis<3u;axis++){plus[axis]=atomicLoad(&tileReach[64u*axis+slot]);minus[axis]=atomicLoad(&tileReach[64u*(3u+axis)+slot]);}
  speeds[owner.tile]=umPackReach(plus,minus);
 }
}
// Separable box maximum over the global reach: a characteristic from tile t
// never leaves that box, so its speed is bounded by the box maximum.
${[0,1,2].map(axis=>/* wgsl */`
@compute @workgroup_size(64) fn spread${axis}(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(tile>=UM_TILES){return;}
 let base=${axis===1?"UM_TILES":"0u"};let into=${axis===1?"0u":"UM_TILES"};
 let reach=i32(umCertificateRadius(bitcast<f32>(atomicLoad(&umSupport[4u*UM_TILES]))));
 let p=vec3i(umTileCoord(tile));var plus=vec3u(0);var minus=vec3u(0);
 for(var q=max(p.${"xyz"[axis]}-reach,0);q<=min(p.${"xyz"[axis]}+reach,i32(UM_T.${"xyz"[axis]})-1);q++){
  var r=p;r.${"xyz"[axis]}=q;let value=speeds[base+umTileAt(vec3u(r))];
  plus=max(plus,umReachPlus(value));minus=max(minus,umReachMinus(value));
 }
 speeds[into+tile]=umPackReach(plus,minus);
}`).join("\n")}
// Summed-volume table of the 4h tiles in speeds[0,T), which is dead after
// spread2: inclusive prefix counts along x, then y, then z, in place, one lane
// per tile line.
${[0,1,2].map(axis=>/* wgsl */`
@compute @workgroup_size(64) fn prefix${axis}(@builtin(global_invocation_id) gid:vec3u){
 let line=gid.x+umDispatchX*64u*gid.y;if(line>=UM_T.${"xyz"[(axis+1)%3]}*UM_T.${"xyz"[(axis+2)%3]}){return;}
 var p=vec3u(0);p.${"xyz"[(axis+1)%3]}=line%UM_T.${"xyz"[(axis+1)%3]};p.${"xyz"[(axis+2)%3]}=line/UM_T.${"xyz"[(axis+1)%3]};var sum=0u;
 for(var i=0u;i<UM_T.${"xyz"[axis]};i++){p.${"xyz"[axis]}=i;let t=umTileAt(p);sum+=${axis===0?"select(0u,1u,umTileWidth(t)!=1u)":"speeds[t]"};speeds[t]=sum;}
}`).join("\n")}
fn umPrefix(p:vec3i)->u32{if(any(p<vec3i(0))){return 0u;}return speeds[umTileAt(vec3u(p))];}
// 4h tiles in the inclusive tile box [low,high] clipped to the lattice (which
// holds the tile itself). Wrapping u32 arithmetic is exact: the count fits.
fn umCoarseTilesIn(low:vec3i,high:vec3i)->u32{
 let a=max(low,vec3i(0))-vec3i(1);let b=min(high,vec3i(UM_T)-vec3i(1));
 return umPrefix(b)-umPrefix(vec3i(a.x,b.y,b.z))-umPrefix(vec3i(b.x,a.y,b.z))-umPrefix(vec3i(b.x,b.y,a.z))
  +umPrefix(vec3i(a.x,a.y,b.z))+umPrefix(vec3i(a.x,b.y,a.z))+umPrefix(vec3i(b.x,a.y,a.z))-umPrefix(a);
}
// Appends aggregate per workgroup: one global atomic per list, not per tile.
var<workgroup> certifyCounts:array<atomic<u32>,2>;
var<workgroup> certifyBases:array<u32,2>;
@compute @workgroup_size(64) fn certify(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=gid.x+umDispatchX*64u*gid.y;let fine=tile<UM_TILES&&umTileWidth(min(tile,UM_TILES-1u))==1u;
 if(lane<2u){atomicStore(&certifyCounts[lane],0u);}workgroupBarrier();
 // Every characteristic from this tile stays in its signed reach box: the
 // spread reach bounds the extended velocities of the global-reach box around
 // it (localSpeed + spread). With no 4h tile in the box every tap is a unit
 // face and the regular sampler is the general one.
 var list=1u;var local=0u;
 if(fine){
  let reach=speeds[UM_TILES+tile];let plus=umReachPlus(reach);let minus=umReachMinus(reach);
  let bounded=all(plus<vec3u(UM_REACH_SATURATED))&&all(minus<vec3u(UM_REACH_SATURATED));
  let p=vec3i(umTileCoord(tile));
  let regular=bounded&&umCoarseTilesIn(p-vec3i(minus),p+vec3i(plus))==0u;
  list=select(2u,1u,regular);local=atomicAdd(&certifyCounts[list-1u],1u);
  // The class for the tiles layer, in the reach word's two spare bits (no
  // reader masks them in: umReachPlus/Minus take bits 0..29).
  speeds[UM_TILES+tile]=reach|(select(select(${UNIFORM_STAGE_CERTIFICATE.coarseInReach}u,${UNIFORM_STAGE_CERTIFICATE.saturated}u,!bounded),${UNIFORM_STAGE_CERTIFICATE.regular}u,regular)<<30u);
 }
 workgroupBarrier();
 if(lane<2u){let count=atomicLoad(&certifyCounts[lane]);if(count>0u){certifyBases[lane]=atomicAdd(&umSupport[4u*UM_TILES+lane+1u],count);}}
 workgroupBarrier();
 if(fine){atomicStore(&umSupport[(4u+list)*UM_TILES+16u+certifyBases[list-1u]+local],tile);}
}
`,counted)});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
    if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
    const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources,this.extendedResources]});
    await Promise.all(["seed","dilate0","dilate1","dilate2","localSpeed","spread0","spread1","spread2","prefix0","prefix1","prefix2","certify"].map(async entryPoint=>{this.pipelines.set(entryPoint,await uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX,...(counted.includes(entryPoint)?{umCountedJobs:entryPoint==="seed"?UNIFORM_MIXED_COUNTED.residentAll:UNIFORM_MIXED_COUNTED.all}:{})}}}));}));
  }
  encode(encoder:GPUCommandEncoder,policy={fineReach:2,shellReach:1,twoLevel:true,shellOnly:true}):void{
    if(this.pipelines.size!==12)throw new Error("Mixed frame plan is not initialized");
    this.device.queue.writeBuffer(this.params,0,new Uint32Array([policy.fineReach,policy.shellReach,!policy.twoLevel?3:!policy.shellOnly?2:0,0]));
    encoder.clearBuffer(this.ownership.support,this.ownership.capacity.tiles*16,64);
    // Seed flags: absent pages' tiles keep this zero (the seed skips them).
    encoder.clearBuffer(this.ownership.support,0,this.ownership.capacity.tiles*4);
    const pass=encoder.beginComputePass({label:"Uniform shared frame plan"});
    pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.group.group);pass.setBindGroup(2,this.extendedGroup.group);
    this.dispatch(pass,["seed","dilate0","dilate1","dilate2"]);
    pass.end();
  }
  /** Fine-tile certificate from the extended field. Encode after extension
   * and before any certified/merged consumer (surface, momentum). */
  /** Each tile's certificate word after encodeCertificate (speeds[T, 2T)):
   * signed reach and, for h tiles, the class (UNIFORM_STAGE_CERTIFICATE). */
  get certificate():{readonly buffer:GPUBuffer;readonly offset:number}{return {buffer:this.ownership.speeds,offset:4*this.ownership.capacity.tiles};}
  encodeCertificate(encoder:GPUCommandEncoder,dt:number):void{
    // Only the certificate reads dt: support and extension are dt-free.
    this.device.queue.writeBuffer(this.params,16,new Float32Array([dt,0,0,0]));
    const pass=encoder.beginComputePass({label:"Uniform local speed certificate"});
    pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.group.group);pass.setBindGroup(2,this.extendedGroup.group);
    // Certified consumers read the list counts themselves (umCertifiedJobCount).
    this.dispatch(pass,["localSpeed","spread0","spread1","spread2","prefix0","prefix1","prefix2","certify"]);
    pass.end();
  }
  private dispatch(pass:GPUComputePassEncoder,entries:readonly string[]):void{
    for(const entry of entries){
      if(counted.includes(entry)){this.ownership.dispatchAllCounted(pass,this.pipelines.get(entry)!);continue;}
      const t=this.ownership.capacity.tileDimensions,line=/^prefix(\d)$/.exec(entry);
      const groups=Math.ceil((line?t[(+line[1]!+1)%3]!*t[(+line[1]!+2)%3]!:this.ownership.capacity.tiles)/64);
      pass.setPipeline(uniformDetailPick(this.pipelines.get(entry)!));pass.dispatchWorkgroups(Math.min(groups,this.ownership.dispatchX),Math.ceil(groups/this.ownership.dispatchX));
    }
  }
  destroy():void{this.params.destroy();}
}
