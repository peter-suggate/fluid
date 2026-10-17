import { RESIDENCY_STAGES } from "./uniform-local-residency";
import { uniformFramePlanSpreadWGSL } from "./uniform-frame-plan-spread";
import {UNIFORM_DETAIL_4H_LOAD} from "../../core/uniform-detail-abi";
import { uniformDetailBindLayout, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, uniformDetailGroup, type UniformDetailGroup } from "./uniform-detail-fields";
import {UNIFORM_TILE_BITS_WORDS,uniformTileBitsWGSL} from "./uniform-block-tables";
import {uniformKeyGroups} from "./uniform-lattice-address";
import type {UniformMixedOwnership} from "./uniform-mixed-ownership";
import {UNIFORM_MIXED_COUNTED,uniformMixedCountedEntriesWGSL,uniformMixedTopologyNeighboursWGSL,uniformMixedTopologyWGSL} from "./uniform-mixed-topology.wgsl";
import {uniformMixedVertexSamplingSource} from "./uniform-mixed-vertex-sampling.wgsl";
import {UNIFORM_STAGE_CERTIFICATE} from "./uniform-stage-grids";
import {uniformRegionHeatWGSL} from "./uniform-region-heat";

/** Owner entries: one lane per owner of every tier, GPU-counted. */
const counted:readonly string[]=["seed","localSpeed"];
/** The entries of the support census and of the certificate, in launch order. */
const SUPPORT_ENTRIES:readonly string[]=["seed","dilate0","dilate1","dilate2"];
const CERTIFICATE_ENTRIES:readonly string[]=["localSpeed","spread0","spread1","spread2","coarseBits","coarseBlocks","certify"];
const PLAN_PASS:GPUComputePassDescriptor={label:"Uniform shared frame plan"},CERTIFICATE_PASS:GPUComputePassDescriptor={label:"Uniform local speed certificate"};
/** The support policy of an encode given none. */
const DEFAULT_POLICY:{fineReach:number;shellReach:number;twoLevel:boolean;shellOnly:boolean}={fineReach:RESIDENCY_STAGES[1].core,shellReach:1,twoLevel:true,shellOnly:true};
/** One launch of a plan pass: an owner entry's GPU-counted launch, or a
 * per-tile (coarseBlocks: per block) entry's grid, fixed with the pool. */
interface PlanLaunch{readonly pipeline:GPUComputePipeline;readonly counted:boolean;readonly x:number;readonly y:number}

/** Start-of-frame support census, independent of simulation ownership. The
 * native fine sampling reach is two tiles; extension gets one additional tile.
 * All stages share the resulting immutable support planes until the next frame.
 * A source/edit must be applied before this census, just as in the native host.
 * This is sampling/extension support, not permission to skip transported mass. */
export class UniformMixedFramePlan {
  readonly allocatedBytes:number;
  private readonly params:GPUBuffer;
  /** The 4h tile bits (uniformTileBitsWGSL): by block, a bit per tile key
   * stored at 4h. */
  private readonly coarse:GPUBuffer;
  private readonly resources:GPUBindGroupLayout;
  private readonly group:UniformDetailGroup;
  private readonly extendedResources:GPUBindGroupLayout;
  private readonly extendedGroup:UniformDetailGroup;
  /** Each tile's seed flags as last seeded (bits 0..1), bit 31 set when they
   * were seeded in a COLD region by a retaining encode: a word per tile key. */
  private readonly held:GPUBuffer;
  private readonly heldResources:GPUBindGroupLayout;
  /** The held flags and the region work tiers (attachHeat): seed's group 3. */
  private heldGroup?:GPUBindGroup;
  private readonly pipelines=new Map<string,GPUComputePipeline>();
  private supportRevision=-1;
  /** The last encoded policy's words (fine reach, shell reach, mode, 0): none yet. */
  private readonly supportSettings:number[]=[-1,-1,-1,-1];
  /** The parameter words of encode (policy) and encodeCertificate (step), and
   * each pass's launches: built at the first encode and replayed. */
  private policyWords?:Uint32Array<ArrayBuffer>;
  private stepWords?:Float32Array<ArrayBuffer>;
  private launches?:{readonly support:readonly PlanLaunch[];readonly held:readonly PlanLaunch[];readonly certificate:readonly PlanLaunch[]};
  private certificateView?:{readonly buffer:GPUBuffer;readonly offset:number};
  constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,volume:GPUTexture,phi:GPUTexture,
    /** The extended field every characteristic samples, valid at encodeCertificate. */
    extended:GPUTexture,extendedNegative:GPUBuffer,directionalCertificate=false,private readonly surfaceOnly=false){
    if(!directionalCertificate)throw new Error("The mixed frame plan certifies signed per-axis reach only");
    this.params=device.createBuffer({label:"Uniform shared support policy",size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.coarse=device.createBuffer({label:"Uniform frame plan 4h tile bits",size:4*UNIFORM_TILE_BITS_WORDS*ownership.capacity.address.blocks,usage:GPUBufferUsage.STORAGE});
    this.held=device.createBuffer({label:"Uniform frame plan held seed flags",size:4*ownership.capacity.tiles,usage:GPUBufferUsage.STORAGE});
    this.allocatedBytes=32+this.coarse.size+this.held.size;
    this.resources=uniformDetailBindLayout(device,{entries:[...[0,1].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]});
    this.group=uniformDetailGroup(device,{layout:this.resources,entries:[...[volume,phi].map((t,binding)=>({binding,resource:t})),{binding:2,resource:{buffer:this.params}}]});
    this.heldResources=device.createBindGroupLayout({label:"Uniform frame plan held seed flags",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}}]});
    this.extendedResources=uniformDetailBindLayout(device,{entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
      {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},...[2,3,4].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}}))]});
    this.extendedGroup=uniformDetailGroup(device,{layout:this.extendedResources,entries:[{binding:0,resource:extended},{binding:1,resource:{buffer:extendedNegative}},{binding:2,resource:{buffer:ownership.speeds}},{binding:3,resource:{buffer:this.coarse}},{binding:4,resource:{buffer:ownership.regionVelocity}}]});
  }
  /** The region work tiers (UniformRegionHeat.tiers), fixed for life: a
   * retaining encode keeps a COLD tile's seed flags. */
  attachHeat(tiers:GPUBuffer):void{
    this.heldGroup=this.device.createBindGroup({label:"Uniform frame plan held seed flags",layout:this.heldResources,entries:[{binding:0,resource:{buffer:this.held}},{binding:1,resource:{buffer:tiers}}]});
  }
  async initialize():Promise<void>{
    const h=this.ownership.capacity.lattice.cellSize_m;
    // The generator reads the lattice and tile count only (fixed by capacity).
    const topology=uniformMixedTopologyWGSL(this.ownership.capacity,0).replace('umSupport:array<u32>','umSupport:array<atomic<u32>>').replace(/umSupport\[([^\]]+)\]/g,'atomicLoad(&umSupport[$1])')+uniformMixedTopologyNeighboursWGSL();
    // seed and localSpeed stride the GPU-counted owners of every tier
    // (umAllOwner); the per-tile passes cover the lattice.
    const module=uniformDetailModule(this.device,{label:"Uniform mixed frame plan",code:uniformMixedCountedEntriesWGSL(topology+/* wgsl */`
@group(1) @binding(0) var volume:texture_3d<f32>;
@group(1) @binding(1) var phi:texture_3d<f32>;
// A retaining seed (the seedHeld variant: the advance's own encodes).
override planRetain:bool=false;
struct PlanPolicy {settings:vec4u,step:vec4f}
@group(1) @binding(2) var<uniform> policy:PlanPolicy;
@group(3) @binding(0) var<storage,read_write> planHeld:array<u32>;
${uniformRegionHeatWGSL(3,1,this.ownership.capacity.tiles/64)}
@group(2) @binding(0) var extended:texture_3d<f32>;
@group(2) @binding(1) var<storage,read> extendedNegative:array<f32>;
@group(2) @binding(2) var<storage,read_write> speeds:array<u32>;
@group(2) @binding(3) var<storage,read_write> coarse:array<u32>;
@group(2) @binding(4) var<storage,read_write> regionVelocity:array<atomic<u32>>;
// umSupport and speeds hold a word per tile KEY: a lane indexes them by its
// own tile's key, and a neighbour's word by umTileKey of the neighbour's
// coordinate, skipped where umKeyStored refuses it. coarse holds the 4h tile
// bits by block (uniformTileBitsWGSL), all written by every certificate.
// A tile with nothing stored is skipped, never read. An AIR tile holds the
// AIR image: no support flag, zero faces, a 4h tile word. So it adds nothing
// to a support dilation. For a separable reach maximum it invalidates
// the regular certificate, since an intermediate path may cross the hole;
// coarseAny also treats it as 4h. A SOLID tile adds nothing to either.
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
// A tile corner: the base block (UNIFORM_DETAIL_4H_LOAD).
fn umLoadCorner(p:vec3u)->f32{return ${UNIFORM_DETAIL_4H_LOAD}textureLoad(phi,vec3i(p),0).x;}
// The same loads of the eight tile corners p + s*umCorner(k,2u): one page-table read where one block holds them.
fn umLoadCorner8(p:vec3u,s:u32)->array<f32,8>{return udLoad48_phi(vec3i(p),s);}
// Tiles, in Chebyshev tile distance, that any characteristic of this frame
// can reach, including cubic taps, wall continuation and the Newton search.
fn umCertificateRadius(speed:f32)->u32{
 let travel=speed*policy.step.x/umSceneMinimumSpacing();
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
// One lane per owner over every h tile and the 4h tiles of resident pages
// (umResidentAllOwner). A tile's owners are contiguous slots, so
// lane-owner.lane is its first lane: the workgroup slot of its flags. 4h
// tiles pack 64 to a workgroup instead of idling 63 lanes. A 4h tile of an
// absent region is certified empty (NB: no negative phi; VOF: V=0 and
// corner phi at least 16h), so its seed flags are the zero encode
// clears to.
// A retaining encode (seedHeld: the advance's own, after the frame head's heat) keeps
// a COLD region's tile flags: a COLD region's phi and closure hold
// (uniform-region-heat), so flags seeded in a COLD region by a retaining
// encode stand while every later encode finds it COLD. Any other seeding
// (a warm or hot region, a non-retaining encode) rewrites the word without
// the held bit, so a held word was seeded in the current COLD stretch.
const UM_PLAN_HELD:u32=0x80000000u;
var<workgroup> seeded:array<atomic<u32>,64>;
@compute @workgroup_size(64) fn seed(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 let owner=umResidentAllOwner(gid);let slot=lane-owner.lane;
 atomicStore(&seeded[lane],0u);workgroupBarrier();
 var cold=false;var held=0u;
 if(owner.width!=0u){
  cold=planRetain&&umRegionTier(umPageOfTile(owner.tile))==UM_TIER_COLD;held=planHeld[owner.tile];
 }
 let keep=cold&&(held&UM_PLAN_HELD)!=0u;
 if(owner.width!=0u&&!keep){
  let width=owner.width;let unit=umTileMaximumWidth(owner.tile)==1u;let origin=umOrigin(owner);
  // A 4h owner's origin and corners are its tile's (umVertexValue of a
  // tile corner is its load): the base blocks.
  // NB velocity work starts at liquid, not at the entire positive phi
  // collar. The dilation below is the sampling/extension dependency set;
  // expanding a pre-expanded distance collar counted that support twice.
  var occupied=false;
  if(width==4u){
   ${this.surfaceOnly?"":`occupied=${UNIFORM_DETAIL_4H_LOAD}textureLoad(volume,vec3i(origin),0).x!=0.0;`}
   let c8=umLoadCorner8(origin,4u);for(var k=0u;k<8u;k++){occupied=occupied||c8[k]<${this.surfaceOnly?"0.0":"(16.0*umSceneMaximumSpacing())"};}
  }else{
   ${this.surfaceOnly?`// The tile flag is the OR over its 64 owners' eight corners: exactly
   // its 125 distinct vertices. Share those vertices across the same lanes
   // instead of loading each interior corner up to eight times.
   let base=origin-umCorner(owner.lane,4u);
   for(var k=owner.lane;k<125u;k+=64u){
    let vertex=base+umCorner(k,5u);var value=0.0;
    if(unit){value=umLoadVertex(vertex);}else{value=umVertexValue(vertex);}
    occupied=occupied||value<0.0;
   }`:`occupied=textureLoad(volume,vec3i(origin),0).x!=0.0;
   for(var k=0u;k<8u;k++){
    let vertex=origin+umCorner(k,2u)*width;
    occupied=occupied||select(umVertexValue(vertex),umLoadVertex(vertex),unit)<(4.0*umSceneMaximumSpacing())*f32(width);
   }`}
  }
  if(occupied){atomicOr(&seeded[slot],3u);}
 }
 workgroupBarrier();
 if(owner.width!=0u&&owner.lane==0u){
  var flags=held&3u;
  if(!keep){flags=atomicLoad(&seeded[slot]);planHeld[owner.tile]=flags|select(0u,UM_PLAN_HELD,cold);}
  atomicStore(&umSupport[owner.tile],flags);
 }
}
${[0,1,2].map(axis=>/* wgsl */`
@compute @workgroup_size(64) fn dilate${axis}(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(!umTileLive(tile)){return;}
 let p=vec3i(umTileCoord(tile));var flags=0u;
 let reach=i32(policy.settings.x+policy.settings.y);
 for(var delta=-reach;delta<=reach;delta++){
  var q=p;q[${axis}]+=delta;let k=umTileKey(q);if(!umKeyStored(k)){continue;}
  let source=atomicLoad(&umSupport[${axis}u*UM_TILES+k]);
  // Crossing the outer shell consumes its one axial dependency. Propagating
  // shell bit 2 again on the next axis turns face reads into diagonal reads.
  ${this.surfaceOnly?"flags|=select((source&1u)<<1u,source&3u,abs(delta)<=i32(policy.settings.x));":"flags|=source&select(2u,3u,abs(delta)<=i32(policy.settings.x));"}
 }
 atomicStore(&umSupport[${axis+1}u*UM_TILES+tile],flags|policy.settings.z);
}`).join("\n")}
// Each tile's signed reach (umPackReach) from the extended canonical faces it
// owns; low: the owner stores its low face along axis
// (umLowOpen). Owner lanes as in seed.
// The spread's global radius (umSupport[4*UM_TILES], umCertificateRadius)
// takes the largest |extended face| over every owner: exactly the faces a
// characteristic reads (an absent region's extended faces are convex
// combinations of sources). A nonfinite face makes it infinite: every
// certificate saturates.
fn umSignedFaceExtent(unit:bool,origin:vec3u,axis:u32,low:bool)->vec3f{
 var lo=3.402823e38;var hi=-3.402823e38;var finite=true;
 if(unit){let v=textureLoad(extended,vec3i(origin),0)[axis];finite=finite&&abs(v)<=3.402823e38;lo=min(lo,v);hi=max(hi,v);}
 else{
  let owner=umOwnerAt(vec3i(origin));let first=umFace(owner,axis,1,0u);
  if(first.width==4u){let v=${UNIFORM_DETAIL_4H_LOAD}textureLoad(extended,first.anchor,0)[axis];finite=finite&&abs(v)<=3.402823e38;lo=min(lo,v);hi=max(hi,v);}
  else{for(var part=0u;part<first.count;part++){let v=textureLoad(extended,umFacePatch(first,part).anchor,0)[axis];finite=finite&&abs(v)<=3.402823e38;lo=min(lo,v);hi=max(hi,v);}}
 }
 if(low){let v=extendedNegative[umLowFaceIndex(origin,axis)];finite=finite&&abs(v)<=3.402823e38;lo=min(lo,v);hi=max(hi,v);}
 return vec3f(lo,hi,f32(finite&&lo<=hi));
}
var<workgroup> tileReach:array<atomic<u32>,384>;
var<workgroup> tileVelocity:array<atomic<u32>,384>;
var<workgroup> planSpeed:atomic<u32>;
// Float-flip orders finite signed floats as unsigned words. An empty word
// is below every finite value; nonfinite inputs poison the whole interval.
fn umVelocityOrder(value:f32)->u32{
 let bits=bitcast<u32>(value);return select(bits^0x80000000u,~bits,(bits&0x80000000u)!=0u);
}
@compute @workgroup_size(64) fn localSpeed(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 let owner=umAllOwner(gid);let slot=lane-owner.lane;
 for(var k=0u;k<6u;k++){atomicStore(&tileReach[64u*k+lane],0u);atomicStore(&tileVelocity[64u*k+lane],0u);}
 if(lane==0u){atomicStore(&planSpeed,0u);}
 workgroupBarrier();
 if(owner.width!=0u){
  let origin=umOrigin(owner);let unit=umTileMaximumWidth(owner.tile)==1u;let scale=policy.step.x/umSceneMinimumSpacing();
  let low=umLowOpen(vec3i(origin));var fastest=0u;
  for(var axis=0u;axis<3u;axis++){
   let e=umSignedFaceExtent(unit,origin,axis,low[axis]);
   // NaN fails both comparisons: it saturates.
   let finite=e.z!=0.0;
   atomicMax(&tileReach[64u*axis+slot],select(UM_REACH_SATURATED,umReach(max(0.0,-e.x)*scale),finite));
   atomicMax(&tileReach[64u*(3u+axis)+slot],select(UM_REACH_SATURATED,umReach(max(0.0,e.y)*scale),finite));
   atomicMax(&tileVelocity[64u*axis+slot],select(0xffffffffu,umVelocityOrder(e.y),finite));
   atomicMax(&tileVelocity[64u*(3u+axis)+slot],select(0xffffffffu,umVelocityOrder(-e.x),finite));
   fastest=max(fastest,select(0x7f800000u,bitcast<u32>(max(abs(e.x),abs(e.y))),finite));
  }
  atomicMax(&planSpeed,fastest);
 }
 workgroupBarrier();
 if(lane==0u){atomicMax(&umSupport[4u*UM_TILES],atomicLoad(&planSpeed));}
 if(owner.width!=0u&&owner.lane==0u){
  var plus=vec3u(0);var minus=vec3u(0);
  for(var axis=0u;axis<3u;axis++){plus[axis]=atomicLoad(&tileReach[64u*axis+slot]);minus[axis]=atomicLoad(&tileReach[64u*(3u+axis)+slot]);}
  speeds[owner.tile]=umPackReach(plus,minus);
  // Reuse the face reads and tile reduction above. Six atomics per tile,
  // no additional field scan, publish current unquantized local motion.
  let region=umPageOfTile(owner.tile);
  for(var k=0u;k<6u;k++){atomicMax(&regionVelocity[6u*region+k],atomicLoad(&tileVelocity[64u*k+slot]));}
 }
}
// Sparse allocation need not contain the rectangular intermediate paths of
// a separable maximum. An AIR hole propagates saturation, making every
// affected certificate use the general mixed sampler. A single global
// allocation-radius comparison cannot establish coverage of a directional
// region; the per-tile certificate and the actual support audits do so.
${[0,1,2].map(axis=>uniformFramePlanSpreadWGSL(axis)).join("\n")}
// The 4h tile bits: a stored tile at 4h. An AIR page is 4h throughout
// (coarseAny); a SOLID page holds none. Past 64 pages a box is answered as
// holding one: no tile with so long a reach is certified regular.
${uniformTileBitsWGSL({name:"coarse",word:(b,i)=>`coarse[${UNIFORM_TILE_BITS_WORDS}u*${b}+${i}]`,store:(b,i,v)=>`coarse[${UNIFORM_TILE_BITS_WORDS}u*${b}+${i}]=${v};`,flag:t=>`umTileWidth(${t})!=1u`,air:true})}
// Appends aggregate per workgroup: one global atomic per list, not per tile.
var<workgroup> certifyCounts:array<atomic<u32>,2>;
var<workgroup> certifyBases:array<u32,2>;
@compute @workgroup_size(64) fn certify(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=gid.x+umDispatchX*64u*gid.y;let fine=umTileLive(tile)&&umTileWidth(min(tile,UM_TILES-1u))==1u;
 if(lane<2u){atomicStore(&certifyCounts[lane],0u);}workgroupBarrier();
 // Every characteristic from this tile stays in its signed reach box: the
 // spread reach bounds the extended velocities of the global-reach box around
 // it (localSpeed + spread). With no 4h tile in the box (an AIR tile is one:
 // coarseAny) every tap is a unit face and the regular sampler is the
 // general one.
 var list=1u;var local=0u;
 if(fine){
  let reach=speeds[UM_TILES+tile];let plus=umReachPlus(reach);let minus=umReachMinus(reach);
  let bounded=all(plus<vec3u(UM_REACH_SATURATED))&&all(minus<vec3u(UM_REACH_SATURATED));
  let p=vec3i(umTileCoord(tile));
  let regular=bounded&&!coarseAny(tile>>9u,p-vec3i(minus),p+vec3i(plus));
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
    const seedLayout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources,this.extendedResources,this.heldResources]});
    await Promise.all(["seed","dilate0","dilate1","dilate2","localSpeed","spread0","spread1","spread2","coarseBits","coarseBlocks","certify"].map(async entryPoint=>{this.pipelines.set(entryPoint,await uniformDetailPipeline(this.device,this.ownership,{layout:entryPoint==="seed"?seedLayout:layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX,...(counted.includes(entryPoint)?{umCountedJobs:entryPoint==="seed"?UNIFORM_MIXED_COUNTED.residentAll:UNIFORM_MIXED_COUNTED.all}:{})}}}));}));
    this.pipelines.set("seedHeld",await uniformDetailPipeline(this.device,this.ownership,{layout:seedLayout,compute:{module,entryPoint:"seed",constants:{umDispatchX:this.ownership.dispatchX,umCountedJobs:UNIFORM_MIXED_COUNTED.residentAll,planRetain:1}}}));
  }
  /** retain: the advance's own encodes after the frame head's heat encode
   * (seedHeld keeps COLD tiles' flags). Any other encode (a remap, an edit,
   * a host extension) seeds every tile and ends every held stretch. */
  encode(encoder:GPUCommandEncoder,policy=DEFAULT_POLICY,retain=false):void{
    if(this.pipelines.size!==12)throw new Error("Mixed frame plan is not initialized");
    if(!this.heldGroup)throw new Error("Mixed frame plan: no region work tiers are attached (attachHeat)");
    const s=this.supportSettings,words=this.policyWords??=new Uint32Array(4);
    s[0]=policy.fineReach;s[1]=policy.shellReach;s[2]=!policy.twoLevel?3:!policy.shellOnly?2:0;s[3]=0;
    words[0]=s[0];words[1]=s[1];words[2]=s[2];words[3]=0;
    this.supportRevision=this.ownership.revision;
    this.device.queue.writeBuffer(this.params,0,words);
    encoder.clearBuffer(this.ownership.support,this.ownership.capacity.tiles*16,64);
    // Seed flags: absent pages' tiles keep this zero (the seed skips them).
    encoder.clearBuffer(this.ownership.support,0,this.ownership.capacity.tiles*4);
    const pass=encoder.beginComputePass(PLAN_PASS);
    pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.group.group);pass.setBindGroup(2,this.extendedGroup.group);
    pass.setBindGroup(3,this.heldGroup);
    this.dispatch(pass,retain?this.launchesOf().held:this.launchesOf().support);
    pass.end();
  }
  /** The advance's tail: the mid-frame plan's support planes stand for the
   * next head unless ownership or the support policy changed since (then a
   * full plan). The certificate takes its speed from the extended field. */
  encodeTail(encoder:GPUCommandEncoder,policy=DEFAULT_POLICY):void{
    const s=this.supportSettings;
    if(!this.surfaceOnly||this.supportRevision!==this.ownership.revision||policy.fineReach!==s[0]||policy.shellReach!==s[1]||(!policy.twoLevel?3:!policy.shellOnly?2:0)!==s[2])this.encode(encoder,policy);
  }
  /** Fine-tile certificate from the extended field. Encode after extension
   * and before any certified/merged consumer (surface, momentum). */
  /** Each tile's certificate word after encodeCertificate (speeds[T, 2T)):
   * signed reach and, for h tiles, the class (UNIFORM_STAGE_CERTIFICATE). */
  get certificate():{readonly buffer:GPUBuffer;readonly offset:number}{return this.certificateView??={buffer:this.ownership.speeds,offset:4*this.ownership.capacity.tiles};}
  encodeCertificate(encoder:GPUCommandEncoder,dt:number):void{
    // Only the certificate reads dt: support and extension are dt-free.
    const step=this.stepWords??=new Float32Array(4);step[0]=dt;
    this.device.queue.writeBuffer(this.params,16,step);
    encoder.clearBuffer(this.ownership.regionVelocity);
    // The spread's speed (localSpeed) and the two list counts (certify).
    encoder.clearBuffer(this.ownership.support,this.ownership.capacity.tiles*16,12);
    const pass=encoder.beginComputePass(CERTIFICATE_PASS);
    pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.group.group);pass.setBindGroup(2,this.extendedGroup.group);
    // Certified consumers read the list counts themselves (umCertifiedJobCount).
    this.dispatch(pass,this.launchesOf().certificate);
    pass.end();
  }
  private dispatch(pass:GPUComputePassEncoder,launches:readonly PlanLaunch[]):void{
    for(let i=0;i<launches.length;i++){
      const launch=launches[i]!;
      if(launch.counted){this.ownership.dispatchAllCounted(pass,launch.pipeline);continue;}
      pass.setPipeline(uniformDetailPick(launch.pipeline));pass.dispatchWorkgroups(launch.x,launch.y);
    }
  }
  /** The lattice, its tiles and the dispatch stride are the ownership's for
   * life, and the pipelines initialize()'s: every grid is known once. The
   * build is its own function: one holding a closure allocates that
   * closure's context at every entry, taken or not. */
  private launchesOf():NonNullable<UniformMixedFramePlan["launches"]>{return this.launches??=this.buildLaunches();}
  private buildLaunches():NonNullable<UniformMixedFramePlan["launches"]>{
    const of=(entry:string):PlanLaunch=>{
      const pipeline=this.pipelines.get(entry)!;
      if(counted.includes(entry))return {pipeline,counted:true,x:0,y:0};
      // coarseBlocks: a lane per block. The rest: a lane per tile key.
      if(entry==="coarseBlocks")return {pipeline,counted:false,x:Math.ceil(this.ownership.capacity.address.blocks/64),y:1};
      const [x,y]=uniformKeyGroups(this.ownership.capacity.tiles,64,this.ownership.dispatchX,[0,0]);
      return {pipeline,counted:false,x,y};
    };
    const support=SUPPORT_ENTRIES.map(of);
    return {support,held:[{...support[0]!,pipeline:this.pipelines.get("seedHeld")!},...support.slice(1)],certificate:CERTIFICATE_ENTRIES.map(of)};
  }
  destroy():void{this.params.destroy();this.coarse.destroy();this.held.destroy();}
}
