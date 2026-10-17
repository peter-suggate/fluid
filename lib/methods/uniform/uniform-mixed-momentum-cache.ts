import { uniformMixedExtensionSnapshotWGSL } from "./uniform-mixed-extension-slots.wgsl";
import { uniformDetailBindLayout, uniformDetailClass, uniformDetailExtent, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, type UniformDetailStorage, uniformDetailGroup, type UniformDetailGroup } from "./uniform-detail-fields";
import {UNIFORM_DETAIL_4H_LOAD} from "../../core/uniform-detail-abi";
import { uniformCoarseCacheExtent, uniformCoarseCacheGroups, uniformCoarseCacheWGSL } from "./uniform-coarse-velocity-cache";
import { uniformCompiledVelocityTapWGSL } from "./uniform-compiled-velocity-taps.wgsl";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedCountedEntriesWGSL, uniformMixedTopologyNeighboursWGSL, uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedFaceAddressWGSL } from "./uniform-mixed-face-dispatch.wgsl";
import { UNIFORM_MIXED_HANGING_TAPS, UNIFORM_UNIT_HALO_EDGE, uniformMixedHangingTapWGSL, uniformMixedVelocitySamplingSource, uniformUnitHaloExtent, uniformUnitHaloWGSL } from "./uniform-mixed-velocity-sampling.wgsl";

/** Workgroups a block of the unit halo fill: one a z-plane of its texels. */
const HALO_SLICES=UNIFORM_UNIT_HALO_EDGE;
const CACHE_PASS:GPUComputePassDescriptor={label:"Uniform mixed 4h sampling cache"},TAPS_PASS:GPUComputePassDescriptor={label:"Uniform mixed hanging fine taps"};

export interface UniformMixedMomentumCacheFields {
  extended:GPUTexture;
  slots?:GPUBufferBinding;
  /** With slots: a uniform whose first word is the live sample count
   * (UniformNarrowBandFlip.liveGate). A slot fill feeds only the samples'
   * old-epoch snapshot, so it does nothing at zero. */
  gate?:GPUBuffer;
  negative:GPUBuffer;
  coarseExtended:GPUTexture;
}
/** 4h MAC sampling restriction of the extended velocity, including negative
 * domain faces. These are interpolation cache values, never an independently
 * advanced velocity grid. The borrowed RGBA output is keyed by block
 * (uniform-coarse-velocity-cache.ts): each stored block's 8³ tiles and the
 * tile around them, so a reader's taps are plain offsets from its own block.
 * Fine patches are averaged once per update. */
export class UniformMixedMomentumCache {
  readonly allocatedBytes=0;
  private pipeline?:GPUComputePipeline;
  private slotPipeline?:GPUComputePipeline;
  private readonly slotResources?:GPUBindGroupLayout;
  private readonly slotGroups=new WeakSet<UniformDetailGroup>();
  private readonly resources:GPUBindGroupLayout;
  /** Workgroups of the cache launch: a lane a texel of the block grid's
   * texture, whatever its blocks hold. */
  private readonly groups:readonly [number,number,number];
  constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership){
    this.groups=uniformCoarseCacheGroups(ownership.capacity.address);
    const entries:GPUBindGroupLayoutEntry[]=[
      {binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
      {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:2,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"3d"}},
    ];this.resources=uniformDetailBindLayout(device,{entries});
    if(device.limits.maxStorageBuffersPerShaderStage>=9)this.slotResources=uniformDetailBindLayout(device,{entries:[...entries,{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]});
  }
  bind(f:UniformMixedMomentumCacheFields):UniformDetailGroup{
    const blocks=uniformCoarseCacheExtent(this.ownership.capacity.address);
    for(const [i,t] of [f.extended,f.coarseExtended].entries())
      if((i===0?uniformDetailClass(t)!=="face":uniformDetailExtent(t).some((n,a)=>n!==blocks[a]))||t.format!=="rgba32float")throw new Error("Mixed momentum cache requires native inputs and a 4h output of 10 texels a block");
    if(f.slots&&!this.slotResources)throw new Error("Slot snapshot is unsupported");
    if(f.slots&&!f.gate)throw new Error("A slot fill needs the live sample gate");
    const group=uniformDetailGroup(this.device,{layout:f.slots?this.slotResources!:this.resources,entries:[
      {binding:0,resource:f.extended},
      {binding:1,resource:{buffer:f.negative}},
      {binding:2,resource:f.coarseExtended},...(f.slots?[{binding:3,resource:f.slots},{binding:4,resource:{buffer:f.gate!,size:16}}]:[]),
    ]});if(f.slots)this.slotGroups.add(group);return group;
  }
  private initialization?:Promise<void>;
  initialize():Promise<void>{return this.initialization??=this.compile();}
  private async compile():Promise<void>{
   for(const slots of this.slotResources?[false,true]:[false]){
    const module=uniformDetailModule(this.device,{label:"Uniform mixed momentum cache",code:uniformMixedTopologyWGSL(this.ownership.capacity,0)+uniformMixedTopologyNeighboursWGSL()+/* wgsl */`
@group(1) @binding(0) var extended:texture_3d<f32>;
@group(1) @binding(1) var<storage,read> negative:array<f32>;
@group(1) @binding(2) var coarseExtended:texture_storage_3d<rgba32float,write>;
${uniformMixedFaceAddressWGSL}
${slots?uniformMixedExtensionSnapshotWGSL(1,3)+"@group(1) @binding(4) var<uniform> umSlotGate:vec4u;":""}
fn umCacheExtended(anchor:vec3i,axis:u32)->f32 {
 // The low-face table is keyed by the cell above the face.
 if(umFaceLow(anchor,axis)){var above=anchor;above[axis]+=1;return negative[umLowFaceIndex(vec3u(above),axis)];}
 return ${slots?"ueSnapshotFace(anchor,axis)":"textureLoad(extended,anchor,0)[axis]"};
}
// One lane per texel: 10 an axis for every block of the grid, a stored
// block's 8³ tiles and the tile around them (uniform-coarse-velocity-cache.ts).
// A block that holds no page is never written. A halo texel holds the faces
// a tap from this page reads there: the tile's own beside a stored or an AIR
// page, the page's edge tile's where a wall holds the tap (umTileNearFromPage)
// and, below the page along the face's own axis, always the tile's own: the
// low face of the page's first tile. A face with no stored cell on either
// side is zero.
@compute @workgroup_size(5,5,5) fn cache(@builtin(global_invocation_id) gid:vec3u){
 ${slots?"if(umSlotGate.x==0u){return;}":""}
 let cell=gid/10u;if(any(cell>=UM_BLOCK_GRID)){return;}
 let b=cell.x|(cell.y<<UM_BLOCK_BITS)|(cell.z<<(2u*UM_BLOCK_BITS));
 if(!umBlockStored(b)||umBlockWord(b)==UM_FREE){return;}
 let page=vec3i(umBlockPage(b));let low=page<<vec3u(3u);let tile=low+vec3i(gid%10u)-vec3i(1);
 let near=umTileNearFromPage(b,page,tile);
 // A 4h tile among 4h tiles (its stencil's widths) has whole 4h faces:
 // nothing to discover, and three loads fill the texel.
 let key=umTileKey(near);var regular=false;
 if(umKeyStored(key)){let stencil=umTileStencil(key);regular=(stencil.x>>27u)==4u&&(stencil.y>>27u)==4u;}
 var e=vec4f(0);
 for(var axis=0u;axis<3u;axis++){
  let u=(axis+1u)%3u;let v=(axis+2u)%3u;
  var id=near;if(tile[axis]<low[axis]){id[axis]=tile[axis];}
  var anchor=id*4;anchor[axis]+=3;var width=4u;
  if(!regular||tile[axis]<low[axis]){
   var owner=umOwnerAt(anchor);var sign=1;
   if(owner.width==0u){var next=anchor;next[axis]+=1;owner=umOwnerAt(next);sign=-1;if(owner.width==0u){continue;}}
   width=umFace(owner,axis,sign,0u).width;
  }
  let side=4u/width;var value=0.0;
  // A whole 4h face is its tile's +face anchor: the base block.
  if(width==4u&&!umFaceLow(anchor,axis)){e[axis]=${slots?"ueSnapshotKnown(anchor,axis,4u)":`${UNIFORM_DETAIL_4H_LOAD}textureLoad(extended,anchor,0)[axis]`};continue;}
  for(var y=0u;y<side;y++){for(var x=0u;x<side;x++){
   var q=anchor;q[u]+=i32(x*width);q[v]+=i32(y*width);value+=umCacheExtended(q,axis);
  }}e[axis]=value/f32(side*side);
 }
 textureStore(coarseExtended,vec3i(gid),e);
}`});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
    const pipeline=await uniformDetailPipeline(this.device,this.ownership,{layout:this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,slots?this.slotResources!:this.resources]}),compute:{module,entryPoint:"cache"}});
    if(slots)this.slotPipeline=pipeline;else this.pipeline=pipeline;
   }
  }
  encode(encoder:GPUCommandEncoder,group:UniformDetailGroup,shared?:GPUComputePassEncoder):void{
    if(!this.pipeline)throw new Error("Mixed momentum cache is not initialized");
    const pass=shared??encoder.beginComputePass(CACHE_PASS),g=this.groups;pass.setPipeline(uniformDetailPick(this.slotGroups.has(group)?this.slotPipeline!:this.pipeline));pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group.group);
    pass.dispatchWorkgroups(g[0],g[1],g[2]);if(!shared)pass.end();
  }
}

export interface UniformMixedHangingTapFields {
  extended:GPUTexture;
  slots?:GPUBufferBinding;
  /** With slots: the live sample gate (UniformMixedMomentumCacheFields.gate). */
  gate?:GPUBuffer;
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
  /** The taps are detail fields: their bytes are the storage's. Their halos
   * (haloOf) are not. */
  get allocatedBytes():number{let bytes=this.haloBlocks.size;for(const t of this.halos.values())bytes+=16*t.width*t.height*t.depthOrArrayLayers;return bytes;}
  /** Each taps texture's haloed copy by block (UNIFORM_UNIT_HALO_EDGE),
   * refilled by every encode that fills the taps but a slot fill. */
  private readonly halos=new Map<GPUTexture,GPUTexture>();
  private readonly haloGroups=new WeakMap<UniformDetailGroup,UniformDetailGroup>();
  private readonly haloResources:GPUBindGroupLayout;
  private haloPipeline?:GPUComputePipeline;
  private haloMarkPipeline?:GPUComputePipeline;
  /** Sixteen words a block: a bit a tile, whether a sample of one of the
   * block's tiles can read the halo texels over it (markUnitHalo, before
   * every fill). The fill skips the texels of unmarked tiles. */
  private readonly haloBlocks:GPUBuffer;
  private readonly haloBlockCount:number;
  /** The unit interpolant's taps as one native MAC texel per cell: each unit
   * tile's stored faces and each slotted tile's fine taps, written with the
   * slots. A sample with nonzero fine weight has every tap in a unit or a
   * slotted tile (a tap within 4h of a unit tile is in a tile beside it, and
   * a coarse tile beside a unit tile is a seam tile), so its unit
   * interpolant is eight direct loads, as all-fine Uniform samples. */
  readonly unitVelocity:GPUTexture;
  private pipeline?:GPUComputePipeline;
  private slotPipeline?:GPUComputePipeline;
  private readonly slotResources?:GPUBindGroupLayout;
  private readonly slotGroups=new WeakSet<UniformDetailGroup>();
  private readonly resources:GPUBindGroupLayout;
  /** detail: the solver's detail storage. The taps are read atlas-only (no
   * tile outside resident patches is ever a hanging tap), but the texture is
   * face-class: the remap borrows it as the census extension's face scratch. */
  constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,private readonly detail:UniformDetailStorage){
    const entries:GPUBindGroupLayoutEntry[]=[
      {binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
      {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:2,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
      {binding:3,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"3d"}},
    ];this.resources=uniformDetailBindLayout(device,{entries});
    if(device.limits.maxStorageBuffersPerShaderStage>=9)this.slotResources=uniformDetailBindLayout(device,{entries:[...entries,{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]});
    // The halo fill: the taps read back (5), the halo written (3).
    this.haloResources=uniformDetailBindLayout(device,{entries:[...entries,{binding:5,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},{binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]},[3]);
    const grid=ownership.capacity.address.blockGrid;this.haloBlockCount=grid[0]*grid[1]*grid[2];
    if(this.haloBlockCount>device.limits.maxComputeWorkgroupsPerDimension)throw new Error(`Unit halo marking launches a workgroup a block: ${this.haloBlockCount} blocks exceed ${device.limits.maxComputeWorkgroupsPerDimension}`);
    this.haloBlocks=device.createBuffer({label:"Uniform mixed unit halo blocks",size:64*this.haloBlockCount,usage:GPUBufferUsage.STORAGE});
    this.unitVelocity=this.createTaps("Uniform mixed unit velocity taps");
  }
  /** Another texture this fill can write: the unit taps of a second field,
   * kept beside unitVelocity (the caller destroys it; its halo is this
   * builder's). */
  createTaps(label:string):GPUTexture{
    const taps=this.detail.createField(label,"atlas","rgba32float");
    // No fine tile: no sample has unit weight, and a stand-in disables the halo.
    const size=this.ownership.capacity.fineTiles>0?uniformUnitHaloExtent(this.ownership.capacity.address):[1,1,1];
    this.halos.set(taps,this.device.createTexture({label:`${label} halo`,size,dimension:"3d",format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING}));
    return taps;
  }
  /** The haloed copy of a createTaps texture (or of unitVelocity). */
  haloOf(taps:GPUTexture):GPUTexture{const halo=this.halos.get(taps);if(!halo)throw new Error("Unit taps without a halo");return halo;}
  /** output: a createTaps texture to fill in place of unitVelocity. */
  bind(f:UniformMixedHangingTapFields,output=this.unitVelocity):UniformDetailGroup{
    const blocks=uniformCoarseCacheExtent(this.ownership.capacity.address);
    for(const [i,t] of [f.extended,f.coarse].entries())
      if((i===0?uniformDetailClass(t)!=="face":uniformDetailExtent(t).some((n,a)=>n!==blocks[a]))||t.format!=="rgba32float")throw new Error("Hanging taps require native velocity and the 4h sampling cache");
    if(f.slots&&!this.slotResources)throw new Error("Slot snapshot is unsupported");
    if(f.slots&&!f.gate)throw new Error("A slot fill needs the live sample gate");
    const group=uniformDetailGroup(this.device,{layout:f.slots?this.slotResources!:this.resources,entries:[
      {binding:0,resource:f.extended},{binding:1,resource:{buffer:f.negative}},{binding:2,resource:f.coarse},
      {binding:3,resource:output},...(f.slots?[{binding:4,resource:f.slots},{binding:5,resource:{buffer:f.gate!,size:16}}]:[]),
    ]});if(f.slots)this.slotGroups.add(group);
    // A slot fill (the paired snapshot's old-epoch taps) has no halo: the
    // slot snapshot reads its taps alone, and the update's own fill
    // refreshes the halo before the next sample can read it.
    if(!f.slots)this.haloGroups.set(group,uniformDetailGroup(this.device,{layout:this.haloResources,entries:[
      {binding:0,resource:f.extended},{binding:1,resource:{buffer:f.negative}},{binding:2,resource:f.coarse},
      {binding:3,resource:this.haloOf(output)},{binding:5,resource:output},{binding:6,resource:{buffer:this.haloBlocks}},
    ]}));
    return group;
  }
  async initialize():Promise<void>{
   {
    // One lane a texel of the halo texture, whatever its block holds
    // (uniformUnitHaloExtent): a stored block's page's taps and the cell
    // around them, each the value the unit interpolant reads for that tap.
    const code=uniformMixedTopologyWGSL(this.ownership.capacity,0)+/* wgsl */`
@group(1) @binding(0) var extended:texture_3d<f32>;
@group(1) @binding(1) var<storage,read> negative:array<f32>;
@group(1) @binding(2) var coarse:texture_3d<f32>;
@group(1) @binding(3) var unitHaloOut:texture_storage_3d<rgba32float,write>;
@group(1) @binding(5) var unitVelocity:texture_3d<f32>;
@group(1) @binding(6) var<storage,read_write> haloBlocks:array<u32>;
${uniformMixedFaceAddressWGSL}
fn umLoadMixedFace(anchor:vec3i,axis:u32)->f32{
 if(umFaceLow(anchor,axis)){var above=anchor;above[axis]+=1;return negative[umLowFaceIndex(vec3u(above),axis)];}
 return textureLoad(extended,anchor,0)[axis];
}
${uniformCoarseCacheWGSL+uniformCompiledVelocityTapWGSL+uniformUnitHaloWGSL+uniformMixedHangingTapWGSL(2)}
// Only a sample with fine weight reads a halo, and only its own tile's
// block's (umUnitHaloHome): fine weight needs a unit tile (umCachedFineTile,
// umFineStencilTile, the APIC home's unit tile) or a blend mask (the
// weighted sampler's boxes; none, weight zero). Both are layout words,
// current at every fill. A sample of tile T reads taps within two cells
// below and one above T's cells, all in T's tiles -1..+1 along each axis:
// a halo texel is read only where its tile (clamped into the page: the
// margin's texels are read by the edge tiles only) is within one tile of
// such a tile of the block. markUnitHalo marks those tiles, a bit a tile,
// sixteen words a block; the fill skips the texels of the rest.
var<workgroup> haloSources:array<atomic<u32>,16>;
var<workgroup> haloReach:array<atomic<u32>,16>;
fn umHaloBit(l:vec3u)->u32{return l.x|(l.y<<3u)|(l.z<<6u);}
@compute @workgroup_size(64) fn markUnitHalo(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let b=group.x;let live=b<UM_BLOCKS&&umBlockStored(b)&&umBlockWord(b)!=UM_FREE;
 if(lane<16u){atomicStore(&haloSources[lane],0u);atomicStore(&haloReach[lane],0u);}
 workgroupBarrier();
 if(live){
  for(var k=lane;k<512u;k+=64u){let key=512u*b+k;if(umTileWidth(key)==1u||umVelocityBlendMask(key)!=0u){atomicOr(&haloSources[k>>5u],1u<<(k&31u));}}
 }
 workgroupBarrier();
 if(live){
  for(var k=lane;k<512u;k+=64u){
   let l=umLocalTile(k);let lo=max(l,vec3u(1u))-vec3u(1u);let hi=min(l+vec3u(1u),vec3u(7u));var near=false;
   for(var z=lo.z;z<=hi.z;z++){for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){
    let n=umHaloBit(vec3u(x,y,z));near=near||(atomicLoad(&haloSources[n>>5u])&(1u<<(n&31u)))!=0u;
   }}}
   if(near){atomicOr(&haloReach[k>>5u],1u<<(k&31u));}
  }
 }
 workgroupBarrier();
 if(lane<16u){haloBlocks[16u*b+lane]=atomicLoad(&haloReach[lane]);}
}
// A block's texels, a workgroup a z-plane (y). A plane lies in one tile row:
// its 10 x 10 tiles' words (marked, unit-or-slotted, SOLID) are staged
// first, so a texel reads none of the tables; an unstored block's workgroups
// and an unmarked tile's texels write nothing.
const UM_HALO_MARKED=1u;const UM_HALO_UNIT=2u;const UM_HALO_SOLID=4u;
var<workgroup> haloTiles:array<u32,100>;
@compute @workgroup_size(64) fn fillUnitHalo(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let b=group.x;let live=b<UM_BLOCKS&&umBlockStored(b)&&umBlockWord(b)!=UM_FREE;
 let origin=(vec3i(umBlockPage(min(b,UM_BLOCKS-1u)))<<vec3u(5u))-vec3i(2);
 let z=group.y;let tz=(z+2u)/4u;
 if(live){
  for(var k=lane;k<100u;k+=64u){
   // Local tile t covers the texels with (l + 2) / 4 == t: its first, l = max(4 t, 2) - 2.
   let t=vec3u(k%10u,k/10u,tz);let tap=origin+vec3i(max(4u*t,vec3u(2u))-vec3u(2u));
   let n=umHaloBit(clamp(t,vec3u(1u),vec3u(8u))-vec3u(1u));var word=select(0u,UM_HALO_MARKED,(haloBlocks[16u*b+(n>>5u)]&(1u<<(n&31u)))!=0u);
   let key=umTileAt(vec3u(tap)/4u);
   if(umTilePresent(tap>>vec3u(2u))&&(umTileWidth(key)==1u||umHanging[key]!=UM_NO_SLOT)){word|=UM_HALO_UNIT;}
   if(umCellKind(tap)==UM_SOLID){word|=UM_HALO_SOLID;}
   haloTiles[k]=word;
  }
 }
 workgroupBarrier();
 if(!live){return;}
 for(var i=lane;i<${UNIFORM_UNIT_HALO_EDGE**2}u;i+=64u){
  let l=vec3u(i%${UNIFORM_UNIT_HALO_EDGE}u,i/${UNIFORM_UNIT_HALO_EDGE}u,z);let t=(l+vec3u(2u))/4u;
  let word=haloTiles[t.x+10u*t.y];
  if((word&UM_HALO_MARKED)!=0u){umFillUnitHaloTexel(b,origin+vec3i(l),l,word);}
 }
}
fn umFillUnitHaloTexel(b:u32,tap:vec3i,l:vec3u,word:u32){
 // The unit interpolant's tap: the texel the cell resolves to. It is
 // umVelocityTap1's too where this fill wrote it: a unit or a slotted tile.
 var value=textureLoad(unitVelocity,tap,0);
 value.w=select(0.0,1.0,(word&UM_HALO_UNIT)!=0u);
 // A SOLID cell's channel on the plane below the low wall along it: the
 // low-wall face, as umVelocityTap1 reads it: the low-face table under a
 // unit tile, the negative-plane tap (as the hanging record holds it) under
 // a coarser one. Its other channels keep the SOLID texel (zero), as the
 // unit sampler's direct loads read them.
 if((word&UM_HALO_SOLID)!=0u){
  value.w=1.0;let wall=umWallLow();let anchor=max(tap,wall);let tile=umTileAt(vec3u(anchor)/4u);let unit=umTileWidth(tile)==1u;
  for(var axis=0u;axis<3u;axis++){
   if(tap[axis]!=wall[axis]-1){continue;}
   if(unit){var face=anchor;face[axis]=tap[axis];value[axis]=umLoadMixedFace(face,axis);}
   else{value[axis]=umCompiledVelocityTap(tile,vec3u(anchor&vec3i(3)),axis,true);}
  }
 }
 textureStore(unitHaloOut,${UNIFORM_UNIT_HALO_EDGE}*umBlockCoord(b)+vec3i(l),value);
}`;
    const module=uniformDetailModule(this.device,{label:"Uniform mixed unit tap halo",code});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
    const haloLayout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.haloResources,this.ownership.hangingLayout]});
    [this.haloMarkPipeline,this.haloPipeline]=await Promise.all(["markUnitHalo","fillUnitHalo"].map(entryPoint=>uniformDetailPipeline(this.device,this.ownership,{layout:haloLayout,compute:{module,entryPoint}})));
   }
   for(const slots of this.slotResources?[false,true]:[false]){
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
${slots?uniformMixedExtensionSnapshotWGSL(1,4)+"@group(1) @binding(5) var<uniform> umSlotGate:vec4u;":""}
// A slot fill launches no job without a live sample (a uniform).
fn umSlotLive()->bool{return ${slots?"umSlotGate.x!=0u":"true"};}
fn umLoadMixedFace(anchor:vec3i,axis:u32)->f32{
 // The low-face table is keyed by the cell above the face.
 if(umFaceLow(anchor,axis)){var above=anchor;above[axis]+=1;return negative[umLowFaceIndex(vec3u(above),axis)];}
 return ${slots?"ueSnapshotFace(anchor,axis)":"textureLoad(extended,anchor,0)[axis]"};
}
${uniformCoarseCacheWGSL+uniformMixedHangingTapWGSL(2)+uniformCompiledVelocityTapWGSL}
// Tile-local lanes: 0..191 fine taps (stored to unitVelocity below),
// 192..239 fine negative boundary plane taps (for tiles on that plane).
var<workgroup> unitTaps:array<f32,192>;
fn umHangingFill(slot:u32,tile:u32,lane:u32){
 if(lane<192u){
  let cell=lane%64u;let axis=lane/64u;let local=vec3u(cell%4u,(cell/4u)%4u,cell/16u);
  let value=umCompiledVelocityTap(tile,local,axis,false);
  unitTaps[lane]=value;
 }else if(lane<${192+UNIFORM_MIXED_HANGING_TAPS}u){
  let offset=lane-192u;let axis=offset/16u;let cell=offset%16u;
  if(!umLowOpen(vec3i(4u*umTileCoord(tile)))[axis]){return;}
  var local=vec3u(0);local[(axis+1u)%3u]=cell%4u;local[(axis+2u)%3u]=cell/4u;
  umHanging[umHangingPlaneAddress(slot,local,axis)]=bitcast<u32>(umCompiledVelocityTap(tile,local,axis,true));
 }
}
// Seam 4h slots follow the seam h slots (the builder's and update()'s order);
// slots past the cache are never visited (the builder flags UM_OVERFLOW_HANGING).
fn umUnitJobs()->u32{return ${slots?"0u":"(umCounts.x+3u)/4u"};}
fn umCoarseSlots()->vec2u{let header=7u*UM_TILES+16u;let fine=umSupport[header];let most=umHangingSlots();return vec2u(min(fine,most),min(fine+umSupport[header+1u],most));}
@compute @workgroup_size(256) fn unitVelocityTaps(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let units=umUnitJobs();
 if(group.x<units){
  let k=4u*group.x+lane/64u;
  if(k<umCounts.x){let tile=umTopology[UM_TILES+k];let cell=vec3i(umTileCoord(tile)*4u+umCorner(lane%64u,4u));textureStore(unitVelocity,cell,textureLoad(extended,cell,0));}
 }else{
  let slot=umCoarseSlots().x+group.x-units;let tile=umHanging[UM_TILES+slot];
  if(tile!=UM_NO_SLOT){umHangingFill(slot,tile,lane);}
  workgroupBarrier();
  if(tile!=UM_NO_SLOT&&lane<64u){textureStore(unitVelocity,vec3i(umTileCoord(tile)*4u+umCorner(lane,4u)),vec4f(unitTaps[lane],unitTaps[lane+64u],unitTaps[lane+128u],0.0));}
 }
}`,["unitVelocityTaps"],"select(0u,umUnitJobs()+umCoarseSlots().y-umCoarseSlots().x,umSlotLive())");
    const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,slots?this.slotResources!:this.resources,this.ownership.hangingLayout]});
    const module=uniformDetailModule(this.device,{label:"Uniform mixed hanging taps",code});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
    const pipeline=await uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module,entryPoint:"unitVelocityTaps",constants:{umDispatchX:this.ownership.dispatchX}}});
    if(slots)this.slotPipeline=pipeline;else this.pipeline=pipeline;
   }
  }
  encode(encoder:GPUCommandEncoder,group:UniformDetailGroup,shared?:GPUComputePassEncoder):void{
    if(!this.pipeline)throw new Error("Mixed hanging taps are not initialized");
    // A fixed grid from capacity; the GPU counts the h tiles (forces
    // viscosity reads every unit tile's texels, so they are refreshed even
    // without a slot) and the seam 4h slots. At most one job per tile.
    const pass=shared??encoder.beginComputePass(TAPS_PASS);pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group.group);pass.setBindGroup(2,this.ownership.hangingGroup);
    this.ownership.dispatchCounted(pass,this.slotGroups.has(group)?this.slotPipeline!:this.pipeline,this.ownership.capacity.tiles,"fineAndSeams");
    // Every fill but a slot fill refreshes its halo: the texels samples can read (markUnitHalo),
    // a fixed grid of HALO_SLICES workgroups a block.
    const halo=this.haloGroups.get(group);
    if(halo){
      pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,halo.group);pass.setBindGroup(2,this.ownership.hangingGroup);
      pass.setPipeline(uniformDetailPick(this.haloMarkPipeline!));pass.dispatchWorkgroups(this.haloBlockCount);
      pass.setPipeline(uniformDetailPick(this.haloPipeline!));pass.dispatchWorkgroups(this.haloBlockCount,HALO_SLICES);
    }
    if(!shared)pass.end();
  }
  destroy():void{this.unitVelocity.destroy();for(const halo of this.halos.values())halo.destroy();this.haloBlocks.destroy();}
}
