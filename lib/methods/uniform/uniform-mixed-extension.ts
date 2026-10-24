import { UNIFORM_COLD_FREEZE } from "./uniform-cold-freeze";
import { uniformMixedExtensionSlotWGSL } from "./uniform-mixed-extension-slots.wgsl";
import { uniformCompiledExtensionNeighborWGSL } from "./uniform-compiled-extension.wgsl";
import { pairedExtensionSource, pairedSnapshotSource, replayedPublicationSource } from "./uniform-mixed-extension-pair.wgsl";
import { uniformDetailBindLayout, uniformDetailClass, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, uniformDetailGroup, type UniformDetailGroup } from "./uniform-detail-fields";
import type { UniformLatticeAddress } from "./uniform-lattice-address";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformExtrapolationWorkBytes, type WebGPUUniformVelocityExtrapolator } from "./webgpu-uniform-velocity-extrapolation";
import { UNIFORM_EXTRAPOLATION_WORK_WORDS } from "./webgpu-uniform-velocity-extrapolation.wgsl";
import { UNIFORM_MIXED_COUNTED,uniformMixedCountedEntriesWGSL,uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { UNIFORM_LIVE_GROUP_GROUPS, UNIFORM_LIVE_KEY_GROUPS, UNIFORM_LIVE_LANE_GROUPS } from "./uniform-live-lists";
import { uniformStageListsWGSL } from "./uniform-stage-lists";
import { uniformMixedFaceAddressWGSL,uniformMixedFaceDispatchWGSL } from "./uniform-mixed-face-dispatch.wgsl";
import { uniformMixedVelocitySamplingSource } from "./uniform-mixed-velocity-sampling.wgsl";
import { uniformRegionHeatWGSL } from "./uniform-region-heat";
import { UNIFORM_BODY_GEOMETRY_REACH } from "./uniform-mixed-solid.wgsl";

type Hierarchy=ReturnType<WebGPUUniformVelocityExtrapolator["prepareMixedContinuation"]>;
interface Fields {
 physical:GPUTexture;phase:GPUTexture;negative:GPUBuffer;output:GPUTexture;outputNegative:GPUBuffer;
 params:GPUBuffer;
 /** A uniform whose first word is the frame's live sample count
  * (UniformNarrowBandFlip.liveGate): the old epoch's retain and snapshot do
  * nothing at zero, and publication then never replays its selections.
  * Without one, samples are always taken as live. */
 gate?:GPUBuffer;
 companion?:{scratch:GPUBuffer;physical:GPUTexture;negative:GPUBuffer};
 /** Existing FLIP fine-tap texture for finalized-slot publication. */
 snapshot?:GPUTexture;
}
/** A patch of generated shader text: `anchor` occurs exactly once. */
function patchOnce(source:string,anchor:string,to:string):string{
 const count=source.split(anchor).length-1;
 if(count!==1)throw new Error(`Mixed extension patch expected 1 occurrence of ${JSON.stringify(anchor)}, found ${count}`);
 return source.replace(anchor,()=>to);
}
/** How the pyramid's origin textures hold a source bound. */
export const uniformExtensionSourceBoundsWGSL=/* wgsl */`// A source bound is a cell of the page window, 11 bits an axis: x, y and the
// low ten bits of z in the origin texel's component, the eleventh bit of z in
// bit 24 + component of the texel's w. The component's mask bit in the value
// texel says whether a bound is there. The pyramid's prolongation reads them.
fn umSourceWord(p:vec3i)->u32{let q=vec3u(p);return q.x|(q.y<<11u)|((q.z&1023u)<<22u);}
// The pyramid's 4h level holds block b's tiles at umBlockCoord(b) in the
// block grid (umTileStoreTexel); a texel's upper source bounds lie this far
// above its lower ones.
const UE_UPPER_BOUNDS:vec3i=vec3i(0,0,i32(8u*UM_BLOCK_GRID.z));
`;
/** umFarValue: the trilinear combination of the known values among the
 * eight 4h pyramid cells around a face. A cell of a page that is not stored
 * holds nothing. */
export function uniformExtensionFarValueWGSL(name:string):string{
 return /* wgsl */`// A face no sweep reached. A 4h cell's value sits at its tile's +face centre:
// the face takes the trilinear combination of the cells around its own
// centre, over the taps that are known and renormalised by their weight, so
// it stays between the least and the greatest of them. A face on a cell
// plane (a tile's -face sits at the cell below's +face centre) has taps of
// zero weight across that plane: only where no tap of positive weight is
// known does a second pass take those, weighted by their tangential
// fractions (each zero factor floored to 2^-10, below every positive one, so
// a positive weight is unchanged). A trailing tile's -face then takes the
// tile above's known value rather than zero. Zero with none.
fn ${name}(face:UMFace)->f32{
 let point=umFaceCenter(face);var q=point/4.0-vec3f(0.5);q[face.axis]-=0.5;
 let base=vec3i(floor(q));let fraction=fract(q);var sum=0.0;var total=0.0;
 for(var farPass=0u;farPass<2u&&total<=0.0;farPass++){for(var k=0u;k<8u;k++){
  let bit=vec3i(umCorner(k,2u));let w=select(vec3f(1)-fraction,fraction,bit==vec3i(1));if((w.x*w.y*w.z>0.0)!=(farPass==0u)){continue;}
  let floored=max(w,vec3f(0.0009765625));let weight=floored.x*floored.y*floored.z;
  let texel=umTileStoreTexel(base+bit);if(texel.x<0){continue;}
  let state=textureLoad(coarse,texel,0);
  if((u32(round(state.w))&(1u<<face.axis))==0u){continue;}
  sum+=weight*state[face.axis];total+=weight;
 }}
 return select(0.0,sum/total,total>0.0);
}
`;
}
/** (value, distance) slots per canonical patch and component at `fineTiles`
 * h tiles. Fixed by the lattice: width-4 patches in a component-major n/4
 * layer, the negative domain-wall planes of each width, a two-word
 * finite-cell mask per tile (UE_FLAGS) and the no-patch slot. Then the unit
 * patches by fine-tile rank, 240 an h tile. See umSlot. */
export function extensionSlots(a:UniformLatticeAddress,fineTiles:number):number{
 // The low-face planes: per h face a low-face table's keys (UM_LOW_FACE_KEYS), per 4h face three planes of 64 a block.
 return 5*a.tileKeys+a.lowFaceKeys+192*a.blocks+1+240*fineTiles;
}
/** One scalar companion per slot and epoch at `fineTiles` h tiles (companionBytesAt). */
export function extensionCompanionBytes(a:UniformLatticeAddress,fineTiles:number):number{
 return 2*Math.ceil(Math.ceil(8*extensionSlots(a,fineTiles)/256)*256/2/256)*256;
}
/** Seam 4h tiles per sweepSeams job (ueSweepCoarsePack). */
const UE_PACK_TILES=2;
/** Regular-bulk mode restricts physical source faces directly to the 4h
 * nearest-source hierarchy, then retains supported h faces on publication.
 * The legacy mode first performs Godunov sweeps on canonical mixed patches
 * in two vec2f slot arrays the extension owns: they persist from one swept
 * encode to the next (the regional head rebuild reads the last one's). */
/** A regional rebuild (the head after bodies moved) follows the frame's
 * tail, which extended this same velocity: since then only a moved body
 * rewrote phase or support (G2P moves particles alone; a page claim or remap
 * ends the retention). A 16h region is changed where a tile's topology word
 * or support bit moved, or where it is not COLD (UniformRegionHeat) and a
 * body's swept support lies within UNIFORM_EXTENSION_BODY_REACH h of it
 * (UniformMixedSolid.bodyRegions): a COLD region is maintained at 16h.
 * A full encode marks, besides, the regions within one region of a non-COLD
 * one (or one whose words moved): only there does the tail publish the old
 * FLIP payload (state 3). G2P takes no FLIP delta at a COLD region's samples
 * (they blend toward the current payload alone), so no COLD sample reads the
 * old payload, and a non-COLD sample's taps lie within one cell of its region. */
const UNIFORM_EXTENSION_BODY_REACH=UNIFORM_BODY_GEOMETRY_REACH+1;
/** Per-frame pass descriptors, built once. */
const EXTENSION_PASS:GPUComputePassDescriptor={label:"Uniform mixed extension"},MARK_PASS:GPUComputePassDescriptor={label:"Uniform mixed extension regions"},RETAIN_PASS:GPUComputePassDescriptor={label:"Uniform mixed extension retain"},
 SNAPSHOT_PASS:GPUComputePassDescriptor={label:"Uniform mixed extension slot snapshot"};
/** ceil(jobs/per) in whole steps: a fractional quotient is a heap number until its caller is optimized. */
const ceilDiv=(jobs:number,per:number):number=>{const up=jobs+per-1;return (up-up%per)/per;};
export class UniformMixedExtension {
 /** The topology stamp and the mark passes' mode words (the slots are the
  * caller's: useSlots). */
 get allocatedBytes():number{return this.stamp.size+16*this.modes.length+this.alwaysLive.size+this.work.size+this.noBodies.size+(this.held?.size??0);}
 /** The held values' bytes at `fineTiles` h tiles: one f32 a slot (heldBytesAt). */
 heldBytesAt(fineTiles:number):number{return Math.ceil(4*extensionSlots(this.address,fineTiles)/256)*256;}
 /** One slot array at `fineTiles` h tiles. */
 private slotBytesAt(fineTiles:number):number{return Math.ceil(8*extensionSlots(this.address,fineTiles)/256)*256;}
 /** The region states after one slot array: a vec2f a 16h region. */
 private get regionBytes():number{return Math.ceil(8*(this.ownership.capacity.tiles/64+1)/256)*256;}
 /** The slot buffer's prefix at `fineTiles` h tiles: both slot arrays,
  * each followed by the region states (useSlots). */
 slotBufferBytesAt(fineTiles:number):number{return 2*(this.slotBytesAt(fineTiles)+this.regionBytes);}
 /** One scalar companion per slot and epoch, borrowed until G2P resampling. */
 companionBytesAt(fineTiles:number):number{return extensionCompanionBytes(this.address,fineTiles);}
 private readonly resources:GPUBindGroupLayout;
 private readonly pairedResources?:GPUBindGroupLayout;
 private readonly snapshotResources?:GPUBindGroupLayout;
 private snapshotPipelines?:{restrict:GPUComputePipeline;fine:GPUComputePipeline;coarse:GPUComputePipeline};
 private replayPublishers?:{regular:Map<string,GPUComputePipeline>;list:Map<string,GPUComputePipeline>;seams:Map<string,GPUComputePipeline>};
 /** The encoder holding a finished slot snapshot, and its sweep count. */
 private snapshotEncoder?:GPUCommandEncoder;
 private snapshotSweeps=0;
 get supportsPair():boolean{return this.pairedResources!==undefined&&!this.regularBulk;}
 /** GPU-counted launches (UNIFORM_MIXED_COUNTED), each of a fixed grid: no
  * launch size, variant or skip reads the host membership. Regular h owners. */
 private readonly regularPipelines=new Map<string,GPUComputePipeline>();
 /** Every regular 4h owner, 64 per job, fused tier or not. */
 private readonly listPipelines=new Map<string,GPUComputePipeline>();
 /** Seed/publication share a seam launch. Sweeps separate h seams from
  * regular and packed seam 4h jobs, so h jobs compile with one staged tile. */
 private readonly seamPipelines=new Map<string,GPUComputePipeline>();
 private fineSeamSweep?:GPUComputePipeline;
 private restrictPipeline?:GPUComputePipeline;
 private readonly pairedRegular=new Map<string,GPUComputePipeline>();
 private readonly pairedList=new Map<string,GPUComputePipeline>();
 private readonly pairedSeams=new Map<string,GPUComputePipeline>();
 private pairedFineSweep?:GPUComputePipeline;
 private retainPipelines?:readonly [GPUComputePipeline,GPUComputePipeline];
 private readonly companionRegular=new Map<string,GPUComputePipeline>();
 private readonly companionList=new Map<string,GPUComputePipeline>();
 private readonly companionSeams=new Map<string,GPUComputePipeline>();
 private companionRestrict?:GPUComputePipeline;
 private readonly address:UniformLatticeAddress;
 /** restrictBand's fixed grid: 64 live tiles a workgroup job. */
 private readonly restrictGroups:number;
 /** The gate of a bind without one: always live. */
 private readonly alwaysLive:GPUBuffer;
 /** The slot arrays (slotBufferBytesAt), created by bind() at the
  * ownership's capacity. */
 private slots?:GPUBuffer;
 private slotArray=0;
 /** Per tile key the topology word and extension support bit the last swept
  * encode saw, per 16h region its work tier then and whether it changed. */
 private readonly stamp:GPUBuffer;
 /** The marks' work words (the hierarchy's heWork): [0] a region is dirty,
  * [1] a region publishes the old FLIP payload; then the held blocks the
  * hierarchy strides, listed before each of its encodes (encodeBlocks). */
 private readonly work:GPUBuffer;
 private readonly blocksLayout:GPUBindGroupLayout;
 private readonly blocksGroup:GPUBindGroup;
 private blocksPipeline?:GPUComputePipeline;
 /** The block list's fixed grid: a held block a lane. */
 private readonly blockGroups:number;
 /** The body mask of a frame without solids: no region near a body. */
 private readonly noBodies:GPUBuffer;
 private bodyRegions?:GPUBuffer;
 /** The mark passes' modes: every region dirty, then regional with the
  * final slots in array 0 or in array 1, then held (the tail's) likewise. */
 private readonly modes:readonly GPUBuffer[];
 private readonly markLayout:GPUBindGroupLayout;
 private markGroups?:readonly GPUBindGroup[];
 private markPipelines?:{stamp:GPUComputePipeline;classify:GPUComputePipeline};
 /** The slot arrays hold the final slots of the last swept encode and the
  * stamp its topology: a regional encode may keep the slots of its clean
  * regions. Cleared by bind() and invalidateRetained(). */
 private retained=false;
 /** The array holding them: the last swept encode's sweep count's parity. */
 private retainedParity=0;
 /** Per canonical slot the value its last publication from swept slots
  * stored (any publication but a held region's): what a held region
  * publishes for its unknown faces. Allocated with the slots (useSlots). */
 private held?:GPUBuffer;
 /** A swept encode has marked every region since the slots were lent or
  * bound: the tail may hold (every region's validity word is its own). */
 private heldReady=false;
 /** The last pair encode's marks held its quiet regions (finishPair's
  * hierarchy then runs only with a dirty region). */
 private heldTail=false;
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,readonly hierarchy:Hierarchy,private readonly regularBulk=false,directRestriction=false,private readonly coldFreeze=false){
  if(ownership.capacity.lattice.dimensions.some(n=>n%4!==0))throw new Error("Mixed extension requires a 4-aligned lattice");
  if(!directRestriction)throw new Error("Mixed extension restricts the face-plane anchors directly");
  this.address=ownership.capacity.address;this.restrictGroups=UNIFORM_LIVE_KEY_GROUPS;
  this.alwaysLive=device.createBuffer({label:"Uniform mixed extension live gate",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(this.alwaysLive,0,new Uint32Array([1,0,0,0]));
  const tiles=ownership.capacity.tiles,regions=tiles/64;
  this.stamp=device.createBuffer({label:"Uniform mixed extension topology stamp",size:4*(2*tiles+2*regions),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  this.work=device.createBuffer({label:"Uniform mixed extension work words",size:uniformExtrapolationWorkBytes(this.address.blocks),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(this.work,0,new Uint32Array([1,1,0,0]));
  this.blocksLayout=device.createBindGroupLayout({label:"Uniform mixed extension held blocks",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
  this.blockGroups=UNIFORM_LIVE_LANE_GROUPS;
  this.blocksGroup=device.createBindGroup({label:"Uniform mixed extension held blocks",layout:this.blocksLayout,entries:[{binding:0,resource:{buffer:this.work}}]});
  this.noBodies=device.createBuffer({label:"Uniform mixed extension no bodies",size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(this.noBodies,0,new Uint32Array([~0,~0,~0,~0]));
  this.modes=[[0,0],[1,0],[1,1],[2,0],[2,1]].map(([regional,parity])=>{const b=device.createBuffer({label:"Uniform mixed extension region mode",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(b,0,new Uint32Array([regional!,parity!,0,0]));return b;});
  this.markLayout=device.createBindGroupLayout({label:"Uniform mixed extension regions",entries:[
   ...[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}})),
   {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
  ]});
  // The hierarchy's kernels read the page pool's tail of the ownership's buffer.
  hierarchy.attach(ownership.presentation.buffer,ownership.pageMirror.offset,this.work);
  const entries:GPUBindGroupLayoutEntry[]=[
   ...[0,1].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}})),
   ...[4,5,9].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
   {binding:10,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"uint",viewDimension:"3d"}},
   {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   ...[8,11].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only" as const,format:"rgba32float" as const,viewDimension:"3d" as const}})),
   {binding:12,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32uint",viewDimension:"3d"}},
   {binding:13,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
   {binding:17,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
  ];
  // The held values: the main layout's alone (the paired and snapshot
  // modules publish no current payload: their accessors are stubs).
  this.resources=uniformDetailBindLayout(device,{entries:[...entries,{binding:18,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
  if(device.limits.maxStorageBuffersPerShaderStage>=9){const paired:GPUBindGroupLayoutEntry[]=[...entries,
   ...[2,3].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}})),
   {binding:14,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
   {binding:15,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
  ];this.pairedResources=uniformDetailBindLayout(device,{entries:paired});
   this.snapshotResources=uniformDetailBindLayout(device,{entries:[...paired,{binding:16,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"3d"}}]},[16]);
  }
 }
 /** Bind the slots to the prefix of `buffer` at the ownership's capacity.
  * The caller lends a buffer other stages share, under one rule: between
  * the closing swept encode of a frame (the tail's) and the next head's
  * encode nothing else writes it, and the caller calls invalidateRetained
  * where its other writers begin (a head's publication has been read by
  * then: the slots are dead until the tail seeds them again). A new buffer
  * or capacity retains nothing. */
 private heatTiers?:GPUBuffer;
 /** The frame's region work tiers (UniformRegionHeat.tiers) and, with
  * solids, the swept-body region mask (UniformMixedSolid.bodyRegions), before useSlots. */
 attachHeat(tiers:GPUBuffer,bodyRegions?:GPUBuffer):void{
  if(tiers.size<this.ownership.capacity.tiles/16)throw new Error("Mixed extension: the work tiers hold fewer regions than the ownership");
  if(bodyRegions&&bodyRegions.size<this.ownership.capacity.tiles/16)throw new Error("Mixed extension: the body mask holds fewer regions than the ownership");
  this.heatTiers=tiers;this.bodyRegions=bodyRegions;
 }
 useSlots(buffer:GPUBuffer):void{
  if(!this.heatTiers)throw new Error("Mixed extension: no region work tiers are attached (attachHeat)");
  const bytes=this.slotBufferBytesAt(this.ownership.capacity.fineTiles);
  if(buffer.size<bytes)throw new Error(`Mixed extension slots need ${bytes} bytes; the lent buffer holds ${buffer.size}`);
  if(this.slots===buffer&&this.slotArray===bytes/2)return;
  this.slots=buffer;this.slotArray=bytes/2;this.retained=false;this.heldReady=false;
  const heldBytes=this.heldBytesAt(this.ownership.capacity.fineTiles);
  if(this.held?.size!==heldBytes){this.held?.destroy();this.held=this.device.createBuffer({label:"Uniform mixed extension held values",size:heldBytes,usage:GPUBufferUsage.STORAGE});}
  const array=this.slotArray;
  this.markGroups=this.modes.map(mode=>this.device.createBindGroup({label:"Uniform mixed extension regions",layout:this.markLayout,entries:[
   {binding:0,resource:{buffer,offset:0,size:array}},{binding:1,resource:{buffer,offset:array,size:array}},
   {binding:2,resource:{buffer:this.stamp}},{binding:3,resource:{buffer:mode,size:16}},
   {binding:4,resource:{buffer:this.heatTiers!}},{binding:5,resource:{buffer:this.bodyRegions??this.noBodies}},{binding:6,resource:{buffer:this.work}},
  ]}));
 }
 /** The next swept encode rebuilds every region (pages were claimed or
  * recycled since the last: a key's slots are never inherited). */
 invalidateRetained():void{this.retained=false;}
 bind(f:Fields):readonly [UniformDetailGroup,UniformDetailGroup]{
  for(const [i,t] of [f.physical,f.phase,f.output].entries())if(t.format!==(i===1?"r32float":"rgba32float")||uniformDetailClass(t)!==(i===1?"cell":"face"))throw new Error("Mixed extension requires native canonical fields");
  if(f.physical===f.output||f.negative===f.outputNegative)throw new Error("Mixed extension outputs must be disjoint");
  const slots=this.slots,size=this.slotArray;this.retained=false;this.heldReady=false;
  if(!slots||size!==this.slotBufferBytesAt(this.ownership.capacity.fineTiles)/2)throw new Error("Mixed extension slots are not lent at this capacity (useSlots)");
  const companionSize=this.companionBytesAt(this.ownership.capacity.fineTiles)/2;
  if(f.companion&&!this.supportsPair)throw new Error("Paired extension is unsupported by this device or mode");
  if(f.companion&&f.companion.scratch.size<2*companionSize)throw new Error("Paired extension companion scratch is too small");
  if(f.snapshot&&!f.companion)throw new Error("Slot snapshot requires paired extension");
  return [0,1].map(parity=>uniformDetailGroup(this.device,{layout:f.snapshot?this.snapshotResources!:f.companion?this.pairedResources!:this.resources,entries:[
   ...(f.snapshot?[{binding:16,resource:f.snapshot}]:[]),
   ...[parity,parity^1].map((slot,binding)=>({binding,resource:{buffer:slots,offset:slot*size,size}})),
   {binding:4,resource:f.physical},{binding:5,resource:f.phase},
   {binding:6,resource:{buffer:f.negative}},{binding:7,resource:{buffer:f.params,size:16}},
   {binding:8,resource:f.output},{binding:9,resource:this.hierarchy.output},
   {binding:10,resource:this.hierarchy.outputOrigins},
   {binding:11,resource:this.hierarchy.input},{binding:12,resource:this.hierarchy.inputOrigins},
   {binding:13,resource:{buffer:f.outputNegative}},
   {binding:17,resource:{buffer:f.gate??this.alwaysLive,size:16}},
   ...(f.companion?[]:[{binding:18,resource:{buffer:this.held!}}]),
   ...(f.companion?[
    ...[parity,parity^1].map((slot,i)=>({binding:2+i,resource:{buffer:f.companion!.scratch,offset:slot*companionSize,size:companionSize}})),
    {binding:14,resource:f.companion.physical},{binding:15,resource:{buffer:f.companion.negative}},
   ]:[]),
  ]})) as unknown as readonly [UniformDetailGroup,UniformDetailGroup];
 }
 /** The active epoch selects the completed scalar payload after its sweeps. */
 companionBindings(buffer:GPUBuffer):readonly [GPUBufferBinding,GPUBufferBinding]{
  const size=this.companionBytesAt(this.ownership.capacity.fineTiles)/2;
  if(buffer.size<2*size)throw new Error("Paired extension companion scratch is too small");
  return [{buffer,offset:0,size},{buffer,offset:size,size}];
 }
 /** The mark passes ahead of a swept encode's seed: stamp, then classify. */
 private async initializeMarks():Promise<void>{
  const code=uniformMixedTopologyWGSL(this.ownership.capacity,0)+uniformStageListsWGSL+/* wgsl */`
@group(1) @binding(0) var<storage,read_write> slots0:array<vec2f>;
@group(1) @binding(1) var<storage,read_write> slots1:array<vec2f>;
@group(1) @binding(2) var<storage,read_write> stamp:array<u32>;
// x: 1 regional (this encode keeps clean regions), 0 every region dirty, 2
// held (the tail's: clean regions publish their held values);
// y: the array holding the final slots (the sweep count's parity).
@group(1) @binding(3) var<uniform> mode:vec4u;
${uniformRegionHeatWGSL(1,4,this.ownership.capacity.tiles/64)}
@group(1) @binding(5) var<storage,read> bodyRegions:array<u32>;
@group(1) @binding(6) var<storage,read_write> work:array<atomic<u32>>;
// stamp: per tile key (topology word, extension support bit), then per
// region bit 0 changed (the regional rebuild's) and bit 1 hot (not COLD, or
// a word moved: the tail's old payload around it), then per region its
// validity: bit 0 the held values hold its last publication from swept
// slots, bit 1 its slots are the last swept encode's final ones.
const MR_CHANGED:u32=2u*UM_TILES;
const MR_VALID:u32=MR_CHANGED+UM_PAGES;
var<workgroup> mrChanged:atomic<u32>;
fn mrBodyNear(region:u32)->bool{return region<arrayLength(&bodyRegions)&&(bodyRegions[region]&0xffu)<=${UNIFORM_EXTENSION_BODY_REACH}u;}
// A region changed when a tile word or support bit differs from the last
// swept encode's, or it is not COLD and beside a moved body (see
// UNIFORM_EXTENSION_BODY_REACH). The stamp then takes the current words.
// A fixed grid strides the held regions (umHeldPage). A region that holds
// no page is never stamped: no reader asks its bits (classifyRegions reads
// those of stored keys alone), and a claim makes it HOT (UniformRegionHeat)
// before its first stamp, so it changes and is hot whatever its old words.
@compute @workgroup_size(64) fn stampRegions(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 // A held encode always publishes (its quiet regions restore their held
 // values over whatever the frame's stages left in the output).
 if(group.x==0u&&lane==0u){let quiet=vec2f(select(0.0,1.0,mode.x==2u),0.0);slots0[arrayLength(&slots0)-1u]=quiet;slots1[arrayLength(&slots1)-1u]=quiet;atomicStore(&work[0],0u);atomicStore(&work[1],0u);}
 for(var visit=group.x;visit<umHeldPageCount();visit+=groups.x){
  let region=umHeldPage(visit);
  if(lane==0u){atomicStore(&mrChanged,0u);}
  workgroupBarrier();
  let block=region>>3u;let stored=umBlockStored(block);let tile=umPageTile(region,lane);
  var word=0xffffffffu;var support=0u;
  if(stored){word=umTopology[tile];support=umTileSupport(tile)&2u;}
  if(stamp[2u*tile]!=word||stamp[2u*tile+1u]!=support){atomicOr(&mrChanged,1u);}
  stamp[2u*tile]=word;stamp[2u*tile+1u]=support;
  workgroupBarrier();
  if(lane==0u){
   let cold=umRegionTier(region)==UM_TIER_COLD;let moved=atomicLoad(&mrChanged)!=0u;
   // A region whose held values are not its last publication rebuilds; a
   // held encode keeps only COLD regions away from bodies.
   var changed=moved||mode.x==0u||(stamp[MR_VALID+region]&1u)==0u;
   if(mode.x==1u){changed=changed||(!cold&&mrBodyNear(region));}
   if(mode.x==2u){changed=changed||!cold||mrBodyNear(region);}
   stamp[MR_CHANGED+region]=select(0u,1u,changed)|select(0u,2u,moved||!cold);
  }
  workgroupBarrier();
 }
}
// A region within one region (Chebyshev) of a changed one is dirty: its
// seeds or the slots its sweeps read (two sweeps reach 8h) changed. One
// within two is a halo. A full encode's regions are dirty, and those within
// one region of a hot one also publish the old payload (3), as do a held
// encode's dirty ones. A clean or halo region is held (+8) where its slots
// are not final (a held encode's all are not): it publishes its held values
// for unknown faces, and a held halo seeds (its dirty neighbours' sweeps
// read its seeds) instead of copying. The states follow each slot array's
// slots, then whether any region is dirty; the work words say whether any
// is dirty or publishes the old payload.
@compute @workgroup_size(64) fn classifyRegions(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 for(var region=gid.x;region<UM_PAGES;region+=64u*groups.x){
  var state=2u;let origin=vec3i(4u*umPageCoord(region));
  if(mode.x!=0u){
   state=0u;let top=select(2u,3u,mode.x==2u);
   for(var k=0u;k<125u&&state<top;k++){
    let d=vec3i(i32(k%5u),i32((k/5u)%5u),i32(k/25u))-vec3i(2);
    let t=umTileKey(origin+4*d);if(!umKeyStored(t)){continue;}
    let bits=stamp[MR_CHANGED+umPageOfTile(t)];let near=all(abs(d)<=vec3i(1));
    if((bits&1u)!=0u){state=max(state,select(1u,2u,near));}
    if(mode.x==2u&&near&&(bits&2u)!=0u){state=3u;}
   }
  }else{
   for(var k=0u;k<27u&&state<3u;k++){
    let d=vec3i(i32(k%3u),i32((k/3u)%3u),i32(k/9u))-vec3i(1);
    let t=umTileKey(origin+4*d);if(!umKeyStored(t)){continue;}
    if((stamp[MR_CHANGED+umPageOfTile(t)]&2u)!=0u){state=3u;}
   }
  }
  let valid=stamp[MR_VALID+region];let held=state<2u&&(mode.x==2u||(valid&2u)==0u);
  stamp[MR_VALID+region]=select(select(valid,valid&1u,held),3u,state>=2u);
  let frozen=${UNIFORM_COLD_FREEZE&&this.coldFreeze}&&umRegionTier(region)==UM_TIER_COLD;
  let value=vec2f(f32(state+select(0u,8u,held)+select(0u,16u,frozen)),f32(mode.y));
  slots0[arrayLength(&slots0)-UM_PAGES-1u+region]=value;slots1[arrayLength(&slots1)-UM_PAGES-1u+region]=value;
  // Any dirty region: the head restricts and publishes (stampRegions cleared it).
  if(state>=2u){slots0[arrayLength(&slots0)-1u]=vec2f(1.0,0.0);slots1[arrayLength(&slots1)-1u]=vec2f(1.0,0.0);atomicStore(&work[0],1u);}
  if(state==3u){atomicStore(&work[1],1u);}
 }
}
`;
  const module=this.device.createShaderModule({label:"Uniform mixed extension regions",code});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.markLayout]});
  const [stamp,classify]=await Promise.all(["stampRegions","classifyRegions"].map(entryPoint=>this.device.createComputePipelineAsync({layout,compute:{module,entryPoint}})));
  this.markPipelines={stamp:stamp!,classify:classify!};
 }
 /** Marks every region's state for this swept encode (regional: keep the
  * clean ones) and stamps the topology it runs on. */
 private encodeMarks(pass:GPUComputePassEncoder,mark:"full"|"regional"|"held",parity:number):void{
  if(!this.markPipelines||!this.markGroups)throw new Error("Mixed extension regions are not initialized");
  const regions=this.ownership.capacity.tiles/64;
  pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.markGroups[mark==="full"?0:mark==="regional"?1+parity:3+parity]!);
  pass.setPipeline(this.markPipelines.stamp);pass.dispatchWorkgroups(Math.min(UNIFORM_LIVE_GROUP_GROUPS,regions));
  pass.setPipeline(this.markPipelines.classify);pass.dispatchWorkgroups(Math.min(65535,ceilDiv(regions,64)));
 }
 destroy():void{this.held?.destroy();this.stamp.destroy();this.work.destroy();this.noBodies.destroy();for(const b of this.modes)b.destroy();this.alwaysLive.destroy();}
 /** The hierarchy's held blocks (its work words' list, umHeldBlock): the
  * hierarchy binds no ownership lists. */
 private async initializeBlocks():Promise<void>{
  const code=uniformMixedTopologyWGSL(this.ownership.capacity,0)+uniformStageListsWGSL+/* wgsl */`
@group(1) @binding(0) var<storage,read_write> work:array<u32>;
@compute @workgroup_size(64) fn blocksList(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 let held=umHeldBlockCount();if(gid.x==0u){work[2]=held;}
 for(var r=gid.x;r<held;r+=64u*groups.x){work[${UNIFORM_EXTRAPOLATION_WORK_WORDS}u+r]=umHeldBlock(r);}
}`;
  const module=this.device.createShaderModule({label:"Uniform mixed extension held blocks",code});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.blocksLayout]});
  this.blocksPipeline=await this.device.createComputePipelineAsync({label:"Uniform mixed extension held blocks",layout,compute:{module,entryPoint:"blocksList"}});
 }
 /** List the held blocks, then the hierarchy over them; group 0 is left rebound. */
 private encodeHierarchy(pass:GPUComputePassEncoder,replay:boolean,gate:0|1|2|3,replayGate:0|1|2|3=0):void{
  if(!this.blocksPipeline)throw new Error("Mixed extension held blocks are not initialized");
  pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.blocksGroup);
  pass.setPipeline(this.blocksPipeline);pass.dispatchWorkgroups(this.blockGroups);
  this.hierarchy.encode(pass,replay,gate,replayGate);
 }
 async initialize():Promise<void>{
  const marks=this.regularBulk?undefined:this.initializeMarks(),blocks=this.initializeBlocks();
  const source=uniformMixedCountedEntriesWGSL(uniformMixedTopologyWGSL(this.ownership.capacity,0)+uniformStageListsWGSL+/* wgsl */`
@group(1) @binding(0) var<storage,read_write> stateIn:array<vec2f>;
@group(1) @binding(1) var<storage,read_write> stateOut:array<vec2f>;
@group(1) @binding(4) var physical:texture_3d<f32>;
@group(1) @binding(5) var phase:texture_3d<f32>;
@group(1) @binding(6) var<storage,read> negative:array<f32>;
@group(1) @binding(7) var<uniform> h:vec4f;
@group(1) @binding(8) var output:texture_storage_3d<rgba32float,write>;
@group(1) @binding(9) var coarse:texture_3d<f32>;
@group(1) @binding(10) var origins:texture_3d<u32>;
@group(1) @binding(11) var coarseOut:texture_storage_3d<rgba32float,write>;
@group(1) @binding(12) var originsOut:texture_storage_3d<rgba32uint,write>;
@group(1) @binding(13) var<storage,read_write> boundary:array<f32>;
@group(1) @binding(17) var<uniform> ueLiveGate:vec4u;
// The frame's live sample count is nonzero (a uniform: barrier-safe exits).
fn ueParticlesLive()->bool{return ueLiveGate.x!=0u;}
${uniformMixedFaceAddressWGSL}
const UM_INF=1e20;
// A closed positive domain wall (the cell across the face is SOLID) is the
// mirror of a negative wall plane: seeded every pass, never swept or
// restricted, published as its physical value. The negative planes have no
// coarse +face, so the pyramid leaves the 4h cell behind a closed wall
// unknown on the wall's axis too: no far lookup takes a value there.
fn umClosedWall(face:UMFace)->bool{
 return face.sign>0&&umAcrossKind(face.anchor,face.axis,1)==UM_SOLID;
}
// One (value, distance) slot per canonical patch and component. The sections
// the lattice fixes come first. Width-4 patches (anchor/4 is their tile) sit
// in a component-major n/4 layer, so neighbouring 4h owners are neighbouring
// words. Negative domain-wall patches (anchor[axis] = -1) have planes of
// their own. Their addresses need the patch width, never a topology load.
${uniformMixedExtensionSlotWGSL}
// The sweep tile jobs stage the words of the 3x3x3 tiles around theirs
// (ueStage), which hold every anchor their faces read (ueFaceLive): those
// jobs take a word from workgroup memory instead of a topology load.
// The tile above a 4h tile may lie outside the 3x3x3: its word is loaded.
// ueMixedJobs: the launch also holds jobs that stage nothing (sweepSeams'
// regular 4h jobs); ueStagedJob then says whether this invocation's job did.
override ueStagedTiles:bool=false;
override ueMixedJobs:bool=false;
var<private> ueStagedJob:bool=false;
// The mixed launch packs its seam 4h tiles UE_PACK a job (ueSweepCoarsePack):
// pack slot s stages its 3x3x3 at 27 s. There a lane's staged tile is its
// pick (xyz: the tile; w: 27 s), set by the job it is serving (a one-tile
// job: ueStage, w = 0); every other launch reads ueJob and slot 0.
const UE_PACK=${UE_PACK_TILES}u;
// Fine-only sweeps use one neighborhood; packed coarse sweeps need two.
override ueStagePack:u32=${UE_PACK_TILES}u;
var<workgroup> ueWords:array<u32,27u*ueStagePack>;
// xyz: the job tile; w: its index, or UM_TILES without a supported owner
// (a sweep's: without an open one).
var<workgroup> ueJob:vec4u;
var<private> uePick:vec4u;
var<private> ueRecipe:vec2u;
var<workgroup> ueJobRecipe:vec2u;
fn ueJobTile()->vec3i{if(ueMixedJobs){return vec3i(uePick.xyz);}return vec3i(ueJob.xyz);}
fn ueJobBase()->u32{if(ueMixedJobs){return uePick.w;}return 0u;}
fn ueTileWord(t:vec3u)->u32{
 if(ueStagedTiles&&(!ueMixedJobs||ueStagedJob)){let r=vec3u(vec3i(t)-ueJobTile()+vec3i(1));if(all(r<=vec3u(2u))){return ueWords[ueJobBase()+r.x+3u*(r.y+3u*r.z)];}}
 return umTopology[umTileAt(t)];
}



// The shared sampler source needs this symbol; no entry of this module samples.
fn umLoadMixedFace(p:vec3i,axis:u32)->f32{return stateIn[umSlot(p,axis,1u)].x;}
// A slot is written only by the owner holding its anchor cell (a low face's:
// the cell above it), wherever that cell is stored: the slot and its support
// word are that cell's own. Off the extension support no pass writes it: its
// seed would be distance INF (a source face needs a liquid owner, which the
// support dilates around), and a value beside an infinite distance is never
// consumed. A slot is read only as the patch of that width at that anchor,
// which its owner writes every pass of a dirty region (ueRegionState). A
// slot no pass wrote holds an earlier encode's words.
fn umSlotState(anchor:vec3i,axis:u32,width:u32)->vec2f{
 let tile=umTileAt(vec3u(max(anchor,umWallLow()))/4u);
 // The support word loads beside the slot's address (a unit slot's tile word).
 let state=stateIn[umSlot(anchor,axis,width)];
 return select(vec2f(0.0,UM_INF),state,(umTileSupport(tile)&2u)!=0u);
}
${uniformMixedVelocitySamplingSource()}
fn umSource(face:UMFace,owner:UMOwner)->bool{
 if(textureLoad(phase,vec3i(umOrigin(owner)),0).x>0.5){return true;}
 return face.neighbor.width!=0u&&textureLoad(phase,vec3i(umOrigin(face.neighbor)),0).x>0.5;
}
fn umPhysical(face:UMFace)->f32{
 if(umFaceLow(face.anchor,face.axis)){return negative[umLowFaceIndex(ueLowCell(face.anchor,face.axis),face.axis)];}
 return textureLoad(physical,face.anchor,0)[face.axis];
}
struct UMNeighbor {value:f32,distance:f32,spacing:f32}
fn umNeighbor(point:vec3f,center:vec3f,component:u32,step:u32,width:u32)->UMNeighbor{
 // The request names the patch anchored below its plane: one with no
 // stored cell at its anchor, nor above a low face's, reaches nothing.
 var offset=vec3f(0.5*f32(width));offset[component]=1.0;
 let anchor=vec3i(round(point-offset));
 if(!ueRequestInside(anchor,component)){return UMNeighbor(0,UM_INF,1);}
 // The tile above the plane, or the anchor's where nothing is stored above.
 var upper=vec3i(point);if(!umCellPresent(upper)){upper=anchor;}
 let tile=umTileAt(vec3u(upper)/4u);
 // Both direct cases read the width-w patch slot below the plane (w is 1 on
 // the all-h stencil path). Its anchor needs no topology, so the support and
 // width words issue beside the slot's address instead of behind its value.
 // The first case below always reads a unit patch (every cell of an all-h
 // stencil, and so the one below the plane, is h); the second reads the
 // width-w patch. Both issue here, before the topology words resolve.
 let unit=umSlotState(anchor,component,1u);
 var direct=unit;if(width!=1u){direct=umSlotState(anchor,component,width);}
 // A unit request whose plane is interior to a unit tile has unit cells on
 // both sides: the case below with lowWidth = highWidth = 1, one load.
 let interior=width==1u&&(u32(round(point[component]))&3u)!=0u&&umTileWidth(tile)==1u;
 if(umRegularFine||interior||umTileMaximumWidth(tile)==1u){return UMNeighbor(unit.x,unit.y,h[step]);}
 // The point is a width-w lattice patch centre (the requesting face has
 // width w). When the owners on both sides of its plane are no finer than w
 // and one of them has width w (or the plane is a domain wall and the inner
 // owner has width w), the lower owner's positive face has patches of width
 // min(low, high) = w, one of them centred here: the search below selects
 // exactly it.
 var below=vec3i(floor(point));let plane=i32(round(point[component]));below[component]=plane-1;
 var above=below;above[component]=plane;
 let wall=umWallLow();
 let lowWidth=select(0u,umTileWidth(umTileAt(vec3u(max(below,wall))/4u)),umCellPresent(below));
 let highWidth=select(0u,umTileWidth(umTileAt(vec3u(max(above,wall))/4u)),umCellPresent(above));
 if(select((lowWidth==width&&(highWidth==0u||highWidth>=width))||(highWidth==width&&lowWidth>width),highWidth==width,lowWidth==0u)){
  // The point is center ± width along step only: the spacing is exactly
  // width*h, as on the fast path above. sqrt(dot) can land an ulp above it,
  // and the root cutoff below sits exactly on whole-cell distances, so which
  // path a face took (the +side tile decides) would decide its reach.
  return UMNeighbor(direct.x,direct.y,abs(point[step]-center[step])*h[step]);
 }
 let site=umVelocitySite(point,component);
 // Candidate real faces; choose geometrically, never read an unowned fine
 // texel. A requested plane inside a coarser cell has two incident faces. A
 // wider patch's centre on finer owners sits on fine cell boundaries in both
 // tangential axes: every tied cell's face is a candidate. Ties are resolved
 // by value, never by order: floor() (and a strict first-wins compare) takes
 // the +side (-side) cell of a tie, and a mirror maps it to the other side.
 // Read only where the plane is interior to this owner, which then holds
 // the cell above it.
 let owner=umOwnerAt(vec3i(floor(point)));
 let u=(component+1u)%3u;let v=(component+2u)%3u;
 var faces:array<UMFace,4>;var n=0u;
 if(site.interior){
  for(var side=0u;side<2u;side++){
   let sign=select(-1,1,side==1u);let first=umFace(owner,component,sign,0u);
   let local=clamp(point-vec3f(umOrigin(owner)),vec3f(0),vec3f(f32(owner.width)-1e-4));
   faces[n]=umFace(owner,component,sign,u32(local[u])/first.width+(owner.width/first.width)*(u32(local[v])/first.width));n++;
  }
 }else if(site.width<width){
  for(var k=0u;k<4u;k++){
   var tied=point;tied[u]+=select(-0.5,0.5,(k&1u)!=0u);tied[v]+=select(-0.5,0.5,(k&2u)!=0u);
   let s=umVelocitySite(tied,component);if(!s.interior){faces[n]=s.face;n++;}
  }
 }else{faces[0]=site.face;n=1u;}
 // Nearest faces, then the least slot distance among them, then the mean
 // value of the faces holding it.
 var spatials:array<f32,4>;var slots:array<vec2f,4>;var nearest=UM_INF;
 for(var i=0u;i<n;i++){
  spatials[i]=UM_INF;let face=faces[i];if(face.width==0u){continue;}
  let location=umFaceCenter(face);let delta=(location-center)*h.xyz;
  if(abs(location[step]-center[step])<1e-5){continue;}
  spatials[i]=dot(delta,delta);slots[i]=umSlotState(face.anchor,component,face.width);nearest=min(nearest,spatials[i]);
 }
 if(nearest>=0.5*UM_INF){return UMNeighbor(0,UM_INF,1);}
 var distance=UM_INF;
 for(var i=0u;i<n;i++){if(spatials[i]<=nearest*(1.0+1e-5)){distance=min(distance,slots[i].y);}}
 var sum=0.0;var count=0.0;
 for(var i=0u;i<n;i++){if(spatials[i]<=nearest*(1.0+1e-5)&&abs(slots[i].y-distance)<=1e-6*max(1.0,distance)){sum+=slots[i].x;count+=1.0;}}
 return UMNeighbor(select(0.0,sum/count,count>0.0),distance,sqrt(nearest));
}
// One patch from its six neighbours (low and high along each axis): the
// Godunov root over the nearer of each pair, cut off at two patch widths and
// at the old distance, then the mean value of the neighbours that reach it.
fn ueCombine(old:vec2f,width:f32,low:array<UMNeighbor,3>,high:array<UMNeighbor,3>)->vec2f{
 var minima:array<f32,3>;var spacing:array<f32,3>;
 for(var axis=0u;axis<3u;axis++){
  let lower=low[axis].distance<=high[axis].distance;
  minima[axis]=select(high[axis].distance,low[axis].distance,lower);
  spacing[axis]=select(high[axis].spacing,low[axis].spacing,lower);
 }
 var sorted=minima;var lengths=spacing;
 for(var i=0u;i<2u;i++){for(var j=i+1u;j<3u;j++){if(sorted[j]<sorted[i]){
  let d=sorted[i];sorted[i]=sorted[j];sorted[j]=d;let s=lengths[i];lengths[i]=lengths[j];lengths[j]=s;
 }}}
 if(sorted[0]>=0.5*UM_INF){return old;}var root=sorted[0]+lengths[0];
 for(var count=2u;count<=3u;count++){
  if(sorted[count-1u]>=0.5*UM_INF||root<=sorted[count-1u]){break;}
  var a=0.0;var b=0.0;var c=-1.0;
  for(var i=0u;i<count;i++){let w=1.0/(lengths[i]*lengths[i]);a+=w;b+=sorted[i]*w;c+=sorted[i]*sorted[i]*w;}
  root=(b+sqrt(max(b*b-a*c,0.0)))/a;
 }
 if(root>2.0*width*min(h.x,min(h.y,h.z))||root>old.y){return old;}
 var weighted=0.0;var total=0.0;let epsilon=max(root,2.0*width*min(h.x,min(h.y,h.z)))*1.1920929e-7;
 for(var axis=0u;axis<3u;axis++){
  if(minima[axis]>=root-epsilon){continue;}var value=0.0;var count=0.0;
  if(abs(low[axis].distance-minima[axis])<=epsilon){value+=low[axis].value;count+=1.0;}
  if(abs(high[axis].distance-minima[axis])<=epsilon){value+=high[axis].value;count+=1.0;}
  let weight=(root-minima[axis])/(spacing[axis]*spacing[axis]);weighted+=weight*value/max(count,1.0);total+=weight;
 }
 return vec2f(select(0.0,weighted/total,total>0.0),root);
}
// Callers evaluate only faces ueFaceLive admits: a face none of whose
// readable slots is finite would return old here.
fn umExtended(face:UMFace,owner:UMOwner)->vec2f{
 // A unit owner's face is its own positive patch (every caller's).
 var slot=0u;if(owner.width==1u){slot=ueOwnSlot(owner,face.axis);}else{slot=umSlot(face.anchor,face.axis,face.width);}
 let old=stateIn[slot];
 if(old.y==0.0||(umTileSupport(owner.tile)&2u)==0u||umClosedWall(face)){return old;}let center=umFaceCenter(face);let width=f32(face.width);
 var low:array<UMNeighbor,3>;var high:array<UMNeighbor,3>;
 for(var axis=0u;axis<3u;axis++){
  var delta=vec3f(0);delta[axis]=width;
  low[axis]=umNeighbor(center-delta,center,face.axis,axis,face.width);high[axis]=umNeighbor(center+delta,center,face.axis,axis,face.width);
 }
 return ueCombine(old,width,low,high);
}
// Frozen equal-width stencils certify one unit patch at every fine face.
fn umUnitExtensionFace(owner:UMOwner,axis:u32,sign:i32)->UMFace{
 let origin=vec3i(umOrigin(owner));var neighbor=origin;neighbor[axis]+=sign;
 var anchor=origin;if(sign<0){anchor[axis]-=1;}
 return UMFace(umOwnerAt(neighbor),anchor,1u,1u,axis,sign);
}
fn umSeedState(face:UMFace,owner:UMOwner)->vec2f{
 // WGSL select evaluates both operands: reject air before loading a payload.
 if(!umSource(face,owner)){return vec2f(0.0,UM_INF);}
 return vec2f(umPhysical(face),0.0);
}
// Seed and sweep outputs. The seed writes every slot and mask into BOTH
// slot arrays. A sweep result whose distance is 0 or INF is its seed value
// (distances only fall, a sweep root is positive, and an INF slot keeps its
// seed value), and so is every slot a sweep skips; the array a sweep writes
// holds that seed value already (from the seed, or from a pass that skipped
// it too). A sweep therefore stores only distances outside {0, INF} and
// never revisits a domain-wall patch (seed-only). Exact for any sweep count.
fn ueSeedStore(slot:u32,value:vec2f){stateOut[slot]=value;stateIn[slot]=value;}
fn ueSweepStore(slot:u32,value:vec2f){if(value.y<0.5*UM_INF&&value.y!=0.0){stateOut[slot]=value;}}
// The regional rebuild (UniformMixedExtension's mark passes): each slot
// array ends in a word per 16h region, 2 dirty (seeded and swept), 1 halo or
// 0 clean (its slots stay the last swept encode's final ones).
// A halo region is clean and within a region of a dirty one: its seed jobs
// copy their tile's final slots into the other array, so a dirty tile's
// sweeps read its neighbours' slots in either parity. Clean tiles do
// nothing. A full encode marks every region dirty.
// The word's y names the array holding the final slots (the sweep count's
// parity): array 0 after an even count, array 1 after an odd one.
fn ueRegionWord(tile:u32)->vec2f{return stateIn[arrayLength(&stateIn)-UM_PAGES-1u+umPageOfTile(tile)];}
// The head's restriction and publication (only theirs: the companion,
// snapshot and replay variants read other slots or write other fields) keep
// what the last swept encode left where nothing is dirty. Their last writer
// before a regional head is the tail's current restriction and publication
// of these same retained slots (UniformMixedExtension.encode).
override ueRegionalRetain:bool=false;
// The tail's old-payload restriction: only the regions publishing it (3);
// the rest keep their texels, the last restriction's.
override ueOldPayload:bool=false;
fn ueRestrictDirty(tile:u32)->bool{if(ueOldPayload){return ueRegionState(tile)>=3u;}return !ueRegionalRetain||ueRegionDirty(tile);}
// No region is dirty: the last publication stands.
fn ueQuiet()->bool{return ueRegionalRetain&&stateIn[arrayLength(&stateIn)-1u].x==0.0;}
var<workgroup> ueQuietWord:u32;
fn ueQuietUniform(lane:u32)->bool{if(lane==0u){ueQuietWord=select(0u,1u,ueQuiet());}return workgroupUniformLoad(&ueQuietWord)!=0u;}
fn ueRegionState(tile:u32)->u32{return u32(ueRegionWord(tile).x)&7u;}
// A held region (clean or halo, its slots not final: the marks' +8) publishes
// its held values for unknown faces; a held halo (9) seeds.
fn ueRegionHeld(tile:u32)->bool{return ${this.regularBulk?"false":"(u32(ueRegionWord(tile).x)&8u)!=0u"};}
fn ueRegionSeedHeld(tile:u32)->bool{return ${this.regularBulk?"false":"(u32(ueRegionWord(tile).x)&15u)==9u"};}
// A frozen region (UniformColdFreeze: COLD at the marks, +16) publishes
// nothing: its texels and low walls hold the frozen field's extension.
fn ueRegionFrozen(tile:u32)->bool{return ${this.regularBulk?"false":"(u32(ueRegionWord(tile).x)&16u)!=0u"};}
fn ueRegionDirty(tile:u32)->bool{return ueRegionState(tile)>=2u;}
// A seed launch binds array 1 as stateIn and array 0 as stateOut: the final
// slot into the other array.
fn ueHaloMove(slot:u32,odd:bool){if(odd){stateOut[slot]=stateIn[slot];}else{stateIn[slot]=stateOut[slot];}}
fn ueHaloCoarse(tile:u32){
 let odd=ueRegionWord(tile).y!=0.0;
 for(var axis=0u;axis<3u;axis++){ueHaloMove(UE_COARSE+axis*UE_TILES+tile,odd);}
 ueHaloMove(UE_FLAGS+2u*tile,odd);ueHaloMove(UE_FLAGS+2u*tile+1u,odd);
}
// Every slot a tile's seed writes, a lane each: an h tile's 240 unit slots
// (its own, and the 4h tiles' below it), a 4h tile's unit patches in the h
// tiles above it, its width-4 patches and mask words. Domain-wall slots are
// seeded into both arrays and never swept or tagged: they already agree.
fn ueHaloTile(tile:u32,lane:u32){
 let word=umTopology[tile];let odd=ueRegionWord(tile).y!=0.0;
 if((word&0x80000000u)!=0u){
  let base=UE_UNIT+15u*((word&0x3fffffffu)>>2u);
  for(var i=lane;i<240u;i+=64u){ueHaloMove(base+i,odd);}
 }else if(lane<48u){
  let axis=lane/16u;var above=vec3i(umTileCoord(tile));above[axis]+=1;
  if(umTilePresent(above)){let upper=umTopology[umTileAt(vec3u(above))];if((upper&0x80000000u)!=0u){ueHaloMove(UE_UNIT+15u*((upper&0x3fffffffu)>>2u)+192u+lane,odd);}}
 }
 if(lane==48u){ueHaloCoarse(tile);}
}
// Negative domain-wall patches are never swept: the seed writes them once.
// Returns whether its distance is finite.
fn umSeedWall(face:UMFace,owner:UMOwner)->bool{
 let source=umSource(face,owner);
 ueSeedStore(umSlot(face.anchor,face.axis,face.width),vec2f(umPhysical(face),select(UM_INF,0.0,source)));
 return source;
}
// Per tile, in the slot array a pass wrote: a 64-bit mask of the tile's
// cells holding the anchor of a finite slot (a low face's: the cell above
// it), as four exact 16-bit floats in two words (UE_FLAGS+2 tile). Only
// supported tiles' masks are current (ueMaskIn gates on support). Masks only
// grow: a pass's mask is its input mask or'ed with the anchors of the finite
// slots it wrote.
// The last float also carries the tile's open bit (65536): the seed left a
// positive patch that is not a source. A seed is a source (distance 0, which
// no sweep changes) or INF, so a tile without the bit stores nothing in any
// sweep and its mask stays its seed mask in both arrays: its sweep job ends
// at this word (ueOpen). The sweeps' tile work is the tiles holding a patch
// still to reach, not every tile of liquid.
fn ueBit(anchor:vec3i)->vec2u{
 let l=vec3u(max(anchor,umWallLow()))%4u;let b=l.x+4u*(l.y+4u*l.z);
 return select(vec2u(1u<<(b&31u),0u),vec2u(0u,1u<<(b&31u)),b>=32u);
}
fn ueMaskIn(tile:u32)->vec2u{
 if((umTileSupport(tile)&2u)==0u){return vec2u(0u);}
 let a=stateIn[UE_FLAGS+2u*tile];let b=stateIn[UE_FLAGS+2u*tile+1u];
 return vec2u(u32(a.x)|(u32(a.y)<<16u),u32(b.x)|((u32(b.y)&0xffffu)<<16u));
}
fn ueOpen(tile:u32)->bool{return stateIn[UE_FLAGS+2u*tile+1u].y>=65536.0;}
fn ueMaskOut(tile:u32,m:vec2u,seed:bool,open:bool){
 let a=vec2f(f32(m.x&0xffffu),f32(m.x>>16u));let b=vec2f(f32(m.y&0xffffu),f32((m.y>>16u)|select(0u,0x10000u,open)));
 stateOut[UE_FLAGS+2u*tile]=a;stateOut[UE_FLAGS+2u*tile+1u]=b;
 if(seed){stateIn[UE_FLAGS+2u*tile]=a;stateIn[UE_FLAGS+2u*tile+1u]=b;}
}
// Tile jobs stage the 3x3x3 tiles around theirs: width (0 outside the
// lattice) and input mask. A 4h tile job is live when any mask is nonzero;
// an h tile job when one of its cells is (ueCellLive). ueLive: bit 0 live;
// an h tile job also notes, per face direction k = 2 axis + side, a 4h face
// neighbour (bit 1 + k) and whether a request into it can read a finite
// slot (bit 7 + k).
var<workgroup> ueLive:atomic<u32>;
var<workgroup> ueNew:array<atomic<u32>,2>;
// Width is already encoded in ueWords. Outside entries are zero: callers
// either ask only whether they are fine, or have checked the domain bounds.
// Avoid a duplicate 27-word width array for each packed job.
fn ueStagedWidth(index:u32)->u32{return select(4u,1u,(ueWords[index]&0x80000000u)!=0u);}
var<workgroup> ueMasks:array<vec2u,27u*ueStagePack>;
var<workgroup> ueFineNeighborBase:array<u32,6>;
fn ueStaged(t:vec3i)->u32{let r=vec3u(t-ueJobTile()+vec3i(1));return ueJobBase()+r.x+3u*(r.y+3u*r.z);}
fn ueCellFinite(cell:vec3i)->bool{
 let t=cell/4;let m=ueMasks[ueStaged(t)];let l=vec3u(cell-4*t);let b=l.x+4u*(l.y+4u*l.z);
 return ((select(m.x,m.y,b>=32u)>>(b&31u))&1u)!=0u;
}
// The cells of a face neighbour's mask in its layer against the job's tile.
fn ueFacing(offset:vec3i)->vec2u{
 if(offset.x>0){return vec2u(0x11111111u);}if(offset.x<0){return vec2u(0x88888888u);}
 if(offset.y>0){return vec2u(0x000f000fu);}if(offset.y<0){return vec2u(0xf000f000u);}
 if(offset.z>0){return vec2u(0x0000ffffu,0u);}return vec2u(0u,0xffff0000u);
}
// A face direction: 2 axis + side, the request order.
fn ueDirection(offset:vec3i)->u32{
 if(offset.x!=0){return select(0u,1u,offset.x>0);}if(offset.y!=0){return select(2u,3u,offset.y>0);}return select(4u,5u,offset.z>0);
}
// The tile faces a cell lies against, by direction.
fn ueEdge(local:vec3u)->u32{
 var edge=0u;
 for(var axis=0u;axis<3u;axis++){if(local[axis]==0u){edge|=1u<<(2u*axis);}if(local[axis]==3u){edge|=2u<<(2u*axis);}}
 return edge;
}
// Whether a unit face anchored at a cell of the job's h tile can differ from
// its old slot (ueFaceLive, for the three components at once). A unit face's
// requests are anchored at the six cells around its anchor (a request past a
// domain wall reads nothing; one against the low wall of its own axis is
// anchored at the cell itself). Where the request's tile is unit, the one
// slot it reads is that cell's: the job tile's mask, spread one cell along
// each axis inside the tile, and the facing layer of each unit face
// neighbour moved onto the layer against it. Where it is a 4h face
// neighbour, every cell ueFaceLive tests lies in that tile, in the tile
// below it on another axis, or is the face's own anchor: when either tile's
// mask is set (ueStage notes it) the whole layer against the neighbour is
// live. Exact while every face neighbour is unit, a superset beside a 4h one
// (a face it admits beyond ueFaceLive's returns its old slot: no store).
fn ueCellLive(local:vec3u,edge:u32,flags:u32)->bool{
 if((edge&(flags>>1u)&(flags>>7u))!=0u){return true;}
 let m=ueMasks[13];
 var d=m|((m&vec2u(0x77777777u))<<vec2u(1u))|((m&vec2u(0xeeeeeeeeu))>>vec2u(1u))|((m&vec2u(0x0fff0fffu))<<vec2u(4u))|((m&vec2u(0xfff0fff0u))>>vec2u(4u))
  |vec2u((m.x<<16u)|(m.x>>16u)|(m.y<<16u),(m.y<<16u)|(m.x>>16u)|(m.y>>16u));
 if(ueStagedWidth(12)==1u){d|=(ueMasks[12]&vec2u(0x88888888u))>>vec2u(3u);}
 if(ueStagedWidth(14)==1u){d|=(ueMasks[14]&vec2u(0x11111111u))<<vec2u(3u);}
 if(ueStagedWidth(10)==1u){d|=(ueMasks[10]&vec2u(0xf000f000u))>>vec2u(12u);}
 if(ueStagedWidth(16)==1u){d|=(ueMasks[16]&vec2u(0x000f000fu))<<vec2u(12u);}
 if(ueStagedWidth(4)==1u){d.x|=ueMasks[4].y>>16u;}
 if(ueStagedWidth(22)==1u){d.y|=(ueMasks[22].x&0xffffu)<<16u;}
 let b=local.x+4u*(local.y+4u*local.z);
 return ((select(d.x,d.y,b>=32u)>>(b&31u))&1u)!=0u;
}
// Stages the neighbourhood of lane 0's owner's tile (every lane of a tile
// job holds an owner of that tile, or none). Workgroup-uniform result.
// fine: an h tile job, live when some ueCellLive of it can hold: its own
// mask is set, or the facing layer of a unit face neighbour, or a 4h face
// neighbour's mask, or the mask of the tile below one on another axis (an
// edge neighbour). A corner tile's mask reaches none of its faces.
fn ueStage(owner:UMOwner,lane:u32,seed:bool,fine:bool)->bool{
 if(lane==0u){
  // UM_TILES: no job; above it, a seed's halo tile (UM_TILES + 1 + tile).
  var index=UM_TILES;
  if(owner.width!=0u&&(umTileSupport(owner.tile)&2u)!=0u&&(seed||ueOpen(owner.tile))){
   let region=ueRegionState(owner.tile);
   if(region>=2u||(seed&&ueRegionSeedHeld(owner.tile))){index=owner.tile;}else if(seed&&region==1u){index=UM_TILES+1u+owner.tile;}
  }
  ueJob=vec4u(umTileCoord(owner.tile),index);
  if(!seed&&(ueMixedJobs||ueSweepKind==1u)&&index<UM_TILES){ueJobRecipe=umTileStencil(index);}
  atomicStore(&ueLive,0u);atomicStore(&ueNew[0],0u);atomicStore(&ueNew[1],0u);
 }
 let job=workgroupUniformLoad(&ueJob);
 if(ueMixedJobs){uePick=vec4u(job.xyz,0u);ueRecipe=ueJobRecipe;}
 if(ueSweepKind==1u){ueRecipe=ueJobRecipe;}
 if(job.w>=UM_TILES){if(job.w>UM_TILES){ueHaloTile(job.w-UM_TILES-1u,lane);}return false;}
 if(seed){return true;}
 if(lane<27u){
  let t=vec3i(job.xyz)+vec3i(umCorner(lane,3u))-vec3i(1);var width=0u;var mask=vec2u(0u);var word=0u;
  if(umTilePresent(t)){let tile=umTileAt(vec3u(t));word=umTopology[tile];width=select(4u,1u,(word&0x80000000u)!=0u);mask=ueMaskIn(tile);}
  ueWords[lane]=word;ueMasks[lane]=mask;
  let offset=vec3i(umCorner(lane,3u))-vec3i(1);let far=abs(offset.x)+abs(offset.y)+abs(offset.z);
  if(ueCacheFineRequests&&fine&&far==1){
   var base=0xffffffffu;
   if(width==1u){base=UE_NONE;if((umTileSupport(umTileAt(vec3u(t)))&2u)!=0u){base=UE_UNIT+15u*((word&0x3fffffffu)>>2u);}}
   ueFineNeighborBase[ueDirection(offset)]=base;
  }
  let filled=any(mask!=vec2u(0u));var live=0u;
  if(!fine||far==0){live=select(0u,1u,filled);}
  else if(far==1){
   let k=ueDirection(offset);
   if(width==4u){live=(2u<<k)|select(0u,128u<<k,filled);}
   else{live=select(0u,1u,any((mask&ueFacing(offset))!=vec2u(0u)));}
  }else if(far==2&&filled){
   for(var axis=0u;axis<3u;axis++){if(offset[axis]<0){var d=offset;d[axis]=0;live|=128u<<ueDirection(d);}}
  }
  if(live!=0u){atomicOr(&ueLive,live);}
 }
 let live=workgroupUniformLoad(&ueLive);return (live&1u)!=0u||((live>>1u)&(live>>7u)&63u)!=0u;
}
// Whether umExtended(face) can differ from its old slot: some slot it may
// read is finite (otherwise it returns old).
// Every slot a request reads is anchored in the cells tested here:
// - old: the face's anchor.
// - request anchor A = the face's anchor +- width e_step (the patch below the
//   plane of the point centre +- width e_step), where ueRequestInside admits
//   it; q its cell (the cell above a low face), Tq the tile of q (in the
//   job's 3x3x3: q is within one width of the face's anchor, whose tile is
//   the job's or, for a 4h owner's patch, whose +axis neighbour is staged).
// - width 1, Tq unit: the unit slot at q only (lowWidth is 1: the direct
//   case; a domain-wall patch at q[axis] = -1 is attributed to q).
// - width 1, Tq 4h: the unit slot at q, and the site owner O (the 4h owner
//   of Tq) patches covering q: its +axis patch in Tq's top layer at q's
//   columns (unit patches) or at O's origin columns (a width-4 patch), and
//   when A's plane is strictly inside O (plane % 4 != 0) its -axis patch in
//   the layer below O at the same columns (a domain wall: layer 0 of Tq);
//   at plane 0 the site is O's -axis domain-wall patch (layer 0 of Tq).
// - width 4: A is a 4h-aligned patch; every direct, site and tied
//   candidate lies in Tq (never interior: its plane is a multiple of four).
fn ueFaceLive(anchor:vec3i,axis:u32,width:u32)->bool{
 var live=ueCellFinite(max(anchor,umWallLow()));
 for(var n=0u;n<6u;n++){
  var request=anchor;request[n/2u]+=select(-i32(width),i32(width),(n&1u)!=0u);
  if(!ueRequestInside(request,axis)){continue;}
  live=live||ueRequestAnchorLive(request,axis,width);
 }
 return live;
}
// One request (ueRequestInside admits its anchor): whether a slot it may
// read is finite.
fn ueRequestAnchorLive(anchor:vec3i,axis:u32,width:u32)->bool{
 let q=max(anchor,umWallLow());let t=q/4;
 if(width!=1u){return any(ueMasks[ueStaged(t)]!=vec2u(0u));}
 if(ueStagedWidth(ueStaged(t))==1u){return ueCellFinite(q);}
 let o=4*t;var top=q;top[axis]=o[axis]+3;var corner=o;corner[axis]=o[axis]+3;
 var live=ueCellFinite(q)||ueCellFinite(top)||ueCellFinite(corner);
 let plane=anchor[axis]+1;
 // A low face (umFaceLow) has no tile below it.
 if(plane%4!=0||umFaceLow(anchor,axis)){
  var below=q;below[axis]=max(o[axis]-1,0);var belowCorner=o;belowCorner[axis]=below[axis];
  live=live||ueCellFinite(below)||ueCellFinite(belowCorner);
 }
 return live;
}
// The anchor bit of a stored (finite, nonzero) sweep result. Lanes gather
// their bits and or them into ueNew once.
fn ueNote(value:vec2f,anchor:vec3i)->vec2u{
 if(value.y<0.5*UM_INF&&value.y!=0.0){return ueBit(anchor);}return vec2u(0u);
}
fn ueGather(bits:vec2u){if(any(bits!=vec2u(0u))){atomicOr(&ueNew[0],bits.x);atomicOr(&ueNew[1],bits.y);}}
// The job's output mask: its input mask or the finite slots it wrote. A
// zero mask is already in the output array (the seed wrote it, and masks
// only grow).
fn ueFlush(){
 let m=ueMasks[13]|vec2u(atomicLoad(&ueNew[0]),atomicLoad(&ueNew[1]));
 if(any(m!=vec2u(0u))){ueMaskOut(ueJob.w,m,false,true);}
}
// Seam requests use compiled lattice recipes in the requesting lane. Their
// geometry is now cheap enough to avoid the shared request/answer queue,
// its per-face atomic reservation and two barriers (3844 bytes at pack 2).
// A positive patch by its anchor (no neighbour lookup: no caller reads it).
fn uePatch(anchor:vec3i,axis:u32,width:u32)->UMFace{return UMFace(UMOwner(),anchor,width,1u,axis,1);}
// umExtended's request n = 2 step + side of a patch centred at center.
fn ueRequestPoint(center:vec3f,width:u32,n:u32)->vec3f{
 var delta=vec3f(0);delta[n/2u]=f32(width);
 if((n&1u)==0u){return center-delta;}
 return center+delta;
}
${uniformCompiledExtensionNeighborWGSL}
// Immutable request geometry comes from the layout recipe; liveness still
// comes from this sweep's staged masks. Classify and consume in one loop,
// retaining the old per-axis combination and candidate reduction order.
override ueCacheFineRequests:bool=false;
var<private> ueFineRequestSlots:array<u32,6>;
// Fine neighbours share an address across components; only the 64-slot
// component offset changes. Resolve each of the six anchors once per cell.
fn uePrepareFineRequests(owner:UMOwner){
 let local=umCorner(owner.lane,4u);let own=ueOwnSlot(owner,0u);
 for(var n=0u;n<6u;n++){
  let axis=n/2u;let sign=select(-1,1,(n&1u)!=0u);let next=i32(local[axis])+sign;
  // The current tile is supported; only requests crossing its edge need
  // the neighbor base shared by all sixteen cells along that face.
  if(next>=0&&next<4){ueFineRequestSlots[n]=u32(i32(own)+sign*i32(1u<<(2u*axis)));continue;}
  let base=ueFineNeighborBase[n];ueFineRequestSlots[n]=base;
  if(base==0xffffffffu||base==UE_NONE){continue;}
  var across=local;across[axis]=select(3u,0u,sign>0);
  ueFineRequestSlots[n]=base+across.x+4u*(across.y+4u*across.z);
 }
}
fn ueExtend(face:UMFace,old:vec2f,need:u32)->vec2f{
 let center=umFaceCenter(face);let origin=4*ueJobTile();
 var low:array<UMNeighbor,3>;var high:array<UMNeighbor,3>;
 for(var n=0u;n<6u;n++){
  var anchor=face.anchor;anchor[n/2u]+=select(-i32(face.width),i32(face.width),(n&1u)!=0u);
  var answer=UMNeighbor(0,UM_INF,1);
  if(ueCacheFineRequests&&face.width==1u&&ueFineRequestSlots[n]!=0xffffffffu){
   let slot=ueFineRequestSlots[n];var value=vec2f(0.0,UM_INF);
   if(slot!=UE_NONE){value=stateIn[slot+64u*face.axis];}
   answer=UMNeighbor(value.x,value.y,h[n/2u]);
  }else if(ueRequestInside(anchor,face.axis)){
   let search=(need&(1u<<n))!=0u&&!ueTopologyDirectAt(ueRecipe,anchor-origin,face.axis,face.width);
   if(search){
    if(ueRequestAnchorLive(anchor,face.axis,face.width)){
     var point=vec3f(anchor)+vec3f(0.5*f32(face.width));point[face.axis]=f32(anchor[face.axis]+1);
     answer=ueCompiledNeighbor(point,center,face.axis,n/2u,face.width);
    }
   }else{
    let state=umSlotState(anchor,face.axis,face.width);
    answer=UMNeighbor(state.x,state.y,f32(face.width)*h[n/2u]);
   }
  }
  if((n&1u)==0u){low[n/2u]=answer;}else{high[n/2u]=answer;}
 }
 return ueCombine(old,f32(face.width),low,high);
}
// The regular 4h launch takes the ownership's packed regular 4h list (one
// lane per owner, no seam tiles to filter), including a tier small enough to
// be fused elsewhere: the owners are the tier list's regular 4h owners.
override ueRegularCoarseList:bool=false;
fn ueOwner(gid:vec3u)->UMOwner{
 if(ueRegularCoarseList){return umRegularCoarseOwner(gid.x+umDispatchX*64u*gid.y);}
 return umOwner(gid);
}
// Off the extension support seed and sweeps write nothing: every reader of
// such a slot goes through umSlotState (or, in restrictBand, skips the tile).
//
// Launches. A launch of a few jobs costs the latency of its longest per-lane
// chain whatever its job count, so each seam tier with a launch of its own
// added that latency to the pass however few tiles it held. Every pass here
// (seed, a sweep, publish) is a Jacobi pass over disjoint outputs, so the
// seam jobs of a pass share one launch and their chains overlap: a job per
// seam h tile, then per seam 4h tile. A sweep's launch first holds the
// regular 4h owners, 64 a job, whose chain is as long: the first h tile then
// adds no launch to a sweep. The regular h kernels stay apart, and the list
// kernel while no h tile can exist: a kernel holding every job kind runs its
// bulk slower (the regular h sweep 1.8x, the list sweep 1.3x).
// ueLaunch: the job counts (seam h tiles, seam 4h tiles, regular 4h packs), loaded
// once per workgroup; a job's kind is workgroup-uniform (its index against
// those counts).
var<workgroup> ueLaunch:vec3u;
fn ueLaunchCounts()->vec3u{
 let header=7u*UM_TILES+16u;
 return vec3u(umSupport[header],umSupport[header+1u],(umSupport[8u*UM_TILES+20u]+63u)/64u);
}
// Seam h tile job: one lane per owner.
fn ueSeamFineJob(job:u32,lane:u32)->UMOwner{
 let tile=umSupport[7u*UM_TILES+20u+job];return UMOwner(tile,lane,1u,(umTopology[tile]&0x3fffffffu)+lane);
}
// Seam 4h tile job (listed after the seam h tiles): the tile's one owner, or
// none off the extension support.
fn ueSeamCoarseJob(job:u32)->UMOwner{
 let header=7u*UM_TILES+16u;let fine=umSupport[header];let tile=umSupport[header+4u+fine+job];
 if((umTileSupport(tile)&2u)==0u){return UMOwner();}
 return UMOwner(tile,0u,4u,umTopology[tile]&0x3fffffffu);
}
// The seed of one owner; returns the anchors of its finite slots (xy) and
// whether it left a patch that is not a source (z: the tile's open bit).
fn ueSeedOwner(owner:UMOwner)->vec3u{
 let origin=umOrigin(owner);let low=umLowOpen(vec3i(origin));var bits=vec2u(0u);var open=0u;
 // A unit owner's faces are the unit patches whatever its neighbours.
 if(umRegularFine||owner.width==1u){
  // A liquid owner certifies all six incident patches as sources before
  // neighbor ownership is resolved. Air owners still resolve their donors.
  let source=textureLoad(phase,vec3i(origin),0).x>0.5;
  for(var axis=0u;axis<3u;axis++){
   if(low[axis]){var face=uePatch(vec3i(origin),axis,1u);face.anchor[axis]-=1;face.sign=-1;if(!source){face=umUnitExtensionFace(owner,axis,-1);}if(umSeedWall(face,owner)){bits|=ueBit(face.anchor);}}
   var face=uePatch(vec3i(origin),axis,1u);if(!source){face=umUnitExtensionFace(owner,axis,1);}
   let value=umSeedState(face,owner);ueSeedStore(ueOwnSlot(owner,axis),value);if(value.y<0.5*UM_INF){bits|=ueBit(face.anchor);}else{open=1u;}
  }
  return vec3u(bits,open);
 }
 // Every patch owns its component slot: no anchor packing across axes.
 for(var axis=0u;axis<3u;axis++){
  if(low[axis]){let face=umFace(owner,axis,-1,0u);if(umSeedWall(face,owner)){bits|=ueBit(face.anchor);}}
  let first=umFace(owner,axis,1,0u);
  for(var part=0u;part<first.count;part++){
   let face=umFace(owner,axis,1,part);
   let value=umSeedState(face,owner);ueSeedStore(umSlot(face.anchor,axis,face.width),value);if(value.y<0.5*UM_INF){bits|=ueBit(face.anchor);}else{open=1u;}
  }
 }
 return vec3u(bits,open);
}
@compute @workgroup_size(64) fn seedList(@builtin(global_invocation_id) gid:vec3u){
 let owner=ueOwner(gid);if(owner.width==0u||(umTileSupport(owner.tile)&2u)==0u){return;}
 // A regular 4h owner's slots are its width-4 patches and mask words.
 let region=ueRegionState(owner.tile);if(region<2u&&!ueRegionSeedHeld(owner.tile)){if(region==1u){ueHaloCoarse(owner.tile);}return;}
 let seeded=ueSeedOwner(owner);ueMaskOut(owner.tile,seeded.xy,true,seeded.z!=0u);
}
// A regular 4h owner's faces are single width-4 patches: every slot they
// read lies in its tile or a face neighbour (ueFaceLive, width 4).
fn ueSweepPack(owner:UMOwner){
 if(owner.width==0u||(umTileSupport(owner.tile)&2u)==0u||!ueOpen(owner.tile)||!ueRegionDirty(owner.tile)){return;}
 let t=vec3i(umTileCoord(owner.tile));let own=ueMaskIn(owner.tile);var live=any(own!=vec2u(0u));
 for(var n=0u;n<6u;n++){
  if(live){break;}
  var s=t;s[n/2u]+=select(-1,1,(n&1u)!=0u);
  if(umTilePresent(s)){live=any(ueMaskIn(umTileAt(vec3u(s)))!=vec2u(0u));}
 }
 if(!live){return;}
 var bits=own;
 for(var axis=0u;axis<3u;axis++){
  let first=umFace(owner,axis,1,0u);
  for(var part=0u;part<first.count;part++){
   let face=umFace(owner,axis,1,part);let value=umExtended(face,owner);
   ueSweepStore(umSlot(face.anchor,axis,face.width),value);if(value.y<0.5*UM_INF){bits|=ueBit(face.anchor);}
  }
 }
 if(any(bits!=vec2u(0u))){ueMaskOut(owner.tile,bits,false,true);}
}
@compute @workgroup_size(64) fn sweepList(@builtin(global_invocation_id) gid:vec3u){
 ueSweepPack(ueOwner(gid));
}
// An h tile job: every lane holds an owner of the one tile (or none).
fn ueSeedFine(owner:UMOwner,lane:u32){
 if(!ueStage(owner,lane,true,true)){return;}
 let bits=ueSeedOwner(owner);atomicOr(&ueNew[0],bits.x);atomicOr(&ueNew[1],bits.y);if(bits.z!=0u){atomicOr(&ueLive,1u);}
 workgroupBarrier();
 if(lane==0u){ueMaskOut(ueJob.w,vec2u(atomicLoad(&ueNew[0]),atomicLoad(&ueNew[1])),true,atomicLoad(&ueLive)!=0u);}
}
@compute @workgroup_size(64) fn seed(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 ueSeedFine(ueOwner(gid),lane);
}
// A regular h tile has no 4h tile in its stencil: each lane evaluates the
// faces of its owner (a unit owner's positive patch is anchored at its
// origin) when its cell is live, and every neighbour is a unit slot.
@compute @workgroup_size(64) fn sweep(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 let owner=ueOwner(gid);
 if(!ueStage(owner,lane,false,true)){return;}
 var bits=vec2u(0u);let local=umCorner(owner.lane,4u);let live=ueCellLive(local,ueEdge(local),atomicLoad(&ueLive));
 for(var axis=0u;axis<3u;axis++){
  if(!live){continue;}
  // Sources are immutable through every sweep. Reject before constructing
  // a face; extension uses only its patch geometry, never its neighbor owner.
  let slot=ueOwnSlot(owner,axis);if(stateIn[slot].y==0.0){continue;}
  let face=uePatch(vec3i(umOrigin(owner)),axis,1u);let value=umExtended(face,owner);
  ueSweepStore(ueOwnSlot(owner,axis),value);bits|=ueNote(value,face.anchor);
 }
 ueGather(bits);
 workgroupBarrier();
 if(lane==0u){ueFlush();}
}
// A seam h tile: each lane evaluates its three unit patches. No values
// remain live across a queue barrier; each patch is planned and consumed.
fn ueSweepSeamFine(owner:UMOwner,lane:u32){
 if(!ueStage(owner,lane,false,true)){return;}
 let origin=vec3i(umOrigin(owner));var bits=vec2u(0u);
 let flags=atomicLoad(&ueLive);let local=umCorner(owner.lane,4u);let edge=ueEdge(local);let live=ueCellLive(local,edge,flags);
 if(ueCacheFineRequests&&live){uePrepareFineRequests(owner);}
 for(var axis=0u;axis<3u;axis++){
  if(!live){continue;}
  let face=uePatch(origin,axis,1u);let old=stateIn[ueOwnSlot(owner,axis)];
  if(old.y==0.0||umClosedWall(face)){continue;}
  let value=ueExtend(face,old,edge&(flags>>1u)&~(1u<<(2u*axis)));
  ueSweepStore(ueOwnSlot(owner,axis),value);bits|=ueNote(value,origin);
 }
 ueGather(bits);
 workgroupBarrier();
 if(lane==0u){ueFlush();}
}
// Coarse seam owners: one job per tile, one lane per patch and component.
// A 4h owner beside h tiles has sixteen patches per face; evaluating them
// serially in one lane made the coarse seam tier a latency chain.
// 4h seam tiles hold one owner with at most sixteen patches per positive
// face. One lane per (axis, patch); each writes its own slot directly.
fn umCoarseCell(local:vec3u)->u32{return local.x+4u*(local.y+4u*local.z);}
fn ueSeedCoarse(owner:UMOwner,lane:u32){
 if(!ueStage(owner,lane,true,false)){return;}
 var bits=vec2u(0u);
 if(lane<48u){
  let axis=lane/16u;let face=umFace(owner,axis,1,lane%16u);
  if(face.width!=0u){let value=umSeedState(face,owner);ueSeedStore(umSlot(face.anchor,axis,face.width),value);if(value.y<0.5*UM_INF){bits=ueBit(face.anchor);}else{atomicOr(&ueLive,1u);}}
 }else if(lane<51u){
  let axis=lane-48u;if(umLowOpen(vec3i(umOrigin(owner)))[axis]){let face=umFace(owner,axis,-1,0u);if(umSeedWall(face,owner)){bits=ueBit(face.anchor);}}
 }
 atomicOr(&ueNew[0],bits.x);atomicOr(&ueNew[1],bits.y);
 workgroupBarrier();
 if(lane==0u){ueMaskOut(ueJob.w,vec2u(atomicLoad(&ueNew[0]),atomicLoad(&ueNew[1])),true,atomicLoad(&ueLive)!=0u);}
}
// The positive patch umFace would return for a 4h seam owner, from the
// staged +axis neighbour width (umFace: min of the owner's and the
// neighbour's, the owner's outside the lattice): item = 16 axis + part.
// Width 0: no such part.
fn ueCoarsePatch(origin:vec3i,item:u32)->UMFace{
 let axis=item/16u;let part=item%16u;var probe=origin;probe[axis]+=4;
 var width=4u;if(umCellPresent(probe)){width=min(4u,ueStagedWidth(ueStaged(probe/4)));}
 let side=4u/width;if(part>=side*side){return UMFace();}
 var anchor=probe;anchor[(axis+1u)%3u]+=i32((part%side)*width);anchor[(axis+2u)%3u]+=i32((part/side)*width);anchor[axis]-=1;
 return uePatch(anchor,axis,width);
}
// A pack of seam 4h tiles (list entries first .. first + UE_PACK, below
// count). A seam 4h tile fills 3 to 48 of a job's 64 lanes and every job
// pays its staging, barriers and flush whatever it holds: a pack shares
// them. Slot s holds one tile: its header (uePackJobs, as ueJob), its staged
// 3x3x3 at 27 s, its live bit (some staged mask nonzero) and its new anchor
// bits. The pack's patches are numbered in (slot, component, part) order
// over the live slots, a lane each: lane l evaluates patch l (and l + 64
// where the pack holds more). Each slot's mask is flushed by one lane.
// A patch's item is 48 slot + 16 axis + part (ueCoarsePatch's item of its
// slot).
const UE_PACK_ROUNDS=${Math.ceil(48*UE_PACK_TILES/64)}u;
const UE_PACK_NONE=0xffffffffu;
var<workgroup> uePackJobs:array<vec4u,UE_PACK>;
var<workgroup> uePackRecipes:array<vec2u,UE_PACK>;
var<workgroup> uePackLive:array<atomic<u32>,UE_PACK>;
var<workgroup> uePackNew:array<atomic<u32>,${2*UE_PACK_TILES}>;
// The patches of a slot's component: sixteen under an h tile (the staged
// +axis neighbour, ueCoarsePatch's width), one otherwise; none in a slot
// without a live job.
fn uePackParts(slot:u32,axis:u32)->u32{
 if(uePackJobs[slot].w>=UM_TILES||atomicLoad(&uePackLive[slot])==0u){return 0u;}
 return select(1u,16u,ueStagedWidth(27u*slot+13u+select(select(9u,3u,axis==1u),1u,axis==0u))==1u);
}
fn uePackItem(at:u32)->u32{
 var base=0u;
 for(var k=0u;k<3u*UE_PACK;k++){
  let parts=uePackParts(k/3u,k%3u);
  if(at<base+parts){return 48u*(k/3u)+16u*(k%3u)+(at-base);}
  base+=parts;
 }
 return UE_PACK_NONE;
}
fn ueSweepCoarsePack(first:u32,count:u32,lane:u32){
 if(lane<UE_PACK){
  var job=vec4u(0u,0u,0u,UM_TILES);
  if(first+lane<count){
   let owner=ueSeamCoarseJob(first+lane);
   if(owner.width!=0u&&ueOpen(owner.tile)&&ueRegionDirty(owner.tile)){job=vec4u(umTileCoord(owner.tile),owner.tile);}
  }
  if(job.w<UM_TILES){uePackRecipes[lane]=umTileStencil(job.w);}
  uePackJobs[lane]=job;atomicStore(&uePackLive[lane],0u);atomicStore(&uePackNew[2u*lane],0u);atomicStore(&uePackNew[2u*lane+1u],0u);
 }
 workgroupBarrier();
 for(var i=lane;i<27u*UE_PACK;i+=64u){
  let slot=i/27u;let job=uePackJobs[slot];var mask=vec2u(0u);var word=0u;
  if(job.w<UM_TILES){
   let t=vec3i(job.xyz)+vec3i(umCorner(i%27u,3u))-vec3i(1);
   if(umTilePresent(t)){let tile=umTileAt(vec3u(t));word=umTopology[tile];mask=ueMaskIn(tile);}
   if(any(mask!=vec2u(0u))){atomicOr(&uePackLive[slot],1u);}
  }
  ueWords[i]=word;ueMasks[i]=mask;
 }
 workgroupBarrier();
 for(var round=0u;round<UE_PACK_ROUNDS;round++){
  let item=uePackItem(64u*round+lane);if(item==UE_PACK_NONE){continue;}
  let slot=item/48u;let job=uePackJobs[slot];uePick=vec4u(job.xyz,27u*slot);ueRecipe=uePackRecipes[slot];
  let face=ueCoarsePatch(vec3i(job.xyz)*4,item%48u);
  if(face.width==0u||!ueFaceLive(face.anchor,face.axis,face.width)){continue;}
  let old=stateIn[umSlot(face.anchor,face.axis,face.width)];
  if(old.y==0.0||umClosedWall(face)){continue;}
  let value=ueExtend(face,old,63u);
  ueSweepStore(umSlot(face.anchor,face.axis,face.width),value);let bits=ueNote(value,face.anchor);
  if(any(bits!=vec2u(0u))){atomicOr(&uePackNew[2u*slot],bits.x);atomicOr(&uePackNew[2u*slot+1u],bits.y);}
 }
 workgroupBarrier();
 if(lane<UE_PACK){
  let job=uePackJobs[lane];
  if(job.w<UM_TILES&&atomicLoad(&uePackLive[lane])!=0u){
   let m=ueMasks[27u*lane+13u]|vec2u(atomicLoad(&uePackNew[2u*lane]),atomicLoad(&uePackNew[2u*lane+1u]));
   if(any(m!=vec2u(0u))){ueMaskOut(job.w,m,false,true);}
  }
 }
}
// Fixed direct grid-stride launches over the GPU job counts: any grid is
// complete. The barrier after a job keeps the next job's staging apart.
@compute @workgroup_size(64) fn seedSeams(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){ueLaunch=ueLaunchCounts();}
 let counts=workgroupUniformLoad(&ueLaunch);
 for(var job=group.x;job<counts.x+counts.y;job+=groups.x){
  if(job<counts.x){ueSeedFine(ueSeamFineJob(job,lane),lane);}
  else{ueSeedCoarse(ueSeamCoarseJob(job-counts.x),lane);}
  workgroupBarrier();
 }
}
// 1: fine seams; 2: all coarse jobs. 0 retains the merged reference used by
// the differential sweep test; production always specializes to 1 or 2.
override ueSweepKind:u32=0u;
@compute @workgroup_size(64) fn sweepSeams(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){ueLaunch=ueLaunchCounts();}
 let counts=workgroupUniformLoad(&ueLaunch);let fine=counts.z+counts.x;
 if(ueSweepKind==1u){
  ueStagedJob=true;
  for(var job=group.x;job<counts.x;job+=groups.x){ueSweepSeamFine(ueSeamFineJob(job,lane),lane);workgroupBarrier();}
  return;
 }
 if(ueSweepKind==2u){
  for(var job=group.x;job<counts.z+(counts.y+UE_PACK-1u)/UE_PACK;job+=groups.x){
   ueStagedJob=job>=counts.z;
   if(job<counts.z){ueSweepPack(umRegularCoarseOwner(64u*job+lane));}
   else{ueSweepCoarsePack(UE_PACK*(job-counts.z),counts.y,lane);}
   workgroupBarrier();
  }
  return;
 }
 for(var job=group.x;job<fine+(counts.y+UE_PACK-1u)/UE_PACK;job+=groups.x){
  ueStagedJob=job>=counts.z;
  if(job<counts.z){ueSweepPack(umRegularCoarseOwner(64u*job+lane));}
  else if(job<fine){ueSweepSeamFine(ueSeamFineJob(job-counts.z,lane),lane);}
  else{ueSweepCoarsePack(UE_PACK*(job-fine),counts.y,lane);}
  workgroupBarrier();
 }
}
${uniformExtensionSourceBoundsWGSL}// One lane per (tile key, component), 64 keys a workgroup: component-major
// lane order, so a SIMD group restricts one component. rbFlags: bit 0
// restricted, bits 1 and 2 the lower and upper bounds' eleventh z bits.
var<workgroup> rbValues:array<f32,192>;
var<workgroup> rbLower:array<u32,192>;
var<workgroup> rbUpper:array<u32,192>;
var<workgroup> rbFlags:array<u32,192>;
@compute @workgroup_size(64,3) fn restrictBand(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_id) id:vec3u){
 // A key that holds no page (a free block, a sentinel) restricts and stores
 // nothing: the jobs are the live tiles (umLiveTile), 64 a workgroup job.
 let live=umLiveTileCount();
 for(var job=group.x;job<(live+63u)/64u;job+=groups.x){
 workgroupBarrier();
 let tile=umLiveTileOr(64u*job+id.x,UM_TILES);let component=id.y;let at=id.x+64u*component;
 // A clean region keeps its texels of the last restriction (ueRegionalRetain).
 let inside=umBlockStored(tile>>9u)&&umTileLive(tile)&&ueRestrictDirty(tile);let gid=umTileCoord(tile);let origin=vec3i(gid)*4;
 // Off the extension support every face keeps its seed distance, and a
 // source face needs a liquid owner, which the support dilates around: no
 // face here is finite, so the restriction is the empty one written below.
 // Every slot read below is anchored in this tile: a supported tile whose
 // final mask is empty restricts to the same empty state.
 let supported=inside&&any(ueMaskIn(tile)!=vec2u(0u));
 rbValues[at]=0.0;rbLower[at]=0u;rbUpper[at]=0u;rbFlags[at]=0u;
 // The tile's +face on a closed wall is seeded, never restricted (umClosedWall).
 var top=origin;top[component]+=3;
 if(supported&&!((gid[component]&7u)==7u&&umAcrossKind(top,component,1)==UM_SOLID)){
  var location=vec3f(origin)+vec3f(2);location[component]+=2.0;
  var best=UM_INF;var sum=0.0;var count=0.0;var lo=vec3i(2048);var hi=vec3i(-1);
  ${this.regularBulk?"let coarse=false;let patchWidth=1u;":`// A 4h tile's one owner has a single positive patch (width 4: only the
  // first plane anchor resolves to it) or sixteen unit patches at the plane
  // anchors; the fallback footprint holds no other patch. An h owner always
  // has one unit patch at its positive anchor, even beside a coarse
  // neighbour. No per-anchor owner or neighbour lookup is needed.
  // Unit slots take the tile's word, and the word of the tile above a 4h
  // tile with unit patches (that tile is h): loaded once for the anchors.
  let word=umTopology[tile];let coarse=(word&0x80000000u)==0u;var patchWidth=1u;var upper=0u;
  if(coarse){patchWidth=umFace(umOwnerAt(origin),component,1,0u).width;if(patchWidth==1u){var above=gid;above[component]+=1u;upper=umTopology[umTileAt(above)];}}`}
  // Restrict real MAC patches. An h tile whose +face plane no sweep reached
  // falls back to its nearest reached face anywhere in the tile, on every
  // axis (native Uniform does so on y alone): its far value then reaches the
  // +plane, which a remap to 4h averages into the coarse owner's face. On y
  // alone, a body moving along x left a coarsened owner's leading face zero.
  for(var fallback=0u;fallback<select(1u,2u,!coarse);fallback++){
   if(count>0.0){break;}
   // Preserve z/y/x order among the 16 face-plane anchors.
   for(var k=0u;k<select(select(16u,64u,fallback!=0u),1u,patchWidth==4u);k++){
    var local=umCorner(k,4u);
    if(fallback==0u){local=vec3u(k%4u,k/4u,3u);if(component==0u){local=vec3u(3u,k%4u,k/4u);}else if(component==1u){local=vec3u(k%4u,3u,k/4u);}}
    let p=origin+vec3i(local);
    ${this.regularBulk?"let o=umOwnerAt(p);let face=umPositiveFaceAtAnchor(o,component,p);if(face.width==0u){continue;}":"let face=UMFace(UMOwner(),p,patchWidth,select(1u,16u,coarse&&patchWidth==1u),component,1);"}
    ${this.regularBulk?"if(!umSource(face,o)){continue;}":"let slot=stateIn[select(UE_COARSE+component*UE_TILES+tile,ueUnitSlotIn(word,upper,local,component),patchWidth==1u)];if(slot.y>=0.5*UM_INF){continue;}"}
    let delta=(umFaceCenter(face)-location)*h.xyz;let distance=dot(delta,delta);let epsilon=1e-6*max(1.0,distance);
    if(distance<best-epsilon){best=distance;sum=0.0;count=0.0;lo=vec3i(2048);hi=vec3i(-1);}
    if(abs(distance-best)<=epsilon){sum+=${this.regularBulk?"umPhysical(face)":"slot.x"};count+=1.0;
     // The whole tangential patch contributes its original support bounds.
     var end=face.anchor+vec3i(i32(face.width)-1);end[component]=face.anchor[component];lo=min(lo,face.anchor);hi=max(hi,end);}
   }
  }
  if(count>0.0){rbValues[at]=sum/count;rbLower[at]=umSourceWord(lo);rbUpper[at]=umSourceWord(hi);rbFlags[at]=1u|((u32(lo.z)>>10u)<<1u)|((u32(hi.z)>>10u)<<2u);}
 }
 workgroupBarrier();
 if(component==0u&&inside){
  var values=vec4f(0);var lower=vec4u(0);var upper=vec4u(0);var mask=0u;
  for(var c=0u;c<3u;c++){
   let flags=rbFlags[id.x+64u*c];values[c]=rbValues[id.x+64u*c];lower[c]=rbLower[id.x+64u*c];upper[c]=rbUpper[id.x+64u*c];
   mask|=(flags&1u)<<c;lower.w|=((flags>>1u)&1u)<<(24u+c);upper.w|=((flags>>2u)&1u)<<(24u+c);
  }
  let texel=(umBlockCoord(tile>>9u)<<vec3u(3u))|vec3i(umLocalTile(tile));
  values.w=f32(mask);textureStore(coarseOut,texel,values);textureStore(originsOut,texel,lower);textureStore(originsOut,texel+UE_UPPER_BOUNDS,upper);
 }
 }
}
${uniformExtensionFarValueWGSL("umFarValue")}
// An h publish job is one tile; every far tap of its faces is one of the 27
// coarse cells around it, staged once in workgroup memory.
var<workgroup> farTile:u32;
var<workgroup> farState:array<vec4f,27>;
// Per coarse cell: bit a set where component a is known. A cell of a page
// that is not stored stages as unknown.
var<workgroup> farKnown:array<u32,27>;
fn umFarSlot(p:vec3i)->u32{let local=vec3u(p-vec3i(umTileCoord(farTile))+vec3i(1));return local.x+3u*(local.y+3u*local.z);}
// umFarValue over the staged cells: the same taps, weights, passes and order.
fn umFarValueStaged(face:UMFace)->f32{
 let point=umFaceCenter(face);var q=point/4.0-vec3f(0.5);q[face.axis]-=0.5;
 let base=vec3i(floor(q));let fraction=fract(q);var sum=0.0;var total=0.0;
 for(var farPass=0u;farPass<2u&&total<=0.0;farPass++){for(var k=0u;k<8u;k++){
  let bit=vec3i(umCorner(k,2u));let w=select(vec3f(1)-fraction,fraction,bit==vec3i(1));if((w.x*w.y*w.z>0.0)!=(farPass==0u)){continue;}
  let floored=max(w,vec3f(0.0009765625));let weight=floored.x*floored.y*floored.z;
  let slot=umFarSlot(base+bit);if((farKnown[slot]&(1u<<face.axis))==0u){continue;}
  sum+=weight*farState[slot][face.axis];total+=weight;
 }}
 return select(0.0,sum/total,total>0.0);
}
// The components some face of the tile needs a far value for: gathered from
// the lanes, then read by all of them.
var<workgroup> farAsked:atomic<u32>;
var<workgroup> farRequired:u32;
@group(1) @binding(18) var<storage,read_write> ueHeld:array<f32>;
fn ueHeldLoad(slot:u32)->f32{return ueHeld[slot];}
fn ueHeldStore(slot:u32,value:f32){ueHeld[slot]=value;}
fn ueHeldKept(slot:u32,value:f32)->f32{ueHeldStore(slot,value);return value;}
// A held region's face: its seed where it is a source (the physical value a
// swept publication takes there too), else its held value.
fn ueHeldFace(face:UMFace,owner:UMOwner,slot:u32)->f32{
 if((umTileSupport(owner.tile)&2u)!=0u&&umSource(face,owner)){return umPhysical(face);}
 return ueHeldLoad(slot);
}
fn ueHeldFine(owner:UMOwner,value:vec4f,held:bool){if(!held){for(var axis=0u;axis<3u;axis++){ueHeldStore(ueOwnSlot(owner,axis),value[axis]);}}}
fn umPublished(face:UMFace)->f32{
 if(umFaceLow(face.anchor,face.axis)||umClosedWall(face)){return umPhysical(face);}
 ${this.regularBulk?`if(umSource(face,umOwnerAt(face.anchor))){return umPhysical(face);}
 return umFarValue(face);`:`let slotIndex=umSlot(face.anchor,face.axis,face.width);
 if(ueRegionHeld(umTileAt(vec3u(face.anchor)/4u))){return ueHeldFace(face,umOwnerAt(face.anchor),slotIndex);}
 let slot=umSlotState(face.anchor,face.axis,face.width);if(slot.y<0.5*UM_INF){return ueHeldKept(slotIndex,slot.x);}
 return ueHeldKept(slotIndex,umFarValue(face));`}
}
fn uePublishFine(owner:UMOwner,lane:u32){
 if(lane==0u){farTile=select(UM_TILES,owner.tile,owner.width!=0u&&!ueRegionFrozen(owner.tile));atomicStore(&farAsked,0u);}
 if(workgroupUniformLoad(&farTile)>=UM_TILES){return;}
 // Resolve seeded and wall faces before preparing the distant-source cache.
 // Many h tiles already have every component: they publish these exact
 // values without loading or decoding the 27-cell fallback neighborhood.
 let origin=umOrigin(owner);let low=umLowOpen(vec3i(origin));var value=vec4f(0);var needsFar=0u;
 // A job is one tile: held or not, its lanes agree.
 let held=ueRegionHeld(farTile);
 for(var axis=0u;axis<3u;axis++){
  if(low[axis]){let face=umUnitExtensionFace(owner,axis,-1);boundary[umLowFaceIndex(origin,axis)]=umPhysical(face);}
  let face=umUnitExtensionFace(owner,axis,1);${this.regularBulk?"let slot=select(vec2f(0,UM_INF),vec2f(umPhysical(face),0),umSource(face,owner));":"let slot=select(vec2f(0.0,UM_INF),stateIn[ueOwnSlot(owner,axis)],(umTileSupport(owner.tile)&2u)!=0u);"}
  if(umClosedWall(face)){value[axis]=umPhysical(face);}
  else if(held){value[axis]=ueHeldFace(face,owner,ueOwnSlot(owner,axis));}
  else if(slot.y<0.5*UM_INF){value[axis]=slot.x;}
  else{needsFar|=1u<<axis;}
 }
 if(needsFar!=0u){atomicOr(&farAsked,needsFar);}
 workgroupBarrier();
 if(lane==0u){farRequired=atomicLoad(&farAsked);atomicStore(&farAsked,0u);}
 let required=workgroupUniformLoad(&farRequired);
 if(required==0u){ueHeldFine(owner,value,held);value.w=textureLoad(physical,vec3i(origin),0).w;textureStore(output,vec3i(origin),value);return;}
 if(lane<27u){
  let texel=umTileStoreTexel(vec3i(umTileCoord(farTile))+vec3i(umCorner(lane,3u))-vec3i(1));farKnown[lane]=0u;
  if(texel.x>=0){let state=textureLoad(coarse,texel,0);farState[lane]=state;farKnown[lane]=u32(round(state.w))&7u;}
 }
 workgroupBarrier();
 for(var axis=0u;axis<3u;axis++){
  if((needsFar&(1u<<axis))==0u){continue;}
  let face=umUnitExtensionFace(owner,axis,1);value[axis]=umFarValueStaged(face);
 }
 ueHeldFine(owner,value,held);value.w=textureLoad(physical,vec3i(origin),0).w;textureStore(output,vec3i(origin),value);
}
@compute @workgroup_size(64) fn publishFine(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 if(ueQuietUniform(lane)){return;}
 uePublishFine(umOwner(gid),lane);
}
var<workgroup> publishCoarseState:array<vec2f,192>;
fn uePublishCoarse(owner:UMOwner,lane:u32){
 for(var i=lane;i<192u;i+=64u){publishCoarseState[i]=vec2f(0);}
 workgroupBarrier();
 let live=owner.width!=0u&&!ueRegionFrozen(owner.tile);
 if(live){
  let origin=umOrigin(owner);
  if(lane<48u){
   let axis=lane/16u;let part=lane%16u;var local=vec3u(0);local[axis]=3u;local[(axis+1u)%3u]=part%4u;local[(axis+2u)%3u]=part/4u;
   let face=umPositiveFaceAtAnchor(owner,axis,vec3i(origin+local));if(face.width!=0u){publishCoarseState[umCoarseCell(local)+64u*axis]=vec2f(umPublished(face),1);}
  }else if(lane<51u){let axis=lane-48u;if(umLowOpen(vec3i(origin))[axis]){boundary[umLowFaceIndex(origin,axis)]=umPublished(umFace(owner,axis,-1,0u));}}
 }
 workgroupBarrier();
 let x=publishCoarseState[lane];let y=publishCoarseState[lane+64u];let z=publishCoarseState[lane+128u];
 if(live&&x.y+y.y+z.y>0.0){let anchor=vec3i(umOrigin(owner)+umCorner(lane,4u));textureStore(output,anchor,vec4f(x.x,y.x,z.x,textureLoad(physical,anchor,0).w));}
}
// h seam owners publish through uePublishFine (the same values: umPublished
// per unit face, with the staged far value); 4h seam tiles through uePublishCoarse.
@compute @workgroup_size(64) fn publishSeams(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){ueLaunch=select(ueLaunchCounts(),vec3u(0u),ueQuiet());}
 let counts=workgroupUniformLoad(&ueLaunch);
 for(var job=group.x;job<counts.x+counts.y;job+=groups.x){
  if(job<counts.x){uePublishFine(ueSeamFineJob(job,lane),lane);}
  else{uePublishCoarse(ueSeamCoarseJob(job-counts.x),lane);}
  workgroupBarrier();
 }
}
${["publishList"].map(name=>patchOnce(uniformMixedFaceDispatchWGSL(name,"umPublished(face)",false,"value.w=textureLoad(physical,ownedFace.anchor,0).w;","ueOwner"),"let origin=umOrigin(owner);",`let origin=umOrigin(owner);
 if(ueQuiet()||ueRegionFrozen(owner.tile)){return;}
 // A regular 4h owner (one-width 3x3x3 stencil) has one width-4 patch per
 // positive face, anchored at origin+3e_axis, and each of those texels holds
 // only that component (no other positive face shares the anchor). Its far
 // value (umFarValue) has one tap of positive weight: the face centre maps
 // to the owner's own coarse cell with zero fraction on every axis, so it
 // is that cell's component when it is known (its mask bit) and zero
 // otherwise. The same values as the
 // general dispatch below, without the face search.
 if(ueRegularCoarseList){
  let cell=(umBlockCoord(owner.tile>>9u)<<vec3u(3u))|vec3i(umLocalTile(owner.tile));let state=textureLoad(coarse,cell,0);let mask=u32(round(state.w));
  let low=umLowOpen(vec3i(origin));let held=ueRegionHeld(owner.tile);
  for(var axis=0u;axis<3u;axis++){
   if(low[axis]){let i=umLowFaceIndex(origin,axis);boundary[i]=negative[i];}
   var anchor=vec3i(origin);anchor[axis]+=3;
   let slot=umSlotState(anchor,axis,4u);let physicalAt=textureLoad(physical,anchor,0);var value=vec4f(0);let slotIndex=umSlot(anchor,axis,4u);
   // umClosedWall: a closed positive domain wall publishes its physical value.
   if(umAcrossKind(anchor,axis,1)==UM_SOLID){value[axis]=physicalAt[axis];}
   else if(held){value[axis]=ueHeldFace(umFace(owner,axis,1,0u),owner,slotIndex);}
   else if(slot.y<0.5*UM_INF){value[axis]=slot.x;}
   else if((mask&(1u<<axis))!=0u){value[axis]=state[axis];}
   if(!held){ueHeldStore(slotIndex,value[axis]);}
   value.w=physicalAt.w;textureStore(output,anchor,value);
  }
  return;
 }
 // A unit owner's faces are the unit patches whatever its neighbours.
 if(umRegularFine||owner.width==1u){
  var value=vec4f(0);let low=umLowOpen(vec3i(origin));
  for(var axis=0u;axis<3u;axis++){
   if(low[axis]){let face=umUnitExtensionFace(owner,axis,-1);boundary[umLowFaceIndex(origin,axis)]=umPublished(face);}
   value[axis]=umPublished(umUnitExtensionFace(owner,axis,1));
  }
  value.w=textureLoad(physical,vec3i(origin),0).w;textureStore(output,vec3i(origin),value);return;
 }`)).join("\n")}
`,["seed","sweep","seedList","sweepList","publishList","publishFine"]);
  const module=uniformDetailModule(this.device,{specializeWorldGeometry:true,label:"Uniform mixed extension",code:source});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]});
  // Only the main restriction and publication retain clean regions (bulk never marks).
  const regionalRetain=+!this.regularBulk;
  this.restrictPipeline=await uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module,entryPoint:"restrictBand",constants:{umDispatchX:this.ownership.dispatchX,ueRegionalRetain:regionalRetain}}});
  await Promise.all((this.regularBulk?["publish"]:["seed","sweep","publish"]).map(async entryPoint=>{
   const retain:Record<string,number>=entryPoint==="publish"?{ueRegionalRetain:regionalRetain}:{};
   const compile=(width:number,counted:number,name:string,list=false)=>uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module,entryPoint:name,constants:{umDispatchX:this.ownership.dispatchX,umCellWidth:width,umRegularTiles:1,umRegularFine:+(width===1),ueRegularCoarseList:+list,umCountedJobs:counted,ueStagedTiles:+(entryPoint==="sweep"&&!list),...retain}}});
   const C=UNIFORM_MIXED_COUNTED;
   this.regularPipelines.set(entryPoint,await compile(1,C.owners,entryPoint==="publish"?"publishFine":entryPoint));
   this.listPipelines.set(entryPoint,await compile(4,C.regularCoarse,`${entryPoint}List`,true));
   // The seam launch reads no launch constant; a sweep's tile jobs stage their tiles.
   this.seamPipelines.set(entryPoint,await uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module,entryPoint:`${entryPoint}Seams`,constants:entryPoint==="sweep"?{ueStagedTiles:1,ueMixedJobs:1,ueSweepKind:2}:retain}}));
   if(entryPoint==="sweep")this.fineSeamSweep=await uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module,entryPoint:"sweepSeams",constants:{ueStagedTiles:1,ueMixedJobs:0,ueSweepKind:1,ueStagePack:1,ueCacheFineRequests:1}}});
  }));
  if(this.supportsPair){
   const pairedSource=pairedExtensionSource(source,true);
   const paired=uniformDetailModule(this.device,{specializeWorldGeometry:true,label:"Uniform mixed paired extension",code:pairedSource});
   const messages=(await paired.getCompilationInfo()).messages.filter(m=>m.type==="error");if(messages.length)throw new Error(messages.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
   const pairLayout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.pairedResources!]});
   const compile=(entryPoint:string,constants:Record<string,number>={})=>uniformDetailPipeline(this.device,this.ownership,{layout:pairLayout,compute:{module:paired,entryPoint,constants}});
   this.retainPipelines=await Promise.all([compile("retainFine"),compile("retainCoarse")]);
   await Promise.all(["seed","sweep"].map(async name=>{
    const C=UNIFORM_MIXED_COUNTED;
    this.pairedRegular.set(name,await compile(`pair_${name}`,{umDispatchX:this.ownership.dispatchX,umCellWidth:1,umRegularTiles:1,umRegularFine:1,umCountedJobs:C.owners,ueStagedTiles:+(name==="sweep")}));
    this.pairedList.set(name,await compile(`pair_${name}List`,{umDispatchX:this.ownership.dispatchX,umCellWidth:4,umRegularTiles:1,umRegularFine:0,ueRegularCoarseList:1,umCountedJobs:C.regularCoarse}));
    this.pairedSeams.set(name,await compile(`pair_${name}Seams`,name==="sweep"?{ueStagedTiles:1,ueMixedJobs:1,ueSweepKind:2}:{}));
   }));
   this.pairedFineSweep=await compile("pair_sweepSeams",{ueStagedTiles:1,ueMixedJobs:0,ueSweepKind:1,ueStagePack:1,ueCacheFineRequests:1});
   this.companionRestrict=await compile("restrictBand",{umDispatchX:this.ownership.dispatchX,ueCompanionValues:1});
   this.companionRegular.set("publish",await compile("publishFine",{umDispatchX:this.ownership.dispatchX,umCellWidth:1,umRegularTiles:1,umRegularFine:1,umCountedJobs:UNIFORM_MIXED_COUNTED.owners,ueCompanionValues:1}));
   this.companionList.set("publish",await compile("publishList",{umDispatchX:this.ownership.dispatchX,umCellWidth:4,umRegularTiles:1,ueRegularCoarseList:1,umCountedJobs:UNIFORM_MIXED_COUNTED.regularCoarse,ueCompanionValues:1}));
   this.companionSeams.set("publish",await compile("publishSeams",{ueCompanionValues:1}));
   // Old publication retains geometric source selections in the dead epoch.
   // The current payload reuses them only after a matching old snapshot.
   const replay=uniformDetailModule(this.device,{specializeWorldGeometry:true,label:"Uniform mixed extension publication replay",code:replayedPublicationSource(source)});
   const replayErrors=(await replay.getCompilationInfo()).messages.filter(m=>m.type==="error");if(replayErrors.length)throw new Error(replayErrors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
   const replayPipeline=(entryPoint:string,constants:Record<string,number>)=>uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module:replay,entryPoint,constants}});
   const C=UNIFORM_MIXED_COUNTED;
   const replayEntries=await Promise.all([
    replayPipeline("publishFine",{umDispatchX:this.ownership.dispatchX,umCellWidth:1,umRegularTiles:1,umRegularFine:1,umCountedJobs:C.owners}),
    replayPipeline("publishList",{umDispatchX:this.ownership.dispatchX,umCellWidth:4,umRegularTiles:1,ueRegularCoarseList:1,umCountedJobs:C.regularCoarse}),
    replayPipeline("publishSeams",{}),
   ]);
   this.replayPublishers={regular:new Map([["publish",replayEntries[0]!]]),list:new Map([["publish",replayEntries[1]!]]),seams:new Map([["publish",replayEntries[2]!]])};
   const snapshot=uniformDetailModule(this.device,{specializeWorldGeometry:true,label:"Uniform mixed extension slot snapshot",code:pairedSnapshotSource(pairedSource,true)});
   const snapshotErrors=(await snapshot.getCompilationInfo()).messages.filter(m=>m.type==="error");if(snapshotErrors.length)throw new Error(snapshotErrors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
   const snapshotLayout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.snapshotResources!]});
   const make=(entryPoint:string)=>uniformDetailPipeline(this.device,this.ownership,{layout:snapshotLayout,compute:{module:snapshot,entryPoint,constants:{umDispatchX:this.ownership.dispatchX,ueCompanionValues:1,ueOldPayload:1}}});
   const [restrict,fine,coarse]=await Promise.all([make("restrictBand"),make("snapshotFine"),make("snapshotCoarse")]);this.snapshotPipelines={restrict,fine,coarse};
  }
  await marks;await blocks;
 }
 /** regional: seed and sweep only the changed regions (not COLD now or at
  * the last swept encode, or a stamped topology word or support bit moved)
  * and those beside them (dirty), keeping the final slots of the rest; the
  * restriction and publication keep the rest's texels, the hierarchy stays
  * whole. A full encode where the last one's slots are not retained or its
  * sweep count's parity differs. The caller asks only where the last swept
  * encode's velocity and parameters still hold (a frame's head after the
  * last tail: phase moves only where phi did, in non-COLD regions). */
 encode(encoder:GPUCommandEncoder,groups:readonly [UniformDetailGroup,UniformDetailGroup],sweeps=2,replayHierarchy=false,regional=false):void{
  this.snapshotEncoder=undefined;this.encodeStages(encoder,groups,sweeps,replayHierarchy,undefined,false,regional);
 }
 /** Before forces/pressure, retain old canonical values in both companion
  * epochs. Their inactive particle bank stays untouched until G2P resampling.
  * A saturated fixed grid strides the GPU counts without empty per-tile jobs. */
 retainPair(encoder:GPUCommandEncoder,groups:readonly [UniformDetailGroup,UniformDetailGroup]):void{
  if(!this.retainPipelines)throw new Error("Paired extension is not initialized");
  const pass=encoder.beginComputePass(RETAIN_PASS),o=this.ownership;
  pass.setBindGroup(0,o.bindGroup);pass.setBindGroup(1,groups[0].group);
  if(o.capacity.fineTiles>0){pass.setPipeline(uniformDetailPick(this.retainPipelines[0]));pass.dispatchWorkgroups(Math.min(4096,o.capacity.fineTiles));}
  pass.setPipeline(uniformDetailPick(this.retainPipelines[1]));pass.dispatchWorkgroups(Math.min(4096,o.capacity.tiles));pass.end();
 }
 /** Seed the current field and shared geometry without overwriting the old
  * scalar payload retained before pressure, then sweep both together. */
 preparePair(encoder:GPUCommandEncoder,groups:readonly [UniformDetailGroup,UniformDetailGroup],sweeps=2):void{this.snapshotEncoder=undefined;this.encodeStages(encoder,groups,sweeps,false,"pair");}
 /** Publish either payload; snapshot consumes the old one before the new
  * publication reuses the same output field. */
 finishPair(encoder:GPUCommandEncoder,groups:readonly [UniformDetailGroup,UniformDetailGroup],sweeps=2,companion=false,replayHierarchy=true):void{
  const replay=!companion&&this.snapshotEncoder===encoder&&this.snapshotSweeps===sweeps;
  this.snapshotEncoder=undefined;this.encodeStages(encoder,groups,sweeps,!companion&&replayHierarchy,companion?"companion":"current",replay);
 }
 /** One epoch's slot array (the final states after sweeps with that
  * parity): its states, then the region words. */
 finalSlots(epoch:number):GPUBufferBinding{
  if(!this.slots)throw new Error("Mixed extension slots are not lent (useSlots)");
  return {buffer:this.slots,offset:(epoch%2)*this.slotArray,size:this.slotArray};
 }
 /** Finalize old canonical slots and fine taps without materializing RGBA. */
 finishSnapshot(encoder:GPUCommandEncoder,groups:readonly [UniformDetailGroup,UniformDetailGroup],sweeps=2):void{
  if(!this.snapshotPipelines)throw new Error("Slot snapshot is not initialized");
  if(!Number.isSafeInteger(sweeps)||sweeps<0)throw new Error("Invalid mixed extension sweep count");
  const pass=encoder.beginComputePass(SNAPSHOT_PASS),p=this.snapshotPipelines,o=this.ownership;
  pass.setBindGroup(0,o.bindGroup);pass.setBindGroup(1,groups[sweeps%2]!.group);pass.setPipeline(uniformDetailPick(p.restrict));
  // The old payload's hierarchy runs only where a region publishes it
  // (gate 2); with none, the current one prolongs instead of replaying.
  pass.dispatchWorkgroups(this.restrictGroups);this.encodeHierarchy(pass,false,2);
  pass.setBindGroup(0,o.bindGroup);pass.setBindGroup(1,groups[sweeps%2]!.group);
  // The fine publisher grid-strides its live tile list; cap empty launch groups.
  if(o.capacity.fineTiles>0){pass.setPipeline(uniformDetailPick(p.fine));pass.dispatchWorkgroups(Math.min(8192,o.capacity.fineTiles));}
  // The coarse publisher strides its 4h list from a fixed grid.
  pass.setPipeline(uniformDetailPick(p.coarse));pass.dispatchWorkgroups(UNIFORM_LIVE_KEY_GROUPS);pass.end();
  this.snapshotEncoder=encoder;this.snapshotSweeps=sweeps;
 }
 private encodeStages(encoder:GPUCommandEncoder,groups:readonly [UniformDetailGroup,UniformDetailGroup],sweeps:number,replayHierarchy:boolean,mode?:"pair"|"companion"|"current",replayPublication=false,regional=false):void{
  if(this.seamPipelines.size!==(this.regularBulk?1:3)||!this.restrictPipeline)throw new Error("Mixed extension is not initialized");
  if(!Number.isSafeInteger(sweeps)||sweeps<0)throw new Error("Invalid mixed extension sweep count");
  // Bulk extension uses only the regular hierarchy. Its restriction reads
  // physical supported faces directly, and publication preserves them at h.
  const swept=!this.regularBulk&&mode!=="companion"&&mode!=="current";
  if(mode==="pair"&&!swept)return;
  // One pass: seed, sweeps, restriction, the hierarchy continuation, publish.
  if(swept){
   // Final slots: groups[sweeps%2]'s stateIn, array sweeps%2.
   // The tail's pair encode holds its quiet regions once every region's
   // validity word is current (heldReady): their slots are not retained.
   const parity=sweeps%2,marks=encoder.beginComputePass(MARK_PASS);regional=regional&&this.retained&&this.retainedParity===parity;
   const held=mode==="pair"&&this.heldReady;this.encodeMarks(marks,held?"held":regional?"regional":"full",parity);marks.end();
   this.retained=true;this.retainedParity=parity;this.heldReady=true;this.heldTail=held;
  }
  const pass=encoder.beginComputePass(EXTENSION_PASS);pass.setBindGroup(0,this.ownership.bindGroup);
  if(swept){this.encodeStage(pass,"seed",groups[1],mode,replayPublication);for(let i=0;i<sweeps;i++)this.encodeStage(pass,"sweep",groups[i%2]!,mode,replayPublication);}
  if(mode==="pair"){pass.end();return;}
  const final=groups[sweeps%2]!;this.encodeStage(pass,"restrictBand",final,mode,replayPublication);
  // A regional encode without a dirty region keeps the hierarchy (gate 1).
  // The tail's current payload replays the old one's selections where the
  // old payload was published (gate 2), else prolongs.
  // A held tail's likewise.
  this.encodeHierarchy(pass,replayHierarchy,(swept&&regional)||(mode==="current"&&this.heldTail)?1:0,mode==="current"&&replayHierarchy?2:0);pass.setBindGroup(0,this.ownership.bindGroup);this.encodeStage(pass,"publish",final,mode,replayPublication);pass.end();
 }
 /** One stage of encodeStages' pass. A method, not a closure there: a closure is a fresh object every frame. */
 private encodeStage(pass:GPUComputePassEncoder,entry:string,group:UniformDetailGroup,mode:"pair"|"companion"|"current"|undefined,replayPublication:boolean):void{
  const o=this.ownership;pass.setBindGroup(1,group.group);
  if(entry==="restrictBand"){
   // 64 live tiles a workgroup job, striding the live list.
   pass.setPipeline(uniformDetailPick((mode==="companion"?this.companionRestrict:this.restrictPipeline)!));
   pass.dispatchWorkgroups(this.restrictGroups);return;
  }
  const publication=replayPublication?this.replayPublishers:undefined;
  const regular=mode==="pair"?this.pairedRegular:mode==="companion"?this.companionRegular:publication?.regular??this.regularPipelines;
  const list=mode==="pair"?this.pairedList:mode==="companion"?this.companionList:publication?.list??this.listPipelines;
  const seams=mode==="pair"?this.pairedSeams:mode==="companion"?this.companionSeams:publication?.seams??this.seamPipelines;
  // Counted launches, each empty (not skipped) when its GPU count is zero
  // unless no h tile can exist.
  const tiles=o.capacity.tiles,packs=ceilDiv(tiles,64);
  o.dispatchTierCounted(pass,regular.get(entry)!,0);
  // Fine seams reserve only their own staged neighborhood. Regular 4h
  // owners share the coarse seam launch to avoid a third sweep dispatch.
  // A coarse-only capacity continues to use the existing list kernel.
  if(entry==="sweep"&&!o.coarseOnly){
   o.dispatchCounted(pass,(mode==="pair"?this.pairedFineSweep:this.fineSeamSweep)!,o.capacity.fineTiles,"seamFine");
   o.dispatchCounted(pass,seams.get(entry)!,ceilDiv(tiles,UE_PACK_TILES)+packs,"seams");return;
  }
  // A regular 4h owner (a 3x3x3 stencil of 4h tiles) seeds from and
  // publishes at tile origins and +face anchors only: the base blocks.
  // Its sweep touches no field.
  const based=entry!=="sweep";
  if(based)pass.setBindGroup(1,group.base);
  o.dispatchRegularCoarseCounted(pass,list.get(entry)!);
  if(based)pass.setBindGroup(1,group.group);
  if(based)o.dispatchCounted(pass,seams.get(entry)!,tiles,"hanging");
 }
}
