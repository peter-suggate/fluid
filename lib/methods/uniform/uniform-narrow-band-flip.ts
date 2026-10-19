import type { UniformRegionalFramePlan } from "./uniform-regional-frame-plan";
import { narrowBandSparseWGSL } from "./uniform-narrow-band-sparse.wgsl";
import { narrowBandAirWGSL, NARROW_BAND_AIR_CHANGED_WORD, NARROW_BAND_AIR_GATE_PLANE, NARROW_BAND_AIR_WORDS_PER_TILE } from "./uniform-narrow-band-air.wgsl";
import { narrowBandParticleStride, narrowBandParticleWGSL, narrowBandApicMomentWGSL, type NarrowBandTransfer } from "./uniform-narrow-band-apic.wgsl";
import { PARTICLE_RESIDENCY_WORDS, particleResidencyTravelWGSL } from "./uniform-local-residency";
import { uniformWindowAddressWGSL } from "./uniform-window-address";
import { narrowBandSeedShellWGSL } from "./uniform-narrow-band-seed.wgsl";
import { uniformMixedExtensionSnapshotWGSL } from "./uniform-mixed-extension-slots.wgsl";
import { narrowBandActivityTriggerWGSL, narrowBandActivityWGSL, NARROW_BAND_ACTIVITY_WORDS_PER_TILE } from "./uniform-narrow-band-activity.wgsl";
import {narrowBandReuseLayout,narrowBandReuseWGSL,type NarrowBandReuseLayout} from "./uniform-narrow-band-reuse.wgsl";
import { narrowBandRedistanceWGSL } from "./uniform-narrow-band-redistance.wgsl";
import { narrowBandMembershipWGSL } from "./uniform-narrow-band-membership.wgsl";
import { narrowBandNearestEntries, narrowBandNearestControlEntries } from "./uniform-narrow-band-nearest.wgsl";
import { NARROW_BAND_TRACE_LIMIT, narrowBandTraceWGSL, type NarrowBandTraceSubsteps } from "./uniform-narrow-band-advection.wgsl";
import { narrowBandNearWGSL } from "./uniform-narrow-band-near.wgsl";
import { narrowBandTerrainContactWGSL, narrowBandRigidContactWGSL } from "./uniform-particle-contact.wgsl";
import { uniformMixedSourceWGSL, UNIFORM_PARAMS_BYTES } from "./uniform-mixed-source.wgsl";
import { NARROW_BAND_PARTICLE_GRID, NARROW_BAND_QUIET_PARTICLE_GRID, UniformNarrowBandOrder } from "./uniform-narrow-band-order";
import { NARROW_BAND_SURFACE_RADIUS, narrowBandSurfaceWGSL, narrowBandFineTransferSource, narrowBandTransferPropertiesSource } from "./uniform-narrow-band-surface.wgsl";
import { particleBrickDepositWGSL } from "./uniform-particle-brick.wgsl";
import { UNIFORM_DETAIL_4H_LOAD } from "../../core/uniform-detail-abi";
import { UNIFORM_DETAIL_H_STORE, UNIFORM_DETAIL_MERGED_SLOTS, uniformDetailBindLayout, uniformDetailGroup, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, type UniformDetailGroup } from "./uniform-detail-fields";
import { uniformMixedTopologyWGSL, uniformMixedCertifiedEntriesWGSL } from "./uniform-mixed-topology.wgsl";
import { UNIFORM_PAGE_TABLE_WORDS, UNIFORM_PAGE_WINDOW, uniformBlockNeighbourWGSL, uniformLatticeAddressWGSL, type UniformLatticeAddress } from "./uniform-lattice-address";
import { uniformMixedFaceAddressWGSL } from "./uniform-mixed-face-dispatch.wgsl";
import { uniformMixedVertexSamplingSource } from "./uniform-mixed-vertex-sampling.wgsl";
import { uniformMixedVelocitySamplingSource } from "./uniform-mixed-velocity-sampling.wgsl";
import { uniformCoarseCacheExtent } from "./uniform-coarse-velocity-cache";
import { uniformMixedSolidWGSL, uniformMixedSolidPipeline, uniformMixedVertexBuriedWGSL, type UniformMixedSolid } from "./uniform-mixed-solid.wgsl";
import { UniformMixedMomentumCache, type UniformMixedHangingTaps } from "./uniform-mixed-momentum-cache";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformPagePoolWGSL, type UniformPagePoolLayout } from "./uniform-page-pool";
import { uniformPageInitBlocksWGSL, uniformPageInitGroups, type UniformPageInitBlocks } from "./uniform-page-initialise";
import type { GPUFluidParticleSource } from "../../core/webgpu-particle-overlay";
import { UNIFORM_COLD_FREEZE, UniformColdFreeze } from "./uniform-cold-freeze";
import type { UniformColdFrame } from "./uniform-cold-frame";
import { UNIFORM_REGION_TIER } from "./uniform-region-heat";

/** How a batch step launches: a fixed 1024-group stride, the ownership's
 * counted launch, one group, or the fixed row that strides the live
 * samples or the occupied tiles (NARROW_BAND_PARTICLE_GRID). */
// Entries of fine-only batches (reconstruct, redistanceFine: encoded only
// while the h-tile capacity is non-zero) on the h side of their group.
const NB_H_STORE_ENTRIES=new Set(["coupleFine","redistanceFine"]);
// Entries that run at any h-tile capacity, also compiled on the h side
// (name@h) for the batches encoded while the capacity is non-zero: their
// twin binds the same group, whose fields are then h textures.
const NB_H_TWIN_ENTRIES=["buildDistance"];
// Pipelines (by name) launched as a solid-free twin (name@free, umSolidFree)
// beside the general pipeline (umSolidTwinned): umSolidSkip leaves the step
// to one. buildDistance's twin is of its h-side pipeline (the redistance
// and metric batches while the h capacity is non-zero).
const NB_FREE_TWIN_ENTRIES=["coupleFine","buildDistance@h"];
const NB_LAUNCH={grid:0,counted:1,one:3,particles:4,regions:5} as const;
/** One launch of a batch, resolved once at initialization: its bind groups
 * by name (bind() replaces the groups), its pipeline and its launch. */
interface NarrowBandStep{readonly groups:string;readonly base:boolean;readonly pipeline:GPUComputePipeline;readonly launch:number}
/** A pass's launches. A frame replays them; it builds no list or label. */
interface NarrowBandBatch{readonly pass:GPUComputePassDescriptor;readonly steps:readonly NarrowBandStep[]}
interface NarrowBandBatches{
 readonly advect:NarrowBandBatch;readonly marking:NarrowBandBatch;readonly countInitial:NarrowBandBatch;readonly seedInitial:NarrowBandBatch;readonly commit:NarrowBandBatch;
 readonly activity:NarrowBandBatch;readonly search:NarrowBandBatch;readonly searchReconstruct:NarrowBandBatch;readonly searchRedistance:NarrowBandBatch;
 readonly diagnose:NarrowBandBatch;readonly reconstruct:NarrowBandBatch;readonly reconstructCoarse:NarrowBandBatch;readonly transfer:NarrowBandBatch;
 readonly snapshot:NarrowBandBatch;readonly retire:NarrowBandBatch;readonly redistanceFine:NarrowBandBatch;readonly redistanceMetric:NarrowBandBatch;readonly metric:NarrowBandBatch;readonly metricFine:NarrowBandBatch;
 readonly update:NarrowBandBatch;readonly resample:NarrowBandBatch;readonly resampleKeep:NarrowBandBatch;readonly clearAir:NarrowBandBatch;
}
const NB_REFINEMENT_PASS:GPUComputePassDescriptor={label:"Narrow-band FLIP surface refinement"};
const NB_SNAPSHOT_PASS:GPUComputePassDescriptor={label:"Narrow-band FLIP snapshot"};
const NB_UPDATE_PASS:GPUComputePassDescriptor={label:"Narrow-band FLIP update"};
const NB_INIT_PASS:GPUComputePassDescriptor={label:"Narrow-band FLIP page words"};

/** Ferstl et al. 2016: persistent samples in a band under the surface carry
 * the liquid's velocity there and overrule the advected level set (Eq. 4).
 * Every sample is liquid; none is set aside as spray.
 * R=4h, r=2h: preserve the axial quadratic footprint; diagonal kernel
 * support beyond the 2h inner collar is truncated (see the alignment notes).
 * Only the inner overlap is routinely reseeded; the outer surface is retained
 * unless persistently sparse. A source adds its whole band that step.
 * R controls lifetime on either grid width; swept crossing cells control h coverage.
 * Coupled samples compare identical liquid-face extrapolations across pressure.
 * nbParticleMode tracks transfer state: 0 coupled, 1 ballistic. A sample no resolved
 * liquid cell supports is ballistic: it keeps its staggered momentum, receives
 * gravity directly, and couples again with that momentum once supported.
 * nbParticleAge measures continuous low density in seconds, independently of
 * activity heat and transfer state; sorting and compaction carry it along. */
export interface UniformNarrowBandSlotSnapshot {
 readonly cache:UniformDetailGroup;readonly taps:UniformDetailGroup;
 readonly particles:readonly [UniformDetailGroup,UniformDetailGroup];
}
export interface NarrowBandSurfaceRevisions{readonly buffer:GPUBuffer;readonly liveOffset:number;readonly scratchOffset:number;readonly regions:number}
export class UniformNarrowBandFlip implements UniformPageInitBlocks {
 readonly label="Narrow-band FLIP page words";
 readonly capacity:number;
 readonly particleStride:number;
 private get apic():boolean{return this.transferMode==="apic";}
 readonly particles:readonly [GPUBuffer,GPUBuffer];
 readonly state:GPUBuffer;
 /** The current frame's live sample count (state word 0 after move()'s
  * commit) as a uniform's first word: the particle consumers between move
  * and update (transfer, snapshot, extension retain and old-epoch snapshot)
  * exit at once on zero. A GPU copy each move, never a host count. The same
  * word is params word 12 for this module's own kernels. */
 readonly liveGate:GPUBuffer;
 private readonly baseAllocatedBytes:number;
 get allocatedBytes():number{return this.baseAllocatedBytes+this.order.allocatedBytes;}
 /** A word per tile key from word `offset` of the buffer: non-zero where a sample's support reached at its last move (markParticleTiles). */
 get particleTiles():{buffer:GPUBuffer;offset:number;allocation:GPUBuffer}{return {buffer:this.bins,offset:2*this.cells+this.tiles,allocation:this.particleAllocation};}
 /** Current particle occupancy and signed travel, rebuilt by refine before the census. */
 private readonly particleAllocation:GPUBuffer;
 private readonly order:UniformNarrowBandOrder;
 count=0;
 reseedClipped=0;
 /** Positive distances in h cells, before union and on the accepted surface.
  * Unresolved samples are reported separately from the resolved liquid band. */
 diagnostics={beforeMaxOutside:NaN,afterMaxOutside:0,outsideSurface:0,unsupported:0,deepInterior:0};
 /** The separation before the union is a diagnostic alone, a sweep of every
  * sample that nothing in the step reads: a step measures it only for a
  * reader that set this before stepping. Unmeasured, it reports NaN, and so
  * do the accepted surface's census (afterMaxOutside, outsideSurface,
  * deepInterior); unsupported is always counted. */
 measureTraces=false;
 /** Completed frame samples, exposed for transfer/conservation verification.
  * Resampling preserves the survivors' cell order, but seeding appends an
  * unsorted tail. The complete buffer is not yet a cell-sorted epoch. */
 get activeParticles():GPUBuffer{return this.particles[this.parity]!;}
 /** After move/order and before update/resample, binding 0's source epoch
  * has been consumed. The working samples live in the other bank. Extension
  * may borrow this bank only across that interval; resample overwrites it. */
 get extensionScratch():GPUBuffer{return this.particles[this.parity]!;}
 /** The completed frame's samples for the particle layer: positions in h cells,
  * live prefix counted by the receipt's first word. Eight samples seed a cell,
  * so a sphere is a little under a quarter cell in radius. update leaves each
  * record the grid velocity it took and its depth; the inner collar is the
  * surface's share of the band. */
 get particleSource():GPUFluidParticleSource{
  const h=this.ownership.capacity.lattice.cellSize_m;
  return {buffer:this.activeParticles,strideFloats:this.particleStride,capacity:this.capacity,positionScale_m:[h[0]!,h[1]!,h[2]!],
   frame:{buffer:this.sourceParams,size:48},radius_m:0.22*Math.min(...h),liveCount:{buffer:this.state,byteOffset:0},
   grid:{velocityFloat:8,depthFloat:7,ballisticFloat:11,ballisticSignBit:!this.apic,surfaceDepth:2}};
 }
 private readonly bins:GPUBuffer;
 private readonly reuse:NarrowBandReuseLayout;
 private readonly reuseControls:GPUBuffer;
 private revisions?:NarrowBandSurfaceRevisions;
 private revisionGeneration?:()=>number;
 private seenRevisionGeneration=-1;
 private reuseParity=0;
 get regionalReuseCapacity():number{return this.reuse.capacity;}
 get regionalReuseAllocatedBytes():number{return 4*this.reuse.words;}
 /** Read-only post-run diagnostics; never used to size simulation launches. */
 get regionalReuseEvidence(){return {buffer:this.bins,regions:this.reuse.capacity,flagsOffset:4*this.reuse.nearestFlags,planeOffset:4*this.reuse.regionPlane};}
 regionalCrossingReuseEnabled=true;
 /** Exact full-scan ablation for the shared frame crossing domain. */
 regionalCrossingPlanEnabled=true;
 get crossingPlanEvidence(){return {buffer:this.bins,offset:4*this.reuse.framePlan,regions:this.reuse.regions};}
 private crossingCanPlan=false;
 private crossingDomain?:NarrowBandBatch;
 private crossingPass?:GPUComputePassEncoder;
 private readonly crossingSearches=new Map<NarrowBandBatch,NarrowBandBatch>();
 private airPlanned?:NarrowBandBatch;
 private airFused?:NarrowBandBatch;
 private crossingIsPlanned=false;
 private crossingPlanSerial=-1;
 private readonly encodeCrossingDomain=(encoder:GPUCommandEncoder):void=>{
  encoder.clearBuffer(this.bins,4*this.reuse.framePlan,16);
  encoder.clearBuffer(this.bins,4*(this.reuse.framePlan+4+6*this.reuse.regions),4);
  const plan=this.framePlan!;
  encoder.copyBufferToBuffer(plan.buffer,4*plan.layout.flags,this.bins,4*(this.reuse.framePlan+4+4*this.reuse.regions),4*this.reuse.regions);
  const pass=this.crossingPass=encoder.beginComputePass(this.compiled.search.pass);
  this.run(encoder,this.crossingDomain!,pass);
 };
 private seenCrossingReuseEnabled=true;
 /** Root epochs are byte-addressed views; no new shader binding is needed. */
 attachSurfaceRevisions(revisions:NarrowBandSurfaceRevisions,generation:()=>number):void{
  if(revisions.regions!==this.reuse.regions)throw new Error("Narrow-band revision region count differs from ownership");
  this.revisions=revisions;this.revisionGeneration=generation;this.seenRevisionGeneration=-1;
 }
 private heatTiers?:GPUBuffer;
 private framePlan?:UniformRegionalFramePlan;
 private tierPlanSerial=-1;
 attachFramePlan(plan:UniformRegionalFramePlan):void{this.framePlan=plan;}
 /** The frame's region work tiers (UniformRegionHeat.tiers): COLD regions'
  * phi producers hold and their consumers retain (narrowBandReuseWGSL). */
 attachHeat(tiers:GPUBuffer):void{
  if(tiers.size<4*this.reuse.regions)throw new Error(`Narrow-band FLIP: the work tiers hold fewer than ${this.reuse.regions} regions`);
  this.heatTiers=tiers;
 }
 /** Copy the tiers in before a stage reads them: a remap or claim may raise
  * a region to HOT after the frame head. */
 private syncTiers(encoder:GPUCommandEncoder):void{
  if(!this.heatTiers)throw new Error("Narrow-band FLIP: no region work tiers are attached");
  // The shared plan is sealed after remap/claim promotions. All subsequent
  // stages consume the same tier snapshot; standalone calls still refresh.
  if(this.framePlan?.active&&this.tierPlanSerial===this.framePlan.serial)return;
  encoder.copyBufferToBuffer(this.heatTiers,0,this.bins,4*this.reuse.tiers,4*this.reuse.regions);
  this.tierPlanSerial=this.framePlan?.active?this.framePlan.serial:-1;
 }
 exportSurfaceRevisions(encoder:GPUCommandEncoder):void{
  if(this.framePlan?.retainsGeometry)return;
  if(this.revisions)encoder.copyBufferToBuffer(this.bins,4*this.reuse.live,this.revisions.buffer,this.revisions.liveOffset,4*this.reuse.regions);
 }
 private syncSurfaceRevisions(encoder:GPUCommandEncoder,scratch=false):void{
  const source=this.revisions;if(!source)return;
  const generation=this.revisionGeneration!();
  if(generation!==this.seenRevisionGeneration){
   encoder.clearBuffer(this.bins,4*this.reuse.base,4*this.reuse.metadataWords);
   encoder.copyBufferToBuffer(source.buffer,source.liveOffset,this.bins,4*this.reuse.live,4*source.regions);
   this.seenRevisionGeneration=generation;this.reuseParity=0;
  }
  if(scratch)encoder.copyBufferToBuffer(source.buffer,source.scratchOffset,this.bins,4*this.reuse.live,4*source.regions);
 }

 private readonly next:GPUBuffer;
 private readonly params:GPUBuffer;
 private resources:GPUBindGroupLayout;
 /** The COLD freeze (attachFreeze): the transfer leaves frozen cells and faces. */
 private freeze?:UniformColdFreeze;
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 private groups:Record<string,readonly [UniformDetailGroup,UniformDetailGroup]>={};
 private batches?:NarrowBandBatches;
 /** The lattice's addressing; its cell, vertex and tile keys are the bins' bank offsets. */
 private readonly address:UniformLatticeAddress;
 private readonly cells:number;
 /** Bank A's words (NB_DEPTH_A to NB_DEPTH_B): a vertex key each, or the two packed nearest banks where those are longer. */
 private readonly vertices:number;
 private readonly surfaceHotWord:number;
 private readonly tiles:number;
 /** Per-frame parameter records, filled in place. */
 private readonly stepWords=new Float32Array(8);
 private readonly activityWords=new Float32Array(4);
 private readonly refinementWords=new Float32Array(8);
 private readonly receiptBits=new Uint32Array(2);
 private readonly receiptDistances=new Float32Array(this.receiptBits.buffer);
 private parity=0;
 private started=false;
 private bandCurrent=false;
 private get activityWord():number{return 3*this.cells+this.vertices+9*this.tiles+4;}
 /** The occupied tile list in the bins: word 0 its length, entries from
  * word 4. A tile key joins it when its coverage (NB_COVERAGE) leaves 0,
  * so the coverage is non-zero only on listed keys; the order's prepare
  * clears exactly the listed keys and empties it. Every per-tile particle
  * launch strides it on a fixed grid: none is sized by the tile keys. */
 private readonly occupiedWord:number;
 /** The far-air gate plane (NARROW_BAND_AIR_GATE_PLANE): a word a tile key,
  * after the activity planes and the air planes before it. A claimed block
  * takes 1 (advect) until the next clear computes it. */
 get airGateWord():number{return this.activityWord+(NARROW_BAND_ACTIVITY_WORDS_PER_TILE+NARROW_BAND_AIR_GATE_PLANE)*this.tiles;}
 /** The bins, which hold the gate plane at airGateWord. */
 get airGateBuffer():GPUBuffer{return this.bins;}
 /** Existing particle-bin coverage, for opt-in spatial cost sampling. */
 get performanceTileCounts():GPUBufferBinding{return {buffer:this.bins,offset:8*this.cells,size:4*this.tiles};}
 /** The particle activity heat plane (f32 a tile key, NB_ACTIVITY_HEAT) in airGateBuffer: the region heat's particle evidence. */
 get activityHeatWord():number{return this.activityWord+2*this.tiles;}
 private updateErrors?:GPUBuffer;
 /** Bind tile update evidence separately from the shape/refinement census. */
 attachUpdateErrors(buffer:GPUBuffer):void{this.updateErrors=buffer;}
 private refinementPipeline?:GPUComputePipeline;
 private refinementLayout?:GPUBindGroupLayout;
 private readonly refinementGroups:({particles:GPUBuffer;target:GPUBuffer;wordOffset:number;size:number;group:GPUBindGroup}|undefined)[]=[];
 private readonly refinementParams:GPUBuffer;
 private readonly coarseVelocity:GPUTexture;
 private cacheGroups:Record<string,UniformDetailGroup>={};
 private slotSnapshotResources?:GPUBindGroupLayout;
 private slotSnapshotPipeline?:GPUComputePipeline;
 private snapshotInput?:{raw:GPUTexture;negative:GPUBuffer};
 get snapshotUnitVelocity():GPUTexture{if(!this.stageTaps)throw new Error("FLIP is not bound");return this.stageTaps.unit;}
 private stageTaps?:{builder:UniformMixedHangingTaps;group:UniformDetailGroup;unit:GPUTexture};
 private initPipeline?:GPUComputePipeline;
 private initGroup?:GPUBindGroup;
 /** pool: the page pool whose commits claim this lattice's blocks (its state buffer and layout); without one no block is ever claimed. */
 constructor(private readonly device:GPUDevice,private readonly ownership:UniformMixedOwnership,private readonly solid:UniformMixedSolid,private readonly sourceParams:GPUBuffer,private readonly cache:UniformMixedMomentumCache,private readonly pool?:{readonly state:GPUBuffer;readonly layout:UniformPagePoolLayout},readonly transferMode:NarrowBandTransfer="flip",readonly seeding:"need"|"activity"="need",readonly fusedSurface=false,readonly sourceSeeding:"need"|"always"="need",readonly traceSubsteps:NarrowBandTraceSubsteps="travel",readonly redistanceSearch:"shared"|"fresh"="fresh",readonly particleDeposit:"tiled"|"gathered"="tiled"){
  const address=this.address=ownership.capacity.address,cells=address.cellKeys,vertices=Math.max(address.vertexKeys,address.cellKeys/2);
  if(pool&&512*pool.layout.blocks!==address.tileKeys)throw new Error(`Narrow-band FLIP: the page pool has ${pool.layout.blocks} blocks, the lattice ${address.tileKeys/512}`);
  this.cells=cells;this.vertices=vertices;this.tiles=address.tileKeys;
  // The band is a surface, but what a scene's surface grows to is not known
  // here. Budget one sample per domain cell, with a two-million floor: twice
  // the previous allocation, still capped by full-domain seeding and the
  // device's per-buffer limit at the original record size: packing must not
  // silently change the maximum sample count on a smaller adapter. A full budget defers optional reseeding and
  // is fatal to liquid a source or the initial band must add (noteReceipt).
  this.particleStride=narrowBandParticleStride(this.apic);const particleBytes=4*this.particleStride;
  this.capacity=Math.min(cells*8,Math.max(2_097_152,cells),Math.floor(device.limits.maxStorageBufferBindingSize/(this.apic?112:64)));
  const buffer=(label:string,size:number,uniform=false)=>device.createBuffer({label:`Narrow-band FLIP ${label}`,size,usage:(uniform?GPUBufferUsage.UNIFORM|GPUBufferUsage.STORAGE:GPUBufferUsage.STORAGE)|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  this.particles=[buffer("particles A",this.capacity*particleBytes),buffer("particles B",this.capacity*particleBytes)];
  this.state=buffer("receipt and particle frame",64);
  // After the air planes: the occupied tile list (occupiedWord).
  this.occupiedWord=3*cells+vertices+address.tileKeys*(9+NARROW_BAND_ACTIVITY_WORDS_PER_TILE+NARROW_BAND_AIR_WORDS_PER_TILE)+4;
  const binWords=this.occupiedWord+4+address.tileKeys;
  const limit=Math.min(device.limits.maxStorageBufferBindingSize,device.limits.maxBufferSize);
  const completeReuse=narrowBandReuseLayout(binWords,8*address.blocks);
  this.reuse=4*(binWords+completeReuse.words+address.vertexKeys)<=limit?completeReuse:narrowBandReuseLayout(binWords,8*address.blocks,0);
  this.surfaceHotWord=binWords+this.reuse.words;
  const binBytes=4*(this.surfaceHotWord+address.vertexKeys);
  if(binBytes>limit)throw new Error(`Narrow-band bins and bounded crossing reuse (${binBytes} bytes) exceed GPU buffer limit ${limit}`);
  this.bins=buffer("cell counts, surface metrics and persistent regional crossing masks",binBytes);
  this.reuseControls=buffer("regional reuse pass controls",72);device.queue.writeBuffer(this.reuseControls,0,new Uint32Array([0,1,1,1,1,1,0,1,0,1,1,0,0,0,0,1,0,0]));
  this.next=buffer("cell starts and compact neighbor samples",cells*4+this.capacity*28);this.params=buffer("parameters",64,true);
  this.liveGate=buffer("live sample gate",16,true);device.queue.writeBuffer(this.liveGate,0,new Uint32Array([1,0,0,0]));
  this.order=new UniformNarrowBandOrder(device,address,ownership.presentation.buffer,this.particles,this.bins,this.next,this.state,this.occupiedWord,undefined,this.apic);
  this.particleAllocation=buffer("particle allocation support",8*address.blocks*PARTICLE_RESIDENCY_WORDS*4);
  this.refinementParams=buffer("refinement prediction",48,true);device.queue.writeBuffer(this.refinementParams,32,new Uint32Array([address.blocks,0,0,0]));
  const size=uniformCoarseCacheExtent(address);
  this.coarseVelocity=device.createTexture({label:"Narrow-band FLIP stage sampling cache",size,dimension:"3d",format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING});
  this.baseAllocatedBytes=[...this.particles,this.state,this.bins,this.next,this.params,this.liveGate,this.particleAllocation,this.refinementParams,this.reuseControls].reduce((n,b)=>n+b.size,0)+size.reduce((n,v)=>n*v,16);
  if(device.limits.maxStorageBuffersPerShaderStage>=9)this.slotSnapshotResources=uniformDetailBindLayout(device,{entries:[
   ...[1,2,4].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}})),
   ...[5,17].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage" as const}})),
   {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   ...[8,10,11].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
  ]});
  this.resources=this.resourceLayout(false);
 }
 /** The stage resources; with the freeze, its region snapshot at binding 18. */
 private resourceLayout(freeze:boolean):GPUBindGroupLayout{
  return uniformDetailBindLayout(this.device,{entries:[
   ...(freeze?[{binding:18,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"2d" as const}}]:[]),
   ...[0,1,2,3,4,5].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:(binding===5?"read-only-storage":"storage") as GPUBufferBindingType}})),
   {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   ...[7,8,10,11,15].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
   {binding:14,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"read-write",format:"r32float",viewDimension:"3d"}},
   {binding:9,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"3d"}},
   {binding:16,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   // The haloed copy of binding 11's taps (UniformMixedHangingTaps.haloOf): no detail field.
   {binding:12,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
  ]},[12]);
 }
 /** The COLD freeze (UniformColdFreeze; coldRegions "frozen16h"), before
  * bind and initialize: the transfer writes no texel of a frozen cell and
  * keeps the departure on a warm texel's frozen faces, and a thawing
  * region's deposits take its coarse term (umThawFaceTerm). */
 /** This advance's cold-frame evidence (beginAdvance). quiet: every live
  * region was COLD at the evidence frame (UniformColdFrame), so no sample is
  * seeded (seeding needs activity target heat, which raises its region off
  * COLD: a lagged region seeds from the frame its tier is the evidence).
  * idle: quiet at this advance and the last, with the freeze: the last
  * advance seeded nothing and the evidence frame's receipt counted no live
  * sample, so none is live now. Every COLD region is then frozen (no
  * transfer write) and holds its phi (reconstruction, air, redistance), so
  * the whole sample and surface machinery leaves its retained state exactly
  * as its launches would: move, reconstruct, clearAir, retire, redistance,
  * transfer, snapshot, update and refine are not encoded. activity is: its
  * plane is the region heat's particle evidence, which ends an idle run. */
 private quiet=false;private quietLast=false;
 private particleGroups=NARROW_BAND_PARTICLE_GRID;
 private idleNow=false;
 get idle():boolean{return this.idleNow;}
 beginAdvance(cold:UniformColdFrame):void{
  const quiet=cold.quiet;
  // Only launch parallelism changes, never the GPU count or seeding policy.
  // HOT/unknown frames retain the populated-throughput grid. The smaller
  // direct grid still strides every sample if work grows before its receipt.
  this.particleGroups=cold.ceiling<UNIFORM_REGION_TIER.hot?NARROW_BAND_QUIET_PARTICLE_GRID:NARROW_BAND_PARTICLE_GRID;
  this.order.workgroups=this.particleGroups;
  this.idleNow=quiet&&this.quietLast&&this.freeze!==undefined&&this.started;
  this.quiet=quiet;this.quietLast=quiet;
 }
 attachFreeze(freeze:UniformColdFreeze|undefined):void{
  if(Object.keys(this.groups).length||this.freeze)throw new Error("Narrow-band FLIP: the freeze attaches once, before bind");
  this.freeze=UNIFORM_COLD_FREEZE?freeze:undefined;
  if(this.freeze)this.resources=this.resourceLayout(true);
 }
 bind(fields:{phi:GPUTexture;phiScratch:GPUTexture;phase:GPUTexture;velocity:GPUTexture;velocityScratch:GPUTexture;departure:GPUTexture;negative:GPUBuffer;negativeScratch:GPUBuffer;negativeDeparture:GPUBuffer;coarseExtended:GPUTexture;unitExtended:GPUTexture;hanging:UniformMixedHangingTaps;sharedStageUnit?:GPUTexture}):void{
  const f=fields;this.snapshotInput={raw:f.departure,negative:f.negativeDeparture};
  // Snapshot, update and seed sample velocityScratch after it is rewritten.
  // Its unit taps are filled first, against this stage's own 4h cache and
  // into the stage tap texture, so a seam tap is one load there as it is
  // in advection. Viscosity requires separate taps on the unpaired path.
  // The transfer writes the stored velocity: momentum was its last reader,
  // and the pressure split rewrites every face of it after the forces.
  // The paired-extension path snapshots after forces, so its old frame taps
  // are dead and can also hold the snapshot/update taps. The unpaired path
  // snapshots before forces and needs separate storage.
  // One texture for the solver's life: a detail field is created before the
  // first layout, and the frame binds its stages again when capacity grows.
  const unit=this.stageTaps?.unit??f.sharedStageUnit??f.hanging.createTaps("Narrow-band FLIP stage unit taps");
  this.stageTaps={builder:f.hanging,unit,group:f.hanging.bind({extended:f.velocityScratch,negative:f.negativeScratch,coarse:this.coarseVelocity},unit)};
  for(const [name,velocity,negative,output] of [["bootstrap",f.velocityScratch,f.negativeScratch,f.departure],["advectParticles",f.velocityScratch,f.negativeScratch,f.departure],["seed",f.departure,f.negativeDeparture,f.velocityScratch],["transfer",f.departure,f.negativeDeparture,f.velocity],["snapshot",f.velocityScratch,f.negativeScratch,f.departure],["surface",f.velocity,f.negative,f.departure],["reconstruct",f.velocity,f.negative,f.departure],["redistance",f.velocity,f.negative,f.departure],["redistanceStore",f.velocity,f.negative,f.departure],["update",f.velocityScratch,f.negativeScratch,f.departure]] as const){
   this.cacheGroups[name]=this.cache.bind({extended:velocity,negative,coarseExtended:this.coarseVelocity});
   this.groups[name]=[0,1].map(parity=>uniformDetailGroup(this.device,{layout:this.resources,entries:[
    {binding:0,resource:{buffer:this.particles[parity]!}},{binding:1,resource:{buffer:this.particles[1-parity]!}},
    {binding:2,resource:{buffer:this.bins}},{binding:3,resource:{buffer:this.next}},{binding:4,resource:{buffer:this.state}},
    {binding:5,resource:{buffer:negative}},{binding:6,resource:{buffer:this.params}},
    {binding:7,resource:name==="reconstruct"||name==="redistanceStore"?f.phiScratch:f.phi},{binding:8,resource:velocity},{binding:9,resource:output},
    {binding:10,resource:name==="advectParticles"?f.coarseExtended:this.coarseVelocity},
    {binding:11,resource:name==="snapshot"||name==="update"?unit:f.unitExtended},{binding:12,resource:f.hanging.haloOf(name==="snapshot"||name==="update"?unit:f.unitExtended)},{binding:14,resource:name==="reconstruct"||name==="redistanceStore"?f.phi:f.phiScratch},
    {binding:15,resource:f.phase},
    {binding:16,resource:{buffer:this.sourceParams,size:UNIFORM_PARAMS_BYTES}},
    ...(this.freeze?[{binding:18,resource:this.freeze.regions.createView()}]:[]),
   ]})) as [UniformDetailGroup,UniformDetailGroup];
  }
 }
 /** Bind the completed old scalar epoch; both particle parities share its values. */
 bindSnapshotSlots(slots:GPUBufferBinding):UniformNarrowBandSlotSnapshot{
  if(!this.slotSnapshotResources||!this.snapshotInput||!this.stageTaps)throw new Error("Slot snapshot is not available");
  const f=this.snapshotInput,unit=this.stageTaps.unit;
  return {
   cache:this.cache.bind({extended:f.raw,negative:f.negative,coarseExtended:this.coarseVelocity,slots,gate:this.liveGate}),
   taps:this.stageTaps.builder.bind({extended:f.raw,negative:f.negative,coarse:this.coarseVelocity,slots,gate:this.liveGate},unit),
   particles:[0,1].map(parity=>uniformDetailGroup(this.device,{layout:this.slotSnapshotResources!,entries:[
    {binding:1,resource:{buffer:this.particles[1-parity]!}},{binding:2,resource:{buffer:this.bins}},{binding:4,resource:{buffer:this.state}},
    {binding:5,resource:{buffer:f.negative}},{binding:6,resource:{buffer:this.params}},
    {binding:8,resource:f.raw},{binding:10,resource:this.coarseVelocity},{binding:11,resource:unit},{binding:17,resource:slots},
   ]})) as [UniformDetailGroup,UniformDetailGroup],
  };
 }
 async initialize():Promise<void>{
  const subgroups=this.device.features.has("subgroups")&&(this.device.adapterInfo.subgroupMinSize??0)>=32;
  await Promise.all([this.cache.initialize(),this.order.initialize()]);
  this.refinementLayout=this.device.createBindGroupLayout({entries:[
   {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
   {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
  ]});
  const refine=this.device.createShaderModule({label:"FLIP surface refinement prediction",code:/* wgsl */`
${narrowBandParticleWGSL(this.apic)}
@group(0) @binding(0) var<storage,read> samples:array<Particle>;
@group(0) @binding(1) var<storage,read> count:array<u32>;
@group(0) @binding(2) var<storage,read_write> joins:array<atomic<u32>>;
// blocks.x: the pool's blocks.
struct Params{dimsDt:vec4f,h:vec4f,blocks:vec4u}
@group(0) @binding(3) var<uniform> p:Params;
@group(0) @binding(4) var<storage,read> bins:array<u32>;
// The box and the blocks are this module's uniform, the pages the
// simulation topology's tail: the address functions read them.
@group(0) @binding(5) var<storage,read> nbRefineTopology:array<u32>;
@group(0) @binding(6) var<storage,read_write> allocation:array<atomic<u32>>;
fn nbRefineCells()->vec3u{return vec3u(p.dimsDt.xyz);}
fn nbRefineTiles()->u32{return 512u*p.blocks.x;}
${uniformWindowAddressWGSL("nbRefineTopology","5u*nbRefineTiles()","p.blocks.x")}
${uniformLatticeAddressWGSL({constant:false,paged:{blocks:"p.blocks.x"}})}
// The swept tiles of a sample, and its own tile's key.
struct Box{low:vec3i,high:vec3i,home:u32}
fn sweptSurface(sample:Particle)->Box{
 let start=sample.position.xyz;let motion=select(sample.before.xyz,sample.velocity.xyz,nbParticleMode(sample)>0.5);
 let travel=p.dimsDt.w*motion/p.h.xyz;
 // A live sample lies in a stored cell: its tile's key is one table read, and its cell's key follows from it.
 let c=vec3u(start);let home=umTileAt(c/4u);
 let bank=2u*UM_CELL_KEYS()+2u*nbRefineTiles()+max(UM_VERTEX_KEYS(),UM_CELL_KEYS()/2u);
 // The nearest-crossing bank holds a tile only within two tiles of a crossing (its reach bit).
 let reach=bins[bank+UM_CELL_KEYS()+4u+2u*nbRefineTiles()+home];
 var nearest=0xffffffffu;if((reach&1u)!=0u){nearest=bins[bank+64u*home+((c.x&3u)|((c.y&3u)<<2u)|((c.z&3u)<<4u))];}
 var low=start-${NARROW_BAND_SURFACE_RADIUS};var high=start+${NARROW_BAND_SURFACE_RADIUS};
 // Interior samples request the actual crossing cell, not their 4h collar.
 // Escaped samples still request their own swept support for reconstruction.
 if(sample.velocity.w<=0.0&&nearest!=0xffffffffu){
  low=vec3f(umCellCoord(nearest));high=low+1.0;
 }
 // No wall clamps the box: a tile that holds no storage has no key and takes
 // no mark (refine). It is kept in the page table's window, which the keys wrap.
 let lo=clamp(vec3i(floor((low+min(travel,vec3f(0))-p.h.w)/4.0)),vec3i(0),vec3i(${8*UNIFORM_PAGE_WINDOW-1}));
 let hi=clamp(vec3i(floor((high+max(travel,vec3f(0))+p.h.w-1e-5)/4.0)),vec3i(0),vec3i(${8*UNIFORM_PAGE_WINDOW-1}));
 return Box(lo,max(lo,hi),home);
}
${particleResidencyTravelWGSL}
// Include newly seeded and ballistic particles, not just last move's tile
// marks. Their own velocities need not be bounded by the extended grid.
fn allocationParticle(sample:Particle){
 if(sample.position.x<0.0||sample.position.w<=0.0){return;}
 let travel=lrParticleTravel(sample.velocity.xyz,sample.before.xyz,nbParticleMode(sample)==1.0,p.dimsDt.w/p.h.xyz);
 let low=travel[0];let high=travel[1];
 // Preserve the particle's two-cell gather/reconstruction support as seeds,
 // exactly as markParticleTiles does. Each stage expands this footprint
 // independently; particle travel must not expand unrelated near-air seeds.
 let lo=vec3i(floor((sample.position.xyz-2.0)/4.0));let hi=vec3i(floor((sample.position.xyz+2.0)/4.0));
 let first=lo>>vec3u(2u);let last=hi>>vec3u(2u);
 for(var z=first.z;z<=last.z;z++){for(var y=first.y;y<=last.y;y++){for(var x=first.x;x<=last.x;x++){
  let region=vec3i(x,y,z);let origin=4*region;let tile=umTileKey(origin);if(!umKeyStored(tile)){continue;}
  let octant=vec3u(region)&vec3u(1u);let page=8u*(tile>>9u)+octant.x+2u*octant.y+4u*octant.z;
  let a=vec3u(max(lo-origin,vec3i(0)));let b=vec3u(min(hi-origin,vec3i(3)));
  let row=((2u<<(b.x-a.x))-1u)<<a.x;var mask=vec2u(0u);
  for(var cz=a.z;cz<=b.z;cz++){for(var cy=a.y;cy<=b.y;cy++){let bit=4u*cy+16u*cz;mask[bit>>5u]|=row<<(bit&31u);}}
  // These words only grow during this dispatch. Avoid serializing the many
  // samples that publish an already-covered region. A stale read can only
  // cause a redundant atomic; every missing bit or larger bound still joins.
  for(var k=0u;k<2u;k++){if(mask[k]!=0u&&(atomicLoad(&allocation[8u*page+k])&mask[k])!=mask[k]){atomicOr(&allocation[8u*page+k],mask[k]);}}
  for(var axis=0u;axis<3u;axis++){
   if(low[axis]>0.0&&atomicLoad(&allocation[8u*page+2u+axis])<bitcast<u32>(low[axis])){atomicMax(&allocation[8u*page+2u+axis],bitcast<u32>(low[axis]));}
   if(high[axis]>0.0&&atomicLoad(&allocation[8u*page+5u+axis])<bitcast<u32>(high[axis])){atomicMax(&allocation[8u*page+5u+axis],bitcast<u32>(high[axis]));}
  }
 }}}

}
@compute @workgroup_size(64) fn refine(@builtin(global_invocation_id) gid:vec3u){
 for(var i=gid.x;i<min(count[0],arrayLength(&samples));i+=65536u){
  let sample=samples[i];allocationParticle(sample);if(sample.velocity.w < -1.0||nbParticleMode(sample)==1.0){continue;}
  let box=sweptSurface(sample);let page=vec3i(vec3u(sample.position.xyz)>>vec3u(5u));
  // Resampling preserves neighboring survivors; the seed tail is unsorted.
  // A superset sweep can mark for both, regardless of their order.
  if(i+1u<min(count[0],arrayLength(&samples))){
   let other=samples[i+1u];
   if(other.velocity.w>=-1.0&&nbParticleMode(other)!=1.0){let b=sweptSurface(other);if(all(b.low<=box.low)&&all(b.high>=box.high)){continue;}}
  }
  for(var z=box.low.z;z<=box.high.z;z++){for(var y=box.low.y;y<=box.high.y;y++){for(var x=box.low.x;x<=box.high.x;x++){
   // A bit a tile key. A tile of the sample's page shares its block; another asks the table.
   let t=vec3i(x,y,z);var tile=(box.home&~511u)|umTileLocal(vec3u(t));
   if(any((t>>vec3u(3u))!=page)){tile=umTileKey(t);if(!umKeyStored(tile)){continue;}}
   atomicOr(&joins[tile/32u],1u<<(tile%32u));
  }}}
 }
}`});
  const refineErrors=(await refine.getCompilationInfo()).messages.filter(m=>m.type==="error");
  if(refineErrors.length)throw new Error(refineErrors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  this.refinementPipeline=await this.device.createComputePipelineAsync({layout:this.device.createPipelineLayout({bindGroupLayouts:[this.refinementLayout]}),compute:{module:refine,entryPoint:"refine"}}).catch(e=>{throw new Error(`Refinement pipeline: ${e.message} ${e.reason}`,{cause:e});});
  if(this.pool){
   // The activity heat and target are the two words a tile key carries from step to step: a claimed block's keys take 0, a tile never active.
   const heat=this.activityWord+2*this.tiles,module=this.device.createShaderModule({label:"Narrow-band FLIP page words",code:/* wgsl */`${uniformPagePoolWGSL(this.pool.layout,0,0)}
@group(0) @binding(1) var<storage,read_write> bins:array<u32>;
${uniformPageInitBlocksWGSL(this.pool.layout,"nbInitBlocks",512,`bins[${heat}u+key]=0u;bins[${heat+this.tiles}u+key]=0u;bins[${this.airGateWord}u+key]=1u;`)}`});
   const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
   if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
   const layout=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
   this.initGroup=this.device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:this.pool.state}},{binding:1,resource:{buffer:this.bins}}]});
   this.initPipeline=await this.device.createComputePipelineAsync({layout:this.device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint:"nbInitBlocks"}});
  }
  const sourceCode=(unitTaps:boolean,slots=false)=>uniformMixedCertifiedEntriesWGSL((this.device.features.has("subgroups")?"enable subgroups;\n":"")+uniformMixedTopologyWGSL(this.ownership.capacity,0)+/* wgsl */`
const nbSparseBins:bool=${this.order.sparse};
${narrowBandParticleWGSL(this.apic)}
@group(1) @binding(0) var<storage,read_write> source:array<Particle>;
@group(1) @binding(1) var<storage,read_write> particles:array<Particle>;
@group(1) @binding(2) var<storage,read_write> bins:array<atomic<u32>>;
@group(1) @binding(3) var<storage,read_write> links:array<u32>;
@group(1) @binding(4) var<storage,read_write> state:array<atomic<u32>>;
@group(1) @binding(5) var<storage,read> negative:array<f32>;
struct Params{hDt:vec4f,settings:vec4f,activity:vec4f,live:vec4u}
@group(1) @binding(6) var<uniform> params:Params;
// This frame's live sample count (move's commit): a uniform, so a kernel
// with barriers may exit on it.
fn nbParticlesLive()->bool{return params.live.x!=0u;}
@group(1) @binding(7) var phi:texture_3d<f32>;
@group(1) @binding(8) var velocity:texture_3d<f32>;
@group(1) @binding(9) var output:texture_storage_3d<rgba32float,write>;
@group(1) @binding(10) var coarseVelocity:texture_3d<f32>;
@group(1) @binding(11) var unitVelocity:texture_3d<f32>;
${unitTaps&&!slots?"@group(1) @binding(12) var unitHalo:texture_3d<f32>;":""}
@group(1) @binding(14) var outputPhi:texture_storage_3d<r32float,read_write>;
@group(1) @binding(15) var phase:texture_3d<f32>;
${slots?UniformColdFreeze.wgsl(undefined,1,18):UniformColdFreeze.wgsl(this.freeze,1,18)}
${this.updateErrors?`// Packed certificate copy: reuse the bins binding (the full FLIP layout is at its buffer limit).
const NB_UPDATE_ERRORS:u32=${this.activityWord+7*this.tiles}u;
fn tuSpeed(t:u32)->f32{
 let at=NB_UPDATE_ERRORS+4u*t;
 if(!umKeyStored(t)||atomicLoad(&bins[at+2u])!=umBlockWord(t>>9u)||bitcast<f32>(atomicLoad(&bins[at+3u]))<=0.0){return 3.402823e38;}
 return bitcast<f32>(atomicLoad(&bins[at]));
}`:"fn tuSpeed(t:u32)->f32{return 3.402823e38;}"}
// A thawing region's coarse term on an h cell's three positive faces.
fn nbThaw(cell:vec3u)->vec3f{return vec3f(umThawFaceTerm(vec3i(cell),0u,1u),umThawFaceTerm(vec3i(cell),1u,1u),umThawFaceTerm(vec3i(cell),2u,1u));}
// The spatial order leaves the samples sorted by cell key, a tile's 64 cells
// together and x fastest inside it. A cell's samples are one contiguous run
// from its start, and the cells of an x row inside a tile follow one another,
// so a gather reads memory in order and follows no list. The bins hold each
// cell's count at its key; the head of the link arena holds its start.
// The samples [x,y) of cells a to b, two cells of one x row of one tile.
// A tile that holds no storage has no samples: the run is empty.
fn nbCellLocal(c:vec3i)->u32{let l=vec3u(c)&vec3u(3u);return l.x|(l.y<<2u)|(l.z<<4u);}
fn nbRun(a:vec3i,b:vec3i)->vec2u{
 let tile=umTileKey(a>>vec3u(2u));if(!umKeyStored(tile)){return vec2u(0);}
 ${this.order.sparse?"if(atomicLoad(&bins[NB_COVERAGE+tile])==0u){return vec2u(0);}":""}
 let last=64u*tile+nbCellLocal(b);return vec2u(links[64u*tile+nbCellLocal(a)],links[last]+atomicLoad(&bins[last]));}
fn nbCellRun(c:vec3i)->vec2u{return nbRun(c,c);}
// Position and motion remain compact streams for the neighbor gathers.
// Reconstruction reads position.w's heat to restrict erasure to its nearest
// represented patch. classify borrows only its sign bit for mode, leaving three
// full-precision motion floats per particle instead of a second vec4.
const NB_SURFACE_HOT_NEAREST=${this.surfaceHotWord}u;
fn nbSurfaceParticleHot(i:u32)->bool{return bitcast<f32>(links[UM_CELL_KEYS+4u*i+3u]&0x7fffffffu)>1.0;}
fn nbPosition(i:u32)->vec3f{let a=UM_CELL_KEYS+4u*i;return bitcast<vec3f>(vec3u(links[a],links[a+1u],links[a+2u]));}
fn nbMotion(i:u32)->vec4f{let a=UM_CELL_KEYS+4u*arrayLength(&particles)+3u*i;let mode=links[UM_CELL_KEYS+4u*i+3u]>>31u;return vec4f(bitcast<vec3f>(vec3u(links[a],links[a+1u],links[a+2u])),f32(mode));}
fn nbStoreMotion(i:u32,p:vec4f){let a=UM_CELL_KEYS+4u*arrayLength(&particles)+3u*i;let v=bitcast<vec3u>(p.xyz);links[a]=v.x;links[a+1u]=v.y;links[a+2u]=v.z;let flag=UM_CELL_KEYS+4u*i+3u;links[flag]=(links[flag]&0x7fffffffu)|(u32(p.w)<<31u);}
// Live particle counts per tile drive sparse cell-bin clearing and ordering.
const NB_COVERAGE:u32=2u*UM_CELL_KEYS;
const NB_SURFACE_TILES:u32=NB_COVERAGE+UM_TILES;
// The occupied tile list (UniformNarrowBandFlip.occupiedWord): its length,
// then a tile key an entry. A key joins when its coverage leaves 0.
const NB_OCCUPIED:u32=${this.occupiedWord}u;
fn nbCover(tile:u32,samples:u32){if(atomicAdd(&bins[NB_COVERAGE+tile],samples)==0u){atomicStore(&bins[NB_OCCUPIED+4u+atomicAdd(&bins[NB_OCCUPIED],1u)],tile);}}
${uniformMixedFaceAddressWGSL}
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
fn umLoadCorner(p:vec3u)->f32{return ${UNIFORM_DETAIL_4H_LOAD}textureLoad(phi,vec3i(p),0).x;}
// The same loads of the eight vertices p + s*umCorner(k,2u): one page-table read where one block holds them.
fn umLoadVertex8(p:vec3u,s:u32)->array<f32,8>{return udLoad8_phi(vec3i(p),s);}
fn umLoadCorner8(p:vec3u,s:u32)->array<f32,8>{return udLoad48_phi(vec3i(p),s);}
${uniformMixedVertexSamplingSource("",true,undefined,"umLoadCorner",true)}
${slots?uniformMixedExtensionSnapshotWGSL(1,17):""}
fn umLoadMixedFace(anchor:vec3i,axis:u32)->f32{
 if(umFaceLow(anchor,axis)){return negative[umLowFaceIndex(umLowFaceAbove(anchor,axis),axis)];}
 return ${slots?"ueSnapshotFace(anchor,axis)":"textureLoad(velocity,anchor,0)[axis]"};
}
${uniformMixedVelocitySamplingSource(false,"coarseVelocity","velocity",undefined,unitTaps?"unitVelocity":undefined,slots?"ueSnapshotUnit":undefined,unitTaps&&!slots?"unitHalo":undefined)}
${uniformMixedSolidWGSL(2,this.solid.coarse?.count)}
${uniformMixedVertexBuriedWGSL}
${narrowBandTerrainContactWGSL}
${narrowBandRigidContactWGSL}
${uniformMixedSourceWGSL(16)}
// advectParticles is compiled twice. Its fast launch (nbFastSample, and
// umSolidFree: it runs only while the scene holds no solid) takes
// only a fine tile's unit interpolant from its page's halo block (exactly
// umSampleVelocity's value there: weight 1, eight loads at plain offsets),
// so the general sampler's code is not in that pipeline. A sample anywhere
// else marks the particle slow: it is listed, untouched, for the general
// launch (advectParticlesDeferred), which advects it from the same record.
override nbFastSample:bool=false;
var<private> nbSlow:bool=false;
fn sampleVelocity(p:vec3f)->vec3f{
 ${unitTaps&&!slots?`if(nbFastSample){
  let s=umSampleTile(p);
  if(!umRegularFine&&umCachedFineTile(s.key)){
   umUnitHome=umUnitHaloHome(s);
   if(umUnitHome.w>=0){
    var v=vec3f(0.0);
    for(var a=0u;a<3u;a++){
     var offset=vec3f(0.5);offset[a]=1.0;
     let b=umHold;var lower=b.lo;lower[a]-=1.0;
     let q=clamp(p-offset,lower,b.hi-vec3f(1.0));
     let base=vec3i(floor(q));var terms:array<f32,8>;
     for(var k=0u;k<8u;k++){terms[k]=textureLoad(unitHalo,base+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u))+umUnitHome.xyz,0)[a];}
     v[a]=0.0+1.0*umVelocityLerp8(terms,fract(q));
    }
    return v;
   }
  }
  nbSlow=true;return vec3f(0.0);
 }`:""}
 return umSampleVelocity(p);
}
${this.apic?/* wgsl */`
// Extend the velocity at domain walls without changing the quadratic moments.
// Interior solid values come from the same extrapolated field as FLIP.
${unitTaps&&!slots?`// A particle of a unit tile reads its taps at plain offsets from its page's
// halo block (every tap is then in a unit or a slotted tile, whose halo
// texel is umVelocityTap1's value: channel w); a tap the halo does not hold
// repeats the sample through umVelocityTap1.
var<private> nbApicHome:vec4i=vec4i(-1);
var<private> nbApicValid:bool=true;`:""}
fn nbApicTap(index:vec3i,axis:u32)->f32{
 var low=vec3i(umHold.lo);low[axis]-=1;
 let held=clamp(index,low,vec3i(umHold.hi)-1);
 ${unitTaps&&!slots?"if(nbApicHome.w>=0){let t=textureLoad(unitHalo,held+nbApicHome.xyz,0);nbApicValid=nbApicValid&&t.w>0.0;return t[axis];}":""}
 return umVelocityTap1(held,axis);
}
${narrowBandApicMomentWGSL}
fn nbApicSample(p:Particle)->Particle{
 umHold=umPointBounds(p.position.xyz,8.0);
 ${unitTaps&&!slots?`let s=umSampleTileHeld(p.position.xyz);nbApicHome=select(vec4i(-1),umUnitHaloHome(s),umTileWidth(s.key)==1u);
 // One row call site: a row is umVelocityTap1's single caller.
 var rows:array<vec4f,3>;
 for(var attempt=0u;attempt<2u;attempt++){
  nbApicValid=true;
  for(var a=0u;a<3u;a++){rows[a]=nbApicRow(p.position.xyz,a);}
  if(nbApicValid||nbApicHome.w<0){break;}
  nbApicHome=vec4i(-1);
 }
 let x=rows[0];let y=rows[1];let z=rows[2];`:"let x=nbApicRow(p.position.xyz,0u);let y=nbApicRow(p.position.xyz,1u);let z=nbApicRow(p.position.xyz,2u);"}
 var result=p;result.velocity=vec4f(x.w,y.w,z.w,p.velocity.w);
 result.cx=vec4f(x.xyz,0);result.cy=vec4f(y.xyz,0);result.cz=vec4f(z.xyz,0);return result;
}`:""}
fn nbSeedParticle(q:vec4f,heat:f32)->Particle{
 var p=Particle();p.position=vec4f(q.xyz,heat);p.velocity.w=q.w;
 ${this.apic?"p=nbApicSample(p);":"p.velocity=vec4f(sampleVelocity(q.xyz),q.w);"}
 p.before=vec4f(${this.apic?"sampleVelocity(q.xyz)":"p.velocity.xyz"},0);return p;
}
fn bandPhi(p:vec3f)->f32{
 let value=umSampleVertex(p);
 return value/min(params.hDt.x,min(params.hDt.y,params.hDt.z));
}
// The block neighbour rows follow the block records in the topology: the
// kind-aware clamps read them.
fn umBlockNeighbourWord(b:u32,i:u32)->u32{return umTopology[UM_BLOCK_RECORD+UM_BLOCKS+27u*b+i];}
${uniformBlockNeighbourWGSL}
${narrowBandNearWGSL}
${narrowBandReuseWGSL(this.reuse)}
${narrowBandMembershipWGSL}
${narrowBandRedistanceWGSL}
${narrowBandActivityTriggerWGSL(this.seeding,this.sourceSeeding)}
${narrowBandActivityWGSL}
${narrowBandAirWGSL}
${narrowBandSparseWGSL}
// The same liquid/retained-mass owners used by velocity extrapolation.
// Negative phi at a particle is insufficient: a sub-cell droplet may have
// no resolved cell at all, so the grid cannot evolve its momentum.
fn gridSupported(p:vec3f)->bool{
 let base=vec3i(floor(p-0.5));let fraction=fract(p-0.5);
 let page=vec3i(floor(p))>>vec3u(5u);let home=nbHomeBlock(page,base>>vec3u(5u),(base+1)>>vec3u(5u));
 for(var k=0u;k<8u;k++){
  let bit=umCorner(k,2u);let w=select(vec3f(1)-fraction,fraction,bit!=vec3u(0));
  if(w.x*w.y*w.z<=1e-6){continue;}
  let c=umCellNearFromPage(home,page,base+vec3i(bit));let owner=umOwnerAt(c);
  if(textureLoad(phase,vec3i(umOrigin(owner)),0).x>0.5){return true;}
 }
 return false;
}
// Liquid a source makes this step, in h cells: the minimum of \`value\` and
// the source's own distance. The source writes the level set after advection;
// here it follows the particle union the same way, and decides which cells
// are seeded as at initialization.
fn nbSourcePhi(p:vec3f,value:f32)->f32{
 let h=min(params.hDt.x,min(params.hDt.y,params.hDt.z));return umSourceuvSourcePhi(p,value*h)/h;
}
// Samples the full pool turned away. Optional reseeding is deferred and
// counted: its cell is short again next step. A hole in liquid a description
// makes (a source's, the initial band) is not repaired: the sticky fatal
// state[3] takes 3 (1 a trajectory past the trace limit, 2 a nonfinite velocity).
fn nbPoolFull(mandatory:bool,samples:u32){if(mandatory){atomicStore(&state[3],3u);}else{atomicAdd(&state[2],samples);}}
// A sample at q, in a page of the box that holds no storage: state[3] takes
// 4 and the sticky state[11] the page, 0x80000000 | x | y<<6 | z<<12 in pages
// of 32 cells (the greatest raised). The host stops the run (noteReceipt);
// the sample is not clamped into a stored cell or counted in one.
fn nbArrived(q:vec3f){
 let page=vec3u(q)>>vec3u(5u);atomicStore(&state[3],4u);atomicMax(&state[11],0x80000000u|page.x|(page.y<<6u)|(page.z<<12u));
}
// A sample made at the end of a step joins the tail of the epoch. The next
// move counts, marks and orders every sample, so nothing is filed for it here.
fn store(p:Particle,mandatory:bool){
 let i=atomicAdd(&state[1],1u);if(i>=arrayLength(&particles)){nbPoolFull(mandatory,1u);return;}
 particles[i]=p;
}
// Gather/addition/velocity support reaches neighboring tiles. Erasure heat
// belongs to occupied tiles: dilating it erodes unseeded neighbors.
// A tile a sample's support reaches lies inside the residency closure. One
// that holds no storage takes no mark; over air it sets the sticky closure
// violation, which the next census makes fatal.
fn nbUnmarked(t:vec3i){if(umTileKind(t)==UM_AIR){umSupport[UM_RESIDENCY+1u]=1u;}}
fn markSurfaceTiles(position:vec3f,heat:f32){
 let home=vec3i(vec3u(position)/4u);
 {let tile=umTileKey(home);if(umKeyStored(tile)){atomicMax(&bins[NB_ACTIVITY_SURFACE_HEAT+tile],bitcast<u32>(max(heat,0.0)));}}
 let lo=nbTileNear(home,vec3i(floor((position-2.0)/4.0)));let hi=nbTileNear(home,vec3i(floor((position+2.0)/4.0)));
 for(var z=lo.z;z<=hi.z;z++){for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){
  let tile=umTileKey(vec3i(x,y,z));if(!umKeyStored(tile)){nbUnmarked(vec3i(x,y,z));continue;}
  atomicStore(&bins[NB_SURFACE_TILES+tile],1u);
  atomicMax(&bins[NB_ACTIVITY_THETA+tile],bitcast<u32>(max(heat,0.0)));
 }}}
}
// Sorted cell runs let one tile combine repeated particle support writes.
var<workgroup> nbMarkedMask:atomic<u32>;
var<workgroup> nbMarkedHeat:array<atomic<u32>,27>;
var<workgroup> nbOwnHeat:atomic<u32>;
var<workgroup> nbMarkedPopulation:u32;
var<workgroup> nbCellStart:array<u32,64>;
var<workgroup> nbCellHeat:array<atomic<u32>,64>;
var<workgroup> nbCellLive:array<atomic<u32>,2>;
var<workgroup> nbStray:atomic<u32>;
var<workgroup> nbMarkedTile:u32;
var<workgroup> nbMarkedCount:u32;
// A fixed grid strides the occupied list: an unoccupied tile has no run.
@compute @workgroup_size(64) fn markParticleTiles(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){nbMarkedCount=min(atomicLoad(&bins[NB_OCCUPIED]),UM_TILES);}
 let count=workgroupUniformLoad(&nbMarkedCount);
 for(var j=group.x+groups.x*group.y;j<count;j+=groups.x*groups.y){
  if(lane==0u){nbMarkedTile=atomicLoad(&bins[NB_OCCUPIED+4u+j]);}
  nbMarkTile(workgroupUniformLoad(&nbMarkedTile),lane);
  workgroupBarrier();
 }
}
fn nbMarkTile(tile:u32,lane:u32){
 if(!umTileLive(tile)){return;}
 let first=64u*tile;
 if(lane==0u){nbMarkedPopulation=links[first+63u]+atomicLoad(&bins[first+63u])-links[first];}
 if(workgroupUniformLoad(&nbMarkedPopulation)==0u){return;}
 if(lane==0u){atomicStore(&nbMarkedMask,0u);atomicStore(&nbOwnHeat,0u);atomicStore(&nbStray,0u);atomicStore(&nbCellLive[0],0u);atomicStore(&nbCellLive[1],0u);}
 if(lane<27u){atomicStore(&nbMarkedHeat[lane],0u);}
 // The tile's samples are one sorted run: lane k holds cell k's start.
 let begin=links[first];nbCellStart[lane]=links[first+lane]-begin;atomicStore(&nbCellHeat[lane],0u);
 let population=nbMarkedPopulation;workgroupBarrier();
 let local=umCorner(lane,4u);let coord=vec3i(umTileCoord(tile));
 // A cell's samples reach the tile below it or the one above, by its half of
 // the tile: the wall across a page face holds that reach in the tile.
 let reachLow=coord+vec3i(local/2u)-1;let reachHigh=coord+vec3i(local/2u);
 let low=umTileNearFromPage(tile>>9u,coord>>vec3u(3u),reachLow);let high=umTileNearFromPage(tile>>9u,coord>>vec3u(3u),reachHigh);
 // The samples stride the lanes, not a lane its cell's run: a crowded cell
 // held its whole wave for its run (the long dam's frame 60 has cells of
 // thousands of samples; the per-cell loop cost 1.6 ms there, this 0.13).
 // A sample's cell is the last run starting at or before it; each cell's
 // heat is the maximum of its samples' (nonnegative, so their bits order
 // as they do: max(heat,h) from 0).
 // The run's links still hold the order's keys: this workgroup alone reads
 // them, so it writes each moved record's position over its slot's, the
 // compact sorted positions every later gather reads.
 for(var j=lane;j<population;j+=64u){
  var cell=0u;for(var step=32u;step>0u;step>>=1u){if(nbCellStart[cell+step]<=j){cell+=step;}}
  let a=UM_CELL_KEYS+4u*(begin+j);let p=particles[begin+j].position;let bits=bitcast<vec4u>(p);
  links[a]=bits.x;links[a+1u]=bits.y;links[a+2u]=bits.z;links[a+3u]=bits.w;
  let q=p.xyz;let h=max(0.0,p.w);
  let l=umCorner(cell,4u);let rl=coord+vec3i(l/2u)-1;let rh=coord+vec3i(l/2u);
  // Preserve the original floating-point boundary decisions exactly.
  let lo=vec3i(floor((q-2.0)/4.0));let hi=vec3i(floor((q+2.0)/4.0));
  if(any(lo!=rl)||any(hi!=rh)){atomicStore(&nbStray,1u);continue;}
  atomicMax(&nbCellHeat[cell],select(bitcast<u32>(h),0u,h==0.0));atomicOr(&nbCellLive[cell>>5u],1u<<(cell&31u));
 }
 // A sample whose support leaves its cell's reach marks its own tiles. That
 // is rare, and kept out of the loop above: inlined there, its loops and
 // atomics doubled the loop's cost though no sample took them. The marks are
 // atomic stores and maxima, so their order changes nothing.
 if(workgroupUniformLoad(&nbStray)!=0u){
  for(var j=lane;j<population;j+=64u){
   var cell=0u;for(var step=32u;step>0u;step>>=1u){if(nbCellStart[cell+step]<=j){cell+=step;}}
   let a=UM_CELL_KEYS+4u*(begin+j);let q=bitcast<vec3f>(vec3u(links[a],links[a+1u],links[a+2u]));
   let l=umCorner(cell,4u);let rl=coord+vec3i(l/2u)-1;let rh=coord+vec3i(l/2u);
   let lo=vec3i(floor((q-2.0)/4.0));let hi=vec3i(floor((q+2.0)/4.0));
   if(any(lo!=rl)||any(hi!=rh)){markSurfaceTiles(q,max(0.0,bitcast<f32>(links[a+3u])));}
  }
 }
 let live=((atomicLoad(&nbCellLive[lane>>5u])>>(lane&31u))&1u)!=0u;let heat=bitcast<f32>(atomicLoad(&nbCellHeat[lane]));
 if(live){
  var mask=0u;
  for(var z=low.z;z<=high.z;z++){for(var y=low.y;y<=high.y;y++){for(var x=low.x;x<=high.x;x++){
   let o=vec3u(vec3i(x,y,z)-coord+1);let bit=o.x+3u*(o.y+3u*o.z);mask|=1u<<bit;
   atomicMax(&nbMarkedHeat[bit],bitcast<u32>(heat));
  }}}
  atomicOr(&nbMarkedMask,mask);atomicMax(&nbOwnHeat,bitcast<u32>(heat));
 }
 workgroupBarrier();
 if(lane==0u){atomicMax(&bins[NB_ACTIVITY_SURFACE_HEAT+tile],atomicLoad(&nbOwnHeat));}
 if(lane<27u&&((atomicLoad(&nbMarkedMask)>>lane)&1u)!=0u){
  let t=coord+vec3i(umCorner(lane,3u))-1;let markedTile=umTileKey(t);
  if(umKeyStored(markedTile)){
   atomicStore(&bins[NB_SURFACE_TILES+markedTile],1u);
   atomicMax(&bins[NB_ACTIVITY_THETA+markedTile],atomicLoad(&nbMarkedHeat[lane]));
  }else{nbUnmarked(t);}
 }
}
${narrowBandTraceWGSL("sampleVelocity",this.traceSubsteps)}
override nbAdvectScalarCollision:bool=false;
// With a static solid in the scene the fast launch (umSolidFree) still
// advects a sample whose trace has no closed cell in its box (every stage
// and end it reaches) while the scene holds no rigid body, terrain or
// procedural boundary: every cell is then whole or closed, the walk moves
// only through cells inside that box, and the general launch's collision
// walk blocks nothing there. The proof is the record's open-cell bits
// (umSolidOpenCells) of the box's tiles, all stored. Any other sample is
// listed slow, untouched. nbSolidRouting off restores the general launch
// for every sample.
override nbSolidRouting:bool=true;
var<private> nbHullLo:vec3f;
var<private> nbHullHi:vec3f;
fn nbHullInclude(x:vec3f){nbHullLo=min(nbHullLo,x);nbHullHi=max(nbHullHi,x);}
fn nbFastSolid()->bool{return ${this.solid.coarse?"nbSolidRouting&&round(umSolidParams.boundary.z)<0.5&&!umSolidHasTerrain()&&umBoundaryBase()==0u":"false"};}
fn nbHullUncut()->bool{
 ${this.solid.coarse?/* wgsl */`let a=vec3i(floor(nbHullLo-vec3f(0.001)));let b=vec3i(floor(nbHullHi+vec3f(0.001)));
 let first=a>>vec3u(2u);let last=b>>vec3u(2u);
 if(any(a<vec3i(0))||any(last-first>vec3i(2))){return false;}
 for(var z=first.z;z<=last.z;z++){for(var y=first.y;y<=last.y;y++){for(var x=first.x;x<=last.x;x++){
  let t=vec3i(x,y,z);let key=umTileKey(t);if(!umKeyStored(key)){return false;}
  let open=umSolidOpenCells(key);if(all(open==vec2u(0xffffffffu))){continue;}
  // The box's cells of this tile: bit x+4y+16z.
  let lo=vec3u(max(a-4*t,vec3i(0)));let hi=vec3u(min(b-4*t,vec3i(3)));
  let row=((2u<<hi.x)-1u)&~((1u<<lo.x)-1u);var box=vec2u(0u);
  for(var cz=lo.z;cz<=hi.z;cz++){for(var cy=lo.y;cy<=hi.y;cy++){
   let bits=row<<(4u*cy+16u*(cz&1u));if(cz<2u){box.x|=bits;}else{box.y|=bits;}
  }}
  if(any((open&box)!=box)){return false;}
 }}}
 return true;`:"return false;"}
}
fn nbListSlow(i:u32){let j=atomicAdd(&state[NB_SLOW_PARTICLES],1u);links[UM_CELL_KEYS+j]=i;}
fn nbAdvect(i:u32){
  nbSlow=false;var p=source[i];var q=p.position.xyz;
  let held=bitcast<vec3i>(vec3u(atomicLoad(&state[12u]),atomicLoad(&state[13u]),atomicLoad(&state[14u])));
  q-=32.0*vec3f(umFramePage()-held);
  let routed=nbFastSample&&umSolidPresent();
  if(routed){nbHullLo=q;nbHullHi=q;if(!nbHullUncut()){nbListSlow(i);return;}}
  // Bodies advance at the previous fluid step's tail. Push samples they
  // overtook out to the moving surface before the static-solid rejection.
  let rigidStart=nbRigidContact(q,p.velocity.xyz);q=rigidStart.position;p.velocity=vec4f(rigidStart.velocity,p.velocity.w);
  // Solid collision, rather than grid ownership, limits particle motion.
  if(nbParticleBlocked(q)){source[i].position.x=-1.0;return;}
  let ballistic=nbParticleMode(p)==1.0||!gridSupported(q);p.before.w=nbParticleState(nbParticleAge(p),select(0.0,1.0,ballistic));
  ${this.apic?"if(ballistic){p.cx=vec4f(0);p.cy=vec4f(0);p.cz=vec4f(0);}":""}
  // A sample takes erasure heat (over 1) at the step's end (update), with its
  // tile's new collar seeded: a collar tile the census triggered eroded the
  // unseeded tiles beside it that same step.
  let tileHeat=nbTargetHeat(q);
  p.position.w=max(select(min(tileHeat,1.0),tileHeat,p.position.w>1.0),max(0.0,p.position.w-select(nbCooling()*params.hDt.w,0.0,ballistic)));
  if(p.position.w<=0.0){source[i].position.x=-1.0;return;}
  // Uniform stores velocity half a step ahead of position. Ballistic drift
  // uses that midpoint velocity; update supplies the next full gravity kick.
  // nbTraceSpan's steps, every field sample of the trace at one call site
  // (Metal inlines each call of the field): stage 0 the field at q (k1; the
  // first sets the substeps), 1 one base substep ahead (k2), 2 the third
  // point, after which the span ends. Ballistic drift samples nothing.
  let h=params.hDt.xyz;
  var steps=1u;var base=0.0;var s=0u;var x=q;var stage=0u;var k1=vec3f(0.0);var k2=vec3f(0.0);var two=false;var probe=false;var travel=1u;
  for(var n=0u;n<${3*NARROW_BAND_TRACE_LIMIT+3}u;n++){
   var end=q;var units=1u;
   if(!ballistic){
    if(routed){nbHullInclude(x);}
    let k=sampleVelocity(x)/h;
    // Nothing of this sample is stored yet: the general launch redoes it.
    if(nbFastSample&&nbSlow){nbListSlow(i);return;}
    if(stage==0u){
     k1=k;
     if(s==0u){
      steps=nbTraceSteps(k1,params.hDt.w);
      // Trajectory refinement leaves the global pressure timestep unchanged.
      if(steps>${NARROW_BAND_TRACE_LIMIT}u){atomicStore(&state[3],1u);source[i].position.x=-1.0;return;}
      travel=steps;if(NB_TRACE_GRADIENT&&steps>1u){steps=1u;probe=true;}
      base=params.hDt.w/f32(steps);
     }
     x=q+base*k1;stage=1u;continue;
    }
    if(stage==1u){
     if(probe){probe=false;let m=nbTraceGradientSteps(k1,k,params.hDt.w,travel);if(m>1u){steps=m;base=params.hDt.w/f32(m);x=q+base*k1;continue;}}
     k2=k;two=nbTraceTwo(base,k1,k2,steps-s);x=nbTraceThird(q,base,k1,k2,two);stage=2u;continue;
    }
    let span=nbTraceEnd(q,base,k1,k2,k,two);end=span.xyz;units=u32(span.w);stage=0u;
   }else{
    if(s==0u){
     steps=nbTraceSteps(p.velocity.xyz/h,params.hDt.w);
     if(steps>${NARROW_BAND_TRACE_LIMIT}u){atomicStore(&state[3],1u);source[i].position.x=-1.0;return;}
     // Drift is straight: gradient substeps take only the 32-cell spans.
     if(NB_TRACE_GRADIENT){steps=(steps+31u)/32u;}
     base=params.hDt.w/f32(steps);
    }
    end=q+base*p.velocity.xyz/h;
   }
   // A SOLID page across a face of the page the span starts in stops the
   // end 0.01 cell inside that face. An AIR page or a stored one does not.
   let endpoint=nbPointNearIn(vec3i(floor(q)),end,0.01);
   if(routed){nbHullInclude(endpoint);}
   // With no embedded solids the segment stays inside the convex container
   // after endpoint clamping. There are no voxel crossings to test; retaining
   // the RK3 endpoint also avoids accumulating rounded collision increments.
   if(!umSolidEnabled()){q=endpoint;}else{
   let travel=endpoint-q;let walk=max(1u,u32(ceil(2.0*max(abs(travel.x),max(abs(travel.y),abs(travel.z))))));
   if(walk>256u){atomicStore(&state[3],1u);break;}
   // A voxel face stops a particle as a container wall does: the crossing
   // axis ends at the face and loses its velocity, the others carry on.
   // A floor is the exception where its tread stands for a sloped bed
   // (umBedSlope): the sample then loses its velocity along that bed's
   // normal, and so keeps the part of its fall that runs down the slope. It
   // gains nothing along an axis a wall has stopped, so it rests in a corner.
   let stride=travel/f32(walk);var landed=false;var floorCell=vec3i(0);var free=vec3f(1);
   for(var j=1u;j<=walk;j++){
    let start=q;
    // A step at most half a cell an axis: bodies matter only where a present
    // body's pose lies within two cells of the start's 16h region (the head's
    // body region mask). Elsewhere the voxel, terrain and boundary walls
    // alone can block it.
    let bodyNear=umBodyCount()>0u&&umBodyPoseNearTile(umTileKey(vec3i(floor(start))>>vec3u(2u)),2u);
    for(var axis=0u;axis<3u;axis++){
     var point=q;point[axis]+=stride[axis];
     var blocked:bool;
     if(nbAdvectScalarCollision){
      // Diagnostic oracle: WGSL select evaluates both values, including the
      // expensive fractional open-cell query that a boundary cell discards.
      blocked=select(umCellOpen(vec3i(point))<0.5,
       nbVoxelBlocked(point),umSolidHasTerrain()||bodyNear||umBoundaryRecord(vec3i(floor(point)))!=0u);
     }else if(umSolidHasTerrain()||bodyNear||umBoundaryRecord(vec3i(floor(point)))!=0u){
      blocked=nbVoxelBlocked(point);
     }else{blocked=umCellOpen(vec3i(point))<0.5;}
     if(blocked){
      let cell=floor(q[axis]);point[axis]=select(cell+0.01,cell+0.99,stride[axis]>0.0);
      if(axis==1u&&stride.y<0.0){landed=true;floorCell=vec3i(q);}else{p.velocity[axis]=0.0;free[axis]=0.0;}
     }
     q=point;
    }
    if(bodyNear){
     let rigidContact=nbRigidContact(q,p.velocity.xyz);
     if(!nbVoxelBlocked(rigidContact.position)){q=rigidContact.position;p.velocity=vec4f(rigidContact.velocity,p.velocity.w);}
    }
    let boundaryContact=nbProceduralContact(start,q,p.velocity.xyz);
    if(!nbVoxelBlocked(boundaryContact.position)&&(!bodyNear||umBodyAt(umSolidTraceWorld(boundaryContact.position))<0)){q=boundaryContact.position;p.velocity=vec4f(boundaryContact.velocity*free,p.velocity.w);}
    if(umSolidHasTerrain()){
     let contact=nbTerrainContact(start,q,p.velocity.xyz);
     // A normal correction must not cross an authored voxel or a body.
     if(!nbVoxelBlocked(contact.position)&&(!bodyNear||umBodyAt(umSolidTraceWorld(contact.position))<0)){q=contact.position;}
     p.velocity=vec4f(contact.velocity*free,p.velocity.w);
    }
   }
   if(landed){
    let slope=umBedSlope(floorCell)*params.hDt.y/params.hDt.xz;
    if(all(slope==vec2f(0))){p.velocity.y=0.0;}else{
     let normal=normalize(vec3f(-slope.x,1.0,-slope.y));
     p.velocity=vec4f((p.velocity.xyz-dot(p.velocity.xyz,normal)*normal)*free,p.velocity.w);
    }
   }
   }
   for(var axis=0u;axis<3u;axis++){if(endpoint[axis]!=end[axis]){p.velocity[axis]=0.0;}}
   s+=units;if(s>=steps){break;}
   x=q;
  }
  if(routed&&!nbHullUncut()){nbListSlow(i);return;}
  // A sample that ends in a page that holds no storage is fatal: it keeps
  // its end, in no cell's count, and the order that follows files it in no run.
  if(!umPointPresent(q)){
   nbArrived(q);p.position=vec4f(q,p.position.w);source[i]=p;return;
  }
  // The sample moves where it is stored: the spatial order that follows
  // reads every slot, drops the ones marked here (a negative x) and takes
  // each cell's count from the bins. Nothing gathers before it.
  // A trajectory can cross from h into a coarse owner during this step.
  // Erasure heat stays with the tile that has it. A hot tile's collar is
  // seeded; a collar's own neighbours need not be, and a grid-supported
  // sample that carried its heat into the collar eroded the unseeded liquid
  // beside it. Spray keeps its heat: nothing else erases what it leaves.
  var heat=p.position.w;
  if(!ballistic){let arrived=nbTargetHeat(q);if(arrived>0.0){heat=min(heat,max(arrived,1.0));}}
  p.position=vec4f(q,heat);source[i]=p;
  atomicAdd(&bins[umCellAt(vec3u(q))],1u);nbCover(umTileAt(vec3u(q)/4u),1u);
}
// A launch over the samples is one fixed row (NARROW_BAND_PARTICLE_GRID): a
// lane takes a sample of its group's 64 a turn and strides the row through
// the count the receipt holds now.
@compute @workgroup_size(64) fn advectParticles(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 // The fast launch is compiled solid-free: with a solid in the scene the
 // general launch takes every sample, unless the fast launch may route them
 // by the record's uncut regions (nbFastSolid).
 if(nbFastSample&&umSolidPresent()&&!nbFastSolid()){return;}
 let n=min(atomicLoad(&state[0]),arrayLength(&source));
 for(var i=gid.x;i<n;i+=64u*groups.x){nbAdvect(i);}
}
// The samples the fast launch listed (state word NB_SLOW_PARTICLES, cleared
// with the step's words), by the general sampler and solid code: all of
// them while a solid is in the scene (the fast launch then listed none).
// The list borrows the position bank, which is dead from update's
// compaction until the order after this pass rewrites it.
const NB_SLOW_PARTICLES:u32=10u;
@compute @workgroup_size(64) fn advectParticlesDeferred(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 // One nbAdvect site (Metal inlines every call).
 let all=umSolidPresent()&&!nbFastSolid();
 let n=min(atomicLoad(&state[select(NB_SLOW_PARTICLES,0u,all)]),arrayLength(&source));
 for(var j=gid.x;j<n;j+=64u*groups.x){nbAdvect(select(links[UM_CELL_KEYS+j],j,all));}
}
// End-of-step sampling uses the reconstructed surface and projected velocity.
// Protect the outer h from crowding; sparse samples have a separate lifetime.
fn nbKept(i:u32)->bool{
 // The depth update just stored, of this position in this band.
 let depth=source[i].velocity.w;let order=umCellAt(vec3u(source[i].position.xyz));
 // Seeding tops a cell up to its count: the samples it keeps and the crowded ones it sheds.
 if(depth<-4.0||source[i].position.w<=0.0||nbSparseExpired(nbParticleAge(source[i]))){atomicSub(&bins[order],1u);return false;}
 // Preserve surface samples even when compressed. Thin structures cannot
 // survive an arbitrary per-cell cap. Only deep overlap samples retire: the
 // source is still in cell order, so a sample's place in its run is its rank.
 return depth>=-1.0||i-links[order]<16u;
}
// Compact positions/motions expire after update. Borrow that part of links
// for per-particle ranks and per-workgroup counts/offsets; cell starts remain
// intact while nbKept uses the sorted rank to retain the overlap samples.
fn nbCompactCounts()->u32{return UM_CELL_KEYS+arrayLength(&source);}
fn nbCompactOffsets()->u32{return nbCompactCounts()+(arrayLength(&source)+63u)/64u;}
var<workgroup> nbCompactScan:array<u32,256>;
// resampleCount is a job a 64-sample block, run by a workgroup together:
// its row strides the blocks (the entries rewrite that ends this source).
${subgroups?/* wgsl */`
// One subgroup owns a 64-particle block, two consecutive samples per lane.
// Its integer scan gives the same stable ranks without shared-memory rounds.
@compute @workgroup_size(32) fn resampleCount(@builtin(workgroup_id) group:vec3u,@builtin(subgroup_invocation_id) lane:u32){
 let job=group.x;let i=64u*job+2u*lane;
 let n=min(atomicLoad(&state[0]),arrayLength(&source));
 var a=0u;var b=0u;if(i<n){a=u32(nbKept(i));}if(i+1u<n){b=u32(nbKept(i+1u));}
 let rank=subgroupExclusiveAdd(a+b);let count=subgroupAdd(a+b);
 if(i<n){links[UM_CELL_KEYS+i]=select(0xffffffffu,rank,a!=0u);}
 if(i+1u<n){links[UM_CELL_KEYS+i+1u]=select(0xffffffffu,rank+a,b!=0u);}
 if(lane==0u&&i<n){links[nbCompactCounts()+job]=count;}
}
`:/* wgsl */`
@compute @workgroup_size(64) fn resampleCount(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 let i=gid.x;let n=min(atomicLoad(&state[0]),arrayLength(&source));
 var kept=0u;if(i<n){kept=u32(nbKept(i));}
 nbCompactScan[lane]=kept;workgroupBarrier();
 for(var stride=1u;stride<64u;stride*=2u){
  var add=0u;if(lane>=stride){add=nbCompactScan[lane-stride];}
  workgroupBarrier();nbCompactScan[lane]+=add;workgroupBarrier();
 }
 if(i<n){links[UM_CELL_KEYS+i]=select(0xffffffffu,nbCompactScan[lane]-kept,kept!=0u);}
 if(lane==0u&&i<n){links[nbCompactCounts()+i/64u]=nbCompactScan[63];}
}
`}
// One small scan of block counts, not a second particle sort. Each lane
// sums its contiguous range before the workgroup scan, then writes offsets.
@compute @workgroup_size(256) fn resamplePrefix(@builtin(local_invocation_index) lane:u32){
 if(lane==0u){atomicStore(&bins[NB_BAND+3u],0u);}
 let blocks=(min(atomicLoad(&state[0]),arrayLength(&source))+63u)/64u;
 let span=(blocks+255u)/256u;let first=lane*span;let end=min(blocks,first+span);
 var total=0u;for(var b=first;b<end;b++){total+=links[nbCompactCounts()+b];}
 nbCompactScan[lane]=total;workgroupBarrier();
 for(var stride=1u;stride<256u;stride*=2u){
  var add=0u;if(lane>=stride){add=nbCompactScan[lane-stride];}
  workgroupBarrier();nbCompactScan[lane]+=add;workgroupBarrier();
 }
 var offset=nbCompactScan[lane]-total;
 for(var b=first;b<end;b++){links[nbCompactOffsets()+b]=offset;offset+=links[nbCompactCounts()+b];}
 if(lane==255u){atomicStore(&state[1],nbCompactScan[255]);}
}
@compute @workgroup_size(64) fn resample(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 let n=min(atomicLoad(&state[0]),arrayLength(&source));
 for(var i=gid.x;i<n;i+=64u*groups.x){
  let rank=links[UM_CELL_KEYS+i];if(rank==0xffffffffu){continue;}
  particles[links[nbCompactOffsets()+i/64u]+rank]=source[i];
 }
}
// Fit the outer lattice to the sphere reconstruction's nodal interface.
// Keep tangential quarter-cell positions: their known offset determines the
// seed depth and preserves a resting plane without any volume correction.
${narrowBandSeedShellWGSL}
fn seedSite(c:vec3u,k:u32,initial:bool,activated:bool)->vec4f{
  let q=vec3f(c)+0.25+0.5*vec3f(umCorner(k,2u));let distance=particleDepth(q);
  if(distance>0.0||distance < -4.0||nbParticleBlocked(q)){return vec4f(-1,0,0,0);}
  // Paper Section 3.3: replenish only the inner [-R,-h] band. The outer
  // layer keeps its transported samples instead of being recreated; a change
  // of ownership or an uncovered tile cannot bootstrap one from the bulk
  // field. Section 3.2: liquid made from a description gets its whole band,
  // at initialization and where a source has just made it.
  if(!initial&&!activated&&distance>-1.0&&nbSourcePhi(q,1.0)>=0.0){return vec4f(-1,0,0,0);}
  // The heat-one overlap supplies support without authority to erase
  // liquid. Its outer shell only seeds on resolved fine owners.
  if(distance>-1.0&&umOwnerAt(vec3i(c)).width!=1u){return vec4f(-1,0,0,0);}
  if(distance<=-NB_SURFACE_RADIUS){return vec4f(q,distance);}
  let shell=seedShell(q,distance);let p=shell.xyz;
  if(shell.w>0.0||nbParticleBlocked(p)){return vec4f(-1,0,0,0);}
  // A sample fitted into a page with no storage is the arrival fatal, and is not made.
  if(!umPointPresent(p)){nbArrived(p);return vec4f(-1,0,0,0);}
  // A shell sample can land in another cell, whose own seeding decision
  // does not see it: no count changes here, whatever the dispatch order.
  // The shell's depth is the depth at its point.
  return shell;
}
fn seedCell(c:vec3u,emit:bool){
 // No tile past the reach is within six cells of the surface.
 if((nbBandReach(c)&1u)==0u){return;}
 let heat=nbTargetHeat(vec3f(c)+0.5);if(heat<=0.0){return;}
 if(!nbParticleCellOpen(vec3i(c))){return;}
 let d=particleDepth(vec3f(c)+0.5);if(d>0.9||d < -4.5){return;}
 let activated=atomicLoad(&bins[NB_ACTIVITY_NEW+nbBandTile(c)])!=0u;
 var seeds:array<vec4f,8>;var seedCount=0u;
 for(var k=0u;k<8u;k++){
  let site=seedSite(c,k,true,activated);if(site.x<0.0){continue;}
  seeds[seedCount]=site;seedCount++;
 }
 // Count first, then use the cell prefix. GPU scheduling cannot change
 // initial particle slots or which cells fit if the pool is exhausted.
 if(seedCount==0u){return;}
 let cell=umCellAt(c);
 if(!emit){atomicStore(&bins[cell],seedCount);nbCover(umTileAt(c/4u),seedCount);return;}
 let first=links[cell];let capacity=arrayLength(&particles);
 let kept=min(seedCount,capacity-min(first,capacity));
 if(kept<seedCount){nbPoolFull(true,seedCount-kept);}
 for(var k=0u;k<kept;k++){
  let q=seeds[k];
  particles[first+k]=nbSeedParticle(q,heat);
 }
}
// Seed the geometric particle band, including its overlap on 4h owners: one
// workgroup for each tile of the band search's list, the only ones seedCell
// passes.
fn nbSeedCell(group:vec3u,lane:u32)->vec3u{
 let tile=atomicLoad(&bins[NB_BAND_SEARCH+group.x]);
 return 4u*umTileCoord(tile)+vec3u(lane%4u,(lane/4u)%4u,lane/16u);
}
// Reserve the remaining slots for new liquid before optional overlap
// reseeding. A single racing dispatch lets the growing pool exhaust the
// budget before the nozzle is visited, punching holes in a continuous jet.
// One cell of source distance encloses every quarter-cell seed; the same
// predicate partitions both passes so a source cell is never seeded twice.
fn nbSourceCell(c:vec3u)->bool{return nbSourcePhi(vec3f(c)+0.5,1.0)<1.0;}
// Compaction has consumed the old cell starts. Reuse that table as a list
// of cells needing candidates, one entry per possible cell at most.
@compute @workgroup_size(64) fn seedCells(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let c=nbSeedCell(group,lane);
 if((nbBandReach(c)&1u)==0u){return;}
 let heat=nbTargetHeat(vec3f(c)+0.5);if(heat<=0.0){return;}
 if(atomicLoad(&bins[umCellAt(c)])>=8u||!nbParticleCellOpen(vec3i(c))){return;}
 let depth=particleDepth(vec3f(c)+0.5);if(depth>0.9||depth < -4.5){return;}
 let slot=atomicAdd(&bins[NB_BAND+3u],1u);
 links[slot]=umCellAt(c)|select(0u,0x80000000u,nbSourceCell(c));
}
fn seedCandidate(group:vec3u,lane:u32,sourcePass:bool)->Particle{
 var empty=Particle();empty.position.x=-1.0;
 let slot=8u*group.x+lane/8u;if(slot>=atomicLoad(&bins[NB_BAND+3u])){return empty;}
 let entry=links[slot];if(((entry&0x80000000u)!=0u)!=sourcePass){return empty;}
 let c=umCellCoord(entry&0x7fffffffu);let k=lane%8u;
 if(k<atomicLoad(&bins[umCellAt(c)])){return empty;}
 let activated=atomicLoad(&bins[NB_ACTIVITY_NEW+nbBandTile(c)])!=0u;
 let site=seedSite(c,k,false,activated);if(site.x<0.0){return empty;}
 let heat=nbTargetHeat(vec3f(c)+0.5);
 return nbSeedParticle(site,heat);
}
fn storeSeedCandidate(p:Particle,lane:u32,mandatory:bool){
 ${subgroups?/* wgsl */`
 let valid=p.position.x>=0.0;let rank=subgroupExclusiveAdd(u32(valid));let count=subgroupAdd(u32(valid));var first=0u;
 if(lane==0u&&count!=0u){first=atomicAdd(&state[1],count);}first=subgroupBroadcastFirst(first);
 if(valid){let at=first+rank;if(at<arrayLength(&particles)){particles[at]=p;}else{nbPoolFull(mandatory,1u);}}
 `:"if(p.position.x>=0.0){store(p,mandatory);}"}
}
@compute @workgroup_size(64) fn seedSources(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32${subgroups?",@builtin(subgroup_invocation_id) subgroupLane:u32":""}){
 storeSeedCandidate(seedCandidate(group,lane,true),${subgroups?"subgroupLane":"0u"},true);
}
@compute @workgroup_size(64) fn seed(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32${subgroups?",@builtin(subgroup_invocation_id) subgroupLane:u32":""}){
 storeSeedCandidate(seedCandidate(group,lane,false),${subgroups?"subgroupLane":"0u"},false);
}

@compute @workgroup_size(64) fn countInitial(@builtin(global_invocation_id) gid:vec3u){
 for(var i=gid.x;i<UM_CELL_KEYS;i+=65536u){if(!umCellLive(i)){continue;}seedCell(umCellCoord(i),false);}
}
@compute @workgroup_size(64) fn seedInitial(@builtin(global_invocation_id) gid:vec3u){
 ${this.order.sparse?"":"if(gid.x==0u){atomicStore(&state[1],links[UM_CELL_KEYS-1u]+atomicLoad(&bins[UM_CELL_KEYS-1u]));}"}
 for(var i=gid.x;i<UM_CELL_KEYS;i+=65536u){if(!umCellLive(i)){continue;}let c=umCellCoord(i);seedCell(c,true);}
}
fn weight(x:f32)->f32{let a=abs(x);if(a<0.5){return 0.75-a*a;}let b=max(0.0,1.5-a);return 0.5*b*b;}
// Also the receipt's census of the samples: the unsupported count always;
// against the surface they made (the largest distance outside it, the
// samples past half a cell outside, the samples deeper than four) only while
// traces are measured (settings.w): it is diagnostics, and its two surface
// queries a sample were most of this pass.
@compute @workgroup_size(64) fn classify(@builtin(global_invocation_id) gid:vec3u){
 var outside=0.0;var counts=vec3u(0);let census=params.settings.w!=0.0;
 if(gid.x==0u){atomicStore(&bins[NB_BAND+3u],0u);atomicStore(&state[9],0u);}
 ${this.particleDeposit==="tiled"?`// The tiled deposit's fine accumulators (transferDeposit).
 if(nbTransferFits()){for(var j=gid.x;j<384u*umCounts.x;j+=65536u){atomicStore(&bins[UM_CELL_KEYS+j],0u);}}`:""}
 for(var i=gid.x;i<min(atomicLoad(&state[1]),arrayLength(&particles));i+=65536u){
  let q=particles[i].position.xyz;let supported=gridSupported(q);counts.y+=u32(!supported);
  if(census){let d=bandPhi(q);outside=max(outside,d);counts+=vec3u(u32(d>0.5),0u,u32(particleDepth(q)<-4.0));}
  particles[i].before.w=nbParticleState(nbParticleAge(particles[i]),select(1.0,0.0,supported));
  nbStoreMotion(i,vec4f(particles[i].velocity.xyz,nbParticleMode(particles[i])));
 }
 atomicMax(&state[5],bitcast<u32>(outside));
 for(var k=0u;k<3u;k++){if(counts[k]!=0u){atomicAdd(&state[6u+k],counts[k]);}}
}
fn transferFace(face:UMFace)->f32{
 let original=umLoadMixedFace(face.anchor,face.axis);
 // A face beside a frozen cell keeps the departure (F there).
 if(face.width==0u||face.anchor[face.axis]<0||face.neighbor.width==0u||umFrozenFace(face.anchor,face.axis,face.width)){return original;}
 let q=umFaceCenter(face);let depth=particleDepth(q);let blend=select(0.0,1.0,depth>=-2.0);
 if(blend==0.0||depth>1.5){return original;}
 // Cells starting exactly at q+1.5 have zero kernel weight throughout.
 let width=1.0;let home=vec3i(floor(q));let lo=nbCellNear(home,vec3i(floor(q-1.5*width)));let hi=nbCellNear(home,vec3i(ceil(q+1.5*width))-1);
 var momentum=0.0;var total=0.0;
 for(var z=lo.z;z<=hi.z;z++){for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){
  let run=nbCellRun(vec3i(x,y,z));
  for(var i=run.x;i<run.y;i++){
   let motion=nbMotion(i);if(motion.w==1.0){continue;}
   let r=(q-nbPosition(i))/width;let w=weight(r.x)*weight(r.y)*weight(r.z);
   var u=motion[face.axis];
   ${this.apic?"let p=particles[i];var row=p.cx.xyz;if(face.axis==1u){row=p.cy.xyz;}if(face.axis==2u){row=p.cz.xyz;}u+=dot(row,r);":""}
   total+=w;momentum+=w*u;
  }
 }}}
 if(total<1e-5){return original;}return mix(original,momentum/total+umThawFaceTerm(face.anchor,face.axis,face.width),blend*nbTransferBlend(q,depth,total));
}
${narrowBandTransferPropertiesSource(subgroups,this.particleDeposit==="tiled")}
${narrowBandFineTransferSource(subgroups,this.apic)}
${this.particleDeposit==="tiled"?particleBrickDepositWGSL(this.apic):""}
// The 4h owners, one a lane. An owner whose whole stencil is 4h has one
// patch a face, and a face centred in a tile past the reach lies more than
// two cells from the surface: it keeps its velocity without a gather. The
// rest are listed for transferSeams. Negative walls are untouched: their
// boundary condition is Uniform's.
@compute @workgroup_size(64) fn transferOwners(@builtin(global_invocation_id) gid:vec3u){
 let owner=umAllOwner(vec3u(umCounts.x*64u+gid.x,0,0));if(owner.width==0u){return;}
 let origin=umOrigin(owner);
 // A frozen owner keeps every texel (UniformColdFreeze): f.velocity holds F.
 if(umFrozenCell(vec3i(origin))){return;}
 // No live sample: every owner keeps its velocity.
 var far=!nbParticlesLive();
 if(!far){
  far=umTileMaximumWidth(owner.tile)==4u&&umTileMinimumWidth(owner.tile)==4u;
  for(var axis=0u;axis<3u;axis++){
   var c=origin+2u;c[axis]+=2u;c=vec3u(umCellNearFromPage(owner.tile>>9u,vec3i(origin>>vec3u(5u)),vec3i(c)));far=far&&(nbBandReach(c)&1u)==0u;
  }
 }
 if(!far){atomicStore(&bins[NB_BAND_SEAM+atomicAdd(&bins[NB_BAND+3u],1u)],owner.tile);return;}
 for(var axis=0u;axis<3u;axis++){
  var anchor=vec3i(origin);anchor[axis]+=3;let original=textureLoad(velocity,anchor,0);
  var value=vec4f(0,0,0,original.w);value[axis]=original[axis];textureStore(output,anchor,value);
 }
}
// A listed 4h owner a workgroup, a lane for each patch of each positive face:
// one beside h cells has sixteen a face, each with its own gather.
var<workgroup> nbSeam:array<vec2f,64>;
@compute @workgroup_size(64) fn transferSeams(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=atomicLoad(&bins[NB_BAND_SEAM+group.x]);let owner=UMOwner(tile,0u,4u,umTopology[tile]&0x3fffffffu);
 let part=lane/3u;let axis=lane%3u;var anchor=vec3i(0);var result=vec2f(0);
 if(lane<48u){let face=umFace(owner,axis,1,part);if(face.width!=0u){anchor=face.anchor;result=vec2f(transferFace(face),1);}}
 nbSeam[lane]=result;workgroupBarrier();
 if(result.y==0.0){return;}
 // One writer a texel: the lowest axis with a patch at this anchor.
 let offset=anchor-vec3i(umOrigin(owner));var value=vec4f(0);var writer=true;
 for(var other=0u;other<3u;other++){
  if(other==axis){value[other]=result.x;continue;}
  let face=umPositiveFaceAtAnchor(owner,other,anchor);if(face.width==0u){continue;}
  if(other<axis){writer=false;}
  let u=(other+1u)%3u;let v=(other+2u)%3u;let side=4u/face.width;
  value[other]=nbSeam[3u*(u32(offset[u])/face.width+side*(u32(offset[v])/face.width))+other].x;
 }
 if(writer){value.w=textureLoad(velocity,anchor,0).w;textureStore(output,anchor,value);}
}

// A COLD region's samples take no FLIP delta (nbUpdate), so the old payload
// is published only around non-COLD regions (UniformMixedExtension state 3)
// and sampled only by their samples.
fn nbSampleCold(q:vec3f)->bool{return nbTileCold(umTileKey(vec3i(floor(q))>>vec3u(2u)));}
@compute @workgroup_size(64) fn snapshot(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 let n=min(atomicLoad(&state[1]),arrayLength(&particles));
 for(var i=gid.x;i<n;i+=64u*groups.x){
  if(nbSampleCold(particles[i].position.xyz)){continue;}
  let mode=nbParticleMode(particles[i]);var v=vec3f(0);if(mode!=1.0){v=sampleVelocity(particles[i].position.xyz);}
  particles[i].before=vec4f(v,particles[i].before.w);
 }
}
fn nbUpdate(i:u32){
  let mode=nbParticleMode(particles[i]);
  var pic=vec3f(0);var v=vec3f(0);
  if(mode!=1.0){
   ${this.apic?"particles[i]=nbApicSample(particles[i]);v=particles[i].velocity.xyz;pic=sampleVelocity(particles[i].position.xyz);":"pic=sampleVelocity(particles[i].position.xyz);var flip=particles[i].velocity.xyz;if(!nbSampleCold(particles[i].position.xyz)){flip+=pic-particles[i].before.xyz;}v=mix(pic,flip,params.settings.x);"}
  }
  if(mode==1.0){
   ${this.apic?"particles[i].cx=vec4f(0);particles[i].cy=vec4f(0);particles[i].cz=vec4f(0);":""}
   v=particles[i].velocity.xyz+vec3f(0,params.settings.z*params.hDt.w,0);
   let q=particles[i].position.xyz;
   // A sample within 0.011 of a cell of a solid page it moves toward loses
   // that axis: one kind an axis, of the cell ahead.
   for(var axis=0u;axis<3u;axis++){
    var ahead=q;ahead[axis]+=select(-0.011,0.011,v[axis]>0.0);
    if((v[axis]<0.0||v[axis]>0.0)&&umPointKind(ahead)==UM_SOLID){v[axis]=0.0;}
   }
  }
  ${this.apic?"if(!all(abs(particles[i].cx.xyz)<vec3f(1e10))||!all(abs(particles[i].cy.xyz)<vec3f(1e10))||!all(abs(particles[i].cz.xyz)<vec3f(1e10))){atomicStore(&state[3],2u);return;}":""}
  if(!all(abs(v)<vec3f(1e10))){atomicStore(&state[3],2u);return;}
  ${this.apic?"particles[i].lifetime.x=nbSparseAge(i,particles[i].position.xyz,nbParticleAge(particles[i]));":"particles[i].before.w=nbParticleState(nbSparseAge(i,particles[i].position.xyz,nbParticleAge(particles[i])),mode);"}
  particles[i].velocity=vec4f(v,particleDepth(particles[i].position.xyz));
  particles[i].position.w=max(particles[i].position.w,nbTargetHeat(particles[i].position.xyz));
  // Retain accepted grid motion for next frame's residency prediction.
  // FLIP residual velocity transfers momentum, but does not trace positions.
  particles[i].before=vec4f(pic,particles[i].before.w);
}
@compute @workgroup_size(64) fn update(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 let n=min(atomicLoad(&state[1]),arrayLength(&particles));
 if(all(gid==vec3u(0))){atomicStore(&state[0],n);atomicStore(&state[1],n);}
 for(var i=gid.x;i<n;i+=64u*groups.x){nbUpdate(i);}
}
// A lane reduces its own samples and touches the shared receipt once.
@compute @workgroup_size(64) fn diagnoseBefore(@builtin(global_invocation_id) gid:vec3u){
 var outside=0.0;
 for(var i=gid.x;i<min(atomicLoad(&state[1]),arrayLength(&particles));i+=65536u){
  outside=max(outside,bandPhi(particles[i].position.xyz));
 }
 atomicMax(&state[4],bitcast<u32>(outside));
}
@compute @workgroup_size(64) fn commitCount(@builtin(global_invocation_id) gid:vec3u){if(gid.x==0u){let n=min(atomicLoad(&state[1]),arrayLength(&particles));atomicStore(&state[0],n);atomicStore(&state[1],n);let f=bitcast<vec3u>(umFramePage());for(var a=0u;a<3u;a++){atomicStore(&state[12u+a],f[a]);}}}
${narrowBandSurfaceWGSL}

`,["resampleCount"],"(min(atomicLoad(&state[0]),arrayLength(&source))+63u)/64u");
  // Advection borrows the exact extended-field unit taps Uniform already
  // prepared; snapshot, update and seed fill their own from the field they
  // sample (stageTaps). The other stages read their own fresh 4h cache and
  // resolve fine seam taps in place.
  const modules=[false,true].map(unitTaps=>uniformDetailModule(this.device,{specializeWorldGeometry:true,label:`Uniform narrow-band FLIP ${unitTaps?"extended":"stage"}`,code:uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(sourceCode(unitTaps),narrowBandNearestEntries,"(atomicLoad(&bins[NB_BAND+1u])+3u)/4u"),["surfaceSplat"],"atomicLoad(&bins[NB_BAND+3u])"),["depthSeeds"],"atomicLoad(&bins[NB_BAND])"),["seed","seedSources"],"(atomicLoad(&bins[NB_BAND+3u])+7u)/8u"),["depthGeometry"],"atomicLoad(&bins[NB_BAND+1u])"),["buildDistance"],"atomicLoad(&bins[NB_BAND+2u])"),["transferProperties"],"umCounts.x"),["transfer"],subgroups?"select(16u*umCounts.x,(atomicLoad(&state[9])+1u)/2u,nbTransferPropertiesFit())":"select(8u*umCounts.x,(atomicLoad(&state[9])+3u)/4u,nbTransferPropertiesFit())"),["transferOwners"],"(umCounts.y+63u)/64u"),["transferSeams"],"atomicLoad(&bins[NB_BAND+3u])"),["couple"],"(umCounts.y+63u)/64u"),["coupleFine"],"umCounts.x"),["redistanceFine"],"umCounts.x"),["seedCells"],"atomicLoad(&bins[NB_BAND+1u])")}));
  for(const module of modules){
   const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
   if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  }
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources,this.solid.tileLayout]});
  await Promise.all(["activity","activitySpread","advectParticles","advectParticlesDeferred","markParticleTiles","surfaceClear","surfaceCells","surfaceSplat","surfaceSplatOccupied","seedCells","seed","seedSources","countInitial","seedInitial","classify","transferProperties","transfer","transferOwners","transferSeams","snapshot","update","couple","coupleFine","commitCount","resampleCount","resamplePrefix","resample","diagnoseBefore","buildDistance","depthPlanSeed","depthPlanExpand","depthTiles","depthSeeds","depthLists",...narrowBandNearestControlEntries,...narrowBandNearestEntries,"depthReuseSelect@withX","depthTiles@noplanes","depthSeeds@noplanes","depthReuseSnapshot@noplanes","depthGeometry","depthCoarse","redistanceFine","redistanceCoarse","airClearCoarse@planned","airPlanSeeds","airPlanPropagate","airPlanSeeds@fused","airPlanPropagate@fused","airSeeds","airSeedsFine","airSpread0","airSpread1","airSpread2","airGate","airClearFine","airClearCoarse",...NB_H_TWIN_ENTRIES.map(entry=>`${entry}@h`),...NB_FREE_TWIN_ENTRIES.map(entry=>`${entry}@free`),this.particleDeposit==="tiled"?"transferDeposit":"transfer@fit"].map(async name=>{
   const entryPoint=name.split("@")[0]!,module=modules[["advectParticles","advectParticlesDeferred","snapshot","update","seed","seedSources"].includes(entryPoint)?1:0]!;
   const p=await uniformMixedSolidPipeline(this.solid,s=>uniformDetailPipeline(this.device,this.ownership,{label:`Narrow-band FLIP ${entryPoint}`,layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX,...(name.endsWith("@fused")?{nbAirFused:1,...(entryPoint==="airPlanPropagate"?{nbAirLanes:256}: {})}:{}),...(name.endsWith("@withX")?{nbXWithSelect:1}:{}),...(name.endsWith("@planned")?{nbAirListed:1}:{}),...(entryPoint==="depthReuseSnapshot"?{nbListWithSnapshot:1}:{}),umCountedJobs:(entryPoint==="couple"||entryPoint==="transfer")?3:0,...(entryPoint==="depthCoarse"?{nbCachedGeometry:0}:{}),...(name.endsWith("@noplanes")?{nbMetricPlanes:0,...(entryPoint==="depthReuseSnapshot"?{}:{nbColdCurrentPhi:1})}:{}),...(NB_H_STORE_ENTRIES.has(entryPoint)||name.includes("@h")?UNIFORM_DETAIL_H_STORE:{}),...(entryPoint==="depthSeeds"?UNIFORM_DETAIL_MERGED_SLOTS:{}),...(entryPoint==="advectParticles"?{nbFastSample:1,umSolidFree:1}:{}),...(name.endsWith("@free")?{umSolidFree:1}:NB_FREE_TWIN_ENTRIES.includes(name)?{umSolidTwinned:1}:{}),...(entryPoint==="transfer"?{nbTransferTwin:name.endsWith("@fit")?1:2}:{}),...s}}})).catch(e=>{throw new Error(`${entryPoint}: ${e.message} ${e.reason}`,{cause:e});});
   this.pipelines.set(name,p);
  }));
  if(this.slotSnapshotResources){
   const module=uniformDetailModule(this.device,{specializeWorldGeometry:true,label:"Narrow-band FLIP slot snapshot",code:sourceCode(true,true)});
   const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
   this.slotSnapshotPipeline=await uniformDetailPipeline(this.device,this.ownership,{layout:this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.slotSnapshotResources]}),compute:{module,entryPoint:"snapshot"}});
  }
  this.airFused=this.compile(["airPlanSeeds@fused","airPlanPropagate@fused"],undefined,"regional air update");
  this.airPlanned=this.compile(["airPlanSeeds","airPlanPropagate","airClearFine","airClearCoarse@planned"],undefined,"retire empty air");
  this.crossingDomain=this.compile(["depthPlanSeed","depthPlanExpand"],"surface","shared frame crossing domain");
  const search=["depthTiles","depthSeeds",...narrowBandNearestControlEntries,...narrowBandNearestEntries];
  this.batches={
   advect:this.compile(["advectParticles","advectParticlesDeferred"],"advectParticles","advectParticles"),
   marking:this.compile(["markParticleTiles","commitCount"],"surface","particle tile marking"),
   countInitial:this.compile(["countInitial"],"bootstrap","countInitial"),seedInitial:this.compile(["seedInitial"],"bootstrap","seedInitial"),commit:this.compile(["commitCount"],undefined,"commitCount"),
   activity:this.compile(["activity","activitySpread"],"bootstrap","activity"),
   search:this.compile(search,undefined,"crossing search"),searchReconstruct:this.compile(search.map(n=>n==="depthTiles"||n==="depthSeeds"||n==="depthReuseSnapshot"?`${n}@noplanes`:n),"reconstruct","crossing search"),searchRedistance:this.compile(search,"redistance","crossing search"),
   diagnose:this.compile(["diagnoseBefore"],"reconstruct","diagnoseBefore"),
   reconstruct:this.compile(["surfaceClear",...(this.fusedSurface?["surfaceSplatOccupied"]:["surfaceCells","surfaceSplat"]),"coupleFine@free","coupleFine","couple"],"reconstruct","surface reconstruction"),
   reconstructCoarse:this.compile(["surfaceClear",...(this.fusedSurface?["surfaceSplatOccupied"]:["surfaceCells","surfaceSplat"]),"couple"],"reconstruct","surface reconstruction"),
   // Classification reads phase and phi. Transfer loads canonical face
   // texels directly in both mixed and all-4h modes; neither samples a cache.
   // The tiled deposit leaves the gather pipeline the layouts it does not fit.
   transfer:this.compile(["classify",...(this.particleDeposit==="tiled"?["transferDeposit","transferProperties"]:["transferProperties","transfer@fit"]),"transfer","transferOwners","transferSeams"],"transfer","transfer"),
   snapshot:this.compile(["snapshot"],undefined,"snapshot"),
   retire:this.compile(["depthCoarse","redistanceCoarse"],undefined,"coarse retirement"),
   clearAir:this.compile(["airSeeds","airSeedsFine","airSpread0","airSpread1","airSpread2","airGate","airClearFine","airClearCoarse"],undefined,"retire empty air"),
   redistanceFine:this.compile(["depthGeometry","buildDistance@h@free","buildDistance@h","redistanceFine"],undefined,"redistance"),
   redistanceMetric:this.compile(["depthGeometry","buildDistance"],undefined,"redistance"),
   metric:this.compile(["depthGeometry","buildDistance"],undefined,"surface metric"),
   metricFine:this.compile(["depthGeometry","buildDistance@h@free","buildDistance@h"],undefined,"surface metric"),
   update:this.compile(["update"],undefined,"update"),
   resample:this.compile(["resampleCount","resamplePrefix","resample","seedCells","seedSources","seed"],"update","resample and seed"),
   resampleKeep:this.compile(["resampleCount","resamplePrefix","resample"],"update","resample"),
  };
  const listedSearch=search.filter(n=>n!=="depthNearestX").map(n=>n==="depthReuseSelect"?"depthReuseSelect@withX":n);
  this.crossingSearches.set(this.batches.search,this.compile(listedSearch,undefined,"crossing search"));
  this.crossingSearches.set(this.batches.searchReconstruct,this.compile(listedSearch.map(n=>n==="depthTiles"||n==="depthSeeds"||n==="depthReuseSnapshot"?`${n}@noplanes`:n),"reconstruct","crossing search"));
  this.crossingSearches.set(this.batches.searchRedistance,this.compile(listedSearch,"redistance","crossing search"));
 }
 private compile(entries:readonly string[],groupName:string|undefined,label:string):NarrowBandBatch{
  return {pass:{label:`Narrow-band FLIP ${label}`},steps:entries.map(name=>{
   const selected=this.pipelines.get(name),entry=name.split("@")[0]!;if(!selected)throw new Error(`Narrow-band FLIP ${name} is not initialized`);
   return {
    groups:groupName??((entry.startsWith("airClear")||name==="airPlanPropagate@fused")?"redistanceStore":entry.startsWith("air")?"surface":entry.startsWith("redistance")?"redistanceStore":entry.startsWith("resample")?"update":entry==="markParticleTiles"||entry==="surfaceCells"||entry==="surfaceSplat"||entry==="diagnoseBefore"||entry==="buildDistance"||entry.startsWith("depth")?"surface":entry.startsWith("transfer")||entry==="classify"?"transfer":entry==="couple"||entry==="coupleFine"||entry==="commitCount"?"surface":entry),
    // In-place redistance reads its own canonical texel: coarse vertices
    // always read the base home, including those outside the resident h ring.
    base:entry==="redistanceCoarse"||entry==="airClearCoarse",pipeline:uniformDetailPick(this.solid.select(selected)),
    launch:["depthTiles","depthPlanSeed","depthPlanExpand","depthReuseSnapshot","depthReuseSelect","airPlanSeeds","airPlanPropagate"].includes(entry)?NB_LAUNCH.regions:entry==="couple"||entry==="transfer"?NB_LAUNCH.counted:entry==="resamplePrefix"?NB_LAUNCH.one
     :entry==="advectParticles"||entry==="snapshot"||entry==="update"||entry==="resampleCount"||entry==="resample"||entry==="markParticleTiles"?NB_LAUNCH.particles:NB_LAUNCH.grid,
   };
  })};
 }
 private get compiled():NarrowBandBatches{if(!this.batches)throw new Error("Narrow-band FLIP is not initialized");return this.batches;}
 /** The metric build: its h-side twin while the h-tile capacity is non-zero. */
 private get metric():NarrowBandBatch{const b=this.compiled;return this.ownership.coarseOnly?b.metric:b.metricFine;}
 private run(encoder:GPUCommandEncoder,batch:NarrowBandBatch,shared?:GPUComputePassEncoder):void{
  const pass=shared??encoder.beginComputePass(batch.pass);
  pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(2,this.solid.tileGroup);
  let bound:GPUBindGroup|undefined;
  const steps=batch.steps;
  for(let i=0;i<steps.length;i++){
   const step=steps[i]!,binding=this.groups[step.groups]![this.parity]!;
   const group=step.base?binding.base:binding.group;
   if(group!==bound){pass.setBindGroup(1,group);bound=group;}
   if(step.launch===NB_LAUNCH.counted){this.ownership.dispatchAllCounted(pass,step.pipeline);continue;}
   pass.setPipeline(step.pipeline);
   if(step.launch===NB_LAUNCH.one)pass.dispatchWorkgroups(1);
   else if(step.launch===NB_LAUNCH.particles)pass.dispatchWorkgroups(this.particleGroups);
   else if(step.launch===NB_LAUNCH.regions)pass.dispatchWorkgroups(Math.min(128,this.reuse.regions));
   else pass.dispatchWorkgroups(1024);
  }
  if(!shared)pass.end();
 }
 /** The step record (parameter words 0-6): cell size, dt, FLIP ratio, open
  * top and gravity. */
 private writeStep(dt:number,openTop:number,gravity:number):void{
  const w=this.stepWords,h=this.ownership.capacity.lattice.cellSize_m;
  w[0]=h[0]!;w[1]=h[1]!;w[2]=h[2]!;w[3]=dt;w[4]=this.apic?0:0.95;w[5]=openTop;w[6]=gravity;w[7]=+this.measureTraces;
  this.device.queue.writeBuffer(this.params,0,w);
 }
 move(encoder:GPUCommandEncoder,dt:number,openTop:boolean,gravity=0):void{
  if(this.idleNow)return;
  const b=this.compiled;
  this.bandCurrent=false;this.crossingCanPlan=false;this.crossingIsPlanned=false;this.framePlan?.invalidateDomain(this);
  this.writeStep(dt,+openTop,gravity);
  // Words 4 to 10 are the step's (10: advect's slow samples); word 11, the arrival fatal's page, stays.
  encoder.clearBuffer(this.state,16,28);
  this.bootstrap(encoder);
  encoder.clearBuffer(this.bins,(this.activityWord+5*this.tiles)*4,2*this.tiles*4);
  // The samples move in place, counting themselves into clear bins and
  // marking the tiles they reach; the order packs them into the other buffer.
  encoder.clearBuffer(this.state,4,8);this.order.prepare(encoder,1-this.parity);
  encoder.clearBuffer(this.bins,this.cells*8+this.tiles*4,this.tiles*4);
  this.run(encoder,b.advect);
  this.order.encode(encoder,1-this.parity);
  this.run(encoder,b.marking);this.crossingCanPlan=true;
  // The count every particle consumer until update gates on (liveGate).
  encoder.copyBufferToBuffer(this.state,0,this.params,48,4);encoder.copyBufferToBuffer(this.state,0,this.liveGate,0,4);
 }
 private bootstrap(encoder:GPUCommandEncoder):void{
  if(this.started)return;
  const b=this.compiled;
  encoder.clearBuffer(this.state,4,8);encoder.clearBuffer(this.bins,0,this.cells*8);
  encoder.clearBuffer(this.bins,this.cells*8+this.tiles*4,this.tiles*4);
  this.measureBand(encoder,b.search);this.buildDistance(encoder,this.metric);this.cache.encode(encoder,this.cacheGroups.bootstrap!);
  // Coverage and the occupied list start empty: countInitial lists its tiles.
  encoder.clearBuffer(this.bins,this.cells*8,this.tiles*4);encoder.clearBuffer(this.bins,4*this.occupiedWord,4);
  this.run(encoder,b.countInitial);this.order.encodeOffsets(encoder,1-this.parity);
  this.run(encoder,b.seedInitial);this.run(encoder,b.commit);
  this.parity=1-this.parity;this.started=true;
 }
 /** Called only for enabled automatic refinement. Explicit Requested/Full
  * layouts keep their authored semantics; particle lifetime is independent. */
 refine(encoder:GPUCommandEncoder,target:{buffer:GPUBuffer;wordOffset:number;words?:number},dt:number,padding=1):void{
  if(this.idleNow)return;
  const lattice=this.ownership.capacity.lattice;
  const w=this.refinementWords,d=this.address.cells,h=lattice.cellSize_m;
  w[0]=d[0]!;w[1]=d[1]!;w[2]=d[2]!;w[3]=dt;w[4]=h[0]!;w[5]=h[1]!;w[6]=h[2]!;w[7]=padding;
  this.device.queue.writeBuffer(this.refinementParams,0,w);
  // One group per particle bank: the bank alternates every step, the census's target does not.
  const particles=this.activeParticles,size=4*Math.ceil(this.address.tileKeys/32);
  let held=this.refinementGroups[this.parity];
  if(held?.particles!==particles||held.target!==target.buffer||held.wordOffset!==target.wordOffset||held.size!==size){
   // The marks are a bit a tile key: a plane of any other length is read by another index.
   if(target.words!==size/4)throw new Error(`The refinement join plane holds ${target.words??"an unstated number of"} words; a bit per tile key needs ${size/4}.`);
   held=this.refinementGroups[this.parity]={particles,target:target.buffer,wordOffset:target.wordOffset,size,group:this.device.createBindGroup({layout:this.refinementLayout!,entries:[
    {binding:0,resource:{buffer:particles}},{binding:1,resource:{buffer:this.state}},
    {binding:2,resource:{buffer:target.buffer,offset:target.wordOffset*4,size}},
    {binding:3,resource:{buffer:this.refinementParams}},
    {binding:4,resource:{buffer:this.bins}},{binding:5,resource:{buffer:this.ownership.presentation.buffer}},
    {binding:6,resource:{buffer:this.particleAllocation}},
   ]})};
  }
  const group=held.group;
  encoder.clearBuffer(this.particleAllocation);
  const pass=encoder.beginComputePass(NB_REFINEMENT_PASS);pass.setPipeline(this.refinementPipeline!);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1024);pass.end();
 }
 /** The page initialise's hook, after the pool's commit and before activity:
  * the heat and target of every tile key of the blocks that commit claimed
  * take 0. One fixed launch that strides the pool's new list. */
 encodeInitBlocks(encoder:GPUCommandEncoder):void{
  if(!this.initPipeline||!this.pool)throw new Error("Narrow-band FLIP has no page pool: no block is claimed");
  const pass=encoder.beginComputePass(NB_INIT_PASS);pass.setPipeline(this.initPipeline);pass.setBindGroup(0,this.initGroup!);pass.dispatchWorkgroups(uniformPageInitGroups(this.pool.layout));pass.end();
 }
 /** The head census runs before remapping and advection. Copy its scores
  * into the existing arena, then publish heat in separate neighbour passes. */
 activity(encoder:GPUCommandEncoder,importance:{buffer:GPUBuffer;offset:number},dt:number,fadeSeconds=0.5,sparseSeconds=2):void{
  const w=this.activityWords;w[0]=2/Math.max(0.05,fadeSeconds);w[1]=sparseSeconds;w[2]=0;w[3]=0;
  this.device.queue.writeBuffer(this.params,32,w);
  this.writeStep(dt,0,0);
  if(this.updateErrors)encoder.copyBufferToBuffer(this.updateErrors,32*this.tiles,this.bins,4*(this.activityWord+7*this.tiles),16*this.tiles);
  encoder.copyBufferToBuffer(importance.buffer,importance.offset,this.bins,this.activityWord*4,this.tiles*8);
  this.run(encoder,this.compiled.activity);
 }
 reconstruct(encoder:GPUCommandEncoder):void{if(this.idleNow)return;this.syncSurfaceRevisions(encoder,true);const b=this.compiled;this.bandCurrent=false;this.measureBand(encoder,b.searchReconstruct);if(this.measureTraces)this.run(encoder,b.diagnose);
  this.run(encoder,this.ownership.coarseOnly?b.reconstructCoarse:b.reconstruct);}
 transfer(encoder:GPUCommandEncoder):void{
  if(this.idleNow)return;
  // Clearing a buried zero invalidates the crossing/metric banks as well
  // as phi. The caller resolved phi before momentum or membership reads it.
  if(!this.bandCurrent){this.measureBand(encoder,this.compiled.search);this.buildDistance(encoder,this.metric);this.bandCurrent=true;}
  this.run(encoder,this.compiled.transfer);
 }
 snapshot(encoder:GPUCommandEncoder,slots?:UniformNarrowBandSlotSnapshot):void{
  if(this.idleNow)return;
  // snapshot and update read the same tiers (nbSampleCold).
  this.syncTiers(encoder);const pass=encoder.beginComputePass(NB_SNAPSHOT_PASS);
  this.cache.encode(encoder,slots?.cache??this.cacheGroups.snapshot!,pass);
  if(slots&&this.ownership.capacity.fineTiles>0)this.stageTaps!.builder.encode(encoder,slots.taps,pass);else if(!slots)this.encodeStageTaps(encoder,pass);
  if(slots){
   if(!this.slotSnapshotPipeline)throw new Error("Slot snapshot is not initialized");
   pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,slots.particles[this.parity]!.group);pass.setPipeline(uniformDetailPick(this.slotSnapshotPipeline));pass.dispatchWorkgroups(this.particleGroups);
  }else this.run(encoder,this.compiled.snapshot,pass);pass.end();
 }
 private encodeStageTaps(encoder:GPUCommandEncoder,shared?:GPUComputePassEncoder):void{if(this.stageTaps&&this.ownership.capacity.fineTiles>0)this.stageTaps.builder.encode(encoder,this.stageTaps.group,shared);}
 /** Adaptive retirement can move a hanging zero when it normalizes coarse
  * corner magnitudes. Store those corners first, from their own metric, so
  * the search and metric redistance builds are the end-step ones. True when
  * it wrote phi: the caller resolves it before redistance. */
 retire(encoder:GPUCommandEncoder):boolean{
  if(this.ownership.coarseOnly||this.idleNow)return false;
  const b=this.compiled;
  this.measureBand(encoder,b.searchRedistance);this.retireSearched=true;this.run(encoder,b.retire);return true;
 }
 /** Fine redistance preserves the crossing set, so its search and metric
  * also serve end-step particle membership. Adaptive coarse corners were
  * stored by retire(); only they could have moved a zero. */
 redistance(encoder:GPUCommandEncoder):void{
  if(this.idleNow)return;
  const b=this.compiled,coarse=this.ownership.coarseOnly;
  // Shared: retirement's search stands (it only stores coarse corners).
  if(this.redistanceSearch==="fresh"||!this.retireSearched)this.measureBand(encoder,b.searchRedistance);
  this.retireSearched=false;
  this.buildDistance(encoder,coarse?b.redistanceMetric:b.redistanceFine);this.bandCurrent=true;
 }
 /** retire() ran the redistance search and no other search has run since. */
 private retireSearched=false;
 /** Clear only the certified empty field outside physical liquid/particle support.
  * It sets state word NARROW_BAND_AIR_CHANGED_WORD where it changes a 4h-lattice vertex. */
 clearAir(encoder:GPUCommandEncoder):void{
  const planned=this.crossingIsPlanned&&this.framePlan?.active&&this.crossingPlanSerial===this.framePlan.serial;
  const fused=planned&&this.framePlan!.scheduleStages&&this.framePlan!.fuseAirClear;
  if(!fused||this.idleNow)encoder.clearBuffer(this.state,4*NARROW_BAND_AIR_CHANGED_WORD,4);
  if(this.idleNow)return;this.syncTiers(encoder);
  this.run(encoder,fused?this.airFused!:planned?this.airPlanned!:this.compiled.clearAir);this.bandCurrent=false;
 }
 private buildDistance(encoder:GPUCommandEncoder,batch:NarrowBandBatch):void{
  encoder.clearBuffer(this.bins,4*this.cells,4);
  this.run(encoder,batch);
 }
 private measureBand(encoder:GPUCommandEncoder,batch:NarrowBandBatch):void{
  this.retireSearched=false;this.syncSurfaceRevisions(encoder);this.syncTiers(encoder);
  if(this.regionalCrossingReuseEnabled!==this.seenCrossingReuseEnabled){
   if(this.reuse.masks>this.reuse.history)encoder.clearBuffer(this.bins,4*this.reuse.history,4*(this.reuse.masks-this.reuse.history));
   this.seenCrossingReuseEnabled=this.regionalCrossingReuseEnabled;
  }
  const parity=this.reuseParity;this.reuseParity=1-parity;
  encoder.copyBufferToBuffer(this.reuseControls,12*(parity+(this.revisions?(this.regionalCrossingReuseEnabled?0:2):4)),this.bins,4*this.reuse.control,12);
  // The plan is sealed only after movement publishes particle support.
  // Bootstrap and standalone calls retain the complete original path.
  const wanted=this.crossingCanPlan&&this.regionalCrossingPlanEnabled&&this.framePlan?.active;
  // Clear counters with the domain's other setup, before opening its pass.
  // The producer and its first consumer then share one compute pass.
  encoder.clearBuffer(this.bins,(3*this.cells+this.vertices+2*this.tiles)*4,(wanted?4:4+3*this.tiles)*4);
  const planned=wanted&&this.framePlan!.encodeDomain(encoder,this,this.encodeCrossingDomain);
  this.crossingIsPlanned=!!planned;this.crossingPlanSerial=planned?this.framePlan!.serial:-1;
  if(!planned)encoder.clearBuffer(this.bins,4*this.reuse.framePlan,4);
  this.run(encoder,planned?this.crossingSearches.get(batch)??batch:batch,this.crossingPass);
  this.crossingPass?.end();this.crossingPass=undefined;
 }
 update(encoder:GPUCommandEncoder):void{
  if(this.idleNow)return;
  const b=this.compiled;
  if(!this.bandCurrent){this.measureBand(encoder,b.search);this.buildDistance(encoder,this.metric);}this.bandCurrent=false;
  this.syncTiers(encoder);const pass=encoder.beginComputePass(NB_UPDATE_PASS);
  this.cache.encode(encoder,this.cacheGroups.update!,pass);this.encodeStageTaps(encoder,pass);
  this.run(encoder,b.update,pass);pass.end();
  // The ordered epoch compacts into the other buffer; the bins keep its counts for the seeding.
  this.parity=1-this.parity;
  this.run(encoder,this.quiet?b.resampleKeep:b.resample);
  this.run(encoder,b.commit);
  this.parity=1-this.parity;
 }
 noteReceipt(words:Uint32Array):void{
  this.count=words[1]!;this.reseedClipped=words[2]!;
  const bits=this.receiptBits,distances=this.receiptDistances,d=this.diagnostics;
  bits[0]=words[4]!;bits[1]=words[5]!;
  d.beforeMaxOutside=this.measureTraces?distances[0]!:NaN;d.unsupported=words[7]!;
  // The surface census runs while traces are measured (classify).
  d.afterMaxOutside=this.measureTraces?distances[1]!:NaN;d.outsideSurface=this.measureTraces?words[6]!:NaN;d.deepInterior=this.measureTraces?words[8]!:NaN;
  // Persistent samples own the budget first. At capacity optional reseeding
  // is deferred (reseedClipped); never destroy a surface to make room for it.
  // Liquid a source or the initial band must add has no such deferral.
  if(words[3]===4){const p=words[11]!;throw new Error(`Narrow-band FLIP: a particle arrived in page (${p&63},${(p>>>6)&63},${(p>>>12)&63}), which holds no storage; the residency closure is too small`);}
  if(words[3])throw new Error(`Narrow-band FLIP ${words[3]===1?`trajectory exceeded ${NARROW_BAND_TRACE_LIMIT} cells`:words[3]===2?"nonfinite velocity"
   :`particle pool is full: ${words[1]} samples live of a capacity of ${this.capacity}, and source or initial liquid still had samples to add`}`);
 }
 destroy():void{this.reuseControls.destroy();this.order.destroy();this.coarseVelocity.destroy();this.refinementParams.destroy();this.particleAllocation.destroy();for(const b of [...this.particles,this.state,this.bins,this.next,this.params,this.liveGate])b.destroy();}
}
