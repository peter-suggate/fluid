import { particleDepositStorage } from "./uniform-particle-deposit-storage";
import {uniformSolidPointOpenWGSL,uniformSolidSlideStartWGSL,uniformSolidPointCellWGSL} from "./uniform-solid-point-open.wgsl";
import type { UniformRegionalFramePlan } from "./uniform-regional-frame-plan";
import { GPUFluidParticleMotion } from "../../core/webgpu-particle-motion";
import { narrowBandSparseWGSL } from "./uniform-narrow-band-sparse.wgsl";
import { narrowBandAirWGSL, NARROW_BAND_AIR_CHANGED_WORD, NARROW_BAND_AIR_GATE_PLANE, NARROW_BAND_AIR_WORDS_PER_TILE } from "./uniform-narrow-band-air.wgsl";
import { narrowBandParticleStride, narrowBandParticleWGSL, narrowBandApicMomentWGSL, type NarrowBandTransfer } from "./uniform-narrow-band-apic.wgsl";
import { PARTICLE_RESIDENCY_WORDS } from "./uniform-local-residency";
import { narrowBandActivityTriggerWGSL, narrowBandActivityWGSL, NARROW_BAND_ACTIVITY_WORDS_PER_TILE } from "./uniform-narrow-band-activity.wgsl";
import {narrowBandReuseLayout,narrowBandReuseWGSL,type NarrowBandReuseLayout} from "./uniform-narrow-band-reuse.wgsl";
import { narrowBandRedistanceWGSL } from "./uniform-narrow-band-redistance.wgsl";
import { narrowBandMembershipWGSL } from "./uniform-narrow-band-membership.wgsl";
import { uniformStageListsWGSL } from "./uniform-stage-lists";
import { narrowBandArenaWGSL, narrowBandArenaCanonicalWGSL } from "./uniform-surface-brick-arena.wgsl";
import { UNIFORM_BRICK_TABLE_ENTRIES, uniformBrickTableClear, uniformBrickTableWGSL, uniformBrickTableWords } from "./uniform-brick-table";
import { narrowBandNearestEntries, narrowBandNearestControlEntries } from "./uniform-narrow-band-nearest.wgsl";
import { GPUBackgroundPipelines, joinPendingPipelines } from "../../core/gpu-background-pipelines";
import { narrowBandNearWGSL } from "./uniform-narrow-band-near.wgsl";
import { narrowBandTerrainContactSource, narrowBandRigidContactSource } from "./uniform-particle-contact.wgsl";
import { narrowBandBodyWindowWGSL } from "./uniform-particle-body-window.wgsl";
import { narrowBandClearWalkWGSL } from "./uniform-particle-clear-walk.wgsl";
import { uniformMixedSourceWGSL, UNIFORM_PARAMS_BYTES } from "./uniform-mixed-source.wgsl";
import { NB_MOVE_TICKET_WORD, NB_SLAB_SEEDS_WORD, particleSlabSpill, particleSlabWGSL, particleSlabWords } from "./uniform-particle-slab.wgsl";
import { NARROW_BAND_CENTRED_BOUND_WGSL, NARROW_BAND_WALL_BOUND_WGSL, narrowBandSurfaceWGSL, narrowBandExpectedMassWGSL, narrowBandFineTransferSource, narrowBandTransferPropertiesSource } from "./uniform-narrow-band-surface.wgsl";
import { particleBrickDepositWGSL } from "./uniform-particle-brick.wgsl";
import { UNIFORM_DETAIL_4H_LOAD } from "../../core/uniform-detail-abi";
import { UNIFORM_DETAIL_H_STORE, UNIFORM_DETAIL_MERGED_SLOTS, uniformDetailBindLayout, uniformDetailGroup, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, type UniformDetailGroup } from "./uniform-detail-fields";
import { uniformMixedTopologyWGSL, uniformMixedCertifiedEntriesWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformBlockNeighbourWGSL, type UniformLatticeAddress } from "./uniform-lattice-address";
import { uniformMixedFaceAddressWGSL } from "./uniform-mixed-face-dispatch.wgsl";
import { uniformMixedVertexSamplingSource } from "./uniform-mixed-vertex-sampling.wgsl";
import { uniformMixedVelocitySamplingSource } from "./uniform-mixed-velocity-sampling.wgsl";
import { uniformCoarseCacheExtent } from "./uniform-coarse-velocity-cache";
import { uniformMixedSolidLibraryWGSL, uniformMixedSolidPipeline, uniformMixedVertexBuriedWGSL, type UniformMixedSolid } from "./uniform-mixed-solid.wgsl";
import { UniformMixedMomentumCache, type UniformMixedHangingTaps } from "./uniform-mixed-momentum-cache";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformPagePoolWGSL, type UniformPagePoolLayout } from "./uniform-page-pool";
import { uniformPageInitBlocksWGSL, uniformPageInitGroups, type UniformPageInitBlocks } from "./uniform-page-initialise";
import type { GPUFluidParticleSource } from "../../core/webgpu-particle-overlay";
import { UNIFORM_COLD_FREEZE, UniformColdFreeze } from "./uniform-cold-freeze";
import type { UniformColdFrame } from "./uniform-cold-frame";
import { uniformFrameExtensionDeltaWGSL } from "./uniform-frame-extension";
import { UNIFORM_REGION_TIER } from "./uniform-region-heat";

/** Workgroups of every launch over the live samples or slabs, one row: a
 * kernel reads its count on the GPU and strides the row through it. */
export const NARROW_BAND_PARTICLE_GRID=4096;
/** The row of a frame whose lagged heat holds no HOT region: still complete
 * at any count, every kernel strides the actual GPU count. */
export const NARROW_BAND_QUIET_PARTICLE_GRID=64;
const NB_CANONICAL_SURFACE_ENTRIES=["arenaPublishCanonical","arenaPublishFinish"] as const;

/** A particle's one-step trace past this many cells an axis, the lattice's
 * extent (64 pages of 32 cells), is fatal: it arrives nowhere. */
const NARROW_BAND_TRACE_REACH=2048;
/** How a batch step launches: a fixed 1024-group stride, the ownership's
 * counted launch, one group, or the fixed row that strides the live
 * samples or the occupied tiles (NARROW_BAND_PARTICLE_GRID). */
// Entries of fine-only batches (reconstruct, the arena's gather and store:
// encoded only while the h-tile capacity is non-zero) on the h side of their
// group. The arena gather's all-4h twin is arenaGather@base.
const NB_H_STORE_ENTRIES=new Set<string>(["coupleFine","arenaGather","arenaStore",...NB_CANONICAL_SURFACE_ENTRIES]);
// Pipelines (by name) launched as a solid-free twin (name@free, umSolidFree)
// beside the general pipeline (umSolidTwinned): umSolidSkip leaves the step
// to one.
const NB_FREE_TWIN_ENTRIES=["coupleFine"];
const NB_LAUNCH={grid:0,counted:1,one:3,particles:4,regions:5} as const;
/** Workgroups of a region launch (a workgroup a region, striding the list):
 * 128 held each ~21 of 2744 regions in series (airPlanSeeds and
 * airPlanPropagate 0.2 ms each in an all-4h frame). */
const NB_REGION_GROUPS=1024;
// Pipelines an advance binds only on some frames, compiled at background
// priority (pipelinesPending says which this advance may bind). Unplanned
// crossings clear air through clearAir; planned but unfused ones through
// airPlanned; diagnoseBefore runs only for a reader measuring traces.
const NB_UNPLANNED_AIR=["airSeeds","airSeedsFine","airSpread0","airSpread1","airSpread2","airGate","airClearFine","airClearCoarse"] as const;
const NB_PLANNED_AIR=["airPlanSeeds","airPlanPropagate","airClearFine","airClearCoarse@planned"] as const;
const NB_DIAGNOSE=["diagnoseBefore"] as const;
const NB_BACKGROUND_PIPELINES=new Set<string>([...NB_UNPLANNED_AIR,...NB_PLANNED_AIR,...NB_DIAGNOSE]);
const NB_NO_BACKGROUND:readonly string[]=[];
/** One launch of a batch, resolved once at initialization: its bind groups
 * by name (bind() replaces the groups), its pipeline and its launch. */
interface NarrowBandStep{readonly groups:string;readonly base:boolean;readonly pipeline:GPUComputePipeline;readonly launch:number}
/** A pass's launches. A frame replays them; it builds no list or label.
 * deferred: its steps wait on background pipelines and resolve at the first
 * run (the advance awaited them: pipelinesPending). unavailable: a pipeline
 * it names is not compiled in this configuration, and running it throws. */
interface NarrowBandBatch{readonly pass:GPUComputePassDescriptor;steps:readonly NarrowBandStep[];deferred?:{readonly entries:readonly string[];readonly groupName:string|undefined};readonly unavailable?:string}
interface NarrowBandBatches{
 readonly advect:NarrowBandBatch;readonly allocate:NarrowBandBatch;readonly sortPrepare:NarrowBandBatch;readonly sort:NarrowBandBatch;readonly countInitial:NarrowBandBatch;readonly seedInitial:NarrowBandBatch;
 readonly activity:NarrowBandBatch;readonly searchReconstruct:NarrowBandBatch;
 readonly arenaSearch:NarrowBandBatch;readonly arenaSearchCoarse:NarrowBandBatch;readonly arenaMetric:NarrowBandBatch;readonly arenaMetricDerived:NarrowBandBatch;readonly arenaStore:NarrowBandBatch;
 readonly diagnose:NarrowBandBatch;readonly reconstruct:NarrowBandBatch;readonly reconstructCoarse:NarrowBandBatch;readonly transfer:NarrowBandBatch;
 readonly retire:NarrowBandBatch;readonly retireListed:NarrowBandBatch;
 readonly seed:NarrowBandBatch;readonly clearAir:NarrowBandBatch;
}
/** The FLIP delta (recordDelta), in its own module beside the stage
 * source: the frame extension's pair tail wrote the projected field's
 * extension minus the transferred field's into its delta pool (binding 19,
 * uniformFrameExtensionDeltaWGSL), a face each. A workgroup an occupied
 * tile, like the deposit: the 3³ tiles' flags and pool indices, then the
 * 540 faces of the tile's samples' MAC stencils (per axis 5 anchors along,
 * 6 across: offsets -1..3 and -1..4), each the pool's delta where the tail
 * swept it, else none. Each sample interpolates the faces that have a
 * value, renormalised; an axis with none (and every sample the record does
 * not reach: a COLD tile's, a seed's, a spilled one's) takes NB_DELTA_NONE,
 * which G2P reads as PIC on that axis.
 *
 * A face is known where the tail swept it: on a tile with the support bit
 * outside a frozen region (umFrozenRegion: this frame's held set, which the
 * extension never writes). There an h tile's face is its row's (by the
 * tile's h list rank), and a 4h tile's lies on its +planes: the canonical
 * delta, or against an h tile the subface's, in the row of the h tile
 * across. Low faces are the negative buffers': none. An h tile whose rank
 * lies past the pool's rows is fatal (11). */
const NB_DELTA_RECORD_WGSL=/* wgsl */`
@group(1) @binding(19) var<storage,read> ufxDelta:array<f32>;
const NB_DELTA_NONE:f32=3e38;const ND_LANES:u32=128u;
var<workgroup> ndJobs:u32;
var<workgroup> ndTile:u32;
var<workgroup> ndCold:u32;
// A window tile: 1 live, 2 known (the support bit, not held), 4 an h tile.
var<workgroup> ndFlags:array<u32,27>;
// A live window tile's pool index: an h tile's rank, a 4h tile's block.
var<workgroup> ndSlot:array<u32,27>;
var<workgroup> ndDelta:array<f32,540>;
// A face of the window: the anchor's offset from the tile origin.
fn ndOffset(s:u32)->vec3i{let axis=s/180u;let k=s%180u;var o=vec3i(0);o[axis]=i32(k%5u)-1;o[(axis+1u)%3u]=i32((k/5u)%6u)-1;o[(axis+2u)%3u]=i32(k/30u)-1;return o;}
fn ndWindow(origin:vec3i,cell:vec3i)->u32{let r=vec3u((cell>>vec3u(2u))-(origin>>vec3u(2u))+1);return r.x+3u*r.y+9u*r.z;}
fn ndFaceDelta(origin:vec3i,s:u32)->f32{
 let axis=s/180u;let anchor=origin+ndOffset(s);let w=ndWindow(origin,anchor);let flags=ndFlags[w];
 if((flags&2u)==0u||umFaceLow(anchor,axis)){return NB_DELTA_NONE;}
 let l=vec3u(anchor&vec3i(3));
 if((flags&4u)!=0u){return ufxDelta[ufxDeltaRowIndex(ndSlot[w],l,axis)];}
 if((anchor[axis]&3)!=3){return NB_DELTA_NONE;}
 var e=vec3i(0);e[axis]=1;
 // Offsets reach 4 along the axis: the cell across lies in the window.
 let x=ndWindow(origin,anchor+e);
 if((ndFlags[x]&5u)==5u){return ufxDelta[ufxDeltaLowIndex(ndSlot[x],axis,l[(axis+1u)%3u],l[(axis+2u)%3u])];}
 return ufxDelta[ufxDelta4Index(ndSlot[w],vec3u(anchor>>vec3u(2u))&vec3u(7u),axis)];
}
// A sample's delta, p relative to the tile origin.
fn ndSample(p:vec3f)->vec3f{
 var d=vec3f(NB_DELTA_NONE);
 for(var axis=0u;axis<3u;axis++){
  var g=p-0.5;g[axis]=p[axis]-1.0;
  let base=vec3i(floor(g))+1;let f=g-floor(g);
  var sum=0.0;var total=0.0;
  for(var k=0u;k<8u;k++){
   let bit=umCorner(k,2u);let w3=select(vec3f(1)-f,f,bit!=vec3u(0));let w=w3.x*w3.y*w3.z;
   let r=vec3u(base+vec3i(bit));let value=ndDelta[axis*180u+r[axis]+5u*r[(axis+1u)%3u]+30u*r[(axis+2u)%3u]];
   if(w>0.0&&value<1e30){sum+=w*value;total+=w;}
  }
  if(total>0.0){d[axis]=sum/total;}
 }
 return d;
}
@compute @workgroup_size(128) fn recordDelta(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){ndJobs=select(0u,min(atomicLoad(&bins[NB_OCCUPIED]),UM_TILES),nbParticlesLive());}
 let count=workgroupUniformLoad(&ndJobs);
 for(var job=group.x;job<count;job+=groups.x){
  workgroupBarrier();
  if(lane==0u){let t=atomicLoad(&bins[NB_OCCUPIED+4u+job]);ndTile=t;ndCold=u32(umTileLive(t)&&nbTileCold(t));}
  let tile=workgroupUniformLoad(&ndTile);let cold=workgroupUniformLoad(&ndCold)!=0u;
  if(!umTileLive(tile)){continue;}
  let bin=nbBinTile(tile);let start=links[bin];let end=links[bin+63u]+atomicLoad(&bins[bin+63u]);
  // A COLD tile's samples take no delta on any axis; one its region thaws by G2P takes PIC.
  if(cold){for(var i=start+lane;i<end;i+=ND_LANES){particles[i].before=vec4f(vec3f(NB_DELTA_NONE),particles[i].before.w);}continue;}
  let origin=vec3i(4u*umTileCoord(tile));
  if(lane<27u){
   let key=umTileKey((origin>>vec3u(2u))+vec3i(i32(lane%3u),i32((lane/3u)%3u),i32(lane/9u))-1);
   var flags=0u;var slot=0u;
   if(umKeyStored(key)&&umTileLive(key)){
    let fine=umTileWidth(key)==1u;
    flags=1u|select(0u,2u,(umTileSupport(key)&2u)!=0u&&!umFrozenRegion(umPageOfTile(key)))|select(0u,4u,fine);
    slot=select(key>>9u,ufxDeltaRank(key),fine);
    if(fine&&slot>=UFX_DELTA_FINE){atomicStore(&state[3],11u);flags=0u;}
   }
   ndFlags[lane]=flags;ndSlot[lane]=slot;
  }
  workgroupBarrier();
  for(var s=lane;s<540u;s+=ND_LANES){ndDelta[s]=ndFaceDelta(origin,s);}
  workgroupBarrier();
  // A ballistic sample's G2P takes no grid velocity: nothing to record.
  for(var i=start+lane;i<end;i+=ND_LANES){
   let s=particles[i];if(!nbLive(s)||nbParticleMode(s)==1.0){continue;}
   // Sorted samples lie in their tile; the clamp only bounds the window.
   particles[i].before=vec4f(ndSample(clamp(s.position.xyz-vec3f(origin),vec3f(0.0),vec3f(3.9999998))),s.before.w);
  }
 }
 // The samples in no tile's run: each current slab's seeds, then the spill reserve.
 let slabs=min(atomicLoad(&state[20u]),UB_BRICKS);let spilled=nbSpillCount(false);
 for(var job=group.x;job<slabs+(spilled+ND_LANES-1u)/ND_LANES;job+=groups.x){
  var first=nbSpillBase(false)+ND_LANES*(job-slabs);var n=min(ND_LANES,spilled-min(ND_LANES*(job-slabs),spilled));
  if(job<slabs){
   let id=atomicLoad(&state[NB_SLAB_OUT+job]);let filed=atomicLoad(&state[nbSlab(id,11u)]);
   first=atomicLoad(&state[nbSlab(id,5u)])+filed;n=nbSlabCount(id)-min(filed,nbSlabCount(id));
  }
  for(var k=lane;k<n;k+=ND_LANES){particles[first+k].before=vec4f(vec3f(NB_DELTA_NONE),particles[first+k].before.w);}
 }
}
`;
const NB_DELTA_PASS:GPUComputePassDescriptor={label:"Narrow-band FLIP delta"};
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
export interface NarrowBandSurfaceRevisions{readonly buffer:GPUBuffer;readonly liveOffset:number;readonly scratchOffset:number;readonly regions:number}
/** The particles' h capacity (scene fluid.tiers): every particle buffer is
 * sized by fineTiles h tiles of 64 cells at perFineCell samples a cell, not
 * by the pool's cell keys. bankBytes floors each bank's buffer: the
 * extension borrows a bank as its paired companion (UniformMixedFrame
 * bindStage), sized by the pool's h tiles; the records stay capacity's. */
/** The surface arena's windows under a particle budget, at least (arenaWindows). */
const NB_ARENA_WINDOW_FLOOR=2048;
/** The reuse cache's fewest payload slots under a particle budget. */
const NB_REUSE_SLOT_FLOOR=256;
export interface NarrowBandParticleBudget{readonly fineTiles:number;readonly perFineCell:number;readonly bankBytes?:number;
 /** The records by rule (uniformParticleRecordRule, M4b): an overlap reseed
  * past them is deferred (reseedClipped), and the slabs hold them at any
  * spread: every stored brick may hold a slab. Unset: 64 x fineTiles x perFineCell. */
 readonly records?:number}
/** Particle records for a budget: its samples, then the slabs' spill reserve
 * (particleSlabSpill) on top, so the slabs alone hold the budget. Under
 * records over `blocks` blocks the slabs are sized for the worst spread:
 * a scratch slab holds a brick's count and a quarter more, at least 32,
 * and a current slab its count, an eighth more, at least 16, and its
 * refused seeds (uniform-particle-slab.wgsl.ts), so R records over B
 * occupied bricks (B at most R and every stored brick, 64 a block) need
 * 1.25 R + 32 B. */
export function narrowBandBudgetCapacity(budget:NarrowBandParticleBudget,blocks=0):number{
 const samples=budget.records!==undefined?Math.ceil(1.25*budget.records)+32*Math.min(budget.records,64*blocks):Math.ceil(64*budget.fineTiles*budget.perFineCell);
 let capacity=samples+Math.max(4096,Math.ceil(samples/15));
 while(capacity-particleSlabSpill(capacity)<samples)capacity++;
 return capacity;
}
/** Diagnostics only (set before a solver is built; off leaves every shader
 * unchanged): count the move's retirements in receipt words from
 * NARROW_BAND_RETIRE_WORD, cumulative: deep, sparse expiry, crowded cell,
 * blocked in a solid, heat expiry, then sparse, blocked and heat expiry of a
 * sample less than a cell inside the advected level set (liquid it may not hold).
 * splitX at or above 0 makes those last three words, at cells x >= splitX:
 * the mass retired deep past splitX + 16, the mass retired deep in
 * [splitX, splitX + 16), and seeds offered. Then (crowdedCells "merge")
 * samples merged into another, excess samples a full gatherer refused (kept
 * live over the cap, a step each), splits, and the mass past 1 of retired
 * merged samples. */
export const NARROW_BAND_RETIRE_WORD=32;
export const NARROW_BAND_RETIRE_WORDS=12;
/** The most sampleSeparation "pairs" moves a coupled sample along an axis, in
 * cells. The move is the mean of its pair corrections (nbSeparation), each
 * m_j/(m_i+m_j)(r_i+r_j-l) < r_i+r_j with r = cbrt(m)/4: so under twice the
 * heaviest sample's radius, cbrt(8)/2 = 1 cell under crowdedCells "merge"
 * (NB_MASS_MAX 8), half a cell with unit samples. */
export interface NarrowBandSeedRank{readonly bits:number;readonly levels:number;readonly keyBits:number}
/** The seed budget's radix shape: levels digits of bits each cover the
 * cell keys (< cells), and their histograms and three control words fit
 * the seam list (one word a tile), which nothing holds during the seed batch. */
export function narrowBandSeedRankShape(cells:number,tiles:number):NarrowBandSeedRank{
 const keyBits=Math.max(1,Math.ceil(Math.log2(cells)));
 if(keyBits>30)throw new Error(`Narrow-band FLIP: ${cells} cell keys exceed the seed list's 30-bit key`);
 for(let bits=10;bits>=1;bits--){const levels=Math.ceil(keyBits/bits);if(levels*(1<<bits)+3<=tiles)return {bits,levels,keyBits};}
 throw new Error(`Narrow-band FLIP: ${tiles} tile keys cannot hold the seed rank's histograms`);
}
/** Overlap reseeds past the seed budget are not refused in launch order.
 * Each candidate's rank key is its cell key XOR a frame mask (a bijection,
 * so the order is canonical while the deferred cells change from frame to
 * frame). The budget admits the first candidates by key, then by site index
 * in the cut cell (nbSeedAdmitted). seedRank@level counts the candidates
 * whose key matches the cut's prefix by their next digit; seedCut@level
 * finds the digit where the count passes the room left. It reads the seed
 * list (seedCells), seedCandidate, the sorted records' count (state[1]) and
 * the step's seeds so far (the source seeds). */
export function narrowBandSeedRankWGSL(rank:NarrowBandSeedRank,budget:number):string{
 const {bits,levels,keyBits}=rank;
 return /* wgsl */`
const NB_SEED_RANK:u32=NB_BAND_SEAM;const NB_SEED_BITS:u32=${bits}u;const NB_SEED_LEVELS:u32=${levels}u;const NB_SEED_BUCKETS:u32=${1<<bits}u;
// The cut's key prefix, the room left inside it, and whether the budget binds.
const NB_SEED_CONTROL:u32=NB_SEED_RANK+NB_SEED_LEVELS*NB_SEED_BUCKETS;
override nbSeedLevel:u32=0u;
var<workgroup> nbSeedSums:array<u32,64>;
var<workgroup> nbSeedKept:array<u32,64>;
// The records' rule: sorted records and this step's source seeds count first.
fn nbSeedRoom()->u32{let used=atomicLoad(&state[1])+atomicLoad(&state[${NB_SLAB_SEEDS_WORD}u]);return select(0u,${budget}u-used,used<${budget}u);}
// No rank is needed where every list entry's eight sites fit (most frames).
fn nbSeedRankIdle()->bool{return 8u*min(atomicLoad(&bins[NB_BAND+3u]),3u*arrayLength(&particles))<=nbSeedRoom();}
// An overlap candidate's rank key (one for the eight lanes of its entry), or NB_NO_BIN.
fn nbSeedRankKey(group:vec3u,lane:u32)->u32{
 let slot=8u*group.x+lane/8u;if(slot>=min(atomicLoad(&bins[NB_BAND+3u]),3u*arrayLength(&particles))){return NB_NO_BIN;}
 let entry=links[nbSeedList()+slot];if((entry&0x80000000u)!=0u){return NB_NO_BIN;}
 var h=params.live.z*0x9e3779b9u;h^=h>>16u;h*=0x7feb352du;h^=h>>15u;
 return ((entry&0x3fffffffu)^h)&${((2**keyBits)-1)>>>0}u;
}
@compute @workgroup_size(64) fn seedRank(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let key=nbSeedRankKey(group,lane);if(key==NB_NO_BIN){return;}
 if(select(nbSeedRankIdle(),atomicLoad(&bins[NB_SEED_CONTROL+2u])==0u,nbSeedLevel>0u)){return;}
 let shift=(NB_SEED_LEVELS-1u-nbSeedLevel)*NB_SEED_BITS;
 if(((key>>shift)>>NB_SEED_BITS)!=atomicLoad(&bins[NB_SEED_CONTROL])){return;}
 if(seedCandidate(group,lane,false,false).position.x<0.0){return;}
 atomicAdd(&bins[NB_SEED_RANK+nbSeedLevel*NB_SEED_BUCKETS+((key>>shift)&(NB_SEED_BUCKETS-1u))],1u);
}
@compute @workgroup_size(64) fn seedCut(@builtin(local_invocation_index) lane:u32){
 let hist=NB_SEED_RANK+nbSeedLevel*NB_SEED_BUCKETS;let per=(NB_SEED_BUCKETS+63u)/64u;
 var s=0u;for(var i=lane*per;i<min(lane*per+per,NB_SEED_BUCKETS);i++){s+=atomicLoad(&bins[hist+i]);}
 nbSeedSums[lane]=s;workgroupBarrier();
 if(lane!=0u){return;}
 var room=0u;
 if(nbSeedLevel==0u){
  if(nbSeedRankIdle()){return;}
  room=nbSeedRoom();
  var total=0u;for(var g=0u;g<64u;g++){total+=nbSeedSums[g];}
  if(total<=room){return;}
  atomicStore(&bins[NB_SEED_CONTROL+2u],1u);
 }else{
  if(atomicLoad(&bins[NB_SEED_CONTROL+2u])==0u){return;}
  room=atomicLoad(&bins[NB_SEED_CONTROL+1u]);
 }
 var below=0u;var g=0u;
 for(;g<63u;g++){if(below+nbSeedSums[g]>room){break;}below+=nbSeedSums[g];}
 var d=g*per;
 for(;d<NB_SEED_BUCKETS-1u;d++){let n=atomicLoad(&bins[hist+d]);if(below+n>room){break;}below+=n;}
 atomicStore(&bins[NB_SEED_CONTROL],(atomicLoad(&bins[NB_SEED_CONTROL])<<NB_SEED_BITS)|d);
 atomicStore(&bins[NB_SEED_CONTROL+1u],room-below);
}
// Admitted: below the cut key, or in the cut cell within the room by site index.
fn nbSeedAdmitted(key:u32,lane:u32)->bool{
 if(atomicLoad(&bins[NB_SEED_CONTROL+2u])==0u){return true;}
 let cut=atomicLoad(&bins[NB_SEED_CONTROL]);
 if(key!=cut){return key<cut;}
 var before=0u;for(var k=lane&~7u;k<lane;k++){before+=nbSeedKept[k];}
 return before<atomicLoad(&bins[NB_SEED_CONTROL+1u]);
}
`;
}
/** A seed's jitter in its eighth of the cell: a hash of the cell, the site
 * (its sample index in the cell) and the salt alone. The reseed's salt is the
 * frame index (params.live.z), so a frame's sites follow from its data, not
 * from the order its seed list was filled in. */
export const narrowBandSeedJitterWGSL=/* wgsl */`
fn nbSeedJitter(c:vec3u,k:u32,salt:u32)->vec3f{
 var h=(c.x*73856093u)^(c.y*19349663u)^(c.z*83492791u)^(k*2654435761u)^(salt*0x9e3779b9u);
 var r=vec3f(0.0);
 for(var a=0u;a<3u;a++){h^=h>>16u;h*=0x7feb352du;h^=h>>15u;h*=0x846ca68bu;h^=h>>16u;r[a]=f32(h&0xffffffu)/16777216.0;}
 return r;
}
`;
export function narrowBandSeparationReach(separation:"off"|"pairs",merge:boolean):number{
 return separation==="pairs"?0.5*Math.cbrt(merge?8:1):0;
}
export const narrowBandRetireCounting={on:false,splitX:-1,splitY:0};
export class UniformNarrowBandFlip implements UniformPageInitBlocks {
 readonly label="Narrow-band FLIP page words";
 readonly capacity:number;
 /** Number of 32-bit words (floats) per particle, not bytes. */
 readonly particleStride:number;
 /** The records' bytes in a bank, its binding (arrayLength) size: a budget's bank buffer can be larger (bankBytes). */
 private readonly recordBytes:number;
 /** A bank buffer's bytes: the records, or a budget's larger bank (the extension companion's). */
 private readonly bankBytes:number;
 private get apic():boolean{return this.transferMode==="apic";}
 readonly particles:readonly [GPUBuffer,GPUBuffer];
 readonly state:GPUBuffer;
 /** The current frame's live sample count (state word 0 after move()'s
  * commit) as a uniform's first word: the particle consumers after the move
  * (transfer, extension retain and recordDelta)
  * exit at once on zero. A GPU copy each move, never a host count. The same
  * word is params word 12 for this module's own kernels. */
 readonly liveGate:GPUBuffer;
 /** The display's motion energy (particleSource.motionEnergy), captured
  * at recordDelta while before.xyz still holds the accepted grid velocity. */
 private readonly displayMotion:GPUFluidParticleMotion;
 private readonly baseAllocatedBytes:number;
 get allocatedBytes():number{return this.baseAllocatedBytes+this.displayMotion.energy.size;}
 /** A word per tile key from word `offset` of the buffer: non-zero where a sample's support reached at its last move (slabSort). */
 get particleTiles():{buffer:GPUBuffer;offset:number;allocation:GPUBuffer}{return {buffer:this.bins,offset:this.countWords+this.tiles,allocation:this.particleAllocation};}
 /** Particle occupancy and signed travel for the census: transferRefine writes them, refine copies them here. */
 private readonly particleAllocation:GPUBuffer;
 count=0;
 reseedClipped=0;
 /** Overlap reseeds deferred at the records' rule, summed over steps. */
 reseedClippedTotal=0;
 /** The records by rule (NarrowBandParticleBudget.records): 0, none. */
 private seedBudget=0;
 /** Positive distances in h cells, before union and on the accepted surface.
  * Unresolved samples are reported separately from the resolved liquid band. */
 diagnostics={beforeMaxOutside:NaN,afterMaxOutside:0,outsideSurface:0,unsupported:0,deepInterior:0};
 /** The separation before the union is a diagnostic alone, a sweep of every
  * sample that nothing in the step reads: a step measures it only for a
  * reader that set this before stepping. Unmeasured, it reports NaN, and so
  * do the accepted surface's census (afterMaxOutside, outsideSurface,
  * deepInterior); unsupported is always counted. */
 measureTraces=false;
 /** Runtime deep-overlap retirement cap; zero disables crowded-cell retirement. */
 particleCellCap=20;
 /** Runtime sparse-retirement threshold (params.live.y); zero disables sparse retirement. */
 sparseNeighbours=16;
 /** Completed frame samples, exposed for transfer/conservation verification.
  * Each brick slab is in cell order up to its seeds, appended after it;
  * holes and retired samples are dead records (heat 0). */
 get activeParticles():GPUBuffer{return this.particles[1]!;}
 /** The scratch bank: the move appends into it and SORT drains it, so
  * from the move's end to the next move it holds nothing. Extension may
  * borrow it across that interval. */
 get extensionScratch():GPUBuffer{return this.particles[0]!;}
 /** The completed frame's samples for the particle layer: positions in h cells,
  * every slab's records counted by the receipt's first word (holes are dead). Eight samples seed a cell,
  * so a sphere is a little under a quarter cell in radius. The move leaves each
  * record the grid velocity it took and its depth; the inner collar is the
  * surface's share of the band. */
 get particleSource():GPUFluidParticleSource{
  const h=this.ownership.capacity.lattice.cellSize_m;
  return {buffer:this.activeParticles,strideFloats:this.particleStride,capacity:this.capacity,positionScale_m:[h[0]!,h[1]!,h[2]!],
   frame:{buffer:this.sourceParams,size:48},radius_m:0.22*Math.min(...h),liveCount:{buffer:this.state,byteOffset:0},motionEnergy:this.displayMotion.energy,
   grid:{velocityFloat:8,depthFloat:7,ballisticFloat:11,ballisticSignBit:!this.apic,surfaceDepth:2}};
 }
 private readonly bins:GPUBuffer;
 /** Minimum spill headroom; each bank also lends its unused slab space to spills. */
 private readonly spill:number;
 private readonly reuse:NarrowBandReuseLayout;
 private readonly reuseControls:GPUBuffer;
 private revisions?:NarrowBandSurfaceRevisions;
 private revisionGeneration?:()=>number;
 private seenRevisionGeneration=-1;
 private reuseParity=0;
 get regionalReuseCapacity():number{return this.reuse.capacity;}
 get regionalReuseAllocatedBytes():number{return 4*this.reuse.words;}
 /** Read-only post-run diagnostics; never used to size simulation launches. */
 get regionalReuseEvidence(){return {buffer:this.bins,regions:this.reuse.capacity?this.reuse.regions:0,flagsOffset:4*this.reuse.nearestFlags};}
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
 /** The stage pipelines compiled at background priority (NB_BACKGROUND_PIPELINES). */
 private readonly background=new GPUBackgroundPipelines();
 /** undefined when every background pipeline this advance may bind is
  * compiled; otherwise their promise, raised to critical priority. The
  * frame's host awaits it before the advance's first queue write. Which air
  * batch clearAir runs follows host flags alone: a frame that runs the
  * stages encodes the shared plan before reconstruct, so its crossings are
  * planned exactly when the plan and the planned search are enabled. */
 pipelinesPending():Promise<void>|undefined{
  const background=this.background;if(background.allReady)return undefined;
  const plan=this.framePlan,planned=this.regionalCrossingPlanEnabled&&plan!==undefined&&plan.enabled;
  const air=!planned?NB_UNPLANNED_AIR:plan!.scheduleStages&&plan!.fuseAirClear?NB_NO_BACKGROUND:NB_PLANNED_AIR;
  return joinPendingPipelines(background.require(air),this.measureTraces?background.require(NB_DIAGNOSE):undefined);
 }
 /** Stage pipelines no batch of this configuration runs, fixed at
  * construction: never compiled. A batch naming one is unavailable and
  * throws if it runs. */
 private omitted?:ReadonlySet<string>;
 private get omittedPipelines():ReadonlySet<string>{
  return this.omitted??=new Set<string>([
   // Every crossing search takes the @cold forms of these.
   "depthTiles","depthSeeds",
   // Reconstruction splats one way: by the plan's occupied list, or cell by cell.
   ...(this.fusedSurface?["surfaceCells","surfaceSplat"]:["surfaceSplatPlan","surfaceSplatOccupied"]),
   // Under previous crossings measureBand runs no search: only the plan's domain launches.
   ...(this.previousCrossings?["depthTiles@cold","depthSeeds@cold",...narrowBandNearestControlEntries,...narrowBandNearestEntries,"depthReuseSelect@withX"]:[]),
   // The canonical surface publishes h vertices itself whenever there are
   // h tiles; with none, redistance stores nothing and its metric is never
   // the derived one (retireSearched is set only by an h-tile retire).
   ...(this.solid.coarse?["arenaStore","arenaGather@derived"]:[]),
  ]);
 }
 private crossingIsPlanned=false;
 private crossingPlanSerial=-1;
 private readonly encodeCrossingDomain=(encoder:GPUCommandEncoder):void=>{
  encoder.clearBuffer(this.bins,4*this.reuse.framePlan,16);
  encoder.clearBuffer(this.bins,4*(this.reuse.framePlan+4+6*this.reuse.regions),4);
  const plan=this.framePlan!;
  encoder.copyBufferToBuffer(plan.buffer,4*plan.layout.flags,this.bins,4*(this.reuse.framePlan+4+4*this.reuse.regions),4*this.reuse.regions);
  const pass=this.crossingPass=encoder.beginComputePass(this.compiled.searchReconstruct.pass);
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
 /** The surface brick arena's windows (uniform-surface-brick-arena.wgsl):
  * 729 vertices a brick-table slot, after the links of binding 3 (no
  * storage binding is spare: the adapter allows ten a stage). */
 private readonly arenaWord:number;
 /** The windows the links buffer holds after arenaWord (NB_ARENA_WINDOWS):
  * every brick of the lattice, or under a particle budget one an h tile with
  * a floor (NB_ARENA_WINDOW_FLOOR). The arena held about a brick an h tile:
  * the letters 2396 bricks at 2466 h tiles, the seep 994 at 931. */
 readonly arenaWindows:number;
 /** The shared brick table's first bins word (uniform-brick-table.ts). */
 private readonly brickWord:number;
 /** transferRefine's prediction for the next census (refine): a join bit a
  * tile key, then PARTICLE_RESIDENCY_WORDS a page, after the brick table. */
 private readonly refineWord:number;
 private readonly refineJoinWords:number;
 /** DEPB's words, in the consumed particle bank when it fits, otherwise
  * after the coarse list: six a fine cell (momentum, then
  * weight, an axis), then a flag a fine tile (uniform-particle-brick.wgsl),
  * for depositTiles fine tiles: every h tile the pool can store unless the
  * device's buffer limit holds fewer. A layout past them takes the general
  * gather (transfer). */
 private readonly depositWord:number;
 private readonly depositTiles:number;
 private readonly sharedDeposit:boolean;
 /** The step's coarse corner list (after the refine words): a count, a
  * build count and two spare words, then coarseListCapacity corner entries
  * and as many build entries. couple, depthCoarse and redistanceCoarse
  * read it instead of visiting every 4h owner of every mapped page. */
 private readonly coarseListWord:number;
 private readonly coarseListCapacity:number;
 /** The frame plan serial whose ownership and tiers built the coarse list (reconstruct), or -1. */
 private coarseListSerial=-1;
 /** The census's fine padding (refine), transferRefine's box margin (params.activity.z). */
 private refinePadding=1;
 private readonly params:GPUBuffer;
 private resources:GPUBindGroupLayout;
 /** The COLD freeze (attachFreeze): the transfer leaves frozen cells and faces. */
 private freeze?:UniformColdFreeze;
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 private groups:Record<string,readonly [UniformDetailGroup,UniformDetailGroup]>={};
 private batches?:NarrowBandBatches;
 private canonicalSurface?:{gather:NarrowBandBatch;far:NarrowBandBatch;publish:NarrowBandBatch;search:NarrowBandBatch};
 /** The lattice's addressing; its cell, vertex and tile keys are the bins' bank offsets. */
 private readonly address:UniformLatticeAddress;
 private readonly cells:number;
 /** Particle brick slots of the cell bins under a budget (0: keyed by cell key), their words, and the count planes' words before NB_COVERAGE. */
 private readonly binSlots:number;
 private readonly binCells:number;
 private readonly countWords:number;
 /** NB_BAND's word: the search's counters, then masks, reach and lists (uniform-narrow-band-membership.wgsl). */
 private readonly bandWord:number;
 /** Bank A's words (NB_DEPTH_A to NB_DEPTH_B): a vertex key each, or the two packed nearest banks where those are longer. */
 private readonly vertices:number;
 private readonly surfaceHotWord:number;
 private readonly tiles:number;
 /** Per-frame parameter records, filled in place. */
 private readonly stepWords=new Float32Array(12);
 private readonly activityWords=this.stepWords.subarray(8,12);
 // params.live.y: live.x is the state's live word, copied in by the GPU.
 private readonly sparseWords=new Uint32Array(1);
 /** params.live.z: the frames moved, the reseed's salt (narrowBandSeedJitterWGSL). */
 private readonly seedFrameWords=new Uint32Array(1);
 /** params.gravity: the step's gravity vector (m/s^2), the ballistic kick. */
 private readonly gravityWords=new Float32Array(4);
 private readonly receiptBits=new Uint32Array(2);
 private readonly receiptDistances=new Float32Array(this.receiptBits.buffer);
 /** The groups' bank order: binding 0 the scratch bank, binding 1 the current. */
 private readonly parity=0;
 private started=false;
 private bandCurrent=false;
 private get activityWord():number{return this.countWords+this.cells+this.vertices+9*this.tiles+4;}
 /** The occupied tile list in the bins: word 0 its length, entries from
  * word 4. A tile key joins it when its coverage (NB_COVERAGE) leaves 0,
  * so the coverage is non-zero only on listed keys; slabPrepare clears
  * exactly the listed keys before the list is emptied. Every per-tile particle
  * launch strides it on a fixed grid: none is sized by the tile keys. */
 private readonly occupiedWord:number;
 /** The far-air gate plane (NARROW_BAND_AIR_GATE_PLANE): a word a tile key,
  * after the activity planes and the air planes before it. A claimed block
  * takes 1 (advect) until the next clear computes it. */
 get airGateWord():number{return this.activityWord+(NARROW_BAND_ACTIVITY_WORDS_PER_TILE+NARROW_BAND_AIR_GATE_PLANE)*this.tiles;}
 /** The bins, which hold the gate plane at airGateWord. */
 get airGateBuffer():GPUBuffer{return this.bins;}
 /** Existing particle-bin coverage, for opt-in spatial cost sampling. */
 get performanceTileCounts():GPUBufferBinding{return {buffer:this.bins,offset:4*this.countWords,size:4*this.tiles};}
 /** The particle activity heat plane (f32 a tile key, NB_ACTIVITY_HEAT) in airGateBuffer: the region heat's particle evidence. */
 get activityHeatWord():number{return this.activityWord+2*this.tiles;}
 /** The surface heat plane (f32 a tile key, NB_ACTIVITY_SURFACE_HEAT) in airGateBuffer: this step's sort writes it before the surface advect. */
 get surfaceHeatWord():number{return this.activityWord+6*this.tiles;}
 private updateErrors?:GPUBuffer;
 /** Bind tile update evidence separately from the shape/refinement census. */
 attachUpdateErrors(buffer:GPUBuffer):void{this.updateErrors=buffer;}
 private readonly coarseVelocity:GPUTexture;
 private cacheGroups:Record<string,UniformDetailGroup>={};
 private deltaResources:GPUBindGroupLayout;
 private deltaPipeline?:GPUComputePipeline;
 private deltaGroup?:UniformDetailGroup;
 private initPipeline?:GPUComputePipeline;
 private initGroup?:GPUBindGroup;
 /** pool: the page pool whose commits claim this lattice's blocks (its state buffer and layout); without one no block is ever claimed. */
 constructor(private readonly device:GPUDevice,private readonly ownership:UniformMixedOwnership,private readonly solid:UniformMixedSolid,private readonly sourceParams:GPUBuffer,private readonly cache:UniformMixedMomentumCache,private readonly pool?:{readonly state:GPUBuffer;readonly layout:UniformPagePoolLayout},readonly transferMode:NarrowBandTransfer="flip",readonly seeding:"need"|"activity"="need",readonly fusedSurface=false,readonly sourceSeeding:"need"|"always"="need",readonly redistanceSearch:"shared"|"fresh"="shared",
  /** Tiles from a crossing tile the arena's redistance reaches (arenaCross's nbReachTiles). */
  readonly redistanceReach:1|2|3=3,
  /** bulkDepth's guard from the last arena search (crossingGuard "previous"):
   * no reconstruct crossing search; the spread reach is copied into the
   * candidate list's words for it (measureBand). */
  readonly previousCrossings=false,
  /** particleBandDepth: cells below the surface the particle band reaches.
   * Seeding fills it and the move retires samples past it. */
  readonly bandDepth=4,
  /** particleDensity "graded": four seed sites a cell (of the eight) in
   * cells deeper than 1.5 cells, a crowding cap of six there, and faces
   * below a cell expect half the mass. */
  readonly gradedDensity=false,
  /** The particles' h capacity (scene fluid.tiers); absent, the pool's cell keys size them. */
  readonly budget?:NarrowBandParticleBudget,
  /** outerSeeding "uncovered": the outer layer (under a cell deep) of a
   * heated tile is also seeded where no sample lies in the cell or the 26
   * around it. There the union keeps the advected level set as it is
   * (particleSurfaceFinish's empty gather), so the seeds add no liquid; they
   * carry liquid that reached the surface without samples (a jet leaving a
   * full throat at ten cells a step) from the next step. "retain" seeds it
   * only where a tile is newly heated or a source adds liquid. */
  readonly outerSeeding:"retain"|"uncovered"="retain",
  /** heatExpiry "held": a sample whose heat runs out is retired only where
   * the level set holds its liquid: a cell deep or more, or in a tile whose
   * surface the census finds resolved (not thin). A shallow sample in a thin
   * tile keeps heat 1 (support without erasure). "anywhere" retires it
   * wherever its tile has cooled. */
  readonly heatExpiry:"anywhere"|"held"="anywhere",
  /** sampleBound "walls": the samples' mass bound in the union holds a
   * wall or floor vertex's plane against the wall (NARROW_BAND_WALL_BOUND_WGSL);
   * "centred" places every vertex at its open cells' centre. */
  readonly sampleBound:"centred"|"walls"="centred",
  /** crowdedCells "merge": a deep cell's samples past particleCellCap fold
   * into its kept samples as mass (nbMergeRun), and a merged sample splits
   * where it reaches a shallow or under-filled cell (nbFinish); "retire"
   * retires them. */
  readonly crowdedCells:"retire"|"merge"="retire",
  /** sampleSeparation "pairs": the move pushes two samples closer than their
   * radii' sum apart (Sato et al. 2018 Section 3.2.2, nbSeparation), so the
   * samples a film carries keep their spacing against a floor instead of
   * packing into its bottom quarter cell; "off" moves each alone. */
  readonly sampleSeparation:"off"|"pairs"="off",
  /** innerReseed "cells": the inner band [-R,-h] is replenished only in cells
   * that lie wholly inside it (centre 1.5 cells down), where the cell's
   * sample count measures that band; "sites" also tops up a cell whose
   * centre is shallower wherever one of its seed sites is a cell deep. */
  readonly innerReseed:"sites"|"cells"="sites",
  /** activationSeeding "uncovered" (with outerSeeding "uncovered"): a newly
   * heated tile seeds its whole band only where no sample lies within a cell
   * (nbUncovered), and elsewhere as any heated tile; "band" seeds the whole
   * band of every newly heated cell under its target count, on top of the
   * samples already carrying its liquid. */
  readonly activationSeeding:"band"|"uncovered"="band"){
  const address=this.address=ownership.capacity.address,cells=address.cellKeys,vertices=Math.max(address.vertexKeys,address.cellKeys/2);
  // The seed list packs a cell key in 30 bits below the source and deep flags.
  if(cells>0x40000000)throw new Error(`Narrow-band FLIP's seed list packs a cell key in 30 bits: ${cells} cell keys exceed 1073741824`);
  if(!(bandDepth>=2&&bandDepth<=4))throw new Error(`Narrow-band FLIP particle band depth ${bandDepth} is outside 2-4 cells`);
  if(pool&&512*pool.layout.blocks!==address.tileKeys)throw new Error(`Narrow-band FLIP: the page pool has ${pool.layout.blocks} blocks, the lattice ${address.tileKeys/512}`);
  // The particle cell bins (counts in bins, starts in links): a word a cell
  // key; under a budget, 512 a particle brick slot (the sort's slab slot,
  // nbBinTile) and a slot word a brick id after them.
  // Under records by rule every stored brick may hold particles (spray
  // scatters them): its bins hold every brick, the hard bound.
  this.binSlots=budget?Math.min(64*address.blocks,budget.records!==undefined?Infinity:budget.fineTiles):0;this.binCells=budget?512*this.binSlots:cells;
  this.countWords=budget?this.binCells+64*address.blocks:2*cells;
  this.cells=cells;this.vertices=vertices;this.tiles=address.tileKeys;this.bandWord=this.countWords+cells+vertices+2*this.tiles;
  // The band is a surface, but what a scene's surface grows to is not known
  // here. Budget one sample per domain cell, with a two-million floor: twice
  // the previous allocation, still capped by full-domain seeding and the
  // device's per-buffer limit at the original record size: packing must not
  // silently change the maximum sample count on a smaller adapter. A full budget defers optional reseeding and
  // is fatal to liquid a source or the initial band must add (noteReceipt).
  this.particleStride=narrowBandParticleStride(this.apic);const particleBytes=4*this.particleStride;
  const deviceRecords=Math.floor(device.limits.maxStorageBufferBindingSize/(this.apic?112:64));
  if(budget&&!(Number.isInteger(budget.fineTiles)&&budget.fineTiles>0&&budget.perFineCell>0))throw new Error(`Narrow-band FLIP particle budget ${JSON.stringify(budget)} needs a positive integer of fine tiles and a positive density`);
  if(budget&&narrowBandBudgetCapacity(budget,address.blocks)>deviceRecords)throw new Error(`Narrow-band FLIP: fluid.tiers (${budget.records!==undefined?`${budget.records} records by rule`:`${budget.fineTiles} h tiles at ${budget.perFineCell} particles a fine cell`}) needs ${narrowBandBudgetCapacity(budget,address.blocks)} particle records; the device's binding limit holds ${deviceRecords}`);
  this.seedBudget=budget?.records??0;
  this.capacity=budget?narrowBandBudgetCapacity(budget,address.blocks):Math.min(cells*8,Math.max(2_097_152,cells),deviceRecords);
  const buffer=(label:string,size:number,uniform=false)=>device.createBuffer({label:`Narrow-band FLIP ${label}`,size,usage:(uniform?GPUBufferUsage.UNIFORM|GPUBufferUsage.STORAGE:GPUBufferUsage.STORAGE)|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  this.recordBytes=this.capacity*particleBytes;const bankBytes=this.bankBytes=Math.max(this.recordBytes,budget?.bankBytes??0);
  this.particles=[buffer("particles A",bankBytes),buffer("particles B",bankBytes)];
  // The receipt, then the slab records (uniform-particle-slab.wgsl).
  this.state=buffer("receipt, particle frame and brick slabs",4*particleSlabWords(address.blocks));
  this.displayMotion=new GPUFluidParticleMotion(device,this.activeParticles,this.state,this.particleStride,this.capacity);
  this.spill=particleSlabSpill(this.capacity);
  // After the air planes: the occupied tile list (occupiedWord).
  this.occupiedWord=this.countWords+cells+vertices+address.tileKeys*(9+NARROW_BAND_ACTIVITY_WORDS_PER_TILE+NARROW_BAND_AIR_WORDS_PER_TILE)+4;
  const binWords=this.occupiedWord+4+address.tileKeys;
  const limit=Math.min(device.limits.maxStorageBufferBindingSize,device.limits.maxBufferSize);
  // Under a particle budget the payload slots follow the planned regions
  // (narrowBandReuseLayout): about one in four of its h tiles' count was
  // measured (seep 4h: 374 regions planned at 1451 h tiles), so half of them,
  // with a floor; a region the slots miss searches afresh.
  const regions=8*address.blocks,slots=budget?Math.max(NB_REUSE_SLOT_FLOOR,Math.ceil(budget.fineTiles/2)):regions;
  const completeReuse=slots<regions?narrowBandReuseLayout(binWords,regions,slots,true):narrowBandReuseLayout(binWords,regions);
  this.reuse=4*(binWords+completeReuse.words+address.vertexKeys)<=limit?completeReuse:narrowBandReuseLayout(binWords,8*address.blocks,0);
  this.surfaceHotWord=binWords+this.reuse.words;
  // The shared brick table (uniform-brick-table.ts) persists after the surface-hot words.
  this.brickWord=this.surfaceHotWord+address.vertexKeys;
  this.refineWord=this.brickWord+uniformBrickTableWords(address.blocks);this.refineJoinWords=Math.ceil(address.tileKeys/32);
  // A canonical 4h corner has one owner: a block's 9³ coarse vertices bound
  // them, twice over for a corner the two rules (couple's, retirement's) see.
  this.coarseListWord=this.refineWord+this.refineJoinWords+8*address.blocks*PARTICLE_RESIDENCY_WORDS;this.coarseListCapacity=2*729*address.blocks;
  // The fine-tile capacity can grow between frames up to the pool's tiles
  // (reserveFine), and the pour's late frames hold over 12k: the deposit's
  // words are sized for the pool, so no layout leaves the fast deposit.
  // Under a particle budget its h tiles bound the deposit: a layout past them takes the general gather.
  // The deposit fits the scratch bank by the buffer's bytes, not its records'
  // (a budget's bank is the companion's): SORT has drained it, and the tail
  // extension's companion is written only after the transfer reads it.
  const deposit=particleDepositStorage(this.coarseListWord+4+2*this.coarseListCapacity,budget?Math.min(budget.fineTiles,address.poolTiles):address.poolTiles,limit,bankBytes);
  this.depositWord=deposit.offsetWords;this.depositTiles=deposit.tiles;this.sharedDeposit=deposit.shared;
  const binBytes=deposit.binsBytes;
  if(binBytes>limit)throw new Error(`Narrow-band bins and bounded crossing reuse (${binBytes} bytes) exceed GPU buffer limit ${limit}`);
  this.bins=buffer("cell counts, surface metrics and persistent regional crossing masks",binBytes);
  this.reuseControls=buffer("regional reuse pass controls",72);device.queue.writeBuffer(this.reuseControls,0,new Uint32Array([0,1,1,1,1,1,0,1,0,1,1,0,0,0,0,1,0,0]));
  // The arena's windows follow: a window (9³ vertices) an arena brick (ubWindow).
  this.arenaWord=this.binCells+7*this.capacity;this.arenaWindows=Math.min(64*address.blocks,budget&&budget.records===undefined?Math.max(NB_ARENA_WINDOW_FLOOR,budget.fineTiles):Infinity);
  const linkBytes=4*(this.arenaWord+729*this.arenaWindows);
  if(linkBytes>limit)throw new Error(`Narrow-band links and arena windows (${linkBytes} bytes) exceed GPU buffer limit ${limit}`);
  this.next=buffer("cell starts, compact neighbor samples and surface brick arena windows",linkBytes);this.params=buffer("parameters",80,true);
  this.liveGate=buffer("live sample gate",16,true);device.queue.writeBuffer(this.liveGate,0,new Uint32Array([1,0,0,0]));
  this.particleAllocation=buffer("particle allocation support",8*address.blocks*PARTICLE_RESIDENCY_WORDS*4);
  const size=uniformCoarseCacheExtent(address);
  this.coarseVelocity=device.createTexture({label:"Narrow-band FLIP stage sampling cache",size,dimension:"3d",format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING});
  this.baseAllocatedBytes=[...this.particles,this.state,this.bins,this.next,this.params,this.liveGate,this.particleAllocation,this.reuseControls].reduce((n,b)=>n+b.size,0)+size.reduce((n,v)=>n*v,16);
  this.resources=this.resourceLayout(false);
  this.deltaResources=this.deltaLayout(false);
 }
 /** The stage resources; with the freeze, its region snapshot at binding 18. */
 /** recordDelta's: the samples (1) and their bins (2, 3, 4), the params (6),
  * the frame extension's delta pool (19); with the freeze, its regions (18). */
 private deltaLayout(freeze:boolean):GPUBindGroupLayout{
  return uniformDetailBindLayout(this.device,{entries:[
   ...(freeze?[{binding:18,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"2d" as const}}]:[]),
   ...[1,2,3,4].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}})),
   {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   {binding:19,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
  ]});
 }
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
  ]},[12,22]);
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
  * transfer, recordDelta and refine are not encoded. activity is: its
  * plane is the region heat's particle evidence, which ends an idle run. */
 private quiet=false;
 private particleGroups=NARROW_BAND_PARTICLE_GRID;
 private idleNow=false;
 get idle():boolean{return this.idleNow;}
 beginAdvance(cold:UniformColdFrame):void{
  const quiet=cold.quiet;
  // Only launch parallelism changes, never the GPU count or seeding policy.
  // HOT/unknown frames retain the populated-throughput grid. The smaller
  // direct grid still strides every sample if work grows before its receipt.
  this.particleGroups=cold.ceiling<UNIFORM_REGION_TIER.hot?NARROW_BAND_QUIET_PARTICLE_GRID:NARROW_BAND_PARTICLE_GRID;
  // this.quiet is still the last advance's here.
  this.idleNow=quiet&&this.quiet&&this.freeze!==undefined&&this.started;
  this.quiet=quiet;
 }
 attachFreeze(freeze:UniformColdFreeze|undefined):void{
  if(Object.keys(this.groups).length||this.freeze)throw new Error("Narrow-band FLIP: the freeze attaches once, before bind");
  this.freeze=UNIFORM_COLD_FREEZE?freeze:undefined;
  if(this.freeze){this.resources=this.resourceLayout(true);this.deltaResources=this.deltaLayout(true);}
 }
 bind(fields:{phi:GPUTexture;phiScratch:GPUTexture;phase:GPUTexture;velocity:GPUTexture;velocityScratch:GPUTexture;departure:GPUTexture;negative:GPUBuffer;negativeScratch:GPUBuffer;negativeDeparture:GPUBuffer;coarseExtended:GPUTexture;unitExtended:GPUTexture;hanging:UniformMixedHangingTaps;extensionDelta:GPUBuffer}):void{
  const f=fields;
  // The transfer writes the stored velocity: momentum was its last reader,
  // and the pressure split rewrites every face of it after the forces.
  // recordDelta reads the frame extension's delta pool, which its pair
  // tail writes (UniformFrameExtension).
  this.deltaGroup=uniformDetailGroup(this.device,{layout:this.deltaResources,entries:[
   {binding:1,resource:{buffer:this.activeParticles,size:this.recordBytes}},{binding:2,resource:{buffer:this.bins}},{binding:3,resource:{buffer:this.next}},{binding:4,resource:{buffer:this.state}},
   {binding:6,resource:{buffer:this.params}},{binding:19,resource:{buffer:f.extensionDelta}},
   ...(this.freeze?[{binding:18,resource:this.freeze.regions.createView()}]:[]),
  ]});
  for(const [name,velocity,negative,output] of [["bootstrap",f.velocityScratch,f.negativeScratch,f.departure],["advectParticles",f.velocityScratch,f.negativeScratch,f.departure],["transfer",f.departure,f.negativeDeparture,f.velocity],["surface",f.velocity,f.negative,f.departure],["reconstruct",f.velocity,f.negative,f.departure],["redistance",f.velocity,f.negative,f.departure],["redistanceStore",f.velocity,f.negative,f.departure]] as const){
   this.cacheGroups[name]=this.cache.bind({extended:velocity,negative,coarseExtended:this.coarseVelocity});
   this.groups[name]=[0,1].map(parity=>uniformDetailGroup(this.device,{layout:this.resources,entries:[
    // Binding 0 spans the bank when it holds the shared deposit (depositScratch aliases it; no shader reads source's length).
    {binding:0,resource:{buffer:this.particles[parity]!,size:this.sharedDeposit?this.bankBytes:this.recordBytes}},{binding:1,resource:{buffer:this.particles[1-parity]!,size:this.recordBytes}},
    {binding:2,resource:{buffer:this.bins}},{binding:3,resource:{buffer:this.next}},{binding:4,resource:{buffer:this.state}},
    {binding:5,resource:{buffer:negative}},{binding:6,resource:{buffer:this.params}},
    {binding:7,resource:name==="reconstruct"||name==="redistanceStore"?f.phiScratch:f.phi},{binding:8,resource:velocity},{binding:9,resource:output},
    {binding:10,resource:name==="advectParticles"?f.coarseExtended:this.coarseVelocity},
    {binding:11,resource:f.unitExtended},{binding:12,resource:f.hanging.haloOf(f.unitExtended)},{binding:14,resource:name==="reconstruct"||name==="redistanceStore"?f.phi:f.phiScratch},
    {binding:15,resource:f.phase},
    {binding:16,resource:{buffer:this.sourceParams,size:UNIFORM_PARAMS_BYTES}},
    ...(this.freeze?[{binding:18,resource:this.freeze.regions.createView()}]:[]),
   ]})) as [UniformDetailGroup,UniformDetailGroup];
  }
 }
 async initialize():Promise<void>{
  await this.displayMotion.initialize();
  const subgroups=this.device.features.has("subgroups")&&(this.device.adapterInfo.subgroupMinSize??0)>=32;
  await this.cache.initialize();
  if(this.pool){
   // The activity heat and target are the two words a tile key carries from step to step: a claimed block's keys take 0, a tile never active.
   const heat=this.activityWord+2*this.tiles,module=this.device.createShaderModule({label:"Narrow-band FLIP page words",code:/* wgsl */`${uniformPagePoolWGSL(this.pool.layout,0,0)}
@group(0) @binding(1) var<storage,read_write> bins:array<u32>;
${uniformPageInitBlocksWGSL(this.pool.layout,"nbInitBlocks",512,`bins[${heat}u+key]=0u;bins[${heat+this.tiles}u+key]=0u;bins[${this.airGateWord}u+key]=1u;${this.previousCrossings?`bins[${this.bandWord+4+2*this.tiles}u+key]=0u;bins[${this.bandWord+4}u+2u*key]=0u;bins[${this.bandWord+5}u+2u*key]=0u;`:""}`)}`});
   const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
   if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
   const layout=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
   this.initGroup=this.device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:this.pool.state}},{binding:1,resource:{buffer:this.bins}}]});
   this.initPipeline=await this.device.createComputePipelineAsync({layout:this.device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint:"nbInitBlocks"}});
  }
  const sourceCode=(unitTaps:boolean,delta=false)=>uniformMixedCertifiedEntriesWGSL((this.device.features.has("subgroups")?"enable subgroups;\n":"")+uniformMixedTopologyWGSL(this.ownership.capacity,0)+uniformStageListsWGSL+/* wgsl */`
const nbSparseBins:bool=true;
const NB_PREVIOUS_CROSSINGS:bool=${this.previousCrossings};
${narrowBandParticleWGSL(this.apic,this.crowdedCells==="merge")}
@group(1) @binding(0) var<storage,read_write> source:array<Particle>;
// Disjoint entry points alias binding 0: only transferClear/Deposit/Properties
// access the atomic view, after SORT consumed source and before tail extension.
${this.sharedDeposit?"@group(1) @binding(0) var<storage,read_write> depositScratch:array<atomic<u32>>;":""}
@group(1) @binding(1) var<storage,read_write> particles:array<Particle>;
@group(1) @binding(2) var<storage,read_write> bins:array<atomic<u32>>;
@group(1) @binding(3) var<storage,read_write> links:array<u32>;
@group(1) @binding(4) var<storage,read_write> state:array<atomic<u32>>;
@group(1) @binding(5) var<storage,read> negative:array<f32>;
struct Params{hDt:vec4f,settings:vec4f,activity:vec4f,live:vec4u,gravity:vec4f}
@group(1) @binding(6) var<uniform> params:Params;
// This frame's live sample count (move's commit): a uniform, so a kernel
// with barriers may exit on it.
fn nbParticlesLive()->bool{return params.live.x!=0u;}
@group(1) @binding(7) var phi:texture_3d<f32>;
@group(1) @binding(8) var velocity:texture_3d<f32>;
@group(1) @binding(9) var output:texture_storage_3d<rgba32float,write>;
@group(1) @binding(10) var coarseVelocity:texture_3d<f32>;
@group(1) @binding(11) var unitVelocity:texture_3d<f32>;
${unitTaps?"@group(1) @binding(12) var unitHalo:texture_3d<f32>;":""}
const NB_ARENA:u32=${this.arenaWord}u;
const NB_ARENA_WINDOWS:u32=${this.arenaWindows}u;
// The particle cell bins: a cell's count at bins[nbBinCell(cell)] and its
// start at links[nbBinCell(cell)]; NB_LINKS words of starts head the links.
// Under a particle budget a covered tile's 64 bins are in its brick's slot
// (the sort's slab slot, NB_SLOT_MAP a brick id); an uncovered tile has none.
const NB_LINKS:u32=${this.binSlots?`${this.binCells}u`:"UM_CELL_KEYS"};const NB_NO_BIN:u32=0xffffffffu;
${this.binSlots?`const NB_SLOT_MAP:u32=${this.binCells}u;
fn nbBinTile(tile:u32)->u32{return 512u*atomicLoad(&bins[NB_SLOT_MAP+ubBrickOfTile(tile)])+64u*((tile&1u)|(((tile>>3u)&1u)<<1u)|(((tile>>6u)&1u)<<2u));}
fn nbBinCell(cell:u32)->u32{if(atomicLoad(&bins[NB_COVERAGE+cell/64u])==0u){return NB_NO_BIN;}return nbBinTile(cell/64u)+cell%64u;}
fn nbBinCount(cell:u32)->u32{let bin=nbBinCell(cell);if(bin==NB_NO_BIN){return 0u;}return atomicLoad(&bins[bin]);}`
:`fn nbBinTile(tile:u32)->u32{return 64u*tile;}
fn nbBinCell(cell:u32)->u32{return cell;}
fn nbBinCount(cell:u32)->u32{return atomicLoad(&bins[cell]);}`}
@group(1) @binding(14) var outputPhi:texture_storage_3d<r32float,read_write>;
@group(1) @binding(15) var phase:texture_3d<f32>;
${UniformColdFreeze.wgsl(this.freeze,1,18)}
${this.updateErrors?`// Packed certificate copy: reuse the bins binding (the full FLIP layout is at its buffer limit).
const NB_UPDATE_ERRORS:u32=${this.activityWord+7*this.tiles}u;
// The tile's deformation: its velocity change across one tile, m/s.
fn tuDeformation(t:u32)->f32{
 let at=NB_UPDATE_ERRORS+4u*t;
 if(!umKeyStored(t)||atomicLoad(&bins[at+2u])!=umBlockWord(t>>9u)||bitcast<f32>(atomicLoad(&bins[at+3u]))<=0.0){return 3.402823e38;}
 return bitcast<f32>(atomicLoad(&bins[at]));
}`:"fn tuDeformation(t:u32)->f32{return 3.402823e38;}"}
// A thawing region's coarse term on an h cell's three positive faces.
fn nbThaw(cell:vec3u)->vec3f{return vec3f(umThawFaceTerm(vec3i(cell),0u,1u),umThawFaceTerm(vec3i(cell),1u,1u),umThawFaceTerm(vec3i(cell),2u,1u));}
// SORT leaves each brick slab's samples sorted by cell key, a tile's 64
// cells together and x fastest inside it. A cell's samples are one contiguous run
// from its start, and the cells of an x row inside a tile follow one another,
// so a gather reads memory in order and follows no list. The bins hold each
// cell's count at its key; the head of the link arena holds its start.
// The samples [x,y) of cells a to b, two cells of one x row of one tile.
// A tile that holds no storage has no samples: the run is empty.
fn nbCellLocal(c:vec3i)->u32{let l=vec3u(c)&vec3u(3u);return l.x|(l.y<<2u)|(l.z<<4u);}
// The move's window holds a tile's first bin (nbWinBin: NB_NO_BIN where
// no run is), the page, coverage and slot reads below.
fn nbRun(a:vec3i,b:vec3i)->vec2u{
 var bin:u32;let w=nbWinIndex(a>>vec3u(2u));
 if(w<64u){bin=nbWinBin[w];if(bin==NB_NO_BIN){return vec2u(0);}}
 else{
  let tile=umTileKey(a>>vec3u(2u));if(!umKeyStored(tile)){return vec2u(0);}
  if(atomicLoad(&bins[NB_COVERAGE+tile])==0u){return vec2u(0);}
  bin=nbBinTile(tile);
 }
 let last=bin+nbCellLocal(b);return vec2u(links[bin+nbCellLocal(a)],links[last]+atomicLoad(&bins[last]));}
fn nbCellRun(c:vec3i)->vec2u{return nbRun(c,c);}
// Position and motion remain compact streams for the neighbor gathers.
// Reconstruction reads position.w's heat to restrict erasure to its nearest
// represented patch. The deposit (DEPB) borrows only its sign bit for mode,
// leaving three full-precision motion floats per particle instead of a second vec4.
const NB_SURFACE_HOT_NEAREST=${this.surfaceHotWord}u;
const NB_COARSE_LIST:u32=${this.coarseListWord}u;const NB_COARSE_CAP:u32=${this.coarseListCapacity}u;
const NB_DEPOSIT:u32=${this.depositWord}u;const NB_DEPOSIT_TILES:u32=${this.depositTiles}u;
fn nbSurfaceParticleHot(i:u32)->bool{return bitcast<f32>(links[NB_LINKS+4u*i+3u]&0x7fffffffu)>1.0;}
fn nbPosition(i:u32)->vec3f{let a=NB_LINKS+4u*i;return bitcast<vec3f>(vec3u(links[a],links[a+1u],links[a+2u]));}
// A sample's mass from the compact flag word (the deposit writes it).
fn nbStreamMass(i:u32)->f32{return ${this.crowdedCells==="merge"?"f32((links[NB_LINKS+4u*i+3u]&7u)+1u)":"1.0"};}
fn nbMotion(i:u32)->vec4f{let a=NB_LINKS+4u*arrayLength(&particles)+3u*i;let mode=links[NB_LINKS+4u*i+3u]>>31u;return vec4f(bitcast<vec3f>(vec3u(links[a],links[a+1u],links[a+2u])),f32(mode));}
// Live particle counts per tile drive sparse cell-bin clearing.
const NB_COVERAGE:u32=${this.binSlots?`${this.binCells}u+64u*UM_BLOCKS`:"2u*UM_CELL_KEYS"};
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
${delta?uniformFrameExtensionDeltaWGSL(this.ownership.capacity.address.blocks,this.ownership.capacity.fineTiles)+NB_DELTA_RECORD_WGSL:""}
fn umLoadMixedFace(anchor:vec3i,axis:u32)->f32{
 if(umFaceLow(anchor,axis)){return negative[umLowFaceIndex(umLowFaceAbove(anchor,axis),axis)];}
 return textureLoad(velocity,anchor,0)[axis];
}
${uniformMixedVelocitySamplingSource(false,"coarseVelocity","velocity",undefined,unitTaps?"unitVelocity":undefined,undefined,unitTaps?"unitHalo":undefined)}
${uniformMixedSolidLibraryWGSL(this.solid,2)}
${uniformMixedVertexBuriedWGSL}
${narrowBandTerrainContactSource("nbMoveBodyAt(q)")}
${narrowBandRigidContactSource("nbMoveBodyMask(point)")}
${uniformMixedSourceWGSL(16)}
// Every particle uses the same compiled general interpolant.
fn sampleVelocity(p:vec3f)->vec3f{return umSampleVelocity(p);}
// The move's sample: umSampleVelocity's value, whose fine tile reads its
// class and halo home from the job's window (staged once a job) in place of
// the page walk, tile word and halo lookups per sample.
fn nbMoveSample(p:vec3f)->vec3f{
 ${unitTaps?`umHold=umPointBounds(p,8.0);let q=clamp(p,umHold.lo,umHold.hi);let tile=umBoundedCell(q,umHold)>>vec3u(2u);
 let w=nbWinIndex(tile);var s=UMSampleTile(q,tile,0u);var fine=false;var weights=1.0;
 if(w<64u&&!umRegularFine&&(nbWinFlags[w]&NB_WIN_FINE)!=0u){fine=true;umUnitHome=nbWinHome[w];}
 else{
  s.key=umTileAt(vec3u(tile));fine=!umRegularFine&&umCachedFineTile(s.key);umUnitHome=umUnitHaloHome(s);
  if(!fine){weights=umVelocitySamplingWeightsAt(s);}
 }
 var v=vec3f(0.0);
 for(var a=0u;a<3u;a++){v[a]=umSampleVelocityAxis(p,s,fine,weights,a);}
 return v;`:"return sampleVelocity(p);"}
}
${this.apic?/* wgsl */`
// Extend the velocity at domain walls without changing the quadratic moments.
// Interior solid values come from the same extrapolated field as FLIP.
${unitTaps?`// A particle of a unit tile reads its taps at plain offsets from its page's
// halo block (every tap is then in a unit or a slotted tile, whose halo
// texel is umVelocityTap1's value: channel w); a tap the halo does not hold
// repeats the sample through umVelocityTap1.
var<private> nbApicHome:vec4i=vec4i(-1);
var<private> nbApicValid:bool=true;`:""}
fn nbApicTap(index:vec3i,axis:u32)->f32{
 var low=vec3i(umHold.lo);low[axis]-=1;
 let held=clamp(index,low,vec3i(umHold.hi)-1);
 ${unitTaps?"if(nbApicHome.w>=0){let t=textureLoad(unitHalo,held+nbApicHome.xyz,0);nbApicValid=nbApicValid&&t.w>0.0;return t[axis];}":""}
 return umVelocityTap1(held,axis);
}
${narrowBandApicMomentWGSL}
fn nbApicSample(p:Particle)->Particle{
 umHold=umPointBounds(p.position.xyz,8.0);
 ${unitTaps?`let s=umSampleTileHeld(p.position.xyz);nbApicHome=select(vec4i(-1),umUnitHaloHome(s),umTileWidth(s.key)==1u);
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
${uniformSolidPointOpenWGSL}
${uniformSolidSlideStartWGSL}
${uniformSolidPointCellWGSL}
${narrowBandRedistanceWGSL}
${uniformBrickTableWGSL("bins",`${this.brickWord}u`)}
${narrowBandArenaWGSL}
${this.solid.coarse?narrowBandArenaCanonicalWGSL:""}
${narrowBandActivityTriggerWGSL(this.seeding,this.sourceSeeding)}
// The particle band (particleBandDepth, particleDensity).
const NB_BAND_DEPTH:f32=${this.bandDepth.toFixed(2)};${narrowBandRetireCounting.on?`
fn nbRetireCount(reason:u32,depth:f32,x:f32,mass:f32){
 atomicAdd(&state[${NARROW_BAND_RETIRE_WORD}u+reason],1u);
 ${narrowBandRetireCounting.splitX>=0?"":`if(mass>1.0){atomicAdd(&state[${NARROW_BAND_RETIRE_WORD+11}u],u32(mass)-1u);}`}
 ${narrowBandRetireCounting.splitX>=0?`if(x>=${narrowBandRetireCounting.splitX.toFixed(1)}&&reason==0u){atomicAdd(&state[${NARROW_BAND_RETIRE_WORD+5}u+select(0u,1u,x<${(narrowBandRetireCounting.splitX+16).toFixed(1)})],u32(round(mass)));}`:`if(depth> -1.0&&(reason==1u||reason==3u||reason==4u)){atomicAdd(&state[${NARROW_BAND_RETIRE_WORD+5}u+select(select(2u,1u,reason==3u),0u,reason==1u)],1u);}`}
}`:""}
const NB_GRADED:bool=${this.gradedDensity};${narrowBandRetireCounting.on?`
fn nbMergeCount(k:u32,n:u32){${narrowBandRetireCounting.splitX>=0?"":`atomicAdd(&state[${NARROW_BAND_RETIRE_WORD+8}u+k],n);`}}`:""}${this.crowdedCells==="merge"?`
// crowdedCells "merge": the mass a sample may hold (the deposit's fixed-point
// sums keep their headroom).
const NB_MASS_MAX:f32=8.0;
// The cap of a deep cell (centre depth): six in a graded band's cells past 1.5.
fn nbCellCap(depth:f32)->u32{return select(u32(params.activity.w),min(u32(params.activity.w),6u),NB_GRADED&&depth<=-1.5);}
// Rank r of a crowded run gathers ranks r+cap, r+2cap, ... (low, high: the
// first and the run's end; stride: the cap) as one sample of their summed
// mass at their mass-weighted position, velocity and FLIP delta (a
// delta absent on an axis for any is absent for the merged one), the most
// heat and the least age. A sample whose mass would pass NB_MASS_MAX is not
// gathered: it stays live, holding the cell over its cap (nbMergeAbsorbed,
// counted). The gathered records are retired unmoved by their own lanes, so
// their slots hold this step's state.
fn nbMergeRun(record:Particle,low:u32,high:u32,stride:u32)->Particle{
 var m=nbParticleMass(record);var x=m*record.position.xyz;var v=m*record.velocity.xyz;var b=m*record.before.xyz;
 var none=abs(record.before.xyz)>=vec3f(1e30);var heat=record.position.w;var age=nbParticleAge(record);
 for(var j=low;j<high;j+=stride){
  let o=particles[j];if(!nbLive(o)){continue;}
  let mo=nbParticleMass(o);
  if(m+mo>NB_MASS_MAX){continue;}
  m+=mo;x+=mo*o.position.xyz;v+=mo*o.velocity.xyz;b+=mo*o.before.xyz;none=none|(abs(o.before.xyz)>=vec3f(1e30));
  heat=max(heat,o.position.w);age=min(age,nbParticleAge(o));
  ${narrowBandRetireCounting.on?"nbMergeCount(0u,1u);":""}
 }
 var p=record;p.position=vec4f(x/m,heat);p.velocity=vec4f(v/m,record.velocity.w);
 p.before=vec4f(select(b/m,vec3f(3e38),none),nbParticleState(age,nbParticleMode(record)));
 return nbWithMass(p,m);
}
// Whether the gatherer of excess rank i (start: the run's first; stride: the
// cap) takes it: the same walk nbMergeRun makes, up to i.
fn nbMergeAbsorbed(start:u32,i:u32,stride:u32)->bool{
 let k=start+(i-start)%stride;var m=nbParticleMass(particles[k]);
 for(var j=k+stride;j<i;j+=stride){
  let o=particles[j];if(!nbLive(o)){continue;}
  let mo=nbParticleMass(o);if(m+mo<=NB_MASS_MAX){m+=mo;}
 }
 return m+nbParticleMass(particles[i])<=NB_MASS_MAX;
}`:""}
fn nbExpectedMass(depth:f32)->f32{return ${narrowBandExpectedMassWGSL("depth",this.gradedDensity,this.crowdedCells==="merge")};}
// The samples a cell holds when seeded full: four of the eight sites in a
// graded band's deep cells (centre depth), else eight.
fn nbSeedTarget(depth:f32)->u32{return select(8u,4u,NB_GRADED&&depth<=-1.5);}
// A graded deep cell's j-th site: the even sites of the 2x2x2 lattice.
fn nbSeedSiteIndex(j:u32,depth:f32)->u32{if(nbSeedTarget(depth)==8u){return j;}return array<u32,4>(0u,3u,5u,6u)[j&3u];}
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
  if(gridLiquidCell(home,page,base+vec3i(bit))){return true;}
 }
 return false;
}
// Whether cell c, read from a sample in page, lies in a liquid owner.
fn gridLiquidCell(home:u32,page:vec3i,c:vec3i)->bool{
 let owner=umOwnerAt(umCellNearFromPage(home,page,c));return textureLoad(phase,vec3i(umOrigin(owner)),0).x>0.5;
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
// A sample at q, in a page of the box that holds no storage: state[3] takes
// 4 and the sticky state[11] the page, 0x80000000 | x | y<<6 | z<<12 in pages
// of 32 cells (the greatest raised). The host stops the run (noteReceipt);
// the sample is not clamped into a stored cell or counted in one.
fn nbArrived(q:vec3f){
 let page=vec3u(q)>>vec3u(5u);atomicStore(&state[3],4u);atomicMax(&state[11],0x80000000u|(page.x&255u)|((page.y&255u)<<8u)|((page.z&255u)<<16u));
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
${particleSlabWGSL(this.capacity-this.spill,this.spill,this.binSlots,this.seedBudget)}
override nbAdvectScalarCollision:bool=false;
// With a static solid in the scene the move skips the collision walk of a
// sample whose trace has no closed cell and no cell the procedural field
// cuts in its box (every stage and end it reaches) while the scene holds no
// rigid body or terrain: every other cell is whole or closed, the walk
// moves only through cells inside that box, blocks nothing there and the
// field's contact finds no hit (its value is positive throughout every
// other cell). The proof is the record's open-cell and field-cut bits
// (umSolidOpenCells, umSolidFieldCut) of the box's tiles, all stored.
// nbSolidRouting off walks every sample.
override nbSolidRouting:bool=true;
var<private> nbHullLo:vec3f;
var<private> nbHullHi:vec3f;
fn nbHullInclude(x:vec3f){nbHullLo=min(nbHullLo,x);nbHullHi=max(nbHullHi,x);}
fn nbFastSolid()->bool{return ${this.solid.coarse?"nbSolidRouting&&round(umSolidParams.boundary.z)<0.5&&!umSolidHasTerrain()":"false"};}
fn nbHullUncut()->bool{
 ${this.solid.coarse?/* wgsl */`let a=vec3i(floor(nbHullLo-vec3f(0.001)));let b=vec3i(floor(nbHullHi+vec3f(0.001)));
 let first=a>>vec3u(2u);let last=b>>vec3u(2u);
 if(any(a<vec3i(0))||any(last-first>vec3i(2))){return false;}
 for(var z=first.z;z<=last.z;z++){for(var y=first.y;y<=last.y;y++){for(var x=first.x;x<=last.x;x++){
  let t=vec3i(x,y,z);
  // The window's proof of the tile (stored; open and uncut) where it holds it.
  let w=nbWinIndex(t);
  if(w<64u){let f=nbWinFlags[w];if((f&NB_WIN_STORED)==0u){return false;}if((f&NB_WIN_UNCUT)!=0u){continue;}}
  let key=umTileKey(t);if(!umKeyStored(key)){return false;}
  let open=umSolidOpenCells(key);let field=umSolidFieldCut(key);
  if(all(open==vec2u(0xffffffffu))&&all(field==vec2u(0u))){continue;}
  // The box's cells of this tile: bit x+4y+16z.
  let lo=vec3u(max(a-4*t,vec3i(0)));let hi=vec3u(min(b-4*t,vec3i(3)));
  let row=((2u<<hi.x)-1u)&~((1u<<lo.x)-1u);var box=vec2u(0u);
  for(var cz=lo.z;cz<=hi.z;cz++){for(var cy=lo.y;cy<=hi.y;cy++){
   let bits=row<<(4u*cy+16u*(cz&1u));if(cz<2u){box.x|=bits;}else{box.y|=bits;}
  }}
  if(any((open&~field&box)!=box)){return false;}
 }}}
 return true;`:"return false;"}
}
// The move's brick context: a slab job stages its brick's 4³-tile window
// (the brick's 2³ tiles and a tile around them), a lane a tile, with what
// the move's per-sample lookups read through the page table: the sampler's
// tile class and halo home, the hull proof, the depth's reach, the target
// heat, G2P's cold tier and the cell runs' first bin. A sample's tile
// inside the window reads them from workgroup memory; outside it, or in a
// flat job, from the device.
// Both are the same functions of the same tile.
const NB_WIN_FINE:u32=1u;const NB_WIN_STORED:u32=2u;const NB_WIN_UNCUT:u32=4u;const NB_WIN_COLD:u32=8u;const NB_WIN_REACH:u32=16u;
var<workgroup> nbWinFlags:array<u32,64>;
var<workgroup> nbWinHeat:array<f32,64>;
var<workgroup> nbWinBin:array<u32,64>;
${unitTaps?"var<workgroup> nbWinHome:array<vec4i,64>;":""}
var<private> nbWinLo:vec3i;
var<private> nbWinOn:bool=false;
fn nbWinIndex(t:vec3i)->u32{let l=t-nbWinLo;if(!nbWinOn||any(l<vec3i(0))||any(l>vec3i(3))){return 64u;}return u32(l.x+4*l.y+16*l.z);}
${narrowBandBodyWindowWGSL}
${narrowBandClearWalkWGSL}
fn nbWinStage(lane:u32){
 let t=nbWinLo+vec3i(umCorner(lane,4u));let at=umTileAt(vec3u(t));let key=umTileKey(t);var flags=0u;
 ${unitTaps?"if(umCachedFineTile(at)){flags|=NB_WIN_FINE;}nbWinHome[lane]=umUnitHaloHome(UMSampleTile(vec3f(0.0),t,at));":""}
 var bin=NB_NO_BIN;
 if(umKeyStored(key)){
  flags|=NB_WIN_STORED;
  if(atomicLoad(&bins[NB_COVERAGE+key])!=0u){bin=nbBinTile(key);}
  ${this.solid.coarse?"if(all(umSolidOpenCells(key)==vec2u(0xffffffffu))&&all(umSolidFieldCut(key)==vec2u(0u))){flags|=NB_WIN_UNCUT;}":""}
 }
 if(nbTileCold(key)){flags|=NB_WIN_COLD;}
 if((atomicLoad(&bins[NB_BAND_REACH+at])&1u)!=0u){flags|=NB_WIN_REACH;}
 flags|=nbBodyTileMask(t)<<8u;
 nbWinFlags[lane]=flags;nbWinBin[lane]=bin;nbWinHeat[lane]=bitcast<f32>(atomicLoad(&bins[NB_ACTIVITY_TARGET+at]));
}
// nbTargetHeat, the move's COLD test and the depth's reach, through the window.
fn nbMoveHeat(p:vec3f)->f32{let t=vec3u(p/4.0);let w=nbWinIndex(vec3i(t));if(w<64u){return nbWinHeat[w];}return bitcast<f32>(atomicLoad(&bins[NB_ACTIVITY_TARGET+umTileAt(t)]));}
${this.heatExpiry==="held"?`// The census found the surface of q's tile thin (UNIFORM_STAGE_IMPORTANCE triggered thin).
fn nbTileThin(q:vec3f)->bool{let t=vec3u(q/4.0);let key=umTileAt(t);if(!umKeyStored(key)){return false;}return ((atomicLoad(&bins[NB_ACTIVITY_SCORES+2u*key+1u])>>17u)&1u)!=0u;}
`:""}fn nbMoveCold(q:vec3f)->bool{let t=vec3i(floor(q))>>vec3u(2u);let w=nbWinIndex(t);if(w<64u){return (nbWinFlags[w]&NB_WIN_COLD)!=0u;}return nbTileCold(umTileKey(t));}
fn nbMoveDepth(p:vec3f)->f32{
 let t=vec3u(p)/4u;let w=nbWinIndex(vec3i(t));var reached:bool;
 if(w<64u){reached=(nbWinFlags[w]&NB_WIN_REACH)!=0u;}else{reached=(atomicLoad(&bins[NB_BAND_REACH+umTileAt(t)])&1u)!=0u;}
 return particleDepthMoveIn(p,reached);
}
// A moved record: status 0 live, 1 retired (or past a fatal).
struct NbMoved{p:Particle,status:u32}
// The move is the step's whole particle stage, one pass a record (i its
// current-bank index):
// - G2P: the last step's end, against the field that step projected (the
//   head's extension of it, which the trace samples too) and the FLIP delta
//   the tail recorded (before.xyz: its extension minus the departure's,
//   NB_DELTA_NONE an axis without, which takes PIC); a sample's depth,
//   isolation age and erasure heat.
// - Retirement: deep or sparse-expired samples are not appended. In the
//   deep overlap, apply the runtime particle cap to each existing cell run.
//   Seeds and spill records outside those runs retain their other rules.
// - The trace: one step of Heun's third-order rule (below), whose first
//   stage is G2P's grid sample.
// A sample's radius, r = cbrt(m)/4 for the integer masses 1..8 it can hold:
// eight unit samples a cell rest half a cell apart. Its centre keeps r from
// every solid it meets, measured along the solid's normal, as it keeps
// r_i+r_j from another sample (nbSeparation): the box walls
// (nbTraceEndpointIn), voxel and SOLID page faces (nbFinish's walk), the
// procedural field, terrain and bodies (their contacts). A sample on a wall
// plane would sit where the grid's normal velocity is zero, carried along the
// face and never off it, its mass in the union's cells beside the wall.
const NB_SAMPLE_RADIUS=array<f32,9>(0.0,0.25,0.31498,0.36056,0.39685,0.42750,0.45428,0.47823,0.5);
fn nbSampleRadius(m:f32)->f32{return NB_SAMPLE_RADIUS[min(u32(m),8u)];}
${this.sampleSeparation==="pairs"?`// sampleSeparation "pairs": Sato et al. 2018 Section 3.2.2 (after Ando et al. 2012): two samples
// closer than the sum of their radii push apart, each by the other's share
// of their mass, d_ij = m_j/(m_i+m_j)(r_i+r_j-|p_i-p_j|) along p_i-p_j.
// Eight samples a cell rest half a cell apart, so r = cbrt(m)/4. Reads the
// current bank's sorted runs, which the move does not write (it appends to
// the scratch bank), of the 2x2x2 cells around the vertex nearest the
// sample: they hold every sample within half a cell along each axis, the
// reach of two unit samples. A merged (deep) sample's wider reach is gathered
// as far as those cells go. Positions are read alone, a mass only in reach.
// (SORT's compact stream is no cheaper here: links is a u32 array, so its
// position is three loads against the record's one; a letters move issued
// 75% more buffer loads through it, 2026-11-09.)
// The pairs are solved together as one Jacobi pass with constraint averaging
// (Macklin et al. 2014, Section 4.2): the sample moves by the mean of its
// pair corrections, so a pile-up of n overlapping samples does not push it n
// pair corrections at once. Each correction is under r_i+r_j, so the move is
// bounded by NARROW_BAND_SEPARATION_REACH, which the residency prediction adds
// to every coupled sample's travel (particleBrickDepositWGSL).
fn nbSeparation(i:u32,q:vec3f,m:f32)->vec3f{
 let base=vec3i(floor(q-0.5));let ri=nbSampleRadius(m);let far=(ri+0.5)*(ri+0.5);let shift=nbHeldShift();var d=vec3f(0.0);var n=0.0;
 for(var k=0u;k<4u;k++){
  let c=vec3i(base.x,base.y+i32(k&1u),base.z+i32(k>>1u));
  let stop=min(base.x+1,c.x|3);
  var runs=array<vec2u,2>(nbRun(c,vec3i(stop,c.y,c.z)),vec2u(0u));
  if(stop<base.x+1){runs[1]=nbRun(vec3i(stop+1,c.y,c.z),vec3i(base.x+1,c.y,c.z));}
  for(var r=0u;r<2u;r++){
   for(var j=runs[r].x;j<runs[r].y;j++){
    if(j==i){continue;}
    let other=particles[j].position;if(other.w<=0.0){continue;}
    let x=q-(other.xyz-shift);let l2=dot(x,x);
    if(l2>=far||l2<=1e-12){continue;}
    let mj=${this.crowdedCells!=="merge"?"1.0":this.apic?"particles[j].lifetime.y+1.0":"f32((bitcast<u32>(particles[j].before.w)&7u)+1u)"};let reach=ri+nbSampleRadius(mj);
    if(l2<reach*reach){let l=sqrt(l2);d+=mj/(m+mj)*(reach-l)*x/l;n+=1.0;}
   }
  }
 }
 return d/max(n,1.0);
}
// The free surface: a sample within a cell of it (its depth over -1) keeps
// only the tangential part of its correction, d - (d.n)n with n the level
// set's normal, the particle-shifting rule at a free surface (Khayyer et al.
// 2017, which cancels the normal shift at the free surface and its
// neighbourhood; Sun et al. 2019, Eq. 44; after Lind et al. 2012). A surface
// sample has no neighbours on the air side, so every overlap pushes it out:
// the normal part moved the surface itself, outward seven times as far as
// inward in a dam's run-up (docs/plans/bulk16-plan-2026-10-28.md, M4b), a
// volume source the union kept. The projection never lengthens d, so
// NARROW_BAND_SEPARATION_REACH still bounds it. The centre and the six
// quarter-cell offsets are read at one nbMoveDepth site (Metal inlines each
// call of it: seven copies spilled the move a further 32 bytes, 2026-11-09).
fn nbSurfaceShift(q:vec3f,d:vec3f)->vec3f{
 var g=vec3f(0.0);
 for(var k=0u;k<7u;k++){
  var x=q;let axis=(k-1u)>>1u;let up=((k-1u)&1u)==0u;
  if(k>0u){x[axis]+=select(-0.25,0.25,up);}
  let depth=nbMoveDepth(x);
  if(k==0u){if(depth<= -1.0){return d;}}else{g[axis]+=select(-depth,depth,up);}
 }
 let l=length(g);if(!(l>1e-6)){return d;}
 let n=g/l;return d-dot(d,n)*n;
}
`:""}fn nbAdvect(record:Particle,i:u32)->NbMoved{
  var p=record;var q=p.position.xyz-nbHeldShift();
  // A seed's first move (its age nbSeedFirst): age 0, and G2P gives
  // it the grid's velocity, as the tail seeds sampled the step's final field.
  let first=bitcast<u32>(nbParticleAge(record))==NB_SEED_FIRST;
  if(first){${this.apic?"p.lifetime.x=0.0;":"p.before.w=nbParticleRestate(record,0.0,nbParticleMode(record));"}}
  ${this.crowdedCells==="merge"?`// crowdedCells "merge": a deep cell (its centre a cell or more under the
  // surface) holding more than its cap keeps the cap's lowest ranks of its
  // sorted run, each gathering the excess ranks congruent to it
  // (nbMergeRun); the excess gathered retire here, unmoved, and the excess
  // a full gatherer refused move on as samples. Every sample in the
  // cell reads the same centre depth, so all agree on the cell.
  if(params.activity.w>0.0){
   let cell=nbBinCell(umCellAt(vec3u(q)));
   if(cell!=NB_NO_BIN){
    let start=links[cell];let count=atomicLoad(&bins[cell]);
    if(i>=start&&i-start<count){
     let centre=floor(q)+0.5;let deep=nbMoveDepth(centre);let cap=nbCellCap(deep);
     if(deep< -1.0&&count>cap){
      if(i-start<cap){p=nbMergeRun(p,i+cap,start+count,cap);q=p.position.xyz-nbHeldShift();}
      else if(nbMergeAbsorbed(start,i,cap)){${narrowBandRetireCounting.on?"nbRetireCount(2u,deep,q.x,1.0);":""}return NbMoved(p,1u);}
      ${narrowBandRetireCounting.on?"else{nbMergeCount(1u,1u);}":""}
     }
    }
   }
  }
  `:""}
  nbHullLo=q;nbHullHi=q;
  let mode=nbParticleMode(p);
  // Uniform stores velocity half a step ahead of position. Ballistic drift
  // uses that midpoint velocity; G2P supplies its full gravity kick.
  // One step of Heun's third-order rule over the whole dt, every field
  // sample at one call site (Metal inlines each call of the field): stage 0
  // the field at q (k1, and G2P's grid velocity), 1 a third of the step
  // along it (k2), 2 two thirds along k2 (k3); the end is q + dt(k1 + 3k3)/4.
  // Its stages stay inside the step, so a trace into a sharp turn (liquid
  // striking the pool) bends with it as the old substeps did; a stage a
  // whole step ahead (Shu and Osher's) overshot there and the band lost a
  // tenth of its samples. Nothing limits the step: a stage the field does
  // not hold is the field's own sample to bound. A ballistic record samples
  // nothing; a record the trace finds ballistic only stage 0.
  let h=params.hDt.xyz;let dt=params.hDt.w;
  var x=q;var k1=vec3f(0.0);var pic=vec3f(0.0);var end=q;var ballistic=true;
  var stage=select(0u,3u,mode==1.0);
  loop{
   if(stage<3u){
    nbHullInclude(x);
    let k=nbMoveSample(x);
    if(stage==1u){x=q+2.0*dt*k/(3.0*h);stage=2u;continue;}
    if(stage==2u){end=q+0.25*dt*(k1+3.0*k/h);break;}
    pic=k;
   }
   // G2P.
   var v=p.velocity.xyz;
   if(mode!=1.0){
    ${this.apic?"p=nbApicSample(p);v=p.velocity.xyz;":"var flip=v;if(!nbMoveCold(q)){let d=p.before.xyz;flip=select(pic,v+d,abs(d)<vec3f(1e30));}v=select(mix(pic,flip,params.settings.x),pic,first);"}
   }else{
    ${this.apic?"p.cx=vec4f(0);p.cy=vec4f(0);p.cz=vec4f(0);":""}
    v+=params.gravity.xyz*dt;
    // A sample within 0.011 of a cell of a solid page it moves toward loses
    // that axis: one kind an axis, of the cell ahead.
    for(var axis=0u;axis<3u;axis++){
     var ahead=q;ahead[axis]+=select(-1.0,1.0,v[axis]>0.0)*(0.001+nbSampleRadius(nbParticleMass(p)));
     if((v[axis]<0.0||v[axis]>0.0)&&umPointKind(ahead)>=UM_SOLID){v[axis]=0.0;}
    }
   }
   ${this.apic?"if(!all(abs(p.cx.xyz)<vec3f(1e10))||!all(abs(p.cy.xyz)<vec3f(1e10))||!all(abs(p.cz.xyz)<vec3f(1e10))){atomicStore(&state[3],2u);return NbMoved(p,1u);}":""}
   if(!all(abs(v)<vec3f(1e10))){atomicStore(&state[3],2u);return NbMoved(p,1u);}
   let age=nbSparseAge(i,q,nbParticleAge(p));let depth=nbMoveDepth(q);
   ${this.apic?"p.lifetime.x=age;":"p.before.w=nbParticleRestate(p,age,mode);"}
   p.velocity=vec4f(v,depth);p.position.w=max(p.position.w,nbMoveHeat(q));
   // The accepted grid motion: the next head's residency prediction.
   // FLIP's residual velocity transfers momentum but traces no position.
   p.before=vec4f(pic,p.before.w);
   if(depth< -NB_BAND_DEPTH||nbSparseExpired(age)){${narrowBandRetireCounting.on?"nbRetireCount(select(1u,0u,depth< -NB_BAND_DEPTH),depth,q.x,nbParticleMass(p));":""}return NbMoved(p,1u);}
   if(${this.crowdedCells==="merge"?"false&&":""}depth<-1.0&&params.activity.w>0.0){
    let cell=nbBinCell(umCellAt(vec3u(q)));
    if(cell!=NB_NO_BIN){
     let rank=i-links[cell];
     let cap=select(u32(params.activity.w),min(u32(params.activity.w),6u),NB_GRADED&&depth<=-1.5);
     if(i>=links[cell]&&rank<atomicLoad(&bins[cell])&&rank>=cap){${narrowBandRetireCounting.on?"nbRetireCount(2u,depth,q.x,1.0);":""}return NbMoved(p,1u);}
    }
   }
   // Bodies advance at the previous fluid step's tail. Push samples they
   // overtook out to the moving surface before the static-solid rejection.
   let rigidStart=nbRigidContact(q,p.velocity.xyz,nbSampleRadius(nbParticleMass(p)));q=rigidStart.position;p.velocity=vec4f(rigidStart.velocity,p.velocity.w);
   // Solid collision, rather than grid ownership, limits particle motion.
   if(nbParticleBlocked(q)){${narrowBandRetireCounting.on?"nbRetireCount(3u,depth,q.x,nbParticleMass(p));":""}return NbMoved(p,1u);}
   ballistic=mode==1.0||!gridSupported(q);p.before.w=nbParticleRestate(p,nbParticleAge(p),select(0.0,1.0,ballistic));
   ${this.apic?"if(ballistic){p.cx=vec4f(0);p.cy=vec4f(0);p.cz=vec4f(0);}":""}
   // A sample takes erasure heat (over 1) at G2P, with its tile's new
   // collar seeded: a collar tile the census triggered eroded the unseeded
   // tiles beside it that same step.
   let tileHeat=nbMoveHeat(q);
   p.position.w=max(select(min(tileHeat,1.0),tileHeat,p.position.w>1.0),max(0.0,p.position.w-select(nbCooling()*params.hDt.w,0.0,ballistic)));
   ${this.heatExpiry==="held"?`// heatExpiry "held": the level set does not hold a shallow sample in a thin tile.
   if(depth> -1.0&&nbTileThin(q)){p.position.w=max(p.position.w,1.0);}
   `:""}if(p.position.w<=0.0){${narrowBandRetireCounting.on?"nbRetireCount(4u,depth,q.x,nbParticleMass(p));":""}return NbMoved(p,1u);}
   // A record its transfer left coupled was bounded by the grid flow alone
   // (its travel record is the accepted grid motion): found unsupported here,
   // it drifts this step by that grid sample. Its FLIP velocity is momentum,
   // and drifts from the next step, whose census reads its speed.
   end=q+dt*select(p.velocity.xyz,pic,mode!=1.0)/h;
   if(ballistic){break;}
   k1=pic/h;x=q+dt*k1/3.0;stage=1u;
  }
  // Reject nonfinite traces before the wall clamp can hide them. The
  // lattice reach applies to the actual collision-bounded trajectory:
  // a far endpoint behind a tank wall still stops at that wall.
  ${this.sampleSeparation==="pairs"?"if(!ballistic){let sepd=nbSeparation(i,q,nbParticleMass(p));end+=nbSurfaceShift(q,sepd);}":""}
  if(!all(abs(end)<vec3f(1e30))){atomicStore(&state[3],2u);return NbMoved(p,1u);}
  let endpoint=nbTraceEndpointIn(vec3i(floor(q)),end,nbSampleRadius(nbParticleMass(p)));
  let reach=abs(endpoint-q);
  if(!(max(reach.x,max(reach.y,reach.z))<${NARROW_BAND_TRACE_REACH}.0)){atomicStore(&state[3],1u);return NbMoved(p,1u);}
  nbHullInclude(endpoint);
  // The box the sample's sphere sweeps: its centre's, out by its radius.
  {let r=nbSampleRadius(nbParticleMass(p));nbHullLo-=vec3f(r);nbHullHi+=vec3f(r);}
  return nbFinish(p,q,end,endpoint,ballistic,nbFastSolid()&&umBodyCount()==0u&&nbHullUncut());
}
// The trace's end: the solid walk from start to endpoint (end held inside
// a SOLID page's face), the arrival and its heat. uncut: the box is proven
// open, so the walk would block nothing.
fn nbFinish(record:Particle,start:vec3f,end:vec3f,endpoint:vec3f,ballistic:bool,uncut:bool)->NbMoved{
  var p=record;var q=start;
  // With no embedded solids the segment stays inside the convex container
  // after endpoint clamping. There are no voxel crossings to test; retaining
  // the RK3 endpoint also avoids accumulating rounded collision increments.
  // Nor are there where the solid's record proves the trace's box open and
  // uncut (nbHullUncut) in a scene with no body or terrain: the walk would
  // block nothing and the field's contact find no hit.
  if(!umSolidEnabled()||uncut){q=endpoint;}else{
   // The long lane walks the chord in strides of at most half a cell an axis.
   let travel=endpoint-q;let walk=max(1u,u32(ceil(2.0*max(abs(travel.x),max(abs(travel.y),abs(travel.z))))));
   // A voxel face stops a particle as a container wall does: the crossing
   // axis ends at the face and loses its velocity, the others carry on.
   // A floor is the exception where its tread stands for a sloped bed
   // (umBedSlope): the sample then loses its velocity along that bed's
   // normal, and so keeps the part of its fall that runs down the slope. It
   // gains nothing along an axis a wall has stopped, so it rests in a corner.
   let stride=travel/f32(walk);var landed=false;var floorCell=vec3i(0);var free=vec3f(1);
   let inset=nbSampleRadius(nbParticleMass(p));
   let clearWalk=nbClearWalk();
   for(var j=1u;j<=walk;j++){
    if(clearWalk){
     for(var axis=0u;axis<3u;axis++){var point=q;point[axis]+=stride[axis];q=point;}
     continue;
    }
    let start=q;
    // Bodies matter only where a present body's pose lies within two cells
    // of the start's 16h region (the head's body region mask). Elsewhere the
    // voxel, terrain and boundary walls alone can block it.
    let bodyNear=umBodyCount()>0u&&umBodyPoseNearTile(umTileKey(vec3i(floor(start))>>vec3u(2u)),2u);
    for(var axis=0u;axis<3u;axis++){
     var point=q;point[axis]+=stride[axis];
     // The sample's leading edge along the stride: its centre keeps inset
     // (its radius) from a voxel or SOLID page face.
     var lead=point;lead[axis]+=sign(stride[axis])*inset;
     var blocked:bool;
     if(nbAdvectScalarCollision){
      // Diagnostic oracle: WGSL select evaluates both values, including the
      // expensive fractional open-cell query that a boundary cell discards.
      blocked=select(umCellOpen(vec3i(lead))<0.5,
       nbVoxelBlocked(lead),umSolidHasTerrain()||bodyNear||umBoundaryRecord(vec3i(floor(lead)))!=0u);
     }else if(umSolidHasTerrain()||bodyNear||umBoundaryRecord(vec3i(floor(lead)))!=0u){
      blocked=nbVoxelBlocked(lead);
     }else{blocked=umCellOpen(vec3i(lead))<0.5;}
     // A SOLID page (the lattice's walls) stops it as a voxel does; only a
     // step that leaves its page reads the page table.
     if(!blocked&&any((vec3i(floor(lead))>>vec3u(5u))!=(vec3i(floor(q))>>vec3u(5u)))){blocked=umPointKind(lead)>=UM_SOLID;}
     if(blocked){
      let face=floor(lead[axis])+select(1.0,0.0,stride[axis]>0.0);point[axis]=face-sign(stride[axis])*inset;
      if(axis==1u&&stride.y<0.0){landed=true;floorCell=vec3i(q);}else{p.velocity[axis]=0.0;free[axis]=0.0;}
     }
     q=point;
    }
    if(bodyNear){
     let rigidContact=nbRigidContact(q,p.velocity.xyz,inset);
     if(!nbVoxelBlocked(rigidContact.position)){q=rigidContact.position;p.velocity=vec4f(rigidContact.velocity,p.velocity.w);}
    }
    let boundaryContact=nbProceduralContact(start,q,p.velocity.xyz,inset);
    if(!nbVoxelBlocked(boundaryContact.position)&&(!bodyNear||nbMoveBodyAt(boundaryContact.position)<0)){q=boundaryContact.position;p.velocity=vec4f(boundaryContact.velocity*free,p.velocity.w);}
    if(umSolidHasTerrain()){
     let contact=nbTerrainContact(start,q,p.velocity.xyz,inset);
     // A normal correction must not cross an authored voxel or a body.
     if(!nbVoxelBlocked(contact.position)&&(!bodyNear||nbMoveBodyAt(contact.position)<0)){q=contact.position;}
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
  // A sample that ends in a page that holds no storage is fatal; it is not
  // appended.
  if(!umPointPresent(q)){nbArrived(q);return NbMoved(p,1u);}
  // A trajectory can cross from h into a coarse owner during this step.
  // Erasure heat stays with the tile that has it. A hot tile's collar is
  // seeded; a collar's own neighbours need not be, and a grid-supported
  // sample that carried its heat into the collar eroded the unseeded liquid
  // beside it. Spray keeps its heat: nothing else erases what it leaves.
  var heat=p.position.w;
  if(!ballistic){let arrived=nbMoveHeat(q);if(arrived>0.0){heat=min(heat,max(arrived,1.0));}}
  p.position=vec4f(q,heat);
  ${this.crowdedCells==="merge"?`// crowdedCells "merge": a merged sample that ends shallow (under a cell
  // deep) or in a cell its run left under half the cap splits in two a
  // quarter cell apart along a direction hashed from its position, about its
  // centroid, each half with its velocity and FLIP delta: the film it spreads
  // into gets back the coverage its mass stands for. The second half
  // appends to its brick's slab (nbMoveAppend). Halves that would leave the
  // storage or enter a solid stay merged this step.
  let mass=nbParticleMass(p);
  if(mass>1.0&&!ballistic&&params.activity.w>0.0){
   let cell=nbBinCell(umCellAt(vec3u(q)));let depth=nbMoveDepth(q);
   if(depth> -1.0||cell==NB_NO_BIN||2u*atomicLoad(&bins[cell])<nbCellCap(depth)){
    let kept=ceil(0.5*mass);let moved=mass-kept;
    let bits=bitcast<vec3u>(q);let hash=(bits.x*73856093u)^(bits.y*19349663u)^(bits.z*83492791u);
    let along=normalize(vec3f(f32(hash&255u),f32((hash>>8u)&255u),f32((hash>>16u)&255u))-127.5);
    let a=q+0.25*(moved/mass)*along;let b=q-0.25*(kept/mass)*along;
    if(umPointPresent(a)&&umPointPresent(b)&&!nbParticleBlocked(a)&&!nbParticleBlocked(b)){
     var other=p;other.position=vec4f(b,heat);nbMoveAppend(nbWithMass(other,moved));
     p.position=vec4f(a,heat);${narrowBandRetireCounting.on?"nbMergeCount(2u,1u);":""}
     return NbMoved(nbWithMass(p,kept),0u);
    }
   }
  }
  `:""}return NbMoved(p,0u);
}
// The move: each workgroup of the row (NARROW_BAND_PARTICLE_GRID) takes
// its next job from the ticket (NB_MOVE_TICKET_WORD) until none is left, the
// spill reserve's flat jobs first: their samples read the device for what a
// slab job's window holds. A row striding the jobs in turn left a launch's
// last groups three or four jobs behind the rest: a letters move (960k
// samples) held its occupancy ceiling to its last 2-3 ms and then fell
// away to nothing (counter capture, 2026-11-09). A job
// is NB_MOVE_CHUNK records of an input slab (last move's current slabs: a dense brick
// is many jobs, so no workgroup serializes one) or 64 entries of the flat lane (the spill reserve).
// Every record goes through the one nbAdvect site (Metal inlines every
// call) and appends to its brick's slab in the scratch bank. A slab job's
// appends to the 27 bricks around its own are counted in workgroup memory
// and take one device atomic a brick a job; the rest take one each.
// sampleSeparation "pairs" takes each record's pair correction inline, from
// the current bank, which the move does not write: the move is one launch.
var<workgroup> nbMoveJobs:vec4u;
var<workgroup> nbMoveJob:vec4u;
var<workgroup> nbMoveTicket:u32;
var<workgroup> nbMoveCount:array<atomic<u32>,27>;
var<workgroup> nbMoveBase:array<u32,27>;
// A neighbour brick's id, slab start and cap (cap 0: no slab this move).
var<workgroup> nbMoveBrick:array<vec3u,27>;
// Slab job j: the input entry whose chunks hold it (the entries' first
// jobs ascend), its chunk's records and the entry's brick.
fn nbSlabJob(j:u32)->vec4u{
 var lo=0u;var hi=min(atomicLoad(&state[19u]),UB_BRICKS);
 while(hi-lo>1u){let mid=(lo+hi)/2u;if(atomicLoad(&state[NB_SLAB_IN+4u*mid+3u])<=j){lo=mid;}else{hi=mid;}}
 let at=NB_SLAB_IN+4u*lo;let chunk=NB_MOVE_CHUNK*(j-atomicLoad(&state[at+3u]));
 return vec4u(0u,atomicLoad(&state[at+1u])+chunk,min(NB_MOVE_CHUNK,atomicLoad(&state[at+2u])-min(chunk,atomicLoad(&state[at+2u]))),atomicLoad(&state[at]));
}
fn nbMoveTurn(r:NbMoved,job:vec4u,lane:u32){
 var n=27u;var rank=0u;
 if(r.status==0u&&job.x==0u){
  let d=(vec3i(floor(r.p.position.xyz))>>vec3u(3u))-(vec3i(ubBrickOrigin(job.w))>>vec3u(3u))+1;
  if(all(d>=vec3i(0))&&all(d<=vec3i(2))){n=u32(d.x+3*d.y+9*d.z);rank=atomicAdd(&nbMoveCount[n],1u);}
 }
 workgroupBarrier();
 if(lane<27u){
  let count=atomicExchange(&nbMoveCount[lane],0u);var at=0u;
  if(count!=0u&&nbMoveBrick[lane].z!=0u){at=atomicAdd(&state[nbSlab(nbMoveBrick[lane].x,2u)],count);}
  nbMoveBase[lane]=at;
 }
 workgroupBarrier();
 if(r.status!=0u){return;}
 if(n==27u){nbMoveAppend(r.p);return;}
 let brick=nbMoveBrick[n];let k=nbMoveBase[n]+rank;
 if(k<brick.z){source[brick.y+k]=r.p;return;}
 // A full slab counts its overflow; a brick with no slab counts nothing.
 if(brick.x!=UB_NONE){atomicAdd(&state[nbSlab(brick.x,4u)],1u);}
 _=nbSpill(r.p,true,true);
}
fn nbMove(group:u32,groups:u32,lane:u32){
 if(lane==0u){
  // Slab/spill placement is an allocation detail, not a numerical choice.
  let entries=atomicLoad(&state[22u]);let flat=nbSpillCount(false);
  nbMoveJobs=vec4u(entries,(flat+63u)/64u,flat,0u);
 }
 let jobs=workgroupUniformLoad(&nbMoveJobs);
 loop{
  if(lane==0u){nbMoveTicket=atomicAdd(&state[${NB_MOVE_TICKET_WORD}u],1u);}
  let t=workgroupUniformLoad(&nbMoveTicket);
  if(t>=jobs.x+jobs.y){break;}
  if(lane==0u){
   if(t>=jobs.y){nbMoveJob=nbSlabJob(t-jobs.y);}
   else{let k=64u*t;nbMoveJob=vec4u(1u,k,min(64u,jobs.z-k),0u);}
  }
  let job=workgroupUniformLoad(&nbMoveJob);
  nbWinOn=job.x==0u;nbWinLo=(vec3i(ubBrickOrigin(job.w))>>vec3u(2u))-1;
  if(nbWinOn){nbWinStage(lane);}
  if(job.x==0u&&lane<27u){
   let id=ubBrickAt(vec3i(ubBrickOrigin(job.w))+8*(vec3i(umCorner(lane,3u))-1));
   var brick=vec3u(UB_NONE,0u,0u);
   if(nbSlabCurrent(id)){brick=vec3u(id,atomicLoad(&state[nbSlab(id,0u)]),atomicLoad(&state[nbSlab(id,1u)]));}
   nbMoveBrick[lane]=brick;
  }
  workgroupBarrier();
  for(var off=0u;off<job.z;off+=64u){
   let k=off+lane;var r=NbMoved();r.status=1u;
   if(k<job.z){
    var i=job.y+k;
    if(job.x==1u){i+=nbSpillBase(false);}
    let record=particles[i];
    if(nbLive(record)){r=nbAdvect(record,i);}
   }
   nbMoveTurn(r,job,lane);
  }
  workgroupBarrier();
 }
}
@compute @workgroup_size(64) fn advectParticles(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){nbMove(group.x,groups.x,lane);}
@compute @workgroup_size(64) fn advectParticlesDeferred(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){nbMove(group.x,groups.x,lane);}
// Jittered sub-cell sampling (Zhu and Bridson 2005; Bridson 2008): a seed
// lies anywhere in its eighth of the cell, hashed from the cell, the site and
// the seeding salt (narrowBandSeedJitterWGSL). Seeds stay where they are
// made: projecting the outer sites to one depth stacked the two sites along
// the normal onto one point, coincident pairs the transport carried and
// separation then pushed apart.
var<private> nbSeedSalt:u32=0u;
${narrowBandSeedJitterWGSL}
fn nbJitter(c:vec3u,k:u32)->vec3f{return nbSeedJitter(c,k,nbSeedSalt);}
${this.outerSeeding==="uncovered"?`// No sample in cell c or the 26 around it (outerSeeding "uncovered").
fn nbUncovered(c:vec3u)->bool{
 for(var k=0u;k<27u;k++){
  let n=vec3i(c)+vec3i(umCorner(k,3u))-1;
  if(umCellPresent(n)&&nbBinCount(umCellAt(vec3u(n)))!=0u){return false;}
 }
 return true;
}
`:""}fn seedSite(c:vec3u,k:u32,initial:bool,activated:bool)->vec4f{
  let q=vec3f(c)+0.5*vec3f(umCorner(k,2u))+0.5*nbJitter(c,k);let distance=particleDepth(q);
  if(distance>0.0||distance< -NB_BAND_DEPTH||nbParticleBlocked(q)){return vec4f(-1,0,0,0);}
  // Paper Section 3.3: replenish only the inner [-R,-h] band. The outer
  // layer keeps its transported samples instead of being recreated; a change
  // of ownership or an uncovered tile cannot bootstrap one from the bulk
  // field. Section 3.2: liquid made from a description gets its whole band,
  // at initialization and where a source has just made it.
  // innerReseed "cells": a cell's sample count measures the inner band only
  // where the whole cell lies in it, its centre at least h + h/2 down.
  // activationSeeding "uncovered": a newly heated tile seeds its outer
  // layer only where no sample describes the liquid (nbUncovered).
  if(!initial&&${this.activationSeeding==="uncovered"&&this.outerSeeding==="uncovered"?"":"!activated&&"}(distance>-1.0${this.innerReseed==="cells"?"||particleDepth(vec3f(c)+0.5)>-1.5":""})&&nbSourcePhi(q,1.0)>=0.0${this.outerSeeding==="uncovered"?"&&!nbUncovered(c)":""}){return vec4f(-1,0,0,0);}
  // The heat-one overlap supplies support without authority to erase
  // liquid. Its outer shell only seeds on resolved fine owners.
  if(distance>-1.0&&umOwnerAt(vec3i(c)).width!=1u){return vec4f(-1,0,0,0);}
  return vec4f(q,distance);
}
// The mass of cell c's filed samples, in units (eight a cell).
fn nbCellMass(c:vec3u)->f32{let run=nbCellRun(vec3i(c));var m=0.0;for(var i=run.x;i<run.y;i++){m+=nbParticleMass(particles[i]);}return m;}
fn seedCell(c:vec3u,emit:bool){
 // No tile past the reach is within six cells of the surface.
 if((nbBandReach(c)&1u)==0u){return;}
 let heat=nbTargetHeat(vec3f(c)+0.5);if(heat<=0.0){return;}
 if(!nbParticleCellOpen(vec3i(c))){return;}
 let d=particleDepth(vec3f(c)+0.5);if(d>0.9||d< -NB_BAND_DEPTH-0.5){return;}
 let activated=atomicLoad(&bins[NB_ACTIVITY_NEW+nbBandTile(c)])!=0u;
 var seeds:array<vec4f,8>;var seedCount=0u;
 for(var j=0u;j<nbSeedTarget(d);j++){
  let site=seedSite(c,nbSeedSiteIndex(j,d),true,activated);if(site.x<0.0){continue;}
  seeds[seedCount]=site;seedCount++;
 }
 // Count first: each seed's brick takes it as demand and is listed, so the
 // slabs that follow hold the whole initial band.
 for(var k=0u;k<seedCount;k++){
  // An emitted seed is a seed too (nbSeedFirst): its first G2P takes the
  // grid's velocity. Unmarked, a tail-recorded FLIP delta once read its
  // before.xyz (its velocity, the travel the residency prediction needs) as
  // a delta and doubled it: translation 0.75 -> 1.46 m/s in one step.
  if(emit){var s=nbWithMass(nbSeedParticle(seeds[k],heat),f32(8u/nbSeedTarget(d)));${this.apic?"s.lifetime.x=nbSeedFirst();":"s.before.w=nbParticleRestate(s,nbSeedFirst(),0.0);"}nbMoveAppend(s);continue;}
  let id=ubBrickAt(vec3i(floor(seeds[k].xyz)));
  if(id!=UB_NONE){atomicAdd(&state[nbSlab(id,10u)],1u);ubMark(id,UB_PARTICLES_CORE);}
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
// The cells needing candidates: a list in the motion words, which nothing
// reads from the transfer to the next deposit (the move reads the cell
// starts). A list past them stops the run (state[3] = 7).
fn nbSeedList()->u32{return NB_LINKS+4u*arrayLength(&particles);}
@compute @workgroup_size(64) fn seedCells(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let c=nbSeedCell(group,lane);
 if((nbBandReach(c)&1u)==0u){return;}
 let heat=nbTargetHeat(vec3f(c)+0.5);if(heat<=0.0){return;}
 if(!nbParticleCellOpen(vec3i(c))){return;}
 let depth=particleDepth(vec3f(c)+0.5);if(depth>0.9||depth< -NB_BAND_DEPTH-0.5){return;}
 if(nbBinCount(umCellAt(c))>=nbSeedTarget(depth)||nbCellMass(c)>=8.0){return;}
 let slot=atomicAdd(&bins[NB_BAND+3u],1u);
 if(slot>=3u*arrayLength(&particles)){atomicStore(&state[3],7u);return;}
 // Bit 31 a source cell, bit 30 a graded band's deep cell (nbSeedTarget 4).
 links[nbSeedList()+slot]=umCellAt(c)|select(0u,0x80000000u,nbSourceCell(c))|select(0u,0x40000000u,nbSeedTarget(depth)==4u);
}
fn seedCandidate(group:vec3u,lane:u32,sourcePass:bool,counting:bool)->Particle{
 var empty=Particle();empty.position.x=-1.0;
 let slot=8u*group.x+lane/8u;if(slot>=min(atomicLoad(&bins[NB_BAND+3u]),3u*arrayLength(&particles))){return empty;}
 let entry=links[nbSeedList()+slot];if(((entry&0x80000000u)!=0u)!=sourcePass){return empty;}
 let c=umCellCoord(entry&0x3fffffffu);let j=lane%8u;let deep=(entry&0x40000000u)!=0u;
 let n0=nbBinCount(umCellAt(c));if(j<n0||j>=select(8u,4u,deep)){return empty;}
 // Paper Section 3.3 reseeds cells with fewer than n = 8 unit samples: a
 // cell's mass, not its record count, once samples carry mass.
 if(nbCellMass(c)+f32(j-n0)*select(1.0,2.0,deep)>=8.0){return empty;}
 let activated=atomicLoad(&bins[NB_ACTIVITY_NEW+nbBandTile(c)])!=0u;
 // Each frame salts its sites (the frame index, params.live.z), so a cell
 // reseeded again does not reuse them; nothing of the list's fill order
 // (the slot) or its count enters a site.
 nbSeedSalt=params.live.z;
 let site=seedSite(c,select(j,array<u32,4>(0u,3u,5u,6u)[j&3u],deep),false,activated);if(site.x<0.0){return empty;}
 ${narrowBandRetireCounting.splitX>=0?`// Diagnostic: seeds at x >= splitX: activated, above the floor band (y 8 past the box), interior, outer.
 if(counting&&f32(c.x)>=${narrowBandRetireCounting.splitX.toFixed(1)}){atomicAdd(&state[${NARROW_BAND_RETIRE_WORD+8}u+select(select(select(3u,2u,site.w<= -1.0),1u,c.y>=${(narrowBandRetireCounting.splitY+8)}u),0u,activated)],1u);}`:""}
 let heat=nbTargetHeat(vec3f(c)+0.5);
 // A cell's liquid is eight units: a graded deep cell's four samples carry
 // two each (crowdedCells "merge" holds the mass), so the union's bound
 // reads them as the cell once they rise and split into a film.
 return nbWithMass(nbSeedParticle(site,heat),select(1.0,2.0,deep));
}
// The step's seeds act from the next step, as the tail seeds did before
// the fused move: seeded at the transfer's head (the step's band, phi and
// counts), a seed joins its brick's slab in the current bank after the
// records SORT filed, which no transfer, splat or seam gather reads (only
// transferRefine, for the census's prediction); the next move's G2P gives
// it the grid's velocity (nbSeedFirst) and files it. No room: a source seed
// spills, an overlap reseed is clipped (its cell is short again next step).
fn storeSeedCandidate(p:Particle,mandatory:bool){
 var s=p;${this.apic?"s.lifetime.x=nbSeedFirst();":"s.before.w=nbParticleRestate(p,nbSeedFirst(),0.0);"}
 if(p.position.x>=0.0){${narrowBandRetireCounting.on&&narrowBandRetireCounting.splitX>=0?`if(p.position.x>=${narrowBandRetireCounting.splitX.toFixed(1)}){atomicAdd(&state[${NARROW_BAND_RETIRE_WORD+7}u],u32(nbParticleMass(p)));}`:""}_=nbSeedAppend(s,mandatory);}
}
@compute @workgroup_size(64) fn seedSources(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 storeSeedCandidate(seedCandidate(group,lane,true,true),true);
}
${this.seedRankWGSL()}
@compute @workgroup_size(64) fn seed(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let p=seedCandidate(group,lane,false,true);
 ${this.seedRank?`// A budget the overlap reseeds overrun keeps the first in rank order
 // (nbSeedAdmitted); a deferred cell is short again next step.
 nbSeedKept[lane]=select(0u,1u,p.position.x>=0.0);workgroupBarrier();
 if(p.position.x>=0.0&&!nbSeedAdmitted(nbSeedRankKey(group,lane),lane)){atomicAdd(&state[2],1u);return;}`:""}
 storeSeedCandidate(p,false);
}

@compute @workgroup_size(64) fn countInitial(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 for(var i=gid.x;i<UM_CELL_KEYS;i+=64u*groups.x){if(!umCellLive(i)){continue;}seedCell(umCellCoord(i),false);}
}
// The initial band appends to the scratch bank's slabs; SORT files it.
@compute @workgroup_size(64) fn seedInitial(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 for(var i=gid.x;i<UM_CELL_KEYS;i+=64u*groups.x){if(!umCellLive(i)){continue;}let c=umCellCoord(i);seedCell(c,true);}
}
fn weight(x:f32)->f32{let a=abs(x);if(a<0.5){return 0.75-a*a;}let b=max(0.0,1.5-a);return 0.5*b*b;}
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
   ${this.crowdedCells==="merge"?"let mass=nbStreamMass(i);total+=mass*w;momentum+=mass*w*u;":"total+=w;momentum+=w*u;"}
  }
 }}}
 if(total<1e-5){return original;}return mix(original,momentum/total+umThawFaceTerm(face.anchor,face.axis,face.width),blend*nbTransferBlend(q,depth,total));
}
${narrowBandTransferPropertiesSource(this.gradedDensity,this.sharedDeposit?"depositScratch":"bins",this.crowdedCells==="merge")}
${narrowBandFineTransferSource(subgroups,this.apic,this.gradedDensity,this.crowdedCells==="merge")}
${particleBrickDepositWGSL(this.apic,{joins:this.refineWord,allocation:this.refineWord+this.refineJoinWords},this.sharedDeposit?"depositScratch":"bins",this.crowdedCells==="merge",narrowBandSeparationReach(this.sampleSeparation,this.crowdedCells==="merge"))}
// The 4h owners, one a lane. An owner whose whole stencil is 4h has one
// patch a face, and a face centred in a tile past the reach lies more than
// two cells from the surface: it keeps its velocity without a gather. The
// rest are listed for transferSeams. Negative walls are untouched: their
// boundary condition is Uniform's.
@compute @workgroup_size(64) fn transferOwners(@builtin(global_invocation_id) gid:vec3u){
 // Deep-COLD owners are frozen: with the freeze, the active regions' 4h tiles.
 var owner=UMOwner();if(umActiveCoarse){owner=umActiveCoarseOwner(gid.x,false);}else{owner=umAllOwner(vec3u(umCounts.x*64u+gid.x,0,0));}
 if(owner.width==0u){return;}
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
 // Without live samples an owner beside h tiles keeps its velocity too: a
 // face toward them is sixteen unit patches, a texel each, and every one
 // keeps its departure (one writer a texel, as transferSeams). Left out,
 // they held the last frame's field: zero where that was air, under every
 // h front the stream advanced into.
 let hanging=umTileMinimumWidth(owner.tile)==1u;
 for(var axis=0u;axis<3u;axis++){
  if(hanging){
   let first=umFace(owner,axis,1,0u);
   if(first.count>1u){
    for(var part=0u;part<first.count;part++){
     let at=umFacePatch(first,part).anchor;let original=textureLoad(velocity,at,0);
     var value=vec4f(0,0,0,original.w);var writer=true;
     for(var other=0u;other<3u;other++){
      if(other!=axis&&umPositiveFaceAtAnchor(owner,other,at).width==0u){continue;}
      if(other<axis){writer=false;}value[other]=original[other];
     }
     if(writer){textureStore(output,at,value);}
    }
    continue;
   }
  }
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

// A lane reduces its own samples and touches the shared receipt once.
@compute @workgroup_size(64) fn diagnoseBefore(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 var outside=0.0;
 for(var i=gid.x;i<nbExtent();i+=64u*groups.x){
  if(!nbLive(particles[i])){continue;}
  outside=max(outside,bandPhi(particles[i].position.xyz));
 }
 atomicMax(&state[4],bitcast<u32>(outside));
}
${this.sampleBound==="walls"?narrowBandSurfaceWGSL.replace(NARROW_BAND_CENTRED_BOUND_WGSL,NARROW_BAND_WALL_BOUND_WGSL):narrowBandSurfaceWGSL}

`,[]);
  // Advection borrows the exact extended-field unit taps Uniform already
  // prepared (the move's G2P and seeds sample them too). The other stages read their own fresh 4h cache and
  // resolve fine seam taps in place.
  const modules=[false,true].map(unitTaps=>uniformDetailModule(this.device,{specializeWorldGeometry:true,label:`Uniform narrow-band FLIP ${unitTaps?"extended":"stage"}`,code:uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(sourceCode(unitTaps),narrowBandNearestEntries,"(atomicLoad(&bins[NB_BAND+1u])+3u)/4u"),["surfaceSplat"],"atomicLoad(&bins[NB_BAND+3u])"),["depthSeeds"],"atomicLoad(&bins[NB_BAND])"),["seed","seedSources",...(this.seedRank?["seedRank"]:[])],"(atomicLoad(&bins[NB_BAND+3u])+7u)/8u"),["transferProperties"],"umCounts.x"),["transfer"],subgroups?"select(16u*umCounts.x,0u,nbTransferFits())":"select(8u*umCounts.x,0u,nbTransferFits())"),["transferOwners"],"select((umCounts.y+63u)/64u,umActiveRegionCount(),umActiveCoarse)"),["transferSeams"],"atomicLoad(&bins[NB_BAND+3u])"),["coupleFine"],"umCounts.x"),["seedCells"],"atomicLoad(&bins[NB_BAND+1u])")}));
  for(const module of modules){
   const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
   if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  }
  if(!this.apic){
   const module=uniformDetailModule(this.device,{specializeWorldGeometry:true,label:"Narrow-band FLIP delta record",code:sourceCode(false,true)});
   const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
   this.deltaPipeline=await uniformDetailPipeline(this.device,this.ownership,{layout:this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.deltaResources]}),compute:{module,entryPoint:"recordDelta"}});
  }
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources,this.solid.tileLayout]});
  await Promise.all(["activity","activitySpread","advectParticlesDeferred","slabMark","slabAllocate","slabPlace","slabOverflow","slabPrepare","slabSort","surfaceClear","surfaceCells","surfaceSplat","surfaceSplatPlan","surfaceSplatOccupied","seedCells","seed","seedSources",...this.seedRankEntries,"countInitial","seedInitial","transferProperties","transfer","transferOwners","transferSeams","couple","coupleFine","diagnoseBefore","depthPlanSeed","depthPlanExpand","depthTiles","depthSeeds","depthLists",...narrowBandNearestControlEntries,...narrowBandNearestEntries,"depthReuseSelect@withX","depthTiles@cold","depthSeeds@cold","coarseCorners","coarseBuilds","depthCoarse","redistanceCoarse",
   "arenaMark",...UNIFORM_BRICK_TABLE_ENTRIES,"arenaGather","arenaGather@base","arenaGather@derived","arenaCross","arenaNearest","arenaDistance","arenaFar","arenaStore",...(this.solid.coarse?[...NB_CANONICAL_SURFACE_ENTRIES,"arenaGather@publication","arenaGather@publication@derived"]:[]),"airClearCoarse@planned","airPlanSeeds","airPlanPropagate","airPlanSeeds@fused","airPlanPropagate@fused","airSeeds","airSeedsFine","airSpread0","airSpread1","airSpread2","airGate","airClearFine","airClearCoarse",...NB_FREE_TWIN_ENTRIES.map(entry=>`${entry}@free`),"transferDeposit","transferRefine",...(this.sharedDeposit?["transferClear"]:[])].map(async name=>{
   const entryPoint=name.split("@")[0]!,module=modules[["advectParticles","advectParticlesDeferred","seed","seedSources","seedRank","seedCut"].includes(entryPoint)?1:0]!;
   if(this.omittedPipelines.has(name))return;
   const compile=()=>uniformMixedSolidPipeline(this.solid,s=>uniformDetailPipeline(this.device,this.ownership,{label:`Narrow-band FLIP ${entryPoint}`,layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX,...(entryPoint==="transferDeposit"?{PB_LANES:this.apic?192:128}:{}),...(entryPoint==="seedRank"||entryPoint==="seedCut"?{nbSeedLevel:Number(name.split("@")[1])}:{}),...(name.endsWith("@fused")?{nbAirFused:1,...(entryPoint==="airPlanPropagate"?{nbAirLanes:256}: {})}:{}),...(name.endsWith("@withX")?{nbXWithSelect:1}:{}),...(name.endsWith("@planned")?{nbAirListed:1}:{}),...(entryPoint==="depthReuseSnapshot"?{nbListWithSnapshot:1}:{}),umCountedJobs:entryPoint==="transfer"?3:0,...(entryPoint==="transferOwners"&&this.freeze?{umActiveCoarse:1}:{}),...(name.endsWith("@cold")?{nbColdCurrentPhi:1}:{}),...(name.endsWith("@derived")?{nbArenaDerived:1}:{}),...(name.includes("@publication")?{nbArenaClearPublication:1}:{}),...(entryPoint==="arenaCross"?{nbReachTiles:this.redistanceReach}:{}),...(NB_H_STORE_ENTRIES.has(entryPoint)&&!name.endsWith("@base")?UNIFORM_DETAIL_H_STORE:{}),...(entryPoint==="depthSeeds"?UNIFORM_DETAIL_MERGED_SLOTS:{}),...(name.endsWith("@free")?{umSolidFree:1}:NB_FREE_TWIN_ENTRIES.includes(name)?{umSolidTwinned:1}:{}),...s}}})).catch(e=>{throw new Error(`${entryPoint}: ${e.message} ${e.reason}`,{cause:e});});
   if(NB_BACKGROUND_PIPELINES.has(name)){this.background.add(name,compile);return;}
   this.pipelines.set(name,await compile());
  }));
  this.airFused=this.compile(["airPlanSeeds@fused","airPlanPropagate@fused"],undefined,"regional air update");
  this.airPlanned=this.compile(["airPlanSeeds","airPlanPropagate","airClearFine","airClearCoarse@planned"],undefined,"retire empty air");
  this.crossingDomain=this.compile(["depthPlanSeed","depthPlanExpand"],"surface","shared frame crossing domain");
  const search=["depthTiles","depthSeeds",...narrowBandNearestControlEntries,...narrowBandNearestEntries];
  this.batches={
   advect:this.compile(["advectParticlesDeferred"],"advectParticles","advectParticles"),
   // The slabs (uniform-particle-slab.wgsl): last move's slabs and the
   // spills mark their bricks, the shared table lists them and their ring,
   // one workgroup sizes the scratch slabs. After the move one workgroup
   // places the current slabs from the exact counts, the overflow joins its
   // bricks, and SORT files every slab by cell into the current bank.
   allocate:this.compile(["slabMark","brickTableList","slabAllocate"],"advectParticles","particle slabs"),
   sortPrepare:this.compile(["slabPlace","slabOverflow","slabPrepare"],"advectParticles","particle placement"),
   sort:this.compile(["slabSort"],"advectParticles","particle sort"),
   countInitial:this.compile(["countInitial","brickTableList","slabAllocate"],"bootstrap","countInitial"),seedInitial:this.compile(["seedInitial"],"bootstrap","seedInitial"),
   activity:this.compile(["activity","activitySpread"],"bootstrap","activity"),
   searchReconstruct:this.compile(search.map(n=>n==="depthTiles"||n==="depthSeeds"?`${n}@cold`:n),"reconstruct","crossing search"),
   // The surface brick arena (uniform-surface-brick-arena.wgsl): the
   // end-step search, the metric (after retire, the gather's derived pass
   // refreshes what the coarse stores moved) and the fine store.
   arenaSearch:this.compile(["arenaMark",...UNIFORM_BRICK_TABLE_ENTRIES,"arenaGather","arenaCross","depthLists","arenaNearest"],"redistance","surface arena search"),
   arenaSearchCoarse:this.compile(["arenaMark",...UNIFORM_BRICK_TABLE_ENTRIES,"arenaGather@base","arenaCross","depthLists","arenaNearest"],"redistance","surface arena search"),
   arenaMetric:this.compile(["arenaDistance","arenaFar"],"redistance","surface arena metric"),
   arenaMetricDerived:this.compile(["arenaGather@derived","arenaDistance","arenaFar"],"redistance","surface arena metric"),
   arenaStore:this.compile(["arenaStore"],"redistanceStore","surface arena store"),
   diagnose:this.compile(["diagnoseBefore"],"reconstruct","diagnoseBefore"),
   reconstruct:this.compile(["coarseCorners","surfaceClear",...(this.fusedSurface?["surfaceSplatPlan","surfaceSplatOccupied"]:["surfaceCells","surfaceSplat"]),"coupleFine@free","coupleFine","couple"],"reconstruct","surface reconstruction"),
   reconstructCoarse:this.compile(["coarseCorners","surfaceClear",...(this.fusedSurface?["surfaceSplatPlan","surfaceSplatOccupied"]:["surfaceCells","surfaceSplat"]),"couple"],"reconstruct","surface reconstruction"),
   // The deposit (DEPB) classifies against phase; FIN finishes the fine
   // cells it reached. Transfer loads canonical face texels directly in both
   // mixed and all-4h modes; nothing samples a cache. The general gather
   // (transfer) serves the layouts the deposit's words do not fit.
   transfer:this.compile([...(this.sharedDeposit?["transferClear"]:[]),"transferDeposit","transferRefine","transferProperties","transfer","transferOwners","transferSeams"],"transfer","transfer"),
   retire:this.compile(["coarseBuilds","depthCoarse","redistanceCoarse"],undefined,"coarse retirement"),
   // A retirement whose step built no coarse list (a standalone call) lists the corners itself.
   retireListed:this.compile(["coarseCorners","coarseBuilds","depthCoarse","redistanceCoarse"],undefined,"coarse retirement"),
   clearAir:this.compile(["airSeeds","airSeedsFine","airSpread0","airSpread1","airSpread2","airGate","airClearFine","airClearCoarse"],undefined,"retire empty air"),
   // The step's seeds, into the current slabs' holes at the transfer's head.
   seed:this.compile(["seedCells","seedSources",...this.seedRankEntries,"seed"],"advectParticles","seed"),
  };
  // The mask view requires the solid tile record. Other solvers keep the
  // baseline chain, which also remains available for metric-only work.
  if(this.solid.coarse)this.canonicalSurface={
   gather:this.compile(["arenaGather@publication@derived"],"redistance","surface canonical view"),
   far:this.compile(["arenaFar"],"redistance","surface canonical far metric"),
   publish:this.compile(["arenaDistance","arenaPublishCanonical","arenaPublishFinish"],"redistanceStore","surface canonical metric and publication"),
   search:this.compile(["arenaMark",...UNIFORM_BRICK_TABLE_ENTRIES,"arenaGather@publication","arenaCross","depthLists","arenaNearest"],"redistance","surface canonical search"),
  };
  const listedSearch=search.filter(n=>n!=="depthNearestX").map(n=>n==="depthReuseSelect"?"depthReuseSelect@withX":n);
  this.crossingSearches.set(this.batches.searchReconstruct,this.compile(listedSearch.map(n=>n==="depthTiles"||n==="depthSeeds"?`${n}@cold`:n),"reconstruct","crossing search"));
 }
 /** The seed budget's canonical admission (narrowBandSeedRankWGSL). */
 private get seedRank():NarrowBandSeedRank|undefined{return this.seedBudget?narrowBandSeedRankShape(this.cells,this.tiles):undefined;}
 private seedRankWGSL():string{const rank=this.seedRank;return rank?narrowBandSeedRankWGSL(rank,this.seedBudget):"";}
 /** The seed batch's rank steps: seedRank@level and seedCut@level. */
 private get seedRankEntries():string[]{const rank=this.seedRank;return rank?Array.from({length:rank.levels},(_,l)=>[`seedRank@${l}`,`seedCut@${l}`]).flat():[];}
 private compile(entries:readonly string[],groupName:string|undefined,label:string):NarrowBandBatch{
  const pass={label:`Narrow-band FLIP ${label}`},omitted=entries.find(name=>this.omittedPipelines.has(name));
  if(omitted!==undefined)return {pass,steps:[],unavailable:`${omitted} is not compiled: no stage of this configuration binds it`};
  if(entries.some(name=>this.background.has(name)))return {pass,steps:[],deferred:{entries,groupName}};
  return {pass,steps:this.steps(entries,groupName)};
 }
 private steps(entries:readonly string[],groupName:string|undefined):NarrowBandStep[]{
  return entries.map(name=>{
   const selected=this.pipelines.get(name)??(this.background.has(name)?this.background.get(name):undefined),entry=name.split("@")[0]!;if(!selected)throw new Error(`Narrow-band FLIP ${name} is not initialized`);
   return {
    groups:groupName??((entry.startsWith("airClear")||name==="airPlanPropagate@fused")?"redistanceStore":entry.startsWith("air")?"surface":entry.startsWith("redistance")?"redistanceStore":entry.startsWith("coarse")?"surface":entry==="surfaceCells"||entry==="surfaceSplat"||entry==="diagnoseBefore"||entry.startsWith("depth")?"surface":entry.startsWith("transfer")?"transfer":entry==="couple"||entry==="coupleFine"?"surface":entry),
    // In-place redistance reads its own canonical texel: coarse vertices
    // always read the base home, including those outside the resident h ring.
    base:entry==="redistanceCoarse"||entry==="airClearCoarse",pipeline:uniformDetailPick(this.solid.select(selected)),
    launch:["depthTiles","depthPlanSeed","depthPlanExpand","depthReuseSnapshot","depthReuseSelect","airPlanSeeds","airPlanPropagate"].includes(entry)?NB_LAUNCH.regions:entry==="transfer"?NB_LAUNCH.counted:entry==="slabAllocate"||entry==="slabPlace"||entry==="seedCut"?NB_LAUNCH.one
     :entry==="advectParticles"?NB_LAUNCH.particles:NB_LAUNCH.grid,
   };
  });
 }
 private get compiled():NarrowBandBatches{if(!this.batches)throw new Error("Narrow-band FLIP is not initialized");return this.batches;}
 private run(encoder:GPUCommandEncoder,batch:NarrowBandBatch,shared?:GPUComputePassEncoder):void{
  // The advance awaited these (pipelinesPending): get throws if it did not.
  if(batch.deferred){batch.steps=this.steps(batch.deferred.entries,batch.deferred.groupName);batch.deferred=undefined;}
  if(batch.unavailable)throw new Error(`${batch.pass.label}: ${batch.unavailable}`);
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
   else if(step.launch===NB_LAUNCH.regions)pass.dispatchWorkgroups(Math.min(NB_REGION_GROUPS,this.reuse.regions));
   else pass.dispatchWorkgroups(1024);
  }
  if(!shared)pass.end();
 }
 /** Refresh the step and cap together, retaining the latest activity settings
  * even when a static layout skips the census. */
 private writeStep(dt:number,openTop:number):void{
  const w=this.stepWords,h=this.ownership.capacity.lattice.cellSize_m;
  w[0]=h[0]!;w[1]=h[1]!;w[2]=h[2]!;w[3]=dt;w[4]=this.apic?0:0.95;w[5]=openTop;w[6]=0;w[7]=+this.measureTraces;
  w[11]=this.particleCellCap;
  this.device.queue.writeBuffer(this.params,0,w);
  this.sparseWords[0]=this.sparseNeighbours;this.device.queue.writeBuffer(this.params,52,this.sparseWords);
 }
 /** gravity: the step's vector, m/s^2. */
 move(encoder:GPUCommandEncoder,dt:number,openTop:boolean,gravity:readonly [number,number,number]):void{
  if(this.idleNow)return;
  const b=this.compiled;
  this.crossingCanPlan=false;this.crossingIsPlanned=false;this.framePlan?.invalidateDomain(this);
  this.writeStep(dt,+openTop);
  // params.live.z: the reseed's salt, this move's frame index (from 1).
  this.seedFrameWords[0]=this.seedFrameWords[0]!+1;this.device.queue.writeBuffer(this.params,56,this.seedFrameWords);
  {const g=this.gravityWords;g[0]=gravity[0];g[1]=gravity[1];g[2]=gravity[2];this.device.queue.writeBuffer(this.params,64,g);}
  // Words 4 to 10 are the step's; word 11, the arrival fatal's page, stays.
  encoder.clearBuffer(this.state,16,28);
  this.bootstrap(encoder);
  // G2P reads the samples' depth from the last step's band (its tiers
  // too: recordDelta's, so a COLD sample takes no delta from either).
  if(!this.bandCurrent){this.arenaSearch(encoder);this.run(encoder,b.arenaMetric);}this.bandCurrent=false;
  encoder.clearBuffer(this.bins,(this.activityWord+5*this.tiles)*4,2*this.tiles*4);
  // The slabs are sized from the last move's, the samples move into them
  // in the scratch bank, and SORT files them by cell into the current bank
  // and marks the tiles they reach.
  encoder.clearBuffer(this.state,4,8);encoder.clearBuffer(this.state,4*NB_SLAB_SEEDS_WORD,4);
  encoder.clearBuffer(this.bins,4*this.countWords+this.tiles*4,this.tiles*4);
  this.clearBrickTable(encoder);this.run(encoder,b.allocate);
  encoder.clearBuffer(this.state,4*NB_MOVE_TICKET_WORD,4);this.run(encoder,b.advect);
  this.sort(encoder);this.crossingCanPlan=true;
  // The count every particle consumer after the move gates on (liveGate).
  encoder.copyBufferToBuffer(this.state,0,this.params,48,4);encoder.copyBufferToBuffer(this.state,0,this.liveGate,0,4);
 }
 private bootstrap(encoder:GPUCommandEncoder):void{
  if(this.started)return;
  const b=this.compiled;
  encoder.clearBuffer(this.state,4,8);encoder.clearBuffer(this.bins,0,4*this.countWords);
  encoder.clearBuffer(this.bins,4*this.countWords+this.tiles*4,this.tiles*4);
  this.arenaSearch(encoder);this.run(encoder,b.arenaMetric);this.cache.encode(encoder,this.cacheGroups.bootstrap!);
  // Coverage and the occupied list start empty: countInitial lists its tiles.
  encoder.clearBuffer(this.bins,4*this.countWords,this.tiles*4);encoder.clearBuffer(this.bins,4*this.occupiedWord,4);
  this.clearBrickTable(encoder);this.run(encoder,b.countInitial);
  this.run(encoder,b.seedInitial);this.sort(encoder);
  this.started=true;
 }
 /** The particle stage builds the shared brick table at its move's head and
  * reads its slots only inside the move (uniform-brick-table.ts). */
 private clearBrickTable(encoder:GPUCommandEncoder):void{
  const table=uniformBrickTableClear(this.brickWord,this.address.blocks);encoder.clearBuffer(this.bins,table.offset,table.size);
 }
 /** SORT: the last epoch's occupied tiles are cleared, then the list. */
 private sort(encoder:GPUCommandEncoder):void{
  const b=this.compiled;
  this.run(encoder,b.sortPrepare);encoder.clearBuffer(this.bins,4*this.occupiedWord,4);this.run(encoder,b.sort);
 }
 /** Called only for enabled automatic refinement. Explicit Requested/Full
  * layouts keep their authored semantics; particle lifetime is independent. */
 refine(encoder:GPUCommandEncoder,target:{buffer:GPUBuffer;wordOffset:number;words?:number},padding=1):void{
  if(this.idleNow)return;
  // transferRefine wrote the census's prediction from the last step's
  // samples (the move's accepted grid motion, DEPB's mode, the band's nearest
  // crossing): copy it to the census's planes, then clear it for the next.
  this.refinePadding=padding;
  if(target.words!==this.refineJoinWords)throw new Error(`The refinement join plane holds ${target.words??"an unstated number of"} words; a bit per tile key needs ${this.refineJoinWords}.`);
  const joins=4*this.refineJoinWords,allocation=this.particleAllocation.size;
  encoder.copyBufferToBuffer(this.bins,4*this.refineWord,target.buffer,4*target.wordOffset,joins);
  encoder.copyBufferToBuffer(this.bins,4*this.refineWord+joins,this.particleAllocation,0,allocation);
  encoder.clearBuffer(this.bins,4*this.refineWord,joins+allocation);
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
  const w=this.activityWords;w[0]=2/Math.max(0.05,fadeSeconds);w[1]=sparseSeconds;w[2]=this.refinePadding;w[3]=this.particleCellCap;
  this.device.queue.writeBuffer(this.params,32,w);
  this.writeStep(dt,0);
  if(this.updateErrors)encoder.copyBufferToBuffer(this.updateErrors,32*this.tiles,this.bins,4*(this.activityWord+7*this.tiles),16*this.tiles);
  encoder.copyBufferToBuffer(importance.buffer,importance.offset,this.bins,this.activityWord*4,this.tiles*8);
  this.run(encoder,this.compiled.activity);
 }
 reconstruct(encoder:GPUCommandEncoder):void{if(this.idleNow)return;this.syncSurfaceRevisions(encoder,true);const b=this.compiled;this.bandCurrent=false;this.measureBand(encoder,b.searchReconstruct);if(this.measureTraces)this.run(encoder,b.diagnose);
  // The coarse corner list starts empty; retirement reuses it under the same frame plan.
  encoder.clearBuffer(this.bins,4*this.coarseListWord,16);this.coarseListSerial=this.framePlan?.active&&this.tierPlanSerial===this.framePlan.serial?this.framePlan.serial:-1;
  this.run(encoder,this.ownership.coarseOnly?b.reconstructCoarse:b.reconstruct);}
 transfer(encoder:GPUCommandEncoder):void{
  if(this.idleNow)return;
  // Clearing a buried zero invalidates the crossing/metric banks as well
  // as phi. The caller resolved phi before momentum or membership reads it.
  if(!this.bandCurrent){this.arenaSearch(encoder);this.run(encoder,this.compiled.arenaMetric);this.bandCurrent=true;}
  // The step's seeds (none in a quiet frame) join the holes past the
  // sorted runs against this step's band, phi and counts, sampling the
  // field the move sampled (storeSeedCandidate). Their candidate count is NB_BAND+3.
  if(!this.quiet){encoder.clearBuffer(this.bins,(this.bandWord+3)*4,4);const rank=this.seedRank;if(rank)encoder.clearBuffer(this.bins,4*(this.bandWord+4+6*this.tiles),4*(rank.levels*(1<<rank.bits)+3));this.run(encoder,this.compiled.seed);}
  // The seam list's length (NB_BAND+3), which transferOwners counts.
  encoder.clearBuffer(this.bins,(this.bandWord+3)*4,4);
  this.run(encoder,this.compiled.transfer);
 }
 /** The FLIP delta into before.xyz, after the pair tail extension: a
  * workgroup an occupied tile stages its samples' face window once (the
  * tail's delta pool, where it swept the face), then interpolates it at
  * each sample. APIC's G2P reads no delta. */
 recordDelta(encoder:GPUCommandEncoder):void{
  if(this.idleNow||this.apic)return;
  if(!this.deltaPipeline||!this.deltaGroup)throw new Error("Narrow-band FLIP delta is not initialized");
  // The move/seeds still carry their accepted grid velocity here. Preserve
  // its display energy before before.xyz becomes a delta (or sentinel).
  this.displayMotion.capture(encoder);
  // recordDelta and the next move's G2P read the same tiers (nbMoveCold).
  this.syncTiers(encoder);const pass=encoder.beginComputePass(NB_DELTA_PASS);
  pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.deltaGroup.group);pass.setPipeline(uniformDetailPick(this.deltaPipeline));
  pass.dispatchWorkgroups(Math.min(4096,this.tiles));pass.end();
 }
 /** Adaptive retirement can move a hanging zero when it normalizes coarse
  * corner magnitudes. Store those corners first, from the arena search's
  * metric, so the redistance is the end-step one. The hanging texels it
  * moves are not resolved before redistance: the arena's derived gather
  * re-forms them from the 4h corners. */
 retire(encoder:GPUCommandEncoder):void{
  if(this.ownership.coarseOnly||this.idleNow)return;
  this.arenaSearch(encoder);
  // The list holds this step's ownership and tiers only under the frame plan reconstruct built it in.
  const listed=this.coarseListSerial>=0&&this.framePlan?.active===true&&this.coarseListSerial===this.framePlan.serial&&this.tierPlanSerial===this.framePlan.serial;
  encoder.clearBuffer(this.bins,4*this.coarseListWord+(listed?4:0),listed?4:16);
  this.run(encoder,listed?this.compiled.retire:this.compiled.retireListed);this.retireSearched=true;
 }
 /** Fine redistance preserves the crossing set, so its search and metric
  * also serve end-step particle membership. Adaptive coarse corners were
  * stored by retire(); only they could have moved a zero. */
 redistance(encoder:GPUCommandEncoder):void{
  if(this.idleNow)return;
  const b=this.compiled;
  // Shared: retirement's search stands (it only stores coarse corners);
  // the derived gather refreshes the windows' hanging vertices.
  if(this.canonicalSurface&&!this.ownership.coarseOnly){
   const shared=this.redistanceSearch!=="fresh"&&this.retireSearched;
   if(!shared)this.arenaSearch(encoder,true);
   // Search already gathered a fresh view. Retirement instead needs the
   // derived view of the coarse corners it changed before metric queries.
   if(shared)this.run(encoder,this.canonicalSurface.gather);
   this.run(encoder,this.canonicalSurface.far);
   this.run(encoder,this.canonicalSurface.publish);
  }else{
   if(this.redistanceSearch==="fresh"||!this.retireSearched){this.arenaSearch(encoder);this.run(encoder,b.arenaMetric);}
   else this.run(encoder,b.arenaMetricDerived);
   if(!this.ownership.coarseOnly)this.run(encoder,b.arenaStore);
  }
  this.retireSearched=false;this.bandCurrent=true;
 }
 /** retire() ran the arena search and no other search has run since. */
 private retireSearched=false;
 /** The surface brick arena's search: the brick table, the windows, the
  * band's masks, reach and lists, and bank B (uniform-surface-brick-arena.wgsl). */
 private arenaSearch(encoder:GPUCommandEncoder,publication=false):void{
  this.retireSearched=false;this.syncSurfaceRevisions(encoder);this.syncTiers(encoder);
  encoder.clearBuffer(this.bins,this.bandWord*4,(4+3*this.tiles)*4);
  const table=uniformBrickTableClear(this.brickWord,this.address.blocks);encoder.clearBuffer(this.bins,table.offset,table.size);
  this.run(encoder,this.ownership.coarseOnly?this.compiled.arenaSearchCoarse:publication&&this.canonicalSurface?.search?this.canonicalSurface.search:this.compiled.arenaSearch);
 }
 /** Clear only the certified empty field outside physical liquid/particle support.
  * It sets state word NARROW_BAND_AIR_CHANGED_WORD where it changes a 4h-lattice vertex. */
 clearAir(encoder:GPUCommandEncoder):void{
  const planned=this.crossingIsPlanned&&this.framePlan?.active&&this.crossingPlanSerial===this.framePlan.serial;
  const fused=planned&&this.framePlan!.scheduleStages&&this.framePlan!.fuseAirClear;
  if(!fused||this.idleNow)encoder.clearBuffer(this.state,4*NARROW_BAND_AIR_CHANGED_WORD,4);
  if(this.idleNow)return;this.syncTiers(encoder);
  this.run(encoder,fused?this.airFused!:planned?this.airPlanned!:this.compiled.clearAir);this.bandCurrent=false;
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
  // Under previousCrossings the masks and reach are the last arena search's:
  // only the counters are cleared, and bulkDepth reads the spread reach
  // (ownership.speeds' second half) from the candidate list's words.
  encoder.clearBuffer(this.bins,this.bandWord*4,(wanted||this.previousCrossings?4:4+3*this.tiles)*4);
  if(this.previousCrossings)encoder.copyBufferToBuffer(this.ownership.speeds,4*this.tiles,this.bins,4*(this.bandWord+4+3*this.tiles),4*this.tiles);
  const planned=wanted&&this.framePlan!.encodeDomain(encoder,this,this.encodeCrossingDomain);
  this.crossingIsPlanned=!!planned;this.crossingPlanSerial=planned?this.framePlan!.serial:-1;
  if(!planned)encoder.clearBuffer(this.bins,4*this.reuse.framePlan,4);
  if(!this.previousCrossings)this.run(encoder,planned?this.crossingSearches.get(batch)??batch:batch,this.crossingPass);
  this.crossingPass?.end();this.crossingPass=undefined;
 }
 /** latest: no later frame's receipt was noted (receipts can finish out of
  * order): only then are its counts the current ones. Every receipt is
  * checked and adds its clipped reseeds. */
 noteReceipt(words:Uint32Array,latest:boolean):void{
  this.reseedClippedTotal+=words[2]!;
  if(latest){
   this.count=words[1]!;this.reseedClipped=words[2]!;
   const bits=this.receiptBits,distances=this.receiptDistances,d=this.diagnostics;
   bits[0]=words[4]!;bits[1]=words[5]!;
   d.beforeMaxOutside=this.measureTraces?distances[0]!:NaN;d.unsupported=words[7]!;
   // The surface census runs while traces are measured (the deposit's).
   d.afterMaxOutside=this.measureTraces?distances[1]!:NaN;d.outsideSurface=this.measureTraces?words[6]!:NaN;d.deepInterior=this.measureTraces?words[8]!:NaN;
  }
  // Persistent samples own the budget first. At capacity optional reseeding
  // is deferred (reseedClipped); never destroy a surface to make room for it.
  // Liquid a source or the initial band must add has no such deferral.
  if(words[3]===4){const p=words[11]!;throw new Error(`Narrow-band FLIP: a particle arrived in page (${p&255},${(p>>>8)&255},${(p>>>16)&255}), which holds no storage; the residency closure is too small`);}
  const budget=this.budget&&(words[3]>=5&&words[3]<=7||words[3]===9||words[3]===10)?`; fluid.tiers.bricks and fluid.tiers.particlesPerFineCell size it (${this.budget.fineTiles} h tiles at ${this.budget.perFineCell} a fine cell)`:"";
  if(words[3])throw new Error(`Narrow-band FLIP ${words[3]===1?`trajectory longer than the lattice (${NARROW_BAND_TRACE_REACH} cells an axis)`:words[3]===2?"nonfinite velocity"
   :words[3]===5?`particle bank is full: brick overflow exhausted the free tail of the ${this.capacity}-record pool`
   :words[3]===6?`particle slabs exceed the pool: ${words[0]} records sized of ${this.capacity-this.spill} before the spill reserve (${words[1]} live)`
   :words[3]===7?`seed candidates exceed their list (${3*this.capacity} cells)`
   :words[3]===8?`coarse corners exceed their list (${this.coarseListCapacity} entries)`
   :words[3]===9?`particle bricks exceed the cell bins' ${this.binSlots} brick slots`
   :words[3]===10?`surface arena bricks (${words[11]}) exceed the arena's ${this.arenaWindows} windows`
   :words[3]===11?`an h tile's rank lies past the frame extension's ${this.ownership.capacity.fineTiles} FLIP delta rows`
   :`unknown particle fatal ${words[3]}`}${budget}`);
 }
 destroy():void{this.displayMotion.destroy();this.reuseControls.destroy();this.coarseVelocity.destroy();this.particleAllocation.destroy();for(const b of [...this.particles,this.state,this.bins,this.next,this.params,this.liveGate])b.destroy();}
}
