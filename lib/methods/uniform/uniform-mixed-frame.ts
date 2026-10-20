import { settleGPUInitialization } from "../../core/gpu-initialization";
import { UniformNarrowBandFlip } from "./uniform-narrow-band-flip";
import { NARROW_BAND_AIR_CHANGED_WORD } from "./uniform-narrow-band-air.wgsl";
import {UNIFORM_MIXED_OVERFLOW_FINE,uniformMixedDetailViolationWord} from "./uniform-mixed-topology.wgsl";
import {uniformMixedHangingBytes} from "./uniform-mixed-velocity-sampling.wgsl";
import {UNIFORM_DETAIL_POLICY} from "./uniform-detail-policy";
import {UNIFORM_DETAIL_VIOLATION,type UniformDetailGroup,type UniformDetailStorage} from "./uniform-detail-fields";
import {UniformPressureSurfaceBand} from "./uniform-pressure-surface-band";
import {UNIFORM_PRESSURE_ARENA_RETURNS_RESIDUAL,UNIFORM_PRESSURE_ARENA_SCHEDULE,UniformPressureArena} from "./uniform-pressure-arena";
import {UniformComputeBlit} from "./uniform-compute-blit";
import {UniformMixedFramePlan} from "./uniform-mixed-frame-plan";
import {UniformMixedOwnershipTransfer,UniformMixedRemap} from "./uniform-mixed-remap";
import {UniformMixedPhiResolve} from "./uniform-mixed-phi-resolve";
import {UniformCoarseVertexPhi} from "./uniform-coarse-vertex-phi";
import {uniformMixedAllCoarseLayout} from "./uniform-mixed-layout";
import {UNIFORM_MIXED_ROOT_FATAL,UNIFORM_MIXED_ROOT_FATAL_MESSAGE,UNIFORM_MIXED_ROOT_UNREACHED,UNIFORM_MIXED_ROOT_UNREACHED_MESSAGE,UniformMixedRootOwners} from "./uniform-mixed-root-owners";
import {UniformBlockNeighbours} from "./uniform-block-neighbours";
import {UNIFORM_RESIDENCY_PAGE_REACH} from "./uniform-mixed-residency";
import type {UniformMixedLayout} from "./uniform-mixed-layout";
import type {UniformScratchArena} from "./uniform-scratch-arena";
import type {WebGPUUniformPressureMultigrid} from "./webgpu-uniform-pressure-multigrid";
import type {WebGPUUniformVelocityExtrapolator} from "./webgpu-uniform-velocity-extrapolation";
import {UNIFORM_WORK_RECEIPT_WORDS,UniformMixedOwnership,type UniformMixedGenerationBuffers} from "./uniform-mixed-ownership";
import {UniformMixedExtension} from "./uniform-mixed-extension";
import {UniformMixedHangingTaps, UniformMixedMomentumCache} from "./uniform-mixed-momentum-cache";
import {uniformCoarseCacheExtent} from "./uniform-coarse-velocity-cache";
import {UniformMixedSurface} from "./uniform-mixed-surface";
import {UniformSurfaceWorkControl} from "./uniform-surface-work-control";
import {NB_VOLUME_RATE_FLOAT,UniformMixedSurfaceVolume} from "./uniform-mixed-surface-volume";
import {UniformMixedSurfaceGeometry} from "./uniform-mixed-surface-geometry";
import {UNIFORM_WORK_RELAYOUT_RESERVE,type UniformWorkEdit} from "./uniform-buffered-work";
import {UniformMixedMomentum,UNIFORM_MIXED_MOMENTUM_LIMITS} from "./uniform-mixed-momentum";
import {UniformMixedForces} from "./uniform-mixed-forces";
import { UniformMixedRegionDynamics } from "./uniform-mixed-region-dynamics";
import { UniformRegionHeat } from "./uniform-region-heat";
import { UniformColdFreeze } from "./uniform-cold-freeze";
import { UNIFORM_COLD_FRAME_EVIDENCE_WORDS, UNIFORM_COLD_FRAME_HEAT_WORDS, UniformColdFrame } from "./uniform-cold-frame";
import {UniformMixedPressureAuthority,uniformMixedRootMaskWords,uniformMixedRootTopologyBytes} from "./uniform-mixed-pressure-authority";
import {uniformMixedPressureStorage} from "./uniform-mixed-pressure-boundary.wgsl";
import {UniformMixedPressureVelocity} from "./uniform-mixed-pressure-velocity";
import {UniformMixedPressureCycles,type UniformMixedPressureCycleLevel} from "./uniform-mixed-pressure-cycles";
import {UniformMixedPressureAcceptance} from "./uniform-mixed-pressure-acceptance";
import {UNIFORM_MIXED_SCHEDULE_FLOOR,UniformMixedPressureSchedule,uniformMixedPressureReserve,uniformMixedPressureSpareFull,type UniformMixedPressurePlan} from "./uniform-mixed-pressure-schedule";
import {UNIFORM_MIXED_FAILURE,UNIFORM_MIXED_STATUS,UNIFORM_MIXED_STATUS_WORDS,describeUniformMixedFrameStatus} from "./uniform-mixed-frame-status";
import {planUniformMixedPressureMemory} from "./uniform-mixed-pressure-memory";
import {DEFAULT_UNIFORM_CM11A_SCHEDULE,UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE,UNIFORM_PRESSURE_BAND_RESIDUAL_TOLERANCE,UNIFORM_PRESSURE_KICK_RESIDUAL_TOLERANCE,UNIFORM_PRESSURE_RELATIVE_REDUCTION,type UniformCM11aSchedule} from "./pressure-policy";
import type {GPUTimestampPhase} from "../../core/performance-trace";
import {UNIFORM_ADVANCE_PHASE as A} from "./uniform-stages";
import {UNIFORM_VOLUME_PHASE as V} from "./uniform-volume-stages";
import {UniformMixedSolid,type UniformMixedSolidResources} from "./uniform-mixed-solid.wgsl";
import {UniformTileContext} from "./uniform-tile-context";
import {UNIFORM_PAGE_POOL_STATUS,UniformPagePool,uniformPagePoolFatalMessage,type UniformPagePoolMirror,type UniformPagePoolResidency} from "./uniform-page-pool";
import {UniformPageInitialise,UniformPageTileWords,uniformPageTopology,type UniformPageInitBlocks} from "./uniform-page-initialise";
import {UNIFORM_STAGE_GRID_HEADER_WORDS,UNIFORM_STAGE_VIEWS,uniformStageBandWords,uniformStageGridHeader,uniformStageGridWords,uniformStageViewWord} from "./uniform-stage-grids";
import {uniformNearSettled,type UniformNearSettled} from "./uniform-near-settled";

export interface UniformMixedFrameTrace {
 instrument(encoder:GPUCommandEncoder):GPUCommandEncoder;
 phase(encoder:GPUCommandEncoder,phase:GPUTimestampPhase):void;
 submit(encoder:GPUCommandEncoder,anchor:GPUBuffer):void;
 submitted():void;
 abort():void;
}

export interface UniformMixedFrameFields {
 arena:UniformScratchArena;
 volume:GPUTexture;volumeScratch:GPUTexture;
 velocity:GPUTexture;velocityScratch:GPUTexture;departure:GPUTexture;
 negative:GPUBuffer;negativeScratch:GPUBuffer;negativeDeparture:GPUBuffer;
 phi:GPUTexture;phiScratch:GPUTexture;
 phase:GPUTexture;centerPhi:GPUTexture;target:GPUTexture;correction:GPUTexture;
 pressure:ReturnType<WebGPUUniformPressureMultigrid["prepareMixedContinuation"]>;
 extension:ReturnType<WebGPUUniformVelocityExtrapolator["prepareMixedContinuation"]>;
 sourceParams:GPUBuffer;
 /** Voxel/terrain/vessel solids and rigid bodies: live voxel edits arrive
  * through editSolids(), bodies through advance(). The host must keep every
  * coarse owner a full tile away from a cut cell liquid can reach. */
 solid:UniformMixedSolidResources;

 /** Two-stage pressure: an all-4h solve, then the h surface band
  * (UniformPressureBand). With solids, the all-4h levels read the static
  * coarse solid record and every liquid solid-coupled tile joins the band.
  * The solve always runs split, an all-4h simulation too (its transfer is the
  * identity and its band empty): in the all-4h ownership with its own surface
  * target and centre phi here, the simulation's own staying intact for the
  * renderer. */
 pressureGeometry:{target:GPUTexture;centerPhi:GPUTexture};
 /** The owner's own copy of the page table and block words (its t=0
  * kernels read it): a mirror of the page pool, beside the two topologies. */
 pages:UniformPagePoolMirror;
 /** The scene's stored set at build (uniformPagePoolResidency): the address
  * was seeded with its AIR pages; its pins are fixed here. */
 residency:UniformPagePoolResidency;
 /** The h fields' detail storage: its table rides the simulation and
  * pressure-root topologies (group 0 of every field stage). */
 detail?:UniformDetailStorage;
}
export interface UniformMixedFrameParameters {
 dt:number;gravity:number;density:number;viscosity:number;surfaceTension:number;
 openTop:boolean;noSlip:boolean;cubic:boolean;
 /** With cubic: h vertices advect phi trilinearly, 4h vertices keep the cubic. */
 fineTrilinear?:boolean;
 /** Encode the existing maximum slot envelope; convergence still stops execution. */
 fullPressureEnvelope?:boolean;
 /** Extra slots beyond the lagged planner's existing spare; bounded by the configured maximum. */
 pressureReserve?:number;
 /** Explicit source volume added this step, in h-cell units (NB scalar budget). */
 addedVolumeCells?:number;
 /** NB-FLIP volume control: seconds over which the liquid's gap to its budget is closed; zero for none. */
 volumeControlSeconds?:number;
 redistance?:boolean;extensionSweeps?:number;
 supportPolicy?:{fineReach:number;shellReach:number;twoLevel:boolean;shellOnly:boolean};
 pressureTolerance:number;
}

/** One owner-driven frame sequence. All persistent fields and large scratch
 * ranges belong to the host. Fine and coarse are layouts of this sequence.
 * Only compact metadata, parameter/receipt buffers and 4h sampling caches are
 * owned here. No readback contains simulation fields. */
/** Plan support, phase and extension are dt-free (the certificate takes dt),
 * so a census extension serves the next frame whatever its step. */
/** Diagnostics: log each frame's pressure slots, coarse solves and band history. */
/** Receipt bytes of the detail violation words (simulation, pressure root). */
const DETAIL_RECEIPT=120+4*UNIFORM_MIXED_STATUS_WORDS;
const WORK_RECEIPT=DETAIL_RECEIPT+8;
/** encodeGeometry's options once only bodies moved since the last gather. */
const BODY_GEOMETRY={bodies:true} as const;
const REMAP_WORK_RECEIPT=WORK_RECEIPT+4*UNIFORM_WORK_RECEIPT_WORDS;
// The root cycle list's count: the width of the root's listed launches.
const ROOT_WORK_RECEIPT=REMAP_WORK_RECEIPT+4;
const FLIP_RECEIPT=ROOT_WORK_RECEIPT+4;
// The page pool's status (UNIFORM_PAGE_POOL_STATUS).
const POOL_RECEIPT=FLIP_RECEIPT+48;
// The region heat header (UniformColdFrame's evidence).
const HEAT_RECEIPT=POOL_RECEIPT+4*UNIFORM_PAGE_POOL_STATUS.words;
const TRACE_RECEIPT=HEAT_RECEIPT+4*UNIFORM_COLD_FRAME_HEAT_WORDS;
// A source's added volume is the scalar budget's alone (beginStep): the
// source itself writes liquid after the head, inside the frame it is due in,
// so a frame with a drop starts from the last tail's extension like any other.
/** The extension key: an advance's parameters less its step and added
 * volume, in the frame's own copy (the caller may reuse its object). Every
 * field starts undefined: a field that only ever held doubles is boxed again
 * by each load until its reader is optimized, where one that held anything
 * else keeps the number by reference. */
type ExtensionKey={-readonly [K in Exclude<keyof UniformMixedFrameParameters,"dt"|"addedVolumeCells">]:UniformMixedFrameParameters[K]|undefined};
const extensionKey=():ExtensionKey=>({gravity:undefined,density:undefined,viscosity:undefined,surfaceTension:undefined,openTop:undefined,noSlip:undefined,cubic:undefined,fineTrilinear:undefined,
 fullPressureEnvelope:undefined,pressureReserve:undefined,volumeControlSeconds:undefined,
 redistance:undefined,extensionSweeps:undefined,supportPolicy:undefined,pressureTolerance:undefined});
/** Two key values agree (NaN is itself). */
const agree=(a:unknown,b:unknown):boolean=>a===b||(a!==a&&b!==b);
const FRAME_ENCODER:GPUCommandEncoderDescriptor={label:"Uniform owner-driven frame"};
/** The authority's parameter words, every frame's and a kick's: the gauge's
 * rate arrives on the GPU (see advance). */
const AUTHORITY_WORDS=new Float32Array([0,-1,0,0]);
/** Frames whose receipts may be unchecked at once: the host's frames-ahead cap. */
export const UNIFORM_MIXED_RECEIPT_RING=2;
/** Rigid bodies this frame (the solid library reads their GPU state):
 * their cut cells move, so the all-4h record, cut widths and geometry are
 * rebuilt and entered cells displaced at the head. couple, when coupling is
 * on, encodes the fluid-to-body exchange and the body integration after the
 * projection, before the census. */
export interface UniformMixedFrameBodies{couple?:(encoder:GPUCommandEncoder)=>void}
/** Dynamic coarsening at horizon one (UniformMixedFrame.setRelayout): the
 * census and layout builder the frame head encodes, with no readback. */
export interface UniformMixedFrameRelayout{
 /** Census and build for this frame's dt, after the head's extension of the
  * state the frame starts from and before the adopt. */
 encode(encoder:GPUCommandEncoder,dt:number,views:boolean):void;
 /** views: the census's band reasons (UNIFORM_STAGE_REASON, one word per
  * tile), written by an encode with views on. */
 readonly reasons:{readonly buffer:GPUBuffer;readonly offset:number};
 /** views: the census's detail importance (UNIFORM_STAGE_IMPORTANCE, two
  * words per tile). Every encode writes them; one with views on scores
  * every criterion. */
 readonly importance:{readonly buffer:GPUBuffer;readonly offset:number};
 /** The built generation's buffers (UniformMixedLayoutBuilder.generation). */
 readonly generation:UniformMixedGenerationBuffers;
 /** The builder's relayout receipt (UNIFORM_MIXED_RELAYOUT_RECEIPT). */
 readonly receipt:{readonly buffer:GPUBuffer;readonly offset:number;readonly words:number};
 /** Encoded in the adopt's blit run (after the build), e.g. a diagnostic
  * copy of the receipt. */
 adopted?(encoder:GPUCommandEncoder):void;
 /** Asked once per advance. False: nothing the GPU decides can have moved
  * the layout since the last build (its requests are the host's and are
  * built; no body, no wet solid, no join), so this frame keeps the
  * generation and its head runs no census. Absent: every frame builds. */
 due?():boolean;
 /** Asked once per advance that is due. A number: since the last build
  * only host requests moved, on a layout only they hold, and the build can
  * change the width of its tiles and the class of its reach at most
  * (UniformWorkEdit). The launch budgets then stay and carry the counts
  * (observeWork) instead of starting at the ceilings.
  * Absent or undefined: the build may admit any tile. */
 edit?():UniformWorkEdit|undefined;
}
/** A frame's receipt is its ring slot's one object, rewritten by the receipt
 * of the next frame to take that slot (UNIFORM_MIXED_RECEIPT_RING frames on
 * at the earliest): a reader that keeps a value past its own reaction copies it. */
export interface UniformMixedFrameReceipt{cycles:number;encoded:number;residual:number;converged:boolean;dustOwners:number;dustMass_cells:number;orphanDustOwners:number;orphanDustMass_cells:number;
 /** The h band this frame re-solved: tiles, completed cycles, final residual (0 when the band is empty). */
 bandTiles:number;bandCycles:number;bandResidual:number}
/** One receipt ring slot: the frame it holds, that frame's words (copied out
 * of the mapped range, under the views its readers take) and the receipt the
 * host is handed. accept and reject are the map's reactions, bound once.
 * target: the frame's residual target, undefined before the first (so the
 * number is kept by reference, like the extension key's). */
interface ReceiptSlot{
 readonly readback:GPUBuffer;unchecked:boolean;frame:number;encoded:number;kick:boolean;target:number|undefined;
 readonly words:Uint32Array<ArrayBuffer>;readonly floats:Float32Array<ArrayBuffer>;
 readonly flip:Uint32Array;readonly status:Uint32Array;readonly work:Uint32Array;
 readonly receipt:UniformMixedFrameReceipt;
 readonly accept:()=>UniformMixedFrameReceipt;readonly reject:(error:unknown)=>never;
}
/** A checked frame's planner output and work counts; frame -1 holds none. */
interface LaggedPlan{frame:number;vCycles:number;fullCycles:number;bandTiles:number;bandBricks:number;bandLevel1:number;bandLevel2:number;bandCycles:number;readonly work:Uint32Array;listed:number;rootListed:number;readonly heat:Uint32Array}
/** The mixed solve's residual target. Tolerance 0 is the fixed-budget
 * control: no user stopping tolerance, so the solve is done at the
 * hierarchy's absolute accuracy (as the native mgCheckCycleConvergence
 * treats it). A literal 0 target can never be met: every frame would fail. */
function uniformMixedPressureTarget(p:{pressureTolerance:number}):number{
 return p.pressureTolerance>0?p.pressureTolerance:UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE;
}

/** The h-tile capacity a frame starts at, and how it follows its layouts.
 * fineTiles: h tiles the owner-indexed buffers hold (UniformMixedCapacity).
 * liquidBand: the pressure band keeps its liquid bound
 * (UniformPressureBand.capacityOf): a GPU relayout whose capacity is every
 * tile the pool can store. Otherwise (host layouts, and a GPU relayout under
 * a reserved capacity) every reserved h tile may be a band tile.
 * byteBudget: bytes the capacity-sized buffers and the detail fields may hold at once, a
 * reservation's old and new together; absent, only the device's buffer and
 * binding limits refuse one. */
export interface UniformMixedFrameCapacity{fineTiles:number;liquidBand:boolean;byteBudget?:number;particleTransfer?:"flip"|"apic";fixedWorldGeometry?:boolean;
 /** COLD regions: "frozen16h" (default) freezes their fine velocity under a 16h coarse term (UniformColdFreeze); "fine" keeps the full fine treatment. */
 coldRegions?:"frozen16h"|"fine";
 /** Host cold-frame skip (UniformColdFrame): "on" (default) leaves out launches of stages with no work in the lagged region heat; "off" encodes every launch. */
 coldFrameSkip?:"on"|"off";
 /** NB-FLIP regionTiering, warmResolution, particleSeeding (default uniformNearSettled()). */
 nearSettled?:UniformNearSettled}

export class UniformMixedFrame {
 /** The NB-FLIP near-settled settings, fixed for the frame's life. */
 readonly nearSettled:UniformNearSettled;
 readonly ownership:UniformMixedOwnership;
 /** The page pool, seeded with the address's stored set and pinned with the
  * residency's pages. The simulation and pressure-root topologies and the
  * owner's page words mirror its table. One commit follows every census
  * (encodePageCommit); its status rides the frame receipt. */
 readonly pagePool:UniformPagePool;
 /** Last completed frame's existing pool receipt; updated without another GPU readback. */
 readonly pagePoolInfo={frame:-1,generation:0,residentPages:0,claimedPages:0,retiredPages:0,freePages:0};
 /** Work counts already returned with each frame. Dependency regions are
  * the union certificate (ownership/stage consumers), not liquid-only pages. */
 readonly workDomainInfo={fineTiles:0,coarseTiles:0,seamFineTiles:0,seamCoarseTiles:0,regularCoarseTiles:0,dependencyRegions16:0,pressureListedOctants:0,remapJobs:0,transferPairs:0};
 readonly levels:readonly UniformMixedPressureCycleLevel[];
 /** Ownership of pressure level 0: the fixed all-4h layout. */
 get pressureOwnership(){return this.levels[0]!.ownership;}
 /** Level 0's live pressure words: all-4h owners plus boundary slots. */
 private get pressureWords(){return uniformMixedPressureStorage(this.pressureOwnership.layout).count;}
 /** The grid overlay's view of the last solve: pressure and pressure phi of
  * pressure level 0, indexed by the pressure owners it was solved on. The
  * phi buffer ends with the frame's stage grids (uniform-stage-grids.ts),
  * which include those owners, so the pair never needs clearing on relayout. */
 readonly presentation:{pressure:GPUBufferBinding;phi:GPUBufferBinding};
 /** Word offsets in presentation.phi of the stage grids' header and of the
  * two-stage h band section in front of it (see uniform-stage-grids). */
 private stageGridWord:number;
 private readonly stageBandWord:number;
 /** Accepted fine pressure and its slot map, including after storage growth. */
 /** Existing band slots, copied only by the opt-in spatial performance sampler. */
 /** Extra producer counters only on explicitly sampled calibration frames. */
 performanceDiagnostics=false;
 get performancePressureCycles():GPUBufferBinding{return {buffer:this.state,offset:28,size:4};}
 get performanceBandCycles():GPUBufferBinding{return {buffer:this.band.index,offset:16,size:4};}
 get performanceIdle():boolean{return this.narrowBandFlip.idle;}
 get performanceSurfaceCounters(){return this.surface.regionCounters;}
 get performanceBandSlots():GPUBufferBinding{return {buffer:this.band.index,offset:this.band.slotMapOffset,size:4*this.ownership.capacity.tiles};}
 get rigidPressure():{buffer:GPUBuffer;word:number}{return {buffer:this.presentation.phi.buffer,word:this.stageBandWord};}
 private stageBandTiles:number;
 private readonly split:{
  transfer:UniformMixedOwnershipTransfer;toPressure:UniformDetailGroup;toSimulation:UniformDetailGroup;
  authority:UniformMixedPressureAuthority;authorityGroup:UniformDetailGroup;
  rhsGroup:UniformDetailGroup;projectionGroup:UniformDetailGroup;
 };
 /** Two-stage pressure: every pressure level is all-4h and
  * UniformPressureArena re-solves the h surface band after it. */
 private readonly band:UniformPressureArena;
 private readonly bandParams:GPUBuffer;
 /** Band tiles of the last accepted advance. */
 bandTiles?:number;
 /** Largest band row residual of the last accepted advance, 1/s. */
 bandResidual?:number;
 readonly solid?:UniformMixedSolid;
 /** The per-frame tile context (solid clearance, neighbour slots, bounds, class bits): built with the solid record and after page commits. */
 readonly tileContext:UniformTileContext;
 /** The AIR image of every field and low-face row of the blocks a commit claims (encodeNewBlocks). */
 private pageInitialise?:UniformPageInitialise;
 /** The pressure root's owners, rebuilt from the page table every step ahead of its first reader. */
 private readonly rootOwners:UniformMixedRootOwners;
 /** The block neighbour rows and wall faces of both topologies' tails, rebuilt from the page table straight after every commit. */
 private readonly blockNeighbours:UniformBlockNeighbours;
 /** Owners of per-key words outside the frame (the census), after the frame's own in the page initialise's order. */
 private readonly pageOwners:UniformPageInitBlocks[]=[];
 /** The census's residency certificate (registerCertificate). */
 private certificate?:{encodeResidency(encoder:GPUCommandEncoder):void};
 /** The owner of the residency certificate: every commit is followed by its
  * mark through the new page table, before anything reads a residency flag. */
 registerCertificate(certificate:{encodeResidency(encoder:GPUCommandEncoder):void}):void{this.certificate=certificate;}
 /** Before initialize: an owner whose words a claimed block's keys must take anew, launched after every commit. */
 registerPageOwner(owner:UniformPageInitBlocks):void{if(this.pageInitialise)throw new Error("The page initialise is built: register its owners before the frame initializes");this.pageOwners.push(owner);}
 /** The remap's extension walls: a low-face buffer of the frame's own. */
 private readonly extensionWalls:GPUBuffer;
 /** A live voxel edit refreshes its geometry at the next frame's head. */
 private solidEditPending=false;
 /** The solid record's simulation widths predate the live layout. */
 private solidWidthsStale=true;
 // Every hanging phi texel is resolved for the live ownership: each advance
 // ends resolved and a host relayout resolves (the head's changed resolve).
 private phiResolved=false;
 /** Band pressure with solids: level 0's topology view (the all-4h record). */
 private readonly solidTopology?:GPUBufferBinding;
 private readonly owned:(GPUTexture|GPUBuffer)[]=[];
 private readonly plan:UniformMixedFramePlan;
 private readonly remap:UniformMixedRemap;
 /** The level set lives at owner resolution: canonical writers (remap, advect,
  * redistance) leave hanging texels stale, and every reader is compiled with
  * the resolved sampler, so each writer is followed by a resolve. */
 private readonly phiResolve:UniformMixedPhiResolve;
 private readonly phiResolveGroups:{phi:UniformDetailGroup;scratch:UniformDetailGroup};
 /** The 4h vertex base consumers outside the solver read: phi at every tile
  * corner, published after the last phi write a presentation can follow. */
 private readonly coarsePhi:UniformCoarseVertexPhi;
 get coarseVertexPhi():GPUTexture{return this.coarsePhi.texture;}
 private readonly extension:UniformMixedExtension;
 private readonly cache:UniformMixedMomentumCache;
 private readonly hanging:UniformMixedHangingTaps;
 private readonly hangingGroup:UniformDetailGroup;
 private readonly surface:UniformMixedSurface;
 readonly surfaceWorkControl:UniformSurfaceWorkControl;
 private readonly surfaceVolume?:UniformMixedSurfaceVolume;
 private readonly geometry:UniformMixedSurfaceGeometry;
 private readonly momentum:UniformMixedMomentum;
 private readonly forces:UniformMixedForces;
 private readonly authority:UniformMixedPressureAuthority;
 private readonly projection:UniformMixedPressureVelocity;
 private readonly cycles:UniformMixedPressureCycles;
 private readonly acceptance:UniformMixedPressureAcceptance;
 private extensionGroups!:readonly [UniformDetailGroup,UniformDetailGroup];
 private flipSnapshotExtensionGroups?:readonly [UniformDetailGroup,UniformDetailGroup];
 private pairedExtensionGroups=new Map<GPUBuffer,{retain:readonly [UniformDetailGroup,UniformDetailGroup];current:readonly [UniformDetailGroup,UniformDetailGroup];old:readonly [UniformDetailGroup,UniformDetailGroup];snapshots:readonly ReturnType<UniformNarrowBandFlip["bindSnapshotSlots"]>[]}>();
 private readonly cacheGroup:UniformDetailGroup;
 private surfaceGroup!:UniformDetailGroup;
 private surfaceVolumeGroup!:UniformDetailGroup;
 private geometryGroup:UniformDetailGroup;
 private readonly momentumGroup:UniformDetailGroup;
 private forceGroup!:UniformDetailGroup;
 /** Per-16h-region force tiers and the projected field's change (stream 1). */
 readonly dynamics:UniformMixedRegionDynamics;
 /** Per-16h-region work tiers (COLD/WARM/HOT), final from the frame head. */
 readonly heat:UniformRegionHeat;
 /** COLD freeze: frozen regions' fine velocity and their 16h term (set at each head from the tiers). */
 readonly freeze?:UniformColdFreeze;
 /** Host cold-frame evidence (uniform-cold-frame): the tiers that may be present this advance. */
 readonly cold:UniformColdFrame;
 /** The work-control generation the last advance saw (a host invalidation moves it). */
 private coldGeneration=-1;
 /** This advance is quiet (UniformColdFrame) with the freeze on: every region is frozen, so the stages whose work is only in non-frozen regions (momentum, its 4h cache and hanging taps, the transfer to simulation) are not encoded; their outputs hold the frozen values. */
 private frozenQuiet=false;
 /** frozenQuiet with the samples idle (UniformNarrowBandFlip.idle; phi held): neither the forces nor any extension is encoded. Every region is frozen, so no publication would write a texel and no snapshot reads the slots; a skipped tail leaves the extension unretained, so the next swept encode marks every region. */
 private extensionIdle=false;
 /** The scratch every non-pressure stage overlaps from offset 0, none of
  * whose contents survives its stage: stageBytesAt the h-tile capacity. The
  * pressure band's rows overlap it too. They live from the band's prepare
  * to its solve, and no stage runs between: forces is the last before the
  * prepare, the closing extension the first after the solve, and each
  * writes what it reads. Rows in a buffer of their own held 5632 bytes an h
  * tile beside this one's 3931 (2.4 and 1.7 GiB on a 384x192x384 lattice). */
 private stageScratch:GPUBuffer;
 private readonly coarseCache:GPUTexture;
 private authorityGroup:UniformDetailGroup;
 private readonly acceptanceGroup:GPUBindGroup;
 private readonly params:Record<"extension"|"surface"|"momentum"|"forces"|"authority"|"projection"|"acceptance",GPUBuffer>;
 private readonly state:GPUBuffer;
 /** The sticky GPU failure record (uniform-mixed-frame-status): the pressure
  * schedule's last gate is the frame's verdict and latches it; a latched
  * record closes every later pressure slot, the band and the transfer back
  * to simulation. Stages that detect their own failures bind it read_write
  * (umLatchFailure). Never cleared: a failed frame fails the sequence. */
 readonly status:GPUBuffer;
 /** Receipt ring: a frame's receipt maps while later frames encode. A slot
  * is unchecked from its frame's submit until that receipt is checked. */
 private readonly receipts:readonly ReceiptSlot[];
 private frameIndex=0;
 /** Each checked frame's planner output, keyed by frame. Frame N encodes
  * frame N-UNIFORM_MIXED_RECEIPT_RING's, never "the newest to resolve":
  * which receipts have mapped depends on wall-clock timing, so a newest-wins
  * plan made free-running and awaited runs diverge. The receipt ring makes
  * frame N reuse that frame's buffer, so its receipt is always checked.
  * A ring by frame: frame N's is taken by frame N+UNIFORM_MIXED_RECEIPT_RING's
  * encode, before that frame's own receipt can write the slot. */
 private readonly lagged:readonly LaggedPlan[]=Array.from({length:UNIFORM_MIXED_RECEIPT_RING},()=>({frame:-1,vCycles:0,fullCycles:0,bandTiles:0,bandBricks:0,bandLevel1:0,bandLevel2:0,bandCycles:0,work:new Uint32Array(UNIFORM_WORK_RECEIPT_WORDS),listed:0,rootListed:0,heat:new Uint32Array(UNIFORM_COLD_FRAME_EVIDENCE_WORDS)}));
 private laggedOf(frame:number):LaggedPlan|undefined{const plan=this.lagged[frame%UNIFORM_MIXED_RECEIPT_RING];return plan&&plan.frame===frame?plan:undefined;}
 private readonly reductions:GPUBuffer;
 private pressureSchedule!:UniformMixedPressureSchedule;
 /** The slot list of the first UNIFORM_MIXED_RECEIPT_RING frames: conservative. */
 private readonly initialPlan:UniformMixedPressurePlan;
 /** Surface-crossing tiles of the current phi: the stage grids' band bits. */
 private readonly surfaceBand:UniformPressureSurfaceBand;
 private ready=false;
 /** The simulation authority's pressure phi, one word per simulation owner
  * the capacity holds. */
 private authorityPhi:GPUBuffer;
 /** See UniformMixedFrameCapacity: a GPU relayout sets it, detaching clears it. */
 private liquidBand:boolean;
 private readonly byteBudget?:number;
 private busy=false;
 private failed=false;
 readonly narrowBandFlip:UniformNarrowBandFlip;
 get narrowBandVolumeBudget():GPUBuffer|undefined{return this.surfaceVolume?.budget;}
 /** The budget's shift and measured volumes are those of the last step that
  * measured them: the first, and the one after each request. */
 private narrowBandVolumeProbe=true;
 requestNarrowBandVolumeProbe():void{this.narrowBandVolumeProbe=true;}
 /** Summed once per set of resources: initialize creates the last of them,
  * and after it only a capacity change (reserveFine) replaces any. */
 private allocatedTotal:number|undefined=undefined;
 get allocatedBytes():number{
  if(this.allocatedTotal!==undefined)return this.allocatedTotal;
  let bytes=this.geometry.allocatedBytes+this.surfaceWorkControl.allocatedBytes+this.cycles.allocatedBytes+(this.narrowBandFlip.allocatedBytes??0)+(this.surfaceVolume?.allocatedBytes??0)+this.coarsePhi.allocatedBytes+this.surfaceBand.allocatedBytes+this.band.allocatedBytes+this.plan.allocatedBytes+this.phiResolve.allocatedBytes+(this.solid?.allocatedBytes??0)+this.surface.allocatedBytes+this.hanging.allocatedBytes+this.ownership.allocatedBytes+this.remap.allocatedBytes+this.split.transfer.allocatedBytes+this.pagePool.allocatedBytes+this.dynamics.allocatedBytes+this.heat.allocatedBytes+(this.freeze?.allocatedBytes??0)+this.extension.allocatedBytes+this.momentum.allocatedBytes+this.split.authority.allocatedBytes+this.tileContext.allocatedBytes;
  for(let i=0;i<this.levels.length;i++){const o=this.levels[i]!.ownership;if(o!==this.ownership)bytes+=o.allocatedBytes;}
  for(let i=0;i<this.owned.length;i++){const r=this.owned[i]!;bytes+="size" in r?r.size:r.width*r.height*r.depthOrArrayLayers*16;}
  if(this.ready)this.allocatedTotal=bytes;
  return bytes;
 }
 private readonly queue:GPUQueue;
 /** One submission's command buffer: the array every submit hands the queue. */
 private readonly commands:GPUCommandBuffer[]=[];
 /** Every ownership an advance holds (the simulation's, then each pressure level's own), and their releases while it runs. */
 private readonly frameOwnerships:UniformMixedOwnership[]=[];
 private readonly releases:(()=>void)[]=[];
 /** write's words, one array per queue write (a write takes the whole array):
  * the cell size and every constant are placed once, here. */
 private readonly words:{readonly frame:Uint32Array<ArrayBuffer>;readonly extension:Float32Array<ArrayBuffer>;readonly surface:Float32Array<ArrayBuffer>;readonly surfaceFlags:Uint32Array<ArrayBuffer>;
  readonly momentum:Float32Array<ArrayBuffer>;readonly momentumFlags:Uint32Array<ArrayBuffer>;readonly forces:Float32Array<ArrayBuffer>;readonly projection:Float32Array<ArrayBuffer>;
  readonly acceptance:Float32Array<ArrayBuffer>;readonly band:Float32Array<ArrayBuffer>};
 /** Byte offset in an ownership's support of its sticky detail violation word. */
 private readonly violationByte:number;
 /** encodeGeometry's options, set in place by each caller that passes any. */
 private readonly geometryOptions:{pressure:boolean;changed:GPUBuffer|undefined}={pressure:false,changed:undefined};
 /** layout: the generation the t=0 fields are in (all h; the first
  * updateLayout remaps from it). capacity: the h tiles the owner-indexed
  * buffers hold until a layout reserves its own, independent of that seed.
  * openTop: the top is open, for the frame's life: both ownerships' page
  * tables hold AIR above the box and pressure is built on it (write). */
 constructor(private readonly device:GPUDevice,layout:UniformMixedLayout,capacity:UniformMixedFrameCapacity,private readonly fields:UniformMixedFrameFields,private readonly openTop:boolean,private readonly schedule:UniformCM11aSchedule=DEFAULT_UNIFORM_CM11A_SCHEDULE){
  const f=fields;
  this.nearSettled=capacity.nearSettled??uniformNearSettled();
  this.liquidBand=capacity.liquidBand;this.byteBudget=capacity.byteBudget;this.queue=device.queue;this.blit=new UniformComputeBlit(device);this.cold=new UniformColdFrame(capacity.coldFrameSkip!=="off");
  const buffer=(label:string,size:number,usage:number)=>{const b=device.createBuffer({label,size,usage});this.owned.push(b);return b;};
  // STORAGE too: a copy into one is a compute launch (UniformComputeBlit).
  const uniform=(name:string,size:number)=>buffer(`Uniform ${name} parameters`,size,GPUBufferUsage.UNIFORM|GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
  this.params={extension:uniform("extension",16),surface:uniform("surface",32),momentum:uniform("momentum",32),forces:uniform("forces",48),authority:uniform("authority",16),projection:uniform("projection",32),acceptance:uniform("acceptance",16)};
  this.state=buffer("Uniform pressure acceptance",32,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST);
  this.status=buffer("Uniform mixed frame status",4*UNIFORM_MIXED_STATUS_WORDS,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST);
  this.initialPlan={vCycles:schedule.vCycles,fullCycles:schedule.fullCycles};
  this.receipts=Array.from({length:UNIFORM_MIXED_RECEIPT_RING},(_,i)=>this.receiptSlot(buffer(`Uniform pressure receipt and mass accounting ${i}`,TRACE_RECEIPT,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ)));
  this.reductions=buffer("Uniform dust accounting",48,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC);
  const caches=Array.from({length:1},(_,i)=>{const t=device.createTexture({label:`Uniform 4h sampling cache ${i}`,size:uniformCoarseCacheExtent(f.arena.address),dimension:"3d",format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING});this.owned.push(t);return t;});
  const coarseLayout=uniformMixedAllCoarseLayout(layout);
  // A static pool takes any reach: every page it could ask for is pinned.
  // Host commands and pins take the pool's fixed capacities: a step that
  // asks for more drop pages, or a scene that pins more, is refused there.
  this.pagePool=new UniformPagePool(device,{...layout.address.pool,closureCap:f.residency.live?4*UNIFORM_RESIDENCY_PAGE_REACH:0xffffffff});this.pagePool.seed(layout.address.seed);this.pagePool.pin(f.residency.pins);
  const solid=this.solid=new UniformMixedSolid(device,f.solid,coarseLayout,this.pagePool);
  this.coarseCache=caches[0]!;
  const o=this.ownership=new UniformMixedOwnership(device,layout,true,capacity.fineTiles,capacity.fixedWorldGeometry);
  this.tileContext=new UniformTileContext(device,o,solid,this.pagePool.state);
  {const h=o.capacity.lattice.cellSize_m,inverse=(x:number)=>Math.fround(1/Math.fround(Math.fround(x)*Math.fround(x)));
   this.words={frame:new Uint32Array(1),extension:new Float32Array([...h,0]),surface:new Float32Array([...h,0]),surfaceFlags:new Uint32Array([0,0,0,4]),
    momentum:new Float32Array([...h,0]),momentumFlags:new Uint32Array([0,0,0,UNIFORM_MIXED_MOMENTUM_LIMITS]),forces:new Float32Array([...h,0,0,0,0,0,0,0,0,0]),projection:new Float32Array([...h,0,0,0,0,0]),
    acceptance:new Float32Array([0,0,UNIFORM_PRESSURE_RELATIVE_REDUCTION,UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE]),band:new Float32Array([...h,0,0,0,0,Math.min(...h),0,inverse(h[0]),inverse(h[1]),inverse(h[2])])};}
  this.violationByte=4*uniformMixedDetailViolationWord(o.capacity.tiles);
  this.cache=new UniformMixedMomentumCache(device,o);
  this.narrowBandFlip=new UniformNarrowBandFlip(device,o,solid,f.sourceParams,this.cache,this.pagePool,capacity.particleTransfer,this.nearSettled.particleSeeding,this.nearSettled.surfaceSplat!=="separate",this.nearSettled.sourceSeeding,this.nearSettled.redistanceSearch,this.nearSettled.particleOrder,this.nearSettled.momentumSampler==="texture32"&&device.features.has("float32-filterable")?"texture32":"software",this.nearSettled.surfaceRedistanceReach);
  this.plan=new UniformMixedFramePlan(device,o,f.volume,f.phi,f.velocityScratch,f.negativeScratch,true,true,solid);
  this.narrowBandFlip.attachUpdateErrors(this.plan.updateErrors);
  this.remap=new UniformMixedRemap(device,o,{volume:f.volume,velocity:f.velocity,phi:f.phi,negative:f.negative},{volume:f.volumeScratch,velocity:f.velocityScratch,phi:f.phiScratch,negative:f.negativeScratch},solid,true);
  // The census extension crosses a relayout in these instead of being
  // rebuilt. Its faces stage in the unit taps: every texel a sampler reads
  // is rewritten by hanging/unitFaces before the first sampler, and nothing
  // reads them from the previous frame's forces until then.
  if(!f.detail)throw new Error("The mixed frame's unit taps are fields of its detail storage");
  // Momentum's unit sampler: texture32 needs float32-filterable, else texture16.
  const sampler=this.nearSettled.momentumSampler==="texture32"&&!device.features.has("float32-filterable")?"texture16":this.nearSettled.momentumSampler;
  this.hanging=new UniformMixedHangingTaps(device,o,f.detail,sampler==="texture16");
  this.remap.bindExtension({volume:f.volume,velocity:f.velocity,phi:f.phi,negative:f.negative},{velocity:this.hanging.unitVelocity,negative:this.extensionWalls=buffer("Uniform mixed remapped extension walls",f.negativeScratch.size,GPUBufferUsage.STORAGE)});
  this.phiResolve=new UniformMixedPhiResolve(device,o,NARROW_BAND_AIR_CHANGED_WORD);this.phiResolveGroups={phi:this.phiResolve.bindHeld(f.phi),scratch:this.phiResolve.bind(f.phiScratch)};
  this.coarsePhi=new UniformCoarseVertexPhi(device,o,f.phi);
  // Domain placement: the published base is the phi field's own base block.
  if(f.detail?.layout.domain)f.detail.adoptBase(f.phi,this.coarsePhi.texture);
  // The arena, low to high: the root, the levels its correction is solved on.
  if(f.arena.rootOffset+f.arena.rootBytes>4*f.arena.levels[0]!.pressure[0])throw new Error("The mixed pressure root overlaps its correction's levels");
  const memory=planUniformMixedPressureMemory(coarseLayout,f.arena.address,f.arena.rootOffset,f.arena.rootBytes);
  const view=(r:{offset:number;size:number}):GPUBufferBinding=>({buffer:f.arena.buffer,...r});
  // Pressure phi, twice: the simulation authority's (simulation owners, at
  // their capacity) for the band's list and rows, and the split authority's
  // (the fixed all-4h owners) for the root solve, the projection and the
  // band's start. Only the first follows the h-tile capacity.
  const phiUsage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST;
  this.authorityPhi=buffer("Uniform mixed pressure phi",4*o.capacity.owners,phiUsage);
  const rootPhi:GPUBufferBinding={buffer:buffer("Uniform mixed root pressure phi",4*(coarseLayout.cellCount+(solid?.coarse?uniformMixedRootMaskWords(coarseLayout):0)),phiUsage)};
  // Pressure's all-4h root keeps the mixed rows; its correction is solved
  // on the levels below it (4h to 32h, rows of blocks as the root is).
  // With solids, the root's topology is the split authority's of each solve:
  // the static all-4h solid record, its cut faces held to the liquid's rows.
  const rootTopology:GPUBufferBinding|undefined=solid?.coarse?{buffer:buffer("Uniform mixed root topology",uniformMixedRootTopologyBytes(coarseLayout),GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC)}:undefined;
  if(rootTopology)this.solidTopology={buffer:rootTopology.buffer,size:rootTopology.buffer.size};
  {const r=memory.root;
   this.levels=[{ownership:new UniformMixedOwnership(device,coarseLayout,false,0,capacity.fixedWorldGeometry),pressure:view(r.pressure),rhs:[view(r.rhs[0]),view(r.rhs[1])],minimum:[view(r.minimum[0]),view(r.minimum[1])],phi:rootPhi,frozen:view(r.frozen),residual:view(r.residual),
    topology:f.solid?{buffer:this.solidTopology!}:undefined}];}
  const root=this.levels[0]!,p=root.ownership;
  this.rootOwners=new UniformMixedRootOwners(device,p,root.phi!,this.solidTopology);
  this.blockNeighbours=new UniformBlockNeighbours(device,layout.address.blocks,[o.pageMirror,p.pageMirror]);
  this.frameOwnerships.push(o);for(const level of this.levels)if(!this.frameOwnerships.includes(level.ownership))this.frameOwnerships.push(level.ownership);
  if(f.detail){f.detail.attach(o);f.detail.attach(p);}
  // The h near-surface extension carries the advancing level-set toe.
  // A regular-only hierarchy previously lost long-dam toe motion.
  this.extension=new UniformMixedExtension(device,o,f.extension,false,true,capacity.coldRegions!=="fine");
  const cacheFields={coarseExtended:caches[0]!};
  this.cacheGroup=this.cache.bind({extended:f.velocityScratch,negative:f.negativeScratch,...cacheFields});
  // Surface and momentum share hanging taps of velocityScratch: nothing
  // between the cache and forces writes it, negativeScratch or cache 0.
  this.hangingGroup=this.hanging.bind({extended:f.velocityScratch,negative:f.negativeScratch,coarse:caches[0]!});
  this.heat=new UniformRegionHeat(device,o,()=>this.surfaceWorkControl.generation);this.plan.attachHeat(this.heat.tiers);this.phiResolve.attachHeat(this.heat.tiers);
  this.surfaceWorkControl=new UniformSurfaceWorkControl(device,o,f.phi,solid,this.pagePool,this.heat);
  this.narrowBandFlip.attachFramePlan(this.surfaceWorkControl.framePlan);
  this.narrowBandFlip.attachSurfaceRevisions(this.surfaceWorkControl.revisions,()=>this.surfaceWorkControl.generation);
  this.narrowBandFlip.attachHeat(this.heat.tiers);this.extension.attachHeat(this.heat.tiers,solid?.coarse?solid.bodyRegions.buffer:undefined);
  this.surface=new UniformMixedSurface(device,o,f.sourceParams,solid,{buffer:this.narrowBandFlip.airGateBuffer,word:this.narrowBandFlip.airGateWord},this.surfaceWorkControl,this.nearSettled.warmResolution==="4h",this.nearSettled.traceSubsteps,this.nearSettled.wallTrace,this.nearSettled.surfaceVelocitySampler==="texture32"&&device.features.has("float32-filterable")?"texture32":"software",this.nearSettled.surfaceTraceSubsteps==="off");
  this.surfaceVolume=new UniformMixedSurfaceVolume(device,o,solid);
  // The full launch after surface volume also writes split pressure's all-4h geometry.
  this.geometry=new UniformMixedSurfaceGeometry(device,o,solid,this.surfaceWorkControl);this.geometryGroup=this.geometry.bind(f.phi,f.target,f.centerPhi,f.pressureGeometry,f.volume);

  // Viscosity reads exact MAC sites from the momentum fill (the extended
  // velocity): h sites in the unit texture, 4h sites in the coarse cache.
  this.dynamics=new UniformMixedRegionDynamics(device,o,{forceParams:this.params.forces,pressure:root.pressure,bodyRegions:solid?.coarse?solid.bodyRegions:undefined,heat:this.heat.tiers,normalDirty:this.geometry.curvatureDirty});
  this.freeze=capacity.coldRegions==="fine"?undefined:new UniformColdFreeze(device,o,this.heat);
  this.narrowBandFlip.attachFreeze(this.freeze);
  this.heat.attach({dynamics:this.dynamics.record,updateErrors:this.plan.updateErrors,bodyRegions:solid?.coarse?solid.bodyRegions:undefined,sourceParams:f.sourceParams,
   particleHeat:{buffer:this.narrowBandFlip.airGateBuffer,word:this.narrowBandFlip.activityHeatWord},coldFreeze:this.freeze,holdRises:capacity.coldFrameSkip!=="off",tiering:this.nearSettled.regionTiering});
  // Retain the projected h velocity detail: restricting all momentum to 4h
  // removes the long-dam toe even when total liquid volume is conserved.
  // Its tiers follow the region heat (UniformMixedMomentumTiers).
  this.momentum=new UniformMixedMomentum(device,o,this.heat.tiers,true,true,f.sourceParams,this.freeze,sampler);
  this.momentumGroup=this.momentum.bind({unitVelocity:this.hanging.unitVelocity,unitHalo:this.hanging.haloOf(this.hanging.unitVelocity),
   unitHaloHalf:sampler==="texture16"?this.hanging.halfHaloOf(this.hanging.unitVelocity):undefined,extended:f.velocityScratch,physical:f.velocity,phase:f.phase,volume:f.volume,centerPhi:f.centerPhi,negative:f.negativeScratch,output:f.departure,outputNegative:f.negativeDeparture,params:this.params.momentum,...cacheFields});
  this.forces=new UniformMixedForces(device,o,true,f.sourceParams,solid,true,true,true,true,true,this.dynamics,this.freeze);
  this.stageBandTiles=this.bandTilesAt(o.capacity.fineTiles,capacity.liquidBand);
  this.stageScratch=buffer("Uniform mixed stage scratch",this.stageBytesAt(o.capacity.fineTiles,this.stageBandTiles),GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST);this.bindStage();
  // Held mode: quiet regions (UniformRegionHeat) keep their owners' values.
  this.authority=new UniformMixedPressureAuthority(device,o,solid,false,false,true,{...this.surfaceWorkControl.revisions,generation:()=>this.surfaceWorkControl.generation});this.authorityGroup=this.bindAuthority();
  // Pressure couples the all-4h owners through the static coarse record.
  this.cycles=new UniformMixedPressureCycles(device,root,view(memory.backup),f.pressure,this.nearSettled.rootReserve!=="parallel",this.nearSettled.rootCoarse!=="tiled",this.nearSettled.rootTiles!=="octant");
  const coarseSolid=!!solid;
  this.projection=new UniformMixedPressureVelocity(device,p,f.sourceParams,solid,coarseSolid,this.cycles.rounding,this.freeze);
  // A cut 4h face's flux sums its h faces of the forced field, which the split
  // leaves in simulation ownership in the scratch pair.
  const fine=coarseSolid?{velocity:f.velocityScratch,negative:f.negativeScratch}:undefined;
  const bindProjection=(input:{velocity:GPUTexture;negative:GPUBuffer},output:{velocity:GPUTexture;negative:GPUBuffer},volume:GPUTexture,fine?:{velocity:GPUTexture;negative:GPUBuffer})=>{
   const common={velocity:input.velocity,negative:{buffer:input.negative},phi:root.phi!,params:this.params.projection,rootTopology};
   return [this.projection.bindRhs({...common,authorityParams:this.params.authority,rhs:root.rhs[0],minimum:root.minimum![0]!,pressure:root.pressure,fine}),
    this.projection.bindProjection({...common,pressure:root.pressure,volume,output:output.velocity,outputNegative:{buffer:output.negative}})] as const;
  };
  const g=f.pressureGeometry;
  {
   // Split: the forced field reaches pressure ownership in velocity/negative
   // and is projected into the scratch pair, then transferred back. Volume
   // reaches pressure ownership in its (free) scratch field.
   const [rhsGroup,projectionGroup]=bindProjection({velocity:f.velocity,negative:f.negative},{velocity:f.velocityScratch,negative:f.negativeScratch},f.volumeScratch,fine);
   const transfer=new UniformMixedOwnershipTransfer(device,o,p,solid,this.freeze);
   const present=(label:string,bytes:number)=>buffer(`Uniform presented ${label}`,bytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST);
   // The record holds any band this frame builds: a larger one is fatal.
   const n=layout.tiles.length;
   // Presented pressure and phi hold the live all-4h words only (level 0
   // keeps the simulation layout's capacity).
   this.stageBandWord=p.layout.cellCount;this.stageGridWord=this.stageBandWord+uniformStageBandWords(n,this.stageBandTiles);
   this.presentation={pressure:{buffer:present("pressure",4*uniformMixedPressureStorage(p.layout).count)},phi:{buffer:present("pressure phi and stage grids",4*(this.stageGridWord+uniformStageGridWords(n)))}};
   // With solids it retains its cut fits and wet topology per 16h region, on the simulation's phi revisions.
   const authority=new UniformMixedPressureAuthority(device,p,solid,coarseSolid,true,true,coarseSolid?{...this.surfaceWorkControl.revisions,generation:()=>this.surfaceWorkControl.generation}:undefined);
   this.split={transfer,rhsGroup,projectionGroup,
    toPressure:transfer.bind({volume:f.volume,velocity:f.velocityScratch,negative:f.negativeScratch},{volume:f.volumeScratch,velocity:f.velocity,negative:f.negative},this.status),
    toSimulation:transfer.bind({volume:f.volume,velocity:f.velocityScratch,negative:f.negativeScratch},{volume:f.volumeScratch,velocity:f.velocity,negative:f.negative},this.status),
    authority,authorityGroup:authority.bind({centerPhi:g.centerPhi,volume:f.volumeScratch,targetFill:g.target,phi:root.phi!,phase:f.phase,correction:f.correction,scratch:coarseSolid?undefined:root.frozen,params:this.params.authority,
     fine:coarseSolid?{phi:f.phi,topology:rootTopology!}:undefined,simulation:{buffer:o.presentation.buffer,size:4*o.capacity.tiles}})};
  }
  this.bandParams=uniform("pressure band",64);
  this.surfaceBand=new UniformPressureSurfaceBand(device,o,f.phi);
  this.band=new UniformPressureArena(device,o,p,{phi:{buffer:this.authorityPhi},coarsePhi:root.phi!,authorityParams:this.params.authority,vertexPhi:f.phi,
   forced:{velocity:f.velocityScratch,negative:f.negativeScratch},velocity:f.velocity,negative:f.negative,copy:f.velocityScratch,
   coarsePressure:root.pressure,params:this.bandParams,presentation:{buffer:this.presentation.phi.buffer,word:this.stageBandWord+layout.tiles.length},capacity:this.stageBandTiles,dynamics:this.dynamics.record,freeze:this.freeze,coldFrame:this.cold,warmTiers:this.nearSettled.warmResolution==="4h"?this.heat.tiers:undefined},{...UNIFORM_PRESSURE_ARENA_SCHEDULE,cycles:this.nearSettled.bandCycles},solid);
  if(this.band.capacity!==this.stageBandTiles)throw new Error("The stage grids' band section must match the band capacity");
  this.acceptance=new UniformMixedPressureAcceptance(device,p);
  this.acceptanceGroup=this.acceptance.bind({residual:root.residual,state:this.state,params:this.params.acceptance});
 }
 /** The stage scratch at `fineTiles` h tiles and `bandTiles` band tiles: the
  * largest of its consumers, each of which checks the buffer against its
  * own need when it binds. */
 private stageBytesAt(fineTiles:number,bandTiles:number):number{
  const n=this.ownership.capacity.tiles;
  // The extension's slots are its prefix: they hold from the tail's
  // extension to the next head's, when no other consumer runs (useSlots).
  return Math.ceil(Math.max(this.extension.slotBufferBytesAt(fineTiles),this.surface.scratchBytes,
   this.surfaceVolume?.scratchBytesAt(fineTiles)??0,UniformMixedForces.normalBytes(63*fineTiles+n))/256)*256;
 }
 /** Every group over the stage scratch (reserveFine binds them again). */
 private bindStage():void{
  // Share taps only when the paired path fits at every possible reservation.
  // Otherwise a later fallback snapshots before viscosity's last tap read.
  const shareUnit=this.extension.supportsPair&&this.narrowBandFlip.particles.every(b=>b.size>=this.extension.companionBytesAt(this.ownership.capacity.address.poolTiles));
  this.narrowBandFlip.bind({...this.fields,coarseExtended:this.coarseCache,unitExtended:this.hanging.unitVelocity,hanging:this.hanging,sharedStageUnit:shareUnit?this.hanging.unitVelocity:undefined});
  const f=this.fields,stage:GPUBufferBinding={buffer:this.stageScratch};
  this.extension.useSlots(this.stageScratch);
  this.extensionGroups=this.extension.bind({physical:f.velocity,phase:f.phase,negative:f.negative,output:f.velocityScratch,outputNegative:f.negativeScratch,params:this.params.extension,gate:this.narrowBandFlip.liveGate});
  this.flipSnapshotExtensionGroups=this.extension.bind({physical:f.velocity,phase:f.phase,negative:f.negativeDeparture,output:f.velocityScratch,outputNegative:f.negativeScratch,params:this.params.extension});
  // The expired particle source bank holds the second scalar payload. A
  // capacity too small for both epochs retains the original two extensions.
  this.pairedExtensionGroups.clear();
  for(const scratch of this.narrowBandFlip.particles){
   if(!this.extension.supportsPair||scratch.size<this.extension.companionBytesAt(this.ownership.capacity.fineTiles))continue;
   const companion={scratch,physical:f.departure,negative:f.negativeDeparture};
   const common={phase:f.phase,output:f.velocityScratch,outputNegative:f.negativeScratch,params:this.params.extension,companion,gate:this.narrowBandFlip.liveGate};
   this.pairedExtensionGroups.set(scratch,{
    retain:this.extension.bind({...common,physical:f.velocity,negative:f.negativeDeparture}),
    current:this.extension.bind({...common,physical:f.velocity,negative:f.negative}),
    old:this.extension.bind({...common,physical:f.departure,negative:f.negativeDeparture,snapshot:this.narrowBandFlip.snapshotUnitVelocity}),
    snapshots:this.extension.companionBindings(scratch).map(slots=>this.narrowBandFlip.bindSnapshotSlots(slots)),
   });
  }
  this.surfaceGroup=this.surface.bind({narrowBandState:this.narrowBandFlip.state,unitVelocity:this.hanging.unitVelocity,unitHalo:this.hanging.haloOf(this.hanging.unitVelocity),velocity:f.velocityScratch,coarseVelocity:this.coarseCache,negative:f.negativeScratch,params:this.params.surface,evidence:stage,phi:f.phi,outputPhi:f.phiScratch});
  if(this.surfaceVolume)this.surfaceVolumeGroup=this.surfaceVolume.bind(f.phi,f.target,stage);
  // NB-FLIP's transfer leaves the advected field in the stored velocity (a
  // kick forces that field where it stands), under momentum's walls.
  this.forceGroup=this.forces.bind({unitVelocity:this.hanging.unitVelocity,advected:f.velocity,phi:f.phi,volume:f.volume,centerPhi:f.centerPhi,coarseVelocity:this.coarseCache,negative:f.negativeDeparture,output:f.velocityScratch,outputNegative:f.negativeScratch,params:this.params.forces,normals:stage,retainedNormals:this.geometry.normalBinding,normalDirty:this.geometry.curvatureDirty});
 }
 private bindAuthority():UniformDetailGroup{
  const f=this.fields;
  return this.authority.bind({centerPhi:f.centerPhi,volume:f.volume,targetFill:f.target,phi:{buffer:this.authorityPhi},phase:f.phase,correction:f.correction,scratch:this.levels[0]!.frozen,params:this.params.authority});
 }
 private bandTilesAt(fineTiles:number,liquidBand:boolean):number{
  // Particle spray can put a wet pressure row in every admitted fine tile.
  // The geometric method's half-domain liquid bound does not bound that set.
  return Math.max(1,fineTiles);
 }
 /** Every buffer the h-tile capacity sizes, at `fineTiles` h tiles and
  * `bandTiles` band tiles: label and bytes. Each is one storage binding. */
 private capacityBuffers(fineTiles:number,bandTiles:number):[string,number][]{
  const n=this.ownership.capacity.tiles,owners=63*fineTiles+this.ownership.capacity.address.boxTiles;
  return [["stage scratch",this.stageBytesAt(fineTiles,bandTiles)],["retained normals and curvature",this.geometry.normalBytesAt(fineTiles)],["velocity tap cache",uniformMixedHangingBytes(n,fineTiles)],["pressure phi",4*owners],
   ["presented pressure phi",4*(this.stageBandWord+uniformStageBandWords(n,bandTiles)+uniformStageGridWords(n))]];
 }
 /** The capacity reserveFine(need) would adopt, or why the device cannot
  * hold it. Changes nothing: admission is decided here, before a layout is
  * adopted. Growth is geometric (UNIFORM_DETAIL_POLICY.poolGrowth), exact
  * where that does not fit; a need under 1/growth^2 of the capacity returns
  * to it. exact: `need` itself, the caller's own rule (a relayout's
  * capacity follows its receipts). */
 fineReservation(need:number,liquidBand=this.liquidBand,exact=false):{fineTiles:number;bandTiles:number;bytes:number;refusal?:string}{
  const c=this.ownership.capacity,n=c.address.poolTiles,held=c.fineTiles,limits=this.device.limits,g=UNIFORM_DETAIL_POLICY.poolGrowth;
  if(!Number.isInteger(need)||need<0||need>n)throw new Error(`A mixed frame whose page pool stores at most ${n} tiles cannot reserve ${need} h tiles`);
  const total=(list:[string,number][])=>list.reduce((sum,[,bytes])=>sum+bytes,0);
  // The detail fields' textures count beside the buffers; a reservation that replaces them
  // (a domain placement crossing zero) holds both generations across the transfer.
  const detail=this.fields.detail,fieldsHeld=detail?.fieldBytesAt(held)??0;
  const fields=(fineTiles:number)=>{const next=detail?.fieldBytesAt(fineTiles)??0;return next===fieldsHeld?0:next;};
  const live=total(this.capacityBuffers(held,this.stageBandTiles))+fieldsHeld,most=Math.min(limits.maxBufferSize,limits.maxStorageBufferBindingSize);
  let refusal:string|undefined;
  for(const fineTiles of exact?[need]:need>held?[Math.min(n,Math.max(need,g*held)),need]:need*g*g<held?[need]:[held]){
   const bandTiles=this.bandTilesAt(fineTiles,liquidBand),buffers=this.capacityBuffers(fineTiles,bandTiles),bytes=total(buffers)+fields(fineTiles);
   const changed=fineTiles!==held||bandTiles!==this.stageBandTiles,over=changed?buffers.find(([,size])=>size>most):undefined;
   refusal=over?`${need} h tiles need a ${over[0]} of ${over[1]} bytes; the device holds ${most} in one buffer`
    :changed&&this.byteBudget!==undefined&&live+bytes>this.byteBudget?`${need} h tiles need ${bytes} bytes of capacity-sized buffers and fields beside the ${live} in use; the budget is ${this.byteBudget}`:undefined;
   if(!refusal)return {fineTiles,bandTiles,bytes};
  }
  return {fineTiles:held,bandTiles:this.stageBandTiles,bytes:live,refusal};
 }
 /** Between frames: hold `need` h tiles. Every buffer the capacity sizes is
  * re-created at it and its groups bound again; none carries a frame's
  * state into the next, so nothing is copied but the hanging slot tables
  * (the ownership's) and the presented phi's 4h words and stage grids (the
  * overlay reads them until the next frame presents). The buffers the
  * submitted frames ran on are destroyed after them. No shader compiles.
  * Returns the device's refusal (fineReservation), with nothing changed.
  * exact under an attached relayout: fewer tiles than held is the caller's
  * claim that a build receipt confirms the running generation fits them. */
 reserveFine(need:number,liquidBand=this.liquidBand,exact=false):string|undefined{
  if(this.busy)throw new Error("The mixed capacity cannot change during an advance");
  const r=this.fineReservation(need,liquidBand,exact);if(r.refusal)return r.refusal;
  this.liquidBand=liquidBand;
  const o=this.ownership,n=o.capacity.tiles,owners=o.capacity.fineTiles!==r.fineTiles,band=this.stageBandTiles!==r.bandTiles;
  if(!owners&&!band)return undefined;
  this.allocatedTotal=undefined;this.cold.trigger();
  const retired:GPUBuffer[]=[];
  const replace=(old:GPUBuffer,size:number)=>{const b=this.device.createBuffer({label:old.label,size,usage:old.usage});this.owned[this.owned.indexOf(old)]=b;retired.push(old);return b;};
  if(owners){
   o.reserveFine(r.fineTiles,exact&&!!this.relayout);
   if(this.geometry.reserveNormals(r.fineTiles)){
    const f=this.fields;this.geometryGroup=this.geometry.bind(f.phi,f.target,f.centerPhi,f.pressureGeometry,f.volume);
    this.geometryCurrent=false;
   }
   this.authorityPhi=replace(this.authorityPhi,4*o.capacity.owners);
   this.authorityGroup=this.bindAuthority();
  }
  // The band's rows are in the stage scratch: either capacity sizes it.
  const stageBytes=this.stageBytesAt(r.fineTiles,r.bandTiles);
  if(owners||stageBytes!==this.stageScratch.size){this.stageScratch=replace(this.stageScratch,stageBytes);this.bindStage();}
  if(band){
   // The record ends the buffer (readers find it from the length), so the band section in front of it moves it.
   const old=this.presentation.phi.buffer,from=this.stageGridWord,grids=uniformStageGridWords(n);
   this.stageBandTiles=r.bandTiles;this.stageGridWord=this.stageBandWord+uniformStageBandWords(n,r.bandTiles);
   const next=replace(old,4*(this.stageGridWord+grids)),e=this.device.createCommandEncoder({label:"Uniform presented phi capacity"});
   // The band section starts empty (no tile has a slot) until the next frame presents its own.
   e.copyBufferToBuffer(old,0,next,0,4*this.stageBandWord);e.copyBufferToBuffer(old,4*from,next,4*this.stageGridWord,4*grids);
   this.device.queue.submit([e.finish()]);this.device.queue.writeBuffer(next,4*(this.stageGridWord+4),Uint32Array.of(r.bandTiles));
   this.presentation.phi={buffer:next};
  }
  this.band.resize({phi:{buffer:this.authorityPhi},presentation:{buffer:this.presentation.phi.buffer,word:this.stageBandWord+n},capacity:r.bandTiles});
  for(const b of retired)b.destroy();
  return undefined;
 }
 /** Every stage compiles at once: an uncached pipeline costs the driver a
  * full Metal compile, and awaiting stages in turn serialized ~350 of them
  * behind one another on a scene's first load. */
 async initialize():Promise<void>{
  const root=this.pressureOwnership;
  this.pressureSchedule=new UniformMixedPressureSchedule(this.device,this.schedule,this.state,
   {fine:root.support,supportWord:9*root.layout.tiles.length+24,status:this.status,band:this.band.index,bandClosedWord:UniformPressureArena.closedWord,acceptance:this.params.acceptance});
  // Every field of the detail storage exists by now: a later one is fatal at the first page initialise.
  const tileWords=new UniformPageTileWords(this.device,this.pagePool,[uniformPageTopology(this.ownership)]);
  const pageInitialise=this.pageInitialise=new UniformPageInitialise(this.device,this.pagePool,this.fields.detail!,{near:16*Math.max(...this.ownership.capacity.lattice.cellSize_m),
   lowFaces:[{buffer:this.fields.negative},{buffer:this.fields.negativeScratch},{buffer:this.fields.negativeDeparture},{buffer:this.extensionWalls}],owners:[tileWords,this.narrowBandFlip,this.surfaceWorkControl,...this.pageOwners]});
  await settleGPUInitialization([this.blit,this.solid,this.tileContext,this.plan,this.remap,this.phiResolve,this.coarsePhi,this.extension,this.cache,this.hanging,this.surfaceWorkControl,this.surface,this.surfaceVolume,this.geometry,this.momentum,this.forces,this.dynamics,this.heat,this.authority,this.projection,this.cycles,this.acceptance,this.split.transfer,this.split.authority,this.pressureSchedule,this.surfaceBand,this.band,this.narrowBandFlip]
   .map(stage=>stage?.initialize()).concat(this.freeze?.initialize(this.status),this.fields.detail?.prepare(this.ownership),this.pagePool.initialize([this.ownership.pageMirror,root.pageMirror,this.fields.pages],this.fields.sourceParams),tileWords.initialize(),pageInitialise.initialize(),this.rootOwners.initialize(),this.blockNeighbours.initialize()));
  this.ready=true;
 }
 /** The last advance's extension key (write), once there has been one, and
  * the support policy it holds. */
 private readonly key=extensionKey();
 private readonly keyPolicy={fineReach:0,shellReach:0,twoLevel:false,shellOnly:false};
 private keyed=false;
 /** The last advance's residual target (write). */
 private pressureTarget:number|undefined=undefined;
 /** A census extension (encodeExtension) under the last advance's key whose
  * plan, phase and extension the next advance may reuse: state is untouched
  * since, and the advance runs with identical parameters (write's answer). */
 private reusableExtension=false;
 /** A body frame's head found the last extension reusable but for the
  * bodies: its extension rebuilds only the regions they reached. */
 private bodyExtension=false;
 /** Any state, parameter or ownership change between frames. */
 invalidateExtension():void{this.cold.trigger();this.reusableExtension=false;this.geometryCurrent=false;this.split.authority.invalidate();}
 /** A live voxel edit rewrote the host's solid mask: rebuild the all-4h
  * record and the cut widths, and displace liquid out of newly solid h
  * cells at the next frame's head. The host has already promoted every
  * tile the edit touches to h (its CPU relayout). */
 editSolids():void{
  this.surfaceWorkControl.invalidate();this.dynamics.invalidate();
  if(!this.solid)throw new Error("This mixed frame was built without solids");
  this.solid.invalidate();this.solidWidthsStale=true;this.invalidateExtension();this.solidEditPending=true;
 }
 /** The simulation target and centre phi were built from the current phi by
  * the last completed advance's gather: nothing has written phi, solids or
  * ownership since. Every such edit clears it with the extension. */
 private geometryCurrent=false;
 /** Bodies moved (the record rebuilt their swept tiles) since the last
  * geometry encode; geometryBodiesOnly: and that geometry was current but
  * for them, so the next plain encode is the body launch (encode {bodies}). */
 private geometryBodies=false;
 private geometryBodiesOnly=false;
 /** The last advance's simulation authority wrote phase on the live layout
  * (under its key); false when rigid bodies moved after it. While geometryCurrent holds, nothing has written volume,
  * phi, solids or ownership since either (every such edit goes through
  * invalidateExtension), and the split's all-4h authority writes no phase:
  * a head's phase launch would store the same values again. */
 private phaseKeyed=false;
 /** same: this advance's key is the last one's (write). */
 private phaseCurrent(same:boolean):boolean{return this.geometryCurrent&&this.phaseKeyed&&same;}
 /** Only rigid bodies moved since the last advance's simulation authority
  * (its geometry was current when this head took them): the authority need
  * rewrite only the owners near their swept support (encodeBodies). Read
  * before this head's geometry encode takes the bodies. */
 private phaseBodiesOnly(same:boolean):boolean{return this.geometryBodies&&this.geometryBodiesOnly&&same&&this.solid!==undefined;}
 /** Re-run the last advance's plan and extension into velocityScratch on
  * the live layout: the field the next advance's surface trace samples.
  * The dynamic census bounds departures from it between frames. */
 encodeExtension(encoder:GPUCommandEncoder):void{
  if(!this.ready||this.busy||this.failed||!this.keyed)throw new Error("Mixed extension needs a completed advance");
  this.encodeExtensionOf(encoder,this.key.supportPolicy,this.key.extensionSweeps);
 }
 /** phase=false: this advance's simulation authority already wrote phase
  * on this layout and nothing has written volume, centre phi, the target or
  * solids since (the split's all-4h authority writes no phase). */
 private encodeExtensionOf(encoder:GPUCommandEncoder,policy:UniformMixedFrameParameters["supportPolicy"],sweeps:number|undefined,phase=true):void{
  this.plan.encode(encoder,policy);
  // Phase only: the advance's authority rewrites phi before its readers
  // (band rows and RHS form the volume correction themselves).
  if(phase)this.authority.encode(encoder,this.authorityGroup,false);
  this.extension.encode(encoder,this.extensionGroups,sweeps??2);
  this.reusableExtension=true;
 }
 /** p's words into every stage's parameters, and p's key into the frame's.
  * True: that key is the last write's. dt and gravity are the caller's
  * reads of p: each number leaves p once, since a double an object holds is
  * boxed again by every load until its reader is optimized. */
 private write(p:UniformMixedFrameParameters,dt:number,gravity:number):boolean{
  if(p.openTop!==this.openTop)throw new Error(`Uniform mixed frame: built with ${this.openTop?"an open":"a closed"} top (page tables, pressure); an advance cannot change it`);
  const density=p.density,viscosity=p.viscosity,tension=p.surfaceTension,tolerance=p.pressureTolerance,open=+p.openTop,step=dt/density,target=tolerance>0?tolerance:UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE;
  const w=this.words,q=this.queue,to=this.params,forces=w.forces,projection=w.projection,band=w.band;
  w.extension[3]=open;q.writeBuffer(to.extension,0,w.extension);
  w.surface[3]=dt;q.writeBuffer(to.surface,0,w.surface);w.surfaceFlags[0]=open;w.surfaceFlags[1]=p.cubic?(p.fineTrilinear?2:1):0;w.surfaceFlags[2]=+this.performanceDiagnostics;q.writeBuffer(to.surface,16,w.surfaceFlags);
  w.momentum[3]=dt;q.writeBuffer(to.momentum,0,w.momentum);w.momentumFlags[0]=open;q.writeBuffer(to.momentum,16,w.momentumFlags);
  forces[3]=dt;forces[4]=gravity;forces[5]=density;forces[6]=viscosity;forces[7]=tension;forces[8]=+p.noSlip;forces[9]=open;q.writeBuffer(to.forces,0,forces);
  q.writeBuffer(to.authority,0,AUTHORITY_WORDS);
  projection[3]=dt;projection[4]=density;projection[5]=open;q.writeBuffer(to.projection,0,projection);
  w.acceptance[0]=step;w.acceptance[1]=target;q.writeBuffer(to.acceptance,0,w.acceptance);
  band[3]=dt;band[4]=density;band[5]=open;band[6]=step;band[8]=Math.min(target,this.nearSettled.bandExit==="returns"?UNIFORM_PRESSURE_ARENA_RETURNS_RESIDUAL:UNIFORM_PRESSURE_BAND_RESIDUAL_TOLERANCE);q.writeBuffer(this.bandParams,0,band);
  const views=this.layoutViews?(this.relayout?UNIFORM_STAGE_VIEWS.previous|UNIFORM_STAGE_VIEWS.reasons|UNIFORM_STAGE_VIEWS.importance:0)|UNIFORM_STAGE_VIEWS.certificate|UNIFORM_STAGE_VIEWS.band:0;
  q.writeBuffer(this.presentation.phi.buffer,4*this.stageGridWord,uniformStageGridHeader(this.ownership.capacity.tiles,this.stageBandTiles,views));
  this.pressureTarget=target;
  const k=this.key,held=k.supportPolicy,policy=p.supportPolicy,reserve=p.pressureReserve,seconds=p.volumeControlSeconds,sweeps=p.extensionSweeps;
  const same=this.keyed&&agree(k.gravity,gravity)&&agree(k.density,density)&&agree(k.viscosity,viscosity)&&agree(k.surfaceTension,tension)&&k.openTop===p.openTop&&k.noSlip===p.noSlip&&k.cubic===p.cubic&&k.fineTrilinear===p.fineTrilinear
   &&k.fullPressureEnvelope===p.fullPressureEnvelope&&agree(k.pressureReserve,reserve)&&agree(k.volumeControlSeconds,seconds)
   &&k.redistance===p.redistance&&agree(k.extensionSweeps,sweeps)&&agree(k.pressureTolerance,tolerance)
   &&(policy?!!held&&agree(held.fineReach,policy.fineReach)&&agree(held.shellReach,policy.shellReach)&&held.twoLevel===policy.twoLevel&&held.shellOnly===policy.shellOnly:!held);
  k.gravity=gravity;k.density=density;k.viscosity=viscosity;k.surfaceTension=tension;k.openTop=p.openTop;k.noSlip=p.noSlip;k.cubic=p.cubic;k.fineTrilinear=p.fineTrilinear;
  k.fullPressureEnvelope=p.fullPressureEnvelope;k.pressureReserve=reserve;k.volumeControlSeconds=seconds;
  k.redistance=p.redistance;k.extensionSweeps=sweeps;k.pressureTolerance=tolerance;
  if(policy){const own=this.keyPolicy;own.fineReach=policy.fineReach;own.shellReach=policy.shellReach;own.twoLevel=policy.twoLevel;own.shellOnly=policy.shellOnly;k.supportPolicy=own;}else k.supportPolicy=undefined;
  this.keyed=true;
  return same;
 }
 /** Record the layout views (the stage grids' previous, reasons and
  * certificate sections) for the tiles, grid and pressure layers. Off, the
  * frame copies nothing and the census records no reasons. */
 setLayoutViews(enabled:boolean):void{this.layoutViews=enabled;}
 private layoutViews=false;
 /** Copies one view section (a word per tile; importance has two) into the stage grids. */
 private recordStageView(encoder:GPUCommandEncoder,buffer:GPUBuffer,offset:number,section:"previous"|"reasons"|"certificate"|"importance"):void{
  const n=this.ownership.capacity.tiles;
  encoder.copyBufferToBuffer(buffer,offset,this.presentation.phi.buffer,4*(this.stageGridWord+uniformStageViewWord(n,section)),4*n*(section==="importance"?2:1));
 }
 /** Copies one stage's tile words (or the band bits) into the stage grids. */
 private recordStageGrid(encoder:GPUCommandEncoder,source:GPUBuffer,range:"transport"|"pressure"|"band"):void{
  const n=this.ownership.capacity.tiles,words=range==="band"?(n+31)>>>5:n;
  const at=this.stageGridWord+UNIFORM_STAGE_GRID_HEADER_WORDS+(range==="transport"?0:range==="pressure"?n:2*n);
  encoder.copyBufferToBuffer(source,0,this.presentation.phi.buffer,4*at,4*words);
 }
 /** Discarded mass uses native sixty-fourths-of-threshold counters, weighted
  * by owner volume. It is a quantized lower bound; counts name owners, not
  * fine cells. Counters are reset once per frame. */
 /** The projected kick U <- P(U + lead g): a frame's head, then forces and
  * the whole pressure solve over `lead` seconds, and no drift. A frame
  * drifts by dt U and then adds g dt, so the stored velocity is the drift's,
  * half a step ahead of the positions; a state seeded at t=0 lagged free
  * fall by g t dt/2 until the host kicked it by dt/2 before its first frame.
  * Gravity alone (no viscosity, capillarity or volume correction) and no
  * transport. The head is the frame's: where a relayout is due its census
  * runs for the frame's step and the kick is solved on the generation it
  * adopts, like every solve (the seed's all-h tiles can outnumber the
  * band's slots); phi and V are written by that remap alone. Converged on
  * the full envelope to UNIFORM_PRESSURE_KICK_RESIDUAL_TOLERANCE, or to the
  * frame's own tolerance where a scene asks for a tighter one; its pressure
  * (Pa: hydrostatic) is presented, the first frame's warm start. The receipt
  * is a frame's, in the same ring and under the same verdict; it plans no
  * frame, so the first two still encode the envelope. bodies: the roster
  * the next frame advances, as solids where they stand. */
 kick(p:UniformMixedFrameParameters,lead:number,bodies=false):Promise<UniformMixedFrameReceipt>{
  if(!(lead>0))throw new Error("The projected kick leads by a positive time");
  return this.advance({...p,dt:lead,viscosity:0,surfaceTension:0,fullPressureEnvelope:true,pressureTolerance:Math.min(uniformMixedPressureTarget(p),UNIFORM_PRESSURE_KICK_RESIDUAL_TOLERANCE)},undefined,false,bodies?{}:undefined,p.dt);
 }
 /** extendTail: extend this frame's final velocity for the next frame
  * head's census (and reuse it there when the key still matches).
  * Encodes and submits synchronously; the promise is the frame's receipt,
  * checked when its map resolves while later frames encode. A failed receipt
  * rejects naming its frame and fails the frame for every later advance.
  * kick: kick()'s encode, ahead of a frame of this step; it takes no frame index. */
 advance(p:UniformMixedFrameParameters,trace?:UniformMixedFrameTrace,extendTail=false,bodies?:UniformMixedFrameBodies,kick=0):Promise<UniformMixedFrameReceipt>{
  if(trace){const measured=trace;trace={...measured,phase:(encoder,phase)=>{this.blit.finishStage(encoder);measured.phase(encoder,phase);}};}
  if(bodies&&!this.solid)throw new Error("This mixed frame was built without solids; rigid bodies need the solid library");
  if(!this.ready||this.busy||this.failed)throw new Error("Unified frame is not ready for an advance");
  let slot:ReceiptSlot|undefined;
  for(let i=0;i<this.receipts.length&&!slot;i++)if(!this.receipts[i]!.unchecked)slot=this.receipts[i];
  if(!slot)throw new Error(`Uniform mixed frame has ${this.receipts.length} unchecked receipts; the host must check one before encoding frame ${this.frameIndex+1}`);
  const readback=slot.readback;
  this.busy=true;const frame=kick?this.frameIndex:++this.frameIndex;
  // The frame a failure latched in this advance is recorded as.
  this.words.frame[0]=frame;this.queue.writeBuffer(this.status,4*UNIFORM_MIXED_STATUS.currentFrame,this.words.frame);
  let held=0;
  try{
   // Use the same fixed-lag receipt as pressure scheduling, independent of
   // map completion timing. This changes direct launch width only: band
   // kernels stride the current GPU count even when it outruns the budget.
   // A receipt from before a relayout attach counts another owner's layout.
   // A relayout that is not due keeps its generation: this frame's head is the host layout's.
   let relayout=this.relayout?.due?.()===false?undefined:this.relayout;
   // A census after frames without one may admit any tile, as at the attach: the ceilings until its receipt arrives.
   // One that answers a counted request edit moves that many tiles at most: the budgets stay, and
   // carry its count until the receipt of the frame that built it is the one observed.
   const edit=relayout?.edit?.(),firstCensus=relayout!==undefined&&!this.censused;
   if(edit)this.noteEdit(frame,edit);
   else if(relayout&&!this.censused){this.workEpoch=frame-1;this.band.observeWork(this.band.capacity);this.band.forgetCycles();this.ownership.forgetWork();this.remap.forgetWork();}
   const edited=this.editedBy(frame);
   const work=!kick&&frame-UNIFORM_MIXED_RECEIPT_RING>this.workEpoch?this.laggedOf(frame-UNIFORM_MIXED_RECEIPT_RING):undefined;
   // Under a census the budgets keep a reserve: it admits tiles the receipts report two frames late.
   // A counted edit's build admits none but its own.
   const reserve=relayout&&!edit?UNIFORM_WORK_RELAYOUT_RESERVE:0;
   // The band's tiles are wet h tiles: an edit adds its own at most.
   if(work){this.band.observeWork(work.bandTiles+(edited?.tiles??0),reserve,work.bandBricks+(edited?.tiles??0),work.bandLevel1+(edited?.tiles??0),work.bandLevel2+(edited?.tiles??0));this.band.observeCycles(work.bandCycles);this.ownership.observeWork(work.work,reserve,edited);this.remap.observeWork(work.listed,reserve,edited?.reach);this.cycles.observeWork(work.rootListed);}
   // Only a census certifies residency: a frame without one must not skip a page the last one did.
   if(!kick||relayout){if(!relayout&&this.censused)this.ownership.resetResidency();this.censused=!!relayout;}
   // Only a census asks the pool for the liquid's pages: without one a live pool would advance into pages it never claimed.
   if(!relayout&&this.fields.residency.live)throw new Error("Uniform mixed frame: an advance without a census on a live page pool");
   for(let i=0;i<this.frameOwnerships.length;i++){this.releases[held]=this.frameOwnerships[i]!.acquireFrame();held++;}
   // same: p's extension key is the last advance's.
   const dt=p.dt,gravity=p.gravity,same=this.write(p,dt,gravity);this.blit.setSharedSchedule(false);let encoder=this.beginEncoder(trace);
   {const generation=this.surfaceWorkControl.generation,moved=generation!==this.coldGeneration;this.coldGeneration=generation;
    this.cold.begin(frame,frame-UNIFORM_MIXED_RECEIPT_RING,work?.heat,kick>0||!same||moved||edit!==undefined||firstCensus||(p.addedVolumeCells??0)!==0);
    this.frozenQuiet=this.freeze!==undefined&&this.cold.quiet;
    // NB-FLIP's idle advance (no live sample, every region frozen and COLD) leaves its stages out.
    this.surfaceWorkControl.framePlan.beginFrame();this.narrowBandFlip.beginAdvance(this.cold);this.extensionIdle=this.frozenQuiet&&this.narrowBandFlip.idle;}
   // An idle advance (NB-FLIP idle: quiet with the freeze this advance and
   // the last, no live sample) runs no census. Everything it reads is held:
   // phi, every region frozen, no sample, and no host trigger (each host
   // census input breaks idle on its own frame). The last generation, its
   // residency certificate and page claims stand (censused stays set, so
   // nothing above reset them); the first advance off idle censuses as before.
   if(relayout&&this.narrowBandFlip.idle)relayout=undefined;
   // The projection alone: at dt 0 the authority's volume correction and deficit balance are zero.
   if(kick)this.queue.writeBuffer(this.params.authority,0,AUTHORITY_WORDS);
   // Bodies moved at the last frame's tail: the record rebuilds the tiles
   // their swept support reaches (its build writes their widths too), and
   // derived geometry refreshes only the regions the swept-body mask names.
   if(bodies){this.solid!.encodeBodies(encoder);this.solid!.invalidate(true);this.geometryBodies=true;this.geometryBodiesOnly=this.geometryCurrent;this.split.authority.noteBodies();this.geometryCurrent=false;this.bodyExtension=this.reusableExtension;this.reusableExtension=false;}
   // Built once for static solids (again after an edit, the moved bodies'
   // tiles every frame bodies exist), with the tile cut map every solid helper
   // gates on, before the band's tile list reads them. Every head also writes
   // the swept-body region mask (UniformMixedSolid.bodyRegions) here.
   const solidBuild=!!this.solid?.recordStale;this.solid?.encodeCoarse(encoder,this.ownership.presentation);if(solidBuild)this.tileContext.encodeBuild(encoder);
   // Where this frame's non-pressure work goes: after the swept-body mask, before every stage that reads the tiers.
   this.heat.encode(encoder,dt,p.density);
   // The freeze follows the final tiers: a relayout head's after its promotions (encodeRelayoutHead).
   if(!relayout)this.freeze?.encodeHead(encoder,dt,gravity,this.frozenQuiet);
   // A host layout: a live solid edit, or the cells bodies entered, lands on
   // the live (host-promoted) ownership. A GPU relayout displaces after its
   // adopt, on the generation that promoted them (encodeRelayoutHead).
   if(!relayout&&(this.solidEditPending||bodies)){this.solidEditPending=false;}
   // An attached relayout that built nothing this frame changed nothing: the view of the generation before is this one.
   if(!relayout&&this.relayout&&this.layoutViews)this.recordStageView(encoder,this.ownership.presentation.buffer,this.ownership.presentation.offset??0,"previous");
   // The frame is submitted in segments as it encodes (flush): the GPU starts each one
   // while the host encodes the next, instead of idling through the whole
   // frame's encode. Queue writes made while a later segment encodes land
   // after the earlier segments, which never read them.
   encoder.clearBuffer(this.reductions);
   // A kick's census is its frame's, for that frame's step.
   if(relayout)this.encodeRelayoutHead(encoder,p,kick||dt,same,relayout,bodies!==undefined,trace);
   else{
   // Which cut tiles the simulation holds at h: the all-4h levels read their h texels.
   if(this.solidWidthsStale){this.solid?.encodeSimulation(encoder,this.ownership.presentation);this.solidWidthsStale=false;}
   if(this.solid)trace?.phase(encoder,V.solids);
   this.recordStageGrid(encoder,this.ownership.presentation.buffer,"transport");
   // The census already planned and extended this exact state: geometry and
   // centre phi are the last frame's (phi is unchanged since), phase is the
   // live layout's, and velocityScratch holds the extension.
   // The h phi field survives bulk relayout unchanged.

   const reuse=this.reusableExtension&&same;this.reusableExtension=false;
   // Bodies alone changed the last extension's inputs: rebuild their regions.
   const regional=this.bodyExtension&&same;this.bodyExtension=false;
   if(kick)this.plan.encode(encoder,p.supportPolicy);else if(!reuse){
    const phase=!this.phaseCurrent(same),bodiesOnly=this.phaseBodiesOnly(same);
    // The plan reads no solid: after bodies alone, the last plan's support holds.
    if(!regional)this.plan.encode(encoder,p.supportPolicy,true);if(!this.geometryCurrent)this.encodeGeometry(encoder);
    // Phase only: the authority below (split: the simulation authority
    // before the band rows, then the pressure authority) rewrites phi before
    // its readers; band rows and RHS form the volume correction themselves.
    if(phase){if(bodiesOnly)this.authority.encodeBodies(encoder,this.authorityGroup);else this.authority.encode(encoder,this.authorityGroup,false);}
    trace?.phase(encoder,V.support);
    // Regional wherever the layout holds: the work tiers name the regions
    // whose inputs moved since the tail's swept encode (a full one otherwise).
    if(!reuse&&!this.extensionIdle){this.extension.encode(encoder,this.extensionGroups,p.extensionSweeps??2,false,same);trace?.phase(encoder,V.extension);}
   }
   }
   // The stage scratch's other consumers start here: the extension's slots
   // are dead until the tail seeds them (UniformMixedExtension.useSlots).
   this.extension.invalidateRetained();
   // A kick moves no liquid: no trace, transport, surface volume or 4h base.
   let pairedExtension:ReturnType<typeof this.pairedExtensionGroups.get>;
   if(!kick){
   // An idle advance (UniformNarrowBandFlip.idle) holds the certificate's
   // inputs: the extension (extensionIdle), the plan's support, ownership and
   // dt (a dt change breaks idle). Its words and region velocities stand.
   const idle=this.narrowBandFlip.idle;
   if(!idle)this.plan.encodeCertificate(encoder,dt);
   this.surfaceWorkControl.framePlan.encode(encoder,idle&&this.geometryCurrent);
   this.blit.setSharedSchedule(this.surfaceWorkControl.framePlan.active&&this.surfaceWorkControl.framePlan.scheduleStages&&this.surfaceWorkControl.framePlan.coalescePasses);
   if(this.layoutViews){const certificate=this.plan.certificate;this.recordStageView(encoder,certificate.buffer,certificate.offset,"certificate");}
   if(!this.frozenQuiet){this.cache.encode(encoder,this.cacheGroup);if(this.ownership.capacity.fineTiles>0)this.hanging.encode(encoder,this.hangingGroup);}
   trace?.phase(encoder,V.transportReach);
   {
    // write holds p's control time in the key.
    const seconds=this.key.volumeControlSeconds??0,control=seconds>0?1/seconds:0;
    // Idle: phi, the budget and the rate's inputs are the last frame's, so is the gauge's rate (the copy below still lands it).
    if(!idle)this.surfaceVolume!.beginStep(encoder,this.surfaceVolumeGroup,dt,p.addedVolumeCells??0,control);
    // The gauge's rate reaches the authorities' parameters on the GPU: this
    // frame's host write left that word zero, and no readback sets it.
    if(control>0)encoder.copyBufferToBuffer(this.surfaceVolume!.budget!,4*NB_VOLUME_RATE_FLOAT,this.params.authority,8,4);
   }
   // The samples' move is the frame's first long stage: it runs while the
   // host encodes the surface stages behind it.
   // The move closes its own share of the phi stage: passes after a
   // submission's last mark are in no stage's time.
   {this.narrowBandFlip.move(encoder,dt,p.openTop,gravity);trace?.phase(encoder,V.phi);encoder=this.flush(encoder,trace);}
   // Idle: the advected scratch's only readers (reconstruct, redistance) do
   // not run, and phi is unchanged, so no resolve has a texel to change.
   if(!idle){this.surface.encode(encoder,this.surfaceGroup);this.phiResolve.encode(encoder,this.phiResolveGroups.scratch);}
   // Cell departures feed conservative volume transport only. NB-FLIP
   // measures volume from phi, and momentum writes this scratch field later.
   {
    // Eq. 4: the samples overrule the advected interface before distance,
    // occupancy and pressure read it.
    // Reconstruction reads the advected scratch and writes the live field.
    // Its metric bank then lets redistance update canonical vertices in place.
    this.narrowBandFlip.reconstruct(encoder);
    if(!idle)this.phiResolve.encode(encoder,this.phiResolveGroups.phi,undefined,true);
   }
   // Retire buried phantom surfaces and stale near-air distances only outside
   // the physical support collar. Resolve shared vertices before all readers.
   // Before redistance: its crossing search and metric then describe the
   // final field, and serve the transfer and update as they stand.
   // Off a kick the reconstruction's resolve already covers this layout, so
   // only clearAir's own 4h-lattice writes can change a hanging value.
   this.narrowBandFlip.clearAir(encoder);if(kick)this.phiResolve.encode(encoder,this.phiResolveGroups.phi,undefined,true);else if(!idle)this.phiResolve.encodeGated(encoder,this.phiResolveGroups.phi,this.narrowBandFlip.state,true);
   if(p.redistance!==false&&!idle){this.narrowBandFlip.retire(encoder);this.narrowBandFlip.redistance(encoder);this.phiResolve.encode(encoder,this.phiResolveGroups.phi,undefined,true);}
   // Surface producers publish exact changes once; derived geometry and the
   // next surface model consume these revisions instead of rescanning phi.
   this.narrowBandFlip.exportSurfaceRevisions(encoder);
   trace?.phase(encoder,V.phi);encoder=this.flush(encoder,trace);

   trace?.phase(encoder,V.coupling);
   // Apply shifts canonical vertices only; resolved readers below and the
   // next advect read the hanging texels.
   // NB-FLIP measures its volume against the budget and does not move the
   // surface to meet it: a shift the samples do not share is undone by the
   // next step's union with them, so it would be paid again every step.
   // That measurement is a diagnostic alone, so a step encodes it only for
   // a reader that asked (requestNarrowBandVolumeProbe).
   {if(this.narrowBandVolumeProbe){this.narrowBandVolumeProbe=false;this.surfaceVolume!.encode(encoder,this.surfaceVolumeGroup,1,false);}}
   // Nothing after this pass writes phi: the next advance starts from it,
   // and the renderer's 4h vertex base is published from it.
   if(!this.surfaceWorkControl.framePlan.retainsGeometry)this.coarsePhi.encode(encoder);
   this.phiResolved=true;
   }

   // Nothing below writes phi before the split reads the pressure geometry.
   // The sealed frame plan retains both geometry families and their
   // measured-volume mirror only when no producer ran since the last gather.
   if(!this.surfaceWorkControl.framePlan.retainsGeometry){const options=this.geometryOptions;options.pressure=true;options.changed=undefined;this.encodeGeometry(encoder,options);}this.geometryCurrent=true;
   trace?.phase(encoder,V.gather);

   // A kick forces the stored field where it stands.
   if(kick){this.copyWhole(encoder,this.fields.velocity,this.fields.departure);encoder.copyBufferToBuffer(this.fields.negative,0,this.fields.negativeDeparture,0,this.fields.negative.size);}
   else this.encodeParticleMomentum(encoder,p);
   if(!kick){
    // G2P must compare the same extrapolation of valid fluid faces on both
    // sides of pressure. Raw air-face clearing is not a physical FLIP delta.
    pairedExtension=this.pairedExtensionGroups.get(this.narrowBandFlip.extensionScratch);
    // An idle frozen advance encodes no extension (extensionIdle).
    if(this.extensionIdle){/* nothing to extend or retain */}
    else if(pairedExtension){
     // Keep only P2G's canonical scalar faces in the expired particle bank
     // while forces and pressure replace velocity. Topology stays fixed.
     this.extension.retainPair(encoder,pairedExtension.retain);
    }else{
     this.extension.encode(encoder,this.flipSnapshotExtensionGroups!,p.extensionSweeps??2);
     this.narrowBandFlip.snapshot(encoder);
    }
   }
   // An idle frozen advance holds phi (no geometry change, so no capillary cache is dirty) and every owner is frozen: the forces write nothing.
   this.dynamics.encodeClassify(encoder);if(!this.extensionIdle)this.forces.encode(encoder,this.forceGroup,p.surfaceTension>0);
   trace?.phase(encoder,A.advectionCorrection);encoder=this.flush(encoder,trace);
   // Pressure stays all-4h; no layout build and no CPU wait. The band rows
   // need this frame's simulation authority and u*, both rewritten in
   // pressure ownership by the split below.
   // The surface census only feeds the overlay: nothing in the solve reads it.
   if(this.layoutViews){this.surfaceBand.encode(encoder);this.recordStageGrid(encoder,this.surfaceBand.band.buffer,"band");}
   // The surface-only authority before transfer already wrote these exact
   // phi and phase values; transfer, forces, extension and snapshot only read
   // them (forces cache curvature in their normal scratch). A kick skips the
   // earlier authority, so it still needs this build, over every owner.
   if(kick)this.authority.encodeAll(encoder,this.authorityGroup);
   this.phaseKeyed=!bodies;this.band.encodePrepare(encoder);
   const split=this.split;
   split.transfer.encodeToPressure(encoder,split.toPressure);
   // The root's owners for the blocks held now, and the sentinel and free blocks' rows: after the transfer's copy of the certificate, ahead of every root reader.
   this.rootOwners.encode(encoder);
   split.authority.encode(encoder,split.authorityGroup);
   // Warm start: the root iterate begins at the last frame's presented
   // pressure (fixed all-4h owners; buildRhs zeroes air). From p=0 every
   // frame stopped at the same accepted residual, and a resting pool kept
   // that one smooth divergence: it expanded each step, conservative
   // transport diluted its V into a layer above phi, and the floor's lost
   // volume later surfaced as a growing cavity.
   {const to=this.levels[0]!.pressure;encoder.copyBufferToBuffer(this.presentation.pressure.buffer,0,to.buffer,to.offset??0,4*this.pressureWords);}
   this.projection.encode(encoder,"rhs",split.rhsGroup);
   // Idle explicitly holds phi and suppresses census/remap. With no moved
   // body or kick, pressure geometry is the previous solve's; only its RHS,
   // bounds and work list need refreshing. Any producer wakeup restores a
   // full rebuild (or exact regional propagation when enabled for diagnosis).
   this.cycles.encodeSetup(encoder,this.narrowBandFlip.idle&&!bodies&&!kick);
   this.cycles.encodeMeasure(encoder);this.acceptance.encode(encoder,this.acceptanceGroup,this.state,"initial");
   trace?.phase(encoder,A.pressureSetup);
   // The conservative schedule is encoded whole; GPU gates run only the
   // slots the last checkpoint and the lagged frame's plan call for. No CPU wait.
   const plannedBy=kick?0:frame-UNIFORM_MIXED_RECEIPT_RING,lagged=this.laggedOf(plannedBy);
   if(plannedBy>0&&!lagged)throw new Error(`Uniform mixed frame ${frame} encoded before frame ${plannedBy}'s receipt was checked`);
   // Taken: its values stand until the next receipt to land in its place, after this encode.
   if(lagged)lagged.frame=-1;
   const schedule=this.pressureSchedule,plan=p.fullPressureEnvelope||!lagged?this.initialPlan:uniformMixedPressureSpareFull(uniformMixedPressureReserve(lagged,this.initialPlan,Math.max(p.pressureReserve??0,UNIFORM_MIXED_SCHEDULE_FLOOR-lagged.vCycles-lagged.fullCycles)),this.initialPlan),vCycles=plan.vCycles;
   schedule.begin(plan);
   // The lagged plan is the window's largest cycle count plus the planner's
   // one spare; every slot from that spare on (the host floor and reserve
   // too) is capacity, closed in all but a few frames: the root launches
   // those narrow. Every slot is the root's one traversal form.
   const expected=p.fullPressureEnvelope||!lagged?schedule.slots:lagged.vCycles+lagged.fullCycles-1;
   for(let slot=0;slot<schedule.slots;slot++){
    const gated=schedule.gate(encoder,slot),capacity=slot>=expected;
    if(slot<vCycles)this.cycles.encodeVCycle(gated,capacity);else this.cycles.encodeFullCycle(gated,capacity);
    this.cycles.encodeMeasure(gated,capacity);this.acceptance.encode(gated,this.acceptanceGroup,this.state,"cycle");
    trace?.phase(encoder,slot<vCycles?A.pressureVCycles:A.pressureFullCycles);
   }
   // Projection reads the accepted iterate; the all-4h root has no seam,
   // so no reconstruction slope. Its gate enables it only on acceptance.
   this.projection.encode(schedule.gate(encoder,schedule.slots),"project",split.projectionGroup);
   schedule.end();
   // Frozen regions take the mean of the projection's corrections into their term.
   this.freeze?.encodeAccumulate(encoder);
   trace?.phase(encoder,A.pressureProjection);
   if(!this.frozenQuiet)split.transfer.encodeToSimulation(encoder,split.toSimulation);
   this.band.encodeSolve(encoder);
   // The projected field is final here: each region's change since the last frame.
   this.dynamics.encodePublish(encoder);
   if(!kick){
    // An idle frozen advance encodes no extension (extensionIdle).
    if(this.extensionIdle){/* nothing to extend or retain */}
    else if(pairedExtension){
     // Phase, topology and wall geometry are unchanged across pressure.
     // The old hierarchy finishes
     // into snapshot slots/taps; the current hierarchy then reuses its geometry.
     this.extension.preparePair(encoder,pairedExtension.current,p.extensionSweeps??2);
     this.extension.finishSnapshot(encoder,pairedExtension.old,p.extensionSweeps??2);
     this.narrowBandFlip.snapshot(encoder,pairedExtension.snapshots[(p.extensionSweeps??2)%2]!);
     this.extension.finishPair(encoder,this.extensionGroups,p.extensionSweeps??2,false);
    }else this.extension.encode(encoder,this.extensionGroups,p.extensionSweeps??2,true);
   }
   const root=this.levels[0]!,rootPressure=root.pressure,rootPhi=root.phi!;
   // Present the live all-4h words only: level 0 keeps the simulation
   // layout's capacity, but its owners and boundary slots are all-4h.
   encoder.copyBufferToBuffer(rootPressure.buffer,rootPressure.offset??0,this.presentation.pressure.buffer,0,4*this.pressureWords);
   encoder.copyBufferToBuffer(rootPhi.buffer,rootPhi.offset??0,this.presentation.phi.buffer,0,4*this.pressureOwnership.layout.cellCount);
   this.recordStageGrid(encoder,this.pressureOwnership.presentation.buffer,"pressure");
   // The h band re-solved this frame (empty on an all-4h simulation); the
   // live band pressures were presented by the band's own slot launch.
   encoder.copyBufferToBuffer(this.band.index,this.band.slotMapOffset,this.presentation.phi.buffer,4*this.stageBandWord,4*this.ownership.capacity.tiles);
   trace?.phase(encoder,V.band);
   // Rigid coupling reads the accepted h pressure and projected field;
   // the census below promotes around the integrated poses.
   if(bodies?.couple){bodies.couple(encoder);trace?.phase(encoder,A.rigidCoupling);}
   encoder.copyBufferToBuffer(this.state,0,readback,0,32);encoder.copyBufferToBuffer(this.reductions,0,readback,32,48);schedule.encodePlanCopy(encoder,readback,80);
   this.band.encodeReceipt(encoder,readback,88);encoder.copyBufferToBuffer(this.status,0,readback,120,4*UNIFORM_MIXED_STATUS_WORDS);
   // Sticky detail-storage violations of both group-0 ownerships.
   const violation=this.violationByte;
   encoder.copyBufferToBuffer(this.ownership.support,violation,readback,DETAIL_RECEIPT,4);encoder.copyBufferToBuffer(this.pressureOwnership.support,violation,readback,DETAIL_RECEIPT+4,4);
   this.ownership.encodeWorkReceipt(encoder,readback,WORK_RECEIPT);
   this.remap.encodeWorkReceipt(encoder,readback,REMAP_WORK_RECEIPT);this.cycles.encodeWorkReceipt(encoder,readback,ROOT_WORK_RECEIPT);
   encoder.copyBufferToBuffer(this.narrowBandFlip.state,0,readback,FLIP_RECEIPT,48);this.pagePool.encodeReceipt(encoder,readback,POOL_RECEIPT);
   encoder.copyBufferToBuffer(this.heat.tiers,4*this.heat.regions,readback,HEAT_RECEIPT,4*UNIFORM_COLD_FRAME_HEAT_WORDS);
   // The next head's census and advection read this extension: it is the
   // frame's velocity extension, priced there, not the census's.
   // Rigid integration changes body state, not the canonical fluid inputs
   // of extension. Its phase uses the current solid record: fine open masks
   // and coarse capacity remain the head's until the next record rebuild.
   // That next head explicitly invalidates/rebuilds before census or transport.
   if(!kick){
    // Phi and ownership have not changed since encodeParticleMomentum's
    // full plan, nor particles since the move (whose G2P takes this
    // projection at the next head): those support planes and the
    // post-pressure extension stand (the certificate takes its speed from it).
    this.plan.encodeTail(encoder,p.supportPolicy);this.reusableExtension=true;
    trace?.phase(encoder,V.extension);
   }else if(extendTail){this.encodeExtensionOf(encoder,p.supportPolicy,p.extensionSweeps,bodies!==undefined);trace?.phase(encoder,V.extension);}
   trace?.submit(encoder,this.fields.negative);this.submit(encoder);trace?.submitted();
   slot.frame=frame;slot.encoded=plan.vCycles+plan.fullCycles;slot.kick=kick>0;slot.target=this.pressureTarget;slot.unchecked=true;
   return this.check(slot);
  }catch(error){trace?.abort();this.failed=true;throw error;}finally{for(let i=0;i<held;i++)this.releases[i]!();this.busy=false;}
 }
 /** The frame encoders' copies and clears run as compute launches (UniformComputeBlit). */
 private readonly blit:UniformComputeBlit;
 private beginEncoder(trace?:UniformMixedFrameTrace):GPUCommandEncoder{return this.blit.adopt(this.device.createCommandEncoder(FRAME_ENCODER),trace?encoder=>trace.instrument(encoder):undefined);}
 private submit(encoder:GPUCommandEncoder):void{this.commands[0]=encoder.finish();this.queue.submit(this.commands);}
 /** Submits the segment encoded so far; the frame goes on in the encoder returned. */
 private flush(encoder:GPUCommandEncoder,trace?:UniformMixedFrameTrace):GPUCommandEncoder{trace?.submit(encoder,this.fields.negative);this.submit(encoder);return this.beginEncoder(trace);}
 /** Shared interval for differential replay; phase is not a momentum input,
  * and frame-plan consumes phi/velocity rather than authority scratch. */
 private encodeParticleMomentum(encoder:GPUCommandEncoder,p:UniformMixedFrameParameters):void{
  if(!this.frozenQuiet)this.momentum.encode(encoder,this.momentumGroup);
  // Idle: phi, velocity, ownership and tiers are the last plan's inputs (the plan reads no solid).
  if(!this.narrowBandFlip.idle)this.plan.encode(encoder,p.supportPolicy,true);
  this.authority.encode(encoder,this.authorityGroup,false);
  this.narrowBandFlip.transfer(encoder);
 }
 /** A ring slot around `readback`, with the frame's reactions to its map. */
 private receiptSlot(readback:GPUBuffer):ReceiptSlot{
  const words=new Uint32Array(TRACE_RECEIPT/4);
  const slot:ReceiptSlot={readback,unchecked:false,frame:0,encoded:0,kick:false,target:undefined,words,floats:new Float32Array(words.buffer),
   flip:words.subarray(FLIP_RECEIPT/4,FLIP_RECEIPT/4+12),status:words.subarray(30,30+UNIFORM_MIXED_STATUS_WORDS),
   work:words.subarray(WORK_RECEIPT/4,REMAP_WORK_RECEIPT/4),
   receipt:{cycles:0,encoded:0,residual:0,converged:true,dustOwners:0,dustMass_cells:0,orphanDustOwners:0,orphanDustMass_cells:0,bandTiles:0,bandCycles:0,bandResidual:0},
   accept:()=>this.receive(slot),reject:error=>this.refuse(slot,error)};
  return slot;
 }
 /** The slot's frame's receipt: fail fast on a rejected or unconverged solve. */
 private check(slot:ReceiptSlot):Promise<UniformMixedFrameReceipt>{
  let mapped:Promise<undefined>;
  try{mapped=slot.readback.mapAsync(GPUMapMode.READ);}catch(error){this.failed=true;slot.unchecked=false;return Promise.reject(error);}
  return mapped.then(slot.accept,slot.reject);
 }
 /** Its own function: one holding a closure allocates that closure's context at every entry, and receive runs every frame. */
 private detailViolation(detail:number,simulation:number,pressure:number,failure:string|undefined):Error{
  return new Error(`Uniform detail storage: a store a non-resident tile cannot hold (${Object.entries(UNIFORM_DETAIL_VIOLATION).filter(([,bit])=>detail&bit).map(([kind])=>kind).join(", ")} field; simulation ${simulation}, pressure ${pressure})${failure?`; ${failure}`:""}`);
 }
 private refuse(slot:ReceiptSlot,error:unknown):never{this.failed=true;slot.unchecked=false;throw error;}
 private receive(slot:ReceiptSlot):UniformMixedFrameReceipt{
  const mapped=slot.words,floats=slot.floats,frame=slot.frame,kick=slot.kick;
  try{mapped.set(new Uint32Array(slot.readback.getMappedRange(),0,TRACE_RECEIPT/4));slot.readback.unmap();}
  catch(error){this.failed=true;throw error;}finally{slot.unchecked=false;}
  try{
   this.narrowBandFlip.noteReceipt(slot.flip);
   const page=this.pagePoolInfo,base=POOL_RECEIPT/4,status=UNIFORM_PAGE_POOL_STATUS;
   // Receipts can finish out of order with multiple frames in flight.
   if(frame>=page.frame){page.frame=frame;page.generation=mapped[base+status.generation]!;
    page.residentPages=mapped[base+status.resident]!;page.claimedPages=mapped[base+status.claimed]!;
    page.retiredPages=mapped[base+status.pending]!;page.freePages=mapped[base+status.free]!;
    const d=this.workDomainInfo,w=slot.work;
    d.fineTiles=w[0]!;d.coarseTiles=w[1]!;d.seamFineTiles=w[4]!;d.seamCoarseTiles=w[5]!;d.regularCoarseTiles=w[6]!;d.dependencyRegions16=w[7]!;
    d.pressureListedOctants=mapped[ROOT_WORK_RECEIPT/4]!;d.remapJobs=mapped[REMAP_WORK_RECEIPT/4]!;d.transferPairs=slot.flip[9]!;}
   // The page pool's sticky fatal explains every later failure: a page the frame needed was never stored.
   const pool=uniformPagePoolFatalMessage(this.pagePool.layout,mapped,POOL_RECEIPT/4);if(pool)throw new Error(pool);
   // The GPU's own verdict first: it names the first failing frame.
   const failure=describeUniformMixedFrameStatus(slot.status);
   // A detail-storage violation explains every later failure.
   if(mapped[DETAIL_RECEIPT/4+1]!&UNIFORM_MIXED_ROOT_FATAL)throw new Error(`${UNIFORM_MIXED_ROOT_FATAL_MESSAGE}${failure?`; ${failure}`:""}`);
   if(mapped[DETAIL_RECEIPT/4+1]!&UNIFORM_MIXED_ROOT_UNREACHED)throw new Error(`${UNIFORM_MIXED_ROOT_UNREACHED_MESSAGE}${failure?`; ${failure}`:""}`);
   const detail=mapped[DETAIL_RECEIPT/4]!|mapped[DETAIL_RECEIPT/4+1]!;
   if(detail)throw this.detailViolation(detail,mapped[DETAIL_RECEIPT/4]!,mapped[DETAIL_RECEIPT/4+1]!,failure);
   if(failure)throw new Error(`${failure}${slot.status[UNIFORM_MIXED_STATUS.residencyAudit]?`; residency audit ${slot.status[UNIFORM_MIXED_STATUS.residencyAudit]} (1: absent near, 2: closure)`:""}${mapped[30]===UNIFORM_MIXED_FAILURE.layoutCapacity&&mapped[33]!&UNIFORM_MIXED_OVERFLOW_FINE?`; the layout holds ${mapped[WORK_RECEIPT/4]} h tiles, capacity ${this.ownership.capacity.fineTiles}`:""}; projection withheld${""}`);
   // Words 0-7 are the acceptance state, 8-19 the dust accounting.
   const residual=floats[1]!,count=mapped[7]!,target=slot.target!;
   if(mapped[4]!==0||mapped[5]===0||!Number.isFinite(residual)||residual<0||residual>target){
    throw new Error(`Uniform mixed pressure ${mapped[4]!==0?"rejected a non-improving cycle":"did not converge"}: candidate ${floats[0]}, accepted ${residual}, tolerance ${target}, ${count} cycles; projection withheld${""}`);
   }
   if(mapped[23]===UniformPressureArena.slotOverflow)throw new Error(`Uniform pressure band needs more than the arena's ${this.band.slotCapacity} brick slots; projection is incomplete`);
   if(mapped[23]!==0)throw new Error(`Uniform pressure band needs ${mapped[22]} tiles, over its capacity; projection is incomplete`);
   const bandTiles=mapped[22]!,bandResidual=floats[24]!;
   this.bandTiles=bandTiles;this.bandResidual=bandResidual;
   // Only an accepted solve plans; a failed one threw above. Frame
   // frame+UNIFORM_MIXED_RECEIPT_RING encodes it, whenever this resolves.
   if(!kick){const plan=this.lagged[frame%UNIFORM_MIXED_RECEIPT_RING]!;plan.frame=frame;plan.vCycles=mapped[20]!;plan.fullCycles=mapped[21]!;plan.bandTiles=bandTiles;plan.bandBricks=mapped[25]!;plan.bandLevel1=mapped[28]!;plan.bandLevel2=mapped[29]!;plan.bandCycles=mapped[26]!;plan.work.set(slot.work);plan.listed=mapped[REMAP_WORK_RECEIPT/4]!;plan.rootListed=mapped[ROOT_WORK_RECEIPT/4]!;for(let w=0;w<UNIFORM_COLD_FRAME_HEAT_WORDS;w++)plan.heat[w]=mapped[HEAT_RECEIPT/4+w]!;plan.heat[UNIFORM_COLD_FRAME_HEAT_WORDS]=slot.flip[0]!;}
   const receipt=slot.receipt;
   receipt.cycles=count;receipt.encoded=slot.encoded;receipt.residual=residual;receipt.dustOwners=mapped[13]!+mapped[18]!;receipt.orphanDustOwners=mapped[18]!;
   receipt.bandTiles=bandTiles;receipt.bandCycles=mapped[26]!;receipt.bandResidual=bandResidual;
   return receipt;
  }catch(error){this.failed=true;throw new Error(`Uniform mixed ${kick?`kick before frame ${frame+1}`:`frame ${frame}`}: ${error instanceof Error?error.message:String(error)}`,{cause:error});}
 }
 /** NB occupancy is a measurement, including immediately after a relayout.
  * No consumer may see stale independently remapped or displaced mass: the
  * geometry's own pass copies every live owner's target into the volume
  * (its mirror), where a whole-field copy followed each launch. */
 private encodeGeometry(encoder:GPUCommandEncoder,options?:Parameters<UniformMixedSurfaceGeometry["encode"]>[2]):void{
  // Only bodies moved since the last gather: their swept regions alone.
  const bodiesOnly=!options&&this.geometryBodies&&this.geometryBodiesOnly;this.geometryBodies=false;
  this.geometry.encode(encoder,this.geometryGroup,bodiesOnly?BODY_GEOMETRY:options);
 }
 /** Two fields of one class: the physical textures copy whole. */
 private copyWhole(encoder:GPUCommandEncoder,from:GPUTexture,to:GPUTexture):void{
  const d=this.fields.detail;if(d){d.copy(encoder,from,to);return;}
  encoder.copyTextureToTexture({texture:from},{texture:to},[from.width,from.height,from.depthOrArrayLayers]);
 }
 /** Takes the ungraded h/4h simulation layout; pressure stays all-4h. The
  * host (CPU) layout path: authored regions, and the initial layout. Its h
  * tiles are reserved first (reserveFine): a capacity the device refuses is
  * returned with nothing changed, the frame still on its last generation. */
 updateLayout(layout:UniformMixedLayout):string|undefined{
  if(!this.ready||this.busy||this.failed)throw new Error("Ownership edits require a completed frame");
  if(this.relayout)throw new Error("A GPU relayout owns the mixed layout: detach it (setRelayout()) before a host relayout");
  const refusal=this.reserveFine(layout.fineTiles.length);if(refusal)return refusal;
  this.cold.trigger();this.surfaceWorkControl.invalidate();this.dynamics.invalidate();
  // An authored refinement is immediate evidence of possible new band work;
  // reserve launch parallelism before its first frame, ahead of the lagged
  // liquid-band receipts. The actual band may occupy fewer of these tiles.
  this.band.observeWork(layout.fineTiles.length);
  this.reusableExtension=false;this.geometryCurrent=false;this.solidWidthsStale=true;
  // The remap reads the all-4h solid record (umTileOpen): a relayout ahead of
  // the first frame builds it. A pending live edit keeps the record of the
  // solids the GPU still holds (the host publishes the mask after this).
  if(this.solid?.recordStale&&!this.solidEditPending){
   const e=this.device.createCommandEncoder({label:"Uniform solid record before a host relayout"});this.solid.encodeCoarse(e,this.ownership.presentation);this.tileContext.encodeBuild(e);this.device.queue.submit([e.finish()]);
  }
  // Detail storage: the first layout installs (packed: deposits the uploads,
  // or emulates the remap from all-h when not every patch is resident);
  // later ones admit their patches before the remap and retire the rest
  // after the phi resolve, as the GPU relayout does.
  const f=this.fields,detail=f.detail,first=detail&&!detail.settled;
  if(first){
   if(detail.install(layout,{volume:f.volume,velocity:f.velocity,velocityScratch:f.velocityScratch}))this.remap.apply(layout);
   else this.ownership.update(layout);
  }else{
   // Slots for the new patches beside the resident ones (both live across the remap).
   detail?.reserve(layout);
   this.remap.apply(layout,detail&&((encoder,target)=>detail.encodeAdmit(encoder,target)));
  }
  const e=this.device.createCommandEncoder({label:"Uniform resolve remapped phi"});this.phiResolve.encode(e,this.phiResolveGroups.phi);
  if(!first)detail?.encodeRetire(e);
  // The pressure root takes the adopted generation's ring bit (a first layout has no admit).
  detail?.encodeSettle(e,true);
  {
   this.solid?.encodeSimulation(e,this.ownership.presentation);this.solidWidthsStale=false;
   this.plan.encode(e);this.encodeGeometry(e);
  }
  // The remap rewrote phi on the new ownership: the 4h vertex base follows.
  this.coarsePhi.encode(e);
  this.device.queue.submit([e.finish()]);
  // A domain placement whose capacity returned to zero: its restricted bases are the fields from here.
  detail?.commitRetire();
  this.phiResolved=true;
  return undefined;
 }
 /** Dynamic coarsening's GPU relayout, run at the head of every advance
  * (encodeRelayoutHead); undefined returns the layout to updateLayout. */
 private relayout?:UniformMixedFrameRelayout;
 /** The last advance ran the relayout's census (its residency certificate is live). */
 private censused=false;
 /** Request edits (UniformMixedFrameRelayout.edit) built in frames whose
  * receipts no advance has observed yet: a ring by frame, each the sum of
  * its frame's (a kick's and its frame's share one). */
 private readonly edits=Array.from({length:UNIFORM_MIXED_RECEIPT_RING},()=>({frame:-UNIFORM_MIXED_RECEIPT_RING,tiles:0,reach:0}));
 private readonly edited={tiles:0,reach:0};
 private noteEdit(frame:number,edit:UniformWorkEdit):void{
  const e=this.edits[frame%UNIFORM_MIXED_RECEIPT_RING]!;
  if(e.frame===frame){e.tiles+=edit.tiles;e.reach+=edit.reach;}else{e.frame=frame;e.tiles=edit.tiles;e.reach=edit.reach;}
 }
 /** The edits `frame`'s budgets carry: those of frames whose receipts it does not observe yet. */
 private editedBy(frame:number):UniformWorkEdit|undefined{
  const sum=this.edited;let any=false;sum.tiles=0;sum.reach=0;
  for(let i=0;i<this.edits.length;i++){const e=this.edits[i]!;if(e.frame>frame-UNIFORM_MIXED_RECEIPT_RING){any=true;sum.tiles+=e.tiles;sum.reach+=e.reach;}}
  return any?sum:undefined;
 }
 /** relayout.adopted on its relayout, bound when it is attached. */
 private relayoutAdopted?:(encoder:GPUCommandEncoder)=>void;
 /** The frame a relayout was attached after: its receipts and older ones
  * describe the host layout it replaced. */
 private workEpoch=0;
 /** reserved: the h tiles the relayout starts with storage for (a request's
  * host count, or the capacity already held); absent, every tile the pool
  * can store, with the band at its liquid bound (a census's first build may
  * ask any stored tile).
  * After that the capacity follows the build receipts: the builder defers a
  * build over its admission and the host moves the storage between frames
  * (reserveFine). The caller asks fineReservation first: a refusal here
  * throws. counted: the first build is a counted edit of the host layout
  * this replaces (UniformMixedFrameRelayout.edit bounds it), so that
  * layout's receipts and budgets stand and the edit's count is added: a
  * small request arriving launches a small frame. */
 setRelayout(relayout?:UniformMixedFrameRelayout,reserved?:number,counted=false):void{
  if(this.busy)throw new Error("The mixed relayout cannot change during an advance");
  // A census's first build has no receipt to size from: storage for every
  // tile the pool can store, the band at its liquid bound, unless the host
  // reserved for its requests. Detached, host layouts reserve their own
  // again (the next updateLayout).
  if(relayout){
   const refusal=reserved===undefined?this.reserveFine(this.ownership.capacity.address.poolTiles,true):this.reserveFine(reserved,false);
   if(refusal)throw new Error(`A GPU relayout holds ${reserved===undefined?"every tile the pool can store":`${reserved} tiles`} at h: ${refusal}`);
  }else{this.liquidBand=false;if(this.censused){this.ownership.resetResidency();this.censused=false;}}
  this.relayout=relayout;this.relayoutAdopted=relayout?.adopted?.bind(relayout);this.cold.trigger();
  // The census may admit any tile in its first frame and its counts reach
  // the host UNIFORM_MIXED_RECEIPT_RING frames late: launch at the ceilings
  // until a receipt of this owner arrives. A budget left at the host
  // layout's (one group from all-4h) runs the new h tiles on a few lanes.
  if(relayout&&!counted){this.workEpoch=this.frameIndex;this.band.observeWork(this.band.capacity);this.ownership.forgetWork();this.remap.forgetWork();}
  // markChanged latches a builder fatal into this frame's record.
  this.remap.bindStatus(this.status);
 }
 /** The dynamic frame head, horizon one: the census classifies the state
  * this frame starts from, for this frame's dt, and the frame advects on the
  * generation it builds. Order, all in the frame's first encoder, with no
  * host round trip:
  *  1. the extension of that state (velocityScratch) on the ownership it
  *     lives in: the last frame's tail extension (encodeExtensionOf) when no
  *     state or parameter changed since (reusableExtension; bodies, solid
  *     edits and host edits invalidate it), otherwise encoded here;
  *  2. census and builder (relayout.encode): band bits, then the generation;
  *  3. adopt (UniformMixedRemap.applyGpu): the builder's buffers into the
  *     target, remap volume, velocity, phi and the extension, adopt into the
  *     live ownership, publish; an unchanged generation lists no tile;
  *  4. phi resolve, geometry (solid edits, bodies' new cells) and the
  *     cut widths on the adopted generation;
  *  5. plan, gather and phase on it: they belong to the ownership, not the
  *     field, so they are rebuilt every frame (the host no longer knows
  *     whether the generation changed).
  * The certificate, momentum cache and hanging taps that follow read the
  * remapped extension. The classifier and the trace sample the same field:
  * the census bounds each tile's departures over all its owners' faces of
  * that extension on both sides of the relayout (it joins the trace's
  * sampled flow with zero), and the remap writes each changed face as a
  * convex combination of old faces of owners in the same tile and, where a
  * 4h face is handed to h faces, of the tile beside it on either tangential
  * axis: remapFace averages remapSample, which interpolates two old faces,
  * each tilted toward the nearer face beside it on its plane by at most
  * half their difference (remapSlope). An unlisted tile's faces are the
  * identity. The census's sampled flow takes the bounds of those tiles
  * (SAMPLE_REACH), so the remapped extension stays inside the census bounds
  * the generation was built for, and no second extension runs before
  * advection. */
 /** The page commit of a census, for the relayout's encode to place straight
  * after it: the census's requests become claims, the idle retire, and the
  * topologies' tails name the new stored set before any reader of it. The
  * census's residency certificate is marked over that set at once. */
 encodePageCommit(encoder:GPUCommandEncoder):void{
  if(!this.certificate)throw new Error("Uniform mixed frame: a page commit with no residency certificate registered (registerCertificate)");
  this.pagePool.encodeCommit(encoder);this.certificate.encodeResidency(encoder);this.blockNeighbours.encode(encoder);this.encodeNewBlocks(encoder);this.tileContext.encodeClaims(encoder);
 }
 /** Every owner's build of the blocks the last commit claimed, in the commit's
  * encoder: the fields' AIR image and the per-key owners' (page initialise),
  * then the solid record over it. */
 encodeNewBlocks(encoder:GPUCommandEncoder):void{
  this.pageInitialise!.encode(encoder);this.extension.invalidateRetained();
  this.solid?.encodeNewBlocks(encoder,this.ownership.presentation);
 }
 private encodeRelayoutHead(encoder:GPUCommandEncoder,p:UniformMixedFrameParameters,dt:number,same:boolean,relayout:UniformMixedFrameRelayout,bodies:boolean,trace?:UniformMixedFrameTrace):void{
  const reuse=this.reusableExtension&&same;this.reusableExtension=false;
  // Bodies alone changed the last extension's inputs: rebuild their regions.
  const regional=this.bodyExtension&&same;this.bodyExtension=false;
  // In the frame's first blit run; the remap's markListed publishes the
  // receipt's generation into the status record (a later failure names it).
  this.remap.encodeClear(encoder);
  if(!reuse){
   if(this.solidWidthsStale){this.solid?.encodeSimulation(encoder,this.ownership.presentation);this.solidWidthsStale=false;}
   const phase=!this.phaseCurrent(same),bodiesOnly=this.phaseBodiesOnly(same);
   // The plan reads no solid: after bodies alone, the last plan's support holds.
   if(!regional)this.plan.encode(encoder,p.supportPolicy,true);if(!this.geometryCurrent)this.encodeGeometry(encoder);
   if(phase){if(bodiesOnly)this.authority.encodeBodies(encoder,this.authorityGroup);else this.authority.encode(encoder,this.authorityGroup,false);}
   if(!this.extensionIdle){this.extension.encode(encoder,this.extensionGroups,p.extensionSweeps??2,false,regional);trace?.phase(encoder,V.extension);}
  }
  // The tiles as the frame starts, for the grid layer's relayout marks.
  if(this.layoutViews)this.recordStageView(encoder,this.ownership.presentation.buffer,this.ownership.presentation.offset??0,"previous");
  relayout.encode(encoder,dt,this.layoutViews);
  if(this.layoutViews){const reasons=relayout.reasons,importance=relayout.importance;this.recordStageView(encoder,reasons.buffer,reasons.offset,"reasons");this.recordStageView(encoder,importance.buffer,importance.offset,"importance");}
  // Detail storage admits the generation's patches ahead of the remap.
  this.fields.detail?.encodeAdmit(encoder,relayout.generation.topology);
  this.remap.applyGpu(encoder,relayout.generation,relayout.receipt,true,this.relayoutAdopted);
  this.geometryCurrent=false;
  trace?.phase(encoder,A.resolutionCensus);
  // Only tiles whose neighbourhood changed width can hang differently.
  const changes=relayout.generation.changes;
  this.phiResolve.encode(encoder,this.phiResolveGroups.phi,this.phiResolved?changes:undefined);
  this.surfaceWorkControl.encodeRemapRevisions(encoder,changes);
  // Claims and remaps raised tiers: the deep COLD flags and epochs follow.
  this.heat.encodeDeep(encoder);this.freeze?.encodeHead(encoder,dt,p.gravity,this.frozenQuiet);
  // The patches the generation no longer needs return to the base.
  this.fields.detail?.encodeRetire(encoder);
  this.fields.detail?.encodeSettle(encoder,false);
  // Only the relayout moved phi since the last geometry unless solids moved.
  // Bodies: current unless no geometry encode has taken their regions yet.
  const solidsMoved=this.solidEditPending||this.geometryBodies;
  if(this.solidEditPending||bodies){this.solidEditPending=false;}
  // Which cut tiles the simulation holds at h: the all-4h levels read their h texels.
  this.solid?.encodeSimulation(encoder,this.ownership.presentation,this.solidWidthsStale?undefined:changes);this.solidWidthsStale=false;
  if(this.solid)trace?.phase(encoder,V.solids);
  this.recordStageGrid(encoder,this.ownership.presentation.buffer,"transport");
  {const options=this.geometryOptions;options.pressure=false;options.changed=solidsMoved?undefined:changes;this.plan.encode(encoder,p.supportPolicy,true);this.encodeGeometry(encoder,options);}
  this.authority.encode(encoder,this.authorityGroup,false);
  trace?.phase(encoder,V.support);
 }
 /** Publishes the 4h vertex base from the live phi: t=0 and any host write
  * of phi outside an advance or a relayout. */
 publishCoarseVertexPhi():void{
  if(!this.ready||this.busy)throw new Error("The 4h vertex phi base publishes between frames");
  this.surfaceWorkControl.invalidate();this.dynamics.invalidate();
  if(this.coarsePhi.adopted)return;
  const e=this.device.createCommandEncoder({label:"Uniform publish 4h vertex phi base"});this.coarsePhi.encode(e);this.device.queue.submit([e.finish()]);
 }
 destroy():void{this.blit.destroy();this.surfaceWorkControl.destroy();this.cycles.destroy();this.surfaceVolume?.destroy();this.narrowBandFlip.destroy();this.coarsePhi.destroy();this.solid?.destroy();this.tileContext.destroy();this.pressureSchedule?.destroy();this.surfaceBand.destroy();this.band.destroy();this.plan.destroy();this.phiResolve.destroy();this.surface.destroy();this.geometry.destroy();this.momentum.destroy();this.hanging.destroy();this.remap.destroy();this.ownership.destroy();this.pageInitialise?.destroy();this.pagePool.destroy();this.split.transfer.destroy();this.extension.destroy();this.freeze?.destroy();for(const l of this.levels)if(l.ownership!==this.ownership)l.ownership.destroy();for(const r of this.owned)r.destroy();}
}
