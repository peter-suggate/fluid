import {UNIFORM_MIXED_OVERFLOW_FINE,uniformMixedDetailViolationWord} from "./uniform-mixed-topology.wgsl";
import {uniformMixedHangingBytes} from "./uniform-mixed-velocity-sampling.wgsl";
import {UNIFORM_DETAIL_POLICY} from "./uniform-detail-policy";
import {UNIFORM_DETAIL_RECEIPT_BYTES,UNIFORM_DETAIL_VIOLATION,type UniformDetailGroup,type UniformDetailStorage} from "./uniform-detail-fields";
import {UniformPressureSurfaceBand} from "./uniform-pressure-surface-band";
import {UniformPressureBand} from "./uniform-pressure-band";
import {uniformMixedDustMass} from "./uniform-mixed-dust-accounting.wgsl";
import {UniformMixedCleanup} from "./uniform-mixed-cleanup";
import {UniformMixedFramePlan} from "./uniform-mixed-frame-plan";
import {UniformMixedOwnershipTransfer,UniformMixedRemap} from "./uniform-mixed-remap";
import {UniformMixedPhiResolve} from "./uniform-mixed-phi-resolve";
import {UniformCoarseVertexPhi} from "./uniform-coarse-vertex-phi";
import {uniformMixedAllCoarseLayout} from "./uniform-mixed-layout";
import type {UniformMixedLayout} from "./uniform-mixed-layout";
import type {UniformScratchArena} from "./uniform-scratch-arena";
import type {WebGPUUniformPressureMultigrid} from "./webgpu-uniform-pressure-multigrid";
import type {WebGPUUniformVelocityExtrapolator} from "./webgpu-uniform-velocity-extrapolation";
import {UniformMixedTransportStage} from "./uniform-mixed-transport";
import {UNIFORM_WORK_RECEIPT_WORDS,UniformMixedOwnership,type UniformMixedGenerationBuffers} from "./uniform-mixed-ownership";
import {UniformMixedExtension} from "./uniform-mixed-extension";
import {UniformMixedHangingTaps, UniformMixedMomentumCache} from "./uniform-mixed-momentum-cache";
import {UniformMixedSurface} from "./uniform-mixed-surface";
import {UniformMixedSurfaceVolume} from "./uniform-mixed-surface-volume";
import {UniformMixedSurfaceGeometry} from "./uniform-mixed-surface-geometry";
import {UniformMixedSharpening} from "./uniform-mixed-sharpening";
import {UNIFORM_WORK_RELAYOUT_RESERVE,type UniformWorkEdit} from "./uniform-buffered-work";
import {UniformMixedMomentum,UNIFORM_MIXED_MOMENTUM_LIMITS} from "./uniform-mixed-momentum";
import {UniformMixedForces} from "./uniform-mixed-forces";
import {UniformMixedPressureAuthority} from "./uniform-mixed-pressure-authority";
import {uniformMixedPressureStorage} from "./uniform-mixed-pressure-boundary.wgsl";
import {UniformMixedPressureVelocity} from "./uniform-mixed-pressure-velocity";
import {UniformMixedPressureCycles,type UniformMixedPressureCycleLevel} from "./uniform-mixed-pressure-cycles";
import {UniformMixedPressureAcceptance} from "./uniform-mixed-pressure-acceptance";
import {UNIFORM_MIXED_SCHEDULE_FLOOR,UniformMixedPressureSchedule,uniformMixedPressureReserve,uniformMixedPressureSpareFull,type UniformMixedPressurePlan} from "./uniform-mixed-pressure-schedule";
import {UNIFORM_MIXED_FAILURE,UNIFORM_MIXED_STATUS,UNIFORM_MIXED_STATUS_WORDS,describeUniformMixedFrameStatus} from "./uniform-mixed-frame-status";
import {planUniformMixedPressureMemory} from "./uniform-mixed-pressure-memory";
import {DEFAULT_UNIFORM_CM11A_SCHEDULE,UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE,UNIFORM_PRESSURE_KICK_RESIDUAL_PER_STEP,UNIFORM_PRESSURE_RELATIVE_REDUCTION,type UniformCM11aSchedule} from "./pressure-policy";
import type {GPUTimestampPhase} from "../../core/performance-trace";
import {UNIFORM_ADVANCE_PHASE as A} from "./uniform-stages";
import {UNIFORM_VOLUME_PHASE as V} from "./uniform-volume-stages";
import {UniformMixedSolid,type UniformMixedSolidResources} from "./uniform-mixed-solid.wgsl";
import {UniformMixedSolidDisplacement} from "./uniform-mixed-solid-displacement";
import {UniformPipelineNeeds} from "./uniform-pipeline-needs";
import {UNIFORM_STAGE_GRID_HEADER_WORDS,UNIFORM_STAGE_VIEWS,uniformStageBandWords,uniformStageGridHeader,uniformStageGridWords,uniformStageViewWord} from "./uniform-stage-grids";

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
 uniformGroup:GPUBindGroup;
 sourceParams?:GPUBuffer;
 /** Voxel/terrain/vessel solids and rigid bodies: live voxel edits arrive
  * through editSolids(), bodies through advance(). The host must keep every
  * coarse owner a full tile away from a cut cell liquid can reach. */
 solid?:UniformMixedSolidResources;
 /** Compile by need: the host's registry (uniform-pipeline-needs). Omitted:
  * every variant is built by initialize(). */
 needs?:UniformPipelineNeeds;
 /** Two-stage pressure: an all-4h solve, then the h surface band
  * (UniformPressureBand). With solids, the all-4h levels read the static
  * coarse solid record and every liquid solid-coupled tile joins the band.
  * The solve always runs split, an all-4h simulation too (its transfer is the
  * identity and its band empty): in the all-4h ownership with its own surface
  * target and centre phi here, the simulation's own staying intact for the
  * renderer. */
 pressureGeometry:{target:GPUTexture;centerPhi:GPUTexture};
 /** The h fields' detail storage: its table rides the simulation and
  * pressure-root topologies (group 0 of every field stage). */
 detail?:UniformDetailStorage;
}
export interface UniformMixedFrameParameters {
 dt:number;gravity:number;density:number;viscosity:number;surfaceTension:number;
 openTop:boolean;noSlip:boolean;cubic:boolean;drain:boolean;
 /** Redistance keeps h vertices beside the surface (umPreserved). */
 preserve?:boolean;
 /** Smooth surface coarsening: rebuild coarse interface distance after material travel. */
 coarseSurfaceTravel?:boolean;
 /** Redistance publishes held 4h vertices' distance corrections for the census (uniformDetailHeldDistance). */
 coarseHeldDistance?:boolean;
 /** Encode the existing maximum slot envelope; convergence still stops execution. */
 fullPressureEnvelope?:boolean;
 /** Extra slots beyond the lagged planner's existing spare; bounded by the configured maximum. */
 pressureReserve?:number;
 totalSurfaceVolume?:boolean;redistance?:boolean;sharpening?:boolean;surfaceDeficitBalancing?:boolean;extensionSweeps?:number;
 supportPolicy?:{fineReach:number;shellReach:number;twoLevel:boolean;shellOnly:boolean};
 dust:number;orphanDust?:number;sharpeningStrength:number;sharpeningDistance:number;pressureTolerance:number;
 /** Sharpening sweeps (even: the last writes V; default 8) and surface-volume Newton rounds (default 2).
  * Zero sweeps or a zero sharpening band skips sharpening; zero rounds skips the surface-volume shift. */
 sharpeningSweeps?:number;surfaceVolumeRounds?:number;
}

/** One owner-driven frame sequence. All persistent fields and large scratch
 * ranges belong to the host. Fine and coarse are layouts of this sequence.
 * Only compact metadata, parameter/receipt buffers and 4h sampling caches are
 * owned here. No readback contains simulation fields. */
/** Plan support, phase and extension are dt-free (the certificate takes dt),
 * so a census extension serves the next frame whatever its step. */
/** Diagnostics: log each frame's pressure slots, coarse solves and band history. */
const PRESSURE_TRACE=typeof process!=="undefined"&&!!process.env.FLUID_MIXED_PRESSURE_TRACE;
/** Receipt bytes of the detail violation words (simulation, pressure root),
 * then the detail storage's residency header. */
const DETAIL_RECEIPT=120+4*UNIFORM_MIXED_STATUS_WORDS;
const DETAIL_RECEIPT_WORDS=2+UNIFORM_DETAIL_RECEIPT_BYTES/4;
const WORK_RECEIPT=DETAIL_RECEIPT+4*DETAIL_RECEIPT_WORDS;
const SHARPEN_WORK_RECEIPT=WORK_RECEIPT+4*UNIFORM_WORK_RECEIPT_WORDS;
const REMAP_WORK_RECEIPT=SHARPEN_WORK_RECEIPT+8;
// The root cycle list's count: the width of the root's listed launches.
const ROOT_WORK_RECEIPT=REMAP_WORK_RECEIPT+4;
const TRACE_RECEIPT=ROOT_WORK_RECEIPT+4;
const extensionKey=(p:UniformMixedFrameParameters)=>JSON.stringify({...p,dt:0});
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
export interface UniformMixedFrameReceipt{cycles:number;encoded:number;residual:number;converged:boolean;dustOwners:number;dustMass_cells:number;orphanDustOwners:number;orphanDustMass_cells:number;
 /** The h band this frame re-solved: tiles, completed cycles, final residual (0 when the band is empty). */
 bandTiles:number;bandCycles:number;bandResidual:number}
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
 * tile. Otherwise (host layouts, and a GPU relayout under a reserved
 * capacity) every reserved h tile may be a band tile.
 * fixed: never reserved again (QA): a layout over it throws.
 * byteBudget: bytes the capacity-sized buffers and the detail fields may hold at once, a
 * reservation's old and new together; absent, only the device's buffer and
 * binding limits refuse one. */
export interface UniformMixedFrameCapacity{fineTiles:number;liquidBand:boolean;fixed?:boolean;byteBudget?:number}

export class UniformMixedFrame {
 readonly transport:UniformMixedTransportStage;
 get ownership(){return this.transport.ownership;}
 /** The surface stage's held-vertex corrections (UniformMixedSurface.held), for the Dynamic census. */
 get heldDistance(){return this.surface.held;}
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
 private stageBandTiles:number;
 private readonly split:{
  transfer:UniformMixedOwnershipTransfer;toPressure:UniformDetailGroup;toSimulation:UniformDetailGroup;
  authority:UniformMixedPressureAuthority;authorityGroup:UniformDetailGroup;
  rhsGroup:UniformDetailGroup;projectionGroup:UniformDetailGroup;
 };
 /** Two-stage pressure: every pressure level is all-4h and
  * UniformPressureBand re-solves the h surface band after it. */
 private readonly band:UniformPressureBand;
 private readonly bandParams:GPUBuffer;
 /** Band tiles of the last accepted advance. */
 bandTiles?:number;
 /** Largest band row residual of the last accepted advance, 1/s. */
 bandResidual?:number;
 readonly solid?:UniformMixedSolid;
 private readonly displacement?:UniformMixedSolidDisplacement;
 /** A live voxel edit awaits its displacement at the next frame's head. */
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
 private readonly cleanup:UniformMixedCleanup;
 private readonly cleanupGroups:readonly [UniformDetailGroup,UniformDetailGroup];
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
 private readonly surfaceVolume:UniformMixedSurfaceVolume;
 private readonly geometry:UniformMixedSurfaceGeometry;
 private readonly sharpen:UniformMixedSharpening;
 private readonly momentum:UniformMixedMomentum;
 private readonly forces:UniformMixedForces;
 private readonly authority:UniformMixedPressureAuthority;
 private readonly projection:UniformMixedPressureVelocity;
 private readonly cycles:UniformMixedPressureCycles;
 private readonly acceptance:UniformMixedPressureAcceptance;
 private extensionGroups!:readonly [UniformDetailGroup,UniformDetailGroup];
 private readonly cacheGroup:UniformDetailGroup;
 private surfaceGroups!:readonly [UniformDetailGroup,UniformDetailGroup];
 private surfaceVolumeGroup!:UniformDetailGroup;
 private readonly geometryGroup:UniformDetailGroup;
 private sharpenGroups!:readonly [UniformDetailGroup,UniformDetailGroup];
 private readonly momentumGroup:UniformDetailGroup;
 private forceGroup!:UniformDetailGroup;
 /** The scratch every non-pressure stage overlaps from offset 0, none of
  * whose contents survives its stage: stageBytesAt the h-tile capacity. */
 private stageScratch:GPUBuffer;
 private readonly coarseCache:GPUTexture;
 private authorityGroup:UniformDetailGroup;
 private readonly acceptanceGroup:GPUBindGroup;
 private readonly params:Record<"extension"|"surface"|"momentum"|"forces"|"authority"|"sharpen"|"projection"|"acceptance",GPUBuffer>;
 private readonly state:GPUBuffer;
 /** The sticky GPU failure record (uniform-mixed-frame-status): the pressure
  * schedule's last gate is the frame's verdict and latches it; a latched
  * record closes every later pressure slot, the band and the transfer back
  * to simulation. Stages that detect their own failures bind it read_write
  * (umLatchFailure). Never cleared: a failed frame fails the sequence. */
 readonly status:GPUBuffer;
 /** Receipt ring: a frame's receipt maps while later frames encode. */
 private readonly readbacks:readonly GPUBuffer[];
 /** Ring slots whose frame's receipt has not been checked yet. */
 private readonly unchecked=new Set<GPUBuffer>();
 private frameIndex=0;
 /** Each checked frame's planner output, keyed by frame. Frame N encodes
  * frame N-UNIFORM_MIXED_RECEIPT_RING's, never "the newest to resolve":
  * which receipts have mapped depends on wall-clock timing, so a newest-wins
  * plan made free-running and awaited runs diverge. The receipt ring makes
  * frame N reuse that frame's buffer, so its receipt is always checked. */
 private readonly lagged=new Map<number,UniformMixedPressurePlan & {bandTiles:number;work:Uint32Array;sharpenWork:Uint32Array;listed:number;rootListed:number}>();
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
 private sharpenList:GPUBuffer;
 /** See UniformMixedFrameCapacity: a GPU relayout sets it, detaching clears it. */
 private liquidBand:boolean;
 private readonly fixedCapacity:boolean;
 private readonly byteBudget?:number;
 private busy=false;
 private failed=false;
 get allocatedBytes():number{return this.coarsePhi.allocatedBytes+(this.displacement?.allocatedBytes??0)+this.surfaceBand.allocatedBytes+this.band.allocatedBytes+this.plan.allocatedBytes+(this.solid?.allocatedBytes??0)+this.surface.allocatedBytes+this.hanging.allocatedBytes+this.transport.allocatedBytes+this.remap.allocatedBytes+this.split.transfer.allocatedBytes+this.levels.filter(l=>l.ownership!==this.ownership).reduce((n,l)=>n+l.ownership.allocatedBytes,0)+this.owned.reduce((n,r)=>n+("size" in r?r.size:r.width*r.height*r.depthOrArrayLayers*16),0);}
 /** layout: the generation the t=0 fields are in (all h; the first
  * updateLayout remaps from it). capacity: the h tiles the owner-indexed
  * buffers hold until a layout reserves its own, independent of that seed. */
 constructor(private readonly device:GPUDevice,layout:UniformMixedLayout,capacity:UniformMixedFrameCapacity,private readonly fields:UniformMixedFrameFields,openTop:boolean,private readonly schedule:UniformCM11aSchedule=DEFAULT_UNIFORM_CM11A_SCHEDULE){
  const f=fields;
  this.liquidBand=capacity.liquidBand;this.fixedCapacity=!!capacity.fixed;this.byteBudget=capacity.byteBudget;
  const buffer=(label:string,size:number,usage:number)=>{const b=device.createBuffer({label,size,usage});this.owned.push(b);return b;};
  const uniform=(name:string,size:number)=>buffer(`Uniform ${name} parameters`,size,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
  this.params={extension:uniform("extension",16),surface:uniform("surface",32),momentum:uniform("momentum",32),forces:uniform("forces",48),authority:uniform("authority",16),sharpen:uniform("sharpen",32),projection:uniform("projection",32),acceptance:uniform("acceptance",16)};
  this.state=buffer("Uniform pressure acceptance",32,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST);
  this.status=buffer("Uniform mixed frame status",4*UNIFORM_MIXED_STATUS_WORDS,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST);
  this.initialPlan={vCycles:schedule.vCycles,fullCycles:schedule.fullCycles};
  this.readbacks=Array.from({length:UNIFORM_MIXED_RECEIPT_RING},(_,i)=>buffer(`Uniform pressure receipt and mass accounting ${i}`,TRACE_RECEIPT+(PRESSURE_TRACE?336:0),GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ));
  this.reductions=buffer("Uniform dust accounting",48,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC);
  const caches=Array.from({length:1},(_,i)=>{const t=device.createTexture({label:`Uniform 4h sampling cache ${i}`,size:layout.lattice.dimensions.map(n=>n/4+2),dimension:"3d",format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING});this.owned.push(t);return t;});
  const coarseLayout=uniformMixedAllCoarseLayout(layout);
  const solid=this.solid=f.solid?new UniformMixedSolid(device,f.solid,coarseLayout):undefined;
  if(solid&&f.needs)solid.needs=f.needs;
  this.coarseCache=caches[0]!;
  this.transport=new UniformMixedTransportStage(device,layout,f.volume,f.volumeScratch,f.departure,{phi:f.phi,params:this.params.sharpen,reductions:this.reductions,resolved:true},f.sourceParams,solid,capacity.fineTiles);
  const o=this.ownership;
  this.displacement=solid?new UniformMixedSolidDisplacement(device,o,solid):undefined;
  this.plan=new UniformMixedFramePlan(device,o,f.volume,f.phi,f.velocity,f.negative,f.velocityScratch,f.negativeScratch,true);
  this.cleanup=new UniformMixedCleanup(device,o,solid,true);
  this.cleanupGroups=[this.cleanup.bind(f.volume,f.volumeScratch,f.phi,this.params.sharpen,this.reductions),this.cleanup.bind(f.volumeScratch,f.volume,f.phi,this.params.sharpen,this.reductions)];
  this.remap=new UniformMixedRemap(device,o,{volume:f.volume,velocity:f.velocity,phi:f.phi,negative:f.negative},{volume:f.volumeScratch,velocity:f.velocityScratch,phi:f.phiScratch,negative:f.negativeScratch},solid);
  // The census extension crosses a relayout in these instead of being
  // rebuilt. Its faces stage in the unit taps: every texel a sampler reads
  // is rewritten by hanging/unitFaces before the first sampler, and nothing
  // reads them from the previous frame's forces until then.
  this.hanging=new UniformMixedHangingTaps(device,o,f.detail);
  this.remap.bindExtension({volume:f.volume,velocity:f.velocity,phi:f.phi,negative:f.negative},{velocity:this.hanging.unitVelocity,negative:buffer("Uniform mixed remapped extension walls",f.negativeScratch.size,GPUBufferUsage.STORAGE)});
  this.phiResolve=new UniformMixedPhiResolve(device,o);this.phiResolveGroups={phi:this.phiResolve.bind(f.phi),scratch:this.phiResolve.bind(f.phiScratch)};
  this.coarsePhi=new UniformCoarseVertexPhi(device,o,f.phi);
  // Domain placement: the published base is the phi field's own base block.
  if(f.detail?.layout.domain)f.detail.adoptBase(f.phi,this.coarsePhi.texture);
  // The arena, low to high: the root, the native levels.
  const native=Math.min(...[f.pressure.pressure,f.pressure.rhs,f.pressure.minimum,f.pressure.phi,f.pressure.topology].map(v=>v.buffer!.offset??0));
  if(f.arena.rootOffset+f.arena.rootBytes>native)throw new Error("The mixed pressure root overlaps the native continuation fields");
  const memory=planUniformMixedPressureMemory(coarseLayout,f.arena.rootOffset,f.arena.rootBytes);
  const view=(r:{offset:number;size:number}):GPUBufferBinding=>({buffer:f.arena.buffer,...r});
  // Pressure phi, twice: the simulation authority's (simulation owners, at
  // their capacity) for the band's list and rows, and the split authority's
  // (the fixed all-4h owners) for the root solve, the projection and the
  // band's start. Only the first follows the h-tile capacity.
  const phiUsage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST;
  this.authorityPhi=buffer("Uniform mixed pressure phi",4*o.capacity.owners,phiUsage);
  const rootPhi:GPUBufferBinding={buffer:buffer("Uniform mixed root pressure phi",4*coarseLayout.cellCount,phiUsage)};
  // Pressure's all-4h root keeps the mixed rows; the native hierarchy
  // solves its correction from its n/4 level.
  // With solids, the root's topology is the static all-4h solid record,
  // read in place: rebuilt only when the solids change, never copied.
  if(solid?.coarse)this.solidTopology={buffer:solid.coarse.record,size:16*solid.coarse.count};
  {const r=memory.root;
   this.levels=[{ownership:new UniformMixedOwnership(device,coarseLayout,false,0),pressure:view(r.pressure),rhs:[view(r.rhs[0]),view(r.rhs[1])],minimum:[view(r.minimum[0]),view(r.minimum[1])],phi:rootPhi,frozen:view(r.frozen),residual:view(r.residual),
    topology:f.solid?{buffer:this.solidTopology!}:undefined}];}
  const root=this.levels[0]!,p=root.ownership;
  if(f.detail){f.detail.attach(o);f.detail.attach(p);}
  // The h near-surface extension carries the advancing level-set toe.
  // A regular-only hierarchy previously lost long-dam toe motion.
  this.extension=new UniformMixedExtension(device,o,f.extension,false,true);
  this.cache=new UniformMixedMomentumCache(device,o);
  const cacheFields={coarseExtended:caches[0]!};
  this.cacheGroup=this.cache.bind({extended:f.velocityScratch,negative:f.negativeScratch,...cacheFields});
  // Surface and momentum share hanging taps of velocityScratch: nothing
  // between the cache and forces writes it, negativeScratch or cache 0.
  this.hangingGroup=this.hanging.bind({extended:f.velocityScratch,negative:f.negativeScratch,coarse:caches[0]!});
  this.surface=new UniformMixedSurface(device,o,f.sourceParams,solid,true,true);
  this.surfaceVolume=new UniformMixedSurfaceVolume(device,o,solid,true);
  // The full launch after surface volume also writes split pressure's all-4h geometry.
  this.geometry=new UniformMixedSurfaceGeometry(device,o,solid,true,true);this.geometryGroup=this.geometry.bind(f.phi,f.target,f.centerPhi,f.pressureGeometry);
  this.sharpenList=buffer("Uniform mixed sharpening tile list",UniformMixedSharpening.workBytes(layout.tiles.length,o.capacity.owners),GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST);
  this.sharpen=new UniformMixedSharpening(device,o,solid,{list:this.sharpenList},true);
  // Retain the projected h velocity detail: restricting all momentum to 4h
  // removes the long-dam toe even when total liquid volume is conserved.
  this.momentum=new UniformMixedMomentum(device,o,true,true,f.sourceParams);
  this.momentumGroup=this.momentum.bind({unitVelocity:this.hanging.unitVelocity,extended:f.velocityScratch,physical:f.velocity,phase:f.phase,volume:f.volume,centerPhi:f.centerPhi,negative:f.negativeScratch,output:f.departure,outputNegative:f.negativeDeparture,params:this.params.momentum,...cacheFields});
  // Viscosity reads exact MAC sites from the momentum fill (the extended
  // velocity): h sites in the unit texture, 4h sites in the coarse cache.
  this.forces=new UniformMixedForces(device,o,true,f.sourceParams,solid,true);
  this.stageScratch=buffer("Uniform mixed stage scratch",this.stageBytesAt(o.capacity.fineTiles),GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST);this.bindStage();
  this.authority=new UniformMixedPressureAuthority(device,o,solid);this.authorityGroup=this.bindAuthority();
  // Pressure couples the all-4h owners through the static coarse record.
  const coarseSolid=!!solid;
  this.projection=new UniformMixedPressureVelocity(device,p,f.sourceParams,solid,coarseSolid);
  // A cut 4h face's flux sums its h faces of the forced field, which the split
  // leaves in simulation ownership in the scratch pair.
  const fine=coarseSolid?{velocity:f.velocityScratch,negative:f.negativeScratch}:undefined;
  const bindProjection=(input:{velocity:GPUTexture;negative:GPUBuffer},output:{velocity:GPUTexture;negative:GPUBuffer},volume:GPUTexture,fine?:{velocity:GPUTexture;negative:GPUBuffer})=>{
   const common={velocity:input.velocity,negative:{buffer:input.negative},phi:root.phi!,params:this.params.projection};
   return [this.projection.bindRhs({...common,correction:f.correction,rhs:root.rhs[0],minimum:root.minimum![0]!,pressure:root.pressure,fine}),
    this.projection.bindProjection({...common,pressure:root.pressure,volume,output:output.velocity,outputNegative:{buffer:output.negative}})] as const;
  };
  const g=f.pressureGeometry;
  {
   // Split: the forced field reaches pressure ownership in velocity/negative
   // and is projected into the scratch pair, then transferred back. Volume
   // reaches pressure ownership in its (free) scratch field.
   const [rhsGroup,projectionGroup]=bindProjection({velocity:f.velocity,negative:f.negative},{velocity:f.velocityScratch,negative:f.negativeScratch},f.volumeScratch,fine);
   const transfer=new UniformMixedOwnershipTransfer(device,o,p,solid);
   const present=(label:string,bytes:number)=>buffer(`Uniform presented ${label}`,bytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST);
   // The record holds any band this frame builds: a larger one is fatal.
   const n=layout.tiles.length;this.stageBandTiles=this.bandTilesAt(o.capacity.fineTiles,capacity.liquidBand);
   // Presented pressure and phi hold the live all-4h words only (level 0
   // keeps the simulation layout's capacity).
   this.stageBandWord=p.layout.cellCount;this.stageGridWord=this.stageBandWord+uniformStageBandWords(n,this.stageBandTiles);
   this.presentation={pressure:{buffer:present("pressure",4*uniformMixedPressureStorage(p.layout).count)},phi:{buffer:present("pressure phi and stage grids",4*(this.stageGridWord+uniformStageGridWords(n)))}};
   const authority=new UniformMixedPressureAuthority(device,p,solid,coarseSolid,true);
   this.split={transfer,rhsGroup,projectionGroup,
    toPressure:transfer.bind({volume:f.volume,velocity:f.velocityScratch,negative:f.negativeScratch},{volume:f.volumeScratch,velocity:f.velocity,negative:f.negative},this.status),
    toSimulation:transfer.bind({volume:f.volume,velocity:f.velocityScratch,negative:f.negativeScratch},{volume:f.volumeScratch,velocity:f.velocity,negative:f.negative},this.status),
    authority,authorityGroup:authority.bind({centerPhi:g.centerPhi,volume:f.volumeScratch,targetFill:g.target,phi:root.phi!,phase:f.phase,correction:f.correction,scratch:root.frozen,params:this.params.authority,
     fine:coarseSolid?{phi:f.phi}:undefined,simulation:{buffer:o.presentation.buffer,size:4*o.capacity.tiles}})};
  }
  this.bandParams=uniform("pressure band",64);
  this.surfaceBand=new UniformPressureSurfaceBand(device,o,f.phi);
  this.band=new UniformPressureBand(device,o,p,{phi:{buffer:this.authorityPhi},coarsePhi:root.phi!,correction:f.correction,vertexPhi:f.phi,
   forced:{velocity:f.velocityScratch,negative:f.negativeScratch},velocity:f.velocity,negative:f.negative,copy:f.velocityScratch,
   coarsePressure:root.pressure,params:this.bandParams,presentation:{buffer:this.presentation.phi.buffer,word:this.stageBandWord+layout.tiles.length},capacity:this.stageBandTiles},undefined,solid);
  if(this.band.capacity!==this.stageBandTiles)throw new Error("The stage grids' band section must match the band capacity");
  // Setup is encoded once per solve, ungated (advance); cycles are pure cycle work.
  this.cycles=new UniformMixedPressureCycles(device,root,view(memory.backup),f.pressure,f.uniformGroup,openTop);
  this.acceptance=new UniformMixedPressureAcceptance(device,p);
  this.acceptanceGroup=this.acceptance.bind({residual:root.residual,state:this.state,params:this.params.acceptance});
 }
 /** The stage scratch at `fineTiles` h tiles: the largest of its consumers,
  * each of which checks the buffer against its own need when it binds. */
 private stageBytesAt(fineTiles:number):number{
  const n=this.ownership.capacity.tiles;
  return Math.ceil(Math.max(UniformMixedTransportStage.scratchRanges(n,fineTiles).bytes,this.extension.scratchBytesAt(fineTiles),this.surface.scratchBytes,
   this.surfaceVolume.scratchBytesAt(fineTiles),UniformMixedSharpening.scratchBytes(63*fineTiles+n,fineTiles),UniformMixedForces.normalBytes(63*fineTiles+n))/256)*256;
 }
 /** Every group over the stage scratch or the sharpening list (reserveFine binds them again). */
 private bindStage():void{
  const f=this.fields,stage:GPUBufferBinding={buffer:this.stageScratch};
  this.transport.bindScratch(this.stageScratch);
  this.extensionGroups=this.extension.bind({physical:f.velocity,phase:f.phase,negative:f.negative,output:f.velocityScratch,outputNegative:f.negativeScratch,scratch:stage,params:this.params.extension});
  const surfaceFields={unitVelocity:this.hanging.unitVelocity,velocity:f.velocityScratch,coarseVelocity:this.coarseCache,volume:f.volume,negative:f.negativeScratch,departures:f.departure,params:this.params.surface,evidence:stage};
  this.surfaceGroups=[this.surface.bind({...surfaceFields,phi:f.phi,outputPhi:f.phiScratch}),this.surface.bind({...surfaceFields,phi:f.phiScratch,outputPhi:f.phi})];
  this.surfaceVolumeGroup=this.surfaceVolume.bind(f.phi,f.volume,f.phi,stage);
  this.sharpenGroups=[this.sharpen.bind(f.volume,f.volumeScratch,f.phi,f.target,f.centerPhi,stage,this.params.sharpen,this.reductions),this.sharpen.bind(f.volumeScratch,f.volume,f.phi,f.target,f.centerPhi,stage,this.params.sharpen,this.reductions)];
  this.forceGroup=this.forces.bind({unitVelocity:this.hanging.unitVelocity,advected:f.departure,phi:f.phi,volume:f.volume,centerPhi:f.centerPhi,coarseVelocity:this.coarseCache,negative:f.negativeDeparture,output:f.velocityScratch,outputNegative:f.negativeScratch,params:this.params.forces,curvature:f.phase,normals:stage});
 }
 private bindAuthority():UniformDetailGroup{
  const f=this.fields;
  return this.authority.bind({centerPhi:f.centerPhi,volume:f.volume,targetFill:f.target,phi:{buffer:this.authorityPhi},phase:f.phase,correction:f.correction,scratch:this.levels[0]!.frozen,params:this.params.authority});
 }
 private bandTilesAt(fineTiles:number,liquidBand:boolean):number{
  return liquidBand?UniformPressureBand.capacityOf(this.ownership.capacity.tiles,fineTiles):Math.max(1,fineTiles);
 }
 /** Every buffer the h-tile capacity sizes, at `fineTiles` h tiles and
  * `bandTiles` band tiles: label and bytes. Each is one storage binding. */
 private capacityBuffers(fineTiles:number,bandTiles:number):[string,number][]{
  const n=this.ownership.capacity.tiles,owners=63*fineTiles+n,band=this.band.bytesAt(bandTiles);
  return [["stage scratch",this.stageBytesAt(fineTiles)],["velocity tap cache",uniformMixedHangingBytes(n,fineTiles)],["pressure phi",4*owners],["sharpening tile list",UniformMixedSharpening.workBytes(n,owners)],
   ...(this.displacement?[["solid displacement deposits",4*owners] as [string,number]]:[]),
   ["pressure band rows",band.rows],["pressure band aggregates",band.aggregates],["pressure band iterate",band.iterate],
   ["presented pressure phi",4*(this.stageBandWord+uniformStageBandWords(n,bandTiles)+uniformStageGridWords(n))]];
 }
 /** QA: the capacity never changes (uniformMixedFixedFineCapacity). */
 get capacityFixed():boolean{return this.fixedCapacity;}
 /** The capacity reserveFine(need) would adopt, or why the device cannot
  * hold it. Changes nothing: admission is decided here, before a layout is
  * adopted. Growth is geometric (UNIFORM_DETAIL_POLICY.poolGrowth), exact
  * where that does not fit; a need under 1/growth^2 of the capacity returns
  * to it. exact: `need` itself, the caller's own rule (a relayout's
  * capacity follows its receipts). A fixed capacity (QA) stays: the layout
  * over it throws. */
 fineReservation(need:number,liquidBand=this.liquidBand,exact=false):{fineTiles:number;bandTiles:number;bytes:number;refusal?:string}{
  const c=this.ownership.capacity,n=c.tiles,held=c.fineTiles,limits=this.device.limits,g=UNIFORM_DETAIL_POLICY.poolGrowth;
  if(!Number.isInteger(need)||need<0||need>n)throw new Error(`A mixed frame of ${n} tiles cannot reserve ${need} h tiles`);
  const total=(list:[string,number][])=>list.reduce((sum,[,bytes])=>sum+bytes,0);
  // The detail fields' textures count beside the buffers; a reservation that replaces them
  // (a domain placement crossing zero) holds both generations across the transfer.
  const detail=this.fields.detail,fieldsHeld=detail?.fieldBytesAt(held)??0;
  const fields=(fineTiles:number)=>{const next=detail?.fieldBytesAt(fineTiles)??0;return next===fieldsHeld?0:next;};
  const live=total(this.capacityBuffers(held,this.stageBandTiles))+fieldsHeld,most=Math.min(limits.maxBufferSize,limits.maxStorageBufferBindingSize);
  let refusal:string|undefined;
  for(const fineTiles of this.fixedCapacity?[held]:exact?[need]:need>held?[Math.min(n,Math.max(need,g*held)),need]:need*g*g<held?[need]:[held]){
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
  const retired:GPUBuffer[]=[];
  const replace=(old:GPUBuffer,size:number)=>{const b=this.device.createBuffer({label:old.label,size,usage:old.usage});this.owned[this.owned.indexOf(old)]=b;retired.push(old);return b;};
  if(owners){
   o.reserveFine(r.fineTiles,exact&&!!this.relayout);
   this.authorityPhi=replace(this.authorityPhi,4*o.capacity.owners);
   this.sharpenList=replace(this.sharpenList,UniformMixedSharpening.workBytes(n,o.capacity.owners));this.sharpen.setWork(this.sharpenList);
   this.stageScratch=replace(this.stageScratch,this.stageBytesAt(r.fineTiles));
   this.authorityGroup=this.bindAuthority();this.bindStage();
  }
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
  this.pressureSchedule=new UniformMixedPressureSchedule(this.device,this.schedule,this.state,this.fields.pressure.tolerance,
   {native:this.fields.pressure.diagnostics,fine:root.support,supportWord:9*root.layout.tiles.length+24,status:this.status,band:this.band.index,bandClosedWord:UniformPressureBand.closedWord,acceptance:this.params.acceptance});
  // Displacement runs on a live voxel edit or a body: built when that state is first prepared.
  const needs=this.fields.needs??this.solid?.needs??new UniformPipelineNeeds();
  await Promise.all([this.solid,{initialize:()=>needs.declare(["displace"],async()=>{await this.displacement?.initialize();})},this.transport,this.plan,this.cleanup,this.remap,this.phiResolve,this.coarsePhi,this.extension,this.cache,this.hanging,this.surface,this.surfaceVolume,this.geometry,this.sharpen,this.momentum,this.forces,this.authority,this.projection,this.cycles,this.acceptance,this.split.transfer,this.split.authority,this.pressureSchedule,this.surfaceBand,this.band]
   .map(stage=>stage?.initialize()).concat(this.fields.detail?.prepare(this.ownership,needs)));
  this.ready=true;
 }
 private lastParameters?:UniformMixedFrameParameters;
 /** Parameters of a census extension (encodeExtension) whose plan, phase and
  * extension the next advance may reuse: state is untouched since, and the
  * advance runs with identical parameters. */
 private reusableExtension?:string;
 /** Any state, parameter or ownership change between frames. */
 invalidateExtension():void{this.reusableExtension=undefined;this.geometryCurrent=false;}
 /** A live voxel edit rewrote the host's solid mask: rebuild the all-4h
  * record and the cut widths, and displace liquid out of newly solid h
  * cells at the next frame's head. The host has already promoted every
  * tile the edit touches to h (its CPU relayout). */
 editSolids():void{
  if(!this.solid)throw new Error("This mixed frame was built without solids");
  this.solid.invalidate();this.solidWidthsStale=true;this.invalidateExtension();this.solidEditPending=true;
 }
 /** The simulation target and centre phi were built from the current phi by
  * the last completed advance's gather: nothing has written phi, solids or
  * ownership since. Every such edit clears it with the extension. */
 private geometryCurrent=false;
 /** The parameters (extensionKey) under which the last advance's simulation
  * authority wrote phase on the live layout; undefined when rigid bodies
  * moved after it. While geometryCurrent holds, nothing has written volume,
  * phi, solids or ownership since either (every such edit goes through
  * invalidateExtension), and the split's all-4h authority writes no phase:
  * a head's phase launch would store the same values again. */
 private phaseKey?:string;
 private phaseCurrent(p:UniformMixedFrameParameters):boolean{return this.geometryCurrent&&this.phaseKey===extensionKey(p);}
 /** Re-run the last advance's plan and extension into velocityScratch on
  * the live layout: the field the next advance's surface trace samples.
  * The dynamic census bounds departures from it between frames. */
 encodeExtension(encoder:GPUCommandEncoder):void{
  const p=this.lastParameters;if(!this.ready||this.busy||this.failed||!p)throw new Error("Mixed extension needs a completed advance");
  this.encodeExtensionOf(encoder,p);
 }
 /** phase=false: this advance's simulation authority already wrote phase
  * on this layout and nothing has written volume, centre phi, the target or
  * solids since (the split's all-4h authority writes no phase). */
 private encodeExtensionOf(encoder:GPUCommandEncoder,p:UniformMixedFrameParameters,phase=true):void{
  this.plan.encode(encoder,p.supportPolicy);
  // Phase only: the advance's authority rewrites phi, correction and balance
  // before their readers (band rows, RHS).
  if(phase)this.authority.encode(encoder,this.authorityGroup,false);
  this.extension.encode(encoder,this.extensionGroups,p.extensionSweeps??2);
  this.reusableExtension=extensionKey(p);
 }
 private write(p:UniformMixedFrameParameters):void{
  this.lastParameters=p;
  const h=this.ownership.capacity.lattice.cellSize_m;
  const floats=(b:GPUBuffer,v:number[])=>this.device.queue.writeBuffer(b,0,new Float32Array(v));
  const flags=(b:GPUBuffer,v:number[])=>this.device.queue.writeBuffer(b,16,new Uint32Array(v));
  floats(this.params.extension,[...h,+p.openTop]);floats(this.params.surface,[...h,p.dt]);flags(this.params.surface,[+p.openTop|(p.preserve?2:0)|(p.coarseSurfaceTravel?4:0)|(p.coarseHeldDistance?8:0),+p.cubic,+p.drain,4]);
  floats(this.params.momentum,[...h,p.dt]);flags(this.params.momentum,[+p.openTop,0,0,UNIFORM_MIXED_MOMENTUM_LIMITS]);
  floats(this.params.forces,[...h,p.dt,p.gravity,p.density,p.viscosity,p.surfaceTension,+p.noSlip,+p.openTop,0,p.dust]);
  floats(this.params.authority,[p.dt,p.surfaceDeficitBalancing===true?0:-1,0,p.dust]);floats(this.params.sharpen,[p.sharpeningStrength,p.sharpeningDistance,p.dust,p.orphanDust??0,0,0,0,0]);
  floats(this.params.projection,[...h,p.dt,p.density,+p.openTop,0,p.dust]);floats(this.params.acceptance,[p.dt/p.density,uniformMixedPressureTarget(p),UNIFORM_PRESSURE_RELATIVE_REDUCTION,UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE]);
  floats(this.bandParams,[...h,p.dt,p.density,+p.openTop,p.dt/p.density,Math.min(...h),0,...h.map(x=>Math.fround(1/Math.fround(Math.fround(x)*Math.fround(x))))]);
  const views=this.layoutViews?(this.relayout?UNIFORM_STAGE_VIEWS.previous|UNIFORM_STAGE_VIEWS.reasons|UNIFORM_STAGE_VIEWS.importance:0)|UNIFORM_STAGE_VIEWS.certificate|UNIFORM_STAGE_VIEWS.band:0;
  this.device.queue.writeBuffer(this.presentation.phi.buffer,4*this.stageGridWord,uniformStageGridHeader(this.ownership.capacity.tiles,this.stageBandTiles,views));
 }
 /** Record the layout views (the stage grids' previous, reasons and
  * certificate sections) for the tiles, grid and pressure layers. Off, the
  * frame copies nothing and the census records no reasons. */
 setLayoutViews(enabled:boolean):void{this.layoutViews=enabled;}
 private layoutViews=false;
 /** Copies one view section (a word per tile; importance has two) into the stage grids. */
 private recordStageView(encoder:GPUCommandEncoder,source:{readonly buffer:GPUBuffer;readonly offset:number},section:"previous"|"reasons"|"certificate"|"importance"):void{
  const n=this.ownership.capacity.tiles;
  encoder.copyBufferToBuffer(source.buffer,source.offset,this.presentation.phi.buffer,4*(this.stageGridWord+uniformStageViewWord(n,section)),4*n*(section==="importance"?2:1));
 }
 /** Copies one stage's tile words (or the band bits) into the stage grids. */
 private recordStageGrid(encoder:GPUCommandEncoder,source:GPUBuffer,range:"transport"|"pressure"|"band"):void{
  const n=this.ownership.capacity.tiles,words=range==="band"?Math.ceil(n/32):n;
  const at=this.stageGridWord+UNIFORM_STAGE_GRID_HEADER_WORDS+(range==="transport"?0:range==="pressure"?n:2*n);
  encoder.copyBufferToBuffer(source,0,this.presentation.phi.buffer,4*at,4*words);
 }
 /** Discarded mass uses native sixty-fourths-of-threshold counters, weighted
  * by owner volume. It is a quantized lower bound; counts name owners, not
  * fine cells. Counters are reset once per frame, before transport cleanup. */
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
  * the full envelope to UNIFORM_PRESSURE_KICK_RESIDUAL_PER_STEP, or to the
  * frame's own tolerance where a scene asks for a tighter one; its pressure
  * (Pa: hydrostatic) is presented, the first frame's warm start. The receipt
  * is a frame's, in the same ring and under the same verdict; it plans no
  * frame, so the first two still encode the envelope. bodies: the roster
  * the next frame advances, as solids where they stand. */
 kick(p:UniformMixedFrameParameters,lead:number,bodies=false):Promise<UniformMixedFrameReceipt>{
  if(!(lead>0))throw new Error("The projected kick leads by a positive time");
  return this.advance({...p,dt:lead,viscosity:0,surfaceTension:0,fullPressureEnvelope:true,pressureTolerance:Math.min(uniformMixedPressureTarget(p),2*lead*UNIFORM_PRESSURE_KICK_RESIDUAL_PER_STEP)},undefined,false,bodies?{}:undefined,p.dt);
 }
 /** extendTail: extend this frame's final velocity for the next frame
  * head's census (and reuse it there when the key still matches).
  * Encodes and submits synchronously; the promise is the frame's receipt,
  * checked when its map resolves while later frames encode. A failed receipt
  * rejects naming its frame and fails the frame for every later advance.
  * kick: kick()'s encode, ahead of a frame of this step; it takes no frame index. */
 advance(p:UniformMixedFrameParameters,trace?:UniformMixedFrameTrace,extendTail=false,bodies?:UniformMixedFrameBodies,kick=0):Promise<UniformMixedFrameReceipt>{
  if(bodies&&!this.solid)throw new Error("This mixed frame was built without solids; rigid bodies need the solid library");
  if(!this.ready||this.busy||this.failed)throw new Error("Unified frame is not ready for an advance");
  const readback=this.readbacks.find(b=>!this.unchecked.has(b));
  if(!readback)throw new Error(`Uniform mixed frame has ${this.readbacks.length} unchecked receipts; the host must check one before encoding frame ${this.frameIndex+1}`);
  this.busy=true;const frame=kick?this.frameIndex:++this.frameIndex;
  // The frame a failure latched in this advance is recorded as.
  this.device.queue.writeBuffer(this.status,4*UNIFORM_MIXED_STATUS.currentFrame,new Uint32Array([frame]));
  const releases:(()=>void)[]=[];
  try{
   // Use the same fixed-lag receipt as pressure scheduling, independent of
   // map completion timing. This changes direct launch width only: band
   // kernels stride the current GPU count even when it outruns the budget.
   // A receipt from before a relayout attach counts another owner's layout.
   // A relayout that is not due keeps its generation: this frame's head is the host layout's.
   const relayout=this.relayout?.due?.()===false?undefined:this.relayout;
   // A census after frames without one may admit any tile, as at the attach: the ceilings until its receipt arrives.
   // One that answers a counted request edit moves that many tiles at most: the budgets stay, and
   // carry its count until the receipt of the frame that built it is the one observed.
   const edit=relayout?.edit?.();
   if(edit)this.edits.push({frame,edit});
   else if(relayout&&!this.censused){this.workEpoch=frame-1;this.band.observeWork(this.band.capacity);this.ownership.forgetWork();this.sharpen.forgetWork();this.remap.forgetWork();}
   while(this.edits.length&&this.edits[0]!.frame<=frame-UNIFORM_MIXED_RECEIPT_RING)this.edits.shift();
   const edited=this.edits.length?{tiles:this.edits.reduce((sum,e)=>sum+e.edit.tiles,0),reach:this.edits.reduce((sum,e)=>sum+e.edit.reach,0)}:undefined;
   const work=!kick&&frame-UNIFORM_MIXED_RECEIPT_RING>this.workEpoch?this.lagged.get(frame-UNIFORM_MIXED_RECEIPT_RING):undefined;
   // Under a census the budgets keep a reserve: it admits tiles the receipts report two frames late.
   // A counted edit's build admits none but its own.
   const reserve=relayout&&!edit?UNIFORM_WORK_RELAYOUT_RESERVE:0;
   // The band's tiles are wet h tiles: an edit adds its own at most.
   if(work){this.band.observeWork(work.bandTiles+(edited?.tiles??0),reserve);this.ownership.observeWork(work.work,reserve,edited);this.sharpen.observeWork(work.sharpenWork,reserve,edited?.reach);this.remap.observeWork(work.listed,reserve,edited?.reach);this.cycles.observeWork(work.rootListed);}
   // Only a census certifies residency: a frame without one must not skip a page the last one did.
   if(!kick||relayout){if(!relayout&&this.censused)this.ownership.resetResidency();this.censused=!!relayout;}
   for(const ownership of new Set([this.ownership,...this.levels.map(l=>l.ownership)]))releases.push(ownership.acquireFrame());
   const makeEncoder=()=>{const raw=this.device.createCommandEncoder({label:"Uniform owner-driven frame"});return trace?.instrument(raw)??raw;};
   const resetSurfaceTravel=this.lastParameters!==undefined&&!!this.lastParameters.coarseSurfaceTravel!==!!p.coarseSurfaceTravel;
   this.write(p);let encoder=makeEncoder();
   // The projection alone: at dt 0 the authority's volume correction and deficit balance are zero.
   if(kick)this.device.queue.writeBuffer(this.params.authority,0,new Float32Array([0,-1,0,p.dust]));
   if(resetSurfaceTravel)this.surface.resetTravel(encoder);
   // Bodies moved at the last frame's tail: their cut cells are new, and the
   // census extension no longer matches the solid record or V.
   if(bodies){this.solid!.encodeBodies(encoder);this.solid!.invalidate(true);this.solidWidthsStale=true;this.invalidateExtension();}
   // Built once for static solids (again after an edit, every frame bodies
   // exist), with the tile cut map every solid helper gates on, before the
   // displacement and the band's tile list read them.
   this.solid?.encodeCoarse(encoder);
   // A host layout: a live solid edit, or the cells bodies entered, lands on
   // the live (host-promoted) ownership. A GPU relayout displaces after its
   // adopt, on the generation that promoted them (encodeRelayoutHead).
   if(!relayout&&(this.solidEditPending||bodies)){this.displacement!.encode(encoder,this.fields.volume,!this.solidEditPending);this.solidEditPending=false;}
   // An attached relayout that built nothing this frame changed nothing: the view of the generation before is this one.
   if(!relayout&&this.relayout&&this.layoutViews)this.recordStageView(encoder,{buffer:this.ownership.presentation.buffer,offset:this.ownership.presentation.offset??0},"previous");
   // Submit the frame in segments as it encodes: the GPU starts each one
   // while the host encodes the next, instead of idling through the whole
   // frame's encode. Queue writes made while a later segment encodes land
   // after the earlier segments, which never read them.
   const flush=()=>{trace?.submit(encoder,this.fields.negative);this.device.queue.submit([encoder.finish()]);encoder=makeEncoder();};
   encoder.clearBuffer(this.reductions);
   // A kick's census is its frame's, for that frame's step.
   if(relayout)this.encodeRelayoutHead(encoder,kick?{...p,dt:kick}:p,relayout,bodies!==undefined,trace);
   else{
   // Which cut tiles the simulation holds at h: the all-4h levels read their h texels.
   if(this.solidWidthsStale){this.solid?.encodeSimulation(encoder,this.ownership.presentation);this.solidWidthsStale=false;}
   if(this.solid)trace?.phase(encoder,V.solids);
   this.recordStageGrid(encoder,this.ownership.presentation.buffer,"transport");
   // The census already planned and extended this exact state: geometry and
   // centre phi are the last frame's (phi is unchanged since), phase is the
   // live layout's, and velocityScratch holds the extension.
   // The h phi field survives bulk relayout unchanged.

   const reuse=this.reusableExtension===extensionKey(p);this.reusableExtension=undefined;
   if(kick)this.plan.encode(encoder,p.supportPolicy);else if(!reuse){
    const phase=!this.phaseCurrent(p);
    this.plan.encode(encoder,p.supportPolicy);if(!this.geometryCurrent)this.geometry.encode(encoder,this.geometryGroup);
    // Phase only: the authority below (split: the simulation authority
    // before the band rows, then the pressure authority) rewrites phi, every
    // correction texel and the balance scratch before their readers.
    if(phase)this.authority.encode(encoder,this.authorityGroup,false);
    trace?.phase(encoder,V.support);
    if(!reuse){this.extension.encode(encoder,this.extensionGroups,p.extensionSweeps??2);trace?.phase(encoder,V.extension);}
   }
   }
   // A kick moves no liquid: no trace, transport, surface volume or 4h base.
   if(!kick){
   this.plan.encodeCertificate(encoder,p.dt);if(this.layoutViews)this.recordStageView(encoder,this.plan.certificate,"certificate");
   this.cache.encode(encoder,this.cacheGroup);this.hanging.encode(encoder,this.hangingGroup);
   trace?.phase(encoder,V.transportReach);
   this.surface.encode(encoder,"advect",this.surfaceGroups[0]);this.phiResolve.encode(encoder,this.phiResolveGroups.scratch);this.surface.encode(encoder,"traceCells",this.surfaceGroups[0]);
   if(p.redistance!==false){this.surface.encode(encoder,"redistance",this.surfaceGroups[1]);this.phiResolve.encode(encoder,this.phiResolveGroups.phi);}else this.copyWhole(encoder,this.fields.phiScratch,this.fields.phi);
   trace?.phase(encoder,V.phi);flush();
   this.transport.encodeTransport(encoder);
   // Cleanup and surface correction read the independent h phi field.
   // Transport leaves V (regular floor applied) in volumeScratch: the orphan census or the copy returns it.
   if(p.dust>0)this.cleanup.encode(encoder,this.cleanupGroups[1]);else this.transport.encodeCopy(encoder);
   trace?.phase(encoder,V.coupling);
   // Apply shifts canonical vertices only; resolved readers below and the
   // next advect read the hanging texels.
   if(p.totalSurfaceVolume!==false&&(p.surfaceVolumeRounds??2)>0){this.surfaceVolume.encode(encoder,this.surfaceVolumeGroup,p.surfaceVolumeRounds??2);this.phiResolve.encode(encoder,this.phiResolveGroups.phi);}
   // Nothing after this pass writes phi: the next advance starts from it,
   // and the renderer's 4h vertex base is published from it.
   this.coarsePhi.encode(encoder);
   this.phiResolved=true;
   }

   // Nothing below writes phi before the split reads the pressure geometry.
   this.geometry.encode(encoder,this.geometryGroup,{pressure:true});this.geometryCurrent=true;
   trace?.phase(encoder,V.gather);
   if(!kick&&p.sharpening!==false&&(p.sharpeningSweeps??8)>0&&p.sharpeningDistance>0){this.sharpen.encodeGeometry(encoder,this.sharpenGroups[0]);this.sharpen.encodeSweeps(encoder,this.sharpenGroups,p.sharpeningSweeps??8);}
   trace?.phase(encoder,V.sharpen);
   // A kick forces the stored field where it stands.
   if(kick){this.copyWhole(encoder,this.fields.velocity,this.fields.departure);encoder.copyBufferToBuffer(this.fields.negative,0,this.fields.negativeDeparture,0,this.fields.negative.size);}
   else this.momentum.encode(encoder,this.momentumGroup);
   this.forces.encode(encoder,this.forceGroup,p.surfaceTension>0,p.coarseSurfaceTravel===true);
   trace?.phase(encoder,A.advectionCorrection);flush();
   // Pressure stays all-4h; no layout build and no CPU wait. The band rows
   // need this frame's simulation authority and u*, both rewritten in
   // pressure ownership by the split below.
   // The surface census only feeds the overlay: nothing in the solve reads it.
   if(this.layoutViews){this.surfaceBand.encode(encoder);this.recordStageGrid(encoder,this.surfaceBand.band.buffer,"band");}
   this.authority.encode(encoder,this.authorityGroup);this.phaseKey=bodies?undefined:extensionKey(p);this.band.encodePrepare(encoder);
   const split=this.split;
   split.transfer.encodeToPressure(encoder,split.toPressure);
   split.authority.encode(encoder,split.authorityGroup);
   // Warm start: the root iterate begins at the last frame's presented
   // pressure (fixed all-4h owners; buildRhs zeroes air). From p=0 every
   // frame stopped at the same accepted residual, and a resting pool kept
   // that one smooth divergence: it expanded each step, conservative
   // transport diluted its V into a layer above phi, and the floor's lost
   // volume later surfaced as a growing cavity.
   {const to=this.levels[0]!.pressure;encoder.copyBufferToBuffer(this.presentation.pressure.buffer,0,to.buffer,to.offset??0,4*this.pressureWords);}
   this.projection.encode(encoder,"rhs",split.rhsGroup);this.cycles.encodeSetup(encoder);
   this.cycles.encodeMeasure(encoder);this.acceptance.encode(encoder,this.acceptanceGroup,this.state,"initial");
   trace?.phase(encoder,A.pressureSetup);
   // The conservative schedule is encoded whole; GPU gates run only the
   // slots the last checkpoint and the lagged frame's plan call for. No CPU wait.
   const plannedBy=kick?0:frame-UNIFORM_MIXED_RECEIPT_RING,lagged=this.lagged.get(plannedBy);
   if(plannedBy>0&&!lagged)throw new Error(`Uniform mixed frame ${frame} encoded before frame ${plannedBy}'s receipt was checked`);
   this.lagged.delete(plannedBy);
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
   trace?.phase(encoder,A.pressureProjection);
   split.transfer.encodeToSimulation(encoder,split.toSimulation);
   this.band.encodeSolve(encoder);
   const root=this.levels[0]!;
   // Present the live all-4h words only: level 0 keeps the simulation
   // layout's capacity, but its owners and boundary slots are all-4h.
   for(const [from,to,words] of [[root.pressure,this.presentation.pressure,this.pressureWords],[root.phi!,this.presentation.phi,this.pressureOwnership.layout.cellCount]] as const)encoder.copyBufferToBuffer(from.buffer,from.offset??0,to.buffer,0,4*words);
   this.recordStageGrid(encoder,this.pressureOwnership.presentation.buffer,"pressure");
   // The h band re-solved this frame (empty on an all-4h simulation); the
   // live band pressures were presented by the band's own slot launch.
   encoder.copyBufferToBuffer(this.band.index,this.band.slotMapOffset,this.presentation.phi.buffer,4*this.stageBandWord,4*this.ownership.capacity.tiles);
   trace?.phase(encoder,V.band);
   // Native coupleRigid and the rigid integration read the projected field;
   // the census below promotes around the integrated poses.
   if(bodies?.couple){bodies.couple(encoder);trace?.phase(encoder,A.rigidCoupling);}
   encoder.copyBufferToBuffer(this.state,0,readback,0,32);encoder.copyBufferToBuffer(this.reductions,0,readback,32,48);schedule.encodePlanCopy(encoder,readback,80);
   this.band.encodeReceipt(encoder,readback,88);encoder.copyBufferToBuffer(this.status,0,readback,120,4*UNIFORM_MIXED_STATUS_WORDS);
   // Sticky detail-storage violations of both group-0 ownerships.
   const violation=4*uniformMixedDetailViolationWord(this.ownership.capacity.tiles,this.ownership.capacity.lattice);
   encoder.copyBufferToBuffer(this.ownership.support,violation,readback,DETAIL_RECEIPT,4);encoder.copyBufferToBuffer(this.pressureOwnership.support,violation,readback,DETAIL_RECEIPT+4,4);
   if(this.fields.detail)this.fields.detail.encodeReceipt(encoder,readback,DETAIL_RECEIPT+8);
   this.ownership.encodeWorkReceipt(encoder,readback,WORK_RECEIPT);
   this.sharpen.encodeWorkReceipt(encoder,readback,SHARPEN_WORK_RECEIPT);
   this.remap.encodeWorkReceipt(encoder,readback,REMAP_WORK_RECEIPT);this.cycles.encodeWorkReceipt(encoder,readback,ROOT_WORK_RECEIPT);
   if(PRESSURE_TRACE){const at=TRACE_RECEIPT;schedule.encodeTraceCopy(encoder,readback,at);encoder.copyBufferToBuffer(this.fields.pressure.diagnostics,0,readback,at+128,112);encoder.copyBufferToBuffer(this.band.index,0,readback,at+240,96);}
   // The next head's census and advection read this extension: it is the
   // frame's velocity extension, priced there, not the census's.
   // Phase is this frame's simulation authority's unless rigid bodies moved since.
   if(extendTail){this.encodeExtensionOf(encoder,p,bodies!==undefined);trace?.phase(encoder,V.extension);}
   trace?.submit(encoder,this.fields.negative);this.device.queue.submit([encoder.finish()]);trace?.submitted();
   this.unchecked.add(readback);
   return this.check(readback,frame,p,plan.vCycles+plan.fullCycles,kick>0);
  }catch(error){trace?.abort();this.failed=true;throw error;}finally{for(const release of releases)release();this.busy=false;}
 }
 private traceCoarse=[0,0];
 private tracePressure(frame:number,w:Uint32Array):void{
  const f=new Float32Array(w.buffer,w.byteOffset,w.length),c=f.subarray(0,32),native=w.subarray(32,60),band=w.subarray(60,84),bf=f.subarray(60,84);
  const slots=w[12]!+w[13]!,ran=w[16]!+w[17]!,accuracy=(i:number)=>["1","0.1","0"][(w[25]!>>(2*i))&3]??"?";
  const iterations=native[26]!-this.traceCoarse[0]!,solves=native[27]!-this.traceCoarse[1]!;this.traceCoarse=[native[26]!,native[27]!];
  console.log(`PTRACE ${JSON.stringify({frame,plan:[w[12],w[13]],ran:[w[16],w[17]],initial:c[8],slots:Array.from({length:ran},(_,i)=>[c[18+i],accuracy(i)]),stalled:w[7],vStalled:w[6],planned:c[9],next:[w[14],w[15]],
   coarse:{solves,iterations},band:{tiles:band[0],cycles:band[4],final:bf[2],history:Array.from(bf.subarray(8,8+6))},slotsEncoded:slots})}`);
 }
 /** Frame `frame`'s receipt: fail fast on a rejected or unconverged solve. */
 private async check(readback:GPUBuffer,frame:number,p:UniformMixedFrameParameters,encoded:number,kick=false):Promise<UniformMixedFrameReceipt>{
  let mapped:Uint32Array;
  try{
   await readback.mapAsync(GPUMapMode.READ);
   mapped=new Uint32Array(readback.getMappedRange(),0,TRACE_RECEIPT/4+(PRESSURE_TRACE?84:0)).slice();readback.unmap();
   if(PRESSURE_TRACE)this.tracePressure(frame,mapped.slice(TRACE_RECEIPT/4));
  }catch(error){this.failed=true;throw error;}finally{this.unchecked.delete(readback);}
  try{
   const state=mapped.slice(0,8),accounting=mapped.slice(8,20);
   // The GPU's own verdict first: it names the first failing frame.
   const failure=describeUniformMixedFrameStatus(mapped.slice(30,30+UNIFORM_MIXED_STATUS_WORDS));
   // A relayout that outran the patch slots explains every later failure.
   this.fields.detail?.noteReceipt(mapped.subarray(DETAIL_RECEIPT/4+2,DETAIL_RECEIPT/4+DETAIL_RECEIPT_WORDS));
   const detail=mapped[DETAIL_RECEIPT/4]!|mapped[DETAIL_RECEIPT/4+1]!;
   if(detail)throw new Error(`Uniform detail storage: a store a non-resident tile cannot hold (${Object.entries(UNIFORM_DETAIL_VIOLATION).filter(([,bit])=>detail&bit).map(([kind])=>kind).join(", ")} field; simulation ${mapped[DETAIL_RECEIPT/4]}, pressure ${mapped[DETAIL_RECEIPT/4+1]})${failure?`; ${failure}`:""}`);
   if(failure)throw new Error(`${failure}${mapped[30]===UNIFORM_MIXED_FAILURE.layoutCapacity&&mapped[33]!&UNIFORM_MIXED_OVERFLOW_FINE?`; the layout holds ${mapped[WORK_RECEIPT/4]} h tiles, capacity ${this.ownership.capacity.fineTiles}`:""}; projection withheld${typeof process!=="undefined"&&process.env.FLUID_MIXED_HOST_DIAGNOSTICS?`; params ${JSON.stringify(p)}; receipt ${[...state]}`:""}`);
   const residual=new Float32Array(state.buffer)[1]!,count=state[7]!;
   if(state[4]!==0||state[5]===0||!Number.isFinite(residual)||residual<0||residual>uniformMixedPressureTarget(p)){
    throw new Error(`Uniform mixed pressure ${state[4]!==0?"rejected a non-improving cycle":"did not converge"}: candidate ${new Float32Array(state.buffer)[0]}, accepted ${residual}, tolerance ${uniformMixedPressureTarget(p)}, ${count} cycles; projection withheld${typeof process!=="undefined"&&process.env.FLUID_MIXED_HOST_DIAGNOSTICS?`; params ${JSON.stringify(p)}; receipt ${[...state]}`:""}`);
   }
   if(mapped[23]!==0)throw new Error(`Uniform pressure band needs ${mapped[22]} tiles, over its capacity; projection is incomplete`);
   this.bandTiles=mapped[22]!;this.bandResidual=new Float32Array(mapped.buffer)[24]!;
   // Only an accepted solve plans; a failed one threw above. Frame
   // frame+UNIFORM_MIXED_RECEIPT_RING encodes it, whenever this resolves.
   if(!kick)this.lagged.set(frame,{vCycles:mapped[20]!,fullCycles:mapped[21]!,bandTiles:mapped[22]!,work:mapped.slice(WORK_RECEIPT/4,SHARPEN_WORK_RECEIPT/4),sharpenWork:mapped.slice(SHARPEN_WORK_RECEIPT/4,REMAP_WORK_RECEIPT/4),listed:mapped[REMAP_WORK_RECEIPT/4]!,rootListed:mapped[ROOT_WORK_RECEIPT/4]!});
   const cells=this.ownership.capacity.lattice.dimensions.reduce((n,d)=>n*d,1);
   const orphanDustMass_cells=uniformMixedDustMass(accounting,10,p.orphanDust??0,cells);
   return {cycles:count,encoded,residual,converged:true,
    dustOwners:accounting[5]!+accounting[10]!,dustMass_cells:uniformMixedDustMass(accounting,5,p.dust,cells)+orphanDustMass_cells,
    orphanDustOwners:accounting[10]!,orphanDustMass_cells,bandTiles:mapped[22]!,bandCycles:mapped[26]!,bandResidual:this.bandResidual};
  }catch(error){this.failed=true;throw new Error(`Uniform mixed ${kick?`kick before frame ${frame+1}`:`frame ${frame}`}: ${error instanceof Error?error.message:String(error)}`,{cause:error});}
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
  const refusal=this.fields.detail?.refusal(layout)??this.reserveFine(layout.fineTiles.length);if(refusal)return refusal;
  // An authored refinement is immediate evidence of possible new band work;
  // reserve launch parallelism before its first frame, ahead of the lagged
  // liquid-band receipts. The actual band may occupy fewer of these tiles.
  this.band.observeWork(layout.fineTiles.length);
  this.reusableExtension=undefined;this.geometryCurrent=false;this.solidWidthsStale=true;
  // The remap reads the all-4h solid record (umTileOpen): a relayout ahead of
  // the first frame builds it. A pending live edit keeps the record of the
  // solids the GPU still holds (the host publishes the mask after this).
  if(this.solid?.recordStale&&!this.solidEditPending){
   const e=this.device.createCommandEncoder({label:"Uniform solid record before a host relayout"});this.solid.encodeCoarse(e);this.device.queue.submit([e.finish()]);
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
 /** Request edits (UniformMixedFrameRelayout.edit) built in frames whose receipts no advance has observed yet. */
 private readonly edits:{frame:number;edit:UniformWorkEdit}[]=[];
 /** The frame a relayout was attached after: its receipts and older ones
  * describe the host layout it replaced. */
 private workEpoch=0;
 /** reserved: the h tiles the relayout starts with storage for (a request's
  * host count, or the capacity already held); absent, every tile, with the
  * band at its liquid bound (a census's first build may ask any tile).
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
  // tile, the band at its liquid bound, unless the host reserved for its
  // requests. Detached, host layouts reserve their own again (the next
  // updateLayout).
  if(relayout){
   const refusal=reserved===undefined?this.reserveFine(this.ownership.capacity.tiles,true):this.reserveFine(reserved,false);
   if(refusal)throw new Error(`A GPU relayout holds ${reserved===undefined?"every tile":`${reserved} tiles`} at h: ${refusal}`);
  }else{this.liquidBand=false;if(this.censused){this.ownership.resetResidency();this.censused=false;}}
  this.relayout=relayout;
  // The census may admit any tile in its first frame and its counts reach
  // the host UNIFORM_MIXED_RECEIPT_RING frames late: launch at the ceilings
  // until a receipt of this owner arrives. A budget left at the host
  // layout's (one group from all-4h) runs the new h tiles on a few lanes.
  if(relayout&&!counted){this.workEpoch=this.frameIndex;this.band.observeWork(this.band.capacity);this.ownership.forgetWork();this.sharpen.forgetWork();this.remap.forgetWork();}
  // The GPU census admits patches with no host round trip: every patch has a slot.
  if(relayout)this.fields.detail?.reserveAll();
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
  *  4. phi resolve, displacement (solid edits, bodies' new cells) and the
  *     cut widths on the adopted generation;
  *  5. plan, gather and phase on it: they belong to the ownership, not the
  *     field, so they are rebuilt every frame (the host no longer knows
  *     whether the generation changed).
  * The certificate, momentum cache and hanging taps that follow read the
  * remapped extension. The classifier and the trace sample the same field:
  * the census bounds each tile's departures over all its owners' faces of
  * that extension on both sides of the relayout (it joins the trace's
  * sampled flow with zero), and the remap writes each changed face as a
  * convex combination of old faces of owners in the same tile (remapFace
  * averages remapSample, which interpolates two old faces); an unlisted
  * tile's faces are the identity. So the remapped extension stays inside
  * the census bounds the generation was built for, and no second extension
  * runs before advection. */
 private encodeRelayoutHead(encoder:GPUCommandEncoder,p:UniformMixedFrameParameters,relayout:UniformMixedFrameRelayout,bodies:boolean,trace?:UniformMixedFrameTrace):void{
  const reuse=this.reusableExtension===extensionKey(p);this.reusableExtension=undefined;
  // In the frame's first blit run; the remap's markListed publishes the
  // receipt's generation into the status record (a later failure names it).
  this.remap.encodeClear(encoder);
  if(!reuse){
   if(this.solidWidthsStale){this.solid?.encodeSimulation(encoder,this.ownership.presentation);this.solidWidthsStale=false;}
   const phase=!this.phaseCurrent(p);
   this.plan.encode(encoder,p.supportPolicy);if(!this.geometryCurrent)this.geometry.encode(encoder,this.geometryGroup);
   if(phase)this.authority.encode(encoder,this.authorityGroup,false);
   this.extension.encode(encoder,this.extensionGroups,p.extensionSweeps??2);trace?.phase(encoder,V.extension);
  }
  // The tiles as the frame starts, for the grid layer's relayout marks.
  if(this.layoutViews)this.recordStageView(encoder,{buffer:this.ownership.presentation.buffer,offset:this.ownership.presentation.offset??0},"previous");
  relayout.encode(encoder,p.dt,this.layoutViews);
  if(this.layoutViews){this.recordStageView(encoder,relayout.reasons,"reasons");this.recordStageView(encoder,relayout.importance,"importance");}
  // Detail storage admits the generation's patches ahead of the remap.
  this.fields.detail?.encodeAdmit(encoder,relayout.generation.topology);
  this.remap.applyGpu(encoder,relayout.generation,relayout.receipt,true,relayout.adopted?.bind(relayout));
  this.geometryCurrent=false;
  trace?.phase(encoder,A.resolutionCensus);
  // Only tiles whose neighbourhood changed width can hang differently.
  const changes=relayout.generation.changes;
  this.phiResolve.encode(encoder,this.phiResolveGroups.phi,this.phiResolved?changes:undefined);
  // The patches the generation no longer needs return to the base.
  this.fields.detail?.encodeRetire(encoder);
  this.fields.detail?.encodeSettle(encoder,false);
  // Only the relayout moved phi since the last geometry unless solids moved.
  const solidsMoved=this.solidEditPending||bodies;
  if(this.solidEditPending||bodies){this.displacement!.encode(encoder,this.fields.volume,!this.solidEditPending);this.solidEditPending=false;}
  // Which cut tiles the simulation holds at h: the all-4h levels read their h texels.
  this.solid?.encodeSimulation(encoder,this.ownership.presentation,this.solidWidthsStale?undefined:changes);this.solidWidthsStale=false;
  if(this.solid)trace?.phase(encoder,V.solids);
  this.recordStageGrid(encoder,this.ownership.presentation.buffer,"transport");
  this.plan.encode(encoder,p.supportPolicy);this.geometry.encode(encoder,this.geometryGroup,{changed:solidsMoved?undefined:changes});
  this.authority.encode(encoder,this.authorityGroup,false);
  trace?.phase(encoder,V.support);
 }
 /** Publishes the 4h vertex base from the live phi: t=0 and any host write
  * of phi outside an advance or a relayout. */
 publishCoarseVertexPhi():void{
  if(!this.ready||this.busy)throw new Error("The 4h vertex phi base publishes between frames");
  if(this.coarsePhi.adopted)return;
  const e=this.device.createCommandEncoder({label:"Uniform publish 4h vertex phi base"});this.coarsePhi.encode(e);this.device.queue.submit([e.finish()]);
 }
 destroy():void{this.coarsePhi.destroy();this.solid?.destroy();this.displacement?.destroy();this.pressureSchedule?.destroy();this.surfaceBand.destroy();this.band.destroy();this.plan.destroy();this.surface.destroy();this.geometry.destroy();this.momentum.destroy();this.hanging.destroy();this.remap.destroy();this.transport.destroy();this.cleanup.destroy();this.split.transfer.destroy();for(const l of this.levels)if(l.ownership!==this.ownership)l.ownership.destroy();for(const r of this.owned)r.destroy();}
}
