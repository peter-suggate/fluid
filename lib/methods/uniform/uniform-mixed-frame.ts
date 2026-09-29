import {UniformPressureSurfaceBand} from "./uniform-pressure-surface-band";
import {UniformPressureBand} from "./uniform-pressure-band";
import {uniformMixedDustMass} from "./uniform-mixed-dust-accounting.wgsl";
import {UniformMixedCleanup} from "./uniform-mixed-cleanup";
import {UniformMixedFramePlan} from "./uniform-mixed-frame-plan";
import {UniformMixedOwnershipTransfer,UniformMixedRemap} from "./uniform-mixed-remap";
import {UniformMixedPhiResolve} from "./uniform-mixed-phi-resolve";
import {uniformMixedAllCoarseLayout} from "./uniform-mixed-layout";
import type {UniformMixedLayout} from "./uniform-mixed-layout";
import {UNIFORM_MIXED_RELAYOUT_RECEIPT} from "./uniform-mixed-layout-builder";
import type {UniformScratchArena} from "./uniform-scratch-arena";
import type {WebGPUUniformPressureMultigrid} from "./webgpu-uniform-pressure-multigrid";
import type {WebGPUUniformVelocityExtrapolator} from "./webgpu-uniform-velocity-extrapolation";
import {UniformMixedTransportStage} from "./uniform-mixed-transport";
import {UniformMixedOwnership,type UniformMixedGenerationBuffers} from "./uniform-mixed-ownership";
import {UniformMixedExtension} from "./uniform-mixed-extension";
import {UniformMixedHangingTaps, UniformMixedMomentumCache} from "./uniform-mixed-momentum-cache";
import {UniformMixedSurface} from "./uniform-mixed-surface";
import {UniformMixedSurfaceVolume} from "./uniform-mixed-surface-volume";
import {UniformMixedSurfaceGeometry} from "./uniform-mixed-surface-geometry";
import {UniformMixedSharpening} from "./uniform-mixed-sharpening";
import {UniformMixedMomentum,UNIFORM_MIXED_MOMENTUM_LIMITS} from "./uniform-mixed-momentum";
import {UniformMixedForces} from "./uniform-mixed-forces";
import {UniformMixedPressureAuthority} from "./uniform-mixed-pressure-authority";
import {uniformMixedPressureStorage} from "./uniform-mixed-pressure-boundary.wgsl";
import {UniformMixedPressureVelocity} from "./uniform-mixed-pressure-velocity";
import {UniformMixedPressureCycles,type UniformMixedPressureCycleLevel} from "./uniform-mixed-pressure-cycles";
import {UniformMixedPressureAcceptance} from "./uniform-mixed-pressure-acceptance";
import {UniformMixedPressureSchedule,type UniformMixedPressurePlan} from "./uniform-mixed-pressure-schedule";
import {UNIFORM_MIXED_STATUS,UNIFORM_MIXED_STATUS_WORDS,describeUniformMixedFrameStatus} from "./uniform-mixed-frame-status";
import {planUniformMixedPressureMemory} from "./uniform-mixed-pressure-memory";
import {DEFAULT_UNIFORM_CM11A_SCHEDULE,UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE,UNIFORM_PRESSURE_RELATIVE_REDUCTION,type UniformCM11aSchedule} from "./pressure-policy";
import type {GPUTimestampPhase} from "../../core/performance-trace";
import {UNIFORM_ADVANCE_PHASE as A} from "./uniform-stages";
import {UNIFORM_VOLUME_PHASE as V} from "./uniform-volume-stages";
import {UniformMixedSolid,type UniformMixedSolidResources} from "./uniform-mixed-solid.wgsl";
import {UniformMixedSolidDisplacement} from "./uniform-mixed-solid-displacement";
import {UNIFORM_STAGE_GRID_HEADER_WORDS,UNIFORM_STAGE_VIEWS,uniformStageBandWords,uniformStageGridHeader,uniformStageGridWords,uniformStageViewWord} from "./uniform-stage-grids";

export interface UniformMixedFrameTrace {
 instrument(encoder:GPUCommandEncoder):GPUCommandEncoder;
 phase(encoder:GPUCommandEncoder,phase:GPUTimestampPhase):void;
 submit(encoder:GPUCommandEncoder,anchor:GPUBuffer):void;
 submitted():void;
 abort():void;
}

export interface UniformMixedFrameFields {
 arena:UniformScratchArena;conditioning:GPUBuffer;
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
 /** Two-stage pressure: an all-4h solve, then the h surface band
  * (UniformPressureBand). With solids, the all-4h levels read the static
  * coarse solid record and every liquid solid-coupled tile joins the band.
  * The solve always runs split, an all-4h simulation too (its transfer is the
  * identity and its band empty): in the all-4h ownership with its own surface
  * target and centre phi here, the simulation's own staying intact for the
  * renderer. */
 pressureGeometry:{target:GPUTexture;centerPhi:GPUTexture};
}
export interface UniformMixedFrameParameters {
 dt:number;gravity:number;density:number;viscosity:number;surfaceTension:number;
 openTop:boolean;noSlip:boolean;cubic:boolean;drain:boolean;
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
 /** The built generation's buffers (UniformMixedLayoutBuilder.generation). */
 readonly generation:UniformMixedGenerationBuffers;
 /** The builder's relayout receipt (UNIFORM_MIXED_RELAYOUT_RECEIPT). */
 readonly receipt:{readonly buffer:GPUBuffer;readonly offset:number;readonly words:number};
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

export class UniformMixedFrame {
 readonly transport:UniformMixedTransportStage;
 get ownership(){return this.transport.ownership;}
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
 private readonly stageGridWord:number;
 private readonly stageBandWord:number;
 private readonly stageBandTiles:number;
 private readonly split:{
  transfer:UniformMixedOwnershipTransfer;toPressure:GPUBindGroup;toSimulation:GPUBindGroup;
  authority:UniformMixedPressureAuthority;authorityGroup:GPUBindGroup;
  rhsGroup:GPUBindGroup;projectionGroup:GPUBindGroup;
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
 /** Band pressure with solids: level 0's topology view (the all-4h record). */
 private readonly solidTopology?:GPUBufferBinding;
 private readonly owned:(GPUTexture|GPUBuffer)[]=[];
 private readonly plan:UniformMixedFramePlan;
 private readonly cleanup:UniformMixedCleanup;
 private readonly cleanupGroups:readonly [GPUBindGroup,GPUBindGroup];
 private readonly remap:UniformMixedRemap;
 /** The level set lives at owner resolution: canonical writers (remap, advect,
  * redistance) leave hanging texels stale, and every reader is compiled with
  * the resolved sampler, so each writer is followed by a resolve. */
 private readonly phiResolve:UniformMixedPhiResolve;
 private readonly phiResolveGroups:{phi:GPUBindGroup;scratch:GPUBindGroup};
 private readonly extension:UniformMixedExtension;
 private readonly cache:UniformMixedMomentumCache;
 private readonly hanging:UniformMixedHangingTaps;
 private readonly hangingGroup:GPUBindGroup;
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
 private readonly extensionGroups:readonly [GPUBindGroup,GPUBindGroup];
 private readonly cacheGroup:GPUBindGroup;
 private readonly surfaceGroups:readonly [GPUBindGroup,GPUBindGroup];
 private readonly surfaceVolumeGroup:GPUBindGroup;
 private readonly geometryGroup:GPUBindGroup;
 private readonly sharpenGroups:readonly [GPUBindGroup,GPUBindGroup];
 private readonly momentumGroup:GPUBindGroup;
 private readonly forceGroup:GPUBindGroup;
 private readonly authorityGroup:GPUBindGroup;
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
 private readonly lagged=new Map<number,UniformMixedPressurePlan>();
 private readonly reductions:GPUBuffer;
 private pressureSchedule!:UniformMixedPressureSchedule;
 /** The slot list of the first UNIFORM_MIXED_RECEIPT_RING frames: conservative. */
 private readonly initialPlan:UniformMixedPressurePlan;
 /** Surface-crossing tiles of the current phi: the stage grids' band bits. */
 private readonly surfaceBand:UniformPressureSurfaceBand;
 private ready=false;
 private busy=false;
 private failed=false;
 get allocatedBytes():number{return (this.displacement?.allocatedBytes??0)+this.surfaceBand.allocatedBytes+this.band.allocatedBytes+this.plan.allocatedBytes+(this.solid?.allocatedBytes??0)+this.surface.allocatedBytes+this.hanging.allocatedBytes+this.transport.allocatedBytes+this.remap.allocatedBytes+this.split.transfer.allocatedBytes+this.levels.filter(l=>l.ownership!==this.ownership).reduce((n,l)=>n+l.ownership.allocatedBytes,0)+this.owned.reduce((n,r)=>n+("size" in r?r.size:r.width*r.height*r.depthOrArrayLayers*16),0);}
 constructor(private readonly device:GPUDevice,layout:UniformMixedLayout,private readonly fields:UniformMixedFrameFields,openTop:boolean,private readonly schedule:UniformCM11aSchedule=DEFAULT_UNIFORM_CM11A_SCHEDULE){
  if(layout.cellCount!==layout.tiles.length*64)throw new Error("Unified frame must reserve its capacity with fine ownership");
  const f=fields;
  const buffer=(label:string,size:number,usage:number)=>{const b=device.createBuffer({label,size,usage});this.owned.push(b);return b;};
  const uniform=(name:string,size:number)=>buffer(`Uniform ${name} parameters`,size,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
  this.params={extension:uniform("extension",16),surface:uniform("surface",32),momentum:uniform("momentum",32),forces:uniform("forces",48),authority:uniform("authority",16),sharpen:uniform("sharpen",32),projection:uniform("projection",32),acceptance:uniform("acceptance",16)};
  this.state=buffer("Uniform pressure acceptance",32,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST);
  this.status=buffer("Uniform mixed frame status",4*UNIFORM_MIXED_STATUS_WORDS,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST);
  this.initialPlan={vCycles:schedule.vCycles,fullCycles:schedule.fullCycles};
  this.readbacks=Array.from({length:UNIFORM_MIXED_RECEIPT_RING},(_,i)=>buffer(`Uniform pressure receipt and mass accounting ${i}`,120+4*UNIFORM_MIXED_STATUS_WORDS,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ));
  this.reductions=buffer("Uniform dust accounting",48,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC);
  const caches=Array.from({length:1},(_,i)=>{const t=device.createTexture({label:`Uniform 4h sampling cache ${i}`,size:layout.lattice.dimensions.map(n=>n/4+2),dimension:"3d",format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING});this.owned.push(t);return t;});
  const coarseLayout=uniformMixedAllCoarseLayout(layout);
  const solid=this.solid=f.solid?new UniformMixedSolid(device,f.solid,coarseLayout):undefined;
  this.transport=new UniformMixedTransportStage(device,layout,f.arena,f.volume,f.volumeScratch,f.departure,f.sourceParams,solid,true);
  const o=this.ownership;
  this.displacement=solid?new UniformMixedSolidDisplacement(device,o,solid):undefined;
  this.plan=new UniformMixedFramePlan(device,o,f.volume,f.phi,f.velocity,f.negative,f.velocityScratch,f.negativeScratch,true);
  this.cleanup=new UniformMixedCleanup(device,o,solid,true);
  this.cleanupGroups=[this.cleanup.bind(f.volume,f.volumeScratch,f.phi,this.params.sharpen,this.reductions),this.cleanup.bind(f.volumeScratch,f.volume,f.phi,this.params.sharpen,this.reductions)];
  this.remap=new UniformMixedRemap(device,o,{volume:f.volume,velocity:f.velocity,phi:f.phi,negative:f.negative},{volume:f.volumeScratch,velocity:f.velocityScratch,phi:f.phiScratch,negative:f.negativeScratch});
  // The census extension crosses a relayout in these instead of being rebuilt.
  {const v=f.velocityScratch,t=device.createTexture({label:"Uniform mixed remapped extension",size:[v.width,v.height,v.depthOrArrayLayers],dimension:"3d",format:v.format,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING});this.owned.push(t);
   this.remap.bindExtension({volume:f.volume,velocity:f.velocity,phi:f.phi,negative:f.negative},{velocity:t,negative:buffer("Uniform mixed remapped extension walls",f.negativeScratch.size,GPUBufferUsage.STORAGE)});}
  this.phiResolve=new UniformMixedPhiResolve(device,o);this.phiResolveGroups={phi:this.phiResolve.bind(f.phi),scratch:this.phiResolve.bind(f.phiScratch)};
  const prefix=Math.min(...[f.pressure.pressure,f.pressure.rhs,f.pressure.minimum,f.pressure.phi,f.pressure.topology].map(v=>v.buffer!.offset??0));
  const memory=planUniformMixedPressureMemory(layout,prefix,f.conditioning.size);
  const view=(r:{offset:number;size:number},external=false):GPUBufferBinding=>({buffer:external?f.conditioning:f.arena.buffer,...r});
  // Pressure's all-4h root keeps the mixed rows; the native hierarchy
  // solves its correction from its n/4 level.
  // With solids, the root's topology is the static all-4h solid record,
  // read in place: rebuilt only when the solids change, never copied.
  if(solid?.coarse)this.solidTopology={buffer:solid.coarse.record,size:16*solid.coarse.count};
  {const r=memory.root;
   this.levels=[{ownership:new UniformMixedOwnership(device,coarseLayout,false),pressure:view(r.pressure),rhs:[view(r.rhs[0]),view(r.rhs[1])],minimum:[view(r.minimum[0]),view(r.minimum[1])],phi:view(r.phi,true),frozen:view(r.frozen),residual:view(r.residual),
    topology:f.solid?{buffer:this.solidTopology!}:undefined}];}
  const root=this.levels[0]!,p=root.ownership;
  // The h near-surface extension carries the advancing level-set toe.
  // A regular-only hierarchy fails uniform-long-dam-front-dawn.test.ts.
  this.extension=new UniformMixedExtension(device,o,f.extension,false,true);
  this.extensionGroups=this.extension.bind({physical:f.velocity,phase:f.phase,negative:f.negative,output:f.velocityScratch,outputNegative:f.negativeScratch,scratch:{buffer:f.arena.buffer},params:this.params.extension});
  this.cache=new UniformMixedMomentumCache(device,o);
  const cacheFields={coarseExtended:caches[0]!};
  this.cacheGroup=this.cache.bind({extended:f.velocityScratch,negative:f.negativeScratch,...cacheFields});
  // Surface and momentum share hanging taps of velocityScratch: nothing
  // between the cache and forces writes it, negativeScratch or cache 0.
  this.hanging=new UniformMixedHangingTaps(device,o);this.hangingGroup=this.hanging.bind({extended:f.velocityScratch,negative:f.negativeScratch,coarse:caches[0]!});
  this.surface=new UniformMixedSurface(device,o,f.sourceParams,solid,true,true);
  const surfaceFields={unitVelocity:this.hanging.unitVelocity,velocity:f.velocityScratch,coarseVelocity:caches[0]!,volume:f.volume,negative:f.negativeScratch,departures:f.departure,params:this.params.surface,evidence:{buffer:f.arena.buffer}};
  this.surfaceGroups=[this.surface.bind({...surfaceFields,phi:f.phi,outputPhi:f.phiScratch}),this.surface.bind({...surfaceFields,phi:f.phiScratch,outputPhi:f.phi})];
  this.surfaceVolume=new UniformMixedSurfaceVolume(device,o,solid,true);
  this.surfaceVolumeGroup=this.surfaceVolume.bind(f.phi,f.volume,f.phi,{buffer:f.arena.buffer});
  // The full launch after surface volume also writes split pressure's all-4h geometry.
  this.geometry=new UniformMixedSurfaceGeometry(device,o,solid,true,true);this.geometryGroup=this.geometry.bind(f.phi,f.target,f.centerPhi,f.pressureGeometry);
  this.sharpen=new UniformMixedSharpening(device,o,solid,{list:buffer("Uniform mixed sharpening tile list",UniformMixedSharpening.workBytes(layout.tiles.length),GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST)},true);
  this.sharpenGroups=[this.sharpen.bind(f.volume,f.volumeScratch,f.phi,f.target,f.centerPhi,{buffer:f.arena.buffer,offset:0,size:f.arena.edgeBytes},this.params.sharpen,this.reductions),this.sharpen.bind(f.volumeScratch,f.volume,f.phi,f.target,f.centerPhi,{buffer:f.arena.buffer,offset:0,size:f.arena.edgeBytes},this.params.sharpen,this.reductions)];
  // Retain the projected h velocity detail: restricting all momentum to 4h
  // removes the long-dam toe even when total liquid volume is conserved.
  this.momentum=new UniformMixedMomentum(device,o,true,true,f.sourceParams);
  this.momentumGroup=this.momentum.bind({unitVelocity:this.hanging.unitVelocity,extended:f.velocityScratch,physical:f.velocity,phase:f.phase,volume:f.volume,centerPhi:f.centerPhi,negative:f.negativeScratch,output:f.departure,outputNegative:f.negativeDeparture,params:this.params.momentum,...cacheFields});
  // Viscosity reads exact MAC sites from the momentum fill (the extended
  // velocity): h sites in the unit texture, 4h sites in the coarse cache.
  this.forces=new UniformMixedForces(device,o,true,f.sourceParams,solid);this.forceGroup=this.forces.bind({unitVelocity:this.hanging.unitVelocity,advected:f.departure,phi:f.phi,volume:f.volume,centerPhi:f.centerPhi,coarseVelocity:caches[0]!,negative:f.negativeDeparture,output:f.velocityScratch,outputNegative:f.negativeScratch,params:this.params.forces});
  this.authority=new UniformMixedPressureAuthority(device,o,solid);this.authorityGroup=this.authority.bind({centerPhi:f.centerPhi,volume:f.volume,targetFill:f.target,phi:root.phi!,phase:f.phase,correction:f.correction,scratch:root.frozen,params:this.params.authority});
  // Pressure couples the all-4h owners through the static coarse record.
  const coarseSolid=!!solid;
  this.projection=new UniformMixedPressureVelocity(device,p,f.sourceParams,solid,coarseSolid);
  // A cut 4h face's flux sums its h faces of the forced field, which the split
  // leaves in simulation ownership in the scratch pair.
  const fine=coarseSolid?{velocity:f.velocityScratch,negative:f.negativeScratch}:undefined;
  const bindProjection=(input:{velocity:GPUTexture;negative:GPUBuffer},output:{velocity:GPUTexture;negative:GPUBuffer},centerPhi:GPUTexture,volume:GPUTexture,fine?:{velocity:GPUTexture;negative:GPUBuffer})=>{
   const common={velocity:input.velocity,negative:{buffer:input.negative},phi:root.phi!,params:this.params.projection};
   return [this.projection.bindRhs({...common,correction:f.correction,rhs:root.rhs[0],minimum:root.minimum![0]!,pressure:root.pressure,fine}),
    this.projection.bindProjection({...common,pressure:root.pressure,centerPhi,volume,output:output.velocity,outputNegative:{buffer:output.negative}})] as const;
  };
  const g=f.pressureGeometry;
  {
   // Split: the forced field reaches pressure ownership in velocity/negative
   // and is projected into the scratch pair, then transferred back. Volume
   // reaches pressure ownership in its (free) scratch field.
   const [rhsGroup,projectionGroup]=bindProjection({velocity:f.velocity,negative:f.negative},{velocity:f.velocityScratch,negative:f.negativeScratch},g.centerPhi,f.volumeScratch,fine);
   const transfer=new UniformMixedOwnershipTransfer(device,o,p);
   const present=(label:string,bytes:number)=>buffer(`Uniform presented ${label}`,bytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
   // The band's capacity never exceeds the tile count (its all-tile
   // diagnostic included), so the record holds any band this frame builds.
   const n=layout.tiles.length;this.stageBandTiles=n;
   // Presented pressure and phi hold the live all-4h words only (level 0
   // keeps the simulation layout's capacity).
   this.stageBandWord=p.layout.cellCount;this.stageGridWord=this.stageBandWord+uniformStageBandWords(n,this.stageBandTiles);
   this.presentation={pressure:{buffer:present("pressure",4*uniformMixedPressureStorage(p.layout).count)},phi:{buffer:present("pressure phi and stage grids",4*(this.stageGridWord+uniformStageGridWords(n)))}};
   const authority=new UniformMixedPressureAuthority(device,p,solid,coarseSolid,true);
   this.split={transfer,rhsGroup,projectionGroup,
    toPressure:transfer.bind({volume:f.volume,velocity:f.velocityScratch,negative:f.negativeScratch},{volume:f.volumeScratch,velocity:f.velocity,negative:f.negative},this.status),
    toSimulation:transfer.bind({volume:f.volume,velocity:f.velocityScratch,negative:f.negativeScratch},{volume:f.volumeScratch,velocity:f.velocity,negative:f.negative},this.status),
    authority,authorityGroup:authority.bind({centerPhi:g.centerPhi,volume:f.volumeScratch,targetFill:g.target,phi:root.phi!,phase:f.phase,correction:f.correction,scratch:root.frozen,params:this.params.authority,
     fine:coarseSolid?{centerPhi:f.centerPhi,volume:f.volume}:undefined})};
  }
  this.bandParams=uniform("pressure band",48);
  this.surfaceBand=new UniformPressureSurfaceBand(device,o,f.phi);
  this.band=new UniformPressureBand(device,o,p,{phi:root.phi!,correction:f.correction,vertexPhi:f.phi,
   forced:{velocity:f.velocityScratch,negative:f.negativeScratch},velocity:f.velocity,negative:f.negative,copy:f.velocityScratch,
   coarsePressure:root.pressure,params:this.bandParams,presentation:{buffer:this.presentation.phi.buffer,word:this.stageBandWord+layout.tiles.length}},undefined,solid);
  if(this.band.capacity!==this.stageBandTiles)throw new Error("The stage grids' band section must match the band capacity");
  // Setup is encoded once per solve, ungated (advance); cycles are pure cycle work.
  this.cycles=new UniformMixedPressureCycles(device,root,view(memory.backup),f.pressure,f.uniformGroup,openTop);
  this.acceptance=new UniformMixedPressureAcceptance(device,p);
  this.acceptanceGroup=this.acceptance.bind({residual:root.residual,state:this.state,params:this.params.acceptance});
 }
 /** Every stage compiles at once: an uncached pipeline costs the driver a
  * full Metal compile, and awaiting stages in turn serialized ~350 of them
  * behind one another on a scene's first load. */
 async initialize():Promise<void>{
  const root=this.pressureOwnership;
  this.pressureSchedule=new UniformMixedPressureSchedule(this.device,this.schedule,this.state,this.fields.pressure.tolerance,
   {native:this.fields.pressure.diagnostics,fine:root.support,supportWord:9*root.layout.tiles.length+24,status:this.status,band:this.band.index,bandClosedWord:UniformPressureBand.closedWord});
  await Promise.all([this.solid,this.displacement,this.transport,this.plan,this.cleanup,this.remap,this.phiResolve,this.extension,this.cache,this.hanging,this.surface,this.surfaceVolume,this.geometry,this.sharpen,this.momentum,this.forces,this.authority,this.projection,this.cycles,this.acceptance,this.split.transfer,this.split.authority,this.pressureSchedule,this.surfaceBand,this.band]
   .map(stage=>stage?.initialize()));
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
 /** Re-run the last advance's plan and extension into velocityScratch on
  * the live layout: the field the next advance's surface trace samples.
  * The dynamic census bounds departures from it between frames. */
 encodeExtension(encoder:GPUCommandEncoder):void{
  const p=this.lastParameters;if(!this.ready||this.busy||this.failed||!p)throw new Error("Mixed extension needs a completed advance");
  this.encodeExtensionOf(encoder,p);
 }
 private encodeExtensionOf(encoder:GPUCommandEncoder,p:UniformMixedFrameParameters):void{
  this.plan.encode(encoder,p.supportPolicy);
  // The last pressure setup left phase in pressure ownership. Phase only:
  // the advance's authority rewrites phi, correction and balance before
  // their readers (band rows, RHS).
  this.authority.encode(encoder,this.authorityGroup,false);
  this.extension.encode(encoder,this.extensionGroups,p.extensionSweeps??2);
  this.reusableExtension=extensionKey(p);
 }
 private write(p:UniformMixedFrameParameters):void{
  this.lastParameters=p;
  const h=this.ownership.capacity.lattice.cellSize_m;
  const floats=(b:GPUBuffer,v:number[])=>this.device.queue.writeBuffer(b,0,new Float32Array(v));
  const flags=(b:GPUBuffer,v:number[])=>this.device.queue.writeBuffer(b,16,new Uint32Array(v));
  floats(this.params.extension,[...h,+p.openTop]);floats(this.params.surface,[...h,p.dt]);flags(this.params.surface,[+p.openTop,+p.cubic,+p.drain,4]);
  floats(this.params.momentum,[...h,p.dt]);flags(this.params.momentum,[+p.openTop,0,0,UNIFORM_MIXED_MOMENTUM_LIMITS]);
  floats(this.params.forces,[...h,p.dt,p.gravity,p.density,p.viscosity,p.surfaceTension,+p.noSlip,+p.openTop,0,p.dust]);
  floats(this.params.authority,[p.dt,p.surfaceDeficitBalancing===true?0:-1,0,p.dust]);floats(this.params.sharpen,[p.sharpeningStrength,p.sharpeningDistance,p.dust,p.orphanDust??0,0,0,0,0]);
  floats(this.params.projection,[...h,p.dt,p.density,+p.openTop,0,p.dust]);floats(this.params.acceptance,[p.dt/p.density,uniformMixedPressureTarget(p),UNIFORM_PRESSURE_RELATIVE_REDUCTION,UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE]);
  floats(this.bandParams,[...h,p.dt,p.density,+p.openTop,p.dt/p.density,Math.min(...h),0,...h.map(x=>Math.fround(1/Math.fround(Math.fround(x)*Math.fround(x))))]);
  const views=this.layoutViews?(this.relayout?UNIFORM_STAGE_VIEWS.previous|UNIFORM_STAGE_VIEWS.reasons:0)|UNIFORM_STAGE_VIEWS.certificate:0;
  this.device.queue.writeBuffer(this.presentation.phi.buffer,4*this.stageGridWord,uniformStageGridHeader(this.ownership.capacity.tiles,this.stageBandTiles,views));
 }
 /** Record the layout views (the stage grids' previous, reasons and
  * certificate sections) for the tiles, grid and pressure layers. Off, the
  * frame copies nothing and the census records no reasons. */
 setLayoutViews(enabled:boolean):void{this.layoutViews=enabled;}
 private layoutViews=false;
 /** Copies one view section (a word per tile) into the stage grids. */
 private recordStageView(encoder:GPUCommandEncoder,source:{readonly buffer:GPUBuffer;readonly offset:number},section:"previous"|"reasons"|"certificate"):void{
  const n=this.ownership.capacity.tiles;
  encoder.copyBufferToBuffer(source.buffer,source.offset,this.presentation.phi.buffer,4*(this.stageGridWord+uniformStageViewWord(n,section)),4*n);
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
 /** extendTail: extend this frame's final velocity for the next frame
  * head's census (and reuse it there when the key still matches).
  * Encodes and submits synchronously; the promise is the frame's receipt,
  * checked when its map resolves while later frames encode. A failed receipt
  * rejects naming its frame and fails the frame for every later advance. */
 advance(p:UniformMixedFrameParameters,trace?:UniformMixedFrameTrace,extendTail=false,bodies?:UniformMixedFrameBodies):Promise<UniformMixedFrameReceipt>{
  if(bodies&&!this.solid)throw new Error("This mixed frame was built without solids; rigid bodies need the solid library");
  if(!this.ready||this.busy||this.failed)throw new Error("Unified frame is not ready for an advance");
  const readback=this.readbacks.find(b=>!this.unchecked.has(b));
  if(!readback)throw new Error(`Uniform mixed frame has ${this.readbacks.length} unchecked receipts; the host must check one before encoding frame ${this.frameIndex+1}`);
  this.busy=true;const frame=++this.frameIndex;
  // The frame a failure latched in this advance is recorded as.
  this.device.queue.writeBuffer(this.status,4*UNIFORM_MIXED_STATUS.currentFrame,new Uint32Array([frame]));
  const releases:(()=>void)[]=[];
  try{
   for(const ownership of new Set([this.ownership,...this.levels.map(l=>l.ownership)]))releases.push(ownership.acquireFrame());
   const makeEncoder=()=>{const raw=this.device.createCommandEncoder({label:"Uniform owner-driven frame"});return trace?.instrument(raw)??raw;};
   this.write(p);let encoder=makeEncoder();
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
   const relayout=this.relayout;
   if(!relayout&&(this.solidEditPending||bodies)){this.displacement!.encode(encoder,this.fields.volume,!this.solidEditPending);this.solidEditPending=false;}
   // Submit the frame in segments as it encodes: the GPU starts each one
   // while the host encodes the next, instead of idling through the whole
   // frame's encode. Queue writes made while a later segment encodes land
   // after the earlier segments, which never read them.
   const flush=()=>{trace?.submit(encoder,this.fields.negative);this.device.queue.submit([encoder.finish()]);encoder=makeEncoder();};
   encoder.clearBuffer(this.reductions);
   if(relayout)this.encodeRelayoutHead(encoder,p,relayout,bodies!==undefined,trace);
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
   if(!reuse){
    this.plan.encode(encoder,p.supportPolicy);if(!this.geometryCurrent)this.geometry.encode(encoder,this.geometryGroup);
    // Phase only: the authority below (split: the simulation authority
    // before the band rows, then the pressure authority) rewrites phi, every
    // correction texel and the balance scratch before their readers.
    this.authority.encode(encoder,this.authorityGroup,false);
    trace?.phase(encoder,V.support);
    if(!reuse){this.extension.encode(encoder,this.extensionGroups,p.extensionSweeps??2);trace?.phase(encoder,V.extension);}
   }
   }
   this.plan.encodeCertificate(encoder,p.dt);if(this.layoutViews)this.recordStageView(encoder,this.plan.certificate,"certificate");
   this.cache.encode(encoder,this.cacheGroup);this.hanging.encode(encoder,this.hangingGroup);
   trace?.phase(encoder,V.transportReach);
   this.surface.encode(encoder,"advect",this.surfaceGroups[0]);this.phiResolve.encode(encoder,this.phiResolveGroups.scratch);this.surface.encode(encoder,"traceCells",this.surfaceGroups[0]);
   if(p.redistance!==false){this.surface.encode(encoder,"redistance",this.surfaceGroups[1]);this.phiResolve.encode(encoder,this.phiResolveGroups.phi);}else this.copyWhole(encoder,this.fields.phiScratch,this.fields.phi);
   trace?.phase(encoder,V.phi);flush();
   this.transport.encodeCopy(encoder);this.transport.encodeTransport(encoder);
   // Cleanup and surface correction read the independent h phi field.
   if(p.dust>0)this.cleanup.encode(encoder,this.cleanupGroups);
   trace?.phase(encoder,V.coupling);
   // Apply shifts canonical vertices only; resolved readers below and the
   // next advect read the hanging texels.
   if(p.totalSurfaceVolume!==false&&(p.surfaceVolumeRounds??2)>0){this.surfaceVolume.encode(encoder,this.surfaceVolumeGroup,p.surfaceVolumeRounds??2);this.phiResolve.encode(encoder,this.phiResolveGroups.phi);}
   // Nothing after this pass writes phi: the next advance starts from it.

   // Nothing below writes phi before the split reads the pressure geometry.
   this.geometry.encode(encoder,this.geometryGroup,{pressure:true});this.geometryCurrent=true;
   trace?.phase(encoder,V.gather);
   if(p.sharpening!==false&&(p.sharpeningSweeps??8)>0&&p.sharpeningDistance>0){this.sharpen.encodeGeometry(encoder,this.sharpenGroups[0]);this.sharpen.encodeSweeps(encoder,this.sharpenGroups,p.sharpeningSweeps??8);}
   trace?.phase(encoder,V.sharpen);
   this.momentum.encode(encoder,this.momentumGroup);
   this.forces.encode(encoder,this.forceGroup);
   trace?.phase(encoder,A.advectionCorrection);flush();
   // Pressure stays all-4h; no layout build and no CPU wait. The band rows
   // need this frame's simulation authority and u*, both rewritten in
   // pressure ownership by the split below.
   this.surfaceBand.encode(encoder);this.recordStageGrid(encoder,this.surfaceBand.band.buffer,"band");
   this.authority.encode(encoder,this.authorityGroup);this.band.encodePrepare(encoder);
   const split=this.split;
   split.transfer.encodeToPressure(encoder,split.toPressure);
   split.authority.encode(encoder,split.authorityGroup);
   this.projection.encode(encoder,"rhs",split.rhsGroup);this.cycles.encodeSetup(encoder);
   this.cycles.encodeMeasure(encoder);this.acceptance.encode(encoder,this.acceptanceGroup,this.state,"initial");
   trace?.phase(encoder,A.pressureSetup);
   // The conservative schedule is encoded whole; GPU gates run only the
   // slots the last checkpoint and the lagged frame's plan call for. No CPU wait.
   const plannedBy=frame-UNIFORM_MIXED_RECEIPT_RING,lagged=this.lagged.get(plannedBy);
   if(plannedBy>0&&!lagged)throw new Error(`Uniform mixed frame ${frame} encoded before frame ${plannedBy}'s receipt was checked`);
   this.lagged.delete(plannedBy);
   const schedule=this.pressureSchedule,plan=lagged??this.initialPlan,vCycles=plan.vCycles;
   schedule.begin(plan);
   for(let slot=0;slot<schedule.slots;slot++){
    const gated=schedule.gate(encoder,slot);
    if(slot<vCycles)this.cycles.encodeVCycle(gated);else this.cycles.encodeFullCycle(gated);
    this.cycles.encodeMeasure(gated);this.acceptance.encode(gated,this.acceptanceGroup,this.state,"cycle");
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
   // The next head's census and advection read this extension: it is the
   // frame's velocity extension, priced there, not the census's.
   if(extendTail){this.encodeExtensionOf(encoder,p);trace?.phase(encoder,V.extension);}
   trace?.submit(encoder,this.fields.negative);this.device.queue.submit([encoder.finish()]);trace?.submitted();
   this.unchecked.add(readback);
   return this.check(readback,frame,p,plan.vCycles+plan.fullCycles);
  }catch(error){trace?.abort();this.failed=true;throw error;}finally{for(const release of releases)release();this.busy=false;}
 }
 /** Frame `frame`'s receipt: fail fast on a rejected or unconverged solve. */
 private async check(readback:GPUBuffer,frame:number,p:UniformMixedFrameParameters,encoded:number):Promise<UniformMixedFrameReceipt>{
  let mapped:Uint32Array;
  try{
   await readback.mapAsync(GPUMapMode.READ);
   mapped=new Uint32Array(readback.getMappedRange(),0,30+UNIFORM_MIXED_STATUS_WORDS).slice();readback.unmap();
  }catch(error){this.failed=true;throw error;}finally{this.unchecked.delete(readback);}
  try{
   const state=mapped.slice(0,8),accounting=mapped.slice(8,20);
   // The GPU's own verdict first: it names the first failing frame.
   const failure=describeUniformMixedFrameStatus(mapped.slice(30,30+UNIFORM_MIXED_STATUS_WORDS));
   if(failure)throw new Error(`${failure}; projection withheld${typeof process!=="undefined"&&process.env.FLUID_MIXED_HOST_DIAGNOSTICS?`; params ${JSON.stringify(p)}; receipt ${[...state]}`:""}`);
   const residual=new Float32Array(state.buffer)[1]!,count=state[7]!;
   if(state[4]!==0||state[5]===0||!Number.isFinite(residual)||residual<0||residual>uniformMixedPressureTarget(p)){
    throw new Error(`Uniform mixed pressure ${state[4]!==0?"rejected a non-improving cycle":"did not converge"}: candidate ${new Float32Array(state.buffer)[0]}, accepted ${residual}, tolerance ${uniformMixedPressureTarget(p)}, ${count} cycles; projection withheld${typeof process!=="undefined"&&process.env.FLUID_MIXED_HOST_DIAGNOSTICS?`; params ${JSON.stringify(p)}; receipt ${[...state]}`:""}`);
   }
   if(mapped[23]!==0)throw new Error(`Uniform pressure band needs ${mapped[22]} tiles, over its capacity; projection is incomplete`);
   if(mapped[25]!==0)throw new Error(`Uniform pressure band solid certificate failed (${mapped[25]&1?"a cut tile the simulation holds at 4h has a liquid row":""}${mapped[25]===3?"; ":""}${mapped[25]&2?"a Neumann face is cut, V<1":""}); projection is incomplete`);
   this.bandTiles=mapped[22]!;this.bandResidual=new Float32Array(mapped.buffer)[24]!;
   // Only an accepted solve plans; a failed one threw above. Frame
   // frame+UNIFORM_MIXED_RECEIPT_RING encodes it, whenever this resolves.
   this.lagged.set(frame,{vCycles:mapped[20]!,fullCycles:mapped[21]!});
   const cells=this.ownership.capacity.lattice.dimensions.reduce((n,d)=>n*d,1);
   const orphanDustMass_cells=uniformMixedDustMass(accounting,10,p.orphanDust??0,cells);
   return {cycles:count,encoded,residual,converged:true,
    dustOwners:accounting[5]!+accounting[10]!,dustMass_cells:uniformMixedDustMass(accounting,5,p.dust,cells)+orphanDustMass_cells,
    orphanDustOwners:accounting[10]!,orphanDustMass_cells,bandTiles:mapped[22]!,bandCycles:mapped[26]!,bandResidual:this.bandResidual};
  }catch(error){this.failed=true;throw new Error(`Uniform mixed frame ${frame}: ${error instanceof Error?error.message:String(error)}`,{cause:error});}
 }
 /** The layout generation hook: copy the GPU's current generation word (one
  * u32 at `offset` of `source`) into the status record, so a failure latched
  * after it names that generation. Encode it wherever a generation is adopted. */
 encodeLayoutGeneration(encoder:GPUCommandEncoder,source:GPUBuffer,offset:number):void{
  encoder.copyBufferToBuffer(source,offset,this.status,4*UNIFORM_MIXED_STATUS.currentGeneration,4);
 }
 private copyWhole(encoder:GPUCommandEncoder,from:GPUTexture,to:GPUTexture):void{
  encoder.copyTextureToTexture({texture:from},{texture:to},[from.width,from.height,from.depthOrArrayLayers]);
 }
 /** Takes the ungraded h/4h simulation layout; pressure stays all-4h. The
  * host (CPU) layout path: authored regions, and the initial layout. */
 updateLayout(layout:UniformMixedLayout):void{
  if(!this.ready||this.busy||this.failed)throw new Error("Ownership edits require a completed frame");
  if(this.relayout)throw new Error("A GPU relayout owns the mixed layout: detach it (setRelayout()) before a host relayout");
  this.reusableExtension=undefined;this.geometryCurrent=false;this.solidWidthsStale=true;
  this.remap.apply(layout);
  {const e=this.device.createCommandEncoder({label:"Uniform resolve remapped phi"});this.phiResolve.encode(e,this.phiResolveGroups.phi);this.device.queue.submit([e.finish()]);}
 }
 /** Dynamic coarsening's GPU relayout, run at the head of every advance
  * (encodeRelayoutHead); undefined returns the layout to updateLayout. */
 private relayout?:UniformMixedFrameRelayout;
 setRelayout(relayout?:UniformMixedFrameRelayout):void{
  if(this.busy)throw new Error("The mixed relayout cannot change during an advance");
  this.relayout=relayout;
  // Only the relayout's census certifies residency: without it every page is resident.
  if(!relayout)this.ownership.resetResidency();
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
  if(!reuse){
   if(this.solidWidthsStale)this.solid?.encodeSimulation(encoder,this.ownership.presentation);
   this.plan.encode(encoder,p.supportPolicy);if(!this.geometryCurrent)this.geometry.encode(encoder,this.geometryGroup);
   this.authority.encode(encoder,this.authorityGroup,false);
   this.extension.encode(encoder,this.extensionGroups,p.extensionSweeps??2);trace?.phase(encoder,V.extension);
  }
  // The tiles as the frame starts, for the grid layer's relayout marks.
  if(this.layoutViews)this.recordStageView(encoder,{buffer:this.ownership.presentation.buffer,offset:this.ownership.presentation.offset??0},"previous");
  relayout.encode(encoder,p.dt,this.layoutViews);
  if(this.layoutViews)this.recordStageView(encoder,relayout.reasons,"reasons");
  this.remap.applyGpu(encoder,relayout.generation,relayout.receipt,true);
  this.encodeLayoutGeneration(encoder,relayout.receipt.buffer,relayout.receipt.offset+4*UNIFORM_MIXED_RELAYOUT_RECEIPT.generation);
  this.geometryCurrent=false;
  trace?.phase(encoder,A.resolutionCensus);
  this.phiResolve.encode(encoder,this.phiResolveGroups.phi);
  // Only the relayout moved phi since the last geometry unless solids moved.
  const solidsMoved=this.solidEditPending||bodies;
  if(this.solidEditPending||bodies){this.displacement!.encode(encoder,this.fields.volume,!this.solidEditPending);this.solidEditPending=false;}
  // Which cut tiles the simulation holds at h: the all-4h levels read their h texels.
  this.solid?.encodeSimulation(encoder,this.ownership.presentation);this.solidWidthsStale=false;
  if(this.solid)trace?.phase(encoder,V.solids);
  this.recordStageGrid(encoder,this.ownership.presentation.buffer,"transport");
  this.plan.encode(encoder,p.supportPolicy);this.geometry.encode(encoder,this.geometryGroup,{changed:!solidsMoved});
  this.authority.encode(encoder,this.authorityGroup,false);
  trace?.phase(encoder,V.support);
 }
 destroy():void{this.solid?.destroy();this.displacement?.destroy();this.pressureSchedule?.destroy();this.surfaceBand.destroy();this.band.destroy();this.plan.destroy();this.surface.destroy();this.geometry.destroy();this.momentum.destroy();this.hanging.destroy();this.remap.destroy();this.transport.destroy();this.cleanup.destroy();this.split.transfer.destroy();for(const l of this.levels)if(l.ownership!==this.ownership)l.ownership.destroy();for(const r of this.owned)r.destroy();}
}
