import {UniformPressureSurfaceBand} from "./uniform-pressure-surface-band";
import {UniformPressureBand} from "./uniform-pressure-band";
import {uniformMixedDustMass} from "./uniform-mixed-dust-accounting.wgsl";
import {UniformMixedCleanup} from "./uniform-mixed-cleanup";
import {UniformMixedFramePlan} from "./uniform-mixed-frame-plan";
import {UniformMixedOwnershipTransfer,UniformMixedRemap} from "./uniform-mixed-remap";
import {UniformMixedPhiResolve} from "./uniform-mixed-phi-resolve";
import {mixedCellWidth,uniformMixedAllCoarseLayout} from "./uniform-mixed-layout";
import type {UniformMixedLayout} from "./uniform-mixed-layout";
import type {UniformScratchArena} from "./uniform-scratch-arena";
import type {WebGPUUniformPressureMultigrid} from "./webgpu-uniform-pressure-multigrid";
import type {WebGPUUniformVelocityExtrapolator} from "./webgpu-uniform-velocity-extrapolation";
import {UniformMixedTransportStage} from "./uniform-mixed-transport";
import {UniformMixedOwnership,uniformMixedHangingCapacity,type UniformMixedBuiltOwnership} from "./uniform-mixed-ownership";
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
import {planUniformMixedPressureMemory} from "./uniform-mixed-pressure-memory";
import {DEFAULT_UNIFORM_CM11A_SCHEDULE,UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE,UNIFORM_PRESSURE_RELATIVE_REDUCTION,type UniformCM11aSchedule} from "./pressure-policy";
import type {GPUTimestampPhase} from "../../core/performance-trace";
import {UNIFORM_ADVANCE_PHASE as A} from "./uniform-stages";
import {UNIFORM_VOLUME_PHASE as V} from "./uniform-volume-stages";
import {UniformMixedSolid,type UniformMixedSolidResources} from "./uniform-mixed-solid.wgsl";
import {UniformMixedSolidDisplacement} from "./uniform-mixed-solid-displacement";
import {UNIFORM_STAGE_GRID_HEADER_WORDS,uniformStageBandWords,uniformStageGridHeader,uniformStageGridWords} from "./uniform-stage-grids";

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
  * Unless the simulation is itself all-4h, the solve runs split: in the
  * all-4h ownership with its own surface target and centre phi here, the
  * simulation's own staying intact for the renderer. */
 pressureGeometry:{target:GPUTexture;centerPhi:GPUTexture};
}
export interface UniformMixedFrameParameters {
 dt:number;gravity:number;density:number;viscosity:number;surfaceTension:number;
 openTop:boolean;noSlip:boolean;cubic:boolean;drain:boolean;
 totalSurfaceVolume?:boolean;redistance?:boolean;sharpening?:boolean;surfaceDeficitBalancing?:boolean;extensionSweeps?:number;
 supportPolicy?:{fineReach:number;shellReach:number;twoLevel:boolean;shellOnly:boolean};
 dust:number;orphanDust?:number;sharpeningStrength:number;sharpeningDistance:number;pressureTolerance:number;
}

/** One owner-driven frame sequence. All persistent fields and large scratch
 * ranges belong to the host. Fine and coarse are layouts of this sequence.
 * Only compact metadata, parameter/receipt buffers and 4h sampling caches are
 * owned here. No readback contains simulation fields. */
/** Frames whose receipts may be unchecked at once: the host's frames-ahead cap. */
export const UNIFORM_MIXED_RECEIPT_RING=2;
/** Rigid bodies this frame (the solid library reads their GPU state):
 * their cut cells move, so the all-4h record, cut widths and geometry are
 * rebuilt and entered cells displaced at the head. couple, when coupling is
 * on, encodes the fluid-to-body exchange and the body integration after the
 * projection, before the census. */
export interface UniformMixedFrameBodies{couple?:(encoder:GPUCommandEncoder)=>void}
export interface UniformMixedFrameReceipt{cycles:number;encoded:number;residual:number;converged:boolean;dustOwners:number;dustMass_cells:number;orphanDustOwners:number;orphanDustMass_cells:number;
 /** The h band this frame re-solved: tiles, completed cycles, final residual (0 when pressure matched bulk). */
 bandTiles:number;bandCycles:number;bandResidual:number}
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
  geometry:UniformMixedSurfaceGeometry;geometryGroup:GPUBindGroup;authority:UniformMixedPressureAuthority;authorityGroup:GPUBindGroup;
  rhsGroup:GPUBindGroup;projectionGroup:GPUBindGroup;
 };
 /** The simulation layout is all-4h: pressure and simulation layouts are
  * identical, so pressure binds the simulation fields directly. */
 private pressureMatchesSimulation=false;
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
 readonly extension:UniformMixedExtension;
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
 private readonly rhsGroup:GPUBindGroup;
 private readonly projectionGroup:GPUBindGroup;
 private readonly acceptanceGroup:GPUBindGroup;
 private readonly params:Record<"extension"|"surface"|"momentum"|"forces"|"authority"|"sharpen"|"projection"|"acceptance",GPUBuffer>;
 private readonly state:GPUBuffer;
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
 private readonly lagged=new Map<number,{plan:UniformMixedPressurePlan;bandTiles:number}>();
 private readonly reductions:GPUBuffer;
 private pressureSchedule!:UniformMixedPressureSchedule;
 /** The slot list of the first UNIFORM_MIXED_RECEIPT_RING frames: conservative. */
 private readonly initialPlan:UniformMixedPressurePlan;
 /** Surface-crossing tiles of the current phi: the stage grids' band bits. */
 private readonly surfaceBand:UniformPressureSurfaceBand;
 private ready=false;
 private busy=false;
 private failed=false;
 get allocatedBytes():number{return (this.displacement?.allocatedBytes??0)+this.surfaceBand.allocatedBytes+this.band.allocatedBytes+this.plan.allocatedBytes+(this.solid?.allocatedBytes??0)+this.surface.allocatedBytes+this.hanging.allocatedBytes+this.transport.allocatedBytes+this.remap.allocatedBytes+(this.split?.transfer.allocatedBytes??0)+this.levels.filter(l=>l.ownership!==this.ownership).reduce((n,l)=>n+l.ownership.allocatedBytes,0)+this.owned.reduce((n,r)=>n+("size" in r?r.size:r.width*r.height*r.depthOrArrayLayers*16),0);}
 constructor(private readonly device:GPUDevice,layout:UniformMixedLayout,private readonly fields:UniformMixedFrameFields,openTop:boolean,private readonly schedule:UniformCM11aSchedule=DEFAULT_UNIFORM_CM11A_SCHEDULE){
  if(layout.cellCount!==layout.tiles.length*64)throw new Error("Unified frame must reserve its capacity with fine ownership");
  const f=fields;
  const buffer=(label:string,size:number,usage:number)=>{const b=device.createBuffer({label,size,usage});this.owned.push(b);return b;};
  const uniform=(name:string,size:number)=>buffer(`Uniform ${name} parameters`,size,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
  this.params={extension:uniform("extension",16),surface:uniform("surface",32),momentum:uniform("momentum",32),forces:uniform("forces",48),authority:uniform("authority",16),sharpen:uniform("sharpen",32),projection:uniform("projection",32),acceptance:uniform("acceptance",16)};
  this.state=buffer("Uniform pressure acceptance",32,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST);
  this.initialPlan={vCycles:schedule.vCycles,fullCycles:schedule.fullCycles};
  this.readbacks=Array.from({length:UNIFORM_MIXED_RECEIPT_RING},(_,i)=>buffer(`Uniform pressure receipt and mass accounting ${i}`,128,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ));
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
  this.phiResolve=new UniformMixedPhiResolve(device,o);this.phiResolveGroups={phi:this.phiResolve.bind(f.phi),scratch:this.phiResolve.bind(f.phiScratch)};
  const prefix=Math.min(...[f.pressure.pressure,f.pressure.rhs,f.pressure.minimum,f.pressure.phi,f.pressure.topology].map(v=>v.buffer!.offset??0));
  const memory=planUniformMixedPressureMemory(layout,prefix,f.conditioning.size);
  const view=(r:{offset:number;size:number},external=false):GPUBufferBinding=>({buffer:external?f.conditioning:f.arena.buffer,...r});
  // Pressure's all-4h root keeps the mixed rows; the native hierarchy
  // solves its correction from its n/4 level.
  // With solids, the root's topology is the static all-4h solid record,
  // copied each advance into the conditioning buffer directly after the
  // all-4h phi (the root reads it as its phi binding's tail).
  if(solid?.coarse){
   const r=memory.root.phi,base=4*Math.ceil(layout.tiles.length*4/256)*64,bytes=16*solid.coarse.count;
   if(base+bytes>r.size)throw new Error("Band solid topology does not fit behind the all-4h phi");
   this.solidTopology={buffer:f.conditioning,offset:r.offset+base,size:bytes};
  }
  {const r=memory.root;
   this.levels=[{ownership:new UniformMixedOwnership(device,coarseLayout,false,true),pressure:view(r.pressure),rhs:[view(r.rhs[0]),view(r.rhs[1])],minimum:[view(r.minimum[0]),view(r.minimum[1])],phi:view(r.phi,true),frozen:view(r.frozen),residual:view(r.residual),
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
  this.geometry=new UniformMixedSurfaceGeometry(device,o,solid,true);this.geometryGroup=this.geometry.bind(f.phi,f.target,f.centerPhi);
  this.sharpen=new UniformMixedSharpening(device,o,solid,{list:buffer("Uniform mixed sharpening tile list",UniformMixedSharpening.workBytes(layout.tiles.length),GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST)},true);
  this.sharpenGroups=[this.sharpen.bind(f.volume,f.volumeScratch,f.phi,f.target,f.centerPhi,{buffer:f.arena.buffer,offset:0,size:f.arena.edgeBytes},this.params.sharpen,this.reductions),this.sharpen.bind(f.volumeScratch,f.volume,f.phi,f.target,f.centerPhi,{buffer:f.arena.buffer,offset:0,size:f.arena.edgeBytes},this.params.sharpen,this.reductions)];
  // Retain the projected h velocity detail: restricting all momentum to 4h
  // removes the long-dam toe even when total liquid volume is conserved.
  this.momentum=new UniformMixedMomentum(device,o,true,true);
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
  // Unsplit (an all-4h simulation) the forced field IS the input, and a
  // writable buffer cannot be bound twice. Its cut-face taps never run there:
  // no tile is simulated at h (umSolidCut), so bind the idle output pair.
  [this.rhsGroup,this.projectionGroup]=bindProjection({velocity:f.velocityScratch,negative:f.negativeScratch},{velocity:f.velocity,negative:f.negative},f.centerPhi,f.volume,
   coarseSolid?{velocity:f.velocity,negative:f.negative}:undefined);
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
   const geometry=new UniformMixedSurfaceGeometry(device,p,solid,false),authority=new UniformMixedPressureAuthority(device,p,solid,coarseSolid);
   this.split={transfer,rhsGroup,projectionGroup,
    toPressure:transfer.bind({volume:f.volume,velocity:f.velocityScratch,negative:f.negativeScratch},{volume:f.volumeScratch,velocity:f.velocity,negative:f.negative}),
    toSimulation:transfer.bind({volume:f.volume,velocity:f.velocityScratch,negative:f.negativeScratch},{volume:f.volumeScratch,velocity:f.velocity,negative:f.negative}),
    // The resolved simulation phi holds every 4h corner's vertex value.
    geometry,geometryGroup:geometry.bind(f.phi,g.target,g.centerPhi),
    authority,authorityGroup:authority.bind({centerPhi:g.centerPhi,volume:f.volumeScratch,targetFill:g.target,phi:root.phi!,phase:f.phase,correction:f.correction,scratch:root.frozen,params:this.params.authority,
     fine:coarseSolid?{centerPhi:f.centerPhi,volume:f.volume}:undefined})};
  }
  this.bandParams=uniform("pressure band",48);
  this.surfaceBand=new UniformPressureSurfaceBand(device,layout,f.phi);
  this.band=new UniformPressureBand(device,o,p,{phi:root.phi!,correction:f.correction,vertexPhi:f.phi,
   forced:{velocity:f.velocityScratch,negative:f.negativeScratch},velocity:f.velocity,negative:f.negative,copy:f.velocityScratch,
   coarsePressure:root.pressure,params:this.bandParams,presentation:{buffer:this.presentation.phi.buffer,word:this.stageBandWord+layout.tiles.length}},undefined,solid);
  if(this.band.capacity!==this.stageBandTiles)throw new Error("The stage grids' band section must match the band capacity");
  // Setup is encoded once per solve, ungated (advance); cycles are pure cycle work.
  this.cycles=new UniformMixedPressureCycles(device,root,view(memory.backup),f.pressure,f.uniformGroup,openTop);
  this.acceptance=new UniformMixedPressureAcceptance(device,p);
  this.acceptanceGroup=this.acceptance.bind({residual:root.residual,state:this.state,params:this.params.acceptance});
 }
 async initialize():Promise<void>{
  await this.solid?.initialize();
  await this.displacement?.initialize();
  for(const stage of [this.transport,this.plan,this.cleanup,this.remap,this.phiResolve,this.extension,this.cache,this.hanging,this.surface,this.surfaceVolume,this.geometry,this.sharpen,this.momentum,this.forces,this.authority,this.projection,this.cycles,this.acceptance,this.split.transfer,this.split.geometry,this.split.authority])await stage.initialize();
  const root=this.pressureOwnership;
  this.pressureSchedule=new UniformMixedPressureSchedule(this.device,this.schedule,this.state,this.fields.pressure.tolerance,
   {native:this.fields.pressure.diagnostics,fine:root.support,supportWord:9*root.layout.tiles.length+24});
  await this.pressureSchedule.initialize();
  await this.surfaceBand.initialize();await this.band.initialize();
  this.ready=true;
 }
 /** Any state, parameter or ownership edit between frames. */
 invalidateExtension():void{this.geometryCurrent=false;}
 /** A live voxel edit rewrote the host's solid mask: rebuild the all-4h
  * record and the cut widths, and displace liquid out of newly solid h
  * cells at the next frame's head. The host has already promoted every
  * tile the edit touches to h (its next relayout or adopt). */
 editSolids():void{
  if(!this.solid)throw new Error("This mixed frame was built without solids");
  this.solid.invalidate();this.solidWidthsStale=true;this.geometryCurrent=false;this.solidEditPending=true;
 }
 /** The simulation target and centre phi were built from the current phi by
  * the last completed advance's gather: nothing has written phi, solids or
  * ownership since. Every such edit clears it. */
 private geometryCurrent=false;
 private write(p:UniformMixedFrameParameters):void{
  const h=this.ownership.capacity.lattice.cellSize_m;
  const floats=(b:GPUBuffer,v:number[])=>this.device.queue.writeBuffer(b,0,new Float32Array(v));
  const flags=(b:GPUBuffer,v:number[])=>this.device.queue.writeBuffer(b,16,new Uint32Array(v));
  floats(this.params.extension,[...h,0]);floats(this.params.surface,[...h,p.dt]);flags(this.params.surface,[+p.openTop,+p.cubic,+p.drain,4]);
  floats(this.params.momentum,[...h,p.dt]);flags(this.params.momentum,[+p.openTop,0,0,UNIFORM_MIXED_MOMENTUM_LIMITS]);
  floats(this.params.forces,[...h,p.dt,p.gravity,p.density,p.viscosity,p.surfaceTension,+p.noSlip,+p.openTop,0,p.dust]);
  floats(this.params.authority,[p.dt,p.surfaceDeficitBalancing===true?0:-1,0,p.dust]);floats(this.params.sharpen,[p.sharpeningStrength,p.sharpeningDistance,p.dust,p.orphanDust??0,0,0,0,0]);
  floats(this.params.projection,[...h,p.dt,p.density,+p.openTop,0,p.dust]);floats(this.params.acceptance,[p.dt/p.density,p.pressureTolerance,UNIFORM_PRESSURE_RELATIVE_REDUCTION,UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE]);
  floats(this.bandParams,[...h,p.dt,p.density,+p.openTop,p.dt/p.density,Math.min(...h),...h.map(x=>Math.fround(1/Math.fround(Math.fround(x)*Math.fround(x)))),0]);
  this.device.queue.writeBuffer(this.presentation.phi.buffer,4*this.stageGridWord,uniformStageGridHeader(this.ownership.capacity.tileCount,this.stageBandTiles));
 }
 /** Copies one stage's tile words (or the band bits) into the stage grids. */
 private recordStageGrid(encoder:GPUCommandEncoder,source:GPUBuffer,range:"transport"|"pressure"|"band"):void{
  const n=this.ownership.capacity.tileCount,words=range==="band"?Math.ceil(n/32):n;
  const at=this.stageGridWord+UNIFORM_STAGE_GRID_HEADER_WORDS+(range==="transport"?0:range==="pressure"?n:2*n);
  encoder.copyBufferToBuffer(source,0,this.presentation.phi.buffer,4*at,4*words);
 }
 /** Discarded mass uses native sixty-fourths-of-threshold counters, weighted
  * by owner volume. It is a quantized lower bound; counts name owners, not
  * fine cells. Counters are reset once per frame, before transport cleanup. */
 /** census: the dynamic classifier and layout builder, encoded at the end
  * of this frame's last submission, so their readback shares the frame's
  * final map instead of costing the host a second round trip. It reads the
  * final physical velocity and the nearest-source hierarchy built from it
  * here (no faces published); the next advance extends on the adopted
  * layout.
  * Encodes and submits synchronously; the promise is the frame's receipt,
  * checked when its map resolves while later frames encode. A failed receipt
  * rejects naming its frame and fails the frame for every later advance. */
 advance(p:UniformMixedFrameParameters,trace?:UniformMixedFrameTrace,census?:(encoder:GPUCommandEncoder)=>(()=>void),bodies?:UniformMixedFrameBodies):Promise<UniformMixedFrameReceipt>{
  if(bodies&&!this.solid)throw new Error("This mixed frame was built without solids; rigid bodies need the solid library");
  if(!this.ready||this.busy||this.failed)throw new Error("Unified frame is not ready for an advance");
  const readback=this.readbacks.find(b=>!this.unchecked.has(b));
  if(!readback)throw new Error(`Uniform mixed frame has ${this.readbacks.length} unchecked receipts; the host must check one before encoding frame ${this.frameIndex+1}`);
  this.busy=true;const frame=++this.frameIndex;
  const releases:(()=>void)[]=[];
  try{
   const makeEncoder=()=>{const raw=this.device.createCommandEncoder({label:"Uniform owner-driven frame"});return trace?.instrument(raw)??raw;};
   let encoder=makeEncoder();
   // A census relayout adopted since the last frame, before ownership is held.
   if(this.pendingAdopts.length){for(const adopt of this.pendingAdopts)adopt(encoder);this.pendingAdopts=[];trace?.phase(encoder,A.resolutionRelayout);}
   // Bodies moved at the last frame's tail: their cut cells are new.
   if(bodies){this.solid!.encodeBodies(encoder);this.solid!.invalidate(true);this.solidWidthsStale=true;this.geometryCurrent=false;}
   // A live solid edit, or the cells bodies entered, lands on the adopted
   // (promoted) ownership.
   if(this.solidEditPending||bodies){this.displacement!.encode(encoder,this.fields.volume,!this.solidEditPending);this.solidEditPending=false;}
   for(const ownership of new Set([this.ownership,...this.levels.map(l=>l.ownership)]))releases.push(ownership.acquireFrame());
   this.write(p);
   // Submit the frame in segments as it encodes: the GPU starts each one
   // while the host encodes the next, instead of idling through the whole
   // frame's encode. Queue writes made while a later segment encodes land
   // after the earlier segments, which never read them.
   const flush=()=>{trace?.submit(encoder,this.fields.negative);this.device.queue.submit([encoder.finish()]);encoder=makeEncoder();};
   encoder.clearBuffer(this.reductions);
   // Built once for static solids (again after an edit, every frame bodies
   // exist), before the band's tile list reads its cut flags.
   this.solid?.encodeCoarse(encoder);
   // Which cut tiles the simulation holds at h: the all-4h levels read their h texels.
   if(this.solidWidthsStale){this.solid?.encodeSimulation(encoder,this.ownership.presentation);this.solidWidthsStale=false;}
   if(this.solid)trace?.phase(encoder,V.solids);
   this.recordStageGrid(encoder,this.ownership.presentation.buffer,"transport");
   // Geometry and centre phi are the last frame's gather unless an edit or
   // an adopt cleared them. The h phi field survives bulk relayout unchanged.

   this.plan.encode(encoder,p.supportPolicy);if(!this.geometryCurrent)this.geometry.encode(encoder,this.geometryGroup);
   // Phase only: the authority below (split: the simulation authority
   // before the band rows, then the pressure authority) rewrites phi, every
   // correction texel and the balance scratch before their readers.
   this.authority.encode(encoder,this.authorityGroup,false);
   trace?.phase(encoder,V.support);
   this.extension.encode(encoder,this.extensionGroups,p.extensionSweeps??2);
   trace?.phase(encoder,V.extension);
   this.plan.encodeCertificate(encoder,p.dt);this.cache.encode(encoder,this.cacheGroup);this.hanging.encode(encoder,this.hangingGroup);
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
   if(p.totalSurfaceVolume!==false){this.surfaceVolume.encode(encoder,this.surfaceVolumeGroup);this.phiResolve.encode(encoder,this.phiResolveGroups.phi);}
   // Nothing after this pass writes phi: the next advance starts from it.

   this.geometry.encode(encoder,this.geometryGroup);this.geometryCurrent=true;
   trace?.phase(encoder,V.gather);
   if(p.sharpening!==false){this.sharpen.encodeGeometry(encoder,this.sharpenGroups[0]);this.sharpen.encodeSweeps(encoder,this.sharpenGroups,8);}
   trace?.phase(encoder,V.sharpen);
   this.momentum.encode(encoder,this.momentumGroup);
   this.forces.encode(encoder,this.forceGroup);
   trace?.phase(encoder,A.advectionCorrection);flush();
   // Pressure stays all-4h; no layout build and no CPU wait. The band rows
   // need this frame's simulation authority and u*, both rewritten in
   // pressure ownership by the split below.
   this.surfaceBand.encode(encoder);this.recordStageGrid(encoder,this.surfaceBand.band.buffer,"band");
   if(!this.pressureMatchesSimulation){this.authority.encode(encoder,this.authorityGroup);this.band.encodePrepare(encoder);}
   const split=this.pressureMatchesSimulation?undefined:this.split;
   if(split){
    split.transfer.encodeToPressure(encoder,split.toPressure);
    split.geometry.encode(encoder,split.geometryGroup);split.authority.encode(encoder,split.authorityGroup);
   }else this.authority.encode(encoder,this.authorityGroup);
   if(this.solidTopology)encoder.copyBufferToBuffer(this.solid!.coarse!.record,0,this.solidTopology.buffer,this.solidTopology.offset!,this.solidTopology.size!);
   this.projection.encode(encoder,"rhs",split?.rhsGroup??this.rhsGroup);this.cycles.encodeSetup(encoder);
   this.cycles.encodeMeasure(encoder);this.acceptance.encode(encoder,this.acceptanceGroup,this.state,"initial");
   trace?.phase(encoder,A.pressureSetup);
   // The conservative schedule is encoded whole; GPU gates run only the
   // slots the last checkpoint and the lagged frame's plan call for. No CPU wait.
   const plannedBy=frame-UNIFORM_MIXED_RECEIPT_RING,lagged=this.lagged.get(plannedBy);
   if(plannedBy>0&&!lagged)throw new Error(`Uniform mixed frame ${frame} encoded before frame ${plannedBy}'s receipt was checked`);
   this.lagged.delete(plannedBy);
   const schedule=this.pressureSchedule,plan=lagged?.plan??this.initialPlan,vCycles=plan.vCycles;
   schedule.begin(plan);
   for(let slot=0;slot<schedule.slots;slot++){
    const gated=schedule.gate(encoder,slot);
    if(slot<vCycles)this.cycles.encodeVCycle(gated);else this.cycles.encodeFullCycle(gated);
    this.cycles.encodeMeasure(gated);this.acceptance.encode(gated,this.acceptanceGroup,this.state,"cycle");
    trace?.phase(encoder,slot<vCycles?A.pressureVCycles:A.pressureFullCycles);
   }
   // Projection reads the accepted iterate; the all-4h root has no seam,
   // so no reconstruction slope. Its gate enables it only on acceptance.
   this.projection.encode(schedule.gate(encoder,schedule.slots),"project",split?.projectionGroup??this.projectionGroup);
   schedule.end();
   trace?.phase(encoder,A.pressureProjection);
   if(split){
    split.transfer.encodeToSimulation(encoder,split.toSimulation);
    this.band.encodeSolve(encoder,lagged?.bandTiles??0);
   }
   const root=this.levels[0]!;
   // Present the live all-4h words only: level 0 keeps the simulation
   // layout's capacity, but its owners and boundary slots are all-4h.
   for(const [from,to,words] of [[root.pressure,this.presentation.pressure,this.pressureWords],[root.phi!,this.presentation.phi,this.pressureOwnership.layout.cellCount]] as const)encoder.copyBufferToBuffer(from.buffer,from.offset??0,to.buffer,0,4*words);
   this.recordStageGrid(encoder,this.pressureOwnership.presentation.buffer,"pressure");
   {
    // The h band re-solved this frame, or none when pressure matched bulk.
    const n=this.ownership.capacity.tileCount,at=4*this.stageBandWord;
    // The live band pressures were presented by the band's own slot launch.
    if(split)encoder.copyBufferToBuffer(this.band.index,this.band.slotMapOffset,this.presentation.phi.buffer,at,4*n);
    else encoder.clearBuffer(this.presentation.phi.buffer,at,4*n);
   }
   trace?.phase(encoder,V.band);
   // Native coupleRigid and the rigid integration read the projected field;
   // the census below promotes around the integrated poses.
   if(bodies?.couple){bodies.couple(encoder);trace?.phase(encoder,A.rigidCoupling);}
   encoder.copyBufferToBuffer(this.state,0,readback,0,32);encoder.copyBufferToBuffer(this.reductions,0,readback,32,48);schedule.encodePlanCopy(encoder,readback,80);
   if(split)this.band.encodeReceipt(encoder,readback,88);else encoder.clearBuffer(readback,88,32);
   // 30: the adopted generation's hanging overflow (seam tiles, 0 if none).
   encoder.copyBufferToBuffer(this.ownership.support,(9*this.ownership.tileCount+25)*4,readback,120,4);
   if(census)this.extension.encodeFar(encoder,this.extensionGroups);
   const submitted=census?.(encoder);
   if(census)trace?.phase(encoder,A.resolutionCensus);
   // The census maps alongside the receipt: one wait for both.
   trace?.submit(encoder,this.fields.negative);this.device.queue.submit([encoder.finish()]);trace?.submitted();submitted?.();
   this.unchecked.add(readback);
   return this.check(readback,frame,p,plan.vCycles+plan.fullCycles);
  }catch(error){trace?.abort();this.failed=true;throw error;}finally{for(const release of releases)release();this.busy=false;}
 }
 /** Frame `frame`'s receipt: fail fast on a rejected or unconverged solve. */
 private async check(readback:GPUBuffer,frame:number,p:UniformMixedFrameParameters,encoded:number):Promise<UniformMixedFrameReceipt>{
  let mapped:Uint32Array;
  try{
   await readback.mapAsync(GPUMapMode.READ);
   mapped=new Uint32Array(readback.getMappedRange(),0,31).slice();readback.unmap();
  }catch(error){this.failed=true;throw error;}finally{this.unchecked.delete(readback);}
  try{
   const state=mapped.slice(0,8),accounting=mapped.slice(8,20);
   const residual=new Float32Array(state.buffer)[1]!,count=state[7]!;
   if(state[4]!==0||state[5]===0||!Number.isFinite(residual)||residual<0||residual>p.pressureTolerance){
    throw new Error(`Uniform mixed pressure ${state[4]!==0?"rejected a non-improving cycle":"did not converge"}: candidate ${new Float32Array(state.buffer)[0]}, accepted ${residual}, tolerance ${p.pressureTolerance}, ${count} cycles; projection withheld${typeof process!=="undefined"&&process.env.FLUID_MIXED_HOST_DIAGNOSTICS?`; params ${JSON.stringify(p)}; receipt ${[...state]}`:""}`);
   }
   if(mapped[30]!==0)throw new Error(`Uniform mixed ownership has ${mapped[30]} seam tiles, over the hanging tap cache's ${uniformMixedHangingCapacity(this.ownership.tileCount)} slots`);
   if(mapped[23]!==0)throw new Error(`Uniform pressure band needs ${mapped[22]} tiles, over its capacity; projection is incomplete`);
   if(mapped[25]!==0)throw new Error(`Uniform pressure band solid certificate failed (${mapped[25]&1?"a cut tile the simulation holds at 4h has a liquid row":""}${mapped[25]===3?"; ":""}${mapped[25]&2?"a Neumann face is cut, V<1":""}); projection is incomplete`);
   this.bandTiles=mapped[22]!;this.bandResidual=new Float32Array(mapped.buffer)[24]!;
   // Only an accepted solve plans; a failed one threw above. Frame
   // frame+UNIFORM_MIXED_RECEIPT_RING encodes it, whenever this resolves.
   this.lagged.set(frame,{plan:{vCycles:mapped[20]!,fullCycles:mapped[21]!},bandTiles:mapped[22]!});
   const cells=this.ownership.capacity.lattice.dimensions.reduce((n,d)=>n*d,1);
   const orphanDustMass_cells=uniformMixedDustMass(accounting,10,p.orphanDust??0,cells);
   return {cycles:count,encoded,residual,converged:true,
    dustOwners:accounting[5]!+accounting[10]!,dustMass_cells:uniformMixedDustMass(accounting,5,p.dust,cells)+orphanDustMass_cells,
    orphanDustOwners:accounting[10]!,orphanDustMass_cells,bandTiles:mapped[22]!,bandCycles:mapped[26]!,bandResidual:this.bandResidual};
  }catch(error){this.failed=true;throw new Error(`Uniform mixed frame ${frame}: ${error instanceof Error?error.message:String(error)}`,{cause:error});}
 }
 private copyWhole(encoder:GPUCommandEncoder,from:GPUTexture,to:GPUTexture):void{
  encoder.copyTextureToTexture({texture:from},{texture:to},[from.width,from.height,from.depthOrArrayLayers]);
 }
 /** Takes the ungraded h/4h simulation layout; pressure stays all-4h. */
 updateLayout(layout:UniformMixedLayout):void{
  if(!this.ready||this.busy||this.failed)throw new Error("Ownership edits require a completed frame");
  this.geometryCurrent=false;this.solidWidthsStale=true;
  this.remap.apply(layout);
  {const e=this.device.createCommandEncoder({label:"Uniform resolve remapped phi"});this.phiResolve.encode(e,this.phiResolveGroups.phi);this.device.queue.submit([e.finish()]);}
  // Every pressure level is the fixed all-4h layout.
  this.pressureMatchesSimulation=layout.tiles.every(word=>mixedCellWidth(word)===4);
 }
 /** Adopt a GPU-built simulation generation (UniformMixedLayoutBuilder) at
  * the head of the next advance, in its encoder: no submit and no host
  * layout. before: encoded ahead of the adopt (a builder re-run). The host
  * cannot tell an unchanged generation, so every adopt remaps (an empty
  * worklist) and the next advance rebuilds geometry and phase and extends
  * once, on the adopted layout. Pressure
  * levels never change; dynamic ownership always solves split (an all-4h
  * simulation band is empty). */
 adoptBuiltLayout(built:UniformMixedBuiltOwnership,before?:(encoder:GPUCommandEncoder)=>void):void{
  if(!this.ready||this.busy||this.failed)throw new Error("Ownership edits require a completed frame");
  if(this.pendingAdopts.length)throw new Error("A GPU-built generation is already waiting for adoption");
  this.invalidateExtension();this.solidWidthsStale=true;
  this.pendingAdopts.push(encoder=>{before?.(encoder);this.remap.applyBuilt(encoder,built);this.phiResolve.encode(encoder,this.phiResolveGroups.phi);});
  this.pressureMatchesSimulation=false;
 }
 private pendingAdopts:((encoder:GPUCommandEncoder)=>void)[]=[];
 destroy():void{this.solid?.destroy();this.displacement?.destroy();this.pressureSchedule?.destroy();this.surfaceBand.destroy();this.band.destroy();this.plan.destroy();this.surface.destroy();this.hanging.destroy();this.remap.destroy();this.transport.destroy();this.cleanup.destroy();this.split.transfer.destroy();for(const l of this.levels)if(l.ownership!==this.ownership)l.ownership.destroy();for(const r of this.owned)r.destroy();}
}
