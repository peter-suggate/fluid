import {UniformMixedCleanup} from "./uniform-mixed-cleanup";
import {nextUniformPressureCorrection} from "./uniform-pressure-continuation";
import {UniformMixedFramePlan} from "./uniform-mixed-frame-plan";
import {UniformMixedRemap} from "./uniform-mixed-remap";
import {uniformMixedPressureLevel} from "./uniform-mixed-layout";
import type {UniformMixedLayout} from "./uniform-mixed-layout";
import type {UniformScratchArena} from "./uniform-scratch-arena";
import type {WebGPUUniformPressureMultigrid} from "./webgpu-uniform-pressure-multigrid";
import type {WebGPUUniformVelocityExtrapolator} from "./webgpu-uniform-velocity-extrapolation";
import {UniformMixedTransportStage} from "./uniform-mixed-transport";
import {UniformMixedOwnership} from "./uniform-mixed-ownership";
import {UniformMixedExtension} from "./uniform-mixed-extension";
import {UniformMixedHangingTaps, UniformMixedMomentumCache} from "./uniform-mixed-momentum-cache";
import {UniformMixedSurface} from "./uniform-mixed-surface";
import {UniformMixedSurfaceVolume} from "./uniform-mixed-surface-volume";
import {UniformMixedVertexTransfer} from "./uniform-mixed-vertex-transfer";
import {UniformMixedSurfaceGeometry} from "./uniform-mixed-surface-geometry";
import {UniformMixedSharpening} from "./uniform-mixed-sharpening";
import {UniformMixedMomentum,UNIFORM_MIXED_MOMENTUM_LIMITS} from "./uniform-mixed-momentum";
import {UniformMixedForces} from "./uniform-mixed-forces";
import {UniformMixedPressureAuthority} from "./uniform-mixed-pressure-authority";
import {UniformMixedPressureVelocity} from "./uniform-mixed-pressure-velocity";
import {UniformMixedPressureContinuation} from "./uniform-mixed-pressure-continuation";
import {UniformMixedPressureCycles,type UniformMixedPressureCycleLevel} from "./uniform-mixed-pressure-cycles";
import {UniformMixedPressureAcceptance} from "./uniform-mixed-pressure-acceptance";
import {planUniformMixedPressureMemory} from "./uniform-mixed-pressure-memory";
import {DEFAULT_UNIFORM_CM11A_SCHEDULE,type UniformCM11aSchedule} from "./pressure-policy";
import type {GPUTimestampPhase} from "../../core/performance-trace";
import {UNIFORM_ADVANCE_PHASE as A} from "./uniform-stages";
import {UNIFORM_VOLUME_PHASE as V} from "./uniform-volume-stages";
import {UniformMixedSolid,type UniformMixedSolidResources} from "./uniform-mixed-solid.wgsl";

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
 /** Static voxel/terrain/vessel solids, fixed for this frame's lifetime.
  * The host must keep every coarse owner a full tile away from a cut cell. */
 solid?:UniformMixedSolidResources;
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
export class UniformMixedFrame {
 readonly transport:UniformMixedTransportStage;
 get ownership(){return this.transport.ownership;}
 readonly levels:readonly UniformMixedPressureCycleLevel[];
 private readonly owned:(GPUTexture|GPUBuffer)[]=[];
 private readonly plan:UniformMixedFramePlan;
 private readonly cleanup:UniformMixedCleanup;
 private readonly cleanupGroups:readonly [GPUBindGroup,GPUBindGroup];
 private readonly remap:UniformMixedRemap;
 private readonly extension:UniformMixedExtension;
 private readonly cache:UniformMixedMomentumCache;
 private readonly hanging:UniformMixedHangingTaps;
 private readonly hangingGroup:GPUBindGroup;
 private readonly forceCacheGroup:GPUBindGroup;
 private readonly forceHangingGroup:GPUBindGroup;
 private readonly surface:UniformMixedSurface;
 private readonly surfaceVolume:UniformMixedSurfaceVolume;
 private readonly copyPhi:UniformMixedVertexTransfer;
 private readonly geometry:UniformMixedSurfaceGeometry;
 private readonly sharpen:UniformMixedSharpening;
 private readonly momentum:UniformMixedMomentum;
 private readonly forces:UniformMixedForces;
 private readonly authority:UniformMixedPressureAuthority;
 private readonly projection:UniformMixedPressureVelocity;
 private readonly continuation:UniformMixedPressureContinuation;
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
 private readonly readback:GPUBuffer;
 private readonly reductions:GPUBuffer;
 private continuationReady=false;
 private ready=false;
 private busy=false;
 private failed=false;
 get allocatedBytes():number{return this.plan.allocatedBytes+this.transport.allocatedBytes+this.remap.allocatedBytes+this.levels.slice(1).reduce((n,l)=>n+l.ownership.allocatedBytes,0)+this.owned.reduce((n,r)=>n+("size" in r?r.size:r.width*r.height*r.depthOrArrayLayers*16),0);}
 constructor(private readonly device:GPUDevice,layout:UniformMixedLayout,private readonly fields:UniformMixedFrameFields,openTop:boolean,private readonly schedule:UniformCM11aSchedule=DEFAULT_UNIFORM_CM11A_SCHEDULE){
  if(layout.cellCount!==layout.tiles.length*64)throw new Error("Unified frame must reserve its capacity with fine ownership");
  const f=fields;
  const buffer=(label:string,size:number,usage:number)=>{const b=device.createBuffer({label,size,usage});this.owned.push(b);return b;};
  const uniform=(name:string,size:number)=>buffer(`Uniform ${name} parameters`,size,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
  this.params={extension:uniform("extension",16),surface:uniform("surface",32),momentum:uniform("momentum",32),forces:uniform("forces",48),authority:uniform("authority",16),sharpen:uniform("sharpen",32),projection:uniform("projection",32),acceptance:uniform("acceptance",16)};
  this.state=buffer("Uniform pressure acceptance",32,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST);
  this.readback=buffer("Uniform pressure receipt and mass accounting",80,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
  this.reductions=buffer("Uniform dust accounting",48,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC);
  const caches=Array.from({length:3},(_,i)=>{const t=device.createTexture({label:`Uniform 4h sampling cache ${i}`,size:layout.lattice.dimensions.map(n=>n/4+2),dimension:"3d",format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING});this.owned.push(t);return t;});
  const solid=f.solid?new UniformMixedSolid(device,f.solid):undefined;
  this.transport=new UniformMixedTransportStage(device,layout,f.arena,f.volume,f.volumeScratch,f.departure,f.sourceParams,solid);
  const o=this.ownership;
  this.plan=new UniformMixedFramePlan(device,o,f.volume,f.phi,f.velocity,f.negative,f.velocityScratch,f.negativeScratch);
  this.cleanup=new UniformMixedCleanup(device,o,solid);
  this.cleanupGroups=[this.cleanup.bind(f.volume,f.volumeScratch,f.phi,this.params.sharpen,this.reductions),this.cleanup.bind(f.volumeScratch,f.volume,f.phi,this.params.sharpen,this.reductions)];
  this.remap=new UniformMixedRemap(device,o,{volume:f.volume,velocity:f.velocity,phi:f.phi,negative:f.negative},{volume:f.volumeScratch,velocity:f.velocityScratch,phi:f.phiScratch,negative:f.negativeScratch});
  const prefix=Math.min(...[f.pressure.pressure,f.pressure.rhs,f.pressure.minimum,f.pressure.phi,f.pressure.topology].map(v=>v.buffer!.offset??0));
  const memory=planUniformMixedPressureMemory(layout,prefix,f.conditioning.size,!!f.solid);
  const view=(r:{offset:number;size:number},external=false):GPUBufferBinding=>({buffer:external?f.conditioning:f.arena.buffer,...r});
  this.levels=memory.levels.map((r,i)=>({ownership:i===0?o:new UniformMixedOwnership(device,memory.layouts[i]!),pressure:view(r.pressure),rhs:[view(r.rhs[0]),view(r.rhs[1])],minimum:r.minimum.map(v=>view(v)),phi:view(r.phi,i===0),slopes:view(r.slopes),frozen:view(r.frozen),residual:view(r.residual),
   topology:!f.solid?undefined:i===0?{texture:f.departure}:{buffer:view(r.topology!)}}));
  const root=this.levels[0]!,last=this.levels[2]!;
  this.extension=new UniformMixedExtension(device,o,f.extension);
  this.extensionGroups=this.extension.bind({physical:f.velocity,phase:f.phase,negative:f.negative,output:f.velocityScratch,outputNegative:f.negativeScratch,scratch:{buffer:f.arena.buffer},params:this.params.extension});
  this.cache=new UniformMixedMomentumCache(device,o);
  const cacheFields={coarseExtended:caches[0]!,coarsePhysical:caches[1]!,coarseWeight:caches[2]!};
  this.cacheGroup=this.cache.bind({extended:f.velocityScratch,physical:f.velocity,phase:f.phase,negative:f.negativeScratch,...cacheFields});
  // Surface and momentum share hanging taps of velocityScratch: nothing
  // between the cache and forces writes it, negativeScratch or cache 0.
  this.hanging=new UniformMixedHangingTaps(device,o);this.hangingGroup=this.hanging.bind({extended:f.velocityScratch,negative:f.negativeScratch,coarse:caches[0]!});
  this.surface=new UniformMixedSurface(device,o,f.sourceParams,solid,true);
  const surfaceFields={velocity:f.velocityScratch,coarseVelocity:caches[0]!,volume:f.volume,negative:f.negativeScratch,departures:f.departure,params:this.params.surface,evidence:{buffer:f.arena.buffer}};
  this.surfaceGroups=[this.surface.bind({...surfaceFields,phi:f.phi,outputPhi:f.phiScratch}),this.surface.bind({...surfaceFields,phi:f.phiScratch,outputPhi:f.phi})];
  this.surfaceVolume=new UniformMixedSurfaceVolume(device,o,solid);
  this.surfaceVolumeGroup=this.surfaceVolume.bind(f.phi,f.volume,f.phiScratch,{buffer:f.arena.buffer});
  this.copyPhi=new UniformMixedVertexTransfer(device,o,f.phiScratch,f.phi,"restrict");
  this.geometry=new UniformMixedSurfaceGeometry(device,o,solid);this.geometryGroup=this.geometry.bind(f.phi,f.target,f.centerPhi);
  this.sharpen=new UniformMixedSharpening(device,o,solid);
  this.sharpenGroups=[this.sharpen.bind(f.volume,f.volumeScratch,f.phi,f.target,f.centerPhi,{buffer:f.arena.buffer,offset:0,size:f.arena.edgeBytes},this.params.sharpen,this.reductions),this.sharpen.bind(f.volumeScratch,f.volume,f.phi,f.target,f.centerPhi,{buffer:f.arena.buffer,offset:0,size:f.arena.edgeBytes},this.params.sharpen,this.reductions)];
  this.momentum=new UniformMixedMomentum(device,o,true,true,true);
  this.momentumGroup=this.momentum.bind({extended:f.velocityScratch,physical:f.velocity,phase:f.phase,volume:f.volume,centerPhi:f.centerPhi,predicted:f.velocity,reversed:f.velocity,negative:f.negativeScratch,predictedNegative:f.negative,reversedNegative:f.negative,output:f.departure,outputNegative:f.negativeDeparture,params:this.params.momentum,...cacheFields});
  // Momentum is the last reader of both sampling caches. Refill them from the
  // field viscosity samples (velocity with the advected negative planes).
  this.forceCacheGroup=this.cache.bind({extended:f.velocity,physical:f.velocity,phase:f.phase,negative:f.negativeDeparture,...cacheFields});
  this.forceHangingGroup=this.hanging.bind({extended:f.velocity,negative:f.negativeDeparture,coarse:caches[0]!});
  this.forces=new UniformMixedForces(device,o,true,f.sourceParams,solid,true);this.forceGroup=this.forces.bind({velocity:f.velocity,advected:f.departure,phi:f.phi,volume:f.volume,centerPhi:f.centerPhi,coarseVelocity:caches[0]!,negative:f.negativeDeparture,output:f.velocityScratch,outputNegative:f.negativeScratch,params:this.params.forces});
  this.authority=new UniformMixedPressureAuthority(device,o,solid);this.authorityGroup=this.authority.bind({centerPhi:f.centerPhi,volume:f.volume,targetFill:f.target,phi:root.phi!,phase:f.phase,correction:f.correction,scratch:root.frozen,params:this.params.authority});
  this.projection=new UniformMixedPressureVelocity(device,o,f.sourceParams,solid);
  // Level-0 pressure topology borrows the departure texture: free between
  // forces (its last reader) and projection.
  const topology=solid?f.departure:undefined;
  const common={velocity:f.velocityScratch,negative:{buffer:f.negativeScratch},phi:root.phi!,params:this.params.projection};
  this.rhsGroup=this.projection.bindRhs({...common,correction:f.correction,rhs:root.rhs[0],minimum:root.minimum![0]!,pressure:root.pressure,topology});
  this.projectionGroup=this.projection.bindProjection({...common,pressure:root.pressure,slopes:root.slopes,centerPhi:f.centerPhi,volume:f.volume,output:f.velocity,outputNegative:{buffer:f.negative},topology});
  this.continuation=new UniformMixedPressureContinuation(device,last.ownership,f.pressure,openTop,!!f.solid);
  const continuationGroups=last.rhs.map(rhs=>this.continuation.bind({pressure:last.pressure,rhs,minimum:last.minimum![0]!,phi:last.phi!,topology:last.topology&&"buffer" in last.topology?last.topology.buffer:undefined}));
  this.cycles=new UniformMixedPressureCycles(device,this.levels,view(memory.backup),(encoder,rhs,kind)=>{this.continuation.encode(encoder,continuationGroups[rhs===last.rhs[0]?0:1]!,f.uniformGroup,kind,!this.continuationReady);this.continuationReady=true;},[true,true],this.schedule,{openTop});
  this.acceptance=new UniformMixedPressureAcceptance(device,o);
  this.acceptanceGroup=this.acceptance.bind({residual:root.residual,state:this.state,params:this.params.acceptance});
 }
 async initialize():Promise<void>{
  for(const stage of [this.transport,this.plan,this.cleanup,this.remap,this.extension,this.cache,this.hanging,this.surface,this.surfaceVolume,this.copyPhi,this.geometry,this.sharpen,this.momentum,this.forces,this.authority,this.projection,this.continuation,this.cycles,this.acceptance])await stage.initialize();
  this.ready=true;
 }
 private write(p:UniformMixedFrameParameters):void{
  const h=this.ownership.layout.lattice.cellSize_m;
  const floats=(b:GPUBuffer,v:number[])=>this.device.queue.writeBuffer(b,0,new Float32Array(v));
  const flags=(b:GPUBuffer,v:number[])=>this.device.queue.writeBuffer(b,16,new Uint32Array(v));
  floats(this.params.extension,[...h,0]);floats(this.params.surface,[...h,p.dt]);flags(this.params.surface,[+p.openTop,+p.cubic,+p.drain,4]);
  floats(this.params.momentum,[...h,p.dt]);flags(this.params.momentum,[+p.openTop,0,0,UNIFORM_MIXED_MOMENTUM_LIMITS]);
  floats(this.params.forces,[...h,p.dt,p.gravity,p.density,p.viscosity,p.surfaceTension,+p.noSlip,+p.openTop,0,p.dust]);
  floats(this.params.authority,[p.dt,p.surfaceDeficitBalancing===false?-1:0,0,p.dust]);floats(this.params.sharpen,[p.sharpeningStrength,p.sharpeningDistance,p.dust,p.orphanDust??0,0,0,0,0]);
  floats(this.params.projection,[...h,p.dt,p.density,+p.openTop,0,p.dust]);floats(this.params.acceptance,[p.dt/p.density,p.pressureTolerance,0,0]);
 }
 private async receipt(encoder:GPUCommandEncoder,trace?:UniformMixedFrameTrace):Promise<Uint32Array>{
  encoder.copyBufferToBuffer(this.state,0,this.readback,0,32);trace?.submit(encoder,this.state);this.device.queue.submit([encoder.finish()]);
  await this.readback.mapAsync(GPUMapMode.READ);const state=new Uint32Array(this.readback.getMappedRange(),0,8).slice();this.readback.unmap();return state;
 }
 /** Discarded mass uses native sixty-fourths-of-threshold counters, weighted
  * by owner volume. It is a quantized lower bound; counts name owners, not
  * fine cells. Counters are reset once per frame, before transport cleanup. */
 async advance(p:UniformMixedFrameParameters,trace?:UniformMixedFrameTrace):Promise<{cycles:number;residual:number;converged:boolean;dustOwners:number;dustMass_cells:number;orphanDustOwners:number;orphanDustMass_cells:number}>{
  if(!this.ready||this.busy||this.failed)throw new Error("Unified frame is not ready for an advance");
  this.busy=true;
  const releases:(()=>void)[]=[];
  try{
   for(const level of this.levels)releases.push(level.ownership.acquireFrame());
   const makeEncoder=()=>{const raw=this.device.createCommandEncoder({label:"Uniform owner-driven frame"});return trace?.instrument(raw)??raw;};
   this.write(p);let encoder=makeEncoder();
   encoder.clearBuffer(this.reductions);
   this.plan.encode(encoder,p.supportPolicy,p.dt);this.geometry.encode(encoder,this.geometryGroup);this.authority.encode(encoder,this.authorityGroup);
   trace?.phase(encoder,A.extensionAuthority);
   this.extension.encode(encoder,this.extensionGroups,p.extensionSweeps??2);this.plan.encodeCertificate(encoder);this.cache.encode(encoder,this.cacheGroup);this.hanging.encode(encoder,this.hangingGroup);
   trace?.phase(encoder,A.extensionHierarchy);
   this.surface.encode(encoder,"advect",this.surfaceGroups[0]);this.surface.encode(encoder,"traceCells",this.surfaceGroups[0]);
   if(p.redistance!==false)this.surface.encode(encoder,"redistance",this.surfaceGroups[1]);else this.copyPhi.encode(encoder);
   trace?.phase(encoder,V.phi);
   this.transport.encodeCopy(encoder);this.transport.encodeTransport(encoder);
   trace?.phase(encoder,V.coupling);
   if(p.dust>0)this.cleanup.encode(encoder,this.cleanupGroups);
   if(p.totalSurfaceVolume!==false){this.surfaceVolume.encode(encoder,this.surfaceVolumeGroup);this.copyPhi.encode(encoder);}
   this.geometry.encode(encoder,this.geometryGroup);
   trace?.phase(encoder,V.gather);
   if(p.sharpening!==false){this.sharpen.encodeGeometry(encoder,this.sharpenGroups[0]);for(let i=0;i<8;i++)this.sharpen.encodeSweep(encoder,this.sharpenGroups[i%2]!,false);}
   trace?.phase(encoder,V.sharpen);
   this.momentum.encode(encoder,this.momentumGroup);
   this.cache.encode(encoder,this.forceCacheGroup);this.hanging.encode(encoder,this.forceHangingGroup);this.forces.encode(encoder,this.forceGroup);
   trace?.phase(encoder,A.advectionCorrection);
   this.continuationReady=false;
   this.authority.encode(encoder,this.authorityGroup);this.projection.encode(encoder,"rhs",this.rhsGroup);this.cycles.encodeSurfaceRestriction(encoder);
   this.cycles.encodeMeasure(encoder);this.acceptance.encode(encoder,this.acceptanceGroup,this.state,"initial");
   trace?.phase(encoder,A.pressureSetup);
   let state:Uint32Array=new Uint32Array(8),count=0;
   const schedule=this.schedule;
   let cycle=0,accuracy=1,previous=Infinity;
   this.fields.pressure.setCoarseAccuracy(accuracy);
   while(cycle<schedule.fullCycles+schedule.vCycles){
    if(cycle<schedule.vCycles)this.cycles.encodeVCycle(encoder);else this.cycles.encodeFullCycle(encoder);
    this.cycles.encodeMeasure(encoder);this.acceptance.encode(encoder,this.acceptanceGroup,this.state,"cycle");count++;
    trace?.phase(encoder,cycle<schedule.vCycles?A.pressureVCycles:A.pressureFullCycles);
    state=await this.receipt(encoder,trace);encoder=makeEncoder();
    if(state[5]!==0||state[4]!==0)break;
    const values=new Float32Array(state.buffer),residual=values[1]!;
    if(count===1)previous=values[2]!;
    const next=nextUniformPressureCorrection(cycle,schedule.vCycles,schedule.fullCycles,residual,previous,accuracy);
    if(!next)break;
    previous=residual;cycle=next.cycle;accuracy=next.accuracy;this.fields.pressure.setCoarseAccuracy(accuracy);
   }
   const residual=new Float32Array(state.buffer)[1]!;
   if(state[4]!==0||state[5]===0||!Number.isFinite(residual)||residual<0||residual>p.pressureTolerance){
    throw new Error(`Uniform mixed pressure ${state[4]!==0?"rejected a non-improving cycle":"did not converge"}: candidate ${new Float32Array(state.buffer)[0]}, accepted ${residual}, tolerance ${p.pressureTolerance}, ${count} cycles; projection withheld${typeof process!=="undefined"&&process.env.FLUID_MIXED_HOST_DIAGNOSTICS?`; params ${JSON.stringify(p)}; receipt ${[...state]}`:""}`);
   }
   // Rebuild slopes from the converged iterate before projection.
   this.cycles.encodeMeasure(encoder);trace?.phase(encoder,A.pressureFinish);
   this.projection.encode(encoder,"project",this.projectionGroup);
   trace?.phase(encoder,A.pressureProjection);
   encoder.copyBufferToBuffer(this.reductions,0,this.readback,32,48);
   trace?.submit(encoder,this.fields.negative);this.device.queue.submit([encoder.finish()]);trace?.submitted();await this.readback.mapAsync(GPUMapMode.READ);
   const accounting=new Uint32Array(this.readback.getMappedRange(),32,12).slice();this.readback.unmap();
   const orphanDustMass_cells=accounting[11]!*(p.orphanDust??0)/64;
   return {cycles:count,residual:new Float32Array(state.buffer)[1]!,converged:state[5]!==0,
    dustOwners:accounting[5]!+accounting[10]!,dustMass_cells:accounting[6]!*p.dust/64+orphanDustMass_cells,
    orphanDustOwners:accounting[10]!,orphanDustMass_cells};
  }catch(error){trace?.abort();this.failed=true;throw error;}finally{for(const release of releases)release();this.busy=false;}
 }
 updateLayout(layout:UniformMixedLayout):void{
  if(!this.ready||this.busy||this.failed)throw new Error("Ownership edits require a completed frame");
  this.remap.apply(layout);
  // Pressure is scratch for the next solve, not a transported state variable.
  // Retire its old indexing before paused diagnostic consumers see new owners.
  const clear=this.device.createCommandEncoder();
  for(const view of [this.levels[0]!.pressure,this.levels[0]!.phi!])clear.clearBuffer(view.buffer,view.offset??0,view.size);
  this.device.queue.submit([clear.finish()]);
  this.levels[1]!.ownership.update(uniformMixedPressureLevel(layout,2));
  this.levels[2]!.ownership.update(uniformMixedPressureLevel(layout,4));
 }
 destroy():void{this.plan.destroy();this.remap.destroy();this.transport.destroy();for(const l of this.levels.slice(1))l.ownership.destroy();for(const r of this.owned)r.destroy();}
}
