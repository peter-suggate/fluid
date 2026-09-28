import { uniformMixedPressureStorage } from "./uniform-mixed-pressure-boundary.wgsl";
import { DEFAULT_UNIFORM_CM11A_SCHEDULE, type UniformCM11aSchedule } from "./pressure-policy";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { UniformMixedPressureBoundsStage, UniformMixedPressureLevelStage, UniformMixedPressurePasses, UniformMixedPressureTransferStage,
  type UniformMixedPressureEntry, type UniformMixedPressureBoundsEntry, type UniformMixedPressureTransferEntry } from "./uniform-mixed-pressure-stage";
import type { UniformMixedPressureTopology } from "./uniform-mixed-pressure-topology.wgsl";

export interface UniformMixedPressureCycleLevel {
  ownership: UniformMixedOwnership;
  pressure: GPUBufferBinding;
  slopes: GPUBufferBinding;
  rhs: readonly [GPUBufferBinding, GPUBufferBinding];
  frozen: GPUBufferBinding;
  residual: GPUBufferBinding;
  /** Original/shifted at L0; one reusable correction minimum below L0. */
  minimum?: readonly GPUBufferBinding[];
  phi?: GPUBufferBinding;
  /** Static solids: both all-4h levels carry buffer records (level 0 the
   * static solid record, level 1 its restriction). */
  topology?: UniformMixedPressureTopology;
}

/** Native CM11a cycle traversal through one all-4h level (the band
 * pressure's bulk solve) whose coarse correction the native hierarchy solves
 * from 4h, entered at the second all-4h level. All fields are borrowed. The
 * callback continues the existing uniform hierarchy at 4h; a large 4h grid
 * must not be dispatched as one coarsest workgroup. Convergence validation
 * and field publication remain the native host's job. */
export class UniformMixedPressureCycles {
  readonly allocatedBytes = 0;
  private stage0?: UniformMixedPressureLevelStage;
  private transfer0?: UniformMixedPressureTransferStage;
  private bound0?: UniformMixedPressureBoundsStage;
  private readonly groups = new Map<string, GPUBindGroup>();
  private readonly bufferIds = new Map<GPUBuffer, number>();
  private shifted = false;
  private readonly constrained: boolean;
  private readonly surface: boolean;
  private readonly solid: boolean;
  constructor(private readonly device: GPUDevice, readonly levels: readonly UniformMixedPressureCycleLevel[],
    private readonly backup: GPUBufferBinding,
    private readonly encodeUniformHierarchy: (encoder: GPUCommandEncoder, rhs: GPUBufferBinding, kind: "v" | "full") => void,
    private readonly schedule: UniformCM11aSchedule = DEFAULT_UNIFORM_CM11A_SCHEDULE,
    private readonly boundary?: {openTop:boolean}) {
    if(levels.length!==2)throw new Error("Mixed pressure traversal requires two all-4h levels");
    this.constrained=!!levels[0]!.minimum; this.surface=!!levels[0]!.phi; this.solid=!!levels[0]!.topology;
    if(levels.some(l=>!!l.topology!==this.solid||(l.topology&&"texture" in l.topology)))
      throw new Error("Solid pressure topology must be buffer records at both all-4h levels");
    levels.forEach((l,i)=>{
      if(l.ownership.layout.tiles.length!==levels[0]!.ownership.layout.tiles.length||l.ownership.layout.coarseTiles.length!==l.ownership.layout.tiles.length)
        throw new Error("Mixed pressure levels must be all-4h");
      if(!!l.minimum!==this.constrained||!!l.phi!==this.surface||(l.minimum&&l.minimum.length!==(i===0?2:1)))
        throw new Error("Mixed pressure level fields disagree");
    });
  }
  async initialize(): Promise<void> {
    const kind=this.solid?"buffer" as const:undefined,[fine,coarse]=this.levels as [UniformMixedPressureCycleLevel,UniformMixedPressureCycleLevel];
    this.stage0=new UniformMixedPressureLevelStage(this.device,fine.ownership,this.constrained,this.surface,this.boundary,kind);await this.stage0.initialize();
    // Native host's paper-destination >= M-C phi policy.
    this.transfer0=new UniformMixedPressureTransferStage(this.device,fine.ownership,coarse.ownership,true,!!this.boundary,kind);await this.transfer0.initialize();
    if(this.constrained){this.bound0=new UniformMixedPressureBoundsStage(this.device,fine.ownership,coarse.ownership,!!this.boundary);await this.bound0.initialize();}
  }
  private get stage(): UniformMixedPressureLevelStage {if(!this.stage0)throw new Error("Mixed pressure cycles are not initialized");return this.stage0;}
  private key(view: GPUBufferBinding): string {
    if(!this.bufferIds.has(view.buffer))this.bufferIds.set(view.buffer,this.bufferIds.size);
    return `${this.bufferIds.get(view.buffer)}:${view.offset??0}:${view.size??view.buffer.size-(view.offset??0)}`;
  }
  private minimum(level: number): GPUBufferBinding | undefined {return this.levels[level]!.minimum?.[level===0&&this.shifted?1:0];}
  private group(level: number,rhs: GPUBufferBinding,result: GPUBufferBinding): GPUBindGroup {
    const l=this.levels[level]!,minimum=this.minimum(level),key=`level:${level}:${this.key(rhs)}:${this.key(result)}:${this.shifted}`;
    let group=this.groups.get(key);
    if(!group){group=this.stage.bind({pressure:l.pressure,slopes:l.slopes,rhs,frozen:l.frozen,result,minimum,phi:l.phi,topology:l.topology});this.groups.set(key,group);}
    return group;
  }
  private encodeEntry(encoder: UniformMixedPressurePasses,level: number,entry: UniformMixedPressureEntry,rhs: GPUBufferBinding,result=this.levels[level]!.residual): void {
    this.stage.encode(encoder,entry,this.group(level,rhs,result));
  }
  private smooth(encoder: UniformMixedPressurePasses,level: number,rhs: GPUBufferBinding,count: number): void {
    const group=this.group(level,rhs,this.levels[level]!.residual);
    // The first sweep of a visit projects every air owner onto p_min; air is
    // then a fixed point, so later sweeps revisit only liquid tiles.
    for(let i=0;i<count;i++)this.stage.encodeSweep(encoder,group,this.surface&&i>0);
  }
  private transfer(encoder: UniformMixedPressurePasses,level: number,entry: UniformMixedPressureTransferEntry,source: GPUBufferBinding,destination: GPUBufferBinding): void {
    const key=`transfer:${level}:${entry}:${this.key(source)}:${this.key(destination)}`;
    let group=this.groups.get(key);
    if(!group){group=this.transfer0!.bind(entry,source,destination);this.groups.set(key,group);}
    let topology:GPUBindGroup|undefined;
    if(this.solid&&entry==="restrictSurfacePhi"){
      const topologyKey=`topology:${level}`;topology=this.groups.get(topologyKey);
      if(!topology){const next=this.levels[level+1]!.topology!;if(!("buffer" in next))throw new Error("Mixed coarse pressure topology must be an arena record");
        topology=this.transfer0!.bindTopology(this.levels[level]!.topology!,next.buffer);this.groups.set(topologyKey,topology);}
    }
    this.transfer0!.encode(encoder,entry,group,topology);
  }
  private bound(encoder: UniformMixedPressurePasses,level: number,entry: UniformMixedPressureBoundsEntry,destination: GPUBufferBinding): void {
    const source=this.minimum(level)!,key=`bound:${level}:${entry}:${this.key(source)}:${this.key(destination)}`;
    let group=this.groups.get(key);
    if(!group){group=this.bound0!.bind(entry,source,this.levels[level]!.pressure,destination);this.groups.set(key,group);}
    this.bound0!.encode(encoder,entry,group);
  }
  /** Zero the native level's correction. It is entered between passes, so
   * its clear splits none. */
  private clearPressure(encoder: UniformMixedPressurePasses): void {
    const l=this.levels[1]!;encoder.commands.clearBuffer(l.pressure.buffer,l.pressure.offset??0,(this.boundary?uniformMixedPressureStorage(l.ownership.layout).count:l.ownership.layout.cellCount)*4);
  }
  /** Each public entry batches its dispatches into as few passes as its
   * encoder commands allow; the label names the whole batch. */
  private batch(encoder: GPUCommandEncoder,label: string,body:(passes:UniformMixedPressurePasses)=>void): void {
    const passes=new UniformMixedPressurePasses(encoder,`Uniform mixed pressure ${label}`);body(passes);passes.end();
  }
  encodeSurfaceRestriction(encoder: GPUCommandEncoder): void {this.batch(encoder,"setup",passes=>this.surfaceRestriction(passes));}
  /** A level's phi for a transfer: its owner phi alone. Level 0's buffer
   * topology follows it in the same buffer, and a transfer binds that record
   * separately. */
  private ownerPhi(level: number): GPUBufferBinding {
    const phi=this.levels[level]!.phi!;
    return {buffer:phi.buffer,offset:phi.offset??0,size:4*this.levels[level]!.ownership.layout.cellCount};
  }
  private surfaceRestriction(encoder: UniformMixedPressurePasses): void {
    // The native continuation extends the 4h phi itself.
    if(this.surface)this.transfer(encoder,0,"restrictSurfacePhi",this.ownerPhi(0),this.ownerPhi(1));
    // Seam records freeze this frame's phi/solid coefficients for every sweep.
    this.stage.encodeRecords(encoder,this.group(0,this.levels[0]!.rhs[0],this.levels[0]!.residual));
  }
  private residual(encoder: UniformMixedPressurePasses,level: number,rhs: GPUBufferBinding,output=this.levels[level]!.residual): void {
    this.stage.encodeRecordEntry(encoder,"residual",this.group(level,rhs,output));
  }
  encodeMeasure(encoder: GPUCommandEncoder): void {
    this.stage.encodeRecordEntry(encoder,"measure",this.group(0,this.levels[0]!.rhs[0],this.levels[0]!.residual));
  }
  encodeVCycle(encoder: GPUCommandEncoder): void {this.batch(encoder,"V-cycle",passes=>this.vCycle(passes,this.levels[0]!.rhs[0]));}
  private vCycle(encoder: UniformMixedPressurePasses,rhs: GPUBufferBinding): void {
    const l=this.levels[0]!,next=this.levels[1]!;
    this.smooth(encoder,0,rhs,this.schedule.preSweeps);this.residual(encoder,0,rhs);
    this.transfer(encoder,0,"restrictValues",l.residual,next.rhs[0]);
    if(this.constrained)this.bound(encoder,0,"downsampleSubtract",next.minimum![0]!);
    // Reads neither field above; last, so the native 4h clear ends the pass
    // the hierarchy would end anyway.
    this.clearPressure(encoder);
    this.encodeUniformHierarchy(encoder.commands,next.rhs[0],"v");this.transfer(encoder,0,"prolongAdd",next.pressure,l.pressure);
    this.smooth(encoder,0,rhs,this.schedule.postSweeps);
  }
  encodeFullCycle(encoder: GPUCommandEncoder): void {this.batch(encoder,"Full-cycle",passes=>this.fullCycle(passes));}
  private fullCycle(encoder: UniformMixedPressurePasses): void {
    const fine=this.levels[0]!;
    // A shader copy supports disjoint views of the same arena, unlike a
    // buffer-to-buffer copy whose source and destination must differ.
    this.encodeEntry(encoder,0,"saveBackup",fine.rhs[0],this.backup);
    if(this.constrained){this.bound(encoder,0,"shiftMinimum",fine.minimum![1]!);this.shifted=true;}
    const coarse=this.levels[1]!;
    this.residual(encoder,0,fine.rhs[0],fine.rhs[1]);
    this.transfer(encoder,0,"restrictValues",fine.rhs[1],coarse.rhs[1]);
    if(this.constrained)this.bound(encoder,0,"downsampleMinimum",coarse.minimum![0]!);
    this.clearPressure(encoder);this.encodeUniformHierarchy(encoder.commands,coarse.rhs[1],"full");
    this.transfer(encoder,0,"prolongAssign",coarse.pressure,fine.pressure);
    this.vCycle(encoder,fine.rhs[1]);
    this.encodeEntry(encoder,0,"addBackup",this.backup);this.shifted=false;
  }
}
