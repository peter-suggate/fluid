import { uniformMixedPressureStorage } from "./uniform-mixed-pressure-boundary.wgsl";
import { mixedCellWidth } from "./uniform-mixed-layout";
import { DEFAULT_UNIFORM_CM11A_SCHEDULE, type UniformCM11aSchedule } from "./pressure-policy";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { UniformMixedPressureBoundsStage, UniformMixedPressureLevelStage, UniformMixedPressureTransferStage,
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
  /** Static solids: the native texture at L0, arena records below it. */
  topology?: UniformMixedPressureTopology;
}

/** Native CM11a cycle traversal through h/2h/4h ownership. All fields are
 * borrowed. The final callback continues the existing uniform hierarchy at
 * 4h; a large 4h grid must not be dispatched as one coarsest workgroup.
 * Convergence validation and field publication remain the native host's job. */
export class UniformMixedPressureCycles {
  readonly allocatedBytes = 0;
  private readonly stages: UniformMixedPressureLevelStage[] = [];
  private readonly transfers: UniformMixedPressureTransferStage[] = [];
  private readonly bounds: UniformMixedPressureBoundsStage[] = [];
  private readonly groups = new Map<string, GPUBindGroup>();
  private readonly bufferIds = new Map<GPUBuffer, number>();
  private shifted = false;
  private readonly constrained: boolean;
  private readonly surface: boolean;
  private readonly solid: boolean;
  constructor(private readonly device: GPUDevice, readonly levels: readonly UniformMixedPressureCycleLevel[],
    private readonly backup: GPUBufferBinding,
    private readonly encodeUniformHierarchy: (encoder: GPUCommandEncoder, rhs: GPUBufferBinding, kind: "v" | "full") => void,
    /** Native host's paper-destination >= M-C policy for each transfer. */
    private readonly preferPositivePhi: readonly boolean[],
    private readonly schedule: UniformCM11aSchedule = DEFAULT_UNIFORM_CM11A_SCHEDULE,
    private readonly boundary?: {openTop:boolean}) {
    if(levels.length !== 3 || preferPositivePhi.length !== 2)throw new Error("Mixed pressure traversal requires h/2h/4h levels");
    this.constrained=!!levels[0]!.minimum; this.surface=!!levels[0]!.phi; this.solid=!!levels[0]!.topology;
    if(levels.some((l,i)=>!!l.topology!==this.solid||(l.topology&&("texture" in l.topology)!==(i===0))))
      throw new Error("Mixed solid pressure topology must be the native texture at h and arena records at 2h/4h");
    levels.forEach((l,i)=>{
      if(l.ownership.layout.tiles.length!==levels[0]!.ownership.layout.tiles.length
        || l.ownership.layout.tiles.some((word,tile)=>mixedCellWidth(word)!==Math.max(mixedCellWidth(levels[0]!.ownership.layout.tiles[tile]!),1<<i)))
        throw new Error("Mixed pressure hierarchy must raise its floor from h to 2h to 4h");
      if(!!l.minimum!==this.constrained||!!l.phi!==this.surface||(l.minimum&&l.minimum.length!==(i===0?2:1)))
        throw new Error("Mixed pressure level fields disagree");
    });
  }
  async initialize(): Promise<void> {
    for(let i=0;i<2;i++) {
      const kind=this.solid?(i===0?"texture" as const:"buffer" as const):undefined;
      const stage=new UniformMixedPressureLevelStage(this.device,this.levels[i]!.ownership,this.constrained,this.surface,this.boundary,kind);
      await stage.initialize();this.stages.push(stage);
      const transfer=new UniformMixedPressureTransferStage(this.device,this.levels[i]!.ownership,this.levels[i+1]!.ownership,this.preferPositivePhi[i],!!this.boundary,kind);
      await transfer.initialize();this.transfers.push(transfer);
      if(this.constrained){const bound=new UniformMixedPressureBoundsStage(this.device,this.levels[i]!.ownership,this.levels[i+1]!.ownership,!!this.boundary);await bound.initialize();this.bounds.push(bound);}
    }
  }
  private key(view: GPUBufferBinding): string {
    if(!this.bufferIds.has(view.buffer))this.bufferIds.set(view.buffer,this.bufferIds.size);
    return `${this.bufferIds.get(view.buffer)}:${view.offset??0}:${view.size??view.buffer.size-(view.offset??0)}`;
  }
  private minimum(level: number): GPUBufferBinding | undefined {return this.levels[level]!.minimum?.[level===0&&this.shifted?1:0];}
  private group(level: number,rhs: GPUBufferBinding,result: GPUBufferBinding): GPUBindGroup {
    const l=this.levels[level]!,minimum=this.minimum(level),key=`level:${level}:${this.key(rhs)}:${this.key(result)}:${this.shifted}`;
    let group=this.groups.get(key);
    if(!group){group=this.stages[level]!.bind({pressure:l.pressure,slopes:l.slopes,rhs,frozen:l.frozen,result,minimum,phi:l.phi,topology:l.topology});this.groups.set(key,group);}
    return group;
  }
  private stage(encoder: GPUCommandEncoder,level: number,entry: UniformMixedPressureEntry,rhs: GPUBufferBinding,result=this.levels[level]!.residual): void {
    this.stages[level]!.encode(encoder,entry,this.group(level,rhs,result));
  }
  private smooth(encoder: GPUCommandEncoder,level: number,rhs: GPUBufferBinding,count: number): void {
    const group=this.group(level,rhs,this.levels[level]!.residual);
    for(let i=0;i<count;i++)this.stages[level]!.encodeSweep(encoder,group);
  }
  private transfer(encoder: GPUCommandEncoder,level: number,entry: UniformMixedPressureTransferEntry,source: GPUBufferBinding,destination: GPUBufferBinding): void {
    const key=`transfer:${level}:${entry}:${this.key(source)}:${this.key(destination)}`;
    let group=this.groups.get(key);
    if(!group){group=this.transfers[level]!.bind(entry,source,destination);this.groups.set(key,group);}
    let topology:GPUBindGroup|undefined;
    if(this.solid&&(entry==="restrictSurfacePhi"||entry==="extrapolateSurfacePhi")){
      const topologyKey=`topology:${level}`;topology=this.groups.get(topologyKey);
      if(!topology){const next=this.levels[level+1]!.topology!;if(!("buffer" in next))throw new Error("Mixed coarse pressure topology must be an arena record");
        topology=this.transfers[level]!.bindTopology(this.levels[level]!.topology!,next.buffer);this.groups.set(topologyKey,topology);}
    }
    this.transfers[level]!.encode(encoder,entry,group,topology);
  }
  private bound(encoder: GPUCommandEncoder,level: number,entry: UniformMixedPressureBoundsEntry,destination: GPUBufferBinding): void {
    const source=this.minimum(level)!,key=`bound:${level}:${entry}:${this.key(source)}:${this.key(destination)}`;
    let group=this.groups.get(key);
    if(!group){group=this.bounds[level]!.bind(entry,source,this.levels[level]!.pressure,destination);this.groups.set(key,group);}
    this.bounds[level]!.encode(encoder,entry,group);
  }
  private clearPressure(encoder: GPUCommandEncoder,level: number): void {
    const l=this.levels[level]!;encoder.clearBuffer(l.pressure.buffer,l.pressure.offset??0,(this.boundary?uniformMixedPressureStorage(l.ownership.layout).count:l.ownership.layout.cellCount)*4);
  }
  encodeSurfaceRestriction(encoder: GPUCommandEncoder): void {
    if(this.surface)for(let level=0;level<2;level++)this.transfer(encoder,level,"restrictSurfacePhi",this.levels[level]!.phi!,this.levels[level+1]!.phi!);
    // Native order: the raw phi/V pyramid first, then one-cell continuation
    // per level. L0 is pressurePhi (already continued) and the native
    // continuation extends 4h itself, so only 2h is extended here.
    if(this.solid)this.transfer(encoder,0,"extrapolateSurfacePhi",this.levels[0]!.phi!,this.levels[1]!.phi!);
    // Seam records freeze this frame's phi/solid coefficients for every sweep.
    for(let level=0;level<2;level++)this.stages[level]!.encodeRecords(encoder,this.group(level,this.levels[level]!.rhs[0],this.levels[level]!.residual));
  }
  encodeResidual(encoder: GPUCommandEncoder,level=0,rhs=this.levels[level]!.rhs[0],output=this.levels[level]!.residual): void {
    this.stages[level]!.encodeRecordEntry(encoder,"residual",this.group(level,rhs,output));
  }
  encodeMeasure(encoder: GPUCommandEncoder): void {
    this.stages[0]!.encodeRecordEntry(encoder,"measure",this.group(0,this.levels[0]!.rhs[0],this.levels[0]!.residual));
  }
  encodeVCycle(encoder: GPUCommandEncoder,level=0,rhs=this.levels[level]!.rhs[0]): void {
    if(level===2){this.encodeUniformHierarchy(encoder,rhs,"v");return;}
    const l=this.levels[level]!,next=this.levels[level+1]!;
    this.smooth(encoder,level,rhs,this.schedule.preSweeps);this.encodeResidual(encoder,level,rhs);
    this.transfer(encoder,level,"restrictValues",l.residual,next.rhs[0]);this.clearPressure(encoder,level+1);
    if(this.constrained)this.bound(encoder,level,"downsampleSubtract",next.minimum![0]!);
    this.encodeVCycle(encoder,level+1,next.rhs[0]);this.transfer(encoder,level,"prolongAdd",next.pressure,l.pressure);
    this.smooth(encoder,level,rhs,this.schedule.postSweeps);
  }
  encodeFullCycle(encoder: GPUCommandEncoder): void {
    const fine=this.levels[0]!;
    // A shader copy supports disjoint views of the same arena, unlike a
    // buffer-to-buffer copy whose source and destination must differ.
    this.stage(encoder,0,"saveBackup",fine.rhs[0],this.backup);
    if(this.constrained){this.bound(encoder,0,"shiftMinimum",fine.minimum![1]!);this.shifted=true;}
    this.encodeResidual(encoder,0,fine.rhs[0],fine.rhs[1]);
    for(let level=0;level<2;level++){
      this.transfer(encoder,level,"restrictValues",this.levels[level]!.rhs[1],this.levels[level+1]!.rhs[1]);
      if(this.constrained)this.bound(encoder,level,"downsampleMinimum",this.levels[level+1]!.minimum![0]!);
    }
    this.clearPressure(encoder,2);this.encodeUniformHierarchy(encoder,this.levels[2]!.rhs[1],"full");
    for(let level=1;level>=0;level--){
      this.transfer(encoder,level,"prolongAssign",this.levels[level+1]!.pressure,this.levels[level]!.pressure);
      this.encodeVCycle(encoder,level,this.levels[level]!.rhs[1]);
    }
    this.stage(encoder,0,"addBackup",this.backup);this.shifted=false;
  }
}
