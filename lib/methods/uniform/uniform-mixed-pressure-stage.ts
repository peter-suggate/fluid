import { uniformMixedPressureBoundaryIndexWGSL, uniformMixedPressureBoundaryWGSL, uniformMixedPressureBoundaryLoop, uniformMixedPressureStorage } from "./uniform-mixed-pressure-boundary.wgsl";
import { uniformMixedPressureSurfaceWGSL } from "./uniform-mixed-pressure-surface.wgsl";
import { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedPressureReconstructionSource } from "./uniform-mixed-pressure-reconstruction.wgsl";
import { uniformMixedPressureOperatorSource } from "./uniform-mixed-pressure-operator.wgsl";
import { uniformMixedPressureTopologyWGSL, type UniformMixedPressureTopology } from "./uniform-mixed-pressure-topology.wgsl";
import { UNIFORM_MIXED_PRESSURE_RECORD_CHUNK, uniformMixedPressureRecordsSource } from "./uniform-mixed-pressure-records.wgsl";

const recordEntries = ["buildRecords", "linkRecords", "reconstructRecords", "freezeRecords", "residualRecords", "measureRecords"] as const;
type RecordEntry = typeof recordEntries[number];
const entries = ["addBackup", "saveBackup"] as const;
export type UniformMixedPressureEntry = typeof entries[number];
export type UniformMixedPressureTransferEntry = "restrictValues" | "restrictSurfacePhi" | "prolongAssign" | "prolongAdd";
export interface UniformMixedPressureFields {
  pressure: GPUBufferBinding;
  slopes: GPUBufferBinding;
  rhs: GPUBufferBinding;
  frozen: GPUBufferBinding;
  result: GPUBufferBinding;
  minimum?: GPUBufferBinding;
  phi?: GPUBufferBinding;
  /** Static-solid scenes: this level's CM11a (open, V) record. */
  topology?: UniformMixedPressureTopology;
}
/** One compute pass shared by consecutive mixed pressure dispatches. A
 * dispatch is its own WebGPU usage scope, so dependent dispatches may share a
 * pass; encoder commands (copies, clears, foreign passes) end it first. */
export class UniformMixedPressurePasses {
  private open?: GPUComputePassEncoder;
  constructor(private readonly encoder: GPUCommandEncoder, readonly label: string) {}
  get pass(): GPUComputePassEncoder { return this.open ??= this.encoder.beginComputePass({ label: this.label }); }
  /** The encoder with no pass open. */
  get commands(): GPUCommandEncoder { this.end(); return this.encoder; }
  end(): void { this.open?.end(); this.open = undefined; }
}
/** A bare encoder gives each stage call its own labelled pass. */
export type UniformMixedPressureTarget = GPUCommandEncoder | UniformMixedPressurePasses;
function beginPass(target: UniformMixedPressureTarget, label: string): { pass: GPUComputePassEncoder; end(): void } {
  if (target instanceof UniformMixedPressurePasses) return { pass: target.pass, end: () => {} };
  const pass = target.beginComputePass({ label }); return { pass, end: () => pass.end() };
}
async function checkedModule(device: GPUDevice, code: string): Promise<GPUShaderModule> {
  const module = device.createShaderModule({ label: "Uniform mixed pressure operators", code });
  const info = await module.getCompilationInfo(), errors = info.messages.filter(m => m.type === "error");
  if (errors.length) throw new Error(errors.map(m => `${m.lineNum}: ${m.message}`).join("\n"));
  return module;
}
// All field bindings use storage/read_write, even logically read-only inputs:
// WebGPU tracks buffer usage for the whole arena, not for each disjoint range.
// Mixing read-only and writable storage aliases would invalidate the pass.

/** Limit every view to its actual field. Disjoint views of one existing arena
 * are valid; overlapping writable fields are not. No field allocation occurs. */
/** Explicit arena views reserve capacity for future ownership generations.
 * Do not shrink them to the first active layout: cached groups survive edits. */
function views(fields: readonly GPUBufferBinding[], sizes: readonly number[]): GPUBufferBinding[] {
  const result = fields.map((field, i) => {
    const offset = field.offset ?? 0, available = field.size ?? field.buffer.size - offset, required = sizes[i]!, size = field.size ?? required;
    if (offset < 0 || !Number.isSafeInteger(offset) || !Number.isSafeInteger(available) || available < required || offset + available > field.buffer.size
      || !(field.buffer.usage & GPUBufferUsage.STORAGE)) throw new Error("Invalid borrowed mixed pressure field");
    return { buffer: field.buffer, offset, size };
  });
  for (let i = 0; i < result.length; i++) for (let j = 0; j < i; j++) {
    const a = result[i]!, b = result[j]!;
    if (a.buffer === b.buffer && a.offset < b.offset + b.size && b.offset < a.offset + a.size)
      throw new Error("Mixed pressure fields overlap");
  }
  return result;
}

/** Every mixed pressure level is all-4h: stages compile the 4h tier alone. */
function assertAllCoarse(owner: UniformMixedOwnership): void {
  if (owner.layout.coarseTiles.length !== owner.layout.tiles.length) throw new Error("Mixed pressure levels must be all-4h");
}
/** Pipelines indexed by tier (ownership.dispatch): only the 4h tier exists.
 * The empty h and tier-1 (reserved) tiers never launch (dispatchTier returns on a zero count). */
function coarseTier(pipeline: GPUComputePipeline): GPUComputePipeline[] {
  const tiers: GPUComputePipeline[] = []; tiers[2] = pipeline; return tiers;
}
/** Both pressure levels are the same all-4h layout, so every owner (and
 * halo slot) has a same-width twin on the other level: transfers inject. */
const twinWGSL = /* wgsl */ `
fn umFineTwin(o:coarseUMOwner)->fineUMOwner{return fineumOwnerAt(vec3i(coarseumOrigin(o)));}
fn umCoarseTwin(o:fineUMOwner)->coarseUMOwner{return coarseumOwnerAt(vec3i(fineumOrigin(o)));}
`;
function validatePressureLevels(fine: UniformMixedOwnership, coarse: UniformMixedOwnership): void {
  if (fine.layout.lattice.dimensions.some((n, a) => n !== coarse.layout.lattice.dimensions[a])
    || fine.layout.lattice.cellSize_m.some((n, a) => n !== coarse.layout.lattice.cellSize_m[a])
    || (["x", "y", "z"] as const).some(a => fine.layout.lattice.origin_m[a] !== coarse.layout.lattice.origin_m[a]))
    throw new Error("Mixed pressure transfer lattices differ");
  assertAllCoarse(fine); assertAllCoarse(coarse);
}

/** GPU stage binding, not another pressure solver. The caller supplies shared
 * ownership and borrowed storage. The caller supplies
 * pressure bounds and ghost-fluid surface phi. Optional domain halo rows
 * retain separating walls; embedded-solid coefficients are not supported.
 */
export class UniformMixedPressureLevelStage {
  readonly allocatedBytes = 0;
  private readonly resources: GPUBindGroupLayout;
  private readonly pipelines = new Map<UniformMixedPressureEntry, GPUComputePipeline[]>();
  private readonly recordPipelines=new Map<RecordEntry,GPUComputePipeline>();
  private readonly regularPipelines=new Map<"residual"|"measure",GPUComputePipeline[]>();
  private recordGroup?:GPUBindGroup;
  /** Buffer topology rides in the phi binding from this f32 index: a separate
   * binding would make the record layout 11 compute storage buffers. */
  private topologyBase?:number;
  /** Ownership generation the records describe; phi/solids are refreshed by encodeRecords. */
  private recordsFor?:object;
  constructor(private readonly device: GPUDevice, readonly ownership: UniformMixedOwnership, private readonly constrained = false, private readonly surface = false, private readonly boundary?: {openTop:boolean},
    /** Embedded static solids: face V from a per-level topology record. */
    private readonly topology?: "buffer") {
    if(boundary&&!constrained)throw new Error("Separating walls require pressure minimum fields");
    if(topology&&(!surface||!boundary))throw new Error("Solid pressure topology requires surface rows and separating walls");
    assertAllCoarse(ownership);
    this.resources = device.createBindGroupLayout({ entries: [...[0, 1, 2, 3, 4, ...(constrained ? [5] : []), ...(surface ? [6] : [])].map(binding => ({ binding,
      visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } }))] });
  }
  async initialize(): Promise<void> {
    const owner = this.ownership, h = owner.layout.lattice.cellSize_m;
    const halo=(body:string)=>this.boundary?uniformMixedPressureBoundaryLoop(body):"";
    // planUniformMixedPressureMemory allocates a coarse level's topology directly after its phi.
    this.topologyBase=this.topology==="buffer"?Math.ceil(owner.layout.cellCount*4/256)*64:undefined;
    const module = await checkedModule(this.device, uniformMixedTopologyWGSL(owner.layout, 0) + /* wgsl */ `
const UM_H=vec3f(${h.map(n => n.toFixed(8)).join(",")});
@group(1) @binding(0) var<storage,read_write> pressures:array<f32>;
@group(1) @binding(1) var<storage,read_write> slopes:array<vec4f>;
@group(1) @binding(2) var<storage,read_write> rhs:array<f32>;
@group(1) @binding(3) var<storage,read_write> frozen:array<f32>;
@group(1) @binding(4) var<storage,read_write> result:array<f32>;
${this.constrained ? "@group(1) @binding(5) var<storage,read_write> minimum:array<f32>;" : ""}
fn umMinimum(o:UMOwner)->f32{return ${this.constrained ? "minimum[o.index]" : "-3.402823e38"};}
fn umPressure(o:UMOwner)->f32{return pressures[o.index];}
fn umPressureSlope(o:UMOwner)->vec3f{return slopes[o.index].xyz;}
${this.surface ? "@group(1) @binding(6) var<storage,read_write> phi:array<f32>;\nfn umPressurePhi(o:UMOwner)->f32{return phi[o.index];}" : ""}
${this.surface ? uniformMixedPressureSurfaceWGSL : ""}
${uniformMixedPressureReconstructionSource(this.surface)}
${this.topology ? uniformMixedPressureTopologyWGSL(this.topology,1,7,"",this.topologyBase) : ""}
${this.boundary ? uniformMixedPressureBoundaryWGSL(owner.layout,this.boundary.openTop,this.surface,!!this.topology) : ""}
${uniformMixedPressureOperatorSource(this.surface,!!this.boundary,!!this.topology)}
${uniformMixedPressureRecordsSource(this.surface,!!this.boundary,!!this.topology,this.constrained)}
@compute @workgroup_size(64) fn residual(@builtin(global_invocation_id) gid:vec3u){
 let o=umOwner(gid);if(o.width!=0u){result[o.index]=${this.surface ? "select(0.0,rhs[o.index]-umPressureApply(o),umPressureLiquid(o))" : "rhs[o.index]-umPressureApply(o)"};${halo("let coefficient=select(umBoundaryCoefficient(o,axis,sign),0.0,umBoundaryOpen(axis,sign));result[halo]=rhs[halo]-coefficient*(pressures[halo]-umPressure(o));")}}
}
@compute @workgroup_size(64) fn measure(@builtin(global_invocation_id) gid:vec3u){
 let o=umOwner(gid);if(o.width==0u){return;}
 ${halo(`let coefficient=select(umBoundaryCoefficient(o,axis,sign),0.0,umBoundaryOpen(axis,sign));
 let p=pressures[halo];let r=rhs[halo]-coefficient*(p-umPressure(o));let low=minimum[halo];let gap=max(0.0,p-low);
 let projected=max(select(abs(r),gap*coefficient,r<0.0&&-r>=gap*coefficient),max(0.0,low-p)*coefficient);
 let finite=(bitcast<u32>(p)&0x7f800000u)!=0x7f800000u&&(bitcast<u32>(r)&0x7f800000u)!=0x7f800000u&&(bitcast<u32>(projected)&0x7f800000u)!=0x7f800000u;
 result[halo]=select(3.402823e38,projected,finite&&projected>=0.0);`)}
 let p=umPressure(o);
${this.surface ? ` if(!umPressureLiquid(o)){
  result[o.index]=select(0.0,3.402823e38,(bitcast<u32>(p)&0x7f800000u)==0x7f800000u);return;
 }` : ""}
 let diagonal=umPressureCoreTerms(o).x;
 let r=rhs[o.index]-umPressureApply(o);let low=umMinimum(o);let gap=max(0.0,p-low);
 let boundActive=r<0.0&&-r>=gap*diagonal;
 let projected=max(select(abs(r),gap*diagonal,boundActive),max(0.0,low-p)*diagonal);
 let finite=(bitcast<u32>(p)&0x7f800000u)!=0x7f800000u
  &&(bitcast<u32>(r)&0x7f800000u)!=0x7f800000u
  &&(bitcast<u32>(projected)&0x7f800000u)!=0x7f800000u;
 // Float diagnostics use a maximal failure sentinel; WGSL rejects constant Inf.
 result[o.index]=select(3.402823e38,projected,finite&&projected>=0.0);
}
@compute @workgroup_size(64) fn saveBackup(@builtin(global_invocation_id) gid:vec3u){
 let o=umOwner(gid);if(o.width!=0u){result[o.index]=pressures[o.index];${halo("result[halo]=pressures[halo];")}}
}
@compute @workgroup_size(64) fn addBackup(@builtin(global_invocation_id) gid:vec3u){
 let o=umOwner(gid);if(o.width!=0u){pressures[o.index]+=rhs[o.index];${halo("pressures[halo]+=rhs[halo];")}}
}
` );
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [owner.bindLayout, this.resources] });
    const compile = async (entryPoint: string, regular = false) => coarseTier(await this.device.createComputePipelineAsync({ layout,
      compute: { module, entryPoint, constants: { umCellWidth: 4, umDispatchX: owner.dispatchX, umRegularTiles:+regular } } }));
    for (const entry of entries) this.pipelines.set(entry, await compile(entry));
    for(const entry of ["residual","measure"] as const)this.regularPipelines.set(entry,await compile(entry,true));
    const recordLayout=this.device.createPipelineLayout({bindGroupLayouts:[owner.bindLayout,this.resources,owner.hangingLayout]});
    for(const entryPoint of recordEntries)this.recordPipelines.set(entryPoint,await this.device.createComputePipelineAsync({layout:recordLayout,
      compute:{module,entryPoint,constants:{umDispatchX:owner.dispatchX}}}));
  }
  bind(fields: UniformMixedPressureFields): GPUBindGroup {
    const n = this.ownership.layout.cellCount, count=this.boundary?uniformMixedPressureStorage(this.ownership.layout).count:n;
    if (this.constrained !== !!fields.minimum) throw new Error("Mixed pressure minimum binding does not match stage mode");
    if (this.surface !== !!fields.phi) throw new Error("Mixed pressure phi binding does not match stage mode");
    const bindings = [fields.pressure, fields.slopes, fields.rhs, fields.frozen, fields.result], sizes = [4*count, 16*n, 4*count, 4*count, 4*count];
    if (fields.minimum) { bindings.push(fields.minimum); sizes.push(4*count); }
    if (fields.phi) {
      if (fields.topology) {
        const phi=fields.phi,topology=fields.topology.buffer,base=this.topologyBase!,records=16*uniformMixedPressureStorage(this.ownership.layout).count;
        if (topology.buffer!==phi.buffer || (topology.offset??0)!==(phi.offset??0)+4*base || (topology.size??records)<records)
          throw new Error("Mixed coarse pressure topology must directly follow its level's phi range");
        bindings.push({buffer:phi.buffer,offset:phi.offset??0,size:4*base+records}); sizes.push(4*base+records);
      } else { bindings.push(fields.phi); sizes.push(4*n); }
    }
    const resources = views(bindings, sizes);
    if (!!this.topology !== !!fields.topology)
      throw new Error("Mixed pressure topology binding does not match stage mode");
    const entries:GPUBindGroupEntry[]=resources.map((resource,index)=>({binding:index===resources.length-1&&this.surface?6:index,resource}));
    return this.device.createBindGroup({layout:this.resources,entries});
  }
  encode(target: UniformMixedPressureTarget, entry: UniformMixedPressureEntry, group: GPUBindGroup): void {
    const pipelines = this.pipelines.get(entry);
    if (!pipelines) throw new Error("Mixed pressure stage is not initialized");
    const { pass, end } = beginPass(target, `Uniform mixed pressure ${entry}`);
    pass.setBindGroup(0, this.ownership.bindGroup); pass.setBindGroup(1, group); this.ownership.dispatch(pass, pipelines); end();
  }
  /** Rebuild the seam records from this group's phi/solids and the current
   * ownership. Required whenever either changes; sweeps build lazily only
   * for a new ownership generation. */
  encodeRecords(target: UniformMixedPressureTarget, group: GPUBindGroup): void {
    const build=this.recordPipelines.get("buildRecords");if(!build)throw new Error("Mixed pressure stage is not initialized");
    this.recordGroup=this.ownership.recordGroup(this.ownership.fusedJobs(true)*UNIFORM_MIXED_PRESSURE_RECORD_CHUNK*4);
    const {pass,end}=beginPass(target,"Uniform mixed pressure records");
    pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);pass.setBindGroup(2,this.recordGroup!);
    this.ownership.dispatchFusedRows(pass,build);this.ownership.dispatchFusedRows(pass,this.recordPipelines.get("linkRecords")!);
    end();
    this.recordsFor=this.ownership.layout;
  }
  /** residual or measure: regular tiles through the regular operator, seam
   * rows and small regular tiers from their records at the current iterate. */
  encodeRecordEntry(target: UniformMixedPressureTarget, entry: "residual"|"measure", group: GPUBindGroup): void {
    const regular=this.regularPipelines.get(entry);if(!regular||this.recordPipelines.size!==recordEntries.length)throw new Error("Mixed pressure stage is not initialized");
    if(this.recordsFor!==this.ownership.layout)this.encodeRecords(target,group);
    const {pass,end}=beginPass(target,`Uniform mixed pressure ${entry}`);
    pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);pass.setBindGroup(2,this.recordGroup!);
    const fused=(name:RecordEntry)=>this.ownership.dispatchFused(pass,this.recordPipelines.get(name)!,true);
    fused("reconstructRecords");fused("freezeRecords");
    this.ownership.dispatchRegular(pass,regular,true);fused(`${entry}Records`);
    end();
  }
}

/** Level transfers between the two identical all-4h levels: every value,
 * phi, halo slot and (static solids) topology record injects from its twin. */
export class UniformMixedPressureTransferStage {
  readonly allocatedBytes = 0;
  private readonly resources: GPUBindGroupLayout;
  private readonly topologyResources?: GPUBindGroupLayout;
  private readonly pipelines = new Map<UniformMixedPressureTransferEntry, GPUComputePipeline[]>();
  constructor(private readonly device: GPUDevice, readonly fine: UniformMixedOwnership, readonly coarse: UniformMixedOwnership, private readonly boundary = false,
    private readonly topology?: "buffer") {
    validatePressureLevels(fine, coarse);
    if (topology && !boundary) throw new Error("Solid pressure topology requires separating walls");
    this.resources = device.createBindGroupLayout({ entries: [0, 1].map(binding => ({ binding,
      visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } })) });
    if (topology) this.topologyResources = device.createBindGroupLayout({ entries: [0, 1].map(binding => ({ binding,
      visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } })) });
  }
  private get solidEntries(): UniformMixedPressureTransferEntry[] { return this.topology ? ["restrictSurfacePhi"] : []; }
  async initialize(): Promise<void> {
    const fine = this.fine, coarse = this.coarse;
    const halo=(body:string,prefix:string)=>this.boundary?uniformMixedPressureBoundaryLoop(body,prefix):"";
    const boundarySource=this.boundary?uniformMixedPressureBoundaryIndexWGSL(fine.layout,"fine")+uniformMixedPressureBoundaryIndexWGSL(coarse.layout,"coarse"):"";
    const solid = this.topology ? /* wgsl */ `
${uniformMixedPressureTopologyWGSL(this.topology, 3, 0, "fine")}
@group(3) @binding(1) var<storage,read_write> topologyOut:array<vec4f>;
@compute @workgroup_size(64) fn restrictSurfacePhi(@builtin(global_invocation_id) gid:vec3u){
 let o=coarseumOwner(gid);if(o.width==0u){return;}
 let twin=umFineTwin(o);
 destination[o.index]=source[twin.index];topologyOut[o.index]=fineumTopo(twin);
 ${halo("topologyOut[halo]=vec4f(fineumPressureWallV(twin,axis,sign),0.0,0.0,0.0);","coarse")}
}` : /* wgsl */ `
@compute @workgroup_size(64) fn restrictSurfacePhi(@builtin(global_invocation_id) gid:vec3u){
 let o=coarseumOwner(gid);if(o.width!=0u){destination[o.index]=source[umFineTwin(o).index];}
}`;
    const module = await checkedModule(this.device, uniformMixedTopologyWGSL(fine.layout, 0, "fine") + uniformMixedTopologyWGSL(coarse.layout, 1, "coarse") + /* wgsl */ `
@group(2) @binding(0) var<storage,read_write> source:array<f32>;
@group(2) @binding(1) var<storage,read_write> destination:array<f32>;
${twinWGSL}
${boundarySource}
${solid}
@compute @workgroup_size(64) fn restrictValues(@builtin(global_invocation_id) gid:vec3u){
 let o=coarseumOwner(gid);if(o.width==0u){return;}let twin=umFineTwin(o);
 destination[o.index]=source[twin.index];${halo("destination[halo]=source[fineumBoundaryIndex(twin,axis,sign)];","coarse")}
}
@compute @workgroup_size(64) fn prolongAssign(@builtin(global_invocation_id) gid:vec3u){
 let o=fineumOwner(gid);if(o.width==0u){return;}let twin=umCoarseTwin(o);
 destination[o.index]=source[twin.index];${halo("destination[halo]=source[coarseumBoundaryIndex(twin,axis,sign)];","fine")}
}
@compute @workgroup_size(64) fn prolongAdd(@builtin(global_invocation_id) gid:vec3u){
 let o=fineumOwner(gid);if(o.width==0u){return;}let twin=umCoarseTwin(o);
 destination[o.index]+=source[twin.index];${halo("destination[halo]+=source[coarseumBoundaryIndex(twin,axis,sign)];","fine")}
}
` );
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [fine.bindLayout, coarse.bindLayout, this.resources] });
    const solidLayout = this.topologyResources ? this.device.createPipelineLayout({ bindGroupLayouts: [fine.bindLayout, coarse.bindLayout, this.resources, this.topologyResources] }) : layout;
    for (const entryPoint of ["restrictValues", "restrictSurfacePhi", "prolongAssign", "prolongAdd"] as const) this.pipelines.set(entryPoint,
      coarseTier(await this.device.createComputePipelineAsync({ layout: this.solidEntries.includes(entryPoint) ? solidLayout : layout, compute: { module, entryPoint,
        constants: { fineumCellWidth: 4, coarseumCellWidth: 4, fineumDispatchX: fine.dispatchX, coarseumDispatchX: coarse.dispatchX } } })));
  }
  bind(entry: UniformMixedPressureTransferEntry, source: GPUBufferBinding, destination: GPUBufferBinding): GPUBindGroup {
    const counts = [this.fine, this.coarse].map(o=>this.boundary&&entry!=="restrictSurfacePhi"?uniformMixedPressureStorage(o.layout).count:o.layout.cellCount);
    if (entry !== "restrictValues" && entry !== "restrictSurfacePhi") counts.reverse();
    const resources = views([source, destination], counts.map(n => n * 4));
    return this.device.createBindGroup({ layout: this.resources, entries: resources.map((resource, binding) => ({ binding, resource })) });
  }
  /** Both levels' arena topology records, sized for owners plus halo slots. */
  bindTopology(source: UniformMixedPressureTopology, destination: GPUBufferBinding): GPUBindGroup {
    if (!this.topologyResources) throw new Error("Mixed pressure transfer topology does not match stage mode");
    const [fine, coarse] = [this.fine, this.coarse].map(o => 16 * uniformMixedPressureStorage(o.layout).count);
    const input = views([source.buffer, destination], [fine!, coarse!]);
    return this.device.createBindGroup({ layout: this.topologyResources, entries: input.map((resource, binding) => ({ binding, resource })) });
  }
  encode(target: UniformMixedPressureTarget, entry: UniformMixedPressureTransferEntry, group: GPUBindGroup, topology?: GPUBindGroup): void {
    const pipelines = this.pipelines.get(entry);
    if (!pipelines) throw new Error("Mixed pressure transfers are not initialized");
    if (this.solidEntries.includes(entry) !== !!topology) throw new Error(`Mixed pressure ${entry} topology binding does not match stage mode`);
    const { pass, end } = beginPass(target, `Uniform mixed pressure ${entry}`);
    pass.setBindGroup(0, this.fine.bindGroup); pass.setBindGroup(1, this.coarse.bindGroup); pass.setBindGroup(2, group); if (topology) pass.setBindGroup(3, topology);
    ((entry === "restrictValues" || entry === "restrictSurfacePhi") ? this.coarse : this.fine).dispatch(pass, pipelines); end();
  }
}

export type UniformMixedPressureBoundsEntry = "shiftMinimum" | "downsampleMinimum" | "downsampleSubtract";
/** The CM11a correction-bound operations between the identical levels:
 * a twin's bound, optionally less its current pressure. */
export class UniformMixedPressureBoundsStage {
  readonly allocatedBytes = 0;
  private readonly resources: GPUBindGroupLayout;
  private readonly pipelines = new Map<UniformMixedPressureBoundsEntry, GPUComputePipeline[]>();
  constructor(private readonly device: GPUDevice, readonly fine: UniformMixedOwnership, readonly coarse: UniformMixedOwnership, private readonly boundary = false) {
    validatePressureLevels(fine, coarse);
    this.resources = device.createBindGroupLayout({ entries: [0, 1, 2].map(binding => ({ binding,
      visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } })) });
  }
  async initialize(): Promise<void> {
    const fine = this.fine, coarse = this.coarse;
    const halo=(body:string,prefix:string)=>this.boundary?uniformMixedPressureBoundaryLoop(body,prefix):"";
    const boundarySource=this.boundary?uniformMixedPressureBoundaryIndexWGSL(fine.layout,"fine")+uniformMixedPressureBoundaryIndexWGSL(coarse.layout,"coarse"):"";
    const module = await checkedModule(this.device, uniformMixedTopologyWGSL(fine.layout, 0, "fine") + uniformMixedTopologyWGSL(coarse.layout, 1, "coarse") + /* wgsl */ `
@group(2) @binding(0) var<storage,read_write> minimum:array<f32>;
@group(2) @binding(1) var<storage,read_write> pressure:array<f32>;
@group(2) @binding(2) var<storage,read_write> destination:array<f32>;
${twinWGSL}
${boundarySource}
fn umTwinBound(i:u32,subtract:bool)->f32{return minimum[i]-select(0.0,pressure[i],subtract);}
@compute @workgroup_size(64) fn shiftMinimum(@builtin(global_invocation_id) gid:vec3u){
 let o=fineumOwner(gid);if(o.width!=0u){destination[o.index]=umTwinBound(o.index,true);${halo("destination[halo]=umTwinBound(halo,true);","fine")}}
}
fn umDownsample(gid:vec3u,subtract:bool){
 let o=coarseumOwner(gid);if(o.width==0u){return;}let twin=umFineTwin(o);
 destination[o.index]=umTwinBound(twin.index,subtract);${halo("destination[halo]=umTwinBound(fineumBoundaryIndex(twin,axis,sign),subtract);","coarse")}
}
@compute @workgroup_size(64) fn downsampleMinimum(@builtin(global_invocation_id) gid:vec3u){umDownsample(gid,false);}
@compute @workgroup_size(64) fn downsampleSubtract(@builtin(global_invocation_id) gid:vec3u){umDownsample(gid,true);}
`);
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [fine.bindLayout, coarse.bindLayout, this.resources] });
    for (const entryPoint of ["shiftMinimum", "downsampleMinimum", "downsampleSubtract"] as const) this.pipelines.set(entryPoint,
      coarseTier(await this.device.createComputePipelineAsync({ layout, compute: { module, entryPoint,
        constants: { fineumCellWidth: 4, coarseumCellWidth: 4, fineumDispatchX: fine.dispatchX, coarseumDispatchX: coarse.dispatchX } } })));
  }
  bind(entry: UniformMixedPressureBoundsEntry, minimum: GPUBufferBinding, pressure: GPUBufferBinding, destination: GPUBufferBinding): GPUBindGroup {
    const count=(o:UniformMixedOwnership)=>this.boundary?uniformMixedPressureStorage(o.layout).count:o.layout.cellCount;
    const n = count(this.fine), m = entry === "shiftMinimum" ? n : count(this.coarse);
    const resources = views([minimum, pressure, destination], [4*n, 4*n, 4*m]);
    return this.device.createBindGroup({ layout: this.resources, entries: resources.map((resource, binding) => ({ binding, resource })) });
  }
  encode(target: UniformMixedPressureTarget, entry: UniformMixedPressureBoundsEntry, group: GPUBindGroup): void {
    const pipelines = this.pipelines.get(entry); if (!pipelines) throw new Error("Mixed pressure bounds are not initialized");
    const { pass, end } = beginPass(target, `Uniform mixed pressure ${entry}`);
    pass.setBindGroup(0, this.fine.bindGroup); pass.setBindGroup(1, this.coarse.bindGroup); pass.setBindGroup(2, group);
    (entry === "shiftMinimum" ? this.fine : this.coarse).dispatch(pass, pipelines); end();
  }
}
