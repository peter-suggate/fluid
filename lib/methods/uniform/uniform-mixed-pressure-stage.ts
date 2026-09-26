import { uniformMixedPressureBoundaryChildrenWGSL, uniformMixedPressureBoundaryTransferWGSL, uniformMixedPressureBoundaryBoundsWGSL } from "./uniform-mixed-pressure-boundary-transfer.wgsl";
import { uniformMixedPressureBoundaryIndexWGSL, uniformMixedPressureBoundaryWGSL, uniformMixedPressureBoundaryLoop, uniformMixedPressureStorage } from "./uniform-mixed-pressure-boundary.wgsl";
import { uniformMixedPressureSurfaceWGSL } from "./uniform-mixed-pressure-surface.wgsl";
import { uniformMixedPressureBoundsWGSL } from "./uniform-mixed-pressure-bounds.wgsl";
import { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { mixedCellWidth } from "./uniform-mixed-layout";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedPressureReconstructionSource } from "./uniform-mixed-pressure-reconstruction.wgsl";
import { uniformMixedPressureOperatorSource } from "./uniform-mixed-pressure-operator.wgsl";
import { uniformMixedPressureTransferWGSL, uniformMixedPressureSurfaceTransferWGSL } from "./uniform-mixed-pressure-transfer.wgsl";

const entries = ["reconstruct", "freezeRhs", "residual", "addBackup", "saveBackup", "measure"] as const;
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

function validatePressureLevels(fine: UniformMixedOwnership, coarse: UniformMixedOwnership): void {
  if (fine.layout.lattice.dimensions.some((n, a) => n !== coarse.layout.lattice.dimensions[a])
    || fine.layout.lattice.cellSize_m.some((n, a) => n !== coarse.layout.lattice.cellSize_m[a])
    || (["x", "y", "z"] as const).some(a => fine.layout.lattice.origin_m[a] !== coarse.layout.lattice.origin_m[a]))
    throw new Error("Mixed pressure transfer lattices differ");
  if (fine.layout.tiles.some((word, i) => {
    const f = mixedCellWidth(word), c = mixedCellWidth(coarse.layout.tiles[i]!);
    return c !== f && c !== 2 * f;
  })) throw new Error("Mixed pressure transfers require adjacent nested levels");
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
  private smoothPipelines: GPUComputePipeline[] = [];
  private regularSmoothPipelines:GPUComputePipeline[]=[];
  private readonly seamPipelines=new Map<"reconstruct"|"freezeRhs",GPUComputePipeline[]>();
  private readonly jacobiGroups = new WeakMap<GPUBindGroup,GPUBindGroup>();
  constructor(private readonly device: GPUDevice, readonly ownership: UniformMixedOwnership, private readonly constrained = false, private readonly surface = false, private readonly boundary?: {openTop:boolean}) {
    if(boundary&&!constrained)throw new Error("Separating walls require pressure minimum fields");
    this.resources = device.createBindGroupLayout({ entries: [0, 1, 2, 3, 4, ...(constrained ? [5] : []), ...(surface ? [6] : [])].map(binding => ({ binding,
      visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } })) });
  }
  async initialize(): Promise<void> {
    const owner = this.ownership, h = owner.layout.lattice.cellSize_m;
    const halo=(body:string)=>this.boundary?uniformMixedPressureBoundaryLoop(body):"";
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
${this.boundary ? uniformMixedPressureBoundaryWGSL(owner.layout,this.boundary.openTop,this.surface) : ""}
${uniformMixedPressureOperatorSource(this.surface,!!this.boundary)}
@compute @workgroup_size(64) fn reconstruct(@builtin(global_invocation_id) gid:vec3u){
 let o=umOwner(gid);if(o.width!=0u){slopes[o.index]=vec4f(umReconstructPressureSlope(o),0.0);}
}
@compute @workgroup_size(64) fn freezeRhs(@builtin(global_invocation_id) gid:vec3u){
 let o=umOwner(gid);if(o.width!=0u){frozen[o.index]=umPressureCorrectedRhs(o,rhs[o.index]);${halo("frozen[halo]=rhs[halo];")}}
}
@compute @workgroup_size(64) fn residual(@builtin(global_invocation_id) gid:vec3u){
 let o=umOwner(gid);if(o.width!=0u){result[o.index]=${this.surface ? "select(0.0,rhs[o.index]-umPressureApply(o),umPressureLiquid(o))" : "rhs[o.index]-umPressureApply(o)"};${halo("let coefficient=select(umBoundaryCoefficient(o,axis,sign),0.0,umBoundaryOpen(axis,sign));result[halo]=rhs[halo]-coefficient*(pressures[halo]-umPressure(o));")}}
}
@compute @workgroup_size(64) fn smoothJacobi(@builtin(global_invocation_id) gid:vec3u){
 let o=umOwner(gid);if(o.width==0u){return;}
 let old=umPressure(o);let core=umPressureCoreTerms(o);
 var next=old;
 if(core.x>0.0){let b=select(frozen[o.index],rhs[o.index],umPressureRegular(o));next=mix(old,(b+core.y)/core.x,0.6666667); }
 result[o.index]=${this.constrained ? "max(next,umMinimum(o))" : "next"};
 ${halo(`let coefficient=select(umBoundaryCoefficient(o,axis,sign),0.0,umBoundaryOpen(axis,sign));
 let next=select(pressures[halo],mix(pressures[halo],umPressure(o)+rhs[halo]/coefficient,0.6666667),coefficient>0.0);
 result[halo]=max(next,minimum[halo]);`)}
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
    const compile = (entryPoint: string, width: number, seams = false, regular = false) => this.device.createComputePipelineAsync({ layout,
      compute: { module, entryPoint, constants: { umCellWidth: width, umDispatchX: owner.dispatchX, umInterfaceTiles:+seams,umRegularTiles:+regular } } });
    for (const entry of entries) this.pipelines.set(entry, await Promise.all([1, 2, 4].map(w => compile(entry, w))));
    this.smoothPipelines = await Promise.all([1,2,4].map(width => compile("smoothJacobi",width,true)));
    this.regularSmoothPipelines=await Promise.all([1,2,4].map(width=>compile("smoothJacobi",width,false,true)));
    for(const entry of ["reconstruct","freezeRhs"] as const)this.seamPipelines.set(entry,await Promise.all([1,2,4].map(width=>compile(entry,width,true))));
  }
  bind(fields: UniformMixedPressureFields): GPUBindGroup {
    const n = this.ownership.layout.cellCount, count=this.boundary?uniformMixedPressureStorage(this.ownership.layout).count:n;
    if (this.constrained !== !!fields.minimum) throw new Error("Mixed pressure minimum binding does not match stage mode");
    if (this.surface !== !!fields.phi) throw new Error("Mixed pressure phi binding does not match stage mode");
    const bindings = [fields.pressure, fields.slopes, fields.rhs, fields.frozen, fields.result], sizes = [4*count, 16*n, 4*count, 4*count, 4*count];
    if (fields.minimum) { bindings.push(fields.minimum); sizes.push(4*count); }
    if (fields.phi) { bindings.push(fields.phi); sizes.push(4*n); }
    const resources = views(bindings, sizes);
    const entries=resources.map((resource,index)=>({binding:index===resources.length-1&&this.surface?6:index,resource}));
    const group=this.device.createBindGroup({layout:this.resources,entries});
    const swapped=entries.map(entry=>({...entry,resource:entry.binding===0?resources[4]!:entry.binding===4?resources[0]!:entry.resource}));
    this.jacobiGroups.set(group,this.device.createBindGroup({layout:this.resources,entries:swapped}));
    return group;
  }
  encode(encoder: GPUCommandEncoder, entry: UniformMixedPressureEntry, group: GPUBindGroup): void {
    const pipelines = this.pipelines.get(entry);
    if (!pipelines) throw new Error("Mixed pressure stage is not initialized");
    const pass = encoder.beginComputePass({ label: `Uniform mixed pressure ${entry}` });
    pass.setBindGroup(0, this.ownership.bindGroup); pass.setBindGroup(1, group); this.ownership.dispatch(pass, pipelines); pass.end();
  }
  encodeSweep(encoder: GPUCommandEncoder, group: GPUBindGroup): void {
    if (this.smoothPipelines.length !== 3) throw new Error("Mixed pressure stage is not initialized");
    // Native Uniform uses two simultaneous projected Jacobi updates per
    // sweep. The residual range is dead here and supplies the ping-pong half;
    // every tier reads the same old iterate, preserving reflection symmetry.
    const pass = encoder.beginComputePass({ label: "Uniform mixed pressure sweep" });
    pass.setBindGroup(0, this.ownership.bindGroup); pass.setBindGroup(1, group);
    this.ownership.dispatchSeams(pass,this.seamPipelines.get("reconstruct")!);
    this.ownership.dispatchSeams(pass,this.seamPipelines.get("freezeRhs")!);
    this.ownership.dispatchRegular(pass,this.regularSmoothPipelines);
    this.ownership.dispatchSeams(pass,this.smoothPipelines);
    pass.setBindGroup(1,this.jacobiGroups.get(group)!);
    this.ownership.dispatchRegular(pass,this.regularSmoothPipelines);
    this.ownership.dispatchSeams(pass,this.smoothPipelines);
    pass.end();
  }
}

export class UniformMixedPressureTransferStage {
  readonly allocatedBytes = 0;
  private readonly resources: GPUBindGroupLayout;
  private readonly pipelines = new Map<UniformMixedPressureTransferEntry, GPUComputePipeline[]>();
  constructor(private readonly device: GPUDevice, readonly fine: UniformMixedOwnership, readonly coarse: UniformMixedOwnership, private readonly preferPositivePhi = false, private readonly boundary = false) {
    validatePressureLevels(fine, coarse);
    this.resources = device.createBindGroupLayout({ entries: [0, 1].map(binding => ({ binding,
      visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } })) });
  }
  async initialize(): Promise<void> {
    const fine = this.fine, coarse = this.coarse;
    const halo=(body:string,prefix:string)=>this.boundary?uniformMixedPressureBoundaryLoop(body,prefix):"";
    const boundarySource=this.boundary?uniformMixedPressureBoundaryIndexWGSL(fine.layout,"fine")+uniformMixedPressureBoundaryIndexWGSL(coarse.layout,"coarse")+uniformMixedPressureBoundaryChildrenWGSL:"";
    const module = await checkedModule(this.device, uniformMixedTopologyWGSL(fine.layout, 0, "fine") + uniformMixedTopologyWGSL(coarse.layout, 1, "coarse") + /* wgsl */ `
@group(2) @binding(0) var<storage,read_write> source:array<f32>;
@group(2) @binding(1) var<storage,read_write> destination:array<f32>;
fn umFineResidual(o:fineUMOwner)->f32{return source[o.index];}
fn umCoarsePressure(o:coarseUMOwner)->f32{return source[o.index];}
${boundarySource}
${this.boundary?uniformMixedPressureBoundaryTransferWGSL:""}
${uniformMixedPressureTransferWGSL}
${uniformMixedPressureSurfaceTransferWGSL}
@compute @workgroup_size(64) fn restrictSurfacePhi(@builtin(global_invocation_id) gid:vec3u){
 let o=coarseumOwner(gid);if(o.width!=0u){destination[o.index]=umRestrictSurfacePhi(o);}
}
@compute @workgroup_size(64) fn restrictValues(@builtin(global_invocation_id) gid:vec3u){
 let o=coarseumOwner(gid);if(o.width!=0u){destination[o.index]=umRestrictPressureResidual(o);${halo("destination[halo]=umRestrictBoundary(o,axis,sign);","coarse")}}
}
@compute @workgroup_size(64) fn prolongAssign(@builtin(global_invocation_id) gid:vec3u){
 let o=fineumOwner(gid);if(o.width!=0u){destination[o.index]=umProlongPressureCorrection(o);${halo("destination[halo]=umProlongBoundary(o,axis,sign);","fine")}}
}
@compute @workgroup_size(64) fn prolongAdd(@builtin(global_invocation_id) gid:vec3u){
 let o=fineumOwner(gid);if(o.width!=0u){destination[o.index]+=umProlongPressureCorrection(o);${halo("destination[halo]+=umProlongBoundary(o,axis,sign);","fine")}}
}
` );
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [fine.bindLayout, coarse.bindLayout, this.resources] });
    for (const entryPoint of ["restrictValues", "restrictSurfacePhi", "prolongAssign", "prolongAdd"] as const) this.pipelines.set(entryPoint,
      await Promise.all([1, 2, 4].map(width => this.device.createComputePipelineAsync({ layout, compute: { module, entryPoint,
        constants: { umPreferPositivePhi: Number(this.preferPositivePhi), fineumCellWidth: width, coarseumCellWidth: width, fineumDispatchX: fine.dispatchX, coarseumDispatchX: coarse.dispatchX } } }))));
  }
  bind(entry: UniformMixedPressureTransferEntry, source: GPUBufferBinding, destination: GPUBufferBinding): GPUBindGroup {
    const counts = [this.fine, this.coarse].map(o=>this.boundary&&entry!=="restrictSurfacePhi"?uniformMixedPressureStorage(o.layout).count:o.layout.cellCount);
    if (entry !== "restrictValues" && entry !== "restrictSurfacePhi") counts.reverse();
    const resources = views([source, destination], counts.map(n => n * 4));
    return this.device.createBindGroup({ layout: this.resources, entries: resources.map((resource, binding) => ({ binding, resource })) });
  }
  encode(encoder: GPUCommandEncoder, entry: UniformMixedPressureTransferEntry, group: GPUBindGroup): void {
    const pipelines = this.pipelines.get(entry);
    if (!pipelines) throw new Error("Mixed pressure transfers are not initialized");
    const pass = encoder.beginComputePass({ label: `Uniform mixed pressure ${entry}` });
    pass.setBindGroup(0, this.fine.bindGroup); pass.setBindGroup(1, this.coarse.bindGroup); pass.setBindGroup(2, group);
    ((entry === "restrictValues" || entry === "restrictSurfacePhi") ? this.coarse : this.fine).dispatch(pass, pipelines); pass.end();
  }
}

export type UniformMixedPressureBoundsEntry = "shiftMinimum" | "downsampleMinimum" | "downsampleSubtract";
/** The existing CM11a correction-bound operations, using borrowed fields. */
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
    const boundarySource=this.boundary?uniformMixedPressureBoundaryIndexWGSL(fine.layout,"fine")+uniformMixedPressureBoundaryIndexWGSL(coarse.layout,"coarse")+uniformMixedPressureBoundaryChildrenWGSL:"";
    const module = await checkedModule(this.device, uniformMixedTopologyWGSL(fine.layout, 0, "fine") + uniformMixedTopologyWGSL(coarse.layout, 1, "coarse") + /* wgsl */ `
@group(2) @binding(0) var<storage,read_write> minimum:array<f32>;
@group(2) @binding(1) var<storage,read_write> pressure:array<f32>;
@group(2) @binding(2) var<storage,read_write> destination:array<f32>;
fn umFineMinimum(o:fineUMOwner)->f32{return minimum[o.index];}
fn umFinePressure(o:fineUMOwner)->f32{return pressure[o.index];}
${boundarySource}
${this.boundary?uniformMixedPressureBoundaryBoundsWGSL:""}
${uniformMixedPressureBoundsWGSL}
@compute @workgroup_size(64) fn shiftMinimum(@builtin(global_invocation_id) gid:vec3u){
 let o=fineumOwner(gid);if(o.width!=0u){destination[o.index]=umFineMinimum(o)-umFinePressure(o);${halo("destination[halo]=minimum[halo]-pressure[halo];","fine")}}
}
@compute @workgroup_size(64) fn downsampleMinimum(@builtin(global_invocation_id) gid:vec3u){
 let o=coarseumOwner(gid);if(o.width!=0u){destination[o.index]=umRestrictPressureMinimum(o,false);${halo("destination[halo]=umRestrictBoundaryMinimum(o,axis,sign,false);","coarse")}}
}
@compute @workgroup_size(64) fn downsampleSubtract(@builtin(global_invocation_id) gid:vec3u){
 let o=coarseumOwner(gid);if(o.width!=0u){destination[o.index]=umRestrictPressureMinimum(o,true);${halo("destination[halo]=umRestrictBoundaryMinimum(o,axis,sign,true);","coarse")}}
}
`);
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [fine.bindLayout, coarse.bindLayout, this.resources] });
    for (const entryPoint of ["shiftMinimum", "downsampleMinimum", "downsampleSubtract"] as const) this.pipelines.set(entryPoint,
      await Promise.all([1, 2, 4].map(width => this.device.createComputePipelineAsync({ layout, compute: { module, entryPoint,
        constants: { fineumCellWidth: width, coarseumCellWidth: width, fineumDispatchX: fine.dispatchX, coarseumDispatchX: coarse.dispatchX } } }))));
  }
  bind(entry: UniformMixedPressureBoundsEntry, minimum: GPUBufferBinding, pressure: GPUBufferBinding, destination: GPUBufferBinding): GPUBindGroup {
    const count=(o:UniformMixedOwnership)=>this.boundary?uniformMixedPressureStorage(o.layout).count:o.layout.cellCount;
    const n = count(this.fine), m = entry === "shiftMinimum" ? n : count(this.coarse);
    const resources = views([minimum, pressure, destination], [4*n, 4*n, 4*m]);
    return this.device.createBindGroup({ layout: this.resources, entries: resources.map((resource, binding) => ({ binding, resource })) });
  }
  encode(encoder: GPUCommandEncoder, entry: UniformMixedPressureBoundsEntry, group: GPUBindGroup): void {
    const pipelines = this.pipelines.get(entry); if (!pipelines) throw new Error("Mixed pressure bounds are not initialized");
    const pass = encoder.beginComputePass({ label: `Uniform mixed pressure ${entry}` });
    pass.setBindGroup(0, this.fine.bindGroup); pass.setBindGroup(1, this.coarse.bindGroup); pass.setBindGroup(2, group);
    (entry === "shiftMinimum" ? this.fine : this.coarse).dispatch(pass, pipelines); pass.end();
  }
}
