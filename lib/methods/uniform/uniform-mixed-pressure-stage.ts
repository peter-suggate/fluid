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
import { uniformMixedPressureTopologyWGSL, type UniformMixedPressureTopology } from "./uniform-mixed-pressure-topology.wgsl";
import { UNIFORM_MIXED_PRESSURE_RECORD_CHUNK, uniformMixedPressureRecordsSource } from "./uniform-mixed-pressure-records.wgsl";

const recordEntries = ["buildRecords", "reconstructRecords", "freezeRecords", "smoothRecords", "residualRecords", "measureRecords"] as const;
type RecordEntry = typeof recordEntries[number];
const entries = ["reconstruct", "freezeRhs", "residual", "addBackup", "saveBackup", "measure"] as const;
export type UniformMixedPressureEntry = typeof entries[number];
export type UniformMixedPressureTransferEntry = "restrictValues" | "restrictSurfacePhi" | "prolongAssign" | "prolongAdd" | "extrapolateSurfacePhi";
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
  private regularSmoothPipelines:GPUComputePipeline[]=[];
  private readonly recordPipelines=new Map<RecordEntry,GPUComputePipeline>();
  private readonly regularPipelines=new Map<"residual"|"measure",GPUComputePipeline[]>();
  private readonly jacobiGroups = new WeakMap<GPUBindGroup,GPUBindGroup>();
  private recordGroup?:GPUBindGroup;
  /** Buffer topology rides in the phi binding from this f32 index: a separate
   * binding would make the record layout 11 compute storage buffers. */
  private topologyBase?:number;
  /** Ownership generation the records describe; phi/solids are refreshed by encodeRecords. */
  private recordsFor?:object;
  constructor(private readonly device: GPUDevice, readonly ownership: UniformMixedOwnership, private readonly constrained = false, private readonly surface = false, private readonly boundary?: {openTop:boolean},
    /** Embedded static solids: face V from a per-level topology record. */
    private readonly topology?: "texture"|"buffer") {
    if(boundary&&!constrained)throw new Error("Separating walls require pressure minimum fields");
    if(topology&&(!surface||!boundary))throw new Error("Solid pressure topology requires surface rows and separating walls");
    this.resources = device.createBindGroupLayout({ entries: [...[0, 1, 2, 3, 4, ...(constrained ? [5] : []), ...(surface ? [6] : [])].map(binding => ({ binding,
      visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } })),
      ...(topology==="texture"?[{binding:7,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}}]
        :[])] });
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
fn umReconstructOwner(o:UMOwner){if(o.width!=0u){slopes[o.index]=vec4f(umReconstructPressureSlope(o),0.0);}}
fn umFreezeOwner(o:UMOwner){if(o.width!=0u){frozen[o.index]=umPressureCorrectedRhs(o,rhs[o.index]);${halo("frozen[halo]=rhs[halo];")}}}
@compute @workgroup_size(64) fn reconstruct(@builtin(global_invocation_id) gid:vec3u){umReconstructOwner(umOwner(gid));}
@compute @workgroup_size(64) fn freezeRhs(@builtin(global_invocation_id) gid:vec3u){umFreezeOwner(umOwner(gid));}
@compute @workgroup_size(64) fn residual(@builtin(global_invocation_id) gid:vec3u){
 let o=umOwner(gid);if(o.width!=0u){result[o.index]=${this.surface ? "select(0.0,rhs[o.index]-umPressureApply(o),umPressureLiquid(o))" : "rhs[o.index]-umPressureApply(o)"};${halo("let coefficient=select(umBoundaryCoefficient(o,axis,sign),0.0,umBoundaryOpen(axis,sign));result[halo]=rhs[halo]-coefficient*(pressures[halo]-umPressure(o));")}}
}
@compute @workgroup_size(64) fn smoothJacobi(@builtin(global_invocation_id) gid:vec3u){umSmoothOwner(umOwner(gid));}
fn umSmoothOwner(o:UMOwner){
 if(o.width==0u){return;}
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
    this.regularSmoothPipelines=await Promise.all([1,2,4].map(width=>compile("smoothJacobi",width,false,true)));
    for(const entry of ["residual","measure"] as const)this.regularPipelines.set(entry,await Promise.all([1,2,4].map(width=>compile(entry,width,false,true))));
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
      if (fields.topology && "buffer" in fields.topology) {
        const phi=fields.phi,topology=fields.topology.buffer,base=this.topologyBase!,records=16*uniformMixedPressureStorage(this.ownership.layout).count;
        if (topology.buffer!==phi.buffer || (topology.offset??0)!==(phi.offset??0)+4*base || (topology.size??records)<records)
          throw new Error("Mixed coarse pressure topology must directly follow its level's phi range");
        bindings.push({buffer:phi.buffer,offset:phi.offset??0,size:4*base+records}); sizes.push(4*base+records);
      } else { bindings.push(fields.phi); sizes.push(4*n); }
    }
    const resources = views(bindings, sizes);
    if (!!this.topology !== !!fields.topology || (fields.topology && ("texture" in fields.topology) !== (this.topology === "texture")))
      throw new Error("Mixed pressure topology binding does not match stage mode");
    const entries:GPUBindGroupEntry[]=resources.map((resource,index)=>({binding:index===resources.length-1&&this.surface?6:index,resource}));
    if (fields.topology && "texture" in fields.topology) entries.push({binding:7,resource:fields.topology.texture.createView()});
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
  /** Rebuild the seam records from this group's phi/solids and the current
   * ownership. Required whenever either changes; sweeps build lazily only
   * for a new ownership generation. */
  encodeRecords(encoder: GPUCommandEncoder, group: GPUBindGroup): void {
    const build=this.recordPipelines.get("buildRecords");if(!build)throw new Error("Mixed pressure stage is not initialized");
    this.recordGroup=this.ownership.recordGroup(this.ownership.fusedJobs(true)*UNIFORM_MIXED_PRESSURE_RECORD_CHUNK*4);
    const pass=encoder.beginComputePass({label:"Uniform mixed pressure records"});
    pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);pass.setBindGroup(2,this.recordGroup!);
    this.ownership.dispatchFused(pass,build,true);pass.end();
    this.recordsFor=this.ownership.layout;
  }
  /** residual or measure: regular tiles through the regular operator, seam
   * rows and small regular tiers from their records at the current iterate. */
  encodeRecordEntry(encoder: GPUCommandEncoder, entry: "residual"|"measure", group: GPUBindGroup): void {
    const regular=this.regularPipelines.get(entry);if(!regular||this.recordPipelines.size!==recordEntries.length)throw new Error("Mixed pressure stage is not initialized");
    if(this.recordsFor!==this.ownership.layout)this.encodeRecords(encoder,group);
    const pass=encoder.beginComputePass({label:`Uniform mixed pressure ${entry}`});
    pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);pass.setBindGroup(2,this.recordGroup!);
    const fused=(name:RecordEntry)=>this.ownership.dispatchFused(pass,this.recordPipelines.get(name)!,true);
    fused("reconstructRecords");fused("freezeRecords");
    this.ownership.dispatchRegular(pass,regular,true);fused(`${entry}Records`);
    pass.end();
  }
  encodeSweep(encoder: GPUCommandEncoder, group: GPUBindGroup): void {
    if (this.recordPipelines.size!==recordEntries.length) throw new Error("Mixed pressure stage is not initialized");
    if(this.recordsFor!==this.ownership.layout)this.encodeRecords(encoder,group);
    // Native Uniform uses two simultaneous projected Jacobi updates per
    // sweep. The residual range is dead here and supplies the ping-pong half;
    // every tier reads the same old iterate, preserving reflection symmetry.
    const pass = encoder.beginComputePass({ label: "Uniform mixed pressure sweep" });
    pass.setBindGroup(0, this.ownership.bindGroup); pass.setBindGroup(1, group); pass.setBindGroup(2,this.recordGroup!);
    // Seam rows and small regular tiers gather their frozen linear records:
    // one launch per step instead of re-walking canonical patches each sweep.
    const fused=(entry:RecordEntry)=>this.ownership.dispatchFused(pass,this.recordPipelines.get(entry)!,true);
    fused("reconstructRecords");fused("freezeRecords");
    this.ownership.dispatchRegular(pass,this.regularSmoothPipelines,true);fused("smoothRecords");
    pass.setBindGroup(1,this.jacobiGroups.get(group)!);
    this.ownership.dispatchRegular(pass,this.regularSmoothPipelines,true);fused("smoothRecords");
    pass.end();
  }
}

/** With static solids (`topology` names the fine level's record kind),
 * restrictSurfacePhi is native mgDownsampleTopology on split owners: open
 * mean, positive-face V from the four children on the face plane (2/8 each),
 * the open-child phi vote, and halo V as the mean of the children's wall V.
 * Persistent owners inject. extrapolateSurfacePhi is mgExtrapolatePhiOneCell
 * on the coarse level, in place: closed owners read only open neighbours. */
export class UniformMixedPressureTransferStage {
  readonly allocatedBytes = 0;
  private readonly resources: GPUBindGroupLayout;
  private readonly topologyResources?: GPUBindGroupLayout;
  private readonly pipelines = new Map<UniformMixedPressureTransferEntry, GPUComputePipeline[]>();
  constructor(private readonly device: GPUDevice, readonly fine: UniformMixedOwnership, readonly coarse: UniformMixedOwnership, private readonly preferPositivePhi = false, private readonly boundary = false,
    private readonly topology?: "texture"|"buffer") {
    validatePressureLevels(fine, coarse);
    if (topology && !boundary) throw new Error("Solid pressure topology requires separating walls");
    this.resources = device.createBindGroupLayout({ entries: [0, 1].map(binding => ({ binding,
      visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } })) });
    if (topology) this.topologyResources = device.createBindGroupLayout({ entries: [
      topology === "texture" ? { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" as const, viewDimension: "3d" as const } }
        : { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } }] });
  }
  private get solidEntries(): UniformMixedPressureTransferEntry[] { return this.topology ? ["restrictSurfacePhi", "extrapolateSurfacePhi"] : []; }
  async initialize(): Promise<void> {
    const fine = this.fine, coarse = this.coarse;
    const halo=(body:string,prefix:string)=>this.boundary?uniformMixedPressureBoundaryLoop(body,prefix):"";
    const boundarySource=this.boundary?uniformMixedPressureBoundaryIndexWGSL(fine.layout,"fine")+uniformMixedPressureBoundaryIndexWGSL(coarse.layout,"coarse")+uniformMixedPressureBoundaryChildrenWGSL:"";
    const solid = this.topology ? /* wgsl */ `
${uniformMixedPressureTopologyWGSL(this.topology, 3, 0, "fine")}
@group(3) @binding(1) var<storage,read_write> topologyOut:array<vec4f>;
fn umSum8Vec4(v:array<vec4f,8>)->vec4f{return ((v[0]+v[5])+(v[1]+v[4]))+((v[2]+v[7])+(v[3]+v[6]));}
@compute @workgroup_size(64) fn restrictSurfacePhi(@builtin(global_invocation_id) gid:vec3u){
 let o=coarseumOwner(gid);if(o.width==0u){return;}
 let origin=coarseumOrigin(o);let first=fineumOwnerAt(vec3i(origin));
 var topology=fineumTopo(first);var phi=source[first.index];
 if(first.width!=o.width){
  var terms:array<vec4f,8>;var faces:array<vec4f,8>;var values:array<f32,8>;var open:array<f32,8>;var openFlags:array<f32,8>;
  var positive:array<f32,8>;var positiveFlags:array<f32,8>;var negativeFlags:array<f32,8>;
  for(var k=0u;k<8u;k++){
   let bit=vec3u(k&1u,(k>>1u)&1u,k>>2u);let child=fineumOwnerAt(vec3i(origin+bit*(o.width/2u)));
   let t=fineumTopo(child);let value=source[child.index];let isOpen=t.x>1e-5;
   terms[k]=t;faces[k]=vec4f(0.0,select(0.0,2.0*t.y,bit.x==1u),select(0.0,2.0*t.z,bit.y==1u),select(0.0,2.0*t.w,bit.z==1u));
   values[k]=value;open[k]=select(0.0,value,isOpen);openFlags[k]=select(0.0,1.0,isOpen);
   positive[k]=select(0.0,value,value>=0.0&&isOpen);positiveFlags[k]=select(0.0,1.0,value>=0.0&&isOpen);negativeFlags[k]=select(0.0,1.0,value<0.0&&isOpen);
  }
  let v=umSum8Vec4(terms);let fv=umSum8Vec4(faces);topology=vec4f(v.x,fv.y,fv.z,fv.w)/8.0;
  let openCount=umPhiSum8(openFlags);let positiveCount=umPhiSum8(positiveFlags);
  let sum=select(umPhiSum8(values),umPhiSum8(open)*8.0/max(openCount,1.0),openCount>0.0);
  phi=select(sum/8.0,umPhiSum8(positive)/max(positiveCount,1.0),umPreferPositivePhi&&positiveCount>0.0&&umPhiSum8(negativeFlags)>0.0);
 }
 destination[o.index]=phi;topologyOut[o.index]=topology;
 ${halo(`var walls:array<f32,8>;for(var k=0u;k<8u;k++){walls[k]=fineumPressureWallV(umBoundaryChild(o,axis,sign,k),axis,sign);}
 topologyOut[halo]=vec4f(umPhiSum8(walls)/8.0,0.0,0.0,0.0);`,"coarse")}
}
@compute @workgroup_size(64) fn extrapolateSurfacePhi(@builtin(global_invocation_id) gid:vec3u){
 let o=coarseumOwner(gid);if(o.width==0u||topologyOut[o.index].x>1e-5){return;}
 let origin=vec3i(coarseumOrigin(o));var terms:array<f32,6>;var weights:array<f32,6>;
 for(var n=0u;n<6u;n++){
  var q=origin;q[n/2u]+=select(-i32(o.width),i32(o.width),(n&1u)==1u);
  let other=coarseumOwnerAt(q);if(other.width==0u){continue;}
  let v=topologyOut[other.index].x;let value=destination[other.index];
  if(v>1e-5&&value<0.0){terms[n]=v*value;weights[n]=v;}
 }
 let sum=((terms[0]+terms[1])+(terms[4]+terms[5]))+(terms[2]+terms[3]);let weight=((weights[0]+weights[1])+(weights[4]+weights[5]))+(weights[2]+weights[3]);
 if(weight>0.0){destination[o.index]=sum/max(weight,1e-9);}
}` : /* wgsl */ `
@compute @workgroup_size(64) fn restrictSurfacePhi(@builtin(global_invocation_id) gid:vec3u){
 let o=coarseumOwner(gid);if(o.width!=0u){destination[o.index]=umRestrictSurfacePhi(o);}
}`;
    const module = await checkedModule(this.device, uniformMixedTopologyWGSL(fine.layout, 0, "fine") + uniformMixedTopologyWGSL(coarse.layout, 1, "coarse") + /* wgsl */ `
@group(2) @binding(0) var<storage,read_write> source:array<f32>;
@group(2) @binding(1) var<storage,read_write> destination:array<f32>;
fn umFineResidual(o:fineUMOwner)->f32{return source[o.index];}
fn umCoarsePressure(o:coarseUMOwner)->f32{return source[o.index];}
${boundarySource}
${this.boundary?uniformMixedPressureBoundaryTransferWGSL:""}
${uniformMixedPressureTransferWGSL}
${uniformMixedPressureSurfaceTransferWGSL}
${solid}
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
    const solidLayout = this.topologyResources ? this.device.createPipelineLayout({ bindGroupLayouts: [fine.bindLayout, coarse.bindLayout, this.resources, this.topologyResources] }) : layout;
    for (const entryPoint of ["restrictValues", "restrictSurfacePhi", "prolongAssign", "prolongAdd", ...(this.topology ? ["extrapolateSurfacePhi"] as const : [])] as const) this.pipelines.set(entryPoint,
      await Promise.all([1, 2, 4].map(width => this.device.createComputePipelineAsync({ layout: this.solidEntries.includes(entryPoint) ? solidLayout : layout, compute: { module, entryPoint,
        constants: { umPreferPositivePhi: Number(this.preferPositivePhi), fineumCellWidth: width, coarseumCellWidth: width, fineumDispatchX: fine.dispatchX, coarseumDispatchX: coarse.dispatchX } } }))));
  }
  bind(entry: UniformMixedPressureTransferEntry, source: GPUBufferBinding, destination: GPUBufferBinding): GPUBindGroup {
    const counts = [this.fine, this.coarse].map(o=>this.boundary&&entry!=="restrictSurfacePhi"&&entry!=="extrapolateSurfacePhi"?uniformMixedPressureStorage(o.layout).count:o.layout.cellCount);
    if (entry !== "restrictValues" && entry !== "restrictSurfacePhi" && entry !== "extrapolateSurfacePhi") counts.reverse();
    const resources = views([source, destination], counts.map(n => n * 4));
    return this.device.createBindGroup({ layout: this.resources, entries: resources.map((resource, binding) => ({ binding, resource })) });
  }
  /** Fine-level record (texture at L0, arena vec4 buffer above) and the
   * coarse level's arena record, both sized for owners plus halo slots. */
  bindTopology(source: UniformMixedPressureTopology, destination: GPUBufferBinding): GPUBindGroup {
    if (!this.topologyResources || ("texture" in source) !== (this.topology === "texture")) throw new Error("Mixed pressure transfer topology does not match stage mode");
    const [fine, coarse] = [this.fine, this.coarse].map(o => 16 * uniformMixedPressureStorage(o.layout).count);
    const out = views([destination], [coarse!])[0]!;
    if ("buffer" in source) { const input = views([source.buffer, destination], [fine!, coarse!]);
      return this.device.createBindGroup({ layout: this.topologyResources, entries: input.map((resource, binding) => ({ binding, resource })) }); }
    return this.device.createBindGroup({ layout: this.topologyResources, entries: [{ binding: 0, resource: source.texture.createView() }, { binding: 1, resource: out }] });
  }
  encode(encoder: GPUCommandEncoder, entry: UniformMixedPressureTransferEntry, group: GPUBindGroup, topology?: GPUBindGroup): void {
    const pipelines = this.pipelines.get(entry);
    if (!pipelines) throw new Error("Mixed pressure transfers are not initialized");
    if (this.solidEntries.includes(entry) !== !!topology) throw new Error(`Mixed pressure ${entry} topology binding does not match stage mode`);
    const pass = encoder.beginComputePass({ label: `Uniform mixed pressure ${entry}` });
    pass.setBindGroup(0, this.fine.bindGroup); pass.setBindGroup(1, this.coarse.bindGroup); pass.setBindGroup(2, group); if (topology) pass.setBindGroup(3, topology);
    ((entry === "restrictValues" || entry === "restrictSurfacePhi" || entry === "extrapolateSurfacePhi") ? this.coarse : this.fine).dispatch(pass, pipelines); pass.end();
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
