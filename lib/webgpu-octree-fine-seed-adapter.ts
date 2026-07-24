/**
 * GPU adapter from compact pressure/topology rows to factor-m fine-level-set
 * seed candidates. It aliases the existing compact topology and adaptive face
 * arena and never allocates from the finest bounding-box volume.
 */

import { PassBroker } from "./webgpu-pass-broker";
import {
  OCTREE_POWER_VELOCITY_VALID,
  type OctreePowerVelocitySource,
} from "./webgpu-octree-power-velocity";
import {
  OCTREE_FINE_SEED_CANDIDATE_BYTES,
  OCTREE_FINE_SEED_INVALID,
  OCTREE_FINE_SEED_LEAF_RECORD_BYTES,
  OCTREE_FINE_SEED_STATE,
  type OctreeFineSeedCandidateSource,
  type OctreeFineSeedLeafResources,
} from "./octree-fine-seed-leaves";
import { OCTREE_COARSE_PHI_FLAG } from "./octree-coarse-levelset";

/**
 * Candidate publication control:
 * [count, dispatch xyz, topology generation, published, error, capacity,
 * delta-authorized].
 * `published` deliberately does not depend on count, so a completed empty
 * liquid generation is distinct from an unpublished/failed empty producer.
 */
export const OCTREE_FINE_SEED_ADAPTER_CONTROL_BYTES = 36;
export const OCTREE_FINE_SEED_ADAPTER_PUBLICATION = Object.freeze({
  count: 0, dispatch: 1, generation: 4, published: 5, error: 6, capacity: 7, delta: 8,
} as const);
export const OCTREE_FINE_SEED_ADAPTER_PARAMETER_BYTES = 64;

export interface OctreeFineSeedAdapterPlan {
  readonly rowCapacity: number;
  readonly leafBytes: number;
  readonly candidateBytes: number;
  readonly allocatedBytes: number;
}

export function planOctreeFineSeedAdapter(rowCapacity: number): OctreeFineSeedAdapterPlan {
  if (!Number.isSafeInteger(rowCapacity) || rowCapacity < 1) {
    throw new RangeError("Octree fine-seed adapter row capacity must be positive");
  }
  const leafBytes = rowCapacity * OCTREE_FINE_SEED_LEAF_RECORD_BYTES;
  const candidateBytes = rowCapacity * OCTREE_FINE_SEED_CANDIDATE_BYTES;
  return {
    rowCapacity,
    leafBytes,
    candidateBytes,
    allocatedBytes: leafBytes + candidateBytes
      + leafBytes + 128
      + OCTREE_FINE_SEED_ADAPTER_CONTROL_BYTES + OCTREE_FINE_SEED_ADAPTER_PARAMETER_BYTES,
  };
}

export interface OctreeFineSeedAdapterTopology {
  /** Existing 48-byte pressure LeafHeader rows. */
  readonly leafHeaders: GPUBuffer;
  /** Compact pressure control; word 0 is the live LeafHeader row count. */
  readonly rowCount: GPUBuffer;
  /** Full compact control arena including its trailing overflow words. */
  readonly publicationControl?: GPUBuffer;
  /** Persistent frontier; word 3 is the completed topology generation. */
  readonly frontier: GPUBuffer;
  readonly dimensions: readonly [number, number, number];
  readonly cellSize: readonly [number, number, number];
  /** Exact topology mask-XOR publication. Delta fine-seed maintenance is
   * authorized only when `topologyReuseWord` is one and the leaf's tile bears
   * the current frontier generation stamp. */
  readonly changeTracking?: {
    readonly compaction: GPUBuffer;
    readonly tileSizeCells: number;
    readonly tileCapacity: number;
    readonly tileChangeFlagsOffsetWords: number;
    readonly topologyReuseWord: number;
  };
}

export interface OctreeFineSeedAdapterOptions {
  /** Only leaves at or below this topology size seed the fine level-set band. */
  readonly finestLeafSize?: number;
  /** Signed-distance halo around the interface, in finest-cell widths. */
  readonly haloCells?: number;
  /** Authored analytic SDF used only before the first fine generation. */
  readonly analyticInitialCondition?: "dam-break" | "tank-fill";
  readonly initialFillFraction?: number;
}

export interface OctreeFineSeedAdapterSource extends OctreeFineSeedLeafResources {
  readonly plan: OctreeFineSeedAdapterPlan;
  readonly candidateCount: GPUBuffer;
}

/** The compact octree field retained outside the factor-m Section 5 band. */
export interface OctreeFineSeedAdapterCoarsePhiSource {
  /** One `{phi, minimumPhi, maximumPhi, flags}` record per compact row. */
  readonly values: GPUBuffer;
  /** Power-coarse publication control; words 2/11 are row count/valid. */
  readonly control: GPUBuffer;
}

function positiveDyadic(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1 || (value & (value - 1)) !== 0) {
    throw new RangeError(`${label} must be a positive power of two`);
  }
  return value;
}

export const octreeFineSeedCandidateShader = /* wgsl */ `
struct FineSeedLeaf { originX:u32,originY:u32,originZ:u32,size:u32,flags:u32,pad0:u32,pad1:u32,pad2:u32,phiGradient:vec4f,motion:vec4f }
struct Candidate { row:u32,flags:u32 }
struct Params { dimsCapacity:vec4u,selection:vec4u,cellHalo:vec4f,change:vec4u }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> fineSeedLeaves:array<FineSeedLeaf>;
@group(0) @binding(3) var<storage,read_write> candidateControl:array<u32>;
@group(0) @binding(4) var<storage,read> rowControl:array<u32>;
@group(0) @binding(5) var<storage,read> frontier:array<u32>;
@group(0) @binding(6) var<storage,read> compaction:array<u32>;
@group(0) @binding(7) var<storage,read_write> compactCandidates:array<Candidate>;
@group(0) @binding(9) var<storage,read_write> previousFineSeedLeaves:array<FineSeedLeaf>;
const CORE=${OCTREE_FINE_SEED_STATE.core}u;const HALO=${OCTREE_FINE_SEED_STATE.halo}u;const LIVE=${OCTREE_FINE_SEED_STATE.live}u;
fn deltaMode()->bool{return params.change.z!=0xffffffffu&&params.change.z<arrayLength(&compaction)&&compaction[params.change.z]==1u;}
fn dirtyLeaf(leaf:FineSeedLeaf)->bool{if(!deltaMode()){return true;}let tileSize=params.change.x;let limits=(params.dimsCapacity.xyz+vec3u(tileSize-1u))/tileSize;let tile=vec3u(leaf.originX,leaf.originY,leaf.originZ)/tileSize;let index=tile.x+limits.x*(tile.y+limits.y*tile.z);return index<params.change.y&&params.change.w+index<arrayLength(&compaction)&&compaction[params.change.w+index]==frontier[3];}
fn selectedCandidate(row:u32)->Candidate{
  if(row>=params.dimsCapacity.w||row>=arrayLength(&fineSeedLeaves)){return Candidate(${OCTREE_FINE_SEED_INVALID}u,0u);}
  let leaf=fineSeedLeaves[row];let delta=deltaMode();let selectedFlags=leaf.flags&(CORE|HALO);
  if((leaf.flags&LIVE)==0u||leaf.size>params.selection.x||!dirtyLeaf(leaf)||(!delta&&selectedFlags==0u)){
    return Candidate(${OCTREE_FINE_SEED_INVALID}u,0u);
  }
  return Candidate(row,selectedFlags);
}
var<workgroup> candidateCounts:array<u32,256>;
var<workgroup> publicationValid:u32;
@compute @workgroup_size(256) fn publishFineSeedCandidates(@builtin(local_invocation_index) lid:u32){
  let rows=params.dimsCapacity.w;let width=rows/256u;let remainder=rows%256u;
  let begin=lid*width+min(lid,remainder);let end=begin+width+select(0u,1u,lid<remainder);
  var localCount=0u;for(var row=begin;row<end;row+=1u){
    localCount+=select(0u,1u,selectedCandidate(row).row!=${OCTREE_FINE_SEED_INVALID}u);
  }
  candidateCounts[lid]=localCount;workgroupBarrier();
  for(var stride=1u;stride<256u;stride<<=1u){
    var add=0u;if(lid>=stride){add=candidateCounts[lid-stride];}
    workgroupBarrier();candidateCounts[lid]+=add;workgroupBarrier();
  }
  var cursor=candidateCounts[lid]-localCount;
  for(var row=begin;row<end;row+=1u){let candidate=selectedCandidate(row);
    if(candidate.row!=${OCTREE_FINE_SEED_INVALID}u){
      if(cursor<arrayLength(&compactCandidates)){compactCandidates[cursor]=candidate;}
      cursor+=1u;
    }
  }
  storageBarrier();workgroupBarrier();
  let count=candidateCounts[255u];let capacity=arrayLength(&compactCandidates);
  let pressureControl=arrayLength(&rowControl)-8u;let topologyError=select(1u,rowControl[pressureControl],arrayLength(&rowControl)>=8u);
  let generation=select(0u,frontier[3],arrayLength(&frontier)>3u);
  let error=select(topologyError,2u,count>capacity);
  if(lid==0u){
    candidateControl[0]=count;candidateControl[1]=(count+255u)/256u;
    candidateControl[2]=1u;candidateControl[3]=1u;candidateControl[4]=generation;
    candidateControl[6]=error;candidateControl[7]=rows;
    candidateControl[8]=select(0u,1u,deltaMode());
    publicationValid=select(0u,1u,generation!=0u&&error==0u);
    candidateControl[5]=publicationValid;
  }
  let valid=workgroupUniformLoad(&publicationValid);
  if(valid!=0u){for(var row=begin;row<end;row+=1u){let candidate=selectedCandidate(row);
    if(candidate.row!=${OCTREE_FINE_SEED_INVALID}u&&row<arrayLength(&previousFineSeedLeaves)){
      previousFineSeedLeaves[row]=fineSeedLeaves[row];
    }
  }}
}`;

/**
 * Produces compact leaf seeds for factor-m topology discovery. Leaf motion is
 * consumed directly from the native power-cell velocity publication.
 */
export class WebGPUOctreeFineSeedAdapter {
  readonly plan: OctreeFineSeedAdapterPlan;
  readonly leaves: GPUBuffer;
  readonly candidates: GPUBuffer;
  readonly countAndDispatch: GPUBuffer;
  private readonly previousFineSeedLeaves:GPUBuffer;
  private readonly coarseFallback:GPUBuffer;
  private readonly params: GPUBuffer;
  private bindGroup: GPUBindGroup;
  private readonly layout:GPUBindGroupLayout;
  private readonly device:GPUDevice;
  private readonly topology:OctreeFineSeedAdapterTopology;
  private powerVelocity?:OctreePowerVelocitySource;
  private coarsePhi?:OctreeFineSeedAdapterCoarsePhiSource;
  private readonly buildPipeline: GPUComputePipeline;
  private readonly publishCandidatesPipeline: GPUComputePipeline;
  private readonly selectBindGroup: GPUBindGroup;
  private readonly workgroups: number;
  private coarsePhiBindings=false;
  private powerVelocityBindings=false;
  private destroyed = false;

  constructor(
    device: GPUDevice,
    topology: OctreeFineSeedAdapterTopology,
    rowCapacity: number,
    options: OctreeFineSeedAdapterOptions = {},
  ) {
    this.device=device;this.topology=topology;
    this.plan=planOctreeFineSeedAdapter(rowCapacity);
    const dimensions = topology.dimensions;
    dimensions.forEach((value, axis) => {
      if (!Number.isSafeInteger(value) || value < 1 || value > 0xffffffff) {
        throw new RangeError(`Octree fine-seed dimension ${axis} must be an unsigned 32-bit integer`);
      }
    });
    topology.cellSize.forEach((value, axis) => {
      if (!Number.isFinite(value) || value <= 0) throw new RangeError(`Octree fine-seed cell size ${axis} must be positive`);
    });
    const finestLeafSize = positiveDyadic(options.finestLeafSize ?? 1, "Octree fine-seed finest leaf size");
    const haloCells = options.haloCells ?? 4;
    if (!Number.isFinite(haloCells) || haloCells < 0) throw new RangeError("Octree fine-seed halo must be finite and non-negative");

    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.leaves = device.createBuffer({ label: "Octree fine-seed leaf records", size: this.plan.leafBytes, usage: storage });
    this.previousFineSeedLeaves=device.createBuffer({label:"Previous octree fine-seed leaves",size:this.plan.leafBytes,usage:storage});
    this.coarseFallback=device.createBuffer({label:"Octree fine-seed adapter coarse-phi sentinel",size:128,usage:GPUBufferUsage.STORAGE});
    this.candidates = device.createBuffer({ label: "Compact octree fine-seed candidates", size: this.plan.candidateBytes, usage: storage });
    this.countAndDispatch = device.createBuffer({
      label: "Octree fine-seed candidate count and dispatch",
      size: OCTREE_FINE_SEED_ADAPTER_CONTROL_BYTES,
      usage: storage | GPUBufferUsage.COPY_SRC,
    });
    this.workgroups = Math.ceil(rowCapacity / 64);
    this.params = device.createBuffer({
      label: "Octree fine-seed adapter parameters",
      size: OCTREE_FINE_SEED_ADAPTER_PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const parameterData = new ArrayBuffer(OCTREE_FINE_SEED_ADAPTER_PARAMETER_BYTES);
    const analyticMode = options.analyticInitialCondition === "dam-break" ? 2
      : options.analyticInitialCondition === "tank-fill" ? 1 : 0;
    const initialFillFraction = options.initialFillFraction ?? 0;
    if (!Number.isFinite(initialFillFraction) || initialFillFraction < 0 || initialFillFraction > 1) {
      throw new RangeError("Octree analytic initial fill fraction must lie in [0, 1]");
    }
    new Uint32Array(parameterData).set([
      dimensions[0], dimensions[1], dimensions[2], rowCapacity,
      finestLeafSize, Math.ceil(haloCells), analyticMode, 0,
    ]);
    new Float32Array(parameterData)[7] = initialFillFraction;
    new Float32Array(parameterData).set([
      topology.cellSize[0], topology.cellSize[1], topology.cellSize[2],
      Math.ceil(haloCells) * Math.min(...topology.cellSize),
    ], 8);
    const change = topology.changeTracking;
    new Uint32Array(parameterData).set(change ? [
      change.tileSizeCells, change.tileCapacity, change.topologyReuseWord,
      change.tileChangeFlagsOffsetWords,
    ] : [1, 0, OCTREE_FINE_SEED_INVALID, 0], 12);
    device.queue.writeBuffer(this.params, 0, parameterData);

    this.layout = device.createBindGroupLayout({ label: "Octree fine-seed adapter layout", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ] });
    const shaderModule = device.createShaderModule({ label: "Octree topology to fine-level-set seeds", code: octreeFineSeedAdapterShader });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    this.buildPipeline = device.createComputePipeline({
      label: "Build octree fine-seed leaves",
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint: "buildFineSeedLeaves" },
    });
    const selectLayout = device.createBindGroupLayout({ label: "Octree fine-seed candidate selection layout", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ] });
    this.publishCandidatesPipeline = device.createComputePipeline({
      label: "Publish exact octree fine-seed candidates",
      layout: device.createPipelineLayout({ bindGroupLayouts: [selectLayout] }),
      compute: {
        module: device.createShaderModule({
          label: "Octree fine-seed candidate publication",
          code: octreeFineSeedCandidateShader,
        }),
        entryPoint: "publishFineSeedCandidates",
      },
    });
    this.selectBindGroup = device.createBindGroup({ label: "Octree fine-seed candidate selection bindings", layout: selectLayout, entries: [
      { binding: 0, resource: { buffer: this.params } },
      { binding: 1, resource: { buffer: this.leaves } },
      { binding: 3, resource: { buffer: this.countAndDispatch } },
      { binding: 4, resource: { buffer: this.topology.publicationControl ?? this.topology.rowCount } },
      { binding: 5, resource: { buffer: this.topology.frontier } },
      { binding: 6, resource: { buffer: this.topology.changeTracking?.compaction ?? this.coarseFallback } },
      { binding: 7, resource: { buffer: this.candidates } },
      { binding: 9, resource: { buffer: this.previousFineSeedLeaves } },
    ] });
    this.bindGroup = this.createBindGroup();
  }

  private createBindGroup(){return this.device.createBindGroup({ label: "Octree power fields to fine-seed-leaf bindings", layout:this.layout, entries: [
      { binding: 0, resource: { buffer:this.topology.leafHeaders } },
      { binding: 1, resource: { buffer:this.topology.rowCount } },
      { binding: 3, resource: { buffer:this.powerVelocity?.control??this.coarseFallback } },
      { binding: 4, resource: { buffer:this.powerVelocity?.velocities??this.coarseFallback } },
      { binding: 6, resource: { buffer: this.params } },
      { binding: 7, resource: { buffer: this.leaves } },
      {binding:10,resource:{buffer:this.coarsePhi?.values??this.coarseFallback}},
      {binding:11,resource:{buffer:this.coarsePhi?.control??this.coarseFallback}},
    ] });}

  /** Routes leaf motion to the native Section 5 power-cell publication. */
  setPowerVelocitySource(source:OctreePowerVelocitySource):void{
    if(source.plan.rowCapacity!==this.plan.rowCapacity){
      throw new RangeError("Octree fine-seed adapter and power-velocity row capacities must match");
    }
    this.powerVelocity=source;this.powerVelocityBindings=true;this.bindGroup=this.createBindGroup();
  }

  /** Routes recurring leaf classification to the paper's coarse-octree phi. */
  setCoarsePhiSource(source:OctreeFineSeedAdapterCoarsePhiSource):void{
    this.coarsePhi=source;this.coarsePhiBindings=true;this.bindGroup=this.createBindGroup();
  }

  /** True only after the adapter no longer depends on bootstrap phi. */
  get hasCoarsePhiBindings(){return this.coarsePhiBindings;}

  /** True only after leaf motion consumes native power-cell velocities. */
  get hasPowerVelocityBindings(){return this.powerVelocityBindings;}

  encode(broker: PassBroker): void {
    if (this.destroyed) return;
    const pass = broker.compute({ label: "Publish compact octree fine-seed leaves" });
    pass.setPipeline(this.buildPipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(this.workgroups);
    const publish = broker.compute({ label: "Publish exact octree fine-seed candidates" });
    publish.setPipeline(this.publishCandidatesPipeline);
    publish.setBindGroup(0, this.selectBindGroup);
    publish.dispatchWorkgroups(1);
  }

  get source(): OctreeFineSeedAdapterSource {
    const candidates: OctreeFineSeedCandidateSource = {
      candidates: this.candidates,
      countAndDispatch: this.countAndDispatch,
      indirectOffsetBytes: 4,
    };
    return { plan: this.plan, leaves: this.leaves, previousFineSeedLeaves: this.previousFineSeedLeaves, candidates, candidateCount: this.countAndDispatch };
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.leaves.destroy();
    this.previousFineSeedLeaves.destroy();
    this.coarseFallback.destroy();
    this.candidates.destroy();
    this.countAndDispatch.destroy();
    this.params.destroy();
  }
}

export const octreeFineSeedAdapterShader = /* wgsl */ `
struct LeafHeader { cell:u32,entryStart:u32,entryCount:u32,size:u32,diagonal:f32,rhs:f32,pad0:u32,pad1:u32,gradient:vec4f }
struct FineSeedLeaf { originX:u32,originY:u32,originZ:u32,size:u32,flags:u32,pad0:u32,pad1:u32,pad2:u32,phiGradient:vec4f,motion:vec4f }
struct Params { dimsCapacity:vec4u, selection:vec4u, cellHalo:vec4f, change:vec4u }
@group(0) @binding(0) var<storage,read> leafHeaders:array<LeafHeader>;
@group(0) @binding(1) var<storage,read> rowControl:array<u32>;
@group(0) @binding(3) var<storage,read> powerVelocityControl:array<u32>;
@group(0) @binding(4) var<storage,read> powerVelocities:array<vec4f>;
@group(0) @binding(6) var<uniform> params:Params;
@group(0) @binding(7) var<storage,read_write> fineSeedLeaves:array<FineSeedLeaf>;
struct CoarsePhi { phi:f32, minimumPhi:f32, maximumPhi:f32, flags:u32 }
@group(0) @binding(10) var<storage,read> coarsePhi:array<CoarsePhi>;
@group(0) @binding(11) var<storage,read> coarseControl:array<u32>;
const INVALID=0xffffffffu;const CORE=${OCTREE_FINE_SEED_STATE.core}u;const HALO=${OCTREE_FINE_SEED_STATE.halo}u;const LIVE=${OCTREE_FINE_SEED_STATE.live}u;
const COARSE_VALID=${OCTREE_COARSE_PHI_FLAG.valid}u;const COARSE_FINITE=${OCTREE_COARSE_PHI_FLAG.finite}u;
const POWER_VELOCITY_VALID=${OCTREE_POWER_VELOCITY_VALID}u;
fn dims()->vec3u{return params.dimsCapacity.xyz;}
fn cellCount()->u32{return dims().x*dims().y*dims().z;}
fn coord(cell:u32)->vec3u{return vec3u(cell%dims().x,(cell/dims().x)%dims().y,cell/(dims().x*dims().y));}
fn leafOrigin(leaf:FineSeedLeaf)->vec3u{return vec3u(leaf.originX,leaf.originY,leaf.originZ);}
fn valid(p:vec3i)->bool{return all(p>=vec3i(0))&&all(p<vec3i(dims()));}
fn analyticInitialPhi(point:vec3f)->f32{
  let fill=bitcast<f32>(params.selection.w);let world=vec3f((point.x/f32(dims().x)-0.5)*params.cellHalo.x*f32(dims().x),point.y*params.cellHalo.y,(point.z/f32(dims().z)-0.5)*params.cellHalo.z*f32(dims().z));
  if(params.selection.z==1u){return world.y-fill*params.cellHalo.y*f32(dims().y);}
  let heightFraction=max(0.92,fill);let footprintFraction=sqrt(fill/max(heightFraction,1e-9));let extent=params.cellHalo.xyz*vec3f(dims());let exposedMaximum=vec3f(-0.5*extent.x+footprintFraction*extent.x,heightFraction*extent.y,-0.5*extent.z+footprintFraction*extent.z);let q=world-exposedMaximum;return length(max(q,vec3f(0)))+min(max(q.x,max(q.y,q.z)),0.0);
}
fn finite(value:f32)->bool{return value==value&&abs(value)<3.402823e38;}
fn coarsePublicationValid()->bool{return arrayLength(&coarseControl)>=12u&&coarseControl[0]==0u
  &&coarseControl[2]>0u&&coarseControl[11]==0x80000000u;}
fn coarseRowValid(row:u32)->bool{return coarsePublicationValid()&&row<coarseControl[2]
  &&row<arrayLength(&coarsePhi)&&(coarsePhi[row].flags&(COARSE_VALID|COARSE_FINITE))==(COARSE_VALID|COARSE_FINITE)
  &&finite(coarsePhi[row].phi)&&finite(coarsePhi[row].minimumPhi)&&finite(coarsePhi[row].maximumPhi);}
fn liveRow(row:u32,header:LeafHeader)->bool{
  return header.cell<cellCount()&&row<rowControl[0];
}
fn powerVelocityPublicationValid()->bool{
  return arrayLength(&powerVelocityControl)>=8u&&powerVelocityControl[0]==POWER_VELOCITY_VALID
    &&powerVelocityControl[2]>0u&&powerVelocityControl[5]==powerVelocityControl[2]
    &&powerVelocityControl[7]!=0u;
}
fn powerVelocityRowValid(row:u32)->bool{
  if(!powerVelocityPublicationValid()||row>=powerVelocityControl[2]||row>=arrayLength(&powerVelocities)){return false;}
  let velocity=powerVelocities[row];
  return velocity.w==1.0&&finite(velocity.x)&&finite(velocity.y)&&finite(velocity.z);
}
@compute @workgroup_size(64) fn buildFineSeedLeaves(@builtin(global_invocation_id) gid:vec3u){
  let row=gid.x;if(row>=params.dimsCapacity.w||row>=arrayLength(&leafHeaders)||row>=arrayLength(&fineSeedLeaves)){return;}let header=leafHeaders[row];if(header.size==0u||!liveRow(row,header)||!powerVelocityRowValid(row)){fineSeedLeaves[row].flags=0u;return;}let origin=coord(header.cell);let centre=vec3f(origin)+vec3f(0.5*f32(header.size));let coarse=coarseRowValid(row);if(!coarse&&params.selection.z==0u){return;}var centrePhi=0.0;var minimumPhi=0.0;var maximumPhi=0.0;var gradient=vec3f(0.0);if(coarse){let sample=coarsePhi[row];centrePhi=sample.phi;minimumPhi=sample.minimumPhi;maximumPhi=sample.maximumPhi;}else{let sampleCell=vec3i(clamp(vec3u(centre),vec3u(0),dims()-vec3u(1)));centrePhi=analyticInitialPhi(vec3f(sampleCell)+vec3f(0.5));gradient=vec3f(0.5*(analyticInitialPhi(vec3f(sampleCell+vec3i(1,0,0))+vec3f(0.5))-analyticInitialPhi(vec3f(sampleCell-vec3i(1,0,0))+vec3f(0.5))),0.5*(analyticInitialPhi(vec3f(sampleCell+vec3i(0,1,0))+vec3f(0.5))-analyticInitialPhi(vec3f(sampleCell-vec3i(0,1,0))+vec3f(0.5))),0.5*(analyticInitialPhi(vec3f(sampleCell+vec3i(0,0,1))+vec3f(0.5))-analyticInitialPhi(vec3f(sampleCell-vec3i(0,0,1))+vec3f(0.5))));let radius=0.5*f32(header.size)*length(params.cellHalo.xyz);minimumPhi=centrePhi-radius;maximumPhi=centrePhi+radius;}let core=minimumPhi<=0.0&&maximumPhi>=0.0;let halo=!core&&abs(centrePhi)<=params.cellHalo.w;let candidateFlags=select(select(0u,HALO,halo),CORE,core);let flags=LIVE|candidateFlags;let motion=powerVelocities[row].xyz;fineSeedLeaves[row]=FineSeedLeaf(origin.x,origin.y,origin.z,header.size,flags,fineSeedLeaves[row].pad0,0u,0u,vec4f(centrePhi,gradient),vec4f(motion,length(motion)));
}
`;
