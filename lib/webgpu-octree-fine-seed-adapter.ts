/**
 * GPU adapter from compact pressure/topology rows to factor-m fine-level-set
 * seed candidates. It aliases the existing compact topology and adaptive face
 * arena and never allocates from the finest bounding-box volume.
 */

import { PassBroker } from "./webgpu-pass-broker";
import type { DirectStructuredVelocitySource } from "./webgpu-octree-structured-velocity-gpu";
import {
  OCTREE_FINE_SEED_CANDIDATE_BYTES,
  OCTREE_FINE_SEED_INVALID,
  OCTREE_FINE_SEED_LEAF_RECORD_BYTES,
  OCTREE_FINE_SEED_STATE,
  type OctreeFineSeedCandidateSource,
  type OctreeFineSeedLeafResources,
} from "./octree-fine-seed-leaves";
import { OCTREE_COARSE_PHI_FLAG } from "./octree-coarse-levelset";
import type { GPUInitializationTask } from "./gpu-initialization";

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
export const OCTREE_FINE_SEED_ADAPTER_PARAMETER_BYTES = 80;
export const OCTREE_FINE_SEED_ADAPTER_DISPATCH_BYTES = 24;
/** Below this authored compact-row capacity, both maintenance kernels stay in
 * one pass and stride the live row count inside one workgroup. */
export const OCTREE_FINE_SEED_PERSISTENT_ROW_CAPACITY = 4096;

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
      + leafBytes + 128 + OCTREE_FINE_SEED_ADAPTER_DISPATCH_BYTES
      + OCTREE_FINE_SEED_ADAPTER_CONTROL_BYTES + OCTREE_FINE_SEED_ADAPTER_PARAMETER_BYTES
      + OCTREE_FINE_SEED_ADAPTER_DISPATCH_BYTES,
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
  /** Optional absolute corner-anchored dam extents in world metres. */
  readonly initialDamBreakDimensions?: readonly [number, number, number];
  /** The +y cutoff is not a solid wall when the authored container is open. */
  readonly openTopBoundary?: boolean;
  /**
   * Imported dense t=0 level set for scenes with no closed-form surface. It is
   * always bound; `analyticInitialCondition` decides which authority the seed
   * kernel actually samples.
   */
  readonly bootstrapLevelSet: GPUTexture;
  readonly deferPipelineCompilation?: boolean;
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
@group(0) @binding(4) var<storage,read> structuredControl:array<u32>;
@group(0) @binding(5) var<storage,read> frontier:array<u32>;
@group(0) @binding(6) var<storage,read> compaction:array<u32>;
@group(0) @binding(7) var<storage,read_write> compactCandidates:array<Candidate>;
@group(0) @binding(9) var<storage,read_write> previousFineSeedLeaves:array<FineSeedLeaf>;
@group(0) @binding(10) var<storage,read> sourceRowCount:array<u32>;
@group(0) @binding(11) var<storage,read_write> dispatchMetadata:array<u32>;
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
@compute @workgroup_size(1) fn publishFineSeedAdapterDispatch(){
  let rawSourceRows=select(0u,sourceRowCount[0],arrayLength(&sourceRowCount)>0u);
  let sourceRows=min(rawSourceRows,params.dimsCapacity.w);
  let validStructured=arrayLength(&structuredControl)>=6u&&structuredControl[0]==0u
    &&structuredControl[3]!=0u&&structuredControl[4]<=1u;
  let sourceValid=validStructured&&rawSourceRows<=params.dimsCapacity.w
    &&sourceRows==structuredControl[2];
  let rows=select(0u,sourceRows,sourceValid);
  dispatchMetadata[0]=(rows+63u)/64u;dispatchMetadata[1]=1u;dispatchMetadata[2]=1u;
  dispatchMetadata[3]=select(0u,1u,rows>0u);dispatchMetadata[4]=1u;dispatchMetadata[5]=1u;
  if(rows==0u){
    let generation=select(0u,frontier[3],arrayLength(&frontier)>3u);
    candidateControl[0]=0u;candidateControl[1]=0u;candidateControl[2]=1u;candidateControl[3]=1u;
    candidateControl[4]=generation;candidateControl[5]=select(0u,1u,sourceValid&&generation!=0u);
    candidateControl[6]=select(1u,0u,sourceValid);candidateControl[7]=params.dimsCapacity.w;
    candidateControl[8]=select(0u,1u,deltaMode());
  }
}
@compute @workgroup_size(256) fn publishFineSeedCandidates(@builtin(local_invocation_index) lid:u32){
  let rows=min(params.dimsCapacity.w,structuredControl[2]);let width=rows/256u;let remainder=rows%256u;
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
  let rawSourceRows=select(0u,sourceRowCount[0],arrayLength(&sourceRowCount)>0u);
  let sourceValid=arrayLength(&structuredControl)>=6u&&structuredControl[0]==0u
    &&structuredControl[3]!=0u&&structuredControl[4]<=1u&&rawSourceRows<=params.dimsCapacity.w
    &&rawSourceRows==structuredControl[2];
  let topologyError=select(1u,0u,sourceValid);
  let generation=select(0u,frontier[3],arrayLength(&frontier)>3u);
  let error=select(topologyError,2u,count>capacity);
  if(lid==0u){
    candidateControl[0]=count;candidateControl[1]=(count+255u)/256u;
    candidateControl[2]=1u;candidateControl[3]=1u;candidateControl[4]=generation;
    candidateControl[6]=error;candidateControl[7]=params.dimsCapacity.w;
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
  /** Fail-closed placeholder used only until all mandatory GPU authorities are wired. */
  private readonly bindingSentinel:GPUBuffer;
  /** Writable binding-11 placeholder for the candidate consumer. It must not
   * alias the read-only unpublished-authority sentinel in the same bind group. */
  private readonly dispatchBindingSentinel:GPUBuffer;
  private readonly params: GPUBuffer;
  /** The planner writes this directly as STORAGE; consumers read it as
   * INDIRECT after the explicit pass boundary. Their bind group points binding
   * 11 at a distinct sentinel, so no same-pass storage/indirect alias exists. */
  private readonly dispatch: GPUBuffer;
  private bindGroup: GPUBindGroup;
  private readonly layout:GPUBindGroupLayout;
  private readonly device:GPUDevice;
  private readonly topology:OctreeFineSeedAdapterTopology;
  private readonly bootstrapLevelSet:GPUTexture;
  private structuredVelocity?:DirectStructuredVelocitySource;
  private coarsePhi?:OctreeFineSeedAdapterCoarsePhiSource;
  private buildPipeline!: GPUComputePipeline;
  private planDispatchPipeline!: GPUComputePipeline;
  private publishCandidatesPipeline!: GPUComputePipeline;
  private readonly pipelineLayout: GPUPipelineLayout;
  private readonly candidatePipelineLayout: GPUPipelineLayout;
  private shaderModule?: GPUShaderModule;
  private candidateModule?: GPUShaderModule;
  private readonly pipelinesDeferred: boolean;
  private plannerBindGroup: GPUBindGroup;
  private selectBindGroup: GPUBindGroup;
  private readonly selectLayout:GPUBindGroupLayout;
  private coarsePhiBindings=false;
  private structuredVelocityBindings=false;
  private destroyed = false;

  constructor(
    device: GPUDevice,
    topology: OctreeFineSeedAdapterTopology,
    rowCapacity: number,
    options: OctreeFineSeedAdapterOptions,
  ) {
    this.device=device;this.topology=topology;
    this.pipelinesDeferred = true;
    this.bootstrapLevelSet=options.bootstrapLevelSet;
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
    this.bindingSentinel=device.createBuffer({label:"Octree fine-seed adapter unpublished-authority sentinel",size:128,usage:GPUBufferUsage.STORAGE});
    this.dispatchBindingSentinel=device.createBuffer({
      label:"Octree fine-seed candidate dispatch binding sentinel",
      size:OCTREE_FINE_SEED_ADAPTER_DISPATCH_BYTES,
      usage:GPUBufferUsage.STORAGE,
    });
    this.candidates = device.createBuffer({ label: "Compact octree fine-seed candidates", size: this.plan.candidateBytes, usage: storage });
    this.countAndDispatch = device.createBuffer({
      label: "Octree fine-seed candidate count and dispatch",
      size: OCTREE_FINE_SEED_ADAPTER_CONTROL_BYTES,
      usage: storage | GPUBufferUsage.COPY_SRC,
    });
    this.dispatch = device.createBuffer({
      label: "Octree fine-seed direct dispatch",
      size: OCTREE_FINE_SEED_ADAPTER_DISPATCH_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_SRC,
    });
    this.params = device.createBuffer({
      label: "Octree fine-seed adapter parameters",
      size: OCTREE_FINE_SEED_ADAPTER_PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const parameterData = new ArrayBuffer(OCTREE_FINE_SEED_ADAPTER_PARAMETER_BYTES);
    const analyticMode = options.analyticInitialCondition === "dam-break" ? 2
      : options.analyticInitialCondition === "tank-fill" ? 1 : 3;
    const initialFillFraction = options.initialFillFraction ?? 0;
    if (!Number.isFinite(initialFillFraction) || initialFillFraction < 0 || initialFillFraction > 1) {
      throw new RangeError("Octree analytic initial fill fraction must lie in [0, 1]");
    }
    const initialDamBreakDimensions = options.initialDamBreakDimensions ?? [0, 0, 0];
    if (initialDamBreakDimensions.some((value, axis) => !Number.isFinite(value) || value < 0
      || value > topology.cellSize[axis] * dimensions[axis])) {
      throw new RangeError("Octree analytic dam dimensions must lie inside the domain");
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
    new Float32Array(parameterData).set(initialDamBreakDimensions, 16);
    new Uint32Array(parameterData)[19] = options.openTopBoundary ? 1 : 0;
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
      { binding: 12, visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
    ] });
    this.pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    this.selectLayout = device.createBindGroupLayout({ label: "Octree fine-seed candidate selection layout", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ] });
    this.candidatePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.selectLayout] });
    this.plannerBindGroup = this.createSelectBindGroup(this.dispatch);
    this.selectBindGroup = this.createSelectBindGroup(this.dispatchBindingSentinel);
    this.bindGroup = this.createBindGroup();
  }

  private buildDescriptor(): GPUComputePipelineDescriptor {
    this.shaderModule ??= this.device.createShaderModule({
      label: "Octree topology to fine-level-set seeds", code: octreeFineSeedAdapterShader,
    });
    return { label: "Build octree fine-seed leaves", layout: this.pipelineLayout,
      compute: { module: this.shaderModule, entryPoint: "buildFineSeedLeaves" } };
  }

  private candidateDescriptor(entryPoint: "publishFineSeedAdapterDispatch" | "publishFineSeedCandidates"): GPUComputePipelineDescriptor {
    this.candidateModule ??= this.device.createShaderModule({
      label: "Octree fine-seed candidate publication", code: octreeFineSeedCandidateShader,
    });
    return { label: entryPoint === "publishFineSeedAdapterDispatch"
      ? "Publish exact octree fine-seed dispatch" : "Publish exact octree fine-seed candidates",
      layout: this.candidatePipelineLayout, compute: { module: this.candidateModule, entryPoint } };
  }

  initializationTasks(): GPUInitializationTask[] {
    if (!this.pipelinesDeferred) return [];
    return [
      { id: "octree.fine-seeds.pipeline.build", phase: "adaptive-topology",
        label: "Compile compact fine-seed builder",
        run: async () => { this.buildPipeline = await this.device.createComputePipelineAsync(this.buildDescriptor()); } },
      { id: "octree.fine-seeds.pipeline.plan", phase: "adaptive-topology",
        label: "Compile fine-seed dispatch planner",
        run: async () => { this.planDispatchPipeline = await this.device.createComputePipelineAsync(
          this.candidateDescriptor("publishFineSeedAdapterDispatch")); } },
      { id: "octree.fine-seeds.pipeline.publish", phase: "adaptive-topology",
        label: "Compile fine-seed candidate publisher",
        run: async () => { this.publishCandidatesPipeline = await this.device.createComputePipelineAsync(
          this.candidateDescriptor("publishFineSeedCandidates")); } },
    ];
  }

  private createSelectBindGroup(dispatchBinding: GPUBuffer){return this.device.createBindGroup({
    label: dispatchBinding === this.dispatch
      ? "Octree fine-seed dispatch planner bindings" : "Octree fine-seed candidate selection bindings",
    layout: this.selectLayout, entries: [
      { binding: 0, resource: { buffer: this.params } },
      { binding: 1, resource: { buffer: this.leaves } },
      { binding: 3, resource: { buffer: this.countAndDispatch } },
      { binding: 4, resource: { buffer: this.structuredVelocity?.control ?? this.bindingSentinel } },
      { binding: 5, resource: { buffer: this.topology.frontier } },
      { binding: 6, resource: { buffer: this.topology.changeTracking?.compaction ?? this.bindingSentinel } },
      { binding: 7, resource: { buffer: this.candidates } },
      { binding: 9, resource: { buffer: this.previousFineSeedLeaves } },
      { binding: 10, resource: { buffer: this.topology.rowCount } },
      { binding: 11, resource: { buffer: dispatchBinding } },
    ] });}

  private createBindGroup(){return this.device.createBindGroup({ label: "Octree power fields to fine-seed-leaf bindings", layout:this.layout, entries: [
      { binding: 0, resource: { buffer:this.topology.leafHeaders } },
      { binding: 1, resource: { buffer:this.topology.rowCount } },
      { binding: 3, resource: { buffer:this.structuredVelocity?.control??this.bindingSentinel } },
      { binding: 4, resource: { buffer:this.structuredVelocity?.rowVelocities??this.bindingSentinel } },
      { binding: 6, resource: { buffer: this.params } },
      { binding: 7, resource: { buffer: this.leaves } },
      {binding:10,resource:{buffer:this.coarsePhi?.values??this.bindingSentinel}},
      {binding:11,resource:{buffer:this.coarsePhi?.control??this.bindingSentinel}},
      {binding:12,resource:this.bootstrapLevelSet.createView({dimension:"3d"})},
    ] });}

  /** Routes leaf motion to the accepted packed structured-velocity bank. */
  setStructuredVelocitySource(source:DirectStructuredVelocitySource):void{
    if(source.plan.rowCapacity!==this.plan.rowCapacity){
      throw new RangeError("Octree fine-seed adapter and structured-velocity row capacities must match");
    }
    this.structuredVelocity=source;this.structuredVelocityBindings=true;this.bindGroup=this.createBindGroup();
    this.plannerBindGroup=this.createSelectBindGroup(this.dispatch);
    this.selectBindGroup=this.createSelectBindGroup(this.dispatchBindingSentinel);
  }

  /** Routes recurring leaf classification to the paper's coarse-octree phi. */
  setCoarsePhiSource(source:OctreeFineSeedAdapterCoarsePhiSource):void{
    this.coarsePhi=source;this.coarsePhiBindings=true;this.bindGroup=this.createBindGroup();
  }

  /** True only after the adapter no longer depends on bootstrap phi. */
  get hasCoarsePhiBindings(){return this.coarsePhiBindings;}

  /** True only after leaf motion consumes the accepted structured authority. */
  get hasStructuredVelocityBindings(){return this.structuredVelocityBindings;}

  encode(broker: PassBroker): void {
    if (this.destroyed) return;
    if (this.plan.rowCapacity <= OCTREE_FINE_SEED_PERSISTENT_ROW_CAPACITY) {
      // Both kernels visit only the GPU-published live rows. Dispatch ordering
      // provides storage visibility, so the compact path needs neither a
      // planner pass nor an indirect-usage transition.
      const pass = broker.compute({ label: "Maintain compact octree fine seeds" });
      pass.setPipeline(this.buildPipeline);
      pass.setBindGroup(0, this.bindGroup);
      pass.dispatchWorkgroups(1);
      pass.setPipeline(this.publishCandidatesPipeline);
      pass.setBindGroup(0, this.selectBindGroup);
      pass.dispatchWorkgroups(1);
      return;
    }
    const planner = broker.compute({ label: "Publish exact octree fine-seed dispatch" });
    planner.setPipeline(this.planDispatchPipeline);
    planner.setBindGroup(0, this.plannerBindGroup);
    planner.dispatchWorkgroups(1);
    broker.fence("fine-seed direct dispatch publication");
    const pass = broker.compute({ label: "Publish compact octree fine-seed leaves" });
    pass.setPipeline(this.buildPipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroupsIndirect(this.dispatch, 0);
    const publish = broker.compute({ label: "Publish exact octree fine-seed candidates" });
    publish.setPipeline(this.publishCandidatesPipeline);
    publish.setBindGroup(0, this.selectBindGroup);
    publish.dispatchWorkgroupsIndirect(this.dispatch, 12);
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
    this.bindingSentinel.destroy();
    this.dispatchBindingSentinel.destroy();
    this.candidates.destroy();
    this.countAndDispatch.destroy();
    this.params.destroy();
    this.dispatch.destroy();
  }
}

export const octreeFineSeedAdapterShader = /* wgsl */ `
struct LeafHeader { cell:u32,entryStart:u32,entryCount:u32,size:u32,diagonal:f32,rhs:f32,pad0:u32,pad1:u32,gradient:vec4f }
struct FineSeedLeaf { originX:u32,originY:u32,originZ:u32,size:u32,flags:u32,pad0:u32,pad1:u32,pad2:u32,phiGradient:vec4f,motion:vec4f }
struct Params { dimsCapacity:vec4u, selection:vec4u, cellHalo:vec4f, change:vec4u, damDimensions:vec4f }
@group(0) @binding(0) var<storage,read> leafHeaders:array<LeafHeader>;
@group(0) @binding(3) var<storage,read> structuredVelocityControl:array<u32>;
@group(0) @binding(4) var<storage,read> structuredRowVelocities:array<vec4f>;
@group(0) @binding(6) var<uniform> params:Params;
@group(0) @binding(7) var<storage,read_write> fineSeedLeaves:array<FineSeedLeaf>;
struct CoarsePhi { phi:f32, minimumPhi:f32, maximumPhi:f32, flags:u32 }
@group(0) @binding(10) var<storage,read> coarsePhi:array<CoarsePhi>;
@group(0) @binding(11) var<storage,read> coarseControl:array<u32>;
const INVALID=0xffffffffu;const CORE=${OCTREE_FINE_SEED_STATE.core}u;const HALO=${OCTREE_FINE_SEED_STATE.halo}u;const LIVE=${OCTREE_FINE_SEED_STATE.live}u;
const COARSE_VALID=${OCTREE_COARSE_PHI_FLAG.valid}u;const COARSE_FINITE=${OCTREE_COARSE_PHI_FLAG.finite}u;
fn dims()->vec3u{return params.dimsCapacity.xyz;}
fn cellCount()->u32{return dims().x*dims().y*dims().z;}
fn coord(cell:u32)->vec3u{return vec3u(cell%dims().x,(cell/dims().x)%dims().y,cell/(dims().x*dims().y));}
fn leafOrigin(leaf:FineSeedLeaf)->vec3u{return vec3u(leaf.originX,leaf.originY,leaf.originZ);}
fn valid(p:vec3i)->bool{return all(p>=vec3i(0))&&all(p<vec3i(dims()));}
@group(0) @binding(12) var bootstrapLevelSetIn:texture_3d<f32>;
// Cell-unit sample of the imported dense t=0 level set. Texture values live at
// coarse cell centres, while an even-sized octree leaf has its geometric centre
// on a cell boundary. Interpolate in the cell-centred lattice instead of
// rounding that centre toward +x/+y/+z; the latter shifts opposite faces in the
// same world direction and destroys reflection symmetry by half a coarse cell.
fn bootstrapTexturePhi(point:vec3f)->f32{
  let lattice=point-vec3f(0.5);let base=vec3i(floor(lattice));let t=fract(lattice);
  var value=0.0;for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    let high=vec3<bool>((corner&1u)!=0u,(corner&2u)!=0u,(corner&4u)!=0u);
    let weight=select(1.0-t.x,t.x,high.x)*select(1.0-t.y,t.y,high.y)
      *select(1.0-t.z,t.z,high.z);
    value+=weight*textureLoad(bootstrapLevelSetIn,
      clamp(base+offset,vec3i(0),vec3i(dims())-vec3i(1)),0).x;
  }return value;}
fn bootstrapPhi(point:vec3f)->f32{
  if(params.selection.z==3u){return bootstrapTexturePhi(point);}
  return analyticInitialPhi(point);}
fn analyticInitialPhi(point:vec3f)->f32{
  let fill=bitcast<f32>(params.selection.w);let world=vec3f((point.x/f32(dims().x)-0.5)*params.cellHalo.x*f32(dims().x),point.y*params.cellHalo.y,(point.z/f32(dims().z)-0.5)*params.cellHalo.z*f32(dims().z));
  if(params.selection.z==1u){return world.y-fill*params.cellHalo.y*f32(dims().y);}
  let heightFraction=max(0.92,fill);let footprintFraction=sqrt(fill/max(heightFraction,1e-9));let extent=params.cellHalo.xyz*vec3f(dims());let fallback=vec3f(footprintFraction*extent.x,heightFraction*extent.y,footprintFraction*extent.z);let authored=any(params.damDimensions.xyz>vec3f(0.0));let damDimensions=select(fallback,params.damDimensions.xyz,authored);let exposedMaximum=vec3f(-0.5*extent.x+damDimensions.x,damDimensions.y,-0.5*extent.z+damDimensions.z);let q=world-exposedMaximum;return length(max(q,vec3f(0)))+min(max(q.x,max(q.y,q.z)),0.0);
}
fn finite(value:f32)->bool{return value==value&&abs(value)<3.402823e38;}
fn coarsePublicationValid()->bool{return arrayLength(&coarseControl)>=12u&&coarseControl[0]==0u
  &&coarseControl[2]>0u&&coarseControl[11]==0x80000000u;}
fn coarseRowValid(row:u32)->bool{return coarsePublicationValid()&&row<coarseControl[2]
  &&row<arrayLength(&coarsePhi)&&(coarsePhi[row].flags&(COARSE_VALID|COARSE_FINITE))==(COARSE_VALID|COARSE_FINITE)
  &&finite(coarsePhi[row].phi)&&finite(coarsePhi[row].minimumPhi)&&finite(coarsePhi[row].maximumPhi);}
fn liveRow(row:u32,header:LeafHeader)->bool{
  return header.cell<cellCount()&&structuredVelocityPublicationValid()
    &&row<structuredVelocityControl[2];
}
fn structuredVelocityPublicationValid()->bool{
  return arrayLength(&structuredVelocityControl)>=6u&&structuredVelocityControl[0]==0u
    &&structuredVelocityControl[2]>0u&&structuredVelocityControl[3]!=0u
    &&structuredVelocityControl[4]<=1u;
}
fn structuredVelocityRowIndex(row:u32)->u32{return structuredVelocityControl[4]*params.dimsCapacity.w+row;}
fn structuredVelocityRowValid(row:u32)->bool{
  let at=structuredVelocityRowIndex(row);
  if(!structuredVelocityPublicationValid()||row>=structuredVelocityControl[2]||at>=arrayLength(&structuredRowVelocities)){return false;}
  let velocity=structuredRowVelocities[at];
  return velocity.w==1.0&&finite(velocity.x)&&finite(velocity.y)&&finite(velocity.z);
}
@compute @workgroup_size(64) fn buildFineSeedLeaves(@builtin(global_invocation_id) gid:vec3u,
 @builtin(num_workgroups) nwg:vec3u){let lanes=max(1u,nwg.x*64u);let liveRows=min(params.dimsCapacity.w,structuredVelocityControl[2]);
 for(var row=gid.x;row<liveRows&&row<arrayLength(&leafHeaders)&&row<arrayLength(&fineSeedLeaves);row+=lanes){let header=leafHeaders[row];if(header.size==0u||!liveRow(row,header)||!structuredVelocityRowValid(row)){fineSeedLeaves[row].flags=0u;continue;}let origin=coord(header.cell);let centre=vec3f(origin)+vec3f(0.5*f32(header.size));let coarse=coarseRowValid(row);if(!coarse&&params.selection.z==0u){continue;}var centrePhi=0.0;var minimumPhi=0.0;var maximumPhi=0.0;var gradient=vec3f(0.0);if(coarse){let sample=coarsePhi[row];centrePhi=sample.phi;minimumPhi=sample.minimumPhi;maximumPhi=sample.maximumPhi;}else{centrePhi=bootstrapPhi(centre);gradient=vec3f(0.5*(bootstrapPhi(centre+vec3f(1,0,0))-bootstrapPhi(centre-vec3f(1,0,0))),0.5*(bootstrapPhi(centre+vec3f(0,1,0))-bootstrapPhi(centre-vec3f(0,1,0))),0.5*(bootstrapPhi(centre+vec3f(0,0,1))-bootstrapPhi(centre-vec3f(0,0,1))));let radius=0.5*f32(header.size)*length(params.cellHalo.xyz);minimumPhi=centrePhi-radius;maximumPhi=centrePhi+radius;}let openTop=bitcast<u32>(params.damDimensions.w)!=0u;
  // Liquid against a closed lid has no in-domain sign-changing edge. Publish
  // the cutoff leaf so the fine topology exists while that surface separates.
  let virtualInterface=!openTop&&origin.y+header.size>=dims().y&&centrePhi<0.0;let core=(minimumPhi<=0.0&&maximumPhi>=0.0)||virtualInterface;let halo=!core&&abs(centrePhi)<=params.cellHalo.w;let candidateFlags=select(select(0u,HALO,halo),CORE,core);let flags=LIVE|candidateFlags;let motion=structuredRowVelocities[structuredVelocityRowIndex(row)].xyz;fineSeedLeaves[row]=FineSeedLeaf(origin.x,origin.y,origin.z,header.size,flags,fineSeedLeaves[row].pad0,0u,0u,vec4f(centrePhi,gradient),vec4f(motion,length(motion)));}
}
`;
