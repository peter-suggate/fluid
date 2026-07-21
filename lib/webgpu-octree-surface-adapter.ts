/**
 * GPU adapter from compact pressure/topology rows to the surface-page ABI.
 *
 * The adapter aliases the existing compact topology and adaptive face arena.
 * Its only persistent storage is one 48-byte SurfaceLeaf and one 8-byte
 * candidate slot per possible compact row, plus the 32-byte publication header.
 * It never allocates from the finest bounding-box volume.
 */

import { OCTREE_CONSUMER_MAX_FACE_CANDIDATES } from "./octree-consumer-sampling";
import type { OctreeFaceMirrorSource } from "./webgpu-octree-face-mirror";
import {
  OCTREE_SURFACE_CANDIDATE_BYTES,
  OCTREE_SURFACE_LEAF_RECORD_BYTES,
  OCTREE_SURFACE_STATE,
  validateOctreeSurfacePageSource,
  type OctreeSurfaceCandidateSource,
  type OctreeSurfacePageResources,
  type OctreeSurfacePageSource,
} from "./webgpu-octree-surface-pages";

/**
 * Candidate publication control:
 * [count, dispatch xyz, topology generation, published, error, capacity].
 * `published` deliberately does not depend on count, so a completed empty
 * liquid generation is distinct from an unpublished/failed empty producer.
 */
export const OCTREE_SURFACE_ADAPTER_CONTROL_BYTES = 32;
export const OCTREE_SURFACE_ADAPTER_PUBLICATION = Object.freeze({
  count: 0, dispatch: 1, generation: 4, published: 5, error: 6, capacity: 7,
} as const);
export const OCTREE_SURFACE_ADAPTER_PARAMETER_BYTES = 48;

export interface OctreeSurfaceAdapterPlan {
  readonly rowCapacity: number;
  readonly leafBytes: number;
  readonly candidateBytes: number;
  readonly allocatedBytes: number;
}

export function planOctreeSurfaceAdapter(rowCapacity: number): OctreeSurfaceAdapterPlan {
  if (!Number.isSafeInteger(rowCapacity) || rowCapacity < 1) {
    throw new RangeError("Octree surface adapter row capacity must be positive");
  }
  const leafBytes = rowCapacity * OCTREE_SURFACE_LEAF_RECORD_BYTES;
  const candidateBytes = rowCapacity * OCTREE_SURFACE_CANDIDATE_BYTES;
  return {
    rowCapacity,
    leafBytes,
    candidateBytes,
    allocatedBytes: leafBytes + candidateBytes
      + leafBytes + 128
      + OCTREE_SURFACE_ADAPTER_CONTROL_BYTES * 2 + OCTREE_SURFACE_ADAPTER_PARAMETER_BYTES,
  };
}

export interface OctreeSurfaceAdapterTopology {
  /** Existing 48-byte pressure LeafHeader rows. */
  readonly leafHeaders: GPUBuffer;
  /** Compact pressure control; word 0 is the live LeafHeader row count. */
  readonly rowCount: GPUBuffer;
  /** Full compact control arena including its trailing overflow words. */
  readonly publicationControl?: GPUBuffer;
  /** Persistent frontier; word 3 is the completed topology generation. */
  readonly frontier: GPUBuffer;
  readonly levelSet: GPUTexture;
  readonly dimensions: readonly [number, number, number];
  readonly cellSize: readonly [number, number, number];
}

export interface OctreeSurfaceAdapterOptions {
  /** Only leaves at or below this topology size receive fine surface pages. */
  readonly finestLeafSize?: number;
  /** Signed-distance halo around the interface, in finest-cell widths. */
  readonly haloCells?: number;
  /** Preserve and sample the previous page generation. Experimental and
   * opt-in so the stable path does not duplicate every compact leaf. */
  readonly directPageSampling?: boolean;
  /** Authored analytic SDF used only before the first page generation. */
  readonly analyticInitialCondition?: "dam-break" | "tank-fill";
  readonly initialFillFraction?: number;
}

export interface OctreeSurfaceAdapterSource extends OctreeSurfacePageResources {
  readonly plan: OctreeSurfaceAdapterPlan;
  readonly candidateCount: GPUBuffer;
}

function positiveDyadic(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1 || (value & (value - 1)) !== 0) {
    throw new RangeError(`${label} must be a positive power of two`);
  }
  return value;
}

export const octreeSurfaceCandidateShader = /* wgsl */ `
struct SurfaceLeaf { packedOrigin:u32,size:u32,flags:u32,pad:u32,phiGradient:vec4f,motion:vec4f }
struct Candidate { row:u32,flags:u32 }
struct Params { dimsCapacity:vec4u,selection:vec4u,cellHalo:vec4f }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> surfaceLeaves:array<SurfaceLeaf>;
@group(0) @binding(2) var<storage,read_write> candidates:array<Candidate>;
@group(0) @binding(3) var<storage,read_write> candidateControl:array<atomic<u32>>;
@group(0) @binding(4) var<storage,read> rowControl:array<u32>;
@group(0) @binding(5) var<storage,read> frontier:array<u32>;
const CORE=${OCTREE_SURFACE_STATE.core}u;const HALO=${OCTREE_SURFACE_STATE.halo}u;const LIVE=${OCTREE_SURFACE_STATE.live}u;
@compute @workgroup_size(64) fn selectSurfaceCandidates(@builtin(global_invocation_id) gid:vec3u){
  let row=gid.x;if(row>=params.dimsCapacity.w||row>=arrayLength(&surfaceLeaves)){return;}let leaf=surfaceLeaves[row];let candidateFlags=leaf.flags&(CORE|HALO);if((leaf.flags&LIVE)==0u||leaf.size>params.selection.x||candidateFlags==0u){return;}let output=atomicAdd(&candidateControl[0],1u);if(output<arrayLength(&candidates)){candidates[output]=Candidate(row,candidateFlags);}
}
@compute @workgroup_size(1) fn finalizeSurfaceCandidates(){
  let capacity=arrayLength(&candidates);let count=atomicLoad(&candidateControl[0]);
  let pressureControl=arrayLength(&rowControl)-8u;let topologyError=select(1u,rowControl[pressureControl],arrayLength(&rowControl)>=8u);
  let generation=select(0u,frontier[3],arrayLength(&frontier)>3u);
  let error=select(topologyError,2u,count>capacity);
  atomicStore(&candidateControl[4],generation);atomicStore(&candidateControl[6],error);
  atomicStore(&candidateControl[5],select(0u,1u,generation!=0u&&error==0u));
}`;

/**
 * Produces the exact input ABI consumed by WebGPUOctreeSurfacePages.
 *
 * Leaf motion uses the shared adaptive consumer sampler: every component is a
 * span-aware Shepard reconstruction over the row's bounded face-incidence
 * slab. The face arena is therefore the only velocity authority at this seam.
 */
export class WebGPUOctreeSurfaceAdapter {
  readonly plan: OctreeSurfaceAdapterPlan;
  readonly leaves: GPUBuffer;
  readonly candidates: GPUBuffer;
  readonly countAndDispatch: GPUBuffer;
  private readonly candidateTemplate: GPUBuffer;
  private readonly previousLeaves:GPUBuffer;
  private readonly ownsPreviousLeaves:boolean;
  private readonly surfaceFallback:GPUBuffer;
  private readonly params: GPUBuffer;
  private bindGroup: GPUBindGroup;
  private readonly layout:GPUBindGroupLayout;
  private readonly device:GPUDevice;
  private readonly topology:OctreeSurfaceAdapterTopology;
  private readonly faces:OctreeFaceMirrorSource;
  private readonly buildPipeline: GPUComputePipeline;
  private readonly selectPipeline: GPUComputePipeline;
  private readonly finalizeCandidatesPipeline: GPUComputePipeline;
  private readonly selectBindGroup: GPUBindGroup;
  private readonly workgroups: number;
  private pageNativePhiBindings=false;
  private destroyed = false;

  constructor(
    device: GPUDevice,
    topology: OctreeSurfaceAdapterTopology,
    faces: OctreeFaceMirrorSource,
    rowCapacity: number,
    options: OctreeSurfaceAdapterOptions = {},
  ) {
    this.device=device;this.topology=topology;this.faces=faces;
    const directPageSampling=options.directPageSampling===true;
    const planned=planOctreeSurfaceAdapter(rowCapacity);
    this.plan=directPageSampling?planned:{...planned,allocatedBytes:planned.allocatedBytes-planned.leafBytes};
    if (faces.plan.rowCapacity !== rowCapacity) {
      throw new RangeError("Octree surface adapter and face row capacities must match");
    }
    const dimensions = topology.dimensions;
    dimensions.forEach((value, axis) => {
      if (!Number.isSafeInteger(value) || value < 1 || value > 1024) {
        throw new RangeError(`Octree surface dimension ${axis} must be in [1, 1024]`);
      }
    });
    topology.cellSize.forEach((value, axis) => {
      if (!Number.isFinite(value) || value <= 0) throw new RangeError(`Octree surface cell size ${axis} must be positive`);
    });
    const finestLeafSize = positiveDyadic(options.finestLeafSize ?? 1, "Octree surface finest leaf size");
    const haloCells = options.haloCells ?? 4;
    if (!Number.isFinite(haloCells) || haloCells < 0) throw new RangeError("Octree surface halo must be finite and non-negative");

    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.leaves = device.createBuffer({ label: "Octree surface leaf adapter", size: this.plan.leafBytes, usage: storage });
    this.ownsPreviousLeaves=directPageSampling;
    this.previousLeaves=directPageSampling
      ? device.createBuffer({label:"Previous octree surface leaves",size:this.plan.leafBytes,usage:storage})
      : topology.leafHeaders;
    this.surfaceFallback=device.createBuffer({label:"Octree surface adapter page fallback",size:128,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.UNIFORM});
    this.candidates = device.createBuffer({ label: "Octree surface candidates", size: this.plan.candidateBytes, usage: storage });
    this.countAndDispatch = device.createBuffer({
      label: "Octree surface candidate count and dispatch",
      size: OCTREE_SURFACE_ADAPTER_CONTROL_BYTES,
      usage: storage,
    });
    this.workgroups = Math.ceil(rowCapacity / 64);
    this.candidateTemplate = device.createBuffer({
      label: "Octree surface candidate control template",
      size: OCTREE_SURFACE_ADAPTER_CONTROL_BYTES,
      usage: GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });
    new Uint32Array(this.candidateTemplate.getMappedRange()).set([
      0, this.workgroups, 1, 1, 0, 0, 0, rowCapacity,
    ]);
    this.candidateTemplate.unmap();
    this.params = device.createBuffer({
      label: "Octree surface adapter parameters",
      size: OCTREE_SURFACE_ADAPTER_PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const parameterData = new ArrayBuffer(OCTREE_SURFACE_ADAPTER_PARAMETER_BYTES);
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
    device.queue.writeBuffer(this.params, 0, parameterData);

    this.layout = device.createBindGroupLayout({ label: "Octree surface adapter layout", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ] });
    const shaderModule = device.createShaderModule({ label: "Octree topology to surface pages", code: octreeSurfaceAdapterShader });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    this.buildPipeline = device.createComputePipeline({
      label: "Build octree surface leaves",
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint: "buildSurfaceLeaves" },
    });
    const selectLayout = device.createBindGroupLayout({ label: "Octree surface candidate selection layout", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ] });
    this.selectPipeline = device.createComputePipeline({
      label: "Select octree surface candidates",
      layout: device.createPipelineLayout({ bindGroupLayouts: [selectLayout] }),
      compute: { module: device.createShaderModule({ label: "Octree surface candidate selection", code: octreeSurfaceCandidateShader }), entryPoint: "selectSurfaceCandidates" },
    });
    this.finalizeCandidatesPipeline = device.createComputePipeline({
      label: "Finalize octree surface candidate publication",
      layout: device.createPipelineLayout({ bindGroupLayouts: [selectLayout] }),
      compute: { module: device.createShaderModule({ label: "Octree surface candidate publication", code: octreeSurfaceCandidateShader }), entryPoint: "finalizeSurfaceCandidates" },
    });
    this.selectBindGroup = device.createBindGroup({ label: "Octree surface candidate selection bindings", layout: selectLayout, entries: [
      { binding: 0, resource: { buffer: this.params } },
      { binding: 1, resource: { buffer: this.leaves } },
      { binding: 2, resource: { buffer: this.candidates } },
      { binding: 3, resource: { buffer: this.countAndDispatch } },
      { binding: 4, resource: { buffer: this.topology.publicationControl ?? this.topology.rowCount } },
      { binding: 5, resource: { buffer: this.topology.frontier } },
    ] });
    this.bindGroup = this.createBindGroup();
  }

  private createBindGroup(surface?:OctreeSurfacePageSource, levelSet:GPUTexture=this.topology.levelSet){return this.device.createBindGroup({ label: "Octree surface adapter bindings", layout:this.layout, entries: [
      { binding: 0, resource: { buffer:this.topology.leafHeaders } },
      { binding: 1, resource: { buffer:this.topology.rowCount } },
      { binding: 2, resource:levelSet.createView() },
      { binding: 3, resource: { buffer:this.faces.control } },
      { binding: 4, resource: { buffer:this.faces.faces } },
      { binding: 5, resource: { buffer:this.faces.incidence } },
      { binding: 6, resource: { buffer: this.params } },
      { binding: 7, resource: { buffer: this.leaves } },
      {binding:10,resource:{buffer:this.previousLeaves}},{binding:11,resource:surface?.arena??{buffer:this.surfaceFallback}},{binding:12,resource:surface?.params??{buffer:this.surfaceFallback}},
    ] });}

  /** Reuses the previous generation's page hierarchy after dense bootstrap. */
  setSurfacePageSource(source:OctreeSurfacePageSource|undefined, sampledFallback?:GPUTexture){if(source)validateOctreeSurfacePageSource(source);this.pageNativePhiBindings=Boolean(source&&sampledFallback&&sampledFallback!==this.topology.levelSet);this.bindGroup=this.createBindGroup(source,sampledFallback);}

  /** True only after the recurring adapter group no longer retains dense phi. */
  get hasPageNativePhiBindings(){return this.pageNativePhiBindings;}

  encode(encoder: GPUCommandEncoder): void {
    if (this.destroyed) return;
    if(this.ownsPreviousLeaves) encoder.copyBufferToBuffer(this.leaves,0,this.previousLeaves,0,this.plan.leafBytes);
    encoder.copyBufferToBuffer(this.candidateTemplate, 0, this.countAndDispatch, 0, OCTREE_SURFACE_ADAPTER_CONTROL_BYTES);
    const pass = encoder.beginComputePass({ label: "Adapt octree topology to surface pages" });
    pass.setPipeline(this.buildPipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(this.workgroups);
    pass.end();
    const select = encoder.beginComputePass({ label: "Select octree surface page candidates" });
    select.setPipeline(this.selectPipeline);
    select.setBindGroup(0, this.selectBindGroup);
    select.dispatchWorkgroups(this.workgroups);
    select.end();
    const finalize = encoder.beginComputePass({ label: "Finalize octree surface candidate publication" });
    finalize.setPipeline(this.finalizeCandidatesPipeline);
    finalize.setBindGroup(0, this.selectBindGroup);
    finalize.dispatchWorkgroups(1);
    finalize.end();
  }

  get source(): OctreeSurfaceAdapterSource {
    const candidates: OctreeSurfaceCandidateSource = {
      candidates: this.candidates,
      countAndDispatch: this.countAndDispatch,
      indirectOffsetBytes: 4,
    };
    return { plan: this.plan, leaves: this.leaves, ...(this.ownsPreviousLeaves ? { previousLeaves: this.previousLeaves } : {}), candidates, candidateCount: this.countAndDispatch };
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.leaves.destroy();
    if(this.ownsPreviousLeaves) this.previousLeaves.destroy();
    this.surfaceFallback.destroy();
    this.candidates.destroy();
    this.countAndDispatch.destroy();this.candidateTemplate.destroy();
    this.params.destroy();
  }
}

export const octreeSurfaceAdapterShader = /* wgsl */ `
struct LeafHeader { cell:u32,entryStart:u32,entryCount:u32,size:u32,diagonal:f32,rhs:f32,pad0:u32,pad1:u32,gradient:vec4f }
struct FaceRecord { negativeRow:u32,positiveRow:u32,packedOrigin:u32,axisSpan:u32,normalVelocity:f32,area:f32 }
struct SurfaceLeaf { packedOrigin:u32,size:u32,flags:u32,pad:u32,phiGradient:vec4f,motion:vec4f }
struct Params { dimsCapacity:vec4u, selection:vec4u, cellHalo:vec4f }
@group(0) @binding(0) var<storage,read> leafHeaders:array<LeafHeader>;
@group(0) @binding(1) var<storage,read> rowControl:array<u32>;
@group(0) @binding(2) var levelSet:texture_3d<f32>;
@group(0) @binding(3) var<storage,read> faceControl:array<u32>;
@group(0) @binding(4) var<storage,read> faces:array<FaceRecord>;
@group(0) @binding(5) var<storage,read> incidence:array<u32>;
@group(0) @binding(6) var<uniform> params:Params;
@group(0) @binding(7) var<storage,read_write> surfaceLeaves:array<SurfaceLeaf>;
struct PageParams { shape:vec4u,offsets0:vec4u,offsets1:vec4u,offsets2:vec4u,cellDt:vec4f,spare0:vec4u,spare1:vec4u,spare2:vec4u }
@group(0) @binding(10) var<storage,read> previousLeaves:array<SurfaceLeaf>;
@group(0) @binding(11) var<storage,read> pageArena:array<u32>;
@group(0) @binding(12) var<uniform> pageParams:PageParams;
const INVALID=0xffffffffu;const CORE=${OCTREE_SURFACE_STATE.core}u;const HALO=${OCTREE_SURFACE_STATE.halo}u;const LIVE=${OCTREE_SURFACE_STATE.live}u;
const INCIDENCE_PER_ROW=${OCTREE_CONSUMER_MAX_FACE_CANDIDATES}u;const MAX_FACE_CANDIDATES=${OCTREE_CONSUMER_MAX_FACE_CANDIDATES}u;
fn dims()->vec3u{return params.dimsCapacity.xyz;}
fn cellCount()->u32{return dims().x*dims().y*dims().z;}
fn coord(cell:u32)->vec3u{return vec3u(cell%dims().x,(cell/dims().x)%dims().y,cell/(dims().x*dims().y));}
fn packOrigin(p:vec3u)->u32{return p.x|(p.y<<10u)|(p.z<<20u);}
fn unpackOrigin(word:u32)->vec3u{return vec3u(word&1023u,(word>>10u)&1023u,(word>>20u)&1023u);}
fn valid(p:vec3i)->bool{return all(p>=vec3i(0))&&all(p<vec3i(dims()));}
fn analyticInitialPhi(point:vec3f)->f32{
  let fill=bitcast<f32>(params.selection.w);let world=vec3f((point.x/f32(dims().x)-0.5)*params.cellHalo.x*f32(dims().x),point.y*params.cellHalo.y,(point.z/f32(dims().z)-0.5)*params.cellHalo.z*f32(dims().z));
  if(params.selection.z==1u){return world.y-fill*params.cellHalo.y*f32(dims().y);}
  let heightFraction=max(0.92,fill);let footprintFraction=sqrt(fill/max(heightFraction,1e-9));let extent=params.cellHalo.xyz*vec3f(dims());let half=0.5*vec3f(footprintFraction*extent.x,heightFraction*extent.y,footprintFraction*extent.z);let centre=vec3f(-0.5*extent.x+half.x,half.y,-0.5*extent.z+half.z);let q=abs(world-centre)-half;return length(max(q,vec3f(0)))+min(max(q.x,max(q.y,q.z)),0.0);
}
// A completed sparse generation remains authoritative even when it contains
// no resident detail pages: every live leaf still carries an affine phi plane.
// Requiring active pages here creates a circular dependency (the adapter must
// first recover candidates before pages can become active), while accepting
// only 4^3 pages silently rejects the production 2^3 configuration.
fn pagedPhiAvailable()->bool{let r=pageParams.shape.z;return (r==2u||r==4u)&&arrayLength(&pageArena)>7u&&pageArena[3]==0u&&pageArena[6]>0u&&pageArena[7]>0u;}
fn surfaceHash(q:vec3u)->u32{var h=(q.x*73856093u)^(q.y*19349663u)^(q.z*83492791u);h^=h>>16u;return h;}
fn previousContains(row:u32,p:vec3f)->bool{let leaf=previousLeaves[row];let o=vec3f(unpackOrigin(leaf.packedOrigin));return all(p>=o)&&all(p<o+vec3f(f32(leaf.size)));}
fn previousRow(p:vec3f)->u32{let mask=pageParams.offsets2.y-1u;for(var size=1u;size<=32u;size<<=1u){var slot=surfaceHash(vec3u(max(vec3f(0),floor(p/f32(size)))))&mask;for(var probe=0u;probe<32u;probe+=1u){let encoded=pageArena[pageParams.offsets1.y+slot];if(encoded==0u){break;}let row=encoded-1u;if(row<pageParams.shape.x&&row<arrayLength(&previousLeaves)&&previousContains(row,p)){return row;}slot=(slot+1u)&mask;}}return INVALID;}
fn airAliasRow(q:vec3u)->u32{let mask=pageParams.spare0.y-1u;var slot=surfaceHash(q)&mask;let key=packOrigin(q)+1u;for(var probe=0u;probe<32u;probe+=1u){let at=pageParams.spare0.x+2u*slot;let stored=pageArena[at];if(stored==0u){break;}if(stored==key){let encoded=pageArena[at+1u];if(encoded!=0u){let row=0xffffffffu-encoded;if(row<pageParams.shape.x&&row<arrayLength(&previousLeaves)){return row;}}}slot=(slot+1u)&mask;}return INVALID;}
fn pageLoad(base:u32,q:vec3u)->f32{let r=pageParams.shape.z;return bitcast<f32>(pageArena[base+q.x+r*(q.y+r*q.z)]);}
fn previousFallback(leaf:SurfaceLeaf,p:vec3f)->f32{let c=vec3f(unpackOrigin(leaf.packedOrigin))+vec3f(0.5*f32(leaf.size));let physicalGradient=leaf.phiGradient.yzw/max(params.cellHalo.xyz,vec3f(1e-9));let boundedGradient=physicalGradient/max(1.0,length(physicalGradient))*params.cellHalo.xyz;return leaf.phiGradient.x+dot(boundedGradient,p-c);}
fn previousPhi(row:u32,p:vec3f)->f32{let leaf=previousLeaves[row];let slot=pageArena[pageParams.offsets0.x+row];if(slot==INVALID||slot>=pageParams.shape.y){return previousFallback(leaf,p);}let o=vec3f(unpackOrigin(leaf.packedOrigin));let r=pageParams.shape.z;let grid=clamp((p-o)/f32(leaf.size)*f32(r)-vec3f(0.5),vec3f(0),vec3f(f32(r-1u)));let a=vec3u(floor(grid));let b=min(a+vec3u(1),vec3u(r-1u));let t=fract(grid);let base=pageParams.offsets1.z+slot*pageParams.shape.w;return mix(mix(mix(pageLoad(base,a),pageLoad(base,vec3u(b.x,a.y,a.z)),t.x),mix(pageLoad(base,vec3u(a.x,b.y,a.z)),pageLoad(base,vec3u(b.x,b.y,a.z)),t.x),t.y),mix(mix(pageLoad(base,vec3u(a.x,a.y,b.z)),pageLoad(base,vec3u(b.x,a.y,b.z)),t.x),mix(pageLoad(base,vec3u(a.x,b.y,b.z)),pageLoad(base,b),t.x),t.y),t.z);}
var<private> currentRow:u32=INVALID;
fn phi(p:vec3i)->f32{let q=clamp(p,vec3i(0),vec3i(dims())-vec3i(1));if(!pagedPhiAvailable()){if(params.selection.z!=0u){return analyticInitialPhi(vec3f(q)+vec3f(0.5));}return textureLoad(levelSet,q,0).x;}let point=vec3f(q)+vec3f(0.5);let row=previousRow(point);if(row!=INVALID){return previousPhi(row,point);}let airRow=airAliasRow(vec3u(q));if(airRow!=INVALID){return clamp(previousFallback(previousLeaves[airRow],point),0.01*min(params.cellHalo.x,min(params.cellHalo.y,params.cellHalo.z)),params.cellHalo.w);}if(currentRow<arrayLength(&previousLeaves)){return previousFallback(previousLeaves[currentRow],point);}return params.cellHalo.w;}
fn faceAxis(face:FaceRecord)->u32{return face.axisSpan&3u;}
fn faceSpan(face:FaceRecord)->u32{return face.axisSpan>>2u;}
fn faceCentre(face:FaceRecord)->vec3f{let axis=faceAxis(face);var p=vec3f(unpackOrigin(face.packedOrigin));let half=0.5*f32(faceSpan(face));p[(axis+1u)%3u]+=half;p[(axis+2u)%3u]+=half;return p;}
fn liveRow(row:u32,header:LeafHeader)->bool{
  // This is the compact row count that drives pressure and face publication in
  // the same command stream. Consuming it directly avoids a second frontier
  // hash lookup and an intermediate copied counter at the sparse-page seam.
  return header.cell<cellCount()&&row<rowControl[0]&&row<faceControl[3];
}
// Same span-aware bounded-incidence reconstruction as octreeConsumerComponent.
fn sampleMotionComponent(point:vec3f,row:u32,axis:u32)->f32{
  if(row>=faceControl[3]||row>=arrayLength(&incidence)){return 0.0;}var weighted=0.0;var weights=0.0;var nearest=0.0;var nearestD2=3.402823e38;
  let count=min(incidence[row],INCIDENCE_PER_ROW);for(var i=0u;i<count&&i<MAX_FACE_CANDIDATES;i+=1u){let at=faceControl[3]+row*INCIDENCE_PER_ROW+i;if(at>=arrayLength(&incidence)){break;}let faceIndex=incidence[at];if(faceIndex>=faceControl[0]||faceIndex>=faceControl[2]||faceIndex>=arrayLength(&faces)){continue;}let face=faces[faceIndex];if(faceAxis(face)!=axis){continue;}let delta=point-faceCentre(face);let d2=dot(delta,delta);if(d2<nearestD2){nearestD2=d2;nearest=face.normalVelocity;}let support=max(1.0,f32(faceSpan(face)));let weight=1.0/max(0.0625*support*support,d2);weighted+=weight*face.normalVelocity;weights+=weight;}return select(nearest,weighted/weights,weights>0.0);
}
fn sampleMotion(point:vec3f,row:u32)->vec3f{return vec3f(sampleMotionComponent(point,row,0u),sampleMotionComponent(point,row,1u),sampleMotionComponent(point,row,2u));}
fn interfaceRange(origin:vec3u,size:u32,centrePhi:f32)->vec2f{let radius=0.5*f32(size)*length(params.cellHalo.xyz);var range=vec2f(centrePhi-radius,centrePhi+radius);let half=i32(size/2u);let c=vec3i(origin)+vec3i(half);for(var axis=0u;axis<3u;axis+=1u){var negative=c;var positive=c;negative[axis]=i32(origin[axis])-1;positive[axis]=i32(origin[axis]+size);if(valid(negative)){let value=phi(negative);range=vec2f(min(range.x,value),max(range.y,value));}if(valid(positive)){let value=phi(positive);range=vec2f(min(range.x,value),max(range.y,value));}}return range;}
@compute @workgroup_size(64) fn buildSurfaceLeaves(@builtin(global_invocation_id) gid:vec3u){
  let row=gid.x;if(row>=params.dimsCapacity.w||row>=arrayLength(&leafHeaders)||row>=arrayLength(&surfaceLeaves)){return;}currentRow=row;let header=leafHeaders[row];if(header.size==0u||!liveRow(row,header)){surfaceLeaves[row].flags=0u;return;}let origin=coord(header.cell);let centre=vec3f(origin)+vec3f(0.5*f32(header.size));let sampleCell=vec3i(clamp(vec3u(centre),vec3u(0),dims()-vec3u(1)));let centrePhi=phi(sampleCell);let dx=0.5*(phi(sampleCell+vec3i(1,0,0))-phi(sampleCell-vec3i(1,0,0)));let dy=0.5*(phi(sampleCell+vec3i(0,1,0))-phi(sampleCell-vec3i(0,1,0)));let dz=0.5*(phi(sampleCell+vec3i(0,0,1))-phi(sampleCell-vec3i(0,0,1)));let range=interfaceRange(origin,header.size,centrePhi);let core=range.x<=0.0&&range.y>=0.0;let halo=!core&&abs(centrePhi)<=params.cellHalo.w;let candidateFlags=select(select(0u,HALO,halo),CORE,core);let flags=LIVE|candidateFlags;let motion=sampleMotion(centre,row);surfaceLeaves[row]=SurfaceLeaf(packOrigin(origin),header.size,flags,0u,vec4f(centrePhi,dx,dy,dz),vec4f(motion,length(motion)));
}
`;
