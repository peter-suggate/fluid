import { OCTREE_CONSUMER_MAX_FACE_CANDIDATES } from "./octree-consumer-sampling";
import { WebGPUOctreeFaceTopologyTransfer } from "./webgpu-octree-face-transfer";
import { validateOctreeSurfacePageSource, type OctreeSurfacePageSource } from "./webgpu-octree-surface-pages";

/**
 * Canonical face ABI: two incident rows, an explicit u32 origin per axis,
 * axis/span, normal velocity, and area.  Origins deliberately do not reuse
 * the legacy 10:10:10 leaf packing: connected worlds may be wider than 1023
 * finest cells on any axis without aliasing their physical faces.
 */
export const OCTREE_GPU_FACE_RECORD_BYTES = 32;
// A row owns at most 6*4 fragments and can receive the same bound from
// neighbouring publishers. Keep both halves in one row-local slab.
export const OCTREE_GPU_FACE_INCIDENCE_PER_ROW = OCTREE_CONSUMER_MAX_FACE_CANDIDATES;

export interface OctreeFaceMirrorPlan {
  rowCapacity: number;
  faceCapacity: number;
  faceBytes: number;
  incidenceBytes: number;
  offsetBytes?: number;
  allocatedBytes: number;
}

/**
 * A leaf publishes at most four fragments on each of its six sides.  Air and
 * domain-boundary faces are one-sided, so there is no global half-incidence
 * argument that can reduce this allocation bound: in the worst legal
 * topology every one of the 24 candidates can be canonical and live.
 *
 * This is deliberately the publisher's proved bound rather than an observed
 * average.  An undersized arena sets the shared overflow word and invalidates
 * every downstream velocity publication.
 */
export const OCTREE_GPU_FACE_CANDIDATES_PER_ROW = 6 * 4;
export const OCTREE_GPU_FACE_DEFAULT_CAPACITY_PER_ROW = OCTREE_GPU_FACE_CANDIDATES_PER_ROW;

export function planOctreeFaceMirror(
  rowCapacity: number,
  facesPerRow = OCTREE_GPU_FACE_DEFAULT_CAPACITY_PER_ROW,
): OctreeFaceMirrorPlan {
  if (!Number.isInteger(rowCapacity) || rowCapacity < 1) throw new RangeError("Octree face mirror row capacity must be positive");
  if (!Number.isFinite(facesPerRow) || facesPerRow <= 0) throw new RangeError("Octree face mirror density must be positive");
  const faceCapacity = Math.max(1, Math.ceil(rowCapacity * facesPerRow));
  const faceBytes = faceCapacity * OCTREE_GPU_FACE_RECORD_BYTES;
  // One atomic count per row followed by a fixed, bounded incidence slab.
  const incidenceBytes = rowCapacity * (1 + OCTREE_GPU_FACE_INCIDENCE_PER_ROW) * 4;
  const offsetBytes = (rowCapacity + 1) * 4;
  return { rowCapacity, faceCapacity, faceBytes, incidenceBytes, offsetBytes, allocatedBytes: 84 + faceBytes + incidenceBytes + offsetBytes };
}

export interface OctreeFaceMirrorResources {
  velocity: GPUTexture;
  levelSet: GPUTexture;
  owners: GPUBuffer;
  leafHeaders: GPUBuffer;
  frontier: GPUBuffer;
  compaction: GPUBuffer;
  params: GPUBuffer;
  pressureA: GPUBuffer;
  pressureB: GPUBuffer;
  projectedVelocity: GPUTexture;
}

export interface OctreeFaceMirrorOptions {
  /** Preserve canonical velocities across deterministic topology rebuilds. */
  preserveTopologyVelocities?: boolean;
  /** Retain face-index transfer records for inspection. Production needs only the counters. */
  retainTopologyTransferRecords?: boolean;
  /** Exact finest-grid bounds for dimension-scaled canonical-key radix work. */
  dimensions?: readonly [number, number, number];
}

export interface OctreeFaceMirrorSource {
  plan: OctreeFaceMirrorPlan;
  control: GPUBuffer;
  faces: GPUBuffer;
  incidence: GPUBuffer;
  parity: GPUBuffer;
  projectionParity?: GPUBuffer;
  projectionParityOffset?: number;
  /** Four words: maximum |divergence|, sampled rows, non-finite rows, sum divergence squared. */
  projectedDivergence?: GPUBuffer;
  projectedDivergenceOffset?: number;
  topologyTransferDiagnostics?: GPUBuffer;
}

/**
 * Adaptive face store. Face indices are deterministic: a count/scan/emit
 * sequence orders them first by compact leaf row and then by local face slot.
 * Incidence append order is irrelevant because reduction sorts the bounded
 * list of face indices before accumulating it.
 */
export class WebGPUOctreeFaceMirror {
  readonly plan: OctreeFaceMirrorPlan;
  readonly control: GPUBuffer;
  readonly faces: GPUBuffer;
  readonly incidence: GPUBuffer;
  readonly parity: GPUBuffer;
  private readonly offsets: GPUBuffer;
  private readonly faceDispatch: GPUBuffer;
  private readonly surfaceFallback: GPUBuffer;
  private readonly countPipeline: GPUComputePipeline;
  private readonly scanPipeline: GPUComputePipeline;
  private readonly emitPipeline: GPUComputePipeline;
  private readonly parityPipeline: GPUComputePipeline;
  private readonly applyRhsPipeline: GPUComputePipeline;
  private readonly projectPipeline: GPUComputePipeline;
  private readonly projectionParityPipeline: GPUComputePipeline;
  private readonly projectedDivergencePipeline: GPUComputePipeline;
  private readonly topologyTransfer?: WebGPUOctreeFaceTopologyTransfer;
  private topologyPublished = false;
  private pagedSurfaceAttached = false;
  private pageNativePhiBindings = false;
  private topologyBindGroup: GPUBindGroup;
  private readonly scanBindGroup: GPUBindGroup;
  private rhsBindGroup: GPUBindGroup;
  private bindGroups: { pressureA: GPUBindGroup; pressureB: GPUBindGroup };
  private readonly rhsLayout: GPUBindGroupLayout;
  private readonly topologyLayout: GPUBindGroupLayout;
  private readonly projectionLayout: GPUBindGroupLayout;
  private readonly device: GPUDevice;
  private readonly resources: OctreeFaceMirrorResources;

  constructor(device: GPUDevice, resources: OctreeFaceMirrorResources, rowCapacity: number, options: OctreeFaceMirrorOptions = {}) {
    this.device = device; this.resources = resources;
    this.plan = planOctreeFaceMirror(rowCapacity);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    // [count, overflow, capacity, rowCapacity, maximum incidence,
    // incidence-overflow appends]. The adapter consumes the authoritative
    // compact-row count directly from compaction rather than duplicating it.
    this.control = device.createBuffer({ label: "Octree face mirror control", size: 24, usage: storage });
    this.faces = device.createBuffer({ label: "Octree canonical face mirror", size: this.plan.faceBytes, usage: storage });
    this.incidence = device.createBuffer({ label: "Octree face incidence mirror", size: this.plan.incidenceBytes, usage: storage });
    this.parity = device.createBuffer({ label: "Octree face RHS, projection parity, and divergence", size: 48, usage: storage });
    this.offsets = device.createBuffer({ label: "Octree deterministic face offsets", size: this.plan.offsetBytes!, usage: storage });
    this.faceDispatch = device.createBuffer({ label: "Octree face indirect dispatch", size: 12, usage: storage | GPUBufferUsage.INDIRECT });
    this.surfaceFallback = device.createBuffer({ label: "Octree face surface-page fallback", size: 128, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM });
    device.queue.writeBuffer(this.control, 8, new Uint32Array([this.plan.faceCapacity, rowCapacity]));
    const layout = device.createBindGroupLayout({ label: "Octree face mirror layout", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ] });
    this.rhsLayout = layout;
    const shaderModule = device.createShaderModule({ label: "Octree face mirror", code: octreeFaceMirrorShader });
    this.topologyLayout = device.createBindGroupLayout({label:"Octree paged face topology layout",entries:[
      ...[0,1].map((binding)=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
      ...[2,5].map((binding)=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage" as const}})),
      ...[3,4,7,8,9,11].map((binding)=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}})),
      {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
      {binding:15,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:16,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
      {binding:17,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
    ]});
    const topologyPipeline = (label: string, entryPoint: string) => device.createComputePipeline({
      label,
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.topologyLayout] }),
      compute: { module: shaderModule, entryPoint },
    });
    this.countPipeline = topologyPipeline("Count canonical octree faces", "countFaces");
    const scanLayout = device.createBindGroupLayout({ label: "Octree face count scan layout", entries: [
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ] });
    this.scanPipeline = device.createComputePipeline({
      label: "Scan canonical octree face counts",
      layout: device.createPipelineLayout({ bindGroupLayouts: [scanLayout] }),
      compute: { module: shaderModule, entryPoint: "scanFaceCounts" },
    });
    this.scanBindGroup = device.createBindGroup({ label: "Octree face count scan bindings", layout: scanLayout, entries: [
      { binding: 5, resource: { buffer: resources.compaction } },
      { binding: 7, resource: { buffer: this.control } },
      { binding: 11, resource: { buffer: this.offsets } },
      { binding: 12, resource: { buffer: this.faceDispatch } },
    ] });
    this.emitPipeline = topologyPipeline("Publish canonical octree faces", "publishFaces");
    this.parityPipeline = device.createComputePipeline({
      label: "Compare octree face mirror RHS",
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module: shaderModule, entryPoint: "reduceRhsParity" },
    });
    this.applyRhsPipeline = device.createComputePipeline({
      label: "Apply octree face mirror RHS",
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module: shaderModule, entryPoint: "applyFaceRhs" },
    });
    this.projectedDivergencePipeline = device.createComputePipeline({
      label: "Reduce projected octree face divergence",
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module: shaderModule, entryPoint: "reduceProjectedDivergence" },
    });
    this.projectionLayout = device.createBindGroupLayout({ label: "Octree face projection layout", entries: [
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 13, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 14, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 15, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 16, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 17, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ] });
    const projectionPipeline = (label: string, entryPoint: string) => device.createComputePipeline({ label, layout: device.createPipelineLayout({ bindGroupLayouts: [this.projectionLayout] }), compute: { module: shaderModule, entryPoint } });
    this.projectPipeline = projectionPipeline("Project canonical octree faces", "projectFaces");
    this.projectionParityPipeline = projectionPipeline("Compare adaptive and dense projected faces", "reduceProjectionParity");
    const commonEntries: GPUBindGroupEntry[] = [
      { binding: 0, resource: resources.velocity.createView() },
      { binding: 1, resource: resources.levelSet.createView() },
      { binding: 2, resource: { buffer: resources.owners } },
      { binding: 3, resource: { buffer: resources.leafHeaders } },
      { binding: 4, resource: { buffer: resources.frontier } },
      { binding: 5, resource: { buffer: resources.compaction } },
      { binding: 6, resource: { buffer: resources.params } },
      { binding: 7, resource: { buffer: this.control } },
      { binding: 8, resource: { buffer: this.faces } },
      { binding: 9, resource: { buffer: this.incidence } },
      { binding: 10, resource: { buffer: this.parity } },
      { binding: 11, resource: { buffer: this.offsets } },
      { binding: 12, resource: { buffer: this.faceDispatch } },
    ];
    this.rhsBindGroup = device.createBindGroup({ label: "Octree face RHS bindings", layout, entries: commonEntries });
    this.topologyBindGroup = this.createTopologyBindGroup();
    const projectionGroup = (pressure: GPUBuffer) => device.createBindGroup({ label: "Octree face projection bindings", layout: this.projectionLayout, entries: [
      { binding: 1, resource: resources.levelSet.createView() }, { binding: 2, resource: { buffer: resources.owners } }, { binding:4,resource:{buffer:resources.frontier} },
      { binding: 6, resource: { buffer: resources.params } }, { binding: 7, resource: { buffer: this.control } },
      { binding: 8, resource: { buffer: this.faces } }, { binding: 10, resource: { buffer: this.parity } },
      { binding: 13, resource: { buffer: pressure } }, { binding: 14, resource: resources.projectedVelocity.createView() },
      { binding:15,resource:{buffer:this.surfaceFallback}},{binding:16,resource:{buffer:this.surfaceFallback}},{binding:17,resource:{buffer:this.surfaceFallback}},
    ] });
    this.bindGroups = { pressureA: projectionGroup(resources.pressureA), pressureB: projectionGroup(resources.pressureB) };
    if (options.preserveTopologyVelocities) {
      this.topologyTransfer = new WebGPUOctreeFaceTopologyTransfer(device, this.source, {
        retainRecords: options.retainTopologyTransferRecords,
        keyDimensions: options.dimensions,
      });
      this.plan = { ...this.plan, allocatedBytes: this.plan.allocatedBytes + this.topologyTransfer.plan.allocatedBytes };
    }
  }

  private createTopologyBindGroup(surface?:OctreeSurfacePageSource, levelSet:GPUTexture=this.resources.levelSet){const r=this.resources;return this.device.createBindGroup({label:"Octree face construction bindings",layout:this.topologyLayout,entries:[
    {binding:0,resource:r.velocity.createView()},{binding:1,resource:levelSet.createView()},{binding:2,resource:{buffer:r.owners}},{binding:3,resource:{buffer:r.leafHeaders}},{binding:4,resource:{buffer:r.frontier}},{binding:5,resource:{buffer:r.compaction}},{binding:6,resource:{buffer:r.params}},{binding:7,resource:{buffer:this.control}},{binding:8,resource:{buffer:this.faces}},{binding:9,resource:{buffer:this.incidence}},{binding:11,resource:{buffer:this.offsets}},{binding:15,resource:surface?.arena??{buffer:this.surfaceFallback}},{binding:16,resource:surface?.params??{buffer:this.surfaceFallback}},{binding:17,resource:surface?.leaves??{buffer:this.surfaceFallback}},
  ]});}

  private createRhsBindGroup(levelSet:GPUTexture){const r=this.resources;return this.device.createBindGroup({label:"Octree face RHS bindings",layout:this.rhsLayout,entries:[
    {binding:0,resource:r.velocity.createView()},{binding:1,resource:levelSet.createView()},{binding:2,resource:{buffer:r.owners}},{binding:3,resource:{buffer:r.leafHeaders}},{binding:4,resource:{buffer:r.frontier}},{binding:5,resource:{buffer:r.compaction}},{binding:6,resource:{buffer:r.params}},{binding:7,resource:{buffer:this.control}},{binding:8,resource:{buffer:this.faces}},{binding:9,resource:{buffer:this.incidence}},{binding:10,resource:{buffer:this.parity}},{binding:11,resource:{buffer:this.offsets}},{binding:12,resource:{buffer:this.faceDispatch}},
  ]});}

  /** Switches free-surface classification to pages after their first lifecycle publication. */
  setSurfacePageSource(source:OctreeSurfacePageSource|undefined, sampledFallback:GPUTexture=this.resources.levelSet){if(source)validateOctreeSurfacePageSource(source);this.pagedSurfaceAttached=Boolean(source);this.pageNativePhiBindings=Boolean(source)&&sampledFallback!==this.resources.levelSet;this.topologyBindGroup=this.createTopologyBindGroup(source,sampledFallback);this.rhsBindGroup=this.createRhsBindGroup(sampledFallback);const projection=(pressure:GPUBuffer)=>this.device.createBindGroup({label:"Octree face projection bindings",layout:this.projectionLayout,entries:[{binding:1,resource:sampledFallback.createView()},{binding:2,resource:{buffer:this.resources.owners}},{binding:4,resource:{buffer:this.resources.frontier}},{binding:6,resource:{buffer:this.resources.params}},{binding:7,resource:{buffer:this.control}},{binding:8,resource:{buffer:this.faces}},{binding:10,resource:{buffer:this.parity}},{binding:13,resource:{buffer:pressure}},{binding:14,resource:this.resources.projectedVelocity.createView()},{binding:15,resource:source?.arena??{buffer:this.surfaceFallback}},{binding:16,resource:source?.params??{buffer:this.surfaceFallback}},{binding:17,resource:source?.leaves??{buffer:this.surfaceFallback}}]});this.bindGroups={pressureA:projection(this.resources.pressureA),pressureB:projection(this.resources.pressureB)};}

  /** True only after every subsequently encoded face group has shed dense phi. */
  get hasPageNativePhiBindings(){return this.pageNativePhiBindings;}

  encode(encoder: GPUCommandEncoder, rowDispatch: GPUBuffer, applyRhs = false): void {
    this.encodeTopology(encoder, rowDispatch);
    this.encodeRhs(encoder, rowDispatch, applyRhs);
  }

  /** Publish faces and restore old canonical velocities onto the new IDs. */
  encodeTopology(encoder: GPUCommandEncoder, rowDispatch: GPUBuffer): void {
    // Face records still contain the previous presentation here, even though
    // leaf topology has already changed. Capture them before publication
    // overwrites the arena; the transfer is applied after the new deterministic
    // IDs exist. The cold publication deliberately retains its dense seed.
    if (this.topologyPublished) this.topologyTransfer?.encodeCapture(encoder);
    encoder.clearBuffer(this.control, 0, 8);
    encoder.clearBuffer(this.control, 16, 8);
    encoder.clearBuffer(this.incidence, 0, this.plan.rowCapacity * 4);
    encoder.clearBuffer(this.offsets);
    const pass = encoder.beginComputePass({ label: "Deterministically publish canonical octree faces" });
    pass.setBindGroup(0, this.topologyBindGroup);
    pass.setPipeline(this.countPipeline);
    pass.dispatchWorkgroupsIndirect(rowDispatch, 0);
    pass.setPipeline(this.scanPipeline); pass.setBindGroup(0, this.scanBindGroup); pass.dispatchWorkgroups(1);
    pass.setBindGroup(0, this.topologyBindGroup);
    pass.setPipeline(this.emitPipeline); pass.dispatchWorkgroupsIndirect(rowDispatch, 0);
    pass.end();
    if (this.topologyPublished) this.topologyTransfer?.encodeTransfer(encoder);
    this.topologyPublished = true;
  }

  /** Reduce transported face fluxes only after topology/transport is final. */
  encodeRhs(encoder: GPUCommandEncoder, rowDispatch: GPUBuffer, applyRhs = false): void {
    encoder.clearBuffer(this.parity);
    const parity = encoder.beginComputePass({ label: "Compare octree face mirror RHS" });
    parity.setPipeline(this.parityPipeline);
    parity.setBindGroup(0, this.rhsBindGroup);
    parity.dispatchWorkgroupsIndirect(rowDispatch, 0);
    parity.end();
    if (applyRhs) {
      const apply = encoder.beginComputePass({ label: "Apply octree face mirror RHS" });
      apply.setPipeline(this.applyRhsPipeline);
      apply.setBindGroup(0, this.rhsBindGroup);
      apply.dispatchWorkgroupsIndirect(rowDispatch, 0);
      apply.end();
    }
  }

  encodeProjection(encoder: GPUCommandEncoder, pressureInA: boolean): void {
    const pass = encoder.beginComputePass({ label: "Project canonical octree face velocities" });
    pass.setPipeline(this.projectPipeline); pass.setBindGroup(0, pressureInA ? this.bindGroups.pressureA : this.bindGroups.pressureB);
    pass.dispatchWorkgroupsIndirect(this.faceDispatch, 0); pass.end();
  }

  encodeProjectionParity(encoder: GPUCommandEncoder, pressureInA: boolean): void {
    // Once pages are authoritative, the dense projected texture is only a
    // compatibility publication with stale deep-band phi. Comparing the two
    // representations reports the publication error, not projection error.
    if (this.pagedSurfaceAttached) { encoder.clearBuffer(this.parity,16,16); return; }
    const pass = encoder.beginComputePass({ label: "Compare adaptive and dense projected faces" });
    pass.setPipeline(this.projectionParityPipeline); pass.setBindGroup(0, pressureInA ? this.bindGroups.pressureA : this.bindGroups.pressureB);
    pass.dispatchWorkgroupsIndirect(this.faceDispatch, 0); pass.end();
  }

  /** Reduce the final published compact fluxes after all projection authorities. */
  encodeProjectedDivergence(encoder: GPUCommandEncoder): void {
    encoder.clearBuffer(this.parity, 32, 16);
    const pass = encoder.beginComputePass({ label: "Measure projected octree face divergence" });
    pass.setPipeline(this.projectedDivergencePipeline);
    pass.setBindGroup(0, this.rhsBindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.plan.rowCapacity / 256));
    pass.end();
  }

  get source(): OctreeFaceMirrorSource {
    return {
      plan: this.plan, control: this.control, faces: this.faces, incidence: this.incidence, parity: this.parity,
      projectionParity: this.parity, projectionParityOffset: 16,
      projectedDivergence: this.parity, projectedDivergenceOffset: 32,
      topologyTransferDiagnostics: this.topologyTransfer?.diagnostics,
    };
  }

  get topologyTransferDiagnostics(): GPUBuffer | undefined { return this.topologyTransfer?.diagnostics; }

  destroy(): void {
    this.control.destroy(); this.faces.destroy(); this.incidence.destroy(); this.parity.destroy();
    this.offsets.destroy(); this.faceDispatch.destroy();
    this.surfaceFallback.destroy();
    this.topologyTransfer?.destroy();
  }
}

export const octreeFaceMirrorShader = /* wgsl */ `
struct Owner { origin: vec3u, size: u32 }
struct Params { dimsMax: vec4u, cellRelax: vec4f, control: vec4u, solve: vec4f, container: vec4f, inflowPositionRadius: vec4f, inflowDirectionLength: vec4f, physical: vec4f, pressureCapacity: vec4u, hydrostatic: vec4f }
struct LeafHeader { cell: u32, entryStart: u32, entryCount: u32, size: u32, diagonal: f32, rhs: f32, pad0: u32, pad1: u32, gradient: vec4f }
struct SurfaceLeaf { originX:u32,originY:u32,originZ:u32,size:u32,flags:u32,pad0:u32,pad1:u32,pad2:u32,phiGradient:vec4f,motion:vec4f }
struct FaceRecord { negativeRow: u32, positiveRow: u32, originX: u32, originY: u32, originZ: u32, axisSpan: u32, normalVelocity: f32, area: f32 }
struct Candidate { valid: bool, axis: u32, side: i32, span: u32, origin: vec3u, neighbor: Owner, neighborLiquid: bool }

@group(0) @binding(0) var velocityIn: texture_3d<f32>;
@group(0) @binding(1) var levelSetIn: texture_3d<f32>;
@group(0) @binding(2) var<storage, read> owners: array<u32>;
@group(0) @binding(3) var<storage, read_write> leafHeaders: array<LeafHeader>;
@group(0) @binding(4) var<storage, read_write> frontier: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> compaction: array<u32>;
@group(0) @binding(6) var<uniform> params: Params;
// face count, overflow, capacity, row capacity
@group(0) @binding(7) var<storage, read_write> control: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> faces: array<FaceRecord>;
// row counts first, then rowCapacity * 24 face indices
@group(0) @binding(9) var<storage, read_write> incidence: array<atomic<u32>>;
// max absolute difference, max reference RHS, mismatched rows, compared rows
@group(0) @binding(10) var<storage, read_write> parity: array<atomic<u32>>;
@group(0) @binding(11) var<storage, read_write> faceOffsets: array<u32>;
@group(0) @binding(12) var<storage, read_write> faceDispatch: array<u32>;
@group(0) @binding(13) var<storage, read> pressure: array<f32>;
@group(0) @binding(14) var projectedVelocity: texture_3d<f32>;
struct SurfaceParams { shape:vec4u,offsets0:vec4u,offsets1:vec4u,offsets2:vec4u,cellDt:vec4f,spare0:vec4u,spare1:vec4u,spare2:vec4u }
@group(0) @binding(15) var<storage,read> surfaceArena:array<u32>;
@group(0) @binding(16) var<uniform> surfaceParams:SurfaceParams;
@group(0) @binding(17) var<storage,read> surfaceLeaves:array<SurfaceLeaf>;

const INVALID = 0xffffffffu;
const INCIDENCE_PER_ROW = ${OCTREE_GPU_FACE_INCIDENCE_PER_ROW}u;
fn dims() -> vec3u { return params.dimsMax.xyz; }
fn valid(p: vec3i) -> bool { return all(p >= vec3i(0)) && all(p < vec3i(dims())); }
fn index(p: vec3u) -> u32 { return p.x + params.dimsMax.x * (p.y + params.dimsMax.y * p.z); }
fn airCellKey(p: vec3u) -> u32 { return index(p) + 1u; }
fn faceOrigin(face: FaceRecord) -> vec3u { return vec3u(face.originX, face.originY, face.originZ); }
fn leafOrigin(leaf: SurfaceLeaf) -> vec3u { return vec3u(leaf.originX, leaf.originY, leaf.originZ); }
fn cellCoord(c: u32) -> vec3u { let nx=params.dimsMax.x;let ny=params.dimsMax.y;return vec3u(c%nx,(c/nx)%ny,c/(nx*ny)); }
fn decodeOwner(word: u32, cell: vec3u) -> Owner {
  if ((word & 0x80000000u) != 0u) { return Owner(cell, 1u); }
  let exponent = word & 7u;
  if (exponent == 0u || exponent > 5u) { return Owner(cell, 1u); }
  let size = 1u << exponent;
  // The queried cell and dyadic size fully determine the aligned owner.
  // Ignoring the legacy embedded coordinates removes their 1023-cell cap
  // while remaining compatible with every already-published owner word.
  return Owner((cell / vec3u(size)) * vec3u(size), size);
}
fn canonicalOwner(cell: vec3u) -> Owner { var size=min(params.dimsMax.w,8u);var origin=(cell/vec3u(size))*vec3u(size);loop{if(all(origin+vec3u(size)<=dims())||size==1u){return Owner(origin,size);}size>>=1u;origin=(cell/vec3u(size))*vec3u(size);} }
fn ownerPageEncoded(logical:u32)->u32{
  let freeListOffset=owners[5];if(freeListOffset<=16u||((freeListOffset-16u)&1u)!=0u){return 0u;}
  let hashCapacity=(freeListOffset-16u)/2u;let key=logical+1u;var slot=(logical*0x9e3779b1u)%hashCapacity;
  for(var probe=0u;probe<hashCapacity;probe+=1u){let observed=owners[16u+slot];if(observed==key){return owners[16u+hashCapacity+slot];}if(observed==0u){break;}slot=select(slot+1u,0u,slot+1u==hashCapacity);}return 0u;
}
fn ownerAt(p: vec3i) -> Owner {
  let cell=vec3u(p);if(arrayLength(&owners)<=15u||owners[15]!=0x4f574e52u){return decodeOwner(owners[index(cell)],cell);}
  let bd=(dims()+vec3u(7u))/8u;let b=cell/8u;let logical=b.x+b.y*bd.x+b.z*bd.x*bd.y;let encoded=ownerPageEncoded(logical);let capacity=owners[3];
  if(encoded==0u||encoded==0xffffffffu||encoded>capacity){return canonicalOwner(cell);}let local=cell%vec3u(8u);let word=owners[owners[6]+(encoded-1u)*512u+local.x+local.y*8u+local.z*64u];
  if(word==0u){return canonicalOwner(cell);}return decodeOwner(word,cell);
}
fn ownerAtIndex(cell: u32) -> Owner { return ownerAt(vec3i(cellCoord(cell))); }
fn pagedPhiAvailable()->bool{let r=surfaceParams.shape.z;return (r==2u||r==4u)&&arrayLength(&surfaceArena)>6u&&surfaceArena[3]==0u&&surfaceArena[6]>0u;}
fn surfaceHash(q:vec3u)->u32{var h=(q.x*73856093u)^(q.y*19349663u)^(q.z*83492791u);h^=h>>16u;return h;}
fn airAliasRow(q:vec3u)->u32{let mask=surfaceParams.spare0.y-1u;var slot=surfaceHash(q)&mask;let key=airCellKey(q);for(var probe=0u;probe<32u;probe+=1u){let at=surfaceParams.spare0.x+2u*slot;let stored=surfaceArena[at];if(stored==0u){break;}if(stored==key){let encoded=surfaceArena[at+1u];if(encoded!=0u){let row=0xffffffffu-encoded;if(row<surfaceParams.shape.x&&row<arrayLength(&surfaceLeaves)){return row;}}}slot=(slot+1u)&mask;}return INVALID;}
fn surfaceLoad(base:u32,q:vec3u)->f32{let r=surfaceParams.shape.z;return bitcast<f32>(surfaceArena[base+q.x+r*(q.y+r*q.z)]);}
fn surfaceContains(leaf:SurfaceLeaf,point:vec3f)->bool{let origin=vec3f(leafOrigin(leaf));return all(point>=origin)&&all(point<origin+vec3f(f32(leaf.size)));}
fn surfaceFallback(leaf:SurfaceLeaf,point:vec3f)->f32{let origin=vec3f(leafOrigin(leaf));let centre=origin+vec3f(0.5*f32(leaf.size));return leaf.phiGradient.x+dot(leaf.phiGradient.yzw,point-centre);}
fn surfacePagePhi(row:u32,point:vec3f)->f32{let leaf=surfaceLeaves[row];let origin=vec3f(leafOrigin(leaf));let slot=surfaceArena[surfaceParams.offsets0.x+row];if(slot==INVALID||slot>=surfaceParams.shape.y||!surfaceContains(leaf,point)){return surfaceFallback(leaf,point);}let r=surfaceParams.shape.z;let grid=clamp((point-origin)/f32(leaf.size)*f32(r)-vec3f(0.5),vec3f(0),vec3f(f32(r-1u)));let a=vec3u(floor(grid));let b=min(a+vec3u(1),vec3u(r-1u));let t=fract(grid);let base=surfaceParams.offsets1.z+slot*surfaceParams.shape.w;return mix(mix(mix(surfaceLoad(base,a),surfaceLoad(base,vec3u(b.x,a.y,a.z)),t.x),mix(surfaceLoad(base,vec3u(a.x,b.y,a.z)),surfaceLoad(base,vec3u(b.x,b.y,a.z)),t.x),t.y),mix(mix(surfaceLoad(base,vec3u(a.x,a.y,b.z)),surfaceLoad(base,vec3u(b.x,a.y,b.z)),t.x),mix(surfaceLoad(base,vec3u(a.x,b.y,b.z)),surfaceLoad(base,b),t.x),t.y),t.z);}
fn phi(p: vec3i) -> f32 { if (!valid(p)) { return 3.402823e38; }if(!pagedPhiAvailable()){return textureLoad(levelSetIn,p,0).x;}let owner=ownerAt(p);let row=frontierRow(owner);if(row!=INVALID&&row<arrayLength(&surfaceLeaves)){return surfacePagePhi(row,vec3f(p)+vec3f(0.5));}let airRow=airAliasRow(vec3u(p));if(airRow!=INVALID){return surfacePagePhi(airRow,vec3f(p)+vec3f(0.5));}return max(params.cellRelax.x,max(params.cellRelax.y,params.cellRelax.z))*max(1.0,params.solve.w); }
fn ownerPhi(owner: Owner) -> f32 {
  let centre = vec3f(owner.origin) + vec3f(0.5 * f32(owner.size - 1u));
  let a = vec3u(floor(centre)); let b = min(a + vec3u(1u), dims() - vec3u(1u)); let t = fract(centre);
  let p000=phi(vec3i(a));let p100=phi(vec3i(vec3u(b.x,a.y,a.z)));let p010=phi(vec3i(vec3u(a.x,b.y,a.z)));let p110=phi(vec3i(vec3u(b.x,b.y,a.z)));
  let p001=phi(vec3i(vec3u(a.x,a.y,b.z)));let p101=phi(vec3i(vec3u(b.x,a.y,b.z)));let p011=phi(vec3i(vec3u(a.x,b.y,b.z)));let p111=phi(vec3i(b));
  return mix(mix(mix(p000,p100,t.x),mix(p010,p110,t.x),t.y),mix(mix(p001,p101,t.x),mix(p011,p111,t.x),t.y),t.z);
}
fn component(v: vec3f, axis: u32) -> f32 { return select(select(v.z,v.y,axis==1u),v.x,axis==0u); }
fn axisVector(axis: u32) -> vec3i { return select(select(vec3i(0,0,1),vec3i(0,1,0),axis==1u),vec3i(1,0,0),axis==0u); }
fn frontierMapBase()->u32{return 4u+2u*atomicLoad(&control[3]);}
fn frontierHashCapacity()->u32{return (arrayLength(&frontier)-frontierMapBase())/2u;}
fn frontierHash(cell:u32)->u32{var h=cell*747796405u+2891336453u;h^=h>>16u;h*=2246822519u;h^=h>>13u;return h;}
fn frontierRow(owner:Owner)->u32{let cell=index(owner.origin);let cap=frontierHashCapacity();var slot=frontierHash(cell)&(cap-1u);let key=cell+1u;for(var probe=0u;probe<32u;probe+=1u){let stored=atomicLoad(&frontier[frontierMapBase()+2u*slot]);if(stored==0u){break;}if(stored==key){let word=atomicLoad(&frontier[frontierMapBase()+2u*slot+1u]);return select(INVALID,word-2u,word>=2u);}slot=(slot+1u)&(cap-1u);}return INVALID;}
fn compactRowIndex(gid: vec3u) -> u32 { return gid.x + gid.y * compaction[2] * 256u; }

fn candidate(header: LeafHeader, slot: u32) -> Candidate {
  let face=slot/4u;let quadrant=slot%4u;let axis=face/2u;let side=select(-1,1,(face&1u)==1u);let size=header.size;let half=max(1u,size/2u);
  let origin=ownerAtIndex(header.cell).origin;var local=vec3u(0u);local[axis]=select(0u,size-1u,side>0);
  local[(axis+1u)%3u]=select(0u,half,(quadrant&1u)!=0u);local[(axis+2u)%3u]=select(0u,half,(quadrant&2u)!=0u);
  let outside=vec3i(origin+local)+side*axisVector(axis);
  if(!valid(outside)){var faceOrigin=origin;faceOrigin[axis]=select(origin[axis],origin[axis]+size,side>0);return Candidate(quadrant==0u,axis,side,size,faceOrigin,Owner(vec3u(0u),0u),false);}
  let neighbor=ownerAt(outside);let finer=neighbor.size<size;if(!finer&&quadrant!=0u){return Candidate(false,axis,side,0u,origin,neighbor,false);}
  let span=select(size,neighbor.size,finer);var faceOrigin=origin;faceOrigin[axis]=select(origin[axis],origin[axis]+size,side>0);
  faceOrigin[(axis+1u)%3u]+=select(0u,half,(quadrant&1u)!=0u);faceOrigin[(axis+2u)%3u]+=select(0u,half,(quadrant&2u)!=0u);
  // The compact frontier is the authoritative liquid-row set after rebuild.
  // Phi can be represented by different page fallbacks on the two incident
  // sides, so using its sign here can make both leaves publish the same face.
  // Internal faces have one canonical owner: the negative leaf (side > 0).
  let neighborLiquid=frontierRow(neighbor)!=INVALID;
  return Candidate(!neighborLiquid||side>0,axis,side,span,faceOrigin,neighbor,neighborLiquid);
}

fn appendIncidence(row: u32, face: u32) {
  if(row==INVALID||row>=atomicLoad(&control[3])){return;}let local=atomicAdd(&incidence[row],1u);
  atomicMax(&control[4],local+1u);
  if(local<INCIDENCE_PER_ROW){atomicStore(&incidence[atomicLoad(&control[3])+row*INCIDENCE_PER_ROW+local],face);}else{atomicStore(&control[1],1u);atomicAdd(&control[5],1u);}
}

@compute @workgroup_size(256)
fn countFaces(@builtin(global_invocation_id) gid: vec3u) {
  let row=compactRowIndex(gid);let liveRows=min(compaction[0],atomicLoad(&control[3]));if(row>=liveRows){return;}let header=leafHeaders[row];var count=0u;
  for(var slot=0u;slot<24u;slot+=1u){count+=select(0u,1u,candidate(header,slot).valid);}faceOffsets[row]=count;
}

// The face arena is still a small migration allocation, so a single-thread
// scan is preferable to another row-capacity-sized hierarchy. It makes face
// IDs bitwise stable while touching only the live compact rows.
@compute @workgroup_size(1)
fn scanFaceCounts() {
  let liveRows=min(compaction[0],atomicLoad(&control[3]));
  if(compaction[0]>atomicLoad(&control[3])){atomicStore(&control[1],1u);}
  var sum=0u;for(var row=0u;row<liveRows;row+=1u){let count=faceOffsets[row];faceOffsets[row]=sum;sum+=count;}
  faceOffsets[liveRows]=sum;atomicStore(&control[0],sum);if(sum>atomicLoad(&control[2])){atomicStore(&control[1],1u);}
  faceDispatch[0]=(min(sum,atomicLoad(&control[2]))+255u)/256u;faceDispatch[1]=1u;faceDispatch[2]=1u;
}

@compute @workgroup_size(256)
fn publishFaces(@builtin(global_invocation_id) gid: vec3u) {
  let row=compactRowIndex(gid);let liveRows=min(compaction[0],atomicLoad(&control[3]));if(row>=liveRows){return;}let header=leafHeaders[row];var local=0u;
  for(var slot=0u;slot<24u;slot+=1u){let c=candidate(header,slot);if(!c.valid){continue;}let faceIndex=faceOffsets[row]+local;local+=1u;if(faceIndex>=atomicLoad(&control[2])){continue;}
    var velocity=0.0;let areaCell=select(select(params.cellRelax.x*params.cellRelax.y,params.cellRelax.x*params.cellRelax.z,c.axis==1u),params.cellRelax.y*params.cellRelax.z,c.axis==0u);
    for(var b=0u;b<c.span;b+=1u){for(var a=0u;a<c.span;a+=1u){var q=c.origin;q[c.axis]-=1u;q[(c.axis+1u)%3u]+=a;q[(c.axis+2u)%3u]+=b;if(all(q<dims())){velocity+=component(textureLoad(velocityIn,vec3i(q),0).xyz,c.axis);}}}
    velocity/=f32(c.span*c.span);let neighborRow=select(INVALID,frontierRow(c.neighbor),c.neighborLiquid);let negative=select(neighborRow,row,c.side>0);let positive=select(row,neighborRow,c.side>0);
    faces[faceIndex]=FaceRecord(negative,positive,c.origin.x,c.origin.y,c.origin.z,c.axis|(c.span<<2u),velocity,areaCell*f32(c.span*c.span));appendIncidence(negative,faceIndex);appendIncidence(positive,faceIndex);
  }
}

fn faceRhs(row: u32) -> f32 { let count=min(atomicLoad(&incidence[row]),INCIDENCE_PER_ROW);var rhs=0.0;var sorted:array<u32,${OCTREE_GPU_FACE_INCIDENCE_PER_ROW}>;
  for(var local=0u;local<count;local+=1u){sorted[local]=atomicLoad(&incidence[atomicLoad(&control[3])+row*INCIDENCE_PER_ROW+local]);}
  for(var i=1u;i<count;i+=1u){let value=sorted[i];var j=i;while(j>0u&&sorted[j-1u]>value){sorted[j]=sorted[j-1u];j-=1u;}sorted[j]=value;}
  for(var local=0u;local<count;local+=1u){let faceIndex=sorted[local];if(faceIndex>=atomicLoad(&control[0])||faceIndex>=atomicLoad(&control[2])){continue;}let face=faces[faceIndex];let sign=select(-1.0,1.0,face.negativeRow==row);rhs+=sign*face.area*face.normalVelocity;}
  return rhs;
}
@compute @workgroup_size(256)
fn reduceRhsParity(@builtin(global_invocation_id) gid: vec3u) {
  let row=compactRowIndex(gid);if(row>=compaction[0]||atomicLoad(&control[1])!=0u){return;}let rhs=faceRhs(row);let reference=leafHeaders[row].rhs;let difference=abs(rhs-reference);atomicMax(&parity[0],bitcast<u32>(difference));atomicMax(&parity[1],bitcast<u32>(abs(reference)));if(difference>max(1e-5,abs(reference)*1e-5)){atomicAdd(&parity[2],1u);}atomicAdd(&parity[3],1u);
}
@compute @workgroup_size(256)
fn applyFaceRhs(@builtin(global_invocation_id) gid: vec3u) {
  let row=compactRowIndex(gid);if(row>=compaction[0]||atomicLoad(&control[1])!=0u){return;}leafHeaders[row].rhs=faceRhs(row);
}
fn atomicAddFloat(address:ptr<storage,atomic<u32>,read_write>,value:f32){var old=atomicLoad(address);loop{let next=bitcast<u32>(bitcast<f32>(old)+value);let exchanged=atomicCompareExchangeWeak(address,old,next);if(exchanged.exchanged){break;}old=exchanged.old_value;}}
@compute @workgroup_size(256)
fn reduceProjectedDivergence(@builtin(global_invocation_id) gid: vec3u) {
  let row=gid.x;if(row>=compaction[0]||row>=atomicLoad(&control[3])||atomicLoad(&control[1])!=0u){return;}
  let size=f32(leafHeaders[row].size);let volume=size*size*size*params.cellRelax.x*params.cellRelax.y*params.cellRelax.z;
  let divergence=faceRhs(row)/max(volume,1e-30);
  if(divergence<=3.402823e38&&divergence>=-3.402823e38){atomicMax(&parity[8],bitcast<u32>(abs(divergence)));atomicAdd(&parity[9],1u);atomicAddFloat(&parity[11],divergence*divergence);}else{atomicAdd(&parity[10],1u);}
}

fn facePressure(row:u32)->f32{if(row==INVALID||row>=atomicLoad(&control[3])){return 0.0;}return pressure[row];}
fn faceDistance(a:Owner,b:Owner,axis:u32,phiA:f32,phiB:f32)->f32{
  let width=component(params.cellRelax.xyz,axis);let full=0.5*f32(a.size+b.size)*width;if((phiA<0.0)==(phiB<0.0)){return full;}
  let liquidPhi=select(phiB,phiA,phiA<0.0);let airPhi=select(phiA,phiB,phiA<0.0);
  // Match the tall-cell velocity publication bound: the matrix may retain a
  // 0.01 ghost floor, but a pressure kick never uses a fluid scale above 20.
  return clamp(abs(liquidPhi)/max(abs(liquidPhi)+abs(airPhi),1e-12),0.05,1.0)*full;
}
@compute @workgroup_size(256)
fn projectFaces(@builtin(global_invocation_id) gid:vec3u){let faceIndex=gid.x;if(faceIndex>=min(atomicLoad(&control[0]),atomicLoad(&control[2]))||atomicLoad(&control[1])!=0u){return;}
  var face=faces[faceIndex];let axis=face.axisSpan&3u;let origin=faceOrigin(face);let negativePoint=vec3i(origin)-axisVector(axis);let positivePoint=vec3i(origin);
  if(!valid(negativePoint)||!valid(positivePoint)){face.normalVelocity=0.0;faces[faceIndex]=face;return;}
  let negativeOwner=ownerAt(negativePoint);let positiveOwner=ownerAt(positivePoint);let negativePhi=ownerPhi(negativeOwner);let positivePhi=ownerPhi(positiveOwner);
  face.normalVelocity-=(facePressure(face.positiveRow)-facePressure(face.negativeRow))/max(faceDistance(negativeOwner,positiveOwner,axis,negativePhi,positivePhi),1e-7);faces[faceIndex]=face;
}
@compute @workgroup_size(256)
fn reduceProjectionParity(@builtin(global_invocation_id) gid:vec3u){let faceIndex=gid.x;if(faceIndex>=min(atomicLoad(&control[0]),atomicLoad(&control[2]))||atomicLoad(&control[1])!=0u){return;}
  let face=faces[faceIndex];let axis=face.axisSpan&3u;let span=face.axisSpan>>2u;let origin=faceOrigin(face);var reference=0.0;
  for(var b=0u;b<span;b+=1u){for(var a=0u;a<span;a+=1u){var q=origin;q[axis]-=1u;q[(axis+1u)%3u]+=a;q[(axis+2u)%3u]+=b;if(all(q<dims())){reference+=component(textureLoad(projectedVelocity,vec3i(q),0).xyz,axis);}}}reference/=f32(span*span);
  let difference=abs(face.normalVelocity-reference);atomicMax(&parity[4],bitcast<u32>(difference));atomicMax(&parity[5],bitcast<u32>(abs(reference)));if(difference>max(1e-5,abs(reference)*1e-5)){atomicAdd(&parity[6],1u);}atomicAdd(&parity[7],1u);
}
`;
