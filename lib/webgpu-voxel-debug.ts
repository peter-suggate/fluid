/**
 * GPU-only inspection renderer for sparse brick octrees.
 *
 * This module intentionally depends on a structural buffer interface rather
 * than the sparse-octree implementation.  The owner retains all source
 * buffers; this renderer only owns its compacted instance and indirect-draw
 * buffers.  Source counts stay on the GPU throughout the frame.
 *
 * Buffer ABI (all values are little-endian WebGPU host-shareable values):
 *   SparseVoxelDebugRecord, 48 bytes
 *     origin           vec4f  // minimum world corner; w reserved
 *     extent           vec4f  // xyz world extent; w reserved
 *     materialAndFlags vec4u  // material id, flags, level, owner id
 *   SparseVoxelDebugMaterial, 32 bytes
 *     baseColor        vec4f  // linear RGB and opacity
 *     emissiveRoughness vec4f // linear emissive RGB and roughness
 *   count bindings expose one u32 at byte offset zero.
 */

import { unifiedLightingShaderLibrary } from "./webgpu-lighting";

export type VoxelRenderMode = "smooth" | "raw-voxels" | "brick-grid";
/** Sparse inspection is deliberately unavailable to the production hybrid mode. */
export type VoxelDebugMode = Exclude<VoxelRenderMode, "smooth">;

export const SPARSE_VOXEL_DEBUG_RECORD_STRIDE = 48;
export const SPARSE_VOXEL_DEBUG_MATERIAL_STRIDE = 32;
export const SPARSE_VOXEL_DEBUG_ACTIVE = 1;

export interface SparseVoxelRenderSource {
  /** Finest active/occupied voxel records. */
  voxelRecords: GPUBufferBinding;
  /** GPU-resident u32 count for voxelRecords. */
  voxelCount: GPUBufferBinding;
  /** Brick/node bounds used by the grid view. */
  brickRecords: GPUBufferBinding;
  /** GPU-resident u32 count for brickRecords. */
  brickCount: GPUBufferBinding;
  /** SparseVoxelDebugMaterial records; baseColor is linear, never premultiplied. */
  materials: GPUBufferBinding;
  voxelCapacity: number;
  brickCapacity: number;
  materialCount: number;
  /** Optional evolving-fluid residency header (16 u32 words). */
  fluidBrickStats?: GPUBufferBinding;
  fluidBrickCapacity?: number;
  /** Allows the caller to expose buffer replacement without implementation coupling. */
  revision: number;
}
/** Compatibility alias; the source ABI is shared with the production voxel renderer. */
export type SparseVoxelDebugSource = SparseVoxelRenderSource;

export interface VoxelDebugPlan {
  enabled: boolean;
  recordKind: "none" | "voxels" | "bricks";
  capacity: number;
  computeWorkgroups: number;
  verticesPerInstance: number;
  topology: "none" | "triangle-list" | "line-list";
  /** Fluid-resident brick outlines drawn on top of both inspection modes. */
  overlayCapacity: number;
  overlayWorkgroups: number;
}

export function voxelDebugPlan(
  mode: VoxelDebugMode,
  source: Pick<SparseVoxelRenderSource, "voxelCapacity" | "brickCapacity">
): VoxelDebugPlan {
  const raw = mode === "raw-voxels";
  const capacity = Math.max(0, Math.floor(raw ? source.voxelCapacity : source.brickCapacity));
  const overlayCapacity = Math.max(0, Math.floor(source.brickCapacity));
  return {
    enabled: capacity > 0,
    recordKind: raw ? "voxels" : "bricks",
    capacity,
    computeWorkgroups: Math.ceil(capacity / 64),
    verticesPerInstance: raw ? 36 : 24,
    topology: raw ? "triangle-list" : "line-list",
    overlayCapacity,
    overlayWorkgroups: Math.ceil(overlayCapacity / 64)
  };
}

export interface SparseVoxelDebugRendererOptions {
  colorFormat: GPUTextureFormat;
  depthFormat?: GPUTextureFormat;
  sampleCount?: number;
}

export interface SparseVoxelDebugEncodeOptions {
  mode: VoxelDebugMode;
  colorTarget: GPUTextureView;
  depthTarget: GPUTextureView;
  /** Column-major world-to-clip matrix. */
  viewProjection: Float32Array | readonly number[];
  cameraPosition: readonly [number, number, number];
  /** Interior tank bounds used to collapse glass boundary voxels into panes. */
  containerBounds: {
    min: readonly [number, number, number];
    max: readonly [number, number, number];
  };
  containerClosedTop: boolean;
  /** World-space direction from the surface toward the key light. */
  lightDirection?: readonly [number, number, number];
  exposure?: number;
  gridOpacity?: number;
  colorLoadOp?: GPULoadOp;
  depthLoadOp?: GPULoadOp;
}

export const voxelDebugComputeShader = /* wgsl */ `
struct SpatialRecord {
  origin: vec4f,
  extent: vec4f,
  materialAndFlags: vec4u,
}
struct Count { value: u32 }
struct DrawArguments {
  vertexCount: u32,
  instanceCount: atomic<u32>,
  firstVertex: u32,
  firstInstance: u32,
}
struct CompactSettings { capacity: u32, overlayCapacity: u32, padding1: u32, padding2: u32 }

@group(0) @binding(0) var<storage, read> voxelRecords: array<SpatialRecord>;
@group(0) @binding(1) var<storage, read> voxelCount: Count;
@group(0) @binding(2) var<storage, read> brickRecords: array<SpatialRecord>;
@group(0) @binding(3) var<storage, read> brickCount: Count;
@group(0) @binding(4) var<storage, read_write> instances: array<SpatialRecord>;
@group(0) @binding(5) var<storage, read_write> drawArguments: DrawArguments;
@group(0) @binding(6) var<uniform> compactSettings: CompactSettings;
@group(0) @binding(7) var<storage, read_write> overlayInstances: array<SpatialRecord>;
@group(0) @binding(8) var<storage, read_write> overlayDrawArguments: DrawArguments;

const ACTIVE: u32 = 1u;
// CORE(2) | HALO(4) | ACTIVATED(8): residency bits are only ever published
// into brick-record flags for fluid solver leaves, so they double as the
// fluid-versus-environment discriminator for the outline overlay.
const FLUID_RESIDENCY: u32 = 14u;

@compute @workgroup_size(1)
fn prepareRaw() {
  drawArguments.vertexCount = 36u;
  atomicStore(&drawArguments.instanceCount, 0u);
  drawArguments.firstVertex = 0u;
  drawArguments.firstInstance = 0u;
  overlayDrawArguments.vertexCount = 24u;
  atomicStore(&overlayDrawArguments.instanceCount, 0u);
  overlayDrawArguments.firstVertex = 0u;
  overlayDrawArguments.firstInstance = 0u;
}

@compute @workgroup_size(1)
fn prepareGrid() {
  drawArguments.vertexCount = 24u;
  atomicStore(&drawArguments.instanceCount, 0u);
  drawArguments.firstVertex = 0u;
  drawArguments.firstInstance = 0u;
  overlayDrawArguments.vertexCount = 24u;
  atomicStore(&overlayDrawArguments.instanceCount, 0u);
  overlayDrawArguments.firstVertex = 0u;
  overlayDrawArguments.firstInstance = 0u;
}

@compute @workgroup_size(64)
fn compactVoxels(@builtin(global_invocation_id) gid: vec3u) {
  let index = gid.x;
  let count = min(min(voxelCount.value, arrayLength(&voxelRecords)), compactSettings.capacity);
  if (index >= count) { return; }
  let record = voxelRecords[index];
  if ((record.materialAndFlags.y & ACTIVE) == 0u || any(record.extent.xyz <= vec3f(0.0))) { return; }
  let slot = atomicAdd(&drawArguments.instanceCount, 1u);
  // The CPU dispatch is bounded by instance capacity, and count cannot exceed
  // the source binding length. This guard also makes malformed bindings safe.
  if (slot < arrayLength(&instances)) { instances[slot] = record; }
}

@compute @workgroup_size(64)
fn compactBricks(@builtin(global_invocation_id) gid: vec3u) {
  let index = gid.x;
  let count = min(min(brickCount.value, arrayLength(&brickRecords)), compactSettings.capacity);
  if (index >= count) { return; }
  let record = brickRecords[index];
  if ((record.materialAndFlags.y & ACTIVE) == 0u || any(record.extent.xyz <= vec3f(0.0))) { return; }
  let slot = atomicAdd(&drawArguments.instanceCount, 1u);
  if (slot < arrayLength(&instances)) { instances[slot] = record; }
}

@compute @workgroup_size(64)
fn compactFluidBricks(@builtin(global_invocation_id) gid: vec3u) {
  let index = gid.x;
  let count = min(min(brickCount.value, arrayLength(&brickRecords)), compactSettings.overlayCapacity);
  if (index >= count) { return; }
  let record = brickRecords[index];
  // Environment and glass lattice records never carry residency bits, so the
  // overlay outlines exactly the RESIDENT fluid bricks and nothing else.
  if ((record.materialAndFlags.y & FLUID_RESIDENCY) == 0u || any(record.extent.xyz <= vec3f(0.0))) { return; }
  let slot = atomicAdd(&overlayDrawArguments.instanceCount, 1u);
  if (slot < arrayLength(&overlayInstances)) { overlayInstances[slot] = record; }
}
`;

export const voxelDebugRenderShader = /* wgsl */ `
struct ViewUniforms {
  viewProjection: mat4x4f,
  cameraPosition: vec4f,
  lightDirection: vec4f,
  // x: mode (1 raw, 2 grid), y: grid opacity, z: exposure, w: material count
  style: vec4f,
  tankMin: vec4f,
  tankMax: vec4f,
}
struct SpatialRecord {
  origin: vec4f,
  extent: vec4f,
  materialAndFlags: vec4u,
}
struct Material {
  baseColor: vec4f,
  emissiveRoughness: vec4f,
}
struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) worldPosition: vec3f,
  @location(1) worldNormal: vec3f,
  @location(2) @interpolate(flat) materialId: u32,
  @location(3) @interpolate(flat) level: u32,
  @location(4) @interpolate(flat) flags: u32,
}

@group(0) @binding(0) var<uniform> view: ViewUniforms;
@group(0) @binding(1) var<storage, read> instances: array<SpatialRecord>;
@group(0) @binding(2) var<storage, read> materials: array<Material>;

${unifiedLightingShaderLibrary}

fn triangleCorner(index: u32) -> vec3f {
  let corners = array<vec3f, 36>(
    vec3f(1,0,0),vec3f(1,1,0),vec3f(1,1,1), vec3f(1,0,0),vec3f(1,1,1),vec3f(1,0,1),
    vec3f(0,0,1),vec3f(0,1,1),vec3f(0,1,0), vec3f(0,0,1),vec3f(0,1,0),vec3f(0,0,0),
    vec3f(0,1,0),vec3f(0,1,1),vec3f(1,1,1), vec3f(0,1,0),vec3f(1,1,1),vec3f(1,1,0),
    vec3f(0,0,1),vec3f(0,0,0),vec3f(1,0,0), vec3f(0,0,1),vec3f(1,0,0),vec3f(1,0,1),
    vec3f(0,0,1),vec3f(1,0,1),vec3f(1,1,1), vec3f(0,0,1),vec3f(1,1,1),vec3f(0,1,1),
    vec3f(1,0,0),vec3f(0,0,0),vec3f(0,1,0), vec3f(1,0,0),vec3f(0,1,0),vec3f(1,1,0)
  );
  return corners[index];
}

fn triangleNormal(index: u32) -> vec3f {
  let normals = array<vec3f, 6>(vec3f(1,0,0),vec3f(-1,0,0),vec3f(0,1,0),vec3f(0,-1,0),vec3f(0,0,1),vec3f(0,0,-1));
  return normals[index / 6u];
}

fn lineCorner(index: u32) -> vec3f {
  let corners = array<vec3f, 24>(
    vec3f(0,0,0),vec3f(1,0,0), vec3f(1,0,0),vec3f(1,1,0),
    vec3f(1,1,0),vec3f(0,1,0), vec3f(0,1,0),vec3f(0,0,0),
    vec3f(0,0,1),vec3f(1,0,1), vec3f(1,0,1),vec3f(1,1,1),
    vec3f(1,1,1),vec3f(0,1,1), vec3f(0,1,1),vec3f(0,0,1),
    vec3f(0,0,0),vec3f(0,0,1), vec3f(1,0,0),vec3f(1,0,1),
    vec3f(1,1,0),vec3f(1,1,1), vec3f(0,1,0),vec3f(0,1,1)
  );
  return corners[index];
}

@vertex
fn rawVertex(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
  let record = instances[instanceIndex];
  let corner = triangleCorner(vertexIndex);
  let world = record.origin.xyz + corner * record.extent.xyz;
  var output: VertexOutput;
  output.position = view.viewProjection * vec4f(world, 1.0);
  output.worldPosition = world;
  output.worldNormal = triangleNormal(vertexIndex);
  output.materialId = record.materialAndFlags.x;
  output.level = record.materialAndFlags.z;
  output.flags = record.materialAndFlags.y;
  return output;
}

@vertex
fn gridVertex(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
  let record = instances[instanceIndex];
  let world = record.origin.xyz + lineCorner(vertexIndex) * record.extent.xyz;
  var output: VertexOutput;
  output.position = view.viewProjection * vec4f(world, 1.0);
  output.worldPosition = world;
  output.worldNormal = vec3f(0.0, 1.0, 0.0);
  output.materialId = record.materialAndFlags.x;
  output.level = record.materialAndFlags.z;
  output.flags = record.materialAndFlags.y;
  return output;
}

@vertex
fn overlayVertex(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
  let record = instances[instanceIndex];
  let world = record.origin.xyz + lineCorner(vertexIndex) * record.extent.xyz;
  var output: VertexOutput;
  output.position = view.viewProjection * vec4f(world, 1.0);
  // Brick outlines lie exactly on payload voxel faces. A small camera-ward
  // depth bias keeps front and silhouette edges from losing the depth tie
  // against the cubes while occluded rear edges still fail the depth test.
  output.position.z -= 0.0015 * output.position.w;
  output.worldPosition = world;
  output.worldNormal = vec3f(0.0, 1.0, 0.0);
  output.materialId = record.materialAndFlags.x;
  output.level = record.materialAndFlags.z;
  output.flags = record.materialAndFlags.y;
  return output;
}

fn paneCorner(index: u32) -> vec2f {
  let corners = array<vec2f, 6>(
    vec2f(0,0), vec2f(1,0), vec2f(1,1),
    vec2f(0,0), vec2f(1,1), vec2f(0,1)
  );
  return corners[index];
}

@vertex
fn glassPaneVertex(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) paneIndex: u32) -> VertexOutput {
  let uv = paneCorner(vertexIndex);
  let mn = view.tankMin.xyz;
  let mx = view.tankMax.xyz;
  var world = vec3f(0.0);
  var normal = vec3f(0.0);
  switch paneIndex {
    case 0u: { world = vec3f(mn.x, mix(mn.y, mx.y, uv.y), mix(mn.z, mx.z, uv.x)); normal = vec3f(-1,0,0); }
    case 1u: { world = vec3f(mx.x, mix(mn.y, mx.y, uv.y), mix(mn.z, mx.z, uv.x)); normal = vec3f(1,0,0); }
    case 2u: { world = vec3f(mix(mn.x, mx.x, uv.x), mn.y, mix(mn.z, mx.z, uv.y)); normal = vec3f(0,-1,0); }
    case 3u: { world = vec3f(mix(mn.x, mx.x, uv.x), mix(mn.y, mx.y, uv.y), mn.z); normal = vec3f(0,0,-1); }
    case 4u: { world = vec3f(mix(mn.x, mx.x, uv.x), mix(mn.y, mx.y, uv.y), mx.z); normal = vec3f(0,0,1); }
    default: { world = vec3f(mix(mn.x, mx.x, uv.x), mx.y, mix(mn.z, mx.z, uv.y)); normal = vec3f(0,1,0); }
  }
  var output: VertexOutput;
  output.position = view.viewProjection * vec4f(world, 1.0);
  output.worldPosition = world;
  output.worldNormal = normal;
  output.materialId = 1u;
  output.level = 0u;
  output.flags = 0u;
  return output;
}

fn shadeMaterial(material: Material, normal: vec3f, worldPosition: vec3f) -> vec4f {
  let l = normalize(view.lightDirection.xyz);
  let v = normalize(view.cameraPosition.xyz - worldPosition);
  let roughness = clamp(material.emissiveRoughness.w, 0.04, 1.0);
  let closure = unifiedMaterial(material.baseColor.rgb, roughness, material.emissiveRoughness.rgb, 0.24, vec3f(0.04), (1.0 - roughness) * 0.22, vec3f(0.0), 0.0);
  let lighting = unifiedLightingInput(normal, v, l, vec3f(1.0));
  let linearColor = shadeUnifiedSurface(closure, lighting);
  return vec4f(linearColor * view.style.z, material.baseColor.a);
}

@fragment
fn rawFragment(input: VertexOutput) -> @location(0) vec4f {
  let materialCount = max(1u, u32(view.style.w));
  let material = materials[min(input.materialId, materialCount - 1u)];
  // Inspection-hidden room shells still occupy the sparse hierarchy. Discard
  // instead of merely returning alpha zero so their fragments do not write
  // depth and conceal desks, chairs, fixtures, or other interior props.
  if (material.baseColor.a <= 0.001) { discard; }
  // Tank glass is rendered in a separate, camera-sorted pass. Leaving it in
  // the parallel compacted draw makes alpha/depth results depend on atomic
  // instance order, which changes between frames and looks like z-fighting.
  if (input.materialId == 1u) { discard; }
  return shadeMaterial(material, normalize(input.worldNormal), input.worldPosition);
}

@fragment
fn glassPaneFragment(input: VertexOutput) -> @location(0) vec4f {
  let materialCount = max(1u, u32(view.style.w));
  let material = materials[min(1u, materialCount - 1u)];
  let towardCamera = normalize(view.cameraPosition.xyz - input.worldPosition);
  let normal = select(-input.worldNormal, input.worldNormal, dot(input.worldNormal, towardCamera) >= 0.0);
  return shadeMaterial(material, normal, input.worldPosition);
}

// Fluid residency palette shared by both inspection modes. ACTIVATED flashes
// green for the single classification frame the flag survives, CORE bricks
// straddling the interface keep the established bright blue, and supporting
// HALO bricks read saturated violet. Alpha is a per-state emphasis weight.
fn fluidResidencyColor(flags: u32) -> vec4f {
  if ((flags & 8u) != 0u) { return vec4f(0.30, 1.0, 0.42, 1.0); }
  if ((flags & 2u) != 0u) { return vec4f(0.08, 0.55, 1.0, 0.95); }
  return vec4f(0.66, 0.34, 1.0, 0.85);
}

@fragment
fn overlayFragment(input: VertexOutput) -> @location(0) vec4f {
  let residency = fluidResidencyColor(input.flags);
  return vec4f(residency.rgb * view.style.z, residency.a);
}

@fragment
fn gridFragment(input: VertexOutput) -> @location(0) vec4f {
  // Resident fluid bricks (residency bits published by the page table) read
  // brighter and more saturated than the dim alternating environment lattice.
  if ((input.flags & 14u) != 0u) {
    let residency = fluidResidencyColor(input.flags);
    return vec4f(residency.rgb * view.style.z, min(1.0, view.style.y * residency.a * 1.25));
  }
  // Alternating level tint makes parent/child brick boundaries legible even
  // when many bounds project onto the same pixels.
  let even = (input.level & 1u) == 0u;
  let color = select(vec3f(0.12, 0.88, 1.0), vec3f(1.0, 0.58, 0.12), even);
  return vec4f(color * view.style.z, view.style.y);
}
`;

function normalizeDirection(direction: readonly [number, number, number]): [number, number, number] {
  const length = Math.hypot(direction[0], direction[1], direction[2]);
  return length > 1e-8 ? [direction[0] / length, direction[1] / length, direction[2] / length] : [0.35, 0.84, 0.41];
}

export class SparseVoxelDebugRenderer {
  private readonly device: GPUDevice;
  private readonly options: Required<SparseVoxelDebugRendererOptions>;
  private computeLayout?: GPUBindGroupLayout;
  private renderLayout?: GPUBindGroupLayout;
  private prepareRawPipeline?: GPUComputePipeline;
  private prepareGridPipeline?: GPUComputePipeline;
  private compactVoxelPipeline?: GPUComputePipeline;
  private compactBrickPipeline?: GPUComputePipeline;
  private compactFluidBrickPipeline?: GPUComputePipeline;
  private rawPipeline?: GPURenderPipeline;
  private glassPanePipeline?: GPURenderPipeline;
  private gridPipeline?: GPURenderPipeline;
  private overlayPipeline?: GPURenderPipeline;
  private uniformBuffer?: GPUBuffer;
  private compactSettingsBuffer?: GPUBuffer;
  private instanceBuffer?: GPUBuffer;
  private indirectBuffer?: GPUBuffer;
  private overlayInstanceBuffer?: GPUBuffer;
  private overlayIndirectBuffer?: GPUBuffer;
  private computeBindGroup?: GPUBindGroup;
  private renderBindGroup?: GPUBindGroup;
  private overlayRenderBindGroup?: GPUBindGroup;
  private source?: SparseVoxelDebugSource;
  private sourceRevision = -1;
  private instanceCapacity = 0;
  private overlayInstanceCapacity = 0;
  private initialized = false;
  private destroyed = false;

  constructor(device: GPUDevice, options: SparseVoxelDebugRendererOptions) {
    this.device = device;
    this.options = {
      colorFormat: options.colorFormat,
      depthFormat: options.depthFormat ?? "depth24plus",
      sampleCount: options.sampleCount ?? 1
    };
  }

  async initialize(): Promise<void> {
    if (this.initialized) return;
    if (this.destroyed) throw new Error("Cannot initialize a destroyed sparse voxel debug renderer");
    const computeModule = this.device.createShaderModule({ label: "Sparse voxel debug compaction", code: voxelDebugComputeShader });
    const renderModule = this.device.createShaderModule({ label: "Sparse voxel debug rendering", code: voxelDebugRenderShader });
    this.computeLayout = this.device.createBindGroupLayout({ label: "Sparse voxel debug compute bindings", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }
    ] });
    this.renderLayout = this.device.createBindGroupLayout({ label: "Sparse voxel debug render bindings", entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } }
    ] });
    const computeLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.computeLayout] });
    const renderLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.renderLayout] });
    const compute = (label: string, entryPoint: string) => this.device.createComputePipelineAsync({ label, layout: computeLayout, compute: { module: computeModule, entryPoint } });
    const depthStencil: GPUDepthStencilState = { format: this.options.depthFormat, depthWriteEnabled: true, depthCompare: "less-equal" };
    [this.prepareRawPipeline, this.prepareGridPipeline, this.compactVoxelPipeline, this.compactBrickPipeline, this.compactFluidBrickPipeline] = await Promise.all([
      compute("Prepare raw voxel draw", "prepareRaw"),
      compute("Prepare sparse brick grid draw", "prepareGrid"),
      compute("Compact visible sparse voxels", "compactVoxels"),
      compute("Compact visible sparse bricks", "compactBricks"),
      compute("Compact resident fluid brick outlines", "compactFluidBricks")
    ]);
    [this.rawPipeline, this.glassPanePipeline, this.gridPipeline, this.overlayPipeline] = await Promise.all([
      this.device.createRenderPipelineAsync({
        label: "Raw sparse voxel cubes", layout: renderLayout,
        vertex: { module: renderModule, entryPoint: "rawVertex" },
        fragment: { module: renderModule, entryPoint: "rawFragment", targets: [{
          format: this.options.colorFormat,
          blend: { color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" }, alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" } }
        }] },
        primitive: { topology: "triangle-list", cullMode: "back" }, depthStencil,
        multisample: { count: this.options.sampleCount }
      }),
      this.device.createRenderPipelineAsync({
        label: "Camera-sorted tank glass panes", layout: renderLayout,
        vertex: { module: renderModule, entryPoint: "glassPaneVertex" },
        fragment: { module: renderModule, entryPoint: "glassPaneFragment", targets: [{
          format: this.options.colorFormat,
          blend: { color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" }, alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" } }
        }] },
        primitive: { topology: "triangle-list", cullMode: "none" },
        depthStencil: { ...depthStencil, depthWriteEnabled: false },
        multisample: { count: this.options.sampleCount }
      }),
      this.device.createRenderPipelineAsync({
        label: "Sparse brick wire grid", layout: renderLayout,
        vertex: { module: renderModule, entryPoint: "gridVertex" },
        fragment: { module: renderModule, entryPoint: "gridFragment", targets: [{
          format: this.options.colorFormat,
          blend: { color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" }, alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" } }
        }] },
        primitive: { topology: "line-list" }, depthStencil: { ...depthStencil, depthWriteEnabled: false },
        multisample: { count: this.options.sampleCount }
      }),
      this.device.createRenderPipelineAsync({
        label: "Resident fluid brick outlines", layout: renderLayout,
        vertex: { module: renderModule, entryPoint: "overlayVertex" },
        fragment: { module: renderModule, entryPoint: "overlayFragment", targets: [{
          format: this.options.colorFormat,
          blend: { color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" }, alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" } }
        }] },
        primitive: { topology: "line-list" }, depthStencil: { ...depthStencil, depthWriteEnabled: false },
        multisample: { count: this.options.sampleCount }
      })
    ]);
    this.uniformBuffer = this.device.createBuffer({ label: "Sparse voxel debug view", size: 144, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.compactSettingsBuffer = this.device.createBuffer({ label: "Sparse voxel debug compaction settings", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.indirectBuffer = this.device.createBuffer({ label: "Sparse voxel debug indirect draw", size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT });
    this.overlayIndirectBuffer = this.device.createBuffer({ label: "Sparse voxel debug overlay indirect draw", size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT });
    this.initialized = true;
    this.rebuildBindings();
  }

  setSource(source?: SparseVoxelDebugSource): void {
    if (this.destroyed) return;
    if (!source) {
      this.source = undefined;
      this.sourceRevision = -1;
      this.computeBindGroup = undefined;
      this.renderBindGroup = undefined;
      this.overlayRenderBindGroup = undefined;
      return;
    }
    if (!Number.isInteger(source.voxelCapacity) || source.voxelCapacity < 0 || !Number.isInteger(source.brickCapacity) || source.brickCapacity < 0) {
      throw new Error("Sparse voxel debug capacities must be non-negative integers");
    }
    if (!Number.isInteger(source.materialCount) || source.materialCount < 1) throw new Error("Sparse voxel debug source requires at least one material");
    this.source = source;
    this.rebuildBindings();
  }

  private rebuildBindings(): void {
    const source = this.source;
    if (!this.initialized || !source || !this.computeLayout || !this.renderLayout || !this.uniformBuffer || !this.compactSettingsBuffer || !this.indirectBuffer || !this.overlayIndirectBuffer) return;
    const requiredCapacity = Math.max(1, source.voxelCapacity, source.brickCapacity);
    if (!this.instanceBuffer || requiredCapacity > this.instanceCapacity) {
      this.instanceBuffer?.destroy();
      this.instanceBuffer = this.device.createBuffer({
        label: `Sparse voxel debug instances (${requiredCapacity})`,
        size: requiredCapacity * SPARSE_VOXEL_DEBUG_RECORD_STRIDE,
        usage: GPUBufferUsage.STORAGE
      });
      this.instanceCapacity = requiredCapacity;
    }
    const requiredOverlayCapacity = Math.max(1, source.brickCapacity);
    if (!this.overlayInstanceBuffer || requiredOverlayCapacity > this.overlayInstanceCapacity) {
      this.overlayInstanceBuffer?.destroy();
      this.overlayInstanceBuffer = this.device.createBuffer({
        label: `Sparse voxel debug overlay instances (${requiredOverlayCapacity})`,
        size: requiredOverlayCapacity * SPARSE_VOXEL_DEBUG_RECORD_STRIDE,
        usage: GPUBufferUsage.STORAGE
      });
      this.overlayInstanceCapacity = requiredOverlayCapacity;
    }
    this.computeBindGroup = this.device.createBindGroup({ layout: this.computeLayout, entries: [
      { binding: 0, resource: source.voxelRecords }, { binding: 1, resource: source.voxelCount },
      { binding: 2, resource: source.brickRecords }, { binding: 3, resource: source.brickCount },
      { binding: 4, resource: { buffer: this.instanceBuffer } }, { binding: 5, resource: { buffer: this.indirectBuffer } },
      { binding: 6, resource: { buffer: this.compactSettingsBuffer } },
      { binding: 7, resource: { buffer: this.overlayInstanceBuffer } }, { binding: 8, resource: { buffer: this.overlayIndirectBuffer } }
    ] });
    this.renderBindGroup = this.device.createBindGroup({ layout: this.renderLayout, entries: [
      { binding: 0, resource: { buffer: this.uniformBuffer } }, { binding: 1, resource: { buffer: this.instanceBuffer } },
      { binding: 2, resource: source.materials }
    ] });
    this.overlayRenderBindGroup = this.device.createBindGroup({ layout: this.renderLayout, entries: [
      { binding: 0, resource: { buffer: this.uniformBuffer } }, { binding: 1, resource: { buffer: this.overlayInstanceBuffer } },
      { binding: 2, resource: source.materials }
    ] });
    this.sourceRevision = source.revision;
  }

  encode(encoder: GPUCommandEncoder, options: SparseVoxelDebugEncodeOptions): boolean {
    if (this.destroyed || !this.initialized || !this.source) return false;
    if (options.viewProjection.length !== 16) throw new Error("Sparse voxel debug viewProjection must contain 16 values");
    if (this.sourceRevision !== this.source.revision) this.rebuildBindings();
    const plan = voxelDebugPlan(options.mode, this.source);
    if (!plan.enabled || !this.computeBindGroup || !this.renderBindGroup || !this.uniformBuffer || !this.indirectBuffer) return false;
    const light = normalizeDirection(options.lightDirection ?? [0.35, 0.84, 0.41]);
    const uniforms = new Float32Array(36);
    uniforms.set(options.viewProjection, 0);
    uniforms.set([...options.cameraPosition, 1], 16);
    uniforms.set([...light, 0], 20);
    uniforms.set([options.mode === "raw-voxels" ? 1 : 2, options.gridOpacity ?? 0.82, options.exposure ?? 1, this.source.materialCount], 24);
    uniforms.set([...options.containerBounds.min, 0], 28);
    uniforms.set([...options.containerBounds.max, options.containerClosedTop ? 1 : 0], 32);
    this.device.queue.writeBuffer(this.uniformBuffer, 0, uniforms);
    this.device.queue.writeBuffer(this.compactSettingsBuffer!, 0, new Uint32Array([plan.capacity, plan.overlayCapacity, 0, 0]));

    const voxelMode = options.mode === "raw-voxels";
    const compute = encoder.beginComputePass({ label: options.mode === "raw-voxels" ? "Prepare raw sparse voxels" : "Prepare sparse brick grid" });
    compute.setBindGroup(0, this.computeBindGroup);
    compute.setPipeline(voxelMode ? this.prepareRawPipeline! : this.prepareGridPipeline!);
    compute.dispatchWorkgroups(1);
    compute.setPipeline(voxelMode ? this.compactVoxelPipeline! : this.compactBrickPipeline!);
    compute.dispatchWorkgroups(plan.computeWorkgroups);
    if (plan.overlayWorkgroups > 0 && this.compactFluidBrickPipeline) {
      compute.setPipeline(this.compactFluidBrickPipeline);
      compute.dispatchWorkgroups(plan.overlayWorkgroups);
    }
    compute.end();

    const pass = encoder.beginRenderPass({
      label: options.mode === "raw-voxels" ? "Raw sparse voxel inspection" : "Sparse brick boundary inspection",
      colorAttachments: [{
        view: options.colorTarget,
        clearValue: { r: 0.008, g: 0.012, b: 0.018, a: 1 },
        loadOp: options.colorLoadOp ?? "load",
        storeOp: "store"
      }],
      depthStencilAttachment: { view: options.depthTarget, depthClearValue: 1, depthLoadOp: options.depthLoadOp ?? "load", depthStoreOp: "store" }
    });
    pass.setPipeline(voxelMode ? this.rawPipeline! : this.gridPipeline!);
    pass.setBindGroup(0, this.renderBindGroup);
    pass.drawIndirect(this.indirectBuffer, 0);
    // Residency-colored fluid brick outlines: the sole brick structure in raw
    // mode, and an emphasis pass over the environment lattice in grid mode.
    // Drawn before the glass panes so the tank tints them naturally.
    if (plan.overlayCapacity > 0 && this.overlayPipeline && this.overlayRenderBindGroup && this.overlayIndirectBuffer) {
      pass.setPipeline(this.overlayPipeline);
      pass.setBindGroup(0, this.overlayRenderBindGroup);
      pass.drawIndirect(this.overlayIndirectBuffer, 0);
      pass.setBindGroup(0, this.renderBindGroup);
    }
    if (voxelMode && this.glassPanePipeline) {
      const [minX, minY, minZ] = options.containerBounds.min;
      const [maxX, maxY, maxZ] = options.containerBounds.max;
      const centers: Array<readonly [number, number, number, number]> = [
        [minX, (minY + maxY) / 2, (minZ + maxZ) / 2, 0],
        [maxX, (minY + maxY) / 2, (minZ + maxZ) / 2, 1],
        [(minX + maxX) / 2, minY, (minZ + maxZ) / 2, 2],
        [(minX + maxX) / 2, (minY + maxY) / 2, minZ, 3],
        [(minX + maxX) / 2, (minY + maxY) / 2, maxZ, 4]
      ];
      if (options.containerClosedTop) centers.push([(minX + maxX) / 2, maxY, (minZ + maxZ) / 2, 5]);
      const [cameraX, cameraY, cameraZ] = options.cameraPosition;
      centers.sort((a, b) => {
        const distanceSquared = (center: readonly [number, number, number, number]) =>
          (center[0] - cameraX) ** 2 + (center[1] - cameraY) ** 2 + (center[2] - cameraZ) ** 2;
        return distanceSquared(b) - distanceSquared(a);
      });
      pass.setPipeline(this.glassPanePipeline);
      for (const center of centers) pass.draw(6, 1, 0, center[3]);
    }
    pass.end();
    return true;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const buffer of [this.uniformBuffer, this.compactSettingsBuffer, this.instanceBuffer, this.indirectBuffer, this.overlayInstanceBuffer, this.overlayIndirectBuffer]) {
      try { buffer?.destroy(); } catch { /* device loss */ }
    }
    this.source = undefined;
    this.computeBindGroup = undefined;
    this.renderBindGroup = undefined;
    this.overlayRenderBindGroup = undefined;
  }
}
