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
import {
  FLUID_BRICK_ACTIVATED,
  FLUID_BRICK_CORE,
  FLUID_BRICK_HALO,
  FLUID_BRICK_RESIDENT,
  FLUID_BRICK_STATE_STRIDE_BYTES,
  FLUID_BRICK_WAS_RESIDENT,
  FLUID_BRICK_WORKLIST_ENTRY_STRIDE_BYTES,
  FLUID_BRICK_WORKLIST_HEADER_WORDS,
  FLUID_BRICK_WORKLIST_WORDS,
} from "./webgpu-fluid-brick-residency";

export type VoxelRenderMode = "smooth" | "raw-voxels" | "brick-grid";
/** Sparse inspection is deliberately unavailable to the production hybrid mode. */
export type VoxelDebugMode = Exclude<VoxelRenderMode, "smooth">;

export const SPARSE_VOXEL_DEBUG_RECORD_STRIDE = 48;
export const SPARSE_VOXEL_DEBUG_MATERIAL_STRIDE = 32;
export const SPARSE_VOXEL_DEBUG_ACTIVE = 1;
export const SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS = Object.freeze({
  resident: FLUID_BRICK_RESIDENT,
  core: FLUID_BRICK_CORE,
  halo: FLUID_BRICK_HALO,
  activated: FLUID_BRICK_ACTIVATED,
  wasResident: FLUID_BRICK_WAS_RESIDENT,
} as const);
export const SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS = FLUID_BRICK_WORKLIST_WORDS;

/**
 * GPU publication-state words. A producer writes `completeGeneration` only in
 * a final pass after every buffer referenced by the structural source is ready.
 * Consumers may cache that word and skip work when it has not changed.
 */
export const SPARSE_VOXEL_PUBLICATION_STATE = Object.freeze({
  strideBytes: 32,
  completeGeneration: 0,
  validFields: 1,
  topologyRevision: 2,
  staticGeometryRevision: 3,
  dynamicSolidRevision: 4,
  coarseFluidRevision: 5,
  fineFluidRevision: 6,
} as const);

/** Bit values stored in SPARSE_VOXEL_PUBLICATION_STATE.validFields. */
export const SPARSE_VOXEL_VALID_FIELDS = Object.freeze({
  topology: 1 << 0,
  staticGeometry: 1 << 1,
  dynamicSolid: 1 << 2,
  coarseFluid: 1 << 3,
  fineFluid: 1 << 4,
  velocity: 1 << 5,
  materialOwner: 1 << 6,
} as const);

export interface SparseVoxelStructuralFieldValidity {
  /** Global availability bit tested against publication.validFields. */
  bit: number;
  /** Distance values are metres with negative values inside the material. */
  signedDistance?: "negative-inside-metres";
  /** Whether distance magnitude is currently a Euclidean metric distance. */
  distanceQuality?: "metric" | "metric-near-interface" | "occupancy-estimate" | "mixed-exact-approximate";
  /** Additional per-leaf condition required before reading this field. */
  residency?: "all-published-leaves" | "fluid-resident-leaves" | "unavailable";
}

export interface SparseVoxelPublicationWord {
  /** Bind the complete state block; `word` is the array<u32> index to read. */
  binding: GPUBufferBinding;
  word: number;
}

export interface SparseVoxelResidencyListView {
  /** GPU count word; consumers clamp it to capacity before indexing. */
  count: SparseVoxelPublicationWord;
  entryOffsetBytes: number;
  entryStrideBytes: number;
  capacity: number;
}

export interface SparseVoxelFilteredResidencyListView extends SparseVoxelResidencyListView {
  /** Candidate list is shared; select entries by this authoritative state bit. */
  requiredStateBit: number;
}

export interface SparseVoxelFluidResidencySource {
  /** One u32 per solver brick: low 8 flag bits, dry-frame count in bits 16..31. */
  states: GPUBufferBinding;
  /** Header, active `(brick, leaf)` pairs, then retired `(brick, leaf)` pairs. */
  worklist: GPUBufferBinding;
  /** Solver-brick lattice embedded in the structural render-brick domain. */
  domain: Readonly<{
    originBricks: readonly [number, number, number];
    dimensionsBricks: readonly [number, number, number];
  }>;
  stateStrideBytes: number;
  stateBits: Readonly<{
    resident: number;
    core: number;
    halo: number;
    activated: number;
    wasResident: number;
  }>;
  active: SparseVoxelResidencyListView;
  /** Core entries are a state-bit-filtered view of `active`. */
  core: SparseVoxelFilteredResidencyListView;
  /** Halo entries are a state-bit-filtered view of `active`. */
  halo: SparseVoxelFilteredResidencyListView;
  retired: SparseVoxelResidencyListView;
  counters: Readonly<{ activated: SparseVoxelPublicationWord }>;
  /** GPU worklist generation, incremented after list contents and dispatches. */
  generation: SparseVoxelPublicationWord;
  /** Structural coarse-fluid revision which owns the completed residency view. */
  revision: SparseVoxelPublicationWord;
  owner: "GPUFluidBrickResidency";
}

export interface SparseVoxelFluidResidencyLayout {
  headerBytes: number;
  activeEntryOffsetBytes: number;
  retiredEntryOffsetBytes: number;
  entryStrideBytes: number;
  stateStrideBytes: number;
  worklistByteLength: number;
}

/** Exact byte layout allocated by `GPUFluidBrickResidency`. */
export function sparseVoxelFluidResidencyLayout(capacity: number): SparseVoxelFluidResidencyLayout {
  if (!Number.isSafeInteger(capacity) || capacity < 1) throw new RangeError("Sparse voxel residency capacity must be a positive integer");
  const headerBytes = FLUID_BRICK_WORKLIST_HEADER_WORDS * Uint32Array.BYTES_PER_ELEMENT;
  const activeEntryOffsetBytes = headerBytes;
  const retiredEntryOffsetBytes = activeEntryOffsetBytes + capacity * FLUID_BRICK_WORKLIST_ENTRY_STRIDE_BYTES;
  return {
    headerBytes,
    activeEntryOffsetBytes,
    retiredEntryOffsetBytes,
    entryStrideBytes: FLUID_BRICK_WORKLIST_ENTRY_STRIDE_BYTES,
    stateStrideBytes: FLUID_BRICK_STATE_STRIDE_BYTES,
    worklistByteLength: headerBytes + capacity * FLUID_BRICK_WORKLIST_ENTRY_STRIDE_BYTES * 2,
  };
}

export interface SparseVoxelFluidResidencyState {
  flags: number;
  dryFrames: number;
  resident: boolean;
  core: boolean;
  halo: boolean;
  activated: boolean;
  wasResident: boolean;
}

export function decodeSparseVoxelFluidResidencyState(stateWord: number): SparseVoxelFluidResidencyState {
  if (!Number.isSafeInteger(stateWord) || stateWord < 0 || stateWord > 0xffff_ffff) {
    throw new RangeError("Sparse voxel residency state must be a uint32");
  }
  const flags = stateWord & 0xff;
  const has = (bit: number) => (flags & bit) !== 0;
  return {
    flags,
    dryFrames: stateWord >>> 16,
    resident: has(SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.resident),
    core: has(SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.core),
    halo: has(SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.halo),
    activated: has(SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.activated),
    wasResident: has(SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.wasResident),
  };
}

/** Binding-free state/list address decode shared by renderer consumers. */
export const sparseVoxelFluidResidencyWGSL = /* wgsl */ `
const SVO_RESIDENCY_RESIDENT:u32=1u;const SVO_RESIDENCY_CORE:u32=2u;const SVO_RESIDENCY_HALO:u32=4u;const SVO_RESIDENCY_ACTIVATED:u32=8u;const SVO_RESIDENCY_WAS_RESIDENT:u32=32u;
fn svoResidencyFlags(stateWord:u32)->u32{return stateWord&0xffu;}
fn svoResidencyDryFrames(stateWord:u32)->u32{return stateWord>>16u;}
fn svoResidencyHas(stateWord:u32,requiredBit:u32)->bool{return (svoResidencyFlags(stateWord)&requiredBit)!=0u;}
fn svoResidencyEntryWord(entryOffsetBytes:u32,entryIndex:u32)->u32{return entryOffsetBytes/4u+entryIndex*2u;}
`;

export interface SparseVoxelStructuralRenderSource {
  /** Counts, capacities, indirect arguments, and topology publication state. */
  control: GPUBufferBinding;
  /** Eight-u32 node records: Morton key, level, child links, leaf, flags. */
  nodes: GPUBufferBinding;
  /** Four-u32 leaf records: node, voxel offset, Morton key. */
  leaves: GPUBufferBinding;
  /** vec4f: fluid SDF, solid SDF, solid fraction, pressure. */
  geometry: GPUBufferBinding;
  /** vec4f: world velocity xyz and reconstructed liquid fraction. */
  velocity: GPUBufferBinding;
  /** u32 packed as owner:u16 | material:u16. */
  materialOwners: GPUBufferBinding;
  /** Per-leaf residency flags; required when reading evolving fluid payload. */
  fluidLeafStates: GPUBufferBinding;
  /** Authoritative producer-owned brick residency; never inferred from payload values. */
  fluidResidency?: SparseVoxelFluidResidencySource;
  capacities: Readonly<{ nodes: number; leaves: number; voxels: number }>;
  strides: Readonly<{
    control: number;
    node: number;
    leaf: number;
    geometry: number;
    velocity: number;
    materialOwner: number;
  }>;
  domain: Readonly<{
    worldOrigin_m: readonly [number, number, number];
    cellSize_m: readonly [number, number, number];
    dimensionsCells: readonly [number, number, number];
    brickSize: 4 | 8;
    maximumDepth: number;
  }>;
  publication: Readonly<{
    /** Eight-u32 state block described by SPARSE_VOXEL_PUBLICATION_STATE. */
    state: GPUBufferBinding;
    /** State-block binding plus the u32 word containing the completion fence. */
    completeGeneration: SparseVoxelPublicationWord;
    validFields: SparseVoxelPublicationWord;
    revisions: Readonly<{
      topology: SparseVoxelPublicationWord;
      staticGeometry: SparseVoxelPublicationWord;
      dynamicSolid: SparseVoxelPublicationWord;
      coarseFluid: SparseVoxelPublicationWord;
      /** Zero until a fine sparse fluid field is attached to this ABI. */
      fineFluid: SparseVoxelPublicationWord;
    }>;
  }>;
  fields: Readonly<{
    topology: SparseVoxelStructuralFieldValidity;
    staticGeometry: SparseVoxelStructuralFieldValidity;
    dynamicSolid: SparseVoxelStructuralFieldValidity;
    coarseFluid: SparseVoxelStructuralFieldValidity;
    fineFluid: SparseVoxelStructuralFieldValidity;
    velocity: SparseVoxelStructuralFieldValidity;
    materialOwner: SparseVoxelStructuralFieldValidity;
  }>;
}

/** Production sparse-scene ABI. It deliberately excludes expanded inspection records. */
export interface SparseVoxelSceneRenderSource {
  /** Dense material-table slot count used by production and inspection shading. */
  materialCount: number;
  /**
   * Optional production PBR table. Records are dense and direct-indexed by the
   * stable sparse material ID; legacy inspection continues to use `materials`.
   */
  pbrMaterials?: SparseVoxelPbrMaterialSource;
  /** Optional authored directional and finite-area light table for production shading. */
  lights?: SparseVoxelLightSource;
  /** Optional image-free diffuse/specular environment fallback for production shading. */
  environmentLighting?: SparseVoxelEnvironmentLightingSource;
  /** Optional evolving-fluid residency header (16 u32 words). */
  fluidBrickStats?: GPUBufferBinding;
  fluidBrickCapacity?: number;
  /** Direct production source. Optional keeps non-structural producers valid. */
  structural?: SparseVoxelStructuralRenderSource;
  /** Allows the caller to expose buffer replacement without implementation coupling. */
  revision: number;
}

/** Expanded records used only by raw-voxel and brick-grid inspection. */
export interface SparseVoxelRenderSource extends SparseVoxelSceneRenderSource {
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
  /** Filled tank panes may conceal page-native fluid cells in raw inspection. */
  drawContainerGlass?: boolean;
  /**
   * Producer-owned switch for the expanded voxel/brick inspection records.
   * Absence means a legacy producer whose inspection publication is always on.
   * Structural publication is independent and must never be gated by this.
   */
  inspectionPublication?: SparseVoxelInspectionPublicationController;
}

export interface SparseVoxelInspectionPublicationController {
  /** Whether the producer will materialize expanded voxel and brick records. */
  readonly enabled: boolean;
  /** Increments only when `enabled` changes. */
  readonly revision: number;
  /** Returns true when the requested state changed. */
  setEnabled(enabled: boolean): boolean;
  /**
   * Materialize records requested by a disabled-to-enabled transition. This
   * lets an inspection repaint publish current records without a simulation
   * advance. Legacy callers may ignore it because producer encode stays on.
   */
  encodePending?(encoder: GPUCommandEncoder): boolean;
}

export interface SparseVoxelInspectionPublicationProducerController
  extends SparseVoxelInspectionPublicationController {
  encodePending(encoder: GPUCommandEncoder): boolean;
  /** Called by the owning producer after its regular encode materializes records. */
  markEncoded(): void;
}

/**
 * Create the controller attached by sparse producers. Default-on preserves the
 * publication behavior of callers that do not opt into production elision.
 */
export function createSparseVoxelInspectionPublicationController(
  initiallyEnabled = true,
  encode?: (encoder: GPUCommandEncoder) => void,
): SparseVoxelInspectionPublicationProducerController {
  let enabled = initiallyEnabled;
  let revision = 1;
  let pending = initiallyEnabled;
  return {
    get enabled() { return enabled; },
    get revision() { return revision; },
    setEnabled(nextEnabled: boolean) {
      if (nextEnabled === enabled) return false;
      enabled = nextEnabled;
      if (enabled) pending = true;
      revision += 1;
      return true;
    },
    encodePending(encoder: GPUCommandEncoder) {
      if (!enabled || !pending || !encode) return false;
      encode(encoder);
      pending = false;
      return true;
    },
    markEncoded() { pending = false; },
  };
}

export interface SparseVoxelPbrMaterialSource {
  binding: GPUBufferBinding;
  /** Dense slot count, including reserved and currently unassigned IDs. */
  count: number;
  strideBytes: number;
  /** Content/schema revision shared by every record in this publication. */
  revision: number;
}

export interface SparseVoxelLightSource {
  binding: GPUBufferBinding;
  count: number;
  strideBytes: number;
  /** Content/schema revision shared by every published light record. */
  revision: number;
}

export interface SparseVoxelEnvironmentLightingSource {
  binding: GPUBufferBinding;
  count: number;
  strideBytes: number;
  revision: number;
  /** Versioned content identity for producer/consumer buffer reuse. */
  cacheKey: string;
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

function voxelDebugDispatch(blocks: number): [number, number, number] {
  const x = Math.min(65_535, Math.max(0, blocks));
  return [x, x > 0 ? Math.ceil(blocks / x) : 1, 1];
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
fn compactVoxels(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let blocks = (compactSettings.capacity + 63u) / 64u;
  let dispatchX = min(blocks, 65535u);
  let index = (wid.x + wid.y * dispatchX) * 64u + lid;
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
fn compactBricks(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let blocks = (compactSettings.capacity + 63u) / 64u;
  let dispatchX = min(blocks, 65535u);
  let index = (wid.x + wid.y * dispatchX) * 64u + lid;
  let count = min(min(brickCount.value, arrayLength(&brickRecords)), compactSettings.capacity);
  if (index >= count) { return; }
  let record = brickRecords[index];
  if ((record.materialAndFlags.y & ACTIVE) == 0u || any(record.extent.xyz <= vec3f(0.0))) { return; }
  let slot = atomicAdd(&drawArguments.instanceCount, 1u);
  if (slot < arrayLength(&instances)) { instances[slot] = record; }
}

@compute @workgroup_size(64)
fn compactFluidBricks(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let blocks = (compactSettings.overlayCapacity + 63u) / 64u;
  let dispatchX = min(blocks, 65535u);
  let index = (wid.x + wid.y * dispatchX) * 64u + lid;
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
  // Unit-cube coordinates make every occupied cell legible even when a
  // contiguous region (notably the initial dam column) has a flat silhouette.
  @location(5) localPosition: vec3f,
  @location(6) @interpolate(flat) voxelSeed: u32,
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
  // Pull every cube slightly toward its centre. A solid dam can otherwise
  // hide every internal cell behind one continuous exterior surface.
  let separatedCorner = mix(vec3f(0.035), vec3f(0.965), corner);
  let world = record.origin.xyz + separatedCorner * record.extent.xyz;
  var output: VertexOutput;
  output.position = view.viewProjection * vec4f(world, 1.0);
  output.worldPosition = world;
  output.worldNormal = triangleNormal(vertexIndex);
  output.materialId = record.materialAndFlags.x;
  output.level = record.materialAndFlags.z;
  output.flags = record.materialAndFlags.y;
  output.localPosition = corner;
  output.voxelSeed = instanceIndex;
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
  output.localPosition = vec3f(0.0);
  output.voxelSeed = 0u;
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
  output.localPosition = vec3f(0.0);
  output.voxelSeed = 0u;
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
  output.localPosition = vec3f(uv, 0.0);
  output.voxelSeed = paneIndex;
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
  // Raw occupancy is a diagnostic representation, so use a finite bounded
  // Lambert + ambient preview rather than allowing a malformed/degenerate PBR
  // half-vector to turn thousands of valid occupied cubes into NaN/black.
  // Production SVO shading and the sorted glass pass retain the shared PBR
  // closure; this branch prioritizes stable material identity and silhouette.
  let normal = normalize(input.worldNormal);
  let lambert = 0.28 + 0.72 * max(dot(normal, normalize(view.lightDirection.xyz)), 0.0);
  var linearColor = material.baseColor.rgb * lambert + material.emissiveRoughness.rgb;
  // Only the two coordinates lying in this cube face describe its boundary;
  // the third coordinate is identically 0 or 1 and would mark the whole face
  // as an edge. Normalize by screen-space derivatives to keep seams visible at
  // a stable pixel width across the very different dam-break cell scales.
  var facePosition = input.localPosition.xy;
  if (abs(normal.x) > 0.5) { facePosition = input.localPosition.yz; }
  else if (abs(normal.y) > 0.5) { facePosition = input.localPosition.xz; }
  let pixelWidth = max(fwidth(facePosition), vec2f(1e-4));
  let boundaryDistance = min(facePosition, vec2f(1.0) - facePosition) / pixelWidth;
  let edge = 1.0 - smoothstep(0.65, 1.35, min(boundaryDistance.x, boundaryDistance.y));
  // Page-native water commonly occupies a contiguous block. Screen-door its
  // face interiors with a per-cell seed so discarded pixels can reveal deeper
  // cells without requiring an impossible global alpha sort. Cell edges stay
  // solid, making this an order-independent structural cutaway.
  if (input.materialId == 3u && edge < 0.35) {
    let pixel = vec2u(input.position.xy);
    var hash = input.voxelSeed * 747796405u + pixel.x * 2891336453u + pixel.y * 277803737u;
    hash = (hash ^ (hash >> 16u)) * 2246822519u;
    if ((hash & 7u) >= 3u) { discard; }
    linearColor *= 1.45;
  }
  linearColor *= mix(1.0, 0.24, edge);
  return vec4f(max(linearColor, vec3f(0.035)) * view.style.z, material.baseColor.a);
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
        // The standalone WebGPU clip transform reaches framebuffer-oriented
        // front-face classification with the opposite winding on the affected
        // backend. Inspection is opt-in and capacity-bounded, so render both
        // sides rather than allowing every occupied cube face to be culled.
        primitive: { topology: "triangle-list", cullMode: "none" }, depthStencil,
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
      // These arenas scale with published voxel capacity (hundreds of MB in
      // wide scenes). Inspection is optional, so detaching must release them
      // rather than merely dropping bind groups.
      this.instanceBuffer?.destroy();
      this.instanceBuffer = undefined;
      this.instanceCapacity = 0;
      this.overlayInstanceBuffer?.destroy();
      this.overlayInstanceBuffer = undefined;
      this.overlayInstanceCapacity = 0;
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
    compute.dispatchWorkgroups(...voxelDebugDispatch(plan.computeWorkgroups));
    if (plan.overlayWorkgroups > 0 && this.compactFluidBrickPipeline) {
      compute.setPipeline(this.compactFluidBrickPipeline);
      compute.dispatchWorkgroups(...voxelDebugDispatch(plan.overlayWorkgroups));
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
    if (voxelMode && this.glassPanePipeline && this.source.drawContainerGlass !== false) {
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
