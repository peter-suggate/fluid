/**
 * Structural render ABI published by sparse brick octrees.
 *
 * This module is a *contract* and holds no GPU resources of its own: it names
 * the buffer bindings, publication words and per-field validity descriptors a
 * sparse producer publishes and a renderer consumes, so neither side has to
 * import the other. The owner retains every source buffer, and counts stay on
 * the GPU throughout the frame.
 *
 * It used to also host a standalone inspection renderer — one 48-byte expanded
 * record per resolved leaf voxel plus a second per-brick arena, compacted and
 * drawn as an additive overlay on top of the real SVO frame. That path cost
 * roughly 295 MB of arenas on the widened ocean scene and was reachable only
 * from the render panel's LEVELS/SURFACE/BRICKS/CONTENT buttons; SHADED and RAW
 * now select `surfaceReconstruction` on the production frame instead, so the
 * whole expanded-record lane was removed. What remains here is exactly the
 * structural half the shaded path depends on.
 */

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
  sceneGeometryRevision: 3,
  dynamicSolidRevision: 4,
  coarseFluidRevision: 5,
  fineFluidRevision: 6,
} as const);

/** Bit values stored in SPARSE_VOXEL_PUBLICATION_STATE.validFields. */
export const SPARSE_VOXEL_VALID_FIELDS = Object.freeze({
  topology: 1 << 0,
  sceneGeometry: 1 << 1,
  dynamicSolid: 1 << 2,
  coarseFluid: 1 << 3,
  fineFluid: 1 << 4,
  velocity: 1 << 5,
  materialOwner: 1 << 6,
} as const);

export interface SparseVoxelStructuralFieldValidity {
  /** Global availability bit tested against publication.validFields. */
  bit: number;
  /**
   * Units of the field's distance values, negative inside the material.
   *
   * `negative-inside-cell-bands` is what a narrowed scene geometry lane reports
   * (see `SparseBrickSceneGeometryFormat`): the value is the distance divided by
   * `SPARSE_BRICK_SCENE_DISTANCE_BAND_RADII * cellRadius` for the voxel's *own*
   * leaf and clamped to +/-1, so it is comparable only within one leaf and only
   * up to a positive scale. That is enough for a gradient and not enough for a
   * ray march; a consumer that needs metres must say so and get `f32x2`.
   */
  signedDistance?: "negative-inside-metres" | "negative-inside-cell-bands";
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
  /**
   * Single physical structural arena. Consumers should bind this once and use
   * `structureOffsetsWords` instead of rebinding its semantic slices.
   */
  structure: GPUBufferBinding;
  /** Word offsets of the structural records within `structure`. */
  structureOffsetsWords: Readonly<{
    control: number;
    publication: number;
    nodes: number;
    leaves: number;
  }>;
  /** Counts, capacities, indirect arguments, and topology publication state. */
  control: GPUBufferBinding;
  /** Eight-u32 node records: Morton key, level, child links, leaf, flags. */
  nodes: GPUBufferBinding;
  /** Four-u32 leaf records: node, voxel offset, Morton key. */
  leaves: GPUBufferBinding;
  /** vec4f: fluid SDF, dynamic-solid SDF/fraction, pressure. */
  geometry: GPUBufferBinding;
  /** vec4f: authored/live scene SDF, coverage, and reserved derived channels. */
  sceneGeometry: GPUBufferBinding;
  /** vec4f: world velocity xyz and reconstructed liquid fraction. */
  velocity: GPUBufferBinding;
  /** u32 packed as owner:u16 | material:u16. */
  materialOwners: GPUBufferBinding;
  /** u32 scene owner/material identity, updated independently from physics. */
  sceneMaterialOwners: GPUBufferBinding;
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
      sceneGeometry: SparseVoxelPublicationWord;
      dynamicSolid: SparseVoxelPublicationWord;
      coarseFluid: SparseVoxelPublicationWord;
      /** Zero until a fine sparse fluid field is attached to this ABI. */
      fineFluid: SparseVoxelPublicationWord;
    }>;
  }>;
  fields: Readonly<{
    topology: SparseVoxelStructuralFieldValidity;
    sceneGeometry: SparseVoxelStructuralFieldValidity;
    dynamicSolid: SparseVoxelStructuralFieldValidity;
    coarseFluid: SparseVoxelStructuralFieldValidity;
    fineFluid: SparseVoxelStructuralFieldValidity;
    velocity: SparseVoxelStructuralFieldValidity;
    materialOwner: SparseVoxelStructuralFieldValidity;
  }>;
}

/** Production sparse-scene ABI. Every consumer of a sparse producer sees this. */
export interface SparseVoxelSceneRenderSource {
  /** Dense material-table slot count addressed by the stable sparse material ID. */
  materialCount: number;
  /**
   * Optional production PBR table. Records are dense and direct-indexed by the
   * stable sparse material ID.
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
  /** Optional generation-matched 4^3 acceleration view derived from the live structural topology. */
  wideFanout?: import("./webgpu-svo-wide-fanout").WebGPUSvoWideFanoutSource;
  /** Optional 16-byte aligned traversal nodes derived from the canonical 32-byte records. */
  compactHierarchy?: import("./webgpu-svo-compact-hierarchy").WebGpuSvoCompactHierarchySource;
  /** Optional page-valid sparse opacity cache derived incrementally from the unified live tree. */
  nodeMipPyramid?: import("./webgpu-svo-node-mip-pyramid").WebGpuSvoNodeMipVisibleGeneration;
  /** Optional directional exitant-radiance generation sharing the opacity page plan and slots. */
  tetrahedralRadiance?: import("./webgpu-svo-tetrahedral-radiance").WebGpuSvoTetrahedralRadianceVisibleGeneration;
  /**
   * Capability of the derived opacity/radiance hierarchy. This is independent
   * of authored light-table support: consumers can visibly select exact SVO
   * visibility when cone data cannot be represented without dropping pages.
   */
  derivedLighting?: Readonly<{
    state: "ready" | "unavailable";
    reason?: "capacity" | "address-plan-invalidated" | "unsupported-level-count" | "initialization-failed";
    detail?: string;
    requiredPages: number;
    capacity: number;
  }>;
  /** Renderer-derived allocation telemetry; absent capabilities report zero bytes. */
  derivedRenderAllocationBytes?: Readonly<{
    wideFanout: number;
    compactHierarchy?: number;
    nodeMipPyramid?: number;
    tetrahedralRadiance?: number;
  }>;
  /** Allows the caller to expose buffer replacement without implementation coupling. */
  revision: number;
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
