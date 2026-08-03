import type { RigidBodyDescription, SceneDescription } from "./model";
import {
  cachedSvoPublication,
  hashSvoPublication,
  internSvoPublication,
} from "./svo-publication-cache";
import {
  buildSvoEnvironmentLighting,
  SVO_ENVIRONMENT_LIGHTING_RECORD_STRIDE_BYTES,
  type SvoEnvironmentLightingRecord,
} from "./svo-environment-lighting";
import {
  buildSvoSceneLights,
  SVO_LIGHT_MAXIMUM_RECORDS,
  SVO_LIGHT_RECORD_STRIDE_BYTES,
  type SvoLightRecord,
} from "./svo-light-abi";
import {
  buildDefaultSvoMaterialRecords,
  packSvoMaterialTable,
  SVO_MATERIAL_RECORD_STRIDE_BYTES,
  svoMaterialFunctionIdForEnvironmentProxy,
  svoMaterialFromEnvironmentProxyMaterial,
} from "./svo-material-abi";
import {
  SparseBrickOctreeGPU,
  SPARSE_BRICK_GPU_LAYOUT,
  packSparseBrickPlan,
  type SparseBrickCoordinate,
  type SparseBrickPlan,
  type SparseBrickPublicationSource,
  type SparseBrickSize
} from "./sparse-brick-octree";
import { planAdaptiveSparseBrickOctree } from "./adaptive-sparse-brick-plan";
import { planSvoWideFanout } from "./svo-wide-fanout";
import { WebGPUSvoWideFanout } from "./webgpu-svo-wide-fanout";
import { packSvoCompactHierarchy } from "./svo-compact-hierarchy";
import { WebGpuSvoCompactHierarchy } from "./webgpu-svo-compact-hierarchy";
import { SVO_NODE_MIP_LAYOUT, planSvoNodeMipPyramid, type SvoNodeMipCoordinate } from "./svo-node-mip-pyramid";
import {
  WebGpuLiveSvoNodeMipPyramid,
  createWebGpuSvoNodeMipDirectPageTable,
  webGpuSvoNodeMipMaximumPages,
} from "./webgpu-svo-node-mip-pyramid";
import { WebGpuLiveSvoTetrahedralRadiance } from "./webgpu-svo-tetrahedral-radiance";
import {
  WebGpuLiveSvoDerivedBuilder,
  WebGpuLiveSvoDerivedWorklistPlanner,
  liveSvoPlanBasePages,
} from "./webgpu-svo-live-derived-builder";
import { WebGpuSvoBrickOccupancyBuilder } from "./webgpu-svo-brick-occupancy";
import {
  WebGpuSparseBrickTopologyMutator,
  packSparseBrickTopologyMutationWorklist,
  sparseBrickTopologyMutationNodeReserve,
} from "./webgpu-sparse-brick-topology-mutation";
import { planSparseSceneDomain } from "./sparse-scene-domain";
import { VOXEL_MATERIAL_IDS, materialIdForRigidShape, packVoxelDebugMaterialTable } from "./voxel-scene";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives, type EnvironmentProxyPrimitive } from "./voxel-environments";
import {
  SparseSceneProxyVoxelizer,
  SPARSE_SCENE_MAINTENANCE_STATE_WORDS,
  sparseScenePrimitiveBounds,
  sparseScenePrimitiveForProxy,
  type SparseSceneAxisAlignedBounds,
  type SparseScenePrimitive,
  type SparseScenePrimitiveUpdate,
  type SparseScenePublication,
} from "./webgpu-sparse-scene-proxies";
import {
  SPARSE_VOXEL_DEBUG_RECORD_STRIDE,
  SPARSE_VOXEL_DEBUG_HAS_CONTENT,
  SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS,
  SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS,
  SPARSE_VOXEL_PUBLICATION_STATE,
  SPARSE_VOXEL_VALID_FIELDS,
  createSparseVoxelInspectionPublicationController,
  sparseVoxelFluidResidencyLayout,
  type SparseVoxelInspectionPublicationProducerController,
  type SparseVoxelRenderSource,
  type SparseVoxelSceneRenderSource,
  type SparseVoxelStructuralRenderSource,
} from "./webgpu-voxel-debug";
import { GPUFluidBrickResidency, type FluidBrickResidencyStats } from "./webgpu-fluid-brick-residency";
import { PassBroker } from "./webgpu-pass-broker";

export interface OctreeSparseBrickWorldOptions {
  brickSize?: SparseBrickSize;
  /** Additional authored-environment subdivision beyond the legacy 2x brick ceiling. */
  environmentBrickRefinementLevels?: number;
  /** Air-side support retained for pressure-topology rebuilds. */
  haloCells?: number;
  /** Keep the independent deep-liquid topology worklist. */
  bulkResidency?: boolean;
  /** Velocity-swept residency support plus downstream neighbor activation. */
  brickPreActivation?: boolean;
  /**
   * Power-of-two bricks per topology-tile axis. Topology rebuilds operate on
   * tiles of max(brickSize, maximumLeafSize) cells so a pressure leaf can
   * never straddle a partial-rebuild boundary; payload residency remains
   * brick-granular.
   */
  topologyTileBricks?: number;
  /** Retain dry pressure-wall tiles and their grading support for owner pages. */
  includePressureBoundarySupport?: boolean;
  pressureBoundaryTopClosed?: boolean;
  includeWholeDomainPressureSupport?: boolean;
  /** Retain a dry wall tile only when liquid is within reach; see the residency option. */
  fluidGatedBoundarySupport?: boolean;
  /** Lifetime budget of previously absent scene bricks that may be activated in-place. */
  sceneMutationBrickCapacity?: number;
}

export interface OctreeSparseBrickDenseFields {
  levelSet: GPUTexture;
  velocity: GPUTexture;
  solidCells: GPUBuffer;
}

export interface OctreeSparseBrickEncodePlan {
  /** Structural topology/payload/publication always remains live. */
  structuralPublication: true;
  inspectionPublication: boolean;
  inspectionCountCopies: 0 | 2;
  inspectionComputePasses: 0 | 2;
  inspectionDispatches: 0 | 2;
}

/** Deterministic work evidence for production and inspection publication. */
export function planOctreeSparseBrickEncode(
  inspectionPublication = true,
): OctreeSparseBrickEncodePlan {
  return {
    structuralPublication: true,
    inspectionPublication,
    inspectionCountCopies: inspectionPublication ? 2 : 0,
    inspectionComputePasses: inspectionPublication ? 2 : 0,
    inspectionDispatches: inspectionPublication ? 2 : 0,
  };
}

/** Environment terminal leaves are at most 2x the solver brick scale. */
export const ENVIRONMENT_MAXIMUM_COARSENING_POWER = 1;

/** One extra level is the production default; zero reproduces the previous plan. */
export const DEFAULT_ENVIRONMENT_BRICK_REFINEMENT_LEVELS = 1;

export function environmentMaximumCoarseningPower(
  refinementLevels = DEFAULT_ENVIRONMENT_BRICK_REFINEMENT_LEVELS,
): number {
  if (!Number.isInteger(refinementLevels) || refinementLevels < 0
    || refinementLevels > ENVIRONMENT_MAXIMUM_COARSENING_POWER) {
    throw new RangeError(`Environment brick refinement must be an integer from 0 to ${ENVIRONMENT_MAXIMUM_COARSENING_POWER}`);
  }
  return ENVIRONMENT_MAXIMUM_COARSENING_POWER - refinementLevels;
}

/** Derive an optional renderer traversal snapshot without taking structural authority. */
export function planOctreeSvoWideFanout(plan: SparseBrickPlan) {
  return planSvoWideFanout({
    sourceGeneration: 1,
    generation: 1,
    maximumDepth: plan.maximumDepth,
    terminals: plan.leaves.map((leaf) => {
      const node = plan.nodes[leaf.nodeIndex];
      return {
        sourceNodeIndex: node.index,
        sourceLeafIndex: leaf.index,
        level: node.level,
        coordinate: [node.coordinate.x, node.coordinate.y, node.coordinate.z] as const,
      };
    }),
  });
}

export function planOctreeBrickCoordinates(dimensions: readonly [number, number, number], brickSize: SparseBrickSize) {
  if (brickSize !== 4 && brickSize !== 8) throw new RangeError("Octree brick size must be 4 or 8");
  for (const value of dimensions) if (!Number.isInteger(value) || value < 1) throw new RangeError("Octree field dimensions must be positive integers");
  const brickDimensions = dimensions.map((value) => Math.ceil(value / brickSize)) as [number, number, number];
  const coordinates: SparseBrickCoordinate[] = [];
  for (let z = 0; z < brickDimensions[2]; z += 1) for (let y = 0; y < brickDimensions[1]; y += 1) for (let x = 0; x < brickDimensions[0]; x += 1) coordinates.push({ x, y, z });
  return { brickDimensions, coordinates };
}

/** Root depth must cover declared empty bounds as well as allocated terminals. */
export function sparseSceneOctreeMaximumDepth(
  brickDimensions: readonly [number, number, number],
  coordinates: readonly SparseBrickCoordinate[],
): number {
  for (const [axis, value] of brickDimensions.entries()) {
    if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`Sparse scene brick dimension ${axis} must be positive`);
  }
  let maximumBrickCoordinate = Math.max(...brickDimensions.map((value) => value - 1));
  for (const coordinate of coordinates) {
    for (const [axis, value] of [coordinate.x, coordinate.y, coordinate.z].entries()) {
      if (!Number.isSafeInteger(value) || value < 0) throw new RangeError(`Sparse scene brick coordinate ${axis} must be non-negative`);
      maximumBrickCoordinate = Math.max(maximumBrickCoordinate, value);
    }
  }
  return maximumBrickCoordinate === 0 ? 0 : Math.ceil(Math.log2(maximumBrickCoordinate + 1));
}

export const ENVIRONMENT_VOXEL_MATERIAL_BASE = 32;
export const OCTREE_SVO_PBR_MATERIAL_REVISION = 2;
export const OCTREE_SVO_LIGHT_REVISION = 1;
export const OCTREE_SVO_ENVIRONMENT_LIGHTING_REVISION = 1;

/** Fixed live-scene arenas. Capacity changes require an explicit new world, never a hidden reallocating bake. */
export const OCTREE_LIVE_SCENE_PRIMITIVE_CAPACITY = 4_096;
export const OCTREE_LIVE_SCENE_DIRTY_REGION_CAPACITY = OCTREE_LIVE_SCENE_PRIMITIVE_CAPACITY * 2;
export const OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK = 64;
export const OCTREE_LIVE_SCENE_MUTATION_BRICK_CAPACITY = 4_096;
export const OCTREE_LIVE_SCENE_MATERIAL_CAPACITY = ENVIRONMENT_VOXEL_MATERIAL_BASE + OCTREE_LIVE_SCENE_PRIMITIVE_CAPACITY;

interface LiveScenePrimitiveState {
  readonly signature: string;
  readonly bounds: SparseSceneAxisAlignedBounds;
}

interface LiveScenePrimitiveEntry {
  readonly primitive: SparseScenePrimitive;
  readonly materialSignature: string;
}

function brickCoordinateKey({ x, y, z }: SparseBrickCoordinate): string { return `${x},${y},${z}`; }

function liveScenePrimitiveSignature(primitive: SparseScenePrimitive): string {
  return JSON.stringify(primitive);
}

function sparseScenePrimitiveForRigidBody(body: RigidBodyDescription, ownerId: number): SparseScenePrimitive {
  const center = [body.position_m.x, body.position_m.y, body.position_m.z] as const;
  const orientation = [body.orientation.x, body.orientation.y, body.orientation.z, body.orientation.w] as const;
  const materialId = materialIdForRigidShape(body.shape);
  if (body.shape === "sphere") {
    return { kind: "ellipsoid", center, radii: [body.dimensions_m.x, body.dimensions_m.x, body.dimensions_m.x], materialId, ownerId };
  }
  if (body.shape === "box") {
    return { kind: "box", center, halfExtents: [body.dimensions_m.x / 2, body.dimensions_m.y / 2, body.dimensions_m.z / 2], orientation, materialId, ownerId };
  }
  if (body.shape === "capsule") {
    return { kind: "capsule", center, radius: body.dimensions_m.x, halfLength: body.dimensions_m.y / 2, orientation, materialId, ownerId };
  }
  return { kind: "cylinder", center, radius: body.dimensions_m.x, halfHeight: body.dimensions_m.y / 2, orientation, materialId, ownerId };
}

function coalesceDirtyRegions(
  regions: readonly SparseSceneAxisAlignedBounds[],
  capacity = OCTREE_LIVE_SCENE_DIRTY_REGION_CAPACITY,
): SparseSceneAxisAlignedBounds[] {
  if (regions.length <= capacity) return [...regions];
  return [{
    minimum: [
      Math.min(...regions.map(({ minimum }) => minimum[0])),
      Math.min(...regions.map(({ minimum }) => minimum[1])),
      Math.min(...regions.map(({ minimum }) => minimum[2])),
    ],
    maximum: [
      Math.max(...regions.map(({ maximum }) => maximum[0])),
      Math.max(...regions.map(({ maximum }) => maximum[1])),
      Math.max(...regions.map(({ maximum }) => maximum[2])),
    ],
  }];
}

export function liveSceneBrickCoordinatesForRegions(
  regions: readonly SparseSceneAxisAlignedBounds[],
  worldOrigin: readonly [number, number, number],
  cellSize: readonly [number, number, number],
  brickSize: number,
  brickDimensions: readonly [number, number, number],
): SparseBrickCoordinate[] {
  const coordinates = new Map<string, SparseBrickCoordinate>();
  for (const region of regions) {
    const minimum = region.minimum.map((value, axis) => Math.floor(
      (value - worldOrigin[axis]) / (cellSize[axis] * brickSize),
    ));
    const maximum = region.maximum.map((value, axis) => Math.floor(
      (value - worldOrigin[axis]) / (cellSize[axis] * brickSize),
    ));
    for (let z = Math.max(0, minimum[2]); z <= Math.min(brickDimensions[2] - 1, maximum[2]); z += 1) {
      for (let y = Math.max(0, minimum[1]); y <= Math.min(brickDimensions[1] - 1, maximum[1]); y += 1) {
        for (let x = Math.max(0, minimum[0]); x <= Math.min(brickDimensions[0] - 1, maximum[0]); x += 1) {
          coordinates.set(`${x},${y},${z}`, { x, y, z });
        }
      }
    }
  }
  return [...coordinates.values()];
}

/** Missing finest-brick coverage only; existing coarse/fine leaves remain reusable. */
export function liveSceneMissingBrickCoordinates(
  regions: readonly SparseSceneAxisAlignedBounds[],
  worldOrigin: readonly [number, number, number],
  cellSize: readonly [number, number, number],
  brickSize: number,
  brickDimensions: readonly [number, number, number],
  covered: ReadonlySet<string>,
): SparseBrickCoordinate[] {
  return liveSceneBrickCoordinatesForRegions(regions, worldOrigin, cellSize, brickSize, brickDimensions)
    .filter((coordinate) => !covered.has(brickCoordinateKey(coordinate)));
}

export function liveSvoDenseFinestPages(dimensions: readonly [number, number, number]): SvoNodeMipCoordinate[] {
  const pages: SvoNodeMipCoordinate[] = [];
  for (let z = 0; z < dimensions[2]; z += 1) for (let y = 0; y < dimensions[1]; y += 1) for (let x = 0; x < dimensions[0]; x += 1) {
    pages.push([x, y, z]);
  }
  return pages;
}

/** Finest 8-cell node-mip page grid for any supported canonical brick size. */
export function liveSvoBasePageDimensions(
  brickDimensions: readonly [number, number, number],
  brickSize: SparseBrickSize,
): readonly [number, number, number] {
  return brickDimensions.map((bricks) => Math.ceil(
    bricks * brickSize / SVO_NODE_MIP_LAYOUT.interiorSize,
  )) as [number, number, number];
}

/** Compact vec4 emission table consumed by the GPU-only live radiance builder. */
function packLiveSvoMaterialEmission(packedRecords: Uint32Array, count: number): Float32Array<ArrayBuffer> {
  const source = new Float32Array(packedRecords.buffer, packedRecords.byteOffset, packedRecords.byteLength / 4);
  const recordWords = SVO_MATERIAL_RECORD_STRIDE_BYTES / 4;
  const result = new Float32Array(OCTREE_LIVE_SCENE_MATERIAL_CAPACITY * 4);
  for (let material = 0; material < count; material += 1) {
    result.set(source.subarray(material * recordWords + 4, material * recordWords + 7), material * 4);
  }
  return result;
}

export interface OctreeSvoPbrMaterialPublicationData {
  packedRecords: Uint32Array<ArrayBuffer>;
  count: number;
  strideBytes: number;
  revision: number;
  contentRevision: string;
  cacheKey: string;
}

const octreeSvoPbrMaterialCache = new Map<string, OctreeSvoPbrMaterialPublicationData>();

/** Dense default table used by the producer and CPU ABI/lifecycle tests. */
export function buildOctreeSvoPbrMaterialPublication(
  revision = OCTREE_SVO_PBR_MATERIAL_REVISION,
  environmentPrimitives: readonly EnvironmentProxyPrimitive[] = [],
): OctreeSvoPbrMaterialPublicationData {
  if (!Number.isSafeInteger(revision) || revision < 1 || revision > 0xffff_ffff) {
    throw new RangeError("SVO PBR material publication revision must be a positive uint32");
  }
  const contentRevision = hashSvoPublication(new Uint32Array(), JSON.stringify({
    revision,
    environmentPrimitives: environmentPrimitives.map(({ key, ownerIndex, group, tags, material }) => ({ key, ownerIndex, group, tags, material })),
  }));
  const cacheKey = `octree-svo-pbr-material-v1:${contentRevision}`;
  const cached = cachedSvoPublication(octreeSvoPbrMaterialCache, cacheKey);
  if (cached) return cached;
  const records = [
    ...buildDefaultSvoMaterialRecords(revision),
    ...environmentPrimitives.map((primitive) => {
      if (!Number.isSafeInteger(primitive.ownerIndex) || primitive.ownerIndex < 0) {
        throw new RangeError(`Environment material owner index for ${primitive.key} must be a non-negative safe integer`);
      }
      const materialId = ENVIRONMENT_VOXEL_MATERIAL_BASE + primitive.ownerIndex;
      if (materialId > 0xffff) throw new RangeError(`Environment material ID for ${primitive.key} does not fit uint16`);
      return svoMaterialFromEnvironmentProxyMaterial(
        materialId,
        primitive.material,
        revision,
        svoMaterialFunctionIdForEnvironmentProxy(primitive),
      );
    }),
  ];
  const packedRecords = packSvoMaterialTable(records);
  return internSvoPublication(octreeSvoPbrMaterialCache, cacheKey, {
    packedRecords,
    count: packedRecords.byteLength / SVO_MATERIAL_RECORD_STRIDE_BYTES,
    strideBytes: SVO_MATERIAL_RECORD_STRIDE_BYTES,
    revision,
    contentRevision,
    cacheKey,
  });
}

export interface OctreeSvoLightPublicationData {
  records: readonly SvoLightRecord[];
  packedRecords: Uint32Array<ArrayBuffer>;
  count: number;
  strideBytes: number;
  revision: number;
  omittedFixtureKeys: readonly string[];
  contentRevision: string;
  cacheKey: string;
}

const octreeSvoLightCache = new Map<string, OctreeSvoLightPublicationData>();

/** Build the selected scene/environment's deterministic bounded light table. */
export function buildOctreeSvoLightPublication(
  scene: SceneDescription,
  options: { revision?: number; maximumRecords?: number } = {},
): OctreeSvoLightPublicationData {
  const revision = options.revision ?? OCTREE_SVO_LIGHT_REVISION;
  const lights = buildSvoSceneLights(scene, {
    revision,
    maximumRecords: options.maximumRecords ?? SVO_LIGHT_MAXIMUM_RECORDS,
  });
  const cacheKey = `octree-${lights.cacheKey}`;
  return internSvoPublication(octreeSvoLightCache, cacheKey, {
    records: lights.records,
    packedRecords: lights.packedRecords,
    count: lights.records.length,
    strideBytes: SVO_LIGHT_RECORD_STRIDE_BYTES,
    revision: lights.revision,
    omittedFixtureKeys: lights.omittedFixtureKeys,
    contentRevision: lights.contentRevision,
    cacheKey,
  });
}

export interface OctreeSvoEnvironmentLightingPublicationData {
  record: SvoEnvironmentLightingRecord;
  packedRecords: Uint32Array<ArrayBuffer>;
  count: 1;
  strideBytes: number;
  revision: number;
  cacheKey: string;
}

/** Build the selected environment's single image-free lighting record. */
export function buildOctreeSvoEnvironmentLightingPublication(
  scene: Pick<SceneDescription, "environment" | "lighting">,
  revision = OCTREE_SVO_ENVIRONMENT_LIGHTING_REVISION,
): OctreeSvoEnvironmentLightingPublicationData {
  const lighting = buildSvoEnvironmentLighting(scene.environment ?? "default", revision, scene.lighting?.environment);
  return {
    record: lighting.record,
    packedRecords: lighting.packedRecord,
    count: 1,
    strideBytes: SVO_ENVIRONMENT_LIGHTING_RECORD_STRIDE_BYTES,
    revision: lighting.record.revision,
    cacheKey: lighting.cacheKey,
  };
}

function storageBuffer(device: GPUDevice, label: string, size: number, data?: ArrayBufferView<ArrayBuffer>) {
  const buffer = device.createBuffer({ label, size: Math.max(4, size), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  if (data && data.byteLength > 0) device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

/** Cover arbitrarily large debug-publication arenas without exceeding WebGPU's per-axis limit. */
export function tiledDebugDispatch(items: number, workgroupSize: number): [number, number, number] {
  const blocks = Math.ceil(Math.max(0, items) / workgroupSize);
  const x = Math.min(65_535, blocks);
  return [x, x > 0 ? Math.ceil(blocks / x) : 1, 1];
}

const debugPublicationShader = /* wgsl */ `
struct Node { address: vec4u, links: vec4u }
struct Leaf { topology: vec4u }
struct DebugRecord { origin: vec4f, extent: vec4f, materialAndFlags: vec4u }
struct Params { dims: vec4u, origin: vec4f, cell: vec4f, settings: vec4u }
@group(0) @binding(0) var<storage, read> control: array<u32>;
@group(0) @binding(1) var<storage, read> nodes: array<Node>;
@group(0) @binding(2) var<storage, read> leaves: array<Leaf>;
@group(0) @binding(3) var<storage, read> materialOwners: array<u32>;
@group(0) @binding(4) var<storage, read> bodyMaterials: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read_write> voxelRecords: array<DebugRecord>;
@group(0) @binding(7) var<storage, read_write> brickRecords: array<DebugRecord>;
@group(0) @binding(8) var<storage, read> fluidLeafStates: array<u32>;

fn keyBit(low: u32, high: u32, bit: u32) -> u32 {
  if (bit < 32u) { return (low >> bit) & 1u; }
  return (high >> (bit - 32u)) & 1u;
}
fn decodeMorton(low: u32, high: u32, level: u32) -> vec3u {
  var result = vec3u(0u);
  for (var bit = 0u; bit < level; bit += 1u) {
    let scale = 1u << bit;
    result.x += keyBit(low, high, 3u * bit) * scale;
    result.y += keyBit(low, high, 3u * bit + 1u) * scale;
    result.z += keyBit(low, high, 3u * bit + 2u) * scale;
  }
  return result;
}
fn recordForVoxel(index: u32) -> DebugRecord {
  let brickSize = params.settings.x;
  let voxelsPerBrick = brickSize * brickSize * brickSize;
  let leafIndex = index / voxelsPerBrick;
  let localIndex = index - leafIndex * voxelsPerBrick;
  let local = vec3u(localIndex % brickSize, (localIndex / brickSize) % brickSize, localIndex / (brickSize * brickSize));
  let leaf = leaves[leafIndex];
  let node = nodes[leaf.topology.x];
  let brick = decodeMorton(leaf.topology.z, leaf.topology.w, node.address.z);
  let scale = 1u << (params.settings.z - node.address.z);
  let cell = (brick * brickSize + local) * scale;
  let inside = all(cell < params.dims.xyz);
  // Debug records are densely expanded in leaf order, but the authoritative
  // field payload can live at a different arena offset for every leaf.
  let payloadIndex = leaf.topology.y + localIndex;
  var packed = 0u;
  if (payloadIndex < arrayLength(&materialOwners)) {
    packed = materialOwners[payloadIndex];
  }
  var material = packed & 0xffffu;
  let owner = packed >> 16u;
  if (material != 0u && owner != 0xffffu && owner < params.settings.y && owner < arrayLength(&bodyMaterials)) { material = bodyMaterials[owner]; }
  // Residency makes a brick schedulable; it does not turn the halo's air
  // voxels into liquid. Raw inspection therefore stays payload-exact while
  // the brick-grid record below visualizes core and halo allocation.
  let isActive = inside && material != 0u;
  let world = params.origin.xyz + vec3f(cell) * params.cell.xyz;
  return DebugRecord(vec4f(world, 0.0), vec4f(f32(scale) * params.cell.xyz, 0.0), vec4u(material, select(0u, 1u, isActive), node.address.z, owner));
}

@compute @workgroup_size(256)
fn materializeVoxels(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let blocks = (u32(arrayLength(&voxelRecords)) + 255u) / 256u;
  let dispatchX = min(blocks, 65535u);
  let index = (wid.x + wid.y * dispatchX) * 256u + lid;
  if (index >= control[2] || index >= arrayLength(&voxelRecords)) { return; }
  voxelRecords[index] = recordForVoxel(index);
}

@compute @workgroup_size(64)
fn materializeBricks(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let blocks = (u32(arrayLength(&brickRecords)) + 63u) / 64u;
  let dispatchX = min(blocks, 65535u);
  let leafIndex = (wid.x + wid.y * dispatchX) * 64u + lid;
  if (leafIndex >= control[1] || leafIndex >= arrayLength(&brickRecords)) { return; }
  let brickSize = params.settings.x;
  let voxelsPerBrick = brickSize * brickSize * brickSize;
  var isActive = false;
  var material = 0u;
  var owner = 0xffffu;
  for (var local = 0u; local < voxelsPerBrick; local += 1u) {
    let payloadIndex = leaves[leafIndex].topology.y + local;
    var packed = 0u;
    if (payloadIndex < arrayLength(&materialOwners)) {
      packed = materialOwners[payloadIndex];
    }
    let candidateOwner = packed >> 16u;
    var candidate = packed & 0xffffu;
    if (candidate != 0u && candidateOwner != 0xffffu && candidateOwner < params.settings.y && candidateOwner < arrayLength(&bodyMaterials)) { candidate = bodyMaterials[candidateOwner]; }
    if (candidate != 0u) { isActive = true; material = candidate; owner = candidateOwner; }
  }
  // Preserve payload occupancy independently from fluid residency. The latter
  // includes empty halo/support allocation and must not masquerade as content
  // in occupied-brick inspection.
  let hasContent = isActive;
  let residency = fluidLeafStates[leafIndex];
  let fluidResident = (residency & 1u) != 0u;
  if (fluidResident) { isActive = true; material = params.settings.w; owner = 0xffffu; }
  let leaf = leaves[leafIndex];
  let node = nodes[leaf.topology.x];
  let brick = decodeMorton(leaf.topology.z, leaf.topology.w, node.address.z);
  let scale = 1u << (params.settings.z - node.address.z);
  let world = params.origin.xyz + vec3f(brick * brickSize * scale) * params.cell.xyz;
  let extent = f32(brickSize * scale) * params.cell.xyz;
  let contentFlag = select(0u, ${SPARSE_VOXEL_DEBUG_HAS_CONTENT}u, hasContent);
  brickRecords[leafIndex] = DebugRecord(vec4f(world, 0.0), vec4f(extent, 0.0), vec4u(material, select(0u, 1u, isActive) | residency | contentFlag, node.address.z, owner));
}
`;

/** Fields a live scene publication makes valid before fluid payload is attached. */
export const OCTREE_SPARSE_BRICK_SCENE_VALID_FIELDS =
  SPARSE_VOXEL_VALID_FIELDS.topology
  | SPARSE_VOXEL_VALID_FIELDS.sceneGeometry
  | SPARSE_VOXEL_VALID_FIELDS.materialOwner;

function structuralPublicationFinalizeShader(sceneMaintenanceStateOffsetWords = 0) { return /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> state: array<u32>;
@group(0) @binding(1) var<storage, read> sceneMaintenance: array<u32>;
@group(0) @binding(2) var<storage, read> topologyMutation: array<u32>;

// Scene geometry is its own live generation. Fluid coverage is intentionally
// published through a separate renderer channel so evolving physics can never
// overwrite authored scene voxels in-place.
const SCENE_VALID_FIELDS: u32 = ${OCTREE_SPARSE_BRICK_SCENE_VALID_FIELDS}u;
const PHYSICS_VALID_FIELDS: u32 = ${
  SPARSE_VOXEL_VALID_FIELDS.dynamicSolid |
  SPARSE_VOXEL_VALID_FIELDS.coarseFluid |
  SPARSE_VOXEL_VALID_FIELDS.velocity
}u;

@compute @workgroup_size(1)
fn finalizePhysics() {
  state[${SPARSE_VOXEL_PUBLICATION_STATE.dynamicSolidRevision}] += 1u;
  state[${SPARSE_VOXEL_PUBLICATION_STATE.coarseFluidRevision}] += 1u;
  state[${SPARSE_VOXEL_PUBLICATION_STATE.validFields}] |= PHYSICS_VALID_FIELDS;
  state[${SPARSE_VOXEL_PUBLICATION_STATE.completeGeneration}] += 1u;
}

@compute @workgroup_size(1)
fn finalizeScene() {
  let maintenanceOffset = ${sceneMaintenanceStateOffsetWords}u;
  let requested = sceneMaintenance[maintenanceOffset + ${SPARSE_SCENE_MAINTENANCE_STATE_WORDS.requestedRevision}u];
  let completed = sceneMaintenance[maintenanceOffset + ${SPARSE_SCENE_MAINTENANCE_STATE_WORDS.completedRevision}u];
  let overflow = sceneMaintenance[maintenanceOffset + ${SPARSE_SCENE_MAINTENANCE_STATE_WORDS.overflowFlags}u];
  if (requested == 0u || completed != requested || overflow != 0u || topologyMutation[3] != 0u) { return; }
  if (topologyMutation[0] != 0u) { state[${SPARSE_VOXEL_PUBLICATION_STATE.topologyRevision}] += 1u; }
  state[${SPARSE_VOXEL_PUBLICATION_STATE.sceneGeometryRevision}] += 1u;
  // Scene publication adds its independently-owned lane; it must never clear
  // already-current fluid/dynamic fields while physics is paused.
  state[${SPARSE_VOXEL_PUBLICATION_STATE.validFields}] |= SCENE_VALID_FIELDS;
  // Last: topology, payload maintenance, and the stores above become visible
  // to every render consumer as one live scene generation.
  state[${SPARSE_VOXEL_PUBLICATION_STATE.completeGeneration}] += 1u;
}
`; }

/**
 * Transitional GPU bridge: the octree solver remains authoritative while its
 * resident level set, velocity and VOS solid field are published into one
 * sparse-brick ABI for scene inspection and subsequent sparse kernels.
 */
export class OctreeSparseBrickWorld {
  readonly tree: SparseBrickOctreeGPU;
  /** Narrow two-sided band used by surface and topology scheduling. */
  readonly residency: GPUFluidBrickResidency;
  /** Full wet-domain topology residency, independent of the surface band. */
  readonly bulkResidency?: GPUFluidBrickResidency;
  readonly sceneSource: SparseVoxelSceneRenderSource;
  private readonly preActivation: boolean;

  private readonly device: GPUDevice;
  private readonly dimensions: readonly [number, number, number];
  private readonly solverGridOriginCells: readonly [number, number, number];
  private readonly finestLevel: number;
  private readonly cellSize: readonly [number, number, number];
  private readonly containerClosedTop: boolean;
  private readonly source: SparseBrickPublicationSource;
  private readonly sourceBuffers: GPUBuffer[];
  private readonly pbrMaterialBuffer: GPUBuffer;
  private readonly materialEmissionBuffer: GPUBuffer;
  private readonly lightBuffer: GPUBuffer;
  private readonly environmentLightingBuffer: GPUBuffer;
  private readonly inspectionMaterialData: Float32Array<ArrayBuffer>;
  private readonly inspectionBodyMaterials: Uint32Array<ArrayBuffer>;
  private readonly inspectionParameterData: ArrayBuffer;
  private inspection?: {
    source: SparseVoxelRenderSource;
    publication: SparseVoxelInspectionPublicationProducerController;
    buffers: GPUBuffer[];
    voxelRecords: GPUBuffer;
    brickRecords: GPUBuffer;
    voxelCount: GPUBuffer;
    brickCount: GPUBuffer;
    voxelPipeline: GPUComputePipeline;
    brickPipeline: GPUComputePipeline;
    bindGroup: GPUBindGroup;
    allocatedBytes: number;
  };
  private readonly baseAllocatedBytes: number;
  private readonly structuralPublicationState: GPUBuffer;
  private structuralPhysicsPipeline!: GPUComputePipeline;
  private structuralScenePipeline!: GPUComputePipeline;
  private structuralModule!: GPUShaderModule;
  private readonly structuralPipelineLayout: GPUPipelineLayout;
  private readonly structuralFinalizeBindGroup: GPUBindGroup;
  private readonly proxyVoxelizer: SparseSceneProxyVoxelizer;
  private readonly topologyMutator: WebGpuSparseBrickTopologyMutator;
  private readonly topologyMutationWorklist: GPUBuffer;
  private readonly topologyMutationCapacity: number;
  private readonly sceneBrickDimensions: readonly [number, number, number];
  private readonly brickSize: SparseBrickSize;
  private readonly sceneWorldOrigin: readonly [number, number, number];
  private readonly brickOccupancyBuilder: WebGpuSvoBrickOccupancyBuilder;
  private readonly wideFanout?: WebGPUSvoWideFanout;
  private readonly compactHierarchy?: WebGpuSvoCompactHierarchy;
  private readonly nodeMipPyramid?: WebGpuLiveSvoNodeMipPyramid;
  private readonly tetrahedralRadiance?: WebGpuLiveSvoTetrahedralRadiance;
  private readonly liveDerivedPlanner?: WebGpuLiveSvoDerivedWorklistPlanner;
  private readonly liveDerivedBuilder?: WebGpuLiveSvoDerivedBuilder;
  private readonly liveDerivedPlannedBasePageKeys = new Set<string>();
  private liveDerivedAddressPlanValid = true;
  private liveDerivedInitial = true;
  private liveScenePrimitiveStates = new Map<string, LiveScenePrimitiveState>();
  private liveScenePrimitives = new Map<string, LiveScenePrimitiveEntry>();
  private readonly coveredSceneBrickCoordinates = new Set<string>();
  private readonly pendingTopologyCoordinates = new Map<string, SparseBrickCoordinate>();
  private pendingScenePublication?: SparseScenePublication;
  private pendingTopologyMutation = false;
  private sceneRevision = 0;
  private topologyPublished = false;
  private pipelineInitialization?: Promise<void>;
  private inspectionInitialization?: Promise<SparseVoxelRenderSource>;
  private destroyed = false;

  constructor(device: GPUDevice, scene: SceneDescription, dimensions: readonly [number, number, number], options: OctreeSparseBrickWorldOptions = {}) {
    this.device = device;
    this.dimensions = dimensions;
    const brickSize = options.brickSize ?? 8;
    this.brickSize = brickSize;
    const environmentCatalog = buildEnvironmentProxyCatalog(scene, scene.environment ?? "default");
    const environmentPrimitives = environmentProxyPrimitives(environmentCatalog, true);
    const sceneDomain = planSparseSceneDomain(
      scene, dimensions, brickSize,
      environmentPrimitives.map((primitive) => ({ min: primitive.aabb_m.min, max: primitive.aabb_m.max })),
      { conservativePaddingCells: 1, worldBounds_m: scene.voxelDomain.bounds_m }
    );
    this.solverGridOriginCells = sceneDomain.solverGridOriginCells;
    this.sceneBrickDimensions = sceneDomain.brickDimensions;
    this.sceneWorldOrigin = [sceneDomain.worldOrigin_m.x, sceneDomain.worldOrigin_m.y, sceneDomain.worldOrigin_m.z];
    const maximumDepth = sparseSceneOctreeMaximumDepth(sceneDomain.brickDimensions, sceneDomain.coordinates);
    const plan = planAdaptiveSparseBrickOctree({
      brickSize,
      solverBricks: sceneDomain.solverBrickCoordinates,
      proxyBricks: sceneDomain.proxyBrickCoordinates.flat(),
      maximumDepth,
      maximumEnvironmentCoarseningPower: Math.min(
        environmentMaximumCoarseningPower(options.environmentBrickRefinementLevels),
        maximumDepth,
      )
    });
    this.finestLevel = plan.maximumDepth;
    for (const leaf of plan.leaves) {
      const node = plan.nodes[leaf.nodeIndex];
      const scale = 2 ** (plan.maximumDepth - node.level);
      const first = [node.coordinate.x * scale, node.coordinate.y * scale, node.coordinate.z * scale];
      for (let z = first[2]; z < Math.min(first[2] + scale, this.sceneBrickDimensions[2]); z += 1) {
        for (let y = first[1]; y < Math.min(first[1] + scale, this.sceneBrickDimensions[1]); y += 1) {
          for (let x = first[0]; x < Math.min(first[0] + scale, this.sceneBrickDimensions[0]); x += 1) {
            this.coveredSceneBrickCoordinates.add(brickCoordinateKey({ x, y, z }));
          }
        }
      }
    }
    const packed = packSparseBrickPlan(plan, 1);
    const sceneBrickVolume = sceneDomain.brickDimensions.reduce((product, value) => product * value, 1);
    const requestedMutationCapacity = options.sceneMutationBrickCapacity ?? OCTREE_LIVE_SCENE_MUTATION_BRICK_CAPACITY;
    if (!Number.isSafeInteger(requestedMutationCapacity) || requestedMutationCapacity < 1) {
      throw new RangeError("Live scene mutation brick capacity must be a positive safe integer");
    }
    this.topologyMutationCapacity = Math.min(sceneBrickVolume, requestedMutationCapacity);
    const nodeCapacity = Math.max(1, plan.nodes.length
      + sparseBrickTopologyMutationNodeReserve(maximumDepth, this.topologyMutationCapacity));
    const leafCapacity = Math.max(1, plan.leaves.length + this.topologyMutationCapacity);
    this.tree = new SparseBrickOctreeGPU(device, {
      brickSize, nodeCapacity, leafCapacity, label: "Octree unified live sparse-brick world",
    });
    this.brickOccupancyBuilder = new WebGpuSvoBrickOccupancyBuilder(device);
    this.topologyMutator = new WebGpuSparseBrickTopologyMutator(device);
    this.topologyMutationWorklist = device.createBuffer({
      label: "Live sparse scene topology mutation worklist",
      size: (8 + this.topologyMutationCapacity * 4) * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.compactHierarchy = plan.nodes.length === 0 ? undefined : new WebGpuSvoCompactHierarchy(device,
      packSvoCompactHierarchy({ nodes: packed.nodes, leaves: packed.leaves,
        publishedNodeCount: plan.nodes.length, publishedLeafCount: plan.leaves.length }, 1));
    // Renderer traversal snapshots are optional views of the canonical terminal
    // set. A later topology revision rejects them by generation; they never own
    // structure and failure simply omits the capability.
    let wideFanout: WebGPUSvoWideFanout | undefined;
    try {
      const widePlan = planOctreeSvoWideFanout(plan);
      wideFanout = new WebGPUSvoWideFanout(device, {
        maximumPages: Math.max(1, widePlan.pages.length),
        maximumDescriptors: Math.max(1, widePlan.descriptorCount),
      });
      const encoder = device.createCommandEncoder({ label: "Publish immutable SVO wide-fanout hierarchy" });
      if (wideFanout.encode(encoder, widePlan) !== "encoded" || !wideFanout.capability()) {
        wideFanout.destroy();
        wideFanout = undefined;
      }
    } catch {
      wideFanout?.destroy();
      wideFanout = undefined;
    }
    this.wideFanout = wideFanout;
    const solverOriginBricks = this.solverGridOriginCells.map((value) => value / brickSize);
    if (solverOriginBricks.some((value) => !Number.isInteger(value))) throw new Error("Shared sparse scene origin must align to the fluid brick lattice");
    const leafByCoordinate = new Map<string, number>();
    for (const leaf of plan.leaves) {
      const node = plan.nodes[leaf.nodeIndex];
      if (node.level === plan.maximumDepth) leafByCoordinate.set(`${leaf.coordinate.x},${leaf.coordinate.y},${leaf.coordinate.z}`, leaf.index);
    }
    const localBrickDimensions = dimensions.map((value) => Math.ceil(value / brickSize)) as [number, number, number];
    const leafIndices = new Uint32Array(localBrickDimensions[0] * localBrickDimensions[1] * localBrickDimensions[2]);
    let mappedBrick = 0;
    for (let z = 0; z < localBrickDimensions[2]; z += 1) for (let y = 0; y < localBrickDimensions[1]; y += 1) for (let x = 0; x < localBrickDimensions[0]; x += 1) {
      const key = `${solverOriginBricks[0] + x},${solverOriginBricks[1] + y},${solverOriginBricks[2] + z}`;
      const leafIndex = leafByCoordinate.get(key);
      if (leafIndex === undefined) throw new Error(`Fluid brick ${key} has no finest scene leaf`);
      leafIndices[mappedBrick++] = leafIndex;
    }
    this.residency = new GPUFluidBrickResidency(device, dimensions, sceneDomain.cellSize_m, {
      brickSize, haloCells: options.haloCells ?? 2, retireAfterFrames: 3, leafIndices, leafCapacity: this.tree.leafCapacity,
      topologyTileBricks: options.topologyTileBricks ?? 1,
    });
    this.preActivation = options.brickPreActivation ?? true;
    if (options.bulkResidency) {
      // Bulk velocity must remain defined throughout deep liquid, while the
      // surface path wins by visiting only a narrow two-sided band. The two
      // schedulers stay independent so topology support never widens surface
      // redistance back to O(wet volume).
      this.bulkResidency = new GPUFluidBrickResidency(device, dimensions, sceneDomain.cellSize_m, {
        brickSize,
        haloCells: options.haloCells ?? 2,
        retireAfterFrames: 3,
        includeLiquidInterior: true,
        includePressureBoundarySupport: options.includePressureBoundarySupport,
        pressureBoundaryTopClosed: options.pressureBoundaryTopClosed,
        includeWholeDomainPressureSupport: options.includeWholeDomainPressureSupport,
        fluidGatedBoundarySupport: options.fluidGatedBoundarySupport,
        leafIndices,
        leafCapacity: this.tree.leafCapacity,
        topologyTileBricks: options.topologyTileBricks ?? 1,
      });
    }

    const counts = storageBuffer(device, "Sparse brick source counts", packed.counts.byteLength, packed.counts);
    const topology = storageBuffer(device, "Sparse brick source topology", packed.topology.byteLength, packed.topology);
    const initialVoxelCount = Math.max(1, plan.voxelCount);
    const geometry = storageBuffer(device, "Sparse brick source geometry", initialVoxelCount * 16);
    const velocity = storageBuffer(device, "Sparse brick source velocity", initialVoxelCount * 16);
    const materialOwners = storageBuffer(device, "Sparse brick source material owners", initialVoxelCount * 4);
    this.sourceBuffers = [counts, topology, geometry, velocity, materialOwners];
    this.source = { counts, topology, geometry, velocity, materialOwners, capacities: {
      nodes: plan.nodes.length, leaves: plan.leaves.length, voxels: plan.voxelCount,
    } };

    const baseMaterials = packVoxelDebugMaterialTable();
    const materialCount = ENVIRONMENT_VOXEL_MATERIAL_BASE + environmentPrimitives.length;
    const materialData = new Float32Array(Math.max(baseMaterials.length, materialCount * 8));
    materialData.set(baseMaterials);
    environmentPrimitives.forEach((primitive) => {
      const offset = (ENVIRONMENT_VOXEL_MATERIAL_BASE + primitive.ownerIndex) * 8;
      const color = primitive.material.colorLinear;
      // Room shells remain part of the sparse model and brick-grid topology,
      // but an opaque front wall would hide every interior prop in raw mode.
      // Alpha zero is an inspection-only visibility flag; material and owner
      // identity stay intact for modelling and collision queries.
      const inspectionAlpha = primitive.tags.includes("shell") ? 0 : 1;
      materialData.set([
        color[0], color[1], color[2], inspectionAlpha,
        color[0] * primitive.material.emission, color[1] * primitive.material.emission, color[2] * primitive.material.emission,
        primitive.material.roughness
      ], offset);
    });
    this.inspectionMaterialData = materialData;
    const pbrMaterials = buildOctreeSvoPbrMaterialPublication(
      OCTREE_SVO_PBR_MATERIAL_REVISION,
      environmentPrimitives,
    );
    this.pbrMaterialBuffer = storageBuffer(
      device,
      "Sparse voxel PBR material table",
      OCTREE_LIVE_SCENE_MATERIAL_CAPACITY * SVO_MATERIAL_RECORD_STRIDE_BYTES,
      pbrMaterials.packedRecords,
    );
    this.materialEmissionBuffer = storageBuffer(
      device,
      "Live SVO material emission table",
      OCTREE_LIVE_SCENE_MATERIAL_CAPACITY * 16,
      packLiveSvoMaterialEmission(pbrMaterials.packedRecords, pbrMaterials.count),
    );
    const lights = buildOctreeSvoLightPublication(scene);
    this.lightBuffer = storageBuffer(
      device,
      "Sparse voxel authored light table",
      lights.packedRecords.byteLength,
      lights.packedRecords,
    );
    const environmentLighting = buildOctreeSvoEnvironmentLightingPublication(scene);
    this.environmentLightingBuffer = storageBuffer(
      device,
      "Sparse voxel environment lighting",
      environmentLighting.packedRecords.byteLength,
      environmentLighting.packedRecords,
    );
    const bodyMaterials = new Uint32Array(Math.max(1, scene.rigidBodies.length));
    scene.rigidBodies.forEach((body, index) => { bodyMaterials[index] = materialIdForRigidShape(body.shape); });
    this.inspectionBodyMaterials = bodyMaterials;
    const c = scene.container;
    this.containerClosedTop = c.top === "closed";
    this.cellSize = sceneDomain.cellSize_m;
    const parameterData = new ArrayBuffer(64), uints = new Uint32Array(parameterData), floats = new Float32Array(parameterData);
    uints.set([sceneDomain.sceneDimensionsCells[0], sceneDomain.sceneDimensionsCells[1], sceneDomain.sceneDimensionsCells[2], 0], 0);
    floats.set([sceneDomain.worldOrigin_m.x, sceneDomain.worldOrigin_m.y, sceneDomain.worldOrigin_m.z, 0], 4);
    floats.set([...this.cellSize, 0], 8);
    uints.set([brickSize, scene.rigidBodies.length, plan.maximumDepth, VOXEL_MATERIAL_IDS.fluid], 12);
    this.inspectionParameterData = parameterData;
    this.proxyVoxelizer = new SparseSceneProxyVoxelizer(device, this.tree, {
      cellSize: this.cellSize,
      worldOrigin: [sceneDomain.worldOrigin_m.x, sceneDomain.worldOrigin_m.y, sceneDomain.worldOrigin_m.z],
      finestLevel: plan.maximumDepth,
      primitiveCapacity: OCTREE_LIVE_SCENE_PRIMITIVE_CAPACITY,
      dirtyRegionCapacity: OCTREE_LIVE_SCENE_DIRTY_REGION_CAPACITY,
      dirtyBrickCapacity: this.tree.leafCapacity,
      candidatesPerDirtyBrick: OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK,
      label: `${environmentCatalog.environmentId} live scene geometry`,
    });
    if (SPARSE_VOXEL_PUBLICATION_STATE.strideBytes !== SPARSE_BRICK_GPU_LAYOUT.publicationStrideBytes) {
      throw new Error("Sparse voxel publication ABI does not match the structural arena");
    }
    this.structuralPublicationState = this.tree.structuralPublication;
    const structuralLayout = device.createBindGroupLayout({
      label: "Sparse voxel structural publication finalizer layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ],
    });
    this.structuralPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [structuralLayout] });
    this.structuralFinalizeBindGroup = device.createBindGroup({
      label: "Sparse voxel structural publication finalizer bindings",
      layout: structuralLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.structuralPublicationState,
            offset: this.tree.structuralPublicationOffsetBytes,
            size: SPARSE_VOXEL_PUBLICATION_STATE.strideBytes,
          },
        },
        { binding: 1, resource: { buffer: this.proxyVoxelizer.maintenanceBinding.buffer } },
        { binding: 2, resource: { buffer: this.topologyMutationWorklist } },
      ],
    });
    const publicationBinding = {
      buffer: this.structuralPublicationState,
      offset: this.tree.structuralPublicationOffsetBytes,
      size: SPARSE_VOXEL_PUBLICATION_STATE.strideBytes,
    };
    const publicationWord = (word: number) => ({ binding: publicationBinding, word });
    const residencyLayout = sparseVoxelFluidResidencyLayout(this.residency.capacity);
    if (residencyLayout.worklistByteLength !== this.residency.worklistByteLength) {
      throw new Error("Sparse voxel residency ABI does not match the producer worklist allocation");
    }
    let nodeMipPyramid: WebGpuLiveSvoNodeMipPyramid | undefined;
    let tetrahedralRadiance: WebGpuLiveSvoTetrahedralRadiance | undefined;
    let liveDerivedPlanner: WebGpuLiveSvoDerivedWorklistPlanner | undefined;
    let liveDerivedBuilder: WebGpuLiveSvoDerivedBuilder | undefined;
    const liveDerivedBasePageDimensions = liveSvoBasePageDimensions(this.sceneBrickDimensions, brickSize);
    const liveDerivedLevelCount = sparseSceneOctreeMaximumDepth(liveDerivedBasePageDimensions, []) + 1;
    const maximumDerivedPages = webGpuSvoNodeMipMaximumPages(device);
    let derivedLighting: NonNullable<SparseVoxelSceneRenderSource["derivedLighting"]> = {
      state: "unavailable",
      reason: "unsupported-level-count",
      detail: `Live SVO derived lighting needs ${liveDerivedLevelCount} mip levels; the runtime supports at most 12`,
      requiredPages: 0,
      capacity: maximumDerivedPages,
    };
    if (liveDerivedLevelCount <= 12) {
      try {
        const occupiedPages = liveSvoPlanBasePages(plan);
        occupiedPages.forEach((page) => this.liveDerivedPlannedBasePageKeys.add(page.join(",")));
        const mipPlan = planSvoNodeMipPyramid({
          generation: 1,
          occupiedPages,
          levelCount: liveDerivedLevelCount,
          capacity: maximumDerivedPages,
        });
        derivedLighting = {
          state: "unavailable",
          reason: "capacity",
          detail: `Live SVO derived lighting needs ${mipPlan.requestedPageCount} sparse pages; this device can address ${maximumDerivedPages}`,
          requiredPages: mipPlan.requestedPageCount,
          capacity: maximumDerivedPages,
        };
        const direct = createWebGpuSvoNodeMipDirectPageTable(mipPlan, device.limits.maxTextureDimension3D);
        const atlasFits = mipPlan.atlas.texels.every((value) => value <= device.limits.maxTextureDimension3D);
        if (!mipPlan.complete || !direct.ready || !atlasFits || mipPlan.pages.length === 0) {
          throw new RangeError("Live SVO derived-page capacity cannot cover the declared editable domain");
        }
        nodeMipPyramid = new WebGpuLiveSvoNodeMipPyramid(device, {
          pageCapacity: mipPlan.pages.length,
          atlasTexels: mipPlan.atlas.texels,
          directPageTableDimensions: direct.dimensions,
          label: "Unified live SVO node mips",
        });
        tetrahedralRadiance = new WebGpuLiveSvoTetrahedralRadiance(device, {
          pageCapacity: mipPlan.pages.length,
          atlasTexels: mipPlan.atlas.texels,
          label: "Unified live SVO radiance",
        });
        const nodeTarget = nodeMipPyramid.prepareGpuUpdate(mipPlan);
        const radianceTarget = tetrahedralRadiance.prepareGpuUpdate(mipPlan);
        const sceneMaintenance = this.proxyVoxelizer.maintenanceBinding;
        const liveDerivedGenerationSource = {
          buffer: this.structuralPublicationState,
          offsetBytes: this.tree.structuralPublicationOffsetBytes
            + SPARSE_VOXEL_PUBLICATION_STATE.completeGeneration * 4,
        };
        liveDerivedPlanner = new WebGpuLiveSvoDerivedWorklistPlanner(device, {
          tree: this.tree,
          nodeMips: nodeTarget,
          dirtyLeafSources: [
            {
              buffer: sceneMaintenance.buffer,
              countOffsetBytes: sceneMaintenance.stateOffsetBytes
                + SPARSE_SCENE_MAINTENANCE_STATE_WORDS.dirtyBrickCount * 4,
              recordOffsetBytes: sceneMaintenance.dirtyBrickOffsetBytes,
              capacity: sceneMaintenance.dirtyBrickCapacity,
              recordStrideWords: 4,
            },
            {
              buffer: this.residency.worklist,
              countOffsetBytes: SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS.activeCount * 4,
              recordOffsetBytes: residencyLayout.activeEntryOffsetBytes + 4,
              capacity: this.residency.capacity,
              recordStrideWords: residencyLayout.entryStrideBytes / 4,
            },
            {
              buffer: this.residency.worklist,
              countOffsetBytes: SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS.retiredCount * 4,
              recordOffsetBytes: residencyLayout.retiredEntryOffsetBytes + 4,
              capacity: this.residency.capacity,
              recordStrideWords: residencyLayout.entryStrideBytes / 4,
            },
          ],
          generationSource: liveDerivedGenerationSource,
          levelCount: liveDerivedLevelCount,
          finestLevel: plan.maximumDepth,
          // Every sparse address is retained, but a level can only emit its
          // own resident pages. Avoid sizing all worklists like the hierarchy.
          pageCapacityPerLevel: Math.max(...Array.from({ length: liveDerivedLevelCount }, (_, level) =>
            mipPlan.pages.filter((page) => page.key.level === level).length)),
          label: "Unified live SVO derived-page planner",
        });
        liveDerivedBuilder = new WebGpuLiveSvoDerivedBuilder(device, {
          tree: this.tree,
          nodeMips: nodeTarget,
          radiance: radianceTarget,
          materialEmission: this.materialEmissionBuffer,
          worklists: liveDerivedPlanner.worklists,
          generationSource: liveDerivedGenerationSource,
          plannedPageCount: mipPlan.pages.length,
          finestLevel: plan.maximumDepth,
          label: "Unified live SVO derived-page builder",
        });
        nodeMipPyramid.acceptGpuUpdate(mipPlan.generation);
        tetrahedralRadiance.acceptGpuUpdate(mipPlan.generation);
        derivedLighting = {
          state: "ready",
          requiredPages: mipPlan.requestedPageCount,
          capacity: maximumDerivedPages,
        };
      } catch (error) {
        liveDerivedBuilder?.destroy();
        liveDerivedPlanner?.destroy();
        nodeMipPyramid?.destroy();
        tetrahedralRadiance?.destroy();
        nodeMipPyramid = undefined;
        tetrahedralRadiance = undefined;
        liveDerivedPlanner = undefined;
        liveDerivedBuilder = undefined;
        if (derivedLighting.requiredPages <= derivedLighting.capacity) {
          derivedLighting = {
            ...derivedLighting,
            reason: "initialization-failed",
            detail: error instanceof Error ? error.message : String(error),
          };
        }
        console.warn("[svo] live derived lighting unavailable; exact visibility will be used", error);
      }
    }
    this.liveDerivedAddressPlanValid = derivedLighting.state === "ready";
    this.nodeMipPyramid = nodeMipPyramid;
    this.tetrahedralRadiance = tetrahedralRadiance;
    this.liveDerivedPlanner = liveDerivedPlanner;
    this.liveDerivedBuilder = liveDerivedBuilder;
    const residencyWorklistBinding = { buffer: this.residency.worklist, size: this.residency.worklistByteLength };
    const residencyWord = (word: number) => ({ binding: residencyWorklistBinding, word });
    const activeResidencyList = {
      count: residencyWord(SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS.activeCount),
      entryOffsetBytes: residencyLayout.activeEntryOffsetBytes,
      entryStrideBytes: residencyLayout.entryStrideBytes,
      capacity: this.residency.capacity,
    };
    const structural: SparseVoxelStructuralRenderSource = {
      structure: { buffer: this.tree.structure, size: this.tree.structure.size },
      structureOffsetsWords: {
        control: this.tree.controlOffsetBytes / Uint32Array.BYTES_PER_ELEMENT,
        publication: this.tree.structuralPublicationOffsetBytes / Uint32Array.BYTES_PER_ELEMENT,
        nodes: this.tree.nodeOffsetBytes / Uint32Array.BYTES_PER_ELEMENT,
        leaves: this.tree.leafOffsetBytes / Uint32Array.BYTES_PER_ELEMENT,
      },
      control: { buffer: this.tree.control, size: SPARSE_BRICK_GPU_LAYOUT.controlStrideBytes },
      nodes: { buffer: this.tree.nodes, offset: this.tree.nodeOffsetBytes, size: this.tree.nodeCapacity * SPARSE_BRICK_GPU_LAYOUT.nodeStrideBytes },
      leaves: { buffer: this.tree.leaves, offset: this.tree.leafOffsetBytes, size: this.tree.leafCapacity * SPARSE_BRICK_GPU_LAYOUT.leafStrideBytes },
      geometry: { buffer: this.tree.geometry, offset: this.tree.geometryOffsetBytes, size: this.tree.voxelCapacity * SPARSE_BRICK_GPU_LAYOUT.geometryStrideBytes },
      sceneGeometry: { buffer: this.tree.sceneGeometry, offset: this.tree.sceneGeometryOffsetBytes, size: this.tree.voxelCapacity * SPARSE_BRICK_GPU_LAYOUT.geometryStrideBytes },
      velocity: { buffer: this.tree.velocity, offset: this.tree.velocityOffsetBytes, size: this.tree.voxelCapacity * SPARSE_BRICK_GPU_LAYOUT.velocityStrideBytes },
      materialOwners: { buffer: this.tree.materialOwners, offset: this.tree.materialOwnerOffsetBytes, size: this.tree.voxelCapacity * SPARSE_BRICK_GPU_LAYOUT.materialOwnerStrideBytes },
      sceneMaterialOwners: { buffer: this.tree.sceneMaterialOwners, offset: this.tree.sceneMaterialOwnerOffsetBytes, size: this.tree.voxelCapacity * SPARSE_BRICK_GPU_LAYOUT.materialOwnerStrideBytes },
      fluidLeafStates: { buffer: this.residency.leafStates, size: this.tree.leafCapacity * Uint32Array.BYTES_PER_ELEMENT },
      fluidResidency: {
        states: { buffer: this.residency.stateBuffer, size: this.residency.capacity * residencyLayout.stateStrideBytes },
        worklist: residencyWorklistBinding,
        domain: {
          originBricks: solverOriginBricks as [number, number, number],
          dimensionsBricks: localBrickDimensions,
        },
        stateStrideBytes: residencyLayout.stateStrideBytes,
        stateBits: SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS,
        active: activeResidencyList,
        core: {
          ...activeResidencyList,
          count: residencyWord(SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS.coreCount),
          requiredStateBit: SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.core,
        },
        halo: {
          ...activeResidencyList,
          count: residencyWord(SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS.haloCount),
          requiredStateBit: SPARSE_VOXEL_FLUID_RESIDENCY_STATE_BITS.halo,
        },
        retired: {
          count: residencyWord(SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS.retiredCount),
          entryOffsetBytes: residencyLayout.retiredEntryOffsetBytes,
          entryStrideBytes: residencyLayout.entryStrideBytes,
          capacity: this.residency.capacity,
        },
        counters: {
          activated: residencyWord(SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS.activatedCount),
        },
        generation: residencyWord(SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS.generation),
        revision: publicationWord(SPARSE_VOXEL_PUBLICATION_STATE.coarseFluidRevision),
        owner: "GPUFluidBrickResidency",
      },
      capacities: { nodes: this.tree.nodeCapacity, leaves: this.tree.leafCapacity, voxels: this.tree.voxelCapacity },
      strides: {
        control: SPARSE_BRICK_GPU_LAYOUT.controlStrideBytes,
        node: SPARSE_BRICK_GPU_LAYOUT.nodeStrideBytes,
        leaf: SPARSE_BRICK_GPU_LAYOUT.leafStrideBytes,
        geometry: SPARSE_BRICK_GPU_LAYOUT.geometryStrideBytes,
        velocity: SPARSE_BRICK_GPU_LAYOUT.velocityStrideBytes,
        materialOwner: SPARSE_BRICK_GPU_LAYOUT.materialOwnerStrideBytes,
      },
      domain: {
        worldOrigin_m: [sceneDomain.worldOrigin_m.x, sceneDomain.worldOrigin_m.y, sceneDomain.worldOrigin_m.z],
        cellSize_m: this.cellSize,
        dimensionsCells: sceneDomain.sceneDimensionsCells,
        brickSize,
        maximumDepth: plan.maximumDepth,
      },
      publication: {
        state: publicationBinding,
        completeGeneration: publicationWord(SPARSE_VOXEL_PUBLICATION_STATE.completeGeneration),
        validFields: publicationWord(SPARSE_VOXEL_PUBLICATION_STATE.validFields),
        revisions: {
          topology: publicationWord(SPARSE_VOXEL_PUBLICATION_STATE.topologyRevision),
          sceneGeometry: publicationWord(SPARSE_VOXEL_PUBLICATION_STATE.sceneGeometryRevision),
          dynamicSolid: publicationWord(SPARSE_VOXEL_PUBLICATION_STATE.dynamicSolidRevision),
          coarseFluid: publicationWord(SPARSE_VOXEL_PUBLICATION_STATE.coarseFluidRevision),
          fineFluid: publicationWord(SPARSE_VOXEL_PUBLICATION_STATE.fineFluidRevision),
        },
      },
      fields: {
        topology: { bit: SPARSE_VOXEL_VALID_FIELDS.topology, residency: "all-published-leaves" },
        sceneGeometry: { bit: SPARSE_VOXEL_VALID_FIELDS.sceneGeometry, signedDistance: "negative-inside-metres", distanceQuality: "mixed-exact-approximate", residency: "all-published-leaves" },
        dynamicSolid: { bit: SPARSE_VOXEL_VALID_FIELDS.dynamicSolid, signedDistance: "negative-inside-metres", distanceQuality: "occupancy-estimate", residency: "fluid-resident-leaves" },
        coarseFluid: { bit: SPARSE_VOXEL_VALID_FIELDS.coarseFluid, signedDistance: "negative-inside-metres", distanceQuality: "metric-near-interface", residency: "fluid-resident-leaves" },
        fineFluid: { bit: SPARSE_VOXEL_VALID_FIELDS.fineFluid, signedDistance: "negative-inside-metres", distanceQuality: "metric", residency: "unavailable" },
        velocity: { bit: SPARSE_VOXEL_VALID_FIELDS.velocity, residency: "fluid-resident-leaves" },
        materialOwner: { bit: SPARSE_VOXEL_VALID_FIELDS.materialOwner, residency: "all-published-leaves" },
      },
    };
    this.sceneSource = {
      pbrMaterials: {
        binding: { buffer: this.pbrMaterialBuffer, size: this.pbrMaterialBuffer.size },
        count: pbrMaterials.count,
        strideBytes: pbrMaterials.strideBytes,
        revision: pbrMaterials.revision,
      },
      lights: {
        binding: { buffer: this.lightBuffer, size: lights.packedRecords.byteLength },
        count: lights.count,
        strideBytes: lights.strideBytes,
        revision: lights.revision,
      },
      environmentLighting: {
        binding: { buffer: this.environmentLightingBuffer, size: environmentLighting.packedRecords.byteLength },
        count: environmentLighting.count,
        strideBytes: environmentLighting.strideBytes,
        revision: environmentLighting.revision,
        cacheKey: environmentLighting.cacheKey,
      },
      materialCount: pbrMaterials.count,
      fluidBrickStats: { buffer: this.residency.worklist }, fluidBrickCapacity: this.residency.capacity,
      structural,
      wideFanout: this.wideFanout?.capability(),
      compactHierarchy: this.compactHierarchy?.capability(),
      nodeMipPyramid: this.nodeMipPyramid?.visibleGeneration() && {
        ...this.nodeMipPyramid.visibleGeneration()!,
        worldOrigin_m: this.sceneWorldOrigin,
        worldExtent_m: sceneDomain.sceneDimensionsCells.map((cells, axis) => cells * sceneDomain.cellSize_m[axis]) as [number, number, number],
      },
      tetrahedralRadiance: this.tetrahedralRadiance?.visibleGeneration(),
      derivedLighting,
      derivedRenderAllocationBytes: {
        wideFanout: this.wideFanout?.allocatedBytes ?? 0,
        compactHierarchy: this.compactHierarchy?.allocatedBytes ?? 0,
        nodeMipPyramid: (this.nodeMipPyramid?.allocatedBytes ?? 0)
          + (this.liveDerivedPlanner?.allocatedBytes ?? 0),
        tetrahedralRadiance: (this.tetrahedralRadiance?.allocatedBytes ?? 0)
          + (this.liveDerivedBuilder?.allocatedBytes ?? 0),
      },
      revision: 1
    };
    this.stageSceneUpdate(scene);
    this.baseAllocatedBytes = this.tree.allocatedBytes + this.residency.allocatedBytes
      + (this.bulkResidency?.allocatedBytes ?? 0)
      + this.sourceBuffers.reduce((sum, buffer) => sum + buffer.size, 0)
      + this.pbrMaterialBuffer.size + this.materialEmissionBuffer.size + this.lightBuffer.size + this.environmentLightingBuffer.size
      + this.topologyMutationWorklist.size + this.topologyMutator.allocatedBytes
      + this.proxyVoxelizer.allocatedBytes + (this.wideFanout?.allocatedBytes ?? 0)
      + (this.compactHierarchy?.allocatedBytes ?? 0)
      + (this.nodeMipPyramid?.allocatedBytes ?? 0)
      + (this.tetrahedralRadiance?.allocatedBytes ?? 0)
      + (this.liveDerivedPlanner?.allocatedBytes ?? 0)
      + (this.liveDerivedBuilder?.allocatedBytes ?? 0);
  }

  /** Compile this world's structural programs and every owned maintenance pipeline. */
  initializePipelines(): Promise<void> {
    if (this.destroyed) return Promise.reject(new Error("Cannot initialize a destroyed sparse-brick world"));
    this.pipelineInitialization ??= (async () => {
      this.structuralModule = this.device.createShaderModule({
        label: "Sparse voxel structural publication finalizer",
        code: structuralPublicationFinalizeShader(
          this.proxyVoxelizer.maintenanceBinding.stateOffsetBytes / 4),
      });
      const structuralPipelines = Promise.all([
        this.device.createComputePipelineAsync({
          label: "Finalize live sparse voxel physics publication",
          layout: this.structuralPipelineLayout,
          compute: { module: this.structuralModule, entryPoint: "finalizePhysics" },
        }),
        this.device.createComputePipelineAsync({
          label: "Finalize live sparse voxel scene publication",
          layout: this.structuralPipelineLayout,
          compute: { module: this.structuralModule, entryPoint: "finalizeScene" },
        }),
      ] as const);
      const [, compiledStructuralPipelines] = await Promise.all([
        Promise.all([
          this.tree.initializePipelines(),
          this.brickOccupancyBuilder.initializePipelines(),
          this.proxyVoxelizer.initializePipelines(),
          this.topologyMutator.initializePipelines(),
          this.liveDerivedPlanner?.initializePipelines(),
          this.liveDerivedBuilder?.initializePipelines(),
        ]),
        structuralPipelines,
      ]);
      [this.structuralPhysicsPipeline, this.structuralScenePipeline] = compiledStructuralPipelines;
    })().catch((error) => {
      this.pipelineInitialization = undefined;
      throw error;
    });
    return this.pipelineInitialization;
  }

  get allocatedBytes(): number { return this.baseAllocatedBytes + (this.inspection?.allocatedBytes ?? 0); }

  /** The already-compiled inspection source, without starting lazy compilation. */
  get inspectionSource(): SparseVoxelRenderSource | undefined { return this.inspection?.source; }

  /** Allocate expanded diagnostic records only when raw/grid inspection asks for them. */
  ensureInspectionSource(): Promise<SparseVoxelRenderSource> {
    if (this.destroyed) throw new Error("Cannot inspect a destroyed sparse-brick world");
    if (this.inspection) return Promise.resolve(this.inspection.source);
    this.inspectionInitialization ??= this.initializeInspectionSource().catch((error) => {
      this.inspectionInitialization = undefined;
      throw error;
    });
    return this.inspectionInitialization;
  }

  private async initializeInspectionSource(): Promise<SparseVoxelRenderSource> {
    const voxelRecords = storageBuffer(this.device, "Sparse voxel debug records", this.tree.voxelCapacity * SPARSE_VOXEL_DEBUG_RECORD_STRIDE);
    const brickRecords = storageBuffer(this.device, "Sparse brick debug records", this.tree.leafCapacity * SPARSE_VOXEL_DEBUG_RECORD_STRIDE);
    const voxelCount = storageBuffer(this.device, "Sparse voxel render count", 4);
    const brickCount = storageBuffer(this.device, "Sparse brick render count", 4);
    const materialBuffer = storageBuffer(this.device, "Sparse voxel material table", this.inspectionMaterialData.byteLength, this.inspectionMaterialData);
    const bodyMaterialBuffer = storageBuffer(this.device, "Sparse voxel body material IDs", this.inspectionBodyMaterials.byteLength, this.inspectionBodyMaterials);
    const params = this.device.createBuffer({ label: "Sparse voxel debug publication parameters", size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(params, 0, this.inspectionParameterData);
    const shaderModule = this.device.createShaderModule({ label: "Octree sparse-brick render publication", code: debugPublicationShader });
    const debugLayout = this.device.createBindGroupLayout({ label: "Octree sparse-brick render publication layout", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }
    ] });
    const debugPipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [debugLayout] });
    const buffers = [voxelRecords, brickRecords, voxelCount, brickCount, materialBuffer, bodyMaterialBuffer, params];
    let voxelPipeline: GPUComputePipeline;
    let brickPipeline: GPUComputePipeline;
    try {
      [voxelPipeline, brickPipeline] = await Promise.all([
        this.device.createComputePipelineAsync({ label: "Materialize sparse voxel records", layout: debugPipelineLayout, compute: { module: shaderModule, entryPoint: "materializeVoxels" } }),
        this.device.createComputePipelineAsync({ label: "Materialize sparse brick records", layout: debugPipelineLayout, compute: { module: shaderModule, entryPoint: "materializeBricks" } }),
      ]);
    } catch (error) {
      for (const buffer of buffers) buffer.destroy();
      throw error;
    }
    if (this.destroyed) {
      for (const buffer of buffers) buffer.destroy();
      throw new Error("Sparse-brick world was destroyed while inspection pipelines compiled");
    }
    const bindGroup = this.device.createBindGroup({ layout: debugLayout, entries: [
      { binding: 0, resource: { buffer: this.tree.control, size: SPARSE_BRICK_GPU_LAYOUT.controlStrideBytes } },
      { binding: 1, resource: { buffer: this.tree.nodes, offset: this.tree.nodeOffsetBytes, size: this.tree.nodeCapacity * SPARSE_BRICK_GPU_LAYOUT.nodeStrideBytes } },
      { binding: 2, resource: { buffer: this.tree.leaves, offset: this.tree.leafOffsetBytes, size: this.tree.leafCapacity * SPARSE_BRICK_GPU_LAYOUT.leafStrideBytes } },
      { binding: 3, resource: { buffer: this.tree.materialOwners, offset: this.tree.materialOwnerOffsetBytes } },
      { binding: 4, resource: { buffer: bodyMaterialBuffer } }, { binding: 5, resource: { buffer: params } },
      { binding: 6, resource: { buffer: voxelRecords } }, { binding: 7, resource: { buffer: brickRecords } },
      { binding: 8, resource: { buffer: this.residency.leafStates } }
    ] });
    const publication = createSparseVoxelInspectionPublicationController(true, (encoder) => this.encodeInspectionPublication(encoder));
    const source: SparseVoxelRenderSource = {
      ...this.sceneSource,
      voxelRecords: { buffer: voxelRecords }, voxelCount: { buffer: voxelCount },
      brickRecords: { buffer: brickRecords }, brickCount: { buffer: brickCount },
      materials: { buffer: materialBuffer },
      voxelCapacity: this.tree.voxelCapacity, brickCapacity: this.tree.leafCapacity,
      inspectionPublication: publication,
    };
    this.inspection = {
      source, publication, buffers, voxelRecords, brickRecords, voxelCount, brickCount,
      voxelPipeline, brickPipeline, bindGroup,
      allocatedBytes: buffers.reduce((sum, buffer) => sum + buffer.size, 0),
    };
    return source;
  }

  /**
   * Stage the newest authored scene as render truth. Multiple editor writes
   * before the next presentation frame coalesce into one publication whose
   * dirty coverage includes every superseded old/new bound.
   */
  stageSceneUpdate(scene: SceneDescription): boolean {
    if (this.destroyed) throw new Error("Cannot update a destroyed sparse-brick world");
    const catalog = buildEnvironmentProxyCatalog(scene, scene.environment ?? "default");
    const authored = environmentProxyPrimitives(catalog, true);
    const liveEntries = [
      ...scene.rigidBodies.map((body, ownerId) => ({
        key: `rigid:${body.id}`,
        primitive: sparseScenePrimitiveForRigidBody(body, ownerId),
        materialSignature: body.shape,
      })),
      ...authored.map((primitive) => ({
        key: primitive.key,
        primitive: sparseScenePrimitiveForProxy(primitive, {
          materialId: ENVIRONMENT_VOXEL_MATERIAL_BASE + primitive.ownerIndex,
          ownerId: scene.rigidBodies.length + primitive.ownerIndex,
        }),
        materialSignature: JSON.stringify(primitive.material),
      })),
    ];
    if (liveEntries.length > OCTREE_LIVE_SCENE_PRIMITIVE_CAPACITY) {
      throw new RangeError(`Live scene needs ${liveEntries.length} primitives but the fixed arena holds ${OCTREE_LIVE_SCENE_PRIMITIVE_CAPACITY}`);
    }
    const initialPublication = this.liveScenePrimitiveStates.size === 0;
    const nextPrimitives = new Map<string, LiveScenePrimitiveEntry>();
    const nextStates = new Map<string, LiveScenePrimitiveState>();
    const dirtyRegions: SparseSceneAxisAlignedBounds[] = [
      ...(this.pendingScenePublication?.dirtyRegions ?? []),
    ];
    const newBounds: SparseSceneAxisAlignedBounds[] = [];
    for (const entry of liveEntries) {
      const live = entry.primitive;
      nextPrimitives.set(entry.key, { primitive: live, materialSignature: entry.materialSignature });
      const state = { signature: `${liveScenePrimitiveSignature(live)}:${entry.materialSignature}`, bounds: sparseScenePrimitiveBounds(live) };
      nextStates.set(entry.key, state);
      const previous = this.liveScenePrimitiveStates.get(entry.key);
      if (!previous || previous.signature !== state.signature) {
        if (previous) dirtyRegions.push(previous.bounds);
        dirtyRegions.push(state.bounds); newBounds.push(state.bounds);
      }
    }
    for (const [key, previous] of this.liveScenePrimitiveStates) {
      if (!nextStates.has(key)) dirtyRegions.push(previous.bounds);
    }
    if (dirtyRegions.length === 0) return false;
    this.liveScenePrimitiveStates = nextStates;
    this.liveScenePrimitives = nextPrimitives;
    const revision = this.advanceSceneRevision();
    const pbrMaterials = buildOctreeSvoPbrMaterialPublication(
      OCTREE_SVO_PBR_MATERIAL_REVISION + revision,
      authored,
    );
    if (pbrMaterials.count > OCTREE_LIVE_SCENE_MATERIAL_CAPACITY) {
      throw new RangeError(`Live scene needs ${pbrMaterials.count} materials but the fixed arena holds ${OCTREE_LIVE_SCENE_MATERIAL_CAPACITY}`);
    }
    this.device.queue.writeBuffer(this.pbrMaterialBuffer, 0, pbrMaterials.packedRecords);
    this.device.queue.writeBuffer(this.materialEmissionBuffer, 0,
      packLiveSvoMaterialEmission(pbrMaterials.packedRecords, pbrMaterials.count));
    const publishedSource = (this as unknown as { sceneSource?: SparseVoxelSceneRenderSource }).sceneSource;
    if (publishedSource) {
      publishedSource.pbrMaterials = {
        binding: { buffer: this.pbrMaterialBuffer, size: this.pbrMaterialBuffer.size },
        count: pbrMaterials.count,
        strideBytes: pbrMaterials.strideBytes,
        revision: pbrMaterials.revision,
      };
      publishedSource.materialCount = pbrMaterials.count;
    }
    this.stagePrimitivePublication(revision, dirtyRegions, newBounds, initialPublication);
    return true;
  }

  /** Allocation-free keyed motion/content updates shared by renderer-only and fluid worlds. */
  stageLivePrimitiveUpdates(updates: readonly SparseScenePrimitiveUpdate[]): boolean {
    if (this.destroyed) throw new Error("Cannot update a destroyed sparse-brick world");
    if (updates.length === 0) return false;
    const dirtyRegions: SparseSceneAxisAlignedBounds[] = [...(this.pendingScenePublication?.dirtyRegions ?? [])];
    const newBounds: SparseSceneAxisAlignedBounds[] = [];
    let changed = false;
    const seen = new Set<string>();
    for (const update of updates) {
      if (!update.key || seen.has(update.key)) throw new RangeError("Live primitive update keys must be unique and nonempty");
      seen.add(update.key);
      const previousEntry = this.liveScenePrimitives.get(update.key);
      const previousState = this.liveScenePrimitiveStates.get(update.key);
      if (!previousEntry || !previousState) throw new RangeError(`Unknown live primitive key ${update.key}`);
      const signature = `${liveScenePrimitiveSignature(update.primitive)}:${previousEntry.materialSignature}`;
      if (signature === previousState.signature) continue;
      const bounds = sparseScenePrimitiveBounds(update.primitive);
      dirtyRegions.push(previousState.bounds, bounds); newBounds.push(bounds); changed = true;
      this.liveScenePrimitives.set(update.key, { ...previousEntry, primitive: update.primitive });
      this.liveScenePrimitiveStates.set(update.key, { signature, bounds });
    }
    if (!changed) return false;
    const revision = this.advanceSceneRevision();
    this.stagePrimitivePublication(revision, dirtyRegions, newBounds, false);
    return true;
  }

  private advanceSceneRevision(): number {
    this.sceneRevision = this.sceneRevision === 0xffff_ffff ? 1 : this.sceneRevision + 1;
    return this.sceneRevision;
  }

  private stagePrimitivePublication(
    revision: number,
    dirtyRegions: readonly SparseSceneAxisAlignedBounds[],
    newBounds: readonly SparseSceneAxisAlignedBounds[],
    initialPublication: boolean,
  ): void {
    const coalescedDirty = coalesceDirtyRegions(dirtyRegions, initialPublication ? 1 : OCTREE_LIVE_SCENE_DIRTY_REGION_CAPACITY);
    this.pendingScenePublication = {
      primitives: [...this.liveScenePrimitives.values()].map(({ primitive }) => primitive),
      dirtyRegions: coalescedDirty,
      revision,
    };
    if (!initialPublication) {
      for (const coordinate of liveSceneMissingBrickCoordinates(
        newBounds, this.sceneWorldOrigin, this.cellSize, this.brickSize,
        this.sceneBrickDimensions, this.coveredSceneBrickCoordinates,
      )) {
        this.pendingTopologyCoordinates.set(brickCoordinateKey(coordinate), coordinate);
        const pageKey = [coordinate.x, coordinate.y, coordinate.z]
          .map((value) => Math.floor(value * this.brickSize / SVO_NODE_MIP_LAYOUT.interiorSize))
          .join(",");
        if (this.liveDerivedAddressPlanValid && !this.liveDerivedPlannedBasePageKeys.has(pageKey)) {
          this.liveDerivedAddressPlanValid = false;
          const current = this.sceneSource.derivedLighting;
          this.sceneSource.derivedLighting = {
            state: "unavailable",
            reason: "address-plan-invalidated",
            detail: `Scene edit activated derived page ${pageKey} outside the fixed sparse address plan; exact visibility is active`,
            requiredPages: current?.requiredPages ?? this.liveDerivedPlannedBasePageKeys.size,
            capacity: current?.capacity ?? webGpuSvoNodeMipMaximumPages(this.device),
          };
          this.sceneSource.nodeMipPyramid = undefined;
          this.sceneSource.tetrahedralRadiance = undefined;
        }
      }
    }
    if (this.pendingTopologyCoordinates.size === 0) {
      this.device.queue.writeBuffer(this.topologyMutationWorklist, 0, new Uint32Array(8));
      this.pendingTopologyMutation = false;
      return;
    }
    const requested = [...this.pendingTopologyCoordinates.values()];
    const packed = packSparseBrickTopologyMutationWorklist(
      requested.slice(0, this.topologyMutationCapacity), revision, this.topologyMutationCapacity,
    );
    packed[0] = requested.length;
    this.device.queue.writeBuffer(this.topologyMutationWorklist, 0, packed);
    this.pendingTopologyMutation = true;
    // Only a topology-changing transaction invalidates topology-keyed wide,
    // compact, and renderer caches. Covered primitive motion leaves it stable.
    this.sceneSource.revision = revision;
  }

  /**
   * Encode bounded maintenance for the latest staged scene revision.
   *
   * This is intentionally callable every presentation frame, including while
   * physics is paused. The scene is always authoritative; topology and voxel
   * payloads are reusable accelerators whose completion generation advances
   * only after the maintenance passes finish.
   */
  encodeSceneMaintenance(encoder: GPUCommandEncoder, deferDerived = false): boolean {
    if (this.destroyed) return false;
    let encoded = false;
    if (!this.topologyPublished) {
      this.tree.encodePublish(encoder, this.source);
      this.topologyPublished = true;
      encoded = true;
    }
    if (this.pendingTopologyMutation) {
      this.topologyMutator.encode(encoder, this.tree, {
        buffer: this.topologyMutationWorklist,
        capacity: this.topologyMutationCapacity,
      }, {
        maximumDepth: this.finestLevel,
        brickDimensions: this.sceneBrickDimensions,
        generation: this.sceneRevision,
        maximumRequests: this.topologyMutationCapacity,
      });
      if (this.pendingTopologyCoordinates.size <= this.topologyMutationCapacity) {
        for (const [key] of this.pendingTopologyCoordinates) this.coveredSceneBrickCoordinates.add(key);
      }
      this.pendingTopologyCoordinates.clear();
      this.pendingTopologyMutation = false;
      encoded = true;
    }
    if (this.pendingScenePublication) {
      this.proxyVoxelizer.publish(this.pendingScenePublication);
      this.pendingScenePublication = undefined;
    }
    encoded = this.proxyVoxelizer.encodeMaintenance(encoder) || encoded;
    if (!encoded) return false;
    const broker = new PassBroker(encoder);
    const finalizer = broker.compute({ label: "Finalize live sparse voxel scene publication" });
    finalizer.setPipeline(this.structuralScenePipeline);
    finalizer.setBindGroup(0, this.structuralFinalizeBindGroup);
    finalizer.dispatchWorkgroups(1);
    broker.fence("live sparse scene publication finalized");
    if (!deferDerived) this.encodeLiveDerivedMaintenance(encoder);
    return true;
  }

  private encodeLiveDerivedMaintenance(encoder: GPUCommandEncoder): void {
    if (!this.liveDerivedAddressPlanValid || !this.liveDerivedPlanner || !this.liveDerivedBuilder) return;
    const initializeEmpty = this.liveDerivedInitial;
    if (initializeEmpty) this.liveDerivedPlanner.encodeInitial(encoder);
    else this.liveDerivedPlanner.encode(encoder);
    this.liveDerivedBuilder.encode(encoder, initializeEmpty);
    this.liveDerivedInitial = false;
    const nodeMip = this.nodeMipPyramid?.visibleGeneration();
    const radiance = this.tetrahedralRadiance?.visibleGeneration();
    this.sceneSource.nodeMipPyramid = nodeMip && {
      ...nodeMip,
      worldOrigin_m: this.sceneWorldOrigin,
      worldExtent_m: this.sceneBrickDimensions.map((bricks, axis) => bricks * this.brickSize * this.cellSize[axis]) as [number, number, number],
    };
    this.sceneSource.tetrahedralRadiance = radiance;
  }

  encode(encoder: GPUCommandEncoder, fields: OctreeSparseBrickDenseFields, dt_s = 0): void {
    if (this.destroyed) return;
    this.residency.encode(encoder, fields.levelSet, fields.velocity, { dt_s, preActivation: this.preActivation });
    this.bulkResidency?.encode(encoder, fields.levelSet, fields.velocity, { dt_s, preActivation: this.preActivation });
    const inspection = this.inspection;
    const encodePlan = planOctreeSparseBrickEncode(inspection?.publication.enabled ?? false);
    this.encodeSceneMaintenance(encoder, true);
    this.tree.encodeFromDenseFields(encoder, {
      levelSet: fields.levelSet.createView(), velocity: fields.velocity.createView(), solidCells: fields.solidCells,
      dimensions: this.dimensions,
      cellSize: this.cellSize,
      fluidMaterialId: VOXEL_MATERIAL_IDS.fluid,
      solidMaterialId: VOXEL_MATERIAL_IDS.terrain,
      containerMaterialId: VOXEL_MATERIAL_IDS.containerGlass,
      containerClosedTop: this.containerClosedTop,
      gridOriginCells: this.solverGridOriginCells,
      finestLevel: this.finestLevel,
      activeBrickWorklist: this.residency.worklist,
    });
    // One conservative terminal summary covers both independently-owned
    // payload lanes; scene and physics writes never overwrite one another.
    this.brickOccupancyBuilder.encodeFluidResidency(encoder, this.tree, this.residency.worklist);
    if (encodePlan.inspectionPublication) {
      this.encodeInspectionPublication(encoder);
      inspection?.publication.markEncoded();
    }
    const finalizerBroker = new PassBroker(encoder);
    const finalizer = finalizerBroker.compute({ label: "Finalize live sparse voxel physics publication" });
    finalizer.setPipeline(this.structuralPhysicsPipeline);
    finalizer.setBindGroup(0, this.structuralFinalizeBindGroup);
    finalizer.dispatchWorkgroups(1);
    finalizerBroker.fence("live sparse physics publication finalized");
    this.encodeLiveDerivedMaintenance(encoder);
  }

  private encodeInspectionPublication(encoder: GPUCommandEncoder): void {
    const inspection = this.inspection;
    if (!inspection) return;
    const broker = new PassBroker(encoder);
    broker.copyBufferToBuffer(this.tree.control, 8, inspection.voxelCount, 0, 4);
    broker.copyBufferToBuffer(this.tree.control, 4, inspection.brickCount, 0, 4);
    const voxelPass = broker.compute({ label: "Publish octree raw voxel records" });
    voxelPass.setPipeline(inspection.voxelPipeline); voxelPass.setBindGroup(0, inspection.bindGroup);
    voxelPass.dispatchWorkgroups(...tiledDebugDispatch(this.tree.voxelCapacity, 256));
    const brickPass = broker.compute({ label: "Publish octree sparse brick records" });
    brickPass.setPipeline(inspection.brickPipeline); brickPass.setBindGroup(0, inspection.bindGroup);
    brickPass.dispatchWorkgroups(...tiledDebugDispatch(this.tree.leafCapacity, 64));
    broker.fence("sparse inspection records published");
  }

  readResidencyStats(): Promise<FluidBrickResidencyStats> { return this.residency.readStats(); }

  readBulkResidencyStats(): Promise<FluidBrickResidencyStats> | undefined { return this.bulkResidency?.readStats(); }

  /** Full wet-domain worklist, independent of the narrow surface band. */
  get bulkResidencyWorklist(): GPUBuffer | undefined { return this.bulkResidency?.worklist; }

  /** Persistent wet-domain topology scheduler on compact authority. */
  get topologyResidency(): GPUFluidBrickResidency { return this.bulkResidency ?? this.residency; }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.tree.destroy();
    this.residency.destroy();
    this.bulkResidency?.destroy();
    this.proxyVoxelizer.destroy();
    this.topologyMutator.destroy();
    this.topologyMutationWorklist.destroy();
    this.wideFanout?.destroy();
    this.compactHierarchy?.destroy();
    this.liveDerivedBuilder?.destroy();
    this.liveDerivedPlanner?.destroy();
    this.nodeMipPyramid?.destroy();
    this.tetrahedralRadiance?.destroy();
    for (const buffer of [...this.sourceBuffers, this.pbrMaterialBuffer, this.materialEmissionBuffer, this.lightBuffer, this.environmentLightingBuffer, ...(this.inspection?.buffers ?? [])]) buffer.destroy();
  }
}

export const octreeSparseBrickDebugPublicationShader = debugPublicationShader;
export const octreeSparseBrickStructuralFinalizeShader = structuralPublicationFinalizeShader();
