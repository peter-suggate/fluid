import type { SceneDescription } from "./model";
import {
  cachedSvoStaticPublication,
  hashSvoStaticPublication,
  internSvoStaticPublication,
} from "./svo-static-publication-cache";
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
  type SparseBrickPublicationSource,
  type SparseBrickSize
} from "./sparse-brick-octree";
import { planAdaptiveSparseBrickOctree } from "./adaptive-sparse-brick-plan";
import { planSparseSceneDomain } from "./sparse-scene-domain";
import { VOXEL_MATERIAL_IDS, materialIdForRigidShape, packVoxelDebugMaterialTable } from "./voxel-scene";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives, type EnvironmentProxyPrimitive } from "./voxel-environments";
import { SparseSceneProxyVoxelizer, type SparseScenePrimitive } from "./webgpu-sparse-scene-proxies";
import {
  SPARSE_VOXEL_DEBUG_RECORD_STRIDE,
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
import {
  WebGPUFluidBrickAtlas,
  type FluidBrickAtlasMode,
  type FluidBrickAtlasSamplingSource,
  type FluidBrickAtlasStats,
} from "./webgpu-brick-atlas";

export interface OctreeSparseBrickWorldOptions {
  brickSize?: SparseBrickSize;
  /** Air-side support retained for pressure-topology rebuilds. */
  haloCells?: number;
  /** Brick-pooled phi/velocity atlas ownership; off avoids atlas allocation. */
  brickAtlas?: "off" | FluidBrickAtlasMode;
  /** Keep the deep-liquid worklist without allocating atlas field payloads. */
  bulkResidencyOnly?: boolean;
  /** Velocity-swept residency support plus downstream neighbor activation. */
  brickPreActivation?: boolean;
  /**
   * Power-of-two bricks per topology-tile axis. Topology rebuilds operate on
   * tiles of max(brickSize, maximumLeafSize) cells so a pressure leaf can
   * never straddle a partial-rebuild boundary; payload residency, the atlas
   * and dense-field clears remain brick-granular.
   */
  topologyTileBricks?: number;
}

export interface OctreeSparseBrickDenseFields {
  levelSet: GPUTexture;
  velocity: GPUTexture;
  solidCells: GPUBuffer;
}

export interface OctreeSparseBrickTimestampWrites {
  residency?: GPUComputePassTimestampWrites;
  publication?: GPUComputePassTimestampWrites;
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

export function planOctreeBrickCoordinates(dimensions: readonly [number, number, number], brickSize: SparseBrickSize) {
  if (brickSize !== 4 && brickSize !== 8) throw new RangeError("Octree brick size must be 4 or 8");
  for (const value of dimensions) if (!Number.isInteger(value) || value < 1) throw new RangeError("Octree field dimensions must be positive integers");
  const brickDimensions = dimensions.map((value) => Math.ceil(value / brickSize)) as [number, number, number];
  const coordinates: SparseBrickCoordinate[] = [];
  for (let z = 0; z < brickDimensions[2]; z += 1) for (let y = 0; y < brickDimensions[1]; y += 1) for (let x = 0; x < brickDimensions[0]; x += 1) coordinates.push({ x, y, z });
  return { brickDimensions, coordinates };
}

export const ENVIRONMENT_VOXEL_MATERIAL_BASE = 32;
export const OCTREE_SVO_PBR_MATERIAL_REVISION = 2;
export const OCTREE_SVO_LIGHT_REVISION = 1;
export const OCTREE_SVO_ENVIRONMENT_LIGHTING_REVISION = 1;

export interface OctreeSvoPbrMaterialPublicationData {
  packedRecords: Uint32Array<ArrayBuffer>;
  count: number;
  strideBytes: number;
  revision: number;
  staticRevision: string;
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
  const staticRevision = hashSvoStaticPublication(new Uint32Array(), JSON.stringify({
    revision,
    environmentPrimitives: environmentPrimitives.map(({ key, ownerIndex, group, tags, material }) => ({ key, ownerIndex, group, tags, material })),
  }));
  const cacheKey = `octree-svo-pbr-material-v1:${staticRevision}`;
  const cached = cachedSvoStaticPublication(octreeSvoPbrMaterialCache, cacheKey);
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
  return internSvoStaticPublication(octreeSvoPbrMaterialCache, cacheKey, {
    packedRecords,
    count: packedRecords.byteLength / SVO_MATERIAL_RECORD_STRIDE_BYTES,
    strideBytes: SVO_MATERIAL_RECORD_STRIDE_BYTES,
    revision,
    staticRevision,
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
  staticRevision: string;
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
  return internSvoStaticPublication(octreeSvoLightCache, cacheKey, {
    records: lights.records,
    packedRecords: lights.packedRecords,
    count: lights.records.length,
    strideBytes: SVO_LIGHT_RECORD_STRIDE_BYTES,
    revision: lights.revision,
    omittedFixtureKeys: lights.omittedFixtureKeys,
    staticRevision: lights.staticRevision,
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
  scene: Pick<SceneDescription, "environment">,
  revision = OCTREE_SVO_ENVIRONMENT_LIGHTING_REVISION,
): OctreeSvoEnvironmentLightingPublicationData {
  const lighting = buildSvoEnvironmentLighting(scene.environment ?? "default", revision);
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
  let residency = fluidLeafStates[leafIndex];
  let fluidResident = (residency & 1u) != 0u;
  if (fluidResident) { isActive = true; material = params.settings.w; owner = 0xffffu; }
  let leaf = leaves[leafIndex];
  let node = nodes[leaf.topology.x];
  let brick = decodeMorton(leaf.topology.z, leaf.topology.w, node.address.z);
  let scale = 1u << (params.settings.z - node.address.z);
  let world = params.origin.xyz + vec3f(brick * brickSize * scale) * params.cell.xyz;
  let extent = f32(brickSize * scale) * params.cell.xyz;
  brickRecords[leafIndex] = DebugRecord(vec4f(world, 0.0), vec4f(extent, 0.0), vec4u(material, select(0u, 1u, isActive) | residency, node.address.z, owner));
}
`;

const structuralPublicationFinalizeShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> state: array<atomic<u32>>;

const VALID_FIELDS: u32 = ${
  SPARSE_VOXEL_VALID_FIELDS.topology |
  SPARSE_VOXEL_VALID_FIELDS.staticGeometry |
  SPARSE_VOXEL_VALID_FIELDS.dynamicSolid |
  SPARSE_VOXEL_VALID_FIELDS.coarseFluid |
  SPARSE_VOXEL_VALID_FIELDS.velocity |
  SPARSE_VOXEL_VALID_FIELDS.materialOwner
}u;

fn finishFrame(first: bool) {
  if (first) {
    atomicStore(&state[${SPARSE_VOXEL_PUBLICATION_STATE.topologyRevision}], 1u);
    atomicStore(&state[${SPARSE_VOXEL_PUBLICATION_STATE.staticGeometryRevision}], 1u);
  }
  atomicAdd(&state[${SPARSE_VOXEL_PUBLICATION_STATE.dynamicSolidRevision}], 1u);
  atomicAdd(&state[${SPARSE_VOXEL_PUBLICATION_STATE.coarseFluidRevision}], 1u);
  // Fine fluid remains explicitly unavailable (validity bit and revision zero)
  // until the sparse surface-band atlas is attached to this source.
  atomicStore(&state[${SPARSE_VOXEL_PUBLICATION_STATE.validFields}], VALID_FIELDS);
  // This is deliberately last: prior passes and the stores above define one
  // complete structural snapshot for consumers later in the command stream.
  atomicAdd(&state[${SPARSE_VOXEL_PUBLICATION_STATE.completeGeneration}], 1u);
}

@compute @workgroup_size(1)
fn finalizeInitial() { finishFrame(true); }

@compute @workgroup_size(1)
fn finalizeFrame() { finishFrame(false); }
`;

/**
 * Transitional GPU bridge: the octree solver remains authoritative while its
 * resident level set, velocity and VOS solid field are published into one
 * sparse-brick ABI for scene inspection and subsequent sparse kernels.
 */
export class OctreeSparseBrickWorld {
  readonly tree: SparseBrickOctreeGPU;
  /** Narrow two-sided band used by surface and topology scheduling. */
  readonly residency: GPUFluidBrickResidency;
  /** Full wet-domain residency used only by authoritative bulk field storage. */
  readonly bulkResidency?: GPUFluidBrickResidency;
  readonly atlas?: WebGPUFluidBrickAtlas;
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
  private readonly structuralInitialPipeline: GPUComputePipeline;
  private readonly structuralFramePipeline: GPUComputePipeline;
  private readonly structuralFinalizeBindGroup: GPUBindGroup;
  private readonly proxyVoxelizer: SparseSceneProxyVoxelizer;
  private published = false;
  private proxiesPublished = false;
  private destroyed = false;

  constructor(device: GPUDevice, scene: SceneDescription, dimensions: readonly [number, number, number], options: OctreeSparseBrickWorldOptions = {}) {
    this.device = device;
    this.dimensions = dimensions;
    const brickSize = options.brickSize ?? 8;
    const environmentCatalog = buildEnvironmentProxyCatalog(scene, scene.environment ?? "default");
    const environmentPrimitives = environmentProxyPrimitives(environmentCatalog, true);
    const sceneDomain = planSparseSceneDomain(
      scene, dimensions, brickSize,
      environmentPrimitives.map((primitive) => ({ min: primitive.aabb_m.min, max: primitive.aabb_m.max })),
      { conservativePaddingCells: 1 }
    );
    this.solverGridOriginCells = sceneDomain.solverGridOriginCells;
    const maximumBrickCoordinate = sceneDomain.coordinates.reduce((maximum, coordinate) => Math.max(maximum, coordinate.x, coordinate.y, coordinate.z), 0);
    const maximumDepth = maximumBrickCoordinate === 0 ? 0 : Math.ceil(Math.log2(maximumBrickCoordinate + 1));
    const plan = planAdaptiveSparseBrickOctree({
      brickSize,
      solverBricks: sceneDomain.solverBrickCoordinates,
      proxyBricks: sceneDomain.proxyBrickCoordinates.flat(),
      maximumDepth,
      maximumEnvironmentCoarseningPower: Math.min(ENVIRONMENT_MAXIMUM_COARSENING_POWER, maximumDepth)
    });
    this.finestLevel = plan.maximumDepth;
    const packed = packSparseBrickPlan(plan, 1);
    this.tree = new SparseBrickOctreeGPU(device, { brickSize, nodeCapacity: Math.max(1, plan.nodes.length), leafCapacity: Math.max(1, plan.leaves.length), label: "Octree unified sparse-brick world" });
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
    const brickAtlasMode = options.brickAtlas ?? "mirror";
    if (brickAtlasMode !== "off" || options.bulkResidencyOnly) {
      // Bulk velocity must remain defined throughout deep liquid, while the
      // surface path wins by visiting only a narrow two-sided band. Keep the
      // two schedulers independent so atlas authority never widens surface
      // redistance back to O(wet volume).
      this.bulkResidency = new GPUFluidBrickResidency(device, dimensions, sceneDomain.cellSize_m, {
        brickSize,
        haloCells: options.haloCells ?? 2,
        retireAfterFrames: 3,
        includeLiquidInterior: true,
        leafIndices,
        leafCapacity: this.tree.leafCapacity,
        topologyTileBricks: options.topologyTileBricks ?? 1,
      });
      if (brickAtlasMode !== "off") {
        this.atlas = new WebGPUFluidBrickAtlas(device, dimensions, this.bulkResidency, {
          brickSize,
          mode: brickAtlasMode,
          preActivation: this.preActivation,
        });
      }
    }

    const counts = storageBuffer(device, "Sparse brick source counts", packed.counts.byteLength, packed.counts);
    const topology = storageBuffer(device, "Sparse brick source topology", packed.topology.byteLength, packed.topology);
    const geometry = storageBuffer(device, "Sparse brick source geometry", this.tree.voxelCapacity * 16);
    const velocity = storageBuffer(device, "Sparse brick source velocity", this.tree.voxelCapacity * 16);
    const materialOwners = storageBuffer(device, "Sparse brick source material owners", this.tree.voxelCapacity * 4);
    this.sourceBuffers = [counts, topology, geometry, velocity, materialOwners];
    this.source = { counts, topology, geometry, velocity, materialOwners, capacities: { nodes: plan.nodes.length, leaves: plan.leaves.length, voxels: this.tree.voxelCapacity } };

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
      pbrMaterials.packedRecords.byteLength,
      pbrMaterials.packedRecords,
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
    this.structuralPublicationState = storageBuffer(
      device,
      "Sparse voxel structural publication state",
      SPARSE_VOXEL_PUBLICATION_STATE.strideBytes,
    );
    const structuralModule = device.createShaderModule({
      label: "Sparse voxel structural publication finalizer",
      code: structuralPublicationFinalizeShader,
    });
    const structuralLayout = device.createBindGroupLayout({
      label: "Sparse voxel structural publication finalizer layout",
      entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }],
    });
    const structuralPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [structuralLayout] });
    this.structuralInitialPipeline = device.createComputePipeline({
      label: "Finalize initial sparse voxel structural publication",
      layout: structuralPipelineLayout,
      compute: { module: structuralModule, entryPoint: "finalizeInitial" },
    });
    this.structuralFramePipeline = device.createComputePipeline({
      label: "Finalize sparse voxel structural frame",
      layout: structuralPipelineLayout,
      compute: { module: structuralModule, entryPoint: "finalizeFrame" },
    });
    this.structuralFinalizeBindGroup = device.createBindGroup({
      label: "Sparse voxel structural publication finalizer bindings",
      layout: structuralLayout,
      entries: [{ binding: 0, resource: { buffer: this.structuralPublicationState } }],
    });
    const proxyPrimitives: SparseScenePrimitive[] = environmentPrimitives.map((primitive) => {
      const identity = {
        center: [primitive.center_m.x, primitive.center_m.y, primitive.center_m.z] as const,
        materialId: ENVIRONMENT_VOXEL_MATERIAL_BASE + primitive.ownerIndex,
        ownerId: scene.rigidBodies.length + primitive.ownerIndex
      };
      if (primitive.kind === "box") return { ...identity, kind: "box", halfExtents: [primitive.halfSize_m.x, primitive.halfSize_m.y, primitive.halfSize_m.z] };
      if (primitive.kind === "cylinder") return { ...identity, kind: "cylinder", radius: primitive.radius_m, halfHeight: primitive.halfHeight_m };
      return { ...identity, kind: "ellipsoid", radii: [primitive.radius_m.x, primitive.radius_m.y, primitive.radius_m.z] };
    });
    this.proxyVoxelizer = new SparseSceneProxyVoxelizer(device, this.tree, proxyPrimitives, {
      cellSize: this.cellSize,
      worldOrigin: [sceneDomain.worldOrigin_m.x, sceneDomain.worldOrigin_m.y, sceneDomain.worldOrigin_m.z],
      finestLevel: plan.maximumDepth,
      label: `${environmentCatalog.environmentId} environment proxies`
    });
    const publicationBinding = { buffer: this.structuralPublicationState };
    const publicationWord = (word: number) => ({ binding: publicationBinding, word });
    const residencyLayout = sparseVoxelFluidResidencyLayout(this.residency.capacity);
    if (residencyLayout.worklistByteLength !== this.residency.worklistByteLength) {
      throw new Error("Sparse voxel residency ABI does not match the producer worklist allocation");
    }
    const residencyWorklistBinding = { buffer: this.residency.worklist, size: this.residency.worklistByteLength };
    const residencyWord = (word: number) => ({ binding: residencyWorklistBinding, word });
    const activeResidencyList = {
      count: residencyWord(SPARSE_VOXEL_FLUID_RESIDENCY_WORKLIST_WORDS.activeCount),
      entryOffsetBytes: residencyLayout.activeEntryOffsetBytes,
      entryStrideBytes: residencyLayout.entryStrideBytes,
      capacity: this.residency.capacity,
    };
    const structural: SparseVoxelStructuralRenderSource = {
      control: { buffer: this.tree.control, size: SPARSE_BRICK_GPU_LAYOUT.controlStrideBytes },
      nodes: { buffer: this.tree.nodes, offset: this.tree.nodeOffsetBytes, size: this.tree.nodeCapacity * SPARSE_BRICK_GPU_LAYOUT.nodeStrideBytes },
      leaves: { buffer: this.tree.leaves, offset: this.tree.leafOffsetBytes, size: this.tree.leafCapacity * SPARSE_BRICK_GPU_LAYOUT.leafStrideBytes },
      geometry: { buffer: this.tree.geometry, offset: this.tree.geometryOffsetBytes, size: this.tree.voxelCapacity * SPARSE_BRICK_GPU_LAYOUT.geometryStrideBytes },
      velocity: { buffer: this.tree.velocity, offset: this.tree.velocityOffsetBytes, size: this.tree.voxelCapacity * SPARSE_BRICK_GPU_LAYOUT.velocityStrideBytes },
      materialOwners: { buffer: this.tree.materialOwners, offset: this.tree.materialOwnerOffsetBytes, size: this.tree.voxelCapacity * SPARSE_BRICK_GPU_LAYOUT.materialOwnerStrideBytes },
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
          staticGeometry: publicationWord(SPARSE_VOXEL_PUBLICATION_STATE.staticGeometryRevision),
          dynamicSolid: publicationWord(SPARSE_VOXEL_PUBLICATION_STATE.dynamicSolidRevision),
          coarseFluid: publicationWord(SPARSE_VOXEL_PUBLICATION_STATE.coarseFluidRevision),
          fineFluid: publicationWord(SPARSE_VOXEL_PUBLICATION_STATE.fineFluidRevision),
        },
      },
      fields: {
        topology: { bit: SPARSE_VOXEL_VALID_FIELDS.topology, residency: "all-published-leaves" },
        staticGeometry: { bit: SPARSE_VOXEL_VALID_FIELDS.staticGeometry, signedDistance: "negative-inside-metres", distanceQuality: "mixed-exact-approximate", residency: "all-published-leaves" },
        dynamicSolid: { bit: SPARSE_VOXEL_VALID_FIELDS.dynamicSolid, signedDistance: "negative-inside-metres", distanceQuality: "occupancy-estimate", residency: "fluid-resident-leaves" },
        coarseFluid: { bit: SPARSE_VOXEL_VALID_FIELDS.coarseFluid, signedDistance: "negative-inside-metres", distanceQuality: "metric-near-interface", residency: "fluid-resident-leaves" },
        fineFluid: { bit: SPARSE_VOXEL_VALID_FIELDS.fineFluid, signedDistance: "negative-inside-metres", distanceQuality: "metric", residency: "unavailable" },
        velocity: { bit: SPARSE_VOXEL_VALID_FIELDS.velocity, residency: "fluid-resident-leaves" },
        materialOwner: { bit: SPARSE_VOXEL_VALID_FIELDS.materialOwner, residency: "all-published-leaves" },
      },
    };
    this.sceneSource = {
      pbrMaterials: {
        binding: { buffer: this.pbrMaterialBuffer, size: pbrMaterials.packedRecords.byteLength },
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
      materialCount: materialData.length / 8,
      fluidBrickStats: { buffer: this.residency.worklist }, fluidBrickCapacity: this.residency.capacity,
      structural,
      revision: 1
    };
    this.baseAllocatedBytes = this.tree.allocatedBytes + this.residency.allocatedBytes
      + (this.bulkResidency?.allocatedBytes ?? 0) + (this.atlas?.allocatedBytes ?? 0)
      + this.sourceBuffers.reduce((sum, buffer) => sum + buffer.size, 0)
      + this.pbrMaterialBuffer.size + this.lightBuffer.size + this.environmentLightingBuffer.size + this.structuralPublicationState.size
      + this.proxyVoxelizer.allocatedBytes;
  }

  get allocatedBytes(): number { return this.baseAllocatedBytes + (this.inspection?.allocatedBytes ?? 0); }

  /** Allocate the expanded legacy records only when raw/grid inspection asks for them. */
  ensureInspectionSource(): SparseVoxelRenderSource {
    if (this.destroyed) throw new Error("Cannot inspect a destroyed sparse-brick world");
    if (this.inspection) return this.inspection.source;
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
    const voxelPipeline = this.device.createComputePipeline({ label: "Materialize sparse voxel records", layout: debugPipelineLayout, compute: { module: shaderModule, entryPoint: "materializeVoxels" } });
    const brickPipeline = this.device.createComputePipeline({ label: "Materialize sparse brick records", layout: debugPipelineLayout, compute: { module: shaderModule, entryPoint: "materializeBricks" } });
    const bindGroup = this.device.createBindGroup({ layout: debugLayout, entries: [
      { binding: 0, resource: { buffer: this.tree.control } },
      { binding: 1, resource: { buffer: this.tree.nodes, offset: this.tree.nodeOffsetBytes } },
      { binding: 2, resource: { buffer: this.tree.leaves, offset: this.tree.leafOffsetBytes } },
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
    const buffers = [voxelRecords, brickRecords, voxelCount, brickCount, materialBuffer, bodyMaterialBuffer, params];
    this.inspection = {
      source, publication, buffers, voxelRecords, brickRecords, voxelCount, brickCount,
      voxelPipeline, brickPipeline, bindGroup,
      allocatedBytes: buffers.reduce((sum, buffer) => sum + buffer.size, 0),
    };
    return source;
  }

  encode(encoder: GPUCommandEncoder, fields: OctreeSparseBrickDenseFields, timings: OctreeSparseBrickTimestampWrites = {}, dt_s = 0, bulkAlreadyRefreshed = false): void {
    if (this.destroyed) return;
    const beginRange = (label: string, writes?: GPUComputePassTimestampWrites) => {
      if (writes?.beginningOfPassWriteIndex === undefined) return;
      const marker = encoder.beginComputePass({ label: `${label} start`, timestampWrites: {
        querySet: writes.querySet, endOfPassWriteIndex: writes.beginningOfPassWriteIndex
      } });
      marker.end();
    };
    const endRange = (label: string, writes?: GPUComputePassTimestampWrites) => {
      if (writes?.endOfPassWriteIndex === undefined) return;
      const marker = encoder.beginComputePass({ label: `${label} end`, timestampWrites: {
        querySet: writes.querySet, beginningOfPassWriteIndex: writes.endOfPassWriteIndex
      } });
      marker.end();
    };
    beginRange("Fluid brick residency", timings.residency);
    this.residency.encode(encoder, fields.levelSet, fields.velocity, { dt_s, preActivation: this.preActivation });
    // Bulk residency normally refreshes at the head of every solver substep.
    // Keep a tail refresh for t=0 publication and non-solver callers, while
    // the solver explicitly suppresses the duplicate publication pass.
    if (!bulkAlreadyRefreshed) {
      if (this.atlas) this.atlas.encodeBulkRefresh(encoder, fields.levelSet, fields.velocity, dt_s);
      else this.bulkResidency?.encode(encoder, fields.levelSet, fields.velocity, { dt_s, preActivation: this.preActivation });
    }
    endRange("Fluid brick residency", timings.residency);
    beginRange("Sparse brick publication", timings.publication);
    const inspection = this.inspection;
    const encodePlan = planOctreeSparseBrickEncode(inspection?.publication.enabled ?? false);
    const initialPublication = !this.published;
    if (initialPublication) {
      this.tree.encodePublish(encoder, this.source);
      this.published = true;
    }
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
      preservedMaterialIdMinimum: ENVIRONMENT_VOXEL_MATERIAL_BASE,
      activeBrickWorklist: this.residency.worklist,
    });
    if (!this.proxiesPublished) {
      this.proxyVoxelizer.encode(encoder);
      this.proxiesPublished = true;
    }
    if (encodePlan.inspectionPublication) {
      this.encodeInspectionPublication(encoder);
      inspection?.publication.markEncoded();
    }
    // Atlas pages were mirrored with the bulk-residency refresh above (or at
    // the head of the final solver substep).
    const finalizer = encoder.beginComputePass({ label: "Finalize sparse voxel structural publication" });
    finalizer.setPipeline(initialPublication ? this.structuralInitialPipeline : this.structuralFramePipeline);
    finalizer.setBindGroup(0, this.structuralFinalizeBindGroup);
    finalizer.dispatchWorkgroups(1);
    finalizer.end();
    endRange("Sparse brick publication", timings.publication);
  }

  private encodeInspectionPublication(encoder: GPUCommandEncoder): void {
    const inspection = this.inspection;
    if (!inspection) return;
    encoder.copyBufferToBuffer(this.tree.control, 8, inspection.voxelCount, 0, 4);
    encoder.copyBufferToBuffer(this.tree.control, 4, inspection.brickCount, 0, 4);
    const voxelPass = encoder.beginComputePass({ label: "Publish octree raw voxel records" });
    voxelPass.setPipeline(inspection.voxelPipeline); voxelPass.setBindGroup(0, inspection.bindGroup);
    voxelPass.dispatchWorkgroups(...tiledDebugDispatch(this.tree.voxelCapacity, 256)); voxelPass.end();
    const brickPass = encoder.beginComputePass({ label: "Publish octree sparse brick records" });
    brickPass.setPipeline(inspection.brickPipeline); brickPass.setBindGroup(0, inspection.bindGroup);
    brickPass.dispatchWorkgroups(...tiledDebugDispatch(this.tree.leafCapacity, 64)); brickPass.end();
  }

  readResidencyStats(): Promise<FluidBrickResidencyStats> { return this.residency.readStats(); }

  readBulkResidencyStats(): Promise<FluidBrickResidencyStats> | undefined { return this.bulkResidency?.readStats(); }

  get atlasSamplingSource(): FluidBrickAtlasSamplingSource | undefined { return this.atlas?.getSamplingSource(); }

  /** Full wet-domain worklist, independent of optional atlas payload storage. */
  get bulkResidencyWorklist(): GPUBuffer | undefined { return this.bulkResidency?.worklist; }

  /** Persistent wet-domain topology scheduler on compact authority. */
  get topologyResidency(): GPUFluidBrickResidency { return this.bulkResidency ?? this.residency; }

  readAtlasStats(): Promise<FluidBrickAtlasStats> | undefined { return this.atlas?.readStats(); }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.tree.destroy();
    this.residency.destroy();
    this.bulkResidency?.destroy();
    this.atlas?.destroy();
    this.proxyVoxelizer.destroy();
    for (const buffer of [...this.sourceBuffers, this.pbrMaterialBuffer, this.lightBuffer, this.environmentLightingBuffer, this.structuralPublicationState, ...(this.inspection?.buffers ?? [])]) buffer.destroy();
  }
}

export const octreeSparseBrickDebugPublicationShader = debugPublicationShader;
export const octreeSparseBrickStructuralFinalizeShader = structuralPublicationFinalizeShader;
