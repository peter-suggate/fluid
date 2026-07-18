import type { SceneDescription } from "./model";
import {
  SparseBrickOctreeGPU,
  packSparseBrickPlan,
  type SparseBrickCoordinate,
  type SparseBrickPublicationSource,
  type SparseBrickSize
} from "./sparse-brick-octree";
import { planAdaptiveSparseBrickOctree } from "./adaptive-sparse-brick-plan";
import { planSparseSceneDomain } from "./sparse-scene-domain";
import { VOXEL_MATERIAL_IDS, materialIdForRigidShape, packVoxelDebugMaterialTable } from "./voxel-scene";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives } from "./voxel-environments";
import { SparseSceneProxyVoxelizer, type SparseScenePrimitive } from "./webgpu-sparse-scene-proxies";
import { SPARSE_VOXEL_DEBUG_RECORD_STRIDE, type SparseVoxelRenderSource } from "./webgpu-voxel-debug";
import { GPUFluidBrickResidency, type FluidBrickResidencyStats } from "./webgpu-fluid-brick-residency";
import { WebGPUFluidBrickAtlas, type FluidBrickAtlasStats } from "./webgpu-brick-atlas";

export interface OctreeSparseBrickWorldOptions {
  brickSize?: SparseBrickSize;
  /** Air-side support retained for pressure-topology rebuilds. */
  haloCells?: number;
  /** Brick-pooled phi/velocity atlas mirrored from the dense fields. */
  brickAtlas?: boolean;
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

function storageBuffer(device: GPUDevice, label: string, size: number, data?: ArrayBufferView<ArrayBuffer>) {
  const buffer = device.createBuffer({ label, size: Math.max(4, size), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  if (data && data.byteLength > 0) device.queue.writeBuffer(buffer, 0, data);
  return buffer;
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
  let packed = materialOwners[index];
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
fn materializeVoxels(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= control[2] || gid.x >= arrayLength(&voxelRecords)) { return; }
  voxelRecords[gid.x] = recordForVoxel(gid.x);
}

@compute @workgroup_size(64)
fn materializeBricks(@builtin(global_invocation_id) gid: vec3u) {
  let leafIndex = gid.x;
  if (leafIndex >= control[1] || leafIndex >= arrayLength(&brickRecords)) { return; }
  let brickSize = params.settings.x;
  let voxelsPerBrick = brickSize * brickSize * brickSize;
  var isActive = false;
  var material = 0u;
  var owner = 0xffffu;
  for (var local = 0u; local < voxelsPerBrick; local += 1u) {
    let packed = materialOwners[leafIndex * voxelsPerBrick + local];
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

/**
 * Transitional GPU bridge: the octree solver remains authoritative while its
 * resident level set, velocity and VOS solid field are published into one
 * sparse-brick ABI for scene inspection and subsequent sparse kernels.
 */
export class OctreeSparseBrickWorld {
  readonly tree: SparseBrickOctreeGPU;
  readonly residency: GPUFluidBrickResidency;
  readonly atlas?: WebGPUFluidBrickAtlas;
  readonly renderSource: SparseVoxelRenderSource;
  readonly allocatedBytes: number;
  private readonly preActivation: boolean;

  private readonly device: GPUDevice;
  private readonly dimensions: readonly [number, number, number];
  private readonly solverGridOriginCells: readonly [number, number, number];
  private readonly finestLevel: number;
  private readonly cellSize: readonly [number, number, number];
  private readonly containerClosedTop: boolean;
  private readonly source: SparseBrickPublicationSource;
  private readonly sourceBuffers: GPUBuffer[];
  private readonly voxelRecords: GPUBuffer;
  private readonly brickRecords: GPUBuffer;
  private readonly voxelCount: GPUBuffer;
  private readonly brickCount: GPUBuffer;
  private readonly materialBuffer: GPUBuffer;
  private readonly bodyMaterialBuffer: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly voxelPipeline: GPUComputePipeline;
  private readonly brickPipeline: GPUComputePipeline;
  private readonly debugBindGroup: GPUBindGroup;
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
    if (options.brickAtlas ?? true) this.atlas = new WebGPUFluidBrickAtlas(device, dimensions, this.residency, { brickSize });

    const counts = storageBuffer(device, "Sparse brick source counts", packed.counts.byteLength, packed.counts);
    const topology = storageBuffer(device, "Sparse brick source topology", packed.topology.byteLength, packed.topology);
    const geometry = storageBuffer(device, "Sparse brick source geometry", this.tree.voxelCapacity * 16);
    const velocity = storageBuffer(device, "Sparse brick source velocity", this.tree.voxelCapacity * 16);
    const materialOwners = storageBuffer(device, "Sparse brick source material owners", this.tree.voxelCapacity * 4);
    this.sourceBuffers = [counts, topology, geometry, velocity, materialOwners];
    this.source = { counts, topology, geometry, velocity, materialOwners, capacities: { nodes: plan.nodes.length, leaves: plan.leaves.length, voxels: this.tree.voxelCapacity } };

    this.voxelRecords = storageBuffer(device, "Sparse voxel debug records", this.tree.voxelCapacity * SPARSE_VOXEL_DEBUG_RECORD_STRIDE);
    this.brickRecords = storageBuffer(device, "Sparse brick debug records", this.tree.leafCapacity * SPARSE_VOXEL_DEBUG_RECORD_STRIDE);
    this.voxelCount = storageBuffer(device, "Sparse voxel render count", 4);
    this.brickCount = storageBuffer(device, "Sparse brick render count", 4);
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
    this.materialBuffer = storageBuffer(device, "Sparse voxel material table", materialData.byteLength, new Float32Array(materialData));
    const bodyMaterials = new Uint32Array(Math.max(1, scene.rigidBodies.length));
    scene.rigidBodies.forEach((body, index) => { bodyMaterials[index] = materialIdForRigidShape(body.shape); });
    this.bodyMaterialBuffer = storageBuffer(device, "Sparse voxel body material IDs", bodyMaterials.byteLength, bodyMaterials);
    this.params = device.createBuffer({ label: "Sparse voxel debug publication parameters", size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const c = scene.container;
    this.containerClosedTop = c.top === "closed";
    this.cellSize = sceneDomain.cellSize_m;
    const parameterData = new ArrayBuffer(64), uints = new Uint32Array(parameterData), floats = new Float32Array(parameterData);
    uints.set([sceneDomain.sceneDimensionsCells[0], sceneDomain.sceneDimensionsCells[1], sceneDomain.sceneDimensionsCells[2], 0], 0);
    floats.set([sceneDomain.worldOrigin_m.x, sceneDomain.worldOrigin_m.y, sceneDomain.worldOrigin_m.z, 0], 4);
    floats.set([...this.cellSize, 0], 8);
    uints.set([brickSize, scene.rigidBodies.length, plan.maximumDepth, VOXEL_MATERIAL_IDS.fluid], 12);
    device.queue.writeBuffer(this.params, 0, parameterData);

    const shaderModule = device.createShaderModule({ label: "Octree sparse-brick render publication", code: debugPublicationShader });
    const debugLayout = device.createBindGroupLayout({ label: "Octree sparse-brick render publication layout", entries: [
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
    const debugPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [debugLayout] });
    this.voxelPipeline = device.createComputePipeline({ label: "Materialize sparse voxel records", layout: debugPipelineLayout, compute: { module: shaderModule, entryPoint: "materializeVoxels" } });
    this.brickPipeline = device.createComputePipeline({ label: "Materialize sparse brick records", layout: debugPipelineLayout, compute: { module: shaderModule, entryPoint: "materializeBricks" } });
    this.debugBindGroup = device.createBindGroup({ layout: debugLayout, entries: [
      { binding: 0, resource: { buffer: this.tree.control } },
      { binding: 1, resource: { buffer: this.tree.nodes, offset: this.tree.nodeOffsetBytes } },
      { binding: 2, resource: { buffer: this.tree.leaves, offset: this.tree.leafOffsetBytes } },
      { binding: 3, resource: { buffer: this.tree.materialOwners, offset: this.tree.materialOwnerOffsetBytes } },
      { binding: 4, resource: { buffer: this.bodyMaterialBuffer } }, { binding: 5, resource: { buffer: this.params } },
      { binding: 6, resource: { buffer: this.voxelRecords } }, { binding: 7, resource: { buffer: this.brickRecords } },
      { binding: 8, resource: { buffer: this.residency.leafStates } }
    ] });
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
    this.renderSource = {
      voxelRecords: { buffer: this.voxelRecords }, voxelCount: { buffer: this.voxelCount },
      brickRecords: { buffer: this.brickRecords }, brickCount: { buffer: this.brickCount },
      materials: { buffer: this.materialBuffer }, voxelCapacity: this.tree.voxelCapacity, brickCapacity: this.tree.leafCapacity,
      materialCount: materialData.length / 8,
      fluidBrickStats: { buffer: this.residency.worklist }, fluidBrickCapacity: this.residency.capacity,
      revision: 1
    };
    this.allocatedBytes = this.tree.allocatedBytes + this.residency.allocatedBytes + (this.atlas?.allocatedBytes ?? 0)
      + this.tree.voxelCapacity * SPARSE_VOXEL_DEBUG_RECORD_STRIDE + this.tree.leafCapacity * SPARSE_VOXEL_DEBUG_RECORD_STRIDE
      + this.sourceBuffers.reduce((sum, buffer) => sum + buffer.size, 0) + this.voxelCount.size + this.brickCount.size
      + this.materialBuffer.size + this.bodyMaterialBuffer.size + this.params.size + this.proxyVoxelizer.allocatedBytes;
  }

  encode(encoder: GPUCommandEncoder, fields: OctreeSparseBrickDenseFields, timings: OctreeSparseBrickTimestampWrites = {}, dt_s = 0): void {
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
    endRange("Fluid brick residency", timings.residency);
    beginRange("Sparse brick publication", timings.publication);
    if (!this.published) {
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
    encoder.copyBufferToBuffer(this.tree.control, 8, this.voxelCount, 0, 4);
    encoder.copyBufferToBuffer(this.tree.control, 4, this.brickCount, 0, 4);
    const voxelPass = encoder.beginComputePass({ label: "Publish octree raw voxel records" });
    voxelPass.setPipeline(this.voxelPipeline); voxelPass.setBindGroup(0, this.debugBindGroup);
    voxelPass.dispatchWorkgroups(Math.ceil(this.tree.voxelCapacity / 256)); voxelPass.end();
    const brickPass = encoder.beginComputePass({ label: "Publish octree sparse brick records" });
    brickPass.setPipeline(this.brickPipeline); brickPass.setBindGroup(0, this.debugBindGroup);
    brickPass.dispatchWorkgroups(Math.ceil(this.tree.leafCapacity / 64)); brickPass.end();
    // Atlas tiles follow the freshly classified residency states within the
    // same publication window: retire freed slots, allocate newly resident
    // ones, then mirror the dense fields (apron included) and validate.
    this.atlas?.encode(encoder, fields.levelSet, fields.velocity);
    endRange("Sparse brick publication", timings.publication);
  }

  readResidencyStats(): Promise<FluidBrickResidencyStats> { return this.residency.readStats(); }

  readAtlasStats(): Promise<FluidBrickAtlasStats> | undefined { return this.atlas?.readStats(); }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.tree.destroy();
    this.residency.destroy();
    this.atlas?.destroy();
    this.proxyVoxelizer.destroy();
    for (const buffer of [...this.sourceBuffers, this.voxelRecords, this.brickRecords, this.voxelCount, this.brickCount, this.materialBuffer, this.bodyMaterialBuffer, this.params]) buffer.destroy();
  }
}

export const octreeSparseBrickDebugPublicationShader = debugPublicationShader;
