/**
 * Sparse-brick octree primitives shared by scene, rendering, and fluid work.
 *
 * The CPU helpers in this file only produce deterministic addresses and testable
 * topology plans. Runtime publication is device-to-device: counts can be
 * authored by an earlier GPU pass and no map/readback is needed to publish the
 * tree or size later compute/draw work.
 */

export type SparseBrickSize = 4 | 8;

export interface SparseBrickCoordinate {
  x: number;
  y: number;
  z: number;
}

export interface SparseBrickNodePlan {
  index: number;
  level: number;
  morton: bigint;
  coordinate: SparseBrickCoordinate;
  childMask: number;
  firstChild: number;
  childCount: number;
  leafIndex: number;
}

export interface SparseBrickLeafPlan {
  index: number;
  nodeIndex: number;
  morton: bigint;
  coordinate: SparseBrickCoordinate;
  voxelOffset: number;
}

export interface SparseBrickPlan {
  brickSize: SparseBrickSize;
  maximumDepth: number;
  /** Global node offsets for levels 0..maximumDepth, followed by the end. */
  levelOffsets: readonly number[];
  nodes: readonly SparseBrickNodePlan[];
  leaves: readonly SparseBrickLeafPlan[];
  voxelCount: number;
}

export interface SparseBrickPlanOptions {
  brickSize: SparseBrickSize;
  /** If omitted, the smallest depth containing every non-negative coordinate is used. */
  maximumDepth?: number;
}

export const SPARSE_BRICK_INVALID_INDEX = 0xffffffff;
export const SPARSE_BRICK_NO_OWNER = 0xffff;
export const SPARSE_BRICK_MAX_MORTON_BITS = 21;

export const SPARSE_BRICK_GPU_LAYOUT = Object.freeze({
  nodeStrideBytes: 32,
  leafStrideBytes: 16,
  geometryStrideBytes: 16,
  velocityStrideBytes: 16,
  materialOwnerStrideBytes: 4,
  controlStrideBytes: 128,
  /** geometry = fluid SDF, solid SDF estimate, solid fraction, pressure */
  geometryChannels: ["fluidSignedDistance", "solidSignedDistance", "solidFraction", "pressure"] as const,
  /** velocity = world velocity xyz, reconstructed liquid volume fraction */
  velocityChannels: ["velocityX", "velocityY", "velocityZ", "liquidFraction"] as const,
  controlWords: {
    publishedNodes: 0,
    publishedLeaves: 1,
    publishedVoxels: 2,
    generation: 3,
    requestedNodes: 4,
    requestedLeaves: 5,
    requestedVoxels: 6,
    requestedGeneration: 7,
    nodeCapacity: 8,
    leafCapacity: 9,
    voxelCapacity: 10,
    brickSize: 11,
    overflowFlags: 12,
    droppedNodes: 13,
    droppedLeaves: 14,
    droppedVoxels: 15,
    leafWordOffset: 16,
    velocityWordOffset: 17,
    materialOwnerWordOffset: 18,
  } as const,
  dispatchIndirectOffsetBytes: 80,
  drawIndirectOffsetBytes: 96,
});

function integerCoordinate(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 0 || value >= 2 ** SPARSE_BRICK_MAX_MORTON_BITS) {
    throw new RangeError(`${name} must be a non-negative integer below 2^${SPARSE_BRICK_MAX_MORTON_BITS}`);
  }
  return value;
}

/** Interleave 21 bits from x/y/z into one deterministic 63-bit address. */
export function mortonEncode3D(x: number, y: number, z: number): bigint {
  const values = [integerCoordinate(x, "x"), integerCoordinate(y, "y"), integerCoordinate(z, "z")];
  let key = 0n;
  for (let bit = 0; bit < SPARSE_BRICK_MAX_MORTON_BITS; bit += 1) {
    for (let axis = 0; axis < 3; axis += 1) {
      if (Math.floor(values[axis] / 2 ** bit) % 2 !== 0) key |= 1n << BigInt(3 * bit + axis);
    }
  }
  return key;
}

export function mortonDecode3D(key: bigint): SparseBrickCoordinate {
  if (key < 0n || key >= 1n << 63n) throw new RangeError("Morton key must be an unsigned 63-bit integer");
  const values = [0, 0, 0];
  for (let bit = 0; bit < SPARSE_BRICK_MAX_MORTON_BITS; bit += 1) {
    for (let axis = 0; axis < 3; axis += 1) {
      if ((key & (1n << BigInt(3 * bit + axis))) !== 0n) values[axis] += 2 ** bit;
    }
  }
  return { x: values[0], y: values[1], z: values[2] };
}

/** Append one xyz child octant (x | y<<1 | z<<2) to a prefix. */
export function mortonChild(parent: bigint, childOctant: number): bigint {
  if (parent < 0n || parent >= 1n << 60n) throw new RangeError("Parent Morton prefix is too deep");
  if (!Number.isInteger(childOctant) || childOctant < 0 || childOctant > 7) throw new RangeError("Child octant must be 0..7");
  return (parent << 3n) | BigInt(childOctant);
}

export function mortonParent(child: bigint): bigint {
  if (child < 0n || child >= 1n << 63n) throw new RangeError("Morton key must be an unsigned 63-bit integer");
  return child >> 3n;
}

export function packMaterialOwner(materialId: number, ownerId: number = SPARSE_BRICK_NO_OWNER): number {
  if (!Number.isInteger(materialId) || materialId < 0 || materialId > 0xffff) throw new RangeError("Material ID must fit uint16");
  if (!Number.isInteger(ownerId) || ownerId < 0 || ownerId > 0xffff) throw new RangeError("Owner ID must fit uint16");
  return ((ownerId << 16) | materialId) >>> 0;
}

export function unpackMaterialOwner(value: number): { materialId: number; ownerId: number } {
  const word = value >>> 0;
  return { materialId: word & 0xffff, ownerId: word >>> 16 };
}

function depthForCoordinates(coordinates: readonly SparseBrickCoordinate[]): number {
  let maximum = 0;
  for (const coordinate of coordinates) maximum = Math.max(maximum, coordinate.x, coordinate.y, coordinate.z);
  return maximum === 0 ? 0 : Math.ceil(Math.log2(maximum + 1));
}

function popcount8(value: number): number {
  let count = 0;
  for (let word = value & 0xff; word !== 0; word >>>= 1) count += word & 1;
  return count;
}

/**
 * Build a canonical level-major, Morton-sorted pointerless topology.
 * Input coordinates address finest-level bricks, not individual voxels.
 */
export function planSparseBrickOctree(
  input: readonly SparseBrickCoordinate[],
  options: SparseBrickPlanOptions,
): SparseBrickPlan {
  if (options.brickSize !== 4 && options.brickSize !== 8) throw new RangeError("Sparse brick size must be 4 or 8");
  const unique = new Map<bigint, SparseBrickCoordinate>();
  for (const value of input) {
    const coordinate = {
      x: integerCoordinate(value.x, "brick x"),
      y: integerCoordinate(value.y, "brick y"),
      z: integerCoordinate(value.z, "brick z"),
    };
    unique.set(mortonEncode3D(coordinate.x, coordinate.y, coordinate.z), coordinate);
  }
  const coordinates = [...unique.values()];
  const requiredDepth = depthForCoordinates(coordinates);
  const maximumDepth = options.maximumDepth ?? requiredDepth;
  if (!Number.isInteger(maximumDepth) || maximumDepth < 0 || maximumDepth > SPARSE_BRICK_MAX_MORTON_BITS) {
    throw new RangeError(`Maximum depth must be 0..${SPARSE_BRICK_MAX_MORTON_BITS}`);
  }
  if (maximumDepth < requiredDepth) throw new RangeError("Maximum depth cannot contain all brick coordinates");

  const levelCoordinates: SparseBrickCoordinate[][] = [];
  const levelOffsets: number[] = [];
  const nodes: SparseBrickNodePlan[] = [];
  const indexByLevelAndMorton = new Map<string, number>();
  for (let level = 0; level <= maximumDepth; level += 1) {
    levelOffsets.push(nodes.length);
    const divisor = 2 ** (maximumDepth - level);
    const levelMap = new Map<bigint, SparseBrickCoordinate>();
    for (const coordinate of coordinates) {
      const ancestor = {
        x: Math.floor(coordinate.x / divisor),
        y: Math.floor(coordinate.y / divisor),
        z: Math.floor(coordinate.z / divisor),
      };
      levelMap.set(mortonEncode3D(ancestor.x, ancestor.y, ancestor.z), ancestor);
    }
    const sorted = [...levelMap.entries()].sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0);
    levelCoordinates.push(sorted.map(([, coordinate]) => coordinate));
    for (const [morton, coordinate] of sorted) {
      const index = nodes.length;
      indexByLevelAndMorton.set(`${level}:${morton}`, index);
      nodes.push({
        index, level, morton, coordinate,
        childMask: 0, firstChild: SPARSE_BRICK_INVALID_INDEX, childCount: 0,
        leafIndex: SPARSE_BRICK_INVALID_INDEX,
      });
    }
  }
  levelOffsets.push(nodes.length);

  for (let level = 0; level < maximumDepth; level += 1) {
    for (let localIndex = 0; localIndex < levelCoordinates[level].length; localIndex += 1) {
      const node = nodes[levelOffsets[level] + localIndex];
      let firstChild = SPARSE_BRICK_INVALID_INDEX;
      let mask = 0;
      for (let childOctant = 0; childOctant < 8; childOctant += 1) {
        const childKey = mortonChild(node.morton, childOctant);
        const childIndex = indexByLevelAndMorton.get(`${level + 1}:${childKey}`);
        if (childIndex === undefined) continue;
        firstChild = Math.min(firstChild, childIndex);
        mask |= 1 << childOctant;
      }
      node.childMask = mask;
      node.childCount = popcount8(mask);
      node.firstChild = firstChild;
    }
  }

  const voxelCountPerBrick = options.brickSize ** 3;
  const leaves: SparseBrickLeafPlan[] = [];
  if (coordinates.length > 0) {
    for (let nodeIndex = levelOffsets[maximumDepth]; nodeIndex < levelOffsets[maximumDepth + 1]; nodeIndex += 1) {
      const node = nodes[nodeIndex];
      const index = leaves.length;
      node.leafIndex = index;
      leaves.push({ index, nodeIndex, morton: node.morton, coordinate: node.coordinate, voxelOffset: index * voxelCountPerBrick });
    }
  }
  return {
    brickSize: options.brickSize,
    maximumDepth,
    levelOffsets,
    nodes,
    leaves,
    voxelCount: leaves.length * voxelCountPerBrick,
  };
}

export interface PackedSparseBrickPlan {
  /** Eight u32 words per node: key lo/hi, level, child mask, first child, child count, leaf, flags. */
  nodes: Uint32Array<ArrayBuffer>;
  /** Four u32 words per leaf: node, voxel offset, key lo/hi. */
  leaves: Uint32Array<ArrayBuffer>;
  /** Nodes followed immediately by leaves; preferred by the portable eight-storage-binding publication path. */
  topology: Uint32Array<ArrayBuffer>;
  counts: Uint32Array<ArrayBuffer>;
}

function splitMorton(key: bigint): [number, number] {
  return [Number(key & 0xffffffffn) >>> 0, Number((key >> 32n) & 0xffffffffn) >>> 0];
}

export function packSparseBrickPlan(plan: SparseBrickPlan, generation = 0): PackedSparseBrickPlan {
  const nodeWords = new Uint32Array(plan.nodes.length * 8);
  for (const node of plan.nodes) {
    const [low, high] = splitMorton(node.morton);
    nodeWords.set([
      low, high, node.level, node.childMask, node.firstChild, node.childCount, node.leafIndex, 0,
    ], node.index * 8);
  }
  const leafWords = new Uint32Array(plan.leaves.length * 4);
  for (const leaf of plan.leaves) {
    const [low, high] = splitMorton(leaf.morton);
    leafWords.set([leaf.nodeIndex, leaf.voxelOffset, low, high], leaf.index * 4);
  }
  const topology = new Uint32Array(nodeWords.length + leafWords.length);
  topology.set(nodeWords); topology.set(leafWords, nodeWords.length);
  return {
    nodes: nodeWords, leaves: leafWords, topology,
    counts: new Uint32Array([plan.nodes.length, plan.leaves.length, plan.voxelCount, generation >>> 0, 0, nodeWords.length]),
  };
}

export interface SparseBrickPublicationSource {
  /** First four u32 words are node, leaf, voxel, and generation counts. */
  counts: GPUBuffer;
  /** Raw u32 topology arena. Count words 4 and 5 give source node/leaf word offsets. */
  topology: GPUBuffer;
  geometry: GPUBuffer;
  velocity: GPUBuffer;
  materialOwners: GPUBuffer;
  /** Allocated source bounds used only to size a conservative GPU dispatch. */
  capacities: { nodes: number; leaves: number; voxels: number };
}

export interface SparseBrickDenseFieldSource {
  levelSet: GPUTextureView;
  velocity: GPUTextureView;
  /** Dense array of `{ fraction: f32, owner: i32 }`, x-major then y then z. */
  solidCells: GPUBuffer;
  dimensions: readonly [number, number, number];
  gridOriginCells?: readonly [number, number, number];
  /** Maximum topology level. Omit for legacy fixed-level brick plans. */
  finestLevel?: number;
  /** Preserve static scene-proxy payloads at or above this material ID. */
  preservedMaterialIdMinimum?: number;
  cellSize: readonly [number, number, number];
  fluidMaterialId: number;
  solidMaterialId: number;
  containerMaterialId?: number;
  containerClosedTop?: boolean;
}

export interface SparseBrickOctreeGPUOptions {
  brickSize: SparseBrickSize;
  nodeCapacity: number;
  leafCapacity: number;
  label?: string;
}

const publicationShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> sourceCounts: array<u32>;
@group(0) @binding(1) var<storage, read> sourceTopology: array<u32>;
@group(0) @binding(2) var<storage, read> sourceGeometry: array<vec4f>;
@group(0) @binding(3) var<storage, read> sourceVelocity: array<vec4f>;
@group(0) @binding(4) var<storage, read> sourceMaterialOwners: array<u32>;
@group(0) @binding(5) var<storage, read_write> control: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> topology: array<u32>;
@group(0) @binding(7) var<storage, read_write> payload: array<u32>;

fn linearIndex(gid: vec3u, groups: vec3u) -> u32 {
  return gid.x + gid.y * groups.x * 256u + gid.z * groups.x * groups.y * 256u;
}
@compute @workgroup_size(256)
fn publish(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let index = linearIndex(gid, groups);
  let requested = vec3u(sourceCounts[0], sourceCounts[1], sourceCounts[2]);
  let capacities = vec3u(atomicLoad(&control[8]), atomicLoad(&control[9]), atomicLoad(&control[10]));
  let overflow = requested > capacities;
  let valid = !any(overflow);
  if (index == 0u) {
    atomicStore(&control[0], select(0u, requested.x, valid));
    atomicStore(&control[1], select(0u, requested.y, valid));
    atomicStore(&control[2], select(0u, requested.z, valid));
    atomicStore(&control[3], select(0u, sourceCounts[3], valid));
    atomicStore(&control[4], requested.x); atomicStore(&control[5], requested.y);
    atomicStore(&control[6], requested.z); atomicStore(&control[7], sourceCounts[3]);
    let flags = select(0u, 1u, overflow.x) | select(0u, 2u, overflow.y) | select(0u, 4u, overflow.z);
    atomicStore(&control[12], flags);
    atomicStore(&control[13], select(0u, requested.x - capacities.x, overflow.x));
    atomicStore(&control[14], select(0u, requested.y - capacities.y, overflow.y));
    atomicStore(&control[15], select(0u, requested.z - capacities.z, overflow.z));
    let blocks = select(0u, (requested.z + 255u) / 256u, valid && requested.z > 0u);
    let dispatchX = min(blocks, 65535u);
    atomicStore(&control[20], dispatchX);
    if (dispatchX > 0u) { atomicStore(&control[21], (blocks + dispatchX - 1u) / dispatchX); }
    else { atomicStore(&control[21], 0u); }
    atomicStore(&control[22], 1u);
    atomicStore(&control[24], select(0u, 36u, valid && requested.y > 0u));
    atomicStore(&control[25], select(0u, requested.y, valid));
    atomicStore(&control[26], 0u); atomicStore(&control[27], 0u);
  }
  if (!valid) { return; }
  if (index < requested.x) {
    let sourceBase = sourceCounts[4] + index * 8u;
    let destinationBase = index * 8u;
    for (var word = 0u; word < 8u; word += 1u) { topology[destinationBase + word] = sourceTopology[sourceBase + word]; }
  }
  if (index < requested.y) {
    let sourceBase = sourceCounts[5] + index * 4u;
    let destinationBase = atomicLoad(&control[16]) + index * 4u;
    for (var word = 0u; word < 4u; word += 1u) { topology[destinationBase + word] = sourceTopology[sourceBase + word]; }
  }
  if (index < requested.z) {
    let geometryBase = index * 4u;
    let velocityBase = atomicLoad(&control[17]) + index * 4u;
    payload[geometryBase] = bitcast<u32>(sourceGeometry[index].x);
    payload[geometryBase + 1u] = bitcast<u32>(sourceGeometry[index].y);
    payload[geometryBase + 2u] = bitcast<u32>(sourceGeometry[index].z);
    payload[geometryBase + 3u] = bitcast<u32>(sourceGeometry[index].w);
    payload[velocityBase] = bitcast<u32>(sourceVelocity[index].x);
    payload[velocityBase + 1u] = bitcast<u32>(sourceVelocity[index].y);
    payload[velocityBase + 2u] = bitcast<u32>(sourceVelocity[index].z);
    payload[velocityBase + 3u] = bitcast<u32>(sourceVelocity[index].w);
    payload[atomicLoad(&control[18]) + index] = sourceMaterialOwners[index];
  }
}
`;

const denseFieldShader = /* wgsl */ `
struct SolidCell { fraction: f32, owner: i32 }
struct Params { dims: vec4u, origin: vec4i, cell: vec4f, materials: vec4u }
@group(0) @binding(0) var<storage, read> control: array<u32>;
@group(0) @binding(1) var<storage, read> topology: array<u32>;
@group(0) @binding(2) var levelSet: texture_3d<f32>;
@group(0) @binding(3) var velocityField: texture_3d<f32>;
@group(0) @binding(4) var<storage, read> solidCells: array<SolidCell>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read_write> payload: array<u32>;

fn keyBit(low: u32, high: u32, bit: u32) -> u32 {
  if (bit >= 32u) { return (high >> (bit - 32u)) & 1u; }
  return (low >> bit) & 1u;
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
fn linearIndex(gid: vec3u, groups: vec3u) -> u32 {
  return gid.x + gid.y * groups.x * 256u + gid.z * groups.x * groups.y * 256u;
}
@compute @workgroup_size(256)
fn materializeDenseFields(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let index = linearIndex(gid, groups);
  let brickSize = control[11];
  let voxelsPerBrick = brickSize * brickSize * brickSize;
  let leafIndex = index / voxelsPerBrick;
  if (leafIndex >= control[1]) { return; }
  let localIndex = index - leafIndex * voxelsPerBrick;
  let local = vec3u(localIndex % brickSize, (localIndex / brickSize) % brickSize, localIndex / (brickSize * brickSize));
  let leafBase = control[16] + leafIndex * 4u;
  let nodeIndex = topology[leafBase];
  let voxelOffset = topology[leafBase + 1u];
  let level = topology[nodeIndex * 8u + 2u];
  let brick = decodeMorton(topology[leafBase + 2u], topology[leafBase + 3u], level);
  var scale = 1u;
  if (params.dims.w != 0xffffffffu && params.dims.w > level) { scale = 1u << (params.dims.w - level); }
  let worldCell = vec3i((brick * brickSize + local) * scale);
  let q = worldCell - params.origin.xyz;
  let output = voxelOffset + localIndex;
  let geometryBase = output * 4u;
  let velocityBase = control[17] + output * 4u;
  if (scale != 1u || any(q < vec3i(0)) || any(q >= vec3i(params.dims.xyz))) {
    // Mixed-level/static environment leaves are initialized by the proxy pass
    // and then remain untouched by per-step dense fluid publication.
    return;
  }
  let dense = u32(q.x) + params.dims.x * (u32(q.y) + params.dims.y * u32(q.z));
  let phi = textureLoad(levelSet, q, 0).x;
  let solid = solidCells[dense];
  let h = min(params.cell.x, min(params.cell.y, params.cell.z));
  let solidPhi = (0.5 - clamp(solid.fraction, 0.0, 1.0)) * 2.0 * h;
  let materialOffset = control[18] + output;
  let previousIdentity = payload[materialOffset];
  let previousMaterial = previousIdentity & 0xffffu;
  let preserveStatic = params.origin.w > 0 && previousMaterial >= u32(params.origin.w);
  var combinedSolidPhi = solidPhi;
  var combinedSolidFraction = solid.fraction;
  if (preserveStatic) {
    combinedSolidPhi = min(combinedSolidPhi, bitcast<f32>(payload[geometryBase + 1u]));
    combinedSolidFraction = max(combinedSolidFraction, bitcast<f32>(payload[geometryBase + 2u]));
  }
  payload[geometryBase] = bitcast<u32>(phi); payload[geometryBase + 1u] = bitcast<u32>(combinedSolidPhi);
  payload[geometryBase + 2u] = bitcast<u32>(combinedSolidFraction); payload[geometryBase + 3u] = 0u;
  let fieldVelocity = textureLoad(velocityField, q, 0).xyz;
  let liquidFraction = clamp(0.5 - phi / max(h, 1e-8), 0.0, 1.0);
  payload[velocityBase] = bitcast<u32>(fieldVelocity.x); payload[velocityBase + 1u] = bitcast<u32>(fieldVelocity.y);
  payload[velocityBase + 2u] = bitcast<u32>(fieldVelocity.z); payload[velocityBase + 3u] = bitcast<u32>(liquidFraction);
  let boundary = q.x == 0 || q.x == i32(params.dims.x) - 1 || q.z == 0 || q.z == i32(params.dims.z) - 1 || q.y == 0 || (params.materials.w != 0u && q.y == i32(params.dims.y) - 1);
  var material = select(0u, params.materials.x, phi < 0.0);
  material = select(material, params.materials.z, boundary);
  material = select(material, params.materials.y, solid.fraction > 0.0);
  // Empty solid cells may be zero-initialized before the first raster pass.
  // Never treat their default owner 0 as rigid body 0 unless occupancy is
  // actually present.
  let owner = select(0xffffu, min(u32(max(solid.owner, 0)), 0xfffeu), solid.fraction > 0.0 && solid.owner >= 0);
  payload[materialOffset] = select((owner << 16u) | (material & 0xffffu), previousIdentity, preserveStatic && material == 0u);
}
`;

export const sparseBrickPublicationShader = publicationShader;
export const sparseBrickDenseFieldShader = denseFieldShader;

function positiveCapacity(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${name} must be a positive safe integer`);
  return value;
}

/** Portable 2D dispatch sizing for streams larger than WebGPU's x dimension. */
export function sparseBrickDispatchDimensions(itemCount: number, workgroupSize = 256): [number, number, number] {
  if (!Number.isSafeInteger(itemCount) || itemCount < 0) throw new RangeError("Item count must be a non-negative safe integer");
  const blocks = Math.ceil(itemCount / workgroupSize);
  if (blocks === 0) return [0, 1, 1];
  const x = Math.min(blocks, 65_535);
  const y = Math.ceil(blocks / x);
  if (y > 65_535) throw new RangeError("Sparse brick dispatch exceeds portable WebGPU dimensions");
  return [x, y, 1];
}

function checkedBytes(items: number, stride: number, name: string): number {
  const bytes = items * stride;
  if (!Number.isSafeInteger(bytes) || bytes > 0xffffffff) throw new RangeError(`${name} allocation is too large`);
  return Math.max(stride, bytes);
}

function alignBytes(value: number, alignment = 256): number {
  return Math.ceil(value / alignment) * alignment;
}

export class SparseBrickOctreeGPU {
  readonly brickSize: SparseBrickSize;
  readonly nodeCapacity: number;
  readonly leafCapacity: number;
  readonly voxelCapacity: number;
  /** Total bytes owned by this class, including control/indirect uniforms. */
  readonly allocatedBytes: number;
  readonly topology: GPUBuffer;
  readonly payload: GPUBuffer;
  readonly controlAndIndirect: GPUBuffer;
  readonly nodes: GPUBuffer;
  readonly nodeOffsetBytes = 0;
  readonly leaves: GPUBuffer;
  readonly leafOffsetBytes: number;
  readonly geometry: GPUBuffer;
  readonly geometryOffsetBytes = 0;
  readonly velocity: GPUBuffer;
  readonly velocityOffsetBytes: number;
  readonly materialOwners: GPUBuffer;
  readonly materialOwnerOffsetBytes: number;
  readonly control: GPUBuffer;
  readonly dispatchIndirect: GPUBuffer;
  readonly dispatchIndirectOffsetBytes = SPARSE_BRICK_GPU_LAYOUT.dispatchIndirectOffsetBytes;
  readonly drawIndirect: GPUBuffer;
  readonly drawIndirectOffsetBytes = SPARSE_BRICK_GPU_LAYOUT.drawIndirectOffsetBytes;

  private readonly device: GPUDevice;
  private readonly publicationPipeline: GPUComputePipeline;
  private readonly denseFieldPipeline: GPUComputePipeline;
  private readonly denseFieldParams: GPUBuffer;
  private destroyed = false;

  constructor(device: GPUDevice, options: SparseBrickOctreeGPUOptions) {
    if (options.brickSize !== 4 && options.brickSize !== 8) throw new RangeError("Sparse brick size must be 4 or 8");
    this.device = device;
    this.brickSize = options.brickSize;
    this.nodeCapacity = positiveCapacity(options.nodeCapacity, "Node capacity");
    this.leafCapacity = positiveCapacity(options.leafCapacity, "Leaf capacity");
    this.voxelCapacity = this.leafCapacity * this.brickSize ** 3;
    const label = options.label ?? "Sparse brick octree";
    const storageUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const indirectUsage = storageUsage | GPUBufferUsage.INDIRECT;
    const nodeBytes = checkedBytes(this.nodeCapacity, SPARSE_BRICK_GPU_LAYOUT.nodeStrideBytes, "Node");
    const leafBytes = checkedBytes(this.leafCapacity, SPARSE_BRICK_GPU_LAYOUT.leafStrideBytes, "Leaf");
    const geometryBytes = checkedBytes(this.voxelCapacity, SPARSE_BRICK_GPU_LAYOUT.geometryStrideBytes, "Geometry");
    const velocityBytes = checkedBytes(this.voxelCapacity, SPARSE_BRICK_GPU_LAYOUT.velocityStrideBytes, "Velocity");
    const materialOwnerBytes = checkedBytes(this.voxelCapacity, SPARSE_BRICK_GPU_LAYOUT.materialOwnerStrideBytes, "Material/owner");
    this.leafOffsetBytes = alignBytes(nodeBytes);
    this.velocityOffsetBytes = alignBytes(geometryBytes);
    this.materialOwnerOffsetBytes = this.velocityOffsetBytes + alignBytes(velocityBytes);
    const topologyBytes = this.leafOffsetBytes + leafBytes;
    const payloadBytes = this.materialOwnerOffsetBytes + materialOwnerBytes;
    this.allocatedBytes = topologyBytes + payloadBytes + SPARSE_BRICK_GPU_LAYOUT.controlStrideBytes + 64;
    this.topology = device.createBuffer({ label: `${label} topology arena`, size: topologyBytes, usage: storageUsage });
    this.payload = device.createBuffer({ label: `${label} payload arena`, size: payloadBytes, usage: storageUsage });
    this.controlAndIndirect = device.createBuffer({ label: `${label} control, overflow, and indirect arguments`, size: SPARSE_BRICK_GPU_LAYOUT.controlStrideBytes, usage: indirectUsage });
    // Aliases plus explicit offsets keep downstream bind/draw APIs ergonomic
    // while publication stays within WebGPU's portable eight-storage limit.
    this.nodes = this.topology; this.leaves = this.topology;
    this.geometry = this.payload; this.velocity = this.payload; this.materialOwners = this.payload;
    this.control = this.controlAndIndirect;
    this.dispatchIndirect = this.controlAndIndirect; this.drawIndirect = this.controlAndIndirect;
    device.queue.writeBuffer(this.control, 8 * 4, new Uint32Array([
      this.nodeCapacity, this.leafCapacity, this.voxelCapacity, this.brickSize,
    ]));
    device.queue.writeBuffer(this.control, 16 * 4, new Uint32Array([
      this.leafOffsetBytes / 4, this.velocityOffsetBytes / 4, this.materialOwnerOffsetBytes / 4,
    ]));
    this.denseFieldParams = device.createBuffer({ label: `${label} dense publication parameters`, size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.publicationPipeline = device.createComputePipeline({
      label: `${label} publication pipeline`, layout: "auto",
      compute: { module: device.createShaderModule({ label: `${label} publication shader`, code: publicationShader }), entryPoint: "publish" },
    });
    this.denseFieldPipeline = device.createComputePipeline({
      label: `${label} dense-field pipeline`, layout: "auto",
      compute: { module: device.createShaderModule({ label: `${label} dense-field shader`, code: denseFieldShader }), entryPoint: "materializeDenseFields" },
    });
  }

  /** Reset only authoritative counts/arguments; stale payload is unreachable. */
  encodeReset(encoder: GPUCommandEncoder): void {
    encoder.clearBuffer(this.control, 0, 8 * 4);
    encoder.clearBuffer(this.control, 12 * 4, 4 * 4);
    encoder.clearBuffer(this.control, this.dispatchIndirectOffsetBytes, 12);
    encoder.clearBuffer(this.control, this.drawIndirectOffsetBytes, 16);
  }

  /** Publish a GPU-authored topology and payload without any CPU count readback. */
  encodePublish(encoder: GPUCommandEncoder, source: SparseBrickPublicationSource): void {
    this.encodeReset(encoder);
    const bindGroup = this.device.createBindGroup({
      label: "Sparse brick publication bind group",
      layout: this.publicationPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: source.counts } },
        { binding: 1, resource: { buffer: source.topology } },
        { binding: 2, resource: { buffer: source.geometry } },
        { binding: 3, resource: { buffer: source.velocity } },
        { binding: 4, resource: { buffer: source.materialOwners } },
        { binding: 5, resource: { buffer: this.control } },
        { binding: 6, resource: { buffer: this.topology } },
        { binding: 7, resource: { buffer: this.payload } },
      ],
    });
    const maximum = Math.max(source.capacities.nodes, source.capacities.leaves, source.capacities.voxels, 1);
    const dispatch = sparseBrickDispatchDimensions(maximum);
    const pass = encoder.beginComputePass({ label: "Publish sparse brick octree" });
    pass.setPipeline(this.publicationPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(...dispatch);
    pass.end();
  }

  /**
   * Fill the current leaves from the octree solver's resident dense fields.
   * Topology must already have been published earlier in this command stream.
   */
  encodeFromDenseFields(encoder: GPUCommandEncoder, source: SparseBrickDenseFieldSource): void {
    const [nx, ny, nz] = source.dimensions;
    const origin = source.gridOriginCells ?? [0, 0, 0];
    for (const [value, name] of [[nx, "nx"], [ny, "ny"], [nz, "nz"]] as const) positiveCapacity(value, name);
    const materials = [source.fluidMaterialId, source.solidMaterialId, source.containerMaterialId ?? 0];
    for (const material of materials) if (!Number.isInteger(material) || material < 0 || material > 0xffff) throw new RangeError("Material IDs must fit uint16");
    const words = new ArrayBuffer(64);
    const uints = new Uint32Array(words);
    const ints = new Int32Array(words);
    const floats = new Float32Array(words);
    const finestLevel = source.finestLevel;
    if (finestLevel !== undefined && (!Number.isInteger(finestLevel) || finestLevel < 0 || finestLevel > SPARSE_BRICK_MAX_MORTON_BITS)) throw new RangeError("Finest topology level is invalid");
    uints.set([nx, ny, nz, finestLevel ?? 0xffffffff], 0);
    ints.set([origin[0], origin[1], origin[2], 0], 4);
    const preservedMaterialIdMinimum = source.preservedMaterialIdMinimum ?? 0;
    if (!Number.isInteger(preservedMaterialIdMinimum) || preservedMaterialIdMinimum < 0 || preservedMaterialIdMinimum > 0xffff) throw new RangeError("Preserved material minimum must fit uint16");
    ints[7] = preservedMaterialIdMinimum;
    floats.set([source.cellSize[0], source.cellSize[1], source.cellSize[2], 0], 8);
    uints.set([source.fluidMaterialId, source.solidMaterialId, source.containerMaterialId ?? 0, source.containerClosedTop ? 1 : 0], 12);
    this.device.queue.writeBuffer(this.denseFieldParams, 0, words);
    const bindGroup = this.device.createBindGroup({
      label: "Sparse brick dense-field bind group",
      layout: this.denseFieldPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.control } },
        { binding: 1, resource: { buffer: this.topology } },
        { binding: 2, resource: source.levelSet },
        { binding: 3, resource: source.velocity },
        { binding: 4, resource: { buffer: source.solidCells } },
        { binding: 5, resource: { buffer: this.denseFieldParams } },
        { binding: 6, resource: { buffer: this.payload } },
      ],
    });
    const dispatch = sparseBrickDispatchDimensions(this.voxelCapacity);
    const pass = encoder.beginComputePass({ label: "Materialize octree dense fields into sparse bricks" });
    pass.setPipeline(this.denseFieldPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(...dispatch);
    pass.end();
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.topology.destroy(); this.payload.destroy(); this.controlAndIndirect.destroy(); this.denseFieldParams.destroy();
  }
}
