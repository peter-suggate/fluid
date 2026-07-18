import {
  SPARSE_BRICK_NO_OWNER,
  packMaterialOwner,
  sparseBrickDispatchDimensions,
  type SparseBrickOctreeGPU,
} from "./sparse-brick-octree";

export type SparseSceneVector3 = readonly [number, number, number];
export type SparseSceneQuaternion = readonly [number, number, number, number];

export const SPARSE_SCENE_PRIMITIVE_STRIDE_BYTES = 48;

export const SPARSE_SCENE_PRIMITIVE_TYPES = Object.freeze({
  box: 1,
  cylinder: 2,
  ellipsoid: 3,
} as const);

interface SparseScenePrimitiveBase {
  center: SparseSceneVector3;
  materialId: number;
  ownerId?: number;
}

export interface SparseSceneBoxPrimitive extends SparseScenePrimitiveBase {
  kind: "box";
  halfExtents: SparseSceneVector3;
}

export interface SparseSceneCylinderPrimitive extends SparseScenePrimitiveBase {
  kind: "cylinder";
  radius: number;
  halfHeight: number;
  /** Local cylinder axis is +Y. Quaternion order is xyzw. */
  orientation?: SparseSceneQuaternion;
}

export interface SparseSceneEllipsoidPrimitive extends SparseScenePrimitiveBase {
  kind: "ellipsoid";
  radii: SparseSceneVector3;
  /** Quaternion order is xyzw. */
  orientation?: SparseSceneQuaternion;
}

export type SparseScenePrimitive =
  | SparseSceneBoxPrimitive
  | SparseSceneCylinderPrimitive
  | SparseSceneEllipsoidPrimitive;

export interface SparseSceneCellSample {
  solidSignedDistance: number;
  solidFraction: number;
  materialOwner: number;
}

export interface SparseSceneProxyVoxelizerOptions {
  cellSize: SparseSceneVector3;
  /** World-space position of cell-coordinate (0, 0, 0)'s minimum corner. */
  worldOrigin?: SparseSceneVector3;
  /** Maximum topology level. Omit for legacy fixed-level brick plans. */
  finestLevel?: number;
  label?: string;
}

function finiteVector(values: readonly number[], name: string): void {
  if (values.some((value) => !Number.isFinite(value))) throw new RangeError(`${name} must contain finite values`);
}

function positiveVector(values: SparseSceneVector3, name: string): void {
  finiteVector(values, name);
  if (values.some((value) => value <= 0)) throw new RangeError(`${name} must contain positive values`);
}

function validateIdentity(materialId: number, ownerId = SPARSE_BRICK_NO_OWNER): void {
  if (!Number.isInteger(materialId) || materialId < 1 || materialId > 0xffff) {
    throw new RangeError("Scene proxy material ID must be a nonzero uint16");
  }
  if (!Number.isInteger(ownerId) || ownerId < 0 || ownerId > 0xffff) {
    throw new RangeError("Scene proxy owner ID must fit uint16");
  }
}

function normalizedQuaternion(value: SparseSceneQuaternion | undefined): SparseSceneQuaternion {
  if (value === undefined) return [0, 0, 0, 1];
  finiteVector(value, "Primitive orientation");
  const length = Math.hypot(...value);
  if (length <= 1e-12) throw new RangeError("Primitive orientation must have nonzero length");
  return [value[0] / length, value[1] / length, value[2] / length, value[3] / length];
}

function primitiveExtent(primitive: SparseScenePrimitive): SparseSceneVector3 {
  if (primitive.kind === "box") {
    positiveVector(primitive.halfExtents, "Box half extents");
    return primitive.halfExtents;
  }
  if (primitive.kind === "cylinder") {
    if (!Number.isFinite(primitive.radius) || primitive.radius <= 0 ||
        !Number.isFinite(primitive.halfHeight) || primitive.halfHeight <= 0) {
      throw new RangeError("Cylinder radius and half height must be positive");
    }
    return [primitive.radius, primitive.halfHeight, primitive.radius];
  }
  positiveVector(primitive.radii, "Ellipsoid radii");
  return primitive.radii;
}

function primitiveType(primitive: SparseScenePrimitive): number {
  return SPARSE_SCENE_PRIMITIVE_TYPES[primitive.kind];
}

/**
 * Pack three vec4 words per primitive. The fourth lane of the first two vec4s
 * is bitcast on GPU to preserve exact type and material/owner integer identity:
 * `{ center.xyz, type }, { extent.xyz, packedIdentity }, { quaternion.xyzw }`.
 */
export function packSparseScenePrimitives(primitives: readonly SparseScenePrimitive[]): Uint32Array<ArrayBuffer> {
  const words = new Uint32Array(new ArrayBuffer(primitives.length * SPARSE_SCENE_PRIMITIVE_STRIDE_BYTES));
  const floats = new Float32Array(words.buffer);
  for (let index = 0; index < primitives.length; index += 1) {
    const primitive = primitives[index];
    finiteVector(primitive.center, "Primitive center");
    validateIdentity(primitive.materialId, primitive.ownerId);
    const extent = primitiveExtent(primitive);
    const orientation = primitive.kind === "box" ? [0, 0, 0, 1] as const : normalizedQuaternion(primitive.orientation);
    const base = index * (SPARSE_SCENE_PRIMITIVE_STRIDE_BYTES / 4);
    floats.set(primitive.center, base);
    words[base + 3] = primitiveType(primitive);
    floats.set(extent, base + 4);
    words[base + 7] = packMaterialOwner(primitive.materialId, primitive.ownerId);
    floats.set(orientation, base + 8);
  }
  return words;
}

function inverseRotate(point: SparseSceneVector3, quaternion: SparseSceneQuaternion): SparseSceneVector3 {
  const [qx, qy, qz, qw] = quaternion;
  // Apply conjugate(q) * point * q using the cross-product quaternion form.
  const tx = 2 * (-qy * point[2] + qz * point[1]);
  const ty = 2 * (-qz * point[0] + qx * point[2]);
  const tz = 2 * (-qx * point[1] + qy * point[0]);
  return [
    point[0] + qw * tx + (-qy * tz + qz * ty),
    point[1] + qw * ty + (-qz * tx + qx * tz),
    point[2] + qw * tz + (-qx * ty + qy * tx),
  ];
}

function boxDistance(point: SparseSceneVector3, halfExtents: SparseSceneVector3): number {
  const q = point.map((value, axis) => Math.abs(value) - halfExtents[axis]) as unknown as SparseSceneVector3;
  const outside = Math.hypot(Math.max(q[0], 0), Math.max(q[1], 0), Math.max(q[2], 0));
  return outside + Math.min(Math.max(q[0], q[1], q[2]), 0);
}

function cylinderDistance(point: SparseSceneVector3, radius: number, halfHeight: number): number {
  const radial = Math.hypot(point[0], point[2]) - radius;
  const vertical = Math.abs(point[1]) - halfHeight;
  return Math.hypot(Math.max(radial, 0), Math.max(vertical, 0)) + Math.min(Math.max(radial, vertical), 0);
}

function ellipsoidDistance(point: SparseSceneVector3, radii: SparseSceneVector3): number {
  const k0 = Math.hypot(point[0] / radii[0], point[1] / radii[1], point[2] / radii[2]);
  const k1 = Math.hypot(point[0] / (radii[0] ** 2), point[1] / (radii[1] ** 2), point[2] / (radii[2] ** 2));
  return k1 > 1e-12 ? k0 * (k0 - 1) / k1 : -Math.min(...radii);
}

/** CPU mirror of the WGSL primitive SDF, useful for topology planning and tests. */
export function sparseScenePrimitiveSignedDistance(
  primitive: SparseScenePrimitive,
  worldPoint: SparseSceneVector3,
): number {
  finiteVector(worldPoint, "World point");
  finiteVector(primitive.center, "Primitive center");
  validateIdentity(primitive.materialId, primitive.ownerId);
  const localOffset: SparseSceneVector3 = [
    worldPoint[0] - primitive.center[0],
    worldPoint[1] - primitive.center[1],
    worldPoint[2] - primitive.center[2],
  ];
  if (primitive.kind === "box") {
    positiveVector(primitive.halfExtents, "Box half extents");
    return boxDistance(localOffset, primitive.halfExtents);
  }
  const local = inverseRotate(localOffset, normalizedQuaternion(primitive.orientation));
  if (primitive.kind === "cylinder") {
    primitiveExtent(primitive);
    return cylinderDistance(local, primitive.radius, primitive.halfHeight);
  }
  positiveVector(primitive.radii, "Ellipsoid radii");
  return ellipsoidDistance(local, primitive.radii);
}

/** Conservative cell-center occupancy mirror used by the GPU voxelization pass. */
export function sampleSparseScenePrimitiveCell(
  primitives: readonly SparseScenePrimitive[],
  cellCenter: SparseSceneVector3,
  cellSize: SparseSceneVector3,
): SparseSceneCellSample {
  positiveVector(cellSize, "Cell size");
  let distance = Number.POSITIVE_INFINITY;
  let identity = packMaterialOwner(0, SPARSE_BRICK_NO_OWNER);
  for (const primitive of primitives) {
    const candidate = sparseScenePrimitiveSignedDistance(primitive, cellCenter);
    if (candidate < distance) {
      distance = candidate;
      identity = packMaterialOwner(primitive.materialId, primitive.ownerId);
    }
  }
  if (primitives.length === 0) return { solidSignedDistance: distance, solidFraction: 0, materialOwner: identity };
  const cellRadius = 0.5 * Math.hypot(...cellSize);
  const fraction = Math.max(0, Math.min(1, 0.5 - distance / (2 * cellRadius)));
  return {
    solidSignedDistance: distance,
    solidFraction: fraction,
    materialOwner: fraction > 0 ? identity : packMaterialOwner(0, SPARSE_BRICK_NO_OWNER),
  };
}

export const sparseSceneProxyVoxelizationShader = /* wgsl */ `
struct ScenePrimitive {
  centerType: vec4f,
  extentIdentity: vec4f,
  rotation: vec4f,
}
struct Params {
  worldOrigin: vec4f,
  cell: vec4f,
  primitiveCount: u32,
  finestLevel: u32,
  _padding: vec2u,
}
@group(0) @binding(0) var<storage, read> control: array<u32>;
@group(0) @binding(1) var<storage, read> topology: array<u32>;
@group(0) @binding(2) var<storage, read_write> payload: array<u32>;
@group(0) @binding(3) var<storage, read> primitives: array<ScenePrimitive>;
@group(0) @binding(4) var<uniform> params: Params;

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
fn inverseRotate(point: vec3f, quaternion: vec4f) -> vec3f {
  let vector = -quaternion.xyz;
  let twiceCross = 2.0 * cross(vector, point);
  return point + quaternion.w * twiceCross + cross(vector, twiceCross);
}
fn boxDistance(point: vec3f, halfExtents: vec3f) -> f32 {
  let q = abs(point) - halfExtents;
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}
fn cylinderDistance(point: vec3f, radius: f32, halfHeight: f32) -> f32 {
  let q = vec2f(length(point.xz) - radius, abs(point.y) - halfHeight);
  return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0);
}
fn ellipsoidDistance(point: vec3f, radii: vec3f) -> f32 {
  let k0 = length(point / radii);
  let k1 = length(point / (radii * radii));
  return select(-min(radii.x, min(radii.y, radii.z)), k0 * (k0 - 1.0) / k1, k1 > 1e-8);
}
fn primitiveDistance(primitive: ScenePrimitive, world: vec3f) -> f32 {
  let primitiveType = bitcast<u32>(primitive.centerType.w);
  let offset = world - primitive.centerType.xyz;
  if (primitiveType == 1u) { return boxDistance(offset, primitive.extentIdentity.xyz); }
  let local = inverseRotate(offset, primitive.rotation);
  if (primitiveType == 2u) { return cylinderDistance(local, primitive.extentIdentity.x, primitive.extentIdentity.y); }
  if (primitiveType == 3u) { return ellipsoidDistance(local, primitive.extentIdentity.xyz); }
  return 1e20;
}
fn linearIndex(gid: vec3u, groups: vec3u) -> u32 {
  return gid.x + gid.y * groups.x * 256u + gid.z * groups.x * groups.y * 256u;
}

@compute @workgroup_size(256)
fn voxelizeSceneProxies(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
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
  if (params.finestLevel != 0xffffffffu && params.finestLevel > level) { scale = 1u << (params.finestLevel - level); }
  let worldCell = (brick * brickSize + local) * scale;
  let world = params.worldOrigin.xyz + (vec3f(worldCell) + 0.5 * f32(scale)) * params.cell.xyz;

  var bestDistance = 1e20;
  var bestIdentity = 0xffff0000u;
  for (var primitiveIndex = 0u; primitiveIndex < params.primitiveCount; primitiveIndex += 1u) {
    let primitive = primitives[primitiveIndex];
    let candidate = primitiveDistance(primitive, world);
    if (candidate < bestDistance) {
      bestDistance = candidate;
      bestIdentity = bitcast<u32>(primitive.extentIdentity.w);
    }
  }

  let output = voxelOffset + localIndex;
  let geometryBase = output * 4u;
  var previousDistance = bitcast<f32>(payload[geometryBase + 1u]);
  let previousFraction = bitcast<f32>(payload[geometryBase + 2u]);
  let materialOffset = control[18] + output;
  let previousIdentity = payload[materialOffset];
  let previousMaterial = previousIdentity & 0xffffu;
  // Freshly published non-solver leaves are zero-filled. Treat that exact
  // state as empty space rather than a zero-distance solid.
  if (previousMaterial == 0u && previousFraction == 0.0 && previousDistance == 0.0) { previousDistance = 1e20; }
  let cellRadius = 0.5 * length(params.cell.xyz * f32(scale));
  let primitiveFraction = clamp(0.5 - bestDistance / (2.0 * cellRadius), 0.0, 1.0);
  payload[geometryBase + 1u] = bitcast<u32>(min(previousDistance, bestDistance));
  payload[geometryBase + 2u] = bitcast<u32>(max(previousFraction, primitiveFraction));

  // Fluid, container, rigid-body, terrain, and any other authored identity wins.
  if (previousMaterial == 0u && primitiveFraction > 0.0 && bestDistance <= previousDistance) {
    payload[materialOffset] = bestIdentity;
  }
}
`;

/** GPU-only static environment voxelization over the octree's currently published leaves. */
export class SparseSceneProxyVoxelizer {
  readonly primitiveCount: number;
  readonly allocatedBytes: number;

  private readonly tree: SparseBrickOctreeGPU;
  private readonly primitiveBuffer: GPUBuffer;
  private readonly paramsBuffer: GPUBuffer;
  private readonly pipeline: GPUComputePipeline;
  private readonly bindGroup: GPUBindGroup;
  private destroyed = false;

  constructor(
    device: GPUDevice,
    tree: SparseBrickOctreeGPU,
    primitives: readonly SparseScenePrimitive[],
    options: SparseSceneProxyVoxelizerOptions,
  ) {
    positiveVector(options.cellSize, "Cell size");
    const worldOrigin = options.worldOrigin ?? [0, 0, 0];
    finiteVector(worldOrigin, "World origin");
    const packed = packSparseScenePrimitives(primitives);
    this.tree = tree;
    this.primitiveCount = primitives.length;
    const primitiveBytes = Math.max(SPARSE_SCENE_PRIMITIVE_STRIDE_BYTES, packed.byteLength);
    this.allocatedBytes = primitiveBytes + 48;
    const label = options.label ?? "Sparse scene proxies";
    this.primitiveBuffer = device.createBuffer({
      label: `${label} primitives`, size: primitiveBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.paramsBuffer = device.createBuffer({
      label: `${label} parameters`, size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    if (packed.byteLength > 0) device.queue.writeBuffer(this.primitiveBuffer, 0, packed);
    const parameterData = new ArrayBuffer(48);
    const parameterFloats = new Float32Array(parameterData);
    const parameterUints = new Uint32Array(parameterData);
    parameterFloats.set(worldOrigin, 0);
    parameterFloats.set(options.cellSize, 4);
    parameterUints[8] = primitives.length;
    if (options.finestLevel !== undefined && (!Number.isInteger(options.finestLevel) || options.finestLevel < 0 || options.finestLevel > 21)) throw new RangeError("Finest topology level is invalid");
    parameterUints[9] = options.finestLevel ?? 0xffffffff;
    device.queue.writeBuffer(this.paramsBuffer, 0, parameterData);
    this.pipeline = device.createComputePipeline({
      label: `${label} voxelization pipeline`, layout: "auto",
      compute: {
        module: device.createShaderModule({ label: `${label} voxelization shader`, code: sparseSceneProxyVoxelizationShader }),
        entryPoint: "voxelizeSceneProxies",
      },
    });
    this.bindGroup = device.createBindGroup({
      label: `${label} voxelization bind group`,
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tree.control } },
        // Bind whole arenas: leaf/material offsets are GPU-resident control words 16/18.
        { binding: 1, resource: { buffer: tree.topology } },
        { binding: 2, resource: { buffer: tree.payload } },
        { binding: 3, resource: { buffer: this.primitiveBuffer } },
        { binding: 4, resource: { buffer: this.paramsBuffer } },
      ],
    });
  }

  /** Encode after dense-field materialization so authored materials retain precedence. */
  encode(encoder: GPUCommandEncoder): void {
    if (this.destroyed || this.primitiveCount === 0) return;
    const pass = encoder.beginComputePass({ label: "Voxelize static scene proxies into sparse bricks" });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(...sparseBrickDispatchDimensions(this.tree.voxelCapacity));
    pass.end();
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.primitiveBuffer.destroy();
    this.paramsBuffer.destroy();
  }
}
