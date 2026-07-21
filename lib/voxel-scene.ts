import { canonicalScene, type Quaternion, type RigidBodyDescription, type RigidShape, type SceneDescription, type Vec3 } from "./model";
import { TERRAIN_DEFAULT_FLAT, TERRAIN_UNION_EXPONENT, type TerrainDescription } from "./terrain";

/** Deterministic CPU planning schema consumed by the future GPU brick builder. */
export const VOXEL_SCENE_PLAN_VERSION = "1.0.0" as const;

export type LinearRgb = readonly [r: number, g: number, b: number];

export const VOXEL_MATERIAL_IDS = {
  containerGlass: 1,
  terrain: 2,
  fluid: 3,
  sphere: 16,
  box: 17,
  capsule: 18,
  cylinder: 19
} as const;

export interface VoxelMaterial {
  /** Stable GPU table index. Zero is intentionally reserved for empty space. */
  id: number;
  key: keyof typeof VOXEL_MATERIAL_IDS;
  name: string;
  closure: "opaque" | "dielectric" | "thin-dielectric";
  baseColorLinear: LinearRgb;
  emissiveLinear: LinearRgb;
  metallic: number;
  roughness: number;
  transmission: number;
  ior: number;
  /** Documents which existing rendering value must remain visually stable. */
  colorProvenance: string;
  terrainPalette?: {
    lawnDarkLinear: LinearRgb;
    lawnLightLinear: LinearRgb;
    sandLinear: LinearRgb;
  };
}

/**
 * The rigid values exactly match webgpu-renderer.ts's scene-linear palette.
 * They must not receive an sRGB-to-linear conversion when uploaded.
 */
export const VOXEL_MATERIALS: ReadonlyArray<VoxelMaterial> = [
  {
    id: VOXEL_MATERIAL_IDS.containerGlass, key: "containerGlass", name: "Container glass", closure: "thin-dielectric",
    baseColorLinear: [0.42, 0.78, 0.72], emissiveLinear: [0, 0, 0], metallic: 0, roughness: 0.06, transmission: 0.96, ior: 1.5,
    colorProvenance: "webgpu-renderer glass tint"
  },
  {
    id: VOXEL_MATERIAL_IDS.terrain, key: "terrain", name: "Terrain", closure: "opaque",
    baseColorLinear: [0.56, 0.5525, 0.5275], emissiveLinear: [0, 0, 0], metallic: 0, roughness: 0.92, transmission: 0, ior: 1.45,
    colorProvenance: "webgpu-environments garden lawn midpoint",
    terrainPalette: {
      lawnDarkLinear: [0.46, 0.455, 0.435],
      lawnLightLinear: [0.66, 0.65, 0.62],
      sandLinear: [0.56, 0.55, 0.52]
    }
  },
  {
    id: VOXEL_MATERIAL_IDS.fluid, key: "fluid", name: "Water", closure: "dielectric",
    // Exact midpoint of the existing renderer's in-water scatter
    // (0.018, 0.34, 0.29) and scientific grid highlight (0.42, 0.96, 0.82).
    // This is intentionally more legible than optically composited water when
    // inspecting raw occupied voxels.
    baseColorLinear: [0.219, 0.65, 0.555], emissiveLinear: [0, 0, 0], metallic: 0, roughness: 0.08, transmission: 0.94, ior: 1.333,
    colorProvenance: "midpoint of webgpu-renderer water scatter and scientific grid highlight"
  },
  {
    id: VOXEL_MATERIAL_IDS.sphere, key: "sphere", name: "Sphere", closure: "opaque",
    baseColorLinear: [0.95, 0.63, 0.29], emissiveLinear: [0, 0, 0], metallic: 0, roughness: 0.48, transmission: 0, ior: 1.45,
    colorProvenance: "webgpu-renderer rigid palette"
  },
  {
    id: VOXEL_MATERIAL_IDS.box, key: "box", name: "Box", closure: "opaque",
    baseColorLinear: [0.48, 0.66, 0.96], emissiveLinear: [0, 0, 0], metallic: 0, roughness: 0.52, transmission: 0, ior: 1.45,
    colorProvenance: "webgpu-renderer rigid palette"
  },
  {
    id: VOXEL_MATERIAL_IDS.capsule, key: "capsule", name: "Capsule", closure: "opaque",
    baseColorLinear: [0.84, 0.42, 0.48], emissiveLinear: [0, 0, 0], metallic: 0, roughness: 0.46, transmission: 0, ior: 1.45,
    colorProvenance: "webgpu-renderer rigid palette"
  },
  {
    id: VOXEL_MATERIAL_IDS.cylinder, key: "cylinder", name: "Cylinder", closure: "opaque",
    baseColorLinear: [0.66, 0.52, 0.92], emissiveLinear: [0, 0, 0], metallic: 0, roughness: 0.5, transmission: 0, ior: 1.45,
    colorProvenance: "webgpu-renderer rigid palette"
  }
];

export interface Vec3i { x: number; y: number; z: number }
export interface VoxelAabb { min: Vec3; max: Vec3 }
export interface VoxelIndexRange { min: Vec3i; maxExclusive: Vec3i }

export interface VoxelCandidateBounds {
  /** Analytic source support before sampling/filter padding. */
  exact_m: VoxelAabb;
  /** Exact support expanded by half a voxel diagonal. */
  conservative_m: VoxelAabb;
  voxelRange: VoxelIndexRange;
  brickRange: VoxelIndexRange;
  /** World/local bounds snapped outwards to complete voxels and bricks. */
  voxelAligned_m: VoxelAabb;
  brickAligned_m: VoxelAabb;
}

export interface SparseBrickLayoutPlan {
  voxelSize_m: number;
  brickCells: number;
  brickSize_m: number;
  /** Fluid-compatible lattice origin: lower x/z container corner and y=0. */
  worldOrigin_m: Vec3;
  interiorVoxelRange: VoxelIndexRange;
  interiorBrickRange: VoxelIndexRange;
  interiorBounds_m: VoxelAabb;
  conservativePadding_m: number;
}

interface VoxelSourceBase {
  id: string;
  materialId: number;
  composition: "union";
  candidate: VoxelCandidateBounds;
  revisionHash: string;
}

export type ContainerBoundarySide = "floor" | "left" | "right" | "front" | "back" | "ceiling";

export interface VoxelContainerBoundarySource extends VoxelSourceBase {
  kind: "container-boundary";
  partition: "static";
  side: ContainerBoundarySide;
  /** Unit normal pointing from the solid shell into the fluid domain. */
  inwardNormal: Vec3;
  /** Exact location of the authored zero-thickness physical boundary. */
  surfaceCoordinate_m: number;
  /** Finite representation shell outside the domain; not physical wall thickness. */
  compilationShellThickness_m: number;
}

export interface VoxelTerrainSource extends VoxelSourceBase {
  kind: "terrain-heightfield";
  partition: "static";
  terrain: TerrainDescription;
  evaluator: {
    unionExponent: number;
    defaultFlat: number;
    clampMinimum_m: 0;
  };
}

export interface VoxelRigidPrimitive {
  kind: RigidShape;
  dimensions_m: Vec3;
  /** Explicitly records the pre-existing non-uniform dimension convention. */
  dimensionSemantics: "radius" | "full-extents" | "radius-and-cylinder-height";
}

export interface VoxelTransform {
  position_m: Vec3;
  orientation: Quaternion;
}

export interface VoxelLocalAllocation {
  origin_m: Vec3;
  candidate: VoxelCandidateBounds;
  brickDimensions: Vec3i;
}

export interface VoxelRigidSource extends VoxelSourceBase {
  kind: "rigid-primitive";
  partition: "static" | "dynamic";
  bodyId: string;
  name: string;
  primitive: VoxelRigidPrimitive;
  transform: VoxelTransform;
  /** Shape/topology revision excludes the transform so local bricks can persist while a body moves. */
  topologyRevisionHash: string;
  transformRevisionHash: string;
  localAllocation: VoxelLocalAllocation;
}

export type VoxelStaticSource = VoxelContainerBoundarySource | VoxelTerrainSource | VoxelRigidSource;

export interface VoxelSceneHashInputs {
  planVersion: typeof VOXEL_SCENE_PLAN_VERSION;
  sceneCanonical: string;
  voxelSize_m: number;
  brickCells: number;
  conservativePadding_m: number;
  materialRevisionHash: string;
}

export interface VoxelSceneRevisions {
  sceneHash: string;
  sceneRevision: number;
  staticHash: string;
  staticRevision: number;
  dynamicTopologyHash: string;
  dynamicTopologyRevision: number;
  dynamicTransformsHash: string;
  dynamicTransformsRevision: number;
}

export interface VoxelScenePlan {
  version: typeof VOXEL_SCENE_PLAN_VERSION;
  sceneId: string;
  layout: SparseBrickLayoutPlan;
  materials: ReadonlyArray<VoxelMaterial>;
  staticSources: ReadonlyArray<VoxelStaticSource>;
  dynamicSources: ReadonlyArray<VoxelRigidSource>;
  hashInputs: VoxelSceneHashInputs;
  revisions: VoxelSceneRevisions;
  bounds_m: VoxelAabb;
}

export interface PlanVoxelSceneOptions {
  /** Defaults to half the voxel diagonal, sufficient for centre-sampled conservative candidates. */
  conservativePadding_m?: number;
}

export function materialIdForRigidShape(shape: RigidShape): number {
  return VOXEL_MATERIAL_IDS[shape];
}

export function voxelMaterial(id: number): VoxelMaterial {
  const material = VOXEL_MATERIALS.find((entry) => entry.id === id);
  if (!material) throw new Error(`Unknown voxel material ID ${id}`);
  return material;
}

/** Matches SparseVoxelDebugMaterial in webgpu-voxel-debug.ts: two vec4f lanes. */
export const VOXEL_DEBUG_MATERIAL_STRIDE_BYTES = 32;
const VOXEL_DEBUG_MATERIAL_FLOATS = VOXEL_DEBUG_MATERIAL_STRIDE_BYTES / Float32Array.BYTES_PER_ELEMENT;

/**
 * Packs a dense, directly indexable GPU debug table. Stable material IDs are
 * sparse (rigid IDs begin at 16), so compact material-array order is never a
 * valid shader index. Slot zero stays empty; unassigned positive slots are
 * diagnostic magenta so an invalid active-voxel material is conspicuous.
 */
export function packVoxelDebugMaterialTable(materials: ReadonlyArray<VoxelMaterial> = VOXEL_MATERIALS): Float32Array {
  const ids = new Set<number>();
  let maximumId = 0;
  for (const material of materials) {
    if (!Number.isInteger(material.id) || material.id <= 0) throw new Error(`Voxel material ID must be a positive integer, received ${material.id}`);
    if (ids.has(material.id)) throw new Error(`Duplicate voxel material ID ${material.id}`);
    ids.add(material.id);
    maximumId = Math.max(maximumId, material.id);
  }
  const packed = new Float32Array((maximumId + 1) * VOXEL_DEBUG_MATERIAL_FLOATS);
  for (let id = 1; id <= maximumId; id += 1) packed.set([1, 0, 1, 1, 0, 0, 0, 1], id * VOXEL_DEBUG_MATERIAL_FLOATS);
  for (const material of materials) {
    packed.set([
      ...material.baseColorLinear, material.closure === "thin-dielectric" ? 0.24 : 1,
      ...material.emissiveLinear, material.roughness
    ], material.id * VOXEL_DEBUG_MATERIAL_FLOATS);
  }
  return packed;
}

function stable(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stable);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(([key, child]) => [key, stable(child)]));
  }
  return value;
}

function hashValue(value: unknown): string {
  const bytes = new TextEncoder().encode(JSON.stringify(stable(value)));
  let hash = 0xcbf29ce484222325n;
  for (const byte of bytes) {
    hash ^= BigInt(byte);
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  return hash.toString(16).padStart(16, "0");
}

function numericRevision(hash: string): number {
  return Number.parseInt(hash.slice(-8), 16) >>> 0;
}

function cloneVec3(value: Vec3): Vec3 {
  return { x: value.x, y: value.y, z: value.z };
}

function expand(bounds: VoxelAabb, amount: number): VoxelAabb {
  return {
    min: { x: bounds.min.x - amount, y: bounds.min.y - amount, z: bounds.min.z - amount },
    max: { x: bounds.max.x + amount, y: bounds.max.y + amount, z: bounds.max.z + amount }
  };
}

function rangeForBounds(bounds: VoxelAabb, origin: Vec3, cellSize: number): VoxelIndexRange {
  return {
    min: {
      x: Math.floor((bounds.min.x - origin.x) / cellSize),
      y: Math.floor((bounds.min.y - origin.y) / cellSize),
      z: Math.floor((bounds.min.z - origin.z) / cellSize)
    },
    maxExclusive: {
      x: Math.ceil((bounds.max.x - origin.x) / cellSize),
      y: Math.ceil((bounds.max.y - origin.y) / cellSize),
      z: Math.ceil((bounds.max.z - origin.z) / cellSize)
    }
  };
}

function boundsForRange(range: VoxelIndexRange, origin: Vec3, cellSize: number): VoxelAabb {
  return {
    min: {
      x: origin.x + range.min.x * cellSize,
      y: origin.y + range.min.y * cellSize,
      z: origin.z + range.min.z * cellSize
    },
    max: {
      x: origin.x + range.maxExclusive.x * cellSize,
      y: origin.y + range.maxExclusive.y * cellSize,
      z: origin.z + range.maxExclusive.z * cellSize
    }
  };
}

function candidateBounds(exact_m: VoxelAabb, origin: Vec3, voxelSize_m: number, brickSize_m: number, padding_m: number): VoxelCandidateBounds {
  const conservative_m = expand(exact_m, padding_m);
  const voxelRange = rangeForBounds(conservative_m, origin, voxelSize_m);
  const brickRange = rangeForBounds(conservative_m, origin, brickSize_m);
  return {
    exact_m, conservative_m, voxelRange, brickRange,
    voxelAligned_m: boundsForRange(voxelRange, origin, voxelSize_m),
    brickAligned_m: boundsForRange(brickRange, origin, brickSize_m)
  };
}

function localRigidBounds(body: RigidBodyDescription): VoxelAabb {
  const d = body.dimensions_m;
  if (body.shape === "sphere") return { min: { x: -d.x, y: -d.x, z: -d.x }, max: { x: d.x, y: d.x, z: d.x } };
  if (body.shape === "box") return { min: { x: -d.x / 2, y: -d.y / 2, z: -d.z / 2 }, max: { x: d.x / 2, y: d.y / 2, z: d.z / 2 } };
  const verticalExtent = body.shape === "capsule" ? d.y / 2 + d.x : d.y / 2;
  return { min: { x: -d.x, y: -verticalExtent, z: -d.x }, max: { x: d.x, y: verticalExtent, z: d.x } };
}

function rotationAbs(qInput: Quaternion): readonly [Vec3, Vec3, Vec3] {
  const length = Math.hypot(qInput.w, qInput.x, qInput.y, qInput.z) || 1;
  const w = qInput.w / length, x = qInput.x / length, y = qInput.y / length, z = qInput.z / length;
  return [
    { x: Math.abs(1 - 2 * (y * y + z * z)), y: Math.abs(2 * (x * y - z * w)), z: Math.abs(2 * (x * z + y * w)) },
    { x: Math.abs(2 * (x * y + z * w)), y: Math.abs(1 - 2 * (x * x + z * z)), z: Math.abs(2 * (y * z - x * w)) },
    { x: Math.abs(2 * (x * z - y * w)), y: Math.abs(2 * (y * z + x * w)), z: Math.abs(1 - 2 * (x * x + y * y)) }
  ];
}

function transformAabb(local: VoxelAabb, transform: VoxelTransform): VoxelAabb {
  const half = {
    x: (local.max.x - local.min.x) / 2,
    y: (local.max.y - local.min.y) / 2,
    z: (local.max.z - local.min.z) / 2
  };
  const rows = rotationAbs(transform.orientation);
  const extent = {
    x: rows[0].x * half.x + rows[0].y * half.y + rows[0].z * half.z,
    y: rows[1].x * half.x + rows[1].y * half.y + rows[1].z * half.z,
    z: rows[2].x * half.x + rows[2].y * half.y + rows[2].z * half.z
  };
  return {
    min: { x: transform.position_m.x - extent.x, y: transform.position_m.y - extent.y, z: transform.position_m.z - extent.z },
    max: { x: transform.position_m.x + extent.x, y: transform.position_m.y + extent.y, z: transform.position_m.z + extent.z }
  };
}

function bodyPrimitive(body: RigidBodyDescription): VoxelRigidPrimitive {
  return {
    kind: body.shape,
    dimensions_m: cloneVec3(body.dimensions_m),
    dimensionSemantics: body.shape === "sphere" ? "radius" : body.shape === "box" ? "full-extents" : "radius-and-cylinder-height"
  };
}

function rigidSource(body: RigidBodyDescription, layout: SparseBrickLayoutPlan): VoxelRigidSource {
  const primitive = bodyPrimitive(body);
  const transform = { position_m: cloneVec3(body.position_m), orientation: { ...body.orientation } };
  const partition = body.motion === "static" ? "static" : "dynamic";
  const topologyRevisionHash = hashValue({
    bodyId: body.id,
    primitive,
    materialId: materialIdForRigidShape(body.shape),
    voxelSize_m: layout.voxelSize_m,
    brickCells: layout.brickCells,
    conservativePadding_m: layout.conservativePadding_m
  });
  const transformRevisionHash = hashValue(transform);
  const localExact = localRigidBounds(body);
  const localCandidate = candidateBounds(localExact, { x: 0, y: 0, z: 0 }, layout.voxelSize_m, layout.brickSize_m, layout.conservativePadding_m);
  const brickDimensions = {
    x: localCandidate.brickRange.maxExclusive.x - localCandidate.brickRange.min.x,
    y: localCandidate.brickRange.maxExclusive.y - localCandidate.brickRange.min.y,
    z: localCandidate.brickRange.maxExclusive.z - localCandidate.brickRange.min.z
  };
  const worldExact = transformAabb(localExact, transform);
  const common = {
    id: `rigid:${body.id}`,
    materialId: materialIdForRigidShape(body.shape),
    composition: "union" as const,
    candidate: candidateBounds(worldExact, layout.worldOrigin_m, layout.voxelSize_m, layout.brickSize_m, layout.conservativePadding_m)
  };
  return {
    ...common,
    revisionHash: hashValue({ topologyRevisionHash, transformRevisionHash, partition }),
    kind: "rigid-primitive",
    partition,
    bodyId: body.id,
    name: body.name,
    primitive,
    transform,
    topologyRevisionHash,
    transformRevisionHash,
    localAllocation: { origin_m: cloneVec3(localCandidate.brickAligned_m.min), candidate: localCandidate, brickDimensions }
  };
}

function boundarySources(scene: SceneDescription, layout: SparseBrickLayoutPlan): VoxelContainerBoundarySource[] {
  const { width_m: width, height_m: height, depth_m: depth, top } = scene.container;
  const x0 = -width / 2, x1 = width / 2, z0 = -depth / 2, z1 = depth / 2;
  const shell = layout.voxelSize_m;
  const entries: Array<{ side: ContainerBoundarySide; inwardNormal: Vec3; surfaceCoordinate_m: number; exact: VoxelAabb }> = [
    { side: "floor", inwardNormal: { x: 0, y: 1, z: 0 }, surfaceCoordinate_m: 0, exact: { min: { x: x0, y: -shell, z: z0 }, max: { x: x1, y: 0, z: z1 } } },
    { side: "left", inwardNormal: { x: 1, y: 0, z: 0 }, surfaceCoordinate_m: x0, exact: { min: { x: x0 - shell, y: 0, z: z0 }, max: { x: x0, y: height, z: z1 } } },
    { side: "right", inwardNormal: { x: -1, y: 0, z: 0 }, surfaceCoordinate_m: x1, exact: { min: { x: x1, y: 0, z: z0 }, max: { x: x1 + shell, y: height, z: z1 } } },
    { side: "front", inwardNormal: { x: 0, y: 0, z: 1 }, surfaceCoordinate_m: z0, exact: { min: { x: x0, y: 0, z: z0 - shell }, max: { x: x1, y: height, z: z0 } } },
    { side: "back", inwardNormal: { x: 0, y: 0, z: -1 }, surfaceCoordinate_m: z1, exact: { min: { x: x0, y: 0, z: z1 }, max: { x: x1, y: height, z: z1 + shell } } }
  ];
  if (top === "closed") entries.push({
    side: "ceiling", inwardNormal: { x: 0, y: -1, z: 0 }, surfaceCoordinate_m: height,
    exact: { min: { x: x0, y: height, z: z0 }, max: { x: x1, y: height + shell, z: z1 } }
  });
  return entries.map((entry) => {
    const candidate = candidateBounds(entry.exact, layout.worldOrigin_m, layout.voxelSize_m, layout.brickSize_m, layout.conservativePadding_m);
    const source = {
      id: `container:${entry.side}`,
      kind: "container-boundary" as const,
      partition: "static" as const,
      side: entry.side,
      materialId: VOXEL_MATERIAL_IDS.containerGlass,
      composition: "union" as const,
      inwardNormal: entry.inwardNormal,
      surfaceCoordinate_m: entry.surfaceCoordinate_m,
      compilationShellThickness_m: shell,
      candidate
    };
    return { ...source, revisionHash: hashValue(source) };
  });
}

function cloneTerrain(terrain: TerrainDescription): TerrainDescription {
  return {
    baseHeight_m: terrain.baseHeight_m,
    features: terrain.features.map((feature) => ({
      ...feature,
      center_m: { ...feature.center_m },
      radius_m: { ...feature.radius_m }
    }))
  };
}

function terrainSource(scene: SceneDescription, layout: SparseBrickLayoutPlan): VoxelTerrainSource | undefined {
  if (!scene.terrain) return undefined;
  const terrain = cloneTerrain(scene.terrain);
  const maximumHeight = Math.min(scene.container.height_m, terrain.baseHeight_m + terrain.features.reduce((sum, feature) => sum + (feature.kind === "mound" ? feature.amount_m : 0), 0));
  const exact: VoxelAabb = {
    min: { x: -scene.container.width_m / 2, y: 0, z: -scene.container.depth_m / 2 },
    max: { x: scene.container.width_m / 2, y: maximumHeight, z: scene.container.depth_m / 2 }
  };
  const source = {
    id: "terrain:heightfield",
    kind: "terrain-heightfield" as const,
    partition: "static" as const,
    materialId: VOXEL_MATERIAL_IDS.terrain,
    composition: "union" as const,
    terrain,
    evaluator: { unionExponent: TERRAIN_UNION_EXPONENT, defaultFlat: TERRAIN_DEFAULT_FLAT, clampMinimum_m: 0 as const },
    candidate: candidateBounds(exact, layout.worldOrigin_m, layout.voxelSize_m, layout.brickSize_m, layout.conservativePadding_m)
  };
  return { ...source, revisionHash: hashValue(source) };
}

function unionBounds(bounds: VoxelAabb[]): VoxelAabb {
  if (bounds.length === 0) return { min: { x: 0, y: 0, z: 0 }, max: { x: 0, y: 0, z: 0 } };
  return bounds.slice(1).reduce((result, value) => ({
    min: { x: Math.min(result.min.x, value.min.x), y: Math.min(result.min.y, value.min.y), z: Math.min(result.min.z, value.min.z) },
    max: { x: Math.max(result.max.x, value.max.x), y: Math.max(result.max.y, value.max.y), z: Math.max(result.max.z, value.max.z) }
  }), bounds[0]);
}

export function planVoxelScene(scene: SceneDescription, options: PlanVoxelSceneOptions = {}): VoxelScenePlan {
  const voxelSize_m = scene.voxelDomain.finestCellSize_m;
  const brickCells = scene.voxelDomain.brickSize_cells;
  if (!(voxelSize_m > 0) || !Number.isFinite(voxelSize_m)) throw new Error("Voxel size must be finite and positive");
  if (!Number.isInteger(brickCells) || brickCells < 2 || brickCells > 32 || (brickCells & (brickCells - 1)) !== 0) {
    throw new Error("Brick cell count must be a power of two in [2, 32]");
  }
  const conservativePadding_m = options.conservativePadding_m ?? voxelSize_m * Math.sqrt(3) / 2;
  if (!(conservativePadding_m >= 0) || !Number.isFinite(conservativePadding_m)) throw new Error("Conservative padding must be finite and non-negative");
  const brickSize_m = voxelSize_m * brickCells;
  const worldOrigin_m = { x: -scene.container.width_m / 2, y: 0, z: -scene.container.depth_m / 2 };
  const interiorBounds_m = {
    min: cloneVec3(worldOrigin_m),
    max: { x: scene.container.width_m / 2, y: scene.container.height_m, z: scene.container.depth_m / 2 }
  };
  const layout: SparseBrickLayoutPlan = {
    voxelSize_m, brickCells, brickSize_m, worldOrigin_m,
    interiorVoxelRange: rangeForBounds(interiorBounds_m, worldOrigin_m, voxelSize_m),
    interiorBrickRange: rangeForBounds(interiorBounds_m, worldOrigin_m, brickSize_m),
    interiorBounds_m,
    conservativePadding_m
  };

  const rigidSources = scene.rigidBodies.map((body) => rigidSource(body, layout));
  const dynamicSources = rigidSources.filter((source) => source.partition === "dynamic");
  const staticSources: VoxelStaticSource[] = [...boundarySources(scene, layout)];
  const terrain = terrainSource(scene, layout);
  if (terrain) staticSources.push(terrain);
  staticSources.push(...rigidSources.filter((source) => source.partition === "static"));

  const materialRevisionHash = hashValue(VOXEL_MATERIALS);
  const hashInputs: VoxelSceneHashInputs = {
    planVersion: VOXEL_SCENE_PLAN_VERSION,
    sceneCanonical: canonicalScene(scene),
    voxelSize_m,
    brickCells,
    conservativePadding_m,
    materialRevisionHash
  };
  const sceneHash = hashValue(hashInputs);
  const staticHash = hashValue(staticSources);
  const dynamicTopologyHash = hashValue(dynamicSources.map(({ topologyRevisionHash, bodyId }) => ({ bodyId, topologyRevisionHash })));
  const dynamicTransformsHash = hashValue(dynamicSources.map(({ transformRevisionHash, bodyId }) => ({ bodyId, transformRevisionHash })));
  return {
    version: VOXEL_SCENE_PLAN_VERSION,
    sceneId: scene.sceneId,
    layout,
    materials: VOXEL_MATERIALS,
    staticSources,
    dynamicSources,
    hashInputs,
    revisions: {
      sceneHash, sceneRevision: numericRevision(sceneHash),
      staticHash, staticRevision: numericRevision(staticHash),
      dynamicTopologyHash, dynamicTopologyRevision: numericRevision(dynamicTopologyHash),
      dynamicTransformsHash, dynamicTransformsRevision: numericRevision(dynamicTransformsHash)
    },
    bounds_m: unionBounds([...staticSources, ...dynamicSources].map((source) => source.candidate.brickAligned_m))
  };
}
