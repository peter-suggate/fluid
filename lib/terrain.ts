import type { SceneDescription } from "./model";

/**
 * Analytic terrain heightfield shared by every solver, the renderer and the
 * rigid-body contact solver. The tall-cell paper models the ground as a
 * per-column height H_{i,k}; here the scene carries a compact analytic spec
 * (a base ground level plus smooth elliptical basins and mounds) and each
 * consumer evaluates or bakes the same closed-form height, so the CPU seed,
 * the GPU kernels and the rendered grass always agree exactly.
 *
 * Heights are metres above the container floor (y = 0). A scene without a
 * `terrain` block behaves exactly as before: a flat solid floor at y = 0.
 */

export interface TerrainFeature {
  kind: "basin" | "mound";
  /** Footprint centre in world metres (container-centred x/z, like bodies). */
  center_m: { x: number; z: number };
  /** Elliptical footprint semi-axes in metres. */
  radius_m: { x: number; z: number };
  /** Basin depth or mound height in metres (always positive). */
  amount_m: number;
  /** Footprint rotation about +Y in radians. */
  rotation_rad?: number;
  /** Fraction of the radius that is a flat plateau/floor, in [0, 1). */
  flat?: number;
}

export interface TerrainDescription {
  /** Ground level in metres above the container floor before features. */
  baseHeight_m: number;
  features: TerrainFeature[];
}

/** Uniform-buffer capacity shared with the WGSL evaluators. */
export const MAX_TERRAIN_FEATURES = 8;
/**
 * Overlapping basins merge through a p-norm smooth maximum of their carve
 * depths: exact for a single basin and in the far field (no equality bias),
 * smooth in overlaps with a bounded 2^(1/p) deepening where basins coincide.
 */
export const TERRAIN_UNION_EXPONENT = 8;
export const TERRAIN_DEFAULT_FLAT = 0.45;

function featureWeight(feature: TerrainFeature, x: number, z: number): number {
  const rotation = feature.rotation_rad ?? 0;
  const cos = Math.cos(rotation);
  const sin = Math.sin(rotation);
  const dx = x - feature.center_m.x;
  const dz = z - feature.center_m.z;
  const localX = (cos * dx + sin * dz) / feature.radius_m.x;
  const localZ = (-sin * dx + cos * dz) / feature.radius_m.z;
  const distance = Math.hypot(localX, localZ);
  const flat = feature.flat ?? TERRAIN_DEFAULT_FLAT;
  if (distance <= flat) return 1;
  if (distance >= 1) return 0;
  const s = 1 - (distance - flat) / (1 - flat);
  return s * s * (3 - 2 * s);
}

/** Ground height in metres above the container floor at world (x, z). */
export function terrainHeightAt(terrain: TerrainDescription | undefined, x: number, z: number): number {
  if (!terrain) return 0;
  let mounds = 0;
  let carvePower = 0;
  for (const feature of terrain.features) {
    const weight = featureWeight(feature, x, z);
    if (feature.kind === "mound") mounds += feature.amount_m * weight;
    else carvePower += (feature.amount_m * weight) ** TERRAIN_UNION_EXPONENT;
  }
  const carve = carvePower > 0 ? carvePower ** (1 / TERRAIN_UNION_EXPONENT) : 0;
  return Math.max(0, terrain.baseHeight_m + mounds - carve);
}

/** Outward terrain surface normal at world (x, z), by central differences. */
export function terrainNormalAt(terrain: TerrainDescription | undefined, x: number, z: number, epsilon_m = 0.02) {
  const gradientX = (terrainHeightAt(terrain, x + epsilon_m, z) - terrainHeightAt(terrain, x - epsilon_m, z)) / (2 * epsilon_m);
  const gradientZ = (terrainHeightAt(terrain, x, z + epsilon_m) - terrainHeightAt(terrain, x, z - epsilon_m)) / (2 * epsilon_m);
  const inverseLength = 1 / Math.hypot(gradientX, 1, gradientZ);
  return { x: -gradientX * inverseLength, y: inverseLength, z: -gradientZ * inverseLength };
}

export function sceneHasTerrain(scene: Pick<SceneDescription, "terrain">): boolean {
  const terrain = scene.terrain;
  return !!terrain && (terrain.baseHeight_m > 0 || terrain.features.length > 0);
}

/**
 * Bake per-column ground heights (metres) at cell centres of an nx-by-nz
 * lattice spanning the container footprint. Row-major, index x + nx * z —
 * the layout every solver's column texture uses.
 */
export function terrainColumnHeights(
  scene: { terrain?: TerrainDescription; container: { width_m: number; height_m: number; depth_m: number } },
  nx: number,
  nz: number
): Float32Array {
  const c = scene.container;
  const heights = new Float32Array(nx * nz);
  if (!sceneHasTerrain(scene)) return heights;
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const worldX = -0.5 * c.width_m + (x + 0.5) * (c.width_m / nx);
    const worldZ = -0.5 * c.depth_m + (z + 0.5) * (c.depth_m / nz);
    heights[x + nx * z] = Math.min(c.height_m, terrainHeightAt(scene.terrain, worldX, worldZ));
  }
  return heights;
}

/** Solid fraction of a grid cell cut by the terrain column height. */
export function terrainCellSolidFraction(columnHeight_m: number, cellBottom_m: number, cellSize_m: number): number {
  if (cellSize_m <= 0) return 0;
  return Math.max(0, Math.min(1, (columnHeight_m - cellBottom_m) / cellSize_m));
}

export function validateTerrain(
  terrain: TerrainDescription,
  container: Pick<SceneDescription["container"], "width_m" | "height_m" | "depth_m">
): string[] {
  const errors: string[] = [];
  if (!(terrain.baseHeight_m >= 0) || terrain.baseHeight_m >= container.height_m) {
    errors.push("Terrain base height must be inside [0, container height)");
  }
  if (!Array.isArray(terrain.features)) {
    errors.push("Terrain features must be an array");
    return errors;
  }
  if (terrain.features.length > MAX_TERRAIN_FEATURES) errors.push(`Terrain supports at most ${MAX_TERRAIN_FEATURES} features`);
  for (const feature of terrain.features) {
    if (feature.kind !== "basin" && feature.kind !== "mound") errors.push(`Unsupported terrain feature kind ${String((feature as { kind?: unknown }).kind)}`);
    if (!(feature.amount_m > 0)) errors.push("Terrain feature amount must be positive");
    if (!(feature.radius_m?.x > 0) || !(feature.radius_m?.z > 0)) errors.push("Terrain feature radii must be positive");
    if (feature.flat !== undefined && !(feature.flat >= 0 && feature.flat < 1)) errors.push("Terrain feature flat fraction must be in [0, 1)");
    if (feature.rotation_rad !== undefined && !Number.isFinite(feature.rotation_rad)) errors.push("Terrain feature rotation must be finite");
    if (feature.kind === "basin" && feature.amount_m > terrain.baseHeight_m + 1e-9) {
      errors.push("Terrain basin depth cannot exceed the base ground height");
    }
  }
  return errors;
}
