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

/**
 * Sampled ground heights on a regular lattice, for brush sculpting that the
 * 8-feature analytic form cannot express.
 *
 * Heights sit at node positions `origin_m + (i, j) * spacing_m`, row-major
 * `i + nx * j` — the same layout `terrainColumnHeights` bakes, so solvers need
 * no kernel change: they already consume terrain as baked column heights.
 */
export interface TerrainGrid {
  kind: "grid";
  /** World position of sample (0, 0), container-centred like features. */
  origin_m: { x: number; z: number };
  spacing_m: number;
  size: { nx: number; nz: number };
  /** Row-major heights in metres above the container floor. */
  heights_m: number[];
}

/** Bound on an authored grid, keeping documents loadable and JSON sane. */
export const MAX_TERRAIN_GRID_SAMPLES = 262_144;
export const MIN_TERRAIN_GRID_SIZE = 2;

export interface TerrainDescription {
  /** Ground level in metres above the container floor before features. */
  baseHeight_m: number;
  features: TerrainFeature[];
  /**
   * Sculpted heights. When present this *supersedes* `baseHeight_m` and
   * `features` for every height query — the analytic fields are retained only
   * as provenance for the bake that produced the grid.
   */
  grid?: TerrainGrid;
}

/** Bilinear sample with clamped edges; the exact form the WGSL mirror must use. */
export function sampleTerrainGrid(grid: TerrainGrid, x: number, z: number): number {
  const { nx, nz } = grid.size;
  const fx = Math.min(nx - 1, Math.max(0, (x - grid.origin_m.x) / grid.spacing_m));
  const fz = Math.min(nz - 1, Math.max(0, (z - grid.origin_m.z) / grid.spacing_m));
  const x0 = Math.min(nx - 2, Math.floor(fx)), z0 = Math.min(nz - 2, Math.floor(fz));
  const tx = fx - x0, tz = fz - z0;
  const at = (i: number, j: number) => grid.heights_m[i + nx * j] ?? 0;
  const lower = at(x0, z0) * (1 - tx) + at(x0 + 1, z0) * tx;
  const upper = at(x0, z0 + 1) * (1 - tx) + at(x0 + 1, z0 + 1) * tx;
  return Math.max(0, lower * (1 - tz) + upper * tz);
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
  if (terrain.grid) return sampleTerrainGrid(terrain.grid, x, z);
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
  return !!terrain && (terrain.baseHeight_m > 0 || terrain.features.length > 0 || !!terrain.grid);
}

/** True when the ground is sculpted rather than analytic. */
export function terrainIsSculpted(terrain: TerrainDescription | undefined): terrain is TerrainDescription & { grid: TerrainGrid } {
  return !!terrain?.grid;
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

/**
 * One-way bake of the analytic form onto a lattice, so the first brush stroke
 * on an authored preset starts from exactly the ground that was being drawn.
 * The lattice spans the container footprint with a one-sample margin on each
 * side, so edge columns are sampled rather than extrapolated.
 */
export function bakeTerrainGrid(
  terrain: TerrainDescription | undefined,
  container: Pick<SceneDescription["container"], "width_m" | "height_m" | "depth_m">,
  spacing_m: number,
): TerrainGrid {
  if (!(spacing_m > 0)) throw new RangeError("Terrain grid spacing must be positive");
  const nx = Math.max(MIN_TERRAIN_GRID_SIZE, Math.ceil(container.width_m / spacing_m) + 1);
  const nz = Math.max(MIN_TERRAIN_GRID_SIZE, Math.ceil(container.depth_m / spacing_m) + 1);
  if (nx * nz > MAX_TERRAIN_GRID_SAMPLES) throw new RangeError(`Terrain grid ${nx}x${nz} exceeds ${MAX_TERRAIN_GRID_SAMPLES} samples`);
  const origin_m = { x: -0.5 * container.width_m, z: -0.5 * container.depth_m };
  const heights_m = new Array<number>(nx * nz);
  for (let j = 0; j < nz; j += 1) for (let i = 0; i < nx; i += 1) {
    const height = terrainHeightAt(terrain, origin_m.x + i * spacing_m, origin_m.z + j * spacing_m);
    heights_m[i + nx * j] = Math.min(container.height_m, Math.max(0, height));
  }
  return { kind: "grid", origin_m, spacing_m, size: { nx, nz }, heights_m };
}

/**
 * Radial brush stroke. The falloff is the same smoothstep the analytic
 * features use, so a sculpted rim reads like an authored one. Heights stay in
 * [0, container height] — the range every solver's column texture assumes.
 */
export function applyTerrainBrush(
  grid: TerrainGrid,
  centerX_m: number,
  centerZ_m: number,
  radius_m: number,
  delta_m: number,
  maximumHeight_m: number,
): TerrainGrid {
  if (!(radius_m > 0)) return grid;
  const { nx, nz } = grid.size;
  const heights_m = grid.heights_m.slice();
  const first = (value: number, origin: number) => Math.max(0, Math.floor((value - radius_m - origin) / grid.spacing_m));
  const last = (value: number, origin: number, count: number) => Math.min(count - 1, Math.ceil((value + radius_m - origin) / grid.spacing_m));
  for (let j = first(centerZ_m, grid.origin_m.z); j <= last(centerZ_m, grid.origin_m.z, nz); j += 1) {
    for (let i = first(centerX_m, grid.origin_m.x); i <= last(centerX_m, grid.origin_m.x, nx); i += 1) {
      const dx = grid.origin_m.x + i * grid.spacing_m - centerX_m;
      const dz = grid.origin_m.z + j * grid.spacing_m - centerZ_m;
      const distance = Math.hypot(dx, dz) / radius_m;
      if (distance >= 1) continue;
      const s = 1 - distance;
      const weight = s * s * (3 - 2 * s);
      const index = i + nx * j;
      heights_m[index] = Math.min(maximumHeight_m, Math.max(0, heights_m[index] + delta_m * weight));
    }
  }
  return { ...grid, heights_m };
}

function validateTerrainGrid(
  grid: TerrainGrid,
  container: Pick<SceneDescription["container"], "width_m" | "height_m" | "depth_m">,
): string[] {
  const errors: string[] = [];
  if (grid.kind !== "grid") errors.push("Terrain grid kind must be \"grid\"");
  if (!Number.isFinite(grid.origin_m?.x) || !Number.isFinite(grid.origin_m?.z)) errors.push("Terrain grid origin must be finite");
  if (!(grid.spacing_m > 0) || !Number.isFinite(grid.spacing_m)) errors.push("Terrain grid spacing must be positive and finite");
  const nx = grid.size?.nx, nz = grid.size?.nz;
  if (!Number.isInteger(nx) || !Number.isInteger(nz) || nx < MIN_TERRAIN_GRID_SIZE || nz < MIN_TERRAIN_GRID_SIZE) {
    errors.push(`Terrain grid size must be at least ${MIN_TERRAIN_GRID_SIZE} samples per axis`);
    return errors;
  }
  if (nx * nz > MAX_TERRAIN_GRID_SAMPLES) errors.push(`Terrain grid exceeds ${MAX_TERRAIN_GRID_SAMPLES} samples`);
  if (!Array.isArray(grid.heights_m) || grid.heights_m.length !== nx * nz) {
    errors.push("Terrain grid heights must be a row-major array of nx * nz samples");
    return errors;
  }
  if (!grid.heights_m.every((height) => Number.isFinite(height) && height >= 0 && height <= container.height_m)) {
    errors.push("Terrain grid heights must be finite and inside [0, container height]");
  }
  return errors;
}

export function validateTerrain(
  terrain: TerrainDescription,
  container: Pick<SceneDescription["container"], "width_m" | "height_m" | "depth_m">
): string[] {
  const errors: string[] = [];
  if (!(terrain.baseHeight_m >= 0) || terrain.baseHeight_m >= container.height_m) {
    errors.push("Terrain base height must be inside [0, container height)");
  }
  if (terrain.grid) errors.push(...validateTerrainGrid(terrain.grid, container));
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
