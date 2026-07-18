import type { SceneDescription, Vec3 } from "./model";
import {
  mortonEncode3D,
  type SparseBrickCoordinate,
  type SparseBrickSize,
} from "./sparse-brick-octree";

export type SparseSceneProxyCoverage = "volume" | "surface-shell";

export interface SparseSceneWorldAabb {
  min: Vec3;
  max: Vec3;
  /** Overrides the planner default for this proxy. */
  coverage?: SparseSceneProxyCoverage;
}

export interface SparseSceneDomainOptions {
  /** Isotropic in cell counts; converted using each solver-axis cell size. Defaults to zero. */
  conservativePaddingCells?: number;
  /** Default coverage for proxies that do not specify one. */
  proxyCoverage?: SparseSceneProxyCoverage;
  /** Thickness retained on each side of a surface-shell proxy. Defaults to one cell. */
  surfaceShellCells?: number;
}

export interface SparseSceneCellRange {
  min: readonly [number, number, number];
  maxExclusive: readonly [number, number, number];
}

export interface SparseSceneWorldBounds {
  min: Vec3;
  max: Vec3;
}

export interface SparseSceneDomainPlan {
  brickSize: SparseBrickSize;
  /** Solver spacing is retained throughout the scene so simulation and proxy addresses share one lattice. */
  cellSize_m: readonly [number, number, number];
  /** World position of scene cell (0, 0, 0), aligned exactly to the solver cell lattice. */
  worldOrigin_m: Vec3;
  /** Solver cell (0, 0, 0), expressed in non-negative scene-cell coordinates. */
  solverGridOriginCells: readonly [number, number, number];
  solverDimensionsCells: readonly [number, number, number];
  sceneDimensionsCells: readonly [number, number, number];
  brickDimensions: readonly [number, number, number];
  solverBounds_m: SparseSceneWorldBounds;
  worldBounds_m: SparseSceneWorldBounds;
  /** Complete solver-domain cover. */
  solverBrickCoordinates: readonly SparseBrickCoordinate[];
  /** Proxy candidates after removing every brick already covered by the solver. */
  environmentBrickCoordinates: readonly SparseBrickCoordinate[];
  /** Per-proxy candidates, useful for later GPU voxelization; overlaps are intentionally retained here. */
  proxyBrickCoordinates: readonly (readonly SparseBrickCoordinate[])[];
  /** Canonical Morton order, ready for planSparseBrickOctree. */
  coordinates: readonly SparseBrickCoordinate[];
}

type MutableTriple = [number, number, number];

function finite(value: number, label: string): number {
  if (!Number.isFinite(value)) throw new RangeError(`${label} must be finite`);
  return value;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive safe integer`);
  return value;
}

function coordinateKey(value: SparseBrickCoordinate): string {
  return `${value.x},${value.y},${value.z}`;
}

function canonicalCoordinates(values: Iterable<SparseBrickCoordinate>): SparseBrickCoordinate[] {
  const unique = new Map<string, SparseBrickCoordinate>();
  for (const value of values) unique.set(coordinateKey(value), value);
  return [...unique.values()].sort((a, b) => {
    const ma = mortonEncode3D(a.x, a.y, a.z);
    const mb = mortonEncode3D(b.x, b.y, b.z);
    return ma < mb ? -1 : ma > mb ? 1 : 0;
  });
}

function validateProxy(proxy: SparseSceneWorldAabb, index: number): void {
  for (const axis of ["x", "y", "z"] as const) {
    finite(proxy.min[axis], `proxy ${index} minimum ${axis}`);
    finite(proxy.max[axis], `proxy ${index} maximum ${axis}`);
    if (proxy.max[axis] < proxy.min[axis]) throw new RangeError(`proxy ${index} maximum ${axis} must not be below its minimum`);
  }
  if (proxy.coverage !== undefined && proxy.coverage !== "volume" && proxy.coverage !== "surface-shell") {
    throw new RangeError(`proxy ${index} has unsupported coverage`);
  }
}

function worldRangeOnSolverLattice(
  proxy: SparseSceneWorldAabb,
  solverOrigin: readonly [number, number, number],
  cellSize: readonly [number, number, number],
  paddingCells: number,
): SparseSceneCellRange {
  const minima = [proxy.min.x, proxy.min.y, proxy.min.z];
  const maxima = [proxy.max.x, proxy.max.y, proxy.max.z];
  const min: MutableTriple = [0, 0, 0];
  const maxExclusive: MutableTriple = [0, 0, 0];
  for (let axis = 0; axis < 3; axis += 1) {
    min[axis] = Math.floor((minima[axis] - solverOrigin[axis]) / cellSize[axis] - paddingCells);
    maxExclusive[axis] = Math.ceil((maxima[axis] - solverOrigin[axis]) / cellSize[axis] + paddingCells);
    // Preserve zero-thickness analytic proxies even when they lie exactly on a lattice plane.
    if (maxExclusive[axis] <= min[axis]) maxExclusive[axis] = min[axis] + 1;
  }
  return { min, maxExclusive };
}

function enumerateBricks(
  range: SparseSceneCellRange,
  sceneLatticeMinimum: readonly [number, number, number],
  brickSize: SparseBrickSize,
  coverage: SparseSceneProxyCoverage,
  surfaceShellCells: number,
): SparseBrickCoordinate[] {
  const sceneMin = range.min.map((value, axis) => value - sceneLatticeMinimum[axis]) as MutableTriple;
  const sceneMax = range.maxExclusive.map((value, axis) => value - sceneLatticeMinimum[axis]) as MutableTriple;
  const brickMin = sceneMin.map((value) => Math.floor(value / brickSize)) as MutableTriple;
  const brickMax = sceneMax.map((value) => Math.ceil(value / brickSize)) as MutableTriple;
  const innerMin = sceneMin.map((value) => value + surfaceShellCells) as MutableTriple;
  const innerMax = sceneMax.map((value) => value - surfaceShellCells) as MutableTriple;
  const hasInterior = innerMin.every((value, axis) => value < innerMax[axis]);
  const result: SparseBrickCoordinate[] = [];
  for (let z = brickMin[2]; z < brickMax[2]; z += 1) {
    for (let y = brickMin[1]; y < brickMax[1]; y += 1) {
      for (let x = brickMin[0]; x < brickMax[0]; x += 1) {
        if (coverage === "surface-shell" && hasInterior) {
          const brickCellMin = [x * brickSize, y * brickSize, z * brickSize];
          const brickCellMax = brickCellMin.map((value) => value + brickSize);
          const whollyInside = brickCellMin.every((value, axis) => value >= innerMin[axis] && brickCellMax[axis] <= innerMax[axis]);
          if (whollyInside) continue;
        }
        result.push({ x, y, z });
      }
    }
  }
  return canonicalCoordinates(result);
}

/**
 * Plans a single non-negative sparse address space for the solver and the visible scene.
 *
 * The tank is fully resident because fluid kernels need every solver cell. Environment
 * proxies contribute only the bricks intersecting their padded AABBs (or conservative
 * AABB shells), so distant furniture does not allocate the empty room between it and
 * the tank.
 */
export function planSparseSceneDomain(
  scene: SceneDescription,
  solverDimensions: readonly [number, number, number],
  brickSize: SparseBrickSize,
  proxies: readonly SparseSceneWorldAabb[],
  options: SparseSceneDomainOptions = {},
): SparseSceneDomainPlan {
  if (brickSize !== 4 && brickSize !== 8) throw new RangeError("Sparse brick size must be 4 or 8");
  const dimensions = solverDimensions.map((value, axis) => positiveInteger(value, `solver dimension ${axis}`)) as MutableTriple;
  const container = scene.container;
  const extents = [container.width_m, container.height_m, container.depth_m];
  extents.forEach((value, axis) => {
    finite(value, `container extent ${axis}`);
    if (!(value > 0)) throw new RangeError(`container extent ${axis} must be positive`);
  });
  const cellSize = extents.map((value, axis) => value / dimensions[axis]) as MutableTriple;
  const solverOrigin: MutableTriple = [-container.width_m / 2, 0, -container.depth_m / 2];
  const paddingCells = options.conservativePaddingCells ?? 0;
  if (!Number.isFinite(paddingCells) || paddingCells < 0) throw new RangeError("Conservative padding must be finite and non-negative");
  const surfaceShellCells = options.surfaceShellCells ?? 1;
  if (!Number.isFinite(surfaceShellCells) || surfaceShellCells < 0) throw new RangeError("Surface shell thickness must be finite and non-negative");
  const defaultCoverage = options.proxyCoverage ?? "volume";
  if (defaultCoverage !== "volume" && defaultCoverage !== "surface-shell") throw new RangeError("Unsupported proxy coverage");

  const proxyRanges = proxies.map((proxy, index) => {
    validateProxy(proxy, index);
    return worldRangeOnSolverLattice(proxy, solverOrigin, cellSize, paddingCells);
  });
  const latticeMinimum: MutableTriple = [0, 0, 0];
  const latticeMaximum: MutableTriple = [...dimensions];
  for (const range of proxyRanges) {
    for (let axis = 0; axis < 3; axis += 1) {
      latticeMinimum[axis] = Math.min(latticeMinimum[axis], range.min[axis]);
      latticeMaximum[axis] = Math.max(latticeMaximum[axis], range.maxExclusive[axis]);
    }
  }
  // A single brick address must mean the same cell extent to the scene and
  // the fluid page table. Align the shared origin so solver-local 8^3 bricks
  // never straddle two scene leaves when distant proxies extend the domain.
  for (let axis = 0; axis < 3; axis += 1) {
    latticeMinimum[axis] = Math.floor(latticeMinimum[axis] / brickSize) * brickSize;
    latticeMaximum[axis] = Math.ceil(latticeMaximum[axis] / brickSize) * brickSize;
  }
  const solverGridOriginCells = latticeMinimum.map((value) => -value) as MutableTriple;
  const sceneDimensionsCells = latticeMaximum.map((value, axis) => value - latticeMinimum[axis]) as MutableTriple;
  const brickDimensions = sceneDimensionsCells.map((value) => Math.ceil(value / brickSize)) as MutableTriple;
  const worldOrigin = solverOrigin.map((value, axis) => value + latticeMinimum[axis] * cellSize[axis]) as MutableTriple;

  const solverRange: SparseSceneCellRange = {
    min: [0, 0, 0],
    maxExclusive: dimensions,
  };
  const solverBrickCoordinates = enumerateBricks(solverRange, latticeMinimum, brickSize, "volume", 0);
  const solverKeys = new Set(solverBrickCoordinates.map(coordinateKey));
  const proxyBrickCoordinates = proxyRanges.map((range, index) => enumerateBricks(
    range,
    latticeMinimum,
    brickSize,
    proxies[index].coverage ?? defaultCoverage,
    surfaceShellCells,
  ));
  const environmentBrickCoordinates = canonicalCoordinates(
    proxyBrickCoordinates.flat().filter((coordinate) => !solverKeys.has(coordinateKey(coordinate))),
  );
  const coordinates = canonicalCoordinates([...solverBrickCoordinates, ...environmentBrickCoordinates]);
  const solverBounds_m: SparseSceneWorldBounds = {
    min: { x: solverOrigin[0], y: solverOrigin[1], z: solverOrigin[2] },
    max: {
      x: solverOrigin[0] + dimensions[0] * cellSize[0],
      y: solverOrigin[1] + dimensions[1] * cellSize[1],
      z: solverOrigin[2] + dimensions[2] * cellSize[2],
    },
  };
  const worldBounds_m: SparseSceneWorldBounds = {
    min: { x: worldOrigin[0], y: worldOrigin[1], z: worldOrigin[2] },
    max: {
      x: worldOrigin[0] + sceneDimensionsCells[0] * cellSize[0],
      y: worldOrigin[1] + sceneDimensionsCells[1] * cellSize[1],
      z: worldOrigin[2] + sceneDimensionsCells[2] * cellSize[2],
    },
  };

  return {
    brickSize,
    cellSize_m: cellSize,
    worldOrigin_m: { x: worldOrigin[0], y: worldOrigin[1], z: worldOrigin[2] },
    solverGridOriginCells,
    solverDimensionsCells: dimensions,
    sceneDimensionsCells,
    brickDimensions,
    solverBounds_m,
    worldBounds_m,
    solverBrickCoordinates,
    environmentBrickCoordinates,
    proxyBrickCoordinates,
    coordinates,
  };
}
