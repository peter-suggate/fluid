import type { SceneDescription, Vec3 } from "./model";
import {
  SPARSE_BRICK_NUMBER_MORTON_BITS,
  mortonNumber3D,
  mortonNumberCoordinate,
  type SparseBrickCoordinate,
  type SparseBrickSize,
} from "../svo/features/construction/sparse-brick-octree";

export type SparseSceneProxyCoverage = "volume" | "surface-shell";

/**
 * Whether the simulation container is claimed as bricks.
 *
 * `container` is the historical — and only correct — answer while a solver
 * runs: fluid kernels address every cell of the tank, so every brick of it must
 * exist whether or not any geometry is in it. On a world with no solver
 * (`scene.systems.fluid === false`) that claim buys nothing and costs the whole
 * container: on `hero-garden-hose` at 6.25 mm it is 10 368 bricks, of which
 * 3 766 are claimed by nothing else in the scene — 26.6 % of the tree, each one
 * a leaf of dry payload plus a node-mip page. `none` drops it and leaves the
 * environment producers (the ground, the authored proxies, the rigid bodies) to
 * claim what they actually occupy.
 */
export type SparseSceneSolverClaim = "container" | "none";

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
  /** Optional minimum world-space address bounds; sparse proxies may extend beyond them. */
  worldBounds_m?: SparseSceneWorldBounds;
  /**
   * Whether the solver's own container is claimed. Defaults to `container`, so
   * every existing caller keeps the full cover; only a solverless world may ask
   * for `none`.
   */
  solverClaim?: SparseSceneSolverClaim;
  /** GPU planners consume bounds directly and must not enumerate host bricks. */
  enumerateProxyBricks?: boolean;
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
  /** Complete solver-domain cover, or empty when {@link SparseSceneDomainOptions.solverClaim} is `none`. */
  solverBrickCoordinates: readonly SparseBrickCoordinate[];
  /** Proxy candidates after removing every brick already covered by the solver. */
  environmentBrickCoordinates: readonly SparseBrickCoordinate[];
  /** Per-proxy candidates, useful for later GPU voxelization; overlaps are intentionally retained here. */
  proxyBrickCoordinates: readonly (readonly SparseBrickCoordinate[])[];
  /** Exact inclusive-exclusive brick ranges, retaining the lattice rounding. */
  proxyBrickRanges: readonly SparseSceneCellRange[];
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

/** An ascending run of Morton keys with its repeats dropped. */
function uniqueAscending(keys: Float64Array): Float64Array {
  let count = 0;
  for (let index = 0; index < keys.length; index += 1) {
    if (count === 0 || keys[index] !== keys[count - 1]) keys[count++] = keys[index]!;
  }
  return keys.subarray(0, count);
}

/** The coordinates a run of Morton keys denotes, in the run's order. */
function coordinatesOf(keys: Float64Array): SparseBrickCoordinate[] {
  const coordinates: SparseBrickCoordinate[] = new Array(keys.length);
  for (let index = 0; index < keys.length; index += 1) coordinates[index] = mortonNumberCoordinate(keys[index]!);
  return coordinates;
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

/** The inclusive-exclusive brick box a cell range occupies on the scene lattice. */
function brickBox(
  range: SparseSceneCellRange,
  sceneLatticeMinimum: readonly [number, number, number],
  brickSize: SparseBrickSize,
): { min: MutableTriple; maxExclusive: MutableTriple } {
  return {
    min: range.min.map((value, axis) => Math.floor((value - sceneLatticeMinimum[axis]) / brickSize)) as MutableTriple,
    maxExclusive: range.maxExclusive.map((value, axis) => Math.ceil((value - sceneLatticeMinimum[axis]) / brickSize)) as MutableTriple,
  };
}

/**
 * The bricks a cell range claims, as ascending Morton keys.
 *
 * Keys are numbers (`mortonNumber3D`) and a claim is a sorted `Float64Array`:
 * a box's bricks are distinct by construction, so canonical order is one
 * native sort, and the union of every proxy's claim is a sort of their
 * concatenation rather than a string per brick.
 */
function enumerateBricks(
  range: SparseSceneCellRange,
  sceneLatticeMinimum: readonly [number, number, number],
  brickSize: SparseBrickSize,
  coverage: SparseSceneProxyCoverage,
  surfaceShellCells: number,
): Float64Array {
  const sceneMin = range.min.map((value, axis) => value - sceneLatticeMinimum[axis]) as MutableTriple;
  const sceneMax = range.maxExclusive.map((value, axis) => value - sceneLatticeMinimum[axis]) as MutableTriple;
  const { min: brickMin, maxExclusive: brickMax } = brickBox(range, sceneLatticeMinimum, brickSize);
  const innerMin = sceneMin.map((value) => value + surfaceShellCells) as MutableTriple;
  const innerMax = sceneMax.map((value) => value - surfaceShellCells) as MutableTriple;
  const shell = coverage === "surface-shell" && innerMin.every((value, axis) => value < innerMax[axis]);
  const keys = new Float64Array(Math.max(0, brickMax[0] - brickMin[0]) * Math.max(0, brickMax[1] - brickMin[1])
    * Math.max(0, brickMax[2] - brickMin[2]));
  let count = 0;
  for (let z = brickMin[2]; z < brickMax[2]; z += 1) {
    const insideZ = z * brickSize >= innerMin[2] && z * brickSize + brickSize <= innerMax[2];
    for (let y = brickMin[1]; y < brickMax[1]; y += 1) {
      const insideY = insideZ && y * brickSize >= innerMin[1] && y * brickSize + brickSize <= innerMax[1];
      for (let x = brickMin[0]; x < brickMax[0]; x += 1) {
        // A shell drops the bricks wholly inside the proxy's interior.
        if (shell && insideY && x * brickSize >= innerMin[0] && x * brickSize + brickSize <= innerMax[0]) continue;
        keys[count++] = mortonNumber3D(x, y, z);
      }
    }
  }
  return keys.subarray(0, count).sort();
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
  const solverClaim = options.solverClaim ?? "container";
  if (solverClaim !== "container" && solverClaim !== "none") throw new RangeError("Unsupported solver claim");

  const proxyRanges = proxies.map((proxy, index) => {
    validateProxy(proxy, index);
    return worldRangeOnSolverLattice(proxy, solverOrigin, cellSize, paddingCells);
  });
  const latticeMinimum: MutableTriple = [0, 0, 0];
  const latticeMaximum: MutableTriple = [...dimensions];
  const explicitBoundsRange = options.worldBounds_m
    ? worldRangeOnSolverLattice(options.worldBounds_m, solverOrigin, cellSize, 0)
    : undefined;
  for (const range of [...proxyRanges, ...(explicitBoundsRange ? [explicitBoundsRange] : [])]) {
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
  // The lattice above is derived from `dimensions` regardless, so dropping the
  // claim moves no address: the same brick coordinate still names the same
  // cells, the container simply stops being a reason for a brick to exist.
  // Every brick the lattice can name has a number for a Morton key.
  if (brickDimensions.some((value) => value > 2 ** SPARSE_BRICK_NUMBER_MORTON_BITS)) {
    throw new RangeError(`Sparse scene domain of ${brickDimensions.join(" x ")} bricks exceeds 2^${SPARSE_BRICK_NUMBER_MORTON_BITS} an axis`);
  }
  const solverKeys = solverClaim === "none"
    ? new Float64Array(0) : enumerateBricks(solverRange, latticeMinimum, brickSize, "volume", 0);
  const solverBox = brickBox(solverRange, latticeMinimum, brickSize);
  const proxyBrickRanges = proxyRanges.map(range => brickBox(range, latticeMinimum, brickSize));
  const proxyKeys = options.enumerateProxyBricks === false ? [] : proxyRanges.map((range, index) => enumerateBricks(
    range,
    latticeMinimum,
    brickSize,
    proxies[index].coverage ?? defaultCoverage,
    surfaceShellCells,
  ));
  // The union of the proxies' claims, less every brick the solver's box holds.
  const claimed = new Float64Array(proxyKeys.reduce((total, keys) => total + keys.length, 0));
  let claimedCount = 0;
  for (const keys of proxyKeys) { claimed.set(keys, claimedCount); claimedCount += keys.length; }
  const union = uniqueAscending(claimed.sort());
  let environmentCount = 0;
  const environmentKeys = new Float64Array(union.length);
  const environmentBrickCoordinates: SparseBrickCoordinate[] = [];
  for (let index = 0; index < union.length; index += 1) {
    const coordinate = mortonNumberCoordinate(union[index]!);
    if (solverClaim !== "none"
      && coordinate.x >= solverBox.min[0] && coordinate.x < solverBox.maxExclusive[0]
      && coordinate.y >= solverBox.min[1] && coordinate.y < solverBox.maxExclusive[1]
      && coordinate.z >= solverBox.min[2] && coordinate.z < solverBox.maxExclusive[2]) continue;
    environmentKeys[environmentCount++] = union[index]!;
    environmentBrickCoordinates.push(coordinate);
  }
  // Two ascending, disjoint runs as one: the whole claim in canonical order.
  const allKeys = new Float64Array(solverKeys.length + environmentCount);
  for (let i = 0, j = 0, count = 0; count < allKeys.length; count += 1) {
    allKeys[count] = j >= environmentCount || (i < solverKeys.length && solverKeys[i]! < environmentKeys[j]!)
      ? solverKeys[i++]! : environmentKeys[j++]!;
  }
  const solverBrickCoordinates = coordinatesOf(solverKeys);
  const proxyBrickCoordinates = proxyKeys.map(coordinatesOf);
  const coordinates = coordinatesOf(allKeys);
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
    proxyBrickRanges,
    coordinates,
  };
}
