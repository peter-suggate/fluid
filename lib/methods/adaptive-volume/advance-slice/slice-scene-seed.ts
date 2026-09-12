import type { SparseAdaptiveMassAtlas } from "../sparse-brick-atlas";
import type { FluidRefinementRegion, SceneDescription } from "../../../core/model";
import type { MethodParamValues } from "../../../core/method-contract";
import type { SolidWorld } from "../../../core/solid-world";
import type { AdaptiveMassSolverOptions } from "../method";

/**
 * Immutable, finest-lattice input to the two-dimensional Sparse CM12 proxy.
 *
 * Density and capacity use the production scalar convention: both are
 * fractions of a full finest cell.  The physical liquid area represented by
 * one entry is `density[i] * cellSizeX * cellSizeY`; fill is density/capacity.
 */
export interface SliceSceneViewport {
  readonly originX: number;
  readonly originY: number;
  readonly cellSizeX: number;
  readonly cellSizeY: number;
  readonly sourceCellSize: number;
  readonly centerCellZ: number;
  /** Physical centre of the containing production voxel. */
  readonly sourceCellCenterZ: number;
  /** Exact geometric centre plane requested by the lab. */
  readonly centerZ: number;
}

export interface SliceSceneDynamicMetadata {
  readonly kind: "rigid" | "source";
  readonly label: string;
  readonly supported: boolean;
  readonly detail?: string;
  readonly payload?: Readonly<Record<string, unknown>>;
}

export interface SliceProductionContext {
  readonly scene: SceneDescription;
  readonly values: MethodParamValues;
  readonly options: AdaptiveMassSolverOptions;
  readonly initiallyActiveBrickKeys: ReadonlySet<number>;
  readonly solidWorld: SolidWorld;
}

export interface SliceSceneSeed {
  readonly id: string;
  readonly label: string;
  readonly note: string;
  readonly dimensions: readonly [number, number];
  readonly boundary: {
    readonly xMin: "closed" | "open";
    readonly xMax: "closed" | "open";
    readonly yMin: "closed" | "open";
    readonly yMax: "closed" | "open";
    readonly z: "symmetry" | "omitted";
  };
  readonly viewport: SliceSceneViewport;
  /** Static/open fraction at t=0, including the centre-z rigid intersection. */
  readonly capacity: Float32Array;
  /** Liquid volume divided by full cell volume, matching production rho. */
  readonly density: Float32Array;
  /** Staggered x face velocity in m/s, `(nx + 1) * ny` values. */
  readonly velocityX: Float32Array;
  /** Staggered canvas-down y face velocity in m/s, `nx * (ny + 1)` values. */
  readonly velocityY: Float32Array;
  readonly apertureX?: Float32Array;
  readonly apertureY?: Float32Array;
  readonly solidVelocityX?: Float32Array;
  readonly solidVelocityY?: Float32Array;
  /** Optional source z velocity retained for dimensional-reduction disclosure. */
  readonly velocityZ?: Float32Array;
  /** Cell material identifiers sampled from the production solid world. */
  readonly materialId: Uint16Array;
  /** Exact production sparse-atlas generation from which this plane was cut. */
  readonly sourceAtlas?: SparseAdaptiveMassAtlas;
  readonly production?: SliceProductionContext;
  /**
   * Enforcement regions this cut obeys, overriding the production document's
   * own when present.
   *
   * Absent means "whatever the scene authored", which is how every seed is
   * built; a lab that draws one writes the whole list here rather than into
   * `production.scene`, because the production document is also the thing the
   * seed was *sampled from* and editing it would make the run disagree with
   * the world it came out of. Read through `sliceSceneRegions`, never off either field directly.
   */
  readonly refinementRegions?: readonly FluidRefinementRegion[];
  /** Acceleration in the 2-D x/canvas-down-y frame, m/s^2. */
  readonly gravity: readonly [number, number];
  readonly dt: number;
  readonly densityKgM3: number;
  readonly dynamic?: readonly SliceSceneDynamicMetadata[];
  readonly limitations?: readonly string[];
}

export function validateSliceSceneSeed(seed: SliceSceneSeed): void {
  const [nx, ny] = seed.dimensions;
  if (!Number.isInteger(nx) || !Number.isInteger(ny) || nx <= 0 || ny <= 0) {
    throw new RangeError("slice scene dimensions must be positive integers");
  }
  const cells = nx * ny;
  if (seed.capacity.length !== cells || seed.density.length !== cells
    || seed.materialId.length !== cells
    || seed.velocityX.length !== (nx + 1) * ny
    || seed.velocityY.length !== nx * (ny + 1)) {
    throw new RangeError("slice scene fields do not match its native dimensions");
  }
  const { cellSizeX, cellSizeY, sourceCellSize } = seed.viewport;
  if (!(seed.dt > 0) || !Number.isFinite(seed.dt)
    || !(cellSizeX > 0) || !(cellSizeY > 0) || !(sourceCellSize > 0)
    || !Number.isFinite(cellSizeX) || !Number.isFinite(cellSizeY)
    || !Number.isFinite(sourceCellSize)) {
    throw new RangeError("slice scene spacing and dt must be finite and positive");
  }
  // Production Sparse CM12 has one cubic finest-cell length.  Silently using
  // rectangular cells would change trace CFL, pressure coefficients and PLIC.
  const spacingTolerance = 16 * Number.EPSILON
    * Math.max(cellSizeX, cellSizeY, sourceCellSize);
  if (Math.abs(cellSizeX - sourceCellSize) > spacingTolerance
    || Math.abs(cellSizeY - sourceCellSize) > spacingTolerance) {
    throw new RangeError("slice proxy requires production-square finest cells");
  }
  for (let i = 0; i < cells; i += 1) {
    const capacity = seed.capacity[i]!;
    const density = seed.density[i]!;
    if (!(capacity >= 0 && capacity <= 1) || !(density >= 0)
      || !Number.isFinite(capacity) || !Number.isFinite(density)) {
      throw new RangeError(`invalid slice scalar at cell ${i}`);
    }
    // Do not reject density above capacity: production deliberately preserves
    // CM12 excess state and reports it instead of clamping the authority.
  }
}

/** Deterministic fallback used only by direct numerical tests and first load. */
export function createDefaultSliceSceneSeed(
  dimensions: readonly [number, number] = [96, 40],
): SliceSceneSeed {
  const [nx, ny] = dimensions;
  const capacity = new Float32Array(nx * ny);
  const density = new Float32Array(nx * ny);
  const materialId = new Uint16Array(nx * ny);
  for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    const i = y * nx + x;
    const open = x > 0 && x + 1 < nx && y > 0 && y + 1 < ny;
    capacity[i] = open ? 1 : 0;
    density[i] = open && x < Math.max(2, Math.floor(nx / 4)) ? 1 : 0;
    materialId[i] = open ? 0 : 1;
  }
  const cellSize = 0.05;
  return {
    id: "fallback-dam-break",
    label: "Fallback dam break",
    note: "Deterministic direct-test seed used before a production scene is selected.",
    dimensions,
    boundary: { xMin: "closed", xMax: "closed", yMin: "closed", yMax: "closed", z: "symmetry" },
    viewport: {
      originX: 0, originY: 0, cellSizeX: cellSize, cellSizeY: cellSize,
      sourceCellSize: cellSize, centerCellZ: 0, sourceCellCenterZ: 0, centerZ: 0,
    },
    capacity, density,
    velocityX: new Float32Array((nx + 1) * ny),
    velocityY: new Float32Array(nx * (ny + 1)),
    materialId,
    gravity: [0, 9.81], dt: 1 / 60, densityKgM3: 998.2,
  };
}
