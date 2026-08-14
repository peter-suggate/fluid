/**
 * How many compact pressure rows a scene is allowed to publish.
 *
 * The row arena is fixed at construction and overflow is a rejected GPU
 * generation, never a truncation, so this file chooses between a solver that
 * wastes memory and one that fails closed on a legal frame. Two independent
 * bounds are taken and the smaller wins: a scene-shaped one (interface band
 * plus closed-wall strip plus coarse bulk) and a footprint-shaped one (the
 * authored liquid plus its integrated inflow, with headroom).
 *
 * The terms are as narrow as they are because the wide versions were measured.
 * Summing all three cross-sections modelled an interface that is simultaneously
 * maximal in every orientation and reserved 384k rows against the widened
 * ocean's 25.6k-cell free surface; reserving the closed-wall strip over the
 * full container height reserved air, against 148,600 rows actually published.
 * Splitting the tree is not publishing a row -- only wet leaves reach the
 * arena -- which is why the wall term is bounded by a wettable envelope.
 */
import type { SceneDescription } from "../../core/model";
import { damBreakBoxContains, initialLiquidContainsCell, sceneDamBreakBox } from "../../core/initial-fluid";
import { integratedInflowVolume } from "../../core/inflow-boundary";

export interface OctreePressureCapacityPlan {
  rowCapacity: number;
  pressureBytes: number;
  headerBytes: number;
}

/**
 * The paper's Section 5 interpolant needs a complete local octree
 * neighbourhood wherever a trajectory can sample velocity.  The generated
 * interior Delaunay catalog has no clipped/ghost sites outside the domain, so
 * the bounded production extension keeps closed walls in the regular
 * unit-cell case.  Three cells match the paper's Section 4.3 boundary-band
 * scale; Section 5 requires the advection band to contain the trajectory, so
 * the configured interface support is used whenever it is larger.
 */
export const OCTREE_POWER_BOUNDARY_STRIP_MIN_CELLS = 3;

interface OctreePowerBoundaryStripPlan {
  readonly widthCells: number;
  /** Exact number of finest cells in the union of the selected closed-wall strips. */
  readonly unitCellUpperBound: number;
  /** Exact number of 8-cubed owner pages intersected by that union. */
  readonly ownerPageUpperBound: number;
}

function planOctreePowerBoundaryStrip(
  dims: { nx: number; ny: number; nz: number },
  interfaceBandCells: number,
  closedTop = false,
): OctreePowerBoundaryStripPlan {
  const dimensions = [dims.nx, dims.ny, dims.nz];
  if (dimensions.some((value) => !Number.isSafeInteger(value) || value < 1)) {
    throw new RangeError("Octree power boundary-strip dimensions must be positive safe integers");
  }
  if (!Number.isFinite(interfaceBandCells) || interfaceBandCells < 0) {
    throw new RangeError("Octree power boundary-strip interface band must be finite and non-negative");
  }
  const widthCells = Math.max(OCTREE_POWER_BOUNDARY_STRIP_MIN_CELLS, Math.ceil(interfaceBandCells));
  const lowWidths = [widthCells, widthCells, widthCells];
  const highWidths = [widthCells, closedTop ? widthCells : 0, widthCells];
  const interiorCells = dimensions.map((value, axis) => Math.max(0,
    value - Math.min(value, lowWidths[axis]) - Math.min(value, highWidths[axis])));
  const volume = dimensions[0] * dimensions[1] * dimensions[2];
  const interiorVolume = interiorCells[0] * interiorCells[1] * interiorCells[2];

  const pageDimensions = dimensions.map((value) => Math.ceil(value / 8));
  const interiorPages = dimensions.map((value, axis) => {
    const first = Math.ceil(Math.min(value, lowWidths[axis]) / 8);
    // A partial terminal page is interior when that side is open; with a
    // closed high wall, only complete pages ending before its strip qualify.
    const lastExclusive = highWidths[axis] === 0
      ? Math.ceil(value / 8)
      : Math.floor((value - Math.min(value, highWidths[axis])) / 8);
    return Math.max(0, lastExclusive - first);
  });
  return {
    widthCells,
    unitCellUpperBound: volume - interiorVolume,
    ownerPageUpperBound: pageDimensions[0] * pageDimensions[1] * pageDimensions[2]
      - interiorPages[0] * interiorPages[1] * interiorPages[2],
  };
}

interface OctreeFluidFootprintBudget {
  readonly initialLiquidCells: number;
  readonly inflowLiquidCells: number;
  readonly maximumLiquidCells: number;
  /** Inclusive/exclusive finest-cell bounds of the authored t=0 liquid. */
  readonly minimumCell: readonly [number, number, number];
  readonly maximumCell: readonly [number, number, number];
}

/** Exact authored liquid-volume budget.  This is construction-time work only:
 * it deliberately shares the initial-condition predicates with the bootstrap
 * so capacity follows fluid rather than the surrounding air arena.  Inflow is
 * integrated over the authored scene duration and converted conservatively to
 * finest-cell volumes.  Runtime overflow remains a rejected GPU publication. */
export function planOctreeFluidFootprintBudget(
  scene: SceneDescription,
  dims: { nx: number; ny: number; nz: number },
): OctreeFluidFootprintBudget {
  const dimensions = [dims.nx, dims.ny, dims.nz] as const;
  if (dimensions.some((value) => !Number.isSafeInteger(value) || value < 1)) {
    throw new RangeError("Octree fluid-footprint dimensions must be positive integers");
  }
  const dam = sceneDamBreakBox(scene); let initialLiquidCells = 0;
  const minimum = [dims.nx, dims.ny, dims.nz] as [number, number, number];
  const maximum = [0, 0, 0] as [number, number, number];
  for (let z = 0; z < dims.nz; z += 1) for (let y = 0; y < dims.ny; y += 1) {
    for (let x = 0; x < dims.nx; x += 1) {
      const baseWet = scene.fluid.initialCondition === "dam-break"
        ? damBreakBoxContains(dam, (x + 0.5) / dims.nx, (y + 0.5) / dims.ny,
          (z + 0.5) / dims.nz)
        : (y + 0.5) / dims.ny <= scene.container.fillFraction;
      const wet = initialLiquidContainsCell(scene, x, y, z, dimensions, baseWet);
      if (!wet) continue;
      initialLiquidCells += 1;
      minimum[0] = Math.min(minimum[0], x); minimum[1] = Math.min(minimum[1], y);
      minimum[2] = Math.min(minimum[2], z);
      maximum[0] = Math.max(maximum[0], x + 1); maximum[1] = Math.max(maximum[1], y + 1);
      maximum[2] = Math.max(maximum[2], z + 1);
    }
  }
  if (initialLiquidCells === 0) { minimum.fill(0); maximum.fill(1); }
  const cellVolume = scene.container.width_m * scene.container.height_m
    * scene.container.depth_m / (dims.nx * dims.ny * dims.nz);
  const inflowVolume = scene.fluid.inflow
    ? integratedInflowVolume(scene.fluid.inflow, 0, Math.max(0, scene.duration_s)) : 0;
  const inflowLiquidCells = Math.ceil(Math.max(0, inflowVolume) / cellVolume);
  const maximumLiquidCells = Math.min(dims.nx * dims.ny * dims.nz,
    initialLiquidCells + inflowLiquidCells);
  return Object.freeze({ initialLiquidCells, inflowLiquidCells, maximumLiquidCells,
    minimumCell: Object.freeze(minimum), maximumCell: Object.freeze(maximum) });
}

/**
 * Deformation/topology headroom on the interface band, matching the physical
 * narrow-band plan in `octree-fine-band-capacity`.  Fixed-size row records do
 * not fragment, so this is pure interface-growth reserve.
 */
const OCTREE_PRESSURE_SURFACE_GROWTH_SAFETY = 1.25;
/** Total row headroom over the authored liquid + finite inflow volume.  A
 * rejected generation is the growth signal; no kernel silently truncates to
 * this budget. */
const OCTREE_PRESSURE_FLUID_FOOTPRINT_HEADROOM = 2;

/**
 * A sloshing or collapsing liquid redistributes within at least this fraction
 * of the container, so the wettable envelope never shrinks below it however
 * shallow the authored fill is.
 */
const OCTREE_PRESSURE_WETTABLE_FLOOR_FRACTION = 0.5;

/**
 * Capacity for the compact pressure publication.  The interface contribution
 * is an area-times-width band, the wall term reserves the closed-wall unit
 * strip that can carry rows, and the fully-coarse term covers the calm bulk.
 * Overflow is detected on-GPU and fail-closed; this is a capacity, not an
 * assumption used by the numerical kernels.
 */
export function planOctreePressureCapacity(
  dims: { nx: number; ny: number; nz: number },
  maximumLeafSize: number,
  interfaceBandCells: number,
  override?: number,
  closedTop = false,
  liquidFillFraction = 1,
  rowCapacityLimit = Number.MAX_SAFE_INTEGER,
  fluidFootprint?: OctreeFluidFootprintBudget,
): OctreePressureCapacityPlan {
  const count = dims.nx * dims.ny * dims.nz;
  const aligned = (value: number) => Math.ceil(value / 256) * 256;
  const bandLayers = Math.max(2, Math.ceil(interfaceBandCells) + 2);
  // One connected interface has the area of a single cross-section.  Summing
  // all three modelled a surface that is simultaneously maximal in every
  // orientation, which no single interface can be, and at the widened ocean it
  // reserved 384k rows against a 25.6k-cell free surface.  This is the same
  // area-times-width shape `planGlobalFineNarrowBandBrickCapacity` uses for the
  // physical band, including its explicit deformation headroom.
  const interfaceArea = Math.max(dims.nx * dims.ny, dims.nx * dims.nz, dims.ny * dims.nz);
  const surfaceRows = Math.ceil(interfaceArea * bandLayers * OCTREE_PRESSURE_SURFACE_GROWTH_SAFETY);
  const coarseRows = 8 * Math.ceil(count / Math.max(1, maximumLeafSize ** 3));
  // Power authority currently uses the generated interior catalog.  Reserve
  // the closed-wall unit-strip bound in addition to the moving interface
  // bound; overlap only makes this conservative.  This prevents the
  // correctness strip from silently converting into a row-arena rollback.
  //
  // `powerClosedWallStripIntersects` splits the whole strip to unit owners,
  // but splitting the tree is not publishing a row: only wet leaves reach the
  // row arena, and the strip above the free surface stays dry.  The widened
  // ocean measures 148,600 published rows against a 480,768-cell closed strip,
  // so reserving the strip over the full container height was reserving air.
  // The wettable envelope is the rest waterline plus the strip width and the
  // interface band, and never less than half the container, which covers the
  // authored collapse and seiche cases.  Beyond it the GPU fails closed on
  // arena overflow rather than corrupting the solve.
  const stripWidth = Math.max(OCTREE_POWER_BOUNDARY_STRIP_MIN_CELLS, Math.ceil(interfaceBandCells));
  const clampedFill = Math.max(0, Math.min(1, liquidFillFraction));
  const wettableCellsY = Math.min(dims.ny,
    Math.max(Math.ceil(dims.ny * OCTREE_PRESSURE_WETTABLE_FLOOR_FRACTION),
      Math.ceil(clampedFill * dims.ny) + stripWidth + bandLayers));
  const wallRows = wettableCellsY >= dims.ny
    ? planOctreePowerBoundaryStrip(dims, interfaceBandCells, closedTop).unitCellUpperBound
    : planOctreePowerBoundaryStrip({ nx: dims.nx, ny: wettableCellsY, nz: dims.nz },
      interfaceBandCells, false).unitCellUpperBound;
  const sceneShapedRequest = surfaceRows + wallRows + coarseRows;
  const footprintRequest = fluidFootprint === undefined ? sceneShapedRequest
    : Math.ceil(Math.max(1, fluidFootprint.maximumLiquidCells)
      * OCTREE_PRESSURE_FLUID_FOOTPRINT_HEADROOM);
  const requested = override === undefined
    ? Math.min(sceneShapedRequest, footprintRequest) : override;
  if (!Number.isSafeInteger(rowCapacityLimit) || rowCapacityLimit < 1) {
    throw new RangeError("Octree pressure row-capacity limit must be a positive safe integer");
  }
  const rowCapacity = Math.max(1,
    Math.min(count, rowCapacityLimit, aligned(Math.max(1, Math.floor(requested)))));
  return {
    rowCapacity,
    pressureBytes: rowCapacity * 2 * 4,
    headerBytes: rowCapacity * 48,
  };
}

/**
 * What the device's storage-binding limit allows, as one row count.
 *
 * The row arena is negotiated once per solver and every lane publishes into
 * it, so this ceiling is engine-wide rather than per lane. The arenas that
 * bind it hardest — the packed A/B structured-velocity authority and the
 * SPGrid pyramid — belong to the power backend, which is why the arithmetic
 * cannot live here: `octree-shared` may not import a method. It is installed
 * instead, by the one module that names every method.
 *
 * Deliberately NOT a lane hook. Both lanes are sized against the same ceiling
 * today, so routing it through the lane would raise Losasso's row capacity on
 * every device — a behaviour change wearing a refactor's clothes. A Losasso
 * solver is therefore bounded by arenas it never allocates; that is the
 * existing contract, stated rather than quietly inherited.
 */
export interface OctreeDeviceRowCapacityQuery {
  readonly dimensions: readonly [number, number, number];
  /** `min(maxStorageBufferBindingSize, maxBufferSize)` for the device. */
  readonly maximumStorageBindingBytes: number;
  /** The scene-shaped capacity, before the device gets a say. */
  readonly plannedRowCapacity: number;
}

export type OctreeDeviceRowCapacityCeiling = (query: OctreeDeviceRowCapacityQuery) => number;

let installedDeviceRowCapacityCeiling: OctreeDeviceRowCapacityCeiling | undefined;

export function registerOctreeDeviceRowCapacityCeiling(
  ceiling: OctreeDeviceRowCapacityCeiling,
): void {
  installedDeviceRowCapacityCeiling = ceiling;
}

export function octreeDeviceRowCapacityCeiling(query: OctreeDeviceRowCapacityQuery): number {
  if (!installedDeviceRowCapacityCeiling) {
    throw new Error("No octree device row-capacity ceiling is installed"
      + " -- import the octree method plugin before constructing a solver");
  }
  return installedDeviceRowCapacityCeiling(query);
}
