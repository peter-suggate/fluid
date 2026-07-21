import { damBreakFractions, type DamBreakFractions } from "./initial-fluid";

export type OctreeAnalyticBootstrapCondition = "dam-break" | "tank-fill";
export type OctreeAnalyticBootstrapVec3 = readonly [number, number, number];

/**
 * The complete dyadic 2:1 transition from a finest interface leaf to a
 * maximum-sized leaf travels less than one maximum-leaf topology tile. This
 * is the same 3x3x3 tile dilation used by the recurring GPU residency path.
 */
export const OCTREE_ANALYTIC_BOOTSTRAP_GRADING_HALO_TILES = 1;

export interface OctreeAnalyticBootstrapInput {
  readonly dimensions: OctreeAnalyticBootstrapVec3;
  readonly containerSize: OctreeAnalyticBootstrapVec3;
  readonly tileSizeCells: number;
  readonly initialCondition: OctreeAnalyticBootstrapCondition;
  readonly fillFraction: number;
  readonly interfaceBandCells: number;
  readonly surfaceDetailStrength?: number;
}

export interface OctreeAnalyticBootstrapTileLimits {
  /** Inclusive tile coordinate. */
  readonly minimum: OctreeAnalyticBootstrapVec3;
  /** Exclusive tile coordinate. */
  readonly maximumExclusive: OctreeAnalyticBootstrapVec3;
}

export interface OctreeAnalyticBootstrapBoundsPlan {
  readonly dimensions: OctreeAnalyticBootstrapVec3;
  readonly containerSize: OctreeAnalyticBootstrapVec3;
  readonly cellSize: OctreeAnalyticBootstrapVec3;
  readonly tileSizeCells: number;
  readonly tileDimensions: OctreeAnalyticBootstrapVec3;
  readonly tileCapacity: number;
  readonly initialCondition: OctreeAnalyticBootstrapCondition;
  readonly fillFraction: number;
  readonly damBreak: DamBreakFractions;
  /** Exact upper bound used by the topology sizing predicate. */
  readonly interfaceSupportCells: number;
  readonly interfaceSupportWorld: number;
  readonly gradingHaloTiles: typeof OCTREE_ANALYTIC_BOOTSTRAP_GRADING_HALO_TILES;
  /**
   * Compact production contract. Analytic dam/tank liquid is anchored at the
   * domain minimum, so liquid + interface support + grading is one box.
   */
  readonly activeTileLimits: OctreeAnalyticBootstrapTileLimits;
  readonly activeTileCount: number;
  readonly outsideWorklist: {
    /** All negative analytic liquid is present in activeTileLimits. */
    readonly sign: "non-negative-air";
    /** Missing rows must not become implicit zero before coarse publication. */
    readonly bootstrapAuthority: "analytic-sdf";
    /** Once coarse rows publish, absence has been analytically proven dry. */
    readonly publishedCoarseAuthority: "positive-air";
  };
}

export interface OctreeAnalyticBootstrapPlan extends OctreeAnalyticBootstrapBoundsPlan {
  /** CPU oracle only; production GPU code should generate this from limits. */
  readonly activeTileIndices: Uint32Array<ArrayBuffer>;
  /** Tiles intersecting negative analytic liquid before support dilation. */
  readonly liquidTileIndices: Uint32Array<ArrayBuffer>;
  readonly liquidTileCount: number;
  /** Tiles intersecting the conservative analytic interface support shell. */
  readonly interfaceTileIndices: Uint32Array<ArrayBuffer>;
  readonly interfaceTileCount: number;
}

interface Bounds3 {
  readonly minimum: OctreeAnalyticBootstrapVec3;
  readonly maximum: OctreeAnalyticBootstrapVec3;
}

const UINT32_MAX = 0xffff_ffff;

function validateInput(input: OctreeAnalyticBootstrapInput): void {
  if (input.dimensions.length !== 3 || input.dimensions.some((value) => !Number.isSafeInteger(value) || value < 1)) {
    throw new RangeError("Analytic bootstrap dimensions must be positive safe integers");
  }
  if (input.containerSize.length !== 3 || input.containerSize.some((value) => !Number.isFinite(value) || value <= 0)) {
    throw new RangeError("Analytic bootstrap container size must be positive and finite");
  }
  if (!Number.isSafeInteger(input.tileSizeCells) || input.tileSizeCells < 1
    || (input.tileSizeCells & (input.tileSizeCells - 1)) !== 0) {
    throw new RangeError("Analytic bootstrap tile size must be a positive power of two");
  }
  if (input.initialCondition !== "dam-break" && input.initialCondition !== "tank-fill") {
    throw new RangeError("Analytic bootstrap supports only dam-break and tank-fill");
  }
  if (!Number.isFinite(input.fillFraction) || input.fillFraction < 0 || input.fillFraction > 1) {
    throw new RangeError("Analytic bootstrap fill fraction must be in [0, 1]");
  }
  if (!Number.isFinite(input.interfaceBandCells) || input.interfaceBandCells < 0) {
    throw new RangeError("Analytic bootstrap interface band must be finite and non-negative");
  }
  const detail = input.surfaceDetailStrength ?? 0;
  if (!Number.isFinite(detail) || detail < 0 || detail > 1) {
    throw new RangeError("Analytic bootstrap surface detail strength must be in [0, 1]");
  }
}

function intersects(a: Bounds3, b: Bounds3, inclusive = false): boolean {
  for (let axis = 0; axis < 3; axis += 1) {
    if (inclusive) {
      if (a.maximum[axis] < b.minimum[axis] || a.minimum[axis] > b.maximum[axis]) return false;
    } else if (a.maximum[axis] <= b.minimum[axis] || a.minimum[axis] >= b.maximum[axis]) return false;
  }
  return true;
}

function containedIn(a: Bounds3, b: Bounds3): boolean {
  return [0, 1, 2].every((axis) => a.minimum[axis] >= b.minimum[axis] && a.maximum[axis] <= b.maximum[axis]);
}

function tileBoundsWorld(
  tile: OctreeAnalyticBootstrapVec3,
  input: OctreeAnalyticBootstrapInput,
  cellSize: OctreeAnalyticBootstrapVec3,
): Bounds3 {
  const origin = [-0.5 * input.containerSize[0], 0, -0.5 * input.containerSize[2]] as const;
  const minimum = [0, 1, 2].map((axis) => origin[axis]
    + tile[axis] * input.tileSizeCells * cellSize[axis]) as unknown as OctreeAnalyticBootstrapVec3;
  const maximum = [0, 1, 2].map((axis) => origin[axis]
    + Math.min(input.dimensions[axis], (tile[axis] + 1) * input.tileSizeCells) * cellSize[axis]) as unknown as OctreeAnalyticBootstrapVec3;
  return { minimum, maximum };
}

function analyticBounds(input: OctreeAnalyticBootstrapInput, damBreak: DamBreakFractions): Bounds3 {
  const [width, height, depth] = input.containerSize;
  return input.initialCondition === "tank-fill" ? {
    minimum: [-0.5 * width, 0, -0.5 * depth],
    maximum: [0.5 * width, input.fillFraction * height, 0.5 * depth],
  } : {
    minimum: [-0.5 * width, 0, -0.5 * depth],
    maximum: [
      -0.5 * width + damBreak.width * width,
      damBreak.height * height,
      -0.5 * depth + damBreak.depth * depth,
    ],
  };
}

function expanded(bounds: Bounds3, radius: number): Bounds3 {
  return {
    minimum: bounds.minimum.map((value) => value - radius) as unknown as OctreeAnalyticBootstrapVec3,
    maximum: bounds.maximum.map((value) => value + radius) as unknown as OctreeAnalyticBootstrapVec3,
  };
}

function contracted(bounds: Bounds3, radius: number): Bounds3 | undefined {
  const minimum = bounds.minimum.map((value) => value + radius) as unknown as OctreeAnalyticBootstrapVec3;
  const maximum = bounds.maximum.map((value) => value - radius) as unknown as OctreeAnalyticBootstrapVec3;
  return [0, 1, 2].every((axis) => minimum[axis] < maximum[axis]) ? { minimum, maximum } : undefined;
}

function flatten(tile: OctreeAnalyticBootstrapVec3, tileDimensions: OctreeAnalyticBootstrapVec3): number {
  return tile[0] + tileDimensions[0] * (tile[1] + tileDimensions[1] * tile[2]);
}

function enumerateLimits(limits: OctreeAnalyticBootstrapTileLimits, dimensions: OctreeAnalyticBootstrapVec3): Uint32Array<ArrayBuffer> {
  const values: number[] = [];
  // Match the existing x-major topology worklist ABI exactly.
  for (let z = limits.minimum[2]; z < limits.maximumExclusive[2]; z += 1) {
    for (let y = limits.minimum[1]; y < limits.maximumExclusive[1]; y += 1) {
      for (let x = limits.minimum[0]; x < limits.maximumExclusive[0]; x += 1) values.push(flatten([x, y, z], dimensions));
    }
  }
  return Uint32Array.from(values);
}

function inclusiveTileExtent(
  extentWorld: number,
  cellSize: number,
  tileSizeCells: number,
  tileCapacity: number,
): number {
  // Interface contact is inclusive: when phi=0 or the support edge lies
  // exactly on a tile boundary, both incident tiles are candidates. The +1
  // therefore intentionally differs from a half-open ceil(extent / width).
  return Math.min(tileCapacity, Math.floor(extentWorld / (cellSize * tileSizeCells)) + 1);
}

/**
 * Constant-time production planner. It allocates no domain-sized arrays and
 * does not enumerate topology tiles.
 *
 * Analytic dam/tank liquid begins at the domain minimum. The union of all
 * negative liquid, the conservative interface support shell, and the paper's
 * required grading 1-ring is consequently one clipped rectangular range.
 */
export function planOctreeAnalyticBootstrapBounds(
  input: OctreeAnalyticBootstrapInput,
): OctreeAnalyticBootstrapBoundsPlan {
  validateInput(input);
  const dimensions = [...input.dimensions] as OctreeAnalyticBootstrapVec3;
  const containerSize = [...input.containerSize] as OctreeAnalyticBootstrapVec3;
  const cellSize = dimensions.map((value, axis) => containerSize[axis] / value) as unknown as OctreeAnalyticBootstrapVec3;
  const tileDimensions = dimensions.map((value) => Math.ceil(value / input.tileSizeCells)) as unknown as OctreeAnalyticBootstrapVec3;
  const tileCapacity = tileDimensions[0] * tileDimensions[1] * tileDimensions[2];
  if (!Number.isSafeInteger(tileCapacity) || tileCapacity > UINT32_MAX) {
    throw new RangeError("Analytic bootstrap tile capacity must fit a WebGPU uint");
  }
  const detail = input.surfaceDetailStrength ?? 0;
  const interfaceSupportCells = input.interfaceBandCells + 8 * detail;
  // The sparse worklist is a conservative outer bound for the anisotropic
  // signed-distance classifier. Using the widest physical cell prevents an
  // axis with larger h from escaping a band expressed in finest-grid cells.
  const interfaceSupportWorld = interfaceSupportCells * Math.max(...cellSize);
  if (!Number.isFinite(interfaceSupportWorld)) {
    throw new RangeError("Analytic bootstrap interface support must remain finite");
  }
  const damBreak = damBreakFractions(input.fillFraction);
  const liquidExtent = input.initialCondition === "tank-fill"
    ? [containerSize[0], input.fillFraction * containerSize[1], containerSize[2]] as const
    : [
      damBreak.width * containerSize[0],
      damBreak.height * containerSize[1],
      damBreak.depth * containerSize[2],
    ] as const;
  // Liquid fills everything from the domain minimum to liquidExtent. Its
  // union with the interface shell therefore fills the expanded upper box;
  // no internal shell representation is needed by production.
  const desiredMaximumExclusive = [0, 1, 2].map((axis) => inclusiveTileExtent(
    liquidExtent[axis] + interfaceSupportWorld,
    cellSize[axis],
    input.tileSizeCells,
    tileDimensions[axis],
  )) as unknown as OctreeAnalyticBootstrapVec3;
  const maximumExclusive = desiredMaximumExclusive.map((value, axis) => Math.min(
    tileDimensions[axis],
    value + OCTREE_ANALYTIC_BOOTSTRAP_GRADING_HALO_TILES,
  )) as unknown as OctreeAnalyticBootstrapVec3;
  const activeTileLimits = { minimum: [0, 0, 0] as const, maximumExclusive } as const;
  const activeTileCount = maximumExclusive[0] * maximumExclusive[1] * maximumExclusive[2];
  return {
    dimensions,
    containerSize,
    cellSize,
    tileSizeCells: input.tileSizeCells,
    tileDimensions,
    tileCapacity,
    initialCondition: input.initialCondition,
    fillFraction: input.fillFraction,
    damBreak,
    interfaceSupportCells,
    interfaceSupportWorld,
    gradingHaloTiles: OCTREE_ANALYTIC_BOOTSTRAP_GRADING_HALO_TILES,
    activeTileLimits,
    activeTileCount,
    outsideWorklist: {
      sign: "non-negative-air",
      bootstrapAuthority: "analytic-sdf",
      publishedCoarseAuthority: "positive-air",
    },
  };
}

/** Signed distance used until the first coarse analytic rows are published. */
export function sampleOctreeAnalyticBootstrapPhi(
  input: Pick<OctreeAnalyticBootstrapInput, "containerSize" | "initialCondition" | "fillFraction">,
  point: OctreeAnalyticBootstrapVec3,
): number {
  if (input.initialCondition === "tank-fill") return point[1] - input.fillFraction * input.containerSize[1];
  const damBreak = damBreakFractions(input.fillFraction);
  const bounds = analyticBounds({
    ...input,
    dimensions: [1, 1, 1], tileSizeCells: 1, interfaceBandCells: 0,
  }, damBreak);
  const center = bounds.minimum.map((value, axis) => 0.5 * (value + bounds.maximum[axis]));
  const half = bounds.minimum.map((value, axis) => 0.5 * (bounds.maximum[axis] - value));
  const q = point.map((value, axis) => Math.abs(value - center[axis]) - half[axis]);
  return Math.hypot(Math.max(q[0], 0), Math.max(q[1], 0), Math.max(q[2], 0))
    + Math.min(Math.max(q[0], q[1], q[2]), 0);
}

/**
 * Exact CPU oracle/contracts for the bounded t=0 analytic topology generator.
 *
 * The paper creates fresh sparse storage from interface blocks and their
 * 1-ring. Pressure topology additionally needs every deep-liquid coarse tile:
 * omitting those tiles would lose negative coarse rows outside the fine band.
 * Dam-break and tank-fill are origin-anchored boxes/slabs, so that union and
 * the complete grading halo remain a single clipped rectangular tile range.
 */
export function planOctreeAnalyticBootstrap(input: OctreeAnalyticBootstrapInput): OctreeAnalyticBootstrapPlan {
  const boundsPlan = planOctreeAnalyticBootstrapBounds(input);
  const { cellSize, tileDimensions, damBreak, interfaceSupportWorld } = boundsPlan;
  const liquid = analyticBounds(input, damBreak);
  const outer = expanded(liquid, interfaceSupportWorld);
  const inner = contracted(liquid, interfaceSupportWorld);
  const liquidIndices: number[] = [], interfaceIndices: number[] = [];

  for (let z = 0; z < tileDimensions[2]; z += 1) for (let y = 0; y < tileDimensions[1]; y += 1) {
    for (let x = 0; x < tileDimensions[0]; x += 1) {
      const tile = [x, y, z] as const;
      const bounds = tileBoundsWorld(tile, input, cellSize);
      // Strict overlap means the tile contains an actually negative point;
      // merely touching phi=0 does not invent a liquid pressure row.
      if (intersects(bounds, liquid)) liquidIndices.push(flatten(tile, tileDimensions));
      // Tank phi is a plane rather than the SDF of its wall-bounded liquid
      // slab. Only the horizontal free surface belongs to its interface shell.
      // Dam phi is the authored box SDF; its L-infinity shell is a conservative
      // superset of |boxSdf| <= radius. Inclusive contact also keeps a
      // zero-width interface discoverable.
      const interfaceTile = input.initialCondition === "tank-fill"
        ? bounds.minimum[1] <= liquid.maximum[1] + interfaceSupportWorld
          && bounds.maximum[1] >= liquid.maximum[1] - interfaceSupportWorld
        : intersects(bounds, outer, true) && (!inner || !containedIn(bounds, inner));
      if (interfaceTile) {
        interfaceIndices.push(flatten(tile, tileDimensions));
      }
    }
  }

  const desired = new Set([...liquidIndices, ...interfaceIndices]);
  const active = new Set<number>();
  for (const index of desired) {
    const x = index % tileDimensions[0];
    const y = Math.floor(index / tileDimensions[0]) % tileDimensions[1];
    const z = Math.floor(index / (tileDimensions[0] * tileDimensions[1]));
    for (let dz = -1; dz <= 1; dz += 1) for (let dy = -1; dy <= 1; dy += 1) for (let dx = -1; dx <= 1; dx += 1) {
      const q = [x + dx, y + dy, z + dz] as const;
      if (q[0] >= 0 && q[1] >= 0 && q[2] >= 0
        && q[0] < tileDimensions[0] && q[1] < tileDimensions[1] && q[2] < tileDimensions[2]) {
        active.add(flatten(q, tileDimensions));
      }
    }
  }

  const activeTileIndices = enumerateLimits(boundsPlan.activeTileLimits, tileDimensions);
  // The origin-anchored analytic shapes must stay rectangular after dilation.
  // Failing here prevents a future analytic-shape extension from silently
  // using algebraic compact limits that omit a classified tile or include an
  // unbounded hole.
  if (activeTileIndices.length !== active.size || activeTileIndices.some((index) => !active.has(index))) {
    throw new Error("Analytic bootstrap compact bounds disagree with the enumerated topology oracle");
  }

  return {
    ...boundsPlan,
    activeTileIndices,
    liquidTileIndices: Uint32Array.from(liquidIndices),
    liquidTileCount: liquidIndices.length,
    interfaceTileIndices: Uint32Array.from(interfaceIndices),
    interfaceTileCount: interfaceIndices.length,
  };
}
