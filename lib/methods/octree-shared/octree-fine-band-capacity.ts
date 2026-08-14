/**
 * Resident brick capacity for the domain-global fine level-set band.
 *
 * The fine grid is never materialized. It is a shell around one interface, so
 * every plan here is area times band width rather than volume -- which is what
 * makes growing all three logical dimensions cost quadratically instead of
 * cubically, and is the whole reason the two-grid construction of Aanjaneya et
 * al. 2017 Section 5 is affordable at all.
 *
 * Two planners exist because two different facts can bound the same band. The
 * scene-wide plan takes the largest cross-section, which is right for a
 * sloshing tank and wildly pessimistic for a droplet; the footprint plan takes
 * the authored liquid box's exposed faces instead. Its companion floor exists
 * because the area estimate silently drops the edge and corner terms of the
 * first dilation: an 8-cubed drop dilated six bricks occupies a clipped box
 * shell, not six unrelated face slabs, and the GPU publisher discovers that
 * volume exactly and rejects a pool sized from the slabs.
 *
 * `resolveGlobalFineBrickCapacity` has the last word, and it refuses rather
 * than silently shrinks: a physical plan the device cannot bind or dispatch is
 * an error, because quietly reducing the band would quietly lose surface.
 */

/** Resolve a physically planned capacity against 2D dispatch and binding limits. */
export function resolveGlobalFineBrickCapacity(
  defaultCapacity: number,
  override: number | undefined,
  maximumWorkgroupsPerDimension: number,
  transportWorkgroupQuantum = 64,
  maximumStorageBufferBindingSize = Number.MAX_SAFE_INTEGER,
  samplesPerBrick = 64,
  summaryLevelCount = 1,
  exactSummaryEntryCapacity?: number,
): number {
  if (!Number.isSafeInteger(defaultCapacity) || defaultCapacity < 1
    || !Number.isSafeInteger(maximumWorkgroupsPerDimension) || maximumWorkgroupsPerDimension < 1
    || !Number.isSafeInteger(transportWorkgroupQuantum) || transportWorkgroupQuantum < 1
    || !Number.isSafeInteger(maximumStorageBufferBindingSize) || maximumStorageBufferBindingSize < 16
    || !Number.isSafeInteger(samplesPerBrick) || samplesPerBrick < 1
    || !Number.isSafeInteger(summaryLevelCount) || summaryLevelCount < 1
    || (exactSummaryEntryCapacity !== undefined
      && (!Number.isSafeInteger(exactSummaryEntryCapacity) || exactSummaryEntryCapacity < 1))) {
    throw new RangeError("Global fine level-set default/device capacities must be positive integers");
  }
  // Per-brick work is tiled over x/y; dispatch shape no longer truncates the
  // physical capacity. Bound the largest persistent buffers instead: one
  // four-byte fine channel per sample and the compact sorted summary directory containing
  // at most one entry per resident brick per hierarchy level at load <= 0.5.
  const twoDimensionalDispatchMaximum = maximumWorkgroupsPerDimension ** 2;
  const payloadSafe = Math.floor(maximumStorageBufferBindingSize / (samplesPerBrick * 4));
  let summaryHashSlots = 1;
  while ((summaryHashSlots * 2) * 32 + 64 <= maximumStorageBufferBindingSize) summaryHashSlots *= 2;
  const summarySafe = exactSummaryEntryCapacity === undefined
    ? Math.floor(summaryHashSlots / (2 * summaryLevelCount))
    : Number.MAX_SAFE_INTEGER;
  const rawDeviceMaximum = Math.min(twoDimensionalDispatchMaximum, payloadSafe, summarySafe);
  const deviceMaximum = Math.floor(rawDeviceMaximum / transportWorkgroupQuantum) * transportWorkgroupQuantum;
  const configured = override ?? defaultCapacity;
  if (!Number.isSafeInteger(configured) || configured < 1) {
    throw new RangeError("Global fine level-set brick capacity must be a positive integer");
  }
  if (configured > deviceMaximum) {
    throw new RangeError(`Global fine level-set brick capacity ${configured} exceeds the sparse binding/dispatch limit ${deviceMaximum}; the physical narrow-band estimate is not reduced implicitly`);
  }
  if (exactSummaryEntryCapacity !== undefined) {
    let exactHashCapacity = 1;
    while (exactHashCapacity < exactSummaryEntryCapacity * 2) exactHashCapacity *= 2;
    if (64 + exactHashCapacity * 32 > maximumStorageBufferBindingSize) {
      throw new RangeError(`Global fine sparse summary requires ${64 + exactHashCapacity * 32} bytes, exceeding the storage binding limit ${maximumStorageBufferBindingSize}`);
    }
  }
  return configured;
}

interface GlobalFineNarrowBandCapacityPlan {
  readonly logicalBrickCount: number;
  readonly maximumInterfaceAreaBricks: number;
  readonly bandLayers: number;
  readonly bandBrickCount: number;
  readonly surfaceGrowthSafety: number;
  readonly surfaceGrowthHeadroomBricks: number;
  readonly maximumResidentBricks: number;
}

/**
 * Power's restored factor-4 surface grid uses the shared candidate-local
 * publisher. Two-times interface-area headroom is the smallest round policy
 * that admits the complete legal band in the compact symmetric-expansion
 * domain; the capacity planner still clamps at the logical brick count.
 */
export const POWER2017_FINE_BAND_SURFACE_GROWTH_SAFETY = 2;

/** Footprint-specialized version of the physical band plan.  Only the three
 * exposed faces of the authored liquid box are needed for a corner/tank seed;
 * the deformation safety and inflow surface term are explicit headroom, not a
 * numerical assumption. */
export function planFluidFootprintFineNarrowBandBrickCapacity(
  logicalBrickDimensions: readonly [number, number, number],
  footprintBrickDimensions: readonly [number, number, number],
  dilationBrickRings: number,
  inflowBrickBudget = 0,
  surfaceGrowthSafety = 1.5,
): GlobalFineNarrowBandCapacityPlan {
  const base = planGlobalFineNarrowBandBrickCapacity(logicalBrickDimensions,
    dilationBrickRings, surfaceGrowthSafety);
  if (footprintBrickDimensions.some((value) => !Number.isSafeInteger(value) || value < 1)
    || !Number.isSafeInteger(inflowBrickBudget) || inflowBrickBudget < 0) {
    throw new RangeError("Fluid-footprint fine-band dimensions are invalid");
  }
  const [x, y, z] = footprintBrickDimensions;
  const authoredArea = x * y + x * z + y * z;
  const inflowArea = inflowBrickBudget === 0 ? 0
    : Math.ceil(3 * inflowBrickBudget ** (2 / 3));
  const maximumInterfaceAreaBricks = Math.min(base.logicalBrickCount,
    authoredArea + inflowArea);
  const bandBrickCount = Math.min(base.logicalBrickCount,
    maximumInterfaceAreaBricks * base.bandLayers);
  const maximumResidentBricks = Math.min(base.logicalBrickCount,
    Math.ceil(bandBrickCount * surfaceGrowthSafety));
  return Object.freeze({ ...base, maximumInterfaceAreaBricks, bandBrickCount,
    surfaceGrowthHeadroomBricks: maximumResidentBricks - bandBrickCount,
    maximumResidentBricks });
}

/**
 * Exact publication floor for a rectangular authored liquid footprint.
 *
 * The rolling area-times-width estimate above is the right asymptotic reserve,
 * but it omits the edge and corner terms of the first Chebyshev dilation. That
 * omission is material for compact seeds: an 8-cubed drop with a six-brick
 * publication radius occupies a clipped box-shell volume, not six unrelated
 * face slabs. The GPU publisher discovers that volume exactly and rejects the
 * generation when the page pool was sized only from the slab estimate. Callers
 * include the extra transport membership ring when planning recurring updates.
 *
 * Bounds are half-open in logical fine-brick coordinates. The returned shell
 * is the largest translated expanded box that fits in the domain minus the
 * box interior farther than `dilationBrickRings` from every authored face.
 * Planning the initial clipped position is insufficient: a lid- or wall-flush
 * seed consumes the un-clipped envelope as soon as it separates.
 */
export function planFluidFootprintFineBandBrickFloor(
  logicalBrickDimensions: readonly [number, number, number],
  footprintMinimumBrick: readonly [number, number, number],
  footprintMaximumBrick: readonly [number, number, number],
  dilationBrickRings: number,
): number {
  if (logicalBrickDimensions.some((value) => !Number.isSafeInteger(value) || value < 1)
    || !Number.isSafeInteger(dilationBrickRings) || dilationBrickRings < 1
    || footprintMinimumBrick.some((value, axis) => !Number.isSafeInteger(value)
      || value < 0 || value >= footprintMaximumBrick[axis]!)
    || footprintMaximumBrick.some((value, axis) => !Number.isSafeInteger(value)
      || value > logicalBrickDimensions[axis]!)) {
    throw new RangeError("Fluid-footprint fine-band bounds are invalid");
  }
  const outerExtents = logicalBrickDimensions.map((dimension, axis) => Math.min(
    dimension,
    footprintMaximumBrick[axis]! - footprintMinimumBrick[axis]!
      + 2 * dilationBrickRings,
  ));
  const innerExtents = logicalBrickDimensions.map((_dimension, axis) => Math.max(0,
    footprintMaximumBrick[axis]! - footprintMinimumBrick[axis]!
      - 2 * dilationBrickRings));
  const volume = (extents: readonly number[]) => extents.reduce((product, value) => {
    const next = product * value;
    if (!Number.isSafeInteger(next)) {
      throw new RangeError("Fluid-footprint fine-band volume exceeds exact integer range");
    }
    return next;
  }, 1);
  return volume(outerExtents) - volume(innerExtents);
}

/**
 * Physical single-interface narrow-band capacity, in global fine bricks.
 *
 * This is deliberately an area-times-width plan. Increasing all logical
 * dimensions while holding the physical brick-band width fixed grows the
 * reserve quadratically rather than materializing the cubic fine lattice.
 * `surfaceGrowthSafety` is explicit deformation/topology headroom; fixed-size
 * physical pages themselves do not incur allocator fragmentation. The 1.5
 * default preserves the evolving fine SPGrid required by Aanjaneya et al.
 * 2017 Section 5 (`docs/papers/aanjaneya-2017-power-liquids.txt`); their
 * two-grid construction updates the fine surface grid every advection step
 * and does not assume that its interface remains planar.
 */
export function planGlobalFineNarrowBandBrickCapacity(
  brickDimensions: readonly [number, number, number],
  dilationBrickRings: number,
  surfaceGrowthSafety = 1.5,
): GlobalFineNarrowBandCapacityPlan {
  if (brickDimensions.some((value) => !Number.isSafeInteger(value) || value < 1)
    || !Number.isSafeInteger(dilationBrickRings) || dilationBrickRings < 1
    || !Number.isFinite(surfaceGrowthSafety) || surfaceGrowthSafety < 1) {
    throw new RangeError("Global fine narrow-band estimate inputs are invalid");
  }
  const [x, y, z] = brickDimensions;
  const logicalBrickCount = x * y * z;
  const maximumInterfaceAreaBricks = Math.max(x * y, x * z, y * z);
  const bandLayers = 2 * dilationBrickRings + 1;
  if (![logicalBrickCount, maximumInterfaceAreaBricks, bandLayers,
    maximumInterfaceAreaBricks * bandLayers].every(Number.isSafeInteger)) {
    throw new RangeError("Global fine narrow-band estimate exceeds exact integer range");
  }
  const bandBrickCount = Math.min(logicalBrickCount, maximumInterfaceAreaBricks * bandLayers);
  const plannedWithHeadroom = Math.ceil(bandBrickCount * surfaceGrowthSafety);
  const maximumResidentBricks = Math.min(logicalBrickCount, plannedWithHeadroom);
  return {
    logicalBrickCount, maximumInterfaceAreaBricks, bandLayers, bandBrickCount,
    surfaceGrowthSafety,
    surfaceGrowthHeadroomBricks: maximumResidentBricks - bandBrickCount,
    maximumResidentBricks,
  };
}

/** Scalar form of the physical narrow-band plan. */
export function estimateGlobalFineNarrowBandBrickCapacity(
  brickDimensions: readonly [number, number, number],
  dilationBrickRings: number,
  surfaceGrowthSafety = 1.5,
): number {
  return planGlobalFineNarrowBandBrickCapacity(
    brickDimensions, dilationBrickRings, surfaceGrowthSafety,
  ).maximumResidentBricks;
}
