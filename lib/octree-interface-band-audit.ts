/**
 * Objective audit of the octree's adaptive update criteria against the
 * refinement invariant Losasso et al. (2004) state for water.
 *
 * Section 6 prescribes "adaptive refinement to a band about the interface
 * (focusing more heavily on the water side)", and the section's fast-marching
 * redistancing is only permitted to discard incomplete T-junction directions
 * "since we coarsen as we move away from the interface". Both statements are
 * about leaf SIZE as a function of DISTANCE to the zero set, so both are
 * measurable from a published coarse level-set directory alone:
 *
 *   1. no leaf coarser than one cell may contain the interface, and
 *   2. the coarsest leaf observed must not decrease with distance.
 *
 * At globalFineLevelSetFactor 1 -- the product default, which publishes no
 * fine narrow band -- this directory is the whole surface representation, so a
 * leaf that violates (1) is a facet the renderer cannot avoid drawing: one phi
 * interval spread over size^3 finest cells.
 *
 * The decode mirrors the PowerCoarseSampleDirectory ABI consumed by
 * `reconstructCoarseOnlyOctreeOccupancyField`; it deliberately re-derives the
 * interface membership from the published interval rather than trusting the
 * producer's `containsInterface` flag, and reports both so a disagreement is
 * itself visible.
 */

import { OCTREE_COARSE_PHI_FLAG } from "./octree-coarse-levelset";

/** Header words before the first eight-word row entry. */
export const OCTREE_COARSE_DIRECTORY_HEADER_WORDS = 8;
/** Words per PowerCoarseSampleDirectory entry. */
export const OCTREE_COARSE_DIRECTORY_ROW_WORDS = 8;

export interface OctreeInterfaceBandRow {
  readonly origin: readonly [number, number, number];
  readonly size: number;
  readonly phi: number;
  readonly minimumPhi: number;
  readonly maximumPhi: number;
  /** Smallest |phi| attained anywhere in the row, in finest cell widths. */
  readonly distanceCells: number;
  /** True when the published interval brackets the zero set. */
  readonly straddles: boolean;
  /** "water" when the row is wholly submerged, "air" when wholly dry. */
  readonly side: "water" | "air" | "interface";
}

export interface OctreeInterfaceBandAudit {
  readonly rows: number;
  readonly cellWidth: number;
  readonly bandCells: number;
  /** Leaf-size histogram over rows whose interval brackets the zero set. */
  readonly straddlingRowsBySize: Readonly<Record<string, number>>;
  /**
   * Invariant (1). Rows coarser than one cell that hold the interface: each is
   * one phi interval smeared over size^3 finest cells, which is a facet.
   */
  readonly straddlingCoarseRows: number;
  /** Finest cells covered by those rows -- the extent of the artifact. */
  readonly straddlingCoarseCells: number;
  /** Rows carrying the producer's containsInterface flag, for cross-check. */
  readonly flaggedInterfaceRows: number;
  /**
   * The band rule itself: rows coarser than one cell whose nearest phi lies
   * within `bandCells` of the interface.
   *
   * This, not `straddlingCoarseRows`, is the load-bearing count. The publisher
   * stores one phi per finest row, so a size-1 row has minimum == maximum and
   * an interval that brackets zero is rare even where the surface is; a leaf
   * sitting two cells from the surface at size 2 is the defect regardless.
   */
  readonly bandCoarseRows: number;
  /** The same rows split by side, for section 6's water-side asymmetry. */
  readonly bandCoarseRowsWaterSide: number;
  readonly bandCoarseRowsAirSide: number;
  /** Coarsest leaf observed at each whole-cell distance from the interface. */
  readonly maximumSizeByDistanceCells: readonly number[];
  /** Invariant (2). Distance buckets where the coarsest leaf shrinks again. */
  readonly monotonicityBreaks: number;
  /** Origin-Y profile of the invariant-(1) violations. */
  readonly straddlingCoarseRowsByOriginY: readonly number[];
  /** Worst offenders, largest first, for locating the artifact. */
  readonly worstStraddlingRows: readonly OctreeInterfaceBandRow[];
}

function finiteFloat(words: ArrayLike<number>, index: number): number {
  const value = words[index];
  if (value === undefined) throw new RangeError("Coarse directory word is out of range");
  return new Float32Array(new Uint32Array([value >>> 0]).buffer)[0]!;
}

/**
 * Decode every published row, with its distance to the interface in cells.
 *
 * The row's phi interval is conservative over its volume, so the closest the
 * zero set comes to the row is zero when the interval brackets it and
 * min(|minimum|, |maximum|) otherwise.
 */
export function decodeOctreeInterfaceBandRows(
  directory: ArrayLike<number>,
  dimensions: readonly [number, number, number],
): readonly OctreeInterfaceBandRow[] {
  if (directory.length < OCTREE_COARSE_DIRECTORY_HEADER_WORDS) {
    throw new RangeError("Coarse level-set directory is truncated");
  }
  const rowCount = directory[2] ?? 0;
  const capacity = (directory.length - OCTREE_COARSE_DIRECTORY_HEADER_WORDS)
    / OCTREE_COARSE_DIRECTORY_ROW_WORDS;
  if (!Number.isSafeInteger(capacity) || rowCount < 0 || rowCount > capacity) {
    throw new RangeError("Coarse level-set directory has an invalid row count");
  }
  const cellWidth = finiteFloat(directory, 7);
  if (!(cellWidth > 0)) throw new RangeError("Coarse level-set cell width is invalid");
  const [nx, ny] = dimensions;
  const rows: OctreeInterfaceBandRow[] = [];
  for (let slot = 0; slot < rowCount; slot += 1) {
    const base = OCTREE_COARSE_DIRECTORY_HEADER_WORDS
      + slot * OCTREE_COARSE_DIRECTORY_ROW_WORDS;
    const cellPlusOne = directory[base] ?? 0;
    const size = directory[base + 1] ?? 0;
    const flags = directory[base + 5] ?? 0;
    const required = OCTREE_COARSE_PHI_FLAG.valid | OCTREE_COARSE_PHI_FLAG.finite;
    if (cellPlusOne === 0 || size === 0 || (flags & required) !== required) continue;
    const phi = finiteFloat(directory, base + 2);
    const minimumPhi = finiteFloat(directory, base + 3);
    const maximumPhi = finiteFloat(directory, base + 4);
    if (!Number.isFinite(phi) || !Number.isFinite(minimumPhi)
      || !Number.isFinite(maximumPhi)) continue;
    const cell = cellPlusOne - 1;
    const origin: [number, number, number] = [
      cell % nx, Math.floor(cell / nx) % ny, Math.floor(cell / (nx * ny)),
    ];
    const straddles = minimumPhi <= 0 && maximumPhi >= 0;
    const nearest = straddles ? 0
      : minimumPhi > 0 ? minimumPhi : -maximumPhi;
    rows.push({
      origin, size, phi, minimumPhi, maximumPhi,
      distanceCells: nearest / cellWidth,
      straddles,
      side: straddles ? "interface" : maximumPhi < 0 ? "water" : "air",
    });
  }
  return rows;
}

/**
 * Score a published directory against both refinement invariants.
 *
 * `bandCells` is the authored `interfaceRefinementBandCells`; it only selects
 * which rows the water-side/air-side reach counters observe. The pass/fail
 * gate is `straddlingCoarseRows` and `monotonicityBreaks`, neither of which
 * depends on a tuning dial.
 */
export function auditOctreeInterfaceBand(
  directory: ArrayLike<number>,
  dimensions: readonly [number, number, number],
  bandCells: number,
  worstRowLimit = 8,
): OctreeInterfaceBandAudit {
  if (!(bandCells >= 0)) throw new RangeError("Band width must be non-negative");
  const rows = decodeOctreeInterfaceBandRows(directory, dimensions);
  const cellWidth = finiteFloat(directory, 7);
  const straddlingBySize = new Map<number, number>();
  const straddlingCoarseRowsByOriginY = new Array<number>(dimensions[1]).fill(0);
  const maximumSizeByDistanceCells: number[] = [];
  let straddlingCoarseRows = 0;
  let straddlingCoarseCells = 0;
  let flaggedInterfaceRows = 0;
  let bandCoarseRowsWaterSide = 0;
  let bandCoarseRowsAirSide = 0;
  const straddlingCoarse: OctreeInterfaceBandRow[] = [];
  for (const row of rows) {
    const bucket = Math.min(Math.floor(row.distanceCells), 4096);
    while (maximumSizeByDistanceCells.length <= bucket) maximumSizeByDistanceCells.push(0);
    maximumSizeByDistanceCells[bucket] = Math.max(
      maximumSizeByDistanceCells[bucket]!, row.size,
    );
    if (row.straddles) {
      straddlingBySize.set(row.size, (straddlingBySize.get(row.size) ?? 0) + 1);
      if (row.size > 1) {
        straddlingCoarseRows += 1;
        straddlingCoarseCells += row.size ** 3;
        straddlingCoarse.push(row);
        const y = Math.min(row.origin[1], dimensions[1] - 1);
        if (y >= 0) straddlingCoarseRowsByOriginY[y]! += 1;
      }
    } else if (row.size > 1 && row.distanceCells <= bandCells) {
      if (row.side === "water") bandCoarseRowsWaterSide += 1;
      else bandCoarseRowsAirSide += 1;
    }
  }
  // A producer flag disagreeing with the published interval would make every
  // interval-derived number above suspect, so surface it rather than assume.
  const rowCount = directory[2] ?? 0;
  for (let slot = 0; slot < rowCount; slot += 1) {
    const base = OCTREE_COARSE_DIRECTORY_HEADER_WORDS
      + slot * OCTREE_COARSE_DIRECTORY_ROW_WORDS;
    if (((directory[base + 5] ?? 0) & OCTREE_COARSE_PHI_FLAG.containsInterface) !== 0) {
      flaggedInterfaceRows += 1;
    }
  }
  // Empty buckets carry no evidence either way; compare only observed ones.
  let monotonicityBreaks = 0;
  let observed = 0;
  for (const size of maximumSizeByDistanceCells) {
    if (size === 0) continue;
    if (size < observed) monotonicityBreaks += 1;
    observed = Math.max(observed, size);
  }
  straddlingCoarse.sort((left, right) => right.size - left.size
    || left.origin[1] - right.origin[1]);
  return Object.freeze({
    rows: rows.length,
    cellWidth,
    bandCells,
    straddlingRowsBySize: Object.freeze(Object.fromEntries(
      [...straddlingBySize.entries()].sort(([left], [right]) => left - right)
        .map(([size, count]) => [String(size), count]),
    )),
    straddlingCoarseRows,
    straddlingCoarseCells,
    flaggedInterfaceRows,
    bandCoarseRows: straddlingCoarseRows + bandCoarseRowsWaterSide + bandCoarseRowsAirSide,
    bandCoarseRowsWaterSide,
    bandCoarseRowsAirSide,
    maximumSizeByDistanceCells: Object.freeze(maximumSizeByDistanceCells),
    monotonicityBreaks,
    straddlingCoarseRowsByOriginY: Object.freeze(straddlingCoarseRowsByOriginY),
    worstStraddlingRows: Object.freeze(straddlingCoarse.slice(0, worstRowLimit)),
  });
}
