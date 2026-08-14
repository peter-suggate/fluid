import type { SceneDescription } from "../core/model";
import { INITIAL_FLUID_BRICK_SIZE } from "../core/initial-fluid";

export function freeFallOracle(scene: SceneDescription, grid: readonly [number, number, number]) {
  const gravity = scene.fluid.gravity_m_s2;
  const g = Math.hypot(gravity.x, gravity.y, gravity.z);
  const cell_m = scene.container.height_m / grid[1];
  const seed = scene.fluid.initialBrickSeeds_m?.[0];
  const seedCellY = seed
    ? Math.min(grid[1] - 1, Math.max(0, Math.floor(seed.y / scene.container.height_m * grid[1])))
    : grid[1] - 1;
  const originY = Math.floor(seedCellY / INITIAL_FLUID_BRICK_SIZE) * INITIAL_FLUID_BRICK_SIZE;
  const initialCentroidY_cells = originY + INITIAL_FLUID_BRICK_SIZE / 2;
  const drop_m = originY * cell_m;
  const impact_s = Math.sqrt(2 * drop_m / g);
  return {
    g, cell_m, initialCentroidY_cells, impact_s,
    centroidY_cells: (time_s: number) =>
      initialCentroidY_cells - 0.5 * g * time_s * time_s / cell_m,
  };
}

/**
 * Per-cell attribution for the free-fall drop oracles.
 *
 * These scenes are the only ones whose exact solution is known at every cell
 * rather than in aggregate: until impact every liquid cell moves at exactly
 * g t along gravity with no lateral component, and because the seeded body is
 * a cube every column falls identically. Two independent defects therefore
 * separate the stages that can hold liquid against a wall:
 *
 *  - a velocity shortfall (the cell is slower than free fall) means the
 *    projection or the Section 5 velocity extension is holding it;
 *  - a correct velocity under a lagging column means the velocity is right and
 *    the surface tracker (fine transport / redistance) is not following it.
 *
 * Bucketing by how many closed domain walls a cell touches localizes the
 * defect: 0 interior, 1 a flat wall, 2 a seam, 3 the corner. Columns bucket by
 * their *vertical* wall count, which is what "fluid climbs the corners" means
 * — gravity is tangential there, so a free-slip wall cannot change a column's
 * fall at all and any lag is numerical.
 */
export function freeFallContactAttribution(
  alpha: Float32Array,
  velocity: Float32Array | undefined,
  grid: readonly [number, number, number],
  gravity: readonly [number, number, number],
  time_s: number,
  analyticCentroidY_cells: number,
  seedFootprint?: Readonly<{ originX: number; originZ: number; size: number }>,
  fineUpperSurfaceField?: Float32Array,
) {
  const [nx, ny, nz] = grid;
  const g = Math.hypot(gravity[0], gravity[1], gravity[2]);
  const unit: readonly [number, number, number] = g > 0
    ? [gravity[0] / g, gravity[1] / g, gravity[2] / g] : [0, -1, 0];
  const expectedSpeed = g * time_s;
  interface CellBucket {
    contacts: number; wetCells: number; sampledCells: number;
    shortfallSum: number; maximumShortfall: number; lateralSum: number; maximumLateral: number;
  }
  interface ColumnBucket { verticalWalls: number; columns: number; lagSum: number; maximumLag: number }
  // Same samples, split the other way: a wall whose outward normal opposes
  // gravity (the lid) is the only one that can physically hold liquid up, so
  // separating "touches the lid" from "touches N vertical walls" tells an
  // interaction between the two apart from either acting alone.
  interface OverheadBucket {
    overhead: boolean; verticalWalls: number; wetCells: number; sampledCells: number; shortfallSum: number;
  }
  const overheadBuckets = new Map<string, OverheadBucket>();
  const cells = new Map<number, CellBucket>();
  const columns = new Map<number, ColumnBucket>();
  let worstColumn: { i: number; k: number; verticalWalls: number; lag_cells: number } | undefined;
  let bestColumn: { i: number; k: number; verticalWalls: number; lag_cells: number } | undefined;
  let worstCell: { i: number; j: number; k: number; contacts: number; shortfallFraction: number } | undefined;
  let bestCell: { i: number; j: number; k: number; contacts: number; shortfallFraction: number } | undefined;
  let footprintRimLag = 0, footprintRimColumns = 0;
  let footprintCenterLag = 0, footprintCenterColumns = 0;
  let footprintTopCenter = 0, footprintTopCenterColumns = 0;
  let footprintTopShoulder = 0, footprintTopShoulderColumns = 0;
  let footprintTopCenterSpeed = 0, footprintTopCenterSpeedColumns = 0;
  let footprintTopShoulderSpeed = 0, footprintTopShoulderSpeedColumns = 0;
  for (let k = 0; k < nz; k += 1) for (let i = 0; i < nx; i += 1) {
    let columnAmount = 0, weightedY = 0;
    let topCell = -1;
    for (let j = 0; j < ny; j += 1) {
      const amount = alpha[i + nx * (j + ny * k)] ?? 0;
      if (!(amount > 0)) continue;
      topCell = j;
      columnAmount += amount; weightedY += amount * (j + 0.5);
      if (!(amount >= 0.5)) continue;
      const contacts = (i === 0 ? 1 : 0) + (i === nx - 1 ? 1 : 0) + (j === 0 ? 1 : 0)
        + (j === ny - 1 ? 1 : 0) + (k === 0 ? 1 : 0) + (k === nz - 1 ? 1 : 0);
      let bucket = cells.get(contacts);
      if (!bucket) {
        bucket = { contacts, wetCells: 0, sampledCells: 0, shortfallSum: 0,
          maximumShortfall: -Infinity, lateralSum: 0, maximumLateral: 0 };
        cells.set(contacts, bucket);
      }
      bucket.wetCells += 1;
      if (!velocity || !(expectedSpeed > 1e-9)) continue;
      const cell = i + nx * (j + ny * k);
      const v: readonly [number, number, number] = [velocity[3 * cell] ?? NaN,
        velocity[3 * cell + 1] ?? NaN, velocity[3 * cell + 2] ?? NaN];
      if (!v.every((value) => Number.isFinite(value))) continue;
      const along = v[0] * unit[0] + v[1] * unit[1] + v[2] * unit[2];
      const lateral = Math.hypot(v[0] - along * unit[0], v[1] - along * unit[1], v[2] - along * unit[2]);
      const shortfallFraction = (expectedSpeed - along) / expectedSpeed;
      bucket.sampledCells += 1;
      bucket.shortfallSum += shortfallFraction;
      bucket.maximumShortfall = Math.max(bucket.maximumShortfall, shortfallFraction);
      bucket.lateralSum += lateral;
      bucket.maximumLateral = Math.max(bucket.maximumLateral, lateral);
      if (!worstCell || shortfallFraction > worstCell.shortfallFraction) {
        worstCell = { i, j, k, contacts, shortfallFraction };
      }
      if (!bestCell || shortfallFraction < bestCell.shortfallFraction) {
        bestCell = { i, j, k, contacts, shortfallFraction };
      }
      // An overhead wall is one whose outward normal opposes gravity, which on
      // this lattice is the +gravity-facing extreme of the gravity-aligned
      // axis. It is the only wall that can hold liquid up.
      const gravityAxis = unit[1] <= -0.5 || unit[1] >= 0.5 ? 1 : unit[0] <= -0.5 || unit[0] >= 0.5 ? 0 : 2;
      const index3 = [i, j, k][gravityAxis]!, extent = [nx, ny, nz][gravityAxis]!;
      const overhead = unit[gravityAxis] < 0 ? index3 === extent - 1 : index3 === 0;
      const verticalWallCount = [i === 0 || i === nx - 1, j === 0 || j === ny - 1, k === 0 || k === nz - 1]
        .filter((touching, axis) => touching && axis !== gravityAxis).length;
      const key = `${overhead ? 1 : 0}:${verticalWallCount}`;
      let overheadBucket = overheadBuckets.get(key);
      if (!overheadBucket) {
        overheadBucket = { overhead, verticalWalls: verticalWallCount, wetCells: 0, sampledCells: 0, shortfallSum: 0 };
        overheadBuckets.set(key, overheadBucket);
      }
      overheadBucket.wetCells += 1; overheadBucket.sampledCells += 1;
      overheadBucket.shortfallSum += shortfallFraction;
    }
    // One cell of liquid is the thinnest column with a meaningful centroid;
    // anything less is a single-sample remnant, not a stuck column.
    if (!(columnAmount >= 1)) continue;
    const verticalWalls = (i === 0 ? 1 : 0) + (i === nx - 1 ? 1 : 0)
      + (k === 0 ? 1 : 0) + (k === nz - 1 ? 1 : 0);
    const lag_cells = weightedY / columnAmount - analyticCentroidY_cells;
    if (seedFootprint && i >= seedFootprint.originX && i < seedFootprint.originX + seedFootprint.size
      && k >= seedFootprint.originZ && k < seedFootprint.originZ + seedFootprint.size) {
      const localX = i - seedFootprint.originX, localZ = k - seedFootprint.originZ;
      const rim = localX === 0 || localZ === 0
        || localX === seedFootprint.size - 1 || localZ === seedFootprint.size - 1;
      if (rim) { footprintRimLag += lag_cells; footprintRimColumns += 1; }
      else if (localX >= 2 && localZ >= 2
        && localX <= seedFootprint.size - 3 && localZ <= seedFootprint.size - 3) {
        footprintCenterLag += lag_cells; footprintCenterColumns += 1;
      }
      // Compare the authoritative fine-phi upper interface, not compact
      // occupancy. A coarse cell-centre phi is also a distance to the side
      // faces, so treating its occupancy as a height creates a false crown.
      const topY_cells = fineUpperSurfaceField?.[i + nx * k] ?? NaN;
      const centerStart = Math.floor((seedFootprint.size - 2) / 2);
      const center = localX >= centerStart && localX < centerStart + 2
        && localZ >= centerStart && localZ < centerStart + 2;
      const shoulder = localX >= 2 && localX < seedFootprint.size - 2
        && localZ >= 2 && localZ < seedFootprint.size - 2 && !center;
      if (Number.isFinite(topY_cells)) {
        if (center) { footprintTopCenter += topY_cells; footprintTopCenterColumns += 1; }
        else if (shoulder) { footprintTopShoulder += topY_cells; footprintTopShoulderColumns += 1; }
      }
      if (velocity && topCell >= 0) {
        const cell = i + nx * (topCell + ny * k);
        const along = (velocity[3 * cell] ?? NaN) * unit[0]
          + (velocity[3 * cell + 1] ?? NaN) * unit[1]
          + (velocity[3 * cell + 2] ?? NaN) * unit[2];
        if (Number.isFinite(along)) {
          if (center) { footprintTopCenterSpeed += along; footprintTopCenterSpeedColumns += 1; }
          else if (shoulder) { footprintTopShoulderSpeed += along; footprintTopShoulderSpeedColumns += 1; }
        }
      }
    }
    let bucket = columns.get(verticalWalls);
    if (!bucket) {
      bucket = { verticalWalls, columns: 0, lagSum: 0, maximumLag: -Infinity };
      columns.set(verticalWalls, bucket);
    }
    bucket.columns += 1; bucket.lagSum += lag_cells;
    bucket.maximumLag = Math.max(bucket.maximumLag, lag_cells);
    if (!worstColumn || lag_cells > worstColumn.lag_cells) {
      worstColumn = { i, k, verticalWalls, lag_cells };
    }
    if (!bestColumn || lag_cells < bestColumn.lag_cells) {
      bestColumn = { i, k, verticalWalls, lag_cells };
    }
  }
  const round = (value: number) => Number(value.toFixed(4));
  return {
    velocityByContact: [...cells.values()].sort((a, b) => a.contacts - b.contacts).map((bucket) => ({
      contacts: bucket.contacts, wetCells: bucket.wetCells, sampledCells: bucket.sampledCells,
      meanShortfallFraction: bucket.sampledCells > 0 ? round(bucket.shortfallSum / bucket.sampledCells) : null,
      maximumShortfallFraction: bucket.sampledCells > 0 ? round(bucket.maximumShortfall) : null,
      meanLateralSpeed_m_s: bucket.sampledCells > 0 ? round(bucket.lateralSum / bucket.sampledCells) : null,
      maximumLateralSpeed_m_s: bucket.sampledCells > 0 ? round(bucket.maximumLateral) : null,
    })),
    velocityByOverheadContact: [...overheadBuckets.values()]
      .sort((a, b) => (a.overhead === b.overhead ? a.verticalWalls - b.verticalWalls : a.overhead ? 1 : -1))
      .map((bucket) => ({
        overhead: bucket.overhead, verticalWalls: bucket.verticalWalls, wetCells: bucket.wetCells,
        meanShortfallFraction: bucket.sampledCells > 0 ? round(bucket.shortfallSum / bucket.sampledCells) : null,
      })),
    columnLagByVerticalWalls: [...columns.values()].sort((a, b) => a.verticalWalls - b.verticalWalls)
      .map((bucket) => ({
        verticalWalls: bucket.verticalWalls, columns: bucket.columns,
        meanLag_cells: round(bucket.lagSum / bucket.columns),
        maximumLag_cells: round(bucket.maximumLag),
      })),
    ...(worstColumn ? { worstColumn: { ...worstColumn, lag_cells: round(worstColumn.lag_cells) } } : {}),
    ...(bestColumn ? { bestColumn: { ...bestColumn, lag_cells: round(bestColumn.lag_cells) } } : {}),
    ...(worstColumn && bestColumn ? {
      columnLagSpread_cells: round(worstColumn.lag_cells - bestColumn.lag_cells),
    } : {}),
    ...(worstCell ? { worstCell: { ...worstCell, shortfallFraction: round(worstCell.shortfallFraction) } } : {}),
    ...(bestCell ? { bestCell: { ...bestCell, shortfallFraction: round(bestCell.shortfallFraction) } } : {}),
    ...(worstCell && bestCell ? {
      velocityShortfallSpread: round(worstCell.shortfallFraction - bestCell.shortfallFraction),
    } : {}),
    ...(footprintCenterColumns > 0 && footprintRimColumns > 0 ? {
      footprintColumnLag: {
        centerMean_cells: round(footprintCenterLag / footprintCenterColumns),
        rimMean_cells: round(footprintRimLag / footprintRimColumns),
        centerToRim_cells: round(footprintCenterLag / footprintCenterColumns
          - footprintRimLag / footprintRimColumns),
        centerColumns: footprintCenterColumns,
        rimColumns: footprintRimColumns,
      },
    } : {}),
    ...(footprintTopCenterColumns > 0 && footprintTopShoulderColumns > 0 ? {
      footprintTopSurface: {
        centerMeanY_cells: round(footprintTopCenter / footprintTopCenterColumns),
        shoulderMeanY_cells: round(footprintTopShoulder / footprintTopShoulderColumns),
        centerProtrusion_cells: round(footprintTopCenter / footprintTopCenterColumns
          - footprintTopShoulder / footprintTopShoulderColumns),
        centerColumns: footprintTopCenterColumns,
        shoulderColumns: footprintTopShoulderColumns,
        ...(footprintTopCenterSpeedColumns > 0 && footprintTopShoulderSpeedColumns > 0 ? {
          centerMeanSpeed_m_s: round(footprintTopCenterSpeed / footprintTopCenterSpeedColumns),
          shoulderMeanSpeed_m_s: round(footprintTopShoulderSpeed / footprintTopShoulderSpeedColumns),
        } : {}),
      },
    } : {}),
  };
}

export type FreeFallContactAttribution = ReturnType<typeof freeFallContactAttribution>;
