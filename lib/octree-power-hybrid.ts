/**
 * Conservative Stage-1 hybrid for the octree power operator.
 *
 * The classification is intentionally a proof, not a heuristic.  A row may
 * use implicit Cartesian addressing only when all six liquid neighbours are
 * same-sized and every local face is the exact regular-grid face.  Everything
 * else remains in the power band.  A face is power-owned when either endpoint
 * is power-band, so the two matrix rows can never disagree at the seam.
 */

import {
  OCTREE_POWER_INVALID_ROW,
  type OctreePowerFaceRecord,
  type OctreePowerOperator,
} from "./octree-power-operator";

export const OCTREE_POWER_HYBRID_ROW_CLASS = Object.freeze({
  regularInterior: 0,
  powerBand: 1,
} as const);

export const OCTREE_POWER_HYBRID_FACE_CLASS = Object.freeze({
  regular: 0,
  power: 1,
} as const);

export interface OctreePowerHybridWork {
  readonly descriptorRows: number;
  readonly catalogLookups: number;
  readonly tetrahedronFans: number;
  readonly pageSlotTraversals: number;
  readonly expensiveUnits: number;
}

export interface OctreePowerHybridPlan {
  readonly operator: OctreePowerOperator;
  readonly rowClasses: Uint8Array;
  readonly faceClasses: Uint8Array;
  readonly regularRows: Uint32Array;
  readonly powerRows: Uint32Array;
  readonly regularFaces: number;
  readonly seamFaces: number;
  readonly powerBandFaces: number;
  readonly fullPowerWork: OctreePowerHybridWork;
  readonly hybridWork: OctreePowerHybridWork;
  readonly workReduction: number;
}

export interface OctreePowerHybridOptions {
  /**
   * Same-epoch GFM, cut-cell, moving-solid, or policy rows supplied by the
   * boundary publisher.  Unknown rows reject the plan instead of being
   * ignored.  The geometric proof below independently catches non-unit open
   * fractions, domain boundaries, and level transitions.
   */
  readonly forcePowerRows?: Iterable<number>;
}

/** Seven-word compact workset ABI published by the structured boundary pass. */
export const OCTREE_POWER_HYBRID_WORKSET_HEADER_WORDS = 7;
export const OCTREE_POWER_HYBRID_ROW_CLASS_COUNT = 4;

export interface OctreePowerHybridRuntimePublication {
  readonly acceptedRows: number;
  readonly acceptedEpoch: number;
  /** The accepted bank's four row-class headers, classes 0..3. */
  readonly headers: readonly Uint32Array[];
  /** The accepted bank's members corresponding to `headers`, classes 0..3. */
  readonly members: readonly Uint32Array[];
}

export interface OctreePowerHybridRuntimeCensus {
  readonly liveRows: number;
  readonly regularRows: number;
  readonly powerRows: number;
  readonly classCounts: readonly [number, number, number, number];
  readonly fullDescriptorRows: number;
  readonly hybridDescriptorRows: number;
  readonly fullCatalogLookups: number;
  readonly hybridCatalogLookups: number;
  readonly fullPageSlotChains: number;
  readonly hybridPageSlotChains: number;
  readonly machineryReduction: number;
}

/**
 * Acceptance verdict for a live shipping census.  Keeping this independent of
 * the smoke runner makes the gate exact and unit-testable: a missing or corrupt
 * GPU publication cannot accidentally be scored as saved work.
 */
export function octreePowerHybridWorkVerdict(
  census: Pick<OctreePowerHybridRuntimeCensus,
    "liveRows" | "regularRows" | "powerRows" | "fullPageSlotChains" | "hybridPageSlotChains"> | null | undefined,
  minimumReduction: number,
): { readonly accepted: boolean; readonly reason?: string } {
  if (!Number.isFinite(minimumReduction) || minimumReduction < 1) return {
    accepted: false,
    reason: `minimum reduction must be finite and at least 1; received ${minimumReduction}`,
  };
  if (!census) return { accepted: false, reason: "shipping GPU census was not published" };
  const full = census.fullPageSlotChains;
  const hybrid = census.hybridPageSlotChains;
  const liquidRows = census.regularRows + census.powerRows;
  if (!Number.isSafeInteger(full) || full < 1 || !Number.isSafeInteger(hybrid) || hybrid < 0
    || !Number.isSafeInteger(liquidRows) || liquidRows < 1 || liquidRows > census.liveRows
    || full !== 18 * liquidRows || hybrid !== 18 * census.powerRows) return {
    accepted: false,
    reason: "shipping GPU census has inconsistent expensive-work totals",
  };
  const reduction = hybrid === 0 ? Number.POSITIVE_INFINITY : full / hybrid;
  if (reduction < minimumReduction) return {
    accepted: false,
    reason: `shipping hybrid machinery reduction ${reduction.toFixed(6)}x is below ${minimumReduction.toFixed(6)}x`,
  };
  return { accepted: true };
}

/**
 * Decode a runtime Bet-4 census only from a complete accepted publication.
 *
 * Counts alone are deliberately insufficient: a duplicated row and a missing
 * row can have the right total while authorizing the wrong stencil.  This
 * decoder verifies the seven-word indirect headers and an exact, disjoint
 * partition of `[0, acceptedRows)` before reporting any saved work.
 */
export function decodeOctreePowerHybridRuntimeCensus(
  publication: OctreePowerHybridRuntimePublication,
): OctreePowerHybridRuntimeCensus | null {
  const { acceptedRows, acceptedEpoch, headers, members } = publication;
  if (!Number.isSafeInteger(acceptedRows) || acceptedRows < 1
    || !Number.isSafeInteger(acceptedEpoch) || acceptedEpoch < 1
    || headers.length !== OCTREE_POWER_HYBRID_ROW_CLASS_COUNT
    || members.length !== OCTREE_POWER_HYBRID_ROW_CLASS_COUNT) return null;
  const seen = new Uint8Array(acceptedRows);
  const counts: number[] = [];
  for (let cls = 0; cls < OCTREE_POWER_HYBRID_ROW_CLASS_COUNT; cls += 1) {
    const header = headers[cls], list = members[cls];
    if (!header || header.length < OCTREE_POWER_HYBRID_WORKSET_HEADER_WORDS || !list) return null;
    const count = header[1]!;
    const blocks = Math.ceil(count / 64);
    const dispatchX = Math.max(1, Math.min(65_535, blocks));
    const dispatchY = Math.ceil(blocks / dispatchX);
    if (header[0] !== acceptedEpoch || count !== list.length
      || header[2]! < acceptedRows || header[3] !== 3
      || header[4] !== dispatchX || header[5] !== dispatchY || header[6] !== 1) return null;
    counts.push(count);
    for (const row of list) {
      if (row >= acceptedRows || seen[row] !== 0) return null;
      seen[row] = 1;
    }
  }
  if (counts.reduce((sum, count) => sum + count, 0) !== acceptedRows
    || seen.some((value) => value !== 1)) return null;
  const classCounts = [counts[0]!, counts[1]!, counts[2]!, counts[3]!] as const;
  const regularRows = classCounts[0];
  const powerRows = acceptedRows - regularRows;
  // The current power image has eighteen canonical channels. Stage 1 removes
  // its descriptor/catalog/address machinery for class 0; all coefficient
  // loads remain until the seam oracle permits the optional second cut.
  const fullPageSlotChains = 18 * acceptedRows;
  const hybridPageSlotChains = 18 * powerRows;
  return {
    liveRows: acceptedRows,
    regularRows,
    powerRows,
    classCounts,
    fullDescriptorRows: acceptedRows,
    hybridDescriptorRows: powerRows,
    fullCatalogLookups: acceptedRows,
    hybridCatalogLookups: powerRows,
    fullPageSlotChains,
    hybridPageSlotChains,
    machineryReduction: powerRows === 0 ? Number.POSITIVE_INFINITY : acceptedRows / powerRows,
  };
}

const DIRECTIONS = [
  [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1],
] as const;

function rowFaceDirection(
  operator: OctreePowerOperator,
  row: number,
  face: OctreePowerFaceRecord,
): number | undefined {
  const neighbor = face.negativeRow === row ? face.positiveRow
    : face.positiveRow === row ? face.negativeRow : OCTREE_POWER_INVALID_ROW;
  if (neighbor === OCTREE_POWER_INVALID_ROW || neighbor >= operator.sites.length) return undefined;
  const site = operator.sites[row]!;
  const other = operator.sites[neighbor]!;
  const h = site.size;
  if (other.size !== h || face.openFraction !== 1 || face.area !== h * h
    || face.inverseDistance !== 1 / h) return undefined;
  const delta = other.center.map((value, axis) => value - site.center[axis]);
  const direction = DIRECTIONS.findIndex((candidate) => candidate.every(
    (value, axis) => delta[axis] === value * h,
  ));
  if (direction < 0) return undefined;
  const outward = face.negativeRow === row ? face.normal : face.normal.map((value) => -value);
  if (!DIRECTIONS[direction]!.every((value, axis) => outward[axis] === value)) return undefined;
  const midpoint = site.center.map((value, axis) => (value + other.center[axis]) / 2);
  if (!face.centroid.every((value, axis) => value === midpoint[axis])) return undefined;
  return direction;
}

function rowIsProvenRegular(operator: OctreePowerOperator, row: number): boolean {
  const site = operator.sites[row]!;
  const compact = operator.rows[row]!;
  if (compact.row !== row || compact.volume !== site.size ** 3
    || compact.entries.length !== 6 || compact.diagonal !== 6 * site.size
    || operator.incidence[row]!.length !== 6) return false;
  const directions = new Set<number>();
  const neighbors = new Set<number>();
  for (const item of operator.incidence[row]!) {
    const face = operator.faces[item.face];
    if (!face) return false;
    const expectedSign = face.negativeRow === row ? 1 : face.positiveRow === row ? -1 : 0;
    if (item.sign !== expectedSign) return false;
    const direction = rowFaceDirection(operator, row, face);
    if (direction === undefined || directions.has(direction)) return false;
    directions.add(direction);
    neighbors.add(face.negativeRow === row ? face.positiveRow : face.negativeRow);
  }
  return directions.size === 6 && compact.entries.every((entry) =>
    neighbors.has(entry.row) && entry.coefficient === site.size);
}

/** Validate the unique-face authority on which the seam rule relies. */
function assertCoherentFaces(operator: OctreePowerOperator): void {
  const counts = new Uint8Array(operator.faces.length);
  const pairKeys = new Set<string>();
  for (let row = 0; row < operator.rows.length; row += 1) {
    const compact = operator.rows[row]!;
    if (compact.row !== row || !Number.isFinite(compact.diagonal) || compact.diagonal < 0
      || compact.entries.some((entry) => !Number.isSafeInteger(entry.row) || entry.row < 0
        || entry.row >= operator.rows.length || entry.row === row
        || !(entry.coefficient > 0) || !Number.isFinite(entry.coefficient))) {
      throw new RangeError("Hybrid classification requires valid compact power rows");
    }
    const seenFaces = new Set<number>();
    for (const item of operator.incidence[row]!) {
      const face = operator.faces[item.face];
      if (!face || seenFaces.has(item.face)
        || (face.negativeRow === row ? item.sign !== 1
          : face.positiveRow === row ? item.sign !== -1 : true)) {
        throw new RangeError("Hybrid classification requires reciprocal unique face incidence");
      }
      seenFaces.add(item.face); counts[item.face]! += 1;
    }
  }
  for (const face of operator.faces) {
    if (![...face.centroid, ...face.normal, face.area, face.inverseDistance, face.openFraction]
      .every(Number.isFinite)
      || !(face.area > 0) || face.inverseDistance < 0 || face.openFraction < 0 || face.openFraction > 1
      || counts[face.id] !== (face.positiveRow === OCTREE_POWER_INVALID_ROW ? 1 : 2)) {
      throw new RangeError("Hybrid classification requires valid unique physical faces");
    }
    if (face.positiveRow !== OCTREE_POWER_INVALID_ROW) {
      const low = Math.min(face.negativeRow, face.positiveRow), high = Math.max(face.negativeRow, face.positiveRow);
      const key = `${low}:${high}`;
      if (pairKeys.has(key)) throw new RangeError("Hybrid classification rejects duplicate row-pair faces");
      pairKeys.add(key);
      const coefficient = face.openFraction * face.area * face.inverseDistance;
      for (const row of [face.negativeRow, face.positiveRow]) {
        const entry = operator.rows[row]!.entries.find((candidate) => candidate.row === (row === low ? high : low));
        if (!entry || entry.coefficient !== coefficient) {
          throw new RangeError("Hybrid classification requires the shared face coefficient in both rows");
        }
      }
    }
  }
}

function work(
  operator: OctreePowerOperator,
  rows: readonly number[] | Uint32Array,
): OctreePowerHybridWork {
  let tetrahedronFans = 0, pageSlotTraversals = 0;
  for (const row of rows) {
    tetrahedronFans += operator.incidence[row]!.length;
    pageSlotTraversals += operator.rows[row]!.entries.length;
  }
  const descriptorRows = rows.length;
  const catalogLookups = rows.length;
  return {
    descriptorRows,
    catalogLookups,
    tetrahedronFans,
    pageSlotTraversals,
    expensiveUnits: descriptorRows + catalogLookups + tetrahedronFans + pageSlotTraversals,
  };
}

/** Build one fail-closed class publication from an immutable power operator. */
export function buildOctreePowerHybridPlan(
  operator: OctreePowerOperator,
  options: OctreePowerHybridOptions = {},
): OctreePowerHybridPlan {
  const rowCount = operator.rows.length;
  if (operator.sites.length !== rowCount || operator.incidence.length !== rowCount
    || operator.faces.some((face, id) => face.id !== id
      || face.negativeRow >= rowCount
      || (face.positiveRow !== OCTREE_POWER_INVALID_ROW && face.positiveRow >= rowCount))) {
    throw new RangeError("Hybrid classification requires a coherent immutable power operator");
  }
  assertCoherentFaces(operator);
  const forced = new Set(options.forcePowerRows ?? []);
  for (const row of forced) if (!Number.isSafeInteger(row) || row < 0 || row >= rowCount) {
    throw new RangeError("Forced power-band row is outside the published operator");
  }
  const rowClasses = new Uint8Array(rowCount);
  const regularRows: number[] = [], powerRows: number[] = [];
  for (let row = 0; row < rowCount; row += 1) {
    const regular = !forced.has(row) && rowIsProvenRegular(operator, row);
    rowClasses[row] = regular
      ? OCTREE_POWER_HYBRID_ROW_CLASS.regularInterior
      : OCTREE_POWER_HYBRID_ROW_CLASS.powerBand;
    (regular ? regularRows : powerRows).push(row);
  }

  const faceClasses = new Uint8Array(operator.faces.length);
  let regularFaces = 0, seamFaces = 0, powerBandFaces = 0;
  for (const face of operator.faces) {
    const negativePower = rowClasses[face.negativeRow] === OCTREE_POWER_HYBRID_ROW_CLASS.powerBand;
    const positivePower = face.positiveRow === OCTREE_POWER_INVALID_ROW
      || rowClasses[face.positiveRow] === OCTREE_POWER_HYBRID_ROW_CLASS.powerBand;
    const power = negativePower || positivePower;
    faceClasses[face.id] = power
      ? OCTREE_POWER_HYBRID_FACE_CLASS.power
      : OCTREE_POWER_HYBRID_FACE_CLASS.regular;
    if (!power) regularFaces += 1;
    else if (face.positiveRow !== OCTREE_POWER_INVALID_ROW && negativePower !== positivePower) seamFaces += 1;
    else powerBandFaces += 1;
  }

  const allRows = Uint32Array.from({ length: rowCount }, (_, row) => row);
  const regular = Uint32Array.from(regularRows);
  const powerBand = Uint32Array.from(powerRows);
  const fullPowerWork = work(operator, allRows);
  const hybridWork = work(operator, powerBand);
  return {
    operator,
    rowClasses,
    faceClasses,
    regularRows: regular,
    powerRows: powerBand,
    regularFaces,
    seamFaces,
    powerBandFaces,
    fullPowerWork,
    hybridWork,
    workReduction: hybridWork.expensiveUnits === 0
      ? Number.POSITIVE_INFINITY
      : fullPowerWork.expensiveUnits / hybridWork.expensiveUnits,
  };
}

function latticeKey(origin: readonly number[], size: number): string {
  return `${size}:${origin[0]},${origin[1]},${origin[2]}`;
}

/**
 * Float64 differential oracle for the Stage-1 operator.
 *
 * Power-band rows execute the full compact power row.  Proven regular rows
 * discover their six neighbors arithmetically from lattice coordinates; only
 * seam faces consult the unique power-face coefficient.  The sorting matches
 * the full oracle's row order so exact regular grids are bit-identical.
 */
export function applyOctreePowerHybridMatrix(
  operator: OctreePowerOperator,
  plan: OctreePowerHybridPlan,
  vector: readonly number[],
): number[] {
  if (plan.operator !== operator) throw new Error("Hybrid plan/operator epoch mismatch");
  if (vector.length !== operator.rows.length || vector.some((value) => !Number.isFinite(value))) {
    throw new RangeError("Hybrid matrix vector must be finite and match the row count");
  }
  const rowByLattice = new Map(operator.sites.map((site, row) => [latticeKey(site.origin, site.size), row]));
  const faceByRows = new Map<string, OctreePowerFaceRecord>();
  for (const face of operator.faces) if (face.positiveRow !== OCTREE_POWER_INVALID_ROW) {
    const low = Math.min(face.negativeRow, face.positiveRow), high = Math.max(face.negativeRow, face.positiveRow);
    const key = `${low}:${high}`;
    if (faceByRows.has(key)) throw new Error("Hybrid operator has duplicate physical faces");
    faceByRows.set(key, face);
  }
  return operator.rows.map((compact, row) => {
    if (plan.rowClasses[row] === OCTREE_POWER_HYBRID_ROW_CLASS.powerBand) {
      return compact.diagonal * vector[row]!
        - compact.entries.reduce((sum, entry) => sum + entry.coefficient * vector[entry.row]!, 0);
    }
    const site = operator.sites[row]!;
    const neighbors = DIRECTIONS.map((direction) => {
      const origin = site.origin.map((value, axis) => value + direction[axis] * site.size);
      const neighbor = rowByLattice.get(latticeKey(origin, site.size));
      if (neighbor === undefined) throw new Error("Proven regular row lost an arithmetic neighbor");
      const low = Math.min(row, neighbor), high = Math.max(row, neighbor);
      const face = faceByRows.get(`${low}:${high}`);
      if (!face) throw new Error("Proven regular row lost its physical face");
      const coefficient = plan.faceClasses[face.id] === OCTREE_POWER_HYBRID_FACE_CLASS.power
        ? face.openFraction * face.area * face.inverseDistance
        : site.size;
      return { row: neighbor, coefficient };
    }).sort((a, b) => a.row - b.row);
    return 6 * site.size * vector[row]!
      - neighbors.reduce((sum, entry) => sum + entry.coefficient * vector[entry.row]!, 0);
  });
}
