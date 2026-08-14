/** Fixed-layout, matrix-free power-row publication oracle.
 *
 * Runtime operator kernels consume this arena and four disjoint worksets.
 * Descriptor lookup, geometry, sorting, and variable-length CSR construction
 * are deliberately absent from this ABI.
 */

import {
  OCTREE_WORKSET_INVALID_ID,
  buildOctreeWorkset,
} from "../octree-shared/webgpu-octree-worksets";
export const OCTREE_POWER_VELOCITY_FAMILIES = 6 as const;
export const OCTREE_POWER_BOUNDARY_SLOTS = 6 as const;
export const OCTREE_POWER_REGULAR_CASE_ID = 0 as const;

export const OCTREE_POWER_ROW_CLASS = Object.freeze({
  regularInterior: 0,
  transitionInterior: 1,
  physicalBoundary: 2,
  transitionBoundary: 3,
  invalid: 4,
} as const);

export type OctreePowerRowClass =
  typeof OCTREE_POWER_ROW_CLASS[keyof typeof OCTREE_POWER_ROW_CLASS];

export interface ResolvedPowerRowPlan {
  readonly rowCapacity: number;
  readonly maximumNeighborSlots: number;
  readonly rowStrideWords: number;
  readonly rowBytes: number;
  readonly arenaBytes: number;
  readonly worksetBytes: number;
  readonly allocatedBytes: number;
}

export interface ResolvedPowerRowInput {
  readonly row: number;
  readonly caseId: number;
  readonly catalogEntry: number;
  readonly physicalPage: number;
  readonly localCell: number;
  readonly volume: number;
  readonly scale: number;
  readonly neighborRows: readonly number[];
  readonly neighborCoefficients: readonly number[];
  readonly velocityFamilyHandles: readonly number[];
  readonly dynamicBoundaryFractions?: readonly number[];
  readonly transition: boolean;
  readonly physicalBoundary: boolean;
  readonly valid?: boolean;
}

export interface ResolvedPowerRow extends ResolvedPowerRowInput {
  readonly rowClass: OctreePowerRowClass;
  readonly epoch: number;
}

export interface ResolvedPowerRowsPublication {
  readonly epoch: number;
  readonly plan: ResolvedPowerRowPlan;
  readonly rows: readonly ResolvedPowerRow[];
  readonly packedRows: Uint32Array;
  readonly regularInteriorRows: Uint32Array;
  readonly transitionInteriorRows: Uint32Array;
  readonly physicalBoundaryRows: Uint32Array;
  readonly transitionBoundaryRows: Uint32Array;
  readonly invalidRows: Uint32Array;
}

export type OctreeRelativeGrading = "same" | "same-or-finer" | "same-or-coarser";

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`${label} must be a positive integer`);
  }
  return value;
}

function uint32(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
    throw new RangeError(`${label} must fit uint32`);
  }
  return value >>> 0;
}

function finitePositive(value: number, label: string): number {
  if (!Number.isFinite(value) || value <= 0) {
    throw new RangeError(`${label} must be finite and positive`);
  }
  return value;
}

function floatBits(value: number): number {
  const bytes = new ArrayBuffer(4);
  new Float32Array(bytes)[0] = value;
  return new Uint32Array(bytes)[0]!;
}

export function planResolvedPowerRows(
  rowCapacityValue: number,
  maximumNeighborSlotsValue: number,
): ResolvedPowerRowPlan {
  const rowCapacity = positiveInteger(rowCapacityValue, "Resolved power row capacity");
  const maximumNeighborSlots = positiveInteger(
    maximumNeighborSlotsValue,
    "Resolved power maximum neighbor slots",
  );
  // class/case, catalog/neighbor-count, epoch, page/local, volume, scale,
  // neighbor row handles, normalized coefficients, six family handles, and
  // six dynamic boundary fractions.
  const rowStrideWords = 6 + 2 * maximumNeighborSlots
    + OCTREE_POWER_VELOCITY_FAMILIES + OCTREE_POWER_BOUNDARY_SLOTS;
  const rowBytes = rowStrideWords * 4;
  const arenaBytes = rowCapacity * rowBytes;
  const worksetBytes = 5 * (7 + rowCapacity) * 4;
  return {
    rowCapacity,
    maximumNeighborSlots,
    rowStrideWords,
    rowBytes,
    arenaBytes,
    worksetBytes,
    allocatedBytes: arenaBytes + worksetBytes,
  };
}

/** Enforce the stronger face/edge 1-ring contract before case lookup. */
export function classifyExclusivePowerGrading(
  anchorLevelValue: number,
  neighborLevels: readonly number[],
): OctreeRelativeGrading {
  const anchorLevel = uint32(anchorLevelValue, "Anchor level");
  if (neighborLevels.length === 0) return "same";
  let finer = false;
  let coarser = false;
  for (let index = 0; index < neighborLevels.length; index += 1) {
    const level = uint32(neighborLevels[index]!, `Neighbor level ${index}`);
    const delta = level - anchorLevel;
    if (Math.abs(delta) > 1) {
      throw new RangeError("Power 1-ring violates 2:1 grading");
    }
    finer ||= delta > 0;
    coarser ||= delta < 0;
  }
  if (finer && coarser) {
    throw new RangeError("Power 1-ring mixes finer and coarser neighbors");
  }
  return finer ? "same-or-finer" : coarser ? "same-or-coarser" : "same";
}

export function classifyResolvedPowerRow(
  transition: boolean,
  physicalBoundary: boolean,
  valid = true,
): OctreePowerRowClass {
  if (!valid) return OCTREE_POWER_ROW_CLASS.invalid;
  if (transition) {
    return physicalBoundary
      ? OCTREE_POWER_ROW_CLASS.transitionBoundary
      : OCTREE_POWER_ROW_CLASS.transitionInterior;
  }
  return physicalBoundary
    ? OCTREE_POWER_ROW_CLASS.physicalBoundary
    : OCTREE_POWER_ROW_CLASS.regularInterior;
}

function validateRow(plan: ResolvedPowerRowPlan, row: ResolvedPowerRowInput): void {
  uint32(row.row, "Resolved power row");
  if (row.row >= plan.rowCapacity) throw new RangeError("Resolved power row exceeds capacity");
  uint32(row.caseId, "Resolved power case ID");
  uint32(row.catalogEntry, "Resolved power catalog entry");
  uint32(row.physicalPage, "Resolved power physical page");
  uint32(row.localCell, "Resolved power local cell");
  finitePositive(row.volume, "Resolved power volume");
  finitePositive(row.scale, "Resolved power scale");
  if (row.neighborRows.length !== row.neighborCoefficients.length
    || row.neighborRows.length > plan.maximumNeighborSlots) {
    throw new RangeError("Resolved power neighbor handles and coefficients must have bounded equal length");
  }
  row.neighborRows.forEach((neighbor, slot) => {
    uint32(neighbor, `Resolved power neighbor ${slot}`);
    if (neighbor >= plan.rowCapacity || neighbor === row.row) {
      throw new RangeError("Resolved power neighbor must name another row in the arena");
    }
    finitePositive(row.neighborCoefficients[slot]!, `Resolved power coefficient ${slot}`);
  });
  if (row.velocityFamilyHandles.length !== OCTREE_POWER_VELOCITY_FAMILIES) {
    throw new RangeError("Resolved power row needs exactly six velocity-family handles");
  }
  row.velocityFamilyHandles.forEach((handle, family) => {
    if (handle !== OCTREE_WORKSET_INVALID_ID) {
      uint32(handle, `Resolved power velocity-family handle ${family}`);
    }
  });
  const fractions = row.dynamicBoundaryFractions ?? [];
  if (fractions.length > OCTREE_POWER_BOUNDARY_SLOTS) {
    throw new RangeError("Resolved power row has too many dynamic-boundary fractions");
  }
  fractions.forEach((fraction, slot) => {
    if (!Number.isFinite(fraction) || fraction < 0 || fraction > 1) {
      throw new RangeError(`Resolved power boundary fraction ${slot} must be in [0,1]`);
    }
  });
  if (!row.transition && !row.physicalBoundary && row.caseId !== OCTREE_POWER_REGULAR_CASE_ID) {
    throw new RangeError("Regular interior rows must use compact case zero");
  }
}

export function buildResolvedPowerRowsPublication(
  plan: ResolvedPowerRowPlan,
  epochValue: number,
  rowInputs: readonly ResolvedPowerRowInput[],
): ResolvedPowerRowsPublication {
  const epoch = uint32(epochValue, "Resolved power epoch");
  if (epoch === 0) throw new RangeError("Resolved power epoch zero is unpublished");
  if (rowInputs.length > plan.rowCapacity) throw new RangeError("Resolved power publication exceeds capacity");
  const seen = new Uint8Array(plan.rowCapacity);
  const rows = rowInputs.map((input): ResolvedPowerRow => {
    validateRow(plan, input);
    if (seen[input.row] !== 0) throw new RangeError("Resolved power publication contains duplicate rows");
    seen[input.row] = 1;
    return Object.freeze({
      ...input,
      valid: input.valid ?? true,
      rowClass: classifyResolvedPowerRow(input.transition, input.physicalBoundary, input.valid),
      epoch,
    });
  }).sort((left, right) => left.row - right.row);
  if (rows.some((row, index) => row.row !== index)) {
    throw new RangeError("Resolved power rows must form a dense live prefix");
  }

  const packedRows = new Uint32Array(plan.rowCapacity * plan.rowStrideWords)
    .fill(OCTREE_WORKSET_INVALID_ID);
  const byClass = new Map<OctreePowerRowClass, number[]>();
  for (const row of rows) {
    const base = row.row * plan.rowStrideWords;
    packedRows[base] = ((row.caseId & 0x00ff_ffff) | (row.rowClass << 24)) >>> 0;
    packedRows[base + 1] = ((row.catalogEntry & 0x00ff_ffff)
      | (row.neighborRows.length << 24)) >>> 0;
    packedRows[base + 2] = epoch;
    packedRows[base + 3] = ((row.physicalPage & 0xffff) | ((row.localCell & 0xffff) << 16)) >>> 0;
    packedRows[base + 4] = floatBits(row.volume);
    packedRows[base + 5] = floatBits(row.scale);
    const neighborBase = base + 6;
    packedRows.set(row.neighborRows, neighborBase);
    const coefficientBase = neighborBase + plan.maximumNeighborSlots;
    packedRows.set(row.neighborCoefficients.map(floatBits), coefficientBase);
    const velocityBase = coefficientBase + plan.maximumNeighborSlots;
    packedRows.set(row.velocityFamilyHandles, velocityBase);
    const boundaryBase = velocityBase + OCTREE_POWER_VELOCITY_FAMILIES;
    packedRows.fill(floatBits(1), boundaryBase, boundaryBase + OCTREE_POWER_BOUNDARY_SLOTS);
    packedRows.set((row.dynamicBoundaryFractions ?? []).map(floatBits), boundaryBase);
    const list = byClass.get(row.rowClass);
    if (list) list.push(row.row); else byClass.set(row.rowClass, [row.row]);
  }

  const workset = (rowClass: OctreePowerRowClass) => buildOctreeWorkset(
    byClass.get(rowClass) ?? [],
    { epoch, capacity: plan.rowCapacity, workgroupSize: 64 },
  );
  return Object.freeze({
    epoch,
    plan,
    rows,
    packedRows,
    regularInteriorRows: workset(OCTREE_POWER_ROW_CLASS.regularInterior),
    transitionInteriorRows: workset(OCTREE_POWER_ROW_CLASS.transitionInterior),
    physicalBoundaryRows: workset(OCTREE_POWER_ROW_CLASS.physicalBoundary),
    transitionBoundaryRows: workset(OCTREE_POWER_ROW_CLASS.transitionBoundary),
    invalidRows: workset(OCTREE_POWER_ROW_CLASS.invalid),
  });
}

/** Internal face coefficients must be bit-identical from both incident rows. */
export function validateResolvedPowerReciprocity(
  publication: ResolvedPowerRowsPublication,
): readonly string[] {
  const issues: string[] = [];
  const byRow = new Map(publication.rows.map((row) => [row.row, row]));
  for (const row of publication.rows) {
    row.neighborRows.forEach((neighbor, slot) => {
      const other = byRow.get(neighbor);
      if (!other) {
        issues.push(`row ${row.row} references absent row ${neighbor}`);
        return;
      }
      const reverse = other.neighborRows.indexOf(row.row);
      if (reverse < 0) {
        issues.push(`row ${row.row} -> ${neighbor} has no reciprocal slot`);
        return;
      }
      if (floatBits(row.neighborCoefficients[slot]!)
        !== floatBits(other.neighborCoefficients[reverse]!)) {
        issues.push(`row ${row.row} <-> ${neighbor} coefficient bits differ`);
      }
    });
  }
  return issues;
}

export const OCTREE_RESOLVED_ROW_GPU_CONTROL_WORDS = 16;
export const OCTREE_RESOLVED_ROW_GPU_CONTROL_BYTES =
  OCTREE_RESOLVED_ROW_GPU_CONTROL_WORDS * 4;
export const OCTREE_RESOLVED_ROW_CLASS_COUNT = 5;

export const OCTREE_RESOLVED_ROW_GPU_ERROR = Object.freeze({
  sourceAuthority: 1 << 0,
  capacity: 1 << 1,
  invalidHeader: 1 << 2,
  invalidEntry: 1 << 3,
  nonFinite: 1 << 4,
  catalog: 1 << 5,
} as const);

export interface WebGPUResolvedPowerRowWorksetBinding {
  readonly buffer: GPUBuffer;
  readonly offset: number;
  readonly size: number;
}

export interface WebGPUResolvedPowerRowSource {
  readonly plan: ResolvedPowerRowPlan;
  readonly params: GPUBuffer;
  readonly rows: GPUBuffer;
  readonly diagonals: GPUBuffer;
  readonly worksets: Readonly<Record<
    "regularInterior" | "transitionInterior" | "physicalBoundary" | "transitionBoundary",
    WebGPUResolvedPowerRowWorksetBinding
  >>;
  readonly invalidRows: WebGPUResolvedPowerRowWorksetBinding;
  readonly control: GPUBuffer;
}
