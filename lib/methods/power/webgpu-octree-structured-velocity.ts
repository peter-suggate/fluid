/**
 * CPU differential oracle for the epoch-published six-family velocity authority.
 *
 * This mirrors the direct GPU publisher's fixed-layout SoA state for numerical
 * fixtures and carry/reciprocity checks. It is not imported by production and
 * intentionally contains no GPU executor or shader.
 */

import {
  OCTREE_POWER_RECONSTRUCTION_FLOATS,
  OCTREE_POWER_ROW_TEMPLATE_FLOATS,
  OCTREE_POWER_ROW_TEMPLATE_HEADER_WORDS,
  unpackOctreePowerRowTemplateSlot,
} from "./octree-power-catalog";
import type { ResolvedPowerRowsPublication } from "./webgpu-octree-power-resolved-rows";
import { OCTREE_POWER_ROW_CLASS } from "./webgpu-octree-power-resolved-rows";
import { OCTREE_WORKSET_INVALID_ID, buildOctreeWorkset } from "../octree-shared/webgpu-octree-worksets";
import type { OctreePowerTopologySource } from "./webgpu-octree-power-topology";

export const OCTREE_STRUCTURED_VELOCITY_FAMILY_COUNT = 6;
export const OCTREE_STRUCTURED_VELOCITY_MAX_ORIENTATIONS = 8;
export const OCTREE_STRUCTURED_VELOCITY_INVALID_HANDLE = OCTREE_WORKSET_INVALID_ID;

export type StructuredVelocityClass =
  | "regularInterior"
  | "transitionInterior"
  | "physicalBoundary"
  | "transitionBoundary";

export const OCTREE_STRUCTURED_VELOCITY_CLASSES = Object.freeze([
  "regularInterior", "transitionInterior", "physicalBoundary", "transitionBoundary",
] as const);

export interface StructuredVelocityCatalogSource {
  readonly rowTemplateHeaders: Uint32Array;
  readonly rowTemplateSlots: Uint32Array;
  readonly rowTemplateData: Float32Array;
  readonly reconstructionData: Float32Array;
  /** Catalog face layout is the existing immutable 12-float generator ABI. */
  readonly faceData: Float32Array;
}

/**
 * Rebuild-only resolution of a case-local catalog slot. `neighborRows` is not
 * retained after publication. A world-boundary slot must use INVALID.
 */
export interface TransientStructuredVelocityRowResolution {
  readonly row: number;
  readonly neighborRows: readonly number[];
  readonly normalVelocities?: readonly number[];
  readonly boundaryFractions?: readonly number[];
}

export interface StructuredVelocityCarry {
  readonly previous: StructuredVelocityPublication;
  /** old row -> new row; INVALID means the old row retired. */
  readonly oldToNewRows: Uint32Array;
}

export interface StructuredVelocityPublication {
  readonly epoch: number;
  readonly rowCapacity: number;
  readonly maximumCaseSlots: number;
  readonly slotCount: number;
  /** Seven offsets delimit the six contiguous family ranges. */
  readonly familyOffsets: Uint32Array;
  readonly values: Float32Array;
  readonly ownerRows: Uint32Array;
  readonly neighborRows: Uint32Array;
  /** family | orientation<<3 | catalog-slot<<6. */
  readonly orientation: Uint32Array;
  readonly coefficients: Float32Array;
  readonly inverseDistances: Float32Array;
  readonly boundaryFractions: Float32Array;
  readonly normalX: Float32Array;
  readonly normalY: Float32Array;
  readonly normalZ: Float32Array;
  /** Exactly six handles per row; each names an eight-orientation block. */
  readonly rowFamilyHandles: Uint32Array;
  readonly rowFamilySlotHandles: Uint32Array;
  readonly rowFamilySlotCounts: Uint32Array;
  /** Fixed rowCapacity*maximumCaseSlots handle table; no runtime CSR. */
  readonly rowSlotHandles: Uint32Array;
  /** Incidence sign stored independently because a shared value has one owner. */
  readonly rowSlotSigns: Int32Array;
  readonly rowCatalogSlots: Uint32Array;
  readonly rowSlotCounts: Uint32Array;
  readonly rowWorksets: Readonly<Record<StructuredVelocityClass, Uint32Array>>;
  readonly familyWorksets: Readonly<Record<StructuredVelocityClass, Uint32Array>>;
  /** Publisher/debug identity only; hot consumers do not receive it. */
  readonly slotStableKeys: readonly string[];
}

const floatBits = (value: number): number => {
  const storage = new ArrayBuffer(4);
  new Float32Array(storage)[0] = value;
  return new Uint32Array(storage)[0]!;
};

const finite = (value: number, label: string): number => {
  if (!Number.isFinite(value)) throw new RangeError(`${label} must be finite`);
  return Math.fround(value);
};

function className(transition: boolean, boundary: boolean): StructuredVelocityClass {
  if (transition) return boundary ? "transitionBoundary" : "transitionInterior";
  return boundary ? "physicalBoundary" : "regularInterior";
}

function rowClassFlags(rowClass: number): readonly [boolean, boolean] {
  return [
    rowClass === OCTREE_POWER_ROW_CLASS.transitionInterior
      || rowClass === OCTREE_POWER_ROW_CLASS.transitionBoundary,
    rowClass === OCTREE_POWER_ROW_CLASS.physicalBoundary
      || rowClass === OCTREE_POWER_ROW_CLASS.transitionBoundary,
  ];
}

interface Candidate {
  row: number;
  local: number;
  catalogSlot: number;
  family: number;
  orientation: number;
  neighbor: number;
  coefficient: number;
  inverseDistance: number;
  fraction: number;
  value: number;
  normal: readonly [number, number, number];
  transition: boolean;
  physicalBoundary: boolean;
  stableKey: string;
}

interface OwnedSlot {
  key: string;
  owner: Candidate;
  other?: Candidate;
  transition: boolean;
  physicalBoundary: boolean;
}

function remappedStableKey(publication: StructuredVelocityPublication, handle: number,
  rowMap: Uint32Array): string | undefined {
  const key = publication.slotStableKeys[handle]!;
  const fields = key.split(":");
  if (fields[0] === "i") {
    const left = rowMap[Number(fields[1])];
    const right = rowMap[Number(fields[2])];
    if (left === undefined || right === undefined
      || left === OCTREE_STRUCTURED_VELOCITY_INVALID_HANDLE
      || right === OCTREE_STRUCTURED_VELOCITY_INVALID_HANDLE) return undefined;
    return `i:${Math.min(left, right)}:${Math.max(left, right)}`;
  }
  const row = rowMap[Number(fields[1])];
  if (row === undefined || row === OCTREE_STRUCTURED_VELOCITY_INVALID_HANDLE) return undefined;
  return `b:${row}:${fields.slice(2).join(":")}`;
}

/**
 * Publish canonical destination slots from catalog cases and resolved rows.
 * Each internal row pair must resolve each other exactly once and must provide
 * coefficient-identical records. There is deliberately no permissive path.
 */
export function publishStructuredVelocityFamilies(
  resolved: ResolvedPowerRowsPublication,
  catalog: StructuredVelocityCatalogSource,
  resolutions: readonly TransientStructuredVelocityRowResolution[],
  carry?: StructuredVelocityCarry,
): StructuredVelocityPublication {
  if (resolved.epoch === 0) throw new RangeError("Structured velocity epoch zero is unpublished");
  const entryCount = catalog.rowTemplateHeaders.length / OCTREE_POWER_ROW_TEMPLATE_HEADER_WORDS;
  if (!Number.isInteger(entryCount)
    || catalog.rowTemplateData.length !== catalog.rowTemplateSlots.length * OCTREE_POWER_ROW_TEMPLATE_FLOATS
    || catalog.reconstructionData.length !== catalog.rowTemplateSlots.length * OCTREE_POWER_RECONSTRUCTION_FLOATS
    || catalog.faceData.length !== catalog.rowTemplateSlots.length * 12) {
    throw new RangeError("Structured velocity catalog arrays have inconsistent lengths");
  }
  const resolutionByRow = new Map(resolutions.map((resolution) => [resolution.row, resolution]));
  if (resolutionByRow.size !== resolutions.length) {
    throw new RangeError("Structured velocity resolution contains duplicate rows");
  }
  const candidates: Candidate[] = [];
  let maximumCaseSlots = 0;
  for (const row of resolved.rows) {
    if (row.rowClass === OCTREE_POWER_ROW_CLASS.invalid) continue;
    if (row.catalogEntry >= entryCount) throw new RangeError(`Row ${row.row} catalog entry is absent`);
    const resolution = resolutionByRow.get(row.row);
    if (!resolution) throw new RangeError(`Row ${row.row} has no structured velocity resolution`);
    const header = row.catalogEntry * OCTREE_POWER_ROW_TEMPLATE_HEADER_WORDS;
    const first = catalog.rowTemplateHeaders[header]!;
    const count = catalog.rowTemplateHeaders[header + 1]!;
    maximumCaseSlots = Math.max(maximumCaseSlots, count);
    if (first > catalog.rowTemplateSlots.length || count > catalog.rowTemplateSlots.length - first
      || resolution.neighborRows.length !== count
      || resolution.normalVelocities && resolution.normalVelocities.length !== count
      || resolution.boundaryFractions && resolution.boundaryFractions.length !== count) {
      throw new RangeError(`Row ${row.row} structured velocity case-slot resolution is incomplete`);
    }
    const [transition, rowBoundary] = rowClassFlags(row.rowClass);
    for (let local = 0; local < count; local += 1) {
      const catalogSlot = first + local;
      const selector = unpackOctreePowerRowTemplateSlot(catalog.rowTemplateSlots[catalogSlot]!);
      if (selector.family >= OCTREE_STRUCTURED_VELOCITY_FAMILY_COUNT
        || selector.orientation >= OCTREE_STRUCTURED_VELOCITY_MAX_ORIENTATIONS) {
        throw new RangeError(`Row ${row.row} catalog slot leaves the structured family ABI`);
      }
      const neighbor = resolution.neighborRows[local]! >>> 0;
      if (selector.worldBoundary !== (neighbor === OCTREE_STRUCTURED_VELOCITY_INVALID_HANDLE)) {
        throw new RangeError(`Row ${row.row} boundary selector and neighbor disagree at slot ${local}`);
      }
      if (!selector.worldBoundary && (neighbor >= resolved.rows.length || neighbor === row.row
        || !row.neighborRows.includes(neighbor))) {
        throw new RangeError(`Row ${row.row} slot ${local} does not resolve to a published neighbor`);
      }
      const neighborIndex = selector.worldBoundary ? -1 : row.neighborRows.indexOf(neighbor);
      const coefficient = selector.worldBoundary
        ? Math.fround(catalog.rowTemplateData[catalogSlot * 3]! / row.scale)
        : Math.fround(row.neighborCoefficients[neighborIndex]!);
      const inverseDistance = Math.fround(catalog.rowTemplateData[catalogSlot * 3 + 2]! / row.scale);
      if (!(coefficient > 0) || !(inverseDistance > 0)) {
        throw new RangeError(`Row ${row.row} structured slot coefficient is not positive`);
      }
      const fraction = finite(resolution.boundaryFractions?.[local]
        ?? row.dynamicBoundaryFractions?.[selector.dynamicBoundarySlot] ?? 1,
      `Row ${row.row} boundary fraction`);
      if (fraction < 0 || fraction > 1) throw new RangeError("Structured boundary fraction must be in [0,1]");
      const value = finite(resolution.normalVelocities?.[local] ?? 0,
        `Row ${row.row} normal velocity`);
      const stableKey = selector.worldBoundary
        ? `b:${row.row}:${selector.family}:${selector.orientation}:${selector.dynamicBoundarySlot}`
        : `i:${Math.min(row.row, neighbor)}:${Math.max(row.row, neighbor)}`;
      candidates.push({
        row: row.row, local, catalogSlot, family: selector.family,
        orientation: selector.orientation, neighbor, coefficient, inverseDistance,
        fraction, value,
        normal: [catalog.faceData[catalogSlot * 12 + 8]!, catalog.faceData[catalogSlot * 12 + 9]!,
          catalog.faceData[catalogSlot * 12 + 10]!],
        transition, physicalBoundary: rowBoundary || selector.worldBoundary, stableKey,
      });
    }
  }
  if (resolutionByRow.size !== resolved.rows.filter((row) => row.rowClass !== OCTREE_POWER_ROW_CLASS.invalid).length) {
    throw new RangeError("Structured velocity resolution contains a row outside the live resolved authority");
  }

  const groups = new Map<string, Candidate[]>();
  for (const candidate of candidates) {
    const list = groups.get(candidate.stableKey);
    if (list) list.push(candidate); else groups.set(candidate.stableKey, [candidate]);
  }
  const owned: OwnedSlot[] = [];
  for (const [key, records] of groups) {
    const boundary = key.startsWith("b:");
    if (records.length !== (boundary ? 1 : 2)) {
      throw new RangeError(`Structured slot ${key} does not have ${boundary ? "one owner" : "two incidences"}`);
    }
    records.sort((left, right) => left.row - right.row || left.local - right.local);
    const owner = records[0]!, other = records[1];
    if (other) {
      if (owner.neighbor !== other.row || other.neighbor !== owner.row) {
        throw new RangeError(`Structured slot ${key} is not reciprocal`);
      }
      if (floatBits(owner.coefficient) !== floatBits(other.coefficient)
        || floatBits(owner.fraction) !== floatBits(other.fraction)
        || (floatBits(owner.value) !== floatBits(other.value)
          && floatBits(owner.value) !== floatBits(-other.value))) {
        throw new RangeError(`Structured slot ${key} has asymmetric authoritative data`);
      }
      const dot = owner.normal[0] * other.normal[0] + owner.normal[1] * other.normal[1]
        + owner.normal[2] * other.normal[2];
      if (dot > -0.9999) throw new RangeError(`Structured slot ${key} normals are not reciprocal`);
    }
    owned.push({
      key, owner, other,
      transition: owner.transition || (other?.transition ?? false),
      physicalBoundary: boundary || owner.physicalBoundary || (other?.physicalBoundary ?? false),
    });
  }
  owned.sort((left, right) => left.owner.family - right.owner.family
    || left.owner.row - right.owner.row || left.owner.local - right.owner.local
    || left.key.localeCompare(right.key));

  const slotCount = owned.length;
  const familyOffsets = new Uint32Array(OCTREE_STRUCTURED_VELOCITY_FAMILY_COUNT + 1);
  let cursor = 0;
  for (let family = 0; family < OCTREE_STRUCTURED_VELOCITY_FAMILY_COUNT; family += 1) {
    familyOffsets[family] = cursor;
    while (cursor < slotCount && owned[cursor]!.owner.family === family) cursor += 1;
  }
  familyOffsets[OCTREE_STRUCTURED_VELOCITY_FAMILY_COUNT] = slotCount;
  const values = new Float32Array(slotCount);
  const ownerRows = new Uint32Array(slotCount);
  const neighborRows = new Uint32Array(slotCount).fill(OCTREE_STRUCTURED_VELOCITY_INVALID_HANDLE);
  const orientation = new Uint32Array(slotCount);
  const coefficients = new Float32Array(slotCount);
  const inverseDistances = new Float32Array(slotCount);
  const boundaryFractions = new Float32Array(slotCount);
  const normalX = new Float32Array(slotCount), normalY = new Float32Array(slotCount),
    normalZ = new Float32Array(slotCount);
  const rowCapacity = resolved.plan.rowCapacity;
  const rowFamilyHandles = new Uint32Array(rowCapacity * OCTREE_STRUCTURED_VELOCITY_FAMILY_COUNT);
  const rowFamilySlotHandles = new Uint32Array(
    rowCapacity * OCTREE_STRUCTURED_VELOCITY_FAMILY_COUNT
      * OCTREE_STRUCTURED_VELOCITY_MAX_ORIENTATIONS,
  ).fill(OCTREE_STRUCTURED_VELOCITY_INVALID_HANDLE);
  const rowFamilySlotCounts = new Uint32Array(
    rowCapacity * OCTREE_STRUCTURED_VELOCITY_FAMILY_COUNT,
  );
  for (let row = 0; row < rowCapacity; row += 1) {
    for (let family = 0; family < OCTREE_STRUCTURED_VELOCITY_FAMILY_COUNT; family += 1) {
      const index = row * OCTREE_STRUCTURED_VELOCITY_FAMILY_COUNT + family;
      rowFamilyHandles[index] = index * OCTREE_STRUCTURED_VELOCITY_MAX_ORIENTATIONS;
    }
  }
  const rowSlotHandles = new Uint32Array(rowCapacity * maximumCaseSlots)
    .fill(OCTREE_STRUCTURED_VELOCITY_INVALID_HANDLE);
  const rowSlotSigns = new Int32Array(rowCapacity * maximumCaseSlots);
  const rowCatalogSlots = new Uint32Array(rowCapacity * maximumCaseSlots)
    .fill(OCTREE_STRUCTURED_VELOCITY_INVALID_HANDLE);
  const rowSlotCounts = new Uint32Array(rowCapacity);
  const familyByClass = new Map<StructuredVelocityClass, number[]>();
  for (const name of OCTREE_STRUCTURED_VELOCITY_CLASSES) familyByClass.set(name, []);

  owned.forEach((slot, handle) => {
    const source = slot.owner;
    values[handle] = source.value;
    ownerRows[handle] = source.row;
    neighborRows[handle] = slot.other?.row ?? OCTREE_STRUCTURED_VELOCITY_INVALID_HANDLE;
    orientation[handle] = source.family | (source.orientation << 3) | (source.catalogSlot << 6);
    coefficients[handle] = source.coefficient;
    inverseDistances[handle] = source.inverseDistance;
    boundaryFractions[handle] = source.fraction;
    normalX[handle] = source.normal[0]; normalY[handle] = source.normal[1]; normalZ[handle] = source.normal[2];
    const incidences: readonly (readonly [Candidate, number])[] = slot.other
      ? [[source, 1], [slot.other, -1]]
      : [[source, 1]];
    for (const [record, sign] of incidences) {
      const index = record.row * maximumCaseSlots + record.local;
      rowSlotHandles[index] = handle;
      rowSlotSigns[index] = sign;
      rowCatalogSlots[index] = record.catalogSlot;
      rowSlotCounts[record.row] = Math.max(rowSlotCounts[record.row]!, record.local + 1);
      const familyIndex = record.row * OCTREE_STRUCTURED_VELOCITY_FAMILY_COUNT + record.family;
      const familySlot = rowFamilyHandles[familyIndex]! + record.orientation;
      if (rowFamilySlotHandles[familySlot] !== OCTREE_STRUCTURED_VELOCITY_INVALID_HANDLE) {
        throw new RangeError(`Row ${record.row} repeats family ${record.family} orientation ${record.orientation}`);
      }
      rowFamilySlotHandles[familySlot] = handle;
      rowFamilySlotCounts[familyIndex] += 1;
    }
    familyByClass.get(className(slot.transition, slot.physicalBoundary))!.push(handle);
  });

  if (carry) {
    if (carry.oldToNewRows.length < carry.previous.rowCapacity) {
      throw new RangeError("Structured velocity carry remap is shorter than the prior row authority");
    }
    const current = new Map(owned.map((slot, handle) => [slot.key, handle]));
    const written = new Uint8Array(slotCount);
    for (let oldHandle = 0; oldHandle < carry.previous.slotCount; oldHandle += 1) {
      const key = remappedStableKey(carry.previous, oldHandle, carry.oldToNewRows);
      if (!key) continue;
      const handle = current.get(key);
      if (handle === undefined) continue;
      if (written[handle]) throw new RangeError(`Structured velocity carry aliases slot ${key}`);
      values[handle] = carry.previous.values[oldHandle]!;
      written[handle] = 1;
    }
  }

  const rowClasses = new Map<StructuredVelocityClass, number[]>();
  for (const name of OCTREE_STRUCTURED_VELOCITY_CLASSES) rowClasses.set(name, []);
  for (const row of resolved.rows) {
    if (row.rowClass === OCTREE_POWER_ROW_CLASS.invalid) continue;
    const [transition, boundary] = rowClassFlags(row.rowClass);
    rowClasses.get(className(transition, boundary))!.push(row.row);
  }
  const worksets = (source: Map<StructuredVelocityClass, number[]>) => Object.freeze(
    Object.fromEntries(OCTREE_STRUCTURED_VELOCITY_CLASSES.map((name) => [name,
      buildOctreeWorkset(source.get(name)!, {
        epoch: resolved.epoch, capacity: Math.max(rowCapacity, slotCount, 1), workgroupSize: 64,
      })])) as Record<StructuredVelocityClass, Uint32Array>,
  );
  return Object.freeze({
    epoch: resolved.epoch, rowCapacity, maximumCaseSlots, slotCount, familyOffsets,
    values, ownerRows, neighborRows, orientation, coefficients, inverseDistances, boundaryFractions,
    normalX, normalY, normalZ, rowFamilyHandles, rowFamilySlotHandles, rowFamilySlotCounts,
    rowSlotHandles, rowSlotSigns, rowCatalogSlots, rowSlotCounts,
    rowWorksets: worksets(rowClasses), familyWorksets: worksets(familyByClass),
    slotStableKeys: Object.freeze(owned.map((slot) => slot.key)),
  });
}

/** The six values written into `ResolvedPowerRow.velocityFamilyHandles`. */
export function structuredVelocityFamilyHandlesForRow(
  publication: StructuredVelocityPublication,
  row: number,
): Uint32Array {
  if (!Number.isSafeInteger(row) || row < 0 || row >= publication.rowCapacity) {
    throw new RangeError("Structured velocity row is outside the authority");
  }
  const first = row * OCTREE_STRUCTURED_VELOCITY_FAMILY_COUNT;
  return publication.rowFamilyHandles.slice(first, first + OCTREE_STRUCTURED_VELOCITY_FAMILY_COUNT);
}

/** Fixed catalog pseudoinverse reconstruction for differential fixtures. */
export function reconstructStructuredRowVelocity(publication: StructuredVelocityPublication,
  catalog: StructuredVelocityCatalogSource, row: number): readonly [number, number, number] {
  if (!Number.isSafeInteger(row) || row < 0 || row >= publication.rowCapacity) {
    throw new RangeError("Structured velocity row is outside the authority");
  }
  let x = 0, y = 0, z = 0;
  const count = publication.rowSlotCounts[row]!;
  for (let local = 0; local < count; local += 1) {
    const index = row * publication.maximumCaseSlots + local;
    const handle = publication.rowSlotHandles[index]!;
    if (handle === OCTREE_STRUCTURED_VELOCITY_INVALID_HANDLE) {
      throw new Error(`Structured velocity row ${row} has an unpublished case slot`);
    }
    const catalogSlot = publication.rowCatalogSlots[index]!;
    const sample = publication.values[handle]! * publication.rowSlotSigns[index]!;
    x += catalog.reconstructionData[catalogSlot * 3]! * sample;
    y += catalog.reconstructionData[catalogSlot * 3 + 1]! * sample;
    z += catalog.reconstructionData[catalogSlot * 3 + 2]! * sample;
  }
  return [x, y, z];
}

/** Destination-row divergence gather in fixed catalog order. */
export function structuredVelocityDivergence(publication: StructuredVelocityPublication,
  row: number, volume: number): number {
  if (!(volume > 0) || !Number.isFinite(volume)) throw new RangeError("Divergence volume must be positive");
  let flux = 0;
  const count = publication.rowSlotCounts[row]!;
  for (let local = 0; local < count; local += 1) {
    const index = row * publication.maximumCaseSlots + local;
    const handle = publication.rowSlotHandles[index]!;
    const area = publication.coefficients[handle]! / publication.inverseDistances[handle]!;
    flux += publication.rowSlotSigns[index]! * publication.values[handle]!
      * area * publication.boundaryFractions[handle]!;
  }
  return flux / volume;
}

/** Unique-slot pressure projection. Each handle is written exactly once. */
export function projectStructuredVelocity(publication: StructuredVelocityPublication,
  pressure: Float32Array, dtOverDensity: number): Float32Array {
  finite(dtOverDensity, "Projection dt/rho");
  const output = publication.values.slice();
  for (let handle = 0; handle < publication.slotCount; handle += 1) {
    const neighbor = publication.neighborRows[handle]!;
    if (neighbor === OCTREE_STRUCTURED_VELOCITY_INVALID_HANDLE) continue;
    const owner = publication.ownerRows[handle]!;
    output[handle] = Math.fround(publication.values[handle]!
      - dtOverDensity * (pressure[neighbor]! - pressure[owner]!)
      * publication.inverseDistances[handle]! * publication.boundaryFractions[handle]!);
  }
  return output;
}

/**
 * Destination-owned Eulerian update. The sampler reconstructs/traces/samples
 * the full vector; this function performs the single final normal projection.
 */
export function advectStructuredVelocityDestinationOwned(
  publication: StructuredVelocityPublication,
  sampleFullVector: (handle: number) => readonly [number, number, number],
): Float32Array {
  const output = new Float32Array(publication.slotCount);
  for (let handle = 0; handle < publication.slotCount; handle += 1) {
    const velocity = sampleFullVector(handle);
    if (velocity.length !== 3 || velocity.some((value) => !Number.isFinite(value))) {
      throw new RangeError(`Structured velocity sample ${handle} is not a finite full vector`);
    }
    output[handle] = Math.fround(velocity[0] * publication.normalX[handle]!
      + velocity[1] * publication.normalY[handle]!
      + velocity[2] * publication.normalZ[handle]!);
  }
  return output;
}
