/**
 * Authoritative bounded brick-page substrate for the octree pressure owner map.
 *
 * Recurring topology publications are exact sorted logical-brick records whose
 * physical page IDs remain stable while the logical brick remains resident.
 * Candidate generations fail closed before replacing the immutable accepted
 * publication.
 */

import { PassBroker } from "./webgpu-pass-broker";

export const OCTREE_OWNER_BRICK_SIZE = 8 as const;
export const OCTREE_OWNER_PAGE_VOXELS = OCTREE_OWNER_BRICK_SIZE ** 3;
export const OCTREE_OWNER_ARENA_CONTROL_WORDS = 16;
export const OCTREE_OWNER_ARENA_MAGIC = 0x4f57_4e52;
export const OCTREE_OWNER_PAGE_WORD_VALID = 0x8000_0000;
/** Resident page belongs to the moving accepted-topology worklist, rather
 * than only to the immutable analytic lookup halo. */
export const OCTREE_OWNER_PAGE_WORD_TOPOLOGY = 0x0020_0000;
export const OCTREE_OWNER_PAGE_TABLE_MISSING = 0;
export const OCTREE_OWNER_PAGE_TABLE_RESERVED = 0xffff_ffff;

/** Publication status bits stored in owner-arena control word 10. */
export const OCTREE_OWNER_PAGE_PUBLICATION_STATUS = Object.freeze({
  ready: 1 << 0,
  unchanged: 1 << 1,
  stale: 1 << 2,
  unpublished: 1 << 3,
  overflow: 1 << 6,
} as const);
/** High status bit selecting immutable owner table/payload bank B. */
export const OCTREE_OWNER_PAGE_ACTIVE_TABLE_B = 0x8000_0000;

/** Exact immutable 16-word owner-arena control ABI. */
export const OCTREE_OWNER_PAGE_CONTROL_WORDS = Object.freeze({
  freeCount: 0,
  residentCount: 1,
  candidateError: 2,
  capacity: 3,
  logicalBrickCount: 4,
  ownerRecordPageOffsetWords: 5,
  ownerPagesOffsetWords: 6,
  acceptedGeneration: 7,
  activatedCount: 8,
  retiredCount: 9,
  status: 10,
  observedGeneration: 11,
  invalidEntryCount: 12,
  tileListCapacity: 13,
  tileSizeCells: 14,
  magic: 15,
} as const);

export interface OctreeOwnerPageControl {
  readonly freeCount: number;
  readonly residentCount: number;
  readonly candidateError: number;
  readonly capacity: number;
  readonly logicalBrickCount: number;
  readonly ownerRecordPageOffsetWords: number;
  readonly ownerPagesOffsetWords: number;
  readonly acceptedGeneration: number;
  readonly activatedCount: number;
  readonly retiredCount: number;
  readonly status: number;
  readonly observedGeneration: number;
  readonly invalidEntryCount: number;
  readonly tileListCapacity: number;
  readonly tileSizeCells: number;
  readonly magic: number;
}

/** Decode the complete owner-arena control packet without legacy field aliases. */
export function unpackOctreeOwnerPageControl(words: ArrayLike<number>): OctreeOwnerPageControl {
  if (words.length < OCTREE_OWNER_ARENA_CONTROL_WORDS) {
    throw new RangeError(`Octree owner control needs ${OCTREE_OWNER_ARENA_CONTROL_WORDS} words`);
  }
  const word = (name: keyof typeof OCTREE_OWNER_PAGE_CONTROL_WORDS) =>
    Number(words[OCTREE_OWNER_PAGE_CONTROL_WORDS[name]]) >>> 0;
  return {
    freeCount: word("freeCount"),
    residentCount: word("residentCount"),
    candidateError: word("candidateError"),
    capacity: word("capacity"),
    logicalBrickCount: word("logicalBrickCount"),
    ownerRecordPageOffsetWords: word("ownerRecordPageOffsetWords"),
    ownerPagesOffsetWords: word("ownerPagesOffsetWords"),
    acceptedGeneration: word("acceptedGeneration"),
    activatedCount: word("activatedCount"),
    retiredCount: word("retiredCount"),
    status: word("status"),
    observedGeneration: word("observedGeneration"),
    invalidEntryCount: word("invalidEntryCount"),
    tileListCapacity: word("tileListCapacity"),
    tileSizeCells: word("tileSizeCells"),
    magic: word("magic"),
  };
}

export type OctreeOwnerLeafSize = 1 | 2 | 4 | 8 | 16 | 32;
export type OctreeOwnerCoordinate = readonly [number, number, number];

export interface OctreeOwnerRecord {
  origin: readonly [number, number, number];
  size: OctreeOwnerLeafSize;
  missing: boolean;
}

export interface OctreeOwnerPagePlanOptions {
  brickSize?: 8;
  maximumResidentFraction?: number;
  maximumPages?: number;
  /** Correctness floor for a bounded bootstrap publication. */
  minimumPages?: number;
  /** Optional device/allocation ceiling. Capacity degrades to fit this bound. */
  maximumArenaBytes?: number;
  /**
   * Compact-simulation capacity model. Pressure rows cover the volumetric
   * topology while fine surface rows cover the two-dimensional refinement
   * sheet. A 50% overlap/headroom allowance is applied before clamping to the
   * logical brick lattice. This is a conservative operational bound, not a
   * proof for an arbitrarily folded interface: an exact smaller bound requires
   * the topology producer to publish its refined-brick count. Until then,
   * overflow remains fail-closed to canonical owners and invalidates the solve.
   */
  adaptiveBounds?: {
    pressureRowCapacity: number;
    fineSeedLeafCapacity: number;
  };
}

export interface OctreeOwnerPagePlan {
  dimensions: readonly [number, number, number];
  brickSize: 8;
  brickDimensions: readonly [number, number, number];
  logicalBrickCount: number;
  requestedCapacity: number;
  minimumCapacity: number;
  adaptiveCapacity?: number;
  capacity: number;
  degraded: boolean;
  pageVoxels: number;
  bytesPerPage: number;
  controlOffsetWords: number;
  /** Sorted logical-brick keys (`logical + 1`) for the accepted generation. */
  ownerRecordKeyOffsetWords: number;
  /** Stable encoded physical page IDs paired one-to-one with sorted keys. */
  ownerRecordPageOffsetWords: number;
  ownerRecordKeyBOffsetWords: number;
  ownerRecordPageBOffsetWords: number;
  ownerRecordCapacity: number;
  /** Direct logical-brick -> encoded physical page directory. Publication
   * updates only added/carried/retired identities; hot consumers never search
   * the sorted diagnostic records. */
  ownerDirectoryOffsetWords: number;
  ownerDirectoryBOffsetWords: number;
  ownerPagesOffsetWords: number;
  ownerPagesBOffsetWords: number;
  allocatedWords: number;
  allocatedBytes: number;
  denseOwnerBytes: number;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}

function coordinate(value: OctreeOwnerCoordinate, label: string): readonly [number, number, number] {
  value.forEach((component, axis) => {
    if (!Number.isSafeInteger(component) || component < 0) throw new RangeError(`${label} axis ${axis} must be a non-negative integer`);
  });
  return value;
}

function leafSize(value: number): OctreeOwnerLeafSize {
  if (![1, 2, 4, 8, 16, 32].includes(value)) throw new RangeError("Octree owner leaf size must be one of 1, 2, 4, 8, 16, or 32");
  return value as OctreeOwnerLeafSize;
}

export function planOctreeOwnerPages(
  dimensions: OctreeOwnerCoordinate,
  options: OctreeOwnerPagePlanOptions = {},
): OctreeOwnerPagePlan {
  dimensions.forEach((value, axis) => positiveInteger(value, `Octree owner dimension ${axis}`));
  if (options.brickSize !== undefined && options.brickSize !== OCTREE_OWNER_BRICK_SIZE) {
    throw new RangeError("Octree owner pages currently require 8-cubed bricks");
  }
  const brickDimensions = dimensions.map((value) => Math.ceil(value / OCTREE_OWNER_BRICK_SIZE)) as [number, number, number];
  const logicalBrickCount = brickDimensions[0] * brickDimensions[1] * brickDimensions[2];
  const fraction = options.maximumResidentFraction === undefined
    ? 1
    : Math.max(1 / logicalBrickCount, Math.min(1, options.maximumResidentFraction));
  if (!Number.isFinite(fraction)) throw new RangeError("Octree owner resident fraction must be finite");
  const fractionalCapacity = Math.max(1, Math.ceil(logicalBrickCount * fraction));
  const hardCapacity = options.maximumPages === undefined
    ? logicalBrickCount
    : Math.max(1, Math.min(logicalBrickCount, Math.floor(options.maximumPages)));
  if (!Number.isFinite(hardCapacity)) throw new RangeError("Octree owner maximum pages must be finite");
  const minimumCapacity = options.minimumPages === undefined
    ? 1
    : Math.max(1, Math.min(logicalBrickCount, Math.floor(options.minimumPages)));
  if (!Number.isFinite(minimumCapacity)) throw new RangeError("Octree owner minimum pages must be finite");
  let adaptiveCapacity: number | undefined;
  if (options.adaptiveBounds !== undefined) {
    const { pressureRowCapacity, fineSeedLeafCapacity } = options.adaptiveBounds;
    positiveInteger(pressureRowCapacity, "Octree owner pressure-row capacity");
    positiveInteger(fineSeedLeafCapacity, "Octree owner fine-seed leaf capacity");
    // Bulk rows amortize over an 8^3 owner page. Fine interface rows amortize
    // only over its 8^2 cross-section. The 3/2 multiplier covers overlap,
    // 2:1 grading, and one-frame residency hysteresis without returning to a
    // fixed percentage of the entire bounding volume.
    const pressurePages = Math.ceil(pressureRowCapacity / OCTREE_OWNER_PAGE_VOXELS);
    const fineSeedPages = Math.ceil(fineSeedLeafCapacity / (OCTREE_OWNER_BRICK_SIZE ** 2));
    adaptiveCapacity = Math.min(logicalBrickCount, Math.max(1, Math.ceil((pressurePages + fineSeedPages) * 3 / 2)));
  }
  const requestedCapacity = Math.min(logicalBrickCount, Math.max(
    minimumCapacity,
    Math.min(fractionalCapacity, hardCapacity, adaptiveCapacity ?? logicalBrickCount),
  ));
  const wordsFor = (residentCapacity: number) =>
    OCTREE_OWNER_ARENA_CONTROL_WORDS + 2 * logicalBrickCount
      + residentCapacity * (4 + 2 * OCTREE_OWNER_PAGE_VOXELS);
  let deviceCapacity = requestedCapacity;
  if (options.maximumArenaBytes !== undefined) {
    if (!Number.isFinite(options.maximumArenaBytes) || options.maximumArenaBytes < 0) {
      throw new RangeError("Octree owner arena byte ceiling must be finite and non-negative");
    }
    const maximumWords = Math.floor(options.maximumArenaBytes / 4);
    let low = 0;
    let high = requestedCapacity;
    while (low < high) {
      const middle = Math.ceil((low + high) / 2);
      if (wordsFor(middle) <= maximumWords) low = middle;
      else high = middle - 1;
    }
    deviceCapacity = low;
    if (deviceCapacity < 1) throw new RangeError("Octree owner arena byte ceiling cannot hold one physical page");
  }
  const capacity = Math.min(requestedCapacity, deviceCapacity);
  const ownerRecordCapacity = capacity;
  const ownerRecordKeyOffsetWords = OCTREE_OWNER_ARENA_CONTROL_WORDS;
  const ownerRecordPageOffsetWords = ownerRecordKeyOffsetWords + ownerRecordCapacity;
  const ownerRecordKeyBOffsetWords = ownerRecordPageOffsetWords + ownerRecordCapacity;
  const ownerRecordPageBOffsetWords = ownerRecordKeyBOffsetWords + ownerRecordCapacity;
  const ownerDirectoryOffsetWords = ownerRecordPageBOffsetWords + ownerRecordCapacity;
  const ownerDirectoryBOffsetWords = ownerDirectoryOffsetWords + logicalBrickCount;
  const ownerPagesOffsetWords = ownerDirectoryBOffsetWords + logicalBrickCount;
  const ownerPagesBOffsetWords = ownerPagesOffsetWords + capacity * OCTREE_OWNER_PAGE_VOXELS;
  const allocatedWords = ownerPagesBOffsetWords + capacity * OCTREE_OWNER_PAGE_VOXELS;
  return {
    dimensions: [...dimensions] as [number, number, number],
    brickSize: OCTREE_OWNER_BRICK_SIZE,
    brickDimensions,
    logicalBrickCount,
    requestedCapacity,
    minimumCapacity,
    adaptiveCapacity,
    capacity,
    degraded: capacity < requestedCapacity,
    pageVoxels: OCTREE_OWNER_PAGE_VOXELS,
    bytesPerPage: (2 + OCTREE_OWNER_PAGE_VOXELS) * 4,
    controlOffsetWords: 0,
    ownerRecordKeyOffsetWords,
    ownerRecordPageOffsetWords,
    ownerRecordKeyBOffsetWords,
    ownerRecordPageBOffsetWords,
    ownerRecordCapacity,
    ownerDirectoryOffsetWords,
    ownerDirectoryBOffsetWords,
    ownerPagesOffsetWords,
    ownerPagesBOffsetWords,
    allocatedWords,
    allocatedBytes: allocatedWords * 4,
    denseOwnerBytes: dimensions[0] * dimensions[1] * dimensions[2] * 8,
  };
}

/** Pack one owner relative to the 8-cubed brick containing `cell`. */
export function packOctreeOwnerPageWord(
  cellValue: OctreeOwnerCoordinate,
  ownerOriginValue: OctreeOwnerCoordinate,
  ownerSizeValue: OctreeOwnerLeafSize,
): number {
  const cell = coordinate(cellValue, "Octree owner cell");
  const origin = coordinate(ownerOriginValue, "Octree owner origin");
  const size = leafSize(ownerSizeValue);
  for (let axis = 0; axis < 3; axis += 1) {
    if (cell[axis] < origin[axis] || cell[axis] >= origin[axis] + size) {
      throw new RangeError("Octree owner cell must lie inside its leaf");
    }
  }
  const brickOrigin = cell.map((value) => Math.floor(value / OCTREE_OWNER_BRICK_SIZE) * OCTREE_OWNER_BRICK_SIZE);
  const delta = origin.map((value, axis) => value - brickOrigin[axis]);
  delta.forEach((value) => {
    if (value < -32 || value > 31) throw new RangeError("Octree owner origin does not fit the signed six-bit brick-relative encoding");
  });
  const exponent = Math.log2(size);
  return (OCTREE_OWNER_PAGE_WORD_VALID
    | ((delta[0] + 32) & 63)
    | (((delta[1] + 32) & 63) << 6)
    | (((delta[2] + 32) & 63) << 12)
    | ((exponent & 7) << 18)) >>> 0;
}

/** Decode a valid packed owner word. Missing words are handled separately. */
export function unpackOctreeOwnerPageWord(
  word: number,
  cellValue: OctreeOwnerCoordinate,
): OctreeOwnerRecord {
  const cell = coordinate(cellValue, "Octree owner cell");
  const packed = word >>> 0;
  if ((packed & OCTREE_OWNER_PAGE_WORD_VALID) === 0) throw new RangeError("Cannot unpack a missing octree owner page word");
  const exponent = (packed >>> 18) & 7;
  if (exponent > 5) throw new RangeError("Packed octree owner leaf exponent is invalid");
  const brickOrigin = cell.map((value) => Math.floor(value / OCTREE_OWNER_BRICK_SIZE) * OCTREE_OWNER_BRICK_SIZE);
  const delta = [packed & 63, (packed >>> 6) & 63, (packed >>> 12) & 63].map((value) => value - 32);
  const origin = brickOrigin.map((value, axis) => value + delta[axis]) as [number, number, number];
  if (origin.some((value) => value < 0)) throw new RangeError("Packed octree owner origin falls outside the non-negative domain");
  return { origin, size: leafSize(1 << exponent), missing: false };
}

/** Canonical coarse air owner used when a logical brick has no physical page. */
export function canonicalMissingAirOwner(
  cellValue: OctreeOwnerCoordinate,
  dimensionsValue: OctreeOwnerCoordinate,
  maximumLeafSizeValue: OctreeOwnerLeafSize,
): OctreeOwnerRecord {
  const cell = coordinate(cellValue, "Octree owner cell");
  const dimensions = coordinate(dimensionsValue, "Octree owner dimensions");
  dimensions.forEach((value, axis) => {
    positiveInteger(value, `Octree owner dimension ${axis}`);
    if (cell[axis] >= value) throw new RangeError("Missing-air owner cell must lie inside the domain");
  });
  let size = leafSize(maximumLeafSizeValue);
  let origin = cell.map((value) => Math.floor(value / size) * size) as [number, number, number];
  while (size > 1 && origin.some((value, axis) => value + size > dimensions[axis])) {
    size = leafSize(size / 2);
    origin = cell.map((value) => Math.floor(value / size) * size) as [number, number, number];
  }
  return { origin, size, missing: true };
}

export function decodeOctreeOwnerPageWord(
  word: number,
  cell: OctreeOwnerCoordinate,
  dimensions: OctreeOwnerCoordinate,
  maximumLeafSize: OctreeOwnerLeafSize,
): OctreeOwnerRecord {
  return ((word >>> 0) & OCTREE_OWNER_PAGE_WORD_VALID) !== 0
    ? unpackOctreeOwnerPageWord(word, cell)
    : canonicalMissingAirOwner(cell, dimensions, maximumLeafSize);
}

/** Stable status bits returned by both the CPU oracle and WGSL lookup ABI. */
export const OCTREE_OWNER_PAGE_LOOKUP_STATUS = Object.freeze({
  missing: 1 << 0,
  invalid: 1 << 1,
  topology: 1 << 2,
} as const);

/** Locate a logical brick in the accepted sorted owner records. */
export function findOctreeOwnerPageRecord(
  arena: ArrayLike<number>,
  plan: Pick<OctreeOwnerPagePlan,
    "ownerRecordKeyOffsetWords" | "ownerRecordKeyBOffsetWords" | "ownerRecordCapacity">,
  logical: number,
): number {
  if (!Number.isSafeInteger(logical) || logical < 0 || logical >= 0xffff_fffe) return -1;
  const key = logical + 1;
  const resident = Math.min(Number(arena[OCTREE_OWNER_PAGE_CONTROL_WORDS.residentCount]) >>> 0,
    plan.ownerRecordCapacity);
  const recordKeyOffset = ((Number(arena[OCTREE_OWNER_PAGE_CONTROL_WORDS.status]) >>> 0)
    & OCTREE_OWNER_PAGE_ACTIVE_TABLE_B) === 0
    ? plan.ownerRecordKeyOffsetWords : plan.ownerRecordKeyBOffsetWords;
  let low = 0;
  let high = resident;
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2);
    const word = recordKeyOffset + middle;
    if (word >= arena.length) return -1;
    const observed = Number(arena[word]) >>> 0;
    if (observed < key) low = middle + 1;
    else high = middle;
  }
  return low < resident
    && (Number(arena[recordKeyOffset + low]) >>> 0) === key ? low : -1;
}

export interface OctreeOwnerPageLookupResult extends OctreeOwnerRecord {
  /** Zero for a resident, valid owner; otherwise `OCTREE_OWNER_PAGE_LOOKUP_STATUS` bits. */
  status: number;
}

function missingOwnerLookup(
  cellValue: readonly [number, number, number],
  dimensions: OctreeOwnerCoordinate,
  maximumLeafSize: OctreeOwnerLeafSize,
  invalid: boolean,
): OctreeOwnerPageLookupResult {
  const cell = cellValue.map((value, axis) => {
    const finite = Number.isFinite(value) ? Math.trunc(value) : 0;
    return Math.max(0, Math.min(dimensions[axis] - 1, finite));
  }) as [number, number, number];
  return {
    ...canonicalMissingAirOwner(cell, dimensions, maximumLeafSize),
    status: OCTREE_OWNER_PAGE_LOOKUP_STATUS.missing
      | (invalid ? OCTREE_OWNER_PAGE_LOOKUP_STATUS.invalid : 0),
  };
}

/**
 * CPU oracle for the owner-page lookup ABI below.
 *
 * Page-table entries are physical slot + 1. Every arena access is bounded;
 * nonresident, reserved, truncated, and malformed pages all fail closed to a
 * deterministic coarse-air owner. Packed owners are relative to the brick
 * containing `cell`, so the same leaf can be decoded from either side of a
 * brick seam (including a size-32 leaf spanning four bricks per axis).
 */
export function lookupOctreeOwnerPage(
  arena: ArrayLike<number>,
  plan: OctreeOwnerPagePlan,
  cellValue: readonly [number, number, number],
  maximumLeafSizeValue: OctreeOwnerLeafSize,
): OctreeOwnerPageLookupResult {
  const maximumLeafSize = leafSize(maximumLeafSizeValue);
  const invalidCell = cellValue.some((value, axis) =>
    !Number.isSafeInteger(value) || value < 0 || value >= plan.dimensions[axis]);
  if (invalidCell) return missingOwnerLookup(cellValue, plan.dimensions, maximumLeafSize, true);
  const cell = cellValue as OctreeOwnerCoordinate;
  const brick = cell.map((value) => Math.floor(value / OCTREE_OWNER_BRICK_SIZE));
  const logical = brick[0]
    + brick[1] * plan.brickDimensions[0]
    + brick[2] * plan.brickDimensions[0] * plan.brickDimensions[1];
  if (logical < 0 || logical >= plan.logicalBrickCount) {
    return missingOwnerLookup(cellValue, plan.dimensions, maximumLeafSize, true);
  }
  const table = ((Number(arena[OCTREE_OWNER_PAGE_CONTROL_WORDS.status]) >>> 0)
    & OCTREE_OWNER_PAGE_ACTIVE_TABLE_B) === 0 ? 0 : 1;
  const directoryOffset = table === 0
    ? plan.ownerDirectoryOffsetWords : plan.ownerDirectoryBOffsetWords;
  const payloadOffset = table === 0 ? plan.ownerPagesOffsetWords : plan.ownerPagesBOffsetWords;
  const valueWord = directoryOffset + logical;
  if (valueWord >= arena.length) return missingOwnerLookup(cellValue, plan.dimensions, maximumLeafSize, true);
  const encodedPage = Number(arena[valueWord]) >>> 0;
  if (encodedPage === OCTREE_OWNER_PAGE_TABLE_MISSING) {
    return missingOwnerLookup(cellValue, plan.dimensions, maximumLeafSize, false);
  }
  if (encodedPage === OCTREE_OWNER_PAGE_TABLE_RESERVED || encodedPage > plan.capacity) {
    return missingOwnerLookup(cellValue, plan.dimensions, maximumLeafSize, true);
  }
  const local = (cell[0] & 7) + (cell[1] & 7) * 8 + (cell[2] & 7) * 64;
  const payloadWord = payloadOffset + (encodedPage - 1) * OCTREE_OWNER_PAGE_VOXELS + local;
  if (!Number.isSafeInteger(payloadWord) || payloadWord < 0 || payloadWord >= arena.length) {
    return missingOwnerLookup(cellValue, plan.dimensions, maximumLeafSize, true);
  }
  const packed = Number(arena[payloadWord]) >>> 0;
  if ((packed & OCTREE_OWNER_PAGE_WORD_VALID) === 0) {
    return missingOwnerLookup(cellValue, plan.dimensions, maximumLeafSize, packed !== 0);
  }
  try {
    const owner = unpackOctreeOwnerPageWord(packed, cell);
    const containsCell = owner.origin.every((origin, axis) =>
      cell[axis] >= origin && cell[axis] < origin + owner.size);
    const insideDomain = owner.origin.every((origin, axis) =>
      origin + owner.size <= plan.dimensions[axis]);
    if (owner.size > maximumLeafSize || !containsCell || !insideDomain) {
      return missingOwnerLookup(cellValue, plan.dimensions, maximumLeafSize, true);
    }
    return { ...owner, status: (packed & OCTREE_OWNER_PAGE_WORD_TOPOLOGY) !== 0
      ? OCTREE_OWNER_PAGE_LOOKUP_STATUS.topology : 0 };
  } catch {
    return missingOwnerLookup(cellValue, plan.dimensions, maximumLeafSize, true);
  }
}

/**
 * Reusable read-only WGSL owner-page lookup ABI.
 *
 * The consuming shader declares these two bindings (at any group/bindings):
 *
 * `var<storage, read> ownerPageArena: array<u32>`
 * `var<uniform> ownerPageLookupParams: OctreeOwnerPageLookupParams`
 *
 * `octreeOwnerPageLookup(vec3i(cell))` returns a resident owner or canonical
 * air. The ABI validates `arrayLength` before every page-table/payload access.
 */
export const octreeOwnerPageLookupWgsl = /* wgsl */ `
struct OctreeOwnerPageLookupParams {
  dimensionsMaximumLeaf: vec4u,          // xyz dimensions, maximum leaf size
  brickDimensionsLogicalCount: vec4u,   // xyz brick dimensions, logical count
  arenaOffsetsCapacity: vec4u,           // direct directory, payload, capacity, page voxels
}
struct OctreeOwnerPageLookupResult {
  origin: vec3u,
  size: u32,
  status: u32,
}

const OWNER_PAGE_LOOKUP_MISSING: u32 = ${OCTREE_OWNER_PAGE_LOOKUP_STATUS.missing}u;
const OWNER_PAGE_LOOKUP_INVALID: u32 = ${OCTREE_OWNER_PAGE_LOOKUP_STATUS.invalid}u;
const OWNER_PAGE_LOOKUP_TOPOLOGY: u32 = ${OCTREE_OWNER_PAGE_LOOKUP_STATUS.topology}u;
const OWNER_PAGE_WORD_VALID: u32 = ${OCTREE_OWNER_PAGE_WORD_VALID}u;
const OWNER_PAGE_WORD_TOPOLOGY: u32 = ${OCTREE_OWNER_PAGE_WORD_TOPOLOGY}u;
const OWNER_PAGE_TABLE_RESERVED: u32 = ${OCTREE_OWNER_PAGE_TABLE_RESERVED}u;
const OWNER_PAGE_BRICK_SIZE: u32 = ${OCTREE_OWNER_BRICK_SIZE}u;
const OWNER_PAGE_VOXELS: u32 = ${OCTREE_OWNER_PAGE_VOXELS}u;

fn ownerPageSupportedLeafSize(size: u32) -> bool {
  return size == 1u || size == 2u || size == 4u || size == 8u || size == 16u || size == 32u;
}

fn ownerPageCanonicalAir(cellValue: vec3i, statusValue: u32) -> OctreeOwnerPageLookupResult {
  let dimensions = ownerPageLookupParams.dimensionsMaximumLeaf.xyz;
  var result: OctreeOwnerPageLookupResult;
  result.origin = vec3u(0u);
  result.size = 1u;
  result.status = statusValue | OWNER_PAGE_LOOKUP_MISSING;
  if (any(dimensions == vec3u(0u)) || any(dimensions > vec3u(0x7fffffffu))) {
    result.status |= OWNER_PAGE_LOOKUP_INVALID;
    return result;
  }
  let upper = vec3i(dimensions - vec3u(1u));
  let cell = vec3u(clamp(cellValue, vec3i(0), upper));
  var size = ownerPageLookupParams.dimensionsMaximumLeaf.w;
  if (!ownerPageSupportedLeafSize(size)) {
    size = 1u;
    result.status |= OWNER_PAGE_LOOKUP_INVALID;
  }
  loop {
    let origin = (cell / vec3u(size)) * vec3u(size);
    let sizeFits = all(vec3u(size) <= dimensions);
    let originFits = sizeFits && all(origin <= dimensions - vec3u(size));
    if (originFits || size == 1u) {
      result.origin = origin;
      result.size = size;
      return result;
    }
    size >>= 1u;
  }
  // The size==1 branch above is terminating for every u32 input; retain an
  // explicit fail-closed return so conservative WGSL validators can prove the
  // function's return contract without reasoning about the loop invariant.
  result.status |= OWNER_PAGE_LOOKUP_INVALID;
  return result;
}

fn ownerPageInvalidAir(cell: vec3i) -> OctreeOwnerPageLookupResult {
  return ownerPageCanonicalAir(cell, OWNER_PAGE_LOOKUP_INVALID);
}

fn octreeOwnerPageLookup(cell: vec3i) -> OctreeOwnerPageLookupResult {
  let dimensions = ownerPageLookupParams.dimensionsMaximumLeaf.xyz;
  let maximumLeaf = ownerPageLookupParams.dimensionsMaximumLeaf.w;
  let brickDimensions = ownerPageLookupParams.brickDimensionsLogicalCount.xyz;
  let logicalCount = ownerPageLookupParams.brickDimensionsLogicalCount.w;
  let directoryAOffset = ownerPageLookupParams.arenaOffsetsCapacity.x;
  let payloadAOffset = ownerPageLookupParams.arenaOffsetsCapacity.y;
  let capacity = ownerPageLookupParams.arenaOffsetsCapacity.z;
  let pageVoxels = ownerPageLookupParams.arenaOffsetsCapacity.w;
  let table = ownerPageArena[${OCTREE_OWNER_PAGE_CONTROL_WORDS.status}u] >> 31u;
  let directoryOffset = directoryAOffset + table * logicalCount;
  let payloadOffset = payloadAOffset + table * capacity * pageVoxels;
  if (any(dimensions == vec3u(0u)) || any(dimensions > vec3u(0x7fffffffu))
      || !ownerPageSupportedLeafSize(maximumLeaf) || any(brickDimensions == vec3u(0u))
      || capacity == 0u || pageVoxels != OWNER_PAGE_VOXELS) {
    return ownerPageInvalidAir(cell);
  }
  if (any(cell < vec3i(0)) || any(vec3u(cell) >= dimensions)) {
    return ownerPageInvalidAir(cell);
  }
  let unsignedCell = vec3u(cell);
  let brick = unsignedCell / vec3u(OWNER_PAGE_BRICK_SIZE);
  if (any(brick >= brickDimensions)) { return ownerPageInvalidAir(cell); }
  if (brickDimensions.x > 0xffffffffu / brickDimensions.y) { return ownerPageInvalidAir(cell); }
  let brickLayer = brickDimensions.x * brickDimensions.y;
  if (brick.z > 0xffffffffu / brickLayer) { return ownerPageInvalidAir(cell); }
  let zOffset = brick.z * brickLayer;
  if (brick.y > (0xffffffffu - zOffset) / brickDimensions.x) { return ownerPageInvalidAir(cell); }
  let yzOffset = zOffset + brick.y * brickDimensions.x;
  if (brick.x > 0xffffffffu - yzOffset) { return ownerPageInvalidAir(cell); }
  let logical = yzOffset + brick.x;
  if (logical >= logicalCount) { return ownerPageInvalidAir(cell); }
  let arenaWords = arrayLength(&ownerPageArena);
  if (payloadAOffset != directoryAOffset + 2u * logicalCount) {
    return ownerPageInvalidAir(cell);
  }
  if (directoryOffset >= arenaWords || logicalCount > arenaWords - directoryOffset) {
    return ownerPageInvalidAir(cell);
  }
  let encodedPage = ownerPageArena[directoryOffset + logical];
  if (encodedPage == 0u) {
    return ownerPageCanonicalAir(cell, 0u);
  }
  if (encodedPage == OWNER_PAGE_TABLE_RESERVED || encodedPage > capacity) {
    return ownerPageInvalidAir(cell);
  }
  let physicalSlot = encodedPage - 1u;
  let local = unsignedCell % vec3u(OWNER_PAGE_BRICK_SIZE);
  let localIndex = local.x + local.y * 8u + local.z * 64u;
  if (payloadOffset >= arenaWords || physicalSlot > (arenaWords - payloadOffset - 1u) / pageVoxels) {
    return ownerPageInvalidAir(cell);
  }
  let pageBase = payloadOffset + physicalSlot * pageVoxels;
  if (localIndex >= arenaWords - pageBase) { return ownerPageInvalidAir(cell); }
  let packed = ownerPageArena[pageBase + localIndex];
  if ((packed & OWNER_PAGE_WORD_VALID) == 0u) {
    return ownerPageCanonicalAir(cell, select(OWNER_PAGE_LOOKUP_INVALID, 0u, packed == 0u));
  }
  let exponent = (packed >> 18u) & 7u;
  if (exponent > 5u) { return ownerPageInvalidAir(cell); }
  let size = 1u << exponent;
  if (size > maximumLeaf) { return ownerPageInvalidAir(cell); }
  let brickOrigin = vec3i(brick * vec3u(OWNER_PAGE_BRICK_SIZE));
  let delta = vec3i(
    i32(packed & 63u) - 32,
    i32((packed >> 6u) & 63u) - 32,
    i32((packed >> 12u) & 63u) - 32,
  );
  // Positive deltas can overflow i32 only at the upper signed boundary.
  if (any(delta > vec3i(0)) && any(brickOrigin > vec3i(0x7fffffff) - max(delta, vec3i(0)))) {
    return ownerPageInvalidAir(cell);
  }
  if (any(delta < vec3i(0)) && any(brickOrigin < -min(delta, vec3i(0)))) {
    return ownerPageInvalidAir(cell);
  }
  let signedOrigin = brickOrigin + delta;
  if (any(signedOrigin < vec3i(0))) { return ownerPageInvalidAir(cell); }
  let origin = vec3u(signedOrigin);
  if (any(vec3u(size) > dimensions) || any(origin > dimensions - vec3u(size))) {
    return ownerPageInvalidAir(cell);
  }
  if (any(unsignedCell < origin) || any(unsignedCell >= origin + vec3u(size))) {
    return ownerPageInvalidAir(cell);
  }
  var result: OctreeOwnerPageLookupResult;
  result.origin = origin;
  result.size = size;
  result.status = select(0u, OWNER_PAGE_LOOKUP_TOPOLOGY,
    (packed & OWNER_PAGE_WORD_TOPOLOGY) != 0u);
  return result;
}
`;

export interface OctreeOwnerPageLifecycleStats {
  resident: number;
  peakResident: number;
  free: number;
  required: number;
  activated: number;
  retired: number;
  overflow: number;
  generation: number;
  capacity: number;
}

export interface OctreeOwnerPagePublicationResult {
  status: number;
  stats: OctreeOwnerPageLifecycleStats;
}

/** Deterministic CPU oracle for sorted, exact owner/page publications. */
export class OctreeOwnerPageLifecycleMirror {
  readonly pageTable: Uint32Array<ArrayBuffer>;
  readonly capacity: number;
  private resident = 0;
  private peakResident = 0;
  private required = 0;
  private activated = 0;
  private retired = 0;
  private overflow = 0;
  private generation = 0;
  private freePages: number[];

  constructor(readonly logicalBrickCount: number, capacity: number) {
    positiveInteger(logicalBrickCount, "Octree owner logical brick count");
    positiveInteger(capacity, "Octree owner physical page capacity");
    if (capacity > logicalBrickCount) throw new RangeError("Octree owner page capacity cannot exceed the logical brick count");
    this.capacity = capacity;
    this.pageTable = new Uint32Array(logicalBrickCount);
    this.freePages = Array.from({ length: capacity }, (_, page) => page + 1);
  }

  private checkedUnique(indices: readonly number[], label: string): number[] {
    const unique = [...new Set(indices)];
    for (const index of unique) {
      if (!Number.isSafeInteger(index) || index < 0 || index >= this.logicalBrickCount) {
        throw new RangeError(`${label} contains an invalid logical brick index`);
      }
    }
    return unique;
  }

  private apply(activeIndices: readonly number[], retiredIndices: readonly number[]): OctreeOwnerPageLifecycleStats {
    const active = this.checkedUnique(activeIndices, "Octree owner active list").sort((a, b) => a - b);
    const retired = this.checkedUnique(retiredIndices, "Octree owner retired list");
    const activeSet = new Set(active);
    if (retired.some((index) => activeSet.has(index))) throw new RangeError("A logical owner page cannot be active and retired in the same publication");
    const previous = new Set<number>();
    for (let logical = 0; logical < this.logicalBrickCount; logical += 1) {
      if (this.pageTable[logical] !== OCTREE_OWNER_PAGE_TABLE_MISSING) previous.add(logical);
    }
    this.required = active.length;
    this.activated = active.filter((logical) => !previous.has(logical)).length;
    this.retired = [...previous].filter((logical) => !activeSet.has(logical)).length;
    this.overflow = Math.max(0, active.length - this.capacity);
    if (this.overflow > 0) return this.stats();
    const retiredPages = [...previous]
      .filter((logical) => !activeSet.has(logical))
      .sort((a, b) => a - b)
      .map((logical) => this.pageTable[logical]);
    const available = [...retiredPages, ...this.freePages];
    const next = new Uint32Array(this.logicalBrickCount);
    let activation = 0;
    for (const logical of active) {
      const carried = this.pageTable[logical];
      if (carried !== OCTREE_OWNER_PAGE_TABLE_MISSING) {
        next[logical] = carried;
        continue;
      }
      const encodedPage = available[activation];
      if (encodedPage === undefined) {
        throw new Error("Octree owner stable page assignment exhausted validated capacity");
      }
      next[logical] = encodedPage;
      activation += 1;
    }
    this.freePages = activation < retiredPages.length
      ? [...this.freePages, ...retiredPages.slice(activation)]
      : this.freePages.slice(activation - retiredPages.length);
    this.pageTable.set(next);
    this.resident = active.length;
    this.peakResident = Math.max(this.peakResident, this.resident);
    return this.stats();
  }

  /** Apply an explicitly fenced publication; stale and unchanged inputs do no work. */
  publish(generation: number, activeIndices: readonly number[], retiredIndices: readonly number[]): OctreeOwnerPagePublicationResult {
    if (!Number.isSafeInteger(generation) || generation < 0 || generation > 0xffff_ffff) {
      throw new RangeError("Octree owner publication generation must fit uint32");
    }
    if (generation === 0) return { status: OCTREE_OWNER_PAGE_PUBLICATION_STATUS.unpublished, stats: this.stats() };
    if (generation < this.generation) return { status: OCTREE_OWNER_PAGE_PUBLICATION_STATUS.stale, stats: this.stats() };
    if (generation === this.generation) return { status: OCTREE_OWNER_PAGE_PUBLICATION_STATUS.unchanged, stats: this.stats() };
    const stats = this.apply(activeIndices, retiredIndices);
    if (stats.overflow === 0) this.generation = generation;
    const status = OCTREE_OWNER_PAGE_PUBLICATION_STATUS.ready
      | (stats.overflow > 0 ? OCTREE_OWNER_PAGE_PUBLICATION_STATUS.overflow : 0);
    return { status, stats };
  }

  update(activeIndices: readonly number[], retiredIndices: readonly number[]): OctreeOwnerPageLifecycleStats {
    return this.publish(this.generation + 1, activeIndices, retiredIndices).stats;
  }

  slot(logicalBrick: number): number | undefined {
    if (!Number.isSafeInteger(logicalBrick) || logicalBrick < 0 || logicalBrick >= this.logicalBrickCount) {
      throw new RangeError("Octree owner logical brick index is invalid");
    }
    const encoded = this.pageTable[logicalBrick];
    return encoded === OCTREE_OWNER_PAGE_TABLE_MISSING ? undefined : encoded - 1;
  }

  stats(): OctreeOwnerPageLifecycleStats {
    return {
      resident: this.resident,
      peakResident: this.peakResident,
      free: this.capacity - this.resident,
      required: this.required,
      activated: this.activated,
      retired: this.retired,
      overflow: this.overflow,
      generation: this.generation,
      capacity: this.capacity,
    };
  }
}

/**
 * Synchronization-free persistent owner publication.
 *
 * One bounded workgroup emits and sorts the exact active page stream, validates
 * the old/new merge, assigns stable physical IDs, and publishes payloads. Two
 * deliberately separate publication dispatches then replace the sorted record
 * table and finally advance its authority header. Retired pages satisfy
 * additions in delta rank order; only net growth consumes the persistent FIFO
 * of already-free IDs. No per-width dispatch schedule, indirect buffer,
 * logical-domain scan, capacity clear, hash, cross-workgroup synchronization, or
 * free-ID search remains.
 */
export const octreeDeterministicOwnerPageLifecycleShader = /* wgsl */ `
struct Params {
  counts: vec4u,
  offsets: vec4u,
  source: vec4u,
  topology: vec4u,
}
@group(0) @binding(0) var<storage, read_write> arena: array<u32>;
@group(0) @binding(1) var<storage, read> worklist: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> scratch: array<u32>;
@group(0) @binding(4) var<storage, read_write> candidateGenerationSource: array<u32>;

const HEADER: u32 = 16u;
const PAGE_VOXELS: u32 = 512u;
const INVALID_KEY: u32 = 0xffffffffu;
const CANDIDATE_VALID: u32 = 0x80000000u;
const OWNER_WORD_TOPOLOGY: u32 = ${OCTREE_OWNER_PAGE_WORD_TOPOLOGY}u;
const CONTROL_FREE_COUNT: u32 = ${OCTREE_OWNER_PAGE_CONTROL_WORDS.freeCount}u;
const CONTROL_RESIDENT_COUNT: u32 = ${OCTREE_OWNER_PAGE_CONTROL_WORDS.residentCount}u;
const CONTROL_CANDIDATE_ERROR: u32 = ${OCTREE_OWNER_PAGE_CONTROL_WORDS.candidateError}u;
const CONTROL_ACCEPTED_GENERATION: u32 = ${OCTREE_OWNER_PAGE_CONTROL_WORDS.acceptedGeneration}u;
const CONTROL_ACTIVATED_COUNT: u32 = ${OCTREE_OWNER_PAGE_CONTROL_WORDS.activatedCount}u;
const CONTROL_RETIRED_COUNT: u32 = ${OCTREE_OWNER_PAGE_CONTROL_WORDS.retiredCount}u;
const CONTROL_STATUS: u32 = ${OCTREE_OWNER_PAGE_CONTROL_WORDS.status}u;
const CONTROL_OBSERVED_GENERATION: u32 = ${OCTREE_OWNER_PAGE_CONTROL_WORDS.observedGeneration}u;
const CONTROL_INVALID_ENTRY_COUNT: u32 = ${OCTREE_OWNER_PAGE_CONTROL_WORDS.invalidEntryCount}u;
const META_OBSERVED_GENERATION: u32 = 22u;
const META_ERROR: u32 = 23u;
const META_GENERATION: u32 = 24u;
const META_OLD_COUNT: u32 = 25u;
const META_SOURCE_SLOTS: u32 = 26u;
const META_NEW_COUNT: u32 = 27u;
const META_VALID: u32 = 28u;
const META_ADDED: u32 = 29u;
const META_RETIRED: u32 = 30u;
const META_FREE_HEAD: u32 = 31u;
override SORT_CAPACITY: u32 = 1u;

fn sortABase() -> u32 { return 32u; }
fn candidateKeyBase() -> u32 { return sortABase() + SORT_CAPACITY; }
fn candidatePageBase() -> u32 { return candidateKeyBase() + params.counts.y; }
fn newFlagBase() -> u32 { return candidatePageBase() + params.counts.y; }
fn oldFlagBase() -> u32 { return newFlagBase() + params.counts.y; }
fn retiredPageBase() -> u32 { return oldFlagBase() + params.counts.y; }
fn activationRowBase() -> u32 { return retiredPageBase() + params.counts.y; }
fn freeQueueBase() -> u32 { return activationRowBase() + params.counts.y; }
fn activeTable() -> u32 { return arena[CONTROL_STATUS] >> 31u; }
fn inactiveTable() -> u32 { return 1u - activeTable(); }
fn recordKeyBase(table:u32)->u32{return params.offsets.x+table*2u*params.counts.y;}
fn recordPageBase(table:u32)->u32{return params.offsets.y+table*2u*params.counts.y;}
fn directoryBase(table:u32) -> u32 {
  return params.offsets.y + 3u*params.counts.y + table*params.counts.x;
}
fn payloadBase(table:u32)->u32{
  return params.offsets.z+table*params.counts.y*PAGE_VOXELS;
}
fn sortedKey(row:u32)->u32 { return scratch[sortABase()+row]; }
fn lowerBoundOld(key:u32,count:u32)->u32 {
  var lo=0u;var hi=count;
  while(lo<hi){let mid=lo+(hi-lo)/2u;
    if(arena[recordKeyBase(activeTable())+mid]<key){lo=mid+1u;}else{hi=mid;}}
  return lo;
}
fn lowerBoundNew(key:u32,count:u32)->u32 {
  var lo=0u;var hi=count;
  while(lo<hi){let mid=lo+(hi-lo)/2u;
    if(sortedKey(mid)<key){lo=mid+1u;}else{hi=mid;}}
  return lo;
}

fn candidatePageKey(slot:u32)->u32 {
  let tileSize=params.source.y;let pagesPerAxis=tileSize/8u;
  let pagesPerTile=pagesPerAxis*pagesPerAxis*pagesPerAxis;
  let item=slot/pagesPerTile;let page=slot%pagesPerTile;
  let dimensions=vec3u(params.counts.z,params.counts.w,params.offsets.w);
  let brickDimensions=(dimensions+vec3u(7u))/8u;
  let tiles=(dimensions+vec3u(tileSize-1u))/tileSize;
  let compactLimits=params.topology.yzw;
  let compactCount=compactLimits.x*compactLimits.y*compactLimits.z;
  let recurring=params.source.w!=0u;
  let worklistCount=select(0u,worklist[0],recurring);
  var tileIndex=INVALID_KEY;
  if(item<worklistCount){tileIndex=worklist[HEADER+item];}
  else if(item-worklistCount<compactCount){
    let compactItem=item-worklistCount;
    let tile=vec3u(compactItem%compactLimits.x,
      (compactItem/compactLimits.x)%compactLimits.y,
      compactItem/(compactLimits.x*compactLimits.y));
    tileIndex=tile.x+tiles.x*(tile.y+tiles.y*tile.z);
  }
  var key=0u;
  if(tileIndex<tiles.x*tiles.y*tiles.z){
    let tile=vec3u(tileIndex%tiles.x,(tileIndex/tiles.x)%tiles.y,
      tileIndex/(tiles.x*tiles.y));
    let local=vec3u(page%pagesPerAxis,(page/pagesPerAxis)%pagesPerAxis,
      page/(pagesPerAxis*pagesPerAxis));
    let brick=tile*pagesPerAxis+local;
    if(all(brick<brickDimensions)){
      let logical=brick.x+brickDimensions.x*(brick.y+brickDimensions.y*brick.z);
      key=select(logical+1u,0u,logical>=params.counts.x);
    }else{key=INVALID_KEY;}
  }
  return key;
}

fn ownerPageWord(cell: vec3u, origin: vec3u, size: u32) -> u32 {
  let brickOrigin = (cell / vec3u(8u)) * vec3u(8u);
  let delta = vec3i(origin) - vec3i(brickOrigin);
  return 0x80000000u
    | (u32(delta.x + 32) & 63u)
    | ((u32(delta.y + 32) & 63u) << 6u)
    | ((u32(delta.z + 32) & 63u) << 12u)
    | ((u32(firstTrailingBit(size)) & 7u) << 18u);
}

var<workgroup> transactionState:array<u32,8>;

@compute @workgroup_size(256)
fn buildOwnerPageCandidate(@builtin(local_invocation_index) lid:u32) {
  if(lid==0u){
    scratch[META_VALID]=0u;scratch[META_GENERATION]=0u;
    scratch[META_NEW_COUNT]=0u;scratch[META_ADDED]=0u;scratch[META_RETIRED]=0u;
    scratch[META_ERROR]=0u;
    // Analytic cold bootstrap owns worklist generation 1. Recurring candidates
    // consume the frontier transaction's host attempt stamp. Failed attempts
    // may create gaps, so validity is wrap-safe newer-than, not contiguous.
    var generation=worklist[15];
    if(params.source.w!=0u){
      // params.source.w is the accepted-generation index plus one. The
      // frontier attempt generation is five words after that accepted index.
      generation=candidateGenerationSource[params.source.w+4u];
    }
    let capacity=params.counts.y;
    let oldCount=arena[CONTROL_RESIDENT_COUNT];let tileSize=params.source.y;
    let compactLimits=params.topology.yzw;
    let compactCount=compactLimits.x*compactLimits.y*compactLimits.z;
    // Cold bootstrap consumes only the compact analytic support box. Every
    // recurring candidate unions that immutable air-side support floor with
    // the moving topology worklist. Sorting below removes overlapping pages.
    // This preserves Section 5 face/edge owner probes without enrolling the
    // support-only tiles in refinement or fine-fluid residency.
    let worklistCount=select(0u,worklist[0],params.source.w!=0u);
    let activeCount=worklistCount+compactCount;
    let pagesPerAxis=tileSize/8u;
    let pagesPerTile=pagesPerAxis*pagesPerAxis*pagesPerAxis;
    let sourceSlots=activeCount*pagesPerTile;
    let age=generation-arena[CONTROL_ACCEPTED_GENERATION];
    let eligible=generation!=0u&&age!=0u&&age<0x80000000u;
    let valid=oldCount<=capacity&&activeCount<=params.source.x
      &&tileSize>=8u&&tileSize<=32u&&(tileSize&(tileSize-1u))==0u
      &&pagesPerTile!=0u&&activeCount<=params.source.z/pagesPerTile
      &&sourceSlots<=params.source.z&&params.source.z<=SORT_CAPACITY;
    scratch[META_OBSERVED_GENERATION]=generation;
    if(eligible&&!valid){scratch[META_ERROR]=1u;}
    transactionState[0]=select(0u,1u,eligible&&valid);
    transactionState[1]=generation;transactionState[2]=oldCount;
    transactionState[3]=sourceSlots;transactionState[4]=0u;
    transactionState[5]=0u;transactionState[6]=0u;transactionState[7]=0u;
  }
  let enabled=workgroupUniformLoad(&transactionState[0]);
  let generation=workgroupUniformLoad(&transactionState[1]);
  let oldCount=workgroupUniformLoad(&transactionState[2]);
  let sourceSlots=workgroupUniformLoad(&transactionState[3]);
  if(enabled!=0u){
    for(var slot=lid;slot<SORT_CAPACITY;slot+=256u){
      scratch[sortABase()+slot]=select(INVALID_KEY,candidatePageKey(slot),slot<sourceSlots);
    }
    storageBarrier();workgroupBarrier();
    for(var width=2u;width<=SORT_CAPACITY;width<<=1u){
      for(var stride=width>>1u;stride>0u;stride>>=1u){
        for(var index=lid;index<SORT_CAPACITY;index+=256u){
          let partner=index^stride;
          if(partner>index){
            let left=sortedKey(index);let right=sortedKey(partner);
            let descending=(index&width)!=0u;let greater=left>right;
            if(greater!=descending){
              scratch[sortABase()+index]=right;scratch[sortABase()+partner]=left;
            }
          }
        }
        storageBarrier();workgroupBarrier();
      }
    }
    if(lid==0u){
      // Both sources are internally unique, but the moving worklist normally
      // overlaps the immutable analytic support floor. Compact the sorted
      // stream to a genuine set union before stable-ID carry classification.
      var count=0u;var previous=INVALID_KEY;
      for(var read=0u;read<sourceSlots;read+=1u){
        let key=sortedKey(read);
        if(key<INVALID_KEY&&key!=previous){scratch[sortABase()+count]=key;count+=1u;previous=key;}
      }
      for(var clear=count;clear<sourceSlots;clear+=1u){scratch[sortABase()+clear]=INVALID_KEY;}
      transactionState[4]=count;scratch[META_NEW_COUNT]=count;
      if(count>params.counts.y||(count>0u&&sortedKey(0u)==0u)){
        transactionState[0]=0u;scratch[META_ERROR]=1u;
      }
    }
    storageBarrier();workgroupBarrier();
    let admitted=workgroupUniformLoad(&transactionState[0]);
    let newCount=workgroupUniformLoad(&transactionState[4]);
    if(admitted!=0u){
      for(var row=lid;row<max(oldCount,newCount);row+=256u){
        if(row<newCount){
          let key=sortedKey(row);let old=lowerBoundOld(key,oldCount);
          let carried=old<oldCount&&arena[recordKeyBase(activeTable())+old]==key;
          scratch[candidateKeyBase()+row]=key;
          var oldPage=0u;if(carried){oldPage=arena[recordPageBase(activeTable())+old];}
          scratch[candidatePageBase()+row]=oldPage;
          let invalid=key==0u||key>params.counts.x||(row>0u&&sortedKey(row-1u)>=key);
          scratch[newFlagBase()+row]=select(1u,0u,carried)|select(0u,2u,invalid);
        }
        if(row<oldCount){
          let key=arena[recordKeyBase(activeTable())+row];
          let page=arena[recordPageBase(activeTable())+row];
          let current=lowerBoundNew(key,newCount);
          let carried=current<newCount&&sortedKey(current)==key;
          let invalid=key==0u||key>params.counts.x||page==0u||page>params.counts.y
            ||(row>0u&&arena[recordKeyBase(activeTable())+row-1u]>=key);
          scratch[oldFlagBase()+row]=select(1u,0u,carried)|select(0u,2u,invalid);
        }
      }
      storageBarrier();workgroupBarrier();
      if(lid==0u){
        var added=0u;var retired=0u;var invalid=0u;
        for(var row=0u;row<newCount;row+=1u){
          let flag=scratch[newFlagBase()+row];invalid|=flag>>1u;
          if((flag&1u)!=0u){scratch[newFlagBase()+row]=added+1u;added+=1u;}
          else{scratch[newFlagBase()+row]=0u;}
        }
        for(var row=0u;row<oldCount;row+=1u){
          let flag=scratch[oldFlagBase()+row];invalid|=flag>>1u;
          if((flag&1u)!=0u){
              scratch[retiredPageBase()+retired]=arena[recordPageBase(activeTable())+row];retired+=1u;
          }
        }
        let free=arena[CONTROL_FREE_COUNT];
        let carriedNew=select(newCount-added,0u,added>newCount);
        let carriedOld=select(oldCount-retired,0u,retired>oldCount);
        let available=free+retired;
        let valid=invalid==0u&&carriedNew==carriedOld&&free<=params.counts.y
          &&available>=added&&newCount==carriedNew+added&&oldCount==carriedOld+retired
          &&params.counts.y-newCount==available-added;
        if(valid){
          scratch[META_GENERATION]=generation;scratch[META_OLD_COUNT]=oldCount;
          scratch[META_SOURCE_SLOTS]=sourceSlots;scratch[META_ADDED]=added;
          scratch[META_RETIRED]=retired;
          transactionState[5]=added;transactionState[6]=retired;transactionState[7]=1u;
          }else{scratch[META_ERROR]=1u;}
      }
      let valid=workgroupUniformLoad(&transactionState[7]);
      let added=workgroupUniformLoad(&transactionState[5]);
      let retired=workgroupUniformLoad(&transactionState[6]);
      if(valid!=0u){
        for(var row=lid;row<newCount;row+=256u){
          let encodedRank=scratch[newFlagBase()+row];
          if(encodedRank!=0u){
            let rank=encodedRank-1u;var page=0u;
            if(rank<retired){page=scratch[retiredPageBase()+rank];}
            else{
              let queueRank=rank-retired;let head=scratch[META_FREE_HEAD];
              page=scratch[freeQueueBase()+(head+queueRank)%params.counts.y];
            }
            scratch[candidatePageBase()+row]=page;scratch[activationRowBase()+rank]=row;
          }
        }
        storageBarrier();workgroupBarrier();
        // Materialize a complete inactive payload bank. Carried physical IDs
        // copy their accepted page; added IDs receive canonical coarse owners.
        for(var row=0u;row<newCount;row+=1u){
          let logical=scratch[candidateKeyBase()+row]-1u;
          let encodedPage=scratch[candidatePageBase()+row];
          let carried=scratch[newFlagBase()+row]==0u;
          let dimensions=vec3u(params.counts.z,params.counts.w,params.offsets.w);
          let brickDimensions=(dimensions+vec3u(7u))/8u;
          let brick=vec3u(logical%brickDimensions.x,
            (logical/brickDimensions.x)%brickDimensions.y,
            logical/(brickDimensions.x*brickDimensions.y));
          let brickOrigin=brick*8u;
          for(var local=lid;local<PAGE_VOXELS;local+=256u){
            var word=arena[payloadBase(activeTable())+(encodedPage-1u)*PAGE_VOXELS+local];
            if(!carried){
              let cell=brickOrigin+vec3u(local%8u,(local/8u)%8u,local/64u);
              word=0u;
              if(all(cell<dimensions)){
                var size=params.source.y;var origin=(cell/vec3u(size))*vec3u(size);
                loop{if(all(origin+vec3u(size)<=dimensions)||size==1u){break;}
                  size>>=1u;origin=(cell/vec3u(size))*vec3u(size);}
                word=ownerPageWord(cell,origin,size);
              }
            }
            // Membership is a leaf property, not a resident-page property.
            // The compact frontier marks accepted leaf origins after it has
            // finished refining this inactive owner bank.
            word&=~OWNER_WORD_TOPOLOGY;
            arena[payloadBase(inactiveTable())+(encodedPage-1u)*PAGE_VOXELS+local]=word;
          }
        }
        storageBarrier();workgroupBarrier();
        if(lid==0u){scratch[META_VALID]=CANDIDATE_VALID;}
      }
    }
  }
}

var<workgroup> commitState:array<u32,4>;
@compute @workgroup_size(256)
fn commitOwnerPageCandidate(@builtin(local_invocation_index) lid:u32) {
  if(lid==0u){
    let valid=scratch[META_VALID]==CANDIDATE_VALID;
    commitState[0]=select(0u,1u,valid);
    commitState[1]=select(0u,scratch[META_NEW_COUNT],valid);
    let added=select(0u,scratch[META_ADDED],valid);
    let retired=select(0u,scratch[META_RETIRED],valid);
    commitState[2]=added;commitState[3]=select(retired-added,0u,retired<added);
  }
  let enabled=workgroupUniformLoad(&commitState[0]);
  let count=workgroupUniformLoad(&commitState[1]);
  let added=workgroupUniformLoad(&commitState[2]);
  if(enabled!=0u){
    let oldCount=scratch[META_OLD_COUNT];
    // Clear only keys previously stored in the inactive bank; this is bounded
    // by physical capacity and never scans the logical domain.
    for(var row=lid;row<params.counts.y;row+=256u){
      let staleKey=arena[recordKeyBase(inactiveTable())+row];
      if(staleKey>0u&&staleKey<=params.counts.x){
        arena[directoryBase(inactiveTable())+staleKey-1u]=0u;
      }
      if(row<count){
        let key=scratch[candidateKeyBase()+row];
        let page=scratch[candidatePageBase()+row];
        arena[recordKeyBase(inactiveTable())+row]=key;
        arena[recordPageBase(inactiveTable())+row]=page;
        if(key>0u&&key<=params.counts.x){
          arena[directoryBase(inactiveTable())+key-1u]=page;
        }
      }else{
        arena[recordKeyBase(inactiveTable())+row]=0u;
        arena[recordPageBase(inactiveTable())+row]=0u;
      }
    }
  }
}

@compute @workgroup_size(1)
fn commitOwnerPageGeneration() {
  if(scratch[META_VALID]!=CANDIDATE_VALID){return;}
  // The owner bank and frontier selector are one authority. Validate every
  // frontier publication condition before writing either active header; there
  // is no second dispatch and therefore no split-epoch failure window.
  if(params.source.w==0u){return;}
  let activeGenerationIndex=params.source.w-1u;
  let frontierBase=activeGenerationIndex-3u;
  let selector=candidateGenerationSource[frontierBase+7u];
  let generation=candidateGenerationSource[frontierBase+8u];
  let candidateCount=select(params.topology.x+1u,
    candidateGenerationSource[frontierBase+selector],selector<=1u);
  if(candidateGenerationSource[frontierBase+6u]!=1u
    ||candidateGenerationSource[frontierBase+9u]!=0u
    ||selector>1u||candidateCount>params.topology.x
    ||generation!=scratch[META_GENERATION]||generation==0u){return;}
  let age=generation-candidateGenerationSource[activeGenerationIndex];
  if(age==0u||age>=0x80000000u){return;}
  let count=scratch[META_NEW_COUNT];let added=scratch[META_ADDED];let retired=scratch[META_RETIRED];
  let consumed=select(added-retired,0u,added<retired);
  let capacity=params.counts.y;
  let surplus=select(retired-added,0u,retired<added);
  let free=arena[CONTROL_FREE_COUNT];
  let tail=(scratch[META_FREE_HEAD]+free)%capacity;
  for(var rank=0u;rank<surplus;rank+=1u){
    scratch[freeQueueBase()+(tail+rank)%capacity]
      =scratch[retiredPageBase()+added+rank];
  }
  scratch[META_FREE_HEAD]=(scratch[META_FREE_HEAD]+consumed)%capacity;
  arena[CONTROL_FREE_COUNT] = params.counts.y - count;
  arena[CONTROL_RESIDENT_COUNT] = count;
  arena[CONTROL_ACCEPTED_GENERATION] = scratch[META_GENERATION];
  arena[CONTROL_CANDIDATE_ERROR] = 0u;
  arena[CONTROL_ACTIVATED_COUNT] = added;
  arena[CONTROL_RETIRED_COUNT] = retired;
  arena[CONTROL_STATUS] = ${OCTREE_OWNER_PAGE_PUBLICATION_STATUS.ready}u
    | (inactiveTable()<<31u);
  arena[CONTROL_OBSERVED_GENERATION] = arena[CONTROL_ACCEPTED_GENERATION];
  arena[CONTROL_INVALID_ENTRY_COUNT] = 0u;
  candidateGenerationSource[frontierBase+2u]=selector;
  candidateGenerationSource[activeGenerationIndex]=generation;
  candidateGenerationSource[frontierBase+6u]=0u;
  scratch[META_VALID] = 0u;
}
`;

export interface OctreeAnalyticOwnerBootstrapSource {
  readonly tileWorklist: GPUBuffer;
  readonly tileSizeCells: number;
  readonly activeTileLimits: readonly [number, number, number];
  readonly activeTileCount: number;
}

/**
 * The recurring spatial owner domain. Unlike the surface-brick worklist, the
 * topology-tile publication already contains the 3x3x3 support closure used
 * by refinement, balancing, frontier replacement, and the face/edge 1-ring
 * required by Aanjaneya et al. (2017), Section 4.2. Section 5's independently
 * allocated fine-phi block one-ring is not allowed to shrink this pressure
 * support set.
 */
export interface OctreeOwnerTopologyResidencySource {
  readonly tileWorklist: GPUBuffer;
  readonly tileSizeCells: number;
  readonly tileListCapacity: number;
  /** GPU-resident active topology clock. The owner candidate is exactly its
   * immediate successor, even when the carried residency list is unchanged. */
  readonly candidateGeneration: {
    readonly buffer: GPUBuffer;
    readonly offsetWords: number;
    readonly frontierListCapacity: number;
  };
}

/** CPU oracle for the production GPU attempt clock. Rejected attempts remain
 * part of the clock, so callers pass the last attempted generation rather
 * than the last accepted generation. Zero is reserved for unpublished state. */
export function resolveOctreeOwnerCandidateGeneration(
  lastAttemptGeneration: number,
): number {
  if (!Number.isSafeInteger(lastAttemptGeneration)
      || lastAttemptGeneration < 0
      || lastAttemptGeneration > 0xffff_ffff) {
    throw new RangeError("Last topology attempt generation must fit uint32");
  }
  return ((lastAttemptGeneration + 1) >>> 0) || 1;
}

export function octreeAnalyticOwnerBootstrapPageCount(
  dimensions: OctreeOwnerCoordinate,
  source: Pick<OctreeAnalyticOwnerBootstrapSource, "tileSizeCells" | "activeTileLimits">,
): number {
  if (!Number.isSafeInteger(source.tileSizeCells) || source.tileSizeCells < 8
      || source.tileSizeCells > 32 || (source.tileSizeCells & (source.tileSizeCells - 1)) !== 0) {
    throw new RangeError("Analytic owner bootstrap tile size must be 8, 16, or 32");
  }
  const coveredCells = source.activeTileLimits.map((limit, axis) => {
    if (!Number.isSafeInteger(limit) || limit < 0) throw new RangeError("Analytic owner bootstrap limits must be non-negative integers");
    return Math.min(dimensions[axis], limit * source.tileSizeCells);
  });
  return coveredCells.reduce((product, cells) => product * Math.ceil(cells / OCTREE_OWNER_BRICK_SIZE), 1);
}

/** GPU-only owner-page lifecycle driven directly by fluid brick residency. */
export class WebGPUOctreeSimulationOwnerPages {
  readonly plan: OctreeOwnerPagePlan;
  readonly arena: GPUBuffer;
  /** Candidate metadata/scratch. Word 28 is the fail-closed ready gate. */
  readonly candidateTransaction: GPUBuffer;
  readonly allocatedBytes: number;
  private readonly params: GPUBuffer;
  private readonly analyticParams?: GPUBuffer;
  private readonly scratch: GPUBuffer;
  private readonly buildCandidate: GPUComputePipeline;
  private readonly commitCandidate: GPUComputePipeline;
  private readonly commit: GPUComputePipeline;
  private readonly analyticBootstrap?: OctreeAnalyticOwnerBootstrapSource;
  private readonly analyticGroup?: GPUBindGroup;
  private readonly topologyResidency?: OctreeOwnerTopologyResidencySource;
  private readonly topologyGroup?: GPUBindGroup;

  constructor(device: GPUDevice, dimensions: OctreeOwnerCoordinate,
    options: OctreeOwnerPagePlanOptions = {}, analyticBootstrap?: OctreeAnalyticOwnerBootstrapSource,
    topologyResidency?: OctreeOwnerTopologyResidencySource) {
    this.analyticBootstrap = analyticBootstrap;
    this.topologyResidency = topologyResidency;
    if (analyticBootstrap && !topologyResidency) {
      throw new Error("Analytic owner bootstrap requires the coupled topology epoch source");
    }
    if (analyticBootstrap && (analyticBootstrap.activeTileCount
      !== analyticBootstrap.activeTileLimits[0] * analyticBootstrap.activeTileLimits[1]
        * analyticBootstrap.activeTileLimits[2])) {
      throw new RangeError("Analytic owner bootstrap count must match its compact limits");
    }
    if (topologyResidency && (!Number.isSafeInteger(topologyResidency.tileSizeCells)
        || topologyResidency.tileSizeCells < 8 || topologyResidency.tileSizeCells > 32
        || (topologyResidency.tileSizeCells & (topologyResidency.tileSizeCells - 1)) !== 0
        || !Number.isSafeInteger(topologyResidency.tileListCapacity)
        || topologyResidency.tileListCapacity < 1
        || !Number.isSafeInteger(topologyResidency.candidateGeneration.offsetWords)
        || topologyResidency.candidateGeneration.offsetWords < 0
        || !Number.isSafeInteger(topologyResidency.candidateGeneration.frontierListCapacity)
        || topologyResidency.candidateGeneration.frontierListCapacity < 1
        || topologyResidency.candidateGeneration.offsetWords * 4 + 4
          > topologyResidency.candidateGeneration.buffer.size)) {
      throw new RangeError("Owner topology residency requires a power-of-two 8..32 cell tile and positive list capacity");
    }
    const analyticMinimumPages = analyticBootstrap
      ? octreeAnalyticOwnerBootstrapPageCount(dimensions, analyticBootstrap)
      : 1;
    const topologyMaximumCandidatePages = topologyResidency
      ? topologyResidency.tileListCapacity
        * (topologyResidency.tileSizeCells / OCTREE_OWNER_BRICK_SIZE) ** 3
      : 0;
    // The recurring candidate is the set union of its moving worklist and the
    // immutable analytic support floor. Reserve for their worst-case unique
    // page count (the planner clips this to the finite logical domain). A
    // smaller adaptive arena can validate the cold set and then deadlock on
    // the first legitimate support-only page added by recurrence.
    const recurringUnionMinimumPages = analyticMinimumPages + topologyMaximumCandidatePages;
    this.plan = planOctreeOwnerPages(dimensions, {
      ...options,
      minimumPages: Math.max(options.minimumPages ?? 1, recurringUnionMinimumPages),
    });
    this.arena = device.createBuffer({ label: "Simulation octree owner pages", size: this.plan.allocatedBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.params = device.createBuffer({ label: "Simulation octree sorted owner-page parameters", size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.analyticParams = analyticBootstrap
      ? device.createBuffer({ label: "Analytic octree sorted owner-page parameters", size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }) : undefined;
    const topologyCandidateSlots = topologyMaximumCandidatePages;
    const analyticCandidateSlots = analyticBootstrap
      ? analyticBootstrap.activeTileCount
        * (analyticBootstrap.tileSizeCells / OCTREE_OWNER_BRICK_SIZE) ** 3
      : 0;
    // Recurring publications retain the immutable analytic air-support floor
    // and add the moving topology worklist. Their candidate stream is a union;
    // the GPU sort deterministically removes pages present in both sources.
    const candidateSlotCapacity = Math.max(1, topologyCandidateSlots + analyticCandidateSlots);
    if (!Number.isSafeInteger(candidateSlotCapacity) || candidateSlotCapacity < 1) {
      throw new RangeError("Octree owner candidate-page capacity is invalid");
    }
    const sortCapacity = 2 ** Math.ceil(Math.log2(candidateSlotCapacity));
    const freeQueueBase = 32 + sortCapacity + this.plan.capacity * 6;
    const scratchWords = freeQueueBase + this.plan.capacity;
    this.scratch = device.createBuffer({ label: "Deterministic octree owner-page transaction",
      size: scratchWords * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.candidateTransaction = this.scratch;
    device.queue.writeBuffer(this.scratch, freeQueueBase * 4,
      Uint32Array.from({ length: this.plan.capacity }, (_, page) => page + 1));
    const initial = new Uint32Array(this.plan.allocatedWords);
    initial[OCTREE_OWNER_PAGE_CONTROL_WORDS.freeCount] = this.plan.capacity;
    initial[OCTREE_OWNER_PAGE_CONTROL_WORDS.capacity] = this.plan.capacity;
    initial[OCTREE_OWNER_PAGE_CONTROL_WORDS.logicalBrickCount] = this.plan.logicalBrickCount;
    initial[OCTREE_OWNER_PAGE_CONTROL_WORDS.ownerRecordPageOffsetWords] =
      this.plan.ownerRecordPageOffsetWords;
    initial[OCTREE_OWNER_PAGE_CONTROL_WORDS.ownerPagesOffsetWords] = this.plan.ownerPagesOffsetWords;
    initial[OCTREE_OWNER_PAGE_CONTROL_WORDS.tileListCapacity] =
      topologyResidency?.tileListCapacity ?? 0;
    initial[OCTREE_OWNER_PAGE_CONTROL_WORDS.tileSizeCells] =
      topologyResidency?.tileSizeCells ?? analyticBootstrap?.tileSizeCells ?? 0;
    initial[OCTREE_OWNER_PAGE_CONTROL_WORDS.magic] = OCTREE_OWNER_ARENA_MAGIC;
    device.queue.writeBuffer(this.arena, 0, initial);
    const parameterWords = (
      tileListCapacity: number,
      tileSize: number,
      candidateGenerationOffsetPlusOne = 0,
      analyticLimits: readonly [number, number, number] = [0, 0, 0],
    ) => new Uint32Array([
      this.plan.logicalBrickCount, this.plan.capacity, dimensions[0], dimensions[1],
      this.plan.ownerRecordKeyOffsetWords, this.plan.ownerRecordPageOffsetWords,
      this.plan.ownerPagesOffsetWords, dimensions[2],
      tileListCapacity, tileSize, candidateSlotCapacity, candidateGenerationOffsetPlusOne,
      topologyResidency?.candidateGeneration.frontierListCapacity ?? 0, ...analyticLimits,
    ]);
    device.queue.writeBuffer(this.params, 0, parameterWords(
      (topologyResidency?.tileListCapacity ?? 0) + (analyticBootstrap?.activeTileCount ?? 0) || 1,
      topologyResidency?.tileSizeCells ?? analyticBootstrap?.tileSizeCells ?? 8,
      topologyResidency ? topologyResidency.candidateGeneration.offsetWords + 1 : 0,
      analyticBootstrap?.activeTileLimits,
    ));
    if (this.analyticParams && analyticBootstrap) {
      device.queue.writeBuffer(this.analyticParams, 0,
        parameterWords(Math.max(1, analyticBootstrap.activeTileCount), analyticBootstrap.tileSizeCells,
          0, analyticBootstrap.activeTileLimits));
    }
    const layout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ] });
    const lifecycleModule = device.createShaderModule({ label: "Deterministic simulation octree owner pages",
      code: octreeDeterministicOwnerPageLifecycleShader });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const pipeline = (entryPoint: string, constants?: Record<string, number>) =>
      device.createComputePipeline({
        layout: pipelineLayout, compute: { module: lifecycleModule, entryPoint, constants },
      });
    this.buildCandidate = pipeline("buildOwnerPageCandidate", { SORT_CAPACITY: sortCapacity });
    this.commitCandidate = pipeline("commitOwnerPageCandidate",
      { SORT_CAPACITY: sortCapacity });
    this.commit = pipeline("commitOwnerPageGeneration");
    const lifecycleEntries = (
      worklist: GPUBuffer,
      params: GPUBuffer,
      generationSource: GPUBuffer,
    ) => [
      { binding: 0, resource: { buffer: this.arena } },
      { binding: 1, resource: { buffer: worklist } },
      { binding: 2, resource: { buffer: params } },
      { binding: 3, resource: { buffer: this.scratch } },
      { binding: 4, resource: { buffer: generationSource } },
    ];
    if (analyticBootstrap && this.analyticParams) {
      this.analyticGroup = device.createBindGroup({ layout,
        entries: lifecycleEntries(
          analyticBootstrap.tileWorklist,
          this.analyticParams,
          topologyResidency!.candidateGeneration.buffer,
        ) });
    }
    if (topologyResidency) {
      this.topologyGroup = device.createBindGroup({ layout,
        entries: lifecycleEntries(
          topologyResidency.tileWorklist,
          this.params,
          topologyResidency.candidateGeneration.buffer,
        ) });
    }
    this.allocatedBytes = this.arena.size + this.params.size + (this.analyticParams?.size ?? 0)
      + this.scratch.size;
  }

  /** Tail of substep N: prepare the next page records/payload, but do not
   * advance the accepted owner generation visible to active consumers. */
  encodeInactiveCandidate(broker: PassBroker): void {
    if (!this.topologyResidency || !this.topologyGroup) {
      throw new Error("Recurring owner pages require exact topology-tile residency");
    }
    try {
      const pass = broker.compute({ label: "Prepare inactive owner-page generation" });
      pass.setBindGroup(0, this.topologyGroup);
      pass.setPipeline(this.buildCandidate); pass.dispatchWorkgroups(1);
      pass.setPipeline(this.commitCandidate); pass.dispatchWorkgroups(1);
      broker.fence("inactive owner-page generation prepared");
    } catch (error) {
      broker.fence("owner-page lifecycle encoding failure");
      throw error;
    }
  }

  /** Beginning of substep N+1: publish only a completely prepared candidate. */
  encodeReadyCommit(broker: PassBroker): void {
    if (!this.topologyResidency || !this.topologyGroup) {
      throw new Error("Recurring owner pages require exact topology-tile residency");
    }
    const pass = broker.compute({ label: "Publish ready owner-page generation" });
    pass.setBindGroup(0, this.topologyGroup);
    pass.setPipeline(this.commit);
    pass.dispatchWorkgroups(1);
    broker.fence("ready owner-page generation published");
  }

  /** One-time GPU-only analytic candidate preparation. The ordinary coupled
   * boundary commit publishes it only after the cold frontier validates. */
  encodeAnalyticBootstrap(broker: PassBroker): void {
    if (!this.analyticBootstrap || !this.analyticGroup) {
      throw new Error("Analytic owner-page bootstrap was not configured");
    }
    const analyticGroup = this.analyticGroup;
    try {
      const pass = broker.compute({ label: "Analytic owner-page generation transaction" });
      pass.setBindGroup(0, analyticGroup);
      pass.setPipeline(this.buildCandidate); pass.dispatchWorkgroups(1);
      pass.setPipeline(this.commitCandidate); pass.dispatchWorkgroups(1);
      broker.fence("analytic owner-page candidate prepared");
    } catch (error) {
      broker.fence("analytic owner-page encoding failure");
      throw error;
    }
  }

  destroy(): void {
    this.arena.destroy(); this.params.destroy(); this.analyticParams?.destroy();
    this.scratch.destroy();
  }
}
