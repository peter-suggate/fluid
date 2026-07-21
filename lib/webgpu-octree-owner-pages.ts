/**
 * Bounded brick-page substrate for the octree pressure owner map.
 *
 * This module deliberately does not integrate with the live projection yet.
 * It supplies the compact owner encoding, capacity accounting, and allocation
 * lifecycle needed for a mirror-first cutover from the dense Owner array.
 */

import {
  SVO_RENDER_RESIDENCY_CONSUMER_CONTROL_WORDS,
  SVO_RENDER_RESIDENCY_CONSUMER_STATUS,
  type WebGPUSvoRenderResidencyConsumer,
} from "./webgpu-svo-render-residency-consumer";
import {
  FLUID_BRICK_ACTIVE_DISPATCH_OFFSET_BYTES,
  FLUID_BRICK_RETIRED_DISPATCH_OFFSET_BYTES,
} from "./webgpu-fluid-brick-residency";

export const OCTREE_OWNER_BRICK_SIZE = 8 as const;
export const OCTREE_OWNER_PAGE_VOXELS = OCTREE_OWNER_BRICK_SIZE ** 3;
export const OCTREE_OWNER_ARENA_CONTROL_WORDS = 16;
export const OCTREE_OWNER_PAGE_WORD_VALID = 0x8000_0000;
export const OCTREE_OWNER_PAGE_TABLE_MISSING = 0;
export const OCTREE_OWNER_PAGE_TABLE_RESERVED = 0xffff_ffff;
export const OCTREE_OWNER_PAGE_HASH_EMPTY = 0;
export const OCTREE_OWNER_PAGE_HASH_TOMBSTONE = 0xffff_ffff;

/** Renderer-owned lifecycle status bits stored in arena control word 10. */
export const SVO_OWNER_PAGE_STATUS = Object.freeze({
  ready: 1 << 0,
  unchanged: 1 << 1,
  stale: 1 << 2,
  unpublished: 1 << 3,
  sourceRejected: 1 << 4,
  invalidEntry: 1 << 5,
  overflow: 1 << 6,
  sourceDegraded: 1 << 7,
} as const);

export const SVO_OWNER_PAGE_CONTROL_WORDS = Object.freeze({
  freeCount: 0,
  residentCount: 1,
  peakResidentCount: 2,
  overflowCount: 3,
  requiredCount: 4,
  activatedCount: 5,
  retiredCount: 6,
  acceptedGeneration: 7,
  ownerMismatchCount: 8,
  comparedOwnerCount: 9,
  status: 10,
  observedGeneration: 11,
  invalidEntryCount: 12,
  stalePublicationCount: 13,
  unchangedPublicationCount: 14,
  unpublishedPublicationCount: 15,
} as const);

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
    surfacePageCapacity: number;
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
  pageTableOffsetWords: number;
  /** Number of open-addressed key/value slots. Bounded by physical capacity, not box volume. */
  pageHashCapacity: number;
  pageTableValueOffsetWords: number;
  freeListOffsetWords: number;
  ownerPagesOffsetWords: number;
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
    const { pressureRowCapacity, surfacePageCapacity } = options.adaptiveBounds;
    positiveInteger(pressureRowCapacity, "Octree owner pressure-row capacity");
    positiveInteger(surfacePageCapacity, "Octree owner surface-page capacity");
    // Bulk rows amortize over an 8^3 owner page. Fine interface rows amortize
    // only over its 8^2 cross-section. The 3/2 multiplier covers overlap,
    // 2:1 grading, and one-frame residency hysteresis without returning to a
    // fixed percentage of the entire bounding volume.
    const pressurePages = Math.ceil(pressureRowCapacity / OCTREE_OWNER_PAGE_VOXELS);
    const surfacePages = Math.ceil(surfacePageCapacity / (OCTREE_OWNER_BRICK_SIZE ** 2));
    adaptiveCapacity = Math.min(logicalBrickCount, Math.max(1, Math.ceil((pressurePages + surfacePages) * 3 / 2)));
  }
  const requestedCapacity = Math.min(logicalBrickCount, Math.max(
    minimumCapacity,
    Math.min(fractionalCapacity, hardCapacity, adaptiveCapacity ?? logicalBrickCount),
  ));
  const hashSlotsFor = (residentCapacity: number) => Math.max(2, Math.ceil(residentCapacity * 4 / 3));
  const wordsFor = (residentCapacity: number) =>
    OCTREE_OWNER_ARENA_CONTROL_WORDS + hashSlotsFor(residentCapacity) * 2
      + residentCapacity * (1 + OCTREE_OWNER_PAGE_VOXELS);
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
  const pageHashCapacity = hashSlotsFor(capacity);
  const pageTableOffsetWords = OCTREE_OWNER_ARENA_CONTROL_WORDS;
  const pageTableValueOffsetWords = pageTableOffsetWords + pageHashCapacity;
  const freeListOffsetWords = pageTableValueOffsetWords + pageHashCapacity;
  const ownerPagesOffsetWords = freeListOffsetWords + capacity;
  const allocatedWords = ownerPagesOffsetWords + capacity * OCTREE_OWNER_PAGE_VOXELS;
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
    bytesPerPage: (1 + OCTREE_OWNER_PAGE_VOXELS) * 4,
    controlOffsetWords: 0,
    pageTableOffsetWords,
    pageHashCapacity,
    pageTableValueOffsetWords,
    freeListOffsetWords,
    ownerPagesOffsetWords,
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
} as const);

function ownerPageHash(logical: number): number {
  return Math.imul(logical >>> 0, 0x9e37_79b1) >>> 0;
}

/** Locate a logical brick in the sparse owner hash. Returns -1 when absent. */
export function findOctreeOwnerPageHashSlot(
  arena: ArrayLike<number>,
  plan: Pick<OctreeOwnerPagePlan, "pageTableOffsetWords" | "pageHashCapacity">,
  logical: number,
): number {
  if (!Number.isSafeInteger(logical) || logical < 0 || logical >= 0xffff_fffe) return -1;
  const key = logical + 1;
  const capacity = plan.pageHashCapacity;
  if (capacity < 1) return -1;
  let slot = ownerPageHash(logical) % capacity;
  for (let probe = 0; probe < capacity; probe += 1) {
    const word = plan.pageTableOffsetWords + slot;
    if (word >= arena.length) return -1;
    const observed = Number(arena[word]) >>> 0;
    if (observed === key) return slot;
    if (observed === OCTREE_OWNER_PAGE_HASH_EMPTY) return -1;
    slot = slot + 1 === capacity ? 0 : slot + 1;
  }
  return -1;
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
  const hashSlot = findOctreeOwnerPageHashSlot(arena, plan, logical);
  if (hashSlot < 0) {
    return missingOwnerLookup(cellValue, plan.dimensions, maximumLeafSize, false);
  }
  const valueWord = plan.pageTableValueOffsetWords + hashSlot;
  if (valueWord >= arena.length) return missingOwnerLookup(cellValue, plan.dimensions, maximumLeafSize, true);
  const encodedPage = Number(arena[valueWord]) >>> 0;
  if (encodedPage === OCTREE_OWNER_PAGE_TABLE_MISSING) return missingOwnerLookup(cellValue, plan.dimensions, maximumLeafSize, true);
  if (encodedPage === OCTREE_OWNER_PAGE_TABLE_RESERVED || encodedPage > plan.capacity) {
    return missingOwnerLookup(cellValue, plan.dimensions, maximumLeafSize, true);
  }
  const local = (cell[0] & 7) + (cell[1] & 7) * 8 + (cell[2] & 7) * 64;
  const payloadWord = plan.ownerPagesOffsetWords + (encodedPage - 1) * OCTREE_OWNER_PAGE_VOXELS + local;
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
    return { ...owner, status: 0 };
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
  arenaOffsetsCapacity: vec4u,           // page table, payload, capacity, page voxels
}
struct OctreeOwnerPageLookupResult {
  origin: vec3u,
  size: u32,
  status: u32,
}

const OWNER_PAGE_LOOKUP_MISSING: u32 = ${OCTREE_OWNER_PAGE_LOOKUP_STATUS.missing}u;
const OWNER_PAGE_LOOKUP_INVALID: u32 = ${OCTREE_OWNER_PAGE_LOOKUP_STATUS.invalid}u;
const OWNER_PAGE_WORD_VALID: u32 = ${OCTREE_OWNER_PAGE_WORD_VALID}u;
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
}

fn ownerPageInvalidAir(cell: vec3i) -> OctreeOwnerPageLookupResult {
  return ownerPageCanonicalAir(cell, OWNER_PAGE_LOOKUP_INVALID);
}

fn octreeOwnerPageLookup(cell: vec3i) -> OctreeOwnerPageLookupResult {
  let dimensions = ownerPageLookupParams.dimensionsMaximumLeaf.xyz;
  let maximumLeaf = ownerPageLookupParams.dimensionsMaximumLeaf.w;
  let brickDimensions = ownerPageLookupParams.brickDimensionsLogicalCount.xyz;
  let logicalCount = ownerPageLookupParams.brickDimensionsLogicalCount.w;
  let pageTableOffset = ownerPageLookupParams.arenaOffsetsCapacity.x;
  let payloadOffset = ownerPageLookupParams.arenaOffsetsCapacity.y;
  let capacity = ownerPageLookupParams.arenaOffsetsCapacity.z;
  let pageVoxels = ownerPageLookupParams.arenaOffsetsCapacity.w;
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
  if (payloadOffset <= pageTableOffset + capacity || ((payloadOffset - pageTableOffset - capacity) & 1u) != 0u) {
    return ownerPageInvalidAir(cell);
  }
  let hashCapacity = (payloadOffset - pageTableOffset - capacity) / 2u;
  if (hashCapacity == 0u || pageTableOffset >= arenaWords || hashCapacity > arenaWords - pageTableOffset) {
    return ownerPageInvalidAir(cell);
  }
  let key = logical + 1u;
  var slot = (logical * 0x9e3779b1u) % hashCapacity;
  var encodedPage = 0u;
  var found = false;
  for (var probe = 0u; probe < hashCapacity; probe += 1u) {
    let observed = ownerPageArena[pageTableOffset + slot];
    if (observed == key) {
      if (pageTableOffset + hashCapacity + slot >= arenaWords) { return ownerPageInvalidAir(cell); }
      encodedPage = ownerPageArena[pageTableOffset + hashCapacity + slot];
      found = true;
      break;
    }
    if (observed == 0u) { break; }
    slot = select(slot + 1u, 0u, slot + 1u == hashCapacity);
  }
  if (!found) { return ownerPageCanonicalAir(cell, 0u); }
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
  result.status = 0u;
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
  /** Dense-owner records compared against their packed page round-trip. */
  comparedOwners: number;
  /** Packed records whose decoded origin/size differs from dense authority. */
  ownerMismatches: number;
}

export interface OctreeOwnerPagePublicationResult {
  status: number;
  stats: OctreeOwnerPageLifecycleStats;
}

/** Deterministic CPU oracle for the GPU page-table/free-list lifecycle. */
export class OctreeOwnerPageLifecycleMirror {
  readonly pageTable: Uint32Array<ArrayBuffer>;
  readonly capacity: number;
  private readonly freeSlots: number[];
  private resident = 0;
  private peakResident = 0;
  private required = 0;
  private activated = 0;
  private retired = 0;
  private overflow = 0;
  private generation = 0;

  constructor(readonly logicalBrickCount: number, capacity: number) {
    positiveInteger(logicalBrickCount, "Octree owner logical brick count");
    positiveInteger(capacity, "Octree owner physical page capacity");
    if (capacity > logicalBrickCount) throw new RangeError("Octree owner page capacity cannot exceed the logical brick count");
    this.capacity = capacity;
    this.pageTable = new Uint32Array(logicalBrickCount);
    this.freeSlots = Array.from({ length: capacity }, (_, index) => capacity - 1 - index);
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
    const active = this.checkedUnique(activeIndices, "Octree owner active list");
    const retired = this.checkedUnique(retiredIndices, "Octree owner retired list");
    const activeSet = new Set(active);
    if (retired.some((index) => activeSet.has(index))) throw new RangeError("A logical owner page cannot be active and retired in the same publication");
    this.required = 0;
    this.activated = 0;
    this.retired = 0;
    this.overflow = 0;
    // Activation intentionally precedes retirement. A retiring page cannot be
    // reused until consumers have invalidated the old topology/frontier.
    for (const logical of active) {
      if (this.pageTable[logical] !== OCTREE_OWNER_PAGE_TABLE_MISSING) continue;
      this.required += 1;
      const slot = this.freeSlots.pop();
      if (slot === undefined) { this.overflow += 1; continue; }
      this.pageTable[logical] = slot + 1;
      this.resident += 1;
      this.activated += 1;
      this.peakResident = Math.max(this.peakResident, this.resident);
    }
    for (const logical of retired) {
      const encoded = this.pageTable[logical];
      if (encoded === OCTREE_OWNER_PAGE_TABLE_MISSING) continue;
      this.pageTable[logical] = OCTREE_OWNER_PAGE_TABLE_MISSING;
      this.freeSlots.push(encoded - 1);
      this.resident -= 1;
      this.retired += 1;
    }
    return this.stats();
  }

  /** Apply an explicitly fenced publication; stale and unchanged inputs do no work. */
  publish(generation: number, activeIndices: readonly number[], retiredIndices: readonly number[]): OctreeOwnerPagePublicationResult {
    if (!Number.isSafeInteger(generation) || generation < 0 || generation > 0xffff_ffff) {
      throw new RangeError("Octree owner publication generation must fit uint32");
    }
    if (generation === 0) return { status: SVO_OWNER_PAGE_STATUS.unpublished, stats: this.stats() };
    if (generation < this.generation) return { status: SVO_OWNER_PAGE_STATUS.stale, stats: this.stats() };
    if (generation === this.generation) return { status: SVO_OWNER_PAGE_STATUS.unchanged, stats: this.stats() };
    const previousGeneration = this.generation;
    this.generation = generation;
    let stats: OctreeOwnerPageLifecycleStats;
    try {
      stats = this.apply(activeIndices, retiredIndices);
    } catch (error) {
      this.generation = previousGeneration;
      throw error;
    }
    const status = SVO_OWNER_PAGE_STATUS.ready | (stats.overflow > 0 ? SVO_OWNER_PAGE_STATUS.overflow : 0);
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
      free: this.freeSlots.length,
      required: this.required,
      activated: this.activated,
      retired: this.retired,
      overflow: this.overflow,
      generation: this.generation,
      capacity: this.capacity,
      comparedOwners: 0,
      ownerMismatches: 0,
    };
  }
}

export const octreeOwnerPageLifecycleShader = /* wgsl */ `
struct LifecycleParams {
  counts: vec4u, // logical bricks, capacity, active count, retired count
  offsets: vec4u, // page table, free list, owner pages, retired-list base
}
@group(0) @binding(0) var<storage, read_write> arena: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> changes: array<u32>;
@group(0) @binding(2) var<uniform> params: LifecycleParams;

const INVALID: u32 = 0xffffffffu;
const PAGE_VOXELS: u32 = 512u;
var<workgroup> activatedSlot: atomic<u32>;

fn activeIndex(wid: vec3u) -> u32 {
  let width = min(params.counts.z, 65535u);
  return wid.x + wid.y * max(1u, width);
}
fn retiredIndex(wid: vec3u) -> u32 {
  let width = min(params.counts.w, 65535u);
  return wid.x + wid.y * max(1u, width);
}
fn popFreeSlot() -> u32 {
  loop {
    let available = atomicLoad(&arena[0]);
    if (available == 0u) { return INVALID; }
    let claim = atomicCompareExchangeWeak(&arena[0], available, available - 1u);
    if (claim.exchanged) { return atomicLoad(&arena[params.offsets.y + available - 1u]); }
  }
  return INVALID;
}

fn pageValueWord(logical: u32, insert: bool) -> u32 {
  let hashCapacity = (params.offsets.y - params.offsets.x) / 2u;
  let key = logical + 1u;
  var slot = (logical * 0x9e3779b1u) % hashCapacity;
  for (var probe = 0u; probe < hashCapacity; probe += 1u) {
    let keyWord = params.offsets.x + slot;
    let observed = atomicLoad(&arena[keyWord]);
    if (observed == key) { return params.offsets.x + hashCapacity + slot; }
    if (observed == 0u || observed == INVALID) {
      if (!insert) { if (observed == 0u) { return INVALID; } }
      else {
        var expected = observed;
        loop {
          let claim = atomicCompareExchangeWeak(&arena[keyWord], expected, key);
          if (claim.exchanged || claim.old_value == key) { return params.offsets.x + hashCapacity + slot; }
          if (claim.old_value != expected) { break; }
        }
      }
    }
    slot = select(slot + 1u, 0u, slot + 1u == hashCapacity);
  }
  return INVALID;
}

@compute @workgroup_size(1)
fn beginLifecycle() {
  atomicStore(&arena[3], 0u); // overflow
  atomicStore(&arena[4], 0u); // required
  atomicStore(&arena[5], 0u); // activated
  atomicStore(&arena[6], 0u); // retired
  atomicAdd(&arena[7], 1u);   // generation
  atomicStore(&arena[8], 0u); // owner mismatches
  atomicStore(&arena[9], 0u); // compared owners
}

@compute @workgroup_size(64)
fn activatePages(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(local_invocation_index) lid: u32,
) {
  let item = activeIndex(wid);
  if (lid == 0u) { atomicStore(&activatedSlot, INVALID); }
  workgroupBarrier();
  if (item >= params.counts.z) { return; }
  let logical = changes[item];
  if (logical >= params.counts.x) { return; }
  if (lid == 0u) {
    let pageWord = pageValueWord(logical, true);
    if (pageWord == INVALID) { atomicAdd(&arena[3], 1u); return; }
    loop {
      let current = atomicLoad(&arena[pageWord]);
      if (current != 0u) { break; }
      let reserve = atomicCompareExchangeWeak(&arena[pageWord], 0u, INVALID);
      if (!reserve.exchanged) { continue; }
      atomicAdd(&arena[4], 1u);
      let slot = popFreeSlot();
      if (slot == INVALID || slot >= params.counts.y) {
        atomicStore(&arena[pageWord], 0u);
        let hashCapacity = (params.offsets.y - params.offsets.x) / 2u;
        atomicStore(&arena[params.offsets.x + (pageWord - params.offsets.x - hashCapacity)], INVALID);
        atomicAdd(&arena[3], 1u);
        break;
      }
      atomicStore(&activatedSlot, slot);
      atomicStore(&arena[pageWord], slot + 1u);
      let resident = atomicAdd(&arena[1], 1u) + 1u;
      atomicMax(&arena[2], resident);
      atomicAdd(&arena[5], 1u);
      break;
    }
  }
  workgroupBarrier();
  let slot = workgroupUniformLoad(&activatedSlot);
  if (slot == INVALID) { return; }
  let base = params.offsets.z + slot * PAGE_VOXELS;
  for (var local = lid; local < PAGE_VOXELS; local += 64u) {
    atomicStore(&arena[base + local], 0u);
  }
}

@compute @workgroup_size(64)
fn retirePages(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(local_invocation_index) lid: u32,
) {
  let item = retiredIndex(wid);
  if (item >= params.counts.w || lid != 0u) { return; }
  let logical = changes[params.offsets.w + item];
  if (logical >= params.counts.x) { return; }
  let pageWord = pageValueWord(logical, false);
  if (pageWord == INVALID) { return; }
  let encoded = atomicExchange(&arena[pageWord], 0u);
  if (encoded == 0u || encoded == INVALID || encoded > params.counts.y) { return; }
  let hashCapacity = (params.offsets.y - params.offsets.x) / 2u;
  atomicStore(&arena[params.offsets.x + (pageWord - params.offsets.x - hashCapacity)], INVALID);
  let freeIndex = atomicAdd(&arena[0], 1u);
  if (freeIndex >= params.counts.y) {
    atomicStore(&arena[0], params.counts.y);
    atomicAdd(&arena[3], 1u);
    return;
  }
  atomicStore(&arena[params.offsets.y + freeIndex], encoded - 1u);
  atomicSub(&arena[1], 1u);
  atomicAdd(&arena[6], 1u);
}

`;

function tiledWorkgroups(items: number): readonly [number, number, number] {
  const x = Math.min(65_535, items);
  const y = x > 0 ? Math.ceil(items / x) : 1;
  if (y > 65_535) throw new RangeError("Octree owner lifecycle exceeds the two-dimensional WebGPU dispatch range");
  return [x, y, 1];
}

export const octreeSimulationOwnerPageLifecycleShader = /* wgsl */ `
struct Params {
  counts: vec4u,  // logical bricks, capacity, cell dimensions xy
  offsets: vec4u, // page table, free list, payload, cell dimension z
}
@group(0) @binding(0) var<storage, read_write> arena: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> worklist: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
const INVALID: u32 = 0xffffffffu;
const HEADER: u32 = 16u;
const PAGE_VOXELS: u32 = 512u;

fn popFree() -> u32 {
  loop {
    let count = atomicLoad(&arena[0]);
    if (count == 0u) { return INVALID; }
    let claim = atomicCompareExchangeWeak(&arena[0], count, count - 1u);
    if (claim.exchanged) { return atomicLoad(&arena[params.offsets.y + count - 1u]); }
  }
}

fn pageValueWord(logical: u32, insert: bool) -> u32 {
  let hashCapacity = (params.offsets.y - params.offsets.x) / 2u;
  let key = logical + 1u;
  var slot = (logical * 0x9e3779b1u) % hashCapacity;
  for (var probe = 0u; probe < hashCapacity; probe += 1u) {
    let keyWord = params.offsets.x + slot;
    let observed = atomicLoad(&arena[keyWord]);
    if (observed == key) { return params.offsets.x + hashCapacity + slot; }
    if (observed == 0u || observed == INVALID) {
      if (!insert) { if (observed == 0u) { return INVALID; } }
      else {
        var expected = observed;
        loop {
          let claim = atomicCompareExchangeWeak(&arena[keyWord], expected, key);
          if (claim.exchanged || claim.old_value == key) { return params.offsets.x + hashCapacity + slot; }
          if (claim.old_value != expected) { break; }
        }
      }
    }
    slot = select(slot + 1u, 0u, slot + 1u == hashCapacity);
  }
  return INVALID;
}

@compute @workgroup_size(1)
fn beginLifecycle() { atomicStore(&arena[2], 0u); atomicAdd(&arena[7], 1u); }

@compute @workgroup_size(64)
fn retirePages(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) numWorkgroups: vec3u, @builtin(local_invocation_index) lid: u32) {
  let linearGroup = wid.x + numWorkgroups.x * wid.y;
  if ((linearGroup & 1u) != 0u) { return; }
  let item = linearGroup >> 1u;
  let count = min(worklist[4], params.counts.x);
  if (item >= count || lid != 0u) { return; }
  let logical = worklist[HEADER + params.counts.x * 2u + item * 2u];
  if (logical >= params.counts.x) { atomicStore(&arena[2], 2u); return; }
  let pageWord = pageValueWord(logical, false);
  if (pageWord == INVALID) { return; }
  let encoded = atomicExchange(&arena[pageWord], 0u);
  if (encoded == 0u || encoded == INVALID || encoded > params.counts.y) { return; }
  let hashCapacity = (params.offsets.y - params.offsets.x) / 2u;
  atomicStore(&arena[params.offsets.x + (pageWord - params.offsets.x - hashCapacity)], INVALID);
  let free = atomicAdd(&arena[0], 1u);
  if (free < params.counts.y) { atomicStore(&arena[params.offsets.y + free], encoded - 1u); atomicSub(&arena[1], 1u); }
  else { atomicStore(&arena[0], params.counts.y); atomicStore(&arena[2], 2u); }
}

@compute @workgroup_size(1)
fn activatePages(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) numWorkgroups: vec3u) {
  let linearGroup = wid.x + numWorkgroups.x * wid.y;
  if ((linearGroup & 1u) != 0u) { return; }
  let item = linearGroup >> 1u;
  let count = min(worklist[0], params.counts.x);
  if (item >= count) { return; }
  let logical = worklist[HEADER + item * 2u];
  if (logical >= params.counts.x) { atomicStore(&arena[2], 2u); return; }
  let table = pageValueWord(logical, true);
  if (table == INVALID) { atomicStore(&arena[2], 1u); return; }
  var reserved = false;
  loop {
    let reserve = atomicCompareExchangeWeak(&arena[table], 0u, INVALID);
    if (reserve.exchanged) { reserved = true; break; }
    if (reserve.old_value != 0u) { break; }
  }
  if (!reserved) { return; }
  let slot = popFree();
  if (slot == INVALID || slot >= params.counts.y) {
    atomicStore(&arena[table], 0u);
    let hashCapacity = (params.offsets.y - params.offsets.x) / 2u;
    atomicStore(&arena[params.offsets.x + (table - params.offsets.x - hashCapacity)], INVALID);
    atomicStore(&arena[2], 1u);
    return;
  }
  atomicStore(&arena[table], slot + 1u);
  atomicAdd(&arena[1], 1u);
  let base = params.offsets.z + slot * PAGE_VOXELS;
  for (var local = 0u; local < PAGE_VOXELS; local += 1u) { atomicStore(&arena[base + local], 0u); }
}

fn analyticOwnerWord(origin: vec3u, size: u32) -> u32 {
  if (size == 1u) { return 0x80000000u; }
  let exponent = u32(firstTrailingBit(size));
  let aligned = origin >> vec3u(exponent);
  return exponent | (aligned.x << 3u) | (aligned.y << 12u) | (aligned.z << 21u);
}

// One workgroup consumes one analytically published topology tile. Each lane
// owns at most one 8^3 page (tileSize <= 32), allocates it from the bounded
// arena, and seeds the exact coarse owner that resetTopology would publish.
// This closes the cold-start gap where brick residency is still empty and a
// missing page would otherwise cap lookup at the synthetic size-8 owner.
@compute @workgroup_size(64)
fn activateAnalyticTopologyPages(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) numWorkgroups: vec3u,
  @builtin(local_invocation_index) lid: u32,
) {
  let tileSlot = wid.x + numWorkgroups.x * wid.y;
  let dimensions = vec3u(params.counts.z, params.counts.w, params.offsets.w);
  let tileSize = atomicLoad(&arena[14]);
  let tileDimensions = (dimensions + vec3u(tileSize - 1u)) / tileSize;
  let tileCapacity = tileDimensions.x * tileDimensions.y * tileDimensions.z;
  let tileCount = min(worklist[0], tileCapacity);
  if (tileSlot >= tileCount || tileSize < 8u || tileSize > 32u || (tileSize & (tileSize - 1u)) != 0u) { return; }
  let tileIndex = worklist[HEADER + tileSlot];
  if (tileIndex >= tileCapacity) { atomicStore(&arena[2], 2u); return; }
  let tile = vec3u(tileIndex % tileDimensions.x,
    (tileIndex / tileDimensions.x) % tileDimensions.y,
    tileIndex / (tileDimensions.x * tileDimensions.y));
  let pagesPerAxis = tileSize / 8u;
  let pagesPerTile = pagesPerAxis * pagesPerAxis * pagesPerAxis;
  if (lid >= pagesPerTile) { return; }
  let localBrick = vec3u(lid % pagesPerAxis,
    (lid / pagesPerAxis) % pagesPerAxis,
    lid / (pagesPerAxis * pagesPerAxis));
  let brick = tile * vec3u(pagesPerAxis) + localBrick;
  let brickDimensions = (dimensions + vec3u(7u)) / 8u;
  if (any(brick >= brickDimensions)) { return; }
  let logical = brick.x + brick.y * brickDimensions.x + brick.z * brickDimensions.x * brickDimensions.y;
  if (logical >= params.counts.x) { atomicStore(&arena[2], 2u); return; }
  let table = pageValueWord(logical, true);
  if (table == INVALID) { atomicStore(&arena[2], 1u); return; }
  var reserved = false;
  loop {
    let reserve = atomicCompareExchangeWeak(&arena[table], 0u, INVALID);
    if (reserve.exchanged) { reserved = true; break; }
    if (reserve.old_value != 0u) { break; }
  }
  if (!reserved) { return; }
  let slot = popFree();
  if (slot == INVALID || slot >= params.counts.y) {
    atomicStore(&arena[table], 0u);
    let hashCapacity = (params.offsets.y - params.offsets.x) / 2u;
    atomicStore(&arena[params.offsets.x + (table - params.offsets.x - hashCapacity)], INVALID);
    atomicStore(&arena[2], 1u);
    return;
  }
  let base = params.offsets.z + slot * PAGE_VOXELS;
  let brickOrigin = brick * 8u;
  for (var local = 0u; local < PAGE_VOXELS; local += 1u) {
    let cell = brickOrigin + vec3u(local % 8u, (local / 8u) % 8u, local / 64u);
    var word = 0u;
    if (all(cell < dimensions)) {
      var size = tileSize;
      var origin = (cell / vec3u(size)) * vec3u(size);
      loop {
        if (all(origin + vec3u(size) <= dimensions) || size == 1u) { break; }
        size >>= 1u;
        origin = (cell / vec3u(size)) * vec3u(size);
      }
      word = analyticOwnerWord(origin, size);
    }
    atomicStore(&arena[base + local], word);
  }
  atomicStore(&arena[table], slot + 1u);
  atomicAdd(&arena[1], 1u);
}
`;

export interface OctreeAnalyticOwnerBootstrapSource {
  readonly tileWorklist: GPUBuffer;
  readonly tileSizeCells: number;
  readonly activeTileLimits: readonly [number, number, number];
  readonly activeTileCount: number;
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
  readonly allocatedBytes: number;
  private readonly params: GPUBuffer;
  private readonly group: GPUBindGroup;
  private readonly begin: GPUComputePipeline;
  private readonly retire: GPUComputePipeline;
  private readonly activate: GPUComputePipeline;
  private readonly worklist: GPUBuffer;
  private readonly analyticBootstrap?: OctreeAnalyticOwnerBootstrapSource;
  private readonly analyticGroup?: GPUBindGroup;
  private readonly activateAnalytic?: GPUComputePipeline;

  constructor(device: GPUDevice, dimensions: OctreeOwnerCoordinate, worklist: GPUBuffer,
    options: OctreeOwnerPagePlanOptions = {}, analyticBootstrap?: OctreeAnalyticOwnerBootstrapSource) {
    this.worklist = worklist;
    this.analyticBootstrap = analyticBootstrap;
    const analyticMinimumPages = analyticBootstrap
      ? octreeAnalyticOwnerBootstrapPageCount(dimensions, analyticBootstrap)
      : 1;
    this.plan = planOctreeOwnerPages(dimensions, {
      ...options,
      minimumPages: Math.max(options.minimumPages ?? 1, analyticMinimumPages),
    });
    this.arena = device.createBuffer({ label: "Simulation octree owner pages", size: this.plan.allocatedBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.params = device.createBuffer({ label: "Simulation octree owner-page parameters", size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const initial = new Uint32Array(this.plan.allocatedWords);
    initial[0] = this.plan.capacity; initial[3] = this.plan.capacity; initial[4] = this.plan.logicalBrickCount;
    initial[5] = this.plan.freeListOffsetWords; initial[6] = this.plan.ownerPagesOffsetWords;
    initial[14] = analyticBootstrap?.tileSizeCells ?? 0;
    initial[15] = 0x4f57_4e52;
    for (let index = 0; index < this.plan.capacity; index += 1) initial[this.plan.freeListOffsetWords + index] = this.plan.capacity - 1 - index;
    device.queue.writeBuffer(this.arena, 0, initial);
    device.queue.writeBuffer(this.params, 0, new Uint32Array([
      this.plan.logicalBrickCount, this.plan.capacity, dimensions[0], dimensions[1],
      this.plan.pageTableOffsetWords, this.plan.freeListOffsetWords, this.plan.ownerPagesOffsetWords, dimensions[2],
    ]));
    const layout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ] });
    const module = device.createShaderModule({ label: "Simulation octree owner-page lifecycle", code: octreeSimulationOwnerPageLifecycleShader });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    this.begin = device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint: "beginLifecycle" } });
    this.retire = device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint: "retirePages" } });
    this.activate = device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint: "activatePages" } });
    this.group = device.createBindGroup({ layout, entries: [
      { binding: 0, resource: { buffer: this.arena } }, { binding: 1, resource: { buffer: worklist } }, { binding: 2, resource: { buffer: this.params } },
    ] });
    if (analyticBootstrap) {
      this.activateAnalytic = device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint: "activateAnalyticTopologyPages" } });
      this.analyticGroup = device.createBindGroup({ layout, entries: [
        { binding: 0, resource: { buffer: this.arena } },
        { binding: 1, resource: { buffer: analyticBootstrap.tileWorklist } },
        { binding: 2, resource: { buffer: this.params } },
      ] });
    }
    this.allocatedBytes = this.arena.size + this.params.size;
  }

  encode(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass({ label: "Evolve simulation octree owner pages" });
    pass.setBindGroup(0, this.group); pass.setPipeline(this.begin); pass.dispatchWorkgroups(1);
    pass.setPipeline(this.retire); pass.dispatchWorkgroupsIndirect(this.worklist, FLUID_BRICK_RETIRED_DISPATCH_OFFSET_BYTES);
    pass.setPipeline(this.activate); pass.dispatchWorkgroupsIndirect(this.worklist, FLUID_BRICK_ACTIVE_DISPATCH_OFFSET_BYTES);
    pass.end();
  }

  /** One-time, GPU-only coarse owner publication for the analytic cold tile set. */
  encodeAnalyticBootstrap(encoder: GPUCommandEncoder): void {
    if (!this.analyticBootstrap || !this.analyticGroup || !this.activateAnalytic) {
      throw new Error("Analytic owner-page bootstrap was not configured");
    }
    const pass = encoder.beginComputePass({ label: "Seed analytic octree owner pages" });
    pass.setBindGroup(0, this.analyticGroup);
    pass.setPipeline(this.begin); pass.dispatchWorkgroups(1);
    if (this.analyticBootstrap.activeTileCount > 0) {
      pass.setPipeline(this.activateAnalytic);
      pass.dispatchWorkgroups(...tiledWorkgroups(this.analyticBootstrap.activeTileCount));
    }
    pass.end();
  }

  destroy(): void { this.arena.destroy(); this.params.destroy(); }
}

/** Self-contained GPU allocation skeleton. It is not bound to projection yet. */
export class GPUOctreeOwnerPageArena {
  readonly plan: OctreeOwnerPagePlan;
  readonly arena: GPUBuffer;
  readonly allocatedBytes: number;
  private readonly changes: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly bindGroup: GPUBindGroup;
  private readonly beginPipeline: GPUComputePipeline;
  private readonly activatePipeline: GPUComputePipeline;
  private readonly retirePipeline: GPUComputePipeline;
  private readonly changeData: Uint32Array<ArrayBuffer>;
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    dimensions: OctreeOwnerCoordinate,
    options: OctreeOwnerPagePlanOptions = {},
  ) {
    this.plan = planOctreeOwnerPages(dimensions, options);
    const storageCopy = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.arena = device.createBuffer({ label: "Octree owner page arena", size: this.plan.allocatedBytes, usage: storageCopy });
    this.changes = device.createBuffer({
      label: "Octree owner page lifecycle changes",
      size: Math.max(8, this.plan.logicalBrickCount * 2 * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.params = device.createBuffer({ label: "Octree owner page lifecycle parameters", size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.changeData = new Uint32Array(this.plan.logicalBrickCount * 2);
    const initial = new Uint32Array(this.plan.allocatedWords);
    initial[0] = this.plan.capacity;
    for (let index = 0; index < this.plan.capacity; index += 1) {
      initial[this.plan.freeListOffsetWords + index] = this.plan.capacity - 1 - index;
    }
    device.queue.writeBuffer(this.arena, 0, initial);

    const layout = device.createBindGroupLayout({ label: "Octree owner page lifecycle layout", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ] });
    const lifecycleModule = device.createShaderModule({ label: "Octree owner page lifecycle", code: octreeOwnerPageLifecycleShader });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    this.beginPipeline = device.createComputePipeline({ label: "Begin octree owner page lifecycle", layout: pipelineLayout, compute: { module: lifecycleModule, entryPoint: "beginLifecycle" } });
    this.activatePipeline = device.createComputePipeline({ label: "Activate octree owner pages", layout: pipelineLayout, compute: { module: lifecycleModule, entryPoint: "activatePages" } });
    this.retirePipeline = device.createComputePipeline({ label: "Retire octree owner pages", layout: pipelineLayout, compute: { module: lifecycleModule, entryPoint: "retirePages" } });
    this.bindGroup = device.createBindGroup({ label: "Octree owner page lifecycle bindings", layout, entries: [
      { binding: 0, resource: { buffer: this.arena } },
      { binding: 1, resource: { buffer: this.changes } },
      { binding: 2, resource: { buffer: this.params } },
    ] });
    this.allocatedBytes = this.arena.size + this.changes.size + this.params.size;
  }

  encodeLifecycle(encoder: GPUCommandEncoder, activeIndices: readonly number[], retiredIndices: readonly number[]): void {
    if (this.destroyed) return;
    const validate = (indices: readonly number[], label: string) => {
      for (const index of indices) if (!Number.isSafeInteger(index) || index < 0 || index >= this.plan.logicalBrickCount) {
        throw new RangeError(`${label} contains an invalid logical brick index`);
      }
    };
    validate(activeIndices, "Octree owner active list");
    validate(retiredIndices, "Octree owner retired list");
    const active = [...new Set(activeIndices)], retired = [...new Set(retiredIndices)];
    const activeSet = new Set(active);
    if (retired.some((index) => activeSet.has(index))) throw new RangeError("A logical owner page cannot be active and retired in the same publication");
    this.changeData.set(active, 0);
    this.changeData.set(retired, this.plan.logicalBrickCount);
    this.device.queue.writeBuffer(this.changes, 0, this.changeData);
    this.device.queue.writeBuffer(this.params, 0, new Uint32Array([
      this.plan.logicalBrickCount, this.plan.capacity, active.length, retired.length,
      this.plan.pageTableOffsetWords, this.plan.freeListOffsetWords, this.plan.ownerPagesOffsetWords, this.plan.logicalBrickCount,
    ]));
    const pass = encoder.beginComputePass({ label: "Evolve octree owner page lifecycle" });
    pass.setBindGroup(0, this.bindGroup);
    pass.setPipeline(this.beginPipeline); pass.dispatchWorkgroups(1);
    if (active.length > 0) { pass.setPipeline(this.activatePipeline); pass.dispatchWorkgroups(...tiledWorkgroups(active.length)); }
    if (retired.length > 0) { pass.setPipeline(this.retirePipeline); pass.dispatchWorkgroups(...tiledWorkgroups(retired.length)); }
    pass.end();
  }

  async readState(): Promise<{ stats: OctreeOwnerPageLifecycleStats; pageTable: Uint32Array<ArrayBuffer> }> {
    if (this.destroyed) throw new Error("Octree owner page arena has been destroyed");
    const words = OCTREE_OWNER_ARENA_CONTROL_WORDS + this.plan.pageHashCapacity * 2;
    const readback = this.device.createBuffer({ label: "Octree owner page state readback", size: words * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder({ label: "Read octree owner page state" });
    encoder.copyBufferToBuffer(this.arena, 0, readback, 0, words * 4);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const data = new Uint32Array(readback.getMappedRange().slice(0));
      const pageTable = new Uint32Array(this.plan.logicalBrickCount);
      for (let slot = 0; slot < this.plan.pageHashCapacity; slot += 1) {
        const key = data[this.plan.pageTableOffsetWords + slot];
        if (key !== OCTREE_OWNER_PAGE_HASH_EMPTY && key !== OCTREE_OWNER_PAGE_HASH_TOMBSTONE && key <= this.plan.logicalBrickCount) {
          pageTable[key - 1] = data[this.plan.pageTableValueOffsetWords + slot];
        }
      }
      return {
        stats: {
          free: data[0], resident: data[1], peakResident: data[2], overflow: data[3],
          required: data[4], activated: data[5], retired: data[6], generation: data[7], capacity: this.plan.capacity,
          ownerMismatches: data[8], comparedOwners: data[9],
        },
        pageTable,
      };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  async readStats(): Promise<OctreeOwnerPageLifecycleStats> {
    if (this.destroyed) throw new Error("Octree owner page arena has been destroyed");
    const readback = this.device.createBuffer({ label: "Octree owner page telemetry readback", size: 40, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder({ label: "Read octree owner page telemetry" });
    encoder.copyBufferToBuffer(this.arena, 0, readback, 0, 40);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const data = new Uint32Array(readback.getMappedRange());
      return {
        free: data[0], resident: data[1], peakResident: data[2], overflow: data[3],
        required: data[4], activated: data[5], retired: data[6], generation: data[7],
        ownerMismatches: data[8], comparedOwners: data[9], capacity: this.plan.capacity,
      };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.arena.destroy();
    this.changes.destroy();
    this.params.destroy();
  }
}

/**
 * Ordered allocator for renderer-owned owner pages.
 *
 * This consumes only the validated, compacted output of
 * `WebGPUSvoRenderResidencyConsumer`. The producer publication remains
 * immutable, all lifecycle state stays on the GPU, and readers must compare
 * control word `acceptedGeneration` before using the page table. Payload words
 * are cleared on activation but are not populated with fine phi by this class.
 */
export const svoRendererOwnerPageAllocatorShader = /* wgsl */ `
struct Params {
  countsOffsets: vec4u, // logical count, capacity, active-list words, retired-list words
  arenaOffsets: vec4u,  // page table, free list, payload, source list capacity
}
@group(0) @binding(0) var<storage, read_write> arena: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> sourceControl: array<u32>;
@group(0) @binding(2) var<storage, read> sourceEntries: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> retiredSlots: array<u32>;

const PAGE_VOXELS: u32 = ${OCTREE_OWNER_PAGE_VOXELS}u;
const READY: u32 = ${SVO_OWNER_PAGE_STATUS.ready}u;
const UNCHANGED: u32 = ${SVO_OWNER_PAGE_STATUS.unchanged}u;
const STALE: u32 = ${SVO_OWNER_PAGE_STATUS.stale}u;
const UNPUBLISHED: u32 = ${SVO_OWNER_PAGE_STATUS.unpublished}u;
const SOURCE_REJECTED: u32 = ${SVO_OWNER_PAGE_STATUS.sourceRejected}u;
const INVALID_ENTRY: u32 = ${SVO_OWNER_PAGE_STATUS.invalidEntry}u;
const OVERFLOW: u32 = ${SVO_OWNER_PAGE_STATUS.overflow}u;
const SOURCE_DEGRADED: u32 = ${SVO_OWNER_PAGE_STATUS.sourceDegraded}u;
const SOURCE_READY: u32 = ${SVO_RENDER_RESIDENCY_CONSUMER_STATUS.ready}u;
const SOURCE_UNCHANGED: u32 = ${SVO_RENDER_RESIDENCY_CONSUMER_STATUS.unchanged}u;
const SOURCE_STALE: u32 = ${SVO_RENDER_RESIDENCY_CONSUMER_STATUS.stale}u;
const SOURCE_INVALID: u32 = ${SVO_RENDER_RESIDENCY_CONSUMER_STATUS.invalidEntry}u;
const SOURCE_DEGRADED_MASK: u32 = ${SVO_RENDER_RESIDENCY_CONSUMER_STATUS.sourceOverflow
  | SVO_RENDER_RESIDENCY_CONSUMER_STATUS.rendererExhausted
  | SVO_RENDER_RESIDENCY_CONSUMER_STATUS.coarseFallback}u;

fn clearAttemptTelemetry(observed: u32) {
  atomicStore(&arena[3], 0u);
  atomicStore(&arena[4], 0u);
  atomicStore(&arena[5], 0u);
  atomicStore(&arena[6], 0u);
  atomicStore(&arena[8], 0u);
  atomicStore(&arena[9], 0u);
  atomicStore(&arena[10], 0u);
  atomicStore(&arena[11], observed);
  atomicStore(&arena[12], 0u);
}

fn invalidateEntry() {
  atomicAdd(&arena[12], 1u);
  atomicOr(&arena[10], INVALID_ENTRY);
}

fn pageValueWord(logical: u32, insert: bool) -> u32 {
  let hashCapacity = (params.arenaOffsets.y - params.arenaOffsets.x) / 2u;
  let key = logical + 1u;
  var slot = (logical * 0x9e3779b1u) % hashCapacity;
  var firstTombstone = 0xffffffffu;
  for (var probe = 0u; probe < hashCapacity; probe += 1u) {
    let observed = atomicLoad(&arena[params.arenaOffsets.x + slot]);
    if (observed == key) { return params.arenaOffsets.x + hashCapacity + slot; }
    if (observed == 0xffffffffu && firstTombstone == 0xffffffffu) { firstTombstone = slot; }
    if (observed == 0u) {
      if (!insert) { return 0xffffffffu; }
      let targetSlot = select(firstTombstone, slot, firstTombstone == 0xffffffffu);
      atomicStore(&arena[params.arenaOffsets.x + targetSlot], key);
      return params.arenaOffsets.x + hashCapacity + targetSlot;
    }
    slot = select(slot + 1u, 0u, slot + 1u == hashCapacity);
  }
  if (insert && firstTombstone != 0xffffffffu) {
    atomicStore(&arena[params.arenaOffsets.x + firstTombstone], key);
    return params.arenaOffsets.x + hashCapacity + firstTombstone;
  }
  return 0xffffffffu;
}

// One invocation owns allocation order. The compact source preserves list
// order, reverse-initialized free storage therefore yields slots 0, 1, ... and
// retired slots are unavailable until all activations in this publication end.
fn activate(logical: u32) {
  if (logical >= params.countsOffsets.x) { invalidateEntry(); return; }
  let pageWord = pageValueWord(logical, true);
  if (pageWord == 0xffffffffu) { atomicAdd(&arena[3], 1u); atomicOr(&arena[10], OVERFLOW); return; }
  if (atomicLoad(&arena[pageWord]) != 0u) { return; }
  atomicAdd(&arena[4], 1u);
  let available = atomicLoad(&arena[0]);
  if (available == 0u) {
    let hashCapacity = (params.arenaOffsets.y - params.arenaOffsets.x) / 2u;
    atomicStore(&arena[params.arenaOffsets.x + (pageWord - params.arenaOffsets.x - hashCapacity)], 0xffffffffu);
    atomicAdd(&arena[3], 1u);
    atomicOr(&arena[10], OVERFLOW);
    return;
  }
  let freeWord = params.arenaOffsets.y + available - 1u;
  let slot = atomicLoad(&arena[freeWord]);
  if (slot >= params.countsOffsets.y) {
    let hashCapacity = (params.arenaOffsets.y - params.arenaOffsets.x) / 2u;
    atomicStore(&arena[params.arenaOffsets.x + (pageWord - params.arenaOffsets.x - hashCapacity)], 0xffffffffu);
    atomicAdd(&arena[3], 1u);
    atomicOr(&arena[10], OVERFLOW | INVALID_ENTRY);
    return;
  }
  atomicStore(&arena[0], available - 1u);
  atomicStore(&arena[pageWord], slot + 1u);
  let resident = atomicAdd(&arena[1], 1u) + 1u;
  atomicMax(&arena[2], resident);
  atomicAdd(&arena[5], 1u);
  let payload = params.arenaOffsets.z + slot * PAGE_VOXELS;
  for (var local = 0u; local < PAGE_VOXELS; local += 1u) {
    atomicStore(&arena[payload + local], 0u);
  }
}

fn retire(logical: u32, item: u32) {
  if (item < arrayLength(&retiredSlots)) { retiredSlots[item] = 0xffffffffu; }
  if (logical >= params.countsOffsets.x) { invalidateEntry(); return; }
  let pageWord = pageValueWord(logical, false);
  if (pageWord == 0xffffffffu) { return; }
  let encoded = atomicLoad(&arena[pageWord]);
  if (encoded == 0u) { return; }
  if (encoded > params.countsOffsets.y) { invalidateEntry(); return; }
  let freeCount = atomicLoad(&arena[0]);
  if (freeCount >= params.countsOffsets.y) { invalidateEntry(); return; }
  atomicStore(&arena[pageWord], 0u);
  let hashCapacity = (params.arenaOffsets.y - params.arenaOffsets.x) / 2u;
  atomicStore(&arena[params.arenaOffsets.x + (pageWord - params.arenaOffsets.x - hashCapacity)], 0xffffffffu);
  if (item < arrayLength(&retiredSlots)) { retiredSlots[item] = encoded - 1u; }
  atomicStore(&arena[params.arenaOffsets.y + freeCount], encoded - 1u);
  atomicStore(&arena[0], freeCount + 1u);
  atomicSub(&arena[1], 1u);
  atomicAdd(&arena[6], 1u);
}

@compute @workgroup_size(1)
fn applyResidencyPublication() {
  let observed = sourceControl[${SVO_RENDER_RESIDENCY_CONSUMER_CONTROL_WORDS.observedGeneration}u];
  let sourceGeneration = sourceControl[${SVO_RENDER_RESIDENCY_CONSUMER_CONTROL_WORDS.acceptedGeneration}u];
  let accepted = atomicLoad(&arena[7]);
  clearAttemptTelemetry(observed);
  if (observed == 0u) {
    atomicStore(&arena[10], UNPUBLISHED);
    atomicAdd(&arena[15], 1u);
    return;
  }
  let sourceStatus = sourceControl[${SVO_RENDER_RESIDENCY_CONSUMER_CONTROL_WORDS.status}u];
  if ((sourceStatus & SOURCE_READY) == 0u || (sourceStatus & SOURCE_INVALID) != 0u) {
    if ((sourceStatus & SOURCE_UNCHANGED) != 0u && sourceGeneration == accepted) {
      atomicStore(&arena[10], UNCHANGED);
      atomicAdd(&arena[14], 1u);
    } else if ((sourceStatus & SOURCE_STALE) != 0u) {
      atomicStore(&arena[10], STALE);
      atomicAdd(&arena[13], 1u);
    } else {
      atomicStore(&arena[10], SOURCE_REJECTED);
    }
    return;
  }
  if (sourceGeneration < accepted) {
    atomicStore(&arena[10], STALE);
    atomicAdd(&arena[13], 1u);
    return;
  }
  if (sourceGeneration == accepted && sourceGeneration != 0u) {
    atomicStore(&arena[10], UNCHANGED);
    atomicAdd(&arena[14], 1u);
    return;
  }
  if (sourceGeneration == 0u) {
    atomicStore(&arena[10], SOURCE_REJECTED);
    return;
  }
  atomicStore(&arena[10], READY | select(0u, SOURCE_DEGRADED, (sourceStatus & SOURCE_DEGRADED_MASK) != 0u));
  let activeCount = min(sourceControl[${SVO_RENDER_RESIDENCY_CONSUMER_CONTROL_WORDS.dirtyActiveCount}u], params.arenaOffsets.w);
  let retiredCount = min(sourceControl[${SVO_RENDER_RESIDENCY_CONSUMER_CONTROL_WORDS.dirtyRetiredCount}u], params.arenaOffsets.w);
  for (var item = 0u; item < activeCount; item += 1u) {
    activate(sourceEntries[params.countsOffsets.z + item * 2u]);
  }
  for (var item = 0u; item < retiredCount; item += 1u) {
    retire(sourceEntries[params.countsOffsets.w + item * 2u], item);
  }
  // Publication becomes visible only after activation, zero-fill, and
  // retirement have completed in command order.
  atomicStore(&arena[7], sourceGeneration);
}
`;

export class WebGPUSvoOwnerPageAllocator {
  readonly plan: OctreeOwnerPagePlan;
  readonly arena: GPUBuffer;
  /** Physical slots released by each compact retired-list item in the last accepted publication. */
  readonly retiredSlots: GPUBuffer;
  readonly allocatedBytes: number;
  private readonly params: GPUBuffer;
  private readonly bindGroup: GPUBindGroup;
  private readonly pipeline: GPUComputePipeline;
  private destroyed = false;

  constructor(
    device: GPUDevice,
    dimensions: OctreeOwnerCoordinate,
    source: WebGPUSvoRenderResidencyConsumer,
    options: OctreeOwnerPagePlanOptions = {},
  ) {
    this.plan = planOctreeOwnerPages(dimensions, options);
    if (source.layout.capacity < 1) throw new RangeError("SVO owner-page source capacity must be positive");
    const storageCopy = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.arena = device.createBuffer({ label: "SVO renderer owner-page arena", size: this.plan.allocatedBytes, usage: storageCopy });
    this.retiredSlots = device.createBuffer({
      label: "SVO renderer owner-page retired slots",
      size: Math.max(4, source.layout.capacity * 4),
      usage: storageCopy,
    });
    this.params = device.createBuffer({ label: "SVO renderer owner-page parameters", size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const initial = new Uint32Array(this.plan.allocatedWords);
    initial[SVO_OWNER_PAGE_CONTROL_WORDS.freeCount] = this.plan.capacity;
    for (let index = 0; index < this.plan.capacity; index += 1) {
      initial[this.plan.freeListOffsetWords + index] = this.plan.capacity - 1 - index;
    }
    device.queue.writeBuffer(this.arena, 0, initial);
    device.queue.writeBuffer(this.params, 0, new Uint32Array([
      this.plan.logicalBrickCount, this.plan.capacity,
      source.layout.entryOffsetsBytes.active / 4, source.layout.entryOffsetsBytes.retired / 4,
      this.plan.pageTableOffsetWords, this.plan.freeListOffsetWords,
      this.plan.ownerPagesOffsetWords, source.layout.capacity,
    ]));
    const layout = device.createBindGroupLayout({ label: "SVO renderer owner-page allocator layout", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ] });
    const shaderModule = device.createShaderModule({ label: "SVO renderer owner-page allocator", code: svoRendererOwnerPageAllocatorShader });
    this.pipeline = device.createComputePipeline({
      label: "Apply SVO renderer owner-page publication",
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module: shaderModule, entryPoint: "applyResidencyPublication" },
    });
    this.bindGroup = device.createBindGroup({ label: "SVO renderer owner-page allocator bindings", layout, entries: [
      { binding: 0, resource: { buffer: this.arena } },
      { binding: 1, resource: { buffer: source.control } },
      { binding: 2, resource: { buffer: source.entries } },
      { binding: 3, resource: { buffer: this.params } },
      { binding: 4, resource: { buffer: this.retiredSlots } },
    ] });
    this.allocatedBytes = this.arena.size + this.retiredSlots.size + this.params.size;
  }

  /** Apply the latest accepted renderer-residency publication entirely on-GPU. */
  encode(encoder: GPUCommandEncoder): void {
    if (this.destroyed) return;
    const pass = encoder.beginComputePass({ label: "Apply SVO renderer owner-page residency" });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
  }

  telemetryBinding(): GPUBufferBinding {
    return { buffer: this.arena, offset: 0, size: OCTREE_OWNER_ARENA_CONTROL_WORDS * 4 };
  }

  /** Bind the aligned whole arena; shaders use the word offsets in `plan`. */
  storageBinding(): GPUBufferBinding {
    return { buffer: this.arena, offset: 0, size: this.plan.allocatedBytes };
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.params.destroy();
    this.retiredSlots.destroy();
    this.arena.destroy();
  }
}
