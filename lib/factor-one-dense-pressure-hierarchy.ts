/**
 * CPU authority for the factor-1 dense-indexed pressure hierarchy.
 *
 * "Dense" describes the coordinate address, not the cell roles: absent cells
 * retain a zero flag and every operation exits for them. The x-fast address is
 * the canonical coordinate-keyed serialization used by differential tests,
 * not a frozen physical GPU ABI. A GPU executor may use versioned channel
 * strides and compact occupied-index worklists while preserving these keys.
 */

export type FactorOneDenseCoordinate = readonly [number, number, number];

export const FACTOR_ONE_DENSE_CELL_ROLE = Object.freeze({
  absent: 0,
  active: 1,
  ghost: 2,
  multigridOnly: 4,
} as const);

export type FactorOneDenseOccupiedRole =
  typeof FACTOR_ONE_DENSE_CELL_ROLE.active
  | typeof FACTOR_ONE_DENSE_CELL_ROLE.ghost
  | typeof FACTOR_ONE_DENSE_CELL_ROLE.multigridOnly;

export interface FactorOneDenseHierarchyPlan {
  readonly dimensions: FactorOneDenseCoordinate;
  readonly levelDimensions: readonly FactorOneDenseCoordinate[];
  readonly levelBases: readonly number[];
  readonly levelVolumes: readonly number[];
  readonly levelCount: number;
  readonly totalSlots: number;
}

export interface FactorOneDenseHierarchySeed {
  readonly level: number;
  readonly coordinate: FactorOneDenseCoordinate;
  readonly role: FactorOneDenseOccupiedRole;
  /** Zero-based accepted pressure row. Required for ACTIVE and GHOST. */
  readonly owner?: number;
}

export interface FactorOneDenseHierarchy {
  readonly plan: FactorOneDenseHierarchyPlan;
  /** Exactly zero or one of ACTIVE/GHOST/MG_ONLY. */
  readonly flags: Uint32Array;
  /** GPU-compatible row + 1 encoding; zero means no owner. */
  readonly owners: Uint32Array;
  /**
   * Strictly increasing x-fast indices local to each level. These select
   * recurring work without changing the direct-addressed channel slots.
   */
  readonly occupiedLocalIndices: readonly Uint32Array[];
}

export const FACTOR_ONE_DENSE_GPU_PHYSICAL_PLAN_VERSION = 1 as const;
export const FACTOR_ONE_DENSE_GPU_CHANNEL_ALIGNMENT_ELEMENTS = 64;
export const FACTOR_ONE_DENSE_GPU_SCALAR_BYTES = 4;
export const FACTOR_ONE_DENSE_GPU_WORKLIST_RECORD_WORDS = 4;

export const FACTOR_ONE_DENSE_GPU_PUBLICATION_WORD = Object.freeze({
  acceptedValid: 0,
  acceptedEpoch: 1,
  acceptedError: 2,
  candidateValid: 4,
  candidateEpoch: 5,
  candidateError: 6,
} as const);

export const FACTOR_ONE_DENSE_GPU_CHANNEL_NAMES = Object.freeze([
  "acceptedFlags",
  "acceptedOwners",
  "acceptedDiagonal",
  "candidateFlags",
  "candidateOwners",
  "candidateDiagonal",
  "rhs",
  "a",
  "b",
] as const);

export type FactorOneDenseGpuChannelName =
  typeof FACTOR_ONE_DENSE_GPU_CHANNEL_NAMES[number];

export interface FactorOneDenseGpuPhysicalRegion {
  readonly name: string;
  readonly scalar: "u32" | "f32";
  readonly offsetBytes: number;
  readonly payloadElements: number;
  readonly allocatedElements: number;
  readonly payloadBytes: number;
  readonly allocatedBytes: number;
}

export interface FactorOneDenseGpuChannelRegion extends FactorOneDenseGpuPhysicalRegion {
  readonly name: FactorOneDenseGpuChannelName;
  /** Global semantic slot `s` is stored at offsetBytes + 4*s. */
  readonly directAddressed: true;
}

export interface FactorOneDenseGpuWorklistLevel {
  readonly level: number;
  readonly capacityElements: number;
  readonly acceptedIndicesOffsetBytes: number;
  readonly candidateIndicesOffsetBytes: number;
  readonly acceptedCountOffsetBytes: number;
  readonly candidateCountOffsetBytes: number;
  /** Three-word WebGPU indirect record immediately after the count word. */
  readonly acceptedDispatchOffsetBytes: number;
  readonly candidateDispatchOffsetBytes: number;
}

/**
 * Experimental host description of the first x-fast SoA GPU image. Versioning
 * makes layout changes explicit; this is intentionally not a frozen ABI.
 */
export interface FactorOneDenseGpuPhysicalPlan {
  readonly version: typeof FACTOR_ONE_DENSE_GPU_PHYSICAL_PLAN_VERSION;
  readonly stability: "prototype-not-frozen";
  readonly storageOrder: "x-fast-soa";
  readonly scalarBytes: typeof FACTOR_ONE_DENSE_GPU_SCALAR_BYTES;
  readonly channelAlignmentElements: typeof FACTOR_ONE_DENSE_GPU_CHANNEL_ALIGNMENT_ELEMENTS;
  readonly channelAlignmentBytes: number;
  readonly channelStrideElements: number;
  readonly channelStrideBytes: number;
  readonly hierarchy: FactorOneDenseHierarchyPlan;
  readonly channels: readonly FactorOneDenseGpuChannelRegion[];
  readonly spectralUpper: FactorOneDenseGpuPhysicalRegion;
  readonly candidateSpectralUpper: FactorOneDenseGpuPhysicalRegion;
  readonly acceptedOccupiedIndices: FactorOneDenseGpuPhysicalRegion;
  readonly candidateOccupiedIndices: FactorOneDenseGpuPhysicalRegion;
  readonly acceptedOccupiedCounts: FactorOneDenseGpuPhysicalRegion;
  readonly candidateOccupiedCounts: FactorOneDenseGpuPhysicalRegion;
  readonly publicationControl: FactorOneDenseGpuPhysicalRegion;
  readonly worklistLevels: readonly FactorOneDenseGpuWorklistLevel[];
  readonly regions: readonly FactorOneDenseGpuPhysicalRegion[];
  readonly payloadBytes: number;
  readonly paddingBytes: number;
  readonly totalBytes: number;
}

const MAXIMUM_LEVELS = 12;
const MAXIMUM_ENCODED_OWNER = 0xffff_fffe;

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`${label} must be a positive integer`);
  }
  return value;
}

function checkedVolume(dimensions: FactorOneDenseCoordinate): number {
  const volume = dimensions[0] * dimensions[1] * dimensions[2];
  if (!Number.isSafeInteger(volume) || volume < 1) {
    throw new RangeError("Factor-1 dense hierarchy volume exceeds exact integer addressing");
  }
  return volume;
}

function alignElements(elements: number): number {
  return Math.ceil(elements / FACTOR_ONE_DENSE_GPU_CHANNEL_ALIGNMENT_ELEMENTS)
    * FACTOR_ONE_DENSE_GPU_CHANNEL_ALIGNMENT_ELEMENTS;
}

/**
 * Plan one accepted/candidate dense image, correction vectors, spectral
 * bounds, and two capacity-sized occupied-index publications in a single
 * 256-byte-aligned arena. Logical channels retain the semantic global slot.
 */
export function planFactorOneDenseGpuPhysicalImage(
  hierarchyPlan: FactorOneDenseHierarchyPlan,
): FactorOneDenseGpuPhysicalPlan {
  const alignmentElements = FACTOR_ONE_DENSE_GPU_CHANNEL_ALIGNMENT_ELEMENTS;
  const scalarBytes = FACTOR_ONE_DENSE_GPU_SCALAR_BYTES;
  const channelAlignmentBytes = alignmentElements * scalarBytes;
  const channelStrideElements = alignElements(hierarchyPlan.totalSlots);
  const channelStrideBytes = channelStrideElements * scalarBytes;
  const regions: FactorOneDenseGpuPhysicalRegion[] = [];
  let cursorBytes = 0;
  const append = (
    name: string,
    scalar: "u32" | "f32",
    payloadElements: number,
    allocatedElements = alignElements(payloadElements),
  ): FactorOneDenseGpuPhysicalRegion => {
    if (cursorBytes % channelAlignmentBytes !== 0) {
      throw new RangeError("Factor-1 dense GPU physical region lost channel alignment");
    }
    const region = Object.freeze({
      name,
      scalar,
      offsetBytes: cursorBytes,
      payloadElements,
      allocatedElements,
      payloadBytes: payloadElements * scalarBytes,
      allocatedBytes: allocatedElements * scalarBytes,
    });
    regions.push(region);
    cursorBytes += region.allocatedBytes;
    return region;
  };

  const channels: FactorOneDenseGpuChannelRegion[] =
    FACTOR_ONE_DENSE_GPU_CHANNEL_NAMES.map((name): FactorOneDenseGpuChannelRegion =>
      Object.freeze({
      ...append(
        name,
        name.endsWith("Flags") || name.endsWith("Owners") ? "u32" : "f32",
        hierarchyPlan.totalSlots,
        channelStrideElements,
      ),
      name,
      directAddressed: true as const,
    }));
  const spectralUpper = append("spectralUpper", "f32", hierarchyPlan.levelCount);
  const candidateSpectralUpper = append(
    "candidateSpectralUpper", "f32", hierarchyPlan.levelCount,
  );
  const acceptedOccupiedIndices = append(
    "acceptedOccupiedIndices", "u32", hierarchyPlan.totalSlots, channelStrideElements,
  );
  const candidateOccupiedIndices = append(
    "candidateOccupiedIndices", "u32", hierarchyPlan.totalSlots, channelStrideElements,
  );
  const acceptedOccupiedCounts = append(
    "acceptedOccupiedCounts", "u32",
    hierarchyPlan.levelCount * FACTOR_ONE_DENSE_GPU_WORKLIST_RECORD_WORDS,
  );
  const candidateOccupiedCounts = append(
    "candidateOccupiedCounts", "u32",
    hierarchyPlan.levelCount * FACTOR_ONE_DENSE_GPU_WORKLIST_RECORD_WORDS,
  );
  const publicationControl = append("publicationControl", "u32", 8);
  const worklistLevels = hierarchyPlan.levelVolumes.map((capacityElements, level) =>
    Object.freeze({
      level,
      capacityElements,
      acceptedIndicesOffsetBytes:
        acceptedOccupiedIndices.offsetBytes + hierarchyPlan.levelBases[level]! * scalarBytes,
      candidateIndicesOffsetBytes:
        candidateOccupiedIndices.offsetBytes + hierarchyPlan.levelBases[level]! * scalarBytes,
      acceptedCountOffsetBytes: acceptedOccupiedCounts.offsetBytes
        + level * FACTOR_ONE_DENSE_GPU_WORKLIST_RECORD_WORDS * scalarBytes,
      candidateCountOffsetBytes: candidateOccupiedCounts.offsetBytes
        + level * FACTOR_ONE_DENSE_GPU_WORKLIST_RECORD_WORDS * scalarBytes,
      acceptedDispatchOffsetBytes: acceptedOccupiedCounts.offsetBytes
        + (level * FACTOR_ONE_DENSE_GPU_WORKLIST_RECORD_WORDS + 1) * scalarBytes,
      candidateDispatchOffsetBytes: candidateOccupiedCounts.offsetBytes
        + (level * FACTOR_ONE_DENSE_GPU_WORKLIST_RECORD_WORDS + 1) * scalarBytes,
    }));
  const payloadBytes = regions.reduce((sum, region) => sum + region.payloadBytes, 0);
  return Object.freeze({
    version: FACTOR_ONE_DENSE_GPU_PHYSICAL_PLAN_VERSION,
    stability: "prototype-not-frozen",
    storageOrder: "x-fast-soa",
    scalarBytes,
    channelAlignmentElements: alignmentElements,
    channelAlignmentBytes,
    channelStrideElements,
    channelStrideBytes,
    hierarchy: hierarchyPlan,
    channels: Object.freeze(channels),
    spectralUpper,
    candidateSpectralUpper,
    acceptedOccupiedIndices,
    candidateOccupiedIndices,
    acceptedOccupiedCounts,
    candidateOccupiedCounts,
    publicationControl,
    worklistLevels: Object.freeze(worklistLevels),
    regions: Object.freeze(regions),
    payloadBytes,
    paddingBytes: cursorBytes - payloadBytes,
    totalBytes: cursorBytes,
  });
}

/** Exact ceil-halving pyramid terminating at one cell. */
export function planFactorOneDensePressureHierarchy(
  dimensions: FactorOneDenseCoordinate,
): FactorOneDenseHierarchyPlan {
  const root = dimensions.map((value) =>
    positiveInteger(value, "Factor-1 dense hierarchy dimension")) as [number, number, number];
  const levelDimensions: Array<FactorOneDenseCoordinate> = [];
  const levelBases: number[] = [];
  const levelVolumes: number[] = [];
  let current: [number, number, number] = root;
  let totalSlots = 0;
  for (;;) {
    const stored = Object.freeze([...current]) as unknown as FactorOneDenseCoordinate;
    const volume = checkedVolume(stored);
    levelDimensions.push(stored);
    levelBases.push(totalSlots);
    levelVolumes.push(volume);
    totalSlots += volume;
    if (!Number.isSafeInteger(totalSlots)) {
      throw new RangeError("Factor-1 dense hierarchy arena exceeds exact integer addressing");
    }
    if (current[0] === 1 && current[1] === 1 && current[2] === 1) break;
    if (levelDimensions.length >= MAXIMUM_LEVELS) {
      throw new RangeError(`Factor-1 dense hierarchy exceeds ${MAXIMUM_LEVELS} levels`);
    }
    current = current.map((value) => Math.ceil(value / 2)) as [number, number, number];
  }
  return Object.freeze({
    dimensions: Object.freeze([...root]) as unknown as FactorOneDenseCoordinate,
    levelDimensions: Object.freeze(levelDimensions),
    levelBases: Object.freeze(levelBases),
    levelVolumes: Object.freeze(levelVolumes),
    levelCount: levelDimensions.length,
    totalSlots,
  });
}

function checkedLevel(plan: FactorOneDenseHierarchyPlan, level: number): number {
  if (!Number.isSafeInteger(level) || level < 0 || level >= plan.levelCount) {
    throw new RangeError("Factor-1 dense hierarchy level is out of range");
  }
  return level;
}

function checkedCoordinate(
  dimensions: FactorOneDenseCoordinate,
  coordinate: FactorOneDenseCoordinate,
): void {
  if (coordinate.some((value, axis) =>
    !Number.isSafeInteger(value) || value < 0 || value >= dimensions[axis])) {
    throw new RangeError("Factor-1 dense hierarchy coordinate is out of range");
  }
}

/** Global x-fast arena slot for one level coordinate. */
export function factorOneDenseSlot(
  plan: FactorOneDenseHierarchyPlan,
  level: number,
  coordinate: FactorOneDenseCoordinate,
): number {
  checkedLevel(plan, level);
  const dimensions = plan.levelDimensions[level]!;
  checkedCoordinate(dimensions, coordinate);
  return plan.levelBases[level]!
    + coordinate[0] + dimensions[0] * (coordinate[1] + dimensions[1] * coordinate[2]);
}

/** Inverse of factorOneDenseSlot for a slot known to belong to `level`. */
export function factorOneDenseCoordinate(
  plan: FactorOneDenseHierarchyPlan,
  level: number,
  slot: number,
): FactorOneDenseCoordinate {
  checkedLevel(plan, level);
  const base = plan.levelBases[level]!, volume = plan.levelVolumes[level]!;
  if (!Number.isSafeInteger(slot) || slot < base || slot >= base + volume) {
    throw new RangeError("Factor-1 dense hierarchy slot is outside the level");
  }
  const dimensions = plan.levelDimensions[level]!;
  const local = slot - base;
  const yz = Math.floor(local / dimensions[0]);
  return Object.freeze([
    local - yz * dimensions[0],
    yz % dimensions[1],
    Math.floor(yz / dimensions[1]),
  ]) as FactorOneDenseCoordinate;
}

function rolePriority(role: FactorOneDenseOccupiedRole): number {
  if (role === FACTOR_ONE_DENSE_CELL_ROLE.active) return 3;
  if (role === FACTOR_ONE_DENSE_CELL_ROLE.ghost) return 2;
  if (role === FACTOR_ONE_DENSE_CELL_ROLE.multigridOnly) return 1;
  throw new RangeError("Invalid factor-1 dense hierarchy role");
}

function encodedOwner(seed: FactorOneDenseHierarchySeed): number {
  const owned = seed.role === FACTOR_ONE_DENSE_CELL_ROLE.active
    || seed.role === FACTOR_ONE_DENSE_CELL_ROLE.ghost;
  if (!owned) {
    if (seed.owner !== undefined) {
      throw new RangeError("Multigrid-only factor-1 dense cells cannot carry an owner");
    }
    return 0;
  }
  if (!Number.isSafeInteger(seed.owner) || seed.owner! < 0 || seed.owner! > MAXIMUM_ENCODED_OWNER) {
    throw new RangeError("Active and ghost factor-1 dense cells require a valid zero-based owner");
  }
  return seed.owner! + 1;
}

/**
 * Build role/owner metadata and close every occupied cell through its unique
 * q/2 aggregate parent. Seed order cannot affect the result.
 */
export function buildFactorOneDensePressureHierarchy(
  dimensions: FactorOneDenseCoordinate,
  seeds: readonly FactorOneDenseHierarchySeed[],
): FactorOneDenseHierarchy {
  if (seeds.length === 0) throw new RangeError("Factor-1 dense hierarchy requires at least one occupied seed");
  const plan = planFactorOneDensePressureHierarchy(dimensions);
  const flags = new Uint32Array(plan.totalSlots);
  const owners = new Uint32Array(plan.totalSlots);
  const publish = (seed: FactorOneDenseHierarchySeed) => {
    const slot = factorOneDenseSlot(plan, seed.level, seed.coordinate);
    const incomingOwner = encodedOwner(seed);
    const oldOwner = owners[slot]!;
    if (oldOwner !== 0 && incomingOwner !== 0 && oldOwner !== incomingOwner) {
      throw new RangeError("Factor-1 dense hierarchy coordinate has conflicting owners");
    }
    const oldRole = flags[slot] as FactorOneDenseOccupiedRole | 0;
    if (oldRole === 0 || rolePriority(seed.role) > rolePriority(oldRole)) flags[slot] = seed.role;
    if (incomingOwner !== 0) owners[slot] = incomingOwner;
  };
  for (const seed of seeds) publish(seed);
  for (let level = 0; level + 1 < plan.levelCount; level += 1) {
    const base = plan.levelBases[level]!, end = base + plan.levelVolumes[level]!;
    for (let slot = base; slot < end; slot += 1) if (flags[slot] !== 0) {
      const coordinate = factorOneDenseCoordinate(plan, level, slot);
      publish({
        level: level + 1,
        coordinate: coordinate.map((value) => Math.floor(value / 2)) as [number, number, number],
        role: FACTOR_ONE_DENSE_CELL_ROLE.multigridOnly,
      });
    }
  }
  const occupiedLocalIndices = Object.freeze(plan.levelVolumes.map((volume, level) => {
    const base = plan.levelBases[level]!;
    const occupied: number[] = [];
    for (let local = 0; local < volume; local += 1) {
      if (flags[base + local] !== 0) occupied.push(local);
    }
    return Uint32Array.from(occupied);
  }));
  const hierarchy = Object.freeze({ plan, flags, owners, occupiedLocalIndices });
  validateFactorOneDensePressureHierarchy(hierarchy);
  return hierarchy;
}

/** Fail-closed validation suitable for a later shadow GPU readback. */
export function validateFactorOneDensePressureHierarchy(
  hierarchy: FactorOneDenseHierarchy,
): void {
  const { plan, flags, owners, occupiedLocalIndices } = hierarchy;
  if (flags.length !== plan.totalSlots || owners.length !== plan.totalSlots) {
    throw new RangeError("Factor-1 dense hierarchy metadata has the wrong length");
  }
  if (occupiedLocalIndices.length !== plan.levelCount) {
    throw new RangeError("Factor-1 dense hierarchy occupied worklists have the wrong level count");
  }
  for (let slot = 0; slot < plan.totalSlots; slot += 1) {
    const role = flags[slot]!;
    const validRole = role === FACTOR_ONE_DENSE_CELL_ROLE.absent
      || role === FACTOR_ONE_DENSE_CELL_ROLE.active
      || role === FACTOR_ONE_DENSE_CELL_ROLE.ghost
      || role === FACTOR_ONE_DENSE_CELL_ROLE.multigridOnly;
    if (!validRole) throw new RangeError("Factor-1 dense hierarchy cell has an invalid role");
    const owned = role === FACTOR_ONE_DENSE_CELL_ROLE.active
      || role === FACTOR_ONE_DENSE_CELL_ROLE.ghost;
    if (owned !== (owners[slot] !== 0)) {
      throw new RangeError("Factor-1 dense hierarchy role and owner disagree");
    }
  }
  for (let level = 0; level < plan.levelCount; level += 1) {
    const base = plan.levelBases[level]!, volume = plan.levelVolumes[level]!;
    const worklist = occupiedLocalIndices[level]!;
    let previous = -1;
    for (const local of worklist) {
      if (local >= volume || local <= previous || flags[base + local] === 0) {
        throw new RangeError("Factor-1 dense hierarchy occupied worklist is invalid");
      }
      previous = local;
    }
    let cursor = 0;
    for (let local = 0; local < volume; local += 1) {
      const listed = cursor < worklist.length && worklist[cursor] === local;
      if (listed) cursor += 1;
      if (listed !== (flags[base + local] !== 0)) {
        throw new RangeError("Factor-1 dense hierarchy occupied worklist is incomplete");
      }
    }
  }
  for (let level = 0; level + 1 < plan.levelCount; level += 1) {
    const base = plan.levelBases[level]!, end = base + plan.levelVolumes[level]!;
    for (let slot = base; slot < end; slot += 1) {
      if (flags[slot] !== 0 && flags[factorOneDenseAggregateParent(plan, level, slot)] === 0) {
        throw new RangeError("Occupied factor-1 dense cell has no aggregate parent");
      }
    }
  }
  const bottom = plan.levelCount - 1;
  const bottomBase = plan.levelBases[bottom]!, bottomVolume = plan.levelVolumes[bottom]!;
  let occupiedBottom = 0;
  for (let slot = bottomBase; slot < bottomBase + bottomVolume; slot += 1) {
    occupiedBottom += Number(flags[slot] !== 0);
  }
  if (bottomVolume !== 1 || occupiedBottom !== 1) {
    throw new RangeError("Factor-1 dense hierarchy must have one occupied bottom cell");
  }
}

/** Unique unit-weight aggregate parent of an occupied or absent fine slot. */
export function factorOneDenseAggregateParent(
  plan: FactorOneDenseHierarchyPlan,
  fineLevel: number,
  fineSlot: number,
): number {
  checkedLevel(plan, fineLevel);
  if (fineLevel + 1 >= plan.levelCount) {
    throw new RangeError("Bottom factor-1 dense hierarchy level has no parent");
  }
  const coordinate = factorOneDenseCoordinate(plan, fineLevel, fineSlot);
  return factorOneDenseSlot(plan, fineLevel + 1,
    coordinate.map((value) => Math.floor(value / 2)) as [number, number, number]);
}

/**
 * In-bounds child slots in the fixed restriction order: x bit changes
 * fastest, then y, then z.
 */
export function factorOneDenseAggregateChildren(
  plan: FactorOneDenseHierarchyPlan,
  fineLevel: number,
  parentSlot: number,
): readonly number[] {
  checkedLevel(plan, fineLevel);
  if (fineLevel + 1 >= plan.levelCount) {
    throw new RangeError("Bottom factor-1 dense hierarchy level has no children");
  }
  const parent = factorOneDenseCoordinate(plan, fineLevel + 1, parentSlot);
  const fineDimensions = plan.levelDimensions[fineLevel]!;
  const children: number[] = [];
  for (let z = 0; z < 2; z += 1) for (let y = 0; y < 2; y += 1) for (let x = 0; x < 2; x += 1) {
    const child: [number, number, number] = [
      2 * parent[0] + x, 2 * parent[1] + y, 2 * parent[2] + z,
    ];
    if (child.every((value, axis) => value < fineDimensions[axis])) {
      children.push(factorOneDenseSlot(plan, fineLevel, child));
    }
  }
  return Object.freeze(children);
}

function requireLevelVector(
  plan: FactorOneDenseHierarchyPlan,
  level: number,
  values: ArrayLike<number>,
  label: string,
): void {
  checkedLevel(plan, level);
  if (values.length !== plan.levelVolumes[level]) {
    throw new RangeError(`${label} length disagrees with the factor-1 dense level`);
  }
}

/** Parent-owned, deterministic P^T for the unit 2x2x2 aggregate mapping. */
export function restrictFactorOneDenseAggregate(
  hierarchy: FactorOneDenseHierarchy,
  fineLevel: number,
  fineValues: ArrayLike<number>,
): Float64Array {
  const { plan, flags } = hierarchy;
  if (fineLevel + 1 >= plan.levelCount) throw new RangeError("Cannot restrict the bottom hierarchy level");
  requireLevelVector(plan, fineLevel, fineValues, "Fine vector");
  const coarseLevel = fineLevel + 1;
  const result = new Float64Array(plan.levelVolumes[coarseLevel]);
  const fineBase = plan.levelBases[fineLevel]!, coarseBase = plan.levelBases[coarseLevel]!;
  for (let coarse = 0; coarse < result.length; coarse += 1) {
    const parentSlot = coarseBase + coarse;
    if (flags[parentSlot] === 0) continue;
    let sum = 0;
    for (const childSlot of factorOneDenseAggregateChildren(plan, fineLevel, parentSlot)) {
      if (flags[childSlot] !== 0) sum += fineValues[childSlot - fineBase]!;
    }
    result[coarse] = sum;
  }
  return result;
}

/** Unit P matching restrictFactorOneDenseAggregate exactly. */
export function prolongFactorOneDenseAggregate(
  hierarchy: FactorOneDenseHierarchy,
  fineLevel: number,
  coarseValues: ArrayLike<number>,
): Float64Array {
  const { plan, flags } = hierarchy;
  if (fineLevel + 1 >= plan.levelCount) throw new RangeError("Cannot prolong into the bottom hierarchy level");
  const coarseLevel = fineLevel + 1;
  requireLevelVector(plan, coarseLevel, coarseValues, "Coarse vector");
  const result = new Float64Array(plan.levelVolumes[fineLevel]);
  const fineBase = plan.levelBases[fineLevel]!, coarseBase = plan.levelBases[coarseLevel]!;
  for (let fine = 0; fine < result.length; fine += 1) {
    const fineSlot = fineBase + fine;
    if (flags[fineSlot] === 0) continue;
    const parent = factorOneDenseAggregateParent(plan, fineLevel, fineSlot);
    result[fine] = coarseValues[parent - coarseBase]!;
  }
  return result;
}

const SIX_FACE_DIRECTIONS: readonly FactorOneDenseCoordinate[] = Object.freeze([
  Object.freeze([1, 0, 0]), Object.freeze([-1, 0, 0]),
  Object.freeze([0, 1, 0]), Object.freeze([0, -1, 0]),
  Object.freeze([0, 0, 1]), Object.freeze([0, 0, -1]),
] as FactorOneDenseCoordinate[]);

/**
 * Matrix-free first-order M1 apply. Every present face uses the same
 * per-level coefficient; absent and out-of-domain neighbours contribute no
 * off-diagonal term.
 */
export function applyFactorOneDenseSixFace(
  hierarchy: FactorOneDenseHierarchy,
  level: number,
  input: ArrayLike<number>,
  diagonal: ArrayLike<number>,
  faceCoefficient: number,
): Float64Array {
  const { plan, flags } = hierarchy;
  requireLevelVector(plan, level, input, "Input vector");
  requireLevelVector(plan, level, diagonal, "Diagonal vector");
  if (!(faceCoefficient >= 0) || !Number.isFinite(faceCoefficient)) {
    throw new RangeError("Factor-1 dense face coefficient must be finite and non-negative");
  }
  const result = new Float64Array(plan.levelVolumes[level]);
  const base = plan.levelBases[level]!, dimensions = plan.levelDimensions[level]!;
  for (let local = 0; local < result.length; local += 1) {
    const slot = base + local;
    if (flags[slot] === 0) continue;
    const source = input[local]!, d = diagonal[local]!;
    if (!Number.isFinite(source) || !(d > 0) || !Number.isFinite(d)) {
      throw new RangeError("Factor-1 dense operator requires finite input and positive diagonal");
    }
    const coordinate = factorOneDenseCoordinate(plan, level, slot);
    let value = d * source;
    for (const direction of SIX_FACE_DIRECTIONS) {
      const neighbour: [number, number, number] = [
        coordinate[0] + direction[0],
        coordinate[1] + direction[1],
        coordinate[2] + direction[2],
      ];
      if (neighbour.some((component, axis) => component < 0 || component >= dimensions[axis])) continue;
      const neighbourSlot = factorOneDenseSlot(plan, level, neighbour);
      if (flags[neighbourSlot] !== 0) {
        const other = input[neighbourSlot - base]!;
        if (!Number.isFinite(other)) {
          throw new RangeError("Factor-1 dense operator requires finite occupied neighbour values");
        }
        value -= faceCoefficient * other;
      }
    }
    result[local] = value;
  }
  return result;
}
