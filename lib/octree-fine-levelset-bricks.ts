/**
 * CPU authority/oracle for the domain-global sparse fine level-set lattice.
 *
 * The fine lattice deliberately has no octree row in its address.  Octree
 * topology may be rebuilt while a brick key and its samples remain stable.
 * This follows the two-mesh construction in Aanjaneya et al. section 5: the
 * fine SPGrid stores phi only; velocity and pressure remain on the octree.
 */

export const FINE_LEVELSET_INVALID = 0xffff_ffff;
export const FINE_LEVELSET_CHANNELS = 4;
export const FINE_LEVELSET_NEIGHBOR_COUNT = 6;
export const FINE_LEVELSET_MAX_HASH_PROBES = 32;

export const FINE_LEVELSET_SAMPLE_FLAGS = Object.freeze({
  valid: 1 << 0,
  interface: 1 << 1,
  known: 1 << 2,
  trial: 1 << 3,
  negative: 1 << 4,
} as const);

export type FineLevelSetFactor = 4 | 8;
export type FineLevelSetBrickResolution = 4 | 8;
export type FineLevelSetVec3 = readonly [number, number, number];

export interface FineLevelSetBrickPlanOptions {
  domainOrigin: FineLevelSetVec3;
  /** Number of finest-effective octree cells along each domain axis. */
  finestCellDimensions: FineLevelSetVec3;
  finestCellWidth: number;
  fineFactor: FineLevelSetFactor;
  brickResolution: FineLevelSetBrickResolution;
  maximumResidentBricks: number;
  maximumHashLoad?: number;
  maximumHashProbes?: number;
}

export interface FineLevelSetBrickPlan {
  domainOrigin: FineLevelSetVec3;
  finestCellDimensions: FineLevelSetVec3;
  finestCellWidth: number;
  fineFactor: FineLevelSetFactor;
  fineCellWidth: number;
  brickResolution: FineLevelSetBrickResolution;
  sampleDimensions: FineLevelSetVec3;
  brickDimensions: FineLevelSetVec3;
  logicalBrickCount: number;
  maximumResidentBricks: number;
  maximumHashLoad: number;
  maximumHashProbes: number;
  samplesPerBrick: number;
  payloadBytesPerBrick: number;
  hashCapacity: number;
  payloadCapacityBytes: number;
  metadataCapacityBytes: number;
  pageTableBytes: number;
  worklistBytes: number;
  allocatedBytes: number;
}

export interface FineLevelSetBrickMemory {
  residentBricks: number;
  activePayloadBytes: number;
  payloadCapacityBytes: number;
  metadataBytes: number;
  pageTableBytes: number;
  worklistBytes: number;
  fragmentationBytes: number;
  allocatedBytes: number;
}

export interface FineLevelSetBrickPage {
  physicalId: number;
  key: number;
  generation: number;
  flags: Uint32Array;
  phi: Float32Array;
  workA: Float32Array;
  workB: Float32Array;
  /** Physical IDs in -X,+X,-Y,+Y,-Z,+Z order. */
  neighborIds: Uint32Array;
}

export interface FineLevelSetPublication {
  generation: number;
  interfaceKeys: Uint32Array;
  desiredKeys: Uint32Array;
  activePhysicalIds: Uint32Array;
  reusedPages: number;
  activatedPages: number;
  retiredPages: number;
}

export interface FineLevelSetGPUGenerationData {
  generation: number;
  activeCount: number;
  /** key, physical ID pairs; empty slots contain INVALID, INVALID. */
  hashPairs: Uint32Array;
  /** physical ID, key, generation, flags, then six cached neighbors. */
  metadataWords: Uint32Array;
  /** count, generation, dispatch x/y/z, then compact physical IDs. */
  worklistWords: Uint32Array;
  flags: Uint32Array;
  phi: Float32Array;
  workA: Float32Array;
  workB: Float32Array;
}

export type FineLevelSetCoarsePhi = (position: FineLevelSetVec3) => number;

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}

function finiteVec3(value: FineLevelSetVec3, label: string): FineLevelSetVec3 {
  if (value.length !== 3 || value.some((entry) => !Number.isFinite(entry))) {
    throw new RangeError(`${label} must contain three finite values`);
  }
  return [value[0], value[1], value[2]];
}

function nextPowerOfTwo(value: number): number {
  let result = 1;
  while (result < value) result *= 2;
  return result;
}

export function planFineLevelSetBricks(options: FineLevelSetBrickPlanOptions): FineLevelSetBrickPlan {
  const domainOrigin = finiteVec3(options.domainOrigin, "Fine level-set domain origin");
  const finestCellDimensions = options.finestCellDimensions.map((value, axis) =>
    positiveInteger(value, `Fine level-set finest dimension ${axis}`)) as unknown as FineLevelSetVec3;
  if (!Number.isFinite(options.finestCellWidth) || options.finestCellWidth <= 0) {
    throw new RangeError("Fine level-set finest cell width must be finite and positive");
  }
  if (options.fineFactor !== 4 && options.fineFactor !== 8) {
    throw new RangeError("Fine level-set factor must be 4 or 8");
  }
  if (options.brickResolution !== 4 && options.brickResolution !== 8) {
    throw new RangeError("Fine level-set brick resolution must be 4 or 8");
  }
  const maximumResidentBricks = positiveInteger(options.maximumResidentBricks, "Fine level-set resident capacity");
  const maximumHashLoad = options.maximumHashLoad ?? 0.5;
  if (!Number.isFinite(maximumHashLoad) || maximumHashLoad <= 0 || maximumHashLoad > 0.5) {
    throw new RangeError("Fine level-set maximum hash load must be in (0, 0.5]");
  }
  const maximumHashProbes = options.maximumHashProbes ?? FINE_LEVELSET_MAX_HASH_PROBES;
  if (!Number.isSafeInteger(maximumHashProbes) || maximumHashProbes < 1 || maximumHashProbes > FINE_LEVELSET_MAX_HASH_PROBES) {
    throw new RangeError(`Fine level-set maximum hash probes must be in [1, ${FINE_LEVELSET_MAX_HASH_PROBES}]`);
  }
  const sampleDimensions = finestCellDimensions.map((value) => value * options.fineFactor) as unknown as FineLevelSetVec3;
  if (sampleDimensions.some((value) => !Number.isSafeInteger(value))) {
    throw new RangeError("Fine level-set sample dimensions exceed exact integer range");
  }
  const brickDimensions = sampleDimensions.map((value) => Math.ceil(value / options.brickResolution)) as unknown as FineLevelSetVec3;
  const logicalBrickCount = brickDimensions[0] * brickDimensions[1] * brickDimensions[2];
  // INVALID is reserved by both CPU and WGSL hash tables.
  if (!Number.isSafeInteger(logicalBrickCount) || logicalBrickCount > FINE_LEVELSET_INVALID) {
    throw new RangeError("Fine level-set brick coordinates do not fit the configured 32-bit key");
  }
  if (maximumResidentBricks > logicalBrickCount) {
    throw new RangeError("Fine level-set resident capacity exceeds the logical brick domain");
  }
  const samplesPerBrick = options.brickResolution ** 3;
  const payloadBytesPerBrick = samplesPerBrick * FINE_LEVELSET_CHANNELS * 4;
  const hashCapacity = nextPowerOfTwo(Math.ceil(maximumResidentBricks / maximumHashLoad));
  if (hashCapacity > 0x8000_0000) throw new RangeError("Fine level-set hash capacity exceeds WebGPU indexing range");
  const payloadCapacityBytes = maximumResidentBricks * payloadBytesPerBrick;
  // Two generation-specific metadata copies: id/key/generation/state + six neighbors.
  const metadataCapacityBytes = 2 * maximumResidentBricks * 10 * 4;
  const pageTableBytes = 2 * hashCapacity * 2 * 4;
  // count/generation/dispatch xyz followed by physical IDs, for both generations.
  const worklistBytes = 2 * (5 + maximumResidentBricks) * 4;
  const allocatedBytes = payloadCapacityBytes + metadataCapacityBytes + pageTableBytes + worklistBytes;
  return {
    domainOrigin, finestCellDimensions, finestCellWidth: options.finestCellWidth,
    fineFactor: options.fineFactor, fineCellWidth: options.finestCellWidth / options.fineFactor,
    brickResolution: options.brickResolution, sampleDimensions, brickDimensions, logicalBrickCount,
    maximumResidentBricks, maximumHashLoad, maximumHashProbes, samplesPerBrick, payloadBytesPerBrick,
    hashCapacity, payloadCapacityBytes, metadataCapacityBytes, pageTableBytes, worklistBytes, allocatedBytes,
  };
}

export function packFineLevelSetBrickKey(plan: FineLevelSetBrickPlan, coord: FineLevelSetVec3): number {
  for (let axis = 0; axis < 3; axis += 1) {
    if (!Number.isSafeInteger(coord[axis]) || coord[axis] < 0 || coord[axis] >= plan.brickDimensions[axis]) {
      throw new RangeError(`Fine level-set brick coordinate ${axis} is outside the configured domain`);
    }
  }
  const key = coord[0] + plan.brickDimensions[0] * (coord[1] + plan.brickDimensions[1] * coord[2]);
  if (!Number.isSafeInteger(key) || key >= FINE_LEVELSET_INVALID) {
    throw new RangeError("Fine level-set brick key cannot be represented by the 32-bit ABI");
  }
  return key >>> 0;
}

export function unpackFineLevelSetBrickKey(plan: FineLevelSetBrickPlan, key: number): [number, number, number] {
  if (!Number.isSafeInteger(key) || key < 0 || key >= plan.logicalBrickCount || key === FINE_LEVELSET_INVALID) {
    throw new RangeError("Fine level-set brick key is outside the configured domain");
  }
  const xy = plan.brickDimensions[0] * plan.brickDimensions[1];
  const z = Math.floor(key / xy);
  const remainder = key - z * xy;
  const y = Math.floor(remainder / plan.brickDimensions[0]);
  return [remainder - y * plan.brickDimensions[0], y, z];
}

export interface FineLevelSetAddress {
  fineCoord: [number, number, number];
  brickCoord: [number, number, number];
  localCoord: [number, number, number];
  brickKey: number;
  localIndex: number;
}

export function fineLevelSetAddressAtPosition(
  plan: FineLevelSetBrickPlan,
  position: FineLevelSetVec3,
): FineLevelSetAddress | undefined {
  const fineCoord = position.map((value, axis) => Math.floor((value - plan.domainOrigin[axis]) / plan.fineCellWidth)) as [number, number, number];
  if (fineCoord.some((value, axis) => value < 0 || value >= plan.sampleDimensions[axis])) return undefined;
  return fineLevelSetAddressAtCoordinate(plan, fineCoord);
}

export function fineLevelSetAddressAtCoordinate(
  plan: FineLevelSetBrickPlan,
  fineCoord: FineLevelSetVec3,
): FineLevelSetAddress {
  for (let axis = 0; axis < 3; axis += 1) {
    if (!Number.isSafeInteger(fineCoord[axis]) || fineCoord[axis] < 0 || fineCoord[axis] >= plan.sampleDimensions[axis]) {
      throw new RangeError(`Fine level-set sample coordinate ${axis} is outside the configured domain`);
    }
  }
  const brickCoord = fineCoord.map((value) => Math.floor(value / plan.brickResolution)) as [number, number, number];
  const localCoord = fineCoord.map((value, axis) => value - brickCoord[axis] * plan.brickResolution) as [number, number, number];
  const localIndex = localCoord[0] + plan.brickResolution * (localCoord[1] + plan.brickResolution * localCoord[2]);
  return { fineCoord: [...fineCoord] as [number, number, number], brickCoord, localCoord,
    brickKey: packFineLevelSetBrickKey(plan, brickCoord), localIndex };
}

function hashKey(key: number, mask: number): number {
  return Math.imul(key ^ (key >>> 16), 0x9e37_79b1) >>> 0 & mask;
}

class BoundedBrickHash {
  readonly pairs: Uint32Array;

  constructor(private readonly plan: FineLevelSetBrickPlan) {
    this.pairs = new Uint32Array(plan.hashCapacity * 2);
    this.pairs.fill(FINE_LEVELSET_INVALID);
  }

  insert(key: number, physicalId: number): void {
    const mask = this.plan.hashCapacity - 1;
    const start = hashKey(key, mask);
    for (let probe = 0; probe < this.plan.maximumHashProbes; probe += 1) {
      const slot = (start + probe) & mask;
      const prior = this.pairs[slot * 2];
      if (prior === FINE_LEVELSET_INVALID || prior === key) {
        this.pairs[slot * 2] = key;
        this.pairs[slot * 2 + 1] = physicalId;
        return;
      }
    }
    throw new RangeError(`Fine level-set hash exceeded ${this.plan.maximumHashProbes} probes`);
  }

  lookup(key: number): number {
    const mask = this.plan.hashCapacity - 1;
    const start = hashKey(key, mask);
    for (let probe = 0; probe < this.plan.maximumHashProbes; probe += 1) {
      const slot = (start + probe) & mask;
      const stored = this.pairs[slot * 2];
      if (stored === key) return this.pairs[slot * 2 + 1];
      if (stored === FINE_LEVELSET_INVALID) return FINE_LEVELSET_INVALID;
    }
    return FINE_LEVELSET_INVALID;
  }
}

const NEIGHBOR_DIRECTIONS = [
  [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1],
] as const;

function sortedUnique(keys: Iterable<number>): number[] {
  return [...new Set(keys)].sort((a, b) => a - b);
}

/** Reference implementation of two-generation interface-plus-ring publication. */
export class FineLevelSetBrickOracle {
  readonly plan: FineLevelSetBrickPlan;
  private pagesByKey = new Map<number, FineLevelSetBrickPage>();
  private pagesById: Array<FineLevelSetBrickPage | undefined>;
  private hash: BoundedBrickHash;
  private generationValue = 0;

  constructor(plan: FineLevelSetBrickPlan) {
    this.plan = plan;
    this.pagesById = new Array(plan.maximumResidentBricks);
    this.hash = new BoundedBrickHash(plan);
  }

  get generation(): number { return this.generationValue; }
  get residentBrickCount(): number { return this.pagesByKey.size; }

  residentPages(): readonly FineLevelSetBrickPage[] {
    return [...this.pagesByKey.values()].sort((a, b) => a.key - b.key);
  }

  pageForKey(key: number): FineLevelSetBrickPage | undefined {
    // Validate before lookup so malformed keys can never alias an empty slot.
    unpackFineLevelSetBrickKey(this.plan, key);
    const id = this.hash.lookup(key);
    return id === FINE_LEVELSET_INVALID ? undefined : this.pagesById[id];
  }

  private createPage(physicalId: number, key: number, generation: number, coarsePhi: FineLevelSetCoarsePhi): FineLevelSetBrickPage {
    const { samplesPerBrick } = this.plan;
    const page: FineLevelSetBrickPage = {
      physicalId, key, generation,
      flags: new Uint32Array(samplesPerBrick),
      phi: new Float32Array(samplesPerBrick),
      workA: new Float32Array(samplesPerBrick),
      workB: new Float32Array(samplesPerBrick),
      neighborIds: new Uint32Array(FINE_LEVELSET_NEIGHBOR_COUNT).fill(FINE_LEVELSET_INVALID),
    };
    const brick = unpackFineLevelSetBrickKey(this.plan, key);
    for (let z = 0; z < this.plan.brickResolution; z += 1) {
      for (let y = 0; y < this.plan.brickResolution; y += 1) {
        for (let x = 0; x < this.plan.brickResolution; x += 1) {
          const local = x + this.plan.brickResolution * (y + this.plan.brickResolution * z);
          const q: FineLevelSetVec3 = [
            brick[0] * this.plan.brickResolution + x,
            brick[1] * this.plan.brickResolution + y,
            brick[2] * this.plan.brickResolution + z,
          ];
          const position = q.map((value, axis) => this.plan.domainOrigin[axis] + (value + 0.5) * this.plan.fineCellWidth) as unknown as FineLevelSetVec3;
          const value = coarsePhi(position);
          if (!Number.isFinite(value)) throw new RangeError("Coarse phi fallback returned a non-finite value");
          page.phi[local] = value;
          page.workA[local] = value;
          page.workB[local] = value;
          page.flags[local] = FINE_LEVELSET_SAMPLE_FLAGS.valid
            | (value < 0 ? FINE_LEVELSET_SAMPLE_FLAGS.negative : 0);
        }
      }
    }
    return page;
  }

  publishInterfaceAndRing(
    interfaceKeysValue: Iterable<number>,
    coarsePhi: FineLevelSetCoarsePhi,
    ringWidth = 1,
  ): FineLevelSetPublication {
    if (!Number.isSafeInteger(ringWidth) || ringWidth < 1) throw new RangeError("Fine level-set ring width must be positive");
    const interfaceKeys = sortedUnique(interfaceKeysValue);
    interfaceKeys.forEach((key) => unpackFineLevelSetBrickKey(this.plan, key));
    let frontier = interfaceKeys;
    const desired = new Set(interfaceKeys);
    for (let ring = 0; ring < ringWidth; ring += 1) {
      const next: number[] = [];
      for (const key of frontier) {
        const coord = unpackFineLevelSetBrickKey(this.plan, key);
        for (const direction of NEIGHBOR_DIRECTIONS) {
          const neighbor = coord.map((value, axis) => value + direction[axis]) as [number, number, number];
          if (neighbor.some((value, axis) => value < 0 || value >= this.plan.brickDimensions[axis])) continue;
          const neighborKey = packFineLevelSetBrickKey(this.plan, neighbor);
          if (!desired.has(neighborKey)) { desired.add(neighborKey); next.push(neighborKey); }
        }
      }
      frontier = sortedUnique(next);
    }
    const desiredKeys = sortedUnique(desired);
    if (desiredKeys.length > this.plan.maximumResidentBricks) {
      throw new RangeError("Fine level-set desired interface band exceeds physical page capacity");
    }

    const nextGeneration = (this.generationValue + 1) >>> 0 || 1;
    const nextByKey = new Map<number, FineLevelSetBrickPage>();
    const oldOnlyIds = [...this.pagesByKey.entries()]
      .filter(([key]) => !desired.has(key)).map(([, page]) => page.physicalId).sort((a, b) => a - b);
    // Array.prototype.map skips holes; physical slots intentionally start as
    // holes, so enumerate the numeric capacity rather than the sparse array.
    const whollyFreeIds = Array.from({ length: this.pagesById.length }, (_, id) =>
      this.pagesById[id] ? -1 : id).filter((id) => id >= 0);
    const availableIds = [...oldOnlyIds, ...whollyFreeIds];
    let reusedPages = 0;
    let activatedPages = 0;
    for (const key of desiredKeys) {
      const existing = this.pagesByKey.get(key);
      let page: FineLevelSetBrickPage;
      if (existing) {
        reusedPages += 1;
        // Keep the currently published metadata immutable until the complete
        // next hash and all coarse initializations have succeeded.
        page = {
          ...existing,
          generation: nextGeneration,
          neighborIds: new Uint32Array(FINE_LEVELSET_NEIGHBOR_COUNT).fill(FINE_LEVELSET_INVALID),
        };
      } else {
        const physicalId = availableIds.shift();
        if (physicalId === undefined) throw new RangeError("Fine level-set physical free list was exhausted");
        page = this.createPage(physicalId, key, nextGeneration, coarsePhi);
        activatedPages += 1;
      }
      nextByKey.set(key, page);
    }

    const nextHash = new BoundedBrickHash(this.plan);
    for (const [key, page] of nextByKey) nextHash.insert(key, page.physicalId);
    for (const [key, page] of nextByKey) {
      const coord = unpackFineLevelSetBrickKey(this.plan, key);
      for (let directionIndex = 0; directionIndex < NEIGHBOR_DIRECTIONS.length; directionIndex += 1) {
        const direction = NEIGHBOR_DIRECTIONS[directionIndex];
        const neighbor = coord.map((value, axis) => value + direction[axis]) as [number, number, number];
        page.neighborIds[directionIndex] = neighbor.some((value, axis) => value < 0 || value >= this.plan.brickDimensions[axis])
          ? FINE_LEVELSET_INVALID
          : nextHash.lookup(packFineLevelSetBrickKey(this.plan, neighbor));
      }
    }
    this.pagesById.fill(undefined);
    for (const page of nextByKey.values()) this.pagesById[page.physicalId] = page;
    const retiredPages = this.pagesByKey.size - reusedPages;
    this.pagesByKey = nextByKey;
    this.hash = nextHash;
    this.generationValue = nextGeneration;
    const activePhysicalIds = Uint32Array.from([...nextByKey.values()].sort((a, b) => a.key - b.key).map((page) => page.physicalId));
    return {
      generation: nextGeneration, interfaceKeys: Uint32Array.from(interfaceKeys),
      desiredKeys: Uint32Array.from(desiredKeys), activePhysicalIds,
      reusedPages, activatedPages, retiredPages,
    };
  }

  /** Finds interface bricks only from resident samples; no logical-domain scan. */
  detectInterfaceKeys(): Uint32Array {
    const result = new Set<number>();
    for (const [key, page] of this.pagesByKey) {
      let negative = false;
      let nonnegative = false;
      for (let index = 0; index < page.phi.length; index += 1) {
        if ((page.flags[index] & FINE_LEVELSET_SAMPLE_FLAGS.valid) === 0) continue;
        if (page.phi[index] < 0) negative = true;
        else nonnegative = true;
        if (negative && nonnegative) { result.add(key); break; }
      }
      // A crossing can straddle a brick boundary even when both individual
      // bricks are single-signed. Check positive faces once and publish both
      // incident bricks. Cached IDs make this O(resident samples), not O(domain).
      for (const [axis, neighborSlot] of [[0, 1], [1, 3], [2, 5]] as const) {
        const neighborId = page.neighborIds[neighborSlot];
        if (neighborId === FINE_LEVELSET_INVALID) continue;
        const neighbor = this.pagesById[neighborId];
        if (!neighbor) continue;
        let crossing = false;
        for (let v = 0; v < this.plan.brickResolution && !crossing; v += 1) {
          for (let u = 0; u < this.plan.brickResolution; u += 1) {
            const high: [number, number, number] = axis === 0
              ? [this.plan.brickResolution - 1, u, v]
              : axis === 1 ? [u, this.plan.brickResolution - 1, v] : [u, v, this.plan.brickResolution - 1];
            const low: [number, number, number] = axis === 0 ? [0, u, v] : axis === 1 ? [u, 0, v] : [u, v, 0];
            const index = (coord: readonly [number, number, number]) =>
              coord[0] + this.plan.brickResolution * (coord[1] + this.plan.brickResolution * coord[2]);
            const highIndex = index(high);
            const lowIndex = index(low);
            if ((page.flags[highIndex] & FINE_LEVELSET_SAMPLE_FLAGS.valid) === 0
              || (neighbor.flags[lowIndex] & FINE_LEVELSET_SAMPLE_FLAGS.valid) === 0) continue;
            if ((page.phi[highIndex] < 0) !== (neighbor.phi[lowIndex] < 0)) { crossing = true; break; }
          }
        }
        if (crossing) { result.add(key); result.add(neighbor.key); }
      }
    }
    return Uint32Array.from(sortedUnique(result));
  }

  sampleOrCoarse(position: FineLevelSetVec3, coarsePhi: FineLevelSetCoarsePhi): number {
    const address = fineLevelSetAddressAtPosition(this.plan, position);
    if (address) {
      const page = this.pageForKey(address.brickKey);
      if (page && (page.flags[address.localIndex] & FINE_LEVELSET_SAMPLE_FLAGS.valid) !== 0) {
        return page.phi[address.localIndex];
      }
    }
    const fallback = coarsePhi(position);
    if (!Number.isFinite(fallback)) throw new RangeError("Coarse phi fallback returned a non-finite value");
    return fallback;
  }

  sampleResidentAtCoordinate(fineCoord: FineLevelSetVec3): number | undefined {
    if (fineCoord.some((value, axis) => !Number.isSafeInteger(value)
      || value < 0 || value >= this.plan.sampleDimensions[axis])) return undefined;
    const address = fineLevelSetAddressAtCoordinate(this.plan, fineCoord);
    const page = this.pageForKey(address.brickKey);
    if (!page || (page.flags[address.localIndex] & FINE_LEVELSET_SAMPLE_FLAGS.valid) === 0) return undefined;
    const value = page.phi[address.localIndex];
    return Number.isFinite(value) ? value : undefined;
  }

  /** Trilinear fine-phi sampling; undefined means the sparse band is incomplete. */
  sampleResidentTrilinear(position: FineLevelSetVec3): number | undefined {
    const lattice = position.map((value, axis) =>
      (value - this.plan.domainOrigin[axis]) / this.plan.fineCellWidth - 0.5) as [number, number, number];
    const base = lattice.map(Math.floor) as [number, number, number];
    const fraction = lattice.map((value, axis) => value - base[axis]) as [number, number, number];
    let result = 0;
    for (let z = 0; z < 2; z += 1) for (let y = 0; y < 2; y += 1) for (let x = 0; x < 2; x += 1) {
      const weight = (x ? fraction[0] : 1 - fraction[0])
        * (y ? fraction[1] : 1 - fraction[1]) * (z ? fraction[2] : 1 - fraction[2]);
      if (weight === 0) continue;
      const value = this.sampleResidentAtCoordinate([base[0] + x, base[1] + y, base[2] + z]);
      if (value === undefined) return undefined;
      result += weight * value;
    }
    return result;
  }

  memoryAccounting(): FineLevelSetBrickMemory {
    const residentBricks = this.pagesByKey.size;
    const activePayloadBytes = residentBricks * this.plan.payloadBytesPerBrick;
    const metadataBytes = 2 * residentBricks * 10 * 4;
    return {
      residentBricks, activePayloadBytes, payloadCapacityBytes: this.plan.payloadCapacityBytes,
      metadataBytes, pageTableBytes: this.plan.pageTableBytes, worklistBytes: this.plan.worklistBytes,
      fragmentationBytes: this.plan.payloadCapacityBytes - activePayloadBytes,
      allocatedBytes: this.plan.allocatedBytes,
    };
  }

  exportGPUGeneration(): FineLevelSetGPUGenerationData {
    const { maximumResidentBricks: capacity, samplesPerBrick } = this.plan;
    const metadataWords = new Uint32Array(capacity * 10).fill(FINE_LEVELSET_INVALID);
    const flags = new Uint32Array(capacity * samplesPerBrick);
    const phi = new Float32Array(capacity * samplesPerBrick);
    const workA = new Float32Array(capacity * samplesPerBrick);
    const workB = new Float32Array(capacity * samplesPerBrick);
    const ordered = [...this.pagesByKey.values()].sort((a, b) => a.key - b.key);
    const worklistWords = new Uint32Array(5 + capacity);
    worklistWords.set([ordered.length, this.generationValue, Math.ceil(ordered.length / 64), 1, 1]);
    ordered.forEach((page, workIndex) => {
      const metadataOffset = page.physicalId * 10;
      metadataWords.set([page.physicalId, page.key, page.generation, 1, ...page.neighborIds], metadataOffset);
      const payloadOffset = page.physicalId * samplesPerBrick;
      flags.set(page.flags, payloadOffset);
      phi.set(page.phi, payloadOffset);
      workA.set(page.workA, payloadOffset);
      workB.set(page.workB, payloadOffset);
      worklistWords[5 + workIndex] = page.physicalId;
    });
    return {
      generation: this.generationValue, activeCount: ordered.length,
      hashPairs: this.hash.pairs.slice(), metadataWords, worklistWords, flags, phi, workA, workB,
    };
  }
}
