import {
  SVO_NODE_MIP_LAYOUT,
  svoNodeMipAtlasPageCoordinate,
  svoNodeMipPageKey,
  svoNodeMipSeedPage,
  svoRadianceLevelFloor,
  type SvoNodeMipCoordinate,
  type SvoNodeMipPagePlan,
  type SvoNodeMipPageSeed,
  type SvoNodeMipPyramidPlan,
} from "./svo-node-mip-pyramid";
import {
  SVO_NODE_MIP_PAGE_HASH,
  findSvoNodeMipPageHashSlot,
  insertSvoNodeMipPageHash,
  removeSvoNodeMipPageHash,
  svoNodeMipPageHashHome,
  svoNodeMipPageHashShape,
} from "./svo-node-mip-page-hash";
import { completeCooperativeBuild } from "../../../core/cooperative-build";

/**
 * Where the live node-mip pyramid's *addresses* are decided, as opposed to its
 * contents.
 *
 * A page's address is a slot, and a slot is found through the page hash
 * (`svo-node-mip-page-hash.ts`) keyed by `(level, coordinate)`. Nothing reads
 * slots in order, so nothing about a slot depends on the pages around it: a new
 * page takes a free slot and one hash entry, an erased page gives both back, and
 * every other page stays where it is. The work an edit asks of this module is
 * therefore the pages that edit adds or removes, never the scene's.
 *
 * Two shapes, in preference order:
 *
 *  1. **Total.** Every page the declared domain can hold gets a slot up front.
 *     No edit can activate a page outside the plan, because there is no outside.
 *  2. **Reserved.** The domain is too large to address whole, so the plan covers
 *     what is occupied plus a bounded reserve, and hands slots out of that
 *     reserve as edits need them. An edit the reserve cannot hold is refused
 *     before it changes anything.
 *
 * The reserve is priced in bytes, not in a fraction, because that is what it
 * actually costs: a slot is one opacity page plus four radiance lobes in the
 * atlas, and one more of each in the builder's scratch.
 *
 * **Two pools.** The radiance atlas is addressed by `slot - radianceSlotOffset`
 * and the derived builder tells a radiance page from an opacity-only one by
 * that same comparison, so pages below the radiance floor live in
 * `[0, radianceSlotOffset)` and pages at or above it in
 * `[radianceSlotOffset, pageCapacity)`. The offset is fixed when the plan is
 * made. So that an edit may still spend the whole reserve on either kind, each
 * pool is sized to hold all of it; what limits an edit is the resident page
 * count against `residentPageLimit`, and the pools merely never bind first.
 * The extra opacity-only slots that costs are a few percent of the reserve's
 * own price.
 */
export const SVO_NODE_MIP_ADDRESS_PLAN = Object.freeze({
  /**
   * Opacity page + four radiance lobes, in the target atlas.
   *
   * Priced at 8 B/texel — the `rgba16float` fallback — deliberately, even though
   * a device with `texture-formats-tier1` allocates the lobes as
   * `rg11b10ufloat` at 4. The reserve is a *byte* budget, so a cheaper page
   * would buy proportionally more reserved slots and move the atlas shape with
   * the device's feature set. Holding the conservative figure keeps the address
   * plan a pure function of the scene: the narrow format then simply spends
   * less than its budget, which is never a fault.
   */
  atlasBytesPerPage: SVO_NODE_MIP_LAYOUT.bytesPerPage + 4 * SVO_NODE_MIP_LAYOUT.physicalSize ** 3 * 8,
  /**
   * What one reserved slot costs in total. Twice the atlas figure: the builder's
   * compact scratch is sized by the largest per-level worklist, and raising the
   * addressable page count raises that worklist by the same amount.
   */
  reservedBytesPerPage: 2 * (SVO_NODE_MIP_LAYOUT.bytesPerPage + 4 * SVO_NODE_MIP_LAYOUT.physicalSize ** 3 * 8),
  /**
   * Physical allocation this module is willing to spend on addresses that hold
   * nothing yet. Sized so an ordinary authored room plans totally — the hero
   * garden needs 196 of these — while a solver-sized domain still gets a usable
   * growth window instead of the whole lattice.
   */
  maximumReserveBytes: 48 * 1024 * 1024,
} as const);

export interface SvoNodeMipAddressPlanOptions {
  /**
   * Virtual pages the current topology covers, each at its own level.
   *
   * A bare coordinate still means a finest-level page, so every producer that
   * only ever had base pages reads unchanged. A coarse leaf seeds the level
   * whose texels are its voxels — see {@link SvoNodeMipSeedPage}.
   */
  occupiedBasePages: readonly SvoNodeMipPageSeed[];
  /** Finest-level page grid of the whole editable domain. */
  basePageDimensions: readonly [number, number, number];
  levelCount: number;
  /** Pages this device can address at all. */
  addressCapacity: number;
  /** Stable for the life of the plan: page-local validity is the live generation authority. */
  generation?: number;
  maximumReserveBytes?: number;
  /** Finest leaf edge in metres; anchors the radiance floor to a world size. */
  cellSize_m?: number;
  /** Explicit radiance floor level. Zero restores a radiance page at every level. */
  radianceFloorLevel?: number;
  /**
   * Finest level that owns an *opacity* page; see `SVO_OPACITY_LEVEL_FLOOR`.
   *
   * The plan carries it so that growth stays inside the shape the first
   * publication chose: an edit hands this module bare level-0 coordinates, and
   * seeding one of those under a floored plan would put a base level back under
   * part of the scene and nowhere else. Zero is the shipped behaviour and every
   * path below is then the identity.
   */
  opacityFloorLevel?: number;
}

/** What a set of base pages asks of the plan, counted without changing it. */
export interface SvoNodeMipAddressPlanGrowth {
  /** Pages the request needs that the plan does not hold. Zero means it is already addressed. */
  newPages: number;
  /** Of those, the ones below the radiance floor, which take an opacity-only slot. */
  newOpacityOnlyPages: number;
  /**
   * Distinct pages on the requested pages' ancestor chains, new or not: every
   * page whose contents the request changes, and so everything the builder
   * owes for it.
   */
  chainPages: number;
  /** The plan can hold the request. When false nothing may be grown and `refusal` says why. */
  fits: boolean;
  refusal?: string;
}

const RESIDENT = 1;
const SEEDED = 2;
const COORDINATE_LIMIT = 2 ** SVO_NODE_MIP_PAGE_HASH.coordinateBits;

function positiveDimensions(value: readonly [number, number, number]): void {
  if (value.length !== 3 || value.some((component) => !Number.isSafeInteger(component) || component <= 0)) {
    throw new RangeError("Node-mip base page dimensions must be three positive safe integers");
  }
}

/** Page grid at each virtual level, given the finest one. Mirrors the plan's `floor(c/2)` ancestry. */
export function svoNodeMipLevelPageDimensions(
  basePageDimensions: readonly [number, number, number],
  levelCount: number,
): readonly (readonly [number, number, number])[] {
  positiveDimensions(basePageDimensions);
  if (!Number.isSafeInteger(levelCount) || levelCount < 1 || levelCount > 32) {
    throw new RangeError("Node-mip level count must be an integer in [1, 32]");
  }
  const levels: (readonly [number, number, number])[] = [];
  let current = [...basePageDimensions] as [number, number, number];
  for (let level = 0; level < levelCount; level += 1) {
    levels.push([...current] as [number, number, number]);
    current = current.map((component) => Math.max(1, Math.ceil(component / 2))) as [number, number, number];
  }
  return levels;
}

/**
 * Pages a plan over *every* base page of the domain would request, without
 * materialising them. Deciding whether the total shape is affordable must not
 * itself cost the allocation being decided against.
 */
export function svoNodeMipDomainPyramidPageCount(
  basePageDimensions: readonly [number, number, number],
  levelCount: number,
): number {
  return svoNodeMipLevelPageDimensions(basePageDimensions, levelCount)
    .reduce((total, dimensions) => total + dimensions[0] * dimensions[1] * dimensions[2], 0);
}

/** Every base page of the domain, x fastest. */
export function svoNodeMipDomainBasePages(
  basePageDimensions: readonly [number, number, number],
): readonly SvoNodeMipCoordinate[] {
  positiveDimensions(basePageDimensions);
  const [width, height, depth] = basePageDimensions;
  const pages: SvoNodeMipCoordinate[] = [];
  for (let z = 0; z < depth; z += 1) for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) pages.push([x, y, z]);
  return pages;
}

/**
 * Atlas page shape for a fixed slot capacity: near-cubic, and a function of the
 * capacity alone, because the atlas has to outlive every edit.
 */
export function svoNodeMipAtlasPagesForCapacity(capacity: number): readonly [number, number, number] {
  if (!Number.isSafeInteger(capacity) || capacity <= 0) throw new RangeError("Node-mip atlas capacity must be positive");
  const x = Math.ceil(Math.cbrt(capacity));
  const y = Math.ceil(Math.sqrt(capacity / x));
  return [x, y, Math.ceil(capacity / (x * y))];
}

function nextPowerOfTwo(value: number): number {
  let result = 1;
  while (result < value) result *= 2;
  return result;
}

/**
 * Open-addressed `(level, coordinate) -> value` map in one typed array, keyed
 * the way the page hash is. It is what finds a scene's distinct pages before
 * their capacity is known, and what keeps one edit from counting a shared
 * ancestor twice.
 */
class PageKeyMap {
  private words: Uint32Array;
  private capacity: number;
  count = 0;

  constructor(expected: number) {
    this.capacity = nextPowerOfTwo(Math.max(64, expected * 2));
    this.words = new Uint32Array(this.capacity * 4);
  }

  /** Empty the map, growing it first when `expected` keys would crowd it. */
  reset(expected: number): void {
    const capacity = nextPowerOfTwo(Math.max(64, expected * 2));
    if (capacity > this.capacity) { this.capacity = capacity; this.words = new Uint32Array(capacity * 4); }
    else this.words.fill(0);
    this.count = 0;
  }

  /** The value held for a key, or -1 after adding the key with `value`. */
  findOrAdd(level: number, x: number, y: number, z: number, value: number): number {
    if (this.count * 2 >= this.capacity) this.grow();
    const key0 = (x | (level << SVO_NODE_MIP_PAGE_HASH.coordinateBits)) >>> 0;
    const mask = this.capacity - 1;
    const words = this.words;
    for (let index = svoNodeMipPageHashHome(level, x, y, z) & mask; ; index = (index + 1) & mask) {
      const base = index * 4;
      if (words[base + 3] === 0) {
        words[base] = key0; words[base + 1] = y; words[base + 2] = z; words[base + 3] = value + 1;
        this.count += 1;
        return -1;
      }
      if (words[base] === key0 && words[base + 1] === y && words[base + 2] === z) return words[base + 3] - 1;
    }
  }

  private grow(): void {
    const old = this.words;
    this.capacity *= 2;
    this.words = new Uint32Array(this.capacity * 4);
    this.count = 0;
    for (let base = 0; base < old.length; base += 4) {
      if (old[base + 3] === 0) continue;
      this.findOrAdd(old[base] >>> SVO_NODE_MIP_PAGE_HASH.coordinateBits, old[base] & (COORDINATE_LIMIT - 1),
        old[base + 1], old[base + 2], old[base + 3] - 1);
    }
  }
}

/** A scene's distinct pages in the order they were first reached: `x, y, z, level` quads. */
class PageDiscovery {
  private readonly map: PageKeyMap;
  pages: Uint32Array<ArrayBuffer>;
  seeded: Uint8Array<ArrayBuffer>;
  count = 0;

  constructor(expected: number, private readonly levelCount: number, private readonly floorLevel: number) {
    this.map = new PageKeyMap(expected * 2);
    this.pages = new Uint32Array(Math.max(64, expected * 2) * 4);
    this.seeded = new Uint8Array(Math.max(64, expected * 2));
  }

  /**
   * Claim a seed's page and every ancestor it does not have yet. The walk stops
   * at the first page already known: that page's own ancestors were added when
   * it was, so residency stays ancestor-closed without visiting them again.
   */
  seed(seedLevel: number, seedX: number, seedY: number, seedZ: number): void {
    let x = seedX, y = seedY, z = seedZ, level = seedLevel;
    if (level < this.floorLevel) {
      const shift = this.floorLevel - level;
      x = Math.floor(x / 2 ** shift); y = Math.floor(y / 2 ** shift); z = Math.floor(z / 2 ** shift);
      level = this.floorLevel;
    }
    if (x >= COORDINATE_LIMIT || y >= COORDINATE_LIMIT || z >= COORDINATE_LIMIT) {
      throw new RangeError(`SVO node-mip coordinates must fit ${SVO_NODE_MIP_PAGE_HASH.coordinateBits} bits per axis`);
    }
    for (let first = true; level < this.levelCount; level += 1, first = false) {
      const known = this.map.findOrAdd(level, x, y, z, this.count);
      if (known >= 0) {
        if (first) this.seeded[known] = 1;
        return;
      }
      if (this.count === this.seeded.length) {
        const pages = new Uint32Array(this.pages.length * 2); pages.set(this.pages); this.pages = pages;
        const seeded = new Uint8Array(this.seeded.length * 2); seeded.set(this.seeded); this.seeded = seeded;
      }
      const base = this.count * 4;
      this.pages[base] = x; this.pages[base + 1] = y; this.pages[base + 2] = z; this.pages[base + 3] = level;
      this.seeded[this.count] = first ? 1 : 0;
      this.count += 1;
      x >>>= 1; y >>>= 1; z >>>= 1;
    }
  }
}

interface SvoNodeMipAddressPlanShape {
  generation: number;
  levelCount: number;
  basePageDimensions: readonly [number, number, number];
  levelDimensions: readonly (readonly [number, number, number])[];
  opacityFloorLevel: number;
  radianceFloorLevel: number;
  /** Pages the scene asked for, whether or not they fit. */
  requestedPageCount: number;
  residentPageLimit: number;
  opacityOnlyPageCapacity: number;
  radiancePageCapacity: number;
  total: boolean;
  domainBasePageCount: number;
  domainPyramidPageCount: number;
  reservePages: number;
  pageCapacityPerLevel: number;
  pageCapacityByLevel: readonly number[];
  maximumReserveBytes: number;
  radiancePageCapacityPerLevel: number;
}

/**
 * The live address plan: which slot every resident page holds, and which slots
 * are free. One object for the life of a world; edits change it in place.
 */
export class SvoNodeMipAddressPlan {
  /** What the pyramid, the radiance atlas and the renderer read. The same object for the life of the plan. */
  readonly plan: SvoNodeMipPyramidPlan;
  /** Fixed for the lifetime of the world, so no edit reallocates an atlas. */
  readonly atlasPages: readonly [number, number, number];
  readonly atlasTexels: readonly [number, number, number];
  /** Slots, and so the size of everything indexed by one. Both pools; see the module comment. */
  readonly pageCapacity: number;
  /** Pages that may be resident at once: what the scene held when it was built plus `reservePages`. */
  readonly residentPageLimit: number;
  readonly levelCount: number;
  readonly basePageDimensions: readonly [number, number, number];
  /** Every page of the declared domain is addressed; growth can never be needed. */
  readonly total: boolean;
  readonly domainBasePageCount: number;
  readonly domainPyramidPageCount: number;
  /** Pages held back, at build, for activations the first publication did not see. */
  readonly reservePages: number;
  /** Worklist depth the deepest level may need, which bounds the builder's scratch. */
  readonly pageCapacityPerLevel: number;
  /**
   * Worklist depth *each* level may need, which sizes the arena's sections: a
   * level's own domain grid, capped by what the reserve could add to what it
   * held at build. Growth refuses a request that would put a level past it.
   */
  readonly pageCapacityByLevel: readonly number[];
  readonly maximumReserveBytes: number;
  /**
   * Finest level that owns a radiance page. See `SVO_RADIANCE_LEVEL_FLOOR`.
   *
   * Everything below is opacity-only, and the fields beneath describe a radiance
   * atlas that is *not* the opacity atlas: fewer slots, its own shape, and its
   * own origin in the shared slot numbering.
   */
  readonly radianceFloorLevel: number;
  /** First slot of the radiance pool, and the size of the opacity-only pool below it. Never moves. */
  readonly radianceSlotOffset: number;
  /** Physical radiance slots: `[radianceSlotOffset, pageCapacity)`. */
  readonly radiancePageCapacity: number;
  readonly radianceAtlasPages: readonly [number, number, number];
  readonly radianceAtlasTexels: readonly [number, number, number];
  /** Radiance worklist depth one level may need; bounds the radiance scratch. */
  readonly radiancePageCapacityPerLevel: number;
  /** Finest level that owns an opacity page. Zero is the shipped pyramid. */
  readonly opacityFloorLevel: number;
  /** Entries in the page hash; a power of two. */
  readonly pageTableCapacity: number;
  /** The page hash exactly as the device holds it: `pageTableCapacity` entries of four words. */
  readonly pageTableWords: Uint32Array<ArrayBuffer>;
  /** `x, y, z, level` of the page in each slot; zero where the slot is free. */
  readonly slotPages: Uint32Array<ArrayBuffer>;

  private readonly levelDimensions: readonly (readonly [number, number, number])[];
  private readonly slotState: Uint8Array;
  /** Resident child pages of each slot's page. A page with none and no seed of its own is returned. */
  private readonly childCounts: Uint8Array;
  private readonly levelCounts: Uint32Array;
  /** Returned slots: opacity-only ones stacked from the front, radiance ones from the back. */
  private readonly freeSlots: Uint32Array;
  private opacityOnlyFree = 0;
  private radianceFree = 0;
  /** Slots of each pool handed out at least once; the untouched reserve starts here. */
  private opacityOnlyUsed = 0;
  private radianceUsed = 0;
  private opacityOnlyCount = 0;
  private radianceCount = 0;
  private seedCount = 0;
  private revision = 0;
  private pagesRevision = -1;
  private pagesCache: readonly SvoNodeMipPagePlan[] = [];
  private readonly pendingEntries: number[] = [];
  private readonly pendingSlots: number[] = [];
  private readonly scratch = new PageKeyMap(64);
  private readonly scratchLevels: Uint32Array;
  private readonly chain: Uint32Array;
  private readonly noteEntry = (entry: number): void => { this.pendingEntries.push(entry); };

  /** Made by `planSvoNodeMipAddresses`; `discovery` is absent for a plan the scene does not fit. */
  constructor(shape: SvoNodeMipAddressPlanShape, discovery: PageDiscovery | undefined) {
    this.levelCount = shape.levelCount;
    this.basePageDimensions = shape.basePageDimensions;
    this.levelDimensions = shape.levelDimensions;
    this.opacityFloorLevel = shape.opacityFloorLevel;
    this.radianceFloorLevel = shape.radianceFloorLevel;
    this.residentPageLimit = shape.residentPageLimit;
    this.radianceSlotOffset = shape.opacityOnlyPageCapacity;
    this.radiancePageCapacity = shape.radiancePageCapacity;
    this.pageCapacity = shape.opacityOnlyPageCapacity + shape.radiancePageCapacity;
    this.total = shape.total;
    this.domainBasePageCount = shape.domainBasePageCount;
    this.domainPyramidPageCount = shape.domainPyramidPageCount;
    this.reservePages = shape.reservePages;
    this.pageCapacityPerLevel = shape.pageCapacityPerLevel;
    this.pageCapacityByLevel = shape.pageCapacityByLevel;
    this.maximumReserveBytes = shape.maximumReserveBytes;
    this.radiancePageCapacityPerLevel = shape.radiancePageCapacityPerLevel;
    const physical = SVO_NODE_MIP_LAYOUT.physicalSize;
    this.atlasPages = svoNodeMipAtlasPagesForCapacity(this.pageCapacity);
    this.atlasTexels = this.atlasPages.map((value) => value * physical) as [number, number, number];
    this.radianceAtlasPages = svoNodeMipAtlasPagesForCapacity(this.radiancePageCapacity);
    this.radianceAtlasTexels = this.radianceAtlasPages.map((value) => value * physical) as [number, number, number];
    this.pageTableCapacity = svoNodeMipPageHashShape(this.pageCapacity, Number.MAX_SAFE_INTEGER).capacity;
    this.pageTableWords = new Uint32Array(this.pageTableCapacity * SVO_NODE_MIP_PAGE_HASH.wordsPerEntry);
    this.slotPages = new Uint32Array(this.pageCapacity * 4);
    this.slotState = new Uint8Array(this.pageCapacity);
    this.childCounts = new Uint8Array(this.pageCapacity);
    this.levelCounts = new Uint32Array(this.levelCount);
    this.freeSlots = new Uint32Array(this.pageCapacity);
    this.scratchLevels = new Uint32Array(this.levelCount);
    this.chain = new Uint32Array(this.levelCount * 3);
    if (discovery) {
      const { pages, seeded } = discovery;
      for (let index = 0; index < discovery.count; index += 1) {
        const base = index * 4;
        const slot = this.place(pages[base + 3], pages[base], pages[base + 1], pages[base + 2]);
        if (seeded[index]) { this.slotState[slot] |= SEEDED; this.seedCount += 1; }
      }
      for (let slot = 0; slot < this.pageCapacity; slot += 1) {
        if (!this.slotState[slot]) continue;
        const base = slot * 4;
        const level = this.slotPages[base + 3];
        if (level + 1 >= this.levelCount) continue;
        this.childCounts[this.find(level + 1, this.slotPages[base] >>> 1, this.slotPages[base + 1] >>> 1, this.slotPages[base + 2] >>> 1)] += 1;
      }
      // The first upload is the whole table, so there is nothing to carry over.
      this.pendingEntries.length = 0;
      this.pendingSlots.length = 0;
    }
    const complete = discovery !== undefined;
    const requestedPageCount = shape.requestedPageCount;
    const atlasCapacity = this.atlasPages[0] * this.atlasPages[1] * this.atlasPages[2];
    const atlasBytes = atlasCapacity * SVO_NODE_MIP_LAYOUT.bytesPerPage;
    // Read live rather than captured: the counts move with every edit.
    const pages = (): readonly SvoNodeMipPagePlan[] => this.materialisePages();
    const resident = (): number => this.residentPageCount;
    const finest = (): number => this.finestResidentLevel;
    this.plan = {
      generation: shape.generation,
      get pages() { return pages(); },
      atlas: { pages: this.atlasPages as SvoNodeMipCoordinate, texels: this.atlasTexels as SvoNodeMipCoordinate, capacity: atlasCapacity },
      get requestedPageCount() { return complete ? resident() : requestedPageCount; },
      get residentPageCount() { return resident(); },
      slotCapacity: this.pageCapacity,
      levelCount: this.levelCount,
      get finestLevel() { return finest(); },
      slotPages: this.slotPages,
      pageTable: { capacity: this.pageTableCapacity, words: this.pageTableWords },
      overflowPageCount: complete ? 0 : requestedPageCount,
      get pagePayloadBytes() { return resident() * SVO_NODE_MIP_LAYOUT.bytesPerPage; },
      atlasBytes,
      directoryBytes: 0,
      allocatedBytes: atlasBytes,
      complete,
    };
  }

  get residentPageCount(): number { return this.opacityOnlyCount + this.radianceCount; }
  /** Pages claimed by the topology itself rather than held as some page's ancestor. */
  get seedPageCount(): number { return this.seedCount; }
  /** Pages an edit may still add: what is left of the reserve, counting returned pages. */
  get remainingReservePages(): number { return Math.max(0, this.residentPageLimit - this.residentPageCount); }
  /** Lowest level holding a page; the opacity floor unless the scene is all coarse leaves. */
  get finestResidentLevel(): number {
    for (let level = 0; level < this.levelCount; level += 1) if (this.levelCounts[level] > 0) return level;
    return this.opacityFloorLevel;
  }
  /** Resident pages at one level. */
  levelPageCount(level: number): number { return this.levelCounts[level] ?? 0; }

  /** The slot a page holds, or -1. */
  find(level: number, x: number, y: number, z: number): number {
    return findSvoNodeMipPageHashSlot(this.pageTableWords, this.pageTableCapacity, level, x, y, z);
  }

  /**
   * Count what base pages would add, in work proportional to those pages and
   * the levels above them. The plan is not changed.
   */
  measureGrowth(requestedBasePages: readonly SvoNodeMipCoordinate[]): SvoNodeMipAddressPlanGrowth {
    const floor = this.opacityFloorLevel;
    const levelNew = this.scratchLevels.fill(0);
    this.scratch.reset(requestedBasePages.length * (this.levelCount - floor));
    let newPages = 0, newOpacityOnlyPages = 0, chainPages = 0;
    for (const page of requestedBasePages) {
      // Domain membership is asked of the *finest* coordinate the caller named,
      // before the floor collapses it: a page one step outside the base grid
      // maps inside the floor's grid, and accepting it would address geometry
      // the plan cannot address.
      if (!this.insideDomain(page)) {
        return { newPages: 0, newOpacityOnlyPages: 0, chainPages: 0, fits: false,
          refusal: `page ${page.join(",")} is outside the ${this.basePageDimensions.join("x")} page domain` };
      }
      let x = page[0] >>> floor, y = page[1] >>> floor, z = page[2] >>> floor;
      for (let level = floor, known = false; level < this.levelCount; level += 1) {
        if (this.scratch.findOrAdd(level, x, y, z, 0) >= 0) break;
        chainPages += 1;
        // Once a page is resident so is every ancestor, so the table is asked
        // only until the first hit.
        if (!known && this.find(level, x, y, z) >= 0) known = true;
        if (!known) {
          newPages += 1;
          levelNew[level] += 1;
          if (level < this.radianceFloorLevel) newOpacityOnlyPages += 1;
        }
        x >>>= 1; y >>>= 1; z >>>= 1;
      }
    }
    const growth: SvoNodeMipAddressPlanGrowth = { newPages, newOpacityOnlyPages, chainPages, fits: true };
    const refuse = (refusal: string): SvoNodeMipAddressPlanGrowth => ({ ...growth, fits: false, refusal });
    if (newPages === 0) return growth;
    if (newPages > this.remainingReservePages) {
      return refuse(`${newPages} new lighting pages against ${this.remainingReservePages} left of the ${this.reservePages} reserved`);
    }
    if (this.opacityOnlyCount + newOpacityOnlyPages > this.radianceSlotOffset) {
      return refuse(`${newOpacityOnlyPages} new opacity pages against ${this.radianceSlotOffset - this.opacityOnlyCount} free slots below the radiance floor`);
    }
    if (this.radianceCount + newPages - newOpacityOnlyPages > this.radiancePageCapacity) {
      return refuse(`${newPages - newOpacityOnlyPages} new radiance pages against ${this.radiancePageCapacity - this.radianceCount} free radiance slots`);
    }
    for (let level = floor; level < this.levelCount; level += 1) {
      if (this.levelCounts[level] + levelNew[level] > this.pageCapacityByLevel[level]) {
        return refuse(`${levelNew[level]} new pages at level ${level} against ${this.pageCapacityByLevel[level] - this.levelCounts[level]} left of its ${this.pageCapacityByLevel[level]}-page worklist`);
      }
    }
    return growth;
  }

  /**
   * Address base pages an edit activated: each takes a free slot and one hash
   * entry, and so does every ancestor it lacks. No resident page moves. Throws,
   * before changing anything, when `measureGrowth` says the request does not fit.
   */
  grow(requestedBasePages: readonly SvoNodeMipCoordinate[]): SvoNodeMipAddressPlanGrowth {
    const growth = this.measureGrowth(requestedBasePages);
    if (!growth.fits) throw new RangeError(`Node-mip address plan cannot grow: ${growth.refusal}`);
    const floor = this.opacityFloorLevel;
    const chain = this.chain;
    let changed = false;
    for (const page of requestedBasePages) {
      let x = page[0] >>> floor, y = page[1] >>> floor, z = page[2] >>> floor;
      let missing = 0, slot = -1;
      for (let level = floor; level < this.levelCount; level += 1) {
        slot = this.find(level, x, y, z);
        if (slot >= 0) break;
        chain[missing * 3] = x; chain[missing * 3 + 1] = y; chain[missing * 3 + 2] = z;
        missing += 1;
        x >>>= 1; y >>>= 1; z >>>= 1;
      }
      // Coarsest first, so each page finds its parent already placed.
      for (let index = missing - 1; index >= 0; index -= 1) {
        const parent = slot;
        slot = this.place(floor + index, chain[index * 3], chain[index * 3 + 1], chain[index * 3 + 2]);
        if (parent >= 0) this.childCounts[parent] += 1;
      }
      if (!(this.slotState[slot] & SEEDED)) { this.slotState[slot] |= SEEDED; this.seedCount += 1; changed = true; }
      changed ||= missing > 0;
    }
    if (changed) this.revision += 1;
    return growth;
  }

  /**
   * Give back the pages of base pages the topology no longer claims.
   *
   * A page is returned when nothing claims it and it has no resident child, and
   * returning it may leave its parent in the same state, so the walk continues
   * upward for as long as that holds. The caller answers for the claim: under an
   * opacity floor several base pages share one floor page, and it must name one
   * only when none of them is occupied any longer. A total plan keeps every
   * page, because its whole point is that no edit ever needs one. Returns the
   * number of pages returned to the reserve.
   */
  release(releasedBasePages: readonly SvoNodeMipCoordinate[]): number {
    if (this.total) return 0;
    const floor = this.opacityFloorLevel;
    let released = 0, changed = false;
    for (const page of releasedBasePages) {
      if (!this.insideDomain(page)) continue;
      let x = page[0] >>> floor, y = page[1] >>> floor, z = page[2] >>> floor;
      let slot = this.find(floor, x, y, z);
      if (slot < 0 || !(this.slotState[slot] & SEEDED)) continue;
      this.slotState[slot] &= ~SEEDED;
      this.seedCount -= 1;
      changed = true;
      for (let level = floor; slot >= 0 && this.slotState[slot] === RESIDENT && this.childCounts[slot] === 0;) {
        removeSvoNodeMipPageHash(this.pageTableWords, this.pageTableCapacity, level, x, y, z, this.noteEntry);
        this.vacate(slot, level);
        released += 1;
        level += 1;
        if (level >= this.levelCount) break;
        x >>>= 1; y >>>= 1; z >>>= 1;
        slot = this.find(level, x, y, z);
        if (slot >= 0) this.childCounts[slot] -= 1;
      }
    }
    if (changed) this.revision += 1;
    return released;
  }

  /**
   * Hand over what changed since the last call and forget it: every hash entry
   * whose texel must be rewritten, and every slot that was given to a new page
   * and so holds contents that are not that page's until it is built.
   */
  drainChanges(entry: (index: number) => void, slot: (index: number) => void): void {
    for (const index of this.pendingEntries) entry(index);
    for (const index of this.pendingSlots) slot(index);
    this.pendingEntries.length = 0;
    this.pendingSlots.length = 0;
  }

  private insideDomain(page: SvoNodeMipCoordinate): boolean {
    for (let axis = 0; axis < 3; axis += 1) {
      const component = page[axis];
      if (!Number.isSafeInteger(component) || component < 0 || component >= this.basePageDimensions[axis]) return false;
    }
    return true;
  }

  /** Give a page a slot and a hash entry. A returned slot is reused before the reserve is touched. */
  private place(level: number, x: number, y: number, z: number): number {
    let slot: number;
    if (level < this.radianceFloorLevel) {
      if (this.opacityOnlyFree > 0) slot = this.freeSlots[this.opacityOnlyFree -= 1];
      else if (this.opacityOnlyUsed < this.radianceSlotOffset) slot = (this.opacityOnlyUsed += 1) - 1;
      else throw new RangeError("Node-mip address plan has no opacity-only slot left");
      this.opacityOnlyCount += 1;
    } else {
      if (this.radianceFree > 0) slot = this.freeSlots[this.pageCapacity - (this.radianceFree -= 1) - 1];
      else if (this.radianceUsed < this.radiancePageCapacity) slot = this.radianceSlotOffset + (this.radianceUsed += 1) - 1;
      else throw new RangeError("Node-mip address plan has no radiance slot left");
      this.radianceCount += 1;
    }
    const base = slot * 4;
    this.slotPages[base] = x; this.slotPages[base + 1] = y; this.slotPages[base + 2] = z; this.slotPages[base + 3] = level;
    this.slotState[slot] = RESIDENT;
    this.childCounts[slot] = 0;
    this.levelCounts[level] += 1;
    this.pendingEntries.push(insertSvoNodeMipPageHash(this.pageTableWords, this.pageTableCapacity, level, x, y, z, slot));
    this.pendingSlots.push(slot);
    return slot;
  }

  private vacate(slot: number, level: number): void {
    this.slotPages.fill(0, slot * 4, slot * 4 + 4);
    this.slotState[slot] = 0;
    this.levelCounts[level] -= 1;
    if (level < this.radianceFloorLevel) {
      this.freeSlots[this.opacityOnlyFree] = slot; this.opacityOnlyFree += 1; this.opacityOnlyCount -= 1;
    } else {
      this.freeSlots[this.pageCapacity - this.radianceFree - 1] = slot; this.radianceFree += 1; this.radianceCount -= 1;
    }
  }

  private materialisePages(): readonly SvoNodeMipPagePlan[] {
    if (this.pagesRevision === this.revision) return this.pagesCache;
    const physical = SVO_NODE_MIP_LAYOUT.physicalSize;
    const pages: SvoNodeMipPagePlan[] = [];
    for (let slot = 0; slot < this.pageCapacity; slot += 1) {
      if (!this.slotState[slot]) continue;
      const base = slot * 4;
      const key = { generation: this.plan.generation, level: this.slotPages[base + 3],
        coordinate: [this.slotPages[base], this.slotPages[base + 1], this.slotPages[base + 2]] as SvoNodeMipCoordinate };
      const atlasPage = svoNodeMipAtlasPageCoordinate(slot, this.atlasPages as SvoNodeMipCoordinate);
      pages.push({ key, keyString: svoNodeMipPageKey(key), slot, atlasPage,
        atlasTexelOrigin: atlasPage.map((component) => component * physical) as unknown as SvoNodeMipCoordinate });
    }
    this.pagesCache = pages;
    this.pagesRevision = this.revision;
    return pages;
  }
}

/** Seeds per yield offer; each walks one chain of at most `levelCount` hash probes. */
const NODE_MIP_ADDRESS_YIELD_BATCH = 4096;

/**
 * Decide the address plan once, for the lifetime of the world.
 *
 * Throws nothing a caller has to interpret: an unusable domain comes back as a
 * plan whose `complete` is false, which is the same signal the caller already
 * treats as "derived lighting is unavailable".
 */
export function planSvoNodeMipAddresses(options: SvoNodeMipAddressPlanOptions): SvoNodeMipAddressPlan {
  return completeCooperativeBuild(planSvoNodeMipAddressesSteps(options));
}

/**
 * The same decision, offered as slices.
 *
 * One pass over the seeds finds the scene's distinct pages — each seed walks up
 * until it meets a page already found — and one pass over those pages gives
 * each its slot. Nothing is sorted and nothing is keyed by a string, so a scene
 * of 145k seeds plans in tens of milliseconds. Every yield is in the
 * derived-lighting *prologue*, which owns no device resource.
 */
export function* planSvoNodeMipAddressesSteps(
  options: SvoNodeMipAddressPlanOptions,
): Generator<unknown, SvoNodeMipAddressPlan, undefined> {
  positiveDimensions(options.basePageDimensions);
  if (!Number.isSafeInteger(options.addressCapacity) || options.addressCapacity <= 0) {
    throw new RangeError("Node-mip address capacity must be a positive safe integer");
  }
  const generation = options.generation ?? 1;
  const levelCount = options.levelCount;
  const levelDimensions = svoNodeMipLevelPageDimensions(options.basePageDimensions, levelCount);
  const opacityFloorLevel = Math.max(0, Math.min(levelCount - 1, Math.trunc(options.opacityFloorLevel ?? 0)));
  if (levelDimensions[opacityFloorLevel].some((component) => component > COORDINATE_LIMIT)) {
    throw new RangeError(`Node-mip page grid ${levelDimensions[opacityFloorLevel].join("x")} exceeds the page hash's ${SVO_NODE_MIP_PAGE_HASH.coordinateBits}-bit coordinates`);
  }
  // Every count below is over the levels that can hold a page. Below the floor
  // there are none, so a floored domain is the floor level's grid and the sum
  // above it — not the base grid, which no longer describes anything resident.
  const denseLevelCounts = levelDimensions.map((dimensions, level) => (level < opacityFloorLevel
    ? 0 : dimensions[0] * dimensions[1] * dimensions[2]));
  const domainBasePageCount = denseLevelCounts[opacityFloorLevel];
  const domainPyramidPageCount = denseLevelCounts.reduce((count, dense) => count + dense, 0);
  const maximumReserveBytes = options.maximumReserveBytes ?? SVO_NODE_MIP_ADDRESS_PLAN.maximumReserveBytes;
  const reserveBudgetPages = Math.max(0, Math.floor(maximumReserveBytes / SVO_NODE_MIP_ADDRESS_PLAN.reservedBytesPerPage));

  let discovery = new PageDiscovery(options.occupiedBasePages.length, levelCount, opacityFloorLevel);
  let visited = 0;
  for (const seed of options.occupiedBasePages) {
    const { level, coordinate } = svoNodeMipSeedPage(seed);
    if (coordinate.length !== 3 || coordinate.some((component) => !Number.isSafeInteger(component) || component < 0)) {
      throw new RangeError("Occupied SVO node-mip page must contain three non-negative safe integers");
    }
    discovery.seed(level, coordinate[0], coordinate[1], coordinate[2]);
    if ((visited += 1) % NODE_MIP_ADDRESS_YIELD_BATCH === 0) yield;
  }
  const occupiedPageCount = discovery.count;

  const reservePages = Math.min(reserveBudgetPages, Math.max(0, domainPyramidPageCount - occupiedPageCount));
  const residentPageLimit = Math.max(1, Math.min(options.addressCapacity, occupiedPageCount + reservePages));
  const total = residentPageLimit >= domainPyramidPageCount;
  if (total) {
    // The plan is the domain: every page at the floor is claimed, and with it
    // every page above. Bounded by the reserve that just admitted it.
    discovery = new PageDiscovery(domainBasePageCount, levelCount, opacityFloorLevel);
    const [width, height, depth] = levelDimensions[opacityFloorLevel];
    for (let z = 0; z < depth; z += 1) {
      for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) discovery.seed(opacityFloorLevel, x, y, z);
      yield;
    }
  }
  // Never below the opacity floor: the radiance atlas rides the opacity
  // pyramid's slot numbering, so a radiance base at a level that owns no page
  // would ask the builder to seed a chain from records that do not exist. The
  // anchored radiance floor is three levels up at the reference leaf and rises
  // with refinement, so this clamp only ever binds when both levers are forced.
  const radianceFloorLevel = Math.max(opacityFloorLevel, svoRadianceLevelFloor({
    levelCount, cellSize_m: options.cellSize_m, override: options.radianceFloorLevel,
  }));
  const planLevelCounts = new Array<number>(levelCount).fill(0);
  let opacityOnlyPages = 0;
  for (let index = 0; index < discovery.count; index += 1) {
    const level = discovery.pages[index * 4 + 3];
    planLevelCounts[level] += 1;
    if (level < radianceFloorLevel) opacityOnlyPages += 1;
  }
  const fits = discovery.count <= residentPageLimit;
  const headroom = Math.max(0, residentPageLimit - discovery.count);
  const denseBelow = denseLevelCounts.slice(0, radianceFloorLevel).reduce((count, dense) => count + dense, 0);
  const denseAbove = denseLevelCounts.slice(radianceFloorLevel).reduce((count, dense) => count + dense, 0);
  // Each pool may take the whole headroom, bounded by its own levels' grids;
  // see the module comment. The radiance pool is never empty so that its atlas
  // exists, and the two together never pass what the device can address.
  const radiancePageCapacity = Math.max(1, Math.min(denseAbove, discovery.count - opacityOnlyPages + headroom,
    options.addressCapacity));
  const opacityOnlyPageCapacity = Math.max(0, Math.min(denseBelow, opacityOnlyPages + headroom,
    options.addressCapacity - radiancePageCapacity));
  const denseLevelMaximum = Math.max(1, ...denseLevelCounts);
  const planLevelMaximum = Math.max(1, ...planLevelCounts);
  // A total plan is already the whole grid, and a partial one can grow by at
  // most the shared `reservePages` — all of which could land on any one level.
  const pageCapacityByLevel = denseLevelCounts.map((dense, level) => Math.max(1, Math.min(dense,
    total ? dense : planLevelCounts[level] + reservePages)));
  const pageCapacityPerLevel = Math.max(1, Math.min(denseLevelMaximum,
    total ? denseLevelMaximum : planLevelMaximum + reservePages));
  const radiancePageCapacityPerLevel = Math.max(1, Math.min(pageCapacityPerLevel,
    Math.max(1, ...denseLevelCounts.slice(radianceFloorLevel))));
  yield;
  return new SvoNodeMipAddressPlan({
    generation, levelCount, basePageDimensions: options.basePageDimensions, levelDimensions,
    opacityFloorLevel, radianceFloorLevel,
    requestedPageCount: discovery.count, residentPageLimit, opacityOnlyPageCapacity, radiancePageCapacity,
    total, domainBasePageCount, domainPyramidPageCount, reservePages,
    pageCapacityPerLevel, pageCapacityByLevel, maximumReserveBytes, radiancePageCapacityPerLevel,
  }, fits && opacityOnlyPages <= opacityOnlyPageCapacity
    && discovery.count - opacityOnlyPages <= radiancePageCapacity ? discovery : undefined);
}
