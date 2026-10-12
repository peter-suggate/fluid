import { signedSpatialCoordinateHash } from "./signed-spatial-hash";
import { SOLID_PATCH_BOX_KIND, solidPatchKindId } from "./solid-patch/kind";
import type { SolidWorld, SolidWorldPage } from "./solid-world";
import { uploadChangedWords } from "../svo/features/scene-publication/svo-table-upload";

export const WEBGPU_SOLID_WORLD_MAGIC = 0x534f_4331;
export const WEBGPU_SOLID_WORLD_VERSION = 2;
export const WEBGPU_SOLID_WORLD_HEADER_WORDS = 24;
export const WEBGPU_SOLID_WORLD_ENTRY_WORDS = 6;
export const WEBGPU_SOLID_WORLD_REGION_WORDS = 8;
export const WEBGPU_SOLID_WORLD_FRACTION_WORDS = 128;
export const WEBGPU_SOLID_WORLD_SDF_WORDS = 256;
export const WEBGPU_SOLID_WORLD_MATERIAL_WORDS = 256;
export const WEBGPU_SOLID_WORLD_PHYSICS_PAGE_WORDS =
  WEBGPU_SOLID_WORLD_FRACTION_WORDS + WEBGPU_SOLID_WORLD_SDF_WORDS;
export const WEBGPU_SOLID_WORLD_RENDER_PAGE_WORDS =
  WEBGPU_SOLID_WORLD_PHYSICS_PAGE_WORDS + WEBGPU_SOLID_WORLD_MATERIAL_WORDS;
/**
 * One bit a voxel, bit `x + 8 (y + 8 z)` low bit first: set where an edit
 * wrote the cell. A terrain scene draws those cells from their page and every
 * other cell from its analytic ground, so the lane follows the render lanes.
 */
export const WEBGPU_SOLID_WORLD_AUTHORED_WORDS = 16;

export interface WebgpuSolidWorldPageLayout {
  readonly baseWords: number;
  readonly pageCapacity: number;
  readonly directoryCapacity: number;
  readonly regionCapacity: number;
  readonly regionBaseWords: number;
  readonly directoryBaseWords: number;
  readonly pageBaseWords: number;
  readonly pageWords: number;
  readonly includesMaterial: boolean;
  readonly includesAuthored: boolean;
  readonly totalWords: number;
  readonly totalBytes: number;
}

const nextPowerOfTwo = (value: number): number => {
  let result = 1;
  while (result < value) result *= 2;
  return result;
};

/**
 * The most a display page image may reserve: its pages, its directory and its
 * header. The image is sized once, when the scene is built, and never grows,
 * so this is what an edit session can ever draw as pages. A scene whose
 * editable domain needs less reserves only that. 64 MiB is 25,599 render pages
 * (2,560 bytes each, plus a directory at no more than half load), which is
 * every page of the editor's default scene two and a half times over.
 */
export const WEBGPU_SOLID_WORLD_DISPLAY_BUDGET_BYTES = 64 * 1024 * 1024;

const pageWordsFor = (includesMaterial: boolean, includesAuthored = false): number => (includesMaterial
  ? WEBGPU_SOLID_WORLD_RENDER_PAGE_WORDS : WEBGPU_SOLID_WORLD_PHYSICS_PAGE_WORDS)
  + (includesAuthored ? WEBGPU_SOLID_WORLD_AUTHORED_WORDS : 0);

/** Bytes an image of `pageCapacity` pages and `regionCapacity` regions takes after its base. */
const imageBytes = (pageCapacity: number, regionCapacity: number, includesMaterial: boolean,
  includesAuthored = false): number =>
  4 * (WEBGPU_SOLID_WORLD_HEADER_WORDS + regionCapacity * WEBGPU_SOLID_WORLD_REGION_WORDS
    + nextPowerOfTwo(2 * pageCapacity) * WEBGPU_SOLID_WORLD_ENTRY_WORDS
    + pageCapacity * pageWordsFor(includesMaterial, includesAuthored));

/**
 * Pages a display image reserves for a scene, decided before anything is
 * allocated: every page its editable domain holds, up to what the budget
 * holds, and never fewer than the scene is built with.
 *
 * A built world larger than the budget is returned as it is, and
 * `createWebgpuSolidWorldPageLayout` then refuses it by name.
 */
export function planWebgpuSolidWorldDisplayPageCapacity(options: {
  /** Pages the scene holds when it is built. */
  readonly builtPageCount: number;
  /** Pages of the solid lattice the editable domain touches. */
  readonly domainPageCount: number;
  readonly includesMaterial?: boolean;
  readonly includesAuthored?: boolean;
  readonly budgetBytes?: number;
}): number {
  const { builtPageCount, domainPageCount } = options;
  if (!Number.isSafeInteger(builtPageCount) || builtPageCount < 0
    || !Number.isSafeInteger(domainPageCount) || domainPageCount < 0) {
    throw new RangeError("SolidWorld display page counts are invalid");
  }
  const includesMaterial = options.includesMaterial === true;
  const includesAuthored = options.includesAuthored === true;
  const budgetBytes = options.budgetBytes ?? WEBGPU_SOLID_WORLD_DISPLAY_BUDGET_BYTES;
  // The directory doubles at powers of two, so the largest count that fits is searched for.
  let low = 0, high = Math.max(0, Math.floor(budgetBytes / (4 * pageWordsFor(includesMaterial, includesAuthored))));
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (imageBytes(middle, 0, includesMaterial, includesAuthored) <= budgetBytes) low = middle;
    else high = middle - 1;
  }
  return Math.max(1, builtPageCount, Math.min(domainPageCount, low));
}

export function createWebgpuSolidWorldPageLayout(options: {
  readonly baseWords: number;
  readonly authoredPageCount: number;
  /**
   * The exact capacity to reserve, as `planWebgpuSolidWorldDisplayPageCapacity`
   * decides it. Without it the image holds half as many pages again as it is
   * authored with, rounded up to a power of two, which is all a fixture needs.
   */
  readonly pageCapacity?: number;
  readonly authoredRegionCount?: number;
  readonly includesMaterial?: boolean;
  /** Adds the authored lane after the material lane, which it therefore needs. */
  readonly includesAuthored?: boolean;
  readonly maximumBytes?: number;
}): WebgpuSolidWorldPageLayout {
  if (!Number.isSafeInteger(options.baseWords) || options.baseWords < 0
    || !Number.isSafeInteger(options.authoredPageCount)
    || options.authoredPageCount < 0) {
    throw new RangeError("SolidWorld GPU page capacities are invalid");
  }
  if (options.pageCapacity !== undefined && (!Number.isSafeInteger(options.pageCapacity)
    || options.pageCapacity < Math.max(1, options.authoredPageCount))) {
    throw new RangeError(`SolidWorld GPU page capacity ${String(options.pageCapacity)} cannot hold `
      + `the ${options.authoredPageCount} pages the scene is built with`);
  }
  const regionCapacity = options.authoredRegionCount ?? 0;
  if (!Number.isSafeInteger(regionCapacity) || regionCapacity < 0) {
    throw new RangeError("SolidWorld GPU region capacity is invalid");
  }
  const pageCapacity = options.pageCapacity ?? nextPowerOfTwo(Math.max(16,
    Math.ceil(1.5 * options.authoredPageCount)));
  const directoryCapacity = nextPowerOfTwo(2 * pageCapacity);
  const regionBaseWords = WEBGPU_SOLID_WORLD_HEADER_WORDS;
  const directoryBaseWords = regionBaseWords
    + regionCapacity * WEBGPU_SOLID_WORLD_REGION_WORDS;
  const pageBaseWords = directoryBaseWords
    + directoryCapacity * WEBGPU_SOLID_WORLD_ENTRY_WORDS;
  const includesMaterial = options.includesMaterial === true;
  const includesAuthored = options.includesAuthored === true;
  if (includesAuthored && !includesMaterial) {
    throw new RangeError("SolidWorld GPU authored lane follows the material lane; the image must include both");
  }
  const pageWords = pageWordsFor(includesMaterial, includesAuthored);
  const localWords = pageBaseWords + pageCapacity * pageWords;
  const maximumBytes = options.maximumBytes ?? WEBGPU_SOLID_WORLD_DISPLAY_BUDGET_BYTES;
  if (4 * localWords > maximumBytes) {
    throw new RangeError(`SolidWorld GPU image of ${pageCapacity} pages requires ${4 * localWords} bytes; `
      + `the fixed budget permits ${maximumBytes}`);
  }
  return Object.freeze({ baseWords: options.baseWords, pageCapacity,
    regionCapacity, regionBaseWords,
    directoryCapacity, directoryBaseWords, pageBaseWords, pageWords,
    includesMaterial, includesAuthored, totalWords: options.baseWords + localWords,
    totalBytes: 4 * (options.baseWords + localWords) });
}

function packPage(destination: Uint32Array, page: SolidWorldPage,
  includesMaterial: boolean, includesAuthored = false): void {
  destination.fill(0);
  if (includesAuthored && page.authored) {
    for (let byte = 0; byte < page.authored.length; byte += 1) {
      destination[WEBGPU_SOLID_WORLD_RENDER_PAGE_WORDS + (byte >>> 2)]! |= page.authored[byte]! << (8 * (byte & 3));
    }
  }
  for (let voxel = 0; voxel < page.solidFraction.length; voxel += 1) {
    destination[voxel >>> 2]! |= page.solidFraction[voxel]! << (8 * (voxel & 3));
    const sdfAt = WEBGPU_SOLID_WORLD_FRACTION_WORDS + (voxel >>> 1);
    destination[sdfAt]! |= (page.signedDistanceQ8[voxel]! & 0xffff)
      << (16 * (voxel & 1));
    if (includesMaterial) {
      const materialAt = WEBGPU_SOLID_WORLD_PHYSICS_PAGE_WORDS + (voxel >>> 1);
      destination[materialAt]! |= page.materialId[voxel]! << (16 * (voxel & 1));
    }
  }
}

export interface WebgpuSolidWorldPhysicalLattice {
  readonly origin_m: readonly [number, number, number];
  readonly cellSize_m: readonly [number, number, number];
}

/**
 * Pages of a solid lattice that a world box touches: every page an edit kept
 * inside that box can make solid. The box need not sit on page faces; a page
 * it only grazes, by less than a millionth of a page, is not counted.
 */
export function webgpuSolidWorldDomainPageCount(lattice: WebgpuSolidWorldPhysicalLattice,
  minimum_m: readonly [number, number, number], maximum_m: readonly [number, number, number]): number {
  let pages = 1;
  for (let axis = 0; axis < 3; axis += 1) {
    const pageEdge = 8 * lattice.cellSize_m[axis]!;
    const first = Math.floor((minimum_m[axis]! - lattice.origin_m[axis]!) / pageEdge + 1e-6);
    const last = Math.ceil((maximum_m[axis]! - lattice.origin_m[axis]!) / pageEdge - 1e-6) - 1;
    if (!Number.isFinite(first) || !Number.isFinite(last)) {
      throw new RangeError("SolidWorld display domain is not a finite box");
    }
    pages *= Math.max(0, last - first + 1);
  }
  return pages;
}

function header(layout: WebgpuSolidWorldPageLayout, pageCount: number,
  originFine: readonly [number, number, number],
  lattice?: WebgpuSolidWorldPhysicalLattice): Uint32Array<ArrayBuffer> {
  const words = new Uint32Array(WEBGPU_SOLID_WORLD_HEADER_WORDS);
  words.set([WEBGPU_SOLID_WORLD_MAGIC, WEBGPU_SOLID_WORLD_VERSION,
    layout.directoryCapacity, layout.directoryCapacity - 1,
    layout.directoryBaseWords, layout.pageBaseWords, layout.pageCapacity,
    pageCount, originFine[0] >>> 0, originFine[1] >>> 0, originFine[2] >>> 0,
    WEBGPU_SOLID_WORLD_ENTRY_WORDS, layout.pageWords,
    (layout.includesMaterial ? 1 : 0) | (layout.includesAuthored ? 2 : 0), layout.regionCapacity,
    layout.regionBaseWords]);
  if (lattice) {
    new Float32Array(words.buffer).set(lattice.origin_m, 16);
    new Float32Array(words.buffer).set(lattice.cellSize_m, 20);
  }
  return words;
}

/** CPU oracle/fixture. Production publication uses `WebgpuSolidWorldPageWriter`. */
export function packWebgpuSolidWorldPages(layout: WebgpuSolidWorldPageLayout,
  world: Pick<SolidWorld, "pages" | "regions">, originFine: readonly [number, number, number],
  lattice?: WebgpuSolidWorldPhysicalLattice): Uint32Array {
  if (world.pages.length > layout.pageCapacity) throw new RangeError(
    `SolidWorld has ${world.pages.length} pages; GPU capacity is ${layout.pageCapacity}`);
  const words = new Uint32Array(layout.totalWords - layout.baseWords);
  words.set(header(layout, world.pages.length, originFine, lattice));
  const regions = world.regions ?? [];
  if (regions.length > layout.regionCapacity) throw new RangeError(
    `SolidWorld has ${regions.length} regions; GPU capacity is ${layout.regionCapacity}`);
  for (let index = 0; index < regions.length; index += 1) {
    const region = regions[index]!;
    // A region record is a box that fills or clears. It has no word for a
    // kind or a repaint, and packing either as a clear would draw a hole.
    if (region.operation === "paint" || solidPatchKindId(region) !== SOLID_PATCH_BOX_KIND) {
      throw new RangeError(`SolidWorld region ${index} is a ${solidPatchKindId(region)} ${region.operation}; `
        + "a GPU region record holds only a box fill or a box clear");
    }
    words.set([region.operation === "fill" ? 1 : 0,
      region.minimum[0] >>> 0, region.minimum[1] >>> 0, region.minimum[2] >>> 0,
      region.maximumExclusive[0] >>> 0, region.maximumExclusive[1] >>> 0,
      region.maximumExclusive[2] >>> 0, region.materialId ?? 1],
    layout.regionBaseWords + index * WEBGPU_SOLID_WORLD_REGION_WORDS);
  }
  const occupied = new Uint8Array(layout.directoryCapacity);
  for (let pageIndex = 0; pageIndex < world.pages.length; pageIndex += 1) {
    const page = world.pages[pageIndex]!;
    const hash = signedSpatialCoordinateHash(page.coordinate);
    let slot = hash & (layout.directoryCapacity - 1);
    while (occupied[slot]) slot = (slot + 1) & (layout.directoryCapacity - 1);
    occupied[slot] = 1;
    words.set([2, hash, page.coordinate[0] >>> 0, page.coordinate[1] >>> 0,
      page.coordinate[2] >>> 0, pageIndex], layout.directoryBaseWords
      + slot * WEBGPU_SOLID_WORLD_ENTRY_WORDS);
    const at = layout.pageBaseWords + pageIndex * layout.pageWords;
    packPage(words.subarray(at, at + layout.pageWords), page, layout.includesMaterial, layout.includesAuthored);
  }
  return words;
}

const samePageLanes = (left: SolidWorldPage, right: SolidWorldPage, layout: WebgpuSolidWorldPageLayout): boolean => {
  const { solidFraction, signedDistanceQ8, materialId } = left;
  for (let voxel = 0; voxel < solidFraction.length; voxel += 1) {
    if (solidFraction[voxel] !== right.solidFraction[voxel]
      || signedDistanceQ8[voxel] !== right.signedDistanceQ8[voxel]
      || (layout.includesMaterial && materialId[voxel] !== right.materialId[voxel])) return false;
  }
  if (!layout.includesAuthored) return true;
  for (let byte = 0; byte < WEBGPU_SOLID_WORLD_AUTHORED_WORDS * 4; byte += 1) {
    if ((left.authored?.[byte] ?? 0) !== (right.authored?.[byte] ?? 0)) return false;
  }
  return true;
};

/**
 * The page image as a device holds it, kept in step with successive page sets.
 *
 * A page keeps the payload slot it was first given for as long as its
 * coordinate stays in the set, so a set that differs from the last by `k`
 * pages writes `k` payloads and the directory entries those pages own: nothing
 * about the other pages moves. Slots a set drops return to a free list. The
 * directory is open addressing with linear probing, and a removal closes the
 * gap it leaves (backward shift), so no tombstones build up and a probe is
 * never longer than a from-scratch pack would make it.
 *
 * The capacity is the layout's and is checked before the first write of a
 * set, so a refused set leaves the device image as it was.
 */
export class WebgpuSolidWorldPageWriter {
  /** The directory as last written: `WEBGPU_SOLID_WORLD_ENTRY_WORDS` an entry. */
  private readonly directory: Uint32Array;
  private readonly slotPage: (SolidWorldPage | undefined)[];
  private readonly slotOfPage = new Map<SolidWorldPage, number>();
  /** The write that last found each slot's page in its set. */
  private readonly slotSeen: Uint32Array;
  private readonly freeSlots: Uint32Array;
  private freeCount: number;
  private generation = 0;
  private liveCount = 0;
  private headerCount = -1;
  private readonly pending: SolidWorldPage[] = [];
  private readonly pendingSlot: number[] = [];
  private readonly entry = new Uint32Array(WEBGPU_SOLID_WORLD_ENTRY_WORDS);
  private readonly payload: Uint32Array<ArrayBuffer>;
  /** The page a slot held before the one being written, packed the same way. */
  private readonly heldPayload: Uint32Array;

  constructor(private readonly queue: Pick<GPUQueue, "writeBuffer">, private readonly destination: GPUBuffer,
    readonly layout: WebgpuSolidWorldPageLayout,
    private readonly originFine: readonly [number, number, number],
    private readonly lattice?: WebgpuSolidWorldPhysicalLattice) {
    this.directory = new Uint32Array(layout.directoryCapacity * WEBGPU_SOLID_WORLD_ENTRY_WORDS);
    this.slotPage = new Array<SolidWorldPage | undefined>(layout.pageCapacity).fill(undefined);
    this.slotSeen = new Uint32Array(layout.pageCapacity);
    this.freeSlots = new Uint32Array(layout.pageCapacity);
    // Popped from the top, so a first set takes slots in the order it lists its pages.
    for (let slot = 0; slot < layout.pageCapacity; slot += 1) this.freeSlots[slot] = layout.pageCapacity - 1 - slot;
    this.freeCount = layout.pageCapacity;
    this.payload = new Uint32Array(layout.pageWords);
    this.heldPayload = new Uint32Array(layout.pageWords);
  }

  /** Pages the device image holds. */
  get pageCount(): number { return this.liveCount; }

  /** The payload slot of a page of the last set, or -1. */
  slotOf(page: SolidWorldPage): number { return this.slotOfPage.get(page) ?? -1; }

  /** Directory entry holding this page coordinate, or -1. */
  private entryOf(coordinate: readonly [number, number, number], hash: number): number {
    const mask = this.layout.directoryCapacity - 1, { directory } = this;
    for (let at = hash & mask; ; at = (at + 1) & mask) {
      const base = at * WEBGPU_SOLID_WORLD_ENTRY_WORDS;
      if (directory[base] === 0) return -1;
      if (directory[base + 1] === hash && directory[base + 2] === coordinate[0] >>> 0
        && directory[base + 3] === coordinate[1] >>> 0 && directory[base + 4] === coordinate[2] >>> 0) return at;
    }
  }

  private writeEntry(at: number): void {
    const base = at * WEBGPU_SOLID_WORLD_ENTRY_WORDS;
    this.entry.set(this.directory.subarray(base, base + WEBGPU_SOLID_WORLD_ENTRY_WORDS));
    this.queue.writeBuffer(this.destination, 4 * (this.layout.baseWords + this.layout.directoryBaseWords + base),
      this.entry);
  }

  /** Empty one entry and move back every later entry of its run that probed past it. */
  private removeEntry(removed: number): void {
    const mask = this.layout.directoryCapacity - 1, { directory } = this, E = WEBGPU_SOLID_WORLD_ENTRY_WORDS;
    let gap = removed;
    for (let at = (gap + 1) & mask; directory[at * E] !== 0; at = (at + 1) & mask) {
      const home = directory[at * E + 1]! & mask;
      // An entry whose home lies cyclically in (gap, at] is still reachable past the gap.
      if (gap <= at ? (gap < home && home <= at) : (gap < home || home <= at)) continue;
      directory.copyWithin(gap * E, at * E, at * E + E);
      this.writeEntry(gap);
      gap = at;
    }
    directory.fill(0, gap * E, gap * E + E);
    this.writeEntry(gap);
  }

  /**
   * Bring the device image to exactly `pages`: one page a coordinate.
   * Returns the payloads uploaded.
   */
  write(pages: readonly SolidWorldPage[]): number {
    const { layout, queue, destination, pending, pendingSlot, slotPage, slotSeen, slotOfPage } = this;
    if (pages.length > layout.pageCapacity) throw new RangeError(
      `SolidWorld has ${pages.length} pages; GPU capacity is ${layout.pageCapacity}`);
    this.generation += 1;
    const generation = this.generation;
    pending.length = 0; pendingSlot.length = 0;
    for (const page of pages) {
      const slot = slotOfPage.get(page);
      if (slot === undefined) pending.push(page);
      else if (slotSeen[slot] === generation) throw new RangeError(
        `SolidWorld lists page ${page.coordinate.join(",")} twice`);
      else slotSeen[slot] = generation;
    }
    // A page not held by identity is either new or another object at a held coordinate.
    let kept = pages.length - pending.length;
    for (const page of pending) {
      const at = this.entryOf(page.coordinate, signedSpatialCoordinateHash(page.coordinate));
      const slot = at < 0 ? -1 : this.directory[at * WEBGPU_SOLID_WORLD_ENTRY_WORDS + 5]!;
      if (slot >= 0) {
        if (slotSeen[slot] === generation) throw new RangeError(
          `SolidWorld lists page ${page.coordinate.join(",")} twice`);
        slotSeen[slot] = generation; kept += 1;
      }
      pendingSlot.push(slot);
    }
    if (kept < this.liveCount) {
      for (let slot = 0; slot < layout.pageCapacity; slot += 1) {
        const page = slotPage[slot];
        if (!page || slotSeen[slot] === generation) continue;
        this.removeEntry(this.entryOf(page.coordinate, signedSpatialCoordinateHash(page.coordinate)));
        slotOfPage.delete(page); slotPage[slot] = undefined;
        this.freeSlots[this.freeCount] = slot; this.freeCount += 1; this.liveCount -= 1;
      }
    }
    let uploads = 0;
    for (let index = 0; index < pending.length; index += 1) {
      const page = pending[index]!;
      let slot = pendingSlot[index]!, changed = true, replaces = false;
      if (slot < 0) {
        this.freeCount -= 1; slot = this.freeSlots[this.freeCount]!; this.liveCount += 1;
        const hash = signedSpatialCoordinateHash(page.coordinate), mask = layout.directoryCapacity - 1;
        let at = hash & mask;
        while (this.directory[at * WEBGPU_SOLID_WORLD_ENTRY_WORDS] !== 0) at = (at + 1) & mask;
        this.directory.set([2, hash, page.coordinate[0] >>> 0, page.coordinate[1] >>> 0,
          page.coordinate[2] >>> 0, slot], at * WEBGPU_SOLID_WORLD_ENTRY_WORDS);
        this.writeEntry(at);
        slotSeen[slot] = generation;
      } else {
        const held = slotPage[slot]!;
        changed = !samePageLanes(held, page, layout);
        // The slot holds the held page as packed, so the words the two packings
        // differ in are all a stroke through the page has to send.
        if (changed) { packPage(this.heldPayload, held, layout.includesMaterial, layout.includesAuthored); replaces = true; }
        slotOfPage.delete(held);
      }
      slotPage[slot] = page; slotOfPage.set(page, slot);
      if (!changed) continue;
      const { payload } = this;
      packPage(payload, page, layout.includesMaterial, layout.includesAuthored);
      const at = 4 * (layout.baseWords + layout.pageBaseWords + slot * layout.pageWords);
      if (replaces) uploadChangedWords(queue, destination, at, this.heldPayload, payload);
      else queue.writeBuffer(destination, at, payload);
      uploads += 1;
    }
    pending.length = 0;
    if (this.headerCount !== this.liveCount) {
      this.headerCount = this.liveCount;
      queue.writeBuffer(destination, 4 * layout.baseWords,
        header(layout, this.liveCount, this.originFine, this.lattice));
    }
    return uploads;
  }
}
