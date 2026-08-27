import { signedSpatialCoordinateHash } from "./signed-spatial-hash";
import type { SolidWorld, SolidWorldPage } from "./solid-world";

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
  readonly totalWords: number;
  readonly totalBytes: number;
}

const nextPowerOfTwo = (value: number): number => {
  let result = 1;
  while (result < value) result *= 2;
  return result;
};

export function createWebgpuSolidWorldPageLayout(options: {
  readonly baseWords: number;
  readonly authoredPageCount: number;
  readonly authoredRegionCount?: number;
  readonly includesMaterial?: boolean;
  readonly maximumBytes?: number;
}): WebgpuSolidWorldPageLayout {
  if (!Number.isSafeInteger(options.baseWords) || options.baseWords < 0
    || !Number.isSafeInteger(options.authoredPageCount)
    || options.authoredPageCount < 0) {
    throw new RangeError("SolidWorld GPU page capacities are invalid");
  }
  const regionCapacity = options.authoredRegionCount ?? 0;
  if (!Number.isSafeInteger(regionCapacity) || regionCapacity < 0) {
    throw new RangeError("SolidWorld GPU region capacity is invalid");
  }
  const pageCapacity = nextPowerOfTwo(Math.max(16,
    Math.ceil(1.5 * options.authoredPageCount)));
  const directoryCapacity = nextPowerOfTwo(2 * pageCapacity);
  const regionBaseWords = WEBGPU_SOLID_WORLD_HEADER_WORDS;
  const directoryBaseWords = regionBaseWords
    + regionCapacity * WEBGPU_SOLID_WORLD_REGION_WORDS;
  const pageBaseWords = directoryBaseWords
    + directoryCapacity * WEBGPU_SOLID_WORLD_ENTRY_WORDS;
  const includesMaterial = options.includesMaterial === true;
  const pageWords = includesMaterial ? WEBGPU_SOLID_WORLD_RENDER_PAGE_WORDS
    : WEBGPU_SOLID_WORLD_PHYSICS_PAGE_WORDS;
  const localWords = pageBaseWords + pageCapacity * pageWords;
  const maximumBytes = options.maximumBytes ?? 64 * 1024 * 1024;
  if (4 * localWords > maximumBytes) {
    throw new RangeError(`SolidWorld GPU image requires ${4 * localWords} bytes; `
      + `the fixed budget permits ${maximumBytes}`);
  }
  return Object.freeze({ baseWords: options.baseWords, pageCapacity,
    regionCapacity, regionBaseWords,
    directoryCapacity, directoryBaseWords, pageBaseWords, pageWords,
    includesMaterial, totalWords: options.baseWords + localWords,
    totalBytes: 4 * (options.baseWords + localWords) });
}

function packPage(destination: Uint32Array, page: SolidWorldPage,
  includesMaterial: boolean): void {
  destination.fill(0);
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

function header(layout: WebgpuSolidWorldPageLayout, pageCount: number,
  originFine: readonly [number, number, number],
  lattice?: WebgpuSolidWorldPhysicalLattice): Uint32Array<ArrayBuffer> {
  const words = new Uint32Array(WEBGPU_SOLID_WORLD_HEADER_WORDS);
  words.set([WEBGPU_SOLID_WORLD_MAGIC, WEBGPU_SOLID_WORLD_VERSION,
    layout.directoryCapacity, layout.directoryCapacity - 1,
    layout.directoryBaseWords, layout.pageBaseWords, layout.pageCapacity,
    pageCount, originFine[0] >>> 0, originFine[1] >>> 0, originFine[2] >>> 0,
    WEBGPU_SOLID_WORLD_ENTRY_WORDS, layout.pageWords,
    layout.includesMaterial ? 1 : 0, layout.regionCapacity,
    layout.regionBaseWords]);
  if (lattice) {
    new Float32Array(words.buffer).set(lattice.origin_m, 16);
    new Float32Array(words.buffer).set(lattice.cellSize_m, 20);
  }
  return words;
}

/** CPU oracle/fixture. Production publication should use the streaming writer. */
export function packWebgpuSolidWorldPages(layout: WebgpuSolidWorldPageLayout,
  world: SolidWorld, originFine: readonly [number, number, number],
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
    packPage(words.subarray(at, at + layout.pageWords), page, layout.includesMaterial);
  }
  return words;
}

/** Stream one canonical sparse page at a time; never builds an arena-sized host image. */
export function writeWebgpuSolidWorldPages(queue: GPUQueue, destination: GPUBuffer,
  layout: WebgpuSolidWorldPageLayout, world: SolidWorld,
  originFine: readonly [number, number, number],
  lattice?: WebgpuSolidWorldPhysicalLattice): void {
  if (world.pages.length > layout.pageCapacity) throw new RangeError(
    `SolidWorld has ${world.pages.length} pages; GPU capacity is ${layout.pageCapacity}`);
  queue.writeBuffer(destination, 4 * layout.baseWords,
    header(layout, world.pages.length, originFine, lattice));
  const occupied = new Uint8Array(layout.directoryCapacity);
  const entry = new Uint32Array(WEBGPU_SOLID_WORLD_ENTRY_WORDS);
  const payload = new Uint32Array(layout.pageWords);
  for (let pageIndex = 0; pageIndex < world.pages.length; pageIndex += 1) {
    const page = world.pages[pageIndex]!;
    const hash = signedSpatialCoordinateHash(page.coordinate);
    let slot = hash & (layout.directoryCapacity - 1);
    while (occupied[slot]) slot = (slot + 1) & (layout.directoryCapacity - 1);
    occupied[slot] = 1;
    entry.set([2, hash, page.coordinate[0] >>> 0, page.coordinate[1] >>> 0,
      page.coordinate[2] >>> 0, pageIndex]);
    queue.writeBuffer(destination, 4 * (layout.baseWords + layout.directoryBaseWords
      + slot * WEBGPU_SOLID_WORLD_ENTRY_WORDS), entry);
    packPage(payload, page, layout.includesMaterial);
    queue.writeBuffer(destination, 4 * (layout.baseWords + layout.pageBaseWords
      + pageIndex * layout.pageWords), payload);
  }
}
