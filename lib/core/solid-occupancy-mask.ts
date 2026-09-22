import { SOLID_WORLD_BRICK_CELLS, type SolidWorld, type SolidWorldPage,
  type SolidWorldVoxelPatch } from "./solid-world";

export const SOLID_OCCUPANCY_MASK_MAGIC = 0x53565731;
export const SOLID_OCCUPANCY_MASK_HEADER_WORDS = 4;

/** Word range of a mask an update rewrote; the rest of the mask is unchanged. */
export interface SolidOccupancyMaskDirtyRange {
  readonly firstWord: number;
  readonly wordCount: number;
}

/**
 * One bit per cell of a dense lattice with a one-cell halo: bit
 * `(x+1) + sx*((y+1) + sy*(z+1))` is set where `sampleSolidWorld` is solid.
 *
 * The mask is built from the world's pages, never by sampling the lattice, and
 * a later world is folded in by the pages it does not share with the previous
 * one. A voxel stroke copies only the pages it writes, so an edit costs its
 * own pages rather than the domain.
 */
export class SolidOccupancyMask {
  readonly words: Uint32Array;
  private readonly size: readonly [number, number, number];
  private world: SolidWorld | undefined;

  constructor(dimensions: readonly [number, number, number]) {
    this.size = [dimensions[0] + 2, dimensions[1] + 2, dimensions[2] + 2];
    this.words = new Uint32Array(SOLID_OCCUPANCY_MASK_HEADER_WORDS
      + Math.ceil(this.size[0] * this.size[1] * this.size[2] / 32));
    this.words.set([SOLID_OCCUPANCY_MASK_MAGIC, ...this.size]);
  }

  /** Make the mask describe `next`; undefined when no bit changed. */
  update(next: SolidWorld): SolidOccupancyMaskDirtyRange | undefined {
    const previous = this.world;
    this.world = next;
    if (previous === next) return undefined;
    const regions = next.regions ?? [];
    const dirty = { low: Number.POSITIVE_INFINITY, high: -1 };
    if (!previous || !sameRegions(previous.regions ?? [], regions)) {
      this.words.fill(0, SOLID_OCCUPANCY_MASK_HEADER_WORDS);
      for (const page of next.pages) this.writePage(page.coordinate, page, [], dirty);
      for (const region of regions) this.writeRegion(region);
      return { firstWord: 0, wordCount: this.words.length };
    }
    const retained = new Set<SolidWorldPage>(previous.pages);
    for (const page of next.pages) if (!retained.has(page)) this.writePage(page.coordinate, page, regions, dirty);
    // A page a clear emptied leaves the world rather than staying behind empty.
    for (const page of previous.pages) {
      if (next.directory.lookup(page.coordinate) === undefined) this.writePage(page.coordinate, undefined, regions, dirty);
    }
    return dirty.high < dirty.low ? undefined
      : { firstWord: dirty.low, wordCount: dirty.high - dirty.low + 1 };
  }

  private writePage(coordinate: readonly [number, number, number], page: SolidWorldPage | undefined,
    regions: readonly SolidWorldVoxelPatch[], dirty: { low: number; high: number }): void {
    const B = SOLID_WORLD_BRICK_CELLS, [sx, sy] = this.size, words = this.words;
    const base = [coordinate[0] * B, coordinate[1] * B, coordinate[2] * B];
    const lo = base.map((value) => Math.max(0, -1 - value));
    const hi = base.map((value, axis) => Math.min(B, this.size[axis]! - 1 - value));
    if (lo.some((value, axis) => value >= hi[axis]!)) return;
    const covering = regions.filter((region) => base.every((value, axis) =>
      region.minimum[axis]! < value + B && region.maximumExclusive[axis]! > value));
    for (let z = lo[2]!; z < hi[2]!; z += 1) for (let y = lo[1]!; y < hi[1]!; y += 1) {
      const local = B * (y + B * z);
      const row = sx * ((base[1]! + y + 1) + sy * (base[2]! + z + 1)) + base[0]! + 1;
      for (let x = lo[0]!; x < hi[0]!; x += 1) {
        let solid = page !== undefined && page.solidFraction[local + x]! > 0;
        for (const region of covering) {
          if (base[0]! + x < region.minimum[0]! || base[0]! + x >= region.maximumExclusive[0]!
            || base[1]! + y < region.minimum[1]! || base[1]! + y >= region.maximumExclusive[1]!
            || base[2]! + z < region.minimum[2]! || base[2]! + z >= region.maximumExclusive[2]!) continue;
          solid = region.operation === "fill";
        }
        const index = row + x, word = SOLID_OCCUPANCY_MASK_HEADER_WORDS + (index >>> 5), bit = 1 << (index & 31);
        const before = words[word]!;
        const after = solid ? before | bit : before & ~bit;
        if (after === before) continue;
        words[word] = after;
        if (word < dirty.low) dirty.low = word;
        if (word > dirty.high) dirty.high = word;
      }
    }
  }

  private writeRegion(region: SolidWorldVoxelPatch): void {
    const [sx, sy] = this.size, words = this.words, fill = region.operation === "fill";
    const lo = region.minimum.map((value) => Math.max(-1, value));
    const hi = region.maximumExclusive.map((value, axis) => Math.min(this.size[axis]! - 1, value));
    for (let z = lo[2]!; z < hi[2]!; z += 1) for (let y = lo[1]!; y < hi[1]!; y += 1) {
      const row = sx * ((y + 1) + sy * (z + 1)) + 1;
      for (let x = lo[0]!; x < hi[0]!; x += 1) {
        const index = row + x, word = SOLID_OCCUPANCY_MASK_HEADER_WORDS + (index >>> 5), bit = 1 << (index & 31);
        words[word] = fill ? words[word]! | bit : words[word]! & ~bit;
      }
    }
  }
}

function sameRegions(left: readonly SolidWorldVoxelPatch[], right: readonly SolidWorldVoxelPatch[]): boolean {
  return left === right || (left.length === right.length && left.every((region, index) => {
    const other = right[index]!;
    return region === other || (region.operation === other.operation
      && (region.materialId ?? 1) === (other.materialId ?? 1)
      && region.minimum.every((value, axis) => value === other.minimum[axis])
      && region.maximumExclusive.every((value, axis) => value === other.maximumExclusive[axis]));
  }));
}
