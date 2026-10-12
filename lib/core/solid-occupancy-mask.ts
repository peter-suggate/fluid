import { solidFractionOccupies } from "./solid-patch/kind";
import { assertSolidWorldRegionIsBox, SOLID_WORLD_BRICK_CELLS, solidWorldChangedPages, solidWorldOnlyRepainted,
  solidWorldPageAt, solidWorldPageOrder, type SolidGround, type SolidWorld, type SolidWorldPage,
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
 * The mask is built from the ground's column heights and the pages patches
 * wrote, never from a page per terrain cell and never by sampling the lattice
 * cell by cell: a column of ground is a run of bits. A later world over the
 * same ground is folded in by the written pages it does not share with the
 * previous one. A voxel stroke copies only the pages it writes, so an edit
 * costs its own pages rather than the domain, and so does undoing it.
 *
 * Pages are the composed patches, so a round or a painted patch reaches the
 * mask as the cells it left behind (`solidFractionOccupies`) and never as its
 * box. Only a region is read as a box, and a region is refused unless it is
 * one.
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

  /** The mask was made from this world: an update to it changes no bit. */
  describes(world: SolidWorld): boolean { return this.world === world; }

  /** Make the mask describe `next`; undefined when no bit changed. */
  update(next: SolidWorld): SolidOccupancyMaskDirtyRange | undefined {
    const previous = this.world;
    this.world = next;
    return this.fold(previous, next, true);
  }

  /**
   * Make the mask describe `next` only if that moves no bit. False, with the
   * mask and the world it describes left exactly as they were, when one would.
   *
   * A consumer holding work derived from the mask asks this first. A paint
   * copies pages and moves nothing, and is then taken up without any of that
   * work being touched; anything else is left for `update` to report.
   */
  adoptIfUnchanged(next: SolidWorld): boolean {
    if (this.fold(this.world, next, false)) return false;
    this.world = next;
    return true;
  }

  /** The words `next` rewrites over `previous`. Without `commit` nothing is written and the first word found is the answer. */
  private fold(previous: SolidWorld | undefined, next: SolidWorld, commit: boolean): SolidOccupancyMaskDirtyRange | undefined {
    if (previous === next) return undefined;
    const regions = next.regions ?? [];
    for (const region of regions) assertSolidWorldRegionIsBox(region);
    const dirty = { low: Number.POSITIVE_INFINITY, high: -1 };
    if (!previous || previous.ground?.key !== next.ground?.key || !sameRegions(previous.regions ?? [], regions)) {
      // Every word is rewritten, which is its own answer to whether any would be.
      if (commit) {
        this.words.fill(0, SOLID_OCCUPANCY_MASK_HEADER_WORDS);
        if (next.ground) this.writeGround(next.ground);
        // A written page is the whole answer for its cells: it clears the ground it dug as it sets what it filled.
        for (const page of next.pages) this.writePage(page.coordinate, page, [], dirty, true);
        for (const region of regions) this.writeRegion(region);
      }
      return { firstWord: 0, wordCount: this.words.length };
    }
    // A repaint copies pages and moves no bit: answered from its strokes,
    // without reading a page.
    if (solidWorldOnlyRepainted(previous, next)) return undefined;
    // A world baked from the last one by strokes names the pages it copied,
    // made and dropped: nothing else is visited.
    const changed = solidWorldChangedPages(previous, next);
    if (changed) {
      for (const coordinate of changed) {
        // A page that left the list answers as the ground again, or as nothing.
        this.writePage(coordinate, solidWorldPageAt(next, coordinate), regions, dirty, commit);
        if (!commit && dirty.high >= dirty.low) break;
      }
    } else {
      // An undo, a branch, a recomposed document. Both lists hold only written
      // pages and ascend in (z, y, x), so one pass over the two finds every
      // page that came, went or was replaced, and the ground is never walked.
      const pages = next.pages;
      let cursor = 0;
      for (const page of previous.pages) {
        while (cursor < pages.length && solidWorldPageOrder(pages[cursor]!.coordinate, page.coordinate) < 0) {
          this.writePage(pages[cursor]!.coordinate, pages[cursor++], regions, dirty, commit);
        }
        const held = cursor < pages.length && solidWorldPageOrder(pages[cursor]!.coordinate, page.coordinate) === 0
          ? pages[cursor++] : undefined;
        // What was written and no longer is reads as the ground under it.
        if (held !== page) this.writePage(page.coordinate, held ?? next.ground?.page(page.coordinate), regions, dirty, commit);
        if (!commit && dirty.high >= dirty.low) break;
      }
      if (commit || dirty.high < dirty.low) {
        while (cursor < pages.length) this.writePage(pages[cursor]!.coordinate, pages[cursor++], regions, dirty, commit);
      }
    }
    return dirty.high < dirty.low ? undefined
      : { firstWord: dirty.low, wordCount: dirty.high - dirty.low + 1 };
  }

  private writePage(coordinate: readonly [number, number, number], page: SolidWorldPage | undefined,
    regions: readonly SolidWorldVoxelPatch[], dirty: { low: number; high: number }, commit: boolean): void {
    const B = SOLID_WORLD_BRICK_CELLS, [sx, sy] = this.size, words = this.words;
    const base = [coordinate[0] * B, coordinate[1] * B, coordinate[2] * B];
    const lo = base.map((value) => Math.max(0, -1 - value));
    const hi = base.map((value, axis) => Math.min(B, this.size[axis]! - 1 - value));
    if (lo.some((value, axis) => value >= hi[axis]!)) return;
    // A paint region recolours cells and decides nothing about occupancy.
    const covering = regions.filter((region) => region.operation !== "paint" && base.every((value, axis) =>
      region.minimum[axis]! < value + B && region.maximumExclusive[axis]! > value));
    for (let z = lo[2]!; z < hi[2]!; z += 1) for (let y = lo[1]!; y < hi[1]!; y += 1) {
      const local = B * (y + B * z);
      const row = sx * ((base[1]! + y + 1) + sy * (base[2]! + z + 1)) + base[0]! + 1;
      for (let x = lo[0]!; x < hi[0]!; x += 1) {
        let solid = page !== undefined && solidFractionOccupies(page.solidFraction[local + x]!);
        for (const region of covering) {
          if (base[0]! + x < region.minimum[0]! || base[0]! + x >= region.maximumExclusive[0]!
            || base[1]! + y < region.minimum[1]! || base[1]! + y >= region.maximumExclusive[1]!
            || base[2]! + z < region.minimum[2]! || base[2]! + z >= region.maximumExclusive[2]!) continue;
          solid = region.operation === "fill";
        }
        const index = row + x, word = SOLID_OCCUPANCY_MASK_HEADER_WORDS + (index >>> 5), bit = 1 << (index & 31);
        const before = words[word]!;
        // Unsigned: a word holding bit 31 must compare equal to its stored value.
        const after = (solid ? before | bit : before & ~bit) >>> 0;
        if (after === before) continue;
        if (word < dirty.low) dirty.low = word;
        if (word > dirty.high) dirty.high = word;
        if (!commit) return;
        words[word] = after;
      }
    }
  }

  /** Set every ground cell of a cleared mask: each column is solid from y = 0 for `solidCells`. */
  private writeGround(ground: SolidGround): void {
    const [sx, sy, sz] = this.size, words = this.words;
    const x0 = Math.max(-1, ground.minimum[0]), x1 = Math.min(sx - 1, ground.maximumExclusive[0]);
    const z0 = Math.max(-1, ground.minimum[2]), z1 = Math.min(sz - 1, ground.maximumExclusive[2]);
    for (let z = z0; z < z1; z += 1) for (let x = x0; x < x1; x += 1) {
      const top = Math.min(sy - 1, ground.solidCells(x, z));
      let index = (x + 1) + sx * (1 + sy * (z + 1));
      for (let y = 0; y < top; y += 1, index += sx) {
        words[SOLID_OCCUPANCY_MASK_HEADER_WORDS + (index >>> 5)]! |= 1 << (index & 31);
      }
    }
  }

  private writeRegion(region: SolidWorldVoxelPatch): void {
    if (region.operation === "paint") return;
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
