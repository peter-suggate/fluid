import { solidFractionOccupies } from "./solid-patch/kind";
import { assertSolidWorldRegionIsBox, SOLID_WORLD_BRICK_CELLS, solidWorldChangedPages, solidWorldOnlyRepainted,
  solidWorldPageOrder, type SolidWorld, type SolidWorldVoxelPatch } from "./solid-world";

/** Cells an axis of a mask page: the lattice's page. */
export const SOLID_MASK_PAGE_CELLS = 32;
/** Pages an axis of the directory, and its words: one a page of the window. */
export const SOLID_MASK_WINDOW = 64;
export const SOLID_MASK_DIRECTORY_WORDS = SOLID_MASK_WINDOW ** 3;
/** Words of a unit, the grain the image is allocated in: one page's 32³ bits. */
export const SOLID_MASK_UNIT_WORDS = SOLID_MASK_PAGE_CELLS ** 3 / 32;
/** Mask codes, the low half of a directory word: no cell of the page is solid, every cell is. Any other is the page's unit. */
export const SOLID_MASK_EMPTY = 0;
export const SOLID_MASK_FULL = 0xfffe;
/** The whole directory word of a page the lattice walls off: the page table's SOLID word. */
export const SOLID_MASK_WALL = 0xffffffff;

/** Word range of the image an update rewrote. */
export interface SolidOccupancyMaskDirtyRange {
  readonly firstWord: number;
  readonly wordCount: number;
}

/** Where the parts of a mask image lie. */
export interface SolidOccupancyMaskLayout {
  /** First word of the lane after the directory, and its words. */
  readonly lane: number;
  readonly laneWords: number;
  /** First word of the tile cut map, one word a tile key. */
  readonly cutMap: number;
  readonly tileKeys: number;
  /** Unit of the first page slot, and the slots after it. */
  readonly firstSlot: number;
  readonly slots: number;
  /** Words of the whole image. */
  readonly words: number;
}

/** The image of `tileKeys` tile keys and `slots` page slots, with `laneWords` words between the directory and the cut map. */
export function solidOccupancyMaskLayout(tileKeys: number, slots: number, laneWords = 0): SolidOccupancyMaskLayout {
  if (!Number.isSafeInteger(tileKeys) || tileKeys < 0 || !Number.isSafeInteger(slots) || slots < 0 || !Number.isSafeInteger(laneWords) || laneWords < 0) {
    throw new RangeError(`A solid mask image needs whole counts: ${tileKeys} tile keys, ${slots} page slots, ${laneWords} lane words`);
  }
  const cutMap = SOLID_MASK_DIRECTORY_WORDS + laneWords, firstSlot = Math.ceil((cutMap + tileKeys) / SOLID_MASK_UNIT_WORDS);
  if (firstSlot + slots > SOLID_MASK_FULL) {
    throw new RangeError(`A solid mask image of ${laneWords} lane words, ${tileKeys} tile keys and ${slots} page slots passes the ${SOLID_MASK_FULL} units a directory word can name`);
  }
  return { lane: SOLID_MASK_DIRECTORY_WORDS, laneWords, cutMap, tileKeys, firstSlot, slots, words: (firstSlot + slots) * SOLID_MASK_UNIT_WORDS };
}

/** What a mask is sized by, for the life of its image. */
export interface SolidOccupancyMaskPages {
  /** A solver with a separate terrain heightfield must not close its cut cells here. */
  readonly separateTerrain?: boolean;
  /** A partly solid cell is solid from half solid rather than from any
   * fraction: a procedural solid the fluid takes as voxels keeps its face. */
  readonly halfSolid?: boolean;
  /** The lattice's page table as seeded, 64³ words: a page whose word is 0xffffffff is walled off for good. */
  readonly table: ArrayLike<number>;
  /** Tile keys of the page pool: the cut map's words. */
  readonly tileKeys: number;
  /** Pages that may be partly solid at once. An edit that needs another is refused. */
  readonly slots: number;
  /** Words kept between the directory and the cut map for a lane its owner writes on the GPU; the mask leaves them zero and never uploads over them after the first whole image. */
  readonly laneWords?: number;
  /** The window page of the world's page (0,0,0), default the window's origin: the
   * directory is by window page, and the world, `solid`, `wall` and `changed` are
   * in the world's own cells. */
  readonly origin?: readonly [number, number, number];
}

/** The directory word of the page holding cell (x, y, z): umSolidPage. */
export function solidOccupancyMaskPage(words: ArrayLike<number>, x: number, y: number, z: number): number {
  return words[((x >> 5) & 63) | (((y >> 5) & 63) << 6) | (((z >> 5) & 63) << 12)]!;
}

/** Whether cell (x, y, z) is solid in an image: umSolidVoxel, in the same integer arithmetic. */
export function solidOccupancyMaskVoxel(words: ArrayLike<number>, x: number, y: number, z: number): boolean {
  const code = solidOccupancyMaskPage(words, x, y, z) & 0xffff;
  if (code === SOLID_MASK_EMPTY) return false;
  if (code >= SOLID_MASK_FULL) return true;
  const i = (x & 31) | ((y & 31) << 5) | ((z & 31) << 10);
  return ((words[(code << 10) + (i >>> 5)]! >>> (i & 31)) & 1) !== 0;
}

/**
 * The image's readers in WGSL over `buffer`, an array<u32> that holds it from
 * its first word: `page(p)`, the directory word of the page of cell p, and
 * `voxel(p)`, whether the cell is solid.
 */
export function solidOccupancyMaskWGSL(buffer: string, page: string, voxel: string, moving = false): string {
  return /* wgsl */ `fn ${page}(p:vec3i)->u32{
 ${moving?`let world=(p>>vec3u(5u))+umFramePage();let e=bitcast<vec3u>(world)&vec3u(63u);let raw=${buffer}[e.x|(e.y<<6u)|(e.z<<12u)];
 let kept=select(raw,0u,(raw==${SOLID_MASK_WALL}u)|any(bitcast<vec3u>(world)>vec3u(63u)));
 return select(kept,${SOLID_MASK_WALL}u,umWorldSolid(world));`
 :`let e=vec3u(p>>vec3u(5u))&vec3u(63u);return ${buffer}[e.x|(e.y<<6u)|(e.z<<12u)];`}
}
fn ${voxel}(p:vec3i)->bool{
 let code=${page}(p)&0xffffu;
 if(code==0u){return false;}
 if(code>=${SOLID_MASK_FULL}u){return true;}
 let l=vec3u(p)&vec3u(31u);let i=l.x|(l.y<<5u)|(l.z<<10u);
 return (${buffer}[(code<<10u)+(i>>5u)]&(1u<<(i&31u)))!=0u;
}`;
}

const popcount = (value: number): number => {
  let v = value - ((value >>> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  return (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
};

/**
 * Which cells of the lattice's page window are solid, as the solver binds it:
 * one image, in words.
 *
 *  - The directory, 64³ words: the word of page (x, y, z), each taken mod 64,
 *    is at `x | y << 6 | z << 12`. Its low half is the page's mask code:
 *    `SOLID_MASK_EMPTY`, `SOLID_MASK_FULL`, or the unit that holds the page's
 *    bits. Its high half is kept as it is found, for a second per-page slot
 *    family beside the mask. A page the lattice walls off holds
 *    `SOLID_MASK_WALL` whole, so it reads as full.
 *  - A lane of `laneWords` words, zero here: its owner writes it on the GPU.
 *  - The tile cut map, a word a tile key, all cut until the solver's record
 *    build writes it. The mask never touches it again.
 *  - Page slots, a unit of 1,024 words each, from the next unit boundary: bit
 *    `x | y << 5 | z << 10` of a page's cells, so the words of unit u start at
 *    `u << 10` and a reader needs no base.
 *
 * A page is folded whole from the ground's column heights, the pages patches
 * wrote into it and the regions over it, then compared with what the image
 * holds, so an update reports exactly the cells that moved and uploads the
 * directory words and the slots it rewrote. A later world over the same
 * ground folds only the pages its strokes wrote; a page that becomes empty or
 * full gives its slot back.
 *
 * Pages are the composed patches, so a round or a painted patch reaches the
 * mask as the cells it left behind (`solidFractionOccupies`) and never as its
 * box. Only a region is read as a box, and a region is refused unless it is
 * one.
 *
 * The image's slots grow as pages take them, up to the slots it was made
 * with, until its owner seals it at a count (seal): the image is then sized
 * for good. A world with more partly solid pages than that is refused, loudly
 * and for good: the image is then part of one world and part of the next.
 */
export class SolidOccupancyMask {
  private readonly separateTerrain: boolean;
  private readonly occupies: (fraction: number) => boolean;
  /** pages.origin, and the same in cells. */
  private readonly origin: readonly [number, number, number];
  readonly originCell: readonly [number, number, number];
  /** The image: the directory, the lane, the cut map and the slots handed out so far (every one once sealed). */
  get words(): Uint32Array<ArrayBuffer> { return this.image; }
  get layout(): SolidOccupancyMaskLayout { return this.shape; }
  private image: Uint32Array<ArrayBuffer>;
  private shape: SolidOccupancyMaskLayout;
  private world: SolidWorld | undefined;
  /** Units given back, the last taken first; then the next never held, below which every held unit lies. */
  private readonly freed: number[] = [];
  private fresh: number;
  /** Pages off the wall whose code is not EMPTY. */
  private occupied = 0;
  private refused: string | undefined;
  /** One page's bits as the world gives them. */
  private readonly page = new Uint32Array(SOLID_MASK_UNIT_WORDS);
  private readonly dirty = { directoryLow: 0, directoryHigh: -1, slotLow: 0, slotHigh: -1 };

  constructor(pages: SolidOccupancyMaskPages) {
    this.separateTerrain = pages.separateTerrain ?? false;
    this.occupies = pages.halfSolid ? fraction => fraction >= 128 : solidFractionOccupies;
    const origin = this.origin = pages.origin ?? [0, 0, 0];
    this.originCell = [origin[0] * SOLID_MASK_PAGE_CELLS, origin[1] * SOLID_MASK_PAGE_CELLS, origin[2] * SOLID_MASK_PAGE_CELLS];
    if (pages.table.length !== SOLID_MASK_DIRECTORY_WORDS) throw new RangeError(`A solid mask takes the lattice's ${SOLID_MASK_DIRECTORY_WORDS}-word page table; it was given ${pages.table.length} words`);
    const layout = this.shape = solidOccupancyMaskLayout(pages.tileKeys, pages.slots, pages.laneWords);
    const words = this.image = new Uint32Array((layout.firstSlot + Math.min(layout.slots, 16)) * SOLID_MASK_UNIT_WORDS);
    for (let e = 0; e < SOLID_MASK_DIRECTORY_WORDS; e += 1) if (pages.table[e] === SOLID_MASK_WALL) words[e] = SOLID_MASK_WALL;
    words.fill(1, layout.cutMap, layout.cutMap + layout.tileKeys);
    this.fresh = layout.firstSlot;
  }

  /** Size the image for good at `slots` page slots: no fewer than it has handed out, no more than it was made with. */
  seal(slots: number): void {
    const shape = this.shape, used = this.fresh - shape.firstSlot;
    if (!Number.isSafeInteger(slots) || slots < used || slots > shape.slots) throw new RangeError(`A solid mask that has handed out ${used} of ${shape.slots} page slots cannot be sealed at ${slots}`);
    const words = (shape.firstSlot + slots) * SOLID_MASK_UNIT_WORDS;
    this.shape = { ...shape, slots, words };
    if (this.image.length !== words) { const image = new Uint32Array(words); image.set(this.image.subarray(0, Math.min(words, this.image.length))); this.image = image; }
  }

  /** A page's unit: the last given back, else the next never held, the image grown to hold it. */
  private take(): number {
    const freed = this.freed.pop();
    if (freed !== undefined) return freed;
    const unit = this.fresh++, units = this.image.length / SOLID_MASK_UNIT_WORDS;
    if (unit >= units) {
      const image = new Uint32Array(Math.min(this.shape.firstSlot + this.shape.slots, Math.max(2 * units, unit + 1)) * SOLID_MASK_UNIT_WORDS);
      image.set(this.image); this.image = image;
    }
    return unit;
  }

  /** The mask was made from this world: an update to it changes no bit. */
  describes(world: SolidWorld): boolean { return this.world === world; }

  /** No cell off the wall is solid. */
  get empty(): boolean { return this.occupied === 0; }

  /** Page slots in use: the pages that are partly solid. */
  get slotsHeld(): number { return this.fresh - this.shape.firstSlot - this.freed.length; }

  /** Whether cell (x, y, z) is solid: what `umSolidVoxel` reads from the image. */
  solid(x: number, y: number, z: number): boolean { const o = this.originCell; return solidOccupancyMaskVoxel(this.words, x + o[0], y + o[1], z + o[2]); }

  /** Whether the lattice walls off the page of cell (x, y, z). */
  wall(x: number, y: number, z: number): boolean { const o = this.originCell; return solidOccupancyMaskPage(this.words, x + o[0], y + o[1], z + o[2]) === SOLID_MASK_WALL; }

  /**
   * Make the mask describe `next`; undefined when no bit changed, else the
   * word ranges of the image to upload: the directory words and the slots
   * that were rewritten. `changed` is told every cell whose bit moved, once.
   */
  update(next: SolidWorld, changed?: (x: number, y: number, z: number, solid: boolean) => void):
    readonly SolidOccupancyMaskDirtyRange[] | undefined {
    const previous = this.world;
    this.world = next;
    const dirty = this.dirty;
    dirty.directoryLow = dirty.slotLow = Number.POSITIVE_INFINITY; dirty.directoryHigh = dirty.slotHigh = -1;
    if (!this.fold(previous, next, true, changed)) return undefined;
    const ranges: SolidOccupancyMaskDirtyRange[] = [];
    if (dirty.directoryHigh >= dirty.directoryLow) ranges.push({ firstWord: dirty.directoryLow, wordCount: dirty.directoryHigh - dirty.directoryLow + 1 });
    if (dirty.slotHigh >= dirty.slotLow) ranges.push({ firstWord: dirty.slotLow, wordCount: dirty.slotHigh - dirty.slotLow + 1 });
    return ranges;
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
    if (this.fold(this.world, next, false, undefined)) return false;
    this.world = next;
    return true;
  }

  /** Whether `next` moves a bit of `previous`. With `commit` the image is rewritten; without, nothing is and the first bit found is the answer. */
  private fold(previous: SolidWorld | undefined, next: SolidWorld, commit: boolean,
    changed: ((x: number, y: number, z: number, solid: boolean) => void) | undefined): boolean {
    if (this.refused) throw new Error(this.refused);
    if (previous === next) return false;
    const regions = next.regions ?? [];
    for (const region of regions) assertSolidWorldRegionIsBox(region);
    // The mask pages to fold, by directory word, each once.
    const pages = new Set<number>(), o = this.origin, W = SOLID_MASK_WINDOW, P = SOLID_MASK_PAGE_CELLS;
    const brick = (coordinate: readonly [number, number, number]): void => {
      const x = (coordinate[0] >> 2) + o[0], y = (coordinate[1] >> 2) + o[1], z = (coordinate[2] >> 2) + o[2];
      if (x >= 0 && y >= 0 && z >= 0 && x < W && y < W && z < W && this.words[x | (y << 6) | (z << 12)] !== SOLID_MASK_WALL) pages.add(x | (y << 6) | (z << 12));
    };
    // The pages off the wall that a box of cells touches.
    const box = (minimum: readonly number[], maximumExclusive: readonly number[]): void => {
      const low = [0, 1, 2].map(k => Math.max(0, Math.floor(minimum[k]! / P) + o[k]!));
      const high = [0, 1, 2].map(k => Math.min(W - 1, Math.floor((maximumExclusive[k]! - 1) / P) + o[k]!));
      for (let z = low[2]!; z <= high[2]!; z += 1) for (let y = low[1]!; y <= high[1]!; y += 1) for (let x = low[0]!; x <= high[0]!; x += 1) {
        const e = x | (y << 6) | (z << 12);
        if (this.words[e] !== SOLID_MASK_WALL) pages.add(e);
      }
    };
    if (!previous || previous.ground?.key !== next.ground?.key || !sameRegions(previous.regions ?? [], regions)) {
      // Every page that may hold a solid cell, before or after: the pages the
      // image holds solid, and every page the next world's ground, written
      // pages and regions reach. Any other page off the wall is empty and stays so.
      for (let e = 0; e < SOLID_MASK_DIRECTORY_WORDS; e += 1) {
        const word = this.words[e]!;
        if (word !== SOLID_MASK_WALL && (word & 0xffff) !== SOLID_MASK_EMPTY) pages.add(e);
      }
      for (const page of next.pages) brick(page.coordinate);
      for (const region of regions) if (region.operation !== "paint") box(region.minimum, region.maximumExclusive);
      if (next.ground && !this.separateTerrain) box(next.ground.minimum, next.ground.maximumExclusive);
    } else {
      // A repaint copies pages and moves no bit: answered from its strokes,
      // without reading a page.
      if (solidWorldOnlyRepainted(previous, next)) return false;
      // A world baked from the last one by strokes names the pages it copied,
      // made and dropped: nothing else is visited.
      const strokes = solidWorldChangedPages(previous, next);
      if (strokes) for (const coordinate of strokes) brick(coordinate);
      else {
        // An undo, a branch, a recomposed document. Both lists hold only written
        // pages and ascend in (z, y, x), so one pass over the two finds every
        // page that came, went or was replaced, and the ground is never walked.
        const held = next.pages;
        let cursor = 0;
        for (const page of previous.pages) {
          while (cursor < held.length && solidWorldPageOrder(held[cursor]!.coordinate, page.coordinate) < 0) brick(held[cursor++]!.coordinate);
          const same = cursor < held.length && solidWorldPageOrder(held[cursor]!.coordinate, page.coordinate) === 0 ? held[cursor++] : undefined;
          if (same !== page) brick(page.coordinate);
        }
        while (cursor < held.length) brick(held[cursor++]!.coordinate);
      }
    }
    let moved = false;
    // A page that needs a slot while none is free waits for the pages after it to give theirs back.
    let waiting: number[] | undefined;
    for (const e of pages) {
      const folded = this.foldPage(e, next, regions, commit, changed);
      if (folded === "waits") (waiting ??= []).push(e);
      else if (folded) { moved = true; if (!commit) return true; }
    }
    if (waiting) {
      for (const e of waiting) {
        if (this.foldPage(e, next, regions, commit, changed) !== "waits") continue;
        this.refused = `The solid mask holds ${this.layout.slots} page slots, sized when the solver was built, and every one is in use: `
          + `this world makes more pages partly solid. The image is never grown mid-run; build the solver with more slots.`;
        throw new Error(this.refused);
      }
      moved = true;
    }
    return moved;
  }

  /**
   * Fold the page at directory word `e` from `world`. False when its bits are
   * the image's already, "waits" when it needs a slot and none is free (the
   * image and `changed` are then untouched), true once a bit moved.
   */
  private foldPage(e: number, world: SolidWorld, regions: readonly SolidWorldVoxelPatch[], commit: boolean,
    changed: ((x: number, y: number, z: number, solid: boolean) => void) | undefined): boolean | "waits" {
    let words = this.image;
    const held = words[e]!;
    if (held === SOLID_MASK_WALL) return false;
    const P = SOLID_MASK_PAGE_CELLS, B = SOLID_WORLD_BRICK_CELLS, page = this.page, o = this.origin;
    // The page's low cell in the world's cells.
    const x0 = ((e & 63) - o[0]) * P, y0 = (((e >> 6) & 63) - o[1]) * P, z0 = ((e >> 12) - o[2]) * P;
    page.fill(0);
    // A row of the page along x is one word: its bits are the row's cells.
    const ground = world.ground;
    if (ground && !this.separateTerrain && y0 >= ground.minimum[1] && y0 < ground.maximumExclusive[1]) {
      const xa = Math.max(x0, ground.minimum[0]), xb = Math.min(x0 + P, ground.maximumExclusive[0]);
      const za = Math.max(z0, ground.minimum[2]), zb = Math.min(z0 + P, ground.maximumExclusive[2]);
      // Each column is solid from y = 0 for `solidCells`.
      for (let z = za; z < zb; z += 1) for (let x = xa; x < xb; x += 1) {
        const top = Math.min(P, ground.solidCells(x, z) - y0), bit = 1 << (x - x0);
        for (let y = 0, row = P * (z - z0); y < top; y += 1) page[row + y]! |= bit;
      }
    }
    // A written page is the whole answer for its cells: it clears the ground it dug as it sets what it filled.
    if (world.pages.length) {
      const per = P / B, coordinate: [number, number, number] = [0, 0, 0];
      for (let k = 0; k < per; k += 1) for (let j = 0; j < per; j += 1) for (let i = 0; i < per; i += 1) {
        coordinate[0] = x0 / B + i; coordinate[1] = y0 / B + j; coordinate[2] = z0 / B + k;
        const index = world.directory.lookup(coordinate);
        if (index === undefined) continue;
        const written = world.pages[index]!, fraction = written.solidFraction, shift = B * i, clear = ~(((1 << B) - 1) << shift);
        for (let z = 0; z < B; z += 1) for (let y = 0; y < B; y += 1) {
          const local = B * (y + B * z), row = (B * j + y) + P * (B * k + z);
          let bits = 0;
          for (let x = 0; x < B; x += 1) {
            const cell = local + x;
            const authored = !this.separateTerrain || !ground
              || (written.authored !== undefined && (written.authored[cell >>> 3]! & (1 << (cell & 7))) !== 0);
            if (authored && this.occupies(fraction[cell]!)) bits |= 1 << x;
          }
          page[row] = (page[row]! & clear) | (bits << shift);
        }
      }
    }
    // Regions override pages in order. A paint region recolours cells and decides nothing about occupancy.
    for (const region of regions) {
      if (region.operation === "paint") continue;
      const xa = Math.max(x0, region.minimum[0]), xb = Math.min(x0 + P, region.maximumExclusive[0]);
      const ya = Math.max(y0, region.minimum[1]), yb = Math.min(y0 + P, region.maximumExclusive[1]);
      const za = Math.max(z0, region.minimum[2]), zb = Math.min(z0 + P, region.maximumExclusive[2]);
      if (xa >= xb || ya >= yb || za >= zb) continue;
      const bits = (0xffffffff >>> (P - (xb - xa))) << (xa - x0), fill = region.operation === "fill";
      for (let z = za; z < zb; z += 1) for (let y = ya; y < yb; y += 1) {
        const row = (y - y0) + P * (z - z0);
        page[row] = fill ? page[row]! | bits : page[row]! & ~bits;
      }
    }
    let count = 0;
    for (let w = 0; w < SOLID_MASK_UNIT_WORDS; w += 1) count += popcount(page[w]!);
    const code = held & 0xffff, slot = code !== SOLID_MASK_EMPTY && code !== SOLID_MASK_FULL ? code << 10 : -1;
    const uniform = code === SOLID_MASK_FULL ? 0xffffffff : 0;
    let moved = false;
    for (let w = 0; w < SOLID_MASK_UNIT_WORDS && !moved; w += 1) moved = (slot < 0 ? uniform : words[slot + w]!) !== page[w]!;
    if (!moved || !commit) return moved;
    const partial = count !== 0 && count !== P ** 3;
    if (partial && slot < 0 && this.slotsHeld === this.shape.slots) return "waits";
    if (changed) {
      for (let w = 0; w < SOLID_MASK_UNIT_WORDS; w += 1) {
        const now = page[w]!;
        let bits = ((slot < 0 ? uniform : words[slot + w]!) ^ now) >>> 0;
        while (bits) {
          const bit = 31 - Math.clz32(bits & -bits); bits = (bits & (bits - 1)) >>> 0;
          changed(x0 + bit, y0 + (w & 31), z0 + (w >> 5), ((now >>> bit) & 1) !== 0);
        }
      }
    }
    const dirty = this.dirty;
    let next = partial ? code : count ? SOLID_MASK_FULL : SOLID_MASK_EMPTY;
    if (partial) {
      if (slot < 0) { next = this.take(); words = this.image; }
      const first = next << 10;
      words.set(page, first);
      dirty.slotLow = Math.min(dirty.slotLow, first); dirty.slotHigh = Math.max(dirty.slotHigh, first + SOLID_MASK_UNIT_WORDS - 1);
    } else if (slot >= 0) this.freed.push(code);
    if (next !== code) {
      this.occupied += (next !== SOLID_MASK_EMPTY ? 1 : 0) - (code !== SOLID_MASK_EMPTY ? 1 : 0);
      words[e] = ((held & 0xffff0000) | next) >>> 0;
      dirty.directoryLow = Math.min(dirty.directoryLow, e); dirty.directoryHigh = Math.max(dirty.directoryHigh, e);
    }
    return true;
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
