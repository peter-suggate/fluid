import { solidFractionOccupies } from "./solid-patch/kind";
import { assertSolidWorldRegionIsBox, SOLID_WORLD_BRICK_CELLS, solidWorldChangedPages, solidWorldOnlyRepainted,
  solidWorldPageOrder, type SolidWorld, type SolidWorldVoxelPatch } from "./solid-world";

/** Cells an axis of a mask page: the lattice's page. */
export const SOLID_MASK_PAGE_CELLS = 32;
/** Pages an axis of the flat directory, and its words: one a page of window pages 0..63. */
export const SOLID_MASK_WINDOW = 64;
export const SOLID_MASK_DIRECTORY_WORDS = SOLID_MASK_WINDOW ** 3;
/** Pages an axis of the lattice's window (the page directory's period), and
 * of a super-block: past the flat directory a page is found by its
 * super-block's word in the top table, then its word in the child table that
 * names. */
export const SOLID_MASK_LATTICE_PAGES = 256;
export const SOLID_MASK_SUPER_BLOCK_PAGES = 4;
/** First word of the top table (after the flat directory), and its words: one a super-block of the lattice's window. */
export const SOLID_MASK_TOP = SOLID_MASK_DIRECTORY_WORDS;
export const SOLID_MASK_TOP_WORDS = (SOLID_MASK_LATTICE_PAGES / SOLID_MASK_SUPER_BLOCK_PAGES) ** 3;
/** Words of a child table: one a page of its super-block. */
export const SOLID_MASK_CHILD_WORDS = SOLID_MASK_SUPER_BLOCK_PAGES ** 3;
/** Words before the lane: the flat directory and the top table. */
export const SOLID_MASK_TABLE_WORDS = SOLID_MASK_DIRECTORY_WORDS + SOLID_MASK_TOP_WORDS;
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
  /** First word of the lane after the directory and the top table, and its words. */
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

/** The image of `tileKeys` tile keys and `slots` page slots, with `laneWords` words between the top table and the cut map. */
export function solidOccupancyMaskLayout(tileKeys: number, slots: number, laneWords = 0): SolidOccupancyMaskLayout {
  if (!Number.isSafeInteger(tileKeys) || tileKeys < 0 || !Number.isSafeInteger(slots) || slots < 0 || !Number.isSafeInteger(laneWords) || laneWords < 0) {
    throw new RangeError(`A solid mask image needs whole counts: ${tileKeys} tile keys, ${slots} page slots, ${laneWords} lane words`);
  }
  const cutMap = SOLID_MASK_TABLE_WORDS + laneWords, firstSlot = Math.ceil((cutMap + tileKeys) / SOLID_MASK_UNIT_WORDS);
  if (firstSlot + slots > SOLID_MASK_FULL) {
    throw new RangeError(`A solid mask image of ${laneWords} lane words, ${tileKeys} tile keys and ${slots} page slots passes the ${SOLID_MASK_FULL} units a directory word can name`);
  }
  return { lane: SOLID_MASK_TABLE_WORDS, laneWords, cutMap, tileKeys, firstSlot, slots, words: (firstSlot + slots) * SOLID_MASK_UNIT_WORDS };
}

/**
 * The lattice's walls, by window page: umWorldSolid's rule (the page
 * directory's world box, uniformWindowWorldWGSL). Policy 1 is a closed tank,
 * walled outside its box; 2 the same under an open top, open above the box's
 * floor over its footprint; 3 open ground, walled below its floor alone.
 */
export interface SolidOccupancyMaskWalls {
  readonly policy: 1 | 2 | 3;
  /** The box's low window page, and its pages an axis. */
  readonly origin: readonly [number, number, number];
  readonly pages: readonly [number, number, number];
}
/** Whether the lattice walls off window page (x, y, z). */
export function solidOccupancyMaskWall(walls: SolidOccupancyMaskWalls, x: number, y: number, z: number): boolean {
  const o = walls.origin, n = walls.pages;
  if (walls.policy === 3) return y < o[1];
  const rx = x - o[0], ry = y - o[1], rz = z - o[2];
  return rx < 0 || ry < 0 || rz < 0 || rx >= n[0] || rz >= n[2] || (walls.policy === 1 && ry >= n[1]);
}
/** The box and its one-page halo lie in the flat directory: every page a lattice reader asks for is one read of it, wrapped as it always was. */
export function solidOccupancyMaskFitsFlat(walls: SolidOccupancyMaskWalls): boolean {
  return [0, 1, 2].every(a => walls.origin[a]! + walls.pages[a]! < SOLID_MASK_WINDOW);
}

/** What a mask is sized by, for the life of its image. */
export interface SolidOccupancyMaskPages {
  /** A solver with a separate terrain heightfield must not close its cut cells here. */
  readonly separateTerrain?: boolean;
  /** A partly solid cell is solid from half solid rather than from any
   * fraction: a procedural solid the fluid takes as voxels keeps its face. */
  readonly halfSolid?: boolean;
  /** The lattice's walls: a page they wall off is walled for good. */
  readonly walls: SolidOccupancyMaskWalls;
  /** Tile keys of the page pool: the cut map's words. */
  readonly tileKeys: number;
  /** Pages that may be partly solid at once. An edit that needs another is refused. */
  readonly slots: number;
  /** Words kept between the top table and the cut map for a lane its owner writes on the GPU; the mask leaves them zero and never uploads over them after the first whole image. */
  readonly laneWords?: number;
  /** The window page of the world's page (0,0,0), default the window's origin: the
   * directory is by window page, and the world, `solid`, `wall` and `changed` are
   * in the world's own cells. */
  readonly origin?: readonly [number, number, number];
}

/**
 * The directory word of the page holding window cell (x, y, z): umSolidPage
 * without the walls. A page of window pages 0..63 is its flat directory word;
 * another of the lattice's window its child table's word, EMPTY where its
 * super-block has none. A page off the lattice's window reads the flat
 * directory wrapped, as a mask that fits it reads its halo.
 */
export function solidOccupancyMaskPage(words: ArrayLike<number>, x: number, y: number, z: number): number {
  const px = x >> 5, py = y >> 5, pz = z >> 5, any = px | py | pz;
  if ((any & ~(SOLID_MASK_WINDOW - 1)) === 0) return words[px | (py << 6) | (pz << 12)]!;
  if ((any & ~(SOLID_MASK_LATTICE_PAGES - 1)) === 0) {
    const top = words[SOLID_MASK_TOP + ((px >> 2) | ((py >> 2) << 6) | ((pz >> 2) << 12))]!;
    return top === 0 ? SOLID_MASK_EMPTY : words[top + ((px & 3) | ((py & 3) << 2) | ((pz & 3) << 4))]!;
  }
  return words[(px & 63) | ((py & 63) << 6) | ((pz & 63) << 12)]!;
}

/** Whether cell (x, y, z) is solid in an image: umSolidVoxel, in the same integer arithmetic. */
export function solidOccupancyMaskVoxel(words: ArrayLike<number>, x: number, y: number, z: number): boolean {
  return solidOccupancyMaskCodeVoxel(words, solidOccupancyMaskPage(words, x, y, z), x, y, z);
}
function solidOccupancyMaskCodeVoxel(words: ArrayLike<number>, word: number, x: number, y: number, z: number): boolean {
  const code = word & 0xffff;
  if (code === SOLID_MASK_EMPTY) return false;
  if (code >= SOLID_MASK_FULL) return true;
  const i = (x & 31) | ((y & 31) << 5) | ((z & 31) << 10);
  return ((words[(code << 10) + (i >>> 5)]! >>> (i & 31)) & 1) !== 0;
}

/** The second level in WGSL: the child word of window page w (all < 256), EMPTY where its super-block has no child table. */
const childWGSL = (buffer: string, w: string) => `let s=${w}>>vec3u(2u);let t=${buffer}[${SOLID_MASK_TOP}u+(s.x|(s.y<<6u)|(s.z<<12u))];
  if(t!=0u){raw=${buffer}[t+((${w}.x&3u)|((${w}.y&3u)<<2u)|((${w}.z&3u)<<4u))];}`;

/**
 * The image's readers in WGSL over `buffer`, an array<u32> that holds it from
 * its first word: `page(p)`, the directory word of the page of cell p, and
 * `voxel(p)`, whether the cell is solid.
 *
 * The moving reader (the solver's) takes p in frame cells and the walls from
 * umWorldSolid; a page of window pages 0..63 is one read of the flat
 * directory, any other of the lattice's window the top table's and its child
 * table's. The fixed reader (the bake's) takes p in window cells: given the
 * walls of a mask that does not fit the flat directory, it reads both levels
 * and the walls by their rule; without, the flat directory alone, wrapped.
 */
export function solidOccupancyMaskWGSL(buffer: string, page: string, voxel: string, moving = false, walls?: SolidOccupancyMaskWalls): string {
  const W = SOLID_MASK_WALL;
  const reader = moving ? `let world=(p>>vec3u(5u))+umFramePage();let w=bitcast<vec3u>(world);
 var raw=${buffer}[(w.x&63u)|((w.y&63u)<<6u)|((w.z&63u)<<12u)];
 if(any(w>vec3u(63u))){
  raw=0u;
  if(all(w<vec3u(${SOLID_MASK_LATTICE_PAGES}u))){${childWGSL(buffer, "w")}}
 }
 return select(select(raw,0u,raw==${W}u),${W}u,umWorldSolid(world));`
    : walls ? `let q=p>>vec3u(5u);let w=bitcast<vec3u>(q);
 if(${page}Wall(q)){return ${W}u;}
 if(all(w<vec3u(64u))){return ${buffer}[w.x|(w.y<<6u)|(w.z<<12u)];}
 var raw=0u;
 if(all(w<vec3u(${SOLID_MASK_LATTICE_PAGES}u))){${childWGSL(buffer, "w")}}
 return raw;`
      : `let e=vec3u(p>>vec3u(5u))&vec3u(63u);return ${buffer}[e.x|(e.y<<6u)|(e.z<<12u)];`;
  const wall = !moving && walls ? `fn ${page}Wall(q:vec3i)->bool{
 let r=q-vec3i(${walls.origin.join(",")});
 ${walls.policy === 3 ? "return r.y<0;" : `return any(r<vec3i(0))||r.x>=${walls.pages[0]}||r.z>=${walls.pages[2]}${walls.policy === 1 ? `||r.y>=${walls.pages[1]}` : ""};`}
}
` : "";
  return /* wgsl */ `${wall}fn ${page}(p:vec3i)->u32{
 ${reader}
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
 *  - The flat directory, 64³ words: the word of window page (x, y, z) of
 *    pages 0..63 is at `x | y << 6 | z << 12`. Its low half is the page's
 *    mask code: `SOLID_MASK_EMPTY`, `SOLID_MASK_FULL`, or the unit that holds
 *    the page's bits. Its high half is kept as it is found, for a second
 *    per-page slot family beside the mask. A page the lattice walls off
 *    holds `SOLID_MASK_WALL` whole, so it reads as full.
 *  - The top table, 64³ words, the second level for the rest of the
 *    lattice's window (256 pages an axis, as the page directory): the word of
 *    super-block (x, y, z) of 4³ pages is at `x | y << 6 | z << 12`, 0 for
 *    none, else the first word of its child table, a word a page (local
 *    `x | y << 2 | z << 4`) as the flat directory's. A child table is made
 *    only for a super-block off the flat directory that holds a page neither
 *    EMPTY nor walled off, and only by the first world the mask folds: an
 *    edit that needs another is refused. Child tables lie in units of their
 *    own among the slots, 16 a unit. A mask that fits the flat directory
 *    (solidOccupancyMaskFitsFlat) has none, and its readers never read the
 *    top table for a page of its lattice.
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
 *
 * The walls are a rule (SolidOccupancyMaskWalls), umWorldSolid's: the flat
 * directory holds them as WALL words, and a page past it is walled when the
 * rule says so, whatever its super-block holds.
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
  readonly walls: SolidOccupancyMaskWalls;
  /** The box and its halo lie in the flat directory (solidOccupancyMaskFitsFlat). */
  readonly fitsFlat: boolean;
  /** Super-block (x | y << 6 | z << 12) to its child table's first word. */
  private readonly children = new Map<number, number>();
  /** The unit child tables are being cut from, and the next table of its 16. */
  private childUnit = 0;
  private childNext = 16;
  /** Only the first world folded makes child tables. */
  private built = false;
  private refused: string | undefined;
  /** One page's bits as the world gives them. */
  private readonly page = new Uint32Array(SOLID_MASK_UNIT_WORDS);
  private readonly dirty = { directoryLow: 0, directoryHigh: -1, topLow: 0, topHigh: -1, slotLow: 0, slotHigh: -1 };

  constructor(pages: SolidOccupancyMaskPages) {
    this.separateTerrain = pages.separateTerrain ?? false;
    this.occupies = pages.halfSolid ? fraction => fraction >= 128 : solidFractionOccupies;
    const origin = this.origin = pages.origin ?? [0, 0, 0];
    this.originCell = [origin[0] * SOLID_MASK_PAGE_CELLS, origin[1] * SOLID_MASK_PAGE_CELLS, origin[2] * SOLID_MASK_PAGE_CELLS];
    const walls = this.walls = pages.walls;
    this.fitsFlat = solidOccupancyMaskFitsFlat(walls);
    const layout = this.shape = solidOccupancyMaskLayout(pages.tileKeys, pages.slots, pages.laneWords);
    const words = this.image = new Uint32Array((layout.firstSlot + Math.min(layout.slots, 16)) * SOLID_MASK_UNIT_WORDS);
    for (let z = 0, e = 0; z < SOLID_MASK_WINDOW; z += 1) for (let y = 0; y < SOLID_MASK_WINDOW; y += 1) for (let x = 0; x < SOLID_MASK_WINDOW; x += 1, e += 1) {
      if (solidOccupancyMaskWall(walls, x, y, z)) words[e] = SOLID_MASK_WALL;
    }
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

  /** The directory word of the page of cell (x, y, z): what `umSolidPage` reads, WALL where the lattice walls it off. */
  pageWord(x: number, y: number, z: number): number {
    const o = this.originCell, wx = x + o[0], wy = y + o[1], wz = z + o[2];
    return solidOccupancyMaskWall(this.walls, wx >> 5, wy >> 5, wz >> 5) ? SOLID_MASK_WALL : solidOccupancyMaskPage(this.words, wx, wy, wz);
  }

  /** Whether cell (x, y, z) is solid: what `umSolidVoxel` reads from the image. */
  solid(x: number, y: number, z: number): boolean { const o = this.originCell; return solidOccupancyMaskCodeVoxel(this.words, this.pageWord(x, y, z), x + o[0], y + o[1], z + o[2]); }

  /** Whether the lattice walls off the page of cell (x, y, z). */
  wall(x: number, y: number, z: number): boolean { return this.pageWord(x, y, z) === SOLID_MASK_WALL; }

  /** The image word that holds window page (x, y, z) (each 0..255): its flat directory word, its child table's, or -1 where its super-block has none. */
  private wordOf(x: number, y: number, z: number): number {
    if (((x | y | z) & ~(SOLID_MASK_WINDOW - 1)) === 0) return x | (y << 6) | (z << 12);
    const child = this.children.get((x >> 2) | ((y >> 2) << 6) | ((z >> 2) << 12));
    return child === undefined ? -1 : child + ((x & 3) | ((y & 3) << 2) | ((z & 3) << 4));
  }

  /** A child table for the super-block of window page (x, y, z): its words EMPTY, or WALL where the walls are. */
  private child(x: number, y: number, z: number): void {
    const sx = x >> 2, sy = y >> 2, sz = z >> 2, key = sx | (sy << 6) | (sz << 12);
    if (this.childNext === 16) {
      if (this.slotsHeld === this.shape.slots) {
        this.refused = `The solid mask holds ${this.layout.slots} units, sized when the solver was built, and every one is in use: its child tables need another.`;
        throw new Error(this.refused);
      }
      this.childUnit = this.take(); this.childNext = 0;
    }
    const first = (this.childUnit << 10) + SOLID_MASK_CHILD_WORDS * this.childNext++, words = this.image;
    for (let l = 0; l < SOLID_MASK_CHILD_WORDS; l += 1) {
      words[first + l] = solidOccupancyMaskWall(this.walls, 4 * sx + (l & 3), 4 * sy + ((l >> 2) & 3), 4 * sz + (l >> 4)) ? SOLID_MASK_WALL : SOLID_MASK_EMPTY;
    }
    this.children.set(key, first);
    const top = SOLID_MASK_TOP + key, dirty = this.dirty;
    words[top] = first;
    dirty.topLow = Math.min(dirty.topLow, top); dirty.topHigh = Math.max(dirty.topHigh, top);
    dirty.slotLow = Math.min(dirty.slotLow, first); dirty.slotHigh = Math.max(dirty.slotHigh, first + SOLID_MASK_CHILD_WORDS - 1);
  }

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
    dirty.directoryLow = dirty.topLow = dirty.slotLow = Number.POSITIVE_INFINITY; dirty.directoryHigh = dirty.topHigh = dirty.slotHigh = -1;
    const moved = this.fold(previous, next, true, changed);
    this.built = true;
    if (!moved) return undefined;
    const ranges: SolidOccupancyMaskDirtyRange[] = [];
    if (dirty.directoryHigh >= dirty.directoryLow) ranges.push({ firstWord: dirty.directoryLow, wordCount: dirty.directoryHigh - dirty.directoryLow + 1 });
    if (dirty.topHigh >= dirty.topLow) ranges.push({ firstWord: dirty.topLow, wordCount: dirty.topHigh - dirty.topLow + 1 });
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
    // The mask pages to fold, by window page x | y << 8 | z << 16 of the lattice's window, each once.
    const pages = new Set<number>(), o = this.origin, W = SOLID_MASK_LATTICE_PAGES, P = SOLID_MASK_PAGE_CELLS, walls = this.walls;
    const brick = (coordinate: readonly [number, number, number]): void => {
      const x = (coordinate[0] >> 2) + o[0], y = (coordinate[1] >> 2) + o[1], z = (coordinate[2] >> 2) + o[2];
      if (x >= 0 && y >= 0 && z >= 0 && x < W && y < W && z < W && !solidOccupancyMaskWall(walls, x, y, z)) pages.add(x | (y << 8) | (z << 16));
    };
    // The pages off the wall that a box of cells touches.
    const box = (minimum: readonly number[], maximumExclusive: readonly number[]): void => {
      const low = [0, 1, 2].map(k => Math.max(0, Math.floor(minimum[k]! / P) + o[k]!));
      const high = [0, 1, 2].map(k => Math.min(W - 1, Math.floor((maximumExclusive[k]! - 1) / P) + o[k]!));
      for (let z = low[2]!; z <= high[2]!; z += 1) for (let y = low[1]!; y <= high[1]!; y += 1) for (let x = low[0]!; x <= high[0]!; x += 1) {
        if (!solidOccupancyMaskWall(walls, x, y, z)) pages.add(x | (y << 8) | (z << 16));
      }
    };
    if (!previous || previous.ground?.key !== next.ground?.key || !sameRegions(previous.regions ?? [], regions)) {
      // Every page that may hold a solid cell, before or after: the pages the
      // image holds solid, and every page the next world's ground, written
      // pages and regions reach. Any other page off the wall is empty and stays so.
      for (let e = 0; e < SOLID_MASK_DIRECTORY_WORDS; e += 1) {
        const word = this.words[e]!;
        if (word !== SOLID_MASK_WALL && (word & 0xffff) !== SOLID_MASK_EMPTY) pages.add((e & 63) | (((e >> 6) & 63) << 8) | ((e >> 12) << 16));
      }
      for (const [key, first] of this.children) for (let l = 0; l < SOLID_MASK_CHILD_WORDS; l += 1) {
        const word = this.words[first + l]!;
        if (word !== SOLID_MASK_WALL && (word & 0xffff) !== SOLID_MASK_EMPTY) {
          pages.add((4 * (key & 63) + (l & 3)) | ((4 * ((key >> 6) & 63) + ((l >> 2) & 3)) << 8) | ((4 * (key >> 12) + (l >> 4)) << 16));
        }
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
   * Fold window page `k` (x | y << 8 | z << 16) from `world`. False when its
   * bits are the image's already, "waits" when it needs a slot and none is
   * free (the image and `changed` are then untouched), true once a bit moved.
   */
  private foldPage(k: number, world: SolidWorld, regions: readonly SolidWorldVoxelPatch[], commit: boolean,
    changed: ((x: number, y: number, z: number, solid: boolean) => void) | undefined): boolean | "waits" {
    let words = this.image;
    const px = k & 255, py = (k >> 8) & 255, pz = k >> 16;
    let e = this.wordOf(px, py, pz);
    const held = e >= 0 ? words[e]! : solidOccupancyMaskWall(this.walls, px, py, pz) ? SOLID_MASK_WALL : SOLID_MASK_EMPTY;
    if (held === SOLID_MASK_WALL) return false;
    const P = SOLID_MASK_PAGE_CELLS, B = SOLID_WORLD_BRICK_CELLS, page = this.page, o = this.origin;
    // The page's low cell in the world's cells.
    const x0 = (px - o[0]) * P, y0 = (py - o[1]) * P, z0 = (pz - o[2]) * P;
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
    if (e < 0 && this.built) {
      this.refused = `The solid mask's child tables were made for the super-blocks the solver was built with: `
        + `this world makes page ${px},${py},${pz} of super-block ${px >> 2},${py >> 2},${pz >> 2}, which has none, solid. Build the solver with this solid.`;
      throw new Error(this.refused);
    }
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
    if (e < 0) { this.child(px, py, pz); words = this.image; e = this.wordOf(px, py, pz); }
    if (partial) {
      if (slot < 0) { next = this.take(); words = this.image; }
      const first = next << 10;
      words.set(page, first);
      dirty.slotLow = Math.min(dirty.slotLow, first); dirty.slotHigh = Math.max(dirty.slotHigh, first + SOLID_MASK_UNIT_WORDS - 1);
    } else if (slot >= 0) this.freed.push(code);
    if (next !== code) {
      this.occupied += (next !== SOLID_MASK_EMPTY ? 1 : 0) - (code !== SOLID_MASK_EMPTY ? 1 : 0);
      words[e] = ((held & 0xffff0000) | next) >>> 0;
      if (e < SOLID_MASK_DIRECTORY_WORDS) { dirty.directoryLow = Math.min(dirty.directoryLow, e); dirty.directoryHigh = Math.max(dirty.directoryHigh, e); }
      else { dirty.slotLow = Math.min(dirty.slotLow, e); dirty.slotHigh = Math.max(dirty.slotHigh, e); }
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
