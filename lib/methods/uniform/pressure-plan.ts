/** The pressure root's correction levels, planned without a GPU. */
import { uniformFaceLevelSlots } from "./uniform-lattice-address";

/** The levels: 4h, 8h, 16h and 32h. Each is a partition of a page (8, 4, 2
 * and 1 cells an axis), so a level's cells are rows of blocks and a pool of
 * any shape has all four, the last a cell to a page and the coarsest. */
export const UNIFORM_PRESSURE_PAGE_LEVELS = 4;

/** A level's rows and its fields, as words of the scratch arena. The rows
 * are the cells by block then local cell (x fastest), then the wall slots by
 * block face (uniformFaceLevel): a slot to a cell of each face of a block,
 * the unknown behind that face where a SOLID page lies across it. Level 0's
 * row of a tile or of a slot is the mixed root's (its tile key, and the tile
 * keys then uniformFaceTile). */
export interface UniformPressureLevelWords {
  /** Cells an axis of a page: 8 >> level. */
  readonly side: number;
  readonly cells: number;
  readonly slots: number;
  /** Both parities of the correction, a word a row. */
  readonly pressure: readonly [number, number];
  /** The V-cycle's RHS and bound, then a Full-Cycle's. */
  readonly rhs: readonly [number, number];
  readonly minimum: readonly [number, number];
  /** Seven words a row: the coefficients of its -x, +x, -y, +y, -z and +z
   * faces, then the liquid mask of the row and of those six neighbours. */
  readonly coefficients: number;
  /** A word a cell: phi as coarsened, then continued one cell into the closed cells. */
  readonly phi: readonly [number, number];
  /** Four words a cell: the open fraction and the +x, +y, +z face fractions. */
  readonly topology: number;
  /** A word a slot: the fraction of the wall face it lies behind (a low face's; a high face's is its cell's own). */
  readonly wall: number;
}

/** Rows of level `level` a block: 896, 160, 32 and 7. */
export const uniformPressureLevelRows = (level: number): number => (8 >> level) ** 3 + uniformFaceLevelSlots(1, level);

/** The levels of a pool of `blocks` blocks (its sentinels included: every
 * block has its rows), laid from arena word `first`; end: the word after. */
export function planUniformPressureLevels(blocks: number, first: number): { readonly levels: readonly UniformPressureLevelWords[]; readonly end: number } {
  if (!Number.isSafeInteger(blocks) || blocks < 3 || !Number.isSafeInteger(first) || first < 0)
    throw new RangeError(`Uniform pressure levels need a pool of at least three blocks at an arena word, not ${blocks} at ${first}`);
  let cursor = first;
  const take = (words: number): number => { const at = cursor; cursor += words; return at; };
  const levels: UniformPressureLevelWords[] = [];
  for (let level = 0; level < UNIFORM_PRESSURE_PAGE_LEVELS; level++) {
    const side = 8 >> level, cells = side ** 3 * blocks, slots = uniformFaceLevelSlots(blocks, level), rows = cells + slots;
    levels.push({ side, cells, slots, pressure: [take(rows), take(rows)], rhs: [take(rows), take(rows)], minimum: [take(rows), take(rows)],
      coefficients: take(7 * rows), phi: [take(cells), take(cells)], topology: take(4 * cells), wall: take(slots) });
  }
  return { levels, end: cursor };
}
