/** The pressure arena's super-block tail, planned without a GPU
 * (uniform-pressure-arena.ts, schedule.tail "superblocks").
 *
 * Above the per-page levels (8h, 16h, 32h: a page's 4^3, 2^3 and one row)
 * two levels group pages: 64h, a row a 2^3 group of pages, and 128h, a row a
 * 4^3 group (a super-block). A group exists where one of its pages holds a
 * 32h row; its row aggregates its children's (D less the couplings toward
 * children that hold rows, the outward couplings summed, both times the
 * scale). Until the super-block directory lands, a group is keyed by page
 * coordinate from the box's low page (the window's 6-bit coordinate less the
 * box origin, mod 64), so groups never straddle the box's origin and a key's
 * neighbour across the window's edge is none. Each level keeps a list of its
 * rows (key words) and a window map (key to row + 1, 0: none), cleared after
 * each solve over the list that wrote it. The 128h rows are the one
 * workgroup's: at most 16^3 = 4096 of them in the window, 8.6G h cells. */
export const UNIFORM_PRESSURE_SUPER6_SIDE = 32;
export const UNIFORM_PRESSURE_SUPER7_SIDE = 16;

export interface UniformPressureSuperWords {
  /** Rows each level can hold: a 64h row has a page with a row, a 128h row a 64h row. */
  readonly capacity6: number;
  readonly capacity7: number;
  /** Index (state) words: the counts, the key lists, the maps; end: the word after. */
  readonly index: { readonly count6: number; readonly count7: number; readonly list6: number; readonly list7: number; readonly map6: number; readonly map7: number; readonly end: number };
  /** Pool words, a plane of capacity words each: 64h X, Y, R, D, six K, eight
   * children (page blocks, UB_NONE_ID none); 128h X, R, D, six K, eight
   * children (64h rows, UB_NONE_ID none); end: the word after. */
  readonly pool: { readonly x6: number; readonly y6: number; readonly r6: number; readonly d6: number; readonly k6: number; readonly c6: number;
    readonly x7: number; readonly r7: number; readonly d7: number; readonly k7: number; readonly c7: number; readonly end: number };
}

export function planUniformPressureSuper(blocks: number, index: number, pool: number): UniformPressureSuperWords {
  if (!Number.isSafeInteger(blocks) || blocks < 1 || !Number.isSafeInteger(index) || index < 0 || !Number.isSafeInteger(pool) || pool < 0)
    throw new RangeError(`Uniform pressure super-block tail: ${blocks} blocks at words ${index}, ${pool}`);
  const capacity6 = Math.min(blocks, UNIFORM_PRESSURE_SUPER6_SIDE ** 3), capacity7 = Math.min(blocks, UNIFORM_PRESSURE_SUPER7_SIDE ** 3);
  let i = index;
  const take = (words: number) => { const at = i; i += words; return at; };
  const count6 = take(1), count7 = take(1); take(2);
  const list6 = take(capacity6), list7 = take(capacity7), map6 = take(UNIFORM_PRESSURE_SUPER6_SIDE ** 3), map7 = take(UNIFORM_PRESSURE_SUPER7_SIDE ** 3);
  let p = pool;
  const plane = (n: number, planes = 1) => { const at = p; p += n * planes; return at; };
  const x6 = plane(capacity6), y6 = plane(capacity6), r6 = plane(capacity6), d6 = plane(capacity6), k6 = plane(capacity6, 6), c6 = plane(capacity6, 8);
  const x7 = plane(capacity7), r7 = plane(capacity7), d7 = plane(capacity7), k7 = plane(capacity7, 6), c7 = plane(capacity7, 8);
  return { capacity6, capacity7, index: { count6, count7, list6, list7, map6, map7, end: i }, pool: { x6, y6, r6, d6, k6, c6, x7, r7, d7, k7, c7, end: p } };
}

/** A page's group key at 64h (shift 1) or 128h (shift 2), from its window
 * page coordinate and the box's low page: the WGSL's sRel and sPack. */
export function uniformPressureSuperKey(page: readonly [number, number, number], box: readonly [number, number, number], shift: 1 | 2): number {
  const bits = shift === 1 ? 5 : 4, g = page.map((v, a) => ((v + 64 - box[a]!) & 63) >> shift);
  return g[0]! | (g[1]! << bits) | (g[2]! << 2 * bits);
}
