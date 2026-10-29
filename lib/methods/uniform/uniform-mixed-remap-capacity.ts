/** One list word and one fitted value per corner; reductions stay in workgroup memory. */
export const UNIFORM_REMAP_PLANE_WORDS = 2;

/** A plane writer is unique at each 4h vertex, each stored tile can
 * contribute at most eight corners, and a page of 8^3 tiles has 9^3
 * vertices. Size for the smallest of these bounds instead of imposing a
 * scene-independent 1,024-corner ceiling: the pool's pages bound it, so the
 * box's size does not (its vertices only cap a small box). The device
 * limit remains a hard bound; the GPU's existing overflow record handles a
 * relayout that exceeds it without publishing a partial projection. */
export function uniformMixedRemapCapacity(
  dimensions: readonly [number, number, number], tiles: number,
  maxBytes: number, solids: boolean,
): { corners: number; bytes: number } {
  const headerWords = 4 + tiles;
  const held = Math.floor((Math.floor(maxBytes / 4) - headerWords) / UNIFORM_REMAP_PLANE_WORDS);
  if (held < 1) throw new Error("Device cannot hold the mixed remap worklist and one plane corner");
  const vertices = dimensions.reduce((count, cells) => count * (cells / 4 + 1), 1);
  const corners = Math.min(held, solids ? Math.min(vertices, 8 * tiles, 729 * Math.ceil(tiles / 512)) : 1);
  return { corners, bytes: 4 * (headerWords + corners * UNIFORM_REMAP_PLANE_WORDS) };
}
