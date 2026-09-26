/** Fixed overlap patterns: one translated box intersects at most (r + 1)^3
 * aligned fragments when its width is r times the sampling width. Duplicate
 * fragments may resolve to one coarser donor; donor normalization sums them.
 */
export type UniformCellWidth = 1 | 2 | 4;
export function uniformVolumeStencilSlots(ratio: UniformCellWidth): number {
  return (ratio + 1) ** 3;
}
/** One implicit donor-base word, overlap weights, and the self-fallback weight.
 * No per-edge donor indices or dynamically sized row storage are required.
 */
export function uniformVolumeStencilBytes(ratio: UniformCellWidth): number {
  return 4 * (uniformVolumeStencilSlots(ratio) + 2);
}
