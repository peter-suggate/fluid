/** Reject a provably impossible uniform dual grid before building its octree.
 * Counts existing occupancy keys only; field sampling and fitting stay on GPU.
 * Fixed maintenance records also consume storage, so passing is a lower-bound
 * check, not a promise that the final arena fits.
 */
export function assertDualGridAttachmentFits(
  occupiedKeys: Iterable<unknown>, brickSize: number, maximumBytes: number, candidatesPerBrick = 0, reservedBricks = 0,
): void {
  const bytesPerBrick = brickSize ** 3 * 16 + (4 + candidatesPerBrick) * 4;
  const maximumBricks = Math.floor(maximumBytes / bytesPerBrick);
  let bricks = reservedBricks;
  for (const _key of occupiedKeys) {
    if (++bricks > maximumBricks) {
      throw new RangeError(`Uniform dual-grid construction needs more than ${maximumBytes} bytes (${bricks} allocated bricks, including the edit reserve); reduce render grid resolution`);
    }
  }
}
