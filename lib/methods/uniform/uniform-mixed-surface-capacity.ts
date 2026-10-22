/** Transient scheduling queues, not fluid/ownership capacity. Overflow traces
 * join the full deferred-vertex path; overflow deferred cubics evaluate inline.
 * 32,768 tiles stage over two million traces without reserving every possible
 * fine tile in a large sparse page pool. */
export const UNIFORM_SURFACE_STAGING_TILES = 32768;
export function uniformMixedSurfaceCapacity(fineTiles:number):{traceEntries:number;cubicEntries:number}{
  if(!Number.isSafeInteger(fineTiles)||fineTiles<0)throw new Error("Surface fine-tile capacity must be a nonnegative safe integer");
  const stagedTiles=Math.min(fineTiles,UNIFORM_SURFACE_STAGING_TILES);
  return {traceEntries:64*stagedTiles,cubicEntries:125*stagedTiles};
}
