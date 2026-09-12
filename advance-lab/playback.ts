import type { AdvanceSlice } from
  "../lib/methods/adaptive-volume/advance-slice/slice-solver";

/** Identity of the mutable solver state consumed by one canvas publication. */
export function slicePresentationRevision(slice: AdvanceSlice): string {
  return `${slice.frame}:${slice.injections}:${slice.topology.accepted.generation}`;
}

/** Play must not mutate a slice whose preceding revision has not been painted. */
export function slicePresentationReady(
  paintedRevision: string | null,
  slice: AdvanceSlice,
): boolean {
  return paintedRevision === slicePresentationRevision(slice);
}
