import type { AdvanceView } from "../lib/physics-wasm/advance-view";

/** Identity of the mutable solver state consumed by one canvas publication. */
export function advancePresentationRevision(view: AdvanceView): string {
  const revision = view.revision;
  return `${revision.runEpoch}:${revision.commandSequence}:${revision.frame}:${revision.injections}:${revision.topologyGeneration}:${revision.surfaceRevision}`;
}

/** Play must not mutate a slice whose preceding revision has not been painted. */
export function advancePresentationReady(
  paintedRevision: string | null,
  view: AdvanceView,
): boolean {
  return paintedRevision === advancePresentationRevision(view);
}
