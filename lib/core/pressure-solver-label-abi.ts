/**
 * Capability markers a pressure solver embeds in its published label.
 *
 * `GPUEulerianInfo.pressureSolver` is free text a solver writes for the panels
 * to display, but two substrings inside it are load-bearing: they say the
 * published iteration counts came from a GPU-resident §4.3 executor whose
 * post-convergence tail dispatches zero workgroups. Panels and diagnostics
 * read those counts differently when that is true, so they need to recognise
 * the marker without importing the executor that stamps it — which is the only
 * reason the recogniser lives here rather than beside the solver.
 *
 * Substring matching, not equality: the label also carries the scene's own
 * shape and tuning, and a consumer that demanded the whole string would go
 * quietly false the next time a solver appended a word to it.
 */

/** Stable capability marker shared by the solver label and UI diagnostics. */
export const OCTREE_PERSISTENT_MGPCG_SOLVER_LABEL_MARKER = "persistent executor";
export const OCTREE_EXACT_REDUCTION_MGPCG_SOLVER_LABEL_MARKER = "exact-reduction executor";

/** Legacy name retained for UI callers; recognizes either production §4.3
 * executor capability marker. */
export function isOctreePersistentMGPCGSolverLabel(label: string | undefined): boolean {
  return label?.includes(OCTREE_PERSISTENT_MGPCG_SOLVER_LABEL_MARKER) === true
    || label?.includes(OCTREE_EXACT_REDUCTION_MGPCG_SOLVER_LABEL_MARKER) === true;
}
