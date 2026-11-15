/**
 * Lane P's pressure rows and projection on the brick pool (brick-abi.md §6): the engine's
 * `operator` (entry nbOperator, F_ALL) and `project` (F_LIQ), extended in place to NB-FLIP's
 * composite system (pressure-composite.wgsl.ts):
 * - `operator` is the composite h rows (rowsF): band faces at the face φ's θ (Gibou et al. 2002;
 *   Enright et al. 2003), seams to native 4h rows with their heads, separating lattice walls
 *   (Batty, Bertails & Bridson 2007; Chentanez & Müller 2012), K_OPEN toward non-resident pages,
 *   and NB-FLIP's volume control in the RHS (Kim et al. 2007). It keeps the entry point, the
 *   FAIL.LIST zero-row rule over F_ALL and the cut's FAIL.UNSUPPORTED check. It writes X (the
 *   start), the start again in B (the seam heads' and the walls' state for `project`), and PPHI.
 * - The native rows (rowsC, C_ALL) run at the head of the registered solver's setup, after
 *   `operator` (brick-pressure-solver.ts), because the engine places one operator over F_ALL.
 * - `project` is the composite h-face projection (umProject's bFace in the slot units), with the
 *   WARM bit. The native 4h faces (C U) and the h tiles' C X mean run at the end of the solver's
 *   finish (projectC), because they need only the solution.
 * The all-h rows are the M1 contract's except at the lattice's faces, which separate: attached
 * while X + lift >= 0, which is the closed face under hydrostatics.
 */
import type { BrickLayout } from "./brick-abi";
import { pressureCompositeKernels } from "./pressure-composite.wgsl";

/** `operator` and `project` (WGSL reserves `operator`: its entry point is nbOperator). `tiered` is the engine's
 * (BrickStages.tiered); the default keeps the tiered engine's absent-page guard. */
export function pressureOperatorKernels(L: BrickLayout, opt: { readonly tiered?: boolean } = {}) {
  const K = pressureCompositeKernels(L, { rowsF: ["operator", "nbOperator"], projectF: ["project", "project"] }, opt);
  return { operator: K.rowsF, project: K.projectF };
}
