/**
 * Lane P's pressure rows and projection on the brick pool (brick-abi.md §6): the engine's
 * `operator` (entry nbOperator, F_ALL) and `project` (F_ALL), extended in place to NB-FLIP's
 * composite system (pressure-composite.wgsl.ts):
 * - `operator` is the composite h rows (rowsF): band faces at the face φ's θ (Gibou et al. 2002;
 *   Enright et al. 2003), seams to native 4h rows with their heads, separating lattice walls
 *   (Batty, Bertails & Bridson 2007; Chentanez & Müller 2012), K_OPEN toward non-resident pages,
 *   and NB-FLIP's volume control in the RHS (Kim et al. 2007). It keeps the entry point, zero
 *   rows on every F_ALL brick off the liquid (P owns its row list, so no class bit gates a row and
 *   there is no FAIL.LIST check) and the cut's FAIL.UNSUPPORTED check. It writes X (the
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
import { mgLayout, mgLive, poolMGPCG } from "./pressure-mgpcg";

/** `operator` and `project` (WGSL reserves `operator`: its entry point is nbOperator). `tiered` is the engine's
 * (BrickStages.tiered); the default keeps the tiered engine's absent-page guard. `faces`: the solver binds pmgF
 * (lane P's MGPCG), so rowsF writes its face cache (mgLayout FC) and project reads it; and with the live lists
 * (mgLive) the MGPCG itself places rowsF over HB and projectF over HP (pressure-mgpcg.ts), so `operator` is the lean
 * pressureRowsList over F_ALL (the lists and the zero rows off them) and `project` pressureProjectRest (the FLAGS of
 * the bricks off HP). */
export function pressureOperatorKernels(L: BrickLayout, options: { readonly tiered?: boolean; readonly faces?: boolean; readonly cuts?: boolean } = {}) {
  // The cuts follow the K pool where the face cache is (a scene with K slots reads them; with none, the text is the dam's).
  const opt = { ...options, cuts: options.cuts ?? (options.faces === true && L.cap.k > 0) };
  if (opt.faces && mgLive(L)) {
    const M = mgLayout(L), live = poolMGPCG(L, { cuts: opt.cuts }).live!;
    const K = pressureCompositeKernels(L, {}, { tiered: opt.tiered, faces: M.FC, live: { hb: M.HBL, hp: M.HPL, rl: M.RLL, rln: M.RLN }, cuts: opt.cuts });
    return { operator: live.rowsList, project: K.projectRest! };
  }
  const K = pressureCompositeKernels(L, { rowsF: ["operator", "nbOperator"], projectF: ["project", "project"] }, { tiered: opt.tiered, faces: opt.faces ? mgLayout(L).FC : undefined, cuts: opt.cuts });
  return { operator: K.rowsF, project: K.projectF };
}
