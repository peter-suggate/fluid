/**
 * LANE P OWNS THIS FILE. It registers P's pressure solver with the brick
 * engine's slot (brick-pressure-slot.ts, brick-abi.md §6.1).
 *
 * The registered solver is the pool MGPCG (pressure-mgpcg.ts) on NB-FLIP's
 * composite rows: the h tiles' root φ (tileRoots) runs before the engine's
 * operator, the native 4h rows (rowsC) open its setup, after the
 * engine's operator (the h rows), and the native faces' projection with the
 * h tiles' C X mean (projectC) closes its finish, before the engine's
 * project (the h faces). The separating walls' outer loop (MGPCGOptions.walls)
 * is on: the composite rows keep the wall state in FLAGS bits 8-10. The bring-up CG (brick-bringup-cg.ts) is for E's
 * tests only and never registers here (coordinator ruling, 2026-11-15).
 */
import { ARGS, LIST } from "./brick-abi";
import type { PressureSolver, PressureSolverFactory } from "./brick-pressure-slot";
import { pressureCompositeKernels } from "./pressure-composite.wgsl";
import { poolMGPCG, type MGPCGOptions } from "./pressure-mgpcg";

/** The composite MGPCG: poolMGPCG with the native rows before its setup and the native projection after its finish. */
export const compositeMGPCG = (options: MGPCGOptions = {}): PressureSolverFactory => (L): PressureSolver => {
  const { publish, live, core, ...mg } = poolMGPCG(L, { ...options, walls: options.walls ?? true }), K = pressureCompositeKernels(L);
  const pages = { indirect: ARGS.PER_LIST * LIST.C_ALL + ARGS.G1 };
  // The live lists (pressure-mgpcg.ts mgLive): the engine's operator is the lists' pass, so the h tiles' held fit and
  // the h rows run over HB at the setup's head, before the native rows, and the h faces' projection over HP closes the
  // finish (the F_ALL path's order: operator, rowsC, ..., projectC, publish, project).
  if (live) return {
    ...mg,
    label: "composite MGPCG (lane P)",
    setup: [...live.args, ...live.tileRoots, ...live.rows, { kernel: K.rowsC, dispatch: pages }, ...core.setup],
    finish: [...core.finish, { kernel: K.projectC, dispatch: pages }, ...publish, ...live.project],
  };
  return {
    ...mg,
    label: "composite MGPCG (lane P)",
    // The h tiles' held fit (root φ) once a step, before the operator's cold starts read it.
    before: [{ kernel: K.tileRoots, dispatch: { indirect: ARGS.PER_LIST * LIST.F_ALL + ARGS.G1 } }],
    setup: [{ kernel: K.rowsC, dispatch: pages }, ...core.setup],
    // Then the region pressure change (pmgC PDELTA) from the final root values, for E's next region heat.
    finish: [...core.finish, { kernel: K.projectC, dispatch: pages }, ...publish],
  };
};

export const brickPressureSolver: PressureSolverFactory | undefined = compositeMGPCG();
