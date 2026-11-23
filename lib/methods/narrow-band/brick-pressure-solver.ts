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
 * With the scene's bodies (the factory's context), the rigid coupling is monolithic in the solve (MGPCGOptions.bodies:
 * the dynamic bodies' rows in A', coordinator ruling 2026-11-17, the brick engine only).
 */
import { ARGS, LIST } from "./brick-abi";
import type { PressureSolver, PressureSolverFactory } from "./brick-pressure-slot";
import { pressureCompositeKernels, ROWS_C_PAGES } from "./pressure-composite.wgsl";
import { mgLayout, poolMGPCG, type MGPCGOptions } from "./pressure-mgpcg";
import type { BrickBodiesFrame } from "./bodies-binding.wgsl";
import type { BrickLayout } from "./brick-abi";

/** What the engine tells a solver of the scene beyond the layout: the rigid bodies' frame when the scene binds bodies. */
export interface BrickPressureContext { readonly bodies?: BrickBodiesFrame; readonly cuts?: boolean }
/** A solver factory that takes the engine's context (assignable to PressureSolverFactory). */
export type BrickPressureFactory = (L: BrickLayout, context?: BrickPressureContext) => PressureSolver;

/** The composite MGPCG: poolMGPCG with the native rows before its setup and the native projection after its finish. */
/** monolithic (default true): the context's bodies implicit in the solve; false keeps the explicit couple (comparisons). */
export const compositeMGPCG = (options: MGPCGOptions & { readonly monolithic?: boolean } = {}): BrickPressureFactory => (L, context): PressureSolver => {
  // The cuts follow the K pool, as the MGPCG's (rowsC latches a liquid native row in a cut brick), unless the engine
  // says no stamp of this schedule takes a K slot (lane E, deferred statics: context.cuts false).
  const cuts = options.cuts ?? context?.cuts ?? L.cap.k > 0;
  const { monolithic = true, ...rest } = options, bodies = rest.bodies ?? (monolithic ? context?.bodies : undefined);
  const { publish, live, core, ...mg } = poolMGPCG(L, { ...rest, cuts, walls: rest.walls ?? true, ...(bodies ? { bodies } : {}) }), K = pressureCompositeKernels(L, {}, { cuts, faces: cuts ? mgLayout(L).FC : undefined });
  const pages = { indirect: ARGS.PER_LIST * LIST.C_ALL + ARGS.G1 }, rowsPages = ROWS_C_PAGES;
  // The live lists (pressure-mgpcg.ts mgLive): the engine's operator is the lists' pass, so the h tiles' held fit and
  // the h rows run over HB at the setup's head, before the native rows, and the h faces' projection over HP closes the
  // finish (the F_ALL path's order: operator, rowsC, ..., projectC, publish, project).
  if (live) return {
    ...mg,
    label: "composite MGPCG (lane P)",
    setup: [...live.args, ...live.tileRoots, ...live.rows, { kernel: K.rowsC, dispatch: rowsPages }, ...core.setup],
    finish: [...core.finish, { kernel: K.projectC, dispatch: pages }, ...publish, ...live.project],
  };
  return {
    ...mg,
    label: "composite MGPCG (lane P)",
    // The h tiles' held fit (root φ) once a step, before the operator's cold starts read it.
    before: [{ kernel: K.tileRoots, dispatch: { indirect: ARGS.PER_LIST * LIST.F_ALL + ARGS.G1 } }],
    setup: [{ kernel: K.rowsC, dispatch: rowsPages }, ...core.setup],
    // Then the region pressure change (pmgC PDELTA) from the final root values, for E's next region heat.
    finish: [...core.finish, { kernel: K.projectC, dispatch: pages }, ...publish],
  };
};

export const brickPressureSolver: PressureSolverFactory | undefined = compositeMGPCG();
