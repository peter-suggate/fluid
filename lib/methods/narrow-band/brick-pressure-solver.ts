/**
 * LANE P OWNS THIS FILE. It registers P's pressure solver with the brick
 * engine's slot (brick-pressure-slot.ts, brick-abi.md §6.1).
 *
 * The registered solver is the pool MGPCG (pressure-mgpcg.ts) on NB-FLIP's
 * composite rows: the native 4h rows (rowsC) open its setup, after the
 * engine's operator (the h rows), and the native faces' projection with the
 * h tiles' C X mean (projectC) closes its finish, before the engine's
 * project (the h faces). The bring-up CG (brick-bringup-cg.ts) is for E's
 * tests only and never registers here (coordinator ruling, 2026-11-15).
 */
import { ARGS, LIST } from "./brick-abi";
import type { PressureSolver, PressureSolverFactory } from "./brick-pressure-slot";
import { pressureCompositeKernels } from "./pressure-composite.wgsl";
import { poolMGPCG, type MGPCGOptions } from "./pressure-mgpcg";

/** The composite MGPCG: poolMGPCG with the native rows before its setup and the native projection after its finish. */
export const compositeMGPCG = (options: MGPCGOptions = {}): PressureSolverFactory => (L): PressureSolver => {
  const mg = poolMGPCG(L, options), K = pressureCompositeKernels(L);
  const pages = { indirect: ARGS.PER_LIST * LIST.C_ALL + ARGS.G1 };
  return {
    ...mg,
    label: "composite MGPCG (lane P)",
    setup: [{ kernel: K.rowsC, dispatch: pages }, ...mg.setup],
    finish: [...mg.finish, { kernel: K.projectC, dispatch: pages }],
  };
};

export const brickPressureSolver: PressureSolverFactory | undefined = compositeMGPCG();
