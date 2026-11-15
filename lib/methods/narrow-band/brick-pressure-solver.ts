/**
 * LANE P OWNS THIS FILE. It registers P's pressure solver with the brick
 * engine's slot (brick-pressure-slot.ts, brick-abi.md §6.1).
 *
 * Export the MGPCG factory here when it runs on the pool. Until then it is
 * undefined, and the method refuses every scene on the brick engine. The
 * bring-up CG (brick-bringup-cg.ts) is for E's tests only and never
 * registers here (coordinator ruling, 2026-11-15).
 */
import type { PressureSolverFactory } from "./brick-pressure-slot";

export const brickPressureSolver: PressureSolverFactory | undefined = undefined;
