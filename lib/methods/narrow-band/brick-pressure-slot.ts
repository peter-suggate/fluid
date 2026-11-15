/**
 * The pressure slot contract (brick-abi.md §6.1): what sits between E's
 * `operator` and `project` and lane P's solver.
 *
 * Inputs (written by `operator` on every cell of every live F brick, F_ALL;
 * outside F_LIQ the rows are zero: K, DIAG, RHS and X are 0):
 * - `pres.RHS`: -h * sum_a (A u*)(c+e_a) - (A u*)(c), m^2/s;
 * - `pres.DIAG`: the sum over the six faces of A (liquid neighbour),
 *   A/theta (air neighbour), or 0 (closed);
 * - `pres.K0..2`: the coupling (>= 0) across the cell's lower face along a,
 *   held by the upper cell of the pair;
 * - `pres.X`: the warm start, 0 at non-liquid cells.
 * The system is DIAG X - sum K X_nbr = RHS over liquid cells, with X = 0 at
 * air. X = p dt / rho.
 *
 * Output: `pres.X` on the liquid cells of F_LIQ bricks.
 *
 * Gates:
 * - P owns args words 512..2047 and header words 128..191.
 * - `setup` opens slot 0 and closes the rest, every step.
 * - Slot k's last placement is a direct one-workgroup `check`, which opens
 *   slot k+1 or leaves it closed.
 * - `setup` clears STATE.PRESSURE_CAP. The last open slot's check sets it
 *   when the solve stops above tolerance.
 */
import type { BrickLayout } from "./brick-abi";
import { ARGS } from "./brick-abi";
import type { BrickPlacement } from "./brick-kernel";

/** P's args region (brick-abi.md §5.5). */
export const PRESSURE_ARGS = { FIRST: ARGS.LANE_P, END: ARGS.LANE_F } as const;

export interface PressureSolver {
  readonly label: string;
  /** True for E's bring-up solver: M1 tests only, never a gate or a timing claim. */
  readonly bringUp: boolean;
  /** Encoded slots record() allots. */
  readonly slots: number;
  /** Placed once, after `operator`. Opens slot 0 and closes the rest. */
  readonly setup: readonly BrickPlacement[];
  /** Slot k's placements, ending in its check. */
  slot(k: number): readonly BrickPlacement[];
  /** Placed once, before `project`. */
  readonly finish: readonly BrickPlacement[];
  /** Lane-owned buffers by name, in words; the engine allocates them zeroed. */
  readonly buffers: Readonly<Record<string, number>>;
  /** Header words (in P's 128..191) the receipt reports: this step's iterations, and the cumulative capped steps. */
  readonly receipt?: { readonly iterations?: number; readonly capped?: number };
}

export type PressureSolverFactory = (layout: BrickLayout) => PressureSolver;

/** Checks that a solver's gated dispatches stay in its args region. */
export function validatePressureSolver(s: PressureSolver): void {
  if (!Number.isInteger(s.slots) || s.slots < 1) throw new Error(`Brick engine: pressure solver ${s.label} declares ${s.slots} slots`);
  const all = [...s.setup, ...s.finish, ...Array.from({ length: s.slots }, (_, k) => s.slot(k)).flat()];
  for (const p of all) {
    if ("indirect" in p.dispatch) {
      const w = p.dispatch.indirect;
      // List arguments (words 0..392) are also valid sources: an ungated list-wide kernel.
      if (!(w >= 0 && w + 3 <= ARGS.FC_G64 + 3) && !(w >= PRESSURE_ARGS.FIRST && w + 3 <= PRESSURE_ARGS.END))
        throw new Error(`Brick engine: ${p.kernel.label} dispatches from args word ${w}, outside the lists and P's region`);
    }
  }
}
