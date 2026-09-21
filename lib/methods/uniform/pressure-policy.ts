/** Backend-neutral CM11a schedule and lagged budget policy. */
export const UNIFORM_CM11A_FULL_CYCLES = 3;
export const UNIFORM_CM11A_V_CYCLES = 4;
// Retain the validated six pre/post sweeps. Full-depth bounds and the
// finest-level acceptance gate protect deeper hierarchies from divergence.
export const UNIFORM_CM11A_PRE_SWEEPS = 6;
export const UNIFORM_CM11A_POST_SWEEPS = 6;
// Bounded projected-smoothing recovery after a rejected multigrid cycle.
// Only a finite iterate with a non-increasing residual can be published.
export const UNIFORM_CM11A_RECOVERY_BATCHES = 8;
export const UNIFORM_CM11A_RECOVERY_SWEEPS = 8;
// Recovery starts from a rejected solve, sometimes already under a loose
// absolute tolerance. Demand useful reduction before declaring recovery done.
export const UNIFORM_CM11A_RECOVERY_REDUCTION = 0.1;
export const UNIFORM_CM11A_PHI_PRESERVATION_LEVELS = 2;
// TallCells reports 1e-4 s^-1 as its GPU/single-precision absolute L-infinity
// tolerance; 1e-8 belongs to its double-precision CPU comparison. CM11a
// itself fixes the cycle schedule but does not prescribe a residual tolerance.
export const UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE = 1e-4;
export const UNIFORM_CM11A_COARSE_SWEEP_CAP = 4096;

export interface UniformCM11aSchedule {
  readonly fullCycles: number;
  readonly vCycles: number;
  readonly preSweeps: number;
  readonly postSweeps: number;
  readonly residualTolerance?: number;
}

export const DEFAULT_UNIFORM_CM11A_SCHEDULE: UniformCM11aSchedule = Object.freeze({
  fullCycles: UNIFORM_CM11A_FULL_CYCLES,
  vCycles: UNIFORM_CM11A_V_CYCLES,
  preSweeps: UNIFORM_CM11A_PRE_SWEEPS,
  postSweeps: UNIFORM_CM11A_POST_SWEEPS,
  residualTolerance: 10,
});

/**
 * Cycles the lagged budget never drops below. One complete cycle always runs,
 * so a step whose demand estimate is stale by a frame still projects against a
 * coarse-corrected pressure rather than against the previous step's field.
 */
export const UNIFORM_CM11A_MINIMUM_CYCLE_BUDGET = 1;
/** Cycles added above the last observed demand when it converged. */
export const UNIFORM_CM11A_DEFAULT_BUDGET_HEADROOM = 1;

export interface UniformCM11aCycleBudgetInput {
  /**
   * Cycles the latest *observed* step executed before its residual gate
   * tripped. Undefined until the first asynchronous diagnostics sample lands.
   */
  readonly lastExecutedCycles?: number;
  /** Startup prefix; omitted preserves the reference solver’s full startup. */
  readonly initialCycles?: number;
  /** Whether that step met tolerance; false requests a larger prefix. */
  readonly lastConverged?: boolean;
  readonly headroom: number;
  readonly minCycles?: number;
  /** The configured schedule: Full-Cycles + V-Cycles actually planned. */
  readonly maxCycles: number;
}

/**
 * How many cycles the next step encodes, from the last demand the async stats
 * readback reported.
 *
 * The GPU-side gate already stops a converged solve early, but a skipped pass
 * still costs its launch floor and its CPU encode, so the saving has to be
 * taken on the host by not encoding the tail at all. The signal is lagged by
 * however many frames the readback takes, which is why the rule is asymmetric:
 * a converged step uses observed demand plus the configured headroom, while a
 * step that used every encoded cycle *and still missed tolerance* doubles, so
 * an impact frame recovers its full schedule within one or two steps instead
 * of climbing one cycle at a time.
 */
export function uniformCM11aCycleBudget(input: UniformCM11aCycleBudgetInput): number {
  const maxCycles = Number.isFinite(input.maxCycles) ? Math.max(0, Math.floor(input.maxCycles)) : 0;
  const minCycles = Math.min(maxCycles, Math.max(0, Math.floor(
    Number.isFinite(input.minCycles) ? input.minCycles! : UNIFORM_CM11A_MINIMUM_CYCLE_BUDGET)));
  const executed = input.lastExecutedCycles;
  if (executed === undefined || !Number.isFinite(executed)) {
    const initial = Number.isFinite(input.initialCycles) ? Math.floor(input.initialCycles!) : maxCycles;
    return Math.min(maxCycles, Math.max(minCycles, initial));
  }
  const observed = Math.max(0, Math.floor(executed));
  const headroom = Number.isFinite(input.headroom) ? Math.max(0, Math.floor(input.headroom)) : 0;
  const demand = input.lastConverged
    ? observed + headroom
    : Math.max(2 * observed, observed + 2);
  return Math.min(maxCycles, Math.max(minCycles, demand));
}
