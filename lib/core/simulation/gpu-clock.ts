/** Clocks are compared in seconds; a step is never smaller than this. */
export const CLOCK_EPSILON_S = 1e-9;

/**
 * WebGPU transport may prepare one step beyond the last completed GPU state,
 * but it must not prepare another until that submitted step has completed.
 */
export function gpuCanAcceptNextStep(requestedTime_s: number, completedTime_s: number): boolean {
  return requestedTime_s <= completedTime_s + CLOCK_EPSILON_S;
}

/** Ignore stale completion callbacks from a solver that was reset or replaced. */
export function commitGPUCompletion(requestedTime_s: number, completedTime_s: number, callbackTime_s: number): number {
  if (!Number.isFinite(callbackTime_s) || callbackTime_s <= completedTime_s + CLOCK_EPSILON_S) return completedTime_s;
  return Math.min(requestedTime_s, callbackTime_s);
}

/** O(1) host-clock collapse for GPU authority; no intermediate host state exists. */
export function collapseGPUFixedSteps(accumulator_s: number, dt_s: number) {
  if (!Number.isFinite(accumulator_s) || accumulator_s < 0 || !Number.isFinite(dt_s) || dt_s <= 0) {
    return { steps: 0, remainder_s: Math.max(0, Number.isFinite(accumulator_s) ? accumulator_s : 0) };
  }
  const steps = Math.floor((accumulator_s + 1e-12) / dt_s);
  return { steps, remainder_s: Math.max(0, accumulator_s - steps * dt_s) };
}
