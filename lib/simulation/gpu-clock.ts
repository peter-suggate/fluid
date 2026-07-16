const CLOCK_EPSILON_S = 1e-9;
const PRESENTATION_QUANTUM_S = 1 / 60;
const MAX_TALL_CELL_BATCH_DEPTH = 8;

/**
 * Number of CPU clock ticks that may be prepared before a GPU queue fence.
 *
 * The restricted tall-cell solver is commonly configured with a 4 ms outer
 * step. Fencing every step then limits it to one 4 ms advance per display
 * refresh even when the GPU has enough headroom to calculate several steps.
 * An uncoupled solve has no intervening CPU state dependency, so batch enough
 * work to cover one 60 Hz presentation interval and fence the batch once.
 *
 * Rigid-body scenes retain the one-step impulse handshake. The adaptive
 * quadtree method keeps its existing shallow batch because topology rebuilds
 * may deliberately stop a submission sequence.
 */
export function gpuBatchDepth(methodId: string, fixedDt_s: number, hasRigidBodies: boolean): number {
  if (hasRigidBodies) return 1;
  if (methodId === "quadtree-tall-cell") return 2;
  if (methodId !== "tall-cell" || !Number.isFinite(fixedDt_s) || fixedDt_s <= 0) return 1;
  return Math.min(MAX_TALL_CELL_BATCH_DEPTH, Math.max(1, Math.ceil((PRESENTATION_QUANTUM_S - CLOCK_EPSILON_S) / fixedDt_s)));
}

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
