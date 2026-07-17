const CLOCK_EPSILON_S = 1e-9;
const MAX_PRESENTATION_BATCH_DEPTH = 32;
const PRESENTATION_IN_FLIGHT_BATCHES = 2;

/**
 * Number of CPU clock ticks that may be prepared before a GPU queue fence.
 *
 * The restricted tall-cell solver is commonly configured with a 4 ms outer
 * step. Fencing every step then limits it to one 4 ms advance per display
 * refresh even when the GPU has enough headroom to calculate several steps.
 * Batch enough cheap GPU work to cover one presentation interval and fence
 * the batch once. Octree solves use the same policy; their topology and surface
 * state remain ordered in the submitted command stream. Tall-cell and octree
 * rigid feedback is deliberately partitioned at the presentation boundary:
 * the solver captures every impulse, the controller merges the frame-sized
 * interval, then distributes that aggregate over the next fixed rigid steps.
 *
 * Other rigid-body methods retain the one-step handshake. The adaptive
 * quadtree method keeps its existing shallow uncoupled batch because topology
 * rebuilds may deliberately stop a submission sequence.
 */
export function gpuBatchDepth(methodId: string, fixedDt_s: number, hasRigidBodies: boolean, targetFps = 60): number {
  const presentationBatched = methodId === "tall-cell" || methodId === "octree";
  if (presentationBatched && Number.isFinite(fixedDt_s) && fixedDt_s > 0) {
    const presentationQuantum_s = 1 / Math.min(120, Math.max(24, Number.isFinite(targetFps) ? targetFps : 60));
    return Math.min(MAX_PRESENTATION_BATCH_DEPTH, Math.max(1, Math.ceil((presentationQuantum_s - CLOCK_EPSILON_S) / fixedDt_s)));
  }
  if (hasRigidBodies) return 1;
  return methodId === "quadtree-tall-cell" ? 2 : 1;
}

/**
 * Maximum CPU-prepared transport window ahead of queue-confirmed GPU state.
 *
 * Keep a second presentation-sized batch prepared for the paths without a
 * same-step CPU dependency so requestAnimationFrame jitter cannot starve the
 * GPU at a completion boundary. Other methods retain their stricter topology
 * or rigid-feedback handshakes.
 */
export function gpuInFlightStepLimit(methodId: string, fixedDt_s: number, hasRigidBodies: boolean, targetFps = 60): number {
  const batchDepth = gpuBatchDepth(methodId, fixedDt_s, hasRigidBodies, targetFps);
  return methodId === "tall-cell" || methodId === "octree" ? batchDepth * PRESENTATION_IN_FLIGHT_BATCHES : batchDepth;
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
