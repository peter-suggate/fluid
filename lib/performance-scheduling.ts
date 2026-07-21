export type PerformanceScheduleInput = {
  targetFps: number;
  gpuAdvance_s: number;
  submissionBatchDepth: number;
  physicsPerAdvance_ms: number;
  renderPerFrame_ms: number;
  pressureSolvesPerAdvance: number;
};

const positive = (value: number, fallback: number) => Number.isFinite(value) && value > 0 ? value : fallback;
const nonNegative = (value: number) => Number.isFinite(value) && value > 0 ? value : 0;

export type GPUUtilizationInput = {
  physics_ms?: number;
  physicsCompletionInterval_ms?: number;
  presentation_ms?: number;
  presentationInterval_ms?: number;
};

export type AdvanceWallTimingInput = {
  cpuAdvanceEncode_ms?: number;
  gpuBatchWall_ms?: number;
  gpuAdvanceWall_ms?: number;
  gpuStep_ms?: number;
};

/** Last completed advance, kept separate from continuously sampled presentation
 * frames so a paused manual-step hitch cannot be overwritten by idle redraws. */
export function advanceWallBreakdown(input: AdvanceWallTimingInput | null | undefined) {
  if (!input) return null;
  const measured = (value: number | undefined) => value !== undefined && Number.isFinite(value) && value >= 0 ? value : undefined;
  const encode_ms = measured(input.cpuAdvanceEncode_ms);
  const queueFence_ms = measured(input.gpuBatchWall_ms);
  const timestampedGPU_ms = measured(input.gpuStep_ms);
  const wall_ms = measured(input.gpuAdvanceWall_ms);
  if (encode_ms === undefined && queueFence_ms === undefined && wall_ms === undefined) return null;
  return {
    encode_ms: encode_ms ?? 0,
    queueFence_ms: queueFence_ms ?? 0,
    timestampedGPU_ms: timestampedGPU_ms ?? 0,
    untimestampedQueue_ms: Math.max(0, (queueFence_ms ?? 0) - (timestampedGPU_ms ?? 0)),
    wall_ms: wall_ms ?? (encode_ms ?? 0) + (queueFence_ms ?? 0),
  };
}

/** Estimate queue occupancy from timestamped work and queue-confirmed cadence. */
export function measuredGPUUtilization(input: GPUUtilizationInput) {
  const share = (busy_ms: number | undefined, interval_ms: number | undefined) =>
    busy_ms !== undefined && interval_ms !== undefined
      && Number.isFinite(busy_ms) && Number.isFinite(interval_ms)
      && busy_ms >= 0 && interval_ms > 0
      ? Math.min(1, Math.max(0, busy_ms / interval_ms))
      : null;
  const physics = share(input.physics_ms, input.physicsCompletionInterval_ms);
  const presentation = share(input.presentation_ms, input.presentationInterval_ms);
  if (physics === null && presentation === null) return null;
  return {
    physics: physics ?? 0,
    presentation: presentation ?? 0,
    total: Math.min(1, (physics ?? 0) + (presentation ?? 0))
  };
}

/** Translate asynchronous GPU batching into presentation-frame-normalized rates. */
export function performanceSchedule(input: PerformanceScheduleInput) {
  const targetFps = positive(input.targetFps, 60);
  const gpuAdvance_ms = nonNegative(input.gpuAdvance_s) * 1000;
  const batchDepth = Math.max(1, Math.round(positive(input.submissionBatchDepth, 1)));
  const physicsPerAdvance_ms = nonNegative(input.physicsPerAdvance_ms);
  const renderPerFrame_ms = nonNegative(input.renderPerFrame_ms);
  const pressureSolvesPerAdvance = positive(input.pressureSolvesPerAdvance, 1);
  const frameBudget_ms = 1000 / targetFps;
  const advancesPerFrame = gpuAdvance_ms > 0 ? frameBudget_ms / gpuAdvance_ms : 0;
  const physicsPerFrame_ms = physicsPerAdvance_ms * advancesPerFrame;
  const gpuDemandPerFrame_ms = physicsPerFrame_ms + renderPerFrame_ms;
  const batchSimulation_ms = batchDepth * gpuAdvance_ms;
  const batchGPU_ms = batchDepth * physicsPerAdvance_ms;

  return {
    frameBudget_ms,
    gpuAdvance_ms,
    advancesPerFrame,
    physicsPerFrame_ms,
    renderPerFrame_ms,
    gpuDemandPerFrame_ms,
    demandPercent: gpuDemandPerFrame_ms / frameBudget_ms * 100,
    headroom_ms: frameBudget_ms - gpuDemandPerFrame_ms,
    pressureSolvesPerFrame: advancesPerFrame * pressureSolvesPerAdvance,
    pressureSolvesPerSecond: gpuAdvance_ms > 0 ? 1000 / gpuAdvance_ms * pressureSolvesPerAdvance : 0,
    pressureSolvesPerAdvance,
    pressureSolvesPerBatch: batchDepth * pressureSolvesPerAdvance,
    batchDepth,
    batchSimulation_ms,
    batchGPU_ms,
    realtimeFramesPerBatch: batchSimulation_ms / frameBudget_ms
  };
}
