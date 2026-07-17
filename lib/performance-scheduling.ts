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
