export interface GPUAdvancePlan {
  dt_s: number;
  nextTime_s: number;
  lag_s: number;
}

export function planGPUAdvance(requestedTime_s: number, currentTime_s: number, maximumDt_s: number): GPUAdvancePlan | undefined {
  if (requestedTime_s < currentTime_s) return undefined;
  const dt_s = Math.min(maximumDt_s, requestedTime_s - currentTime_s);
  const nextTime_s = currentTime_s + dt_s;
  return { dt_s, nextTime_s, lag_s: Math.max(0, requestedTime_s - nextTime_s) };
}

export interface TallCellStabilitySignals {
  nonFiniteCount: number;
  pressureRelativeResidual: number;
  maxComponentCfl: number;
  maxDivergenceBefore_s: number;
  maxDivergenceAfter_s: number;
  dt_s: number;
}

export function classifyTallCellStability(signals: TallCellStabilitySignals): string[] {
  const flags: string[] = [];
  const divergenceRatio = signals.maxDivergenceAfter_s / Math.max(signals.maxDivergenceBefore_s, 1e-30);
  const dimensionlessPostDivergence = signals.maxDivergenceAfter_s * signals.dt_s;
  if (signals.nonFiniteCount > 0) flags.push("non-finite-state");
  if (signals.pressureRelativeResidual > 0.1) flags.push("pressure-residual");
  if (signals.maxComponentCfl > 1) flags.push("advective-cfl");
  if (dimensionlessPostDivergence > 0.5) flags.push("post-projection-divergence");
  if (dimensionlessPostDivergence > 0.5 && divergenceRatio > 1.05) flags.push("projection-amplified-divergence");
  return flags;
}
