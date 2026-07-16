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
  highCflCellCount: number;
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
  // Both reference methods deliberately use semi-Lagrangian transport at CFL
  // well above one. Flag escalation (the speed-rail threshold, or a genuinely
  // widespread excursion), not the papers' normal operating regime.
  if (signals.maxComponentCfl > 4 || signals.highCflCellCount >= 32) flags.push("advective-cfl");
  if (dimensionlessPostDivergence > 0.5) flags.push("post-projection-divergence");
  if (dimensionlessPostDivergence > 0.5 && divergenceRatio > 1.05) flags.push("projection-amplified-divergence");
  return flags;
}

export interface DriftOscillationSummary {
  driftSignChanges: number;
  latePeakToPeakDrift: number;
}

/** Summarize controller ringing over the late half of an ordered drift trace. */
export function summarizeDriftOscillation(drift: readonly number[]): DriftOscillationSummary {
  if (drift.length === 0) return { driftSignChanges: 0, latePeakToPeakDrift: 0 };
  const median3 = drift.map((value, index) => {
    if (index === 0 || index === drift.length - 1) return value;
    const values = [drift[index - 1], value, drift[index + 1]].sort((a, b) => a - b);
    return values[1];
  });
  const late = median3.slice(Math.floor(median3.length / 2));
  let driftSignChanges = 0;
  let previousSign = 0;
  for (let index = 1; index < late.length; index += 1) {
    const difference = late[index] - late[index - 1];
    const sign = difference > 0 ? 1 : difference < 0 ? -1 : 0;
    if (sign === 0) continue;
    if (previousSign !== 0 && sign !== previousSign) driftSignChanges += 1;
    previousSign = sign;
  }
  return {
    driftSignChanges,
    latePeakToPeakDrift: Math.max(...late) - Math.min(...late)
  };
}
