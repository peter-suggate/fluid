/** Adaptive plans contain all V-cycles, then all Full-Cycles. */
export function nextUniformPressureCorrection(current: number, vCycles: number, fullCycles: number,
  residual: number, previous: number, accuracy: number): {cycle: number; accuracy: number} | undefined {
  const stalled = !Number.isFinite(residual) || residual > previous * 0.5;
  let cycle = current + 1;
  if (stalled && cycle < vCycles && fullCycles > 0) cycle = vCycles;
  if (cycle >= vCycles + fullCycles) return undefined;
  return {cycle, accuracy: stalled ? (accuracy === 1 ? 0.1 : 0) : accuracy};
}
