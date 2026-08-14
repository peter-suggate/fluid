/**
 * Queue-complete wall time for the solver's measured stepping window.
 *
 * `completedAt_ms` must be captured only after the final solver submission has
 * completed. Diagnostic work is timed separately, after the queue boundary it
 * depends on, so subtracting it cannot accidentally subtract the solver drain.
 */
export function queueCompleteSimulationWall_ms(
  runStarted_ms: number,
  completedAt_ms: number,
  diagnosticWall_ms: number,
): number {
  if (![runStarted_ms, completedAt_ms, diagnosticWall_ms].every(Number.isFinite)
    || completedAt_ms < runStarted_ms || diagnosticWall_ms < 0) {
    throw new RangeError("Queue-complete simulation timing requires finite ordered clocks and non-negative diagnostics");
  }
  return Math.max(0, completedAt_ms - runStarted_ms - diagnosticWall_ms);
}
