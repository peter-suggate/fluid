/** Simulation step used by every example in CM12 Sec. 4. */
export const UNIFORM_PAPER_DT_S = 1 / 30;

/** Paper mode only admits complete 1/30 s advances. */
export function uniformPaperAdvanceReady(requestedTime_s: number, currentTime_s: number): boolean {
  return requestedTime_s - currentTime_s >= UNIFORM_PAPER_DT_S - 1e-9;
}
