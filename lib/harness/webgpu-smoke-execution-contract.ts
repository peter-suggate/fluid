export interface WebGPUSmokeExecutionResult {
  readonly method: string;
  readonly steps: number;
  readonly grid: readonly [number, number, number];
  readonly info: {
    readonly encodedSteps?: number;
    readonly submittedTime_s?: number;
    readonly completedTime_s?: number;
  };
}

export interface WebGPUSmokeExecutionContract {
  readonly exactSteps?: number;
  readonly maxDt_s?: number;
  readonly timeTolerance_s?: number;
}

/** Executor invariants are scenario-independent protocol checks, not physics assertions. */
export function webGPUSmokeExecutionFailures(
  results: readonly WebGPUSmokeExecutionResult[],
  contract: WebGPUSmokeExecutionContract,
): string[] {
  const failures: string[] = [];
  const fail = (condition: boolean, message: string) => { if (!condition) failures.push(message); };
  const referenceGrid = results[0]?.grid;
  for (const result of results) {
    if (referenceGrid) fail(result.grid.every((value, axis) => value === referenceGrid[axis]),
      `${result.method} grid ${result.grid.join("x")} differs from ${referenceGrid.join("x")}`);
    if (contract.exactSteps === undefined) continue;
    const dt_s = contract.maxDt_s;
    fail(dt_s !== undefined && Number.isFinite(dt_s) && dt_s > 0,
      `exact-step execution requires a finite positive maxDt_s`);
    if (!(dt_s !== undefined && Number.isFinite(dt_s) && dt_s > 0)) continue;
    const expectedTime_s = contract.exactSteps * dt_s;
    const tolerance = contract.timeTolerance_s ?? 1e-9;
    fail(result.steps === contract.exactSteps,
      `${result.method} accepted ${result.steps} outer steps; expected exactly ${contract.exactSteps}`);
    fail(result.info.encodedSteps === contract.exactSteps,
      `${result.method} encoded ${result.info.encodedSteps ?? "unknown"} steps; expected exactly ${contract.exactSteps}`);
    fail(Math.abs((result.info.submittedTime_s ?? -Infinity) - expectedTime_s) <= tolerance,
      `${result.method} submitted time ${result.info.submittedTime_s} differs from exact checkpoint ${expectedTime_s}`);
    fail(Math.abs((result.info.completedTime_s ?? -Infinity) - expectedTime_s) <= tolerance,
      `${result.method} completed time ${result.info.completedTime_s} differs from fenced checkpoint ${expectedTime_s}`);
  }
  return failures;
}
