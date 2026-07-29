import assert from "node:assert/strict";
import test from "node:test";
import { webGPUSmokeExecutionFailures } from "../tools/webgpu-smoke-execution-contract";

const result = (method: string, steps = 2, grid: [number, number, number] = [4, 3, 2]) => ({
  method, steps, grid,
  info: { encodedSteps: steps, submittedTime_s: 0.008, completedTime_s: 0.008 },
});

test("exact execution protocol is scenario agnostic", () => {
  assert.deepEqual(webGPUSmokeExecutionFailures([result("a"), result("b")], {
    exactSteps: 2, maxDt_s: 0.004,
  }), []);
  assert.match(webGPUSmokeExecutionFailures([result("a", 1)], {
    exactSteps: 2, maxDt_s: 0.004,
  }).join("\n"), /accepted 1 outer steps/);
});

test("cross-method result grids must be collocated", () => {
  assert.match(webGPUSmokeExecutionFailures([result("a"), result("b", 2, [5, 3, 2])], {})[0],
    /grid 5x3x2 differs/);
});
