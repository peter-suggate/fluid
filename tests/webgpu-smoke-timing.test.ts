import assert from "node:assert/strict";
import test from "node:test";
import { queueCompleteSimulationWall_ms } from "../tools/webgpu-smoke-timing";

test("Dawn stepping wall includes the final queue drain and excludes only diagnostics", () => {
  assert.equal(queueCompleteSimulationWall_ms(100, 190, 12), 78);
});

test("Dawn stepping wall rejects malformed clock evidence", () => {
  assert.throws(() => queueCompleteSimulationWall_ms(100, 99, 0), /finite ordered clocks/);
  assert.throws(() => queueCompleteSimulationWall_ms(100, 110, -1), /finite ordered clocks/);
  assert.throws(() => queueCompleteSimulationWall_ms(100, Number.NaN, 0), /finite ordered clocks/);
});
