import assert from "node:assert/strict";
import test from "node:test";
import { sparseCM12DistanceSweeps, sparseCM12ReturnPasses } from "../lib/methods/adaptive-volume/sharpening-controls";
import { ALGORITHM_TUNING_PARAMS } from "../lib/methods/adaptive-volume/features/algorithms/definition";

test("adaptive return controls retain zero and bound the dispatch budget", () => {
  for (const [key, clamp, fallback] of [
    ["distanceSweeps", sparseCM12DistanceSweeps, 8],
    ["returnPasses", sparseCM12ReturnPasses, 4],
  ] as const) {
    assert.equal(clamp(undefined), fallback);
    assert.equal(clamp(NaN), fallback);
    assert.equal(clamp(0), 0);
    assert.equal(clamp(-3), 0);
    assert.equal(clamp(100), 16);
    assert.equal(clamp(2.7), 3);
    const spec = ALGORITHM_TUNING_PARAMS.find(p => p.key === key)!;
    assert.equal(spec.default, fallback);
    assert.equal(spec.kind, "number");
  }
});
