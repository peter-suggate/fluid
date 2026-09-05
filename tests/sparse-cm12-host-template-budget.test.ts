import assert from "node:assert/strict";
import test from "node:test";

import { sparseCM12HostTemplateVariantsEnabled, sparseCM12TopologyPagePoolPlan } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

test("host rerung admission budgets expanded cells and rows, not only bricks", () => {
  assert.equal(sparseCM12HostTemplateVariantsEnabled(
    1_152 * 64, 1_152 * 240, 1_152, 8,
  ), true, "the established long-dam compatibility envelope must remain admitted");

  assert.equal(sparseCM12HostTemplateVariantsEnabled(
    1_900 * 64, 1_900 * 240, 1_900, 8,
  ), false,
  "a B4-heavy atlas must not hide its million-cell all-rung expansion behind 1,900 leaves");

  assert.equal(sparseCM12HostTemplateVariantsEnabled(
    250_001, 700_000, 100, 8,
  ), true,
  "crossing the legacy accepted-cell cutoff must not disable an affordable rerung catalogue");
});

test("topology page capacity records demand separately from its physical budget", () => {
  const limited = sparseCM12TopologyPagePoolPlan(700, true, 8, 640);
  assert.equal(limited.requestedPageCapacity, 700);
  assert.equal(limited.pageBudget, 640);
  assert.equal(limited.pageCapacity, 640);

  const admitted = sparseCM12TopologyPagePoolPlan(600, true, 8, 1_024);
  assert.equal(admitted.requestedPageCapacity, 600);
  assert.equal(admitted.pageBudget, 1_024);
  assert.equal(admitted.pageCapacity, 600);
});

test("topology budgets reject non-finite or fractional allocation sizes", () => {
  for (const invalid of [NaN, Infinity, -1, 0.5, Number.MAX_SAFE_INTEGER + 1]) {
    assert.throws(() => sparseCM12TopologyPagePoolPlan(700, true, 8, invalid), RangeError);
    assert.throws(() => sparseCM12TopologyPagePoolPlan(invalid, true, 8, 512), RangeError);
  }
  assert.equal(sparseCM12TopologyPagePoolPlan(700, true, 8, 0).pageCapacity, 0);
});
