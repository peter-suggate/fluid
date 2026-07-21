import assert from "node:assert/strict";
import test from "node:test";

import { buildSVORenderStageBenchmarkPlan } from "../tools/svo-render-stage-benchmark-contract";

test("render-stage plan rotates fixed-resolution shadow and temporal isolation variants", () => {
  const plan = buildSVORenderStageBenchmarkPlan({
    revision: "deadbeef", baseUrl: "http://localhost:3000/lab?existing=1",
    resolution: { width: 1280, height: 720 }, cycles: 3,
  });
  assert.equal(plan.runs.length, 9);
  assert.deepEqual(plan.runs.map(({ variant }) => variant), [
    "production", "full-rate-shadows", "primary-only",
    "full-rate-shadows", "primary-only", "production",
    "primary-only", "production", "full-rate-shadows",
  ]);
  assert.ok(plan.runs.every(({ outputResolution, internalResolution }) => outputResolution.width === 1280
    && outputResolution.height === 720 && internalResolution.width === 1280 && internalResolution.height === 720));
  assert.match(plan.runs[0].url, /existing=1/);
  assert.match(plan.runs[0].url, /svoShadowVisibility=1/);
  assert.match(plan.runs[0].url, /svoTemporal=1/);
  assert.equal(plan.runs[2].expectedTimingContextFragment, "shadow-off:temporal-off:smooth:svo");
  assert.match(plan.captureInstructions[0], /1280x720/);
});

test("render-stage plan rejects ambiguous dimensions and non-web targets", () => {
  assert.throws(() => buildSVORenderStageBenchmarkPlan({ revision: "x", baseUrl: "file:///tmp/app", resolution: { width: 1, height: 1 } }), /HTTP/);
  assert.throws(() => buildSVORenderStageBenchmarkPlan({ revision: "x", baseUrl: "http://localhost", resolution: { width: 0, height: 720 } }), /positive/);
});

