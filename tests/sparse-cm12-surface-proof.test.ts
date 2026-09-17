import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { resolveMethodValues } from "../lib/core/method-contract";
import {
  adaptiveMassMethod,
  adaptiveMassSolverOptions,
} from "../lib/methods/adaptive-volume/method";
import { SPARSE_CM12_ACTIVITY_POLICY, sparseCM12ActivityPolicy } from "../lib/methods/adaptive-volume/features/adaptivity/policy";

const resident = readFileSync(new URL(
  "../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts",
  import.meta.url,
), "utf8");
const shader = readFileSync(new URL(
  "../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts",
  import.meta.url,
), "utf8");

test("surface coarsening policy is enabled, bounded, and keeps QA forcing private", () => {
  assert.equal(SPARSE_CM12_ACTIVITY_POLICY.surfaceCoarseningEnabled, true);
  assert.equal(SPARSE_CM12_ACTIVITY_POLICY.surfaceDisplacementToleranceCells, 1);

  const sanitized = sparseCM12ActivityPolicy({
    activitySignals: true,
    surfaceCoarseningEnabled: false,
    surfaceDisplacementToleranceCells: -4,
    forcedSurfaceResolutionForQA: 4,
  });
  assert.equal(sanitized.activitySignals, true);
  assert.equal(sanitized.surfaceCoarseningEnabled, false);
  assert.equal(sanitized.surfaceDisplacementToleranceCells, 0);
  assert.equal(sanitized.forcedSurfaceResolutionForQA, 4);
  assert.equal(sparseCM12ActivityPolicy({
    forcedSurfaceResolutionForQA: 2,
  }).forcedSurfaceResolutionForQA, 2);
  assert.equal(sparseCM12ActivityPolicy({
    forcedSurfaceResolutionForQA: 3,
  }).forcedSurfaceResolutionForQA, undefined);
});

test("coarse-first is the production default", () => {
  assert.equal(adaptiveMassMethod.params.some((param) =>
    param.key === "resolutionMode"), false,
  "fixed all-fine/all-coarse modes must not remain in the production UI");
  assert.equal("resolutionMode" in adaptiveMassSolverOptions({ resolutionMode: "all-fine" }),
    false, "stale fixed-mode state must be discarded at the production boundary");
  const selector = adaptiveMassMethod.params.find((param) =>
    param.key === "selectorMode");
  assert.equal(selector?.kind, "select");
  if (selector?.kind === "select") assert.equal(selector.default, "coarse-first");
  assert.equal(adaptiveMassMethod.presetFor("balanced").selectorMode, "coarse-first");

  const values = resolveMethodValues(adaptiveMassMethod, "balanced", {});
  assert.equal(values.selectorMode, "coarse-first");
  assert.equal(adaptiveMassSolverOptions(values).activityPolicy?.activitySignals, true);
});

