import assert from "node:assert/strict";
import test from "node:test";

import { diagnoseAdaptiveSurfaceBreakup, type AdaptiveSurfaceBreakupSample } from
  "../tools/adaptive-surface-breakup-diagnostic";

const sample = (overrides: Partial<AdaptiveSurfaceBreakupSample> = {}): AdaptiveSurfaceBreakupSample => ({
  step: 0, topologyEpoch: 1, conservedMass_m3: 1, handoffDrift_m3: 0,
  massVisibleVolume_m3: 1, reconstructedVolume_m3: 1, renderedVolume_m3: 1,
  massComponents: 1, reconstructedComponents: 1, renderedComponents: 1,
  ...overrides,
});

test("breakup diagnostic separates exact rho mass from reconstructed and rendered volume", () => {
  const failures = diagnoseAdaptiveSurfaceBreakup([
    sample(),
    sample({ step: 1, reconstructedVolume_m3: 0.8, reconstructedComponents: 3,
      renderedVolume_m3: 0.7, renderedComponents: 4 }),
  ], { mass_m3: 1e-8, volume_m3: 1e-3 });
  assert.deepEqual(failures.map((failure) => failure.stage), ["reconstruction", "renderer"]);
  assert.equal(failures.some((failure) => failure.stage === "mass"), false,
    "an exact mass receipt must not hide or be blamed for phi breakup");
});

test("breakup diagnostic attributes repeated topology-handoff drift before downstream breakup", () => {
  const failures = diagnoseAdaptiveSurfaceBreakup([
    sample(),
    sample({ step: 1, topologyEpoch: 2, conservedMass_m3: 1.0002,
      handoffDrift_m3: 0.0002, massVisibleVolume_m3: 0.95, massComponents: 2,
      reconstructedVolume_m3: 0.95, reconstructedComponents: 2,
      renderedVolume_m3: 0.95, renderedComponents: 2 }),
    sample({ step: 2, topologyEpoch: 3, conservedMass_m3: 0.9998,
      handoffDrift_m3: -0.0004, massVisibleVolume_m3: 0.9, massComponents: 3,
      reconstructedVolume_m3: 0.9, reconstructedComponents: 3,
      renderedVolume_m3: 0.9, renderedComponents: 3 }),
  ], { mass_m3: 1e-6, volume_m3: 1e-3 });
  assert.deepEqual(failures.slice(0, 3).map((failure) => failure.stage),
    ["mass", "handoff", "mass"]);
  assert.ok(failures.filter((failure) => failure.stage === "handoff").length === 2);
});

test("breakup diagnostic accepts stable repeated handoffs within fixed-point tolerance", () => {
  const failures = diagnoseAdaptiveSurfaceBreakup([
    sample(),
    sample({ step: 1, topologyEpoch: 2, conservedMass_m3: 1 + 2e-9,
      handoffDrift_m3: 2e-9 }),
    sample({ step: 2, topologyEpoch: 3, conservedMass_m3: 1 - 3e-9,
      handoffDrift_m3: -5e-9 }),
  ], { mass_m3: 1e-8, volume_m3: 1e-3 });
  assert.deepEqual(failures, []);
});

test("breakup diagnostic rejects unordered Dawn checkpoint samples", () => {
  assert.throws(() => diagnoseAdaptiveSurfaceBreakup([
    sample({ step: 2 }), sample({ step: 1 }),
  ], { mass_m3: 1e-8, volume_m3: 1e-3 }), /step ordered/);
});
