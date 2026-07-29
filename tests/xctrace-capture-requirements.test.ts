import assert from "node:assert/strict";
import test from "node:test";
import { assertCompleteOccupancyReport } from "../tools/profile-mini-dam-xctrace";

const report = (overrides: Record<string, unknown> = {}) => ({
  attribution: { mode: "full", compositeBuckets: 0 },
  counters: { meanOccupancy: 0.5, partitionCount: 4,
    occupancyTrace: [{ t: 0, p: [0.5], label: "stage" }] },
  passes: [{ counterSamples: 4, occupancy: 0.5 }],
  frames: { count: 1, samples: [{}], captures: [{}],
    anchor: "Open coupled topology ready-commit gate" },
  timeline: { intervals: [{ encoderId: "0x1", start: 0,
    label: "Open coupled topology ready-commit gate" }] },
  ...overrides,
}) as never;

test("xctrace publication requires full labels and task-attributed occupancy", () => {
  assert.doesNotThrow(() => assertCompleteOccupancyReport(report()));
  assert.throws(() => assertCompleteOccupancyReport(report({
    attribution: { mode: "scoped", compositeBuckets: 8 },
  })), /label isolation is scoped/);
  assert.throws(() => assertCompleteOccupancyReport(report({
    counters: { partitionCount: 0, occupancyTrace: [] },
  })), /Compute Occupancy was not captured/);
  assert.throws(() => assertCompleteOccupancyReport(report({
    passes: [{ counterSamples: 0 }],
  })), /no labelled GPU task received occupancy/);
  assert.throws(() => assertCompleteOccupancyReport(report({
    frames: { count: 2, samples: [{}, {}], captures: [{}, {}] },
  })), /instead of exactly one/);
  assert.throws(() => assertCompleteOccupancyReport(report({
    timeline: { intervals: [{ encoderId: "0x1" }, { encoderId: "0x1" }] },
  })), /appears more than once/);
  assert.throws(() => assertCompleteOccupancyReport(report({
    frames: { count: 1, samples: [{}], captures: [{}], anchor: "Pressure interior" },
  })), /periodic but not a semantic GPU start/);
  assert.throws(() => assertCompleteOccupancyReport(report({
    frames: { count: 1, samples: [{}], captures: [{}],
      anchor: "Open coupled topology ready-commit gate" },
    timeline: { intervals: [{ encoderId: "0x1", start: 0, label: "Pressure interior" }] },
  })), /frame origin does not contain/);
  assert.doesNotThrow(() => assertCompleteOccupancyReport(report({
    timeline: { intervals: [
      { encoderId: "0x1", start: 8, label: "Publish deterministic fine-seed brick residency" },
      { encoderId: "0x2", start: 1_200, label: "Open coupled topology ready-commit gate" },
    ] },
  })), "legitimate work from an overlapping command buffer may precede the semantic gate");
  assert.throws(() => assertCompleteOccupancyReport(report({
    timeline: { intervals: [
      { encoderId: "0x1", start: 0, label: "Publish deterministic fine-seed brick residency" },
      { encoderId: "0x2", start: 5_001, label: "Open coupled topology ready-commit gate" },
    ] },
  })), /frame origin does not contain/);
});
