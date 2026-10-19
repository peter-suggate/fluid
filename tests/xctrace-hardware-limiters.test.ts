import assert from "node:assert/strict";
import test from "node:test";
import { exclusiveHardwareSpans, hardwareSampleStage } from "../tools/xctrace-hardware-limiters";

test("hardware counters exclude foreign work, composites and boundary samples", () => {
  const spans = exclusiveHardwareSpans([
    { start: 0, end: 100, ours: true, labels: ["advect"] },
    { start: 30, end: 50, ours: false, labels: ["browser"] },
    { start: 100, end: 200, ours: true, labels: ["solve", "copy"] },
    { start: 210, end: 300, ours: true, labels: ["transfer"] },
    { start: 240, end: 270, ours: true, labels: ["transfer"] },
  ]);
  assert.equal(hardwareSampleStage(spans, 15, 10), "advect");
  assert.equal(hardwareSampleStage(spans, 29, 10), undefined);
  assert.equal(hardwareSampleStage(spans, 40, 10), undefined);
  assert.equal(hardwareSampleStage(spans, 55, 10), undefined);
  assert.equal(hardwareSampleStage(spans, 70, 10), "advect");
  assert.equal(hardwareSampleStage(spans, 150, 10), undefined);
  assert.equal(hardwareSampleStage(spans, 205, 10), undefined);
  assert.equal(hardwareSampleStage(spans, 250, 10), "transfer");
  assert.equal(hardwareSampleStage(spans, 305, 10), undefined);
  // Counter exports can revisit earlier timestamps after another ring's rows.
  assert.equal(hardwareSampleStage(spans, 15, 10), "advect");
});

test("overlapping own stages cannot lend counters to the latest starting stage", () => {
  const spans = exclusiveHardwareSpans([
    { start: 0, end: 100, ours: true, labels: ["advect"] },
    { start: 25, end: 125, ours: true, labels: ["transfer"] },
  ]);
  assert.equal(hardwareSampleStage(spans, 15, 5), "advect");
  assert.equal(hardwareSampleStage(spans, 50, 5), undefined);
  assert.equal(hardwareSampleStage(spans, 115, 5), "transfer");
});
