import assert from "node:assert/strict";
import test from "node:test";
import { ProgressHistory } from "../lib/core/progress-history";
import type { ResourceActivity, ResourceReadinessSnapshot } from "../lib/core/resource-readiness";

const plugin = { id: "fluid", lane: "fluid" } as unknown as ResourceReadinessSnapshot["plugins"][string]["plugin"];

function activity(phase: string, work?: { completed: number; total: number }): ResourceActivity {
  return { id: `fluid.${phase}`, pluginId: "fluid", lane: "fluid", label: phase, phase, completed: 0, total: 3,
    startedAt_ms: 0, retainingPrevious: false, ...(work ? { work: { ...work, unit: "pipelines" } } : {}) };
}
const preparing = (value: ResourceActivity) => ({ plugins: { fluid: { state: "preparing", label: "", usable: false, activity: value, plugin } } }) as unknown as ResourceReadinessSnapshot;
const ready = { plugins: { fluid: { state: "ready", label: "", usable: true, plugin } } } as unknown as ResourceReadinessSnapshot;

function memory() {
  const items = new Map<string, string>();
  return { getItem: (key: string) => items.get(key) ?? null, setItem: (key: string, value: string) => { items.set(key, value); } };
}

test("phases are weighted by how long they took last time, and only a finished run is recorded", () => {
  const storage = memory();
  const history = new ProgressHistory(storage);
  // No record: no estimate, however far along the count is.
  history.observe("scene", preparing(activity("allocation")), 0);
  assert.equal(history.estimate("scene", activity("allocation"), undefined, 100), undefined);
  history.observe("scene", preparing(activity("solver-pipelines")), 1000);
  history.observe("scene", ready, 10_000);

  // Same work again: allocation took 1 s of 10, compiling 9 s.
  const next = new ProgressHistory(storage);
  next.observe("scene", preparing({ ...activity("allocation"), startedAt_ms: 20_000 }), 20_000);
  next.observe("scene", preparing({ ...activity("solver-pipelines"), startedAt_ms: 20_000 }), 21_000);
  const halfway = next.estimate("scene", { ...activity("solver-pipelines", { completed: 62, total: 124 }), startedAt_ms: 20_000 }, 0.5, 25_500)!;
  assert.equal(halfway.fraction, 0.55);
  // Covered 5.5 s of record in 5.5 s: on pace, 4.5 s left.
  assert.equal(Math.round(halfway.remaining_ms), 4500);
  // A different scene shares nothing.
  assert.equal(next.estimate("other", activity("solver-pipelines"), 0.5, 25_500), undefined);
});

test("comparison panes keep independent runs while sharing completed timings", () => {
  const storage = memory();
  const history = new ProgressHistory(storage);
  history.observe("scene", preparing(activity("allocation")), 0, "a");
  history.observe("scene", preparing(activity("allocation")), 100, "b");
  history.observe("scene", preparing(activity("solver-pipelines")), 1000, "a");
  // B finishing must not delete A's run or include A's compile phase.
  history.observe("scene", ready, 2100, "b");
  history.observe("scene", ready, 10_000, "a");
  const record = JSON.parse(storage.getItem("fluid.progress-history.v1")!)["fluid|scene"];
  assert.deepEqual(record, { phases: [["allocation", 1000], ["solver-pipelines", 9000]], total_ms: 10_000 });

  const a = { ...activity("allocation"), startedAt_ms: 20_000 };
  const b = { ...activity("allocation"), startedAt_ms: 21_000 };
  history.observe("scene", preparing(a), 20_000, "a");
  history.observe("scene", preparing(b), 21_000, "b");
  history.observe("scene", preparing({ ...a, phase: "solver-pipelines" }), 21_000, "a");
  assert.equal(history.estimate("scene", a, 0.5, 21_500, "a")?.fraction, 0.55);
  assert.equal(history.estimate("scene", b, 0.5, 21_500, "b")?.fraction, 0.05);
  assert.equal(history.estimate("scene", a, 0.5, 21_500, "b"), undefined);
  assert.equal(history.estimate("other", a, 0.5, 21_500, "a"), undefined);
});

test("a run that is superseded rather than finished leaves the record alone", () => {
  const storage = memory();
  const history = new ProgressHistory(storage);
  history.observe("scene", preparing(activity("allocation")), 0);
  history.observe("scene", preparing({ ...activity("allocation"), startedAt_ms: 5 }), 4000);
  history.observe("scene", { plugins: { fluid: { state: "idle", label: "", usable: false, plugin } } } as unknown as ResourceReadinessSnapshot, 6000);
  assert.equal(storage.getItem("fluid.progress-history.v1"), null);
});
