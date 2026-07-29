import assert from "node:assert/strict";
import test from "node:test";
import {
  performanceShaderVariant,
  usePerformanceInstrumentationStore,
} from "../lib/stores/performance-instrumentation-store";

test("timeline capture does not select an instrumented shader variant", () => {
  const store = usePerformanceInstrumentationStore.getState();
  store.setMode("off");
  const productionGeneration = usePerformanceInstrumentationStore.getState().shaderGeneration;

  usePerformanceInstrumentationStore.getState().setMode("timeline");
  const timeline = usePerformanceInstrumentationStore.getState();
  assert.equal(timeline.enabled, true);
  assert.equal(timeline.shaderActivityEnabled, false);
  assert.equal(timeline.shaderGeneration, productionGeneration);
  assert.deepEqual(performanceShaderVariant(), {
    enabled: false,
    generation: productionGeneration,
    cacheKey: "production",
  });
});

test("activity capture changes the compile-time shader generation", () => {
  usePerformanceInstrumentationStore.getState().setMode("timeline");
  const before = usePerformanceInstrumentationStore.getState().shaderGeneration;

  usePerformanceInstrumentationStore.getState().setMode("activity");
  const activity = performanceShaderVariant();
  assert.equal(activity.enabled, true);
  assert.equal(activity.generation, before + 1);
  assert.equal(activity.cacheKey, `activity-${before + 1}`);

  usePerformanceInstrumentationStore.getState().setMode("off");
  const production = performanceShaderVariant();
  assert.equal(production.enabled, false);
  assert.equal(production.generation, before + 2);
  assert.equal(production.cacheKey, "production");
});

test("the backward-compatible enabled state selects the production-WGSL timeline", () => {
  usePerformanceInstrumentationStore.getState().setMode("off");
  const before = usePerformanceInstrumentationStore.getState().shaderGeneration;
  usePerformanceInstrumentationStore.getState().setEnabled(true);
  const enabled = usePerformanceInstrumentationStore.getState();
  assert.equal(enabled.mode, "timeline");
  assert.equal(enabled.shaderActivityEnabled, false);
  assert.equal(enabled.shaderGeneration, before);
});
