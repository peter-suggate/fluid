import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const read = (path: string) => readFileSync(new URL(path, import.meta.url), "utf8");

test("the old profiler toggle and stage-capture UI are absent", () => {
  const sources = [
    read("../lib/stores/ui-store.ts"),
    read("../components/WebGPUViewport.tsx"),
    read("../components/PerformancePanel.tsx"),
    read("../lib/webgpu-renderer.ts"),
    read("../lib/webgpu-water-pipeline.ts"),
  ].join("\n");
  assert.doesNotMatch(sources, /performanceReadbacksEnabled|setPerformanceReadbacksEnabled|gpuStageCapture/);
});

test("performance panel can bypass every trace producer without disabling correctness synchronization", () => {
  const panel = read("../components/PerformancePanel.tsx");
  const store = read("../lib/stores/performance-instrumentation-store.ts");
  const controller = read("../lib/simulation/controller.ts");
  const renderer = read("../lib/webgpu-renderer.ts");
  const uniform = read("../lib/webgpu-uniform-eulerian.ts");
  const tall = read("../lib/webgpu-eulerian.ts");
  assert.equal((panel.match(/aria-label="Detailed performance capture"/g) ?? []).length, 1,
    "capture mode has exactly one binary control");
  assert.equal((panel.match(/role="switch"/g) ?? []).length, 1);
  assert.doesNotMatch(panel, /MEASUREMENT LOAD|<select|Timeline only/);
  assert.match(panel, /Correctness synchronization remains active/);
  // Renegotiated by docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md item P0.1: the product
  // default is uninstrumented. "activity" compiles per-workgroup atomics into
  // every MGPCG entry point, so it is opt-in through the panel (or through
  // FLUID_PERFORMANCE_TRACES in the Dawn harness) rather than always paid for.
  // The assertion still pins that the initial mode is authored explicitly.
  assert.match(store, /mode: "off"/);
  assert.match(store, /enabled: false/,
    "detailed trace and activity capture are opt-in, not the default");
  assert.match(store, /shaderActivityEnabled: false/);
  for (const source of [controller, renderer, uniform, tall]) {
    assert.match(source, /usePerformanceInstrumentationStore\.getState\(\)/);
  }
  assert.match(renderer, /measurementInstrumentationEnabled[\s\S]*shouldTracePresentation/);
  assert.match(uniform, /measurementInstrumentationEnabled[\s\S]*shouldTracePhysics/);
  assert.match(tall, /shouldTrace=measurementInstrumentationEnabled&&/);
});

test("new accounting uses one generic trace schema", () => {
  const source = read("../lib/performance-trace.ts");
  assert.match(source, /interface PerformanceTrace/);
  assert.match(source, /decodeGPUTimestampPartition/);
  assert.match(source, /Performance intervals overlap/);
});

test("GPU lanes fall back to explicit queue-wall samples without timestamp-query", () => {
  const recorder = read("../lib/performance-trace.ts");
  const renderer = read("../lib/webgpu-renderer.ts");
  const uniform = read("../lib/webgpu-uniform-eulerian.ts");
  const tall = read("../lib/webgpu-eulerian.ts");
  assert.match(recorder, /class GPUQueueWallPerformanceTraceRecorder/);
  assert.match(recorder, /measurementSource: "gpu-queue-wall"/);
  for (const source of [renderer, uniform, tall]) {
    assert.match(source, /GPUQueueWallPerformanceTraceRecorder/);
    assert.match(source, /QueueTraceRead/);
    assert.match(source, /trace\s*\?\?\s*.*QueueTraceRead/);
  }
});
