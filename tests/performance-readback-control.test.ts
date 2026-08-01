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

test("new accounting uses one generic trace schema", () => {
  const source = read("../lib/performance-trace.ts");
  assert.match(source, /interface PerformanceTrace/);
  assert.match(source, /decodeGPUTimestampPartition/);
  assert.match(source, /Performance intervals overlap/);
});

