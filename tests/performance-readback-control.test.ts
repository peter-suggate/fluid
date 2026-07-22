import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { useUIStore } from "../lib/stores/ui-store";

const read = (path: string) => readFileSync(new URL(path, import.meta.url), "utf8");

test("performance readbacks are user-switchable and default on", () => {
  const initial = useUIStore.getState().performanceReadbacksEnabled;
  assert.equal(initial, true);
  useUIStore.getState().setPerformanceReadbacksEnabled(false);
  assert.equal(useUIStore.getState().performanceReadbacksEnabled, false);
  useUIStore.getState().setPerformanceReadbacksEnabled(initial);
});

test("maximum-throughput mode gates every recurring profiler readback at encode or submit time", () => {
  const renderer = read("../lib/webgpu-renderer.ts");
  const uniform = read("../lib/webgpu-uniform-eulerian.ts");
  const tallCell = read("../lib/webgpu-eulerian.ts");
  const water = read("../lib/webgpu-water-pipeline.ts");
  const viewport = read("../components/WebGPUViewport.tsx");
  const panel = read("../components/PerformancePanel.tsx");

  assert.match(renderer, /this\.performanceReadbacksEnabled&&now_ms-this\.lastGPUReadbackAt_ms>=250/,
    "solver statistics polling must stop");
  assert.match(renderer, /sampleRenderGPU=Boolean\(this\.performanceReadbacksEnabled&&this\.renderQuerySet/,
    "presentation timestamps and their resolve must not be encoded");
  assert.match(uniform, /if \(!this\.performanceReadbacksEnabled \|\| !this\.querySet/,
    "uniform and octree timing ranges must not be allocated");
  assert.match(uniform, /const productionPhaseProbeActive = this\.performanceReadbacksEnabled/,
    "queue-boundary command-buffer splitting must remain disabled in maximum-throughput mode");
  assert.match(uniform, /productionPhaseProbeActive \? \(phase, completedEncoder\) =>/,
    "the octree surface must receive no split callback outside an active intrusive sample");
  assert.match(uniform, /if \(!this\.performanceReadbacksEnabled \|\| \(this\.info\.encodedSteps/,
    "uniform and octree diagnostic maps must return before encoding copies");
  assert.match(tallCell, /if\(!this\.performanceReadbacksEnabled\|\|!this\.querySet/,
    "restricted tall-cell timing ranges must not be allocated");
  assert.match(tallCell, /if\(!this\.performanceReadbacksEnabled\|\|this\.stepIndex===0\)return this\.info/,
    "restricted tall-cell diagnostic maps must return before encoding copies");
  assert.match(water, /if \(!this\.performanceReadbacksEnabled \|\| this\.adaptiveDiagnosticPending/,
    "adaptive presentation diagnostic copies must not be encoded");
  assert.match(viewport, /renderer\.setPerformanceReadbacksEnabled\(ui\.performanceReadbacksEnabled\)/,
    "the live UI state must reach the renderer before each draw");
  assert.match(viewport, /simulation\.backend === "webgpu" && rendererRef\.current && useUIStore\.getState\(\)\.performanceReadbacksEnabled/,
    "GPU picking must fall back to the CPU bounds path rather than map a result");
  assert.match(panel, /OFF · MAX SPEED/);
});
