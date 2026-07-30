import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = (path: string) => readFileSync(new URL(path, import.meta.url), "utf8");

test("cone-traced SVO presentation records every encoded render-path item", () => {
  const renderer = source("../lib/webgpu-renderer.ts");
  const water = source("../lib/webgpu-water-pipeline.ts");
  const dry = source("../lib/webgpu-svo-dry-scene.ts");

  assert.match(renderer, /svoRenderMode === "svo"[^]*voxelRenderMode === "smooth"[^]*svoLightingMode === "cone" \|\| svoLightingMode === "gi"/);
  assert.match(renderer, /new GPUStageTimestampRecorder\(/);
  assert.match(renderer, /const detailedPresentationTrace = traceDetailedSvoRenderPath \? presentationTrace : undefined/);
  assert.match(renderer, /completeDetailedPresentationPhase\(\{ id: "inspection-overlay"/);
  assert.match(renderer, /completeDetailedPresentationPhase\(\{ id: "present"/);

  for (const phase of [
    "surface-extraction",
    "water-caustics",
    "water-front-interface",
    "water-back-interface",
    "optical-composite",
  ]) assert.match(water, new RegExp(`tracePhase\\?\\.\\(\\{ id: "${phase}"`), phase);

  for (const phase of ["svo-cone-lighting", "svo-primary", "svo-temporal"]) {
    assert.match(dry, new RegExp(`tracePhase\\?\\.\\(\\{ id: "${phase}"`), phase);
  }
});

test("fluid-only presentation keeps the compact fixed trace", () => {
  const renderer = source("../lib/webgpu-renderer.ts");

  assert.match(renderer, /if \(phase && !traceDetailedSvoRenderPath\) presentationTrace\?\.completePhase/);
  assert.match(renderer, /PRESENTATION_TRACE_PHASES\[fixedPresentationPhase\]/);
});

test("presentation boundaries ride the frame's passes and add no measurement fence", () => {
  const renderer = source("../lib/webgpu-renderer.ts");

  assert.match(renderer, /const encoder = presentationTrace\?\.instrument\(rawEncoder\) \?\? rawEncoder/,
    "the presentation encoder must be the instrumented one so boundaries can attach to real passes");
  assert.doesNotMatch(renderer, /presentationTrace\?\.boundary\(/,
    "marker passes are no longer how a presentation boundary is recorded");
});
