import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(new URL("../tools/benchmark-svo-dry-frame-gpu.ts", import.meta.url), "utf8");

test("SVO dry-frame benchmark uses the generic one-shot GPU trace", () => {
  assert.match(source, /new GPUPerformanceTraceRecorder\(/);
  assert.match(source, /\[\{ id: "dry-scene", label: "SVO traversal \+ dry shading" \}\]/);
  assert.match(source, /traceRecorder\.boundary\(encoder, `\$\{label\} trace start`\)/);
  assert.match(source, /traceRecorder\.boundary\(encoder, `\$\{label\} trace end`\)/);
  assert.match(source, /traceRecorder\.resolve\(encoder\)/);
  assert.match(source, /samples\.push\(trace\.total_ms\)/);
  assert.doesNotMatch(source, /createQuerySet|resolveQuerySet|timestampWrites|GPU(?:Compute|Render)PassTimestampWrites/);
});

test("SVO dry-frame benchmark can capture the configured production phase partition", () => {
  assert.match(source, /FLUID_SVO_DRY_FRAME_PHASE_TRACE=1/);
  assert.match(source, /await DynamicGPUPerformanceTraceRecorder\.create\(/);
  assert.match(source, /attempt < 3 && !configuredPhaseTrace/);
  assert.match(source, /recorder\.completePhase\(encoder, phase\)/);
  assert.match(source, /configuredPhaseTrace,/);
});

test("SVO dry-frame benchmark exposes a clean render-only xctrace lane", () => {
  assert.match(source, /FLUID_SVO_DRY_FRAME_PROFILE_SECONDS/);
  assert.match(source, /FLUID_SVO_DRY_FRAME_SCENE/);
  assert.match(source, /phase: "constructed"/);
  assert.match(source, /label: `SVO render frame \$\{frame\}`/);
  assert.match(source, /phase: "result"/);
  assert.match(source, /if \(profileSeconds > 0\)[^]*process\.exit\(0\);[^]*Timing helpers/);
});

test("SVO dry-frame benchmark can force serialized wall timing without changing the rendered frame", () => {
  assert.match(source, /FLUID_SVO_DRY_FRAME_TIMING selects wall \(default\)/);
  assert.match(source, /const timingMode = process\.env\.FLUID_SVO_DRY_FRAME_TIMING \?\? "wall"/);
  assert.match(source, /const forceWallTiming = timingMode === "wall"/);
  assert.match(source, /GPUPerformanceTraceRecorder\.supported\(device\) && !forceWallTiming/);
  assert.match(source, /device\.queue\.onSubmittedWorkDone\(\)/);
});

test("SVO dry-frame benchmark measures only the production GLOBAL path", () => {
  assert.doesNotMatch(source, /FLUID_SVO_DRY_FRAME_LIGHTING|SVO_LIGHTING_MODES|setLightingMode|lightingMode/);
  assert.match(source, /const shadingPathRaw = process\.env\.FLUID_SVO_DRY_FRAME_SHADING \?\? "split"/);
  assert.match(source, /renderer\.setLightingOptions/);
  assert.match(source, /solver\.encodeSceneMaintenance\(initialScenePublication\)/,
    "the benchmark must publish the staged live scene before timing render consumers");
});

test("SVO dry-frame benchmark isolates primary-seam-closure cost and quality", () => {
  assert.match(source, /const SVO_VIEW_UNIFORM_FLOATS = 104/);
  assert.match(source, /size: SVO_VIEW_UNIFORM_FLOATS \* Float32Array\.BYTES_PER_ELEMENT/);
  assert.match(source, /FLUID_SVO_DRY_FRAME_PRIMARY_SEAM_CLOSURE/);
  assert.match(source, /same-process-interleaved-off-on/);
  assert.match(source, /cycle % 2 === 0 \? \[false, true\][^]*: \[true, false\]/);
  assert.match(source, /disabledMedian_ms:/);
  assert.match(source, /enabledMedian_ms:/);
  assert.match(source, /overheadPercent:/);
  assert.match(source, /pairedMedianOverhead_ms:/);
  assert.match(source, /disabledVsFull:/);
  assert.match(source, /enabledVsFull:/);
  assert.match(source, /enabledCloserToFull/);
  assert.match(source, /darkenedByRefinement/);
  assert.match(source, /silhouetteSignedDifference/);
});

test("SVO dry-frame benchmark exposes a render-only brick-size override and topology accounting", () => {
  assert.match(source, /FLUID_SVO_DRY_FRAME_BRICK_SIZE/);
  assert.match(source, /renderBrickSize === undefined \? \{\} : \{ renderBrickSize \}/);
  assert.match(source, /authoredBrickSize: scene\.voxelDomain\.brickSize_cells/);
  assert.match(source, /structuralBytes:/);
  assert.match(source, /materialOwners:/);
  assert.match(source, /payload:/);
  assert.match(source, /FLUID_SVO_DRY_FRAME_RAW_OUT/);
  assert.match(source, /FLUID_SVO_DRY_FRAME_CONFIGURED_RAW_OUT/);
  assert.match(source, /configuredImageHashFnv1a32/);
});

test("SVO dry-frame benchmark exposes exact static-primary coherence accounting", () => {
  assert.match(source, /FLUID_SVO_DRY_FRAME_COHERENCE/);
  assert.match(source, /static-primary requires FLUID_SVO_DRY_FRAME_SHADING=split/);
  assert.match(source, /const primaryCoherenceKey = rayCoherenceMode === "static-primary"/);
  assert.match(source, /steadyPrimaryRaysTracedPerFrame: 0/);
  assert.match(source, /shadowAndConeRaysRemainPerFrame: true/);
  assert.match(source, /residentBytesPerPixel: SVO_DRY_SPLIT_RESIDENT_BYTES_PER_PIXEL/);
});

test("brick-size comparison fixes the real hose frame and compares packed image error", () => {
  const comparison = readFileSync(new URL("../tools/benchmark-svo-brick-size-gpu.ts", import.meta.url), "utf8");
  assert.match(comparison, /FLUID_SVO_BRICK_SIZE_SCENE \?\? "hose-tank"/);
  assert.match(comparison, /FLUID_SVO_BRICK_SIZE_WIDTH \?\? 660/);
  assert.match(comparison, /FLUID_SVO_BRICK_SIZE_HEIGHT \?\? 662/);
  assert.match(comparison, /FLUID_SVO_DRY_FRAME_BRICK_SIZE: String\(brickSize\)/);
  assert.match(comparison, /simulation: "absent\/frozen"/);
  assert.match(comparison, /differingHalfWords/);
  assert.match(comparison, /relativeLuminance:/);
  assert.match(comparison, /speedup4Vs8/);
});
