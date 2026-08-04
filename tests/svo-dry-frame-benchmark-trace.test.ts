import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(new URL("../tools/benchmark-svo-dry-frame-gpu.ts", import.meta.url), "utf8");

/**
 * Everything except the paired record-scale lane.
 *
 * That lane owns its own `beginningOfPassWriteIndex`/`endOfPassWriteIndex` pair
 * per pass, and it has to: both shared recorders read a frame as one *chain* of
 * boundaries, and on this path Dawn/Metal schedules a standalone marker pass
 * away from the work it brackets — measured, a 95 ms frame decoded as five
 * boundaries spanning 327 us with one of them 90 ms out of order, so
 * `decodeGPUTimestampPartition` correctly rejected every attempt. The
 * prohibition below is about the *frame-timing* number the benchmark reports,
 * which must come from the shared recorder so the tool and the app cannot
 * measure a frame two different ways; it was never about forbidding a lane from
 * asking a pass how long it took.
 */
const frameTimingSource = (() => {
  const start = source.indexOf("if (recordScaleMultipliers) {");
  const end = source.indexOf("// Build the shipped garden lighting-study world");
  assert.ok(start > 0 && end > start, "the record-scale lane must stay one contiguous block");
  return source.slice(0, start) + source.slice(end);
})();

test("SVO dry-frame benchmark uses pass-local GPU timestamps for an honest frame span", () => {
  assert.match(source, /new GPUPassTimestampRecorder\(/);
  assert.match(source, /recorder\.instrument\(encoder\)/);
  assert.match(source, /recorder\.resolve\(encoder\)/);
  assert.match(source, /samples\.push\(reading\.span_ms\)/);
  assert.doesNotMatch(frameTimingSource, /createQuerySet|resolveQuerySet|timestampWrites|GPU(?:Compute|Render)PassTimestampWrites/);
  // And the record-scale lane still reports per-pass hardware time rather than
  // a wall clock, which is the property W0's gate is actually stated in.
  assert.match(source, /FLUID_SVO_DRY_FRAME_RECORD_SCALE/);
  assert.match(source, /passSamples\.get\(arm\.multiplier\)!\.push\(reading\)/);
});

test("SVO dry-frame benchmark can capture the configured production phase partition", () => {
  assert.match(source, /FLUID_SVO_DRY_FRAME_PHASE_TRACE=1/);
  assert.match(source, /new GPUPassTimestampRecorder\(device, 512, "SVO configured phase trace"\)/);
  assert.match(source, /attempt < 3 && !configuredPhaseTrace/);
  assert.match(source, /passTimestampPerformanceTrace\(/);
  assert.match(source, /frameSpan_ms: Number\(reading\.span_ms\.toFixed\(4\)\)/);
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
  assert.match(source, /GPUPassTimestampRecorder\.supported\(device\) && !forceWallTiming/);
  assert.match(source, /device\.queue\.onSubmittedWorkDone\(\)/);
});

test("SVO dry-frame benchmark measures only the production GLOBAL path", () => {
  assert.doesNotMatch(source, /FLUID_SVO_DRY_FRAME_LIGHTING|SVO_LIGHTING_MODES|setLightingMode|lightingMode/);
  assert.match(source, /const shadingPathRaw = process\.env\.FLUID_SVO_DRY_FRAME_SHADING \?\? "split"/);
  assert.match(source, /renderer\.setLightingOptions/);
  assert.match(source, /solver\.encodeSceneMaintenance\(initialScenePublication\)/,
    "the benchmark must publish the staged live scene before timing render consumers");
  assert.match(source, /FLUID_SVO_DRY_FRAME_RADIANCE_FEEDBACK/);
  assert.match(source, /const radianceFeedbackEnabled = process\.env\.FLUID_SVO_DRY_FRAME_RADIANCE_FEEDBACK === "1"/);
  assert.match(source, /FLUID_SVO_DRY_FRAME_FEEDBACK_FRAMES/);
  assert.match(source, /for \(let frame = 1; frame < radianceFeedbackFrames; frame \+= 1\)[^]*solver\.encodeSceneMaintenance\(feedbackEncoder\)/,
    "the Dawn image must settle the same continuous radiance feedback that browser presentation frames encode");
  assert.match(source, /scene: \{[^]*radianceFeedbackFrames,/,
    "the capture must report its exact feedback-settlement depth");
  assert.match(source, /staticFeedbackIdle = !solver\.encodeSceneMaintenance\(idleEncoder\)/);
  assert.match(source, /assert\.equal\(staticFeedbackIdle, true,[^]*static live scene must stop encoding feedback/,
    "the Dawn smoke must reject a renderer that keeps processing a converged static scene");
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
