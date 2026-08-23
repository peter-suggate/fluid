import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  createSparseCM12IncrementalActivityInitialWords,
  createSparseCM12IncrementalActivityLayout,
} from "../lib/methods/adaptive-mass/sparse-cm12-incremental-activity";
import { createSparseCM12IncrementalActivityWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-incremental-activity.wgsl";

test("ACT1 uses stamp masks without allocating a brick list", () => {
  const layout = createSparseCM12IncrementalActivityLayout({
    baseWords: 128,
    brickCount: 8,
  });
  assert.equal(layout.headerBaseWords, 128);
  assert.equal(layout.brickStampBaseWords, 144);
  assert.equal(layout.brickVelocityStampBaseWords, 152);
  assert.equal(layout.brickTopologyStateBaseWords, 160);
  assert.equal(layout.brickCensusStateBaseWords, 168);
  assert.equal(layout.scoreHistogramBaseWords, 176);
  assert.equal(layout.totalWords, 432);
  assert.equal("brickListBaseWords" in layout, false);

  const words = createSparseCM12IncrementalActivityInitialWords(layout);
  assert.equal(words.length, layout.totalWords - layout.headerBaseWords);
  assert.deepEqual([...words.subarray(32, 40)], Array(8).fill(0xffff_ffff));
});

test("ACT1 claims and consumes brick stamps directly", () => {
  const layout = createSparseCM12IncrementalActivityLayout({
    baseWords: 0,
    brickCount: 32,
  });
  const wgsl = createSparseCM12IncrementalActivityWGSL(layout, 2);
  assert.match(wgsl, /atomicExchange\(&activity\[ACTIVITY_BRICK_STAMP\+brick\],generation\)/);
  assert.match(wgsl,
    /atomicExchange\(&activity\[ACTIVITY_BRICK_VELOCITY_STAMP\+brick\],\s*generation\)/);
  assert.match(wgsl,
    /if\(previous!=generation\)\{incrementalActivityPublishFaceBrickClosure\(brick\);\}/);
  assert.match(wgsl,
    /if\(dx>=0&&dx<span&&dy>=0&&dy<span&&dz>=0&&dz<span\)\{continue;\}/);
  assert.doesNotMatch(wgsl,
    /generation\);\s*_\s*=incrementalActivityClaimBrick\(brick\);\s*\/\/ Closure depends/);
  assert.match(wgsl, /fn finalizeIncrementalActivityMasks\(\)/);
  assert.match(wgsl, /return select\(INVALID,invocation,invocation<ACTIVITY_BRICK_COUNT/);
  assert.doesNotMatch(wgsl, /ACTIVITY_BRICK_LIST/);
  assert.doesNotMatch(wgsl, /atomicCompareExchangeWeak/);
});

test("resident activity consumers use direct brick-domain dispatch", () => {
  const source = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
    import.meta.url,
  ), "utf8");
  assert.match(source, /dispatch\("measureBrickActivity", this\.incrementalActivityLayout\.brickCount\)/);
  assert.match(source, /dispatch\("clearSparseCM12RetiredFaceVelocitySupport",\s*this\.incrementalActivityLayout\.brickCount\)/);
  assert.doesNotMatch(source, /activityIndirectArguments/);
  assert.doesNotMatch(source, /dispatchActivity\(/);
});

test("BFA1 prepares and projects accepted rows without a transient DFRM plane", () => {
  const resident = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
    import.meta.url,
  ), "utf8");
  assert.match(resident, /dispatch\("prepareSparseCM12InteriorFaceTiles"/);
  assert.match(resident, /dispatch\("prepareSparseCM12SeamFacePackets"/);
  assert.doesNotMatch(resident, /DirtyFaceRowMask|compileSparseCM12DirtyFaceRowMasks/);
  assert.match(resident, /dispatchAccepted\("measureDivergenceDiagnostics", "cell"\)/);
});
