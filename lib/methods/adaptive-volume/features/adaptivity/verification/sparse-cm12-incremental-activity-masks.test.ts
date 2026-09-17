import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  createSparseCM12IncrementalActivityInitialWords,
  createSparseCM12IncrementalActivityLayout,
} from "../sparse-cm12-incremental-activity";
import { createSparseCM12IncrementalActivityWGSL } from
  "../sparse-cm12-incremental-activity.wgsl";

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
  assert.equal(layout.brickBoundaryLiquidFaceBaseWords, 432);
  assert.equal(layout.totalWords, 440);
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


test("one sibling-octet owner preserves the exhaustive detail maximum", () => {
  for (const resolution of [2, 4, 8]) {
    for (let count = 1; count <= resolution ** 3; count += 1) {
      const fixed = Array.from({ length: count }, (_, cell) =>
        (97 * cell + 13 * count) % 1000 - 500);
      let repeatedMaximum = 0;
      let electedMaximum = 0;
      for (let local = 0; local < count; local += 1) {
        const x = local % resolution;
        const y = Math.floor(local / resolution) % resolution;
        const z = Math.floor(local / (resolution * resolution));
        const group = [x & ~1, y & ~1, z & ~1];
        let sum = 0;
        for (let dz = 0; dz < 2; dz += 1) for (let dy = 0; dy < 2; dy += 1) {
          for (let dx = 0; dx < 2; dx += 1) {
            const child = group[0]! + dx + resolution * (group[1]! + dy
              + resolution * (group[2]! + dz));
            if (child < count) sum += fixed[child]!;
          }
        }
        repeatedMaximum = Math.max(repeatedMaximum, Math.abs(8 * fixed[local]! - sum));
      }
      for (let local = 0; local < count; local += 1) {
        const x = local % resolution;
        const y = Math.floor(local / resolution) % resolution;
        const z = Math.floor(local / (resolution * resolution));
        if ((x & 1) !== 0 || (y & 1) !== 0 || (z & 1) !== 0) continue;
        const children: number[] = [];
        for (let dz = 0; dz < 2; dz += 1) for (let dy = 0; dy < 2; dy += 1) {
          for (let dx = 0; dx < 2; dx += 1) {
            const child = x + dx + resolution * (y + dy + resolution * (z + dz));
            if (child < count) children.push(fixed[child]!);
          }
        }
        const sum = children.reduce((total, value) => total + value, 0);
        for (const value of children) {
          electedMaximum = Math.max(electedMaximum, Math.abs(8 * value - sum));
        }
      }
      assert.equal(electedMaximum, repeatedMaximum, `${resolution}^3 count ${count}`);
    }
  }
});

test("Face preparation reuses accepted rows without a transient DFRM plane", () => {
  const resident = readFileSync(new URL(
    "../../../webgpu-sparse-cm12-resident.ts",
    import.meta.url,
  ), "utf8");
  assert.match(resident, /dispatchAccepted\("prepareSparseCM12AcceptedFaceRows", "row"\)/);
  assert.doesNotMatch(resident, /prepareSparseCM12InteriorFaceTiles|prepareSparseCM12SeamFacePackets/);
  assert.doesNotMatch(resident, /DirtyFaceRowMask|compileSparseCM12DirtyFaceRowMasks/);
  assert.doesNotMatch(resident, /dispatchAccepted\("measureDivergenceDiagnostics"/);
  assert.match(resident, /dispatchAccepted\("collocateAndDiagnose", "cell"\)/);
  assert.match(resident, /dispatch\("reduceDivergenceDiagnostics", 1\)/);
});

