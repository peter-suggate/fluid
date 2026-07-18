import assert from "node:assert/strict";
import test from "node:test";
import { decodeRenderStageTimestamps } from "../lib/webgpu-renderer";

const timestamps = [
  0n, 2_000_000n,
  3_000_000n, 6_000_000n,
  7_000_000n, 11_000_000n,
  12_000_000n, 17_000_000n,
  18_000_000n, 24_000_000n,
  25_000_000n, 32_000_000n,
  33_000_000n, 35_000_000n,
  36_000_000n, 39_000_000n
];

test("raster timestamps retain each presentation stage", () => {
  assert.deepEqual(decodeRenderStageTimestamps(timestamps, true), {
    total_ms: 32,
    surfaceExtraction_ms: 2,
    dryScene_ms: 3,
    interfaces_ms: 9,
    sprayFront_ms: 2,
    sprayBack_ms: 3,
    sprayRender_ms: 5,
    opticalComposite_ms: 6,
    upscale_ms: 7
  });
});

test("unchanged surfaces contribute no stale extraction time", () => {
  const result = decodeRenderStageTimestamps(timestamps, false);
  assert.equal(result.surfaceExtraction_ms, undefined);
  assert.equal(result.total_ms, 30);
});

test("disabled spray contributes no stale render time", () => {
  const result = decodeRenderStageTimestamps(timestamps, true, false);
  assert.equal(result.sprayFront_ms, 0);
  assert.equal(result.sprayBack_ms, 0);
  assert.equal(result.sprayRender_ms, 0);
  assert.equal(result.total_ms, 27);
});
