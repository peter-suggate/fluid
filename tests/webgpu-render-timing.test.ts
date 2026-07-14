import assert from "node:assert/strict";
import test from "node:test";
import { decodeRenderStageTimestamps } from "../lib/webgpu-renderer";

const timestamps = [
  0n, 2_000_000n,
  3_000_000n, 6_000_000n,
  7_000_000n, 11_000_000n,
  12_000_000n, 17_000_000n,
  18_000_000n, 24_000_000n,
  25_000_000n, 32_000_000n
];

test("raster timestamps retain each presentation stage", () => {
  assert.deepEqual(decodeRenderStageTimestamps(timestamps, true, true), {
    total_ms: 27,
    surfaceExtraction_ms: 2,
    dryScene_ms: 3,
    interfaces_ms: 9,
    opticalComposite_ms: 6,
    upscale_ms: 7
  });
});

test("unchanged surfaces contribute no stale extraction time", () => {
  const result = decodeRenderStageTimestamps(timestamps, true, false);
  assert.equal(result.surfaceExtraction_ms, undefined);
  assert.equal(result.total_ms, 25);
});

test("ray-march timestamps use the ray pass and shared upscale only", () => {
  assert.deepEqual(decodeRenderStageTimestamps(timestamps, false, false), {
    total_ms: 9,
    opticalComposite_ms: 2,
    upscale_ms: 7
  });
});
