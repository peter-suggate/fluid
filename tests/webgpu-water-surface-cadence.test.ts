import assert from "node:assert/strict";
import test from "node:test";

import { frameInterval_ms } from "../lib/core/frame-pacing";
import { shouldUpdateWaterSurface } from "../lib/core/webgpu-water-pipeline";

test("paused manual revisions bypass the wall-clock surface cadence", () => {
  const lastExtractionAt_ms = 100;
  const insideCadenceAt_ms = lastExtractionAt_ms + 0.25 * frameInterval_ms();

  assert.equal(shouldUpdateWaterSurface(
    7, 8, lastExtractionAt_ms, insideCadenceAt_ms,
  ), false, "running presentation should retain its paced surface");
  assert.equal(shouldUpdateWaterSurface(
    7, 8, lastExtractionAt_ms, insideCadenceAt_ms, true,
  ), true, "a paused manual step must extract its new revision immediately");
  assert.equal(shouldUpdateWaterSurface(
    8, 8, lastExtractionAt_ms, insideCadenceAt_ms, true,
  ), false, "a paused repaint without a simulation step must retain its surface");
});

test("running surface extraction remains paced", () => {
  const lastExtractionAt_ms = 100;
  assert.equal(shouldUpdateWaterSurface(
    3, 4, lastExtractionAt_ms, lastExtractionAt_ms + frameInterval_ms(),
  ), true);
  assert.equal(shouldUpdateWaterSurface(
    -1, 0, lastExtractionAt_ms, lastExtractionAt_ms,
  ), true, "the first extraction is never paced away");
});
