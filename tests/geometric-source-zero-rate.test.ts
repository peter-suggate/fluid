import assert from "node:assert/strict";
import test from "node:test";

import { createGeometricSourceWGSL } from
  "../lib/methods/adaptive-volume/geometric-source.wgsl";

test("a stopped continuous source hides stale per-cell rates", () => {
  const shader = createGeometricSourceWGSL({
    ledgerBaseFloats: 0,
    brickScratchBaseFloats: 64,
    brickCapacity: 8,
    sourceRateBaseFloats: 128,
    componentBaseWords: 256,
  });
  assert.match(shader,
    /fn geometricSourceRate\(cell:u32\)->f32\{\s*if\(!gsEnabled\(\)\)\{return 0\.0;\}return state\[GS_RATE\+cell\];\s*\}/);
  assert.match(shader,
    /fn initializeContinuousGeometricSource[\s\S]*let weight=gsWeight\(cell\);state\[GS_RATE\+cell\]=weight[\s\S]*gsCellWithWeight\(cell,weight\)/);
  assert.match(shader,
    /fn connectContinuousGeometricSource[\s\S]*if\(gsMember\(cell\)\)[\s\S]*if\(!gsMember\(cell\)\)\{continue;\}/);
  assert.match(shader,
    /fn gatherContinuousGeometricSourceWeights[\s\S]*let weight=state\[GS_RATE\+cell\]/);
  assert.match(shader,
    /fn publishContinuousGeometricSourceRates[\s\S]*let weight=state\[GS_RATE\+cell\]/);
  assert.equal(shader.match(/gsWeight\(cell\)/g)?.length, 1,
    "source weight should be defined once and evaluated only by initialization");
});
