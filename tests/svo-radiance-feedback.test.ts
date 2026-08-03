import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import {
  LIVE_SVO_RADIANCE_FEEDBACK,
  liveSvoDerivedWorklistWGSL,
  liveSvoRadianceFeedbackWGSL,
} from "../lib/webgpu-svo-live-derived-builder";

const worldSource = readFileSync(new URL("../lib/webgpu-octree-sparse-bricks.ts", import.meta.url), "utf8");

test("live SVO diffuse feedback is opt-in, contractive, and temporally partitioned", () => {
  assert.equal(LIVE_SVO_RADIANCE_FEEDBACK.enabledByDefault, false,
    "in-place partial generations must not ship until feedback publishes through ping-pong atlases");
  assert.equal(LIVE_SVO_RADIANCE_FEEDBACK.phaseCount, 4);
  assert.equal(LIVE_SVO_RADIANCE_FEEDBACK.settleCycleCount, 24);
  assert.equal(LIVE_SVO_RADIANCE_FEEDBACK.settleFrameCount, 96);
  assert.equal(LIVE_SVO_RADIANCE_FEEDBACK.maximumTransportAlbedo, 0.85);
  assert.match(liveSvoDerivedWorklistWGSL, /partitioned=params\.limits\.w==2u/);
  assert.match(liveSvoDerivedWorklistWGSL, /dirty%max\(params\.source\.w,1u\)!=params\.source\.z/);
  assert.match(worldSource, /options\.radianceFeedback \?\? LIVE_SVO_RADIANCE_FEEDBACK\.enabledByDefault/);
  assert.match(worldSource, /encodeRadianceFeedback\([\s\S]*liveDerivedFeedbackPhase/);
  assert.match(worldSource, /if \(!encoded\) return !deferDerived && this\.encodeLiveRadianceFeedback\(encoder\)/,
    "static presentation frames must finish feedback instead of freezing the first partition");
  assert.match(worldSource, /liveDerivedFeedbackFramesRemaining = this\.liveDerivedBuilder\.radianceFeedbackEnabled[^]*LIVE_SVO_RADIANCE_FEEDBACK\.settleFrameCount/,
    "every real source rebuild must restart one bounded convergence window");
  assert.match(worldSource, /private encodeLiveRadianceFeedback[^]*liveDerivedFeedbackFramesRemaining <= 0[^]*liveDerivedFeedbackPhase = \(this\.liveDerivedFeedbackPhase \+ 1\)[^]*liveDerivedFeedbackFramesRemaining -= 1/,
    "feedback must rotate only while convergence work remains, then become idle");
});

test("feedback replaces previous radiance with emission plus bounded reflected light", () => {
  assert.match(liveSvoRadianceFeedbackWGSL, /transportAlbedo=clamp\([^;]*vec3f\(0\.85\)\)/);
  assert.match(liveSvoRadianceFeedbackWGSL, /let outgoing=max\(material\.emissiveRoughness\.rgb,vec3f\(0\.0\)\)\+transportAlbedo\*incident/);
  assert.doesNotMatch(liveSvoRadianceFeedbackWGSL, /outgoing\s*=\s*previous|previous\s*\+\s*outgoing/);
  assert.match(liveSvoRadianceFeedbackWGSL, /distance=2\.5[^]*distance=6\.0[^]*distance=18\.0[^]*distance=54\.0/);
  assert.match(liveSvoRadianceFeedbackWGSL, /svoEnvironmentDiffuseIrradiance\(environment\[0\],normal\)\/PI/);
  assert.match(liveSvoRadianceFeedbackWGSL, /incident\+=directIncident\(worldPosition,originCells,normal\)/,
    "authored direct light becomes a first-bounce source before temporal feedback adds higher orders");
  assert.match(liveSvoRadianceFeedbackWGSL, /textureSampleLevel\(opacitySource,atlasSampler,uv,0\.0\)/,
    "feedback uses filtered atlas samples instead of nearest-voxel contour steps");
  assert.match(liveSvoRadianceFeedbackWGSL, /transmittance\*=1\.0-clamp\(sample\.coverage,0\.0,1\.0\)/);
  assert.doesNotMatch(liveSvoRadianceFeedbackWGSL, /sample\.coverage>\.2/,
    "direct feedback visibility must remain continuous across fractional coverage");
});
