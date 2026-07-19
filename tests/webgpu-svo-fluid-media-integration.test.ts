import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { resolveSvoFluidRenderOwnership } from "../lib/svo-fluid-media-path";
import { rasterWaterStagePlan } from "../lib/webgpu-water-pipeline";
import { svoDrySceneShader } from "../lib/webgpu-svo-dry-scene";
import { svoFluidPrimaryModeWord } from "../lib/webgpu-svo-fluid-primary";

const waterSource = readFileSync(new URL("../lib/webgpu-water-pipeline.ts", import.meta.url), "utf8");
const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");

test("direct media requires explicit validation while the production default retains every raster-water stage", () => {
  assert.equal(svoFluidPrimaryModeWord(undefined), 0);
  assert.equal(svoFluidPrimaryModeWord("direct-structural-media"), 0);
  assert.equal(svoFluidPrimaryModeWord("direct-structural-media", true), 2);
  assert.deepEqual(rasterWaterStagePlan(resolveSvoFluidRenderOwnership()), {
    dryTarget: "internal-refraction", extractSurface: true, renderInterfaces: true, compositeOptics: true,
  });
  assert.deepEqual(rasterWaterStagePlan(resolveSvoFluidRenderOwnership("direct-structural-media", false)), {
    dryTarget: "internal-refraction", extractSurface: true, renderInterfaces: true, compositeOptics: true,
  });
});

test("validated direct media writes the presentation target and suppresses extraction, interfaces, and composite atomically", () => {
  const ownership = resolveSvoFluidRenderOwnership("direct-structural-media", true);
  assert.deepEqual(rasterWaterStagePlan(ownership), {
    dryTarget: "presentation-output", extractSurface: false, renderInterfaces: false, compositeOptics: false,
  });
  assert.throws(() => rasterWaterStagePlan({ ...ownership, legacyInterfaces: true }), /atomically suppress/);
  const branch = waterSource.indexOf('stagePlan.dryTarget === "presentation-output"');
  const geometry = waterSource.indexOf("const geometryDimensions", branch);
  const extraction = waterSource.indexOf('label:"Extract water isosurface"', branch);
  assert.ok(branch >= 0 && geometry > branch && extraction > geometry,
    "the direct branch must return before allocating or encoding any legacy-water work");
  assert.match(waterSource, /drySceneReplacement\?\.\(encoder, output, timestamps\?\.scene\)/,
    "direct media renders to the presentation output, not the raster compositor's internal dry target");
});

test("the production dry shader composes structural water and scene callbacks into the bounded media path", () => {
  assert.match(svoDrySceneShader, /fn svoFluidMediaQueryWater\(/);
  assert.match(svoDrySceneShader, /svoTraceStructuralFluidPrimary\(ray\.origin_m,ray\.direction,ray\.maximumDistance_m/);
  assert.match(svoDrySceneShader, /fn svoFluidMediaQueryScene\(/);
  assert.match(svoDrySceneShader, /let solid=traceDrySolidScene/);
  assert.match(svoDrySceneShader, /let glass=traceGlass/);
  assert.match(svoDrySceneShader, /fn svoTraceStructuralFluidMedia\(/);
  assert.match(svoDrySceneShader, /dry\.fluidDomainMode\.w==2u/);
  assert.match(svoDrySceneShader, /media\.throughput\*transmitted/);
  assert.match(svoDrySceneShader, /media\.inscatter/);
  assert.match(svoDrySceneShader, /SVO_FLUID_MEDIA_RESULT_INVALID/);
  assert.match(svoDrySceneShader, /SVO_FLUID_MEDIA_RESULT_EXHAUSTED/);
  assert.match(svoDrySceneShader, /SVO_GBUFFER_FIELD_FLUID_COARSE/);
});

test("the renderer passes the dry renderer's single ownership decision into the water pipeline", () => {
  assert.match(rendererSource, /const fluidRenderOwnership = useSvoDryScene \? this\.svoDryScenePipeline\?\.fluidRenderOwnership : undefined/);
  assert.match(rendererSource, /drySceneReplacement,\s*fluidRenderOwnership,/);
  assert.doesNotMatch(rendererSource, /directFluidMediaEndToEndValidated\s*:\s*true/,
    "no shipped scene may silently opt into the experimental direct-water path");
});
