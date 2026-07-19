import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  beerLambert,
  dielectricFresnel,
  GLASS_OPTICS,
  sceneLinearToDisplay,
  unifiedDisplayTransferShaderLibrary,
  unifiedLightingShaderLibrary,
  WATER_OPTICS
} from "../lib/webgpu-lighting";
import { compositeShader, sceneShader } from "../lib/webgpu-water-pipeline";
import { svoDrySceneShader } from "../lib/webgpu-svo-dry-scene";
import { voxelDebugRenderShader } from "../lib/webgpu-voxel-debug";

const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");

test("shared lighting contract is resource-layout independent and scene-linear", () => {
  assert.match(unifiedLightingShaderLibrary, /struct UnifiedLightingMaterial/);
  assert.match(unifiedLightingShaderLibrary, /struct UnifiedLightingInput/);
  assert.match(unifiedLightingShaderLibrary, /fn shadeUnifiedSurface/);
  assert.match(unifiedLightingShaderLibrary, /fn unifiedDielectricFresnel/);
  assert.match(unifiedLightingShaderLibrary, /fn unifiedAbsorbingTransmission/);
  assert.doesNotMatch(unifiedLightingShaderLibrary, /@group|@binding/, "the closure must compose with any renderer bind-group ABI");
  assert.doesNotMatch(unifiedLightingShaderLibrary, /pow\([^\n]*1\.0\s*\/\s*2\.2/, "display transfer belongs only in the final output pass");
});

test("CPU optical mirrors retain physical endpoint and attenuation invariants", () => {
  assert.equal(dielectricFresnel(1, WATER_OPTICS.fresnelF0), WATER_OPTICS.fresnelF0);
  assert.equal(dielectricFresnel(0, WATER_OPTICS.fresnelF0), 1);
  assert.equal(dielectricFresnel(2, GLASS_OPTICS.fresnelF0), GLASS_OPTICS.fresnelF0, "cosine is clamped");
  assert.equal(dielectricFresnel(-1, GLASS_OPTICS.fresnelF0), 1, "negative cosine is clamped");

  assert.deepEqual(beerLambert(WATER_OPTICS.absorption, 0), [1, 1, 1]);
  const transmission = beerLambert(WATER_OPTICS.absorption, 2);
  assert.ok(transmission[0] < transmission[1] && transmission[1] < transmission[2], "clean water attenuates red first");
  assert.deepEqual(beerLambert(WATER_OPTICS.absorption, -2), [1, 1, 1], "negative optical distance cannot amplify light");
});

test("scene-linear lighting reaches the presentation target through exactly one display transfer", () => {
  assert.deepEqual(sceneLinearToDisplay([0, -1, Number.NaN]), [0, 0, 0]);
  assert.deepEqual(sceneLinearToDisplay([1, 1, 1]), [
    0.5 ** (1 / 2.2),
    0.5 ** (1 / 2.2),
    0.5 ** (1 / 2.2),
  ]);
  assert.match(unifiedDisplayTransferShaderLibrary, /nonNegative \/ \(nonNegative \+ vec3f\(1\.0\)\)/);
  assert.equal((unifiedDisplayTransferShaderLibrary.match(/pow\(/g) ?? []).length, 1);
  assert.doesNotMatch(sceneShader, /unifiedDisplayTransfer|1\.0\s*\/\s*2\.2/,
    "raster dry lighting must remain scene-linear");
  assert.doesNotMatch(svoDrySceneShader, /unifiedDisplayTransfer|1\.0\s*\/\s*2\.2/,
    "SVO dry lighting must remain scene-linear");
  assert.equal((compositeShader.match(/fn unifiedDisplayTransfer\(/g) ?? []).length, 1);
  assert.equal((compositeShader.match(/unifiedDisplayTransfer\(c\)/g) ?? []).length, 1);
  assert.match(compositeShader, /fn finish\([^}]+return vec4f\(unifiedDisplayTransfer\(c\),1\);\}/);
});

test("raster bodies and optical water/glass consume the canonical closure", () => {
  assert.match(sceneShader, /shadeUnifiedSurface\(material,lighting\)/);
  assert.match(svoDrySceneShader, /shadeUnifiedSurface\(directClosure,lighting\)/,
    "SVO dry materials must use the same resource-independent closure as raster bodies");
  assert.match(voxelDebugRenderShader, /shadeUnifiedSurface\(closure, lighting\)/, "raw voxel materials must use the same closure");
  assert.match(compositeShader, /unifiedDielectricFresnel\(cosine,0\.04\)/);
  assert.match(compositeShader, /unifiedDielectricFresnel\(cosine,0\.02037\)/);
  assert.match(compositeShader, /unifiedAbsorbingTransmission/);
  assert.match(compositeShader, /unifiedSpecularLobe\(n,-rd,light,180\.0\)/);
});

test("analytic tank glass remains enabled for the hybrid octree smooth scene", () => {
  assert.doesNotMatch(sceneShader, /u\.options\.w<0\.5&&environmentIndex\(\)!=7/);
  const glassFunction = compositeShader.slice(compositeShader.indexOf("fn compositeFrontGlass"), compositeShader.indexOf("fn finish"));
  assert.doesNotMatch(glassFunction, /u\.options\.w/, "voxel scene selection must not suppress the raster glass presentation");
  assert.match(glassFunction, /if\(environmentIndex\(\)==7\)\{return color;\}/, "the open garden remains vessel-free");
});

test("raw voxel glass uses a separate stable pane pass", () => {
  assert.match(voxelDebugRenderShader, /fn glassPaneVertex/);
  assert.match(voxelDebugRenderShader, /fn glassPaneFragment/);
  assert.match(voxelDebugRenderShader, /input\.materialId == 1u\) \{ discard/);
  assert.match(rendererSource, /containerBounds: \{/);
  assert.match(rendererSource, /containerClosedTop: scene\.container\.top === "closed"/);
});
