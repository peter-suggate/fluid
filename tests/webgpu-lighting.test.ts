import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  beerLambert,
  dielectricFresnel,
  GLASS_OPTICS,
  integratedEnvironmentBrdf,
  packWaterSceneOptics,
  REC709_LUMINANCE,
  resolveWaterKeyLight,
  resolveWaterOptics,
  resolveDisplayGrade,
  waterSceneOpticsShaderLibrary,
  sceneLinearToDisplay,
  unifiedDisplayTransferShaderLibrary,
  unifiedLightingShaderLibrary,
  WATER_CAUSTIC_DEFAULT_STRENGTH,
  WATER_OPTICS
} from "../lib/webgpu-lighting";
import { compositeShader } from "../lib/webgpu-water-pipeline";
import { svoDrySceneShader } from "../lib/webgpu-svo-dry-scene";

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

test("rough environment BRDF does not turn a matte silhouette into a mirror", () => {
  const f0 = [0.04, 0.04, 0.04] as const;
  const matteGrazing = integratedEnvironmentBrdf(0, 0.78, f0);
  const matteFacing = integratedEnvironmentBrdf(1, 0.78, f0);
  const mirrorGrazing = integratedEnvironmentBrdf(0, 0, f0);

  assert.ok(matteGrazing.every((channel) => channel < 0.08),
    `rough stone grazing response must stay diffuse-scale, received ${matteGrazing}`);
  assert.ok(matteGrazing.every((channel, index) => channel >= matteFacing[index]),
    "the integrated response retains a restrained grazing lift");
  assert.ok(mirrorGrazing.every((channel) => channel > 0.95),
    "a genuinely smooth dielectric still reflects the environment at its silhouette");
  assert.match(unifiedLightingShaderLibrary, /fn unifiedEnvironmentBrdf\(/);
  assert.match(unifiedLightingShaderLibrary, /exp2\(-9\.28\*nDotV\)/,
    "the GPU closure must carry the same view/roughness integration as the CPU oracle");
});

test("scene-linear lighting reaches the presentation target through exactly one display transfer", () => {
  assert.deepEqual(sceneLinearToDisplay([0, -1, Number.NaN]), [0, 0, 0]);
  assert.deepEqual(sceneLinearToDisplay([1, 1, 1]), [
    0.5 ** (1 / 2.2),
    0.5 ** (1 / 2.2),
    0.5 ** (1 / 2.2),
  ]);
  assert.match(unifiedDisplayTransferShaderLibrary, /nonNegative \/ \(nonNegative \+ vec3f\(1\.0\)\)/);
  // One gamma per entry point, and the graded entry point applies its curve
  // once: two `pow` in the library means `unifiedDisplayTransfer` and
  // `unifiedDisplayGrade`, never a transfer applied twice inside either.
  assert.equal((unifiedDisplayTransferShaderLibrary.match(/pow\(/g) ?? []).length, 2);
  assert.equal((unifiedDisplayTransferShaderLibrary.match(/1\.0 \/ 2\.2/g) ?? []).length, 2);
  assert.doesNotMatch(svoDrySceneShader, /unifiedDisplayTransfer|1\.0\s*\/\s*2\.2/,
    "SVO dry lighting must remain scene-linear");
  assert.equal((compositeShader.match(/fn unifiedDisplayTransfer\(/g) ?? []).length, 1);
  assert.equal((compositeShader.match(/unifiedDisplayTransfer\(c\)/g) ?? []).length, 0,
    "the final compositor must choose the document's resolved grade");
  assert.match(compositeShader,
    /fn finish\([^}]+return vec4f\(unifiedDisplayGradeBalanced\(c,waterDisplayExposure\(\),waterDisplayToneCurve\(\),waterDisplayWhiteBalance\(\)\),1\);\}/);
});

/**
 * The grade is additive, not a replacement: `sceneLinearToDisplay` with no
 * grade must stay the exact expression every existing frame was written with,
 * or the change re-grades scenes that never asked for one.
 */
test("an unauthored grade is the identity against the historical transfer", () => {
  const neutral = resolveDisplayGrade(undefined);
  assert.deepEqual(neutral, { exposure: 1, toneCurve: "reinhard", whiteBalance: [1, 1, 1] });
  assert.deepEqual(resolveDisplayGrade({}), neutral);
  for (const radiance of [0, 0.18, 1, 4.5] as const) {
    assert.deepEqual(sceneLinearToDisplay([radiance, radiance, radiance], neutral),
      sceneLinearToDisplay([radiance, radiance, radiance]));
  }
  // The trade the filmic curve is bought for: unit radiance stops reading as a
  // light grey, so a white surface no longer needs an over-driven key.
  const [reinhard] = sceneLinearToDisplay([1, 1, 1]);
  const [aces] = sceneLinearToDisplay([1, 1, 1], resolveDisplayGrade({ toneCurve: "aces" }));
  assert.ok(Math.abs(reinhard - 0.7297) < 1e-3, `Reinhard puts unit radiance at ${reinhard}`);
  assert.ok(Math.abs(aces - 0.9061) < 1e-3, `ACES puts unit radiance at ${aces}`);
  // Exposure is applied before the curve, so it is not a display-space scale.
  assert.deepEqual(sceneLinearToDisplay([0.5, 0.5, 0.5], resolveDisplayGrade({ exposure: 2 })),
    sceneLinearToDisplay([1, 1, 1]));
});

test("live SVO and optical water/glass consume the canonical closure", () => {
  assert.match(svoDrySceneShader, /shadeUnifiedSurface\(directClosure,lighting\)/,
    "SVO dry materials must use the same resource-independent closure as raster bodies");
  // The tank's glass is a renderer constant and stays inlined; the water's
  // optics became a scene property and are now read from a uniform, so the
  // assertion moved from "the number 0.02037 appears" to "the composite asks
  // the scene". `WATER_OPTICS` is still the value it is asked for by default —
  // `resolveWaterOptics` is what proves that, not the shader text.
  assert.match(compositeShader, /unifiedDielectricFresnel\(cosine,0\.04\)/);
  assert.doesNotMatch(compositeShader, /unifiedDielectricFresnel\(cosine,0\.02037\)/,
    "water optics must not be inlined into WGSL at build time");
  assert.match(compositeShader, /unifiedDielectricFresnel\(cosine,waterFresnelF0\(\)\)/);
  assert.match(compositeShader, /unifiedAbsorbingTransmission\([^)]*,waterAbsorption\(\),waterScatter\(\),/);
  assert.match(compositeShader, /unifiedSpecularLobe\(n,-rd,waterKeyDirection\(\),180\.0\)/);
  assert.deepEqual(resolveWaterOptics(), {
    absorption_mInv: [...WATER_OPTICS.absorption],
    scatter: [...WATER_OPTICS.scatter],
    indexOfRefraction: WATER_OPTICS.indexOfRefraction,
    fresnelF0: WATER_OPTICS.fresnelF0,
    causticStrength: WATER_CAUSTIC_DEFAULT_STRENGTH,
  }, "a scene that authors nothing gets the frozen table verbatim");
});

test("analytic tank glass remains enabled for the hybrid octree smooth scene", () => {
  const glassFunction = compositeShader.slice(compositeShader.indexOf("fn compositeFrontGlass"), compositeShader.indexOf("fn finish"));
  assert.doesNotMatch(glassFunction, /u\.options\.w/, "voxel scene selection must not suppress the raster glass presentation");
  assert.match(glassFunction, /if\(environmentIndex\(\)==7\)\{return color;\}/, "the open garden remains vessel-free");
});

/**
 * The inspection overlay drew the tank as five camera-sorted alpha panes in a
 * pass of its own, because leaving glass in the parallel compacted draw made
 * the alpha/depth result depend on atomic instance order. That renderer is
 * gone, and with it the only consumer of a `containerBounds` render option, so
 * the frame's glass is once again the raster composite's alone.
 */
test("the raster composite is the only glass presentation left", () => {
  assert.doesNotMatch(rendererSource, /containerBounds: \{/,
    "no render pass may still take inspection container bounds");
  assert.match(compositeShader, /fn compositeFrontGlass/,
    "the authoritative water composite still owns the tank panes");
});


// ---------------------------------------------------------------------------
// White balance — H1 of docs/hero-fidelity-1000x-handoff.md
// ---------------------------------------------------------------------------

const waterOpticsLibrary = waterSceneOpticsShaderLibrary(0, 0, 1);

test("white balance changes chromaticity and never luminance", () => {
  // The orthogonality claim the solve depends on: exposure owns brightness,
  // balance owns hue. A knob that did both could be solved for neither.
  for (const authored of [[1.2, 1, 0.8], [0.6, 1, 1.5], [3, 2, 1]] as const) {
    const grade = resolveDisplayGrade({ whiteBalance: authored });
    const luminance =
      REC709_LUMINANCE[0] * grade.whiteBalance[0] +
      REC709_LUMINANCE[1] * grade.whiteBalance[1] +
      REC709_LUMINANCE[2] * grade.whiteBalance[2];
    assert.ok(Math.abs(luminance - 1) < 1e-9, `balance ${authored} has luminance ${luminance}`);
  }
  // A neutral grey keeps its display luminance under a warm balance, while its
  // channels separate.
  const warm = resolveDisplayGrade({ whiteBalance: [1.15, 1, 0.82], toneCurve: "aces" });
  const graded = sceneLinearToDisplay([0.4, 0.4, 0.4], warm);
  assert.ok(graded[0] > graded[1] && graded[1] > graded[2], `warm balance must order R>G>B, got ${graded}`);
});

test("an authored white balance survives the uniform round trip through two spare floats", () => {
  // Green is not transported; it is reconstructed from the luminance
  // constraint. If the resolver ever stops normalising, this is what catches it.
  const grade = resolveDisplayGrade({ whiteBalance: [1.15, 1, 0.82] });
  const packed = packWaterSceneOptics(resolveWaterOptics(undefined), resolveWaterKeyLight(undefined), grade);
  // Lane 6 begins at float 24: exposure, tone-curve code, then the two spare
  // slots the balance was folded into.
  const red = packed[26];
  const blue = packed[27];
  assert.ok(Math.abs(red - grade.whiteBalance[0]) < 1e-6);
  assert.ok(Math.abs(blue - grade.whiteBalance[2]) < 1e-6);
  const green = (1 - REC709_LUMINANCE[0] * red - REC709_LUMINANCE[2] * blue) / REC709_LUMINANCE[1];
  assert.ok(
    Math.abs(green - grade.whiteBalance[1]) < 1e-6,
    `reconstructed green ${green} does not match the resolved ${grade.whiteBalance[1]}`,
  );
});

test("the white-balance reconstruction constants agree between resolver and shader", () => {
  // Three numbers in three places is how a warm grade quietly becomes a warm
  // and darker one. The shader must spell the resolver's own weights.
  assert.ok(waterOpticsLibrary.includes(`${REC709_LUMINANCE[0]}`), "shader is missing the red luminance weight");
  assert.ok(waterOpticsLibrary.includes(`${REC709_LUMINANCE[1]}`), "shader is missing the green luminance weight");
  assert.ok(waterOpticsLibrary.includes(`${REC709_LUMINANCE[2]}`), "shader is missing the blue luminance weight");
});

test("an out-of-range channel gain clamps rather than casting the frame", () => {
  const extreme = resolveDisplayGrade({ whiteBalance: [100, 1, 0.0001] });
  for (const gain of extreme.whiteBalance) {
    assert.ok(Number.isFinite(gain) && gain > 0, `clamped gain ${gain} is not usable`);
  }
});
