import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { createBiasedSvoVisibilityRay, SVO_VISIBILITY_LIMITS } from "../lib/svo-visibility-rays";
import {
  directionalLightSceneExitDistance,
  SVO_DRY_SCENE_SHADOW_BIAS_CELLS,
  svoDrySceneShader,
} from "../lib/webgpu-svo-dry-scene";

const drySceneSource = readFileSync(new URL("../lib/webgpu-svo-dry-scene.ts", import.meta.url), "utf8");
const waterSource = readFileSync(new URL("../lib/webgpu-water-pipeline.ts", import.meta.url), "utf8");

test("directional visibility clips to the finite authored scene exit", () => {
  const bounds = { width_m: 4, height_m: 2, depth_m: 6 };
  assert.equal(directionalLightSceneExitDistance(
    { x: 0, y: 0.2, z: 0 },
    { x: 0, y: 3, z: 0 },
    bounds,
  ), 1.8);

  const direction = { x: 1, y: 1, z: 0 };
  const exitDistance = directionalLightSceneExitDistance({ x: 0, y: 1, z: 0 }, direction, bounds);
  assert.ok(Math.abs(exitDistance - Math.SQRT2) < 1e-12);

  const biased = createBiasedSvoVisibilityRay({
    surfacePosition_m: [0, 1, 0],
    geometricNormal: [0.6, 0.8, 0],
    directionToLight: [1, 1, 0],
    maximumLightDistance_m: exitDistance,
    cellSize_m: [0.02, 0.01, 0.04],
  }, { originBiasCells: SVO_DRY_SCENE_SHADOW_BIAS_CELLS });
  assert.ok(biased.originBias_m > 0);
  assert.ok(biased.tMax_m > 0 && biased.tMax_m < exitDistance,
    "the biased ray must still end at the original directional-light exit plane");

  assert.equal(directionalLightSceneExitDistance({ x: 3, y: 1, z: 0 }, { x: 0, y: 1, z: 0 }, bounds), 0);
  assert.equal(directionalLightSceneExitDistance({ x: 0, y: 1, z: 0 }, { x: 0, y: 0, z: 0 }, bounds), 0);
});

test("dry-scene direct PBR is geometry-normal aware and hard visibility modulates only direct light", () => {
  assert.match(svoDrySceneShader, /unifiedPbrMaterial\(surface\.baseColor,surface\.metallic,surface\.roughness,vec3f\(0\.0\),0\.0/);
  assert.match(svoDrySceneShader, /unifiedLightingInputWithGeometry\(hit\.normal,hit\.normal,-rd,sample\.towardLight,sample\.radiance\*visibility\/f32\(sampleCount\)\)/);
  assert.match(svoDrySceneShader, /direct\+=shadeUnifiedSurface\(directClosure,lighting\)/);
  assert.match(svoDrySceneShader, /let visibility=dryLightVisibility\(position,hit\.normal,sample\.towardLight,sample\.finiteDistance_m\)/);
  assert.doesNotMatch(svoDrySceneShader, /unifiedPbrMaterial\([^)]*visibility/,
    "hard visibility belongs on incident direct radiance, not material or ambient/emissive state");
});

test("bounded hard-shadow visibility covers opaque sources and transmissive panes", () => {
  assert.match(svoDrySceneShader, /fn svoBiasedVisibilityRay/);
  assert.match(svoDrySceneShader, /fn directionalLightSceneExitDistance/);
  assert.match(svoDrySceneShader, new RegExp(
    `SvoVisibilityBudget\\(${SVO_VISIBILITY_LIMITS.nodeVisits}u,${SVO_VISIBILITY_LIMITS.leafVisits}u,${SVO_VISIBILITY_LIMITS.workItems}u,4u\\)`,
  ));
  assert.match(svoDrySceneShader, /svoTraceVisibility\([^;]*true,0\.001,max\(ray\.originBias_m,1e-6\)\)/,
    "the hard ray must bound pane transmission and advance beyond each sheet");

  const adapterStart = svoDrySceneShader.indexOf("fn svoVisibilityNext(");
  const adapterEnd = svoDrySceneShader.indexOf("fn dryLightVisibility(", adapterStart);
  assert.ok(adapterStart >= 0 && adapterEnd > adapterStart);
  assert.ok(adapterStart < svoDrySceneShader.indexOf("fn svoTraceVisibility("),
    "Chrome requires the renderer adapter declaration before the shared trace body calls it");
  const adapter = svoDrySceneShader.slice(adapterStart, adapterEnd);
  assert.match(adapter, /bodyHit\(ray\.origin_m,ray\.direction,bodies\[bodyIndex\]\)/,
    "dynamic rigid bodies must cast hard shadows");
  assert.match(adapter, /primitiveHit\(record,ray\.origin_m,ray\.direction\)/,
    "small authored primitive catalogs must cast hard shadows");
  assert.match(adapter, /traceLeafPayloadVisibility/,
    "large or generated catalogs must retain SVO payload shadow traversal");
  assert.match(adapter, /traceTerrain\(ray\.origin_m,ray\.direction\)/,
    "analytic terrain must cast hard shadows");
  assert.match(adapter, /traceGlass\(ray\.origin_m,ray\.direction,tMin_m,bestT,false\)/,
    "finite panes must attenuate rather than become opaque shadow blockers");
  assert.match(adapter, /optics\.netTransmittance[^]*dryVisibilityTransmissionStep/);
  assert.doesNotMatch(adapter, /shadeUnifiedSurface|dryHardVisibility/,
    "visibility intersection must never recurse into shading");
});

test("invalid or exhausted shadow work fails closed and raster/timing fallback remains intact", () => {
  assert.match(svoDrySceneShader, /publicationState\[0\]==0u[^]*SVO_VIS_STEP_INVALID/);
  assert.match(svoDrySceneShader, /SVO_STATUS_WORK_EXHAUSTED\|\|leaf\.status==SVO_STATUS_STACK_OVERFLOW\|\|leaf\.status==SVO_STATUS_SOURCE_OVERFLOW[^]*SVO_VIS_STEP_EXHAUSTED/);
  assert.match(svoDrySceneShader, /fn svoVisibilityFail\([^]*vec3f\(0\.0\)/,
    "shared invalid/exhausted/occluded results must carry zero direct visibility");
  assert.match(drySceneSource, /encode\(encoder: GPUCommandEncoder, target: GPUTexture \| GPUTextureView, timestampWrites\?: TimestampRange, temporalFrame\?: SparseVoxelTemporalFrameState\): boolean/);
  assert.match(drySceneSource, /timestampWrites \? \{ timestampWrites \} : \{\}/,
    "SVO scene timing must remain attached to the replacement pass");
  assert.match(waterSource, /if \(!sparseSceneEncoded\) \{[^]*label:"Dry scene"/,
    "the unchanged raster pass remains the fallback when SVO declines a frame");
});
