import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { createBiasedSvoVisibilityRay, SVO_VISIBILITY_LIMITS } from "../lib/svo-visibility-rays";
import {
  directionalLightSceneExitDistance,
  SVO_DRY_SCENE_SHADOW_BIAS_CELLS,
  svoDrySceneShader,
} from "../lib/webgpu-svo-dry-scene";
import {
  SVO_CAMERA_CHANGING_FRAME,
  SVO_SHADOW_HISTORY_WARMUP_FRAMES,
  svoDrySceneTemporalFrame,
  svoShadowTemporalFrame,
} from "../lib/webgpu-renderer";

const drySceneSource = readFileSync(new URL("../lib/webgpu-svo-dry-scene.ts", import.meta.url), "utf8");
const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
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
  assert.match(svoDrySceneShader, /let visibility=dryLightVisibility\(position,hit\.normal,hit\.ownerId,sample\.towardLight,sample\.finiteDistance_m\)/);
  assert.match(svoDrySceneShader, /sample\.valid==0u\|\|dot\(hit\.normal,sample\.towardLight\)<=0\.0[^]*continue[^]*dryLightVisibility/,
    "back-facing opaque samples must skip the full primary/shadow visibility traversal");
  assert.doesNotMatch(svoDrySceneShader, /unifiedPbrMaterial\([^)]*visibility/,
    "hard visibility belongs on incident direct radiance, not material or ambient/emissive state");
  assert.match(svoDrySceneShader, /light\.positionRange\.w>0\.0&&distanceSquared>=light\.positionRange\.w\*light\.positionRange\.w[^]*return dryInvalidLightSample\(\)/,
    "finite-range samples with exactly zero contribution must stop before square roots and shadow traversal");
  assert.match(svoDrySceneShader, /let radiance=baseRadiance\*\(rangeFade\*shapeScale\);if\(max\(max\(radiance\.x,radiance\.y\),radiance\.z\)<=0\.0\)\{return dryInvalidLightSample\(\);\}/,
    "zero-radiance area samples, including back-facing emitters, must never launch visibility rays");
});

test("checkerboard hard visibility is enabled only with temporal reconstruction", () => {
  assert.match(rendererSource, /new URLSearchParams\(location\.search\)\.get\("svoTemporal"\) !== "0"/);
  assert.equal(SVO_SHADOW_HISTORY_WARMUP_FRAMES, 2);
  assert.equal(svoShadowTemporalFrame(true, 0, 12), -1);
  assert.equal(svoShadowTemporalFrame(true, 1, 12), -1);
  assert.equal(svoShadowTemporalFrame(true, 2, 12), 12);
  assert.equal(svoShadowTemporalFrame(false, 20, 12), -1);
  assert.equal(svoDrySceneTemporalFrame(12, 0), SVO_CAMERA_CHANGING_FRAME);
  assert.equal(svoDrySceneTemporalFrame(-1, 2), -1,
    "a settled camera remains distinguishable when checkerboard shadows are disabled");
  assert.equal(svoDrySceneTemporalFrame(12, 2), 12);
  assert.match(rendererSource, /shadowStabilityKey !== this\.svoShadowStabilityKey[^]*this\.svoDryScenePipeline\?\.invalidateTemporalHistory\(\)/,
    "camera, body, scene, or diagnostic changes must force a full-rate shadow frame and discard stale shadows");
  assert.match(rendererSource, /shadowTemporalFrame = svoShadowTemporalFrame\(checkerboardShadowsEligible, this\.svoShadowStableFrames, this\.presentationFrameIndex\)/);
  assert.match(rendererSource, /cameraStabilityKey !== this\.svoCameraStabilityKey[^]*this\.svoCameraStableFrames = 0[^]*drySceneTemporalFrame = svoDrySceneTemporalFrame\(shadowTemporalFrame, this\.svoCameraStableFrames\)/,
    "camera movement must be detected from the view basis, independently of pointer state");
  assert.match(svoDrySceneShader, /temporalShadowSampling=uniforms\.viewport\.w>=0\.0&&\(dry\.materialPublication\.w&2u\)!=0u/);
  assert.match(svoDrySceneShader, /shadowParity==0u/);
  assert.match(svoDrySceneShader, /dryShadowTracingEnabled==0u\)\{return vec3f\(1\.0\);\}/);
  assert.match(svoDrySceneShader, /DRY_GBUFFER_SHADOW_DEFERRED<<20u/,
    "deferred visibility must be explicit in the G-buffer rather than inferred from color");
});

test("bounded hard-shadow visibility covers opaque sources and transmissive panes", () => {
  assert.match(svoDrySceneShader, /fn svoBiasedVisibilityRay/);
  assert.match(svoDrySceneShader, /fn dryBiasedVisibilityRayUnit\([^]*projectedCellWidth=dot\(abs\(geometricNormal\),cellSize_m\)[^]*maximumLightDistance_m-dot\(offset,directionToLight\)/,
    "the renderer-local unit-vector path must preserve the shared projected-cell bias and original endpoint");
  assert.match(svoDrySceneShader, /let ray=dryBiasedVisibilityRayUnit\(position,geometricNormal,towardLight,maximumDistance/,
    "hard shadows must avoid renormalizing already-unit light directions and surface normals");
  // Deliberate cone-banding fix: the cone origin escapes the receiver surface
  // along the geometric normal and its march end shortens by the escape's
  // projection so it still stops at the emitter surface; the rigid blocker
  // keeps the exact bias-adjusted ray.
  assert.match(svoDrySceneShader, /dryConeVisibility\(ray\.origin_m\+geometricNormal\*coneEscape_m,towardLight,\.065,max\(0\.0,ray\.tMax_m-coneEscape_m\*dot\(geometricNormal,towardLight\)\)\)[^]*rigidBlocker\.t<ray\.tMax_m/,
    "cone and rigid visibility must stop at the same bias-adjusted emitter endpoint as exact traversal");
  assert.match(svoDrySceneShader, /visibilityDistance=select\(distance,max\(0\.0,distance-light\.shape\.x\),light\.identity\.x==SVO_LIGHT_POINT\)/,
    "point attenuation uses center distance while visibility stops at the conservative emitter surface");
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
  assert.match(adapter, /let body=bodies\[bodyIndex\][^]*bodyHit\(ray\.origin_m,ray\.direction,body\)/,
    "dynamic rigid bodies must cast hard shadows");
  assert.match(adapter, /bodyIndex==dryVisibilityIgnoredOwner[^]*continue/,
    "a convex dynamic receiver must not exact-test itself along its outward front-facing shadow ray");
  assert.match(adapter, /!bodyBoundingSphereVisible\(ray\.origin_m,ray\.direction,body,tMin_m,bestT\)[^]*continue/,
    "a conservative world-space sphere must reject distant bodies before quaternion transforms");
  assert.match(adapter, /shape>=2&&!bodyCandidateVisible\(ray\.origin_m,ray\.direction,body,tMin_m,bestT\)/,
    "capsules and cylinders must retain conservative rejection while box/sphere exact tests avoid a duplicate local transform");
  assert.match(adapter, /tracePrimitiveCandidates\(ray\.origin_m,ray\.direction,tMin_m,bestT,[^,]+,true\)/,
    "small authored catalogs must spatially reject candidates before exact shadow intersections");
  const candidateTraversalStart = svoDrySceneShader.indexOf("fn tracePrimitiveCandidates(");
  const candidateTraversalEnd = svoDrySceneShader.indexOf("fn traceLeafPayload(", candidateTraversalStart);
  const candidateTraversal = svoDrySceneShader.slice(candidateTraversalStart, candidateTraversalEnd);
  assert.match(candidateTraversal, /opaqueAnyHit&&candidate\.t<DRY_MISS[^]*return DryCandidateTrace\(candidate,leftOrPrimitive,DRY_CANDIDATE_COMPLETE,workItems\)/,
    "opaque shadow candidate traversal must stop at its first exact blocker rather than resolve nearest identity");
  assert.match(adapter, /candidate\.hit\.t<bestT\)\{return dryVisibilityStep\(SVO_VIS_STEP_HIT/,
    "any opaque analytic blocker must terminate visibility without nearest-identity work");
  assert.match(adapter, /onlyIgnoredReceiver=dry\.metadata\.x==1u&&svoPrimitiveOwnerId\(primitives\[0\]\)==dryVisibilityIgnoredOwner/,
    "the common one-primitive convex receiver must bypass its entire static candidate query");
  assert.match(svoDrySceneShader, /owner==dry\.metadata\.z\|\|owner==dryVisibilityIgnoredOwner/,
    "multi-primitive candidate traversal must skip the exact convex receiver while retaining every other blocker");
  assert.match(adapter, /payload\.status==SVO_VIS_STEP_HIT\)\{return dryVisibilityStep\(SVO_VIS_STEP_HIT/,
    "any opaque SVO payload blocker must terminate before terrain and glass work");
  assert.doesNotMatch(adapter, /for\(var primitiveIndex=0u;primitiveIndex<dry\.metadata\.x/,
    "small authored catalogs must never return to a full exact-primitive shadow loop");
  assert.match(adapter, /traceLeafPayloadVisibility/,
    "large or generated catalogs must retain SVO payload shadow traversal");
  assert.match(adapter, /traceTerrain\(ray\.origin_m,ray\.direction\)/,
    "analytic terrain must cast hard shadows");
  assert.match(adapter, /traceGlass\(ray\.origin_m,ray\.direction,tMin_m,bestT,false\)/,
    "finite panes must attenuate rather than become opaque shadow blockers");
  assert.match(svoDrySceneShader, /fn dryGlassBoundingSphereVisible\([^]*record\.extentIorEpsilon\.xy[^]*record\.centerThickness\.w[^]*radius\*radius/,
    "pane tracing must conservatively reject distant finite sheets in world space before local transforms");
  assert.match(svoDrySceneShader, /compositeOwned\|\|thickReplaced\|\|!dryGlassBoundingSphereVisible\(record,ro,rd,tMin_m,bestT\)[^]*continue[^]*svoThinGlassIntersect/,
    "both primary and shadow pane queries must apply the conservative gate before exact intersection");
  assert.match(adapter, /optics\.netTransmittance[^]*dryVisibilityTransmissionStep/);
  assert.doesNotMatch(adapter, /shadeUnifiedSurface|dryHardVisibility/,
    "visibility intersection must never recurse into shading");
});

test("cone visibility is generation-checked and falls back to exact SVO visibility", () => {
  assert.match(svoDrySceneShader, /fn dryNodeMipReady\(\)->bool\{return dry\.nodeMip\.w!=0u&&dry\.nodeMip\.x!=0u&&dry\.nodeMip\.x==publicationState\[2\]&&dry\.nodeMip\.y>0u&&dry\.nodeMip\.z>0u;\}/,
    "the sampled cache must require a complete matching directory publication");
  const lightStart = svoDrySceneShader.indexOf("fn dryLightVisibility(");
  const lightEnd = svoDrySceneShader.indexOf("fn dryContactVisibilityRadius", lightStart);
  const lightVisibility = svoDrySceneShader.slice(lightStart, lightEnd);
  assert.match(lightVisibility, /dry\.materialPublication\.w&4u[^]*dryConeVisibility\([^]*if\(cone\.valid!=0u\)\{[^]*return vec3f\(cone\.transmittance\);\}/);
  assert.ok(lightVisibility.indexOf("svoTraceVisibility", lightVisibility.indexOf("dryConeVisibility")) > 0,
    "a missing or stale cone result must continue through exact bounded visibility");
  const contactStart = svoDrySceneShader.indexOf("fn dryContactVisibility(");
  const contactEnd = svoDrySceneShader.indexOf("fn dryEnvironment(", contactStart);
  assert.match(svoDrySceneShader.slice(contactStart, contactEnd), /dry\.materialPublication\.w&4u[^]*dryNodeMipReady\(\)[^]*for\(var sampleIndex=0u;sampleIndex<4u/,
    "cone AO uses four bounded hemisphere samples only when the cache is ready");
  assert.match(svoDrySceneShader.slice(contactStart, contactEnd), /cone\.valid==0u\)\{coneValid=false;break;\}[^]*if\(coneValid\)\{return[^]*svoTraceVisibility/,
    "an unavailable cone sample must fall through to exact bounded AO instead of leaking ambient light");
  assert.match(svoDrySceneShader, /diffuseEnvironment=[^;]*\*contactVisibility\/UNIFIED_PI[^]*specularEnvironment=dryEnvironment/,
    "AO must modulate diffuse environment only, leaving direct light, emission, and specular environment intact");
});

test("invalid or exhausted shadow work fails closed and raster/timing fallback remains intact", () => {
  assert.match(svoDrySceneShader, /if\(\(dry\.materialPublication\.w&2u\)==0u\)\{return vec3f\(1\.0\);\}/,
    "the shadow-disabled production path must return before traversal");
  assert.match(rendererSource, /checkerboardShadowsEligible = this\.svoTemporalAccumulationEnabled && svoLightingOptions\.shadowsEnabled/,
    "the user-facing shadow option must drive the temporally reconstructed visibility path");
  assert.match(svoDrySceneShader, /publicationState\[0\]==0u[^]*SVO_VIS_STEP_INVALID/);
  assert.match(svoDrySceneShader, /SVO_STATUS_WORK_EXHAUSTED\|\|leaf\.status==SVO_STATUS_STACK_OVERFLOW\|\|leaf\.status==SVO_STATUS_SOURCE_OVERFLOW[^]*SVO_VIS_STEP_EXHAUSTED/);
  assert.match(svoDrySceneShader, /fn svoVisibilityFail\([^]*vec3f\(0\.0\)/,
    "shared invalid/exhausted/occluded results must carry zero direct visibility");
  assert.match(drySceneSource, /encode\(encoder: GPUCommandEncoder, target: GPUTexture \| GPUTextureView, timestampWrites\?: TimestampRange, temporalFrame\?: SparseVoxelTemporalFrameState, temporalTimestampWrites\?: TimestampRange, reuseKey\?: string\): DrySceneReplacementResult \| false/);
  assert.match(drySceneSource, /timestampWrites \? \{ timestampWrites \} : \{\}/,
    "SVO scene timing must remain attached to the replacement pass");
  assert.match(waterSource, /if \(!sparseSceneResult\) \{[^]*label:"Dry scene"/,
    "the unchanged raster pass remains the fallback when SVO declines a frame");
});
