import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { SVO_MATERIAL_RECORD_STRIDE_BYTES } from "../lib/svo-material-abi";
import {
  createSvoDryConeMarcherWGSL,
  createSvoDrySceneFragmentWGSL,
  SparseVoxelDrySceneRenderer,
  SVO_DRY_CONE_PREPASS_CONTRACT,
  SVO_DRY_SILHOUETTE_REFINEMENT_CONTRACT,
  SVO_DRY_VOXEL_LIGHT_CACHE_CONTRACT,
  svoConePrepassSize,
  svoDrySceneShader,
} from "../lib/webgpu-svo-dry-scene";
import type { SparseVoxelRenderSource } from "../lib/webgpu-voxel-debug";
import { svoDrySceneFixture } from "./svo-dry-scene-test-fixture";

const rendererSource = readFileSync(new URL("../lib/webgpu-svo-dry-scene.ts", import.meta.url), "utf8");

test("the Phase-1 voxel visibility cache is runtime-demanded, bounded, and fail-soft", () => {
  const cached = createSvoDrySceneFragmentWGSL(0.5, "hybrid", "off", "split", 0,
    false, false, false, false, { voxelLightCache: true });
  const uncached = createSvoDrySceneFragmentWGSL(0.5, "hybrid", "off", "split", 0,
    false, false, false, false, { voxelLightCache: false });

  assert.equal(SVO_DRY_VOXEL_LIGHT_CACHE_CONTRACT.populationBudget, 16_384);
  assert.equal(SVO_DRY_VOXEL_LIGHT_CACHE_CONTRACT.format, "rg32uint");
  assert.match(cached, /@compute @workgroup_size\(8,8\) fn dryVoxelLightDemandMain/);
  assert.match(cached, /atomicOr\(&dryVoxelLightRequests\[word\],bit\)/,
    "the frame-local request bitset must deduplicate pixels naming one voxel");
  assert.match(cached, /@compute @workgroup_size\(64\) fn dryVoxelLightPopulateMain/);
  assert.match(cached, /dryConeVisibility\(ray\.origin_m\+normal\*escape/,
    "population reuses the live cone marcher at runtime; no baked visibility input exists");
  assert.match(cached, /dryVoxelLightConsumerEligible=select\(0u,1u,hit\.motionKind==DRY_GBUFFER_MOTION_STATIC\)/);
  assert.match(cached, /if\(cachedVoxelVisibility\.y>0\.0\)[^]*nearestBodyIgnoring/,
    "cached static visibility retains the current-frame rigid blocker overlay");
  assert.match(cached, /dryCurrentLightSlot=0xffffffffu/,
    "missing or rejected exclusive-tier entries must continue into the live fallback chain");
  assert.doesNotMatch(uncached, /dryVoxelLightDemandMain|dryVoxelLightPopulateMain|dryVoxelLightCacheRead/);

  assert.match(rendererSource, /if \(usePrepass && !this\.voxelLightExclusive\)/,
    "the settled eligible tier cuts over by withholding the old screen-space cone pass");
  assert.match(rendererSource, /setVoxelLightCacheEnabled\(enabled: boolean\)/,
    "production retains an immediate runtime fallback switch");
  assert.match(rendererSource, /this\.invalidateVoxelLightCache\(\)/,
    "authored dynamic changes invalidate runtime entries instead of relying on pre-baked data");
});

test("scale 1 preserves the production shader byte-for-byte (fingerprint contract)", () => {
  assert.equal(createSvoDrySceneFragmentWGSL(1), svoDrySceneShader,
    "the factory default must return the exact historical string so the bit-exact frame fingerprint reproduces");
  assert.doesNotMatch(svoDrySceneShader, /dryPrepass|@group\(1\)/,
    "the inline path must carry no prepass declarations, bindings, or code");
  assert.doesNotMatch(svoDrySceneShader, /anyBodyBlockerIgnoring/,
    "the reduced-only blocker specialization must not perturb the scale-1 shader");
});

test("reduced scales use declared receiver and full-rate edge tiers without hiding page failures", () => {
  for (const scale of [0.5, 0.25, 0.125] as const) {
    const reduced = createSvoDrySceneFragmentWGSL(scale);
    assert.match(reduced, /@group\(1\) @binding\(0\) var dryPrepassVisibilityKeyTexture:texture_2d<u32>/);
    assert.match(reduced, /@group\(1\) @binding\(1\) var dryPrepassGeometryTexture:texture_2d<f32>/);
    assert.match(reduced, /@group\(1\) @binding\(2\) var dryPrepassIdentityTexture:texture_2d<u32>/);
    assert.match(reduced, /@group\(1\) @binding\(3\) var dryPrepassRadianceTexture:texture_2d<f32>/);
    assert.match(reduced, /@fragment fn dryPrepassGeometryMain/,
      "the reduced primary trace must be isolated from cone-march register pressure");
    assert.match(reduced, /@fragment fn dryPrepassVisibilityMain/,
      "the cone marcher must consume the freshly traced reduced geometry in its own phase");
    assert.match(reduced, /dryPrepassVisibilityMain[^]*textureLoad\(dryPrepassGeometryTexture,coordinate,0\)[^]*textureLoad\(dryPrepassIdentityTexture,coordinate,0\)[^]*return dryPrepassTraceVisibility/,
      "the cone phase reconstructs the dynamic hit instead of tracing primary visibility again");
    assert.match(reduced, /@fragment fn dryPrepassShadeMain/,
      "opaque shading must have a separate reduced-rate entry point so it cannot inflate cone-pass register pressure");
    assert.ok(reduced.includes(createSvoDryConeMarcherWGSL({ branchlessMorton: true, rangedDirectorySearch: true, fluidCoverage: true, directPageTable: true })),
      "the reduced variant must embed the identical optimized marcher block");
    assert.match(reduced, /if\(weightSum<0\.05\)\{[^]*dryPrepassRecoverExactReceiver[^]*dryPrepassExactEdgeState=1u;return;/,
      "sub-prepass-pixel surfaces must enter the declared live full-rate edge tier");
    assert.match(reduced, /let cone=dryConeVisibility\(origin,rotated[^]*if\(cone\.valid==0u\)\{return DRY_PREPASS_INVALID_PACKED;\}[^]*visibility\+=cone\.transmittance/,
      "a dirty AO page must invalidate the reduced texel instead of fabricating visibility");
    assert.match(reduced, /let cone=dryConeVisibility\(ray\.origin_m\+geometricNormal\*coneEscape_m[^]*if\(cone\.valid==0u\)\{return DRY_PREPASS_INVALID_PACKED;\}[^]*visibility\+=mix/,
      "a dirty shadow page must invalidate the reduced texel instead of being packed as fully visible");
    // Deliberate cone-banding fix: shadow-cone origins escape the receiver's
    // trilinear support along the geometric normal, and finite emitters clear
    // the march end by one cone-support width before marching.
    assert.match(reduced, /let cone=dryConeVisibility\(ray\.origin_m\+geometricNormal\*coneEscape_m,towardLight,dry\.tuningRays1\.y,coneMax_m,geometricNormal,finiteDistance_m>0\.0\)/,
      "valid cone-mode shadow queries retain the declared live cone algorithm");
    assert.match(reduced, /dryCurrentLightSlot<8u/,
      "every user-shadable light slot must reuse the reduced-rate visibility cache");
    assert.match(reduced, /if\(index<4u\)\{return dryPrepassData0\[index\];\}[^]*if\(index<8u\)\{return dryPrepassData1\[index-4u\];\}[^]*dryPrepassData2\[min\(index-8u,3u\)\]/,
      "AO and all eight light slots must decode from their documented packed planes");
    assert.match(reduced, /if\(lightIndex<3u\)\{visibility0\[1u\+lightIndex\]=packedVisibility;\}[^]*else if\(lightIndex<7u\)\{visibility1\[lightIndex-3u\]=packedVisibility;\}[^]*else\{visibility2\.x=packedVisibility;\}/,
      "the prepass writer and full-resolution reader must agree on every packed light channel");
    assert.match(reduced, /fn dryPrepassQuantize7\(value:f32\)->u32\{return u32\(round\(clamp\(value,0\.0,1\.0\)\*127\.0\)\);\}/);
    assert.match(reduced, /fn dryPrepassPack\([^]*clamp\(data0\.x,0\.0,1\.0\)\*255\.0/);
    assert.match(reduced, /fn dryPrepassUnpack0[^]*\/255\.0[^]*\/127\.0[^]*fn dryPrepassUnpack1[^]*fn dryPrepassUnpack2/,
      "AO must retain eight bits while all eight lights round-trip through seven-bit lanes");
    assert.match(reduced, /let packed=textureLoad\(dryPrepassVisibilityKeyTexture,texel,0\)[^]*accumulated0\+=dryPrepassUnpack0\(packed\)[^]*accumulated1\+=dryPrepassUnpack1\(packed\)[^]*accumulated2\+=dryPrepassUnpack2\(packed\)/,
      "one coherent integer fetch must provide every visibility lane and guide the 2x2 reconstruction");
    assert.match(reduced, /let prepassRigidBlocked=anyBodyBlockerIgnoring\(ray\.origin_m,towardLight,ownerId,ray\.tMax_m\);let raw=select\(dryPrepassChannel\(1u\+dryCurrentLightSlot\),0\.0,prepassRigidBlocked\)/,
      "rigid-body blocker terms stay inline at full resolution on the upsampled shadow path");
    assert.match(reduced, /prepassUnblocked\+=select\(1\.0,0\.0,prepassRigidBlocked\)/,
      "rigid AO blocker sampling stays inline at full resolution on the upsampled AO path");
    assert.match(reduced, /fn anyBodyBlockerIgnoring\([^]*if\(bodyHit\(ro,rd,body\)\.t<tMax\)\{return true;\}/,
      "blocker-only paths must early out without carrying the full nearest-hit payload");
    assert.match(reduced, /fn dryPrepassReceiverCompatible[^]*materialMatches[^]*metadata==dryPrepassHitMetadata\(hit\)[^]*hit\.motionKind==DRY_GBUFFER_MOTION_STATIC\|\|ownerMatches/,
      "static seams may share fully classified receivers while moving surfaces retain exact ownership");
    assert.match(reduced, /let identityMatches=dryPrepassReceiverCompatible\(textureLoad\(dryPrepassIdentityTexture,texel,0\)\.x,u32\(round\(geometry\.w\)\),hit\)/,
      "radiance reconstruction must apply the declared receiver compatibility contract");
    assert.match(reduced, /if\(!identityMatches\)\{if\(bilinear>1e-6\)\{linearSafe=0u;\}continue;\}[^]*accumulated2\+=dryPrepassUnpack2\(packed\)\*weight/,
      "visibility and radiance reconstruction must share one compatible receiver at discontinuities");
    assert.match(reduced, /if\(all\(packed\.xy==DRY_PREPASS_INVALID_PACKED\)\)\{linearSafe=0u;continue;\}/,
      "an invalid neighbouring receiver must be excluded instead of poisoning unrelated full-rate pixels");
    assert.match(reduced, /if\(!identityMatches\)\{if\(bilinear>1e-6\)\{linearSafe=0u;\}continue;\}[^]*if\(bilinear>1e-6&&\(depthWeight<0\.25\|\|normalWeight<0\.25\)\)\{linearSafe=0u;\}/,
      "hardware filtering must be disabled before it can cross an identity, depth, or normal edge");
    assert.match(reduced, /textureSampleLevel\(dryPrepassRadianceTexture,nodeMipSampler,pixel\/max\(uniforms\.viewport\.xy,vec2f\(1\.0\)\),0\.0\)/,
      "the gated-linear mode must use the resident linear sampler without another pass");
    assert.match(reduced, /accumulatedRadiance\+=textureLoad\(dryPrepassRadianceTexture,texel,0\)\*weight;radianceWeightSum\+=weight/,
      "joint-bilateral mode must reuse the visibility guide weights for edge-aware radiance reconstruction");
    assert.match(reduced, /materialPublication\.w&16u\)!=0u[^]*tuningCounts2\.w==3u[^]*tuningCounts2\.w==4u[^]*accumulatedGi\+=textureLoad\(dryPrepassRadianceTexture,texel,0\)\*weight;giWeightSum\+=weight/,
      "relight modes must reconstruct current-frame environmental GI only from exact-identity neighbours");
    assert.match(reduced, /if\(giWeightSum>=0\.05\)\{dryPrepassGi=accumulatedGi\/giWeightSum;dryPrepassGiState=1u;\}[^]*else if[^]*dryDerivedPageFailure\|=8u/,
      "insufficient geometry-guided GI support must fail visibly without a hidden full-resolution retry");
    assert.match(reduced, /if\(dryPrepassGiState==1u\)\{[^]*return DryGlobalIllumination/,
      "a valid reduced GI surface summary must return before any full-resolution 3D cone taps");
    assert.match(reduced, /let weight=bilinear\*select\(guidedWeight,1\.0,dry\.tuningCounts2\.w==3u\)/,
      "wide relight must aggressively reconstruct shadow factors with unmodified bilinear weights");
    assert.match(reduced, /tuningCounts2\.w!=3u&&dry\.tuningCounts2\.w!=4u&&bestRadianceWeight>0\.0/,
      "both relight modes must bypass every reduced-radiance shortcut");
    assert.match(reduced, /fn dryPrepassPackIdentity\(hit:DryHit\)->u32\{return \(hit\.materialId&0xffffu\)\|\(\(hit\.ownerId&0xffffu\)<<16u\);\}/);
    assert.match(reduced, /let opaque=DryHit\(geometry\.x,dryPrepassDecodeNormal\(geometry\.yz\),identity&0xffffu,identity>>16u/);
    assert.match(reduced, /return vec4f\(shadeDryOpaque\(opaque,ro,rd\),opaque\.t\)/,
      "the isolated reduced-rate pass must shade the reconstructed coarse hit without another primary trace");
    const noGiStart = reduced.indexOf("fn dryPrepassShadeNoGi(");
    const noGiEnd = reduced.indexOf("@fragment fn dryPrepassShadeMain", noGiStart);
    const noGiTier = reduced.slice(noGiStart, noGiEnd);
    assert.ok(noGiStart >= 0 && noGiEnd > noGiStart);
    assert.doesNotMatch(noGiTier, /dryGlobalIllumination|dryLightVisibility|dryContactVisibility/,
      "the directly-lit tier must consume packed visibility without re-entering any live cone or exact-ray closure");
    assert.match(noGiTier, /dryPrepassChannel\(1u\+lightIndex\)/);
    assert.match(reduced, /if\(!packedValid\)[^]*return vec4f\(0\.0,0\.0,0\.0,-1\.0\)/,
      "an unresolved reduced receiver must fall through to exact full-rate lighting");
    assert.match(reduced, /let gi=dryGlobalIllumination\(ro\+rd\*opaque\.t,opaque\.normal,ignoredBodyOwner\)[^]*if\(dry\.tuningCounts2\.w==3u[^]*\|\|dry\.tuningCounts2\.w==4u\)\{return vec4f\(gi\.radiance,select\(-1\.0,gi\.visibility,gi\.valid!=0u\)\);\}[^]*dryPrepassGiState=1u;[^]*return vec4f\(shadeDryOpaque\(opaque,ro,rd\),opaque\.t\)/,
      "relight modes store a GI closure while reconstruction modes store a complete reduced-rate material result");
    assert.match(reduced, /if\(hit\.t>=DRY_MISS\)[^]*dryPrepassRadianceState==1u&&hit\.motionKind==DRY_GBUFFER_MOTION_STATIC[^]*let position=ro\+rd\*hit\.t;let surface=dryEvaluateSurfaceMaterial/,
      "exact-matched static radiance must return before full-resolution procedural material evaluation");
  }
  const productionCone = createSvoDrySceneFragmentWGSL(0.5, "canonical-parametric", "off", "split");
  assert.doesNotMatch(productionCone, /dryConeFallback/);
  const directStart = productionCone.indexOf("fn dryLightVisibility(");
  const directEnd = productionCone.indexOf("fn dryContactVisibilityRadius", directStart);
  const directVisibility = productionCone.slice(directStart, directEnd);
  const directConeMode = directVisibility.indexOf("if((dry.materialPublication.w&4u)!=0u)");
  const directInvalid = directVisibility.indexOf("if(cone.valid==0u)", directConeMode);
  const directConeReturn = directVisibility.indexOf("return mix(vec3f(1.0),raw,dry.tuningRays0.y);", directInvalid);
  const directExactMode = directVisibility.indexOf("let result=svoTraceVisibility", directConeReturn);
  assert.ok(directConeMode >= 0 && directInvalid > directConeMode && directConeReturn > directInvalid && directExactMode > directConeReturn,
    "cone-mode direct visibility must return before the explicitly selected exact tracer");
  const giStart = productionCone.indexOf("fn dryGlobalIllumination(");
  const giEnd = productionCone.indexOf("fn dryDiagnosticControl", giStart);
  assert.doesNotMatch(productionCone.slice(giStart, giEnd), /svoTraceVisibility/,
    "cone-mode GI must not silently invoke the exact tracer");
  assert.doesNotMatch(rendererSource, /lightingMode/,
    "GLOBAL must not retain a selectable lighting backend");
  assert.match(rendererSource, /label: "Sparse voxel persistent world GI cache"[^]*gi\.setPipeline\(this\.worldGiFramePipeline!\)[^]*gi\.setPipeline\(this\.worldGiCachePipeline!\)/,
    "split GLOBAL must produce the reduced GI target through the persistent world-space cache");
  assert.match(rendererSource, /if \(!reconstructReducedRadiance\)[^]*label: "Sparse voxel persistent world GI cache"/,
    "GI-only cache output must be restricted to full-rate relight modes");
  assert.match(rendererSource, /if \(usePrepass && reconstructReducedRadiance[^]*label: "Sparse voxel reduced-rate opaque shading"/,
    "radiance reconstruction modes must execute the complete reduced-rate material pass");
  assert.match(rendererSource, /entryPoint: "dryReconstructedLightingMain"[^]*if \(usePrepass && reconstructReducedRadiance\)[^]*splitReconstructedLightingPipeline[^]*splitLightingPipeline/,
    "reconstructed receivers and exact edge fallbacks must run as separate occupancy tiers");
  const compactVisibility = rendererSource.indexOf('label: "Sparse voxel compact cone visibility"');
  const persistentGi = rendererSource.indexOf('label: "Sparse voxel persistent world GI cache"', compactVisibility);
  const deferredLighting = rendererSource.indexOf('label: "Sparse voxel deferred dry lighting"', persistentGi);
  assert.ok(compactVisibility >= 0 && persistentGi > compactVisibility && deferredLighting > persistentGi,
    "split GLOBAL must evaluate environmental GI after current-frame compact visibility and before deferred lighting");
});

test("edge reconstruction is bounded, motion-safe, and promotes only unresolved pixels", () => {
  const shader = createSvoDrySceneFragmentWGSL(0.5, "canonical-parametric", "off", "split");
  assert.match(shader, /for\(var j=0u;j<2u;j\+=1u\)\{for\(var i=0u;i<2u;i\+=1u\)/,
    "the common shipping path must remain bounded to its ordinary four receiver candidates");
  assert.match(shader, /if\(!identityMatches\)[^]*continue;[^]*if\(all\(packed\.xy==DRY_PREPASS_INVALID_PACKED\)\)[^]*continue;/,
    "other identities and invalid pages must be rejected before they enter the reconstruction");
  assert.match(shader, /for\(var radius=1;radius<=2;radius\+=1\)[^]*dryPrepassUseExactReceiver/,
    "the exceptional receiver search must have a hard two-texel radius");
  assert.match(shader, /dryPrepassRecoverExactReceiver\(coordinate,dims,depth,normal,hit\)[^]*dryPrepassExactEdgeState=1u;return;/,
    "only pixels with no lawful receiver may enter full-rate live cone evaluation");
  assert.match(shader, /if\(cone\.valid==0u\)\{dryDerivedPageFailure\|=2u;return vec3f\(0\.0\);\}/,
    "an unavailable page in the edge tier must remain explicitly fail-visible");
});

test("primary seam closure brackets background with opposing foreground surfaces", () => {
  const reduced = createSvoDrySceneFragmentWGSL(0.5, "hybrid", "off", "split");
  assert.match(reduced, /fn dryPublicationWord\(index:u32\)->u32\{return svoStructure\[dry\.structureOffsets\.y\+index\];\}/,
    "exact refinement addresses publication through the unified structural arena");
  const classifierStart = reduced.indexOf("fn drySilhouetteAmbiguous");
  const refinerStart = reduced.indexOf("fn drySilhouetteRefineMain");
  const exactStart = reduced.indexOf("fn drySilhouetteTraceVisibilityExact");
  const exactEnd = reduced.indexOf("@fragment fn dryPrepassVisibilityMain", exactStart);
  const refinerEnd = reduced.indexOf("\n}", refinerStart) + 2;
  assert.ok(classifierStart >= 0 && refinerStart > classifierStart && refinerEnd > refinerStart);
  const classifier = reduced.slice(classifierStart, refinerStart);
  const refiner = reduced.slice(refinerStart, refinerEnd);

  assert.match(classifier, /drySplitGeometryAt\(coordinate\)[^]*drySplitIdentityAt\(coordinate\)[^]*array<vec2i,4>/,
    "ambiguity is classified from the current full-rate primary publication, independent of scene topology");
  assert.match(classifier, /!sameSurface\|\|!depthClose/,
    "hit\/miss, owner\/material identity, and meaningful depth discontinuities enter the worklist");
  assert.doesNotMatch(classifier, /normalClose|\.9999/,
    "smoothly varying normals must not classify an entire curved surface as a silhouette");
  assert.match(classifier, /if\(!drySilhouetteAmbiguous\(coordinate,dimensions\)\)\{return;\}let queueIndex=atomicAdd/,
    "only classified pixels may consume expensive full-resolution visibility work");
  assert.match(refiner, /dryPrepassBoundaryQueue\.coordinates\[queueIndex\][^]*drySplitGeometryAt[^]*drySilhouetteTraceVisibilityExact/,
    "the refiner consumes the compacted current-frame hit and invokes its explicit authoritative visibility tier");
  assert.doesNotMatch(refiner, /traceOpaqueScene/,
    "the primary current-frame hit must not be retraced");
  const exact = reduced.slice(exactStart, exactEnd);
  assert.match(exact, /svoTraceVisibility/,
    "queued pixels deliberately run the existing bounded exact visibility traversal");
  assert.doesNotMatch(exact, /dryConeVisibility/,
    "live-derived cone page availability cannot invalidate the authoritative silhouette tier");
  assert.match(refiner, /exact\.status==DRY_SILHOUETTE_EXACT_EXHAUSTED[^]*invalidAoPages[^]*failedRefinements[^]*exact\.status==DRY_SILHOUETTE_EXACT_INVALID[^]*invalidDirectPages[^]*failedRefinements/,
    "bounded exhaustion and invalid exact traversal are counted separately and both remain explicit failures");
  assert.match(reduced, /fn dryPrimarySeamSample\(coordinate:vec2i\)[^]*array<vec4i,4>[^]*dryPrimarySeamForeground\(first\.w,centreDepth\)[^]*dryPrimarySeamForeground\(second\.w,centreDepth\)/,
    "closure requires foreground hits on both opposing sides of the candidate pixel");
  assert.match(reduced, /differentSurface[^]*if\(!differentSurface\)\{continue;\}[^]*candidateDepth=max\(first\.w,second\.w\)/,
    "closure is restricted to seams between distinct surfaces and extends the rear surface");
  assert.match(reduced, new RegExp(`materialPublication\\.w&${128}u[^]*dryPrimarySeamSample\\(coordinate\\)[^]*geometry=seam\\.geometry`),
    "the deferred surface pass consumes the same seam decision that patched primary depth");

  assert.equal(SVO_DRY_CONE_PREPASS_CONTRACT.silhouetteRefinementFormat, "rg32uint");
  assert.equal(SVO_DRY_CONE_PREPASS_CONTRACT.silhouetteRefinementStateFormat, "r32uint");
  assert.equal(SVO_DRY_SILHOUETTE_REFINEMENT_CONTRACT.workgroupSize, 64);
  assert.equal(SVO_DRY_SILHOUETTE_REFINEMENT_CONTRACT.counterSizeBytes, 8);
  assert.equal(SVO_DRY_SILHOUETTE_REFINEMENT_CONTRACT.diagnosticCounterSizeBytes, 16);
});

test("primary seam closure patches the G-buffer before sky and deferred lighting", () => {
  const primary = rendererSource.indexOf('label: "Sparse voxel exact live-scene primitive visibility"');
  const seam = rendererSource.indexOf('label: "Sparse voxel primary seam closure"', primary);
  const lighting = rendererSource.indexOf('label: "Sparse voxel deferred dry lighting"', seam);
  assert.ok(primary >= 0 && seam > primary && lighting > seam,
    "the optional coverage patch must run after all primary producers and before the sky/surface partition");
  assert.match(rendererSource, /label: `Sparse voxel primary seam closure x\$\{scale\}`[^]*entryPoint: "dryPrimarySeamMain"[^]*depthWriteEnabled: true[^]*depthCompare: SVO_GBUFFER_RENDER_TARGET_CONTRACT\.depthCompare/,
    "the pass updates authoritative reversed-Z depth rather than painting over the final color");
  assert.match(rendererSource, /if \(this\.silhouetteRefinementEnabled\)[^]*gBufferViews\.packedSurface[^]*gBufferViews\.identityMedia[^]*gBufferViews\.hardwareDepth[^]*seam\.draw\(3\)/,
    "the explicit toggle controls one full-screen G-buffer seam pass");
  assert.doesNotMatch(rendererSource, /beginComputePass\(\{ label: "Sparse voxel silhouette refinement/,
    "the retired lighting-visibility worklist must not execute");
});

test("occupancy experiments alter only their intended reduced-shader mechanisms", () => {
  // The per-pixel cost counters are gone from every variant: render stage
  // views read published planes, so nothing observes an invocation-private
  // tally and no shader pays to keep one live.
  const reduced = createSvoDrySceneFragmentWGSL(
    0.25, "canonical-parametric", "off", "split", 0, false, false, false, false,
  );
  assert.doesNotMatch(reduced,
    /dryCostOverlay|dryPrimaryNodeVisits|dryPrimaryLeafVisits|dryPrimaryVoxelWorkItems|dryShadowNodeVisits|dryMipSteps|dryTraversalFailure/);

  const inlineBoundaries = createSvoDrySceneFragmentWGSL(
    0.25, "canonical-parametric", "off", "split", 0, false, false, false, false,
    { inlineConeBoundaries: true },
  );
  assert.match(inlineBoundaries, /if\(!homogeneous\)\{let opaque=traceOpaqueScene/);
  assert.doesNotMatch(inlineBoundaries, /if\(!homogeneous\)\{let queueIndex=atomicAdd/);

  const uncachedGi = createSvoDrySceneFragmentWGSL(
    0.25, "canonical-parametric", "off", "split", 0, false, false, false, false,
    { dropGiPageCache: true },
  );
  assert.doesNotMatch(uncachedGi, /var<private> dryGiPageCache/);
  assert.match(uncachedGi, /fn svoTetraRadianceConeLoad[^]*var pageCache=DryNodeMipPageCache/);

  const half = createSvoDrySceneFragmentWGSL(
    0.25, "canonical-parametric", "off", "split", 0, false, false, false, false,
    { halfPrecisionLighting: true },
  );
  assert.match(half, /^enable f16;/);
  assert.match(half, /var<private> dryPrepassData0:vec4h/);
  assert.doesNotMatch(half, /var<private> dryPrepassData0:vec4f/);

  const shortStack = createSvoDrySceneFragmentWGSL(
    0.25, "canonical-parametric", "off", "split", 0, false, false, false, false,
    { shortTraversalStack: true },
  );
  assert.match(shortStack, /const SVO_STACK_CAPACITY: u32 = 16u/);
  assert.match(shortStack, /stack: array<SvoStackEntry, 16>/);
  assert.doesNotMatch(shortStack, /array<SvoStackEntry, 32>/);
  assert.equal(createSvoDrySceneFragmentWGSL(1), svoDrySceneShader,
    "all occupancy experiments remain opt-in and preserve scale-1 production bytes");
});

test("prepass target contract and sizing", () => {
  assert.equal(SVO_DRY_CONE_PREPASS_CONTRACT.visibilityFormat, "rg32uint");
  assert.equal(SVO_DRY_CONE_PREPASS_CONTRACT.visibilityTargetCount, 1);
  assert.equal(SVO_DRY_CONE_PREPASS_CONTRACT.geometryFormat, "rgba16float");
  assert.equal(SVO_DRY_CONE_PREPASS_CONTRACT.identityFormat, "r32uint");
  assert.equal(SVO_DRY_CONE_PREPASS_CONTRACT.radianceFormat, "rgba16float");
  assert.equal(SVO_DRY_CONE_PREPASS_CONTRACT.silhouetteRefinementFormat, "rg32uint");
  assert.equal(SVO_DRY_CONE_PREPASS_CONTRACT.silhouetteRefinementStateFormat, "r32uint");
  assert.equal(SVO_DRY_CONE_PREPASS_CONTRACT.maximumPrepassLights, 8);
  assert.deepEqual(svoConePrepassSize(1280, 720, 0.5), [640, 360]);
  assert.deepEqual(svoConePrepassSize(1280, 720, 0.25), [320, 180]);
  assert.deepEqual(svoConePrepassSize(1280, 720, 0.125), [160, 90]);
  assert.deepEqual(svoConePrepassSize(1281, 721, 0.5), [641, 361]);
  assert.deepEqual(svoConePrepassSize(1281, 721, 0.125), [160, 90]);
  assert.deepEqual(svoConePrepassSize(1, 1, 0.25), [1, 1], "prepass targets never collapse below 1x1");
  assert.deepEqual(svoConePrepassSize(1, 1, 0.125), [1, 1]);
  assert.deepEqual(svoConePrepassSize(1280, 720, 1), [1280, 720]);
});

function mockSource(): SparseVoxelRenderSource {
  const resource = { buffer: {} as GPUBuffer };
  return {
    materialCount: 8,
    pbrMaterials: { binding: resource, count: 8, strideBytes: SVO_MATERIAL_RECORD_STRIDE_BYTES, revision: 1 },
    structural: {
      structure: resource,
      structureOffsetsWords: { control: 0, publication: 0, nodes: 0, leaves: 0 },
      control: resource, nodes: resource, leaves: resource, geometry: resource,
      velocity: resource, materialOwners: resource, fluidLeafStates: resource,
      publication: { state: resource, byteLength: 32 },
      domain: { worldOrigin_m: [0, 0, 0], cellSize_m: [.1, .1, .1], dimensionsCells: [16, 16, 16], brickSize: 8, maximumDepth: 1 },
      capacities: { nodes: 8, leaves: 8, geometryVoxels: 4096, velocityVoxels: 4096, materialOwnerVoxels: 4096, fluidLeafStates: 8 },
      strides: { control: 4, node: 32, leaf: 16, geometry: 16, velocity: 16, materialOwner: 4, fluidLeafState: 4 },
      fields: {
        topology: { residency: "all-published-leaves", validity: "published-generation", revision: 1 },
        sceneGeometry: { residency: "all-published-leaves", validity: "published-generation", revision: 1 },
        materialOwner: { residency: "all-published-leaves", validity: "published-generation", revision: 1 },
        dynamicSolid: { residency: "unavailable", validity: "unavailable", revision: 0 },
        coarseFluid: { residency: "unavailable", validity: "unavailable", revision: 0 },
        fineFluid: { residency: "unavailable", validity: "unavailable", revision: 0 },
      },
      generation: { published: 1, completed: 1 },
    },
  } as unknown as SparseVoxelRenderSource;
}

test("the cone-lighting scale is an optional lighting option that defaults to the inline path", () => {
  const previousBufferUsage = globalThis.GPUBufferUsage, previousTextureUsage = globalThis.GPUTextureUsage;
  Object.assign(globalThis, {
    GPUBufferUsage: { UNIFORM: 1, COPY_DST: 2, STORAGE: 4, MAP_READ: 8 },
    GPUTextureUsage: { TEXTURE_BINDING: 1, RENDER_ATTACHMENT: 2 },
  });
  const writes: Array<{ label?: string }> = [];
  const device = {
    createBuffer(descriptor: { label?: string }) { return { label: descriptor.label, destroy() {} }; },
    createTexture() { return { createView() { return {}; }, destroy() {} }; },
    createSampler() { return {}; },
    queue: {
      writeBuffer(target: { label?: string }) { writes.push({ label: target.label }); },
    },
  } as unknown as GPUDevice;
  try {
    const renderer = new SparseVoxelDrySceneRenderer(device, {} as GPUBuffer, {} as GPUBuffer);
    assert.deepEqual(renderer.presentationBundleStatus, { state: "ready" });
    renderer.setSource(mockSource());
    renderer.publishScene(svoDrySceneFixture);
    assert.equal(renderer.coneLightingScale, 1, "callers without the option keep the historical inline path");
    const paramsWrites = () => writes.filter(({ label }) => label === "Sparse voxel dry scene parameters").length;
    const beforeRepeat = paramsWrites();
    renderer.setLightingOptions({ shadowsEnabled: true, ambientOcclusionEnabled: true });
    assert.equal(paramsWrites(), beforeRepeat, "unchanged options (including implicit scale 1) must short-circuit");
    renderer.setLightingOptions({ shadowsEnabled: true, ambientOcclusionEnabled: true, coneLightingScale: 0.5 });
    assert.equal(renderer.coneLightingScale, 0.5);
    assert.deepEqual(renderer.presentationBundleStatus, {
      state: "compiling", detail: "Compiling requested SVO cone bundle at scale 0.5",
    });
    renderer.setLightingOptions({ shadowsEnabled: true, ambientOcclusionEnabled: true, coneLightingScale: 0.5 });
    renderer.setLightingOptions({ shadowsEnabled: true, ambientOcclusionEnabled: true, coneLightingScale: 0.125 });
    assert.equal(renderer.coneLightingScale, 0.125, "the 8x8 option reaches the renderer without normalization loss");
    renderer.setLightingOptions({ shadowsEnabled: true, ambientOcclusionEnabled: true });
    assert.equal(renderer.coneLightingScale, 1, "omitting the option returns to the inline path");
    renderer.destroy();
  } finally {
    Object.assign(globalThis, { GPUBufferUsage: previousBufferUsage, GPUTextureUsage: previousTextureUsage });
  }
});

test("primary seam closure is opt-in for every split lighting rate", () => {
  const previousBufferUsage = globalThis.GPUBufferUsage, previousTextureUsage = globalThis.GPUTextureUsage;
  Object.assign(globalThis, {
    GPUBufferUsage: { UNIFORM: 1, COPY_DST: 2, STORAGE: 4, MAP_READ: 8 },
    GPUTextureUsage: { TEXTURE_BINDING: 1, RENDER_ATTACHMENT: 2 },
  });
  const device = {
    createBuffer(descriptor: { label?: string; size?: number }) {
      return { label: descriptor.label, size: descriptor.size ?? 0, destroy() {} };
    },
    createTexture() { return { createView() { return {}; }, destroy() {} }; },
    createSampler() { return {}; },
    queue: { writeBuffer() {} },
  } as unknown as GPUDevice;
  try {
    const renderer = new SparseVoxelDrySceneRenderer(
      device, {} as GPUBuffer, {} as GPUBuffer, "rgba16float", "hybrid", "off", "split",
    );
    assert.deepEqual(renderer.silhouetteRefinementStatus, { state: "disabled" },
      "scale 1 also keeps primary seam closure opt-in");
    renderer.setLightingOptions({ shadowsEnabled: true, ambientOcclusionEnabled: true, coneLightingScale: 0.5 });
    assert.deepEqual(renderer.silhouetteRefinementStatus, { state: "disabled" },
      "the reduced split path keeps experimental seam treatment opt-in");
    renderer.setLightingOptions({ shadowsEnabled: true, ambientOcclusionEnabled: true, coneLightingScale: 0.5,
      silhouetteRefinementEnabled: true });
    assert.equal(renderer.silhouetteRefinementStatus.state, "compiling",
      "an explicit request reports resource readiness rather than silently substituting another path");
    renderer.setLightingOptions({ shadowsEnabled: true, ambientOcclusionEnabled: true, coneLightingScale: 0.5,
      coneTracingMode: "exact", silhouetteRefinementEnabled: true });
    assert.equal(renderer.silhouetteRefinementStatus.state, "compiling",
      "primary coverage does not depend on the selected secondary visibility algorithm");
    renderer.destroy();
  } finally {
    Object.assign(globalThis, { GPUBufferUsage: previousBufferUsage, GPUTextureUsage: previousTextureUsage });
  }
});

const modulePath = process.env.WEBGPU_NODE_MODULE;
test("reduced shader variants compile with both entry points on the GPU backend", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU cone-prepass checks",
}, async () => {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]), adapter = await gpu.requestAdapter({ powerPreference: "high-performance" }); assert.ok(adapter);
  const device = await adapter.requestDevice();
  try {
    for (const scale of [0.5, 0.25, 0.125] as const) {
      const code = createSvoDrySceneFragmentWGSL(scale, "hybrid", "off", "split");
      const module = device.createShaderModule({ label: `Cone-prepass dry shader validation x${scale}`, code });
      const info = await module.getCompilationInfo();
      assert.deepEqual(info.messages.filter(({ type }) => type === "error"), []);
      for (const entryPoint of [
        "drySilhouetteResetMain",
        "drySilhouetteClassifyMain",
        "drySilhouetteFinalizeMain",
        "drySilhouetteRefineMain",
      ]) {
        try {
          await device.createComputePipelineAsync({
            label: `Silhouette refinement ${entryPoint} x${scale}`,
            layout: "auto",
            compute: { module, entryPoint },
          });
        } catch (error) {
          throw new Error(`Metal rejected ${entryPoint} at scale ${scale}: ${String(error)}`, { cause: error });
        }
      }
    }

    const width = 4, height = 4, bytesPerRow = 256;
    const code = createSvoDrySceneFragmentWGSL(0.5, "hybrid", "off", "split");
    const module = device.createShaderModule({ label: "Executable silhouette classifier", code });
    const classifier = await device.createComputePipelineAsync({
      label: "Executable production silhouette classifier", layout: "auto",
      compute: { module, entryPoint: "drySilhouetteClassifyMain" },
    });
    const geometry = device.createTexture({
      label: "Synthetic primary geometry", size: [width, height], format: "rgba32float",
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
    });
    const identity = device.createTexture({
      label: "Synthetic primary identity", size: [width, height], format: "rg32uint",
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
    });
    const refinement = device.createTexture({
      label: "Synthetic refinement publication", size: [width, height], format: "rg32uint",
      usage: GPUTextureUsage.STORAGE_BINDING,
    });
    const refinementState = device.createTexture({
      label: "Synthetic refinement state", size: [width, height], format: "r32uint",
      usage: GPUTextureUsage.STORAGE_BINDING,
    });
    const queue = device.createBuffer({
      label: "Synthetic silhouette worklist",
      size: Uint32Array.BYTES_PER_ELEMENT * (SVO_DRY_SILHOUETTE_REFINEMENT_CONTRACT.queueHeaderWords + width * height),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const readback = device.createBuffer({
      label: "Synthetic silhouette count readback", size: Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const geometryRows = new Float32Array(bytesPerRow / Float32Array.BYTES_PER_ELEMENT * height);
    const identityRows = new Uint32Array(bytesPerRow / Uint32Array.BYTES_PER_ELEMENT * height);
    for (const [x, y] of [[1, 1], [2, 1], [1, 2], [2, 2]] as const) {
      const geometryIndex = y * (bytesPerRow / 4) + x * 4;
      geometryRows.set([0, 0, 1, 1], geometryIndex);
      const identityIndex = y * (bytesPerRow / 4) + x * 2;
      identityRows.set([1, 7], identityIndex);
    }
    device.queue.writeTexture({ texture: geometry }, geometryRows,
      { bytesPerRow, rowsPerImage: height }, { width, height });
    device.queue.writeTexture({ texture: identity }, identityRows,
      { bytesPerRow, rowsPerImage: height }, { width, height });
    device.queue.writeBuffer(queue, 0, new Uint32Array(SVO_DRY_SILHOUETTE_REFINEMENT_CONTRACT.queueHeaderWords));
    const classifyResources = device.createBindGroup({
      layout: classifier.getBindGroupLayout(1), entries: [
        { binding: 7, resource: { buffer: queue } },
        { binding: 9, resource: refinement.createView() },
        { binding: 12, resource: refinementState.createView() },
      ],
    });
    const primaryResources = device.createBindGroup({
      layout: classifier.getBindGroupLayout(2), entries: [
        { binding: 1, resource: geometry.createView() },
        { binding: 5, resource: identity.createView() },
      ],
    });
    const encoder = device.createCommandEncoder({ label: "Execute production silhouette classifier" });
    const pass = encoder.beginComputePass();
    pass.setPipeline(classifier);
    pass.setBindGroup(1, classifyResources);
    pass.setBindGroup(2, primaryResources);
    pass.dispatchWorkgroups(1, 1);
    pass.end();
    encoder.copyBufferToBuffer(queue, 0, readback, 0, Uint32Array.BYTES_PER_ELEMENT);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    assert.equal(new Uint32Array(readback.getMappedRange())[0], 4,
      "four current-frame hit\/miss silhouette pixels must execute into the production worklist");
    readback.unmap();
    readback.destroy(); queue.destroy(); refinementState.destroy(); refinement.destroy(); identity.destroy(); geometry.destroy();
  } finally { device.destroy(); }
});
