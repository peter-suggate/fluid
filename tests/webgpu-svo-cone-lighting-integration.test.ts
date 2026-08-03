import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { SVO_MATERIAL_RECORD_STRIDE_BYTES } from "../lib/svo-material-abi";
import { SVO_FLUID_COVERAGE_LAYOUT } from "../lib/svo-fluid-coverage";
import {
  createSvoDrySceneFragmentWGSL,
  SparseVoxelDrySceneRenderer,
  SVO_DRY_DERIVED_FAILURE,
  SVO_DRY_DERIVED_FAILURE_COUNTERS,
  SVO_DRY_NODE_MIP_PUBLICATION_MODE,
  SVO_DRY_SCENE_PARAMS_LAYOUT,
  SVO_DRY_VISIBILITY_FLAGS,
  SVO_DRY_WORLD_GI_CACHE_CONTRACT,
  svoDryPrimitiveArenaCacheInvalidation,
  svoDrySceneShader,
} from "../lib/webgpu-svo-dry-scene";
import type { SparseVoxelRenderSource } from "../lib/webgpu-voxel-debug";
import { SVO_CONE_RADIANCE_RECONSTRUCTION_CODES } from "../lib/svo-render-tuning";
import { svoDrySceneFixture } from "./svo-dry-scene-test-fixture";

const drySource = readFileSync(new URL("../lib/webgpu-svo-dry-scene.ts", import.meta.url), "utf8");
const worldSource = readFileSync(new URL("../lib/webgpu-octree-sparse-bricks.ts", import.meta.url), "utf8");
const sourceAbi = readFileSync(new URL("../lib/webgpu-voxel-debug.ts", import.meta.url), "utf8");

test("live derived page validity uses sampled uint bindings with zero fallbacks", () => {
  assert.doesNotMatch(drySource, /dryDerivedFailureColor|mix\(shaded,dryDerivedFailure/,
    "typed page failures must not be painted into production scene colour");
  assert.match(drySource, /nodeMipPageValidityFallback = device\.createTexture\([^]*size: \[1, 1\][^]*format: "r32uint"/);
  assert.match(drySource, /tetrahedralRadiancePageValidityFallback = device\.createTexture\([^]*size: \[1, 1\][^]*format: "r32uint"/);
  assert.match(drySource, /binding: 26, resource: nodeMipPageValidity \?\? this\.nodeMipPageValidityFallbackView/);
  assert.match(drySource, /binding: 27, resource: tetrahedralRadiancePageValidity \?\? this\.tetrahedralRadiancePageValidityFallbackView/);
  const reduced = createSvoDrySceneFragmentWGSL(0.5, "canonical-parametric", "off", "split", 0, false, false, false, true);
  assert.match(reduced, /if\(all\(packed\.xy==DRY_PREPASS_INVALID_PACKED\)\)\{linearSafe=0u;continue;\}/,
    "one invalid fan-out receiver must be isolated from unrelated identities");
  assert.match(reduced, /if\(weightSum<0\.05\)\{[^]*dryPrepassRecoverExactReceiver[^]*dryPrepassExactEdgeState=1u;return;/,
    "fan-out invalidity must enter only the declared live edge tier when no compatible receiver exists");
  assert.match(reduced, new RegExp(`if\\(cone\\.valid==0u\\)\\{dryDerivedPageFailure\\|=${SVO_DRY_DERIVED_FAILURE.directVisibilityPage}u;return vec3f\\(0\\.0\\);\\}`),
    "the edge tier must retain typed failure publication for genuinely unavailable live pages");
});

function source(): SparseVoxelRenderSource {
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

test("GLOBAL lighting and its visibility effects write independent flags", () => {
  assert.deepEqual(SVO_DRY_VISIBILITY_FLAGS, {
    exactContact: 1, exactShadow: 2, coneLightingRequested: 4, ambientOcclusion: 8,
    globalIllumination: 16, globalIlluminationOcclusion: 32, globalIlluminationRequested: 64,
    silhouetteRefinement: 128,
  });
  assert.match(drySource, /const coneTracingEnabled = coneTracingMode === "cones" && this\.derivedLightingReady\(\)/);
  assert.doesNotMatch(drySource, /lightingMode|setLightingMode/);
  const previousBufferUsage = globalThis.GPUBufferUsage, previousTextureUsage = globalThis.GPUTextureUsage;
  Object.assign(globalThis, {
    GPUBufferUsage: { UNIFORM: 1, COPY_DST: 2, STORAGE: 4, MAP_READ: 8 },
    GPUTextureUsage: { TEXTURE_BINDING: 1 },
  });
  const writes: Array<{ label?: string; words: Uint32Array }> = [];
  const device = {
    createBuffer(descriptor: { label?: string }) { return { label: descriptor.label, destroy() {} }; },
    createTexture() { return { createView() { return {}; }, destroy() {} }; },
    createSampler() { return {}; },
    queue: {
      writeBuffer(target: { label?: string }, _offset: number, data: ArrayBuffer | ArrayBufferView) {
        const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        writes.push({ label: target.label, words: new Uint32Array(bytes.slice().buffer) });
      },
    },
  } as unknown as GPUDevice;
  try {
    const renderer = new SparseVoxelDrySceneRenderer(device, {} as GPUBuffer, {} as GPUBuffer);
    renderer.setSource(source());
    renderer.publishScene(svoDrySceneFixture);
    const params = () => writes.filter(({ label }) => label === "Sparse voxel dry scene parameters");
    const flagWord = (write: { words: Uint32Array }) => write.words[SVO_DRY_SCENE_PARAMS_LAYOUT.materialPublicationWordOffset + 3];
    assert.equal(flagWord(params().at(-1)!) & SVO_DRY_VISIBILITY_FLAGS.exactShadow, SVO_DRY_VISIBILITY_FLAGS.exactShadow);
    assert.equal(flagWord(params().at(-1)!) & SVO_DRY_VISIBILITY_FLAGS.ambientOcclusion, SVO_DRY_VISIBILITY_FLAGS.ambientOcclusion);
    assert.equal(flagWord(params().at(-1)!) & SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested, 0,
      "an unavailable cone hierarchy selects exact visibility instead of poisoning the frame");
    assert.deepEqual(renderer.lightingVisibilityStatus, {
      state: "exact", fallback: true,
      detail: "Complete cone-lighting hierarchy is unavailable; exact SVO shadows and AO are active",
    });
    renderer.setLightingOptions({ shadowsEnabled: false, ambientOcclusionEnabled: true });
    assert.equal(flagWord(params().at(-1)!) & SVO_DRY_VISIBILITY_FLAGS.exactShadow, 0);
    assert.equal(flagWord(params().at(-1)!) & SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested, 0);
    renderer.setLightingOptions({ shadowsEnabled: false, ambientOcclusionEnabled: false });
    assert.equal(flagWord(params().at(-1)!) & (SVO_DRY_VISIBILITY_FLAGS.exactContact | SVO_DRY_VISIBILITY_FLAGS.exactShadow | SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested | SVO_DRY_VISIBILITY_FLAGS.ambientOcclusion), 0);
    renderer.setLightingOptions({ shadowsEnabled: true, ambientOcclusionEnabled: true });
    const requestedWithoutDerivedFlags = flagWord(params().at(-1)!);
    assert.equal(requestedWithoutDerivedFlags & SVO_DRY_VISIBILITY_FLAGS.globalIlluminationRequested, 0,
      "the effective exact path must not ask shader stages to sample an unavailable GI atlas");
    assert.equal(requestedWithoutDerivedFlags & SVO_DRY_VISIBILITY_FLAGS.globalIllumination, 0);
    const giSource = source() as unknown as SparseVoxelRenderSource & Record<string, unknown>;
    const plan = { generation: 1, complete: true, pages: [{ key: { generation: 1, level: 0, coordinate: [0, 0, 0] }, slot: 0 }], atlas: { texels: [10, 10, 10] } };
    Object.assign(giSource, {
      nodeMipPyramid: { generation: 1, plan, worldOrigin_m: [0, 0, 0], pageValidity: { view: {} } },
      tetrahedralRadiance: { generation: 1, plan, views: [{}, {}, {}, {}] },
    });
    renderer.setSource(giSource as unknown as SparseVoxelRenderSource);
    renderer.publishScene(svoDrySceneFixture);
    const giFlags = flagWord(params().at(-1)!);
    assert.equal(giFlags & SVO_DRY_VISIBILITY_FLAGS.globalIllumination, SVO_DRY_VISIBILITY_FLAGS.globalIllumination);
    assert.equal(giFlags & SVO_DRY_VISIBILITY_FLAGS.globalIlluminationRequested,
      SVO_DRY_VISIBILITY_FLAGS.globalIlluminationRequested);
    assert.equal(giFlags & SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested, SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested);
    assert.deepEqual(renderer.lightingVisibilityStatus, { state: "cones" });
    assert.equal(giFlags & SVO_DRY_VISIBILITY_FLAGS.globalIlluminationOcclusion,
      SVO_DRY_VISIBILITY_FLAGS.globalIlluminationOcclusion, "AO enables broad GI-cone visibility");
    assert.equal(giFlags & (SVO_DRY_VISIBILITY_FLAGS.exactContact | SVO_DRY_VISIBILITY_FLAGS.ambientOcclusion), 0,
      "GI replaces the standalone AO/contact cones");
    assert.equal(params().at(-1)!.words[SVO_DRY_SCENE_PARAMS_LAYOUT.nodeMipWordOffset + 3],
      SVO_DRY_NODE_MIP_PUBLICATION_MODE.pageValidity,
      "live derived pages remain globally usable across topology revisions; page validity owns freshness");
    assert.deepEqual([...params().at(-1)!.words.slice(SVO_DRY_SCENE_PARAMS_LAYOUT.tetrahedralRadianceWordOffset,
      SVO_DRY_SCENE_PARAMS_LAYOUT.tetrahedralRadianceWordOffset + 4)], [1, 1, 0, 0]);
    assert.equal(params().at(-1)!.words[SVO_DRY_SCENE_PARAMS_LAYOUT.tuningWordOffset + 11],
      SVO_CONE_RADIANCE_RECONSTRUCTION_CODES["full-res-relight"],
      "GLOBAL must not reconstruct an unwritten reduced-radiance plane as black");
    const giParams = new Float32Array(params().at(-1)!.words.buffer);
    assert.deepEqual([...giParams.slice(SVO_DRY_SCENE_PARAMS_LAYOUT.giLightingWordOffset,
      SVO_DRY_SCENE_PARAMS_LAYOUT.giLightingWordOffset + 4)], [1.7999999523162842, 0.6499999761581421, 0.8500000238418579, 1]);
    assert.deepEqual([...giParams.slice(SVO_DRY_SCENE_PARAMS_LAYOUT.giConesWordOffset,
      SVO_DRY_SCENE_PARAMS_LAYOUT.giConesWordOffset + 2)], [1.0499999523162842, 4]);
    renderer.setLightingOptions({ shadowsEnabled: true, ambientOcclusionEnabled: false });
    assert.equal(flagWord(params().at(-1)!) & SVO_DRY_VISIBILITY_FLAGS.globalIlluminationOcclusion, 0,
      "the AO switch remains meaningful in GLOBAL mode");
    renderer.destroy();
  } finally {
    Object.assign(globalThis, { GPUBufferUsage: previousBufferUsage, GPUTextureUsage: previousTextureUsage });
  }
});

test("missing or stale node-mip samples fail closed without an exact traversal escape", () => {
  const start = svoDrySceneShader.indexOf("fn dryLightVisibility(");
  const end = svoDrySceneShader.indexOf("fn dryContactVisibilityRadius", start);
  const visibility = svoDrySceneShader.slice(start, end);
  assert.match(visibility, /if\(cone\.valid==0u\)\{dryDerivedPageFailure\|=2u;return vec3f\(0\.0\);\}/);
  const coneMode = visibility.indexOf("if((dry.materialPublication.w&4u)!=0u)");
  const invalidCone = visibility.indexOf("if(cone.valid==0u)", coneMode);
  const coneReturn = visibility.indexOf("return mix(vec3f(1.0),raw,dry.tuningRays0.y);", invalidCone);
  const exactMode = visibility.indexOf("let result=svoTraceVisibility", coneReturn);
  assert.ok(coneMode >= 0 && invalidCone > coneMode && coneReturn > invalidCone && exactMode > coneReturn,
    "the exact tracer must remain reachable only after cone mode has returned");
  assert.doesNotMatch(drySource,
    /coneLightingRequested[^\n]*globalIllumination[^\n]*==0u/,
    "GLOBAL reuses hierarchical shadow visibility instead of forcing exact SVO rays for every light");
  assert.match(svoDrySceneShader, /let generationReady=dry\.nodeMip\.w==2u\|\|dry\.nodeMip\.x==dryPublicationWord\(2u\)/,
    "live page-local validity survives structural revisions while generation-fenced sources remain exact");
});

test("cone steps reuse a matching mip page and search only on page, LOD, or generation changes", () => {
  const lookupStart = svoDrySceneShader.indexOf("fn dryNodeMipAt(");
  const lookupEnd = svoDrySceneShader.indexOf("struct DryConeVisibility", lookupStart);
  const lookup = svoDrySceneShader.slice(lookupStart, lookupEnd);
  assert.match(svoDrySceneShader, /struct DryNodeMipPageCache\{coordinate:vec3u,level:u32,pageOrigin:vec3u,generation:u32,resident:u32,pageIndex:u32,blackRadiance:u32\}/);
  assert.match(lookup, /pageCache:ptr<function,DryNodeMipPageCache>/);
  assert.match(lookup, /generation!=dry\.nodeMip\.x\|\|\(\*pageCache\)\.level!=level\|\|any\(\(\*pageCache\)\.coordinate!=pageCoordinate\)/);
  assert.match(lookup, /\*pageCache=DryNodeMipPageCache\(pageCoordinate,level,vec3u\(0u\),dry\.nodeMip\.x,0u,0xffffffffu,0u\);let pageIndex=dryNodeMipFind\(level,pageCoordinate\)/,
    "the queried key is cached as non-resident before searching so failed searches are reusable");
  assert.match(lookup, /pageIndex!=0xffffffffu[^]*\*pageCache=DryNodeMipPageCache\(pageCoordinate,level,entry\.pageOrigin,entry\.generation,1u,pageIndex,0u\)/);
  assert.match(lookup, /if\(\(\*pageCache\)\.resident==0u\)\{return DryNodeMipLookup\(SvoNodeMipSample\(0\.0,0\.0,0\.0,0\.0\),1u\);\}/,
    "cached sparse-directory misses sample as transparent without another search");
  assert.match(lookup, /if\(!dryNodeMipPageValid\(\(\*pageCache\)\.pageIndex\)\)\{return DryNodeMipLookup\(SvoNodeMipSample\(0\.0,0\.0,0\.0,0\.0\),0u\);\}/,
    "a present but dirty page is invalid, never equivalent to empty space");
  assert.match(svoDrySceneShader, /var pageCache=DryNodeMipPageCache\([^]*dryNodeMipAt\([^]*&pageCache\)/);
  assert.match(svoDrySceneShader, /var black=2u;if\(dryTetraRadiancePageValid\(pageIndex\)\)\{black=textureLoad\(tetraRadianceBlackPages,vec2u\(pageIndex,0u\),0\)\.x;\}/);
  assert.match(svoDrySceneShader, /if\(dryGiPageCache\.blackRadiance==1u\)/);
  assert.match(svoDrySceneShader, /if\(dryGiPageCache\.blackRadiance==2u\)[^]*,1u,0u\);\}/,
    "dirty radiance pages retain current opacity but never sample stale radiance");
  assert.match(svoDrySceneShader, /if\(result\.valid==0u\|\|result\.missingRadianceSamples!=0u\)\{dryDerivedPageFailure\|=4u;return DryGlobalIllumination\(vec3f\(0\.0\),1\.0,0u\);\}/,
    "dirty opacity or radiance pages publish invalid GI without exact traversal");
  assert.match(svoDrySceneShader,
    /return SvoTetraRadianceConeSourceSample\(opacity\.solidMean,SvoTetraRadiance\(vec3f\(0\.0\),vec3f\(0\.0\),vec3f\(0\.0\),vec3f\(0\.0\)\),1u,1u\)/,
    "a certified-black page remains a valid sample while avoiding all four lobe fetches");

  const coherentSteps = Array.from({ length: 48 }, (_, step) => ({
    generation: 17,
    level: 0,
    coordinate: [Math.floor((0.75 + step) / 8), 0, 0] as const,
  }));
  let searches = 0;
  let previous: (typeof coherentSteps)[number] | undefined;
  for (const step of coherentSteps) {
    if (!previous || previous.generation !== step.generation || previous.level !== step.level
      || previous.coordinate.some((component, axis) => component !== step.coordinate[axis])) searches += 1;
    previous = step;
  }
  assert.equal(searches, 6, "an axis-aligned 48-step path searches once per crossed 8-voxel page");
  assert.equal(coherentSteps.length - searches, 42);
  assert.equal(1 - searches / coherentSteps.length, 0.875, "the representative path eliminates 87.5% of directory searches");

  const absentPageSteps = Array.from({ length: 12 }, () => ({ generation: 17, level: 0, coordinate: [9, 4, 2] as const }));
  searches = 0;
  let absentPrevious: (typeof absentPageSteps)[number] | undefined;
  for (const step of absentPageSteps) {
    if (!absentPrevious || absentPrevious.generation !== step.generation || absentPrevious.level !== step.level
      || absentPrevious.coordinate.some((component, axis) => component !== step.coordinate[axis])) searches += 1;
    absentPrevious = step;
  }
  assert.equal(searches, 1, "repeated samples in one non-resident sparse page reuse the negative lookup");
});

test("reduced split GI uses a bounded camera-independent world cache", () => {
  assert.deepEqual(SVO_DRY_WORLD_GI_CACHE_CONTRACT, {
    entryCount: 262_144,
    entryBytes: 16,
    probeBytes: 8,
    payloadBytes: 8,
    probeCount: 4,
    allocatedBytes: 4_194_304,
    frameBytes: 144,
    dynamicInfluenceCells: 12,
    dynamicInfluenceBodyRadii: 3,
  });
  const shader = createSvoDrySceneFragmentWGSL(0.5, "canonical-parametric", "off", "split");
  const keyStart = shader.indexOf("fn dryWorldGiKey(");
  const keyEnd = shader.indexOf("fn dryWorldGiFind(", keyStart);
  const key = shader.slice(keyStart, keyEnd);
  assert.ok(keyStart >= 0 && keyEnd > keyStart);
  assert.doesNotMatch(key, /cameraPosition|cameraTarget|viewport/,
    "camera motion changes queried world samples but must not invalidate their cached values");
  assert.match(key, /position-dry\.nodeMipOrigin/);
  assert.match(key, /dry\.nodeMip\.x/);
  assert.match(shader, /@compute @workgroup_size\(1\) fn dryWorldGiFrameMain/);
  assert.match(shader, /dryWorldGiFrame\.cameraForwardAspect=vec4f\(forward,/);
  assert.match(key, /dryWorldGiSpatialStart\(quantized\)/,
    "neighbouring world samples must retain a spatially coherent cache-set index");
  assert.doesNotMatch(key, /positionRadius|orientation/);
  assert.match(shader, /motion\.linearVelocityDisplacement\.w>1e-7\|\|motion\.angularVelocityAngle\.w>1e-7/);
  assert.match(shader, /fn dryWorldGiBodyInfluence\([^]*dot\(delta,delta\)<=influence\*influence/);
  assert.match(shader, /bodyMask\|=bodyBit[^]*signature=dryWorldGiHashAdd\(signature,dryWorldGiFrame\.bodySignatures\[bodyIndex\]\)/,
    "each receiver namespace must include only bodies in its conservative influence");
  assert.match(shader, /if\(influence\.movingMask!=0u\)\{[^]*dryWorldGiBodyMask=influence\.bodyMask/,
    "only a locally influential moving body may force an exact GI retrace");
  assert.match(shader, /let bodyAware=influence\.bodyMask!=0u;let bodyNamespace=select\(0x4f1bbcdcu,influence\.signature,bodyAware\)/,
    "unrelated body motion must not change a static neighbourhood's cache namespace");
  assert.match(shader, /dryWorldGiIgnoreRigidBodies=select\(1u,0u,bodyAware\);dryWorldGiBodyMask=influence\.bodyMask;dryPrepassGiState=0u;/,
    "only bounded body neighbourhoods specialize cached GI with rigid geometry");
  assert.match(shader, /nearestBodyMaskIgnoring\(origin,direction,ignoredBodyOwner,dryWorldGiBodyMask\)/,
    "cached GI cones must test only the bodies represented by their local cache key");
  assert.match(shader, /struct DryWorldGiCache\{[^]*metadata:array<DryWorldGiCacheMetadata,262144>[^]*payload:array<DryWorldGiCachePayload,262144>/,
    "miss probes must scan packed metadata without pulling the half-float payload plane");
  assert.match(shader, /atomicCompareExchangeWeak\(&dryWorldGiCache\.metadata\[slot\]\.state,claimState,1u\)/);
  assert.match(shader, /atomicStore\(&dryWorldGiCache\.metadata\[slot\]\.state,key\.readyState\)/,
    "the ready state must publish after the packed half-float payload");
  assert.match(shader, /if\(state!=1u\)\{claimSlot=slot;claimState=state;\}/,
    "a moving camera may replace an old ready entry but never a writer in progress");
  assert.match(shader, /@compute @workgroup_size\(8,8\) fn dryWorldGiCacheMain/);
});

test("derived-page failures expose a bounded renderer readback ABI", () => {
  assert.deepEqual(SVO_DRY_DERIVED_FAILURE_COUNTERS, {
    ambientOcclusionPageWord: 0,
    directVisibilityPageWord: 1,
    globalIlluminationPageWord: 2,
    wordCount: 3,
    sizeBytes: 12,
  });
  const shader = createSvoDrySceneFragmentWGSL(0.5, "canonical-parametric", "off", "split");
  assert.match(shader, /invalidAoPages:atomic<u32>,invalidDirectPages:atomic<u32>/);
  assert.match(shader, /atomicStore\(&dryPrepassBoundaryQueue\.invalidAoPages,0u\)/);
  assert.match(shader, /atomicAdd\(&dryPrepassBoundaryQueue\.invalidAoPages,1u\)/);
  assert.match(shader, /atomicAdd\(&dryPrepassBoundaryQueue\.invalidDirectPages,1u\)/);
  assert.match(shader, /invalidGiPages:atomic<u32>/);
  assert.match(shader, /atomicAdd\(&dryWorldGiFrame\.invalidGiPages,1u\)/);
  assert.match(drySource, /copyDerivedPageFailureCounters\([^]*copyBufferToBuffer\(this\.coneDerivedFailureSnapshot, 0[^]*copyBufferToBuffer\(this\.worldGiFrameBuffer, 16/);
});

test("analytic transform publications retain derived caches by exact dependency", () => {
  const dirtyBounds = [{ minimum: [-1, 0, -1] as const, maximum: [1, 2, 1] as const }];
  assert.deepEqual(svoDryPrimitiveArenaCacheInvalidation({ dirtyBounds, derivedLighting: "unchanged" }), {
    worldGi: false,
    directionalVisibility: false,
  });
  assert.deepEqual(svoDryPrimitiveArenaCacheInvalidation({ dirtyBounds, derivedLighting: "global" }), {
    worldGi: true,
    directionalVisibility: true,
  });
  assert.match(drySource, /this\.primitiveDirtyBounds = change\.dirtyBounds/,
    "arbitrary old/new bounds remain attached for the future sparse-page publication");
  assert.match(drySource, /if \(invalidation\.worldGi\) this\.worldGiCacheDirty = true/);
  assert.match(drySource, /if \(invalidation\.directionalVisibility\) this\.invalidateVoxelLightCache\(\)/);
});

test("node-mip sampling publishes its own world origin inside the live uniform block", () => {
  assert.equal(SVO_DRY_SCENE_PARAMS_LAYOUT.nodeMipOriginWordOffset, 60);
  // The static-lighting block still ends exactly where it always did. Evolving
  // fluid coverage is appended past it rather than repacking anything below, so
  // every offset a cone-lighting frame reads is unmoved.
  assert.equal((SVO_DRY_SCENE_PARAMS_LAYOUT.nodeMipOriginWordOffset + 4) * 4, 256);
  assert.equal(SVO_DRY_SCENE_PARAMS_LAYOUT.fluidCoverageWordOffset * 4, 256,
    "the fluid frame must start where the static block ends");
  assert.equal(SVO_DRY_SCENE_PARAMS_LAYOUT.tuningWordOffset,
    SVO_DRY_SCENE_PARAMS_LAYOUT.fluidCoverageWordOffset + SVO_FLUID_COVERAGE_LAYOUT.frameWords,
    "runtime tuning must immediately follow the 12-word fluid frame");
  assert.equal(SVO_DRY_SCENE_PARAMS_LAYOUT.rigidBoundsWordOffset,
    SVO_DRY_SCENE_PARAMS_LAYOUT.giConesWordOffset + 4,
    "the whole-scene rigid sphere must immediately follow the GI controls");
  assert.equal(SVO_DRY_SCENE_PARAMS_LAYOUT.sizeBytes,
    (SVO_DRY_SCENE_PARAMS_LAYOUT.derivedTraversalWordOffset + 4) * Uint32Array.BYTES_PER_ELEMENT,
    "the uniform allocation must end after the four-arena offset metadata");
  assert.match(drySource, /floats\.set\(nodeMip\?\.worldOrigin_m \?\? structural\.domain\.worldOrigin_m, SVO_DRY_SCENE_PARAMS_LAYOUT\.nodeMipOriginWordOffset\)/);
  assert.match(svoDrySceneShader, /virtualVoxel=\(position_m-dry\.nodeMipOrigin\.xyz\)/,
    "topology experiments must not reinterpret an unchanged opacity atlas in the structural tree's coordinate frame");
  assert.match(worldSource, /worldOrigin_m: this\.sceneWorldOrigin/);
  assert.match(worldSource, /worldExtent_m: sceneDomain\.sceneDimensionsCells\.map/);
  assert.match(svoDrySceneShader, /fn dryNodeMipSceneExitDistance\([^]*maximum=minimum\+dry\.nodeMipExtent\.xyz/,
    "GI cones must survey the surrounding authored scene, not stop at the solver tank");
});

test("sparse-brick world exposes, accounts, and retires its optional node-mip capability", () => {
  assert.match(sourceAbi, /nodeMipPyramid\?: import\("\.\/webgpu-svo-node-mip-pyramid"\)\.WebGpuSvoNodeMipVisibleGeneration/);
  assert.match(worldSource, /nodeMipPyramid: this\.nodeMipPyramid\?\.visibleGeneration\(\)/);
  assert.match(worldSource, /nodeMipPyramid: \(this\.nodeMipPyramid\?\.allocatedBytes \?\? 0\)/);
  assert.match(worldSource, /\+ \(this\.nodeMipPyramid\?\.allocatedBytes \?\? 0\)/);
  assert.match(worldSource, /this\.nodeMipPyramid\?\.destroy\(\)/);
  assert.match(sourceAbi, /tetrahedralRadiance\?: import\("\.\/webgpu-svo-tetrahedral-radiance"\)\.WebGpuSvoTetrahedralRadianceVisibleGeneration/);
  assert.match(worldSource, /tetrahedralRadiance: this\.tetrahedralRadiance\?\.visibleGeneration\(\)/);
  assert.match(worldSource, /WebGpuLiveSvoNodeMipPyramid/);
  assert.match(worldSource, /WebGpuLiveSvoTetrahedralRadiance/);
});

const modulePath = process.env.WEBGPU_NODE_MODULE;
test("production dry shader compiles sampled node-mip atlas and uint directory bindings", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU cone-lighting checks",
}, async () => {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]), adapter = await gpu.requestAdapter({ powerPreference: "high-performance" }); assert.ok(adapter);
  const device = await adapter.requestDevice();
  try {
    const module = device.createShaderModule({ label: "Cone-lighting dry shader validation", code: svoDrySceneShader });
    const info = await module.getCompilationInfo();
    assert.deepEqual(info.messages.filter(({ type }) => type === "error"), []);
    const cacheModule = device.createShaderModule({
      label: "Persistent world GI cache validation",
      code: createSvoDrySceneFragmentWGSL(0.5, "canonical-parametric", "off", "split"),
    });
    const cacheInfo = await cacheModule.getCompilationInfo();
    assert.deepEqual(cacheInfo.messages.filter(({ type }) => type === "error"), []);
    assert.match(svoDrySceneShader, /@binding\(16\) var nodeMipAtlas:texture_3d<f32>/);
    assert.match(svoDrySceneShader, /@binding\(18\) var nodeMipDirectory:texture_2d<u32>/);
    assert.match(svoDrySceneShader, /@binding\(20\) var nodeMipPageTable:texture_3d<u32>/);
    assert.match(svoDrySceneShader, /@binding\(26\) var nodeMipPageValidity:texture_2d<u32>/);
    assert.match(svoDrySceneShader, /@binding\(27\) var tetraRadiancePageValidity:texture_2d<u32>/);
  } finally { device.destroy(); }
});
