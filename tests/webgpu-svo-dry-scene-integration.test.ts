import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";

import { DEFAULT_SVO_LIGHTING_OPTIONS } from "../lib/svo-render-options";
import {
  canEncodeSparseVoxelDryScene,
  SVO_DRY_SCENE_BINDING_CONTRACT,
  svoDrySceneShader,
  svoDrySceneVertexShader,
  type SparseVoxelDrySceneData,
} from "../lib/webgpu-svo-dry-scene";
import type { SparseVoxelRenderSource } from "../lib/webgpu-voxel-debug";
import type { DrySceneReplacementEncoder } from "../lib/webgpu-water-pipeline";
import { svoDrySceneFixture } from "./svo-dry-scene-test-fixture";

const rendererUrl = new URL("../lib/webgpu-renderer.ts", import.meta.url);
const waterUrl = new URL("../lib/webgpu-water-pipeline.ts", import.meta.url);
const drySceneUrl = new URL("../lib/webgpu-svo-dry-scene.ts", import.meta.url);
const viewportUrl = new URL("../components/WebGPUViewport.tsx", import.meta.url);
const rendererSource = readFileSync(rendererUrl, "utf8");
const waterSource = readFileSync(waterUrl, "utf8");
const drySceneSource = existsSync(drySceneUrl) ? readFileSync(drySceneUrl, "utf8") : "";
const viewportSource = readFileSync(viewportUrl, "utf8");

function expectSource(source: string, pattern: RegExp, message: string): void {
  assert.ok(pattern.test(source), message);
}

test("GLOBAL SVO is the sole production presentation", () => {
  assert.deepEqual(DEFAULT_SVO_LIGHTING_OPTIONS,
    { shadowsEnabled: true, ambientOcclusionEnabled: true, coneTracingMode: "cones", primaryTraversal: "raster" });
  assert.doesNotMatch(rendererSource, /svoRenderMode|svoLightingMode|SvoRenderMode|SvoLightingMode/);
  expectSource(rendererSource, /type SvoLightingOptions[^]*from "\.\/svo-render-options"/,
    "renderer must retain only GLOBAL visibility effects");
  expectSource(viewportSource, /ui\.voxelRenderMode,[^]*shadowsEnabled: ui\.svoShadowsEnabled,[^]*ambientOcclusionEnabled: ui\.svoAmbientOcclusionEnabled,[^]*stageView: ui\.svoStageView/,
    "viewport must pass lighting effects before the diagnostics argument in the renderer contract");
});

test("the water pipeline replacement callback owns the dry scene without a substitute path", () => {
  const sampledTargetView = {} as GPUTextureView;
  const replacement: DrySceneReplacementEncoder = () => ({ encoded: true, sampledTargetView });
  assert.deepEqual(replacement({} as GPUCommandEncoder, {} as GPUTexture), { encoded: true, sampledTargetView });
  expectSource(waterSource, /drySceneReplacement\?\.\(encoder, this\.sceneTexture, tracePhase\) \?\? false/,
    "water pipeline must let the replacement explicitly accept or reject a frame");
  expectSource(waterSource, /if \(!sparseSceneResult\) \{[^]*label:"SVO dry-scene unavailable"/,
    "a rejected live SVO frame must enter the explicit fail-closed pass");
  assert.doesNotMatch(waterSource, /drySceneOverlay/,
    "a sparse production scene must never be composed after the analytic pass");
  assert.doesNotMatch(waterSource, /scenePipeline|sceneBindGroup|sceneShader|Render dry scene for water refraction/);
});

test("missing live SVO publication fails closed and retains only real fluid interfaces", () => {
  const failure = waterSource.slice(
    waterSource.indexOf("if (!sparseSceneResult)"),
    waterSource.indexOf("traceBoundary?.();", waterSource.indexOf("if (!sparseSceneResult)")),
  );
  expectSource(failure, /clearValue:\{r:\.18,g:0,b:\.045,a:65504\}/,
    "the failure plane must be conspicuous and remain at far depth for water diagnosis");
  expectSource(failure, /SVO dry-scene unavailable · fail closed/,
    "performance diagnostics must name the rejection rather than a fallback");
  assert.doesNotMatch(rendererSource, /setPendingSvoBackground|svoPresentationExpected/);
  expectSource(waterSource, /if \(this\.sceneHasFluid\) \{[^]*interfacePass\("Water \+ spray front interfaces"/,
    "actual fluid interfaces remain available after the dry scene fails closed");
});

test("the direct renderer exposes a source-aware replacement texture contract", () => {
  assert.ok(drySceneSource, "lib/webgpu-svo-dry-scene.ts must implement the production dry-scene renderer");
  expectSource(drySceneSource, /export class SparseVoxelDrySceneRenderer/,
    "direct renderer class must be public to the presentation owner");
  expectSource(drySceneSource, /setSource\(source: SparseVoxelSceneRenderSource \| undefined\)/,
    "the renderer attaches the mutable SVO acceleration independently");
  expectSource(drySceneSource, /publishScene\(scene: SparseVoxelDrySceneData\): boolean/,
    "the renderer hot-publishes live analytic, material, glass, and lighting data");
  expectSource(drySceneSource, /encode\([^)]*encoder: GPUCommandEncoder[^)]*target: GPUTexture \| GPUTextureView[^)]*\): DrySceneReplacementResult \| false/,
    "encode must report both successful ownership and the texture the next stage should sample");
  expectSource(drySceneSource, /if \(!this\.pipeline \|\| !this\.bindGroup\) return false/,
    "an absent or unpublished source must reject the frame for fail-closed presentation");
  expectSource(drySceneSource, /loadOp:\s*"clear"/,
    "a successful replacement owns the complete dry-scene target");
  assert.doesNotMatch(drySceneSource, /SparseVoxelDebugRecord|voxelRecords|brickRecords/,
    "production traversal must not expand or consume debug cube records");
  const primitiveHitStart = drySceneSource.indexOf("fn primitiveHit(");
  const primitiveHitEnd = drySceneSource.indexOf("fn traceLeafPayload(", primitiveHitStart);
  const primitiveHit = drySceneSource.slice(primitiveHitStart, primitiveHitEnd);
  assert.match(primitiveHit, /svoIntersectPrimitiveExact\(record,ro,rd,max\(tMin,1e-4\),tMax\)/,
    "occupied SVO payload cells must use the shared five-kind analytic ray contract");
  assert.doesNotMatch(primitiveHit, /svoEvaluatePrimitive|svoPrimitiveDistance_m|svoEllipsoidClosestPoint_m/,
    "ray hits must not run the bounded ellipsoid closest-point distance solve to recover a normal");
  assert.match(drySceneSource, /fn nearestBodyMaskIgnoring\([^]*bodyBoundingSphereVisible\(ro,rd,body,0\.0,best\.t\)/,
    "primary rays must reject distant dynamic bodies in world space before exact local intersection");
  assert.match(drySceneSource,
    /fn dryGlobalIllumination\(position:vec3f,normal:vec3f,ignoredBodyOwner:u32\)[^]*if\(dryWorldGiIgnoreRigidBodies==0u\)\{rigidHit=nearestBodyMaskIgnoring\(origin,direction,ignoredBodyOwner,dryWorldGiBodyMask\);\}[^]*min\(sceneExit,rigidHit\.t\)[^]*select\(visibleThroughStatic,0\.0,rigidBlocked\)/,
    "environmental GI must clip static radiance cones against the live rigid overlay in the current frame");
  assert.match(drySceneSource,
    /if\(influence\.movingMask!=0u\)[^]*dryWorldGiIgnoreRigidBodies=0u[^]*dryGlobalIllumination\(position,opaque\.normal,ignoredBodyOwner\)/,
    "moving rigid bodies must force exact body-aware GI only inside their conservative local influence");
  assert.match(drySceneSource,
    /ignoredBodyOwner=select\(DRY_OWNER_NONE,hit\.ownerId,hit\.motionKind==DRY_GBUFFER_MOTION_RIGID\)/,
    "a rigid GI receiver must ignore only itself while static receivers test every live body");
  assert.match(drySceneSource,
    /rootIndex=0u;rootIndex<2u[^]*candidate=\(-b\+select\(-root,root,rootIndex!=0u\)\)\/a/,
    "capsule and cylinder side intersections must retain the positive exit root for inside rays");
  for (const binding of ["structural.structure", "structural.sceneMaterialOwners", "this.sceneArenaBuffer"]) {
    assert.ok(drySceneSource.includes(binding), `direct rendering must bind ${binding}`);
  }
});

test("the fullscreen vertex stage compiles from a small module isolated from the dry fragment graph", () => {
  assert.ok(svoDrySceneVertexShader.length < 1_024);
  assert.match(svoDrySceneVertexShader, /@vertex fn vertexMain/);
  assert.doesNotMatch(svoDrySceneVertexShader, /@fragment|var<storage|svoTraverse/);
  assert.doesNotMatch(svoDrySceneShader, /@vertex fn vertexMain/);
  assert.match(svoDrySceneShader, /@fragment fn fragmentMain/);
  expectSource(drySceneSource, /const \[vertexModule, fragmentModule\] = await Promise\.all/,
    "the Metal vertex compiler must not receive the monolithic dry fragment module");
  expectSource(drySceneSource, /vertex: \{ module: vertexModule[^]*fragment: \{ module: fragmentModule/,
    "the render pipeline must preserve distinct stage modules");
});

test("every dry-shader group-zero declaration has one layout and bind-group entry", () => {
  const declarations = [...svoDrySceneShader.matchAll(/@group\(0\)\s+@binding\((\d+)\)\s+var(?:(?:<(uniform|storage,\s*read)>)|\s+[^:]+:\s*(texture_3d<f32>|texture_3d<u32>|texture_2d<u32>|sampler))/g)]
    .map((match) => ({
      binding: Number(match[1]),
      type: match[2] === "uniform" ? "uniform" : match[2] ? "read-only-storage"
        : match[3] === "texture_3d<f32>" ? "texture-3d-float"
        : match[3] === "texture_3d<u32>" ? "texture-3d-uint"
        : match[3] === "texture_2d<u32>" ? "texture-2d-uint" : "filtering-sampler",
    }))
    .sort((a, b) => a.binding - b.binding);
  assert.deepEqual(declarations, [...SVO_DRY_SCENE_BINDING_CONTRACT],
    "the production layout contract must enumerate every shader declaration, including optional uniform binders");
  assert.equal(new Set(declarations.map(({ binding }) => binding)).size, declarations.length, "shader bindings must be unique");

  const rebuildStart = drySceneSource.indexOf("this.bindGroup = this.device.createBindGroup");
  // Bound the scan by this call's own terminator. Reaching for the next "]);"
  // anywhere in the file silently swallowed later bind groups — a diagnostic
  // one, for instance — and reported their bindings as production duplicates.
  const rebuildEnd = drySceneSource.indexOf("] });", rebuildStart);
  assert.ok(rebuildStart >= 0 && rebuildEnd > rebuildStart, "the production bind group must be built in one call");
  const resources = [...drySceneSource.slice(rebuildStart, rebuildEnd).matchAll(/\{ binding: (\d+), resource:/g)]
    .map((match) => Number(match[1])).sort((a, b) => a - b);
  assert.deepEqual(resources, SVO_DRY_SCENE_BINDING_CONTRACT.map(({ binding }) => binding),
    "every declared/layout binding must have one production resource expression");
  assert.match(drySceneSource, /\.\.\.\(derivedTraversal \? \[\{ binding: 5, resource: derivedTraversal \}\] : \[\]\)/,
    "derived traversal is bound only for entrypoints that actually consume it");
  assert.equal(SVO_DRY_SCENE_BINDING_CONTRACT.filter(({ type }) => type === "read-only-storage").length, 4,
    "the dry pass has a hard four-storage-buffer ceiling");
  assert.deepEqual(SVO_DRY_SCENE_BINDING_CONTRACT.filter(({ binding }) => binding === 11 || binding === 12), []);
  assert.deepEqual(SVO_DRY_SCENE_BINDING_CONTRACT.slice(-12).map(({ binding, type }) => [binding, type]), [
    [16, "texture-3d-float"], [17, "filtering-sampler"], [18, "texture-2d-uint"],
    // Evolving fluid coverage shares the node-mip sampler rather than adding a
    // second one: both want clamp-to-edge linear filtering.
    [19, "texture-3d-float"],
    [20, "texture-3d-uint"],
    [21, "texture-3d-float"], [22, "texture-3d-float"],
    [23, "texture-3d-float"], [24, "texture-3d-float"],
    [25, "texture-2d-uint"],
    [26, "texture-2d-uint"], [27, "texture-2d-uint"],
  ], "cone lighting must consume sampled resources rather than another fragment storage buffer");
  assert.match(drySceneSource, /nodeMip\?\.view \?\? this\.nodeMipFallbackAtlasView/);
  assert.match(drySceneSource, /nodeMip\?\.sampler \?\? this\.nodeMipFallbackSampler/);
  assert.match(drySceneSource, /nodeMip\?\.directoryView \?\? this\.nodeMipFallbackDirectoryView/);
  assert.match(drySceneSource, /nodeMip\?\.directPageTableView \?\? this\.nodeMipFallbackDirectPageTableView/);
});

test("unavailable structural fields reject live SVO before GPU encoding", () => {
  const source = {
    materialCount: 2,
    pbrMaterials: { binding: { buffer: {} as GPUBuffer }, count: 8, strideBytes: 96, revision: 1 },
    structural: {
      fields: {
        topology: { residency: "all-published-leaves" },
        sceneGeometry: { residency: "all-published-leaves" },
        materialOwner: { residency: "all-published-leaves" },
      },
    },
  } as unknown as SparseVoxelRenderSource;
  const scene: SparseVoxelDrySceneData = { ...svoDrySceneFixture, ownerBase: 32 };
  assert.equal(canEncodeSparseVoxelDryScene(undefined, scene), false);
  assert.equal(canEncodeSparseVoxelDryScene(source, undefined), false);
  assert.equal(canEncodeSparseVoxelDryScene(source, { ...scene, primitiveRecords: new Uint32Array(0) }), false);
  assert.equal(canEncodeSparseVoxelDryScene(source, scene), true);
  const unavailable = {
    ...source,
    structural: {
      ...source.structural!,
      fields: {
        ...source.structural!.fields,
        sceneGeometry: { ...source.structural!.fields.sceneGeometry, residency: "unavailable" as const },
      },
    },
  };
  assert.equal(canEncodeSparseVoxelDryScene(unavailable, scene), false);
});

test("renderer atomically replaces structural sources before retiring the previous solver", () => {
  expectSource(rendererSource, /private svoDryScenePipeline\?: SparseVoxelDrySceneRenderer/,
    "FluidLabRenderer must own the direct renderer lifecycle");
  expectSource(rendererSource, /this\.svoDryScenePipeline\?\.setSource\(sparseSceneSource\)/,
    "solver attachment must replace only the mutable acceleration source");
  expectSource(rendererSource, /this\.svoDryScenePipeline && !this\.svoDryScenePipeline\.publishScene\(publication\)/,
    "presentation scene publication must remain independent of solver attachment");

  const attach = rendererSource.indexOf("this.svoDryScenePipeline?.setSource(sparseSceneSource)");
  const retire = rendererSource.indexOf("this.retireGPUFluid(previous)", attach);
  assert.ok(attach >= 0, "the warmed structural source must replace the active binding");
  assert.ok(retire > attach, "the new binding must be installed before the previous solver is retired");
  expectSource(rendererSource, /this\.svoDryScenePipeline\?\.destroy\(\)/,
    "renderer teardown must destroy direct-renderer-owned GPU resources");
});

test("SVO is offered to the water pipeline beneath every structural view", () => {
  expectSource(rendererSource, /const drySceneReplacement = \(/,
    "structural views must preserve GLOBAL as their underlying presentation");
  expectSource(rendererSource, /this\.svoDryScenePipeline\?\.encode\(/,
    "SVO mode must offer a replacement encoder");
  expectSource(rendererSource, /this\.waterPipeline\.encode\([^]*drySceneReplacement/s,
    "the replacement callback must target the water pipeline's internal HDR dry-scene attachment");
});

test("raw voxels and brick-grid are overlays on the GLOBAL frame", () => {
  assert.match(rendererSource, /this\.voxelInspectionSource = requestedVoxelDebugGeneration >= 0 \? this\.gpuFluid\?\.sparseVoxelRenderSource : undefined/,
    "debug mode materialization must remain gated by inspection visibility");
  assert.match(rendererSource, /this\.voxelDebugPipeline\?\.setSource\(this\.voxelInspectionSource\)/,
    "debug modes consume expanded records only while inspection is visible");
  assert.match(rendererSource, /if \(voxelRenderMode !== "smooth" && this\.voxelDebugDepth\)/);
  assert.match(rendererSource, /mode: voxelRenderMode/);
  assert.match(rendererSource, /Structural views diagnose the same GLOBAL frame[^]*colorLoadOp: "load"/,
    "inspection must alpha-blend over the GLOBAL dry scene");
});
