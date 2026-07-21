import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";

import { DEFAULT_SVO_LIGHTING_MODE, DEFAULT_SVO_LIGHTING_OPTIONS, DEFAULT_SVO_RENDER_MODE } from "../lib/svo-render-mode";
import {
  canConsumeSparseVoxelPrimitiveCandidates,
  canEncodeSparseVoxelDryScene,
  SVO_DRY_SCENE_BINDING_CONTRACT,
  svoDrySceneShader,
  svoDrySceneVertexShader,
  type SparseVoxelDrySceneData,
} from "../lib/webgpu-svo-dry-scene";
import type { SparseVoxelRenderSource } from "../lib/webgpu-voxel-debug";
import type { DrySceneReplacementEncoder } from "../lib/webgpu-water-pipeline";
import { candidateBackedDrySceneFixture } from "./svo-dry-scene-test-fixture";

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

test("SVO presentation is the WebGPU default while raster remains selectable", () => {
  assert.equal(DEFAULT_SVO_RENDER_MODE, "svo");
  assert.equal(DEFAULT_SVO_LIGHTING_MODE, "cone");
  assert.deepEqual(DEFAULT_SVO_LIGHTING_OPTIONS, { shadowsEnabled: true, ambientOcclusionEnabled: true });
  expectSource(rendererSource, /svoRenderMode: SvoRenderMode = DEFAULT_SVO_RENDER_MODE/,
    "callers which do not override presentation must use sparse voxels");
  expectSource(rendererSource, /DEFAULT_SVO_LIGHTING_MODE[^]*DEFAULT_SVO_RENDER_MODE[^]*type SvoLightingMode[^]*type SvoRenderMode[^]*from "\.\/svo-render-mode"/,
    "renderer must consume the canonical render and lighting toggles");
  expectSource(viewportSource, /ui\.svoLightingMode,[^]*shadowsEnabled: ui\.svoShadowsEnabled,[^]*ambientOcclusionEnabled: ui\.svoAmbientOcclusionEnabled,[^]*overlay: ui\.svoCostOverlay/,
    "viewport must pass lighting effects before the diagnostics argument in the renderer contract");
});

test("the water pipeline replacement callback is fail-safe and replaces rather than overlays raster", () => {
  const sampledTargetView = {} as GPUTextureView;
  const replacement: DrySceneReplacementEncoder = () => ({ encoded: true, sampledTargetView });
  assert.deepEqual(replacement({} as GPUCommandEncoder, {} as GPUTexture), { encoded: true, sampledTargetView });
  expectSource(waterSource, /drySceneReplacement\?\.\(encoder, this\.sceneTexture, timestamps\?\.scene\) \?\? false/,
    "water pipeline must let the replacement explicitly accept or reject a frame");
  expectSource(waterSource, /if \(!sparseSceneResult\) \{[^]*label:"Dry scene"/,
    "the analytic pass is encoded only when no replacement accepted the frame");
  assert.doesNotMatch(waterSource, /drySceneOverlay/,
    "a sparse production scene must never be composed after the analytic pass");
  const replacementCall = waterSource.indexOf("const sparseSceneResult = drySceneReplacement");
  const rasterFallback = waterSource.indexOf("if (!sparseSceneResult)", replacementCall);
  assert.ok(replacementCall >= 0 && rasterFallback > replacementCall);
});

test("the direct renderer exposes a source-aware replacement texture contract", () => {
  assert.ok(drySceneSource, "lib/webgpu-svo-dry-scene.ts must implement the production dry-scene renderer");
  expectSource(drySceneSource, /export class SparseVoxelDrySceneRenderer/,
    "direct renderer class must be public to the presentation owner");
  expectSource(drySceneSource, /setSource\(source: SparseVoxelSceneRenderSource \| undefined, scene: SparseVoxelDrySceneData \| undefined\)/,
    "the renderer accepts structural arenas, their parent material table, and analytic-owner data together");
  expectSource(drySceneSource, /encode\([^)]*encoder: GPUCommandEncoder[^)]*target: GPUTexture \| GPUTextureView[^)]*timestampWrites\?: TimestampRange[^)]*\): DrySceneReplacementResult \| false/,
    "encode must report both successful ownership and the texture the next stage should sample");
  expectSource(drySceneSource, /if \(!this\.pipeline \|\| !this\.bindGroup\) return false/,
    "an absent or unpublished source must trigger the raster fallback");
  expectSource(drySceneSource, /loadOp:\s*"clear"/,
    "a successful replacement owns the complete dry-scene target");
  assert.doesNotMatch(drySceneSource, /SparseVoxelDebugRecord|voxelRecords|brickRecords/,
    "production traversal must not expand or consume debug cube records");
  const primitiveHitStart = drySceneSource.indexOf("fn primitiveHit(");
  const primitiveHitEnd = drySceneSource.indexOf("const DRY_CANDIDATE_COMPLETE", primitiveHitStart);
  const primitiveHit = drySceneSource.slice(primitiveHitStart, primitiveHitEnd);
  assert.match(primitiveHit, /svoIntersectPrimitiveExact\(record,ro,rd,max\(tMin,1e-4\),tMax\)/,
    "candidate leaves must use the shared five-kind analytic ray contract");
  assert.doesNotMatch(primitiveHit, /svoEvaluatePrimitive|svoPrimitiveDistance_m|svoEllipsoidClosestPoint_m/,
    "ray hits must not run the bounded ellipsoid closest-point distance solve to recover a normal");
  assert.match(drySceneSource, /fn nearestBodyIgnoring\([^]*bodyBoundingSphereVisible\(ro,rd,body,0\.0,best\.t\)/,
    "primary rays must reject distant dynamic bodies in world space before exact local intersection");
  for (const binding of ["structural.control", "structural.nodes", "structural.leaves", "structural.materialOwners", "structural.publication.state", "source.pbrMaterials!.binding"]) {
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
  const declarations = [...svoDrySceneShader.matchAll(/@group\(0\)\s+@binding\((\d+)\)\s+var(?:(?:<(uniform|storage,\s*read)>)|\s+[^:]+:\s*(texture_3d<f32>|texture_2d<u32>|sampler))/g)]
    .map((match) => ({
      binding: Number(match[1]),
      type: match[2] === "uniform" ? "uniform" : match[2] ? "read-only-storage"
        : match[3] === "texture_3d<f32>" ? "texture-3d-float"
        : match[3] === "texture_2d<u32>" ? "texture-2d-uint" : "filtering-sampler",
    }))
    .sort((a, b) => a.binding - b.binding);
  assert.deepEqual(declarations, [...SVO_DRY_SCENE_BINDING_CONTRACT],
    "the production layout contract must enumerate every shader declaration, including optional uniform binders");
  assert.equal(new Set(declarations.map(({ binding }) => binding)).size, declarations.length, "shader bindings must be unique");

  const rebuildStart = drySceneSource.indexOf("this.bindGroup = this.device.createBindGroup");
  const rebuildEnd = drySceneSource.indexOf("]);", rebuildStart);
  const resources = [...drySceneSource.slice(rebuildStart, rebuildEnd).matchAll(/\{ binding: (\d+), resource:/g)]
    .map((match) => Number(match[1])).sort((a, b) => a - b);
  assert.deepEqual(resources, SVO_DRY_SCENE_BINDING_CONTRACT.map(({ binding }) => binding),
    "every declared/layout binding must have a resource in the sole production bind-group variant");
  assert.equal(SVO_DRY_SCENE_BINDING_CONTRACT.filter(({ type }) => type === "read-only-storage").length, 10,
    "the dry pass includes the two optional wide-fanout traversal payloads");
  assert.deepEqual(SVO_DRY_SCENE_BINDING_CONTRACT.filter(({ binding }) => binding === 11 || binding === 12), [
    { binding: 11, type: "read-only-storage" }, { binding: 12, type: "read-only-storage" },
  ]);
  assert.deepEqual(SVO_DRY_SCENE_BINDING_CONTRACT.slice(-3).map(({ binding, type }) => [binding, type]), [
    [16, "texture-3d-float"], [17, "filtering-sampler"], [18, "texture-2d-uint"],
  ], "cone lighting must consume sampled resources rather than another fragment storage buffer");
  assert.match(drySceneSource, /nodeMip\?\.view \?\? this\.nodeMipFallbackAtlasView/);
  assert.match(drySceneSource, /nodeMip\?\.sampler \?\? this\.nodeMipFallbackSampler/);
  assert.match(drySceneSource, /nodeMip\?\.directoryView \?\? this\.nodeMipFallbackDirectoryView/);
});

test("unavailable structural fields fail over to raster before GPU encoding", () => {
  const source = {
    materialCount: 2,
    pbrMaterials: { binding: { buffer: {} as GPUBuffer }, count: 8, strideBytes: 96, revision: 1 },
    structural: {
      fields: {
        topology: { residency: "all-published-leaves" },
        staticGeometry: { residency: "all-published-leaves" },
        materialOwner: { residency: "all-published-leaves" },
      },
    },
  } as unknown as SparseVoxelRenderSource;
  const scene: SparseVoxelDrySceneData = { ...candidateBackedDrySceneFixture, ownerBase: 32 };
  assert.equal(canEncodeSparseVoxelDryScene(undefined, scene), false);
  assert.equal(canEncodeSparseVoxelDryScene(source, undefined), false);
  assert.equal(canEncodeSparseVoxelDryScene(source, { ...scene, primitiveRecords: new Uint32Array(0) }), false);
  assert.equal(canEncodeSparseVoxelDryScene(source, scene), true);
  assert.equal(canConsumeSparseVoxelPrimitiveCandidates(scene), true);
  assert.equal(canEncodeSparseVoxelDryScene(source, { ...scene, primitiveCandidates: undefined }), false,
    "small analytic catalogs must fail over when their conservative candidate publication is unavailable");
  assert.equal(canConsumeSparseVoxelPrimitiveCandidates({ ...scene, primitiveCandidates: undefined }), false);
  const unavailable = {
    ...source,
    structural: {
      ...source.structural!,
      fields: {
        ...source.structural!.fields,
        staticGeometry: { ...source.structural!.fields.staticGeometry, residency: "unavailable" as const },
      },
    },
  };
  assert.equal(canEncodeSparseVoxelDryScene(unavailable, scene), false);
});

test("renderer atomically replaces structural sources before retiring the previous solver", () => {
  expectSource(rendererSource, /private svoDryScenePipeline\?: SparseVoxelDrySceneRenderer/,
    "FluidLabRenderer must own the direct renderer lifecycle");
  expectSource(rendererSource, /this\.svoDryScenePipeline\?\.setSource\(sparseSceneSource,drySceneData\)/,
    "solver attachment must pass the complete source and its analytic-owner data");

  const attach = rendererSource.indexOf("this.svoDryScenePipeline?.setSource(sparseSceneSource,drySceneData)");
  const retire = rendererSource.indexOf("this.retireGPUFluid(previous)", attach);
  assert.ok(attach >= 0, "the warmed structural source must replace the active binding");
  assert.ok(retire > attach, "the new binding must be installed before the previous solver is retired");
  expectSource(rendererSource, /this\.svoDryScenePipeline\?\.destroy\(\)/,
    "renderer teardown must destroy direct-renderer-owned GPU resources");
});

test("SVO is offered to the water pipeline only for smooth production presentation", () => {
  expectSource(rendererSource, /svoRenderMode === "svo" && voxelRenderMode === "smooth"/,
    "inspection modes and the production renderer are separate switches");
  expectSource(rendererSource, /this\.svoDryScenePipeline\?\.encode\(/,
    "SVO mode must offer a replacement encoder");
  expectSource(rendererSource, /this\.waterPipeline\.encode\([^]*drySceneReplacement/s,
    "the replacement callback must target the water pipeline's internal HDR dry-scene attachment");
});

test("raw voxels and brick-grid remain independent clear-and-replace inspection modes", () => {
  assert.match(rendererSource, /this\.voxelInspectionSource = requestedVoxelDebugGeneration >= 0 \? this\.gpuFluid\?\.sparseVoxelRenderSource : undefined/,
    "debug mode materialization must remain gated by inspection visibility");
  assert.match(rendererSource, /this\.voxelDebugPipeline\?\.setSource\(this\.voxelInspectionSource\)/,
    "debug modes consume expanded records only while inspection is visible");
  assert.match(rendererSource, /if \(voxelRenderMode !== "smooth" && this\.voxelDebugDepth\)/);
  assert.match(rendererSource, /mode: voxelRenderMode/);
  assert.match(rendererSource, /colorLoadOp: "clear"/,
    "inspection remains a complete representation switch, not an SVO dry-scene overlay");
});
