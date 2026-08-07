import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { SVO_MATERIAL_RECORD_STRIDE_BYTES } from "../lib/svo-material-abi";
import {
  canEncodeSparseVoxelDryScene,
  SparseVoxelDrySceneRenderer,
  SVO_DRY_SCENE_PARAMS_LAYOUT,
  svoDrySceneShader,
  type SparseVoxelDrySceneData,
} from "../lib/webgpu-svo-dry-scene";
import type { SparseVoxelSceneRenderSource } from "../lib/webgpu-voxel-debug";
import { svoDrySceneFixture } from "./svo-dry-scene-test-fixture";

const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
const dryRendererSource = readFileSync(new URL("../lib/webgpu-svo-dry-scene.ts", import.meta.url), "utf8");
const panelSource = readFileSync(new URL("../components/VisualPanel.tsx", import.meta.url), "utf8");

function structuralSource(
  pbrBinding: GPUBufferBinding = { buffer: {} as GPUBuffer, size: 8 * SVO_MATERIAL_RECORD_STRIDE_BYTES },
): SparseVoxelSceneRenderSource {
  const resource = { buffer: {} as GPUBuffer };
  return {
    materialCount: 8,
    pbrMaterials: { binding: pbrBinding, count: 8, strideBytes: SVO_MATERIAL_RECORD_STRIDE_BYTES, revision: 3 },
    structural: {
      structure: resource,
      structureOffsetsWords: { control: 0, publication: 64, nodes: 128, leaves: 640 },
      control: resource, nodes: resource, leaves: resource, geometry: resource,
      sceneGeometry: resource, velocity: resource, materialOwners: resource, scenePayload: resource, scenePayloadLanes: { mode: "dense" as const, materialOwnerWords: 0, occupancyWords: 0, recordMaskWords: 0, headerWords: 0, blobWords: 0, recordWords: 0 }, fluidLeafStates: resource,
      publication: { state: resource, byteLength: 32 },
      domain: { worldOrigin_m: [-2, 0, -2], cellSize_m: [0.04, 0.04, 0.04], dimensionsCells: [64, 64, 64], brickSize: 16, maximumDepth: 4 },
      capacities: { nodes: 64, leaves: 32, geometryVoxels: 1024, velocityVoxels: 1024, materialOwnerVoxels: 1024, fluidLeafStates: 32 },
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
  } as unknown as SparseVoxelSceneRenderSource;
}

const scene: SparseVoxelDrySceneData = svoDrySceneFixture;

test("live material publication is independent of producer material bindings", () => {
  const valid = structuralSource();
  assert.equal(canEncodeSparseVoxelDryScene(valid, scene), true);
  assert.equal(canEncodeSparseVoxelDryScene({ ...valid, pbrMaterials: undefined }, scene), true);
  assert.equal(canEncodeSparseVoxelDryScene(valid, { ...scene, materialRecords: new Uint32Array(1) }), false);
  assert.equal(canEncodeSparseVoxelDryScene(valid, { ...scene, materialRevision: 0 }), false);
});

test("binding 4 consumes the fixed renderer-owned authored scene arena", () => {
  assert.match(svoDrySceneShader, /@group\(0\) @binding\(4\) var<storage,read> drySceneArena:array<u32>/);
  assert.match(dryRendererSource, /\{ binding: 4, resource: \{ buffer: this\.sceneArenaBuffer \} \}/);
  assert.doesNotMatch(dryRendererSource, /resource: source\.materials/);
  assert.doesNotMatch(dryRendererSource, /@group\(0\) @binding\(10\)/);
  assert.doesNotMatch(dryRendererSource, /svoStructuralGeometry|svoStructuralLeafStates/,
    "the dry pass must not retain structural fluid-march payload bindings");
  assert.doesNotMatch(readFileSync(new URL("../lib/webgpu-voxel-debug.ts", import.meta.url), "utf8"), /materials: GPUBufferBinding/,
    "the inspection material table went with the expanded-record renderer");
});

test("published count, revision, direct identity, flags, and material functions are enforced in WGSL", () => {
  assert.deepEqual(SVO_DRY_SCENE_PARAMS_LAYOUT, {
    sizeBytes: 624, terrainWordOffset: 24, terrainMaterialWordOffset: 28, materialPublicationWordOffset: 32,
    nodeMipWordOffset: 36, nodeMipAtlasWordOffset: 40,
    wideFanoutWordOffset: 44, nodeMipLevelStartWordOffset: 48,
    nodeMipOriginWordOffset: 60, fluidCoverageWordOffset: 64, tuningWordOffset: 76,
    nodeMipDirectWordOffset: 96, nodeMipDirectLevelZWordOffset: 100, tetrahedralRadianceWordOffset: 112, nodeMipExtentWordOffset: 116,
    giLightingWordOffset: 120, giConesWordOffset: 124, rigidBoundsWordOffset: 128, primitiveCandidatesWordOffset: 132,
    structureOffsetsWordOffset: 136, derivedTraversalWordOffset: 140, lodWordOffset: 144,
    payloadLaneWordOffset: 148, payloadLane1WordOffset: 152,
  });
  assert.match(dryRendererSource, /const visibilityFlags = \(!giReady && ambientOcclusionEnabled \? SVO_DRY_VISIBILITY_FLAGS\.exactContact \| SVO_DRY_VISIBILITY_FLAGS\.ambientOcclusion : 0\)[^]*SVO_DRY_VISIBILITY_FLAGS\.exactShadow[^]*SVO_DRY_VISIBILITY_FLAGS\.coneLightingRequested[^]*SVO_DRY_VISIBILITY_FLAGS\.globalIllumination/,
    "the visibility lane keeps ambient occlusion, shadows, and requested cone lighting independently switchable");
  assert.match(dryRendererSource, /words\.set\(\[materialCount, scene\.materialRevision, SVO_MATERIAL_RECORD_STRIDE_BYTES, visibilityFlags\]/);
  assert.match(svoDrySceneShader, /fn dryPublishedMaterialValid\(material:SvoMaterialRecord,index:u32\)->bool/);
  assert.match(svoDrySceneShader, /svoMaterialValid\(material,index\)&&material\.identity\.y==dry\.materialPublication\.y&&\(material\.identity\.w&SVO_MATERIAL_FLAG_OPAQUE\)!=0u/);
  // The shader no longer branches on `material.identity.z` at all: the two
  // world-position colour policies it selected between are deleted, and a
  // surface's colour is the published record's own base colour.
  assert.doesNotMatch(svoDrySceneShader, /material\.identity\.z==SVO_MATERIAL_FUNCTION_GARDEN_TERRAIN/);
  assert.match(svoDrySceneShader, /if\(surface\.valid==0u\)\{return vec3f\(0\.0\);\}/,
    "record-content failures must render fail-closed instead of sampling invalid fields");
});

test("shared PBR consumes all opaque surface fields from the producer record", () => {
  assert.match(svoDrySceneShader, /material\.emissiveRoughness\.w/);
  assert.match(svoDrySceneShader, /material\.emissiveRoughness\.xyz\+selectedEmission/);
  assert.match(svoDrySceneShader, /material\.surface\.x,vec3f\(svoMaterialDielectricF0\(material\)\),material\.surface\.y/);
  assert.match(svoDrySceneShader, /unifiedPbrMaterial\(surface\.baseColor,surface\.metallic,surface\.roughness,vec3f\(0\.0\),0\.0,surface\.specularF0,surface\.specularWeight/);
  assert.match(svoDrySceneShader, /surface\.emissive\+diffuseEnvironment\+specularEnvironment\+direct/);
});

test("renderer binds its live material arena, never the producer PBR table", () => {
  const previousUsage = globalThis.GPUBufferUsage;
  const previousTextureUsage = globalThis.GPUTextureUsage;
  Object.assign(globalThis, { GPUBufferUsage: { UNIFORM: 1, COPY_DST: 2, STORAGE: 4 } });
  Object.assign(globalThis, { GPUTextureUsage: { TEXTURE_BINDING: 1 } });
  const pbrBuffer = {} as GPUBuffer;
  let entries: readonly GPUBindGroupEntry[] = [];
  const device = {
    createBuffer(descriptor: { label?: string }) { return { label: descriptor.label, destroy() {} }; },
    createTexture() { return { createView() { return {}; }, destroy() {} }; },
    createSampler() { return {}; },
    createBindGroup(descriptor: { entries: readonly GPUBindGroupEntry[] }) { entries = descriptor.entries; return {}; },
    queue: { writeBuffer() {} },
  } as unknown as GPUDevice;
  try {
    const source = structuralSource({ buffer: pbrBuffer, size: 8 * SVO_MATERIAL_RECORD_STRIDE_BYTES });
    const renderer = new SparseVoxelDrySceneRenderer(device, {} as GPUBuffer, {} as GPUBuffer, "rgba16float", "canonical");
    const internals = renderer as unknown as { layout: GPUBindGroupLayout; pipeline: GPURenderPipeline };
    internals.layout = {} as GPUBindGroupLayout;
    internals.pipeline = {} as GPURenderPipeline;
    renderer.setSource(source);
    renderer.publishScene(scene);
    assert.equal(((entries.find(({ binding }) => binding === 4)?.resource as GPUBufferBinding).buffer as unknown as { label?: string }).label, "Live authored scene arena (materials, primitives/BVH, thin glass, terrain)");
    assert.notEqual((entries.find(({ binding }) => binding === 4)?.resource as GPUBufferBinding).buffer, pbrBuffer);
    renderer.destroy();
  } finally {
    Object.assign(globalThis, { GPUBufferUsage: previousUsage });
    Object.assign(globalThis, { GPUTextureUsage: previousTextureUsage });
  }
});

test("renderer assembles materials on each changed presentation scene", () => {
  assert.match(rendererSource, /const materialRecords = packSvoMaterialTable/);
  assert.match(rendererSource, /this\.svoMaterialsSupported = materialRecords\.byteLength > 0/);
  assert.match(rendererSource, /terrainSupported: this\.svoTerrainSupported,[^]*glassSupported: this\.svoGlassSupported,[^]*materialsSupported: this\.svoMaterialsSupported/);
  assert.match(rendererSource, /failureReason: "missing-pbr-materials"/);
  assert.match(panelSource, /"missing-pbr-materials": "production PBR material table is unavailable"/);
});
