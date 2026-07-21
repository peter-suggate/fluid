import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { SVO_MATERIAL_RECORD_STRIDE_BYTES } from "../lib/svo-material-abi";
import {
  canConsumeSparseVoxelPbrMaterials,
  canEncodeSparseVoxelDryScene,
  SparseVoxelDrySceneRenderer,
  SVO_DRY_SCENE_PARAMS_LAYOUT,
  svoDrySceneShader,
  type SparseVoxelDrySceneData,
} from "../lib/webgpu-svo-dry-scene";
import type { SparseVoxelRenderSource } from "../lib/webgpu-voxel-debug";
import { candidateBackedDrySceneFixture } from "./svo-dry-scene-test-fixture";

const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
const dryRendererSource = readFileSync(new URL("../lib/webgpu-svo-dry-scene.ts", import.meta.url), "utf8");
const panelSource = readFileSync(new URL("../components/VisualPanel.tsx", import.meta.url), "utf8");

function structuralSource(
  pbrBinding: GPUBufferBinding = { buffer: {} as GPUBuffer, size: 8 * SVO_MATERIAL_RECORD_STRIDE_BYTES },
): SparseVoxelRenderSource {
  const resource = { buffer: {} as GPUBuffer };
  return {
    materialCount: 8,
    materials: resource,
    pbrMaterials: { binding: pbrBinding, count: 8, strideBytes: SVO_MATERIAL_RECORD_STRIDE_BYTES, revision: 3 },
    structural: {
      control: resource, nodes: resource, leaves: resource, geometry: resource,
      velocity: resource, materialOwners: resource, fluidLeafStates: resource,
      publication: { state: resource, byteLength: 32 },
      domain: { worldOrigin_m: [-2, 0, -2], cellSize_m: [0.04, 0.04, 0.04], dimensionsCells: [64, 64, 64], brickSize: 16, maximumDepth: 4 },
      capacities: { nodes: 64, leaves: 32, geometryVoxels: 1024, velocityVoxels: 1024, materialOwnerVoxels: 1024, fluidLeafStates: 32 },
      strides: { control: 4, node: 32, leaf: 16, geometry: 16, velocity: 16, materialOwner: 4, fluidLeafState: 4 },
      fields: {
        topology: { residency: "all-published-leaves", validity: "published-generation", revision: 1 },
        staticGeometry: { residency: "all-published-leaves", validity: "published-generation", revision: 1 },
        materialOwner: { residency: "all-published-leaves", validity: "published-generation", revision: 1 },
        dynamicSolid: { residency: "unavailable", validity: "unavailable", revision: 0 },
        coarseFluid: { residency: "unavailable", validity: "unavailable", revision: 0 },
        fineFluid: { residency: "unavailable", validity: "unavailable", revision: 0 },
      },
      generation: { published: 1, completed: 1 },
    },
  } as unknown as SparseVoxelRenderSource;
}

const scene: SparseVoxelDrySceneData = candidateBackedDrySceneFixture;

test("production PBR publication validation is exact and malformed sources fail over before encoding", () => {
  const valid = structuralSource();
  assert.equal(canConsumeSparseVoxelPbrMaterials(valid), true);
  assert.equal(canEncodeSparseVoxelDryScene(valid, scene), true);
  assert.equal(canConsumeSparseVoxelPbrMaterials({ ...valid, pbrMaterials: undefined }), false);
  assert.equal(canConsumeSparseVoxelPbrMaterials({ ...valid, pbrMaterials: { ...valid.pbrMaterials!, strideBytes: 32 } }), false);
  assert.equal(canConsumeSparseVoxelPbrMaterials({ ...valid, pbrMaterials: { ...valid.pbrMaterials!, count: 1 } }), false);
  assert.equal(canConsumeSparseVoxelPbrMaterials({ ...valid, pbrMaterials: { ...valid.pbrMaterials!, count: 2.5 } }), false);
  assert.equal(canConsumeSparseVoxelPbrMaterials({ ...valid, pbrMaterials: { ...valid.pbrMaterials!, count: 0x1_0000_0000 } }), false);
  assert.equal(canConsumeSparseVoxelPbrMaterials({ ...valid, pbrMaterials: { ...valid.pbrMaterials!, revision: 0 } }), false);
  assert.equal(canConsumeSparseVoxelPbrMaterials({ ...valid, pbrMaterials: { ...valid.pbrMaterials!, revision: 0x1_0000_0000 } }), false);
  assert.equal(canConsumeSparseVoxelPbrMaterials(structuralSource({ buffer: {} as GPUBuffer, size: 767 })), false);
  assert.equal(canEncodeSparseVoxelDryScene({ ...valid, pbrMaterials: undefined }, scene), false);
});

test("binding 6 consumes the 96-byte producer table while the legacy debug ABI remains available", () => {
  assert.match(svoDrySceneShader, /@group\(0\) @binding\(6\) var<storage,read> materials:array<SvoMaterialRecord>/);
  assert.match(dryRendererSource, /\{ binding: 6, resource: source\.pbrMaterials!\.binding \}/);
  assert.doesNotMatch(dryRendererSource, /binding: 6, resource: source\.materials/);
  assert.match(dryRendererSource, /@group\(0\) @binding\(10\)/);
  assert.doesNotMatch(dryRendererSource, /svoStructuralGeometry|svoStructuralLeafStates/,
    "the dry pass must not retain structural fluid-march payload bindings");
  assert.match(readFileSync(new URL("../lib/webgpu-voxel-debug.ts", import.meta.url), "utf8"), /materials: GPUBufferBinding/,
    "inspection keeps its compact legacy material binding");
});

test("published count, revision, direct identity, flags, and material functions are enforced in WGSL", () => {
  assert.deepEqual(SVO_DRY_SCENE_PARAMS_LAYOUT, {
    sizeBytes: 192, terrainWordOffset: 24, terrainMaterialWordOffset: 28, materialPublicationWordOffset: 32,
    primitiveCandidateWordOffset: 36, nodeMipWordOffset: 40, nodeMipAtlasWordOffset: 44,
  });
  assert.match(dryRendererSource, /const visibilityFlags = \(ambientOcclusionEnabled \? SVO_DRY_VISIBILITY_FLAGS\.exactContact \| SVO_DRY_VISIBILITY_FLAGS\.ambientOcclusion : 0\)[^]*SVO_DRY_VISIBILITY_FLAGS\.exactShadow[^]*SVO_DRY_VISIBILITY_FLAGS\.coneLightingRequested/,
    "the visibility lane keeps ambient occlusion, shadows, and requested cone lighting independently switchable");
  assert.match(dryRendererSource, /words\.set\(\[pbrMaterials\.count, pbrMaterials\.revision, pbrMaterials\.strideBytes, visibilityFlags\], SVO_DRY_SCENE_PARAMS_LAYOUT\.materialPublicationWordOffset\)/);
  assert.match(svoDrySceneShader, /fn dryPublishedMaterialValid\(material:SvoMaterialRecord,index:u32\)->bool/);
  assert.match(svoDrySceneShader, /svoMaterialValid\(material,index\)&&material\.identity\.y==dry\.materialPublication\.y&&\(material\.identity\.w&SVO_MATERIAL_FLAG_OPAQUE\)!=0u/);
  assert.match(svoDrySceneShader, /material\.identity\.z==SVO_MATERIAL_FUNCTION_GARDEN_TERRAIN/);
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

test("renderer binds producer PBR identity, never the legacy inspection buffer", () => {
  const previousUsage = globalThis.GPUBufferUsage;
  const previousTextureUsage = globalThis.GPUTextureUsage;
  Object.assign(globalThis, { GPUBufferUsage: { UNIFORM: 1, COPY_DST: 2, STORAGE: 4 } });
  Object.assign(globalThis, { GPUTextureUsage: { TEXTURE_BINDING: 1 } });
  const legacyBuffer = {} as GPUBuffer;
  const pbrBuffer = {} as GPUBuffer;
  let entries: readonly GPUBindGroupEntry[] = [];
  const device = {
    createBuffer() { return { destroy() {} }; },
    createTexture() { return { createView() { return {}; }, destroy() {} }; },
    createSampler() { return {}; },
    createBindGroup(descriptor: { entries: readonly GPUBindGroupEntry[] }) { entries = descriptor.entries; return {}; },
    queue: { writeBuffer() {} },
  } as unknown as GPUDevice;
  try {
    const source = structuralSource({ buffer: pbrBuffer, size: 8 * SVO_MATERIAL_RECORD_STRIDE_BYTES });
    source.materials = { buffer: legacyBuffer };
    const renderer = new SparseVoxelDrySceneRenderer(device, {} as GPUBuffer, {} as GPUBuffer);
    const internals = renderer as unknown as { layout: GPUBindGroupLayout; pipeline: GPURenderPipeline };
    internals.layout = {} as GPUBindGroupLayout;
    internals.pipeline = {} as GPURenderPipeline;
    renderer.setSource(source, scene);
    assert.equal((entries.find(({ binding }) => binding === 6)?.resource as GPUBufferBinding).buffer, pbrBuffer);
    assert.notEqual((entries.find(({ binding }) => binding === 6)?.resource as GPUBufferBinding).buffer, legacyBuffer);
    renderer.destroy();
  } finally {
    Object.assign(globalThis, { GPUBufferUsage: previousUsage });
    Object.assign(globalThis, { GPUTextureUsage: previousTextureUsage });
  }
});

test("missing PBR publication has an explicit raster fallback reason", () => {
  assert.match(rendererSource, /svoMaterialsSupported=canConsumeSparseVoxelPbrMaterials\(sparseSceneSource\)/);
  assert.match(rendererSource, /terrainSupported: this\.svoTerrainSupported,[^]*glassSupported: this\.svoGlassSupported,[^]*materialsSupported: this\.svoMaterialsSupported/);
  assert.match(rendererSource, /fallbackReason: "missing-pbr-materials"/);
  assert.match(panelSource, /"missing-pbr-materials": "production PBR material table is unavailable"/);
});
