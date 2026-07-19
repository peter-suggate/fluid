import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { gardenPoolTerrain, GARDEN_WATERLINE_M } from "../lib/garden-scene";
import { cloneScene, defaultScene } from "../lib/model";
import { buildSvoTerrainMaterial, SVO_TERRAIN_MATERIAL_METADATA_STRIDE_BYTES } from "../lib/svo-terrain-material";
import {
  canEncodeSparseVoxelDryScene,
  SparseVoxelDrySceneRenderer,
  SVO_DRY_SCENE_PARAMS_LAYOUT,
  svoDrySceneShader,
  type SparseVoxelDrySceneData,
} from "../lib/webgpu-svo-dry-scene";
import type { SparseVoxelRenderSource } from "../lib/webgpu-voxel-debug";

const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");

function structuralSource(): SparseVoxelRenderSource {
  const resource = { buffer: {} as GPUBuffer };
  return {
    materialCount: 8,
    materials: resource,
    pbrMaterials: { binding: resource, count: 8, strideBytes: 96, revision: 1 },
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

function gardenMaterial() {
  const scene = cloneScene(defaultScene);
  scene.terrain = gardenPoolTerrain();
  scene.container.height_m = 1;
  scene.container.fillFraction = GARDEN_WATERLINE_M;
  return buildSvoTerrainMaterial(scene);
}

test("production garden metadata is packed into the existing dry uniform without a new binding", () => {
  assert.match(rendererSource, /const terrainMaterial=scenePrimitives\.analyticTerrain\?buildSvoTerrainMaterial\(scene\):undefined/);
  assert.match(rendererSource, /terrainMaterialMetadata:terrainMaterial\?\.packedMetadata,terrainMaterialCacheKey:terrainMaterial\?\.cacheKey/);
  assert.equal(SVO_TERRAIN_MATERIAL_METADATA_STRIDE_BYTES, 16);
  assert.deepEqual(SVO_DRY_SCENE_PARAMS_LAYOUT, {
    sizeBytes: 160, terrainWordOffset: 24, terrainMaterialWordOffset: 28, materialPublicationWordOffset: 32, fluidDomainWordOffset: 36,
  });
  assert.match(svoDrySceneShader, /terrainMaterial:SvoTerrainMaterialMetadata/);
  assert.match(svoDrySceneShader, /@binding\(11\) var<storage,read> svoStructuralGeometry/);
  assert.match(svoDrySceneShader, /@group\(0\) @binding\(10\) var<storage,read> glassPanes/);
});

test("analytic terrain evaluates the exact raster world-space material before PBR", () => {
  assert.match(svoDrySceneShader, /fn svoTerrainMaterial\(metadata:SvoTerrainMaterialMetadata,p:vec3f,normalIn:vec3f\)/);
  assert.match(svoDrySceneShader, /floor\(p\.xz\*26\.0\)/);
  assert.match(svoDrySceneShader, /smoothstep\(metadata\.waterline_m-\.02,metadata\.waterline_m\+\.04,p\.y\)/);
  assert.match(svoDrySceneShader, /let terrainSample=svoTerrainMaterial\(dry\.terrainMaterial,position,hit\.normal\);base=terrainSample\.colorLinear/);
  assert.match(svoDrySceneShader, /unifiedPbrMaterial\(surface\.baseColor,surface\.metallic,surface\.roughness,vec3f\(0\.0\),0\.0/,
    "procedural color and the stable terrain-table roughness must feed the PBR closure");
});

test("terrain region and seeded variation identity have an explicit pending G-buffer adapter", () => {
  assert.match(svoDrySceneShader, /struct DrySurfaceMaterial[^]*regionId:u32,variationFlags:u32/);
  assert.match(svoDrySceneShader, /regionId=terrainSample\.regionId;variationFlags=terrainSample\.variationFlags/);
  assert.match(svoDrySceneShader, /DRY_SURFACE_REGION_NONE:u32=0xffffffffu/);
  assert.match(svoDrySceneShader, /terrainPolicyValid=material\.identity\.z==SVO_MATERIAL_FUNCTION_GARDEN_TERRAIN&&dry\.terrainMaterial\.policyVersion==1u&&dry\.terrainMaterial\.materialId==materialId&&materialId==dry\.terrain\.x/,
    "non-garden and mismatched material IDs must preserve default table shading");
});

test("terrain metadata validation and dry-parameter uploads are exact and content cached", () => {
  const source = structuralSource();
  const build = gardenMaterial();
  const base: SparseVoxelDrySceneData = {
    primitiveRecords: new Uint32Array(16), ownerBase: 0, terrainMaterialId: build.metadata.materialId,
    terrainMaterialMetadata: build.packedMetadata, terrainMaterialCacheKey: build.cacheKey,
  };
  assert.equal(canEncodeSparseVoxelDryScene(source, base), true);
  assert.equal(canEncodeSparseVoxelDryScene(source, { ...base, terrainMaterialMetadata: new Uint32Array(3) }), false);
  assert.equal(canEncodeSparseVoxelDryScene(source, { ...base, terrainMaterialMetadata: undefined }), true,
    "non-garden/default table shading remains encodable");

  const previousUsage = globalThis.GPUBufferUsage;
  Object.assign(globalThis, { GPUBufferUsage: { UNIFORM: 1, COPY_DST: 2, STORAGE: 4 } });
  const writes: Array<{ label?: string; words: Uint32Array }> = [];
  const device = {
    createBuffer(descriptor: { label?: string }) {
      return { label: descriptor.label, destroy() {} };
    },
    queue: {
      writeBuffer(target: { label?: string }, _offset: number, data: ArrayBuffer | ArrayBufferView) {
        const bytes = data instanceof ArrayBuffer
          ? new Uint8Array(data)
          : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        writes.push({ label: target.label, words: new Uint32Array(bytes.slice().buffer) });
      },
    },
  } as unknown as GPUDevice;
  try {
    const renderer = new SparseVoxelDrySceneRenderer(device, {} as GPUBuffer, {} as GPUBuffer);
    renderer.setSource(source, base);
    const parameterWrites = () => writes.filter(({ label }) => label === "Sparse voxel dry scene parameters");
    assert.equal(parameterWrites().length, 1);
    assert.deepEqual(
      [...parameterWrites()[0].words.slice(
        SVO_DRY_SCENE_PARAMS_LAYOUT.terrainMaterialWordOffset,
        SVO_DRY_SCENE_PARAMS_LAYOUT.materialPublicationWordOffset,
      )],
      [...build.packedMetadata],
    );
    renderer.setSource(source, base);
    assert.equal(parameterWrites().length, 1, "unchanged packed metadata must not rewrite the uniform");
    const changed = gardenMaterial();
    const changedWords = Uint32Array.from(changed.packedMetadata);
    new Float32Array(changedWords.buffer)[1] += 0.01;
    renderer.setSource(source, { ...base, terrainMaterialMetadata: changedWords, terrainMaterialCacheKey: `${changed.cacheKey}:changed` });
    assert.equal(parameterWrites().length, 2);
    renderer.destroy();
  } finally {
    Object.assign(globalThis, { GPUBufferUsage: previousUsage });
  }
});
