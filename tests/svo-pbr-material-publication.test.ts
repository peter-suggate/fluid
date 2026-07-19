import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  SVO_MATERIAL_FLAGS,
  SVO_MATERIAL_RECORD_STRIDE_BYTES,
  unpackSvoMaterialRecord,
} from "../lib/svo-material-abi";
import {
  buildOctreeSvoPbrMaterialPublication,
  ENVIRONMENT_VOXEL_MATERIAL_BASE,
  OCTREE_SVO_PBR_MATERIAL_REVISION,
} from "../lib/webgpu-octree-sparse-bricks";
import { cloneScene, defaultScene } from "../lib/model";
import { environmentIds } from "../lib/environments";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives } from "../lib/voxel-environments";
import type { SparseVoxelRenderSource } from "../lib/webgpu-voxel-debug";
import { VOXEL_MATERIALS } from "../lib/voxel-scene";

test("PBR publication is optional and leaves the legacy debug source ABI intact", () => {
  const binding = { buffer: {} as GPUBuffer };
  const legacy = {
    voxelRecords: binding,
    voxelCount: binding,
    brickRecords: binding,
    brickCount: binding,
    materials: binding,
    voxelCapacity: 64,
    brickCapacity: 8,
    materialCount: 20,
    revision: 3,
  } satisfies SparseVoxelRenderSource;
  assert.equal(legacy.materials, binding);

  const modern = {
    ...legacy,
    pbrMaterials: { binding, count: 20, strideBytes: 96, revision: 7 },
  } satisfies SparseVoxelRenderSource;
  assert.equal(modern.pbrMaterials.binding, binding);
  assert.equal(modern.materials, legacy.materials);
});

test("producer table is a dense 96-byte direct-index publication with explicit revision", () => {
  const publication = buildOctreeSvoPbrMaterialPublication(17);
  const repeated = buildOctreeSvoPbrMaterialPublication(17);
  const maximumId = Math.max(...VOXEL_MATERIALS.map(({ id }) => id));
  assert.equal(publication.strideBytes, SVO_MATERIAL_RECORD_STRIDE_BYTES);
  assert.equal(publication.count, maximumId + 1);
  assert.equal(publication.packedRecords.byteLength, publication.count * publication.strideBytes);
  assert.equal(publication.revision, 17);
  assert.strictEqual(repeated, publication);
  assert.strictEqual(repeated.packedRecords, publication.packedRecords);
  assert.equal(repeated.cacheKey, publication.cacheKey);
  assert.notEqual(buildOctreeSvoPbrMaterialPublication(18).cacheKey, publication.cacheKey);
  for (const material of VOXEL_MATERIALS) {
    const record = unpackSvoMaterialRecord(publication.packedRecords, material.id);
    assert.equal(record.materialId, material.id, `${material.key} remains its direct table index`);
    assert.equal(record.revision, publication.revision);
  }
  assert.throws(() => buildOctreeSvoPbrMaterialPublication(0), /positive uint32/);
});

test("every shipped environment primitive publishes a finite non-black opaque PBR closure", () => {
  const scene = cloneScene(defaultScene);
  for (const environmentId of environmentIds) {
    const primitives = environmentProxyPrimitives(buildEnvironmentProxyCatalog(scene, environmentId), true);
    const publication = buildOctreeSvoPbrMaterialPublication(23, primitives);
    for (const primitive of primitives) {
      const materialId = ENVIRONMENT_VOXEL_MATERIAL_BASE + primitive.ownerIndex;
      assert.ok(materialId < publication.count, `${primitive.key} material is inside the direct-index table`);
      const material = unpackSvoMaterialRecord(publication.packedRecords, materialId);
      assert.equal(material.materialId, materialId, `${primitive.key} keeps its stable material ID`);
      assert.equal(material.revision, publication.revision);
      assert.equal(material.flags, SVO_MATERIAL_FLAGS.opaque);
      const shadingInputs = [
        ...material.baseColorLinear,
        ...material.emissiveLinear,
        material.roughness,
        material.metallic,
        material.specularWeight,
      ];
      assert.ok(shadingInputs.every(Number.isFinite), `${primitive.key} shading inputs are finite`);
      assert.ok(
        material.baseColorLinear.some((channel) => channel > 0)
          || material.emissiveLinear.some((channel) => channel > 0),
        `${primitive.key} cannot resolve to the invalid black closure`,
      );
    }
  }
  const primitive = environmentProxyPrimitives(buildEnvironmentProxyCatalog(scene, "default"), true)[0];
  assert.throws(
    () => buildOctreeSvoPbrMaterialPublication(1, [{ ...primitive, ownerIndex: 0xffff }]),
    /does not fit uint16/,
  );
  assert.throws(
    () => buildOctreeSvoPbrMaterialPublication(1, [{ ...primitive, ownerIndex: -1 }]),
    /non-negative safe integer/,
  );
});

test("octree world owns, accounts, publishes, and destroys the PBR buffer once", () => {
  const source = readFileSync(new URL("../lib/webgpu-octree-sparse-bricks.ts", import.meta.url), "utf8");
  assert.equal(OCTREE_SVO_PBR_MATERIAL_REVISION, 2);
  assert.match(source, /private readonly pbrMaterialBuffer: GPUBuffer/);
  assert.match(source, /this\.pbrMaterialBuffer = storageBuffer\(/);
  assert.match(source, /buildOctreeSvoPbrMaterialPublication\([\s\S]*environmentPrimitives/,
    "the production publication includes the selected environment catalog");
  assert.match(source, /binding: \{ buffer: this\.pbrMaterialBuffer, size: pbrMaterials\.packedRecords\.byteLength \}/);
  assert.match(source, /count: pbrMaterials\.count/);
  assert.match(source, /strideBytes: pbrMaterials\.strideBytes/);
  assert.match(source, /revision: pbrMaterials\.revision/);
  assert.match(source, /this\.sourceBuffers\.reduce\(\(sum, buffer\) => sum \+ buffer\.size, 0\)/,
    "the legacy inspection material buffer remains covered by the source-buffer accounting set");
  assert.match(source, /\+ this\.pbrMaterialBuffer\.size/,
    "the expanded production table is accounted independently");
  const destroy = source.slice(source.indexOf("  destroy(): void {"));
  assert.match(destroy, /\.\.\.this\.sourceBuffers, this\.pbrMaterialBuffer, this\.lightBuffer, this\.environmentLightingBuffer, this\.structuralPublicationState/);
  assert.equal((destroy.match(/this\.pbrMaterialBuffer/g) ?? []).length, 1);
});
