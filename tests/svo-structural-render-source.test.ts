import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { SPARSE_BRICK_GPU_LAYOUT } from "../lib/sparse-brick-octree";
import { octreeSparseBrickStructuralFinalizeShader } from "../lib/webgpu-octree-sparse-bricks";
import {
  SPARSE_VOXEL_PUBLICATION_STATE,
  SPARSE_VOXEL_VALID_FIELDS,
  type SparseVoxelRenderSource,
} from "../lib/webgpu-voxel-debug";

test("structural source remains optional for legacy debug producers", () => {
  const binding = { buffer: {} as GPUBuffer };
  const legacy = {
    voxelRecords: binding,
    voxelCount: binding,
    brickRecords: binding,
    brickCount: binding,
    materials: binding,
    voxelCapacity: 64,
    brickCapacity: 8,
    materialCount: 3,
    revision: 7,
  } satisfies SparseVoxelRenderSource;
  assert.equal(legacy.revision, 7);
});

test("structural publication words and field flags are stable and non-overlapping", () => {
  const words = [
    SPARSE_VOXEL_PUBLICATION_STATE.completeGeneration,
    SPARSE_VOXEL_PUBLICATION_STATE.validFields,
    SPARSE_VOXEL_PUBLICATION_STATE.topologyRevision,
    SPARSE_VOXEL_PUBLICATION_STATE.sceneGeometryRevision,
    SPARSE_VOXEL_PUBLICATION_STATE.dynamicSolidRevision,
    SPARSE_VOXEL_PUBLICATION_STATE.coarseFluidRevision,
    SPARSE_VOXEL_PUBLICATION_STATE.fineFluidRevision,
  ];
  assert.equal(new Set(words).size, words.length);
  assert.ok(Math.max(...words) * 4 < SPARSE_VOXEL_PUBLICATION_STATE.strideBytes);

  const flags = Object.values(SPARSE_VOXEL_VALID_FIELDS);
  assert.equal(new Set(flags).size, flags.length);
  for (const flag of flags) assert.equal(flag & (flag - 1), 0, `${flag} must be one bit`);
});

test("production source publishes native sparse arenas with explicit offsets and strides", () => {
  const source = readFileSync(new URL("../lib/webgpu-octree-sparse-bricks.ts", import.meta.url), "utf8");
  assert.match(source, /structure: \{ buffer: this\.tree\.structure/);
  assert.match(source, /structureOffsetsWords:/);
  for (const member of ["control", "nodes", "leaves", "geometry", "velocity", "materialOwners", "fluidLeafStates"]) {
    assert.match(source, new RegExp(`${member}: \\{ buffer:`));
  }
  assert.match(source, /offset: this\.tree\.leafOffsetBytes/);
  assert.match(source, /offset: this\.tree\.velocityOffsetBytes/);
  assert.match(source, /offset: this\.tree\.materialOwnerOffsetBytes/);
  assert.match(source, /worldOrigin_m:/);
  assert.match(source, /cellSize_m: this\.cellSize/);
  assert.match(source, /dimensionsCells: sceneDomain\.sceneDimensionsCells/);
  assert.match(source, /maximumDepth: plan\.maximumDepth/);
  assert.equal(SPARSE_BRICK_GPU_LAYOUT.nodeStrideBytes, 32);
  assert.equal(SPARSE_BRICK_GPU_LAYOUT.leafStrideBytes, 16);
  assert.equal(SPARSE_BRICK_GPU_LAYOUT.geometryStrideBytes, 16);
  assert.equal(SPARSE_BRICK_GPU_LAYOUT.velocityStrideBytes, 16);
  assert.equal(SPARSE_BRICK_GPU_LAYOUT.materialOwnerStrideBytes, 4);
  assert.equal(SPARSE_BRICK_GPU_LAYOUT.publicationOffsetBytes % 256, 0);
  assert.equal(SPARSE_BRICK_GPU_LAYOUT.topologyOffsetBytes % 256, 0);
  assert.ok(SPARSE_BRICK_GPU_LAYOUT.topologyOffsetBytes > SPARSE_BRICK_GPU_LAYOUT.publicationOffsetBytes);
});

test("completion generation advances only after a completed live-scene maintenance revision", () => {
  const shader = octreeSparseBrickStructuralFinalizeShader;
  const sceneFinalize = shader.slice(shader.indexOf("fn finalizeScene()"));
  assert.match(shader, /@compute @workgroup_size\(1\)\s*fn finalizeScene/);
  assert.match(sceneFinalize, /requested == 0u \|\| completed != requested \|\| overflow != 0u/);
  assert.match(sceneFinalize, new RegExp(`state\\[${SPARSE_VOXEL_PUBLICATION_STATE.topologyRevision}\\] \\+= 1u;`));
  assert.match(sceneFinalize, new RegExp(`state\\[${SPARSE_VOXEL_PUBLICATION_STATE.sceneGeometryRevision}\\] \\+= 1u;`));
  assert.doesNotMatch(sceneFinalize, new RegExp(`state\\[${SPARSE_VOXEL_PUBLICATION_STATE.dynamicSolidRevision}\\]\\s*(?:\\+)?=`));
  assert.doesNotMatch(sceneFinalize, new RegExp(`state\\[${SPARSE_VOXEL_PUBLICATION_STATE.coarseFluidRevision}\\]\\s*(?:\\+)?=`));
  assert.doesNotMatch(sceneFinalize,
    new RegExp(`state\\[${SPARSE_VOXEL_PUBLICATION_STATE.fineFluidRevision}\\]\\s*(?:\\+)?=`),
    "scene maintenance must not claim or mutate fluid publications");
  assert.ok(
    sceneFinalize.indexOf(`state[${SPARSE_VOXEL_PUBLICATION_STATE.validFields}]`) <
      sceneFinalize.indexOf(`state[${SPARSE_VOXEL_PUBLICATION_STATE.completeGeneration}]`),
    "complete generation must be the final publication-state write",
  );
});

test("live scene proxies are maintained before the structural completion fence", () => {
  const source = readFileSync(new URL("../lib/webgpu-octree-sparse-bricks.ts", import.meta.url), "utf8");
  const encodeStart = source.indexOf("  encodeSceneMaintenance(encoder: GPUCommandEncoder");
  const encodeBody = source.slice(encodeStart, source.indexOf("\n  encode(encoder: GPUCommandEncoder", encodeStart));
  assert.ok(encodeBody.indexOf("this.proxyVoxelizer.encodeMaintenance(encoder)") >= 0);
  assert.ok(encodeBody.indexOf("this.proxyVoxelizer.encodeMaintenance(encoder)")
    < encodeBody.indexOf("Finalize live sparse voxel scene publication"));
  assert.doesNotMatch(encodeBody, /mapAsync|getMappedRange/);
});
