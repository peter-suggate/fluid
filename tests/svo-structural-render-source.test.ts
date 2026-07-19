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
    SPARSE_VOXEL_PUBLICATION_STATE.staticGeometryRevision,
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
});

test("completion generation advances only after scoped revisions and validity", () => {
  const shader = octreeSparseBrickStructuralFinalizeShader;
  assert.match(shader, /fn finalizeInitial\(\) \{ finishFrame\(true\); \}/);
  assert.match(shader, /fn finalizeFrame\(\) \{ finishFrame\(false\); \}/);
  assert.match(shader, new RegExp(`atomicStore\\(&state\\[${SPARSE_VOXEL_PUBLICATION_STATE.topologyRevision}\\], 1u\\)`));
  assert.match(shader, new RegExp(`atomicStore\\(&state\\[${SPARSE_VOXEL_PUBLICATION_STATE.staticGeometryRevision}\\], 1u\\)`));
  assert.match(shader, new RegExp(`atomicAdd\\(&state\\[${SPARSE_VOXEL_PUBLICATION_STATE.dynamicSolidRevision}\\], 1u\\)`));
  assert.match(shader, new RegExp(`atomicAdd\\(&state\\[${SPARSE_VOXEL_PUBLICATION_STATE.coarseFluidRevision}\\], 1u\\)`));
  assert.doesNotMatch(shader, new RegExp(`atomic(?:Store|Add)\\(&state\\[${SPARSE_VOXEL_PUBLICATION_STATE.fineFluidRevision}\\]`));
  assert.ok(
    shader.indexOf(`state[${SPARSE_VOXEL_PUBLICATION_STATE.validFields}]`) <
      shader.indexOf(`state[${SPARSE_VOXEL_PUBLICATION_STATE.completeGeneration}]`),
    "complete generation must be the final publication-state write",
  );
});

test("dry static proxies are encoded before the structural completion fence", () => {
  const source = readFileSync(new URL("../lib/webgpu-octree-sparse-bricks.ts", import.meta.url), "utf8");
  const encodeStart = source.indexOf("  encode(encoder: GPUCommandEncoder");
  const encodeBody = source.slice(encodeStart, source.indexOf("  readResidencyStats", encodeStart));
  assert.ok(encodeBody.indexOf("this.proxyVoxelizer.encode(encoder)") >= 0);
  assert.ok(encodeBody.indexOf("this.proxyVoxelizer.encode(encoder)") < encodeBody.indexOf("Finalize sparse voxel structural publication"));
  assert.ok(encodeBody.indexOf("this.atlas?.encode") < encodeBody.indexOf("Finalize sparse voxel structural publication"));
  assert.doesNotMatch(encodeBody, /mapAsync|getMappedRange/);
});
