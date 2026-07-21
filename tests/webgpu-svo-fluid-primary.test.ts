import assert from "node:assert/strict";
import test from "node:test";

import { SPARSE_BRICK_GPU_LAYOUT } from "../lib/sparse-brick-octree";
import {
  canConsumeSvoFineFluidCapability,
  canEncodeSparseVoxelDryScene,
  SVO_DRY_SCENE_PARAMS_LAYOUT,
  svoDrySceneShader,
  type SparseVoxelDrySceneData,
} from "../lib/webgpu-svo-dry-scene";
import { SVO_FINE_PHI_CONTROL_WORDS, type SvoFineFluidGpuCapability } from "../lib/webgpu-svo-fine-phi-stager";
import {
  canConsumeSparseVoxelCoarseFluidPrimary,
  createSvoStructuralFluidPrimaryWGSL,
  DEFAULT_SVO_FLUID_PRIMARY_MODE,
  SVO_STRUCTURAL_FLUID_PRIMARY_LIMITS,
  SVO_STRUCTURAL_FLUID_PRIMARY_STORAGE_BINDINGS,
} from "../lib/webgpu-svo-fluid-primary";
import { createWebgpuSvoTraversalWGSL } from "../lib/webgpu-svo-traversal";
import {
  SPARSE_VOXEL_PUBLICATION_STATE,
  SPARSE_VOXEL_VALID_FIELDS,
  type SparseVoxelSceneRenderSource,
} from "../lib/webgpu-voxel-debug";
import { candidateBackedDrySceneFixture } from "./svo-dry-scene-test-fixture";

function buffer(size: number): GPUBuffer {
  return { size } as GPUBuffer;
}

function source(): SparseVoxelSceneRenderSource {
  const capacities = { nodes: 64, leaves: 32, voxels: 1024 };
  const publication = { buffer: buffer(SPARSE_VOXEL_PUBLICATION_STATE.strideBytes) };
  const publicationWord = (word: number) => ({ binding: publication, word });
  return {
    materialCount: 2,
    revision: 1,
    pbrMaterials: { binding: { buffer: buffer(2 * 96), size: 2 * 96 }, count: 2, strideBytes: 96, revision: 1 },
    structural: {
      control: { buffer: buffer(SPARSE_BRICK_GPU_LAYOUT.controlStrideBytes), size: SPARSE_BRICK_GPU_LAYOUT.controlStrideBytes },
      nodes: { buffer: buffer(capacities.nodes * SPARSE_BRICK_GPU_LAYOUT.nodeStrideBytes), size: capacities.nodes * SPARSE_BRICK_GPU_LAYOUT.nodeStrideBytes },
      leaves: { buffer: buffer(capacities.leaves * SPARSE_BRICK_GPU_LAYOUT.leafStrideBytes), size: capacities.leaves * SPARSE_BRICK_GPU_LAYOUT.leafStrideBytes },
      geometry: { buffer: buffer(capacities.voxels * SPARSE_BRICK_GPU_LAYOUT.geometryStrideBytes), size: capacities.voxels * SPARSE_BRICK_GPU_LAYOUT.geometryStrideBytes },
      velocity: { buffer: buffer(capacities.voxels * SPARSE_BRICK_GPU_LAYOUT.velocityStrideBytes) },
      materialOwners: { buffer: buffer(capacities.voxels * SPARSE_BRICK_GPU_LAYOUT.materialOwnerStrideBytes) },
      fluidLeafStates: { buffer: buffer(capacities.leaves * 4), size: capacities.leaves * 4 },
      capacities,
      strides: {
        control: SPARSE_BRICK_GPU_LAYOUT.controlStrideBytes,
        node: SPARSE_BRICK_GPU_LAYOUT.nodeStrideBytes,
        leaf: SPARSE_BRICK_GPU_LAYOUT.leafStrideBytes,
        geometry: SPARSE_BRICK_GPU_LAYOUT.geometryStrideBytes,
        velocity: SPARSE_BRICK_GPU_LAYOUT.velocityStrideBytes,
        materialOwner: SPARSE_BRICK_GPU_LAYOUT.materialOwnerStrideBytes,
      },
      domain: {
        worldOrigin_m: [-2, 0, -2],
        cellSize_m: [0.04, 0.05, 0.04],
        dimensionsCells: [96, 64, 96],
        brickSize: 8,
        maximumDepth: 4,
      },
      publication: {
        state: publication,
        completeGeneration: publicationWord(SPARSE_VOXEL_PUBLICATION_STATE.completeGeneration),
        validFields: publicationWord(SPARSE_VOXEL_PUBLICATION_STATE.validFields),
        revisions: {
          topology: publicationWord(SPARSE_VOXEL_PUBLICATION_STATE.topologyRevision),
          staticGeometry: publicationWord(SPARSE_VOXEL_PUBLICATION_STATE.staticGeometryRevision),
          dynamicSolid: publicationWord(SPARSE_VOXEL_PUBLICATION_STATE.dynamicSolidRevision),
          coarseFluid: publicationWord(SPARSE_VOXEL_PUBLICATION_STATE.coarseFluidRevision),
          fineFluid: publicationWord(SPARSE_VOXEL_PUBLICATION_STATE.fineFluidRevision),
        },
      },
      fields: {
        topology: { bit: SPARSE_VOXEL_VALID_FIELDS.topology, residency: "all-published-leaves" },
        staticGeometry: { bit: SPARSE_VOXEL_VALID_FIELDS.staticGeometry, residency: "all-published-leaves" },
        dynamicSolid: { bit: SPARSE_VOXEL_VALID_FIELDS.dynamicSolid, residency: "fluid-resident-leaves" },
        coarseFluid: { bit: SPARSE_VOXEL_VALID_FIELDS.coarseFluid, signedDistance: "negative-inside-metres", distanceQuality: "metric-near-interface", residency: "fluid-resident-leaves" },
        fineFluid: { bit: SPARSE_VOXEL_VALID_FIELDS.fineFluid, signedDistance: "negative-inside-metres", distanceQuality: "metric", residency: "unavailable" },
        velocity: { bit: SPARSE_VOXEL_VALID_FIELDS.velocity, residency: "fluid-resident-leaves" },
        materialOwner: { bit: SPARSE_VOXEL_VALID_FIELDS.materialOwner, residency: "all-published-leaves" },
      },
    },
  };
}

test("coarse structural primary ABI validation requires explicit metric SDF residency and complete bindings", () => {
  const valid = source();
  assert.equal(canConsumeSparseVoxelCoarseFluidPrimary(valid), true);
  assert.equal(DEFAULT_SVO_FLUID_PRIMARY_MODE, "legacy-compositor");
  assert.equal(SVO_STRUCTURAL_FLUID_PRIMARY_STORAGE_BINDINGS, 10);
  assert.deepEqual(SVO_STRUCTURAL_FLUID_PRIMARY_LIMITS, { leafVisits: 48, fieldSteps: 256, refinementIterations: 8 });

  assert.equal(canConsumeSparseVoxelCoarseFluidPrimary({ ...valid, structural: { ...valid.structural!, fields: { ...valid.structural!.fields, coarseFluid: { ...valid.structural!.fields.coarseFluid, residency: "unavailable" } } } }), false);
  assert.equal(canConsumeSparseVoxelCoarseFluidPrimary({ ...valid, structural: { ...valid.structural!, fields: { ...valid.structural!.fields, coarseFluid: { ...valid.structural!.fields.coarseFluid, distanceQuality: "occupancy-estimate" } } } }), false);
  assert.equal(canConsumeSparseVoxelCoarseFluidPrimary({ ...valid, structural: { ...valid.structural!, geometry: { buffer: buffer(16), size: 16 } } }), false);
  assert.equal(canConsumeSparseVoxelCoarseFluidPrimary({ ...valid, structural: { ...valid.structural!, fluidLeafStates: { buffer: buffer(4), size: 4 } } }), false);
});

test("primary marcher remaps real structural arrays and enforces generation, residency, and work failure", () => {
  const shader = createSvoStructuralFluidPrimaryWGSL({
    control: "dryControl",
    nodes: "dryNodes",
    leaves: "dryLeaves",
    geometry: "dryGeometry",
    leafStates: "dryLeafStates",
    publication: "dryPublication",
    domainFunction: "dryFluidDomain",
  });
  assert.match(shader, /dryPublication\[0u\]/);
  assert.match(shader, /dryPublication\[5u\]/);
  assert.match(shader, /dryLeafStates\[leaf\.leafIndex\]&SVO_STRUCTURAL_RESIDENT/);
  assert.match(shader, /svoStructuralCoarseFluidTrilinear\(domain/);
  assert.match(shader, /svoFluidRefineZero\(/);
  assert.match(shader, /svoFluidGradientNormal\(/);
  assert.match(shader, /SVO_FLUID_PRIMARY_EXHAUSTED/);
  assert.doesNotMatch(shader, /svoStructuralFluidDomain/);
  assert.doesNotMatch(shader, /svoStructuralGeometry/);
  assert.throws(() => createSvoStructuralFluidPrimaryWGSL({ geometry: "not-valid!" }), /Invalid WGSL identifier/);
  assert.throws(() => createSvoStructuralFluidPrimaryWGSL({ nodeWordFunction: "nodeWord" }), /requires both word and word-length/);
});

test("shared traversal and primary marcher compose without duplicate structural storage declarations", () => {
  const traversal = createWebgpuSvoTraversalWGSL({ control: 2, nodes: 3, leaves: 4 })
    .replaceAll("svoControl", "svoStructuralControl")
    .replaceAll("svoNodes", "svoStructuralNodes")
    .replaceAll("svoLeaves", "svoStructuralLeaves");
  const primary = createSvoStructuralFluidPrimaryWGSL({
    nodeWordFunction: "dryNodeWord",
    nodeWordLengthFunction: "dryNodeWordLength",
    leafWordFunction: "dryLeafWord",
    leafWordLengthFunction: "dryLeafWordLength",
  });
  assert.equal((`${traversal}\n${primary}`.match(/var<storage, read> svoStructuralControl/g) ?? []).length, 1);
  assert.doesNotMatch(primary, /svoStructuralNodes\[/);
  assert.match(primary, /dryNodeWord\(nodeBase\+2u\)/);
  assert.match(primary, /dryLeafWordLength\(\)/);
  assert.match(primary, /var<private> svoStructuralFluidPrimaryExpectedGeneration:u32/);
  assert.match(primary, /domain\.settings\.y!=0u&&generation!=domain\.settings\.y/,
    "every nested point lookup must compare against the trace-captured generation");
});

test("validated fine phi exclusively refines primary samples and preserves the ten-storage adapter contract", () => {
  const primary = createSvoStructuralFluidPrimaryWGSL({
    fineSampleFunction: "rendererFineSample",
    fineGradientFunction: "rendererFineGradient",
  });
  assert.match(primary, /svoResolveFluidPhi\(SvoFluidFieldValue\(coarse\.phi_m[^]*SvoFluidFieldValue\(fine\.phi_m,fine\.valid\)\)/,
    "one resolver gives valid fine data exclusive ownership and exact coarse fallback");
  assert.match(primary, /previous\.owner/);
  assert.match(primary, /root\.owner/);
  assert.match(primary, /owner==SVO_FLUID_OWNER_FINE[^]*rendererFineGradient/);
  assert.match(svoDrySceneShader, /dryFluidFieldSource\(first\.fieldOwner\)/,
    "the existing G-buffer metadata distinguishes coarse and fine roots without ABI growth");
  assert.equal((svoDrySceneShader.match(/var<storage,\s*read>/g) ?? []).length, SVO_STRUCTURAL_FLUID_PRIMARY_STORAGE_BINDINGS);

  const paramsWords = new Uint32Array(40);
  paramsWords.set([2, 1, 1, 8], 0);
  paramsWords.set([4, 2, 2, 2], 4);
  paramsWords.set([2, 1, 1, 2], 8);
  paramsWords.set([4, 2, 2, 0], 12);
  paramsWords.set([2, 2, 2, 16], 16);
  paramsWords.set([64, 64, 66, 128], 20);
  const capability = {
    arena: { buffer: { size: 4096 } as GPUBuffer, size: 4096 },
    params: { buffer: { size: 160 } as GPUBuffer },
    statusWord: SVO_FINE_PHI_CONTROL_WORDS.status,
    acceptedStructuralGenerationWord: SVO_FINE_PHI_CONTROL_WORDS.acceptedStructuralGeneration,
    acceptedFineGenerationWord: SVO_FINE_PHI_CONTROL_WORDS.acceptedFineGeneration,
    pageGenerationOffsetWords: 64,
    ownerPageTableOffsetWords: 66,
    payloadOffsetWords: 128,
    paramsWords,
    publicationMirrorWords: 8,
    coarseFallbackRequired: true,
    directWaterOwnership: false,
  } satisfies SvoFineFluidGpuCapability;
  assert.equal(canConsumeSvoFineFluidCapability(capability), true);
  assert.equal(canConsumeSvoFineFluidCapability({ ...capability, directWaterOwnership: true } as unknown as SvoFineFluidGpuCapability), false);
  assert.equal(canConsumeSvoFineFluidCapability({ ...capability, paramsWords: new Uint32Array(39) }), false);
});

test("production dry renderer binds fluid only through an explicit non-overlapping primary mode", () => {
  const valid = source();
  const legacy: SparseVoxelDrySceneData = candidateBackedDrySceneFixture;
  const diagnostic: SparseVoxelDrySceneData = { ...legacy, fluidPrimaryMode: "coarse-opaque-diagnostic" };
  assert.equal(canEncodeSparseVoxelDryScene(valid, legacy), true);
  assert.equal(canEncodeSparseVoxelDryScene(valid, diagnostic), true);
  assert.equal(canEncodeSparseVoxelDryScene({ ...valid, structural: { ...valid.structural!, fields: { ...valid.structural!.fields, coarseFluid: { ...valid.structural!.fields.coarseFluid, residency: "unavailable" } } } }, legacy), true,
    "legacy compositor must not depend on the new coarse field");
  assert.equal(canEncodeSparseVoxelDryScene({ ...valid, structural: { ...valid.structural!, fields: { ...valid.structural!.fields, coarseFluid: { ...valid.structural!.fields.coarseFluid, residency: "unavailable" } } } }, diagnostic), false,
    "an explicitly requested direct fluid path must fall back before encoding malformed residency");
  assert.deepEqual(SVO_DRY_SCENE_PARAMS_LAYOUT, {
    sizeBytes: 368,
    terrainWordOffset: 24,
    terrainMaterialWordOffset: 28,
    materialPublicationWordOffset: 32,
    fluidDomainWordOffset: 36,
    primitiveCandidateWordOffset: 40,
    finePhiWordOffset: 44,
    nodeMipWordOffset: 84,
    nodeMipAtlasWordOffset: 88,
  });
  assert.match(svoDrySceneShader, /@group\(0\) @binding\(11\) var<storage,read> svoStructuralGeometry/);
  assert.match(svoDrySceneShader, /@group\(0\) @binding\(12\) var<storage,read> svoStructuralLeafStates/);
  assert.match(svoDrySceneShader, /dry\.fluidDomainMode\.w==1u/);
  assert.match(svoDrySceneShader, /SVO_GBUFFER_FIELD_FLUID_COARSE/);
  assert.match(svoDrySceneShader, /dryFluidPrimaryFailure=DRY_GBUFFER_WORK_EXHAUSTED/);
  assert.match(svoDrySceneShader, /dryFluidPrimaryFailure=DRY_GBUFFER_INVALID_FIELD/);
});
