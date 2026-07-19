import assert from "node:assert/strict";
import test from "node:test";
import {
  SPARSE_BRICK_GPU_LAYOUT,
  packSparseBrickPlan,
  planSparseBrickOctree,
  type SparseBrickCoordinate,
} from "../lib/sparse-brick-octree";
import {
  SVO_STRUCTURAL_CLEARED_PHI_MIN,
  SVO_STRUCTURAL_POINT_LOOKUP_MAX_VISITS,
  gradientSvoStructuralCoarseFluid,
  lookupSvoStructuralCoarseFluidCell,
  sampleSvoStructuralCoarseFluidAtWorld,
  sampleSvoStructuralFluidExclusive,
  svoStructuralCoarseFluidSamplingWGSL,
  type SvoStructuralFluidPackedFixture,
} from "../lib/svo-fluid-structural-sampling";
import { FLUID_BRICK_RESIDENT } from "../lib/webgpu-fluid-brick-residency";
import { SPARSE_VOXEL_PUBLICATION_STATE, SPARSE_VOXEL_VALID_FIELDS } from "../lib/webgpu-voxel-debug";

const close = (actual: number, expected: number, tolerance = 1e-6) => {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} should be within ${tolerance} of ${expected}`);
};

const cellSize = [0.5, 1, 2] as const;
const worldOrigin = [10, 20, 30] as const;

function phiForCell(cell: readonly [number, number, number]): number {
  const position = cell.map((component, axis) => worldOrigin[axis] + (component + 0.5) * cellSize[axis]);
  return position[0] + 2 * position[1] + 3 * position[2] - 150;
}

function fixtureFor(coordinates: readonly SparseBrickCoordinate[] = [{ x: 0, y: 0, z: 0 }, { x: 1, y: 0, z: 0 }]): SvoStructuralFluidPackedFixture {
  const plan = planSparseBrickOctree(coordinates, { brickSize: 4, maximumDepth: 1 });
  const packed = packSparseBrickPlan(plan, 7);
  const geometry = new Float32Array(plan.voxelCount * 4);
  for (const leaf of plan.leaves) {
    for (let z = 0; z < plan.brickSize; z += 1) for (let y = 0; y < plan.brickSize; y += 1) for (let x = 0; x < plan.brickSize; x += 1) {
      const localIndex = x + y * plan.brickSize + z * plan.brickSize ** 2;
      const cell = [
        leaf.coordinate.x * plan.brickSize + x,
        leaf.coordinate.y * plan.brickSize + y,
        leaf.coordinate.z * plan.brickSize + z,
      ] as const;
      geometry[(leaf.voxelOffset + localIndex) * 4] = phiForCell(cell);
    }
  }
  const control = new Uint32Array(32);
  control[SPARSE_BRICK_GPU_LAYOUT.controlWords.publishedNodes] = plan.nodes.length;
  control[SPARSE_BRICK_GPU_LAYOUT.controlWords.publishedLeaves] = plan.leaves.length;
  control[SPARSE_BRICK_GPU_LAYOUT.controlWords.publishedVoxels] = plan.voxelCount;
  control[SPARSE_BRICK_GPU_LAYOUT.controlWords.brickSize] = plan.brickSize;
  const publicationState = new Uint32Array(SPARSE_VOXEL_PUBLICATION_STATE.strideBytes / 4);
  publicationState[SPARSE_VOXEL_PUBLICATION_STATE.completeGeneration] = 7;
  publicationState[SPARSE_VOXEL_PUBLICATION_STATE.validFields] = SPARSE_VOXEL_VALID_FIELDS.topology
    | SPARSE_VOXEL_VALID_FIELDS.coarseFluid;
  publicationState[SPARSE_VOXEL_PUBLICATION_STATE.topologyRevision] = 3;
  publicationState[SPARSE_VOXEL_PUBLICATION_STATE.coarseFluidRevision] = 6;
  return {
    control,
    nodes: packed.nodes,
    leaves: packed.leaves,
    geometry,
    fluidLeafStates: new Uint32Array(plan.leaves.length).fill(FLUID_BRICK_RESIDENT),
    publicationState,
    domain: {
      worldOrigin_m: worldOrigin,
      cellSize_m: cellSize,
      dimensionsCells: [8, 4, 4],
      brickSize: 4,
      maximumDepth: 1,
    },
    expectedCompleteGeneration: 7,
  };
}

function copyFixture(source = fixtureFor()): SvoStructuralFluidPackedFixture {
  return {
    ...source,
    control: new Uint32Array(source.control),
    nodes: new Uint32Array(source.nodes),
    leaves: new Uint32Array(source.leaves),
    geometry: new Float32Array(source.geometry),
    fluidLeafStates: new Uint32Array(source.fluidLeafStates),
    publicationState: new Uint32Array(source.publicationState),
  };
}

test("packed lookup follows actual node/leaf offsets and returns geometry.x in negative-inside metres", () => {
  const source = fixtureFor();
  const first = lookupSvoStructuralCoarseFluidCell(source, [0, 0, 0]);
  assert.equal(first.status, "valid");
  if (first.status !== "valid") return;
  close(first.phi_m, phiForCell([0, 0, 0]));
  assert.ok(first.phi_m < 0);
  assert.equal(first.voxelIndex, 0);
  assert.equal(first.nodeVisits, 2);

  const secondBrick = lookupSvoStructuralCoarseFluidCell(source, [4, 0, 0]);
  assert.equal(secondBrick.status, "valid");
  if (secondBrick.status !== "valid") return;
  close(secondBrick.phi_m, phiForCell([4, 0, 0]));
  assert.notEqual(secondBrick.leafIndex, first.leafIndex);
  assert.equal(secondBrick.voxelIndex, 64);
});

test("world lookup uses anisotropic cell extents and returns explicit outside-domain misses", () => {
  const source = fixtureFor();
  const sample = sampleSvoStructuralCoarseFluidAtWorld(source, [10.74, 21.9, 35.9]);
  assert.equal(sample.status, "valid");
  if (sample.status === "valid") {
    assert.deepEqual(sample.cell, [1, 1, 2]);
    close(sample.phi_m, phiForCell([1, 1, 2]));
  }
  assert.deepEqual(sampleSvoStructuralCoarseFluidAtWorld(source, [9.999, 21, 31]), {
    status: "miss", reason: "outside-domain", nodeVisits: 0,
  });
  assert.deepEqual(lookupSvoStructuralCoarseFluidCell(source, [8, 0, 0]), {
    status: "miss", reason: "outside-domain", nodeVisits: 0,
  });
});

test("anisotropic gradients traverse cross-brick neighbors and use one-sided domain boundaries", () => {
  const source = fixtureFor();
  const acrossBrick = gradientSvoStructuralCoarseFluid(source, [11.75, 21.5, 33]);
  assert.equal(acrossBrick.status, "valid");
  if (acrossBrick.status !== "valid") return;
  assert.deepEqual(acrossBrick.center.cell, [3, 1, 1]);
  close(acrossBrick.gradient[0], 1);
  close(acrossBrick.gradient[1], 2);
  close(acrossBrick.gradient[2], 3);
  close(acrossBrick.normal?.[0] ?? 0, 1 / Math.sqrt(14));
  close(acrossBrick.normal?.[1] ?? 0, 2 / Math.sqrt(14));
  close(acrossBrick.normal?.[2] ?? 0, 3 / Math.sqrt(14));
  assert.deepEqual(acrossBrick.schemes, ["central", "central", "central"]);

  const boundary = gradientSvoStructuralCoarseFluid(source, [10.25, 20.5, 31]);
  assert.equal(boundary.status, "valid");
  if (boundary.status === "valid") {
    close(boundary.gradient[0], 1);
    close(boundary.gradient[1], 2);
    close(boundary.gradient[2], 3);
    assert.deepEqual(boundary.schemes, ["forward", "forward", "forward"]);
  }
});

test("sparse missing branches are misses while malformed topology is invalid", () => {
  const sparse = fixtureFor([{ x: 0, y: 0, z: 0 }]);
  assert.deepEqual(lookupSvoStructuralCoarseFluidCell(sparse, [4, 0, 0]), {
    status: "miss", reason: "missing-branch", nodeVisits: 1,
  });

  const malformedCount = copyFixture();
  malformedCount.nodes[5] += 1;
  assert.equal(lookupSvoStructuralCoarseFluidCell(malformedCount, [0, 0, 0]).status, "invalid");
  const malformedBacklink = copyFixture();
  malformedBacklink.leaves[0] = 0;
  const backlink = lookupSvoStructuralCoarseFluidCell(malformedBacklink, [0, 0, 0]);
  assert.equal(backlink.status, "invalid");
  if (backlink.status === "invalid") assert.equal(backlink.reason, "invalid-topology");
});

test("publication validity, complete generation, overflow, residency, and payload are independent gates", () => {
  const unpublished = copyFixture();
  unpublished.publicationState[SPARSE_VOXEL_PUBLICATION_STATE.completeGeneration] = 0;
  assert.equal(lookupSvoStructuralCoarseFluidCell(unpublished, [0, 0, 0]).status, "invalid");

  const stale = copyFixture();
  stale.expectedCompleteGeneration = 8;
  const staleResult = lookupSvoStructuralCoarseFluidCell(stale, [0, 0, 0]);
  assert.equal(staleResult.status, "invalid");
  if (staleResult.status === "invalid") assert.equal(staleResult.reason, "generation-mismatch");

  const noField = copyFixture();
  noField.publicationState[SPARSE_VOXEL_PUBLICATION_STATE.validFields] &= ~SPARSE_VOXEL_VALID_FIELDS.coarseFluid;
  const noFieldResult = lookupSvoStructuralCoarseFluidCell(noField, [0, 0, 0]);
  assert.equal(noFieldResult.status, "invalid");
  if (noFieldResult.status === "invalid") assert.equal(noFieldResult.reason, "missing-valid-fields");

  const overflow = copyFixture();
  overflow.control[SPARSE_BRICK_GPU_LAYOUT.controlWords.overflowFlags] = 1;
  const overflowResult = lookupSvoStructuralCoarseFluidCell(overflow, [0, 0, 0]);
  assert.equal(overflowResult.status, "invalid");
  if (overflowResult.status === "invalid") assert.equal(overflowResult.reason, "source-overflow");

  const nonresident = copyFixture();
  nonresident.fluidLeafStates[0] = 0;
  const nonresidentResult = lookupSvoStructuralCoarseFluidCell(nonresident, [0, 0, 0]);
  assert.equal(nonresidentResult.status, "invalid");
  if (nonresidentResult.status === "invalid") assert.equal(nonresidentResult.reason, "nonresident-leaf");

  const cleared = copyFixture();
  cleared.geometry[0] = SVO_STRUCTURAL_CLEARED_PHI_MIN;
  const clearedResult = lookupSvoStructuralCoarseFluidCell(cleared, [0, 0, 0]);
  assert.equal(clearedResult.status, "invalid");
  if (clearedResult.status === "invalid") assert.equal(clearedResult.reason, "invalid-payload");
});

test("future fine pages exclusively own valid samples and otherwise fall back to structural coarse phi", () => {
  const source = fixtureFor();
  const position = [10.25, 20.5, 31] as const;
  assert.deepEqual(sampleSvoStructuralFluidExclusive(source, position, () => ({ phi_m: -0.001, valid: true })), {
    status: "valid", owner: "fine", phi_m: -0.001,
  });
  const fallback = sampleSvoStructuralFluidExclusive(source, position, () => ({ phi_m: 0, valid: false }));
  assert.equal(fallback.status, "valid");
  if (fallback.status === "valid") {
    assert.equal(fallback.owner, "coarse");
    close(fallback.phi_m, phiForCell([0, 0, 0]));
  }
  assert.deepEqual(sampleSvoStructuralFluidExclusive(source, position, () => ({ phi_m: Number.NaN, valid: true })), {
    status: "invalid", owner: "none", reason: "invalid-fine-sample",
  });

  const brokenCoarse = copyFixture(source);
  brokenCoarse.fluidLeafStates.fill(0);
  assert.equal(sampleSvoStructuralFluidExclusive(brokenCoarse, position, () => ({ phi_m: -0.2, valid: true })).owner, "fine");
});

test("layout constants and bounded binding-free WGSL mirror match the structural source ABI", () => {
  assert.equal(SVO_STRUCTURAL_POINT_LOOKUP_MAX_VISITS, 24);
  assert.equal(SPARSE_BRICK_GPU_LAYOUT.nodeStrideBytes, 32);
  assert.equal(SPARSE_BRICK_GPU_LAYOUT.leafStrideBytes, 16);
  assert.equal(SPARSE_BRICK_GPU_LAYOUT.geometryStrideBytes, 16);
  assert.equal(SPARSE_BRICK_GPU_LAYOUT.controlWords.publishedNodes, 0);
  assert.equal(SPARSE_BRICK_GPU_LAYOUT.controlWords.publishedLeaves, 1);
  assert.equal(SPARSE_BRICK_GPU_LAYOUT.controlWords.publishedVoxels, 2);
  assert.equal(SPARSE_BRICK_GPU_LAYOUT.controlWords.brickSize, 11);
  assert.equal(SPARSE_BRICK_GPU_LAYOUT.controlWords.overflowFlags, 12);
  assert.equal(FLUID_BRICK_RESIDENT, 1);
  assert.doesNotMatch(svoStructuralCoarseFluidSamplingWGSL, /@group|@binding/);
  assert.match(svoStructuralCoarseFluidSamplingWGSL, /const SVO_STRUCTURAL_MAX_VISITS:u32=24u/);
  assert.match(svoStructuralCoarseFluidSamplingWGSL, /let phi=svoStructuralGeometry\[voxelIndex\]\.x/);
  assert.match(svoStructuralCoarseFluidSamplingWGSL, /svoStructuralLeafStates\[leafIndex\]&SVO_STRUCTURAL_RESIDENT/);
  assert.match(svoStructuralCoarseFluidSamplingWGSL, /SVO_STRUCTURAL_REQUIRED_FIELDS/);
  assert.match(svoStructuralCoarseFluidSamplingWGSL, /for\(var visits=1u;visits<=SVO_STRUCTURAL_MAX_VISITS/);
  assert.match(svoStructuralCoarseFluidSamplingWGSL, /fn svoStructuralCoarseFluidGradientWorld/);
  assert.match(svoStructuralCoarseFluidSamplingWGSL, /domain\.cellSize_m\[axis\]/);
  assert.match(svoStructuralCoarseFluidSamplingWGSL, /svoStructuralCoarseFluidCell\(domain,backwardCell\)/);
});
