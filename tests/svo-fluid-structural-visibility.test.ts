import assert from "node:assert/strict";
import test from "node:test";
import { SPARSE_BRICK_GPU_LAYOUT, packSparseBrickPlan, planSparseBrickOctree } from "../lib/sparse-brick-octree";
import {
  SVO_STRUCTURAL_INTERPOLATION_MAX_CORNERS,
  SVO_STRUCTURAL_VISIBILITY_DEFAULT_MAX_STEPS,
  SVO_STRUCTURAL_VISIBILITY_DEFAULT_NODE_BUDGET,
  SVO_STRUCTURAL_VISIBILITY_MAX_NODE_BUDGET,
  SVO_STRUCTURAL_VISIBILITY_MAX_STEPS,
  sampleSvoStructuralCoarseFluidTrilinear,
  svoStructuralFluidVisibilityWGSL,
  traceSvoStructuralCoarseFluid,
} from "../lib/svo-fluid-structural-visibility";
import type { SvoStructuralFluidPackedFixture } from "../lib/svo-fluid-structural-sampling";
import { FLUID_BRICK_RESIDENT } from "../lib/webgpu-fluid-brick-residency";
import { SPARSE_VOXEL_PUBLICATION_STATE, SPARSE_VOXEL_VALID_FIELDS } from "../lib/webgpu-voxel-debug";

const close = (actual: number, expected: number, tolerance = 1e-4) => {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} should be within ${tolerance} of ${expected}`);
};

function sampledFixture(
  dimensionsCells: readonly [number, number, number],
  cellSize_m: readonly [number, number, number],
  phiAt: (position_m: readonly [number, number, number]) => number,
): SvoStructuralFluidPackedFixture {
  const brickSize = 4;
  const brickDimensions = dimensionsCells.map((size) => Math.ceil(size / brickSize));
  const maximumCoordinate = Math.max(...brickDimensions) - 1;
  const maximumDepth = maximumCoordinate === 0 ? 0 : Math.ceil(Math.log2(maximumCoordinate + 1));
  const coordinates = [];
  for (let z = 0; z < brickDimensions[2]; z += 1) for (let y = 0; y < brickDimensions[1]; y += 1) for (let x = 0; x < brickDimensions[0]; x += 1) {
    coordinates.push({ x, y, z });
  }
  const plan = planSparseBrickOctree(coordinates, { brickSize, maximumDepth });
  const packed = packSparseBrickPlan(plan, 13);
  const geometry = new Float32Array(plan.voxelCount * 4);
  for (const leaf of plan.leaves) {
    for (let z = 0; z < brickSize; z += 1) for (let y = 0; y < brickSize; y += 1) for (let x = 0; x < brickSize; x += 1) {
      const cell = [leaf.coordinate.x * brickSize + x, leaf.coordinate.y * brickSize + y, leaf.coordinate.z * brickSize + z] as const;
      const position = cell.map((component, axis) => (component + 0.5) * cellSize_m[axis]) as [number, number, number];
      const local = x + y * brickSize + z * brickSize ** 2;
      geometry[(leaf.voxelOffset + local) * 4] = phiAt(position);
    }
  }
  const control = new Uint32Array(32);
  control[SPARSE_BRICK_GPU_LAYOUT.controlWords.publishedNodes] = plan.nodes.length;
  control[SPARSE_BRICK_GPU_LAYOUT.controlWords.publishedLeaves] = plan.leaves.length;
  control[SPARSE_BRICK_GPU_LAYOUT.controlWords.publishedVoxels] = plan.voxelCount;
  control[SPARSE_BRICK_GPU_LAYOUT.controlWords.brickSize] = brickSize;
  const publicationState = new Uint32Array(SPARSE_VOXEL_PUBLICATION_STATE.strideBytes / 4);
  publicationState[SPARSE_VOXEL_PUBLICATION_STATE.completeGeneration] = 13;
  publicationState[SPARSE_VOXEL_PUBLICATION_STATE.validFields] = SPARSE_VOXEL_VALID_FIELDS.topology | SPARSE_VOXEL_VALID_FIELDS.coarseFluid;
  publicationState[SPARSE_VOXEL_PUBLICATION_STATE.topologyRevision] = 4;
  publicationState[SPARSE_VOXEL_PUBLICATION_STATE.coarseFluidRevision] = 12;
  return {
    control,
    nodes: packed.nodes,
    leaves: packed.leaves,
    geometry,
    fluidLeafStates: new Uint32Array(plan.leaves.length).fill(FLUID_BRICK_RESIDENT),
    publicationState,
    domain: { worldOrigin_m: [0, 0, 0], cellSize_m, dimensionsCells, brickSize, maximumDepth },
    expectedCompleteGeneration: 13,
  };
}

function copied(source: SvoStructuralFluidPackedFixture): SvoStructuralFluidPackedFixture {
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

test("trilinear coarse sampling is continuous and exact for an anisotropic plane across brick boundaries", () => {
  const source = sampledFixture([8, 4, 4], [0.5, 1, 2], (position) => position[0] + 2 * position[1] - 3);
  const left = sampleSvoStructuralCoarseFluidTrilinear(source, [1.999, 1.5, 3]);
  const right = sampleSvoStructuralCoarseFluidTrilinear(source, [2.001, 1.5, 3]);
  assert.equal(left.status, "valid");
  assert.equal(right.status, "valid");
  if (left.status !== "valid" || right.status !== "valid") return;
  close(left.phi_m, 1.999 + 3 - 3);
  close(right.phi_m, 2.001 + 3 - 3);
  assert.ok(left.leafIndices.length > 1);
  assert.ok(right.leafIndices.length > 1);
});

test("domain-only nearest clamping is zero-set safe while required retired interior corners invalidate", () => {
  const source = sampledFixture([8, 4, 4], [1, 1, 1], (position) => position[0] - 4);
  const boundary = sampleSvoStructuralCoarseFluidTrilinear(source, [0, 1.5, 1.5]);
  assert.equal(boundary.status, "valid");
  if (boundary.status === "valid") {
    assert.equal(boundary.fallback, "domain-clamp");
    close(boundary.phi_m, -3.5);
  }

  const retired = copied(source);
  const secondLeaf = sampleSvoStructuralCoarseFluidTrilinear(source, [4.5, 1.5, 1.5]);
  assert.equal(secondLeaf.status, "valid");
  if (secondLeaf.status !== "valid") return;
  retired.fluidLeafStates[secondLeaf.anchorLeafIndex] = 0;
  const crossBoundary = sampleSvoStructuralCoarseFluidTrilinear(retired, [3.9, 1.5, 1.5]);
  assert.equal(crossBoundary.status, "invalid");
  if (crossBoundary.status === "invalid") assert.equal(crossBoundary.reason, "nonresident-leaf");
});

test("leaf-aware trace finds the nearest cross-brick plane root with generation/source diagnostics", () => {
  const source = sampledFixture([8, 4, 4], [1, 1, 1], (position) => position[0] - 4);
  const result = traceSvoStructuralCoarseFluid(source, {
    origin_m: [0.5, 1.5, 1.5], direction: [7, 0, 0], tMax_m: 7,
  }, { step_m: 10, tTolerance_m: 1e-7, phiTolerance_m: 1e-7 });
  assert.equal(result.status, "hit");
  if (result.status !== "hit") return;
  close(result.t_m, 3.5, 2e-5);
  close(result.position_m[0], 4, 2e-5);
  close(result.normal[0], 1, 2e-5);
  assert.equal(result.insideFluidAtStart, true);
  assert.equal(result.diagnostics.completeGeneration, 13);
  assert.equal(result.diagnostics.coarseFluidRevision, 12);
  assert.equal(result.diagnostics.source, "structural-coarse");
  assert.ok(result.diagnostics.crossLeafSamples > 0);
});

test("sampled sphere trace finds nearest entry and inside-fluid exit", () => {
  const center = [4, 4, 4] as const;
  const radius = 2;
  const sphere = sampledFixture([16, 16, 16], [0.5, 0.5, 0.5], (position) => (
    Math.hypot(position[0] - center[0], position[1] - center[1], position[2] - center[2]) - radius
  ));
  const entry = traceSvoStructuralCoarseFluid(sphere, {
    origin_m: [0.25, 4, 4], direction: [1, 0, 0], tMax_m: 7.5,
  }, { step_m: 0.2, tTolerance_m: 1e-5, phiTolerance_m: 1e-5 });
  assert.equal(entry.status, "hit");
  if (entry.status === "hit") {
    close(entry.position_m[0], 2, 0.04);
    assert.equal(entry.insideFluidAtStart, false);
    assert.ok(entry.normal[0] < -0.99);
  }

  const exit = traceSvoStructuralCoarseFluid(sphere, {
    origin_m: center, direction: [2, 0, 0], tMax_m: 4,
  }, { step_m: 0.2, tTolerance_m: 1e-5, phiTolerance_m: 1e-5 });
  assert.equal(exit.status, "hit");
  if (exit.status === "hit") {
    close(exit.position_m[0], 6, 0.04);
    assert.equal(exit.insideFluidAtStart, true);
    assert.ok(exit.normal[0] > 0.99);
  }
});

test("invalid neighbors and stale publication abort tracing without inventing a crossing", () => {
  const source = sampledFixture([8, 4, 4], [1, 1, 1], (position) => position[0] - 4);
  const retired = copied(source);
  const target = sampleSvoStructuralCoarseFluidTrilinear(source, [4.5, 1.5, 1.5]);
  assert.equal(target.status, "valid");
  if (target.status !== "valid") return;
  retired.fluidLeafStates[target.anchorLeafIndex] = 0;
  const invalid = traceSvoStructuralCoarseFluid(retired, {
    origin_m: [1.5, 1.5, 1.5], direction: [1, 0, 0], tMax_m: 5,
  });
  assert.equal(invalid.status, "invalid-field");
  assert.equal(invalid.diagnostics.failureReason, "nonresident-leaf");

  const stale = copied(source);
  stale.expectedCompleteGeneration = 14;
  const staleResult = traceSvoStructuralCoarseFluid(stale, {
    origin_m: [1.5, 1.5, 1.5], direction: [1, 0, 0], tMax_m: 5,
  });
  assert.equal(staleResult.status, "invalid-field");
  assert.equal(staleResult.diagnostics.failureReason, "generation-mismatch");
  assert.equal(staleResult.diagnostics.completeGeneration, 13);
});

test("topology and step work caps terminate explicitly", () => {
  const source = sampledFixture([8, 4, 4], [1, 1, 1], () => 1);
  const nodeCapped = traceSvoStructuralCoarseFluid(source, {
    origin_m: [0.5, 1.5, 1.5], direction: [1, 0, 0], tMax_m: 7,
  }, { maximumNodeVisits: 24 });
  assert.equal(nodeCapped.status, "work-exhausted");
  assert.equal(nodeCapped.diagnostics.failureReason, "node-budget-exhausted");
  assert.ok(nodeCapped.diagnostics.topologyNodeVisits <= 24);

  const stepCapped = traceSvoStructuralCoarseFluid(source, {
    origin_m: [0.5, 1.5, 1.5], direction: [1, 0, 0], tMax_m: 7,
  }, { maximumSteps: 1, maximumNodeVisits: 2_000, step_m: 0.1 });
  assert.equal(stepCapped.status, "work-exhausted");
  assert.equal(stepCapped.steps, 1);
});

test("fixed work caps and binding-free composition helpers are explicit", () => {
  assert.equal(SVO_STRUCTURAL_INTERPOLATION_MAX_CORNERS, 8);
  assert.equal(SVO_STRUCTURAL_VISIBILITY_DEFAULT_MAX_STEPS, 512);
  assert.equal(SVO_STRUCTURAL_VISIBILITY_MAX_STEPS, 65_536);
  assert.equal(SVO_STRUCTURAL_VISIBILITY_DEFAULT_NODE_BUDGET, 65_536);
  assert.equal(SVO_STRUCTURAL_VISIBILITY_MAX_NODE_BUDGET, 1_048_576);
  assert.doesNotMatch(svoStructuralFluidVisibilityWGSL, /@group|@binding/);
  assert.match(svoStructuralFluidVisibilityWGSL, /fn svoStructuralCoarseFluidTrilinear/);
  assert.match(svoStructuralFluidVisibilityWGSL, /fn svoFluidSampleAt/);
  assert.match(svoStructuralFluidVisibilityWGSL, /fn svoFluidRefineZero/);
  assert.match(svoStructuralFluidVisibilityWGSL, /fn svoFluidGradientNormal/);
  assert.match(svoStructuralFluidVisibilityWGSL, /SVO_STRUCTURAL_MAX_VISITS:u32=24u/);
});
