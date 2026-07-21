import assert from "node:assert/strict";
import test from "node:test";
import {
  OCTREE_MGPCG_PRECONDITIONER_KIND,
  OCTREE_SECTION43_BOUNDARY_BAND_LAYERS,
  OCTREE_SECTION43_MAXIMUM_PCG_ITERATIONS,
  OCTREE_SECTION43_BOUNDARY_SMOOTHING_ITERATIONS,
  WebGPUOctreeMGPCG,
  octreeMGPCGShader,
  planOctreeMGPCG,
} from "../lib/webgpu-octree-mgpcg";
import { octreePowerOperatorShader } from "../lib/webgpu-octree-power-operator";
import { WebGPUOctreeProjection } from "../lib/webgpu-octree";

test("MGPCG allocation is bounded by compact rows and sparse levels", () => {
  const small = planOctreeMGPCG({ dimensions: [64, 48, 32], rowCapacity: 10_000, maximumLeafSize: 16 });
  const wide = planOctreeMGPCG({ dimensions: [1024, 48, 32], rowCapacity: 10_000, maximumLeafSize: 16 });
  assert.equal(small.rowCapacity, wide.rowCapacity);
  assert.ok(wide.hierarchyLevelCount > small.hierarchyLevelCount);
  assert.equal(wide.hierarchyBytes, wide.hierarchyLevelCount * wide.rowCapacity * 5 * 4);
  assert.ok(!("cellCount" in wide), "planner must not expose a finest-domain allocation");
});

test("Section 4.3 hybrid is opt-in, separately sized, and requires an explicit SPD L1 V-cycle", () => {
  const aggregate = planOctreeMGPCG({ dimensions: [64, 48, 32], rowCapacity: 10_000, maximumLeafSize: 16 });
  const hybrid = planOctreeMGPCG({ dimensions: [64, 48, 32], rowCapacity: 10_000, maximumLeafSize: 16,
    preconditionerKind: "section43-hybrid" });
  assert.equal(aggregate.preconditionerKind, "aggregate");
  assert.equal(aggregate.hybridBytes, 0);
  assert.equal(hybrid.preconditionerKind, "section43-hybrid");
  assert.equal(hybrid.hierarchyBytes, 0, "aggregate hierarchy is not the hybrid's L1 V-cycle");
  assert.equal(hybrid.hybridBytes, hybrid.vectorBytes * 6);
  assert.throws(() => new WebGPUOctreeMGPCG({} as GPUDevice, {
    leafHeaders: {} as GPUBuffer, leafEntries: {} as GPUBuffer, rowCount: {} as GPUBuffer,
  }, { dimensions: [64, 48, 32], rowCapacity: 10_000, maximumLeafSize: 16,
    maximumIterations: 16, preconditionerKind: "section43-hybrid" }),
  /requires an explicit SPD first-order V-cycle/);
});

test("Section 4.3 hybrid has a three-layer boundary/transition band and paired k=8 L2 smoothing", () => {
  assert.equal(OCTREE_SECTION43_BOUNDARY_BAND_LAYERS, 3);
  assert.equal(OCTREE_SECTION43_BOUNDARY_SMOOTHING_ITERATIONS, 8);
  assert.equal(OCTREE_SECTION43_MAXIMUM_PCG_ITERATIONS, 16);
  assert.match(octreeMGPCGShader, /boundaryGap=h\.diagonal-offDiagonalSum/);
  assert.match(octreeMGPCGShader, /headers\[e\.row\]\.size!=h\.size/);
  assert.match(octreeMGPCGShader, /dilateHybridBandAtoB/);
  assert.match(octreeMGPCGShader, /dilateHybridBandBtoA/);
  assert.match(octreeMGPCGShader, /formHybridL1Residual/);
  assert.match(octreeMGPCGShader, /addHybridL1Correction/);
  const source = WebGPUOctreeMGPCG.toString();
  assert.match(source, /firstOrderVCycle|encodeCorrection/);
  assert.match(source, /OCTREE_SECTION43_BOUNDARY_SMOOTHING_ITERATIONS/);
  assert.match(WebGPUOctreeProjection.toString(), /3 graph-ring band approximation/,
    "the visible solver label must not describe graph dilation as an exact three-voxel paper band");
});

test("authoritative power projection constructs and selects the Section 4.3 L1 V-cycle", () => {
  const source = WebGPUOctreeProjection.toString();
  assert.match(source, /new WebGPUOctreeFirstOrderVCycle/);
  assert.match(source, /preconditionerKind:\s*this\.powerPolicy\.authoritative\s*\?\s*"section43-hybrid"/);
  assert.match(source, /firstOrderVCycle\?\.encodeCapture\(encoder\)/,
    "L1 rows must be captured before power publication replaces the shared CSR");
});

test("matrix-free aggregate PCG uses additive transfers and GPU-only convergence", () => {
  assert.equal(OCTREE_MGPCG_PRECONDITIONER_KIND, "additive-geometric-aggregate-diagonal");
  assert.match(octreeMGPCGShader, /value-=e\.coefficient\*fieldValue/);
  assert.match(octreeMGPCGShader, /restrictResidual/);
  assert.match(octreeMGPCGShader, /prolongateCorrection/);
  assert.match(octreeMGPCGShader, /r\*preconditioned\[row\]/);
  assert.match(octreeMGPCGShader, /atomicStore\(&control\[1\],1u\)/);
  assert.doesNotMatch(WebGPUOctreeProjection.prototype.encode.toString(), /mapAsync|getMappedRange/);
});

test("aggregate preconditioner is not mislabeled as the paper Section 4.3 hybrid V-cycle", () => {
  const source = WebGPUOctreeProjection.toString();
  assert.doesNotMatch(source, /matrix-free MGPCG/);
  assert.doesNotMatch(source, /"multigrid"/);
  assert.doesNotMatch(octreeMGPCGShader, /GhostValuePropagate|GhostValueAccumulate/);
});

test("power projection publication is gated by MGPCG success", () => {
  assert.match(octreePowerOperatorShader, /preparePowerProjectionMGPCG/);
  assert.match(octreePowerOperatorShader, /atomicLoad\(&solverControl\[0\]\)!=0u/);
  assert.match(octreePowerOperatorShader, /atomicLoad\(&solverControl\[1\]\)==0u/);
  assert.match(WebGPUOctreeProjection.prototype.encode.toString(), /this\.mgpcg.*encode/);
});
