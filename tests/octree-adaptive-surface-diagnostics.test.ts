import assert from "node:assert/strict";
import test from "node:test";
import { analyzeAdaptiveSurfaceFeatureGeometry, analyzeAdaptiveSurfacePublication } from
  "../lib/octree-adaptive-surface-diagnostics";

const bits = (value: number) => new Uint32Array(new Float32Array([value]).buffer)[0]!;

test("adaptive surface audit identifies coarse crossings and renderer/constraint drift", () => {
  const leaves = new Uint32Array(16);
  leaves.set([0, 0, 0, 2], 0);
  leaves.set([0, 1, 2, 3, 4, 5, 6, 7], 8);
  const nodes = new Uint32Array(8 * 4);
  for (let node = 0; node < 8; node += 1) {
    const x = node & 1 ? 2 : 0, y = node & 2 ? 2 : 0, z = node & 4 ? 2 : 0;
    nodes[4 * node] = x + 5 * (y + 5 * z);
  }
  const phi = new Uint32Array(16);
  [-1, 1, -1, 1, -1, 1, -1, 1].forEach((value, node) => {
    phi[2 * node] = bits(value); phi[2 * node + 1] = bits(value);
  });
  const renderer = new Uint32Array(24);
  renderer[2] = 1; renderer[14] = 0;
  for (let corner = 0; corner < 8; corner += 1) renderer[16 + corner] = phi[2 * corner]!;
  const result = analyzeAdaptiveSurfacePublication({
    graphControl: Uint32Array.from([1, 1, 8]), phiControl: new Uint32Array(20),
    leaves, nodalPhi: phi, nodes, constraints: new Uint32Array(8 * 12), renderer,
    dimensions: [4, 4, 4],
  });
  assert.deepEqual(result.interfaceLeafCountsBySize, { 2: 1 });
  assert.equal(result.coarseInterfaceLeafCount, 1);
  assert.equal(result.coarseStrictInterfaceLeafCount, 1);
  assert.equal(result.coarseTouchingInterfaceLeafCount, 0);
  assert.equal(result.maximumStoredConstraintError, 0);
  assert.equal(result.maximumRendererCornerError, 0);
});

test("adaptive surface audit evaluates hanging nodes before comparing renderer corners", () => {
  const leaves = new Uint32Array(16);
  leaves.set([0, 0, 0, 1], 0);
  leaves.set([0, 1, 2, 3, 4, 5, 6, 7], 8);
  const nodes = new Uint32Array(8 * 4);
  const phi = new Uint32Array(8 * 2);
  for (let node = 0; node < 8; node += 1) {
    nodes[4 * node] = node;
    phi[2 * node] = bits(node === 1 ? -1 : node === 2 ? 1 : 2);
  }
  phi[0] = bits(7);
  const constraints = new Uint32Array(8 * 12);
  constraints[1] = 2;
  constraints[2] = 2;
  constraints[4] = 1;
  constraints[5] = 2;
  constraints[8] = 1;
  constraints[9] = 1;
  const renderer = new Uint32Array(24);
  renderer[2] = 1;
  renderer[14] = 0;
  renderer[16] = bits(0.25);
  for (let corner = 1; corner < 8; corner += 1) {
    renderer[16 + corner] = phi[2 * corner]!;
  }
  const result = analyzeAdaptiveSurfacePublication({
    graphControl: Uint32Array.from([1, 1, 8]), phiControl: new Uint32Array(20),
    leaves, nodalPhi: phi, nodes, constraints, renderer, dimensions: [1, 1, 1],
  });
  assert.equal(result.constrainedNodeCount, 1);
  assert.equal(result.maximumStoredConstraintError, 7,
    "stored hanging payload is diagnosed against its two-master interpolation");
  assert.equal(result.maximumRendererCornerError, 0.25,
    "renderer comparison uses the constrained value, not the stale hanging payload");
});

test("adaptive surface feature audit reconstructs top heights and active zero-set cubes", () => {
  const leaves = new Uint32Array(16);
  leaves.set([0, 0, 0, 2], 0);
  leaves.set([0, 1, 2, 3, 4, 5, 6, 7], 8);
  const nodes = new Uint32Array(8 * 4);
  const phi = new Uint32Array(8 * 2);
  for (let corner = 0; corner < 8; corner += 1) {
    phi[2 * corner] = bits(corner & 2 ? 1 : -1);
  }
  const result = analyzeAdaptiveSurfaceFeatureGeometry({
    graphControl: Uint32Array.from([1, 1, 8]), phiControl: new Uint32Array(20),
    leaves, nodalPhi: phi, nodes, constraints: new Uint32Array(8 * 12),
    renderer: new Uint32Array(8), dimensions: [2, 2, 2],
  }, 1, [
    { name: "face-center", points: [[1, 1]] },
    { name: "corners", points: [[0, 0], [0, 2], [2, 0], [2, 2]] },
  ]);
  assert.equal(result.coveredNodalSamples, 27);
  assert.equal(result.missingNodalSamples, 0);
  assert.equal(result.maximumSharedNodeMismatch, 0);
  assert.equal(result.activeCubeCount, 8);
  assert.deepEqual(result.zeroSetExtentsCells,
    { minimum: [0, 0, 0], maximum: [2, 2, 2] });
  assert.deepEqual(result.topFeatures["face-center"], {
    sampleCount: 1,
    phiAtReferenceTop: { minimum: 0, maximum: 0, mean: 0 },
    surfaceHeightCells: { minimum: 1, maximum: 1, mean: 1 },
    meanRetreatFromReferenceTopCells: 0,
  });
  assert.equal(result.topFeatures.corners?.sampleCount, 4);
  assert.equal(result.topFeatures.corners?.surfaceHeightCells.mean, 1);
});
