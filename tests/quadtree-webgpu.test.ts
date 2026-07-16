import assert from "node:assert/strict";
import test from "node:test";
import { simulationMethods } from "../lib/methods";
import { quadtreeTallCellMethod } from "../lib/methods/quadtree-tall-cell";
import { quadtreeDispatchShader, quadtreeTallCellProjectionShader } from "../lib/webgpu-quadtree-tall-cell";
import { packedQuadtreeRootMap, quadtreeConstructionShader } from "../lib/webgpu-quadtree-builder";

test("the old optical-layer method is replaced by quadtree tall cells", () => {
  assert.ok(simulationMethods.includes(quadtreeTallCellMethod));
  assert.ok(!simulationMethods.some((method) => method.id === "adaptive-optical-layer"));
  assert.equal(quadtreeTallCellMethod.shortLabel, "Adaptive");
  assert.match(quadtreeTallCellMethod.detail, /T-junction/);
});

test("quadtree updates evaluate sizing, subdivide, and smooth on WebGPU", () => {
  for (const entry of ["advectLevelSet", "seedDistance", "jumpFlood", "finalizeDistance", "evaluateSizing", "refine", "smoothTopology"]) assert.match(quadtreeConstructionShader, new RegExp(`fn ${entry}\\b`));
  assert.match(quadtreeConstructionShader, /for \(var y = 0u; y < params\.dims\.y; y \+= 1u\)/, "sizing must vertically reduce each column on the GPU");
  assert.match(quadtreeConstructionShader, /sizingField\[index2\(q\)\] = maximum/);
  assert.match(quadtreeConstructionShader, /neighborTooFine/);
  assert.match(quadtreeConstructionShader, /demand > 1\.0 \/ testedWidth/);
  const roots = packedQuadtreeRootMap(16, 8, 8);
  assert.equal(roots.length, 128);
  assert.equal((roots[0] >>> 20) & 1023, 8);
  assert.equal((roots[15] >>> 20) & 1023, 8);
});

test("WebGPU pressure path is variational PCG rather than Jacobi pressure smoothing", () => {
  for (const entry of ["initialize", "multiply", "reduceDenominator", "updateSolution", "reduceResidual", "updateDirection", "project"]) {
    assert.match(quadtreeTallCellProjectionShader, new RegExp(`fn ${entry}\\b`));
  }
  assert.match(quadtreeDispatchShader, /fn updateDispatch\b/);
  assert.match(quadtreeTallCellProjectionShader, /rowProduct/);
  assert.match(quadtreeTallCellProjectionShader, /faceGradient/);
  assert.match(quadtreeTallCellProjectionShader, /face\.weights\.y \* faceGradient\(face\)/);
  assert.match(quadtreeTallCellProjectionShader, /struct SolverState/);
  assert.doesNotMatch(quadtreeTallCellProjectionShader, /atomic(Add|Max|Min)/);
});

test("cubic pressure projection conservatively prolongs the solved variational face correction", () => {
  assert.match(quadtreeTallCellProjectionShader, /fn solvedFaceGradient\(face: Face\)/);
  assert.match(quadtreeTallCellProjectionShader, /face\.weights\.y \/ face\.weights\.x/);
  assert.match(quadtreeTallCellProjectionShader, /value\[axis\] -= fluidScale \* solvedFaceGradient\(face\)/);
  assert.doesNotMatch(quadtreeTallCellProjectionShader, /cellPressure\(plus\) - cellPressure\(gid\)/);
});

test("monolithic rigid coupling and MLS mapping are wired into the solve", () => {
  assert.match(quadtreeTallCellProjectionShader, /fn coupleReduce/);
  assert.match(quadtreeTallCellProjectionShader, /fn coupleApply/);
  assert.match(quadtreeTallCellProjectionShader, /fn coupleImpulse/);
  assert.match(quadtreeTallCellProjectionShader, /fn mlsRowGradient/);
  assert.match(quadtreeTallCellProjectionShader, /face\.weights\.x \* faceVelocity\(face\) \+ face\.solidFlux/);
  assert.match(quadtreeTallCellProjectionShader, /state\[dof\]\.matrixDirection \+= sum/);
});

test("corrected inner ghost velocity averages every replaced vertical face", () => {
  // Every vertical face, ghost or single-cell, averages its leaf's full x/z
  // footprint; only horizontal faces take the transverse-row branch.
  assert.match(quadtreeTallCellProjectionShader, /if \(axis != 1u\) \{/);
  assert.match(quadtreeTallCellProjectionShader, /for \(var y = face\.bounds\.z; y < face\.bounds\.w; y \+= 1u\)/);
  assert.match(quadtreeTallCellProjectionShader, /sum \/ max\(1\.0, count\)/);
});
