import assert from "node:assert/strict";
import test from "node:test";
import { simulationMethods } from "../lib/methods";
import { quadtreeTallCellMethod } from "../lib/methods/quadtree-tall-cell";
import { quadtreeTallCellProjectionShader } from "../lib/webgpu-quadtree-tall-cell";

test("the old optical-layer method is replaced by quadtree tall cells", () => {
  assert.ok(simulationMethods.includes(quadtreeTallCellMethod));
  assert.ok(!simulationMethods.some((method) => method.id === "adaptive-optical-layer"));
  assert.equal(quadtreeTallCellMethod.shortLabel, "Adaptive");
  assert.match(quadtreeTallCellMethod.detail, /T-junction/);
});

test("WebGPU pressure path is variational PCG rather than Jacobi pressure smoothing", () => {
  for (const entry of ["initialize", "multiply", "reduceDenominator", "updateSolution", "reduceResidual", "updateDirection", "project"]) {
    assert.match(quadtreeTallCellProjectionShader, new RegExp(`fn ${entry}\\b`));
  }
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

test("corrected inner ghost velocity averages every replaced vertical face", () => {
  assert.match(quadtreeTallCellProjectionShader, /let ghost = faceGhost\(face\)/);
  assert.match(quadtreeTallCellProjectionShader, /for \(var y = face\.bounds\.z; y < face\.bounds\.w; y \+= 1u\)/);
  assert.match(quadtreeTallCellProjectionShader, /sum \/ max\(1\.0, count\)/);
});
