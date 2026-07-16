import assert from "node:assert/strict";
import test from "node:test";
import { simulationMethods } from "../lib/methods";
import { quadtreeTallCellMethod } from "../lib/methods/quadtree-tall-cell";
import { nextQuadtreeIterationBudget, quadtreeDispatchShader, quadtreeIterationBudget, quadtreeTallCellProjectionShader } from "../lib/webgpu-quadtree-tall-cell";
import { packedQuadtreeRootMap, quadtreeConstructionShader, quadtreeSurfaceShader } from "../lib/webgpu-quadtree-builder";

test("the old optical-layer method is replaced by quadtree tall cells", () => {
  assert.ok(simulationMethods.includes(quadtreeTallCellMethod));
  assert.ok(!simulationMethods.some((method) => method.id === "adaptive-optical-layer"));
  assert.equal(quadtreeTallCellMethod.shortLabel, "Adaptive");
  assert.match(quadtreeTallCellMethod.detail, /T-junction/);
});

test("quadtree updates evaluate sizing, subdivide, and smooth on WebGPU", () => {
  for (const entry of ["evaluateSizing", "refine", "smoothTopology", "sampleLeafProfiles"]) assert.match(quadtreeConstructionShader, new RegExp(`fn ${entry}\\b`));
  assert.match(quadtreeConstructionShader, /for \(var y = 0u; y < params\.dims\.y; y \+= 1u\)/, "sizing must vertically reduce each column on the GPU");
  assert.match(quadtreeConstructionShader, /sizingField\[index2\(q\)\] = maximum/);
  assert.match(quadtreeConstructionShader, /neighborTooFine/);
  assert.match(quadtreeConstructionShader, /demand > 1\.0 \/ testedWidth/);
  const roots = packedQuadtreeRootMap(16, 8, 8);
  assert.equal(roots.length, 128);
  assert.equal((roots[0] >>> 20) & 1023, 8);
  assert.equal((roots[15] >>> 20) & 1023, 8);
});

test("resident phi uses saved-grid semi-Lagrangian advection with narrow-band volume feedback", () => {
  for (const entry of ["advectLevelSet", "reduceVolume", "seedDistance", "jumpFlood", "finalizeDistance"]) assert.match(quadtreeSurfaceShader, new RegExp(`fn ${entry}\\b`));
  assert.match(quadtreeSurfaceShader, /fn centredMacVelocity/);
  assert.match(quadtreeSurfaceShader, /let departure = vec3f\(gid\) - centredMacVelocity\(vec3i\(gid\)\) \* params\.cellAndDt\.w \/ params\.cellAndDt\.xyz/);
  assert.match(quadtreeSurfaceShader, /own\.x \+ loadVelocity\(q - vec3i\(1, 0, 0\)\)\.x/);
  assert.match(quadtreeSurfaceShader, /distanceSeedsOut\[index3\(gid\)\] = select\(0xffffffffu, packSeed\(gid\), crosses\)/);
  assert.match(quadtreeSurfaceShader, /fn volumeCorrectedPhi/);
  assert.match(quadtreeSurfaceShader, /value - params\.control\.x \* h \* params\.cellAndDt\.w/);
  assert.match(quadtreeSurfaceShader, /abs\(value\) < 1\.5 \* h/);
  assert.match(quadtreeSurfaceShader, /value \/ \(4\.0 \* params\.cellAndDt\.y\)/);
  assert.match(quadtreeSurfaceShader, /atomicAdd\(&reductions\[0\]/);
  assert.doesNotMatch(quadtreeSurfaceShader, /reverseLevelSet|correctLevelSet|MacCormack/);
  assert.doesNotMatch(quadtreeSurfaceShader, /volumeIn|loadVolume|alpha/);
  assert.match(quadtreeConstructionShader, /fn effectiveWet[\s\S]*return loadAdvancedPhi\(clamp3\(q\)\) < 0\.0/);
  assert.match(quadtreeConstructionShader, /columnProfiles\[leaf \* params\.dims\.y \+ y\]/);
});

test("pressure iterations consume precomputed face masks, fluxes, and row activity", () => {
  assert.match(quadtreeTallCellProjectionShader, /fn refreshRows/);
  assert.match(quadtreeTallCellProjectionShader, /liquidMask \|= 1u << slot/);
  assert.match(quadtreeTallCellProjectionShader, /face\.flux = face\.weights\.x \* faceVelocity\(face\) \+ face\.solidFlux/);
  assert.match(quadtreeTallCellProjectionShader, /fn dofActive\(row: u32\) -> bool \{ return state\[stateIndex\(row, ACTIVE_FLAG\)\] != 0u; \}/);
  assert.match(quadtreeTallCellProjectionShader, /rhs \+= item\.coefficient \* face\.flux/);
  const iterationPath = quadtreeTallCellProjectionShader.slice(quadtreeTallCellProjectionShader.indexOf("fn rowProduct"), quadtreeTallCellProjectionShader.indexOf("fn cellIndex"));
  assert.doesNotMatch(iterationPath, /faceSamplePhi|faceVelocity/);
  assert.match(iterationPath, /matrixCoefficient\(entry\) \* stateF\(matrixNode\(entry\), DIRECTION\)/);
});

test("WebGPU pressure path is variational PCG rather than Jacobi pressure smoothing", () => {
  for (const entry of ["initialize", "multiply", "applyStep", "finishIteration", "project"]) {
    assert.match(quadtreeTallCellProjectionShader, new RegExp(`fn ${entry}\\b`));
  }
  assert.match(quadtreeDispatchShader, /fn updateDispatch\b/);
  assert.match(quadtreeTallCellProjectionShader, /rowProduct/);
  assert.match(quadtreeTallCellProjectionShader, /faceGradient/);
  assert.match(quadtreeTallCellProjectionShader, /matrixBaseCoefficient\(entry\) \* face\.weights\.y/);
  assert.match(quadtreeTallCellProjectionShader, /alias SolverField = u32/);
  assert.match(quadtreeTallCellProjectionShader, /fn preconditionJacobi/);
  assert.match(quadtreeTallCellProjectionShader, /fn preconditionLine/);
  assert.match(quadtreeTallCellProjectionShader, /fn preconditionPolynomialStart/);
  for (const entry of ["applyStepPartial", "applyStepFinalize", "applyStepUpdate", "finishIterationPartial", "finishIterationFinalize", "finishIterationUpdate"]) assert.match(quadtreeTallCellProjectionShader, new RegExp(`fn ${entry}\\b`));
  assert.doesNotMatch(quadtreeTallCellProjectionShader, /atomic(Add|Max|Min)/);
});

test("pressure command budgets follow convergence feedback without lowering the hard cap", () => {
  const initial = quadtreeIterationBudget(12_322, { pressureIterations: 96 });
  assert.equal(initial.hardBudget, 445);
  assert.equal(initial.encodedBudget, 445);
  const converged = nextQuadtreeIterationBudget(initial, 200, true);
  assert.equal(converged.hardBudget, 445);
  assert.ok(converged.encodedBudget < initial.encodedBudget);
  const capped = nextQuadtreeIterationBudget({ ...converged, encodedBudget: 128 }, 128, false);
  assert.equal(capped.encodedBudget, 256);
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
  assert.match(quadtreeTallCellProjectionShader, /face\.flux = face\.weights\.x \* faceVelocity\(face\) \+ face\.solidFlux/);
  assert.match(quadtreeTallCellProjectionShader, /addStateF\(dof, MATRIX_DIRECTION, sum\)/);
});

test("corrected inner ghost velocity averages every replaced vertical face", () => {
  // Every vertical face, ghost or single-cell, averages its leaf's full x/z
  // footprint; only horizontal faces take the transverse-row branch.
  assert.match(quadtreeTallCellProjectionShader, /if \(axis != 1u\) \{/);
  assert.match(quadtreeTallCellProjectionShader, /for \(var y = face\.bounds\.z; y < face\.bounds\.w; y \+= 1u\)/);
  assert.match(quadtreeTallCellProjectionShader, /sum \/ max\(1\.0, count\)/);
});
