import assert from "node:assert/strict";
import test from "node:test";
import {
  solveOctreePipelinedPCG,
  type OctreePipelinedPCGLinearMap,
} from "../lib/octree-pipelined-pcg";

function denseMap(matrix: readonly (readonly number[])[]): OctreePipelinedPCGLinearMap {
  return (input) => matrix.map((row) =>
    row.reduce((sum, value, column) => sum + value * input[column], 0));
}

function multiply(matrix: readonly (readonly number[])[], vector: readonly number[]): number[] {
  return matrix.map((row) =>
    row.reduce((sum, value, column) => sum + value * vector[column], 0));
}

/** Independent conventional PCG, deliberately not the merged recurrence. */
function conventionalPCG(
  matrix: readonly (readonly number[])[],
  inverseDiagonal: readonly number[],
  rightHandSide: readonly number[],
): number[] {
  const dot = (left: readonly number[], right: readonly number[]) =>
    left.reduce((sum, value, index) => sum + value * right[index], 0);
  const solution = rightHandSide.map(() => 0);
  const residual = [...rightHandSide];
  let preconditioned = residual.map((value, index) => inverseDiagonal[index] * value);
  let direction = [...preconditioned], gamma = dot(residual, preconditioned);
  for (let iteration = 0; iteration < rightHandSide.length; iteration += 1) {
    const image = multiply(matrix, direction);
    const alpha = gamma / dot(direction, image);
    for (let index = 0; index < solution.length; index += 1) {
      solution[index] += alpha * direction[index];
      residual[index] -= alpha * image[index];
    }
    preconditioned = residual.map((value, index) => inverseDiagonal[index] * value);
    const nextGamma = dot(residual, preconditioned);
    if (Math.sqrt(dot(residual, residual)) < 1e-13) break;
    const beta = nextGamma / gamma;
    direction = preconditioned.map((value, index) => value + beta * direction[index]);
    gamma = nextGamma;
  }
  return solution;
}

test("direct-curvature float64 PCG matches an independent conventional SPD solve", () => {
  const matrix = [
    [7, -2, 0, 0, 0],
    [-2, 6, -1, 0, 0],
    [0, -1, 5, -1, 0],
    [0, 0, -1, 4, -1],
    [0, 0, 0, -1, 3],
  ];
  const inverseDiagonal = matrix.map((row, index) => 1 / row[index]);
  const rightHandSide = [1.25, -0.75, 2.5, 0.125, -1.5];
  const reference = conventionalPCG(matrix, inverseDiagonal, rightHandSide);
  const result = solveOctreePipelinedPCG({
    rightHandSide,
    applyOperator: denseMap(matrix),
    applyPreconditioner: (input) =>
      input.map((value, index) => inverseDiagonal[index] * value),
  }, { maximumIterations: 10, relativeTolerance: 1e-12 });

  assert.equal(result.accepted, true, result.diagnostics.failure);
  assert.equal(result.diagnostics.converged, true);
  assert.ok(result.diagnostics.iterations <= matrix.length);
  assert.equal(result.diagnostics.reductions,
    Math.max(1, 2 * result.diagnostics.iterations),
    "each nonterminal update owns residual and direct-curvature reductions");
  assert.equal(result.diagnostics.history.length, result.diagnostics.iterations + 1);
  assert.ok(result.solution.every((value, index) => Math.abs(value - reference[index]) < 2e-12));
  assert.ok(result.diagnostics.residualNorm <= result.diagnostics.stoppingThreshold);
  assert.ok(result.residual.every(Number.isFinite));
});

test("direct-curvature PCG solves a manufactured variable-coefficient Poisson row set", () => {
  const count = 17;
  const faceWeights = Array.from({ length: count + 1 }, (_, face) =>
    0.75 + ((face * 13) % 11) / 9);
  const matrix = Array.from({ length: count }, () => new Array<number>(count).fill(0));
  for (let row = 0; row < count; row += 1) {
    matrix[row][row] = faceWeights[row] + faceWeights[row + 1] + 0.2;
    if (row > 0) matrix[row][row - 1] = -faceWeights[row];
    if (row + 1 < count) matrix[row][row + 1] = -faceWeights[row + 1];
  }
  const manufactured = Array.from({ length: count }, (_, row) =>
    Math.sin((row + 1) * 0.37) + 0.2 * Math.cos((row + 1) * 0.91));
  const rightHandSide = multiply(matrix, manufactured);
  const result = solveOctreePipelinedPCG({
    rightHandSide,
    applyOperator: denseMap(matrix),
    applyPreconditioner: (input) =>
      input.map((value, row) => value / matrix[row][row]),
  }, { maximumIterations: 40, relativeTolerance: 5e-13 });

  assert.equal(result.accepted, true, result.diagnostics.failure);
  assert.ok(result.solution.every((value, row) => Math.abs(value - manufactured[row]) < 2e-11),
    `manufactured error ${Math.max(...result.solution.map((value, row) => Math.abs(value - manufactured[row])))}`);
  assert.equal(result.diagnostics.reductions, 2 * result.diagnostics.iterations);
  assert.ok(result.diagnostics.history.every((entry, index) => entry.iteration === index));
});

test("oracle validates an initial guess and reports zero outer iterations", () => {
  const matrix = [[3, -1], [-1, 2]];
  const solution = [0.75, -1.25];
  const rightHandSide = multiply(matrix, solution);
  const result = solveOctreePipelinedPCG({
    rightHandSide,
    applyOperator: denseMap(matrix),
    initialGuess: solution,
  }, { relativeTolerance: 1e-14 });

  assert.equal(result.accepted, true);
  assert.equal(result.diagnostics.iterations, 0);
  assert.equal(result.diagnostics.reductions, 1);
  assert.equal(result.diagnostics.residualNorm, 0);
});

test("oracle rejects non-linear preconditioners before entering the recurrence", () => {
  const result = solveOctreePipelinedPCG({
    rightHandSide: [1, 2, -1],
    applyOperator: denseMap([[4, -1, 0], [-1, 3, -1], [0, -1, 2]]),
    applyPreconditioner: (input) => input.map((value) => value * Math.abs(value)),
  });

  assert.equal(result.accepted, false);
  assert.equal(result.diagnostics.failure, "preconditioner-is-not-linear");
  assert.equal(result.diagnostics.iterations, 0);
  assert.equal(result.diagnostics.reductions, 0);
});

test("oracle fails closed on indefinite curvature and on an exhausted budget", () => {
  const indefinite = solveOctreePipelinedPCG({
    rightHandSide: [1, 0],
    applyOperator: denseMap([[1, 0], [0, -1]]),
  });
  assert.equal(indefinite.accepted, false);
  assert.equal(indefinite.diagnostics.failure, "operator-is-not-positive");
  assert.equal(indefinite.diagnostics.iterations, 0);

  const matrix = [[4, -1, 0], [-1, 4, -1], [0, -1, 3]];
  const exhausted = solveOctreePipelinedPCG({
    rightHandSide: [1, 2, 3],
    applyOperator: denseMap(matrix),
  }, { maximumIterations: 1, relativeTolerance: 1e-15 });
  assert.equal(exhausted.accepted, false);
  assert.equal(exhausted.diagnostics.failure, "maximum-iterations");
  assert.equal(exhausted.diagnostics.iterations, 1);
  assert.equal(exhausted.diagnostics.reductions, 2);
  assert.ok(exhausted.diagnostics.residualNorm > exhausted.diagnostics.stoppingThreshold);
});

test("malformed vector and tolerance contracts throw deterministically", () => {
  const identity: OctreePipelinedPCGLinearMap = (input) => input;
  assert.throws(() => solveOctreePipelinedPCG({
    rightHandSide: [1, Number.NaN],
    applyOperator: identity,
  }), /non-finite/);
  assert.throws(() => solveOctreePipelinedPCG({
    rightHandSide: [1, 2],
    initialGuess: [0],
    applyOperator: identity,
  }), /length 2/);
  assert.throws(() => solveOctreePipelinedPCG({
    rightHandSide: [1],
    applyOperator: identity,
  }, { relativeTolerance: -1 }), /non-negative/);
});
