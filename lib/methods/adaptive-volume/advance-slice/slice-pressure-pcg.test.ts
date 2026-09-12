import assert from "node:assert/strict";
import test from "node:test";
import { solveSlicePressurePCG } from "./slice-pressure-pcg";

const f = Math.fround;

function bits(values: Float32Array): number[] {
  return [...new Uint32Array(values.buffer, values.byteOffset, values.length)];
}

test("Chronopoulos-Gear fixture preserves production recurrence and delayed gate", () => {
  // This matrix fixture is deliberately expressed as the face-jump SPD image,
  // while expected f32 words were captured from the production-order kernel
  // transcription. They catch replacement by ordinary two-reduction PCG.
  const matrix = [[4, -1, 0], [-1, 4, -1], [0, -1, 3]] as const;
  const result = solveSlicePressurePCG({
    diagonal: new Float32Array([4, 4, 3]),
    rhs: new Float32Array([15, 10, 10]),
    pressure: new Float32Array([0.25, -0.5, 0.75]),
    member: new Uint8Array([1, 1, 1]),
    maximumIterations: 16,
    relativeTolerance: 1e-6,
    apply(input, output) {
      for (let row = 0; row < 3; row++) {
        let sum = 0;
        for (let column = 0; column < 3; column++) {
          sum = f(sum + f(matrix[row]![column]! * input[column]!));
        }
        output[row] = sum;
      }
    },
  });
  assert.deepEqual(bits(result.pressure), [1084227584, 1084227583, 1084227584]);
  assert.deepEqual(bits(result.residual), [0, 905969664, 0]);
  assert.equal(result.iterations, 6);
  assert.equal(result.firstToleranceIteration, 6);
  assert.equal(result.curvatureRecoveries, 1);
  assert.equal(result.initialTrueResidualSquared, 403.8125);
  assert.equal(result.finalTrueResidualSquared, 3.637978807091713e-12);
  assert.equal(result.recursiveResidualSquared, 7.553497241886739e-30);
  assert.equal(result.residualDrift, true);
  assert.equal(result.converged, true);
});

test("an exact solution waits for the eight-iteration true-residual gate", () => {
  const result = solveSlicePressurePCG({
    diagonal: new Float32Array([2, 3]), rhs: new Float32Array([4, 9]),
    pressure: new Float32Array(2), member: new Uint8Array([1, 1]),
    maximumIterations: 9, relativeTolerance: 1e-6,
    apply(input, output) {
      output[0] = f(2 * input[0]!); output[1] = f(3 * input[1]!);
    },
  });
  assert.deepEqual(bits(result.pressure), [1073741824, 1077936128]);
  assert.equal(result.iterations, 1,
    "the second recurrence detects zero curvature; encoded work remains gated until iteration 8");
  assert.equal(result.encodedIterations, 9);
  assert.equal(result.firstToleranceIteration, 1);
  assert.equal(result.curvatureRecoveries, 1);
  assert.equal(result.finalTrueResidualSquared, 0);
});

test("curvature recovery re-applies the operator and can close the device gate", () => {
  let applications = 0;
  const result = solveSlicePressurePCG({
    diagonal: new Float32Array([1]), rhs: new Float32Array([1]),
    pressure: new Float32Array([0]), member: new Uint8Array([1]),
    maximumIterations: 9, relativeTolerance: 1e-6,
    apply(_input, output) { applications++; output[0] = 0; },
  });
  assert.equal(result.iterations, 0);
  assert.equal(result.curvatureRecoveries, 1);
  assert.equal(result.converged, false);
  assert.equal(result.finalTrueResidualSquared, 1);
  assert.equal(applications, 6,
    "warm image, initial receipt, failed seed, cadence receipt, failed recovery and final receipt");
});

test("compact execution order controls the production f32 reduction tree", () => {
  const count = 130;
  let seed = 1;
  const rhs = Float32Array.from({ length: count }, () => {
    seed = (1664525 * seed + 1013904223) >>> 0;
    const exponent = seed % 25 - 12;
    return f(((seed >>> 8) & 1 ? -1 : 1) * 2 ** exponent * (1 + (seed & 255) / 256));
  });
  const ascending = Uint32Array.from({ length: count }, (_, id) => id);
  const reversed = Uint32Array.from(ascending).reverse();
  const run = (executionOrder: Uint32Array) => solveSlicePressurePCG({
    diagonal: new Float32Array(count).fill(1), rhs,
    pressure: new Float32Array(count), member: new Uint8Array(count).fill(1),
    executionOrder, maximumIterations: 0, relativeTolerance: 0,
    apply(_input, output) { output.fill(0); },
  });
  assert.equal(run(ascending).rhsSquared, 160549664);
  assert.equal(run(reversed).rhsSquared, 160549680);
});
