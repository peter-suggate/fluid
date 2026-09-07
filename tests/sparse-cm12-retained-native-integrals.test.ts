import assert from "node:assert/strict";
import test from "node:test";
import { compileRetainedNativeIntegrals } from "../lib/methods/adaptive-mass/sparse-cm12-retained-native-integrals";
import { compileRetainedSceneFineMeans, retainedSceneDensity, type RetainedSceneBox } from "../lib/methods/adaptive-mass/sparse-cm12-retained-scene-density";

function catalogue(boxes: readonly RetainedSceneBox[]) {
  const words = new Uint32Array(16 + 8 * boxes.length), floats = new Float32Array(words.buffer);
  words[6] = 16;
  boxes.forEach((box, i) => {
    const at = 16 + 8 * i;
    for (let axis = 0; axis < 3; axis++) {
      floats[at + axis] = (box.lower[axis] + box.upper[axis]) / 2;
      floats[at + 4 + axis] = box.upper[axis] - box.lower[axis];
    }
  });
  return words;
}
const h = .125, dimensions = [16, 16, 16] as const;
const pool = retainedSceneDensity({ generation: 1, transitionWidth: h,
  domain: { lower: [-1, 0, -1], upper: [1, 2, 1] },
  primitives: [{ kind: "quadratic-height", center: [0, .43, 0], curvature: [0, 0, 0] }] });
const fine = { means: compileRetainedSceneFineMeans(pool, dimensions, h), dimensions };
const close = (a: number, b: number, tolerance = 3e-7) => assert.ok(Math.abs(a - b) < tolerance, `${a} != ${b}`);

// Independent one-dimensional positive-part antiderivative, with hard domain
// clipping. No retained evaluator or source finest moments enter this oracle.
function poolAmount(box: RetainedSceneBox) {
  const p = pool.primitives[0]; assert.ok(p.kind === "quadratic-height");
  const low = Math.max(0, box.lower[1] * h), high = Math.min(2, box.upper[1] * h);
  const xWidth = Math.max(0, Math.min(16, box.upper[0]) - Math.max(0, box.lower[0])) * h;
  const zWidth = Math.max(0, Math.min(16, box.upper[2]) - Math.max(0, box.lower[2])) * h;
  if (high <= low) return 0;
  const positiveSquare = (x: number) => Math.max(x, 0) ** 2;
  const upperRamp = p.center[1] + pool.transitionWidth / 2, lowerRamp = p.center[1] - pool.transitionWidth / 2;
  const integral = (positiveSquare(upperRamp - low) - positiveSquare(upperRamp - high)
    - positiveSquare(lowerRamp - low) + positiveSquare(lowerRamp - high)) / (2 * pool.transitionWidth);
  return xWidth * zWidth * integral;
}

test("all integer native rungs restrict a non-grid-aligned pool to its analytic physical amounts", () => {
  for (const width of [8, 4, 2, 1, 4, 8]) {
    const boxes: RetainedSceneBox[] = [];
    for (let z = 0; z < 16; z += width) for (let y = 0; y < 16; y += width) for (let x = 0; x < 16; x += width) {
      boxes.push({ lower: [x, y, z], upper: [x + width, y + width, z + width] });
    }
    const means = compileRetainedNativeIntegrals(pool, catalogue(boxes), boxes.length, h, fine);
    let amount = 0;
    for (let i = 0; i < boxes.length; i++) {
      const volume = (width * h) ** 3;
      close(means[i], poolAmount(boxes[i]) / volume); amount += means[i] * volume;
    }
    close(amount, poolAmount({ lower: [0, 0, 0], upper: [16, 16, 16] }), 1e-8);
  }
});

test("clipped, anisotropic and outside-domain native boxes retain their true full-box denominator", () => {
  const boxes: RetainedSceneBox[] = [
    { lower: [13, 0, 11], upper: [16, 7, 16] },
    { lower: [-2, -1, 4], upper: [4, 3, 8] },
    { lower: [15, 2, -3], upper: [19, 7, 2] },
    { lower: [20, 0, 20], upper: [24, 4, 24] },
  ];
  const means = compileRetainedNativeIntegrals(pool, catalogue(boxes), boxes.length, h, fine);
  boxes.forEach((box, i) => {
    const volume = box.upper.reduce((v, hi, axis) => v * (hi - box.lower[axis]) * h, 1);
    close(means[i], poolAmount(box) / volume);
  });
});

test("fractional native bounds require physical-field integration, never rounded finest moments", () => {
  const boxes: RetainedSceneBox[] = [{ lower: [0, 3, 0], upper: [.5, 3.5, .5] },
    { lower: [.5, 3.5, .5], upper: [1, 4, 1] }];
  const topology = catalogue(boxes);
  assert.throws(() => compileRetainedNativeIntegrals(pool, topology, 2, h, fine), /not aligned/);
  const actual = compileRetainedNativeIntegrals(pool, topology, 2, h);
  boxes.forEach((box, i) => close(actual[i], poolAmount(box) / (h / 2) ** 3));
  assert.ok(Math.abs(actual[0] - actual[1]) > .3, "different subcell halves contain different physical amounts");
});

test("malformed moment lattices and degenerate native boxes fail before publishing NaN", () => {
  const topology = catalogue([{ lower: [0, 0, 0], upper: [1, 1, 1] }]);
  assert.throws(() => compileRetainedNativeIntegrals(pool, topology, 1, h, { means: new Float32Array(1), dimensions }), /does not match/);
  assert.throws(() => compileRetainedNativeIntegrals(pool, topology.slice(0, 20), 1, h, fine), /Truncated/);
  const floats = new Float32Array(topology.buffer); floats[20] = 0;
  assert.throws(() => compileRetainedNativeIntegrals(pool, topology, 1, h, fine), /Invalid retained native geometry/);
});
