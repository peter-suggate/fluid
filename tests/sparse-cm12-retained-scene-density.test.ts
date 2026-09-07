import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene } from "../lib/core/model";
import { createCoarseFirstPoolImpactHalfScene, createCoarseFirstPoolImpactQuarterScene } from "../lib/core/scenes";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice-dimensions";
import {
  compileRetainedSceneDensity, compileRetainedSceneFineMeans, evaluateRetainedSceneDensity,
  evaluateRetainedScenePhi, integrateRetainedSceneDensity, integrateRetainedSceneVertical,
  packRetainedSceneDensity, retainedSceneDensity, type RetainedSceneDensity, type RetainedScenePoint, type RetainedSceneFineMeansReceipt,
} from "../lib/methods/adaptive-mass/sparse-cm12-retained-scene-density";

const close = (actual: number, expected: number, tolerance = 1e-12) => assert.ok(Math.abs(actual - expected) <= tolerance,
  `expected ${actual} to be within ${tolerance} of ${expected}`);
function field(primitives: RetainedSceneDensity["primitives"], width = .05) {
  return retainedSceneDensity({ generation: 3, transitionWidth: width,
    domain: { lower: [-1, 0, -1], upper: [1, 2, 1] }, primitives });
}
function radialAmount(f: RetainedSceneDensity) {
  const p = f.primitives.find(p => p.kind === "ellipsoid"); assert.ok(p?.kind === "ellipsoid");
  const a = f.transitionWidth / Math.min(...p.radii);
  return 4 * Math.PI * p.radii[0] * p.radii[1] * p.radii[2] / (15 * a)
    * ((1 + a) ** 2.5 - Math.max(0, 1 - a) ** 2.5);
}

test("authored pool and sphere compile into immutable numeric density with an explicit retained width", () => {
  const scene = createCoarseFirstPoolImpactQuarterScene(), compiled = compileRetainedSceneDensity(scene); assert.ok(compiled);
  assert.equal(compiled.transitionWidth, Math.fround(.05));
  assert.equal(compiled.primitives.length, 2);
  const before = packRetainedSceneDensity(compiled);
  scene.container.fillFraction = .9; scene.fluid.initialLiquidVolumes = [];
  assert.deepEqual(packRetainedSceneDensity(compiled), before);
  assert.ok(Object.isFrozen(compiled)); assert.ok(Object.isFrozen(compiled.primitives));
  assert.ok(compiled.primitives.every(p => Object.isFrozen(p)));
  const pool = compiled.primitives[0]; assert.equal(pool.kind, "quadratic-height");
  if (pool.kind === "quadratic-height") close(evaluateRetainedSceneDensity(compiled, [.7, pool.center[1], .7]), .5);
  const sphere = compiled.primitives[1]; assert.equal(sphere.kind, "ellipsoid");
  if (sphere.kind === "ellipsoid") {
    close(evaluateRetainedSceneDensity(compiled, [sphere.center[0] + sphere.radii[0], sphere.center[1], sphere.center[2]]), .5);
    assert.ok(evaluateRetainedSceneDensity(compiled, [0, sphere.center[1], 0]) > .999);
  }
});

test("packing matches the production float-arena ABI exactly", () => {
  const f = field([{ kind: "box", lower: [-.2, .3, -.4], upper: [.2, .7, .4] },
    { kind: "ellipsoid", center: [.1, 1, .2], radii: [.2, .3, .4] },
    { kind: "quadratic-height", center: [.2, .4, -.1], curvature: [.5, 0, -.25] }]);
  const packed = packRetainedSceneDensity(f);
  assert.equal(packed.length, 64); assert.deepEqual([...packed.slice(0, 4)], [1, 3, 3, f.transitionWidth]);
  assert.deepEqual([...packed.slice(4, 7)], f.domain.lower); assert.deepEqual([...packed.slice(8, 11)], f.domain.upper);
  assert.deepEqual([...packed.slice(12, 15)], f.domain.lower);
  assert.equal(packed[16], 1); assert.equal(packed[32], 2); assert.equal(packed[48], 3);
  assert.deepEqual([...packed.slice(52, 55)], [Math.fround(.2), Math.fround(.4), Math.fround(-.1)]);
  assert.deepEqual([...packed.slice(56, 59)], [.5, 0, -.25]);
});

test("ellipsoid half-density surface and physical ramp match the declared quadratic", () => {
  const f = field([{ kind: "ellipsoid", center: [.13, 1, -.07], radii: [.2, .3, .4] }]);
  const p = f.primitives[0]; assert.equal(p.kind, "ellipsoid"); if (p.kind !== "ellipsoid") return;
  for (let i = 0; i < 101; i++) {
    const theta = i * 2.399963229728653, y = 1 - 2 * (i + .5) / 101, radius = Math.sqrt(1 - y * y);
    const normal = [radius * Math.cos(theta), y, radius * Math.sin(theta)];
    const surface = normal.map((v, axis) => p.center[axis] + p.radii[axis] * v) as unknown as RetainedScenePoint;
    close(evaluateRetainedScenePhi(f, surface), 0, 5e-16); close(evaluateRetainedSceneDensity(f, surface), .5, 1e-14);
    for (const s of [.4, .7, 1.1, 1.3]) {
      const x = normal.map((v, axis) => p.center[axis] + s * p.radii[axis] * v) as unknown as RetainedScenePoint;
      close(evaluateRetainedSceneDensity(f, x), Math.max(0, Math.min(1, .5 + Math.min(...p.radii) / (2 * f.transitionWidth) * (1 - s * s))), 2e-14);
    }
  }
});

test("vertical envelope integrates overlapping box, pool, and ellipsoid without double counting", () => {
  const f = field([{ kind: "quadratic-height", center: [0, .53, 0], curvature: [.3, 0, -.2] },
    { kind: "box", lower: [-.4, .4, -.4], upper: [.35, .9, .3] },
    { kind: "ellipsoid", center: [.1, 1, -.1], radii: [.3, .3, .35] }], .1);
  const samples = 20000;
  for (const [x, z] of [[0, 0], [.3, .1], [-.3, .25], [.42, -.2]]) {
    let oracle = 0;
    for (let i = 0; i < samples; i++) oracle += evaluateRetainedSceneDensity(f, [x, (i + .5) * 1.5 / samples, z]) * 1.5 / samples;
    close(integrateRetainedSceneVertical(f, x, z, 0, 1.5), oracle, 2e-8);
  }
});

test("full ellipsoid diffuse amount and octant restriction use the analytic radial integral", () => {
  for (const width of [.05, .4, 1e-7]) {
    const f = field([{ kind: "ellipsoid", center: [0, 1, 0], radii: [.2, .3, .4] }], width);
    const full = integrateRetainedSceneDensity(f, f.domain, { absoluteTolerance: 1e-14 });
    close(full.amount, radialAmount(f), width < 1e-6 ? 2e-11 : 2e-15); assert.equal(full.toleranceMet, true);
    assert.equal(full.verticalEvaluations, 0);
    const half = integrateRetainedSceneDensity(f, { lower: [0, 1, 0], upper: [1, 2, 1] }, { absoluteTolerance: 1e-14 });
    close(half.amount * 8, full.amount, 2e-15);
  }
});

test("shallow quadratic height integrates analytically when both ramp ends fit within the slab", () => {
  const f = field([{ kind: "quadratic-height", center: [.1, .6, -.2], curvature: [.2, 0, -.1] }], .1);
  const p = f.primitives[0]; assert.ok(p.kind === "quadratic-height");
  const box = { lower: [-.4, 0, -.3], upper: [.4, 1, .3] } as const;
  const meanSquareX = .8 ** 2 / 12 + p.center[0] ** 2;
  const meanSquareZ = .6 ** 2 / 12 + p.center[2] ** 2;
  const expected = .8 * .6 * (p.center[1] + p.curvature[0] * meanSquareX + p.curvature[2] * meanSquareZ);
  const r = integrateRetainedSceneDensity(f, box, { absoluteTolerance: 1e-12 });
  assert.equal(r.toleranceMet, true); close(r.amount, expected, 2e-15);
});

test("arbitrary ellipsoid box integrals converge and commute with a nonuniform partition", () => {
  const f = field([{ kind: "ellipsoid", center: [.03, .91, -.02], radii: [.27, .31, .23] }]);
  const box = { lower: [-.15, .66, -.19], upper: [.23, 1.14, .15] } as const;
  const whole = integrateRetainedSceneDensity(f, box, { absoluteTolerance: 1e-8 }); assert.equal(whole.toleranceMet, true);
  let amount = 0, error = whole.estimatedAbsoluteError;
  for (const [a, b] of [[-.15, -.07], [-.07, .09], [.09, .23]]) {
    const r = integrateRetainedSceneDensity(f, { lower: [a, box.lower[1], box.lower[2]], upper: [b, box.upper[1], box.upper[2]] }, { absoluteTolerance: 2e-9 });
    assert.equal(r.toleranceMet, true); amount += r.amount; error += r.estimatedAbsoluteError;
  }
  close(amount, whole.amount, Math.max(error, 1e-10));
  const exhausted = integrateRetainedSceneDensity(f, box, { absoluteTolerance: 1e-15, maximumRectangles: 1 });
  assert.equal(exhausted.toleranceMet, false); assert.equal(exhausted.rectangles, 1);
});

test("domain walls bound support without introducing density loss inside a filled reservoir", () => {
  const f = field([{ kind: "box", lower: [-1, 0, -1], upper: [1, .5, 1] }]);
  close(evaluateRetainedSceneDensity(f, [-1, 0, -1]), 1);
  assert.equal(evaluateRetainedSceneDensity(f, [-1.001, .2, 0]), 0);
  close(integrateRetainedSceneDensity(f, { lower: [-2, -.3, -2], upper: [2, 1, 2] }).amount, 2, 2e-12);
});

test("compiler preserves empty seed replacement and rejects unsupported liquid sources", () => {
  const scene = cloneScene(defaultScene); scene.fluid.initialCondition = "tank-fill";
  scene.fluid.initialLiquidVolumes = []; scene.fluid.initialBrickSeeds_m = [];
  assert.equal(compileRetainedSceneDensity(scene)!.primitives.length, 0);
  scene.fluid.initialBrickSeedsAdditive = true;
  assert.ok(compileRetainedSceneDensity(scene)!.primitives.length > 0);
  scene.fluid.initialLiquidVolumes = [{ shape: "torus", center_m: { x: 0, y: 1, z: 0 }, radius_m: .3, tubeRadius_m: .1 }];
  assert.equal(compileRetainedSceneDensity(scene), null);
  scene.fluid.initialLiquidVolumes = []; scene.fluid.initialHeightField = { kind: "cosine", baseHeight_m: .4, amplitude_m: .1, wavelength_m: 1, originX_m: 0 };
  assert.equal(compileRetainedSceneDensity(scene), null);
});

for (const make of [createCoarseFirstPoolImpactQuarterScene, createCoarseFirstPoolImpactHalfScene]) test(
  `${make().sceneId}: finest moments conserve analytic diffuse amount and reflection symmetry`, () => {
    const scene = make(), f = compileRetainedSceneDensity(scene); assert.ok(f);
    const dimensions = sceneLatticeDimensions(scene), h = Math.fround(scene.voxelDomain.finestCellSize_m);
    let receipt: RetainedSceneFineMeansReceipt | undefined;
    const means = compileRetainedSceneFineMeans(f, dimensions, h, { onReceipt: r => { receipt = r; } });
    assert.ok(receipt); assert.ok(receipt.integratedCells < means.length / 20);
    assert.ok(receipt.maximumEstimatedMeanError <= 2e-7);
    const p = f.primitives[0]; assert.ok(p.kind === "quadratic-height");
    const expected = dimensions[0] * h * dimensions[2] * h * p.center[1] + radialAmount(f);
    close(means.reduce((sum, q) => sum + q, 0) * h ** 3, expected, 2e-7);
    const at = (x: number, y: number, z: number) => x + dimensions[0] * (y + dimensions[1] * z);
    for (let z = 0; z < dimensions[2]; z++) for (let y = 0; y < dimensions[1]; y++) for (let x = 0; x < dimensions[0]; x++) {
      const q = means[at(x, y, z)]; assert.ok(q >= 0 && q <= 1);
      assert.equal(q, means[at(dimensions[0] - 1 - x, y, z)]); assert.equal(q, means[at(x, y, dimensions[2] - 1 - z)]);
    }
  });
