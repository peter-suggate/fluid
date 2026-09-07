import assert from "node:assert/strict";
import test from "node:test";
import { packFineLevelSetSample, unpackFineLevelSetPackedPhi } from "../lib/core/fine-levelset-packed-sample";
import { compileRetainedSceneDensity, evaluateRetainedSceneDensity, evaluateRetainedScenePhi } from "../lib/methods/adaptive-mass/sparse-cm12-retained-scene-density";
import { exactPoolImpactImplicitPhi, exactVerticalCrossings, measurePublishedPoolImpact, poolImpactBudgets, poolImpactOracle,
  POOL_IMPACT_SCENES } from "../tools/implicit-density/pool-impact-oracle";

for (const id of POOL_IMPACT_SCENES) test(`${id}: actual catalog geometry and original minmax8 region have an independent oracle`, () => {
  const oracle = poolImpactOracle(id), quarter = id.endsWith("quarter");
  assert.deepEqual(oracle.dimensions, quarter ? [32, 24, 32] : [64, 48, 64]);
  assert.equal(oracle.h, .05);
  assert.ok(Math.abs(oracle.poolHeight - (quarter ? .4 : .8)) < 1e-15);
  assert.deepEqual(oracle.sphereCenter, [0, quarter ? .9125 : 1.825, 0]);
  assert.equal(oracle.sphereRadius, quarter ? .25 : .5);
  const region = oracle.scene.fluid.refinementRegions![0]!;
  assert.equal(region.minimumCellSize_cells, 8); assert.equal(region.maximumCellSize_cells, 8);
  assert.deepEqual(region.min_m, quarter ? { x: -.8, y: 0, z: -.8 } : { x: -1.6, y: 0, z: -1.6 });
  // Preserve the literal original URL: rounding 66.6667% puts the half scene
  // just beyond the parser's snap tolerance, so its outward snap reaches 2m.
  const expectedMaximum = quarter ? { x: -.4, y: .8, z: .8 } : { x: -.8, y: 2, z: 1.6 };
  for (const axis of ["x", "y", "z"] as const)
    assert.ok(Math.abs(region.max_m[axis] - expectedMaximum[axis]) < 1e-14);
  const [nx, ny, nz] = oracle.dimensions;
  const phi = new Float32Array(nx * ny * nz), ballRemoved = new Float32Array(phi.length);
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const wx = oracle.origin[0] + (x + .5) * oracle.h, wy = (y + .5) * oracle.h;
    const wz = oracle.origin[2] + (z + .5) * oracle.h, r = oracle.sphereRadius;
    const spherePhi = (wx * wx + (wy - oracle.sphereCenter[1]) ** 2 + wz * wz - r * r) / (2 * r);
    const index = x + nx * (y + ny * z), planePhi = wy - oracle.poolHeight;
    phi[index] = unpackFineLevelSetPackedPhi(packFineLevelSetSample(Math.min(planePhi, spherePhi), 1));
    ballRemoved[index] = planePhi;
  }
  const metrics = measurePublishedPoolImpact(phi, oracle), budgets = poolImpactBudgets(oracle);
  assert.equal(metrics.missingOrExtraCrossingColumns, 0, JSON.stringify(metrics));
  assert.ok(metrics.sphereCrossings > 100, "inspect both sides on every sphere-covered vertical ray");
  assert.ok(metrics.maximumPoolHeightError_m < budgets.poolPlanarity_m);
  assert.ok(metrics.maximumSphereDistanceError_m < budgets.spherePublishedRadial_m);
  assert.equal(metrics.missingAnalyticSamples, 0);
  assert.ok(metrics.maximumSamplePrecisionBudgetRatio <= 1);
  const missing = measurePublishedPoolImpact(ballRemoved, oracle);
  assert.equal(missing.missingOrExtraCrossingColumns, metrics.sphereCrossings / 2,
    "an intact pool cannot conceal a missing suspended sphere");
  assert.equal(exactVerticalCrossings(oracle, 0, 0).length, 3,
    "the oracle includes pool plus lower and upper sphere surfaces");
});

for (const id of POOL_IMPACT_SCENES) test(`${id}: compiled retained field agrees with analytic geometry at arbitrary physical points`, () => {
  const oracle = poolImpactOracle(id), field = compileRetainedSceneDensity(oracle.scene);
  assert.ok(field);
  // Fibonacci sphere points avoid favorable grid axes, symmetric samples, and
  // cell centers. Compare the retained defining function at the actual sphere,
  // and its density at physical offsets through the same fixed-width ramp.
  let maximumSurfaceResidual_m = 0, maximumDensityError = 0, maximumNormalVectorError = 0;
  for (let i = 0; i < 1000; i++) {
    const ny = 1 - 2 * (i + .5) / 1000, theta = i * Math.PI * (3 - Math.sqrt(5));
    const radial = Math.sqrt(1 - ny * ny), normal = [radial * Math.cos(theta), ny, radial * Math.sin(theta)];
    const point = normal.map((v, axis) => oracle.sphereCenter[axis]! + oracle.sphereRadius * v) as [number, number, number];
    maximumSurfaceResidual_m = Math.max(maximumSurfaceResidual_m, Math.abs(evaluateRetainedScenePhi(field, point)));
    const epsilon = 1e-5;
    const gradient = [0, 1, 2].map(axis => {
      const plus = [...point] as [number, number, number], minus = [...point] as [number, number, number];
      plus[axis] += epsilon; minus[axis] -= epsilon;
      return (evaluateRetainedScenePhi(field, plus) - evaluateRetainedScenePhi(field, minus)) / (2 * epsilon);
    });
    const length = Math.hypot(...gradient);
    maximumNormalVectorError = Math.max(maximumNormalVectorError, Math.hypot(...gradient.map((v, axis) => v / length - normal[axis]!)));
    for (const offset of [-.031, -.017, .006, .023]) {
      const query = point.map((v, axis) => v + offset * normal[axis]!) as [number, number, number];
      const expectedPhi = exactPoolImpactImplicitPhi(oracle, query);
      const expectedDensity = Math.max(0, Math.min(1, .5 - expectedPhi / oracle.h));
      maximumDensityError = Math.max(maximumDensityError, Math.abs(evaluateRetainedSceneDensity(field, query) - expectedDensity));
    }
  }
  // Float32 canonical coordinates are accepted; each explicit budget is far
  // below a finest cell and unrelated to the operation-cell width.
  assert.ok(maximumSurfaceResidual_m < 2e-7, `${id}: sphere zero surface residual ${maximumSurfaceResidual_m} m`);
  assert.ok(maximumDensityError < 5e-6, `${id}: retained diffuse density error ${maximumDensityError}`);
  assert.ok(maximumNormalVectorError < 1e-6, `${id}: sphere normal error ${maximumNormalVectorError}`);
  for (const x of [-.731, -.183, .017, .389]) for (const z of [-.637, -.109, .277, .513])
    assert.ok(Math.abs(evaluateRetainedScenePhi(field, [x, oracle.poolHeight, z])) < 1e-7,
      `${id}: authored flat pool remains the exact zero plane`);
});
