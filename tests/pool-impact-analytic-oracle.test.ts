import assert from "node:assert/strict";
import test from "node:test";
import { packFineLevelSetSample, unpackFineLevelSetPackedPhi } from "../lib/core/fine-levelset-packed-sample";
import { exactVerticalCrossings, measurePublishedPoolImpact, poolImpactBudgets, poolImpactOracle,
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
  const missing = measurePublishedPoolImpact(ballRemoved, oracle);
  assert.equal(missing.missingOrExtraCrossingColumns, metrics.sphereCrossings / 2,
    "an intact pool cannot conceal a missing suspended sphere");
  assert.equal(exactVerticalCrossings(oracle, 0, 0).length, 3,
    "the oracle includes pool plus lower and upper sphere surfaces");
});
