import assert from "node:assert/strict";
import test from "node:test";
import { createOctreePowerSite, powerBoxBoundary } from "../lib/octree-power-geometry";
import {
  applyOctreePowerMatrix,
  buildOctreePowerOperator,
  OCTREE_POWER_INVALID_ROW,
  octreePowerMatrixEnergy,
  octreePowerBoundaryDistance,
  octreePowerDivergence,
  planOctreePowerStorage,
  projectOctreePowerFaceVelocities,
  reconstructPowerVelocity,
} from "../lib/octree-power-operator";

const close = (actual: number, expected: number, tolerance = 1e-8) => assert.ok(Math.abs(actual - expected) <= tolerance,
  `expected ${actual} to be within ${tolerance} of ${expected}`);

function transitionPatch() {
  const sites = [createOctreePowerSite("coarse", [0, 0, 0], 2)];
  for (let z = 0; z < 3; z += 1) for (let y = 0; y < 3; y += 1) for (let x = 0; x < 3; x += 1) {
    if (x < 2 && y < 2 && z < 2) continue;
    sites.push(createOctreePowerSite(`${x}${y}${z}`, [x, y, z], 1));
  }
  return { sites, boundaries: powerBoxBoundary([0, 0, 0], [3, 3, 3]) };
}

test("power operator emits one face with reciprocal signed incidence", () => {
  const { sites, boundaries } = transitionPatch();
  const operator = buildOctreePowerOperator(sites, boundaries);
  assert.ok(operator.faces.length > 0);
  for (const face of operator.faces) {
    const negative = operator.incidence[face.negativeRow].find((item) => item.face === face.id);
    assert.deepEqual(negative, { face: face.id, sign: 1 });
    if (face.positiveRow !== 0xffff_ffff) {
      const positive = operator.incidence[face.positiveRow].find((item) => item.face === face.id);
      assert.deepEqual(positive, { face: face.id, sign: -1 });
      assert.ok(face.inverseDistance > 0);
    }
    assert.ok(face.area > 0 && Number.isFinite(face.area));
  }
  assert.equal(new Set(operator.faces.map((face) => face.key)).size, operator.faces.length);
});

test("compact power matrix is symmetric positive semidefinite with constant null space", () => {
  const { sites, boundaries } = transitionPatch();
  const operator = buildOctreePowerOperator(sites, boundaries);
  for (const row of operator.rows) for (const entry of row.entries) {
    const reciprocal = operator.rows[entry.row].entries.find((candidate) => candidate.row === row.row);
    assert.ok(reciprocal);
    close(reciprocal.coefficient, entry.coefficient, 2e-8);
  }
  applyOctreePowerMatrix(operator, operator.rows.map(() => 1)).forEach((value) => close(value, 0, 2e-8));
  for (let trial = 0; trial < 20; trial += 1) {
    const vector = operator.rows.map((_, row) => Math.sin((row + 1) * (trial + 0.37)));
    assert.ok(octreePowerMatrixEnergy(operator, vector) >= -1e-8);
  }
});

test("face-derived RHS has zero integrated divergence for a constant velocity", () => {
  const { sites, boundaries } = transitionPatch();
  const velocity = [1.25, -0.75, 0.5] as const;
  const operator = buildOctreePowerOperator(sites, boundaries, {
    normalVelocity: (_centroid, normal) => normal[0] * velocity[0] + normal[1] * velocity[1] + normal[2] * velocity[2],
  });
  operator.rows.forEach((row) => close(row.rhs, 0, 2e-8));
});

test("projection uses the same inverse dual distance and reduces divergence", () => {
  const { sites, boundaries } = transitionPatch();
  const pressure = [...sites].sort((a, b) => a.key.localeCompare(b.key)).map((site) => 0.5 * site.center[0] - site.center[1] + 0.25 * site.center[2]);
  const initial = buildOctreePowerOperator(sites, boundaries, {
    normalVelocity: (_centroid, _normal, negative, positive) => positive
      ? pressure[[...sites].sort((a, b) => a.key.localeCompare(b.key)).findIndex((site) => site.key === positive.key)]
        / Math.hypot(...positive.center.map((value, axis) => value - negative.center[axis]))
        - pressure[[...sites].sort((a, b) => a.key.localeCompare(b.key)).findIndex((site) => site.key === negative.key)]
        / Math.hypot(...positive.center.map((value, axis) => value - negative.center[axis]))
      : 0,
  });
  const before = octreePowerDivergence(initial);
  const projected = projectOctreePowerFaceVelocities(initial, pressure);
  const after = octreePowerDivergence(initial, projected);
  assert.ok(Math.max(...before.map(Math.abs)) > 1e-3);
  assert.ok(Math.max(...after.map(Math.abs)) < 1e-7);
});

test("partial apertures use the same open area in assembly and projection", () => {
  const { sites, boundaries } = transitionPatch();
  const operator = buildOctreePowerOperator(sites, boundaries, {
    openFraction: (_negative, positive) => positive ? 0.25 : 1,
    normalVelocity: (_centroid, normal) => normal[0],
  });
  const interior = operator.faces.find((face) => face.positiveRow !== OCTREE_POWER_INVALID_ROW);
  assert.ok(interior);
  const pressure = operator.rows.map((_row, index) => index * 2);
  const projected = projectOctreePowerFaceVelocities(operator, pressure, 0.5);
  const expected = interior.normalVelocity - 0.5
    * (pressure[interior.positiveRow] - pressure[interior.negativeRow])
    * interior.inverseDistance * interior.openFraction;
  assert.ok(Math.abs(projected[interior.id] - expected) < 1e-12);
  const coefficient = operator.rows[interior.negativeRow].entries
    .find((entry) => entry.row === interior.positiveRow)?.coefficient;
  assert.equal(coefficient, interior.area * interior.inverseDistance * interior.openFraction);
});

test("ghost-fluid boundary distance follows the exact dual-edge crossing", () => {
  close(octreePowerBoundaryDistance(-0.25, 0.75, 2), 0.5);
  close(octreePowerBoundaryDistance(-1e-6, 1, 2), 2e-6 / 1.000001);
  assert.throws(() => octreePowerBoundaryDistance(1, -1, 2), /liquid\/air phi/);
  assert.throws(() => octreePowerBoundaryDistance(0, 1, 2), /liquid\/air phi/);
});

test("CPU oracle includes an open zero-pressure boundary in assembly and projection", () => {
  const site = createOctreePowerSite("liquid", [0, 0, 0], 1);
  const operator = buildOctreePowerOperator([site], powerBoxBoundary([0, 0, 0], [1, 1, 1]), {
    boundaryDistance: (_negative, face) => face.boundaryKey === "x+" ? 0.25 : undefined,
    normalVelocity: (_centroid, _normal, _negative, positive) => positive ? 0 : 1,
  });
  const open = operator.faces.find((face) => face.boundaryKey === "x+");
  assert.ok(open);
  close(open.inverseDistance, 4);
  close(operator.rows[0].diagonal, open.area * open.inverseDistance);
  const pressure = [-0.25];
  const projected = projectOctreePowerFaceVelocities(operator, pressure);
  close(projected[open.id], 0);
});

test("uniform center row exactly reproduces six-point coefficients", () => {
  const sites = [
    createOctreePowerSite("anchor", [0, 0, 0], 1),
    createOctreePowerSite("x-", [-1, 0, 0], 1), createOctreePowerSite("x+", [1, 0, 0], 1),
    createOctreePowerSite("y-", [0, -1, 0], 1), createOctreePowerSite("y+", [0, 1, 0], 1),
    createOctreePowerSite("z-", [0, 0, -1], 1), createOctreePowerSite("z+", [0, 0, 1], 1),
  ];
  const operator = buildOctreePowerOperator(sites, powerBoxBoundary([-1, -1, -1], [2, 2, 2]));
  const row = operator.rows.find((candidate) => candidate.siteKey === "anchor")!;
  close(row.diagonal, 6);
  assert.equal(row.entries.length, 6);
  row.entries.forEach((entry) => close(entry.coefficient, 1));
  close(row.volume, 1);
});

test("least-squares reconstruction handles Cartesian and general normals", () => {
  const velocity = [2, -3, 4] as const;
  const normals = [
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    [1 / Math.sqrt(3), 1 / Math.sqrt(3), 1 / Math.sqrt(3)],
  ] as const;
  const result = reconstructPowerVelocity(normals.map((normal, index) => ({
    normal,
    normalVelocity: normal[0] * velocity[0] + normal[1] * velocity[1] + normal[2] * velocity[2],
    weight: index + 1,
  })));
  assert.equal(result.usedFallback, false);
  result.velocity.forEach((value, axis) => close(value, velocity[axis]));
  assert.deepEqual(reconstructPowerVelocity([{ normal: [1, 0, 0], normalVelocity: 3, weight: 1 }], [7, 8, 9]).velocity, [7, 8, 9]);
});

test("power storage planner is compact-capacity-scaled", () => {
  const plan = planOctreePowerStorage(100, 400, 32);
  assert.equal(plan.rowMetricBytes, 1_600);
  assert.equal(plan.faceBytes, 12_800);
  assert.equal(plan.incidenceOffsetBytes, 404);
  assert.equal(plan.incidenceEntryBytes, 3_200);
  assert.equal(plan.incidenceBytes, 3_604);
  assert.equal(plan.maximumIncidencePerRow, 32);
  assert.equal(plan.allocatedBytes, 18_004);
  assert.throws(() => planOctreePowerStorage(100, 400, 0), /positive integer/);
});
