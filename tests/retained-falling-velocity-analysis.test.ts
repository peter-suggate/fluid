import assert from "node:assert/strict";
import test from "node:test";
import { analyzeNativeVelocity, gradientInvariants, nativeLocalGradient,
  selectSnapshotVelocity, type NativeVelocitySample, type Point3 } from "../tools/retained-falling-velocity-analysis";

const close = (actual: readonly number[], expected: readonly number[], tolerance = 1e-12) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, i) => assert.ok(Math.abs(value - expected[i]!) < tolerance,
    `component ${i}: ${value} != ${expected[i]}`));
};
const sample = (point: Point3, velocity: Point3, id = 0, volume = 1): NativeVelocitySample =>
  ({ point, velocity, id, volume });

test("precommit snapshots select source VEX and destination collocated bank independently of scalar parity", () => {
  // Include unequal parities: selecting a velocity bank from scalar parity can
  // appear correct in ordinary frames yet read the wrong state after edits.
  for (const scalar of [0, 1]) for (const face of [0, 1]) {
    assert.deepEqual(selectSnapshotVelocity("transport-velocity-extension", scalar, face),
      { densityParity: scalar, velocityPlane: "effective", extended: true });
    assert.deepEqual(selectSnapshotVelocity("velocity-projection", scalar, face),
      { densityParity: scalar ^ 1, velocityPlane: face ? "cellVelocityA" : "cellVelocityB", extended: false });
  }
  assert.throws(() => selectSnapshotVelocity("frame-commit", 0, 0), /Unsupported/);
  for (const invalid of [-1, 2, .5, NaN]) {
    assert.throws(() => selectSnapshotVelocity("velocity-projection", invalid, 0), /parity/);
    assert.throws(() => selectSnapshotVelocity("transport-velocity-extension", 0, invalid), /parity/);
  }
});

test("unequal native volumes recover an affine physical velocity and its invariants", () => {
  const gradient = [1, 2, 3, -4, 5, 6, 7, -8, -6];
  const intercept: Point3 = [2, -3, 4];
  const points: Point3[] = [[-.1, .2, -.4], [.7, .1, .3], [0, .8, -.2], [.1, -.1, .9], [.9, .6, .5]];
  const samples = points.map((point, i) => sample(point,
    intercept.map((value, axis) => value + point.reduce((sum, p, j) => sum + gradient[3 * axis + j]! * p, 0)) as unknown as Point3,
    i, [1, 8, 1, 64, 8][i]! * .001));
  const result = analyzeNativeVelocity(samples, intercept);
  assert.ok(result.affine);
  close(result.affine.gradient_per_s, gradient);
  close(result.affine.vorticity_per_s, [-14, -4, -6]);
  assert.ok(Math.abs(result.affine.divergence_per_s) < 1e-12);
  assert.ok(result.maximumAffineResidual_m_s! < 1e-12);
  assert.ok(result.maximumUniformResidual_m_s! > 1);
  const total = samples.reduce((sum, cell) => sum + cell.volume, 0);
  close(result.meanVelocity_m_s!, [0, 1, 2].map(axis => samples.reduce((sum, cell) => sum + cell.volume * cell.velocity[axis]!, 0) / total));
});

test("coarse/fine face offsets recover a full affine gradient rather than axis-only secants", () => {
  const gradient = [0, -2, 3, 2, 0, -4, -3, 4, 0];
  const center = sample([.2, -.1, .3], [1, -2, 3]);
  const offsets: Point3[] = [[-.3, 0, 0], [.2, -.1, -.1], [.2, .1, .1], [0, -.3, 0], [0, .2, 0], [0, 0, .2], [0, 0, -.3]];
  const neighbors = offsets.map((d, i) => ({ area: i === 1 || i === 2 ? .01 : .04,
    sample: sample(d.map((value, axis) => value + center.point[axis]) as unknown as Point3,
      center.velocity.map((value, axis) => value + d.reduce((sum, q, j) => sum + gradient[3 * axis + j]! * q, 0)) as unknown as Point3, i + 1) }));
  const result = nativeLocalGradient(center, neighbors);
  assert.ok(result);
  close(result.gradient_per_s, gradient);
  assert.ok(result.strainFrobenius_per_s < 1e-12);
  close(result.vorticity_per_s, [8, 6, 4]);
});

test("non-affine quadratic velocity retains nonzero residual while symmetric local differences recover its derivative", () => {
  const velocity = ([x, y, z]: Point3): Point3 => [x * x + 2 * y, y * y - z, z * z + x];
  const points: Point3[] = [];
  for (const x of [-1, 0, 1]) for (const y of [-1, 0, 1]) for (const z of [-1, 0, 1]) points.push([x, y, z]);
  const result = analyzeNativeVelocity(points.map((p, i) => sample(p, velocity(p), i)), [0, 0, 0]);
  assert.ok(result.affine);
  close(result.affine.gradient_per_s, [0, 2, 0, 0, 0, -1, 1, 0, 0]);
  assert.ok(Math.abs(result.affineRmsResidual_m_s! - Math.sqrt(2 / 3)) < 1e-12);
  const p: Point3 = [.3, -.2, .4], center = sample(p, velocity(p));
  const neighbors = [0, 1, 2].flatMap(axis => [-1, 1].map(side => {
    const q = [...p] as [number, number, number]; q[axis]! += side * .05;
    return { area: .0025, sample: sample(q, velocity(q)) };
  }));
  const local = nativeLocalGradient(center, neighbors);
  assert.ok(local);
  close(local.gradient_per_s, [.6, 2, 0, 0, -.4, -1, 1, 0, .8]);
});

test("rank-deficient native samples do not manufacture a 3D derivative", () => {
  const samples = [-1, 0, 1].map(x => sample([x, 0, 0], [x, 2, 3]));
  assert.equal(analyzeNativeVelocity(samples, [0, 2, 3]).affine, null);
  assert.equal(nativeLocalGradient(samples[1]!, [samples[0]!, samples[2]!].map(s => ({ sample: s, area: 1 }))), undefined);
  assert.deepEqual(analyzeNativeVelocity([], [0, 0, 0]), { count: 0 });
  close(gradientInvariants([1, 0, 0, 0, 2, 0, 0, 0, 3]).vorticity_per_s, [0, 0, 0]);
});
