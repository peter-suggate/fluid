import assert from "node:assert/strict";
import test from "node:test";
import { clampedAffineMean, evaluate, frameForBox, mean, mergeEquivalentFields,
  polynomial, positiveAffineMean, rebase, splitBox, splitField, volume, type Box } from "../tools/implicit-density/field";
import { fixtures } from "../tools/implicit-density/ladder-fixtures";
import { runFixture, verifyUnrepresentableMerge } from "../tools/implicit-density/ladder";

for (const fixture of fixtures) test(`${fixture.id}: independent field/integral/gradient oracle through 100 mixed partitions`, () => {
  const result = runFixture(fixture);
  assert.ok(result.passed, JSON.stringify(result));
  assert.ok(result.normalProbeCount > 0, "measure surface crossings and normals against analytic geometry");
  if (fixture.id.startsWith("edge-")) assert.ok(result.creaseProbeCount > 0,
    "keep exact sharp ties distinct from smooth surface normals");
  assert.equal(result.maximumLeaves, 260, "must actually subdivide the query partition");
  assert.deepEqual(result.checkpoints.map(c => c.cycle), [1, 10, 100]);
});

test("30% planar fill: integrate a fixed physical ramp, never interpolate cell means", () => {
  const box: Box = { lower: [0, 0, 0], upper: [1, 1, 1] };
  const field = polynomial({ origin: [0, 0, 0], scale: [1, 1, 1] }, [8, 0, -25, 0, 0, 0, 0, 0, 0, 0]);
  assert.ok(Math.abs(clampedAffineMean(field, box) - 0.3) < 1e-13);
  assert.ok(Math.abs(evaluate(field, [0.1, 0.3, 0.7]) - 0.5) < 1e-13);
  let mass = 0;
  for (const child of splitBox(box)) {
    const q = rebase(field, frameForBox(child));
    const density = clampedAffineMean(q, child);
    assert.ok(Math.abs(density - (child.lower[1] === 0 ? 0.6 : 0)) < 1e-13);
    mass += density * volume(child);
  }
  assert.ok(Math.abs(mass - 0.3) < 1e-13);
  // Correct means used as point samples would still give the wrong contour.
  assert.ok(Math.abs((0.25 + (0.6 - 0.5) / 0.6 * 0.5) - 1 / 3) < 1e-14);
});

test("oblique affine positive-part integral agrees with independent known simplex volumes", () => {
  const box: Box = { lower: [0, 0, 0], upper: [1, 1, 1] };
  const frame = { origin: [0, 0, 0] as const, scale: [1, 1, 1] as const };
  // Integral of max(1-x-y-z,0) over the unit cube is 1/24.
  for (const permutation of [[-1, -1, -1], [1, -1, -1], [1, 1, -1]]) {
    const positives = permutation.filter(x => x > 0).length;
    const f = polynomial(frame, [1 - positives, ...permutation, 0, 0, 0, 0, 0, 0]);
    assert.ok(Math.abs(positiveAffineMean(f, box) - 1 / 24) < 1e-13);
    const childSum = splitBox(box).reduce((s, child) => s + positiveAffineMean(f, child) / 8, 0);
    assert.ok(Math.abs(childSum - 1 / 24) < 1e-13);
  }
});

test("new child detail cannot be discarded by a conservative-mean merge", () => {
  assert.equal(verifyUnrepresentableMerge(), true);
  const fixture = fixtures[0], children = splitField(fixture.field, fixture.box);
  const merged = mergeEquivalentFields(children.map(c => c.field), frameForBox(fixture.box));
  assert.ok(merged);
  assert.ok(Math.abs(mean(merged, fixture.box) - fixture.exactMean(fixture.box)) < 1e-13);
  assert.equal(mergeEquivalentFields([], frameForBox(fixture.box)), null);
});

test("unsupported curved feature integration fails instead of silently approximating", () => {
  const curved = fixtures.find(f => f.id === "curved-shallow-bowl")!.field;
  const plane = fixtures[0].field;
  assert.equal(curved.kind, "polynomial"); assert.equal(plane.kind, "polynomial");
  if (curved.kind !== "polynomial" || plane.kind !== "polynomial") return;
  assert.throws(() => mean({ kind: "minimum", branches: [curved, plane] }, fixtures[0].box), /affine/);
});
