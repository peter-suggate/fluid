import assert from "node:assert/strict";
import test from "node:test";
import { packFineLevelSetSample, unpackFineLevelSetPackedPhi } from "../lib/core/fine-levelset-packed-sample";
import { measurePartitionAnalyticField, partitionAnalyticPhi,
  type PartitionFixture } from "../tools/implicit-density/partition-oracle";

function samples(fixture: PartitionFixture) {
  return Float32Array.from({ length: 16 ** 3 }, (_, i) => partitionAnalyticPhi(fixture,
    [-.4 + (i % 16 + .5) * .05, (Math.floor(i / 16) % 16 + .5) * .05,
      -.4 + (Math.floor(i / 256) + .5) * .05]));
}

for (const fixture of ["flat", "quadratic", "sphere-pool", "sharp-box"] as const) test(
  `${fixture}: independent analytic partition oracle accepts exact samples and detects a changed field`, () => {
    const field = samples(fixture);
    const exact = measurePartitionAnalyticField(fixture, field, [16, 16, 16], .05);
    assert.equal(exact.missingSamples, 0);
    assert.ok(exact.sampleCount > 100);
    assert.ok(exact.maximumSamplePrecisionRatio <= 1);
    assert.equal(exact.unresolvedAnalyticColumns, 0);
    assert.equal(exact.changedCrossingColumns, 0);
    assert.equal(exact.observedCrossings, exact.analyticCrossings);
    assert.ok(exact.maximumRootPrecisionRatio <= 1);
    assert.ok(exact.maximumZeroDistance_m <= exact.surfaceBudget_m);
    const stored = measurePartitionAnalyticField(fixture,
      field.map(value => unpackFineLevelSetPackedPhi(packFineLevelSetSample(value, 1))), [16, 16, 16], .05);
    assert.equal(stored.changedCrossingColumns, 0);
    assert.ok(stored.maximumSamplePrecisionRatio <= 1);
    assert.ok(stored.maximumRootPrecisionRatio <= 1, "root precision includes binary16 storage near tangent rays");
    assert.ok(stored.maximumZeroDistance_m <= stored.surfaceBudget_m);
    const shifted = measurePartitionAnalyticField(fixture, field.map(value => value + .01), [16, 16, 16], .05);
    assert.ok(shifted.maximumSamplePrecisionRatio > 1, "a retained wrong baseline must fail");
    const missing = field.slice();
    missing[Array.from(field).findIndex(value => Math.abs(value) < .05)] = NaN;
    assert.equal(measurePartitionAnalyticField(fixture, missing, [16, 16, 16], .05).missingSamples, 1);
  });

test("sharp-box oracle distinguishes exact authored top from a sampled active-face change", () => {
  assert.equal(partitionAnalyticPhi("sharp-box", [.125, .57, .025]), 0);
  const result = measurePartitionAnalyticField("sharp-box", samples("sharp-box"), [16, 16, 16], .05);
  assert.ok(result.maximumZeroDistance_m > .004, "the lateral face can limit the nearest-surface distance");
  assert.ok(result.maximumZeroDistance_m < .025, "the discrepancy is bounded by finest sampling");
});
