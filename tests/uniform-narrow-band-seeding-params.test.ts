import assert from "node:assert/strict";
import test from "node:test";
import { uniformNearSettled } from "../lib/methods/uniform/uniform-near-settled";
import { narrowBandFlipValues, uniformNarrowBandMethod } from "../lib/methods/uniform/uniform-narrow-band-method";
import { narrowBandRetireCounting } from "../lib/methods/uniform/uniform-narrow-band-flip";

test("seedScoring and outerSeeding default to the census and reseeding they replace", () => {
  assert.equal(uniformNearSettled({}).seedScoring, "unshaped");
  assert.equal(uniformNearSettled({ seedScoring: "bogus" }).seedScoring, "unshaped");
  assert.equal(uniformNearSettled({ seedScoring: "every" }).seedScoring, "every");
  assert.equal(uniformNearSettled({}).outerSeeding, "retain");
  assert.equal(uniformNearSettled({ outerSeeding: "bogus" }).outerSeeding, "retain");
  assert.equal(uniformNearSettled({ outerSeeding: "uncovered" }).outerSeeding, "uncovered");
  const values = narrowBandFlipValues({});
  assert.equal(values.seedScoring, "unshaped");
  assert.equal(values.outerSeeding, "retain");
  for (const key of ["seedScoring", "outerSeeding"]) {
    const param = uniformNarrowBandMethod.params.find(p => p.key === key);
    assert.ok(param && param.kind === "select" && param.update === "solver", `${key} is a solver-restarting select`);
  }
});

test("retirement counting is a diagnostic that is off unless a probe turns it on", () => {
  assert.equal(narrowBandRetireCounting.on, false);
});
