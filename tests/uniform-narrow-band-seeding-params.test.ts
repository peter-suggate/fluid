import assert from "node:assert/strict";
import test from "node:test";
import { uniformNearSettled } from "../lib/methods/uniform/uniform-near-settled";
import { narrowBandFlipValues, uniformNarrowBandMethod } from "../lib/methods/uniform/uniform-narrow-band-method";
import { narrowBandRetireCounting } from "../lib/methods/uniform/uniform-narrow-band-flip";
import { NARROW_BAND_CENTRED_BOUND_WGSL, NARROW_BAND_WALL_BOUND_WGSL, narrowBandSurfaceWGSL } from "../lib/methods/uniform/uniform-narrow-band-surface.wgsl";

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

test("heatExpiry and sampleBound default to the retirement and union bound they replace", () => {
  assert.equal(uniformNearSettled({}).heatExpiry, "anywhere");
  assert.equal(uniformNearSettled({ heatExpiry: "held" }).heatExpiry, "held");
  assert.equal(uniformNearSettled({}).sampleBound, "centred");
  assert.equal(uniformNearSettled({ sampleBound: "walls" }).sampleBound, "walls");
  const values = narrowBandFlipValues({});
  assert.equal(values.heatExpiry, "anywhere");
  assert.equal(values.sampleBound, "centred");
});

test("the wall bound replaces the centred one once, and keeps its expression where all eight cells are open", () => {
  assert.equal(narrowBandSurfaceWGSL.split(NARROW_BAND_CENTRED_BOUND_WGSL).length - 1, 1, "particleSurfaceFinish holds the centred bound once");
  const centred = "particle=max(particle,min(bulk,1.0-f32(held)/(4.0*f32(open))));";
  assert.ok(NARROW_BAND_CENTRED_BOUND_WGSL.includes(`if(open>0u){${centred}}`));
  assert.ok(NARROW_BAND_WALL_BOUND_WGSL.includes(`if(open==8u){${centred}}`));
  assert.match(NARROW_BAND_WALL_BOUND_WGSL, /bound=select\(-sparse,1\.0-max\(low,high\)-sparse,denseOpen\);/);
});
