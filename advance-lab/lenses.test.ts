import assert from "node:assert/strict";
import test from "node:test";
import {
  FRACTION_FLOOR, fractionReadout, fractionResidueRamp, interfaceSegments,
  rdfMinorityAreaDistorted,
} from "./lenses";

test("minority-area guard ignores roundoff and rejects amplified corner cuts", () => {
  assert.equal(rdfMinorityAreaDistorted(3.999774, 3.78, 4), true,
    "a 0.000226 accepted air sliver must not become a 0.22-cell corner hole");
  assert.equal(rdfMinorityAreaDistorted(3.999774, 3.9996, 4), false,
    "sub-per-mille cell-area variation remains RDF presentation smoothing");
  assert.equal(rdfMinorityAreaDistorted(0.2, 0.3, 1), false,
    "a resolved minority phase may vary without forcing a PLIC seam");
});

test("the fraction readout sizes itself to the answer it has to keep", () => {
  assert.equal(fractionReadout(0.42), ".42", "two decimals carry the ordinary range");
  assert.equal(fractionReadout(1), "1", "a full cell is one character, not 1.00");
  assert.equal(fractionReadout(0.995), "1",
    "half a percent under full rounds to full rather than drawing .99 forever");
  assert.equal(fractionReadout(1.04), "1.04",
    "overfull keeps the digit that says how far past capacity the cell is");
  assert.equal(fractionReadout(1e-4), "1e-4",
    "a resolved residue cell must not read as .00, which is what vacuum reads as");
  assert.equal(fractionReadout(FRACTION_FLOOR), "1e-6", "the floor still states itself");
});

test("the residue ramp spends a fixed share on each decade", () => {
  assert.equal(fractionResidueRamp(FRACTION_FLOOR), 0);
  assert.equal(fractionResidueRamp(0.5), 1);
  const decades = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1].map(fractionResidueRamp);
  for (let i = 1; i < decades.length; i += 1) {
    const step = decades[i]! - decades[i - 1]!;
    assert.ok(Math.abs(step - 0.1755) < 1e-3,
      `a decade must be a fixed share of the ramp, not ${step.toFixed(4)}`);
  }
  /* The claim the log ramp exists for: on a linear ramp over [0, 1/2] these two
   * cells differ by under a percent of the range and draw as the same nothing. */
  assert.ok(fractionResidueRamp(1e-3) - fractionResidueRamp(1e-4) > 0.15,
    "two residue cells a decade apart must separate");
});

test("an interface chord is the part of the plane that is not the cell's own edge", () => {
  /* Liquid above the waterline in canvas terms: the normal points down, out of
   * it, and the plane sits halfway down the cell. */
  const halved = interfaceSegments({ nx: 0, ny: 1, clipNx: 0, clipNy: 1, offset: 0.5 });
  assert.equal(halved.length, 1, "a plane through the cell cuts exactly one chord");
  const [ax, ay, bx, by] = halved[0]!;
  assert.equal(ay, 0.5);
  assert.equal(by, 0.5);
  assert.deepEqual([ax, bx].sort(), [0, 1], "the chord spans the cell");
  assert.equal(
    interfaceSegments({ nx: 0, ny: 1, clipNx: 0, clipNy: 1, offset: 2 }).length, 0,
    "a plane the cell does not reach contributes no chord, only its own box edges");
});
