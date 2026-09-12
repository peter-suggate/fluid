import assert from "node:assert/strict";
import test from "node:test";
import { productionSceneSliceSeedById } from
  "../lib/methods/adaptive-volume/advance-slice/production-scene-slice";
import { buildSliceLattice, createSliceLattice, latticePlane } from
  "../lib/methods/adaptive-volume/advance-slice/slice-lattice";
import { reconstructSliceSharedRdf } from
  "../lib/methods/adaptive-volume/advance-slice/slice-presentation-publication";
import {
  advanceSlice, clipUnitSquare, createAdvanceSlice, polygonArea, sliceCell, UNIT_SQUARE,
} from "../lib/methods/adaptive-volume/advance-slice/slice-solver";
import {
  FRACTION_FLOOR, fractionReadout, fractionResidueRamp, interfaceSegments,
  rdfMinorityAreaDistorted, sliceRdfPlicFallbackCells,
} from "./lenses";

test("minority-area guard ignores roundoff and rejects amplified corner cuts", () => {
  assert.equal(rdfMinorityAreaDistorted(3.999774, 3.78, 4), true,
    "a 0.000226 accepted air sliver must not become a 0.22-cell corner hole");
  assert.equal(rdfMinorityAreaDistorted(3.999774, 3.9996, 4), false,
    "sub-per-mille cell-area variation remains RDF presentation smoothing");
  assert.equal(rdfMinorityAreaDistorted(0.2, 0.3, 1), false,
    "a resolved minority phase may vary without forcing a PLIC seam");
});

test("frame-four impact keeps supported cells on one shared RDF field", () => {
  const slice = createAdvanceSlice(productionSceneSliceSeedById("coarse-first-pool-impact-half"));
  for (let frame = 0; frame < 4; frame += 1) advanceSlice(slice, { pressureIterations: 28 });
  const rdf = reconstructSliceSharedRdf(slice.topology.accepted, slice.fields,
    slice.numericalTopology);
  const lattice = createSliceLattice(slice); buildSliceLattice(lattice, slice);
  const fallback = sliceRdfPlicFallbackCells(slice, lattice, rdf);
  assert.equal(fallback.size, 0,
    "finite full-capacity RDF cells must not be replaced by owner-local PLIC");
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

test("a coarse cell's PLIC line is drawn where its own fraction puts it", () => {
  /* The offset the resident publishes is measured in finest cells, so a cell on
   * a coarse rung is more than one of them across. Clipping it against the unit
   * square without that scale puts the plane a whole rung outside the box: an
   * 8² cell holding 36% liquid drew as empty. Every accepted cut cell in every
   * scene the lab can seed is checked here, because the error is invisible on
   * the finest rung — where the scale is 1 — and that is the only rung the
   * small scenes have. */
  for (const id of ["hose-tank", "high-resolution-dam-break", "ocean-seiche",
    "coarse-first-pool-impact-half"]) {
    const s = createAdvanceSlice(productionSceneSliceSeedById(id));
    for (let frame = 0; frame < 4; frame += 1) advanceSlice(s, { pressureIterations: 20 });
    const lattice = createSliceLattice(s); buildSliceLattice(lattice, s);
    let coarse = 0;
    for (const cell of lattice.cells) {
      const plane = latticePlane(lattice, cell);
      if (!cell.open || !plane) continue;
      if (cell.width > 1 || cell.height > 1) coarse += 1;
      const drawn = polygonArea(
        clipUnitSquare(UNIT_SQUARE, plane.clipNx, plane.clipNy, plane.offset));
      assert.ok(Math.abs(drawn - cell.fill) < 1e-6,
        `${id}: a ${cell.width}×${cell.height} cell holding ${cell.fill.toFixed(4)}`
        + ` drew ${drawn.toFixed(4)} of itself`);
    }
    assert.ok(coarse > 0, `${id} must exercise at least one cut cell above the finest rung`);
  }
});

test("every drawn normal points from the liquid towards the air", () => {
  /* The record is published y-up about the cell centre and drawn y-down from the
   * corner. A sign lost in that reflection is the failure this overlay would
   * otherwise present as a finding: arrows that all point into the water. */
  const s = createAdvanceSlice(productionSceneSliceSeedById("coarse-first-pool-impact-half"));
  for (let frame = 0; frame < 6; frame += 1) advanceSlice(s, { pressureIterations: 28 });
  const lattice = createSliceLattice(s); buildSliceLattice(lattice, s);
  const fillAt = (x: number, y: number): number => {
    if (x < 0 || y < 0 || x >= s.nx || y >= s.ny) return -1;
    const dense = sliceCell(s, x, y), capacity = s.K[dense]!;
    return capacity > 1e-8 ? s.V[dense]! / capacity : -1;
  };
  let tested = 0, agreed = 0;
  for (const cell of lattice.cells) {
    const plane = latticePlane(lattice, cell);
    if (!cell.open || !plane) continue;
    const magnitude = Math.hypot(plane.nx, plane.ny);
    const cx = cell.x0 + cell.width / 2, cy = cell.y0 + cell.height / 2;
    const reach = Math.max(cell.width, cell.height);
    const air = fillAt(Math.floor(cx + plane.nx / magnitude * reach),
      Math.floor(cy + plane.ny / magnitude * reach));
    const wet = fillAt(Math.floor(cx - plane.nx / magnitude * reach),
      Math.floor(cy - plane.ny / magnitude * reach));
    if (air < 0 || wet < 0 || Math.abs(air - wet) < 1e-3) continue;
    tested += 1;
    if (wet > air) agreed += 1;
  }
  assert.ok(tested > 20, `too few resolved cut cells to conclude anything (${tested})`);
  assert.equal(agreed, tested,
    `${tested - agreed} of ${tested} normals point into the water they came out of`);
});
