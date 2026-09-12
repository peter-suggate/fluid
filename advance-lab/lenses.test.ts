import assert from "node:assert/strict";
import test from "node:test";
import { productionSceneSliceSeedById } from
  "../lib/methods/adaptive-volume/advance-slice/production-scene-slice";
import { createSliceLattice, buildSliceLattice } from
  "../lib/methods/adaptive-volume/advance-slice/slice-lattice";
import { reconstructSliceSharedRdf } from
  "../lib/methods/adaptive-volume/advance-slice/slice-presentation-publication";
import { advanceSlice, createAdvanceSlice } from
  "../lib/methods/adaptive-volume/advance-slice/slice-solver";
import { rdfMinorityAreaDistorted, sliceRdfPlicFallbackCells } from "./lenses";

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
