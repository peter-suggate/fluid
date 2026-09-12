import assert from "node:assert/strict";
import test from "node:test";
import { productionSceneSliceSeedById } from
  "../lib/methods/adaptive-volume/advance-slice/production-scene-slice";
import { reconstructSliceSharedRdf } from
  "../lib/methods/adaptive-volume/advance-slice/slice-presentation-publication";
import { buildSliceLattice, createSliceLattice } from
  "../lib/methods/adaptive-volume/advance-slice/slice-lattice";
import { advanceSlice, createAdvanceSlice, resetAdvanceSlice } from
  "../lib/methods/adaptive-volume/advance-slice/slice-solver";
import { inspectSliceRdfDisplay } from "../advance-lab/lenses";
import { contourQualityReceipt, measureSphereDisplayContour } from
  "./support/advance-slice-contour-quality";

const CENTER_FINE = [32, 36.5] as const;
const RADIUS_FINE = 10;

function quality(slice: ReturnType<typeof createAdvanceSlice>) {
  return measureSphereDisplayContour(slice,
    reconstructSliceSharedRdf(slice.topology.accepted, slice.fields),
    CENTER_FINE, RADIUS_FINE);
}

test("the displayed frame-zero sphere uses one smooth shared RDF contour", () => {
  const slice = createAdvanceSlice(
    productionSceneSliceSeedById("coarse-first-pool-impact-half"));
  const result = quality(slice);
  assert.equal(result.partialCellCount, 68,
    "the production sphere sampling fixture changed");
  assert.equal(result.supportedFallbackCellCount, 0,
    "supported sphere cells must not splice PLIC segments into the shared RDF contour");
  assert.equal(result.plicSegmentCount, 0);
  assert.equal(result.sharedVertexConflictCount, 0,
    "every incident owner must consume the same canonical shared RDF vertex value");
  assert.equal(result.danglingEndpointCount, 0);
  assert.ok(result.maximumEndpointGapFine <= 1e-4);
  assert.ok(result.radialRmsFine <= 0.1,
    `frame-zero sphere radial RMS is ${result.radialRmsFine} fine cells`);
  assert.ok(result.radialMaximumFine <= 0.2,
    `frame-zero sphere radial maximum is ${result.radialMaximumFine} fine cells`);
  assert.ok(result.tangentRmsDegrees <= 7,
    `frame-zero sphere tangent RMS is ${result.tangentRmsDegrees} degrees`);
  assert.ok(result.tangentMaximumDegrees <= 20,
    `frame-zero sphere tangent maximum is ${result.tangentMaximumDegrees} degrees`);
  assert.ok(result.longestNearCollinearRunFine <= 2,
    `frame-zero sphere contains a ${result.longestNearCollinearRunFine}-cell flat facet`);
});

test("one accepted step and reset preserve the shared contour contract", () => {
  const seed = productionSceneSliceSeedById("coarse-first-pool-impact-half");
  let slice = createAdvanceSlice(seed);
  const initial = contourQualityReceipt(quality(slice));
  advanceSlice(slice, { pressureIterations: 4 });
  const stepped = quality(slice);
  assert.ok(stepped.minimumAcceptedMinorityFraction < 1e-4,
    "the accepted first step must cover a resolved sub-per-mille minority phase");
  assert.equal(stepped.supportedFallbackCellCount, 0,
    "the first accepted step must not reintroduce a PLIC/RDF hybrid contour");
  assert.equal(stepped.plicSegmentCount, 0);
  assert.equal(stepped.sharedVertexConflictCount, 0);
  assert.equal(stepped.danglingEndpointCount, 0);
  assert.ok(stepped.maximumEndpointGapFine <= 1e-4);
  assert.ok(stepped.maximumNeighbourTangentJumpDegrees <= 25,
    `the first accepted step has a ${stepped.maximumNeighbourTangentJumpDegrees}-degree corner`);
  slice = resetAdvanceSlice(slice, seed);
  assert.deepEqual(contourQualityReceipt(quality(slice)), initial,
    "reset must reproduce the exact frame-zero contour receipt");
});

test("PLIC fallback is reserved for unsupported cut or nonfinite RDF cells", () => {
  const slice = createAdvanceSlice(productionSceneSliceSeedById("garden-pond"));
  const rdf = reconstructSliceSharedRdf(slice.topology.accepted, slice.fields);
  const lattice = createSliceLattice(slice); buildSliceLattice(lattice, slice);
  const decisions = inspectSliceRdfDisplay(slice, lattice, rdf);
  const plic = decisions.filter(decision => decision.mode === "plic");
  assert.ok(plic.some(decision => decision.reason === "cut-cell"),
    "fixture must exercise the unsupported solid-boundary fallback");
  assert.ok(plic.every(decision => decision.reason === "cut-cell"
    || decision.reason === "nonfinite-rdf"));
  assert.ok(decisions.filter(decision => decision.reason === "shared-rdf")
    .every(decision => decision.mode === "rdf"));
});
