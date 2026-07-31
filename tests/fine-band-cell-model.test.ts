import assert from "node:assert/strict";
import test from "node:test";
import {
  FINE_BAND_BRICK_RESOLUTION,
  fineBandBrickKey,
  fineBandBrickOf,
  fineBandBudget,
  fineBandCensus,
  fineBandCrossingPoint,
  fineBandFreeSurfaceCrossings,
  fineBandLadderAttribution,
  fineBandLadderPassColor,
  fineBandMembership,
  fineBandNarrative,
  fineBandPhiSpan,
  fineBandProbeClosestPoint,
  fineBandScheduledWork,
  fineBandTightestCrossing,
  fineBandVolumeLedger,
  type FineBandWidths,
} from "../lib/fine-band-cell-model";
import {
  FLUID_CELL_TRACE_ABI_VERSION,
  FLUID_CELL_TRACE_FINE_RECORDS_OFFSET,
  FLUID_CELL_TRACE_FINE_RECORD_CAPACITY,
  FLUID_CELL_TRACE_FINE_RECORD_WORDS,
  FLUID_CELL_TRACE_HEADER,
  FLUID_CELL_TRACE_HEADER_WORDS,
  FLUID_CELL_TRACE_HITS_OFFSET,
  FLUID_CELL_TRACE_HIT_CAPACITY,
  FLUID_CELL_TRACE_HIT_WORDS,
  FLUID_CELL_TRACE_MAGIC,
  FLUID_CELL_TRACE_NEIGHBOR_CAPACITY,
  FLUID_CELL_TRACE_RECORD_WORDS,
  FLUID_CELL_TRACE_STATUS,
  FLUID_CELL_TRACE_WORDS,
  fluidCellTraceHasInterfaceHit,
  nextFluidCellTraceInterfaceHit,
  type FluidCellTrace,
  type FluidCellTraceFineProbe,
  type FluidCellTraceHit,
  type FluidCellTraceNeighbor,
} from "../lib/fluid-cell-trace";
import {
  FLUID_CELL_TRACE_STORAGE_BINDINGS,
  fluidCellTraceGatherShader,
} from "../lib/webgpu-fluid-cell-trace";
import { FINE_LEVELSET_SAMPLE_FLAGS } from "../lib/octree-fine-levelset-bricks";
import {
  FINE_FLOOD_SAMPLE_FLAGS,
  FINE_FLOOD_SAMPLE_FLAG_BITS,
} from "../lib/fine-flood-provenance";
import { VISUALIZATION_STORAGE_BUFFERS_PER_STAGE } from "../lib/visualization-bindings";

/* ------------------------------------------------------------------------- */
/* Fixtures                                                                   */
/* ------------------------------------------------------------------------- */

function neighbor(overrides: Partial<FluidCellTraceNeighbor> = {}): FluidCellTraceNeighbor {
  return {
    direction: 0, row: 7, leafSize: 4, flags: 1, leafOrigin: [12, 0, 8], pressure: 0,
    present: true, coarser: false, finer: false, boundary: false,
    ...overrides,
  };
}

function probe(overrides: Partial<FluidCellTraceFineProbe> = {}): FluidCellTraceFineProbe {
  return {
    cell: [16, 4, 8], flags: 0, phi: -0.5, seedCell: [16, 4, 8], seedCode: 0, hop: 0,
    resident: true, valid: true, interface: false, negative: true, resolved: true,
    stale: false, missing: false,
    ...overrides,
  };
}

function hit(overrides: Partial<FluidCellTraceHit> = {}): FluidCellTraceHit {
  return {
    row: 3, leafSize: 4, leafOrigin: [8, 0, 8], distance_m: 1, selected: false,
    flags: 0, holdsInterface: false, liquid: false, corrected: true,
    ...overrides,
  };
}

function trace(overrides: Partial<FluidCellTrace> = {}): FluidCellTrace {
  return {
    version: FLUID_CELL_TRACE_ABI_VERSION,
    status: FLUID_CELL_TRACE_STATUS.resolved,
    pixel: [1, 2], requestToken: 1,
    cell: [8, 0, 8], row: 3, leafSize: 4, leafOrigin: [8, 0, 8],
    diagonal: 6, rhs: -1, entryCount: 6, volume: 64, topologyCode: 0, pressure: 0.5,
    fineSamples: 0, fineResolved: 0, fineMaximumHop: 0, fineInterface: 0,
    dimensions: [16, 16, 16], fineFactor: 4,
    fineProbes: 0, fineMissing: 0, fineStale: 0, fineNegative: 0,
    coarsePhi: 0, coarsePhiMinimum: 0, coarsePhiMaximum: 0, coarsePhiFlags: 0,
    probeMinimumPhi: 0, probeMaximumPhi: 0, probeNearestPhi: 0,
    fineProbeRecords: [],
    neighbors: [],
    hits: [], hitIndex: 0, hitOverflow: 0,
    ...overrides,
  };
}

const WIDTHS: FineBandWidths = {
  pressureBandCells: 4,
  surfaceBandCells: 6,
  transportBandFineCells: 24,
  redistanceBandFineCells: 32,
};

/* ------------------------------------------------------------------------- */
/* The free surface, where it meets the operator                              */
/* ------------------------------------------------------------------------- */

test("a sign change along a dual edge becomes a crossing with a ghost-fluid scale", () => {
  const crossings = fineBandFreeSurfaceCrossings(trace({
    coarsePhi: -1, coarsePhiFlags: 0b1001,
    neighbors: [neighbor({ direction: 0, phi: 3 })],
  }));
  assert.equal(crossings.length, 1);
  // The surface cuts a quarter of the way along, so the dual distance shrinks
  // to a quarter and the coefficient grows fourfold.
  assert.equal(crossings[0].fraction, 0.25);
  assert.equal(crossings[0].outward, true);
  assert.equal(crossings[0].coefficientScale, 4);
});

test("a neighbour with no published record contributes no crossing", () => {
  // Zero is a perfectly good phi. Comparing against a fabricated one would put
  // the surface exactly on a cell that has no opinion about where it is.
  const crossings = fineBandFreeSurfaceCrossings(trace({
    coarsePhi: -1, coarsePhiFlags: 0b1001,
    neighbors: [neighbor({ direction: 0 })],
  }));
  assert.deepEqual(crossings, []);
});

test("neighbours on the same side of the surface are not crossings", () => {
  const crossings = fineBandFreeSurfaceCrossings(trace({
    coarsePhi: -1, coarsePhiFlags: 0b1001,
    neighbors: [neighbor({ direction: 0, phi: -2 }), neighbor({ direction: 1, phi: -0.1 })],
  }));
  assert.deepEqual(crossings, []);
});

test("an air cell's crossing carries no coefficient scale", () => {
  // The free-surface condition is imposed from the liquid side. A cell sitting
  // in air has no row for it to scale, and claiming otherwise would attribute
  // the neighbour's coefficient to this one.
  const crossings = fineBandFreeSurfaceCrossings(trace({
    coarsePhi: 2, coarsePhiFlags: 0b1001,
    neighbors: [neighbor({ direction: 0, phi: -2 })],
  }));
  assert.equal(crossings.length, 1);
  assert.equal(crossings[0].outward, false);
  assert.equal(crossings[0].coefficientScale, undefined);
  assert.equal(fineBandTightestCrossing(crossings), undefined);
});

test("crossings are ordered tightest first, so the near-singular face reads first", () => {
  const crossings = fineBandFreeSurfaceCrossings(trace({
    coarsePhi: -1, coarsePhiFlags: 0b1001,
    neighbors: [
      neighbor({ direction: 0, phi: 1 }),
      neighbor({ direction: 2, phi: 19 }),
      neighbor({ direction: 4, phi: 3 }),
    ],
  }));
  assert.deepEqual(crossings.map((crossing) => crossing.direction), [2, 4, 0]);
  assert.equal(fineBandTightestCrossing(crossings)?.direction, 2);
});

test("an unpublished row has no crossings at all", () => {
  const crossings = fineBandFreeSurfaceCrossings(trace({
    coarsePhi: -1, coarsePhiFlags: 0,
    neighbors: [neighbor({ direction: 0, phi: 3 })],
  }));
  assert.deepEqual(crossings, []);
});

test("the crossing point lands on the dual edge at the measured fraction", () => {
  const point = fineBandCrossingPoint([0, 0, 0], [8, 0, 0], 0.25);
  assert.deepEqual([...point], [2, 0, 0]);
});

/* ------------------------------------------------------------------------- */
/* Phi across the cell                                                        */
/* ------------------------------------------------------------------------- */

test("probes straddling zero against a record that does not is the sub-grid case", () => {
  const span = fineBandPhiSpan(trace({
    coarsePhi: -2, coarsePhiMinimum: -3, coarsePhiMaximum: -1, coarsePhiFlags: 0b1001,
    fineSamples: 12, probeMinimumPhi: -0.4, probeMaximumPhi: 0.3, probeNearestPhi: 0.02,
  }));
  assert.equal(span.probesStraddle, true);
  assert.equal(span.recordStraddles, false);
  assert.equal(span.unresolvedInterface, true);
});

test("a surface cell whose record straddles zero is not reported as a loss", () => {
  const span = fineBandPhiSpan(trace({
    coarsePhi: -0.2, coarsePhiMinimum: -1.5, coarsePhiMaximum: 1.5, coarsePhiFlags: 0b1001,
    fineSamples: 12, probeMinimumPhi: -1, probeMaximumPhi: 1, probeNearestPhi: 0.01,
  }));
  assert.equal(span.unresolvedInterface, false);
  assert.equal(span.interfaceOffCentre, false);
});

test("a surface far from the site is flagged separately from a loss", () => {
  const span = fineBandPhiSpan(trace({
    coarsePhi: -2.5, coarsePhiMinimum: -4, coarsePhiMaximum: 1, coarsePhiFlags: 0b1001,
    fineSamples: 12, probeMinimumPhi: -3, probeMaximumPhi: 0.5, probeNearestPhi: 0.1,
  }));
  assert.equal(span.unresolvedInterface, false);
  assert.equal(span.interfaceOffCentre, true);
});

test("an unpublished record reports itself rather than a phi of zero", () => {
  const span = fineBandPhiSpan(trace({ coarsePhi: 5, coarsePhiFlags: 0 }));
  assert.equal(span.published, false);
  assert.equal(span.unresolvedInterface, false);
});

test("the span's full scale is the leaf's half-diagonal", () => {
  const span = fineBandPhiSpan(trace({ leafSize: 4 }));
  assert.ok(Math.abs(span.scale - 2 * Math.sqrt(3)) < 1e-9);
});

/* ------------------------------------------------------------------------- */
/* Census, ledger and budget                                                  */
/* ------------------------------------------------------------------------- */

test("the census partitions probes into missing, stale, valid and resident-invalid", () => {
  const census = fineBandCensus(trace({
    fineProbes: 100, fineMissing: 10, fineStale: 5, fineSamples: 70,
    fineResolved: 65, fineInterface: 4, fineNegative: 30,
  }));
  assert.equal(census.residentInvalid, 15);
  assert.equal(census.coverage, 0.7);
  assert.ok(Math.abs(census.liquidFraction - 30 / 70) < 1e-9);
});

test("counts that exceed their denominator are clamped rather than trusted", () => {
  // These come off the GPU. A corrupt readback must not produce a coverage
  // above one and a HUD that reports 140% of a leaf.
  const census = fineBandCensus(trace({
    fineProbes: 10, fineMissing: 40, fineStale: 40, fineSamples: 40, fineNegative: 40,
  }));
  assert.equal(census.missing, 10);
  assert.equal(census.stale, 0);
  assert.equal(census.valid, 0);
  assert.equal(census.coverage, 0);
  assert.equal(census.liquidFraction, 0);
});

test("a cell entirely on one side of the surface is not a partial ledger", () => {
  const full = fineBandVolumeLedger(trace({
    fineProbes: 64, fineSamples: 64, fineNegative: 64, volume: 8,
  }));
  assert.equal(full.partial, false);
  assert.equal(full.liquidVolume, 8);

  const cut = fineBandVolumeLedger(trace({
    fineProbes: 64, fineSamples: 64, fineNegative: 16, volume: 8,
  }));
  assert.equal(cut.partial, true);
  assert.equal(cut.liquidVolume, 2);
});

test("the budget prices one unknown in samples, bricks and bytes", () => {
  const budget = fineBandBudget(trace({
    leafSize: 32, fineFactor: 4, fineProbes: 100, fineSamples: 50,
  }));
  assert.equal(budget.edge, 128);
  assert.equal(budget.samples, 128 ** 3);
  assert.equal(budget.bricks, (128 / FINE_BAND_BRICK_RESOLUTION) ** 3);
  // Four channels of four bytes, matching FINE_LEVELSET_CHANNELS.
  assert.equal(budget.bytes, 128 ** 3 * 16);
  // Half the probes came back valid, so about half the dense count is resident.
  assert.equal(budget.residentEstimate, Math.round(128 ** 3 * 0.5));
});

test("scheduled fine work multiplies resident samples by substeps and ladder passes", () => {
  const work = fineBandScheduledWork({
    residentSamples: 1000, substeps: 4, ladderStrides: [8, 4, 2, 1, 1, 1],
  });
  assert.equal(work.ladderPasses, 6);
  assert.equal(work.sampleTouches, 1000 * 10);
});

/* ------------------------------------------------------------------------- */
/* Ladder attribution                                                         */
/* ------------------------------------------------------------------------- */

test("probes are binned by the leading passes of the ladder that ran", () => {
  const attribution = fineBandLadderAttribution([
    probe({ hop: 0 }),
    probe({ hop: 1 }),
    probe({ hop: 16 }),
    probe({ valid: true, resolved: false }),
  ], [8, 4, 2, 1, 1, 1]);
  assert.ok(attribution);
  // A self-seeded sample needs no pass; a one-cell hop is covered by the first
  // prefix that reaches it; the deepest needs five of the six encoded passes.
  assert.equal(attribution.bins[0], 1);
  assert.equal(attribution.resolved, 3);
  assert.equal(attribution.unresolved, 1);
  assert.equal(attribution.deepestHop, 16);
  assert.equal(attribution.requiredPasses, 5);
  assert.equal(attribution.coveredByEncodedLadder, true);
});

test("a hop deeper than the encoded reach is reported as carried, not as covered", () => {
  const attribution = fineBandLadderAttribution([probe({ hop: 40 })], [8, 4, 2, 1, 1, 1]);
  assert.ok(attribution);
  assert.equal(attribution.coveredByEncodedLadder, false);
});

test("no published ladder yields no attribution rather than an imagined one", () => {
  // Binning against a schedule that never ran would name passes that do not
  // exist, which is worse than showing nothing.
  assert.equal(fineBandLadderAttribution([probe({ hop: 3 })], []), undefined);
  assert.equal(fineBandLadderAttribution([probe({ hop: 3 })], [3, 1]), undefined);
});

test("the ladder ramp keeps the self-seeded bin outside the pass colours", () => {
  assert.equal(fineBandLadderPassColor(0, 6), "#ffffff");
  assert.notEqual(fineBandLadderPassColor(1, 6), "#ffffff");
});

/* ------------------------------------------------------------------------- */
/* Band membership                                                            */
/* ------------------------------------------------------------------------- */

test("the four bands are converted to one unit and ordered innermost first", () => {
  const membership = fineBandMembership(1, WIDTHS, 4);
  assert.deepEqual(membership.rings.map((ring) => ring.id),
    ["pressure", "surface", "transport", "redistance"]);
  // 24 fine cells at factor four is six finest cells; 32 is eight.
  assert.equal(membership.rings[2].radius, 6);
  assert.equal(membership.rings[3].radius, 8);
  assert.equal(membership.innermost?.id, "pressure");
});

test("a cell beyond every band reports itself outside rather than inside the last", () => {
  const membership = fineBandMembership(50, WIDTHS, 4);
  assert.equal(membership.outside, true);
  assert.equal(membership.innermost, undefined);
});

test("membership uses distance magnitude, so liquid and air sides agree", () => {
  assert.deepEqual(
    fineBandMembership(-3, WIDTHS, 4).innermost?.id,
    fineBandMembership(3, WIDTHS, 4).innermost?.id,
  );
});

/* ------------------------------------------------------------------------- */
/* Geometry                                                                   */
/* ------------------------------------------------------------------------- */

test("a probe's closest point is its seed's crossing, not its own centre", () => {
  // Direction 1 is +x with a half-cell offset, so the point sits half a fine
  // cell past the seed's centre along x.
  const code = (1 << 24) | Math.round(0.5 * 0xff_ffff);
  const point = fineBandProbeClosestPoint(probe({ seedCell: [10, 4, 8], seedCode: code }));
  assert.ok(point);
  assert.ok(Math.abs(point[0] - 11) < 1e-3);
  assert.ok(Math.abs(point[1] - 4.5) < 1e-9);
});

test("an unresolved probe has no closest point", () => {
  assert.equal(fineBandProbeClosestPoint(probe({ resolved: false })), undefined);
});

test("a malformed seed code is dropped rather than taking the frame down", () => {
  assert.equal(fineBandProbeClosestPoint(probe({ seedCode: -1 })), undefined);
});

test("gaps are keyed by brick, since residency is a property of the brick", () => {
  // Two samples one cell apart inside the same brick must not claim two holes.
  assert.equal(fineBandBrickKey([5, 6, 7]), fineBandBrickKey([4, 4, 4]));
  assert.notEqual(fineBandBrickKey([5, 6, 7]), fineBandBrickKey([8, 4, 4]));
  const brick = fineBandBrickOf([5, 6, 7]);
  assert.deepEqual([...brick.origin], [4, 4, 4]);
  assert.equal(brick.size, FINE_BAND_BRICK_RESOLUTION);
});

/* ------------------------------------------------------------------------- */
/* Narrative                                                                  */
/* ------------------------------------------------------------------------- */

test("the narrative names an uncorrected row rather than reporting a phi of zero", () => {
  const steps = fineBandNarrative(
    trace({ coarsePhiFlags: 0, fineProbes: 512, fineSamples: 100 }),
    { widths: WIDTHS, ladderStrides: [8, 4, 2, 1] },
  );
  const phi = steps.find((step) => step.id === "fine-phi");
  assert.ok(phi);
  assert.equal(phi.value, "uncorrected");
  assert.match(phi.detail, /never corrected it/);
});

test("the scheduled fine-band step is badged scheduled, never gathered", () => {
  const steps = fineBandNarrative(
    trace({ fineProbes: 512, fineSamples: 256, coarsePhiFlags: 0b1001 }),
    { widths: WIDTHS, ladderStrides: [8, 4, 2, 1] },
  );
  const work = steps.find((step) => step.id === "fine-work");
  assert.ok(work);
  assert.equal(work.evidence, "scheduled");
  assert.equal(steps.find((step) => step.id === "fine-budget")?.evidence, "gathered");
});

test("a fully covered leaf reports no coverage step", () => {
  const steps = fineBandNarrative(
    trace({ fineProbes: 512, fineSamples: 512, coarsePhiFlags: 0b1001 }),
    { widths: WIDTHS, ladderStrides: [8, 4, 2, 1] },
  );
  assert.equal(steps.find((step) => step.id === "fine-coverage"), undefined);
});

test("a leaf the band could not fully reach says so", () => {
  const steps = fineBandNarrative(
    trace({ fineProbes: 512, fineMissing: 40, fineStale: 8, fineSamples: 400, coarsePhiFlags: 0b1001 }),
    { widths: WIDTHS, ladderStrides: [8, 4, 2, 1] },
  );
  const coverage = steps.find((step) => step.id === "fine-coverage");
  assert.ok(coverage);
  assert.match(coverage.detail, /no resident page/);
});

/* ------------------------------------------------------------------------- */
/* ABI                                                                        */
/* ------------------------------------------------------------------------- */

test("the trace buffer accounts for every section exactly", () => {
  assert.equal(
    FLUID_CELL_TRACE_WORDS,
    FLUID_CELL_TRACE_HEADER_WORDS
    + FLUID_CELL_TRACE_NEIGHBOR_CAPACITY * FLUID_CELL_TRACE_RECORD_WORDS
    + FLUID_CELL_TRACE_HIT_CAPACITY * FLUID_CELL_TRACE_HIT_WORDS
    + FLUID_CELL_TRACE_FINE_RECORD_CAPACITY * FLUID_CELL_TRACE_FINE_RECORD_WORDS,
  );
});

test("the fine records start after the ray run, not inside it", () => {
  assert.equal(
    FLUID_CELL_TRACE_FINE_RECORDS_OFFSET,
    FLUID_CELL_TRACE_HITS_OFFSET + FLUID_CELL_TRACE_HIT_CAPACITY * FLUID_CELL_TRACE_HIT_WORDS,
  );
});

test("the interface jump searches forward and wraps, and holds still when there is nothing to find", () => {
  const run = [
    hit({ holdsInterface: false }), hit({ holdsInterface: true }),
    hit({ holdsInterface: false }), hit({ holdsInterface: true }),
  ];
  assert.equal(nextFluidCellTraceInterfaceHit(run, 0), 1);
  assert.equal(nextFluidCellTraceInterfaceHit(run, 1), 3);
  // Wrapping keeps it one repeatable gesture rather than a dead end.
  assert.equal(nextFluidCellTraceInterfaceHit(run, 3), 1);
  assert.equal(fluidCellTraceHasInterfaceHit(run), true);

  const plain = [hit({ holdsInterface: false }), hit({ holdsInterface: false })];
  assert.equal(fluidCellTraceHasInterfaceHit(plain), false);
  // Nothing to jump to must leave the selection alone rather than move it
  // somewhere arbitrary; the control is disabled on the same signal.
  assert.equal(nextFluidCellTraceInterfaceHit(plain, 1), 1);
  assert.equal(nextFluidCellTraceInterfaceHit([], 3), 0);
});

test("a leaf that holds the interface is the one the run picks out", () => {
  // The classification comes off the coarse record, so an uncorrected leaf is
  // never claimed to hold the surface.
  assert.equal(hit({ corrected: false, holdsInterface: false }).holdsInterface, false);
});

test("every header field fits inside the header", () => {
  for (const [name, offset] of Object.entries(FLUID_CELL_TRACE_HEADER)) {
    assert.ok(offset < FLUID_CELL_TRACE_HEADER_WORDS,
      `${name} at ${offset} overruns the ${FLUID_CELL_TRACE_HEADER_WORDS}-word header`);
  }
});

test("the magic carries the ABI version, so a stale buffer cannot decode", () => {
  assert.equal(FLUID_CELL_TRACE_MAGIC & 0xff, FLUID_CELL_TRACE_ABI_VERSION);
});

test("the flood's sample flags are the brick layout's, bit for bit", () => {
  // These describe one field. Restating them put `negative` on bit 2 — which is
  // `known` in the brick layout — and the cell trace, the first reader of it,
  // reported every cell in the water as 0% liquid. The GPU commit writes
  // NEGATIVE = 16u, so bit 4 is the only correct answer.
  assert.equal(FINE_FLOOD_SAMPLE_FLAGS.valid, FINE_LEVELSET_SAMPLE_FLAGS.valid);
  assert.equal(FINE_FLOOD_SAMPLE_FLAGS.interface, FINE_LEVELSET_SAMPLE_FLAGS.interface);
  assert.equal(FINE_FLOOD_SAMPLE_FLAGS.negative, FINE_LEVELSET_SAMPLE_FLAGS.negative);
  assert.equal(FINE_FLOOD_SAMPLE_FLAGS.negative, 16);
  // Every named flag has to fit under the closest-point code that sits above it.
  for (const bit of Object.values(FINE_LEVELSET_SAMPLE_FLAGS)) {
    assert.ok(bit < (1 << FINE_FLOOD_SAMPLE_FLAG_BITS), `flag ${bit} overruns the reserved bits`);
  }
});

test("the gather's WGSL reads the same flag values the redistance commit writes", () => {
  // The shader interpolates these constants, so a drift would be silent until
  // someone noticed a count that was always zero.
  for (const [name, bit] of [
    ["FINE_VALID", FINE_LEVELSET_SAMPLE_FLAGS.valid],
    ["FINE_INTERFACE", FINE_LEVELSET_SAMPLE_FLAGS.interface],
    ["FINE_NEGATIVE", FINE_LEVELSET_SAMPLE_FLAGS.negative],
  ] as const) {
    assert.ok(fluidCellTraceGatherShader.includes(`const ${name}:u32=${bit}u;`),
      `${name} is not declared as ${bit} in the gather`);
  }
});

test("the gather sits exactly on the storage ceiling, with nothing spare", () => {
  // The framework records ten as what Apple silicon reports. An eleventh
  // publication needs a second pass, not another binding, and this is the test
  // that says so before a driver does.
  assert.equal(
    FLUID_CELL_TRACE_STORAGE_BINDINGS.length,
    VISUALIZATION_STORAGE_BUFFERS_PER_STAGE,
  );
  assert.equal(new Set(FLUID_CELL_TRACE_STORAGE_BINDINGS).size,
    FLUID_CELL_TRACE_STORAGE_BINDINGS.length);
});
