import assert from "node:assert/strict";
import test from "node:test";
import {
  FINE_FLOOD_HISTOGRAM_PASS_BINS,
  FINE_FLOOD_SAMPLE_FLAG_BITS,
  decodeFineFloodClosestPointCode,
  describeFineFloodLadder,
  fineFloodAxisHop,
  fineFloodClosestPoint,
  fineFloodClosestPointCode,
  fineFloodDescendingPassesForReach,
  fineFloodLadderPrefixForReach,
  summarizeFineFloodLadder,
  type FineFloodHistogram,
} from "../lib/fine-flood-provenance";
import { planFineLevelSetJFAStrides } from "../lib/webgpu-octree-fine-levelset-redistance";

const FRACTION_SCALE = 16_777_215;

/** Build the packed word the shader stores, so the decode is tested against the writer's layout. */
function packSampleWord(flags: number, direction: number, fraction: number): number {
  const quantized = Math.round(Math.max(0, Math.min(1, fraction)) * FRACTION_SCALE);
  const code = (direction << 24) | (quantized & 0x00ff_ffff);
  return (flags | (code << FINE_FLOOD_SAMPLE_FLAG_BITS)) >>> 0;
}

function emptyBins(): number[] {
  return Array.from({ length: FINE_FLOOD_HISTOGRAM_PASS_BINS }, () => 0);
}

test("closest-point codes decode to the shader's three cases", () => {
  assert.deepEqual(decodeFineFloodClosestPointCode(0), { kind: "absent", fraction: 0 });
  // Direction 6 is the shader's "phi is exactly zero here" sentinel; 7 is a
  // crossing whose fraction quantized onto the sample centre.
  assert.equal(decodeFineFloodClosestPointCode(6 << 24).kind, "coincident");
  assert.equal(decodeFineFloodClosestPointCode(7 << 24).kind, "coincident");
  const axis = decodeFineFloodClosestPointCode((3 << 24) | Math.round(0.25 * FRACTION_SCALE));
  assert.equal(axis.kind, "axis");
  assert.equal(axis.direction, 3);
  assert.ok(Math.abs(axis.fraction - 0.25) < 1e-6);
});

test("a packed sample word yields both its state bits and its closest point", () => {
  const word = packSampleWord(0b101, 1, 0.5);
  assert.equal(word & ((1 << FINE_FLOOD_SAMPLE_FLAG_BITS) - 1), 0b101);
  const point = fineFloodClosestPoint([4, 9, 2], fineFloodClosestPointCode(word));
  // Direction 1 is +x, so the crossing sits half a cell past the centre.
  assert.ok(Math.abs(point[0] - 5.0) < 1e-5);
  assert.ok(Math.abs(point[1] - 9.5) < 1e-9);
  assert.ok(Math.abs(point[2] - 2.5) < 1e-9);
});

test("a coincident code leaves the closest point on the seed centre", () => {
  assert.deepEqual(fineFloodClosestPoint([1, 2, 3], 6 << 24), [1.5, 2.5, 3.5]);
});

test("the hop is Chebyshev, because the flood gathers over a cube", () => {
  assert.equal(fineFloodAxisHop([0, 0, 0], [3, -5, 1]), 5);
  assert.equal(fineFloodAxisHop([7, 7, 7], [7, 7, 7]), 0);
});

test("descending passes follow the ladder's own 2P-1 reach rule", () => {
  assert.equal(fineFloodDescendingPassesForReach(0), 0);
  assert.equal(fineFloodDescendingPassesForReach(1), 1);
  // Reach 1 covers hop 1; hop 2 needs a first stride of 2, so two passes.
  assert.equal(fineFloodDescendingPassesForReach(2), 2);
  assert.equal(fineFloodDescendingPassesForReach(3), 2);
  assert.equal(fineFloodDescendingPassesForReach(4), 3);
  assert.equal(fineFloodDescendingPassesForReach(7), 3);
  assert.equal(fineFloodDescendingPassesForReach(8), 4);
});

test("required passes agree with the ladder the planner sizes for the same reach", () => {
  // The planner sizes a cold ladder to cover the whole band. Asking this module
  // how many passes that band needs must return the ladder it actually built,
  // or the surplus reported to the user is measuring two different things.
  for (const band of [1, 2, 3, 4, 7, 8, 15, 16, 23, 31, 64]) {
    const strides = planFineLevelSetJFAStrides(band, band, 0);
    assert.equal(
      fineFloodDescendingPassesForReach(band), strides.length,
      `band ${band} planned ${strides.length} passes`,
    );
  }
});

test("a ladder splits into its descending part and its collar repairs", () => {
  const plan = describeFineFloodLadder(planFineLevelSetJFAStrides(23, 23, 5));
  assert.deepEqual([...plan.strides], [16, 8, 4, 2, 1, 1, 1, 1, 1, 1]);
  assert.equal(plan.descendingPasses, 5);
  assert.equal(plan.collarRepairPasses, 5);
  assert.equal(plan.descendingReach, 31);
  // The repairs extend reach by one cell each, so the schedule reaches further
  // than the descending part alone. Judging a hop against 2P - 1 alone reports
  // an impossible negative surplus for hops the repairs legitimately closed.
  assert.equal(plan.encodedReach, 36);
  assert.deepEqual([...plan.prefixReach], [16, 24, 28, 30, 31, 32, 33, 34, 35, 36]);
});

test("the covering prefix is measured against the ladder that actually ran", () => {
  // The mini dam's recurring schedule: four descending passes reaching 15, then
  // two collar repairs that carry it to 17.
  const plan = describeFineFloodLadder([8, 4, 2, 1, 1, 1]);
  assert.equal(plan.encodedReach, 17);
  assert.equal(fineFloodLadderPrefixForReach(plan.prefixReach, 0), 0);
  assert.equal(fineFloodLadderPrefixForReach(plan.prefixReach, 8), 1);
  assert.equal(fineFloodLadderPrefixForReach(plan.prefixReach, 9), 2);
  assert.equal(fineFloodLadderPrefixForReach(plan.prefixReach, 15), 4);
  assert.equal(fineFloodLadderPrefixForReach(plan.prefixReach, 16), 5);
  // A hop deeper than the whole ladder saturates instead of naming a pass that
  // was never encoded; `coveredByEncodedLadder` is what reports the shortfall.
  assert.equal(fineFloodLadderPrefixForReach(plan.prefixReach, 999), 6);
});

test("a ladder longer than the encoder's budget is rejected", () => {
  assert.throws(() => describeFineFloodLadder(Array.from({ length: 14 }, () => 1)), RangeError);
});

test("a single-stride ladder is all descending", () => {
  const plan = describeFineFloodLadder([1]);
  assert.equal(plan.descendingPasses, 1);
  assert.equal(plan.collarRepairPasses, 0);
});

test("non-power-of-two strides are rejected rather than silently binned", () => {
  assert.throws(() => describeFineFloodLadder([3, 1]), RangeError);
  assert.throws(() => describeFineFloodLadder([]), RangeError);
});

test("a flood whose hops stay short reports the trailing passes it did not need", () => {
  const bins = emptyBins();
  bins[0] = 800;   // self-seeded interface samples
  bins[1] = 150;   // covered by the first encoded pass
  bins[2] = 50;    // covered by the second
  const histogram: FineFloodHistogram = { bins, unresolved: 0, maximumAxisHop: 3 };
  const summary = summarizeFineFloodLadder({
    strides: planFineLevelSetJFAStrides(23, 23, 5),
    resident: 1_200,
    histogram,
  });
  assert.equal(summary.resolved, 1_000);
  assert.equal(summary.selfSeeded, 800);
  // The first pass of a band-23 ladder strides 16, so it alone covers a hop of 3.
  assert.equal(summary.requiredLadderPasses, 1);
  assert.equal(summary.surplusLadderPasses, 9);
  assert.equal(summary.coveredByEncodedLadder, true);
  // A ladder built from scratch for a reach of 3 would be two passes; that is a
  // different question from how much of this ladder ran, and is reported apart.
  assert.equal(summary.idealLadderPasses, 2);
  // Resident counts samples with no distance this generation, so the shares
  // must normalise over the resolved set rather than over residency.
  assert.ok(Math.abs(summary.cumulativeResolvedShare[0] - 0.8) < 1e-9);
  assert.ok(Math.abs(summary.cumulativeResolvedShare[2] - 1) < 1e-9);
});

test("a hop deeper than the whole ladder is reported, not clamped away", () => {
  const bins = emptyBins();
  bins[6] = 100;
  const summary = summarizeFineFloodLadder({
    strides: [8, 4, 2, 1, 1, 1],
    resident: 100,
    histogram: { bins, unresolved: 0, maximumAxisHop: 40 },
  });
  assert.equal(summary.coveredByEncodedLadder, false);
  assert.equal(summary.requiredLadderPasses, 6);
  assert.equal(summary.surplusLadderPasses, 0);
});

test("unresolved samples stay out of every pass bin", () => {
  const bins = emptyBins();
  bins[0] = 10;
  const summary = summarizeFineFloodLadder({
    strides: [1],
    resident: 100,
    histogram: { bins, unresolved: 90, maximumAxisHop: 0 },
  });
  assert.equal(summary.resolved, 10);
  assert.equal(summary.unresolved, 90);
  assert.equal(summary.requiredLadderPasses, 0);
});

test("a histogram of the wrong width is rejected", () => {
  assert.throws(() => summarizeFineFloodLadder({
    strides: [1], resident: 1, histogram: { bins: [1, 2], unresolved: 0, maximumAxisHop: 0 },
  }), RangeError);
});
