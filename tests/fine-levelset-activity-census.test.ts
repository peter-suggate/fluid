import assert from "node:assert/strict";
import test from "node:test";
import { decodeFineLevelSetActivityCensus }
  from "../lib/fine-levelset-activity-census";

test("fine activity census decodes the finalized page-delta receipt", () => {
  const worklist = new Uint32Array([91, 80, 128, 3]);
  const delta = new Uint32Array(16);
  delta.set([14, 91, 24, 40, 12, 10, 2, 4, 10, 3, 1, 0, 0, 24, 10, 1]);
  assert.deepEqual(decodeFineLevelSetActivityCensus(7, worklist, delta, 5), {
    step: 7,
    generation: 91,
    pageDeltaGeneration: 91,
    receiptValid: true,
    liveBandPages: 80,
    dirtyPages: 10,
    dirtyHaloPages: 24,
    supportHaloPages: 40,
    changedPages: 14,
    addedPages: 2,
    retiredPages: 4,
    maximumDisplacementFineCells: 3,
    activeTransportPages: 80,
    sleepingTransportPages: 0,
    transportInputPages: 80,
    transportReceiptValid: true,
    executedSolveIterations: 5,
    transportActivityFraction: 1,
    dirtyFraction: 0.125,
    dirtyHaloFraction: 0.3,
    supportHaloFraction: 0.5,
  });
});

test("fine activity census measures compact Losasso transport activity across retirements", () => {
  const worklist = new Uint32Array([12, 60, 128, 3]);
  const delta = new Uint32Array(16); delta[1] = 12; delta[15] = 1;
  const transport = new Uint32Array(16); transport[3] = 1; transport[14] = 8; transport[15] = 72;
  const census = decodeFineLevelSetActivityCensus(2, worklist, delta, 4, transport)!;
  assert.equal(census.receiptValid, true);
  assert.equal(census.activeTransportPages, 8);
  assert.equal(census.sleepingTransportPages, 72);
  assert.equal(census.transportInputPages, 80);
  assert.equal(census.transportActivityFraction, 0.1);
});

test("fine activity census retains and marks mismatched or unpublished generations", () => {
  const worklist = new Uint32Array([91, 8, 16, 3]);
  const delta = new Uint32Array(16);
  delta[1] = 90;
  delta[15] = 1;
  assert.equal(decodeFineLevelSetActivityCensus(1, worklist, delta, 0)?.receiptValid, false);
  delta[1] = 91;
  delta[15] = 0;
  assert.equal(decodeFineLevelSetActivityCensus(1, worklist, delta, 0)?.receiptValid, false);
});
