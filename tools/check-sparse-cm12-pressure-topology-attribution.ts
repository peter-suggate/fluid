#!/usr/bin/env node
/** Static gate for prior-frame pressure-topology performance attribution. */
import assert from "node:assert/strict";
import { adaptiveMassPressureTopologyChip } from
  "../lib/methods/adaptive-mass/adaptive-mass-frame-pipeline";
import { sparseCM12PressureTopologyAttribution } from
  "../lib/methods/adaptive-mass/sparse-cm12-pressure-topology-attribution";

const pcm = (generation = 12, fault = 0) => ({
  cell: { phase: 1, fault, firstFault: 0xffff_ffff, dirtyCount: 3,
    totalCount: 80, candidateGeneration: generation, acceptedGeneration: generation },
  row: { phase: 1, fault, firstFault: 0xffff_ffff, dirtyCount: 5,
    totalCount: 140, candidateGeneration: generation, acceptedGeneration: generation },
});
const work = { acceptedCellCount: 100, acceptedRowCount: 220,
  pressureCellCount: 80, pressureActiveRowCount: 140, pcm: pcm() };

const adjacent = sparseCM12PressureTopologyAttribution({
  prior: { encodedStep: 9, topologyGeneration: 17, committedBrickCount: 4 },
  current: { encodedStep: 10, topologyGeneration: 18, committedBrickCount: 2 },
  work,
});
assert.equal(adjacent.status, "matched");
assert.equal(adjacent.inputTopologyGeneration, 17);
assert.equal(adjacent.priorCommittedBrickCount, 4);
assert.equal(adjacent.currentEndFrameTopologyGeneration, 18);
assert.equal(adjacent.currentEndFrameCommittedBrickCount, 2);
assert.equal(adjacent.pcmMatched, true);

const unchangedGap = sparseCM12PressureTopologyAttribution({
  prior: { encodedStep: 3, topologyGeneration: 17, committedBrickCount: 9 },
  current: { encodedStep: 8, topologyGeneration: 17, committedBrickCount: 0 },
  work,
});
assert.equal(unchangedGap.status, "matched");
assert.equal(unchangedGap.inputTopologyGeneration, 17);
assert.equal(unchangedGap.priorCommittedBrickCount, 0);

const changedGap = sparseCM12PressureTopologyAttribution({
  prior: { encodedStep: 3, topologyGeneration: 17, committedBrickCount: 0 },
  current: { encodedStep: 8, topologyGeneration: 19, committedBrickCount: 1 },
  work: { ...work, pcm: pcm(12, 1) },
});
assert.equal(changedGap.status, "unavailable");
assert.equal(changedGap.inputTopologyGeneration, undefined);
assert.equal(changedGap.priorCommittedBrickCount, undefined);
assert.equal(changedGap.pcmMatched, false);

const chip = adaptiveMassPressureTopologyChip({
  adaptivePressureTopologyAttribution: adjacent,
  adaptiveAcceptedSameLevelCoarseRowCount: 12,
  adaptiveAcceptedMixedSeamRowCount: 3,
});
assert.match(chip, /Input topology gen 17 · prior commit 4 bricks/);
assert.match(chip, /Matched work: accepted 100 cells \/ 220 rows/);
assert.match(chip, /PCM gen 12\/12 · dirty leaves 3\/5 · matched/);
assert.match(chip, /End-frame → topology gen 18 · 2 committed bricks \(next repair input\)/);

console.log("Sparse CM12 pressure-topology attribution: PASS");
