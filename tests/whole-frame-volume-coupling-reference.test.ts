import assert from "node:assert/strict";
import test from "node:test";

import { referenceSharpenVolume, referenceWholeFrameVolumeCoupling,
  sharpeningFaceHasLiquidConnection, type VolumeBoxCell } from
  "../lib/methods/adaptive-volume/whole-frame-volume-coupling-reference";
import { createGeometricVolumeResidentWGSL, WHOLE_FRAME_VOLUME_ENTRY_POINTS,
  type SparseGeometricVolumeLayout } from
  "../lib/methods/adaptive-volume/resident-volume.wgsl";

const cell = (x0: number, x1: number, capacity: number, amount: number): VolumeBoxCell => ({
  minimum: [x0, 0, 0], maximum: [x1, 1, 1],
  donorCapacity: capacity, receiverCapacity: capacity, amount,
});

test("stationary mixed-width boxes are an identity and conserve extensive volume", () => {
  const cells = [cell(0, 1, 1, .25), cell(1, 3, 2, 1.5), cell(3, 4, 1, .75)];
  const result = referenceWholeFrameVolumeCoupling(cells, cells.map(() => [0, 0, 0] as const));
  assert.deepEqual(result.amounts, cells.map(value => value.amount));
  assert.equal(result.amounts.reduce((a, b) => a + b, 0), 2.5);
});

test("long translated boxes conserve donor amount and permit receiver excess", () => {
  const cells = [cell(0, 1, 1, 2), cell(1, 2, 1, 0), cell(2, 3, 1, 0)];
  const displacements = [[2, 0, 0], [-1, 0, 0], [-1, 0, 0]] as const;
  const forward = referenceWholeFrameVolumeCoupling(cells, displacements);
  const reverse = referenceWholeFrameVolumeCoupling(cells, displacements, true);
  for (const result of [forward, reverse]) {
    assert.ok(Math.abs(result.amounts.reduce((a, b) => a + b, 0) - 2) < 1e-12);
    assert.ok(Math.max(...result.amounts) > 1, "V may exceed open capacity");
    assert.ok(result.amounts.every(value => value >= 0));
    const columns = cells.map((_cell, donor) => result.edges
      .filter(edge => edge.donor === donor).reduce((sum, edge) => sum + edge.weight, 0));
    columns.forEach((sum, donor) => assert.ok(Math.abs(sum - cells[donor]!.donorCapacity) < 1e-12));
  }
});

test("a closing donor without an open receiver is an explicit fault", () => {
  const closing: VolumeBoxCell = {
    minimum: [0, 0, 0], maximum: [1, 1, 1], donorCapacity: 1, receiverCapacity: 0, amount: 1,
  };
  assert.throws(() => referenceWholeFrameVolumeCoupling([closing], [[0, 0, 0]]),
    /has no open receiver/);
});

test("resident shader exposes a fixed whole-frame schedule without FCT microsteps", () => {
  const keys = [
    "currentVolume", "lowVolume", "positiveLimiter", "negativeLimiter",
    "rowSubfaceRanges", "cellSubfaceRanges", "cellSubfaceEntries", "subfaceMetadata",
    "subfaceFluxes", "subfaceRoundoff", "supportControlBaseWords", "controlBaseWords",
    "subfaceCapacity", "airDiagonal", "airControlBaseWords", "airComponentBaseWords",
    "transportEdgeCapacity", "transportEdgeMetadata", "transportEdgeWeightsA",
    "transportEdgeWeightsB", "wholeFrameControlBaseWords",
    "transportReceiverHeadsBaseWords", "transportDonorHeadsBaseWords",
  ] as const;
  const layout = Object.fromEntries(keys.map((key, index) => [key, index + 64])) as unknown as
    SparseGeometricVolumeLayout;
  const shader = createGeometricVolumeResidentWGSL(layout);
  for (const entry of WHOLE_FRAME_VOLUME_ENTRY_POINTS) {
    assert.match(shader, new RegExp(`fn ${entry}\\b`));
  }
  assert.doesNotMatch(shader, /GeometricFCT|gvMicroActive|advanceGeometricVolumeSubstep/);
});

test("V-only sharpening is face-local, conservative, and creates no new excess", () => {
  const before = [1.5, 0.1, 0.4, 0.25];
  const capacity = [1, 1, 1, 0.2];
  const target = [0.5, 0.8, 0.7, 0.1];
  const after = referenceSharpenVolume(before, target, capacity, [[0, 1], [1, 2]]);
  assert.ok(Math.abs(after.reduce((a, b) => a + b, 0) - before.reduce((a, b) => a + b, 0)) < 1e-12);
  after.forEach((value, i) => {
    const priorExcess = Math.max(0, before[i]! - capacity[i]!);
    assert.ok(value >= 0 && value <= capacity[i]! + priorExcess + 1e-12);
  });
  assert.equal(after[3], before[3], "a disconnected cell cannot exchange V");
  assert.ok(Math.abs(after[1]! - target[1]!) < Math.abs(before[1]! - target[1]!));
});

test("sharpening face proof rejects a thin air gap between drops", () => {
  assert.equal(sharpeningFaceHasLiquidConnection(0.01, true, 2), false,
    "positive phi at the shared-face midpoint separates the liquid components");
  assert.equal(sharpeningFaceHasLiquidConnection(-0.01, true, 2), true);
  assert.equal(sharpeningFaceHasLiquidConnection(0, true, 2), true,
    "a resolved contour on the shared face remains connected");
  assert.equal(sharpeningFaceHasLiquidConnection(-0.01, false, 2), false,
    "unsupported phi cannot certify component connectivity");
});
