import assert from "node:assert/strict";
import test from "node:test";

import {
  requireLevelSetVolumeReceipt,
  summarizeResearchTransport,
} from "../tools/native-level-set-volume-report";

const receipt = () => ({
  initialLiquidVolume: 10,
  finalLiquidVolume: 10,
  signedVolumeDrift: 0,
  absoluteVolumeDrift: 0,
  volumeRoundoffBound: 1e-5,
  maximumNormalizedRowResidual: 1e-4,
  maximumDonorResidual: 1e-6,
  zeroWeightDonors: 0,
  invalidPhiSamples: 0,
  maximumTraceDistance: 4,
  maximumTraceCourant: 2,
  overCapacityCellCount: 2,
  maximumVolumeOverCapacity: 0.1,
  totalVolumeOverCapacity: 0.15,
  maximumOverCapacityRatio: 0.02,
  phiImpliedLiquidVolume: 10,
  signedPhiVolumeMismatch: 0,
  absolutePhiVolumeMismatch: 0,
  insideBandAbsolutePhiVolumeMismatch: 0,
  outsideBandAbsolutePhiVolumeMismatch: 0,
  maximumAbsolutePhiVolumeMismatch: 0,
  maximumNormalizedPhiVolumeMismatch: 0,
  sharpening: {
    componentCount: 1,
    ambiguousCellCount: 0,
    orphanVolume: 0,
    initialBandAbsoluteMismatch: 0,
    finalBandAbsoluteMismatch: 0,
    initialDistanceWeightedMismatch: 0,
    finalDistanceWeightedMismatch: 0,
    initialOverCapacityVolume: 0,
    finalOverCapacityVolume: 0,
    initialOverCapacityCount: 0,
    finalOverCapacityCount: 0,
    relocatedVolume: 0,
    donorCount: 0,
    receiverCount: 0,
    unresolvedEligibleResidual: 0,
    maximumRelocationDistance: 0,
    crossComponentPairCount: 0,
    boundViolationCount: 0,
    globalConservationResidual: 0,
    maximumComponentConservationResidual: 0,
  },
  traceNanoseconds: 1,
  volumeGatherNanoseconds: 2,
  phiGatherNanoseconds: 3,
  planeFitNanoseconds: 4,
  rdfNanoseconds: 5,
});

test("level-set-volume receipt requires all lab measurements", () => {
  assert.equal(requireLevelSetVolumeReceipt(receipt(), 1).planeFitNanoseconds, 4);
  const missing = receipt() as Record<string, unknown>;
  delete missing.rdfNanoseconds;
  assert.throws(() => requireLevelSetVolumeReceipt(missing, 1), /rdfNanoseconds/);
});

test("level-set-volume report preserves failed conservation evidence", () => {
  assert.equal(requireLevelSetVolumeReceipt({
    ...receipt(), absoluteVolumeDrift: 2e-5,
  }, 3).absoluteVolumeDrift, 2e-5);
});

const worldTimings = {
  available: true,
  totalAdvance: 28,
  fieldBuild: 1,
  primaryPressure: 2,
  supportPlanningTransfer: 3,
  postSupportPressure: 4,
  transport: 5,
  postTransport: 6,
  resolutionPublication: 7,
  other: 0,
};
const frame = (number: number, transport: "baseline" | "level-set-volume") => ({
  receipt: {
    frame: number,
    liquidMeasure: 10,
    fault: null,
    microsteps: transport === "baseline" ? 2 : 0,
    pressure: { iterations: 9, encodedIterations: 256 },
    stageTimings: worldTimings,
    interfaceSeams: {
      comparisonCount: 3,
      skippedInvalidPlaneCount: 1,
      meanAbsoluteOffsetDifference: 0.2,
      rmsOffsetDifference: 0.3,
      maximumAbsoluteOffsetDifference: 0.4,
    },
    ...(transport === "level-set-volume" ? { levelSetVolume: receipt() } : {}),
  },
});

test("research summary keeps baseline microsteps and level-set substeps distinct", () => {
  const baseline = summarizeResearchTransport({
    frames: [frame(0, "baseline"), frame(1, "baseline")], failure: null,
  }, "baseline");
  assert.equal(baseline.configuration.levelSetVolumeSubsteps, null);
  assert.equal(baseline.materialMicrosteps.total, 2);
  const levelSet = summarizeResearchTransport({
    frames: [frame(0, "level-set-volume"), frame(1, "level-set-volume")], failure: null,
  }, "level-set-volume");
  assert.equal(levelSet.configuration.levelSetVolumeSubsteps, 0);
  assert.equal(levelSet.overCapacity.maximumNormalizedRowResidual, 1e-4);
});
