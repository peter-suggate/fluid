import assert from "node:assert/strict";
import test from "node:test";

import {
  energyFromCompactMoments,
  summarizeLevelSetVolumeEnergy,
  type NativeWorldOutput,
} from "../tools/verify-level-set-volume-energy-native";

test("compact moments include bulk and fluctuating kinetic energy plus origin-referenced PE", () => {
  const energy = energyFromCompactMoments({
    liquidMeasure: 4,
    centroid: [3, 5],
    meanVelocity: [2, -1],
    velocityCovariance: [[4, 99], [99, 9]],
    gravityFineCellsPerSecondSquared: [0, -10],
  });
  assert.deepEqual(energy, {
    kineticEnergyDiagnostic: 36,
    gravitationalPotentialEnergyAboveFloorDiagnostic: 200,
    mechanicalEnergyDiagnostic: 236,
  });
});

function timing(totalAdvance: number, transport: number) {
  return {
    available: true,
    totalAdvance,
    fieldBuild: totalAdvance - transport,
    primaryPressure: 0,
    supportPlanningTransfer: 0,
    postSupportPressure: 0,
    transport,
    postTransport: 0,
    resolutionPublication: 0,
    other: 0,
  };
}

function fixture(advancedMicrosteps = 0): NativeWorldOutput {
  const frame = (number: number, measure: number, converged: boolean) => ({
    liquidCentroid: [0, 2 - 0.5 * number],
    liquidMeanVelocity: [number, 0],
    liquidVelocityCovariance: [[1, 0], [0, 3]],
    receipt: {
      frame: number,
      time: number / 30,
      fault: null,
      liquidMeasure: measure,
      microsteps: number === 0 ? 1 : advancedMicrosteps,
      maxVelocity: number + 2,
      pressure: {
        converged,
        iterations: number === 0 ? 0 : 7,
        initialResidual: number === 0 ? 0 : 10,
        residual: number === 0 ? 0 : 1e-9,
      },
      stageTimings: timing(number === 0 ? 0 : 100 + number, number === 0 ? 0 : 40),
      ...(number === 0 ? {} : { levelSetVolume: {
        overCapacityCellCount: number,
        maximumVolumeOverCapacity: number / 10,
        totalVolumeOverCapacity: number / 5,
        maximumOverCapacityRatio: number / 20,
      } }),
    },
  });
  return { frames: [frame(0, 10, false), frame(1, 10.1, false), frame(2, 9.9, true)], failure: null };
}

test("summary reports unconverged flags without treating them as a failure", () => {
  const report = summarizeLevelSetVolumeEnergy(fixture(), {
    sceneId: "fixture",
    cellSizeMetres: 0.5,
    gravityMetresPerSecondSquared: [0, -5],
  });
  assert.equal(report.configuration.dtSeconds, 1 / 30);
  assert.equal(report.configuration.substeps, 0);
  assert.equal(report.summary.pressureConvergedFlagFrames, 1);
  assert.equal(report.summary.pressureUnconvergedFlagFrames, 1);
  assert.equal(report.summary.maximumPressureRelativeResidual, 1e-10);
  assert.ok(Math.abs(report.summary.finalRelativeMassDrift + 0.01) < 1e-15);
  assert.equal(report.energyDiagnostic.availableEnergyAboveTankRestState, null);
  assert.equal(report.energyDiagnostic.kind, "compact-volume-weighted-moment-approximation");
});

test("summary rejects transport substeps and non-1/30 dt", () => {
  assert.throws(() => summarizeLevelSetVolumeEnergy(fixture(1), {
    sceneId: "fixture",
    cellSizeMetres: 1,
    gravityMetresPerSecondSquared: [0, -10],
  }), /forbids transport substeps/);
  assert.throws(() => summarizeLevelSetVolumeEnergy(fixture(), {
    sceneId: "fixture",
    cellSizeMetres: 1,
    gravityMetresPerSecondSquared: [0, -10],
  }, { dtSeconds: 1 / 60 }), /requires fixed dt=1\/30/);
});

test("summary validates archived receipt time and uses neutral phases for other scenes", () => {
  const output = fixture();
  const report = summarizeLevelSetVolumeEnergy(output, {
    sceneId: "another-scene",
    cellSizeMetres: 1,
    gravityMetresPerSecondSquared: [0, -10],
  });
  assert.equal(report.summary.timing.diagnosticWindow.label, "early-window");
  (output.frames[1]!.receipt as Record<string, unknown>).time = 1 / 60;
  assert.throws(() => summarizeLevelSetVolumeEnergy(output, {
    sceneId: "another-scene",
    cellSizeMetres: 1,
    gravityMetresPerSecondSquared: [0, -10],
  }), /does not match fixed dt=1\/30/);
});
