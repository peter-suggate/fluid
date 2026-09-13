/** No-UI regression for the exact Advance Lab CM12 Figure 7 selection. */
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { findSceneDefinition } from "../lib/core/scenes";
import { sceneDocument } from "../lib/core/scene-definition";

const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
const argument = (name: string, fallback: string): string => {
  const prefix = `--${name}=`;
  return process.argv.find(value => value.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};
const frames = Number(argument("frames", "90"));
assert.ok(Number.isSafeInteger(frames) && frames >= 1, "--frames must be a positive integer");
const diagnostic = argument("diagnostic", "false") === "true";
const binary = resolve(root, argument("binary", "rust/target/release/examples/verify_world"));

const definition = findSceneDefinition("cm12-figure-7");
assert.ok(definition, "CM12 Figure 7 must remain in the live scene catalog");
const document = sceneDocument(definition);
const input = {
  scene: document,
  productionOptions: { dtS: 1 / 30, timeStep: "paper" },
  worldOptions: {
    pressureIterations: 256,
    pressureRelativeTolerance: 1e-6,
    transportExperiment: {
      mode: "cellwise-remap",
      traceSegments: 1,
      edgeSamples: 1,
      closure: "band-projection",
    },
  },
  frames,
  receiptsOnly: true,
  requireCellwiseCommit: true,
  observeStageMetrics: diagnostic,
};

const run = spawnSync(binary, { cwd: root, input: JSON.stringify(input), encoding: "utf8",
  maxBuffer: 64 * 1024 * 1024 });
if (run.status !== 0) {
  process.stderr.write(run.stderr || run.stdout);
  process.exit(run.status ?? 1);
}
const output = JSON.parse(run.stdout) as {
  frames: Array<Record<string, any>>;
  stages: Array<Array<Record<string, any>>>;
};
assert.equal(output.frames.length, frames + 1);

const initial = output.frames[0]!;
assert.equal(initial.receipt.seededVolume, 1252, "live catalog centre slice changed");
assert.deepEqual(initial.liquidBounds, { minimum: [44, 70], maximum: [84, 110] });
const initialSpanX = initial.liquidBounds.maximum[0] - initial.liquidBounds.minimum[0];
let sawImpact = false;
let sawPostImpactSpread = false;
let sawAdaptiveWidths = false;
let sawWetMixedSeam = false;
let sawPressureCleanup = false;
let maximumCleanupBefore = 0;
let maximumCleanupAfter = 0;
let maximumCleanupBound = 0;
let maximumCleanupRateChange = 0;
let maximumEffectiveTraceSegments = 0;
let maximumEffectiveEdgeSamples = 0;
let maximumAdaptiveEdgeRefinementPasses = 0;
let maximumAdaptiveEdgePointsInserted = 0;
let maximumAdaptiveEdgeDyadicDepth = 0;
let maximumAdaptiveEdgePointsPerSubface = 0;
let maximumTraces = 0;
let maximumRkEvaluations = 0;
let maximumChainPoints = 0;
let maximumCourantAfterClosure = 0;
let minimumWidth4WetCells = Number.POSITIVE_INFINITY;
let minimumWidth4LiquidMeasure = Number.POSITIVE_INFINITY;
let previousLiquidMeasure = initial.receipt.liquidMeasure as number;
let maximumStepRelativeLiquidDrift = 0;

const topologyMetric = (frame: Record<string, any>) => {
  const widths = Object.entries(frame.cellsByWidth ?? {}) as Array<[string, Record<string, number>]>;
  const total = widths.reduce((sum, [, value]) => sum + value.liquidMeasure, 0);
  const coarse = widths.filter(([width]) => Number(width) > 1);
  const covariance = frame.liquidCovariance as [[number, number], [number, number]];
  const trace = covariance[0][0] + covariance[1][1];
  const discriminant = Math.hypot(covariance[0][0] - covariance[1][1], 2 * covariance[0][1]);
  const minor = 0.5 * (trace - discriminant);
  const major = 0.5 * (trace + discriminant);
  const resolutionBricks = frame.resolution?.bricks ?? [];
  return {
    frame: frame.receipt.frame,
    centroid: frame.liquidCentroid,
    meanVelocity: frame.liquidMeanVelocity,
    verticalVelocityRmsDeviation: Math.sqrt(Math.max(0, frame.liquidVelocityCovariance[1][1])),
    representedBounds: frame.representedBounds,
    shapeEigenvalueRatio: major / minor,
    coarseLiquidFraction: coarse.reduce((sum, [, value]) => sum + value.liquidMeasure, 0) / total,
    fullCoarseLiquidFraction:
      coarse.reduce((sum, [, value]) => sum + value.fullLiquidMeasure, 0) / total,
    partialCoarseLiquidFraction:
      coarse.reduce((sum, [, value]) => sum + value.partialLiquidMeasure, 0) / total,
    widths: frame.cellsByWidth,
    remapWork: frame.receipt.cellwiseRemap ? {
      traces: frame.receipt.cellwiseRemap.traces,
      rkEvaluations: frame.receipt.cellwiseRemap.rkEvaluations,
      chainPoints: frame.receipt.cellwiseRemap.chainPoints,
      adaptiveEdgePointsInserted: frame.receipt.cellwiseRemap.adaptiveEdgePointsInserted,
      adaptiveEdgeRefinementPasses: frame.receipt.cellwiseRemap.adaptiveEdgeRefinementPasses,
      effectiveTraceSegments: frame.receipt.cellwiseRemap.traceSegments,
      effectiveEdgeSamples: frame.receipt.cellwiseRemap.edgeSamples,
    } : null,
    resolutionReasons: {
      surface: resolutionBricks.filter((brick: any) => (brick.reasons & 1) !== 0).length,
      velocityFloor: resolutionBricks.filter((brick: any) => (brick.reasons & 128) !== 0).length,
      densitySurface: resolutionBricks.filter((brick: any) => (brick.reasons & 16_384) !== 0).length,
      bulkDemotion: resolutionBricks.filter((brick: any) => brick.planReasons === 2_048).length,
      coarsePromotion: resolutionBricks.filter((brick: any) => brick.planReasons === 4).length,
    },
  };
};

const initialTopology = topologyMetric(initial);
const initialWidth4 = initial.cellsByWidth?.["4"];
assert.ok(initialWidth4?.wetCells > 0 && initialWidth4?.liquidMeasure > 0,
  "initial Figure 7 sphere has no width-4 liquid cells");
minimumWidth4WetCells = initialWidth4.wetCells;
minimumWidth4LiquidMeasure = initialWidth4.liquidMeasure;
const dt = input.productionOptions.dtS;
const gravityFineY = document.fluid.gravity_m_s2.y / document.voxelDomain.finestCellSize_m;
const freefallEnvelope = (frame: Record<string, any>) => {
  const step = frame.receipt.frame;
  const displacement = gravityFineY * dt * dt * step * (step + 1) / 2;
  const lower = initial.liquidBounds.minimum[1] + displacement - 1;
  const upper = initial.liquidBounds.maximum[1] + displacement + 1;
  const bins = frame.liquidMeasureByFineY as number[];
  const total = bins.reduce((sum, value) => sum + value, 0);
  const below = bins.reduce((sum, value, y) => sum + (y + 0.5 < lower ? value : 0), 0);
  const above = bins.reduce((sum, value, y) => sum + (y + 0.5 > upper ? value : 0), 0);
  return {
    expectedFreefallCentroidY: initial.liquidCentroid[1] + displacement,
    expectedFreefallVelocityY: gravityFineY * dt * step,
    freefallEnvelopeLeakFraction: (below + above) / total,
    laggingUpperLiquidFraction: above / total,
  };
};

for (const frame of output.frames.slice(1)) {
  const receipt = frame.receipt;
  const remap = receipt.cellwiseRemap;
  assert.equal(receipt.fault, null, `frame ${receipt.frame}: numerical fault`);
  assert.equal(receipt.microsteps, 0, `frame ${receipt.frame}: baseline transport ran`);
  assert.equal(remap.materialCommitted, true, `frame ${receipt.frame}: material commit rejected`);
  assert.equal(remap.closureAccepted, true, `frame ${receipt.frame}: closure rejected`);
  assert.equal(remap.pregeometryCertificateViolations, 0,
    `frame ${receipt.frame}: continuity certificate rejected`);
  assert.ok(remap.traces > 0 && remap.gatherClips > 0,
    `frame ${receipt.frame}: material remap did not commit`);
  assert.equal(remap.supportExtrapolationSamples, 0,
    `frame ${receipt.frame}: receiver path left the continuous streamfunction support`);
  assert.equal(remap.traceVelocityFallbackSamples, 0,
    `frame ${receipt.frame}: receiver path used the local RT0 fallback`);
  for (const field of [
    "preCorrectionReceiverBandFolds",
    "correctedReceiverBandFolds",
    "preCorrectionBboxLiquidFolds",
    "correctedBboxLiquidFolds",
    "preCorrectionConvexHullLiquidFolds",
    "correctedConvexHullLiquidFolds",
    "preCorrectionLiquidReceiverFolds",
    "correctedLiquidReceiverFolds",
  ]) {
    assert.equal(typeof remap[field], "number",
      `frame ${receipt.frame}: missing ${field} receipt`);
  }
  assert.equal(remap.preCorrectionConvexHullLiquidFolds, 0,
    `frame ${receipt.frame}: uncorrected folded hull can intersect liquid`);
  assert.equal(remap.correctedConvexHullLiquidFolds, 0,
    `frame ${receipt.frame}: corrected folded hull can intersect liquid`);
  assert.equal(remap.preCorrectionLiquidReceiverFolds, 0,
    `frame ${receipt.frame}: uncorrected material receiver folded`);
  assert.equal(remap.correctedLiquidReceiverFolds, 0,
    `frame ${receipt.frame}: corrected material receiver folded`);
  assert.equal(remap.adaptiveEdgeRefinementExhausted, false,
    `frame ${receipt.frame}: receiver-band geometry refinement exhausted its bound`);
  assert.ok(remap.gatherAbsoluteVolumeError <= remap.gatherVolumeRoundoffBound,
    `frame ${receipt.frame}: gather volume error ${remap.gatherAbsoluteVolumeError} exceeded ${remap.gatherVolumeRoundoffBound}`);
  assert.ok(remap.gatherWorstDonorVolumeError <= remap.gatherWorstDonorRoundoffBound,
    `frame ${receipt.frame}: donor ${remap.gatherWorstDonor} coverage error ${remap.gatherWorstDonorVolumeError} exceeded ${remap.gatherWorstDonorRoundoffBound}`);
  const stepRelativeLiquidDrift = Math.abs(receipt.liquidMeasure - previousLiquidMeasure)
    / initial.receipt.liquidMeasure;
  maximumStepRelativeLiquidDrift = Math.max(maximumStepRelativeLiquidDrift,
    stepRelativeLiquidDrift);
  assert.ok(stepRelativeLiquidDrift <= 1e-7,
    `frame ${receipt.frame}: remap changed liquid by ${stepRelativeLiquidDrift} relative`);
  previousLiquidMeasure = receipt.liquidMeasure;
  assert.ok(Math.abs(receipt.drift) <= 1e-7,
    `frame ${receipt.frame}: relative liquid drift ${receipt.drift}`);
  assert.ok(frame.densityRange[0] >= -1e-6 && frame.densityRange[1] <= 1.000001,
    `frame ${receipt.frame}: density out of bounds ${frame.densityRange}`);
  const width4 = frame.cellsByWidth?.["4"];
  assert.ok(width4?.wetCells > 0 && width4?.liquidMeasure > 0,
    `frame ${receipt.frame}: geometric scene lost all width-4 liquid cells`);
  minimumWidth4WetCells = Math.min(minimumWidth4WetCells, width4.wetCells);
  minimumWidth4LiquidMeasure = Math.min(minimumWidth4LiquidMeasure, width4.liquidMeasure);

  const before = remap.pressureRateCleanupMaxNormalizedDivergenceBefore;
  const after = remap.pressureRateCleanupMaxNormalizedDivergenceAfter;
  const bound = remap.pressureRateCleanupMaxNormalizedRoundoffBound;
  assert.ok(before <= bound + Number.EPSILON,
    `frame ${receipt.frame}: pressure cleanup exceeded its derived roundoff bound`);
  sawPressureCleanup ||= remap.pressureRateCleanupChangedSubfaces > 0;
  maximumCleanupBefore = Math.max(maximumCleanupBefore, before);
  maximumCleanupAfter = Math.max(maximumCleanupAfter, after);
  maximumCleanupBound = Math.max(maximumCleanupBound, bound);
  maximumCleanupRateChange = Math.max(maximumCleanupRateChange,
    remap.pressureRateCleanupMaxAbsoluteRateChange);
  assert.equal(remap.requestedTraceSegments, 1,
    `frame ${receipt.frame}: live scene trace-segment request changed`);
  maximumEffectiveTraceSegments = Math.max(maximumEffectiveTraceSegments, remap.traceSegments);
  assert.equal(remap.requestedEdgeSamples, 1,
    `frame ${receipt.frame}: live scene edge-sample request changed`);
  assert.ok(remap.edgeSamples >= remap.requestedEdgeSamples,
    `frame ${receipt.frame}: effective edge sampling regressed below its request`);
  maximumEffectiveEdgeSamples = Math.max(maximumEffectiveEdgeSamples, remap.edgeSamples);
  maximumAdaptiveEdgeRefinementPasses = Math.max(maximumAdaptiveEdgeRefinementPasses,
    remap.adaptiveEdgeRefinementPasses);
  maximumAdaptiveEdgePointsInserted = Math.max(maximumAdaptiveEdgePointsInserted,
    remap.adaptiveEdgePointsInserted);
  maximumAdaptiveEdgeDyadicDepth = Math.max(maximumAdaptiveEdgeDyadicDepth,
    remap.adaptiveEdgeMaxDyadicDepth);
  maximumAdaptiveEdgePointsPerSubface = Math.max(maximumAdaptiveEdgePointsPerSubface,
    remap.adaptiveEdgeMaxPointsPerSubface);
  maximumTraces = Math.max(maximumTraces, remap.traces);
  maximumRkEvaluations = Math.max(maximumRkEvaluations, remap.rkEvaluations);
  maximumChainPoints = Math.max(maximumChainPoints, remap.chainPoints);
  maximumCourantAfterClosure = Math.max(maximumCourantAfterClosure,
    remap.maxCourantAfterClosure);

  if (receipt.frame <= 20) {
    const topology = topologyMetric(frame);
    const expectedCentroidY = initial.liquidCentroid[1]
      + gravityFineY * dt * dt * receipt.frame * (receipt.frame + 1) / 2;
    if (!diagnostic) {
      assert.ok(Math.abs(topology.centroid[0] - initialTopology.centroid[0]) <= 0.25,
        `frame ${receipt.frame}: falling sphere drifted laterally`);
      assert.ok(Math.abs(topology.centroid[1] - expectedCentroidY) <= 0.25,
        `frame ${receipt.frame}: centroid ${topology.centroid[1]} != freefall ${expectedCentroidY}`);
      assert.ok(topology.shapeEigenvalueRatio <= 1.05,
        `frame ${receipt.frame}: pre-impact sphere aspect ${topology.shapeEigenvalueRatio}`);
    }
    if (!diagnostic) {
      assert.ok(topology.coarseLiquidFraction >= 0.4,
        `frame ${receipt.frame}: coarse liquid fraction ${topology.coarseLiquidFraction}`);
      assert.ok(topology.fullCoarseLiquidFraction >= 0.4,
        `frame ${receipt.frame}: full coarse liquid fraction ${topology.fullCoarseLiquidFraction}`);
    }
  }

  const spanX = frame.liquidBounds.maximum[0] - frame.liquidBounds.minimum[0];
  sawImpact ||= frame.liquidBounds.minimum[1] <= 4;
  sawPostImpactSpread ||= sawImpact && spanX > initialSpanX;
  sawAdaptiveWidths ||= frame.cellWidths.length > 1;
  sawWetMixedSeam ||= frame.wetMixedSeamCount > 0;
}

const last = output.frames.at(-1)!;
const topologyFrames = new Set(
  [0, 1, 5, 10, 11, 12, 13, 14, 15, 20, 25, 26, 36, 40]
    .filter(frame => frame <= frames),
);
const preImpactTopology = output.frames
  .filter(frame => topologyFrames.has(frame.receipt.frame))
  .map(frame => ({
    ...topologyMetric(frame),
    ...freefallEnvelope(frame),
    stageVelocity: frame.receipt.frame === 0 ? [] : output.stages[frame.receipt.frame - 1],
  }));
if (frames >= 90) {
  assert.ok(sawImpact, `${frames} frames did not reach the tank floor`);
  assert.ok(sawPostImpactSpread, `${frames} frames did not cover post-impact spreading`);
}
if (frames >= 20 && !diagnostic) {
  const frame20 = topologyMetric(output.frames[20]!);
  assert.ok(frame20.coarseLiquidFraction >= 0.4,
    `frame 20: coarse liquid fraction ${frame20.coarseLiquidFraction}`);
  assert.ok(frame20.fullCoarseLiquidFraction >= 0.4,
    `frame 20: full coarse liquid fraction ${frame20.fullCoarseLiquidFraction}`);
}
assert.ok(sawAdaptiveWidths, "run did not exercise adaptive cell widths");
assert.ok(sawWetMixedSeam, "run did not transport liquid across a mixed seam");
assert.ok(sawPressureCleanup, "run did not exercise the bounded pressure-rate cleanup");
assert.ok(last.receipt.topologyGeneration > initial.receipt.topologyGeneration,
  "adaptive topology generation did not advance");

process.stdout.write(`${JSON.stringify({
  scene: definition.id,
  frames,
  initialLiquidMeasure: initial.receipt.liquidMeasure,
  finalLiquidMeasure: last.receipt.liquidMeasure,
  finalRelativeDrift: last.receipt.drift,
  finalTopologyGeneration: last.receipt.topologyGeneration,
  finalLiquidBounds: last.liquidBounds,
  maximumEffectiveTraceSegments,
  maximumEffectiveEdgeSamples,
  maximumAdaptiveEdgeRefinementPasses,
  maximumAdaptiveEdgePointsInserted,
  maximumAdaptiveEdgeDyadicDepth,
  maximumAdaptiveEdgePointsPerSubface,
  maximumTraces,
  maximumRkEvaluations,
  maximumChainPoints,
  maximumCourantAfterClosure,
  minimumWidth4WetCells,
  minimumWidth4LiquidMeasure,
  maximumStepRelativeLiquidDrift,
  pressureRateCleanup: {
    maximumNormalizedDivergenceBefore: maximumCleanupBefore,
    maximumNormalizedDivergenceAfter: maximumCleanupAfter,
    maximumNormalizedRoundoffBound: maximumCleanupBound,
    maximumAbsoluteRateChange: maximumCleanupRateChange,
  },
  sawImpact,
  sawPostImpactSpread,
  sawAdaptiveWidths,
  sawWetMixedSeam,
  preImpactTopology,
}, null, 2)}\n`);
