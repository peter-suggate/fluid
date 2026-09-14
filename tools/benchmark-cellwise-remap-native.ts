/** Native, no-UI timing arm for sparse geometric-remap work-shape changes. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import { findSceneDefinition } from "../lib/core/scenes";
import { sceneDocument } from "../lib/core/scene-definition";
import {
  type CellwiseStageTimings,
  requireNativeCellwiseStageTimings,
  requireNativeWorldStageTimings,
  summarizeNativeStageTimings,
  type WorldStageTimings,
} from "./native-cellwise-stage-timings";

const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
const argument = (name: string, fallback: string): string => {
  const prefix = `--${name}=`;
  return process.argv.find(value => value.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};
const sceneId = argument("scene", "cm12-figure-7");
assert.ok(["cm12-figure-7", "coarse-first-pool-impact-half"].includes(sceneId),
  "--scene must name one of the two exact native regression scenes");
const mode = argument("mode", "cellwise-remap");
assert.ok(mode === "baseline" || mode === "cellwise-remap",
  "--mode must be baseline or cellwise-remap");
const defaultFrames = sceneId === "cm12-figure-7" ? "30" : "10";
const frames = Number(argument("frames", defaultFrames));
assert.ok(Number.isSafeInteger(frames) && frames > 0, "--frames must be positive");
const arm = argument("arm", mode);
const rawOutputPath = argument("raw-output", "");
const inputOutputPath = argument("input-output", "");
const binary = resolve(root, argument("binary", "rust/target/release/examples/verify_world"));
const sha256 = (path: string) => createHash("sha256").update(readFileSync(path)).digest("hex");
const sourcePaths = [
  "rust/crates/fluid-core/src/world.rs",
  "rust/crates/fluid-core/src/band_projection.rs",
  "rust/crates/fluid-core/src/numerics.rs",
  "rust/crates/fluid-core/src/adaptive_remap.rs",
  "rust/crates/fluid-core/src/resolution.rs",
  "rust/crates/fluid-core/examples/verify_world.rs",
] as const;
const git = (args: string[]) => spawnSync("git", args, { cwd: root, encoding: "utf8" }).stdout.trim();
// Snapshot source provenance before launching the native binary. Another
// worker may begin the next arm's edits while this process is still running.
const sourceProvenance = {
  gitCommit: git(["rev-parse", "HEAD"]),
  gitStatus: git(["status", "--short"]),
  binary,
  binarySha256: sha256(binary),
  sourceHashesCapturedBeforeRun: true,
  sourceSha256: Object.fromEntries(sourcePaths.map(path => [path, sha256(resolve(root, path))])),
};
const definition = findSceneDefinition(sceneId);
assert.ok(definition, `live scene ${sceneId} is missing`);

const transportExperiment = mode === "baseline" ? "baseline" : {
  mode: "cellwise-remap",
  traceSegments: 1,
  edgeSamples: 1,
  closure: "band-projection",
};
const input = {
  scene: sceneDocument(definition),
  productionOptions: { dtS: 1 / 30, timeStep: "paper" },
  worldOptions: {
    pressureIterations: 256,
    pressureRelativeTolerance: 1e-6,
    transportExperiment,
  },
  frames,
  receiptsOnly: true,
  requireCellwiseCommit: mode === "cellwise-remap",
  captureFailure: true,
};
if (inputOutputPath) writeFileSync(resolve(root, inputOutputPath), `${JSON.stringify(input)}\n`);

const started = process.hrtime.bigint();
const run = spawnSync(binary, {
  cwd: root,
  input: JSON.stringify(input),
  encoding: "utf8",
  maxBuffer: 64 * 1024 * 1024,
});
const wrapperNanoseconds = Number(process.hrtime.bigint() - started);
if (run.status !== 0) {
  process.stderr.write(run.stderr || run.stdout);
  process.exit(run.status ?? 1);
}
if (rawOutputPath) writeFileSync(resolve(root, rawOutputPath), run.stdout);
const output = JSON.parse(run.stdout) as {
  frames: Array<Record<string, any>>;
  failure: unknown;
};
assert.ok(output.frames.length >= 1 && output.frames.length <= frames + 1);

const worldTimings: WorldStageTimings[] = [];
const cellwiseTimings: CellwiseStageTimings[] = [];
let totalMicrosteps = 0;
let maximumMicrosteps = 0;
let maximumCells = 0;
let maximumTraces = 0;
let maximumRkEvaluations = 0;
let maximumChainPoints = 0;
let maximumEffectiveTraceSegments = 0;
let maximumEffectiveEdgeSamples = 0;
const frameRows: Array<Record<string, unknown>> = [];
const gateErrors: string[] = [];
for (const frame of output.frames.slice(1)) {
  const receipt = frame.receipt;
  const cellCount = frame.cellCount ?? frame.graph?.cells?.length;
  assert.ok(Number.isSafeInteger(cellCount) && cellCount >= 0,
    `frame ${receipt.frame}: missing cell count`);
  try {
    worldTimings.push(requireNativeWorldStageTimings(receipt, receipt.frame));
  } catch (error) {
    gateErrors.push(error instanceof Error ? error.message : String(error));
    break;
  }
  totalMicrosteps += receipt.microsteps;
  maximumMicrosteps = Math.max(maximumMicrosteps, receipt.microsteps);
  maximumCells = Math.max(maximumCells, cellCount);
  if (mode === "baseline") {
    try {
      assert.equal(receipt.cellwiseRemap, undefined,
        `frame ${receipt.frame}: baseline unexpectedly published a remap receipt`);
    } catch (error) {
      gateErrors.push(error instanceof Error ? error.message : String(error));
    }
    frameRows.push({
      frame: receipt.frame,
      cells: cellCount,
      microsteps: receipt.microsteps,
      drift: receipt.drift,
      worldStageTimings: worldTimings.at(-1),
      cellwiseStageTimings: null,
    });
    if (receipt.fault !== null) {
      gateErrors.push(`frame ${receipt.frame}: numerical fault ${JSON.stringify(receipt.fault)}`);
      break;
    }
    continue;
  }
  const remap = receipt.cellwiseRemap;
  try {
    cellwiseTimings.push(requireNativeCellwiseStageTimings(remap, receipt.frame));
  } catch (error) {
    gateErrors.push(error instanceof Error ? error.message : String(error));
    break;
  }
  maximumTraces = Math.max(maximumTraces, remap.traces);
  maximumRkEvaluations = Math.max(maximumRkEvaluations, remap.rkEvaluations);
  maximumChainPoints = Math.max(maximumChainPoints, remap.chainPoints);
  maximumEffectiveTraceSegments = Math.max(maximumEffectiveTraceSegments, remap.traceSegments);
  maximumEffectiveEdgeSamples = Math.max(maximumEffectiveEdgeSamples, remap.edgeSamples);
  frameRows.push({
    frame: receipt.frame,
    cells: cellCount,
    microsteps: receipt.microsteps,
    drift: receipt.drift,
    traces: remap.traces,
    rkEvaluations: remap.rkEvaluations,
    chainPoints: remap.chainPoints,
    effectiveTraceSegments: remap.traceSegments,
    effectiveEdgeSamples: remap.edgeSamples,
    supportExtrapolationSamples: remap.supportExtrapolationSamples,
    traceVelocityFallbackSamples: remap.traceVelocityFallbackSamples,
    preCorrectionReceiverBandFolds: remap.preCorrectionReceiverBandFolds,
    correctedReceiverBandFolds: remap.correctedReceiverBandFolds,
    preCorrectionMaterialFolds: remap.preCorrectionLiquidReceiverFolds,
    correctedMaterialFolds: remap.correctedLiquidReceiverFolds,
    gatherAbsoluteVolumeError: remap.gatherAbsoluteVolumeError,
    gatherVolumeRoundoffBound: remap.gatherVolumeRoundoffBound,
    gatherWorstDonorVolumeError: remap.gatherWorstDonorVolumeError,
    gatherWorstDonorRoundoffBound: remap.gatherWorstDonorRoundoffBound,
    worldStageTimings: worldTimings.at(-1),
    cellwiseStageTimings: cellwiseTimings.at(-1),
  });
  try {
    assert.equal(receipt.fault, null, `frame ${receipt.frame}: numerical fault`);
    assert.equal(receipt.microsteps, 0, `frame ${receipt.frame}: baseline transport ran`);
    assert.equal(remap.materialCommitted, true, `frame ${receipt.frame}: material did not commit`);
    assert.equal(remap.closureAccepted, true, `frame ${receipt.frame}: closure rejected`);
    assert.equal(remap.supportExtrapolationSamples, 0,
      `frame ${receipt.frame}: trace left streamfunction support`);
    assert.equal(remap.traceVelocityFallbackSamples, 0,
      `frame ${receipt.frame}: trace used a fallback velocity`);
    assert.equal(remap.preCorrectionLiquidReceiverFolds, 0);
    assert.equal(remap.correctedLiquidReceiverFolds, 0);
    assert.equal(remap.preCorrectionConvexHullLiquidFolds, 0);
    assert.equal(remap.correctedConvexHullLiquidFolds, 0);
    assert.equal(remap.adaptiveEdgeRefinementExhausted, false);
    assert.ok(remap.gatherAbsoluteVolumeError <= remap.gatherVolumeRoundoffBound);
    assert.ok(remap.gatherWorstDonorVolumeError <= remap.gatherWorstDonorRoundoffBound);
  } catch (error) {
    gateErrors.push(error instanceof Error ? error.message : String(error));
    break;
  }
}

const timingWindow = (first: number, last: number) => {
  const start = Math.max(1, first) - 1;
  const end = Math.min(worldTimings.length, last);
  if (start >= end) return null;
  return summarizeNativeStageTimings(
    worldTimings.slice(start, end),
    mode === "cellwise-remap" ? cellwiseTimings.slice(start, end) : undefined,
  );
};
const last = output.frames.at(-1)!;
const accepted = output.failure === null
  && gateErrors.length === 0
  && output.frames.length === frames + 1;
const selectedFrames = new Set([1, 5, 10, 20, 25, 28, 30].filter(value => value <= frames));
process.stdout.write(`${JSON.stringify({
  arm,
  scene: sceneId,
  mode,
  frames,
  configuration: {
    dtSeconds: 1 / 30,
    pressureIterations: 256,
    pressureRelativeTolerance: 1e-6,
    traceSegments: mode === "cellwise-remap" ? 1 : null,
    edgeSamples: mode === "cellwise-remap" ? 1 : null,
    closure: mode === "cellwise-remap" ? "band-projection" : null,
  },
  provenance: sourceProvenance,
  wrapper: {
    totalNanoseconds: wrapperNanoseconds,
    includesProcessStartupAndJson: true,
  },
  acceptance: {
    accepted,
    requestedFrames: frames,
    producedFrames: output.frames.length - 1,
    firstFailure: output.failure,
    gateErrors,
  },
  timings: {
    all: worldTimings.length > 0 ? summarizeNativeStageTimings(
      worldTimings,
      mode === "cellwise-remap" ? cellwiseTimings : undefined,
    ) : null,
    preImpact: sceneId === "cm12-figure-7" ? timingWindow(1, 19) : null,
    frames20Through30: sceneId === "cm12-figure-7" ? timingWindow(20, 30) : null,
  },
  work: {
    totalMicrosteps,
    maximumMicrosteps,
    maximumCells,
    maximumTraces,
    maximumRkEvaluations,
    maximumChainPoints,
    maximumEffectiveTraceSegments,
    maximumEffectiveEdgeSamples,
  },
  selectedFrameRows: frameRows.filter(row => selectedFrames.has(row.frame as number)),
  failureFrameRow: output.failure === null ? null : frameRows.at(-1),
  result: {
    initialLiquidMeasure: output.frames[0]!.receipt.liquidMeasure,
    finalLiquidMeasure: last.receipt.liquidMeasure,
    finalRelativeDrift: last.receipt.drift,
    finalTopologyGeneration: last.receipt.topologyGeneration,
  },
}, null, 2)}\n`);
if (!accepted) process.exitCode = 1;
