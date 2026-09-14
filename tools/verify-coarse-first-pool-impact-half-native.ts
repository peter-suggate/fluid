/** No-UI regression for the live coarse-first half-pool impact scene. */
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
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
import {
  type ResearchTransport,
  summarizeResearchTransport,
} from "./native-level-set-volume-report";

const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
const argument = (name: string, fallback: string): string => {
  const prefix = `--${name}=`;
  return process.argv.find(value => value.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};
const frames = Number(argument("frames", "10"));
assert.ok(Number.isSafeInteger(frames) && frames >= 1, "--frames must be positive");
const transport = argument("transport", "cellwise-remap");
assert.ok(["cellwise-remap", "level-set-volume", "baseline"].includes(transport),
  "--transport must be cellwise-remap, level-set-volume, or baseline");
const outputPath = argument("output", "");
const binary = resolve(root, argument("binary", "rust/target/release/examples/verify_world"));
const definition = findSceneDefinition("coarse-first-pool-impact-half");
assert.ok(definition, "live coarse-first half-pool scene is missing");

const input = {
  scene: sceneDocument(definition),
  productionOptions: { dtS: 1 / 30, timeStep: "paper" },
  worldOptions: {
    pressureIterations: 256,
    pressureRelativeTolerance: 1e-6,
    transportExperiment: transport === "cellwise-remap" ? {
      mode: "cellwise-remap",
      traceSegments: 1,
      edgeSamples: 1,
      closure: "band-projection",
    } : transport,
  },
  frames,
  receiptsOnly: true,
  requireCellwiseCommit: transport === "cellwise-remap",
};

const run = spawnSync(binary, {
  cwd: root,
  input: JSON.stringify(input),
  encoding: "utf8",
  maxBuffer: 64 * 1024 * 1024,
});
if (run.status !== 0) {
  process.stderr.write(run.stderr || run.stdout);
  process.exit(run.status ?? 1);
}
const output = JSON.parse(run.stdout) as { frames: Array<Record<string, any>>; failure: unknown };
assert.equal(output.failure, null);
assert.equal(output.frames.length, frames + 1);
assert.equal(output.frames[0]?.receipt.seededVolume, 1337, "live scene seed changed");
if (transport !== "cellwise-remap") {
  const report = `${JSON.stringify({
    scene: definition.id,
    ...summarizeResearchTransport(output, transport as ResearchTransport),
  }, null, 2)}\n`;
  if (outputPath) writeFileSync(resolve(root, outputPath), report);
  process.stdout.write(report);
  process.exit(0);
}
const stageTimingFrames: CellwiseStageTimings[] = [];
const worldStageTimingFrames: WorldStageTimings[] = [];
for (const frame of output.frames.slice(1)) {
  const receipt = frame.receipt;
  const remap = receipt.cellwiseRemap;
  worldStageTimingFrames.push(requireNativeWorldStageTimings(receipt, receipt.frame));
  const remapTiming = requireNativeCellwiseStageTimings(remap, receipt.frame);
  stageTimingFrames.push(remapTiming);
  assert.equal(receipt.fault, null, `frame ${receipt.frame}: numerical fault`);
  assert.equal(receipt.microsteps, 0, `frame ${receipt.frame}: baseline transport ran`);
  assert.equal(remap.materialCommitted, true, `frame ${receipt.frame}: material commit rejected`);
  assert.equal(remap.closureAccepted, true, `frame ${receipt.frame}: closure rejected`);
  assert.equal(remap.pregeometryCertificateViolations, 0,
    `frame ${receipt.frame}: continuity certificate rejected`);
  assert.equal(remap.requestedTraceSegments, 1);
  assert.equal(remap.requestedEdgeSamples, 1);
  assert.equal(remap.supportExtrapolationSamples, 0);
  assert.equal(remap.traceVelocityFallbackSamples, 0);
  assert.equal(remap.preCorrectionConvexHullLiquidFolds, 0);
  assert.equal(remap.correctedConvexHullLiquidFolds, 0);
  assert.equal(remap.preCorrectionLiquidReceiverFolds, 0);
  assert.equal(remap.correctedLiquidReceiverFolds, 0);
  assert.equal(remap.adaptiveEdgeRefinementExhausted, false);
  assert.ok(remap.gatherAbsoluteVolumeError <= remap.gatherVolumeRoundoffBound);
  assert.ok(remap.gatherWorstDonorVolumeError <= remap.gatherWorstDonorRoundoffBound);
  assert.ok(Math.abs(receipt.drift) <= 1e-7,
    `frame ${receipt.frame}: liquid drift ${receipt.drift}`);
}

const last = output.frames.at(-1)!;
process.stdout.write(`${JSON.stringify({
  scene: definition.id,
  frames,
  initialLiquidMeasure: output.frames[0]!.receipt.liquidMeasure,
  finalLiquidMeasure: last.receipt.liquidMeasure,
  finalRelativeDrift: last.receipt.drift,
  finalTopologyGeneration: last.receipt.topologyGeneration,
  stageTimings: summarizeNativeStageTimings(worldStageTimingFrames, stageTimingFrames),
}, null, 2)}\n`);
