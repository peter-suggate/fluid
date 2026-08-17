#!/usr/bin/env node
/** CPU-only source/receipt gate for all six production coherence stages. */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  SPARSE_CM12_VEX_ACTIVITY_BATCH_GRID_RECEIPTS,
  SPARSE_CM12_VEX_ACTIVITY_BATCH_GRID_REQUIRED_FIELDS,
} from "../lib/methods/adaptive-mass/sparse-cm12-vex-activity-batch";
import {
  SPARSE_CM12_SIX_STAGE_AUTHORITY_EXPECTATIONS,
  SPARSE_CM12_SIX_STAGE_REQUIRED_RECEIPT_FIELDS,
  assertSparseCM12GridOverlayFailClosedSource,
  assertSparseCM12SixStageFramePlanLayout,
  assertSparseCM12SixStageProductionSources,
  inspectSparseCM12SixStageReceipts,
  type SparseCM12LogicalStageReceipt,
  type SparseCM12SixStageAuthorityId,
  type SparseCM12SixStageReceipt,
} from "../lib/methods/adaptive-mass/sparse-cm12-six-stage-cutover-gate";

assert.equal(SPARSE_CM12_VEX_ACTIVITY_BATCH_GRID_RECEIPTS.length, 6);
for (let stage = 0; stage < 6; stage += 1) {
  const receipt = SPARSE_CM12_VEX_ACTIVITY_BATCH_GRID_RECEIPTS.find(
    (entry) => entry.stage === stage,
  );
  assert.ok(receipt, `Grid receipt stage ${stage} is absent`);
  assert.equal(receipt.absent, "unknown-magenta");
  assert.equal(receipt.direct, "required");
  assert.equal(receipt.closure, "required");
  assert.equal(receipt.disposition, "executed-or-skipped");
}
for (const field of ["directMaskLow", "closureMaskLow", "executedMaskLow",
  "skippedMaskLow", "unknownCount", "uncoveredWriteCount", "faultCount"] as const) {
  assert.ok((SPARSE_CM12_VEX_ACTIVITY_BATCH_GRID_REQUIRED_FIELDS as readonly string[])
    .includes(field), `legacy six-stage receipt contract omitted ${field}`);
}
assert.deepEqual(SPARSE_CM12_SIX_STAGE_REQUIRED_RECEIPT_FIELDS.slice(0, 5),
  ["directCount", "closureCount", "executedCount", "skippedCount", "unknownCount"]);

const stageReceipt = (stage: 0 | 1 | 2 | 3 | 4 | 5): SparseCM12LogicalStageReceipt => ({
  stage,
  authorityIds: SPARSE_CM12_SIX_STAGE_AUTHORITY_EXPECTATIONS.filter((entry) =>
    (entry.stages as readonly number[]).includes(stage)).map((entry) => entry.id),
  validTileCount: 10, directCount: 2, closureCount: 3, executedCount: 5,
  skippedCount: 5, unknownCount: 0, uncoveredWriteCount: 0,
  expectedCoverageCount: 5, coveredCoverageCount: 5,
  producerGeneration: 12, consumerGeneration: 12, causeMask: 3,
  maximumClosureDepth: 2, faultCount: 0, firstFaultId: 0xffff_ffff,
});
const complete: SparseCM12SixStageReceipt = { acceptedGeneration: 12,
  topologyGeneration: 7, stages: [0, 1, 2, 3, 4, 5].map((stage) =>
    stageReceipt(stage as 0 | 1 | 2 | 3 | 4 | 5)) };
assert.deepEqual(inspectSparseCM12SixStageReceipts(undefined).state, "unknown");
assert.deepEqual(inspectSparseCM12SixStageReceipts(complete).state, "matched");
const missingMassAuthority: SparseCM12SixStageReceipt = { ...complete,
  stages: complete.stages.map((entry) => entry.stage === 1 ? { ...entry,
    authorityIds: entry.authorityIds.filter((id) => id !== "SRR1.mass") } : entry) };
assert.equal(inspectSparseCM12SixStageReceipts(missingMassAuthority).state, "fault");
assert.match(inspectSparseCM12SixStageReceipts(missingMassAuthority).issues.join(" "),
  /stage 1 omits SRR1.mass/);
const falselyClean: SparseCM12SixStageReceipt = { ...complete,
  stages: complete.stages.map((entry) => entry.stage === 2 ? { ...entry,
    skippedCount: 4 } : entry) };
assert.match(inspectSparseCM12SixStageReceipts(falselyClean).issues.join(" "),
  /executed\/skipped partition is incomplete/);

const localSource = "fn dirtyInvocation(id:u32)->u32{return id;} dispatchWorkgroupsIndirect";
const sources = Object.fromEntries(SPARSE_CM12_SIX_STAGE_AUTHORITY_EXPECTATIONS.map(
  (entry) => [entry.id, localSource],
)) as Record<SparseCM12SixStageAuthorityId, string>;
assert.doesNotThrow(() => assertSparseCM12SixStageProductionSources(sources));
assert.throws(() => assertSparseCM12SixStageProductionSources({ ...sources,
  "SAW1.gamma": `${localSource} acceptedTemplateCellInvocation`,
}), /SAW1.gamma uses forbidden accepted\/global token acceptedTemplateCellInvocation/);
assert.throws(() => assertSparseCM12SixStageProductionSources({ ...sources,
  PCA1: `${localSource} bakeBrickAggregateEdges`,
}), /PCA1 uses forbidden accepted\/global token bakeBrickAggregateEdges/);
assert.throws(() => assertSparseCM12SixStageProductionSources({ ...sources,
  FPP1: "dispatchWorkgroups(physicalBrickCapacity)",
}), /FPP1 lacks a local GPU authority/);

const overlaySource = readFileSync(new URL("../lib/core/webgpu-grid-overlay.ts",
  import.meta.url), "utf8");
assert.doesNotThrow(() => assertSparseCM12GridOverlayFailClosedSource(overlaySource));
assert.doesNotThrow(() => assertSparseCM12SixStageFramePlanLayout({ packetCount: 6 }));
assert.throws(() => assertSparseCM12SixStageFramePlanLayout({ packetCount: 5 }),
  /must publish all six/);

console.log("Sparse CM12 six-stage cutover contract fixtures: PASS");

const residentTS = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts", import.meta.url), "utf8");
const residentWGSL = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url),
"utf8");

const stageSlice = (id: string): string => {
  const marker = `stage("${id}", () => {`;
  const start = residentTS.indexOf(marker);
  if (start < 0) return "";
  const next = residentTS.indexOf("\n    stage(\"", start + marker.length);
  return residentTS.slice(start, next < 0 ? residentTS.length : next);
};
const methodSlice = (marker: string, nextMarker: string): string => {
  const start = residentTS.indexOf(marker);
  if (start < 0) return "";
  const end = residentTS.indexOf(nextMarker, start + marker.length);
  return residentTS.slice(start, end < 0 ? residentTS.length : end);
};
const liveIssues: string[] = [];
const requireTokens = (stage: string, source: string, tokens: readonly string[]): void => {
  if (!source) liveIssues.push(`${stage}: resident scheduling slice is absent`);
  for (const token of tokens) if (!source.includes(token)) {
    liveIssues.push(`${stage}: missing production token ${token}`);
  }
};
const rejectTokens = (stage: string, source: string, tokens: readonly string[]): void => {
  for (const token of tokens) if (source.includes(token)) {
    liveIssues.push(`${stage}: forbidden accepted/global runtime token ${token}`);
  }
};

const extension = stageSlice("transport-velocity-extension");
const facePreparation = stageSlice("face-preparation");
const mass = stageSlice("conservative-transport");
const gamma = stageSlice("gamma-diffusion");
const surface = stageSlice("surface-sharpening");
const pressureTopology = stageSlice("pressure-topology");
const projection = stageSlice("velocity-projection");
const presentation = methodSlice("private encodeFramePlanPresentation(",
  "\n  private encode");

requireTokens("stage0/VEX1", residentTS, [
  "createSparseCM12VexActivityBatchLayout", "createSparseCM12VexActivityBatchSchedule",
  "createSparseCM12VexActivityBatchPipelineDescriptors",
]);
requireTokens("stage0/VEX1", extension, [
  "beginVelocityExtensionCandidate", "advanceVelocityExtensionCandidates",
]);
rejectTokens("stage0/VEX1", extension, [
  "dispatchAccepted(\"initializeTransportVelocity\"",
  "extrapolateTransportVelocityToSource", "extrapolateTransportVelocityToDestination",
]);
requireTokens("stage0/FPA1-preparation", facePreparation, [
  "beginSparseCM12FacePreparationAuthority", "dispatchFaceProjectionAuthority",
  "finalizeSparseCM12FacePreparationExecution",
]);
requireTokens("stage0/FPA1-projection", projection, [
  "beginSparseCM12FaceProjectionAuthority", "dispatchFaceProjectionAuthority",
  "finalizeSparseCM12FaceProjectionExecution",
]);

requireTokens("stage1/SRR1", residentTS, [
  "scalarResultAuthority.encodePlan(encoder, this.activity)",
  "scalarResultAuthority.encodePublish(encoder, this.activity)",
]);
rejectTokens("stage1/mass", mass, [
  "dispatchAccepted(\"traceGammaAndBeta\"", "dispatchAccepted(\"scatterDensityDeficit\"",
  "dispatchAccepted(\"gatherConservativeDensity\"",
]);
requireTokens("stage1/mass", mass, [
  "dispatchScalarResult(\"traceGammaAndBeta\"",
  "dispatchScalarResult(\"scatterDensityDeficit\"",
  "dispatchScalarResult(\"gatherConservativeDensity\"",
  "dispatchScalarResult(\"compareSparseCM12MassResult\"",
  "scalarResultAuthority.encodePublish(encoder, this.activity)",
]);
rejectTokens("stage2/gamma", gamma, [
  "dispatchAccepted(\"scatterGammaSnapshot\"", "dispatchAccepted(\"finalizeGammaSnapshot\"",
  "dispatchAccepted(\"scatterGammaRefinement\"",
  "dispatchAccepted(\"finalizeGammaRefinement\"",
]);
requireTokens("stage2/gamma", gamma, ["scalarAuthority.encodeExecution(encoder, 1)"]);
rejectTokens("stage3/surface", surface, [
  "dispatchAccepted(\"prepareSharpeningField\"",
  "dispatchAccepted(\"scatterSharpeningMass\"",
  "dispatchAccepted(\"finalizeSharpening\"",
]);
requireTokens("stage3/surface", surface, [
  "scalarAuthority.encodeExecution(encoder, 2)", "scalarAuthority.encodeCommit(encoder)",
]);

requireTokens("stage4/PCF-PCA", pressureTopology, [
  "dispatchPersistentPressureCache", "finalizePersistentPressureCache",
]);
requireTokens("stage4/PSA", residentTS, [
  "dispatchPressureSolveAuthority", "finalizeSparseCM12PressureSolveAuthority",
]);
requireTokens("stage5/FPP1", presentation, [
  "dispatchWorkgroupsIndirect(this.presentationIndirectArguments",
  "executeSparseCM12FramePlanPresentationPacket",
  "commitSparseCM12FramePlanPresentationPacket",
  "finalizeSparseCM12FramePlanPresentationExecution",
]);
rejectTokens("stage5/FPL-FPP", presentation, [
  "dispatch(\"initializeSparseCM12FramePlanNext\", bricks)",
  "dispatch(\"populateSparseCM12PresentationFramePlan\", bricks)",
  "dispatch(\"resolveSparseCM12FramePlanNextClosure\", bricks)",
  "dispatch(\"sealSparseCM12FramePlanNextBricks\", bricks)",
  "dispatch(\"buildSparseCM12FramePlanPresentationPacket\", bricks)",
  "execute.dispatchWorkgroups(bricks)",
]);

requireTokens("all-stages/A4D2-FPL1", residentWGSL, [
  "createSparseCM12VexActivityBatchWGSL", "cm12ActivityPublishFramePlanRoot",
  "cm12ActivityPublishExactBrick", "cm12BatchVelocityExtensionRoot",
  "cm12BatchVelocityExtensionClosure", "cm12BatchVelocityExtensionScheduled",
  "cm12BatchVelocityExtensionFault",
]);
rejectTokens("stages0-4/FPL1", residentWGSL, [
  "markSparseCM12GlobalFramePlanReceipts", "0x1fu,0u",
  "actual\n  // global blast radius",
]);

if (liveIssues.length > 0) {
  console.error("Sparse CM12 live six-stage cutover: FAIL");
  for (const issue of liveIssues) console.error(`- ${issue}`);
  process.exitCode = 1;
} else {
  console.log("Sparse CM12 live six-stage cutover: PASS");
}
