#!/usr/bin/env node
/** No-GPU source contract for complete Sparse CM12 stage-cost receipts. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

const [capture, recorder, resident, probe, baseline] = await Promise.all([
  readFile(new URL("../lib/methods/adaptive-mass/adaptive-mass-frame-pipeline.ts",
    import.meta.url), "utf8"),
  readFile(new URL("../lib/core/performance-trace.ts", import.meta.url), "utf8"),
  readFile(new URL("../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
    import.meta.url), "utf8"),
  readFile(new URL("./probe-sparse-cm12-stage-cost.ts", import.meta.url), "utf8"),
  readFile(new URL("../artifacts/sparse-cm12-ocean-b16-p16-stage-cost-baseline.json",
    import.meta.url)),
]);

const seamStart = capture.indexOf("readonly residentStageSeams");
const seamEnd = capture.indexOf("  /** Stage the query readback", seamStart);
assert(seamStart >= 0 && seamEnd > seamStart, "adaptive capture seam block is missing");
const seam = capture.slice(seamStart, seamEnd);
assert.match(seam, /this\.gpu\?\.completePhase\(this\.encoder, phase\)/,
  "every resident stage, including the final stage, must close normally");
assert.doesNotMatch(seam, /completeFinalPhaseOnNextPass|SPARSE_CM12_RESIDENT_FINAL_STAGE/,
  "adaptive capture must not close a multipass final stage on its first pass");

const recorderStart = recorder.indexOf("export class GPUStageTimestampRecorder");
const resolveStart = recorder.indexOf("  resolve(encoder: GPUCommandEncoder): void {",
  recorderStart);
const readStart = recorder.indexOf("  async read(): Promise<PerformanceTrace", resolveStart);
assert(resolveStart >= 0 && readStart > resolveStart, "timestamp resolve block is missing");
const resolve = recorder.slice(resolveStart, readStart);
assert.match(resolve, /endOfPassWriteIndex/);
assert.match(resolve, /GPU stage trace close/);
assert.match(resolve, /marker\.dispatchWorkgroups\(1\)/,
  "the trailing boundary must use the prepared observable marker");
assert.match(resolve, /this\.finalBoundaryAnchor\.source[\s\S]*this\.markerResources\.buffer/,
  "the trailing marker must consume the resident's final publication");

const publication = resident.slice(
  resident.indexOf('stage("presentation-publication", () => {'),
  resident.indexOf("  /**\n   * Turn the marker view", resident.indexOf(
    'stage("presentation-publication", () => {')),
);
assert.match(publication, /this\.encodeVelocityExtensionPlan/);
assert.match(publication, /this\.encodeFramePlanPresentation/);
assert.match(publication, /encoder\.copyBufferToBuffer\(this\.topologyArena/,
  "the final accepted-indirect copy must remain ahead of timestamp resolve");
assert.match(publication, /seams\?\.anchorFinalBoundary\?\.\(this\.acceptedIndirectArguments/,
  "the final boundary must anchor after the accepted-indirect publication");
assert.match(resident,
  /Sparse CM12 accepted indirect dispatch snapshot[\s\S]*GPUBufferUsage\.COPY_SRC/,
  "the accepted-indirect publication must be usable as a timestamp dependency");
assert.match(seam, /anchorFinalBoundary[\s\S]*this\.gpu\?\.anchorFinalBoundary/,
  "adaptive capture must forward the resident's final dependency anchor");

assert.match(probe,
  /argument\("enforce-pressure-receipts", "1"\) === "1"/,
  "pressure receipt enforcement must default on");
assert.match(probe, /pressureAuthorityInspection\.complete[\s\S]*ptr\.fault === 0/,
  "pressure receipt enforcement must reject a PTR fault");
assert.match(probe, /candidate\.context === expectedTraceContext/,
  "timestamp polling must reject a late prior-frame trace");
assert.match(probe, /GPUStageTimestampRecorder\.prepare\(device\)/,
  "the observable trailing marker must be prepared before capture");
assert.match(probe, /GPU_WORK_CHUNK_BY_LABEL[\s\S]*workChunkSamples/,
  "the stage-cost receipt must publish disjoint concrete work chunks");
assert.match(probe, /if \(stage\.costInsideStage\) continue;/,
  "shared diagram nodes must not be emitted as independent numeric stages");
assert.match(probe, /maximumStageChunkError_ms/,
  "every concrete stage rollup must reconcile against its owned chunks");
assert.match(probe, /createProcessRetainedDawnGPU/,
  "native Dawn must remain retained through process teardown");
assert.match(resident,
  /readFinalScalarMaskHeaderQA\(\)[\s\S]*SPARSE_CM12_FINAL_SCALAR_MASK_HEADER_WORDS/,
  "per-advance final-scalar diagnosis must use the fixed FSM1 header readback");
assert.match(resident,
  /readVelocityExtensionHeaderQA\(\)[\s\S]*SPARSE_CM12_VELOCITY_EXTENSION_HEADER_WORDS/,
  "VEX fault triage must expose a fixed 128-byte header readback");
assert.match(probe,
  /priorFinalScalarMaskGeneration === 0[\s\S]*generation === expectedFrameControlGeneration/,
  "the exact FSM1 successor oracle must admit its generation-0 construction bootstrap");
assert.match(probe,
  /if \(vexHeader\.firstFault\)[\s\S]*qaSolver\.readVelocityExtensionQA\(\)/,
  "full VEX QA may be materialized only after an exact header first-fault receipt");
assert.match(probe,
  /writeFile\(outputPath[\s\S]*assert\.equal\(diagnosticFailure, undefined/,
  "the diagnostic artifact must be persisted before a receipt failure is raised");
assert.match(probe, /manager\.whenIdle\(\)[\s\S]*queue\.onSubmittedWorkDone\(\)[\s\S]*teardownSolver\?\.destroy\(\)[\s\S]*device\.destroy\(\)/,
  "teardown must drain compilation and queue work before destroying the solver and device");
for (const token of ["gitCommit", "gitDirty", "methodProfile", "ladder",
  "PINNED_BASELINE_SHA256"]) {
  assert(probe.includes(token), `stage-cost provenance is missing ${token}`);
}

const baselineSha256 = createHash("sha256").update(baseline).digest("hex");
assert.equal(baselineSha256,
  "87f2463d688dd36c2313258550992eaab8a9e9c8581888ba553c4f254ebbf4e2");

console.log(JSON.stringify({ passed: true, noGPU: true,
  finalBoundary: "ordinary completion through observable trailing marker",
  pressureReceiptsDefault: "enforced",
  pinnedBaselineSha256: baselineSha256 }, null, 2));
