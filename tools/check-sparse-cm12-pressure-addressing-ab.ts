#!/usr/bin/env node
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  SPARSE_CM12_PRESSURE_ADDRESSING_AB_HEADER as H,
  SPARSE_CM12_PRESSURE_ADDRESSING_AB_PHASE,
  createSparseCM12PressureAddressingABInitialWords,
  createSparseCM12PressureAddressingABLayout,
  createSparseCM12PressureAddressingABPipelineDescriptors,
  createSparseCM12ProductionPressureAddressingLayout,
  inspectSparseCM12PressureAddressingABReceipt,
  sparseCM12PressureAddressingABPipelineConstants,
  sparseCM12PressureAddressingABReceiptAccepted,
} from "../lib/methods/adaptive-mass/sparse-cm12-pressure-addressing-ab";
import { createSparseCM12PressureAddressingABWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-pressure-addressing-ab.wgsl";

const layout = createSparseCM12PressureAddressingABLayout({
  baseWords: 113, cellCapacity: 4_097,
  brickFineResolution: 16, presentationPageResolution: 16,
  constructionMode: "qa-pressure-addressing-ab",
});
assert.equal(layout.baseWords, 128);
assert.equal(layout.listBaseWords % 64, 0);
assert.equal(layout.totalBytes, 4 * layout.totalWords);
assert.equal(layout.constructionOnly, true);
assert.equal(layout.runtimeSelectable, false);
const productionLayout = createSparseCM12ProductionPressureAddressingLayout({
  baseWords: 113, cellCapacity: 4_097,
  brickFineResolution: 16, presentationPageResolution: 16,
});
assert.equal(productionLayout.constructionOnly, false);
assert.equal(productionLayout.productionAddressAuthority, true);
assert.equal(productionLayout.runtimeSelectable, false);
assert.throws(() => createSparseCM12PressureAddressingABLayout({
  cellCapacity: 10, brickFineResolution: 16, presentationPageResolution: 8 as 16,
  constructionMode: "qa-pressure-addressing-ab",
}), /B16\/P16/);

const initial = createSparseCM12PressureAddressingABInitialWords(layout);
assert.equal(initial.length, layout.totalWords - layout.baseWords);
assert.equal(initial[H.listBaseWords], layout.listBaseWords);
assert.equal(initial[layout.listBaseWords - layout.baseWords], 0xffff_ffff);

const acceptedArena = new Uint32Array(layout.totalWords);
acceptedArena.set(initial, layout.baseWords);
const set = (word: number, value: number) => { acceptedArena[layout.baseWords + word] = value; };
set(H.phase, SPARSE_CM12_PRESSURE_ADDRESSING_AB_PHASE.accepted);
set(H.expectedPCMGeneration, 7);set(H.materializedPCMGeneration, 7);
set(H.expectedCount, 3);set(H.materializedCount, 3);
set(H.materializedExecutions, 3);set(H.verifiedExecutions, 3);
set(H.materializedHash, 0x1234);set(H.verifiedHash, 0x1234);
set(H.acceptedReceipts, 1);
const receipt = inspectSparseCM12PressureAddressingABReceipt(acceptedArena, layout);
assert.equal(sparseCM12PressureAddressingABReceiptAccepted(receipt), true);
set(H.mismatchCount, 1);
assert.equal(sparseCM12PressureAddressingABReceiptAccepted(
  inspectSparseCM12PressureAddressingABReceipt(acceptedArena, layout)), false);

assert.deepEqual(sparseCM12PressureAddressingABPipelineConstants("canonicalRankSelect"),
  { CM12_PRESSURE_ADDRESS_MODE: 0 });
assert.deepEqual(sparseCM12PressureAddressingABPipelineConstants("materializedList"),
  { CM12_PRESSURE_ADDRESS_MODE: 1 });
for (const mode of ["canonicalRankSelect", "materializedList"] as const) {
  const descriptors = createSparseCM12PressureAddressingABPipelineDescriptors(mode);
  assert.equal(descriptors.length, 5);
  assert(descriptors.every((entry) => entry.constructionOnly
    && !entry.runtimeSelectable && !entry.outputSelectable));
}

const helpers = createSparseCM12PressureAddressingABWGSL({
  layout, arenaName: "arena", workgroupSize: 64,
});
const productionHelpers = createSparseCM12PressureAddressingABWGSL({
  layout: productionLayout, arenaName: "arena", workgroupSize: 64,
  fixedMode: "materializedList",
});
assert.match(helpers, /CM12_PRESSURE_ADDRESS_MODE==PAB_MODE_RANK_SELECT/);
assert.match(helpers, /CM12_PRESSURE_ADDRESS_MODE!=PAB_MODE_LIST/);
assert.match(helpers, /return PAB_INVALID;/);
assert.doesNotMatch(helpers,
  /PAB_MODE_LIST[\s\S]{0,500}return pcmCellRankSelect\(rank\)/,
  "materialized-list arm must never fall back to rank-select");
assert.match(helpers, /actual=pabLoad\(PAB_LIST_BASE\+rank\)/);
assert.match(helpers, /expected=pcmCellRankSelect\(rank\)/);
assert.match(helpers, /fn pabPressureAddressingReady\(\)->bool/);
assert.doesNotMatch(productionHelpers,
  /pabLoad\([^\n]*verifiedExecutions[^\n]*\)!=count/,
  "production finalization must not repeat the construction rank-select oracle");

// Negative policy fixture: the exact forbidden fallback must be detected.
const forbidden = helpers.replace("return PAB_INVALID;}",
  "return pcmCellRankSelect(rank);}");
assert.match(forbidden,
  /PAB_MODE_LIST[\s\S]{0,500}return pcmCellRankSelect\(rank\)/);

const resident = readFileSync(fileURLToPath(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts", import.meta.url,
)), "utf8");
const residentWGSL = readFileSync(fileURLToPath(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url,
)), "utf8");
const probe = readFileSync(fileURLToPath(new URL(
  "./probe-sparse-cm12-pressure-addressing-ab.ts", import.meta.url,
)), "utf8");
assert.match(resident,
  /pressureCellCount:\s*pcmCell\[\s*SPARSE_CM12_CANONICAL_MEMBERSHIP_DOMAIN_HEADER\.totalCount\s*\]/,
  "pressure-cell diagnostics must use the canonical PCM accepted count");
assert.doesNotMatch(resident,
  /copyBufferToBuffer\(this\.conditioning,\s*0,\s*this\.diagnosticsReadback/,
  "pressure-cell diagnostics must never reinterpret reusable conditioning scratch");
const residentHooks = {
  rankFactory: resident.includes("createPressureAddressingRankSelectForQA"),
  listFactory: resident.includes("createPressureAddressingMaterializedListForQA"),
  generatedWGSL: residentWGSL.includes("createSparseCM12PressureAddressingABWGSL"),
  addressHelper: residentWGSL.includes("pabPressureCellAddress(invocation)"),
  materializerDispatch: resident.includes("materializePressureCellAddresses"),
  verifierDispatch: resident.includes("verifyPressureCellAddresses"),
  receiptReader: resident.includes("readPressureAddressingABQA"),
};
// This checker is the serial handoff ledger. Once any hook lands, all must
// land atomically; a partial cutover would make the A/B incomparable.
const installedHookCount = Object.values(residentHooks).filter(Boolean).length;
assert(installedHookCount === 0 || installedHookCount === Object.keys(residentHooks).length,
  `partial PAB1 resident integration ${JSON.stringify(residentHooks)}`);
if (installedHookCount === 0) {
  assert.match(residentWGSL,
    /fn pressureCellInvocation\(invocation:u32\)[\s\S]{0,160}pcmCellRankSelect\(invocation\)/,
    "pre-handoff resident pressure address root moved unexpectedly");
} else {
assert.match(residentWGSL,
    /fn pressureCellInvocation\(invocation:u32\)[\s\S]{0,160}pabPressureCellAddress\(invocation\)/,
    "PAB1 resident pressure address helper is not installed");
}
assert.match(resident, /pressureAddressingModeForQA \?\? "materializedList"/,
  "ordinary production must compile the immutable materialized pressure list");
assert.match(resident, /createSparseCM12ProductionPressureAddressingLayout/,
  "ordinary production must allocate the canonical pressure address list");
assert.match(resident, /Sparse CM12 production pressure-address materialization/,
  "production must publish the address list after PCM finalization");
assert.doesNotMatch(resident, /Sparse CM12 production pressure-address verification/,
  "construction-only rank-select verification must not run in production");
assert.match(resident,
  /descriptor\.key !== "verifyPressureCellAddresses"/,
  "production must not compile the construction-only verifier pipeline");
assert.match(residentWGSL,
  /fn psaPressureAddressingReady\(\)->bool\{return pabPressureAddressingReady\(\);\}/,
  "PSA must reject a missing or stale pressure-address receipt");
assert.match(probe, /createPressureAddressingRankSelectForQA/);
assert.match(probe, /createPressureAddressingMaterializedListForQA/);
assert.match(probe, /createAsync fallback is forbidden/);
assert.doesNotMatch(probe, /\?\s*WebGPUAdaptiveMassSolver\.createAsync/,
  "PAB1 probe must not substitute the production constructor");
assert.match(probe, /createSymmetricExpansionScene/);
assert.match(probe, /const dimensions = \[32, 16, 32\] as const/);
assert.match(probe, /fixedDt_s = scene\.numerics\.maxDt_s = CM12_PAPER_DT_S/);
assert.match(probe,
  /PAB1 symmetric correctness ladder requires warmup 0 and frames 5, 20, or 60/);
assert.match(probe, /symmetricCorrectnessFrameRungs: \[5, 20, 60\]/);
assert.match(probe, /gitStatusSha256/);
assert.match(probe, /gitDirty: gitStatus\.trim\(\)\.length > 0/);
assert.match(probe, /resolvedMethodValues: values/);
assert.match(probe, /methodProfile: "balanced"/);
assert.match(probe, /--arm-order=rank-first/);
assert.match(probe, /argument\("arm-order", "rank-first"\)/);
assert.match(probe, /arm-order must be rank-first or list-first/);
assert.match(probe, /armOrder === "rank-first"/);
assert.match(probe, /armOrder, counterbalanced: false/);
assert.match(probe, /performanceClaim: false/);
assert.match(probe, /sequential-order-confounded/);
assert.doesNotMatch(probe,
  /performanceClaim: !symmetricCorrectnessOnly && allSRRFaultZero/,
  "one sequential PAB1 artifact must not publish a production performance claim");
assert.match(probe, /correctness-only-no-performance-claim/);
assert.match(probe, /physicalHashesIdenticalEveryMeasuredFrame/);
assert.match(probe, /PAB1 symmetric physical receipt differs/);
assert.match(probe, /constructionAttributionPrimedWithoutPhysicsAdvance/);
assert.match(probe, /constructionAttribution\.encodedStep, 0/);
assert.match(probe, /rawPressureAuthority/);
assert.match(probe, /psaGenerations/);
assert.match(probe, /partialFailureFrames/);
assert.match(probe, /await device\.queue\.onSubmittedWorkDone\(\)/);
assert.match(probe, /candidate\.sampleId > priorSampleId/);
assert.match(probe, /candidate\.context === expectedContext/);
assert.match(probe, /adaptive-mass:sim-\$\{\(step \* dt_s\)\.toFixed\(6\)\}/);
assert.match(probe, /observedContext/);
assert.match(probe,
  /captureGap_ms - \(performance\.now\(\) - pollStarted_ms\)/);
assert.match(probe, /enforcedInterAdvanceWait_ms/);
assert.match(probe, /if \(step < warmup \+ frames\)/);
assert.match(probe, /captureGap_ms - \(performance\.now\(\) - pollStarted_ms\)/);
assert.match(probe, /setTimeout\(done, Math\.min\(5, remaining_ms\)\)/);
assert.match(probe, /hardwareTracePoll/);
assert.doesNotMatch(probe, /adaptivePressureCellCount/,
  "paired PAB1 domain equality must use canonical PCM QA, not stats counters");

const destroyStart = resident.indexOf("destroy(): void {");
const destroyEnd = resident.indexOf("private assertLive()", destroyStart);
assert(destroyStart >= 0 && destroyEnd > destroyStart,
  "resident destroy lifecycle block is missing");
const destroySource = resident.slice(destroyStart, destroyEnd);
const requiredResidentIndirectBufferDestroys = [
  "frameControlIndirectArguments",
  "pressureTopologyRepairIndirectArguments",
  "persistentPressureCacheIndirectArguments",
  "pressureSolveAuthorityIndirectArguments",
  "faceProjectionAuthorityIndirectArguments",
] as const;
for (const buffer of requiredResidentIndirectBufferDestroys) {
  assert.match(destroySource, new RegExp(`this\\.${buffer}[,\\.]`),
    `${buffer} is not destroyed with the resident`);
}

const standalone = /* wgsl */ `
@group(0) @binding(0) var<storage,read_write> arena:array<atomic<u32>>;
fn pcmCellAcceptedGeneration()->u32{return 7u;}
fn pcmCellAcceptedCount()->u32{return 3u;}
fn pcmCellRankSelect(rank:u32)->u32{return select(0xffffffffu,rank+1u,rank<3u);}
fn pcmCellContains(cell:u32)->bool{return cell>=1u&&cell<=3u;}
fn cellActive(cell:u32)->bool{return pcmCellContains(cell);}
${helpers}
`;
const directory = mkdtempSync(join(tmpdir(), "sparse-cm12-pab1-"));
try {
  const path = join(directory, "pab1.wgsl");
  writeFileSync(path, standalone);
  const checked = spawnSync(process.env.NAGA ?? "naga", [path], { encoding: "utf8" });
  if (checked.error) throw checked.error;
  if (checked.status !== 0) throw new Error(checked.stderr || checked.stdout);
} finally {
  rmSync(directory, { recursive: true, force: true });
}

process.stdout.write(`${JSON.stringify({
  abi: "PAB1/v1", constructionOnlyQA: true,
  productionMaterializedAuthority: true, runtimeFallback: false,
  cellCapacity: layout.cellCapacity, listBaseWords: layout.listBaseWords,
  totalWords: layout.totalWords, receipt: "CPU accepted/rejected",
  naga: "materializer+verifier+dual-address helper valid",
  residentHooks,
  requiredResidentIndirectBufferDestroys,
  productionResidentHook: installedHookCount === 0
    ? "absent by design until serial handoff" : "complete",
}, null, 2)}\n`);
