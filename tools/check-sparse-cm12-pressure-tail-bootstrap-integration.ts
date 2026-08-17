#!/usr/bin/env node
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  SPARSE_CM12_PRESSURE_TAIL_BANK_BYTES,
  acceptSparseCM12PressureTailGeneration,
  type SparseCM12PressureTailGenerationReceipt,
} from "../lib/methods/adaptive-mass/sparse-cm12-pressure-tail-authority";

const residentPath = resolve("lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts");
const residentWGSLPath = resolve(
  "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
);
const observabilityPath = resolve(
  "lib/methods/adaptive-mass/sparse-cm12-pressure-cutover-observability.ts",
);
const source = readFileSync(residentPath, "utf8");
const wgslSource = readFileSync(residentWGSLPath, "utf8");
const observabilitySource = readFileSync(observabilityPath, "utf8");

assert.equal(SPARSE_CM12_PRESSURE_TAIL_BANK_BYTES, 48);

const receipt = (overrides: Partial<SparseCM12PressureTailGenerationReceipt> = {}):
SparseCM12PressureTailGenerationReceipt => ({
  phaseAccepted: true,
  acceptedGeneration: 7,
  publishedGeneration: 7,
  activeBank: 0,
  fault: 0,
  predicate: "ordinary",
  predicateActive: true,
  expectedFamilyWorkgroups: [17, 5, 3, 1],
  familyWorkgroups: [17, 5, 3, 1],
  ...overrides,
});

assert.equal(acceptSparseCM12PressureTailGeneration(receipt()).accepted, true);
assert.equal(acceptSparseCM12PressureTailGeneration(
  receipt({ publishedGeneration: 0 }),
).reason, "missing-publication");
assert.equal(acceptSparseCM12PressureTailGeneration(
  receipt({ publishedGeneration: 6 }),
).reason, "stale-publication");
assert.equal(acceptSparseCM12PressureTailGeneration(
  receipt({ fault: 14, predicateActive: false, familyWorkgroups: [0, 0, 0, 0] }),
).reason, "psa-fault");
assert.equal(acceptSparseCM12PressureTailGeneration(
  receipt({ fault: 14, predicateActive: false, familyWorkgroups: [17, 0, 0, 0] }),
).reason, "fail-open-work");
assert.equal(acceptSparseCM12PressureTailGeneration(
  receipt({ predicate: "recovery", predicateActive: true }),
).accepted, true);
assert.equal(acceptSparseCM12PressureTailGeneration(
  receipt({ predicate: "recovery", predicateActive: false,
    familyWorkgroups: [0, 0, 0, 0] }),
).accepted, true);
assert.equal(acceptSparseCM12PressureTailGeneration(
  receipt({ predicate: "recovery", predicateActive: false }),
).reason, "predicate-work");

const firstCopy = source.indexOf(".encodeCopy(encoder, tailBank)");
const tokens = {
  compiledA: source.includes('"publishSparseCM12PressureTailA"'),
  compiledB: source.includes('"publishSparseCM12PressureTailB"'),
  adapterAllocated: source.includes("new WebGPUSparseCM12PressureTailAuthority("),
  adapterDestroyed: source.includes("pressureTailAuthority.destroy()"),
  publisherScheduled: source.includes('dispatch("publishSparseCM12PressureTailA", 1)')
    || source.includes('dispatch("publishSparseCM12PressureTailB", 1)'),
  passClosedBeforeCopy: false,
  copied: source.includes(".encodeCopy(encoder, tailBank)"),
  copiedDispatch: source.includes(".dispatch(activePass, tailBank, family)"),
  exactOrdinaryPredicate: wgslSource.includes(
    "fn psaPressureArithmeticActive()->bool{return pipelinedPressureActive();}",
  ),
  receiptGate: observabilitySource.includes(
    "receipt.psa.tailPublishedGeneration !== receipt.psa.acceptedGeneration",
  ),
  directTailArithmeticPreserved: [
    'dispatchPressureCell("updatePipelinedState")',
    'dispatchPressureCell("applyPipelinedImage")',
    'dispatchPressureCell("applyPipelinedRecovery")',
  ].every((token) => source.includes(token)),
};

const initializeGate = source.indexOf('dispatch("reducePipelinedInitialize", 1)');
const firstPublisher = Math.min(...[
  source.indexOf('dispatch("publishSparseCM12PressureTailA", 1)'),
  source.indexOf('dispatch("publishSparseCM12PressureTailB", 1)'),
].filter((index) => index >= 0));
tokens.passClosedBeforeCopy = Number.isFinite(firstPublisher) && firstCopy > firstPublisher
  && source.slice(firstPublisher, firstCopy).includes("closePass()");
const firstTailConsumer = source.indexOf('dispatchPressureCell("updatePipelinedState")');
const bootstrapOrder = initializeGate >= 0 && Number.isFinite(firstPublisher)
  && firstPublisher > initializeGate
  && (firstTailConsumer < 0 || firstPublisher < firstTailConsumer);

const bootstrapMissing = Object.entries({
  "PTL1 adapter allocation": tokens.adapterAllocated,
  "PTL1 adapter destruction": tokens.adapterDestroyed,
  "scheduled GPU tail publisher": tokens.publisherScheduled,
  "publisher after pipelined initialization and before the first tail consumer": bootstrapOrder,
  "storage-to-indirect pass boundary": tokens.passClosedBeforeCopy,
  "48-byte matching-bank copy": tokens.copied,
  "exact ordinary pressure predicate": tokens.exactOrdinaryPredicate,
  "accepted/published generation receipt gate": tokens.receiptGate,
  "bootstrap landing preserves the existing direct tail arithmetic":
    tokens.directTailArithmeticPreserved,
}).filter(([, present]) => !present).map(([requirement]) => requirement);

const report = {
  abi: "PTL1/v1",
  resident: residentPath,
  publishersCompiled: tokens.compiledA && tokens.compiledB,
  bootstrapIntegrated: bootstrapMissing.length === 0,
  arithmeticCutoverIntegrated: tokens.copiedDispatch
    && !tokens.directTailArithmeticPreserved,
  observedRootCause: tokens.compiledA && tokens.compiledB && !tokens.publisherScheduled
    ? "publishers are compiled but never encoded" : null,
  bootstrapMissing,
  laterArithmeticCutoverMissing: tokens.copiedDispatch ? []
    : ["copied-bank indirect consumers"],
  exactResidentAnchors: [
    "after topologyArena creation: allocate WebGPUSparseCM12PressureTailAuthority from topologyArena and pressureSolveAuthorityLayout, then pass it into the resident constructor",
    "pressure-rhs after journalRecord(0): publish bootstrap bank A, closePass, encodeCopy exactly 48 bytes; do not consume the snapshot yet",
    "replace psaPressureArithmeticActive true with pipelinedPressureActive so the copied bootstrap tuple is exact-or-zero",
    "retain direct updatePipelinedState/applyPipelinedImage/applyPipelinedRecovery dispatches in this landing",
    "destroy both PTL1 snapshot buffers with the resident",
    "later full PTL: after each arithmetic gate publish construction-fixed A/B, close, copy, then dispatch eligible families",
  ],
  bootstrapReceiptInvariant: "existing pressure-cutover receipt requires PSA fault 0,"
    + " acceptedGeneration==candidateGeneration, tailPublishedGeneration>0, and"
    + " tailPublishedGeneration==acceptedGeneration",
  laterFullReceiptInvariant: "activeBank in {0,1}; copied x counts exactly match"
    + " the PSA tuple for the selected ordinary/recovery predicate, otherwise all four"
    + " triplets are zero",
};

process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
if (!report.bootstrapIntegrated) {
  throw new Error(`PSA tail bootstrap integration incomplete: ${bootstrapMissing.join("; ")}`);
}
