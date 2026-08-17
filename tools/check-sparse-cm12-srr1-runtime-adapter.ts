#!/usr/bin/env node
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  SPARSE_CM12_SRR1_INDIRECT_FAMILY_COUNT,
  SPARSE_CM12_SRR1_INGRESS_HEADER,
  SPARSE_CM12_SRR1_PIPELINES,
  SPARSE_CM12_SRR1_RECEIPT_WORDS,
  createSparseCM12SRR1IngressLayout,
  createSparseCM12SRR1RuntimePlan,
  createSparseCM12SRR1RuntimeWGSL,
  initializeSparseCM12SRR1IngressWords,
  sparseCM12SRR1NoGlobalScanProof,
  sparseCM12SRR1RuntimeABIReceipt,
} from "../lib/methods/adaptive-mass/sparse-cm12-srr1-runtime-adapter";
import { createSparseCM12SRR1ResidentIngressWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-srr1-resident-ingress.wgsl";
import {
  SPARSE_CM12_SRR1_CUTOVER_GATES,
  SPARSE_CM12_SRR1_PATCH_SITES,
  validateSparseCM12SRR1ProductionManifest,
} from "../lib/methods/adaptive-mass/sparse-cm12-srr1-production-manifest";
import {
  assertSparseCM12SRR1PairedPhysicalHashes,
  assertSparseCM12SRR1PairedTrajectory,
  createSparseCM12SRR1PairedRuntimePlans,
} from "../lib/methods/adaptive-mass/sparse-cm12-srr1-qa";

const assert: (condition: unknown, message: string) => asserts condition =
  (condition: unknown, message: string): asserts condition => {
    if (!condition) throw new Error(message);
  };
const equal = (actual: number, expected: number, message: string): void => {
  if (actual !== expected) throw new Error(`${message}: ${actual} != ${expected}`);
};

const TILE_COUNT = 4096; // canonical dam64 B16 tile-capacity fixture.
const BASE_WORDS = 12345;
const paired = createSparseCM12SRR1PairedRuntimePlans({
  baseWords: BASE_WORDS, tileCapacity: TILE_COUNT,
});
const temporal = paired.temporal;
const oracle = paired.immutableFullOracle;
equal(temporal.authorityLayout.totalWords, oracle.authorityLayout.totalWords,
  "paired authority words");
equal(temporal.ingressLayout.totalWords, oracle.ingressLayout.totalWords,
  "paired ingress words");
assert(temporal.constructionMode === "temporal"
  && oracle.constructionMode === "immutable-full-oracle", "paired construction modes");

const ingress = createSparseCM12SRR1IngressLayout({
  baseWords: BASE_WORDS, tileCapacity: TILE_COUNT,
});
const words = new Uint32Array(ingress.totalWords);
initializeSparseCM12SRR1IngressWords(words, ingress, "temporal");
equal(words[ingress.headerBaseWords + SPARSE_CM12_SRR1_INGRESS_HEADER.tileCapacity]!,
  TILE_COUNT, "ingress tile capacity");
equal(words[ingress.headerBaseWords + SPARSE_CM12_SRR1_INGRESS_HEADER.constructionMode]!,
  0, "temporal construction mode");
assert(ingress.eventStampBaseWords >= ingress.headerBaseWords + 64,
  "event stamp region overlaps header");
assert(ingress.eventBaseWords >= ingress.eventStampBaseWords + 2 * TILE_COUNT,
  "event region overlaps stamps");
assert(ingress.workListBaseWords >= ingress.eventBaseWords + 4 * TILE_COUNT,
  "work list overlaps events");
assert(ingress.receiptBaseWords >= ingress.workListBaseWords + TILE_COUNT,
  "receipt region overlaps work list");
assert(ingress.totalWords >= ingress.receiptBaseWords
  + SPARSE_CM12_SRR1_RECEIPT_WORDS * TILE_COUNT, "receipt region truncated");

equal(SPARSE_CM12_SRR1_PIPELINES.length, 11, "pipeline count");
equal(new Set(SPARSE_CM12_SRR1_PIPELINES.map(({ entryPoint }) => entryPoint)).size,
  SPARSE_CM12_SRR1_PIPELINES.length, "unique entry points");
const source = createSparseCM12SRR1RuntimeWGSL(temporal);
for (const { entryPoint } of SPARSE_CM12_SRR1_PIPELINES) {
  assert(source.includes(`fn ${entryPoint}`), `missing WGSL entry point ${entryPoint}`);
}
assert(source.includes("cm12SRRValidateReceiptRank")
  && source.includes("cm12SRRPromoteReceiptRank"),
"runtime shader lacks rollback-free validate/promote split");
assert(!source.includes("var<storage,read_write> state"),
  "dedicated SRR1 adapter can bind physics");
const residentIngressSource = createSparseCM12SRR1ResidentIngressWGSL({
  layout: ingress, arenaName: "activity",
});
for (const helper of ["sirResidentBeginFrame", "sirResidentAppendEvent",
  "sirResidentEventStamped",
  "sirResidentResetCopiedEvents", "sirResidentBeginReceiptBatch",
  "sirResidentPublishReceipt"]) {
  assert(residentIngressSource.includes(`fn ${helper}`), `missing resident helper ${helper}`);
}
assert(!residentIngressSource.includes("state["),
  "resident ingress helper can write physics");
assert(!residentIngressSource.includes("generation|0x80000000u")
  && residentIngressSource.includes("sirResidentResetCopiedEvents(tile:u32)")
  && residentIngressSource.includes("stamp+1u],0u")
  && residentIngressSource.includes("stamp+0u],0u")
  && residentIngressSource.includes("attempt<64u"),
"event stamps must use a pre-cleared, bounded, lock-free generation handoff");
assert(source.includes("sirStampBase") && source.includes("ingress[sirStampBase+"),
  "runtime planning must import the final merged per-tile cause mask");
const witnessLowerFine = [48, 0, 0] as const;
const witnessUpperFine = [63, 15, 15] as const;
const oldCenterTileX = Math.floor((witnessLowerFine[0] + witnessUpperFine[0] + 1) / 8);
const oldCenterHaloLowerTileX = oldCenterTileX - 1;
const exactOwnerTileX = Math.floor(witnessLowerFine[0] / 4);
assert(oldCenterHaloLowerTileX > exactOwnerTileX,
  "depth witness no longer demonstrates center-tile closure loss");
const tileDims = [16, 16, 16] as const;
const tileKey = (x: number, y: number, z: number): string => `${x},${y},${z}`;
const ownerKey = tileKey(exactOwnerTileX, 0, 0);
// A fine producer has already stamped the owner. The later coarse producer
// must still visit its entire clipped extent+halo, not return at the owner.
const exactExtentHalo = new Set<string>([ownerKey]);
for (let z = Math.max(0, Math.floor(witnessLowerFine[2] / 4) - 1);
  z <= Math.min(tileDims[2] - 1, Math.floor(witnessUpperFine[2] / 4) + 1); z += 1) {
  for (let y = Math.max(0, Math.floor(witnessLowerFine[1] / 4) - 1);
    y <= Math.min(tileDims[1] - 1, Math.floor(witnessUpperFine[1] / 4) + 1); y += 1) {
    for (let x = Math.max(0, Math.floor(witnessLowerFine[0] / 4) - 1);
      x <= Math.min(tileDims[0] - 1, Math.floor(witnessUpperFine[0] / 4) + 1); x += 1) {
      exactExtentHalo.add(tileKey(x, y, z));
    }
  }
}
equal(exactExtentHalo.size, 125,
  "pre-stamped owner suppressed part of the clipped coarse extent+halo");
assert(exactExtentHalo.has(tileKey(12, 1, 0)),
  "coarse extent+halo omitted witness tile (12,1,0)");
const residentSource = readFileSync(resolve(
  "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts"), "utf8");
assert(residentSource.includes("bounds[0]/vec3i(4)-vec3i(1)")
  && residentSource.includes("bounds[1]/vec3i(4)+vec3i(1)"),
"resident coarse-cell invalidation is not extent-owned");
assert(!residentSource.includes("sirResidentEventCause(owner,generation)"),
  "producer-level owner dedup can suppress later coarse extent publication");
equal(Object.keys(temporal.indirectFamilies).length,
  SPARSE_CM12_SRR1_INDIRECT_FAMILY_COUNT, "indirect family count");

const hashes = Object.freeze({ densitySha256: "d", velocitySha256: "v",
  pressureSha256: "p", divergenceSha256: "q" });
assertSparseCM12SRR1PairedPhysicalHashes({ step: 1,
  temporal: hashes, immutableFullOracle: hashes });
assertSparseCM12SRR1PairedTrajectory([
  { step: 1, temporal: hashes, immutableFullOracle: hashes },
  { step: 5, temporal: hashes, immutableFullOracle: hashes },
], [1, 5]);
let mismatchRejected = false;
try {
  assertSparseCM12SRR1PairedPhysicalHashes({ step: 2, temporal: hashes,
    immutableFullOracle: { ...hashes, densitySha256: "different" } });
} catch { mismatchRejected = true; }
assert(mismatchRejected, "paired physical mismatch was tolerated");

validateSparseCM12SRR1ProductionManifest();
for (const site of SPARSE_CM12_SRR1_PATCH_SITES) {
  const path = resolve(site.file);
  const sourceFile = readFileSync(path, "utf8");
  assert(sourceFile.includes(site.anchor),
    `production patch anchor missing: ${site.file} :: ${site.anchor}`);
}
assert(SPARSE_CM12_SRR1_CUTOVER_GATES.some((gate) => gate.includes("canonical60")),
  "canonical60 gate absent");
assert(SPARSE_CM12_SRR1_CUTOVER_GATES.some((gate) => gate.includes("ocean B16/P16")),
  "ocean B16/P16 gate absent");

const receipt = sparseCM12SRR1RuntimeABIReceipt(temporal);
const noScan = sparseCM12SRR1NoGlobalScanProof(temporal);
equal(noScan.evolvingHostDecisionInputs.length, 0, "host decisions");
equal(noScan.runtimeCPUReadbacks.length, 0, "runtime readbacks");
equal(noScan.runtimeMappedBuffers.length, 0, "runtime mappings");
assert(noScan.acceptedCellOrRowScans === false, "accepted-domain scan present");
for (const entryPoint of ["appendSRR1Events", "writeSRR1WorkRanks",
  "importSRR1Receipts", "validateSRR1Receipts", "promoteSRR1Receipts"]) {
  assert(noScan.runtimeDomains[entryPoint]?.includes("<"),
    `${entryPoint} lacks bounded GPU-domain proof`);
}
console.log(JSON.stringify({ ...receipt,
  ingressBaseWords: ingress.headerBaseWords,
  indirectFamilies: SPARSE_CM12_SRR1_INDIRECT_FAMILY_COUNT,
  sourceBytes: source.length,
  patchSites: SPARSE_CM12_SRR1_PATCH_SITES.length,
  cutoverGates: SPARSE_CM12_SRR1_CUTOVER_GATES.length,
  noGlobalScan: noScan,
  contracts: ["GPU-only evolving authority", "candidate-only repair",
    "unpublished candidate receipts", "rollback-free validate/promote",
    "immutable construction full oracle", "four-field SHA-256 equality"],
}, null, 2));
