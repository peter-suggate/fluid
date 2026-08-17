#!/usr/bin/env node
/** Static contract for construction-only actual FPA VEX xyz-read census. */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import {
  createSparseCM12FpaVexReadCensusInitialWords,
  createSparseCM12FpaVexReadCensusLayout,
  inspectSparseCM12FpaVexReadCensusQA,
  inspectSparseCM12FpaVexReadCensusSummaryQA,
} from "../lib/methods/adaptive-mass/sparse-cm12-fpa-vex-read-census";
import { createSparseCM12FpaVexReadCensusWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-fpa-vex-read-census.wgsl";
import { createWebgpuSparseCM12ResidentWGSL } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl";
import {
  SPARSE_CM12_VEX_REVERSE_POLICY,
  buildSparseCM12VexReverseDependency,
  sparseCM12VexReverseDependencyPolicyHash,
  sparseCM12VexRowsScheduledByChangedTiles,
  summarizeSparseCM12VexDonorTiles,
} from "../lib/methods/adaptive-mass/sparse-cm12-vex-reverse-dependency";

const censusWGSLSource = readFileSync(new URL(
  "../lib/methods/adaptive-mass/sparse-cm12-fpa-vex-read-census.wgsl.ts",
  import.meta.url), "utf8");
const residentWGSLSource = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
  import.meta.url), "utf8");

assert.match(residentWGSLSource,
  /fn prepareTransportFaces[\s\S]*?row!=INVALID&&fpaPreparationRowLive\(row\)[\s\S]*?prepareTransportFaceRow\(row\)/,
  "construction full-FPA oracle must execute only the production live-row domain");
for (const entry of [
  "captureSparseCM12PriorFaceForOracle",
  "captureSparseCM12FpaOracleAndRestore",
  "verifySparseCM12FpaOracle",
]) {
  assert.match(censusWGSLSource,
    new RegExp(`fn ${entry}\\([\\s\\S]*?!fpaPreparationRowLive\\(row\\)`),
    `${entry} must ignore rows outside the exact production FPA live domain`);
}

const layout = createSparseCM12FpaVexReadCensusLayout({
  baseWords: 3, cellCapacity: 97, rowCapacity: 131, tileCapacity: 17,
});
assert.equal(layout.baseWords % 64, 0);
assert.equal(layout.maximumTilesPerRow, 32);
assert(layout.changedTileBitsBaseWords > layout.changedCellBitsBaseWords);
assert(layout.directChangedCellBitsBaseWords > layout.predictedRowBitsBaseWords);
assert(layout.oracleChangedRowBitsBaseWords > layout.sourceFaceChangedRowBitsBaseWords);
assert(layout.xyzDonorCellBitsBaseWords[0] > layout.oracleChangedRowBitsBaseWords);
assert(layout.xyzDonorCellBitsBaseWords[1] > layout.xyzDonorCellBitsBaseWords[0]);
assert(layout.rowTilesBaseWords[0] > layout.rowTileCountBaseWords[1]);
assert(layout.rowTilesBaseWords[1] > layout.rowTilesBaseWords[0]);
assert(layout.velocityBitsBaseWords[0] >= layout.rowTilesBaseWords[1]
  + layout.rowCapacity * layout.maximumTilesPerRow);
const initial = createSparseCM12FpaVexReadCensusInitialWords(layout);
const receipt = inspectSparseCM12FpaVexReadCensusQA(initial, layout);
assert.equal(receipt.fault, 0); assert.equal(receipt.rowTiles.length, 131);
assert.equal(receipt.changedEffectiveCellCount, 0);
const compact = inspectSparseCM12FpaVexReadCensusSummaryQA(
  initial.subarray(0, layout.compactSummaryWords), layout);
assert.equal(compact.changedEffectiveTileCount, 0);

const wgsl = createSparseCM12FpaVexReadCensusWGSL(layout);
for (const entry of ["beginSparseCM12FpaVexReadCensus",
  "clearSparseCM12FpaVexReadCensus", "captureSparseCM12ChangedEffectiveTransport",
  "captureSparseCM12PriorFaceForOracle", "captureSparseCM12FpaOracleAndRestore",
  "scheduleSparseCM12FpaFromAcceptedVexReads", "verifySparseCM12FpaOracle",
  "finalizeSparseCM12FpaVexReadSummary", "commitSparseCM12FpaVexReadCensus"])
  assert(wgsl.includes(`fn ${entry}`), `missing ${entry}`);
const changedCapture = wgsl.slice(
  wgsl.indexOf("fn captureSparseCM12ChangedEffectiveTransport"),
  wgsl.indexOf("fn captureSparseCM12PriorFaceForOracle"));
assert.match(changedCapture,
  /acceptedXyz=fvrCellBit\(fvrXyzDonorCellsBase\(accepted\),cell\)[\s\S]*if\(!acceptedXyz[\s\S]*return;\}[\s\S]*let effective=cm12ExtensionTransportVelocity\(cell\)/,
  "pre-schedule xyz capture must call the VEX accessor for accepted donor cells only");
assert.equal(changedCapture.match(/cm12ExtensionTransportVelocity\(cell\)/g)?.length, 1,
  "accepted-donor capture must evaluate the accessor exactly once");
assert.match(wgsl,
  /fn fvrTransportValidityClassNoFault[\s\S]*AcceptedGeneration\(\)==0u[\s\S]*ExtensionFaulted\(\)[\s\S]*sourceDensity\(\)\+cell[\s\S]*BlastStamp\+cell[\s\S]*AcceptedOwner\+cell[\s\S]*AcceptedDepth\+cell[\s\S]*<9u/,
  "effective-w direct classifier must mirror every accessor branch without faulting");
assert.match(wgsl, /incrementalActivityStableTile\(cell\)\.x/,
  "donors must use canonical stable 4^3 tile ids");
assert.match(wgsl, /atomicStore\(&topologyArena\[counts\+row\],count\)/,
  "each row must publish an exact deduplicated donor-tile list");
assert.match(wgsl, /if\(count>=FVR_MAX_ROW_TILES\)[\s\S]*atomicStore[^\n]*,3u\)[\s\S]*atomicMin[^\n]*row/,
  "row tile overflow must fail closed with provenance");
assert.match(wgsl, /FVR_PRIOR_FACE.*FVR_ORACLE_FACE/s,
  "full oracle must retain the prior result independently");
assert.match(wgsl, /state\[destinationFaceVelocity\(\)\+row\]=bitcast<f32>\(prior\)/,
  "observational full oracle must restore before production FPA");
assert.match(wgsl, /FVR_ORACLE_FACE\+row[\s\S]*fpaPreparedAuthorityBits\(row\)/,
  "production FPA must be verified against the unchanged full oracle");
assert.match(wgsl,
  /scheduleSparseCM12FpaFromAcceptedVexReads[\s\S]*fvrAcceptedParity\(\)[\s\S]*FVR_CHANGED_TILE_BITS/,
  "predicted S must come exclusively from accepted donor tiles intersected with changes");
assert.match(censusWGSLSource,
  /acceptedTopology!=topologyGeneration[\s\S]*topologyRebuildPublished[\s\S]*scheduleSparseCM12FpaFromAcceptedVexReads[\s\S]*topologyRebuildPublished/,
  "construction census must explicitly rebuild and schedule every live row on topology epoch changes");
assert.doesNotMatch(censusWGSLSource,
  /acceptedTopology!=topologyGeneration[\s\S]{0,200}fault\)}\],7u/,
  "topology rebuild must not masquerade as a census fault");
const clear = wgsl.slice(wgsl.indexOf("fn clearSparseCM12FpaVexReadCensus"),
  wgsl.indexOf("fn captureSparseCM12ChangedEffectiveTransport"));
assert.match(clear, /fvrRowCountBase\(fvrCandidateParity\(\)\)/,
  "frame clear must target only the candidate graph bank");
assert.doesNotMatch(clear, /fvrAcceptedParity\(\)/,
  "frame clear must never erase the accepted prior graph");
const begin = wgsl.slice(wgsl.indexOf("fn beginSparseCM12FpaVexReadCensus"),
  wgsl.indexOf("fn clearSparseCM12FpaVexReadCensus"));
assert.doesNotMatch(begin, /for\(var word=/,
  "begin must reset explicit transient words, never an accepted-header range");
assert.doesNotMatch(begin,
  new RegExp(`atomicStore\\(&topologyArena\\[${layout.headerBaseWords
    + 20}u\\]`), "begin must preserve accepted parity");
assert.doesNotMatch(begin,
  new RegExp(`atomicStore\\(&topologyArena\\[${layout.headerBaseWords
    + 21}u\\]`), "begin must preserve accepted generation");
assert.match(wgsl,
  /FVR_DENSITY_CLASS_A[\s\S]*FVR_VALIDITY_CLASS_A[\s\S]*FVR_DIRECT_CHANGED_CELL_BITS/,
  "phase and effective-w changes must enter the direct producer union");
assert.match(wgsl,
  /sourceFaceVelocity\(\)\+row[\s\S]*FVR_SOURCE_FACE_CHANGED_BITS/,
  "source-face changes must enter the direct producer union");
assert.match(wgsl,
  /scheduleSparseCM12FpaFromAcceptedVexReads[\s\S]*FVR_SOURCE_FACE_CHANGED_BITS[\s\S]*FVR_DIRECT_CHANGED_CELL_BITS/,
  "predicted S must union accepted xyz dependencies with exact direct producers");
assert.match(wgsl, /hasRigidBodies\(\)[\s\S]*,8u\)/,
  "unsupported rigid epochs must reject instead of weakening C-subset-S");
assert.match(censusWGSLSource,
  /acceptedTopology!=topologyGeneration[\s\S]*topologyRebuildPublished/,
  "topology epoch changes must invalidate the stale graph and request an explicit QA rebuild");
assert.match(wgsl,
  /let accepted=fvrAcceptedParity\(\);let candidate=fvrCandidateParity\(\)[\s\S]*fvrVelocityBase\(accepted\)[\s\S]*fvrVelocityBase\(candidate\)/,
  "effective xyz capture must read accepted and write candidate snapshots");
assert.match(wgsl,
  /fvrDensityClassBase\(accepted\)[\s\S]*fvrDensityClassBase\(candidate\)/,
  "density-class capture must be transactional");
assert.match(wgsl,
  /fvrValidityClassBase\(accepted\)[\s\S]*fvrValidityClassBase\(candidate\)/,
  "effective-w capture must be transactional");
assert.match(wgsl,
  /fvrSourceFaceBase\(fvrAcceptedParity\(\)\)[\s\S]*fvrSourceFaceBase\(fvrCandidateParity\(\)\)/,
  "source-face capture must be transactional");
assert.match(wgsl,
  /FVR_ORACLE_CHANGED_ROW_BITS[\s\S]*if\(oracleChanged&&!predicted\)[\s\S]*atomicAdd[\s\S]*atomicMin/,
  "full oracle must publish exact C-minus-S omission provenance");
assert.match(wgsl,
  /FVR_PRIOR_ACCEPTED_AUTHORITY[\s\S]*fpaPreparedAuthorityBits\(row\)[\s\S]*FVR_PRIOR_ACCEPTED_AUTHORITY\+row\]\)!=oracle[\s\S]*FVR_ORACLE_CHANGED_ROW_BITS/,
  "C must compare against persistent accepted FPA authority, not alternating destination bank");
assert.match(wgsl,
  /commitSparseCM12FpaVexReadCensus[\s\S]*\|\|atomicLoad[\s\S]*fvrCandidateParity/,
  "A-to-B promotion must occur only after a zero-omission clean verify");
assert.match(censusWGSLSource,
  /scheduleSparseCM12FpaFromAcceptedVexReads[\s\S]*fpaPreparationRowLive[\s\S]*acceptedGeneration[\s\S]*==0u/,
  "construction must conservatively schedule all live rows without pre-verify promotion");
const schedule = wgsl.slice(wgsl.indexOf("fn scheduleSparseCM12FpaFromAcceptedVexReads"),
  wgsl.indexOf("fn finalizeSparseCM12FpaVexReadSummary"));
assert.match(schedule, /!fpaPreparationRowLive\(row\)\)\{return;\}/,
  "predicted S must contain schedulable live rows only");
const finalize = censusWGSLSource.slice(
  censusWGSLSource.indexOf("fn finalizeSparseCM12FpaVexReadSummary"),
  censusWGSLSource.indexOf("fn captureSparseCM12FpaOracleAndRestore"));
assert.match(finalize,
  /!fpaPreparationRowLive\(row\)\)\{return;\}[\s\S]*liveRowCount[\s\S]*FVR_ROW_TILE_HISTOGRAM/,
  "row-tile histogram must count exactly the live accepted rows");
assert.doesNotMatch(censusWGSLSource, /fn bootstrapSparseCM12FpaVexReadCensus/,
  "construction must not publish an unverified candidate before production verify");
assert.match(censusWGSLSource,
  /commitSparseCM12FpaVexReadCensus[\s\S]*construction[\s\S]*constructionBootstrapPublished/,
  "construction bootstrap receipt must publish only inside clean post-verify commit");
assert.doesNotMatch(createSparseCM12FpaVexReadCensusWGSL(undefined),
  /fvrRecord|FVR|topologyArena|@compute/,
  "ordinary source seam must be absent, not merely allocation/dispatch free");
const ordinaryResidentWGSL = createWebgpuSparseCM12ResidentWGSL();
assert.doesNotMatch(ordinaryResidentWGSL, /fvr|FVR|censusRow/,
  "ordinary resident hot path must contain no FVR census token or parameter");
const qaResidentWGSL = createWebgpuSparseCM12ResidentWGSL(
  16, 16,
  undefined, undefined, undefined, undefined, undefined, undefined,
  undefined, undefined, undefined, undefined, undefined, undefined,
  undefined, undefined, undefined, undefined, undefined,
  layout,
);
assert.match(qaResidentWGSL,
  /let effective=cm12ExtensionTransportVelocity\(cell\);\s*fvrRecordXyzRead\(censusRow,cell,effective\.xyz\);\s*result\+=wx\*wy\*wz\*effective\.xyz/,
  "actual xyz reads must evaluate the accessor once before recording and accumulation");

// Candidate promotion replaces row 0's donor tile 0 with tile 1. The next
// schedule must be driven by the accepted candidate graph; tile 0 is stale.
const policyHash = sparseCM12VexReverseDependencyPolicyHash(
  SPARSE_CM12_VEX_REVERSE_POLICY);
const topologyHash = [1, 2, 3, 4] as const;
const build = (generation: number, tile: number) => {
  const dependency = buildSparseCM12VexReverseDependency({ cellCount: 3, rowCount: 2,
    policy: SPARSE_CM12_VEX_REVERSE_POLICY,
    journal: { topologyGeneration: generation, topologyHash, policyHash,
      policyEpoch: 0, generation, rows: [
        { row: 0, resultGeneration: generation, dependencyGeneration: generation,
          actualReadsComplete: true, reads: [
            { cell: tile, kind: "interpolation-corner" as const }] },
        { row: 1, resultGeneration: generation, dependencyGeneration: generation,
          actualReadsComplete: true, reads: [
            { cell: 2, kind: "final-interpolation-corner" as const }] },
      ] },
  });
  return summarizeSparseCM12VexDonorTiles({ dependency, tileCount: 3,
    tileForCell: (cell) => cell });
};
const accepted = build(1, 0), candidate = build(2, 1);
assert.deepEqual(sparseCM12VexRowsScheduledByChangedTiles(accepted, [0]), [0]);
assert.deepEqual(sparseCM12VexRowsScheduledByChangedTiles(candidate, [0]), [],
  "stale donor edge survived candidate promotion");
assert.deepEqual(sparseCM12VexRowsScheduledByChangedTiles(candidate, [1]), [0]);
const changedOracleRows = [0];
const predictedFromAcceptedCandidate = new Set(
  sparseCM12VexRowsScheduledByChangedTiles(candidate, [1]));
assert.deepEqual(changedOracleRows.filter((row) =>
  !predictedFromAcceptedCandidate.has(row)), [], "C-minus-S must be empty");

// Mirror the shader's two-bank state transition, not merely VRD construction:
// A remains readable while B collects, clean verify flips to B, and clearing
// the next candidate A retires its stale tile-0 edge without touching B.
const banks = [[[0], [2]], [[], []]] as number[][][];
let acceptedParity = 0;
const candidateParity = () => 1 - acceptedParity;
banks[candidateParity()] = [[], []];
banks[candidateParity()]![0] = [1]; banks[candidateParity()]![1] = [2];
assert.deepEqual(banks[acceptedParity], [[0], [2]], "candidate clear touched accepted A");
acceptedParity = candidateParity();
assert.deepEqual(banks[acceptedParity], [[1], [2]], "clean A-to-B promotion failed");
banks[candidateParity()] = [[], []];
assert.deepEqual(banks[acceptedParity], [[1], [2]], "next clear touched accepted B");
assert.deepEqual(banks[candidateParity()], [[], []], "stale A edges survived retirement");

const snapshotBanks = [[10], [0]];
acceptedParity = 0;
snapshotBanks[candidateParity()]![0] = 11;
const candidateFaulted = true;
if (!candidateFaulted) acceptedParity = candidateParity();
assert.equal(snapshotBanks[acceptedParity]![0], 10,
  "faulted candidate advanced the accepted input snapshot");
const sameChangeRediscovered = snapshotBanks[acceptedParity]![0] !== 11;
assert.equal(sameChangeRediscovered, true,
  "faulted candidate suppressed the same change on the following frame");
snapshotBanks[candidateParity()]![0] = 11;
acceptedParity = candidateParity();
assert.equal(snapshotBanks[acceptedParity]![0], 11,
  "clean graph+snapshot promotion was not atomic");
const donorBitBanks = [new Set([3]), new Set<number>()];
acceptedParity = 0;
donorBitBanks[candidateParity()]!.add(7);
assert.deepEqual([...donorBitBanks[acceptedParity]!], [3],
  "candidate donor capture mutated accepted donor bits before verify");
if (!candidateFaulted) acceptedParity = candidateParity();
assert.deepEqual([...donorBitBanks[acceptedParity]!], [3],
  "faulted candidate advanced accepted donor bits");

// Alternating destination storage is physical restore state, never the
// logical accepted FPA authority used to define C. This fixture makes the two
// disagree: destination parity would invent row 0 and omit the real row 1.
const acceptedPrepared = [10, 20], parityOldDestination = [9, 21];
const fullOraclePrepared = [10, 21], predictedRows = new Set([1]);
const changedFrom = (baseline: readonly number[]) => fullOraclePrepared
  .flatMap((bits, row) => bits === baseline[row] ? [] : [row]);
assert.deepEqual(changedFrom(parityOldDestination), [0],
  "parity mutation control did not expose the false C authority");
assert.deepEqual(changedFrom(acceptedPrepared), [1]);
assert.deepEqual(changedFrom(acceptedPrepared)
  .filter((row) => !predictedRows.has(row)), [],
  "accepted prepared authority produced a false C-minus-S omission");

const root = resolve(import.meta.dirname, "..");
const production = readFileSync(resolve(root,
  "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts"), "utf8");
const sharedWGSL = readFileSync(resolve(root,
  "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts"), "utf8");
const harness = readFileSync(resolve(root,
  "tools/probe-sparse-cm12-temporal-seed-ab.ts"), "utf8");
const integrated = production.includes("FpaVexReadCensusLayout");
assert.match(production,
  /const fpaVexReadCensusLayout = temporalSeedModeForQA === undefined\s*\? undefined/,
  "ordinary construction must omit FVR1 allocation");
assert.match(production,
  /size: Math\.max\(4, fpaVexReadCensusLayout\?\.totalBytes[\s\S]*faceProjectionAuthorityLayout\.totalBytes\)/,
  "ordinary topology arena extent must fall through unchanged");
assert.match(production,
  /const fpaVexReadCensusEntries = fpaVexReadCensusLayout[\s\S]*\? await Promise\.all/,
  "ordinary construction must compile no FVR1 pipelines");
const oracleBegin = production.indexOf('dispatch("beginSparseCM12FpaVexReadCensus"');
const recording = production.indexOf('dispatch("beginSparseCM12FpaVexReadRecording"');
const fullOracle = production.indexOf('dispatchAccepted("prepareTransportFaces", "row")',
  recording);
const restore = production.indexOf('dispatchAccepted("captureSparseCM12FpaOracleAndRestore"');
const productionBegin = production.indexOf('dispatch("beginSparseCM12FacePreparationAuthority"');
const productionFinish = production.indexOf('dispatch("finalizeSparseCM12FacePreparationExecution"');
const verify = production.indexOf('dispatchAccepted("verifySparseCM12FpaOracle"');
const commit = production.indexOf('dispatch("commitSparseCM12FpaVexReadCensus"');
assert(oracleBegin >= 0 && recording > oracleBegin && fullOracle > recording
  && restore > fullOracle && productionBegin > restore && productionFinish > productionBegin
  && verify > productionFinish && commit > verify,
"FVR1 must snapshot/full-record/restore before unchanged production and verify/commit after");
assert.match(sharedWGSL, /const fvrSampleRead = fpaVexReadCensusLayout/,
  "actual-read tracking must remain generator-specialized to the QA resident");
assert.match(sharedWGSL, /override TEMPORAL_CHANGE_SEED_QA:bool=true;/,
  "production exact temporal seed default must remain true");
const readbackMethod = production.slice(production.indexOf("async readFpaVexReadCensusQA"),
  production.indexOf("async readPressureAddressingABQA"));
assert.match(readbackMethod, /const bytes = layout\.compactSummaryBytes;/,
  "FVR1 readback must be compact and bounded");
assert.doesNotMatch(readbackMethod, /layout\.totalWords|rowTilesBaseWords|velocityBitsBaseWords/,
  "FVR1 readback must not copy graph/snapshot storage");
assert(harness.includes("readFpaVexReadCensusQA"));
for (const field of ["rowTileHistogram", "rowTileQuantiles", "tileFanoutHistogram",
  "tileFanoutQuantiles", "changedEffectiveCellCount", "changedEffectiveTileCount",
  "pairHash", "omittedChangedRowCount", "constructionBootstrapPublished",
  "liveRowCount", "compactSummaryBytes", "totalQAArenaBytes"]) {
  assert(harness.includes("fpaVexReadCensus") && censusWGSLSource.includes(field)
      || readFileSync(resolve(root,
        "lib/methods/adaptive-mass/sparse-cm12-fpa-vex-read-census.ts"), "utf8")
        .includes(field), `FVR1 artifact lacks ${field}`);
}
console.log(JSON.stringify({ passed: true,
  contract: "sparse-cm12-fpa-vex-read-census", constructionOnly: true,
  actualRead: "cm12ExtensionTransportVelocity(cell).xyz",
  stableTile: "4^3", maximumTilesPerRow: layout.maximumTilesPerRow,
  candidatePromotionFixture: true, staleEdgeRetirementFixture: true,
  acceptedCandidateBankFixture: true, omittedChangedRows: 0,
  acceptedDonorOnlyCapture: true, nonfaultingValidityClassifier: true,
  faultedCandidateRetainsAcceptedDonorBits: true,
  faultedCandidateRetainsAcceptedSnapshots: true,
  repeatedChangeRediscoveredAfterFault: true,
  fullOracle: "snapshot/full/restore/verify", productionIntegrated: integrated,
  integrationGap: integrated ? [] : [
    "optional resident layout/allocation", "FPA accessor record seam",
    "QA full-oracle encode schedule", "bounded readback and paired harness",
  ],
}, null, 2));
