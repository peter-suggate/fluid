#!/usr/bin/env node
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";

import {
  SPARSE_CM12_VEX_REVERSE_DEPENDENCY_HEADER as H,
  SPARSE_CM12_VEX_REVERSE_DEPENDENCY_MAGIC,
  SPARSE_CM12_VEX_REVERSE_DEPENDENCY_SOURCE_MANIFEST,
  SPARSE_CM12_VEX_REVERSE_POLICY,
  advanceSparseCM12VexReverseDependency,
  buildSparseCM12VexReverseDependency,
  createSparseCM12VexDependencyJournalInitialWords,
  createSparseCM12VexDependencyJournalLayout,
  sparseCM12VexReverseDependencyPolicyHash,
  sparseCM12VexReverseRows,
  sparseCM12VexRowsScheduledByChangedTiles,
  sparseCM12VexRowsScheduledByChangedCells,
  summarizeSparseCM12VexDonorTiles,
  type SparseCM12VexReverseDependencyPolicy,
} from "../lib/methods/adaptive-mass/sparse-cm12-vex-reverse-dependency";
import { createSparseCM12VexDependencyJournalWGSL,
  createSparseCM12VexReverseDependencyLookupWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-vex-reverse-dependency.wgsl";

const policy: SparseCM12VexReverseDependencyPolicy = {
  ...SPARSE_CM12_VEX_REVERSE_POLICY,
};
assert(!("maximumCflFineCells" in policy),
  "accepted actual-read dependency must not masquerade as a geometric CFL radius");
const policyHash = sparseCM12VexReverseDependencyPolicyHash(policy);
const topologyHash = [0x10203040, 0x50607080, 0x90a0b0c0, 0xd0e0f000] as const;
const journal = {
  topologyGeneration: 9, topologyHash, policyHash,
  policyEpoch: 3,
  generation: 1,
  // Deliberately shuffled with duplicate forward reads. Output must be stable.
  rows: [
    { row: 3, resultGeneration: 1, dependencyGeneration: 1,
      actualReadsComplete: true, reads: [
        { cell: 1, kind: "probe-owner-span" },
        { cell: 4, kind: "interpolation-corner" },
        { cell: 1, kind: "probe-owner-span" }] },
    { row: 0, resultGeneration: 1, dependencyGeneration: 1,
      actualReadsComplete: true, reads: [
        { cell: 2, kind: "incident-validity" },
        { cell: 1, kind: "final-interpolation-corner" }] },
    { row: 2, resultGeneration: 1, dependencyGeneration: 1,
      actualReadsComplete: true, reads: [] },
    { row: 1, resultGeneration: 1, dependencyGeneration: 1,
      actualReadsComplete: true, reads: [
        { cell: 4, kind: "interpolation-corner" },
        { cell: 2, kind: "probe-owner-span" }] },
  ],
} as const;
const dependency = buildSparseCM12VexReverseDependency({
  baseWords: 7, cellCount: 6, rowCount: 4, edgeCapacity: 12, policy, journal,
});
assert.equal(dependency.layout.baseWords % 64, 0);
assert.equal(dependency.words[H.magic], SPARSE_CM12_VEX_REVERSE_DEPENDENCY_MAGIC);
assert.equal(dependency.layout.edgeCount, 6);
assert.equal(dependency.layout.rowBytes, 24);
assert.equal(dependency.layout.unusedCapacityBytes, 24);
assert.equal(dependency.layout.maximumRowsPerCell, 2);
assert.deepEqual(sparseCM12VexReverseRows(dependency, 0), []);
assert.deepEqual(sparseCM12VexReverseRows(dependency, 1), [0, 3]);
assert.deepEqual(sparseCM12VexReverseRows(dependency, 2), [0, 1]);
assert.deepEqual(sparseCM12VexReverseRows(dependency, 4), [1, 3]);

// Safe donor-tile coarsening excludes incident validity and topology-probe
// reads. Cells 1/2 share tile 0 and cell 4 occupies tile 1.
const donorTiles = summarizeSparseCM12VexDonorTiles({ dependency, tileCount: 3,
  tileForCell: (cell) => Math.floor(cell / 3) });
assert.deepEqual(donorTiles.rowTiles, [[0], [1], [], [1]]);
assert.deepEqual(donorTiles.tileRows, [[0], [1, 3], []]);
assert.equal(donorTiles.edgeCount, 3);
assert.deepEqual(donorTiles.rowTileCount,
  { p50: 1, p90: 1, p99: 1, maximum: 1 });
assert.deepEqual(donorTiles.tileSubscriberCount,
  { p50: 1, p90: 1, p99: 1, maximum: 2 });
assert.deepEqual(sparseCM12VexRowsScheduledByChangedTiles(donorTiles, [0]), [0]);
assert.deepEqual(sparseCM12VexRowsScheduledByChangedTiles(donorTiles, [1]), [1, 3]);
assert.throws(() => sparseCM12VexRowsScheduledByChangedTiles(donorTiles, [3]),
  /exceeds capacity/);

const reordered = buildSparseCM12VexReverseDependency({
  cellCount: 6, rowCount: 4, policy,
  journal: { ...journal, rows: [...journal.rows].reverse() },
});
for (let cell = 0; cell < 6; cell += 1) {
  assert.deepEqual(sparseCM12VexReverseRows(reordered, cell),
    sparseCM12VexReverseRows(dependency, cell));
}
assert.deepEqual(reordered.contentHash, dependency.contentHash,
  "content provenance must ignore certificate input order and duplicate mentions");
assert.throws(() => buildSparseCM12VexReverseDependency({
  cellCount: 6, rowCount: 4, policy: { ...policy, maximumSubsteps: 8 } as never,
  journal,
}), /unsupported trace policy/);
assert.throws(() => buildSparseCM12VexReverseDependency({
  cellCount: 6, rowCount: 4, policy,
  journal: { ...journal, policyHash: [0, 0, 0, 0] },
}), /policy hash mismatch/);
assert.throws(() => buildSparseCM12VexReverseDependency({
  cellCount: 6, rowCount: 4, policy,
  journal: { ...journal, rows: journal.rows.slice(1) },
}), /does not cover every stable row/);
assert.throws(() => buildSparseCM12VexReverseDependency({
  cellCount: 6, rowCount: 4, edgeCapacity: 5, policy, journal,
}), /capacity overflow/);

// Redirect fixture: accepted row 0 read A. A changes, so row 0 executes and its
// paired candidate journal redirects to B. On the following frame B alone must
// schedule row 0; A must no longer do so. Unexecuted rows retain prior reads.
const redirectPolicyHash = sparseCM12VexReverseDependencyPolicyHash(policy);
const redirectAccepted = buildSparseCM12VexReverseDependency({
  cellCount: 3, rowCount: 2, edgeCapacity: 4, policy,
  journal: { topologyGeneration: 9, topologyHash, policyHash: redirectPolicyHash,
    policyEpoch: 3,
    generation: 7, rows: [
      { row: 0, resultGeneration: 7, dependencyGeneration: 7,
        actualReadsComplete: true, reads: [
          { cell: 0, kind: "interpolation-corner" }] },
      { row: 1, resultGeneration: 7, dependencyGeneration: 7,
        actualReadsComplete: true, reads: [
          { cell: 2, kind: "incident-validity" }] },
    ] },
});
assert.deepEqual(sparseCM12VexRowsScheduledByChangedCells(redirectAccepted, [0]), [0]);
assert.deepEqual(sparseCM12VexRowsScheduledByChangedCells(redirectAccepted, [1]), []);
const redirectCandidate = advanceSparseCM12VexReverseDependency({
  accepted: redirectAccepted, candidateGeneration: 8, policy,
  executedRows: [{ row: 0, resultGeneration: 8, dependencyGeneration: 8,
    actualReadsComplete: true, reads: [
      { cell: 1, kind: "final-interpolation-corner" }] }],
});
assert.deepEqual(sparseCM12VexRowsScheduledByChangedCells(redirectCandidate, [0]), []);
assert.deepEqual(sparseCM12VexRowsScheduledByChangedCells(redirectCandidate, [1]), [0]);
assert.deepEqual(sparseCM12VexRowsScheduledByChangedCells(redirectCandidate, [2]), [1],
  "unexecuted row must inherit accepted reads");
assert.throws(() => advanceSparseCM12VexReverseDependency({
  accepted: redirectAccepted, candidateGeneration: 8, policy,
  executedRows: [{ row: 0, resultGeneration: 8, dependencyGeneration: 7,
    actualReadsComplete: true, reads: [] }],
}), /generation pair is incomplete/);
assert.throws(() => advanceSparseCM12VexReverseDependency({
  accepted: redirectAccepted, candidateGeneration: 8, candidatePolicyEpoch: 4, policy,
  executedRows: [{ row: 0, resultGeneration: 8, dependencyGeneration: 8,
    actualReadsComplete: true, reads: [] }],
}), /policy epoch change requires every affected row journal/);

const journalLayout = createSparseCM12VexDependencyJournalLayout({
  rowCapacity: 4, cellCapacity: 6, chunkCapacity: 8,
});
assert.equal(journalLayout.maximumRawReads, 256);
assert.equal(journalLayout.totalBytes % 256, 0);
const initialJournal = createSparseCM12VexDependencyJournalInitialWords(journalLayout);
assert.equal(initialJournal[0], 1);

assert.equal(SPARSE_CM12_VEX_REVERSE_DEPENDENCY_SOURCE_MANIFEST.runtimeMutationRequired,
  true);
assert.equal(SPARSE_CM12_VEX_REVERSE_DEPENDENCY_SOURCE_MANIFEST.productionIntegrated,
  false);
assert(SPARSE_CM12_VEX_REVERSE_DEPENDENCY_SOURCE_MANIFEST.liveMissing.includes(
  "GPU canonical candidate-journal to accepted-index replacement"));
assert(SPARSE_CM12_VEX_REVERSE_DEPENDENCY_SOURCE_MANIFEST.invariants.includes(
  "no runtime accepted-row scan"));
const library = createSparseCM12VexReverseDependencyLookupWGSL({ dependency });
const journalLibrary = createSparseCM12VexDependencyJournalWGSL({
  layout: journalLayout,
});
assert.match(library, /var<storage,read>vexReverseDependency/);
assert.match(library, /fn fpeaVexReverseProvenanceValid/);
assert.match(library, /fn fpeaVexReverseBegin/);
assert.match(library, /fn fpeaVexReverseEnd/);
assert.match(library, /fn fpeaVexReverseRow/);
assert.doesNotMatch(library, /atomic|storage,read_write|acceptedTemplateRowInvocation|fallback/i);
assert.match(journalLibrary, /fn vrdjRecord/);
assert.match(journalLibrary, /fn vrdjSealRow/);
assert.match(journalLibrary, /fn vrdjAllocateChunk/);
assert.doesNotMatch(journalLibrary, /acceptedTemplateRowInvocation|fallback/i);

const source = `${library}\nfn vrdjExpectedExecutedRows()->u32{return 1u;}\n${journalLibrary}
@compute @workgroup_size(1)
fn lookupFixture(){
  if(!fpeaVexReverseProvenanceValid()){return;}
  let begin=fpeaVexReverseBegin(1u);let end=fpeaVexReverseEnd(1u);
  if(begin!=0xffffffffu&&begin<end){_=fpeaVexReverseRow(begin);}
  var cursor=vrdjBeginRow(0u,1u);_=vrdjRecord(&cursor,1u,2u);
  _=vrdjSealRow(&cursor,1u);
}`;
const directory = mkdtempSync(join(tmpdir(), "fluid-vrd1-"));
try {
  const path = join(directory, "vrd1.wgsl"); writeFileSync(path, source);
  const result = spawnSync(process.env.NAGA ?? "naga", [path], { encoding: "utf8" });
  if (result.status !== 0) throw new Error(result.stderr || result.stdout);
} finally {
  rmSync(directory, { recursive: true, force: true });
}

console.log(JSON.stringify({ passed: true,
  contract: "sparse-cm12-vex-reverse-dependency",
  policy, edgeCount: dependency.layout.edgeCount,
  maximumRowsPerCell: dependency.layout.maximumRowsPerCell,
  memory: { totalBytes: dependency.layout.totalBytes,
    offsetBytes: dependency.layout.offsetBytes, rowBytes: dependency.layout.rowBytes,
    unusedCapacityBytes: dependency.layout.unusedCapacityBytes,
    journalBytes: journalLayout.totalBytes,
    maximumRawReads: journalLayout.maximumRawReads },
  stableRows: Array.from({ length: dependency.layout.cellCount }, (_, cell) =>
    sparseCM12VexReverseRows(dependency, cell)),
  donorTileSummary: {
    edgeCount: donorTiles.edgeCount,
    rowTileCount: donorTiles.rowTileCount,
    tileSubscriberCount: donorTiles.tileSubscriberCount,
    canonicalPairHash: donorTiles.canonicalPairHash,
  },
  redirectFixture: { retiredAEdge: true, addedBEdge: true,
    bOnlyNextFrameSchedulesRow: true },
  naga: "PASS", runtimeScan: false, fallback: false,
}, null, 2));
