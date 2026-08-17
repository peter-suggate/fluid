#!/usr/bin/env node
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_CAUSE,
  SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER,
  SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_INVALID,
  createSparseCM12PressureTopologyRepairInitialWords,
  createSparseCM12PressureTopologyRepairLayout,
  evaluateSparseCM12PressureTopologyRepairQA,
  sparseCM12PressureTopologyRepairIndirectByteOffset,
} from "../lib/methods/adaptive-mass/sparse-cm12-pressure-topology-repair";
import { createSparseCM12PressureTopologyRepairWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-pressure-topology-repair.wgsl";

const layout = createSparseCM12PressureTopologyRepairLayout({
  brickCapacity: 8, rowCapacity: 96, baseWords: 128,
  brickFineResolution: 16, presentationPageResolution: 16,
});
const initial = createSparseCM12PressureTopologyRepairInitialWords(layout);
assert.equal(initial.byteLength, layout.totalBytes);
assert.equal(initial[layout.baseWords + SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER.totalWords],
  layout.totalWords);
assert.equal(sparseCM12PressureTopologyRepairIndirectByteOffset(layout, "brick", "repair"),
  4 * (layout.brick.headerBaseWords + 10));
assert(layout.row.candidateGenerationBaseWords
  > layout.brick.treeLevelBaseWords.at(-1)!);
assert(layout.brickOldStateBaseWords > layout.row.treeLevelBaseWords.at(-1)!);

// Exact two-brick dynamic fixture: brick 1 changes B16->B8 and brick 6 is
// created at B16. Records arrive out of order and one producer repeats an
// identical record. Cell incidences intentionally overlap heavily.
const OLD_B16 = 0x8000_0010;
const NEW_B8 = 0x8000_0008;
const NEW_B16 = 0x8000_1010;
const incidences = Array.from({ length: 16 }, (_, cell) =>
  Object.freeze([cell % 7, 20 + (cell % 5), cell % 7]));
const receipt = evaluateSparseCM12PressureTopologyRepairQA({
  brickCapacity: 8,
  rowCapacity: 96,
  records: [
    { brick: 6, oldState: SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_INVALID,
      newState: NEW_B16, cause: SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_CAUSE.topologyCreated },
    { brick: 1, oldState: OLD_B16, newState: NEW_B8,
      cause: SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_CAUSE.resolutionChanged },
    { brick: 6, oldState: SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_INVALID,
      newState: NEW_B16, cause: SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_CAUSE.identityChanged },
  ],
  cellRange: (brick, state) => {
    if (brick === 1 && state === OLD_B16) return [0, 6];
    if (brick === 1 && state === NEW_B8) return [6, 4];
    if (brick === 6 && state === NEW_B16) return [10, 6];
    throw new Error(`unexpected fixture state ${brick}:${state}`);
  },
  incidences,
  membership: (cell, current) => current ? (cell & 1) === 0 : cell % 3 === 0,
});
assert.deepEqual(receipt.changedBricks, [1, 6]);
assert.deepEqual(receipt.retiredCells, [0, 1, 2, 3, 4, 5]);
assert.deepEqual(receipt.currentCells, [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
assert.deepEqual(receipt.dirtyRows, [0, 1, 2, 3, 4, 5, 6, 20, 21, 22, 23, 24]);
assert.deepEqual(receipt.pcfTopologyCells,
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
assert.deepEqual(receipt.pcfMembershipCells, [2, 3, 4, 8, 9, 10, 14, 15]);
assert.throws(() => evaluateSparseCM12PressureTopologyRepairQA({
  brickCapacity: 8, rowCapacity: 96,
  records: [
    { brick: 1, oldState: OLD_B16, newState: NEW_B8, cause: 1 },
    { brick: 1, oldState: OLD_B16, newState: NEW_B16, cause: 2 },
  ],
  cellRange: () => [0, 0], incidences, membership: () => false,
}), /conflicting changed-brick provenance/);

const helpers = createSparseCM12PressureTopologyRepairWGSL({
  layout, arenaName: "arena", prefix: "ptr", workgroupSize: 64,
});
assert.match(helpers, /@workgroup_size\(64\) fn repairSparseCM12PressureTopologyChangedBricks/);
assert.match(helpers, /for\(var local=lane;local<oldRange\.y;local\+=64u\)/);
assert.match(helpers, /fn importSparseCM12PressureTopologyChangedBricks/);
assert.doesNotMatch(helpers, /for\(var brick=0u;/,
  "PTR1 must never walk brick capacity from a singleton finalizer");

const wgsl = /* wgsl */ `
@group(0) @binding(0) var<storage,read_write> arena:array<atomic<u32>>;
fn ptrFrameGeneration()->u32{return 2u;}
fn ptrTopologyGeneration()->u32{return 7u;}
fn ptrPCMCellCandidateGeneration()->u32{return 3u;}
fn ptrPCMRowCandidateGeneration()->u32{return 4u;}
fn ptrPCFCandidateGeneration()->u32{return 5u;}
fn ptrPCMCellAcceptedGeneration()->u32{return 3u;}
fn ptrPCMRowAcceptedGeneration()->u32{return 4u;}
fn ptrPCFAcceptedGeneration()->u32{return 5u;}
fn ptrExpectedProducerReceipts()->u32{return ptrTopologyJournalCount();}
fn ptrTopologyJournalCount()->u32{return 2u;}
fn ptrTopologyJournalRecord(at:u32)->vec4u{
 return select(vec4u(1u,${OLD_B16}u,${NEW_B8}u,4u),
  vec4u(6u,0xffffffffu,${NEW_B16}u,1u),at==1u);}
fn ptrCellCapacity()->u32{return 16u;}
fn ptrBrickCellRange(brick:u32,state:u32)->vec2u{
 if(brick==1u&&state==${OLD_B16}u){return vec2u(0u,6u);}
 if(brick==1u&&state==${NEW_B8}u){return vec2u(6u,4u);}
 if(brick==6u&&state==${NEW_B16}u){return vec2u(10u,6u);}
 return vec2u(0u);}
fn ptrCellIncidenceRange(cell:u32)->vec2u{return vec2u(cell,cell+1u);}
fn ptrIncidenceRow(at:u32)->u32{return at%12u;}
fn ptrApplyPressureCellClassification(cell:u32,current:bool)->bool{
 return current&&(cell&1u)==0u;}
fn ptrClassifyPressureRow(row:u32)->bool{return (row&1u)==0u;}
fn ptrPressureTheta(row:u32)->f32{return f32(row)*0.01;}
fn pcmCellContains(cell:u32)->bool{return cell%3u==0u;}
fn pcmCellSetCandidate(cell:u32,enabled:bool,cause:u32,direct:bool)->bool{
 _=cell;_=enabled;_=cause;_=direct;return true;}
fn pcmRowSetCandidate(row:u32,enabled:bool,cause:u32,direct:bool)->bool{
 _=row;_=enabled;_=cause;_=direct;return true;}
fn pcfRecordCellMembershipEvent(cell:u32)->bool{_=cell;return true;}
fn pcfRecordTopologyCellEvent(cell:u32)->bool{_=cell;return true;}
fn pcfStoreThetaAndRecord(row:u32,theta:f32)->bool{_=row;_=theta;return true;}
${helpers}
`;

const directory = mkdtempSync(join(tmpdir(), "sparse-cm12-ptr1-"));
try {
  const path = join(directory, "ptr1-b16-p16.wgsl");
  writeFileSync(path, wgsl);
  const checked = spawnSync(process.env.NAGA ?? "naga", [path], { encoding: "utf8" });
  if (checked.error) throw checked.error;
  if (checked.status !== 0) throw new Error(checked.stderr || checked.stdout);
} finally {
  rmSync(directory, { recursive: true, force: true });
}

process.stdout.write(`${JSON.stringify({
  abi: "PTR1/v1", fixture: "exact-two-brick-B16-to-B8-plus-B16-create",
  changedBricks: receipt.changedBricks, retiredCells: receipt.retiredCells.length,
  currentCells: receipt.currentCells.length, deduplicatedRows: receipt.dirtyRows,
  leafBits: 256, brickWorkgroupLanes: 64, stableRankSelect: true,
  compactFinalizer: true, pcfEvents: receipt.pcfTopologyCells.length,
  totalWords: layout.totalWords, naga: "valid",
}, null, 2)}\n`);
