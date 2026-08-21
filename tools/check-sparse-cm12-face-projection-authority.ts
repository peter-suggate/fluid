#!/usr/bin/env node
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import {
  SPARSE_CM12_FACE_PROJECTION_HEADER,
  SPARSE_CM12_FACE_PROJECTION_INVALID,
  SPARSE_CM12_FACE_PROJECTION_VERSION,
  compareSparseCM12FaceProjectionAuthorityQA,
  createSparseCM12FaceProjectionAuthorityInitialWords,
  createSparseCM12FaceProjectionAuthorityLayout,
} from "../lib/methods/adaptive-mass/sparse-cm12-face-projection-authority";
import { createSparseCM12FaceProjectionAuthorityWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-face-projection-authority.wgsl";

const layout = createSparseCM12FaceProjectionAuthorityLayout({
  rowCapacity: 1025, cellCapacity: 513, baseWords: 256,
});
const initial = createSparseCM12FaceProjectionAuthorityInitialWords(layout);
assert.equal(initial.byteLength, layout.totalBytes);
assert.equal(initial[layout.baseWords + SPARSE_CM12_FACE_PROJECTION_HEADER.totalWords],
  layout.totalWords);
assert(layout.preparation.activeBitsBaseWords >= layout.preparedAuthorityBaseWords + 1025);
assert(layout.preparationCertificateBaseWords
  >= layout.preparedAuthorityBaseWords + 1025);
assert(layout.acceptedPressureBitsBaseWords
  >= layout.preparationCertificateBaseWords + 1025);
assert(layout.projection.activeBitsBaseWords > layout.preparation.treeLevelBaseWords.at(-1)!);

// Exact-authority QA: stable non-zero solenoidal rows are reused, while three
// bit changes and one pressure-only change execute in canonical row order.
const acceptedPreparedFaceBits = Uint32Array.from({ length: 1025 }, (_, row) =>
  (0x3f00_0000 + row) >>> 0);
const acceptedFaceBits = Uint32Array.from(acceptedPreparedFaceBits,
  (bits, row) => (bits ^ (row % 3 === 0 ? 0x0000_1000 : 0x0000_2000)) >>> 0);
const changed = new Set([2, 300, 1024]);
const pressureOnly = 777;
const prepareHead = (row: number) => changed.has(row)
  ? (acceptedPreparedFaceBits[row]! ^ 0x0000_0080) >>> 0
  : acceptedPreparedFaceBits[row]!;
const projectHead = (row: number, prepared: number) => (prepared
  ^ (row === pressureOnly ? 0x0000_0400 : (row % 3 === 0 ? 0x0000_1000 : 0x0000_2000))) >>> 0;
const qa = compareSparseCM12FaceProjectionAuthorityQA({
  preparationLive: new Uint8Array(1025).fill(1),
  projectionLive: new Uint8Array(1025).fill(1),
  preparationDirtyRows: [...changed], projectionDirtyRows: [pressureOnly],
  acceptedFaceBits, acceptedPreparedFaceBits, prepareHead, projectHead,
});
assert.equal(qa.firstMismatchRow, SPARSE_CM12_FACE_PROJECTION_INVALID,
  `FPA1 CPU oracle mismatch at row ${qa.firstMismatchRow}`);
const missing = compareSparseCM12FaceProjectionAuthorityQA({
  preparationLive: new Uint8Array(1025).fill(1),
  projectionLive: new Uint8Array(1025).fill(1),
  preparationDirtyRows: [2, 1024], projectionDirtyRows: [pressureOnly],
  acceptedFaceBits, acceptedPreparedFaceBits, prepareHead, projectHead,
});
assert.equal(missing.firstMismatchRow, 300,
  "FPA1 QA oracle did not expose the first uncovered stable row");

const helpers = createSparseCM12FaceProjectionAuthorityWGSL({
  layout, arenaName: "arena", prefix: "fpa", workgroupSize: 64,
});
const wgsl = /* wgsl */ `
@group(0) @binding(0) var<storage,read_write> arena:array<atomic<u32>>;
fn cm12HotHeaderValid()->bool{return true;}
fn cm12HotRowValid(row:u32)->bool{return row<1025u;}
fn cm12HotIncidenceRange(cell:u32)->vec2u{_=cell;return vec2u(0u);}
fn cm12HotIncidence(index:u32)->vec2u{_=index;return vec2u(0u);}
fn cm12HotRowTermCount(row:u32)->u32{_=row;return 0u;}
fn cm12HotRowTermCell(row:u32,ordinal:u32)->u32{_=row;_=ordinal;return 0xffffffffu;}
fn fpaFrameGeneration()->u32{return 1u;}
fn fpaTopologyGeneration()->u32{return 1u;}
fn fpaPCMGeneration()->u32{return 1u;}
fn fpaSourceParity()->u32{return 1u;}
fn fpaPolicyBits()->u32{return 0x3c23d70au;}
fn fpaExpectedPreparationReceipts()->u32{return 0u;}
fn fpaExpectedProjectionReceipts()->u32{return 0u;}
fn fpaPreparationRowLive(row:u32)->bool{return row<1025u;}
fn fpaProjectionRowLive(row:u32)->bool{return row<1025u&&(row&1u)==0u;}
fn fpaPreparationDependencyGeneration(row:u32)->u32{return 1u+row;}
fn fpaProjectionDependencyGeneration(row:u32)->u32{return 2u+row;}
${helpers}
@compute @workgroup_size(1)
fn exerciseFPA(){_=fpaMarkPreparationRow(2u,1u,0u,false);
  _=fpaMarkProjectionRow(4u,2u,0u,false);_=fpaMarkTopologyCellBlast(0u,32u);
  _=fpaPreparationRowInvocation(0u);_=fpaProjectionRowInvocation(0u);
  fpaStorePreparedAuthority(2u,0x3f800000u);_=fpaPreparedAuthorityBits(2u);
  fpaStorePreparationCertificate(2u,1u);_=fpaPreparationCertificate(2u);
  _=fpaPreparationComplete(2u);_=fpaProjectionComplete(4u);
  _=fpaPreparationMustMirrorUnprojected(3u);_=fpaProjectionMustMirror(4u);}
`;

const directory = mkdtempSync(join(tmpdir(), "sparse-cm12-fpa1-"));
const path = join(directory, "fpa1.wgsl");
try {
  writeFileSync(path, wgsl);
  const checked = spawnSync(process.env.NAGA ?? "naga", [path], { encoding: "utf8" });
  if (checked.error) throw checked.error;
  if (checked.status !== 0) throw new Error(checked.stderr || checked.stdout);
} finally {
  rmSync(directory, { recursive: true, force: true });
}

process.stdout.write(`${JSON.stringify({
  abi: `FPA1/v${SPARSE_CM12_FACE_PROJECTION_VERSION}`, rowCapacity: layout.rowCapacity,
  leafCount: layout.preparation.leafCount,
  totalWords: layout.totalWords,
  cpuOracle: "bit-exact", naga: "valid",
}, null, 2)}\n`);
