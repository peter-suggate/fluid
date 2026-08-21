#!/usr/bin/env node
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import {
  SPARSE_CM12_FACE_PROJECTION_HEADER,
  SPARSE_CM12_FACE_PROJECTION_VERSION,
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
assert(layout.projection.activeBitsBaseWords
  >= layout.acceptedPressureBitsBaseWords + layout.cellCapacity);

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
fn fpaExpectedProjectionReceipts()->u32{return 0u;}
fn fpaProjectionRowLive(row:u32)->bool{return row<1025u&&(row&1u)==0u;}
fn fpaProjectionDependencyGeneration(row:u32)->u32{return 2u+row;}
${helpers}
@compute @workgroup_size(1)
fn exerciseFPA(){_=fpaMarkProjectionRow(4u,2u,0u,false);
  _=fpaMarkProjectionPressureCell(0u,32u,2u);
  _=fpaProjectionRowInvocation(0u);_=fpaProjectionComplete(4u);
  _=fpaProjectionMustMirror(4u);}
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
  leafCount: layout.projection.leafCount,
  totalWords: layout.totalWords,
  naga: "valid",
}, null, 2)}\n`);
