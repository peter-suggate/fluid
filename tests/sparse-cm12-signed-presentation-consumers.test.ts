import assert from "node:assert/strict";
import test from "node:test";

import { makeFineLevelSetSortedWorklistLookupWGSL } from
  "../lib/core/fine-levelset-brick-abi";
import {
  decodeFluidCellTrace,
  FLUID_CELL_TRACE_FINE_RECORD,
  FLUID_CELL_TRACE_FINE_RECORDS_OFFSET,
  FLUID_CELL_TRACE_HEADER,
  FLUID_CELL_TRACE_MAGIC,
  FLUID_CELL_TRACE_WORDS,
} from "../lib/core/fluid-cell-trace";
import { fluidCellTraceGatherShader } from "../lib/core/webgpu-fluid-cell-trace";
import { octreeTechniqueFineLifecycleShader } from
  "../lib/methods/octree-shared/webgpu-octree-technique-overlay";
import {
  decodeSparseCM12SignedPresentationKey,
  encodeSparseCM12SignedPresentationKey,
  sparseCM12SignedPresentationInitialWorldFits,
} from "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

test("signed sparse presentation keys preserve negative vertical pages", () => {
  const coordinate = [-17, -23, 41] as const;
  assert.deepEqual(decodeSparseCM12SignedPresentationKey(
    encodeSparseCM12SignedPresentationKey(coordinate)), coordinate);
  assert.deepEqual(decodeSparseCM12SignedPresentationKey(
    encodeSparseCM12SignedPresentationKey([0, -512, 0])), [0, -512, 0]);
  assert.deepEqual(decodeSparseCM12SignedPresentationKey(
    encodeSparseCM12SignedPresentationKey([0, 511, 0])), [0, 511, 0]);
  assert.throws(() => encodeSparseCM12SignedPresentationKey([0, -513, 0]),
    /not representable/);
  assert.throws(() => encodeSparseCM12SignedPresentationKey([0, 512, 0]),
    /not representable/);
});

test("signed sparse presentation construction reserves actual pages, not the pool in every direction", () => {
  assert.equal(sparseCM12SignedPresentationInitialWorldFits([24, 12, 4]), true,
    "the long-dam authored page box fits the signed presentation ABI");
  assert.equal(sparseCM12SignedPresentationInitialWorldFits([1, 513, 1]), false,
    "an authored page outside the signed Y envelope is still rejected");
});

test("shared fine-page lookup accepts packed signed SparseWorld keys", () => {
  const wgsl = makeFineLevelSetSortedWorklistLookupWGSL(
    "fine", "metadata", "worklist", "pageOf",
  );
  assert.match(wgsl,
    /if\(\(worklist\[3\]&0x40000000u\)==0u&&key>=logicalCount\)\{return INVALID;\}/,
  "only dense-key publishers may compare a key with the dense logical volume");
  assert.doesNotMatch(wgsl, /\n\s*if\(key>=logicalCount\)\{return INVALID;\}/);
});

test("fluid-cell trace addresses and decodes signed sparse presentation pages", () => {
  assert.match(fluidCellTraceGatherShader,
    /fn fineSignedSparseAddressing\(\)->bool/);
  assert.match(fluidCellTraceGatherShader,
    /let key=finePageKey\(vec3i\(brick\)\)/,
  "probe lookup must not construct the retired dense atlas key directly");
  assert.match(fluidCellTraceGatherShader,
    /brick=vec3i\(i32\(key&0x7ffu\)-1024,i32\(\(key>>11u\)&0x3ffu\)-512,\s*i32\(\(key>>21u\)&0x7ffu\)-1024\)/,
  "seed provenance must decode the same signed key that presentation publishes");
  assert.doesNotMatch(fluidCellTraceGatherShader,
    /let key=brick\.x\+fine\.brickDimensions\.x\*\(brick\.y/);

  const words = new Uint32Array(FLUID_CELL_TRACE_WORDS);
  words[FLUID_CELL_TRACE_HEADER.magic] = FLUID_CELL_TRACE_MAGIC;
  words[FLUID_CELL_TRACE_HEADER.fineRecordCount] = 1;
  const record = FLUID_CELL_TRACE_FINE_RECORDS_OFFSET;
  words[record + FLUID_CELL_TRACE_FINE_RECORD.flags] = 1;
  words[record + FLUID_CELL_TRACE_FINE_RECORD.seedCell] = (-8) >>> 0;
  words[record + FLUID_CELL_TRACE_FINE_RECORD.seedCell + 1] = 3;
  words[record + FLUID_CELL_TRACE_FINE_RECORD.seedCell + 2] = (-16) >>> 0;
  assert.deepEqual(decodeFluidCellTrace(words)?.fineProbeRecords[0]?.seedCell,
    [-8, 3, -16], "signed WDR seed coordinates must survive the QA readback");
});

test("fine-lifecycle overlay uses the publication's signed key mode", () => {
  assert.match(octreeTechniqueFineLifecycleShader,
    /fn finePageKey\(brick:vec3u\)->u32/);
  assert.match(octreeTechniqueFineLifecycleShader,
    /fn finePageCoordinate\(key:u32\)->vec3i/);
  assert.match(octreeTechniqueFineLifecycleShader,
    /let key=finePageKey\(brick\)/);
  assert.match(octreeTechniqueFineLifecycleShader,
    /let brick=finePageCoordinate\(metadata\[id\*4u\+1u\]\)/,
  "flood provenance must decode metadata through the signed-key path");
  assert.doesNotMatch(octreeTechniqueFineLifecycleShader,
    /let key=brick\.x\+fine\.brickDimensions\.x\*\(brick\.y/);
  const floodSample = octreeTechniqueFineLifecycleShader.slice(
    octreeTechniqueFineLifecycleShader.indexOf("fn floodSampleCell"),
    octreeTechniqueFineLifecycleShader.indexOf("fn floodPass"),
  );
  assert.doesNotMatch(floodSample,
    /let bz=key\/xy|let brick=vec3u\(brickRemainder/,
  "flood provenance must not retain a second dense-only metadata decoder");
});
