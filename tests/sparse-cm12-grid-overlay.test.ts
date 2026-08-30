import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { gridOverlayShader } from "../lib/core/webgpu-grid-overlay";

const resident = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts", import.meta.url,
), "utf8");

test("Sparse CM12 publishes the brick resolution consumed by visual overlays", () => {
  const writeParameters = resident.slice(
    resident.indexOf("private writeParameters("),
    resident.indexOf("private writeDynamicParameters("),
  );
  assert.ok(writeParameters.length > 0,
    "the sparse parameter publication must remain identifiable");
  assert.match(writeParameters,
    /u\.set\(\[\.\.\.this\.dimensions,\s*this\.brickFineResolution\s*<<\s*1\],\s*4\)/,
  "dimensions.w must retain the brick-resolution consumer ABI");
  assert.doesNotMatch(writeParameters,
    /u\.set\(\[\.\.\.this\.dimensions,\s*0\],\s*4\)/,
  "retiring an unrelated flag must not erase the brick resolution");
  assert.match(gridOverlayShader,
    /fn sparseBrickFineResolution\(\)->u32\{return max\(1u,sparseP\.dimensions\.w>>1u\);\}/);
});

test("the grid overlay consumes the producer's live activity-record stride", () => {
  assert.match(resident, /const ACTIVITY_RECORD_WORDS = 42/,
    "the surface-proof receipt extends the activity record to 42 words");
  assert.match(resident,
    /activityRecordWords:\s*ACTIVITY_RECORD_WORDS/,
    "the sparse consumer source must publish its record ABI");
  assert.match(gridOverlayShader,
    /fn sparseActivityRecordWords\(\)->u32\{return sparseOverlayP\.worldDirectory\.w;\}/);
  assert.match(gridOverlayShader,
    /SPARSE_ACTIVITY_HEADER_WORDS\+sparseActivityRecordWords\(\)\*brick\+10u/);
  assert.doesNotMatch(gridOverlayShader, /SPARSE_ACTIVITY_RECORD_WORDS:u32=/,
    "renderer ownership must not freeze a second copy of the producer stride");
});

test("the structure overlay omits non-liquid SparseWorld support halos", () => {
  assert.match(gridOverlayShader,
    /fn sparseBrickOccupied\(brick:u32\)[\s\S]*sparseActivity\[at\]&64u/);
  assert.match(gridOverlayShader,
    /sparseStructureHalo=sparseGridEnabled\(\)&&fieldMode==0[\s\S]*!sparseBrickOccupied/);
  assert.match(gridOverlayShader,
    /if\(sparseStructureHalo\)\{fill=vec3f\(0\.0\);alpha=0\.0;line=0\.0;sampleDot=0\.0;\}/);
});

test("the grid overlay follows SparseWorld's signed ownership directory", () => {
  assert.match(resident,
    /worldDirectoryBaseWords:\s*this\.worldDirectoryLayout\.baseWords/,
  "the consumer source must locate WDR1 inside the shared topology arena");
  assert.match(resident,
    /worldDirectoryInitialLeaves:\s*this\.worldDirectoryLayout\.initialLeaves/,
  "the consumer source must identify WDR1's dynamic-leaf boundary");
  assert.match(gridOverlayShader, /const SPARSE_WORLD_DIRECTORY_MAGIC:u32=0x57445231u/);
  assert.match(gridOverlayShader,
    /if\(directory\)\{brick=sparseWorldOwnerAt\(querySigned\);\}/,
  "represented-cell lookup must use the signed world authority");
  assert.match(gridOverlayShader,
    /if\(directory\)\{brickCoordinate=sparseWorldLeafCoordinate\(brick\);\}/,
  "cell placement must use the signed leaf coordinate, not a dense key decode");
  assert.match(gridOverlayShader,
    /minimumFine=sparseBounds\.minimum;maximumFine=sparseBounds\.maximum/,
  "slice and volume bounds must follow the live sparse-world extent");
  assert.doesNotMatch(gridOverlayShader,
    /if\(!sparseGridEnabled\(\)\|\|any\(q<vec3i\(0\)\)/,
  "signed sparse owners must not be clipped to the authored lattice");
});

test("the grid overlay addresses synthesized frontier-leaf cell pages", () => {
  assert.match(gridOverlayShader,
    /brick>=sparseOverlayP\.worldDirectory\.z/,
  "dynamic WDR1 leaves must not index the immutable atlas range table");
  assert.match(gridOverlayShader,
    /let first=sparseTopologyArena\[2u\]\+page\*count/,
  "frontier cells must use the resident's page-local dynamic cell tail");
  assert.match(gridOverlayShader,
    /if\(resolution!=brickFine\)\{return vec2u\(0u\);\}/,
  "a synthesized frontier leaf is representable only at its fixed B rung");
});

test("signed-distance visualization uses compact signed fine-page keys", () => {
  assert.match(gridOverlayShader,
    /fn sparseSignedFineAddressing\(\)->bool\{return \(sparseFineWorklist\[3\]&0x40000000u\)!=0u;\}/);
  assert.match(gridOverlayShader,
    /return u32\(page\.x\+1024\)\|\(u32\(page\.y\+512\)<<11u\)\|\(u32\(page\.z\+1024\)<<21u\);/,
  "the overlay key must match the resident's signed fine-page ABI");
  assert.match(gridOverlayShader,
    /let pageCoordinate=sparseFloorDiv\(q,i32\(pageResolution\)\);/,
  "negative fine coordinates require floor division rather than truncation");
});
