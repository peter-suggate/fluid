import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const resident = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
  import.meta.url,
), "utf8");
const shader = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
  import.meta.url,
), "utf8");

test("candidate rerung synthesis excludes reserved dynamic capacity", () => {
  assert.match(resident,
    /dispatch\("synthesizeCandidateCellPages", this\.worldDirectoryLayout\.initialLeaves\)/);
  assert.doesNotMatch(resident,
    /dispatch\("synthesizeCandidateCellPages", leafCapacity\)/);
  assert.doesNotMatch(resident,
    /dispatchTopology\("synthesizeCandidateCellPages", leafCapacity\)/);
});

test("SparseWorld pages store only mutable or hot topology records", () => {
  assert.doesNotMatch(resident, /GPU_TOPOLOGY_CELL_RECORD_WORDS/);
  assert.doesNotMatch(resident,
    /GPU_TOPOLOGY_CELL_PAGE_HEADER_WORDS\s*\n\s*\+ brickFineResolution \*\* 3 \* 8/);
  const synthesis = shader.slice(shader.indexOf("fn synthesizeSparseWorldFrontierPages"),
    shader.indexOf("fn connectSparseWorldFrontierPages"));
  assert.match(synthesis, /let rowBase=16u;/);
  assert.match(synthesis, /pageBase\+6u\],0u/);
  assert.doesNotMatch(synthesis, /pageBase\+cellBase\+8u\*local/);
  assert.match(synthesis, /let termBase=rowBase\+7u\*faceCount/);
  assert.doesNotMatch(synthesis, /rowBase\+[78]u\*faceCount\+row/);
  assert.match(synthesis,
    /let incidenceRecords=termBase\+4u\*faceCount/);
  assert.doesNotMatch(synthesis, /incidenceOffsets\+local/);
  assert.doesNotMatch(synthesis, /2u\*\(6u\*local\+side\)/);
  assert.match(synthesis, /dynamicIncidenceOverrideAt\(pageBase,local,side\)/);
  assert.match(synthesis, /pageBase\+termBase\+4u\*row/);
});
