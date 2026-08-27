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

test("activity swept prediction uses the computed cell center", () => {
  const measurementStart = shader.indexOf("fn measureBrickActivity");
  assert.notEqual(measurementStart, -1);
  const measurement = shader.slice(measurementStart);
  assert.match(measurement, /let center=cellCenter\(cell\);/);
  assert.match(measurement,
    /floor\(\(center\s*\+p\.activityTiming\.x\*p\.frame\.x\*ownVelocity\)/);
  assert.doesNotMatch(measurement, /floor\(\(cellCenter\s*\+/);
});

test("SolidWorld initializes every immutable topology rung", () => {
  const cells = shader.slice(shader.indexOf("fn refreshSparseCM12SolidWorldCells"),
    shader.indexOf("fn refreshSparseCM12SolidWorldRow"));
  const rows = shader.slice(shader.indexOf("fn refreshSparseCM12SolidWorldRows"),
    shader.indexOf("fn candidateFaceBoundaryRowRange"));
  assert.match(cells, /let cell=gid\.x;if\(cell>=ta\(2u\)\)\{return;\}/);
  assert.doesNotMatch(cells, /acceptedTemplateCellInvocation/);
  assert.match(rows, /let row=gid\.x;if\(row>=ta\(3u\)\)\{return;\}/);
  assert.doesNotMatch(rows, /acceptedTemplateRowInvocation/);
  assert.match(resident,
    /refreshSparseCM12SolidWorldCells!\);\s*pass\.dispatchWorkgroups\(Math\.ceil\(this\.templateCellCount/);
  assert.match(resident,
    /refreshSparseCM12SolidWorldRows!\);\s*pass\.dispatchWorkgroups\(Math\.ceil\(this\.templateRowCount/);
});

test("static SolidWorld detail is measured only by the cold refresh", () => {
  const evidence = shader.slice(
    shader.indexOf("fn refreshSparseCM12StaticSolidGeometryEvidence"),
    shader.indexOf("var<workgroup>candidateFaceActive"));
  const measurement = shader.slice(shader.indexOf("fn measureBrickActivity"),
    shader.indexOf("fn planBrickResolution"));
  assert.match(evidence,
    /staticSolidRestrictionError\(origin,max\(1u,BRICK_FINE_RESOLUTION\/2u\),lane\)/);
  assert.match(evidence,
    /staticSolidRestrictionError\(origin,max\(1u,BRICK_FINE_RESOLUTION\/4u\),lane\)/);
  assert.match(evidence,
    /staticSolidRestrictionError\(origin,max\(1u,BRICK_FINE_RESOLUTION\/8u\),lane\)/);
  assert.match(evidence, /staticSolidRestrictionError\(origin,1u,lane\)/);
  assert.match(evidence, /reduced\.x>p\.activityDensity\.w/);
  assert.doesNotMatch(measurement, /cm12SolidVoxelFractionQ8/);
  assert.match(measurement,
    /var reasons=atomicLoad\(&activity\[output\+1u\]\)&0x3c00u/);
});

test("sparse-air metadata requires an actually one-sided accepted row", () => {
  const measurement = shader.slice(shader.indexOf("fn measureBrickActivity"),
    shader.indexOf("fn planBrickResolution"));
  assert.match(measurement, /var hasOpenOpposingTerm=false/);
  assert.match(measurement, /hasOpenOpposingTerm=true/);
  assert.match(measurement,
    /airPort&&!hasOpenOpposingTerm&&ownWet/);
  assert.doesNotMatch(measurement, /var crosses=airPort&&ownWet/);
});
