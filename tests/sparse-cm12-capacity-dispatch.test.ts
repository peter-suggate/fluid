import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { planSparseCM12LinearDispatch } from
  "../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident";

const resident = readFileSync(new URL(
  "../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts",
  import.meta.url,
), "utf8");
const shader = readFileSync(new URL(
  "../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts",
  import.meta.url,
), "utf8");

test("authored rerung uses its backed catalogue without allocating duplicate pages", () => {
  assert.doesNotMatch(resident, /["'](?:allocateCandidateTopologyPages|synthesizeCandidateCellPages)["']/);
  assert.doesNotMatch(shader, /fn (?:allocateCandidateTopologyPages|synthesizeCandidateCellPages)\(/);
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
  assert.match(rows, /let group=wid\.x\+nwg\.x\*wid\.y;/);
  assert.match(rows, /let row=64u\*group\+lane;/);
  assert.doesNotMatch(rows, /acceptedTemplateRowInvocation/);
  assert.match(resident,
    /refreshSparseCM12SolidWorldCells!\);\s*pass\.dispatchWorkgroups\(Math\.ceil\(this\.templateCellCount/);
  assert.match(resident,
    /refreshSparseCM12SolidWorldRows!\);\s*pass\.dispatchWorkgroups\(\.\.\.planSparseCM12LinearDispatch\(\s*Math\.ceil\(this\.templateRowCount/);
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
