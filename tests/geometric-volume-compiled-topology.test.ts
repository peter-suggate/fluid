import assert from "node:assert/strict";
import test from "node:test";

import { createGeometricVolumeResidentWGSL, WHOLE_FRAME_VOLUME_CONTROL,
  type SparseGeometricVolumeLayout } from
  "../lib/methods/adaptive-volume/resident-volume.wgsl";

const layout = Object.fromEntries([
  "currentVolume", "lowVolume", "positiveLimiter", "negativeLimiter",
  "rowSubfaceRanges", "cellSubfaceRanges", "cellSubfaceEntries",
  "subfaceMetadata", "subfaceFluxes", "subfaceRoundoff",
  "supportControlBaseWords", "controlBaseWords", "subfaceCapacity",
  "airDiagonal", "airControlBaseWords", "airComponentBaseWords",
  "transportEdgeCapacity", "transportEdgeMetadata", "transportEdgeWeightsA",
  "transportEdgeWeightsB", "wholeFrameControlBaseWords",
  "transportReceiverHeadsBaseWords", "transportDonorHeadsBaseWords",
].map((key, value) => [key, value])) as unknown as SparseGeometricVolumeLayout;

const shader = createGeometricVolumeResidentWGSL(layout);

test("physical faces and cell CSR are compiled only for a full CNX generation", () => {
  assert.match(shader,
    /fn beginGeometricVolumeTopologyCompilation\(\)[\s\S]*if\(!cnxBuilding\(\)\)\{return;\}[\s\S]*cnxPhysicalFaceCountBuildingUnchecked\(\)/);
  assert.match(shader,
    /fn compileGeometricVolumeSubfaces[\s\S]*acceptedTemplateRowInvocation\(cnxLinearInvocation\(gid\)\)[\s\S]*cnxAllocatePhysicalFaces\(count,row\)/);
  assert.match(shader,
    /fn compileGeometricVolumeCellFaces[\s\S]*acceptedTemplateCellInvocation\(cnxLinearInvocation\(gid\)\)[\s\S]*cnxAllocatePhysicalFaceEntries\(count,cell\)/);
  assert.match(shader,
    /fn publishGeometricVolumeTopology\(\)\{cnxPublishTransportView\(\);\}/);
  const build = shader.slice(shader.indexOf("fn beginGeometricVolumeTopologyCompilation"),
    shader.indexOf("fn gvAuditMaterialCoverage"));
  assert.doesNotMatch(build, /atomicAdd\(&conditioning\[GV_CONTROL\]/);
  assert.doesNotMatch(build, /atomicAdd\(&conditioning\[GV_SUPPORT\+32u\]/);
});

test("each whole-frame transport retains the sealed physical graph", () => {
  const begin = shader.slice(shader.indexOf("fn beginWholeFrameVolumeTransport"),
    shader.indexOf("fn gvWriteFace"));
  assert.match(begin, /cnxTransportViewValidForAcceptedTopology\(\)/);
  assert.match(begin, /gvStore\(0u,cnxPhysicalFaceCount\(\)\)/);
  assert.doesNotMatch(begin, /GV_SUPPORT\+32u/);
  assert.match(shader, /fn gvCellFaceRange\(cell:u32\)->vec2u\{return cnxCellFaceRangeUnchecked\(cell\);\}/);
  assert.match(shader, /fn gvArea\(face:u32\)->f32\{return cnxPhysicalFaceAreaUnchecked\(face\);\}/);
});

test("whole-frame transport uses sparse receiver/donor edges and no FCT microsteps", () => {
  assert.match(shader, /fn buildWholeFrameVolumeCoupling[\s\S]*ownerCellAt\(vec3i\(x,y,z\)\)/);
  assert.match(shader, /fn gvAppendCouplingEdge[\s\S]*GV_RECEIVER_HEADS[\s\S]*GV_DONOR_HEADS/);
  assert.match(shader, /fn normalizeWholeFrameVolumeRowsAtoB/);
  assert.match(shader, /fn normalizeWholeFrameVolumeDonorsBtoA/);
  assert.match(shader, /fn auditWholeFrameVolumeMarginals/);
  assert.equal(WHOLE_FRAME_VOLUME_CONTROL.maximumTraceDisplacement, 18);
  assert.equal(WHOLE_FRAME_VOLUME_CONTROL.sharpeningCutCellSkipCount, 19);
  assert.equal(WHOLE_FRAME_VOLUME_CONTROL.sharpeningBlockedFaceSkipCount, 20);
  assert.equal(WHOLE_FRAME_VOLUME_CONTROL.sharpeningDisconnectedFaceSkipCount, 21);
  assert.match(shader,
    /GV_WHOLE_FRAME_CONTROL\+18u\][\s\S]*length\(traced-centre\)/);
  assert.doesNotMatch(shader, /gvMicroActive|advanceGeometricVolumeSubstep|GeometricFCT/);
});

test("whole-frame sharpening stays inside open metric phi connectivity", () => {
  assert.match(shader, /capacity<cellVolume\(cell\)-gvRoundoff/,
    "cut cells retain transported V without a spatial solid/liquid intersection");
  assert.match(shader, /rowOpenFraction\(row\)<1\.0-9\.5367431640625e-7/,
    "blocked or spatially unresolved cut rows cannot carry sharpening volume");
  assert.match(shader, /lsvSampleAt\(gvSharpeningFaceCentre\(face,cells\)\)/,
    "a metric face-band sample gates every sharpening transfer");
});

test("transport frontier reads compiled row endpoints without term discovery", () => {
  const projected = shader.slice(shader.indexOf("fn markProjectedGeometricTransportReceivers"),
    shader.indexOf("fn activateProjectedGeometricTransportReceivers"));
  assert.match(projected,
    /cnxPhysicalFaceRangeUnchecked\(row\)[\s\S]*cnxPhysicalFaceCellsUnchecked\(face\)/);
  assert.doesNotMatch(projected, /rowTermRange|termCell|termCoefficient/);

  const velocity = shader.slice(shader.indexOf("fn gatherGeometricTransportVelocityBounds"),
    shader.indexOf("fn gatherGeometricPreflightVelocityBounds"));
  assert.match(velocity,
    /cnxPhysicalFaceRangeUnchecked\(row\)[\s\S]*cnxPhysicalFaceCellsUnchecked\(face\)/);
  assert.doesNotMatch(velocity, /rowTermRange|termCell|termCoefficient/);

  const termHasMaterial = (terms: readonly number[], material: ReadonlySet<number>) =>
    terms.some(cell => material.has(cell));
  const faceHasMaterial = (faces: readonly (readonly [number, number | undefined])[],
    material: ReadonlySet<number>) => faces.some(([negative, positive]) =>
      material.has(negative) || (positive !== undefined && material.has(positive)));
  const terms = [4, 7, 8];
  const mixedFaces = [[4, 7], [4, 8]] as const;
  for (const material of [new Set([4]), new Set([7]), new Set([8]), new Set<number>()]) {
    assert.equal(faceHasMaterial(mixedFaces, material), termHasMaterial(terms, material));
  }
  assert.equal(faceHasMaterial([[4, undefined]], new Set([4])), true,
    "a compiled one-sided boundary face retains its sole material endpoint");
});

test("the full build refuses an omitted accepted geometric neighbour", () => {
  assert.match(shader,
    /fn gvExcludedRowHasGeometricNeighbor\(row:u32,ownTerm:u32\)->bool[\s\S]*own\*other>=0\.0[\s\S]*geometricSubface\(row,negative,positive\)\.status!=0u/);
  assert.match(shader,
    /if\(!gvAcceptedPhysicalRow\(row\)\)\{[\s\S]*rowAccepted\(row\)&&gvExcludedRowHasGeometricNeighbor\([\s\S]*gvTopologyBuildMalformed\(1u,row/);

  const excludedRequiresFault = (coefficients: readonly number[], own: number,
    geometricStatus: (own: number, other: number) => 0 | 1 | 2) =>
    coefficients.some((coefficient, other) =>
      coefficients[own]! * coefficient < 0 && geometricStatus(own, other) !== 0);
  assert.equal(excludedRequiresFault([-1], 0, () => 1), false,
    "a superseded one-sided boundary row has no PLIC neighbour to cover");
  assert.equal(excludedRequiresFault([-1, 0.25, 0.25, 0.25, 0.25], 0,
    (_own, other) => other === 2 ? 0 : 1), true,
  "an omitted mixed row with any physical opposite-side patch must refuse seal");
  assert.equal(excludedRequiresFault([-1, 1], 0, () => 2), true,
    "malformed opposite-side geometry must also refuse seal");
});
