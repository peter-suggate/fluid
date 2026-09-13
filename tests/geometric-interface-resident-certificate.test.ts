import assert from "node:assert/strict";
import test from "node:test";

import { geometricInterfaceResidentWGSL } from
  "../lib/methods/adaptive-volume/geometric-interface-resident.wgsl";

const tolerance = 9.5367431640625e-7;
const certifiedFill = (fill: number): number | undefined =>
  !(fill >= -tolerance && fill <= 1 + tolerance)
    ? undefined : Math.min(1, Math.max(0, fill));

test("interface certification accepts and locally clamps transport roundoff", () => {
  assert.equal(certifiedFill(-1.862645149230957e-9), 0);
  assert.equal(certifiedFill(1 + 2 ** -22), 1);
  assert.equal(certifiedFill(-2 * tolerance), undefined);
  assert.equal(certifiedFill(1 + 2 * tolerance), undefined);
  assert.equal(certifiedFill(Number.NaN), undefined);
  assert.equal(certifiedFill(Number.POSITIVE_INFINITY), undefined);
  assert.equal(certifiedFill(Number.NEGATIVE_INFINITY), undefined);
});

test("uniform 2-D and 3-D reconstruction share the certified fill observation", () => {
  assert.match(geometricInterfaceResidentWGSL,
    /fn geometricResidentCertifiedFill[\s\S]*!\(fill >= -tolerance && fill <= 1\.0\+tolerance\)[\s\S]*clamp\(fill,0\.0,1\.0\)/);
  assert.match(geometricInterfaceResidentWGSL,
    /fn geometricResidentFitUniformExtrusion[\s\S]*geometricResidentCertifiedFill\(other,densityOffset\)/);
  assert.match(geometricInterfaceResidentWGSL,
    /fn geometricResidentUniformSample[\s\S]*return geometricResidentCertifiedFill\(cell,densityOffset\)/);
});

test("ELVIRA candidates require orientation evidence on their integration axis", () => {
  assert.match(geometricInterfaceResidentWGSL,
    /fn geometricResidentIntegrationSupported[\s\S]*normal\[axis\]!=0\.0/);
  assert.match(geometricInterfaceResidentWGSL,
    /for\(var direction=0u;direction<2u[\s\S]*geometricResidentIntegrationSupported\(normal,integration\)/);
  assert.match(geometricInterfaceResidentWGSL,
    /for\(var integration=0u;integration<3u[\s\S]*geometricResidentIntegrationSupported\(fallback\.normal,integration\)/);
});

test("all PLIC neighbour consumers use the sealed physical-face CSR", () => {
  assert.match(geometricInterfaceResidentWGSL,
    /fn geometricResidentAdaptiveExtrusionCertified[\s\S]*cnxCellFaceRangeUnchecked\(cell\)[\s\S]*cnxPhysicalFaceRowUnchecked\(face\)[\s\S]*geometricResidentSameProjection/);
  assert.match(geometricInterfaceResidentWGSL,
    /fn geometricResidentProjectedGradient[\s\S]*cnxCellFaceRangeUnchecked\(cell\)[\s\S]*cnxPhysicalFaceCellsUnchecked\(face\)[\s\S]*geometricResidentSameProjection/);
  assert.match(geometricInterfaceResidentWGSL,
    /fn geometricResidentReconstructInterface[\s\S]*if\(!cnxTransportViewValidForAcceptedTopology\(\)\)\{return result;\}[\s\S]*let faces=cnxCellFaceRangeUnchecked\(cell\)[\s\S]*cnxCellFaceEntryUnchecked\(adjacency\)[\s\S]*cnxPhysicalFaceAreaUnchecked\(face\)/);
  assert.match(geometricInterfaceResidentWGSL,
    /fn geometricResidentSupportedInterface[\s\S]*if\(!cnxTransportViewValidForAcceptedTopology\(\)\)\{return result;\}[\s\S]*let faces=cnxCellFaceRangeUnchecked\(cell\)[\s\S]*cnxPhysicalFaceCellsUnchecked\(face\)[\s\S]*cnxPhysicalFaceRowUnchecked\(face\)/);
  assert.doesNotMatch(geometricInterfaceResidentWGSL,
    /geometricResidentOverlap|incidenceBegin|incidenceEnd|incidenceRow|incidenceTerm|rowTermRange|termCoefficient|termCell/);
});

type CompiledFace = Readonly<{
  row: number;
  face: number;
  projection: string | undefined;
}>;

/** Mirrors the row/face restart used by the compiled adaptive PLIC consumers. */
const projectionLeaders = (entries: readonly CompiledFace[]): number[] => {
  const leaders: number[] = [];
  let rowFirst = 0;
  let previousRow = -1;
  let previousFace = -1;
  for (let at = 0; at < entries.length; at += 1) {
    const entry = entries[at]!;
    if (entry.row !== previousRow || (previousFace >= 0 && entry.face <= previousFace)) {
      rowFirst = at;
    }
    previousRow = entry.row;
    previousFace = entry.face;
    if (entry.projection === undefined) continue;
    const earlier = entries.slice(rowFirst, at).some(candidate =>
      candidate.row === entry.row && candidate.projection === entry.projection);
    if (!earlier) leaders.push(at);
  }
  return leaders;
};

test("compiled PLIC grouping preserves mixed, boundary, and repeated-incidence order", () => {
  const entries: CompiledFace[] = [
    { row: 7, face: 10, projection: "coarse-port" },
    { row: 7, face: 11, projection: "coarse-port" },
    { row: 7, face: 12, projection: "side-port" },
    { row: 9, face: 20, projection: undefined },
    // A duplicate incidence restarts when its ascending face sequence repeats.
    { row: 7, face: 10, projection: "coarse-port" },
    { row: 7, face: 11, projection: "coarse-port" },
  ];
  assert.deepEqual(projectionLeaders(entries), [0, 2, 4]);
});
