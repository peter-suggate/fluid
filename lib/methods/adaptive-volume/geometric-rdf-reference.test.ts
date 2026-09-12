import assert from "node:assert/strict";
import test from "node:test";
import { rdfCellCentreValue, rdfPlaneBoxCentroid,
  type RdfPlaneSample3, type RdfVec3 } from "./geometric-rdf-reference";

const normalize = (value: RdfVec3): RdfVec3 => {
  const length = Math.hypot(...value);
  return [value[0] / length, value[1] / length, value[2] / length];
};

test("PLIC box centroid lies on an oblique interface and respects box symmetry", () => {
  const normal = normalize([2, -1, 3]);
  const sample: RdfPlaneSample3 = { center: [5, 7, 11], widths: [2, 4, 6],
    normal, offset: 0.35 };
  const centroid = rdfPlaneBoxCentroid(sample);
  const relative = centroid.map((value, axis) => value - sample.center[axis]!) as
    unknown as RdfVec3;
  assert.ok(Math.abs(normal[0] * relative[0] + normal[1] * relative[1]
    + normal[2] * relative[2] - sample.offset) < 1e-12);
  for (let axis = 0; axis < 3; axis += 1)
    assert.ok(Math.abs(relative[axis]) <= sample.widths[axis]! / 2 + 1e-12);
});

test("orientation-weighted RDF exactly reproduces one global plane", () => {
  const normal = normalize([2, 1, -3]);
  const anchor: RdfVec3 = [4.2, 3.7, 5.1];
  const globalOffset = normal[0] * anchor[0] + normal[1] * anchor[1]
    + normal[2] * anchor[2];
  const sources: RdfPlaneSample3[] = [];
  for (let z = 2; z <= 6; z += 1) for (let y = 2; y <= 6; y += 1)
    for (let x = 2; x <= 6; x += 1) {
      const center: RdfVec3 = [x + 0.5, y + 0.5, z + 0.5];
      const offset = globalOffset - normal[0] * center[0] - normal[1] * center[1]
        - normal[2] * center[2];
      if (Math.abs(offset) <= (Math.abs(normal[0]) + Math.abs(normal[1])
        + Math.abs(normal[2])) / 2)
        sources.push({ center, widths: [1, 1, 1], normal, offset });
    }
  for (const target of [[3.5, 3.5, 3.5], [4.5, 4.5, 4.5],
    [5.5, 3.5, 4.5]] as const) {
    const value = rdfCellCentreValue(target, sources);
    const expected = normal[0] * target[0] + normal[1] * target[1]
      + normal[2] * target[2] - globalOffset;
    assert.ok(value !== null && Math.abs(value - expected) < 1e-12);
  }
});

test("opposed nearby planes retain a signed thin region instead of cancelling normals", () => {
  const left: RdfPlaneSample3 = { center: [0, 0, 0], widths: [1, 1, 1],
    normal: [-1, 0, 0], offset: -0.25 };
  const right: RdfPlaneSample3 = { center: [1, 0, 0], widths: [1, 1, 1],
    normal: [1, 0, 0], offset: -0.25 };
  const inside = rdfCellCentreValue([0.5, 0, 0], [left, right]);
  assert.ok(inside !== null);
  assert.ok(inside < 0, "the RDF averages signed distances, not opposing normals");
});
