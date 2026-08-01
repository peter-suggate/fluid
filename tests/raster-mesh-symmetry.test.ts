import assert from "node:assert/strict";
import test from "node:test";
import { rasterMeshSymmetryMetrics, sharpPatchRasterMetrics } from "../tools/raster-mesh-symmetry";

function vertex(position: readonly [number, number, number], normal: readonly [number, number, number]) {
  return [...position, 1, ...normal, 0];
}

test("raster mesh audit accepts an exact D4 vertex set independent of order and duplicates", () => {
  const records = [
    vertex([1, 2, 3], [1, 0, 0]), vertex([-1, 2, 3], [-1, 0, 0]),
    vertex([1, 2, -3], [1, 0, 0]), vertex([-1, 2, -3], [-1, 0, 0]),
    vertex([3, 2, 1], [0, 0, 1]), vertex([-3, 2, 1], [0, 0, 1]),
    vertex([3, 2, -1], [0, 0, -1]), vertex([-3, 2, -1], [0, 0, -1]),
  ];
  records.push(records[0], records[0]);
  const packed = new Float32Array(records.flat().reverse().reverse());
  const result = rasterMeshSymmetryMetrics(packed, records.length);
  assert.equal(result.exactMismatchCount, 0);
  assert.equal(result.exactPositionMismatchCount, 0);
  assert.equal(result.nonFiniteCount, 0);
});

test("raster mesh audit detects one-sided geometry and non-finite normals", () => {
  const packed = new Float32Array(vertex([1, 2, 3], [Number.NaN, 0, 0]));
  const result = rasterMeshSymmetryMetrics(packed, 1);
  assert.ok(result.exactMismatchCount > 0);
  assert.ok(result.exactPositionMismatchCount > 0);
  assert.equal(result.nonFiniteCount, 1);
});

test("raster mesh audit distinguishes a closed triangle complex from a missing face", () => {
  const points = [
    [1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1],
  ] as const;
  const faces = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]] as const;
  const mesh = (selected: readonly (readonly [number, number, number])[]) => new Float32Array(
    selected.flatMap((face) => face.flatMap((index) => vertex(points[index], [0, 1, 0]))),
  );
  const closed = rasterMeshSymmetryMetrics(mesh(faces), faces.length * 3);
  assert.equal(closed.degenerateTriangleCount, 0);
  assert.equal(closed.openEdgeCount, 0);
  assert.equal(closed.nonManifoldEdgeCount, 0);
  const openFaces = faces.slice(1);
  const open = rasterMeshSymmetryMetrics(mesh(openFaces), openFaces.length * 3);
  assert.equal(open.openEdgeCount, 3);
});

test("sharp patch audit rejects the former multi-tetra allocation", () => {
  const rectangle = [
    vertex([0.5, 0, 0], [1, 0, 0]), vertex([0.5, 1, 0], [1, 0, 0]),
    vertex([0.5, 1, 1], [1, 0, 0]), vertex([0.5, 0, 0], [1, 0, 0]),
    vertex([0.5, 1, 1], [1, 0, 0]), vertex([0.5, 0, 1], [1, 0, 0]),
  ];
  const vertices = new Float32Array(rectangle.flat());
  const cubes = new Uint32Array([0, (1 | (6 << 8)) << 16]);
  const direct = sharpPatchRasterMetrics(vertices, 6, cubes, new Uint32Array([0, 6, 6, 6, 6, 6]), 1);
  assert.deepEqual(direct, { patchCount: 1, invalidPatchCount: 0 });
  const squeezed = sharpPatchRasterMetrics(vertices, 6, cubes, new Uint32Array([0, 3, 6, 6, 6, 6]), 1);
  assert.equal(squeezed.patchCount, 1);
  assert.equal(squeezed.invalidPatchCount, 1);
  assert.match(squeezed.firstInvalidPatch?.reason ?? "", /six-vertex rectangle/);
});
