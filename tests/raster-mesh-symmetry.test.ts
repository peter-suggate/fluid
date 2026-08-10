import assert from "node:assert/strict";
import test from "node:test";
import { rasterCubeSymmetryMetrics, rasterMeshSymmetryMetrics,
  sharpPatchRasterMetrics } from "../tools/raster-mesh-symmetry";

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

test("raster cube audit reflects the optical ghost lattice about D+1", () => {
  const dimensions = [4, 2, 4] as const;
  const orbit = (seed: readonly [number, number, number, number]) => {
    const records = new Map<string, [number, number, number, number]>([[seed.join(":"), [...seed]]]);
    for (let changed = true; changed;) {
      changed = false;
      for (const [x, y, z, span] of [...records.values()]) for (const record of [
        [dimensions[0] + 1 - x - span, y, z, span],
        [x, y, dimensions[2] + 1 - z - span, span],
        [z, y, x, span],
      ] as [number, number, number, number][]) if (!records.has(record.join(":"))) {
        records.set(record.join(":"), record); changed = true;
      }
    }
    return [...records.values()];
  };
  const records = [...orbit([0, 0, 1, 1]), ...orbit([0, 1, 1, 2])];
  const cubes = new Uint32Array(records.flatMap(([x, y, z, span]) => [
    x | z << 16, y | span << 16,
  ]));
  const result = rasterCubeSymmetryMetrics(cubes, records.length, dimensions);
  assert.equal(result.transforms["reflect-x"].exactIdentityMismatchCount, 0);
  assert.equal(result.transforms["reflect-z"].exactIdentityMismatchCount, 0);
  assert.equal(result.transforms["swap-xz"].exactIdentityMismatchCount, 0);
});

test("raster cube audit reflects zero-span adaptive descriptors about native D", () => {
  const dimensions = [4, 2, 4] as const;
  const bases = [0, 1, 2, 3].flatMap(x => [0, 1, 2, 3].map(z => [x, z] as const));
  const cubes = new Uint32Array(bases.flatMap(([x, z]) => [x | z << 16, 0]));
  const result = rasterCubeSymmetryMetrics(cubes, bases.length, dimensions);
  assert.equal(result.transforms["reflect-x"].exactIdentityMismatchCount, 0);
  assert.equal(result.transforms["reflect-z"].exactIdentityMismatchCount, 0);
  assert.equal(result.transforms["swap-xz"].exactIdentityMismatchCount, 0);
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
