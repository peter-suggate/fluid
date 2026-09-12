import assert from "node:assert/strict";
import test from "node:test";
import { sliceRdfTriangles, type SliceRdfVertex } from "./slice-rdf-triangulation";

function canonical(triangles: readonly (readonly SliceRdfVertex[])[]): string[] {
  return triangles.map(triangle => triangle.map(vertex => JSON.stringify(vertex)).sort().join(";")).sort();
}

test("RDF tessellation reflects with its field across either centre axis", () => {
  // Unequal corner values expose the diagonal bias hidden by planar fixtures.
  const original = sliceRdfTriangles(0, 0, -3, 1, -0.25, 2);
  const reflectedX = original.map(triangle => triangle.map(([x, y, phi]) =>
    [1 - x, y, phi] as const));
  const reflectedY = original.map(triangle => triangle.map(([x, y, phi]) =>
    [x, 1 - y, phi] as const));
  assert.deepEqual(canonical(sliceRdfTriangles(0, 0, 1, -3, 2, -0.25)), canonical(reflectedX));
  assert.deepEqual(canonical(sliceRdfTriangles(0, 0, 2, -0.25, 1, -3)), canonical(reflectedY));
});

test("RDF centre fan preserves affine scalar fields and covers one cell", () => {
  const triangles = sliceRdfTriangles(2, 3, -2, 1, 3, 0);
  let area = 0;
  for (const triangle of triangles) {
    const [a, b, c] = triangle;
    area += Math.abs((b![0] - a![0]) * (c![1] - a![1])
      - (b![1] - a![1]) * (c![0] - a![0])) / 2;
    for (const [x, y, phi] of triangle)
      assert.equal(phi, -2 + 3 * (x - 2) + 2 * (y - 3));
  }
  assert.equal(area, 1);
});
