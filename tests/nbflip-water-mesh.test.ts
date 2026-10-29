import assert from "node:assert/strict";
import test from "node:test";
import { waterMeshObj } from "../tools/blender/nbflip-water-mesh";

test("offline water export preserves metres, orientation, winding and ignores non-position lanes", () => {
  const vertices = new Float32Array([
    1, 2, 3, 99, 0, 1, 0, 99,
    2, 2, 3, 99, 0, 1, 0, 99,
    1, 2, 2, 99, 0, 1, 0, 99,
  ]);
  const result = waterMeshObj(vertices, "frame");
  assert.deepEqual(result.bounds, { min: [1, -3, 2], max: [2, -2, 2] });
  const positions = result.obj.split("\n").filter(line => line.startsWith("v ")).map(line => line.slice(2).split(" ").map(Number));
  assert.deepEqual(positions, [[1, -3, 2], [2, -3, 2], [1, -2, 2]]);
  const [a, b, c] = positions;
  const normalZ = (b![0]! - a![0]!) * (c![1]! - a![1]!) - (b![1]! - a![1]!) * (c![0]! - a![0]!);
  assert.equal(normalZ, 1, "upward application triangle must still face upward in Blender");
  assert.match(result.obj, /f 1 2 3\n$/);
});

test("offline water export refuses missing, truncated or non-finite geometry", () => {
  assert.throws(() => waterMeshObj(new Float32Array(), "empty"), /complete triangles/);
  assert.throws(() => waterMeshObj(new Float32Array(16), "partial"), /complete triangles/);
  const bad = new Float32Array(24); bad[9] = NaN;
  assert.throws(() => waterMeshObj(bad, "broken"), /non-finite/);
});

test("optical OBJ normals preserve the production lane independently of geometric face normals", () => {
  const vertices = new Float32Array([
    1, 2, 3, 99, 1, 2, 0, 7,
    2, 2, 3, 99, 0, 2, 1, 7,
    1, 2, 2, 99, 0, 3, 0, 7,
  ]);
  const legacy = waterMeshObj(vertices, "frame"), optical = waterMeshObj(vertices, "frame", { opticalNormals: true });
  assert.deepEqual(optical.bounds, legacy.bounds);
  const lines = (obj: string, prefix: string) => obj.split("\n").filter(line => line.startsWith(prefix));
  assert.deepEqual(lines(optical.obj, "v "), lines(legacy.obj, "v "), "optics must not change triangle positions");
  const normals = lines(optical.obj, "vn ").map(line => line.slice(3).split(" ").map(Number));
  assert.deepEqual(normals, [[1 / Math.sqrt(5), 0, 2 / Math.sqrt(5)], [0, -1 / Math.sqrt(5), 2 / Math.sqrt(5)], [0, 0, 1]]);
  normals.forEach(normal => assert.ok(Math.abs(Math.hypot(...normal) - 1) < 1e-14));
  assert.match(optical.obj, /f 1\/\/1 2\/\/2 3\/\/3\n$/);
  assert.ok(normals[0]![0]! > 0, "preserve tilted phi gradient rather than recomputing the flat face normal");
  const invalid = vertices.slice(); invalid[4] = NaN;
  assert.throws(() => waterMeshObj(invalid, "invalid", { opticalNormals: true }), /invalid optical normals/);
  assert.doesNotThrow(() => waterMeshObj(invalid, "legacy"), "legacy exporter must continue ignoring non-position lanes");
  invalid[4] = invalid[5] = invalid[6] = 0;
  assert.throws(() => waterMeshObj(invalid, "zero", { opticalNormals: true }), /invalid optical normals/);
});
