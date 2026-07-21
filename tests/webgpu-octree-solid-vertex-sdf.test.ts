import assert from "node:assert/strict";
import test from "node:test";
import {
  materializeOctreeTerrainVertexSdf,
  octreeSolidVertexSdfShader,
  planOctreeSolidVertexSdf,
} from "../lib/webgpu-octree-solid-vertex-sdf";

test("solid vertex-SDF storage scales with compact owner rows", () => {
  const plan = planOctreeSolidVertexSdf(37);
  assert.equal(plan.sampleCapacity, 37 * 8);
  assert.equal(plan.sdfBytes, 37 * 8 * 4);
  assert.equal(plan.allocatedBytes, 37 * 8 * 4 + 64 + 64);
  assert.throws(() => planOctreeSolidVertexSdf(0), /positive integer/);
});

test("terrain SDF is materialized at the eight actual vertices of each sparse owner", () => {
  const heights = new Float32Array(4 * 4).fill(1);
  const values = materializeOctreeTerrainVertexSdf([
    { cell: 0, size: 2 },
    { cell: 2 + 4 * 2 + 16 * 2, size: 2 },
  ], heights, [4, 4, 4], [0.5, 0.5, 0.5]);
  assert.equal(values.length, 16, "no finest-box vertex lattice is materialized");
  assert.deepEqual(Array.from(values.slice(0, 8)), [-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5]);
  assert.deepEqual(Array.from(values.slice(8, 16)), [0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5]);
});

test("cell-centred terrain heights are bilinearly sampled at owner vertices", () => {
  const values = materializeOctreeTerrainVertexSdf(
    [{ cell: 0, size: 2 }],
    new Float32Array([0, 2, 0, 2]),
    [2, 2, 2],
    [1, 1, 1],
  );
  // x=0 clamps to the first cell centre; x=2 clamps to the second. The four
  // upper corners differ only by owner y, never by an invented square face.
  assert.deepEqual(Array.from(values), [0, -2, 2, 0, 0, -2, 2, 0]);
});

test("terrain vertex publication rejects malformed sparse owners", () => {
  assert.throws(() => materializeOctreeTerrainVertexSdf(
    [{ cell: 1, size: 2 }], new Float32Array(16), [4, 4, 4], [1, 1, 1],
  ), /canonical octree owner/);
  assert.throws(() => materializeOctreeTerrainVertexSdf(
    [{ cell: 0, size: 2 }], new Float32Array([Number.NaN, 0, 0, 0]), [2, 2, 2], [1, 1, 1],
  ), /non-finite/);
});

test("GPU publication is generation-tagged and fail-closed", () => {
  assert.match(octreeSolidVertexSdfShader, /solid vertex SDF|publishSolidVertexSdf/i);
  assert.match(octreeSolidVertexSdfShader, /atomicStore\(&arena\.control\[4\],params\.publication\.x\)/);
  assert.match(octreeSolidVertexSdfShader, /atomicLoad\(&arena\.control\[3\]\)==count\*8u/);
  assert.match(octreeSolidVertexSdfShader, /atomicStore\(&arena\.control\[5\],VALID\)/);
  assert.match(octreeSolidVertexSdfShader, /rollbackSeedControl\[5\]==params\.publication\.x/);
  assert.doesNotMatch(octreeSolidVertexSdfShader, /dims\.x\*params\.dims\.y\*params\.dims\.z\*8u/,
    "storage must remain compact-row-scaled");
});
