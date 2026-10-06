import assert from "node:assert/strict";
import test from "node:test";
import { summarizeSurfaceDistribution } from "../tools/uniform-surface-work-audit";

test("surface demand model preserves sparse lane tails and spatial specialization", () => {
  const d = [8, 4, 4];
  const data = new Uint32Array(9 * 5 * 5);
  // One four-iteration search in each tile, separated into sampler variants.
  data[1 + 9 * (1 + 5)] = 6;
  data[5 + 9 * (1 + 5)] = 65536 + 6;
  const r = summarizeSurfaceDistribution(data, d);
  assert.equal(r.owned, 2); assert.equal(r.admitted, 2); assert.equal(r.iterations, 8);
  assert.equal(r.current.laneIterations, 256);
  assert.equal(r.spatialCompact[2]!.laneIterations, 256);
  // Same sampling specialization permits a spatial block to pack both.
  data[5 + 9 * (1 + 5)] = 6;
  assert.equal(summarizeSurfaceDistribution(data, d).spatialCompact[2]!.laneIterations, 128);
});

test("surface demand model counts negative closure rounds separately", () => {
  const data = new Uint32Array(125);
  data[1 + 5 * (1 + 5)] = 5; // Interior round, three iterations.
  data[0] = 4; // Negative corner closure, two iterations.
  data[4 + 5 * (4 + 5 * 4)] = 1; // Owned but rejected.
  const r = summarizeSurfaceDistribution(data, [4, 4, 4]);
  assert.equal(r.owned, 3); assert.equal(r.admitted, 2); assert.equal(r.iterations, 5);
  assert.equal(r.current.activeGroups, 2);
  assert.equal(r.current.laneIterations, 160);
  assert.equal(r.spatialCompact[1]!.laneIterations, 96);
});
