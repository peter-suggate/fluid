import assert from "node:assert/strict";
import test from "node:test";

import { createSliceRetirementAuthority, publishSliceRetirementAuthority } from "./slice-retirement-authority";
import { compileSliceTopology } from "./slice-topology";

test("post-topology retirement marks exact changed and retired leaf slots", () => {
  const before = compileSliceTopology([{ id: 4, key: 4, coordinate: [0, 0],
    resolution: 1, density: new Float32Array([0]) }], [8, 8]);
  const after = compileSliceTopology([{ id: 4, key: 4, coordinate: [0, 0],
    resolution: 1, active: false }], [8, 8], 2);
  const authority = publishSliceRetirementAuthority(createSliceRetirementAuthority(before),
    before, after, new Float32Array([0]));
  assert.deepEqual(Array.from(authority.receipt.topologyChangedBrickIds), [4]);
  assert.deepEqual(Array.from(authority.receipt.retiredBrickIds), [4]);
  assert.equal(authority.receipt.retiredResidueMassFineCells, 0);
  assert.equal(authority.receipt.pendingDynamicReleaseIds.length, 0);
});

