import assert from "node:assert/strict";
import test from "node:test";

import { sparseBrickPayloadProfileForOwnership } from
  "../lib/svo/sparse-brick-octree";

test("a fluid scene's renderer-only sidecar does not allocate solver lanes", () => {
  assert.equal(sparseBrickPayloadProfileForOwnership(true, true, "dry"), "dry");
  assert.equal(sparseBrickPayloadProfileForOwnership(true, false, "dry"), "full",
    "a solver-owned fluid octree must retain its mutable lanes");
  assert.equal(sparseBrickPayloadProfileForOwnership(false, false, "dry"), "dry");
  assert.equal(sparseBrickPayloadProfileForOwnership(true, true, "full"), "full",
    "the existing rollback lever must still cover sidecars");
});
