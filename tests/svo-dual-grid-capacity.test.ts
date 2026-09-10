import assert from "node:assert/strict";
import test from "node:test";
import { assertDualGridAttachmentFits } from "../lib/svo/features/meshing/dual-grid-capacity";

test("dual attachment capacity check rejects impossible grids without enumerating the remainder", () => {
  let visits = 0;
  function* occupied() { for (let i = 0; i < 1_000_000; i++) { visits++; yield i; } }
  assert.throws(() => assertDualGridAttachmentFits(occupied(), 8, 8208 * 100), /Uniform dual-grid construction/);
  assert.equal(visits, 101);
  assert.throws(() => assertDualGridAttachmentFits([0, 1], 8, 8208 * 2, 512), /Uniform dual-grid/);
  assert.throws(() => assertDualGridAttachmentFits([0, 1], 8, 8208 * 2, 0, 1), /Uniform dual-grid/);
  assert.doesNotThrow(() => assertDualGridAttachmentFits([0, 1], 8, 8208 * 2));
  assert.throws(() => assertDualGridAttachmentFits([0, 1], 8, 8208 * 2 - 1), /Uniform dual-grid/);
});
