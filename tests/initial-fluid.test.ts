import assert from "node:assert/strict";
import test from "node:test";
import { damBreakFractions } from "../lib/initial-fluid";

test("dam-break corner reservoir preserves requested volume", () => {
  for (const fill of [0.05, 0.22, 0.5, 0.9, 1]) {
    const dam = damBreakFractions(fill);
    assert.equal(dam.width, dam.depth);
    assert.ok(dam.width <= 1 && dam.height <= 1 && dam.depth <= 1);
    assert.ok(Math.abs(dam.width * dam.height * dam.depth - fill) < 1e-12);
  }
});
