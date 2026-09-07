import assert from "node:assert/strict";
import test from "node:test";
import { resolveSvoPrimaryTraversal, SVO_PRIMARY_TRAVERSAL_OVERRIDE } from "../../../pipeline/svo-render-options";

test("raster requests never silently select tracing at any scene scale", () => {
  assert.equal(SVO_PRIMARY_TRAVERSAL_OVERRIDE, undefined, "Run the product-policy test without a diagnostic override");
  for (const scale of [{}, { environmentRefinementDepth: 3, leafBricks: 227137, targetPixels: 368000 },
    { environmentRefinementDepth: 6, leafBricks: 1000000, targetPixels: 4000000 }]) {
    assert.equal(resolveSvoPrimaryTraversal("raster", scale), "mesh");
    assert.equal(resolveSvoPrimaryTraversal("mesh", scale), "mesh");
    assert.equal(resolveSvoPrimaryTraversal("traced", scale), "traced", "Explicit reference tracing remains selectable");
  }
});
