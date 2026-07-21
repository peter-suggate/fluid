import assert from "node:assert/strict";
import test from "node:test";
import { octreePowerCatalogWGSL } from "../lib/octree-power-wgsl";

test("shared power WGSL exposes the exact 32-byte face ABI and catalog reconstruction", () => {
  assert.match(octreePowerCatalogWGSL, /struct PowerFaceRecord/);
  assert.match(octreePowerCatalogWGSL, /geometryCode:u32/);
  assert.match(octreePowerCatalogWGSL, /normalVelocity:f32/);
  assert.match(octreePowerCatalogWGSL, /fn inversePowerTransform/);
  assert.match(octreePowerCatalogWGSL, /fn reconstructPowerCatalogFace/);
  assert.doesNotMatch(octreePowerCatalogWGSL, /axisSpan|faceAxis/);
});
