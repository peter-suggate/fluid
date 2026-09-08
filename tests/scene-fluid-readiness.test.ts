import assert from "node:assert/strict";
import test from "node:test";
import { enableWaterLockReason } from "../lib/core/scene-fluid-readiness";

test("water setup waits for dry raster publication even when the SVO status is idle", () => {
  assert.match(enableWaterLockReason(null, { state: "idle" })!, /finish loading/);
  assert.ok(enableWaterLockReason({ initialRasterSurfaceReady: false }, { state: "ready" }));
});

test("a retained ready raster cannot enable water while replacement SVO work is preparing", () => {
  assert.ok(enableWaterLockReason({ initialRasterSurfaceReady: true }, { state: "preparing" }));
  assert.equal(enableWaterLockReason({ initialRasterSurfaceReady: true }, { state: "ready" }), undefined);
});
