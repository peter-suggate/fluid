import assert from "node:assert/strict";
import test from "node:test";
import { createUIStore } from "../lib/core/stores/ui-store";
import { lightingQuery } from "../lib/svo/features/lighting-visibility/persistence";
import { DEFAULT_SVO_LIGHTING_OPTIONS } from "../lib/svo/pipeline/svo-render-options";
import { SVO_DRY_VISIBILITY_FLAGS } from "../lib/svo/features/shading/program";

test("voxel surface image detail defaults on and roundtrips independently of lighting visibility", () => {
  assert.equal(DEFAULT_SVO_LIGHTING_OPTIONS.surfaceTexturesEnabled, true);
  const ui = createUIStore(), before = ui.getState();
  assert.equal(before.svoSurfaceTexturesEnabled, true);
  before.setSvoSurfaceTexturesEnabled(false);
  assert.equal(ui.getState().svoSurfaceTexturesEnabled, false);
  assert.equal(ui.getState().svoPrimaryTraversal, before.svoPrimaryTraversal);
  assert.equal(ui.getState().svoConeTracingMode, before.svoConeTracingMode);
  assert.equal(lightingQuery.read(new URLSearchParams()).svoSurfaceTexturesEnabled, true);
  for (const mode of ["raster-ao", "cones", "exact", "off"]) for (const enabled of [false, true]) {
    const query = new URLSearchParams(`svoCones=${mode}`), state = lightingQuery.read(query);
    state.svoSurfaceTexturesEnabled = enabled;
    lightingQuery.write(query, state);
    const restored = lightingQuery.read(query);
    assert.equal(restored.svoSurfaceTexturesEnabled, enabled);
    assert.equal(restored.svoConeTracingMode, mode);
  }
  const bits = Object.values(SVO_DRY_VISIBILITY_FLAGS);
  assert.equal(new Set(bits).size, bits.length, "material detail must not alias a visibility flag");
});
