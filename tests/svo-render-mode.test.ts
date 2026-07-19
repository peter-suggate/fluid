import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { DEFAULT_SVO_RENDER_MODE, isSvoRenderMode, SVO_RENDER_MODES } from "../lib/svo-render-mode";

test("SVO production renderer modes have a stable raster fallback", () => {
  assert.equal(DEFAULT_SVO_RENDER_MODE, "svo");
  assert.deepEqual(SVO_RENDER_MODES, ["raster", "svo"]);
  assert.equal(isSvoRenderMode("raster"), true);
  assert.equal(isSvoRenderMode("svo"), true);
  for (const invalid of [undefined, null, "smooth", "ray-marched", 1]) assert.equal(isSvoRenderMode(invalid), false);
});

test("visual controls clearly separate production renderer choice from debug representation", () => {
  const panel = readFileSync(new URL("../components/VisualPanel.tsx", import.meta.url), "utf8");
  assert.match(panel, /aria-label="Scene renderer"/);
  assert.match(panel, /setSvoRenderMode\("raster"\)[^]*>Raster<\/button>/);
  assert.match(panel, /setSvoRenderMode\("svo"\)[^]*>Sparse voxels<\/button>/);
  assert.match(panel, /setVoxelRenderMode\("raw-voxels"\)[^]*>Raw voxels<\/button>/);
});
