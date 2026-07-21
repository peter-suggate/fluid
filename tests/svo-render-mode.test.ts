import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { DEFAULT_SVO_RENDER_MODE, isSvoRenderMode, SVO_RENDER_MODES } from "../lib/svo-render-mode";

test("renderer modes use sparse voxels as the WebGPU default and retain raster as an override", () => {
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
  assert.match(panel, /setVoxelRenderMode\("surface-voxels"\)[^]*>Finest surface<\/button>/);
  assert.match(panel, /aria-label="SVO lighting quality"[^]*setSvoLightingMode\("direct"\)[^]*>Direct<\/button>[^]*setSvoLightingMode\("cone"\)[^]*>Beautiful<\/button>/);
  assert.match(panel, /aria-label="SVO lighting effects"[^]*setSvoShadowsEnabled[^]*>Shadows<\/button>[^]*setSvoAmbientOcclusionEnabled[^]*>Ambient occlusion<\/button>/);
});

test("scene configuration exposes the unified voxel lattice instead of method-level columns", () => {
  const scenePanel = readFileSync(new URL("../components/SceneConfigPopover.tsx", import.meta.url), "utf8");
  const method = readFileSync(new URL("../lib/methods/octree.ts", import.meta.url), "utf8");
  assert.match(scenePanel, /data-testid="voxel-domain-controls"/);
  assert.match(scenePanel, /label="Finest cell"/);
  assert.match(scenePanel, /ariaLabel="Sparse voxel brick size"/);
  assert.match(scenePanel, /disabled: fluidEnabled/);
  assert.doesNotMatch(method, /key: "surfaceColumns"/);
});
