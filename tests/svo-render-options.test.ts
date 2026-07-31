import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { DEFAULT_SVO_LIGHTING_OPTIONS } from "../lib/svo-render-options";

test("GLOBAL SVO exposes effects without retaining alternate render or lighting modes", () => {
  assert.deepEqual(DEFAULT_SVO_LIGHTING_OPTIONS, {
    shadowsEnabled: true,
    ambientOcclusionEnabled: true,
  });
  const options = readFileSync(new URL("../lib/svo-render-options.ts", import.meta.url), "utf8");
  for (const retired of ["SVO_RENDER_MODES", "SVO_LIGHTING_MODES", "SvoRenderMode", "SvoLightingMode"]) {
    assert.doesNotMatch(options, new RegExp(retired));
  }
});

test("visual controls expose one GLOBAL path with structural SVO overlays", () => {
  const panel = readFileSync(new URL("../components/VisualPanel.tsx", import.meta.url), "utf8");
  assert.doesNotMatch(panel, /aria-label="Renderer"|>RASTER<\/button>|>DIRECT<\/button>|>BEAUTIFUL<\/button>/);
  assert.match(panel, /GLOBAL SVO VIEW/);
  assert.match(panel, /selectRepresentation\("smooth"\)[^]*>SHADED<\/button>/);
  assert.match(panel, /selectRepresentation\("raw-voxels"\)[^]*>RAW<\/button>/);
  assert.match(panel, /selectRepresentation\("surface-voxels"\)[^]*>SURFACE<\/button>/);
  assert.match(panel, /selectRepresentation\("occupied-bricks"\)[^]*>CONTENT<\/button>/);
  assert.match(panel, /aria-label="SVO lighting effects"[^]*<Toggle label="Shadows"[^]*onChange=\{setSvoShadowsEnabled\}[^]*<Toggle label="AO"[^]*onChange=\{setSvoAmbientOcclusionEnabled\}/);
  assert.match(panel, /aria-label="SVO primary tracing optimizations"[^]*<Toggle label="Reuse stationary visibility"[^]*stationaryPrimaryReuseEnabled/,
    "the default-off primary coherence optimization must be exposed in the SVO controls");

  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  assert.match(renderer, /requested\.push\("svo-dry-scene"\)/,
    "GLOBAL remains active for every structural view");
  assert.match(renderer, /Structural views diagnose the same GLOBAL frame[^]*colorLoadOp: "load"/,
    "structural views blend over GLOBAL instead of clearing it");
  assert.doesNotMatch(renderer, /fallbackReason: "inspection-mode"/);
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
