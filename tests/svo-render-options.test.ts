import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { DEFAULT_SVO_LIGHTING_OPTIONS } from "../lib/svo-render-options";

test("GLOBAL SVO exposes effects without retaining alternate render or lighting modes", () => {
  assert.deepEqual(DEFAULT_SVO_LIGHTING_OPTIONS, {
    shadowsEnabled: true,
    ambientOcclusionEnabled: true,
    silhouetteRefinementEnabled: false,
    coneTracingMode: "cones",
    primaryTraversal: "raster",
  });
  const options = readFileSync(new URL("../lib/svo-render-options.ts", import.meta.url), "utf8");
  for (const retired of ["SVO_RENDER_MODES", "SVO_LIGHTING_MODES", "SvoRenderMode", "SvoLightingMode"]) {
    assert.doesNotMatch(options, new RegExp(retired));
  }
});

test("visual controls expose one GLOBAL path with no surface arms", () => {
  const panel = readFileSync(new URL("../components/VisualPanel.tsx", import.meta.url), "utf8");
  assert.doesNotMatch(panel, /aria-label="Renderer"|>RASTER<\/button>|>DIRECT<\/button>|>BEAUTIFUL<\/button>/);
  // EXACT / SHADED / RAW selected between three ways of reconstructing the
  // shading normal per pixel. The voxeliser bakes the normal into the voxel now,
  // so there is one arm and the strip is gone with it — a control with no
  // second arm behind it is a lie about what the frame can be.
  assert.doesNotMatch(panel, /selectSurface|surfaceReconstruction/,
    "no control may still offer a surface-reconstruction arm");
  // LEVELS / SURFACE / BRICKS / CONTENT selected the expanded-record inspection
  // overlay — a second renderer, fed from its own 48-byte-per-voxel arenas
  // (~295 MB on the widened ocean scene), drawn additively over the real SVO
  // frame. It went with those arenas, and with it the store field the whole
  // strip switched on; the surface arms were the last thing in that strip, so
  // there is no view strip left at all.
  assert.doesNotMatch(panel, /voxelRenderMode|selectRepresentation|setVoxelRenderMode/,
    "no control may still reach for the removed inspection representation");
  assert.doesNotMatch(panel, />LEVELS<\/button>|>SURFACE<\/button>|>BRICKS<\/button>|>CONTENT<\/button>/,
    "the inspection representation buttons are gone with their renderer");
  assert.match(panel, /aria-label="SVO lighting effects"[^]*<Toggle label="Shadows"[^]*onChange=\{setSvoShadowsEnabled\}[^]*<Toggle label="AO"[^]*onChange=\{setSvoAmbientOcclusionEnabled\}/);
  assert.match(panel, /<Toggle label="Close primary seams"[^]*checked=\{silhouetteRefinementEnabled\}[^]*onChange=\{setSilhouetteRefinementEnabled\}/,
    "primary seam closure must be an explicit render control");
  assert.match(panel, /data-testid="silhouette-refinement-status"[^]*silhouetteRefinementStatus\.state\.toUpperCase\(\)/,
    "the requested pass lifecycle must be visible rather than silently substituted");
  assert.match(panel, /aria-label="SVO primary tracing optimizations"[^]*<Toggle label="Reuse stationary visibility"[^]*stationaryPrimaryReuseEnabled/,
    "the default-off primary coherence optimization must be exposed in the SVO controls");

  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  assert.match(renderer, /requested\.push\("svo-dry-scene"\)/,
    "GLOBAL remains the one production presentation");
  assert.doesNotMatch(renderer, /voxelRenderMode|voxelDebug|voxelInspection|"voxel-debug"/,
    "the renderer must hold no attachment, depth target or encode path for the removed overlay");
  assert.doesNotMatch(renderer, /failureReason: "inspection-mode"/);
});

test("primary visibility is switchable between the raster and traced paths from the render panel", () => {
  const panel = readFileSync(new URL("../components/VisualPanel.tsx", import.meta.url), "utf8");
  assert.match(panel, /<span>Primary visibility<\/span>[^]*mode: "raster"[^]*mode: "traced"[^]*setSvoPrimaryTraversal\(mode\)/,
    "both primary traversals must be selectable side by side");
  // Reuse caches the traced primary and cannot engage against the rastered one,
  // so the control that offers it has to go quiet rather than read as available.
  assert.match(panel, /<Toggle label="Reuse stationary visibility"[^]*disabled=\{svoPrimaryTraversal === "raster"\}/,
    "stationary reuse must present as unavailable while the raster primary is selected");

  const viewport = readFileSync(new URL("../components/WebGPUViewport.tsx", import.meta.url), "utf8");
  assert.match(viewport, /primaryTraversal: ui\.svoPrimaryTraversal/,
    "the selection must reach the renderer with the rest of the lighting options");
  assert.match(viewport, /silhouetteRefinementEnabled: ui\.silhouetteRefinementEnabled/,
    "the silhouette-refinement request must reach the renderer with the shared lighting options");

  // The mode is compiled into the dry-scene shader variants and render-pass
  // shape, so it can only change by retiring the pipeline that baked it in.
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  const swapStart = renderer.indexOf("private applyPrimaryTraversalRequest");
  assert.ok(swapStart > 0, "the renderer must act on a changed traversal request");
  const swap = renderer.slice(swapStart, renderer.indexOf("private ensureOptionalPipeline<T>", swapStart));
  assert.match(swap, /this\.failedOptionalPipelines\.delete\("svo-dry-scene"\)/,
    "a sticky failure of the previous mode must not block the mode being asked for now");
  assert.match(swap, /this\.svoDryScenePipeline = undefined/,
    "the retired pipeline must be cleared so the next sweep rebuilds it");
  assert.match(swap, /retired\.destroy\(\)/, "the retired pipeline must release its GPU resources");
  // The request is resolved against the scene's scale on the way in — a scene
  // that emits more brick proxies than the target has pixels takes the
  // megakernel whatever the toggle says — so the call carries the structural
  // terms too. It still has to precede the sweep.
  assert.match(renderer, /this\.applyPrimaryTraversalRequest\(svoLightingOptions\.primaryTraversal \?\? "raster", \{[^]*?\}\);\s*\n\s*\}\s*\n\s*this\.ensureRequestedOptionalPipelines\(/,
    "the swap must run before the sweep so a toggle rebuilds in the same frame");
  assert.match(swap, /resolveSvoPrimaryTraversal\(requested, scale\)/,
    "the traversal a frame runs must be the shared rule's answer, not the raw toggle");
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
