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
    // On by default, and a real switch rather than an exposure: withheld, the
    // cone gather and the world-GI cache pass are never encoded, so the frame
    // gets that time back. `giBounceStrength: 0` costs exactly what it cost
    // before, which is why the two are separate controls.
    globalIlluminationEnabled: true,
    primaryTraversal: "traced",
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
  assert.match(panel, /aria-label="SVO lighting effects"[^]*<PipeToggle label="Shadows"[^]*onChange=\{setSvoShadowsEnabled\}[^]*<PipeToggle label="AO"[^]*onChange=\{setSvoAmbientOcclusionEnabled\}/);
  assert.match(panel, /data-testid="silhouette-refinement-status"[^]*silhouetteRefinementStatus\.state\.toUpperCase\(\)/,
    "the requested pass lifecycle must be visible rather than silently substituted");
  // Seam closure and stationary reuse are pipeline nodes now, so their switch is
  // the node lamp rather than a free-standing toggle. What has to survive the
  // move is that both stay reachable and that each still moves the one field the
  // frame actually reads.
  const graph = readFileSync(new URL("../lib/render-pipeline-graph.ts", import.meta.url), "utf8");
  for (const node of ["seam-closure", "stationary-reuse"]) {
    assert.match(graph, new RegExp(`id: "${node}"[^]*?toggleable: true`),
      `${node} must present as a switchable node`);
  }
  assert.match(panel, /id === "seam-closure"\) setSilhouetteRefinementEnabled\(!silhouetteRefinementEnabled\)/,
    "the seam-closure lamp must move the silhouette-refinement request");
  assert.match(panel, /id === "stationary-reuse"\) updateTuning\("stationaryPrimaryReuseEnabled"/,
    "the reuse lamp must move the primary coherence tuning");

  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  assert.match(renderer, /requested\.push\("svo-dry-scene"\)/,
    "GLOBAL remains the one production presentation");
  assert.doesNotMatch(renderer, /voxelRenderMode|voxelDebug|voxelInspection|"voxel-debug"/,
    "the renderer must hold no attachment, depth target or encode path for the removed overlay");
  assert.doesNotMatch(renderer, /failureReason: "inspection-mode"/);
});

test("the render panel offers no primary-traversal control, because the frame has no choice", () => {
  // `resolveSvoPrimaryTraversal` answers `traced` unconditionally, so a
  // RASTER/TRACED strip selected an arm that never ran — and the store defaulted
  // to `raster`, so the panel lit the arm the renderer never takes while the
  // megakernel drew every frame. Worse, that dead selection was the `disabled`
  // predicate on stationary reuse, which is live on exactly the path that does
  // run. The raster arm stays compiled and stays reachable through the
  // environment override; what it does not get is a control claiming to switch
  // the frame.
  const panel = readFileSync(new URL("../components/VisualPanel.tsx", import.meta.url), "utf8");
  assert.doesNotMatch(panel, /Primary visibility|setSvoPrimaryTraversal|svoPrimaryTraversal/,
    "the panel must not offer a traversal the resolver ignores");

  const options = readFileSync(new URL("../lib/svo-render-options.ts", import.meta.url), "utf8");
  assert.match(options, /FLUID_SVO_PRIMARY_TRAVERSAL/,
    "the raster arm must remain reachable for the paired-worktree measurement");

  const viewport = readFileSync(new URL("../components/WebGPUViewport.tsx", import.meta.url), "utf8");
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

test("global illumination is a switch that withholds work, not an exposure that scales it", () => {
  // `giBounceStrength: 0` multiplies a gather that has already run, so it costs
  // exactly what it cost before. This is the control that buys the time back:
  // the `globalIllumination` flags are withheld outright and the persistent
  // world-GI cache pass is never encoded. Cone shadows and AO are untouched,
  // which is what makes it worth having beside CONES/EXACT/OFF.
  const renderer = readFileSync(new URL("../lib/webgpu-svo-dry-scene.ts", import.meta.url), "utf8");
  assert.match(renderer, /const globalIlluminationEnabled = this\.lightingOptions\.globalIlluminationEnabled !== false;/,
    "the renderer must read the switch when it writes its visibility flags");
  assert.match(renderer, /const giReady = coneTracingEnabled && globalIlluminationEnabled &&/,
    "a withheld gather must not report itself ready");
  assert.match(renderer, /coneTracingEnabled && globalIlluminationEnabled \? SVO_DRY_VISIBILITY_FLAGS\.globalIlluminationRequested : 0/,
    "the request flag must be withheld too, or the shader still marches the cones");
  // Three conditions guard the cache now: the reconstruction family, this
  // switch, and the stage ablation. Match only the middle one — the others have
  // their own tests, and pinning the whole expression here would make this test
  // fail for reasons that have nothing to do with the gather.
  assert.match(renderer, /if \(!reconstructReducedRadiance && this\.lightingOptions\.globalIlluminationEnabled !== false/,
    "the world-GI cache pass must not be encoded when the gather is off");
  // A changed switch has to reach the flags, and the cache it filled under the
  // old setting has to go with it.
  assert.match(renderer, /\|\| globalIlluminationEnabled !== previousGlobalIllumination;/,
    "toggling the gather must invalidate the world-GI cache");

  const viewport = readFileSync(new URL("../components/WebGPUViewport.tsx", import.meta.url), "utf8");
  assert.match(viewport, /globalIlluminationEnabled: ui\.svoGlobalIlluminationEnabled/,
    "the switch must reach the renderer with the rest of the lighting options");

  const graph = readFileSync(new URL("../lib/render-pipeline-graph.ts", import.meta.url), "utf8");
  assert.match(graph, /id: "gi-composition"[^]*?toggleable: true/,
    "the GI node's lamp must be a control rather than a readout");
  const panel = readFileSync(new URL("../components/VisualPanel.tsx", import.meta.url), "utf8");
  assert.match(panel, /id === "gi-composition"\) setSvoGlobalIlluminationEnabled\(!svoGlobalIlluminationEnabled\)/,
    "the GI lamp must move the switch");
});
