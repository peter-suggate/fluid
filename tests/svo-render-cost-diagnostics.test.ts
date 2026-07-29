import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  DEFAULT_SVO_RENDER_DIAGNOSTICS,
  SVO_COST_OVERLAY_DEFINITIONS,
  SVO_COST_OVERLAY_MODES,
  SVO_COST_SEQUENTIAL_LEGEND,
  normalizeSvoRenderDiagnostics,
  svoCostOverlayCode,
} from "../lib/svo-render-diagnostics";
import { useUIStore } from "../lib/stores/ui-store";
import { createSvoDrySceneFragmentWGSL, svoDrySceneShader } from "../lib/webgpu-svo-dry-scene";

const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
const panelSource = readFileSync(new URL("../components/VisualPanel.tsx", import.meta.url), "utf8");
const viewportSource = readFileSync(new URL("../components/WebGPUViewport.tsx", import.meta.url), "utf8");

test("SVO cost diagnostic controls are bounded and retain a production-off default", () => {
  assert.deepEqual(DEFAULT_SVO_RENDER_DIAGNOSTICS, {
    overlay: "off", maximumTraversalDepth: 21, maximumNodeVisits: 256, overlayOpacity: 0.9,
  });
  assert.equal(svoCostOverlayCode("traversal-depth"), 1);
  assert.equal(svoCostOverlayCode("brick-tests"), 3);
  assert.equal(svoCostOverlayCode("candidate-work"), 5);
  assert.equal(svoCostOverlayCode("shadow-work"), 6);
  assert.equal(svoCostOverlayCode("mip-steps"), 7);
  assert.equal(svoCostOverlayCode("total-cost"), 8);
  assert.equal(svoCostOverlayCode("exhaustion"), 9);
  assert.equal(svoCostOverlayCode("voxel-work"), 10);
  assert.equal(svoCostOverlayCode("prepass-fallback"), 14);
  assert.equal(svoCostOverlayCode("work-composition"), 15);
  assert.deepEqual(normalizeSvoRenderDiagnostics({
    overlay: "total-cost", maximumTraversalDepth: 99, maximumNodeVisits: 0, overlayOpacity: 2,
  }), { overlay: "total-cost", maximumTraversalDepth: 21, maximumNodeVisits: 1, overlayOpacity: 1 });

  const initial = useUIStore.getInitialState();
  useUIStore.setState(initial, true);
  useUIStore.getState().setSvoCostOverlay("node-visits");
  useUIStore.getState().setSvoMaximumTraversalDepth(0);
  useUIStore.getState().setSvoMaximumNodeVisits(999);
  useUIStore.getState().setSvoOverlayOpacity(-1);
  assert.deepEqual({
    overlay: useUIStore.getState().svoCostOverlay,
    depth: useUIStore.getState().svoMaximumTraversalDepth,
    visits: useUIStore.getState().svoMaximumNodeVisits,
    opacity: useUIStore.getState().svoOverlayOpacity,
  }, { overlay: "node-visits", depth: 1, visits: 256, opacity: 0 });
  useUIStore.setState(initial, true);
});

test("SVO scene visualizations publish complete descriptions and high-contrast legends", () => {
  assert.equal(new Set(SVO_COST_OVERLAY_MODES).size, SVO_COST_OVERLAY_MODES.length);
  assert.deepEqual(Object.keys(SVO_COST_OVERLAY_DEFINITIONS), [...SVO_COST_OVERLAY_MODES]);
  for (const mode of SVO_COST_OVERLAY_MODES) {
    const definition = SVO_COST_OVERLAY_DEFINITIONS[mode];
    assert.equal(definition.mode, mode);
    assert.ok(definition.label.length > 0);
    assert.ok(definition.description.length > 20, `${mode} needs an explanatory sentence`);
    assert.ok(definition.legend.length >= 3, `${mode} needs a readable legend`);
  }
  assert.deepEqual(SVO_COST_SEQUENTIAL_LEGEND.map(({ color }) => color), [
    "#08051f", "#2447ff", "#00d9ff", "#00ff85", "#eaff00", "#ff8500", "#ff174d", "#ffffff",
  ]);
  assert.deepEqual(SVO_COST_OVERLAY_DEFINITIONS["work-composition"].legend.map(({ label }) => label),
    ["Primary", "Shadow", "Cone / mip"]);
  assert.deepEqual(SVO_COST_OVERLAY_DEFINITIONS["prepass-fallback"].legend.map(({ label }) => label),
    ["Inline / unavailable", "Prepass reused", "Full-res fallback"]);
});

test("render panel and viewport expose selectable scene heatmaps with an in-scene legend", () => {
  assert.match(panelSource, /SVO traversal cost/);
  assert.match(panelSource, /Maximum traversal depth/);
  assert.match(panelSource, /Maximum node visits/);
  assert.match(panelSource, /SVO_COST_OVERLAY_MODES\.map/);
  assert.match(panelSource, /SVO_COST_OVERLAY_DEFINITIONS\[mode\]/);
  assert.match(panelSource, /mode === svoCostOverlay && mode !== "off" \? "off" : mode/);
  assert.match(viewportSource, /className="svo-cost-legend"/);
  assert.match(viewportSource, /SVO_COST_OVERLAY_DEFINITIONS\[svoCostOverlay\]/);
  assert.match(viewportSource, /maximumTraversalDepth: ui\.svoMaximumTraversalDepth/);
});

test("render panel contains rendering controls only; solver fields stay in performance", () => {
  assert.doesNotMatch(panelSource, /useMethodStore|gridOverlay|paperPipeline|finePublicationGate/);
  assert.doesNotMatch(panelSource, />Solver grid<|Paper pipeline inspector|CFL load|Projected divergence/);
  assert.match(panelSource, /RENDER OBSERVATORY/);
  assert.match(panelSource, /SCENE REPRESENTATION/);
  assert.match(panelSource, /LIVE PER-PIXEL SCENE HEATMAPS/);
});

test("dry shader measures topology, candidate, mip, and shadow work before applying the heatmap", () => {
  assert.match(svoDrySceneShader, /fn dryConfiguredMapping\(\)->SvoMapping/);
  assert.match(svoDrySceneShader, /mapping\.maxVisits=min\(mapping\.maxVisits,dryDiagnosticMaximumNodeVisits\(\)\)/);
  assert.match(svoDrySceneShader, /fn dryTraverse\([^]*svoTraverseWithDepthLimit/);
  assert.match(svoDrySceneShader, /dryPrimaryNodeVisits\+=leaf\.visits/);
  assert.match(svoDrySceneShader, /dryPrimaryLeafVisits\+=1u/);
  assert.match(svoDrySceneShader, /dryPrimaryEmptyBrickSkips\+=1u/);
  assert.match(svoDrySceneShader, /dryPrimaryVoxelWorkItems\+=1u/);
  assert.match(svoDrySceneShader, /dryCandidateWorkItems\+=candidate\.workItems/);
  assert.match(svoDrySceneShader, /dryShadowNodeVisits\+=result\.nodeVisits/);
  assert.match(svoDrySceneShader, /dryMipSteps\+=1u/);
  assert.match(svoDrySceneShader, /fn dryLogCost\([^]*log2\(1\.0\+max\(value,0\.0\)\)/);
  assert.match(svoDrySceneShader, /primaryCost\*vec3f\(0\.0,\.851,1\.0\)[^]*shadowCost\*vec3f\(1\.0,\.078,\.576\)[^]*mipCost\*vec3f\(1\.0,\.902,0\.0\)/);
  assert.match(svoDrySceneShader, /targets\.radianceDepth=dryCostOverlay\(targets\.radianceDepth\)/);
  assert.match(rendererSource, /svoCostOverlayCode\(activeSvoDiagnostics\.overlay\)/);
  assert.match(rendererSource, /diagnosticsKey/);
});

test("reduced cone lighting visualizes prepass reuse and full-resolution fallback", () => {
  const reducedShader = createSvoDrySceneFragmentWGSL(0.5);
  const prepassMode = svoCostOverlayCode("prepass-fallback");
  assert.match(reducedShader, new RegExp(`mode==${prepassMode}u`));
  assert.match(reducedShader, /dryPrepassState==1u[^]*vec3f\(0\.0,\.9,1\.0\)/);
  assert.match(reducedShader, /dryConeFallback==1u[^]*vec3f\(1\.0,\.09,\.30\)/);
  assert.match(reducedShader, /dryPrepassData=vec4f\(1\.0\);dryPrepassState=0u;dryConeFallback=0u/);
});
