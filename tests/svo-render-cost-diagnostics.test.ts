import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  DEFAULT_SVO_RENDER_DIAGNOSTICS,
  normalizeSvoRenderDiagnostics,
  svoCostOverlayCode,
} from "../lib/svo-render-diagnostics";
import { useUIStore } from "../lib/stores/ui-store";
import { svoDrySceneShader } from "../lib/webgpu-svo-dry-scene";

const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
const panelSource = readFileSync(new URL("../components/VisualPanel.tsx", import.meta.url), "utf8");
const viewportSource = readFileSync(new URL("../components/WebGPUViewport.tsx", import.meta.url), "utf8");

test("SVO cost diagnostic controls are bounded and retain a production-off default", () => {
  assert.deepEqual(DEFAULT_SVO_RENDER_DIAGNOSTICS, {
    overlay: "off", maximumTraversalDepth: 21, maximumNodeVisits: 256, overlayOpacity: 0.82,
  });
  assert.equal(svoCostOverlayCode("traversal-depth"), 1);
  assert.equal(svoCostOverlayCode("exhaustion"), 7);
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

test("render panel and viewport expose selectable scene heatmaps with an in-scene legend", () => {
  assert.match(panelSource, /SVO traversal cost/);
  assert.match(panelSource, /Maximum traversal depth/);
  assert.match(panelSource, /Maximum node visits/);
  assert.match(panelSource, /SVO_COST_OVERLAY_MODES\.map/);
  assert.match(viewportSource, /className="svo-cost-legend"/);
  assert.match(viewportSource, /maximumTraversalDepth: ui\.svoMaximumTraversalDepth/);
});

test("dry shader measures topology, field, candidate, and shadow work before applying the heatmap", () => {
  assert.match(svoDrySceneShader, /fn dryConfiguredMapping\(\)->SvoMapping/);
  assert.match(svoDrySceneShader, /mapping\.maxVisits=min\(mapping\.maxVisits,dryDiagnosticMaximumNodeVisits\(\)\)/);
  assert.match(svoDrySceneShader, /fn dryTraverse\([^]*svoTraverseWithDepthLimit/);
  assert.match(svoDrySceneShader, /dryPrimaryNodeVisits\+=leaf\.visits/);
  assert.match(svoDrySceneShader, /dryPrimaryFieldSteps\+=fluid\.fieldSteps/);
  assert.match(svoDrySceneShader, /dryCandidateWorkItems\+=candidate\.workItems/);
  assert.match(svoDrySceneShader, /dryShadowNodeVisits\+=result\.nodeVisits/);
  assert.match(svoDrySceneShader, /targets\.radianceDepth=dryCostOverlay\(targets\.radianceDepth\)/);
  assert.match(rendererSource, /svoCostOverlayCode\(activeSvoDiagnostics\.overlay\)/);
  assert.match(rendererSource, /diagnosticsKey/);
});
