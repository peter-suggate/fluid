import assert from "node:assert/strict";
import test from "node:test";
import { DEFAULT_RIGHT_PANEL_WIDTH, MAX_RIGHT_PANEL_WIDTH, MIN_RIGHT_PANEL_WIDTH, useUIStore } from "../lib/stores/ui-store";

test("viewport utility panels are mutually exclusive", () => {
  const initial = useUIStore.getInitialState();
  useUIStore.setState(initial, true);

  useUIStore.getState().setRightPanel("visual");
  assert.equal(useUIStore.getState().rightPanel, "visual");
  assert.equal(useUIStore.getState().diagnosticsOpen, false);

  useUIStore.getState().setDiagnosticsOpen(true);
  assert.equal(useUIStore.getState().rightPanel, "diagnostics");
  assert.equal(useUIStore.getState().diagnosticsOpen, true);

  useUIStore.getState().setRightPanel("bodies");
  assert.equal(useUIStore.getState().rightPanel, "bodies");
  assert.equal(useUIStore.getState().diagnosticsOpen, false);

  useUIStore.getState().setRightPanel("performance");
  assert.equal(useUIStore.getState().rightPanel, "performance");
  assert.equal(useUIStore.getState().diagnosticsOpen, false);

  useUIStore.setState(initial, true);
});

test("right panel width defaults to performance width and clamps resize input", () => {
  const initial = useUIStore.getInitialState();
  useUIStore.setState(initial, true);

  assert.equal(useUIStore.getState().rightPanelWidth, DEFAULT_RIGHT_PANEL_WIDTH);
  useUIStore.getState().setRightPanelWidth(713.6);
  assert.equal(useUIStore.getState().rightPanelWidth, 714);
  useUIStore.getState().setRightPanelWidth(1);
  assert.equal(useUIStore.getState().rightPanelWidth, MIN_RIGHT_PANEL_WIDTH);
  useUIStore.getState().setRightPanelWidth(10_000);
  assert.equal(useUIStore.getState().rightPanelWidth, MAX_RIGHT_PANEL_WIDTH);

  useUIStore.setState(initial, true);
});

test("presentation defaults to beautiful sparse voxels", () => {
  const initial = useUIStore.getInitialState();
  useUIStore.setState(initial, true);
  assert.equal(useUIStore.getState().svoRenderMode, "svo");
  assert.equal(useUIStore.getState().svoLightingMode, "cone");
  assert.equal(useUIStore.getState().svoShadowsEnabled, true);
  assert.equal(useUIStore.getState().svoAmbientOcclusionEnabled, true);
  useUIStore.getState().setSvoRenderMode("svo");
  useUIStore.getState().setSvoLightingMode("direct");
  useUIStore.getState().setSvoShadowsEnabled(false);
  useUIStore.getState().setSvoAmbientOcclusionEnabled(false);
  assert.equal(useUIStore.getState().svoRenderMode, "svo");
  assert.equal(useUIStore.getState().svoLightingMode, "direct");
  assert.equal(useUIStore.getState().svoShadowsEnabled, false);
  assert.equal(useUIStore.getState().svoAmbientOcclusionEnabled, false);
  useUIStore.setState(initial, true);
});
