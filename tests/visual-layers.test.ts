import "../lib/methods";
import assert from "node:assert/strict";
import test from "node:test";
import { VISUAL_LAYERS, visualLayers, readVisualLayers, writeVisualLayers, legacyVisualLayers, toggleVisualLayer, scalarLayerPaint } from "../lib/core/visual-layers";
import { uniformLabQuery } from "../advance-lab/uniform-lab-state";
import { createUIStore } from "../lib/core/stores/ui-store";
import { parseQueryState, serializeQueryState } from "../lib/core/url-state";

test("layer selection is canonical, independent and preserves hidden selections", () => {
  const ids = VISUAL_LAYERS.map(l => l.id);
  let state = visualLayers([]);
  for (const id of [...ids].reverse()) state = toggleVisualLayer(state, id);
  assert.deepEqual(state.enabled, ids);
  state = { ...state, visible: false, opacity: { volume: 0.25 } };
  assert.deepEqual(readVisualLayers(writeVisualLayers(state)), state);
  assert.deepEqual(toggleVisualLayer(state, "pressure").enabled, ids.filter(id => id !== "pressure"));
  assert.deepEqual(readVisualLayers('{"enabled":["bogus","grid","grid"],"opacity":{"grid":99}}'), { enabled: ["grid"], visible: true, opacity: { grid: 1 } });
});
test("old lab links migrate surface/released faces and the independent grid", () => {
  const state = uniformLabQuery.read(new URLSearchParams("field=release&grid=1"));
  assert.deepEqual(state.layers, legacyVisualLayers("release", true));
  const query = new URLSearchParams();
  uniformLabQuery.write(query, { ...state, layers: visualLayers([]) });
  assert.deepEqual(uniformLabQuery.read(query).layers.enabled, []);
});
test("3D links roundtrip combined layers and opacity", () => {
  const state = parseQueryState("method=uniform-volume&grid=z");
  state.ui.visualLayers = { ...visualLayers(["volume", "window", "tiles", "release"]), opacity: { volume: 0.2 } };
  const query = serializeQueryState("", { presetId: state.presetId, scene: state.scene }, state, state.ui);
  assert.deepEqual(parseQueryState(query).ui.visualLayers, state.ui.visualLayers);
});
test("shared scalar semantics distinguish volume, density, signed pressure and empty air", () => {
  assert.equal(scalarLayerPaint("density", 0).alpha, 0);
  assert.equal(scalarLayerPaint("volume", 0.5).alpha, 0.5);
  assert.notDeepEqual(scalarLayerPaint("pressure", 5000).color, scalarLayerPaint("pressure", -5000).color);
  assert.equal(scalarLayerPaint("pressure", 5000).alpha, 0.5);
});

test("legacy controls hide composed layers and explicit mode picks replace them", () => {
  const store = createUIStore();
  store.setState({ visualLayers: visualLayers(["volume", "grid"]) });
  store.getState().setGridOverlayAxis("off");
  assert.equal(store.getState().visualLayers?.visible, false);
  store.getState().setGridOverlayAxis("x");
  assert.equal(store.getState().visualLayers?.visible, true);
  store.getState().setGridOverlayMode("phi");
  assert.equal(store.getState().visualLayers, undefined);
});
