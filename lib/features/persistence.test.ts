import "../methods";
import assert from "node:assert/strict";
import test from "node:test";
import { uiFeatureQuery } from "./persistence";
import { createPaneSession } from "../core/session/session";
import { parseQueryState, serializeQueryState, applyQueryStateToSession } from "../core/url-state";
test("feature UI state round-trips through the application URL host", () => {
  const parsed = parseQueryState("?fluidSurface=wireframe&svoShadows=0&svoAO=0&svoPrimary=traced&svoCones=exact");
  assert.equal(parsed.ui.fluidSurfaceRenderMode, "wireframe");
  const query = serializeQueryState("?external=keep", parsed, parsed, parsed.ui, undefined, undefined, parsed);
  assert.equal(new URLSearchParams(query).get("external"), "keep");
  const restored = parseQueryState(query);
  assert.deepEqual(restored.ui, parsed.ui);
});
test("feature defaults are canonical", () => {
  const query = new URLSearchParams();
  uiFeatureQuery.write(query, uiFeatureQuery.read(query));
  assert.equal(query.toString(), "");
});

test("reflection selection round-trips without modifying the scene or another pane", () => {
  const a = createPaneSession("a"), b = createPaneSession("b");
  const scene = a.scene.getState().scene;
  a.ui.getState().setWaterReflectionMode("environment");
  assert.equal(a.scene.getState().scene, scene);
  assert.equal(b.ui.getState().waterReflectionMode, "screen-space");
  const params = new URLSearchParams();
  uiFeatureQuery.write(params, a.ui.getState());
  assert.equal(params.get("waterReflections"), "environment");
  assert.equal(parseQueryState(`?${params}`).ui.waterReflectionMode, "environment");
  assert.equal(parseQueryState("?waterReflections=invalid").ui.waterReflectionMode, "screen-space");
  a.ui.getState().setWaterReflectionMode("world-space");
  uiFeatureQuery.write(params, a.ui.getState());
  assert.equal(parseQueryState(`?${params}`).ui.waterReflectionMode, "world-space");
  assert.equal(a.scene.getState().scene, scene);
});

test("hydration and reset use feature state without leaking into another pane", () => {
  const a = createPaneSession("a"), b = createPaneSession("b");
  applyQueryStateToSession(a, "?fluidSurface=wireframe&svoPrimary=traced");
  assert.equal(a.ui.getState().fluidSurfaceRenderMode, "wireframe");
  assert.equal(a.ui.getState().svoPrimaryTraversal, "traced");
  assert.equal(b.ui.getState().fluidSurfaceRenderMode, "simple");
  a.runtime.getState().resetSimulationTime();
  assert.equal(a.ui.getState().fluidSurfaceRenderMode, "wireframe");
});


test("World refraction is opt-in, pane-local, and independent of reflection selection", () => {
  const a = createPaneSession("a"), b = createPaneSession("b");
  const scene = a.scene.getState().scene;
  assert.equal(a.ui.getState().waterWorldRefractionEnabled, false);
  a.ui.getState().setWaterWorldRefractionEnabled(true);
  assert.equal(a.ui.getState().waterReflectionMode, "screen-space");
  assert.equal(a.scene.getState().scene, scene);
  assert.equal(b.ui.getState().waterWorldRefractionEnabled, false);
  const params = new URLSearchParams();
  uiFeatureQuery.write(params, a.ui.getState());
  assert.equal(params.get("waterWorldRefraction"), "1");
  assert.equal(parseQueryState(`?${params}`).ui.waterWorldRefractionEnabled, true);
  applyQueryStateToSession(a, "?waterWorldRefraction=0&waterReflections=world-space");
  assert.equal(a.ui.getState().waterWorldRefractionEnabled, false);
  assert.equal(a.ui.getState().waterReflectionMode, "world-space");
});
