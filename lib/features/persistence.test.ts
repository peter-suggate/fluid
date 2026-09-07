import "../methods";
import assert from "node:assert/strict";
import test from "node:test";
import { uiFeatureQuery } from "./persistence";
import { initialRuntimeFeatures, resetRuntimeFeatures } from "./runtime-lifecycle";
import { createPaneSession } from "../core/session/session";
import { parseQueryState, serializeQueryState, applyQueryStateToSession } from "../core/url-state";
test("feature UI state round-trips through the application URL host", () => {
  const parsed = parseQueryState("?fluidSurface=wireframe&svoShadows=0&svoAO=0&svoPrimary=traced&svoCones=exact&freezeTopology=1");
  assert.equal(parsed.ui.fluidSurfaceRenderMode, "wireframe");
  assert.equal(parsed.topologyFrozen, true);
  const query = serializeQueryState("?external=keep", parsed, parsed, parsed.ui, undefined, undefined, parsed);
  assert.equal(new URLSearchParams(query).get("external"), "keep");
  const restored = parseQueryState(query);
  assert.deepEqual(restored.ui, parsed.ui);
  assert.equal(restored.topologyFrozen, true);
});
test("feature defaults are canonical and simulation reset clears topology freeze", () => {
  const query = new URLSearchParams();
  uiFeatureQuery.write(query, uiFeatureQuery.read(query));
  assert.equal(query.toString(), "");
  assert.deepEqual(initialRuntimeFeatures(), { topologyFrozen: false });
  assert.deepEqual(resetRuntimeFeatures("simulation"), { topologyFrozen: false });
});

test("hydration and reset use feature state without leaking into another pane", () => {
  const a = createPaneSession("a"), b = createPaneSession("b");
  applyQueryStateToSession(a, "?freezeTopology=1&fluidSurface=wireframe&svoPrimary=traced");
  assert.equal(a.runtime.getState().topologyFrozen, true);
  assert.equal(a.ui.getState().fluidSurfaceRenderMode, "wireframe");
  assert.equal(a.ui.getState().svoPrimaryTraversal, "traced");
  assert.equal(b.runtime.getState().topologyFrozen, false);
  assert.equal(b.ui.getState().fluidSurfaceRenderMode, "simple");
  a.runtime.getState().resetSimulationTime();
  assert.equal(a.runtime.getState().topologyFrozen, false);
  assert.equal(a.ui.getState().fluidSurfaceRenderMode, "wireframe");
});
