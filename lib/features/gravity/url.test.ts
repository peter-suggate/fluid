import "../../methods";
import assert from "node:assert/strict";
import test from "node:test";
import { parseQueryState, serializeQueryState } from "../../core/url-state";
import { setGravity, toggleGravity } from "./state";

test("disabled gravity remembers the authored vector across a URL round trip", () => {
  const initial = parseQueryState("");
  initial.scene.fluid = toggleGravity(setGravity(initial.scene.fluid, { x: 2, y: -3, z: 4 }));
  const query = serializeQueryState("", initial, initial);
  const restored = parseQueryState(query);
  assert.deepEqual(restored.scene.fluid.gravity_m_s2, { x: 0, y: 0, z: 0 });
  assert.deepEqual(toggleGravity(restored.scene.fluid).gravity_m_s2, { x: 2, y: -3, z: 4 });
});

test("malformed gravity memory cannot enter a hydrated scene", () => {
  const query = new URLSearchParams({ "scene.fluid.rememberedGravity_m_s2": JSON.stringify({ y: -3 }) });
  const restored = parseQueryState(query.toString());
  assert.equal(restored.scene.fluid.rememberedGravity_m_s2, undefined);
});
