import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { interactiveMethodId, interactiveSimulationMethods } from "../lib/methods";

const methodPanelSource = readFileSync(new URL("../components/MethodPanel.tsx", import.meta.url), "utf8");
const urlStateSource = readFileSync(new URL("../lib/url-state.ts", import.meta.url), "utf8");

test("interactive method picker exposes octree and regular tall cells", () => {
  assert.deepEqual(interactiveSimulationMethods.map((method) => method.id), ["octree", "tall-cell"]);
  assert.match(methodPanelSource, /ariaLabel="Simulation method"/);
  assert.match(methodPanelSource, /Regular tall cells/);
  assert.match(methodPanelSource, /Experimental/);
});

test("UI hydration restores supported methods and rejects offline-only methods", () => {
  assert.equal(interactiveMethodId("tall-cell"), "tall-cell");
  assert.equal(interactiveMethodId("octree"), "octree");
  assert.equal(interactiveMethodId("uniform"), "octree");
  assert.match(urlStateSource, /methodId: interactiveMethodId\(state\.methodId\)/);
  assert.doesNotMatch(urlStateSource, /methodId: "octree", quality: state\.quality/);
});
