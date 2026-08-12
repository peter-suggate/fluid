import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { getMethod, interactiveMethodId, interactiveSimulationMethods } from "../lib/methods";

const methodPanelSource = readFileSync(new URL("../components/MethodPanel.tsx", import.meta.url), "utf8");
const urlStateSource = readFileSync(new URL("../lib/url-state.ts", import.meta.url), "utf8");

test("interactive method picker exposes adaptive and uniform GPU runtimes", () => {
  assert.deepEqual(interactiveSimulationMethods.map((method) => method.id), ["octree", "uniform"]);
  assert.match(methodPanelSource, /ariaLabel="Simulation method"/);
  assert.doesNotMatch(methodPanelSource, /Regular tall cells/);
  assert.doesNotMatch(methodPanelSource, /Experimental/);
});

test("UI hydration restores supported methods and rejects offline-only methods", () => {
  assert.equal(interactiveMethodId("tall-cell"), "uniform");
  assert.equal(interactiveMethodId("octree"), "octree");
  assert.equal(interactiveMethodId("uniform"), "uniform");
  assert.equal(getMethod("unknown").id, "uniform");
  assert.match(urlStateSource, /methodId: interactiveMethodId\(state\.methodId\)/);
  assert.doesNotMatch(urlStateSource, /methodId: "octree", quality: state\.quality/);
});
