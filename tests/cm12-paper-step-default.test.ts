import test from "node:test";
import assert from "node:assert/strict";
import { CM12_SHARPENING_DISTANCE_CELLS } from "../lib/core/cm12-numerics";
import {
  resolveMethodValues,
  type SimulationMethod,
} from "../lib/core/method-contract";
import { SCENE_CATALOG } from "../lib/core/scenes";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import { uniformMethod } from "../lib/methods/uniform/method";

/**
 * Every CM12 lane runs at the paper's 1/30 s unless something opts out on
 * purpose.
 *
 * Sec. 3.5 sharpening only balances transport diffusion at that per-step dose,
 * so a scene that lands on the authored dt is running a method whose surface
 * treatment is out of balance with its transport — and doing it silently,
 * because a profile override is not visible anywhere the step is read. The
 * deliberate opt-outs are elsewhere by construction: harness lanes name
 * `timeStep: "scene"` in the smoke catalog, tools pass it in their own
 * overrides, and the step slider releases it through the controller. What must
 * not happen is a scene arriving off the paper step by default.
 */

const CM12_METHODS: ReadonlyArray<readonly [string, SimulationMethod]> = [
  ["uniform", uniformMethod],
  ["adaptive-mass", adaptiveMassMethod],
];

test("both CM12 methods declare the paper step as their default", () => {
  for (const [id, method] of CM12_METHODS) {
    const spec = method.params.find((candidate) => candidate.key === "timeStep");
    assert.ok(spec, `${id} has no timeStep parameter`);
    assert.equal(spec.default, "paper", `${id} does not default to the paper step`);
    assert.equal(method.presetFor?.("balanced")?.timeStep ?? "paper", "paper",
      `${id}'s balanced preset leaves the paper step`);
  }
});

test("both CM12 methods use the shared sharpening return distance", () => {
  for (const [id, method] of CM12_METHODS) {
    const spec = method.params.find((candidate) =>
      candidate.key === "sharpeningDistance");
    assert.ok(spec, `${id} has no sharpeningDistance parameter`);
    assert.equal(spec.default, CM12_SHARPENING_DISTANCE_CELLS,
      `${id} does not expose the shared sharpening default`);
    assert.equal(
      resolveMethodValues(method, "balanced", {}).sharpeningDistance,
      CM12_SHARPENING_DISTANCE_CELLS,
      `${id}'s balanced preset overrides the shared sharpening default`,
    );
  }
});

test("no scene profile sends a CM12 lane off the paper step", () => {
  const cm12Ids = new Set(CM12_METHODS.map(([id]) => id));
  const offPaper = SCENE_CATALOG
    .filter((definition) => definition.methodProfile
      && cm12Ids.has(definition.methodProfile.methodId)
      && (definition.methodProfile.overrides?.timeStep ?? "paper") !== "paper")
    .map((definition) => definition.id);
  assert.deepEqual(offPaper, [],
    `these scenes default a CM12 lane to the authored dt: ${offPaper.join(", ")}`);
});

test("a CM12 scene profile that names the step names the paper one", () => {
  const named = SCENE_CATALOG.filter((definition) =>
    definition.methodProfile?.overrides?.timeStep !== undefined);
  // Not a coverage claim: only that whatever exists today is on the paper step.
  for (const definition of named) {
    assert.equal(definition.methodProfile!.overrides!.timeStep, "paper",
      `${definition.id} pins a non-paper step`);
  }
});
