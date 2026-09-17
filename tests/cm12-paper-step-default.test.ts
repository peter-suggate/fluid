import test from "node:test";
import assert from "node:assert/strict";
import { CM12_SHARPENING_DISTANCE_CELLS } from "../lib/core/cm12-numerics";
import {
  resolveMethodValues,
  type SimulationMethod,
} from "../lib/core/method-contract";
import { adaptiveMassMethod } from "../lib/methods/adaptive-volume/method";
import { uniformMethod } from "../lib/methods/uniform/method";

// Method defaults remain the paper step. Analytic study scenes deliberately
// select the scene timestep; their motion contracts are tested with those scenes.

const CM12_METHODS: ReadonlyArray<readonly [string, SimulationMethod]> = [
  ["uniform", uniformMethod],
  ["adaptive-volume", adaptiveMassMethod],
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


