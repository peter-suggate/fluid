import assert from "node:assert/strict";
import test from "node:test";
import { uniformVolumeMethod } from "../lib/methods/uniform/uniform-volume-method";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";
import { resolveUniformGeometricValues, UNIFORM_GEOMETRIC_NATIVE_PARAMS } from "../lib/methods/uniform/uniform-geometric-parameters";
import { methodConfigurationImpact } from "../lib/core/method-lifecycle";

test("coarse preparation rebuilds, while activation uses the runtime path", () => {
  const prepared = { prepareCoarseSimulation: "on" };
  assert.equal(uniformGeometricSolverOptions().prepareCoarseSimulation, false);
  assert.equal(uniformGeometricSolverOptions(prepared).prepareCoarseSimulation, true);
  assert.equal(methodConfigurationImpact(uniformVolumeMethod, "balanced", {}, prepared), "rebuild");
  assert.equal(methodConfigurationImpact(uniformVolumeMethod, "balanced", prepared,
    { ...prepared, coarseSimulation: "on" }), "live");
  assert.equal(methodConfigurationImpact(uniformVolumeMethod, "balanced",
    { ...prepared, coarseSimulation: "on" }, prepared), "live");
  assert.ok(uniformVolumeMethod.runtimeParamKeys?.includes("coarseSimulation"));
});

test("an unprepared configuration cannot replay a saved coarse request", () => {
  assert.equal(resolveUniformGeometricValues({ coarseSimulation: "on" }).coarseSimulation, "off");
  assert.equal(resolveUniformGeometricValues({ prepareCoarseSimulation: "off", coarseSimulation: "on" }).coarseSimulation, "off");
  assert.equal(resolveUniformGeometricValues({ prepareCoarseSimulation: "on", coarseSimulation: "on" }).coarseSimulation, "on");
  assert.ok(!UNIFORM_GEOMETRIC_NATIVE_PARAMS.some(p => p.key === "coarseSimulation" || p.key === "prepareCoarseSimulation"));
});
