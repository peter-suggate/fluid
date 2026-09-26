import type {MethodParamValues} from "../lib/core/method-contract";
import assert from "node:assert/strict";
import test from "node:test";
import { uniformVolumeMethod } from "../lib/methods/uniform/uniform-volume-method";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";
import { resolveUniformGeometricValues, UNIFORM_GEOMETRIC_NATIVE_PARAMS } from "../lib/methods/uniform/uniform-geometric-parameters";

test("Uniform production resolution has no preparation or activation parameters", () => {
  for(const values of [{}, {prepareCoarseSimulation:"on",coarseSimulation:"on"}] as MethodParamValues[]){
    const resolved=resolveUniformGeometricValues(values);
    assert.equal("prepareCoarseSimulation" in resolved,false);
    assert.equal("coarseSimulation" in resolved,false);
    assert.equal(uniformGeometricSolverOptions(values).prepareCoarseSimulation,undefined);
  }
  assert.ok(!uniformVolumeMethod.runtimeParamKeys?.includes("coarseSimulation"));
  assert.ok(!UNIFORM_GEOMETRIC_NATIVE_PARAMS.some(p=>p.key==="coarseSimulation"||p.key==="prepareCoarseSimulation"));
});

test("saved prototype mixed settings cannot select the retired solver or discard native controls", () => {
  const values = resolveUniformGeometricValues({ prepareMixedSimulation: "on", mixedSimulation: "on", sharpeningStrength: .375 });
  assert.equal('mixedSimulation' in values, false);
  assert.equal('prepareMixedSimulation' in values, false);
  assert.equal("coarseSimulation" in values, false);
  const options = uniformGeometricSolverOptions(values);
  assert.equal('prepareMixedSimulation' in options, false);
  assert.equal(options.prepareCoarseSimulation, undefined);
  assert.equal(options.geometricVolume, true);
  assert.equal(options.sharpeningStrength, .375);
});
