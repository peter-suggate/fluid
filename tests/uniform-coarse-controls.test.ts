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
    assert.equal("prepareCoarseSimulation" in uniformGeometricSolverOptions(values),false);
  }
  assert.ok(!uniformVolumeMethod.runtimeParamKeys?.includes("coarseSimulation"));
  assert.ok(!UNIFORM_GEOMETRIC_NATIVE_PARAMS.some(p=>p.key==="coarseSimulation"||p.key==="prepareCoarseSimulation"));
});
