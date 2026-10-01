import type {MethodParamValues} from "../lib/core/method-contract";
import assert from "node:assert/strict";
import test from "node:test";
import {getSceneDefinition} from "../lib/core/scenes";
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

test("smooth surface coarsening is disabled, including saved experimental settings", () => {
  for (const values of [{}, {coarseningSurfaceTolerance:0.5}, {coarseningSurfaceTolerance:1}] as MethodParamValues[]) {
    assert.equal("coarseningSurfaceTolerance" in resolveUniformGeometricValues(values),false);
    assert.equal(uniformGeometricSolverOptions(values).mixedCoarseningSurfaceTolerance,0);
  }
  assert.ok(!uniformVolumeMethod.runtimeParamKeys?.includes("coarseningSurfaceTolerance"));
  assert.ok(!uniformVolumeMethod.params.some(p=>p.key==="coarseningSurfaceTolerance"));
  assert.ok(!UNIFORM_GEOMETRIC_NATIVE_PARAMS.some(p=>p.key==="coarseningSurfaceTolerance"));
});

test("the authored still-pond profile preserves fine-surface hydrostatic balance",()=>{
  const profile=getSceneDefinition("hero-garden-hose-x10").methodProfile!;
  assert.equal(profile.methodId,"uniform-volume");
  assert.equal(uniformGeometricSolverOptions(profile.overrides).mixedCoarseningSurfaceTolerance,0);
  assert.equal(resolveUniformGeometricValues(profile.overrides).pressureResidualTolerance,1e-3);
});
