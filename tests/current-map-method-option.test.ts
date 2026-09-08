import assert from "node:assert/strict";
import test from "node:test";
import { resolveMethodValues } from "../lib/core/method-contract";
import { createCoarseFirstPoolImpactQuarterScene, getScenePreset } from "../lib/core/scenes";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice-dimensions";
import { adaptiveMassMethod, adaptiveMassSolverOptions } from "../lib/methods/adaptive-mass/method";
import { bindRetainedSceneSupportLattice, compileRetainedSceneDensity, packRetainedSceneDensity,
  retainedSceneDensity } from "../lib/methods/adaptive-mass/sparse-cm12-retained-scene-density";

test("production quarter keeps coarse-first with the previous native surface and paper timestep", () => {
  const preset = getScenePreset("coarse-first-pool-impact-quarter");
  const profile = preset.methodProfile!;
  assert.equal(profile.methodId, "adaptive-mass");
  const values = resolveMethodValues(adaptiveMassMethod, profile.quality, profile.overrides);
  assert.equal(values.densityTransport, "native-cm12");
  assert.equal(values.selectorMode, "coarse-first");
  assert.equal(values.timeStep, "paper");
  assert.equal(preset.create().numerics.fixedDt_s, 1 / 30);
});

test("current spatial field selection rebuilds the solver and preserves the validation default", () => {
  const param = adaptiveMassMethod.params.find(param => param.key === "densityTransport");
  assert.ok(param);
  assert.equal(param.update, "solver");
  assert.equal(param.default, "native-cm12");
  assert.ok(!adaptiveMassMethod.runtimeParamKeys?.includes("densityTransport"));
  assert.equal(adaptiveMassSolverOptions(resolveMethodValues(adaptiveMassMethod, "balanced", {})).densityTransport,
    "native-cm12");
  const selected = resolveMethodValues(adaptiveMassMethod, "balanced", { densityTransport: "current-map" });
  assert.equal(selected.densityTransport, "current-map");
  assert.equal(adaptiveMassSolverOptions(selected).densityTransport, "current-map");
  const retained = resolveMethodValues(adaptiveMassMethod, "balanced", { densityTransport: "retained-cm12" });
  assert.equal(adaptiveMassSolverOptions(retained).densityTransport, "retained-cm12");
});

test("current-map authoring keeps sphere/pool geometry and survives immutable support binding", () => {
  const scene = createCoarseFirstPoolImpactQuarterScene();
  const legacy = compileRetainedSceneDensity(scene);
  const current = compileRetainedSceneDensity(scene, { transport: "current-map" });
  assert.ok(legacy && current);
  assert.equal(legacy.transport, undefined);
  assert.equal(current.transport, "current-map");
  assert.deepEqual(packRetainedSceneDensity(current), packRetainedSceneDensity(legacy));
  const bound = bindRetainedSceneSupportLattice(current, sceneLatticeDimensions(scene), scene.voxelDomain.finestCellSize_m);
  const cloned = retainedSceneDensity({ ...bound, generation: bound.generation + 1 });
  assert.equal(bound.transport, "current-map");
  assert.equal(cloned.transport, "current-map");
  assert.ok(Object.isFrozen(bound) && Object.isFrozen(cloned));
  assert.deepEqual(cloned.primitives, legacy.primitives);
});
