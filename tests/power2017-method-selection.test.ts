import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { getMethod, resolveMethodValues } from "../lib/methods";
import { octreeCoarseBackendPolicy } from "../lib/octree-coarse-backend";
import { octreeSolverOptions } from "../lib/methods/octree";
import { getSceneWebGPUSmokeLane } from "../lib/scene-webgpu-smoke-catalog";
import {
  createSymmetricExpansionScene,
  POWER2017_FACTOR4_BENCHMARK_METHOD_PROFILE,
} from "../lib/scenes";
import { parseQueryState, serializeQueryState } from "../lib/url-state";

const methodPanelSource = readFileSync(
  new URL("../components/MethodPanel.tsx", import.meta.url), "utf8",
);

test("selecting Power 2017 resolves one canonical factor-4 benchmark tuple", () => {
  const method = getMethod("octree");
  const losasso = resolveMethodValues(method, "balanced", {});
  assert.equal(losasso.coarseBackend, "losasso");
  assert.equal(losasso.globalFineLevelSetFactor, "1",
    "adding the benchmark must not change the Losasso product default");

  const power = resolveMethodValues(method, "balanced", {
    coarseBackend: "power2017",
    globalFineLevelSetFactor: "1",
    topologyCadenceAdvances: 8,
  });
  assert.equal(power.coarseBackend, "power2017");
  assert.equal(power.globalFineLevelSetFactor, "4");
  assert.equal(power.topologyCadenceAdvances, 1);
  assert.equal(octreeCoarseBackendPolicy("power2017").requiresSeparateFineLevelSet, true);

  const options = octreeSolverOptions(createSymmetricExpansionScene(), "balanced", power);
  assert.equal(options.coarseDynamics.backend, "power2017");
  assert.equal(options.coarseDynamics.pressureExecutor, "persistent-power-mgpcg");
  assert.equal(options.octree.globalFineLevelSetFactor, 4);
  assert.equal(options.coarseDynamics.topology.advancesPerEpoch, 1);
});

test("a Power selection survives a shared-link round trip as factor four", () => {
  const parsed = parseQueryState(
    "?scene=symmetric-expansion&param.octree.coarseBackend=power2017",
  );
  const initialValues = resolveMethodValues(getMethod(parsed.methodId), parsed.quality,
    parsed.overrides[parsed.methodId] ?? {});
  assert.equal(initialValues.coarseBackend, "power2017");
  assert.equal(initialValues.globalFineLevelSetFactor, "4");

  const query = serializeQueryState("", {
    presetId: parsed.presetId,
    scene: parsed.scene,
  }, {
    methodId: parsed.methodId,
    quality: parsed.quality,
    overrides: parsed.overrides,
  }, parsed.ui);
  const resumed = parseQueryState(`?${query}`);
  const resumedValues = resolveMethodValues(getMethod(resumed.methodId), resumed.quality,
    resumed.overrides[resumed.methodId] ?? {});
  assert.equal(resumedValues.coarseBackend, "power2017");
  assert.equal(resumedValues.globalFineLevelSetFactor, "4");
});

test("symmetric expansion has a dedicated Power lane beside the Losasso lane", () => {
  assert.deepEqual(POWER2017_FACTOR4_BENCHMARK_METHOD_PROFILE.overrides, {
    coarseBackend: "power2017",
    maximumLeafSize: "32",
    interfaceRefinementBandCells: 4,
    globalFineLevelSetFactor: "4",
    topologyCadenceAdvances: 1,
  });
  const power = getSceneWebGPUSmokeLane("symmetric-expansion", "power2017-factor-4");
  assert.deepEqual(power.methods[0]?.overrides,
    POWER2017_FACTOR4_BENCHMARK_METHOD_PROFILE.overrides);
  assert.equal(power.collect.raster, "initial-final");
  assert.equal(power.collect.globalFineGeneration, true);
  assert.equal(power.stop.exactSteps, 3);
  assert.equal(power.stop.simulatedTime_s, 0.012);
  assert.ok(power.acceptance.some(({ id }) => id === "power2017-coarse-backend"));

  const losasso = getSceneWebGPUSmokeLane("symmetric-expansion", "fine-factor-4");
  assert.equal(losasso.methods[0]?.overrides.coarseBackend, "losasso");
  assert.match(methodPanelSource, /disabled=\{fixedPowerFineBand\}/);
});
