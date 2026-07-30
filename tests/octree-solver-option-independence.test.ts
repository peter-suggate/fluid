import assert from "node:assert/strict";
import test from "node:test";
import { octreeMethod, octreeSolverOptions } from "../lib/methods/octree";
import { defaultScene } from "../lib/model";

test("octree exposes one production pressure authority", () => {
  const commonValues = {
    maximumLeafSize: "16",
    interfaceRefinementBandCells: 4,
    globalFineLevelSetFactor: "4",
  };
  const options = octreeSolverOptions(defaultScene, "balanced", {
    ...commonValues,
  });

  assert.equal("powerPressureSolver" in options.octree, false);
  assert.equal(options.octree.adaptivity, 1);
  assert.equal(octreeSolverOptions(defaultScene, "balanced", {
    ...commonValues, globalFineLevelSetMaximumBricks: 6_144,
  }).octree.globalFineLevelSetMaximumBricks, 6_144);
  assert.equal(octreeSolverOptions(defaultScene, "balanced", {
    ...commonValues, globalFineLevelSetMaximumBricks: 0,
  }).octree.globalFineLevelSetMaximumBricks, undefined);
});

test("UI has no retired pressure-solver selector", () => {
  const selector = octreeMethod.params.find((parameter) => parameter.key === "powerPressureSolver");
  assert.equal(selector, undefined);
  assert.equal(octreeMethod.params.some((parameter) => parameter.key === "adaptivity"), false);
  assert.equal("powerPressureSolver" in octreeMethod.presetFor("balanced"), false);
});
