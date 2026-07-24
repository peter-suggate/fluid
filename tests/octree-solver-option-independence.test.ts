import assert from "node:assert/strict";
import test from "node:test";
import { octreeMethod, octreeSolverOptions } from "../lib/methods/octree";
import { defaultScene } from "../lib/model";

test("pressure solver selection does not change octree adaptivity", () => {
  const commonValues = {
    maximumLeafSize: "16",
    interfaceRefinementBandCells: 4,
    globalFineLevelSetFactor: "4",
  };
  const galerkin = octreeSolverOptions(defaultScene, "balanced", {
    ...commonValues,
    powerPressureSolver: "galerkin",
  });
  const mgpcg = octreeSolverOptions(defaultScene, "balanced", {
    ...commonValues,
    powerPressureSolver: "section43-mgpcg",
  });

  assert.equal(galerkin.octree.powerPressureSolver, "galerkin");
  assert.equal(mgpcg.octree.powerPressureSolver, "section43-mgpcg");
  assert.equal(galerkin.octree.adaptivity, 1);
  assert.equal(mgpcg.octree.adaptivity, galerkin.octree.adaptivity);
});

test("UI exposes both pressure algorithms on the adaptive representation and defaults to Galerkin", () => {
  const selector = octreeMethod.params.find((parameter) => parameter.key === "powerPressureSolver");
  assert.ok(selector && selector.kind === "select");
  assert.equal(selector.label, "Adaptive pressure solver");
  assert.equal(selector.default, "galerkin");
  assert.deepEqual(selector.options, [
    { value: "galerkin", label: "Adaptive native-L2 Galerkin" },
    { value: "section43-mgpcg", label: "Adaptive Section 4.3 MGPCG" },
  ]);
  assert.equal(octreeMethod.params.some((parameter) => parameter.key === "adaptivity"), false);
  assert.equal(octreeMethod.presetFor("balanced").powerPressureSolver, "galerkin");
});
