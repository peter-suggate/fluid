import assert from "node:assert/strict";
import test from "node:test";
import { createSvoDrySceneFragmentWGSL } from "../program";
import { shadingExperimentSource } from "../../../../../tools/svo-shading-experiment";

test("production specialized source matches the benchmarked closure", () => {
  const generate = (fast: boolean) => createSvoDrySceneFragmentWGSL(.5, "raster-primary", "off", "split", 0, false, false, true,
    { surfaceMesh: true, voxelLightCache: false, globalIlluminationAbsent: true, opaqueDirectionalCones: fast });
  const normalize = (source: string) => source.replace(/\/\/[^\n]*/g, "").replace(/\s+/g, "");
  const reference = shadingExperimentSource(generate(false), "cone-only-opaque-one-light-visibility-guide");
  assert.equal(normalize(generate(true)), normalize(reference));
});
