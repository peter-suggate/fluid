import assert from "node:assert/strict";
import test from "node:test";
import { cameraBasis, dot, length, mulberry32, orbit, pan, zoom } from "../lib/math";
import { canonicalScene, cloneScene, createRunManifest, defaultCamera, defaultScene, parseScene, serializeScene, validateScene } from "../lib/model";
import { runShellValidation } from "../lib/validation";

test("S2-01 default scene passes the SI scene contract", () => {
  assert.deepEqual(validateScene(defaultScene), []);
});

test("S2-02 scene serialization is byte-stable after round-trip", () => {
  const serialized = serializeScene(defaultScene);
  assert.equal(serializeScene(parseScene(serialized)), serialized);
  assert.equal(canonicalScene(parseScene(serialized)), canonicalScene(defaultScene));
});

test("S2-03 camera reset is immutable and deterministic", () => {
  const moved = orbit(defaultCamera, 20, -10);
  assert.notDeepEqual(moved, defaultCamera);
  assert.deepEqual(defaultCamera, {
    azimuth_rad: 0.72,
    elevation_rad: 0.42,
    distance_m: 2.65,
    target_m: { x: 0, y: 0.38, z: 0 }
  });
});

test("S2-04 camera basis is orthonormal after orbit, pan, and zoom", () => {
  const cameras = [defaultCamera, orbit(defaultCamera, 110, -70), pan(defaultCamera, 40, -20), zoom(defaultCamera, 850)];
  for (const camera of cameras) {
    const { forward, right, up } = cameraBasis(camera);
    assert.ok(Math.abs(length(forward) - 1) < 1e-10);
    assert.ok(Math.abs(length(right) - 1) < 1e-10);
    assert.ok(Math.abs(length(up) - 1) < 1e-10);
    assert.ok(Math.abs(dot(forward, right)) < 1e-10);
    assert.ok(Math.abs(dot(forward, up)) < 1e-10);
    assert.ok(Math.abs(dot(right, up)) < 1e-10);
  }
});

test("S2-05 seeded random stream is reproducible", () => {
  const a = mulberry32(9);
  const b = mulberry32(9);
  const c = mulberry32(10);
  const streamA = Array.from({ length: 1000 }, a);
  assert.deepEqual(streamA, Array.from({ length: 1000 }, b));
  assert.notDeepEqual(streamA, Array.from({ length: 1000 }, c));
});

test("S2-06 invalid physical fields are rejected without changing thresholds", () => {
  const scene = cloneScene(defaultScene);
  scene.container.width_m = -1;
  scene.container.fillFraction = 1.01;
  scene.fluid.density_kg_m3 = 0;
  scene.fluid.dynamicViscosity_Pa_s = -1;
  scene.nominalResolution.length_m = 0;
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s * 2;
  assert.ok(validateScene(scene).length >= 6);
});

test("S2-10 exported run manifest carries reproducibility identity", () => {
  const manifest = createRunManifest(defaultScene, "compare", "test-adapter");
  assert.equal(manifest.runSchemaVersion, "1.0.0");
  assert.equal(manifest.solverMode, "compare");
  assert.deepEqual(manifest.precision, { cpu: "binary64", gpu: "f32" });
  assert.equal(manifest.webgpuAdapter, "test-adapter");
  assert.equal(manifest.scene.randomSeed, defaultScene.randomSeed);
});

test("in-app validation agrees with the unit contract", () => {
  const results = runShellValidation();
  assert.ok(results.length >= 5);
  assert.ok(results.every((result) => result.passed), JSON.stringify(results, null, 2));
});
