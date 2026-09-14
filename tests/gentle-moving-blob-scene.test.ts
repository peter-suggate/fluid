import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { createGentleMovingBlobScene } from "../lib/core/gentle-moving-blob-scene";
import { initialLiquidFractionAtCell } from "../lib/core/initial-fluid";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneDocument } from "../lib/core/scene-definition";
import { validateScene } from "../lib/core/model";

test("gentle moving blob is a fully contained spherical transport probe", () => {
  const scene = createGentleMovingBlobScene();
  assert.deepEqual(validateScene(scene), []);
  assert.deepEqual(sceneLatticeDimensions(scene), [32, 24, 16]);
  assert.deepEqual(scene.fluid.initialVelocity_m_s, { x: 0.08, y: 0, z: 0 });
  assert.deepEqual(scene.fluid.gravity_m_s2, { x: 0, y: 0, z: 0 });
  assert.equal(scene.fluid.dynamicViscosity_Pa_s, 0);
  assert.equal(scene.fluid.surfaceTension_N_m, 0);
  assert.equal(scene.numerics.fixedDt_s, 1 / 30);
  assert.equal(scene.numerics.maxDt_s, 1 / 30);
  assert.equal(scene.fluid.refinementRegions, undefined);

  const sphere = scene.fluid.initialLiquidVolumes?.[0];
  assert.ok(sphere && sphere.shape === "sphere");
  assert.deepEqual(sphere.center_m, { x: -0.25, y: 0.6, z: 0 });
  assert.equal(sphere.radius_m, 0.3);
  const domain = {
    min: { x: -scene.container.width_m / 2, y: 0, z: -scene.container.depth_m / 2 },
    max: {
      x: scene.container.width_m / 2,
      y: scene.container.height_m,
      z: scene.container.depth_m / 2,
    },
  };
  for (const axis of ["x", "y", "z"] as const) {
    assert.ok(sphere.center_m[axis] - sphere.radius_m > domain.min[axis]);
    assert.ok(sphere.center_m[axis] + sphere.radius_m < domain.max[axis]);
  }
  assert.ok(Math.abs(domain.max.z - (sphere.center_m.z + sphere.radius_m) - 0.1) < 1e-12);
  assert.ok(Math.abs(sphere.center_m.z - sphere.radius_m - domain.min.z - 0.1) < 1e-12);
});

test("gentle moving blob source occupies an interior, z-symmetric lattice extent", () => {
  const scene = sceneDocument(getSceneDefinition("gentle-moving-blob"));
  const dimensions = sceneLatticeDimensions(scene);
  const occupied: Array<readonly [number, number, number, number]> = [];
  for (let z = 0; z < dimensions[2]; z++) {
    for (let y = 0; y < dimensions[1]; y++) {
      for (let x = 0; x < dimensions[0]; x++) {
        const fraction = initialLiquidFractionAtCell(scene, x, y, z, dimensions, false);
        if (fraction > 0) occupied.push([x, y, z, fraction]);
      }
    }
  }
  assert.ok(occupied.length > 0);
  const zValues = occupied.map((sample) => sample[2]);
  assert.equal(Math.min(...zValues), 2);
  assert.equal(Math.max(...zValues), 13);
  assert.ok(Math.min(...zValues) > 0 && Math.max(...zValues) < dimensions[2] - 1);

  const byZ = new Map<number, number>();
  for (const [, , z, fraction] of occupied) byZ.set(z, (byZ.get(z) ?? 0) + fraction);
  for (let z = 0; z < dimensions[2]; z++) {
    assert.ok(Math.abs((byZ.get(z) ?? 0) - (byZ.get(dimensions[2] - 1 - z) ?? 0)) < 1e-12);
  }
});
