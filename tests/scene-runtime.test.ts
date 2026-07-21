import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene } from "../lib/model";
import { planSceneRuntime } from "../lib/scene-runtime";

function sceneWithBodies(motion: "dynamic" | "static") {
  const scene = cloneScene(defaultScene);
  scene.rigidBodies = [{
    id: `body-${motion}`,
    name: `${motion} body`,
    shape: "sphere",
    dimensions_m: { x: 0.1, y: 0.1, z: 0.1 },
    density_kg_m3: 1_000,
    position_m: { x: 0, y: 0.2, z: 0 },
    orientation: { w: 1, x: 0, y: 0, z: 0 },
    linearVelocity_m_s: { x: 0, y: 0, z: 0 },
    angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
    restitution: 0,
    friction: 0,
    motion,
  }];
  return scene;
}

test("omitted fluid declaration preserves the legacy solver even at zero fill", () => {
  const scene = cloneScene(defaultScene);
  delete scene.systems;
  scene.container.fillFraction = 0;
  delete scene.fluid.initialBrickSeeds_m;
  delete scene.fluid.inflow;

  const plan = planSceneRuntime(scene);
  assert.equal(plan.content.fluidEnabled, true);
  assert.equal(plan.fluidSolver, true);
  assert.equal(plan.capabilities["fluid-authority"], true);
  assert.equal(plan.readiness.fluidAuthority.state, "required");
  assert.equal(plan.readiness.transport.state, "required");
});

test("static rigid bodies are content but do not request dynamics or coupling", () => {
  const plan = planSceneRuntime(sceneWithBodies("static"));
  assert.equal(plan.content.rigidBodyCount, 1);
  assert.equal(plan.content.dynamicRigidBodyCount, 0);
  assert.equal(plan.capabilities["rigid-dynamics"], false);
  assert.equal(plan.rigidCoupling, false);
});

test("dynamic bodies in a fluid scene request dynamics and fluid coupling", () => {
  const plan = planSceneRuntime(sceneWithBodies("dynamic"));
  assert.equal(plan.capabilities["rigid-dynamics"], true);
  assert.equal(plan.capabilities["fluid-rigid-coupling"], true);
  assert.equal(plan.rigidCoupling, true);
  assert.deepEqual(plan.readiness.transport.requires, [
    "fluid-authority",
    "water-presentation",
    "rigid-dynamics",
    "fluid-rigid-coupling",
  ]);
});

test("dry static scenes mark fluid and transport gates not required", () => {
  const scene = sceneWithBodies("static");
  scene.systems = { fluid: false };
  scene.container.fillFraction = 0;

  const plan = planSceneRuntime(scene, { methodId: "octree" });
  assert.equal(plan.fluidSolver, false);
  assert.equal(plan.capabilities["water-presentation"], false);
  assert.equal(plan.readiness.fluidAuthority.state, "not-required");
  assert.deepEqual(plan.readiness.fluidAuthority.requires, []);
  assert.equal(plan.readiness.transport.state, "not-required");
  assert.deepEqual(plan.readiness.transport.requires, []);
  assert.equal(plan.readiness.presentation.state, "required");
  assert.deepEqual(plan.readiness.presentation.requires, ["static-world", "sparse-voxel-presentation"]);
});
