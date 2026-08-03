import assert from "node:assert/strict";
import test from "node:test";
import { canonicalScene, parseScene, serializeScene, validateScene } from "../lib/model";
import { createEmptyScene, SCENE_STARTERS } from "../lib/empty-scene";
import { planSceneRuntime } from "../lib/scene-runtime";
import { sceneLatticeDimensions } from "../lib/scene-lattice";
import { environmentIds } from "../lib/environments";

test("a fresh empty scene is a valid document", () => {
  assert.deepEqual(validateScene(createEmptyScene()), []);
});

test("an empty scene owns no content and asks for no solver", () => {
  const scene = createEmptyScene();
  const plan = planSceneRuntime(scene);
  assert.equal(plan.fluidSolver, false, "systems.fluid must be an explicit false, not an omission");
  assert.equal(plan.content.rigidBodyCount, 0);
  assert.equal(plan.content.terrain, false);
  assert.equal(scene.container.fillFraction, 0);
  assert.equal(scene.fluid.inflow, undefined);
  assert.equal(scene.fluid.initialBrickSeeds_m, undefined);
  assert.equal(scene.fluid.initialDamBreakDimensions_m, undefined);
  assert.equal(scene.container.top, "open", "nothing can be dropped into a lidded room");
});

test("the empty document round-trips through the schema unchanged", () => {
  const scene = createEmptyScene();
  assert.equal(canonicalScene(parseScene(serializeScene(scene))), canonicalScene(scene));
});

test("the default extents are exactly the authored 32x24x32 lattice", () => {
  const scene = createEmptyScene();
  assert.deepEqual(sceneLatticeDimensions(scene), [32, 24, 32]);
  // The extents came out of the cell counts, so they must go back into them
  // exactly — this is the equality an axis of decimal literals loses.
  assert.equal(scene.container.width_m, 32 * scene.voxelDomain.finestCellSize_m);
  assert.equal(scene.container.height_m, 24 * scene.voxelDomain.finestCellSize_m);
  assert.equal(scene.container.depth_m, 32 * scene.voxelDomain.finestCellSize_m);
});

test("requested extents are snapped to whole cells rather than authored raw", () => {
  const scene = createEmptyScene({ extents_m: { x: 1.63, y: 1.2, z: 1.6 }, finestCellSize_m: 0.05 });
  assert.deepEqual(sceneLatticeDimensions(scene), [33, 24, 32]);
  assert.equal(scene.container.width_m, 33 * 0.05);
});

test("the created document carries its scenery, so no environment lookup remains", () => {
  const scene = createEmptyScene({ environment: "garden" });
  assert.equal(scene.environment, "garden");
  assert.ok(scene.scenery && scene.scenery.nodes.length > 0);
});

test("the room is an enclosure and a light, and nothing else", () => {
  const scene = createEmptyScene();
  const nodes = scene.scenery?.nodes ?? [];
  // Anything beyond these two is set dressing the author would have to find and
  // delete before their own scene reads.
  assert.deepEqual(nodes.map((node) => node.kind), ["room-shell", "box"]);
  const [shell, light] = nodes;
  assert.equal(shell.kind === "room-shell" && shell.materialModel, "room",
    "an unstaged room must not borrow a staged set's material model");
  assert.ok(light.tags?.includes("light"), "an unlit room is a black frame");
  assert.equal(nodes.some((node) => node.id.startsWith("cyc/") || node.id.startsWith("calibration/")), false,
    "the calibration studio's cyclorama and step wedge belong to scenes that read numbers off water");
});

test("a fresh scene has no tank in it", () => {
  // The container is still the solver's boundary; it is simply not drawn. An
  // aquarium standing in an empty room presumes the scene is about water.
  const scene = createEmptyScene();
  assert.equal(scene.container.vessel, "none");
  assert.deepEqual(validateScene({ ...scene, container: { ...scene.container, vessel: "glass" } }), [],
    "adding one back is an ordinary edit, not a different kind of document");
});

test("every starter is a distinct, valid, fluid-free room", () => {
  const ids = new Set(SCENE_STARTERS.map((starter) => starter.id));
  assert.equal(ids.size, SCENE_STARTERS.length, "starter IDs must be unique");
  assert.deepEqual([...ids], ["empty-room", "small-room", "hall"]);
  const lattices = new Set<string>();
  for (const starter of SCENE_STARTERS) {
    const scene = starter.create();
    assert.deepEqual(validateScene(scene), [], `${starter.id} must validate`);
    assert.equal(planSceneRuntime(scene).fluidSolver, false, `${starter.id} must stay fluid-free`);
    assert.equal(scene.rigidBodies.length, 0, `${starter.id} must be empty`);
    assert.equal(scene.terrain, undefined, `${starter.id} must not author terrain`);
    assert.equal(scene.container.vessel, "none", `${starter.id} must not stand a tank in the room`);
    assert.ok(scene.environment && environmentIds.includes(scene.environment), `${starter.id} names a real environment`);
    assert.ok(starter.blurb.length > 0 && starter.name.length > 0);
    // The lattice is the structural tier of the solver key and the one thing a
    // starter chooses on the author's behalf, so two starters that resolve to
    // the same domain would be offering the same choice twice.
    lattices.add(sceneLatticeDimensions(scene).join("x"));
    for (const axis of sceneLatticeDimensions(scene)) {
      assert.equal(axis % 8, 0, `${starter.id} must be whole 8-cell bricks on every axis`);
    }
  }
  assert.equal(lattices.size, SCENE_STARTERS.length);
});
