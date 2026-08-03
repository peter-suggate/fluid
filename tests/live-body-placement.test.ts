import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { cloneScene, type RigidBodyDescription, type SceneDescription } from "../lib/model";
import { getScenePreset } from "../lib/scenes";
import { adoptRigidBodyRoster, initializeRigidBodies } from "../lib/rigid-body";
import { sceneEditRequiresReset } from "../lib/webgpu-renderer";

const base = getScenePreset("dam-break-boxes").create();

/** A body two seconds into a run: moved, turned and moving. */
function settled(scene: SceneDescription) {
  const bodies = initializeRigidBodies(scene.rigidBodies.map((body) => ({ ...body })));
  for (const body of bodies) {
    body.position_m = { x: body.position_m.x, y: 0.05, z: body.position_m.z };
    body.linearVelocity_m_s = { x: 0.3, y: -0.1, z: 0 };
    body.angularVelocity_rad_s = { x: 0, y: 0.4, z: 0 };
  }
  return bodies;
}

function dropped(scene: SceneDescription, id = "dropped-in"): RigidBodyDescription {
  return { ...cloneScene(scene).rigidBodies[0], id, position_m: { x: 0, y: 0.7, z: 0 } };
}

test("moving a body in running water never resets the clock", () => {
  // At t = 2 s, dragging or reshaping something must leave the simulation
  // running rather than restart it at t = 0.
  const nudged = (patch: Partial<RigidBodyDescription>) =>
    ({ ...base, rigidBodies: [{ ...base.rigidBodies[0], ...patch }, ...base.rigidBodies.slice(1)] });
  for (const after of [
    nudged({ position_m: { x: 0.2, y: 0.4, z: 0 } }),
    nudged({ dimensions_m: { x: 0.15, y: 0.15, z: 0.15 } }),
    nudged({ friction: 0.2 }),
  ]) assert.equal(sceneEditRequiresReset(base, after), false);
});

test("the edits that genuinely cannot be applied live still say so", () => {
  const solidFree = { ...base, rigidBodies: [] };
  const cases: ReadonlyArray<[string, SceneDescription, SceneDescription]> = [
    // Environment proxies are numbered `rigidBodies.length + ownerIndex` by the
    // render source, so the roster length is part of its ABI: adding a body
    // renumbers every scenery object. Decoupling that — pinning the base at the
    // fixed rigid capacity — is what would make this one warm.
    ["a body arriving", base, { ...base, rigidBodies: [...base.rigidBodies, dropped(base)] }],
    ["the first body in a solid-free scene", solidFree, { ...solidFree, rigidBodies: [dropped(base)] }],
    ["the last body leaving", base, solidFree],
    ["a resized tank", base, { ...base, container: { ...base.container, width_m: base.container.width_m + 0.4 } }],
    ["painted water", base, { ...base, fluid: { ...base.fluid, initialBrickSeeds_m: [{ x: 0, y: 0.1, z: 0 }] } }],
    ["sculpted terrain", base, { ...base, terrain: { baseHeight_m: 0.1, features: [] } }],
  ];
  for (const [label, before, after] of cases) assert.equal(sceneEditRequiresReset(before, after), true, label);
});

test("a body already in flight keeps the state it has reached", () => {
  const live = settled(base);
  const adopted = adoptRigidBodyRoster(live, [...base.rigidBodies, dropped(base)]);
  assert.equal(adopted.length, base.rigidBodies.length + 1);
  for (let index = 0; index < live.length; index += 1) {
    assert.equal(adopted[index], live[index],
      "an untouched body must be the same object, not a re-derived copy of its authored pose");
  }
  const newcomer = adopted[adopted.length - 1];
  assert.equal(newcomer.position_m.y, 0.7, "the new body starts where it was dropped");
  assert.ok(newcomer.mass_kg > 0, "and with mass properties of its own");
});

test("a reshaped body is re-derived where it fell, not where it was authored", () => {
  const live = settled(base);
  const grown: RigidBodyDescription = { ...base.rigidBodies[0], dimensions_m: { x: 0.2, y: 0.2, z: 0.2 } };
  const [adopted] = adoptRigidBodyRoster(live, [grown, ...base.rigidBodies.slice(1)]);
  assert.notEqual(adopted, live[0], "a changed description must re-derive mass properties");
  assert.deepEqual(adopted.position_m, live[0].position_m, "but not teleport the body back to its authored pose");
  assert.deepEqual(adopted.linearVelocity_m_s, live[0].linearVelocity_m_s);
  assert.notEqual(adopted.mass_kg, live[0].mass_kg, "a bigger box at the same density is heavier");
  // `syncBodies` signs a body with its *authored* transform; only the simulated
  // state above came from the run.
  assert.deepEqual(adopted.description.position_m, grown.position_m);
  assert.deepEqual(adopted.description.dimensions_m, grown.dimensions_m);
});

test("a removed body leaves, and an unrelated roster is untouched", () => {
  const live = settled(base);
  const kept = adoptRigidBodyRoster(live, base.rigidBodies.slice(1));
  assert.equal(kept.length, base.rigidBodies.length - 1);
  assert.equal(kept.some((body) => body.description.id === base.rigidBodies[0].id), false);
  assert.deepEqual(adoptRigidBodyRoster(live, base.rigidBodies), live,
    "an edit that touched no body must produce the identical roster");
});

test("a commit only resets when the edit requires it", () => {
  // The source guard is the point of the phase: `commitEdit` used to reset
  // unconditionally, so every gesture threw away the run in order to apply it.
  const controller = readFileSync(new URL("../lib/simulation/controller.ts", import.meta.url), "utf8");
  assert.match(controller, /options\.reseed && sceneEditRequiresReset\(pending\.snapshot\.scene, committed\)/);
  assert.match(controller, /else this\.adoptRigidBodies\(committed\)/,
    "an edit that skips the reset must still reconcile the roster the solver is stepping");
});
