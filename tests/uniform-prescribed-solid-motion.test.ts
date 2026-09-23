import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene } from "../lib/core/model";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { UniformPrescribedSolidMotion } from "../lib/methods/uniform/uniform-prescribed-solid-motion";

const fixture = () => {
  const scene = cloneScene(defaultScene);
  assert.ok(scene.rigidBodies.length);
  return initializeRigidBodies(scene.rigidBodies)[0]!;
};

test("held pose commands supply translation and rotation for one simulation step", () => {
  const motion = new UniformPrescribedSolidMotion(), body = fixture();
  body.held = true; body.orientation = { w: 1, x: 0, y: 0, z: 0 };
  motion.sample([body], 0);
  body.position_m.x += 0.05;
  body.orientation = { w: Math.cos(0.1), x: 0, y: Math.sin(0.1), z: 0 };
  body.linearVelocity_m_s = { x: 6, y: 0, z: 0 }; // stale pointer-event speed
  const moved = motion.sample([body], 0.1)[0]!;
  assert.ok(Math.abs(moved.linearVelocity_m_s.x - 0.5) < 1e-12);
  assert.ok(Math.abs(moved.angularVelocity_rad_s.y - 2) < 1e-12);
  assert.equal(body.linearVelocity_m_s.x, 6, "caller roster is untouched");
  const stopped = motion.sample([body], 0.1)[0]!;
  assert.deepEqual(stopped.linearVelocity_m_s, { x: 0, y: 0, z: 0 });
  assert.ok(Math.hypot(...Object.values(stopped.angularVelocity_rad_s)) < 1e-12);
});

test("free GPU poses and the first grab are not differentiated from stale CPU state", () => {
  const motion = new UniformPrescribedSolidMotion(), body = fixture();
  body.description.motion = "dynamic"; body.held = false;
  motion.sample([body], 0);
  body.position_m.x += 10;
  assert.equal(motion.sample([body], 0.1)[0], body);
  body.held = true; body.position_m.x += 10;
  assert.equal(motion.sample([body], 0.1)[0], body);
  motion.sample([], 0.1);
  assert.equal(motion.sample([body], 0.1)[0], body, "removed IDs do not retain stale history");
});

test("authored static edits differentiate poses and quaternion sign flips do not spin", () => {
  const motion = new UniformPrescribedSolidMotion(), body = fixture();
  body.description.motion = "static"; body.held = false;
  body.orientation = { w: 1, x: 0, y: 0, z: 0 }; motion.sample([body], 0);
  body.position_m.z += 0.1; body.orientation = { w: -1, x: 0, y: 0, z: 0 };
  const moved = motion.sample([body], 0.05)[0]!;
  assert.ok(Math.abs(moved.linearVelocity_m_s.z - 2) < 1e-12);
  assert.ok(Math.hypot(...Object.values(moved.angularVelocity_rad_s)) < 1e-12);
});
