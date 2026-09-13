import assert from "node:assert/strict";
import test from "node:test";
import {
  PHYSICS_EXECUTION_BACKEND_KEY,
  physicsExecutionBackendParams,
  resolvePhysicsExecutionBackend,
} from "./physics-execution-backend";

test("physics backend defaults to GPU and hides unavailable CPU physics", () => {
  assert.deepEqual(physicsExecutionBackendParams(false), []);
  assert.equal(resolvePhysicsExecutionBackend({}), "gpu");
  assert.equal(resolvePhysicsExecutionBackend({ [PHYSICS_EXECUTION_BACKEND_KEY]: "invalid" }), "gpu");
});

test("physics backend offers CPU only when its real world is available", () => {
  const [setting] = physicsExecutionBackendParams(true);
  assert.equal(setting?.kind, "select");
  assert.equal(setting?.default, "gpu");
  assert.equal(resolvePhysicsExecutionBackend({ [PHYSICS_EXECUTION_BACKEND_KEY]: "cpu" }), "cpu");
});
