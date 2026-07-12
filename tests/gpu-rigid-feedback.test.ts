import assert from "node:assert/strict";
import test from "node:test";
import { consumeGPURigidLoad, mergeGPURigidLoads, type GPURigidLoad } from "../lib/webgpu-eulerian";

const load = (impulse: number, interval: number): GPURigidLoad => ({
  bodyId: "body",
  impulse_N_s: { x: impulse, y: 0, z: 0 },
  angularImpulse_N_m_s: { x: 0, y: impulse / 2, z: 0 },
  couplingInterval_s: interval,
  displacedVolume_m3: 0.001
});

test("GPU rigid impulse is consumed once over its source interval", () => {
  const pending = load(0.8, 0.008);
  const first = consumeGPURigidLoad(pending, 0.004), second = consumeGPURigidLoad(pending, 0.004), third = consumeGPURigidLoad(pending, 0.004);
  assert.ok(Math.abs(first.impulse_N_s.x - 0.4) < 1e-12);
  assert.ok(Math.abs(second.impulse_N_s.x - 0.4) < 1e-12);
  assert.equal(third.impulse_N_s.x, 0);
  assert.ok(Math.abs(first.impulse_N_s.x + second.impulse_N_s.x - 0.8) < 1e-12);
});

test("GPU readbacks queue impulses instead of replacing or repeating them", () => {
  const [pending] = mergeGPURigidLoads([load(0.25, 0.004)], [load(0.35, 0.004)]);
  assert.ok(Math.abs(pending.impulse_N_s.x - 0.6) < 1e-12);
  assert.ok(Math.abs(pending.couplingInterval_s - 0.008) < 1e-12);
  const delivered = consumeGPURigidLoad(pending, 0.004).impulse_N_s.x + consumeGPURigidLoad(pending, 0.004).impulse_N_s.x;
  assert.ok(Math.abs(delivered - 0.6) < 1e-12);
});
