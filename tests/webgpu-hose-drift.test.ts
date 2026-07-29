import assert from "node:assert/strict";
import test from "node:test";
import { createPaperScenario } from "../lib/paper-scenarios";
import { auditHoseJetDrift, hoseJetDriftFailures } from "../tools/webgpu-hose-drift";

test("hose drift audit separates gravity bend from nozzle-local sideways drift", () => {
  const scene = createPaperScenario("hose-tank"), inflow = scene.fluid.inflow!;
  // Rotate away from every grid axis. The audit frame must follow the authored
  // nozzle, not select a dominant Cartesian component.
  inflow.center_m = { x: 0, y: 0.45, z: 0 };
  inflow.velocity_m_s = { x: 1, y: 1, z: 0 };
  const dimensions = [6, 6, 6] as const;
  const volume = new Float32Array(6 ** 3), velocity = new Float32Array(3 * volume.length);
  for (let z = 2; z <= 3; z += 1) for (let y = 3; y <= 4; y += 1) for (let x = 3; x <= 4; x += 1) {
    const cell = x + 6 * (y + 6 * z);
    volume[cell] = 1;
    velocity.set([1, 1, 0], 3 * cell);
  }
  const audit = auditHoseJetDrift(volume, velocity, dimensions, scene.container,
    inflow, { ...inflow.center_m }, scene.fluid.gravity_m_s2);
  assert.ok(Math.abs(audit.outletSideSpeed_m_s) < 1e-12);
  assert.ok(Math.abs(audit.outletAngleError_deg) < 1e-6);
  assert.ok(audit.maximumSideCentroidOffset_m <= scene.container.depth_m / dimensions[2] / 2 + 1e-12);
  const momentumPassing = { sampledAirborneBins: 4, minimumAirborneAxialRetentionRatio: 0.95,
    maximumAdjacentAxialSpeedDropRatio: 0.1, maximumAdjacentMomentumFluxDropRatio: 0.2,
    minimumBallisticGravityVelocityRatio: 0.95, maximumBallisticGravityVelocityRatio: 1.05,
    maximumBallisticCenterlineRelativeError: 0.1 };
  assert.deepEqual(hoseJetDriftFailures({ ...audit, maximumSideCentroidOffset_m: 0,
    ...momentumPassing }, 0.02), []);
  assert.deepEqual(hoseJetDriftFailures({ ...audit, outletSideSpeed_m_s: 4.9,
    maximumSideCentroidOffset_m: 0.19, ...momentumPassing }, 0.02), [
    "outlet sideways speed 4.9 m/s exceeds 0.10 m/s",
    "jet sideways centroid 0.19 m exceeds 0.01 m",
  ]);
  assert.ok(hoseJetDriftFailures({ ...audit, maximumSideCentroidOffset_m: 0,
    ...momentumPassing, maximumBallisticGravityVelocityRatio: 3.5 }, 0.02)
    .some((failure) => failure.includes("velocity amplification")));
});

test("hose drift audit rejects abrupt airborne forward-momentum loss", () => {
  const scene = createPaperScenario("hose-tank"), inflow = scene.fluid.inflow!;
  const dimensions = [40, 30, 20] as const;
  const volume = new Float32Array(dimensions[0] * dimensions[1] * dimensions[2]);
  const velocity = new Float32Array(3 * volume.length);
  const outlet = { x: -0.34, y: inflow.center_m.y, z: 0 };
  for (let x = 9; x < 21; x += 1) for (let y = 16; y < 20; y += 1) for (let z = 8; z < 12; z += 1) {
    const cell = x + dimensions[0] * (y + dimensions[1] * z);
    volume[cell] = 1;
    const worldX = -0.5 * scene.container.width_m + (x + 0.5) * scene.container.width_m / dimensions[0];
    const axial = worldX - outlet.x;
    velocity[3 * cell] = axial < 0.12 ? 0.8 : 0.35;
  }
  const audit = auditHoseJetDrift(volume, velocity, dimensions, scene.container,
    inflow, outlet, scene.fluid.gravity_m_s2);
  assert.ok(audit.sampledAirborneBins >= 3);
  assert.ok(audit.minimumAirborneAxialRetentionRatio < 0.7);
  assert.ok(audit.maximumAdjacentAxialSpeedDropRatio > 0.25);
  assert.ok(hoseJetDriftFailures({ ...audit, maximumSideCentroidOffset_m: 0 }, 0.02)
    .some((failure) => failure.includes("axial speed retention")));
});
