import assert from "node:assert/strict";
import test from "node:test";
import { EulerianFluidSolver } from "../lib/eulerian-solver";
import { cloneScene, defaultScene } from "../lib/model";
import { initializeRigidBodies } from "../lib/rigid-body";
import { appendSimulationFrame, createSimulationRecording, simulationFrameAt } from "../lib/simulation-recording";

test("recording keeps independent snapshots and replaces a same-time endpoint", () => {
  const scene = cloneScene(defaultScene);
  const bodies = initializeRigidBodies(scene.rigidBodies);
  const solver = new EulerianFluidSolver(scene);
  const recording = createSimulationRecording(scene, "webgpu", "balanced", bodies, solver.getRenderState());

  bodies[0].position_m.y += 1;
  solver.fluid[0] = 123;
  appendSimulationFrame(recording, 0, bodies, solver.getRenderState(), true);

  assert.equal(recording.frames.length, 1);
  assert.equal(recording.frames[0].bodies[0].position_m.y, bodies[0].position_m.y);
  assert.equal(recording.frames[0].fluid.occupancy[0], 123);
  bodies[0].position_m.y += 1;
  solver.fluid[0] = 0;
  assert.notEqual(recording.frames[0].bodies[0].position_m.y, bodies[0].position_m.y);
  assert.equal(recording.frames[0].fluid.occupancy[0], 123);
});

test("recording samples at display cadence and finds the latest frame at a playback time", () => {
  const scene = cloneScene(defaultScene);
  const bodies = initializeRigidBodies(scene.rigidBodies);
  const solver = new EulerianFluidSolver(scene);
  const recording = createSimulationRecording(scene, "cpu-reference", "high", bodies, solver.getRenderState());

  assert.equal(appendSimulationFrame(recording, 0.005, bodies, solver.getRenderState()), false);
  bodies[0].position_m.x = 1;
  assert.equal(appendSimulationFrame(recording, 0.02, bodies, solver.getRenderState()), true);
  bodies[0].position_m.x = 2;
  assert.equal(appendSimulationFrame(recording, 0.04, bodies, solver.getRenderState()), true);

  assert.equal(simulationFrameAt(recording, 0.039).bodies[0].position_m.x, 1);
  assert.equal(simulationFrameAt(recording, 0.04).bodies[0].position_m.x, 2);
  assert.equal(recording.duration_s, 0.04);
});
