import assert from "node:assert/strict";
import test from "node:test";
import {
  CPU_PHYSICS_ACTIVITY_TASKS,
  createCPUPerformanceActivityProfiler,
} from "../lib/cpu-performance-activity";
import { EulerianFluidSolver } from "../lib/eulerian-solver";
import { cloneScene, defaultScene } from "../lib/model";

test("the CPU Eulerian step exposes every coarse physics workload through one profiler argument", () => {
  const scene = cloneScene(defaultScene);
  scene.numerics.fixedDt_s = 0.004;
  const solver = new EulerianFluidSolver(scene, {
    dimensions: { nx: 4, ny: 4, nz: 4 },
    markerSamplesPerAxis: 1,
  });
  const profiler = createCPUPerformanceActivityProfiler({
    enabled: true,
    identity: { frameId: "cpu-frame", generation: 1, submissionId: "cpu-step" },
    resourceId: "cpu.main",
    resourceLabel: "CPU main thread",
    resourceKind: "cpu-main",
  });

  solver.step(scene.numerics.fixedDt_s, profiler);

  const measured = new Set(profiler.output().spans.map((span) => span.taskId));
  for (const task of [
    CPU_PHYSICS_ACTIVITY_TASKS.timestep,
    CPU_PHYSICS_ACTIVITY_TASKS.inflow,
    CPU_PHYSICS_ACTIVITY_TASKS.forces,
    CPU_PHYSICS_ACTIVITY_TASKS.velocityAdvection,
    CPU_PHYSICS_ACTIVITY_TASKS.viscosity,
    CPU_PHYSICS_ACTIVITY_TASKS.divergenceBefore,
    CPU_PHYSICS_ACTIVITY_TASKS.pressure,
    CPU_PHYSICS_ACTIVITY_TASKS.divergenceAfter,
    CPU_PHYSICS_ACTIVITY_TASKS.markerAdvection,
    CPU_PHYSICS_ACTIVITY_TASKS.diagnostics,
  ]) assert.ok(measured.has(task.id), task.label);
});
