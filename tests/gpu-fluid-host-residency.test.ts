import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { collapseGPUFixedSteps } from "../lib/simulation/controller";

const source = (relative: string) => readFileSync(new URL(`../${relative}`, import.meta.url), "utf8");

test("GPU target-clock debt collapses in O(1) without changing fixed-step time", () => {
  assert.deepEqual(collapseGPUFixedSteps(0.1, 0.004), { steps: 25, remainder_s: 0 });
  const partial = collapseGPUFixedSteps(0.0105, 0.004);
  assert.equal(partial.steps, 2);
  assert.ok(Math.abs(partial.remainder_s - 0.0025) < 1e-12);
  assert.deepEqual(collapseGPUFixedSteps(Number.NaN, 0.004), { steps: 0, remainder_s: 0 });
});

test("WebGPU controller and renderer retain no CPU fluid oracle or dense upload authority", () => {
  const controller = source("lib/simulation/controller.ts");
  const renderer = source("lib/webgpu-renderer.ts");
  assert.match(controller, /private fluidSolver\?: EulerianFluidSolver/);
  assert.match(controller, /if \(this\.backend === "cpu-reference"\) this\.fluidSolver = this\.buildFluidSolver/);
  assert.match(controller, /if \(backend === "webgpu"\) \{[\s\S]*collapseGPUFixedSteps/);
  assert.match(renderer, /if \(backend === "cpu-reference"\) this\.uploadFluid\(fluid\)/);
});

test("GPU volume correction rejects mapped control state and cadence exposes its remaining exception", () => {
  const solver = source("lib/webgpu-uniform-eulerian.ts");
  const surface = source("lib/webgpu-quadtree-builder.ts");
  assert.match(solver, /Remaining residency exception/);
  assert.match(solver, /hostSchedulingUsesReadback: true/);
  assert.match(surface, /this\.gpuVolumeCorrection \? 0 : this\.correctionSpeed/);
  assert.match(surface, /if \(!this\.gpuVolumeCorrection\) this\.correctionSpeed/);
});

test("power-volume publication reuses its initialization-time bind group", () => {
  const octree = source("lib/webgpu-octree.ts");
  assert.match(octree, /private powerVolumeGroup\?: GPUBindGroup/);
  assert.match(octree, /this\.powerVolumeGroup = this\.device\.createBindGroup/);
  const encode = octree.slice(octree.indexOf("private encodePowerAssemblyMirror"), octree.indexOf("private encodePowerProjectionMirror"));
  assert.doesNotMatch(encode, /createBindGroup/);
});
