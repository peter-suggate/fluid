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

test("GPU volume correction rejects mapped control state and octree cadence has no host readback", () => {
  const solver = source("lib/webgpu-uniform-eulerian.ts");
  const surface = source("lib/webgpu-quadtree-builder.ts");
  assert.doesNotMatch(solver, /Remaining residency exception/);
  assert.match(solver, /hostSchedulingUsesReadback: false/);
  assert.match(solver, /const substeps = this\.quadtreeProjection \? proactiveQuadtreeSubsteps/);
  assert.match(surface, /this\.gpuVolumeCorrection \? 0 : this\.correctionSpeed/);
  assert.match(surface, /if \(!this\.gpuVolumeCorrection\) this\.correctionSpeed/);
});

test("octree uses one active-page dispatch with a bounded GPU fine-transport schedule", () => {
  const solver = source("lib/webgpu-uniform-eulerian.ts");
  const transport = source("lib/webgpu-octree-fine-levelset-transport.ts");
  const advance = solver.slice(solver.indexOf("advanceTo(time_s"), solver.indexOf("async readStats()"));
  assert.match(advance, /const substeps = this\.quadtreeProjection \? proactiveQuadtreeSubsteps/,
    "only the retained quadtree backend may select a host subdivision count");
  assert.doesNotMatch(advance, /this\.octreeProjection\s*\?\s*proactiveQuadtreeSubsteps/);
  assert.match(transport, /FINE_LEVELSET_TRANSPORT_MAXIMUM_ENCODED_SUBSTEPS = 64/);
  assert.match(transport,
    /owners\.dispatchWorkgroupsIndirect\(this\.indirectDispatch, 16\)/,
    "owner publication must use the exact active-page indirect record");
  assert.match(transport,
    /classify\.dispatchWorkgroupsIndirect\(this\.indirectDispatch, 160\)/,
    "classification must use the exact active-page indirect record");
  assert.match(transport,
    /transport\.dispatchWorkgroupsIndirect\(this\.indirectDispatch, \(4 \+ 7 \* index \+ 4\) \* 4\)/,
    "each specialized transport must use its exact compact class workset");
  assert.match(transport,
    /commit\.dispatchWorkgroupsIndirect\(this\.indirectDispatch, 160\)/,
    "commit must use the same exact active-page publication");
  assert.doesNotMatch(transport, /dispatchWorkgroups\(.*maximumResidentBricks/,
    "fine transport must never dispatch the resident-capacity ceiling");
  assert.doesNotMatch(transport.slice(transport.indexOf("encode(broker:")), /createBindGroup/,
    "recurring fine transport must reuse construction-time bind groups");
});

test("power-volume publication reuses its initialization-time bind group", () => {
  const octree = source("lib/webgpu-octree.ts");
  assert.match(octree, /private powerVolumeGroup\?: GPUBindGroup/);
  assert.match(octree, /this\.powerVolumeGroup = this\.device\.createBindGroup/);
  const encode = octree.slice(octree.indexOf("private encodeNativePowerAssembly"), octree.indexOf("private encodeNativePowerProjection"));
  assert.doesNotMatch(encode, /createBindGroup/);
});
